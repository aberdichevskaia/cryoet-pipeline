from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext, TiltAlignmentBackend
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc


class CommandRunner(Protocol):
    """Injectable external-command runner used by backend contract tests."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        """Run a command and return captured text output."""
        ...


class ImodTiltXcorrAlignmentBackend:
    """Run IMOD Tiltxcorr for coarse translational tilt-series alignment."""

    name = "imod_tiltxcorr"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def align(
        self,
        tilt_stack: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Produce canonical transforms and an IMOD-compatible `.xf` export."""

        stack_shape = _validate_tilt_stack(tilt_stack, manifest)
        input_binning = _positive_int_parameter(context, "binning", default=8)
        _validate_binning(stack_shape, input_binning)
        min_std_ratio = _float_parameter(
            context,
            "min_std_ratio",
            default=0.2,
            minimum=0.0,
            maximum=1.0,
        )
        filter_sigma1 = _float_parameter(
            context,
            "filter_sigma1",
            default=0.03,
            minimum=0.0,
            maximum=0.5,
        )
        filter_radius2 = _float_parameter(
            context,
            "filter_radius2",
            default=0.25,
            minimum=0.0,
            maximum=0.5,
        )
        filter_sigma2 = _float_parameter(
            context,
            "filter_sigma2",
            default=0.05,
            minimum=0.0,
            maximum=0.5,
        )
        tilt_axis_angle_deg = _tilt_axis_angle(manifest, context)
        tiltxcorr_executable = resolve_imod_executable("tiltxcorr", context)
        xftoxg_executable = resolve_imod_executable("xftoxg", context)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _alignment_paths(context.output_dir, manifest)
        _require_available_output_paths(paths, overwrite=overwrite)
        alignment_order = sorted(
            range(manifest.num_tilts),
            key=lambda index: manifest.images[index].tilt_angle_deg,
        )

        paths.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-tiltxcorr-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            binned_stack_path = temporary_root / f"{manifest.tilt_series_id}_bin.st"
            tilt_angles_path = temporary_root / f"{manifest.tilt_series_id}.tlt"
            raw_transform_path = temporary_root / f"{manifest.tilt_series_id}.prexf"
            global_transform_path = temporary_root / f"{manifest.tilt_series_id}.prexg"

            projection_std_in_alignment_order = write_binned_mrc_stack(
                tilt_stack.path,
                binned_stack_path,
                binning=input_binning,
                pixel_spacing_angstrom=tilt_stack.pixel_spacing_angstrom,
                section_indices=alignment_order,
            )
            write_tilt_angles_file(
                tilt_angles_path,
                [
                    manifest.images[index].tilt_angle_deg
                    for index in alignment_order
                ],
            )
            median_std = float(np.median(projection_std_in_alignment_order))
            if median_std <= 0.0:
                raise ValueError("input tilt stack has no measurable intensity variance")
            excluded_alignment_indices = [
                index
                for index, projection_std in enumerate(
                    projection_std_in_alignment_order
                )
                if projection_std < median_std * min_std_ratio
            ]
            if len(excluded_alignment_indices) == manifest.num_tilts:
                raise ValueError("input QC excluded every tilt image")
            command = [
                str(tiltxcorr_executable),
                "-input",
                str(binned_stack_path),
                "-output",
                str(raw_transform_path),
                "-tiltfile",
                str(tilt_angles_path),
                "-rotation",
                f"{tilt_axis_angle_deg:.6f}",
                "-sigma1",
                f"{filter_sigma1:.6f}",
                "-radius2",
                f"{filter_radius2:.6f}",
                "-sigma2",
                f"{filter_sigma2:.6f}",
                "-verbose",
                "1",
            ]
            if excluded_alignment_indices:
                command.extend(
                    [
                        "-skip",
                        ",".join(
                            str(index + 1)
                            for index in excluded_alignment_indices
                        ),
                    ]
                )
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(tiltxcorr_executable, context),
            )
            _write_command_log(paths.tiltxcorr_log, command, result)
            if result.returncode != 0:
                raise RuntimeError(
                    f"IMOD tiltxcorr failed with exit code {result.returncode}; "
                    f"see {paths.tiltxcorr_log}"
                )
            if not raw_transform_path.is_file():
                raise RuntimeError(
                    f"IMOD tiltxcorr did not write transforms: {raw_transform_path}"
                )

            raw_transforms = parse_imod_xf(raw_transform_path)
            if len(raw_transforms) != manifest.num_tilts:
                raise ValueError(
                    f"IMOD tiltxcorr returned {len(raw_transforms)} transforms for "
                    f"{manifest.num_tilts} tilts"
                )

            xftoxg_command = [
                str(xftoxg_executable),
                "-input",
                str(raw_transform_path),
                "-goutput",
                str(global_transform_path),
                "-nfit",
                "0",
            ]
            xftoxg_result = self._command_runner(
                xftoxg_command,
                cwd=temporary_root,
                env=imod_environment(xftoxg_executable, context),
            )
            _write_command_log(paths.xftoxg_log, xftoxg_command, xftoxg_result)
            if xftoxg_result.returncode != 0:
                raise RuntimeError(
                    f"IMOD xftoxg failed with exit code {xftoxg_result.returncode}; "
                    f"see {paths.xftoxg_log}"
                )
            if not global_transform_path.is_file():
                raise RuntimeError(
                    f"IMOD xftoxg did not write global transforms: "
                    f"{global_transform_path}"
                )
            global_transforms = parse_imod_xf(global_transform_path)

        if len(global_transforms) != manifest.num_tilts:
            raise ValueError(
                f"IMOD xftoxg returned {len(global_transforms)} transforms for "
                f"{manifest.num_tilts} tilts"
            )

        transforms_by_manifest_index = {
            manifest_index: transform
            for manifest_index, transform in zip(
                alignment_order,
                global_transforms,
                strict=True,
            )
        }
        projection_std_by_manifest_index = {
            manifest_index: projection_std
            for manifest_index, projection_std in zip(
                alignment_order,
                projection_std_in_alignment_order,
                strict=True,
            )
        }
        excluded_manifest_indices = {
            alignment_order[index] for index in excluded_alignment_indices
        }
        normalized_transforms = [
            AlignmentTransform(
                z_value=image.z_value,
                tilt_angle_deg=image.tilt_angle_deg,
                a11=values[0],
                a12=values[1],
                a21=values[2],
                a22=values[3],
                shift_x_px=values[4] * input_binning,
                shift_y_px=values[5] * input_binning,
            )
            for manifest_index, image in enumerate(manifest.images)
            for values in (
                transforms_by_manifest_index[manifest_index],
            )
        ]
        alignment = TiltAlignment(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            stage="coarse",
            input_stack_id=tilt_stack.id,
            input_binning=input_binning,
            tilt_axis_angle_deg=tilt_axis_angle_deg,
            transform_semantics="global",
            transforms=normalized_transforms,
            input_projection_std=[
                projection_std_by_manifest_index[index]
                for index in range(manifest.num_tilts)
            ],
            excluded_z_values=[
                image.z_value
                for index, image in enumerate(manifest.images)
                if index in excluded_manifest_indices
            ],
        )
        _write_alignment_outputs(
            alignment,
            canonical_path=paths.canonical,
            imod_xf_path=paths.imod_xf,
        )

        return Artifact(
            id=f"{manifest.tilt_series_id}:alignment:coarse",
            kind=ArtifactKind.ALIGNMENT,
            path=paths.canonical,
            parent_ids=[tilt_stack.id],
            shape=(manifest.num_tilts, 6),
            dtype="float64",
            parameters={
                "backend": self.name,
                "stage": "coarse",
                "tilt_series_id": manifest.tilt_series_id,
                "input_binning": input_binning,
                "alignment_order": "tilt_angle_ascending",
                "transform_semantics": "global",
                "min_std_ratio": min_std_ratio,
                "excluded_z_values": alignment.excluded_z_values,
                "filter_sigma1": filter_sigma1,
                "filter_radius2": filter_radius2,
                "filter_sigma2": filter_sigma2,
                "tilt_axis_angle_deg": tilt_axis_angle_deg,
                "imod_xf_path": str(paths.imod_xf),
                "tiltxcorr_log_path": str(paths.tiltxcorr_log),
                "xftoxg_log_path": str(paths.xftoxg_log),
                "translation_units": "full_resolution_pixels",
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.canonical,
                paths.imod_xf,
                paths.tiltxcorr_log,
                paths.xftoxg_log,
            ),
        )


def align_and_register(
    backend: TiltAlignmentBackend,
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> Artifact:
    """Run tilt-series alignment and register the canonical result."""

    artifact = backend.align(tilt_stack, manifest, context)
    registry.add(artifact, replace=replace_existing)
    return artifact


def parse_imod_xf(path: Path) -> list[tuple[float, float, float, float, float, float]]:
    """Parse six-column IMOD affine transforms with finite-value validation."""

    transforms: list[tuple[float, float, float, float, float, float]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(
                f"{path}:{line_number}: expected 6 IMOD transform values, "
                f"found {len(fields)}"
            )
        try:
            transform = tuple(float(field) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid IMOD transform value"
            ) from exc
        if not all(np.isfinite(value) for value in transform):
            raise ValueError(f"{path}:{line_number}: transform values must be finite")
        transforms.append(cast(tuple[float, float, float, float, float, float], transform))

    if not transforms:
        raise ValueError(f"{path}: no IMOD transforms found")
    return transforms


class _AlignmentPaths:
    def __init__(
        self,
        output_dir: Path,
        canonical: Path,
        imod_xf: Path,
        tiltxcorr_log: Path,
        xftoxg_log: Path,
    ) -> None:
        self.output_dir = output_dir
        self.canonical = canonical
        self.imod_xf = imod_xf
        self.tiltxcorr_log = tiltxcorr_log
        self.xftoxg_log = xftoxg_log

    @property
    def outputs(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.canonical,
            self.imod_xf,
            self.tiltxcorr_log,
            self.xftoxg_log,
        )


def _alignment_paths(output_root: Path, manifest: TiltSeriesManifest) -> _AlignmentPaths:
    output_dir = output_root / "alignments" / manifest.tilt_series_id
    prefix = f"{manifest.tilt_series_id}_coarse"
    return _AlignmentPaths(
        output_dir=output_dir,
        canonical=output_dir / f"{prefix}_alignment.json",
        imod_xf=output_dir / f"{prefix}.xf",
        tiltxcorr_log=output_dir / f"{prefix}_tiltxcorr.log",
        xftoxg_log=output_dir / f"{prefix}_xftoxg.log",
    )


def _require_available_output_paths(paths: _AlignmentPaths, *, overwrite: bool) -> None:
    existing = [path for path in paths.outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"alignment outputs already exist: {joined}")


def _validate_tilt_stack(
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[int, int, int]:
    if tilt_stack.kind is not ArtifactKind.TILT_STACK:
        raise ValueError(f"expected tilt stack artifact, got {tilt_stack.kind}")
    if not tilt_stack.path.exists():
        raise FileNotFoundError(f"tilt stack not found: {tilt_stack.path}")

    shape = _stack_shape(tilt_stack.path)
    if len(shape) != 3:
        raise ValueError(f"expected tilt stack with shape (tilts, y, x), got {shape}")
    if shape[0] != manifest.num_tilts:
        raise ValueError(
            f"tilt stack has {shape[0]} images, manifest has {manifest.num_tilts}"
        )
    if tilt_stack.shape is not None and tuple(tilt_stack.shape) != shape:
        raise ValueError(
            f"tilt stack artifact shape {tilt_stack.shape} does not match data {shape}"
        )
    return shape


def _stack_shape(path: Path) -> tuple[int, ...]:
    if path.suffix == ".zarr" or path.is_dir():
        stack = cast(Any, zarr.open(path, mode="r"))
        return tuple(int(axis_size) for axis_size in stack.shape)
    return validate_complete_mrc(path).shape


def _validate_binning(stack_shape: tuple[int, int, int], binning: int) -> None:
    if binning > min(stack_shape[1:]):
        raise ValueError(
            f"alignment binning {binning} exceeds stack image shape {stack_shape[1:]}"
        )


def write_binned_mrc_stack(
    source_path: Path,
    output_path: Path,
    *,
    binning: int,
    pixel_spacing_angstrom: float | None,
    section_indices: Sequence[int],
) -> list[float]:
    if source_path.suffix == ".zarr" or source_path.is_dir():
        source = cast(Any, zarr.open(source_path, mode="r"))
        return _write_binned_array(
            source,
            output_path,
            binning=binning,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            section_indices=section_indices,
        )
    with mrcfile.open(source_path, permissive=True) as source_mrc:
        return _write_binned_array(
            source_mrc.data,
            output_path,
            binning=binning,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            section_indices=section_indices,
        )


def _write_binned_array(
    source: Any,
    output_path: Path,
    *,
    binning: int,
    pixel_spacing_angstrom: float | None,
    section_indices: Sequence[int],
) -> list[float]:
    source_tilts, height, width = (int(axis_size) for axis_size in source.shape)
    if not section_indices:
        raise ValueError("section_indices must not be empty")
    if len(section_indices) != len(set(section_indices)):
        raise ValueError("section_indices must be unique")
    if any(index < 0 or index >= source_tilts for index in section_indices):
        raise ValueError("section_indices contain an out-of-range stack index")
    num_tilts = len(section_indices)
    projection_std: list[float] = []
    output_height = height // binning
    output_width = width // binning
    cropped_height = output_height * binning
    cropped_width = output_width * binning
    start_y = (height - cropped_height) // 2
    start_x = (width - cropped_width) // 2

    with mrcfile.new_mmap(
        output_path,
        shape=(num_tilts, output_height, output_width),
        mrc_mode=2,
        overwrite=True,
    ) as output_mrc:
        for output_index, source_index in enumerate(section_indices):
            plane = np.asarray(
                source[
                    source_index,
                    start_y : start_y + cropped_height,
                    start_x : start_x + cropped_width,
                ],
                dtype=np.float32,
            )
            if binning > 1:
                plane = plane.reshape(
                    output_height,
                    binning,
                    output_width,
                    binning,
                ).mean(axis=(1, 3), dtype=np.float32)
            output_mrc.data[output_index, :, :] = plane
            projection_std.append(float(plane.std(dtype=np.float64)))
        if pixel_spacing_angstrom is not None:
            output_mrc.voxel_size = pixel_spacing_angstrom * binning
    return projection_std


def write_tilt_angles_file(path: Path, tilt_angles_deg: Sequence[float]) -> None:
    path.write_text("".join(f"{angle:.6f}\n" for angle in tilt_angles_deg))


def _tilt_axis_angle(manifest: TiltSeriesManifest, context: BackendContext) -> float:
    configured = context.parameters.get("tilt_axis_angle_deg")
    if configured is not None:
        if isinstance(configured, bool) or not isinstance(configured, (int, float)):
            raise TypeError("context parameter 'tilt_axis_angle_deg' must be numeric")
        if not np.isfinite(configured):
            raise ValueError("context parameter 'tilt_axis_angle_deg' must be finite")
        return float(configured)

    rotation_angles = [
        image.rotation_angle_deg
        for image in manifest.images
        if image.rotation_angle_deg is not None
    ]
    if len(rotation_angles) != manifest.num_tilts:
        raise ValueError(
            "tilt-axis angle is unavailable; provide tilt_axis_angle_deg explicitly"
        )
    first = rotation_angles[0]
    if any(not np.isclose(angle, first, atol=1e-6) for angle in rotation_angles[1:]):
        raise ValueError("manifest rotation angles are inconsistent across tilt images")
    return _normalize_angle(first - 90.0)


def _normalize_angle(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def resolve_imod_executable(program: str, context: BackendContext) -> Path:
    """Resolve one IMOD executable from explicit config, PATH, or IMOD_DIR."""

    configured_key = f"{program}_executable"
    configured = context.parameters.get(configured_key)
    candidates: list[Path] = []
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError(
                f"context parameter {configured_key!r} must be a path string"
            )
        candidates.append(Path(configured).expanduser())
    else:
        discovered = shutil.which(program)
        if discovered is not None:
            candidates.append(Path(discovered))

        imod_dir = context.parameters.get("imod_dir", os.environ.get("IMOD_DIR"))
        if imod_dir is not None:
            if not isinstance(imod_dir, (str, Path)):
                raise TypeError("context parameter 'imod_dir' must be a path string")
            candidates.append(Path(imod_dir).expanduser() / "bin" / program)
        candidates.append(Path("/Applications/IMOD/bin") / program)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"IMOD {program} executable not found; configure IMOD_DIR or "
        f"{configured_key}"
    )


def imod_environment(executable: Path, context: BackendContext) -> dict[str, str]:
    environment = os.environ.copy()
    configured_imod_dir = context.parameters.get("imod_dir")
    if configured_imod_dir is not None:
        imod_dir = Path(str(configured_imod_dir)).expanduser().resolve()
    elif "IMOD_DIR" in environment:
        imod_dir = Path(environment["IMOD_DIR"]).expanduser().resolve()
    else:
        imod_dir = executable.parent.parent

    environment["IMOD_DIR"] = str(imod_dir)
    environment["PATH"] = f"{imod_dir / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
    qtlib = imod_dir / "qtlib"
    if qtlib.is_dir():
        environment["IMOD_QTLIBDIR"] = str(qtlib)
    return environment


def _write_command_log(
    path: Path,
    command: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    path.write_text(
        f"$ {shlex.join(command)}\n\n"
        f"[stdout]\n{result.stdout or ''}\n"
        f"[stderr]\n{result.stderr or ''}\n"
        f"[exit_code]\n{result.returncode}\n"
    )


def _write_alignment_outputs(
    alignment: TiltAlignment,
    *,
    canonical_path: Path,
    imod_xf_path: Path,
) -> None:
    canonical_temporary = canonical_path.with_suffix(canonical_path.suffix + ".tmp")
    imod_temporary = imod_xf_path.with_suffix(imod_xf_path.suffix + ".tmp")
    try:
        canonical_temporary.write_text(
            json.dumps(alignment.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )
        imod_temporary.write_text(
            "".join(
                f"{transform.a11:.7f} {transform.a12:.7f} "
                f"{transform.a21:.7f} {transform.a22:.7f} "
                f"{transform.shift_x_px:.3f} {transform.shift_y_px:.3f}\n"
                for transform in alignment.transforms
            )
        )
        imod_temporary.replace(imod_xf_path)
        canonical_temporary.replace(canonical_path)
    except OSError:
        canonical_temporary.unlink(missing_ok=True)
        imod_temporary.unlink(missing_ok=True)
        raise


def _positive_int_parameter(
    context: BackendContext,
    key: str,
    *,
    default: int,
) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < 1:
        raise ValueError(f"context parameter {key!r} must be at least 1")
    return value


def _float_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(
            f"context parameter {key!r} must be between {minimum} and {maximum}"
        )
    return normalized


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
