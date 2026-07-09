from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    resolve_imod_executable,
    run_command,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.protocols import BackendContext, ReconstructionBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
    TomogramQc,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc
from cryoet_pipeline.runtime import DevicePreference


class ImodTiltReconstructionBackend:
    """Reconstruct a fine-aligned tomogram and canonicalize it to Zarr."""

    name = "imod_tilt"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def reconstruct(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return canonical Zarr tomogram and reconstruction QC artifacts."""

        stack_shape = _validate_final_aligned_stack(tilt_stack)
        alignment_result = _load_alignment(alignment, tilt_stack, manifest)
        output_binning = tilt_stack.binning or 1
        geometry = _reconstruction_geometry(
            context,
            alignment,
            output_binning=output_binning,
        )
        radial_cutoff = _bounded_float_parameter(
            context,
            "radial_cutoff",
            default=0.35,
            minimum=0.0,
            maximum=0.5,
        )
        radial_falloff = _bounded_float_parameter(
            context,
            "radial_falloff",
            default=0.05,
            minimum=0.0,
            maximum=0.5,
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        included_z_values = _included_z_values(tilt_stack, stack_shape[0])
        tilt_angles = _solved_tilt_angles(tilt_stack, stack_shape[0])
        voxel_spacing_angstrom = _voxel_spacing(
            tilt_stack,
            manifest,
            output_binning,
        )
        paths = _reconstruction_paths(
            context.output_dir,
            manifest,
            alignment_stage=alignment_result.stage,
            output_binning=output_binning,
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        executable = resolve_imod_executable("tilt", context)

        paths.tomogram_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-reconstruct-",
            dir=paths.tomogram_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_tilt_angles = temporary_root / f"{manifest.tilt_series_id}.tlt"
            temporary_rec = temporary_root / f"{manifest.tilt_series_id}.rec"
            temporary_zarr = temporary_root / f"{manifest.tilt_series_id}.zarr"
            temporary_xy = temporary_root / "central_xy.mrc"
            temporary_xz = temporary_root / "central_xz.mrc"
            temporary_yz = temporary_root / "central_yz.mrc"
            temporary_report = temporary_root / "reconstruction_qc.json"

            write_tilt_angles_file(temporary_tilt_angles, tilt_angles)
            command = [
                str(executable),
                "-input",
                str(tilt_stack.path.resolve()),
                "-output",
                str(temporary_rec),
                "-TILTFILE",
                str(temporary_tilt_angles),
                "-THICKNESS",
                str(geometry.thickness_px),
                "-XAXISTILT",
                f"{geometry.x_axis_tilt_deg:.6f}",
                "-SHIFT",
                f"0.000000,{geometry.z_shift_px:.6f}",
                "-RADIAL",
                f"{radial_cutoff:.6f},{radial_falloff:.6f}",
                "-FalloffIsTrueSigma",
                "-MODE",
                "2",
                "-PERPENDICULAR",
                "-AdjustOrigin",
            ]
            if context.device is DevicePreference.CUDA:
                command.extend(
                    [
                        "-UseGPU",
                        "0",
                        "-ActionIfGPUFails",
                        "2,2",
                    ]
                )
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(executable, context),
            )
            _write_command_log(paths.log, command, result)
            if result.returncode != 0:
                raise RuntimeError(
                    f"IMOD tilt failed with exit code {result.returncode}; see {paths.log}"
                )
            if not temporary_rec.is_file():
                raise RuntimeError(f"IMOD tilt did not write reconstruction: {temporary_rec}")

            rec_info = validate_complete_mrc(temporary_rec)
            expected_rec_shape = (
                stack_shape[1],
                geometry.thickness_px,
                stack_shape[2],
            )
            if rec_info.shape != expected_rec_shape:
                raise ValueError(
                    f"unexpected IMOD reconstruction shape {rec_info.shape}; "
                    f"expected YZX {expected_rec_shape}"
                )
            with mrcfile.open(temporary_rec, mode="r+") as rec:
                rec.voxel_size = voxel_spacing_angstrom

            statistics = _write_canonical_zarr(
                temporary_rec,
                temporary_zarr,
                voxel_spacing_angstrom=voxel_spacing_angstrom,
            )
            canonical_shape = (
                geometry.thickness_px,
                stack_shape[1],
                stack_shape[2],
            )
            _write_central_slices(
                temporary_zarr,
                xy_path=temporary_xy,
                xz_path=temporary_xz,
                yz_path=temporary_yz,
                voxel_spacing_angstrom=voxel_spacing_angstrom,
            )

            tomogram_id = f"{manifest.tilt_series_id}:tomogram:fine"
            warnings = [
                "CTF correction was not applied",
            ]
            report = TomogramQc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                tomogram_id=tomogram_id,
                shape=canonical_shape,
                voxel_spacing_angstrom=voxel_spacing_angstrom,
                minimum=statistics.minimum,
                maximum=statistics.maximum,
                mean=statistics.mean,
                standard_deviation=statistics.standard_deviation,
                finite=True,
                ctf_corrected=False,
                alignment_stage=alignment_result.stage,
                central_slice_paths={
                    "xy": paths.central_xy,
                    "xz": paths.central_xz,
                    "yz": paths.central_yz,
                },
                status=QcStatus.WARNING,
                warnings=warnings,
            )
            temporary_report.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )

            _replace_output(temporary_rec, paths.imod_rec)
            _replace_output(temporary_zarr, paths.canonical_zarr)
            _replace_output(temporary_tilt_angles, paths.tilt_angles)
            _replace_output(temporary_xy, paths.central_xy)
            _replace_output(temporary_xz, paths.central_xz)
            _replace_output(temporary_yz, paths.central_yz)
            _replace_output(temporary_report, paths.report)

        tomogram_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:tomogram:fine",
            kind=ArtifactKind.TOMOGRAM,
            path=paths.canonical_zarr,
            parent_ids=[tilt_stack.id, alignment.id],
            shape=(geometry.thickness_px, stack_shape[1], stack_shape[2]),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=voxel_spacing_angstrom,
            binning=output_binning,
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "tomogram_branch": "full",
                "alignment_stage": alignment_result.stage,
                "ctf_corrected": False,
                "included_z_values": included_z_values,
                "excluded_z_values": alignment_result.excluded_z_values,
                "thickness": geometry.thickness_px,
                "x_axis_tilt_deg": geometry.x_axis_tilt_deg,
                "z_shift_px": geometry.z_shift_px,
                "positioning_source": geometry.source,
                "radial_cutoff": radial_cutoff,
                "radial_falloff": radial_falloff,
                "imod_rec_path": str(paths.imod_rec),
                "imod_rec_axis_order": "yzx",
                "tilt_angles_path": str(paths.tilt_angles),
                "log_path": str(paths.log),
            },
            software_versions={
                "imod": "external",
                "zarr": _package_version("zarr"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=(
                _path_size_bytes(paths.canonical_zarr)
                + paths.imod_rec.stat().st_size
                + paths.tilt_angles.stat().st_size
                + paths.log.stat().st_size
            ),
        )
        qc_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:reconstruction:fine",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[tomogram_artifact.id],
            parameters={
                "qc_type": "fine_reconstruction",
                "tilt_series_id": manifest.tilt_series_id,
                "status": QcStatus.WARNING.value,
                "central_slice_paths": {
                    "xy": str(paths.central_xy),
                    "xz": str(paths.central_xz),
                    "yz": str(paths.central_yz),
                },
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.report,
                paths.central_xy,
                paths.central_xz,
                paths.central_yz,
            ),
        )
        return [tomogram_artifact, qc_artifact]


def reconstruct_and_register(
    backend: ReconstructionBackend,
    tilt_stack: Artifact,
    alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Reconstruct a tomogram and register tomogram then QC artifacts."""

    artifacts = backend.reconstruct(tilt_stack, alignment, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _ReconstructionPaths:
    def __init__(
        self,
        *,
        tomogram_dir: Path,
        qc_dir: Path,
        canonical_zarr: Path,
        imod_rec: Path,
        tilt_angles: Path,
        log: Path,
        report: Path,
        central_xy: Path,
        central_xz: Path,
        central_yz: Path,
    ) -> None:
        self.tomogram_dir = tomogram_dir
        self.qc_dir = qc_dir
        self.canonical_zarr = canonical_zarr
        self.imod_rec = imod_rec
        self.tilt_angles = tilt_angles
        self.log = log
        self.report = report
        self.central_xy = central_xy
        self.central_xz = central_xz
        self.central_yz = central_yz

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.canonical_zarr,
            self.imod_rec,
            self.tilt_angles,
            self.log,
            self.report,
            self.central_xy,
            self.central_xz,
            self.central_yz,
        )


class _VolumeStatistics:
    def __init__(
        self,
        *,
        minimum: float,
        maximum: float,
        mean: float,
        standard_deviation: float,
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.mean = mean
        self.standard_deviation = standard_deviation


@dataclass(frozen=True)
class _ReconstructionGeometry:
    thickness_px: int
    x_axis_tilt_deg: float
    z_shift_px: float
    source: str


def _reconstruction_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    alignment_stage: str,
    output_binning: int,
) -> _ReconstructionPaths:
    tomogram_dir = output_root / "tomograms" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "reconstruction"
    prefix = f"{manifest.tilt_series_id}_{alignment_stage}_bin{output_binning}"
    return _ReconstructionPaths(
        tomogram_dir=tomogram_dir,
        qc_dir=qc_dir,
        canonical_zarr=tomogram_dir / f"{prefix}.zarr",
        imod_rec=tomogram_dir / f"{prefix}.rec",
        tilt_angles=tomogram_dir / f"{prefix}.tlt",
        log=tomogram_dir / f"{prefix}_tilt.log",
        report=qc_dir / f"{prefix}_qc.json",
        central_xy=qc_dir / f"{prefix}_central_xy.mrc",
        central_xz=qc_dir / f"{prefix}_central_xz.mrc",
        central_yz=qc_dir / f"{prefix}_central_yz.mrc",
    )


def _validate_final_aligned_stack(tilt_stack: Artifact) -> tuple[int, int, int]:
    if tilt_stack.kind is not ArtifactKind.ALIGNED_TILT_STACK:
        raise ValueError(f"expected aligned tilt stack artifact, got {tilt_stack.kind}")
    if tilt_stack.parameters.get("purpose") != "final_alignment":
        raise ValueError("reconstruction requires a final fine-aligned stack")
    if tilt_stack.parameters.get("alignment_stage") != "fine":
        raise ValueError("final aligned stack must record alignment_stage='fine'")
    if not tilt_stack.path.is_file():
        raise FileNotFoundError(f"aligned tilt stack not found: {tilt_stack.path}")
    info = validate_complete_mrc(tilt_stack.path)
    if len(info.shape) != 3:
        raise ValueError(f"expected aligned tilt stack shape (tilts, y, x), got {info.shape}")
    shape = info.shape
    if tilt_stack.shape is not None and tuple(tilt_stack.shape) != shape:
        raise ValueError(f"aligned stack artifact shape {tilt_stack.shape} does not match {shape}")
    return shape


def _load_alignment(
    alignment: Artifact,
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
) -> TiltAlignment:
    if alignment.kind is not ArtifactKind.ALIGNMENT:
        raise ValueError(f"expected alignment artifact, got {alignment.kind}")
    result = TiltAlignment.model_validate_json(alignment.path.read_text())
    if result.tilt_series_id != manifest.tilt_series_id:
        raise ValueError(
            f"alignment is for {result.tilt_series_id}, expected {manifest.tilt_series_id}"
        )
    if alignment.id not in tilt_stack.parent_ids:
        raise ValueError(f"aligned stack {tilt_stack.id} is not derived from {alignment.id}")
    if result.stage != "fine" or alignment.parameters.get("stage") != "fine":
        raise ValueError("reconstruction requires fine alignment")
    return result


def _included_z_values(tilt_stack: Artifact, num_tilts: int) -> list[int]:
    value = tilt_stack.parameters.get("included_z_values")
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("aligned stack must record integer included_z_values")
    included = cast(list[int], value)
    if len(included) != num_tilts:
        raise ValueError(f"aligned stack has {num_tilts} images but {len(included)} z values")
    if len(included) != len(set(included)):
        raise ValueError("aligned stack included_z_values must be unique")
    return included


def _solved_tilt_angles(tilt_stack: Artifact, num_tilts: int) -> list[float]:
    value = tilt_stack.parameters.get("tilt_file_path")
    if not isinstance(value, (str, Path)):
        raise ValueError("final aligned stack must record tilt_file_path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"solved tilt-angle file not found: {path}")
    angles: list[float] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            angle = float(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid solved tilt angle") from exc
        if not math.isfinite(angle):
            raise ValueError(f"{path}:{line_number}: tilt angle must be finite")
        angles.append(angle)
    if len(angles) != num_tilts:
        raise ValueError(
            f"final aligned stack has {num_tilts} images but {len(angles)} solved tilt angles"
        )
    return angles


def _reconstruction_geometry(
    context: BackendContext,
    alignment: Artifact,
    *,
    output_binning: int,
) -> _ReconstructionGeometry:
    explicit = any(
        key in context.parameters for key in ("thickness", "x_axis_tilt_deg", "z_shift_px")
    )
    if "thickness" in context.parameters:
        thickness = _positive_int_parameter(context, "thickness", default=1)
    else:
        unbinned_thickness = _artifact_numeric_parameter(
            alignment,
            "recommended_unbinned_thickness_px",
            positive=True,
        )
        thickness = math.ceil(unbinned_thickness / output_binning)

    if "x_axis_tilt_deg" in context.parameters:
        x_axis_tilt = _finite_float_parameter(context, "x_axis_tilt_deg")
    else:
        x_axis_tilt = _artifact_numeric_parameter(
            alignment,
            "recommended_x_axis_tilt_deg",
        )

    if "z_shift_px" in context.parameters:
        z_shift = _finite_float_parameter(context, "z_shift_px")
    else:
        if alignment.parameters.get("axis_z_shift_applied_in_alignment") is not True:
            raise ValueError(
                "fine alignment must apply AxisZShift before reconstruction, "
                "or reconstruction requires an explicit z_shift_px"
            )
        z_shift = 0.0

    return _ReconstructionGeometry(
        thickness_px=thickness,
        x_axis_tilt_deg=x_axis_tilt,
        z_shift_px=z_shift,
        source=(
            "explicit_override"
            if explicit
            else str(
                alignment.parameters.get(
                    "positioning_source",
                    "fine_alignment",
                )
            )
        ),
    )


def _voxel_spacing(
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
    preview_binning: int,
) -> float:
    if tilt_stack.pixel_spacing_angstrom is not None:
        return tilt_stack.pixel_spacing_angstrom
    if manifest.raw_pixel_spacing_angstrom is None:
        raise ValueError("voxel spacing is unavailable for reconstruction")
    return manifest.raw_pixel_spacing_angstrom * preview_binning


def _write_canonical_zarr(
    imod_rec_path: Path,
    output_path: Path,
    *,
    voxel_spacing_angstrom: float,
) -> _VolumeStatistics:
    with mrcfile.open(imod_rec_path, permissive=True) as rec:
        source_y, depth, width = (int(axis_size) for axis_size in rec.data.shape)
        chunks = (
            min(depth, 16),
            min(source_y, 64),
            min(width, 256),
        )
        output = cast(
            Any,
            zarr.open(
                output_path,
                mode="w",
                shape=(depth, source_y, width),
                chunks=chunks,
                dtype=np.float32,
            ),
        )
        output.attrs.update(
            {
                "axis_order": "zyx",
                "source_axis_order": "yzx",
                "voxel_spacing_angstrom": voxel_spacing_angstrom,
            }
        )

        count = 0
        value_sum = 0.0
        value_square_sum = 0.0
        minimum = math.inf
        maximum = -math.inf
        for start_y in range(0, source_y, chunks[1]):
            stop_y = min(start_y + chunks[1], source_y)
            block_yzx = np.asarray(rec.data[start_y:stop_y, :, :], dtype=np.float32)
            if not np.isfinite(block_yzx).all():
                raise ValueError("IMOD reconstruction contains nonfinite values")
            block_zyx = block_yzx.transpose(1, 0, 2)
            output[:, start_y:stop_y, :] = block_zyx
            block_float64 = block_yzx.astype(np.float64, copy=False)
            count += block_float64.size
            value_sum += float(block_float64.sum())
            value_square_sum += float(np.square(block_float64).sum())
            minimum = min(minimum, float(block_float64.min()))
            maximum = max(maximum, float(block_float64.max()))

    mean = value_sum / count
    variance = max(value_square_sum / count - mean * mean, 0.0)
    return _VolumeStatistics(
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        standard_deviation=math.sqrt(variance),
    )


def _write_central_slices(
    tomogram_path: Path,
    *,
    xy_path: Path,
    xz_path: Path,
    yz_path: Path,
    voxel_spacing_angstrom: float,
) -> None:
    tomogram = cast(Any, zarr.open(tomogram_path, mode="r"))
    depth, height, width = (int(axis_size) for axis_size in tomogram.shape)
    slices = {
        xy_path: np.asarray(tomogram[depth // 2, :, :], dtype=np.float32),
        xz_path: np.asarray(tomogram[:, height // 2, :], dtype=np.float32),
        yz_path: np.asarray(tomogram[:, :, width // 2], dtype=np.float32),
    }
    for path, data in slices.items():
        with mrcfile.new(path, overwrite=True) as output:
            output.set_data(data)
            output.voxel_size = voxel_spacing_angstrom


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


def _replace_output(source: Path, destination: Path) -> None:
    if destination.exists():
        _remove_path(destination)
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"reconstruction outputs already exist: {joined}")


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


def _bounded_float_parameter(
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
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(f"context parameter {key!r} must be between {minimum} and {maximum}")
    return normalized


def _finite_float_parameter(context: BackendContext, key: str) -> float:
    value = context.parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"context parameter {key!r} must be finite")
    return normalized


def _artifact_numeric_parameter(
    artifact: Artifact,
    key: str,
    *,
    positive: bool = False,
) -> float:
    value = artifact.parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"alignment artifact parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"alignment artifact parameter {key!r} must be finite")
    if positive and normalized <= 0.0:
        raise ValueError(f"alignment artifact parameter {key!r} must be positive")
    return normalized


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
