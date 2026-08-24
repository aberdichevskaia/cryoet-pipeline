from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    TemplatePreparationBackend,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TemplatePreparationQc,
    TiltSeriesManifest,
)


class PytomTemplateBackend:
    """Build a matching template and mask with pytom-match-pick's helper tools.

    `pytom_create_template.py` resamples a reference map to the tomogram's voxel
    size; `pytom_create_mask.py` builds the matching spherical mask. The two
    must share a box size, which is checked here rather than left to fail
    obscurely inside template matching.
    """

    name = "pytom"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def prepare(
        self,
        source_map: Path,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a template, a template mask, and a QC artifact."""

        if not source_map.is_file():
            raise FileNotFoundError(f"reference map not found: {source_map}")
        _validate_tomogram(tomogram, manifest)
        output_voxel_size = _tomogram_voxel_size(tomogram)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        invert = _bool_parameter(context, "template_invert", default=True)
        center = _bool_parameter(context, "template_center", default=False)
        mirror = _bool_parameter(context, "template_mirror", default=False)
        low_pass = _optional_positive_float_parameter(context, "template_low_pass_angstrom")
        box_size = _optional_positive_int_parameter(context, "template_box_size_px")
        input_voxel_size = _optional_positive_float_parameter(
            context, "template_input_voxel_size_angstrom"
        )
        mask_sigma = _optional_positive_float_parameter(context, "template_mask_sigma_px")
        emdb_id = _optional_string_parameter(context, "template_emdb_id")

        paths = _template_paths(

            context.output_dir,

            manifest,

            backend_name=self.name,

            label=variant_label(context),

        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)

        create_template = _resolve_pytom_executable(
            context, "pytom_create_template_executable", default_name="pytom_create_template.py"
        )
        create_mask = _resolve_pytom_executable(
            context, "pytom_create_mask_executable", default_name="pytom_create_mask.py"
        )

        command_log: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]] = []
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-template-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_template = temporary_root / "template.mrc"
            temporary_mask = temporary_root / "mask.mrc"

            template_command = [
                str(create_template),
                "-i",
                str(source_map.resolve()),
                "-o",
                str(temporary_template),
                "--output-voxel-size-angstrom",
                f"{output_voxel_size:.6f}",
            ]
            if input_voxel_size is not None:
                template_command.extend(
                    ["--input-voxel-size-angstrom", f"{input_voxel_size:.6f}"]
                )
            if low_pass is not None:
                template_command.extend(["--low-pass", f"{low_pass:.6f}"])
            if box_size is not None:
                template_command.extend(["-b", str(box_size)])
            if center:
                template_command.append("--center")
            if invert:
                template_command.append("--invert")
            if mirror:
                template_command.append("--mirror")

            result = self._command_runner(
                template_command, cwd=temporary_root, env=os.environ.copy()
            )
            command_log.append((template_command, result))
            if result.returncode != 0:
                _write_command_log(paths.log, command_log)
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"pytom_create_template.py failed with exit code {result.returncode}: "
                    f"{detail}; see {paths.log}"
                )
            if not temporary_template.is_file():
                _write_command_log(paths.log, command_log)
                raise RuntimeError(
                    f"pytom_create_template.py did not write {temporary_template}"
                )

            template_statistics = _inspect_volume(temporary_template)
            actual_box_size = template_statistics.box_size
            mask_radius = _mask_radius_px(
                context,
                voxel_size_angstrom=output_voxel_size,
                box_size=actual_box_size,
            )

            mask_command = [
                str(create_mask),
                "-b",
                str(actual_box_size),
                "-o",
                str(temporary_mask),
                "-r",
                f"{mask_radius:.6f}",
                "--voxel-size",
                f"{output_voxel_size:.6f}",
            ]
            if mask_sigma is not None:
                mask_command.extend(["-s", f"{mask_sigma:.6f}"])

            result = self._command_runner(mask_command, cwd=temporary_root, env=os.environ.copy())
            command_log.append((mask_command, result))
            _write_command_log(paths.log, command_log)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"pytom_create_mask.py failed with exit code {result.returncode}: "
                    f"{detail}; see {paths.log}"
                )
            if not temporary_mask.is_file():
                raise RuntimeError(f"pytom_create_mask.py did not write {temporary_mask}")

            mask_statistics = _inspect_volume(temporary_mask)
            if mask_statistics.box_size != actual_box_size:
                raise RuntimeError(
                    f"mask box size {mask_statistics.box_size} does not match the template's "
                    f"{actual_box_size}; template matching requires them to agree"
                )

            _replace_output(temporary_template, paths.template)
            _replace_output(temporary_mask, paths.mask)

        warnings: list[str] = []
        status = QcStatus.PASS
        if not _contrast_matches_inversion(template_statistics, inverted=invert):
            status = QcStatus.WARNING
            expected = "dark (negative)" if invert else "bright (positive)"
            warnings.append(
                f"template density does not look {expected} after applying invert={invert}; "
                "check that the contrast matches the tomogram, since a mismatch makes "
                "template matching quietly find nothing"
            )

        template_id = (

            f"{manifest.tilt_series_id}:template:{self.name}"

            f"{variant_suffix(context)}"

        )
        mask_id = (
            f"{manifest.tilt_series_id}:template_mask:{self.name}"
            f"{variant_suffix(context)}"
        )
        report = TemplatePreparationQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            source_map_path=str(source_map.resolve()),
            emdb_id=emdb_id,
            template_id=template_id,
            template_mask_id=mask_id,
            input_voxel_size_angstrom=(
                input_voxel_size
                if input_voxel_size is not None
                else _source_voxel_size(source_map)
            ),
            output_voxel_size_angstrom=output_voxel_size,
            box_size_px=actual_box_size,
            inverted=invert,
            low_pass_angstrom=low_pass,
            mask_radius_px=mask_radius,
            mask_sigma_px=mask_sigma,
            template_minimum=template_statistics.minimum,
            template_maximum=template_statistics.maximum,
            template_median=template_statistics.median,
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        shared_parameters = {
            "backend": self.name,
            "tilt_series_id": manifest.tilt_series_id,
            "tomogram_id": tomogram.id,
            "source_map_path": str(source_map.resolve()),
            "emdb_id": emdb_id,
            "box_size_px": actual_box_size,
            "inverted": invert,
            "command_log_path": str(paths.log),
            "qc_path": str(paths.report),
        }
        template_artifact = Artifact(
            id=template_id,
            variant=variant_label(context),
            kind=ArtifactKind.TEMPLATE,
            path=paths.template,
            parent_ids=[tomogram.id],
            shape=(actual_box_size, actual_box_size, actual_box_size),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=output_voxel_size,
            parameters={
                **shared_parameters,
                "low_pass_angstrom": low_pass,
                "centered": center,
                "mirrored": mirror,
            },
            software_versions={"pytom-match-pick": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.template.stat().st_size,
        )
        mask_artifact = Artifact(
            id=mask_id,
            variant=variant_label(context),
            kind=ArtifactKind.TEMPLATE_MASK,
            path=paths.mask,
            parent_ids=[template_artifact.id],
            shape=(actual_box_size, actual_box_size, actual_box_size),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=output_voxel_size,
            parameters={
                **shared_parameters,
                "template_id": template_artifact.id,
                "mask_radius_px": mask_radius,
                "mask_sigma_px": mask_sigma,
            },
            software_versions={"pytom-match-pick": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.mask.stat().st_size,
        )
        qc_artifact = Artifact(
            id=(
                f"{manifest.tilt_series_id}:qc:template_preparation:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[template_artifact.id],
            parameters={
                "qc_type": "template_preparation",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "status": status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [template_artifact, mask_artifact, qc_artifact]


def prepare_template_and_register(
    backend: TemplatePreparationBackend,
    source_map: Path,
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Prepare a template and register it, its mask, and the QC artifact."""

    artifacts = backend.prepare(source_map, tomogram, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


@dataclass(frozen=True)
class _VolumeStatistics:
    box_size: int
    minimum: float
    maximum: float
    median: float
    voxel_size_angstrom: float


def _inspect_volume(path: Path) -> _VolumeStatistics:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x)
    if data.ndim != 3:
        raise RuntimeError(f"expected a 3D volume, got shape {data.shape}: {path}")
    if len(set(data.shape)) != 1:
        raise RuntimeError(f"expected a cubic volume, got shape {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"volume contains nonfinite values: {path}")
    return _VolumeStatistics(
        box_size=int(data.shape[0]),
        minimum=float(data.min()),
        maximum=float(data.max()),
        median=float(np.median(data)),
        voxel_size_angstrom=voxel_size,
    )


def _contrast_matches_inversion(
    statistics: _VolumeStatistics,
    *,
    inverted: bool,
) -> bool:
    """Check the template's density points the way the invert flag implies.

    A reference map is a compact blob against a flat background, so the signal
    sits on one side of the median. After `--invert` the blob should be the
    darker excursion; without it, the brighter one.
    """

    below = statistics.median - statistics.minimum
    above = statistics.maximum - statistics.median
    if math.isclose(below, above, rel_tol=1e-6, abs_tol=1e-12):
        return True
    return below > above if inverted else above > below


def _source_voxel_size(path: Path) -> float:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(
            f"reference map has no usable voxel size: {path}; "
            "pass template_input_voxel_size_angstrom explicitly"
        )
    return voxel_size


def _mask_radius_px(
    context: BackendContext,
    *,
    voxel_size_angstrom: float,
    box_size: int,
) -> float:
    explicit = _optional_positive_float_parameter(context, "template_mask_radius_px")
    if explicit is not None:
        radius = explicit
    else:
        diameter = _optional_positive_float_parameter(
            context, "template_particle_diameter_angstrom"
        )
        if diameter is None:
            raise TypeError(
                "provide either 'template_mask_radius_px' or "
                "'template_particle_diameter_angstrom' to size the mask"
            )
        radius = diameter / 2.0 / voxel_size_angstrom
    if radius * 2.0 > box_size:
        raise ValueError(
            f"mask diameter {radius * 2.0:.1f} px does not fit in the template box of "
            f"{box_size} px; enlarge the box with template_box_size_px"
        )
    return radius


@dataclass(frozen=True)
class _TemplatePaths:
    output_dir: Path
    template: Path
    mask: Path
    log: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.template, self.mask, self.log, self.report)


def _template_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    label: str | None = None,
) -> _TemplatePaths:
    template_dir = output_root / "templates" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "template"
    prefix = f"{manifest.tilt_series_id}_{backend_name}" + (
        f"_{label}" if label else ""
    )
    return _TemplatePaths(
        output_dir=template_dir,
        template=template_dir / f"{prefix}_template.mrc",
        mask=template_dir / f"{prefix}_mask.mrc",
        log=template_dir / f"{prefix}.log",
        report=qc_dir / f"{prefix}_qc.json",
    )


def _validate_tomogram(tomogram: Artifact, manifest: TiltSeriesManifest) -> None:
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if tomogram.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError(
            f"tomogram is for {tomogram.parameters.get('tilt_series_id')}, "
            f"expected {manifest.tilt_series_id}"
        )


def _tomogram_voxel_size(tomogram: Artifact) -> float:
    value = tomogram.pixel_spacing_angstrom
    if value is None:
        raise ValueError("tomogram voxel spacing is required to scale the template")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tomogram voxel spacing must be finite and positive")
    return value


def _resolve_pytom_executable(
    context: BackendContext,
    key: str,
    *,
    default_name: str,
) -> Path:
    configured = context.parameters.get(key)
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError(f"context parameter {key!r} must be a path string")
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"pytom-match-pick executable not found: {candidate}")
        return candidate.resolve()
    resolved = shutil.which(default_name)
    if resolved is not None:
        return Path(resolved)
    raise FileNotFoundError(
        f"{default_name} not found; pass --{key.replace('_', '-')} or add it to PATH"
    )


def _write_command_log(
    path: Path,
    command_results: Sequence[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
) -> None:
    sections = []
    for command, result in command_results:
        sections.append(
            f"$ {shlex.join(str(token) for token in command)}\n\n"
            f"[stdout]\n{result.stdout or ''}\n"
            f"[stderr]\n{result.stderr or ''}\n"
            f"[exit_code]\n{result.returncode}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _replace_output(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"template outputs already exist: {joined}")


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _optional_positive_float_parameter(context: BackendContext, key: str) -> float | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and greater than zero")
    return normalized


def _optional_positive_int_parameter(context: BackendContext, key: str) -> int | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < 1:
        raise ValueError(f"context parameter {key!r} must be at least 1")
    return value


def _optional_string_parameter(context: BackendContext, key: str) -> str | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"context parameter {key!r} must be a string")
    return value
