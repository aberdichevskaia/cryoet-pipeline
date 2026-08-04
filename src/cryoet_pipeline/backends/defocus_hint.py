from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
from cryoet_pipeline.backends.protocols import BackendContext, DefocusEstimationBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    DefocusHintQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)


class Ctffind5DefocusHintBackend:
    """Estimate a single-image defocus hint with CTFFIND5, for ctfplotter seeding."""

    name = "ctffind5"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def estimate(
        self,
        corrected_projection: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a defocus-hint artifact and its QC artifact."""

        _validate_corrected_projection(corrected_projection, manifest)
        pixel_spacing_angstrom = _voxel_spacing(corrected_projection)
        tilt_angle_deg = _tilt_angle(corrected_projection)
        overwrite = _bool_parameter(context, "overwrite", default=False)

        voltage_kv = _required_positive_float_parameter(context, "ctf_voltage_kv")
        spherical_aberration_mm = _required_positive_float_parameter(
            context,
            "ctf_spherical_aberration_mm",
        )
        amplitude_contrast = _bounded_float_parameter(
            context,
            "ctf_amplitude_contrast",
            default=0.07,
            minimum=0.0,
            maximum=1.0,
        )
        box_size = _positive_int_parameter(context, "defocus_hint_box_size", default=512)
        min_resolution_angstrom = _positive_float_parameter(
            context,
            "defocus_hint_min_resolution_angstrom",
            default=30.0,
        )
        max_resolution_angstrom = _positive_float_parameter(
            context,
            "defocus_hint_max_resolution_angstrom",
            default=5.0,
        )
        min_defocus_angstrom = _positive_float_parameter(
            context,
            "defocus_hint_min_defocus_angstrom",
            default=5000.0,
        )
        max_defocus_angstrom = _positive_float_parameter(
            context,
            "defocus_hint_max_defocus_angstrom",
            default=80000.0,
        )
        step_angstrom = _positive_float_parameter(
            context,
            "defocus_hint_step_angstrom",
            default=500.0,
        )

        paths = _defocus_hint_paths(context.output_dir, manifest)
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)
        executable = _resolve_ctffind_executable(context)

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-defocus-hint-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_input = temporary_root / f"{manifest.tilt_series_id}.mrc"
            temporary_diagnostic = temporary_root / f"{manifest.tilt_series_id}_ctffind"
            temporary_summary = temporary_root / f"{manifest.tilt_series_id}_ctffind.txt"
            temporary_log = temporary_root / f"{manifest.tilt_series_id}_ctffind.log"
            temporary_report = temporary_root / "defocus_hint_qc.json"

            _materialize_projection_mrc(
                corrected_projection.path,
                temporary_input,
                voxel_spacing_angstrom=pixel_spacing_angstrom,
            )

            stdin_text = _ctffind_stdin(
                input_path=temporary_input,
                diagnostic_path=temporary_diagnostic,
                pixel_spacing_angstrom=pixel_spacing_angstrom,
                voltage_kv=voltage_kv,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast=amplitude_contrast,
                box_size=box_size,
                min_resolution_angstrom=min_resolution_angstrom,
                max_resolution_angstrom=max_resolution_angstrom,
                min_defocus_angstrom=min_defocus_angstrom,
                max_defocus_angstrom=max_defocus_angstrom,
                step_angstrom=step_angstrom,
            )
            result = self._command_runner(
                [str(executable)],
                cwd=temporary_root,
                env={},
                stdin=stdin_text,
            )
            temporary_log.write_text(
                f"[stdin]\n{stdin_text}\n\n"
                f"[stdout]\n{result.stdout or ''}\n"
                f"[stderr]\n{result.stderr or ''}\n"
                f"[exit_code]\n{result.returncode}\n"
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"CTFFIND5 failed with exit code {result.returncode}; see {temporary_log}"
                )
            if not temporary_summary.is_file():
                raise RuntimeError(f"CTFFIND5 did not write a summary file: {temporary_summary}")

            fit = _parse_ctffind_summary(temporary_summary)
            defocus_um = (fit.defocus1_angstrom + fit.defocus2_angstrom) / 2.0 / 10000.0
            astigmatism_um = abs(fit.defocus1_angstrom - fit.defocus2_angstrom) / 2.0 / 10000.0

            defocus_hint_id = f"{manifest.tilt_series_id}:defocus_hint:ctffind5"
            report = DefocusHintQc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                input_projection_id=corrected_projection.id,
                defocus_hint_id=defocus_hint_id,
                tilt_angle_deg=tilt_angle_deg,
                defocus_um=defocus_um,
                astigmatism_um=astigmatism_um,
                azimuth_deg=fit.azimuth_deg,
                cross_correlation=fit.cross_correlation,
                fit_resolution_angstrom=fit.fit_resolution_angstrom,
                status=QcStatus.PASS,
            )
            temporary_report.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )

            _replace_output(temporary_summary, paths.summary)
            _replace_output(temporary_log, paths.log)
            _replace_output(temporary_report, paths.report)

        defocus_hint_artifact = Artifact(
            id=defocus_hint_id,
            kind=ArtifactKind.DEFOCUS_HINT,
            path=paths.summary,
            parent_ids=[corrected_projection.id],
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "input_projection_id": corrected_projection.id,
                "tilt_angle_deg": tilt_angle_deg,
                "defocus_um": defocus_um,
                "astigmatism_um": astigmatism_um,
                "azimuth_deg": fit.azimuth_deg,
                "cross_correlation": fit.cross_correlation,
                "fit_resolution_angstrom": fit.fit_resolution_angstrom,
                "voltage_kv": voltage_kv,
                "spherical_aberration_mm": spherical_aberration_mm,
                "amplitude_contrast": amplitude_contrast,
                "log_path": str(paths.log),
            },
            software_versions={"ctffind": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(paths.summary, paths.log),
        )
        qc_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:defocus_hint:{self.name}",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[defocus_hint_artifact.id],
            parameters={
                "qc_type": "defocus_hint",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "status": QcStatus.PASS.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [defocus_hint_artifact, qc_artifact]


def estimate_defocus_hint_and_register(
    backend: DefocusEstimationBackend,
    corrected_projection: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Estimate a defocus hint and register it plus its QC artifact."""

    artifacts = backend.estimate(corrected_projection, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _DefocusHintPaths:
    def __init__(self, *, output_dir: Path, summary: Path, log: Path, report: Path) -> None:
        self.output_dir = output_dir
        self.summary = summary
        self.log = log
        self.report = report

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.summary, self.log, self.report)


class _CtffindFit:
    def __init__(
        self,
        *,
        defocus1_angstrom: float,
        defocus2_angstrom: float,
        azimuth_deg: float,
        cross_correlation: float,
        fit_resolution_angstrom: float,
    ) -> None:
        self.defocus1_angstrom = defocus1_angstrom
        self.defocus2_angstrom = defocus2_angstrom
        self.azimuth_deg = azimuth_deg
        self.cross_correlation = cross_correlation
        self.fit_resolution_angstrom = fit_resolution_angstrom


def _defocus_hint_paths(output_root: Path, manifest: TiltSeriesManifest) -> _DefocusHintPaths:
    ctf_dir = output_root / "ctf" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "defocus_hint"
    prefix = f"{manifest.tilt_series_id}_defocus_hint"
    return _DefocusHintPaths(
        output_dir=ctf_dir,
        summary=ctf_dir / f"{prefix}.txt",
        log=ctf_dir / f"{prefix}_ctffind.log",
        report=qc_dir / f"{prefix}_qc.json",
    )


def _validate_corrected_projection(
    corrected_projection: Artifact,
    manifest: TiltSeriesManifest,
) -> None:
    if corrected_projection.kind is not ArtifactKind.CORRECTED_PROJECTION:
        raise ValueError(
            f"expected corrected projection artifact, got {corrected_projection.kind}"
        )
    if corrected_projection.parameters.get("z_value") is None:
        raise ValueError("corrected projection must record z_value")
    if not corrected_projection.path.exists():
        raise FileNotFoundError(f"corrected projection not found: {corrected_projection.path}")


def _voxel_spacing(corrected_projection: Artifact) -> float:
    value = corrected_projection.pixel_spacing_angstrom
    if value is None:
        raise ValueError("corrected projection voxel spacing is required for CTFFIND5")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("corrected projection voxel spacing must be finite and positive")
    return value


def _tilt_angle(corrected_projection: Artifact) -> float:
    value = corrected_projection.parameters.get("tilt_angle_deg")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("corrected projection must record a numeric tilt_angle_deg")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("corrected projection tilt_angle_deg must be finite")
    return normalized


def _materialize_projection_mrc(
    source_path: Path,
    output_path: Path,
    *,
    voxel_spacing_angstrom: float,
) -> None:
    if source_path.suffix == ".zarr" or source_path.is_dir():
        array = cast(Any, zarr.open(source_path, mode="r"))
        data = np.asarray(array[:], dtype=np.float32)
    else:
        with mrcfile.open(source_path, permissive=True) as mrc:
            data = np.asarray(mrc.data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected a 2D projection, got shape {data.shape} from {source_path}")
    if not np.isfinite(data).all():
        raise ValueError("corrected projection contains nonfinite values")
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_spacing_angstrom


def _ctffind_stdin(
    *,
    input_path: Path,
    diagnostic_path: Path,
    pixel_spacing_angstrom: float,
    voltage_kv: float,
    spherical_aberration_mm: float,
    amplitude_contrast: float,
    box_size: int,
    min_resolution_angstrom: float,
    max_resolution_angstrom: float,
    min_defocus_angstrom: float,
    max_defocus_angstrom: float,
    step_angstrom: float,
) -> str:
    """Build the interactive answer sequence for a single, non-movie CTFFIND5 run."""

    answers = [
        str(input_path),
        "no",  # is this a movie stack of frames
        str(diagnostic_path),
        f"{pixel_spacing_angstrom:.6f}",
        f"{voltage_kv:.6f}",
        f"{spherical_aberration_mm:.6f}",
        f"{amplitude_contrast:.6f}",
        str(box_size),
        f"{min_resolution_angstrom:.6f}",
        f"{max_resolution_angstrom:.6f}",
        f"{min_defocus_angstrom:.6f}",
        f"{max_defocus_angstrom:.6f}",
        f"{step_angstrom:.6f}",
        "no",  # do you know astigmatism ahead of time
        "no",  # slower, more exhaustive search
        "no",  # restrain astigmatism
        "no",  # find additional phase shift
        "no",  # determine sample tilt (CTFFIND5)
        "no",  # determine sample thickness (CTFFIND5)
        "no",  # set expert options
    ]
    return "\n".join(answers) + "\n"


def _parse_ctffind_summary(path: Path) -> _CtffindFit:
    data_lines = [
        line for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")
    ]
    if not data_lines:
        raise ValueError(f"CTFFIND5 summary file has no data rows: {path}")
    columns = data_lines[0].split()
    if len(columns) < 7:
        raise ValueError(f"CTFFIND5 summary file has too few columns: {path}")
    return _CtffindFit(
        defocus1_angstrom=float(columns[1]),
        defocus2_angstrom=float(columns[2]),
        azimuth_deg=float(columns[3]),
        cross_correlation=float(columns[5]),
        fit_resolution_angstrom=float(columns[6]),
    )


def _resolve_ctffind_executable(context: BackendContext) -> Path:
    configured = context.parameters.get("ctffind_executable")
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError("context parameter 'ctffind_executable' must be a path string")
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"CTFFIND executable not found: {candidate}")
        return candidate.resolve()
    for name in ("ctffind5", "ctffind"):
        resolved = shutil.which(name)
        if resolved is not None:
            return Path(resolved)
    raise FileNotFoundError(
        "CTFFIND executable not found; pass --ctffind-executable or add it to PATH"
    )


def _replace_output(source: Path, destination: Path) -> None:
    if destination.exists():
        _remove_path(destination)
    source.replace(destination)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _require_available_paths(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"defocus hint outputs already exist: {joined}")


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _positive_int_parameter(context: BackendContext, key: str, *, default: int) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < 1:
        raise ValueError(f"context parameter {key!r} must be at least 1")
    return value


def _positive_float_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and positive")
    return normalized


def _required_positive_float_parameter(context: BackendContext, key: str) -> float:
    value = context.parameters.get(key)
    if value is None:
        raise TypeError(f"context parameter {key!r} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and positive")
    return normalized


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


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)
