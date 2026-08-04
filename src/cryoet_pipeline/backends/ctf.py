from __future__ import annotations

import json
import math
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    resolve_imod_executable,
    run_command,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.protocols import BackendContext, CtfCorrectionBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    CtfEstimationQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc


class ImodCtfPlotterBackend:
    """Estimate and phase-flip-correct CTF with IMOD ctfplotter/ctfphaseflip."""

    name = "imod-ctfplotter"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def correct(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a CTF model, a CTF-corrected aligned stack, and a QC artifact."""

        stack_shape = _validate_final_aligned_stack(tilt_stack)
        alignment_result = _load_alignment(alignment, tilt_stack, manifest)
        pixel_spacing_angstrom = _voxel_spacing(tilt_stack)
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
        expected_defocus_um = _expected_defocus_um(context, manifest)
        defocus_tol_nm = _positive_float_parameter(context, "ctf_defocus_tol_nm", default=200.0)
        left_defocus_tol_nm = _positive_float_parameter(
            context,
            "ctf_left_defocus_tol_nm",
            default=2000.0,
        )
        right_defocus_tol_nm = _positive_float_parameter(
            context,
            "ctf_right_defocus_tol_nm",
            default=2000.0,
        )
        tile_size = _positive_int_parameter(context, "ctf_tile_size", default=256)
        autofit_range_deg = _positive_float_parameter(
            context,
            "ctf_autofit_range_deg",
            default=16.0,
        )
        autofit_step_deg = _positive_float_parameter(context, "ctf_autofit_step_deg", default=2.0)
        ps_resolution = _positive_int_parameter(context, "ctf_ps_resolution", default=101)
        interpolation_width = _positive_int_parameter(
            context,
            "ctf_interpolation_width",
            default=15,
        )

        paths = _ctf_paths(context.output_dir, manifest)
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)
        ctfplotter = resolve_imod_executable("ctfplotter", context)
        ctfphaseflip = resolve_imod_executable("ctfphaseflip", context)

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-ctf-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_angles = temporary_root / f"{manifest.tilt_series_id}.tlt"
            temporary_defocus = temporary_root / f"{manifest.tilt_series_id}.defocus"
            temporary_corrected = temporary_root / f"{manifest.tilt_series_id}_ctf.st"
            temporary_report = temporary_root / "ctf_qc.json"

            write_tilt_angles_file(
                temporary_angles,
                [transform.tilt_angle_deg for transform in alignment_result.transforms],
            )

            plot_command = [
                str(ctfplotter),
                "-InputStack",
                str(tilt_stack.path.resolve()),
                "-AngleFile",
                str(temporary_angles),
                "-DefocusFile",
                str(temporary_defocus),
                "-AxisAngle",
                f"{alignment_result.tilt_axis_angle_deg:.6f}",
                "-PixelSize",
                f"{pixel_spacing_angstrom / 10.0:.6f}",
                "-ExpectedDefocus",
                f"{expected_defocus_um * 1000.0:.6f}",
                "-AutoFitRangeAndStep",
                f"{autofit_range_deg:.6f},{autofit_step_deg:.6f}",
                "-Voltage",
                str(int(round(voltage_kv))),
                "-SphericalAberration",
                f"{spherical_aberration_mm:.6f}",
                "-AmplitudeContrast",
                f"{amplitude_contrast:.6f}",
                "-DefocusTol",
                str(int(round(defocus_tol_nm))),
                "-PSResolution",
                str(ps_resolution),
                "-LeftDefTol",
                f"{left_defocus_tol_nm:.6f}",
                "-RightDefTol",
                f"{right_defocus_tol_nm:.6f}",
                "-tileSize",
                str(tile_size),
                "-SaveAndExit",
            ]
            plot_result = self._command_runner(
                plot_command,
                cwd=temporary_root,
                env=imod_environment(ctfplotter, context),
            )
            _write_command_log(paths.plot_log, plot_command, plot_result)
            if plot_result.returncode != 0:
                raise RuntimeError(
                    f"IMOD ctfplotter failed with exit code {plot_result.returncode}; "
                    f"see {paths.plot_log}"
                )
            if not temporary_defocus.is_file():
                raise RuntimeError(
                    f"IMOD ctfplotter did not write a defocus file: {temporary_defocus}"
                )

            defocus_values_nm = _parse_defocus_file(temporary_defocus)
            if not defocus_values_nm:
                raise RuntimeError(f"IMOD ctfplotter defocus file is empty: {temporary_defocus}")

            flip_command = [
                str(ctfphaseflip),
                "-InputStack",
                str(tilt_stack.path.resolve()),
                "-AngleFile",
                str(temporary_angles),
                "-OutputFileName",
                str(temporary_corrected),
                "-DefocusFile",
                str(temporary_defocus),
                "-Voltage",
                str(int(round(voltage_kv))),
                "-SphericalAberration",
                f"{spherical_aberration_mm:.6f}",
                "-DefocusTol",
                str(int(round(defocus_tol_nm))),
                "-PixelSize",
                f"{pixel_spacing_angstrom / 10.0:.6f}",
                "-AmplitudeContrast",
                f"{amplitude_contrast:.6f}",
                "-InterpolationWidth",
                str(interpolation_width),
            ]
            flip_result = self._command_runner(
                flip_command,
                cwd=temporary_root,
                env=imod_environment(ctfphaseflip, context),
            )
            _write_command_log(paths.flip_log, flip_command, flip_result)
            if flip_result.returncode != 0:
                raise RuntimeError(
                    f"IMOD ctfphaseflip failed with exit code {flip_result.returncode}; "
                    f"see {paths.flip_log}"
                )
            if not temporary_corrected.is_file():
                raise RuntimeError(
                    f"IMOD ctfphaseflip did not write a corrected stack: {temporary_corrected}"
                )

            info = validate_complete_mrc(temporary_corrected)
            if info.shape != stack_shape:
                raise ValueError(
                    f"CTF-corrected stack has shape {info.shape}, expected {stack_shape}"
                )

            defocus_values_um = [value / 1000.0 for value in defocus_values_nm]
            ctf_model_id = f"{manifest.tilt_series_id}:ctf_model:fine"
            corrected_stack_id = f"{manifest.tilt_series_id}:aligned_tilt_stack:ctf_corrected"
            report = CtfEstimationQc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                input_stack_id=tilt_stack.id,
                ctf_model_id=ctf_model_id,
                corrected_stack_id=corrected_stack_id,
                num_tilts=stack_shape[0],
                voltage_kv=voltage_kv,
                spherical_aberration_mm=spherical_aberration_mm,
                amplitude_contrast=amplitude_contrast,
                axis_angle_deg=alignment_result.tilt_axis_angle_deg,
                mean_defocus_um=sum(defocus_values_um) / len(defocus_values_um),
                minimum_defocus_um=min(defocus_values_um),
                maximum_defocus_um=max(defocus_values_um),
                status=QcStatus.PASS,
            )
            temporary_report.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )

            _replace_output(temporary_defocus, paths.defocus)
            _replace_output(temporary_corrected, paths.corrected_stack)
            _replace_output(temporary_angles, paths.tilt_angles)
            _replace_output(temporary_report, paths.report)

        ctf_model_artifact = Artifact(
            id=ctf_model_id,
            kind=ArtifactKind.CTF_MODEL,
            path=paths.defocus,
            parent_ids=[tilt_stack.id, alignment.id],
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "axis_angle_deg": alignment_result.tilt_axis_angle_deg,
                "voltage_kv": voltage_kv,
                "spherical_aberration_mm": spherical_aberration_mm,
                "amplitude_contrast": amplitude_contrast,
                "plot_log_path": str(paths.plot_log),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.defocus.stat().st_size,
        )
        corrected_stack_artifact = Artifact(
            id=corrected_stack_id,
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=paths.corrected_stack,
            parent_ids=[tilt_stack.id, ctf_model_artifact.id],
            shape=info.shape,
            dtype=info.dtype,
            axis_order=AxisOrder.TYX,
            pixel_spacing_angstrom=tilt_stack.pixel_spacing_angstrom,
            binning=tilt_stack.binning,
            parameters={
                **{
                    key: value
                    for key, value in tilt_stack.parameters.items()
                    if key != "purpose"
                },
                "purpose": "ctf_corrected",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "ctf_model_id": ctf_model_artifact.id,
                "flip_log_path": str(paths.flip_log),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CACHE,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.corrected_stack,
                paths.tilt_angles,
                paths.flip_log,
            ),
        )
        qc_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:ctf:{self.name}",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[corrected_stack_artifact.id],
            parameters={
                "qc_type": "ctf_estimation",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "status": QcStatus.PASS.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [ctf_model_artifact, corrected_stack_artifact, qc_artifact]


def correct_ctf_and_register(
    backend: CtfCorrectionBackend,
    tilt_stack: Artifact,
    alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Estimate/correct CTF and register the model, corrected stack, and QC artifacts."""

    artifacts = backend.correct(tilt_stack, alignment, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _CtfPaths:
    def __init__(
        self,
        *,
        output_dir: Path,
        defocus: Path,
        corrected_stack: Path,
        tilt_angles: Path,
        plot_log: Path,
        flip_log: Path,
        report: Path,
    ) -> None:
        self.output_dir = output_dir
        self.defocus = defocus
        self.corrected_stack = corrected_stack
        self.tilt_angles = tilt_angles
        self.plot_log = plot_log
        self.flip_log = flip_log
        self.report = report

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.defocus,
            self.corrected_stack,
            self.tilt_angles,
            self.plot_log,
            self.flip_log,
            self.report,
        )


def _ctf_paths(output_root: Path, manifest: TiltSeriesManifest) -> _CtfPaths:
    ctf_dir = output_root / "ctf" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "ctf"
    prefix = f"{manifest.tilt_series_id}_fine"
    return _CtfPaths(
        output_dir=ctf_dir,
        defocus=ctf_dir / f"{prefix}.defocus",
        corrected_stack=ctf_dir / f"{prefix}_ctf.st",
        tilt_angles=ctf_dir / f"{prefix}.tlt",
        plot_log=ctf_dir / f"{prefix}_ctfplotter.log",
        flip_log=ctf_dir / f"{prefix}_ctfphaseflip.log",
        report=qc_dir / f"{prefix}_qc.json",
    )


def _validate_final_aligned_stack(tilt_stack: Artifact) -> tuple[int, int, int]:
    if tilt_stack.kind is not ArtifactKind.ALIGNED_TILT_STACK:
        raise ValueError(f"expected aligned tilt stack artifact, got {tilt_stack.kind}")
    if tilt_stack.parameters.get("purpose") != "final_alignment":
        raise ValueError("CTF correction requires the plain final fine-aligned stack")
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
        raise ValueError("CTF correction requires fine alignment")
    return result


def _voxel_spacing(tilt_stack: Artifact) -> float:
    pixel_spacing = tilt_stack.pixel_spacing_angstrom
    if pixel_spacing is None:
        raise ValueError("aligned stack voxel spacing is required for CTF correction")
    if not math.isfinite(pixel_spacing) or pixel_spacing <= 0:
        raise ValueError("aligned stack voxel spacing must be finite and positive")
    return pixel_spacing


def _expected_defocus_um(context: BackendContext, manifest: TiltSeriesManifest) -> float:
    override = context.parameters.get("ctf_expected_defocus_um")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, (int, float)):
            raise TypeError("context parameter 'ctf_expected_defocus_um' must be numeric")
        value = float(override)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("context parameter 'ctf_expected_defocus_um' must be positive")
        return value
    candidates = [image for image in manifest.images if image.defocus_um is not None]
    if not candidates:
        raise ValueError(
            "no nominal defocus recorded in the manifest; "
            "pass ctf_expected_defocus_um explicitly"
        )
    closest = min(candidates, key=lambda image: abs(image.tilt_angle_deg))
    nominal = closest.defocus_um
    if nominal is None or not math.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("manifest nominal defocus must be finite and positive")
    return nominal


def _parse_defocus_file(path: Path) -> list[float]:
    """Parse the plain (non-astigmatism) 5-column IMOD ctfplotter defocus file."""

    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return []
    second_line_columns = lines[1].split() if len(lines) > 1 else []
    is_plain_format = len(lines) == 1 or len(second_line_columns) == 5
    if not is_plain_format:
        raise ValueError(
            f"unsupported IMOD defocus file format in {path}: expected the plain "
            "5-column format (no astigmatism/phase/cut-on search)"
        )
    defocus_nm: list[float] = []
    for index, line in enumerate(lines):
        columns = line.split()
        if index == 0:
            columns = columns[:-1]
        if len(columns) != 5:
            raise ValueError(f"{path}: line {index + 1}: expected 5 columns, got {len(columns)}")
        defocus_nm.append(float(columns[4]))
    return defocus_nm


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
        raise FileExistsError(f"CTF correction outputs already exist: {joined}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


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
