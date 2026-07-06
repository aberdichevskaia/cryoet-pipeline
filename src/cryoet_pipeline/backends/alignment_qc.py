from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    parse_imod_xf,
    resolve_imod_executable,
    run_command,
    write_binned_mrc_stack,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.protocols import BackendContext, CoarseAlignmentQcBackend
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    CoarseAlignmentQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc


class ImodCoarseAlignmentQcBackend:
    """Apply coarse transforms with IMOD and measure residual alignment shifts."""

    name = "imod_coarse_alignment_qc"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def evaluate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Write a reduced prealigned stack and a machine-readable QC report."""

        stack_shape = _validate_stack(tilt_stack, manifest)
        alignment_result = _load_alignment(alignment, tilt_stack, manifest)
        preview_binning = _positive_int_parameter(
            context,
            "preview_binning",
            default=16,
        )
        _validate_preview_binning(stack_shape, preview_binning)
        residual_warning_px = _nonnegative_float_parameter(
            context,
            "residual_warning_px",
            default=2.0,
        )
        residual_fail_px = _nonnegative_float_parameter(
            context,
            "residual_fail_px",
            default=5.0,
        )
        if residual_fail_px < residual_warning_px:
            raise ValueError(
                "residual_fail_px must be greater than or equal to "
                "residual_warning_px"
            )

        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _qc_paths(
            context.output_dir,
            manifest,
            preview_binning=preview_binning,
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        newstack_executable = resolve_imod_executable("newstack", context)
        tiltxcorr_executable = resolve_imod_executable("tiltxcorr", context)

        transforms_by_z = {
            transform.z_value: transform
            for transform in alignment_result.transforms
        }
        excluded_z_values = set(alignment_result.excluded_z_values)
        included_indices = [
            index
            for index, image in enumerate(manifest.images)
            if image.z_value not in excluded_z_values
        ]
        if not included_indices:
            raise ValueError("coarse alignment excludes every tilt image")
        preview_order = sorted(
            included_indices,
            key=lambda index: manifest.images[index].tilt_angle_deg,
        )

        paths.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-coarse-qc-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            input_stack_path = temporary_root / f"{manifest.tilt_series_id}_input.st"
            scaled_xf_path = temporary_root / f"{manifest.tilt_series_id}_bin.xf"
            tilt_angles_path = temporary_root / f"{manifest.tilt_series_id}.tlt"
            aligned_stack_path = temporary_root / f"{manifest.tilt_series_id}_aligned.st"
            residual_xf_path = temporary_root / f"{manifest.tilt_series_id}_residual.xf"

            write_binned_mrc_stack(
                tilt_stack.path,
                input_stack_path,
                binning=preview_binning,
                pixel_spacing_angstrom=tilt_stack.pixel_spacing_angstrom,
                section_indices=preview_order,
            )
            _write_scaled_xf(
                scaled_xf_path,
                preview_order,
                manifest,
                transforms_by_z,
                binning=preview_binning,
            )
            write_tilt_angles_file(
                tilt_angles_path,
                [
                    manifest.images[index].tilt_angle_deg
                    for index in preview_order
                ],
            )

            newstack_command = [
                str(newstack_executable),
                "-input",
                str(input_stack_path),
                "-output",
                str(aligned_stack_path),
                "-xform",
                str(scaled_xf_path),
                "-mode",
                "2",
            ]
            newstack_result = self._command_runner(
                newstack_command,
                cwd=temporary_root,
                env=imod_environment(newstack_executable, context),
            )
            _write_command_log(paths.newstack_log, newstack_command, newstack_result)
            _require_success(
                newstack_result,
                program="IMOD newstack",
                log_path=paths.newstack_log,
            )
            if not aligned_stack_path.is_file():
                raise RuntimeError(
                    f"IMOD newstack did not write aligned preview: {aligned_stack_path}"
                )

            residual_command = [
                str(tiltxcorr_executable),
                "-input",
                str(aligned_stack_path),
                "-output",
                str(residual_xf_path),
                "-tiltfile",
                str(tilt_angles_path),
                "-rotation",
                f"{alignment_result.tilt_axis_angle_deg:.6f}",
                "-verbose",
                "1",
            ]
            residual_result = self._command_runner(
                residual_command,
                cwd=temporary_root,
                env=imod_environment(tiltxcorr_executable, context),
            )
            _write_command_log(paths.residual_log, residual_command, residual_result)
            _require_success(
                residual_result,
                program="IMOD residual tiltxcorr",
                log_path=paths.residual_log,
            )
            if not residual_xf_path.is_file():
                raise RuntimeError(
                    f"IMOD residual tiltxcorr did not write transforms: "
                    f"{residual_xf_path}"
                )
            residual_transforms = parse_imod_xf(residual_xf_path)
            if len(residual_transforms) != len(preview_order):
                raise ValueError(
                    f"residual alignment returned {len(residual_transforms)} "
                    f"transforms for {len(preview_order)} included tilts"
                )

            if paths.preview.exists():
                paths.preview.unlink()
            aligned_stack_path.replace(paths.preview)

        residual_x = [transform[4] for transform in residual_transforms]
        residual_y = [transform[5] for transform in residual_transforms]
        residual_magnitude = np.hypot(residual_x, residual_y)
        residual_rms_px = float(np.sqrt(np.mean(np.square(residual_magnitude))))
        residual_p95_px = float(np.percentile(residual_magnitude, 95))
        residual_max_px = float(np.max(residual_magnitude))
        warnings: list[str] = []
        if alignment_result.excluded_z_values:
            warnings.append(
                "coarse alignment excluded low-variance tilt images: "
                f"{alignment_result.excluded_z_values}"
            )
        if residual_p95_px > residual_warning_px:
            warnings.append(
                f"residual p95 shift {residual_p95_px:.3f} px exceeds "
                f"warning threshold {residual_warning_px:.3f} px"
            )
        if residual_max_px > residual_fail_px:
            status = QcStatus.FAIL
            warnings.append(
                f"residual maximum shift {residual_max_px:.3f} px exceeds "
                f"failure threshold {residual_fail_px:.3f} px"
            )
        elif warnings:
            status = QcStatus.WARNING
        else:
            status = QcStatus.PASS

        included_z_values = [
            manifest.images[index].z_value for index in preview_order
        ]
        report = CoarseAlignmentQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            input_stack_id=tilt_stack.id,
            alignment_id=alignment.id,
            preview_path=paths.preview,
            preview_binning=preview_binning,
            included_z_values=included_z_values,
            excluded_z_values=alignment_result.excluded_z_values,
            residual_shift_x_px=residual_x,
            residual_shift_y_px=residual_y,
            residual_rms_px=residual_rms_px,
            residual_p95_px=residual_p95_px,
            residual_max_px=residual_max_px,
            input_max_abs_shift_fraction=_max_abs_shift_fraction(
                alignment_result,
                stack_shape,
            ),
            status=status,
            warnings=warnings,
        )
        _write_report(paths.report, report)
        preview_info = validate_complete_mrc(paths.preview)

        preview_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:coarse_alignment:preview",
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=paths.preview,
            parent_ids=[tilt_stack.id, alignment.id],
            shape=preview_info.shape,
            dtype=preview_info.dtype,
            axis_order=AxisOrder.TYX,
            pixel_spacing_angstrom=(
                tilt_stack.pixel_spacing_angstrom * preview_binning
                if tilt_stack.pixel_spacing_angstrom is not None
                else None
            ),
            binning=preview_binning,
            parameters={
                "qc_type": "coarse_alignment_preview",
                "tilt_series_id": manifest.tilt_series_id,
                "included_z_values": included_z_values,
                "excluded_z_values": alignment_result.excluded_z_values,
                "order": "tilt_angle_ascending",
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.preview.stat().st_size,
        )
        report_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:coarse_alignment:report",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[tilt_stack.id, alignment.id, preview_artifact.id],
            parameters={
                "qc_type": "coarse_alignment_report",
                "tilt_series_id": manifest.tilt_series_id,
                "status": report.status.value,
                "newstack_log_path": str(paths.newstack_log),
                "residual_log_path": str(paths.residual_log),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.report,
                paths.newstack_log,
                paths.residual_log,
            ),
        )
        return [preview_artifact, report_artifact]


def evaluate_coarse_alignment_and_register(
    backend: CoarseAlignmentQcBackend,
    tilt_stack: Artifact,
    alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Run coarse-alignment QC and register preview then report artifacts."""

    artifacts = backend.evaluate(tilt_stack, alignment, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _QcPaths:
    def __init__(
        self,
        output_dir: Path,
        preview: Path,
        report: Path,
        newstack_log: Path,
        residual_log: Path,
    ) -> None:
        self.output_dir = output_dir
        self.preview = preview
        self.report = report
        self.newstack_log = newstack_log
        self.residual_log = residual_log

    @property
    def outputs(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.preview,
            self.report,
            self.newstack_log,
            self.residual_log,
        )


def _qc_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    preview_binning: int,
) -> _QcPaths:
    output_dir = (
        output_root
        / "qc"
        / manifest.tilt_series_id
        / "coarse_alignment"
    )
    prefix = f"{manifest.tilt_series_id}_coarse"
    return _QcPaths(
        output_dir=output_dir,
        preview=output_dir / f"{prefix}_aligned_bin{preview_binning}.st",
        report=output_dir / f"{prefix}_alignment_qc.json",
        newstack_log=output_dir / f"{prefix}_newstack.log",
        residual_log=output_dir / f"{prefix}_residual_tiltxcorr.log",
    )


def _validate_stack(
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[int, int, int]:
    if tilt_stack.kind is not ArtifactKind.TILT_STACK:
        raise ValueError(f"expected tilt stack artifact, got {tilt_stack.kind}")
    if not tilt_stack.path.exists():
        raise FileNotFoundError(f"tilt stack not found: {tilt_stack.path}")
    if tilt_stack.shape is None or len(tilt_stack.shape) != 3:
        raise ValueError("tilt stack artifact must record shape (tilts, y, x)")
    shape = (
        int(tilt_stack.shape[0]),
        int(tilt_stack.shape[1]),
        int(tilt_stack.shape[2]),
    )
    if shape[0] != manifest.num_tilts:
        raise ValueError(
            f"tilt stack has {shape[0]} images, manifest has {manifest.num_tilts}"
        )
    return shape


def _load_alignment(
    alignment: Artifact,
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
) -> TiltAlignment:
    if alignment.kind is not ArtifactKind.ALIGNMENT:
        raise ValueError(f"expected alignment artifact, got {alignment.kind}")
    if not alignment.path.is_file():
        raise FileNotFoundError(f"alignment result not found: {alignment.path}")
    result = TiltAlignment.model_validate_json(alignment.path.read_text())
    if result.tilt_series_id != manifest.tilt_series_id:
        raise ValueError(
            f"alignment is for {result.tilt_series_id}, expected "
            f"{manifest.tilt_series_id}"
        )
    if result.input_stack_id != tilt_stack.id:
        raise ValueError(
            f"alignment input stack is {result.input_stack_id}, expected "
            f"{tilt_stack.id}"
        )
    if len(result.transforms) != manifest.num_tilts:
        raise ValueError(
            f"alignment has {len(result.transforms)} transforms for "
            f"{manifest.num_tilts} tilts"
        )
    return result


def _validate_preview_binning(
    stack_shape: tuple[int, int, int],
    preview_binning: int,
) -> None:
    if preview_binning > min(stack_shape[1:]):
        raise ValueError(
            f"preview binning {preview_binning} exceeds image shape "
            f"{stack_shape[1:]}"
        )


def _write_scaled_xf(
    path: Path,
    preview_order: Sequence[int],
    manifest: TiltSeriesManifest,
    transforms_by_z: dict[int, AlignmentTransform],
    *,
    binning: int,
) -> None:
    lines: list[str] = []
    for index in preview_order:
        z_value = manifest.images[index].z_value
        transform = transforms_by_z[z_value]
        lines.append(
            f"{transform.a11:.7f} {transform.a12:.7f} "
            f"{transform.a21:.7f} {transform.a22:.7f} "
            f"{transform.shift_x_px / binning:.3f} "
            f"{transform.shift_y_px / binning:.3f}\n"
        )
    path.write_text("".join(lines))


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


def _require_success(
    result: subprocess.CompletedProcess[str],
    *,
    program: str,
    log_path: Path,
) -> None:
    if result.returncode != 0:
        raise RuntimeError(
            f"{program} failed with exit code {result.returncode}; see {log_path}"
        )


def _write_report(path: Path, report: CoarseAlignmentQc) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def _max_abs_shift_fraction(
    alignment: TiltAlignment,
    stack_shape: tuple[int, int, int],
) -> float:
    excluded = set(alignment.excluded_z_values)
    included = [
        transform
        for transform in alignment.transforms
        if transform.z_value not in excluded
    ]
    max_x = max(abs(transform.shift_x_px) for transform in included) / stack_shape[2]
    max_y = max(abs(transform.shift_y_px) for transform in included) / stack_shape[1]
    return max(max_x, max_y)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"coarse-alignment QC outputs already exist: {joined}")


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


def _nonnegative_float_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0:
        raise ValueError(f"context parameter {key!r} must be finite and nonnegative")
    return normalized


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)
