from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    parse_imod_xf,
    resolve_imod_executable,
    run_command,
)
from cryoet_pipeline.backends.protocols import BackendContext, FineAlignmentBackend
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    FineAlignmentQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)

_RESIDUAL_RE = re.compile(
    r"Residual error mean and sd:\s+"
    r"(?P<mean>[-+.\deE]+)\s+(?P<sd>[-+.\deE]+)\s+nm"
)
_COUNTS_RE = re.compile(
    r"(?P<views>\d+)\s+views,.*?"
    r"(?P<fiducials>\d+)\s+3-D points,\s+"
    r"(?P<points>\d+)\s+projection points"
)
_ROTATION_RE = re.compile(
    r"At minimum tilt, rotation angle is\s+(?P<rotation>[-+.\deE]+)"
)
_LEAVE_OUT_RE = re.compile(
    r"Global\s+(?:robust\s+)?leave-out error\s+\(\d+\s+pts\):\s+"
    r"(?P<error>[-+.\deE]+)\s+nm"
)
_X_AXIS_TILT_RE = re.compile(
    r"X axis tilt needed\s*=\s*(?P<value>[-+.\deE]+)"
)
_UNBINNED_THICKNESS_RE = re.compile(
    r"Unbinned thickness needed to contain centers of all fiducials\s*=\s*"
    r"(?P<value>[-+.\deE]+)"
)
_UNBINNED_Z_SHIFT_RE = re.compile(
    r"Total unbinned shift needed to center range of fiducials in Z\s*=\s*"
    r"(?P<value>[-+.\deE]+)"
)
_INCREMENTAL_UNBINNED_Z_SHIFT_RE = re.compile(
    r"Incremental unbinned shift needed to center range of fiducials in Z\s*=\s*"
    r"(?P<value>[-+.\deE]+)"
)
_INCREMENTAL_TILT_ANGLE_RE = re.compile(
    r"Incremental tilt angle change\s*=\s*(?P<value>[-+.\deE]+)"
)
_TOTAL_TILT_ANGLE_RE = re.compile(
    r"Total tilt angle change\s*=\s*(?P<value>[-+.\deE]+)"
)


@dataclass(frozen=True)
class TiltalignLogSummary:
    """Selected stable metrics parsed from Tiltalign text output."""

    image_count: int
    fiducial_count: int
    projection_point_count: int
    residual_mean_nm: float
    residual_sd_nm: float
    minimum_tilt_rotation_deg: float
    global_leave_out_error_nm: float | None


@dataclass(frozen=True)
class ResidualVectorSummary:
    """Distribution of per-point residual magnitudes in tracking-stack pixels."""

    count: int
    rms_px: float
    p95_px: float
    max_px: float
    outlier_count: int


@dataclass(frozen=True)
class TomogramPositioningSummary:
    """Tiltalign surface-analysis recommendations in unbinned coordinates."""

    x_axis_tilt_deg: float
    unbinned_thickness_px: float
    unbinned_z_shift_px: float
    incremental_unbinned_z_shift_px: float
    total_tilt_angle_deg: float
    incremental_tilt_angle_deg: float


@dataclass(frozen=True)
class AppliedTomogramPositioning:
    """Positioning values applied in the final Tiltalign round."""

    tilt_angle_offset_deg: float
    axis_z_shift_unbinned_px: float


class ImodTiltalignBackend:
    """Solve fiducial geometry with IMOD Tiltalign and compose final transforms."""

    name = "imod_tiltalign"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def align(
        self,
        tracking_stack: Artifact,
        fiducial_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Write canonical fine alignment, IMOD transforms, and residual QC."""

        metadata = _validate_inputs(tracking_stack, fiducial_model, manifest)
        residual_warning_nm = _nonnegative_float_parameter(
            context,
            "residual_warning_nm",
            default=0.8,
        )
        residual_failure_nm = _nonnegative_float_parameter(
            context,
            "residual_failure_nm",
            default=1.5,
        )
        if residual_failure_nm < residual_warning_nm:
            raise ValueError(
                "residual_failure_nm must be greater than or equal to "
                "residual_warning_nm"
            )
        residual_max_warning_px = _nonnegative_float_parameter(
            context,
            "residual_max_warning_tracking_px",
            default=5.0,
        )
        residual_max_failure_px = _nonnegative_float_parameter(
            context,
            "residual_max_failure_tracking_px",
            default=20.0,
        )
        if residual_max_failure_px < residual_max_warning_px:
            raise ValueError(
                "residual_max_failure_tracking_px must be greater than or equal "
                "to residual_max_warning_tracking_px"
            )
        cross_validate = _bool_parameter(context, "cross_validate", default=True)
        robust_fitting = _bool_parameter(context, "robust_fitting", default=True)
        auto_prune_outliers = _bool_parameter(
            context,
            "auto_prune_outliers",
            default=True,
        )
        apply_surface_positioning = _bool_parameter(
            context,
            "apply_surface_positioning",
            default=True,
        )
        max_positioning_rounds = _positive_int_parameter(
            context,
            "max_positioning_rounds",
            default=3,
        )
        max_pruned_fraction = _fraction_parameter(
            context,
            "max_pruned_fraction",
            default=0.02,
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _fine_alignment_paths(context.output_dir, manifest)
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        tiltalign = resolve_imod_executable("tiltalign", context)
        xfproduct = resolve_imod_executable("xfproduct", context)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-tiltalign-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_model = temporary_root / "fiducials.3dmod"
            residual_file = temporary_root / "fine.resid"
            fiducial_xyz = temporary_root / "fiducials.xyz"
            solved_tilt_file = temporary_root / "fine.tlt"
            x_axis_tilt_file = temporary_root / "fine.xtilt"
            fine_transform = temporary_root / "fine.tltxf"
            filled_model = temporary_root / "filled.fid"
            final_transform = temporary_root / "fine.xf"
            alignment_input_model = temporary_root / "alignment_input.fid"
            cleaned_model = temporary_root / "alignment_input_cleaned.fid"
            shutil.copyfile(fiducial_model.path, alignment_input_model)

            command = [
                str(tiltalign),
                "-ModelFile",
                str(alignment_input_model),
                "-ImageFile",
                str(tracking_stack.path.resolve()),
                "-ImagesAreBinned",
                str(metadata.tracking_binning),
                "-UnbinnedPixelSize",
                f"{metadata.raw_pixel_spacing_angstrom / 10.0:.8f}",
                "-OutputModelFile",
                str(output_model),
                "-OutputResidualFile",
                str(residual_file),
                "-OutputFilledInModel",
                str(filled_model),
                "-OutputFidXYZFile",
                str(fiducial_xyz),
                "-OutputTiltFile",
                str(solved_tilt_file),
                "-OutputXAxisTiltFile",
                str(x_axis_tilt_file),
                "-OutputTransformFile",
                str(fine_transform),
                "-RotationAngle",
                f"{metadata.tilt_axis_angle_deg:.6f}",
                "-TiltFile",
                str(metadata.raw_tilt_file.resolve()),
                "-AngleOffset",
                "0.0",
                "-NoSeparateTiltGroups",
                "1",
                "-BeamTiltOption",
                "0",
                "-RotOption",
                "1",
                "-RotDefaultGrouping",
                "5",
                "-TiltOption",
                "5",
                "-TiltDefaultGrouping",
                "5",
                "-MagReferenceView",
                "1",
                "-MagOption",
                "1",
                "-MagDefaultGrouping",
                "4",
                "-XStretchOption",
                "0",
                "-SkewOption",
                "0",
                "-XTiltOption",
                "0",
                "-ResidualReportCriterion",
                "3.0",
                "-SurfacesToAnalyze",
                "2",
                "-MetroFactor",
                "0.25",
                "-MaximumCycles",
                "1000",
                "-AxisZShift",
                "0.0",
                "-ShiftZFromOriginal",
                "-KFactorScaling",
                "1.0",
                "-MinFidsTotalAndEachSurface",
                "10,4",
            ]
            if cross_validate:
                command.extend(["-CrossValidate", "1"])
            if robust_fitting:
                command.extend(["-RobustFitting", "-WarnOnRobustFailure"])
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(tiltalign, context),
            )
            _write_command_log(paths.initial_tiltalign_log, command, result)
            _require_success(
                result,
                program="IMOD tiltalign",
                log_path=paths.initial_tiltalign_log,
            )
            required_outputs = (
                output_model,
                residual_file,
                fiducial_xyz,
                solved_tilt_file,
                x_axis_tilt_file,
                fine_transform,
                filled_model,
            )
            missing = [path for path in required_outputs if not path.is_file()]
            if missing:
                raise RuntimeError(
                    "IMOD tiltalign did not write required outputs: "
                    + ", ".join(str(path) for path in missing)
                )
            log_summary = parse_tiltalign_log(result.stdout or "")
            residual_summary = parse_tiltalign_residual_file(
                residual_file,
                outlier_threshold_px=residual_max_warning_px,
            )
            alignment_rounds = 1
            pruned_point_count = 0
            if (
                auto_prune_outliers
                and residual_summary.max_px > residual_max_warning_px
            ):
                model2point = resolve_imod_executable("model2point", context)
                point2model = resolve_imod_executable("point2model", context)
                pruned_point_count = _prune_fiducial_model(
                    input_model=alignment_input_model,
                    output_model=cleaned_model,
                    image_file=tracking_stack.path,
                    residual_file=residual_file,
                    threshold_px=residual_max_warning_px,
                    max_pruned_fraction=max_pruned_fraction,
                    model2point=model2point,
                    point2model=point2model,
                    context=context,
                    command_runner=self._command_runner,
                    cwd=temporary_root,
                    log_path=paths.prune_log,
                )
                _replace_file(cleaned_model, alignment_input_model)
                for output in required_outputs:
                    output.unlink(missing_ok=True)
                result = self._command_runner(
                    command,
                    cwd=temporary_root,
                    env=imod_environment(tiltalign, context),
                )
                _write_command_log(paths.preposition_tiltalign_log, command, result)
                _require_success(
                    result,
                    program="IMOD tiltalign after outlier pruning",
                    log_path=paths.preposition_tiltalign_log,
                )
                missing = [path for path in required_outputs if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        "IMOD tiltalign did not write required outputs after "
                        "outlier pruning: "
                        + ", ".join(str(path) for path in missing)
                    )
                log_summary = parse_tiltalign_log(result.stdout or "")
                residual_summary = parse_tiltalign_residual_file(
                    residual_file,
                    outlier_threshold_px=residual_max_warning_px,
                )
                alignment_rounds = 2
            else:
                shutil.copyfile(
                    paths.initial_tiltalign_log,
                    paths.preposition_tiltalign_log,
                )
                paths.prune_log.write_text(
                    "No fiducial points pruned; initial residuals were within "
                    "the configured failure threshold.\n"
                )
            positioning_summary = parse_tiltalign_surface_analysis(
                result.stdout or ""
            )
            applied_positioning: AppliedTomogramPositioning | None = None
            positioning_round = 0
            while (
                apply_surface_positioning
                and positioning_summary is not None
                and positioning_round < max_positioning_rounds
                and (
                    positioning_round == 0
                    or not _positioning_is_converged(positioning_summary)
                )
            ):
                applied_positioning = AppliedTomogramPositioning(
                    tilt_angle_offset_deg=positioning_summary.total_tilt_angle_deg,
                    axis_z_shift_unbinned_px=(
                        positioning_summary.unbinned_z_shift_px
                    ),
                )
                positioned_command = _replace_command_option(
                    command,
                    "-AngleOffset",
                    f"{applied_positioning.tilt_angle_offset_deg:.6f}",
                )
                positioned_command = _replace_command_option(
                    positioned_command,
                    "-AxisZShift",
                    f"{applied_positioning.axis_z_shift_unbinned_px:.6f}",
                )
                for output in required_outputs:
                    output.unlink(missing_ok=True)
                result = self._command_runner(
                    positioned_command,
                    cwd=temporary_root,
                    env=imod_environment(tiltalign, context),
                )
                if positioning_round == 0:
                    _write_command_log(
                        paths.tiltalign_log,
                        positioned_command,
                        result,
                    )
                else:
                    _append_command_log(
                        paths.tiltalign_log,
                        positioned_command,
                        result,
                    )
                _require_success(
                    result,
                    program="IMOD tiltalign after tomogram positioning",
                    log_path=paths.tiltalign_log,
                )
                missing = [path for path in required_outputs if not path.is_file()]
                if missing:
                    raise RuntimeError(
                        "IMOD tiltalign did not write required outputs after "
                        "tomogram positioning: "
                        + ", ".join(str(path) for path in missing)
                    )
                log_summary = parse_tiltalign_log(result.stdout or "")
                residual_summary = parse_tiltalign_residual_file(
                    residual_file,
                    outlier_threshold_px=residual_max_warning_px,
                )
                positioning_summary = parse_tiltalign_surface_analysis(
                    result.stdout or ""
                )
                positioning_round += 1
                alignment_rounds += 1
            if positioning_round == 0:
                shutil.copyfile(
                    paths.preposition_tiltalign_log,
                    paths.tiltalign_log,
                )

            xfproduct_command = [
                str(xfproduct),
                "-in1",
                str(metadata.prealign_transform.resolve()),
                "-in2",
                str(fine_transform),
                "-output",
                str(final_transform),
            ]
            xfproduct_result = self._command_runner(
                xfproduct_command,
                cwd=temporary_root,
                env=imod_environment(xfproduct, context),
            )
            _write_command_log(
                paths.xfproduct_log,
                xfproduct_command,
                xfproduct_result,
            )
            _require_success(
                xfproduct_result,
                program="IMOD xfproduct",
                log_path=paths.xfproduct_log,
            )
            if not final_transform.is_file():
                raise RuntimeError(
                    f"IMOD xfproduct did not write final transforms: "
                    f"{final_transform}"
                )

            fine_transforms = parse_imod_xf(fine_transform)
            final_transforms = parse_imod_xf(final_transform)
            solved_tilt_angles = _read_float_lines(solved_tilt_file)
            expected = len(metadata.included_z_values)
            if len(fine_transforms) != expected:
                raise ValueError(
                    f"tiltalign returned {len(fine_transforms)} transforms "
                    f"for {expected} included tilts"
                )
            if len(final_transforms) != expected:
                raise ValueError(
                    f"xfproduct returned {len(final_transforms)} transforms "
                    f"for {expected} included tilts"
                )
            if len(solved_tilt_angles) != expected:
                raise ValueError(
                    f"tiltalign returned {len(solved_tilt_angles)} tilt angles "
                    f"for {expected} included tilts"
                )

            image_by_z = {image.z_value: image for image in manifest.images}
            canonical_transforms = [
                AlignmentTransform(
                    z_value=z_value,
                    tilt_angle_deg=solved_angle,
                    a11=values[0],
                    a12=values[1],
                    a21=values[2],
                    a22=values[3],
                    shift_x_px=values[4],
                    shift_y_px=values[5],
                )
                for z_value, solved_angle, values in zip(
                    metadata.included_z_values,
                    solved_tilt_angles,
                    final_transforms,
                    strict=True,
                )
                if z_value in image_by_z
            ]
            if len(canonical_transforms) != expected:
                raise ValueError("fine alignment refers to unknown manifest z values")
            alignment = TiltAlignment(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                stage="fine",
                input_stack_id=tracking_stack.id,
                input_binning=metadata.tracking_binning,
                tilt_axis_angle_deg=log_summary.minimum_tilt_rotation_deg,
                transform_semantics="global",
                transforms=canonical_transforms,
                excluded_z_values=metadata.excluded_z_values,
            )
            report = _fine_alignment_qc(
                manifest=manifest,
                summary=log_summary,
                raw_pixel_spacing_angstrom=metadata.raw_pixel_spacing_angstrom,
                residual_warning_nm=residual_warning_nm,
                residual_failure_nm=residual_failure_nm,
                residual_summary=residual_summary,
                residual_max_warning_px=residual_max_warning_px,
                residual_max_failure_px=residual_max_failure_px,
                pruned_point_count=pruned_point_count,
                alignment_rounds=alignment_rounds,
                positioning_summary=positioning_summary,
                applied_positioning=applied_positioning,
            )

            _replace_file(output_model, paths.output_model)
            _replace_file(residual_file, paths.residual_file)
            _replace_file(fiducial_xyz, paths.fiducial_xyz)
            _replace_file(solved_tilt_file, paths.solved_tilt_file)
            _replace_file(x_axis_tilt_file, paths.x_axis_tilt_file)
            _replace_file(fine_transform, paths.fine_transform)
            _replace_file(filled_model, paths.filled_model)
            _replace_file(final_transform, paths.final_transform)
            _replace_file(alignment_input_model, paths.alignment_input_model)

        _write_json(paths.canonical, alignment.model_dump(mode="json"))
        _write_json(paths.report, report.model_dump(mode="json"))
        alignment_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:alignment:fine",
            kind=ArtifactKind.ALIGNMENT,
            path=paths.canonical,
            parent_ids=[tracking_stack.id, fiducial_model.id],
            shape=(len(metadata.included_z_values), 6),
            dtype="float64",
            parameters={
                "backend": self.name,
                "stage": "fine",
                "tilt_series_id": manifest.tilt_series_id,
                "transform_semantics": "global",
                "included_z_values": metadata.included_z_values,
                "excluded_z_values": metadata.excluded_z_values,
                "raw_pixel_spacing_angstrom": metadata.raw_pixel_spacing_angstrom,
                "imod_xf_path": str(paths.final_transform),
                "imod_tltxf_path": str(paths.fine_transform),
                "solved_tilt_file_path": str(paths.solved_tilt_file),
                "x_axis_tilt_file_path": str(paths.x_axis_tilt_file),
                "residual_file_path": str(paths.residual_file),
                "filled_fiducial_model_path": str(paths.filled_model),
                "output_3d_model_path": str(paths.output_model),
                "fiducial_xyz_path": str(paths.fiducial_xyz),
                "tiltalign_log_path": str(paths.tiltalign_log),
                "preposition_tiltalign_log_path": str(
                    paths.preposition_tiltalign_log
                ),
                "xfproduct_log_path": str(paths.xfproduct_log),
                "initial_tiltalign_log_path": str(paths.initial_tiltalign_log),
                "alignment_input_model_path": str(paths.alignment_input_model),
                "outlier_prune_log_path": str(paths.prune_log),
                "pruned_point_count": report.pruned_point_count,
                "alignment_rounds": report.alignment_rounds,
                "translation_units": "full_resolution_pixels",
                **_positioning_parameters(
                    positioning_summary,
                    applied_positioning,
                ),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.canonical,
                paths.final_transform,
                paths.fine_transform,
                paths.solved_tilt_file,
                paths.x_axis_tilt_file,
                paths.residual_file,
                paths.filled_model,
                paths.output_model,
                paths.fiducial_xyz,
                paths.tiltalign_log,
                paths.preposition_tiltalign_log,
                paths.xfproduct_log,
                paths.initial_tiltalign_log,
                paths.alignment_input_model,
                paths.prune_log,
            ),
        )
        report_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:alignment:fine",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[alignment_artifact.id],
            parameters={
                "qc_type": "fine_alignment",
                "tilt_series_id": manifest.tilt_series_id,
                "status": report.status.value,
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [alignment_artifact, report_artifact]


def fine_align_and_register(
    backend: FineAlignmentBackend,
    tracking_stack: Artifact,
    fiducial_model: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Run fine alignment and register canonical alignment then QC."""

    artifacts = backend.align(
        tracking_stack,
        fiducial_model,
        manifest,
        context,
    )
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def parse_tiltalign_log(text: str) -> TiltalignLogSummary:
    """Parse stable summary lines from Tiltalign stdout."""

    residual_match = _RESIDUAL_RE.search(text)
    counts_match = _COUNTS_RE.search(text)
    rotation_match = _ROTATION_RE.search(text)
    if residual_match is None:
        raise ValueError("Tiltalign log has no residual mean and standard deviation")
    if counts_match is None:
        raise ValueError("Tiltalign log has no view/fiducial/point counts")
    if rotation_match is None:
        raise ValueError("Tiltalign log has no minimum-tilt rotation angle")
    leave_out_match = _LEAVE_OUT_RE.search(text)
    values = TiltalignLogSummary(
        image_count=int(counts_match.group("views")),
        fiducial_count=int(counts_match.group("fiducials")),
        projection_point_count=int(counts_match.group("points")),
        residual_mean_nm=float(residual_match.group("mean")),
        residual_sd_nm=float(residual_match.group("sd")),
        minimum_tilt_rotation_deg=float(rotation_match.group("rotation")),
        global_leave_out_error_nm=(
            float(leave_out_match.group("error"))
            if leave_out_match is not None
            else None
        ),
    )
    numeric = (
        values.residual_mean_nm,
        values.residual_sd_nm,
        values.minimum_tilt_rotation_deg,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("Tiltalign log summary values must be finite")
    return values


def parse_tiltalign_surface_analysis(
    text: str,
) -> TomogramPositioningSummary | None:
    """Parse optional specimen-positioning recommendations from Tiltalign."""

    x_axis_matches = list(_X_AXIS_TILT_RE.finditer(text))
    thickness_match = _UNBINNED_THICKNESS_RE.search(text)
    z_shift_match = _UNBINNED_Z_SHIFT_RE.search(text)
    incremental_z_shift_match = _INCREMENTAL_UNBINNED_Z_SHIFT_RE.search(text)
    total_tilt_matches = list(_TOTAL_TILT_ANGLE_RE.finditer(text))
    incremental_tilt_matches = list(_INCREMENTAL_TILT_ANGLE_RE.finditer(text))
    matches = (
        bool(x_axis_matches),
        thickness_match is not None,
        z_shift_match is not None,
        incremental_z_shift_match is not None,
        bool(total_tilt_matches),
        bool(incremental_tilt_matches),
    )
    if not any(matches):
        return None
    if not all(matches):
        raise ValueError("Tiltalign surface analysis is incomplete")
    assert thickness_match is not None
    assert z_shift_match is not None
    assert incremental_z_shift_match is not None

    values = TomogramPositioningSummary(
        x_axis_tilt_deg=float(x_axis_matches[-1].group("value")),
        unbinned_thickness_px=float(thickness_match.group("value")),
        unbinned_z_shift_px=float(z_shift_match.group("value")),
        incremental_unbinned_z_shift_px=float(
            incremental_z_shift_match.group("value")
        ),
        total_tilt_angle_deg=float(total_tilt_matches[-1].group("value")),
        incremental_tilt_angle_deg=float(
            incremental_tilt_matches[-1].group("value")
        ),
    )
    numeric = (
        values.x_axis_tilt_deg,
        values.unbinned_thickness_px,
        values.unbinned_z_shift_px,
        values.incremental_unbinned_z_shift_px,
        values.total_tilt_angle_deg,
        values.incremental_tilt_angle_deg,
    )
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("Tiltalign surface analysis values must be finite")
    if values.unbinned_thickness_px <= 0.0:
        raise ValueError("Tiltalign recommended thickness must be positive")
    return values


def parse_tiltalign_residual_file(
    path: Path,
    *,
    outlier_threshold_px: float,
) -> ResidualVectorSummary:
    """Parse Tiltalign residual vectors and summarize their magnitudes."""

    magnitudes = _read_residual_magnitudes(path)
    values = np.asarray(magnitudes, dtype=np.float64)
    return ResidualVectorSummary(
        count=len(magnitudes),
        rms_px=float(np.sqrt(np.mean(np.square(values)))),
        p95_px=float(np.percentile(values, 95)),
        max_px=float(np.max(values)),
        outlier_count=int(np.count_nonzero(values > outlier_threshold_px)),
    )


def _replace_command_option(
    command: Sequence[str],
    option: str,
    value: str,
) -> list[str]:
    updated = list(command)
    try:
        option_index = updated.index(option)
    except ValueError as exc:
        raise ValueError(f"command does not contain option {option}") from exc
    value_index = option_index + 1
    if value_index >= len(updated):
        raise ValueError(f"command option {option} has no value")
    updated[value_index] = value
    return updated


def _positioning_is_converged(summary: TomogramPositioningSummary) -> bool:
    return (
        abs(summary.incremental_tilt_angle_deg) <= 0.1
        and abs(summary.incremental_unbinned_z_shift_px) <= 1.0
    )


def _read_residual_magnitudes(path: Path) -> list[float]:
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path}: residual file is empty")
    header_fields = lines[0].split()
    if len(header_fields) < 2 or header_fields[1] != "residuals":
        raise ValueError(f"{path}: malformed residual count header")
    try:
        declared_count = int(header_fields[0])
    except ValueError as exc:
        raise ValueError(f"{path}: invalid residual count") from exc

    magnitudes: list[float] = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(
                f"{path}:{line_number}: expected at least five residual values"
            )
        try:
            residual_x = float(fields[-2])
            residual_y = float(fields[-1])
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid residual vector"
            ) from exc
        magnitude = float(np.hypot(residual_x, residual_y))
        if not np.isfinite(magnitude):
            raise ValueError(f"{path}:{line_number}: residual must be finite")
        magnitudes.append(magnitude)
    if len(magnitudes) != declared_count:
        raise ValueError(
            f"{path}: declares {declared_count} residuals, found {len(magnitudes)}"
        )
    if not magnitudes:
        raise ValueError(f"{path}: no residual vectors found")
    return magnitudes


@dataclass(frozen=True)
class _FineAlignmentMetadata:
    tracking_binning: int
    raw_pixel_spacing_angstrom: float
    tilt_axis_angle_deg: float
    prealign_transform: Path
    raw_tilt_file: Path
    included_z_values: list[int]
    excluded_z_values: list[int]


@dataclass(frozen=True)
class _FineAlignmentPaths:
    output_dir: Path
    canonical: Path
    report: Path
    final_transform: Path
    fine_transform: Path
    solved_tilt_file: Path
    x_axis_tilt_file: Path
    residual_file: Path
    filled_model: Path
    output_model: Path
    fiducial_xyz: Path
    tiltalign_log: Path
    preposition_tiltalign_log: Path
    initial_tiltalign_log: Path
    xfproduct_log: Path
    alignment_input_model: Path
    prune_log: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.canonical,
            self.report,
            self.final_transform,
            self.fine_transform,
            self.solved_tilt_file,
            self.x_axis_tilt_file,
            self.residual_file,
            self.filled_model,
            self.output_model,
            self.fiducial_xyz,
            self.tiltalign_log,
            self.preposition_tiltalign_log,
            self.initial_tiltalign_log,
            self.xfproduct_log,
            self.alignment_input_model,
            self.prune_log,
        )


def _fine_alignment_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
) -> _FineAlignmentPaths:
    output_dir = output_root / "alignments" / manifest.tilt_series_id / "fine"
    prefix = f"{manifest.tilt_series_id}_fine"
    return _FineAlignmentPaths(
        output_dir=output_dir,
        canonical=output_dir / f"{prefix}_alignment.json",
        report=output_dir / f"{prefix}_alignment_qc.json",
        final_transform=output_dir / f"{prefix}.xf",
        fine_transform=output_dir / f"{prefix}.tltxf",
        solved_tilt_file=output_dir / f"{prefix}.tlt",
        x_axis_tilt_file=output_dir / f"{prefix}.xtilt",
        residual_file=output_dir / f"{prefix}.resid",
        filled_model=output_dir / f"{prefix}_filled.fid",
        output_model=output_dir / f"{prefix}.3dmod",
        fiducial_xyz=output_dir / f"{prefix}_fid.xyz",
        tiltalign_log=output_dir / f"{prefix}_tiltalign.log",
        preposition_tiltalign_log=(
            output_dir / f"{prefix}_tiltalign_preposition.log"
        ),
        initial_tiltalign_log=output_dir / f"{prefix}_tiltalign_initial.log",
        xfproduct_log=output_dir / f"{prefix}_xfproduct.log",
        alignment_input_model=output_dir / f"{prefix}_alignment_input.fid",
        prune_log=output_dir / f"{prefix}_outlier_prune.log",
    )


def _validate_inputs(
    tracking_stack: Artifact,
    fiducial_model: Artifact,
    manifest: TiltSeriesManifest,
) -> _FineAlignmentMetadata:
    if tracking_stack.kind is not ArtifactKind.ALIGNED_TILT_STACK:
        raise ValueError(
            f"expected aligned tracking stack, got {tracking_stack.kind}"
        )
    if tracking_stack.parameters.get("purpose") != "fiducial_tracking":
        raise ValueError("aligned stack is not a fiducial-tracking stack")
    if not tracking_stack.path.is_file():
        raise FileNotFoundError(f"tracking stack not found: {tracking_stack.path}")
    if fiducial_model.kind is not ArtifactKind.FIDUCIAL_MODEL:
        raise ValueError(f"expected fiducial model, got {fiducial_model.kind}")
    if not fiducial_model.path.is_file():
        raise FileNotFoundError(f"fiducial model not found: {fiducial_model.path}")
    if tracking_stack.id not in fiducial_model.parent_ids:
        raise ValueError("fiducial model does not belong to tracking stack")
    if fiducial_model.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError("fiducial model is for a different tilt series")

    included = _int_list_parameter(tracking_stack, "included_z_values")
    excluded = _int_list_parameter(tracking_stack, "excluded_z_values")
    if set(included) & set(excluded):
        raise ValueError("included and excluded z values must be disjoint")
    if tracking_stack.shape is None or tracking_stack.shape[0] != len(included):
        raise ValueError("tracking stack shape does not match included z values")
    if tracking_stack.binning is None or tracking_stack.binning < 1:
        raise ValueError("tracking stack must record positive binning")
    raw_pixel_spacing = _positive_numeric_parameter(
        tracking_stack,
        "raw_pixel_spacing_angstrom",
    )
    tilt_axis_angle = _finite_numeric_parameter(
        tracking_stack,
        "tilt_axis_angle_deg",
    )
    return _FineAlignmentMetadata(
        tracking_binning=tracking_stack.binning,
        raw_pixel_spacing_angstrom=raw_pixel_spacing,
        tilt_axis_angle_deg=tilt_axis_angle,
        prealign_transform=_path_parameter(
            tracking_stack,
            "prealign_transform_path",
        ),
        raw_tilt_file=_path_parameter(tracking_stack, "tilt_file_path"),
        included_z_values=included,
        excluded_z_values=excluded,
    )


def _fine_alignment_qc(
    *,
    manifest: TiltSeriesManifest,
    summary: TiltalignLogSummary,
    raw_pixel_spacing_angstrom: float,
    residual_warning_nm: float,
    residual_failure_nm: float,
    residual_summary: ResidualVectorSummary,
    residual_max_warning_px: float,
    residual_max_failure_px: float,
    pruned_point_count: int,
    alignment_rounds: int,
    positioning_summary: TomogramPositioningSummary | None,
    applied_positioning: AppliedTomogramPositioning | None,
) -> FineAlignmentQc:
    warnings: list[str] = []
    if summary.residual_mean_nm > residual_failure_nm:
        status = QcStatus.FAIL
        warnings.append(
            f"mean residual {summary.residual_mean_nm:.3f} nm exceeds "
            f"failure threshold {residual_failure_nm:.3f} nm"
        )
    elif summary.residual_mean_nm > residual_warning_nm:
        status = QcStatus.WARNING
        warnings.append(
            f"mean residual {summary.residual_mean_nm:.3f} nm exceeds "
            f"warning threshold {residual_warning_nm:.3f} nm"
        )
    else:
        status = QcStatus.PASS
    if residual_summary.max_px > residual_max_failure_px:
        status = QcStatus.FAIL
        warnings.append(
            f"maximum point residual {residual_summary.max_px:.3f} tracking px "
            f"exceeds failure threshold {residual_max_failure_px:.3f}"
        )
    elif residual_summary.max_px > residual_max_warning_px:
        if status is QcStatus.PASS:
            status = QcStatus.WARNING
        warnings.append(
            f"maximum point residual {residual_summary.max_px:.3f} tracking px "
            f"exceeds warning threshold {residual_max_warning_px:.3f}"
        )
    if (
        applied_positioning is not None
        and positioning_summary is not None
        and abs(positioning_summary.incremental_tilt_angle_deg) > 0.1
    ):
        if status is QcStatus.PASS:
            status = QcStatus.WARNING
        warnings.append(
            "post-positioning tilt-angle correction "
            f"{positioning_summary.incremental_tilt_angle_deg:.3f} deg "
            "exceeds 0.100 deg"
        )
    if (
        applied_positioning is not None
        and positioning_summary is not None
        and abs(positioning_summary.incremental_unbinned_z_shift_px) > 1.0
    ):
        if status is QcStatus.PASS:
            status = QcStatus.WARNING
        warnings.append(
            "post-positioning Z correction "
            f"{positioning_summary.incremental_unbinned_z_shift_px:.3f} "
            "unbinned px exceeds 1.000 px"
        )
    return FineAlignmentQc(
        tilt_series_id=manifest.tilt_series_id,
        backend="imod_tiltalign",
        alignment_id=f"{manifest.tilt_series_id}:alignment:fine",
        image_count=summary.image_count,
        fiducial_count=summary.fiducial_count,
        projection_point_count=summary.projection_point_count,
        residual_mean_nm=summary.residual_mean_nm,
        residual_sd_nm=summary.residual_sd_nm,
        residual_mean_unbinned_px=(
            summary.residual_mean_nm * 10.0 / raw_pixel_spacing_angstrom
        ),
        residual_rms_tracking_px=residual_summary.rms_px,
        residual_p95_tracking_px=residual_summary.p95_px,
        residual_max_tracking_px=residual_summary.max_px,
        residual_outlier_count=residual_summary.outlier_count,
        pruned_point_count=pruned_point_count,
        alignment_rounds=alignment_rounds,
        global_leave_out_error_nm=summary.global_leave_out_error_nm,
        minimum_tilt_rotation_deg=summary.minimum_tilt_rotation_deg,
        recommended_x_axis_tilt_deg=(
            positioning_summary.x_axis_tilt_deg
            if positioning_summary is not None
            else None
        ),
        recommended_unbinned_thickness_px=(
            positioning_summary.unbinned_thickness_px
            if positioning_summary is not None
            else None
        ),
        recommended_unbinned_z_shift_px=(
            positioning_summary.unbinned_z_shift_px
            if positioning_summary is not None
            else None
        ),
        applied_tilt_angle_offset_deg=(
            applied_positioning.tilt_angle_offset_deg
            if applied_positioning is not None
            else None
        ),
        applied_axis_z_shift_unbinned_px=(
            applied_positioning.axis_z_shift_unbinned_px
            if applied_positioning is not None
            else None
        ),
        positioning_incremental_tilt_angle_deg=(
            positioning_summary.incremental_tilt_angle_deg
            if positioning_summary is not None
            else None
        ),
        positioning_incremental_z_shift_unbinned_px=(
            positioning_summary.incremental_unbinned_z_shift_px
            if positioning_summary is not None
            else None
        ),
        status=status,
        warnings=warnings,
    )


def _positioning_parameters(
    summary: TomogramPositioningSummary | None,
    applied: AppliedTomogramPositioning | None,
) -> dict[str, object]:
    if summary is None:
        return {}
    parameters: dict[str, object] = {
        "positioning_source": "tiltalign_two_surface_analysis",
        "recommended_x_axis_tilt_deg": summary.x_axis_tilt_deg,
        "recommended_unbinned_thickness_px": summary.unbinned_thickness_px,
        "recommended_unbinned_z_shift_px": summary.unbinned_z_shift_px,
        "positioning_incremental_tilt_angle_deg": (
            summary.incremental_tilt_angle_deg
        ),
        "positioning_incremental_z_shift_unbinned_px": (
            summary.incremental_unbinned_z_shift_px
        ),
    }
    if applied is not None:
        parameters.update(
            {
                "applied_tilt_angle_offset_deg": (
                    applied.tilt_angle_offset_deg
                ),
                "applied_axis_z_shift_unbinned_px": (
                    applied.axis_z_shift_unbinned_px
                ),
                "axis_z_shift_applied_in_alignment": True,
            }
        )
    return parameters


def _read_float_lines(path: Path) -> list[float]:
    values: list[float] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: invalid floating-point value"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"{path}:{line_number}: value must be finite")
        values.append(value)
    if not values:
        raise ValueError(f"{path}: no values found")
    return values


def _prune_fiducial_model(
    *,
    input_model: Path,
    output_model: Path,
    image_file: Path,
    residual_file: Path,
    threshold_px: float,
    max_pruned_fraction: float,
    model2point: Path,
    point2model: Path,
    context: BackendContext,
    command_runner: CommandRunner,
    cwd: Path,
    log_path: Path,
) -> int:
    points_path = cwd / "fiducial_points.txt"
    cleaned_points_path = cwd / "fiducial_points_cleaned.txt"
    export_command = [
        str(model2point),
        "-object",
        "-zero",
        "-zcoord",
        str(input_model),
        str(points_path),
    ]
    export_result = command_runner(
        export_command,
        cwd=cwd,
        env=imod_environment(model2point, context),
    )
    if export_result.returncode != 0:
        _write_command_log(log_path, export_command, export_result)
        raise RuntimeError(
            f"IMOD model2point failed with exit code {export_result.returncode}; "
            f"see {log_path}"
        )
    if not points_path.is_file():
        raise RuntimeError(f"IMOD model2point did not write points: {points_path}")

    point_lines = [
        line for line in points_path.read_text().splitlines() if line.strip()
    ]
    magnitudes = _read_residual_magnitudes(residual_file)
    if len(point_lines) != len(magnitudes):
        raise ValueError(
            f"fiducial model has {len(point_lines)} points but residual file has "
            f"{len(magnitudes)} vectors"
        )
    keep_mask = [magnitude <= threshold_px for magnitude in magnitudes]
    pruned_count = len(keep_mask) - sum(keep_mask)
    if pruned_count < 1:
        raise ValueError("outlier pruning was requested but found no points to prune")
    if pruned_count / len(keep_mask) > max_pruned_fraction:
        raise ValueError(
            f"refusing to prune {pruned_count}/{len(keep_mask)} fiducial points; "
            f"configured maximum fraction is {max_pruned_fraction:.4f}"
        )

    original_by_contour: dict[tuple[int, int], int] = {}
    remaining_by_contour: dict[tuple[int, int], int] = {}
    kept_lines: list[str] = []
    for line, keep in zip(point_lines, keep_mask, strict=True):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError("model2point output must include object and contour")
        key = (int(fields[0]), int(fields[1]))
        original_by_contour[key] = original_by_contour.get(key, 0) + 1
        if keep:
            kept_lines.append(line)
            remaining_by_contour[key] = remaining_by_contour.get(key, 0) + 1
    invalid_contours = [
        key
        for key, original_count in original_by_contour.items()
        if original_count >= 3 and remaining_by_contour.get(key, 0) < 3
    ]
    if invalid_contours:
        raise ValueError(
            "outlier pruning would leave fewer than three points in contours: "
            f"{invalid_contours}"
        )
    cleaned_points_path.write_text("\n".join(kept_lines) + "\n")

    import_command = [
        str(point2model),
        "-open",
        "-zero",
        "-zcoord",
        "-image",
        str(image_file.resolve()),
        str(cleaned_points_path),
        str(output_model),
    ]
    import_result = command_runner(
        import_command,
        cwd=cwd,
        env=imod_environment(point2model, context),
    )
    _write_prune_log(
        log_path,
        export_command,
        export_result,
        import_command,
        import_result,
        pruned_count=pruned_count,
        total_count=len(point_lines),
        threshold_px=threshold_px,
    )
    if import_result.returncode != 0:
        raise RuntimeError(
            f"IMOD point2model failed with exit code {import_result.returncode}; "
            f"see {log_path}"
        )
    if not output_model.is_file():
        raise RuntimeError(
            f"IMOD point2model did not write cleaned model: {output_model}"
        )
    return pruned_count


def _write_prune_log(
    path: Path,
    export_command: Sequence[str],
    export_result: subprocess.CompletedProcess[str],
    import_command: Sequence[str],
    import_result: subprocess.CompletedProcess[str],
    *,
    pruned_count: int,
    total_count: int,
    threshold_px: float,
) -> None:
    path.write_text(
        f"Pruned {pruned_count}/{total_count} points above "
        f"{threshold_px:.6f} tracking pixels.\n\n"
        f"$ {shlex.join(export_command)}\n"
        f"[stdout]\n{export_result.stdout or ''}\n"
        f"[stderr]\n{export_result.stderr or ''}\n"
        f"[exit_code]\n{export_result.returncode}\n\n"
        f"$ {shlex.join(import_command)}\n"
        f"[stdout]\n{import_result.stdout or ''}\n"
        f"[stderr]\n{import_result.stderr or ''}\n"
        f"[exit_code]\n{import_result.returncode}\n"
    )


def _int_list_parameter(artifact: Artifact, key: str) -> list[int]:
    value = artifact.parameters.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ValueError(f"artifact parameter {key!r} must be an integer list")
    return list(value)


def _path_parameter(artifact: Artifact, key: str) -> Path:
    value = artifact.parameters.get(key)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"artifact parameter {key!r} must be a path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"artifact sidecar not found: {path}")
    return path


def _finite_numeric_parameter(artifact: Artifact, key: str) -> float:
    value = artifact.parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"artifact parameter {key!r} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"artifact parameter {key!r} must be finite")
    return normalized


def _positive_numeric_parameter(artifact: Artifact, key: str) -> float:
    value = _finite_numeric_parameter(artifact, key)
    if value <= 0.0:
        raise ValueError(f"artifact parameter {key!r} must be positive")
    return value


def _write_command_log(
    path: Path,
    command: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    path.write_text(_command_log_text(command, result))


def _append_command_log(
    path: Path,
    command: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    with path.open("a") as log:
        log.write("\n--- positioning iteration ---\n\n")
        log.write(_command_log_text(command, result))


def _command_log_text(
    command: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> str:
    return (
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


def _replace_file(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    source.replace(destination)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"fine-alignment outputs already exist: {joined}")


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
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and nonnegative")
    return normalized


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


def _fraction_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
) -> float:
    value = _nonnegative_float_parameter(context, key, default=default)
    if value > 1.0:
        raise ValueError(f"context parameter {key!r} must not exceed 1")
    return value


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)
