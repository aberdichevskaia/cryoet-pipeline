from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, isclose
from pathlib import Path
from statistics import median

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    resolve_imod_executable,
    run_command,
    write_binned_mrc_stack,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    FiducialSeedBackend,
    FiducialTrackingBackend,
)
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    FiducialModelQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc


@dataclass(frozen=True)
class ImodModelSummary:
    """Small backend-independent summary parsed from `imodinfo -a`."""

    max_xyz: tuple[int, int, int]
    points_per_fiducial: tuple[int, ...]

    @property
    def num_fiducials(self) -> int:
        return len(self.points_per_fiducial)

    @property
    def num_points(self) -> int:
        return sum(self.points_per_fiducial)

    def coverage_fraction(self, image_count: int) -> float:
        if not self.points_per_fiducial:
            return 0.0
        return self.num_points / (self.num_fiducials * image_count)


class ImodAutofidseedBackend:
    """Generate a fiducial tracking stack and seed model with IMOD."""

    name = "imod_autofidseed"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def generate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Create an independently reproducible IMOD seed model."""

        stack_shape, alignment_result = _validate_seed_inputs(
            tilt_stack,
            alignment,
            manifest,
        )
        tracking_binning = _positive_int_parameter(
            context,
            "tracking_binning",
            default=4,
        )
        if tracking_binning > min(stack_shape[1:]):
            raise ValueError(
                f"tracking binning {tracking_binning} exceeds image shape "
                f"{stack_shape[1:]}"
            )
        fiducial_diameter_nm = _positive_float_parameter(
            context,
            "fiducial_diameter_nm",
            default=10.0,
        )
        target_beads = _positive_int_parameter(
            context,
            "target_beads",
            default=150,
        )
        min_seed_fiducials = _positive_int_parameter(
            context,
            "min_seed_fiducials",
            default=10,
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        raw_pixel_spacing = _raw_pixel_spacing(manifest, tilt_stack, context)
        pixel_size_nm = raw_pixel_spacing / 10.0
        configured_bead_diameter_px = context.parameters.get(
            "fiducial_diameter_unbinned_px"
        )
        if configured_bead_diameter_px is None:
            bead_diameter_px = fiducial_diameter_nm * 10.0 / raw_pixel_spacing
            bead_diameter_source = "physical_diameter_and_pixel_size"
        else:
            bead_diameter_px = _positive_float_parameter(
                context,
                "fiducial_diameter_unbinned_px",
                default=1.0,
            )
            bead_diameter_source = "user_override"
        beadtrack_box_size_px = _even_box_size(bead_diameter_px)

        included_indices = _included_indices(alignment_result, manifest)
        paths = _seed_paths(
            context.output_dir,
            manifest,
            tracking_binning=tracking_binning,
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        newstack = resolve_imod_executable("newstack", context)
        autofidseed = resolve_imod_executable("autofidseed", context)
        imodinfo = resolve_imod_executable("imodinfo", context)

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-fiducial-seed-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            input_stack = temporary_root / "input_bin.mrc"
            scaled_xf = temporary_root / "scaled.prexg"
            aligned_stack = temporary_root / "tracking_preali.mrc"
            full_resolution_xf = temporary_root / "tracking.prexg"
            tilt_file = temporary_root / "tracking.rawtlt"
            seed_model = temporary_root / "tracking.seed"
            future_fiducial_model = temporary_root / "tracking.fid"
            track_command = temporary_root / "track.com"
            info_file = temporary_root / "autofidseed.info"
            autofidseed_temp = temporary_root / "autofidseed.dir"

            write_binned_mrc_stack(
                tilt_stack.path,
                input_stack,
                binning=tracking_binning,
                pixel_spacing_angstrom=raw_pixel_spacing,
                section_indices=included_indices,
            )
            transforms_by_z = {
                transform.z_value: transform
                for transform in alignment_result.transforms
            }
            _write_imod_transforms(
                scaled_xf,
                included_indices,
                manifest,
                transforms_by_z,
                shift_scale=1.0 / tracking_binning,
            )
            _write_imod_transforms(
                full_resolution_xf,
                included_indices,
                manifest,
                transforms_by_z,
                shift_scale=1.0,
            )
            write_tilt_angles_file(
                tilt_file,
                [
                    manifest.images[index].tilt_angle_deg
                    for index in included_indices
                ],
            )

            newstack_command = [
                str(newstack),
                "-input",
                str(input_stack),
                "-output",
                str(aligned_stack),
                "-xform",
                str(scaled_xf),
                "-mode",
                "2",
            ]
            newstack_result = self._command_runner(
                newstack_command,
                cwd=temporary_root,
                env=imod_environment(newstack, context),
            )
            _write_command_log(paths.newstack_log, newstack_command, newstack_result)
            _require_success(
                newstack_result,
                program="IMOD newstack",
                log_path=paths.newstack_log,
            )
            if not aligned_stack.is_file():
                raise RuntimeError(
                    f"IMOD newstack did not write fiducial tracking stack: "
                    f"{aligned_stack}"
                )

            track_command.write_text(
                _track_command_text(
                    image_file=aligned_stack,
                    seed_model=seed_model,
                    output_model=future_fiducial_model,
                    prealign_transform=full_resolution_xf,
                    tilt_file=tilt_file,
                    tracking_binning=tracking_binning,
                    tilt_axis_angle_deg=alignment_result.tilt_axis_angle_deg,
                    pixel_size_nm=pixel_size_nm,
                    bead_diameter_px=bead_diameter_px,
                    box_size_px=beadtrack_box_size_px,
                )
            )
            autofidseed_command = [
                str(autofidseed),
                "-track",
                str(track_command),
                "-number",
                str(target_beads),
                "-spacing",
                "0.85",
                "-peak",
                "1.0",
                "-elongated",
                "1",
                "-output",
                str(seed_model),
                "-info",
                str(info_file),
                "-tempdir",
                str(autofidseed_temp),
            ]
            autofidseed_result = self._command_runner(
                autofidseed_command,
                cwd=temporary_root,
                env=imod_environment(autofidseed, context),
            )
            _write_command_log(
                paths.autofidseed_log,
                autofidseed_command,
                autofidseed_result,
            )
            _require_success(
                autofidseed_result,
                program="IMOD autofidseed",
                log_path=paths.autofidseed_log,
            )
            if not seed_model.is_file():
                raise RuntimeError(
                    f"IMOD autofidseed did not write a seed model: {seed_model}"
                )
            if not info_file.is_file():
                raise RuntimeError(
                    f"IMOD autofidseed did not write its info file: {info_file}"
                )

            summary = _inspect_imod_model(
                seed_model,
                imodinfo=imodinfo,
                context=context,
                command_runner=self._command_runner,
                log_path=paths.imodinfo_log,
                cwd=temporary_root,
            )
            report = _fiducial_qc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                stage="seed",
                model_id=f"{manifest.tilt_series_id}:fiducial_seed",
                summary=summary,
                image_count=len(included_indices),
                minimum_fiducials=min_seed_fiducials,
            )

            _replace_file(aligned_stack, paths.tracking_stack)
            _replace_file(full_resolution_xf, paths.prealign_transform)
            _replace_file(tilt_file, paths.tilt_file)
            _replace_file(seed_model, paths.seed_model)
            _replace_file(info_file, paths.autofidseed_info)

        paths.track_command.write_text(
            _track_command_text(
                image_file=paths.tracking_stack,
                seed_model=paths.seed_model,
                output_model=paths.output_dir / f"{manifest.tilt_series_id}.fid",
                prealign_transform=paths.prealign_transform,
                tilt_file=paths.tilt_file,
                tracking_binning=tracking_binning,
                tilt_axis_angle_deg=alignment_result.tilt_axis_angle_deg,
                pixel_size_nm=pixel_size_nm,
                bead_diameter_px=bead_diameter_px,
                box_size_px=beadtrack_box_size_px,
            )
        )
        _write_json(paths.report, report.model_dump(mode="json"))
        tracking_info = validate_complete_mrc(paths.tracking_stack)

        tracking_stack_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:aligned_tilt_stack:fiducial",
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=paths.tracking_stack,
            parent_ids=[tilt_stack.id, alignment.id],
            shape=tracking_info.shape,
            dtype=tracking_info.dtype,
            axis_order=AxisOrder.TYX,
            pixel_spacing_angstrom=raw_pixel_spacing * tracking_binning,
            binning=tracking_binning,
            parameters={
                "purpose": "fiducial_tracking",
                "tilt_series_id": manifest.tilt_series_id,
                "included_z_values": [
                    manifest.images[index].z_value for index in included_indices
                ],
                "excluded_z_values": alignment_result.excluded_z_values,
                "order": "tilt_angle_ascending",
                "prealign_transform_path": str(paths.prealign_transform),
                "tilt_file_path": str(paths.tilt_file),
                "newstack_log_path": str(paths.newstack_log),
                "tilt_axis_angle_deg": alignment_result.tilt_axis_angle_deg,
                "raw_pixel_spacing_angstrom": raw_pixel_spacing,
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CACHE,
            retention_policy=RetentionPolicy.RECOMPUTE,
            can_recompute=True,
            size_bytes=paths.tracking_stack.stat().st_size,
        )
        seed_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:fiducial_seed",
            kind=ArtifactKind.FIDUCIAL_SEED_MODEL,
            path=paths.seed_model,
            parent_ids=[tracking_stack_artifact.id],
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "target_beads": target_beads,
                "num_fiducials": summary.num_fiducials,
                "num_points": summary.num_points,
                "fiducial_diameter_nm": fiducial_diameter_nm,
                "fiducial_diameter_unbinned_px": bead_diameter_px,
                "fiducial_diameter_px_source": bead_diameter_source,
                "beadtrack_box_size_unbinned_px": beadtrack_box_size_px,
                "tracking_binning": tracking_binning,
                "track_command_path": str(paths.track_command),
                "autofidseed_info_path": str(paths.autofidseed_info),
                "autofidseed_log_path": str(paths.autofidseed_log),
                "imodinfo_log_path": str(paths.imodinfo_log),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.seed_model,
                paths.track_command,
                paths.autofidseed_info,
                paths.autofidseed_log,
                paths.imodinfo_log,
            ),
        )
        report_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:fiducial_seed",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[seed_artifact.id],
            parameters={
                "qc_type": "fiducial_seed",
                "tilt_series_id": manifest.tilt_series_id,
                "status": report.status.value,
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [tracking_stack_artifact, seed_artifact, report_artifact]


class ImodBeadtrackBackend:
    """Track a seed model through an aligned tilt series with IMOD Beadtrack."""

    name = "imod_beadtrack"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def track(
        self,
        tracking_stack: Artifact,
        seed_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Create a full fiducial model plus machine-readable coverage QC."""

        metadata = _validate_tracking_inputs(
            tracking_stack,
            seed_model,
            manifest,
        )
        rounds = _positive_int_parameter(context, "rounds", default=2)
        min_tracked_fiducials = _positive_int_parameter(
            context,
            "min_tracked_fiducials",
            default=10,
        )
        coverage_warning = _fraction_parameter(
            context,
            "coverage_warning",
            default=0.8,
        )
        coverage_failure = _fraction_parameter(
            context,
            "coverage_failure",
            default=0.5,
        )
        if coverage_failure > coverage_warning:
            raise ValueError(
                "coverage_failure must be less than or equal to coverage_warning"
            )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _tracking_paths(context.output_dir, manifest)
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        beadtrack = resolve_imod_executable("beadtrack", context)
        imodinfo = resolve_imod_executable("imodinfo", context)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-beadtrack-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_model = temporary_root / f"{manifest.tilt_series_id}.fid"
            command = _beadtrack_command(
                beadtrack=beadtrack,
                image_file=tracking_stack.path,
                seed_model=seed_model.path,
                output_model=output_model,
                prealign_transform=metadata.prealign_transform,
                tilt_file=metadata.tilt_file,
                tracking_binning=metadata.tracking_binning,
                tilt_axis_angle_deg=metadata.tilt_axis_angle_deg,
                pixel_size_nm=metadata.raw_pixel_spacing_angstrom / 10.0,
                bead_diameter_px=metadata.fiducial_diameter_unbinned_px,
                box_size_px=metadata.beadtrack_box_size_unbinned_px,
                rounds=rounds,
            )
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(beadtrack, context),
            )
            _write_command_log(paths.beadtrack_log, command, result)
            _require_success(
                result,
                program="IMOD beadtrack",
                log_path=paths.beadtrack_log,
            )
            if not output_model.is_file():
                raise RuntimeError(
                    f"IMOD beadtrack did not write a fiducial model: {output_model}"
                )

            summary = _inspect_imod_model(
                output_model,
                imodinfo=imodinfo,
                context=context,
                command_runner=self._command_runner,
                log_path=paths.imodinfo_log,
                cwd=temporary_root,
            )
            report = _tracked_fiducial_qc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                model_id=f"{manifest.tilt_series_id}:fiducial_model",
                summary=summary,
                image_count=metadata.image_count,
                minimum_fiducials=min_tracked_fiducials,
                coverage_warning=coverage_warning,
                coverage_failure=coverage_failure,
            )
            _replace_file(output_model, paths.fiducial_model)

        paths.track_command.write_text(
            _track_command_text(
                image_file=tracking_stack.path,
                seed_model=seed_model.path,
                output_model=paths.fiducial_model,
                prealign_transform=metadata.prealign_transform,
                tilt_file=metadata.tilt_file,
                tracking_binning=metadata.tracking_binning,
                tilt_axis_angle_deg=metadata.tilt_axis_angle_deg,
                pixel_size_nm=metadata.raw_pixel_spacing_angstrom / 10.0,
                bead_diameter_px=metadata.fiducial_diameter_unbinned_px,
                box_size_px=metadata.beadtrack_box_size_unbinned_px,
                rounds=rounds,
            )
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        model_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:fiducial_model",
            kind=ArtifactKind.FIDUCIAL_MODEL,
            path=paths.fiducial_model,
            parent_ids=[tracking_stack.id, seed_model.id],
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "num_fiducials": summary.num_fiducials,
                "num_points": summary.num_points,
                "coverage_fraction": summary.coverage_fraction(
                    metadata.image_count
                ),
                "tracking_binning": metadata.tracking_binning,
                "rounds": rounds,
                "track_command_path": str(paths.track_command),
                "beadtrack_log_path": str(paths.beadtrack_log),
                "imodinfo_log_path": str(paths.imodinfo_log),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.fiducial_model,
                paths.track_command,
                paths.beadtrack_log,
                paths.imodinfo_log,
            ),
        )
        report_artifact = Artifact(
            id=f"{manifest.tilt_series_id}:qc:fiducial_tracking",
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[model_artifact.id],
            parameters={
                "qc_type": "fiducial_tracking",
                "tilt_series_id": manifest.tilt_series_id,
                "status": report.status.value,
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [model_artifact, report_artifact]


def generate_seed_and_register(
    backend: FiducialSeedBackend,
    tilt_stack: Artifact,
    alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Generate and register tracking-stack, seed-model, and QC artifacts."""

    artifacts = backend.generate(tilt_stack, alignment, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def track_fiducials_and_register(
    backend: FiducialTrackingBackend,
    tracking_stack: Artifact,
    seed_model: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Track and register the full fiducial model and its QC report."""

    artifacts = backend.track(tracking_stack, seed_model, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def parse_imod_model_ascii(text: str) -> ImodModelSummary:
    """Parse dimensions and contour point counts from `imodinfo -a` output."""

    max_xyz: tuple[int, int, int] | None = None
    points_per_fiducial: list[int] = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "max" and len(fields) == 4:
            try:
                max_xyz = (int(fields[1]), int(fields[2]), int(fields[3]))
            except ValueError as exc:
                raise ValueError("invalid IMOD model dimensions") from exc
        elif fields[0] == "contour" and len(fields) >= 4:
            try:
                point_count = int(fields[3])
            except ValueError as exc:
                raise ValueError("invalid IMOD contour point count") from exc
            if point_count < 1:
                raise ValueError("IMOD fiducial contours must contain points")
            points_per_fiducial.append(point_count)

    if max_xyz is None or any(axis_size < 1 for axis_size in max_xyz):
        raise ValueError("IMOD model summary has no valid max dimensions")
    return ImodModelSummary(
        max_xyz=max_xyz,
        points_per_fiducial=tuple(points_per_fiducial),
    )


@dataclass(frozen=True)
class _TrackingMetadata:
    prealign_transform: Path
    tilt_file: Path
    tracking_binning: int
    tilt_axis_angle_deg: float
    raw_pixel_spacing_angstrom: float
    fiducial_diameter_unbinned_px: float
    beadtrack_box_size_unbinned_px: int
    image_count: int


@dataclass(frozen=True)
class _SeedPaths:
    output_dir: Path
    tracking_stack: Path
    prealign_transform: Path
    tilt_file: Path
    seed_model: Path
    track_command: Path
    autofidseed_info: Path
    report: Path
    newstack_log: Path
    autofidseed_log: Path
    imodinfo_log: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.tracking_stack,
            self.prealign_transform,
            self.tilt_file,
            self.seed_model,
            self.track_command,
            self.autofidseed_info,
            self.report,
            self.newstack_log,
            self.autofidseed_log,
            self.imodinfo_log,
        )


@dataclass(frozen=True)
class _TrackingPaths:
    output_dir: Path
    fiducial_model: Path
    track_command: Path
    report: Path
    beadtrack_log: Path
    imodinfo_log: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.fiducial_model,
            self.track_command,
            self.report,
            self.beadtrack_log,
            self.imodinfo_log,
        )


def _seed_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    tracking_binning: int,
) -> _SeedPaths:
    output_dir = output_root / "fiducials" / manifest.tilt_series_id
    prefix = manifest.tilt_series_id
    return _SeedPaths(
        output_dir=output_dir,
        tracking_stack=output_dir
        / f"{prefix}_fiducial_bin{tracking_binning}_preali.mrc",
        prealign_transform=output_dir / f"{prefix}_fiducial.prexg",
        tilt_file=output_dir / f"{prefix}_fiducial.rawtlt",
        seed_model=output_dir / f"{prefix}.seed",
        track_command=output_dir / f"{prefix}_track.com",
        autofidseed_info=output_dir / f"{prefix}_autofidseed.info",
        report=output_dir / f"{prefix}_seed_qc.json",
        newstack_log=output_dir / f"{prefix}_fiducial_newstack.log",
        autofidseed_log=output_dir / f"{prefix}_autofidseed.log",
        imodinfo_log=output_dir / f"{prefix}_seed_imodinfo.log",
    )


def _tracking_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
) -> _TrackingPaths:
    output_dir = output_root / "fiducials" / manifest.tilt_series_id
    prefix = manifest.tilt_series_id
    return _TrackingPaths(
        output_dir=output_dir,
        fiducial_model=output_dir / f"{prefix}.fid",
        track_command=output_dir / f"{prefix}_beadtrack.com",
        report=output_dir / f"{prefix}_tracking_qc.json",
        beadtrack_log=output_dir / f"{prefix}_beadtrack.log",
        imodinfo_log=output_dir / f"{prefix}_fiducial_imodinfo.log",
    )


def _validate_seed_inputs(
    tilt_stack: Artifact,
    alignment: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[tuple[int, int, int], TiltAlignment]:
    if tilt_stack.kind is not ArtifactKind.TILT_STACK:
        raise ValueError(f"expected tilt stack artifact, got {tilt_stack.kind}")
    if not tilt_stack.path.exists():
        raise FileNotFoundError(f"tilt stack not found: {tilt_stack.path}")
    if tilt_stack.shape is None or len(tilt_stack.shape) != 3:
        raise ValueError("tilt stack artifact must record shape (tilts, y, x)")
    stack_shape = tuple(int(axis_size) for axis_size in tilt_stack.shape)
    if stack_shape[0] != manifest.num_tilts:
        raise ValueError(
            f"tilt stack has {stack_shape[0]} images, "
            f"manifest has {manifest.num_tilts}"
        )
    if alignment.kind is not ArtifactKind.ALIGNMENT:
        raise ValueError(f"expected alignment artifact, got {alignment.kind}")
    if not alignment.path.is_file():
        raise FileNotFoundError(f"alignment result not found: {alignment.path}")
    result = TiltAlignment.model_validate_json(alignment.path.read_text())
    if result.input_stack_id != tilt_stack.id:
        raise ValueError(
            f"alignment input stack is {result.input_stack_id}, "
            f"expected {tilt_stack.id}"
        )
    if result.tilt_series_id != manifest.tilt_series_id:
        raise ValueError(
            f"alignment is for {result.tilt_series_id}, "
            f"expected {manifest.tilt_series_id}"
        )
    if len(result.transforms) != manifest.num_tilts:
        raise ValueError(
            f"alignment has {len(result.transforms)} transforms for "
            f"{manifest.num_tilts} tilts"
        )
    return (
        (stack_shape[0], stack_shape[1], stack_shape[2]),
        result,
    )


def _validate_tracking_inputs(
    tracking_stack: Artifact,
    seed_model: Artifact,
    manifest: TiltSeriesManifest,
) -> _TrackingMetadata:
    if tracking_stack.kind is not ArtifactKind.ALIGNED_TILT_STACK:
        raise ValueError(
            f"expected aligned tilt stack artifact, got {tracking_stack.kind}"
        )
    if tracking_stack.parameters.get("purpose") != "fiducial_tracking":
        raise ValueError("aligned stack is not a fiducial-tracking stack")
    if not tracking_stack.path.is_file():
        raise FileNotFoundError(
            f"fiducial tracking stack not found: {tracking_stack.path}"
        )
    if seed_model.kind is not ArtifactKind.FIDUCIAL_SEED_MODEL:
        raise ValueError(
            f"expected fiducial seed model artifact, got {seed_model.kind}"
        )
    if not seed_model.path.is_file():
        raise FileNotFoundError(f"fiducial seed model not found: {seed_model.path}")
    if seed_model.parent_ids != [tracking_stack.id]:
        raise ValueError("fiducial seed model does not belong to tracking stack")

    included_z_values = tracking_stack.parameters.get("included_z_values")
    if not isinstance(included_z_values, list) or not all(
        isinstance(value, int) for value in included_z_values
    ):
        raise ValueError("tracking stack must record included_z_values")
    if tracking_stack.shape is None or tracking_stack.shape[0] != len(
        included_z_values
    ):
        raise ValueError("tracking stack shape does not match included_z_values")

    prealign_transform = _path_parameter(
        tracking_stack,
        "prealign_transform_path",
    )
    tilt_file = _path_parameter(tracking_stack, "tilt_file_path")
    tracking_binning = tracking_stack.binning
    if tracking_binning is None or tracking_binning < 1:
        raise ValueError("tracking stack must record a positive binning")
    tilt_axis_angle_deg = _numeric_artifact_parameter(
        tracking_stack,
        "tilt_axis_angle_deg",
    )
    raw_pixel_spacing = _numeric_artifact_parameter(
        tracking_stack,
        "raw_pixel_spacing_angstrom",
    )
    bead_diameter = _numeric_artifact_parameter(
        seed_model,
        "fiducial_diameter_unbinned_px",
    )
    box_size = seed_model.parameters.get("beadtrack_box_size_unbinned_px")
    if isinstance(box_size, bool) or not isinstance(box_size, int) or box_size < 2:
        raise ValueError(
            "fiducial seed model must record beadtrack_box_size_unbinned_px"
        )
    if seed_model.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError("fiducial seed model is for a different tilt series")
    return _TrackingMetadata(
        prealign_transform=prealign_transform,
        tilt_file=tilt_file,
        tracking_binning=tracking_binning,
        tilt_axis_angle_deg=tilt_axis_angle_deg,
        raw_pixel_spacing_angstrom=raw_pixel_spacing,
        fiducial_diameter_unbinned_px=bead_diameter,
        beadtrack_box_size_unbinned_px=box_size,
        image_count=len(included_z_values),
    )


def _raw_pixel_spacing(
    manifest: TiltSeriesManifest,
    tilt_stack: Artifact,
    context: BackendContext,
) -> float:
    configured = context.parameters.get("raw_pixel_spacing_angstrom")
    if configured is not None:
        return _positive_float_parameter(
            context,
            "raw_pixel_spacing_angstrom",
            default=1.0,
        )
    spacing = manifest.raw_pixel_spacing_angstrom
    if spacing is None or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("manifest must record positive raw pixel spacing")
    if tilt_stack.pixel_spacing_angstrom is None:
        raise ValueError("tilt stack artifact must record pixel spacing")
    if not isclose(
        tilt_stack.pixel_spacing_angstrom,
        spacing,
        rel_tol=1e-5,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"tilt stack pixel spacing {tilt_stack.pixel_spacing_angstrom} "
            f"does not match raw MRC spacing {spacing}; rerun tilt-series preparation"
        )
    return spacing


def _included_indices(
    alignment: TiltAlignment,
    manifest: TiltSeriesManifest,
) -> list[int]:
    excluded = set(alignment.excluded_z_values)
    included = [
        index
        for index, image in enumerate(manifest.images)
        if image.z_value not in excluded
    ]
    if len(included) < 3:
        raise ValueError("fiducial tracking requires at least three tilt images")
    return sorted(
        included,
        key=lambda index: manifest.images[index].tilt_angle_deg,
    )


def _write_imod_transforms(
    path: Path,
    included_indices: Sequence[int],
    manifest: TiltSeriesManifest,
    transforms_by_z: Mapping[int, AlignmentTransform],
    *,
    shift_scale: float,
) -> None:
    lines: list[str] = []
    for index in included_indices:
        z_value = manifest.images[index].z_value
        try:
            transform = transforms_by_z[z_value]
        except KeyError as exc:
            raise ValueError(f"alignment has no transform for z={z_value}") from exc
        lines.append(
            f"{transform.a11:.7f} {transform.a12:.7f} "
            f"{transform.a21:.7f} {transform.a22:.7f} "
            f"{transform.shift_x_px * shift_scale:.3f} "
            f"{transform.shift_y_px * shift_scale:.3f}\n"
        )
    path.write_text("".join(lines))


def _track_command_text(
    *,
    image_file: Path,
    seed_model: Path,
    output_model: Path,
    prealign_transform: Path,
    tilt_file: Path,
    tracking_binning: int,
    tilt_axis_angle_deg: float,
    pixel_size_nm: float,
    bead_diameter_px: float,
    box_size_px: int,
    rounds: int = 2,
) -> str:
    return (
        "$beadtrack -StandardInput\n"
        f"ImageFile\t{image_file.resolve()}\n"
        f"ImagesAreBinned\t{tracking_binning}\n"
        f"InputSeedModel\t{seed_model.resolve()}\n"
        f"OutputModel\t{output_model.resolve()}\n"
        f"PrealignTransformFile\t{prealign_transform.resolve()}\n"
        f"RotationAngle\t{tilt_axis_angle_deg:.6f}\n"
        f"PixelSize\t{pixel_size_nm:.8f}\n"
        f"TiltFile\t{tilt_file.resolve()}\n"
        "TiltDefaultGrouping\t7\n"
        "MagDefaultGrouping\t5\n"
        "RotDefaultGrouping\t1\n"
        f"BeadDiameter\t{bead_diameter_px:.6f}\n"
        f"BoxSizeXandY\t{box_size_px},{box_size_px}\n"
        "FillGaps\n"
        "MaxGapSize\t5\n"
        f"RoundsOfTracking\t{rounds}\n"
        "LocalAreaTracking\t1\n"
        "LocalAreaTargetSize\t1000\n"
        "MinBeadsInArea\t8\n"
        "MinOverlapBeads\t5\n"
        "MinViewsForTiltalign\t4\n"
        "MinTiltRangeToFindAxis\t10.0\n"
        "MinTiltRangeToFindAngles\t20.0\n"
        "MaxBeadsToAverage\t4\n"
    )


def _beadtrack_command(
    *,
    beadtrack: Path,
    image_file: Path,
    seed_model: Path,
    output_model: Path,
    prealign_transform: Path,
    tilt_file: Path,
    tracking_binning: int,
    tilt_axis_angle_deg: float,
    pixel_size_nm: float,
    bead_diameter_px: float,
    box_size_px: int,
    rounds: int,
) -> list[str]:
    return [
        str(beadtrack),
        "-ImageFile",
        str(image_file.resolve()),
        "-ImagesAreBinned",
        str(tracking_binning),
        "-InputSeedModel",
        str(seed_model.resolve()),
        "-OutputModel",
        str(output_model),
        "-PrealignTransformFile",
        str(prealign_transform.resolve()),
        "-RotationAngle",
        f"{tilt_axis_angle_deg:.6f}",
        "-PixelSize",
        f"{pixel_size_nm:.8f}",
        "-TiltFile",
        str(tilt_file.resolve()),
        "-TiltDefaultGrouping",
        "7",
        "-MagDefaultGrouping",
        "5",
        "-RotDefaultGrouping",
        "1",
        "-BeadDiameter",
        f"{bead_diameter_px:.6f}",
        "-BoxSizeXandY",
        f"{box_size_px},{box_size_px}",
        "-FillGaps",
        "-MaxGapSize",
        "5",
        "-RoundsOfTracking",
        str(rounds),
        "-LocalAreaTracking",
        "-LocalAreaTargetSize",
        "1000",
        "-MinBeadsInArea",
        "8",
        "-MinOverlapBeads",
        "5",
        "-MinViewsForTiltalign",
        "4",
        "-MinTiltRangeToFindAxis",
        "10.0",
        "-MinTiltRangeToFindAngles",
        "20.0",
        "-MaxBeadsToAverage",
        "4",
    ]


def _inspect_imod_model(
    model_path: Path,
    *,
    imodinfo: Path,
    context: BackendContext,
    command_runner: CommandRunner,
    log_path: Path,
    cwd: Path,
) -> ImodModelSummary:
    command = [str(imodinfo), "-a", str(model_path)]
    result = command_runner(
        command,
        cwd=cwd,
        env=imod_environment(imodinfo, context),
    )
    _write_command_log(log_path, command, result)
    _require_success(result, program="IMOD imodinfo", log_path=log_path)
    return parse_imod_model_ascii(result.stdout or "")


def _fiducial_qc(
    *,
    tilt_series_id: str,
    backend: str,
    stage: str,
    model_id: str,
    summary: ImodModelSummary,
    image_count: int,
    minimum_fiducials: int,
) -> FiducialModelQc:
    warnings: list[str] = []
    if summary.num_fiducials < minimum_fiducials:
        status = QcStatus.FAIL
        warnings.append(
            f"found {summary.num_fiducials} fiducials; "
            f"minimum is {minimum_fiducials}"
        )
    else:
        status = QcStatus.PASS
    minimum, midpoint, maximum = _point_statistics(summary)
    return FiducialModelQc(
        tilt_series_id=tilt_series_id,
        backend=backend,
        stage=stage,
        model_id=model_id,
        image_count=image_count,
        num_fiducials=summary.num_fiducials,
        num_points=summary.num_points,
        min_points_per_fiducial=minimum,
        median_points_per_fiducial=midpoint,
        max_points_per_fiducial=maximum,
        coverage_fraction=summary.coverage_fraction(image_count),
        status=status,
        warnings=warnings,
    )


def _tracked_fiducial_qc(
    *,
    tilt_series_id: str,
    backend: str,
    model_id: str,
    summary: ImodModelSummary,
    image_count: int,
    minimum_fiducials: int,
    coverage_warning: float,
    coverage_failure: float,
) -> FiducialModelQc:
    report = _fiducial_qc(
        tilt_series_id=tilt_series_id,
        backend=backend,
        stage="tracked",
        model_id=model_id,
        summary=summary,
        image_count=image_count,
        minimum_fiducials=minimum_fiducials,
    )
    coverage = report.coverage_fraction
    warnings = list(report.warnings)
    if coverage < coverage_failure:
        status = QcStatus.FAIL
        warnings.append(
            f"fiducial coverage {coverage:.3f} is below failure threshold "
            f"{coverage_failure:.3f}"
        )
    elif coverage < coverage_warning:
        status = (
            QcStatus.FAIL
            if report.status is QcStatus.FAIL
            else QcStatus.WARNING
        )
        warnings.append(
            f"fiducial coverage {coverage:.3f} is below warning threshold "
            f"{coverage_warning:.3f}"
        )
    else:
        status = report.status
    return report.model_copy(update={"status": status, "warnings": warnings})


def _point_statistics(
    summary: ImodModelSummary,
) -> tuple[int, float, int]:
    if not summary.points_per_fiducial:
        return 0, 0.0, 0
    return (
        min(summary.points_per_fiducial),
        float(median(summary.points_per_fiducial)),
        max(summary.points_per_fiducial),
    )


def _even_box_size(bead_diameter_px: float) -> int:
    return max(2, 2 * ceil((bead_diameter_px * 3.3) / 2.0))


def _path_parameter(artifact: Artifact, key: str) -> Path:
    value = artifact.parameters.get(key)
    if not isinstance(value, (str, Path)):
        raise ValueError(f"artifact parameter {key!r} must be a path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"artifact sidecar not found: {path}")
    return path


def _numeric_artifact_parameter(artifact: Artifact, key: str) -> float:
    value = artifact.parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"artifact parameter {key!r} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"artifact parameter {key!r} must be positive and finite")
    return normalized


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
        raise FileExistsError(f"fiducial outputs already exist: {joined}")


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
    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be positive and finite")
    return normalized


def _fraction_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
) -> float:
    value = _positive_float_parameter(context, key, default=default)
    if value > 1.0:
        raise ValueError(f"context parameter {key!r} must not exceed 1")
    return value


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)
