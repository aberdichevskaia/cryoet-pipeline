from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import torch
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    ParticleExtractionBackend,
    SubtomogramAveragingBackend,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    ClassAverageSummary,
    ParticleExtractionQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    SubtomogramAveragingQc,
    SubtomogramClassificationQc,
    TiltSeriesManifest,
)
from cryoet_pipeline.rotations import (
    cone_offsets,
    matrix_to_relion_euler,
    perturbation_matrix,
    relion_euler_to_matrix,
    resample_volume_zyx,
)
from cryoet_pipeline.runtime import DevicePreference


@dataclass(frozen=True)
class ParticleRecord:
    """One row of per-particle metadata carried alongside a particle stack."""

    index: int
    x_px: float
    y_px: float
    z_px: float
    rot_deg: float
    tilt_deg: float
    psi_deg: float
    shift_x_px: float
    shift_y_px: float
    shift_z_px: float
    score: float | None
    class_number: int
    random_subset: int


@dataclass(frozen=True)
class _PytomStarRow:
    index: int
    x_px: float
    y_px: float
    z_px: float
    rot_deg: float
    tilt_deg: float
    psi_deg: float
    lccmax: float | None


class CubicCropExtractionBackend:
    """Crop cubic subvolumes directly from the canonical tomogram at pick locations."""

    name = "cubic-crop"

    def extract(
        self,
        tomogram: Artifact,
        picks: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a particle-stack artifact and its QC artifact."""

        tomogram_shape = _validate_tomogram(tomogram, manifest)
        _validate_picks(picks, tomogram, manifest)
        box_size = _positive_int_parameter(context, "box_size_px")
        minimum_score = _optional_float_parameter(context, "minimum_score")
        overwrite = _bool_parameter(context, "overwrite", default=False)

        paths = _extraction_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.particles_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)
        if paths.particle_stack.exists():
            _remove_path(paths.particle_stack)

        rows = _read_pytom_star(picks.path)
        requested_count = len(rows)
        half = box_size // 2

        tomogram_array = cast(Any, zarr.open(tomogram.path, mode="r"))
        kept: list[tuple[_PytomStarRow, tuple[int, int, int]]] = []
        dropped_edge = 0
        dropped_score = 0
        for row in rows:
            if minimum_score is not None and row.lccmax is not None and row.lccmax < minimum_score:
                dropped_score += 1
                continue
            bounds = _crop_start(row, box_size=box_size, half=half, shape=tomogram_shape)
            if bounds is None:
                dropped_edge += 1
                continue
            kept.append((row, bounds))

        particle_count = len(kept)
        particle_stack = cast(
            Any,
            zarr.open(
                paths.particle_stack,
                mode="w",
                shape=(particle_count, box_size, box_size, box_size),
                chunks=(1, box_size, box_size, box_size),
                dtype=np.float32,
            ),
        )
        particle_stack.attrs.update({"axis_order": AxisOrder.NZYX.value})

        records: list[ParticleRecord] = []
        for output_index, (row, (z_start, y_start, x_start)) in enumerate(kept):
            crop = np.asarray(
                tomogram_array[
                    z_start : z_start + box_size,
                    y_start : y_start + box_size,
                    x_start : x_start + box_size,
                ],
                dtype=np.float32,
            )
            particle_stack[output_index] = crop
            records.append(
                ParticleRecord(
                    index=output_index,
                    x_px=row.x_px,
                    y_px=row.y_px,
                    z_px=row.z_px,
                    rot_deg=row.rot_deg,
                    tilt_deg=row.tilt_deg,
                    psi_deg=row.psi_deg,
                    shift_x_px=0.0,
                    shift_y_px=0.0,
                    shift_z_px=0.0,
                    score=row.lccmax,
                    class_number=1,
                    random_subset=1 + (output_index % 2),
                )
            )
        write_particle_star(paths.particle_metadata, records)

        warnings: list[str] = []
        status = QcStatus.PASS
        if particle_count == 0:
            status = QcStatus.FAIL
            warnings.append("no particles survived extraction")
        elif requested_count > 0 and dropped_edge / requested_count > 0.5:
            status = QcStatus.WARNING
            warnings.append(
                f"{dropped_edge}/{requested_count} particles dropped for being too close to "
                "the tomogram edge for the requested box size"
            )

        particle_stack_id = (
            f"{manifest.tilt_series_id}:particle_stack:{self.name}"
            f"{variant_suffix(context)}"

        )
        report = ParticleExtractionQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            input_tomogram_id=tomogram.id,
            picks_id=picks.id,
            particle_stack_id=particle_stack_id,
            box_size_px=box_size,
            requested_particle_count=requested_count,
            extracted_particle_count=particle_count,
            dropped_edge_count=dropped_edge,
            dropped_score_count=dropped_score,
            minimum_score_threshold=minimum_score,
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        particle_stack_artifact = Artifact(
            id=particle_stack_id,
            variant=variant_label(context),
            kind=ArtifactKind.PARTICLE_STACK,
            path=paths.particle_stack,
            parent_ids=[tomogram.id, picks.id],
            shape=(particle_count, box_size, box_size, box_size),
            dtype="float32",
            axis_order=AxisOrder.NZYX,
            pixel_spacing_angstrom=tomogram.pixel_spacing_angstrom,
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "input_tomogram_id": tomogram.id,
                "picks_id": picks.id,
                "box_size_px": box_size,
                "particle_count": particle_count,
                "metadata_path": str(paths.particle_metadata),
                "qc_path": str(paths.report),
            },
            software_versions={"zarr": _package_version("zarr")},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(paths.particle_stack)
            + _path_size_bytes(paths.particle_metadata),
        )
        qc_artifact = Artifact(
            id=(
                f"{manifest.tilt_series_id}:qc:particle_extraction:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[particle_stack_artifact.id],
            parameters={
                "qc_type": "particle_extraction",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "status": status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [particle_stack_artifact, qc_artifact]


class FftRefinementAveragingBackend:
    """Iteratively refine per-particle orientation/shift and average, using a
    coarse-to-fine local search scored by normalized cross-correlation.

    This is a deliberately simple baseline: a brute-force local grid search
    (not a smart optimizer), a single class (no RELION-style multi-class
    responsibilities), and no missing-wedge or CTF weighting.
    """

    name = "fft-refinement"

    def average(
        self,
        particles: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return an average map, refined per-particle metadata, and a QC artifact."""

        _validate_particle_stack(particles, manifest)
        num_iterations = _positive_int_parameter(context, "num_iterations", default=5)
        initial_angular_step_deg = _positive_float_parameter(
            context, "initial_angular_step_deg", default=6.0
        )
        initial_shift_step_px = _positive_float_parameter(
            context, "initial_shift_step_px", default=2.0
        )
        initial_reference_path = _optional_path_parameter(context, "initial_reference_path")
        initial_reference_correlation_threshold = _positive_float_parameter(
            context, "initial_reference_correlation_threshold", default=0.3
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        device = torch_device_for_preference(context.device)

        paths = _averaging_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        particle_stack = cast(Any, zarr.open(particles.path, mode="r"))
        particle_count = int(particle_stack.shape[0])
        if particle_count == 0:
            raise ValueError("particle stack has no particles to average")
        box_size = int(particle_stack.shape[1])
        metadata_path = Path(cast(str, particles.parameters["metadata_path"]))
        records = read_particle_star(metadata_path)

        raw_volumes = [
            torch.as_tensor(np.asarray(particle_stack[index]), device=device)
            for index in range(particle_count)
        ]

        aligned = torch.stack(
            [
                resample_volume_zyx(
                    raw_volumes[record.index],
                    relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
                    np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px]),
                    device=device,
                )
                for record in records
            ]
        )

        initial_reference_correlation_mean: float | None = None
        initial_reference_correlation_fraction_above_threshold: float | None = None
        if initial_reference_path is not None:
            initial_reference = _load_reference_mrc(
                initial_reference_path, expected_box_size=box_size, device=device
            )
            initial_correlations = [
                _normalized_correlation(aligned[index], initial_reference)
                for index in range(particle_count)
            ]
            initial_reference_correlation_mean = float(np.mean(initial_correlations))
            initial_reference_correlation_fraction_above_threshold = float(
                np.mean(
                    [
                        correlation > initial_reference_correlation_threshold
                        for correlation in initial_correlations
                    ]
                )
            )
            reference = initial_reference
        else:
            reference = aligned.mean(dim=0)

        final_correlations = [math.nan] * particle_count
        for iteration in range(num_iterations):
            annealing = 2**iteration
            angular_step_deg = initial_angular_step_deg / annealing
            shift_step_px = initial_shift_step_px / annealing
            new_aligned = torch.empty_like(aligned)
            for record in records:
                candidate = _search_local_alignment(
                    raw_volumes[record.index],
                    reference,
                    record,
                    angular_step_deg=angular_step_deg,
                    shift_step_px=shift_step_px,
                    device=device,
                )
                records[record.index] = replace(
                    record,
                    rot_deg=candidate.rot_deg,
                    tilt_deg=candidate.tilt_deg,
                    psi_deg=candidate.psi_deg,
                    shift_x_px=candidate.shift_x_px,
                    shift_y_px=candidate.shift_y_px,
                    shift_z_px=candidate.shift_z_px,
                )
                new_aligned[record.index] = candidate.aligned_volume
                final_correlations[record.index] = candidate.correlation
            aligned = new_aligned
            reference = aligned.mean(dim=0)

        write_particle_star(paths.particle_metadata, records)
        _write_average_mrc(
            paths.average_map,
            reference.detach().cpu().numpy(),
            pixel_spacing_angstrom=particles.pixel_spacing_angstrom,
        )

        warnings: list[str] = []
        status = QcStatus.PASS
        mean_final_correlation = float(np.mean(final_correlations))
        minimum_final_correlation = float(np.min(final_correlations))
        if minimum_final_correlation < 0.0:
            status = QcStatus.WARNING
            warnings.append(
                f"minimum final correlation {minimum_final_correlation:.3f} is negative; "
                "some particles never converged onto the average"
            )
        if (
            initial_reference_correlation_fraction_above_threshold is not None
            and initial_reference_correlation_fraction_above_threshold < 0.1
        ):
            status = QcStatus.WARNING
            warnings.append(
                "fewer than 10% of particles correlate with the initial reference above "
                f"{initial_reference_correlation_threshold:.2f} before refinement - check the "
                "rot/tilt/psi convention and initial reference box size/pixel spacing"
            )

        average_id = (
            f"{manifest.tilt_series_id}:subtomogram_average:{self.name}"
            f"{variant_suffix(context)}"

        )
        report = SubtomogramAveragingQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            particle_stack_id=particles.id,
            average_id=average_id,
            particle_count=particle_count,
            iterations_run=num_iterations,
            mean_final_correlation=mean_final_correlation,
            minimum_final_correlation=minimum_final_correlation,
            initial_reference_correlation_mean=initial_reference_correlation_mean,
            initial_reference_correlation_fraction_above_threshold=(
                initial_reference_correlation_fraction_above_threshold
            ),
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        average_artifact = Artifact(
            id=average_id,
            variant=variant_label(context),
            kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
            path=paths.average_map,
            parent_ids=[particles.id],
            shape=(box_size, box_size, box_size),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=particles.pixel_spacing_angstrom,
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "particle_stack_id": particles.id,
                "particle_count": particle_count,
                "iterations_run": num_iterations,
                "metadata_path": str(paths.particle_metadata),
                "qc_path": str(paths.report),
            },
            software_versions={"torch": _package_version("torch")},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(paths.average_map)
            + _path_size_bytes(paths.particle_metadata),
        )
        qc_artifact = Artifact(
            id=(
                f"{manifest.tilt_series_id}:qc:subtomogram_averaging:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[average_artifact.id],
            parameters={
                "qc_type": "subtomogram_averaging",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "status": status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [average_artifact, qc_artifact]


class ConeSearchRefinementBackend:
    """Multi-reference refinement and classification, modelled on DYNAMO.

    This is a native torch implementation of DYNAMO's approach, not an adapter:
    nothing here shells out to DYNAMO or MATLAB. Unlike the tool-named backends
    elsewhere in this pipeline, it has no external dependency.

    Differs from `FftRefinementAveragingBackend` in three ways that matter:

    - Orientations are searched on a *cone* plus an in-plane sweep, rather than
      by nudging each Euler angle independently. Independent Euler nudges sample
      orientation space unevenly and degenerate near zero tilt.
    - Shifts come from the cross-correlation peak, so the whole allowed shift
      range is explored by one FFT instead of a grid of resamplings.
    - Several references can be refined at once, with each particle hard-assigned
      to its best-scoring class, which is what makes this classification rather
      than plain averaging.

    Search ranges and their sampling shrink together each iteration, so the
    number of candidates stays constant while the search tightens.
    """

    name = "cone-search"

    def average(
        self,
        particles: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return one average per class, refined metadata, and a QC artifact."""

        _validate_particle_stack(particles, manifest)
        num_classes = _positive_int_parameter(context, "num_classes", default=1)
        num_iterations = _positive_int_parameter(context, "num_iterations", default=5)
        cone_range_deg = _nonnegative_float_parameter(context, "cone_range_deg", default=30.0)
        cone_sampling_deg = _positive_float_parameter(context, "cone_sampling_deg", default=10.0)
        inplane_range_deg = _nonnegative_float_parameter(
            context, "inplane_range_deg", default=30.0
        )
        inplane_sampling_deg = _positive_float_parameter(
            context, "inplane_sampling_deg", default=10.0
        )
        shift_range_px = _nonnegative_float_parameter(context, "shift_range_px", default=2.0)
        refine_factor = _positive_float_parameter(context, "refine_factor", default=2.0)
        seed = _int_parameter(context, "seed", default=0)
        initial_reference_path = _optional_path_parameter(context, "initial_reference_path")
        overwrite = _bool_parameter(context, "overwrite", default=False)
        device = torch_device_for_preference(context.device)

        paths = _averaging_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        particle_stack = cast(Any, zarr.open(particles.path, mode="r"))
        particle_count = int(particle_stack.shape[0])
        if particle_count == 0:
            raise ValueError("particle stack has no particles to average")
        if num_classes > particle_count:
            raise ValueError(
                f"cannot refine {num_classes} classes from {particle_count} particles"
            )
        box_size = int(particle_stack.shape[1])
        pixel_spacing_angstrom = _pixel_spacing_angstrom(particles)
        particle_diameter_angstrom = _positive_float_parameter(
            context,
            "particle_diameter_angstrom",
            default=box_size * pixel_spacing_angstrom,
        )
        target_resolution_angstrom = _positive_float_parameter(
            context,
            "target_resolution_angstrom",
            default=2.0 * pixel_spacing_angstrom,
        )
        minimum_sampling_deg = crowther_angular_increment_deg(
            particle_diameter_angstrom, target_resolution_angstrom
        )
        metadata_path = Path(cast(str, particles.parameters["metadata_path"]))
        records = read_particle_star(metadata_path)

        raw_volumes = [
            torch.as_tensor(np.asarray(particle_stack[index]), device=device)
            for index in range(particle_count)
        ]
        aligned = [
            resample_volume_zyx(
                raw_volumes[record.index],
                relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
                np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px]),
                device=device,
            )
            for record in records
        ]

        references, assignments = _initial_references(
            aligned,
            num_classes=num_classes,
            seed=seed,
            initial_reference_path=initial_reference_path,
            box_size=box_size,
            device=device,
        )
        for record, class_number in zip(records, assignments, strict=True):
            records[record.index] = replace(record, class_number=class_number)

        warnings: list[str] = []
        correlations = [math.nan] * particle_count
        reassignment_fraction: float | None = None
        final_cone_sampling_deg = cone_sampling_deg
        clamped_sampling = False
        for iteration in range(num_iterations):
            scale = refine_factor**iteration
            # Refining past the point where a rotation moves the outermost
            # density by less than one resolution element buys nothing, so the
            # sampling stops shrinking there even as the range keeps closing in.
            cone_sampling = max(cone_sampling_deg / scale, minimum_sampling_deg)
            inplane_sampling = max(inplane_sampling_deg / scale, minimum_sampling_deg)
            clamped_sampling = clamped_sampling or cone_sampling_deg / scale < minimum_sampling_deg
            final_cone_sampling_deg = cone_sampling

            offsets = cone_offsets(cone_range_deg / scale, cone_sampling)
            in_plane_offsets = _symmetric_offsets(inplane_range_deg / scale, inplane_sampling)
            # Shifts are found at whole-pixel accuracy from the correlation peak,
            # so never close the window below a single pixel.
            iteration_shift_range = max(shift_range_px / scale, 1.0)

            reassigned = 0
            new_aligned: list[torch.Tensor] = [torch.empty(0)] * particle_count
            for record in records:
                candidate = _search_multi_reference(
                    raw_volumes[record.index],
                    references,
                    record,
                    cone=offsets,
                    in_plane_offsets=in_plane_offsets,
                    max_shift_px=iteration_shift_range,
                    device=device,
                )
                if candidate.class_number != record.class_number:
                    reassigned += 1
                rot_deg, tilt_deg, psi_deg = matrix_to_relion_euler(candidate.matrix)
                records[record.index] = replace(
                    record,
                    rot_deg=rot_deg,
                    tilt_deg=tilt_deg,
                    psi_deg=psi_deg,
                    shift_x_px=float(candidate.shift[0]),
                    shift_y_px=float(candidate.shift[1]),
                    shift_z_px=float(candidate.shift[2]),
                    class_number=candidate.class_number,
                )
                new_aligned[record.index] = candidate.aligned_volume
                correlations[record.index] = candidate.correlation

            aligned = new_aligned
            reassignment_fraction = reassigned / particle_count
            references, empty_classes = _recompute_references(
                aligned, records, references, num_classes=num_classes
            )
            for class_number in empty_classes:
                warnings.append(
                    f"class {class_number} kept no particles in iteration {iteration + 1}; "
                    "its previous reference was carried over"
                )

        write_particle_star(paths.particle_metadata, records)

        class_summaries: list[ClassAverageSummary] = []
        average_artifacts: list[Artifact] = []
        for class_number, reference in enumerate(references, start=1):
            members = [record for record in records if record.class_number == class_number]
            average_path = _class_average_path(paths, class_number, num_classes=num_classes)
            _write_average_mrc(
                average_path,
                reference.detach().cpu().numpy(),
                pixel_spacing_angstrom=particles.pixel_spacing_angstrom,
            )
            average_id = _class_average_id(
                manifest.tilt_series_id, self.name, class_number, num_classes=num_classes
            )
            member_correlations = [
                correlations[record.index]
                for record in members
                if not math.isnan(correlations[record.index])
            ]
            class_summaries.append(
                ClassAverageSummary(
                    class_number=class_number,
                    average_id=average_id,
                    particle_count=len(members),
                    mean_correlation=(
                        float(np.mean(member_correlations)) if member_correlations else None
                    ),
                )
            )
            average_artifacts.append(
                Artifact(
                    id=average_id,
                    variant=variant_label(context),
                    kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
                    path=average_path,
                    parent_ids=[particles.id],
                    shape=(box_size, box_size, box_size),
                    dtype="float32",
                    axis_order=AxisOrder.ZYX,
                    pixel_spacing_angstrom=particles.pixel_spacing_angstrom,
                    parameters={
                        "backend": self.name,
                        "tilt_series_id": manifest.tilt_series_id,
                        "particle_stack_id": particles.id,
                        "class_number": class_number,
                        "num_classes": num_classes,
                        "particle_count": len(members),
                        "iterations_run": num_iterations,
                        "cone_range_deg": cone_range_deg,
                        "cone_sampling_deg": cone_sampling_deg,
                        "inplane_range_deg": inplane_range_deg,
                        "inplane_sampling_deg": inplane_sampling_deg,
                        "shift_range_px": shift_range_px,
                        "metadata_path": str(paths.particle_metadata),
                        "qc_path": str(paths.report),
                    },
                    software_versions={"torch": _package_version("torch")},
                    storage_role=StorageRole.CANONICAL,
                    retention_policy=RetentionPolicy.KEEP,
                    can_recompute=True,
                    size_bytes=_path_size_bytes(average_path),
                )
            )

        status = QcStatus.PASS
        finite_correlations = [value for value in correlations if not math.isnan(value)]
        mean_correlation = float(np.mean(finite_correlations)) if finite_correlations else None
        minimum_correlation = float(np.min(finite_correlations)) if finite_correlations else None
        if minimum_correlation is not None and minimum_correlation < 0.0:
            status = QcStatus.WARNING
            warnings.append(
                f"minimum final correlation {minimum_correlation:.3f} is negative; "
                "some particles never converged onto their class average"
            )
        if clamped_sampling:
            warnings.append(
                f"angular sampling hit the {minimum_sampling_deg:.2f} deg Crowther "
                f"increment for a {particle_diameter_angstrom:.0f} A particle at "
                f"{target_resolution_angstrom:.1f} A; further iterations narrow the "
                "range, not the step"
            )
        if num_classes > 1 and reassignment_fraction is not None and reassignment_fraction > 0.1:
            status = QcStatus.WARNING
            warnings.append(
                f"{reassignment_fraction:.0%} of particles changed class in the last "
                "iteration; classification has not converged, so run more iterations"
            )
        if warnings and status is QcStatus.PASS:
            status = QcStatus.WARNING

        report = SubtomogramClassificationQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            particle_stack_id=particles.id,
            particle_count=particle_count,
            num_classes=num_classes,
            iterations_run=num_iterations,
            cone_range_deg=cone_range_deg,
            cone_sampling_deg=cone_sampling_deg,
            inplane_range_deg=inplane_range_deg,
            inplane_sampling_deg=inplane_sampling_deg,
            shift_range_px=shift_range_px,
            particle_diameter_angstrom=particle_diameter_angstrom,
            target_resolution_angstrom=target_resolution_angstrom,
            crowther_increment_deg=minimum_sampling_deg,
            final_cone_sampling_deg=final_cone_sampling_deg,
            classes=class_summaries,
            mean_final_correlation=mean_correlation,
            minimum_final_correlation=minimum_correlation,
            reassignment_fraction_last_iteration=reassignment_fraction,
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        qc_artifact = Artifact(
            id=(
                f"{manifest.tilt_series_id}:qc:subtomogram_classification:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[average_artifacts[0].id],
            parameters={
                "qc_type": "subtomogram_classification",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "num_classes": num_classes,
                "status": status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [*average_artifacts, qc_artifact]


@dataclass(frozen=True)
class _MultiReferenceCandidate:
    class_number: int
    matrix: np.ndarray
    shift: np.ndarray
    correlation: float
    aligned_volume: torch.Tensor


def _search_multi_reference(
    raw: torch.Tensor,
    references: Sequence[torch.Tensor],
    record: ParticleRecord,
    *,
    cone: Sequence[tuple[float, float]],
    in_plane_offsets: Sequence[float],
    max_shift_px: float,
    device: torch.device,
) -> _MultiReferenceCandidate:
    """Score one particle against every reference over the cone, keeping the best."""

    current_matrix = relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg)
    current_shift = np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px])

    best_class = record.class_number
    best_matrix = current_matrix
    best_shift = current_shift
    best_correlation = -math.inf
    for polar, azimuth in cone:
        for in_plane in in_plane_offsets:
            candidate_matrix = current_matrix @ perturbation_matrix(polar, azimuth, in_plane)
            rotated = resample_volume_zyx(raw, candidate_matrix, current_shift, device=device)
            for class_number, reference in enumerate(references, start=1):
                delta, correlation = _best_shift_and_correlation(
                    rotated, reference, max_shift_px=max_shift_px
                )
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_class = class_number
                    best_matrix = candidate_matrix
                    best_shift = current_shift + candidate_matrix @ delta

    aligned_volume = resample_volume_zyx(raw, best_matrix, best_shift, device=device)
    return _MultiReferenceCandidate(
        class_number=best_class,
        matrix=best_matrix,
        shift=best_shift,
        correlation=best_correlation,
        aligned_volume=aligned_volume,
    )


def _best_shift_and_correlation(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_shift_px: float,
) -> tuple[np.ndarray, float]:
    """Return the best shift within `max_shift_px` and its correlation.

    The whole allowed shift range is evaluated at once through the
    cross-correlation theorem, so widening the range costs nothing extra.
    The returned shift is in reference-frame `(x, y, z)` and is meant to be
    folded into an existing sampling shift as `shift + matrix @ delta`.
    """

    centered_candidate = (candidate - candidate.mean()).to(torch.float32)
    centered_reference = (reference - reference.mean()).to(torch.float32)
    norm = torch.sqrt(
        torch.sum(centered_candidate**2) * torch.sum(centered_reference**2)
    )
    if float(norm) <= 1e-12:
        return np.zeros(3), 0.0

    spectrum = torch.conj(torch.fft.rfftn(centered_reference)) * torch.fft.rfftn(
        centered_candidate
    )
    correlation_map = torch.fft.fftshift(
        torch.fft.irfftn(spectrum, s=tuple(centered_candidate.shape))
    ) / norm

    size = int(centered_candidate.shape[0])
    center = size // 2
    radius = int(min(max(int(math.floor(max_shift_px)), 0), center))
    window = correlation_map[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    flat_index = int(torch.argmax(window).item())
    span = 2 * radius + 1
    delta_z = flat_index // (span * span) - radius
    delta_y = (flat_index // span) % span - radius
    delta_x = flat_index % span - radius
    return (
        np.array([float(delta_x), float(delta_y), float(delta_z)]),
        float(window.reshape(-1)[flat_index].item()),
    )


def crowther_angular_increment_deg(
    diameter_angstrom: float,
    resolution_angstrom: float,
) -> float:
    """Angular increment from the Crowther criterion.

    With `r_max = 1 / d_res` the maximum spatial frequency and `D` the particle
    diameter, `dtheta = 180 / (pi * r_max * D)`, i.e.

        dtheta = degrees(d_res / D)

    Sampling finer than this cannot resolve a new orientation: the criterion is
    the angular spacing at which adjacent views become independent. Note that
    RELION's initial-sampling heuristic, `arctan(2 * d_res / D)`, is about twice
    this at small angles; this function is the Crowther value itself.
    """

    if diameter_angstrom <= 0.0:
        raise ValueError("particle diameter must be greater than zero")
    if resolution_angstrom <= 0.0:
        raise ValueError("target resolution must be greater than zero")
    return math.degrees(resolution_angstrom / diameter_angstrom)


def _pixel_spacing_angstrom(particles: Artifact) -> float:
    value = particles.pixel_spacing_angstrom
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "particle stack pixel spacing is required to apply the Crowther criterion; "
            "pass particle_diameter_angstrom and target_resolution_angstrom explicitly"
        )
    return value


def _symmetric_offsets(range_deg: float, sampling_deg: float) -> list[float]:
    """Offsets centred on zero, so the current orientation is always a candidate."""

    if sampling_deg <= 0.0:
        raise ValueError("sampling must be greater than zero")
    half = range_deg / 2.0
    steps = int(math.floor(half / sampling_deg + 1e-9))
    offsets = [0.0]
    for step in range(1, steps + 1):
        offsets.extend([step * sampling_deg, -step * sampling_deg])
    return offsets


def _initial_references(
    aligned: Sequence[torch.Tensor],
    *,
    num_classes: int,
    seed: int,
    initial_reference_path: Path | None,
    box_size: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[int]]:
    """Seed references and the starting class assignment.

    A single class starts from the global average, or from a supplied reference.
    Several classes start from a deterministic random partition, which is the
    usual way to break the symmetry between otherwise identical references.
    """

    if num_classes == 1:
        if initial_reference_path is not None:
            reference = _load_reference_mrc(
                initial_reference_path, expected_box_size=box_size, device=device
            )
        else:
            reference = torch.stack(list(aligned)).mean(dim=0)
        return [reference], [1] * len(aligned)

    generator = np.random.default_rng(seed)
    assignments = [
        int(value) for value in generator.integers(1, num_classes + 1, size=len(aligned))
    ]
    # Guarantee every class starts with at least one particle.
    for class_number in range(1, num_classes + 1):
        if class_number not in assignments:
            assignments[class_number - 1] = class_number

    references = []
    for class_number in range(1, num_classes + 1):
        members = [
            volume
            for volume, assigned in zip(aligned, assignments, strict=True)
            if assigned == class_number
        ]
        references.append(torch.stack(members).mean(dim=0))
    return references, assignments


def _recompute_references(
    aligned: Sequence[torch.Tensor],
    records: Sequence[ParticleRecord],
    previous: Sequence[torch.Tensor],
    *,
    num_classes: int,
) -> tuple[list[torch.Tensor], list[int]]:
    references: list[torch.Tensor] = []
    empty_classes: list[int] = []
    for class_number in range(1, num_classes + 1):
        members = [
            aligned[record.index] for record in records if record.class_number == class_number
        ]
        if members:
            references.append(torch.stack(members).mean(dim=0))
        else:
            empty_classes.append(class_number)
            references.append(previous[class_number - 1])
    return references, empty_classes


def _class_average_id(
    tilt_series_id: str,
    backend_name: str,
    class_number: int,
    *,
    num_classes: int,
) -> str:
    if num_classes == 1:
        return f"{tilt_series_id}:subtomogram_average:{backend_name}"
    return f"{tilt_series_id}:subtomogram_average:{backend_name}:class{class_number:02d}"


def _class_average_path(
    paths: _AveragingPaths,
    class_number: int,
    *,
    num_classes: int,
) -> Path:
    if num_classes == 1:
        return paths.average_map
    return paths.average_map.with_name(
        paths.average_map.name.replace("_average.mrc", f"_class{class_number:02d}_average.mrc")
    )


def extract_and_register(
    backend: ParticleExtractionBackend,
    tomogram: Artifact,
    picks: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Extract particles and register the particle-stack and QC artifacts."""

    artifacts = backend.extract(tomogram, picks, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def average_and_register(
    backend: SubtomogramAveragingBackend,
    particles: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Average particles and register the average-map and QC artifacts."""

    artifacts = backend.average(particles, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


@dataclass(frozen=True)
class _AlignmentCandidate:
    rot_deg: float
    tilt_deg: float
    psi_deg: float
    shift_x_px: float
    shift_y_px: float
    shift_z_px: float
    correlation: float
    aligned_volume: torch.Tensor


def _search_local_alignment(
    raw: torch.Tensor,
    reference: torch.Tensor,
    record: ParticleRecord,
    *,
    angular_step_deg: float,
    shift_step_px: float,
    device: torch.device,
) -> _AlignmentCandidate:
    """Brute-force local search: refine rotation first, then shift, both as a
    fixed 3-position-per-axis grid (-step, 0, +step) around the current
    estimate, scored by normalized cross-correlation against `reference`.
    Callers anneal `angular_step_deg`/`shift_step_px` down across iterations
    for coarse-to-fine refinement.
    """

    current_shift = np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px])
    angle_deltas = (-angular_step_deg, 0.0, angular_step_deg)
    shift_deltas = (-shift_step_px, 0.0, shift_step_px)

    best_rot, best_tilt, best_psi = record.rot_deg, record.tilt_deg, record.psi_deg
    best_correlation = -math.inf
    for delta_rot in angle_deltas:
        for delta_tilt in angle_deltas:
            for delta_psi in angle_deltas:
                candidate_rot = record.rot_deg + delta_rot
                candidate_tilt = record.tilt_deg + delta_tilt
                candidate_psi = record.psi_deg + delta_psi
                matrix = relion_euler_to_matrix(candidate_rot, candidate_tilt, candidate_psi)
                candidate = resample_volume_zyx(raw, matrix, current_shift, device=device)
                correlation = _normalized_correlation(candidate, reference)
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_rot, best_tilt, best_psi = candidate_rot, candidate_tilt, candidate_psi

    best_matrix = relion_euler_to_matrix(best_rot, best_tilt, best_psi)
    best_shift = current_shift
    best_volume = resample_volume_zyx(raw, best_matrix, best_shift, device=device)
    for delta_x in shift_deltas:
        for delta_y in shift_deltas:
            for delta_z in shift_deltas:
                shift = current_shift + np.array([delta_x, delta_y, delta_z])
                candidate = resample_volume_zyx(raw, best_matrix, shift, device=device)
                correlation = _normalized_correlation(candidate, reference)
                if correlation > best_correlation:
                    best_correlation = correlation
                    best_shift = shift
                    best_volume = candidate

    return _AlignmentCandidate(
        rot_deg=best_rot,
        tilt_deg=best_tilt,
        psi_deg=best_psi,
        shift_x_px=float(best_shift[0]),
        shift_y_px=float(best_shift[1]),
        shift_z_px=float(best_shift[2]),
        correlation=best_correlation,
        aligned_volume=best_volume,
    )


def _normalized_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two volumes at zero additional
    shift (no missing-wedge or CTF weighting).
    """

    a_centered = (a - a.mean()).to(torch.float32)
    b_centered = (b - b.mean()).to(torch.float32)
    numerator = torch.sum(a_centered * b_centered)
    denominator = torch.sqrt(torch.sum(a_centered**2) * torch.sum(b_centered**2))
    if denominator <= 1e-12:
        return 0.0
    return float((numerator / denominator).item())


def _load_reference_mrc(
    path: Path, *, expected_box_size: int, device: torch.device
) -> torch.Tensor:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
    expected_shape = (expected_box_size, expected_box_size, expected_box_size)
    if data.shape != expected_shape:
        raise ValueError(
            f"initial reference shape {data.shape} does not match particle box size "
            f"{expected_shape}: {path}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"initial reference contains nonfinite values: {path}")
    return torch.as_tensor(data, device=device)


def _write_average_mrc(
    path: Path, data: np.ndarray, *, pixel_spacing_angstrom: float | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        if pixel_spacing_angstrom is not None:
            mrc.voxel_size = pixel_spacing_angstrom


def torch_device_for_preference(preference: DevicePreference) -> torch.device:
    if preference is DevicePreference.CUDA:
        return torch.device("cuda")
    if preference is DevicePreference.MPS:
        return torch.device("mps")
    if preference is DevicePreference.CPU:
        return torch.device("cpu")
    raise ValueError(f"unresolved device preference: {preference.value!r}")


def _crop_start(
    row: _PytomStarRow, *, box_size: int, half: int, shape: tuple[int, ...]
) -> tuple[int, int, int] | None:
    z_start = int(round(row.z_px)) - half
    y_start = int(round(row.y_px)) - half
    x_start = int(round(row.x_px)) - half
    if z_start < 0 or y_start < 0 or x_start < 0:
        return None
    if (
        z_start + box_size > shape[0]
        or y_start + box_size > shape[1]
        or x_start + box_size > shape[2]
    ):
        return None
    return z_start, y_start, x_start


def _validate_tomogram(tomogram: Artifact, manifest: TiltSeriesManifest) -> tuple[int, ...]:
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if tomogram.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError(
            f"tomogram is for {tomogram.parameters.get('tilt_series_id')}, "
            f"expected {manifest.tilt_series_id}"
        )
    if tomogram.axis_order is not AxisOrder.ZYX:
        raise ValueError("particle extraction requires a canonical ZYX tomogram")
    if not tomogram.path.is_dir():
        raise FileNotFoundError(f"canonical tomogram Zarr not found: {tomogram.path}")
    zarr_array = cast(Any, zarr.open(tomogram.path, mode="r"))
    shape = tuple(int(axis_size) for axis_size in zarr_array.shape)
    if len(shape) != 3:
        raise ValueError(f"expected tomogram shape (z, y, x), got {shape}")
    return shape


def _validate_picks(picks: Artifact, tomogram: Artifact, manifest: TiltSeriesManifest) -> None:
    if picks.kind is not ArtifactKind.PICKS:
        raise ValueError(f"expected picks artifact, got {picks.kind}")
    if picks.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError("picks are for a different tilt series")
    if picks.parameters.get("input_tomogram_id") != tomogram.id:
        raise ValueError("picks were generated from a different tomogram")
    if not picks.path.is_file():
        raise FileNotFoundError(f"picks STAR file not found: {picks.path}")


def _validate_particle_stack(particles: Artifact, manifest: TiltSeriesManifest) -> None:
    if particles.kind is not ArtifactKind.PARTICLE_STACK:
        raise ValueError(f"expected particle stack artifact, got {particles.kind}")
    if particles.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError("particle stack is for a different tilt series")
    if not particles.path.is_dir():
        raise FileNotFoundError(f"particle stack Zarr not found: {particles.path}")
    metadata_path = particles.parameters.get("metadata_path")
    if not isinstance(metadata_path, str):
        raise ValueError("particle stack artifact is missing metadata_path")
    if not Path(metadata_path).is_file():
        raise FileNotFoundError(f"particle metadata file not found: {metadata_path}")


_PYTOM_STAR_REQUIRED_COLUMNS = (
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
)


def _read_star_loop(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text().splitlines()
    columns: list[str] = []
    in_loop = False
    data_start: int | None = None
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line == "loop_":
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            columns.append(line.split()[0].lstrip("_"))
            continue
        if in_loop and columns and line:
            data_start = line_index
            break
    if data_start is None:
        raise ValueError(f"no STAR data rows found in {path}")
    rows = [line.split() for line in lines[data_start:] if line.strip()]
    return columns, rows


def _read_pytom_star(path: Path) -> list[_PytomStarRow]:
    columns, rows = _read_star_loop(path)
    missing = [name for name in _PYTOM_STAR_REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing required STAR columns: {missing}")
    x_index = columns.index("rlnCoordinateX")
    y_index = columns.index("rlnCoordinateY")
    z_index = columns.index("rlnCoordinateZ")
    rot_index = columns.index("rlnAngleRot")
    tilt_index = columns.index("rlnAngleTilt")
    psi_index = columns.index("rlnAnglePsi")
    score_index = columns.index("rlnLCCmax") if "rlnLCCmax" in columns else None
    return [
        _PytomStarRow(
            index=row_index,
            x_px=float(fields[x_index]),
            y_px=float(fields[y_index]),
            z_px=float(fields[z_index]),
            rot_deg=float(fields[rot_index]),
            tilt_deg=float(fields[tilt_index]),
            psi_deg=float(fields[psi_index]),
            lccmax=float(fields[score_index]) if score_index is not None else None,
        )
        for row_index, fields in enumerate(rows)
    ]


_PARTICLE_STAR_COLUMNS = (
    "particleIndex",
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "shiftXPx",
    "shiftYPx",
    "shiftZPx",
    "rlnLCCmax",
    "rlnClassNumber",
    "rlnRandomSubset",
)


def write_particle_star(path: Path, records: list[ParticleRecord]) -> None:
    lines = ["data_particles", "", "loop_"]
    lines.extend(f"_{name} #{position}" for position, name in enumerate(_PARTICLE_STAR_COLUMNS, 1))
    for record in records:
        lines.append(
            " ".join(
                [
                    str(record.index),
                    f"{record.x_px:.3f}",
                    f"{record.y_px:.3f}",
                    f"{record.z_px:.3f}",
                    f"{record.rot_deg:.6f}",
                    f"{record.tilt_deg:.6f}",
                    f"{record.psi_deg:.6f}",
                    f"{record.shift_x_px:.6f}",
                    f"{record.shift_y_px:.6f}",
                    f"{record.shift_z_px:.6f}",
                    f"{record.score:.6f}" if record.score is not None else "nan",
                    str(record.class_number),
                    str(record.random_subset),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_particle_star(path: Path) -> list[ParticleRecord]:
    columns, rows = _read_star_loop(path)
    missing = [name for name in _PARTICLE_STAR_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    positions = {name: columns.index(name) for name in _PARTICLE_STAR_COLUMNS}
    records = []
    for fields in rows:
        score_text = fields[positions["rlnLCCmax"]]
        records.append(
            ParticleRecord(
                index=int(fields[positions["particleIndex"]]),
                x_px=float(fields[positions["rlnCoordinateX"]]),
                y_px=float(fields[positions["rlnCoordinateY"]]),
                z_px=float(fields[positions["rlnCoordinateZ"]]),
                rot_deg=float(fields[positions["rlnAngleRot"]]),
                tilt_deg=float(fields[positions["rlnAngleTilt"]]),
                psi_deg=float(fields[positions["rlnAnglePsi"]]),
                shift_x_px=float(fields[positions["shiftXPx"]]),
                shift_y_px=float(fields[positions["shiftYPx"]]),
                shift_z_px=float(fields[positions["shiftZPx"]]),
                score=None if score_text == "nan" else float(score_text),
                class_number=int(fields[positions["rlnClassNumber"]]),
                random_subset=int(fields[positions["rlnRandomSubset"]]),
            )
        )
    records.sort(key=lambda record: record.index)
    return records


@dataclass(frozen=True)
class _ExtractionPaths:
    particles_dir: Path
    qc_dir: Path
    particle_stack: Path
    particle_metadata: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.particle_stack, self.particle_metadata, self.report)


def _extraction_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    label: str | None = None,
) -> _ExtractionPaths:
    particles_dir = output_root / "particles" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "particle_extraction"
    prefix = f"{manifest.tilt_series_id}_{backend_name}" + (
        f"_{label}" if label else ""
    )
    return _ExtractionPaths(
        particles_dir=particles_dir,
        qc_dir=qc_dir,
        particle_stack=particles_dir / f"{prefix}_particles.zarr",
        particle_metadata=particles_dir / f"{prefix}_particles.star",
        report=qc_dir / f"{prefix}_qc.json",
    )


@dataclass(frozen=True)
class _AveragingPaths:
    average_dir: Path
    qc_dir: Path
    average_map: Path
    particle_metadata: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.average_map, self.particle_metadata, self.report)


def _averaging_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    label: str | None = None,
) -> _AveragingPaths:
    average_dir = output_root / "averages" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "subtomogram_averaging"
    prefix = f"{manifest.tilt_series_id}_{backend_name}" + (
        f"_{label}" if label else ""
    )
    return _AveragingPaths(
        average_dir=average_dir,
        qc_dir=qc_dir,
        average_map=average_dir / f"{prefix}_average.mrc",
        particle_metadata=average_dir / f"{prefix}_particles.star",
        report=qc_dir / f"{prefix}_qc.json",
    )


def _require_available_paths(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"outputs already exist: {joined}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _remove_path(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _positive_int_parameter(
    context: BackendContext, key: str, *, default: int | None = None
) -> int:
    value = context.parameters.get(key, default)
    if value is None:
        raise TypeError(f"context parameter {key!r} is required")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value <= 0:
        raise ValueError(f"context parameter {key!r} must be greater than zero")
    return value


def _positive_float_parameter(
    context: BackendContext, key: str, *, default: float | None = None
) -> float:
    value = context.parameters.get(key, default)
    if value is None:
        raise TypeError(f"context parameter {key!r} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and greater than zero")
    return normalized


def _optional_float_parameter(context: BackendContext, key: str) -> float | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"context parameter {key!r} must be finite")
    return normalized


def _optional_path_parameter(context: BackendContext, key: str) -> Path | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"context parameter {key!r} must be a path string")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path.resolve()


def _nonnegative_float_parameter(
    context: BackendContext, key: str, *, default: float
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and not negative")
    return normalized


def _int_parameter(context: BackendContext, key: str, *, default: int) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    return value


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value
