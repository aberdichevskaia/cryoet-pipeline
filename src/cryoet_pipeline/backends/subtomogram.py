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

from cryoet_pipeline.alignment import (
    best_masked_shift_and_correlation,
    prepare_masked_ncc,
    soft_spherical_alignment_mask,
)
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    ParticleExtractionBackend,
    SubtomogramAveragingBackend,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.fsc import FscCurve, compute_fsc_curve
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
from cryoet_pipeline.particles import (
    ParticleRecord,
    read_particle_star,
    read_star_loop,
    write_particle_star,
)
from cryoet_pipeline.rotations import (
    center_particle_volume,
    cone_offsets,
    matrix_to_relion_euler,
    perturbation_matrix,
    relion_euler_to_matrix,
    resample_volume_zyx,
)
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.wedge import (
    WedgeGeometry,
    constrained_shift_and_correlation,
    hermitian_weights_rfft,
    wedge_present_mask_rfft,
)


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


GOLD_STANDARD_FSC_THRESHOLD = 0.143
"""The crossing used for the inter-half FSC, per Rosenthal & Henderson."""

_LOW_PASS_MIN_DRIVING_SHELL = 3
"""A crossing this low says the halves agree nowhere, not that the map is coarse.

The first shell holds a handful of Fourier samples, so its correlation scatters
across roughly [-0.37, +0.56] on independent noise averages — a crossing there is
a draw, and the median draw on noise-dominated data lands exactly there. Driving
the filter from it moved the limit coarser than the operator's start and kept it
there, so below this the schedule holds instead.
"""

_MINIMUM_PARTICLES_PER_HALF = 1
"""An empty half has nothing to compare against, so refinement refuses.

One particle is enough to refuse *nothing*: it can be averaged (into itself),
and the map built from every particle stays meaningful. What one particle per
half cannot support is a resolution, which is a separate claim — see
`_MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION`. Conflating the two would mean
refusing to average a handful of particles because the FSC would be
uninterpretable, which is the wrong thing to refuse.
"""

_MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION = 10
"""Below this the inter-half FSC is reported but flagged as uninterpretable.

PROVISIONAL. A lone low-frequency shell holds few voxels, so with a handful of
particles per half the correlation there scatters widely — measured on this
pipeline's own path, the first shell ranges over roughly [-0.37, +0.56] across
seeds for two independent noise averages of twenty particles each. A crossing
drawn from that scatter is a draw, not an estimate. The number is still recorded
rather than suppressed, because a reader who knows the count can judge it.
"""


@dataclass(frozen=True)
class _HalfSetRefinement:
    """What independently refining the two half-sets produced.

    `half_references` are the two maps that never saw each other's particles, so
    the FSC between them is the gold-standard estimate. `combined_reference` is
    the map built from every particle at its refined orientation — the one to
    deposit — and is exactly the count-weighted mean of the two halves, which is
    why the aligned volumes need not be kept.
    """

    records: list[ParticleRecord]
    half_references: dict[int, torch.Tensor]
    combined_reference: torch.Tensor
    correlations: list[float]
    starting_correlation_mean: float
    low_pass_trace: list[float]
    resolution_trace: list[float | None]
    agreement_trace: list[float]
    half_counts: dict[int, int]


def half_set_indices(records: Sequence[ParticleRecord]) -> dict[int, list[int]]:
    """Group record positions by their persisted half-set assignment.

    The assignment is made once at extraction and written to `rlnRandomSubset`;
    nothing recomputes it, so a particle keeps its half across refinement, across
    a re-average, and across any later selection that carries the STAR rows.
    """

    grouped: dict[int, list[int]] = {1: [], 2: []}
    for record in records:
        if record.random_subset not in grouped:
            raise ValueError(
                f"particle {record.index} has random_subset={record.random_subset}; "
                "the half-set assignment must be 1 or 2"
            )
        grouped[record.random_subset].append(record.index)
    return grouped


def next_low_pass_angstrom(
    curve: FscCurve,
    *,
    current_angstrom: float,
    coarsest_angstrom: float,
    finest_angstrom: float,
    margin_shells: float,
    box_size: int,
    pixel_spacing_angstrom: float,
) -> float:
    """The resolution limit for the next iteration, from the halves' agreement.

    This is the feedback that makes a shared starting reference legitimate: the
    refinement is only ever allowed to use frequencies the two independent halves
    already agree on, plus a small margin so that the estimate has somewhere to
    grow into. Without the margin the limit can never improve, since nothing is
    ever refined beyond where it already is; with too generous a margin the
    refinement reaches into frequencies where only noise agrees, and that noise
    then pushes the next estimate further out — the runaway the split exists to
    prevent.

    Bounded on BOTH sides, which the first version of this was not, and the
    missing side was the one that bit. A crossing at the first or second shell is
    the median outcome on noise-dominated data — measured, not supposed — and
    with only a `finest` bound the rule turned that into a limit COARSER than the
    operator's own starting filter and then sat there: from 60 A to 86 A at box
    48, to 115 A at box 64. A refinement filtered more heavily than it was asked
    to be is worse than one that simply held the start.

    Two guards, and they overlap on purpose. A crossing below
    `_LOW_PASS_MIN_DRIVING_SHELL` carries no information, so the limit holds
    rather than being recomputed from it; and the result can never go coarser
    than `coarsest_angstrom` regardless, as the backstop for whatever the first
    guard fails to catch.

    PROVISIONAL: the margin is a fixed shell count, the driving threshold is the
    reporting threshold, and the rule works in angstrom rather than in integer
    shells. Better versions of all three exist and are recorded in the roadmap;
    this function is the only place any of them would change.

    A curve that never crosses the threshold carries no estimate either, so that
    also holds rather than inventing one.
    """

    if curve.crossing_resolution_angstrom is None:
        return current_angstrom
    shell_frequency = 1.0 / (box_size * pixel_spacing_angstrom)
    crossing_shell = 1.0 / (curve.crossing_resolution_angstrom * shell_frequency)
    if crossing_shell < _LOW_PASS_MIN_DRIVING_SHELL:
        return current_angstrom
    crossing_frequency = 1.0 / curve.crossing_resolution_angstrom
    target = 1.0 / (crossing_frequency + margin_shells * shell_frequency)
    return min(max(target, finest_angstrom), coarsest_angstrom)


def _refine_half_sets_fft(
    *,
    records: list[ParticleRecord],
    raw_volumes: Sequence[Any],
    aligned: torch.Tensor,
    start_reference: torch.Tensor | None,
    num_iterations: int,
    initial_angular_step_deg: float,
    initial_shift_step_px: float,
    pixel_spacing_angstrom: float,
    initial_low_pass_angstrom: float,
    finest_low_pass_angstrom: float,
    low_pass_margin_shells: float,
    device: Any,
) -> _HalfSetRefinement:
    """Refine both half-sets, each against a reference built only from itself.

    The two halves advance in lockstep rather than one after the other, because
    the resolution limit they share is derived from the FSC between them and so
    has to be recomputed between iterations. Independence is about whose
    particles build a reference, not about when the work happens: no particle in
    half 1 is ever compared against anything half 2 contributed to.

    When no starting reference is given each half builds its own from its own
    particles, which is stronger than the standard requires. When one is given
    both start from it, which is standard practice and is sound only because the
    low-pass keeps the shared start away from the frequencies being measured.

    The independence is structural, not total, and the exception is deliberate:
    exactly one number can cross between the halves, the resolution limit, and it
    is derived from their FSC. When that FSC is informative, half 1's filter — and
    so half 1's map — depends on half 2's content, through that scalar and nothing
    else. This is what RELION does, and it is the feedback that stops the
    refinement reaching into frequencies where only noise agrees; a fully sealed
    pair would have nothing to set its limit from.

    The channel is conditional, which is worth stating precisely because the
    obvious phrasing overstates it: on noise-dominated data the crossing lands on
    a shell too low to carry information, the schedule holds, and nothing crosses
    at all. Three tests pin the pieces — the sealed single iteration, the shut
    channel under noise, and the rule that decides which regime applies.
    """

    grouped = half_set_indices(records)
    for subset, indices in grouped.items():
        if len(indices) < _MINIMUM_PARTICLES_PER_HALF:
            raise ValueError(
                f"half-set {subset} is empty, so there is nothing to refine it "
                "against and nothing to compare the other half with. Gold-standard "
                "refinement splits the particles before aligning them, so both "
                "halves have to exist; check rlnRandomSubset in the particle STAR"
            )

    references: dict[int, torch.Tensor] = {}
    for subset, indices in grouped.items():
        if start_reference is not None:
            references[subset] = start_reference
        else:
            references[subset] = torch.stack([aligned[index] for index in indices]).mean(dim=0)

    # Against the reference each particle will actually face: its own half's, and
    # FILTERED, because that is what the search is scored against. Comparing a
    # start measured on an unfiltered reference with a finish measured on a
    # filtered one is not a before and after, it is two different questions — and
    # it reads as a refinement that made things worse.
    starting_references = {
        subset: low_pass_filter_volume(
            reference,
            cutoff_angstrom=initial_low_pass_angstrom,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
        )
        for subset, reference in references.items()
    }
    starting_correlation_mean = float(
        np.mean(
            [
                _normalized_correlation(aligned[index], starting_references[subset])
                for subset, indices in grouped.items()
                for index in indices
            ]
        )
    )

    correlations = [math.nan] * len(records)
    low_pass = initial_low_pass_angstrom
    low_pass_trace: list[float] = []
    resolution_trace: list[float | None] = []
    agreement_trace: list[float] = []

    for iteration in range(num_iterations):
        annealing = 2**iteration
        angular_step_deg = initial_angular_step_deg / annealing
        shift_step_px = initial_shift_step_px / annealing
        low_pass_trace.append(low_pass)

        for subset, indices in grouped.items():
            filtered_reference = low_pass_filter_volume(
                references[subset],
                cutoff_angstrom=low_pass,
                pixel_spacing_angstrom=pixel_spacing_angstrom,
            )
            half_aligned: list[torch.Tensor] = []
            for index in indices:
                candidate = _search_local_alignment(
                    raw_volumes[index],
                    filtered_reference,
                    records[index],
                    angular_step_deg=angular_step_deg,
                    shift_step_px=shift_step_px,
                    device=device,
                )
                records[index] = replace(
                    records[index],
                    rot_deg=candidate.rot_deg,
                    tilt_deg=candidate.tilt_deg,
                    psi_deg=candidate.psi_deg,
                    shift_x_px=candidate.shift_x_px,
                    shift_y_px=candidate.shift_y_px,
                    shift_z_px=candidate.shift_z_px,
                )
                half_aligned.append(candidate.aligned_volume)
                correlations[index] = candidate.correlation
            references[subset] = torch.stack(half_aligned).mean(dim=0)

        curve = compute_fsc_curve(
            references[1],
            references[2],
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            threshold=GOLD_STANDARD_FSC_THRESHOLD,
        )
        resolution_trace.append(curve.crossing_resolution_angstrom)
        # The crossing is quantised to Fourier shells — at box 48 the steps near
        # 36 A are 2.6 A wide — so a refinement improving by less than a shell
        # reads as perfectly flat, which is what the first version of this trace
        # showed. The mean correlation across shells moves continuously with
        # agreement at any frequency, so it can tell "not improving" apart from
        # "improving by less than a shell".
        agreement_trace.append(float(np.mean(curve.correlation)))
        low_pass = next_low_pass_angstrom(
            curve,
            current_angstrom=low_pass,
            coarsest_angstrom=initial_low_pass_angstrom,
            finest_angstrom=finest_low_pass_angstrom,
            margin_shells=low_pass_margin_shells,
            box_size=int(references[1].shape[0]),
            pixel_spacing_angstrom=pixel_spacing_angstrom,
        )

    counts = {subset: len(indices) for subset, indices in grouped.items()}
    total = counts[1] + counts[2]
    combined = (counts[1] * references[1] + counts[2] * references[2]) / total
    return _HalfSetRefinement(
        records=records,
        half_references=references,
        combined_reference=combined,
        correlations=correlations,
        starting_correlation_mean=starting_correlation_mean,
        low_pass_trace=low_pass_trace,
        resolution_trace=resolution_trace,
        agreement_trace=agreement_trace,
        half_counts=counts,
    )


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
        particles: Sequence[Artifact],
        manifests: Sequence[TiltSeriesManifest],
        context: BackendContext,
    ) -> list[Artifact]:
        """Return an average map, refined per-particle metadata, and a QC artifact.

        Takes several particle stacks because averaging over particles from many
        tomograms is the point of averaging; one stack is the same call with a
        list of one, and produces byte-identical ids and paths to before.
        """

        if len(particles) != len(manifests):
            raise ValueError(
                f"{len(particles)} particle stacks were given for {len(manifests)} "
                "manifests; each stack needs the manifest of the series it came from"
            )
        if not particles:
            raise ValueError("averaging needs at least one particle stack")
        for stack, stack_manifest in zip(particles, manifests, strict=True):
            _validate_particle_stack(stack, stack_manifest)
        # Sorted so that the average does not depend on the order the manifests
        # were listed in — neither its contents nor the name it is filed under.
        pairs = sorted(
            zip(particles, manifests, strict=True),
            key=lambda pair: pair[1].tilt_series_id,
        )
        num_iterations = _positive_int_parameter(context, "num_iterations", default=5)
        if _optional_float_parameter(context, "alignment_mask_diameter_angstrom") is not None:
            raise ValueError("alignment_mask_diameter_angstrom is supported only by cone-search")
        if _bool_parameter(context, "missing_wedge", default=False):
            raise ValueError("missing_wedge is supported only by cone-search")
        initial_angular_step_deg = _non_negative_float_parameter(
            context, "initial_angular_step_deg", default=6.0
        )
        initial_shift_step_px = _non_negative_float_parameter(
            context, "initial_shift_step_px", default=2.0
        )
        initial_reference_path = _optional_path_parameter(context, "initial_reference_path")
        initial_reference_correlation_threshold = _positive_float_parameter(
            context, "initial_reference_correlation_threshold", default=0.3
        )
        # The reference is filtered every iteration, which is what lets the two
        # half-sets share a starting map without sharing what is being measured.
        # 60 A is the customary starting limit; the floor is Nyquist, since no
        # limit finer than that is representable in the box at all.
        initial_low_pass_angstrom = _positive_float_parameter(
            context, "initial_low_pass_angstrom", default=60.0
        )
        low_pass_margin_shells = _non_negative_float_parameter(
            context, "low_pass_margin_shells", default=2.0
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        device = torch_device_for_preference(context.device)

        combined = _combine_particle_stacks(pairs, device=device)
        # Refining one class is the step between classifying and having a
        # structure: the assignment was written into the STAR and read by
        # nothing. Only on this backend, because it is the one that refines
        # the half-sets independently — a class picked out of a joint
        # multi-class run should be refined on the gold-standard path.
        supplied = _optional_path_parameter(context, "particle_metadata_path")
        if supplied is not None:
            combined = replace_records_from_star(combined, supplied)
        only_class = _optional_positive_int_parameter(context, "class_number")
        if only_class is not None:
            combined = filter_combined_by_class(combined, only_class)
        scope = averaging_scope(combined.tilt_series_ids)
        paths = _averaging_paths(
            context.output_dir,
            scope,
            backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        particle_count = combined.particle_count
        box_size = combined.box_size
        records = combined.records
        raw_volumes = combined.volumes

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
        start_reference: torch.Tensor | None = None
        if initial_reference_path is not None:
            start_reference = _load_reference_mrc(
                initial_reference_path, expected_box_size=box_size, device=device
            )
            initial_correlations = [
                _normalized_correlation(aligned[index], start_reference)
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

        pixel_spacing_angstrom = pairs[0][0].pixel_spacing_angstrom
        finest_low_pass_angstrom = 2.0 * pixel_spacing_angstrom
        refinement = _refine_half_sets_fft(
            records=records,
            raw_volumes=raw_volumes,
            aligned=aligned,
            start_reference=start_reference,
            num_iterations=num_iterations,
            initial_angular_step_deg=initial_angular_step_deg,
            initial_shift_step_px=initial_shift_step_px,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            initial_low_pass_angstrom=initial_low_pass_angstrom,
            finest_low_pass_angstrom=finest_low_pass_angstrom,
            low_pass_margin_shells=low_pass_margin_shells,
            device=device,
        )
        records = refinement.records
        final_correlations = refinement.correlations
        starting_correlation_mean = refinement.starting_correlation_mean
        reference = refinement.combined_reference

        write_particle_star(paths.particle_metadata, records)
        _write_average_mrc(
            paths.average_map,
            reference.detach().cpu().numpy(),
            pixel_spacing_angstrom=pixel_spacing_angstrom,
        )
        for subset, half_path in paths.half_maps.items():
            _write_average_mrc(
                half_path,
                refinement.half_references[subset].detach().cpu().numpy(),
                pixel_spacing_angstrom=pixel_spacing_angstrom,
            )

        smallest_half = min(refinement.half_counts.values())
        resolution_is_interpretable = (
            smallest_half >= _MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION
        )
        resolution_angstrom = (
            refinement.resolution_trace[-1] if refinement.resolution_trace else None
        )

        warnings: list[str] = []
        status = QcStatus.PASS
        if not resolution_is_interpretable:
            status = QcStatus.WARNING
            warnings.append(
                f"the smaller half-set holds {smallest_half} particles, fewer than "
                f"{_MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION}; the refinement is still "
                "gold-standard but the resolution beside it is drawn from a wide "
                "distribution and should not be read as an estimate"
            )
        if resolution_angstrom is None:
            status = QcStatus.WARNING
            warnings.append(
                "the two half-sets never disagreed enough to cross "
                f"{GOLD_STANDARD_FSC_THRESHOLD:.3f}, so no resolution was measured; with "
                "few particles this means the curve is dominated by the shells they "
                "share rather than that the map is that good"
            )
        mean_final_correlation = float(np.mean(final_correlations))
        minimum_final_correlation = float(np.min(final_correlations))
        if mean_final_correlation <= starting_correlation_mean:
            status = QcStatus.WARNING
            warnings.append(
                f"refinement did not improve the mean correlation: it started at "
                f"{starting_correlation_mean:.3f} and ended at {mean_final_correlation:.3f}. "
                "The search found nothing better than the orientations it was given"
            )
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
            f"{scope}:subtomogram_average:{self.name}"
            f"{variant_suffix(context)}"

        )
        report = SubtomogramAveragingQc(
            tilt_series_ids=combined.tilt_series_ids,
            backend=self.name,
            particle_stack_ids=combined.stack_ids,
            average_id=average_id,
            particle_count=particle_count,
            particle_counts=combined.counts,
            iterations_run=num_iterations,
            starting_correlation_mean=starting_correlation_mean,
            mean_final_correlation=mean_final_correlation,
            minimum_final_correlation=minimum_final_correlation,
            initial_reference_correlation_mean=initial_reference_correlation_mean,
            initial_reference_correlation_fraction_above_threshold=(
                initial_reference_correlation_fraction_above_threshold
            ),
            half_set_counts=dict(refinement.half_counts),
            gold_standard=True,
            resolution_angstrom=resolution_angstrom,
            resolution_trace_angstrom=list(refinement.resolution_trace),
            low_pass_trace_angstrom=list(refinement.low_pass_trace),
            half_set_agreement_trace=list(refinement.agreement_trace),
            resolution_is_interpretable=resolution_is_interpretable,
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        average_artifact = Artifact(
            id=average_id,
            variant=variant_label(context),
            kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
            path=paths.average_map,
            parent_ids=list(combined.stack_ids),
            shape=(box_size, box_size, box_size),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
            parameters={
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
                "particle_stack_ids": list(combined.stack_ids),
                "particle_count": particle_count,
                "iterations_run": num_iterations,
                "metadata_path": str(paths.particle_metadata),
                "qc_path": str(paths.report),
                # Recorded so the resolution can be rechecked by someone who did
                # not run the job: these are the two maps it is a claim about.
                "half_map_paths": {
                    str(subset): str(path) for subset, path in sorted(paths.half_maps.items())
                },
                "gold_standard": True,
                "resolution_angstrom": resolution_angstrom,
            },
            software_versions={"torch": _package_version("torch")},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(paths.average_map)
            + sum(_path_size_bytes(path) for path in paths.half_maps.values())
            + _path_size_bytes(paths.particle_metadata),
        )
        qc_artifact = Artifact(
            id=(
                f"{scope}:qc:subtomogram_averaging:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[average_artifact.id],
            parameters={
                "qc_type": "subtomogram_averaging",
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
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
    - Shifts come from FFT cross-correlations instead of a grid of resamplings;
      the optional masked scorer also evaluates shift-dependent local moments.
    - Several references can be refined at once, with each particle hard-assigned
      to its best-scoring class, which is what makes this classification rather
      than plain averaging.

    Search ranges and their sampling shrink together each iteration, so the
    number of candidates stays constant while the search tightens.
    """

    name = "cone-search"

    def average(
        self,
        particles: Sequence[Artifact],
        manifests: Sequence[TiltSeriesManifest],
        context: BackendContext,
    ) -> list[Artifact]:
        """Return one average per class, refined metadata, and a QC artifact.

        Takes several particle stacks for the same reason the other averaging
        backend does: an average over particles from many tomograms is the point,
        and one stack is this call with a list of one.
        """

        if len(particles) != len(manifests):
            raise ValueError(
                f"{len(particles)} particle stacks were given for {len(manifests)} "
                "manifests; each stack needs the manifest of the series it came from"
            )
        if not particles:
            raise ValueError("averaging needs at least one particle stack")
        for stack, stack_manifest in zip(particles, manifests, strict=True):
            _validate_particle_stack(stack, stack_manifest)
        pairs = sorted(
            zip(particles, manifests, strict=True),
            key=lambda pair: pair[1].tilt_series_id,
        )
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

        combined = _combine_particle_stacks(pairs, device=device)
        scope = averaging_scope(combined.tilt_series_ids)
        paths = _averaging_paths(
            context.output_dir,
            scope,
            backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        particle_count = combined.particle_count
        if num_classes > particle_count:
            raise ValueError(
                f"cannot refine {num_classes} classes from {particle_count} particles"
            )
        box_size = combined.box_size
        pixel_spacing_angstrom = _pixel_spacing_angstrom(pairs[0][0])
        initial_low_pass_angstrom = _positive_float_parameter(
            context, "initial_low_pass_angstrom", default=60.0
        )
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
        mask_diameter = _optional_float_parameter(context, "alignment_mask_diameter_angstrom")
        alignment_mask = None
        if mask_diameter is not None:
            alignment_mask = soft_spherical_alignment_mask(
                box_size,
                pixel_spacing_angstrom=pixel_spacing_angstrom,
                diameter_angstrom=mask_diameter,
                device=device,
            )
        missing_wedge = _bool_parameter(context, "missing_wedge", default=False)
        wedge_tilt_axis_deg = _nonnegative_float_parameter(
            context, "wedge_tilt_axis_deg", default=0.0
        )
        wedge_min_tilt_deg = _optional_float_parameter(context, "wedge_min_tilt_deg")
        wedge_max_tilt_deg = _optional_float_parameter(context, "wedge_max_tilt_deg")
        if not missing_wedge and (
            wedge_min_tilt_deg is not None or wedge_max_tilt_deg is not None
        ):
            raise ValueError(
                "a wedge tilt range was given but missing_wedge is off, so it would be "
                "read by nothing; pass missing_wedge to use it"
            )
        if missing_wedge and alignment_mask is not None:
            # Not a limitation to work around silently. The masked score is a
            # real-space weighted correlation and the constrained score is a
            # restriction of the Fourier support; combining them exactly is not
            # a matter of applying both, and the approximation (wedge-filter the
            # reference, then score inside the mask) is untested here. The mask
            # was also measured NOT to address the wedge bias — mean error 8.28
            # vs 6.12 degrees with it — so there is nothing to be gained by
            # guessing at the combination.
            raise ValueError(
                "missing_wedge and alignment_mask_diameter_angstrom cannot be combined yet: "
                "one restricts the Fourier support and the other weights real space, and "
                "their exact combination is not implemented. Run them separately"
            )
        wedge_geometries: list[WedgeGeometry] | None = None
        hermitian_weights: torch.Tensor | None = None
        if missing_wedge:
            wedge_geometries = _wedge_geometry_per_particle(
                pairs,
                combined,
                min_tilt_override=wedge_min_tilt_deg,
                max_tilt_override=wedge_max_tilt_deg,
                tilt_axis_deg=wedge_tilt_axis_deg,
            )
            hermitian_weights = hermitian_weights_rfft(box_size, device)
        distinct_ranges = sorted(
            {
                (geometry.min_tilt_deg, geometry.max_tilt_deg)
                for geometry in (wedge_geometries or [])
            }
        )
        if alignment_mask is not None:
            alignment_score = "weighted-ncc"
        elif missing_wedge:
            alignment_score = "wedge-constrained-ncc"
        else:
            alignment_score = "whole-box-ncc"
        # Annotated because it is splatted into a pydantic model: without it the
        # inferred value union is compared field by field and every field
        # mismatches the union rather than its own entry.
        score_parameters: dict[str, Any] = {
            "alignment_mask_diameter_angstrom": mask_diameter,
            "alignment_mask_flat_radius_fraction": 0.8 if alignment_mask is not None else None,
            "alignment_score": alignment_score,
            "missing_wedge_weighting": missing_wedge,
            "wedge_tilt_ranges_deg": [list(one) for one in distinct_ranges],
            "wedge_tilt_axis_deg": wedge_tilt_axis_deg if missing_wedge else None,
        }
        records = combined.records
        raw_volumes = combined.volumes
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

        # Against the references the search actually starts from, always. The
        # final correlation on its own is a number without a scale: it says
        # nothing about whether the refinement moved anything. Recording where
        # it began is what turns it into a before and after.
        starting_references = [
            low_pass_filter_volume(
                reference,
                cutoff_angstrom=initial_low_pass_angstrom,
                pixel_spacing_angstrom=pixel_spacing_angstrom,
            )
            for reference in references
        ]
        starting_reference_spectra = (
            [torch.fft.rfftn(one - one.mean()) for one in starting_references]
            if missing_wedge
            else None
        )
        starting_correlation_mean = float(
            np.mean(
                [
                    _correlation_at_current_pose(
                        aligned[record.index],
                        starting_references[record.class_number - 1],
                        (
                            starting_reference_spectra[record.class_number - 1]
                            if starting_reference_spectra is not None
                            else None
                        ),
                        matrix=relion_euler_to_matrix(
                            record.rot_deg, record.tilt_deg, record.psi_deg
                        ),
                        alignment_mask=alignment_mask,
                        geometry=(
                            wedge_geometries[record.index]
                            if wedge_geometries is not None
                            else None
                        ),
                        hermitian=hermitian_weights,
                        box_size=box_size,
                        device=device,
                    )
                    for record in records
                ]
            )
        )

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
            # Filtered every iteration, at a limit held constant. Joint refinement
            # is not exempt from a reference imprinting its own noise on the
            # particles that built it — an unfiltered reference lets every
            # frequency imprint, including the ones a resolution would later be
            # claimed at. Held constant rather than driven by an FSC because this
            # path has no trustworthy one: the halves are refined together, and
            # with several classes there is no defined pairing between half 1's
            # class k and half 2's class k to compute one from.
            filtered_references = [
                low_pass_filter_volume(
                    reference,
                    cutoff_angstrom=initial_low_pass_angstrom,
                    pixel_spacing_angstrom=pixel_spacing_angstrom,
                )
                for reference in references
            ]
            iteration_reference_spectra = (
                [torch.fft.rfftn(one - one.mean()) for one in filtered_references]
                if missing_wedge
                else None
            )
            for record in records:
                wedge_scoring = None
                if wedge_geometries is not None:
                    assert hermitian_weights is not None
                    assert iteration_reference_spectra is not None
                    wedge_scoring = _WedgeScoring(
                        geometry=wedge_geometries[record.index],
                        hermitian=hermitian_weights,
                        reference_spectra=iteration_reference_spectra,
                        box_size=box_size,
                    )
                candidate = _search_multi_reference(
                    raw_volumes[record.index],
                    filtered_references,
                    record,
                    cone=offsets,
                    in_plane_offsets=in_plane_offsets,
                    max_shift_px=iteration_shift_range,
                    device=device,
                    alignment_mask=alignment_mask,
                    wedge=wedge_scoring,
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
                pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
            )
            average_id = _class_average_id(
                scope,
                self.name,
                class_number,
                num_classes=num_classes,
                variant=variant_suffix(context),
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
                    parent_ids=list(combined.stack_ids),
                    shape=(box_size, box_size, box_size),
                    dtype="float32",
                    axis_order=AxisOrder.ZYX,
                    pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
                    parameters={
                        "backend": self.name,
                        "tilt_series_ids": list(combined.tilt_series_ids),
                        "particle_stack_ids": list(combined.stack_ids),
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
                        **score_parameters,
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
        if mean_correlation is not None and mean_correlation <= starting_correlation_mean:
            warnings.append(
                f"refinement did not improve the mean correlation: it started at "
                f"{starting_correlation_mean:.3f} and ended at {mean_correlation:.3f}. "
                "The search found nothing better than the orientations it was given"
            )
        if not missing_wedge:
            status = QcStatus.WARNING
            warnings.append(
                "the alignment score ignores the missing wedge. Measured on this "
                "pipeline's own code with a control: on wedge-limited particles seeded "
                "with exactly correct orientations and no noise, this score moved 100% of "
                "them by a mean of 5.90 deg and dropped the average's agreement with the "
                "known truth from 0.972 to 0.911, while the same loop on the same "
                "particles with the wedge removed moved nothing. Pass missing_wedge to "
                "restrict the score to the frequencies each particle measured"
            )
        if warnings and status is QcStatus.PASS:
            status = QcStatus.WARNING

        report = SubtomogramClassificationQc(
            **score_parameters,
            tilt_series_ids=combined.tilt_series_ids,
            backend=self.name,
            particle_stack_ids=combined.stack_ids,
            particle_count=particle_count,
            particle_counts=combined.counts,
            num_classes=num_classes,
            seed=seed,
            refine_factor=refine_factor,
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
            starting_correlation_mean=starting_correlation_mean,
            mean_final_correlation=mean_correlation,
            minimum_final_correlation=minimum_correlation,
            reassignment_fraction_last_iteration=reassignment_fraction,
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        qc_artifact = Artifact(
            id=(
                f"{scope}:qc:subtomogram_classification:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[average_artifacts[0].id],
            parameters={
                "qc_type": "subtomogram_classification",
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
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


@dataclass(frozen=True)
class _WedgeScoring:
    """Everything the constrained score needs that does not change per candidate.

    `geometry` is the particle's OWN wedge, so this is built per particle: two
    stacks averaged together can come from series with different tilt ranges,
    and scoring one series' particles against the other's wedge would reintroduce
    exactly the mismatch the constraint exists to remove.
    """

    geometry: WedgeGeometry
    hermitian: torch.Tensor
    reference_spectra: Sequence[torch.Tensor]
    box_size: int


def _search_multi_reference(
    raw: torch.Tensor,
    references: Sequence[torch.Tensor],
    record: ParticleRecord,
    *,
    cone: Sequence[tuple[float, float]],
    in_plane_offsets: Sequence[float],
    max_shift_px: float,
    device: torch.device,
    alignment_mask: torch.Tensor | None = None,
    wedge: _WedgeScoring | None = None,
) -> _MultiReferenceCandidate:
    """Score one particle against every reference over the cone, keeping the best.

    With `wedge` the score is restricted to the frequencies this particle
    actually measured, at the candidate's own orientation. Without it the score
    is the whole box, which on this pipeline's own measurement moves 100% of
    exactly-correct orientations off the truth on wedge-limited data — see
    `cryoet_pipeline.wedge` for the numbers and the control.
    """

    current_matrix = relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg)
    current_shift = np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px])

    best_class = record.class_number
    best_matrix = current_matrix
    best_shift = current_shift
    best_correlation = -math.inf
    masked_references = (
        [prepare_masked_ncc(reference, alignment_mask) for reference in references]
        if alignment_mask is not None
        else None
    )
    for polar, azimuth in cone:
        for in_plane in in_plane_offsets:
            candidate_matrix = current_matrix @ perturbation_matrix(polar, azimuth, in_plane)
            rotated = resample_volume_zyx(raw, candidate_matrix, current_shift, device=device)
            if wedge is not None:
                # One mask and one transform per candidate orientation, shared by
                # every class: the support depends on the pose, not on which
                # reference the pose is being compared against.
                present = wedge_present_mask_rfft(
                    wedge.box_size,
                    wedge.geometry,
                    matrix=candidate_matrix,
                    device=device,
                )
                candidate_spectrum = torch.fft.rfftn(rotated - rotated.mean())
            for class_number, reference in enumerate(references, start=1):
                if wedge is not None:
                    delta, correlation = constrained_shift_and_correlation(
                        candidate_spectrum,
                        wedge.reference_spectra[class_number - 1],
                        present,
                        wedge.hermitian,
                        size=wedge.box_size,
                        max_shift_px=max_shift_px,
                    )
                elif masked_references is None:
                    delta, correlation = _best_shift_and_correlation(
                        rotated, reference, max_shift_px=max_shift_px
                    )
                else:
                    delta, correlation = best_masked_shift_and_correlation(
                        rotated,
                        masked_references[class_number - 1],
                        max_shift_px=max_shift_px,
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


def _correlation_at_current_pose(
    aligned_volume: torch.Tensor,
    reference: torch.Tensor,
    reference_spectrum: torch.Tensor | None,
    *,
    matrix: np.ndarray,
    alignment_mask: torch.Tensor | None,
    geometry: WedgeGeometry | None,
    hermitian: torch.Tensor | None,
    box_size: int,
    device: torch.device,
) -> float:
    """The score of the pose a particle already has, by whichever rule is in use.

    Exists so that the recorded start and the recorded finish are the same
    question. They were not once before — the start was measured against an
    unfiltered reference while the search faced a filtered one — and a
    refinement that made things worse read as an improvement. Adding a third
    scoring rule was the point at which a shared helper became cheaper than a
    third branch inlined in the caller.
    """

    if geometry is not None:
        if hermitian is None or reference_spectrum is None:
            raise ValueError("a wedge-constrained score needs the weights and the spectrum")
        _, correlation = constrained_shift_and_correlation(
            torch.fft.rfftn(aligned_volume - aligned_volume.mean()),
            reference_spectrum,
            wedge_present_mask_rfft(box_size, geometry, matrix=matrix, device=device),
            hermitian,
            size=box_size,
            max_shift_px=0,
        )
        return correlation
    if alignment_mask is not None:
        return _best_shift_and_correlation(
            aligned_volume, reference, max_shift_px=0, alignment_mask=alignment_mask
        )[1]
    return _normalized_correlation(aligned_volume, reference)


def _wedge_geometry_per_particle(
    pairs: Sequence[tuple[Artifact, TiltSeriesManifest]],
    combined: _CombinedParticles,
    *,
    min_tilt_override: float | None,
    max_tilt_override: float | None,
    tilt_axis_deg: float,
) -> list[WedgeGeometry]:
    """One wedge per particle, taken from the series the particle came from.

    A record cannot say which tilt-series it belongs to — the combined index is
    positional, as `ParticleRecord.index` states — so the series is recovered
    from `counts`, the same arithmetic `filter_combined_by_class` uses.

    The range defaults to the extremes of the manifest's tilt angles, which is
    the honest default and wrong in one knowable way: views excluded during
    alignment are still in the manifest, so a series that lost its extreme views
    gets a wedge slightly narrower than the truth. The overrides exist for that,
    and for data that entered mid-pipeline without a manifest worth trusting.
    """

    geometries: list[WedgeGeometry] = []
    for (_, manifest), count in zip(pairs, combined.counts, strict=True):
        angles = manifest.tilt_angles_deg
        if not angles and (min_tilt_override is None or max_tilt_override is None):
            raise ValueError(
                f"{manifest.tilt_series_id} records no tilt angles, so its missing wedge "
                "cannot be derived; pass wedge_min_tilt_deg and wedge_max_tilt_deg"
            )
        geometry = WedgeGeometry(
            min_tilt_deg=(
                min_tilt_override if min_tilt_override is not None else float(min(angles))
            ),
            max_tilt_deg=(
                max_tilt_override if max_tilt_override is not None else float(max(angles))
            ),
            tilt_axis_deg=tilt_axis_deg,
        )
        geometries.extend([geometry] * count)
    if len(geometries) != combined.particle_count:
        raise ValueError(
            f"built {len(geometries)} wedges for {combined.particle_count} particles; "
            "the per-series counts do not cover the combined stack"
        )
    return geometries


def _best_shift_and_correlation(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_shift_px: float,
    alignment_mask: torch.Tensor | None = None,
) -> tuple[np.ndarray, float]:
    """Return the best shift within `max_shift_px` and its correlation.

    The whole allowed shift range is evaluated at once through the
    cross-correlation theorem, so widening the range costs nothing extra.
    The returned shift is in reference-frame `(x, y, z)` and is meant to be
    folded into an existing sampling shift as `shift + matrix @ delta`.
    """

    if alignment_mask is not None:
        return best_masked_shift_and_correlation(
            candidate,
            prepare_masked_ncc(reference, alignment_mask),
            max_shift_px=max_shift_px,
        )

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
    variant: str = "",
) -> str:
    """The artifact id for one class average, INCLUDING the run's variant.

    The variant was missing, and its absence made every classification after the
    first collide: `_class_average_path` carries the label, because it derives
    from `_averaging_paths`, whose prefix appends it — so the maps on disk were
    correctly separated per run while their artifact ids were identical. Two runs
    of the same scope and backend therefore produced the same four ids, and the
    second refused with "duplicate artifact id:
    <scope>:subtomogram_average:cone-search:class01".

    Which made `--overwrite` look like the answer, and that is the damaging part:
    it succeeds, both runs' maps sit side by side on disk, and the registry keeps
    only the last one's entries — pointing at the last one's files. The registry
    then claims one class01 average for a scope that has several, and nothing
    says which run the surviving entry describes. That is exactly the confusion
    a registry exists to prevent.

    `variant` is a suffix already formatted by `variant_suffix`, matching how
    every other backend builds an id, and it defaults to empty so that a caller
    without a variant produces the ids that are already on disk.
    """

    if num_classes == 1:
        return f"{tilt_series_id}:subtomogram_average:{backend_name}{variant}"
    return (
        f"{tilt_series_id}:subtomogram_average:{backend_name}"
        f"{variant}:class{class_number:02d}"
    )


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
    particles: Sequence[Artifact],
    manifests: Sequence[TiltSeriesManifest],
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Average particles and register the average-map and QC artifacts."""

    artifacts = backend.average(particles, manifests, context)
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


def _search_deltas(step: float) -> tuple[float, ...]:
    """The offsets a local search tries on one axis: three, or one if the step is zero."""

    return (0.0,) if step == 0.0 else (-step, 0.0, step)


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
    # A zero step means "do not search this axis", so it offers one candidate
    # rather than three identical ones — which is 27x less work per particle, and
    # says what the caller meant instead of arriving at it by coincidence.
    angle_deltas = _search_deltas(angular_step_deg)
    shift_deltas = _search_deltas(shift_step_px)

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
    if not tomogram.covers(manifest.tilt_series_id):
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
    if not picks.covers(manifest.tilt_series_id):
        raise ValueError("picks are for a different tilt series")
    if picks.parameters.get("input_tomogram_id") != tomogram.id:
        raise ValueError("picks were generated from a different tomogram")
    if not picks.path.is_file():
        raise FileNotFoundError(f"picks STAR file not found: {picks.path}")


def _validate_particle_stack(particles: Artifact, manifest: TiltSeriesManifest) -> None:
    if particles.kind is not ArtifactKind.PARTICLE_STACK:
        raise ValueError(f"expected particle stack artifact, got {particles.kind}")
    if not particles.covers(manifest.tilt_series_id):
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


def _read_pytom_star(path: Path) -> list[_PytomStarRow]:
    columns, rows = read_star_loop(path)
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
    half_maps: dict[int, Path]
    """The two independently refined maps, written because they are the evidence.

    A resolution is a claim about these two files agreeing; without them the
    number cannot be rechecked, remasked, or recomputed at another threshold by
    anyone who did not run the job.
    """
    particle_metadata: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.average_map,
            *sorted(self.half_maps.values()),
            self.particle_metadata,
            self.report,
        )


@dataclass(frozen=True)
class _CombinedParticles:
    """Several particle stacks presented as one indexable set of particles.

    The stacks are not concatenated on disk. Records already carry an index into
    their own stack, so offsetting those indices is enough to make everything
    downstream — which only ever sees the flat lists — work unchanged, and no
    gigabytes are copied to achieve it.
    """

    volumes: list[Any]
    records: list[ParticleRecord]
    box_size: int
    tilt_series_ids: list[str]
    stack_ids: list[str]
    counts: list[int]

    @property
    def particle_count(self) -> int:
        return len(self.volumes)


def averaging_scope(tilt_series_ids: Sequence[str]) -> str:
    """The name an average is filed under, given what contributed to it.

    A single series keeps exactly the name it had before this could take several,
    so no existing id or path changes. Several are named after the first with a
    count, which stays short enough for a directory and says at a glance that the
    average is not one series'.

    Callers sort the ids first, so the name does not depend on the order the
    manifests were listed in.
    """

    if not tilt_series_ids:
        raise ValueError("an average needs at least one tilt-series")
    if len(tilt_series_ids) == 1:
        return tilt_series_ids[0]
    return f"{tilt_series_ids[0]}_and_{len(tilt_series_ids) - 1}_more"


def replace_records_from_star(
    combined: _CombinedParticles, metadata_path: Path
) -> _CombinedParticles:
    """Take the per-particle metadata from a supplied STAR instead of the stacks'.

    The step that made `--class-number` usable. Classification writes the class it
    assigned into its OWN output STAR, in `averages/`, and never back into the
    particle stacks — whose extraction STAR says `class_number=1` for everything.
    So a filter reading the stacks could only ever see one class, and the flag
    built for "classify, then refine the good class" could not see the
    classification.

    Supplying the refined STAR carries the orientations with the classes, which is
    what refining a chosen class should start from: the point classification left
    it at, not the point extraction did.

    The indices must line up exactly. A refined STAR from a cross-series average
    holds combined indices over the same stacks in the same order, so it does —
    and if it does not, every subsequent subscript would silently address the
    wrong particle, which is why this is an error rather than a warning.
    """

    records = read_particle_star(metadata_path)
    if len(records) != len(combined.volumes):
        raise ValueError(
            f"{metadata_path} describes {len(records)} particles but the stacks hold "
            f"{len(combined.volumes)}. A supplied STAR has to be the one written for "
            "these stacks, in this order — `average-particles` sorts them by "
            "tilt-series id, so list them the same way it did"
        )
    positions = [record.index for record in records]
    if positions != list(range(len(records))):
        raise ValueError(
            f"{metadata_path} does not index the stacks contiguously from zero, so "
            "its rows cannot be paired with the particles. Use the refined STAR "
            "written by average-particles over these same stacks"
        )
    return replace(combined, records=records)


def filter_combined_by_class(
    combined: _CombinedParticles, class_number: int
) -> _CombinedParticles:
    """Keep only the particles classification assigned to one class.

    The step the workflow needs and did not have: classify into K classes, look
    at the K maps, then average or refine the one that is not junk. Without this
    the class assignment is written into the STAR and read by nothing.

    Per-series counts are recovered by arithmetic rather than from the records,
    because a record cannot say which tilt-series it came from — the combined
    index is positional. The original offsets are known from `counts`, so the
    range each kept index falls in identifies its series. That is the recovery
    the roadmap argues makes a per-particle identity column unnecessary for now,
    and this is it being used rather than asserted.
    """

    boundaries: list[int] = []
    running = 0
    for count in combined.counts:
        running += count
        boundaries.append(running)

    kept = [record for record in combined.records if record.class_number == class_number]
    if not kept:
        available = sorted({record.class_number for record in combined.records})
        raise ValueError(
            f"no particles are in class {class_number}; classification assigned "
            f"{len(combined.records)} particles to classes {available}"
        )

    per_series = [0] * len(combined.counts)
    for record in kept:
        for position, boundary in enumerate(boundaries):
            if record.index < boundary:
                per_series[position] += 1
                break

    volumes = [combined.volumes[record.index] for record in kept]
    records = [replace(record, index=position) for position, record in enumerate(kept)]
    return _CombinedParticles(
        volumes=volumes,
        records=records,
        box_size=combined.box_size,
        tilt_series_ids=list(combined.tilt_series_ids),
        stack_ids=list(combined.stack_ids),
        counts=per_series,
    )


def _combine_particle_stacks(
    pairs: Sequence[tuple[Artifact, TiltSeriesManifest]],
    *,
    device: Any,
) -> _CombinedParticles:
    """Load every stack and present them as one flat set of particles."""

    volumes: list[Any] = []
    records: list[ParticleRecord] = []
    ids: list[str] = []
    stack_ids: list[str] = []
    counts: list[int] = []
    box_size: int | None = None

    for artifact, manifest in pairs:
        array = cast(Any, zarr.open(artifact.path, mode="r"))
        count = int(array.shape[0])
        if count == 0:
            raise ValueError(f"{artifact.id}: particle stack has no particles to average")
        stack_box = int(array.shape[1])
        if box_size is None:
            box_size = stack_box
        elif stack_box != box_size:
            raise ValueError(
                f"{artifact.id} has a box of {stack_box} px but the first stack has "
                f"{box_size} px; particles can only be averaged together at one box "
                "size, so re-extract them with a single --box-size-px"
            )
        metadata_path = Path(cast(str, artifact.parameters["metadata_path"]))
        offset = len(volumes)
        # Centered here, at the one place every averaging and classification run
        # loads its particles, rather than at extraction: the stacks on disk stay
        # exactly as written, so nothing has to be re-extracted.
        volumes.extend(
            center_particle_volume(torch.as_tensor(np.asarray(array[index]), device=device))
            for index in range(count)
        )
        records.extend(
            replace(record, index=record.index + offset)
            for record in read_particle_star(metadata_path)
        )
        ids.append(manifest.tilt_series_id)
        stack_ids.append(artifact.id)
        counts.append(count)

    assert box_size is not None
    if len(records) != len(volumes):
        raise ValueError(
            f"the stacks hold {len(volumes)} particles but their metadata describes "
            f"{len(records)}; each stack's star file must cover its own particles"
        )
    return _CombinedParticles(
        volumes=volumes,
        records=records,
        box_size=box_size,
        tilt_series_ids=ids,
        stack_ids=stack_ids,
        counts=counts,
    )


def _averaging_paths(
    output_root: Path,
    scope: str,
    *,
    backend_name: str,
    label: str | None = None,
) -> _AveragingPaths:
    average_dir = output_root / "averages" / scope
    qc_dir = output_root / "qc" / scope / "subtomogram_averaging"
    prefix = f"{scope}_{backend_name}" + (f"_{label}" if label else "")
    return _AveragingPaths(
        average_dir=average_dir,
        qc_dir=qc_dir,
        average_map=average_dir / f"{prefix}_average.mrc",
        half_maps={
            subset: average_dir / f"{prefix}_half{subset}.mrc" for subset in (1, 2)
        },
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


def _non_negative_float_parameter(
    context: BackendContext, key: str, *, default: float | None = None
) -> float:
    """Read a float parameter for which zero is a setting rather than a mistake.

    A search step of zero means "do not search": average the particles at the
    orientations they arrived with. That is the cleanest test of whether a set of
    picks is real, because it gives the refinement no opportunity to manufacture
    agreement — so it has to be expressible. A low-pass margin of zero likewise
    means "no margin", which is the conservative choice, not an invalid one.

    Separate from `_positive_float_parameter` rather than a flag on it, because
    for most parameters here zero really is a mistake: a sampling of zero degrees
    or a refine factor of zero would loop forever or divide by nothing.
    """

    value = context.parameters.get(key, default)
    if value is None:
        raise TypeError(f"context parameter {key!r} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and not negative")
    return normalized


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


def _optional_positive_int_parameter(context: BackendContext, key: str) -> int | None:
    """Read an optional positive int, or None when it was not asked for.

    The sibling of `_optional_float_parameter`; absent until a caller needed to
    say "only this class" and have "no class filter" be a distinct answer from
    "class zero".
    """

    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value <= 0:
        raise ValueError(f"context parameter {key!r} must be greater than zero")
    return value


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
