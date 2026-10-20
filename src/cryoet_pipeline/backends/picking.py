from __future__ import annotations

import json
import math
import os
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
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
from cryoet_pipeline.backends.command_log import write_command_log
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    PickingBackend,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.imod_defocus import read_imod_defocus_file
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    PicksQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
    TomogramBranch,
)
from cryoet_pipeline.particles import pytom_template_shape
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.score_preview import score_preview_from_mrc, write_score_preview

_PREVIEW_FALLBACK_SLAB_PX = 10
"""Slab depth for the map preview when neither a diameter nor a template box is known."""


class PytomMatchPickBackend:
    """Run pytom-match-pick template matching, optionally masked by a segmentation."""

    name = "pytom-match-pick"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def pick(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        supporting_artifacts: Sequence[Artifact] = (),
    ) -> Artifact:
        """Return a canonical STAR picks artifact."""

        _validate_input_tomogram(tomogram, manifest)
        voxel_spacing = _voxel_spacing(tomogram)
        branch = _tomogram_branch(tomogram)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        segmentation = _optional_segmentation_artifact(supporting_artifacts)
        gpu_ids = _gpu_ids_parameter(context)
        volume_split = _volume_split(context)
        paths = _picking_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            branch=branch,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)

        match_template_executable = _resolve_pytom_executable(
            context,
            "pytom_match_template_executable",
            default_name="pytom_match_template.py",
        )
        extract_candidates_executable = _resolve_pytom_executable(
            context,
            "pytom_extract_candidates_executable",
            default_name="pytom_extract_candidates.py",
        )
        template = _resolve_required_path_parameter(context, "pytom_template")
        template_mask = _resolve_required_path_parameter(context, "pytom_template_mask")

        paths.picks_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-pytom-match-pick-",
            dir=paths.picks_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_tomogram = temporary_root / f"{manifest.tilt_series_id}.mrc"
            temporary_tilt_angles = temporary_root / f"{manifest.tilt_series_id}.rawtlt"
            destination_dir = temporary_root / "match_output"
            temporary_report = temporary_root / "pytom_match_pick_qc.json"

            destination_dir.mkdir()
            _write_zarr_to_mrc(
                tomogram.path,
                temporary_tomogram,
                voxel_spacing_angstrom=voxel_spacing,
            )
            view_count = _write_tilt_angles(manifest, context, temporary_tilt_angles)

            temporary_mask: Path | None = None
            mask_coverage: float | None = None
            if segmentation is not None:
                temporary_mask = temporary_root / f"{manifest.tilt_series_id}_mask.mrc"
                mask_coverage = _write_tomogram_mask(
                    segmentation,
                    temporary_mask,
                    tomogram=tomogram,
                    voxel_spacing_angstrom=voxel_spacing,
                    dilation_px=_optional_positive_int_parameter(
                        context, "pytom_segmentation_dilation_px"
                    ),
                )

            # Extraction options are checked here rather than where the extract
            # command is built: that is after the GPU match, so a typo cost the
            # whole run.
            _validate_extraction_options(context)

            defocus_path = _write_defocus_micrometres(
                context,
                temporary_root / f"{manifest.tilt_series_id}_defocus.txt",
                expected_views=view_count,
                included_z_values=_included_z_values(tomogram),
            )
            dose_path = _write_dose_accumulation(
                manifest,
                temporary_root / f"{manifest.tilt_series_id}_dose.txt",
                expected_views=view_count,
                included_z_values=_included_z_values(tomogram),
            )
            match_command = _match_template_command(
                match_template_executable,
                context,
                defocus_path=defocus_path,
                dose_path=dose_path,
                tomogram_path=temporary_tomogram,
                template_path=template,
                template_mask_path=template_mask,
                tilt_angles_path=temporary_tilt_angles,
                tomogram_mask_path=temporary_mask,
                destination_dir=destination_dir,
                gpu_ids=gpu_ids,
                voxel_spacing_angstrom=voxel_spacing,
            )
            match_result = self._command_runner(
                match_command,
                cwd=temporary_root,
                env=os.environ.copy(),
            )
            if match_result.returncode != 0:
                write_command_log(
                paths.log,
                *[(match_command, match_result)],
            )
                detail = (match_result.stderr or match_result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"pytom_match_template.py failed with exit code {match_result.returncode}: "
                    f"{detail}; see {paths.log}"
                )

            keep_volumes = _bool_parameter(context, "pytom_keep_score_volume", default=False)
            job_json = _find_output_file(destination_dir, suffix="_job.json")
            runs: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]] = [
                (match_command, match_result)
            ]
            requested = _optional_positive_int_parameter(context, "pytom_number_of_particles")
            recommended: int | None = None
            if requested is None:
                # No count given: take twice what pytom recommends, the peaks above
                # its own false-positive cut-off, so that classification has the
                # doubtful half to sort as well as the sure one.
                recommend_command = _extract_candidates_command(
                    extract_candidates_executable,
                    context,
                    job_json_path=job_json,
                    tomogram_mask_path=temporary_mask,
                    number_of_particles=_RECOMMENDATION_CAP,
                    cut_off=None,
                )
                recommend_result = self._command_runner(
                    recommend_command,
                    cwd=temporary_root,
                    env=os.environ.copy(),
                )
                runs.append((recommend_command, recommend_result))
                if recommend_result.returncode != 0:
                    write_command_log(paths.log, *runs)
                    detail = (
                        recommend_result.stderr or recommend_result.stdout or "no process output"
                    ).strip()
                    raise RuntimeError(
                        f"pytom_extract_candidates.py failed with exit code "
                        f"{recommend_result.returncode}: {detail}; see {paths.log}"
                    )
                recommended = _read_star_statistics(
                    _find_output_file(temporary_root, suffix="_particles.star")
                ).particle_count
                if recommended == 0:
                    write_command_log(paths.log, *runs)
                    raise RuntimeError(
                        "pytom found no peak above its false-positive cut-off, so there is "
                        "no count to double; pass --pytom-number-of-particles to take a "
                        "fixed number of the best peaks"
                    )
                number_of_particles = 2 * recommended
                cut_off: float | None = (
                    _optional_non_negative_float_parameter(context, "pytom_cut_off") or 0.0
                )
            else:
                number_of_particles = requested
                cut_off = _optional_non_negative_float_parameter(context, "pytom_cut_off")
            extract_command = _extract_candidates_command(
                extract_candidates_executable,
                context,
                job_json_path=job_json,
                tomogram_mask_path=temporary_mask,
                number_of_particles=number_of_particles,
                cut_off=cut_off,
            )
            extract_result = self._command_runner(
                extract_command,
                cwd=temporary_root,
                env=os.environ.copy(),
            )
            runs.append((extract_command, extract_result))
            write_command_log(paths.log, *runs)
            if extract_result.returncode != 0:
                detail = (
                    extract_result.stderr or extract_result.stdout or "no process output"
                ).strip()
                raise RuntimeError(
                    f"pytom_extract_candidates.py failed with exit code "
                    f"{extract_result.returncode}: {detail}; see {paths.log}"
                )

            particles_star = _find_output_file(temporary_root, suffix="_particles.star")
            statistics = _read_star_statistics(particles_star)
            picks_id = (
                f"{manifest.tilt_series_id}:picks:{branch.value}:{self.name}"
                f"{variant_suffix(context)}"
            )
            uniform_z_sd, spread_warning = _z_spread_verdict(
                statistics,
                tomogram_shape=tomogram.shape,
                template_box_px=_template_box_px(job_json),
            )
            search = _search_statistics(job_json)
            significance = _significance_verdict(statistics, search)
            warnings = [one for one in (spread_warning, significance.warning) if one]
            report = PicksQc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                input_tomogram_id=tomogram.id,
                segmentation_id=segmentation.id if segmentation is not None else None,
                picks_id=picks_id,
                particle_count=statistics.particle_count,
                minimum_score=statistics.minimum_score,
                maximum_score=statistics.maximum_score,
                mean_score=statistics.mean_score,
                coordinate_z_median=statistics.coordinate_z_median,
                coordinate_z_sd=statistics.coordinate_z_sd,
                uniform_z_sd=uniform_z_sd,
                search_sigma=significance.sigma,
                search_space=significance.search_space,
                score_max_sigma=significance.score_max_sigma,
                score_cut_off_sigma=significance.score_cut_off_sigma,
                one_false_positive_sigma=significance.one_false_positive_sigma,
                picks_above_one_false_positive=significance.picks_above_floor,
                excess_over_noise_at_cut_off=significance.excess_at_cut_off,
                requested_particle_count=number_of_particles,
                recommended_particle_count=recommended,
                status=QcStatus.WARNING if warnings else QcStatus.PASS,
                warnings=warnings,
            )
            temporary_report.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )
            # A picture of the correlation map, kept from every match: the
            # volume is a gigabyte and kept only on request, and without one or
            # the other nothing can show afterwards where the peaks were.
            temporary_preview = temporary_root / "scores_preview.npz"
            scores_preview_error = _write_scores_preview(
                destination_dir,
                temporary_preview,
                slab_thickness_px=_preview_slab_thickness_px(context, voxel_spacing, job_json),
            )
            if scores_preview_error is None:
                _replace_output(temporary_preview, paths.scores_preview)
            elif paths.scores_preview.exists():
                # An earlier run's picture beside these picks would show a map
                # they were not read off.
                _remove_path(paths.scores_preview)

            _replace_output(job_json, paths.job_json)
            _replace_output(particles_star, paths.particles_star)
            _replace_output(temporary_report, paths.report)
            # pytom's plot of the score distribution against the noise fit it
            # cut at: the only picture of why the cut-off landed where it did.
            # Kilobytes, so always kept. Optional because pytom only draws it
            # when its plotting dependencies are installed.
            graph = sorted(temporary_root.rglob("*_extraction_graph.svg"))
            if graph:
                _replace_output(graph[0], paths.extraction_graph)
            if keep_volumes:
                # The correlation volume is what the half hour of GPU actually
                # bought; the candidates are one cheap reading of it. Keeping it
                # turns the next question about a cut-off or an exclusion
                # distance from a re-match into seconds.
                for suffix, destination in (
                    ("_scores.mrc", paths.scores_volume),
                    ("_angles.mrc", paths.angles_volume),
                ):
                    _replace_output(_find_output_file(destination_dir, suffix=suffix), destination)
                # Kilobytes, and re-extraction is not comparable without it: the
                # whitening filter sets the scale the scores are on. Keeping the
                # volumes and not this is what made the first attempt at
                # re-extraction useless. Optional because it only exists when
                # spectral whitening was asked for.
                whitening = sorted(destination_dir.rglob("*_whitening_filter.npy"))
                if whitening:
                    _replace_output(whitening[0], paths.whitening_filter)

        return Artifact(
            id=(
                f"{manifest.tilt_series_id}:picks:{branch.value}:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.PICKS,
            path=paths.particles_star,
            parent_ids=[tomogram.id, *(artifact.id for artifact in supporting_artifacts)],
            shape=(statistics.particle_count,),
            pixel_spacing_angstrom=voxel_spacing,
            binning=tomogram.binning,
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "tomogram_branch": branch.value,
                "input_tomogram_id": tomogram.id,
                "segmentation_id": segmentation.id if segmentation is not None else None,
                "picking_method": "template_matching",
                # How many peaks extraction was asked for, and, when the count was
                # left to the rule, what pytom recommended: twice that was asked.
                "pytom_number_of_particles": number_of_particles,
                "pytom_recommended_particles": recommended,
                "pytom_template_path": str(template),
                "tomogram_mask_coverage": mask_coverage,
                # Recorded because its absence is otherwise invisible: a match
                # run without dose weighting looks exactly like one with it, and
                # whether the mdoc carried PriorRecordDose is not something a
                # later reader can recover.
                "per_tilt_weighting": _bool_parameter(
                    context, "pytom_per_tilt_weighting", default=False
                ),
                "dose_weighted": dose_path is not None,
                # Nothing else records it: pytom hands the split to its job
                # runner, not to the job, so the job JSON is silent. And a split
                # is not neutral. pytom averages the pieces' noise variances
                # instead of pooling them and skips a piece the tomogram mask
                # covers nowhere, and the false-positive cut-off and the
                # significance in the QC report are computed from those numbers.
                "pytom_volume_split": list(volume_split),
                # What the report filters its particle pictures with, so that they
                # show the particles as the search saw them.
                "pytom_low_pass_angstrom": _optional_positive_float_parameter(
                    context, "pytom_low_pass"
                ),
                "pytom_template_mask_path": str(template_mask),
                "pytom_job_json_path": str(paths.job_json),
                "pytom_scores_volume_path": (
                    str(paths.scores_volume) if paths.scores_volume.is_file() else None
                ),
                "pytom_angles_volume_path": (
                    str(paths.angles_volume) if paths.angles_volume.is_file() else None
                ),
                "pytom_scores_preview_path": (
                    str(paths.scores_preview) if scores_preview_error is None else None
                ),
                "pytom_scores_preview_error": scores_preview_error,
                "pytom_whitening_filter_path": (
                    str(paths.whitening_filter) if paths.whitening_filter.is_file() else None
                ),
                "pytom_extraction_graph_path": (
                    str(paths.extraction_graph) if paths.extraction_graph.is_file() else None
                ),
                "command_log_path": str(paths.log),
                "qc_path": str(paths.report),
            },
            software_versions={
                "pytom-match-pick": "external",
                "zarr": _package_version("zarr"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=(
                _path_size_bytes(paths.particles_star)
                + _path_size_bytes(paths.job_json)
                + paths.report.stat().st_size
            ),
        )


def _template_match_artifact(
    picks: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    *,
    tomogram: Artifact,
) -> Artifact | None:
    """The search itself, as an artifact, from what the picks already recorded.

    Its path is the job JSON, which is small and always kept; the correlation and
    orientation volumes are recorded as parameters because they are about a
    gigabyte each and keeping them is a choice. So the artifact always exists and
    always says whether it can be re-read: a match without its volumes is still a
    record of what was searched, and `extract-picks` refuses it with a reason
    rather than failing to find a file.

    Returns None when the backend recorded no job JSON, which is how a picking
    backend that is not pytom stays unaffected.
    """

    job_json = picks.parameters.get("pytom_job_json_path")
    if not isinstance(job_json, str):
        return None
    scores = picks.parameters.get("pytom_scores_volume_path")
    angles = picks.parameters.get("pytom_angles_volume_path")
    preview = picks.parameters.get("pytom_scores_preview_path")
    volumes_kept = isinstance(scores, str) and isinstance(angles, str)
    return Artifact(
        id=(
            f"{manifest.tilt_series_id}:template_match:{picks.parameters['backend']}"
            f"{variant_suffix(context)}"
        ),
        variant=variant_label(context),
        kind=ArtifactKind.TEMPLATE_MATCH,
        path=Path(job_json),
        parent_ids=list(picks.parent_ids),
        pixel_spacing_angstrom=picks.pixel_spacing_angstrom,
        binning=tomogram.binning,
        parameters={
            "backend": picks.parameters["backend"],
            "tilt_series_id": manifest.tilt_series_id,
            "tomogram_branch": picks.parameters.get("tomogram_branch"),
            "input_tomogram_id": tomogram.id,
            "scores_volume_path": scores,
            "angles_volume_path": angles,
            "volumes_retained": volumes_kept,
            "scores_preview_path": preview,
        },
        storage_role=StorageRole.CANONICAL,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=sum(
            _path_size_bytes(Path(one))
            for one in (job_json, scores, angles, preview)
            if isinstance(one, str)
        ),
    )


def pick_and_register(
    backend: PickingBackend,
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    supporting_artifacts: Sequence[Artifact] = (),
    replace_existing: bool = False,
) -> list[Artifact]:
    """Pick particles and register picks plus QC artifacts."""

    picks = backend.pick(tomogram, manifest, context, supporting_artifacts=supporting_artifacts)
    match = _template_match_artifact(picks, manifest, context, tomogram=tomogram)
    if match is not None:
        # Picks descend from the search, and the search from the tomogram. Before
        # this the search had no name, so a pick list restated the search's
        # parameters instead of pointing at it — and re-reading the same search at
        # another cut-off had nothing to point at either.
        picks = Artifact.model_validate(
            {**picks.model_dump(mode="json"), "parent_ids": [match.id]}
        )
    qc_path = picks.parameters.get("qc_path")
    if not isinstance(qc_path, str):
        raise ValueError("picking backend must record qc_path")
    qc = Artifact(
        id=(
            f"{manifest.tilt_series_id}:qc:picking:{picks.parameters['backend']}"
            f"{variant_suffix(context)}"
        ),
        variant=variant_label(context),
        kind=ArtifactKind.QC,
        path=Path(qc_path),
        parent_ids=[picks.id],
        parameters={
            "qc_type": "particle_picking",
            "backend": picks.parameters["backend"],
            "tilt_series_id": manifest.tilt_series_id,
            "tomogram_branch": picks.parameters.get("tomogram_branch", "full"),
            "status": QcStatus.PASS.value,
        },
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=Path(qc_path).stat().st_size,
    )
    # The match first: the registry refuses a child whose parent it has not seen,
    # and picks are now the search's child rather than the tomogram's.
    artifacts = [match, picks, qc] if match is not None else [picks, qc]
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _PickingPaths:
    def __init__(
        self,
        *,
        picks_dir: Path,
        qc_dir: Path,
        particles_star: Path,
        job_json: Path,
        report: Path,
        log: Path,
        scores_volume: Path,
        angles_volume: Path,
        scores_preview: Path,
        whitening_filter: Path,
        extraction_graph: Path,
    ) -> None:
        self.picks_dir = picks_dir
        self.qc_dir = qc_dir
        self.particles_star = particles_star
        self.job_json = job_json
        self.report = report
        self.log = log
        self.scores_volume = scores_volume
        self.angles_volume = angles_volume
        self.scores_preview = scores_preview
        self.whitening_filter = whitening_filter
        self.extraction_graph = extraction_graph

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.particles_star, self.job_json, self.report, self.log)

    @property
    def volumes(self) -> tuple[Path, ...]:
        """The correlation volumes, kept only when asked for.

        Not in `outputs`, so an existing pair does not block a re-run and a run
        that does not keep them is unaffected. They are about a gigabyte each per
        series at bin8, which is why keeping them is a choice — but the match
        that produces them costs half an hour of GPU and extracting candidates
        from them costs seconds, so during parameter tuning the choice is
        obvious in the other direction.
        """

        return (self.scores_volume, self.angles_volume)


@dataclass(frozen=True)
class _PicksStatistics:
    particle_count: int
    minimum_score: float | None
    maximum_score: float | None
    mean_score: float | None
    coordinate_z_median: float | None = None
    coordinate_z_sd: float | None = None
    scores: tuple[float, ...] = ()
    """Every pick's score, so significance needs no second pass over the STAR.

    A tuple rather than a list because this dataclass is frozen and the scores
    are read once and never edited.
    """


def _picking_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    branch: TomogramBranch,
    label: str | None = None,
) -> _PickingPaths:
    picks_dir = output_root / "picks" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "picking"
    prefix = f"{manifest.tilt_series_id}_{branch.value}_{backend_name}" + (
        f"_{label}" if label else ""
    )
    return _PickingPaths(
        picks_dir=picks_dir,
        qc_dir=qc_dir,
        particles_star=picks_dir / f"{prefix}_particles.star",
        job_json=picks_dir / f"{prefix}_job.json",
        scores_volume=picks_dir / f"{prefix}_scores.mrc",
        angles_volume=picks_dir / f"{prefix}_angles.mrc",
        scores_preview=qc_dir / f"{prefix}_scores_preview.npz",
        whitening_filter=picks_dir / f"{prefix}_whitening_filter.npy",
        extraction_graph=qc_dir / f"{prefix}_extraction_graph.svg",
        report=qc_dir / f"{prefix}_qc.json",
        log=picks_dir / f"{prefix}.log",
    )


def _validate_input_tomogram(
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[int, int, int]:
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if not tomogram.covers(manifest.tilt_series_id):
        raise ValueError(
            f"tomogram is for {tomogram.parameters.get('tilt_series_id')}, "
            f"expected {manifest.tilt_series_id}"
        )
    if tomogram.axis_order is not AxisOrder.ZYX:
        raise ValueError("picking requires a canonical ZYX tomogram")
    if not tomogram.path.is_dir():
        raise FileNotFoundError(f"canonical tomogram Zarr not found: {tomogram.path}")
    zarr_array = cast(Any, zarr.open(tomogram.path, mode="r"))
    shape = tuple(int(axis_size) for axis_size in zarr_array.shape)
    if len(shape) != 3:
        raise ValueError(f"expected tomogram shape (z, y, x), got {shape}")
    if tomogram.shape is not None and tuple(tomogram.shape) != shape:
        raise ValueError(f"tomogram artifact shape {tomogram.shape} does not match {shape}")
    return shape


def _voxel_spacing(tomogram: Artifact) -> float:
    if tomogram.pixel_spacing_angstrom is None:
        raise ValueError("tomogram voxel spacing is required for pytom-match-pick")
    if not math.isfinite(tomogram.pixel_spacing_angstrom) or tomogram.pixel_spacing_angstrom <= 0:
        raise ValueError("tomogram voxel spacing must be finite and positive")
    return tomogram.pixel_spacing_angstrom


def _tomogram_branch(tomogram: Artifact) -> TomogramBranch:
    value = tomogram.parameters.get("tomogram_branch", TomogramBranch.FULL.value)
    return TomogramBranch(value)


def _optional_segmentation_artifact(
    supporting_artifacts: Sequence[Artifact],
) -> Artifact | None:
    segmentations = [
        artifact for artifact in supporting_artifacts if artifact.kind == ArtifactKind.SEGMENTATION
    ]
    if len(segmentations) > 1:
        raise ValueError(
            f"expected at most one segmentation supporting artifact, found {len(segmentations)}"
        )
    return segmentations[0] if segmentations else None


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
            raise FileNotFoundError(
                f"pytom-match-pick executable not found: {candidate} "
                f"(from --{key.replace('_', '-')})"
            )
        return candidate.resolve()
    resolved = shutil.which(default_name)
    if resolved is not None:
        return Path(resolved)
    raise FileNotFoundError(
        f"{default_name} not found; pass --{key.replace('_', '-')} or add it to PATH"
    )


def _resolve_required_path_parameter(context: BackendContext, key: str) -> Path:
    configured = context.parameters.get(key)
    if not isinstance(configured, (str, Path)):
        raise TypeError(f"context parameter {key!r} must be a path string")
    path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{key} not found: {path}")
    return path.resolve()


def _gpu_ids_parameter(context: BackendContext) -> list[str]:
    if context.device is not DevicePreference.CUDA:
        raise ValueError(
            f"pytom-match-pick requires a CUDA GPU, got device {context.device.value!r}"
        )
    configured = context.parameters.get("pytom_gpu_ids")
    if configured is None:
        raise TypeError("context parameter 'pytom_gpu_ids' is required for pytom-match-pick")
    if not isinstance(configured, str):
        raise TypeError("context parameter 'pytom_gpu_ids' must be a string")
    gpu_ids = [token.strip() for token in configured.split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("context parameter 'pytom_gpu_ids' must list at least one GPU id")
    return gpu_ids


def _write_tilt_angles(
    manifest: TiltSeriesManifest,
    context: BackendContext,
    output_path: Path,
) -> int:
    """Write the tilt angles pytom should model the missing wedge from.

    The manifest lists everything that was collected; fine alignment drops some
    views, so the reconstruction has fewer tilts than the manifest has images.
    Template matching weights frequencies by these angles, so the set has to be
    the one the tomogram was built from. The solved angles are used when the
    caller can point at them, and the manifest only when nothing else is known.
    """

    solved = context.parameters.get("pytom_tilt_angles_path")
    if solved is not None:
        if not isinstance(solved, (str, Path)):
            raise TypeError("context parameter 'pytom_tilt_angles_path' must be a path")
        angles = [
            float(line) for line in Path(solved).read_text().splitlines() if line.strip()
        ]
    else:
        angles = list(manifest.tilt_angles_deg)
    output_path.write_text("".join(f"{angle:.6f}\n" for angle in angles))
    return len(angles)


def _match_template_command(
    executable: Path,
    context: BackendContext,
    *,
    tomogram_path: Path,
    template_path: Path,
    template_mask_path: Path,
    tilt_angles_path: Path,
    tomogram_mask_path: Path | None,
    destination_dir: Path,
    gpu_ids: Sequence[str],
    voxel_spacing_angstrom: float,
    defocus_path: Path | None = None,
    dose_path: Path | None = None,
) -> list[str]:
    command = [
        str(executable),
        "-v",
        str(tomogram_path),
        "-t",
        str(template_path),
        "-m",
        str(template_mask_path),
        "-d",
        str(destination_dir),
        "-a",
        str(tilt_angles_path),
        "--voxel-size-angstrom",
        f"{voxel_spacing_angstrom:.6f}",
        "-g",
        *gpu_ids,
    ]
    particle_diameter = _optional_positive_float_parameter(context, "pytom_particle_diameter")
    angular_search = _optional_positive_float_parameter(context, "pytom_angular_search")
    if (particle_diameter is None) == (angular_search is None):
        raise ValueError(
            "exactly one of 'pytom_particle_diameter' or 'pytom_angular_search' is required"
        )
    if particle_diameter is not None:
        command.extend(["--particle-diameter", f"{particle_diameter:.6f}"])
    if angular_search is not None:
        command.extend(["--angular-search", f"{angular_search:.6f}"])
    z_axis_symmetry = _optional_positive_int_parameter(context, "pytom_z_axis_symmetry")
    if z_axis_symmetry is not None:
        command.extend(["--z-axis-rotational-symmetry", str(z_axis_symmetry)])
    if _bool_parameter(context, "pytom_non_spherical_mask", default=False):
        # pytom's fast path precomputes the mask's normalisation once, which is
        # only valid while the mask is rotation-invariant. A mask that is not
        # makes the scores depend on orientation rather than on fit.
        command.append("--non-spherical-mask")
    if _bool_parameter(context, "pytom_per_tilt_weighting", default=False):
        command.append("--per-tilt-weighting")
        # Dose only enters through per-tilt weighting, so it is passed with it
        # rather than on its own: pytom would have nothing to do with a dose
        # file while weighting the template as a single average.
        if dose_path is not None:
            command.extend(["--dose-accumulation", str(dose_path.resolve())])
    command.extend(_ctf_weighting_arguments(context, defocus_path))
    if _bool_parameter(context, "pytom_spectral_whitening", default=False):
        command.append("--spectral-whitening")
    rng_seed = _optional_positive_int_parameter(context, "pytom_rng_seed")
    if rng_seed is not None:
        command.extend(["--rng-seed", str(rng_seed)])
    search_z = _optional_search_range(context, "pytom_search_z")
    if search_z is not None:
        command.extend(["--search-z", str(search_z[0]), str(search_z[1])])
    volume_split = _volume_split(context)
    if volume_split != _UNSPLIT:
        # Left out at 1 1 1, pytom's own default, so an unsplit search is the
        # same command it always was.
        command.extend(["--volume-split", *(str(pieces) for pieces in volume_split)])
    if tomogram_mask_path is not None:
        command.extend(["--tomogram-mask", str(tomogram_mask_path)])
    low_pass = _optional_positive_float_parameter(context, "pytom_low_pass")
    if low_pass is not None:
        command.extend(["--low-pass", f"{low_pass:.6f}"])
    high_pass = _optional_positive_float_parameter(context, "pytom_high_pass")
    if high_pass is not None:
        command.extend(["--high-pass", f"{high_pass:.6f}"])
    if _bool_parameter(context, "pytom_random_phase_correction", default=False):
        command.append("--random-phase-correction")
    return command


_RECOMMENDATION_CAP = 1_000_000
"""The -n of the pass that asks pytom what it recommends. pytom stops at its own
cut-off, so this only has to exceed any count a tomogram can hold above it."""


def _extract_candidates_command(
    executable: Path,
    context: BackendContext,
    *,
    job_json_path: Path,
    tomogram_mask_path: Path | None,
    number_of_particles: int,
    cut_off: float | None,
) -> list[str]:
    """pytom's extraction: `cut_off` as given, or its false-positive estimate when None."""

    _validate_extraction_options(context)
    command = [
        str(executable),
        "-j",
        str(job_json_path),
        "-n",
        str(number_of_particles),
    ]
    if cut_off is not None:
        command.extend(["-c", f"{cut_off:.6f}"])
    else:
        false_positives = (
            _optional_positive_float_parameter(context, "pytom_number_of_false_positives")
            or 1.0
        )
        command.extend(["--number-of-false-positives", f"{false_positives:.6f}"])
    # Extraction uses the diameter to keep peaks apart: the exclusion distance is
    # half of it. That is a different quantity from the one that sets the angular
    # sampling, and for anything elongated the two differ.
    extraction_diameter = _extraction_particle_diameter(context)
    if extraction_diameter is not None:
        command.extend(["--particle-diameter", f"{extraction_diameter:.6f}"])
    if tomogram_mask_path is not None:
        command.extend(["--tomogram-mask", str(tomogram_mask_path)])
    if _bool_parameter(context, "pytom_tophat_filter", default=False):
        command.append("--tophat-filter")
    return command


def _find_output_file(directory: Path, *, suffix: str) -> Path:
    candidates = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.name.endswith(suffix)
    )
    if not candidates:
        raise RuntimeError(f"pytom-match-pick did not write a {suffix} file in {directory}")
    if len(candidates) > 1:
        joined = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"pytom-match-pick wrote multiple {suffix} files: {joined}")
    return candidates[0]


def _read_star_statistics(star_path: Path) -> _PicksStatistics:
    lines = star_path.read_text().splitlines()
    columns: list[str] = []
    in_loop = False
    # No data line after the columns means pytom found nothing above its cut-off,
    # a result rather than an error. Starting from 0 instead read the whole file,
    # starfile's "# Created by ... package" comment included, as particle rows.
    data_start = len(lines)
    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line == "loop_":
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            columns.append(line.split()[0].lstrip("_"))
            continue
        if in_loop and columns and line:
            data_start = index
            break
    rows = [line.split() for line in lines[data_start:] if line.strip()]
    particle_count = len(rows)
    if "rlnLCCmax" not in columns or particle_count == 0:
        return _PicksStatistics(
            particle_count=particle_count,
            minimum_score=None,
            maximum_score=None,
            mean_score=None,
        )
    score_index = columns.index("rlnLCCmax")
    scores = [float(row[score_index]) for row in rows]
    z_median: float | None = None
    z_sd: float | None = None
    if "rlnCoordinateZ" in columns:
        z_values = np.asarray(
            [float(row[columns.index("rlnCoordinateZ")]) for row in rows],
            dtype=np.float64,
        )
        z_median = float(np.median(z_values))
        z_sd = float(z_values.std())
    return _PicksStatistics(
        particle_count=particle_count,
        minimum_score=min(scores),
        maximum_score=max(scores),
        mean_score=sum(scores) / len(scores),
        scores=tuple(scores),
        coordinate_z_median=z_median,
        coordinate_z_sd=z_sd,
    )


@dataclass(frozen=True)
class _SearchStatistics:
    """What pytom knows about the noise it searched, and never told anyone.

    pytom accumulates the standard deviation of the locally-normalised
    cross-correlation across its rotation loop and records it, with the size of
    the space it searched, in the job JSON. Together they turn a score into a
    significance: how far above the noise a peak sits, and how high a peak the
    noise alone reaches over a space that large.

    Read here rather than in a script afterwards because it is the number that
    says whether a half hour of matching found anything, and a run that does not
    report it leaves the question to be re-asked later from files.
    """

    sigma: float
    search_space: float

    def z(self, score: float) -> float:
        return score / self.sigma

    @property
    def one_false_positive_z(self) -> float:
        """The highest peak a peak-free search over this space gives.

        Where the expected number of Gaussian samples above it is exactly one.
        By bisection so that the definition stays visible and no new dependency
        is needed.
        """

        low, high = 0.0, 40.0
        for _ in range(200):
            middle = (low + high) / 2.0
            if self.search_space * 0.5 * math.erfc(middle / math.sqrt(2.0)) > 1.0:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0

    def expected_above(self, z: float) -> float:
        """How many picks the noise alone puts above `z`."""

        return self.search_space * 0.5 * math.erfc(z / math.sqrt(2.0))


@dataclass(frozen=True)
class _SignificanceVerdict:
    """Every significance field for the report, and the one warning worth raising."""

    sigma: float | None = None
    search_space: float | None = None
    score_max_sigma: float | None = None
    score_cut_off_sigma: float | None = None
    one_false_positive_sigma: float | None = None
    picks_above_floor: int | None = None
    excess_at_cut_off: float | None = None
    warning: str | None = None


def _significance_verdict(
    statistics: _PicksStatistics, search: _SearchStatistics | None
) -> _SignificanceVerdict:
    """Turn scores into significance, or return nothing if pytom did not say.

    The one warning here is scale-free on purpose. Whether the best pick clears
    the height a peak-free search of this size reaches needs no external
    benchmark and no assumption about how another package normalises its
    sigma — a comparison already got wrong once here, between two packages that
    normalise differently. Everything that needs a benchmark is left for a reader
    who has one.
    """

    if search is None or statistics.maximum_score is None:
        return _SignificanceVerdict()

    floor = search.one_false_positive_z
    z_max = search.z(statistics.maximum_score)
    z_cut = (
        search.z(statistics.minimum_score) if statistics.minimum_score is not None else None
    )
    above = (
        sum(1 for score in statistics.scores if search.z(score) > floor)
        if statistics.scores
        else None
    )
    excess = (
        statistics.particle_count - search.expected_above(z_cut) if z_cut is not None else None
    )

    warning = None
    if z_max <= floor:
        warning = (
            f"the best pick is {z_max:.2f} sigma, at or below the {floor:.2f} sigma a "
            "peak-free search over this space reaches on its own, so no single pick "
            "here is a detection. An excess of picks over the noise prediction can "
            "still be real signal; the count is what to read, not the peak height"
        )

    return _SignificanceVerdict(
        sigma=search.sigma,
        search_space=search.search_space,
        score_max_sigma=z_max,
        score_cut_off_sigma=z_cut,
        one_false_positive_sigma=floor,
        picks_above_floor=above,
        excess_at_cut_off=excess,
        warning=warning,
    )


def _search_statistics(job_json: Path) -> _SearchStatistics | None:
    """The noise scale and search size pytom recorded, or None if absent.

    Optional by design: a job JSON from another version may not carry
    `job_stats`, and picking must not start failing over a statistic it did not
    used to read.
    """

    try:
        payload = json.loads(job_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    stats = payload.get("job_stats")
    if not isinstance(stats, dict):
        return None
    sigma = stats.get("std")
    search_space = stats.get("search_space")
    if not isinstance(sigma, (int, float)) or not isinstance(search_space, (int, float)):
        return None
    if sigma <= 0.0 or search_space <= 0.0:
        return None
    return _SearchStatistics(sigma=float(sigma), search_space=float(search_space))


def _template_box_px(job_json: Path) -> int | None:
    """The largest side of the template box pytom actually used, from its job."""

    shape = pytom_template_shape(job_json)
    return max(shape) if shape is not None else None


def _preview_slab_thickness_px(
    context: BackendContext, voxel_spacing_angstrom: float, job_json: Path
) -> int:
    """One particle diameter in voxels, the depth one particle's peak occupies.

    From the diameter matching was given when there is one. A search given an
    angular step instead has no diameter, and half the template box stands in:
    a box holds its particle with room to spare, so half of it is a little
    under one diameter, which errs towards splitting a peak over two slabs
    rather than stacking two particles in one.
    """

    diameter = _optional_positive_float_parameter(context, "pytom_particle_diameter")
    if diameter is not None:
        return max(1, round(diameter / voxel_spacing_angstrom))
    box = _template_box_px(job_json)
    if box is not None:
        return max(1, box // 2)
    return _PREVIEW_FALLBACK_SLAB_PX


def _write_scores_preview(
    match_output: Path, destination: Path, *, slab_thickness_px: int
) -> str | None:
    """Write the correlation map's slab maxima, or say why not instead of failing.

    By now the match has cost its GPU time, so a picture that cannot be made
    must not take the picks down with it. The reason is returned to be recorded,
    so that a missing picture is explained rather than silent.
    """

    candidates = sorted(match_output.rglob("*_scores.mrc"))
    if len(candidates) != 1:
        return f"pytom wrote {len(candidates)} *_scores.mrc files where one was expected"
    try:
        preview = score_preview_from_mrc(candidates[0], slab_thickness_px=slab_thickness_px)
        write_score_preview(preview, destination)
    except (OSError, ValueError, MemoryError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _z_spread_verdict(
    statistics: _PicksStatistics,
    *,
    tomogram_shape: tuple[int, ...] | None,
    template_box_px: int | None,
) -> tuple[float | None, str | None]:
    """Are these picks confined to something, or spread through the volume?

    A specimen occupies part of the reconstructed thickness, so its particles do
    too; noise peaks fill whatever they are allowed to fill. The count cannot
    tell the two apart — a threshold loose enough will always produce plenty —
    and this is the readout that can.

    The comparison needs no reference dataset and no assumption about contrast.
    Uniform picks inside the region where a template box can be centred spread by
    `(thickness - box) / sqrt(12)`, both numbers being known here. The verdict is
    whether the observed spread is *significantly* below that, judged by the
    sampling error of a standard deviation, `sd / sqrt(2n)`, at three sigma —
    rather than by a fraction picked to look reasonable.

    PROVISIONAL. This encodes an expectation that has not been confirmed by
    anyone who reads these tomograms: that a reconstruction places a confined
    specimen in a confined band of Z, so that its particles are confined too.
    Every attempt to locate the lamella in Z directly failed — a variance profile
    is too insensitive at this dose, and a mean profile cannot see a slab at all
    because weighted backprojection multiplies by |k| and removes exactly the
    lowest frequencies a slab boundary lives in. So the premise stands on
    reasoning rather than on measurement, and it is out for review.

    There is one case where uniform is simply correct and this still fires: a
    thickness sized snugly around the specimen leaves it filling the volume, so
    its particles fill it too. That is a *good* reconstruction, and the warning
    names it rather than claiming a fault. Sizing snugly is worth doing, so the
    case is not hypothetical.

    If it turns out that picks legitimately fill a generous thickness as well,
    this warning fires on correct results and should be deleted; the recorded
    numbers are worth keeping either way, since they state a fact rather than an
    interpretation.
    """

    observed = statistics.coordinate_z_sd
    count = statistics.particle_count
    if observed is None or count < 2 or tomogram_shape is None or template_box_px is None:
        return None, None
    if len(tomogram_shape) != 3:
        return None, None

    # Canonical tomograms are (Z, Y, X), so the thickness is the first axis.
    thickness = tomogram_shape[0]
    centreable = thickness - template_box_px
    if centreable <= 0:
        return None, None
    uniform = centreable / math.sqrt(12.0)

    standard_error = observed / math.sqrt(2.0 * count)
    if standard_error <= 0.0:
        return uniform, None
    shortfall = (uniform - observed) / standard_error
    if shortfall >= 3.0:
        return uniform, None
    return uniform, (
        f"picks spread {observed:.1f} px through Z where uniform over the "
        f"{centreable} px a template box can be centred in would spread "
        f"{uniform:.1f} px — {shortfall:.1f} standard errors below uniform, so "
        "not distinguishable from it. Three readings, and this cannot tell them "
        "apart: the picks are noise; or the reconstruction does not place the "
        "specimen in a band of Z; or the thickness is snug around the specimen, "
        "which fills the volume and makes uniform the correct answer. Compare "
        "the reconstruction thickness against the specimen's own before treating "
        "this as a fault"
    )


def _write_tomogram_mask(
    segmentation: Artifact,
    output_path: Path,
    *,
    tomogram: Artifact,
    voxel_spacing_angstrom: float,
    dilation_px: int | None,
) -> float:
    """Write the search mask, grown if asked, and return the fraction it admits."""

    mask = tomogram_search_mask(segmentation, tomogram=tomogram, dilation_px=dilation_px)
    coverage = float(mask.mean())
    if coverage <= 0.0:
        raise ValueError(
            f"segmentation {segmentation.id} admits no voxels, so every candidate "
            "would be discarded"
        )
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = voxel_spacing_angstrom
    return coverage


def tomogram_search_mask(
    segmentation: Artifact, *, tomogram: Artifact, dilation_px: int | None
) -> np.ndarray:
    """The mask pytom searches `tomogram` inside: the segmentation, grown if asked.

    pytom zeroes the score at every voxel the mask does not cover, with no
    tolerance and no warning, so a mask drawn on a surface admits only
    candidates centred exactly on that surface. Anything lying a fixed distance
    from it — a lattice under a membrane — is discarded silently unless the mask
    is grown to reach it. That is what the margin is for, and it is also what
    makes an approximate segmentation useful: it need only be roughly right
    about where to look, not about what it found.

    Public for scripts/null_decoys.py, which builds the mask a search will use
    to show it is the one the real picks were found in.
    """

    source = cast(Any, zarr.open(segmentation.path, mode="r"))
    data = np.asarray(source[:], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("segmentation contains nonfinite values")
    expected = tuple(tomogram.shape) if tomogram.shape is not None else None
    if expected is not None and tuple(data.shape) != expected:
        raise ValueError(
            f"segmentation {segmentation.id} has shape {tuple(data.shape)} but "
            f"{tomogram.id} has {expected}; pytom requires them to match. Segment "
            "the tomogram you are picking in, passing the same --variant"
        )
    mask = (data > 0.0).astype(np.float32)
    if dilation_px:
        mask = _dilate(mask, dilation_px)
    return mask


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a mask by a box of the given radius, one axis at a time."""

    grown = mask
    for axis in range(grown.ndim):
        padding = [
            (radius, radius) if index == axis else (0, 0) for index in range(grown.ndim)
        ]
        padded = np.pad(grown, padding)
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1, axis=axis)
        grown = windows.max(axis=-1)
    return grown


def _write_zarr_to_mrc(
    zarr_path: Path,
    output_path: Path,
    *,
    voxel_spacing_angstrom: float,
) -> None:
    source = cast(Any, zarr.open(zarr_path, mode="r"))
    data = np.asarray(source[:], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("input volume contains nonfinite values")
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_spacing_angstrom



def _replace_output(source: Path, destination: Path) -> None:
    if destination.exists():
        _remove_path(destination)
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"picking outputs already exist: {joined}")


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _optional_non_negative_float_parameter(
    context: BackendContext, key: str
) -> float | None:
    """Read an optional float for which zero is a setting, not a mistake.

    A score cut-off of zero means "take the top N peaks with any positive
    correlation, and apply no significance threshold" — which is what the
    published work on EMPIAR-11538 did: top 800 per tomogram, then classification
    found the real 13.8% among them. A threshold is a decision rule rather than a
    property of the data, and refusing to express "no threshold" is how a real
    minority gets discarded before anything can separate it.

    The CLI already advertised `min=0.0` for this option, so the two sides
    disagreed and a run that asked for zero failed after argument parsing rather
    than at it.
    """

    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and not negative")
    return normalized


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
        raise TypeError(f"context parameter {key!r} must be an integer")
    if value <= 0:
        raise ValueError(f"context parameter {key!r} must be greater than zero")
    return value


def _write_defocus_micrometres(
    context: BackendContext,
    output_path: Path,
    *,
    expected_views: int,
    included_z_values: Sequence[int] | None = None,
) -> Path | None:
    """Restate the fitted defocus in the one form pytom cannot misread.

    pytom accepts an IMOD `.defocus` file, but its version-3 reader averages
    columns five and six as the two astigmatic axes. Our writer stamps version 3
    whether or not astigmatism was fitted, so without it column six is either
    absent — pytom raises — or the phase shift, which pytom averages with the
    defocus and silently halves it. A plain list of micrometres, one per view,
    has no columns to misinterpret.
    """

    source = context.parameters.get("pytom_ctf_defocus_path")
    if source is None:
        return None
    if not isinstance(source, (str, Path)):
        raise TypeError("context parameter 'pytom_ctf_defocus_path' must be a path")
    entries = read_imod_defocus_file(Path(source))
    # The model may cover views the tomogram was not built from: correct-ctf
    # leaves out the ones CTFFIND did not fit. Its rows name their z values.
    model_z = context.parameters.get("pytom_ctf_defocus_z_values")
    if (
        included_z_values is not None
        and isinstance(model_z, list)
        and len(model_z) == len(entries)
    ):
        row_of = {z: row for row, z in enumerate(model_z)}
        missing = [z for z in included_z_values if z not in row_of]
        if missing:
            raise ValueError(
                f"the tomogram was built from z values {missing} that the CTF model "
                "has no defocus for"
            )
        entries = [entries[row_of[z]] for z in included_z_values]
    if len(entries) != expected_views:
        raise ValueError(
            f"the CTF model covers {len(entries)} views but the tomogram was "
            f"reconstructed from {expected_views}; pytom pairs defocus with tilt "
            "angles by position, so the two must describe the same views"
        )
    output_path.write_text(
        "".join(f"{entry.mean_defocus_nm / 1000.0:.6f}\n" for entry in entries)
    )
    return output_path


def _included_z_values(tomogram: Artifact) -> list[int] | None:
    """The views the tomogram was reconstructed from, when it recorded them.

    Reconstruction writes this because fine alignment can drop views, and
    everything downstream that pairs per-view metadata needs to know which ones
    survived. Older tomograms predate the field, and None keeps them working —
    they then rely on the count matching, which it does whenever nothing was
    dropped.
    """

    value = tomogram.parameters.get("included_z_values")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError("tomogram included_z_values must be a list of integers")
    return list(value)


def _write_dose_accumulation(
    manifest: TiltSeriesManifest,
    output_path: Path,
    *,
    expected_views: int,
    included_z_values: Sequence[int] | None = None,
) -> Path | None:
    """Accumulated dose per view, so the template can be weighted per tilt.

    SerialEM records `PriorRecordDose`: the dose the specimen had already taken
    when that exposure was made. It starts at zero for the view acquired first
    and grows in acquisition order, which is what makes it usable here — a
    dose-symmetric series is *written* in tilt order and *acquired* in another,
    so accumulating `ExposureDose` down the file would weight the views inside
    out, calling the freshest ones the most damaged.

    The manifest lists every view collected, while fine alignment may have
    dropped some, so `included_z_values` says which ones the tomogram was built
    from. Without it the two only agree when nothing was dropped, and the count
    check below is all that stands between a silent mispairing and a refusal.

    Returns None when the mdoc did not record it, since dose weighting is an
    improvement rather than a requirement.
    """

    images = list(manifest.images)
    if included_z_values is not None:
        wanted = list(included_z_values)
        by_z = {image.z_value: image for image in images}
        missing = [z for z in wanted if z not in by_z]
        if missing:
            raise ValueError(
                f"{manifest.tilt_series_id}: the tomogram was built from z values "
                f"{missing} that the manifest does not contain"
            )
        images = [by_z[z] for z in wanted]

    raw = [image.metadata.get("PriorRecordDose") for image in images]
    if any(value is None for value in raw):
        return None
    doses: list[float] = []
    for image, value in zip(images, raw, strict=True):
        try:
            doses.append(float(str(value)))
        except ValueError as exc:
            raise ValueError(
                f"{manifest.tilt_series_id}: PriorRecordDose for ZValue "
                f"{image.z_value} is not a number: {value!r}"
            ) from exc
    if len(doses) != expected_views:
        raise ValueError(
            f"the mdoc covers {len(doses)} views but the tomogram was "
            f"reconstructed from {expected_views}; pytom pairs dose with tilt "
            "angles by position, so the two must describe the same views"
        )
    output_path.write_text("".join(f"{dose:.6f}\n" for dose in doses))
    return output_path


def _ctf_weighting_arguments(
    context: BackendContext,
    defocus_path: Path | None,
) -> list[str]:
    """CTF weighting for the template, or nothing.

    pytom builds the weighting from a defocus source plus the three optical
    constants and refuses a partial set. The check is repeated here so the
    refusal happens before the GPU match rather than inside it.
    """

    if defocus_path is None:
        return []
    optics = {
        "--voltage": _optional_positive_float_parameter(context, "pytom_ctf_voltage_kv"),
        "--spherical-aberration": _optional_positive_float_parameter(
            context, "pytom_ctf_spherical_aberration_mm"
        ),
        # A fraction, and zero is a legitimate value for it — unlike the two
        # above, which are positive by nature.
        "--amplitude-contrast": _optional_fraction_parameter(
            context, "pytom_ctf_amplitude_contrast"
        ),
    }
    missing = sorted(flag for flag, value in optics.items() if value is None)
    if missing:
        raise ValueError(
            "CTF weighting needs the voltage, spherical aberration and amplitude "
            f"contrast alongside the defocus; missing {', '.join(missing)}"
        )
    arguments = ["--defocus", str(defocus_path.resolve())]
    for flag, value in optics.items():
        arguments.extend([flag, f"{cast(float, value):.6f}"])
    model = context.parameters.get("pytom_tomogram_ctf_model")
    if model is not None:
        if not isinstance(model, str):
            raise TypeError("context parameter 'pytom_tomogram_ctf_model' must be a string")
        arguments.extend(["--tomogram-ctf-model", model])
    return arguments


def _optional_fraction_parameter(context: BackendContext, key: str) -> float | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"context parameter {key!r} must be a fraction between 0 and 1")
    return normalized


def _optional_search_range(context: BackendContext, key: str) -> tuple[int, int] | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"context parameter {key!r} must be a 'min,max' string")
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError(f"context parameter {key!r} must be 'min,max', got {value!r}")
    try:
        low, high = (int(part.strip()) for part in parts)
    except ValueError as error:
        raise ValueError(f"context parameter {key!r} must be two integers") from error
    if low < 0 or high <= low:
        raise ValueError(f"context parameter {key!r} must satisfy 0 <= min < max")
    return low, high


_UNSPLIT = (1, 1, 1)


def parse_volume_split(value: str) -> tuple[int, int, int]:
    """Read 'x,y,z', how many pieces pytom cuts the search into along each axis."""

    parts = value.split(",")
    if len(parts) != 3:
        raise ValueError(f"volume split must be 'x,y,z', got {value!r}")
    try:
        x, y, z = (int(part.strip()) for part in parts)
    except ValueError as error:
        raise ValueError(f"volume split must be three integers, got {value!r}") from error
    if min(x, y, z) < 1:
        raise ValueError(f"volume split must be positive integers, got {value!r}")
    return x, y, z


def _volume_split(context: BackendContext) -> tuple[int, int, int]:
    """The split pytom searches with: its default, unsplit, unless one was asked for."""

    value = context.parameters.get("pytom_volume_split")
    if value is None:
        return _UNSPLIT
    if not isinstance(value, str):
        raise TypeError("context parameter 'pytom_volume_split' must be an 'x,y,z' string")
    try:
        return parse_volume_split(value)
    except ValueError as error:
        raise ValueError(f"context parameter 'pytom_volume_split': {error}") from error


def _required_positive_int_parameter(context: BackendContext, key: str) -> int:
    value = _optional_positive_int_parameter(context, key)
    if value is None:
        raise TypeError(f"context parameter {key!r} is required")
    return value


def _extraction_particle_diameter(context: BackendContext) -> float | None:
    """The diameter extraction uses to keep peaks apart, in angstrom.

    Extraction removes a region of half this size around each accepted peak, so
    it is a minimum peak separation, not a statement about shape. Matching uses
    a diameter for something else entirely — the angular sampling — and for an
    elongated or extended reference the two numbers are different. They fall
    back to one another so the ordinary case needs only one.
    """

    explicit = _optional_positive_float_parameter(context, "pytom_extract_particle_diameter")
    if explicit is not None:
        return explicit
    return _optional_positive_float_parameter(context, "pytom_particle_diameter")


def _validate_extraction_options(context: BackendContext) -> None:
    """Fail before the GPU match on anything extraction will refuse afterwards."""

    # Optional: without it, twice what pytom recommends.
    _optional_positive_int_parameter(context, "pytom_number_of_particles")
    _optional_non_negative_float_parameter(context, "pytom_cut_off")
    if (
        _optional_positive_float_parameter(context, "pytom_number_of_false_positives")
        is not None
        # Zero is a cut-off too ("no floor"), so it is read as one here.
        and _optional_non_negative_float_parameter(context, "pytom_cut_off") is not None
    ):
        raise ValueError(
            "at most one of 'pytom_number_of_false_positives' or 'pytom_cut_off' may be set"
        )
    if _extraction_particle_diameter(context) is None:
        raise TypeError(
            "extraction needs a particle diameter to keep peaks apart; set "
            "'pytom_extract_particle_diameter', or 'pytom_particle_diameter' if the "
            "same value also drives the angular sampling"
        )


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


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
