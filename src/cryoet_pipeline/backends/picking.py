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
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
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
from cryoet_pipeline.runtime import DevicePreference


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
            _write_tilt_angles(manifest, temporary_tilt_angles)

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
                expected_views=len(manifest.tilt_angles_deg),
            )
            match_command = _match_template_command(
                match_template_executable,
                context,
                defocus_path=defocus_path,
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
                _write_command_log(paths.log, [(match_command, match_result)])
                detail = (match_result.stderr or match_result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"pytom_match_template.py failed with exit code {match_result.returncode}: "
                    f"{detail}; see {paths.log}"
                )

            job_json = _find_output_file(destination_dir, suffix="_job.json")
            extract_command = _extract_candidates_command(
                extract_candidates_executable,
                context,
                job_json_path=job_json,
                tomogram_mask_path=temporary_mask,
            )
            extract_result = self._command_runner(
                extract_command,
                cwd=temporary_root,
                env=os.environ.copy(),
            )
            _write_command_log(
                paths.log,
                [(match_command, match_result), (extract_command, extract_result)],
            )
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
                status=QcStatus.PASS,
            )
            temporary_report.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )

            _replace_output(job_json, paths.job_json)
            _replace_output(particles_star, paths.particles_star)
            _replace_output(temporary_report, paths.report)

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
                "pytom_template_path": str(template),
                "tomogram_mask_coverage": mask_coverage,
                "pytom_template_mask_path": str(template_mask),
                "pytom_job_json_path": str(paths.job_json),
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
    artifacts = [picks, qc]
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
    ) -> None:
        self.picks_dir = picks_dir
        self.qc_dir = qc_dir
        self.particles_star = particles_star
        self.job_json = job_json
        self.report = report
        self.log = log

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.particles_star, self.job_json, self.report, self.log)


@dataclass(frozen=True)
class _PicksStatistics:
    particle_count: int
    minimum_score: float | None
    maximum_score: float | None
    mean_score: float | None


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
        report=qc_dir / f"{prefix}_qc.json",
        log=picks_dir / f"{prefix}.log",
    )


def _validate_input_tomogram(
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[int, int, int]:
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if tomogram.parameters.get("tilt_series_id") != manifest.tilt_series_id:
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


def _write_tilt_angles(manifest: TiltSeriesManifest, output_path: Path) -> None:
    output_path.write_text(
        "".join(f"{tilt_angle:.6f}\n" for tilt_angle in manifest.tilt_angles_deg)
    )


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
    command.extend(_ctf_weighting_arguments(context, defocus_path))
    if _bool_parameter(context, "pytom_spectral_whitening", default=False):
        command.append("--spectral-whitening")
    rng_seed = _optional_positive_int_parameter(context, "pytom_rng_seed")
    if rng_seed is not None:
        command.extend(["--rng-seed", str(rng_seed)])
    search_z = _optional_search_range(context, "pytom_search_z")
    if search_z is not None:
        command.extend(["--search-z", str(search_z[0]), str(search_z[1])])
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


def _extract_candidates_command(
    executable: Path,
    context: BackendContext,
    *,
    job_json_path: Path,
    tomogram_mask_path: Path | None,
) -> list[str]:
    _validate_extraction_options(context)
    command = [
        str(executable),
        "-j",
        str(job_json_path),
        "-n",
        str(_required_positive_int_parameter(context, "pytom_number_of_particles")),
    ]
    cut_off = _optional_positive_float_parameter(context, "pytom_cut_off")
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
    data_start = 0
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
    return _PicksStatistics(
        particle_count=particle_count,
        minimum_score=min(scores),
        maximum_score=max(scores),
        mean_score=sum(scores) / len(scores),
    )


def _write_tomogram_mask(
    segmentation: Artifact,
    output_path: Path,
    *,
    tomogram: Artifact,
    voxel_spacing_angstrom: float,
    dilation_px: int | None,
) -> float:
    """Write the search mask, grown if asked, and return the fraction it admits.

    pytom zeroes the score at every voxel the mask does not cover, with no
    tolerance and no warning, so a mask drawn on a surface admits only
    candidates centred exactly on that surface. Anything lying a fixed distance
    from it — a lattice under a membrane — is discarded silently unless the mask
    is grown to reach it. That is what the margin is for, and it is also what
    makes an approximate segmentation useful: it need only be roughly right
    about where to look, not about what it found.
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


def _write_command_log(
    path: Path,
    command_results: Sequence[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
) -> None:
    sections = []
    for command, result in command_results:
        sections.append(
            f"$ {shlex.join(command)}\n\n"
            f"[stdout]\n{result.stdout or ''}\n"
            f"[stderr]\n{result.stderr or ''}\n"
            f"[exit_code]\n{result.returncode}\n"
        )
    path.write_text("\n\n".join(sections))


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
    if len(entries) != expected_views:
        raise ValueError(
            f"the CTF model covers {len(entries)} views but this tilt series has "
            f"{expected_views}; the defocus would not line up with the tilt angles"
        )
    output_path.write_text(
        "".join(f"{entry.mean_defocus_nm / 1000.0:.6f}\n" for entry in entries)
    )
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

    _required_positive_int_parameter(context, "pytom_number_of_particles")
    if (
        _optional_positive_float_parameter(context, "pytom_number_of_false_positives")
        is not None
        and _optional_positive_float_parameter(context, "pytom_cut_off") is not None
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
