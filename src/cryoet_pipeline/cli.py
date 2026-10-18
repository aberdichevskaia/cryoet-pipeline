from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    ImodTiltXcorrAlignmentBackend,
    align_and_register,
)
from cryoet_pipeline.backends.alignment_qc import (
    ImodCoarseAlignmentQcBackend,
    evaluate_coarse_alignment_and_register,
)
from cryoet_pipeline.backends.ctf import (
    Ctffind5CtfEstimationBackend,
    ImodCtfPhaseFlipBackend,
    ImodCtfPlotterEstimationBackend,
    correct_ctf_and_register,
    estimate_ctf_and_register,
)
from cryoet_pipeline.backends.defocus_hint import (
    Ctffind5DefocusHintBackend,
    estimate_defocus_hint_and_register,
)
from cryoet_pipeline.backends.dynamo_search import (
    DynamoAverageBackend,
    DynamoSearchRefinementBackend,
)
from cryoet_pipeline.backends.fiducials import (
    ImodAutofidseedBackend,
    ImodBeadtrackBackend,
    ImodPatchSeedBackend,
    ImodPatchTrackBackend,
    generate_seed_and_register,
    track_fiducials_and_register,
)
from cryoet_pipeline.backends.final_stack import (
    ImodFinalAlignedStackBackend,
    build_final_stack_and_register,
)
from cryoet_pipeline.backends.fine_alignment import (
    ImodTiltalignBackend,
    fine_align_and_register,
)
from cryoet_pipeline.backends.motion import (
    AverageMotionCorrectionBackend,
    MotionCor3MotionCorrectionBackend,
    PhaseCorrelationMotionCorrectionBackend,
    correct_and_register,
)
from cryoet_pipeline.backends.picking import (
    PytomMatchPickBackend,
    parse_volume_split,
    pick_and_register,
)
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    CoarseAlignmentQcBackend,
    CtfCorrectionBackend,
    CtfEstimationBackend,
    DefocusEstimationBackend,
    DenoisingBackend,
    FiducialSeedBackend,
    FiducialTrackingBackend,
    FinalAlignedStackBackend,
    FineAlignmentBackend,
    MotionCorrectionBackend,
    ParticleExtractionBackend,
    PickingBackend,
    ReconstructionBackend,
    SegmentationBackend,
    SubtomogramAveragingBackend,
    TemplatePreparationBackend,
    TiltAlignmentBackend,
)
from cryoet_pipeline.backends.reconstruction import (
    ImodTiltReconstructionBackend,
    reconstruct_and_register,
)
from cryoet_pipeline.backends.restoration import (
    IsoNet2RestorationBackend,
    restore_and_register,
)
from cryoet_pipeline.backends.segmentation import (
    MemBrainSegSegmentationBackend,
    segment_and_register,
)
from cryoet_pipeline.backends.stack import SimpleTiltStackBackend, build_stack_and_register
from cryoet_pipeline.backends.subtomogram import (
    ConeSearchRefinementBackend,
    CubicCropExtractionBackend,
    FftRefinementAveragingBackend,
    average_and_register,
    extract_and_register,
    torch_device_for_preference,
)
from cryoet_pipeline.backends.template import (
    PytomTemplateBackend,
    prepare_template_and_register,
)
from cryoet_pipeline.bypass_external_inputs import (
    import_external_picks_for_extraction,
    import_external_tomogram_for_extraction,
)
from cryoet_pipeline.emdb import fetch_emdb_map, normalize_emdb_id
from cryoet_pipeline.empiar import (
    DEFAULT_EMPIAR_10164_LAYOUT,
    DEFAULT_EMPIAR_ID,
    DEFAULT_TILT_SERIES,
    build_empiar_file_list,
    download_files,
    load_dataset_layout,
)
from cryoet_pipeline.external_stack import import_external_tilt_stack
from cryoet_pipeline.fsc import (
    build_half_set_average,
    compute_and_write_fsc_report,
    load_mask,
    load_mrc_volume,
    plot_fsc_curve,
)
from cryoet_pipeline.models import (
    AlignmentFrame,
    Artifact,
    ArtifactKind,
    ProjectConfig,
    TiltSeriesManifest,
)
from cryoet_pipeline.pick_geometry import (
    AXIS_DIRECTIONS,
    TEMPLATE_AXES,
    MembraneRule,
    filter_picks_by_membrane,
    summary_lines,
)
from cryoet_pipeline.pick_selection import (
    ScoreThreshold,
    default_thresholds_path,
    measure_score_selection,
    score_hints,
    select_picks_by_score,
    selection_paths,
    thresholds_from_json,
    write_thresholds_template,
)
from cryoet_pipeline.picks_visualization import (
    resolve_sphere_radius,
    write_picks_visualization,
)
from cryoet_pipeline.project import initialize_project
from cryoet_pipeline.reports.collect import SeriesArtifacts
from cryoet_pipeline.runtime import normalize_device, resolve_device
from cryoet_pipeline.storage import resolve_storage_policy

app = typer.Typer(help="cryo-ET preprocessing pipeline MVP.")

DEFAULT_DOWNLOAD_TILT_SERIES = list(DEFAULT_TILT_SERIES)


@app.command()
def init(
    frames: Annotated[
        Path,
        typer.Option(help="Directory containing the multiframe movies (MRC or TIFF)."),
    ],
    out: Annotated[Path, typer.Option(help="Project output directory.")],
    mdocs: Annotated[
        Path | None,
        typer.Option(help="Directory containing SerialEM .mdoc files."),
    ] = None,
    frames_pattern: Annotated[
        str | None,
        typer.Option(
            help=(
                "Instead of --mdocs: a regular expression matched against each whole "
                "movie file name, with the named groups series, tilt and angle, e.g. "
                r"'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'."
            ),
        ),
    ] = None,
    pixel_size: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help=(
                "Raw movie pixel size in angstrom, overriding the movie headers. "
                "Omitted, it is read from them: the MRC header, or the resolution "
                "tags SerialEM writes into a TIFF."
            ),
        ),
    ] = None,
    tilt_series: Annotated[
        list[str] | None,
        typer.Option(
            "--tilt-series",
            help=(
                "Tilt-series id to include; repeat for several. Omitted: every series "
                "found, by --frames-pattern or one per <id>.mrc.mdoc."
            ),
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
) -> None:
    """Validate local inputs and write project manifests."""

    if (mdocs is None) == (frames_pattern is None):
        raise typer.BadParameter(
            "give exactly one of --mdocs and --frames-pattern", param_hint="mdocs"
        )
    config = ProjectConfig(
        frames_dir=frames,
        mdocs_dir=mdocs,
        frames_pattern=frames_pattern,
        raw_pixel_spacing_angstrom=pixel_size,
        output_dir=out,
        tilt_series=tilt_series or [],
        device=normalize_device(device),
    )
    result = initialize_project(config)
    if not tilt_series:
        typer.echo(f"found {len(result.tilt_series)} series: {' '.join(result.tilt_series)}")
    typer.echo(f"wrote {result.project_path}")
    for path in result.manifest_paths:
        manifest = TiltSeriesManifest.model_validate_json(path.read_text())
        angles = manifest.tilt_angles_deg
        frames_per_tilt = ", ".join(str(n) for n in sorted(manifest.num_subframes_set))
        # Printed so that a pattern which matched the wrong part of the names
        # shows up here, as a strange angle range or tilt count, before any job.
        typer.echo(
            f"wrote {path}: {manifest.num_tilts} tilts, {min(angles):g} to "
            f"{max(angles):g} deg, {frames_per_tilt} frames per tilt, "
            f"{manifest.raw_pixel_spacing_angstrom:g} A/px"
        )


@app.command("import-tilt-stack")
def import_tilt_stack(
    mdoc: Annotated[
        Path,
        typer.Option(help="Path to the tilt-series' SerialEM .mdoc file."),
    ],
    stack: Annotated[
        Path,
        typer.Option(help="Path to the pre-assembled tilt-series MRC stack."),
    ],
    tilt_series_id: Annotated[
        str,
        typer.Option(help="Tilt-series id, for example TS_0002."),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Output directory for the manifest, stack, and registry."),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite manifest, tilt stack, and registry entries."),
    ] = False,
) -> None:
    """Import an already-assembled tilt-series stack, bypassing motion correction.

    Use this instead of `init` + `prepare-tilt-series` for depositions that do
    not provide individually addressable raw per-tilt movies (unlike
    EMPIAR-10164) -- only a single already-combined MRC stack per tilt-series,
    with one already-corrected image per tilt.
    """

    result = import_external_tilt_stack(
        mdoc_path=mdoc,
        stack_path=stack,
        tilt_series_id=tilt_series_id,
        out=out,
        overwrite=overwrite,
    )
    typer.echo(f"wrote manifest: {result.manifest_path}")
    typer.echo(f"wrote tilt stack: {result.tilt_stack.path}")
    typer.echo(f"wrote tilt angles: {result.tilt_angles.path}")
    typer.echo(f"updated {result.artifact_registry_path}")


@app.command("download-empiar")
def download_empiar(
    out: Annotated[
        Path,
        typer.Option(help="Output root for the downloaded EMPIAR files."),
    ],
    empiar_id: Annotated[
        str,
        typer.Option(help="Numeric EMPIAR deposition id, for example 10164."),
    ] = DEFAULT_EMPIAR_ID,
    tilt_series: Annotated[
        list[str],
        typer.Option(
            "--tilt-series",
            help="Tilt-series ids to download; defaults only apply to EMPIAR-10164.",
        ),
    ] = DEFAULT_DOWNLOAD_TILT_SERIES,
    layout: Annotated[
        Path | None,
        typer.Option(
            help=(
                "YAML file describing this deposition's directory layout; "
                "required for any EMPIAR id other than 10164."
            ),
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print files without downloading.")] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Redownload files even if they exist."),
    ] = False,
) -> None:
    """Download selected tilt-series from one EMPIAR deposition without browser ZIPs."""

    if layout is not None:
        selected_layout = load_dataset_layout(layout)
    elif empiar_id == DEFAULT_EMPIAR_ID:
        selected_layout = DEFAULT_EMPIAR_10164_LAYOUT
    else:
        raise typer.BadParameter(
            f"--layout is required for EMPIAR-{empiar_id} "
            "(only EMPIAR-10164 has a built-in layout)",
            param_hint="layout",
        )

    files = build_empiar_file_list(empiar_id, tilt_series, selected_layout)
    typer.echo(f"selected {len(files)} files")
    for file in files:
        typer.echo(f"{file.relative_path} <- {file.url}")

    if dry_run:
        return

    download_files(files, output_root=out, overwrite=overwrite, progress=typer.echo)


@app.command("correct-motion")
def correct_motion(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for corrected projections.")],
    backend: Annotated[
        str,
        typer.Option(help="Motion-correction backend to use."),
    ] = "average",
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite corrected projection files and registry entries."),
    ] = False,
    storage_policy: Annotated[
        str,
        typer.Option(help="Storage policy: debug, working, or minimal."),
    ] = "debug",
    motioncor3_executable: Annotated[
        Path | None,
        typer.Option(help="MotionCor3 executable path for the motioncor3 backend."),
    ] = None,
    motioncor3_gpu: Annotated[
        int,
        typer.Option(min=0, help="GPU id passed to MotionCor3."),
    ] = 0,
    motioncor3_patch_x: Annotated[
        int,
        typer.Option(min=1, help="MotionCor3 patch count along x."),
    ] = 5,
    motioncor3_patch_y: Annotated[
        int,
        typer.Option(min=1, help="MotionCor3 patch count along y."),
    ] = 5,
    motioncor3_pixel_size: Annotated[
        float | None,
        typer.Option(min=0.0, help="MotionCor3 raw pixel size override in angstrom."),
    ] = None,
    motioncor3_gain: Annotated[
        Path | None,
        typer.Option(help="Optional gain-reference MRC passed to MotionCor3."),
    ] = None,
    motioncor3_gain_rotation: Annotated[
        int,
        typer.Option(min=0, max=3, help="MotionCor3 gain rotation: 0, 1, 2, or 3."),
    ] = 0,
    motioncor3_gain_flip: Annotated[
        int,
        typer.Option(min=0, max=2, help="MotionCor3 gain flip: 0, 1, or 2."),
    ] = 0,
    motioncor3_ftbin: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "MotionCor3 Fourier binning of the sums (-FtBin); 2 brings "
                "super-resolution frames to the physical pixel. The tilt stack and "
                "every step after it work at the movie pixel times this."
            ),
        ),
    ] = 1,
) -> None:
    """Correct multiframe tilt movies and register corrected projections."""

    motion_backend = _motion_backend(backend)
    try:
        resolved_storage_policy = resolve_storage_policy(storage_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="storage-policy") from exc
    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters={
            "overwrite": overwrite,
            "artifact_format": resolved_storage_policy.artifact_format,
            "storage_role": resolved_storage_policy.storage_role,
            "retention_policy": resolved_storage_policy.retention_policy,
            "can_recompute": resolved_storage_policy.can_recompute,
            **_motioncor3_cli_parameters(
                executable=motioncor3_executable,
                gpu=motioncor3_gpu,
                patch_x=motioncor3_patch_x,
                patch_y=motioncor3_patch_y,
                pixel_size=motioncor3_pixel_size,
                gain=motioncor3_gain,
                gain_rotation=motioncor3_gain_rotation,
                gain_flip=motioncor3_gain_flip,
                ftbin=motioncor3_ftbin,
            ),
        },
    )

    artifacts = correct_and_register(
        motion_backend,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    typer.echo(f"wrote {len(artifacts)} corrected projections")
    typer.echo(f"updated {registry}")
    typer.echo(f"storage policy: {resolved_storage_policy.name.value}")
    typer.echo(f"registered size: {artifact_registry.total_size_bytes} bytes")


@app.command("prepare-tilt-series")
def prepare_tilt_series(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for prepared tilt-series data.")],
    motion_backend: Annotated[
        str,
        typer.Option(help="Motion-correction backend to use."),
    ] = "average",
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite prepared files and registry entries."),
    ] = False,
    storage_policy: Annotated[
        str,
        typer.Option(help="Storage policy: debug, working, or minimal."),
    ] = "debug",
    motioncor3_executable: Annotated[
        Path | None,
        typer.Option(help="MotionCor3 executable path for the motioncor3 backend."),
    ] = None,
    motioncor3_gpu: Annotated[
        int,
        typer.Option(min=0, help="GPU id passed to MotionCor3."),
    ] = 0,
    motioncor3_patch_x: Annotated[
        int,
        typer.Option(min=1, help="MotionCor3 patch count along x."),
    ] = 5,
    motioncor3_patch_y: Annotated[
        int,
        typer.Option(min=1, help="MotionCor3 patch count along y."),
    ] = 5,
    motioncor3_pixel_size: Annotated[
        float | None,
        typer.Option(min=0.0, help="MotionCor3 raw pixel size override in angstrom."),
    ] = None,
    motioncor3_gain: Annotated[
        Path | None,
        typer.Option(help="Optional gain-reference MRC passed to MotionCor3."),
    ] = None,
    motioncor3_gain_rotation: Annotated[
        int,
        typer.Option(min=0, max=3, help="MotionCor3 gain rotation: 0, 1, 2, or 3."),
    ] = 0,
    motioncor3_gain_flip: Annotated[
        int,
        typer.Option(min=0, max=2, help="MotionCor3 gain flip: 0, 1, or 2."),
    ] = 0,
    motioncor3_ftbin: Annotated[
        int,
        typer.Option(
            min=1,
            help=(
                "MotionCor3 Fourier binning of the sums (-FtBin); 2 brings "
                "super-resolution frames to the physical pixel. The tilt stack and "
                "every step after it work at the movie pixel times this."
            ),
        ),
    ] = 1,
) -> None:
    """Correct movies and prepare an alignment-ready tilt-series stack."""

    selected_motion_backend = _motion_backend(motion_backend)
    try:
        resolved_storage_policy = resolve_storage_policy(storage_policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="storage-policy") from exc

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters={
            "overwrite": overwrite,
            "artifact_format": resolved_storage_policy.artifact_format,
            "storage_role": resolved_storage_policy.storage_role,
            "retention_policy": resolved_storage_policy.retention_policy,
            "can_recompute": resolved_storage_policy.can_recompute,
            **_motioncor3_cli_parameters(
                executable=motioncor3_executable,
                gpu=motioncor3_gpu,
                patch_x=motioncor3_patch_x,
                patch_y=motioncor3_patch_y,
                pixel_size=motioncor3_pixel_size,
                gain=motioncor3_gain,
                gain_rotation=motioncor3_gain_rotation,
                gain_flip=motioncor3_gain_flip,
                ftbin=motioncor3_ftbin,
            ),
        },
    )

    corrected_artifacts = correct_and_register(
        selected_motion_backend,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    stack_artifacts = build_stack_and_register(
        SimpleTiltStackBackend(),
        corrected_artifacts,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    typer.echo(f"wrote {len(corrected_artifacts)} corrected projections")
    typer.echo(f"wrote {len(stack_artifacts)} tilt-series preparation artifacts")
    typer.echo(f"updated {registry}")
    typer.echo(f"storage policy: {resolved_storage_policy.name.value}")
    typer.echo(f"registered size: {artifact_registry.total_size_bytes} bytes")


@app.command("align-tilt-series")
def align_tilt_series(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for alignment artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="Tilt-alignment backend to use."),
    ] = "imod-xcorr",
    binning: Annotated[
        int,
        typer.Option(min=1, help="Temporary image binning used for coarse alignment."),
    ] = 8,
    min_std_ratio: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Exclude tilts whose standard deviation is below this median ratio.",
        ),
    ] = 0.2,
    tilt_axis_angle: Annotated[
        float | None,
        typer.Option(help="IMOD tilt-axis angle in degrees; defaults to mdoc RotationAngle - 90."),
    ] = None,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite alignment files and the registry entry."),
    ] = False,
) -> None:
    """Coarsely align a prepared tilt series and register normalized transforms."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tilt_stack = _tilt_stack_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _alignment_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "binning": binning,
        "min_std_ratio": min_std_ratio,
    }
    if tilt_axis_angle is not None:
        parameters["tilt_axis_angle_deg"] = tilt_axis_angle
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir

    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    alignment = align_and_register(
        selected_backend,
        tilt_stack,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    typer.echo(f"wrote coarse alignment: {alignment.path}")
    typer.echo(f"wrote IMOD transforms: {alignment.parameters['imod_xf_path']}")
    typer.echo(f"excluded z values: {alignment.parameters['excluded_z_values']}")
    typer.echo(f"updated {registry}")


@app.command("qc-coarse-alignment")
def qc_coarse_alignment(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for QC artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="Coarse-alignment QC backend to use."),
    ] = "imod",
    preview_binning: Annotated[
        int,
        typer.Option(min=1, help="Binning used for the retained aligned preview."),
    ] = 16,
    residual_warning_px: Annotated[
        float,
        typer.Option(min=0.0, help="Residual p95 warning threshold in preview pixels."),
    ] = 2.0,
    residual_fail_px: Annotated[
        float,
        typer.Option(min=0.0, help="Residual maximum failure threshold in preview pixels."),
    ] = 5.0,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite QC files and registry entries."),
    ] = False,
) -> None:
    """Create a reduced prealigned preview and coarse-alignment QC report."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tilt_stack = _tilt_stack_artifact(artifact_registry, tilt_series_manifest)
    alignment = _alignment_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _coarse_alignment_qc_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "preview_binning": preview_binning,
        "residual_warning_px": residual_warning_px,
        "residual_fail_px": residual_fail_px,
    }
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = evaluate_coarse_alignment_and_register(
        selected_backend,
        tilt_stack,
        alignment,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    preview, report = artifacts
    typer.echo(f"wrote coarse-alignment preview: {preview.path}")
    typer.echo(f"wrote coarse-alignment QC: {report.path}")
    typer.echo(f"QC status: {report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("generate-fiducial-seed")
def generate_fiducial_seed(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for fiducial artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="Fiducial seed-generation backend to use."),
    ] = "imod-autofidseed",
    tracking_binning: Annotated[
        int,
        typer.Option(min=1, help="Binning used for fiducial finding and tracking."),
    ] = 4,
    fiducial_diameter_nm: Annotated[
        float,
        typer.Option(min=0.0, help="Nominal gold fiducial diameter in nanometers."),
    ] = 10.0,
    target_beads: Annotated[
        int,
        typer.Option(min=1, help="Desired number of seed fiducials."),
    ] = 150,
    min_seed_fiducials: Annotated[
        int,
        typer.Option(min=1, help="Minimum seed count required for QC pass."),
    ] = 10,
    pre_binning: Annotated[
        int,
        typer.Option(
            min=1,
            help="Binning used for patch tracking (imod-patch-seed backend only).",
        ),
    ] = 8,
    num_patches_x: Annotated[
        int,
        typer.Option(
            min=1,
            help="Patch grid columns (imod-patch-seed backend only).",
        ),
    ] = 5,
    num_patches_y: Annotated[
        int,
        typer.Option(
            min=1,
            help="Patch grid rows (imod-patch-seed backend only).",
        ),
    ] = 5,
    patch_overlap: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Fractional overlap between neighboring patches "
            "(imod-patch-seed backend only).",
        ),
    ] = 0.33,
    patch_margin_px: Annotated[
        int,
        typer.Option(
            min=0,
            help="Unused border on each side of the binned image, in pixels "
            "(imod-patch-seed backend only).",
        ),
    ] = 100,
    raw_pixel_spacing: Annotated[
        float | None,
        typer.Option(
            "--raw-pixel-spacing",
            min=0.0,
            help="Optional calibrated raw pixel spacing in angstrom.",
        ),
    ] = None,
    fiducial_diameter_px: Annotated[
        float | None,
        typer.Option(
            "--fiducial-diameter-px",
            min=0.0,
            help="Optional unbinned bead diameter override in pixels.",
        ),
    ] = None,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite seed, tracking-stack, QC, and registry entries."),
    ] = False,
) -> None:
    """Prepare an aligned tracking stack and generate a fiducial seed model."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tilt_stack = _tilt_stack_artifact(artifact_registry, tilt_series_manifest)
    alignment = _alignment_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _fiducial_seed_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "tracking_binning": tracking_binning,
        "fiducial_diameter_nm": fiducial_diameter_nm,
        "target_beads": target_beads,
        "min_seed_fiducials": min_seed_fiducials,
        "pre_binning": pre_binning,
        "num_patches_x": num_patches_x,
        "num_patches_y": num_patches_y,
        "patch_overlap": patch_overlap,
        "patch_margin_px": patch_margin_px,
    }
    if raw_pixel_spacing is not None:
        parameters["raw_pixel_spacing_angstrom"] = raw_pixel_spacing
    if fiducial_diameter_px is not None:
        parameters["fiducial_diameter_unbinned_px"] = fiducial_diameter_px
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = generate_seed_and_register(
        selected_backend,
        tilt_stack,
        alignment,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    tracking_stack, seed_model, report = artifacts
    typer.echo(f"wrote fiducial tracking stack: {tracking_stack.path}")
    typer.echo(f"wrote fiducial seed model: {seed_model.path}")
    typer.echo(f"wrote fiducial seed QC: {report.path}")
    typer.echo(f"QC status: {report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("track-fiducials")
def track_fiducials(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for fiducial artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="Fiducial-tracking backend to use."),
    ] = "imod-beadtrack",
    rounds: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Beadtrack tracking rounds (default 2). imod-beadtrack only: "
            "patch tracking is a single correlation pass and has no rounds.",
        ),
    ] = None,
    patch_filter_sigma1: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=0.5,
            help="Tiltxcorr high-pass sigma in cycles/pixel, above zero "
            "(default 0.03). imod-patch-track only.",
        ),
    ] = None,
    patch_filter_radius2: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=0.5,
            help="Tiltxcorr low-pass cutoff radius in cycles/pixel, above zero "
            "(default 0.25). imod-patch-track only.",
        ),
    ] = None,
    patch_filter_sigma2: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=0.5,
            help="Tiltxcorr low-pass falloff sigma in cycles/pixel, above zero "
            "(default 0.05). imod-patch-track only.",
        ),
    ] = None,
    min_tracked_fiducials: Annotated[
        int,
        typer.Option(min=1, help="Minimum tracked fiducials required for QC pass."),
    ] = 10,
    coverage_warning: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Tracking coverage warning threshold."),
    ] = 0.8,
    coverage_failure: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Tracking coverage failure threshold."),
    ] = 0.5,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite fiducial model, QC, and registry entries."),
    ] = False,
) -> None:
    """Track the generated fiducial seed model through the tilt series."""

    # Check the option/backend combination before touching the filesystem, so a
    # flag the chosen backend cannot read is reported at once rather than after
    # the manifest and registry have been parsed.
    selected_backend = _fiducial_tracking_backend(backend)
    backend_only_options = {
        "--rounds": ("rounds", rounds),
        "--patch-filter-sigma1": ("filter_sigma1", patch_filter_sigma1),
        "--patch-filter-radius2": ("filter_radius2", patch_filter_radius2),
        "--patch-filter-sigma2": ("filter_sigma2", patch_filter_sigma2),
    }
    _reject_foreign_tracking_options(selected_backend.name, backend_only_options)

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tracking_stack = _fiducial_tracking_stack_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
    )
    seed_model = _fiducial_seed_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
    )
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "min_tracked_fiducials": min_tracked_fiducials,
        "coverage_warning": coverage_warning,
        "coverage_failure": coverage_failure,
    }
    # Left out when not given, so each backend's own default applies rather than
    # this command deciding one on its behalf.
    for key, value in backend_only_options.values():
        if value is not None:
            parameters[key] = value
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = track_fiducials_and_register(
        selected_backend,
        tracking_stack,
        seed_model,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    fiducial_model, report = artifacts
    typer.echo(f"wrote tracked fiducial model: {fiducial_model.path}")
    typer.echo(f"wrote fiducial tracking QC: {report.path}")
    typer.echo(f"QC status: {report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("fine-align-tilt-series")
def fine_align_tilt_series(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for alignment artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="Fine-alignment backend to use."),
    ] = "imod-tiltalign",
    residual_warning_nm: Annotated[
        float,
        typer.Option(min=0.0, help="Mean residual warning threshold in nanometers."),
    ] = 0.8,
    residual_failure_nm: Annotated[
        float,
        typer.Option(min=0.0, help="Mean residual failure threshold in nanometers."),
    ] = 1.5,
    residual_max_warning_px: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Maximum point-residual warning threshold in tracking pixels.",
        ),
    ] = 5.0,
    residual_max_failure_px: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Maximum point-residual failure threshold in tracking pixels.",
        ),
    ] = 20.0,
    cross_validate: Annotated[
        bool,
        typer.Option(help="Run Tiltalign global leave-out cross-validation."),
    ] = True,
    robust_fitting: Annotated[
        bool,
        typer.Option(help="Use Tiltalign robust fitting to down-weight outliers."),
    ] = True,
    auto_prune_outliers: Annotated[
        bool,
        typer.Option(help="Prune failure-level fiducial points and rerun Tiltalign."),
    ] = True,
    max_pruned_fraction: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Maximum fraction of fiducial points removable in one run.",
        ),
    ] = 0.02,
    max_positioning_rounds: Annotated[
        int,
        typer.Option(
            min=1,
            help="Maximum Tiltalign reruns for surface-positioning convergence.",
        ),
    ] = 3,
    surfaces_to_analyze: Annotated[
        int,
        typer.Option(
            min=1,
            max=2,
            help="Planes Tiltalign fits to the fiducials when recommending "
            "specimen positioning. Two suits gold beads, which sit on both faces "
            "of a section. Use one for patch tracking, whose points are coplanar: "
            "a two-plane fit then sorts them onto a single surface plus a stray "
            "point and reports that point's offset as the thickness. QC warns "
            "when the split degenerates, whichever value was used.",
        ),
    ] = 2,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite fine-alignment, QC, and registry entries."),
    ] = False,
) -> None:
    """Solve fiducial fine alignment and compose final global transforms."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tracking_stack = _fiducial_tracking_stack_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
    )
    fiducial_model = _fiducial_model_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
    )
    selected_backend = _fine_alignment_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "residual_warning_nm": residual_warning_nm,
        "residual_failure_nm": residual_failure_nm,
        "residual_max_warning_tracking_px": residual_max_warning_px,
        "residual_max_failure_tracking_px": residual_max_failure_px,
        "cross_validate": cross_validate,
        "robust_fitting": robust_fitting,
        "auto_prune_outliers": auto_prune_outliers,
        "max_pruned_fraction": max_pruned_fraction,
        "max_positioning_rounds": max_positioning_rounds,
        "surfaces_to_analyze": surfaces_to_analyze,
    }
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = fine_align_and_register(
        selected_backend,
        tracking_stack,
        fiducial_model,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    alignment, report = artifacts
    typer.echo(f"wrote fine alignment: {alignment.path}")
    typer.echo(f"wrote final IMOD transforms: {alignment.parameters['imod_xf_path']}")
    typer.echo(f"wrote fine-alignment QC: {report.path}")
    typer.echo(f"QC status: {report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("build-final-aligned-stack")
def build_final_aligned_stack(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for aligned data.")],
    backend: Annotated[
        str,
        typer.Option(help="Final aligned-stack backend to use."),
    ] = "imod-newstack",
    output_binning: Annotated[
        int,
        typer.Option(min=1, help="Binning for the final aligned stack."),
    ] = 8,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite final aligned-stack and registry entry."),
    ] = False,
) -> None:
    """Apply fine transforms and build a reconstruction-ready tilt stack."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tilt_stack = _tilt_stack_artifact(artifact_registry, tilt_series_manifest)
    fine_alignment = _fine_alignment_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
    )
    selected_backend = _final_aligned_stack_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "output_binning": output_binning,
    }
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifact = build_final_stack_and_register(
        selected_backend,
        tilt_stack,
        fine_alignment,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    typer.echo(f"wrote final aligned stack: {artifact.path}")
    typer.echo(f"wrote solved tilt angles: {artifact.parameters['tilt_file_path']}")
    typer.echo(f"updated {registry}")


@app.command("estimate-defocus-hint")
def estimate_defocus_hint(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for the defocus hint.")],
    ctf_voltage_kv: Annotated[
        float,
        typer.Option(min=0.0, help="Microscope accelerating voltage in kV."),
    ],
    ctf_spherical_aberration_mm: Annotated[
        float,
        typer.Option(min=0.0, help="Objective spherical aberration (Cs) in mm."),
    ],
    backend: Annotated[
        str,
        typer.Option(help="Defocus-hint backend to use."),
    ] = "ctffind5",
    ctf_amplitude_contrast: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Amplitude contrast fraction."),
    ] = 0.07,
    ctffind_executable: Annotated[
        Path | None,
        typer.Option(help="CTFFIND executable path when it is not on PATH."),
    ] = None,
    defocus_hint_box_size: Annotated[
        int,
        typer.Option(min=1, help="CTFFIND power-spectrum box size, pixels."),
    ] = 512,
    defocus_hint_min_resolution_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Minimum resolution to fit, angstrom."),
    ] = 30.0,
    defocus_hint_max_resolution_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Maximum resolution to fit, angstrom."),
    ] = 5.0,
    defocus_hint_min_defocus_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Minimum defocus to search, angstrom."),
    ] = 5000.0,
    defocus_hint_max_defocus_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Maximum defocus to search, angstrom."),
    ] = 80000.0,
    defocus_hint_step_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Defocus search step, angstrom."),
    ] = 500.0,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite defocus hint, QC, and registry entries."),
    ] = False,
) -> None:
    """Estimate a single-image defocus hint to seed CTF correction."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    corrected_projection = _zero_tilt_corrected_projection_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    selected_backend = _defocus_estimation_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "ctf_voltage_kv": ctf_voltage_kv,
        "ctf_spherical_aberration_mm": ctf_spherical_aberration_mm,
        "ctf_amplitude_contrast": ctf_amplitude_contrast,
        "defocus_hint_box_size": defocus_hint_box_size,
        "defocus_hint_min_resolution_angstrom": defocus_hint_min_resolution_angstrom,
        "defocus_hint_max_resolution_angstrom": defocus_hint_max_resolution_angstrom,
        "defocus_hint_min_defocus_angstrom": defocus_hint_min_defocus_angstrom,
        "defocus_hint_max_defocus_angstrom": defocus_hint_max_defocus_angstrom,
        "defocus_hint_step_angstrom": defocus_hint_step_angstrom,
    }
    if ctffind_executable is not None:
        parameters["ctffind_executable"] = ctffind_executable
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = estimate_defocus_hint_and_register(
        selected_backend,
        corrected_projection,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    defocus_hint, qc_report = artifacts
    typer.echo(f"wrote defocus hint: {defocus_hint.path}")
    typer.echo(f"defocus_um: {defocus_hint.parameters['defocus_um']}")
    typer.echo(f"wrote defocus hint QC: {qc_report.path}")
    typer.echo(f"updated {registry}")


@app.command("estimate-ctf")
def estimate_ctf(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for CTF artifacts.")],
    ctf_voltage_kv: Annotated[
        float,
        typer.Option(min=0.0, help="Microscope accelerating voltage in kV."),
    ],
    ctf_spherical_aberration_mm: Annotated[
        float,
        typer.Option(min=0.0, help="Objective spherical aberration (Cs) in mm."),
    ],
    backend: Annotated[
        str,
        typer.Option(help="CTF estimation backend: ctffind5 or imod-ctfplotter."),
    ] = "ctffind5",
    ctf_amplitude_contrast: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="Amplitude contrast fraction."),
    ] = 0.07,
    ctf_expected_defocus_um: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Expected defocus, um. For ctffind5 this replaces the zero-tilt "
            "first pass; else a registered defocus hint or the manifest is used.",
        ),
    ] = None,
    ctffind_box_size: Annotated[
        int,
        typer.Option(min=1, help="CTFFIND power-spectrum box size, pixels."),
    ] = 512,
    ctffind_min_resolution_angstrom: Annotated[
        float | None,
        typer.Option(min=0.0, help="CTFFIND low-resolution fit limit, A (default 30)."),
    ] = None,
    ctffind_max_resolution_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="CTFFIND high-resolution fit limit, A. Defaults to the Nyquist limit "
            "of the stack's pixel size; finer requests are clamped to Nyquist.",
        ),
    ] = None,
    ctffind_min_defocus_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="First-pass defocus search minimum, A."),
    ] = 5000.0,
    ctffind_max_defocus_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="First-pass defocus search maximum, A."),
    ] = 50000.0,
    ctffind_defocus_step_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Defocus search step, A."),
    ] = 100.0,
    ctffind_defocus_search_window_angstrom: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="Half-width of the second-pass defocus search around the zero-tilt hint, A.",
        ),
    ] = 1000.0,
    ctffind_include_astigmatism: Annotated[
        bool,
        typer.Option(help="Write per-view astigmatism into the defocus file."),
    ] = True,
    ctffind_exhaustive_search: Annotated[
        bool,
        typer.Option(help="Use CTFFIND's slower, more exhaustive search."),
    ] = False,
    ctffind_restrain_astigmatism_angstrom: Annotated[
        float | None,
        typer.Option(min=0.0, help="Restrain astigmatism to this tolerance, A."),
    ] = None,
    ctffind_search_phase_shift: Annotated[
        bool,
        typer.Option(help="Search for an additional phase shift (phase plate)."),
    ] = False,
    ctffind_determine_sample_tilt: Annotated[
        bool,
        typer.Option(
            help="Let CTFFIND5 fit each view's tilt and the defocus gradient it "
            "makes across the image. Not offered with a phase-shift search.",
        ),
    ] = False,
    ctf_max_excluded_views: Annotated[
        int,
        typer.Option(
            min=0,
            help="Views CTFFIND finds no Thon rings on are left out of the corrected "
            "stack when there are at most this many, and kept with a defocus "
            "interpolated from the fitted views when there are more. Either way "
            "the QC warns.",
        ),
    ] = 5,
    ctffind_min_phase_shift_rad: Annotated[
        float,
        typer.Option(help="Minimum searched phase shift, radians."),
    ] = 0.0,
    ctffind_max_phase_shift_rad: Annotated[
        float,
        typer.Option(help="Maximum searched phase shift, radians."),
    ] = 3.15,
    ctffind_phase_shift_step_rad: Annotated[
        float,
        typer.Option(min=0.0, help="Phase shift search step, radians."),
    ] = 0.2,
    ctffind_executable: Annotated[
        Path | None,
        typer.Option(help="CTFFIND5 executable when it is not on PATH."),
    ] = None,
    ctf_defocus_tol_nm: Annotated[
        float,
        typer.Option(min=0.0, help="ctfplotter defocus tolerance for center strips, nm."),
    ] = 200.0,
    ctf_left_defocus_tol_nm: Annotated[
        float,
        typer.Option(min=0.0, help="ctfplotter tolerance for strips left of center, nm."),
    ] = 2000.0,
    ctf_right_defocus_tol_nm: Annotated[
        float,
        typer.Option(min=0.0, help="ctfplotter tolerance for strips right of center, nm."),
    ] = 2000.0,
    ctf_tile_size: Annotated[
        int,
        typer.Option(min=1, help="ctfplotter square tile size, pixels."),
    ] = 256,
    ctf_autofit_range_deg: Annotated[
        float,
        typer.Option(min=0.0, help="ctfplotter autofit angular range, degrees."),
    ] = 16.0,
    ctf_autofit_step_deg: Annotated[
        float,
        typer.Option(min=0.0, help="ctfplotter autofit angular step, degrees."),
    ] = 2.0,
    ctf_ps_resolution: Annotated[
        int,
        typer.Option(min=1, help="ctfplotter power spectrum resolution."),
    ] = 101,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the CTF model, QC, and registry entries."),
    ] = False,
) -> None:
    """Estimate per-tilt defocus and write an IMOD-format CTF model."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    # Fitted on the stack built without a label, so that `correct-ctf` in any lane
    # can fall back to the resulting model. Named lanes rebuild the same stage at
    # their own binning, so without this the fit stack becomes unfindable as soon
    # as one of them has run — and re-running the lane from fine alignment, when
    # both stacks exist, could not get past this step.
    aligned_stack = _final_aligned_stack_artifact(
        artifact_registry, tilt_series_manifest, variant=variant, shared_fallback=True
    )
    alignment = _fine_alignment_artifact(
        artifact_registry, tilt_series_manifest, variant=variant
    )
    selected_backend = _ctf_estimation_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "ctf_voltage_kv": ctf_voltage_kv,
        "ctf_spherical_aberration_mm": ctf_spherical_aberration_mm,
        "ctf_amplitude_contrast": ctf_amplitude_contrast,
        "ctffind_box_size": ctffind_box_size,
        "ctffind_min_defocus_angstrom": ctffind_min_defocus_angstrom,
        "ctffind_max_defocus_angstrom": ctffind_max_defocus_angstrom,
        "ctffind_defocus_step_angstrom": ctffind_defocus_step_angstrom,
        "ctffind_defocus_search_window_angstrom": ctffind_defocus_search_window_angstrom,
        "ctffind_include_astigmatism": ctffind_include_astigmatism,
        "ctffind_exhaustive_search": ctffind_exhaustive_search,
        "ctffind_search_phase_shift": ctffind_search_phase_shift,
        "ctffind_determine_sample_tilt": ctffind_determine_sample_tilt,
        "ctf_max_excluded_views": ctf_max_excluded_views,
        "ctffind_min_phase_shift_rad": ctffind_min_phase_shift_rad,
        "ctffind_max_phase_shift_rad": ctffind_max_phase_shift_rad,
        "ctffind_phase_shift_step_rad": ctffind_phase_shift_step_rad,
        "ctf_defocus_tol_nm": ctf_defocus_tol_nm,
        "ctf_left_defocus_tol_nm": ctf_left_defocus_tol_nm,
        "ctf_right_defocus_tol_nm": ctf_right_defocus_tol_nm,
        "ctf_tile_size": ctf_tile_size,
        "ctf_autofit_range_deg": ctf_autofit_range_deg,
        "ctf_autofit_step_deg": ctf_autofit_step_deg,
        "ctf_ps_resolution": ctf_ps_resolution,
    }
    if ctffind_min_resolution_angstrom is not None:
        parameters["ctffind_min_resolution_angstrom"] = ctffind_min_resolution_angstrom
    if ctffind_max_resolution_angstrom is not None:
        parameters["ctffind_max_resolution_angstrom"] = ctffind_max_resolution_angstrom
    if ctffind_restrain_astigmatism_angstrom is not None:
        parameters["ctffind_restrain_astigmatism_angstrom"] = (
            ctffind_restrain_astigmatism_angstrom
        )
    if ctffind_executable is not None:
        parameters["ctffind_executable"] = ctffind_executable
    if ctf_expected_defocus_um is not None:
        parameters["ctf_expected_defocus_um"] = ctf_expected_defocus_um
    else:
        defocus_hint = _optional_defocus_hint_artifact(artifact_registry, tilt_series_manifest)
        if defocus_hint is not None:
            parameters["ctf_expected_defocus_um"] = defocus_hint.parameters["defocus_um"]
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = estimate_ctf_and_register(
        selected_backend,
        aligned_stack,
        alignment,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    ctf_model, qc_report = artifacts
    typer.echo(f"wrote CTF model: {ctf_model.path}")
    diagnostic_path = ctf_model.parameters.get("diagnostic_path")
    if isinstance(diagnostic_path, str) and Path(diagnostic_path).exists():
        typer.echo(f"wrote CTF diagnostic image: {diagnostic_path}")
    typer.echo(f"wrote CTF estimation QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("correct-ctf")
def correct_ctf(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for CTF artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="CTF correction backend to use."),
    ] = "imod-ctfphaseflip",
    ctf_model_backend: Annotated[
        str | None,
        typer.Option(
            help="Which estimation backend's CTF model to correct with, when several "
            "are registered."
        ),
    ] = None,
    ctf_defocus_tol_nm: Annotated[
        float,
        typer.Option(min=0.0, help="Defocus tolerance for center strips, nm."),
    ] = 200.0,
    ctf_interpolation_width: Annotated[
        int,
        typer.Option(min=1, help="ctfphaseflip strip interpolation width, pixels."),
    ] = 15,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the corrected stack, QC, and registry entries."),
    ] = False,
) -> None:
    """Phase-flip the final aligned stack using a registered CTF model.

    Voltage, spherical aberration, and amplitude contrast are taken from the
    CTF model, so correction cannot silently disagree with estimation.
    """

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    aligned_stack = _final_aligned_stack_artifact(
        artifact_registry, tilt_series_manifest, variant=variant
    )
    ctf_model = _ctf_model_artifact(
        artifact_registry,
        tilt_series_manifest,
        backend=ctf_model_backend,
        variant=variant,
    )
    selected_backend = _ctf_correction_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "ctf_defocus_tol_nm": ctf_defocus_tol_nm,
        "ctf_interpolation_width": ctf_interpolation_width,
    }
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = correct_ctf_and_register(
        selected_backend,
        aligned_stack,
        ctf_model,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    corrected_stack, qc_report = artifacts
    typer.echo(f"used CTF model: {ctf_model.path}")
    typer.echo(f"wrote CTF-corrected stack: {corrected_stack.path}")
    typer.echo(f"wrote CTF correction QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("reconstruct-tomogram")
def reconstruct_tomogram(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for tomogram artifacts.")],
    backend: Annotated[
        str,
        typer.Option(help="Tomogram reconstruction backend to use."),
    ] = "imod-tilt",
    thickness: Annotated[
        int | None,
        typer.Option(
            min=1,
            help=(
                "Reconstructed depth in output voxels; defaults to the "
                "fine-alignment surface analysis."
            ),
        ),
    ] = None,
    x_axis_tilt: Annotated[
        float | None,
        typer.Option(
            min=-90.0,
            max=90.0,
            help=(
                "Specimen X-axis tilt in degrees; defaults to the fine-alignment surface analysis."
            ),
        ),
    ] = None,
    z_shift: Annotated[
        float | None,
        typer.Option(
            help=(
                "Tomogram Z shift in output pixels; defaults to the "
                "fine-alignment surface analysis."
            ),
        ),
    ] = None,
    radial_cutoff: Annotated[
        float,
        typer.Option(min=0.0, max=0.5, help="IMOD radial filter cutoff."),
    ] = 0.35,
    radial_falloff: Annotated[
        float,
        typer.Option(min=0.0, max=0.5, help="IMOD radial filter falloff."),
    ] = 0.05,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    from_stack: Annotated[
        str,
        typer.Option(
            help="Which stack to reconstruct from: auto prefers a CTF-corrected "
            "stack when one exists, ctf-corrected requires one, aligned ignores "
            "it. Pair with --variant to keep both reconstructions."
        ),
    ] = "auto",
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite tomogram, export, QC, and registry entries."),
    ] = False,
) -> None:
    """Reconstruct a positioned tomogram from the final fine-aligned stack."""

    if from_stack not in {"auto", "ctf-corrected", "aligned"}:
        raise typer.BadParameter(
            f"unsupported stack choice {from_stack!r}; expected: auto, "
            "ctf-corrected, aligned",
            param_hint="from-stack",
        )

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    aligned_stack = _reconstruction_input_stack_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
        from_stack=from_stack,
    )
    alignment = _fine_alignment_artifact(
        artifact_registry,
        tilt_series_manifest,
        variant=variant,
    )
    selected_backend = _reconstruction_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "variant": variant,
        "radial_cutoff": radial_cutoff,
        "radial_falloff": radial_falloff,
    }
    if thickness is not None:
        parameters["thickness"] = thickness
    if x_axis_tilt is not None:
        parameters["x_axis_tilt_deg"] = x_axis_tilt
    if z_shift is not None:
        parameters["z_shift_px"] = z_shift
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = reconstruct_and_register(
        selected_backend,
        aligned_stack,
        alignment,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    tomogram, qc_report = artifacts
    typer.echo(f"wrote canonical tomogram: {tomogram.path}")
    typer.echo(f"wrote IMOD reconstruction: {tomogram.parameters['imod_rec_path']}")
    typer.echo(f"wrote reconstruction QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("restore-tomogram")
def restore_tomogram(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for restored tomograms.")],
    backend: Annotated[
        str,
        typer.Option(help="Tomogram restoration backend to use."),
    ] = "isonet2",
    isonet2_executable: Annotated[
        Path | None,
        typer.Option(help="IsoNet2 executable path when it is not on PATH."),
    ] = None,
    isonet2_model: Annotated[
        Path | None,
        typer.Option(help="Trained IsoNet .h5 model path."),
    ] = None,
    isonet2_gpu_id: Annotated[
        str | None,
        typer.Option(help="GPU id string passed to IsoNet --gpuID, for example 0 or 0,1."),
    ] = None,
    isonet2_cube_size: Annotated[
        int,
        typer.Option(min=1, help="IsoNet prediction cube size."),
    ] = 64,
    isonet2_crop_size: Annotated[
        int,
        typer.Option(min=1, help="IsoNet prediction crop size."),
    ] = 96,
    isonet2_batch_size: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Optional IsoNet prediction batch size.",
        ),
    ] = None,
    isonet2_number_subtomos: Annotated[
        int,
        typer.Option(min=1, help="Number of subtomograms recorded in the IsoNet star file."),
    ] = 100,
    isonet2_normalize_percentile: Annotated[
        bool,
        typer.Option(help="Pass IsoNet --normalize_percentile for prediction."),
    ] = True,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite restored tomogram, QC, and registry entries."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
) -> None:
    """Restore a reconstructed tomogram and register a canonical Zarr result."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest, variant=variant)
    selected_backend = _restoration_backend(backend)
    parameters: dict[str, object] = {
        "variant": variant,
        "overwrite": overwrite,
    }
    if isonet2_executable is not None:
        parameters["isonet2_executable"] = isonet2_executable
    if isonet2_model is not None:
        parameters["isonet2_model"] = isonet2_model
    if isonet2_gpu_id is not None:
        parameters["isonet2_gpu_id"] = isonet2_gpu_id
    parameters["isonet2_cube_size"] = isonet2_cube_size
    parameters["isonet2_crop_size"] = isonet2_crop_size
    if isonet2_batch_size is not None:
        parameters["isonet2_batch_size"] = isonet2_batch_size
    parameters["isonet2_number_subtomos"] = isonet2_number_subtomos
    parameters["isonet2_normalize_percentile"] = isonet2_normalize_percentile
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = restore_and_register(
        selected_backend,
        tomogram,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    restored, qc_report = artifacts
    typer.echo(f"wrote restored tomogram: {restored.path}")
    typer.echo(f"wrote restoration QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("segment-tomogram")
def segment_tomogram(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for segmentations.")],
    backend: Annotated[
        str,
        typer.Option(help="Tomogram segmentation backend to use."),
    ] = "membrain-seg",
    membrain_executable: Annotated[
        Path | None,
        typer.Option(help="MemBrain-seg executable path when membrain is not on PATH."),
    ] = None,
    membrain_model: Annotated[
        Path | None,
        typer.Option(help="Pre-trained MemBrain-seg checkpoint path."),
    ] = None,
    membrain_rescale_patches: Annotated[
        bool,
        typer.Option(help="Pass MemBrain-seg patch rescaling for inference."),
    ] = True,
    membrain_out_pixel_size: Annotated[
        float,
        typer.Option(min=0.0, help="MemBrain-seg internal patch pixel size in angstrom."),
    ] = 10.0,
    membrain_store_probabilities: Annotated[
        bool,
        typer.Option(help="Ask MemBrain-seg to store probability maps."),
    ] = False,
    membrain_store_connected_components: Annotated[
        bool,
        typer.Option(help="Ask MemBrain-seg to store connected-component labels."),
    ] = False,
    membrain_connected_component_threshold: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Remove connected components smaller than this voxel count.",
        ),
    ] = None,
    membrain_test_time_augmentation: Annotated[
        bool,
        typer.Option(help="Use MemBrain-seg test-time augmentation."),
    ] = True,
    membrain_store_uncertainty_map: Annotated[
        bool,
        typer.Option(help="Ask MemBrain-seg to store an uncertainty map."),
    ] = False,
    membrain_segmentation_threshold: Annotated[
        float,
        typer.Option(help="MemBrain-seg membrane score threshold."),
    ] = 0.0,
    membrain_sliding_window_size: Annotated[
        int,
        typer.Option(min=1, help="MemBrain-seg sliding-window size."),
    ] = 160,
    membrain_output_name: Annotated[
        str | None,
        typer.Option(help="Specific MemBrain-seg output .mrc/.rec name to canonicalize."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite segmentation, QC, and registry entries."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
) -> None:
    """Segment a reconstructed tomogram and register a canonical Zarr mask."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest, variant=variant)
    selected_backend = _segmentation_backend(backend)
    parameters: dict[str, object] = {
        "variant": variant,
        "overwrite": overwrite,
        "membrain_rescale_patches": membrain_rescale_patches,
        "membrain_out_pixel_size": membrain_out_pixel_size,
        "membrain_store_probabilities": membrain_store_probabilities,
        "membrain_store_connected_components": membrain_store_connected_components,
        "membrain_test_time_augmentation": membrain_test_time_augmentation,
        "membrain_store_uncertainty_map": membrain_store_uncertainty_map,
        "membrain_segmentation_threshold": membrain_segmentation_threshold,
        "membrain_sliding_window_size": membrain_sliding_window_size,
    }
    if membrain_executable is not None:
        parameters["membrain_executable"] = membrain_executable
    if membrain_model is not None:
        parameters["membrain_model"] = membrain_model
    if membrain_connected_component_threshold is not None:
        parameters["membrain_connected_component_threshold"] = (
            membrain_connected_component_threshold
        )
    if membrain_output_name is not None:
        parameters["membrain_output_name"] = membrain_output_name
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = segment_and_register(
        selected_backend,
        tomogram,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    segmentation, qc_report = artifacts
    typer.echo(f"wrote segmentation: {segmentation.path}")
    typer.echo(f"wrote segmentation QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


def _read_qc_report(report: Artifact) -> dict[str, Any]:
    """Load the JSON a QC artifact points at."""

    return cast(dict[str, Any], json.loads(report.path.read_text()))


def _echo_qc_warnings(report: dict[str, Any]) -> None:
    """Print what a QC report says, not only that it has something to say.

    Every command prints its QC status and none prints the reason, so a run that
    ends in "QC status: warning" tells the reader that something was noticed and
    not what. The text is already written to the report; this only puts it where
    it will be seen.
    """

    for warning in report.get("warnings", []):
        typer.echo(f"  warning: {warning}")


def _echo_picks_summary(report: dict[str, Any]) -> None:
    """Print how many particles were found, and how confident and how spread out.

    The count is the number that decides whether half an hour of template
    matching was worth anything, and it was not printed: a log ending in "QC
    status: pass" could not distinguish two particles from two thousand. Which is
    exactly the question an operator scrolling four series of output is asking.

    The score range comes with it because a count alone cannot say whether the
    picks are peaks or the tail of the noise distribution, and the spread in Z
    because these are FIB lamellae — particles confined to a slab, so a spread
    matching a uniform fill of the whole tomogram is worth seeing next to the
    count rather than buried in a report nobody opens.
    """

    count = report.get("particle_count")
    if count is None:
        return
    typer.echo(f"particles found: {count}")
    minimum = report.get("minimum_score")
    mean = report.get("mean_score")
    maximum = report.get("maximum_score")
    if None not in (minimum, mean, maximum):
        typer.echo(f"score: {minimum:.4f} min, {mean:.4f} mean, {maximum:.4f} max")
    observed = report.get("coordinate_z_sd")
    uniform = report.get("uniform_z_sd")
    median = report.get("coordinate_z_median")
    if observed is not None and uniform is not None and median is not None:
        typer.echo(
            f"z spread: {observed:.1f} px sd about z={median:.0f}, "
            f"against {uniform:.1f} px for a uniform fill"
        )


def _echo_averaging_summary(report: dict[str, Any]) -> None:
    """Print how many particles went in, how they split, and what came out.

    The resolution is printed only when the refinement earned the right to call it
    one: `gold_standard` false means the halves were split after a joint
    refinement, and such a number is not comparable with a published resolution.
    Printing it with a caveat attached would be worse than not printing it, since
    the number travels and the caveat does not.
    """

    count = report.get("particle_count")
    if count is None:
        return
    counts = report.get("half_set_counts") or {}
    split = " + ".join(str(counts[key]) for key in sorted(counts)) if counts else None
    across = f" ({split} across the half-sets)" if split else ""
    typer.echo(f"particles averaged: {count}{across}")

    per_series = report.get("particle_counts") or []
    series_ids = report.get("tilt_series_ids") or []
    if len(per_series) == len(series_ids) > 1:
        joined = ", ".join(f"{name} {n}" for name, n in zip(series_ids, per_series, strict=True))
        typer.echo(f"  from: {joined}")

    starting = report.get("starting_correlation_mean")
    final = report.get("mean_final_correlation")
    if starting is not None and final is not None:
        typer.echo(f"correlation: {starting:.4f} before refinement, {final:.4f} after")

    if report.get("iterations_run") == 0:
        # An average at the given poses: nothing was refined, so there are no
        # half-sets whose agreement could be called a resolution.
        return
    if not report.get("gold_standard", False):
        typer.echo(
            "resolution: not reported - the half-sets were refined against a shared "
            "reference, so an FSC between them is not a gold-standard estimate"
        )
        return
    resolution = report.get("resolution_angstrom")
    if resolution is None:
        typer.echo("resolution: no crossing - the two half-sets did not disagree anywhere")
    else:
        caveat = (
            ""
            if report.get("resolution_is_interpretable", True)
            else " (too few particles per half to be read as an estimate)"
        )
        typer.echo(
            f"resolution: {resolution:.1f} A at FSC 0.143 between the half-sets{caveat}"
        )
    trace = report.get("resolution_trace_angstrom") or []
    if len(trace) > 1:
        shown = ", ".join("none" if value is None else f"{value:.0f}" for value in trace)
        typer.echo(f"  resolution per iteration: {shown} A")
    agreement = report.get("half_set_agreement_trace") or []
    if len(agreement) > 1:
        # The continuous one. A rising then levelling curve is a refinement that
        # worked; a flat one next to a flat resolution means it genuinely did not
        # improve, which the shell-quantised crossing alone cannot establish.
        typer.echo(
            "  half-set agreement:       "
            + ", ".join(f"{value:+.3f}" for value in agreement)
        )
    limits = report.get("low_pass_trace_angstrom") or []
    if len(limits) > 1:
        # Printed beside the resolution because the two ARE a feedback loop: the
        # limit comes from the previous iteration's agreement, and the agreement
        # is measured on a reference cut at that limit. A resolution that never
        # moves while the limit never moves either says the loop is not running;
        # a resolution better than the limit says the halves agree in a band the
        # refinement never used, which is what template bias looks like.
        typer.echo(
            "  low-pass per iteration:   "
            + ", ".join(f"{value:.0f}" for value in limits)
            + " A"
        )


def _echo_extraction_summary(report: dict[str, Any]) -> None:
    """Print what survived extraction and what did not, with the reason.

    A stack silently smaller than its picks is the kind of thing that is noticed
    three steps later, when a count does not match. Both drop reasons are already
    recorded; neither was shown.
    """

    extracted = report.get("extracted_particle_count")
    requested = report.get("requested_particle_count")
    if extracted is None or requested is None:
        return
    typer.echo(f"particles extracted: {extracted} of {requested}")
    dropped_edge = report.get("dropped_edge_count") or 0
    dropped_score = report.get("dropped_score_count") or 0
    if dropped_edge or dropped_score:
        typer.echo(
            f"  dropped: {dropped_edge} too close to the tomogram edge for the box, "
            f"{dropped_score} below the score threshold"
        )


@app.command("prepare-template")
def prepare_template(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for template artifacts.")],
    emdb_id: Annotated[
        str | None,
        typer.Option(help="EMDB accession to fetch, e.g. EMD-2638. Cached between runs."),
    ] = None,
    source_map: Annotated[
        Path | None,
        typer.Option(help="Local reference map MRC/CCP4, instead of fetching from EMDB."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(help="Template preparation backend to use."),
    ] = "pytom",
    template_particle_diameter_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Particle diameter where it is widest, A. Sizes the spherical mask "
            "(that radius plus 3 px) and the box (twice the diameter), unless they "
            "are given.",
        ),
    ] = None,
    template_mask_radius_px: Annotated[
        float | None,
        typer.Option(min=0.0, help="Mask radius in template pixels, overriding the diameter."),
    ] = None,
    template_mask_radius_y_px: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Mask semi-axis along y, pixels. Give this and the z radius to draw "
            "an ellipsoid instead of a sphere, for a reference that is not a compact "
            "particle: a lattice or membrane patch is thin along one axis.",
        ),
    ] = None,
    template_mask_radius_z_px: Annotated[
        float | None,
        typer.Option(min=0.0, help="Mask semi-axis along z, pixels. See the y radius."),
    ] = None,
    template_mask_sigma_px: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Gaussian falloff at the mask edge, pixels; 1 when not given, and "
            "0 for a hard edge.",
        ),
    ] = None,
    template_box_size_px: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Template box size. Defaults to twice the particle diameter when one "
            "is given, else to what the resampling yields; pytom keeps the map's own "
            "box when that is larger.",
        ),
    ] = None,
    template_low_pass_angstrom: Annotated[
        float | None,
        typer.Option(min=0.0, help="Low-pass filter the template to this resolution, A."),
    ] = None,
    template_invert: Annotated[
        bool,
        typer.Option(
            help="Multiply the template by -1. Needed when the tomogram has dark "
            "particles and the reference map has bright density, which is the usual "
            "cryo-ET case."
        ),
    ] = True,
    template_center: Annotated[
        bool,
        typer.Option(help="Center the density in the box before resampling."),
    ] = False,
    template_mirror: Annotated[
        bool,
        typer.Option(help="Mirror the template, for the opposite handedness."),
    ] = False,
    template_input_voxel_size_angstrom: Annotated[
        float | None,
        typer.Option(min=0.0, help="Override the reference map's own voxel size, A."),
    ] = None,
    emdb_cache_dir: Annotated[
        Path | None,
        typer.Option(help="Where to cache downloaded EMDB maps (default: <out>/emdb-cache)."),
    ] = None,
    pytom_create_template_executable: Annotated[
        Path | None,
        typer.Option(help="pytom_create_template.py path when it is not on PATH."),
    ] = None,
    pytom_create_mask_executable: Annotated[
        Path | None,
        typer.Option(help="pytom_create_mask.py path when it is not on PATH."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite template, mask, QC, and registry entries."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    output_variant: Annotated[
        str | None,
        typer.Option(
            help="Label the OUTPUT with this instead of --variant, while still "
            "selecting inputs by --variant. Without it the two are one string, so a "
            "run that varies a parameter over the same inputs either finds nothing "
            "or overwrites what it is being compared against."
        ),
    ] = None,
) -> None:
    """Fetch a reference map and build a matching template plus mask."""

    if (emdb_id is None) == (source_map is None):
        raise typer.BadParameter(
            "pass exactly one of --emdb-id or --source-map",
            param_hint="emdb_id",
        )

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest, variant=variant)

    if emdb_id is not None:
        cache_dir = emdb_cache_dir if emdb_cache_dir is not None else out / "emdb-cache"
        resolved_map = fetch_emdb_map(
            emdb_id,
            cache_dir=cache_dir,
            progress=typer.echo,
        )
    else:
        assert source_map is not None
        resolved_map = source_map

    selected_backend = _template_preparation_backend(backend)
    parameters: dict[str, object] = {
        # Inputs were resolved by `variant`; this only labels what is written.
        "variant": output_variant if output_variant is not None else variant,
        "overwrite": overwrite,
        "template_invert": template_invert,
        "template_center": template_center,
        "template_mirror": template_mirror,
    }
    if emdb_id is not None:
        parameters["template_emdb_id"] = normalize_emdb_id(emdb_id)
    if template_particle_diameter_angstrom is not None:
        parameters["template_particle_diameter_angstrom"] = (
            template_particle_diameter_angstrom
        )
    if template_mask_radius_px is not None:
        parameters["template_mask_radius_px"] = template_mask_radius_px
    if template_mask_radius_y_px is not None:
        parameters["template_mask_radius_y_px"] = template_mask_radius_y_px
    if template_mask_radius_z_px is not None:
        parameters["template_mask_radius_z_px"] = template_mask_radius_z_px
    if template_mask_sigma_px is not None:
        parameters["template_mask_sigma_px"] = template_mask_sigma_px
    if template_box_size_px is not None:
        parameters["template_box_size_px"] = template_box_size_px
    if template_low_pass_angstrom is not None:
        parameters["template_low_pass_angstrom"] = template_low_pass_angstrom
    if template_input_voxel_size_angstrom is not None:
        parameters["template_input_voxel_size_angstrom"] = template_input_voxel_size_angstrom
    if pytom_create_template_executable is not None:
        parameters["pytom_create_template_executable"] = pytom_create_template_executable
    if pytom_create_mask_executable is not None:
        parameters["pytom_create_mask_executable"] = pytom_create_mask_executable
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = prepare_template_and_register(
        selected_backend,
        resolved_map,
        tomogram,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    template, mask, qc_report = artifacts
    report = _read_qc_report(qc_report)
    voxel_size = template.pixel_spacing_angstrom or 0.0
    extents = (
        report["density_extent_x_px"],
        report["density_extent_y_px"],
        report["density_extent_z_px"],
    )
    radius = report["mask_radius_px"]

    typer.echo(f"wrote template: {template.path}")
    typer.echo(f"wrote template mask: {mask.path}")
    typer.echo(f"box size: {template.parameters['box_size_px']} px")
    typer.echo(f"voxel size: {voxel_size} A")
    # Printed rather than left in the report, because these are the two numbers
    # a wrong template gets wrong, and nobody opens a JSON file to check a run
    # that said it succeeded.
    in_pixels = "/".join(f"{extent:.1f}" for extent in extents)
    in_angstrom = "/".join(f"{extent * voxel_size:.0f}" for extent in extents)
    typer.echo(f"density extent: {in_pixels} px ({in_angstrom} A)")

    derived = ""
    if report["mask_radius_derived_from_density"]:
        derived = " - derived from the map, no size was given"
    typer.echo(f"mask radius: {radius:.1f} px ({radius * voxel_size:.0f} A){derived}")
    typer.echo(f"wrote template QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    _echo_qc_warnings(report)
    typer.echo(f"updated {registry}")


@app.command("pick-particles")
def pick_particles(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for picks.")],
    pytom_template: Annotated[
        Path | None,
        typer.Option(
            help="Template density map (MRC). Defaults to the template registered by "
            "prepare-template."
        ),
    ] = None,
    pytom_template_mask: Annotated[
        Path | None,
        typer.Option(
            help="Template mask (MRC), same box size as the template. Defaults to the "
            "registered one."
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(help="Particle picking backend to use."),
    ] = "pytom-match-pick",
    use_segmentation_mask: Annotated[
        bool,
        typer.Option(
            help="Constrain picking to the registered membrane segmentation, if present."
        ),
    ] = True,
    pytom_match_template_executable: Annotated[
        Path | None,
        typer.Option(
            help="pytom_match_template.py path when it is not on PATH."
        ),
    ] = None,
    pytom_extract_candidates_executable: Annotated[
        Path | None,
        typer.Option(
            help="pytom_extract_candidates.py path when it is not on PATH."
        ),
    ] = None,
    pytom_gpu_ids: Annotated[
        str,
        typer.Option(help="Comma-separated GPU ids passed to pytom_match_template.py -g."),
    ] = "0",
    pytom_particle_diameter: Annotated[
        float | None,
        typer.Option(min=0.0, help="Particle diameter, angstrom (exclusive with angular-search)."),
    ] = None,
    pytom_angular_search: Annotated[
        float | None,
        typer.Option(min=0.0, help="Angular search in degrees (mutually exclusive with diameter)."),
    ] = None,
    pytom_number_of_particles: Annotated[
        int,
        typer.Option(
            min=1,
            help="Cap on how many candidates extraction returns. Required by "
            "pytom. For a lattice this is a sampling density, not a particle count.",
        ),
    ] = 1000,
    pytom_extract_particle_diameter: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Diameter, angstrom, that extraction uses to keep peaks apart: it "
            "discards a region of half this size around each accepted peak. Defaults "
            "to --pytom-particle-diameter. Set it separately when the size that keeps "
            "peaks apart differs from the one that sets the angular sampling.",
        ),
    ] = None,
    pytom_non_spherical_mask: Annotated[
        bool | None,
        typer.Option(
            help="Tell pytom the template mask is not rotation-invariant. Detected "
            "from the mask artifact when it was prepared here; set it explicitly for "
            "a mask from elsewhere. Roughly doubles the run.",
        ),
    ] = None,
    segmentation_mask_dilation_px: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Grow the segmentation by this many voxels before using it as a "
            "search mask. pytom discards every candidate whose centre the mask "
            "does not cover, with no tolerance, so a mask drawn on a membrane "
            "admits nothing lying under it.",
        ),
    ] = None,
    pytom_ctf_weighting: Annotated[
        bool,
        typer.Option(
            help="Weight the template by the CTF the tomogram carries. Phase "
            "flipping fixes the sign but not the amplitude, so a corrected "
            "tomogram still carries |CTF| while a deposited reference map does "
            "not. Values come from the CTF model that corrected this tomogram, "
            "so they cannot disagree with it. Off when nothing corrected it."
        ),
    ] = True,
    pytom_per_tilt_weighting: Annotated[
        bool,
        typer.Option(
            help="Weight each tilt separately instead of a binary missing wedge. "
            "Uses the tilt angles that are already passed.",
        ),
    ] = False,
    pytom_spectral_whitening: Annotated[
        bool,
        typer.Option(help="Whiten the spectrum of both tomogram and template."),
    ] = False,
    pytom_rng_seed: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Seed for the random-phase noise map, so two runs differ only by "
            "what you changed.",
        ),
    ] = None,
    pytom_search_z: Annotated[
        str | None,
        typer.Option(
            help="Restrict the search along z to 'min,max' in voxels, keeping the "
            "reconstruction's own top and bottom edges out of the correlation.",
        ),
    ] = None,
    pytom_volume_split: Annotated[
        str | None,
        typer.Option(
            help="Match the search in 'x,y,z' pieces (pytom's --volume-split), for a "
            "tomogram whose search does not fit in GPU memory. 1,1,1 is the unsplit "
            "search.",
        ),
    ] = None,
    pytom_z_axis_symmetry: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Rotational symmetry of the template about its z axis. Cuts the "
            "orientation search by this factor, so it is only correct when the box is "
            "centred on the symmetry axis.",
        ),
    ] = None,
    pytom_low_pass: Annotated[
        float | None,
        typer.Option(min=0.0, help="Low-pass filter resolution in angstrom."),
    ] = None,
    pytom_high_pass: Annotated[
        float | None,
        typer.Option(min=0.0, help="High-pass filter resolution in angstrom."),
    ] = None,
    pytom_random_phase_correction: Annotated[
        bool,
        typer.Option(help="Enable phase-randomized background control (roughly doubles runtime)."),
    ] = False,
    pytom_number_of_false_positives: Annotated[
        float | None,
        typer.Option(min=0.0, help="Expected false positives for the automatic score cutoff."),
    ] = None,
    pytom_cut_off: Annotated[
        float | None,
        typer.Option(min=0.0, help="Hard LCCmax cutoff, overriding number-of-false-positives."),
    ] = None,
    pytom_tophat_filter: Annotated[
        bool,
        typer.Option(help="Apply the tophat-transform false-positive filter during extraction."),
    ] = False,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite picks, QC, and registry entries."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    output_variant: Annotated[
        str | None,
        typer.Option(
            help="Label the OUTPUT with this instead of --variant, while still "
            "selecting inputs by --variant. Without it the two are one string, so a "
            "run that varies a parameter over the same inputs either finds nothing "
            "or overwrites what it is being compared against."
        ),
    ] = None,
    pytom_keep_score_volume: Annotated[
        bool,
        typer.Option(
            "--pytom-keep-score-volume",
            help="Keep the correlation and orientation volumes beside the picks. "
            "They are what the half hour of matching actually produced — extracting "
            "candidates from them takes seconds — so without them every change to a "
            "cut-off or an exclusion distance costs a full re-match. About a gigabyte "
            "each per series at bin8, which is why this is off by default.",
        ),
    ] = False,
) -> None:
    """Pick particles from a reconstructed tomogram and register a canonical STAR result."""

    if pytom_volume_split is not None:
        try:
            parse_volume_split(pytom_volume_split)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="pytom-volume-split") from exc
    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest, variant=variant)
    selected_backend = _picking_backend(backend)
    supporting_artifacts: list[Artifact] = []
    if use_segmentation_mask:
        segmentation = _optional_segmentation_artifact(
            artifact_registry, tilt_series_manifest, variant=variant
        )
        if segmentation is not None:
            supporting_artifacts.append(segmentation)
    registered_template = _template_artifacts(
        artifact_registry, tilt_series_manifest, variant=variant
    )
    if pytom_template is None or pytom_template_mask is None:
        if registered_template is None:
            raise typer.BadParameter(
                "no template registered for this tilt series; run prepare-template first, "
                "or pass --pytom-template and --pytom-template-mask explicitly",
                param_hint="pytom_template",
            )
        template_artifact, mask_artifact = registered_template
        if pytom_template is None:
            pytom_template = template_artifact.path
        if pytom_template_mask is None:
            pytom_template_mask = mask_artifact.path
        supporting_artifacts.extend([template_artifact, mask_artifact])
        if pytom_non_spherical_mask is None:
            # The mask recorded its own shape when it was prepared, so whether
            # pytom's rotation-invariant fast path is valid is a fact about the
            # artifact rather than something to remember on the command line.
            recorded = mask_artifact.parameters.get("mask_is_spherical")
            if isinstance(recorded, bool):
                pytom_non_spherical_mask = not recorded
    parameters: dict[str, object] = {
        # Inputs were resolved by `variant`; this only labels what is written.
        "variant": output_variant if output_variant is not None else variant,
        "pytom_keep_score_volume": pytom_keep_score_volume,
        "overwrite": overwrite,
        "pytom_template": pytom_template,
        "pytom_template_mask": pytom_template_mask,
        "pytom_gpu_ids": pytom_gpu_ids,
        "pytom_random_phase_correction": pytom_random_phase_correction,
        "pytom_tophat_filter": pytom_tophat_filter,
    }
    if pytom_match_template_executable is not None:
        parameters["pytom_match_template_executable"] = pytom_match_template_executable
    if pytom_extract_candidates_executable is not None:
        parameters["pytom_extract_candidates_executable"] = pytom_extract_candidates_executable
    if pytom_particle_diameter is not None:
        parameters["pytom_particle_diameter"] = pytom_particle_diameter
    if pytom_angular_search is not None:
        parameters["pytom_angular_search"] = pytom_angular_search
    tilt_angles_path = _tilt_angles_for_tomogram(
        artifact_registry, tilt_series_manifest, tomogram
    )
    if tilt_angles_path is not None:
        parameters["pytom_tilt_angles_path"] = tilt_angles_path
    weighting = (
        _ctf_weighting_for_tomogram(artifact_registry, tilt_series_manifest, tomogram)
        if pytom_ctf_weighting
        else None
    )
    if weighting is not None:
        parameters["pytom_ctf_defocus_path"] = weighting.defocus_path
        parameters["pytom_ctf_voltage_kv"] = weighting.voltage_kv
        parameters["pytom_ctf_spherical_aberration_mm"] = weighting.spherical_aberration_mm
        parameters["pytom_ctf_amplitude_contrast"] = weighting.amplitude_contrast
        parameters["pytom_tomogram_ctf_model"] = weighting.model
        if weighting.z_values is not None:
            parameters["pytom_ctf_defocus_z_values"] = weighting.z_values
    if segmentation_mask_dilation_px is not None:
        parameters["pytom_segmentation_dilation_px"] = segmentation_mask_dilation_px
    parameters["pytom_number_of_particles"] = pytom_number_of_particles
    parameters["pytom_per_tilt_weighting"] = pytom_per_tilt_weighting
    parameters["pytom_spectral_whitening"] = pytom_spectral_whitening
    if pytom_extract_particle_diameter is not None:
        parameters["pytom_extract_particle_diameter"] = pytom_extract_particle_diameter
    if pytom_non_spherical_mask is not None:
        parameters["pytom_non_spherical_mask"] = pytom_non_spherical_mask
    if pytom_rng_seed is not None:
        parameters["pytom_rng_seed"] = pytom_rng_seed
    if pytom_search_z is not None:
        parameters["pytom_search_z"] = pytom_search_z
    if pytom_volume_split is not None:
        parameters["pytom_volume_split"] = pytom_volume_split
    if pytom_z_axis_symmetry is not None:
        parameters["pytom_z_axis_symmetry"] = pytom_z_axis_symmetry
    if pytom_low_pass is not None:
        parameters["pytom_low_pass"] = pytom_low_pass
    else:
        derived, source = _derived_low_pass_angstrom(
            artifact_registry, tilt_series_manifest, variant=variant
        )
        parameters["pytom_low_pass"] = derived
        parameters["pytom_low_pass_derived_from"] = source
        typer.echo(
            f"low-pass: {derived:.1f} A, derived from a fine-alignment residual of "
            f"{source} — pass --pytom-low-pass to override"
        )
    if pytom_high_pass is not None:
        parameters["pytom_high_pass"] = pytom_high_pass
    if pytom_number_of_false_positives is not None:
        parameters["pytom_number_of_false_positives"] = pytom_number_of_false_positives
    if pytom_cut_off is not None:
        parameters["pytom_cut_off"] = pytom_cut_off
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = pick_and_register(
        selected_backend,
        tomogram,
        tilt_series_manifest,
        context,
        artifact_registry,
        supporting_artifacts=supporting_artifacts,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    *match, picks, qc_report = artifacts
    report = _read_qc_report(qc_report)
    if match:
        typer.echo(f"registered template match: {match[0].id}")
        if match[0].parameters.get("volumes_retained"):
            typer.echo(
                "  its score volume is kept, so extract-picks can re-read it at "
                "another cut-off without matching again"
            )
    typer.echo(f"wrote picks: {picks.path}")
    _echo_picks_summary(report)
    typer.echo(f"wrote picking QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    _echo_qc_warnings(report)
    typer.echo(f"updated {registry}")


@app.command("filter-picks")
def filter_picks(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Project output directory.")],
    output_variant: Annotated[
        str,
        typer.Option(
            help="Label for the kept picks. Extraction, the Dynamo export and averaging "
            "select them by it, as `--variant <label>`."
        ),
    ],
    variant: Annotated[
        str | None,
        typer.Option(help="Label of the picks to filter, e.g. the lane's own."),
    ] = None,
    max_membrane_distance_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="Drop picks further than this from the nearest voxel of the membrane "
            "segmentation the search was masked with.",
        ),
    ] = None,
    min_membrane_distance_angstrom: Annotated[
        float,
        typer.Option(min=0.0, help="Drop picks closer than this to the nearest membrane voxel."),
    ] = 0.0,
    normal_angle_deg: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=90.0,
            help="Expected angle between the template axis and the membrane normal: 0 for "
            "a target whose axis runs along the normal, as the Gag lattice's does. Leave "
            "it out for a target with no fixed orientation to the membrane.",
        ),
    ] = None,
    angle_tolerance_deg: Annotated[
        float | None,
        typer.Option(min=0.0, max=90.0, help="How far from the expected angle a pick may be."),
    ] = None,
    axis_direction: Annotated[
        str,
        typer.Option(
            help="toward, away or either: whether the template axis must point toward the "
            "nearest membrane voxel. Once real picks show which way it points, this tells "
            "the two sides of a membrane apart. Read a dry run's counts first.",
        ),
    ] = "either",
    template_axis: Annotated[
        str,
        typer.Option(help="Template axis compared with the membrane normal: x, y or z."),
    ] = "z",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Measure and report only: the per-pick table and the histograms, with "
            "nothing registered. The rule is chosen from these.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the filter's outputs and registry entries."),
    ] = False,
) -> None:
    """Keep the picks that sit where the target can: at its distance from the membrane,
    oriented to it as the target is."""

    if axis_direction not in AXIS_DIRECTIONS:
        raise typer.BadParameter(
            f"expected one of {', '.join(AXIS_DIRECTIONS)}, got {axis_direction!r}",
            param_hint="axis-direction",
        )
    if template_axis not in TEMPLATE_AXES:
        raise typer.BadParameter(
            f"expected one of {', '.join(TEMPLATE_AXES)}, got {template_axis!r}",
            param_hint="template-axis",
        )
    try:
        rule = MembraneRule(
            min_distance_angstrom=min_membrane_distance_angstrom,
            max_distance_angstrom=max_membrane_distance_angstrom,
            normal_angle_deg=normal_angle_deg,
            angle_tolerance_deg=angle_tolerance_deg,
            axis_direction=axis_direction,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    picks = _picks_artifact(artifact_registry, tilt_series_manifest, variant=variant)
    # The membrane the search itself was masked with, so the filter and the
    # search agree on where the membranes are.
    segmentation_id = picks.parameters.get("segmentation_id")
    if not isinstance(segmentation_id, str):
        raise typer.BadParameter(
            f"{picks.id} was not searched with a segmentation, so there is no membrane "
            "to measure it against",
            param_hint="variant",
        )
    try:
        segmentation = artifact_registry.get(segmentation_id)
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(
            f"{picks.id} names segmentation {segmentation_id!r}, which is not in the registry",
            param_hint="registry",
        ) from exc

    result = filter_picks_by_membrane(
        picks,
        segmentation,
        tilt_series_manifest,
        output_root=out,
        label=output_variant,
        rule=rule,
        template_axis=template_axis,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    typer.echo(f"measured {picks.id} against {segmentation.id}")
    for line in summary_lines(result.summary):
        typer.echo(line)
    typer.echo(f"wrote per-pick table: {result.table_path}")
    typer.echo(f"wrote filter QC: {result.report_path}")
    for warning in cast(list[str], result.summary["warnings"]):
        typer.echo(f"warning: {warning}")
    if dry_run:
        typer.echo("dry run: nothing registered")
        return
    artifact_registry.extend(result.artifacts, replace=overwrite)
    artifact_registry.commit(registry)
    typer.echo(f"wrote picks: {result.star_path}")
    typer.echo(f"registered {result.artifacts[0].id}")
    typer.echo(f"updated {registry}")


@app.command("select-picks")
def select_picks(
    manifest: Annotated[
        list[Path],
        typer.Option(
            help="Tilt-series manifest JSON written by init. Repeat it to select several "
            "series in one run, each at its own threshold or all at one."
        ),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Project output directory.")],
    output_variant: Annotated[
        str,
        typer.Option(
            help="Label for the kept picks. Extraction, the Dynamo export, averaging and "
            "the report select them by it, as `--variant <label>`."
        ),
    ],
    variant: Annotated[
        str | None,
        typer.Option(help="Label of the picks to select from, e.g. the lane's own."),
    ] = None,
    minimum_sigma: Annotated[
        float | None,
        typer.Option(
            help="Keep picks at or above this many σ of their own search's noise, the unit "
            "the report's score curves are drawn in. One value for every series."
        ),
    ] = None,
    minimum_score: Annotated[
        float | None,
        typer.Option(
            help="Keep picks at or above this raw pytom LCCmax. One value for every series."
        ),
    ] = None,
    thresholds_sigma: Annotated[
        Path | None,
        typer.Option(
            help="JSON object giving each series its own threshold in σ: "
            '{"TS_0306": 7.5, ...}. Without any threshold option the file in its fixed '
            "place is used, <out>/picks/<variant>_thresholds_sigma.json, which a first "
            "run writes with every series null for you to fill in."
        ),
    ] = None,
    thresholds_score: Annotated[
        Path | None,
        typer.Option(
            help="JSON object giving each series its own raw pytom LCCmax, the layout "
            "scripts/qc_dynamo_picks.py writes as thresholds.json."
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Count what each threshold would keep and write the QC, registering nothing.",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the selection's outputs and registry entries."),
    ] = False,
) -> None:
    """Keep the picks that score at least a threshold, per series or one for all.

    With no threshold option, the per-series thresholds come from
    <out>/picks/<variant>_thresholds_sigma.json. The first run writes that file, every
    series null beside hints on where its scores lie, and stops; it selects once
    each series there has a number. One threshold for all needs no file.
    """

    given = {
        "minimum-sigma": minimum_sigma,
        "minimum-score": minimum_score,
        "thresholds-sigma": thresholds_sigma,
        "thresholds-score": thresholds_score,
    }
    chosen = [name for name, value in given.items() if value is not None]
    if len(chosen) > 1:
        raise typer.BadParameter(
            "give at most one of --minimum-sigma, --minimum-score, --thresholds-sigma and "
            f"--thresholds-score; got {', '.join('--' + name for name in chosen)}",
            param_hint="minimum-sigma",
        )
    hint = "--" + (chosen[0] if chosen else "thresholds-sigma")
    manifests = [TiltSeriesManifest.model_validate_json(path.read_text()) for path in manifest]
    names = [one.tilt_series_id for one in manifests]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise typer.BadParameter(
            f"the same tilt-series was given more than once: {', '.join(duplicates)}",
            param_hint="manifest",
        )
    artifact_registry = ArtifactRegistry.load(registry)
    if not chosen:
        # No threshold given: the per-series file in its fixed place, made here
        # the first time with every series null, and used once each is filled.
        template = default_thresholds_path(out, variant)
        try:
            hints = {
                one.tilt_series_id: score_hints(
                    _picks_artifact(artifact_registry, one, variant=variant)
                )
                for one in manifests
            }
            unfilled = write_thresholds_template(template, hints)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint=hint) from exc
        if unfilled:
            typer.echo(f"thresholds file: {template}")
            for series in unfilled:
                facts = hints[series]
                typer.echo(
                    f"  {series}: to fill · {facts['picks']} picks"
                    + (
                        f", best {facts['best_sigma']}σ, lowest {facts['lowest_sigma']}σ, "
                        f"one false positive at {facts['one_false_positive_sigma']}σ"
                        if "best_sigma" in facts
                        else ""
                    )
                )
            typer.echo(
                "Give each a threshold in σ there and run this again, or pass "
                "--minimum-sigma for one threshold for every series."
            )
            raise typer.Exit(code=1)
        thresholds_sigma = template
    try:
        if thresholds_sigma is not None:
            thresholds = thresholds_from_json(thresholds_sigma, "sigma", names)
        elif thresholds_score is not None:
            thresholds = thresholds_from_json(thresholds_score, "score", names)
        elif minimum_sigma is not None:
            one = ScoreThreshold(minimum_sigma, "sigma", "--minimum-sigma")
            thresholds = dict.fromkeys(names, one)
        else:
            assert minimum_score is not None
            one = ScoreThreshold(minimum_score, "score", "--minimum-score")
            thresholds = dict.fromkeys(names, one)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint=hint) from exc
    # Every series is measured and checked before any is written, so a threshold
    # that keeps nothing in the third series, or outputs already there, leave the
    # project as it was rather than two series selected and one not.
    planned = []
    for tilt_series_manifest in manifests:
        series = tilt_series_manifest.tilt_series_id
        picks = _picks_artifact(artifact_registry, tilt_series_manifest, variant=variant)
        threshold = thresholds[series]
        if picks.variant == output_variant:
            raise typer.BadParameter(
                f"the selected picks need a label of their own; {picks.id} already carries "
                f"{output_variant!r}",
                param_hint="output-variant",
            )
        try:
            measurement = measure_score_selection(picks, threshold)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint=hint) from exc
        if not dry_run and not measurement.keep_rows:
            raise typer.BadParameter(
                f"{series}: no pick reaches {threshold.describe()}; no series was selected",
                param_hint=hint,
            )
        star_path, report_path = selection_paths(picks, series, out, output_variant)
        outputs = [report_path] if dry_run else [star_path, report_path]
        existing = [path for path in outputs if path.exists()]
        if existing and not overwrite:
            raise typer.BadParameter(
                f"outputs already exist: {', '.join(map(str, existing))}; pass --overwrite "
                "to replace them",
                param_hint="output-variant",
            )
        planned.append((tilt_series_manifest, picks, threshold, measurement))

    results = []
    for tilt_series_manifest, picks, threshold, measurement in planned:
        result = select_picks_by_score(
            picks,
            tilt_series_manifest,
            output_root=out,
            label=output_variant,
            threshold=threshold,
            overwrite=overwrite,
            dry_run=dry_run,
            measurement=measurement,
        )
        summary = result.summary
        converted = summary["minimum_score"]
        sigma = summary["minimum_sigma"]
        typer.echo(
            f"{tilt_series_manifest.tilt_series_id}: keeps {summary['kept']} of "
            f"{summary['picks']} at {threshold.describe()}"
            + (
                f" (LCCmax {converted:.4f})"
                if threshold.unit == "sigma"
                else (f" ({sigma:.2f}σ)" if isinstance(sigma, float) else "")
            )
        )
        for warning in cast(list[str], summary["warnings"]):
            typer.echo(f"  warning: {warning}")
        results.append(result)
    if dry_run:
        typer.echo("dry run: nothing registered")
        return
    for result in results:
        artifact_registry.extend(result.artifacts, replace=overwrite)
    artifact_registry.commit(registry)
    for result in results:
        typer.echo(f"registered {result.artifacts[0].id}: {result.star_path}")
    typer.echo(f"updated {registry}")


@app.command("visualize-picks")
def visualize_picks(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to read."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for the export.")],
    sphere_radius_px: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Radius, in tomogram voxels, to draw each pick at in 3dmod. "
            "Defaults to the radius the template mask was built with.",
        ),
    ] = None,
    imod_dir: Annotated[
        Path | None,
        typer.Option(help="IMOD installation directory when it is not configured."),
    ] = None,
    point2model_executable: Annotated[
        Path | None,
        typer.Option(help="IMOD point2model executable when it is not on PATH."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite an existing export."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
) -> None:
    """Export a tomogram and its picks for inspection in ChimeraX and in 3dmod."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    # The picks are found by label; the tomogram comes from what the picks
    # themselves recorded rather than from a second lookup on the same label.
    # One `--variant` cannot select both: picks made by a parameter sweep carry
    # the sweep's label while the tomogram they were found in keeps the lane's,
    # so asking for one label found the picks and lost the tomogram. Following
    # the recorded lineage cannot mismatch them, and needs no extra flag.
    picks = _picks_artifact(artifact_registry, tilt_series_manifest, variant=variant)
    tomogram = _tomogram_from_picks(
        artifact_registry, picks, tilt_series_manifest, variant=variant
    )

    # The template and the segmentation come from the picks' lineage too, for
    # the same reason: picks narrowed by filter-picks carry a label of their
    # own and share both with the list they came from, so a lookup by their
    # label finds neither. The label stays the fallback for picks whose lineage
    # does not record them, such as a match against a template file passed on
    # the command line.
    registered_template = _template_from_picks(
        artifact_registry, picks
    ) or _template_artifacts(artifact_registry, tilt_series_manifest, variant=variant)
    segmentations = artifact_registry.ancestors_of_kind(picks.id, ArtifactKind.SEGMENTATION)
    segmentation = (
        segmentations[0]
        if len(segmentations) == 1
        else _optional_segmentation_artifact(
            artifact_registry, tilt_series_manifest, variant=variant
        )
    )
    parameters: dict[str, Any] = {}
    if imod_dir is not None:
        parameters["imod_dir"] = imod_dir
    if point2model_executable is not None:
        parameters["point2model_executable"] = point2model_executable
    # point2model is a converter, not a compute step: the device is only here
    # because every backend context carries one.
    context = BackendContext(
        output_dir=out,
        device=resolve_device("cpu"),
        parameters=parameters,
    )
    sphere_radius = resolve_sphere_radius(
        requested_px=sphere_radius_px,
        template_mask=registered_template[1] if registered_template else None,
        voxel_size_angstrom=tomogram.pixel_spacing_angstrom,
    )
    export = write_picks_visualization(
        template=registered_template[0] if registered_template else None,
        segmentation=segmentation,
        tomogram=tomogram,
        picks=picks,
        out=out,
        context=context,
        sphere_radius=sphere_radius,
        overwrite=overwrite,
    )

    typer.echo(f"wrote tomogram: {export.tomogram_path}")
    typer.echo(f"wrote picks: {export.picks_path}")
    if export.template_path is not None:
        typer.echo(f"wrote template: {export.template_path}")
    if export.segmentation_path is not None:
        typer.echo(f"wrote segmentation: {export.segmentation_path}")
    typer.echo(f"wrote ChimeraX script: {export.chimerax_script_path}")
    typer.echo(f"wrote IMOD model: {export.imod_model_path}")
    typer.echo(
        f"  {export.pick_count} picks, drawn as spheres of "
        f"{export.sphere_radius.px} px — {export.sphere_radius.source}"
    )
    typer.echo(f"open with: chimerax {export.chimerax_script_path}")
    typer.echo(f"open with: 3dmod {export.tomogram_path} {export.imod_model_path}")


@app.command("report")
def report(
    project: Annotated[
        Path,
        typer.Option(help="Project directory: the one holding artifacts.json and manifests/."),
    ],
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label of the picks to report on, as given to pick-particles or "
            "filter-picks. Each series' tomogram, CTF fit, membranes and template are then "
            "found through those picks' own lineage."
        ),
    ] = None,
    series: Annotated[
        list[str] | None,
        typer.Option(
            "--series",
            help="Tilt-series to report on; repeat for several. Omitted: every series with "
            "a manifest in the project.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            help="PDF to write. Omitted: <project>/reports/report.pdf, or "
            "report-<variant>.pdf with --variant."
        ),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option(
            help="Lane TOML to append to the report verbatim, so that the PDF says what "
            "was run."
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing PDF."),
    ] = False,
) -> None:
    """Write a PDF for biologists: each series' tomogram, CTF, membranes and picks."""

    try:
        from cryoet_pipeline.reports.pdf import write_report
    except ImportError as exc:
        raise typer.BadParameter(
            f"the report draws with matplotlib, which is not installed ({exc}); install the "
            "viz extra: pip install -e '.[viz]'",
            param_hint="project",
        ) from exc
    from cryoet_pipeline.reports.collect import load_series

    registry_path = project / "artifacts.json"
    if not registry_path.is_file():
        raise typer.BadParameter(f"no registry at {registry_path}", param_hint="project")
    manifests = [
        TiltSeriesManifest.model_validate_json(path.read_text())
        for path in sorted((project / "manifests").glob("*.json"))
    ]
    if series:
        known = {manifest.tilt_series_id for manifest in manifests}
        unknown = sorted(set(series) - known)
        if unknown:
            raise typer.BadParameter(
                f"no manifest for {', '.join(unknown)} in {project / 'manifests'}",
                param_hint="series",
            )
        manifests = [manifest for manifest in manifests if manifest.tilt_series_id in series]
    if not manifests:
        raise typer.BadParameter(f"no manifests in {project / 'manifests'}", param_hint="project")
    destination = out or project / "reports" / (
        f"report-{variant}.pdf" if variant else "report.pdf"
    )
    if destination.exists() and not overwrite:
        raise typer.BadParameter(
            f"{destination} exists; pass --overwrite to replace it", param_hint="out"
        )

    artifact_registry = ArtifactRegistry.load(registry_path)
    reports = []
    for manifest in manifests:
        typer.echo(f"reading {manifest.tilt_series_id}")
        resolved = _report_artifacts(artifact_registry, manifest, variant=variant)
        reports.append(load_series(artifact_registry, resolved))
    typer.echo("drawing")
    outcome = write_report(
        destination,
        reports,
        project=project.resolve().name,
        variant=variant,
        written=datetime.now().strftime("%Y-%m-%d %H:%M"),
        software=_software_description(),
        config=(config.name, config.read_text()) if config is not None else None,
    )
    typer.echo(f"wrote {outcome.path} ({outcome.pages} pages)")
    if outcome.problems:
        typer.echo(f"{len(outcome.problems)} parts could not be shown:")
        for problem in outcome.problems:
            typer.echo(f"  {problem}")


def _report_artifacts(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None,
) -> SeriesArtifacts:
    """What one series' report pages are drawn from, by the lookups every command uses.

    The picks by label, and the rest through the picks' own lineage, as
    visualize-picks does: picks narrowed by filter-picks carry a label of their
    own that the tomogram they were found in does not. Without picks, the
    tomogram by label, so that a series which stopped early still shows how far
    it got. A lookup that fails becomes a line in the report instead of an
    error, so one ambiguous series does not cost the others their pages.
    """

    problems: list[str] = []

    def attempt(what: str, lookup: Callable[[], Any]) -> Any:
        try:
            return lookup()
        except typer.BadParameter as exc:
            problems.append(f"{what}: {exc.message}")
            return None

    def registered(kind: ArtifactKind) -> bool:
        return any(
            artifact.covers(manifest.tilt_series_id) for artifact in registry.by_kind(kind)
        )

    picks = (
        attempt("picks", lambda: _picks_artifact(registry, manifest, variant=variant))
        if registered(ArtifactKind.PICKS)
        else None
    )
    if picks is not None:
        tomogram = attempt(
            "tomogram",
            lambda: _tomogram_from_picks(registry, picks, manifest, variant=variant),
        )
        templates = _template_from_picks(registry, picks) or attempt(
            "template", lambda: _template_artifacts(registry, manifest, variant=variant)
        )
        segmentations = registry.ancestors_of_kind(picks.id, ArtifactKind.SEGMENTATION)
        segmentation = (
            segmentations[0]
            if len(segmentations) == 1
            else attempt(
                "membranes",
                lambda: _optional_segmentation_artifact(registry, manifest, variant=variant),
            )
        )
    else:
        tomogram = (
            attempt("tomogram", lambda: _tomogram_artifact(registry, manifest, variant=variant))
            if registered(ArtifactKind.TOMOGRAM)
            else None
        )
        templates = attempt(
            "template", lambda: _template_artifacts(registry, manifest, variant=variant)
        )
        segmentation = attempt(
            "membranes",
            lambda: _optional_segmentation_artifact(registry, manifest, variant=variant),
        )
    if (
        segmentation is not None
        and tomogram is not None
        and not registry.descends_from(segmentation.id, tomogram.id)
    ):
        problems.append(
            f"membranes: {segmentation.id} was not segmented from {tomogram.id}, so it is "
            "not drawn over it"
        )
        segmentation = None

    # The fit the tomogram was corrected with, when it was; otherwise the series'
    # fit by label, as picking finds it for CTF weighting.
    fitted = (
        registry.ancestors_of_kind(tomogram.id, ArtifactKind.CTF_MODEL)
        if tomogram is not None
        else []
    )
    ctf_model = fitted[0] if fitted else None
    if ctf_model is None and registered(ArtifactKind.CTF_MODEL):
        ctf_model = attempt(
            "CTF", lambda: _ctf_model_artifact(registry, manifest, variant=variant)
        )
    return SeriesArtifacts(
        manifest=manifest,
        picks=picks,
        tomogram=tomogram,
        template=templates[0] if templates else None,
        template_mask=templates[1] if templates else None,
        segmentation=segmentation,
        ctf_model=ctf_model,
        problems=tuple(problems),
    )


def _software_description() -> str:
    """This package's version and, from a git checkout, the commit it was run from."""

    try:
        description = f"cryoet-pipeline {package_version('cryoet-pipeline')}"
    except PackageNotFoundError:
        description = "cryoet-pipeline"
    source = str(Path(__file__).resolve().parent)
    try:
        commit = subprocess.run(
            ["git", "-C", source, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        changes = subprocess.run(
            ["git", "-C", source, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return description
    if commit.returncode == 0 and commit.stdout.strip():
        description += f", commit {commit.stdout.strip()}"
        if changes.returncode == 0 and changes.stdout.strip():
            description += " with uncommitted changes"
    return description


@app.command("extract-particles")
def extract_particles(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for extracted particles.")],
    box_size_px: Annotated[
        int,
        typer.Option(min=1, help="Cubic subvolume side length, in tomogram voxels."),
    ],
    backend: Annotated[
        str,
        typer.Option(help="Particle extraction backend to use."),
    ] = "cubic-crop",
    minimum_score: Annotated[
        float | None,
        typer.Option(help="Drop picks with rlnLCCmax below this value."),
    ] = None,
    pick_template_box_px: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Template box pytom matched with, for picks that do not record their "
            "pytom job (external STAR files). An even box puts each pick half a voxel "
            "above its particle's centre; extraction corrects that and will not guess it.",
        ),
    ] = None,
    tomogram_mrc: Annotated[
        Path | None,
        typer.Option(
            help="BYPASS (temporary): use this external MRC tomogram instead of looking "
            "one up in the registry. For testing against tomograms reconstructed "
            "outside this pipeline."
        ),
    ] = None,
    picks_star: Annotated[
        Path | None,
        typer.Option(
            help="BYPASS (temporary): use this external pytom-match-pick STAR file "
            "instead of looking one up in the registry."
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the particle stack, QC, and registry entries."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
) -> None:
    """Crop cubic subvolumes from a tomogram at picked particle locations."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    if tomogram_mrc is None and picks_star is None:
        # The picks are found by label; the tomogram comes from what the picks
        # themselves recorded rather than from a second lookup on the same label.
        # One `--variant` cannot select both: picks made by a parameter sweep carry
        # the sweep's label while the tomogram they were found in keeps the lane's,
        # so asking for one label found the picks and lost the tomogram. Following
        # the recorded lineage cannot mismatch them, and needs no extra flag.
        picks = _picks_artifact(artifact_registry, tilt_series_manifest, variant=variant)
        tomogram = _tomogram_from_picks(
            artifact_registry, picks, tilt_series_manifest, variant=variant
        )
    else:
        # BYPASS: an external input is resolved tomogram-first, because external
        # picks are told which tomogram they refer to rather than read from it.
        if tomogram_mrc is not None:
            tomogram = import_external_tomogram_for_extraction(
                mrc_path=tomogram_mrc,
                tilt_series_id=tilt_series_manifest.tilt_series_id,
                out=out,
            )
            artifact_registry.add(tomogram, replace=overwrite)
        else:
            tomogram = _tomogram_artifact(
                artifact_registry, tilt_series_manifest, variant=variant
            )
        if picks_star is not None:
            picks = import_external_picks_for_extraction(
                star_path=picks_star,
                tomogram=tomogram,
                tilt_series_id=tilt_series_manifest.tilt_series_id,
                template_box_px=pick_template_box_px,
            )
            artifact_registry.add(picks, replace=overwrite)
        else:
            picks = _picks_artifact(
                artifact_registry, tilt_series_manifest, variant=variant
            )

    selected_backend = _particle_extraction_backend(backend)
    parameters: dict[str, object] = {
        "variant": variant,
        "overwrite": overwrite,
        "box_size_px": box_size_px,
    }
    if minimum_score is not None:
        parameters["minimum_score"] = minimum_score
    if pick_template_box_px is not None:
        parameters["pick_template_box_px"] = pick_template_box_px
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = extract_and_register(
        selected_backend,
        tomogram,
        picks,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    particle_stack, qc_report = artifacts
    report = _read_qc_report(qc_report)
    typer.echo(f"wrote particle stack: {particle_stack.path}")
    _echo_extraction_summary(report)
    typer.echo(f"wrote extraction QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    _echo_qc_warnings(report)
    typer.echo(f"updated {registry}")


@app.command("average-particles")
def average_particles(
    manifest: Annotated[
        list[Path],
        typer.Option(
            help="Tilt-series manifest JSON written by init. Repeat it to average "
            "particles from several tilt-series together, which is what averaging "
            "is for; one manifest behaves exactly as before."
        ),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for the average map.")],
    backend: Annotated[
        str,
        typer.Option(
            help="Subtomogram averaging backend: dynamo-search (Dynamo's local search "
            "and Dynamo's project averaging; one class), dynamo-average (Dynamo's "
            "dynamo_average at the stacks' own poses: the map before any refinement, "
            "no search), cone-search (cone orientation search, multi-reference "
            "classification) or fft-refinement (simple baseline)."
        ),
    ] = "dynamo-search",
    num_classes: Annotated[
        int,
        typer.Option(min=1, help="[cone-search] Number of references to refine simultaneously."),
    ] = 1,
    cone_range_deg: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search, dynamo-search] Full aperture of the orientation cone searched "
            "first. Start near the angular step used for template matching. Default: 30 "
            "(cone-search), 15 (dynamo-search).",
        ),
    ] = None,
    cone_sampling_deg: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search, dynamo-search] Step within the cone, first iteration or "
            "level. Default: 10 (cone-search), 5 (dynamo-search).",
        ),
    ] = None,
    inplane_range_deg: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search, dynamo-search] Full in-plane search range. Default: 30 "
            "(cone-search), 15 (dynamo-search).",
        ),
    ] = None,
    inplane_sampling_deg: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search, dynamo-search] In-plane step. Default: 10 (cone-search), "
            "5 (dynamo-search).",
        ),
    ] = None,
    shift_range_px: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search, dynamo-search] Shift search half-width in pixels. cone-search "
            "(default 2) never closes it below 1 px; dynamo-search (default 3) re-centres "
            "it on each refinement level's winner, as Dynamo does.",
        ),
    ] = None,
    refine_factor: Annotated[
        float | None,
        typer.Option(
            min=1.0,
            help="[cone-search, dynamo-search] Divisor applied to the ranges and steps each "
            "iteration (cone-search) or refinement level (dynamo-search). Default: 2.",
        ),
    ] = None,
    refine_levels: Annotated[
        int | None,
        typer.Option(
            min=0,
            max=4,
            help="[dynamo-search] Refinement levels per iteration after the first grid. "
            "Default: 2.",
        ),
    ] = None,
    search_highpass_px: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[dynamo-search] High-pass radius of the search band in Fourier pixels "
            "(box * spacing / radius is the resolution in A). Default: 2.",
        ),
    ] = None,
    search_lowpass_px: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[dynamo-search] Low-pass radius of the search band in Fourier pixels; "
            "at most half the box. Default: 16 (30 A for a 48-voxel box at 10.012 A).",
        ),
    ] = None,
    gold_standard: Annotated[
        bool,
        typer.Option(
            "--gold-standard/--no-gold-standard",
            help="[dynamo-search] Refine the two halves set by rlnRandomSubset independently "
            "from one start band-passed to --gold-standard-start-lowpass-angstrom, as "
            "Dynamo's half-set projects do; write both half maps and their FSC to QC. The "
            "map is every particle at the pose its own half gave it. Default: off.",
        ),
    ] = False,
    gold_standard_start_lowpass_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[dynamo-search, --gold-standard] Low-pass of the halves' common start in "
            "A, applied as Dynamo's dynamo_bandpass [1 low 2]. Default: 60.",
        ),
    ] = None,
    shift_search_mode: Annotated[
        int | None,
        typer.Option(
            min=1,
            max=2,
            help="[dynamo-search] Dynamo's limm (area_search_modus): 2 re-centres the "
            "shift sphere on each level's winner, 1 keeps it on the box centre. "
            "Default: 2.",
        ),
    ] = None,
    workers: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="[dynamo-search] Processes that search particles in parallel; the poses "
            "do not depend on it. Default: the CPUs this job was given "
            "(SLURM_CPUS_PER_TASK, else the process's CPU affinity).",
        ),
    ] = None,
    particle_diameter_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search] Particle diameter in A, used for the Crowther angular "
            "increment degrees(resolution / diameter). [dynamo-search] The default search "
            "mask diameter. Defaults to the full box.",
        ),
    ] = None,
    target_resolution_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search] Target resolution in A for that increment. Defaults to the "
            "Nyquist limit of the particle pixel size.",
        ),
    ] = None,
    alignment_mask_diameter_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search] Enable reference-frame weighted NCC inside a soft sphere "
            "of this diameter in A (flat to 80% radius, cosine edge). Default: no mask. "
            "The mask plus shift window must fit inside the box. [dynamo-search] Diameter "
            "of the binary search mask, which also normalises the particles before "
            "averaging; default: the particle diameter. Output maps are not masked.",
        ),
    ] = None,
    alignment_mask: Annotated[
        Path | None,
        typer.Option(
            help="[dynamo-search] Alignment mask file (MRC, the particle box), used as "
            "Dynamo uses one: its voxels above 0.5 are searched, and its nonzero voxels "
            "normalise each particle before averaging. Instead of "
            "--alignment-mask-diameter-angstrom.",
        ),
    ] = None,
    alignment_frame: Annotated[
        AlignmentFrame,
        typer.Option(
            help="[cone-search] Scoring grid: reference (legacy) or particle (opt-in). "
            "Particle keeps the raw data and wedge fixed and rotates an unfiltered "
            "reference before matched-band scoring. Requires --missing-wedge and "
            "--shift-range-px 0; spatial masks are not supported in this mode.",
        ),
    ] = AlignmentFrame.REFERENCE,
    missing_wedge: Annotated[
        bool,
        typer.Option(
            "--missing-wedge/--no-missing-wedge",
            help="[cone-search] Restrict the alignment score to the frequencies each "
            "particle actually measured, at the candidate orientation. Default: off, "
            "which is the whole-box score that was MEASURED to move exactly-correct "
            "orientations by a mean of 5.9 deg on noiseless wedge-limited particles. "
            "Off is the default only so that runs already on disk stay comparable; the "
            "QC report says so either way. Cannot be combined with an alignment mask.",
        ),
    ] = False,
    wedge_min_tilt_deg: Annotated[
        float | None,
        typer.Option(
            help="[cone-search] Override the lowest tilt angle used, in degrees. "
            "Defaults to the minimum in each series' manifest, which includes views "
            "that alignment may later have excluded.",
        ),
    ] = None,
    wedge_max_tilt_deg: Annotated[
        float | None,
        typer.Option(help="[cone-search] Override the highest tilt angle used, in degrees."),
    ] = None,
    wedge_tilt_axis_deg: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="[cone-search] Tilt-axis direction in the tomogram xy plane, measured "
            "from y. An IMOD-aligned stack puts the axis on y, so zero; it is a flag "
            "rather than an assumption because a tomogram from elsewhere need not.",
        ),
    ] = 0.0,
    fourier_compensation: Annotated[
        bool,
        typer.Option(
            "--fourier-compensation/--no-fourier-compensation",
            help="[cone-search] Build each reference as sum(W*F)/sum(W) over the "
            "particles' own wedges instead of a plain mean, so a frequency measured by "
            "few particles is not left attenuated relative to one measured by many. "
            "Requires --missing-wedge. Default: off.",
        ),
    ] = False,
    seed: Annotated[
        int,
        typer.Option(help="[cone-search] Seed for the initial random class partition."),
    ] = 0,
    num_iterations: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Number of refinement iterations. Default: 1 (dynamo-search), 5 (others).",
        ),
    ] = None,
    initial_angular_step_deg: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="First-iteration local rotation search step, in degrees (halves each "
            "iteration).",
        ),
    ] = 6.0,
    initial_shift_step_px: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="First-iteration local shift search step, in pixels (halves each iteration).",
        ),
    ] = 2.0,
    initial_low_pass_angstrom: Annotated[
        float,
        typer.Option(
            # No `min=0.0`: zero is not a resolution, and advertising it as the
            # floor is how the search-step options came to promise a value the
            # backend then refused. The backend states the real constraint.
            help="[fft-refinement] Resolution the reference is filtered to in the first "
            "iteration. Later iterations tighten it from the FSC between the two "
            "half-sets, and never loosen it past this value. 60 A is customary; pick it "
            "for the binning, not for the resolution you hope for.",
        ),
    ] = 60.0,
    low_pass_margin_shells: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="[fft-refinement] How many Fourier shells beyond the measured "
            "agreement the reference is allowed to keep. Too small and the estimate can "
            "never improve; too large and the refinement reaches into shells where only "
            "noise agrees.",
        ),
    ] = 2.0,
    initial_reference: Annotated[
        Path | None,
        typer.Option(
            help="Optional starting reference map (e.g. the picking template); also "
            "enables a rot/tilt/psi convention self-check reported in QC."
        ),
    ] = None,
    initial_reference_correlation_threshold: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Self-check threshold: particles are expected to correlate with the "
            "initial reference above this value before any refinement.",
        ),
    ] = 0.3,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the average map, QC, and registry entries."),
    ] = False,
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
    output_variant: Annotated[
        str | None,
        typer.Option(
            help="Label the OUTPUT with this instead of --variant, while still "
            "selecting inputs by --variant. Without it the two are the same string, "
            "so two averaging settings over one set of particle stacks cannot coexist: "
            "a new label finds no inputs, and the old one overwrites the first result. "
            "Averaging the same particles twice is what a parameter sweep is."
        ),
    ] = None,
    class_number: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="[fft-refinement] Average only the particles classification put in "
            "this class. The step between classifying into K classes and refining the "
            "one that is not junk — without it the class assignment is written and "
            "read by nothing.",
        ),
    ] = None,
    particle_metadata: Annotated[
        Path | None,
        typer.Option(
            help="[fft-refinement] Take per-particle metadata from this STAR instead "
            "of the stacks' own. Point it at the refined STAR a classification "
            "wrote to refine one of its classes: the class assignment lives there "
            "and never in the stacks, whose extraction STAR says class 1 for "
            "everything, so --class-number alone cannot see it.",
        ),
    ] = None,
) -> None:
    """Iteratively refine per-particle orientation/shift and average subtomograms."""

    if not manifest:
        raise typer.BadParameter("at least one --manifest is required", param_hint="manifest")
    if alignment_frame is AlignmentFrame.PARTICLE:
        if backend != "cone-search" or not missing_wedge or shift_range_px != 0:
            raise typer.BadParameter(
                "particle frame requires cone-search, --missing-wedge and --shift-range-px 0",
                param_hint="alignment-frame",
            )
        if alignment_mask_diameter_angstrom is not None:
            raise typer.BadParameter(
                "particle frame does not support an alignment mask", param_hint="alignment-frame"
            )
    dynamo = backend in ("dynamo-search", "dynamo-average")
    if alignment_mask_diameter_angstrom is not None and not (dynamo or backend == "cone-search"):
        raise typer.BadParameter(
            "an alignment mask is supported only by cone-search, dynamo-search and "
            "dynamo-average",
            param_hint="alignment-mask-diameter-angstrom",
        )
    if alignment_mask is not None and not dynamo:
        raise typer.BadParameter(
            "an alignment mask file is supported only by dynamo-search and dynamo-average",
            param_hint="alignment-mask",
        )
    if backend == "dynamo-average":
        # Refused rather than ignored: a grid given to an average that does not
        # search would look as though it had been used.
        searched = {
            "num-iterations": num_iterations,
            "cone-range-deg": cone_range_deg,
            "cone-sampling-deg": cone_sampling_deg,
            "inplane-range-deg": inplane_range_deg,
            "inplane-sampling-deg": inplane_sampling_deg,
            "shift-range-px": shift_range_px,
            "refine-factor": refine_factor,
            "initial-reference": initial_reference,
        }
        for hint, value in searched.items():
            if value is not None:
                raise typer.BadParameter(
                    f"--{hint} is not read by dynamo-average, which averages at the "
                    "stacks' own poses without searching",
                    param_hint=hint,
                )
    if alignment_mask is not None and alignment_mask_diameter_angstrom is not None:
        raise typer.BadParameter(
            "give the alignment mask as a file or as a diameter, not both",
            param_hint="alignment-mask",
        )
    if backend == "dynamo-search" and (missing_wedge or fourier_compensation):
        raise typer.BadParameter(
            "dynamo-search always models the wedge and always averages with Fourier "
            "compensation; drop --missing-wedge and --fourier-compensation",
            param_hint="missing-wedge",
        )
    if missing_wedge and backend != "cone-search":
        raise typer.BadParameter(
            "missing-wedge weighting is supported only by cone-search",
            param_hint="missing-wedge",
        )
    if fourier_compensation and backend != "cone-search":
        raise typer.BadParameter(
            "Fourier compensation is supported only by cone-search",
            param_hint="fourier-compensation",
        )
    dynamo_only = {
        "refine-levels": refine_levels,
        "search-highpass-px": search_highpass_px,
        "search-lowpass-px": search_lowpass_px,
        "shift-search-mode": shift_search_mode,
        "gold-standard-start-lowpass-angstrom": gold_standard_start_lowpass_angstrom,
        "workers": workers,
    }
    if gold_standard and backend != "dynamo-search":
        raise typer.BadParameter(
            "--gold-standard is read only by dynamo-search; fft-refinement always splits",
            param_hint="gold-standard",
        )
    if gold_standard_start_lowpass_angstrom is not None and not gold_standard:
        raise typer.BadParameter(
            "the start low-pass is read only with --gold-standard",
            param_hint="gold-standard-start-lowpass-angstrom",
        )
    for hint, value in dynamo_only.items():
        if value is not None and backend != "dynamo-search":
            raise typer.BadParameter(
                f"--{hint} is read only by dynamo-search", param_hint=hint
            )
    if dynamo and num_classes != 1:
        raise typer.BadParameter(
            f"{backend} makes one class; use cone-search to classify",
            param_hint="num-classes",
        )
    if fourier_compensation and not missing_wedge:
        raise typer.BadParameter(
            "Fourier compensation divides by the wedges, so it needs --missing-wedge",
            param_hint="fourier-compensation",
        )
    if (
        (wedge_min_tilt_deg is not None or wedge_max_tilt_deg is not None)
        and not missing_wedge
        and not dynamo
    ):
        raise typer.BadParameter(
            "a wedge tilt range is read only by dynamo-search and dynamo-average, or "
            "with --missing-wedge",
            param_hint="wedge-min-tilt-deg",
        )
    tilt_series_manifests = [
        TiltSeriesManifest.model_validate_json(path.read_text()) for path in manifest
    ]
    seen = [one.tilt_series_id for one in tilt_series_manifests]
    duplicates = sorted({name for name in seen if seen.count(name) > 1})
    if duplicates:
        raise typer.BadParameter(
            f"the same tilt-series was given more than once: {', '.join(duplicates)}; "
            "each series contributes its particles once",
            param_hint="manifest",
        )
    artifact_registry = ArtifactRegistry.load(registry)
    particle_stacks = [
        _particle_stack_artifact(artifact_registry, one, variant=variant)
        for one in tilt_series_manifests
    ]
    selected_backend = _subtomogram_averaging_backend(backend)
    parameters: dict[str, object] = {
        # Inputs were resolved by `variant` above; the backend only ever uses this
        # to label what it writes, so the two can differ.
        "variant": output_variant if output_variant is not None else variant,
        "overwrite": overwrite,
        "initial_angular_step_deg": initial_angular_step_deg,
        "initial_shift_step_px": initial_shift_step_px,
        "initial_low_pass_angstrom": initial_low_pass_angstrom,
        "low_pass_margin_shells": low_pass_margin_shells,
        "initial_reference_correlation_threshold": initial_reference_correlation_threshold,
        "num_classes": num_classes,
        "class_number": class_number,
        "particle_metadata_path": particle_metadata,
        "seed": seed,
        "alignment_mask_diameter_angstrom": alignment_mask_diameter_angstrom,
        "alignment_frame": alignment_frame.value,
        "missing_wedge": missing_wedge,
        "wedge_tilt_axis_deg": wedge_tilt_axis_deg,
        "fourier_compensation": fourier_compensation,
    }
    # Passed only when given, so each backend applies its own defaults: a grid
    # tuned for cone-search is not a default for Dynamo's search.
    for key, value in {
        "num_iterations": num_iterations,
        "cone_range_deg": cone_range_deg,
        "cone_sampling_deg": cone_sampling_deg,
        "inplane_range_deg": inplane_range_deg,
        "inplane_sampling_deg": inplane_sampling_deg,
        "shift_range_px": shift_range_px,
        "refine_factor": refine_factor,
        "refine_levels": refine_levels,
        "search_highpass_px": search_highpass_px,
        "search_lowpass_px": search_lowpass_px,
        "shift_search_mode": shift_search_mode,
        "workers": workers,
    }.items():
        if value is not None:
            parameters[key] = value
    if wedge_min_tilt_deg is not None:
        parameters["wedge_min_tilt_deg"] = wedge_min_tilt_deg
    if wedge_max_tilt_deg is not None:
        parameters["wedge_max_tilt_deg"] = wedge_max_tilt_deg
    if particle_diameter_angstrom is not None:
        parameters["particle_diameter_angstrom"] = particle_diameter_angstrom
    if target_resolution_angstrom is not None:
        parameters["target_resolution_angstrom"] = target_resolution_angstrom
    if initial_reference is not None:
        parameters["initial_reference_path"] = initial_reference
    if alignment_mask is not None:
        parameters["alignment_mask_path"] = alignment_mask
    if gold_standard:
        parameters["gold_standard"] = True
    if gold_standard_start_lowpass_angstrom is not None:
        parameters["gold_standard_start_lowpass_angstrom"] = gold_standard_start_lowpass_angstrom
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = average_and_register(
        selected_backend,
        particle_stacks,
        tilt_series_manifests,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.commit(registry)

    *average_maps, qc_report = artifacts
    report = _read_qc_report(qc_report)
    for average_map in average_maps:
        class_number = average_map.parameters.get("class_number")
        label = f" (class {class_number})" if class_number is not None else ""
        typer.echo(f"wrote subtomogram average{label}: {average_map.path}")
        for subset, half_path in sorted(
            (average_map.parameters.get("half_map_paths") or {}).items()
        ):
            typer.echo(f"  half-set {subset}: {half_path}")
    _echo_averaging_summary(report)
    typer.echo(f"wrote averaging QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    _echo_qc_warnings(report)
    typer.echo(f"updated {registry}")


@app.command("compute-fsc")
def compute_fsc_command(
    mode: Annotated[
        str,
        typer.Option(help="FSC mode: half-set or map-to-model."),
    ],
    out: Annotated[Path, typer.Option(help="Output path for the FSC report JSON.")],
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Correlation threshold for the resolution crossing (default: 0.143 for "
            "half-set, 0.5 for map-to-model)."
        ),
    ] = None,
    particle_stack: Annotated[
        Path | None,
        typer.Option(help="[half-set] Particle stack Zarr from extract-particles."),
    ] = None,
    metadata: Annotated[
        Path | None,
        typer.Option(help="[half-set] Refined particle metadata STAR from average-particles."),
    ] = None,
    pixel_spacing_angstrom: Annotated[
        float | None,
        typer.Option(help="[half-set] Pixel spacing, angstrom."),
    ] = None,
    average_map: Annotated[
        Path | None,
        typer.Option(help="[map-to-model] Final average map MRC."),
    ] = None,
    model_map: Annotated[
        Path | None,
        typer.Option(help="[map-to-model] Reference/template map MRC to compare against."),
    ] = None,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    plot: Annotated[
        Path | None,
        typer.Option(help="Optional PNG path: FSC (y) vs spatial frequency, 1/A (x)."),
    ] = None,
    mask: Annotated[
        Path | None,
        typer.Option(
            help="Optional real-space mask MRC (same box size), applied to both volumes "
            "before computing FSC, so background noise isn't counted."
        ),
    ] = None,
) -> None:
    """Compute a Fourier shell correlation curve: half-set or map-to-model."""

    normalized_mode = mode.lower()
    torch_device = torch_device_for_preference(resolve_device(device))

    if normalized_mode == "half-set":
        if particle_stack is None or metadata is None or pixel_spacing_angstrom is None:
            raise typer.BadParameter(
                "half-set mode requires --particle-stack, --metadata, and "
                "--pixel-spacing-angstrom",
                param_hint="mode",
            )
        volume_a = build_half_set_average(
            particle_stack_path=particle_stack,
            metadata_path=metadata,
            half_set=1,
            device=torch_device,
        )
        volume_b = build_half_set_average(
            particle_stack_path=particle_stack,
            metadata_path=metadata,
            half_set=2,
            device=torch_device,
        )
        resolved_pixel_spacing = pixel_spacing_angstrom
        resolved_threshold = threshold if threshold is not None else 0.143
    elif normalized_mode == "map-to-model":
        if average_map is None or model_map is None:
            raise typer.BadParameter(
                "map-to-model mode requires --average-map and --model-map",
                param_hint="mode",
            )
        volume_a, voxel_size_a = load_mrc_volume(average_map)
        volume_b, voxel_size_b = load_mrc_volume(model_map)
        if not math.isclose(voxel_size_a, voxel_size_b, rel_tol=1e-3):
            raise typer.BadParameter(
                f"--average-map voxel size {voxel_size_a:.4f} does not match --model-map "
                f"voxel size {voxel_size_b:.4f}",
                param_hint="model_map",
            )
        volume_a = volume_a.to(torch_device)
        volume_b = volume_b.to(torch_device)
        if volume_a.shape != volume_b.shape:
            # Same voxel size, different box: the template is built at the size
            # its density needs and the average at the extraction box, so these
            # routinely differ — 39 against 48 for a ribosome here. Padding the
            # smaller one centrally is the correct reconciliation because it
            # leaves the pixel size untouched. Across odd/even parity a
            # half-voxel translation also puts both geometric centres together.
            volume_a, volume_b, padded = _pad_to_common_box(volume_a, volume_b)
            typer.echo(
                f"padded the smaller map from {padded} to {tuple(volume_a.shape)} so the "
                "shells line up; the voxel size is unchanged"
            )
            if (volume_a.shape[0] - padded[0]) % 2:
                typer.echo(
                    "centred-grid correction: translated the smaller map by +0.5 voxel "
                    "on each axis using Fourier phases (real Nyquist convention)"
                )
        resolved_pixel_spacing = (
            pixel_spacing_angstrom if pixel_spacing_angstrom is not None else voxel_size_a
        )
        resolved_threshold = threshold if threshold is not None else 0.5
    else:
        raise typer.BadParameter(
            f"unsupported FSC mode {mode!r}; expected: half-set, map-to-model",
            param_hint="mode",
        )

    if mask is not None:
        mask_tensor = load_mask(mask, expected_shape=tuple(volume_a.shape)).to(torch_device)
        volume_a = volume_a * mask_tensor
        volume_b = volume_b * mask_tensor

    report = compute_and_write_fsc_report(
        volume_a,
        volume_b,
        mode=normalized_mode,
        pixel_spacing_angstrom=resolved_pixel_spacing,
        threshold=resolved_threshold,
        out=out,
    )

    if plot is not None:
        plot_fsc_curve(
            report.resolution_angstrom,
            report.correlation,
            threshold=report.threshold,
            crossing_resolution_angstrom=report.crossing_resolution_angstrom,
            out=plot,
            title=f"FSC ({normalized_mode})",
        )
        typer.echo(f"wrote FSC plot: {plot}")

    typer.echo(f"wrote FSC report: {out}")
    if report.crossing_resolution_angstrom is not None:
        typer.echo(
            f"resolution at threshold {resolved_threshold:.3f}: "
            f"{report.crossing_resolution_angstrom:.2f} A"
        )
    else:
        typer.echo(f"correlation never dropped below threshold {resolved_threshold:.3f}")
    typer.echo(f"QC status: {report.status.value}")


def _parse_map_spec(spec: str) -> tuple[str, Path]:
    """`LABEL=PATH`, or a bare `PATH` labelled by its stem; `\\n` in a label breaks the line."""

    label, separator, path = spec.partition("=")
    if not separator:
        return Path(spec).stem, Path(spec)
    if not label or not path:
        raise typer.BadParameter(f"expected LABEL=PATH, got {spec!r}", param_hint="map")
    return label.replace("\\n", "\n"), Path(path)


@app.command("compare-maps")
def compare_maps_command(
    map_specs: Annotated[
        list[str],
        typer.Option(
            "--map",
            help="A map to show, as LABEL=PATH (or just PATH, labelled by its file name). "
            "Repeat once per column, in display order. '\\n' in a label starts a new line.",
        ),
    ],
    out: Annotated[Path, typer.Option(help="Output image; the format follows the suffix.")],
    protein_bright: Annotated[
        list[str] | None,
        typer.Option(
            "--protein-bright",
            help="Label (or path) of a map whose protein is positive — EMDB maps and "
            "simulated templates usually are. Inverted so protein is dark everywhere. "
            "Repeatable.",
        ),
    ] = None,
    planes: Annotated[
        str, typer.Option(help="Comma-separated central planes: any of xy, xz, yz.")
    ] = "xy,xz",
    low_pass: Annotated[
        list[float] | None,
        typer.Option(
            "--low-pass",
            help="Repeat every plane low-passed at this resolution, angstrom. Repeatable.",
        ),
    ] = None,
    sphere_diameter: Annotated[
        float | None,
        typer.Option(
            help="Diameter of the normalisation sphere, angstrom (default: inscribed in the box)."
        ),
    ] = None,
    sigma_min: Annotated[float, typer.Option(help="Black point, in sigma.")] = -3.0,
    sigma_max: Annotated[float, typer.Option(help="White point, in sigma.")] = 1.5,
    slab: Annotated[
        float, typer.Option(help="Slab thickness averaged into each section, voxels.")
    ] = 1.0,
    title: Annotated[
        str | None, typer.Option(help="Figure title (default: describes the display).")
    ] = None,
    voxel_size: Annotated[
        float | None,
        typer.Option(
            help="Voxel size, angstrom, for maps whose header has none (Dynamo writes 0). "
            "Maps that do carry one must agree with it."
        ),
    ] = None,
) -> None:
    """Show several maps (template, averages, classes) sliced and scaled identically."""

    from cryoet_pipeline.map_comparison import ComparedMap, load_map, render_map_comparison

    parsed = [_parse_map_spec(spec) for spec in map_specs]
    bright = {name.replace("\\n", "\n") for name in protein_bright or []}
    known = {label for label, _ in parsed} | {str(path) for _, path in parsed}
    unknown = bright - known
    if unknown:
        raise typer.BadParameter(
            f"--protein-bright names no --map: {', '.join(sorted(unknown))}",
            param_hint="protein_bright",
        )

    loaded = [(label, path, *load_map(path)) for label, path in parsed]
    headers = [(path, size) for _, path, _, size in loaded if size is not None]
    if voxel_size is None:
        if len(headers) < len(loaded):
            missing = [str(path) for _, path, _, size in loaded if size is None]
            raise typer.BadParameter(
                f"no voxel size in the header of {', '.join(missing)}; pass --voxel-size",
                param_hint="voxel_size",
            )
        reference_path, voxel_size = headers[0]
    else:
        reference_path = Path("--voxel-size")
    for path, size in headers:
        if not math.isclose(size, voxel_size, rel_tol=1e-3):
            raise typer.BadParameter(
                f"{path} has voxel size {size:.4f} A but {reference_path} has "
                f"{voxel_size:.4f} A; resample one of them first",
                param_hint="map",
            )
    target = max(int(volume.shape[0]) for _, _, volume, _ in loaded)

    import torch

    maps = []
    for label, path, volume, _ in loaded:
        if int(volume.shape[0]) < target:
            # Same centring as compute-fsc, including the half-voxel Fourier
            # shift across odd/even parity, so this figure and an FSC between the
            # same files compare the same superposition.
            volume, _, original = _pad_to_common_box(volume, torch.zeros((target,) * 3))
            typer.echo(f"padded {label!r} from {original} to {tuple(volume.shape)}")
        maps.append(
            ComparedMap(
                label=label,
                volume=volume,
                protein_bright=label in bright or str(path) in bright,
            )
        )

    try:
        render_map_comparison(
            maps,
            out=out,
            voxel_size_angstrom=voxel_size,
            planes=[plane.strip() for plane in planes.split(",") if plane.strip()],
            low_pass_angstrom=low_pass or [],
            sphere_diameter_angstrom=sphere_diameter,
            sigma_range=(sigma_min, sigma_max),
            slab_voxels=slab,
            title=title,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"wrote map comparison: {out}")


@app.command()
def run(
    project: Annotated[Path, typer.Option(help="Project directory created by init.")],
) -> None:
    """Run the pipeline for the configured tilt-series."""

    raise typer.BadParameter(
        f"pipeline execution is not implemented yet; project={project}",
        param_hint="project",
    )


@app.command()
def export(
    project: Annotated[Path, typer.Option(help="Project directory.")],
    format: Annotated[str, typer.Option(help="Export format.")] = "imod",
) -> None:
    """Export artifacts for external tools."""

    raise typer.BadParameter(
        f"export is not implemented yet; project={project}, format={format}",
        param_hint="project",
    )


@app.command()
def qc(project: Annotated[Path, typer.Option(help="Project directory.")]) -> None:
    """Generate or inspect QC outputs."""

    raise typer.BadParameter(
        f"QC is not implemented yet; project={project}",
        param_hint="project",
    )


def _motioncor3_cli_parameters(
    *,
    executable: Path | None,
    gpu: int,
    patch_x: int,
    patch_y: int,
    pixel_size: float | None,
    gain: Path | None,
    gain_rotation: int,
    gain_flip: int,
    ftbin: int = 1,
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "motioncor3_gpu_ids": [gpu],
        "motioncor3_patch_x": patch_x,
        "motioncor3_patch_y": patch_y,
        "motioncor3_ftbin": ftbin,
        "motioncor3_gain_rotation": gain_rotation,
        "motioncor3_gain_flip": gain_flip,
    }
    if executable is not None:
        parameters["motioncor3_executable"] = executable
    if pixel_size is not None:
        parameters["motioncor3_pixel_size_angstrom"] = pixel_size
    if gain is not None:
        parameters["motioncor3_gain_reference"] = gain
    return parameters


def _motion_backend(name: str) -> MotionCorrectionBackend:
    normalized = name.lower()
    if normalized == "average":
        return AverageMotionCorrectionBackend()
    if normalized in ("phase-corr", "phase-corr"):
        return PhaseCorrelationMotionCorrectionBackend()
    if normalized in ("motioncor3", "motioncor-3"):
        return MotionCor3MotionCorrectionBackend()

    raise typer.BadParameter(
        "unsupported motion-correction backend "
        f"{name!r}; expected: average, phase-corr, motioncor3",
        param_hint="backend",
    )


def _alignment_backend(name: str) -> TiltAlignmentBackend:
    normalized = name.lower()
    if normalized == "imod-xcorr":
        return ImodTiltXcorrAlignmentBackend()

    raise typer.BadParameter(
        f"unsupported tilt-alignment backend {name!r}; expected: imod-xcorr",
        param_hint="backend",
    )


def _coarse_alignment_qc_backend(name: str) -> CoarseAlignmentQcBackend:
    normalized = name.lower()
    if normalized == "imod":
        return ImodCoarseAlignmentQcBackend()

    raise typer.BadParameter(
        f"unsupported coarse-alignment QC backend {name!r}; expected: imod",
        param_hint="backend",
    )


def _fiducial_seed_backend(name: str) -> FiducialSeedBackend:
    normalized = name.lower()
    if normalized == "imod-autofidseed":
        return ImodAutofidseedBackend()
    if normalized == "imod-patch-seed":
        return ImodPatchSeedBackend()

    raise typer.BadParameter(
        f"unsupported fiducial seed backend {name!r}; "
        "expected: imod-autofidseed, imod-patch-seed",
        param_hint="backend",
    )


def _fiducial_tracking_backend(name: str) -> FiducialTrackingBackend:
    normalized = name.lower()
    if normalized == "imod-beadtrack":
        return ImodBeadtrackBackend()
    if normalized == "imod-patch-track":
        return ImodPatchTrackBackend()

    raise typer.BadParameter(
        f"unsupported fiducial tracking backend {name!r}; "
        "expected: imod-beadtrack, imod-patch-track",
        param_hint="backend",
    )


# Which tracking options each backend actually reads. Bead tracking iterates and
# has rounds; patch tracking is a single tiltxcorr correlation pass and instead
# exposes that correlation's bandpass.
_TRACKING_BACKEND_OPTIONS: dict[str, frozenset[str]] = {
    "imod-beadtrack": frozenset({"--rounds"}),
    "imod-patch-track": frozenset(
        {
            "--patch-filter-sigma1",
            "--patch-filter-radius2",
            "--patch-filter-sigma2",
        }
    ),
}


def _reject_foreign_tracking_options(
    backend_name: str,
    supplied: dict[str, tuple[str, object]],
) -> None:
    """Refuse a backend-specific option the chosen backend would ignore.

    Accepting one silently is worse than refusing it: the run looks configured
    and behaves exactly as if the flag had never been passed.
    """

    owned = _TRACKING_BACKEND_OPTIONS.get(backend_name, frozenset())
    for flag, (_, value) in supplied.items():
        if value is None or flag in owned:
            continue
        owner = next(
            (name for name, flags in _TRACKING_BACKEND_OPTIONS.items() if flag in flags),
            None,
        )
        raise typer.BadParameter(
            f"{flag} is read by {owner}, not by {backend_name}",
            param_hint=flag,
        )


def _fine_alignment_backend(name: str) -> FineAlignmentBackend:
    normalized = name.lower()
    if normalized == "imod-tiltalign":
        return ImodTiltalignBackend()

    raise typer.BadParameter(
        f"unsupported fine-alignment backend {name!r}; expected: imod-tiltalign",
        param_hint="backend",
    )


def _final_aligned_stack_backend(name: str) -> FinalAlignedStackBackend:
    normalized = name.lower()
    if normalized == "imod-newstack":
        return ImodFinalAlignedStackBackend()

    raise typer.BadParameter(
        f"unsupported final aligned-stack backend {name!r}; expected: imod-newstack",
        param_hint="backend",
    )


def _defocus_estimation_backend(name: str) -> DefocusEstimationBackend:
    normalized = name.lower()
    if normalized == "ctffind5":
        return Ctffind5DefocusHintBackend()

    raise typer.BadParameter(
        f"unsupported defocus-hint backend {name!r}; expected: ctffind5",
        param_hint="backend",
    )


def _ctf_estimation_backend(name: str) -> CtfEstimationBackend:
    normalized = name.lower()
    if normalized == "ctffind5":
        return Ctffind5CtfEstimationBackend()
    if normalized == "imod-ctfplotter":
        return ImodCtfPlotterEstimationBackend()

    raise typer.BadParameter(
        f"unsupported CTF estimation backend {name!r}; expected: ctffind5, imod-ctfplotter",
        param_hint="backend",
    )


def _ctf_correction_backend(name: str) -> CtfCorrectionBackend:
    normalized = name.lower()
    if normalized == "imod-ctfphaseflip":
        return ImodCtfPhaseFlipBackend()

    raise typer.BadParameter(
        f"unsupported CTF correction backend {name!r}; expected: imod-ctfphaseflip",
        param_hint="backend",
    )


@dataclass(frozen=True)
class _CtfWeighting:
    """What pytom needs to weight a template like the tomogram it searches.

    Phase flipping corrects the sign of the transfer function but not its
    amplitude, so a corrected tomogram still carries |CTF| while a deposited
    reference map does not. Correlating the two is a mismatched filter: the
    template contributes frequencies the data never recorded. These values let
    pytom apply the same weighting to the template.
    """

    defocus_path: Path
    voltage_kv: float
    spherical_aberration_mm: float
    amplitude_contrast: float
    model: str
    z_values: list[int] | None = None
    """The z value of each row of the defocus file, so that picking can take the
    rows of the views the tomogram kept."""


def _tilt_angles_for_tomogram(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    tomogram: Artifact,
) -> Path | None:
    """The solved tilt angles this tomogram was actually reconstructed from.

    The manifest lists every image that was collected, but fine alignment
    excludes some, so the manifest and the reconstruction disagree on how many
    tilts there are. Template matching models the missing wedge from these
    angles, so handing it the collected set would describe a tilt the data never
    contributed.

    The tomogram's own record comes first: reconstruction writes the angles it
    used. An ancestor stack may hold more views than the tomogram, since
    correct-ctf can leave out views CTFFIND did not fit.
    """

    own = tomogram.parameters.get("tilt_angles_path")
    if isinstance(own, (str, Path)) and Path(own).is_file():
        return Path(own)
    stacks = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.covers(manifest.tilt_series_id)
        and registry.descends_from(tomogram.id, artifact.id)
    ]
    for stack in stacks:
        value = stack.parameters.get("tilt_file_path")
        if isinstance(value, (str, Path)) and Path(value).is_file():
            return Path(value)
    return None


def _ctf_weighting_for_tomogram(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    tomogram: Artifact,
) -> _CtfWeighting | None:
    """Derive the weighting from the tomogram's own lineage, or None.

    The corrected stack records which CTF model corrected it, so the values used
    here are the ones the correction actually used rather than a second set that
    could disagree with it.
    """

    corrected = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("purpose") == "ctf_corrected"
        and registry.descends_from(tomogram.id, artifact.id)
    ]
    if not corrected:
        return None
    model_id = corrected[0].parameters.get("ctf_model_id")
    if not isinstance(model_id, str):
        return None
    try:
        model = registry.get(model_id)
    except KeyError:
        return None
    optics = {
        key: model.parameters.get(key)
        for key in ("voltage_kv", "spherical_aberration_mm", "amplitude_contrast")
    }
    if not all(isinstance(value, (int, float)) for value in optics.values()):
        return None
    if not model.path.is_file():
        return None
    z_values = model.parameters.get("z_values")
    return _CtfWeighting(
        defocus_path=model.path,
        voltage_kv=float(cast(float, optics["voltage_kv"])),
        spherical_aberration_mm=float(cast(float, optics["spherical_aberration_mm"])),
        amplitude_contrast=float(cast(float, optics["amplitude_contrast"])),
        model="phase-flip",
        z_values=(
            list(z_values)
            if isinstance(z_values, list)
            and all(isinstance(z, int) and not isinstance(z, bool) for z in z_values)
            else None
        ),
    )


def _ctf_model_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    backend: str | None = None,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.CTF_MODEL)
        if artifact.covers(manifest.tilt_series_id)
        and (backend is None or artifact.parameters.get("backend") == backend)
    ]
    if variant is not None and matches:
        labelled = [artifact for artifact in matches if _labelled(artifact, variant)]
        matches = labelled or [artifact for artifact in matches if _label_of(artifact) is None]
    if not matches:
        available = sorted(
            str(artifact.parameters.get("backend"))
            for artifact in registry.by_kind(ArtifactKind.CTF_MODEL)
            if artifact.covers(manifest.tilt_series_id)
        )
        detail = f"; registered estimation backends: {', '.join(available)}" if available else ""
        raise typer.BadParameter(
            f"no CTF model for {manifest.tilt_series_id}"
            f"{f' from backend {backend!r}' if backend else ''}{detail}",
            param_hint="registry",
        )
    if len(matches) > 1:
        backends = ", ".join(
            sorted(str(artifact.parameters.get("backend")) for artifact in matches)
        )
        raise typer.BadParameter(
            f"found {len(matches)} CTF models for {manifest.tilt_series_id}; "
            f"pass --ctf-model-backend to choose one of: {backends}",
            param_hint="registry",
        )
    return matches[0]


def _reconstruction_backend(name: str) -> ReconstructionBackend:
    normalized = name.lower()
    if normalized == "imod-tilt":
        return ImodTiltReconstructionBackend()

    raise typer.BadParameter(
        f"unsupported reconstruction backend {name!r}; expected: imod-tilt",
        param_hint="backend",
    )


def _restoration_backend(name: str) -> DenoisingBackend:
    normalized = name.lower()
    if normalized == "isonet2":
        return IsoNet2RestorationBackend()

    raise typer.BadParameter(
        f"unsupported restoration backend {name!r}; expected: isonet2",
        param_hint="backend",
    )


def _segmentation_backend(name: str) -> SegmentationBackend:
    normalized = name.lower()
    if normalized == "membrain-seg":
        return MemBrainSegSegmentationBackend()

    raise typer.BadParameter(
        f"unsupported segmentation backend {name!r}; expected: membrain-seg",
        param_hint="backend",
    )


def _picking_backend(name: str) -> PickingBackend:
    normalized = name.lower()
    if normalized == "pytom-match-pick":
        return PytomMatchPickBackend()

    raise typer.BadParameter(
        f"unsupported picking backend {name!r}; expected: pytom-match-pick",
        param_hint="backend",
    )


def _template_preparation_backend(name: str) -> TemplatePreparationBackend:
    normalized = name.lower()
    if normalized == "pytom":
        return PytomTemplateBackend()

    raise typer.BadParameter(
        f"unsupported template preparation backend {name!r}; expected: pytom",
        param_hint="backend",
    )


def _template_artifacts(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> tuple[Artifact, Artifact] | None:
    """Return the registered template and its mask, if a pair is present."""

    templates = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TEMPLATE)
        if artifact.covers(manifest.tilt_series_id)
    ]
    if not templates:
        return None
    template = _select_variant(
        templates,
        what="template",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )
    masks = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TEMPLATE_MASK)
        if artifact.parameters.get("template_id") == template.id
    ]
    if len(masks) != 1:
        raise typer.BadParameter(
            f"expected exactly one template mask for {template.id}, found {len(masks)}",
            param_hint="registry",
        )
    return template, masks[0]


def _template_from_picks(
    registry: ArtifactRegistry, picks: Artifact
) -> tuple[Artifact, Artifact] | None:
    """The template and mask these picks were matched with, from their lineage.

    Picks matched against template files given on the command line have none in
    their lineage. They still record the files, and when those are a registered
    template and its mask, under whatever label, that pair is the answer: the
    pick ablation makes its templates under labels of their own, which a lookup
    by the picks' label cannot find.

    None when neither way names exactly one template with exactly one mask, so
    the caller falls back to looking them up by label.
    """

    templates = registry.ancestors_of_kind(picks.id, ArtifactKind.TEMPLATE)
    recorded_mask: Path | None = None
    if not templates:
        recorded = picks.parameters.get("pytom_template_path")
        mask_path = picks.parameters.get("pytom_template_mask_path")
        if not isinstance(recorded, str) or not isinstance(mask_path, str):
            return None
        target = Path(recorded).resolve()
        templates = [
            artifact
            for artifact in registry.by_kind(ArtifactKind.TEMPLATE)
            if artifact.path.resolve() == target
        ]
        recorded_mask = Path(mask_path).resolve()
    if len(templates) != 1:
        return None
    masks = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TEMPLATE_MASK)
        if artifact.parameters.get("template_id") == templates[0].id
    ]
    if len(masks) != 1:
        return None
    if recorded_mask is not None and masks[0].path.resolve() != recorded_mask:
        # The template's own mask is not the one the match used.
        return None
    return templates[0], masks[0]


def _particle_extraction_backend(name: str) -> ParticleExtractionBackend:
    normalized = name.lower()
    if normalized == "cubic-crop":
        return CubicCropExtractionBackend()

    raise typer.BadParameter(
        f"unsupported particle extraction backend {name!r}; expected: cubic-crop",
        param_hint="backend",
    )


def _subtomogram_averaging_backend(name: str) -> SubtomogramAveragingBackend:
    normalized = name.lower()
    if normalized == "fft-refinement":
        return FftRefinementAveragingBackend()
    if normalized == "cone-search":
        return ConeSearchRefinementBackend()
    if normalized == "dynamo-search":
        return DynamoSearchRefinementBackend()
    if normalized == "dynamo-average":
        return DynamoAverageBackend()

    raise typer.BadParameter(
        f"unsupported subtomogram averaging backend {name!r}; "
        "expected: dynamo-search, dynamo-average, cone-search, fft-refinement",
        param_hint="backend",
    )


def _optional_segmentation_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact | None:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.SEGMENTATION)
        if artifact.covers(manifest.tilt_series_id)
    ]
    if not matches:
        return None
    return _select_variant(
        matches,
        what="segmentation",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _tilt_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TILT_STACK)
        if artifact.covers(manifest.tilt_series_id)
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one tilt stack for {manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _label_of(artifact: Artifact) -> str | None:
    """The lane label this artifact carries, if any.

    New artifacts record it as a field. Ones written before the field existed
    carry it only as the last segment of their id, which cannot be recovered
    without knowing the canonical form — so for those, absence of the field is
    read as "shared" and `_labelled` matches on the id instead.
    """

    return artifact.variant


def _labelled(artifact: Artifact, variant: str) -> bool:
    return artifact.variant == variant or (
        artifact.variant is None and artifact.id.endswith(f":{variant}")
    )


def _pad_to_common_box(first: Any, second: Any) -> tuple[Any, Any, tuple[int, ...]]:
    # Annotated `Any` rather than `torch.Tensor` because this module imports torch
    # lazily on purpose — a missing optional dependency must not break unrelated
    # commands — and a module-level annotation would defeat that.
    """Pad the smaller cube onto the larger cube's centred grid; say which grew.

    Centred so that a map whose density sits at the box centre — which both of
    these do — stays at the centre. Zeros because outside a template's own box
    there is no density, and because the maps are mean-centred before an FSC
    anyway, so zero is the neutral value rather than an invented one. An odd
    size difference needs a half-voxel translation after integer padding. Use
    Fourier phases, not linear interpolation (which would low-pass the map).
    This is a grid-origin correction, not a fitted registration to the template.
    """

    import torch

    if first.shape == second.shape:
        return first, second, tuple(first.shape)
    smaller, larger = (first, second) if first.shape[0] < second.shape[0] else (second, first)
    original = tuple(smaller.shape)
    target = larger.shape[0]
    if any(size != smaller.shape[0] for size in smaller.shape) or any(
        size != target for size in larger.shape
    ):
        raise typer.BadParameter(
            f"only cubic boxes can be padded to match, got {tuple(first.shape)} and "
            f"{tuple(second.shape)}",
            param_hint="model_map",
        )
    pad = target - smaller.shape[0]
    before = pad // 2
    padded = torch.nn.functional.pad(
        smaller, (before, pad - before) * 3, mode="constant", value=0.0
    )
    if pad % 2:
        # Integer padding leaves the density centre half a voxel below the
        # target centre on each axis. Translate by +0.5 on the padded grid.
        spectrum = torch.fft.fftn(padded)
        frequencies = torch.fft.fftfreq(target, device=padded.device, dtype=padded.dtype)
        phase = torch.exp(-1j * torch.pi * frequencies)
        if target % 2 == 0:
            # The unpaired Nyquist coefficient must stay real. At a half-pixel
            # offset its cosine sample is zero; do not create complex density.
            phase[target // 2] = 0
        for axis in range(3):
            shape = [1, 1, 1]
            shape[axis] = target
            spectrum = spectrum * phase.reshape(shape)
        padded = torch.fft.ifftn(spectrum).real
    if first.shape[0] < second.shape[0]:
        return padded, second, original
    return first, padded, original


def _select_variant(
    matches: list[Artifact],
    *,
    what: str,
    tilt_series_id: str,
    variant: str | None,
    shared_fallback: bool = False,
) -> Artifact:
    """Narrow a lookup to one artifact, using the variant label when it applies.

    Runs that should coexist rather than replace each other carry a `--variant`
    label. When several candidates remain, the error lists them so the caller can
    see what to pass.

    A label marks where one lane parts from another, so at the parting point the
    inputs are still the shared ones and carry no label yet: requiring them to
    carry it would make the divergence itself impossible to express — labelling
    a rebuild at a different binning would demand a labelled alignment that only
    a labelled rebuild could have produced. So the lookup prefers a labelled
    candidate and otherwise falls back to the unlabelled artifact every lane
    shares. It never falls back to another lane's, so lanes cannot be crossed.

    `shared_fallback` extends that to a lookup carrying no label at all, for the
    stage whose input is the shared artifact by design: CTF is fitted on a stack
    built without a label so that any lane can use the model. Everywhere else,
    several candidates and no label stays an error — that refusal is what lets a
    second reconstruction coexist instead of being chosen by accident.
    """

    if variant is not None:
        labelled = [artifact for artifact in matches if _labelled(artifact, variant)]
        matches = labelled or [artifact for artifact in matches if _label_of(artifact) is None]
    elif shared_fallback and len(matches) > 1:
        # Only where an unlabelled lookup means the artifact every lane shares
        # rather than "whichever one exists". Everywhere else two candidates and
        # no label is a real ambiguity, and reporting it is what lets a second
        # reconstruction coexist instead of being picked by accident.
        shared = [artifact for artifact in matches if _label_of(artifact) is None]
        if len(shared) == 1:
            matches = shared
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise typer.BadParameter(
            f"no {what} for {tilt_series_id}"
            + (f" with variant {variant!r}" if variant else ""),
            param_hint="registry",
        )
    available = ", ".join(sorted(artifact.id for artifact in matches))
    raise typer.BadParameter(
        f"found {len(matches)} candidates for {what} of {tilt_series_id}; "
        f"pass --variant to choose one of: {available}",
        param_hint="registry",
    )


def _tomogram_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("tomogram_branch", "full") == "full"
    ]
    return _select_variant(
        matches,
        what="full tomogram",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _tomogram_from_picks(
    registry: ArtifactRegistry,
    picks: Artifact,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    """The tomogram these picks were found in, from the picks' own record.

    Preferred over a second label-based lookup because the two cannot then
    disagree: extraction crops from the volume the coordinates refer to, and any
    other volume would put every box in the wrong place. Falls back to the label
    only for picks written before the id was recorded.
    """

    recorded = picks.parameters.get("input_tomogram_id")
    if isinstance(recorded, str):
        try:
            return registry.get(recorded)
        except (KeyError, ValueError) as exc:
            raise typer.BadParameter(
                f"{picks.id} was found in tomogram {recorded!r}, which is not in the "
                "registry. The coordinates refer to that volume, and cropping from "
                "another would place every box wrongly",
                param_hint="registry",
            ) from exc
    return _tomogram_artifact(registry, manifest, variant=variant)


def _picks_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    """The picks for this series, narrowed by variant like every other lookup.

    This used to demand exactly one and ignore `--variant`, so the moment a
    second labelled pick list existed for a series — which is what
    `--output-variant` exists to produce — both extraction and the ChimeraX
    export refused, with a count and no way to choose. Labelling an output is
    only useful if something downstream can select by the label.
    """

    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PICKS)
        if artifact.covers(manifest.tilt_series_id)
    ]
    return _select_variant(
        matches,
        what="picks",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _particle_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PARTICLE_STACK)
        if artifact.covers(manifest.tilt_series_id)
    ]
    if not matches:
        raise typer.BadParameter(
            f"no particle stack for {manifest.tilt_series_id}", param_hint="registry"
        )
    return _select_variant(
        matches,
        what="particle stack",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _final_aligned_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
    shared_fallback: bool = False,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("purpose") == "final_alignment"
    ]
    return _select_variant(
        matches,
        what="final aligned stack",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
        shared_fallback=shared_fallback,
    )


def _reconstruction_input_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
    from_stack: str = "auto",
) -> Artifact:
    """Choose which stack to reconstruct from.

    `auto` prefers a CTF-corrected stack when one exists, which is almost always
    what is wanted and was the only behaviour available. It could not be
    overridden, so once a corrected stack was registered there was no way to
    reconstruct without CTF — not even to compare the two. `aligned` and
    `ctf-corrected` name the route outright and fail rather than substitute.
    """

    corrected_matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("purpose") == "ctf_corrected"
    ]
    if from_stack == "aligned":
        return _final_aligned_stack_artifact(registry, manifest, variant=variant)
    if from_stack == "ctf-corrected" and not corrected_matches:
        raise typer.BadParameter(
            f"no CTF-corrected stack for {manifest.tilt_series_id}; run correct-ctf "
            "first, or pass --from-stack aligned to reconstruct without CTF correction",
            param_hint="from-stack",
        )
    if corrected_matches:
        return _select_variant(
            corrected_matches,
            what="CTF-corrected stack",
            tilt_series_id=manifest.tilt_series_id,
            variant=variant,
        )
    return _final_aligned_stack_artifact(registry, manifest, variant=variant)


def _zero_tilt_corrected_projection_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    id_prefix = f"{manifest.tilt_series_id}:corrected_projection:"
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.CORRECTED_PROJECTION)
        if artifact.id.startswith(id_prefix)
    ]
    if not matches:
        raise typer.BadParameter(
            f"expected at least one corrected projection for {manifest.tilt_series_id}, found 0",
            param_hint="registry",
        )

    def _tilt_angle(artifact: Artifact) -> float:
        value = artifact.parameters.get("tilt_angle_deg")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise typer.BadParameter(
                f"corrected projection {artifact.id} is missing a numeric tilt_angle_deg",
                param_hint="registry",
            )
        return float(value)

    return min(matches, key=lambda artifact: abs(_tilt_angle(artifact)))


def _optional_defocus_hint_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact | None:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.DEFOCUS_HINT)
        if artifact.covers(manifest.tilt_series_id)
    ]
    if len(matches) > 1:
        raise typer.BadParameter(
            f"expected at most one defocus hint for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0] if matches else None


def _fiducial_tracking_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("purpose") == "fiducial_tracking"
    ]
    return _select_variant(
        matches,
        what="fiducial tracking stack",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _fiducial_seed_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.FIDUCIAL_SEED_MODEL)
        if artifact.covers(manifest.tilt_series_id)
    ]
    return _select_variant(
        matches,
        what="fiducial seed model",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _fiducial_model_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.FIDUCIAL_MODEL)
        if artifact.covers(manifest.tilt_series_id)
    ]
    return _select_variant(
        matches,
        what="tracked fiducial model",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _derived_low_pass_angstrom(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> tuple[float, str]:
    """The matching low-pass from this series' own alignment, or a refusal.

    Nothing derived this before, so a run that omitted `--pytom-low-pass` matched
    across the whole band out to Nyquist. That is not an error anywhere: it
    produces a flat score volume, an extraction cut-off flush against its
    maximum, and a pick list of two with `status: pass`. Deriving it removes the
    guess; refusing when it cannot be derived removes the silence.

    Refusing rather than falling back to Nyquist is the point. A default that
    happens to be wrong is indistinguishable from one that happens to be right
    until someone counts the picks, and by then half an hour of GPU is spent.
    """

    from cryoet_pipeline.resolution import low_pass_from_alignment_residual

    try:
        _fine_alignment_artifact(registry, manifest, variant=variant)
    except typer.BadParameter as exc:
        raise typer.BadParameter(
            "--pytom-low-pass was not given and cannot be derived: no fine alignment "
            f"is registered for {manifest.tilt_series_id} ({exc}). Matching across the "
            "whole band to Nyquist is what returns a handful of picks from a flat "
            "score volume, so pass --pytom-low-pass explicitly instead",
            param_hint="pytom_low_pass",
        ) from exc

    # The residual lives in the QC report, and the alignment artifact does not
    # carry a path to it — so it is found as its own artifact, by kind and type.
    # Going through the registry rather than adding a `qc_path` parameter is what
    # lets this work on projects already on disk, which is the whole point: five
    # series are aligned already and nobody should re-run them for a field.
    reports = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.QC)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("qc_type") == "fine_alignment"
    ]
    residual = None
    for candidate in reports:
        if candidate.path.is_file():
            value = json.loads(candidate.path.read_text()).get("residual_mean_nm")
            if isinstance(value, (int, float)) and value > 0.0:
                residual = value
                break
    if not isinstance(residual, (int, float)) or residual <= 0.0:
        raise typer.BadParameter(
            "--pytom-low-pass was not given and cannot be derived: the fine alignment "
            f"for {manifest.tilt_series_id} records no usable residual_mean_nm. Pass "
            "--pytom-low-pass explicitly",
            param_hint="pytom_low_pass",
        )
    return low_pass_from_alignment_residual(float(residual)), f"{residual:.3f} nm"


def _fine_alignment_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    variant: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNMENT)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("stage") == "fine"
    ]
    return _select_variant(
        matches,
        what="fine alignment",
        tilt_series_id=manifest.tilt_series_id,
        variant=variant,
    )


def _alignment_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNMENT)
        if artifact.covers(manifest.tilt_series_id)
        and artifact.parameters.get("stage") == "coarse"
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one coarse alignment for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]
