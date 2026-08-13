from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated

import typer

from cryoet_pipeline.artiax_export import export_picks_for_artiax
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
from cryoet_pipeline.models import Artifact, ArtifactKind, ProjectConfig, TiltSeriesManifest
from cryoet_pipeline.project import initialize_project
from cryoet_pipeline.runtime import normalize_device, resolve_device
from cryoet_pipeline.storage import resolve_storage_policy

app = typer.Typer(help="cryo-ET preprocessing pipeline MVP.")

DEFAULT_INIT_TILT_SERIES = list(DEFAULT_TILT_SERIES)
DEFAULT_DOWNLOAD_TILT_SERIES = list(DEFAULT_TILT_SERIES)


@app.command()
def init(
    frames: Annotated[
        Path,
        typer.Option(help="Directory containing multiframe MRC files."),
    ],
    mdocs: Annotated[
        Path,
        typer.Option(help="Directory containing SerialEM .mdoc files."),
    ],
    out: Annotated[Path, typer.Option(help="Project output directory.")],
    tilt_series: Annotated[
        list[str],
        typer.Option("--tilt-series", help="Tilt-series ids to include in the project."),
    ] = DEFAULT_INIT_TILT_SERIES,
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
) -> None:
    """Validate local inputs and write project manifests."""

    config = ProjectConfig(
        frames_dir=frames,
        mdocs_dir=mdocs,
        output_dir=out,
        tilt_series=tilt_series,
        device=normalize_device(device),
    )
    result = initialize_project(config)
    typer.echo(f"wrote {result.project_path}")
    for path in result.manifest_paths:
        typer.echo(f"wrote {path}")


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
    artifact_registry.write(registry)

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
    artifact_registry.write(registry)

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
    artifact_registry.write(registry)

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
    artifact_registry.write(registry)

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
    artifact_registry.write(registry)

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
        int,
        typer.Option(min=1, help="Number of Beadtrack tracking rounds."),
    ] = 2,
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

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tracking_stack = _fiducial_tracking_stack_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    seed_model = _fiducial_seed_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    selected_backend = _fiducial_tracking_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "rounds": rounds,
        "min_tracked_fiducials": min_tracked_fiducials,
        "coverage_warning": coverage_warning,
        "coverage_failure": coverage_failure,
    }
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
    artifact_registry.write(registry)

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
        typer.Option(help="Overwrite fine-alignment, QC, and registry entries."),
    ] = False,
) -> None:
    """Solve fiducial fine alignment and compose final global transforms."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tracking_stack = _fiducial_tracking_stack_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    fiducial_model = _fiducial_model_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    selected_backend = _fine_alignment_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "residual_warning_nm": residual_warning_nm,
        "residual_failure_nm": residual_failure_nm,
        "residual_max_warning_tracking_px": residual_max_warning_px,
        "residual_max_failure_tracking_px": residual_max_failure_px,
        "cross_validate": cross_validate,
        "robust_fitting": robust_fitting,
        "auto_prune_outliers": auto_prune_outliers,
        "max_pruned_fraction": max_pruned_fraction,
        "max_positioning_rounds": max_positioning_rounds,
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
    artifact_registry.write(registry)

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
    )
    selected_backend = _final_aligned_stack_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
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
    artifact_registry.write(registry)

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
    artifact_registry.write(registry)

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
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the CTF model, QC, and registry entries."),
    ] = False,
) -> None:
    """Estimate per-tilt defocus and write an IMOD-format CTF model."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    aligned_stack = _final_aligned_stack_artifact(artifact_registry, tilt_series_manifest)
    alignment = _fine_alignment_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _ctf_estimation_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
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
    artifact_registry.write(registry)

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
    aligned_stack = _final_aligned_stack_artifact(artifact_registry, tilt_series_manifest)
    ctf_model = _ctf_model_artifact(
        artifact_registry,
        tilt_series_manifest,
        backend=ctf_model_backend,
    )
    selected_backend = _ctf_correction_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
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
    artifact_registry.write(registry)

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

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    aligned_stack = _reconstruction_input_stack_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    alignment = _fine_alignment_artifact(
        artifact_registry,
        tilt_series_manifest,
    )
    selected_backend = _reconstruction_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
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
    artifact_registry.write(registry)

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
) -> None:
    """Restore a reconstructed tomogram and register a canonical Zarr result."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _restoration_backend(backend)
    parameters: dict[str, object] = {
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
    artifact_registry.write(registry)

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
) -> None:
    """Segment a reconstructed tomogram and register a canonical Zarr mask."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _segmentation_backend(backend)
    parameters: dict[str, object] = {
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
    artifact_registry.write(registry)

    segmentation, qc_report = artifacts
    typer.echo(f"wrote segmentation: {segmentation.path}")
    typer.echo(f"wrote segmentation QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


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
        typer.Option(min=0.0, help="Particle diameter, A; sizes the spherical mask."),
    ] = None,
    template_mask_radius_px: Annotated[
        float | None,
        typer.Option(min=0.0, help="Mask radius in template pixels, overriding the diameter."),
    ] = None,
    template_mask_sigma_px: Annotated[
        float | None,
        typer.Option(min=0.0, help="Gaussian falloff at the mask edge, pixels."),
    ] = None,
    template_box_size_px: Annotated[
        int | None,
        typer.Option(min=1, help="Template box size; defaults to what the resampling yields."),
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
) -> None:
    """Fetch a reference map and build a matching template plus mask."""

    if (emdb_id is None) == (source_map is None):
        raise typer.BadParameter(
            "pass exactly one of --emdb-id or --source-map",
            param_hint="emdb_id",
        )

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest)

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
    artifact_registry.write(registry)

    template, mask, qc_report = artifacts
    typer.echo(f"wrote template: {template.path}")
    typer.echo(f"wrote template mask: {mask.path}")
    typer.echo(f"box size: {template.parameters['box_size_px']} px")
    typer.echo(f"voxel size: {template.pixel_spacing_angstrom} A")
    typer.echo(f"wrote template QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
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
) -> None:
    """Pick particles from a reconstructed tomogram and register a canonical STAR result."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _picking_backend(backend)
    supporting_artifacts: list[Artifact] = []
    if use_segmentation_mask:
        segmentation = _optional_segmentation_artifact(artifact_registry, tilt_series_manifest)
        if segmentation is not None:
            supporting_artifacts.append(segmentation)
    registered_template = _template_artifacts(artifact_registry, tilt_series_manifest)
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
    parameters: dict[str, object] = {
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
    if pytom_low_pass is not None:
        parameters["pytom_low_pass"] = pytom_low_pass
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
    artifact_registry.write(registry)

    picks, qc_report = artifacts
    typer.echo(f"wrote picks: {picks.path}")
    typer.echo(f"wrote picking QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("export-picks")
def export_picks(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to read."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for the export.")],
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite an existing export."),
    ] = False,
) -> None:
    """Export a tomogram and its picks for inspection in ChimeraX with ArtiaX."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest)
    picks = _picks_artifact(artifact_registry, tilt_series_manifest)

    export = export_picks_for_artiax(
        tomogram=tomogram,
        picks=picks,
        out=out,
        overwrite=overwrite,
    )

    typer.echo(f"wrote tomogram: {export.tomogram_path}")
    typer.echo(f"wrote picks: {export.picks_path}")
    typer.echo(f"wrote ChimeraX script: {export.script_path}")
    typer.echo(f"open with: chimerax {export.script_path}")


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
    device: Annotated[
        str,
        typer.Option(help="Runtime device: auto, cuda, mps, or cpu."),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Overwrite the particle stack, QC, and registry entries."),
    ] = False,
) -> None:
    """Crop cubic subvolumes from a tomogram at picked particle locations."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    tomogram = _tomogram_artifact(artifact_registry, tilt_series_manifest)
    picks = _picks_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _particle_extraction_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "box_size_px": box_size_px,
    }
    if minimum_score is not None:
        parameters["minimum_score"] = minimum_score
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
    artifact_registry.write(registry)

    particle_stack, qc_report = artifacts
    typer.echo(f"wrote particle stack: {particle_stack.path}")
    typer.echo(f"wrote extraction QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
    typer.echo(f"updated {registry}")


@app.command("average-particles")
def average_particles(
    manifest: Annotated[
        Path,
        typer.Option(help="Tilt-series manifest JSON written by init."),
    ],
    registry: Annotated[
        Path,
        typer.Option(help="Artifact registry JSON to update."),
    ],
    out: Annotated[Path, typer.Option(help="Output directory for the average map.")],
    backend: Annotated[
        str,
        typer.Option(
            help="Subtomogram averaging backend: cone-search (cone orientation search, "
            "multi-reference classification) or fft-refinement (simple baseline)."
        ),
    ] = "cone-search",
    num_classes: Annotated[
        int,
        typer.Option(min=1, help="[cone-search] Number of references to refine simultaneously."),
    ] = 1,
    cone_range_deg: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="[cone-search] Full aperture of the orientation cone searched in the first "
            "iteration. Start near the angular step used for template matching.",
        ),
    ] = 30.0,
    cone_sampling_deg: Annotated[
        float,
        typer.Option(min=0.0, help="[cone-search] Step within the cone, first iteration."),
    ] = 10.0,
    inplane_range_deg: Annotated[
        float,
        typer.Option(min=0.0, help="[cone-search] In-plane search range, first iteration."),
    ] = 30.0,
    inplane_sampling_deg: Annotated[
        float,
        typer.Option(min=0.0, help="[cone-search] In-plane step, first iteration."),
    ] = 10.0,
    shift_range_px: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="[cone-search] Shift search half-width in pixels; never closes below 1 px, "
            "since the correlation peak is located at whole-pixel accuracy.",
        ),
    ] = 2.0,
    refine_factor: Annotated[
        float,
        typer.Option(
            min=1.0,
            help="[cone-search] Divisor applied to every range and step each iteration. "
            "Two to three is the usual choice.",
        ),
    ] = 2.0,
    particle_diameter_angstrom: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            help="[cone-search] Particle diameter in A, used for the Crowther angular "
            "increment degrees(resolution / diameter). Defaults to the full box.",
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
    seed: Annotated[
        int,
        typer.Option(help="[cone-search] Seed for the initial random class partition."),
    ] = 0,
    num_iterations: Annotated[
        int,
        typer.Option(min=1, help="Number of coarse-to-fine refinement iterations."),
    ] = 5,
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
) -> None:
    """Iteratively refine per-particle orientation/shift and average subtomograms."""

    tilt_series_manifest = TiltSeriesManifest.model_validate_json(manifest.read_text())
    artifact_registry = ArtifactRegistry.load(registry)
    particle_stack = _particle_stack_artifact(artifact_registry, tilt_series_manifest)
    selected_backend = _subtomogram_averaging_backend(backend)
    parameters: dict[str, object] = {
        "overwrite": overwrite,
        "num_iterations": num_iterations,
        "initial_angular_step_deg": initial_angular_step_deg,
        "initial_shift_step_px": initial_shift_step_px,
        "initial_reference_correlation_threshold": initial_reference_correlation_threshold,
        "num_classes": num_classes,
        "cone_range_deg": cone_range_deg,
        "cone_sampling_deg": cone_sampling_deg,
        "inplane_range_deg": inplane_range_deg,
        "inplane_sampling_deg": inplane_sampling_deg,
        "shift_range_px": shift_range_px,
        "refine_factor": refine_factor,
        "seed": seed,
    }
    if particle_diameter_angstrom is not None:
        parameters["particle_diameter_angstrom"] = particle_diameter_angstrom
    if target_resolution_angstrom is not None:
        parameters["target_resolution_angstrom"] = target_resolution_angstrom
    if initial_reference is not None:
        parameters["initial_reference_path"] = initial_reference
    context = BackendContext(
        output_dir=out,
        device=resolve_device(device),
        parameters=parameters,
    )
    artifacts = average_and_register(
        selected_backend,
        particle_stack,
        tilt_series_manifest,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.write(registry)

    *average_maps, qc_report = artifacts
    for average_map in average_maps:
        class_number = average_map.parameters.get("class_number")
        label = f" (class {class_number})" if class_number is not None else ""
        typer.echo(f"wrote subtomogram average{label}: {average_map.path}")
    typer.echo(f"wrote averaging QC: {qc_report.path}")
    typer.echo(f"QC status: {qc_report.parameters['status']}")
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
            random_subset=1,
            device=torch_device,
        )
        volume_b = build_half_set_average(
            particle_stack_path=particle_stack,
            metadata_path=metadata,
            random_subset=2,
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
) -> dict[str, object]:
    parameters: dict[str, object] = {
        "motioncor3_gpu_ids": [gpu],
        "motioncor3_patch_x": patch_x,
        "motioncor3_patch_y": patch_y,
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
    if normalized in ("phase-corr", "phase_corr"):
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


def _ctf_model_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
    *,
    backend: str | None = None,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.CTF_MODEL)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and (backend is None or artifact.parameters.get("backend") == backend)
    ]
    if not matches:
        available = sorted(
            str(artifact.parameters.get("backend"))
            for artifact in registry.by_kind(ArtifactKind.CTF_MODEL)
            if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
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
) -> tuple[Artifact, Artifact] | None:
    """Return the registered template and its mask, if a pair is present."""

    templates = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TEMPLATE)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if not templates:
        return None
    if len(templates) > 1:
        raise typer.BadParameter(
            f"expected at most one template for {manifest.tilt_series_id}, "
            f"found {len(templates)}",
            param_hint="registry",
        )
    template = templates[0]
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

    raise typer.BadParameter(
        f"unsupported subtomogram averaging backend {name!r}; "
        "expected: cone-search, fft-refinement",
        param_hint="backend",
    )


def _optional_segmentation_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact | None:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.SEGMENTATION)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if len(matches) > 1:
        raise typer.BadParameter(
            f"expected at most one segmentation for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0] if matches else None


def _tilt_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TILT_STACK)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one tilt stack for {manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _tomogram_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and artifact.parameters.get("tomogram_branch", "full") == "full"
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one full tomogram for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _picks_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PICKS)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one picks artifact for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _particle_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PARTICLE_STACK)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one particle stack for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _final_aligned_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and artifact.parameters.get("purpose") == "final_alignment"
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one final aligned stack for "
            f"{manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _reconstruction_input_stack_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    """Prefer a CTF-corrected stack over the plain final aligned stack, if present."""

    corrected_matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and artifact.parameters.get("purpose") == "ctf_corrected"
    ]
    if len(corrected_matches) > 1:
        raise typer.BadParameter(
            f"expected at most one CTF-corrected stack for "
            f"{manifest.tilt_series_id}, found {len(corrected_matches)}",
            param_hint="registry",
        )
    if len(corrected_matches) == 1:
        return corrected_matches[0]
    return _final_aligned_stack_artifact(registry, manifest)


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
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
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
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and artifact.parameters.get("purpose") == "fiducial_tracking"
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one fiducial tracking stack for "
            f"{manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _fiducial_seed_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.FIDUCIAL_SEED_MODEL)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one fiducial seed model for "
            f"{manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _fiducial_model_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.FIDUCIAL_MODEL)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one tracked fiducial model for "
            f"{manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _fine_alignment_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNMENT)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and artifact.parameters.get("stage") == "fine"
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one fine alignment for "
            f"{manifest.tilt_series_id}, found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]


def _alignment_artifact(
    registry: ArtifactRegistry,
    manifest: TiltSeriesManifest,
) -> Artifact:
    matches = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.ALIGNMENT)
        if artifact.parameters.get("tilt_series_id") == manifest.tilt_series_id
        and artifact.parameters.get("stage") == "coarse"
    ]
    if len(matches) != 1:
        raise typer.BadParameter(
            f"expected exactly one coarse alignment for {manifest.tilt_series_id}, "
            f"found {len(matches)}",
            param_hint="registry",
        )
    return matches[0]
