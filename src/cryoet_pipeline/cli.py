from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

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
from cryoet_pipeline.bypass_external_inputs import (
    import_external_picks_for_extraction,
    import_external_tomogram_for_extraction,
)
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
    aligned_stack = _final_aligned_stack_artifact(
        artifact_registry, tilt_series_manifest, variant=variant
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
    artifact_registry.write(registry)

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
        typer.Option(min=0.0, help="Particle diameter, A; sizes the spherical mask."),
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
    artifact_registry.write(registry)

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
    artifact_registry.write(registry)

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


@app.command("export-chimerax")
def export_chimerax(
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
    variant: Annotated[
        str | None,
        typer.Option(
            help="Label this run so it coexists with other runs of the same stage "
            "instead of replacing them, and select inputs carrying the same label. "
            "Use it to compare backends or binnings side by side."
        ),
    ] = None,
) -> None:
    """Export a tomogram and its picks for inspection in ChimeraX with ArtiaX."""

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

    registered_template = _template_artifacts(
        artifact_registry, tilt_series_manifest, variant=variant
    )
    export = export_picks_for_artiax(
        template=registered_template[0] if registered_template else None,
        segmentation=_optional_segmentation_artifact(
            artifact_registry, tilt_series_manifest, variant=variant
        ),
        tomogram=tomogram,
        picks=picks,
        out=out,
        overwrite=overwrite,
    )

    typer.echo(f"wrote tomogram: {export.tomogram_path}")
    typer.echo(f"wrote picks: {export.picks_path}")
    if export.template_path is not None:
        typer.echo(f"wrote template: {export.template_path}")
    if export.segmentation_path is not None:
        typer.echo(f"wrote segmentation: {export.segmentation_path}")
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
        "num_iterations": num_iterations,
        "initial_angular_step_deg": initial_angular_step_deg,
        "initial_shift_step_px": initial_shift_step_px,
        "initial_low_pass_angstrom": initial_low_pass_angstrom,
        "low_pass_margin_shells": low_pass_margin_shells,
        "initial_reference_correlation_threshold": initial_reference_correlation_threshold,
        "num_classes": num_classes,
        "class_number": class_number,
        "particle_metadata_path": particle_metadata,
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
        particle_stacks,
        tilt_series_manifests,
        context,
        artifact_registry,
        replace_existing=overwrite,
    )
    artifact_registry.write(registry)

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
        if volume_a.shape != volume_b.shape:
            # Same voxel size, different box: the template is built at the size
            # its density needs and the average at the extraction box, so these
            # routinely differ — 39 against 48 for a ribosome here. Padding the
            # smaller one centrally is the correct reconciliation because it
            # leaves the pixel size, and therefore every shell's resolution,
            # untouched; resampling either would not.
            volume_a, volume_b, padded = _pad_to_common_box(volume_a, volume_b)
            typer.echo(
                f"padded the smaller map from {padded} to {tuple(volume_a.shape)} so the "
                "shells line up; the voxel size is unchanged"
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
    """

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
    return _CtfWeighting(
        defocus_path=model.path,
        voltage_kv=float(cast(float, optics["voltage_kv"])),
        spherical_aberration_mm=float(cast(float, optics["spherical_aberration_mm"])),
        amplitude_contrast=float(cast(float, optics["amplitude_contrast"])),
        model="phase-flip",
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
    """Zero-pad the smaller cube to the larger, centred, and say which grew.

    Centred so that a map whose density sits at the box centre — which both of
    these do — stays at the centre. Zeros because outside a template's own box
    there is no density, and because the maps are mean-centred before an FSC
    anyway, so zero is the neutral value rather than an invented one.
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
    if first.shape[0] < second.shape[0]:
        return padded, second, original
    return first, padded, original


def _select_variant(
    matches: list[Artifact],
    *,
    what: str,
    tilt_series_id: str,
    variant: str | None,
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
    """

    if variant is not None:
        labelled = [artifact for artifact in matches if _labelled(artifact, variant)]
        matches = labelled or [artifact for artifact in matches if _label_of(artifact) is None]
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
