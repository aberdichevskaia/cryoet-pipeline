from __future__ import annotations

from pathlib import Path
from typing import Annotated

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
from cryoet_pipeline.backends.fiducials import (
    ImodAutofidseedBackend,
    ImodBeadtrackBackend,
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
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    CoarseAlignmentQcBackend,
    FiducialSeedBackend,
    FiducialTrackingBackend,
    FinalAlignedStackBackend,
    FineAlignmentBackend,
    MotionCorrectionBackend,
    ReconstructionBackend,
    TiltAlignmentBackend,
)
from cryoet_pipeline.backends.reconstruction import (
    ImodTiltReconstructionBackend,
    reconstruct_and_register,
)
from cryoet_pipeline.backends.stack import SimpleTiltStackBackend, build_stack_and_register
from cryoet_pipeline.empiar import (
    DEFAULT_TILT_SERIES,
    build_empiar_10164_file_list,
    download_files,
)
from cryoet_pipeline.models import Artifact, ArtifactKind, ProjectConfig, TiltSeriesManifest
from cryoet_pipeline.project import initialize_project
from cryoet_pipeline.runtime import normalize_device, resolve_device
from cryoet_pipeline.storage import resolve_storage_policy

app = typer.Typer(help="cryo-ET preprocessing pipeline MVP.")

DEFAULT_OUTPUT_ROOT = Path("data/empiar-10164")
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


@app.command("download-empiar-10164")
def download_empiar_10164(
    out: Annotated[
        Path,
        typer.Option(help="Output root for EMPIAR-10164 files."),
    ] = DEFAULT_OUTPUT_ROOT,
    tilt_series: Annotated[
        list[str],
        typer.Option("--tilt-series", help="Tilt-series ids to download."),
    ] = DEFAULT_DOWNLOAD_TILT_SERIES,
    dry_run: Annotated[bool, typer.Option(help="Print files without downloading.")] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Redownload files even if they exist."),
    ] = False,
) -> None:
    """Download selected EMPIAR-10164 tilt-series without using browser ZIPs."""

    files = build_empiar_10164_file_list(tilt_series)
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
        typer.Option(
            help="IMOD tilt-axis angle in degrees; defaults to mdoc RotationAngle - 90."
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
                "Specimen X-axis tilt in degrees; defaults to the "
                "fine-alignment surface analysis."
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
    aligned_stack = _final_aligned_stack_artifact(
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

    raise typer.BadParameter(
        f"unsupported fiducial seed backend {name!r}; "
        "expected: imod-autofidseed",
        param_hint="backend",
    )


def _fiducial_tracking_backend(name: str) -> FiducialTrackingBackend:
    normalized = name.lower()
    if normalized == "imod-beadtrack":
        return ImodBeadtrackBackend()

    raise typer.BadParameter(
        f"unsupported fiducial tracking backend {name!r}; "
        "expected: imod-beadtrack",
        param_hint="backend",
    )


def _fine_alignment_backend(name: str) -> FineAlignmentBackend:
    normalized = name.lower()
    if normalized == "imod-tiltalign":
        return ImodTiltalignBackend()

    raise typer.BadParameter(
        f"unsupported fine-alignment backend {name!r}; "
        "expected: imod-tiltalign",
        param_hint="backend",
    )


def _final_aligned_stack_backend(name: str) -> FinalAlignedStackBackend:
    normalized = name.lower()
    if normalized == "imod-newstack":
        return ImodFinalAlignedStackBackend()

    raise typer.BadParameter(
        f"unsupported final aligned-stack backend {name!r}; "
        "expected: imod-newstack",
        param_hint="backend",
    )


def _reconstruction_backend(name: str) -> ReconstructionBackend:
    normalized = name.lower()
    if normalized == "imod-tilt":
        return ImodTiltReconstructionBackend()

    raise typer.BadParameter(
        f"unsupported reconstruction backend {name!r}; expected: imod-tilt",
        param_hint="backend",
    )


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
            f"expected exactly one tilt stack for {manifest.tilt_series_id}, "
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
