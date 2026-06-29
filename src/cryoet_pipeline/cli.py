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
from cryoet_pipeline.backends.motion import (
    AverageMotionCorrectionBackend,
    correct_and_register,
)
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    CoarseAlignmentQcBackend,
    MotionCorrectionBackend,
    TiltAlignmentBackend,
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


def _motion_backend(name: str) -> MotionCorrectionBackend:
    normalized = name.lower()
    if normalized == "average":
        return AverageMotionCorrectionBackend()

    raise typer.BadParameter(
        f"unsupported motion-correction backend {name!r}; expected: average",
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
