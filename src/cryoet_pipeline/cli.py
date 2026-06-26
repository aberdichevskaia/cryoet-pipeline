from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cryoet_pipeline.empiar import (
    DEFAULT_TILT_SERIES,
    build_empiar_10164_file_list,
    download_files,
)
from cryoet_pipeline.models import ProjectConfig
from cryoet_pipeline.project import initialize_project
from cryoet_pipeline.runtime import normalize_device

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
