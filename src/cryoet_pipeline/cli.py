from __future__ import annotations

from pathlib import Path

import typer

from cryoet_pipeline.models import ProjectConfig

app = typer.Typer(help="cryo-ET preprocessing pipeline MVP.")


@app.command()
def init(
    frames: Path = typer.Option(..., help="Directory containing multiframe MRC files."),
    mdocs: Path = typer.Option(..., help="Directory containing SerialEM .mdoc files."),
    out: Path = typer.Option(..., help="Project output directory."),
    tilt_series: list[str] = typer.Option(
        ["TS_01", "TS_43"],
        "--tilt-series",
        help="Tilt-series ids to include in the project.",
    ),
    device: str = typer.Option("auto", help="Runtime device: auto, cuda, mps, or cpu."),
) -> None:
    """Validate and print the initial project config.

    This is a placeholder command until the ingest workflow writes project state.
    """

    config = ProjectConfig(
        frames_dir=frames,
        mdocs_dir=mdocs,
        output_dir=out,
        tilt_series=tilt_series,
        device=device,
    )
    typer.echo(config.model_dump_json(indent=2))


@app.command()
def run(project: Path = typer.Option(..., help="Project directory created by init.")) -> None:
    """Run the pipeline for the configured tilt-series."""

    raise typer.BadParameter(
        f"pipeline execution is not implemented yet; project={project}",
        param_hint="project",
    )


@app.command()
def export(
    project: Path = typer.Option(..., help="Project directory."),
    format: str = typer.Option("imod", help="Export format."),
) -> None:
    """Export artifacts for external tools."""

    raise typer.BadParameter(
        f"export is not implemented yet; project={project}, format={format}",
        param_hint="project",
    )


@app.command()
def qc(project: Path = typer.Option(..., help="Project directory.")) -> None:
    """Generate or inspect QC outputs."""

    raise typer.BadParameter(
        f"QC is not implemented yet; project={project}",
        param_hint="project",
    )
