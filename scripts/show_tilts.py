#!/usr/bin/env python3
"""Draw a few raw tilts into a PDF: are there gold beads, and which way does the tilt axis run.

    python3 scripts/show_tilts.py <frames folder> --mdocs <mdoc folder>
    python3 scripts/show_tilts.py <frames folder> --pattern '<regular expression>'
    python3 scripts/show_tilts.py <frames folder> --pattern '...' --series 039 --series 096
    python3 scripts/show_tilts.py --stack TS_1002.mrc --mdoc TS_1002.mrc.mdoc

For a site without 3dmod, before a lane exists: it answers lane_questions'
question 3 (beads or patches, and the bead size) and question 7 (`[align]
tilt_axis_angle`) from the tilts themselves. It reads the folder the way
`cryoet init` does (find_tilt_series, load_tilt_series_manifest,
validate_tilt_series_manifest), or a stack the way import-tilt-stack does
(check_external_tilt_stack), so the arguments are the ones show_folder.py's
fragment holds.

Per series it reads at most nine tilts: the one nearest 0°, the lowest, the
highest, and the four highest on each side of 0°, which give the axis. Frames
are added up without gain or motion correction. What it measures and how is
cryoet_pipeline.tilt_preview's docstring; the pages are
cryoet_pipeline.reports.tilts.

Without --series or --all, only the first series is read: on K3 TIFF movies a
series is about nine gigabytes to read.

Writes only the PDF, by default outputs/tilt-previews/<folder>_<series>.pdf, or
<series>.pdf for a stack.
Exit status: 0 when every series asked for was drawn; 1 when one was refused or
failed (the PDF still holds the others); 2 for arguments that cannot work.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

from pydantic import ValidationError

from cryoet_pipeline.external_stack import check_external_tilt_stack
from cryoet_pipeline.models import ProjectConfig
from cryoet_pipeline.project import (
    find_tilt_series,
    load_tilt_series_manifest,
    validate_tilt_series_manifest,
)
from cryoet_pipeline.reports.tilts import (
    beads_line,
    measured_axis_line,
    recorded_axis_line,
    write_tilt_preview,
)
from cryoet_pipeline.tilt_preview import SeriesPreview, preview_series

ROOT = Path(__file__).resolve().parents[1]
# init never writes here; ProjectConfig only needs a path.
_NO_OUTPUT = Path(os.devnull)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("frames", nargs="?", type=Path,
                        help="the folder of movies, as [data] frames")
    parser.add_argument("--mdocs", type=Path, help="the folder of SerialEM mdocs")
    parser.add_argument("--pattern", help="the movie names' pattern, as [data] frames_pattern")
    parser.add_argument("--pixel", type=float,
                        help="raw pixel size in Å, for movies whose headers have none")
    parser.add_argument("--stack", type=Path, help="an assembled tilt stack (MRC)")
    parser.add_argument("--mdoc", type=Path, help="the stack's mdoc")
    parser.add_argument("--series", action="append", default=[],
                        help="a series to read; repeat for more (a stack's id: its name)")
    parser.add_argument("--all", action="store_true", help="read every series found")
    parser.add_argument("--out", type=Path, help="the PDF to write")
    return parser.parse_args(argv)


def _problem(args: argparse.Namespace) -> str | None:
    if args.stack is not None:
        if args.frames is not None or args.mdocs or args.pattern:
            return "give a frames folder or --stack with --mdoc, not both"
        if args.mdoc is None:
            return "--stack needs --mdoc: the tilt angles come from it"
        if len(args.series) > 1 or args.all:
            return "a stack is one series"
        return None
    if args.frames is None:
        return "give a frames folder (with --mdocs or --pattern) or --stack with --mdoc"
    if (args.mdocs is None) == (args.pattern is None):
        return "give exactly one of --mdocs and --pattern, as [data] does"
    if args.mdoc is not None:
        return "--mdoc goes with --stack; a frames folder takes --mdocs"
    if args.all and args.series:
        return "--all reads every series; --series names some: not both"
    return None


def _print(line: str) -> None:
    print(line, flush=True)


def _software() -> str:
    try:
        description = f"cryoet-pipeline {package_version('cryoet-pipeline')}"
    except PackageNotFoundError:
        description = "cryoet-pipeline"
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        commit = None
    if commit is not None and commit.returncode == 0 and commit.stdout.strip():
        description += f" at {commit.stdout.strip()}"
    return description + ", scripts/show_tilts.py"


def summary(preview: SeriesPreview) -> list[str]:
    angles = preview.angles_deg
    return [
        f"{preview.series}: {len(angles)} tilts, {min(angles):+.1f} to {max(angles):+.1f} deg, "
        f"{preview.raw_pixel_angstrom:.4g} A/px, read {len(preview.pictures)} of them",
        f"  gold beads: {beads_line(preview)}",
        f"  tilt axis from the tilts: {measured_axis_line(preview)}",
        f"  tilt axis the mdoc gives: {recorded_axis_line(preview)}",
        *(f"  note: {note}" for note in preview.notes),
    ]


def read_frames(args: argparse.Namespace, previews: list[SeriesPreview],
                failures: list[str]) -> str:
    try:
        config = ProjectConfig(
            frames_dir=args.frames,
            mdocs_dir=args.mdocs,
            frames_pattern=args.pattern,
            raw_pixel_spacing_angstrom=args.pixel,
            output_dir=_NO_OUTPUT,
            tilt_series=args.series,
        )
        found = args.series or find_tilt_series(config)
    except (ValidationError, ValueError, OSError) as error:
        failures.append(f"refused, as init would refuse it: {error}")
        return args.frames.name
    if not found:
        failures.append("no series found, as init would find none")
        return args.frames.name
    chosen = found if (args.series or args.all) else found[:1]
    if len(chosen) < len(found):
        _print(f"reading {chosen[0]}, the first of {len(found)} series "
               f"({', '.join(found[:8])}{', ...' if len(found) > 8 else ''}); "
               "--series names others, --all reads every one")
    for series in chosen:
        try:
            manifest = load_tilt_series_manifest(config, series)
            validate_tilt_series_manifest(manifest)
            previews.append(preview_series(manifest, progress=_print))
        except (ValueError, OSError, KeyError) as error:
            failures.append(f"{series}: {error}")
    return args.frames.name


def read_stack(args: argparse.Namespace, previews: list[SeriesPreview],
               failures: list[str]) -> str:
    series = args.series[0] if args.series else args.stack.name.split(".")[0]
    try:
        manifest, _ = check_external_tilt_stack(mdoc_path=args.mdoc, stack_path=args.stack,
                                                tilt_series_id=series)
        previews.append(preview_series(manifest, stack=args.stack, progress=_print))
    except (ValueError, OSError) as error:
        failures.append(f"{series}: refused, as import-tilt-stack would refuse it: {error}")
    return series


def default_out(label: str, series: list[str]) -> Path:
    """outputs/tilt-previews/<folder>_<series>.pdf: a folder named frames is common."""

    named = "-".join(series) if len(series) <= 3 else f"{len(series)}-series"
    stem = label if series == [label] else f"{label}_{named}"
    return Path("outputs") / "tilt-previews" / f"{stem}.pdf"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problem = _problem(args)
    if problem is not None:
        print(f"show_tilts.py: {problem}", file=sys.stderr)
        return 2
    previews: list[SeriesPreview] = []
    failures: list[str] = []
    if args.stack is not None:
        label = read_stack(args, previews, failures)
    else:
        label = read_frames(args, previews, failures)

    print()
    for preview in previews:
        for line in summary(preview):
            print(line)
    for failure in failures:
        print(f"not drawn: {failure}")
    if not previews:
        return 1
    out = args.out or default_out(label, [preview.series for preview in previews])
    outcome = write_tilt_preview(out, previews, label=label,
                                 written=date.today().isoformat(), software=_software())
    for problem_line in outcome.problems:
        print(f"page not drawn: {problem_line}")
    print(f"pdf: {outcome.path} ({outcome.pages} pages)")
    return 1 if failures or outcome.problems else 0


if __name__ == "__main__":
    sys.exit(main())
