#!/usr/bin/env python3
"""Per-view CTFFIND fits of one series, side by side across estimate-ctf variants.

    python3 scripts/ctf_fit_table.py outputs/ryr1 039 ctfw1 ctfw5 ctfw10 ctfw5-tilt

Reads <project>/ctf/<series>/<series>_ctffind5_<label>_ctffind.txt and the
.defocus file next to it, which carries the views' tilt angles in the same
order. One row per view: the tilt, then for each label the mean defocus and the
spacing CTFFIND fitted Thon rings to. A variant run with sample-tilt
determination also gets CTFFIND's own estimate of the tilt, in brackets.

Below the table, per label: how many views fit finer than --good-angstrom, and
a straight line through those views' defocus against sin(tilt). A defocus that
moves with sin(tilt) is what an image centre off the tilt axis gives, and its
slope is the number that says how wide the per-view search has to be: the fit
of a view whose defocus lies outside the window has to come back into it on its
own.

Disposable.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from cryoet_pipeline.imod_defocus import parse_ctffind_summary, read_imod_defocus_file


def _tilt_columns(summary: Path) -> list[tuple[float, float] | None]:
    """CTFFIND5's estimated (tilt axis, tilt angle), columns 8 and 9, per view."""

    estimates: list[tuple[float, float] | None] = []
    for line in summary.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [float(field) for field in line.split()]
        estimates.append((fields[7], fields[8]) if len(fields) > 8 else None)
    # Without tilt determination CTFFIND5 still writes the columns, as zeros.
    if all(estimate is None or estimate == (0.0, 0.0) for estimate in estimates):
        return [None] * len(estimates)
    return estimates


def _line(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Least-squares intercept, slope and rms residual, or None for under 3 points."""

    if len(xs) < 3:
        return None
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
    intercept = mean_y - slope * mean_x
    rms = math.sqrt(
        sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys)) / len(xs)
    )
    return intercept, slope, rms


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path)
    parser.add_argument("series")
    parser.add_argument("labels", nargs="+", help="estimate-ctf --variant labels")
    parser.add_argument(
        "--good-angstrom", type=float, default=20.0,
        help="a view counts as fitted when CTFFIND reached finer than this (default 20)",
    )
    args = parser.parse_args()

    tables = []
    for label in args.labels:
        stem = args.project / "ctf" / args.series / f"{args.series}_ctffind5_{label}"
        summary = Path(f"{stem}_ctffind.txt")
        rows = parse_ctffind_summary(summary)
        angles = [entry.angle_start_deg for entry in read_imod_defocus_file(Path(f"{stem}.defocus"))]
        if len(angles) != len(rows):
            raise SystemExit(f"{stem}: {len(rows)} fitted views but {len(angles)} angles")
        tables.append((label, angles, rows, _tilt_columns(summary)))

    reference = tables[0][1]
    for label, angles, _, _ in tables[1:]:
        if any(abs(a - b) > 0.01 for a, b in zip(angles, reference, strict=True)):
            raise SystemExit(f"{label} was fitted on other views than {tables[0][0]}")

    header = "  tilt " + "".join(f" | {label:>22}" for label, *_ in tables)
    print(header)
    print("        " + "".join(f" | {'defocus um':>11} {'fit A':>10}" for _ in tables))
    for index, angle in enumerate(reference):
        cells = []
        for _, _, rows, tilts in tables:
            row = rows[index]
            fit = row.fit_resolution_angstrom
            cell = f"{row.mean_defocus_angstrom / 1e4:11.3f} {fit if fit is not None else float('nan'):6.1f}"
            mark = "*" if fit is not None and fit < args.good_angstrom else " "
            estimate = tilts[index]
            cells.append(
                cell + mark + (f" ({estimate[1]:+.0f})" if estimate else "    ")
            )
        print(f"{angle:+6.1f} " + "".join(f" | {cell:>22}" for cell in cells))
    print(f"\n* fitted finer than {args.good_angstrom:g} A; (n) CTFFIND's own tilt estimate")

    print()
    for label, angles, rows, _ in tables:
        good = [
            (angle, row.mean_defocus_angstrom / 1e4)
            for angle, row in zip(angles, rows)
            if row.fit_resolution_angstrom is not None
            and row.fit_resolution_angstrom < args.good_angstrom
        ]
        line = _line([math.sin(math.radians(a)) for a, _ in good], [d for _, d in good])
        span = (
            f"{min(a for a, _ in good):+.0f} to {max(a for a, _ in good):+.0f} deg"
            if good else "none"
        )
        trend = (
            f"defocus = {line[0]:.3f} {line[1]:+.3f} x sin(tilt) um, rms {line[2]:.3f} um"
            if line is not None else "too few fitted views for a line"
        )
        print(f"{label:>12}: {len(good):2d}/{len(rows)} views fitted ({span}); {trend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
