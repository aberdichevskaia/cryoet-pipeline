#!/usr/bin/env python3
"""Several FSC reports side by side, with a statistic that is not quantised.

    python3 scripts/fsc_table.py outputs/ribo/fsc/real-cls0*-vs-template.json
    python3 scripts/fsc_table.py --shells "masked=fsc_a.json" "unmasked=fsc_b.json"

A report can be given as LABEL=PATH, and the label then names it in every
table; a bare path is named by its file name.

The threshold crossing is a poor way to compare curves that are close together,
and it misled here. It can only land on a shell: for a 48 box at 9.784 A the
shells near the crossing are 469.63/9 = 52.2 A and 469.63/10 = 47.0 A, with
nothing in between. Four class averages came back at 47.0, 52.2, 52.2 and 52.2 A
and were read as "all four agree", which the statistic cannot support — it has
one-shell resolution and cannot see a difference of a few percent.

So the crossings are read off the curve itself, at every level in `--levels`
(0.5 and 0.143 by default): the first shell to fall below the level, placed
between it and the shell before by linear interpolation in spatial frequency.
That also makes them independent of the threshold a report was written with,
which differs between half-set and map-to-model reports. They are printed
alongside two continuous numbers:

  band mean   the mean correlation across a stated band, by default from 200 A
              to 35 A. The lower limit is the resolution the tilt-series
              alignment supports on this data (residuals 0.69-0.92 nm), beyond
              which no curve carries signal; the upper limit skips the coarsest
              shells, which hold a handful of Fourier coefficients each and
              wander across any threshold.
  peak        the highest correlation and where it sits, which says whether two
              curves differ in amplitude rather than in where they end.

Neither is a resolution and neither should be quoted as one. They exist to
compare curves against each other, and above all to compare a real curve against
its control: a map-to-model FSC against the template the particles were oriented
by is confounded toward agreement, so the number that matters is the difference
between the real curve and the same measurement on random-position boxes.

Reads the reports rather than the maps, so it needs no GPU and no re-run.

Disposable, like the rest of `scripts/`. Its replacement is `compute-fsc`
recording a band mean next to the crossing, and recording which two maps it
compared — which the report currently does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def labelled(argument: str) -> tuple[str, Path]:
    """LABEL=PATH, or a bare PATH named by its file name."""

    label, separator, path = argument.partition("=")
    if not separator:
        return Path(argument).name, Path(argument)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {argument!r}")
    return label, Path(path)


def crossing(resolutions: list[float], values: list[float], level: float) -> float | None:
    """Where the curve first falls below `level`, interpolated in spatial frequency.

    The first fall only: a curve that dips through the level and rises again is
    read at the dip, as the half-map QC reads its crossings before a dip. None
    when it never falls below.
    """

    for index in range(1, len(values)):
        if values[index] < level <= values[index - 1]:
            coarse, fine = 1.0 / resolutions[index - 1], 1.0 / resolutions[index]
            fraction = (values[index - 1] - level) / (values[index - 1] - values[index])
            return 1.0 / (coarse + fraction * (fine - coarse))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        type=labelled,
        help="FSC report JSONs, each as PATH or LABEL=PATH.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[0.5, 0.143],
        help="Correlation levels to read crossings at, interpolated between shells.",
    )
    parser.add_argument(
        "--band-coarse-angstrom",
        type=float,
        default=200.0,
        help="Coarse end of the averaging band. Above this the shells hold too "
        "few coefficients to mean anything.",
    )
    parser.add_argument(
        "--band-fine-angstrom",
        type=float,
        default=35.0,
        help="Fine end. Default is what the alignment residuals support; beyond "
        "it every curve is at zero and averaging over it only dilutes.",
    )
    parser.add_argument(
        "--search-blind-angstrom",
        type=float,
        default=47.8,
        help="Where the orientation search stops being able to see, and so stops "
        "being able to print. The cone-search references are low-passed at "
        "initial_low_pass_angstrom (default 60 A) and filters.py gives that a "
        "raised-cosine edge two shells wide, which for a 48 box at 9.784 A/voxel "
        "reaches exactly zero at 47.8 A. The correlation numerator uses the "
        "FILTERED reference, so orientation ranking is blind beyond this and a "
        "template-printing artefact cannot reach here. Any difference between a "
        "real curve and its control in these shells is therefore not printing.",
    )
    parser.add_argument(
        "--shells",
        action="store_true",
        help="Print the curves shell by shell, and the difference when exactly "
        "two reports are given. A band mean averages shells the search controls "
        "together with shells it cannot touch, which hides the discriminating "
        "part; this shows it.",
    )
    args = parser.parse_args()

    crossing_header = "".join(f" {f'{level:g} at':>10}" for level in args.levels)
    print(
        f"{'report':<34}{crossing_header} {'band mean':>10} {'peak':>7} "
        f"{'peak at':>9} {'shells':>7} {'blind mean':>10}"
    )
    print("-" * (86 + 11 * len(args.levels)))

    rows: list[tuple[float, str]] = []
    curves: list[tuple[str, list[float], list[float]]] = []
    for label, path in args.reports:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{label:<34} could not be read: {exc}")
            continue
        correlation = payload.get("correlation") or []
        resolutions = payload.get("resolution_angstrom") or []
        if len(correlation) != len(resolutions) or not correlation:
            print(f"{label:<34} report holds no usable curve")
            continue

        in_band = [
            (value, resolution)
            for value, resolution in zip(correlation, resolutions, strict=True)
            if args.band_fine_angstrom <= resolution <= args.band_coarse_angstrom
        ]
        if not in_band:
            print(
                f"{label:<34} no shells between "
                f"{args.band_coarse_angstrom:.0f} and {args.band_fine_angstrom:.0f} A"
            )
            continue
        band_mean = sum(value for value, _ in in_band) / len(in_band)
        peak = max(correlation)
        peak_at = resolutions[correlation.index(peak)]
        crossings = ""
        for level in args.levels:
            found = crossing(resolutions, correlation, level)
            crossings += f" {'none' if found is None else f'{found:.1f} A':>10}"

        rows.append((band_mean, label))
        curves.append((label, correlation, resolutions))
        blind = [
            value
            for value, resolution in zip(correlation, resolutions, strict=True)
            if args.band_fine_angstrom <= resolution < args.search_blind_angstrom
        ]
        blind_mean = sum(blind) / len(blind) if blind else float("nan")
        print(
            f"{label[-33:]:<34}{crossings} {band_mean:>10.3f} "
            f"{peak:>7.3f} {peak_at:>7.1f} A {len(in_band):>7} {blind_mean:>10.4f}"
        )

        for warning in payload.get("warnings") or []:
            print(f"    warning: {warning}")

    if not rows:
        return 1

    if args.shells and curves:
        print(
            f"\nshell by shell. The search's reference is blind beyond "
            f"{args.search_blind_angstrom:.1f} A, so nothing below that line can "
            "have been printed by it:"
        )
        header = f"  {'shell':>5} {'A':>8}"
        for name, _, _ in curves:
            header += f" {name[:20]:>21}"
        if len(curves) == 2:
            header += f" {'difference':>12}"
        print(header)
        longest = max(len(one[1]) for one in curves)
        crossed_the_line = False
        for index in range(longest):
            resolution = next(
                (res[index] for _, _, res in curves if index < len(res)), None
            )
            if resolution is None:
                continue
            if not crossed_the_line and resolution < args.search_blind_angstrom:
                print(
                    f"  ----- {args.search_blind_angstrom:.1f} A: below here the "
                    "orientation search has zero response -----"
                )
                crossed_the_line = True
            line = f"  {index + 1:>5} {resolution:>8.1f}"
            values: list[float] = []
            for _, correlation, _ in curves:
                if index < len(correlation):
                    values.append(correlation[index])
                    line += f" {correlation[index]:>21.4f}"
                else:
                    line += f" {'':>21}"
            if len(curves) == 2 and len(values) == 2:
                line += f" {values[0] - values[1]:>+12.4f}"
            print(line)
        print(
            "\n  Read the difference column only BELOW the line. Above it both "
            "curves\n  are partly painted by the same mechanism, so agreement "
            "there is cheap;\n  below it the search could not rank orientations at "
            "all, so a difference\n  is either real signal or noise — and its size "
            "against the shells' own\n  scatter is what says which."
        )

    print(
        f"\nband mean is over {args.band_coarse_angstrom:.0f} to "
        f"{args.band_fine_angstrom:.0f} A, the range where these curves carry "
        "anything."
    )
    print("Ranked by it:")
    for band_mean, name in sorted(rows, reverse=True):
        print(f"  {band_mean:.3f}  {name}")
    if len(rows) > 1:
        spread = rows and max(one for one, _ in rows) - min(one for one, _ in rows)
        print(
            f"\nspread {spread:.3f} across {len(rows)} curves. Whether that is large "
            "depends entirely on what the control does — a curve against the "
            "template that oriented the particles agrees with it for free."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
