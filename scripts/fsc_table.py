#!/usr/bin/env python3
"""Several FSC reports side by side, with a statistic that is not quantised.

    python3 scripts/fsc_table.py outputs/ribo/fsc/real-cls0*-vs-template.json

The threshold crossing is a poor way to compare curves that are close together,
and it misled here. It can only land on a shell: for a 48 box at 9.784 A the
shells near the crossing are 469.63/9 = 52.2 A and 469.63/10 = 47.0 A, with
nothing in between. Four class averages came back at 47.0, 52.2, 52.2 and 52.2 A
and were read as "all four agree", which the statistic cannot support — it has
one-shell resolution and cannot see a difference of a few percent.

So the crossing is printed alongside two continuous numbers:

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="FSC report JSONs.")
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
    args = parser.parse_args()

    print(
        f"{'report':<34} {'crossing':>10} {'band mean':>10} {'peak':>7} "
        f"{'peak at':>9} {'shells':>7}"
    )
    print("-" * 84)

    rows: list[tuple[float, str]] = []
    for path in args.reports:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{path.name:<34} could not be read: {exc}")
            continue
        correlation = payload.get("correlation") or []
        resolutions = payload.get("resolution_angstrom") or []
        if len(correlation) != len(resolutions) or not correlation:
            print(f"{path.name:<34} report holds no usable curve")
            continue

        in_band = [
            (value, resolution)
            for value, resolution in zip(correlation, resolutions, strict=True)
            if args.band_fine_angstrom <= resolution <= args.band_coarse_angstrom
        ]
        if not in_band:
            print(
                f"{path.name:<34} no shells between "
                f"{args.band_coarse_angstrom:.0f} and {args.band_fine_angstrom:.0f} A"
            )
            continue
        band_mean = sum(value for value, _ in in_band) / len(in_band)
        peak = max(correlation)
        peak_at = resolutions[correlation.index(peak)]
        crossing = payload.get("crossing_resolution_angstrom")
        crossing_text = f"{crossing:.1f} A" if crossing is not None else "none"

        rows.append((band_mean, path.name))
        print(
            f"{path.name[-33:]:<34} {crossing_text:>10} {band_mean:>10.3f} "
            f"{peak:>7.3f} {peak_at:>7.1f} A {len(in_band):>7}"
        )

        for warning in payload.get("warnings") or []:
            print(f"    warning: {warning}")

    if not rows:
        return 1

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
