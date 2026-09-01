#!/usr/bin/env python3
"""Are the template-matching picks detections, or the tail of the noise?

    python3 scripts/score_significance.py --project outputs/ribo

Answers the question no average can answer, because averaging picks at the
orientations a template assigned them reproduces that template even from pure
noise — measured on this pipeline: 100 noise volumes oriented by a global search
against a template average to a map correlating +0.32 with it, against +0.03 at
random orientations. The bias points toward a false yes, so a shape in the
average is not evidence.

What is evidence: how far above the noise a peak sits, in units of the search's
own standard deviation. pytom already computes that and writes it into the job
JSON we keep as an artifact — `job_stats["std"]` and `job_stats["search_space"]`
— and this pipeline reads only `template_shape` from that file. So the
measurement has been on disk all along, unread.

Two numbers decide it:

  z_max    the highest pick, in sigma
  z_1fp    the highest peak a search over this space produces with NO particle
           present at all, from extreme-value statistics

If z_max does not clear z_1fp, no pick in that series is a detection. Clearing it
is NOT the converse: the cut-off was placed to admit around a hundred expected
false positives per tomogram, so some picks sit above the one-false-positive
level whether or not anything is there. The verdict that matters is the
observed-versus-null table, not the single number. For scale,
published in-cell work on 80S ribosomes reports around 40 sigma, nuclear pore
complexes around 10, and a 100 kDa tubulin dimer around 6 (Cruz-Leon et al. 2024,
Nat Commun 15:3992) — so a ribosome ought to be unmissable, not marginal.

The observed-versus-null table is the fuller answer: if the counts track the
Gaussian null across the whole retained range, the picks ARE the tail. An excess
above the null curve is a real population, and its height is how many there are.

One caveat this prints rather than hides. `--random-phase-correction` reports
`scores - noise_scores + noise_scores.mean()`, an additive pedestal, while sigma
is accumulated from the uncorrected map. So absolute z carries an unknown
constant offset. A constant offset preserves the SLOPE of ln N(>z) against z,
which under the null equals -z, so the fitted slope gives an offset-free estimate
of the true z. Both are printed; where they disagree, trust the slope.

Disposable, like the rest of `scripts/`. Its replacement is reading these stats
in the picking backend and recording them in `PicksQc`, so that a run says
whether it detected anything at the time it runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def _read_star_columns(path: Path) -> dict[str, np.ndarray]:
    """Column name to float array, for the first loop_ block."""

    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]
    columns = [line.split()[0].lstrip("_") for line in lines if line.startswith("_")]
    rows = [
        line.split()
        for line in lines
        if line.strip() and not line.startswith(("data_", "loop_", "_", "#"))
    ]
    table: dict[str, np.ndarray] = {}
    for index, name in enumerate(columns):
        try:
            table[name] = np.array([float(row[index]) for row in rows])
        except (ValueError, IndexError):
            continue
    return table


def _largest_noise_peak_sigma(search_space: float) -> float:
    """The highest peak a peak-free search over `search_space` samples gives.

    The value z where the expected number of Gaussian samples above it is one:
    `search_space * Q(z) = 1`. Solved by bisection rather than with an inverse
    error function so that this has no scipy dependency, and so that the
    definition being used is visible in the code.
    """

    def expected_above(z: float) -> float:
        return search_space * 0.5 * math.erfc(z / math.sqrt(2.0))

    low, high = 0.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if expected_above(middle) > 1.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _fitted_slope(scores: np.ndarray, sigma: float) -> float | None:
    """Local slope of ln N(>z) against z over the upper half of the retained range.

    Under the Gaussian null this slope is -z, so its magnitude is an estimate of
    the true z that survives any additive pedestal in the reported scores.
    """

    z_values = np.sort(scores / sigma)
    if z_values.size < 8:
        return None
    upper = z_values[z_values.size // 2 :]
    counts = np.arange(upper.size, 0, -1, dtype=float)
    slope = np.polyfit(upper, np.log(counts), 1)[0]
    return float(slope)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()

    pattern = str(args.project / "picks" / "*" / "*_particles.star")
    stars = sorted(Path(one) for one in glob.glob(pattern))
    if not stars:
        print(f"no picks found under {args.project / 'picks'}")
        return 1

    header = f"{'series':10} {'picks':>6} {'z_cut':>7} {'z_max':>7} {'z_1fp':>7} {'slope':>7}"
    print(f"{header}  verdict")
    print("-" * 78)

    tables: list[tuple[str, np.ndarray, float, float]] = []
    for star in stars:
        job = star.with_name(star.name.replace("_particles.star", "_job.json"))
        if not job.is_file():
            print(f"{star.parent.name:10} no job JSON beside {star.name}")
            continue
        stats = json.loads(job.read_text()).get("job_stats") or {}
        sigma = stats.get("std")
        search_space = stats.get("search_space")
        table = _read_star_columns(star)
        scores = table.get("rlnLCCmax")
        if scores is None or scores.size == 0:
            print(f"{star.parent.name:10} no rlnLCCmax column")
            continue
        # pytom writes these per row when it has them; prefer them over job_stats.
        if "rlnSearchStd" in table:
            sigma = float(table["rlnSearchStd"][0])
        if sigma is None or search_space is None:
            keys = ", ".join(sorted(stats)) or "none"
            print(f"{star.parent.name:10} job_stats lacks std/search_space (has: {keys})")
            continue

        series = star.parent.name
        z = scores / float(sigma)
        z_one = _largest_noise_peak_sigma(float(search_space))
        cut = table.get("rlnCutOff")
        z_cut = float(cut[0]) / float(sigma) if cut is not None and cut.size else float(z.min())
        slope = _fitted_slope(scores, float(sigma))
        # Deliberately NOT "z_max > z_1fp": the cut-off was placed to admit ~100
        # expected false positives, so picks above the one-false-positive level
        # are there by construction. Testing a synthetic no-signal case against
        # that criterion returned DETECTION, which is how this was caught. The
        # excess of observed counts over the null is the measurement; this column
        # only flags the case that needs no table at all.
        verdict = (
            "below the one-false-positive level - nothing here"
            if z.max() <= z_one
            else "read the table below"
        )
        slope_text = f"{-slope:7.2f}" if slope is not None else "      -"
        print(
            f"{series:10} {z.size:>6} {z_cut:>7.2f} {z.max():>7.2f} {z_one:>7.2f} "
            f"{slope_text}  {verdict}"
        )
        tables.append((series, z, z_one, float(search_space)))

    if not tables:
        print()
        print("Nothing could be scored. The job JSON is kept as an artifact by")
        print("pick-particles, so check the paths above rather than re-running anything.")
        return 1

    print()
    print("Observed picks above each threshold, against what the Gaussian null predicts.")
    print("Counts that track the null column ARE the tail; an excess is a real population.")
    for series, z, _z_one, search_space in tables:
        print(f"\n  {series}")
        print(f"    {'z >':>7} {'observed':>9} {'null':>12}")
        for threshold in np.linspace(z.min(), z.max(), 8):
            observed = int((z > threshold).sum())
            null = search_space * 0.5 * math.erfc(threshold / math.sqrt(2.0))
            print(f"    {threshold:>7.2f} {observed:>9d} {null:>12.2f}")

    print()
    print("For scale, published in-cell values: 80S ribosome ~40 sigma, nuclear pore")
    print("~10, a 100 kDa tubulin dimer ~6 (Cruz-Leon et al. 2024, Nat Commun 15:3992).")
    print()
    print("Absolute z carries an unknown additive offset when")
    print("--random-phase-correction was used, because the reported score is")
    print("shifted while sigma comes from the uncorrected map. The fitted slope is")
    print("offset-free: under the null it equals -z, so where the two disagree the")
    print("slope is the one to trust.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
