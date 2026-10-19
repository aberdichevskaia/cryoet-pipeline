#!/usr/bin/env python3
"""Do two classifications of the same particles agree, or did they partition noise?

    python3 scripts/compare_classifications.py \\
        --first <seed1.star> --second <seed7.star> \\
        --qc <the seed-1 classification QC json> \\
        --registry outputs/ribo/artifacts.json

Classification of pure noise also produces classes, and each will look like
something. The classic way to be fooled is to classify once, pick the class that
looks best, refine it, and get a convincing average of nothing — which is a
danger here specifically, because averaging particles at template-matched
orientations already reproduces the template from noise (+0.32 correlation,
measured on this pipeline).

What distinguishes a real class from a partition of noise is that a real one
REPRODUCES: run the classification from a different starting point and the same
particles cluster together again. Nothing else available here separates the two.

The labels are not comparable directly — class 1 in one run may be class 3 in the
other — so the comparison has to be permutation-invariant. The adjusted Rand
index is that: it counts pairs of particles that both runs either put together or
kept apart, corrected for the agreement two random partitions of the same sizes
would reach by chance. Zero means chance. One means identical.

An empirical null is printed alongside it, because "0.04" only means something
next to what shuffling produces.

Disposable, like the rest of `scripts/`. Its replacement is this comparison
recorded when a classification runs against a previous one, so nobody has to
remember to check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cryoet_pipeline.class_comparison import adjusted_rand_index as _adjusted_rand_index
from cryoet_pipeline.class_comparison import best_pairing as _best_pairing
from cryoet_pipeline.class_comparison import contingency as _contingency
from cryoet_pipeline.particles import read_particle_star


def _empirical_null(
    first: np.ndarray, second: np.ndarray, *, rounds: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the statistics reach when one partition is shuffled.

    Three distributions, because the per-class share needed one and did not have
    it. The first version compared a share against `1 / K` — 25% for four
    classes — and called anything under twice that chance. But the labels are
    paired by whichever assignment maximises the overlap, and that selection
    alone lifts the best share to about 28%, with 31% reachable in four hundred
    shuffles. So an observed 36% read as "indistinguishable from chance" when no
    shuffle had ever come near it.

    The lesson is narrow and worth keeping: a statistic computed after choosing
    the best of several needs a null computed the same way. Comparing it against
    the null of an unselected statistic understates the evidence, and the
    invented factor that papered over the gap hid a real result for an hour.
    """

    rng = np.random.default_rng(0)
    shuffled = second.copy()
    indices, best_shares, all_shares = [], [], []
    for _ in range(rounds):
        rng.shuffle(shuffled)
        table, _, _ = _contingency(first, shuffled)
        indices.append(_adjusted_rand_index(table))
        sizes = table.sum(axis=1)
        shares = [
            table[row, column] / sizes[row] if sizes[row] else 0.0
            for row, column in _best_pairing(table)
        ]
        best_shares.append(max(shares))
        all_shares.extend(shares)
    return np.array(indices), np.array(best_shares), np.array(all_shares)


def _series_composition(qc: Path, registry: Path | None) -> tuple[list[str], list[int]]:
    """Series names and their particle counts, read rather than typed.

    This took a list of counts on the command line for one version, and the
    first use of it pasted the four CLASS sizes — which summed to the same 4000,
    passed the only check there was, and produced a series column computed
    against boundaries that do not exist. The run has five series, so five counts
    were needed; nothing could tell.

    So the counts are read from the run's own report instead. `particle_counts`
    is recorded there now, and for reports written before that field existed the
    counts are recovered from the registry, since every particle-stack artifact
    records its own `particle_count` and the report lists the stacks in order.
    """

    payload = json.loads(qc.read_text())
    names = [str(one) for one in payload.get("tilt_series_ids") or []]
    stack_ids = [str(one) for one in payload.get("particle_stack_ids") or []]
    counts = [int(one) for one in payload.get("particle_counts") or []]
    if counts:
        return names, counts

    if registry is None:
        raise SystemExit(
            f"{qc.name} records no particle_counts — it predates that field. Pass\n"
            "--registry outputs/<project>/artifacts.json and the counts will be\n"
            "recovered from the particle-stack artifacts it names."
        )
    entries = {
        str(one["id"]): one
        for one in json.loads(registry.read_text()).get("artifacts") or []
    }
    recovered: list[int] = []
    for stack_id in stack_ids:
        entry = entries.get(stack_id)
        if entry is None:
            raise SystemExit(
                f"{qc.name} names the particle stack {stack_id}, which is not in\n"
                f"{registry}. Without it the per-series counts cannot be recovered."
            )
        value = (entry.get("parameters") or {}).get("particle_count")
        if value is None:
            raise SystemExit(f"{stack_id} records no particle_count")
        recovered.append(int(value))
    return names, recovered


def _class_profile(
    records, *, particle_counts: list[int] | None, series_names: list[str] | None
) -> None:
    """What each class tracks besides shape, for the classes that reproduce.

    A class can reproduce across seeds without being a population of particles:
    anything that varies between boxes and is stable per box reproduces
    perfectly. Depth in the lamella, correlation score, and which tilt-series a
    box came from are all of that kind, all plausible here, and none of them
    makes a class ribosomal. They are also invisible in the map — a class split
    by ice thickness looks exactly like a class split by nothing.

    So this is printed next to the reproducibility verdict rather than instead of
    it. A class whose mean z sits far from the others is separating depth; one
    whose mean score does is separating the score it was selected on; one drawn
    from a single series is separating microscope conditions.
    """

    header = "           n      z mean     z sd    score mean"
    if particle_counts:
        names = series_names or [f"s{one}" for one in range(1, len(particle_counts) + 1)]
        header += "".join(f"  {one[-7:]:>8}" for one in names)
    print("what the first run's classes hold, besides a shape:")
    print(header)
    boundaries = np.cumsum(particle_counts) if particle_counts else None
    for number in sorted({record.class_number for record in records}):
        members = [record for record in records if record.class_number == number]
        depths = np.array([record.z_px for record in members])
        scores = np.array(
            [record.score for record in members if record.score is not None]
        )
        composition = ""
        if boundaries is not None:
            # A combined index is a position in the concatenated stacks, so the
            # cumulative per-series counts are the only thing that says which
            # series a record came from — `ParticleRecord` cannot.
            which = np.searchsorted(
                boundaries, [record.index for record in members], side="right"
            )
            shares = np.bincount(which, minlength=len(particle_counts))
            composition = "".join(
                f"  {100 * value / max(len(members), 1):>7.0f}%" for value in shares
            )
        mean_score = f"{scores.mean():11.4f}" if scores.size else "          -"
        print(
            f"  {number:<3} {len(members):>6} {depths.mean():10.1f} "
            f"{depths.std():8.1f} {mean_score}{composition}"
        )
    if boundaries is not None:
        # What even shares mean is worth stating, because the expected share is
        # the series' share of the whole, not 1/n_series: the series contribute
        # different numbers of particles.
        total = float(sum(particle_counts))
        expected = "  ".join(f"{100 * one / total:.0f}%" for one in particle_counts)
        print(f"  a class that is not separating series would show {expected}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="Refined STAR, run one.")
    parser.add_argument("--second", required=True, type=Path, help="Refined STAR, run two.")
    parser.add_argument(
        "--qc",
        type=Path,
        default=None,
        help="The run's own QC JSON, for the per-series composition column. A "
        "record cannot say which series it came from — only the per-series counts "
        "can, and typing them by hand is how the four class sizes got pasted in "
        "their place. Omit it and the column is skipped rather than guessed.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Needed only when --qc predates the particle_counts field, to recover "
        "the counts from the particle-stack artifacts the report names.",
    )
    parser.add_argument(
        "--null-rounds",
        type=int,
        default=400,
        help="Shuffles used for the baselines. A maximum cannot be read "
        "off twenty rounds, which is what the first version used.",
    )
    args = parser.parse_args()

    one = read_particle_star(args.first)
    other = read_particle_star(args.second)
    print(f"{args.first.name}: {len(one)} particles")
    print(f"{args.second.name}: {len(other)} particles\n")

    by_index = {record.index: record.class_number for record in other}
    shared = [record for record in one if record.index in by_index]
    if len(one) != len(other):
        # Refused rather than warned about. The first version printed this and
        # then printed the whole comparison anyway, so a reader who skimmed the
        # warning got a confident "indistinguishable from chance" computed over
        # unrelated particles — which is worse than no answer, because it looks
        # like one. `ParticleRecord.index` is a position, not a name: index 0 in
        # a 706-particle run and index 0 in a 4000-particle run are different
        # objects, and nothing here can detect that beyond the count.
        print(
            f"REFUSING: the two runs describe {len(one)} and {len(other)} particles.\n"
            "\nA classification's indices are positions in the stacks it read, so two\n"
            "runs are only comparable if they read the same stacks — which means the\n"
            "same --variant. Comparing across variants pairs unrelated particles and\n"
            "returns a number that looks like an answer.\n"
            "\nPass the two seeds of ONE variant. In this project that is a pair like\n"
            "  ..._bin8-top800-cls4-s1_particles.star\n"
            "  ..._bin8-top800-cls4-s7_particles.star"
        )
        return 1
    if not shared:
        print("no particles in common — check that both runs used the same --variant")
        return 1

    first = np.array([record.class_number for record in shared])
    second = np.array([by_index[record.index] for record in shared])

    table, left, right = _contingency(first, second)
    print("how the first run's classes fall across the second run's:")
    header = "         " + "".join(f"  s2/{name:<3}" for name in right) + "    total"
    print(header)
    for row, name in enumerate(left):
        counts = "".join(f"{value:7d}" for value in table[row])
        print(f"  s1/{name:<3} {counts} {table[row].sum():8d}")
    totals = "".join(f"{value:7d}" for value in table.sum(axis=0))
    print(f"  total  {totals} {table.sum():8d}\n")

    names: list[str] | None = None
    counts: list[int] | None = None
    if args.qc is not None:
        names, counts = _series_composition(args.qc, args.registry)
        if sum(counts) != len(shared):
            # Refused rather than warned about: a composition computed against
            # the wrong boundaries is not a weaker answer, it is a different
            # question answered confidently.
            print(
                f"REFUSING the series column: {args.qc.name} accounts for "
                f"{sum(counts)} particles across {len(counts)} series "
                f"({', '.join(str(one) for one in counts)}) but these STARs "
                f"describe {len(shared)}. The QC and the STARs are from "
                "different runs.\n"
            )
            names, counts = None, None
        elif len(names) != len(counts):
            print(
                f"{args.qc.name} lists {len(names)} series and {len(counts)} "
                "counts; the columns will be numbered instead of named.\n"
            )
            names = None
    _class_profile(shared, particle_counts=counts, series_names=names)

    index = _adjusted_rand_index(table)
    null_index, null_best, null_shares = _empirical_null(
        first, second, rounds=args.null_rounds
    )
    print(f"adjusted Rand index: {index:+.3f}")
    print(
        f"shuffled null over {args.null_rounds} rounds: "
        f"median {np.median(null_index):+.3f}, "
        f"range [{null_index.min():+.3f}, {null_index.max():+.3f}]"
    )

    # The baseline is what shuffling reaches, not 1/K. Pairing the labels by the
    # assignment that maximises overlap lifts a share above 1/K on its own, and
    # comparing against 1/K therefore understates every share by that much.
    share_ceiling = float(null_shares.max())
    best_ceiling = float(null_best.max())
    print(
        f"\nper-class agreement. Shuffling reaches {np.median(null_shares):.0%} for a "
        f"class and at most {share_ceiling:.0%} in {args.null_rounds} rounds; the best "
        f"class of a shuffled run reaches at most {best_ceiling:.0%}:"
    )
    shares: list[tuple[float, int, int, int]] = []
    for row, column in _best_pairing(table):
        overlap = int(table[row, column])
        size = int(table[row].sum())
        share = overlap / size if size else 0.0
        shares.append((share, left[row], right[column], size))
        verdict = "above every shuffle" if share > share_ceiling else "within chance"
        print(
            f"  s1/{left[row]} -> s2/{right[column]}: {overlap} of {size} particles "
            f"stayed together ({share:.0%}) — {verdict}"
        )

    best_share, best_left, best_right, best_size = max(shares)
    reproducing = [one for one in shares if one[0] > share_ceiling]
    print()
    if not np.isfinite(index):
        print("The index is undefined — one run put everything in a single class.")
        return 0

    # The per-class share decides, not the global index. One real class of 200
    # among 800, with the rest random, gives a share of 100% and an adjusted Rand
    # index of only 0.18 — three chance-level classes dilute any global measure.
    # So the index is reported and the shares are read.
    if not reproducing:
        print("No class holds together beyond what shuffling reaches. These classes")
        print("are a partition of noise, and one that looks like a particle is looking")
        print("like one by accident. Do not refine it.")
        return 0

    names = ", ".join(f"s1/{one[1]}" for one in sorted(reproducing, reverse=True))
    print(f"Reproducing beyond every shuffle: {names}.")
    print(
        f"The strongest is s1/{best_left} at {best_share:.0%} of its {best_size} "
        f"particles, against {best_ceiling:.0%} for the best class of a shuffled run."
    )
    print()
    print("That is real structure: the same particles group together again from a")
    print("different start. What it is remains open — look at these classes' maps")
    print("and galleries before refining, because reproducible is not the same as")
    print("ribosomal, and a shared artefact reproduces just as faithfully as a")
    print("shared particle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
