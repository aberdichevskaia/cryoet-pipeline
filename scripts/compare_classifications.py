#!/usr/bin/env python3
"""Do two classifications of the same particles agree, or did they partition noise?

    python3 scripts/compare_classifications.py --first <seed1.star> --second <seed7.star>

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
from pathlib import Path

import numpy as np

from cryoet_pipeline.particles import read_particle_star


def _contingency(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """How many particles each pair of classes shares.

    Counted with `bincount` rather than a loop of masks, because a trustworthy
    null needs hundreds of shuffles and the nested version made twenty the
    practical limit — which is far too few to read a maximum off.
    """

    left = sorted(set(first.tolist()))
    right = sorted(set(second.tolist()))
    rows = np.searchsorted(np.array(left), first)
    columns = np.searchsorted(np.array(right), second)
    flat = np.bincount(
        rows * len(right) + columns, minlength=len(left) * len(right)
    )
    return flat.reshape(len(left), len(right)).astype(np.int64), left, right


def _adjusted_rand_index(table: np.ndarray) -> float:
    """Agreement between two partitions, corrected for chance.

    Implemented here rather than imported so that the definition being used is
    visible: pairs counted together or apart by both, minus what partitions of
    these sizes would agree on at random, over the most they could agree on.
    """

    def pairs(counts: np.ndarray) -> float:
        return float(np.sum(counts * (counts - 1) / 2.0))

    total = float(table.sum())
    if total < 2:
        return float("nan")
    index = pairs(table)
    rows = pairs(table.sum(axis=1))
    columns = pairs(table.sum(axis=0))
    expected = rows * columns / (total * (total - 1) / 2.0)
    largest = 0.5 * (rows + columns)
    if largest == expected:
        return float("nan")
    return (index - expected) / (largest - expected)


def _best_pairing(table: np.ndarray) -> list[tuple[int, int]]:
    """Match each class in one run to the one it most overlaps in the other.

    A proper assignment rather than a greedy pick per row, so that two classes
    cannot both claim the same partner and overstate the agreement.
    """

    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-table)
    return list(zip(rows.tolist(), columns.tolist(), strict=True))


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="Refined STAR, run one.")
    parser.add_argument("--second", required=True, type=Path, help="Refined STAR, run two.")
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
