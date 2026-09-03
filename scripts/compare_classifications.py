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
    """How many particles each pair of classes shares."""

    left = sorted(set(first.tolist()))
    right = sorted(set(second.tolist()))
    table = np.zeros((len(left), len(right)), dtype=np.int64)
    for row, one in enumerate(left):
        for column, other in enumerate(right):
            table[row, column] = int(np.sum((first == one) & (second == other)))
    return table, left, right


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


def _empirical_null(first: np.ndarray, second: np.ndarray, *, rounds: int) -> np.ndarray:
    """What the index reaches when one partition is shuffled.

    The adjusted index is already chance-corrected, so this should sit near zero.
    Printed anyway because a corrected statistic is only trustworthy if its null
    behaves, and because a reader needs a scale for the number above it.
    """

    rng = np.random.default_rng(0)
    shuffled = second.copy()
    values = []
    for _ in range(rounds):
        rng.shuffle(shuffled)
        table, _, _ = _contingency(first, shuffled)
        values.append(_adjusted_rand_index(table))
    return np.array(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path, help="Refined STAR, run one.")
    parser.add_argument("--second", required=True, type=Path, help="Refined STAR, run two.")
    parser.add_argument("--null-rounds", type=int, default=20)
    args = parser.parse_args()

    one = read_particle_star(args.first)
    other = read_particle_star(args.second)
    print(f"{args.first.name}: {len(one)} particles")
    print(f"{args.second.name}: {len(other)} particles\n")

    by_index = {record.index: record.class_number for record in other}
    shared = [record for record in one if record.index in by_index]
    if len(shared) != len(one) or len(one) != len(other):
        print(
            f"comparing the {len(shared)} particles present in both. A mismatch means "
            "the two runs did not see the same particle stacks, in which case the "
            "indices do not refer to the same particles and nothing below is valid.\n"
        )
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
    null = _empirical_null(first, second, rounds=args.null_rounds)
    print(f"adjusted Rand index: {index:+.3f}")
    print(
        f"shuffled null over {args.null_rounds} rounds: "
        f"median {np.median(null):+.3f}, range [{null.min():+.3f}, {null.max():+.3f}]"
    )

    print("\nper-class agreement, on the best pairing of labels:")
    chance = 1.0 / max(len(right), 1)
    shares: list[tuple[float, int, int, int]] = []
    for row, column in _best_pairing(table):
        overlap = int(table[row, column])
        size = int(table[row].sum())
        share = overlap / size if size else 0.0
        shares.append((share, left[row], right[column], size))
        print(
            f"  s1/{left[row]} -> s2/{right[column]}: {overlap} of {size} particles "
            f"stayed together ({share:.0%}, chance is about {chance:.0%})"
        )

    best_share, best_left, best_right, best_size = max(shares)
    print()
    if not np.isfinite(index):
        print("The index is undefined — one run put everything in a single class.")
        return 0

    # The per-class share is the sensitive statistic and the global index is not.
    # One real class of 200 among 800, with the rest random, gives a per-class
    # share of 100% and an adjusted Rand index of only 0.18 — because three
    # chance-level classes dilute a global measure. So the verdict is read off the
    # best class, with the index and the null only establishing that anything at
    # all is above chance.
    above_chance = index > max(0.05, float(null.max()))
    strong_class = best_share > 2.0 * chance and best_size >= 30

    if strong_class and above_chance:
        print(f"Class s1/{best_left} reproduces: {best_share:.0%} of its "
              f"{best_size} particles cluster together again from a different")
        print(f"start, against a chance level of {chance:.0%}. That is a real")
        print("population. Look at its map and its gallery, then refine it alone.")
    elif strong_class:
        print(f"Class s1/{best_left} holds together at {best_share:.0%} against a")
        print(f"chance level of {chance:.0%}, but the partitions as a whole are not")
        print("above chance. A minority class dilutes the global index, so this is")
        print("the expected shape of a small real population — treat it as a lead")
        print("and check its gallery before spending anything on it.")
    elif above_chance:
        print("The partitions agree beyond chance but no single class holds")
        print("together, so what reproduces is a gradient rather than a group.")
        print("Classifying into more classes may separate it; refining any one of")
        print("these will not.")
    else:
        print("Indistinguishable from chance, and no class holds together. These")
        print("classes are a partition of noise, and one that looks like a particle")
        print("is looking like one by accident. Do not refine it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
