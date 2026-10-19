"""How alike two classifications of the same particles are, against chance.

Classification of pure noise also produces classes, and averaging at
template-matched orientations reproduces the template from noise (+0.32
correlation, measured on this pipeline), so a class that looks right proves
little. What a real class does that a partition of noise does not is reproduce:
run the classification again from another start, or with other particles mixed
in, and the same particles cluster together again.

Labels are not comparable directly — class 1 of one run may be class 3 of the
other — so everything here is invariant to relabelling: the adjusted Rand index
over the whole table, the best one-to-one pairing of classes, and the overlap of
two named groups (say, the classes each run seeded with noise or a blob). Each is
read against chance: shuffled labels for the first two, which keep both runs'
class sizes, and the hypergeometric distribution for the overlap.

These are the definitions scripts/compare_classifications.py has used for the
pipeline's own classes; it imports them from here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def contingency(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """How many particles each pair of classes shares: rows `first`'s, columns `second`'s.

    Counted with `bincount` rather than a loop of masks, because a trustworthy
    null needs hundreds of shuffles and the nested version made twenty the
    practical limit — which is far too few to read a maximum off.
    """

    left = sorted(set(np.asarray(first).tolist()))
    right = sorted(set(np.asarray(second).tolist()))
    rows = np.searchsorted(np.array(left), first)
    columns = np.searchsorted(np.array(right), second)
    flat = np.bincount(rows * len(right) + columns, minlength=len(left) * len(right))
    return flat.reshape(len(left), len(right)).astype(np.int64), left, right


def adjusted_rand_index(table: np.ndarray) -> float:
    """Agreement between two partitions, corrected for chance: 0 at chance, 1 when equal.

    Written out rather than imported so that the definition in use is visible:
    pairs counted together or apart by both, minus what partitions of these
    sizes would agree on at random, over the most they could agree on.
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


def best_pairing(table: np.ndarray) -> list[tuple[int, int]]:
    """Match each class of one run to the one it most overlaps in the other.

    A proper assignment rather than a greedy pick per row, so that two classes
    cannot both claim the same partner and overstate the agreement.
    """

    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

    rows, columns = linear_sum_assignment(-table)
    return list(zip(rows.tolist(), columns.tolist(), strict=True))


def paired_share(table: np.ndarray) -> float:
    """The share of particles in the same class of both runs, under the best pairing."""

    total = table.sum()
    if not total:
        return float("nan")
    return float(sum(table[row, column] for row, column in best_pairing(table)) / total)


@dataclass(frozen=True)
class ShuffledNull:
    """What the statistics reach when one run's labels are shuffled among its particles."""

    rand_index: np.ndarray
    paired_share: np.ndarray

    @staticmethod
    def p_value(null: np.ndarray, observed: float) -> float:
        """The share of shuffles that reach the observed value, counting the observation."""

        return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))


def shuffled_null(first: np.ndarray, second: np.ndarray, *, rounds: int,
                  seed: int = 0) -> ShuffledNull:
    """The adjusted Rand index and the paired share over `rounds` shuffles of `second`.

    The paired share picks the best of several pairings, and that selection
    alone lifts it above an unselected share, so its null is computed the same
    way: a statistic computed after choosing the best of several needs a null
    chosen alike (scripts/compare_classifications.py learnt this at a cost).
    """

    rng = np.random.default_rng(seed)
    shuffled = np.array(second, copy=True)
    indices, shares = [], []
    for _ in range(rounds):
        rng.shuffle(shuffled)
        table, _, _ = contingency(first, shuffled)
        indices.append(adjusted_rand_index(table))
        shares.append(paired_share(table))
    return ShuffledNull(np.array(indices), np.array(shares))


@dataclass(frozen=True)
class GroupOverlap:
    """How two groups of particles, one named in each run, overlap."""

    both: int
    first_only: int
    second_only: int
    neither: int

    @property
    def total(self) -> int:
        return self.both + self.first_only + self.second_only + self.neither

    @property
    def first(self) -> int:
        return self.both + self.first_only

    @property
    def second(self) -> int:
        return self.both + self.second_only

    @property
    def expected_both(self) -> float:
        """The overlap two independent groups of these sizes have on average."""

        return self.first * self.second / self.total if self.total else math.nan

    @property
    def ratio(self) -> float:
        return self.both / self.expected_both if self.expected_both else math.nan

    @property
    def p_more(self) -> float:
        """Hypergeometric chance of an overlap this large or larger."""

        from scipy.stats import hypergeom  # type: ignore[import-untyped]

        return float(hypergeom.sf(self.both - 1, self.total, self.first, self.second))

    @property
    def p_less(self) -> float:
        """Hypergeometric chance of an overlap this small or smaller."""

        from scipy.stats import hypergeom

        return float(hypergeom.cdf(self.both, self.total, self.first, self.second))


def group_overlap(in_first: Sequence[bool], in_second: Sequence[bool]) -> GroupOverlap:
    """Count the four cells of two yes/no labellings of the same particles."""

    first = np.asarray(in_first, dtype=bool)
    second = np.asarray(in_second, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("the two groups must label the same particles")
    return GroupOverlap(
        int(np.count_nonzero(first & second)),
        int(np.count_nonzero(first & ~second)),
        int(np.count_nonzero(~first & second)),
        int(np.count_nonzero(~first & ~second)),
    )
