"""Do picks land on particles? Their average against averages at random positions.

One of the checks an import makes (docs/entry_and_exit.md, Checks an import
makes). It uses nothing of what the import was told except the positions it
produced, so it tests the answers that made them: the scale, the axes, the
orientation the picks were taken in. Any of those wrong, and the boxes land on
whatever is there, and their average is as flat as an average of boxes at random
positions.

The null is `scripts/decoy_picks.py`'s: as many boxes as there are picks, x and y
uniform over the volume, the depths drawn from the picks' own, so that both sets
sample the same layers of the specimen. Nineteen such sets give a picks average
that beats all of them a chance of 1 in 20 under the null.

The picks are also set against their own mirrors along each axis, as a volume
is set against its eight axis reversals: picks taken in the other orientation of a
mirror pair land better mirrored.

The statistic is the spread of the average inside the box's inscribed sphere,
after a light low-pass (a Gaussian of one voxel): an average of boxes centred on
a particle holds its density, one of random boxes converges to the local mean.

The whole volume is read once, in slabs of its own z chunks, and every box adds
its overlap with the slab to its set's sum; nothing per box is kept.
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import numpy as np

DECOY_SETS = 19
# In standard deviations of the random sets' statistic. Under the null a set of
# picks stays below FLAT_Z; one that clears LANDS_Z is not chance.
FLAT_Z = 3.0
LANDS_Z = 5.0
# A mirror of the picks has to land this much better than the picks to say so.
MIRROR_MARGIN = 1.2
MAX_TESTED = 300
MIN_TESTED = 10
MIN_INSIDE_FRACTION = 0.5
SEED = 20261116

Verdict = Literal["lands", "weak", "flat", "mirrored", "outside", "not testable"]


@dataclass(frozen=True)
class PicksCheck:
    count: int
    inside: int
    """Picks whose whole box lies in the volume."""
    tested: int
    box_px: int
    statistic: float
    decoy_statistics: list[float]
    beaten: int
    """How many decoy sets the picks' average beats."""
    z_score: float
    """The statistic in standard deviations of the random sets' above their mean."""
    mirror_statistics: dict[str, float]
    """The statistic for the picks mirrored along each axis."""
    verdict: Verdict
    reason: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def box_starts(positions_xyz: np.ndarray, box: int) -> np.ndarray:
    """Each box's first voxel, (z, y, x), as extraction rounds a pick to one."""

    half = box // 2
    rounded = np.rint(np.asarray(positions_xyz, dtype=float)).astype(int)
    return np.column_stack([rounded[:, 2], rounded[:, 1], rounded[:, 0]]) - half


def _inside(starts: np.ndarray, box: int, shape: tuple[int, int, int]) -> np.ndarray:
    upper = np.asarray(shape) - box
    inside: np.ndarray = np.all((starts >= 0) & (starts <= upper), axis=1)
    return inside


def box_averages(volume: Any, sets: list[np.ndarray], box: int) -> list[np.ndarray]:
    """The average box of each set of starts, reading the volume once by z slabs."""

    depth = int(volume.shape[0])
    sums = [np.zeros((box, box, box)) for _ in sets]
    entries = sorted(
        (int(start[0]), index, int(start[1]), int(start[2]))
        for index, starts in enumerate(sets)
        for start in starts
    )
    if not entries:
        return sums
    keys = [entry[0] for entry in entries]
    chunks = getattr(volume, "chunks", None)
    step = int(chunks[0]) if chunks else 16
    low = entries[0][0]
    high = min(depth, entries[-1][0] + box)
    for slab_start in range(low - low % step, high, step):
        slab_stop = min(slab_start + step, depth)
        slab = np.asarray(volume[slab_start:slab_stop], dtype=np.float64)
        first = bisect.bisect_left(keys, slab_start - box + 1)
        last = bisect.bisect_left(keys, slab_stop)
        for z0, index, y0, x0 in entries[first:last]:
            top = max(z0, slab_start)
            bottom = min(z0 + box, slab_stop)
            if top >= bottom:
                continue
            sums[index][top - z0 : bottom - z0] += slab[
                top - slab_start : bottom - slab_start, y0 : y0 + box, x0 : x0 + box
            ]
    return [total / max(len(starts), 1) for total, starts in zip(sums, sets, strict=True)]


def _gaussian_low_pass(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    frequencies = [np.fft.fftfreq(size) for size in volume.shape]
    squared = sum(
        np.square(f).reshape([-1 if axis == index else 1 for axis in range(3)])
        for index, f in enumerate(frequencies)
    )
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * squared)
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * kernel))


def average_contrast(average: np.ndarray) -> float:
    """The spread of a low-passed average inside the box's inscribed sphere."""

    box = average.shape[0]
    smooth = _gaussian_low_pass(average - average.mean())
    centre = (box - 1) / 2
    grid = np.indices(average.shape) - centre
    sphere = np.sqrt(np.square(grid).sum(axis=0)) <= box / 2
    return float(smooth[sphere].std())


def _mirrors(positions: np.ndarray, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """The picks mirrored along each axis of the volume, (x, y, z) as given."""

    depth, height, width = shape
    x, y, z = (positions[:, axis] for axis in range(3))
    return {
        "mirrored in z": np.column_stack([x, y, (depth - 1) - z]),
        "mirrored in y": np.column_stack([x, (height - 1) - y, z]),
        "mirrored in x": np.column_stack([(width - 1) - x, y, z]),
    }


def check_picks(volume: Any, positions_xyz: np.ndarray, box: int) -> PicksCheck:
    """Set the picks' average against random positions and against its own mirrors.

    The picks' statistic is read in standard deviations of the random sets':
    under `FLAT_Z` it is `flat`, chance; from `LANDS_Z` on it `lands`; between,
    `weak`. A mirror that lands clearly better (`MIRROR_MARGIN`, and itself past
    `LANDS_Z`) makes it `mirrored`: the picks were taken in the other orientation
    of a mirror pair.
    """

    shape = cast(tuple[int, int, int], tuple(int(size) for size in volume.shape))
    positions = np.asarray(positions_xyz, dtype=float).reshape(-1, 3)
    starts = box_starts(positions, box)
    inside = _inside(starts, box, shape)
    count = len(positions)
    if count and inside.mean() < MIN_INSIDE_FRACTION:
        return PicksCheck(count, int(inside.sum()), 0, box, 0.0, [], 0, 0.0, {}, "outside",
                          f"{count - int(inside.sum())} of {count} picks put their box outside "
                          f"the volume {shape} (z, y, x): a scale or an axis is wrong")
    if int(inside.sum()) < MIN_TESTED:
        return PicksCheck(count, int(inside.sum()), int(inside.sum()), box, 0.0, [], 0, 0.0,
                          {}, "not testable",
                          f"{int(inside.sum())} picks inside the volume; the check needs "
                          f"{MIN_TESTED}")
    rng = np.random.default_rng(SEED)
    chosen = np.flatnonzero(inside)
    if len(chosen) > MAX_TESTED:
        chosen = rng.choice(chosen, size=MAX_TESTED, replace=False)
    kept = starts[chosen]
    # x and y uniform over the picks' own extent, the depths the picks' own: the
    # random sets sample the region the picks do, not the empty margins too.
    decoys = []
    for _ in range(DECOY_SETS):
        y = rng.integers(kept[:, 1].min(), kept[:, 1].max() + 1, size=len(kept))
        x = rng.integers(kept[:, 2].min(), kept[:, 2].max() + 1, size=len(kept))
        z = rng.choice(kept[:, 0], size=len(kept), replace=True)
        decoys.append(np.column_stack([z, y, x]))
    mirrors = {}
    for name, mirrored in _mirrors(positions[chosen], shape).items():
        mirrored_starts = box_starts(mirrored, box)
        mirrors[name] = mirrored_starts[_inside(mirrored_starts, box, shape)]
    averages = box_averages(volume, [kept, *decoys, *mirrors.values()], box)
    statistic = average_contrast(averages[0])
    decoy_statistics = [average_contrast(a) for a in averages[1 : 1 + DECOY_SETS]]
    mirror_statistics = {
        name: average_contrast(average)
        for name, average, starts_of in zip(mirrors, averages[1 + DECOY_SETS :],
                                            mirrors.values(), strict=True)
        if len(starts_of) >= MIN_TESTED
    }
    beaten = int(sum(statistic > value for value in decoy_statistics))
    centre = float(np.mean(decoy_statistics))
    spread = max(float(np.std(decoy_statistics, ddof=1)), 1e-12 * max(abs(centre), 1.0))

    def z_of(value: float) -> float:
        return (value - centre) / spread

    z_score = z_of(statistic)
    ranked = (f"the picks' average {statistic:.4g}, {z_score:.1f} standard deviations above "
              f"boxes at random positions in the same region ({min(decoy_statistics):.4g} to "
              f"{max(decoy_statistics):.4g}), beating {beaten} of {DECOY_SETS}")
    better = {name: value for name, value in mirror_statistics.items()
              if value > MIRROR_MARGIN * statistic and z_of(value) >= LANDS_Z}
    verdict: Verdict
    if better:
        verdict = "mirrored"
        name, value = max(better.items(), key=lambda item: item[1])
        reason = (f"the picks land better {name} ({value:.4g}, {z_of(value):.1f} standard "
                  f"deviations, against {statistic:.4g}): they were likely taken in the other "
                  "orientation of a mirror pair")
    elif z_score < FLAT_Z:
        verdict = "flat"
        reason = (f"{ranked}: their boxes hold no more than boxes anywhere, so a scale, an axis "
                  "or the orientation they were taken in is likely wrong")
    elif z_score >= LANDS_Z:
        verdict = "lands"
        reason = f"{ranked}; no mirror of them lands better"
    else:
        verdict = "weak"
        reason = ranked
    return PicksCheck(count, int(inside.sum()), len(kept), box, statistic, decoy_statistics,
                      beaten, z_score, mirror_statistics, verdict, reason)


__all__ = ["PicksCheck", "average_contrast", "box_averages", "box_starts", "check_picks"]
