"""Volumes into pictures: slabs, crops, depths and gallery bands. Numpy only.

Three choices here are not cosmetic, because each has cost a round trip on
this project before:

- **Slabs, not slices.** A single bin8 slice is noisy enough to have misled
  this investigation more than once (scripts/review_package.sh says so), so
  every tomogram picture is the mean of a slab about 10 nm thick, lightly
  smoothed in plane, and says so in its caption.
- **One grey scale per series.** Each gallery tile stretched to its own range
  makes noise look like structure; on a shared scale a box with nothing in it
  looks as faint as it is.
- **Protein stays dark.** The values are shown as reconstructed, as in 3dmod,
  rather than negated as some of the scripts here do.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cryoet_pipeline.score_preview import ScorePreview

SLAB_ANGSTROM = 100.0
"""How thick a tomogram picture's slab is: 10 nm, about the size of a ribosome."""

SMOOTHING_SIGMA_PX = 1.0
"""The in-plane Gaussian every picture is smoothed with, in voxels."""

GALLERY_SLAB_PX = 5
"""How many voxels a gallery tile's central slab averages."""

MAX_DRAWN_PX = 512
"""Pictures larger than this are block-averaged down before they go into the PDF."""


def slab_half_width_px(voxel_spacing_angstrom: float) -> int:
    """Half the slab, in voxels, so that 2 h + 1 voxels span about `SLAB_ANGSTROM`."""

    return max(0, round(SLAB_ANGSTROM / voxel_spacing_angstrom / 2.0 - 0.5))


def slab_mean(volume: Any, axis: int, center: int, half_width: int) -> np.ndarray:
    """The mean of the slices `center - h .. center + h` along one axis, clipped to the volume."""

    size = int(volume.shape[axis])
    start = max(0, center - half_width)
    stop = min(size, center + half_width + 1)
    if start >= stop:
        raise ValueError(f"slab at {center} lies outside an axis of {size}")
    index: list[slice] = [slice(None)] * 3
    index[axis] = slice(start, stop)
    block = np.asarray(volume[tuple(index)], dtype=np.float32)
    return np.asarray(block.mean(axis=axis))


def smooth(image: np.ndarray, sigma_px: float = SMOOTHING_SIGMA_PX) -> np.ndarray:
    """Gaussian low-pass in Fourier space; NaN padding stays NaN.

    NaN marks the outside of the volume in a crop. It is filled with the mean
    for the transform, so that the edge does not ring, and put back afterwards.
    """

    if sigma_px <= 0.0:
        return image
    missing = ~np.isfinite(image)
    filled = np.where(missing, np.nanmean(image) if (~missing).any() else 0.0, image)
    rows = np.fft.fftfreq(image.shape[0])[:, None]
    columns = np.fft.rfftfreq(image.shape[1])[None, :]
    response = np.exp(-2.0 * (math.pi * sigma_px) ** 2 * (rows**2 + columns**2))
    smoothed = np.fft.irfft2(np.fft.rfft2(filled) * response, s=image.shape)
    smoothed[missing] = np.nan
    return smoothed.astype(np.float32)


def reduction_factor(shape: Sequence[int], max_px: int = MAX_DRAWN_PX) -> int:
    return max(1, math.ceil(max(shape) / max_px))


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Average `factor` x `factor` blocks, dropping a remainder at the far edges.

    A drawing that places the result with an extent of `(0, n * factor)` in the
    original pixels keeps every overlay in place, because block k covers
    original pixels `k * factor .. (k + 1) * factor - 1`.
    """

    if factor == 1:
        return image
    height = image.shape[0] // factor
    width = image.shape[1] // factor
    trimmed = image[: height * factor, : width * factor]
    return np.asarray(trimmed.reshape(height, factor, width, factor).mean(axis=(1, 3)))


def block_max(image: np.ndarray, factor: int) -> np.ndarray:
    """Like `block_mean`, keeping each block's highest value: a reduced map keeps its peaks."""

    if factor == 1:
        return image
    height = image.shape[0] // factor
    width = image.shape[1] // factor
    trimmed = image[: height * factor, : width * factor]
    return np.asarray(trimmed.reshape(height, factor, width, factor).max(axis=(1, 3)))


def mean_over_depths(volume: Any, start: int, stop: int, chunk: int = 16) -> np.ndarray:
    """The mean of slices `start .. stop - 1`, read a few slices at a time."""

    start = max(0, start)
    stop = min(int(volume.shape[0]), stop)
    if start >= stop:
        raise ValueError(f"no slices between {start} and {stop}")
    total = np.zeros(volume.shape[1:], dtype=np.float64)
    for first in range(start, stop, chunk):
        total += np.asarray(volume[first : min(stop, first + chunk)], dtype=np.float64).sum(axis=0)
    return (total / (stop - start)).astype(np.float32)


def display_range(
    images: Sequence[np.ndarray], low: float = 1.0, high: float = 99.0
) -> tuple[float, float]:
    """One grey scale for a set of pictures: the given percentiles of all their pixels."""

    values = np.concatenate([image[np.isfinite(image)].ravel() for image in images])
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = np.percentile(values, [low, high])
    if upper <= lower:
        upper = lower + 1.0
    return float(lower), float(upper)


def crop_centered(volume: Any, center_zyx: Sequence[int], half: int) -> np.ndarray:
    """A cube of side 2 h + 1 centred on a voxel, NaN where it leaves the volume.

    Odd, so the pick's own voxel is the centre of every view; extraction's
    even boxes put it half a voxel off, which does not matter for a picture
    and would for nothing else here.
    """

    side = 2 * half + 1
    box = np.full((side, side, side), np.nan, dtype=np.float32)
    source: list[slice] = []
    target: list[slice] = []
    for axis, center in enumerate(center_zyx):
        size = int(volume.shape[axis])
        start = center - half
        lo = max(0, start)
        hi = min(size, start + side)
        if lo >= hi:
            return box
        source.append(slice(lo, hi))
        target.append(slice(lo - start, hi - start))
    box[tuple(target)] = np.asarray(volume[tuple(source)], dtype=np.float32)
    return box


def central_views(box: np.ndarray, slab_px: int = GALLERY_SLAB_PX) -> list[np.ndarray]:
    """XY, XZ and YZ: central slabs through a cube, each with its second axis horizontal."""

    half = slab_px // 2
    views = []
    for axis in range(3):
        middle = box.shape[axis] // 2
        indices = range(max(0, middle - half), min(box.shape[axis], middle + half + 1))
        with np.errstate(invalid="ignore"):
            views.append(np.nanmean(np.take(box, list(indices), axis=axis), axis=axis))
    # take() along z leaves (Y, X); along y (Z, X); along x (Z, Y).
    return views


@dataclass(frozen=True)
class Band:
    name: str
    first_rank: int
    """1-based, inclusive."""
    last_rank: int
    positions: np.ndarray
    """Positions in the ranked order that the gallery shows, spread evenly through the band."""


def rank_bands(count: int, per_band: int) -> list[Band]:
    """Thirds of the ranked picks, each sampled at evenly spaced ranks.

    Evenly spaced rather than random, so that a band's tiles reach from its
    best pick to its worst and a rerun shows the same picks.
    """

    edges = [0, round(count / 3), round(2 * count / 3), count]
    bands = []
    for name, lo, hi in zip(("high", "middle", "low"), edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        shown = min(per_band, hi - lo)
        positions = np.unique(np.rint(np.linspace(lo, hi - 1, shown)).astype(np.int64))
        bands.append(Band(name, lo + 1, hi, positions))
    return bands


def stable_seed(text: str) -> int:
    """A seed that is the same for the same text in every run and on every machine."""

    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def random_positions(
    shape_zyx: Sequence[int], half: int, depths: np.ndarray, count: int, seed: int
) -> np.ndarray:
    """Positions anywhere in the tomogram, at depths drawn from the picks' own.

    The control the low band is judged against: the same layers of the
    specimen, with no selection of particles. Drawing the depths from the picks
    means no estimate of where the specimen sits has to be trusted. x and y keep
    a box clear of the edge when the volume is large enough for that.
    """

    rng = np.random.default_rng(seed)
    positions = np.empty((count, 3), dtype=np.int64)
    positions[:, 0] = (
        np.rint(rng.choice(depths, size=count)).astype(np.int64)
        if len(depths)
        else shape_zyx[0] // 2
    )
    for axis in (1, 2):
        lo = half if shape_zyx[axis] > 2 * half else 0
        hi = shape_zyx[axis] - 1 - lo
        positions[:, axis] = rng.integers(lo, hi + 1, size=count)
    return positions


@dataclass(frozen=True)
class Depth:
    """One depth the pages look at: a centre, and the z range picks there are drawn from."""

    center: int
    start: int
    stop: int
    """One past the last z of the range."""


def choose_depths(
    depth: int,
    picks_z: np.ndarray | None,
    preview: ScorePreview | None,
    half_width: int,
) -> list[Depth]:
    """Three depths through the specimen, on the correlation map's slabs when there is one.

    Where the picks are is the best available estimate of where the specimen
    is. Without a map their quartiles set the depths, or, without picks, the
    middle of the volume and a slab either side. With a map, the depths are the
    centres of the three map slabs holding the most picks, nearest the median
    pick on a tie, so that a tomogram panel and the map panel below it show the
    same layer, and a thin specimen gets its neighbouring slabs rather than the
    same slab three times.
    """

    has_picks = picks_z is not None and picks_z.size > 0
    if has_picks:
        assert picks_z is not None
        targets = [float(value) for value in np.quantile(picks_z, [0.25, 0.5, 0.75])]
    else:
        targets = [depth * 3 / 8, depth / 2, depth * 5 / 8]
    if preview is None:
        depths = []
        for target in targets:
            center = min(max(round(target), 0), depth - 1)
            depths.append(
                Depth(center, max(0, center - half_width), min(depth, center + half_width + 1))
            )
        return depths

    slab_count = len(preview.slab_stops)
    counts = np.zeros(slab_count, dtype=np.int64)
    if has_picks:
        assert picks_z is not None
        for z in picks_z:
            counts[preview.slab_containing(float(z))] += 1
    middle = preview.slab_containing(targets[1])
    order = sorted(range(slab_count), key=lambda slab: (-counts[slab], abs(slab - middle)))
    chosen = sorted(order[:3])
    return [
        Depth(
            center=int((preview.slab_starts[slab] + preview.slab_stops[slab] - 1) // 2),
            start=int(preview.slab_starts[slab]),
            stop=int(preview.slab_stops[slab]),
        )
        for slab in chosen
    ]


def nice_length(target: float) -> float:
    """The largest of 1, 2 or 5 times a power of ten that does not exceed `target`."""

    if target <= 0:
        return 0.0
    power = 10.0 ** math.floor(math.log10(target))
    for multiple in (5.0, 2.0, 1.0):
        if multiple * power <= target:
            return multiple * power
    return power
