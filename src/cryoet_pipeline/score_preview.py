"""A picture of the correlation map, small enough to keep from every match.

pytom's correlation volume is what template matching actually produced: for
every voxel, the best score over all orientations. It is also about a gigabyte
per series at bin8, which is why `--pytom-keep-score-volume` is off by default —
and why, without it, nothing could show afterwards what the map looked like.

A picture does not need the volume. The maximum over a slab one particle
diameter thick keeps every peak whose centre lies in that slab, and a stack of
such slabs covering the whole depth is megabytes rather than a gigabyte.

Max rather than mean because a peak is what the picks were read off: averaging
a slab would dilute a peak in one slice with the noise of all the others. One
particle diameter because that is the depth one particle occupies: thinner and
a particle's peak could be split between two slabs, thicker and particles at
different depths project onto one another.

The volume is read one slab at a time through a memory map, so a bin4 map,
eight times the size, costs no more memory than one slab of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np


@dataclass(frozen=True)
class ScorePreview:
    """Slab maxima of a correlation volume, in the volume's own voxel frame."""

    slab_max: np.ndarray
    """(slabs, Y, X): the highest score in each slab, per column."""
    slab_starts: np.ndarray
    """First z index of each slab."""
    slab_stops: np.ndarray
    """One past the last z index of each slab."""
    voxel_spacing_angstrom: float | None = None

    @property
    def depth(self) -> int:
        return int(self.slab_stops[-1])

    def slab_containing(self, z_px: float) -> int:
        """The slab whose z range holds this coordinate, clamped to the volume.

        A pick's z is a voxel index rounded the same way extraction rounds it,
        so the slab that holds `round(z)` is the slab whose maximum includes
        that pick's own peak.
        """

        z = min(max(int(round(z_px)), 0), self.depth - 1)
        return int(np.searchsorted(self.slab_stops, z, side="right"))


def slab_bounds(depth: int, thickness: int) -> list[tuple[int, int]]:
    """Consecutive [start, stop) ranges of `thickness` covering `depth` exactly.

    The last slab is shorter when the depth is not a multiple of the thickness,
    rather than reaching past the volume or leaving its top slices out.
    """

    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    if thickness < 1:
        raise ValueError(f"slab thickness must be positive, got {thickness}")
    return [(start, min(start + thickness, depth)) for start in range(0, depth, thickness)]


def score_preview_from_volume(
    volume: Any,
    *,
    slab_thickness_px: int,
    voxel_spacing_angstrom: float | None = None,
) -> ScorePreview:
    """Slab maxima of a (Z, Y, X) volume, reading one slab at a time."""

    shape = tuple(int(size) for size in volume.shape)
    if len(shape) != 3:
        raise ValueError(f"expected a (Z, Y, X) volume, got shape {shape}")
    bounds = slab_bounds(shape[0], slab_thickness_px)
    slab_max = np.empty((len(bounds), shape[1], shape[2]), dtype=np.float32)
    for index, (start, stop) in enumerate(bounds):
        slab = np.asarray(volume[start:stop], dtype=np.float32)
        if not np.isfinite(slab).all():
            raise ValueError(f"score volume has nonfinite values in slices {start}..{stop - 1}")
        slab_max[index] = slab.max(axis=0)
    return ScorePreview(
        slab_max=slab_max,
        slab_starts=np.asarray([start for start, _ in bounds], dtype=np.int64),
        slab_stops=np.asarray([stop for _, stop in bounds], dtype=np.int64),
        voxel_spacing_angstrom=voxel_spacing_angstrom,
    )


def score_preview_from_mrc(path: Path, *, slab_thickness_px: int) -> ScorePreview:
    """Slab maxima of a correlation volume pytom wrote as MRC.

    pytom transposes on read and again on write, so the file holds the same
    (Z, Y, X) layout as the tomogram it was given — the layout pick coordinates
    index. A reader that doubts this can compare the map at each pick with the
    pick's own score, which the report does.
    """

    with mrcfile.mmap(path, mode="r", permissive=True) as handle:
        voxel = float(handle.voxel_size.x) if handle.voxel_size.x > 0 else None
        return score_preview_from_volume(
            handle.data,
            slab_thickness_px=slab_thickness_px,
            voxel_spacing_angstrom=voxel,
        )


def write_score_preview(preview: ScorePreview, path: Path) -> None:
    """Store a preview as compressed half floats.

    Half precision keeps three significant digits, which is more than a picture
    needs and halves the file: scores are of order 0.1 and their differences
    that matter of order 0.01.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            slab_max=preview.slab_max.astype(np.float16),
            slab_starts=preview.slab_starts,
            slab_stops=preview.slab_stops,
            voxel_spacing_angstrom=np.float64(
                preview.voxel_spacing_angstrom
                if preview.voxel_spacing_angstrom is not None
                else np.nan
            ),
        )


def read_score_preview(path: Path) -> ScorePreview:
    with np.load(path) as stored:
        voxel = float(stored["voxel_spacing_angstrom"])
        return ScorePreview(
            slab_max=np.asarray(stored["slab_max"], dtype=np.float32),
            slab_starts=np.asarray(stored["slab_starts"], dtype=np.int64),
            slab_stops=np.asarray(stored["slab_stops"], dtype=np.int64),
            voxel_spacing_angstrom=voxel if np.isfinite(voxel) else None,
        )
