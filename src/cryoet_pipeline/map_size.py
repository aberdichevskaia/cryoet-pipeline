"""How large the density in a map is, at a contour level.

A lane's particle diameter sets the template's mask and box, the angular step
and the extraction box (docs/agent/lane_questions.md, question 2), and the
person asked for it often does not know it. The template map does: above the
authors' contour, its density is the particle as they published it.
`scripts/show_emdb.py measure` reads it with this module.

Two numbers, because prepare-template neither centres the map nor turns it
about anything but its box centre (`template_center` is off unless asked for):

- the widest extent, the largest distance across the density (its Feret
  diameter), which is what a person measures in ChimeraX;
- twice the distance from the box centre to the farthest density: the diameter
  a mask drawn around that centre needs to hold it. It is never less than the
  widest extent, and it is more when the particle sits off the centre.

This is not `backends.template._density_extent_px`, which reads per-axis
profiles of the low-passed template at a tenth of their peak, so that a QC
warning holds up in noise. Here the level is the authors', on the map as it was
deposited.

The density above the level falls into pieces. Those of at least `KEEP_SHARE`
of the largest are measured together, so that a complex that comes apart at
the contour is measured whole; the rest, noise mostly, is left out and counted.
Density that reaches a face of the box is more than one particle. Its sizes
are then the map's, which `faces_reached` says.

What reaches the faces is told on the map's outermost voxels. Noise falls as far
below their median as the level is above it as often as it rises above the
level; a membrane, a neighbour or a lattice adds density above the level only.
When the density reaches a face and the outermost voxels are noise by that count
(`EDGE_NOISE`), the level is in the noise (`in_noise`): noise above it has joined
the particle into one piece, as on EMD-4805, whose authors' contour is 0.03 sd of
the map. The level is not judged in sds: EMD-4805's noise is strong in a ball
about the centre and weak in the box's corners, and no one sd describes both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mrcfile  # type: ignore[import-untyped]
import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]
from scipy.spatial import ConvexHull, QhullError  # type: ignore[import-untyped]
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

KEEP_SHARE = 0.1
"""A piece at least this share of the largest piece, by volume, is measured with it.

A complex that comes apart at the contour stays whole, and a speck of noise is
left out. What is left out is counted, and the largest of it reported, so that
nothing is dropped unseen.
"""

MOST_PIECES = 4
"""More kept pieces than this are not taken for one particle.

A complex can come apart into a few pieces at the contour. Hundreds of pieces
of much the same size are noise, or a level above the particle's density: on
EMD-16721, five times its authors' contour leaves 321 specks, which measured
together would have proposed a 202 A ribosome.
"""

LEVEL_STEP = 1.5
"""The second level the size is read at, as a multiple of the first.

The authors' contour is a choice made for display, and the size moves with the
level. How far is shown, not judged: on EMD-16721, a ribosome, the mask's
diameter falls 18% at 1.5 times the authors' contour, so a threshold that
flagged a membrane would flag ordinary maps too.
"""

EDGE_NOISE = 0.5
"""What reaches the faces is noise when, of the map's outermost voxels, those as far
below their median as the level is above it are at least this share of those above
the level.

Noise falls below as often as it rises above, so the share is near 1 where noise
reaches the faces: 0.90 to 1.01 for a ball in correlated noise at 0.5 to 1.5 sd
(two seeds), for EMD-16721 at 0.0015 and 0.001 (its authors' contour is 0.0033),
and for EMD-4805 from 0.04, its authors' contour, to 1.0, every level at which the
density reached a face. A membrane across the box gave 0.00 to 0.13 at 2.5 and
3.5 times its noise's sd, and 0.63 to 0.65 at 1.5 sd, where noise does join it.
One with a negative band two voxels wide along each side, 0.6 of its height deep,
as a sharpened map may have, gave 0.59 to 0.62 at 2.5 sd and 0.23 to 0.25 at
3.5 sd: it reads as noise until the level is raised.
"""

_ALL_NEIGHBOURS = np.ones((3, 3, 3), dtype=bool)
"""Voxels that touch at a face, an edge or a corner are one piece.

Measured on EMD-16721, a ribosome, at its authors' contour, which lies 2.9 sd
above the noise in the map's outer shell: with pieces of face neighbours only,
the ribosome comes apart, its largest piece holds 45% of the volume above the
level and is 245 A across; with all 26 neighbours it holds 73% and is 276 A
across. A speck of noise touching the particle may join it; the mask's 3 px
margin is larger than such a speck.
"""


@dataclass(frozen=True)
class MapSize:
    """The density above one level, measured."""

    level: float
    voxel_size_angstrom: float
    box_px: tuple[int, int, int]
    """The map's box, x, y, z."""
    pieces: int
    """The connected pieces above the level."""
    kept: int
    """Of them, the pieces measured: at least `KEEP_SHARE` of the largest."""
    kept_share: float
    """The kept pieces' share of the volume above the level."""
    largest_left_out: float
    """The largest piece left out, as a share of the largest piece; 0 when none is."""
    widest_angstrom: float
    """The largest distance across the kept density."""
    from_centre_angstrom: float
    """The distance from the box centre to the farthest kept density."""
    faces_reached: int
    """The faces of the box, of 6, that the kept density touches."""
    inside_angstrom: float
    """How far the kept density stays from the nearest face; 0 when it reaches one."""
    share_of_map: float
    """The volume above the level as a share of the map's."""
    edge_median: float
    """The median of the map's outermost voxels, those on its six faces."""
    edge_above: float
    """The share of the outermost voxels above the level."""
    edge_below: float
    """The share of the outermost voxels as far below their median as the level is
    above it: below 2 * `edge_median` - level."""

    @property
    def mask_diameter_angstrom(self) -> float:
        """The diameter a mask around the box centre needs to hold the kept density."""

        return 2.0 * self.from_centre_angstrom

    @property
    def in_noise(self) -> bool:
        """The density reaches a face, and what reaches the faces is noise: the level
        is in the noise (`EDGE_NOISE`)."""

        return self.faces_reached > 0 and self.edge_below >= EDGE_NOISE * self.edge_above


def read_map(path: Path) -> tuple[np.ndarray, float]:
    """A map's density and its voxel size in A, refusing what cannot be measured as is."""

    with mrcfile.open(path, permissive=True) as mrc:
        if mrc.data is None:
            raise ValueError(f"{path} is not a map mrcfile can read")
        data = np.asarray(mrc.data, dtype=np.float32)
        sizes = (float(mrc.voxel_size.x), float(mrc.voxel_size.y), float(mrc.voxel_size.z))
    if data.ndim != 3:
        raise ValueError(f"{path} is not a 3D map: its shape is {data.shape}")
    if not all(math.isfinite(size) and size > 0.0 for size in sizes):
        raise ValueError(f"{path} gives no voxel size in its header: {sizes}")
    if max(sizes) - min(sizes) > 1e-3 * max(sizes):
        raise ValueError(f"{path} has voxels of unequal sides, {sizes} A: distances "
                         "across it would depend on their direction")
    return data, sizes[0]


def measure_density(data: np.ndarray, voxel_size_angstrom: float, level: float) -> MapSize:
    """Measure the density above `level` in a map of cubic voxels."""

    volume = np.asarray(data)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D map, got shape {volume.shape}")
    above = volume > level
    if not above.any():
        raise ValueError(f"nothing in the map is above {level:g}; its maximum is "
                         f"{float(volume.max()):g}")
    labels, pieces = ndimage.label(above, structure=_ALL_NEIGHBOURS)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.max())
    keep = sizes >= KEEP_SHARE * largest
    keep[0] = False
    kept = keep[labels]
    del labels
    left_out = sizes[~keep]

    surface = kept & ~ndimage.binary_erosion(kept)
    points = np.argwhere(surface).astype(np.float64)
    outer = _outer_points(points)
    centre = (np.asarray(volume.shape, dtype=np.float64) - 1.0) / 2.0
    # Voxel centres are measured, and each end voxel reaches half a voxel further.
    widest = (_largest_distance(outer) + 1.0) * voxel_size_angstrom
    from_centre = (float(np.sqrt(((outer - centre) ** 2).sum(axis=1)).max()) + 0.5) \
        * voxel_size_angstrom
    # The voxels left empty between the density and each face, low side then high.
    gaps = np.concatenate([points.min(axis=0), np.asarray(volume.shape) - 1 - points.max(axis=0)])
    edge = _outermost(volume)
    edge_median = float(np.median(edge))
    return MapSize(
        level=level,
        voxel_size_angstrom=voxel_size_angstrom,
        box_px=(int(volume.shape[2]), int(volume.shape[1]), int(volume.shape[0])),
        pieces=int(pieces),
        kept=int(keep.sum()),
        kept_share=float(sizes[keep].sum()) / float(sizes.sum()),
        largest_left_out=float(left_out.max()) / largest if left_out.size else 0.0,
        widest_angstrom=widest,
        from_centre_angstrom=from_centre,
        faces_reached=int((gaps == 0).sum()),
        inside_angstrom=float(gaps.min()) * voxel_size_angstrom,
        share_of_map=float(sizes.sum()) / float(volume.size),
        edge_median=edge_median,
        edge_above=float((edge > level).mean()),
        edge_below=float((edge < 2.0 * edge_median - level).mean()),
    )


def _outermost(volume: np.ndarray) -> np.ndarray:
    """The voxels on the six faces of the box, each once."""

    if min(volume.shape) < 3:
        return volume.ravel()
    return np.concatenate([
        volume[0].ravel(), volume[-1].ravel(),
        volume[1:-1, 0].ravel(), volume[1:-1, -1].ravel(),
        volume[1:-1, 1:-1, 0].ravel(), volume[1:-1, 1:-1, -1].ravel(),
    ])


def _outer_points(points: np.ndarray) -> np.ndarray:
    """The points on the convex hull: the farthest pair, and the point farthest from any
    centre, are among them."""

    if len(points) < 5:
        return points
    try:
        hull = ConvexHull(points)
    except QhullError:
        # A flat or straight piece: joggled, qhull still finds its corners.
        hull = ConvexHull(points, qhull_options="QJ")
    vertices: np.ndarray = points[hull.vertices]
    return vertices


def _largest_distance(points: np.ndarray, chunk: int = 512) -> float:
    if len(points) < 2:
        return 0.0
    return max(float(cdist(points[start:start + chunk], points).max())
               for start in range(0, len(points), chunk))
