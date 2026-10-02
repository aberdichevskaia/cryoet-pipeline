"""Several maps side by side, sliced at the same places and shown on one scale.

Built for the question an average keeps raising: does this variant — another
backend, another refinement, one class out of a classification — look like the
others and like the template? Columns are maps, rows are planes through the box
centre, optionally repeated after a low-pass. Every choice below exists so that
a difference between two columns is a difference between the maps rather than
between their displays:

- **One grey scale for every panel.** Each map is z-scored on its own (so that
  arbitrary scale and offset, which differ between programs, drop out) and then
  every panel is drawn over the same fixed sigma range. Per-panel autoscaling
  would stretch a weak map until it looked as confident as a strong one.
- **The z-score is taken inside a sphere, over the whole volume.** Over the full
  box the statistics depend on how much empty corner there is, so the same
  particle would be scaled differently in a 48 box and a 64 box. Over the whole
  volume rather than per slice, because per slice a plane that misses the
  particle gets stretched into noise that looks like density.
- **Protein is dark in every column.** Averages from tomograms already have it
  dark; maps from EMDB, and anything simulated, have it bright. Which is which
  cannot be read off the file, so the caller says so per map (`protein_bright`)
  instead of the code guessing from a histogram.
- **Slices pass through the geometric centre, `(N - 1) / 2`**, the convention
  the rest of the pipeline uses since geometric-center-v1. For an even box that
  centre lies between two planes, and the slab is weighted accordingly instead
  of silently taking plane `N // 2`, which would be half a voxel off.
- **Low-pass rows use `low_pass_filter_volume`**, the same filter the averaging
  and FSC code use, and the z-score is recomputed after filtering, so each row
  is on the same sigma scale as every other.

Images are drawn with the origin at the lower left: in the xy panel x runs right
and y up, in xz x runs right and z up, in yz y runs right and z up — the way
IMOD and ChimeraX show them, so a feature can be found again there.

Maps must already share a voxel size and a box; the `compare-maps` command pads
smaller boxes onto the largest one with the same centring `compute-fsc` uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cryoet_pipeline.filters import low_pass_filter_volume

PLANES = ("xy", "xz", "yz")
"""Planes through the box centre. Arrays are indexed (z, y, x), as mrcfile reads them."""


@dataclass(frozen=True)
class ComparedMap:
    """One column of the comparison."""

    label: str
    volume: torch.Tensor
    protein_bright: bool = False


def load_map(path: Path) -> tuple[torch.Tensor, float | None]:
    """A 3D MRC and its header voxel size, or None when the header has none.

    Dynamo writes its maps with a zero voxel size, so a missing one is ordinary
    here rather than a sign of a broken file; the caller supplies it instead.
    """

    import mrcfile  # type: ignore[import-untyped]

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}: {path}")
    return torch.as_tensor(data), voxel_size if voxel_size > 0.0 else None


def sphere_mask(size: int, *, diameter_voxels: float) -> torch.Tensor:
    """Voxels whose centre lies inside a sphere about the geometric box centre."""

    if diameter_voxels <= 0.0:
        raise ValueError(f"sphere diameter must be positive, got {diameter_voxels}")
    axis = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return (z**2 + y**2 + x**2) <= (diameter_voxels / 2) ** 2


def z_score_inside(volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scale a volume so that the voxels under `mask` have mean 0 and SD 1."""

    inside = volume[mask].to(torch.float64)
    if inside.numel() < 2:
        raise ValueError("the normalisation sphere contains fewer than two voxels")
    deviation = float(inside.std())
    if deviation == 0.0 or not math.isfinite(deviation):
        raise ValueError("the map is constant inside the normalisation sphere")
    return ((volume.to(torch.float64) - float(inside.mean())) / deviation).to(torch.float32)


def central_slab_weights(size: int, thickness_voxels: float) -> np.ndarray:
    """How much of each plane falls inside a slab centred on `(size - 1) / 2`.

    Plane `i` covers `[i - 0.5, i + 0.5]`; its weight is the overlap with the
    slab, normalised to sum to one. A one-voxel slab in an odd box is exactly the
    middle plane; in an even box it is the two middle planes at one half each.
    """

    if thickness_voxels <= 0.0:
        raise ValueError(f"slab thickness must be positive, got {thickness_voxels}")
    if thickness_voxels > size:
        raise ValueError(f"slab of {thickness_voxels} voxels is thicker than the box ({size})")
    centre = (size - 1) / 2
    low, high = centre - thickness_voxels / 2, centre + thickness_voxels / 2
    planes = np.arange(size, dtype=np.float64)
    overlap = np.clip(np.minimum(planes + 0.5, high) - np.maximum(planes - 0.5, low), 0.0, None)
    normalised: np.ndarray = overlap / overlap.sum()
    return normalised


def central_section(volume: np.ndarray, plane: str, thickness_voxels: float) -> np.ndarray:
    """The slab-averaged section through the box centre, laid out for `origin="lower"`.

    Returned as (rows, columns) = (vertical axis, horizontal axis): (y, x) for
    xy, (z, x) for xz, (z, y) for yz.
    """

    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; expected one of {', '.join(PLANES)}")
    # The axis being averaged away: z for xy, y for xz, x for yz.
    axis = {"xy": 0, "xz": 1, "yz": 2}[plane]
    weights = central_slab_weights(volume.shape[axis], thickness_voxels)
    return np.tensordot(weights, np.moveaxis(volume, axis, 0), axes=1)


def _display_volume(
    compared: ComparedMap,
    *,
    mask: torch.Tensor,
    cutoff_angstrom: float | None,
    voxel_size_angstrom: float,
) -> np.ndarray:
    volume = compared.volume.to(torch.float32)
    if compared.protein_bright:
        volume = -volume
    if cutoff_angstrom is not None:
        volume = low_pass_filter_volume(
            volume, cutoff_angstrom=cutoff_angstrom, pixel_spacing_angstrom=voxel_size_angstrom
        )
    return z_score_inside(volume, mask).cpu().numpy()


def render_map_comparison(
    maps: Sequence[ComparedMap],
    *,
    out: Path,
    voxel_size_angstrom: float,
    planes: Sequence[str] = ("xy", "xz"),
    low_pass_angstrom: Sequence[float] = (),
    sphere_diameter_angstrom: float | None = None,
    sigma_range: tuple[float, float] = (-3.0, 1.5),
    slab_voxels: float = 1.0,
    title: str | None = None,
    panel_inches: float = 2.2,
    dpi: int = 150,
) -> None:
    """Draw the comparison grid and save it to `out` (format from the suffix).

    Rows are every plane unfiltered, then every plane again at each low-pass
    cutoff in turn. `sphere_diameter_angstrom` defaults to the sphere inscribed in
    the box.
    """

    if not maps:
        raise ValueError("nothing to compare: no maps given")
    shape = tuple(maps[0].volume.shape)
    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError(f"expected cubic volumes, got shape {shape} for {maps[0].label!r}")
    for compared in maps[1:]:
        if tuple(compared.volume.shape) != shape:
            raise ValueError(
                f"{compared.label!r} has shape {tuple(compared.volume.shape)}, "
                f"but {maps[0].label!r} has {shape}; pad them to one box first"
            )
    for plane in planes:
        if plane not in PLANES:
            raise ValueError(f"unknown plane {plane!r}; expected one of {', '.join(PLANES)}")
    if not planes:
        raise ValueError("at least one plane is needed")
    low, high = sigma_range
    if not low < high:
        raise ValueError(f"sigma range must be increasing, got {sigma_range}")

    size = shape[0]
    box_angstrom = size * voxel_size_angstrom
    diameter_angstrom = (
        box_angstrom if sphere_diameter_angstrom is None else sphere_diameter_angstrom
    )
    mask = sphere_mask(size, diameter_voxels=diameter_angstrom / voxel_size_angstrom)

    cutoffs: list[float | None] = [None, *low_pass_angstrom]
    rows: list[tuple[str, int]] = []  # (row label, index into cutoffs)
    for index, cutoff in enumerate(cutoffs):
        for plane in planes:
            rows.append((plane if cutoff is None else f"{plane}, {cutoff:g} Å", index))

    displayed = [
        [
            _display_volume(
                compared,
                mask=mask,
                cutoff_angstrom=cutoff,
                voxel_size_angstrom=voxel_size_angstrom,
            )
            for cutoff in cutoffs
        ]
        for compared in maps
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if title is None:
        title = (
            f"Same display: z-score inside a {diameter_angstrom:.0f} Å sphere, "
            f"common range {low:g}…{high:+g}σ, protein dark"
        )

    figure, axes = plt.subplots(
        len(rows),
        len(maps),
        figsize=(panel_inches * len(maps), panel_inches * len(rows) + 0.6),
        squeeze=False,
    )
    for row, (row_label, cutoff_index) in enumerate(rows):
        plane = row_label.split(",")[0]
        for column, compared in enumerate(maps):
            ax = axes[row][column]
            section = central_section(displayed[column][cutoff_index], plane, slab_voxels)
            ax.imshow(section, cmap="gray", vmin=low, vmax=high, origin="lower",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(compared.label, fontsize=10)
            if column == 0:
                ax.set_ylabel(row_label, fontsize=11)
    figure.suptitle(title, fontsize=12)
    figure.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=dpi)
    plt.close(figure)
