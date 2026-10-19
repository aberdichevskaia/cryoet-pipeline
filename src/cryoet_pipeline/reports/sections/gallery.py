"""Galleries of picks by score, each particle shown as the search saw it and in one frame.

Three choices make one tile comparable with another, and with the template at the
top of the page:

- **The search's low-pass.** pytom compared the template with the tomogram at that
  resolution, so finer detail played no part in choosing a pick, and showing it
  only adds noise.
- **The template's frame.** pytom's angles turn each particle the way
  subtomogram averaging turns it: the same rotation function, the same rotation
  centre half a voxel off an even template's pick. Particles of one structure then
  look alike and like the template, and a tile that does not is a pick whose
  particle, or whose orientation, is in doubt.
- **A slab, not a slice.** Each view is the mean of a slab about 10 nm thick
  through the particle's centre, as on the tomogram pages.

Random positions are turned too, by the angles of randomly chosen picks, so the
control differs from the picks only in where it was taken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure

from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.particles import pick_center_offset_px
from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import SeriesReport, TemplateInfo
from cryoet_pipeline.reports.imaging import (
    Band,
    central_views,
    crop_centered,
    display_range,
    random_positions,
    slab_half_width_px,
    smooth,
    stable_seed,
)
from cryoet_pipeline.reports.sections.tomogram import open_volume
from cryoet_pipeline.rotations import (
    center_particle_volume,
    relion_euler_to_matrix,
    resample_volume_zyx,
)

TILES_PER_BAND = 8
"""Gallery tiles per band: two rows of four on a landscape page."""

BAND_TITLES = {"high": "High scores", "middle": "Middle scores", "low": "Low scores"}

_CPU = torch.device("cpu")


@dataclass(frozen=True)
class Tile:
    image: np.ndarray
    """XY | XZ | YZ side by side, NaN in the gaps and outside the volume."""
    title: str


@dataclass(frozen=True)
class GalleryBand:
    key: str
    header: str
    tiles: list[Tile]


@dataclass(frozen=True)
class Gallery:
    bands: dict[str, GalleryBand]
    limits: tuple[float, float]
    template: Tile | None
    template_limits: tuple[float, float] | None
    aligned: bool
    """Whether the particles were turned by pytom's angles; False when the list has none."""
    low_pass_angstrom: float | None
    slab_px: int
    voxel_spacing_angstrom: float


def gallery_half_width(report: SeriesReport) -> int:
    """Half the tile, in voxels: the particle and about one radius of its surroundings."""

    radius = report.particle_radius_px
    half = math.ceil(2.0 * radius)
    if report.template is not None:
        half = min(half, report.template.box_px // 2)
    return max(4, half)


def oriented_views(
    volume: Any,
    centre_zyx: tuple[float, float, float],
    matrix: np.ndarray | None,
    *,
    offset_px: float,
    half: int,
    low_pass_angstrom: float | None,
    voxel_spacing_angstrom: float,
    slab_px: int,
) -> list[np.ndarray]:
    """XY, XZ and YZ slabs through one particle, low-passed and turned into the template frame.

    The crop reaches √3 times the tile's half width, so that no turn brings a
    corner of the tile in from outside it; it is filtered whole, so the filter
    has room at the tile's edges too. Turning samples the crop at
    `matrix @ x + shift` for each voxel `x` of the tile, as averaging does, with
    the shift putting the tile's centre on the pick's rotation centre:
    `offset_px` above it along every axis for an even template box.
    """

    reach = math.ceil(half * math.sqrt(3.0)) + 2
    voxel = tuple(int(round(value)) for value in centre_zyx)
    crop = crop_centered(volume, voxel, reach)
    inside = np.isfinite(crop)
    side = 2 * half + 1
    if not inside.any():
        return [np.full((side, side), np.nan, dtype=np.float32)] * 3
    particle = torch.from_numpy(np.where(inside, crop, float(crop[inside].mean())))
    particle = center_particle_volume(particle.to(torch.float32))
    if low_pass_angstrom is not None:
        particle = low_pass_filter_volume(
            particle,
            cutoff_angstrom=low_pass_angstrom,
            pixel_spacing_angstrom=voxel_spacing_angstrom,
        )
    z, y, x = centre_zyx
    shift = np.array(
        [x - offset_px - voxel[2], y - offset_px - voxel[1], z - offset_px - voxel[0]]
    )
    rotation = matrix if matrix is not None else np.eye(3)
    turned = resample_volume_zyx(particle, rotation, shift, device=_CPU).numpy()
    covered = resample_volume_zyx(
        torch.from_numpy(inside.astype(np.float32)), rotation, shift, device=_CPU
    ).numpy()
    if low_pass_angstrom is None:
        turned = np.stack([smooth(plane) for plane in turned])
    window = slice(reach - half, reach + half + 1)
    box = turned[window, window, window]
    box = np.where(covered[window, window, window] >= 0.5, box, np.nan).astype(np.float32)
    return central_views(box, slab_px)


def template_views(
    template: TemplateInfo, *, half: int, low_pass_angstrom: float | None, slab_px: int
) -> list[np.ndarray]:
    """The template's three central slabs, filtered as the tiles are, on the tiles' grid."""

    volume = torch.from_numpy(template.template.astype(np.float32))
    if low_pass_angstrom is not None and template.voxel_spacing_angstrom is not None:
        volume = low_pass_filter_volume(
            volume,
            cutoff_angstrom=low_pass_angstrom,
            pixel_spacing_angstrom=template.voxel_spacing_angstrom,
        )
    side = 2 * half + 1
    return [_fit(view, side) for view in central_views(volume.numpy(), slab_px)]


def _fit(view: np.ndarray, side: int) -> np.ndarray:
    """Cut or pad a view, about its centre, to `side` x `side`; padding is NaN."""

    out = np.full((side, side), np.nan, dtype=np.float32)
    source = []
    target = []
    for size in view.shape:
        start = (size - side) // 2
        lo, hi = max(0, start), min(size, start + side)
        source.append(slice(lo, hi))
        target.append(slice(lo - start, hi - start))
    out[tuple(target)] = view[tuple(source)]
    return out


def _tile_image(views: list[np.ndarray]) -> np.ndarray:
    gap = np.full((views[0].shape[0], max(1, views[0].shape[1] // 10)), np.nan, np.float32)
    return np.concatenate([views[0], gap, views[1], gap, views[2]], axis=1)


def build_gallery(report: SeriesReport, bands: list[Band]) -> Gallery:
    picks = report.picks
    info = report.tomogram
    assert picks is not None and info is not None
    volume = open_volume(info.artifact.path)
    half = gallery_half_width(report)
    voxel = info.voxel_spacing_angstrom
    slab_px = min(2 * slab_half_width_px(voxel) + 1, 2 * half + 1)
    low_pass = picks.low_pass_angstrom
    aligned = picks.has_orientations
    offset = (
        pick_center_offset_px(picks.template_box_px) if picks.template_box_px is not None else 0.0
    )
    ranked = picks.ranked()
    sigma = picks.sigma

    def matrix(index: int) -> np.ndarray | None:
        if not aligned:
            return None
        assert picks.rot_deg is not None and picks.tilt_deg is not None
        assert picks.psi_deg is not None
        return relion_euler_to_matrix(
            float(picks.rot_deg[index]), float(picks.tilt_deg[index]), float(picks.psi_deg[index])
        )

    def views(centre: tuple[float, float, float], rotation: np.ndarray | None,
              offset_px: float) -> list[np.ndarray]:
        return oriented_views(volume, centre, rotation, offset_px=offset_px, half=half,
                              low_pass_angstrom=low_pass, voxel_spacing_angstrom=voxel,
                              slab_px=slab_px)

    gallery: dict[str, GalleryBand] = {}
    for band in bands:
        tiles = []
        for position in band.positions:
            index = int(ranked[position])
            centre = (float(picks.z_px[index]), float(picks.y_px[index]),
                      float(picks.x_px[index]))
            label = (f"{sigma[index]:.1f}σ" if sigma is not None
                     else f"{picks.score[index]:.3f}")
            tiles.append(Tile(_tile_image(views(centre, matrix(index), offset)),
                              f"rank {position + 1} · {label}"))
        header = (
            f"{BAND_TITLES[band.name]} · ranks {band.first_rank}–{band.last_rank} "
            f"of {picks.count}"
        )
        if sigma is not None:
            first = sigma[ranked[band.first_rank - 1]]
            last = sigma[ranked[band.last_rank - 1]]
            header += f" · {first:.1f} to {last:.1f}σ"
        gallery[band.name] = GalleryBand(band.name, header, tiles)

    seed = stable_seed(report.series)
    positions = random_positions(info.shape, half, picks.z_px, TILES_PER_BAND, seed)
    borrowed = np.random.default_rng(seed + 1).integers(0, picks.count, size=len(positions))
    gallery["random"] = GalleryBand(
        "random",
        "Random positions · anywhere in x and y, at z values drawn from the picks', "
        "turned by the angles of randomly chosen picks",
        [
            Tile(_tile_image(views((float(z), float(y), float(x)), matrix(int(pick)), 0.0)),
                 f"random {number}")
            for number, ((z, y, x), pick) in enumerate(zip(positions, borrowed, strict=True), 1)
        ],
    )
    limits = display_range([one.image for band in gallery.values() for one in band.tiles])

    template_tile = None
    template_limits = None
    if report.template is not None:
        image = _tile_image(template_views(report.template, half=half, low_pass_angstrom=low_pass,
                                           slab_px=slab_px))
        template_tile = Tile(image, "Template, shown the same way")
        template_limits = display_range([image])
    return Gallery(gallery, limits, template_tile, template_limits, aligned, low_pass, slab_px,
                   voxel)


def gallery_page(report: SeriesReport, gallery: Gallery, keys: tuple[str, str]) -> Figure:
    shown = [gallery.bands[key] for key in keys if key in gallery.bands]
    names = " and ".join(band.key for band in shown)
    figure = style.new_page(f"{report.series} · Picks by score: {names}", _subtitle(gallery))
    cmap = style.GRAY_WITH_GAPS
    if gallery.template is not None and gallery.template_limits is not None:
        axes = figure.add_axes((0.765, 0.868, 0.2, 0.075))
        axes.imshow(gallery.template.image, cmap=cmap, vmin=gallery.template_limits[0],
                    vmax=gallery.template_limits[1], origin="lower")
        style.image_axes(axes)
        axes.set_title(gallery.template.title, loc="left", fontsize=style.FONT_SMALL)
    top, bottom = 0.86, style.PICTURE_BOTTOM - 0.01
    block = (top - bottom) / 2.0
    columns = 4
    left, right, gap = 0.035, 0.965, 0.012
    width = (right - left - gap * (columns - 1)) / columns
    row_height = (block - 0.055) / 2.0
    for block_index, band in enumerate(shown):
        block_top = top - block_index * block
        figure.text(left, block_top - 0.005, band.header, fontsize=style.FONT_TEXT,
                    fontweight="bold", va="top")
        for number, one in enumerate(band.tiles):
            row, column = divmod(number, columns)
            axes = figure.add_axes((
                left + column * (width + gap),
                block_top - 0.045 - (row + 1) * row_height,
                width,
                row_height - 0.03,
            ))
            axes.imshow(one.image, cmap=cmap, vmin=gallery.limits[0], vmax=gallery.limits[1],
                        origin="lower")
            style.image_axes(axes, one.title)
    style.caption(figure, _caption(gallery))
    return figure


def _subtitle(gallery: Gallery) -> str:
    parts = ["XY | XZ | YZ through each particle's centre"]
    if gallery.aligned:
        parts.append("turned to the template's orientation")
    if gallery.low_pass_angstrom is not None:
        parts.append(f"low-passed to {gallery.low_pass_angstrom:g} Å")
    return " · ".join(parts)


def _caption(gallery: Gallery) -> str:
    thickness_nm = gallery.slab_px * gallery.voxel_spacing_angstrom / 10.0
    text = f"Each view is the mean of a {gallery.slab_px}-voxel ({thickness_nm:.0f} nm) slab. "
    if gallery.low_pass_angstrom is not None:
        text += (f"Low-passed to {gallery.low_pass_angstrom:g} Å, the resolution the search "
                 "ran at. ")
    else:
        text += "The search recorded no low-pass, so the tiles are smoothed by one voxel. "
    if gallery.aligned:
        text += ("Each particle is turned by the angles pytom found, as averaging turns it, so "
                 "particles of one kind look alike and like the template. ")
    else:
        text += "This pick list has no angles, so the particles are shown as they lie. "
    text += ("One grey scale for all tiles of this tilt-series. Ranks are spread evenly through "
             "each third. If the low third looks like the random positions, those picks are "
             "probably not particles.")
    return text
