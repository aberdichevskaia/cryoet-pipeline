"""The picks: where they are, how they score against noise, the map, and what they look like."""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.patches import Circle

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import PickTable, SeriesReport
from cryoet_pipeline.reports.imaging import (
    GALLERY_SLAB_PX,
    Band,
    Depth,
    block_max,
    block_mean,
    central_views,
    crop_centered,
    display_range,
    mean_over_depths,
    random_positions,
    reduction_factor,
    slab_half_width_px,
    smooth,
    stable_seed,
)
from cryoet_pipeline.reports.sections.tomogram import (
    DEPTH_LETTERS,
    add_scale_bar,
    open_volume,
    picture,
    show,
)

TILES_PER_BAND = 8
"""Gallery tiles per band: two rows of four on a landscape page."""

BAND_TITLES = {"high": "High scores", "middle": "Middle scores", "low": "Low scores"}


def picks_page(report: SeriesReport, bands: list[Band]) -> Figure:
    picks = report.picks
    assert picks is not None
    figure = style.new_page(f"{report.series} · Picks", _subtitle(report))
    grid = GridSpec(
        2, 3, figure=figure, left=0.055, right=0.975, top=0.84, bottom=0.16,
        hspace=0.45, wspace=0.3,
    )
    _map_panel(figure.add_subplot(grid[0, 0]), report)
    _height_panel(figure.add_subplot(grid[0, 1]), report)
    _template_panel(figure, grid[0, 2], report)
    _rank_panel(figure.add_subplot(grid[1, 0]), picks, bands)
    _noise_panel(figure.add_subplot(grid[1, 1]), picks)
    _numbers_panel(figure.add_subplot(grid[1, 2]), report)
    style.caption(
        figure,
        "σ: a score divided by the standard deviation of the search's own noise. The "
        "one-false-positive height is the score noise alone reaches about once in this whole "
        "search: a pick above it is unlikely to be noise on its own. Below it, picks count as "
        "evidence only as a group, through how many more there are than noise alone gives "
        "(the gap between the two lines, bottom middle). Dots on the score curve are the picks "
        "shown in the galleries.",
    )
    return figure


def _subtitle(report: SeriesReport) -> str:
    picks = report.picks
    assert picks is not None
    parts = [f"{picks.count} picks"]
    if picks.above_floor is not None and picks.floor_sigma is not None:
        parts.append(
            f"{picks.above_floor} above the one-false-positive height ({picks.floor_sigma:.2f}σ)"
        )
    sigma = picks.sigma
    if sigma is not None and sigma.size:
        parts.append(f"best {float(sigma.max()):.2f}σ")
    if report.artifacts.picks is not None:
        backend = report.artifacts.picks.parameters.get("backend")
        if isinstance(backend, str):
            parts.append(backend)
    return " · ".join(parts)


def _map_panel(axes: Axes, report: SeriesReport) -> None:
    picks = report.picks
    assert picks is not None
    info = report.tomogram
    if info is None or picks.count == 0:
        style.image_axes(axes, "Where the picks are, seen from above")
        axes.text(0.5, 0.5, "no tomogram" if info is None else "no picks",
                  transform=axes.transAxes, ha="center", color=style.INK_SECONDARY)
        return
    volume = open_volume(info.artifact.path)
    low, high = np.quantile(picks.z_px, [0.05, 0.95])
    image = smooth(mean_over_depths(volume, int(math.floor(low)), int(math.ceil(high)) + 1))
    factor = reduction_factor(image.shape)
    reduced = block_mean(image, factor)
    show(axes, reduced, factor, display_range([reduced]))
    axes.scatter(
        picks.x_px, picks.y_px, s=9, facecolors="none", edgecolors=style.PICK_COLOUR,
        linewidths=0.5,
    )
    style.image_axes(axes, "Where the picks are, seen from above")
    add_scale_bar(axes, info, info.shape[2], info.shape[1])


def _height_panel(axes: Axes, report: SeriesReport) -> None:
    picks = report.picks
    assert picks is not None
    axes.set_title("How high the picks sit", loc="left")
    info = report.tomogram
    if picks.count == 0:
        axes.text(0.5, 0.5, "no picks", transform=axes.transAxes, ha="center")
        return
    voxel_nm = info.voxel_spacing_angstrom / 10.0 if info is not None else None
    if info is None or voxel_nm is None:
        heights = picks.z_px
        top = float(heights.max()) + 1.0
        axes.set_xlabel("Height (voxels)")
    else:
        heights = picks.z_px * voxel_nm
        top = info.depth_nm
        axes.set_xlabel("Height above the bottom of the volume (nm)")
    edges = np.linspace(0.0, top, 31)
    axes.hist(heights, bins=edges.tolist(), color=style.SERIES_BLUE, rwidth=0.85)
    if voxel_nm is not None and report.template is not None:
        margin = report.template.box_px / 2.0 * voxel_nm
        if 2 * margin < top:
            for edge in (margin, top - margin):
                axes.axvline(edge, color=style.INK_MUTED, linewidth=0.7)
            even = picks.count * (edges[1] - edges[0]) / (top - 2 * margin)
            axes.hlines(even, margin, top - margin, color=style.INK_SECONDARY, linewidth=0.9)
            axes.annotate(
                "if spread evenly", (margin, even), xytext=(2, 3),
                textcoords="offset points", ha="left", fontsize=6.5,
                color=style.INK_SECONDARY,
            )
    axes.set_xlim(0.0, top)
    axes.set_ylabel("Picks")
    axes.grid(True, axis="y")


def _template_panel(figure: Figure, cell: SubplotSpec, report: SeriesReport) -> None:
    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=cell, height_ratios=[1.0, 0.8],
                                    hspace=0.15, wspace=0.08)
    template = report.template
    if template is None:
        axes = figure.add_subplot(inner[:, :])
        axes.axis("off")
        axes.set_title("Template", loc="left")
        axes.text(0.0, 0.8, "The template these picks were matched with\nis not on disk.",
                  va="top", color=style.INK_SECONDARY)
        return
    volume = template.template
    mask = template.mask
    center = [size // 2 for size in volume.shape]
    sections = [
        (volume[center[0]], mask[center[0]] if mask is not None else None, "XY"),
        (volume[:, center[1], :], mask[:, center[1], :] if mask is not None else None, "XZ"),
        (volume[:, :, center[2]], mask[:, :, center[2]] if mask is not None else None, "YZ"),
    ]
    low, high = np.percentile(volume, [1, 99])
    for column, (section, mask_section, name) in enumerate(sections):
        axes = figure.add_subplot(inner[0, column])
        axes.imshow(section, cmap="gray", vmin=low, vmax=high, origin="lower")
        if mask_section is not None and mask_section.max() > 0:
            axes.contour(mask_section, levels=[0.5 * float(mask_section.max())],
                         colors=style.MASK_COLOUR, linewidths=0.7)
        style.image_axes(axes, ("Template · " if column == 0 else "") + name)
    text = figure.add_subplot(inner[1, :])
    text.axis("off")
    lines = []
    if template.source:
        lines.append(f"From {template.source}")
    box = f"Box {template.box_px} voxels"
    if template.voxel_spacing_angstrom is not None:
        box += f" at {template.voxel_spacing_angstrom:.2f} Å"
    lines.append(box)
    if template.low_pass_angstrom is not None:
        lines.append(f"Low-passed to {template.low_pass_angstrom:g} Å")
    if template.mask_radius_px is not None:
        lines.append(f"Mask radius {template.mask_radius_px:g} voxels (blue)")
    if template.inverted is not None:
        lines.append("Contrast inverted to match the tomogram" if template.inverted
                     else "Contrast as in the source map")
    text.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=7, linespacing=1.45,
              color=style.INK_SECONDARY)


def _rank_panel(axes: Axes, picks: PickTable, bands: list[Band]) -> None:
    axes.set_title("Scores from best to worst", loc="left")
    if not picks.has_scores:
        axes.text(0.5, 0.5, "these picks carry no scores", transform=axes.transAxes,
                  ha="center", color=style.INK_SECONDARY)
        return
    sigma = picks.sigma
    values = sigma if sigma is not None else picks.score
    ordered = values[picks.ranked()]
    ranks = np.arange(1, ordered.size + 1)
    axes.plot(ranks, ordered, color=style.SERIES_BLUE, linewidth=1.3)
    for band in bands:
        shown = band.positions
        axes.scatter(ranks[shown], ordered[shown], s=14, color=style.SERIES_BLUE,
                     edgecolors="white", linewidths=0.6, zorder=3)
        if band.first_rank > 1:
            axes.axvline(band.first_rank - 0.5, color=style.INK_MUTED, linewidth=0.6)
        axes.text((band.first_rank + band.last_rank) / 2.0, 0.98, band.name,
                  transform=axes.get_xaxis_transform(), ha="center", va="top",
                  fontsize=6.5, color=style.INK_SECONDARY)
    if sigma is not None and picks.floor_sigma is not None:
        axes.axhline(picks.floor_sigma, color=style.INK_SECONDARY, linewidth=0.8)
        axes.annotate("one-false-positive height", (1.0, picks.floor_sigma),
                      xycoords=("axes fraction", "data"), xytext=(0, 2),
                      textcoords="offset points", ha="right", va="bottom", fontsize=6.5,
                      color=style.INK_SECONDARY)
    axes.set_xlim(0.5, ordered.size + 0.5)
    axes.set_xlabel("Rank")
    axes.set_ylabel("Score (σ)" if sigma is not None else "Score (pytom LCCmax)")
    axes.grid(True, axis="y")


def _noise_panel(axes: Axes, picks: PickTable) -> None:
    axes.set_title("Picks above each height, against noise alone", loc="left")
    sigma = picks.sigma
    if sigma is None or picks.search_space is None or sigma.size == 0:
        axes.text(0.5, 0.5, "pytom recorded no noise scale\nfor this search",
                  transform=axes.transAxes, ha="center", va="center",
                  color=style.INK_SECONDARY)
        axes.set_xticks([])
        axes.set_yticks([])
        return
    heights = np.linspace(float(sigma.min()), float(sigma.max()), 200)
    observed = np.asarray([np.count_nonzero(sigma >= height) for height in heights])
    expected = picks.expected_from_noise(heights)
    assert expected is not None
    axes.step(heights, observed, where="post", color=style.SERIES_BLUE, linewidth=1.3,
              label="picks")
    axes.plot(heights, expected, color=style.INK_SECONDARY, linewidth=1.0,
              label="noise alone (expected)")
    if picks.floor_sigma is not None:
        axes.axvline(picks.floor_sigma, color=style.INK_MUTED, linewidth=0.6)
    axes.set_yscale("log")
    axes.set_ylim(bottom=0.5, top=max(float(observed.max()), float(expected.max())) * 2.0)
    axes.set_xlabel("Height (σ)")
    axes.set_ylabel("Peaks at or above it")
    axes.legend(loc="upper right")
    axes.grid(True, axis="y")


def _numbers_panel(axes: Axes, report: SeriesReport) -> None:
    picks = report.picks
    assert picks is not None
    axes.axis("off")
    axes.set_title("In numbers", loc="left")
    lines = [f"Picks: {picks.count}"]
    artifact = report.artifacts.picks
    membrane_filter = artifact.parameters.get("membrane_filter") if artifact else None
    if isinstance(membrane_filter, dict) and "picks" in membrane_filter:
        lines.append(f"  kept {membrane_filter.get('kept')} of {membrane_filter['picks']} "
                     "by distance to a membrane")
    if picks.floor_sigma is not None:
        lines.append(f"One-false-positive height: {picks.floor_sigma:.2f}σ")
        if picks.above_floor is not None:
            lines.append(f"  picks above it: {picks.above_floor}")
    sigma = picks.sigma
    if sigma is not None and sigma.size:
        lines.append(f"Best pick: {float(sigma.max()):.2f}σ")
        lines.append(f"Lowest pick: {float(sigma.min()):.2f}σ")
    if picks.excess_at_cut_off is not None:
        lines.append(f"Picks beyond what noise gives: {picks.excess_at_cut_off:+.0f}")
    if picks.search_space is not None:
        lines.append(f"Positions × orientations searched: {picks.search_space:.2e}")
    info = report.tomogram
    if info is not None and picks.z_sd_px is not None:
        voxel_nm = info.voxel_spacing_angstrom / 10.0
        spread = f"Height spread: {picks.z_sd_px * voxel_nm:.0f} nm"
        if picks.uniform_z_sd_px is not None:
            spread += f" (if even: {picks.uniform_z_sd_px * voxel_nm:.0f} nm)"
        lines.append(spread)
    lines.append(
        "Correlation map: " + (report.correlation.source if report.correlation else "not saved")
    )
    record = report.qc.get("particle_picking")
    if record is not None:
        lines.append("")
        lines.append(f"QC: {style.STATUS_LABELS.get(record.status, record.status)}")
        for warning in record.warnings:
            lines.extend(textwrap.wrap(warning, 58))
    axes.text(0.0, 0.98, "\n".join(lines), transform=axes.transAxes, va="top", fontsize=7,
              linespacing=1.45)


def correlation_page(report: SeriesReport, depths: list[Depth]) -> Figure:
    picks = report.picks
    correlation = report.correlation
    info = report.tomogram
    assert picks is not None and correlation is not None and info is not None
    preview = correlation.preview
    if preview.depth != info.shape[0] or preview.slab_max.shape[1:] != info.shape[1:]:
        raise ValueError(
            f"the correlation map covers {preview.depth} × {preview.slab_max.shape[1:]} voxels "
            f"but the tomogram is {info.shape}, so it is not a map of this tomogram"
        )
    volume = open_volume(info.artifact.path)
    half = slab_half_width_px(info.voxel_spacing_angstrom)
    tops = [picture(volume, 0, depth.center, half) for depth in depths]
    limits = display_range([image for image, _ in tops])
    scale = picks.search_sigma if picks.search_sigma is not None else 1.0
    unit = "σ" if picks.search_sigma is not None else "LCCmax"
    maps = []
    for depth in depths:
        slab = preview.slab_max[preview.slab_containing(depth.center)] / scale
        factor = reduction_factor(slab.shape)
        maps.append((block_max(slab, factor), factor))
    values = np.concatenate([image.ravel() for image, _ in maps])
    upper = float(values.max())
    if picks.floor_sigma is not None and picks.search_sigma is not None:
        upper = max(upper, picks.floor_sigma + 0.5)
    map_limits = (float(np.median(values)), upper)

    thickness_nm = (depths[0].stop - depths[0].start) * info.voxel_spacing_angstrom / 10.0
    figure = style.new_page(
        f"{report.series} · Correlation map",
        f"Best score in each {thickness_nm:.0f} nm slab, in {unit} · map {correlation.source} · "
        "picks whose centre lies in the slab are circled",
    )
    grid = GridSpec(
        2, 4, figure=figure, left=0.035, right=0.965, top=0.85, bottom=0.16,
        width_ratios=[1.0, 1.0, 1.0, 0.04], hspace=0.16, wspace=0.06,
    )
    radius = report.particle_radius_px
    mappable = None
    for column, (letter, depth, (image, factor), (slab, map_factor)) in enumerate(
        zip(DEPTH_LETTERS, depths, tops, maps, strict=False)
    ):
        inside = (np.rint(picks.z_px) >= depth.start) & (np.rint(picks.z_px) < depth.stop)
        top_nm = depth.start * info.voxel_spacing_angstrom / 10.0
        bottom_nm = depth.stop * info.voxel_spacing_angstrom / 10.0
        axes = figure.add_subplot(grid[0, column])
        show(axes, image, factor, limits)
        _circles(axes, picks.x_px[inside], picks.y_px[inside], radius)
        style.image_axes(axes, f"{letter} · {top_nm:.0f}–{bottom_nm:.0f} nm up · "
                               f"{int(inside.sum())} picks")
        if column == 0:
            add_scale_bar(axes, info, info.shape[2], info.shape[1])
        axes = figure.add_subplot(grid[1, column])
        show(axes, slab, map_factor, map_limits, cmap=style.MAGNITUDE_MAP)
        mappable = axes.images[0]
        _circles(axes, picks.x_px[inside], picks.y_px[inside], radius)
        style.image_axes(axes)
    assert mappable is not None
    colorbar = figure.colorbar(mappable, cax=figure.add_subplot(grid[1, 3]))
    colorbar.set_label(f"Best score in the slab ({unit})", fontsize=7)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=6)
    if picks.floor_sigma is not None and picks.search_sigma is not None:
        colorbar.ax.axhline(picks.floor_sigma, color=style.INK, linewidth=0.8)

    note = (
        "Top: the tomogram at the middle of each slab (protein dark, 10 nm average). "
        "Bottom: the correlation map over the whole slab; bright is where the template fits. "
        "A circle without a bright spot should not happen; a bright spot without a circle is a "
        "peak the picking left out: below the cut-off, outside the search mask, or too close to "
        "a better pick."
    )
    if picks.floor_sigma is not None and picks.search_sigma is not None:
        note += " The line on the colour bar is the one-false-positive height."
    if correlation.agreement is not None:
        note += (
            f" Check: at {correlation.agreement:.0%} of picks the map holds the pick's own "
            "score, as it should at all of them; much less means the map and the coordinates "
            "disagree, and the circles are misplaced."
        )
    style.caption(figure, note)
    return figure


def _circles(axes: Axes, x: np.ndarray, y: np.ndarray, radius: float) -> None:
    if x.size == 0:
        return
    patches = [Circle((float(cx), float(cy)), radius) for cx, cy in zip(x, y, strict=True)]
    axes.add_collection(
        PatchCollection(patches, facecolor="none", edgecolor=style.PICK_COLOUR, linewidth=0.7)
    )


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


def gallery_half_width(report: SeriesReport) -> int:
    """Half the tile, in voxels: the particle and about one radius of its surroundings."""

    radius = report.particle_radius_px
    half = math.ceil(2.0 * radius)
    if report.template is not None:
        half = min(half, report.template.box_px // 2)
    return max(4, half)


def build_gallery(report: SeriesReport, bands: list[Band]) -> Gallery:
    picks = report.picks
    info = report.tomogram
    assert picks is not None and info is not None
    volume = open_volume(info.artifact.path)
    half = gallery_half_width(report)
    ranked = picks.ranked()
    sigma = picks.sigma

    def tile(center: tuple[int, int, int], title: str) -> Tile:
        views = [smooth(view) for view in central_views(crop_centered(volume, center, half))]
        gap = np.full((views[0].shape[0], max(1, (2 * half + 1) // 10)), np.nan, np.float32)
        return Tile(np.concatenate([views[0], gap, views[1], gap, views[2]], axis=1), title)

    def center_of(index: int) -> tuple[int, int, int]:
        return (int(round(picks.z_px[index])), int(round(picks.y_px[index])),
                int(round(picks.x_px[index])))

    gallery: dict[str, GalleryBand] = {}
    for band in bands:
        tiles = []
        for position in band.positions:
            index = int(ranked[position])
            label = (f"{sigma[index]:.1f}σ" if sigma is not None
                     else f"{picks.score[index]:.3f}")
            tiles.append(tile(center_of(index), f"rank {position + 1} · {label}"))
        header = (
            f"{BAND_TITLES[band.name]} · ranks {band.first_rank}–{band.last_rank} "
            f"of {picks.count}"
        )
        if sigma is not None:
            first = sigma[ranked[band.first_rank - 1]]
            last = sigma[ranked[band.last_rank - 1]]
            header += f" · {first:.1f} to {last:.1f}σ"
        gallery[band.name] = GalleryBand(band.name, header, tiles)

    positions = random_positions(info.shape, half, picks.z_px, TILES_PER_BAND,
                                 stable_seed(report.series))
    gallery["random"] = GalleryBand(
        "random",
        "Random positions · anywhere in x and y, at depths drawn from the picks'",
        [tile((int(z), int(y), int(x)), f"random {number}")
         for number, (z, y, x) in enumerate(positions, 1)],
    )
    limits = display_range([one.image for band in gallery.values() for one in band.tiles])
    return Gallery(gallery, limits)


def gallery_page(report: SeriesReport, gallery: Gallery, keys: tuple[str, str]) -> Figure:
    shown = [gallery.bands[key] for key in keys if key in gallery.bands]
    names = " and ".join(band.key for band in shown)
    figure = style.new_page(f"{report.series} · Picks by score: {names}",
                            "Each tile: XY | XZ | YZ through the pick's centre")
    cmap = style.GRAY_WITH_GAPS
    top, bottom = 0.885, 0.13
    block = (top - bottom) / 2.0
    columns = 4
    left, right, gap = 0.035, 0.965, 0.012
    width = (right - left - gap * (columns - 1)) / columns
    row_height = (block - 0.05) / 2.0
    for block_index, band in enumerate(shown):
        block_top = top - block_index * block
        figure.text(left, block_top - 0.005, band.header, fontsize=9, fontweight="bold",
                    va="top")
        for number, one in enumerate(band.tiles):
            row, column = divmod(number, columns)
            axes = figure.add_axes((
                left + column * (width + gap),
                block_top - 0.04 - (row + 1) * row_height,
                width,
                row_height - 0.025,
            ))
            axes.imshow(one.image, cmap=cmap, vmin=gallery.limits[0], vmax=gallery.limits[1],
                        origin="lower")
            style.image_axes(axes, one.title)
    style.caption(
        figure,
        f"{GALLERY_SLAB_PX}-voxel central slabs through each pick, protein dark, smoothed like "
        "the tomogram pages, all tiles of this tilt-series on one grey scale. Ranks are spread "
        "evenly through each third of the picks. Compare the low third with the random "
        "positions: if they look alike, the low-scoring picks are probably not particles.",
    )
    return figure

