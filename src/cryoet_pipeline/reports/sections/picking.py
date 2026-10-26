"""The picks: where they are, how they score against noise, the map, and what they look like."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import PickTable, SeriesReport, TomogramInfo
from cryoet_pipeline.reports.imaging import (
    Band,
    Depth,
    block_max,
    display_range,
    rank_bands,
    reduction_factor,
    slab_half_width_px,
    sphere_sections,
)
from cryoet_pipeline.reports.sections.gallery import TILES_PER_BAND
from cryoet_pipeline.reports.sections.tomogram import (
    DEPTH_LETTERS,
    Z_NOTE,
    add_scale_bar,
    depth_title,
    mark_depths,
    open_volume,
    picture,
    show,
    slab_caption,
    z_nm,
)


def picks_page(report: SeriesReport, depths: list[Depth]) -> Figure:
    """The picks on the tomogram: three XY slices and one XZ slice, as on the tomogram page."""

    picks = report.picks
    info = report.tomogram
    assert picks is not None and info is not None
    radius, radius_source = report.particle_radius()
    return locations_page(
        f"{report.series} · Picks", _subtitle(report), info, picks.x_px, picks.y_px, picks.z_px,
        radius=radius, radius_source=radius_source, depths=depths,
        box_px=report.template.box_px if report.template is not None else None,
    )


def locations_page(
    title: str,
    subtitle: str,
    info: TomogramInfo,
    x_px: np.ndarray,
    y_px: np.ndarray,
    z_px: np.ndarray,
    *,
    radius: float,
    radius_source: str,
    depths: list[Depth],
    box_px: float | None,
    item: str = "pick",
) -> Figure:
    """Points on their tomogram: three XY slices, one XZ slice and where they sit in z.

    The picks page draws a series' picks with it; the classification report draws
    each class's particles, so the two read alike. `box_px` (tomogram voxels) is the
    box the points were cut in, when known: the z histogram's even level leaves out
    the half box at either face.
    """

    volume = open_volume(info.artifact.path)
    half = slab_half_width_px(info.voxel_spacing_angstrom)
    depth_px, height_px, width_px = info.shape
    items = f"{item}s"

    tops = [picture(volume, 0, depth.center, half) for depth in depths]
    side = picture(volume, 1, height_px // 2, half)
    limits = display_range([image for image, _ in (*tops, side)])

    figure = style.new_page(title, subtitle)
    grid = GridSpec(
        2, 3, figure=figure, left=0.035, right=0.975, top=style.CONTENT_TOP,
        bottom=style.CONTENT_BOTTOM - 0.02, height_ratios=[1.25, 1.0], hspace=0.22,
        wspace=0.08,
    )
    # The histogram's axis labels need the room the pictures above do not.
    lower = GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[1, :], wspace=0.3,
                                    width_ratios=[1.0, 1.0, 0.85])
    for column, (letter, depth, (image, factor)) in enumerate(
        zip(DEPTH_LETTERS, depths, tops, strict=False)
    ):
        axes = figure.add_subplot(grid[0, column])
        show(axes, image, factor, limits)
        sections = sphere_sections(z_px, depth.center, half, radius)
        _circles(axes, x_px, y_px, sections)
        cut = int(np.count_nonzero(np.isfinite(sections)))
        style.image_axes(axes, f"{depth_title(letter, depth, info)} · {cut} {items}")
        if column == 0:
            add_scale_bar(axes, info.voxel_spacing_angstrom, width_px, height_px)

    axes = figure.add_subplot(lower[0, 0:2])
    image, factor = side
    show(axes, image, factor, limits, aspect="equal")
    sections = sphere_sections(y_px, height_px // 2, half, radius)
    _circles(axes, x_px, z_px, sections)
    mark_depths(axes, depths, width_px)
    cut = int(np.count_nonzero(np.isfinite(sections)))
    style.image_axes(axes, f"XZ slice through the middle in y: x across, z up · {cut} {items}")
    add_scale_bar(axes, info.voxel_spacing_angstrom, width_px, depth_px)

    _height_panel(figure.add_subplot(lower[0, 2]), z_px, info, box_px, items)

    style.caption(
        figure,
        f"Circles: a sphere with {radius_source} ({radius:.1f} voxels, "
        f"{radius * info.voxel_spacing_angstrom / 10:.1f} nm) around every {item}, cut by the "
        "slab as 3dmod draws spheres: full size where the slab passes through "
        f"a {item}'s centre, smaller for {items} just above or below it. " + slab_caption(info),
    )
    return figure


def scores_page(report: SeriesReport, bands: list[Band]) -> Figure:
    picks = report.picks
    assert picks is not None
    figure = style.new_page(f"{report.series} · Pick scores", _subtitle(report))
    grid = GridSpec(
        2, 3, figure=figure, left=0.065, right=0.975, top=style.CONTENT_TOP,
        bottom=style.CONTENT_BOTTOM, hspace=0.5, wspace=0.32,
    )
    ranks = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0, 0:2], wspace=0.06)
    log_axes = figure.add_subplot(ranks[0, 0])
    _rank_panels(log_axes, figure.add_subplot(ranks[0, 1], sharey=log_axes), picks, bands)
    _noise_panel(figure.add_subplot(grid[1, 0:2]), picks)
    _template_panel(figure, grid[0, 2], report)
    _numbers_panel(figure.add_subplot(grid[1, 2]), report)
    style.caption(
        figure,
        "σ: a score divided by the standard deviation of the search's own noise. The "
        "one-false-positive height is the score noise alone reaches about once in this whole "
        "search: a pick above it is unlikely to be noise on its own; below it, picks count only "
        "as a group, by how far the blue line runs above the grey one. Top: rank on a log axis "
        "(left), giving the best picks room, and on a linear one (right), each third of the "
        "picks a third of the width. Dots: the picks the galleries show.",
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


def _height_panel(axes: Axes, z_px: np.ndarray, info: TomogramInfo, box_px: float | None,
                  items: str) -> None:
    axes.set_title(f"Where the {items} sit in z", loc="left")
    count = int(z_px.size)
    if count == 0:
        axes.text(0.5, 0.5, f"no {items}", transform=axes.transAxes, ha="center")
        return
    voxel_nm = info.voxel_spacing_angstrom / 10.0
    heights = z_px * voxel_nm
    top = info.depth_nm
    edges = np.linspace(0.0, top, 31)
    axes.hist(heights, bins=edges.tolist(), color=style.SERIES_BLUE, rwidth=0.85)
    if box_px is not None:
        margin = box_px / 2.0 * voxel_nm
        if 2 * margin < top:
            for edge in (margin, top - margin):
                axes.axvline(edge, color=style.INK_MUTED, linewidth=0.7)
            even = count * (edges[1] - edges[0]) / (top - 2 * margin)
            axes.hlines(even, margin, top - margin, color=style.INK_SECONDARY, linewidth=0.9)
            axes.annotate(
                "if spread evenly", (margin, even), xytext=(2, 3),
                textcoords="offset points", ha="left", fontsize=style.FONT_TINY,
                color=style.INK_SECONDARY,
            )
    axes.set_xlim(0.0, top)
    axes.set_xlabel("z (nm above the bottom of the tomogram)")
    axes.set_ylabel(items.capitalize())
    axes.grid(True, axis="y")


def _template_panel(figure: Figure, cell: SubplotSpec, report: SeriesReport) -> None:
    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=cell, height_ratios=[1.0, 1.0],
                                    hspace=0.1, wspace=0.08)
    template = report.template
    if template is None:
        axes = figure.add_subplot(inner[:, :])
        axes.axis("off")
        axes.set_title("Template", loc="left")
        axes.text(0.0, 0.8, "The template these picks were\nmatched with is not on disk.",
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
                         colors=style.MASK_COLOUR, linewidths=0.8)
        style.image_axes(axes, name)
    text = figure.add_subplot(inner[1, :])
    text.axis("off")
    lines = [f"Template from {template.source}" if template.source else "Template"]
    box = f"Box {template.box_px} voxels"
    if template.voxel_spacing_angstrom is not None:
        box += f" at {template.voxel_spacing_angstrom:.2f} Å"
    lines.append(box)
    if template.low_pass_angstrom is not None:
        lines.append(f"Low-passed to {template.low_pass_angstrom:g} Å")
    if template.mask_radius_px is not None:
        lines.append(f"Mask radius {template.mask_radius_px:g} voxels (blue)")
    if template.inverted is not None:
        lines.append("Contrast inverted to match" if template.inverted
                     else "Contrast as in the source map")
    text.text(0.0, 0.95, "\n".join(lines), va="top", fontsize=style.FONT_SMALL,
              linespacing=1.45, color=style.INK_SECONDARY)


def _rank_panels(log_axes: Axes, linear_axes: Axes, picks: PickTable, bands: list[Band]) -> None:
    log_axes.set_title("Scores from best to worst", loc="left")
    if not picks.has_scores:
        log_axes.text(0.5, 0.5, "these picks carry no scores", transform=log_axes.transAxes,
                      ha="center", color=style.INK_SECONDARY)
        linear_axes.axis("off")
        return
    rank_curve(log_axes, picks, bands, log_scale=True, labels=True)
    rank_curve(linear_axes, picks, bands, log_scale=False, labels=True)
    log_axes.set_xlabel("Rank (log scale)")
    linear_axes.set_xlabel("Rank (linear scale)")
    log_axes.set_ylabel(_score_label(picks))
    linear_axes.tick_params(labelleft=False)


def rank_curve(
    axes: Axes, picks: PickTable, bands: list[Band], *, log_scale: bool, labels: bool
) -> None:
    """Scores from best to worst, with the gallery's thirds marked, on one rank axis.

    Every page that shows it draws it twice, on a log and on a linear rank axis,
    and always through this function, so that the pages cannot disagree about
    what a band is. The log axis gives room to the start, where a curve that bends
    after its first few dozen picks does its bending; the linear one gives each
    third of the picks a third of the width, as the galleries do. `labels` names
    the bands and the one-false-positive line, on the linear axis only: there
    every third is wide enough to carry its name, and on the log axis the last
    two are squeezed into its right edge.
    """

    sigma = picks.sigma
    values = sigma if sigma is not None else picks.score
    ordered = values[picks.ranked()]
    ranks = np.arange(1, ordered.size + 1)
    named = labels and not log_scale
    axes.plot(ranks, ordered, color=style.SERIES_BLUE, linewidth=1.4)
    for band in bands:
        shown = band.positions
        axes.scatter(ranks[shown], ordered[shown], s=18, color=style.SERIES_BLUE,
                     edgecolors="white", linewidths=0.6, zorder=3)
        if band.first_rank > 1:
            axes.axvline(band.first_rank - 0.5, color=style.INK_MUTED, linewidth=0.6)
        if named:
            axes.text((band.first_rank + band.last_rank) / 2.0, 0.98, band.name,
                      transform=axes.get_xaxis_transform(), ha="center", va="top",
                      fontsize=style.FONT_TINY, color=style.INK_SECONDARY)
    if sigma is not None and picks.floor_sigma is not None:
        axes.axhline(picks.floor_sigma, color=style.INK_SECONDARY, linewidth=0.8)
        if named:
            axes.annotate("one-false-positive height", (1.0, picks.floor_sigma),
                          xycoords=("axes fraction", "data"), xytext=(-4, 3),
                          textcoords="offset points", ha="right", va="bottom",
                          fontsize=style.FONT_TINY, color=style.INK_SECONDARY,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0,
                                "alpha": 0.85})
    if log_scale:
        axes.set_xscale("log")
        axes.set_xlim(0.8, ordered.size * 1.25)
        axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    else:
        axes.set_xlim(0.5, ordered.size + 0.5)
    axes.grid(True, axis="y")


def _score_label(picks: PickTable) -> str:
    return "Score (σ)" if picks.sigma is not None else "Score (pytom LCCmax)"


OVERVIEW_ROWS = 3
"""Tomograms on one overview page: one per row, its log and linear curves side by side."""


def score_overview_pages(
    reports: Sequence[SeriesReport], contents: Mapping[str, int]
) -> list[Callable[[], Figure]]:
    """Every tomogram's score curves, one tomogram to a row, three to a page.

    One score axis for all of them when every search recorded its noise
    scale: raw pytom scores are not comparable between tomograms, scores in
    sigma are, and comparing tomograms is what this page is for. `contents` is
    read when a page is drawn, after every series has its page number.
    """

    scored = [report for report in reports if report.picks is not None and report.picks.has_scores]
    limits = _shared_sigma_limits(scored)
    return [
        partial(_score_overview_page, scored[start : start + OVERVIEW_ROWS], contents, limits)
        for start in range(0, len(scored), OVERVIEW_ROWS)
    ]


def _shared_sigma_limits(reports: Sequence[SeriesReport]) -> tuple[float, float] | None:
    """One y range for every panel, or None when some search did not record sigma."""

    values: list[np.ndarray] = []
    floors: list[float] = []
    for report in reports:
        assert report.picks is not None
        sigma = report.picks.sigma
        if sigma is None:
            return None
        values.append(sigma)
        if report.picks.floor_sigma is not None:
            floors.append(report.picks.floor_sigma)
    if not values:
        return None
    low = min(float(np.min(sigma)) for sigma in values)
    high = max(float(np.max(sigma)) for sigma in values)
    if floors:
        low = min(low, *floors)
        high = max(high, *floors)
    margin = 0.05 * (high - low) if high > low else 1.0
    return low - margin, high + margin


def _score_overview_page(
    reports: list[SeriesReport],
    contents: Mapping[str, int],
    limits: tuple[float, float] | None,
) -> Figure:
    figure = style.new_page(
        "Scores by tomogram",
        "Each tomogram's picks from best to worst, "
        + ("all on one score axis, in σ" if limits is not None else "each on its own score axis"),
    )
    grid = GridSpec(
        OVERVIEW_ROWS, 2, figure=figure, left=0.065, right=0.975, top=style.CONTENT_TOP,
        bottom=style.CONTENT_BOTTOM, hspace=0.75, wspace=0.06,
    )
    for row, report in enumerate(reports):
        picks = report.picks
        assert picks is not None
        bands = rank_bands(picks.count, TILES_PER_BAND)
        log_axes = figure.add_subplot(grid[row, 0])
        linear_axes = figure.add_subplot(grid[row, 1], sharey=log_axes)
        rank_curve(log_axes, picks, bands, log_scale=True, labels=False)
        rank_curve(linear_axes, picks, bands, log_scale=False, labels=row == 0)
        if limits is not None:
            log_axes.set_ylim(*limits)
        title = f"{report.series} · {picks.count:,} picks"
        if picks.above_floor is not None:
            title += f", {picks.above_floor:,} above the one-false-positive height"
        if report.series in contents:
            title += f" · p. {contents[report.series]}"
        log_axes.set_title(title, loc="left", fontsize=style.FONT_TEXT)
        log_axes.set_ylabel(_score_label(picks))
        linear_axes.tick_params(labelleft=False)
        log_axes.set_xlabel("Rank (log scale)")
        linear_axes.set_xlabel("Rank (linear scale)")
    style.caption(
        figure,
        "Left: rank on a log axis, which gives the best picks room; right: the same picks on a "
        "linear axis, where each third takes a third of the width. A steep start that flattens "
        "into a long tail is a group of high-scoring peaks above many similar weak ones. The "
        "grey line is the one-false-positive height. Thin vertical lines split the picks into "
        "thirds, as the galleries do; the dots are the picks the galleries show. p. is where the "
        "tomogram's own pages start.",
    )
    return figure


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
    axes.step(heights, observed, where="post", color=style.SERIES_BLUE, linewidth=1.4,
              label="picks")
    axes.plot(heights, expected, color=style.INK_SECONDARY, linewidth=1.1,
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
        lines.append(f"  kept {membrane_filter.get('kept')} of {membrane_filter['picks']}")
        lines.append("  by distance to a membrane")
    if picks.floor_sigma is not None:
        lines.append(f"One-false-positive height: {picks.floor_sigma:.2f}σ")
        if picks.above_floor is not None:
            lines.append(f"  picks above it: {picks.above_floor}")
    sigma = picks.sigma
    if sigma is not None and sigma.size:
        lines.append(f"Best pick: {float(sigma.max()):.2f}σ")
        lines.append(f"Lowest pick: {float(sigma.min()):.2f}σ")
    if picks.excess_at_cut_off is not None:
        lines.append(f"Beyond what noise gives: {picks.excess_at_cut_off:+.0f}")
    if picks.search_space is not None:
        lines.append(f"Positions × orientations: {picks.search_space:.2e}")
    info = report.tomogram
    if info is not None and picks.z_sd_px is not None:
        spread = f"Spread in z: {z_nm(picks.z_sd_px, info):.0f} nm"
        if picks.uniform_z_sd_px is not None:
            spread += f" (even: {z_nm(picks.uniform_z_sd_px, info):.0f} nm)"
        lines.append(spread)
    lines.append(
        "Correlation map: " + ("saved" if report.correlation else "not saved")
    )
    record = report.qc.get("particle_picking")
    if record is not None:
        lines.append("")
        lines.append(f"QC: {style.STATUS_LABELS.get(record.status, record.status)}")
        for warning in record.warnings:
            lines.extend(textwrap.wrap(warning, 40))
    axes.text(0.0, 0.98, "\n".join(lines), transform=axes.transAxes, va="top",
              fontsize=style.FONT_SMALL, linespacing=1.45)


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

    thickness_nm = z_nm(depths[0].stop - depths[0].start, info)
    figure = style.new_page(
        f"{report.series} · Correlation map",
        f"Best score in each {thickness_nm:.0f} nm slab of z, in {unit} · picks whose centre "
        "lies in the slab are circled",
    )
    grid = GridSpec(
        2, 4, figure=figure, left=0.035, right=0.955, top=style.CONTENT_TOP,
        bottom=style.PICTURE_BOTTOM + 0.02, width_ratios=[1.0, 1.0, 1.0, 0.04], hspace=0.18,
        wspace=0.06,
    )
    radius = report.particle_radius_px
    mappable = None
    for column, (letter, depth, (image, factor), (slab, map_factor)) in enumerate(
        zip(DEPTH_LETTERS, depths, tops, maps, strict=False)
    ):
        inside = (np.rint(picks.z_px) >= depth.start) & (np.rint(picks.z_px) < depth.stop)
        radii = np.where(inside, radius, np.nan)
        axes = figure.add_subplot(grid[0, column])
        show(axes, image, factor, limits)
        _circles(axes, picks.x_px, picks.y_px, radii)
        style.image_axes(
            axes,
            f"{letter} · z = {z_nm(depth.start, info):.0f}–{z_nm(depth.stop, info):.0f} nm · "
            f"{int(inside.sum())} picks",
        )
        if column == 0:
            add_scale_bar(axes, info.voxel_spacing_angstrom, info.shape[2], info.shape[1])
        axes = figure.add_subplot(grid[1, column])
        show(axes, slab, map_factor, map_limits, cmap=style.MAGNITUDE_MAP)
        mappable = axes.images[0]
        _circles(axes, picks.x_px, picks.y_px, radii)
        style.image_axes(axes)
    assert mappable is not None
    colorbar = figure.colorbar(mappable, cax=figure.add_subplot(grid[1, 3]))
    style.colour_bar_style(colorbar, f"Best score in the slab ({unit})")
    if picks.floor_sigma is not None and picks.search_sigma is not None:
        colorbar.ax.axhline(picks.floor_sigma, color=style.INK, linewidth=0.8)

    note = (
        "Top: the tomogram at the middle of each slab (protein dark). Bottom: the correlation "
        "map over the whole slab; bright is where the template fits. A circle without a bright "
        "spot should not happen; a bright spot without one is a peak picking left out (below "
        "the cut-off, outside the mask, or next to a better pick)."
    )
    if picks.floor_sigma is not None and picks.search_sigma is not None:
        note += " The line on the colour bar is the one-false-positive height."
    if correlation.agreement is not None:
        note += (
            f" Check: at {correlation.agreement:.0%} of picks the map holds the pick's own "
            "score, as it should at all of them."
        )
    note += " " + Z_NOTE
    style.caption(figure, note)
    return figure


def _circles(axes: Axes, x: np.ndarray, y: np.ndarray, radii: np.ndarray) -> None:
    """A ring per pick, each its own size; NaN radii are left out."""

    keep = np.isfinite(radii)
    if not keep.any():
        return
    patches = [
        Circle((float(cx), float(cy)), float(radius))
        for cx, cy, radius in zip(x[keep], y[keep], radii[keep], strict=True)
    ]
    axes.add_collection(
        PatchCollection(patches, facecolor="none", edgecolor=style.PICK_COLOUR, linewidth=0.8)
    )
