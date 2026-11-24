"""The raw-tilts PDF: a few tilts per series, the dark round spots, and the tilt axis.

Drawn from what cryoet_pipeline.tilt_preview measured; nothing is read here but
those measurements. Written for the person who fills a lane config before any
lane has run, and has no 3dmod: one title page with a table, then three pages a
series (tilts, beads, axis).
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.imaging import display_range, smooth
from cryoet_pipeline.reports.sections.front import error_page, fit_width
from cryoet_pipeline.reports.sections.tomogram import add_scale_bar
from cryoet_pipeline.tilt_preview import (
    AXIS_AGREEMENT_DEG,
    GALLERY_SPOTS,
    SPOT_MAX_DARKNESS,
    PairScan,
    SeriesPreview,
    TiltPicture,
    axis_difference,
    flatten,
    overlay_pair,
    pair_factor,
    typical_diameter,
)

AXIS_COLOUR = style.SERIES_BLUE
RECORDED_COLOUR = style.SERIES_ORANGE

MIN_TRACKED_BEADS = 10
"""[fiducials] min_tracked_fiducials' default: the bead tracking QC fails below it."""

HOW_TO_READ = (
    "Pictures lie as 3dmod will show the motion-corrected stack: row 0 at the bottom. "
    "A TIFF movie is turned upside down to get there, as MotionCor3 reads a TIFF bottom row "
    "first. Frames are added up without gain or motion correction, and each row's and "
    "column's mean is evened out, which removes the stripes of a gain not applied.",
    "Gold beads are dark round discs, all of one size, far darker than anything biological. "
    f"A spot is counted when its centre is at most {SPOT_MAX_DARKNESS:g} of the mean around "
    f"it. The bead route's tracking QC wants at least {MIN_TRACKED_BEADS} tracked beads; "
    "with a handful of spots, or spots that are not round discs, the route is patches "
    "(block B of example.toml).",
    "The axis angle is IMOD's, the one [align] tilt_axis_angle takes: from the picture's "
    "y axis, counterclockwise positive. Near 0 or 180 the axis runs up and down, near 90 "
    "or -90 left to right. It is measured as a line without a sign: φ and φ + 180 make the "
    "same pictures, and the two give tomograms of opposite hand. An mdoc's axis has the "
    "sign; without one, check the hand on the first tomogram.",
)


@dataclass(frozen=True)
class Outcome:
    path: Path
    pages: int
    problems: list[str]


def write_tilt_preview(
    path: Path,
    previews: Sequence[SeriesPreview],
    *,
    label: str,
    written: str,
    software: str,
) -> Outcome:
    planned: list[tuple[str, Callable[[], Figure]]] = [
        ("title", lambda: title_page(previews, label=label, written=written, software=software)),
    ]
    for preview in previews:
        planned.append((f"{preview.series} · tilts", _bind(tilts_page, preview)))
        planned.append((f"{preview.series} · beads", _bind(beads_page, preview)))
        planned.append((f"{preview.series} · axis", _bind(axis_page, preview)))

    problems: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    metadata = {"Title": f"Raw tilts · {label}", "Creator": software}
    with matplotlib.rc_context(cast(Any, style.RC)), PdfPages(partial, metadata=metadata) as pdf:
        for number, (page_label, build) in enumerate(planned, 1):
            try:
                figure = build()
            except Exception as error:  # noqa: BLE001 - one page must not lose the others
                problems.append(f"{page_label}: could not be drawn: {type(error).__name__}: "
                                f"{error}")
                figure = error_page(page_label, error)
            style.footer(figure, f"{label} · {written}", number, len(planned))
            style.tidy_figure_text(figure)
            pdf.savefig(figure)
    partial.replace(path)
    return Outcome(path, len(planned), problems)


def _bind(build: Callable[[SeriesPreview], Figure], preview: SeriesPreview
          ) -> Callable[[], Figure]:
    return lambda: build(preview)


# --- Words shared by the table, the pages and the script's print -------------------------


def axis_text(angle_deg: float, near: float | None = None) -> str:
    """An axis as the two signed angles that give it, the one nearer `near` first."""

    first = (angle_deg + 180.0) % 360.0 - 180.0
    second = first - 180.0 if first > 0 else first + 180.0

    def distance(angle: float) -> float:
        return abs((angle - (near or 0.0) + 180.0) % 360.0 - 180.0)

    if distance(second) < distance(first):
        first, second = second, first
    return f"{first:.1f}° (or {second:.1f}°)"


def measured_axis_line(preview: SeriesPreview) -> str:
    axis = preview.axis
    if axis.angle_deg is None:
        return f"not measured: {axis.reason}"
    bests = [pair.best_deg for pair in axis.pairs]
    spread = ", ".join(f"{_signed(best, axis.angle_deg):.1f}" for best in bests)
    return (f"{axis_text(axis.angle_deg, preview.recorded.angle_deg)}, {axis.reading}; "
            f"{axis.agreeing} of {len(bests)} pairs within {AXIS_AGREEMENT_DEG:g}° "
            f"({spread})")


def recorded_axis_line(preview: SeriesPreview) -> str:
    recorded = preview.recorded
    if recorded.angle_deg is None:
        return recorded.source
    text = f"{recorded.angle_deg:.1f}° ({recorded.source})"
    if preview.axis.angle_deg is not None:
        difference = axis_difference(preview.axis.angle_deg, recorded.angle_deg)
        text += (f": the tilts agree within {abs(difference):.1f}°"
                 if abs(difference) <= AXIS_AGREEMENT_DEG
                 else f": the tilts differ by {difference:+.1f}°")
    return text


def beads_line(preview: SeriesPreview) -> str:
    counted = preview.bead_like
    size = typical_diameter(preview.spots)
    if not counted:
        return (f"no spot at most {SPOT_MAX_DARKNESS:g} as dark as its surroundings "
                f"({len(preview.spots)} weaker dark spots)")
    return (f"{len(counted)} dark round spots, most about {size:g} nm across "
            f"(centre at most {SPOT_MAX_DARKNESS:g} of its surroundings)")


def _signed(best: float, centre: float) -> float:
    """A pair's best, written on the same side as the estimate."""

    return centre + axis_difference(best, centre)


# --- Title page ----------------------------------------------------------------------------


def title_page(previews: Sequence[SeriesPreview], *, label: str, written: str,
               software: str) -> Figure:
    figure = style.new_page(
        f"Raw tilts · {label}",
        f"{len(previews)} tilt-series · gold beads and the tilt axis, before any lane · "
        f"written {written}",
    )
    header = ["Series", "Tilts", "Angles", "Bead-like spots", "Size",
              "Axis (tilts)", "Runs", "Axis (mdoc)"]
    rows = []
    for preview in previews:
        angles = preview.angles_deg
        size = typical_diameter(preview.spots)
        axis = preview.axis.angle_deg
        rows.append([
            preview.series,
            str(len(angles)),
            f"{min(angles):+.0f} to {max(angles):+.0f}°",
            str(len(preview.bead_like)),
            f"{size:g} nm" if size is not None else "–",
            axis_text(axis, preview.recorded.angle_deg) if axis is not None else "not measured",
            preview.axis.reading if axis is not None else "–",
            (f"{preview.recorded.angle_deg:.1f}°" if preview.recorded.angle_deg is not None
             else "none"),
        ])
    table_axes = figure.add_axes((0.035, 0.52, 0.93, 0.32))
    table_axes.axis("off")
    table = table_axes.table(cellText=rows[:12], colLabels=header, loc="upper left",
                             cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(style.FONT_SMALL)
    table.scale(1.0, 1.5)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID)
        if row == 0:
            cell.set_text_props(fontweight="bold")
    fit_width(table, table_axes)
    if len(rows) > 12:
        figure.text(0.035, 0.5, f"and {len(rows) - 12} more series, each on its own pages",
                    fontsize=style.FONT_SMALL, color=style.INK_SECONDARY)

    figure.text(0.035, 0.45, "How to read these pages", fontsize=style.FONT_PANEL_TITLE + 1,
                fontweight="bold", va="top")
    y = 0.41
    for paragraph in HOW_TO_READ:
        lines = textwrap.wrap(paragraph, 135)
        figure.text(0.05, y, "•", fontsize=style.FONT_TEXT, va="top")
        figure.text(0.062, y, "\n".join(lines), fontsize=style.FONT_TEXT, va="top",
                    linespacing=1.35)
        y -= 0.027 * len(lines) + 0.02
    figure.text(0.035, 0.05, f"Written by {software}.", fontsize=style.FONT_SMALL,
                color=style.INK_SECONDARY)
    return figure


# --- Tilts ---------------------------------------------------------------------------------


def _shown(picture: TiltPicture) -> np.ndarray:
    return smooth(flatten(picture.image).astype(np.float32))


def _draw(axes: Axes, image: np.ndarray, limits: tuple[float, float] | None = None) -> None:
    low, high = limits if limits is not None else display_range([image])
    axes.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower",
                interpolation="nearest")
    axes.set_xlim(-0.5, image.shape[1] - 0.5)
    axes.set_ylim(-0.5, image.shape[0] - 0.5)


def draw_axis(axes: Axes, angle_deg: float, shape: tuple[int, ...], *, colour: str,
              style_: str = "-") -> None:
    """IMOD's axis through the picture's centre: along (x, y) = (-sin φ, cos φ)."""

    phi = math.radians(angle_deg)
    centre_x, centre_y = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    reach = math.hypot(*shape)
    dx, dy = -math.sin(phi) * reach, math.cos(phi) * reach
    axes.plot([centre_x - dx, centre_x + dx], [centre_y - dy, centre_y + dy], style_,
              color=colour, linewidth=1.4, alpha=0.9)


def _axis_legend(figure: Figure, preview: SeriesPreview, y: float) -> None:
    handles = []
    labels = []
    if preview.axis.angle_deg is not None:
        handles.append(Line2D([], [], color=AXIS_COLOUR, linewidth=1.4))
        labels.append(f"axis from the tilts, {axis_text(preview.axis.angle_deg)}")
    if preview.recorded.angle_deg is not None:
        handles.append(Line2D([], [], color=RECORDED_COLOUR, linewidth=1.4, linestyle="--"))
        labels.append(f"axis the mdoc gives, {preview.recorded.angle_deg:.1f}°")
    if handles:
        figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.035, y),
                      frameon=False, ncol=len(handles), fontsize=style.FONT_SMALL)


def tilts_page(preview: SeriesPreview) -> Figure:
    lowest = preview.pictures[0]
    highest = preview.pictures[-1]
    figure = style.new_page(
        f"{preview.series} · Tilts",
        f"{len(preview.angles_deg)} tilts from {min(preview.angles_deg):+.1f} to "
        f"{max(preview.angles_deg):+.1f}° · {preview.zero.pixel_angstrom:.1f} Å per pixel "
        f"(binned {preview.zero.binning}x from {preview.raw_pixel_angstrom:.3g} Å) · "
        f"read as {preview.entry}",
    )
    grid = GridSpec(1, 3, figure=figure, left=0.035, right=0.965, top=0.8,
                    bottom=style.PICTURE_BOTTOM + 0.04, wspace=0.05)
    for column, picture in enumerate((lowest, preview.zero, highest)):
        axes = figure.add_subplot(grid[0, column])
        image = _shown(picture)
        _draw(axes, image)
        if preview.axis.angle_deg is not None:
            draw_axis(axes, preview.axis.angle_deg, image.shape, colour=AXIS_COLOUR)
        if preview.recorded.angle_deg is not None:
            draw_axis(axes, preview.recorded.angle_deg, image.shape, colour=RECORDED_COLOUR,
                      style_="--")
        title = f"{picture.angle_deg:+.1f}° · {picture.source}"
        fraction = preview.exposure_fraction(picture)
        if fraction is not None and picture is not preview.zero:
            title += f"\nmean {100 * fraction:.0f}% of the {preview.zero.angle_deg:+.1f}° tilt's"
        style.image_axes(axes, title)
        if column == 0:
            add_scale_bar(axes, picture.pixel_angstrom, image.shape[1], image.shape[0])
    _axis_legend(figure, preview, 0.86)
    style.caption(
        figure,
        "The lowest tilt, the one nearest 0° and the highest, each on its own grey scale. At "
        "high tilt the picture is shortened across the tilt axis and darker, since the beam "
        "crosses more ice; along the axis nothing changes. A dark band at one edge of every "
        "tilt is more often the grid or the lamella's edge than the axis."
        + "".join(f" Note: {note}." for note in preview.notes),
    )
    return figure


# --- Beads ---------------------------------------------------------------------------------


def _crop(image: np.ndarray, y: float, x: float, half: int) -> np.ndarray:
    out = np.full((2 * half + 1, 2 * half + 1), np.nan, dtype=np.float32)
    top, left = int(round(y)) - half, int(round(x)) - half
    source_rows = slice(max(0, top), min(image.shape[0], top + 2 * half + 1))
    source_cols = slice(max(0, left), min(image.shape[1], left + 2 * half + 1))
    out[source_rows.start - top:source_rows.stop - top,
        source_cols.start - left:source_cols.stop - left] = image[source_rows, source_cols]
    return out


def beads_page(preview: SeriesPreview) -> Figure:
    zero = preview.zero
    counted = preview.bead_like
    figure = style.new_page(f"{preview.series} · Gold beads",
                            f"The tilt nearest 0°, {zero.angle_deg:+.1f}° · "
                            + beads_line(preview))
    overview = figure.add_axes((0.035, style.PICTURE_BOTTOM + 0.02, 0.43, 0.66))
    image = _shown(zero)
    _draw(overview, image)
    for spot in preview.spots:
        overview.add_patch(Circle(
            (spot.x, spot.y), radius=max(3.0, 1.8 * spot.radius_px), fill=False,
            edgecolor=style.FIDUCIAL_COLOUR if spot.counts else style.INK_MUTED,
            linewidth=0.9 if spot.counts else 0.5,
        ))
    style.image_axes(overview, f"{len(counted)} counted (fuchsia), "
                               f"{len(preview.spots) - len(counted)} weaker (grey)")
    add_scale_bar(overview, zero.pixel_angstrom, image.shape[1], image.shape[0])

    darkest = sorted(preview.spots, key=lambda spot: spot.darkness)[:GALLERY_SPOTS]
    columns, rows = 6, 4
    grid = GridSpec(rows, columns, figure=figure, left=0.5, right=0.965, top=0.8,
                    bottom=style.PICTURE_BOTTOM + 0.02, wspace=0.08, hspace=0.3)
    scale = zero.binning / preview.fine_binning
    largest = max((spot.radius_px for spot in darkest), default=5.0)
    half = max(12, int(round(3.0 * largest * scale)))
    crops = [smooth(_crop(preview.fine, spot.y * scale + (scale - 1) / 2.0,
                          spot.x * scale + (scale - 1) / 2.0, half))
             for spot in darkest]
    limits = display_range(crops) if crops else (0.0, 1.0)
    for number in range(rows * columns):
        axes = figure.add_subplot(grid[divmod(number, columns)])
        style.image_axes(axes)
        if number >= len(crops):
            axes.axis("off")
            continue
        _draw(axes, crops[number], limits)
        spot = darkest[number]
        axes.set_title(f"{spot.darkness:.2f} · {spot.diameter_nm:g} nm",
                       fontsize=style.FONT_TINY,
                       color=style.INK if spot.counts else style.INK_MUTED)
    if not crops:
        figure.text(0.5, 0.5, "No dark round spot stood out at all.", fontsize=style.FONT_TEXT,
                    color=style.INK_SECONDARY)
    figure.text(0.5, 0.835, f"The {len(crops)} darkest spots, at "
                f"{preview.fine_pixel_angstrom:.1f} Å per pixel, on one grey scale: "
                "darkness · size", fontsize=style.FONT_TEXT, va="bottom")
    style.caption(
        figure,
        "Each spot is titled with its darkness, the mean of its centre over the mean of a "
        "ring around it, and the filter size that answered it best. Gold shows as many "
        "uniform round discs of one size, often with a pale rim from the defocus. Ice, "
        "carbon edges and dense organelles give larger, irregular or fainter spots; a "
        "detector defect gives a few tiny ones of the smallest size. "
        f"The bead route's tracking QC fails below {MIN_TRACKED_BEADS} tracked beads.",
    )
    return figure


# --- Axis ----------------------------------------------------------------------------------


def axis_page(preview: SeriesPreview) -> Figure:
    axis = preview.axis
    figure = style.new_page(f"{preview.series} · Tilt axis", measured_axis_line(preview))
    figure.text(0.035, 0.865, "The mdoc: " + recorded_axis_line(preview),
                fontsize=style.FONT_TEXT, color=style.INK_SECONDARY, va="top")
    plot = figure.add_axes((0.07, style.CONTENT_BOTTOM + 0.02, 0.34, 0.58))
    if axis.angle_deg is None:
        plot.axis("off")
        plot.text(0.0, 0.5, f"Not measured: {axis.reason}.", fontsize=style.FONT_TEXT)
        style.caption(figure, "The axis is read from pairs of neighbouring high tilts.")
        return figure
    centre = axis.angle_deg
    for number, pair in enumerate(axis.pairs):
        shifted = np.array([_signed(value, centre) for value in pair.candidates_deg])
        order = np.argsort(shifted)
        colour = style.MAGNITUDE_MAP(0.35 + 0.65 * number / max(1, len(axis.pairs) - 1))
        plot.plot(shifted[order], pair.scores[order], color=colour, linewidth=1.0,
                  label=f"{pair.lower_deg:+.0f}° and {pair.higher_deg:+.0f}°")
        plot.plot([_signed(pair.best_deg, centre)], [pair.scores.max()], "o", color=colour,
                  markersize=3.5)
    plot.axvline(centre, color=AXIS_COLOUR, linewidth=1.2)
    if preview.recorded.angle_deg is not None:
        plot.axvline(_signed(preview.recorded.angle_deg, centre), color=RECORDED_COLOUR,
                     linewidth=1.2, linestyle="--")
    plot.set_xlim(centre - 90.0, centre + 90.0)
    plot.set_xlabel("axis tried, degrees (IMOD's angle, from y, counterclockwise)")
    plot.set_ylabel("match of the pair, higher tilt stretched across it")
    plot.grid(True, axis="x")
    plot.legend(loc="lower center", fontsize=style.FONT_TINY, frameon=False, ncol=2,
                title="pairs of tilts", title_fontsize=style.FONT_TINY)

    pair = shown_pair(preview)
    lower = preview.picture(pair.lower_deg)
    higher = preview.picture(pair.higher_deg)
    across = (centre + 90.0) % 180.0
    for column, (tried, title) in enumerate((
        (centre, f"across the axis found, {axis_text(centre).split(' (')[0]}"),
        (across, f"across the axis turned by 90°, {axis_text(across).split(' (')[0]}"),
    )):
        first, second = overlay_pair(lower, higher, tried)
        axes = figure.add_axes((0.45 + 0.26 * column, style.CONTENT_BOTTOM + 0.0, 0.255, 0.62))
        image = _overlay(first, second)
        axes.imshow(image, origin="lower", interpolation="nearest")
        draw_axis(axes, tried, image.shape, colour="white", style_=":")
        axes.set_xlim(-0.5, image.shape[1] - 0.5)
        axes.set_ylim(-0.5, image.shape[0] - 0.5)
        style.image_axes(axes, title)
        axes.title.set_fontsize(style.FONT_SMALL)
    figure.text(0.45, 0.8, f"{pair.higher_deg:+.1f}° (cyan) stretched by "
                f"{1 / pair_factor(pair.lower_deg, pair.higher_deg):.3f} onto "
                f"{pair.lower_deg:+.1f}° (red)", fontsize=style.FONT_TEXT, va="bottom")
    style.caption(
        figure,
        "Left: for each pair of neighbouring high tilts, how well the higher one matches the "
        "lower once stretched by cos(lower) / cos(higher) across each axis tried; a thin "
        "specimen matches only across the real axis. The estimate (blue) is the median of the "
        "pairs' bests. Right: the pair with the most stretch among those that agree, overlaid "
        "red and cyan: grey where the two agree, coloured fringes where they do not; the "
        "dotted line is the axis tried. "
        "Stretched across the right axis the pair turns grey; across the wrong one it keeps "
        "fringes towards two edges."
        + "".join(f" Note: {note}." for note in preview.notes),
    )
    return figure


def shown_pair(preview: SeriesPreview) -> PairScan:
    """The pair drawn overlaid: the most stretch among those that agree with the estimate."""

    axis = preview.axis
    assert axis.angle_deg is not None
    centre = axis.angle_deg
    agreeing = [pair for pair in axis.pairs
                if abs(axis_difference(pair.best_deg, centre)) <= AXIS_AGREEMENT_DEG]
    return min(agreeing or list(axis.pairs),
               key=lambda pair: pair_factor(pair.lower_deg, pair.higher_deg))


def _overlay(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    height, width = first.shape
    trim_y, trim_x = int(height * 0.15), int(width * 0.15)
    window = (slice(trim_y, height - trim_y), slice(trim_x, width - trim_x))

    def unit(image: np.ndarray) -> np.ndarray:
        part = smooth(image.astype(np.float32), 1.5)[window]
        low, high = np.percentile(part, [2.0, 98.0])
        return np.clip((part - low) / max(high - low, 1e-9), 0.0, 1.0)

    red, cyan = unit(first), unit(second)
    return np.dstack([red, cyan, cyan])
