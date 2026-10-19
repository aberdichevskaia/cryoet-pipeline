"""The look every report page shares: page size, type, ink, colours, header and footer.

Colours follow one rule each. Text is ink, never a series colour. Data series
take categorical hues in a fixed order. Magnitudes take one blue ramp, light to
dark. Status colours mean only pass, warning and fail, and always sit beside
the word, so that no verdict is carried by colour alone.

Overlays keep one colour per thing across every page: picks are orange,
membranes aqua, the template mask and tracked fiducials blue.

Type sizes are roles rather than numbers, so that making the report larger or
smaller is one change here and every page follows.
"""

from __future__ import annotations

from typing import Any

from matplotlib import colormaps, patheffects
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

PAGE_SIZE = (11.69, 8.27)
"""A4 landscape, in inches: wide enough for three tomogram panels and a summary table."""

FONT_PAGE_TITLE = 17.0
FONT_SUBTITLE = 11.0
FONT_PANEL_TITLE = 11.0
FONT_TEXT = 10.0
"""Body text, axis labels, the titles of picture panels."""
FONT_CAPTION = 9.5
FONT_SMALL = 9.0
"""Tick labels, legends, table cells, footers."""
FONT_TINY = 8.5
"""Labels written inside a plot or a picture: reference lines, scale bars, colour bars."""
FONT_HERO = 30.0

TITLE_Y = 0.958
SUBTITLE_Y = 0.906
CONTENT_TOP = 0.85
"""Where the panels of a page may start, below the title and subtitle."""
CAPTION_Y = 0.118
"""The top of a page's caption; four lines of it end above the footer."""
CONTENT_BOTTOM = 0.2
"""Where plots with axis labels end, leaving the caption its room."""
PICTURE_BOTTOM = 0.16
"""Where pictures without axis labels end."""
FOOTER_Y = 0.014

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
SPINE = "#c9c8c2"
NEUTRAL_FILL = "#d9d8d2"

SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
SERIES_AQUA = "#1baf7a"

PICK_COLOUR = SERIES_ORANGE
MEMBRANE_COLOUR = SERIES_AQUA
MASK_COLOUR = SERIES_BLUE
FIDUCIAL_COLOUR = SERIES_BLUE

_BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

MAGNITUDE_MAP = LinearSegmentedColormap.from_list("report-blue", list(_BLUE_RAMP))
"""For continuous fields such as the correlation map: near zero recedes to the page."""

MARKER_MAP = LinearSegmentedColormap.from_list("report-blue-marks", list(_BLUE_RAMP[3:]))
"""For marks coloured by a magnitude: starts dark enough that the lightest mark still shows."""

GRAY_WITH_GAPS = colormaps["gray"].with_extremes(bad="#f1f0ec")
"""Grey for pictures, with gaps and the outside of the volume in a light neutral."""

STATUS_COLOURS = {
    "pass": "#0ca30c",
    "warning": "#fab219",
    "fail": "#d03b3b",
    "unknown": NEUTRAL_FILL,
}
STATUS_LABELS = {"pass": "PASS", "warning": "WARN", "fail": "FAIL", "unknown": "?"}

INTERPOLATED_COLOUR = STATUS_COLOURS["warning"]
"""A tilt CTFFIND found no Thon rings on, kept with an interpolated defocus."""
LEFT_OUT_COLOUR = STATUS_COLOURS["fail"]
"""A tilt CTFFIND found no Thon rings on, left out of the tomogram."""

RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": FONT_TEXT,
    "text.color": INK,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": FONT_TEXT,
    "axes.titlesize": FONT_PANEL_TITLE,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": FONT_SMALL,
    "ytick.labelsize": FONT_SMALL,
    "legend.fontsize": FONT_SMALL,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "image.interpolation": "nearest",
    "pdf.compression": 9,
    "pdf.fonttype": 42,
}


def tint(colour: str, strength: float = 0.35) -> tuple[float, float, float]:
    """A status colour thinned towards white, as a cell background under ink text."""

    red, green, blue = to_rgb(colour)
    return (
        1.0 - strength * (1.0 - red),
        1.0 - strength * (1.0 - green),
        1.0 - strength * (1.0 - blue),
    )


def new_page(title: str, subtitle: str | None = None) -> Figure:
    figure = Figure(figsize=PAGE_SIZE)
    figure.text(0.035, TITLE_Y, title, fontsize=FONT_PAGE_TITLE, fontweight="bold", va="top")
    if subtitle:
        figure.text(0.035, SUBTITLE_Y, subtitle, fontsize=FONT_SUBTITLE, color=INK_SECONDARY,
                    va="top")
    return figure


def footer(figure: Figure, left: str, page: int, pages: int) -> None:
    figure.text(0.035, FOOTER_Y, left, fontsize=FONT_SMALL, color=INK_MUTED)
    figure.text(0.965, FOOTER_Y, f"page {page} of {pages}", fontsize=FONT_SMALL,
                color=INK_MUTED, ha="right")


def caption(figure: Figure, text: str, y: float = CAPTION_Y) -> None:
    """Explanatory text under a page's panels, in secondary ink."""

    figure.text(0.035, y, text, fontsize=FONT_CAPTION, color=INK_SECONDARY, va="top", wrap=True,
                linespacing=1.3)


def image_axes(axes: Axes, title: str | None = None) -> None:
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)
    if title:
        axes.set_title(title, fontsize=FONT_TEXT, loc="left")


def colour_bar_style(colorbar: Any, label: str | None = None) -> None:
    if label:
        colorbar.set_label(label, fontsize=FONT_SMALL)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=FONT_TINY)


def scale_bar(axes: Axes, length_px: float, label: str, *, image_width_px: float,
              image_height_px: float) -> None:
    """A white bar with a dark edge, readable on any part of a grey picture."""

    x0 = 0.04 * image_width_px
    y0 = 0.06 * image_height_px
    height = max(1.0, 0.012 * image_height_px)
    axes.add_patch(
        Rectangle(
            (x0, y0),
            length_px,
            height,
            facecolor="white",
            edgecolor=INK,
            linewidth=0.4,
            zorder=5,
        )
    )
    text = axes.text(
        x0 + length_px / 2.0,
        y0 + height * 2.0,
        label,
        color="white",
        fontsize=FONT_TINY,
        ha="center",
        va="bottom",
        zorder=5,
    )
    text.set_path_effects([patheffects.withStroke(linewidth=1.8, foreground=INK)])


def status_cell(status: str) -> tuple[str, tuple[float, float, float]]:
    """The word and the background for one status cell of a table."""

    key = status if status in STATUS_COLOURS else "unknown"
    return STATUS_LABELS[key], tint(STATUS_COLOURS[key])
