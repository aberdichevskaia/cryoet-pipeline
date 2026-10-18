"""A classification report: what a multireference classification started from, and
what each of its iterations made of it.

The pages, in order: the title; the parameters; a gallery of particles at their
starting poses; the average of all particles at those poses; the starting
references; the class sizes over the iterations; every iteration's classes; the
last classes by tomogram and by picking score; a gallery of each last class at
its own poses; anything that could not be read.

A map is drawn as Dynamo's `dslices` shows one: a run of successive z sections
through the box centre, on one grey scale per map, values as they are (not
inverted). A particle in a gallery is rotated and shifted into the references'
frame by its pose, then shown as the mean of its 10 central z sections, smoothed
by one voxel, on one grey scale per page.

The pages see only `ClassificationRun`. `reports.dynamo_project` fills it from a
Dynamo project; the pipeline's own classification can fill it the same way.
"""

from __future__ import annotations

import math
import textwrap
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

import matplotlib
import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.imaging import display_range, nice_length, smooth

T = TypeVar("T")

SECTIONS = 10
SECTION_STEP_PX = 3
"""Ten sections three voxels apart: 27 voxels about the centre, a ribosome's depth at bin8."""
GALLERY_SLAB_PX = 10
GALLERY_COLUMNS = 8
GALLERY_ROWS = 4
CLASS_GALLERY_TILES = 8
MAPS_PER_PAGE = 3
PARAMETER_ROWS_PER_PAGE = 33
PROBLEM_LINES_PER_PAGE = 30
GALLERY_SEED = 20261016
"""Which particles a gallery shows: fixed, so two reports of one run show the same ones."""

CHANNEL_COLOURS = (
    style.SERIES_BLUE,
    style.SERIES_ORANGE,
    style.SERIES_AQUA,
    "#8e5bd0",
    "#c9407a",
    "#7a7a72",
)

HOW_TO_READ = (
    "Every channel starts from its own reference. In each iteration every channel "
    "searches every particle near its pose, and each particle goes to the channel whose "
    "reference it correlates with best; each channel's average of its particles is its "
    "next reference. A channel that receives no particle is dropped for good.",
    "Maps are drawn as Dynamo's dslices shows them: successive z sections through the box "
    "centre, labelled with their distance from it, on one grey scale per map. Values are "
    "shown as they are, not inverted.",
    "Gallery particles are rotated and shifted into the references' frame by their pose, "
    "the way Dynamo averages them, and shown as the mean of their 10 central z sections, "
    "smoothed by one voxel, on one grey scale per page.",
    "An average at template-matching poses looks like the template even when its particles "
    "are noise, so a class that looks like the particle is not by itself evidence that its "
    "particles are.",
)


@dataclass(frozen=True)
class Volume:
    """A map the report draws, read only when its page is drawn."""

    source: str
    load: Callable[[], np.ndarray]
    """Returns the map as a (z, y, x) array."""


@dataclass(frozen=True)
class ClassResult:
    channel: int
    members: np.ndarray
    """Tags of the particles the class holds after the iteration, ascending."""
    average: Volume | None
    median_cc: float | None


@dataclass(frozen=True)
class IterationResult:
    number: int
    searched: tuple[int, ...]
    """The channels that searched in this iteration."""
    classes: dict[int, ClassResult]
    """The channels that kept particles, by channel."""
    started: str | None = None
    finished: str | None = None


@dataclass(frozen=True)
class Parameter:
    group: str
    name: str
    value: str


Pose = tuple[int, int] | None
"""None for a particle's starting pose, or (iteration, channel) for that channel's
pose after that iteration."""


@dataclass
class ClassificationRun:
    name: str
    source: Path
    software: str
    box_px: int
    pixel_size_angstrom: float | None
    tags: np.ndarray
    """Every particle of the starting table."""
    references: dict[int, Volume]
    """The starting reference of every channel."""
    reference_notes: dict[int, str]
    parameters: list[Parameter]
    iterations: list[IterationResult]
    """The finished iterations, in order."""
    aligned_particle: Callable[[int, Pose], np.ndarray]
    """A particle in the references' frame, as (z, y, x), NaN where no voxel of it reaches."""
    initial_average: Volume | None
    planned_iterations: int | None = None
    series: dict[int, str] = field(default_factory=dict)
    """Each tag's tilt-series, when known."""
    scores: dict[int, float] = field(default_factory=dict)
    """Each tag's picking score, when known."""
    score_label: str = "picking score"
    decoys: frozenset[int] = frozenset()
    """Tags of boxes at random positions mixed in as a control, when there are any."""
    problems: list[str] = field(default_factory=list)
    """Everything that could not be read, one line each."""


@dataclass(frozen=True)
class PlannedPage:
    label: str
    build: Callable[[], Figure]


@dataclass(frozen=True)
class ClassificationReportOutcome:
    path: Path
    pages: int
    problems: list[str]


def write_classification_report(
    path: Path, run: ClassificationRun, *, written: str, software: str
) -> ClassificationReportOutcome:
    """Write the report to `path`; a page that fails is replaced by one saying why."""

    planned = plan_pages(run, written=written, software=software)
    problems = list(run.problems)
    footer = f"{run.name} · {written}"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    metadata = {
        "Title": f"Classification report · {run.name}",
        "Subject": "cryo-ET subtomogram classification",
        "Creator": software,
    }
    with matplotlib.rc_context(cast(Any, style.RC)), PdfPages(partial, metadata=metadata) as pdf:
        for number, page in enumerate(planned, 1):
            try:
                figure = page.build()
            except Exception as error:  # noqa: BLE001 - one page must not lose the report
                problems.append(f"{page.label}: could not be drawn: {type(error).__name__}: "
                                f"{error}")
                figure = _error_page(page.label, error)
            style.footer(figure, footer, number, len(planned))
            pdf.savefig(figure)
    partial.replace(path)
    return ClassificationReportOutcome(path, len(planned), problems)


def plan_pages(run: ClassificationRun, *, written: str, software: str) -> list[PlannedPage]:
    """Every page the run gets, decided by what exists for it, before any is drawn."""

    pages = [PlannedPage("title", partial(title_page, run, written=written, software=software))]
    pages.extend(
        PlannedPage("parameters", build) for build in parameter_pages(run.parameters)
    )
    pages.append(PlannedPage("starting gallery", partial(starting_gallery_page, run)))
    if run.initial_average is not None:
        pages.append(PlannedPage("initial average", partial(initial_average_page, run)))
    for channels in _chunks(sorted(run.references), MAPS_PER_PAGE):
        pages.append(PlannedPage("starting references", partial(references_page, run, channels)))
    if run.iterations:
        pages.append(PlannedPage("class sizes", partial(class_sizes_page, run)))
        if run.decoys:
            pages.append(PlannedPage("decoys", partial(decoys_page, run)))
    for iteration in run.iterations:
        for channels in _chunks(sorted(iteration.classes), MAPS_PER_PAGE) or [[]]:
            pages.append(PlannedPage(f"iteration {iteration.number}",
                                     partial(iteration_page, run, iteration, channels)))
    if run.iterations and (run.series or run.scores):
        pages.append(PlannedPage("last classes", partial(composition_page, run)))
    if run.iterations:
        for channels in _chunks(sorted(run.iterations[-1].classes), MAPS_PER_PAGE):
            pages.append(PlannedPage("class galleries", partial(class_gallery_page, run, channels)))
    for lines in _chunks(run.problems, PROBLEM_LINES_PER_PAGE):
        pages.append(PlannedPage("not read", partial(problems_page, lines)))
    return pages


def _chunks(items: Sequence[T], size: int) -> list[list[T]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# Pages


def title_page(run: ClassificationRun, *, written: str, software: str) -> Figure:
    figure = style.new_page(
        f"Classification report · {run.name}",
        f"{run.software} · written {written}",
    )
    last = run.iterations[-1] if run.iterations else None
    finished = str(len(run.iterations))
    if run.planned_iterations:
        finished += f" of {run.planned_iterations}"
    tiles = (
        (f"{len(run.tags):,}", "particles"),
        (str(len(run.references)), "starting references"),
        (finished, "iterations finished"),
        (str(len(last.classes)) if last is not None else "–", "classes left"),
    )
    for index, (value, label) in enumerate(tiles):
        x = 0.035 + index * 0.235
        figure.text(x, 0.835, value, fontsize=style.FONT_HERO, va="top")
        figure.text(x, 0.765, label, fontsize=style.FONT_TEXT, color=style.INK_SECONDARY,
                    va="top")

    figure.text(0.035, 0.675, "How to read this report", fontsize=style.FONT_PANEL_TITLE + 1,
                fontweight="bold", va="top")
    y = 0.63
    for paragraph in HOW_TO_READ:
        lines = textwrap.wrap(paragraph, 128)
        figure.text(0.05, y, "•", fontsize=style.FONT_TEXT, va="top")
        figure.text(0.062, y, "\n".join(lines), fontsize=style.FONT_TEXT, va="top",
                    linespacing=1.35)
        y -= 0.0285 * len(lines) + 0.02
    figure.text(0.035, 0.08, f"Read from {run.source}", fontsize=style.FONT_SMALL,
                color=style.INK_SECONDARY)
    figure.text(0.035, 0.05, f"Written by {software}. It reads the run's files and changes "
                "none of them.", fontsize=style.FONT_SMALL, color=style.INK_SECONDARY)
    return figure


def parameter_pages(parameters: Sequence[Parameter]) -> list[Callable[[], Figure]]:
    rows = list(parameters) or [Parameter("", "none recorded", "")]
    return [partial(_parameter_page, chunk) for chunk in _chunks(rows, PARAMETER_ROWS_PER_PAGE)]


def _parameter_page(rows: list[Parameter]) -> Figure:
    figure = style.new_page("Parameters", "As the run recorded them")
    text = []
    previous = None
    for row in rows:
        text.append([row.group if row.group != previous else "", row.name, row.value])
        previous = row.group
    axes = figure.add_axes((0.035, 0.05, 0.93, style.CONTENT_TOP - 0.05))
    axes.axis("off")
    table = axes.table(cellText=text, colLabels=["", "Parameter", "Value"], cellLoc="left",
                       colLoc="left", loc="upper left", colWidths=[0.16, 0.24, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(style.FONT_SMALL)
    table.scale(1.0, 1.15)
    for (row_index, column), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_text_props(fontweight="bold", color=style.INK_SECONDARY)
        elif column == 0:
            cell.set_text_props(fontweight="bold")
    return figure


def starting_gallery_page(run: ClassificationRun) -> Figure:
    count = GALLERY_ROWS * GALLERY_COLUMNS
    tags = sample_tags(run.tags, count)
    figure = style.new_page(
        "Particles at their starting poses",
        f"{len(tags)} of {len(run.tags):,} particles, chosen at random",
    )
    tiles = [(int(tag), gallery_tile(run.aligned_particle(int(tag), None))) for tag in tags]
    _draw_tiles(figure, run, tiles, top=style.CONTENT_TOP - 0.035, bottom=style.PICTURE_BOTTOM,
                columns=GALLERY_COLUMNS)
    style.caption(
        figure,
        "Each particle rotated and shifted by its starting pose into the frame its "
        f"references and averages use, then the mean of its {GALLERY_SLAB_PX} central z "
        "sections, smoothed by one voxel. One grey scale for the page. Labels: tag, "
        "tilt-series and picking score where known.",
    )
    return figure


def initial_average_page(run: ClassificationRun) -> Figure:
    volume = cast(Volume, run.initial_average)
    data = np.asarray(volume.load(), dtype=float)
    figure = style.new_page(
        "Average of all particles at their starting poses",
        f"{len(run.tags):,} particles · {volume.source}",
    )
    _draw_sections(figure, data, run.pixel_size_angstrom, x0=0.035, y0=0.6, width=0.93,
                   height=0.2)
    views = central_views(data)
    limits = display_range(views)
    names = ("XY, central z section", "XZ, central y section", "YZ, central x section")
    for index, (view, name) in enumerate(zip(views, names, strict=True)):
        axes = figure.add_axes((0.035 + index * 0.315, 0.2, 0.3, 0.33))
        axes.imshow(view, cmap=style.GRAY_WITH_GAPS, vmin=limits[0], vmax=limits[1],
                    origin="lower")
        style.image_axes(axes, name)
        axes.set_aspect("equal")
        if index == 0:
            _scale_bar(axes, run.pixel_size_angstrom, view.shape)
    style.caption(
        figure,
        "Top: successive z sections through the box centre, as Dynamo's dslices shows a map. "
        "Bottom: the central section along each axis. Before any classification: the "
        "reference every class has to improve on. When computed for this report, it is "
        "Dynamo's dynamo_average with fcompensate, fmin 1 and normalisation under the "
        "alignment mask, through the pipeline's port of it.",
    )
    return figure


def references_page(run: ClassificationRun, channels: Sequence[int]) -> Figure:
    figure = style.new_page("Starting references",
                            f"{len(run.references)} channels; each map on its own grey scale")
    for row, channel in enumerate(channels):
        volume = run.references[channel]
        note = run.reference_notes.get(channel, "")
        label = f"channel {channel}" + (f" · {note}" if note else "") + f" · {volume.source}"
        _map_row(figure, run, row, label, np.asarray(volume.load(), dtype=float),
                 CHANNEL_COLOURS[(channel - 1) % len(CHANNEL_COLOURS)])
    style.caption(
        figure,
        "What each channel searched against in the first iteration. " + MAP_ROW_GUIDE,
    )
    return figure


def class_sizes_page(run: ClassificationRun) -> Figure:
    figure = style.new_page("Class sizes",
                            "Particles in each class after every iteration")
    channels = sorted(run.references)
    total = len(run.tags)
    changed = class_changes(run.iterations)
    rows = []
    for iteration, moved in zip(run.iterations, changed, strict=True):
        cells = [str(iteration.number)]
        for channel in channels:
            result = iteration.classes.get(channel)
            if result is not None:
                cells.append(f"{len(result.members):,} ({100 * len(result.members) / total:.0f}%)")
            elif channel in iteration.searched:
                cells.append("dropped")
            else:
                cells.append("–")
        cells.append("–" if moved is None else f"{moved:,}")
        rows.append(cells)
    labels = ["Iteration", *(f"Channel {k}" for k in channels), "Changed class"]

    with_chart = len(run.iterations) > 1
    table_axes = figure.add_axes((0.035, 0.2, 0.42 if with_chart else 0.93,
                                  style.CONTENT_TOP - 0.2))
    table_axes.axis("off")
    table = table_axes.table(cellText=rows, colLabels=labels, cellLoc="center",
                             loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(style.FONT_SMALL)
    table.scale(1.0, 1.45)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_text_props(fontweight="bold", color=style.INK_SECONDARY)
    table.auto_set_column_width(list(range(len(labels))))

    if with_chart:
        axes = figure.add_axes((0.55, 0.24, 0.41, style.CONTENT_TOP - 0.28))
        numbers = [iteration.number for iteration in run.iterations]
        for channel in channels:
            sizes = [
                len(iteration.classes[channel].members) if channel in iteration.classes
                else (0 if channel in iteration.searched else np.nan)
                for iteration in run.iterations
            ]
            axes.plot(numbers, sizes, marker="o",
                      color=CHANNEL_COLOURS[(channel - 1) % len(CHANNEL_COLOURS)],
                      label=f"channel {channel}")
        axes.set_xlabel("iteration")
        axes.set_ylabel("particles")
        axes.set_xticks(numbers)
        axes.set_ylim(bottom=0)
        axes.grid(True, axis="y")
        axes.legend(loc="best")
    style.caption(
        figure,
        "Dropped: the channel searched in that iteration but no particle chose it, so it "
        "searches no more. Changed class: particles whose class differs from the iteration "
        "before; when it nears zero the classes have settled.",
        y=0.17,
    )
    return figure


def decoy_shares(run: ClassificationRun, iteration: IterationResult) -> dict[int, DecoyShare]:
    """Each class's share of all decoys and of all real particles after `iteration`."""

    decoys = len(run.decoys)
    real = len(run.tags) - decoys
    shares = {}
    for channel, result in sorted(iteration.classes.items()):
        held = sum(1 for tag in result.members if int(tag) in run.decoys)
        shares[channel] = DecoyShare(held, decoys, len(result.members) - held, real)
    return shares


@dataclass(frozen=True)
class DecoyShare:
    decoys: int
    all_decoys: int
    real: int
    all_real: int

    @property
    def decoy_share(self) -> float:
        return self.decoys / self.all_decoys if self.all_decoys else math.nan

    @property
    def real_share(self) -> float:
        return self.real / self.all_real if self.all_real else math.nan

    @property
    def enrichment(self) -> float:
        """Decoy share over real share: 1 when the class takes decoys as it takes picks."""

        return self.decoy_share / self.real_share if self.real_share else math.inf

    def decoy_interval(self, z: float = 1.96) -> tuple[float, float]:
        """Wilson's 95% interval for the decoy share."""

        n = self.all_decoys
        if not n:
            return math.nan, math.nan
        p = self.decoys / n
        denominator = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return centre - half, centre + half


def decoys_page(run: ClassificationRun) -> Figure:
    decoys = len(run.decoys)
    total = len(run.tags)
    figure = style.new_page(
        "Where the decoys went",
        f"{decoys:,} boxes at random positions among {total:,} particles "
        f"({100 * decoys / total:.0f}%), classified blind with the rest",
    )
    channels = sorted(run.references)
    rows = []
    for iteration in run.iterations:
        shares = decoy_shares(run, iteration)
        cells = [str(iteration.number)]
        for channel in channels:
            share = shares.get(channel)
            if share is None:
                cells.append("dropped" if channel in iteration.searched else "–")
            else:
                cells.append(f"{100 * share.decoy_share:.0f}% vs {100 * share.real_share:.0f}% "
                             f"(×{share.enrichment:.2f})")
        rows.append(cells)
    labels = ["Iteration", *(f"Channel {k}" for k in channels)]
    axes = figure.add_axes((0.035, 0.5, 0.5, style.CONTENT_TOP - 0.5))
    axes.axis("off")
    table = axes.table(cellText=rows, colLabels=labels, cellLoc="center", loc="upper left")
    table.auto_set_font_size(False)
    table.set_fontsize(style.FONT_SMALL)
    table.scale(1.0, 1.45)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_text_props(fontweight="bold", color=style.INK_SECONDARY)
    table.auto_set_column_width(list(range(len(labels))))

    last = run.iterations[-1]
    lines = [f"After iteration {last.number}:"]
    for channel, share in decoy_shares(run, last).items():
        low, high = share.decoy_interval()
        lines.append(
            f"class {channel}: {share.decoys:,} of {share.all_decoys:,} decoys "
            f"({100 * share.decoy_share:.1f}%, 95% interval {100 * low:.1f}–{100 * high:.1f}%) "
            f"against {share.real:,} of {share.all_real:,} real particles "
            f"({100 * share.real_share:.1f}%): ×{share.enrichment:.2f}"
        )
    figure.text(0.035, 0.44, "\n".join(lines), fontsize=style.FONT_TEXT, va="top",
                linespacing=1.5)

    if len(run.iterations) > 1:
        chart = figure.add_axes((0.62, 0.5, 0.34, style.CONTENT_TOP - 0.53))
        numbers = [iteration.number for iteration in run.iterations]
        for channel in channels:
            values = [
                decoy_shares(run, iteration)[channel].enrichment
                if channel in iteration.classes else np.nan
                for iteration in run.iterations
            ]
            chart.plot(numbers, values, marker="o",
                       color=CHANNEL_COLOURS[(channel - 1) % len(CHANNEL_COLOURS)],
                       label=f"channel {channel}")
        chart.axhline(1.0, color=style.INK_MUTED, linewidth=0.8, linestyle="--")
        chart.set_xlabel("iteration")
        chart.set_ylabel("decoy share / real share")
        chart.set_xticks(numbers)
        chart.set_ylim(bottom=0)
        chart.grid(True, axis="y")
        chart.legend(loc="best")
    style.caption(
        figure,
        "In each cell: the share of all decoys the class holds, against the share of all real "
        "particles it holds, and their ratio. A ratio of 1 everywhere means the classification "
        "treats a box centred on nothing like a pick. The reading, written before the run: a class "
        "collects junk if it holds at least twice the share of decoys it holds of real "
        "particles, and the 95% interval of its decoy share leaves out its real share.",
        y=0.2,
    )
    return figure


def iteration_page(run: ClassificationRun, iteration: IterationResult,
                   channels: Sequence[int]) -> Figure:
    dropped = [k for k in iteration.searched if k not in iteration.classes]
    subtitle = f"{len(iteration.classes)} classes"
    if dropped:
        subtitle += " · dropped: " + ", ".join(f"channel {k}" for k in dropped)
    if iteration.started and iteration.finished:
        subtitle += f" · ran {iteration.started} to {iteration.finished}"
    figure = style.new_page(f"Iteration {iteration.number}", subtitle)
    total = len(run.tags)
    for row, channel in enumerate(channels):
        result = iteration.classes[channel]
        label = (f"class {channel} · {_particles(len(result.members))} "
                 f"({100 * len(result.members) / total:.0f}%)")
        if result.median_cc is not None:
            label += f" · median CC {result.median_cc:.3f}"
        colour = CHANNEL_COLOURS[(channel - 1) % len(CHANNEL_COLOURS)]
        if result.average is None:
            figure.text(0.035, _row_top(row), label + " · no average was written",
                        fontsize=style.FONT_TEXT, color=colour, va="top")
            continue
        _map_row(figure, run, row, label, np.asarray(result.average.load(), dtype=float), colour)
    if not channels:
        figure.text(0.035, style.CONTENT_TOP, "No class kept any particle.",
                    fontsize=style.FONT_TEXT, va="top", color=style.INK_SECONDARY)
    style.caption(
        figure,
        "Each class's average after this iteration, the reference it searches with in the "
        f"next. {MAP_ROW_GUIDE} Median CC: the class members' correlation with their own "
        "class's reference.",
    )
    return figure


def composition_page(run: ClassificationRun) -> Figure:
    last = run.iterations[-1]
    channels = sorted(last.classes)
    subtitle = "Where their particles come from, and how they scored in picking"
    if run.decoys:
        subtitle += " · real particles only"
    figure = style.new_page(f"The classes after iteration {last.number}", subtitle)
    real = [int(tag) for tag in run.tags if int(tag) not in run.decoys]
    members_of = {k: [int(tag) for tag in last.classes[k].members if int(tag) not in run.decoys]
                  for k in channels}
    with_scores = bool(run.scores)
    if run.series:
        names = sorted({run.series[tag] for tag in real if tag in run.series})
        totals = {name: sum(1 for tag in real if run.series.get(tag) == name) for name in names}
        rows = []
        for channel in channels:
            members = members_of[channel]
            cells = [f"class {channel}"]
            for name in names:
                count = sum(1 for tag in members if run.series.get(int(tag)) == name)
                share = 100 * count / totals[name] if totals[name] else 0.0
                cells.append(f"{count:,} ({share:.0f}%)")
            rows.append(cells)
        rows.append(["all", *(f"{totals[name]:,}" for name in names)])
        axes = figure.add_axes((0.035, 0.24, 0.5 if with_scores else 0.93,
                                style.CONTENT_TOP - 0.24))
        axes.axis("off")
        table = axes.table(cellText=rows, colLabels=["", *names], cellLoc="center",
                           loc="upper left")
        table.auto_set_font_size(False)
        table.set_fontsize(style.FONT_SMALL)
        table.scale(1.0, 1.45)
        for (row_index, _), cell in table.get_celld().items():
            cell.set_edgecolor(style.GRID)
            cell.set_linewidth(0.5)
            if row_index == 0:
                cell.set_text_props(fontweight="bold", color=style.INK_SECONDARY)
    if with_scores:
        left = 0.6 if run.series else 0.08
        axes = figure.add_axes((left, 0.26, 0.96 - left, style.CONTENT_TOP - 0.3))
        groups = [np.array([run.scores[tag] for tag in members_of[k] if tag in run.scores])
                  for k in channels]
        everyone = np.array([run.scores[tag] for tag in real if tag in run.scores])
        data = [*groups, everyone]
        labels = [f"class {k}" for k in channels] + ["all"]
        shown = [values if values.size else np.array([np.nan]) for values in data]
        # Labels set by hand: boxplot's own keyword was renamed between matplotlib versions.
        parts = axes.boxplot(shown, showfliers=False, patch_artist=True, widths=0.6)
        axes.set_xticks(range(1, len(labels) + 1), labels)
        colours = [CHANNEL_COLOURS[(k - 1) % len(CHANNEL_COLOURS)] for k in channels]
        for patch, colour in zip(parts["boxes"], [*colours, style.INK_MUTED], strict=True):
            patch.set_facecolor(style.tint(colour, 0.45))
            patch.set_edgecolor(style.INK_SECONDARY)
        for median in parts["medians"]:
            median.set_color(style.INK)
        axes.set_ylabel(run.score_label)
        axes.grid(True, axis="y")
    style.caption(
        figure,
        "Left: each class's particles by tilt-series, with the share of that tilt-series' "
        "particles the class holds. Right: the picking scores of each class's particles "
        "(box: middle half, line: median, whiskers: 1.5 box lengths). A class of picks "
        "with lower scores than the rest is a candidate for false picks; similar scores "
        "say nothing either way."
        + (" Decoys are left out: their scores are copies of real picks' scores."
           if run.decoys else ""),
        y=0.2,
    )
    return figure


def class_gallery_page(run: ClassificationRun, channels: Sequence[int]) -> Figure:
    last = run.iterations[-1]
    figure = style.new_page(
        f"Particles of each class after iteration {last.number}",
        f"Up to {CLASS_GALLERY_TILES} per class, chosen at random, at the class's own poses",
    )
    rows: list[tuple[int, list[tuple[int, np.ndarray]]]] = []
    for channel in channels:
        members = last.classes[channel].members
        tags = sample_tags(members, CLASS_GALLERY_TILES)
        rows.append((channel, [
            (int(tag), gallery_tile(run.aligned_particle(int(tag), (last.number, channel))))
            for tag in tags
        ]))
    limits = display_range([tile for _, tiles in rows for _, tile in tiles] or [np.zeros(1)])
    total = len(run.tags)
    for row, (channel, tiles) in enumerate(rows):
        top = _row_top(row)
        count = len(last.classes[channel].members)
        figure.text(0.035, top, f"class {channel} · {_particles(count)} "
                    f"({100 * count / total:.0f}%)", fontsize=style.FONT_TEXT,
                    color=CHANNEL_COLOURS[(channel - 1) % len(CHANNEL_COLOURS)], va="top")
        _draw_tile_row(figure, run, tiles, limits, top=top - 0.055, height=0.105)
    style.caption(
        figure,
        f"Each particle at its pose in its class's channel, then the mean of its "
        f"{GALLERY_SLAB_PX} central z sections, smoothed by one voxel. One grey scale for "
        "the page.",
    )
    return figure


def problems_page(lines: Sequence[str]) -> Figure:
    figure = style.new_page("What could not be read", "The rest of the report is unaffected")
    text = "\n".join(textwrap.fill(line, 140) for line in lines)
    figure.text(0.035, style.CONTENT_TOP, text, fontsize=style.FONT_SMALL, va="top",
                color=style.INK_SECONDARY, linespacing=1.35)
    return figure


def _error_page(label: str, error: BaseException) -> Figure:
    figure = style.new_page(f"{label} · could not be drawn",
                            "The rest of the report is unaffected")
    message = textwrap.fill(f"{type(error).__name__}: {error}", 125)
    figure.text(0.035, style.CONTENT_TOP, message, fontsize=style.FONT_TEXT, va="top",
                color=style.INK_SECONDARY)
    return figure


# ---------------------------------------------------------------------------
# Pictures


def z_sections(volume: np.ndarray, count: int = SECTIONS,
               step: int = SECTION_STEP_PX) -> list[tuple[int, np.ndarray]]:
    """Successive z sections about the box centre, `(index, section)` with 0-based z.

    The centre is `(N - 1) / 2`, where the export and Dynamo's averages put a
    particle; an even box has no voxel there, so the sections sit symmetrically
    about it, half a voxel off the nearest ones.
    """

    depth = volume.shape[0]
    centre = (depth - 1) / 2
    offsets = step * (np.arange(count) - (count - 1) / 2)
    indices = sorted({int(np.clip(math.floor(centre + offset + 0.5), 0, depth - 1))
                      for offset in offsets})
    return [(index, np.asarray(volume[index], dtype=float)) for index in indices]


def gallery_tile(volume: np.ndarray, slices: int = GALLERY_SLAB_PX) -> np.ndarray:
    """The mean of a volume's `slices` central z sections, smoothed by one voxel."""

    depth = volume.shape[0]
    start = max(0, (depth - slices) // 2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        slab = np.nanmean(np.asarray(volume[start : start + slices], dtype=float), axis=0)
    return smooth(slab)


def sample_tags(tags: np.ndarray, count: int, seed: int = GALLERY_SEED) -> np.ndarray:
    """Up to `count` tags at random, the same ones for the same tags, in ascending order."""

    tags = np.asarray(tags)
    if len(tags) <= count:
        return np.sort(tags)
    chosen = np.random.default_rng(seed).choice(len(tags), size=count, replace=False)
    return np.sort(tags[chosen])


def class_changes(iterations: Sequence[IterationResult]) -> list[int | None]:
    """How many particles changed class since the iteration before; None for the first."""

    changes: list[int | None] = []
    previous: dict[int, int] | None = None
    for iteration in iterations:
        current = {int(tag): channel for channel, result in iteration.classes.items()
                   for tag in result.members}
        if previous is None:
            changes.append(None)
        else:
            changes.append(sum(1 for tag, channel in current.items()
                               if previous.get(tag) != channel))
        previous = current
    return changes


def central_views(volume: np.ndarray) -> list[np.ndarray]:
    """XY, XZ and YZ sections through `(N - 1) / 2`: the middle voxel of an odd box, the
    mean of the two about it in an even one. Second axis horizontal."""

    views = []
    for axis in range(3):
        size = volume.shape[axis]
        middle = sorted({(size - 1) // 2, size // 2})
        views.append(np.mean(np.take(volume, middle, axis=axis), axis=axis))
    return views


ROW_STEP = 0.215
SECTION_COLUMNS = 5
MAP_ROW_GUIDE = (
    "Left: the central section along each axis. Right: successive z sections through the "
    "box centre, as Dynamo's dslices shows a map, left to right and top to bottom, each "
    "marked with its height above the centre in nm. One grey scale per map."
)


def _row_top(row: int) -> float:
    return style.CONTENT_TOP - row * ROW_STEP


def _map_row(figure: Figure, run: ClassificationRun, row: int, label: str,
             volume: np.ndarray, colour: str) -> None:
    """One map: its central XY, XZ and YZ sections, then its z sections in rows of five.

    One grey scale for all of them, so a section and a central view compare.
    """

    top = _row_top(row)
    figure.text(0.035, top, label, fontsize=style.FONT_TEXT, color=colour, va="top")
    sections = z_sections(volume)
    views = central_views(volume)
    limits = display_range([*views, *(section for _, section in sections)])
    aspect = style.PAGE_SIZE[1] / style.PAGE_SIZE[0]
    image_top = top - 0.03
    height = ROW_STEP - 0.045
    x = 0.035
    for index, (view, name) in enumerate(zip(views, ("XY", "XZ", "YZ"), strict=True)):
        axes = figure.add_axes((x, image_top - height, height * aspect, height))
        _show(axes, view, limits)
        _corner_label(axes, name)
        if index == 0:
            _scale_bar(axes, run.pixel_size_angstrom, view.shape)
        x += height * aspect + 0.008
    x += 0.02
    gap = 0.006
    rows = max(1, math.ceil(len(sections) / SECTION_COLUMNS))
    small = (height - gap * (rows - 1)) / rows
    centre = (volume.shape[0] - 1) / 2
    for index, (z, section) in enumerate(sections):
        line, column = divmod(index, SECTION_COLUMNS)
        axes = figure.add_axes((x + column * (small * aspect + gap),
                                image_top - (line + 1) * small - line * gap,
                                small * aspect, small))
        _show(axes, section, limits)
        _corner_label(axes, _offset_label(z - centre, run.pixel_size_angstrom, unit=False))


def _show(axes: Axes, image: np.ndarray, limits: tuple[float, float]) -> None:
    axes.imshow(image, cmap=style.GRAY_WITH_GAPS, vmin=limits[0], vmax=limits[1],
                origin="lower")
    style.image_axes(axes)
    axes.set_aspect("equal")


def _corner_label(axes: Axes, text: str) -> None:
    """A label inside a picture's top left corner, white on a dark edge, as Dynamo's dslices
    numbers its sections."""

    label = axes.text(0.05, 0.95, text, transform=axes.transAxes, ha="left", va="top",
                      color="white", fontsize=style.FONT_TINY - 1)
    label.set_path_effects([patheffects.withStroke(linewidth=1.6, foreground=style.INK)])


def _draw_sections(figure: Figure, volume: np.ndarray, pixel_size: float | None, *,
                   x0: float, y0: float, width: float, height: float) -> None:
    """A run of z sections side by side in one row, on one grey scale."""

    sections = z_sections(volume)
    limits = display_range([section for _, section in sections])
    centre = (volume.shape[0] - 1) / 2
    gap = 0.006
    panel = (width - gap * (len(sections) - 1)) / len(sections)
    side = min(panel, height * style.PAGE_SIZE[1] / style.PAGE_SIZE[0])
    for index, (z, section) in enumerate(sections):
        axes = figure.add_axes((x0 + index * (panel + gap), y0, side, height))
        axes.imshow(section, cmap=style.GRAY_WITH_GAPS, vmin=limits[0], vmax=limits[1],
                    origin="lower")
        style.image_axes(axes)
        axes.set_aspect("equal")
        axes.set_title(_offset_label(z - centre, pixel_size), fontsize=style.FONT_TINY,
                       color=style.INK_SECONDARY)
        if index == 0:
            _scale_bar(axes, pixel_size, section.shape)


def _particles(count: int) -> str:
    return f"{count:,} particle" + ("" if count == 1 else "s")


def _offset_label(offset_px: float, pixel_size: float | None, *, unit: bool = True) -> str:
    """A section's height above the box centre: in nm when the pixel size is known."""

    if pixel_size:
        value = f"{offset_px * pixel_size / 10:+.1f}"
        return f"z {value} nm" if unit else value
    return f"z {offset_px:+.1f} px" if unit else f"{offset_px:+.1f} px"


def _draw_tiles(figure: Figure, run: ClassificationRun, tiles: list[tuple[int, np.ndarray]], *,
                top: float, bottom: float, columns: int) -> None:
    limits = display_range([tile for _, tile in tiles] or [np.zeros(1)])
    rows = max(1, math.ceil(len(tiles) / columns))
    row_height = (top - bottom) / rows
    for row in range(rows):
        _draw_tile_row(figure, run, tiles[row * columns : (row + 1) * columns], limits,
                       top=top - row * row_height, height=row_height - 0.04,
                       scale_bar=row == 0)


def _draw_tile_row(figure: Figure, run: ClassificationRun, tiles: list[tuple[int, np.ndarray]],
                   limits: tuple[float, float], *, top: float, height: float,
                   scale_bar: bool = True) -> None:
    width = 0.93 / GALLERY_COLUMNS
    side_height = height
    side_width = min(width - 0.008, side_height * style.PAGE_SIZE[1] / style.PAGE_SIZE[0])
    for column, (tag, tile) in enumerate(tiles):
        axes = figure.add_axes((0.035 + column * width, top - side_height, side_width,
                                side_height))
        axes.imshow(tile, cmap=style.GRAY_WITH_GAPS, vmin=limits[0], vmax=limits[1],
                    origin="lower")
        style.image_axes(axes)
        axes.set_aspect("equal")
        axes.set_title(_tile_label(run, tag), fontsize=style.FONT_TINY,
                       color=style.INK_SECONDARY, pad=2)
        if column == 0 and scale_bar:
            _scale_bar(axes, run.pixel_size_angstrom, tile.shape)


def _tile_label(run: ClassificationRun, tag: int) -> str:
    """Tag and score on one line, the tilt-series under them: one line would not fit a tile."""

    if tag in run.decoys:
        return f"#{tag}\ndecoy" + (f" · {run.series[tag]}" if tag in run.series else "")
    first = f"#{tag}" + (f" · {run.scores[tag]:.3f}" if tag in run.scores else "")
    return first + (f"\n{run.series[tag]}" if tag in run.series else "")


def _scale_bar(axes: Axes, pixel_size: float | None, shape: tuple[int, ...]) -> None:
    if not pixel_size:
        return
    height, width = shape[0], shape[1]
    length_nm = nice_length(0.4 * width * pixel_size / 10)
    if length_nm <= 0:
        return
    label = f"{length_nm:g} nm"
    style.scale_bar(axes, length_nm * 10 / pixel_size, label, image_width_px=width,
                    image_height_px=height)
