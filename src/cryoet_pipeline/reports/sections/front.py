"""The pages around the series: title, contents, the table of all series, warnings, appendix."""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import numpy as np
from matplotlib.figure import Figure

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import (
    QC_STEPS,
    SeriesReport,
    SummaryRow,
    qc_step,
)

TABLE_ROWS_PER_PAGE = 18
CONTENTS_ROWS_PER_COLUMN = 24
WARNING_LINES_PER_PAGE = 32
CONFIG_LINES_PER_PAGE = 46

HOW_TO_READ = (
    "Each tilt-series has its own pages, listed in the contents: the tilt alignment, the "
    "aligned tilts, the CTF fit, the tomogram, its membranes if they were segmented, the "
    "picks on the tomogram, their scores, the correlation map, and galleries of picks by "
    "score.",
    "Grey pictures show protein dark, as in 3dmod. They are averages of slabs about 10 nm "
    "thick, lightly smoothed, because a single slice at this binning is mostly noise. z is "
    "the height above the bottom of the tomogram.",
    "Scores are in σ: how many standard deviations of the search's own noise a peak stands "
    "above zero. The one-false-positive height is the score that noise alone reaches about "
    "once in the whole search, so a pick above it is unlikely to be noise on its own. Below "
    "it, picks are evidence only as a group: through how many more of them there are than "
    "noise alone would give.",
    "The galleries split the picks into thirds by score, and add random places at the same "
    "depths. If the low third looks like the random places, those picks are probably not "
    "particles.",
    "PASS, WARN and FAIL are the verdicts each step recorded about its own output. This "
    "report adds none of its own: it only reads the project's records and the files they "
    "name, and changes nothing.",
)


def title_page(
    rows: Sequence[SummaryRow], *, project: str, variant: str | None, written: str,
    software: str,
) -> Figure:
    variant_text = f"variant {variant}" if variant else "no variant label"
    figure = style.new_page(
        f"Picking report · {project}",
        f"{variant_text} · {len(rows)} tilt-series · written {written}",
    )
    with_picks = [row for row in rows if row.picks is not None]
    above = [row.above_floor for row in with_picks if row.above_floor is not None]
    tiles = (
        (str(len(rows)), "tilt-series"),
        (str(len(with_picks)), "with picks"),
        (f"{sum(row.picks or 0 for row in with_picks):,}", "picks in total"),
        (f"{sum(above):,}" if above else "–", "above the one-false-positive height"),
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
    figure.text(0.035, 0.05, f"Written by {software}.", fontsize=style.FONT_SMALL,
                color=style.INK_SECONDARY)
    return figure


def contents_pages(
    rows: Sequence[SummaryRow], contents: Mapping[str, int]
) -> list[Callable[[], Figure]]:
    """Contents builders; they read `contents` when drawn, after every page has a number."""

    per_page = 2 * CONTENTS_ROWS_PER_COLUMN
    return [
        partial(_contents_page, list(rows[start : start + per_page]), contents)
        for start in range(0, max(1, len(rows)), per_page)
    ]


def _contents_page(rows: list[SummaryRow], contents: Mapping[str, int]) -> Figure:
    figure = style.new_page("Contents", "Where each tilt-series' pages start")
    for index, row in enumerate(rows):
        column, line = divmod(index, CONTENTS_ROWS_PER_COLUMN)
        x = 0.035 + column * 0.47
        y = style.CONTENT_TOP - line * 0.03
        figure.text(x, y, row.series, fontsize=style.FONT_TEXT, fontweight="bold", va="top")
        figure.text(x + 0.14, y, _contents_note(row), fontsize=style.FONT_SMALL,
                    color=style.INK_SECONDARY, va="top")
        figure.text(x + 0.43, y, str(contents.get(row.series, "")), fontsize=style.FONT_TEXT,
                    ha="right", va="top")
    return figure


def _contents_note(row: SummaryRow) -> str:
    if row.picks is None:
        return "no picks"
    note = f"{row.picks:,} picks"
    if row.above_floor is not None:
        note += f", {row.above_floor:,} above the height"
    return note


def summary_table_pages(rows: Sequence[SummaryRow]) -> list[Callable[[], Figure]]:
    present = {qc_type for row in rows for qc_type in row.statuses}
    known = [step.qc_type for step in QC_STEPS if step.qc_type in present]
    steps = known + sorted(present - set(known))
    return [
        partial(_summary_table, list(rows[start : start + TABLE_ROWS_PER_PAGE]), steps)
        for start in range(0, max(1, len(rows)), TABLE_ROWS_PER_PAGE)
    ]


_TABLE_COLUMNS = (
    "Series",
    "Tilts\nused / all",
    "Tilt range\n(°)",
    "Defocus\n(µm)",
    "CTF fit\nat 0° (Å)",
    "Alignment\nresidual (nm)",
    "Membrane\n(% volume)",
    "Picks",
    "Above\nthe height",
    "Best\n(σ)",
)


def _summary_table(rows: list[SummaryRow], steps: list[str]) -> Figure:
    figure = style.new_page("All tilt-series", "One row per tilt-series; the verdict columns are "
                            "each step's own QC")
    labels = list(_TABLE_COLUMNS) + [qc_step(step).short for step in steps]
    text: list[list[str]] = []
    colours: list[list[str | tuple[float, float, float]]] = []
    for row in rows:
        values = [
            row.series,
            _views(row),
            f"{row.tilt_range_deg[0]:+.0f} to {row.tilt_range_deg[1]:+.0f}"
            if row.tilt_range_deg else "–",
            _number(row.mean_defocus_um, "{:.2f}"),
            _number(row.ctf_fit_near_zero_angstrom, "{:.1f}"),
            _number(row.residual_nm, "{:.2f}"),
            _number(row.membrane_percent, "{:.1f}"),
            _number(row.picks, "{:,}"),
            _number(row.above_floor, "{:,}"),
            _number(row.best_sigma, "{:.1f}"),
        ]
        cells: list[str | tuple[float, float, float]] = ["white"] * len(values)
        for step in steps:
            status = row.statuses.get(step)
            if status is None:
                values.append("")
                cells.append("white")
            else:
                word, background = style.status_cell(status)
                values.append(word)
                cells.append(background)
        text.append(values)
        colours.append(cells)

    axes = figure.add_axes((0.035, 0.24, 0.93, style.CONTENT_TOP - 0.24))
    axes.axis("off")
    table = axes.table(
        cellText=text or [[""] * len(labels)],
        colLabels=labels,
        cellColours=colours or None,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(style.FONT_SMALL)
    table.scale(1.0, 1.55)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_text_props(fontweight="bold", color=style.INK_SECONDARY)
            cell.set_height(cell.get_height() * 1.8)
    table.auto_set_column_width(list(range(len(labels))))

    legend = "; ".join(f"{qc_step(step).short}: {qc_step(step).label.lower()}" for step in steps)
    style.caption(
        figure,
        "Tilts: used in the tomogram / collected. CTF fit at 0°: how far CTFFIND fitted Thon "
        "rings on the tilt nearest 0°. Alignment residual: mean fine-alignment residual. Above "
        "the height: picks above the one-false-positive height. "
        + (f"Verdict columns: {legend}." if legend else ""),
        y=0.2,
    )
    return figure


def _views(row: SummaryRow) -> str:
    if row.views_used is None:
        return f"– / {row.views_total}"
    return f"{row.views_used} / {row.views_total}"


def _number(value: float | int | None, pattern: str) -> str:
    return pattern.format(value) if value is not None else "–"


def picks_per_tomogram_page(rows: Sequence[SummaryRow]) -> Figure:
    shown = [row for row in rows if row.picks is not None]
    figure = style.new_page("Picks per tomogram",
                            f"{len(shown)} tilt-series with picks")
    # About a third of an inch per bar, so a few bars do not swell to fill the page.
    height = min(0.7, 0.04 * len(shown) + 0.04)
    axes = figure.add_axes((0.17, 0.84 - height, 0.76, height))
    positions = np.arange(len(shown))[::-1]
    totals = np.asarray([row.picks or 0 for row in shown])
    axes.barh(positions, totals, height=0.7, color=style.NEUTRAL_FILL, label="all picks")
    above = [row.above_floor for row in shown]
    if any(value is not None for value in above):
        axes.barh(
            positions,
            [value or 0 for value in above],
            height=0.7,
            color=style.SERIES_BLUE,
            label="above the one-false-positive height",
        )
        axes.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2)
    axes.set_yticks(positions, labels=[row.series for row in shown],
                    fontsize=style.FONT_SMALL if len(shown) <= 30 else style.FONT_TINY - 2)
    if len(shown) <= 25:
        for position, total in zip(positions, totals, strict=True):
            axes.annotate(f"{total:,}", (total, position), xytext=(3, 0),
                          textcoords="offset points", va="center", fontsize=style.FONT_TINY,
                          color=style.INK_SECONDARY)
    axes.set_xlabel("Picks")
    axes.grid(True, axis="x")
    axes.set_axisbelow(True)
    style.caption(
        figure,
        "Grey: every pick kept. Blue: the part of them above the one-false-positive height, "
        "each unlikely to be noise on its own.",
    )
    return figure


def warning_lines(reports: Sequence[SeriesReport]) -> list[tuple[str, bool]]:
    """Every warning and loading problem, as (line, is_heading), wrapped for the page."""

    lines: list[tuple[str, bool]] = []
    for report in reports:
        entries: list[str] = []
        ordered = [step.qc_type for step in QC_STEPS if step.qc_type in report.qc]
        ordered += sorted(set(report.qc) - set(ordered))
        for qc_type in ordered:
            record = report.qc[qc_type]
            label = qc_step(qc_type).label
            for warning in record.warnings:
                entries.append(f"{label}: {warning}")
            if record.status == "fail" and not record.warnings:
                entries.append(f"{label}: failed, with no reason recorded")
        entries.extend(f"Not shown: {problem}" for problem in report.problems)
        if not entries:
            continue
        lines.append((report.series, True))
        for entry in entries:
            wrapped = textwrap.wrap(entry, 140)
            lines.append(("• " + wrapped[0], False))
            lines.extend(("   " + rest, False) for rest in wrapped[1:])
        lines.append(("", False))
    return lines


def warning_pages(reports: Sequence[SeriesReport]) -> list[Callable[[], Figure]]:
    lines = warning_lines(reports) or [
        ("No step recorded a warning, and every part of every tilt-series could be read.", False)
    ]
    return [
        partial(_warning_page, lines[start : start + WARNING_LINES_PER_PAGE])
        for start in range(0, len(lines), WARNING_LINES_PER_PAGE)
    ]


def _warning_page(lines: list[tuple[str, bool]]) -> Figure:
    figure = style.new_page(
        "Warnings and problems",
        "What each step's QC warned about, and what this report could not read",
    )
    y = style.CONTENT_TOP
    for text, heading in lines:
        if heading and y < style.CONTENT_TOP:
            y -= 0.008
        figure.text(0.035, y, text, fontsize=style.FONT_TEXT if heading else style.FONT_SMALL,
                    fontweight="bold" if heading else "normal", va="top",
                    color=style.INK if heading else style.INK_SECONDARY)
        y -= 0.0235
    return figure


def artifact_pages(reports: Sequence[SeriesReport]) -> list[Callable[[], Figure]]:
    return [
        partial(_artifact_table, list(reports[start : start + TABLE_ROWS_PER_PAGE]))
        for start in range(0, max(1, len(reports)), TABLE_ROWS_PER_PAGE)
    ]


def _artifact_table(reports: list[SeriesReport]) -> Figure:
    figure = style.new_page(
        "Appendix · What each page was drawn from",
        "Registry ids, without the tilt-series prefix",
    )
    labels = ["Series", "Picks", "Tomogram", "CTF model", "Membranes", "Template"]
    text = []
    for report in reports:
        artifacts = report.artifacts
        row = [report.series]
        for artifact in (artifacts.picks, artifacts.tomogram, artifacts.ctf_model,
                         artifacts.segmentation, artifacts.template):
            row.append(_short_id(artifact.id, report.series) if artifact is not None else "–")
        text.append(row)
    axes = figure.add_axes((0.035, 0.1, 0.93, style.CONTENT_TOP - 0.1))
    axes.axis("off")
    table = axes.table(cellText=text or [[""] * len(labels)], colLabels=labels,
                       cellLoc="left", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(style.FONT_TINY)
    table.scale(1.0, 1.5)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor(style.GRID)
        cell.set_linewidth(0.5)
        if row_index == 0:
            cell.set_text_props(fontweight="bold", color=style.INK_SECONDARY)
    table.auto_set_column_width(list(range(len(labels))))
    return figure


def _short_id(artifact_id: str, series: str) -> str:
    prefix = f"{series}:"
    return artifact_id[len(prefix):] if artifact_id.startswith(prefix) else artifact_id


def config_pages(name: str, text: str) -> list[Callable[[], Figure]]:
    lines = text.splitlines() or [""]
    return [
        partial(_config_page, name, lines[start : start + CONFIG_LINES_PER_PAGE])
        for start in range(0, len(lines), CONFIG_LINES_PER_PAGE)
    ]


def _config_page(name: str, lines: list[str]) -> Figure:
    figure = style.new_page(f"Appendix · Lane configuration · {name}",
                            "As given to the report, verbatim")
    figure.text(0.035, style.CONTENT_TOP, "\n".join(line[:150] for line in lines),
                family="DejaVu Sans Mono", fontsize=style.FONT_TINY, va="top", linespacing=1.3)
    return figure


def error_page(label: str, error: BaseException) -> Figure:
    figure = style.new_page(f"{label} · could not be drawn",
                            "The rest of the report is unaffected")
    message = textwrap.fill(f"{type(error).__name__}: {error}", 125)
    figure.text(0.035, style.CONTENT_TOP, message, fontsize=style.FONT_TEXT, va="top",
                color=style.INK_SECONDARY)
    return figure


def empty_series_page(report: SeriesReport) -> Figure:
    figure = style.new_page(report.series, "Nothing to draw for this tilt-series yet")
    lines = [textwrap.fill(problem, 125) for problem in report.problems] or [
        "The registry holds no tomogram, CTF fit or picks for it."
    ]
    figure.text(0.035, style.CONTENT_TOP, "\n\n".join(lines), fontsize=style.FONT_TEXT, va="top",
                color=style.INK_SECONDARY)
    return figure
