"""Assemble a report: plan every page first, then draw them one at a time.

Planning first is what lets the contents carry page numbers: how many pages a
series gets depends only on what exists for it, so every number is known before
anything is drawn. Drawing one page at a time keeps one figure in memory, so a
fifty-series report costs no more memory than one page.

A page that fails to draw is replaced by a page saying why, and the failure is
returned: one odd series must not cost the other forty-nine their report.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import SeriesReport, summary_row
from cryoet_pipeline.reports.imaging import choose_depths, rank_bands, slab_half_width_px
from cryoet_pipeline.reports.sections import alignment, ctf, front, gallery, picking, tomogram


@dataclass(frozen=True)
class PlannedPage:
    label: str
    """Which page this is, for a failure message."""
    build: Callable[[], Figure]


@dataclass(frozen=True)
class ReportOutcome:
    path: Path
    pages: int
    problems: list[str]
    """Everything that could not be read or drawn, one line each."""


def write_report(
    path: Path,
    reports: Sequence[SeriesReport],
    *,
    project: str,
    variant: str | None,
    written: str,
    software: str,
    config: tuple[str, str] | None = None,
) -> ReportOutcome:
    """Write the whole report to `path`; `config` is a lane file's name and text to append."""

    rows = [summary_row(report) for report in reports]
    contents: dict[str, int] = {}
    front_matter = [
        PlannedPage(
            "title",
            lambda: front.title_page(
                rows, project=project, variant=variant, written=written, software=software
            ),
        ),
        *(PlannedPage("contents", build) for build in front.contents_pages(rows, contents)),
        *(PlannedPage("summary table", build) for build in front.summary_table_pages(rows)),
    ]
    # One bar is not a chart: with a single picked series its count is on the title page.
    if sum(row.picks is not None for row in rows) > 1:
        front_matter.append(
            PlannedPage("picks per tomogram", lambda: front.picks_per_tomogram_page(rows))
        )
    # Likewise one curve: a single picked series has it on its own picks page.
    if sum(report.picks is not None and report.picks.has_scores for report in reports) > 1:
        front_matter.extend(
            PlannedPage("scores by tomogram", build)
            for build in picking.score_overview_pages(reports, contents)
        )
    front_matter.extend(
        PlannedPage("warnings", build) for build in front.warning_pages(reports)
    )

    sections = [series_pages(report) for report in reports]
    number = len(front_matter) + 1
    for report, pages in zip(reports, sections, strict=True):
        contents[report.series] = number
        number += len(pages)

    back_matter = [PlannedPage("appendix", build) for build in front.artifact_pages(reports)]
    if config is not None:
        back_matter.extend(
            PlannedPage("configuration", build) for build in front.config_pages(*config)
        )

    planned = [*front_matter, *itertools.chain.from_iterable(sections), *back_matter]
    problems = [f"{report.series}: {problem}" for report in reports for problem in report.problems]
    footer = f"{project}" + (f" · {variant}" if variant else "") + f" · {written}"

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    metadata = {
        "Title": f"Picking report · {project}",
        "Subject": "cryo-ET processing and particle picking",
        "Creator": software,
    }
    with matplotlib.rc_context(cast(Any, style.RC)), PdfPages(partial, metadata=metadata) as pdf:
        for page_number, page in enumerate(planned, 1):
            try:
                figure = page.build()
            except Exception as error:  # noqa: BLE001 - one page must not lose the report
                problems.append(f"{page.label}: could not be drawn: {type(error).__name__}: "
                                f"{error}")
                figure = front.error_page(page.label, error)
            style.footer(figure, footer, page_number, len(planned))
            pdf.savefig(figure)
    partial.replace(path)
    return ReportOutcome(path, len(planned), problems)


def series_pages(report: SeriesReport) -> list[PlannedPage]:
    """The pages one series gets, in the pipeline's order, decided by what exists for it."""

    label = report.series
    picks = report.picks
    info = report.tomogram
    bands = rank_bands(picks.count, gallery.TILES_PER_BAND) if picks and picks.has_scores else []
    depths = (
        choose_depths(
            info.shape[0],
            picks.z_px if picks is not None and picks.count else None,
            report.correlation.preview if report.correlation is not None else None,
            slab_half_width_px(info.voxel_spacing_angstrom),
        )
        if info is not None
        else []
    )

    pages: list[PlannedPage] = []
    if report.alignment is not None:
        pages.append(PlannedPage(f"{label} · tilt alignment",
                                 lambda: alignment.alignment_page(report)))
        if report.alignment.aligned_stack is not None:
            pages.append(PlannedPage(f"{label} · aligned tilts",
                                     lambda: alignment.aligned_tilts_page(report)))
    if report.ctf is not None:
        pages.append(PlannedPage(f"{label} · CTF", lambda: ctf.ctf_page(report)))
    if info is not None:
        pages.append(PlannedPage(f"{label} · tomogram",
                                 lambda: tomogram.tomogram_page(report, depths)))
        if report.artifacts.segmentation is not None:
            pages.append(PlannedPage(f"{label} · membranes",
                                     lambda: tomogram.membranes_page(report, depths)))
    if picks is not None:
        if info is not None and picks.count:
            pages.append(PlannedPage(f"{label} · picks",
                                     lambda: picking.picks_page(report, depths)))
        pages.append(PlannedPage(f"{label} · pick scores",
                                 lambda: picking.scores_page(report, bands)))
        if info is not None and report.correlation is not None and picks.count:
            pages.append(PlannedPage(f"{label} · correlation map",
                                     lambda: picking.correlation_page(report, depths)))
        if info is not None and bands:
            cache: dict[str, gallery.Gallery] = {}

            def built() -> gallery.Gallery:
                if "gallery" not in cache:
                    cache["gallery"] = gallery.build_gallery(report, bands)
                return cache["gallery"]

            pages.append(PlannedPage(
                f"{label} · gallery",
                lambda: gallery.gallery_page(report, built(), ("high", "middle")),
            ))
            pages.append(PlannedPage(
                f"{label} · gallery",
                lambda: gallery.gallery_page(report, built(), ("low", "random")),
            ))
    if not pages:
        pages.append(PlannedPage(label, lambda: front.empty_series_page(report)))
    return pages
