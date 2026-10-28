"""The tilt alignment: how well the tracked points fit, tilt by tilt, and the aligned tilts."""

from __future__ import annotations

import textwrap

import mrcfile  # type: ignore[import-untyped]
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

from cryoet_pipeline.reports import style, timing
from cryoet_pipeline.reports.collect import AlignmentInfo, SeriesReport, qc_step
from cryoet_pipeline.reports.imaging import block_mean, display_range, reduction_factor, smooth
from cryoet_pipeline.reports.sections.tomogram import add_scale_bar, show

SHOWN_TILTS = (-60.0, -30.0, 0.0, 30.0, 60.0, -45.0, 45.0)
"""Tilts shown on the aligned-tilts page, nearest first; the worst residual is added."""

ALIGNMENT_STEPS = ("coarse_alignment_report", "fiducial_seed", "fiducial_tracking",
                   "fine_alignment")

TIMED_BY_QC = {
    "coarse_alignment_report": ("align-tilt-series", "qc-coarse-alignment"),
    "fiducial_seed": ("generate-fiducial-seed",),
    "fiducial_tracking": ("track-fiducials",),
    "fine_alignment": ("fine-align-tilt-series",),
}
"""The steps whose time goes on each verdict line: the coarse alignment with its check."""


def alignment_page(report: SeriesReport) -> Figure:
    alignment = report.alignment
    assert alignment is not None
    figure = style.new_page(f"{report.series} · Tilt alignment", _subtitle(report, alignment))
    grid = GridSpec(
        2, 3, figure=figure, left=0.065, right=0.975, top=style.CONTENT_TOP,
        bottom=style.CONTENT_BOTTOM, width_ratios=[1.0, 1.0, 0.95], hspace=0.5, wspace=0.3,
    )
    _residual_panel(figure.add_subplot(grid[0, 0:2]), report, alignment)
    _points_panel(figure.add_subplot(grid[1, 0:2]), alignment)
    _tracked_panel(figure.add_subplot(grid[0, 2]), alignment)
    _numbers_panel(figure.add_subplot(grid[1, 2]), report, alignment)
    things = "beads" if alignment.method == "gold beads" else "points"
    style.caption(
        figure,
        "The residual is the distance between where a tracked point was found on a tilt and "
        "where the solved alignment puts it, averaged over that tilt's points. Small and even "
        "across tilts is good; a tilt that sticks out is aligned worse than its neighbours. "
        f"Fewer {things} at high tilt is normal; a tilt with few of them is held in place by "
        "few measurements.",
    )
    return figure


def _subtitle(report: SeriesReport, alignment: AlignmentInfo) -> str:
    parts = [alignment.method]
    if alignment.fiducial_count is not None:
        noun = "beads" if alignment.method == "gold beads" else "patches"
        parts.append(f"{alignment.fiducial_count} {noun} tracked")
    total = len(report.artifacts.manifest.images)
    parts.append(f"{alignment.z_value.size} of {total} tilts aligned")
    if alignment.tilt_axis_angle_deg is not None:
        parts.append(f"tilt axis at {alignment.tilt_axis_angle_deg:.1f}°")
    document = _document(report, "fine_alignment")
    mean = document.get("residual_mean_nm")
    if isinstance(mean, (int, float)):
        parts.append(f"mean residual {mean:.2f} nm")
    took = timing.took(report.timing.page("alignment"))
    if took is not None:
        parts.append(took)
    return " · ".join(parts)


def _document(report: SeriesReport, qc_type: str) -> dict[str, object]:
    record = report.qc.get(qc_type)
    return record.document if record is not None else {}


def _residual_panel(axes: Axes, report: SeriesReport, alignment: AlignmentInfo) -> None:
    axes.set_title("Residual on each tilt", loc="left")
    axes.set_xlabel("Tilt angle (°)")
    if alignment.residual_nm is None:
        axes.text(0.5, 0.5, "tiltalign's per-tilt table could not be read",
                  transform=axes.transAxes, ha="center", color=style.INK_SECONDARY)
        axes.set_yticks([])
        return
    angles = alignment.tilt_angle_deg
    axes.plot(angles, alignment.residual_nm, color=style.SERIES_BLUE, linewidth=1.0,
              marker="o", markersize=5)
    if report.tomogram is not None:
        pixel_nm = report.tomogram.voxel_spacing_angstrom / 10.0
        if pixel_nm < 3.0 * float(np.max(alignment.residual_nm)):
            axes.axhline(pixel_nm, color=style.INK_MUTED, linewidth=0.8)
            axes.annotate(f"one tomogram pixel ({pixel_nm:.2f} nm)", (1.0, pixel_nm),
                          xycoords=("axes fraction", "data"), ha="right", va="bottom",
                          fontsize=style.FONT_TINY, color=style.INK_SECONDARY)
    excluded = _excluded_angles(report, alignment)
    if excluded:
        axes.scatter(excluded, [0.0] * len(excluded), marker="x", s=30,
                     color=style.INK_MUTED, zorder=3, clip_on=False)
        axes.annotate("× left out", (excluded[0], 0.0), xytext=(6, 4),
                      textcoords="offset points", fontsize=style.FONT_TINY,
                      color=style.INK_SECONDARY)
    axes.set_ylim(bottom=0.0)
    axes.set_ylabel("Mean residual (nm)")
    axes.grid(True, axis="y")


def _excluded_angles(report: SeriesReport, alignment: AlignmentInfo) -> list[float]:
    angle_by_z = {image.z_value: image.tilt_angle_deg
                  for image in report.artifacts.manifest.images}
    return [angle_by_z[z] for z in alignment.excluded_z_values if z in angle_by_z]


def _points_panel(axes: Axes, alignment: AlignmentInfo) -> None:
    beads = alignment.method == "gold beads"
    axes.set_title("Beads found on each tilt" if beads else "Tracked points on each tilt",
                   loc="left")
    axes.set_xlabel("Tilt angle (°)")
    if alignment.points_per_view is None:
        axes.text(0.5, 0.5, "no residual file, so no count per tilt",
                  transform=axes.transAxes, ha="center", color=style.INK_SECONDARY)
        axes.set_yticks([])
        return
    axes.plot(alignment.tilt_angle_deg, alignment.points_per_view, color=style.SERIES_BLUE,
              linewidth=1.0, marker="o", markersize=5)
    if alignment.fiducial_count is not None:
        axes.axhline(alignment.fiducial_count, color=style.INK_MUTED, linewidth=0.8)
        noun = "beads" if beads else "patches"
        axes.annotate(f"all {alignment.fiducial_count} {noun} in the model",
                      (1.0, alignment.fiducial_count), xycoords=("axes fraction", "data"),
                      ha="right", va="bottom", fontsize=style.FONT_TINY,
                      color=style.INK_SECONDARY)
    axes.set_ylim(bottom=0.0)
    axes.set_ylabel("Points used in the fit")
    axes.grid(True, axis="y")


def _tracked_panel(axes: Axes, alignment: AlignmentInfo) -> None:
    view = alignment.nearest_view(0.0)
    angle = alignment.tilt_angle_deg[view]
    if alignment.tracking_stack is None or alignment.point_xy is None:
        style.image_axes(axes, "Tracked points")
        axes.text(0.5, 0.5, "the tracking stack is not on disk", transform=axes.transAxes,
                  ha="center", color=style.INK_SECONDARY)
        return
    with mrcfile.mmap(alignment.tracking_stack, mode="r", permissive=True) as handle:
        image = smooth(np.asarray(handle.data[view], dtype=np.float32))
    factor = reduction_factor(image.shape)
    reduced = block_mean(image, factor)
    show(axes, reduced, factor, display_range([reduced]))
    assert alignment.point_view is not None
    here = alignment.point_view == view
    radius = 0.02 * max(image.shape)
    rings = [Circle((float(x), float(y)), radius) for x, y in alignment.point_xy[here]]
    axes.add_collection(PatchCollection(rings, facecolor="none",
                                        edgecolor=style.FIDUCIAL_COLOUR, linewidth=1.0))
    style.image_axes(axes, f"{int(here.sum())} points on the tilt at {angle:+.1f}°")
    if alignment.tracking_pixel_angstrom is not None:
        add_scale_bar(axes, alignment.tracking_pixel_angstrom, image.shape[1], image.shape[0])


def _numbers_panel(axes: Axes, report: SeriesReport, alignment: AlignmentInfo) -> None:
    axes.axis("off")
    axes.set_title("In numbers", loc="left")
    fine = _document(report, "fine_alignment")
    lines = []
    mean, sd = fine.get("residual_mean_nm"), fine.get("residual_sd_nm")
    if isinstance(mean, (int, float)) and isinstance(sd, (int, float)):
        lines.append(f"Mean residual {mean:.2f} ± {sd:.2f} nm")
    leave_out = fine.get("global_leave_out_error_nm")
    if isinstance(leave_out, (int, float)):
        lines.append(f"Leave-one-out error {leave_out:.2f} nm")
    count, pruned = fine.get("projection_point_count"), fine.get("pruned_point_count")
    if isinstance(count, int):
        text = f"Points in the fit: {count}"
        if isinstance(pruned, int) and pruned:
            text += f" ({pruned} dropped)"
        lines.append(text)
    offset = fine.get("applied_tilt_angle_offset_deg")
    if isinstance(offset, (int, float)):
        lines.append(f"Tilt angles shifted by {offset:+.2f}°")
    x_tilt = fine.get("recommended_x_axis_tilt_deg")
    if isinstance(x_tilt, (int, float)):
        lines.append(f"X-axis tilt {x_tilt:+.2f}°")
    excluded = _excluded_angles(report, alignment)
    if excluded:
        lines.append("Left out: " + ", ".join(f"{angle:+.0f}°" for angle in excluded))
    no_rings = _left_out_for_ctf(alignment)
    if no_rings:
        lines.append("Left out, no Thon rings: "
                     + ", ".join(f"{angle:+.0f}°" for angle in no_rings))
    lines.append("")
    steps = report.timing.page("alignment")
    for qc_type in ALIGNMENT_STEPS:
        record = report.qc.get(qc_type)
        # Each step's time on its verdict line, so that the panel grows no longer.
        took = timing.took(steps.of(*TIMED_BY_QC[qc_type]))
        items = []
        if record is not None:
            items.append(style.STATUS_LABELS.get(record.status, record.status))
        if took is not None:
            items.append(took)
        if not items:
            continue
        lines.append(f"{qc_step(qc_type).label}: " + " · ".join(items))
        for warning in record.warnings if record is not None else ():
            lines.extend("  " + line for line in textwrap.wrap(warning, 40))
    axes.text(0.0, 0.98, "\n".join(lines), transform=axes.transAxes, va="top",
              fontsize=style.FONT_SMALL, linespacing=1.45)


def aligned_tilts_page(report: SeriesReport) -> Figure:
    alignment = report.alignment
    assert alignment is not None and alignment.aligned_stack is not None
    views = _shown_views(alignment)
    worst = _worst_view(alignment, _stack_views(alignment))
    subtitle = "The aligned tilt series the tomogram was reconstructed from"
    if alignment.aligned_pixel_angstrom is not None:
        subtitle += f", {alignment.aligned_pixel_angstrom:.2f} Å per pixel"
    took = timing.took(report.timing.page("aligned tilts"))
    if took is not None:
        subtitle += f" · {took}"
    figure = style.new_page(f"{report.series} · Aligned tilts", subtitle)
    grid = GridSpec(2, 3, figure=figure, left=0.035, right=0.965, top=style.CONTENT_TOP,
                    bottom=style.PICTURE_BOTTOM, hspace=0.14, wspace=0.05)
    with mrcfile.mmap(alignment.aligned_stack, mode="r", permissive=True) as handle:
        stack = handle.data
        for number, view in enumerate(views):
            index = _slice_of(alignment, view, int(stack.shape[0]))
            image = smooth(np.asarray(stack[index], dtype=np.float32))
            factor = reduction_factor(image.shape)
            reduced = block_mean(image, factor)
            axes = figure.add_subplot(grid[divmod(number, 3)])
            show(axes, reduced, factor, display_range([reduced]))
            title = f"{alignment.tilt_angle_deg[view]:+.1f}°"
            if alignment.residual_nm is not None:
                title += f" · residual {alignment.residual_nm[view]:.2f} nm"
            if view == worst:
                title += " · the largest"
            style.image_axes(axes, title)
            if number == 0 and alignment.aligned_pixel_angstrom is not None:
                add_scale_bar(axes, alignment.aligned_pixel_angstrom, image.shape[1],
                              image.shape[0])
    style.caption(
        figure,
        "Each tilt on its own grey scale, protein dark. In the aligned series the tilt axis "
        "runs up and down, so from one tilt to the next a feature should only slide left or "
        "right, further the further it lies from the middle in z. A feature that jumps up or "
        "down between tilts, or a tilt much blurrier than its neighbours, points to that tilt.",
    )
    return figure


def _shown_views(alignment: AlignmentInfo) -> list[int]:
    """Six tilts: the nearest to each of the usual angles, and the worst residual.

    Only from the views the stack holds: correct-ctf leaves out views CTFFIND
    found no Thon rings on, and those are often the highest tilts, the ones the
    usual angles reach for first.
    """

    available = _stack_views(alignment)
    views: list[int] = []
    worst = _worst_view(alignment, available)
    for angle in SHOWN_TILTS:
        view = min(available, key=lambda one: abs(alignment.tilt_angle_deg[one] - angle))
        if view not in views:
            views.append(view)
        if len(views) == (5 if worst is not None else 6):
            break
    if worst is not None and worst not in views:
        views.append(worst)
    return sorted(views[:6], key=lambda view: alignment.tilt_angle_deg[view])


def _left_out_for_ctf(alignment: AlignmentInfo) -> list[float]:
    """Solved angles of the aligned views the reconstructed stack does not hold.

    correct-ctf leaves out the views CTFFIND found no Thon rings on; fine
    alignment's own exclusions are not aligned views, so they are not counted here.
    """

    if alignment.aligned_z_values is None:
        return []
    return [
        float(alignment.tilt_angle_deg[view])
        for view in range(alignment.z_value.size)
        if int(alignment.z_value[view]) not in alignment.aligned_z_values
    ]


def _stack_views(alignment: AlignmentInfo) -> list[int]:
    """The views of the alignment that have a slice in the aligned stack."""

    views = list(range(alignment.z_value.size))
    if alignment.aligned_z_values is None:
        return views
    held = [view for view in views if int(alignment.z_value[view]) in alignment.aligned_z_values]
    if not held:
        raise ValueError("the aligned stack holds none of the alignment's views")
    return held


def _worst_view(alignment: AlignmentInfo, views: list[int]) -> int | None:
    """The view with the largest residual among `views`, or None without residuals."""

    residuals = alignment.residual_nm
    if residuals is None:
        return None
    return max(views, key=lambda view: float(residuals[view]))


def _slice_of(alignment: AlignmentInfo, view: int, slices: int) -> int:
    """The slice of the aligned stack that holds a view of the alignment."""

    z_value = int(alignment.z_value[view])
    if alignment.aligned_z_values is not None:
        if z_value not in alignment.aligned_z_values:
            raise ValueError(f"the aligned stack has no slice for z value {z_value}")
        return alignment.aligned_z_values.index(z_value)
    if slices != alignment.z_value.size:
        raise ValueError(
            f"the aligned stack has {slices} slices for {alignment.z_value.size} aligned tilts"
        )
    return view
