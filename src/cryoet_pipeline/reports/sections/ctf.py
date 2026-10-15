"""Defocus and the reach of the CTF fit, per tilt, against tilt angle and dose."""

from __future__ import annotations

import textwrap
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import CtfTable, SeriesReport


def ctf_page(report: SeriesReport) -> Figure:
    ctf = report.ctf
    assert ctf is not None
    figure = style.new_page(f"{report.series} · CTF ({ctf.backend})", _subtitle(ctf))
    grid = GridSpec(
        2, 4, figure=figure, left=0.065, right=0.965, top=style.CONTENT_TOP,
        bottom=style.PICTURE_BOTTOM, height_ratios=[1.0, 0.9], hspace=0.42, wspace=0.3,
    )

    colours = _view_colours(ctf)
    defocus = figure.add_subplot(grid[0, 0:2])
    _defocus_plot(defocus, ctf, colours)
    fit = figure.add_subplot(grid[0, 2:4])
    _fit_plot(fit, ctf, colours)
    if ctf.dose is not None:
        colorbar = figure.colorbar(
            ScalarMappable(_dose_norm(ctf), style.MARKER_MAP),
            ax=[defocus, fit],
            fraction=0.025,
            pad=0.015,
        )
        style.colour_bar_style(colorbar, "Dose before the tilt (e⁻/Å²)")

    shown = _thumbnail_views(ctf)
    if ctf.diagnostic_path is not None:
        with mrcfile.mmap(ctf.diagnostic_path, mode="r", permissive=True) as handle:
            stack = handle.data
            for column, view in enumerate(shown):
                axes = figure.add_subplot(grid[1, column])
                image = np.asarray(stack[view] if stack.ndim == 3 else stack, dtype=np.float32)
                low, high = np.percentile(image, [1, 99])
                axes.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower")
                style.image_axes(axes, _thumbnail_title(ctf, view))
    text = figure.add_subplot(grid[1, 3])
    text.axis("off")
    text.text(0.0, 1.0, _numbers(report, ctf), va="top", fontsize=style.FONT_SMALL,
              linespacing=1.5)

    note = (
        "Each point is one tilt; its bar spans the two defocus values of an astigmatic fit. "
        "The fit limit is the finest spacing to which CTFFIND still found Thon rings: lower is "
        "better, and it normally worsens with tilt, as the sample gets thicker in projection, "
        "and with dose."
    )
    if ctf.dose is None:
        note += " The mdoc records no PriorRecordDose, so the tilts are not coloured by dose."
    if ctf.diagnostic_path is not None:
        note += " Bottom: CTFFIND's own diagnostic images for three of the tilts."
    style.caption(figure, note)
    return figure


def fit_sampling(ctf: CtfTable) -> str | None:
    """Which binning and pixel the CTF was fitted at, in words."""

    if ctf.pixel_spacing_angstrom is None:
        return None
    text = f"{ctf.pixel_spacing_angstrom:.2f} Å per pixel"
    if ctf.nyquist_angstrom is not None:
        text += f", Nyquist {ctf.nyquist_angstrom:.1f} Å"
    if ctf.fit_binning is not None:
        return f"fitted at bin {ctf.fit_binning} ({text})"
    return f"fitted at {text}"


def _subtitle(ctf: CtfTable) -> str:
    parts = []
    sampling = fit_sampling(ctf)
    if sampling is not None:
        parts.append(sampling)
    if ctf.voltage_kv is not None:
        parts.append(f"{ctf.voltage_kv:.0f} kV")
    if ctf.spherical_aberration_mm is not None:
        parts.append(f"Cs {ctf.spherical_aberration_mm:g} mm")
    if ctf.amplitude_contrast is not None:
        parts.append(f"amplitude contrast {ctf.amplitude_contrast:g}")
    parts.append(f"{len(ctf.z_value)} tilts")
    return " · ".join(parts)


def _dose_norm(ctf: CtfTable) -> Normalize:
    assert ctf.dose is not None
    low, high = float(np.min(ctf.dose)), float(np.max(ctf.dose))
    return Normalize(vmin=low, vmax=high if high > low else low + 1.0)


def _view_colours(ctf: CtfTable) -> np.ndarray:
    """One RGBA row per tilt: by dose when the mdoc recorded it, else all the series blue."""

    if ctf.dose is None:
        return np.tile(to_rgba(style.SERIES_BLUE), (len(ctf.z_value), 1))
    norm = _dose_norm(ctf)
    return np.asarray([style.MARKER_MAP(norm(float(value))) for value in ctf.dose])


def _reference_label(axes: Axes, text: str, y: float) -> None:
    axes.annotate(
        text,
        (1.0, y),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=style.FONT_TINY,
        color=style.INK_SECONDARY,
    )


def _defocus_plot(axes: Axes, ctf: CtfTable, colours: np.ndarray) -> None:
    angles = ctf.tilt_angle_deg
    low = np.minimum(ctf.defocus1_um, ctf.defocus2_um)
    high = np.maximum(ctf.defocus1_um, ctf.defocus2_um)
    axes.vlines(angles, low, high, colors=cast(Any, colours), linewidth=1.3)
    axes.scatter(angles, ctf.mean_defocus_um, c=colours, s=24, zorder=3, linewidths=0)
    if ctf.hint_defocus_um is not None:
        axes.axhline(ctf.hint_defocus_um, color=style.INK_MUTED, linewidth=0.8, zorder=1)
        _reference_label(axes, "estimate at 0°", ctf.hint_defocus_um)
    axes.set_xlabel("Tilt angle (°)")
    axes.set_ylabel("Defocus (µm)")
    axes.set_title("Defocus per tilt", loc="left")
    axes.grid(True, axis="y")


def _fit_plot(axes: Axes, ctf: CtfTable, colours: np.ndarray) -> None:
    axes.set_xlabel("Tilt angle (°)")
    axes.set_title("How far the CTF fit reached", loc="left")
    if ctf.fit_resolution_angstrom is None:
        axes.text(
            0.5,
            0.5,
            f"{ctf.backend} records no per-tilt fit limit",
            transform=axes.transAxes,
            ha="center",
            color=style.INK_SECONDARY,
        )
        axes.set_yticks([])
        return
    fit = ctf.fit_resolution_angstrom
    finite = np.isfinite(fit)
    axes.scatter(
        ctf.tilt_angle_deg[finite], fit[finite], c=colours[finite], s=24, zorder=3, linewidths=0
    )
    if ctf.nyquist_angstrom is not None:
        axes.axhline(ctf.nyquist_angstrom, color=style.INK_MUTED, linewidth=0.8, zorder=1)
        _reference_label(axes, "Nyquist of the fitted stack", ctf.nyquist_angstrom)
        axes.set_ylim(bottom=min(ctf.nyquist_angstrom, float(np.nanmin(fit))) * 0.9)
    axes.set_ylabel("Fit limit (Å, lower is better)")
    axes.grid(True, axis="y")


def _thumbnail_views(ctf: CtfTable) -> list[int]:
    """The tilt nearest 0°, the one nearest 30° on the other side, and the most tilted."""

    most_tilted = int(np.argmax(np.abs(ctf.tilt_angle_deg)))
    other_side = 30.0 if ctf.tilt_angle_deg[most_tilted] < 0 else -30.0
    views: list[int] = []
    for view in (ctf.nearest_view(0.0), ctf.nearest_view(other_side), most_tilted):
        if view not in views:
            views.append(view)
    return views


def _thumbnail_title(ctf: CtfTable, view: int) -> str:
    title = f"{ctf.tilt_angle_deg[view]:+.0f}°"
    if ctf.fit_resolution_angstrom is not None and np.isfinite(ctf.fit_resolution_angstrom[view]):
        title += f" · fit to {ctf.fit_resolution_angstrom[view]:.1f} Å"
    return title


def _numbers(report: SeriesReport, ctf: CtfTable) -> str:
    mean = ctf.mean_defocus_um
    astigmatism_nm = np.abs(ctf.defocus1_um - ctf.defocus2_um) * 1000.0
    lines = []
    sampling = fit_sampling(ctf)
    if sampling is not None:
        lines.extend(textwrap.wrap(sampling[0].upper() + sampling[1:], 36))
    lines += [
        f"Mean defocus {float(np.mean(mean)):.2f} µm",
        f"  range {float(np.min(mean)):.2f} to {float(np.max(mean)):.2f} µm",
        f"Mean astigmatism {float(np.mean(astigmatism_nm)):.0f} nm",
    ]
    if ctf.fit_resolution_angstrom is not None:
        fit = ctf.fit_resolution_angstrom
        best = int(np.nanargmin(fit))
        near_zero = ctf.nearest_view(0.0)
        lines.append(f"Best fit {fit[best]:.1f} Å, at {ctf.tilt_angle_deg[best]:+.0f}°")
        lines.append(f"At {ctf.tilt_angle_deg[near_zero]:+.0f}°: {fit[near_zero]:.1f} Å")
    if ctf.dose is not None:
        lines.append(f"Dose {float(np.min(ctf.dose)):.0f} to {float(np.max(ctf.dose)):.0f} e⁻/Å²")
    record = report.qc.get("ctf_estimation")
    if record is not None:
        lines.append("")
        lines.append(f"QC: {style.STATUS_LABELS.get(record.status, record.status)}")
        for warning in record.warnings:
            lines.extend(textwrap.wrap(warning, 38))
    return "\n".join(lines)
