"""The tomogram, seen at three depths and from two sides, and its membranes."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import zarr
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from cryoet_pipeline.reports import style
from cryoet_pipeline.reports.collect import SeriesReport, TomogramInfo
from cryoet_pipeline.reports.imaging import (
    SMOOTHING_SIGMA_PX,
    Depth,
    block_mean,
    display_range,
    nice_length,
    reduction_factor,
    slab_half_width_px,
    slab_mean,
    smooth,
)

DEPTH_LETTERS = "abc"


def open_volume(path: Any) -> Any:
    return cast(Any, zarr.open(path, mode="r"))


def slab_caption(info: TomogramInfo) -> str:
    half = slab_half_width_px(info.voxel_spacing_angstrom)
    thickness_nm = (2 * half + 1) * info.voxel_spacing_angstrom / 10.0
    return (
        f"Protein is dark. Each panel is the mean of a {thickness_nm:.0f} nm slab "
        f"({2 * half + 1} voxels), smoothed with a {SMOOTHING_SIGMA_PX:g}-voxel Gaussian; "
        "all panels on this page share one grey scale (1st to 99th percentile). "
        "y and z increase upwards, as in 3dmod."
    )


def picture(volume: Any, axis: int, center: int, half: int) -> tuple[np.ndarray, int]:
    """A smoothed slab, block-averaged for the page, and the factor it was reduced by."""

    image = smooth(slab_mean(volume, axis, center, half))
    factor = reduction_factor(image.shape)
    return block_mean(image, factor), factor


def show(
    axes: Axes, image: np.ndarray, factor: int, limits: tuple[float, float], **kwargs: Any
) -> None:
    """Draw a reduced picture in the original voxel coordinates, so overlays need no scaling."""

    height, width = image.shape
    axes.imshow(
        image,
        cmap=kwargs.pop("cmap", "gray"),
        vmin=limits[0],
        vmax=limits[1],
        origin="lower",
        extent=(-0.5, width * factor - 0.5, -0.5, height * factor - 0.5),
        **kwargs,
    )


def add_scale_bar(axes: Axes, info: TomogramInfo, width_px: float, height_px: float) -> None:
    width_nm = width_px * info.voxel_spacing_angstrom / 10.0
    length_nm = nice_length(width_nm / 5.0)
    style.scale_bar(
        axes,
        length_nm * 10.0 / info.voxel_spacing_angstrom,
        f"{length_nm:g} nm",
        image_width_px=width_px,
        image_height_px=height_px,
    )


def depth_title(letter: str, depth: Depth, info: TomogramInfo) -> str:
    return f"{letter} · {depth.center * info.voxel_spacing_angstrom / 10.0:.0f} nm up"


def mark_depths(axes: Axes, depths: list[Depth], width_px: float) -> None:
    for letter, depth in zip(DEPTH_LETTERS, depths, strict=False):
        axes.axhline(depth.center, color="white", linewidth=0.6, alpha=0.8)
        label = axes.text(
            0.01 * width_px,
            depth.center,
            letter,
            color="white",
            fontsize=7,
            va="bottom",
        )
        label.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground=style.INK)])


def tomogram_page(report: SeriesReport, depths: list[Depth]) -> Figure:
    info = report.tomogram
    assert info is not None
    volume = open_volume(info.artifact.path)
    half = slab_half_width_px(info.voxel_spacing_angstrom)
    depth_px, height_px, width_px = info.shape

    tops = [picture(volume, 0, depth.center, half) for depth in depths]
    side_xz = picture(volume, 1, height_px // 2, half)
    side_yz = picture(volume, 2, width_px // 2, half)
    limits = display_range([image for image, _ in (*tops, side_xz, side_yz)])

    figure = style.new_page(f"{report.series} · Tomogram", _tomogram_subtitle(report))
    grid = GridSpec(
        2, 3, figure=figure, left=0.035, right=0.965, top=0.85, bottom=0.15,
        height_ratios=[1.25, 1.0], hspace=0.18, wspace=0.06,
    )
    for column, (letter, depth, (image, factor)) in enumerate(
        zip(DEPTH_LETTERS, depths, tops, strict=False)
    ):
        axes = figure.add_subplot(grid[0, column])
        show(axes, image, factor, limits)
        style.image_axes(axes, depth_title(letter, depth, info) + " · XY")
        if column == 0:
            add_scale_bar(axes, info, width_px, height_px)

    axes = figure.add_subplot(grid[1, 0:2])
    show(axes, side_xz[0], side_xz[1], limits, aspect="equal")
    mark_depths(axes, depths, width_px)
    style.image_axes(axes, "XZ, through the middle in y")
    add_scale_bar(axes, info, width_px, depth_px)

    axes = figure.add_subplot(grid[1, 2])
    show(axes, side_yz[0], side_yz[1], limits, aspect="equal")
    mark_depths(axes, depths, height_px)
    style.image_axes(axes, "YZ, through the middle in x")

    style.caption(
        figure,
        slab_caption(info) + " Letters a to c mark the depths of the top row on the side views.",
    )
    return figure


def _tomogram_subtitle(report: SeriesReport) -> str:
    info = report.tomogram
    assert info is not None
    z, y, x = info.shape
    parts = [f"{x} × {y} × {z} voxels at {info.voxel_spacing_angstrom:.2f} Å"]
    manifest = report.artifacts.manifest
    if info.included_z_values is not None:
        angle_by_z = {image.z_value: image.tilt_angle_deg for image in manifest.images}
        angles = [
            angle_by_z[z_value] for z_value in info.included_z_values if z_value in angle_by_z
        ]
        text = f"{len(info.included_z_values)} of {len(manifest.images)} views"
        if angles:
            text += f", {min(angles):+.0f}° to {max(angles):+.0f}°"
        parts.append(text)
    if info.ctf_corrected is not None:
        parts.append("CTF corrected" if info.ctf_corrected else "not CTF corrected")
    return " · ".join(parts)


def membranes_page(report: SeriesReport, depths: list[Depth]) -> Figure:
    info = report.tomogram
    segmentation = report.artifacts.segmentation
    assert info is not None and segmentation is not None
    volume = open_volume(info.artifact.path)
    mask = open_volume(segmentation.path)
    if tuple(mask.shape) != info.shape:
        raise ValueError(
            f"the segmentation is {tuple(mask.shape)} voxels but the tomogram {info.shape}, "
            "so it is not of this tomogram"
        )
    half = slab_half_width_px(info.voxel_spacing_angstrom)
    depth_px, height_px, width_px = info.shape

    tops = [picture(volume, 0, depth.center, half) for depth in depths]
    side = picture(volume, 1, height_px // 2, half)
    limits = display_range([image for image, _ in (*tops, side)])

    fraction = None
    if "tomogram_segmentation" in report.qc:
        fraction = report.qc["tomogram_segmentation"].document.get("foreground_fraction")
    subtitle = "MemBrain-seg"
    if isinstance(fraction, (int, float)):
        subtitle += f" · {fraction:.1%} of the volume segmented as membrane"
    figure = style.new_page(f"{report.series} · Membranes", subtitle)
    grid = GridSpec(
        2, 3, figure=figure, left=0.035, right=0.965, top=0.85, bottom=0.15,
        height_ratios=[1.25, 1.0], hspace=0.18, wspace=0.06,
    )
    for column, (letter, depth, (image, factor)) in enumerate(
        zip(DEPTH_LETTERS, depths, tops, strict=False)
    ):
        axes = figure.add_subplot(grid[0, column])
        show(axes, image, factor, limits)
        outline = _mask_slab(mask, 0, depth.center, half)
        _outline(axes, block_mean(outline, factor), factor)
        style.image_axes(axes, depth_title(letter, depth, info) + " · XY")
        if column == 0:
            add_scale_bar(axes, info, width_px, height_px)

    axes = figure.add_subplot(grid[1, 0])
    share = _depth_share(mask)
    factor = reduction_factor(share.shape)
    reduced = block_mean(share, factor)
    show(axes, reduced, factor, (0.0, max(float(reduced.max()), 1e-6)), cmap=style.MAGNITUDE_MAP)
    style.image_axes(axes, "Seen from above: share of the depth that is membrane")
    colorbar = figure.colorbar(
        axes.images[0], ax=axes, orientation="horizontal", fraction=0.05, pad=0.03,
        format=PercentFormatter(xmax=1.0, decimals=0),
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=6)

    axes = figure.add_subplot(grid[1, 1:3])
    image, factor = side
    show(axes, image, factor, limits, aspect="equal")
    _outline(axes, block_mean(_mask_slab(mask, 1, height_px // 2, half), factor), factor)
    mark_depths(axes, depths, width_px)
    style.image_axes(axes, "XZ, through the middle in y")

    style.caption(
        figure,
        slab_caption(info)
        + " Aqua outlines the voxels MemBrain-seg labelled as membrane within each slab.",
    )
    return figure


def _mask_slab(mask: Any, axis: int, center: int, half: int) -> np.ndarray:
    size = int(mask.shape[axis])
    index: list[slice] = [slice(None)] * 3
    index[axis] = slice(max(0, center - half), min(size, center + half + 1))
    return np.asarray((np.asarray(mask[tuple(index)]) != 0).any(axis=axis), dtype=np.float32)


def _depth_share(mask: Any, chunk: int = 32) -> np.ndarray:
    """The fraction of each column's depth that is membrane, read a few slices at a time."""

    depth = int(mask.shape[0])
    count = np.zeros(mask.shape[1:], dtype=np.int64)
    for start in range(0, depth, chunk):
        count += (np.asarray(mask[start : start + chunk]) != 0).sum(axis=0)
    return (count / depth).astype(np.float32)


def _outline(axes: Axes, reduced_mask: np.ndarray, factor: int) -> None:
    if not reduced_mask.any():
        return
    height, width = reduced_mask.shape
    rows = (np.arange(height) + 0.5) * factor - 0.5
    columns = (np.arange(width) + 0.5) * factor - 0.5
    axes.contour(
        columns,
        rows,
        reduced_mask,
        levels=[0.5],
        colors=style.MEMBRANE_COLOUR,
        linewidths=0.7,
    )
