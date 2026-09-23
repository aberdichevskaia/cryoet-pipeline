"""Fourier-space filters applied to whole volumes.

Its own module because both subtomogram averaging and the FSC need it, and
`fsc.py` already imports from `backends/subtomogram.py` — so the dependency can
only run one way, and a shared low-level home is the only shape that does not
create a cycle.

The resolution convention here is the one `compute_fsc_curve` already uses: for a
box of `size` voxels at `spacing` angstrom per voxel, Fourier shell `k` is
`size * spacing / k` angstrom, so a resolution of `d` angstrom is the spatial
frequency `1 / d` in cycles per angstrom. Keeping the two in step matters,
because a low-pass limit derived from an FSC curve is fed straight back in here.
"""

from __future__ import annotations

import math

import torch


def _radial_frequency_cycles_per_angstrom(
    shape: tuple[int, int, int],
    *,
    pixel_spacing_angstrom: float,
    device: torch.device,
) -> torch.Tensor:
    """The frequency magnitude of every bin of an `rfftn` of this shape."""

    size_z, size_y, size_x = shape
    frequency_z = torch.fft.fftfreq(size_z, d=pixel_spacing_angstrom, device=device)
    frequency_y = torch.fft.fftfreq(size_y, d=pixel_spacing_angstrom, device=device)
    # rfftn keeps only the non-redundant half of the last axis.
    frequency_x = torch.fft.rfftfreq(size_x, d=pixel_spacing_angstrom, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(
        frequency_z, frequency_y, frequency_x, indexing="ij"
    )
    return torch.sqrt(grid_z**2 + grid_y**2 + grid_x**2)


def low_pass_filter_volume(
    volume: torch.Tensor,
    *,
    cutoff_angstrom: float,
    pixel_spacing_angstrom: float,
    edge_width_shells: float = 2.0,
) -> torch.Tensor:
    """Low-pass a volume at `cutoff_angstrom`, with a raised-cosine edge.

    The edge is soft rather than a hard cutoff, and its width is given in Fourier
    shells so that it means the same thing at any box size: a brick-wall cutoff
    convolves the map with a sinc and rings, which puts concentric shells around
    every feature — exactly the kind of structure that a reader, or an FSC, can
    mistake for real density.

    Why a low-pass is needed here at all, rather than being a refinement nicety:
    two half-sets refined independently may still start from a shared reference,
    and that is only legitimate because the reference is filtered — the
    gold-standard guarantee holds above the filter frequency, not below it. Refine
    against an unfiltered reference and each half's own noise is free to imprint
    at every frequency, which is the self-enhancing loop the whole split exists to
    avoid.

    `cutoff_angstrom` is where the pass band ends; the response reaches zero
    `edge_width_shells` shells beyond it. The DC term is always passed, so this
    never changes a volume's mean.
    """

    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {tuple(volume.shape)}")
    if cutoff_angstrom <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff_angstrom}")
    if pixel_spacing_angstrom <= 0.0:
        raise ValueError(f"pixel spacing must be positive, got {pixel_spacing_angstrom}")
    if edge_width_shells < 0.0:
        raise ValueError(f"edge width cannot be negative, got {edge_width_shells}")

    shape = (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2]))
    response = low_pass_response_rfft(
        shape,
        cutoff_angstrom=cutoff_angstrom,
        pixel_spacing_angstrom=pixel_spacing_angstrom,
        edge_width_shells=edge_width_shells,
        device=volume.device,
    )
    spectrum = torch.fft.rfftn(volume.to(torch.float32))
    filtered = torch.fft.irfftn(spectrum * response, s=shape)
    return filtered.to(volume.dtype)


def low_pass_response_rfft(
    shape: tuple[int, int, int],
    *,
    cutoff_angstrom: float,
    pixel_spacing_angstrom: float,
    device: torch.device,
    edge_width_shells: float = 2.0,
) -> torch.Tensor:
    """The same raised-cosine response used by `low_pass_filter_volume`.

    Exposed so a scorer can apply exactly one identical filter to the observed
    particle and its prediction, after putting both on the particle's grid.
    """

    if len(shape) != 3 or any(size < 1 for size in shape):
        raise ValueError(f"expected a positive 3D shape, got {shape}")
    if not math.isfinite(cutoff_angstrom) or cutoff_angstrom <= 0:
        raise ValueError("cutoff must be finite and positive")
    if not math.isfinite(pixel_spacing_angstrom) or pixel_spacing_angstrom <= 0:
        raise ValueError("pixel spacing must be finite and positive")
    if not math.isfinite(edge_width_shells) or edge_width_shells < 0:
        raise ValueError("edge width must be finite and nonnegative")
    frequency = _radial_frequency_cycles_per_angstrom(
        shape, pixel_spacing_angstrom=pixel_spacing_angstrom, device=device
    )

    pass_frequency = 1.0 / cutoff_angstrom
    # One shell is 1 / (size * spacing) in frequency, by the same convention the
    # FSC uses, so an edge given in shells converts without reference to the box.
    shell_frequency = 1.0 / (shape[0] * pixel_spacing_angstrom)
    stop_frequency = pass_frequency + edge_width_shells * shell_frequency

    response = torch.ones_like(frequency)
    if stop_frequency > pass_frequency:
        in_edge = (frequency > pass_frequency) & (frequency < stop_frequency)
        position = (frequency[in_edge] - pass_frequency) / (stop_frequency - pass_frequency)
        response[in_edge] = 0.5 * (1.0 + torch.cos(math.pi * position))
    response[frequency >= stop_frequency] = 0.0

    return response
