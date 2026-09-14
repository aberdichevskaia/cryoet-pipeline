"""Soft alignment weights and shift-dependent weighted normalized correlation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def soft_spherical_alignment_mask(
    size: int,
    *,
    pixel_spacing_angstrom: float,
    diameter_angstrom: float,
    device: torch.device,
) -> torch.Tensor:
    """A reference-frame weight: flat to 80% radius, cosine taper to full radius."""
    if not math.isfinite(pixel_spacing_angstrom) or pixel_spacing_angstrom <= 0:
        raise ValueError("alignment mask needs a positive finite voxel spacing")
    if (
        not math.isfinite(diameter_angstrom)
        or not 0 < diameter_angstrom < size * pixel_spacing_angstrom
    ):
        raise ValueError("alignment mask diameter must be positive and fit inside the box")
    axis = (
        torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2
    ) * pixel_spacing_angstrom
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    radius = diameter_angstrom / 2
    phase = ((torch.sqrt(x * x + y * y + z * z) - 0.8 * radius) / (0.2 * radius)).clamp(0, 1)
    return 0.5 * (1 + torch.cos(math.pi * phase))


@dataclass(frozen=True)
class MaskedNccReference:
    shape: tuple[int, ...]
    weight_spectrum_conj: torch.Tensor
    reference_spectrum_conj: torch.Tensor
    weight_sum: torch.Tensor
    reference_energy: torch.Tensor
    margin_px: int


def prepare_masked_ncc(reference: torch.Tensor, weight: torch.Tensor) -> MaskedNccReference:
    """Cache reference-only terms; the soft weight enters each moment once."""
    if reference.ndim != 3 or len(set(reference.shape)) != 1 or weight.shape != reference.shape:
        raise ValueError("masked NCC requires equally shaped cubic reference and weights")
    if weight.device != reference.device:
        raise ValueError("masked NCC reference and weights must share a device")
    if (
        not bool(torch.isfinite(weight).all())
        or bool((weight < 0).any())
        or bool((weight > 1).any())
    ):
        raise ValueError("alignment weights must be finite and between zero and one")
    weight = weight.to(torch.float32)
    total = weight.sum()
    if float(total) <= 0:
        raise ValueError("alignment mask is empty")
    ref = reference.to(torch.float32)
    ref = ref - (weight * ref).sum() / total
    energy = (weight * ref.square()).sum()
    if not bool(torch.isfinite(energy)) or float(energy) <= 0:
        raise ValueError("reference has no finite variance inside the alignment mask")
    support = torch.nonzero(weight > 0)
    size = reference.shape[0]
    margin = int(torch.minimum(support, size - 1 - support).min())
    return MaskedNccReference(
        tuple(reference.shape),
        torch.fft.rfftn(weight).conj(),
        torch.fft.rfftn(weight * ref).conj(),
        total,
        energy,
        margin,
    )


def best_masked_shift_and_correlation(
    candidate: torch.Tensor,
    reference: MaskedNccReference,
    *,
    max_shift_px: float,
) -> tuple[np.ndarray, float]:
    """Weighted Pearson NCC at each allowed integer shift, in reference XYZ.

    A fixed mask weights q(x) and p(x + delta). Its candidate mean and variance
    therefore change with delta. FFTs compute all three required moments. The
    support/margin guard excludes circular wrapping and makes each sample agree
    with a direct zero-padded translation of the rotated candidate.
    """
    if tuple(candidate.shape) != reference.shape or candidate.device != reference.weight_sum.device:
        raise ValueError("masked NCC candidate shape/device differs from reference")
    size = reference.shape[0]
    radius = min(max(int(math.floor(max_shift_px)), 0), size // 2)
    if radius > reference.margin_px:
        raise ValueError(
            "alignment mask plus shift window reaches the box edge; reduce mask diameter or shift range"
        )
    p = (candidate - candidate.mean()).to(torch.float32)
    spectrum = torch.fft.rfftn(p)

    def correlate(other: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfftn(other * values, s=reference.shape)

    numerator = correlate(reference.reference_spectrum_conj, spectrum)
    first = correlate(reference.weight_spectrum_conj, spectrum)
    second = correlate(reference.weight_spectrum_conj, torch.fft.rfftn(p.square()))
    variance = (second - first.square() / reference.weight_sum).clamp_min(0)
    norm = torch.sqrt(variance * reference.reference_energy)
    score = torch.where(norm > 1e-12, numerator / norm, -torch.inf)
    center = size // 2
    window = torch.fft.fftshift(score)[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    flat = int(window.reshape(-1).argmax())
    correlation = float(window.reshape(-1)[flat])
    if not math.isfinite(correlation):
        raise ValueError("candidate has no finite NCC inside the alignment mask and shift window")
    span = 2 * radius + 1
    delta = np.array(
        [flat % span - radius, (flat // span) % span - radius, flat // (span * span) - radius],
        dtype=float,
    )
    return delta, correlation
