"""`dynamo_search.refine_particle`, batched over orientations on a torch device.

The same search, step by step: the particle bandpassed twice and through its
wedge, normalized, its FFTs taken once; at each refinement level the candidates of
Dynamo's CPU grid (`angle_list`, run on the host as before), each a rotation of
the prepared reference with the MEX's float32 trilinear interpolation and
eight-corner support, filtered, correlated with Roseman's moving-mask local
normalization, its peak taken inside the shift sphere and refined by Dynamo's
three-sample spline. Candidates are evaluated a chunk at a time instead of one by
one; the winner is chosen as the CPU loop chooses it, the last candidate whose
score is at least the best so far.

`dtype` float64 is for checking this module against the numpy kernel: the
rotation is float32 in both, as in Dynamo, and the rest agrees to FFT rounding,
so the poses agree but for exact ties. float32 is what a GPU runs, as Dynamo's
CUDA motor does: the CCs then differ from float64 by float32 rounding, and a
candidate within that of the winner can win instead. That is the near-tie
disagreement measured between Dynamo's own GPU and CPU modes (Gag, 2026-10-13:
rotation median ~1.3 deg, one finest step), and it is the tolerance this module
is to be held to against Dynamo's GPU, not bitwise parity.

Where the numpy kernel raises (a rotated mask with fewer than two voxels, a
zero-variance template) this raises the same errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch

from cryoet_pipeline.backends.dynamo_search import (
    DEFAULT_SETTINGS,
    SearchResult,
    SearchSettings,
    _cube,
    _triple,
    _xyz,
    angle_list,
    bandpass,
    euler_matrix,
    wedge,
)

DEFAULT_CHUNK = 64
"""Candidates evaluated at once: about 11 complex volumes each, 1.1 MB apiece at 52^3."""

# The order the CPU kernel sums the eight interpolation corners in.
_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0), (1, 1, 1))


@lru_cache(maxsize=8)
def _grid(n: int, device: torch.device) -> torch.Tensor:
    """`_xyz(n)` on a device, once per size and device."""

    return torch.from_numpy(_xyz(n).copy()).to(device)


def _matrices(candidates: np.ndarray) -> np.ndarray:
    """The float32 rotation matrices the numpy `resample` builds, one per candidate."""

    return np.stack([euler_matrix(_triple(angles).astype(np.float32)).astype(np.float32)
                     for angles in candidates])


def resample_batch(volume: torch.Tensor, matrices: torch.Tensor,
                   support: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """`resample(volume, angles, mask=support)` for every matrix, zero shift.

    `volume` (n, n, n) float32; `matrices` (B, 3, 3) float32; `support` (n, n, n)
    bool or None. Returns float32 values (B, n, n, n), zero where undefined, and
    the bool support of each.
    """

    n = volume.shape[0]
    x, y, z = _grid(n, volume.device)
    m = matrices[:, :, :, None, None, None]
    q = [m[:, i, 0] * x + m[:, i, 1] * y + m[:, i, 2] * z for i in range(3)]
    floors = [torch.floor(component).to(torch.int32) for component in q]
    frac = [component - floor.to(torch.float32) for component, floor in zip(q, floors, strict=True)]
    base = [floor + n // 2 for floor in floors]
    valid = torch.ones_like(base[0], dtype=torch.bool)
    for component in base:
        valid &= (component >= 0) & (component + 1 < n)
    base = [component.clamp(0, n - 2).to(torch.int64) for component in base]
    flat = volume.reshape(-1)
    mask_flat = support.reshape(-1) if support is not None else None
    result = torch.zeros_like(q[0])
    for dx, dy, dz in _CORNERS:
        index = ((base[2] + dz) * n + (base[1] + dy)) * n + (base[0] + dx)
        if mask_flat is not None:
            valid &= mask_flat[index]
        value = flat[index]
        for axis, step in enumerate((dx, dy, dz)):
            value = value * (frac[axis] if step else 1 - frac[axis])
        result += value
    result = torch.where(valid, result, torch.zeros_like(result))
    return result, valid


def _centred(volume: torch.Tensor) -> torch.Tensor:
    """`np.roll(np.fft.fftshift(v)[::-1, ::-1, ::-1], (1, 1, 1), (0, 1, 2))` per volume."""

    dims = (-3, -2, -1)
    shifted = torch.fft.fftshift(volume, dim=dims)
    return torch.roll(torch.flip(shifted, dims=dims), shifts=(1, 1, 1), dims=dims)


def local_correlation_batch(templates: torch.Tensor, supports: torch.Tensor,
                            particle_fft: torch.Tensor,
                            particle_squared_fft: torch.Tensor) -> torch.Tensor:
    """`local_correlation` for each (template, support) against one particle."""

    dims = (-3, -2, -1)
    valid = supports
    counts = valid.sum(dim=dims)
    if bool((counts < 2).any()):
        raise ValueError("rotated mask has fewer than two supported voxels")
    zero = torch.zeros((), dtype=templates.dtype, device=templates.device)
    means = torch.where(valid, templates, zero).sum(dim=dims) / counts
    t = torch.where(valid, templates - means[:, None, None, None], zero)
    norms = torch.linalg.vector_norm(t, dim=dims)
    if bool((norms == 0).any()):
        raise ValueError("zero-variance rotated template")
    fm = torch.fft.fftn(valid.to(templates.dtype), dim=dims)
    conj_p, conj_p2 = particle_fft.conj(), particle_squared_fft.conj()
    numerator = torch.fft.ifftn(torch.fft.fftn(t, dim=dims) * conj_p, dim=dims).real
    sum_p = torch.fft.ifftn(fm * conj_p, dim=dims).real
    sum_p2 = torch.fft.ifftn(fm * conj_p2, dim=dims).real
    count = counts.to(templates.dtype)[:, None, None, None]
    variance = torch.clamp(sum_p2 - sum_p**2 / count, min=0)
    denominator = torch.sqrt(variance) * norms[:, None, None, None]
    cc = torch.full_like(numerator, -1.0)
    positive = denominator > 0
    cc[positive] = numerator[positive] / denominator[positive]
    return _centred(cc)


def spline_peaks(y1: np.ndarray, y2: np.ndarray, y3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`_spline_peak` over arrays, element by element, with its formulas in float64."""

    a, b, c = (np.asarray(v, dtype=np.float64) for v in (y1, y2, y3))
    flat = (a == c) | (np.abs(b - a) >= 2) | (np.abs(b - c) >= 2)
    left = -(a**2) / 4 + a * c / 2 + b**2 - 2 * b * c + 3 * c**2 / 4
    right = 3 * a**2 / 4 - 2 * a * b + a * c / 2 + b**2 - c**2 / 4
    u, w = np.emath.sqrt(left), np.emath.sqrt(right)
    d, e = a + 2 * b - 3 * c, 2 * b - 3 * a + c
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        positions = np.stack(
            [-2 * u / d, 2 * u / d, (e + 2 * w) / e, -(3 * a - 2 * b - c + 2 * w) / e]
        )
        values = np.stack(
            [
                (2 * a * b - 3 * a * c + a * u - 2 * b * u + c * u + a**2) / d,
                (2 * a * b - 3 * a * c - a * u + 2 * b * u - c * u + a**2) / d,
                c - w / 2 + 2 * np.emath.power(right, 1.5) / e**2 - 2 * (b - c) * w / e,
                c + w / 2 - 2 * np.emath.power(right, 1.5) / e**2 + 2 * (b - c) * w / e,
            ]
        )
        good = (
            np.isfinite(values)
            & np.isfinite(positions)
            & (values.imag == 0)
            & (positions.imag == 0)
            & (positions.real >= 0)
            & (positions.real <= 1)
        )
    scores = np.where(good, values.real, -np.inf)
    index = np.argmax(scores, axis=0)
    picked = np.take_along_axis(scores, index[None], axis=0)[0]
    offsets = np.take_along_axis(positions.real, index[None], axis=0)[0] - (index < 2)
    use = ~flat & good.any(axis=0)
    return np.where(use, picked, b), np.where(use, offsets, 0.0)


def restricted_peaks(cc: torch.Tensor, shift: np.ndarray,
                     radius: float) -> tuple[np.ndarray, np.ndarray]:
    """`restricted_peak` for each map of a batch, all about the same centre."""

    n = cc.shape[-1]
    if not 0 < radius < n / 2:
        raise ValueError("shift radius must be in (0, N/2)")
    old = _triple(shift)
    steps = (np.sign(old) * np.floor(np.abs(old) + 0.5)).astype(int)
    sphere = np.sum(_xyz(n) ** 2, axis=0) <= np.ceil(radius) ** 2
    sphere = np.roll(sphere, tuple(steps[::-1]), axis=(0, 1, 2))
    inside = torch.from_numpy(sphere).to(cc.device)
    restricted = torch.where(inside, cc, torch.full_like(cc, -1.0))
    flat_index = torch.argmax(restricted.reshape(len(cc), -1), dim=1)
    zi, yi, xi = (flat_index // (n * n)), (flat_index // n) % n, flat_index % n
    index = torch.stack([zi, yi, xi], dim=1)
    batch = torch.arange(len(cc), device=cc.device)
    y2 = restricted[batch, zi, yi, xi]
    sides = []
    for axis in range(3):
        low, high = index.clone(), index.clone()
        low[:, axis] -= 1
        high[:, axis] += 1
        inner = (index[:, axis] > 0) & (index[:, axis] < n - 1)
        low = low.clamp(0, n - 1)
        high = high.clamp(0, n - 1)
        y1 = torch.where(inner, cc[batch, low[:, 0], low[:, 1], low[:, 2]], y2)
        y3 = torch.where(inner, cc[batch, high[:, 0], high[:, 1], high[:, 2]], y2)
        sides.append((y1, y3))
    # Read back to the host before widening: an MPS device has no float64.
    host = [value.cpu().numpy().astype(np.float64) for pair in sides for value in pair]
    centre = y2.cpu().numpy().astype(np.float64)
    location = index.cpu().numpy().astype(np.float64)
    peaks = []
    for axis in range(3):
        value, offset = spline_peaks(host[2 * axis], centre, host[2 * axis + 1])
        peaks.append(value)
        location[:, axis] += offset
    values = np.max(np.stack(peaks), axis=0)
    values = np.where((values > 1) & (values < 1.05), 1.0, values)
    return location[:, ::-1] - n / 2, values


def _apply_filter(volumes: torch.Tensor, centred_filter: torch.Tensor) -> torch.Tensor:
    dims = (-3, -2, -1)
    spectrum = torch.fft.fftn(volumes, dim=dims) * torch.fft.ifftshift(centred_filter, dim=dims)
    return torch.fft.ifftn(spectrum, dim=dims).real


def refine_particle_torch(
    particle: np.ndarray, reference: np.ndarray, mask: np.ndarray,
    angles: np.ndarray | Sequence[float], shift: np.ndarray | Sequence[float],
    tilt_range: tuple[float, float], settings: SearchSettings = DEFAULT_SETTINGS, *,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    chunk: int = DEFAULT_CHUNK,
) -> SearchResult:
    """`refine_particle`, with each level's candidates evaluated `chunk` at a time."""

    p, ref, mask = _cube(particle), _cube(reference), _cube(mask)
    if not p.shape == ref.shape == mask.shape or np.any(mask < 0) or not np.any(mask):
        raise ValueError("shared shape and a nonnegative/nonempty mask are required")
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    device = torch.device(device)
    n = len(p)
    # Narrowed on the host, then moved: an MPS device takes no float64 at all.
    shell = torch.from_numpy(bandpass(n, settings.high, settings.low)).to(dtype).to(device)
    filter_mask = shell * torch.from_numpy(wedge(n, tilt_range)).to(dtype).to(device)
    volume = torch.from_numpy(p).to(dtype).to(device)
    volume = _apply_filter(_apply_filter(volume, shell), filter_mask)
    volume = (volume - volume.mean()) / volume.std(correction=1)
    dims = (-3, -2, -1)
    fp = torch.fft.fftn(volume, dim=dims)
    fp2 = torch.fft.fftn(volume * volume, dim=dims)
    reference_f32 = torch.from_numpy(ref.astype(np.float32)).to(device)
    support = torch.from_numpy(mask != 0).to(device)

    best_angles, best_shift = _triple(angles).copy(), _triple(shift).copy()
    cr, cs, ir, ins = (settings.cone_aperture, settings.cone_sampling,
                       settings.inplane_range, settings.inplane_sampling)
    trace = []
    best_score = -np.inf
    for level in range(settings.refine + 1):
        candidates = angle_list(best_angles, cr, cs, ir, ins)
        center_shift = best_shift.copy() if settings.area_search_modus == 2 else np.zeros(3)
        best_score = -np.inf
        for start in range(0, len(candidates), chunk):
            part = candidates[start:start + chunk]
            matrices = torch.from_numpy(_matrices(part)).to(device)
            rotated, valid = resample_batch(reference_f32, matrices, support)
            rotated = _apply_filter(rotated.to(dtype), filter_mask)
            cc = local_correlation_batch(rotated, valid, fp, fp2)
            shifts, scores = restricted_peaks(cc, center_shift, settings.shift_limit)
            # The CPU loop keeps a candidate whose score is at least the best so far,
            # so of equal scores the last one wins.
            last = len(scores) - 1 - int(np.argmax(scores[::-1]))
            if scores[last] >= best_score:
                best_score = float(scores[last])
                best_angles, best_shift = part[last].copy(), shifts[last].copy()
        trace.append(
            {
                "level": level,
                "candidate_count": len(candidates),
                "angles": best_angles.tolist(),
                "shift": best_shift.tolist(),
                "correlation": best_score,
                "cone_sampling": cs,
            }
        )
        cr, ir = settings.refine_factor * cs, settings.refine_factor * ins
        cs, ins = cs / settings.refine_factor, ins / settings.refine_factor
    return SearchResult(best_angles.copy(), best_shift.copy(), best_score, tuple(trace))


def search_particles_torch(
    particles: Sequence[np.ndarray], reference: np.ndarray, mask: np.ndarray,
    starts: Sequence[tuple[np.ndarray, np.ndarray]], tilt_ranges: Sequence[tuple[float, float]],
    settings: SearchSettings, *, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32, chunk: int = DEFAULT_CHUNK,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """`search_particles` on a torch device: every particle against one fixed reference."""

    results = []
    for particle, (angles, shift), tilt_range in zip(particles, starts, tilt_ranges, strict=True):
        result = refine_particle_torch(particle, reference, mask, angles, shift, tilt_range,
                                       settings, device=device, dtype=dtype, chunk=chunk)
        results.append((result.angles, result.shift, result.correlation))
    return results
