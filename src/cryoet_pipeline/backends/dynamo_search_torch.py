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

What is done differently, for speed, gives the same numbers. The rotation sums the
eight corners with the same float32 weights in the same order, bit for bit; a
cell's eight support tests are read from one map of whole cells. The filters and
correlations use real-input FFTs: the real part of a full-FFT product with a real
filter is the product with the filter's Hermitian part, which differs from the
filter only on the Nyquist planes, where the wedge is not point-symmetric. The
peak is read from the shift sphere's voxels and their neighbours only, mapped into
the uncentred correlation map, instead of centring the whole map. The particles of
one call share the reference and the mask's support on the device, and those of a
tilt range their filters (`search_results_torch`).

`dtype` float64 is for checking this module against the numpy kernel: the
rotation is float32 in both, as in Dynamo, and the rest agrees to FFT rounding,
so the poses agree but for exact ties. float32 is what a GPU runs, as Dynamo's
CUDA motor does: the CCs then differ from float64 by float32 rounding, and a
candidate within that of the winner can win instead. That is the near-tie
disagreement measured between Dynamo's own GPU and CPU modes (Gag, 2026-10-13:
rotation median ~1.3 deg, one finest step), and it is the tolerance this module
is to be held to against Dynamo's GPU, not bitwise parity.

`MultiDeviceSearch` deals the particles among processes, one or more per device.

Where the numpy kernel raises (a rotated mask with fewer than two voxels, a
zero-variance template) this raises the same errors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

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

DEFAULT_CHUNK = 256
"""Candidates evaluated at once: about 11 volumes each, 0.6 MB apiece at 52^3 in float32."""

DIMS = (-3, -2, -1)

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


def cell_support(support: torch.Tensor) -> torch.Tensor:
    """Whether each voxel, as a 2x2x2 cell's low corner, has all eight corners in `support`."""

    s = support
    cells = torch.zeros_like(s)
    cells[:-1, :-1, :-1] = (s[:-1, :-1, :-1] & s[1:, :-1, :-1] & s[:-1, 1:, :-1]
                            & s[:-1, :-1, 1:] & s[1:, 1:, :-1] & s[1:, :-1, 1:]
                            & s[:-1, 1:, 1:] & s[1:, 1:, 1:])
    return cells


def support_reach(support: torch.Tensor) -> torch.Tensor:
    """The output voxels a rotation of `support` can give support to, as linear indices.

    A rotation about the box centre keeps distances to it, so an output voxel more
    than the support's farthest voxel plus 2 from the centre pulls from a cell whose
    corner nearest the centre lies outside the support: it is undefined, zero, for
    every rotation, and need not be computed. For a sphere mask in a box twice the
    particle's size that is four fifths of the box.
    """

    n = support.shape[0]
    x, y, z = _grid(n, support.device)
    squared = x * x + y * y + z * z
    farthest = float(squared[support].max()) if bool(support.any()) else 0.0
    return torch.nonzero((squared <= (farthest**0.5 + 2) ** 2).reshape(-1))[:, 0]


def resample_batch(volume: torch.Tensor, matrices: torch.Tensor,
                   support: torch.Tensor | None, *, cells: torch.Tensor | None = None,
                   reach: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """`resample(volume, angles, mask=support)` for every matrix, zero shift.

    `volume` (n, n, n) float32; `matrices` (B, 3, 3) float32; `support` (n, n, n)
    bool or None; `cells`, `cell_support(support)`, and `reach`,
    `support_reach(support)`, when the caller keeps them: only the voxels in reach
    are computed, the rest are undefined anyway. Returns float32 values
    (B, n, n, n), zero where undefined, and the bool support of each.
    """

    n = volume.shape[0]
    x, y, z = _grid(n, volume.device)
    if reach is not None:
        x, y, z = (component.reshape(-1)[reach] for component in (x, y, z))
        m = matrices[:, :, :, None]
    else:
        m = matrices[:, :, :, None, None, None]
    q = [m[:, i, 0] * x + m[:, i, 1] * y + m[:, i, 2] * z for i in range(3)]
    floors = [torch.floor(component) for component in q]
    frac = [component - floor for component, floor in zip(q, floors, strict=True)]
    near = [1 - weight for weight in frac]
    base = [floor.to(torch.int64) + n // 2 for floor in floors]
    valid = torch.ones_like(base[0], dtype=torch.bool)
    for component in base:
        valid &= (component >= 0) & (component + 1 < n)
    base = [component.clamp(0, n - 2) for component in base]
    low = (base[2] * n + base[1]) * n + base[0]
    if support is not None:
        if cells is None:
            cells = cell_support(support)
        valid &= cells.reshape(-1)[low]
    flat = volume.reshape(-1)
    result = torch.zeros_like(q[0])
    for dx, dy, dz in _CORNERS:
        value = flat[low + (dz * n + dy) * n + dx]
        value = value * (frac[0] if dx else near[0])
        value = value * (frac[1] if dy else near[1])
        value = value * (frac[2] if dz else near[2])
        result += value
    result = torch.where(valid, result, torch.zeros_like(result))
    if reach is None:
        return result, valid
    full = torch.zeros((len(matrices), n * n * n), dtype=result.dtype, device=result.device)
    full_valid = torch.zeros((len(matrices), n * n * n), dtype=torch.bool, device=result.device)
    full[:, reach] = result
    full_valid[:, reach] = valid
    return full.reshape(-1, n, n, n), full_valid.reshape(-1, n, n, n)


def half_filter(centred_filter: torch.Tensor) -> torch.Tensor:
    """A real centred filter as `rfftn` spectra take it: its Hermitian part, halved.

    The real part of `ifftn(fftn(v) * F)` for a real `v` is `ifftn(fftn(v) * H)`
    with H(k) = (F(k) + F(-k)) / 2, and H, being symmetric, multiplies an `rfftn`
    spectrum exactly. F and H differ only where F is not point-symmetric: on a
    Nyquist plane, which has no partner, under a wedge that is not symmetric.
    """

    n = centred_filter.shape[-1]
    full = torch.fft.ifftshift(centred_filter, dim=DIMS)
    mirrored = torch.roll(torch.flip(full, dims=DIMS), shifts=(1, 1, 1), dims=DIMS)
    return ((full + mirrored) / 2)[..., : n // 2 + 1]


def _apply_half_filter(volumes: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    n = volumes.shape[-1]
    spectrum = torch.fft.rfftn(volumes, dim=DIMS) * half
    return torch.fft.irfftn(spectrum, s=(n, n, n), dim=DIMS)


def _centred(volume: torch.Tensor) -> torch.Tensor:
    """`np.roll(np.fft.fftshift(v)[::-1, ::-1, ::-1], (1, 1, 1), (0, 1, 2))` per volume.

    Its voxel i holds the uncentred map's voxel (n/2 - i) mod n along each axis.
    """

    shifted = torch.fft.fftshift(volume, dim=DIMS)
    return torch.roll(torch.flip(shifted, dims=DIMS), shifts=(1, 1, 1), dims=DIMS)


def local_correlation_batch(templates: torch.Tensor, supports: torch.Tensor,
                            particle_rfft: torch.Tensor,
                            particle_squared_rfft: torch.Tensor) -> torch.Tensor:
    """`local_correlation` for each (template, support) against one particle, uncentred.

    The particle comes as `rfftn` spectra of itself and of its square; `_centred`
    puts a result in the numpy kernel's frame.
    """

    n = templates.shape[-1]
    shape = (n, n, n)
    valid = supports
    counts = valid.sum(dim=DIMS)
    if bool((counts < 2).any()):
        raise ValueError("rotated mask has fewer than two supported voxels")
    zero = torch.zeros((), dtype=templates.dtype, device=templates.device)
    means = torch.where(valid, templates, zero).sum(dim=DIMS) / counts
    t = torch.where(valid, templates - means[:, None, None, None], zero)
    norms = torch.linalg.vector_norm(t, dim=DIMS)
    if bool((norms == 0).any()):
        raise ValueError("zero-variance rotated template")
    fm = torch.fft.rfftn(valid.to(templates.dtype), dim=DIMS)
    conj_p, conj_p2 = particle_rfft.conj(), particle_squared_rfft.conj()
    numerator = torch.fft.irfftn(torch.fft.rfftn(t, dim=DIMS) * conj_p, s=shape, dim=DIMS)
    sum_p = torch.fft.irfftn(fm * conj_p, s=shape, dim=DIMS)
    sum_p2 = torch.fft.irfftn(fm * conj_p2, s=shape, dim=DIMS)
    count = counts.to(templates.dtype)[:, None, None, None]
    variance = torch.clamp(sum_p2 - sum_p**2 / count, min=0)
    denominator = torch.sqrt(variance) * norms[:, None, None, None]
    positive = denominator > 0
    one = torch.ones((), dtype=templates.dtype, device=templates.device)
    return torch.where(positive, numerator / torch.where(positive, denominator, one), -one)


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


def _uncentred(n: int, zyx: np.ndarray) -> np.ndarray:
    """Linear indices in the uncentred map of voxels given in the centred frame."""

    raw = (n // 2 - zyx) % n
    return (raw[..., 0] * n + raw[..., 1]) * n + raw[..., 2]


def restricted_peaks(cc: torch.Tensor, shift: np.ndarray,
                     radius: float) -> tuple[np.ndarray, np.ndarray]:
    """`restricted_peak` for each uncentred map of a batch, all about the same centre.

    The maximum is taken over the sphere's voxels in the centred frame's C order,
    as numpy's `argmax` over the restricted map takes it, and the spline reads the
    winner's unrestricted neighbours. A map whose sphere holds nothing above -1,
    where the full restricted map would put its first -1 outside the sphere, is
    centred whole and read as the numpy kernel reads it.
    """

    batch, n = cc.shape[0], cc.shape[-1]
    if not 0 < radius < n / 2:
        raise ValueError("shift radius must be in (0, N/2)")
    old = _triple(shift)
    steps = (np.sign(old) * np.floor(np.abs(old) + 0.5)).astype(int)
    sphere = np.sum(_xyz(n) ** 2, axis=0) <= np.ceil(radius) ** 2
    sphere = np.roll(sphere, tuple(steps[::-1]), axis=(0, 1, 2))
    inside = np.argwhere(sphere)
    flat = cc.reshape(batch, -1)
    gathered = flat[:, torch.from_numpy(_uncentred(n, inside)).to(cc.device)]
    best = torch.argmax(gathered, dim=1)
    index = inside[best.cpu().numpy()]
    y2 = gathered.gather(1, best[:, None])[:, 0].cpu().numpy().astype(np.float64)
    degenerate = np.flatnonzero(y2 <= -1)
    if degenerate.size:
        centred = _centred(cc[torch.from_numpy(degenerate).to(cc.device)])
        restricted = torch.where(torch.from_numpy(sphere).to(cc.device), centred,
                                 torch.full_like(centred, -1.0))
        full = torch.argmax(restricted.reshape(len(degenerate), -1), dim=1).cpu().numpy()
        index[degenerate] = np.stack(np.unravel_index(full, (n, n, n)), axis=1)
        y2[degenerate] = -1.0
    neighbours = []
    for axis in range(3):
        low, high = index.copy(), index.copy()
        low[:, axis] -= 1
        high[:, axis] += 1
        neighbours.extend([low % n, high % n])
    lookup = torch.from_numpy(_uncentred(n, np.stack(neighbours, axis=1))).to(cc.device)
    values = flat.gather(1, lookup).cpu().numpy().astype(np.float64)
    location = index.astype(np.float64)
    peaks = []
    for axis in range(3):
        inner = (index[:, axis] > 0) & (index[:, axis] < n - 1)
        y1 = np.where(inner, values[:, 2 * axis], y2)
        y3 = np.where(inner, values[:, 2 * axis + 1], y2)
        value, offset = spline_peaks(y1, y2, y3)
        peaks.append(value)
        location[:, axis] += offset
    scores = np.max(np.stack(peaks), axis=0)
    scores = np.where((scores > 1) & (scores < 1.05), 1.0, scores)
    return location[:, ::-1] - n / 2, scores


@lru_cache(maxsize=64)
def _filters(n: int, high: float, low: float, tilt_range: tuple[float, float],
             dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The bandpass, and the bandpass through a tilt range's wedge, as `rfftn` spectra
    take them, on the device: every particle of a tomogram shares them, so they are
    made once instead of for each particle.

    Built on the host in float64 and narrowed there, then moved: an MPS device takes
    no float64 at all.
    """

    shell = torch.from_numpy(bandpass(n, high, low))
    filter_mask = shell * torch.from_numpy(wedge(n, tilt_range))
    return (half_filter(shell).to(dtype).to(device),
            half_filter(filter_mask).to(dtype).to(device))


@dataclass(frozen=True)
class _Shared:
    """What every particle searched against one reference uses, on the device."""

    reference: torch.Tensor
    support: torch.Tensor
    cells: torch.Tensor
    reach: torch.Tensor


def _share(reference: np.ndarray, mask: np.ndarray, shape: tuple[int, ...],
           device: torch.device) -> _Shared:
    ref, mask = _cube(reference), _cube(mask)
    if not shape == ref.shape == mask.shape or np.any(mask < 0) or not np.any(mask):
        raise ValueError("shared shape and a nonnegative/nonempty mask are required")
    support = torch.from_numpy(mask != 0).to(device)
    return _Shared(torch.from_numpy(ref.astype(np.float32)).to(device), support,
                   cell_support(support), support_reach(support))


def refine_particle_torch(
    particle: np.ndarray, reference: np.ndarray, mask: np.ndarray,
    angles: np.ndarray | Sequence[float], shift: np.ndarray | Sequence[float],
    tilt_range: tuple[float, float], settings: SearchSettings = DEFAULT_SETTINGS, *,
    device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32,
    chunk: int = DEFAULT_CHUNK,
) -> SearchResult:
    """`refine_particle`, with each level's candidates evaluated `chunk` at a time."""

    return search_results_torch([particle], reference, mask, [(angles, shift)], [tilt_range],
                                settings, device=device, dtype=dtype, chunk=chunk)[0]


def search_results_torch(
    particles: Sequence[np.ndarray], reference: np.ndarray, mask: np.ndarray,
    starts: Sequence[tuple[np.ndarray, np.ndarray]], tilt_ranges: Sequence[tuple[float, float]],
    settings: SearchSettings = DEFAULT_SETTINGS, *, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32, chunk: int = DEFAULT_CHUNK,
) -> list[SearchResult]:
    """`refine_particle_torch` for every particle against one reference, in order.

    What the particles share goes to the device once: the reference and the mask's
    support for the call, the filters for each tilt range (`_filters`). Each particle
    is then searched by the same operations on the same values as alone.
    """

    device = torch.device(device)
    shared = None
    results = []
    for particle, (angles, shift), tilt_range in zip(particles, starts, tilt_ranges, strict=True):
        p = _cube(particle)
        if shared is None:
            shared = _share(reference, mask, p.shape, device)
            if chunk < 1:
                raise ValueError("chunk must be at least 1")
        elif p.shape != tuple(shared.reference.shape):
            raise ValueError("shared shape and a nonnegative/nonempty mask are required")
        results.append(_refine(p, shared, angles, shift, tilt_range, settings,
                               device=device, dtype=dtype, chunk=chunk))
    return results


def _refine(p: np.ndarray, shared: _Shared, angles: np.ndarray | Sequence[float],
            shift: np.ndarray | Sequence[float], tilt_range: tuple[float, float],
            settings: SearchSettings, *, device: torch.device, dtype: torch.dtype,
            chunk: int) -> SearchResult:
    n = len(p)
    lo, hi = tilt_range
    shell_half, filter_half = _filters(n, float(settings.high), float(settings.low),
                                       (float(lo), float(hi)), dtype, device)
    volume = torch.from_numpy(p).to(dtype).to(device)
    volume = _apply_half_filter(_apply_half_filter(volume, shell_half), filter_half)
    volume = (volume - volume.mean()) / volume.std(correction=1)
    fp = torch.fft.rfftn(volume, dim=DIMS)
    fp2 = torch.fft.rfftn(volume * volume, dim=DIMS)

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
            rotated, valid = resample_batch(shared.reference, matrices, shared.support,
                                            cells=shared.cells, reach=shared.reach)
            rotated = _apply_half_filter(rotated.to(dtype), filter_half)
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

    return [(result.angles, result.shift, result.correlation)
            for result in search_results_torch(particles, reference, mask, starts, tilt_ranges,
                                               settings, device=device, dtype=dtype, chunk=chunk)]


_WORKER: tuple[torch.Tensor, torch.device, torch.dtype, int] | None = None


def _start_worker(particles: torch.Tensor, devices: Any, dtype_name: str, chunk: int) -> None:
    """A process's device and inputs; on a GPU, one CPU thread for torch.

    Each of torch's CPU operations wakes its whole OpenMP team and waits for every
    thread of it (ATen/ParallelOpenMP.h opens `omp parallel` without a thread count),
    so processes that each keep a team as large as the job's CPUs wait on one another:
    on max (2026-10-27, 8 cores, 16 threads a team) two processes searched 0.43 s a
    particle each, on one A40 or on two, against one process's 0.097 s. A GPU process
    leaves the host only elementwise work, the same on one thread; on the CPU the
    search's own reductions would change with the team, so it keeps it.
    """

    global _WORKER
    device = torch.device(devices.get())
    if device.type != "cpu":
        torch.set_num_threads(1)
    _WORKER = (particles, device, getattr(torch, dtype_name), chunk)


def _search_block(task: tuple[Any, ...]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    assert _WORKER is not None
    particles, device, dtype, chunk = _WORKER
    block, reference, mask, starts, tilt_ranges, settings = task
    return search_particles_torch([particles[int(i)].numpy() for i in block], reference, mask,
                                  starts, tilt_ranges, settings, device=device, dtype=dtype,
                                  chunk=chunk)


class MultiDeviceSearch:
    """`search_particles_torch` with the particles dealt among devices, a process for
    each name in `devices`: a name given twice is a device shared by two processes.

    Called as `search_particles` is. The particles are put in shared memory and
    the processes started on the first call; later calls must pass the same
    particles, in the same order, and send each process only its block's starts.
    The blocks are contiguous and in order, so the results are each particle's
    own, as one device would give them. `close` ends the processes.
    """

    def __init__(self, devices: Sequence[str], *, dtype: str = "float32",
                 chunk: int = DEFAULT_CHUNK) -> None:
        if not devices:
            raise ValueError("at least one device")
        self.devices = list(devices)
        self.dtype = dtype
        self.chunk = chunk
        self._pool: Any = None
        self._count = 0

    def _start(self, particles: Sequence[np.ndarray]) -> None:
        context = torch.multiprocessing.get_context("spawn")
        shared = torch.from_numpy(np.stack([np.asarray(p) for p in particles])).share_memory_()
        devices = context.Queue()
        for device in self.devices:
            devices.put(device)
        self._pool = context.Pool(len(self.devices), initializer=_start_worker,
                                  initargs=(shared, devices, self.dtype, self.chunk))
        self._count = len(particles)

    def __call__(self, particles: Sequence[np.ndarray], reference: np.ndarray, mask: np.ndarray,
                 starts: Sequence[tuple[np.ndarray, np.ndarray]],
                 tilt_ranges: Sequence[tuple[float, float]], settings: SearchSettings, *,
                 workers: int = 1) -> list[tuple[np.ndarray, np.ndarray, float]]:
        if self._pool is None:
            self._start(particles)
        elif len(particles) != self._count:
            raise ValueError(f"started with {self._count} particles, called with {len(particles)}")
        blocks = [block for block in np.array_split(np.arange(len(particles)), len(self.devices))
                  if len(block)]
        tasks = [(block, reference, mask, [starts[i] for i in block],
                  [tilt_ranges[i] for i in block], settings) for block in blocks]
        return [result for part in self._pool.map(_search_block, tasks) for result in part]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
