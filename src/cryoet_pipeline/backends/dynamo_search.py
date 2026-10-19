"""Dynamo's local alignment search with this pipeline's averaging: `dynamo-search`.

Why this is the default search. On the 223 CZII ribosomes, with identical
particles, pytom poses and extraction, this search left the angles neutral
where `cone-search` cost about 3 A, and it found the half voxel every particle
had been extracted off by (`particles.CROP_POSE_CONVENTION`), which no internal
control could see. Its poses were then averaged by this pipeline's Fourier-
compensated reducer, which with the same poses beat Dynamo's own averaging by
about 7 A. The measurements are in the benchmark's `DYNAMO_NATIVE_HYBRID.md`
and `DYNAMO_HYBRID_ABLATION.md`; `docs/dynamo_search.md` has the details.

One iteration: every particle is searched against the current reference, which
stays fixed while it is, then the reference is rebuilt from the raw particles
at the new poses by `subtomogram._compensated_reference`, the reducer every
other backend uses. K = 1, and not gold-standard: each particle is scored
against an average that contains it.

The search kernel below is a CPU numerical adaptation of Dynamo 1.1.511,
validated against the real program on those 223 particles (221 within the
declared tolerances; the benchmark's `DYNAMO_CORE_REPLAY.md`). It works in
Dynamo's own conventions: ZYX arrays, XYZ shifts, `[tdrot, tilt, narot]`
degrees, rotation centre `N / 2`. The backend converts native poses, whose
centre is `(N - 1) / 2`, with `native_to_dynamo` and `dynamo_to_native`; the
kernel never sees a native pose. Scope: even cubes, one wedge about y, C1,
`use_CC=1`, `localnc=1`, isotropic shift limits (modus 2), no rejection.

Numerical adaptation of the Dynamo routines listed in docs/dynamo_search.md.
The BSD option of the following original joint notice is selected for these
adaptations; this does not relicense any other Dynamo component.

Author: Daniel Castano-Diez, April 2012 (daniel.castano@unibas.ch)
Copyright (c) 2012 Daniel Castano-Diez and Henning Stahlberg
Center for Cellular Imaging and Nano Analytics
Biocenter, University of Basel

This software is issued under a joint BSD/GNU license. You may use the
source code in this file under either license. However, note that the
complete Dynamo software packages have some GPL dependencies,
so you are responsible for compliance with the licenses of these packages
if you opt to use BSD licensing. The warranty disclaimer below holds
in either instance.

This complete copyright notice must be included in any revised version of the
source code. Additional authorship citations may be added, but existing
author citations must be preserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
"""

from __future__ import annotations

import math
import multiprocessing
import os
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
import torch

from cryoet_pipeline.backends.protocols import BackendContext, variant_label, variant_suffix
from cryoet_pipeline.backends.subtomogram import (
    _MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION,
    _averaging_paths,
    _bool_parameter,
    _class_average_id,
    _class_average_path,
    _combine_particle_stacks,
    _CombinedParticles,
    _int_parameter,
    _load_reference_mrc,
    _nonnegative_float_parameter,
    _optional_float_parameter,
    _optional_path_parameter,
    _package_version,
    _path_size_bytes,
    _pixel_spacing_angstrom,
    _positive_float_parameter,
    _positive_int_parameter,
    _require_available_paths,
    _validate_particle_stack,
    _wedge_geometry_per_particle,
    wedge_ranges_by_series,
    _write_average_mrc,
    _write_json,
    averaging_scope,
    half_set_indices,
    pose_change_summary,
    torch_device_for_preference,
)
from cryoet_pipeline.fsc import compute_fsc_curve, rise_toward_nyquist
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    DynamoAverageQc,
    DynamoSearchQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import ParticleRecord, write_particle_star
from cryoet_pipeline.rotations import matrix_to_relion_euler, relion_euler_to_matrix
from cryoet_pipeline.wedge import WEDGE_BOUNDARY_POLICY


@dataclass(frozen=True)
class SearchSettings:
    cone_aperture: float = 15.0
    cone_sampling: float = 5.0
    inplane_range: float = 15.0
    inplane_sampling: float = 5.0
    refine: int = 2
    refine_factor: float = 2.0
    high: float = 2.0
    low: float = 16.0
    shift_limit: float = 3.0
    # Dynamo's limm: 2 centres the shift sphere on the preceding winner, 1 on the box.
    area_search_modus: int = 2

    def __post_init__(self):
        numbers = np.array(list(vars(self).values()), dtype=float)
        if not np.isfinite(numbers).all():
            raise ValueError("search settings must be finite")
        if not (0 < self.high < self.low and self.shift_limit > 0):
            raise ValueError("require 0 < high < low and shift_limit > 0")
        if type(self.area_search_modus) is not int or self.area_search_modus not in (1, 2):
            raise ValueError("area_search_modus must be 1 (box centre) or 2 (preceding winner)")
        if not (0 <= self.cone_aperture <= 180 and 0 <= self.inplane_range <= 360):
            raise ValueError("invalid angular range")
        if min(self.cone_sampling, self.inplane_sampling) <= 0:
            raise ValueError("sampling must be positive")
        if type(self.refine) is not int or not 0 <= self.refine <= 4 or self.refine_factor <= 1:
            raise ValueError("require integer refine in [0, 4] and refine_factor > 1")


@dataclass(frozen=True)
class SearchResult:
    angles: np.ndarray
    shift: np.ndarray
    correlation: float
    levels: tuple[dict, ...]


DEFAULT_SETTINGS = SearchSettings()


def _cube(volume):
    v = np.asarray(volume, dtype=np.float64)
    if v.ndim != 3 or len(set(v.shape)) != 1 or v.shape[0] < 4 or v.shape[0] % 2:
        raise ValueError("expected an even cubic volume, side >= 4")
    if not np.isfinite(v).all():
        raise ValueError("volume must be finite")
    return v


def _triple(values):
    a = np.asarray(values, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("expected three finite values")
    return a


@lru_cache(maxsize=4)
def _xyz(size):
    z, y, x = np.indices((size,) * 3, dtype=np.float32) - size // 2
    xyz = np.stack((x, y, z))
    xyz.flags.writeable = False
    return xyz


def euler_matrix(angles):
    """Dynamo M = Rz(narot) Rx(tilt) Rz(tdrot), in XYZ coordinates."""
    td, ti, na = np.deg2rad(_triple(angles))
    a, b, c, d, e, f = np.cos(td), np.sin(td), np.cos(ti), np.sin(ti), np.cos(na), np.sin(na)
    return np.array(
        [
            [a * e - b * c * f, -e * b - a * c * f, d * f],
            [a * f + b * c * e, -b * f + a * c * e, -d * e],
            [b * d, a * d, c],
        ]
    )


def matrix_euler(m):
    """Keep Dynamo's pole tolerance, including its effect on the search grid."""
    if abs(m[2, 2] - 1) < 1e-4:
        return np.array([0.0, 0.0, np.rad2deg(np.arctan2(m[1, 0], m[0, 0]))])
    if abs(m[2, 2] + 1) < 1e-4:
        return np.array([0.0, 180.0, np.rad2deg(np.arctan2(m[1, 0], m[0, 0]))])
    return np.rad2deg(
        [
            np.arctan2(m[2, 0], m[2, 1]),
            np.arccos(np.clip(m[2, 2], -1, 1)),
            np.arctan2(m[0, 2], -m[1, 2]),
        ]
    )


def angle_list(old_angles, aperture, sampling, inplane_range, inplane_sampling):
    """Ordered CPU Dynamo grid, including the legacy in-plane seed formula.

    This is intentionally not our cone grid. The formula can generate large
    SO(3) changes near singular directions; report them rather than hiding them.
    """
    old = _triple(old_angles)
    if old[1] < 0 or old[1] > 180:
        old = matrix_euler(euler_matrix(old))
    if min(sampling, inplane_sampling) <= 0 or min(aperture, inplane_range) < 0:
        raise ValueError("invalid grid")
    m_old = euler_matrix([old[0], old[1], 0])
    alpha, beta = euler_matrix(old)[0, 0], euler_matrix(old)[1, 0]
    rows, previous = [], None
    for polar in sampling * np.arange(int(np.floor(aperture / 2 / sampling + 1e-12)) + 1):
        if polar == 0 or polar == 180:
            azimuths = [0.0]
        else:
            co, si = np.cos(np.deg2rad(polar)), np.sin(np.deg2rad(polar))
            argument = (np.cos(np.deg2rad(sampling)) - co * co) / (si * si)
            if not -1 <= argument <= 1:
                azimuths = [0.0]
            else:
                sectors = int(np.floor(360 / np.rad2deg(np.arccos(argument)))) + 1
                azimuths = np.arange(sectors) * 360 / sectors
        for azimuth in azimuths:
            td, ti, _ = matrix_euler(euler_matrix([azimuth, polar, 0]) @ m_old)
            t, u = np.deg2rad([td, ti])
            a = alpha * np.cos(t) + beta * np.sin(t) * np.cos(u)
            b = beta * np.cos(t) - alpha * np.sin(t) * np.cos(u)
            seed = np.rad2deg(np.arctan2(b, a))
            count = int(np.floor(inplane_range / 2 / inplane_sampling + 1e-12))
            offsets = list(range(count + 1)) + list(range(-1, -count - 1, -1))
            for offset in offsets:
                row = np.array([td, ti, seed + offset * inplane_sampling])
                matrix = euler_matrix(row)
                if previous is None or np.max(np.abs(matrix - previous)) > 1e-8:
                    rows.append(row)
                    previous = matrix
    return np.array(rows)


def resample(volume, angles, shift=(0.0, 0.0, 0.0), *, mask=None, sphere=False):
    """Pull at c + M @ (x-c) - shift; return values and valid support.

    Match the MEX float32 rotation and strict eight-corner support. A nonzero
    input mask is boolean, not a soft multiplier. Undefined outputs are zero.
    """
    v = _cube(volume).astype(np.float32)
    n = v.shape[0]
    m = euler_matrix(_triple(angles).astype(np.float32)).astype(np.float32)
    d = _triple(shift).astype(np.float32)
    xyz = _xyz(n)
    q = np.stack([m[i, 0] * xyz[0] + m[i, 1] * xyz[1] + m[i, 2] * xyz[2] - d[i] for i in range(3)])
    floors = np.floor(q).astype(np.int32)
    frac = q - floors.astype(np.float32)
    base = floors + n // 2
    valid = np.all((base >= 0) & (base + 1 < n), axis=0)
    if sphere:
        valid &= np.sum(xyz * xyz, axis=0) < (n / 2) ** 2
    support = None
    if mask is not None:
        support = _cube(mask) != 0
        if support.shape != v.shape:
            raise ValueError("mask shape differs from volume")
    base = np.clip(base, 0, n - 2)
    result = np.zeros(v.shape, dtype=np.float32)
    # Order matches the eight-term sum in the CPU interpolation kernel.
    corners = [
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
    ]
    for dx, dy, dz in corners:
        idx = (base[2] + dz, base[1] + dy, base[0] + dx)
        if support is not None:
            valid &= support[idx]
        value = v[idx]
        for axis, step in enumerate((dx, dy, dz)):
            value = value * (frac[axis] if step else 1 - frac[axis])
        result += value
    result[~valid] = 0
    return result.astype(float), valid


def bandpass(size, high, low, smoothing=2.0):
    """Fourier-pixel radii; Gaussian shoulders truncated at weight 0.13."""
    if not (0 < high < low <= size / 2 and smoothing > 0):
        raise ValueError("require 0 < high < low <= Nyquist and smoothing > 0")
    r = np.sqrt(np.sum(_xyz(size).astype(float) ** 2, axis=0))
    distance = np.maximum(np.maximum(high - r, r - low), 0)
    weights = np.exp(-((distance / smoothing) ** 2))
    weights[weights <= 0.13] = 0
    return weights.astype(np.float32).astype(float)


def dynamo_ytilt(tilt_range: tuple[float, float]) -> tuple[float, float]:
    """A tomogram's tilt range as Dynamo's `ytilt`, for the volumes this pipeline stores.

    The solved angles put the measured frequencies where `wedge.py`'s beam
    direction d(t) = (sin t, 0, cos t) puts them in a stored tomogram, and `wedge`
    below, Dynamo's own, puts the same range on the mirror side. Measured
    2026-10-19: noise views over -20..+60 reconstructed by IMOD `tilt
    -PERPENDICULAR` and written as reconstruct-tomogram writes them hold a mean
    power of 6.5e6 where only `wedge(n, (-60, 20))` has support and 4.1e5 where
    only `wedge(n, (-20, 60))` has it, against 3.0e5 outside both. So Dynamo is
    given the range negated. A symmetric range is its own negation, which is why
    the Gag and CZII runs could not have shown it; the stored volume being a
    mirror image of the specimen (the z-reversing transpose) is the likely root.
    """

    low, high = tilt_range
    return -high, -low


def wedge(size, tilt_range):
    """Dynamo ftype=1, crop=1. Fourier sphere radius is N/2 + 0.5."""
    lo, hi = np.asarray(tilt_range, dtype=float)
    if not np.isfinite([lo, hi]).all() or not -90 <= lo < 0 < hi <= 90:
        raise ValueError("require a finite Y tilt range spanning zero within [-90, 90]")
    x, y, z = _xyz(size)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = z / x
    tlo, thi = np.tan(np.deg2rad([lo, hi])).astype(np.float32)
    support = ((ratio > tlo) & (ratio < thi)) | ((x == 0) & (z == 0))
    support &= (x * x + y * y + z * z) < (size / 2 + 0.5) ** 2
    return support.astype(float)


def apply_filter(volume, centered_filter):
    return np.fft.ifftn(np.fft.fftn(volume) * np.fft.ifftshift(centered_filter)).real


def normalize_roi(volume, mask=None):
    v = np.asarray(volume, dtype=float)
    selected = v.ravel() if mask is None else v[np.asarray(mask) != 0]
    if selected.size < 2 or not np.isfinite(selected).all():
        raise ValueError("normalization requires at least two finite ROI voxels")
    mean, std = selected.mean(), selected.std(ddof=1)
    if std <= 0:
        raise ValueError("zero-variance normalization ROI")
    return (v - mean) / std


def local_correlation(template, mask, particle_fft, particle_squared_fft):
    """Roseman moving-mask NCC, indexed by centered XYZ shifts in ZYX storage."""
    valid = np.asarray(mask) == 1
    count = int(valid.sum())
    if count < 2:
        raise ValueError("rotated mask has fewer than two supported voxels")
    t = np.where(valid, template - np.asarray(template)[valid].mean(), 0)
    norm = np.linalg.norm(t)
    if norm == 0:
        raise ValueError("zero-variance rotated template")
    fm = np.fft.fftn(valid.astype(float))
    numerator = np.fft.ifftn(np.fft.fftn(t) * particle_fft.conj()).real
    sum_p = np.fft.ifftn(fm * particle_fft.conj()).real
    sum_p2 = np.fft.ifftn(fm * particle_squared_fft.conj()).real
    denominator = np.sqrt(np.maximum(sum_p2 - sum_p**2 / count, 0)) * norm
    cc = np.full(t.shape, -1.0, dtype=float)
    np.divide(numerator, denominator, out=cc, where=denominator > 0)
    return np.roll(np.fft.fftshift(cc)[::-1, ::-1, ::-1], (1, 1, 1), (0, 1, 2))


def _spline_peak(y1, y2, y3):
    """The legacy three-sample spline extremum, NOT a parabolic fit.

    From dynamo_peak_subpixel_direction. Fail safely on
    degenerate no-finite-extremum cases instead of propagating MATLAB -Inf/NaN.
    """
    if y1 == y3 or abs(y2 - y1) >= 2 or abs(y2 - y3) >= 2:
        return float(y2), 0.0
    a, b, c = np.float64(y1), np.float64(y2), np.float64(y3)
    left = -(a**2) / 4 + a * c / 2 + b**2 - 2 * b * c + 3 * c**2 / 4
    right = 3 * a**2 / 4 - 2 * a * b + a * c / 2 + b**2 - c**2 / 4
    u, w = np.emath.sqrt(left), np.emath.sqrt(right)
    d, e = a + 2 * b - 3 * c, 2 * b - 3 * a + c
    with np.errstate(divide="ignore", invalid="ignore"):
        positions = np.array(
            [-2 * u / d, 2 * u / d, (e + 2 * w) / e, -(3 * a - 2 * b - c + 2 * w) / e]
        )
        values = np.array(
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
    if not good.any():
        return float(y2), 0.0
    scores = np.where(good, values.real, -np.inf)
    index = int(np.argmax(scores))
    return float(scores[index]), float(positions[index].real - (index < 2))


def restricted_peak(cc, shift, radius):
    """Integer sphere around `shift`, then subpixel peak.

    Mode 2 passes the preceding winner, so the sphere moves again at each
    refinement level, not only each run; mode 1 passes zero, the box centre.
    As in Dynamo, interpolation uses UNRESTRICTED neighbours and may leave the
    sphere.
    """
    cc = _cube(cc)
    n = cc.shape[0]
    if not 0 < radius < n / 2:
        raise ValueError("shift radius must be in (0, N/2)")
    old = _triple(shift)
    steps = (np.sign(old) * np.floor(np.abs(old) + 0.5)).astype(int)
    support = np.sum(_xyz(n) ** 2, axis=0) <= np.ceil(radius) ** 2
    support = np.roll(support, tuple(steps[::-1]), axis=(0, 1, 2))
    restricted = np.where(support, cc, -1.0)
    index = np.unravel_index(np.argmax(restricted), cc.shape)
    y2 = restricted[index]
    location = np.array(index, dtype=float)
    peaks = []
    for axis in range(3):
        low, high = list(index), list(index)
        low[axis] -= 1
        high[axis] += 1
        y1, y3 = (cc[tuple(low)], cc[tuple(high)]) if 0 < index[axis] < n - 1 else (y2, y2)
        value, offset = _spline_peak(y1, y2, y3)
        peaks.append(value)
        location[axis] += offset
    value = max(peaks)
    if 1 < value < 1.05:
        value = 1.0
    return location[::-1] - n / 2, float(value)


def refine_particle(
    particle, reference, mask, angles, shift, tilt_range, settings=DEFAULT_SETTINGS
):
    """One iteration against an immutable, already prepared reference.

    Reproduce the CPU project path's TWO bandpass applications to the particle
    (database preprocessing, then data filter); template is bandpassed once
    after rotating it and its support. No GT or oracle result enters the search.
    """
    p, ref, mask = _cube(particle), _cube(reference), _cube(mask)
    if not p.shape == ref.shape == mask.shape or np.any(mask < 0) or not np.any(mask):
        raise ValueError("shared shape and a nonnegative/nonempty mask are required")
    n = len(p)
    shell = bandpass(n, settings.high, settings.low)
    filter_mask = shell * wedge(n, tilt_range)
    p = normalize_roi(apply_filter(apply_filter(p, shell), filter_mask))
    fp, fp2 = np.fft.fftn(p), np.fft.fftn(p * p)
    best_angles, best_shift = _triple(angles).copy(), _triple(shift).copy()
    cr, cs, ir, ins = (
        settings.cone_aperture,
        settings.cone_sampling,
        settings.inplane_range,
        settings.inplane_sampling,
    )
    trace = []
    for level in range(settings.refine + 1):
        candidates = angle_list(best_angles, cr, cs, ir, ins)
        center_shift = best_shift.copy() if settings.area_search_modus == 2 else np.zeros(3)
        best_score = -np.inf
        for candidate in candidates:
            rotated, support = resample(ref, candidate, mask=mask)
            rotated = apply_filter(rotated, filter_mask)
            cc = local_correlation(rotated, support, fp, fp2)
            candidate_shift, score = restricted_peak(cc, center_shift, settings.shift_limit)
            if score >= best_score:
                best_score, best_angles, best_shift = score, candidate, candidate_shift
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


_SINGLE_THREADED = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}
_WORKER_INPUTS: tuple | None = None


def available_cpus() -> int:
    """The CPUs this process was given: SLURM's allocation, else its affinity mask.

    SLURM first because a node that does not bind tasks to cores reports every
    core in the affinity mask, and a job would then start one process per core
    of a machine it shares.
    """
    allocated = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if allocated.isdigit() and int(allocated) > 0:
        return int(allocated)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


@contextmanager
def _single_threaded_children() -> Iterator[None]:
    """Spawned workers read these at start-up; N workers should be N cores."""
    saved = {name: os.environ.get(name) for name in _SINGLE_THREADED}
    os.environ.update(_SINGLE_THREADED)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _initialize_worker(reference, mask, settings) -> None:
    global _WORKER_INPUTS
    _WORKER_INPUTS = (reference, mask, settings)


def _search(task, reference, mask, settings):
    result = refine_particle(task[0], reference, mask, *task[1:], settings)
    return result.angles, result.shift, result.correlation


def _search_task(task):
    return _search(task, *_WORKER_INPUTS)


def search_particles(particles, reference, mask, starts, tilt_ranges, settings, *, workers=1):
    """`refine_particle` for every particle against one fixed reference, in order.

    The reference does not change until every particle is searched, so the
    particles are independent and can be shared among processes. Each is still
    searched by the same code on the same float64 inputs, and nothing in the
    search sums across threads (the rotation is elementwise, NumPy's FFT is
    single-threaded), so the result does not depend on `workers`. Workers are
    spawned, not forked, because a fork after torch has started its thread pools
    can deadlock; a script that reaches this with `workers > 1` therefore needs
    an `if __name__ == "__main__":` guard, as spawned workers import it again.
    """
    tasks = [
        (particle, angles, shift, tilt_range)
        for particle, (angles, shift), tilt_range in zip(
            particles, starts, tilt_ranges, strict=True
        )
    ]
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if workers == 1 or len(tasks) < 2:
        return [_search(task, reference, mask, settings) for task in tasks]
    workers = min(workers, len(tasks))
    with (
        _single_threaded_children(),
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(reference, mask, settings),
        ) as pool,
    ):
        return list(pool.map(_search_task, tasks, chunksize=max(1, len(tasks) // (4 * workers))))


# The fsc=true merge in Dynamo 1.1.511 hard-codes 0.1, ignoring fmin.
FOURIER_WEIGHT_THRESHOLD = 0.1


def average_particles(particles: Iterable[np.ndarray], angles, shifts, tilt_ranges, mask):
    """Raw-data average and Fourier compensation; no search/interpolation reuse.

    Use the same helper for fixed and refined poses to avoid comparing different
    averaging estimators. The `dynamo-search` backend averages every iteration
    with it (replayed against Dynamo: 8e-5 on refine223, 6.5e-5 on sel45_k1).
    Normalization uses all NONZERO mask voxels (ddof=1).
    Match the CPU project's fsc=true assembly: normalize odd/even raw averages
    separately, merge them equally, compensate at weight >= 0.1, and do NOT
    re-normalize the final map. These are not independently refined half-maps.
    Returns compensated average, unweighted average, and merged weights.
    """
    mask = _cube(mask)
    rows = list(zip(angles, shifts, tilt_ranges, strict=True))
    if len(rows) < 2 or np.any(mask < 0) or not np.any(mask):
        raise ValueError("at least two particles and a nonnegative/nonempty mask required")
    total = np.zeros((2, *mask.shape))
    weights, contributions = np.zeros_like(total), np.zeros_like(total)
    count = 0
    for particle, (angle, shift, tilts) in zip(particles, rows, strict=True):
        if _cube(particle).shape != mask.shape:
            raise ValueError("particle and mask shapes differ")
        inverse_angles = -_triple(angle)[::-1]
        aligned, valid = resample(particle, inverse_angles, -_triple(shift))
        aligned = normalize_roi(aligned, mask)
        aligned[~valid] = 0
        half = count % 2
        total[half] += aligned
        contributions[half] += valid
        rotated_wedge, _ = resample(wedge(len(mask), tilts), inverse_angles, sphere=True)
        weights[half] += rotated_wedge
        count += 1
    # Dynamo's explicitBalanceNan defaults to 1: count only valid resamplings.
    raw_halves = np.divide(total, contributions, out=np.zeros_like(total), where=contributions > 0)
    raw = (normalize_roi(raw_halves[0], mask) + normalize_roi(raw_halves[1], mask)) / 2
    weights = (weights[0] / weights[0].max() + weights[1] / weights[1].max()) / 2
    good = weights >= FOURIER_WEIGHT_THRESHOLD
    spectrum = np.fft.fftshift(np.fft.fftn(raw))
    corrected = np.zeros_like(spectrum)
    np.divide(spectrum, weights, out=corrected, where=good)
    average = np.fft.ifftn(np.fft.ifftshift(corrected)).real
    return average, raw, weights


@dataclass(frozen=True)
class ClassAverage:
    average: np.ndarray
    """Compensated, then normalized over the mask: the channel's next reference, or
    the backend's starting one."""
    raw: np.ndarray
    fweight: np.ndarray
    fmask: np.ndarray
    threshold: float
    members: int


def fweight_divide(volume, fweight, threshold: float) -> np.ndarray:
    """`dynamo_fweight_divide`: divide the centred spectrum by the weight where the
    weight reaches `threshold`, and zero every other component."""
    spectrum = np.fft.fftshift(np.fft.fftn(np.asarray(volume, dtype=float)))
    fweight = np.asarray(fweight, dtype=float)
    keep = np.isfinite(fweight) & (fweight >= threshold)
    divided = np.zeros_like(spectrum)
    np.divide(spectrum, fweight, out=divided, where=keep)
    return np.fft.ifftn(np.fft.ifftshift(divided)).real


def average_class(particles, angles, shifts, tilt_ranges, nmask, *, fmin: int = 1) -> ClassAverage:
    """A class average as the MRA plugin asks `dynamo_average` for it.

    Also the console call `dynamo_average(..., 'fcompensate', true, 'fmin', 1,
    'nmask', mask)` that makes a project's starting template from the initial
    poses (sel45_avg.em, replayed to 4e-7), and so the `dynamo-search` backend's
    starting reference when none is given.

    Per particle, the path of the project's own averaging, which the search
    kernel's `average_particles` reproduces (replay: 8e-5 against Dynamo): the
    inverse pose, out-of-box voxels zeroed, normalization over the mask's nonzero
    voxels, zeroed again, and the table wedge rotated with the implicit sphere
    for the Fourier weight. Unlike that `fsc = true` assembly: one sum over all
    members in their given order (ascending tags in Dynamo), no odd/even halves;
    the compensation threshold `fmin / N` of the max-normalized weight, not 0.1;
    and the compensated average normalized over the mask once more.
    """
    nmask = _cube(nmask)
    if np.any(nmask < 0) or not np.any(nmask):
        raise ValueError("a class needs at least one particle and a nonnegative, nonempty mask")
    if type(fmin) is not int or fmin < 1:
        raise ValueError(f"fmin counts particles and must be a positive integer, got {fmin!r}")
    total = np.zeros(nmask.shape)
    contributions = np.zeros(nmask.shape)
    weights = np.zeros(nmask.shape)
    count = 0
    # One particle at a time, so `particles` may be a generator reading them from disk.
    for particle, angle, shift, tilts in zip(particles, angles, shifts, tilt_ranges, strict=True):
        if _cube(particle).shape != nmask.shape:
            raise ValueError("particle and mask shapes differ")
        inverse_angles = -_triple(angle)[::-1]
        aligned, valid = resample(particle, inverse_angles, -_triple(shift))
        aligned = normalize_roi(aligned, nmask)
        aligned[~valid] = 0
        total += aligned
        contributions += valid
        rotated_wedge, _ = resample(wedge(len(nmask), tilts), inverse_angles, sphere=True)
        weights += rotated_wedge
        count += 1
    if not count:
        raise ValueError("a class needs at least one particle and a nonnegative, nonempty mask")
    # explicitBalanceNan: every voxel divided by the particles that really reached it.
    raw = np.divide(total, contributions, out=np.zeros_like(total), where=contributions > 0)
    raw = normalize_roi(raw, nmask)
    fweight = weights / weights.max()
    threshold = fmin / count if fmin < count else 1 / count
    average = normalize_roi(fweight_divide(raw, fweight, threshold), nmask)
    return ClassAverage(
        average, raw, fweight, (fweight >= threshold).astype(float), threshold, count
    )


def _dynamo_centre_offset(box: int) -> np.ndarray:
    """`cD - cN`: Dynamo rotates about `N / 2`, the native convention about `(N - 1) / 2`."""
    return np.full(3, box / 2 - (box - 1) / 2)


def dynamo_angles_from_matrix(matrix) -> np.ndarray:
    """Exact `[tdrot, tilt, narot]` with `euler_matrix(angles) == matrix`.

    Not `matrix_euler`: that keeps Dynamo's 1e-4 pole tolerance, which belongs
    inside the search grid and would move a near-pole pose by up to 0.8 degrees
    if used to convert one.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.isfinite(m).all():
        raise ValueError("expected a finite 3x3 rotation")
    tilt = math.degrees(math.acos(float(np.clip(m[2, 2], -1.0, 1.0))))
    if math.hypot(m[2, 0], m[2, 1]) > 1e-12:
        angles = np.array(
            [
                math.degrees(math.atan2(m[2, 0], m[2, 1])),
                tilt,
                math.degrees(math.atan2(m[0, 2], -m[1, 2])),
            ]
        )
    else:
        pole = 0.0 if m[2, 2] > 0 else 180.0
        angles = np.array([0.0, pole, math.degrees(math.atan2(m[1, 0], m[0, 0]))])
    if not np.allclose(euler_matrix(angles), m, rtol=0, atol=1e-9):
        raise ValueError("the matrix has no exact Dynamo Euler form; is it a proper rotation?")
    return angles


def native_to_dynamo(matrix, shift, box: int) -> tuple[np.ndarray, np.ndarray]:
    """Native `x_in = cN + A (x_out - cN) + s` as Dynamo's `(angles, d)`.

    The same pull map written about Dynamo's centre: `M = A.T` and
    `d = s + (A - I) (cD - cN)`, the conversion the benchmark's exporter uses and
    its tests pin at the poles and with nonzero shifts.
    """
    a = np.asarray(matrix, dtype=float)
    offset = _dynamo_centre_offset(box)
    return dynamo_angles_from_matrix(a.T), np.asarray(shift, dtype=float) + (a - np.eye(3)) @ offset


def dynamo_to_native(angles, shift, box: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `native_to_dynamo`."""
    a = euler_matrix(angles).T
    return a, np.asarray(shift, dtype=float) - (a - np.eye(3)) @ _dynamo_centre_offset(box)


def prepare_reference(template, mask, settings: SearchSettings = DEFAULT_SETTINGS) -> np.ndarray:
    """What Dynamo's iteration setup makes of a reference before its search uses it.

    `+dpkproject/preprocessReferences.m` runs `dynamo_database_volume_process(
    template, 'template', ..., {'sym','dim','band','norm'})` on every starting
    reference, the first and every rebuilt one. With C1 and no resizing that is
    the project bandpass, the one particles also get in database preprocessing,
    then normalization over the project's `mask` item, the alignment mask
    thresholded at 0.5. On refine223 it rebuilds
    `starting_reference_transformed_ref_001_ite_0001.em` from the project's
    `template_initial_ref_001.em` to 2.4e-8 relative L2 (float32 storage);
    normalizing over the soft mask's nonzero voxels instead leaves 0.33. Left
    out, it moved all 25 of 25 refine223 poses off Dynamo's, by a median of one
    finest-level step.
    """
    t = _cube(template)
    return normalize_roi(
        apply_filter(t, bandpass(len(t), settings.high, settings.low)), np.asarray(mask) > 0.5
    )


def search_mask(box: int, diameter_px: float) -> np.ndarray:
    """The binary alignment mask, a sphere about the particle's centre `(N - 1) / 2`.

    Native poses put the particle at `cN`, so that is where the reference holds
    it; the kernel rotates the mask with the reference, which keeps the two
    together. A sphere about the kernel's rotation centre `N / 2` would sit
    (0.5, 0.5, 0.5) voxels off the particle: on 25 of refine223's particles,
    with Dynamo's reference, that alone moved all 25 poses off Dynamo's, by a
    median of one finest-level step. refine223's own mask is this sphere at
    39 px, voxel for voxel. Dynamo thresholds its alignment mask at >0.5 before
    searching; a binary sphere is what that leaves.
    """
    if not 0 < diameter_px <= box:
        raise ValueError(f"alignment mask diameter {diameter_px:.2f} px must fit in a {box} px box")
    offset = np.arange(box) - (box - 1) / 2
    z, y, x = np.meshgrid(offset, offset, offset, indexing="ij")
    return (np.sqrt(x * x + y * y + z * z) < diameter_px / 2).astype(float)


def grid_reach_deg(settings: SearchSettings) -> float:
    """The widest rotation one iteration's levels can compose to, by the triangle bound.

    Each level moves at most half its cone aperture plus half its in-plane range
    from the previous winner, and `refine_particle` narrows both after every
    level. A particle that moved further than this took the legacy in-plane seed.
    """
    aperture, sampling = settings.cone_aperture, settings.cone_sampling
    inplane, inplane_sampling = settings.inplane_range, settings.inplane_sampling
    reach = 0.0
    factor = settings.refine_factor
    for _ in range(settings.refine + 1):
        reach += aperture / 2 + inplane / 2
        aperture, inplane = factor * sampling, factor * inplane_sampling
        sampling, inplane_sampling = sampling / factor, inplane_sampling / factor
    return reach


def _rotation_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    cosine = (float(np.trace(first @ second.T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _dynamo_poses(matrices, shifts, box: int):
    poses = [native_to_dynamo(m, s, box) for m, s in zip(matrices, shifts, strict=True)]
    return [angles for angles, _ in poses], [shift for _, shift in poses]


def _project_average(particles, matrices, shifts, tilt_ranges, mask, box: int):
    """Dynamo's project averaging (`average_particles`) at native poses: map and weights."""
    angles, dynamo_shifts = _dynamo_poses(matrices, shifts, box)
    average, _, fweight = average_particles(particles, angles, dynamo_shifts, tilt_ranges, mask)
    return average, fweight


def _fourier_weight_summary(
    fweight: np.ndarray, threshold: float = FOURIER_WEIGHT_THRESHOLD
) -> dict[str, float]:
    """The merged weight inside the Nyquist sphere; the cube's corners are always empty."""
    radius = np.sqrt(np.sum(_xyz(len(fweight)).astype(float) ** 2, axis=0))
    inside = fweight[radius < len(fweight) / 2]
    return {
        "minimum": float(inside.min()),
        "median": float(np.median(inside)),
        "fraction_below_threshold": float(np.mean(inside < threshold)),
    }


def _load_alignment_mask(path: Path, box: int) -> np.ndarray:
    """A Dynamo alignment mask: its voxels above 0.5 are searched, its nonzero ones
    normalise each particle before averaging."""
    with mrcfile.open(path, permissive=True) as handle:
        mask = np.array(handle.data, dtype=np.float64)
    if mask.shape != (box, box, box):
        raise ValueError(f"alignment mask {path} is {mask.shape}, the particles {(box,) * 3}")
    if not np.isfinite(mask).all() or np.any(mask < 0):
        raise ValueError(f"alignment mask {path} must be finite and nonnegative")
    if np.count_nonzero(mask > 0.5) < 2:
        raise ValueError(f"alignment mask {path} has no voxels above 0.5 to search")
    return mask


INITIAL_REFERENCE_DEFAULT = "dynamo_average of the initial poses (fcompensate, fmin 1)"
GOLD_STANDARD_START_LOWPASS_ANGSTROM = 60.0
"""The common start of the halves is band-passed as `dynamo_bandpass(v, [1 low 2])`.

60 A is the CZII session's `[1 8 2]` at a 48-voxel box of 10.012 A: below it the
halves share the start, above it each half builds its own map from its own data.
"""


def _refine(raw, matrices, shifts, tilt_ranges, reference, mask, normalization_mask,
            settings, *, iterations, workers, box, half=None):
    """`iterations` rounds of searching every given particle against a fixed reference
    and rebuilding it from them: Dynamo's project iteration, on whatever subset it gets.

    Returns the new poses, the search CCs, the last reference and its Fourier weight,
    and one summary per iteration.
    """
    matrices = [m.copy() for m in matrices]
    shifts = [np.asarray(v, dtype=float).copy() for v in shifts]
    correlations = [math.nan] * len(raw)
    fweight = None
    summaries: list[dict[str, float]] = []
    for iteration in range(iterations):
        # Prepared as Dynamo prepares every starting reference, the first and
        # every rebuilt one, before the search sees it.
        fixed = prepare_reference(reference, mask, settings)
        before = [m.copy() for m in matrices], [v.copy() for v in shifts]
        results = search_particles(
            raw,
            fixed,
            mask,
            [native_to_dynamo(m, v, box) for m, v in zip(matrices, shifts, strict=True)],
            tilt_ranges,
            settings,
            workers=workers,
        )
        for index, (angles, shift, correlation) in enumerate(results):
            matrices[index], shifts[index] = dynamo_to_native(angles, shift, box)
            correlations[index] = correlation
        reference, fweight = _project_average(
            raw, matrices, shifts, tilt_ranges, normalization_mask, box
        )
        summary = {
            "iteration": float(iteration + 1),
            "mean_correlation": float(np.mean(correlations)),
            "minimum_correlation": float(np.min(correlations)),
            "rotation_deg_mean": float(
                np.mean([
                    _rotation_between_deg(a, b)
                    for a, b in zip(before[0], matrices, strict=True)
                ])
            ),
            "shift_px_mean": float(
                np.mean([np.linalg.norm(a - b) for a, b in zip(before[1], shifts, strict=True)])
            ),
        }
        if half is not None:
            summary["half"] = float(half)
        summaries.append(summary)
    return matrices, shifts, correlations, reference, fweight, summaries


def _half_map_fsc(half_maps, mask, spacing: float) -> dict[str, object]:
    """FSC between the two half maps, with the alignment mask (applied as `compute-fsc
    --mask` applies one) and without, read at 0.5 and 0.143, with any rise toward
    Nyquist (`fsc.rise_toward_nyquist`)."""
    report: dict[str, object] = {}
    for label, weight in (("masked", mask), ("unmasked", None)):
        a, b = (h if weight is None else h * weight for h in (half_maps[1], half_maps[2]))
        a, b = torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(np.ascontiguousarray(b))
        curves = {t: compute_fsc_curve(a, b, pixel_spacing_angstrom=spacing, threshold=t)
                  for t in (0.5, 0.143)}
        rises = {t: rise_toward_nyquist(curves[t], len(mask)) for t in curves}
        rise = rises[0.143]
        report[label] = {
            "resolution_angstrom": curves[0.5].resolution_angstrom,
            "correlation": curves[0.5].correlation,
            "crossing_0_5_angstrom": curves[0.5].crossing_resolution_angstrom,
            "crossing_0_143_angstrom": curves[0.143].crossing_resolution_angstrom,
            "rise_toward_nyquist": None if rise is None else {
                "dip_resolution_angstrom": rise.dip_resolution_angstrom,
                "dip_correlation": rise.dip_correlation,
                "peak_resolution_angstrom": rise.peak_resolution_angstrom,
                "peak_correlation": rise.peak_correlation,
                "sigma": rise.sigma,
                # The final descent lands beyond the dip, or nowhere; these do not.
                "crossing_before_dip_0_5_angstrom": rises[0.5].crossing_before_dip_angstrom,
                "crossing_before_dip_0_143_angstrom": rise.crossing_before_dip_angstrom,
            },
        }
    return report


@dataclass(frozen=True)
class _DynamoInputs:
    """What both Dynamo backends average: the particles, their poses, the mask, the wedges."""

    pairs: list[tuple[Artifact, TiltSeriesManifest]]
    combined: _CombinedParticles
    box: int
    spacing: float
    mask_path: Path | None
    mask_diameter: float | None
    """The normalisation sphere's diameter in A; None when the mask is a file."""
    normalization_mask: np.ndarray
    wedge_ranges: list[tuple[float, float]]
    """The distinct tilt ranges, as the QC reports them."""
    records: list[ParticleRecord]
    matrices: list[np.ndarray]
    shifts: list[np.ndarray]
    raw: list[np.ndarray]
    tilt_ranges: list[tuple[float, float]]
    """One per particle, from its own series."""


def _dynamo_inputs(
    backend: str,
    particles: Sequence[Artifact],
    manifests: Sequence[TiltSeriesManifest],
    context: BackendContext,
    device: torch.device,
) -> _DynamoInputs:
    """Load and check the stacks, poses, mask and wedges for a Dynamo backend."""

    if len(particles) != len(manifests):
        raise ValueError(
            f"{len(particles)} particle stacks were given for {len(manifests)} "
            "manifests; each stack needs the manifest of the series it came from"
        )
    if not particles:
        raise ValueError("averaging needs at least one particle stack")
    for stack, stack_manifest in zip(particles, manifests, strict=True):
        _validate_particle_stack(stack, stack_manifest)
    pairs = sorted(
        zip(particles, manifests, strict=True),
        key=lambda pair: pair[1].tilt_series_id,
    )
    for key, allowed in (("num_classes", 1), ("alignment_frame", "reference")):
        value = context.parameters.get(key)
        if value is not None and value != allowed:
            raise ValueError(f"{backend} does not support {key}={value!r}")
    for key in ("particle_metadata_path", "class_number"):
        if context.parameters.get(key) is not None:
            raise ValueError(f"{backend} does not read {key}; start from the stack's poses")
    if _nonnegative_float_parameter(context, "wedge_tilt_axis_deg", default=0.0) != 0.0:
        raise ValueError(f"{backend} models one wedge about y; the tilt axis must be 0")

    combined = _combine_particle_stacks(pairs, device=device)
    box = combined.box_size
    if box < 4 or box % 2:
        raise ValueError(f"{backend} needs an even box of at least 4 voxels, got {box}")
    spacing = _pixel_spacing_angstrom(pairs[0][0])
    particle_diameter = _positive_float_parameter(
        context, "particle_diameter_angstrom", default=box * spacing
    )
    mask_path = _optional_path_parameter(context, "alignment_mask_path")
    mask_diameter = _optional_float_parameter(context, "alignment_mask_diameter_angstrom")
    if mask_path is not None:
        if mask_diameter is not None:
            raise ValueError(
                f"{backend} takes the alignment mask as a file or as a diameter, not both"
            )
        # One mask, two uses, as in Dynamo: thresholded for the search, its
        # whole support for normalising the particles before averaging.
        normalization_mask = _load_alignment_mask(mask_path, box)
    else:
        mask_diameter = mask_diameter or particle_diameter
        normalization_mask = search_mask(box, mask_diameter / spacing)
    geometries = _wedge_geometry_per_particle(
        pairs,
        combined,
        min_tilt_override=_optional_float_parameter(context, "wedge_min_tilt_deg"),
        max_tilt_override=_optional_float_parameter(context, "wedge_max_tilt_deg"),
        tilt_axis_deg=0.0,
        ranges_by_series=wedge_ranges_by_series(context),
    )
    wedge_ranges = sorted({(g.min_tilt_deg, g.max_tilt_deg) for g in geometries})
    for low, high in wedge_ranges:
        if not low < 0 < high:
            raise ValueError(
                f"{backend}'s wedge needs a tilt range spanning zero, got [{low}, {high}]"
            )
    records = list(combined.records)
    if [record.index for record in records] != list(range(len(records))):
        raise ValueError("combined particle records are not in stack order")
    return _DynamoInputs(
        pairs=pairs,
        combined=combined,
        box=box,
        spacing=spacing,
        mask_path=mask_path,
        mask_diameter=mask_diameter,
        normalization_mask=normalization_mask,
        wedge_ranges=wedge_ranges,
        records=records,
        matrices=[relion_euler_to_matrix(r.rot_deg, r.tilt_deg, r.psi_deg) for r in records],
        shifts=[np.array([r.shift_x_px, r.shift_y_px, r.shift_z_px]) for r in records],
        raw=[volume.detach().cpu().numpy().astype(np.float64) for volume in combined.volumes],
        tilt_ranges=[dynamo_ytilt((g.min_tilt_deg, g.max_tilt_deg)) for g in geometries],
    )


class DynamoSearchRefinementBackend:
    """Dynamo's local search and Dynamo's project averaging. The default backend."""

    name = "dynamo-search"

    def average(
        self,
        particles: Sequence[Artifact],
        manifests: Sequence[TiltSeriesManifest],
        context: BackendContext,
    ) -> list[Artifact]:
        """Return the refined average and a QC artifact; the metadata sits beside them."""

        num_iterations = _positive_int_parameter(context, "num_iterations", default=1)
        workers = _positive_int_parameter(context, "workers", default=available_cpus())
        settings = SearchSettings(
            cone_aperture=_nonnegative_float_parameter(
                context, "cone_range_deg", default=DEFAULT_SETTINGS.cone_aperture
            ),
            cone_sampling=_positive_float_parameter(
                context, "cone_sampling_deg", default=DEFAULT_SETTINGS.cone_sampling
            ),
            inplane_range=_nonnegative_float_parameter(
                context, "inplane_range_deg", default=DEFAULT_SETTINGS.inplane_range
            ),
            inplane_sampling=_positive_float_parameter(
                context, "inplane_sampling_deg", default=DEFAULT_SETTINGS.inplane_sampling
            ),
            refine=_int_parameter(context, "refine_levels", default=DEFAULT_SETTINGS.refine),
            refine_factor=_positive_float_parameter(
                context, "refine_factor", default=DEFAULT_SETTINGS.refine_factor
            ),
            high=_positive_float_parameter(
                context, "search_highpass_px", default=DEFAULT_SETTINGS.high
            ),
            low=_positive_float_parameter(
                context, "search_lowpass_px", default=DEFAULT_SETTINGS.low
            ),
            shift_limit=_positive_float_parameter(
                context, "shift_range_px", default=DEFAULT_SETTINGS.shift_limit
            ),
            area_search_modus=_int_parameter(
                context, "shift_search_mode", default=DEFAULT_SETTINGS.area_search_modus
            ),
        )
        initial_reference_path = _optional_path_parameter(context, "initial_reference_path")
        gold_standard = _bool_parameter(context, "gold_standard", default=False)
        start_lowpass = _optional_float_parameter(context, "gold_standard_start_lowpass_angstrom")
        if start_lowpass is not None and not gold_standard:
            raise ValueError("gold_standard_start_lowpass_angstrom is read only with gold_standard")
        overwrite = _bool_parameter(context, "overwrite", default=False)
        device = torch_device_for_preference(context.device)

        inputs = _dynamo_inputs(self.name, particles, manifests, context, device)
        pairs, combined, box, spacing = inputs.pairs, inputs.combined, inputs.box, inputs.spacing
        if settings.low > box / 2:
            raise ValueError(f"search_lowpass_px {settings.low} lies beyond Nyquist ({box / 2})")
        mask_path, mask_diameter = inputs.mask_path, inputs.mask_diameter
        normalization_mask = inputs.normalization_mask
        mask = (normalization_mask > 0.5).astype(float)
        ranges = inputs.wedge_ranges
        scope = averaging_scope(combined.tilt_series_ids)
        paths = _averaging_paths(
            context.output_dir, scope, backend_name=self.name, label=variant_label(context)
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        records = list(inputs.records)
        starting_poses = [
            (r.rot_deg, r.tilt_deg, r.psi_deg, r.shift_x_px, r.shift_y_px, r.shift_z_px)
            for r in records
        ]
        matrices, shifts = list(inputs.matrices), list(inputs.shifts)
        raw, tilt_ranges = inputs.raw, inputs.tilt_ranges
        if initial_reference_path is not None:
            reference = (
                _load_reference_mrc(initial_reference_path, expected_box_size=box, device=device)
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            initial_reference = str(initial_reference_path)
        else:
            angles, dynamo_shifts = _dynamo_poses(matrices, shifts, box)
            reference = average_class(
                raw, angles, dynamo_shifts, tilt_ranges, normalization_mask, fmin=1
            ).average
            initial_reference = INITIAL_REFERENCE_DEFAULT

        half_maps: dict[int, np.ndarray] = {}
        half_counts: dict[int, int] = {}
        if gold_standard:
            # Dynamo's half-set scheme: one start for both halves, band-passed so they
            # share nothing beyond it, then each half refined from its own data alone.
            start_lowpass = start_lowpass or GOLD_STANDARD_START_LOWPASS_ANGSTROM
            low_px = box * spacing / start_lowpass
            if not 1.0 < low_px <= box / 2:
                raise ValueError(
                    f"a {start_lowpass:g} A start low-pass is {low_px:.2f} Fourier pixels in "
                    f"this {box}-voxel box; it has to lie between 1 and {box // 2}"
                )
            reference = apply_filter(reference, bandpass(box, 1.0, low_px, 2.0))
            halves = half_set_indices(records)
            half_counts = {half: len(indices) for half, indices in halves.items()}
            if min(half_counts.values()) < 2:
                raise ValueError(
                    "gold_standard needs at least two particles in each half; "
                    f"rlnRandomSubset puts {half_counts[1]} and {half_counts[2]}"
                )
            correlations = [math.nan] * len(records)
            iteration_summaries: list[dict[str, float]] = []
            for half, indices in sorted(halves.items()):
                found, moved_to, found_cc, half_maps[half], _, summaries = _refine(
                    [raw[i] for i in indices],
                    [matrices[i] for i in indices],
                    [shifts[i] for i in indices],
                    [tilt_ranges[i] for i in indices],
                    reference,
                    mask,
                    normalization_mask,
                    settings,
                    iterations=num_iterations,
                    workers=workers,
                    box=box,
                    half=half,
                )
                for position, index in enumerate(indices):
                    matrices[index], shifts[index] = found[position], moved_to[position]
                    correlations[index] = found_cc[position]
                iteration_summaries.extend(summaries)
            # The map is every particle at the pose its own half gave it.
            reference, fweight = _project_average(
                raw, matrices, shifts, tilt_ranges, normalization_mask, box
            )
        else:
            matrices, shifts, correlations, reference, fweight, iteration_summaries = _refine(
                raw,
                matrices,
                shifts,
                tilt_ranges,
                reference,
                mask,
                normalization_mask,
                settings,
                iterations=num_iterations,
                workers=workers,
                box=box,
            )

        for index, record in enumerate(records):
            rot, tilt, psi = matrix_to_relion_euler(matrices[index])
            records[index] = replace(
                record,
                rot_deg=rot,
                tilt_deg=tilt,
                psi_deg=psi,
                shift_x_px=float(shifts[index][0]),
                shift_y_px=float(shifts[index][1]),
                shift_z_px=float(shifts[index][2]),
                class_number=1,
            )
        write_particle_star(paths.particle_metadata, records)
        average_path = _class_average_path(paths, 1, num_classes=1)
        _write_average_mrc(
            average_path,
            reference,
            pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
        )
        half_map_fsc = None
        for half, half_map in half_maps.items():
            _write_average_mrc(
                paths.half_maps[half],
                half_map,
                pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
            )
        if half_maps:
            half_map_fsc = _half_map_fsc(half_maps, normalization_mask, spacing)

        reach = grid_reach_deg(settings)
        starting = [relion_euler_to_matrix(*pose[:3]) for pose in starting_poses]
        moved = [_rotation_between_deg(a, b) for a, b in zip(starting, matrices, strict=True)]
        beyond = [angle for angle in moved if angle > reach * num_iterations + 1e-6]
        warnings: list[str] = []
        minimum = float(np.min(correlations))
        if minimum < 0.0:
            warnings.append(
                f"minimum final correlation {minimum:.3f} is negative; some particles "
                "never matched the reference"
            )
        if beyond:
            warnings.append(
                f"{len(beyond)} particles moved further than the search grid reaches "
                f"({reach * num_iterations:.1f} deg), up to {max(beyond):.1f} deg: Dynamo's "
                "legacy in-plane seed, kept as Dynamo keeps it"
            )
        if half_counts and min(half_counts.values()) < _MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION:
            warnings.append(
                f"a half holds {min(half_counts.values())} particles, fewer than "
                f"{_MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION}: its FSC is reported but not "
                "interpretable as a resolution"
            )
        for label, curve in (half_map_fsc or {}).items():
            rise = curve["rise_toward_nyquist"]
            if rise is not None:
                before = [
                    f"{threshold} at {rise[key]:.1f} A"
                    for threshold, key in (
                        ("0.5", "crossing_before_dip_0_5_angstrom"),
                        ("0.143", "crossing_before_dip_0_143_angstrom"),
                    )
                    if rise[key] is not None
                ]
                warnings.append(
                    f"the {label} half-map FSC rises again toward Nyquist, from "
                    f"{rise['dip_correlation']:.2f} at {rise['dip_resolution_angstrom']:.1f} A to "
                    f"{rise['peak_correlation']:.2f} at {rise['peak_resolution_angstrom']:.1f} A "
                    f"({rise['sigma']:.0f} sigma of shell noise), so its crossings are read "
                    "beyond the dip or not at all"
                    + (f"; before the dip it fell through {', '.join(before)}" if before else "")
                )
        status = QcStatus.WARNING if warnings else QcStatus.PASS
        report = DynamoSearchQc(
            backend=self.name,
            tilt_series_ids=combined.tilt_series_ids,
            particle_stack_ids=combined.stack_ids,
            particle_count=len(records),
            particle_counts=combined.counts,
            iterations_run=num_iterations,
            workers=workers,
            cone_aperture_deg=settings.cone_aperture,
            cone_sampling_deg=settings.cone_sampling,
            inplane_range_deg=settings.inplane_range,
            inplane_sampling_deg=settings.inplane_sampling,
            refine_levels=settings.refine,
            refine_factor=settings.refine_factor,
            search_highpass_px=settings.high,
            search_lowpass_px=settings.low,
            shift_limit_px=settings.shift_limit,
            shift_search_mode=settings.area_search_modus,
            alignment_mask_diameter_angstrom=mask_diameter,
            alignment_mask_path=None if mask_path is None else str(mask_path),
            search_mask_voxels=int(np.count_nonzero(mask)),
            normalization_mask_voxels=int(np.count_nonzero(normalization_mask)),
            grid_reach_deg=reach,
            wedge_tilt_ranges_deg=[list(one) for one in ranges],
            dynamo_ytilt_ranges_deg=[list(dynamo_ytilt(one)) for one in ranges],
            wedge_boundary_policy=WEDGE_BOUNDARY_POLICY,
            fourier_weight_threshold=FOURIER_WEIGHT_THRESHOLD,
            fourier_weight=None if fweight is None else _fourier_weight_summary(fweight),
            initial_reference=initial_reference,
            gold_standard=gold_standard,
            gold_standard_start_lowpass_angstrom=start_lowpass if gold_standard else None,
            half_counts={str(h): n for h, n in half_counts.items()} or None,
            half_map_fsc=half_map_fsc,
            mean_final_correlation=float(np.mean(correlations)),
            minimum_final_correlation=minimum,
            iterations=iteration_summaries,
            pose_change=pose_change_summary(starting_poses, records),
            beyond_grid_count=len(beyond),
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))
        average_id = _class_average_id(
            scope, self.name, 1, num_classes=1, variant=variant_suffix(context)
        )
        average = Artifact(
            id=average_id,
            variant=variant_label(context),
            kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
            path=average_path,
            parent_ids=list(combined.stack_ids),
            shape=(box, box, box),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
            parameters={
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
                "particle_stack_ids": list(combined.stack_ids),
                "class_number": 1,
                "num_classes": 1,
                "particle_count": len(records),
                "iterations_run": num_iterations,
                "workers": workers,
                "cone_aperture_deg": settings.cone_aperture,
                "cone_sampling_deg": settings.cone_sampling,
                "inplane_range_deg": settings.inplane_range,
                "inplane_sampling_deg": settings.inplane_sampling,
                "refine_levels": settings.refine,
                "refine_factor": settings.refine_factor,
                "search_highpass_px": settings.high,
                "search_lowpass_px": settings.low,
                "shift_limit_px": settings.shift_limit,
                "shift_search_mode": settings.area_search_modus,
                "alignment_mask_diameter_angstrom": mask_diameter,
                "alignment_mask_path": None if mask_path is None else str(mask_path),
                "averaging": "dynamo project assembly (average_particles)",
                "wedge_boundary_policy": WEDGE_BOUNDARY_POLICY,
                "metadata_path": str(paths.particle_metadata),
                "qc_path": str(paths.report),
                "gold_standard": gold_standard,
                # As fft-refinement records them: the two maps a resolution is a claim about.
                **(
                    {
                        "half_map_paths": {
                            str(h): str(paths.half_maps[h]) for h in sorted(half_maps)
                        },
                        "resolution_angstrom": half_map_fsc["masked"]["crossing_0_143_angstrom"],
                    }
                    if half_map_fsc is not None
                    else {}
                ),
            },
            software_versions={
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(average_path)
            + sum(_path_size_bytes(paths.half_maps[h]) for h in half_maps),
        )
        qc = Artifact(
            id=f"{scope}:qc:subtomogram_averaging:{self.name}{variant_suffix(context)}",
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[average.id],
            parameters={
                "qc_type": "subtomogram_averaging",
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
                "status": status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [average, qc]


SEARCH_PARAMETERS = (
    "num_iterations",
    "workers",
    "cone_range_deg",
    "cone_sampling_deg",
    "inplane_range_deg",
    "inplane_sampling_deg",
    "refine_levels",
    "refine_factor",
    "search_highpass_px",
    "search_lowpass_px",
    "shift_range_px",
    "shift_search_mode",
    "initial_reference_path",
    "gold_standard",
    "gold_standard_start_lowpass_angstrom",
)
"""What `dynamo-search` reads to search, which `dynamo-average` refuses rather than ignores."""

DYNAMO_AVERAGE_FMIN = 1
"""Dynamo's `fmin`: a Fourier component is kept where the summed wedges reach `fmin`
particles' worth, `fmin / N` of their maximum. The value of the console call that
makes a project's starting average, and so of `dynamo-search`'s own start."""


class DynamoAverageBackend:
    """Dynamo's `dynamo_average` at the poses the stacks give: an average with no search.

    What `dynamo_average(data, 'table', t, 'fcompensate', true, 'fmin', 1, 'nmask',
    mask)` makes in Dynamo's console, and what `dynamo-search` starts from when it
    is given no reference: `average_class`, replayed against Dynamo to 4e-7. The
    particles keep their poses, so this is the map before any refinement. With
    template-matched poses it is also what a template prints into noise, which
    only the same average at random positions can tell apart
    (scripts/decoy_picks.py).
    """

    name = "dynamo-average"

    def average(
        self,
        particles: Sequence[Artifact],
        manifests: Sequence[TiltSeriesManifest],
        context: BackendContext,
    ) -> list[Artifact]:
        """Return the average at the stacks' own poses and a QC artifact."""

        searching = [
            key for key in SEARCH_PARAMETERS if context.parameters.get(key) not in (None, False)
        ]
        if searching:
            raise ValueError(
                f"{self.name} does not search, so it does not read {', '.join(searching)}"
            )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        device = torch_device_for_preference(context.device)
        inputs = _dynamo_inputs(self.name, particles, manifests, context, device)
        combined = inputs.combined
        scope = averaging_scope(combined.tilt_series_ids)
        paths = _averaging_paths(
            context.output_dir, scope, backend_name=self.name, label=variant_label(context)
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        angles, dynamo_shifts = _dynamo_poses(inputs.matrices, inputs.shifts, inputs.box)
        result = average_class(
            inputs.raw,
            angles,
            dynamo_shifts,
            inputs.tilt_ranges,
            inputs.normalization_mask,
            fmin=DYNAMO_AVERAGE_FMIN,
        )
        # The poses go out as they came in: the STAR says what was averaged.
        records = [replace(record, class_number=1) for record in inputs.records]
        write_particle_star(paths.particle_metadata, records)
        spacing = inputs.pairs[0][0].pixel_spacing_angstrom
        average_path = _class_average_path(paths, 1, num_classes=1)
        _write_average_mrc(average_path, result.average, pixel_spacing_angstrom=spacing)

        # Each rlnRandomSubset half averaged alone, the same way at the same poses,
        # so the two can be compared before anything was selected or refined.
        warnings: list[str] = []
        halves = half_set_indices(records)
        half_counts = {half: len(indices) for half, indices in halves.items()}
        half_maps: dict[int, np.ndarray] = {}
        if min(half_counts.values()) < 2:
            warnings.append(
                f"no half maps: rlnRandomSubset puts {half_counts[1]} and {half_counts[2]} "
                "particles in the halves, and each needs at least two"
            )
        else:
            for half, indices in sorted(halves.items()):
                half_maps[half] = average_class(
                    [inputs.raw[i] for i in indices],
                    [angles[i] for i in indices],
                    [dynamo_shifts[i] for i in indices],
                    [inputs.tilt_ranges[i] for i in indices],
                    inputs.normalization_mask,
                    fmin=DYNAMO_AVERAGE_FMIN,
                ).average
                _write_average_mrc(
                    paths.half_maps[half], half_maps[half], pixel_spacing_angstrom=spacing
                )
            if min(half_counts.values()) < _MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION:
                warnings.append(
                    f"a half holds {min(half_counts.values())} particles, fewer than "
                    f"{_MINIMUM_PARTICLES_PER_HALF_FOR_RESOLUTION}: its FSC is noise"
                )
        half_map_fsc = (
            _half_map_fsc(half_maps, inputs.normalization_mask, inputs.spacing)
            if half_maps
            else None
        )

        report = DynamoAverageQc(
            backend=self.name,
            tilt_series_ids=combined.tilt_series_ids,
            particle_stack_ids=combined.stack_ids,
            particle_count=len(records),
            particle_counts=combined.counts,
            alignment_mask_diameter_angstrom=inputs.mask_diameter,
            alignment_mask_path=None if inputs.mask_path is None else str(inputs.mask_path),
            normalization_mask_voxels=int(np.count_nonzero(inputs.normalization_mask)),
            wedge_tilt_ranges_deg=[list(one) for one in inputs.wedge_ranges],
            dynamo_ytilt_ranges_deg=[list(dynamo_ytilt(one)) for one in inputs.wedge_ranges],
            wedge_boundary_policy=WEDGE_BOUNDARY_POLICY,
            fmin=DYNAMO_AVERAGE_FMIN,
            fourier_weight_threshold=result.threshold,
            fourier_weight=_fourier_weight_summary(result.fweight, result.threshold),
            half_counts={str(h): n for h, n in half_counts.items()} if half_maps else None,
            half_map_fsc=half_map_fsc,
            status=QcStatus.WARNING if warnings else QcStatus.PASS,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))
        average = Artifact(
            id=_class_average_id(
                scope, self.name, 1, num_classes=1, variant=variant_suffix(context)
            ),
            variant=variant_label(context),
            kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
            path=average_path,
            parent_ids=list(combined.stack_ids),
            shape=(inputs.box, inputs.box, inputs.box),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=spacing,
            parameters={
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
                "particle_stack_ids": list(combined.stack_ids),
                "class_number": 1,
                "num_classes": 1,
                "particle_count": len(records),
                "iterations_run": 0,
                "alignment_mask_diameter_angstrom": inputs.mask_diameter,
                "alignment_mask_path": None if inputs.mask_path is None else str(inputs.mask_path),
                "averaging": report.averaging,
                "fmin": DYNAMO_AVERAGE_FMIN,
                "wedge_boundary_policy": WEDGE_BOUNDARY_POLICY,
                "metadata_path": str(paths.particle_metadata),
                "qc_path": str(paths.report),
                "gold_standard": False,
                # The maps only: their FSC is in the QC, and no resolution is
                # recorded, since the halves were not refined apart.
                **(
                    {"half_map_paths": {str(h): str(paths.half_maps[h]) for h in sorted(half_maps)}}
                    if half_maps
                    else {}
                ),
            },
            software_versions={
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(average_path)
            + sum(_path_size_bytes(paths.half_maps[h]) for h in half_maps),
        )
        qc = Artifact(
            id=f"{scope}:qc:subtomogram_averaging:{self.name}{variant_suffix(context)}",
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[average.id],
            parameters={
                "qc_type": "subtomogram_averaging",
                "backend": self.name,
                "tilt_series_ids": list(combined.tilt_series_ids),
                "status": report.status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [average, qc]
