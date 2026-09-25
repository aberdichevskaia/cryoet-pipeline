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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from cryoet_pipeline.backends.protocols import BackendContext, variant_label, variant_suffix
from cryoet_pipeline.backends.subtomogram import (
    _averaging_paths,
    _bool_parameter,
    _class_average_id,
    _class_average_path,
    _combine_particle_stacks,
    _compensated_reference,
    _Compensation,
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
    _write_average_mrc,
    _write_json,
    averaging_scope,
    pose_change_summary,
    torch_device_for_preference,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    DynamoSearchQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import write_particle_star
from cryoet_pipeline.rotations import (
    matrix_to_relion_euler,
    relion_euler_to_matrix,
    resample_volume_zyx,
)
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

    def __post_init__(self):
        numbers = np.array(list(vars(self).values()), dtype=float)
        if not np.isfinite(numbers).all():
            raise ValueError("search settings must be finite")
        if not (0 < self.high < self.low and self.shift_limit > 0):
            raise ValueError("require 0 < high < low and shift_limit > 0")
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
    """Mode 2: integer sphere around the previous shift, then subpixel peak.

    As in Dynamo, interpolation uses UNRESTRICTED neighbours and may leave the
    sphere. The sphere moves again at each refinement level, not only each run.
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
        center_shift = best_shift.copy()
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


def average_particles(particles: Iterable[np.ndarray], angles, shifts, tilt_ranges, mask):
    """Raw-data average and Fourier compensation; no search/interpolation reuse.

    Use the same helper for fixed and refined poses to avoid comparing different
    averaging estimators. Normalization uses all NONZERO mask voxels (ddof=1).
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
    # The fsc=true merge in Dynamo 1.1.511 hard-codes 0.1, ignoring fmin.
    good = weights >= 0.1
    spectrum = np.fft.fftshift(np.fft.fftn(raw))
    corrected = np.zeros_like(spectrum)
    np.divide(spectrum, weights, out=corrected, where=good)
    average = np.fft.ifftn(np.fft.ifftshift(corrected)).real
    return average, raw, weights


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


def _compensated_average(volumes, matrices, shifts, compensation, geometries, device):
    """This pipeline's reducer on the raw particles at the given native poses."""
    aligned = [
        resample_volume_zyx(volume, matrix, shift, device=device)
        for volume, matrix, shift in zip(volumes, matrices, shifts, strict=True)
    ]
    return _compensated_reference(aligned, matrices, compensation, geometries)


class DynamoSearchRefinementBackend:
    """Dynamo's local search, this pipeline's compensated averaging. The default backend."""

    name = "dynamo-search"

    def average(
        self,
        particles: Sequence[Artifact],
        manifests: Sequence[TiltSeriesManifest],
        context: BackendContext,
    ) -> list[Artifact]:
        """Return the refined average and a QC artifact; the metadata sits beside them."""

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
                raise ValueError(f"dynamo-search does not support {key}={value!r}")
        for key in ("particle_metadata_path", "class_number"):
            if context.parameters.get(key) is not None:
                raise ValueError(f"dynamo-search does not read {key}; start from the stack's poses")
        num_iterations = _positive_int_parameter(context, "num_iterations", default=1)
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
        )
        if _nonnegative_float_parameter(context, "wedge_tilt_axis_deg", default=0.0) != 0.0:
            raise ValueError("dynamo-search models one wedge about y; the tilt axis must be 0")
        initial_reference_path = _optional_path_parameter(context, "initial_reference_path")
        overwrite = _bool_parameter(context, "overwrite", default=False)
        device = torch_device_for_preference(context.device)

        combined = _combine_particle_stacks(pairs, device=device)
        box = combined.box_size
        if box < 4 or box % 2:
            raise ValueError(f"dynamo-search needs an even box of at least 4 voxels, got {box}")
        if settings.low > box / 2:
            raise ValueError(f"search_lowpass_px {settings.low} lies beyond Nyquist ({box / 2})")
        spacing = _pixel_spacing_angstrom(pairs[0][0])
        particle_diameter = _positive_float_parameter(
            context, "particle_diameter_angstrom", default=box * spacing
        )
        mask_diameter = (
            _optional_float_parameter(context, "alignment_mask_diameter_angstrom")
            or particle_diameter
        )
        mask = search_mask(box, mask_diameter / spacing)
        geometries = _wedge_geometry_per_particle(
            pairs,
            combined,
            min_tilt_override=_optional_float_parameter(context, "wedge_min_tilt_deg"),
            max_tilt_override=_optional_float_parameter(context, "wedge_max_tilt_deg"),
            tilt_axis_deg=0.0,
        )
        ranges = sorted({(g.min_tilt_deg, g.max_tilt_deg) for g in geometries})
        for low, high in ranges:
            if not low < 0 < high:
                raise ValueError(
                    f"dynamo-search's wedge needs a tilt range spanning zero, got [{low}, {high}]"
                )
        scope = averaging_scope(combined.tilt_series_ids)
        paths = _averaging_paths(
            context.output_dir, scope, backend_name=self.name, label=variant_label(context)
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.average_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)

        compensation = _Compensation(geometries=geometries, box_size=box, device=device)
        records = list(combined.records)
        if [record.index for record in records] != list(range(len(records))):
            raise ValueError("combined particle records are not in stack order")
        starting_poses = [
            (r.rot_deg, r.tilt_deg, r.psi_deg, r.shift_x_px, r.shift_y_px, r.shift_z_px)
            for r in records
        ]
        matrices = [relion_euler_to_matrix(r.rot_deg, r.tilt_deg, r.psi_deg) for r in records]
        shifts = [np.array([r.shift_x_px, r.shift_y_px, r.shift_z_px]) for r in records]
        volumes = combined.volumes
        raw = [volume.detach().cpu().numpy().astype(np.float64) for volume in volumes]
        if initial_reference_path is not None:
            reference = _load_reference_mrc(
                initial_reference_path, expected_box_size=box, device=device
            )
            initial_reference = str(initial_reference_path)
        else:
            reference, _ = _compensated_average(
                volumes, matrices, shifts, compensation, geometries, device
            )
            initial_reference = "compensated average of the initial poses"

        correlations = [math.nan] * len(records)
        coverage = None
        iteration_summaries: list[dict[str, float]] = []
        for iteration in range(num_iterations):
            fixed = reference.detach().cpu().numpy().astype(np.float64)
            before = [matrix.copy() for matrix in matrices], [shift.copy() for shift in shifts]
            for index, geometry in enumerate(geometries):
                angles, shift = native_to_dynamo(matrices[index], shifts[index], box)
                result = refine_particle(
                    raw[index],
                    fixed,
                    mask,
                    angles,
                    shift,
                    (geometry.min_tilt_deg, geometry.max_tilt_deg),
                    settings,
                )
                matrices[index], shifts[index] = dynamo_to_native(result.angles, result.shift, box)
                correlations[index] = result.correlation
            reference, coverage = _compensated_average(
                volumes, matrices, shifts, compensation, geometries, device
            )
            iteration_summaries.append(
                {
                    "iteration": float(iteration + 1),
                    "mean_correlation": float(np.mean(correlations)),
                    "minimum_correlation": float(np.min(correlations)),
                    "rotation_deg_mean": float(
                        np.mean(
                            [
                                _rotation_between_deg(a, b)
                                for a, b in zip(before[0], matrices, strict=True)
                            ]
                        )
                    ),
                    "shift_px_mean": float(
                        np.mean(
                            [
                                np.linalg.norm(a - b)
                                for a, b in zip(before[1], shifts, strict=True)
                            ]
                        )
                    ),
                }
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
            reference.detach().cpu().numpy(),
            pixel_spacing_angstrom=pairs[0][0].pixel_spacing_angstrom,
        )

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
        status = QcStatus.WARNING if warnings else QcStatus.PASS
        floor = max(compensation.floor_fraction, 1.0 / len(records))
        report = DynamoSearchQc(
            backend=self.name,
            tilt_series_ids=combined.tilt_series_ids,
            particle_stack_ids=combined.stack_ids,
            particle_count=len(records),
            particle_counts=combined.counts,
            iterations_run=num_iterations,
            cone_aperture_deg=settings.cone_aperture,
            cone_sampling_deg=settings.cone_sampling,
            inplane_range_deg=settings.inplane_range,
            inplane_sampling_deg=settings.inplane_sampling,
            refine_levels=settings.refine,
            refine_factor=settings.refine_factor,
            search_highpass_px=settings.high,
            search_lowpass_px=settings.low,
            shift_limit_px=settings.shift_limit,
            alignment_mask_diameter_angstrom=mask_diameter,
            grid_reach_deg=reach,
            wedge_tilt_ranges_deg=[list(one) for one in ranges],
            wedge_boundary_policy=WEDGE_BOUNDARY_POLICY,
            wedge_coverage_floor_fraction=compensation.floor_fraction,
            wedge_coverage=(
                None
                if coverage is None
                else {
                    "minimum": float(coverage.min()),
                    "median": float(coverage.median()),
                    "fraction_on_the_floor": float((coverage < floor).to(torch.float32).mean()),
                }
            ),
            initial_reference=initial_reference,
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
                "cone_aperture_deg": settings.cone_aperture,
                "cone_sampling_deg": settings.cone_sampling,
                "inplane_range_deg": settings.inplane_range,
                "inplane_sampling_deg": settings.inplane_sampling,
                "refine_levels": settings.refine,
                "refine_factor": settings.refine_factor,
                "search_highpass_px": settings.high,
                "search_lowpass_px": settings.low,
                "shift_limit_px": settings.shift_limit,
                "alignment_mask_diameter_angstrom": mask_diameter,
                "wedge_boundary_policy": WEDGE_BOUNDARY_POLICY,
                "metadata_path": str(paths.particle_metadata),
                "qc_path": str(paths.report),
            },
            software_versions={
                "numpy": _package_version("numpy"),
                "torch": _package_version("torch"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(average_path),
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
