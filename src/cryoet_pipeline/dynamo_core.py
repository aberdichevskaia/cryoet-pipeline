"""Experimental CPU K=1 kernel for replaying Dynamo 1.1.511, not a backend default.

Arrays are ZYX; poses use Dynamo XYZ shifts and [tdrot, tilt, narot] degrees.
The rotation centre is N/2, NOT the native geometric centre (N-1)/2.
Callers must use the explicit exporter conversion, never pass native poses here.

Scope: one fixed-reference iteration, even cubes, single Y wedge, use_CC=1,
localnc=1, C1, isotropic shift limits (modus 2), no particle rejection. Outputs
are raw re-sampled averages, not filtered search buffers. No training or GT.

Numerical adaptation of the Dynamo routines listed in docs/dynamo_core.md.
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

from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


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
