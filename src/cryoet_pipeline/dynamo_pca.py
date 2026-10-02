"""Dynamo 1.1.511's PCA classification, ported for replay before integration.

Standalone, as `dynamo_mra.py` is: nothing in the pipeline calls this yet. It is
to be checked against a real Dynamo ccmatrix project first (benchmark
`DYNAMO_CLEANUP.md`, arm P, run in project mode), then wired in.

What Dynamo's project-mode chain does, read from `dynamo_ccmatrix_compute.m`,
`dynamo_ccmatrix_assemble.m`, `dynamo_PCA_Xmatrix.m`, `dynamo_PCA.m`,
`dynamo_kmeans.m` and the `dynamo_particle.m` actions they call:

1. Every particle is brought into the reference frame by its table pose. This is
   the inverse shift-rotate of the averaging. Voxels the rotation leaves
   undefined become 0, and the project band then filters the particle
   (`aligned`).
2. Its Fourier sampling is the table's single-axis wedge, evaluated analytically
   in the particle's rotated frame (`rotated_wedge`).
3. Every pair is compared on the frequencies both of them sampled: the product
   of the two binarized samplings. Each particle is filtered by it, and the two
   are Pearson-correlated over the classification mask's nonzero voxels. A NaN
   becomes -1, and the matrix is symmetric with a unit diagonal (`cc_matrix`).
4. PCA keeps the ten eigenvectors of largest |eigenvalue| (`eigs(..., 'LM')`).
   The data matrix X holds each aligned particle's mask voxels, and the
   eigenvolumes are V' X (`pca`). The mean particle is not removed in project
   mode: `dynamo_PCA_Xmatrix` computes the centred blocks but discards them
   (its `normalize_block` result is never stored), so the blocks in the
   database, which `dynamo_PCA` reads, hold the particles themselves.
5. A particle's coefficient on an eigenvolume is the Pearson correlation of the
   two after both are filtered by the particle's own sampling map, over the
   mask. That map is used as `dynamo_fsampling_forcemap` returns it, without the
   CC matrix's binarization. Here the particle is aligned with undefined voxels
   set to its minimum, not 0. Dynamo stores the coefficients from table
   column 41 on (`coefficients`).
6. k-means runs on coefficients 1..5 into K clusters: MATLAB's `kmeans` with
   one k-means++ start, squared Euclidean distance and batch updates only.
   Its random start cannot be replayed bit for bit. `lloyd` runs the same
   updates from given centroids, so a replay can test Dynamo's clusters as a
   fixed point (`kmeans`).

Scope, refused rather than guessed: project mode (the uncentred Xmatrix),
single-axis wedges (table ftype 1), no resizing, binning or symmetrization, the
actions `align` and `band` and nothing else, and coefficients with the fmask
compensation that `dynamo_PCA` uses by default. Dynamo also multiplies the
common sampling by a sphere of radius N/2 - 1 that a MEX draws
(`dynamo__sphere_mex`, not read). `cc_matrix` refuses particles with any
frequency within 3 pixels of that radius, where the sphere is 1 whatever the
MEX's centre convention, and leaves it out.

The rotated wedge follows Dynamo's own MATLAB rendering of its MEX
(`rotated_wedge_single_tilt_local`), with the crop sphere of the unrotated wedge
the kernel already reproduces. It still has to be checked against the MEX
itself (`dynamo_wedge_mask`).

Numerical adaptation of the Dynamo routines named above. The BSD option of the
following original joint notice is selected for these adaptations; this does
not relicense any other Dynamo component.

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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cryoet_pipeline.backends.dynamo_search import (
    _cube,
    _triple,
    _xyz,
    apply_filter,
    bandpass,
    euler_matrix,
    resample,
    wedge,
)

NEIGS = 10
"""`dynamo_PCA`'s default number of eigenvectors."""
KMEANS_COEFFICIENTS = 5
"""`dynamo_kmeans`'s default: coefficients 1..5."""
KMEANS_MAX_ITERATIONS = 100
"""MATLAB `kmeans`' default `MaxIter`."""


def rotated_wedge(size, tilt_range, angles) -> np.ndarray:
    """Dynamo's single-axis wedge (ftype 1, crop 1) in a frame rotated by `angles`.

    It evaluates the wedge test at X' = M[0]·k and Z' = M[2]·k, with
    M = `euler_matrix(angles)`. A frequency with |X'| < 1e-8 is unsampled; the
    slab |Z'| <= 0.5 is sampled; outside the crop sphere nothing is. With zero
    angles it is the kernel's `wedge`, whose rules there differ on the axis.
    """
    a = _triple(angles)
    if not np.any(a):
        return wedge(size, tilt_range)
    lo, hi = np.asarray(tilt_range, dtype=float)
    if not np.isfinite([lo, hi]).all() or not -90 <= lo < 0 < hi <= 90:
        raise ValueError("require a finite Y tilt range spanning zero within [-90, 90]")
    m = euler_matrix(a)
    x, y, z = _xyz(size).astype(float)
    big_x = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z
    big_z = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z
    on_axis = np.abs(big_x) < 1e-8
    ratio = np.divide(big_z, big_x, out=np.zeros_like(big_z), where=~on_axis)
    tlo, thi = np.tan(np.deg2rad([lo, hi]))
    support = (ratio < thi) & (ratio > tlo) & ~on_axis
    support |= np.abs(big_z) <= 0.5
    support &= (x * x + y * y + z * z) < (size / 2 + 0.5) ** 2
    return support.astype(float)


def aligned(particle, angles, shift, high, low, *, undefined: str = "zero") -> np.ndarray:
    """`dynamo_particle` with `align`, `nan` and `band`, in that order.

    The inverse pose, as the averaging applies it. Then the voxels it leaves
    undefined become 0 (`undefined="zero"`, the CC matrix and the Xmatrix) or the
    particle's minimum (`"min"`, the coefficients). Last comes the band.
    """
    if undefined not in {"zero", "min"}:
        raise ValueError("undefined voxels become 'zero' or 'min'")
    volume, valid = resample(particle, -_triple(angles)[::-1], -_triple(shift))
    if not valid.any():
        raise ValueError("the pose leaves no voxel of the particle defined")
    if undefined == "min":
        volume[~valid] = volume[valid].min()
    return apply_filter(volume, bandpass(len(volume), high, low))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """`dynamo_pearson3d` on already selected voxels; NaN when either is constant."""
    x, y = x - x.mean(), y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def _support(mask, shape) -> np.ndarray:
    support = _cube(mask) != 0
    if support.shape != shape or not support.any():
        raise ValueError("the mask must match the particles and select voxels")
    return support


def _half(centred: np.ndarray) -> np.ndarray:
    """A centred Fourier map on `rfftn`'s half grid."""
    return np.fft.ifftshift(centred)[..., : centred.shape[2] // 2 + 1]


def _spectrum(volume, shape) -> np.ndarray:
    v = _cube(volume)
    if v.shape != shape:
        raise ValueError("all volumes must share one shape")
    return np.fft.rfftn(v, axes=(0, 1, 2))


def _inverse(spectrum, shape) -> np.ndarray:
    return np.fft.irfftn(spectrum, s=shape, axes=(0, 1, 2))


def cc_matrix(volumes: Sequence[np.ndarray], wedges: Sequence[np.ndarray], mask) -> np.ndarray:
    """Every pair on the frequencies both sampled, Pearson over the mask's nonzero voxels.

    The samplings are binarized at 0.5, as `dynamo_binarize` does. A particle
    with frequencies within 3 pixels of the corner sphere's radius N/2 - 1 is
    refused, since the sphere is left out (module docstring).
    """
    if len(volumes) != len(wedges) or len(volumes) < 2:
        raise ValueError("need one wedge per particle and at least two particles")
    shape = _cube(volumes[0]).shape
    support = _support(mask, shape)
    radius = np.sqrt(np.sum(_xyz(shape[0]).astype(float) ** 2, axis=0))
    beyond = _half(radius >= shape[0] / 2 - 4)
    spectra = [_spectrum(v, shape) for v in volumes]
    for index, spectrum in enumerate(spectra):
        size = np.abs(spectrum)
        if size[beyond].max(initial=0) > 1e-9 * size.max(initial=0):
            raise ValueError(
                f"particle {index} has frequencies within 3 pixels of the corner sphere "
                f"(radius {shape[0] / 2 - 1:g}), which is not ported"
            )
    samplings = [_half(_cube(w) >= 0.5) for w in wedges]
    matrix = np.eye(len(volumes))
    for i in range(len(volumes)):
        for j in range(i + 1, len(volumes)):
            common = samplings[i] * samplings[j]
            a = _inverse(spectra[i] * common, shape)[support]
            b = _inverse(spectra[j] * common, shape)[support]
            value = pearson(a, b)
            matrix[i, j] = matrix[j, i] = -1.0 if np.isnan(value) else value
    return matrix


@dataclass(frozen=True)
class Pca:
    eigenvalues: np.ndarray
    """Largest |eigenvalue| first, as `eigs(..., 'LM')` orders them."""
    eigenvectors: np.ndarray
    """One column per eigenvalue; each column's sign is arbitrary, as in ARPACK."""
    eigenvolumes: np.ndarray
    """V' X placed back into the mask; zero outside it."""


def pca(cc: np.ndarray, volumes: Sequence[np.ndarray], mask, neigs: int = NEIGS) -> Pca:
    """`eigs` of the CC matrix and project-mode `dynamo_PCA`'s eigenvolumes.

    X is the aligned particles' mask voxels, uncentred (module docstring). As in
    Dynamo, more eigenvectors than particles means all of them.
    """
    cc = np.asarray(cc, dtype=float)
    if cc.ndim != 2 or cc.shape[0] != cc.shape[1] or cc.shape[0] != len(volumes):
        raise ValueError("the CC matrix must be square, one row per particle")
    if not np.allclose(cc, cc.T, rtol=0, atol=1e-12):
        raise ValueError("the CC matrix must be symmetric")
    if type(neigs) is not int or neigs < 1:
        raise ValueError("neigs must be a positive integer")
    neigs = min(neigs, len(volumes))
    values, vectors = np.linalg.eigh(cc)
    order = np.argsort(-np.abs(values), kind="stable")[:neigs]
    values, vectors = values[order], vectors[:, order]
    shape = _cube(volumes[0]).shape
    support = _support(mask, shape)
    data = np.stack([_cube(v)[support] for v in volumes])
    eigenvolumes = np.zeros((neigs, *shape))
    eigenvolumes[:, support] = vectors.T @ data
    return Pca(values, vectors, eigenvolumes)


def coefficients(eigenvolumes, volumes, wedges, mask) -> np.ndarray:
    """`dynamo_pearson3d_fmask` of every particle with every eigenvolume.

    Both are filtered by the particle's sampling map as given, not binarized,
    then Pearson-correlated over the mask's nonzero voxels: one row per
    particle, one column per eigenvolume, as eigentable columns 41 on.
    """
    if len(volumes) != len(wedges):
        raise ValueError("need one wedge per particle")
    eigenvolumes = np.asarray(eigenvolumes, dtype=float)
    shape = eigenvolumes.shape[1:]
    support = _support(mask, shape)
    eigen_spectra = [_spectrum(e, shape) for e in eigenvolumes]
    table = np.empty((len(volumes), len(eigenvolumes)))
    for i, (volume, sampling) in enumerate(zip(volumes, wedges, strict=True)):
        f = _half(_cube(sampling))
        particle = _inverse(_spectrum(volume, shape) * f, shape)[support]
        for k, spectrum in enumerate(eigen_spectra):
            table[i, k] = pearson(_inverse(spectrum * f, shape)[support], particle)
    return table


def lloyd(data, centroids, max_iterations: int = KMEANS_MAX_ITERATIONS):
    """MATLAB `kmeans`' batch phase from given centroids: labels 1..K, centroids.

    Squared Euclidean distance. Each point starts at its nearest centroid, a tie
    going to the lower cluster as `min` takes the first, and the centroids
    become their clusters' means. A step then moves every point strictly nearer
    another centroid, so a tie stays put, and updates the means. A cluster left
    empty takes the point farthest from its own centroid (`EmptyAction`
    'singleton'). The phase ends when no point moves, when the summed distance
    stops falling (that step is undone), or after `max_iterations` updates.
    """
    data = np.asarray(data, dtype=float)
    centroids = np.array(centroids, dtype=float)
    if data.ndim != 2 or centroids.ndim != 2 or centroids.shape[1] != data.shape[1]:
        raise ValueError("data and centroids must share their dimension")
    if not 0 < len(centroids) <= len(data):
        raise ValueError("need between one centroid and one per point")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    rows = np.arange(len(data))

    def distances(c):
        return ((data[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)

    def means(labels):
        empty = np.full(data.shape[1], np.nan)
        clusters = range(len(centroids))
        return np.stack(
            [data[labels == k].mean(axis=0) if np.any(labels == k) else empty for k in clusters]
        )

    labels = distances(centroids).argmin(axis=1)
    previous, previous_total = labels, np.inf
    for iteration in range(1, max_iterations + 1):
        centroids = means(labels)
        d = distances(centroids)
        for k in np.flatnonzero(np.bincount(labels, minlength=len(centroids)) == 0):
            counts = np.bincount(labels, minlength=len(centroids))
            lonely = int(np.argmax(d[rows, labels]))
            source = labels[lonely]
            if counts[source] < 2:
                source = int(np.flatnonzero(counts > 1)[0])
                lonely = int(np.flatnonzero(labels == source)[0])
            labels = labels.copy()
            labels[lonely] = k
            centroids[k] = data[lonely]
            centroids[source] = data[labels == source].mean(axis=0)
            d = distances(centroids)
        total = float(d[rows, labels].sum())
        if previous_total <= total:
            labels = previous
            centroids = means(labels)
            break
        if iteration == max_iterations:
            break
        previous, previous_total = labels, total
        nearest = d.argmin(axis=1)
        moved = d[rows, labels] > d[rows, nearest]
        if not moved.any():
            break
        labels = np.where(moved, nearest, labels)
    return labels + 1, centroids


def kmeans(data, k: int, rng: np.random.Generator, replicates: int = 1):
    """k-means++ starts, then `lloyd`; the best of `replicates` by summed distance.

    This is the algorithm `dynamo_kmeans` calls, not its draw: MATLAB's
    generator chooses Dynamo's start.
    """
    data = np.asarray(data, dtype=float)
    if type(k) is not int or not 0 < k <= len(data):
        raise ValueError("k must be a positive integer no larger than the point count")
    best = None
    for _ in range(replicates):
        centroids = [data[rng.integers(len(data))]]
        for _ in range(1, k):
            nearest = np.min([((data - c) ** 2).sum(axis=1) for c in centroids], axis=0)
            centroids.append(data[rng.choice(len(data), p=nearest / nearest.sum())])
        labels, final = lloyd(data, np.stack(centroids))
        total = float(((data - final[labels - 1]) ** 2).sum())
        if best is None or total < best[0]:
            best = (total, labels, final)
    return best[1], best[2]


@dataclass(frozen=True)
class Classification:
    cc: np.ndarray
    pca: Pca
    coefficients: np.ndarray
    labels: np.ndarray
    """1..K per particle, in the order the particles were given."""


def classify(
    particles,
    angles,
    shifts,
    tilt_ranges,
    mask,
    *,
    high: float,
    low: float,
    k: int,
    rng: np.random.Generator,
    neigs: int = NEIGS,
    n_coefficients: int = KMEANS_COEFFICIENTS,
) -> Classification:
    """Dynamo's chain from particles at their table poses to K clusters."""
    rows = list(zip(particles, angles, shifts, tilt_ranges, strict=True))
    if not 0 < n_coefficients <= neigs:
        raise ValueError("k-means uses coefficients 1..n, n no larger than neigs")
    wedges = [rotated_wedge(len(_cube(p)), r, -_triple(a)[::-1]) for p, a, _, r in rows]
    zero = [aligned(p, a, s, high, low) for p, a, s, _ in rows]
    minimum = [aligned(p, a, s, high, low, undefined="min") for p, a, s, _ in rows]
    cc = cc_matrix(zero, wedges, mask)
    found = pca(cc, zero, mask, neigs)
    table = coefficients(found.eigenvolumes, minimum, wedges, mask)
    labels, _ = kmeans(table[:, :n_coefficients], k, rng)
    return Classification(cc, found, table, labels)
