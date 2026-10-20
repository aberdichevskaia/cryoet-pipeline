#!/usr/bin/env python3
"""How close is a map to itself rotated about its z axis, and about which axis is it closest?

    python scripts/rotational_symmetry.py \\
        --map gag_half_eo/results/ite_0003/averages/average_ref_001_ite_0003.em \\
        --map gag_half_eo/results/ite_0003/averages/average_ref_002_ite_0003.em \\
        --mask ../mask_sphere_250A_box64.mrc

For each map, prints the Pearson correlation with itself rotated by 30, 60, ...
180 deg inside the mask (voxels above 0.5, where both copies are defined), twice:
about the z axis through the box centre, and about the axis that fits best. The
fit moves the axis off the centre (x, y offset in voxels) and tilts it, maximising
the mean correlation at 60, 120 and 180 (Nelder-Mead from the box axis).
CC(-a) equals CC(a) up to interpolation, so 0..180 covers the circle.

The question it answers: GPU and CPU Dynamo put three Gag particles about 60 deg
apart (gag_gpu_em_eo vs gag_half_eo, 2026-10-12). The far poses come from the grid:
near tdrot +-90, tilt 90 Dynamo's in-plane seed formula fails and the six ring
directions of a 15/5 cone get seeds about 60 deg apart. What the grid cannot
explain is a far candidate beating every near one; a map pseudo-sixfold about z,
two poses the reference can barely tell apart, would. A sixfold map correlates
high at 60, 120 and 180 and low at 30, 90 and 150, which are the control: for a
sixfold map they are the worst angles. A threefold or twofold map peaks at 120 or
180 only. No peaks means the map is not symmetric about z.

--order n (default 6) sets the symmetry looked for: the angles are the multiples
of 180/n up to 180, the peaks the even ones (multiples of 360/n) and the control
the odd ones, so for n = 4 peaks at 90 and 180 and controls at 45 and 135, and the
axis is fitted to the peaks. Before imposing C4 on RyR1 (2026-10-20) the question
is the same one: Dynamo rotates about the z axis through voxel N/2 (zero-based;
dynamo_csym turns with dynamo_rot, whose 3D centre is the EM convention), the
export centres particles on (N - 1) / 2, and this script's box centre is N // 2,
Dynamo's, so the fitted offset is how far the map's axis is from the one Dynamo's
c4 would use.

On the gag averages (2026-10-13) the peaks fell with the angle, 0.84 at 60, 0.72 at
120. A sixfold axis displaced from the box centre, or tilted from z, does that:
rotating about the wrong axis adds a translation or tilt growing as sin(a/2). The
fit answers it: peaks that come out level about the fitted axis were the axis,
peaks that still fall are the map, which is then only roughly sixfold. That is
what matters before imposing C6, which Dynamo does about the box z axis.

Interpolation smooths noise and so raises a correlation; how much depends on where
the samples fall between voxels, so every compared copy has to be interpolated
alike. Two ways it was not: against the unrotated map, rotations by 90, 180 and 270
about the box axis map the grid onto itself and are not interpolated at all, and
read low (first version: 0.09 at 180 against 0.12-0.13 at 60/120 on a noisy
synthetic C6); and any rotation about z keeps z on the grid, so a tilted axis,
which does not, gained correlation from smoothing alone (a centred noisy C6 fitted
a 2 deg tilt). So both copies are sampled on the box grid turned by one fixed
oblique rotation about the centre: copy a is V(p + R_a (D (x - c) + c - p)), D
fixed, R_a the rotation by a about the candidate axis through p. A map symmetric
about that axis gives identical copies for any D, and every copy, whatever the axis,
falls between voxels the same way. The mask is applied in the box frame and turned
with it, so it should be a sphere about the centre.

The rotation is the pull `resample` in the dynamo-search backend does, c + M (x - c)
- shift with trilinear interpolation and eight-corner support, written here for a
rotation matrix: resample takes Euler angles, and their pole snapping would quantise
a small tilt. The z axis is the reference's: whether it is the hexamer axis depends
on how the template was oriented. Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import mrcfile
import numpy as np
from scipy.optimize import minimize

# EM data types (byte 3 of the header) this reads, as written by Dynamo's dwrite.
_EM_TYPES = {2: "<i2", 4: "<i4", 5: "<f4", 9: "<f8"}


def angle_sets(order: int) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """All angles k * 180 / order up to 180, the peaks (even k) and the control (odd k)."""

    angles = tuple(180.0 * k / order for k in range(1, order + 1))
    return angles, angles[1::2], angles[0::2]


def read_em(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw[0] != 6:
        raise SystemExit(f"{path}: EM machine byte {raw[0]}, only 6 (PC, little endian) is read")
    if raw[3] not in _EM_TYPES:
        raise SystemExit(f"{path}: EM data type {raw[3]} is not read here")
    nx, ny, nz = np.frombuffer(raw, dtype="<i4", count=3, offset=4)
    data = np.frombuffer(raw, dtype=_EM_TYPES[raw[3]], count=nx * ny * nz, offset=512)
    return data.reshape(nz, ny, nx).astype(np.float64)


def read_volume(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".em":
        return read_em(path)
    with mrcfile.open(path, permissive=True) as handle:
        return np.array(handle.data, dtype=np.float64)


@lru_cache(maxsize=2)
def _xyz(size: int) -> np.ndarray:
    z, y, x = np.indices((size,) * 3, dtype=np.float32) - size // 2
    return np.stack((x, y, z)).reshape(3, -1)


def pull(volume: np.ndarray, matrix: np.ndarray, shift: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Values at c + matrix @ (x - c) - shift, trilinear, zero where a corner is outside."""
    v = volume.astype(np.float32)
    n = v.shape[0]
    q = matrix.astype(np.float32) @ _xyz(n) - shift.astype(np.float32)[:, None]
    floors = np.floor(q)
    frac = q - floors
    base = floors.astype(np.int32) + n // 2
    valid = np.all((base >= 0) & (base + 1 < n), axis=0)
    base = np.clip(base, 0, n - 2)
    result = np.zeros(q.shape[1], dtype=np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = ((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1])
                          * (frac[2] if dz else 1 - frac[2]))
                result += weight * v[base[2] + dz, base[1] + dy, base[0] + dx]
    result[~valid] = 0
    return result.reshape(v.shape).astype(np.float64), valid.reshape(v.shape)


def axis_rotation(direction: np.ndarray, degrees: float) -> np.ndarray:
    u = direction / np.linalg.norm(direction)
    a = np.deg2rad(degrees)
    cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.cos(a) * np.eye(3) + np.sin(a) * cross + (1 - np.cos(a)) * np.outer(u, u)


# The fixed oblique turn of the sampling grid; any rotation that leaves no axis on
# the grid would do.
_OBLIQUE = axis_rotation(np.array([1.0, 0.37, 0.21]), 9.7)


def rotated(volume, degrees, axis):
    """The map rotated about the axis through the box centre + (ox, oy, 0), direction
    (tx, ty, 1), sampled on the obliquely turned grid."""
    ox, oy, tx, ty = axis
    rotation = axis_rotation(np.array([tx, ty, 1.0]), degrees)
    point = np.array([ox, oy, 0.0])
    return pull(volume, rotation @ _OBLIQUE, (rotation - np.eye(3)) @ point)


def correlation(first: np.ndarray, second: np.ndarray, where: np.ndarray) -> float:
    a, b = first[where], second[where]
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def profile(volume, inside, axis, angles) -> dict[float, float]:
    base, base_valid = rotated(volume, 0.0, axis)
    result = {}
    for angle in angles:
        copy, valid = rotated(volume, float(angle), axis)
        result[angle] = correlation(base, copy, inside & base_valid & valid)
    return result


def fit_axis(volume, inside, peaks):
    def cost(axis):
        return -np.mean(list(profile(volume, inside, axis, peaks).values()))

    # Steps of half a voxel and about one degree to start; the box axis is the start.
    simplex = np.vstack([np.zeros(4), np.diag([0.5, 0.5, 0.02, 0.02])])
    result = minimize(cost, np.zeros(4), method="Nelder-Mead",
                      options={"initial_simplex": simplex, "xatol": 0.01, "fatol": 1e-5, "maxfev": 400})
    return result.x, int(result.nfev)


def row(name: str, cc: dict[float, float], order: int) -> str:
    angles, peaks, between = angle_sets(order)
    at_peaks = [cc[a] for a in peaks]
    values = " ".join(f"{cc[a]:+6.3f}" for a in angles)
    return (f"  {name:8s} {values}   {np.mean(at_peaks):+.3f}  {np.mean([cc[a] for a in between]):+.3f}"
            f"   {max(at_peaks) - min(at_peaks):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, action="append", required=True, help="repeat for several maps")
    parser.add_argument("--mask", type=Path, required=True, help="voxels above 0.5 are compared")
    parser.add_argument("--order", type=int, default=6, help="the n of Cn looked for (default 6)")
    args = parser.parse_args()
    if args.order < 2:
        parser.error("--order must be 2 or more")
    angles, peaks, between = angle_sets(args.order)

    mask = read_volume(args.mask)
    inside = mask > 0.5
    header = " ".join(f"{a:>6g}" for a in angles)
    span = lambda group: f"{group[0]:g}-{group[-1]:g}"
    for path in args.map:
        volume = read_volume(path)
        if volume.shape != mask.shape:
            raise SystemExit(f"{path} is {volume.shape}, the mask {mask.shape}")
        box = profile(volume, inside, np.zeros(4), angles)
        axis, evaluations = fit_axis(volume, inside, peaks)
        fitted = profile(volume, inside, axis, angles)
        ox, oy, tx, ty = axis
        print(f"== {path}")
        print(f"  axis     {header}   {span(peaks):>6s}  {span(between):>6s}  spread {span(peaks)}")
        print(row("box z", box, args.order))
        print(row("fitted", fitted, args.order))
        print(f"  fitted axis: through centre + ({ox:+.2f}, {oy:+.2f}) voxels, tilted "
              f"{np.degrees(np.arctan(np.hypot(tx, ty))):.2f} deg towards azimuth "
              f"{np.degrees(np.arctan2(ty, tx)):+.0f} deg ({evaluations} evaluations)")


if __name__ == "__main__":
    main()
