#!/usr/bin/env python3
"""How close is a map to itself rotated about its z axis, angle by angle?

    python scripts/rotational_symmetry.py \\
        --map gag_half_eo/results/ite_0003/averages/average_ref_001_ite_0003.em \\
        --map gag_half_eo/results/ite_0003/averages/average_ref_002_ite_0003.em \\
        --mask ../mask_sphere_250A_box64.mrc

For each map, rotates it about the reference z axis through the box centre in
steps of --step degrees and prints the Pearson correlation with the unrotated map
inside the mask (voxels above 0.5, where the rotated map is defined).

The question it answers: GPU and CPU Dynamo put three Gag particles about 60 deg
apart (gag_gpu_em_eo vs gag_half_eo, 2026-10-12), which would be harmless if the
map is pseudo-sixfold about z, two poses the reference can barely tell apart. A
sixfold map correlates high at 60, 120, 180, 240 and 300 and low at the 30-degree
points between them, which are the control: for a sixfold map they are the worst
angles. A threefold or twofold map peaks at 120/240 or 180 only. No peaks means
the map is not symmetric about z, and the outliers need another explanation.

The rotation is `resample` from the dynamo-search backend, the same pull at M @ x
and trilinear interpolation the replays matched Dynamo with, angles (a, 0, 0).
Interpolation lowers every nonzero angle alike; only 0 deg escapes it, so compare
the nonzero angles with each other, never with 0. The z axis is the reference's:
whether it is the hexamer axis depends on how the template was oriented, and a
flat profile can also mean it is not. Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np

from cryoet_pipeline.backends.dynamo_search import resample

# EM data types (byte 3 of the header) this reads, as written by Dynamo's dwrite.
_EM_TYPES = {2: "<i2", 4: "<i4", 5: "<f4", 9: "<f8"}
_SIXFOLD = (60, 120, 180, 240, 300)
_BETWEEN = (30, 90, 150, 210, 270, 330)


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


def correlation(first: np.ndarray, second: np.ndarray, where: np.ndarray) -> float:
    a, b = first[where], second[where]
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def profile(volume: np.ndarray, mask: np.ndarray, angles) -> dict[int, float]:
    inside = mask > 0.5
    result = {}
    for angle in angles:
        rotated, valid = resample(volume, (float(angle), 0.0, 0.0))
        result[int(angle)] = correlation(volume, rotated, inside & valid)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", type=Path, action="append", required=True, help="repeat for several maps")
    parser.add_argument("--mask", type=Path, required=True, help="voxels above 0.5 are compared")
    parser.add_argument("--step", type=int, default=10, help="degrees; 360 and 30 must be multiples of it")
    args = parser.parse_args()
    if 360 % args.step or 30 % args.step:
        raise SystemExit("--step must divide 30")

    mask = read_volume(args.mask)
    angles = range(0, 360, args.step)
    for path in args.map:
        volume = read_volume(path)
        if volume.shape != mask.shape:
            raise SystemExit(f"{path} is {volume.shape}, the mask {mask.shape}")
        cc = profile(volume, mask, angles)
        print(f"== {path}")
        for angle in angles:
            tag = " sixfold" if angle in _SIXFOLD else (" between" if angle in _BETWEEN else "")
            print(f"  {angle:3d} deg  CC {cc[angle]:+.3f}{tag}")
        sixfold, between = np.mean([cc[a] for a in _SIXFOLD]), np.mean([cc[a] for a in _BETWEEN])
        print(f"  mean at 60..300: {sixfold:+.3f}   mean at 30..330: {between:+.3f}   "
              f"120/240: {cc[120]:+.3f} {cc[240]:+.3f}   180: {cc[180]:+.3f}")


if __name__ == "__main__":
    main()
