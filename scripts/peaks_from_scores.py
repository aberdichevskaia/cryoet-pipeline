#!/usr/bin/env python3
"""Pull peak positions straight out of a pytom score volume.

    python3 peaks_from_scores.py TS_0005_scores.mrc --count 400

Exists to salvage a match whose second output file never landed. pytom writes
the score volume first and the angle volume second, so a run that died on the
angles still has every coordinate — and the question being asked of these picks
is where they sit in Z, which needs no orientation at all.

This is not a replacement for `pytom_extract_candidates`: it applies no top-hat
filter and calibrates no cut-off against a background model. It takes the N
strongest local maxima no closer together than one particle, which is enough to
ask whether the picks are confined to a lamella or spread through the volume.

The readout is the standard deviation of Z. For a 320 px volume with a 39 px
template box, pytom cannot centre a box within ~19 px of either face, so uniform
picks give sd = 282/sqrt(12) = 81.4. A 177 px lamella would give about 51.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument(
        "--particle-diameter-angstrom",
        type=float,
        default=290.0,
        help="minimum separation between peaks",
    )
    parser.add_argument("--voxel-size-angstrom", type=float, default=9.784)
    args = parser.parse_args()

    with mrcfile.open(args.scores, permissive=True) as handle:
        volume = np.array(handle.data, dtype=np.float32)  # (Z, Y, X)

    radius = max(1, int(round(args.particle_diameter_angstrom / args.voxel_size_angstrom / 2)))
    print(f"volume {volume.shape} (Z, Y, X)   separation {2 * radius} voxels")

    # A sphere of offsets, punched out around each peak as it is taken, which is
    # how pytom keeps two picks off the same particle.
    span = np.arange(-radius, radius + 1)
    dz, dy, dx = np.meshgrid(span, span, span, indexing="ij")
    sphere = (dz**2 + dy**2 + dx**2) <= radius**2

    depth, height, width = volume.shape
    picks: list[tuple[float, int, int, int]] = []
    for _ in range(args.count):
        flat = int(np.argmax(volume))
        score = float(volume.flat[flat])
        if not np.isfinite(score):
            break
        z, y, x = np.unravel_index(flat, volume.shape)
        picks.append((score, int(z), int(y), int(x)))
        z0, z1 = max(0, z - radius), min(depth, z + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        mask = sphere[
            radius - (z - z0) : radius + (z1 - z),
            radius - (y - y0) : radius + (y1 - y),
            radius - (x - x0) : radius + (x1 - x),
        ]
        volume[z0:z1, y0:y1, x0:x1][mask] = -np.inf

    scores = np.array([p[0] for p in picks])
    z_values = np.array([p[1] for p in picks], dtype=float)
    print(f"peaks {len(picks)}   scores {scores.min():.5f}..{scores.max():.5f}")
    print(f"Z median {np.median(z_values):.0f}   sd {z_values.std():.1f}")
    print("  uniform within the maskable region would be 81.4")
    print("  a 177 px lamella would be about 51")
    counts, edges = np.histogram(z_values, bins=16, range=(0, depth))
    for count, low in zip(counts, edges[:-1], strict=True):
        bar = "#" * int(count * 40 / max(counts.max(), 1))
        print(f"  z {low:4.0f}-{low + depth / 16:4.0f} {count:4d} {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
