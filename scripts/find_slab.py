#!/usr/bin/env python3
"""Where the specimen is in Z, from the tomogram's own variance profile.

    python3 scripts/find_slab.py \\
        --tomogram outputs/ribo/tomograms/TS_0005/TS_0005_fine_bin8_bin8.zarr

Prints the boundaries, and the `--pytom-search-z` value to paste.

A reconstruction is as thick as it was asked to be, not as thick as the
specimen. On EMPIAR-11538 the lamella fills about half of it, and template
matching searched all of it — so roughly half the picks sat in vacuum, where a
particle cannot be. They were not distinguishable by score from the rest, which
is what made the pick list read as noise: it partly was, by construction.

The boundary is found rather than eyeballed because eyeballing gives a number
nobody can reproduce, and because the pipeline should be able to make this
measurement itself. Vacuum has little structure, so the per-slice standard
deviation drops outside the specimen; the specimen shows as a plateau between
two falls. That is the same reasoning roadmap section 12 applies to deriving the
reconstruction thickness, applied to locating what was reconstructed.

Disposable, like the rest of `scripts/`. Its replacement is this measurement
recorded in the reconstruction's QC, so that a later step can restrict itself
without anyone running a script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np


def _slice_deviation(tomogram: Any, *, stride: int) -> np.ndarray:
    """Standard deviation of every Z slice, from a subsample of each.

    Subsampled because the volume is about a gigabyte and a slice's spread is
    well estimated from a fraction of its voxels — the profile's shape is what
    matters, not the fourth digit.
    """

    depth = tomogram.shape[0]
    return np.array(
        [
            float(np.asarray(tomogram[index, ::stride, ::stride], dtype=np.float64).std())
            for index in range(depth)
        ]
    )


def _boundaries(profile: np.ndarray, *, fraction: float) -> tuple[int, int]:
    """First and last slice whose spread clears the vacuum baseline.

    The baseline is taken from the quietest tenth of slices rather than from the
    edges, because a reconstruction's outermost slices carry their own artefacts
    and are not a clean sample of vacuum. The threshold sits a `fraction` of the
    way from that baseline to the peak.
    """

    quietest = np.sort(profile)[: max(len(profile) // 10, 1)]
    baseline = float(quietest.mean())
    peak = float(profile.max())
    if peak <= baseline:
        raise ValueError("the profile is flat: no slab boundary can be found")
    threshold = baseline + fraction * (peak - baseline)
    inside = np.flatnonzero(profile >= threshold)
    return int(inside[0]), int(inside[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tomogram", required=True, type=Path, help="Canonical Zarr.")
    parser.add_argument("--stride", type=int, default=4, help="Subsample within each slice.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Where between the vacuum baseline and the peak to put the boundary. "
        "0.5 is the half-height, which is the conventional choice and is not "
        "sensitive to the exact shape of the falloff.",
    )
    parser.add_argument(
        "--margin-px",
        type=int,
        default=0,
        help="Widen the range by this much on each side, for a search that should "
        "not clip a particle whose centre sits near the boundary.",
    )
    args = parser.parse_args()

    import zarr

    tomogram = cast(Any, zarr.open(args.tomogram, mode="r"))
    apix = float(tomogram.attrs.get("voxel_spacing_angstrom") or 0.0)
    depth = int(tomogram.shape[0])
    print(f"{args.tomogram.name}: {tomogram.shape} at {apix or '?'} A/voxel\n")

    profile = _slice_deviation(tomogram, stride=args.stride)
    low, high = _boundaries(profile, fraction=args.fraction)
    low = max(low - args.margin_px, 0)
    high = min(high + args.margin_px, depth - 1)

    peak = profile.max()
    step = max(depth // 40, 1)
    print("per-slice spread (specimen is the plateau):")
    for index in range(0, depth, step):
        marker = "|" if low <= index <= high else " "
        bar = "#" * int(profile[index] / peak * 46)
        print(f"  z {index:4d} {marker} {bar}")

    thickness = high - low + 1
    print(f"\nslab: z {low} to {high}, {thickness} of {depth} voxels "
          f"({100 * thickness / depth:.0f}% of the reconstruction)")
    if apix:
        print(f"      {thickness * apix / 10:.0f} nm thick, in a "
              f"{depth * apix / 10:.0f} nm reconstruction")
    print(f"\n  --pytom-search-z {low},{high}")
    print("\nEverything outside that range is vacuum as far as a particle is")
    print("concerned, so searching it can only add false peaks and dilute the")
    print("standard deviation the scores are normalised by.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
