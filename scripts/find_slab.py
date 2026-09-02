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


def _profiles(tomogram: Any, *, stride: int, smooth: int) -> dict[str, np.ndarray]:
    """Three per-slice statistics, because only one of them sees the specimen.

    The first attempt used the raw per-slice standard deviation and found
    nothing: it varied by 15% across a reconstruction whose specimen fills half.
    Back-projection spreads noise through the whole volume, so vacuum is not
    empty — it holds reconstruction noise of much the same amplitude, and
    variance cannot tell the two apart.

    What does tell them apart is correlation. Real density is smooth between
    neighbouring voxels and noise is not, so blurring a slice keeps the specimen
    and cancels the vacuum — which is exactly why the lamella becomes visible in
    3dmod when ten slices are averaged. `smoothed` is that measurement.

    `mean` is offered too because protein is dark, so a slice through the
    specimen should sit lower than one through vacuum, and that needs no
    assumption about correlation lengths at all.
    """

    from scipy.ndimage import uniform_filter

    depth = tomogram.shape[0]
    raw, smoothed, means = [], [], []
    for index in range(depth):
        plane = np.asarray(tomogram[index, ::stride, ::stride], dtype=np.float64)
        raw.append(float(plane.std()))
        means.append(float(plane.mean()))
        # Blur in-plane: what survives is what was spatially correlated.
        smoothed.append(float(uniform_filter(plane, size=smooth).std()))
    return {
        "smoothed sd": np.array(smoothed),
        "raw sd": np.array(raw),
        "mean": -np.array(means),  # negated: protein is dark, so lower is more
    }


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
        "--smooth-px",
        type=int,
        default=8,
        help="In-plane blur before measuring spread. Larger keeps only coarser "
        "structure, which is what distinguishes specimen from reconstruction noise.",
    )
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

    profiles = _profiles(tomogram, stride=args.stride, smooth=args.smooth_px)
    step = max(depth // 40, 1)
    contrasts: dict[str, float] = {}
    for name, profile in profiles.items():
        baseline = float(np.sort(profile)[: max(len(profile) // 10, 1)].mean())
        span = float(profile.max()) - baseline
        contrasts[name] = span / abs(baseline) if baseline else 0.0

    print("contrast between the quietest slices and the peak, per statistic:")
    for name, value in contrasts.items():
        print(f"  {name:12} {value:+7.2%}")
    chosen = max(contrasts, key=lambda name: contrasts[name])
    print(f"\nusing {chosen!r} — the one that separates specimen from vacuum best.")
    print("A statistic that shows a few percent is not seeing the specimen at all;")
    print("compare the numbers above before trusting the boundary.\n")

    profile = profiles[chosen]
    low, high = _boundaries(profile, fraction=args.fraction)
    low = max(low - args.margin_px, 0)
    high = min(high + args.margin_px, depth - 1)

    floor = float(profile.min())
    peak = float(profile.max())
    print(f"per-slice {chosen} (specimen is the plateau):")
    for index in range(0, depth, step):
        marker = "|" if low <= index <= high else " "
        scaled = (profile[index] - floor) / (peak - floor) if peak > floor else 0.0
        print(f"  z {index:4d} {marker} {'#' * int(scaled * 46)}")

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
