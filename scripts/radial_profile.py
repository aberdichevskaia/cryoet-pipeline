#!/usr/bin/env python3
"""Is there a body inside the map, or only a shell at the edge of the box?

    python3 scripts/radial_profile.py ~/Downloads/*_class0*_average.mrc

Every class average of this dataset has been described as looking like a shell.
There is a reason it would look like one with no particle involved, and it is
geometric: `resample_volume_zyx` rotates each particle with
`padding_mode="zeros"` (src/cryoet_pipeline/rotations.py:175), and the class
average is a plain mean over a fixed member count
(src/cryoet_pipeline/backends/subtomogram.py:1415). A voxel at radius r from the
centre keeps its radius under rotation, so:

- inside the inscribed sphere, r <= box/2, every particle contributes at every
  voxel, and the noise in the average has amplitude sigma/sqrt(N);
- outside it, only the particles whose rotation happened to keep that voxel
  inside the cube contribute, but the divisor is still N — so the amplitude
  falls, reaching roughly a third of the core value at the corners.

That is a ball of the box's own width with quiet corners, put there by the shape
of the box. It is the same artefact whose DC component fabricated a 65 A FSC
peak and was removed by centring each particle; centring killed the step, not
the amplitude gradient, and the gradient is what a surface threshold shows.

So the outline of these maps carries no information. What does carry information
is whether anything sits INSIDE it. A ribosome is 25-30 nm across — at 9.784
A/voxel that is 26-31 voxels, radius 13-16, comfortably inside the r=24 boundary
of a 48 box. Signal and artefact therefore live at different radii, and one
number separates them: the RMS amplitude inside the particle radius over the RMS
in the annulus between it and the box boundary.

- Near 1.0: the map is flat out to the geometric boundary. There is no body,
  only the box.
- Above 1.0: density is concentrated where a particle would be.

No threshold is chosen anywhere here, which is the point — `sdLevel` is a
choice, and four maps compared at one level still leave the reader deciding what
counts as a shape.

Disposable, like the rest of `scripts/`. Its replacement is this ratio recorded
in the averaging QC, so a run that produced only the box says so itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _profile(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RMS amplitude per integer radial shell, about the box centre.

    RMS rather than the mean, because the mean of a mean-subtracted noise map is
    zero at every radius and would show nothing at all. Amplitude is what both
    the signal and the artefact modulate.
    """

    size_z, size_y, size_x = volume.shape
    grid = np.ogrid[
        : size_z, : size_y, : size_x
    ]
    centre = [(size - 1) / 2.0 for size in volume.shape]
    radius = np.sqrt(
        sum((axis - middle) ** 2 for axis, middle in zip(grid, centre, strict=True))
    )
    shells = radius.astype(int)
    counts = np.bincount(shells.ravel())
    total = np.bincount(shells.ravel(), weights=(volume.astype(np.float64) ** 2).ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(total / counts)
    return rms, counts


def _pooled(rms: np.ndarray, counts: np.ndarray, selector: np.ndarray) -> float:
    """RMS over a set of shells, weighted by how many voxels each holds.

    Weighted because an outer shell holds many more voxels than an inner one, and
    an unweighted mean of shell values would let the handful of voxels near the
    centre outvote the rest of the map.
    """

    weight = counts[selector]
    values = rms[selector]
    good = np.isfinite(values) & (weight > 0)
    if not good.any():
        return float("nan")
    return float(
        np.sqrt(np.sum(weight[good] * values[good] ** 2) / np.sum(weight[good]))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", nargs="+", type=Path, help="Average MRC files.")
    parser.add_argument(
        "--particle-diameter-angstrom",
        type=float,
        default=280.0,
        help="What the particle is expected to span. A human ribosome is 250-300 A; "
        "the ratio is not sensitive to the exact figure, but it is reported so "
        "that the radius it implies is visible rather than assumed.",
    )
    parser.add_argument(
        "--pixel-spacing-angstrom",
        type=float,
        default=None,
        help="Overrides the value in the MRC header, which is what should normally "
        "be trusted.",
    )
    args = parser.parse_args()

    import mrcfile

    print(
        f"{'map':<44} {'box':>4} {'r_particle':>10} {'r_box':>6} "
        f"{'inside':>8} {'annulus':>8} {'ratio':>7}"
    )
    print("-" * 96)
    ratios: list[tuple[float, str]] = []
    for path in args.maps:
        with mrcfile.open(path, permissive=True) as handle:
            volume = np.asarray(handle.data, dtype=np.float64)
            header_apix = float(handle.voxel_size.x) or 0.0
        apix = args.pixel_spacing_angstrom or header_apix
        if not apix:
            print(f"{path.name}: no pixel spacing in the header; pass one")
            continue
        if volume.ndim != 3:
            print(f"{path.name}: not a 3D map ({volume.shape})")
            continue

        rms, counts = _profile(volume)
        box = volume.shape[0]
        boundary = box / 2.0
        particle_radius = args.particle_diameter_angstrom / 2.0 / apix
        if particle_radius >= boundary:
            print(
                f"{path.name}: a {args.particle_diameter_angstrom:.0f} A particle needs "
                f"radius {particle_radius:.0f} px, which is outside the r={boundary:.0f} "
                "boundary of this box — the box is too small for the comparison."
            )
            continue

        shells = np.arange(len(rms))
        inside = shells <= particle_radius
        annulus = (shells > particle_radius) & (shells <= boundary)
        core = _pooled(rms, counts, inside)
        ring = _pooled(rms, counts, annulus)
        ratio = core / ring if ring else float("nan")
        ratios.append((ratio, path.name))
        print(
            f"{path.name[-43:]:<44} {box:>4} {particle_radius:>10.1f} "
            f"{boundary:>6.1f} {core:>8.3f} {ring:>8.3f} {ratio:>7.3f}"
        )

    if not ratios:
        return 1

    print(
        f"\ninside  = RMS within radius {args.particle_diameter_angstrom / 2:.0f} A, "
        "where a particle's density would be"
    )
    print("annulus = RMS between that radius and the box boundary, where the")
    print("          rotation artefact lives and a particle does not")
    print()
    spread = max(one for one, _ in ratios) - min(one for one, _ in ratios)
    print(f"ratios span {spread:.3f} across {len(ratios)} maps.")
    print()
    print("Read them together, not one at a time. Four maps at 1.0 mean four boxes")
    print("and no particles, however different their surfaces look at a chosen")
    print("sdLevel. One map clearly above the others is the class worth refining,")
    print("and it is the same class the seed-to-seed comparison should have named —")
    print("if those two disagree, neither is evidence yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
