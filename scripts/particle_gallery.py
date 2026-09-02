#!/usr/bin/env python3
"""A montage of aligned particles, to be looked at rather than measured.

    python3 scripts/particle_gallery.py \\
        --stack outputs/ribo/particles/TS_0006/TS_0006_cubic-crop_bin8_particles.zarr \\
        --metadata outputs/ribo/averages/TS_0005_and_4_more/..._particles.star \\
        --out gallery.png

Every number computed on this data so far answers a narrow question and several
have been misread — a benchmark on the wrong sigma scale, an excess manufactured
by a template mask, a resolution pinned by shell quantisation. Looking at the
particles is the check none of those can substitute for, and it is the one a
person with the right training can make in seconds.

Three decisions here are not cosmetic, because each one has already cost a round
trip on this project:

- **Density is negated.** Protein is dark in a tomogram, so the raw values are
  negative where the particle is, and displaying them unchanged shows solvent.
  Half an hour went into looking at the wrong side of a histogram before this
  was noticed.
- **Each particle is low-passed** to the resolution the tilt-series alignment
  supports. Below that there is no signal at any binning, so unfiltered voxels
  show speckle whether or not a particle is there.
- **Each particle is shown at its refined orientation**, so the gallery answers
  "do the aligned particles look alike" rather than "what was in the box".

Disposable, like the rest of `scripts/`. It reads this project's own formats
rather than a third-party tool's, so the failure modes are ours.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import zarr

from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.particles import ParticleRecord, read_particle_star
from cryoet_pipeline.rotations import (
    center_particle_volume,
    relion_euler_to_matrix,
    resample_volume_zyx,
)


def _aligned_slice(
    stack: Any,
    record: ParticleRecord,
    *,
    low_pass_angstrom: float,
    pixel_spacing_angstrom: float,
    thickness: int,
) -> np.ndarray:
    """One particle at its refined orientation, as a central slab projection.

    A slab rather than a single slice because one voxel plane through a noisy
    particle is mostly noise: averaging a few planes is what makes a shape
    visible, and it is what a person does by eye when scrolling through Z.
    """

    volume = center_particle_volume(
        torch.as_tensor(np.asarray(stack[record.index]), dtype=torch.float32)
    )
    aligned = resample_volume_zyx(
        volume,
        relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
        np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px]),
        device=torch.device("cpu"),
    )
    aligned = low_pass_filter_volume(
        aligned,
        cutoff_angstrom=low_pass_angstrom,
        pixel_spacing_angstrom=pixel_spacing_angstrom,
    )
    middle = aligned.shape[0] // 2
    half = max(thickness // 2, 1)
    slab = aligned[middle - half : middle + half].mean(dim=0)
    # Negated so protein reads bright: it is dark in the tomogram, and showing
    # it unchanged shows the solvent instead.
    return -slab.numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", required=True, type=Path, help="Particle stack Zarr.")
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Particle STAR. Use the REFINED one from average-particles to see "
        "aligned particles; the extraction STAR shows them as they were boxed.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pixel-spacing-angstrom", type=float, default=9.784)
    parser.add_argument(
        "--low-pass-angstrom",
        type=float,
        default=35.0,
        help="What the tilt-series alignment supports. Finer shows speckle.",
    )
    parser.add_argument("--slab-thickness-px", type=int, default=6)
    parser.add_argument("--count", type=int, default=64, help="How many to show.")
    parser.add_argument(
        "--class-number",
        type=int,
        default=None,
        help="Show only this class, to look at what classification separated.",
    )
    parser.add_argument(
        "--order",
        choices=("score", "file"),
        default="score",
        help="score puts the best-matching first, which is where a real particle "
        "is most likely to be; file keeps the original order.",
    )
    args = parser.parse_args()

    records = read_particle_star(args.metadata)
    if args.class_number is not None:
        records = [one for one in records if one.class_number == args.class_number]
        if not records:
            print(f"no particles in class {args.class_number}")
            return 1
    if args.order == "score":
        records.sort(key=lambda one: (one.score is None, -(one.score or 0.0)))

    stack = cast(Any, zarr.open(args.stack, mode="r"))
    available = int(stack.shape[0])
    usable = [one for one in records if one.index < available]
    if len(usable) != len(records):
        print(
            f"{len(records) - len(usable)} of {len(records)} records index past the "
            f"end of this stack ({available} particles). A refined STAR from a "
            "cross-tilt-series average indexes every stack it averaged, so pair it "
            "with the stack it was written for, or pass a single series' STAR."
        )
    if not usable:
        return 1
    shown = usable[: args.count]

    tiles = [
        _aligned_slice(
            stack,
            record,
            low_pass_angstrom=args.low_pass_angstrom,
            pixel_spacing_angstrom=args.pixel_spacing_angstrom,
            thickness=args.slab_thickness_px,
        )
        for record in shown
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.3))
    flat = np.atleast_1d(axes).ravel()
    # One contrast range across the whole montage, so tiles are comparable: a
    # per-tile range makes pure noise look as structured as a particle.
    stacked = np.stack(tiles)
    low, high = np.percentile(stacked, [2.0, 98.0])
    for panel, tile, record in zip(flat, tiles, shown, strict=False):
        panel.imshow(tile, cmap="gray", vmin=low, vmax=high)
        panel.set_title(
            f"{record.index}" + (f" · {record.score:.3f}" if record.score else ""),
            fontsize=5,
        )
    for panel in flat:
        panel.set_xticks([])
        panel.set_yticks([])
    for panel in flat[len(tiles) :]:
        panel.axis("off")
    figure.suptitle(
        f"{len(tiles)} of {len(usable)} aligned particles, low-passed to "
        f"{args.low_pass_angstrom:.0f} A, protein bright",
        fontsize=8,
    )
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=180)
    plt.close(figure)

    print(f"wrote {args.out} — {len(tiles)} particles")
    print("One contrast range across all tiles, so they can be compared by eye.")
    print("A real population looks alike from tile to tile; noise does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
