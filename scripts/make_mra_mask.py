#!/usr/bin/env python3
"""A soft cylinder mask for Dynamo MRA that holds the particle but not its membrane.

    python3 scripts/make_mra_mask.py --box 52 --pixel 13.44 --diameter 340 \\
        --z-from -1 --z-to 10 --out outputs/ryr1/dynamo/ryr1_cylinder_mask.mrc \\
        --average outputs/ryr1/dynamo/bin8-2x/mra_bbn/results/ite_0005/averages/average_ref_001_ite_0005.em

The sphere scripts/dynamo_mra_setup.m makes by default holds a membrane-bound
particle's membrane too: in mra_bbn (RyR1, 2026-10-19) a 47 nm sphere took in the
whole bilayer, which outweighs the protein, and the classes sorted by membrane
(class 3 was nearly all bilayer and had the best CC). This mask is a cylinder about
the box's z axis, which for a template of a membrane protein is the membrane's
normal (RyR1's C4 axis), from --z-from to --z-to pixels about the box centre,
with the radius where the particle is widest plus 3 px (from --diameter, unless
--radius is given) and a raised-cosine edge of --edge px all round, so no edge is
hard. The z range has to be read off an average of these particles: the
defaults, -1 to +10 px at 13.44 A/px, were read by eye off the picture of the
average at the picks' template-matching poses in mra_bbn's classification report
(page 4: the membrane's middle near -6 nm, the protein from about 0 to +11 nm),
not measured, and the membrane there is curved. The overlay is the check.

The centre is the box centre as Dynamo's sphere masks have it, (N - 1) / 2 in
zero-based voxels, where the export puts each particle. With --average, a PNG
beside the mask shows that map's central sections and a strip of z sections with
the mask's half-height contour on them: look at it before using the mask.

Give the mask to the MRA job as MASK_FILE (an absolute path). Disposable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np


def cylinder_mask(
    box: int, radius: float, z_from: float, z_to: float, edge: float
) -> np.ndarray:
    """1 inside the cylinder, a raised cosine over `edge` px outside, 0 beyond; (z, y, x)."""

    centre = (box - 1) / 2.0
    z, y, x = np.indices((box, box, box), dtype=np.float64) - centre
    radial = np.maximum(np.hypot(x, y) - radius, 0.0)
    axial = np.maximum(np.maximum(z - z_to, z_from - z), 0.0)
    distance = np.hypot(radial, axial)
    if edge <= 0:
        return (distance == 0).astype(np.float32)
    ramp = 0.5 * (1.0 + np.cos(np.pi * np.clip(distance / edge, 0.0, 1.0)))
    return ramp.astype(np.float32)


def _overlay(average: np.ndarray, mask: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    box = average.shape[0]
    centre = box // 2
    sections = [
        ("XY, central z", average[centre], mask[centre]),
        ("XZ, central y", average[:, centre, :], mask[:, centre, :]),
        ("YZ, central x", average[:, :, centre], mask[:, :, centre]),
    ]
    strip = list(range(max(0, centre - 15), min(box, centre + 16), 3))
    figure = plt.figure(figsize=(12, 5.6))
    grid = figure.add_gridspec(2, max(len(strip), 3), height_ratios=[2.2, 1], hspace=0.25)
    low, high = np.percentile(average, [1, 99])
    for column, (name, image, outline) in enumerate(sections):
        axes = figure.add_subplot(grid[0, column * len(strip) // 3 : (column + 1) * len(strip) // 3])
        axes.imshow(image, cmap="gray", vmin=low, vmax=high, origin="lower")
        axes.contour(outline, levels=[0.5], colors="#2a78d6", linewidths=1.2)
        axes.set_title(name, fontsize=9)
        axes.set_xticks([])
        axes.set_yticks([])
    for column, z in enumerate(strip):
        axes = figure.add_subplot(grid[1, column])
        axes.imshow(average[z], cmap="gray", vmin=low, vmax=high, origin="lower")
        if mask[z].max() >= 0.5:
            axes.contour(mask[z], levels=[0.5], colors="#2a78d6", linewidths=1.0)
        axes.set_title(f"z {z - (box - 1) / 2:+.1f} px", fontsize=8)
        axes.set_xticks([])
        axes.set_yticks([])
    figure.suptitle(title, fontsize=10)
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--box", type=int, required=True, help="the particles' box, px")
    parser.add_argument("--pixel", type=float, required=True, help="their pixel size, A")
    parser.add_argument("--diameter", type=float, default=None,
                        help="the particle where it is widest, A; the radius is half of it + 3 px")
    parser.add_argument("--radius", type=float, default=None, help="the radius, px, instead")
    parser.add_argument("--z-from", type=float, default=-1.0,
                        help="the cylinder's lower end, px from the box centre (default -1)")
    parser.add_argument("--z-to", type=float, default=10.0,
                        help="its upper end, px from the box centre (default +10)")
    parser.add_argument("--edge", type=float, default=2.0, help="raised-cosine edge, px")
    parser.add_argument("--out", type=Path, required=True, help="the mask, .mrc")
    parser.add_argument("--average", type=Path, default=None,
                        help="an average of these particles (EM or MRC) to draw the mask on")
    args = parser.parse_args()

    if (args.radius is None) == (args.diameter is None):
        parser.error("give one of --diameter and --radius")
    radius = (
        args.radius if args.radius is not None else round(args.diameter / 2 / args.pixel + 3)
    )
    if not args.z_from < args.z_to:
        parser.error("--z-from must lie below --z-to")
    reach = max(radius, abs(args.z_from), abs(args.z_to)) + args.edge
    if reach > (args.box - 1) / 2.0:
        parser.error(f"the mask reaches {reach:g} px from the centre, past the "
                     f"{args.box} px box's {(args.box - 1) / 2:g}")

    mask = cylinder_mask(args.box, radius, args.z_from, args.z_to, args.edge)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(args.out, overwrite=True) as handle:
        handle.set_data(mask)
        handle.voxel_size = args.pixel
    height = args.z_to - args.z_from
    print(f"wrote {args.out}: cylinder r {radius:g} px ({radius * args.pixel:.0f} A), "
          f"z {args.z_from:+g}..{args.z_to:+g} px ({height * args.pixel:.0f} A tall), "
          f"edge {args.edge:g} px; {int((mask > 0.5).sum())} voxels above 0.5")

    if args.average is not None:
        from cryoet_pipeline.reports.dynamo_project import read_map

        average = read_map(args.average)
        if average.shape != mask.shape:
            raise SystemExit(f"{args.average} is {average.shape}, the mask {mask.shape}")
        picture = args.out.with_suffix(".png")
        _overlay(average, mask, picture,
                 f"{args.out.name} on {args.average.name}: blue is the mask's half height")
        print(f"wrote {picture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
