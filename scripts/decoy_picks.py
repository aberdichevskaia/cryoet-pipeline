#!/usr/bin/env python3
"""A pick list at random positions, to run the whole pipeline on nothing.

    python3 scripts/decoy_picks.py \\
        --registry outputs/ribo/registry.json \\
        --picks-id TS_0005:picks:full:pytom-match-pick:bin8-top800 \\
        --slab 40,110 \\
        --out outputs/ribo/picks/TS_0005

Four class averages of this dataset came back looking alike, and two of them
reproduced across classification seeds beyond anything four hundred shuffles
reached. Both observations are consistent with real particles and equally
consistent with an artefact shared by every box, and nothing computed on the
real picks alone can separate those two readings. This makes the control that
can: the same pipeline, the same tomograms, the same slab, the same number of
boxes, the same starting orientations — and coordinates that are random, so no
box is centred on anything.

Everything except position is copied from the real pick list on purpose:

- **Orientations are carried over row by row**, so the distribution the
  refinement starts from is identical. What is broken is the correspondence
  between an orientation and the density it was matched to, which is the whole
  question.
- **Scores are carried over** too, so any behaviour that depends on the score
  column — ordering, a floor, a gallery sorted by it — is identical.
- **Z is confined to the slab**, because the specimen fills about half the
  reconstruction. A control drawn from the whole volume would sit in vacuum half
  the time and look emptier than the real data for a reason that has nothing to
  do with particles. Get the range from `find_slab.py`.

Then extract, average and classify with the same commands and the same two
seeds, and compare:

- The control's classes look like the real ones and two of them reproduce —
  then the appearance and the reproducibility are both manufactured, and lane B
  has measured the pipeline rather than the specimen.
- The control's classes are featureless and nothing reproduces — then the real
  classes 2 and 4 hold something, and refining the better of them is the next
  step.

Disposable, like the rest of `scripts/`. Its replacement is a `--decoy` flag on
picking that registers this alongside every real pick list, so the control is
paid for automatically rather than when somebody suspects the answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.subtomogram import _read_pytom_star
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    RetentionPolicy,
    StorageRole,
)

_STAR_COLUMNS = (
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnLCCmax",
)


def _draw_positions(
    count: int,
    *,
    bounds: tuple[tuple[float, float], ...],
    minimum_separation_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """`count` positions inside `bounds`, no two closer than the exclusion.

    Drawn in blocks and filtered rather than one at a time, because a rejection
    loop over four thousand positions in a thin slab spends most of its time
    rejecting. If the slab cannot hold the requested number at this separation,
    that is reported rather than looped on forever — a control with a third of
    the boxes is still a control, but only if the difference is known.
    """

    kept = np.empty((count, 3))
    found = 0
    for _ in range(200):
        if found >= count:
            break
        block = np.column_stack(
            [rng.uniform(low, high, size=max(count, 1024)) for low, high in bounds]
        )
        for candidate in block:
            if found >= count:
                break
            if found and np.min(np.linalg.norm(kept[:found] - candidate, axis=1)) < (
                minimum_separation_px
            ):
                continue
            kept[found] = candidate
            found += 1
    return kept[:found]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--picks-id",
        required=True,
        help="The real picks artifact to mirror. Its count, orientations, scores "
        "and tomogram are all taken from it.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Directory for the STAR.")
    parser.add_argument(
        "--slab",
        required=True,
        help="z range the specimen occupies, as LOW,HIGH in tomogram voxels. From "
        "find_slab.py. Sampling outside it puts control boxes in vacuum.",
    )
    parser.add_argument(
        "--variant",
        default="decoy",
        help="Label for the control lane. Everything downstream is selected by it, "
        "so it must not collide with a real lane's label.",
    )
    parser.add_argument(
        "--box-size-px",
        type=int,
        default=48,
        help="Only used to keep positions a half-box from the edges, so the control "
        "loses no boxes to clipping that the real run kept.",
    )
    parser.add_argument(
        "--minimum-separation-px",
        type=float,
        default=None,
        help="Exclusion between control positions. Defaults to the box size, which "
        "is the separation two boxes need to not be largely the same voxels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    low_text, _, high_text = args.slab.partition(",")
    slab_low, slab_high = float(low_text), float(high_text)
    if not slab_high > slab_low:
        print(f"--slab {args.slab} is not a range")
        return 1

    registry = ArtifactRegistry.load(args.registry)
    real = registry.get(args.picks_id)
    if real.kind is not ArtifactKind.PICKS:
        print(f"{args.picks_id} is a {real.kind}, not picks")
        return 1
    tomogram_id = real.parameters.get("input_tomogram_id")
    if not isinstance(tomogram_id, str):
        print(f"{args.picks_id} records no input_tomogram_id")
        return 1
    tomogram = registry.get(tomogram_id)
    if tomogram.shape is None or len(tomogram.shape) != 3:
        print(f"{tomogram_id} records no 3D shape, so the bounds are unknown")
        return 1
    depth, height, width = (int(one) for one in tomogram.shape)
    series = real.parameters["tilt_series_id"]

    rows = _read_pytom_star(real.path)
    print(f"{real.path.name}: {len(rows)} real picks in {series}")
    print(f"{tomogram_id}: {depth} x {height} x {width} (z, y, x)")

    half = args.box_size_px / 2.0
    separation = (
        args.minimum_separation_px
        if args.minimum_separation_px is not None
        else float(args.box_size_px)
    )
    # x/y first, matching the STAR's column order; z last and clipped to the slab.
    bounds = (
        (half, width - half),
        (half, height - half),
        (max(slab_low, half), min(slab_high, depth - half)),
    )
    for name, (low, high) in zip("xyz", bounds, strict=True):
        if not high > low:
            print(f"the {name} range is empty at box {args.box_size_px}: [{low}, {high}]")
            return 1
    span = bounds[2][1] - bounds[2][0]
    print(
        f"drawing inside x [{bounds[0][0]:.0f}, {bounds[0][1]:.0f}], "
        f"y [{bounds[1][0]:.0f}, {bounds[1][1]:.0f}], "
        f"z [{bounds[2][0]:.0f}, {bounds[2][1]:.0f}] — "
        f"{span:.0f} of {depth} voxels in z, at {separation:.0f} px separation"
    )

    rng = np.random.default_rng(args.seed)
    positions = _draw_positions(
        len(rows), bounds=bounds, minimum_separation_px=separation, rng=rng
    )
    if len(positions) < len(rows):
        print(
            f"\nonly {len(positions)} of {len(rows)} positions fit at {separation:.0f} px "
            "separation. The control will have fewer boxes than the real run, which "
            "weakens the comparison but does not invalidate it — lower "
            "--minimum-separation-px to close the gap, and say which was used."
        )

    # Orientations and scores by row, so only position differs from the real list.
    # `strict=False` deliberately: a slab that could not hold every position
    # yields a shorter control, which the message above has already said.
    lines = ["data_particles", "", "loop_"]
    lines.extend(f"_{name} #{position}" for position, name in enumerate(_STAR_COLUMNS, 1))
    for row, (x, y, z) in zip(rows, positions, strict=False):
        lines.append(
            f"{x:.3f} {y:.3f} {z:.3f} "
            f"{row.rot_deg:.6f} {row.tilt_deg:.6f} {row.psi_deg:.6f} "
            f"{row.lccmax if row.lccmax is not None else 0.0:.6f}"
        )
    star = args.out / f"{series}_decoy_{args.variant}_particles.star"
    star.parent.mkdir(parents=True, exist_ok=True)
    star.write_text("\n".join(lines) + "\n")

    artifact = Artifact(
        id=f"{series}:picks:{real.parameters.get('tomogram_branch', 'full')}:decoy:{args.variant}",
        variant=args.variant,
        kind=ArtifactKind.PICKS,
        path=star,
        parent_ids=[tomogram.id],
        shape=(len(positions),),
        pixel_spacing_angstrom=tomogram.pixel_spacing_angstrom,
        binning=tomogram.binning,
        parameters={
            "backend": "decoy",
            "tilt_series_id": series,
            "tomogram_branch": real.parameters.get("tomogram_branch", "full"),
            "input_tomogram_id": tomogram.id,
            "picking_method": "random_positions",
            "mirrors_picks_id": real.id,
            "slab_z": [slab_low, slab_high],
            "minimum_separation_px": separation,
            "seed": args.seed,
            "particle_count": len(positions),
        },
        # Temporary and recomputable: the whole control is a seed plus a slab, so
        # keeping its outputs past the comparison would be storing a gigabyte of
        # particle stack that a one-line command regenerates.
        storage_role=StorageRole.TEMPORARY,
        retention_policy=RetentionPolicy.RECOMPUTE,
        can_recompute=True,
    )
    registry.add(artifact, replace=True)
    registry.write(args.registry)

    print(f"\nwrote {star}")
    print(f"registered {artifact.id}\n")
    print("Extract and average it exactly as the real lane was, with the same two")
    print(f"seeds, selecting it with --variant {args.variant}. Then run")
    print("compare_classifications.py on the control's two seeds.")
    print()
    print("The control's own numbers are the baseline that has been missing: a")
    print("share above the shuffle ceiling means reproducible, and the control")
    print("says whether reproducible needs a particle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
