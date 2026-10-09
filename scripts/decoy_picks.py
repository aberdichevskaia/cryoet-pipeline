#!/usr/bin/env python3
"""A pick list at random positions, to run the whole pipeline on nothing.

    python3 scripts/decoy_picks.py \\
        --registry outputs/ribo/artifacts.json \\
        --picks-id TS_0005:picks:full:pytom-match-pick:bin8-top800 \\
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
- **Depths are a permutation of the real picks' own z values**, so the control
  samples the same depths as the real list without anyone having to know where
  the specimen is. The first version asked for a slab range from
  `find_slab.py` instead, and that script returned 84% of the reconstruction on
  the first real tomogram — its profile had no plateau, so first-crossing to
  last-crossing spanned almost everything. Building the control on that range
  would have drawn it from the whole volume: the exact unfairness the slab
  argument existed to prevent.
- **How much density each box holds is measured, not assumed.** Permuting the
  depths fixes the z marginal but not a tilted lamella, where membership depends
  on x and y too. `--check-variance` compares the in-box standard deviation of
  real and control boxes and says whether the control is sampling the same
  material, so a difference in the averages can be attributed to centring rather
  than to one set being emptier.

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
    depths: np.ndarray | None,
    minimum_separation_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """`count` positions, no two closer than the exclusion.

    x and y are uniform inside `bounds`. z is drawn from `depths` — a permutation
    of the real picks' own z values — when it is given, and uniform inside the
    third bound otherwise.

    Permuting the real depths is the default because it needs no estimate of
    where the specimen is. Asking for a slab range first looked like the careful
    option, and then `find_slab.py` returned 84% of the reconstruction on the
    first tomogram it was pointed at: its profile had no plateau, so
    first-crossing-to-last-crossing spanned nearly everything. A control built on
    that range would have been drawn from the whole volume, which is exactly the
    unfair control the slab argument existed to avoid.

    Taking the depths from the real picks makes the control's z distribution
    identical to the real one by construction, so the question of where the
    specimen is never arises. What it does not fix is a tilted lamella, where
    membership depends on x and y as well; `--check-variance` measures that
    residual difference instead of assuming it away.

    Drawn in blocks and filtered rather than one at a time, because a rejection
    loop over four thousand positions spends most of its time rejecting. If the
    volume cannot hold the requested number at this separation, that is reported
    rather than looped on forever — a control with fewer boxes is still a
    control, but only if the difference is known.
    """

    kept = np.empty((count, 3))
    found = 0
    for _ in range(200):
        if found >= count:
            break
        size = max(count, 1024)
        columns = [rng.uniform(low, high, size=size) for low, high in bounds[:2]]
        if depths is not None:
            columns.append(rng.choice(depths, size=size, replace=True))
        else:
            columns.append(rng.uniform(bounds[2][0], bounds[2][1], size=size))
        for candidate in np.column_stack(columns):
            if found >= count:
                break
            if found and np.min(np.linalg.norm(kept[:found] - candidate, axis=1)) < (
                minimum_separation_px
            ):
                continue
            kept[found] = candidate
            found += 1
    return kept[:found]


def _nearest_neighbour(positions: np.ndarray) -> np.ndarray:
    """Distance from each position to its closest other position.

    Computed on the real list to set the control's exclusion, because guessing
    it was wrong in a way that was invisible until it was printed. The first
    version defaulted the exclusion to the box size, 48 voxels, on the reasoning
    that two boxes closer than that are largely the same voxels. That reasoning
    is fine and the number is still wrong here, twice over:

    - It does not fit. 800 positions at 48 voxels apart in a 464 x 464 x ~260
      volume is past the jamming limit of random sequential addition, so the
      sampler saturated at about 425 and reported the shortfall — the real cause
      being an impossible constraint, not an unlucky draw.
    - It does not match. pytom removes further peaks within HALF the diameter it
      is given, so the real list already contains pairs far closer than 48, and a
      control held to 48 is more spread out than the thing it controls.

    Measuring the real list settles both: an exclusion equal to its own minimum
    separation is achievable by construction, and no control pair is closer than
    any real pair.
    """

    if len(positions) < 2:
        return np.zeros(0)
    # Direct N^2 on a few thousand points: about 6e6 distances, well under a
    # second, and it avoids a scipy dependency for one number.
    distances = np.linalg.norm(
        positions[:, None, :] - positions[None, :, :], axis=-1
    )
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _describe(values: np.ndarray, label: str) -> None:
    """Percentiles rather than mean and sd alone.

    A mean cannot show a tail, and a tail is what a real minority among mostly
    noise looks like: the published work on this dataset found about 14% real
    among the top 800, and 14% of boxes holding a dense particle moves a mean by
    almost nothing while moving the upper percentiles visibly.
    """

    if values.size == 0:
        print(f"  {label}: nothing measured")
        return
    points = np.percentile(values, [1, 10, 50, 90, 99])
    print(
        f"  {label}: p1 {points[0]:.1f}  p10 {points[1]:.1f}  "
        f"median {points[2]:.1f}  p90 {points[3]:.1f}  p99 {points[4]:.1f}  "
        f"(n={values.size})"
    )


def _box_spread(
    tomogram_path: Path,
    positions: np.ndarray,
    *,
    box_size: int,
    sample: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard deviation inside the box at each of `sample` random positions.

    The one number that says whether a control is fair. The control differs from
    the real pick list in exactly one intended way — nothing is centred on
    anything — but it can differ in a second, unintended way: if the real picks
    sit preferentially inside the specimen and the control lands partly outside
    it, then the control has less density in the box and would look emptier for a
    reason that has nothing to do with centring. Comparing the two spread
    distributions makes that visible. Matching distributions mean the control is
    sampling the same material; a shifted one means the comparison needs
    subsampling to match, and says by how much.

    Subsampled because reading four thousand boxes out of a chunked Zarr costs
    minutes and a few hundred is plenty to compare two distributions.
    """

    import zarr

    array = zarr.open(tomogram_path, mode="r")
    half = box_size // 2
    chosen = positions[
        rng.choice(len(positions), size=min(sample, len(positions)), replace=False)
    ]
    spreads = []
    for x, y, z in chosen:
        z_start, y_start, x_start = (int(round(one)) - half for one in (z, y, x))
        if min(z_start, y_start, x_start) < 0:
            continue
        crop = np.asarray(
            array[
                z_start : z_start + box_size,
                y_start : y_start + box_size,
                x_start : x_start + box_size,
            ],
            dtype=np.float32,
        )
        if crop.shape != (box_size, box_size, box_size):
            continue
        spreads.append(float(crop.std()))
    return np.array(spreads)


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
        default=None,
        help="z range to draw from, as LOW,HIGH in tomogram voxels. Normally leave "
        "it off: without it the control takes its depths from the real picks, so "
        "the z distributions match by construction and no estimate of where the "
        "specimen is has to be trusted. Pass a range only to test a different "
        "depth distribution on purpose.",
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
        help="Exclusion between control positions. Defaults to the real list's own "
        "minimum nearest-neighbour distance, which is achievable by construction "
        "and leaves no control pair closer than any real pair. Setting it to the "
        "box size instead is what saturated the sampler at 425 of 800.",
    )
    parser.add_argument(
        "--check-variance",
        type=int,
        default=300,
        help="Boxes to sample from each set when comparing how much density the "
        "real and control boxes contain — the check that says whether the control "
        "is fair. Zero skips it, which means the fairness is assumed rather than "
        "measured.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    slab: tuple[float, float] | None = None
    if args.slab:
        low_text, _, high_text = args.slab.partition(",")
        slab = (float(low_text), float(high_text))
        if not slab[1] > slab[0]:
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

    real_positions = np.array([[row.x_px, row.y_px, row.z_px] for row in rows])
    real_neighbour = _nearest_neighbour(real_positions)
    half = args.box_size_px / 2.0
    if args.minimum_separation_px is not None:
        separation = args.minimum_separation_px
    elif real_neighbour.size:
        separation = float(real_neighbour.min())
    else:
        separation = float(args.box_size_px)
    print("\nnearest-neighbour distance among the real picks, in voxels:")
    _describe(real_neighbour, "real ")
    print(
        f"  exclusion for the control: {separation:.1f} px"
        + ("" if args.minimum_separation_px is None else " (given)")
    )
    # `.size` first: np.percentile on an empty array does not return a value to
    # compare against.
    if real_neighbour.size and separation > float(np.percentile(real_neighbour, 1)):
        closer = int((real_neighbour < separation).sum())
        print(
            f"  NOTE: {closer} of {len(real_neighbour)} real picks sit closer to a "
            "neighbour than the control is allowed to, so the control is slightly "
            "more spread out than the list it controls."
        )

    # x/y first, matching the STAR's column order; z last.
    bounds = (
        (half, width - half),
        (half, height - half),
        (max(slab[0], half), min(slab[1], depth - half)) if slab else (half, depth - half),
    )
    for name, (low, high) in zip("xyz", bounds, strict=True):
        if not high > low:
            print(f"the {name} range is empty at box {args.box_size_px}: [{low}, {high}]")
            return 1

    real_depths = real_positions[:, 2]
    depths = None
    if slab is None:
        # Clipped to the half-box margin so a real pick near the edge cannot
        # place a control box outside the volume; the real run lost those to
        # `_crop_start` anyway.
        depths = np.clip(real_depths, bounds[2][0], bounds[2][1])
        print(
            f"z taken from the real picks: mean {real_depths.mean():.0f}, "
            f"sd {real_depths.std():.0f}, range "
            f"[{real_depths.min():.0f}, {real_depths.max():.0f}] of {depth} — "
            "so the depth distributions match by construction"
        )
    else:
        print(
            f"z drawn uniformly in [{bounds[2][0]:.0f}, {bounds[2][1]:.0f}], "
            f"against real picks at mean {real_depths.mean():.0f} "
            f"sd {real_depths.std():.0f} — the distributions will NOT match"
        )
    print(
        f"x/y drawn uniformly in [{bounds[0][0]:.0f}, {bounds[0][1]:.0f}] and "
        f"[{bounds[1][0]:.0f}, {bounds[1][1]:.0f}], at {separation:.0f} px separation"
    )

    rng = np.random.default_rng(args.seed)
    positions = _draw_positions(
        len(rows),
        bounds=bounds,
        depths=depths,
        minimum_separation_px=separation,
        rng=rng,
    )
    if len(positions) < len(rows):
        print(
            f"\nonly {len(positions)} of {len(rows)} positions fit at "
            f"{separation:.1f} px separation. Past the jamming limit of random "
            "sequential addition, the sampler cannot place more however long it "
            "tries, so this is an impossible constraint rather than an unlucky "
            "draw — lower --minimum-separation-px, and say which was used."
        )
    else:
        _describe(_nearest_neighbour(positions), "decoy")
        print(
            "  A control whose neighbours are much further apart than the real "
            "list's overlaps less between boxes, which is a second difference "
            "besides centring. Compare the medians on the two lines above."
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

    spread_summary: dict[str, float] | None = None
    if args.check_variance > 0:
        print(f"\ncomparing how much density is in the box, {args.check_variance} each ...")
        try:
            real_spread = _box_spread(
                tomogram.path,
                real_positions,
                box_size=args.box_size_px,
                sample=args.check_variance,
                rng=np.random.default_rng(args.seed + 1),
            )
            decoy_spread = _box_spread(
                tomogram.path,
                positions,
                box_size=args.box_size_px,
                sample=args.check_variance,
                rng=np.random.default_rng(args.seed + 2),
            )
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"  could not read the tomogram at {tomogram.path}: {exc}")
            print("  the control is still written; its fairness is now unmeasured.")
        else:
            if real_spread.size and decoy_spread.size:
                ratio = float(decoy_spread.mean() / real_spread.mean())
                spread_summary = {
                    "real_mean": float(real_spread.mean()),
                    "decoy_mean": float(decoy_spread.mean()),
                    "ratio": ratio,
                }
                print(
                    f"  real  picks: mean {real_spread.mean():.2f} "
                    f"+/- {real_spread.std():.2f} over {real_spread.size} boxes"
                )
                print(
                    f"  decoy picks: mean {decoy_spread.mean():.2f} "
                    f"+/- {decoy_spread.std():.2f} over {decoy_spread.size} boxes"
                )
                print(f"  ratio decoy/real: {ratio:.3f}")
                # The percentiles are printed for the LEFT tail, not the right.
                #
                # They were added expecting the opposite: a minority of boxes
                # holding a dense particle should lift p90 while barely moving the
                # mean, so p90 looked like a way to see the published 14% among
                # the top 800. On the real data p90 came back identical to 0.1 for
                # both sets in all five series, and the reason is that this
                # statistic has no headroom. A box is 110592 voxels, so its
                # standard deviation estimates the LOCAL one closely, and in a
                # bin8 tomogram that quantity is set by reconstruction noise
                # rather than by biology — a ribosome occupies about a quarter of
                # the box at a contrast far below the noise, and moves the box
                # standard deviation by less than the spread between boxes.
                #
                # So this cannot detect particles, and no version of it can. What
                # it does measure is the left tail: boxes that sit partly outside
                # the specimen are genuinely quieter, and comparing p1 and p10
                # says whether the two sets sample the same material. That is the
                # question the control needs answered, and it is the only one
                # answered here.
                _describe(real_spread, "real ")
                _describe(decoy_spread, "decoy")
                lower = float(
                    np.percentile(real_spread, 10) / np.percentile(decoy_spread, 10)
                )
                print(
                    f"  p10 real / p10 decoy: {lower:.3f} — above 1 means the real "
                    "picks avoid the emptiest boxes more than random positions do, "
                    "which is template matching preferring material over vacuum "
                    "and not evidence of a particle"
                )
                spread_summary["p10_ratio"] = lower
                if 0.9 <= ratio <= 1.1:
                    print(
                        "  Within a tenth, so the control boxes hold as much "
                        "material as the real ones and the only difference left "
                        "is that nothing is centred. That is a fair control."
                    )
                else:
                    direction = "less" if ratio < 1 else "more"
                    print(
                        f"  The control boxes hold {direction} material than the "
                        "real ones. A difference in the averages could then come "
                        "from that rather than from centring, so read the "
                        "comparison as a bound rather than a match, and say this "
                        "number alongside it."
                    )

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
            "depth_source": "real_picks" if slab is None else "uniform_in_slab",
            "slab_z": list(slab) if slab else None,
            "minimum_separation_px": separation,
            "seed": args.seed,
            "particle_count": len(positions),
            # Recorded because it is what makes the control's result readable
            # later: a comparison whose control held different material is a
            # different comparison, and that fact is not recoverable from the
            # coordinates.
            "box_spread_vs_real": spread_summary,
        },
        # Temporary and recomputable: the whole control is a seed and a
        # permutation, so keeping its outputs past the comparison would be
        # storing a gigabyte of particle stack that one command regenerates.
        storage_role=StorageRole.TEMPORARY,
        retention_policy=RetentionPolicy.RECOMPUTE,
        can_recompute=True,
    )
    registry.add(artifact, replace=True)
    registry.commit(args.registry)

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
