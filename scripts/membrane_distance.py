#!/usr/bin/env python3
"""How far each pick sits from the nearest membrane, against the same for random ones.

    python3 scripts/membrane_distance.py \\
        --registry outputs/ribo/artifacts.json \\
        --real bin8-top800 --decoy decoy \\
        --series TS_0005 TS_0006 TS_0007 TS_0008 TS_0009

The only test of these picks left that does not run through the template.

Everything else lane B measured about identity turned out to be printed by the
orientation search: a global search seeded by the template gives 0.914 FSC
against that template and 2.277 radial concentration from boxes with nothing at
their centre. Both statistics are about SHAPE, and shape here is derived from
orientations that came from the template being compared against, so the
reasoning is circular and no refinement parameter fixes it.

Positions are not. A pick's coordinate owes nothing to the template's shape, and
where ribosomes sit relative to ER membranes is biology: a translating
membrane-bound ribosome has its exit tunnel at the translocon, so its CENTRE
sits about one ribosome radius from the membrane — of order 120 to 150 A. Free
cytosolic ribosomes sit anywhere. So a real population should show a bump at
that offset on top of a broad background, and random positions cannot produce
one.

**Read the shape, not the mean.** This is the trap the whole test turns on. Real
picks might sit nearer membranes than random positions for a reason that has
nothing to do with ribosomes: both membranes and template-matching peaks prefer
dense cytoplasm over ice, so a mean shift is expected either way. What a density
preference cannot manufacture is a peak at one particular distance. The box
density check in `decoy_picks.py` already showed the two sets hold equal
material to within 1.4%, which weakens that confound, and looking at the
histogram rather than its mean removes what is left of it.

A null that is not a null: if the segmentation is empty, or fills the volume,
every distance is degenerate and the comparison is meaningless. Both are
refused rather than reported.

Disposable, like the rest of `scripts/`. Its replacement is roadmap section 9,
the geometric layer, where a distance-to-surface is a property the pipeline
records for every pick rather than something a script recomputes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.subtomogram import _read_pytom_star
from cryoet_pipeline.models import Artifact, ArtifactKind

_MEMBRANE_FRACTION_FLOOR = 1e-5
_MEMBRANE_FRACTION_CEILING = 0.5


def _picks_for(registry: ArtifactRegistry, series: str, variant: str) -> Artifact:
    matches = [
        one
        for one in registry.by_kind(ArtifactKind.PICKS)
        if one.parameters.get("tilt_series_id") == series and one.variant == variant
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one picks artifact for {series} with variant "
            f"{variant!r}, found {len(matches)}"
            + ("" if not matches else ": " + ", ".join(one.id for one in matches))
        )
    return matches[0]


def _segmentation_for(registry: ArtifactRegistry, series: str) -> Artifact:
    matches = [
        one
        for one in registry.by_kind(ArtifactKind.SEGMENTATION)
        if one.parameters.get("tilt_series_id") == series
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one segmentation for {series}, found {len(matches)}"
        )
    return matches[0]


def _distance_field(segmentation: Artifact, threshold: float) -> tuple[Any, float]:
    """Distance in angstrom from every voxel to the nearest membrane voxel."""

    from scipy.ndimage import distance_transform_edt

    import zarr

    array = cast(Any, zarr.open(segmentation.path, mode="r"))
    volume = np.asarray(array[:], dtype=np.float32)
    membrane = volume >= threshold
    fraction = float(membrane.mean())
    if fraction < _MEMBRANE_FRACTION_FLOOR:
        raise SystemExit(
            f"{segmentation.id}: only {fraction:.2%} of voxels are above "
            f"{threshold}, which is no membrane at all. Every distance would be "
            "measured to whatever few voxels survived; lower --threshold or check "
            "whether this segmentation holds probabilities or a binary mask."
        )
    if fraction > _MEMBRANE_FRACTION_CEILING:
        raise SystemExit(
            f"{segmentation.id}: {fraction:.2%} of voxels are above {threshold}, so "
            "'distance to membrane' is a distance to almost everywhere and cannot "
            "discriminate. Raise --threshold."
        )
    spacing = segmentation.pixel_spacing_angstrom
    if spacing is None or spacing <= 0.0:
        raise SystemExit(f"{segmentation.id} records no voxel spacing")
    # The distance transform runs on the COMPLEMENT: it measures how far each
    # voxel is from the nearest True, so the membrane must be the False set.
    distance = distance_transform_edt(~membrane) * spacing
    return distance, fraction


def _sample(distance: Any, rows: list[Any]) -> np.ndarray:
    """The distance field at each pick, in angstrom.

    The STAR carries x, y, z in tomogram voxels and the volume is indexed
    (z, y, x), so the order is reversed here rather than anywhere else.
    """

    depth, height, width = distance.shape
    out: list[float] = []
    for row in rows:
        z = int(round(row.z_px))
        y = int(round(row.y_px))
        x = int(round(row.x_px))
        if not (0 <= z < depth and 0 <= y < height and 0 <= x < width):
            continue
        out.append(float(distance[z, y, x]))
    return np.array(out)


def _histogram(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(counts.sum(), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--series", required=True, nargs="+")
    parser.add_argument("--real", required=True, help="Variant of the real picks.")
    parser.add_argument("--decoy", required=True, help="Variant of the control picks.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Membrane cut-off on the segmentation. The resulting membrane voxel "
        "fraction is printed; a sane one is a fraction of a per cent to a few per "
        "cent, and the script refuses outside that.",
    )
    parser.add_argument(
        "--bin-angstrom", type=float, default=50.0, help="Histogram bin width."
    )
    parser.add_argument(
        "--max-angstrom",
        type=float,
        default=800.0,
        help="Furthest distance shown. Beyond this everything is bulk cytosol.",
    )
    parser.add_argument(
        "--expected-offset-angstrom",
        type=float,
        nargs=2,
        default=(100.0, 180.0),
        help="Where a membrane-bound ribosome's CENTRE should sit: about one "
        "ribosome radius from the membrane. The excess in this band is summarised "
        "separately, but the histogram is what to read.",
    )
    args = parser.parse_args()

    registry = ArtifactRegistry.load(args.registry)
    edges = np.arange(0.0, args.max_angstrom + args.bin_angstrom, args.bin_angstrom)
    real_all: list[np.ndarray] = []
    decoy_all: list[np.ndarray] = []

    for series in args.series:
        real = _picks_for(registry, series, args.real)
        decoy = _picks_for(registry, series, args.decoy)
        segmentation = _segmentation_for(registry, series)

        # The one check that makes the coordinates comparable at all. Picks and
        # segmentation are both indexed in tomogram voxels, so they only share a
        # frame if they came from the SAME tomogram — a segmentation run on a
        # differently binned volume would put every distance silently wrong.
        frames = {
            "real picks": real.parameters.get("input_tomogram_id"),
            "control picks": decoy.parameters.get("input_tomogram_id"),
            "segmentation": segmentation.parameters.get("input_tomogram_id"),
        }
        if len(set(frames.values())) != 1:
            print(f"{series}: these do not share a coordinate frame:")
            for what, tomogram_id in frames.items():
                print(f"  {what:<15} {tomogram_id}")
            return 1

        distance, fraction = _distance_field(segmentation, args.threshold)
        real_values = _sample(distance, _read_pytom_star(real.path))
        decoy_values = _sample(distance, _read_pytom_star(decoy.path))
        real_all.append(real_values)
        decoy_all.append(decoy_values)
        print(
            f"{series}: membrane {fraction:.3%} of voxels, "
            f"{real_values.size} real and {decoy_values.size} control picks, "
            f"median distance {np.median(real_values):.0f} vs "
            f"{np.median(decoy_values):.0f} A"
        )

    real_values = np.concatenate(real_all)
    decoy_values = np.concatenate(decoy_all)
    real_share = _histogram(real_values, edges)
    decoy_share = _histogram(decoy_values, edges)

    print(f"\npooled: {real_values.size} real, {decoy_values.size} control\n")
    print(f"  {'A from membrane':>18} {'real':>8} {'control':>8} {'excess':>8}")
    for index in range(len(edges) - 1):
        low, high = edges[index], edges[index + 1]
        excess = real_share[index] - decoy_share[index]
        bar = "#" * int(round(abs(excess) * 400))
        sign = "" if excess >= 0 else "-"
        print(
            f"  {low:>7.0f} to {high:<7.0f} {real_share[index]:>8.3%} "
            f"{decoy_share[index]:>8.3%} {excess:>+8.3%} {sign}{bar}"
        )

    low, high = args.expected_offset_angstrom
    in_band = (edges[:-1] >= low) & (edges[:-1] < high)
    real_band = float(real_share[in_band].sum())
    decoy_band = float(decoy_share[in_band].sum())
    print(
        f"\nin the {low:.0f} to {high:.0f} A band, where a membrane-bound "
        f"ribosome's centre belongs:\n"
        f"  real {real_band:.2%}, control {decoy_band:.2%}, "
        f"excess {real_band - decoy_band:+.2%}"
    )
    print()
    print("Read the histogram, not that one number. A density preference shifts")
    print("the whole distribution and would show as an excess at every short")
    print("distance; a bound population shows as a BUMP at one offset with the")
    print("far bins unchanged or depleted. Only the second shape is evidence,")
    print("and neither is evidence of a ribosome specifically — any large")
    print("membrane-associated complex would look the same.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
