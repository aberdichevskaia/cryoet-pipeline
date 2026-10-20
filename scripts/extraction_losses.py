#!/usr/bin/env python3
"""Where the picks that extraction dropped were, series by series.

    python3 scripts/extraction_losses.py --project outputs/ryr1 --variant bin8-2x \\
        --box 52 --series 039 --series 096 --series 097 --series 125 --series 161

extract-particles keeps a pick only if its whole box lies inside the tomogram
(`_crop_start` in backends/subtomogram.py) and records how many it dropped, but
not on which side they fell out. This repeats the test for every pick of the
labelled pick list against its own tomogram's shape and says, per series, how
many picks fall out through the top or bottom in z, and how many through the
sides in y or x (a pick can fall out of more than one); then where in z the picks
sit against the tomogram's thickness, and how many a smaller box would keep.
If extraction recorded its own count, that is printed beside it, so the two can
be checked against each other.

Reads the registry and the files it names; writes nothing. Disposable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.cli import _particle_stack_artifact, _picks_artifact, _tomogram_from_picks
from cryoet_pipeline.models import TiltSeriesManifest
from cryoet_pipeline.particles import read_star_loop


def _outside(position: float, half: int, box: int, size: int) -> tuple[bool, bool]:
    """Whether a box about `position` falls out below 0, and above `size`, as extraction tests."""

    start = int(round(position)) - half
    return start < 0, start + box > size


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--variant", required=True, help="the label of the picks extracted")
    parser.add_argument("--box", type=int, required=True, help="the extraction box, px")
    parser.add_argument("--series", action="append", required=True)
    parser.add_argument(
        "--smaller", type=int, action="append", default=[],
        help="also count what this box would keep (repeat for several)",
    )
    args = parser.parse_args()

    registry = ArtifactRegistry.load(args.project / "artifacts.json")
    smaller = sorted(set(args.smaller) | {args.box * 3 // 4 // 2 * 2, args.box // 2 // 2 * 2})
    totals = {"picks": 0, "kept": 0}
    for series in args.series:
        manifest = TiltSeriesManifest.model_validate_json(
            (args.project / "manifests" / f"{series}.json").read_text()
        )
        picks = _picks_artifact(registry, manifest, variant=args.variant)
        tomogram = _tomogram_from_picks(registry, picks, manifest, variant=args.variant)
        shape = tuple(int(size) for size in cast(Any, zarr.open(tomogram.path, mode="r")).shape)
        columns, rows = read_star_loop(picks.path)
        coordinates = [
            tuple(float(row[columns.index(f"rlnCoordinate{axis}")]) for axis in "ZYX")
            for row in rows
        ]

        def fate(box: int) -> dict[str, int]:
            half = box // 2
            counts = {"kept": 0, "z low": 0, "z high": 0, "y": 0, "x": 0}
            for z, y, x in coordinates:
                z_low, z_high = _outside(z, half, box, shape[0])
                y_out = any(_outside(y, half, box, shape[1]))
                x_out = any(_outside(x, half, box, shape[2]))
                counts["z low"] += z_low
                counts["z high"] += z_high
                counts["y"] += y_out
                counts["x"] += x_out
                counts["kept"] += not (z_low or z_high or y_out or x_out)
            return counts

        here = fate(args.box)
        recorded = None
        try:
            stack = _particle_stack_artifact(registry, manifest, variant=args.variant)
            qc_path = stack.parameters.get("qc_path")
            if isinstance(qc_path, str) and Path(qc_path).is_file():
                recorded = json.loads(Path(qc_path).read_text())
        except Exception:  # no extraction under this label: nothing to compare with
            recorded = None
        z_values = sorted(z for z, _, _ in coordinates)
        print(f"{series}: tomogram z {shape[0]} y {shape[1]} x {shape[2]} px, "
              f"{len(coordinates)} picks, box {args.box}")
        print(f"  kept {here['kept']}; out through z: {here['z low']} below, "
              f"{here['z high']} above; through y {here['y']}, through x {here['x']}")
        if recorded is not None:
            print(f"  extraction recorded: {recorded.get('extracted_particle_count')} kept, "
                  f"{recorded.get('dropped_edge_count')} at the edge, "
                  f"{recorded.get('dropped_score_count')} below a score, "
                  f"box {recorded.get('box_size_px')}")
        if z_values:
            quartiles = [z_values[int(q * (len(z_values) - 1))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
            print("  picks in z (min, quartiles, max): "
                  + ", ".join(f"{value:.0f}" for value in quartiles)
                  + f" of 0..{shape[0] - 1}; a box of {args.box} needs "
                  f"{args.box // 2}..{shape[0] - args.box + args.box // 2}")
        for box in smaller:
            print(f"  a box of {box} would keep {fate(box)['kept']}")
        totals["picks"] += len(coordinates)
        totals["kept"] += here["kept"]
    print(f"all: {totals['kept']} of {totals['picks']} picks fit a box of {args.box}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
