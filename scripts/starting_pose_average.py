#!/usr/bin/env python3
"""Write the average of a Dynamo project's particles at their starting poses to a file.

    python scripts/starting_pose_average.py outputs/ryr1/dynamo/bin8-2x/mra_bbn \\
        --out outputs/ryr1/dynamo/bin8-2x/tm_pose_average.mrc

The map the classification report draws as the average of all particles
(scripts/dynamo_classification_report.py, the page after the particles), made
the same way: read_dynamo_classification averages the project's starting table
as Dynamo's console dynamo_average would (fcompensate, fmin 1, normalised under
the project's alignment mask), and this writes it as MRC with the project's
pixel size instead of only drawing it. About a minute for 2,700 particles of 48^3.
In a project of scripts/dynamo_mra_setup.m the starting table is the export's
initial.tbl, so these are the picks' template-matching poses: where every new run
on that export begins.

Why (2026-10-20): RyR1's class 1 at iteration 5 of mra_bbn is fourfold about an
axis 4.7 deg off the box z axis (scripts/rotational_symmetry.py --order 4), and
Dynamo imposes c4 about the box z axis. Whether the starting poses carry that
tilt, or five C1 iterations inside a sphere let the whole average turn, is read
off this map; it is also the one to draw the membrane-free mask on
(scripts/make_mra_mask.py --average). Disposable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np

from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="the Dynamo project folder")
    parser.add_argument("--out", type=Path, required=True, help="the average, .mrc")
    parser.add_argument("--manifest", type=Path,
                        help="export_manifest.json; default the one beside the project, if any")
    args = parser.parse_args()

    run = read_dynamo_classification(args.project, manifest=args.manifest)
    if run.initial_average is None:
        raise SystemExit(f"{args.project}: no particles folder or settings/mask.em to "
                         "average the starting poses with")
    print(f"averaging {len(run.tags):,} particles of {run.name} at their starting poses")
    average = np.asarray(run.initial_average.load(), dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(args.out, overwrite=True) as handle:
        handle.set_data(average)
        if run.pixel_size_angstrom:
            handle.voxel_size = run.pixel_size_angstrom
    pixel = f", {run.pixel_size_angstrom:.3f} A/px" if run.pixel_size_angstrom else ""
    print(f"wrote {args.out}: {average.shape[0]} px box{pixel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
