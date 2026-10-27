#!/usr/bin/env python3
"""A Dynamo MRA project run again with the pipeline's port, beside it, and compared with it.

    python scripts/mra_port_run.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_cone5 --device cuda

reads the Dynamo project's plan (dynamo_mra_project.plan_from_dynamo: its starting
table, seeds, soft mask, particles, and every iteration's search from that
iteration's card), runs it with the port and writes the run beside the Dynamo
project, as a project of the same layout, named --name (default <project>_port):

- --device cuda (or mps, cpu): the torch kernel, float32 unless --dtype float64,
  with the particles dealt among --gpus devices, a process each (all the GPUs the
  job sees unless given);
- --device numpy: the numpy kernel on --workers processes, the one the replays of
  Dynamo's CPU path checked.

Then it compares the two runs iteration by iteration: how many particles each puts
in the same class, and for those, how far apart their poses are (rotation in
degrees, shift in px) and their CCs. Two exact runs agree everywhere; float32
against Dynamo's float64 CPU parts with the particles whose best candidates tie
within rounding, and from them, through the class averages, more and more. The
same comparison between Dynamo's own GPU and CPU runs of one project is the
yardstick (cone5 against cone5_gpu).

A project the setup made and Dynamo never computed (scripts/dynamo_mra_mdc.sbatch
with SETUP_ONLY=1) is run the same way: Dynamo writes every iteration's card when
the setup unfolds a project, so the port follows the plan Dynamo would have
followed, and there is nothing to compare. That is how the pipeline's own runs are
made. A global round must then be set up at the full box (GLOBAL_FULL_BOX=1): the
port searches only there, and refuses Dynamo's CPU downsampling (dim).

The comparison of classes alone, with chance levels, is
scripts/compare_dynamo_classes.py; the report, scripts/dynamo_classification_report.py.
Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from cryoet_pipeline.backends.dynamo_search import euler_matrix, search_particles
from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project, torch_search
from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification, read_table


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = euler_matrix(first) @ euler_matrix(second).T
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1) / 2))))


def _classes(project: Path) -> dict[int, dict[int, int]]:
    """Each finished iteration's class of every particle."""

    run = read_dynamo_classification(project, compute_initial_average=False)
    return {iteration.number: {int(tag): channel for channel, result in iteration.classes.items()
                               for tag in result.members}
            for iteration in run.iterations}


def computed_by_dynamo(project: Path) -> bool:
    """Has Dynamo finished an iteration of the project, or was it only set up?"""

    return any(project.glob("results/ite_*/averages/refined_table_ref_*_ite_*.tbl"))


def compare(ours: Path, theirs: Path) -> None:
    """Iteration by iteration: same class, and for those, the poses and CCs."""

    mine, dynamo = _classes(ours), _classes(theirs)
    print(f"{ours.name} against {theirs.name}:")
    for number in sorted(set(mine) & set(dynamo)):
        tags = sorted(set(mine[number]) & set(dynamo[number]))
        same = [tag for tag in tags if mine[number][tag] == dynamo[number][tag]]
        angles, shifts, ccs = [], [], []
        tables: dict[tuple[str, int], dict[int, np.ndarray]] = {}
        for tag in same:
            channel = mine[number][tag]
            rows = []
            for label, project in (("ours", ours), ("dynamo", theirs)):
                key = (label, channel)
                if key not in tables:
                    path = (project / "results" / f"ite_{number:04d}" / "averages"
                            / f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl")
                    tables[key] = {int(row[0]): row for row in read_table(path)}
                rows.append(tables[key][tag])
            angles.append(_angle(rows[0][6:9], rows[1][6:9]))
            shifts.append(float(np.linalg.norm(rows[0][3:6] - rows[1][3:6])))
            ccs.append(float(rows[0][9] - rows[1][9]))
        if not same:
            print(f"  iteration {number}: no particle in the same class")
            continue
        angle, shift, cc = (np.array(values) for values in (angles, shifts, ccs))
        print(f"  iteration {number}: same class {len(same):,} of {len(tags):,} "
              f"({100 * len(same) / len(tags):.1f}%); their poses: rotation median "
              f"{np.median(angle):.3f}, 95% {np.percentile(angle, 95):.3f}, max "
              f"{angle.max():.2f} deg; shift median {np.median(shift):.3f}, max "
              f"{shift.max():.2f} px; CC difference median {np.median(np.abs(cc)):.1e}, max "
              f"{np.abs(cc).max():.1e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="the Dynamo MRA project to run again")
    parser.add_argument("--name", default=None, help="the port's project; default <project>_port")
    parser.add_argument("--device", default="cuda",
                        help="cuda, mps or cpu for the torch kernel; numpy for the numpy one")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--chunk", type=int, default=256, help="torch: candidates per batch")
    parser.add_argument("--workers", type=int, default=1, help="numpy: processes")
    parser.add_argument("--gpus", type=int, default=0,
                        help="torch on cuda: how many GPUs to deal the particles among; "
                        "0 (default) for all the job sees")
    parser.add_argument("--iterations", type=int, default=None,
                        help="only the first so many of the project's iterations")
    parser.add_argument("--compare-only", action="store_true",
                        help="compare an existing port run with the project, run nothing")
    args = parser.parse_args()

    project = args.project.resolve()
    out = project.parent / (args.name or f"{project.name}_port")
    computed = computed_by_dynamo(project)
    if args.compare_only and not computed:
        raise SystemExit(f"{project}: Dynamo has not computed it, so there is nothing to compare")
    if not args.compare_only:
        plan = plan_from_dynamo(project, iterations=args.iterations)
        manifest = project.parent / "export_manifest.json"
        spacing = (json.loads(manifest.read_text()).get("pixel_spacing_angstrom")
                   if manifest.exists() else None)
        if args.device == "numpy":
            search, destination = search_particles, f"cryoet-pipeline numpy, {args.workers} workers"
        else:
            devices = max(1, args.gpus)
            if args.device == "cuda" and args.gpus == 0:
                import torch

                devices = max(1, torch.cuda.device_count())
            search = torch_search(args.device, args.dtype, args.chunk, devices=devices)
            destination = f"cryoet-pipeline torch {args.device} {args.dtype} x{devices}"
        run_mra_project(plan, out, search=search, destination=destination,
                        workers=args.workers, spacing=spacing)
    if not computed:
        print(f"{project.name} was set up and not computed by Dynamo: nothing to compare")
        return 0
    compare(out, project)
    print(f"classes with chance levels: python scripts/compare_dynamo_classes.py {project} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
