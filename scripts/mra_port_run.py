#!/usr/bin/env python3
"""An MRA run with the pipeline's port: a Dynamo project's run again, beside it and
compared with it, or a new run set up from an export folder without Dynamo.

    python scripts/mra_port_run.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_cone5 --device cuda
    python scripts/mra_port_run.py --export outputs/ryr1/dynamo/bin8-2x-pad \\
        --name mra_pad_ssn_g2atan3_port --seeds "subset subset noise" \\
        --schedule coarse_to_fine --first-lowpass 53.76 --last-lowpass 29.4 \\
        --diameter 340 --global-rounds 2 --device cuda

With --export, the pipeline's setup (`mra_setup.plan_from_export`) makes the plan from
the export folder and scripts/dynamo_mra_setup.m's variables, as that script made a
Dynamo project from them (the same seeds but for new noise in a noise seed, the same
mask and searches; scripts/compare_mra_setup.py checks it against Dynamo's), and the
run is written to <export>/<name>, with nothing to compare. That is how the
pipeline's own runs are made (scripts/mra_port_run.sbatch with EXPORT and NAME). The
rest of this text is about a Dynamo project given instead.

reads the Dynamo project's plan (dynamo_mra_project.plan_from_dynamo: its starting
table, seeds, soft mask, particles, and every iteration's search from that
iteration's card), runs it with the port and writes the run beside the Dynamo
project, as a project of the same layout, named --name (default <project>_port):

- --device cuda (or mps, cpu): the torch kernel, float32 unless --dtype float64,
  with the particles dealt among --gpus devices (all the GPUs the job sees unless
  given), --per-gpu processes on each (1 unless given);
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
followed, and there is nothing to compare. That is how the pipeline's runs were
made until 2026-11-02, and how those runs carry on. A global round runs at the
card's dim, downsampled as Dynamo's CPU path does it (GLOBAL_FULL_BOX=1 at the setup
searches it at the full box instead).

A run already under --name carries on from its last finished iteration, so a job
that ran out of time is submitted again as it was; the run must follow the same
plan (the same seeds, mask, starting table and searches in every iteration it
finished), or it is refused. Each iteration starts from what the one before it
wrote, as Dynamo's do, so the run carried on ends as the unstopped one would have,
bit for bit wherever the kernel repeats itself bit for bit (the numpy one does, and
so did the two cone5 runs on A40s). A run that finished every iteration computes
nothing and is compared again.

The comparison of classes alone, with chance levels, is
scripts/compare_dynamo_classes.py; the report, scripts/dynamo_classification_report.py.
Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.backends.dynamo_search import euler_matrix, search_particles
from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project, torch_search
from cryoet_pipeline.mra_setup import plan_from_export
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


def _spacing(export: Path) -> float | None:
    manifest = export / "export_manifest.json"
    return (json.loads(manifest.read_text()).get("pixel_spacing_angstrom")
            if manifest.exists() else None)


def _kernel(args: argparse.Namespace) -> tuple[Any, str]:
    """The search the arguments ask for, and its name for the run's cards."""

    if args.device == "numpy":
        return search_particles, f"cryoet-pipeline numpy, {args.workers} workers"
    devices = max(1, args.gpus)
    if args.device == "cuda" and args.gpus == 0:
        import torch

        devices = max(1, torch.cuda.device_count())
    search = torch_search(args.device, args.dtype, args.chunk, devices=devices,
                          per_device=args.per_gpu)
    destination = f"cryoet-pipeline torch {args.device} {args.dtype} x{devices}"
    if args.per_gpu > 1:
        destination += f", {args.per_gpu} processes each"
    return search, destination


def new_run(args: argparse.Namespace) -> int:
    """A run set up from an export folder without Dynamo: nothing to compare it with."""

    export = args.export.resolve()
    try:
        plan = plan_from_export(
            export, args.seeds.split(), reference_file=args.reference, schedule=args.schedule,
            first_lowpass=args.first_lowpass, last_lowpass=args.last_lowpass,
            diameter=args.diameter, lowpass_steps=args.lowpass_steps,
            mask_radius=args.mask_radius, mask_edge=args.mask_edge, mask_file=args.mask_file,
            global_rounds=args.global_rounds, global_step=args.global_step,
            global_full_box=args.global_full_box,
            cone_steps=[float(step) for step in args.cone_steps.split()],
            data_folder=args.data_folder,
        )
    except ValueError as error:
        raise SystemExit(f"REFUSED: the setup: {error}") from None
    if args.iterations is not None:
        first = slice(0, args.iterations)
        plan = replace(plan, settings=plan.settings[first], cards=plan.cards[first],
                       mra=plan.mra[first])
    search, destination = _kernel(args)
    run_mra_project(plan, export / args.name, search=search, destination=destination,
                    workers=args.workers, spacing=_spacing(export), resume=True)
    print(f"{args.name} was set up from {export}: no Dynamo project to compare it with")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, nargs="?",
                        help="the Dynamo MRA project to run again (or --export)")
    parser.add_argument("--name", default=None,
                        help="the port's project; default <project>_port, given with --export")
    parser.add_argument("--device", default="cuda",
                        help="cuda, mps or cpu for the torch kernel; numpy for the numpy one")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--chunk", type=int, default=256, help="torch: candidates per batch")
    parser.add_argument("--workers", type=int, default=1, help="numpy: processes")
    parser.add_argument("--gpus", type=int, default=0,
                        help="torch on cuda: how many GPUs to deal the particles among; "
                        "0 (default) for all the job sees")
    parser.add_argument("--per-gpu", type=int, default=1,
                        help="torch: processes sharing each device")
    parser.add_argument("--iterations", type=int, default=None,
                        help="only the first so many of the project's iterations")
    parser.add_argument("--compare-only", action="store_true",
                        help="compare an existing port run with the project, run nothing")
    setup = parser.add_argument_group(
        "a new run, set up from an export folder without Dynamo, with the variables of "
        "scripts/dynamo_mra_setup.m (see its header)")
    setup.add_argument("--export", type=Path, default=None,
                       help="an export folder of scripts/export_dynamo.py: particles/, initial.tbl")
    setup.add_argument("--seeds", default=None,
                       help='one kind per channel, e.g. "subset subset noise"; or "map" with '
                       "--reference")
    setup.add_argument("--reference", type=Path, default=None,
                       help="the map a 'map' seed refines from, in the particles' box")
    setup.add_argument("--schedule", default="fixed", choices=("fixed", "coarse_to_fine"))
    setup.add_argument("--first-lowpass", type=float, default=40.0, help="A (coarse_to_fine)")
    setup.add_argument("--last-lowpass", type=float, default=22.0, help="A (coarse_to_fine)")
    setup.add_argument("--diameter", type=float, default=290.0,
                       help="A, the particle where it is widest: the angular steps and the mask")
    setup.add_argument("--lowpass-steps", default="angstrom", choices=("angstrom", "fourier"))
    setup.add_argument("--mask-radius", type=float, default=None,
                       help="px; default the particle's radius plus 3 px")
    setup.add_argument("--mask-edge", type=float, default=2.0, help="px")
    setup.add_argument("--mask-file", type=Path, default=None,
                       help="a mask of any shape in the particles' box, instead of the sphere")
    setup.add_argument("--global-rounds", type=int, default=0,
                       help="the first rounds that search every orientation")
    setup.add_argument("--global-step", type=float, default=None,
                       help="deg: a coarser grid for the global rounds, refined to their step")
    setup.add_argument("--global-full-box", action="store_true",
                       help="search the global rounds at the full box, not downsampled")
    setup.add_argument("--cone-steps", default="3",
                       help='the cones\' width in steps: one number or one per round, "5 5 3 3 3"')
    setup.add_argument("--data-folder", default="particles")
    args = parser.parse_args()

    if (args.project is None) == (args.export is None):
        parser.error("give a Dynamo project, or --export for a new run; one of them")
    if args.export is not None:
        if args.name is None or args.seeds is None:
            parser.error("--export needs --name and --seeds")
        if args.compare_only:
            parser.error("a new run has no Dynamo project to compare with")
        return new_run(args)

    project = args.project.resolve()
    out = project.parent / (args.name or f"{project.name}_port")
    computed = computed_by_dynamo(project)
    if args.compare_only and not computed:
        raise SystemExit(f"{project}: Dynamo has not computed it, so there is nothing to compare")
    if not args.compare_only:
        plan = plan_from_dynamo(project, iterations=args.iterations)
        search, destination = _kernel(args)
        run_mra_project(plan, out, search=search, destination=destination,
                        workers=args.workers, spacing=_spacing(project.parent), resume=True)
    if not computed:
        print(f"{project.name} was set up and not computed by Dynamo: nothing to compare")
        return 0
    compare(out, project)
    print(f"classes with chance levels: python scripts/compare_dynamo_classes.py {project} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
