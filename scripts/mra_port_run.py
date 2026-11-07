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
run is written to <export>/<name>, with nothing to compare. With --source, --lane and
--series instead, the plan is made from the particle stacks of those series in the
pipeline project's registry, read where they are (`mra_setup.plan_from_registry`,
docs/mra_from_registry.md: the particles an export of the same arguments would copy),
and the run is written to <source>/mra/<name>. With --split and --half, a consensus
half of a split scripts/consensus_halves.py wrote of two runs of the registry is
refined from the split's start (`mra_setup.plan_from_split`, docs/halves_from_runs.md),
the run written beside the split. That is how the pipeline's own runs are made
(scripts/mra_port_run.sbatch with SOURCE, EXPORT or SPLIT, and NAME). The rest of
this text is about a Dynamo project given instead.

A run is written in the run-record layout (docs/run_records.md, `mra_runs.run_plan`):
plan.json, then setup/ and ite_0001/, ite_0002/, ... each with its run.json, the
particles' poses as Parquet and the maps as MRC. A run begun in Dynamo's layout before
2026-11-05 (settings/, results/, log.txt) is carried on in it to its end, by the writer
that began it (`dynamo_mra_project.run_mra_project`).

reads the Dynamo project's plan (dynamo_mra_project.plan_from_dynamo: its starting
table, seeds, soft mask, particles, and every iteration's search from that
iteration's card), runs it with the port and writes the run beside the Dynamo
project, named --name (default <project>_port):

- --device cuda (or mps, cpu): the torch kernel, float32 unless --dtype float64,
  with the particles dealt among --gpus devices (all the GPUs the job sees unless
  given), --per-gpu processes on each (1 unless given);
- --device numpy: the numpy kernel on --workers processes, the one the replays of
  Dynamo's CPU path checked.

Whatever the device, the seeds and the class averages are made on --workers threads;
they do not depend on it (`dynamo_search.average_class`).

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
nothing and is compared again. Both layouts are read through
`reports.classification_reader`, so either can be compared with a Dynamo project.

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
from cryoet_pipeline.dynamo_mra_project import (
    MraPlan,
    plan_from_dynamo,
    run_mra_project,
    torch_search,
)
from cryoet_pipeline.mra_runs import Refused, run_plan
from cryoet_pipeline.mra_setup import (
    SHIFT_LIMIT,
    plan_from_export,
    plan_from_registry,
    plan_from_split,
)
from cryoet_pipeline.reports.classification import ClassificationRun, PoseTable
from cryoet_pipeline.reports.classification_reader import read_classification


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = euler_matrix(first) @ euler_matrix(second).T
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1) / 2))))


def _classes(run: ClassificationRun) -> dict[int, dict[int, int]]:
    """Each finished iteration's class of every particle."""

    return {iteration.number: {int(tag): channel for channel, result in iteration.classes.items()
                               for tag in result.members}
            for iteration in run.iterations}


def computed_by_dynamo(project: Path) -> bool:
    """Has Dynamo finished an iteration of the project, or was it only set up?"""

    return any(project.glob("results/ite_*/averages/refined_table_ref_*_ite_*.tbl"))


def compare(ours: Path, theirs: Path) -> None:
    """Iteration by iteration: same class, and for those, the poses and CCs."""

    runs = [read_classification(path, compute_initial_average=False) for path in (ours, theirs)]
    mine, dynamo = (_classes(run) for run in runs)
    print(f"{ours.name} against {theirs.name}:")
    for number in sorted(set(mine) & set(dynamo)):
        tags = sorted(set(mine[number]) & set(dynamo[number]))
        same = [tag for tag in tags if mine[number][tag] == dynamo[number][tag]]
        angles, shifts, ccs = [], [], []
        tables: dict[tuple[int, int], tuple[PoseTable, dict[int, int]]] = {}
        for tag in same:
            channel = mine[number][tag]
            rows = []
            for side, run in enumerate(runs):
                if (side, channel) not in tables:
                    assert run.poses is not None
                    table = run.poses((number, channel))
                    tables[(side, channel)] = (table, {int(one): row
                                                       for row, one in enumerate(table.tags)})
                table, index = tables[(side, channel)]
                rows.append((table, index[tag]))
            (first, i), (second, j) = rows
            angles.append(_angle(first.angles[i], second.angles[j]))
            shifts.append(float(np.linalg.norm(first.shifts[i] - second.shifts[j])))
            ccs.append(float(first.cc[i] - second.cc[j]))
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


def _kernel(args: argparse.Namespace) -> tuple[Any, str, dict[str, Any]]:
    """The search the arguments ask for, its name for the cards of a run in Dynamo's
    layout, and the execution parameters a run record keeps."""

    if args.device == "numpy":
        return (search_particles, f"cryoet-pipeline numpy, {args.workers} workers",
                {"kernel": "numpy", "workers": args.workers})
    devices = max(1, args.gpus)
    if args.device == "cuda" and args.gpus == 0:
        import torch

        devices = max(1, torch.cuda.device_count())
    search = torch_search(args.device, args.dtype, args.chunk, devices=devices,
                          per_device=args.per_gpu)
    destination = f"cryoet-pipeline torch {args.device} {args.dtype} x{devices}"
    if args.per_gpu > 1:
        destination += f", {args.per_gpu} processes each"
    return search, destination, {"kernel": "torch", "device": args.device, "dtype": args.dtype,
                                 "devices": devices, "per_device": args.per_gpu,
                                 "chunk": args.chunk}


def dynamo_layout(run: Path) -> bool:
    """A run of the port begun in Dynamo's layout, before 2026-11-05, which is carried on
    in it: the port's writer of that layout marked each of its runs with
    settings/mirrors.json, which Dynamo's own projects do not have. Any other folder
    with files in it, a Dynamo project included, is refused by `run_plan`."""

    return (run / "settings" / "mirrors.json").is_file()


def compute(plan: MraPlan, run: Path, args: argparse.Namespace, spacing: float | None) -> None:
    """The plan into `run`, from where it stopped: in the run-record layout, or in Dynamo's
    for a run begun in it."""

    search, destination, execution = _kernel(args)
    if dynamo_layout(run):
        print(f"{run.name} was begun in Dynamo's layout (before 2026-11-05) and is carried on "
              "in it")
        run_mra_project(plan, run, search=search, destination=destination,
                        workers=args.workers, spacing=spacing, resume=True)
        return
    try:
        run_plan(plan, run, search=search, execution=execution, workers=args.workers)
    except Refused as error:
        raise SystemExit(f"REFUSED: {error}") from None


def new_run(args: argparse.Namespace) -> int:
    """A run set up without Dynamo, from an export folder or from a project's registry:
    nothing to compare it with."""

    options: dict[str, Any] = dict(
        schedule=args.schedule,
        first_lowpass=args.first_lowpass, last_lowpass=args.last_lowpass,
        diameter=args.diameter, lowpass_steps=args.lowpass_steps,
        mask_radius=args.mask_radius, mask_edge=args.mask_edge, mask_file=args.mask_file,
        global_rounds=args.global_rounds, global_step=args.global_step,
        global_full_box=args.global_full_box,
        cone_steps=[float(step) for step in args.cone_steps.split()],
        shift_limit=args.shift_limit, antialias=args.antialias, workers=args.workers,
    )
    try:
        if args.export is not None:
            where = args.export.resolve()
            plan = plan_from_export(where, args.seeds.split(), data_folder=args.data_folder,
                                    reference_file=args.reference, **options)
            run, spacing = where / args.name, _spacing(where)
        elif args.source is not None:
            where = args.source.resolve()
            plan = plan_from_registry(where, args.lane, args.series, args.seeds.split(),
                                      decoy_label=args.decoy_lane,
                                      reference_file=args.reference, **options)
            run, spacing = where / "mra" / args.name, None
        else:
            where = args.split.resolve()
            plan = plan_from_split(where, args.half, **options)
            run, spacing = where.parent / args.name, None
    except ValueError as error:
        raise SystemExit(f"REFUSED: the setup: {error}") from None
    if args.iterations is not None:
        first = slice(0, args.iterations)
        plan = replace(plan, settings=plan.settings[first], cards=plan.cards[first],
                       mra=plan.mra[first])
    compute(plan, run, args, spacing)
    print(f"{args.name} was set up from {where}: no Dynamo project to compare it with")
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
    parser.add_argument("--workers", type=int, default=1,
                        help="threads averaging the seeds and classes; with --device numpy, "
                        "also the processes searching")
    parser.add_argument("--gpus", type=int, default=0,
                        help="torch on cuda: how many GPUs to deal the particles among; "
                        "0 (default) for all the job sees")
    parser.add_argument("--per-gpu", type=int, default=1,
                        help="torch: processes sharing each device")
    parser.add_argument("--iterations", type=int, default=None,
                        help="only the first so many of the project's iterations")
    parser.add_argument("--compare-only", action="store_true",
                        help="compare an existing port run with the project, run nothing")
    registry = parser.add_argument_group(
        "a new run from a pipeline project's registry: the particle stacks of --series, read "
        "where they are; the setup's variables below")
    registry.add_argument("--source", type=Path, default=None,
                          help="the pipeline project (its artifacts.json and manifests/)")
    registry.add_argument("--lane", default=None,
                          help="the label of the particle stacks; leave out for unlabelled ones")
    registry.add_argument("--series", nargs="+", default=None,
                          help="the tilt-series, in the order to number them")
    registry.add_argument("--decoy-lane", default=None,
                          help="the label of decoy stacks of the same series, after the particles")
    halves = parser.add_argument_group(
        "a consensus half: one half of a split of two runs of the registry, refined from "
        "the split's start; the schedule's variables below")
    halves.add_argument("--split", type=Path, default=None,
                        help="the split, <project>/mra/<name> of consensus_halves.py split")
    halves.add_argument("--half", type=int, choices=(1, 2), default=None)
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
    setup.add_argument("--antialias", action="store_true",
                       help="not Dynamo's: low-pass every map to the global rounds' smaller box "
                       "before shrinking it, so finer noise does not fold into their band")
    setup.add_argument("--cone-steps", default="3",
                       help='the cones\' width in steps: one number or one per round, "5 5 3 3 3"')
    setup.add_argument("--shift-limit", type=float, default=SHIFT_LIMIT,
                       help="px of the particles' box: how far a particle may move from its "
                       f"box centre in every round (default {SHIFT_LIMIT:g})")
    setup.add_argument("--data-folder", default="particles")
    args = parser.parse_args()

    given = [args.project is not None, args.export is not None, args.source is not None,
             args.split is not None]
    if sum(given) != 1:
        parser.error("give a Dynamo project, or --export, --source or --split for a new run; "
                     "one of them")
    if args.source is None and (args.series or args.lane or args.decoy_lane):
        parser.error("--series, --lane and --decoy-lane go with --source")
    if (args.split is None) != (args.half is None):
        parser.error("--split and --half go together")
    if args.split is not None:
        if args.name is None:
            parser.error("a half needs --name")
        if args.seeds is not None or args.reference is not None:
            parser.error("a half's seed is its split's start_reference.mrc; leave out --seeds "
                         "and --reference")
        if args.data_folder != "particles":
            parser.error("--data-folder is an export's; a half reads the stacks")
        if args.compare_only:
            parser.error("a new run has no Dynamo project to compare with")
        return new_run(args)
    if args.export is not None or args.source is not None:
        if args.name is None or args.seeds is None:
            parser.error("a new run needs --name and --seeds")
        if args.source is not None and not args.series:
            parser.error("--source needs --series")
        if args.source is not None and args.data_folder != "particles":
            parser.error("--data-folder is an export's; a run from the registry reads the stacks")
        if args.compare_only:
            parser.error("a new run has no Dynamo project to compare with")
        return new_run(args)

    if args.antialias:
        parser.error("--antialias sets up a new run; a Dynamo project searches as Dynamo set "
                     "it up")
    project = args.project.resolve()
    out = project.parent / (args.name or f"{project.name}_port")
    computed = computed_by_dynamo(project)
    if args.compare_only and not computed:
        raise SystemExit(f"{project}: Dynamo has not computed it, so there is nothing to compare")
    if not args.compare_only:
        compute(plan_from_dynamo(project, iterations=args.iterations), out, args,
                _spacing(project.parent))
    if not computed:
        print(f"{project.name} was set up and not computed by Dynamo: nothing to compare")
        return 0
    compare(out, project)
    print(f"classes with chance levels: python scripts/compare_dynamo_classes.py {project} {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
