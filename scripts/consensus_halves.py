#!/usr/bin/env python3
"""Two halves of the particles two MRA runs agree on, refined apart from one start.

    python scripts/consensus_halves.py split \\
        outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan:1 \\
        --agree outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssb_g2atan:1 \\
        --start-lowpass 53.76 --out outputs/ryr1/dynamo/ryr1-consensus
    EXPORT=outputs/ryr1/dynamo/ryr1-consensus-half1 NAME=refine_h1 SEEDS=map \\
        REFERENCE=outputs/ryr1/dynamo/ryr1-consensus-half1/start_reference.mrc \\
        SCHEDULE=coarse_to_fine FIRST_LOWPASS=53.76 LAST_LOWPASS=29.4 DIAMETER=340 \\
        sbatch --export=ALL --job-name=refine_h1 --gres=gpu:a40:1 scripts/mra_port_run.sbatch
    (and the same for half 2, NAME=refine_h2)
    python scripts/consensus_halves.py maps \\
        outputs/ryr1/dynamo/ryr1-consensus-half1/refine_h1 \\
        outputs/ryr1/dynamo/ryr1-consensus-half2/refine_h2 \\
        --out outputs/ryr1/dynamo/ryr1-consensus

Runs of a project's registry (mra_port_run.sbatch with SOURCE, in <project>/mra) are
split with no export (docs/halves_from_runs.md):

    python scripts/consensus_halves.py split outputs/ryr1/mra/mra_ssn:1 \
        --agree outputs/ryr1/mra/mra_ssb:1 --start-lowpass 53.76 \
        --out outputs/ryr1/mra/ryr1-consensus
    SPLIT=outputs/ryr1/mra/ryr1-consensus HALF=1 NAME=refine_h1 SCHEDULE=coarse_to_fine \
        FIRST_LOWPASS=53.76 LAST_LOWPASS=29.4 DIAMETER=340 \
        sbatch --export=ALL --job-name=refine_h1 --gres=gpu:a40:1 scripts/mra_port_run.sbatch
    (and the same for half 2, HALF=2 NAME=refine_h2)
    python scripts/consensus_halves.py maps outputs/ryr1/mra/refine_h1 \
        outputs/ryr1/mra/refine_h2 --out outputs/ryr1/mra/ryr1-consensus

With --start-poses picked (runs of a registry only), each kept particle starts from the
pose its run started it from, its pick's (pytom's for RyR1), and the common start is the
kept particles' average at those poses, low-passed to --start-lowpass, instead of the
first run's refined poses and its class average: no orientation the MRA refined against
all the particles together reaches either half (the gold-standard split, 2026-11-09).
That average is computed here, with --workers threads (on a Mac, 2026-11-09: 0.14 s a
particle in a 102 px box on one thread, 0.06 s on four, so 2.4 and 1.1 min for 1,052):

    python scripts/consensus_halves.py split outputs/ryr1/mra/mra_bin4_ssn:2 \
        --agree outputs/ryr1/mra/mra_bin4_ssb:2 --start-lowpass 53.76 \
        --start-poses picked --out outputs/ryr1/mra/ryr1-picked-bin4

The split is then a run of its own in <project>/mra (its run.json, halves.parquet with
each kept particle's half and starting pose, start_reference.mrc), its halves two
selections of its record; each half run reads the stacks the two runs read, its half's
particles at the split's poses, from the split's start; `maps` knows the two halves by
the split they name. Both runs must have read the same particles. The rest of this
text says how the halves are chosen, dealt and started, which is the same for both,
and what an export's split writes.

`split` takes the members of one class, PROJECT:CHANNEL, after the last iteration its
run finished, that are also members of the --agree class, from another run on the
same export; decoys are left out. Their poses are the first run's: each one's shifts
and angles (columns 4-9) in that channel's refined table of that iteration. Every
other column is the export's initial.tbl, so nothing the MRA wrote into its tables
(class, CC, ...) comes along. The particles are dealt into two halves at random,
series by series (--seed), so both halves hold every series in the same proportion.
Each half is an export of its own, <out>-half1 and <out>-half2, written as
scripts/split_export_by_view.py writes its groups (links to the export's particle
files, initial.tbl, export_manifest.json with the split recorded under
"consensus_half", export_complete.json), so the MRA jobs and the reports take it as
any export. Both get the same start_reference.mrc: the first run's class average of
that iteration, low-passed to --start-lowpass (A) with a raised-cosine edge
(cryoet_pipeline.filters.low_pass_filter_volume). Each half is then refined on its
own from that start, one channel and no MRA, by the pipeline's port without Dynamo:
scripts/mra_port_run.sbatch with EXPORT=<half>, SEEDS=map and
REFERENCE=<half>/start_reference.mrc, the run <half>/<NAME> (<half>/<NAME>_port for
runs begun before 2026-11-05, in Dynamo's layout). Until 2026-11-02
Dynamo refined them (scripts/dynamo_mra_mdc.sbatch with the same variables, which
it still can as a control), as refine_h1 and refine_h2 of RyR1 at bin8 were. The
port ran those two again on max (PROJECT=, 2026-11-02), and `maps` read the same:
FSC unmasked through 0.5 at 54.4 A and 0.143 at 39.1 A in both, masked at 36.8 and
32.9 A against Dynamo's 36.9 and 33.0, the particles turned from the start by a
median 1.7 and 1.6 deg and shifted by 0.12 and 0.11 px in both.

`maps` reads the last iteration the two half runs finished (the port's runs in either
layout, or Dynamo's projects, all through reports.classification_reader) and writes,
as MRC at
the export's pixel size: <out>_half1.mrc and <out>_half2.mrc, their mean
<out>_mean.mrc to look at, and <out>_mask.mrc, the halves' alignment mask. It prints
how far each half moved its particles from their starting poses and where the FSC
between the halves falls through 0.5 and 0.143, masked and unmasked, read as
scripts/fsc_phase_randomization.py reads it (interpolated in frequency), with any
rise toward Nyquist (cryoet_pipeline.fsc.rise_toward_nyquist; Dynamo's averaging
makes one). It writes the curves to <out>_fsc.json and prints the command for that
script's phase-randomisation control of the masked curve.

What the FSC can say (2026-10-26, RyR1 at bin8). The halves share no map finer than
the start's low-pass, but they share their particles' starting poses, which the MRA
refined against all the particles together down to its last low-pass (29.4 A in the
RyR1 runs). Particles that keep those poses carry that joint fit into both halves and
can raise their FSC down to that resolution. At bin8, where Nyquist is 26.9 A, the
FSC can therefore show that the halves disagree, not how far they agree; the
resolution test is the same split at bin4. How far the halves moved their particles
says how much of the shared start they kept.

With --start-poses picked the halves share no pose the MRA refined. They still share
the choice of particles (the two runs' classes, which saw all of them), the start below
its low-pass, and the picks' poses, which template matching found against the template
low-passed for matching (17.5 A for RyR1 at bin4). A pick that fitted noise to the
template carries the template's features into both halves, so the FSC can be raised by
the template at resolutions coarser than that low-pass, and not at finer ones.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import torch
from fsc_phase_randomization import crossing_text, falls
from split_export_by_view import write_subexport

from cryoet_pipeline.backends.dynamo_search import _half_map_fsc, euler_matrix
from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.models import ArtifactKind
from cryoet_pipeline.mra_runs import HALVES, PLAN, START, STEP_SPLIT, node_folder
from cryoet_pipeline.particle_tables import DYNAMO_POSES, write_particle_table
from cryoet_pipeline.reports.classification import ClassificationRun, Pose, PoseTable
from cryoet_pipeline.reports.classification_reader import read_classification, starting_average
from cryoet_pipeline.runs import RunInput, Selection, completed, writing

SPLIT_VERSION = 1
"""Declared: raised by hand when a change moves what a split of two runs writes."""

POSE = slice(3, 9)
LEVELS = (0.5, 0.143)


def _class_spec(text: str) -> tuple[Path, int]:
    folder, colon, channel = text.rpartition(":")
    if not colon or not folder or not channel.isdigit():
        raise argparse.ArgumentTypeError(
            f"{text!r}: give PROJECT:CHANNEL, e.g. <export>/mra_pad_ssn_g2atan:1")
    return Path(folder).resolve(), int(channel)


def _last_classes(project: Path) -> tuple[ClassificationRun, int]:
    run = read_classification(project, compute_initial_average=False)
    if not run.iterations:
        raise SystemExit(f"{project}: no finished iteration")
    for problem in run.problems:
        print(f"note, {project.name}: {problem}")
    return run, run.iterations[-1].number


def _members(project: Path, channel: int) -> tuple[ClassificationRun, int, set[int]]:
    """The class's members after the run's last finished iteration, decoys left out."""

    run, number = _last_classes(project)
    classes = run.iterations[-1].classes
    if channel not in classes:
        raise SystemExit(f"{project}: iteration {number} has no class {channel}; it has "
                         f"{sorted(classes)}")
    members = {int(tag) for tag in classes[channel].members} - set(run.decoys)
    print(f"{project.name} class {channel} after iteration {number}: {len(members):,} particles"
          + (" (decoys left out)" if run.decoys else ""))
    return run, number, members


def _poses(run: ClassificationRun, pose: Pose) -> dict[int, np.ndarray]:
    """Each tag's shifts and angles at `pose`, as columns 4-9 of a Dynamo table hold them."""

    assert run.poses is not None
    table: PoseTable = run.poses(pose)
    rows = np.column_stack([table.shifts, table.angles])
    return {int(tag): row for tag, row in zip(table.tags, rows, strict=True)}


def _write_mrc(path: Path, volume: np.ndarray, spacing: float) -> None:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = spacing


def _from_registry(run: Path) -> bool:
    """A run of the run-record layout that read its particles from a registry."""

    path = run / PLAN
    return path.is_file() and isinstance(json.loads(path.read_text()).get("particles"), dict)


def _deal(chosen: list[int], series: Mapping[int, str], seed: int) -> dict[int, list[int]]:
    """The particles dealt into two halves at random, series by series.

    Dealt alternately after a shuffle within each series. A series with an odd count
    gives its extra particle to the half it started with, so the next one starts with
    the other half.
    """

    rng = np.random.default_rng(seed)
    by_series: dict[str, list[int]] = defaultdict(list)
    for tag in chosen:
        by_series[series[tag]].append(tag)
    halves: dict[int, list[int]] = {1: [], 2: []}
    first = 1
    for name in sorted(by_series):
        for index, tag in enumerate(rng.permutation(by_series[name])):
            halves[first if index % 2 == 0 else 3 - first].append(int(tag))
        if len(by_series[name]) % 2:
            first = 3 - first
    return halves


def _start(run: ClassificationRun, project: Path, channel: int, number: int, lowpass: float,
           spacing: float) -> tuple[np.ndarray, str]:
    """The halves' common start: the class average low-passed, and what it is in words."""

    found = run.iterations[-1].classes[channel].average
    if found is None:
        raise SystemExit(f"{project}: iteration {number} wrote no average of class {channel}")
    start = low_pass_filter_volume(torch.from_numpy(found.load()), cutoff_angstrom=lowpass,
                                   pixel_spacing_angstrom=spacing).numpy()
    return start, (f"{project.name}'s class {channel} average of iteration {number}, "
                   f"low-passed to {lowpass:g} A")


def _picked_start(project: Path, tags: list[int], lowpass: float, spacing: float,
                  workers: int) -> tuple[np.ndarray, str]:
    """The halves' common start from the picks: the kept particles averaged at the poses
    the run started them from, low-passed, and what it is in words."""

    found = starting_average(project, tags, workers=workers)
    start = low_pass_filter_volume(torch.from_numpy(found), cutoff_angstrom=lowpass,
                                   pixel_spacing_angstrom=spacing).numpy()
    return start, (f"the {len(tags):,} kept particles averaged at the poses {project.name} "
                   f"started them from (their picks'), low-passed to {lowpass:g} A")


def split(args: argparse.Namespace) -> int:
    (project, channel), (agree, agree_channel) = args.poses, args.agree
    if args.workers < 1:
        raise SystemExit(f"--workers {args.workers}: give at least 1")
    for path in (project, agree):
        if not path.is_dir():
            raise SystemExit(f"{path}: no such run")
    registry = [_from_registry(path) for path in (project, agree)]
    if any(registry):
        if not all(registry):
            raise SystemExit("a run from an export and a run from the registry are not split "
                             f"together: {project} and {agree}")
        return split_runs(args)
    export = project.parent
    if agree.parent != export:
        raise SystemExit(f"{agree} is not in {export}: both runs must be on the export split")
    if not (export / "export_complete.json").is_file():
        raise SystemExit(f"{export}: no export_complete.json, so no finished export to split; "
                         f"{project.name} is not a run of a registry either (its plan.json would "
                         "name the stacks it read)")
    if args.start_poses == "picked":
        raise SystemExit("--start-poses picked is built for runs of a registry "
                         f"(docs/halves_from_runs.md); {project.name} and {agree.name} are runs "
                         f"on the export {export}")
    out = args.out.resolve()
    folders = {half: out.with_name(f"{out.name}-half{half}") for half in (1, 2)}
    for folder in folders.values():
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(f"{folder} exists and is not empty; split into new folders")

    run, number, members = _members(project, channel)
    _, agree_number, agreed = _members(agree, agree_channel)
    chosen = sorted(members & agreed)
    print(f"in both: {len(chosen):,}")
    if len(chosen) < 2:
        raise SystemExit("fewer than two particles to split")
    spacing = run.pixel_size_angstrom
    if not spacing:
        raise SystemExit(f"{project}: no pixel size (export manifest or the project's apix)")
    if not 2 * spacing < args.start_lowpass:
        raise SystemExit(f"--start-lowpass {args.start_lowpass} A is not coarser than Nyquist "
                         f"({2 * spacing:.2f} A)")

    refined = _poses(run, (number, channel))
    lines = {int(float(line.split()[0])): line
             for line in (export / "initial.tbl").read_text().splitlines() if line.strip()}
    manifest = json.loads((export / "export_manifest.json").read_text())
    series = {int(entry["tag"]): str(entry["run"]) for entry in manifest["particles"]}
    missing = [tag for tag in chosen if tag not in refined or tag not in lines or tag not in series]
    if missing:
        raise SystemExit(f"{len(missing)} particles are missing from the refined table, the "
                         f"export's initial.tbl or its manifest, e.g. tag {missing[0]}")

    def posed(tag: int) -> str:
        tokens = lines[tag].split()
        tokens[POSE] = [f"{value:.12g}" for value in refined[tag]]
        return " ".join(tokens)

    halves = _deal(chosen, series, args.seed)
    start, start_note = _start(run, project, channel, number, args.start_lowpass, spacing)
    written: Path | None = None
    for half, folder in folders.items():
        tags = sorted(halves[half])
        write_subexport(export, folder, [(tag, posed(tag)) for tag in tags], manifest, {
            "consensus_half": {
                "half": half, "other_half": str(folders[3 - half]), "from": str(export),
                "poses": f"{project.name} class {channel}, iteration {number}",
                "agree": f"{agree.name} class {agree_channel}, iteration {agree_number}",
                "particles": len(tags), "seed": args.seed,
                "start_reference": "start_reference.mrc: " + start_note,
                "start_lowpass_angstrom": args.start_lowpass,
            },
        })
        # One file, copied: mrcfile stamps every file it writes with the time to the
        # second, so two writes of one map could differ in that byte.
        target = folder / "start_reference.mrc"
        if written is None:
            _write_mrc(target, start, spacing)
            written = target
        else:
            shutil.copyfile(written, target)
        counts = defaultdict(int)
        for tag in tags:
            counts[series[tag]] += 1
        print(f"half {half}: {len(tags):,} particles in {folder}; by series: "
              + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())))
    print(f"start_reference.mrc in both: {start_note}")
    return 0


def split_runs(args: argparse.Namespace) -> int:
    """Two runs of the registry split into halves written as a run of its own,
    <project>/mra/<OUT> (docs/halves_from_runs.md): halves.parquet, each kept particle's
    half and starting pose, and start_reference.mrc; its record lists `half 1` and
    `half 2` as selections."""

    (project, channel), (agree, agree_channel) = args.poses, args.agree
    plans = [json.loads((path / PLAN).read_text()) for path in (project, agree)]
    particles = plans[0]["particles"]
    if any("selection" in plan["particles"] for plan in plans):
        raise SystemExit("a run of part of a set (a half) is not split again")
    if plans[1]["particles"] != particles:
        raise SystemExit(f"{project} and {agree} read different particles: "
                         f"{particles} and {plans[1]['particles']}")
    home = (Path(particles["project"]) / "mra").resolve()
    out = args.out.resolve()
    if out.parent != home:
        raise SystemExit(f"a split of runs from the registry lives in {home}; give --out "
                         f"{home}/<name>")
    if out.exists():
        raise SystemExit(f"{out} exists; give the split a new name")

    run, number, members = _members(project, channel)
    agree_run, agree_number, agreed = _members(agree, agree_channel)
    chosen = sorted(members & agreed)
    print(f"in both: {len(chosen):,}")
    if len(chosen) < 2:
        raise SystemExit("fewer than two particles to split")
    other = [tag for tag in chosen if run.identity.get(tag) != agree_run.identity.get(tag)]
    if other:
        raise SystemExit(f"{len(other)} tags name different particles in the two runs, e.g. "
                         f"tag {other[0]}: {run.identity.get(other[0])} and "
                         f"{agree_run.identity.get(other[0])}")
    spacing = run.pixel_size_angstrom
    if not spacing:
        raise SystemExit(f"{project}: no pixel size")
    if not 2 * spacing < args.start_lowpass:
        raise SystemExit(f"--start-lowpass {args.start_lowpass} A is not coarser than Nyquist "
                         f"({2 * spacing:.2f} A)")
    picked = args.start_poses == "picked"
    starts = _poses(run, None if picked else (number, channel))
    pose_note = (f"the poses {project.name} started the particles from (their picks')"
                 if picked else f"{project.name}'s refined poses in class {channel}, "
                 f"iteration {number}")
    halves = _deal(chosen, run.series, args.seed)

    nodes = [(project, number), (agree, agree_number)]
    records = [completed(node_folder(path, at)) for path, at in nodes]
    inputs = []
    for slot, (path, at), record in zip(("poses", "agree"), nodes, records, strict=True):
        if record is None:
            raise SystemExit(f"{node_folder(path, at)}: no record that succeeded")
        inputs.append(RunInput(slot=slot, path=str(node_folder(path, at)),
                               fingerprint=record.fingerprint))
    if picked:
        start, start_note = _picked_start(project, chosen, args.start_lowpass, spacing,
                                          args.workers)
    else:
        start, start_note = _start(run, project, channel, number, args.start_lowpass, spacing)
    content: dict[str, Any] = {
        "poses": {"run": project.name, "channel": channel, "iteration": number},
        "agree": {"run": agree.name, "channel": agree_channel, "iteration": agree_number},
        "seed": args.seed, "start_lowpass_angstrom": float(args.start_lowpass),
        # Where the project is, the runs' plans say: paths stay out of fingerprints.
        "particles": {key: value for key, value in particles.items() if key != "project"},
    }
    if picked:
        # Named only when asked for, so the splits made before keep their fingerprints.
        content["start_poses"] = "picked"
    tags = sorted(chosen)
    half_of = {tag: half for half, members_ in halves.items() for tag in members_}
    poses = np.array([starts[tag] for tag in tags], dtype=float).reshape(len(tags), 6)
    with writing(out, step=STEP_SPLIT, step_version=SPLIT_VERSION, backend="cryoet-pipeline",
                 backend_version=SPLIT_VERSION, content_params=content,
                 inputs=inputs) as attempt:
        write_particle_table(attempt.add(ArtifactKind.PARTICLE_TABLE, HALVES), {
            "particle_id": np.array(tags, dtype=np.int64),
            "half": np.array([half_of[tag] for tag in tags], dtype=np.int8),
            "tdrot_deg": poses[:, 3], "tilt_deg": poses[:, 4], "narot_deg": poses[:, 5],
            "shift_x_px": poses[:, 0], "shift_y_px": poses[:, 1], "shift_z_px": poses[:, 2],
        }, {"set": "each kept particle's consensus half and starting pose", **DYNAMO_POSES,
            "pixel_size_angstrom": spacing, "box_px": run.box_px, "poses": pose_note,
            "start_reference": start_note})
        _write_mrc(attempt.add(ArtifactKind.CLASS_REFERENCE, START), start, spacing)
        for half in (1, 2):
            attempt.selections.append(Selection(
                name=f"half {half}", kind="particles",
                definition={"column": "half", "equals": half}, snapshot=HALVES))
            attempt.metrics[f"half_{half}_particles"] = float(len(halves[half]))
    for half in (1, 2):
        counts: dict[str, int] = defaultdict(int)
        for tag in halves[half]:
            counts[run.series[tag]] += 1
        print(f"half {half}: {len(halves[half]):,} particles; by series: "
              + ", ".join(f"{name} {count}" for name, count in sorted(counts.items())))
    print(f"starting poses: {pose_note}")
    print(f"{START}: {start_note}")
    print(f"split written to {out}; refine each half with scripts/mra_port_run.sbatch, "
          f"SPLIT={out} HALF=1 NAME={out.name}-half1 (and HALF=2) and the schedule's variables")
    return 0


def _split_half(run: Path) -> dict[str, Any] | None:
    """What a half run of the run-record layout says of its split: its half, and the
    split's record (by fingerprint), seed and start's low-pass. None for any other run."""

    if not (run / PLAN).is_file():
        return None
    setup = completed(node_folder(run, 0))
    if setup is None or setup.selection is None:
        return None
    found = [item for item in setup.inputs if item.slot == "selection"]
    if len(found) != 1:
        return None
    named = Path(found[0].path)
    record = completed(named)
    if record is None or record.step != STEP_SPLIT:
        raise SystemExit(f"{run}: the split it was set up from, {named}, is not there")
    if record.fingerprint != found[0].fingerprint:
        raise SystemExit(f"{run}: the split at {named} is not the one it was set up from")
    return {"half": int(setup.selection.split()[-1]), "from": record.fingerprint,
            "seed": record.content_params["seed"], "split": str(named),
            "start_lowpass_angstrom": record.content_params["start_lowpass_angstrom"],
            "start_poses": record.content_params.get("start_poses", "refined")}


def _half(project: Path) -> tuple[ClassificationRun, int, np.ndarray, dict[str, Any]]:
    run, number = _last_classes(project)
    classes = run.iterations[-1].classes
    if len(classes) != 1:
        raise SystemExit(f"{project}: {len(classes)} classes; a half is refined as one")
    result = next(iter(classes.values()))
    if result.average is None:
        raise SystemExit(f"{project}: iteration {number} wrote no average")
    record = _split_half(project)
    if record is None:
        manifest_path = project.parent / "export_manifest.json"
        record = (json.loads(manifest_path.read_text()).get("consensus_half")
                  if manifest_path.exists() else None)
    if not record:
        raise SystemExit(f"{project}: not a half of scripts/consensus_halves.py split")
    return run, number, result.average.load(), record


def _moves(run: ClassificationRun, number: int) -> tuple[np.ndarray, np.ndarray]:
    """Each particle's turn (deg) and shift (px) from its starting pose to the last refined one."""

    starting = _poses(run, None)
    (channel,) = run.iterations[-1].classes
    turns, shifts = [], []
    for tag, row in _poses(run, (number, channel)).items():
        before = starting[tag]
        product = euler_matrix(row[3:6]) @ euler_matrix(before[3:6]).T
        turns.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(product) - 1) / 2)))))
        shifts.append(float(np.linalg.norm(row[:3] - before[:3])))
    return np.array(turns), np.array(shifts)


def maps(args: argparse.Namespace) -> int:
    projects = [path.resolve() for path in args.halves]
    found = [_half(project) for project in projects]
    (run, number, first, record), (_, other_number, second, other_record) = found
    if {record["half"], other_record["half"]} != {1, 2} or record["from"] != other_record["from"] \
            or record["seed"] != other_record["seed"]:
        raise SystemExit("these are not the two halves of one split: "
                         f"{projects[0].parent} and {projects[1].parent}")
    if number != other_number:
        raise SystemExit(f"the halves finished different iterations, {number} and "
                         f"{other_number}; compare them once both have finished the same")
    runs = [run for run, _, _, _ in found]
    if any(one.mask is None for one in runs):
        raise SystemExit("a half has no alignment mask to read")
    masks = [one.mask.load() for one in runs if one.mask is not None]
    if not np.array_equal(*masks):
        raise SystemExit("the two halves were refined under different masks")
    spacing = run.pixel_size_angstrom
    if not spacing:
        raise SystemExit(f"{projects[0]}: no pixel size")
    halves = {record["half"]: first, other_record["half"]: second}

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: out.with_name(f"{out.name}_{name}.mrc")
             for name in ("half1", "half2", "mean", "mask")}
    _write_mrc(paths["half1"], halves[1], spacing)
    _write_mrc(paths["half2"], halves[2], spacing)
    _write_mrc(paths["mean"], (halves[1] + halves[2]) / 2, spacing)
    _write_mrc(paths["mask"], masks[0], spacing)
    start_poses = record.get("start_poses", "refined")
    print(f"iteration {number} of both halves, {spacing:g} A/px: "
          + ", ".join(str(path) for path in paths.values()))
    print("the halves started from " + ("their picks' poses" if start_poses == "picked"
                                        else "the poses the first run refined"))

    for project, (half_run, _, _, own) in zip(projects, found, strict=True):
        turns, shifts = _moves(half_run, number)
        print(f"half {own['half']} ({project.name}, {len(turns):,} particles) moved from its "
              f"start: turned median {np.median(turns):.1f} deg (90% within "
              f"{np.percentile(turns, 90):.1f}), shifted median {np.median(shifts):.2f} px "
              f"(90% within {np.percentile(shifts, 90):.2f})")

    # Crossings read as fsc_phase_randomization.py reads them, interpolated in frequency,
    # so the numbers here and in its table are the same numbers.
    curves: dict[str, Any] = {}
    for label, curve in _half_map_fsc(halves, masks[0], spacing).items():
        assert isinstance(curve, dict)
        resolutions, values = curve["resolution_angstrom"], curve["correlation"]
        curves[label] = {"resolution_angstrom": resolutions, "correlation": values,
                         **{f"falls_{level}_angstrom": falls(resolutions, values, level)
                            for level in LEVELS},
                         "rise_toward_nyquist": curve["rise_toward_nyquist"]}
    for label in ("unmasked", "masked"):
        curve = curves[label]
        texts = {level: crossing_text(curve["resolution_angstrom"], curve["correlation"], level)
                 for level in LEVELS}
        line = f"FSC {label:9s}: " + ", ".join(
            f"{level:g} " + ("never crossed" if text == "none" else f"at {text}")
            for level, text in texts.items())
        rise = curve["rise_toward_nyquist"]
        if rise:
            line += (f"; it rises again toward Nyquist, {rise['dip_correlation']:.2f} at "
                     f"{rise['dip_resolution_angstrom']:.1f} A to {rise['peak_correlation']:.2f} "
                     f"at {rise['peak_resolution_angstrom']:.1f} A ({rise['sigma']:.0f} sigma)")
        print(line)
    print("  (interpolated in frequency; a/b: the curve falls through the level more than "
          "once, first and last)")
    fsc_path = out.with_name(f"{out.name}_fsc.json")
    fsc_path.write_text(json.dumps({
        "halves": [str(project) for project in projects], "iteration": number,
        "start_poses": start_poses, "pixel_spacing_angstrom": spacing, **curves},
        indent=1) + "\n")
    print(f"curves: {fsc_path}")

    nyquist = 2 * spacing
    cutoffs = sorted({round(float(record.get("start_lowpass_angstrom") or 2 * nyquist), 2),
                      round(1.5 * nyquist, 2)}, reverse=True)
    print("What the mask adds to the masked curve, by phase randomisation:\n"
          f"  python scripts/fsc_phase_randomization.py {paths['half1']} {paths['half2']} "
          f"--mask {paths['mask']} --randomize-beyond-angstrom "
          + " ".join(f"{cutoff:g}" for cutoff in cutoffs) + " --shells")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("split", help="write the two half exports and their start")
    making.add_argument("poses", type=_class_spec,
                        help="PROJECT:CHANNEL, the class whose refined poses the halves start from")
    making.add_argument("--agree", type=_class_spec, required=True,
                        help="PROJECT:CHANNEL of another run on the same export; only particles "
                        "in both classes are kept")
    making.add_argument("--start-lowpass", type=float, required=True,
                        help="A; the common start is the class average low-passed to this")
    making.add_argument("--out", type=Path, required=True,
                        help="the halves go to <out>-half1 and <out>-half2; for runs of the "
                        "registry, the split itself, <project>/mra/<name>")
    making.add_argument("--seed", type=int, default=20261026, help="for the random split")
    making.add_argument("--start-poses", choices=("refined", "picked"), default="refined",
                        help="refined (the default): each particle starts from its pose in the "
                        "first run's class at that iteration, and the start is that class's "
                        "average; picked: from the pose the first run started it from, its "
                        "pick's, and the start is the kept particles' average at those poses, "
                        "so no pose the MRA refined reaches the halves. picked is for runs of "
                        "a registry")
    making.add_argument("--workers", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or 1),
                        help="threads for the average at the picked poses; the map does not "
                        "depend on them (default: $SLURM_CPUS_PER_TASK, else 1)")
    making.set_defaults(run=split)
    reading = commands.add_parser("maps", help="the halves' maps, their FSC and mask")
    reading.add_argument("halves", type=Path, nargs=2, help="the two half projects")
    reading.add_argument("--out", type=Path, required=True,
                         help="prefix: <out>_half1.mrc, _half2.mrc, _mean.mrc, _mask.mrc, "
                         "_fsc.json")
    reading.set_defaults(run=maps)
    args = parser.parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
