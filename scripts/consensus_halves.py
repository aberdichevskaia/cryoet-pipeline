#!/usr/bin/env python3
"""Two halves of the particles two MRA runs agree on, refined apart from one start.

    python scripts/consensus_halves.py split \\
        outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan:1 \\
        --agree outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssb_g2atan:1 \\
        --start-lowpass 53.76 --out outputs/ryr1/dynamo/ryr1-consensus
    python scripts/consensus_halves.py maps \\
        outputs/ryr1/dynamo/ryr1-consensus-half1/refine_h1 \\
        outputs/ryr1/dynamo/ryr1-consensus-half2/refine_h2 \\
        --out outputs/ryr1/dynamo/ryr1-consensus

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
"consensus_half", export_complete.json), so scripts/dynamo_mra_mdc.sbatch and the
reports take it as any export. Both get the same start_reference.mrc: the first
run's class average of that iteration, low-passed to --start-lowpass (A) with a
raised-cosine edge (cryoet_pipeline.filters.low_pass_filter_volume). Each half is
then refined on its own from that start, one channel and no MRA:
scripts/dynamo_mra_mdc.sbatch with SEEDS=map REFERENCE=<half>/start_reference.mrc.

`maps` reads the last iteration the two half projects finished and writes, as MRC at
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

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import torch
from fsc_phase_randomization import crossing_text, falls
from split_export_by_view import write_subexport

from cryoet_pipeline.backends.dynamo_search import _half_map_fsc, euler_matrix
from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.reports.classification import ClassificationRun
from cryoet_pipeline.reports.dynamo_project import (
    _starting_table,
    read_dynamo_classification,
    read_em,
    read_table,
)

POSE = slice(3, 9)
LEVELS = (0.5, 0.143)


def _class_spec(text: str) -> tuple[Path, int]:
    folder, colon, channel = text.rpartition(":")
    if not colon or not folder or not channel.isdigit():
        raise argparse.ArgumentTypeError(
            f"{text!r}: give PROJECT:CHANNEL, e.g. <export>/mra_pad_ssn_g2atan:1")
    return Path(folder).resolve(), int(channel)


def _last_classes(project: Path) -> tuple[ClassificationRun, int]:
    run = read_dynamo_classification(project, compute_initial_average=False)
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


def _results(project: Path, number: int) -> Path:
    return project / "results" / f"ite_{number:04d}" / "averages"


def _write_mrc(path: Path, volume: np.ndarray, spacing: float) -> None:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = spacing


def split(args: argparse.Namespace) -> int:
    (project, channel), (agree, agree_channel) = args.poses, args.agree
    export = project.parent
    if agree.parent != export:
        raise SystemExit(f"{agree} is not in {export}: both runs must be on the export split")
    if not (export / "export_complete.json").is_file():
        raise SystemExit(f"{export}: no export_complete.json, so no finished export to split")
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

    refined = {int(row[0]): row for row in read_table(
        _results(project, number) / f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl")}
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
        tokens[POSE] = [f"{value:.12g}" for value in refined[tag][POSE]]
        return " ".join(tokens)

    # Dealt alternately after a shuffle within each series. A series with an odd count
    # gives its extra particle to the half it started with, so the next one starts with
    # the other half.
    rng = np.random.default_rng(args.seed)
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

    average = read_em(_results(project, number) / f"average_ref_{channel:03d}_ite_{number:04d}.em")
    start = low_pass_filter_volume(torch.from_numpy(average), cutoff_angstrom=args.start_lowpass,
                                   pixel_spacing_angstrom=spacing).numpy()
    start_note = (f"{project.name}'s class {channel} average of iteration {number}, low-passed "
                  f"to {args.start_lowpass:g} A")
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


def _half(project: Path) -> tuple[ClassificationRun, int, np.ndarray, dict[str, Any]]:
    run, number = _last_classes(project)
    classes = run.iterations[-1].classes
    if len(classes) != 1:
        raise SystemExit(f"{project}: {len(classes)} classes; a half is refined as one")
    result = next(iter(classes.values()))
    if result.average is None:
        raise SystemExit(f"{project}: iteration {number} wrote no average")
    manifest_path = project.parent / "export_manifest.json"
    record = (json.loads(manifest_path.read_text()).get("consensus_half")
              if manifest_path.exists() else None)
    if not record:
        raise SystemExit(f"{project.parent}: not a half of scripts/consensus_halves.py split")
    return run, number, result.average.load(), record


def _moves(project: Path, number: int) -> tuple[np.ndarray, np.ndarray]:
    """Each particle's turn (deg) and shift (px) from its starting pose to the last refined one."""

    starting = {int(row[0]): row for row in _starting_table(project, [])}
    final = read_table(_results(project, number) / f"refined_table_ref_001_ite_{number:04d}.tbl")
    turns, shifts = [], []
    for row in final:
        before = starting[int(row[0])]
        product = euler_matrix(row[6:9]) @ euler_matrix(before[6:9]).T
        turns.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(product) - 1) / 2)))))
        shifts.append(float(np.linalg.norm(row[3:6] - before[3:6])))
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
    masks = [read_em(project / "settings" / "mask.em") for project in projects]
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
    print(f"iteration {number} of both halves, {spacing:g} A/px: "
          + ", ".join(str(path) for path in paths.values()))

    for project, (_, _, _, own) in zip(projects, found, strict=True):
        turns, shifts = _moves(project, number)
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
        "pixel_spacing_angstrom": spacing, **curves}, indent=1) + "\n")
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
                        help="the halves go to <out>-half1 and <out>-half2")
    making.add_argument("--seed", type=int, default=20261026, help="for the random split")
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
