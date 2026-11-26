#!/usr/bin/env python
"""Export a class of an MRA run, or picks or particle stacks, to a folder RELION 5 runs as it is.

After an MRA run someone says which class looks right; this writes that class
for RELION's high-resolution refinement (docs/relion_export.md):

    python scripts/export_relion.py --run <project>/mra/<NAME> --class 1 --dry-run
    python scripts/export_relion.py --run <project>/mra/<NAME> --class 1

The dry run prints what would be written and writes nothing: the particles, the
decoys and duplicates left out, the halves, every tilt series with its views,
stack size and what its rows fall back on. The person confirms those numbers;
the export writes <project>/relion/<NAME>-class<C> (or --out), in the
run-record layout (run.json written last, the folder renamed into place):

    tomograms.star, tilt_series/   the tilt images the tomograms were made from,
                                   unaligned, and each one's geometry, CTF, dose
    particles.star                 the class: positions, poses, rlnRandomSubset
    optimisation_set.star
    reference/                     the class average and the MRA mask, in RELION's
                                   frame (mirrored in z), at the MRA's pixel and box
    relion_commands.sh             Extract and Refine3D, at the MRA's pixel first
    selection.json                 each particle's tag, RELION name and half; the
                                   duplicates left out and what they duplicate
    README.md

Picks or particle stacks leave the same way without an MRA run, for someone who goes
on in RELION after picking (docs/entry_and_exit.md, Exits):

    python scripts/export_relion.py --project <project> \\
        --particles TS_01:picks:pytom TS_02:picks:pytom --box 48 --dry-run

Each particle is placed and posed as an MRA run of its extracted stack would start
(relion_export.read_particle_set), and then converted as a class's particles are.
Such an export has no reference/: relion_commands.sh extracts and, when the picks
have poses, reconstructs the average at them; refinement is left to the person.
Picks with no poses, or only some, go as positions only.

Nothing is submitted, and relion_commands.sh has not been run by the export.

Every dry run and export goes into the project's journal.jsonl, through
scripts/submit_lanes.py's journal, and an export from an agentic client (the
assistant) needs --reason and must be the same person's last dry run of the same
run and class, unchanged, at most 30 minutes old, as a submission must.

--relion-check <relion_tomo_reconstruct_particle> then has RELION reconstruct the
exported class from the tilt images at the exported poses (no CTF, at the MRA's
pixel) and compares it with the exported reference: the same check as
scripts/relion_particle_check.py, on the real class. It passes when the reference
as written correlates best of its eight axis reversals, twice the next, within
half a voxel.

Exit status 0 on success (and a passing check), 1 when the check fails, 2 on a
usage error or a run that cannot be exported.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import ArtifactKind
from cryoet_pipeline.relion_export import (
    ParticleSet,
    RelionParticle,
    RunClass,
    SeriesSources,
    assign_halves,
    box_start,
    export_series,
    optimisation_set_text,
    particle_centre,
    read_particle_set,
    read_run_class,
    relion_pose,
    remove_duplicates,
    series_notes,
    series_sources,
    standard_frame_reference,
    write_particles,
    write_tomogram_set,
)
from cryoet_pipeline.runs import RunInput, files_fingerprint, writing

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit_lanes  # noqa: E402  (the journal and the dry-run gate of submissions)

# Step version 2 (11 November 2026): halves balanced over the whole class.
STEP, STEP_VERSION, BACKEND, BACKEND_VERSION = "export-relion", 3, "relion5", 1
JOURNAL_ACTION = "export-relion"
"""What the project's journal calls this script's dry runs and exports."""
"""Declared: raised by hand when a change moves what the export writes."""
MIN_STACK_PIXEL = 1.0
"""Angstrom: the default stack binning is the least that brings the tilt images to
this pixel or coarser (super-resolution sums binned to the camera's pixel)."""


@dataclass(frozen=True)
class SeriesPlan:
    name: str
    sources: SeriesSources
    stack_binning: int
    relion_bin: int
    views: int
    stack_gb: float
    particles: int
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ExportPlan:
    chosen: RunClass | ParticleSet
    project: Path
    out: Path
    kept: list[int]
    duplicates: list[tuple[int, int]]
    halves: dict[int, int]
    linked_sizes: tuple[int, ...]  # groups of neighbours kept in one half, largest first
    centres: dict[int, np.ndarray]
    series: list[SeriesPlan]
    min_distance: float
    seed: int


def _class_spec(text: str) -> tuple[Path, int]:
    folder, colon, channel = text.rpartition(":")
    if not colon or not folder or not channel.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r}: give RUN:CLASS, e.g. <project>/mra/run2:1")
    return Path(folder), int(channel)


def default_stack_binning(tomogram_binning: int, raw_pixel: float) -> int:
    for binning in range(1, tomogram_binning + 1):
        if tomogram_binning % binning == 0 and raw_pixel * binning >= MIN_STACK_PIXEL:
            return binning
    return tomogram_binning


def particles_name(ids: tuple[str, ...]) -> str:
    """The default export folder's name for picks or stacks: their ids, ':' as '_'."""

    names = [artifact_id.replace(":", "_") for artifact_id in ids]
    return "-".join(names) if len(names) <= 2 else f"{names[0]}-and-{len(names) - 1}-more"


def make_plan(args: argparse.Namespace) -> ExportPlan:
    chosen: RunClass | ParticleSet
    tomogram_of_series: dict[str, str] = {}
    if args.particles:
        project = Path(args.project).resolve()
        chosen = read_particle_set(project, tuple(args.particles), box=args.box)
        tomogram_of_series = dict(chosen.tomograms)
    else:
        chosen = read_run_class(args.run, args.channel, iteration=args.iteration, agree=args.agree)
        project = (args.project or chosen.project).resolve()
        registry = ArtifactRegistry.load(project / "artifacts.json")
        tomogram_of_stack: dict[str, str] = {}
        for stack_id in {member.stack_id for member in chosen.members}:
            stack = registry.get(stack_id)
            parents = [registry.get(parent) for parent in stack.parent_ids]
            tomograms = [parent.id for parent in parents if parent.kind is ArtifactKind.TOMOGRAM]
            if len(tomograms) != 1:
                raise ValueError(
                    f"particle stack {stack_id} has {len(tomograms)} tomograms as parents"
                )
            tomogram_of_stack[stack_id] = tomograms[0]
        for member in chosen.members:
            tomogram = tomogram_of_stack[member.stack_id]
            if tomogram_of_series.setdefault(member.series, tomogram) != tomogram:
                raise ValueError(f"{member.series}'s particles come from two tomograms")

    centres = {member.tag: particle_centre(member, chosen.box) for member in chosen.members}
    series_of = {member.tag: member.series for member in chosen.members}
    min_distance = args.min_distance if args.min_distance is not None else chosen.box / 4
    kept, duplicates = remove_duplicates(
        centres, series_of, {member.tag: member.cc for member in chosen.members}, min_distance
    )
    halves, linked = assign_halves(
        {tag: centres[tag] for tag in kept}, series_of, float(chosen.box), args.seed
    )
    counts = Counter(series_of[tag] for tag in kept)
    plans = []
    for name in sorted(counts):
        sources = series_sources(project, name, tomogram_of_series[name])
        tomogram_binning = int(sources.tomogram.binning or 1)
        raw_px = float(sources.final_stack.parameters["raw_pixel_spacing_angstrom"])
        binning = args.stack_binning or default_stack_binning(tomogram_binning, raw_px)
        if tomogram_binning % binning:
            raise ValueError(
                f"{name}: stack binning {binning} does not divide the tomogram's {tomogram_binning}"
            )
        _, ny, nx = (int(size) for size in sources.raw_stack.shape or (0, 0, 0))
        views = len(
            sources.tomogram.parameters.get("included_z_values") or sources.alignment.transforms
        )
        plans.append(
            SeriesPlan(
                name=name,
                sources=sources,
                stack_binning=binning,
                relion_bin=tomogram_binning // binning,
                views=views,
                stack_gb=views * (nx // binning) * (ny // binning) * 4 / 1e9,
                particles=counts[name],
                notes=tuple(series_notes(sources, dose_per_tilt=args.dose_per_tilt)),
            )
        )
    if len({plan.relion_bin for plan in plans}) != 1:
        raise ValueError(
            "the series would need different RELION binnings to reach the MRA's "
            "pixel; give --stack-binning"
        )
    if isinstance(chosen, ParticleSet):
        default = project / "relion" / particles_name(chosen.artifact_ids)
    else:
        default = project / "relion" / f"{chosen.run.name}-class{chosen.channel}"
    out = args.out or default
    return ExportPlan(
        chosen=chosen,
        project=project,
        out=out,
        kept=kept,
        duplicates=duplicates,
        halves=halves,
        linked_sizes=tuple(linked),
        centres=centres,
        series=plans,
        min_distance=min_distance,
        seed=args.seed,
    )


POSES = {
    "given": "the picks' own, the poses an MRA run of their stack starts from",
    "none": "none, so positions only",
    "partial": "only some particles have one, so positions only",
}


def describe(plan: ExportPlan) -> list[str]:
    chosen = plan.chosen
    if isinstance(chosen, ParticleSet):
        lines = [f"{chosen.source} {', '.join(chosen.artifact_ids)}: "
                 f"{len(chosen.members):,} particles"]
    else:
        total = len(chosen.members) + chosen.decoys_left_out + chosen.not_agreed
        lines = [
            f"class {chosen.channel} of {chosen.run} after iteration {chosen.iteration}: "
            f"{total:,} particles",
        ]
        if chosen.decoys_left_out:
            lines.append(f"  decoys left out: {chosen.decoys_left_out:,} (decoys in this class)")
        if chosen.agreed_with:
            lines.append(f"  not in {chosen.agreed_with}: {chosen.not_agreed:,} left out")
    lines.append(
        f"  duplicates left out: {len(plan.duplicates):,} (centres within "
        f"{plan.min_distance:g} voxels of a better-scoring particle)"
    )
    halves = Counter(plan.halves.values())
    linked = plan.linked_sizes
    together = (
        f"{len(linked):,} groups of neighbours within a box kept together, the largest "
        f"{linked[0]:,}"
        if linked
        else "no neighbours within a box"
    )
    lines.append(
        f"  exported: {len(plan.kept):,}; half 1 {halves[1]:,}, half 2 {halves[2]:,} "
        f"({together}; seed {plan.seed})"
    )
    if isinstance(chosen, ParticleSet):
        lines.append(
            f"  box {chosen.box} voxels at {chosen.pixel_angstrom:.3f} A, the tomograms' "
            f"pixel, for RELION to extract; poses: {POSES[chosen.angles]}"
        )
        lines.extend(f"  note: {note}" for note in chosen.notes)
        pixel_of = "the tomograms' pixel"
    else:
        lines.append(
            f"  box {chosen.box} voxels at {chosen.pixel_angstrom:.3f} A, the MRA's; "
            f"particles traced through {chosen.source}"
        )
        pixel_of = "the MRA's pixel"
    for series in plan.series:
        lines.append(
            f"  {series.name}: {series.particles:,} particles, {series.views} views, "
            f"tilt images binned {series.stack_binning} "
            f"({series.stack_gb:.1f} GB as float32), RELION at --bin {series.relion_bin} "
            f"for {pixel_of}"
        )
        lines.extend(f"    note: {note}" for note in series.notes)
    lines.append(f"  to {plan.out}")
    return lines


def plan_record(plan: ExportPlan, args: argparse.Namespace) -> dict[str, Any]:
    """What the dry run showed, as the journal keeps it: an export from the assistant
    must be this record unchanged (scripts/submit_lanes.py's `unshown`). The run, the
    class and the iteration name the thread, as a config and its series do for a lane."""

    chosen = plan.chosen
    if isinstance(chosen, ParticleSet):
        thread = {"config": str(plan.project),
                  "series": "particles " + ", ".join(chosen.artifact_ids),
                  "agree": None, "decoys_left_out": 0, "not_agreed_left_out": 0,
                  "box": chosen.box, "angles": chosen.angles}
    else:
        thread = {"config": str(chosen.run.resolve()),
                  "series": f"class {chosen.channel} after iteration {chosen.iteration}",
                  "agree": chosen.agreed_with, "decoys_left_out": chosen.decoys_left_out,
                  "not_agreed_left_out": chosen.not_agreed}
    return {
        "recipe": JOURNAL_ACTION,
        "config": thread.pop("config"),
        "series": thread.pop("series"),
        "project": str(plan.project),
        "agree": thread.pop("agree"),
        "particles": len(plan.kept),
        "decoys_left_out": thread.pop("decoys_left_out"),
        "not_agreed_left_out": thread.pop("not_agreed_left_out"),
        **thread,
        "duplicates_left_out": len(plan.duplicates),
        "halves": {
            str(half): count for half, count in sorted(Counter(plan.halves.values()).items())
        },
        "seed": plan.seed,
        "min_distance": plan.min_distance,
        "stacks": {
            series.name: {
                "binning": series.stack_binning,
                "views": series.views,
                "relion_bin": series.relion_bin,
                "notes": list(series.notes),
            }
            for series in plan.series
        },
        "out": str(plan.out),
        "hand": args.hand,
        "dose_per_tilt": args.dose_per_tilt,
        "ini_high": args.ini_high,
        "relion_check": bool(args.relion_check),
        "env": {},
    }


def refusal(
    record: dict[str, Any],
    who: dict[str, Any],
    reason: str | None,
    rows: list[dict[str, Any]],
    now: datetime,
) -> str | None:
    """Why an export may not be written now, or None. Only an agentic client is held
    to the dry run it showed, as for a submission: --reason, and the person's own last
    dry run of this run and class, unchanged and at most 30 minutes old."""

    if not who["assistant"]:
        return None
    if not (reason or "").strip():
        return "an export from the assistant needs --reason: what was asked, and by whom"
    why = submit_lanes.unshown(rows, record, who["user"], now)
    if why:
        return f"{why}; dry-run it, show the dry run, and export after a yes"
    return None


def _write_mrc(path: Path, volume: np.ndarray, pixel: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = pixel


RELION_ENVIRONMENT = """if [ -n "${RELION5_MODULE:-}" ]; then
    if ! type module >/dev/null 2>&1 && [ -n "${MODULESHOME:-}" ]; then
        source "$MODULESHOME/init/bash"
    fi
    # One or more module names, split on spaces.
    # shellcheck disable=SC2086
    module load $RELION5_MODULE
fi
if [ -n "${RELION5_HOME:-}" ]; then
    export PATH="$RELION5_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$RELION5_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
for program in @PROGRAMS@; do
    if ! command -v "$program" >/dev/null 2>&1; then
        echo "$program not found: set RELION5_MODULE and/or RELION5_HOME, or put RELION 5" \\
            "on PATH" >&2
        exit 2
    fi
done
# A RELION whose libraries this node lacks fails here, not after an hour of Extract.
if ! version=$(relion_refine --version 2>&1); then
    echo "relion_refine does not start on $(hostname): $version" >&2
    exit 2
fi
echo "$version" | head -1
"""


# Why Extract writes 2D stacks (--stack2d), RELION's own default since 4.1. On
# cryoem.se on 26 November 2026, RELION 5.1.1 extracted 3D pseudo-subtomograms from a
# RyR1 export and Refine3D stopped at its first expectation: "subtomo multiplicity
# correction only applies to ctf_premultiplied data". 5.1.1's relion_tomo_subtomo still
# multiplies 3D pseudo-subtomograms by the CTF but marks them as not premultiplied
# (5.0 marked them premultiplied); 2D stacks are marked premultiplied in both.
STACK2D_COMMENT = (
    "# As 2D stacks: RELION 5.1.1 marks 3D pseudo-subtomograms as not CTF-premultiplied,\n"
    "# though it multiplies them, and its Refine3D refuses them (docs/relion_export.md)."
)


def relion_environment(programs: str) -> str:
    """Where RELION comes from, and a stop before anything runs when `programs`, or
    RELION itself, cannot start on this node."""

    return RELION_ENVIRONMENT.replace("@PROGRAMS@", programs).rstrip("\n")


def commands_text(plan: ExportPlan, diameter: float, ini_high: float) -> str:
    box, relion_bin = plan.chosen.box, plan.series[0].relion_bin
    return f"""#!/usr/bin/env bash
# RELION 5, run from this folder in a job with MPI (and GPUs for Refine3D). Written by
# scripts/export_relion.py and not run by it: review before submitting (README.md says
# why each value is what it is).
#
# RELION comes from PATH, from the modules RELION5_MODULE names (one or more, loaded
# here), and from RELION5_HOME, an installation whose bin/ and lib/ go first. At MDC
# (10 November 2026): RELION5_MODULE=openmpi/v5.0.5-xchu and
# RELION5_HOME=/fast/AG_Kudryashev/software/relion-5.0.1/build-gpu_sm86_cu11.7, as the
# module relion/v5/gpu/sm86-cu11.7 brings an Open MPI that lacks its UCX libraries.
# Refine3D uses the GPUs the job sees, unless USE_GPU=0. THREADS and MPI set the
# threads per rank and the ranks; a Slurm job needs --ntasks of at least MPI, as mpirun
# takes its slots from Slurm. mpirun keeps its session files and shared memory in
# RELION5_MPI_TMPDIR, by default /dev/shm, not in a TMPDIR on a shared file system.
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${{THREADS:-8}}
MPI=${{MPI:-3}}

{relion_environment("relion_tomo_subtomo relion_refine relion_refine_mpi mpirun")}
if [ -n "${{SLURM_NTASKS:-}}" ] && [ "$SLURM_NTASKS" -lt "$MPI" ]; then
    echo "this job has $SLURM_NTASKS task(s) and mpirun needs $MPI: submit it with" \\
        "--ntasks=$MPI, or set MPI" >&2
    exit 2
fi
if [ -n "${{RELION5_MPI_TMPDIR:-}}" ]; then
    export PRTE_MCA_prte_tmpdir_base="$RELION5_MPI_TMPDIR"
elif [ -z "${{PRTE_MCA_prte_tmpdir_base:-}}" ] && [ -d /dev/shm ] && [ -w /dev/shm ]; then
    export PRTE_MCA_prte_tmpdir_base=/dev/shm
fi
GPU_ARGS=()
if [ "${{USE_GPU:-1}}" != 0 ] && command -v nvidia-smi >/dev/null 2>&1 \\
        && nvidia-smi -L >/dev/null 2>&1; then
    GPU_ARGS=(--gpu "")
fi

# Pseudo-subtomograms at the MRA's pixel ({plan.chosen.pixel_angstrom:.3f} A) and box,
# so the reference fits them as it is.
{STACK2D_COMMENT}
mkdir -p Extract/job001
relion_tomo_subtomo --i optimisation_set.star --o Extract/job001/ \\
    --b {2 * box} --crop {box} --bin {relion_bin} --stack2d --float16 --j "$THREADS"

# Gold-standard refinement from the MRA poses: local searches only, C1.
mkdir -p Refine3D/job002
mpirun -n "$MPI" relion_refine_mpi --ios Extract/job001/optimisation_set.star \\
    --ref reference/class_{plan.chosen.channel}.mrc --solvent_mask reference/mask.mrc \\
    --firstiter_cc --ini_high {ini_high:g} --o Refine3D/job002/run \\
    --auto_refine --split_random_halves --low_resol_join_halves 40 \\
    --particle_diameter {diameter:g} --flatten_solvent --zero_mask --ctf --pad 2 \\
    --healpix_order 4 --auto_local_healpix_order 4 --sigma_ang 3 \\
    --offset_range 3 --offset_step 1 --sym C1 --norm --scale \\
    --dont_combine_weights_via_disc --pool 10 --j "$THREADS" \\
    ${{GPU_ARGS[@]+"${{GPU_ARGS[@]}}"}}
"""


def particles_commands_text(plan: ExportPlan) -> str:
    """Extract, and with poses the average at them: no refinement, whose first step
    (Class3D or Refine3D, global or local) is a choice about the data."""

    chosen = plan.chosen
    assert isinstance(chosen, ParticleSet)
    box, relion_bin = chosen.box, plan.series[0].relion_bin
    programs = "relion_tomo_subtomo relion_refine" + (
        " relion_tomo_reconstruct_particle" if chosen.angles == "given" else ""
    )
    text = f"""#!/usr/bin/env bash
# RELION 5, run from this folder in a job. Written by scripts/export_relion.py and not
# run by it: review before submitting (README.md says why each value is what it is).
#
# RELION comes from PATH, from the modules RELION5_MODULE names (one or more, loaded
# here), and from RELION5_HOME, an installation whose bin/ and lib/ go first; README.md
# says what works at MDC. THREADS sets the threads.
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${{THREADS:-8}}

{relion_environment(programs)}

# Pseudo-subtomograms at the tomograms' pixel ({chosen.pixel_angstrom:.3f} A), {box} voxels.
{STACK2D_COMMENT}
mkdir -p Extract/job001
relion_tomo_subtomo --i optimisation_set.star --o Extract/job001/ \\
    --b {2 * box} --crop {box} --bin {relion_bin} --stack2d --float16 --j "$THREADS"
"""
    if chosen.angles == "given":
        text += f"""
# The average at the picks' poses, made from the tilt images: a reference in RELION's
# own frame for what comes next.
mkdir -p Reconstruct/job002
relion_tomo_reconstruct_particle --i optimisation_set.star --o Reconstruct/job002/ \\
    --b {box} --bin {relion_bin} --j "$THREADS"
"""
    return text


def submit_hint(out: Path) -> str:
    """The line that runs an export's commands in a Slurm job, with the folder filled
    in: a person who has to guess the folder submits from the wrong one (MDC, job
    3914512, 10 November 2026: a folder with no export, exit 127). The GPU options are
    the site's: at MDC --partition=gpu --gres=gpu:a40:2. One node: --gres counts GPUs
    per node, and mpirun's ranks would follow tasks Slurm spread over several."""

    return (
        "sbatch --nodes=1 --ntasks=3 --cpus-per-task=8 --time=1-00:00:00 "
        "<GPU partition and --gres> "
        f'--chdir={Path(out).resolve()} --wrap "./relion_commands.sh"'
    )


def particles_submit_hint(out: Path) -> str:
    """The same for an export of picks or stacks: one task, no GPU."""

    return (
        "sbatch --nodes=1 --ntasks=1 --cpus-per-task=8 --time=12:00:00 "
        f'--chdir={Path(out).resolve()} --wrap "./relion_commands.sh"'
    )


def readme_text(plan: ExportPlan, written: dict, diameter: float, ini_high: float) -> str:
    notes = sorted({note for series in written["series"] for note in series["notes"]})
    lines = [
        f"# Class {plan.chosen.channel} of {plan.chosen.run.name}, for RELION 5",
        "",
        "Written by scripts/export_relion.py (docs/relion_export.md). The numbers:",
        "",
        "```text",
        *describe(plan),
        "```",
        "",
        "## Frame",
        "",
        "RELION is given IMOD's standard orientation of each tomogram, the one its own",
        "IMOD import gives. The pipeline's tomograms and MRA maps are its mirror in z, so",
        "`reference/` holds the class average and mask mirrored in z about the box centre,",
        "and the positions and poses in particles.star are mirrored with them. A map RELION",
        "makes from this folder is the mirror of the pipeline's MRA maps.",
        "",
        "## What the rows fall back on",
        "",
        *([f"- {note}" for note in notes] or ["- nothing: CTF, dose and order are the data's own"]),
        f"- rlnTomoHand is {written['hand']:+d}, not measured for these data (RELION's",
        "  documentation: it has to be determined experimentally; always -1 in their hands).",
        f"- RELION's model drops tiltalign's per-view magnification: up to "
        f"{written['dropped_angstrom_max']:.1f} A at the corners of the aligned images.",
        "",
        "## Commands",
        "",
        "`relion_commands.sh` extracts pseudo-subtomograms, as 2D stacks, at the MRA's pixel",
        "and box (so the reference fits them as it is, RELION refuses a reference of another",
        "size) and",
        "refines with local searches from the MRA poses (healpix 4, sigma 3 deg, offsets of",
        f"3 pixels), C1, from the reference low-passed to {ini_high:g} A, with a particle",
        f"diameter of {diameter:g} A. They are starting points; finer pixels come after, by",
        "extracting again at a lower --bin.",
        "",
        "Run it in a job with MPI: as many tasks as `MPI` ranks (3 by default), `THREADS`",
        "CPUs each (8), and GPUs for Refine3D, which it uses unless `USE_GPU=0`. RELION comes",
        "from PATH, from the modules `RELION5_MODULE` names, and from `RELION5_HOME` (its",
        "bin/ and lib/); the script's header says what works at MDC. It first checks that",
        "RELION starts, and stops before running anything if it does not. From any",
        "folder (at MDC the GPU options are `--partition=gpu --gres=gpu:a40:2`):",
        "",
        "```bash",
        submit_hint(plan.out),
        "```",
        "",
        "## Checks",
        "",
        "The geometry and the particle conversion were checked with RELION's own",
        "reconstructions (docs/relion_export.md). To check them on these data:",
        "`scripts/relion_geometry_check.py` and `scripts/relion_particle_check.py` on a",
        "series of this project, or `scripts/export_relion.py --relion-check`.",
        "",
    ]
    return "\n".join(lines)


def particles_readme_text(plan: ExportPlan, written: dict) -> str:
    chosen = plan.chosen
    assert isinstance(chosen, ParticleSet)
    notes = sorted({note for series in written["series"] for note in series["notes"]})
    posed = chosen.angles == "given"
    lines = [
        f"# {chosen.source.capitalize()} {', '.join(chosen.artifact_ids)}, for RELION 5",
        "",
        "Written by scripts/export_relion.py from picks or particle stacks, without an MRA",
        "run (docs/relion_export.md, docs/entry_and_exit.md). The numbers:",
        "",
        "```text",
        *describe(plan),
        "```",
        "",
        "## Frame",
        "",
        "RELION is given IMOD's standard orientation of each tomogram, the one its own",
        "IMOD import gives. The pipeline's tomograms are its mirror in z, so the positions",
        "and poses in particles.star are mirrored with them.",
        "",
        "## Poses",
        "",
        *(
            [
                "Each particle has the pose it was picked with, the pose an MRA run of its",
                "extracted stack starts from: template matching's, for pytom's picks. It is a pose",
                "of the template as the pipeline keeps it, which in RELION's frame is that",
                "template mirrored in z.",
            ]
            if posed
            else [
                "The picks have no poses, or only some do, so particles.star holds positions",
                "only, as RELION's own picking writes them, and what comes next has to search",
                "globally. A particle is where the identity pose puts it.",
            ]
        ),
        "",
        "## What the rows fall back on",
        "",
        *([f"- {note}" for note in chosen.notes]),
        *([f"- {note}" for note in notes] or ["- nothing: CTF, dose and order are the data's own"]),
        f"- rlnTomoHand is {written['hand']:+d}, not measured for these data (RELION's",
        "  documentation: it has to be determined experimentally; always -1 in their hands).",
        f"- RELION's model drops tiltalign's per-view magnification: up to "
        f"{written['dropped_angstrom_max']:.1f} A at the corners of the aligned images.",
        "",
        "## Commands",
        "",
        "`relion_commands.sh` extracts pseudo-subtomograms, as 2D stacks, at the tomograms'",
        "pixel and a box",
        f"of {chosen.box} voxels"
        + (
            ", and reconstructs the average at the picks' poses from the tilt images, a"
            if posed
            else "."
        ),
        *(
            [
                "reference in RELION's own frame. It writes no refinement: whether Class3D or",
                "Refine3D comes first, and with what searches, is a choice about the data.",
            ]
            if posed
            else ["It writes no refinement or classification: those are choices about the data."]
        ),
        "",
        "RELION comes from PATH, from the modules `RELION5_MODULE` names, and from",
        "`RELION5_HOME` (its bin/ and lib/). At MDC (10 November 2026):",
        "`RELION5_MODULE=openmpi/v5.0.5-xchu` and",
        "`RELION5_HOME=/fast/AG_Kudryashev/software/relion-5.0.1/build-gpu_sm86_cu11.7`.",
        "From any folder:",
        "",
        "```bash",
        particles_submit_hint(plan.out),
        "```",
        "",
        "## Checks",
        "",
        "The geometry and the particle conversion were checked with RELION's own",
        "reconstructions (docs/relion_export.md). `scripts/export_relion.py --particles ...",
        "--relion-check` has RELION reconstruct these particles at the exported poses and",
        "compares that with the pipeline's own average of the same boxes at the same poses.",
        "",
    ]
    return "\n".join(lines)


def write(plan: ExportPlan, args: argparse.Namespace) -> dict:
    chosen = plan.chosen
    shared = {
        "stack_binning": {s.name: s.stack_binning for s in plan.series},
        "min_distance_voxels": plan.min_distance,
        "seed": plan.seed,
        "hand": args.hand,
        "dose_per_tilt": args.dose_per_tilt,
    }
    if isinstance(chosen, ParticleSet):
        content: dict[str, Any] = {
            "particles": list(chosen.artifact_ids),
            "box": chosen.box,
            "angles": chosen.angles,
            **shared,
        }
        inputs = [
            RunInput(slot=f"particles {artifact_id}", path=str(plan.project), fingerprint=value)
            for artifact_id, value in chosen.record_fingerprints.items()
        ]
        class_number, posed = 1, chosen.angles == "given"
    else:
        content = {
            "run": str(chosen.run.resolve()),
            "iteration": chosen.iteration,
            "class": chosen.channel,
            "agree": chosen.agreed_with,
            **shared,
        }
        inputs = [
            RunInput(slot=slot, path=str(chosen.run.resolve()), fingerprint=value)
            for slot, value in chosen.record_fingerprints.items()
        ]
        class_number, posed = chosen.channel, True
    inputs.append(
        RunInput(
            slot="registry",
            path=str(plan.project / "artifacts.json"),
            fingerprint=files_fingerprint([plan.project / "artifacts.json"]),
        )
    )
    with writing(
        plan.out,
        step=STEP,
        step_version=STEP_VERSION,
        backend=BACKEND,
        backend_version=BACKEND_VERSION,
        content_params=content,
        inputs=inputs,
    ) as attempt:
        folder = attempt.folder
        exported = {}
        for series in plan.series:
            exported[series.name] = export_series(
                series.sources,
                folder,
                stack_binning=series.stack_binning,
                dose_per_tilt=args.dose_per_tilt,
            )
            attempt.add(ArtifactKind.TILT_STACK, f"tilt_series/{series.name}.mrcs")
        optics = {e.optics for e in exported.values()}
        if len(optics) != 1:
            raise ValueError(f"the series have different optics: {optics}")
        write_tomogram_set(
            folder,
            {
                name: (e.views, e.placement.size_xyz(), e.placement.stack_pixel_angstrom)
                for name, e in exported.items()
            },
            optics=optics.pop(),
            hand=args.hand,
        )
        attempt.add(ArtifactKind.DATASET_EXPORT, "tomograms.star")

        member_of = {member.tag: member for member in chosen.members}
        particles, names = [], {}
        numbers: Counter[str] = Counter()
        for tag in plan.kept:
            member = member_of[tag]
            numbers[member.series] += 1
            centred, angles = relion_pose(
                box_start_xyz=box_start(member.pick_xyz, chosen.box),
                box=chosen.box,
                dynamo_angles_deg=member.angles_deg,
                dynamo_shift_px=member.shifts_px,
                placement=exported[member.series].placement,
            )
            particles.append(
                RelionParticle(
                    member.series,
                    numbers[member.series],
                    tuple(centred),
                    angles,
                    class_number=class_number,
                    half_set=plan.halves[tag],
                )
            )
            names[tag] = f"{member.series}/{numbers[member.series]}"
        write_particles(folder / "particles.star", particles, angles=posed)
        attempt.add(ArtifactKind.DATASET_EXPORT, "particles.star")
        (folder / "optimisation_set.star").write_text(
            optimisation_set_text("particles.star", "tomograms.star")
        )
        attempt.add(ArtifactKind.DATASET_EXPORT, "optimisation_set.star")

        commands = folder / "relion_commands.sh"
        if isinstance(chosen, ParticleSet):
            commands.write_text(particles_commands_text(plan))
        else:
            _write_mrc(
                folder / "reference" / f"class_{chosen.channel}.mrc",
                standard_frame_reference(chosen.average),
                chosen.pixel_angstrom,
            )
            _write_mrc(
                folder / "reference" / "mask.mrc",
                standard_frame_reference(chosen.mask),
                chosen.pixel_angstrom,
            )
            attempt.add(ArtifactKind.CLASS_AVERAGE, f"reference/class_{chosen.channel}.mrc")
            attempt.add(ArtifactKind.ALIGNMENT_MASK, "reference/mask.mrc")
            diameter = particle_diameter(chosen.run, chosen.box, chosen.pixel_angstrom)
            ini_high = float(args.ini_high)
            commands.write_text(commands_text(plan, diameter, ini_high))
        commands.chmod(0o755)
        attempt.add(ArtifactKind.DATASET_EXPORT, "relion_commands.sh")

        dropped_max = max(
            e.dropped_px_max * e.placement.stack_pixel_angstrom for e in exported.values()
        )
        written = {
            "hand": args.hand,
            "dropped_angstrom_max": dropped_max,
            "series": [
                {
                    "name": name,
                    "notes": list(e.notes),
                    "views": len(e.views),
                    "stack_pixel_angstrom": e.placement.stack_pixel_angstrom,
                }
                for name, e in exported.items()
            ],
        }
        selection: dict[str, Any] = {
            "particles": [
                {
                    "tag": tag,
                    "relion_name": names[tag],
                    "half": plan.halves[tag],
                    "cc": member_of[tag].cc,
                }
                for tag in plan.kept
            ],
            "duplicates_left_out": [
                {"tag": tag, "duplicates": other} for tag, other in plan.duplicates
            ],
        }
        if isinstance(chosen, ParticleSet):
            # Where each particle came from: its picks or stack, and its row there.
            rows: dict[int, int] = {}
            seen: Counter[str] = Counter()
            for member in chosen.members:
                rows[member.tag] = seen[member.stack_id]
                seen[member.stack_id] += 1
            for entry in selection["particles"]:
                entry["source"] = member_of[entry["tag"]].stack_id
                entry["row"] = rows[entry["tag"]]
            readme = particles_readme_text(plan, written)
        else:
            selection["decoys_left_out"] = chosen.decoys_left_out
            selection["not_agreed_left_out"] = chosen.not_agreed
            readme = readme_text(plan, written, diameter, ini_high)
        (folder / "selection.json").write_text(json.dumps(selection, indent=1) + "\n")
        attempt.add(ArtifactKind.DATASET_EXPORT, "selection.json")
        (folder / "README.md").write_text(readme)
        attempt.add(ArtifactKind.DATASET_EXPORT, "README.md")
        attempt.metrics.update(
            {
                "particles": float(len(plan.kept)),
                "duplicates_left_out": float(len(plan.duplicates)),
                "decoys_left_out": float(
                    0 if isinstance(chosen, ParticleSet) else chosen.decoys_left_out
                ),
            }
        )
    return written


def particle_diameter(run: Path, box: int, pixel: float) -> float:
    """The particle's diameter in A for Refine3D: the MRA setup's, from the run's
    plan.json; a Dynamo project, or a port run in Dynamo's layout, has none, and then
    half the box, as the box is twice the particle (docs/agent/lane_questions.md)."""

    plan = run / "plan.json"
    found = _find(json.loads(plan.read_text()), "diameter_angstrom") if plan.is_file() else None
    return float(found) if found else box * pixel / 2


def _find(value: object, key: str) -> object:
    """The first value of `key` anywhere in a JSON document."""

    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find(item, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find(item, key)
            if found is not None:
                return found
    return None


def particles_average(plan: ExportPlan, mask: np.ndarray) -> tuple[np.ndarray, int]:
    """The pipeline's own average of the exported picks or stacks, in its own frame.

    Each box is cut where extraction cuts it and averaged at the particle's pose by
    the MRA's `average_class`, with its series' wedge as the MRA's table gives it
    (`dynamo_ytilt`). A box leaving the tomogram is left out; their count is returned.
    """

    import zarr

    from cryoet_pipeline.backends.dynamo_search import average_class, dynamo_ytilt
    from cryoet_pipeline.tomogram_export import tilt_angles_of

    chosen = plan.chosen
    box = chosen.box
    registry = ArtifactRegistry.load(plan.project / "artifacts.json")
    member_of = {member.tag: member for member in chosen.members}
    boxes, angles, shifts, wedges = [], [], [], []
    left_out = 0
    for series in plan.series:
        tomogram = series.sources.tomogram
        volume = zarr.open_array(str(tomogram.path), mode="r")
        found = tilt_angles_of(registry, tomogram, series.name)
        tilts = (
            [float(word) for word in found.read_text().split()]
            if found is not None
            else series.sources.manifest.tilt_angles_deg
        )
        wedge = dynamo_ytilt((min(tilts), max(tilts)))
        depth, height, width = (int(size) for size in volume.shape)
        for tag in plan.kept:
            member = member_of[tag]
            if member.series != series.name:
                continue
            x, y, z = box_start(member.pick_xyz, box)
            if min(x, y, z) < 0 or z + box > depth or y + box > height or x + box > width:
                left_out += 1
                continue
            boxes.append(np.asarray(volume[z : z + box, y : y + box, x : x + box], dtype=float))
            angles.append(member.angles_deg)
            shifts.append(member.shifts_px)
            wedges.append(wedge)
    if not boxes:
        raise ValueError("no exported particle's box lies inside its tomogram")
    average = average_class(boxes, angles, shifts, wedges, mask, workers=os.cpu_count() or 1)
    return np.asarray(average.raw, dtype=np.float64), left_out


def relion_check(plan: ExportPlan, relion: str, threads: int) -> bool:
    """RELION reconstructs the exported particles at the MRA's pixel and their poses.
    For a class it must be the exported reference; for picks or stacks, the
    pipeline's own average of the same boxes at the same poses (`particles_average`).
    The comparison is scripts/relion_particle_check.py's."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from relion_particle_check import _sphere, compare

    box = plan.chosen.box
    folder = plan.out
    checks = folder / "checks"
    checks.mkdir(exist_ok=True)
    command = [
        relion,
        "--i",
        "optimisation_set.star",
        "--o",
        "checks/rec_class/",
        "--b",
        str(box),
        "--bin",
        str(plan.series[0].relion_bin),
        "--no_ctf",
        "--j",
        str(threads),
    ]
    result = subprocess.run(command, cwd=folder, capture_output=True, text=True)
    (checks / "relion_reconstruct_particle.log").write_text(
        " ".join(command) + "\n\n" + result.stdout + result.stderr
    )
    target = checks / "rec_class" / "data_merged.mrc"
    if result.returncode != 0 or not target.is_file():
        sys.exit(
            f"RELION did not reconstruct the class (exit {result.returncode}); "
            f"see {checks / 'relion_reconstruct_particle.log'}"
        )
    relion_map = -mrcfile.read(target).astype(np.float64)
    mask = _sphere(box, box / 2 - 2)
    if isinstance(plan.chosen, ParticleSet):
        ours, left_out = particles_average(plan, mask)
        reference = standard_frame_reference(ours)
        _write_mrc(checks / "pipeline_average_relion_frame.mrc", reference,
                   plan.chosen.pixel_angstrom)
        against = "the pipeline's average of the same boxes at the same poses"
        if left_out:
            against += f" ({left_out} boxes leaving their tomogram left out of it)"
    else:
        reference = mrcfile.read(
            folder / "reference" / f"class_{plan.chosen.channel}.mrc"
        ).astype(np.float64)
        against = "the exported reference"
    outcome = compare(reference, relion_map, mask)
    # For information: RELION's weighted map (merged.mrc). Its division by the Fourier
    # weights is noisy for few particles, so the verdict stays on the backprojected data.
    merged = checks / "rec_class" / "merged.mrc"
    if merged.is_file():
        weighted = compare(reference, -mrcfile.read(merged).astype(np.float64), mask)
        outcome["merged_ncc_as_written"] = weighted["ncc"]["z0 y0 x0"]
    (checks / "class_check.json").write_text(json.dumps(outcome, indent=2) + "\n")
    ranked = sorted(outcome["ncc"].items(), key=lambda item: item[1], reverse=True)
    shift = outcome["shift_zyx_voxel"]
    print(
        f"RELION's reconstruction against {against}: best "
        f"{ranked[0][0]} {ranked[0][1]:+.3f}, next {ranked[1][0]} {ranked[1][1]:+.3f}; "
        f"offset (z, y, x) {shift[0]:+.2f} {shift[1]:+.2f} {shift[2]:+.2f}: "
        + ("pass" if outcome["passed"] else "FAIL: " + "; ".join(outcome["reasons"]))
    )
    return bool(outcome["passed"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, help="<project>/mra/<NAME>, with --class")
    parser.add_argument("--class", dest="channel", type=int)
    parser.add_argument(
        "--particles",
        nargs="+",
        metavar="ARTIFACT_ID",
        help="instead of a class: picks or particle stacks of --project, one per series, "
        "by their ids in the registry",
    )
    parser.add_argument(
        "--box",
        type=int,
        help="with --particles: the box RELION extracts, even (default: the stacks' own; "
        "picks have none)",
    )
    parser.add_argument("--iteration", type=int, help="default: the last finished")
    parser.add_argument(
        "--agree", type=_class_spec, help="RUN:CLASS: keep only particles also in this run's class"
    )
    parser.add_argument("--project", type=Path, help="default: the run's own")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--stack-binning",
        type=int,
        help=f"default: the least that makes the stack's pixel {MIN_STACK_PIXEL} A or more",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        help="voxels; duplicates are closer (default: a quarter of the box)",
    )
    parser.add_argument("--seed", type=int, default=0, help="for the halves")
    parser.add_argument("--hand", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--dose-per-tilt", type=float, help="e/A^2, when the mdoc has none")
    parser.add_argument(
        "--ini-high",
        type=float,
        default=40.0,
        help="A: Refine3D's starting low-pass of the reference",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--reason",
        help="what was asked and by whom, in English, for the project's journal; required "
        "from the assistant",
    )
    parser.add_argument("--relion-check", help="relion_tomo_reconstruct_particle (RELION 5)")
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    )
    args = parser.parse_args(argv)
    if args.particles:
        if args.run or args.channel is not None or args.agree or args.iteration is not None:
            parser.error("--particles exports picks or stacks, not a class: leave out "
                         "--run, --class, --iteration and --agree")
        if args.project is None:
            parser.error("--particles needs --project, the project they are in")
    else:
        if args.run is None or args.channel is None:
            parser.error("give --run and --class, or --particles with --project")
        if args.box is not None:
            parser.error("--box is for --particles; a class has the MRA's box")
    try:
        plan = make_plan(args)
    except (KeyError, ValueError) as error:
        print(f"cannot export: {error}", file=sys.stderr)
        return 2
    for line in describe(plan):
        print(line)
    if plan.out.exists():
        print(f"{plan.out} is there already; an export is never written over", file=sys.stderr)
        return 2
    record = plan_record(plan, args)
    who = submit_lanes._who()
    project = str(plan.project)
    if not args.dry_run:
        why = refusal(
            record, who, args.reason, submit_lanes.read_journal(project), datetime.now(UTC)
        )
        if why:
            print(f"refused: {why}", file=sys.stderr)
            return 2
    # Journaled before anything is written, as a submission is before its sbatch: one
    # dry run stands for one export.
    entry: dict[str, Any] = {
        **who,
        "at": submit_lanes._now(),
        "recipe": JOURNAL_ACTION,
        "config": record["config"],
        "argv": [sys.argv[0], *(sys.argv[1:] if argv is None else argv)],
        "plan": record,
        "plan_sha256": submit_lanes._digest(record),
        "reason": args.reason,
    }
    if args.dry_run:
        entry |= {"event": "dry-run", "problems": []}
    else:
        entry |= {
            "event": "submit",
            "id": uuid.uuid4().hex[:12],
            "code": submit_lanes.code_version(),
        }
    try:
        journal = submit_lanes.write_journal(project, entry)
    except OSError as error:
        if not args.dry_run:
            print(f"refused: cannot write the project's journal: {error}", file=sys.stderr)
            return 2
        print(f"not journaled: {error}", file=sys.stderr)
    else:
        print(f"recorded in {journal}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    written = write(plan, args)
    for series in written["series"]:
        for note in series["notes"]:
            print(f"note, {series['name']}: {note}")
    print(f"wrote {plan.out}")
    hint = (
        particles_submit_hint(plan.out)
        if isinstance(plan.chosen, ParticleSet)
        else submit_hint(plan.out)
    )
    print(f"next, RELION in a job (README.md): {hint}")
    if args.relion_check:
        return 0 if relion_check(plan, args.relion_check, args.threads) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
