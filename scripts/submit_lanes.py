#!/usr/bin/env python3
"""Submit a lane as chained Slurm jobs, from the [run] section of its config.

Run from the repo root on the submit node:

    python3 scripts/submit_lanes.py configs/lanes/ribo-bin8.toml
    python3 scripts/submit_lanes.py configs/lanes/ribo-bin8.toml configs/lanes/gag-bin8.toml
    python3 scripts/submit_lanes.py configs/lanes/gag-bin8.toml --from-stage gpu
    python3 scripts/submit_lanes.py configs/lanes/gag-bin8.toml --from-stage gpu --overwrite
    python3 scripts/submit_lanes.py configs/lanes/gag-bin8.toml --dry-run

[run] names the project, its series, and how the chain in scripts/run_series.py
is cut into jobs. Each stage is one scripts/lane.sbatch job over every series,
steps `from`..`until`, with its own sbatch resources and time limit; stages are
joined by afterok, so one starts only when the previous finished without a
failed series. The cut is not the order: stages must be consecutive slices of
the fixed chain, and a gap or an overlap is refused here, before anything is
submitted. The step order stays in code (see scripts/run_series.py).

A stage whose `from` and `until` are both `mra` runs the MRA (docs/mra_in_lanes.md):
one job of scripts/mra_port_run.sbatch over all the lane's series at once, on the
particles the stage before it extracted, with the lane's [mra] over the defaults.
--until-stage stops a submission before it, to look at the picks first, and
--from-stage mra submits it later on its own. The dry run shows the MRA's run and
every setting with its source. An MRA job reads its series rather than writing
them: another MRA of other settings may run beside it, a lane's steps may not.

Two lanes given together run side by side; every config is checked before any
job is submitted. The environment goes to the jobs with --export=ALL, so set
what the config's `$VAR` values name (PYTOMO_MP_DIR, MOTIONCOR3, MEMBRAIN_EXE,
...) before running; a missing one is refused here rather than in the job.
OVERWRITE is the exception. Steps replace what they wrote before only with
--overwrite, which lane.sbatch receives as OVERWRITE=1 for this submission
alone. OVERWRITE set in the environment is refused: it would apply to every
submission from that shell, including the ones nobody meant to overwrite with.

Every dry run and every submission is appended to the project's journal.jsonl:
the plan, who ran it and through which client, and for a submission also the
config's text, how it differs from the lane file it was copied from, the commit,
--reason and the job IDs. The one exception is a dry run that would refuse, for a
project whose folder does not exist yet: it makes no folder for its entry, says
so, and leaves nothing behind. A submission from the assistant is refused unless it
gives --reason and the config's latest journal entry is the same person's dry
run of exactly this plan, without problems, at most 30 minutes old: what is
submitted is what was shown, once. The assistant is known by CRYOET_ASSISTANT,
which its launcher sets, or by the AI_AGENT or CLAUDECODE Claude Code sets, or
the CODEX_THREAD_ID Codex sets (see docs/agent_architecture.md, "First
increment").

Two jobs on one series write the same files, so a series that a queued or
running job of the same project works on is refused, in a dry run as in a
submission. squeue lists the jobs. The journal says which series each of them
works on, or else the job's log does, once the job runs. A job named like the
project's own jobs that neither accounts for may be on any of its series. Only
the user's own jobs are asked about, since a project lives in its user's clone.

Slurm itself is asked about every stage before anything is submitted, with
`sbatch --test-only` and the stage's own options: the partition, the reservation,
the GPU type, the cores, memory and time a node gives are the site's to judge,
not rules kept here. A stage it refuses is refused, in a dry run as in a
submission, so no stage is queued without the ones after it. A dry run prints
what Slurm said of each stage; with no sbatch, it says the check was not made.

For a new frames project, the dry run also reads the frames and mdocs as
`cryoet init` would (cryoet_pipeline.project.read_inputs), writes nothing, and
refuses what init would refuse: a pattern that matches nothing, a series not
among the movies, a movie it cannot read. Otherwise it prints a line per series
init would find. A stage that runs motion correction needs the file its
[motion] gain_reference names.

The dry run and the submission list every key the config sets that the lab's
rules leave unset, with the rule, as docs/agent/lane_questions.md's table of
every key says it ("rule" its only source, "unset" its rule): a value copied
with a whole lane is that lane's choice, and changes results without anyone
having asked for it. Listed, not refused.

The recipes, experiments written in advance that run a lane's steps another way
(scripts/ctf_window_test.sbatch and the like), are submitted by
scripts/submit_recipe.py, with these checks and into the same journal.

[run] series may be left out, to take every series: a frames lane then gets the
ones `init` finds, or, for a project that exists, the ones it has manifests for;
a stack lane gets every series its [data] stack path matches, with {series} as
the wildcard, and a tomograms or picks lane every series its [data] tomogram path
matches; a particles lane gets the series of [data] particles_series.

A lane of tomograms, picks or particles made elsewhere starts with their import
(docs/entry_and_exit.md). Its files and the answers in its [data] that are not
on the imports' lists are checked here, before the queue; what the files hold is
checked by the imports themselves, in the job.

For a frames lane whose project does not exist yet, `cryoet init` runs here
first, from [data] frames and either mdocs or frames_pattern (tilt metadata
read from the movie file names). A lane naming series that an existing project
has no manifests for is refused, with the `cryoet init ... --add-series` that
adds them and keeps the project's registry; adding series is left to the person,
and is not done here. Not done here either: downloading data and caching EMDB
maps — see the README.

Disposable, like scripts/run_series.py.
"""

from __future__ import annotations

import argparse
import difflib
import getpass
import glob
import hashlib
import json
import os
import re
import select
import shlex
import shutil
import socket
import subprocess
import sys
import time
import tomllib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_project  # noqa: E402

LANE_SBATCH = ROOT / "scripts" / "lane.sbatch"
# A stage that runs the step mra is one job of this script over all the lane's
# series (docs/mra_in_lanes.md).
MRA_SBATCH = ROOT / "scripts" / "mra_port_run.sbatch"
# What scripts/mra_port_run.sbatch reads from the environment. A lane's MRA job gets
# CONFIG, SOURCE, SERIES, NAME and CONFIG_SHA256 from here and none of the rest from
# the shell: its settings are the lane's [mra], and a SEEDS or a PROJECT left in a
# shell by a run typed by hand would change what it runs, or stop it.
MRA_VARIABLES = (
    "CONFIG", "CONFIG_SHA256", "SOURCE", "SERIES", "NAME", "LANE", "DECOY_LANE", "EXPORT",
    "PROJECT", "SPLIT", "HALF", "SEEDS", "REFERENCE", "SCHEDULE", "FIRST_LOWPASS",
    "LAST_LOWPASS", "DIAMETER", "LOWPASS_STEPS", "MASK_RADIUS", "MASK_EDGE", "MASK_FILE",
    "GLOBAL_ROUNDS", "GLOBAL_STEP", "GLOBAL_FULL_BOX", "CONE_STEPS", "SHIFT_LIMIT",
    "ANTIALIAS", "DATA_FOLDER", "GPUS", "SYMMETRY", "AXIS_OFFSET", "TEMPLATE", "DEVICE",
    "DTYPE", "PER_GPU", "MRA_PORT_RUN_LOCK", "OVERWRITE",
)
_MRA_SET = ("CONFIG", "CONFIG_SHA256", "SOURCE", "SERIES", "NAME")
_ENV_REFERENCE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
# Each project's record of what was shown and what was submitted: one JSON object
# per line, appended to and never rewritten.
JOURNAL = "journal.jsonl"
# How long a dry run stands for the plan it showed. Past it the queue and the
# outputs may have moved on, and so may what the person remembers agreeing to.
DRY_RUN_FRESH = timedelta(minutes=30)
# Any of these, set and not empty, means an agentic client runs this.
# CRYOET_ASSISTANT is set by the assistant's launcher. Claude Code sets AI_AGENT
# and CLAUDECODE in every shell it starts, and some other clients set AI_AGENT
# too. Codex (0.167.2) sets CODEX_THREAD_ID in every command it runs, its
# session's id.
ASSISTANT_MARKERS = ("CRYOET_ASSISTANT", "AI_AGENT", "CLAUDECODE", "CODEX_THREAD_ID")
# Copied into the journal as they are, when set: the client, the model (the
# launcher's CRYOET_ASSISTANT_MODEL), and the session whose transcript holds the
# conversation (Claude Code's CLAUDE_CODE_SESSION_ID, Codex's CODEX_THREAD_ID).
# Nothing else from the environment is, apart from a plan's $VARs.
CLIENT_VARIABLES = (
    *ASSISTANT_MARKERS, "CRYOET_ASSISTANT_MODEL", "ANTHROPIC_MODEL", "CLAUDE_CODE_SESSION_ID",
)
# What a job's log says it works on, in the header lane.sbatch writes and the
# recipes copy: "project P ..." with "series A B", or "series A B in P".
_HEADER_PROJECT = re.compile(r"^project\s+(?P<project>[^\s,]+)")
_HEADER_SERIES_IN = re.compile(r"^series\s+(?P<series>.+?)\s+in\s+(?P<project>\S+)\s*$")
_HEADER_SERIES = re.compile(r"^series\s+(?P<series>.+?)\s*$")
# The job number sbatch --test-only names: Slurm draws one for every test
# (628553 to 628555 on cryoem.se, 24 November 2026), but no job holds it.
_TEST_JOB = re.compile(r"^(?:sbatch: )?Job \d+ ")
# The version line of CTFFIND's start-up banner, as the backend reads it
# (cryoet_pipeline.backends.ctf.ctffind_version): anchored, so that CTFFIND 5's
# "Library Version" line is not taken for it.
_CTFFIND_VERSION = re.compile(r"^\s*Version\s*:\s*(?P<version>\d+(?:\.\d+)*)", re.MULTILINE)
# How long CTFFIND may take to print its banner when asked here.
CTFFIND_PROBE_SECONDS = 5.0
# How long one `sbatch --test-only` may take; a busy controller answers in seconds.
SBATCH_TEST_SECONDS = 60.0


@dataclass(frozen=True)
class Stage:
    name: str
    start: str
    end: str
    time: str
    sbatch: tuple[str, ...]
    env: frozenset[str]  # the $VARs the stage's commands name

    @property
    def is_mra(self) -> bool:
        """Runs the MRA, a job of MRA_SBATCH, rather than steps of lane.sbatch."""
        return self.start == run_series.MRA_STEP


@dataclass(frozen=True)
class Plan:
    config: Path
    project: str
    series: tuple[str, ...]
    stages: tuple[Stage, ...]
    # The lane's MRA (cryoet_pipeline.mra_lane.LaneMra), when a stage runs it.
    mra: Any = None


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"{where} must be a list of strings")
    return tuple(value)


def plan(cfg: dict[str, Any], config: Path) -> Plan:
    """The lane's stages, checked against its chain. Refuses rather than guesses."""

    # Before anything is read from it: a misspelt [run] series would otherwise be
    # taken for no list at all, which means every series.
    run_series.check_keys(cfg, config)
    run = cfg.get("run")
    if not isinstance(run, dict):
        raise SystemExit(f"{config}: no [run] section")
    project = run.get("project")
    if not isinstance(project, str) or not project:
        raise SystemExit(f"{config}: [run] project must be a path")
    if run.get("series") is None:
        # Every series. A stack, tomograms or picks lane finds them now, in its file
        # paths, and a particles lane in [data] particles_series; a frames lane only
        # once init has run (see project_series).
        series = every_series(cfg, config)
    else:
        series = _string_list(run.get("series"), f"{config}: [run] series")
        if not series:
            raise SystemExit(f"{config}: [run] series is empty; leave it out for every series")
    if len(set(series)) != len(series):
        raise SystemExit(f"{config}: [run] series lists a series twice")
    raw_stages = run.get("stage")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise SystemExit(f"{config}: [run] has no [[run.stage]]")

    # The chain is the same for every series, and a frames lane may not know its
    # series yet: any id renders it. The MRA follows it, over all the series at once.
    example = series[0] if series else "TS_X"
    chain = [name for name, _ in run_series.build_steps(cfg, Path(project), example)]
    chain.append(run_series.MRA_STEP)
    mra_at = len(chain) - 1
    stages: list[Stage] = []
    for raw in raw_stages:
        where = f"{config}: [[run.stage]] {raw.get('name', '?')!r}"
        missing = [key for key in ("name", "from", "until", "time", "sbatch") if key not in raw]
        if missing:
            raise SystemExit(f"{where}: missing {', '.join(missing)}")
        for key in ("from", "until"):
            if raw[key] not in chain:
                raise SystemExit(
                    f"{where}: unknown step {raw[key]!r} in {key}\n"
                    f"known steps: {', '.join(chain)}"
                )
        lo, hi = chain.index(raw["from"]), chain.index(raw["until"])
        if lo > hi:
            raise SystemExit(f"{where}: {raw['from']!r} comes after {raw['until']!r}")
        if stages:
            expected = chain.index(stages[-1].end) + 1
            if lo != expected:
                gap = "skips" if lo > expected else "repeats"
                between = ", ".join(chain[min(lo, expected) : max(lo, expected)])
                raise SystemExit(
                    f"{where}: starts at {raw['from']!r}, but the previous stage ends at "
                    f"{stages[-1].end!r}, so it {gap} {between}"
                )
        if raw["name"] in {stage.name for stage in stages}:
            raise SystemExit(f"{where}: two stages share the name")
        if lo <= mra_at <= hi and lo != hi:
            raise SystemExit(
                f"{where}: runs {run_series.MRA_STEP} together with other steps; the MRA "
                f"classifies all the series at once, in a stage of its own (from = until = "
                f"{run_series.MRA_STEP!r}) after the one that ends at {chain[mra_at - 1]!r}"
            )

        if lo == mra_at:
            env: frozenset[str] = frozenset()
        else:
            # Every series renders the same $VARs; the paths differ, the variables do not.
            sliced = run_series.slice_steps(
                run_series.build_steps(cfg, Path(project), example), raw["from"], raw["until"]
            )
            env = frozenset(
                match
                for _, command in sliced
                for part in command
                for match in _ENV_REFERENCE.findall(part)
            )
        stages.append(
            Stage(
                name=raw["name"],
                start=raw["from"],
                end=raw["until"],
                time=str(raw["time"]),
                sbatch=_string_list(raw["sbatch"], f"{where}: sbatch"),
                env=env,
            )
        )

    mra = None
    if any(stage.is_mra for stage in stages):
        try:
            from cryoet_pipeline.mra_lane import LaneMraError, lane_mra
        except ImportError as error:
            raise SystemExit(
                f"{config}: the MRA stage is checked with the cryoet-pipeline package, which "
                f"this python cannot import ({error}); activate its environment first "
                "(source scripts/activate_env.sh)"
            ) from None
        try:
            mra = lane_mra(cfg)
        except LaneMraError as error:
            raise SystemExit(f"{config}: {error}") from None
    elif "mra" in cfg:
        raise SystemExit(
            f"{config}: [mra] is set, but no [[run.stage]] runs {run_series.MRA_STEP}, so it "
            f"would be ignored; add a stage with from = until = {run_series.MRA_STEP!r} after "
            "the one that ends at extract-particles, or delete [mra]"
        )
    return Plan(config=config, project=project, series=series, stages=tuple(stages), mra=mra)


def every_series(cfg: dict[str, Any], config: Path) -> tuple[str, ...]:
    """The series of a lane without [run] series, where its config can say them:
    empty for a frames lane, whose series init finds."""

    source = run_series.lane_input(cfg)
    if source == "stack":
        return stack_series(cfg, config)
    if source in ("tomograms", "picks"):
        return stack_series(cfg, config, key="tomogram")
    if source == "particles":
        return tuple(sorted(run_series.particles_series(cfg)))
    return ()


def stack_series(cfg: dict[str, Any], config: Path, key: str = "stack") -> tuple[str, ...]:
    """Every series a lane's [data] path matches, {series} its wildcard: the stacks of
    a stack lane, or the tomograms of a tomograms or picks lane (key "tomogram")."""

    template = cfg.get("data", {}).get(key)
    if not isinstance(template, str) or template.count("{series}") != 1:
        raise SystemExit(
            f"{config}: without [run] series, [data] {key} must contain {{series}} once, "
            "so that the series can be found"
        )
    prefix, suffix = os.path.join(ROOT, template).split("{series}")
    found = sorted(
        path[len(prefix) : len(path) - len(suffix)]
        for path in glob.glob(glob.escape(prefix) + "*" + glob.escape(suffix))
    )
    if not found:
        raise SystemExit(f"{config}: no file matches [data] {key} {template!r}")
    return tuple(found)


def movie_formats(cfg: dict[str, Any]) -> frozenset[str]:
    """The formats of the movies in a frames lane's [data] frames folder.

    A file's format is the pipeline's own reading of its suffix
    (cryoet_pipeline.movies.movie_format), so a file it does not take for a movie,
    such as an mdoc or a `.gain`, does not count. Empty for a stack lane, and when
    the folder cannot be listed; for a new project, problems() says so.
    """
    folder = cfg.get("data", {}).get("frames")
    if run_series.lane_input(cfg) != "frames" or not isinstance(folder, str):
        return frozenset()
    try:
        paths = list((ROOT / folder).iterdir())
    except OSError:
        return frozenset()
    # Here, not at the top: only a frames lane needs the package and its numpy.
    from cryoet_pipeline.movies import movie_format

    formats = set()
    for path in paths:
        try:
            formats.add(movie_format(path).value)
        except ValueError:
            continue
    return frozenset(formats)


def project_series(project: str) -> tuple[str, ...]:
    """The series a project has manifests for: what init wrote, or import-tilt-stack."""

    return tuple(sorted(path.stem for path in (ROOT / project / "manifests").glob("*.json")))


# Printed once under the refusals when a $VAR is not set. On max, a first dry run
# (2026-11-12) refused five of them, and "is not set" was read as "not installed".
UNSET_HINT = (
    "the $ variables above say where the tools are, and the jobs take them from this "
    "shell: set them in ~/.bashrc, then open a new terminal (README, \"0. One-time "
    "setup\"; at MDC, the lines that source "
    "/fast/AG_Kudryashev/software/cryoet-tools.sh and set PYTOMO_MP_DIR)"
)


# The table of every lane key, with where each value comes from (tests/test_lane_questions.py
# reads the same rows). A row whose only source is the lab's rules and whose rule begins
# "unset" names keys a lane leaves out unless someone asked otherwise.
LANE_QUESTIONS = ROOT / "docs" / "agent" / "lane_questions.md"
_KEY_ROW = re.compile(
    r"^\| `\[(?P<section>[a-z_.]+)\] (?P<keys>[a-z0-9_, ]+)` \| (?P<sources>[a-z, ]+) \| "
    r"(?P<how>.+) \|$"
)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def set_against_rules(cfg: dict[str, Any]) -> list[str]:
    """The keys the config sets that the lab's rules leave unset, each with its rule.

    A lane copied whole from another carries that lane's choices with it: the
    assistant's ryr1-assistant lane on cryoem.se, 26 November 2026, took
    number_of_particles = 2000 and number_of_false_positives = 100 from ryr1-bin8
    and picked 755 on series 125 where the rule's twice-the-peaks gave 632. Listed,
    not refused: a person may have asked for any of them."""

    try:
        text = LANE_QUESTIONS.read_text()
    except OSError as error:
        return [f"not compared with the lab's rules: {error}"]
    found = []
    for line in text.splitlines():
        row = _KEY_ROW.match(line)
        if not row or {s.strip() for s in row["sources"].split(",")} != {"rule"}:
            continue
        if not row["how"].lower().startswith("unset"):
            continue
        table = cfg.get(row["section"])
        if not isinstance(table, dict):
            continue
        for key in (k.strip() for k in row["keys"].split(",")):
            if key in table:
                how = _LINK.sub(r"\1", row["how"])
                found.append(f"[{row['section']}] {key} = {json.dumps(table[key])}; "
                             f"the rule: {how}")
    return found


def unset_variables(stages: tuple[Stage, ...]) -> list[tuple[str, Stage]]:
    """The $VARs the stages' commands name that this shell leaves empty."""

    return [
        (name, stage)
        for stage in stages
        for name in sorted(stage.env)
        if not os.environ.get(name)
    ]


def problems(cfg: dict[str, Any], lane: Plan, stages: tuple[Stage, ...]) -> list[str]:
    """What would make the submitted jobs fail at once, found on the submit node."""

    found = [
        f"${name} is not set (used by stage {stage.name!r})"
        for name, stage in unset_variables(stages)
    ]
    found += mra_problems(lane, stages)
    # The MRA is no step of the chain each series runs: its checks are mra_problems'.
    stages = tuple(stage for stage in stages if not stage.is_mra)
    chain = run_series.build_steps(
        cfg, Path(lane.project), lane.series[0] if lane.series else "TS_X"
    )
    steps = {
        name for stage in stages
        for name, _ in run_series.slice_steps(chain, stage.start, stage.end)
    }
    found += ctffind_problems(cfg, steps)
    # Without its gain reference MotionCor3 fails after the queue wait.
    gain = cfg.get("motion", {}).get("gain_reference")
    if "prepare-tilt-series" in steps and isinstance(gain, str):
        path = os.path.expandvars(gain)
        if "$" not in path and not (ROOT / path).is_file():
            found.append(f"[motion] gain_reference: no file {ROOT / path}")
    if cfg.get("fine_align", {}).get("backend") == "aretomo":
        # AreTomo2 runs only on a GPU; a CPU node fails it after the queue wait.
        for stage in stages:
            steps = {name for name, _ in run_series.slice_steps(chain, stage.start, stage.end)}
            asks_gpu = any(
                arg.startswith(("--gres=gpu", "--gpus", "-G")) for arg in stage.sbatch
            )
            if "fine-align-tilt-series" in steps and not asks_gpu:
                found.append(
                    f"stage {stage.name!r} runs fine-align-tilt-series with AreTomo2, "
                    "which needs a GPU; give that stage --partition=gpu and --gres=gpu:..."
                )
    if run_series.lane_input(cfg) in run_series.IMPORTED_INPUTS:
        return found + import_problems(cfg, lane)
    if run_series.lane_input(cfg) != "frames":
        # A stack lane writes its own manifests in import-tilt-stack; only its
        # inputs can be checked here, which saves a queue wait on a typo.
        data = cfg.get("data", {})
        for series in lane.series:
            for key in ("mdoc", "stack"):
                if isinstance(data.get(key), str):
                    path = ROOT / data[key].format(series=series)
                    if not path.is_file():
                        found.append(f"[data] {key}: no file {path}")
        return found
    if init_command(cfg, lane) is not None:
        return found + init_data_problems(cfg)
    if not lane.series and not project_series(lane.project):
        found.append(f"{lane.project} exists but has no manifests to take every series from")
    manifests = ROOT / lane.project / "manifests"
    absent = [s for s in lane.series if not (manifests / f"{s}.json").is_file()]
    if absent:
        # Plain init refuses a project with a registry; --add-series writes the
        # missing manifests and keeps the registry. It is the person's to run.
        add = [*_init_arguments(cfg, lane.project, absent), "--add-series"]
        found.append(
            f"{lane.project} already exists but has no manifest for {', '.join(absent)}; "
            f"add them to it, keeping what it holds, with: {shlex.join(add)}"
        )
    return found


def init_data_problems(cfg: dict[str, Any]) -> list[str]:
    """The [data] keys `cryoet init` needs for a frames lane's new project."""

    data = cfg.get("data", {})
    found = []
    if ("mdocs" in data) == ("frames_pattern" in data):
        found.append(
            "[data] needs exactly one of mdocs and frames_pattern; init reads "
            "the tilt metadata from one of them"
        )
    for key in ("frames", "mdocs"):
        if key == "mdocs" and "mdocs" not in data:
            continue
        if not isinstance(data.get(key), str):
            found.append(f"[data] {key} is not set; init needs it for a new project")
        elif not (ROOT / data[key]).is_dir():
            found.append(f"[data] {key}: no directory {ROOT / data[key]}")
    return found


def init_preview(cfg: dict[str, Any], lane: Plan) -> tuple[list[str], list[str]]:
    """What `cryoet init` would find for a frames lane's new project: the problems,
    and a line per series it would write a manifest for.

    Read by init's own code, cryoet_pipeline.project.read_inputs, with nothing
    written, so a dry run refuses what init would refuse and shows the series,
    tilts, angles and pixel size init would read. Until 24 November 2026 a dry run
    checked only that the frames folder exists, and a pattern that matched nothing,
    a series not among the movies or a missing movie came out at the submission,
    after the yes. Only for a dry run: a submission runs init itself, before any job.
    """

    # Here, not at the top: only a frames lane needs the package.
    from cryoet_pipeline.models import ProjectConfig
    from cryoet_pipeline.project import describe_manifest, read_inputs

    data = cfg.get("data", {})
    try:
        inputs = read_inputs(ProjectConfig(
            frames_dir=ROOT / data["frames"],
            mdocs_dir=ROOT / data["mdocs"] if "mdocs" in data else None,
            frames_pattern=data.get("frames_pattern"),
            raw_pixel_spacing_angstrom=data.get("pixel_size_angstrom"),
            output_dir=ROOT / lane.project,
            tilt_series=list(lane.series),
        ))
    except Exception as error:  # whatever it is, init would stop on it the same way
        said = " ".join(str(error).split())
        return [f"cryoet init would refuse: {type(error).__name__}: {said}"], []
    return [], [f"{one.tilt_series_id}: {describe_manifest(one)}" for one in inputs.manifests]


def slurm_check(
    lane: Plan, stages: tuple[Stage, ...],
) -> tuple[list[str], list[str]] | None:
    """Each stage's job as Slurm judges it, asked with `sbatch --test-only`, which
    submits nothing: the problems, and what Slurm said of each stage it would take.
    None where there is no sbatch to ask.

    The command is the submission's own, without the dependency on the stage
    before, which has no job yet. So the partition, the reservation, the GPU type,
    the cores and memory a node has, the time limits and the batch script's own
    #SBATCH lines are judged by the site's Slurm, not by rules kept here. Until
    24 November 2026 nothing asked it: a lane carried from MDC to Stockholm
    (--partition=gpu --gres=gpu:a40:1) passed its dry run and would have been
    refused only at the submission, after the yes, with the stages before the
    refused one already queued.
    """

    prefix = Path(lane.project).name
    found, said_of = [], []
    for stage in stages:
        script = MRA_SBATCH if stage.is_mra else LANE_SBATCH
        command = [
            "sbatch", "--test-only", *stage.sbatch, f"--time={stage.time}",
            f"--job-name={prefix}-{stage.name}", str(script.relative_to(ROOT)),
        ]
        try:
            result = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                timeout=SBATCH_TEST_SECONDS,
            )
        except FileNotFoundError:
            return None
        except subprocess.TimeoutExpired:
            found.append(
                f"sbatch --test-only gave no answer in {SBATCH_TEST_SECONDS:g} s, so "
                f"whether Slurm takes stage {stage.name!r} is unknown"
            )
            continue
        lines = [line.strip() for line in (result.stderr + result.stdout).splitlines()]
        said = " ".join(line for line in lines if line) or f"exit {result.returncode}"
        if result.returncode != 0:
            asked = shlex.join([*stage.sbatch, f"--time={stage.time}"])
            found.append(f"Slurm refuses stage {stage.name!r} ({asked}): {said}")
        else:
            said_of.append(f"{stage.name}: {_TEST_JOB.sub('', said, count=1)}")
    return found, said_of


def import_problems(cfg: dict[str, Any], lane: Plan) -> list[str]:
    """What the imports of a tomograms, picks or particles lane would refuse at once,
    found before the queue: their files, the answers that are not on the imports'
    lists, and one file named for several series where its format holds one. The
    checks the imports make on what the files hold, the wedge and whether the picks
    land on particles, are theirs, made in the job."""

    # Here, not at the top: only these lanes need the package.
    from cryoet_pipeline.picks_import import check_format
    from cryoet_pipeline.tomogram_import import ImportRefused, orientation, parse_processing

    source = run_series.lane_input(cfg)
    data = cfg.get("data", {})
    series = lane.series
    found: list[str] = []

    def files(key: str, folder: bool = False) -> None:
        value = data.get(key)
        if not isinstance(value, str):
            return
        for one in series if "{series}" in value else series[:1]:
            path = ROOT / value.format(series=one)
            if not (path.is_dir() if folder else path.is_file()):
                found.append(f"[data] {key}: no {'directory' if folder else 'file'} {path}")

    def answer(key: str, check: Any) -> None:
        value = data.get(key)
        if value is None:
            return
        try:
            check(",".join(value) if isinstance(value, list) else str(value))
        except ImportRefused as error:
            found.append(f"[data] {key}: {error}")

    if source == "particles":
        files("particles_folder", folder=True)
        files("particles_table")
        named = run_series.particles_series(cfg)
        absent = [one for one in series if one not in named]
        if absent:
            found.append(f"[data] particles_series names no tomogram number for "
                         f"{', '.join(absent)}")
        table = ROOT / str(data.get("particles_table"))
        if table.is_file():
            from cryoet_pipeline.particles_import import TOMO, _table

            try:
                numbers = sorted({int(round(value)) for value in _table(table)[:, TOMO]})
            except (ImportRefused, ValueError) as error:
                found.append(f"[data] particles_table: {error}")
            else:
                unnamed = [number for number in numbers if number not in named.values()]
                if unnamed:
                    found.append(f"[data] particles_table holds rows of tomograms {unnamed} "
                                 "(column 20), which [data] particles_series gives no series")
                missing = [name for name, number in named.items() if number not in numbers]
                if missing:
                    found.append(f"[data] particles_table has no rows for "
                                 f"{', '.join(missing)}, by [data] particles_series")
        return found

    files("tomogram")
    files("tilt_angles")
    answer("orientation", orientation)
    answer("processing", parse_processing)
    shared = len(series) > 1
    if shared and "{series}" not in str(data.get("tomogram", "{series}")):
        found.append(f"[data] tomogram names one volume for {len(series)} series; give each "
                     "its own with {series}")
    if source == "picks":
        files("picks")
        files("relion_tomograms_star")
        answer("picks_format", check_format)
        answer("picked_in", orientation)
        chosen = data.get("picks_format")
        if shared and "{series}" not in str(data.get("picks", "{series}")):
            if chosen == "relion5-star":
                if "{series}" not in str(data.get("relion_tomo_name", "{series}")):
                    found.append("[data] relion_tomo_name names one tomogram for every series; "
                                 "leave it out for the series ids, or write it with {series}")
            elif chosen == "dynamo-table":
                if "dynamo_tomograms" not in data:
                    found.append(f"[data] picks is one Dynamo table for {len(series)} series; "
                                 "[data] dynamo_tomograms says which tomogram number (column "
                                 "20) is each, as {TS_01 = 1, TS_02 = 2}")
            else:
                found.append(f"[data] picks names one file for {len(series)} series, and a "
                             f"{chosen} file holds no series of its rows; give each series its "
                             "own with {series}")
    return found


def mra_alone(stages: tuple[Stage, ...]) -> bool:
    """Whether a submission runs the MRA on particles extracted before it. Stages are
    submitted as one unbroken slice, and the MRA's follows the extraction's."""
    return bool(stages) and all(stage.is_mra for stage in stages)


def mra_particles(lane: Plan) -> tuple[Any, str | None, str | None]:
    """The particles the lane's MRA would read now, as the job will read them: the
    lane's (cryoet_pipeline.particle_sources.registry_particles), or the classes of an
    earlier run (mra_setup.class_selection). Returns them or None, why not, and a line
    saying which they are."""
    from cryoet_pipeline.particle_sources import registry_particles

    mra = lane.mra
    series = lane.series or project_series(lane.project)
    if not series:
        return None, f"{lane.project} has no series yet", None
    quiet = lambda _: None  # noqa: E731
    try:
        if mra.from_run is None:
            found = registry_particles(ROOT / lane.project, mra.label, series, mra.decoy_label,
                                       log=quiet)
            return found, None, None
        from cryoet_pipeline.mra_setup import class_selection, selected_particles

        chosen = class_selection(ROOT / lane.project / "mra" / mra.from_run, mra.from_classes,
                                 iteration=mra.from_iteration, start_poses=mra.start_poses)
        named = chosen.particles
        if named["label"] != mra.label:
            return None, (f"run {mra.from_run} read the particles of {named['label']!r}, not "
                          f"the lane's {mra.label!r}"), None
        if list(named["series"]) != list(series):
            return None, (f"run {mra.from_run} read series {' '.join(named['series'])}, and "
                          f"the lane has {' '.join(series)}"), None
        found = selected_particles(chosen, log=quiet)
    except (OSError, ValueError, KeyError) as error:
        return None, str(error), None
    held = ", ".join(f"class {channel} {count:,}" for channel, count in chosen.sizes.items()
                     if count)
    return found, None, (f"{chosen.name} of {mra.from_run}, iteration {chosen.iteration}: "
                         f"{len(chosen.tags):,} particles (its classes: {held})")


def mra_problems(lane: Plan, stages: tuple[Stage, ...]) -> list[str]:
    """What would stop the lane's MRA job, when this submission has it."""

    submitted = [stage for stage in stages if stage.is_mra]
    if not submitted or lane.mra is None:
        return []
    stage, mra = submitted[0], lane.mra
    run = mra.run(ROOT)
    shown = _repo_relative(run)
    found = []
    if not any(arg.startswith(("--gres=gpu", "--gpus", "-G")) for arg in stage.sbatch):
        found.append(
            f"stage {stage.name!r} runs the MRA, which searches on GPUs; give that stage "
            "--partition=... and --gres=gpu:..."
        )
    if not mra_alone(stages):
        if mra.from_run is not None:
            found.append(
                f"[mra] from_run runs on the classes of {mra.from_run}, which name the "
                "particles extracted before; this submission extracts them again, so submit "
                "the MRA alone (--from-stage)"
            )
        if run.exists():
            found.append(
                f"the MRA run {shown} exists, and this submission extracts the particles "
                "again, which that run cannot carry on with; give [mra] name another value"
            )
        return found

    from cryoet_pipeline.mra_lane import existing_request
    from cryoet_pipeline.runs import completed

    particles, why, _ = mra_particles(lane)
    if particles is None and mra.from_run is not None:
        found.append(f"the MRA reads part of run {mra.from_run}, and cannot: {why}")
    elif particles is None:
        labels = repr(mra.label) + (f" and {mra.decoy_label!r}" if mra.decoy_label else "")
        found.append(
            f"the MRA reads the particles extracted under {labels}, and cannot: {why}; "
            "extract them first, or submit the stages before it too"
        )
    if particles is not None:
        found += [
            f"the MRA's setup would refuse it: {problem}"
            for problem in mra.setup_problems(particles.box, particles.pixel_size,
                                              len(particles.particles))
        ]
    if not (run / "plan.json").is_file():
        return found
    try:
        request = existing_request(run)
        setup = completed(run / "setup")
    except (OSError, ValueError) as error:
        return [*found, f"the MRA run {shown} exists, and its record cannot be read: {error}"]
    if request is None:
        found.append(
            f"the MRA run {shown} exists and was not asked for by a lane's [mra]; give "
            "[mra] name another value"
        )
    elif particles is not None:
        now = {**mra.resolved(particles.pixel_size), "label": mra.label,
               "series": list(particles.series)}
        before = {**(request.get("resolved") or {}), "label": request.get("label"),
                  "series": request.get("series")}
        differ = sorted(key for key in set(now) | set(before) if now.get(key) != before.get(key))
        if differ:
            found.append(
                f"the MRA run {shown} exists with other settings ({', '.join(differ)}); give "
                "[mra] name another value, or set [mra] as that run had it"
            )
        elif setup is not None and not (
            {(one.slot, one.fingerprint) for one in particles.inputs()}
            <= {(one.slot, one.fingerprint) for one in setup.inputs}
        ):
            found.append(
                f"the MRA run {shown} was set up from particles that have been extracted "
                "again since, so it cannot carry on; give [mra] name another value"
            )
    return found


def mra_lines(lane: Plan, stages: tuple[Stage, ...]) -> list[str]:
    """What the dry run says of the lane's MRA: its run, its particles and every setting
    with its source."""

    submitted = [stage for stage in stages if stage.is_mra]
    if not submitted or lane.mra is None:
        return []
    mra, stage = lane.mra, submitted[0]
    run = mra.run(ROOT)
    lines = [f"mra {_repo_relative(run)}: stage {stage.name!r}, scripts/mra_port_run.sbatch"]
    described = mra.describe()
    if mra_alone(stages):
        particles, _, which = mra_particles(lane)
        if (run / "plan.json").is_file():
            lines.append("  the run exists: with the same settings and particles, the job "
                         "carries it on from its last finished iteration, or only reports it "
                         "again")
        if which is not None:
            lines.append(f"  {which}")
        if particles is not None:
            described = mra.describe(particles.box, particles.pixel_size,
                                     len(particles.particles))
    else:
        before = stages[stages.index(stage) - 1]
        made = "imports" if before.end == "import-particles" else "extracts"
        lines.append(f"  after stage {before.name!r}, on the particles it {made}")
    lines += [f"  {line}" for line in described]
    left = sorted(name for name in MRA_VARIABLES if name not in _MRA_SET and os.environ.get(name))
    if left:
        lines.append(f"  not passed to the MRA job, which takes its settings from [mra]: "
                     f"{', '.join(left)}")
    return lines


def ctffind_problems(cfg: dict[str, Any], steps: set[str]) -> list[str]:
    """A CTFFIND that estimate-ctf's backend would refuse after the queue wait,
    as a problem.

    estimate-ctf runs the CTFFIND the config names in [executables] ctffind, or
    else ctffind5, or else ctffind, from PATH; the job's environment, PATH and
    the config's $VARs included, is this one (--export=ALL). Its backend refuses
    a named path that is not a file, and any CTFFIND 4; one it may not execute
    fails as it starts. Which version it is, its start-up banner says. A CTFFIND
    that cannot be found on PATH or asked here is noted rather than refused,
    since the job's environment may still find one.
    """

    if "estimate-ctf" not in steps:
        return []
    configured = (cfg.get("executables") or {}).get("ctffind")
    if configured is not None:
        return named_ctffind_problems(str(configured))
    # The backend's order (cryoet_pipeline.backends.ctf._resolve_ctffind_executable).
    found = next((path for name in ("ctffind5", "ctffind") if (path := shutil.which(name))), None)
    if found is None:
        print("not checked which CTFFIND estimate-ctf will run: neither ctffind5 nor ctffind "
              "is on PATH here", file=sys.stderr)
        return []
    if (version := ctffind_before_5(found)) is None:
        return []
    return [
        f"estimate-ctf would run {found}, which is CTFFIND {version}, and its backend answers "
        "CTFFIND 5's questions: name CTFFIND 5 in [executables] ctffind (\"$CTFFIND5\", as "
        "ryr1-bin8 does), or put ctffind5 on PATH ahead of it"
    ]


def named_ctffind_problems(configured: str) -> list[str]:
    """[executables] ctffind as the job will be given it, checked as the backend
    checks it (cryoet_pipeline.backends.ctf._resolve_ctffind_executable).

    scripts/run_series.py expands its $VARs, the backend its ~, and lane.sbatch
    runs both from the repo root, so a relative path is from there. The job of
    2026-10-29 that this catches (ribo-mdc, 3789938) had CTFFIND5 set to the
    directory CTFFIND 5 was installed in, and stopped 11 minutes in.
    """

    if any(not os.environ.get(name) for name in _ENV_REFERENCE.findall(configured)):
        # Refused already in problems(), as not set.
        return []
    found = ROOT / Path(os.path.expandvars(configured)).expanduser()
    where = f'[executables] ctffind "{configured}"'
    if found.is_dir():
        inside = ctffind_inside(found)
        return [
            f"{where}: {found} is a directory, and estimate-ctf would stop on it; name the "
            "CTFFIND program in it instead" + (f", such as {inside}" if inside else "")
        ]
    if not found.is_file():
        return [f"{where}: no file {found}, and estimate-ctf would stop on it"]
    if not os.access(found, os.X_OK):
        return [f"{where}: {found} is not executable, and estimate-ctf would stop on it"]
    if (version := ctffind_before_5(str(found))) is None:
        return []
    return [
        f"{where}: {found} is CTFFIND {version}, and estimate-ctf's backend answers "
        "CTFFIND 5's questions; name a CTFFIND 5 instead"
    ]


def ctffind_inside(directory: Path) -> Path | None:
    """A CTFFIND program in a directory named in its place, where an install puts
    it (MDC's is cisTEM/bin/ctffind under ctffind-5.0.2): ctffind5 before ctffind,
    as on PATH. None if there is none."""

    for name in ("ctffind5", "ctffind"):
        for pattern in (name, f"bin/{name}", f"*/bin/{name}"):
            for path in sorted(directory.glob(pattern)):
                if path.is_file() and os.access(path, os.X_OK):
                    return path
    return None


def ctffind_before_5(executable: str) -> str | None:
    """The version of a CTFFIND older than 5, which the backend refuses, as its
    banner states it; None for a later one, and for one that states none here,
    which is noted."""

    version = ctffind_version(executable)
    if version is None:
        print(f"not checked which CTFFIND {executable} is: it printed no version within "
              f"{CTFFIND_PROBE_SECONDS:g} s", file=sys.stderr)
        return None
    return version if int(version.split(".")[0]) < 5 else None


def ctffind_version(executable: str, limit: int = 1 << 16) -> str | None:
    """The version a CTFFIND's start-up banner states, read from what it prints
    before its first question; None if it states none in CTFFIND_PROBE_SECONDS.

    It is given no input and stopped once the banner is read, so it fits nothing,
    and at most `limit` bytes are read from a program that keeps asking.
    """
    try:
        process = subprocess.Popen(
            [executable], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        return None
    assert process.stdout is not None
    output = b""
    deadline = time.monotonic() + CTFFIND_PROBE_SECONDS
    try:
        while len(output) < limit:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([process.stdout], [], [], left)[0]:
                break
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            output += chunk
            if match := _CTFFIND_VERSION.search(output.decode(errors="replace")):
                return match["version"]
    finally:
        process.kill()
        process.wait()
        process.stdout.close()
    return None


def init_command(cfg: dict[str, Any], lane: Plan) -> list[str] | None:
    """`cryoet init` for a frames lane whose project does not exist yet, else None.

    init is project-level and quick (it reads the mdocs and checks the movies are
    there), so it runs here on the submit node rather than in a job. On an
    existing project plain init refuses, since it would start the registry empty;
    series are added to one with --add-series, which the refusal names.
    """
    if run_series.lane_input(cfg) != "frames":
        return None
    if (ROOT / lane.project / "artifacts.json").exists():
        return None
    return _init_arguments(cfg, lane.project, lane.series)


def _init_arguments(cfg: dict[str, Any], project: str, series: Sequence[str]) -> list[str]:
    """`cryoet init` from a frames lane's [data], for these series of `project`."""
    data = cfg.get("data", {})
    command = ["cryoet", "init", "--frames", str(data.get("frames"))]
    if "frames_pattern" in data:
        command += ["--frames-pattern", str(data["frames_pattern"])]
    else:
        command += ["--mdocs", str(data.get("mdocs"))]
    if "pixel_size_angstrom" in data:
        command += ["--pixel-size", str(data["pixel_size_angstrom"])]
    command += ["--out", project]
    for one in series:
        command += ["--tilt-series", one]
    return command


def _repo_relative(path: Path) -> str:
    """lane.sbatch cds to the repo, so a relative path has to be from there."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


class SubmitError(Exception):
    """A stage could not be submitted. The stages before it already were."""

    def __init__(self, message: str, job_ids: list[str]) -> None:
        super().__init__(message)
        self.job_ids = job_ids


def submit(
    lane: Plan, stages: tuple[Stage, ...], dry_run: bool, overwrite: bool = False,
    config_sha256: str | None = None,
) -> list[str]:
    job_ids: list[str] = []
    prefix = Path(lane.project).name
    # OVERWRITE reaches the jobs only from --overwrite: main() refuses one set in
    # the environment, and an empty one is dropped here.
    inherited = {key: value for key, value in os.environ.items() if key != "OVERWRITE"}
    for stage in stages:
        after = (
            [f"--dependency=afterok:{job_ids[-1]}", "--kill-on-invalid-dep=yes"]
            if job_ids
            else []
        )
        script = MRA_SBATCH if stage.is_mra else LANE_SBATCH
        command = [
            "sbatch", "--parsable", "--export=ALL",
            *stage.sbatch, *after,
            f"--time={stage.time}", f"--job-name={prefix}-{stage.name}",
            str(script.relative_to(ROOT)),
        ]
        # Through the environment rather than --export=VAR=value, whose list is
        # comma-separated: SERIES holds spaces, and nothing has to survive a parser.
        if not lane.series and not dry_run:
            raise SubmitError("no series to submit", job_ids)
        # Unknown in a dry run of a new project without [run] series: init, which
        # finds them, has not run.
        series = " ".join(lane.series) or "<every series init finds>"
        base = inherited
        if stage.is_mra:
            # The MRA's settings are the lane's [mra], read by the job from the config
            # it was shown with; nothing else of the MRA's reaches it from the shell.
            base = {key: value for key, value in inherited.items() if key not in MRA_VARIABLES}
            env = {
                "CONFIG": _repo_relative(lane.config),
                "CONFIG_SHA256": config_sha256
                or hashlib.sha256(lane.config.read_text().encode()).hexdigest(),
                "SOURCE": lane.project,
                "SERIES": series,
                "NAME": lane.mra.name,
            }
        else:
            env = {
                "PROJECT": lane.project,
                "CONFIG": _repo_relative(lane.config),
                "SERIES": series,
                "FROM": stage.start,
                "UNTIL": stage.end,
            }
            if overwrite:
                env["OVERWRITE"] = "1"
        if dry_run:
            shown = " ".join(f"{key}={value!r}" for key, value in env.items())
            print(f"{shown} {' '.join(command)}")
            job_ids.append(f"<{stage.name}>")
            continue
        try:
            result = subprocess.run(
                command, cwd=ROOT, env={**base, **env},
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as error:
            said = (error.stderr or "").strip() or f"exit {error.returncode}"
            raise SubmitError(f"sbatch refused stage {stage.name!r}: {said}", job_ids) from error
        except OSError as error:
            raise SubmitError(f"could not run sbatch: {error}", job_ids) from error
        # --parsable prints "id;cluster" on a multi-cluster setup.
        job_ids.append(result.stdout.strip().split(";")[0])
    return job_ids


@dataclass
class Submission:
    """One config's part of a dry run or of a submission."""

    cfg: dict[str, Any]
    text: str
    lane: Plan
    stages: tuple[Stage, ...]
    problems: list[str]
    record: dict[str, Any]
    replaces: dict[str, list[str]]
    id: str | None = None  # of its journal entries, once the first is written
    # Shown by a dry run: a line per series init would find, and what Slurm's
    # sbatch --test-only said of each stage.
    found_by_init: tuple[str, ...] = ()
    slurm_said: tuple[str, ...] = ()


def plan_record(
    cfg: dict[str, Any], lane: Plan, stages: tuple[Stage, ...], text: str, overwrite: bool
) -> dict[str, Any]:
    """What a dry run shows, as JSON. A submission from the assistant must repeat
    it exactly."""

    series = lane.series
    if not series and init_command(cfg, lane) is None:
        series = project_series(lane.project)
    record = {
        "config": _repo_relative(lane.config),
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "project": lane.project,
        "series": list(series),
        "stages": [
            {"name": stage.name, "from": stage.start, "until": stage.end,
             "time": stage.time, "sbatch": list(stage.sbatch),
             **({"kind": "mra"} if stage.is_mra else {})}
            for stage in stages
        ],
        "overwrite": overwrite,
        # What the jobs will find in the variables their commands name, since
        # --export=ALL hands them this environment.
        "env": {
            name: os.environ.get(name, "")
            for name in sorted({name for stage in stages for name in stage.env})
        },
    }
    if lane.mra is not None and any(stage.is_mra for stage in stages):
        # The MRA as the dry run shows it: its run, and every setting as asked for,
        # which the job turns into numbers once it has read the particles.
        record["mra"] = {"run": _repo_relative(lane.mra.run(ROOT)),
                         "label": lane.mra.label, **lane.mra.resolved(None)}
    return record


def recorded_runs(
    cfg: dict[str, Any], lane: Plan, stages: tuple[Stage, ...]
) -> dict[str, list[str]]:
    """Per series, the steps of these stages that have run before: what
    --overwrite may replace.

    Read from step_timings.jsonl, so it is what is known rather than everything:
    a run from before the project kept timings is missing, and one from before
    the rows named their lane counts for every lane.
    """

    series = lane.series or project_series(lane.project)
    if not series:
        return {}
    steps = run_series.build_steps(cfg, Path(lane.project), series[0])
    labelled = show_project.step_labels(steps)
    # The MRA replaces nothing: a run of its name carries on, or is refused.
    wanted = [
        name
        for stage in stages
        if not stage.is_mra
        for name, _ in run_series.slice_steps(steps, stage.start, stage.end)
    ]
    runs = show_project.last_runs(
        ROOT / lane.project, set(series), labelled, cfg["lane"]["variant"]
    )
    found = {one: [step for step in wanted if (one, step) in runs] for one in series}
    return {one: names for one, names in found.items() if names}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def from_assistant() -> bool:
    return any(os.environ.get(name, "").strip() for name in ASSISTANT_MARKERS)


def _who() -> dict[str, Any]:
    """Who runs this, where, and through which client: part of every entry."""
    return {
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "assistant": from_assistant(),
        "client": {name: os.environ[name] for name in CLIENT_VARIABLES if os.environ.get(name)},
    }


def read_journal(project: str) -> list[dict[str, Any]]:
    """The project's journal, oldest entry first."""
    try:
        lines = (ROOT / project / JOURNAL).read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut short
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_journal(project: str, row: dict[str, Any]) -> Path:
    """Append one entry, in a single write, so that two writers' lines do not mix."""
    path = ROOT / project / JOURNAL
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not escaped to ASCII, so that a reason written in Russian stays readable.
    data = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError(f"{path}: the entry was written only in part")
    finally:
        os.close(descriptor)
    return path


def queued_jobs() -> list[dict[str, str]] | None:
    """This user's jobs that Slurm still holds, waiting or running; None where
    there is no squeue to ask. A squeue that fails raises."""
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%j|%T"],
            check=True, capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3:
            jobs.append(dict(zip(("id", "name", "state"), parts, strict=True)))
    return jobs


def journal_series(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The series of every job the journal says was submitted, by job ID."""
    planned = {
        row.get("id"): (row.get("plan") or {}).get("series") or []
        for row in rows
        if row.get("event") == "submit"
    }
    found: dict[str, list[str]] = {}
    for row in rows:
        if row.get("event") in ("submitted", "submit-failed"):
            series = row.get("series") or planned.get(row.get("id"), [])
            for job in (row.get("jobs") or {}).values():
                found[str(job)] = list(series)
    return found


def log_header(root: Path, job_id: str) -> dict[str, Any] | None:
    """The project and series a job's log names, or None while it has no log:
    a job that waits has not written one yet."""
    for path in sorted(root.glob(f"slurm-*-{glob.escape(job_id)}.log")):
        header: dict[str, Any] = {}
        try:
            with path.open(errors="replace") as handle:
                for line in handle:
                    if header and not line.strip():
                        break  # the header ends at its first blank line
                    if match := _HEADER_SERIES_IN.match(line):
                        header.setdefault("series", match["series"].split())
                        header.setdefault("project", match["project"])
                    elif match := _HEADER_PROJECT.match(line):
                        header.setdefault("project", match["project"])
                    elif match := _HEADER_SERIES.match(line):
                        header.setdefault("series", match["series"].split())
                    if "project" in header and "series" in header:
                        break
        except OSError:
            continue
        return header
    return None


def journal_mra_runs(rows: list[dict[str, Any]]) -> dict[str, str]:
    """The MRA run of every MRA job the journal says was submitted, by job ID."""
    planned = {
        row.get("id"): row.get("plan") or {}
        for row in rows
        if row.get("event") == "submit"
    }
    found: dict[str, str] = {}
    for row in rows:
        if row.get("event") not in ("submitted", "submit-failed"):
            continue
        plan = planned.get(row.get("id"), {})
        kinds = {
            stage.get("name"): stage.get("kind")
            for stage in plan.get("stages") or []
            if isinstance(stage, dict)
        }
        for stage, job in (row.get("jobs") or {}).items():
            if kinds.get(stage) == "mra":
                found[str(job)] = str((plan.get("mra") or {}).get("run", ""))
    return found


def busy(
    cfg: dict[str, Any], lane: Plan, jobs: list[dict[str, str]], root: Path = ROOT,
    submitting: str = "this lane", reading: str | None = None,
) -> list[str]:
    """Problems: jobs of this project, still queued or running, on series this
    lane (or a recipe, scripts/submit_recipe.py) would run.

    `reading` is the MRA run of a submission that only runs the MRA, which reads
    its series: an MRA job of another run reads them too, and is no problem for it.
    """

    series = lane.series
    if not series and init_command(cfg, lane) is None:
        series = project_series(lane.project)
    project = (root / lane.project).resolve()
    rows = read_journal(lane.project)
    journaled = journal_series(rows)
    readers = journal_mra_runs(rows)
    found = []
    for job in jobs:
        # squeue names an array job's tasks "<id>_<task>", and sbatch printed "<id>".
        theirs = journaled.get(job["id"], journaled.get(job["id"].split("_", 1)[0]))
        their_run = readers.get(job["id"], readers.get(job["id"].split("_", 1)[0]))
        if reading is not None and their_run is not None and their_run != reading:
            continue
        if theirs is None:
            header = log_header(root, job["id"])
            if header is not None and "project" in header:
                if (root / header["project"]).resolve() != project:
                    continue
                theirs = header.get("series", [])
            elif job["name"].startswith(f"{Path(lane.project).name}-"):
                theirs = []
            else:
                continue
        what = f"job {job['id']} ({job['name']}, {job['state'].lower()})"
        if not theirs:
            found.append(
                f"{what} is this project's, and nothing on record says which series it "
                "works on, so it may be any of them; submit after it has ended"
            )
            continue
        # Without series of its own the lane takes every series.
        shared = [one for one in theirs if not series or one in series]
        if not shared:
            continue
        if their_run is not None and reading is not None:
            found.append(
                f"{what} is the MRA run {their_run}, which {submitting} would run too; its "
                "lock lets one job at a time on a run, so submit after it has ended"
            )
        elif their_run is not None:
            found.append(
                f"{what} is the MRA run {their_run}, which reads the particles of "
                f"{', '.join(shared)}; {submitting} would run steps on them, and a new "
                "extraction would change them under it, so submit after it has ended"
            )
        elif reading is not None:
            found.append(
                f"{what} works on {', '.join(shared)}, whose particles {submitting} reads; "
                "they may change under it, so submit after it has ended"
            )
        else:
            found.append(
                f"{what} works on {', '.join(shared)}, which {submitting} would run too; "
                "two jobs on one series write the same files, so submit after it has ended"
            )
    return found


def queue_problems(
    targets: list[tuple[dict[str, Any], Plan]], submitting: str = "this lane",
    reading: list[str | None] | None = None,
) -> list[list[str]]:
    """Per lane, the jobs already queued or running on its series, as problems.
    `reading` holds, per lane, the MRA run of a submission that only runs the MRA."""
    try:
        jobs = queued_jobs()
    except subprocess.CalledProcessError as error:
        said = (error.stderr or "").strip() or f"exit {error.returncode}"
        return [
            [f"squeue failed ({said}), so it is unknown whether the series are free"]
            for _ in targets
        ]
    if jobs is None:
        print("not checked whether the series are free: no squeue here", file=sys.stderr)
    reading = reading or [None] * len(targets)
    return [
        busy(cfg, lane, jobs or [], submitting=submitting, reading=run)
        for (cfg, lane), run in zip(targets, reading, strict=True)
    ]


# How a submission can differ from its dry run, in words, by plan key: a lane's
# (plan_record) or a recipe's (scripts/submit_recipe.py).
_DIFFERS = {
    "config_sha256": "the config file has changed",
    "script_sha256": "the recipe's script has changed",
    "project": "the project differs",
    "series": "the series differ",
    "stages": "the stages differ",
    "parameters": "the recipe's parameters differ",
    "files": "a file the recipe reads has changed",
    "sbatch": "the sbatch options differ",
    "overwrite": "only one of them has --overwrite",
    "mra": "the MRA's settings differ",
}


def _thread(plan: dict[str, Any]) -> tuple[Any, ...]:
    """What a dry run is a dry run of: a lane's config, or a recipe on a config's
    series. A dry run counts only for a submission of the same."""
    if plan.get("recipe") is None:
        return (plan.get("config"),)
    return (plan.get("config"), plan["recipe"], plan.get("series"))


def unshown(
    rows: list[dict[str, Any]], record: dict[str, Any], user: str, now: datetime
) -> str | None:
    """Why a submission from the assistant is not the dry run it showed, or None.

    It is that dry run when the latest journal entry of the same config (and,
    for a recipe, of the same recipe on the same series) is this user's dry run
    of exactly this plan, without problems, at most DRY_RUN_FRESH old. A
    submission is journaled before its first sbatch, so one dry run stands for
    one submission.
    """

    mine = [
        row for row in rows
        if row.get("event") in ("dry-run", "submit")
        and _thread(row.get("plan") or {}) == _thread(record)
    ]
    if not mine:
        return "no dry run of it is on record"
    last = mine[-1]
    when = last.get("at", "?")
    if last["event"] == "submit":
        return f"it was submitted at {when}, after its last dry run"
    if last.get("user") != user:
        return f"its last dry run, at {when}, was {last.get('user')}'s"
    if last.get("problems"):
        return f"its last dry run, at {when}, found problems"
    shown = last.get("plan") or {}
    # Every key, so that a part of a plan this list does not name yet still counts.
    keys = [*_DIFFERS, *sorted((set(shown) | set(record)) - set(_DIFFERS) - {"env"})]
    differs = [
        _DIFFERS.get(key, f"its {key} differs")
        for key in keys
        if shown.get(key) != record.get(key)
    ]
    env = shown.get("env") or {}
    variables = sorted(
        f"${name}"
        for name in set(env) | set(record["env"])
        if env.get(name) != record["env"].get(name)
    )
    if variables:
        differs.append(f"{', '.join(variables)} differ{'s' if len(variables) == 1 else ''}")
    if differs:
        return f"it differs from its dry run at {when}: {'; '.join(differs)}"
    try:
        age = now - datetime.fromisoformat(when)
    except (TypeError, ValueError):
        return f"its last dry run has no time that can be read ({when!r})"
    if age > DRY_RUN_FRESH:
        return (
            f"its dry run at {when} is {int(age.total_seconds() // 60)} minutes old, "
            f"and a dry run stands for {int(DRY_RUN_FRESH.total_seconds() // 60)}"
        )
    return None


def config_diff(cfg: dict[str, Any], config: Path, text: str) -> dict[str, str] | None:
    """How a config copy differs from the lane file it was made from: the one in
    configs/lanes with the same [lane] name and [run] project. None for a lane
    file itself, and when not exactly one file matches."""

    lanes = ROOT / "configs" / "lanes"
    if config.resolve().parent == lanes.resolve():
        return None
    key = (cfg["lane"].get("name"), cfg["run"].get("project"))
    matches = []
    for path in sorted(lanes.glob("*.toml")):
        try:
            other = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if (other.get("lane", {}).get("name"), other.get("run", {}).get("project")) == key:
            matches.append(path)
    if len(matches) != 1:
        return None
    lines = difflib.unified_diff(
        matches[0].read_text().splitlines(keepends=True),
        text.splitlines(keepends=True),
        _repo_relative(matches[0]),
        _repo_relative(config),
    )
    return {"from": _repo_relative(matches[0]), "diff": "".join(lines)}


def code_version() -> dict[str, Any] | None:
    """The commit checked out at submission, and the tracked files changed since."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        changed = git("status", "--porcelain", "--untracked-files=no").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "modified": [line[3:] for line in changed]}


def next_step(problems_found: bool) -> str:
    """What the assistant does after a dry run."""
    if problems_found:
        return "to submit: once the problems above are resolved, run the dry run again"
    return (
        "to submit: show this dry run, and after a yes run the same command without "
        f"--dry-run and with --reason, within {int(DRY_RUN_FRESH.total_seconds() // 60)} "
        "minutes"
    )


def _close(one: Submission, event: str, **fields: Any) -> None:
    """The entry that says how a journaled submission ended."""
    try:
        write_journal(one.lane.project, {
            "at": _now(), "event": event, "config": one.record["config"], "id": one.id,
            "user": getpass.getuser(), **fields,
        })
    except OSError as error:
        # The jobs already in the queue stay there whether or not this is written.
        print(f"not journaled: {error}", file=sys.stderr)


def _abandon(lanes: list[Submission], why: str) -> None:
    """Close the journal entries of submissions that will not be made."""
    for one in lanes:
        if one.id is not None:
            _close(one, "submit-failed", jobs={}, error=why)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("configs", nargs="+", type=Path, help="lane configs with a [run] section")
    parser.add_argument(
        "--from-stage",
        default=None,
        help="submit this stage and the ones after it, e.g. to resume after a failed "
        "job; with several configs, every one of them must have it",
    )
    parser.add_argument(
        "--until-stage",
        default=None,
        help="submit no stage after this one, e.g. to look at the picks before the MRA "
        "stage, which --from-stage submits later; with several configs, every one of "
        "them must have it",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the sbatch commands and any problems; submit nothing",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="let every step of the submitted stages replace what it wrote before, for "
        "this submission only; the dry run lists the recorded runs it may replace",
    )
    parser.add_argument(
        "--reason", default=None,
        help="what was asked and by whom, in English, for the project's journal; required "
        "from the assistant",
    )
    args = parser.parse_args(argv)
    command_line = ["scripts/submit_lanes.py", *(sys.argv[1:] if argv is None else argv)]

    if os.environ.get("OVERWRITE"):
        raise SystemExit(
            "OVERWRITE is set in the environment, and submit_lanes.py does not pass it on: "
            "give --overwrite to replace outputs in this submission only, or unset OVERWRITE"
        )
    who = _who()
    if who["assistant"] and not args.dry_run and not (args.reason or "").strip():
        raise SystemExit(
            "a submission from the assistant needs --reason: what was asked, and by whom"
        )

    # Everything is checked before the first sbatch, so a broken second config
    # does not leave the first lane submitted alone.
    lanes: list[Submission] = []
    no_sbatch = False
    for config in args.configs:
        text, cfg = run_series.read_config(config)
        lane = plan(cfg, config)
        names = [stage.name for stage in lane.stages]
        for option, given in (("--from-stage", args.from_stage),
                              ("--until-stage", args.until_stage)):
            if given is not None and given not in names:
                raise SystemExit(f"{config}: {option}: no stage {given!r}; stages: "
                                 f"{', '.join(names)}")
        first = names.index(args.from_stage) if args.from_stage else 0
        last = names.index(args.until_stage) if args.until_stage else len(names) - 1
        if first > last:
            raise SystemExit(f"{config}: --from-stage {args.from_stage!r} comes after "
                             f"--until-stage {args.until_stage!r}")
        stages = lane.stages[first : last + 1]
        lane_problems = problems(cfg, lane, stages)
        found_by_init: list[str] = []
        if args.dry_run and init_command(cfg, lane) is not None and not init_data_problems(cfg):
            trouble, found_by_init = init_preview(cfg, lane)
            lane_problems += trouble
        slurm = slurm_check(lane, stages)
        if slurm is None:
            no_sbatch = True
        else:
            lane_problems += slurm[0]
        lanes.append(Submission(
            cfg, text, lane, stages, lane_problems,
            plan_record(cfg, lane, stages, text, args.overwrite),
            recorded_runs(cfg, lane, stages) if args.overwrite else {},
            found_by_init=tuple(found_by_init),
            slurm_said=tuple(slurm[1]) if slurm else (),
        ))
    if no_sbatch:
        print("not checked whether Slurm takes the stages' resources: no sbatch here",
              file=sys.stderr)

    projects = [one.lane.project for one in lanes]
    if len(set(projects)) != len(projects):
        # Two jobs on the same series write the same files.
        raise SystemExit("two configs submit to the same project; submit them one after the other")
    # For the same reason, no job already queued or running may be on the series; an
    # MRA alone only reads them, as an MRA job of another run does.
    targets = [(one.cfg, one.lane) for one in lanes]
    reading = [
        _repo_relative(one.lane.mra.run(ROOT)) if mra_alone(one.stages) else None
        for one in lanes
    ]
    for one, taken in zip(lanes, queue_problems(targets, reading=reading), strict=True):
        one.problems += taken

    found = [f"{one.lane.config}: {problem}" for one in lanes for problem in one.problems]
    for problem in found:
        print(f"{'would refuse' if args.dry_run else 'refused'}: {problem}", file=sys.stderr)
    if any(unset_variables(one.stages) for one in lanes):
        print(UNSET_HINT, file=sys.stderr)
    if found and not args.dry_run:
        return 1

    if who["assistant"] and not args.dry_run:
        # The person agreed to the dry run they were shown, and to nothing else.
        now = datetime.now(UTC)
        refused = [
            f"{one.lane.config}: {why}; dry-run it, show the dry run, and submit after a yes"
            for one in lanes
            if (why := unshown(read_journal(one.lane.project), one.record, who["user"], now))
        ]
        for problem in refused:
            print(f"refused: {problem}", file=sys.stderr)
        if refused:
            return 1

    # A submission is journaled before its first sbatch, so that one which dies
    # half-way is on record, and so that its dry run cannot be used again.
    journaled = []
    for index, one in enumerate(lanes):
        entry = {
            **who, "at": _now(), "config": one.record["config"], "argv": command_line,
            "plan": one.record, "plan_sha256": _digest(one.record),
            "reason": args.reason, "replaces": one.replaces,
        }
        if args.dry_run and one.problems and not (ROOT / one.lane.project).exists():
            # A dry run that would refuse makes no project folder to hold its entry:
            # no submission may follow it, and the folder would stay behind, as
            # outputs/ryr1 did on cryoem.se on 24 November 2026 for a lane of MDC's.
            print(f"not journaled: {one.lane.project} does not exist yet, and a dry run "
                  "that would refuse does not make it", file=sys.stderr)
            continue
        if args.dry_run:
            entry |= {"event": "dry-run", "problems": one.problems}
        else:
            one.id = uuid.uuid4().hex[:12]
            entry |= {
                "event": "submit", "id": one.id, "config_text": one.text,
                "config_diff": config_diff(one.cfg, one.lane.config, one.text),
                "code": code_version(), "init": init_command(one.cfg, one.lane),
            }
        try:
            journaled.append(write_journal(one.lane.project, entry))
        except OSError as error:
            if args.dry_run:
                print(f"not journaled: {error}", file=sys.stderr)
                continue
            one.id = None
            _abandon(lanes[:index], f"not submitted: {one.lane.config} could not be journaled")
            print(f"refused: {one.lane.config}: cannot write its journal: {error}", file=sys.stderr)
            return 1

    for index, one in enumerate(lanes):
        cfg, lane, stages = one.cfg, one.lane, one.stages
        print(f"# {lane.config}: {lane.project}, "
              + (f"{len(lane.series)} series" if lane.series else "every series"))
        combination = run_series.lane_combination(cfg, movie_formats(cfg))
        if combination not in run_series.RUN_ON_REAL_DATA:
            print(f"  warning: {run_series.combination_text(combination)} has not run on real "
                  "data yet, so this lane is the first: expect it to need fixing, and tell the "
                  "maintainers what happened (docs/agent/pipeline_map.md, \"Which combinations "
                  "have run on real data\")")
        init = init_command(cfg, lane)
        try:
            if init is not None:
                print(("would run: " if args.dry_run else "") + shlex.join(init))
                for line in one.found_by_init:
                    print(f"  init would find {line}")
                if not args.dry_run:
                    subprocess.run(init, cwd=ROOT, check=True)
            if not lane.series and (init is None or not args.dry_run):
                # Every series: the ones init just wrote manifests for, or the ones
                # the project already has.
                lane = replace(lane, series=project_series(lane.project))
                print(f"  series: {' '.join(lane.series)}")
            ids = submit(lane, stages, args.dry_run, args.overwrite,
                         config_sha256=one.record["config_sha256"])
        except subprocess.CalledProcessError as error:
            queued, why = [], f"init exited with {error.returncode}"
        except SubmitError as error:
            queued, why = error.job_ids, str(error)
        else:
            queued, why = ids, None
        # Shorter than the stages when sbatch refused one part-way.
        jobs = dict(zip((stage.name for stage in stages), queued, strict=False))
        if why is not None:
            _close(one, "submit-failed", jobs=jobs, error=why)
            _abandon(lanes[index + 1 :], f"not submitted: {lane.config} failed before it")
            print(f"refused: {lane.config}: {why}", file=sys.stderr)
            if jobs:
                print(
                    "  already in the queue, without the stages after it: "
                    + ", ".join(f"{name} {job}" for name, job in jobs.items())
                    + f"; to take it back: scancel {' '.join(jobs.values())}",
                    file=sys.stderr,
                )
            return 1
        print("  " + " -> ".join(f"{name} {job}" for name, job in jobs.items()))
        against = set_against_rules(cfg)
        if against:
            print("  set where the lab's rules leave the key unset "
                  "(docs/agent/lane_questions.md, \"Every key\"); each needs its reason, "
                  "a request or a comment in the lane:")
            for line in against:
                print(f"    {line}")
        if args.dry_run and one.slurm_said:
            print("  Slurm, asked with sbatch --test-only (nothing submitted; each stage "
                  "alone, without waiting for the one before):")
            for line in one.slurm_said:
                print(f"    {line}")
        for line in mra_lines(lane, stages):
            print(f"  {line}")
        if args.overwrite:
            if one.replaces:
                print("  --overwrite may replace what these recorded runs wrote:")
                for series, steps in one.replaces.items():
                    print(f"    {series}: {', '.join(steps)}")
            else:
                print("  --overwrite: no recorded run of these steps to replace")
        if not args.dry_run:
            _close(one, "submitted", jobs=jobs, series=list(lane.series))

    shown = ", ".join(_repo_relative(path) for path in journaled)
    if args.dry_run:
        if journaled:
            print(f"recorded in {shown}")
        if who["assistant"]:
            print(next_step(bool(found)))
    else:
        print(f"journal: {shown}")
        print("watch: squeue -u $USER   logs: slurm-<job name>-<id>.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
