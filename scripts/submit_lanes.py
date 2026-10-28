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
--reason and the job IDs. A submission from the assistant is refused unless it
gives --reason and the config's latest journal entry is the same person's dry
run of exactly this plan, without problems, at most 30 minutes old: what is
submitted is what was shown, once. The assistant is known by CRYOET_ASSISTANT,
which its launcher sets, or by the AI_AGENT or CLAUDECODE its client sets (see
docs/agent_architecture.md, "First increment").

Two jobs on one series write the same files, so a series that a queued or
running job of the same project works on is refused, in a dry run as in a
submission. squeue lists the jobs. The journal says which series each of them
works on, or else the job's log does, once the job runs. A job named like the
project's own jobs that neither accounts for may be on any of its series. Only
the user's own jobs are asked about, since a project lives in its user's clone.

The recipes, experiments written in advance that run a lane's steps another way
(scripts/ctf_window_test.sbatch and the like), are submitted by
scripts/submit_recipe.py, with these checks and into the same journal.

[run] series may be left out, to take every series: a frames lane then gets the
ones `init` finds, or, for a project that exists, the ones it has manifests for;
a stack lane gets every series its [data] stack path matches, with {series} as
the wildcard.

For a frames lane whose project does not exist yet, `cryoet init` runs here
first, from [data] frames and either mdocs or frames_pattern (tilt metadata
read from the movie file names). It is never run on an existing project:
it starts the registry empty. Not done here: downloading data and caching EMDB
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
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_project  # noqa: E402

LANE_SBATCH = ROOT / "scripts" / "lane.sbatch"
_ENV_REFERENCE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
# Each project's record of what was shown and what was submitted: one JSON object
# per line, appended to and never rewritten.
JOURNAL = "journal.jsonl"
# How long a dry run stands for the plan it showed. Past it the queue and the
# outputs may have moved on, and so may what the person remembers agreeing to.
DRY_RUN_FRESH = timedelta(minutes=30)
# Any of these, set and not empty, means an agentic client runs this.
# CRYOET_ASSISTANT is set by the assistant's launcher. Claude Code sets the other
# two in every shell it starts, and some other clients set AI_AGENT too.
ASSISTANT_MARKERS = ("CRYOET_ASSISTANT", "AI_AGENT", "CLAUDECODE")
# Copied into the journal as they are, when set: the client, the model (the
# launcher's CRYOET_ASSISTANT_MODEL), and the session whose transcript holds the
# conversation. Nothing else from the environment is, apart from a plan's $VARs.
CLIENT_VARIABLES = (
    *ASSISTANT_MARKERS, "CRYOET_ASSISTANT_MODEL", "ANTHROPIC_MODEL", "CLAUDE_CODE_SESSION_ID",
)
# What a job's log says it works on, in the header lane.sbatch writes and the
# recipes copy: "project P ..." with "series A B", or "series A B in P".
_HEADER_PROJECT = re.compile(r"^project\s+(?P<project>[^\s,]+)")
_HEADER_SERIES_IN = re.compile(r"^series\s+(?P<series>.+?)\s+in\s+(?P<project>\S+)\s*$")
_HEADER_SERIES = re.compile(r"^series\s+(?P<series>.+?)\s*$")
# The version line of CTFFIND's start-up banner, as the backend reads it
# (cryoet_pipeline.backends.ctf.ctffind_version): anchored, so that CTFFIND 5's
# "Library Version" line is not taken for it.
_CTFFIND_VERSION = re.compile(r"^\s*Version\s*:\s*(?P<version>\d+(?:\.\d+)*)", re.MULTILINE)
# How long CTFFIND may take to print its banner when asked here.
CTFFIND_PROBE_SECONDS = 5.0


@dataclass(frozen=True)
class Stage:
    name: str
    start: str
    end: str
    time: str
    sbatch: tuple[str, ...]
    env: frozenset[str]  # the $VARs the stage's commands name


@dataclass(frozen=True)
class Plan:
    config: Path
    project: str
    series: tuple[str, ...]
    stages: tuple[Stage, ...]


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
        # Every series. A stack lane finds them now, in its stack paths; a frames
        # lane only once init has run (see project_series).
        series = stack_series(cfg, config) if run_series.lane_input(cfg) == "stack" else ()
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
    # series yet: any id renders it.
    example = series[0] if series else "TS_X"
    chain = [name for name, _ in run_series.build_steps(cfg, Path(project), example)]
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
    return Plan(config=config, project=project, series=series, stages=tuple(stages))


def stack_series(cfg: dict[str, Any], config: Path) -> tuple[str, ...]:
    """Every series a stack lane's [data] stack path matches, {series} its wildcard."""

    template = cfg.get("data", {}).get("stack")
    if not isinstance(template, str) or template.count("{series}") != 1:
        raise SystemExit(
            f"{config}: without [run] series, [data] stack must contain {{series}} once, "
            "so that the series can be found"
        )
    prefix, suffix = os.path.join(ROOT, template).split("{series}")
    found = sorted(
        path[len(prefix) : len(path) - len(suffix)]
        for path in glob.glob(glob.escape(prefix) + "*" + glob.escape(suffix))
    )
    if not found:
        raise SystemExit(f"{config}: no file matches [data] stack {template!r}")
    return tuple(found)


def project_series(project: str) -> tuple[str, ...]:
    """The series a project has manifests for: what init wrote, or import-tilt-stack."""

    return tuple(sorted(path.stem for path in (ROOT / project / "manifests").glob("*.json")))


def problems(cfg: dict[str, Any], lane: Plan, stages: tuple[Stage, ...]) -> list[str]:
    """What would make the submitted jobs fail at once, found on the submit node."""

    found = [
        f"${name} is not set (used by stage {stage.name!r})"
        for stage in stages
        for name in sorted(stage.env)
        if not os.environ.get(name)
    ]
    chain = run_series.build_steps(
        cfg, Path(lane.project), lane.series[0] if lane.series else "TS_X"
    )
    found += ctffind_problems(cfg, {
        name for stage in stages
        for name, _ in run_series.slice_steps(chain, stage.start, stage.end)
    })
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
    data = cfg.get("data", {})
    if init_command(cfg, lane) is not None:
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
    if not lane.series and not project_series(lane.project):
        found.append(f"{lane.project} exists but has no manifests to take every series from")
    manifests = ROOT / lane.project / "manifests"
    absent = [s for s in lane.series if not (manifests / f"{s}.json").is_file()]
    if absent:
        # init writes a new, empty registry, so it is never run on a project that
        # already has one: that would throw away every series processed so far.
        found.append(
            f"{lane.project} already exists but has no manifest for {', '.join(absent)}, "
            "and init would erase the project to add them; use a new project for them"
        )
    return found


def ctffind_problems(cfg: dict[str, Any], steps: set[str]) -> list[str]:
    """A CTFFIND 4 that estimate-ctf would find on PATH, as a problem.

    Only where the config names no CTFFIND of its own ([executables] ctffind):
    estimate-ctf then takes ctffind5, or else ctffind, from PATH, the job's PATH
    being this one (--export=ALL), and its backend refuses a CTFFIND 4 after the
    queue wait. Which one it is, its start-up banner says. A CTFFIND that cannot
    be found or asked here is noted rather than refused, since the job's
    environment may still find one.
    """

    if "estimate-ctf" not in steps or (cfg.get("executables") or {}).get("ctffind"):
        return []
    # The backend's order (cryoet_pipeline.backends.ctf._resolve_ctffind_executable).
    found = next((path for name in ("ctffind5", "ctffind") if (path := shutil.which(name))), None)
    if found is None:
        print("not checked which CTFFIND estimate-ctf will run: neither ctffind5 nor ctffind "
              "is on PATH here", file=sys.stderr)
        return []
    version = ctffind_version(found)
    if version is None:
        print(f"not checked which CTFFIND {found} is: it printed no version within "
              f"{CTFFIND_PROBE_SECONDS:g} s", file=sys.stderr)
        return []
    if int(version.split(".")[0]) >= 5:
        return []
    return [
        f"estimate-ctf would run {found}, which is CTFFIND {version}, and its backend answers "
        "CTFFIND 5's questions: name CTFFIND 5 in [executables] ctffind (\"$CTFFIND5\", as "
        "ryr1-bin8 does), or put ctffind5 on PATH ahead of it"
    ]


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
    there), so it runs here on the submit node rather than in a job. It is never
    run on an existing project, because it starts the registry empty.
    """
    if run_series.lane_input(cfg) != "frames":
        return None
    if (ROOT / lane.project / "artifacts.json").exists():
        return None
    data = cfg.get("data", {})
    command = ["cryoet", "init", "--frames", str(data.get("frames"))]
    if "frames_pattern" in data:
        command += ["--frames-pattern", str(data["frames_pattern"])]
    else:
        command += ["--mdocs", str(data.get("mdocs"))]
    if "pixel_size_angstrom" in data:
        command += ["--pixel-size", str(data["pixel_size_angstrom"])]
    command += ["--out", lane.project]
    for series in lane.series:
        command += ["--tilt-series", series]
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
    lane: Plan, stages: tuple[Stage, ...], dry_run: bool, overwrite: bool = False
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
        command = [
            "sbatch", "--parsable", "--export=ALL",
            *stage.sbatch, *after,
            f"--time={stage.time}", f"--job-name={prefix}-{stage.name}",
            str(LANE_SBATCH.relative_to(ROOT)),
        ]
        # Through the environment rather than --export=VAR=value, whose list is
        # comma-separated: SERIES holds spaces, and nothing has to survive a parser.
        if not lane.series and not dry_run:
            raise SubmitError("no series to submit", job_ids)
        env = {
            "PROJECT": lane.project,
            "CONFIG": _repo_relative(lane.config),
            # Unknown in a dry run of a new project without [run] series: init,
            # which finds them, has not run.
            "SERIES": " ".join(lane.series) or "<every series init finds>",
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
                command, cwd=ROOT, env={**inherited, **env},
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


def plan_record(
    cfg: dict[str, Any], lane: Plan, stages: tuple[Stage, ...], text: str, overwrite: bool
) -> dict[str, Any]:
    """What a dry run shows, as JSON. A submission from the assistant must repeat
    it exactly."""

    series = lane.series
    if not series and init_command(cfg, lane) is None:
        series = project_series(lane.project)
    return {
        "config": _repo_relative(lane.config),
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "project": lane.project,
        "series": list(series),
        "stages": [
            {"name": stage.name, "from": stage.start, "until": stage.end,
             "time": stage.time, "sbatch": list(stage.sbatch)}
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
    labelled = {name for name, command in steps if "--variant" in command}
    wanted = [
        name
        for stage in stages
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


def busy(
    cfg: dict[str, Any], lane: Plan, jobs: list[dict[str, str]], root: Path = ROOT,
    submitting: str = "this lane",
) -> list[str]:
    """Problems: jobs of this project, still queued or running, on series this
    lane (or a recipe, scripts/submit_recipe.py) would run."""

    series = lane.series
    if not series and init_command(cfg, lane) is None:
        series = project_series(lane.project)
    project = (root / lane.project).resolve()
    journaled = journal_series(read_journal(lane.project))
    found = []
    for job in jobs:
        # squeue names an array job's tasks "<id>_<task>", and sbatch printed "<id>".
        theirs = journaled.get(job["id"], journaled.get(job["id"].split("_", 1)[0]))
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
        if shared:
            found.append(
                f"{what} works on {', '.join(shared)}, which {submitting} would run too; "
                "two jobs on one series write the same files, so submit after it has ended"
            )
    return found


def queue_problems(
    targets: list[tuple[dict[str, Any], Plan]], submitting: str = "this lane"
) -> list[list[str]]:
    """Per lane, the jobs already queued or running on its series, as problems."""
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
    return [busy(cfg, lane, jobs or [], submitting=submitting) for cfg, lane in targets]


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
        help="what was asked and by whom, for the project's journal; required from the "
        "assistant",
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
    for config in args.configs:
        text, cfg = run_series.read_config(config)
        lane = plan(cfg, config)
        names = [stage.name for stage in lane.stages]
        if args.from_stage is not None and args.from_stage not in names:
            raise SystemExit(f"{config}: no stage {args.from_stage!r}; stages: {', '.join(names)}")
        first = names.index(args.from_stage) if args.from_stage else 0
        stages = lane.stages[first:]
        lanes.append(Submission(
            cfg, text, lane, stages, problems(cfg, lane, stages),
            plan_record(cfg, lane, stages, text, args.overwrite),
            recorded_runs(cfg, lane, stages) if args.overwrite else {},
        ))

    projects = [one.lane.project for one in lanes]
    if len(set(projects)) != len(projects):
        # Two jobs on the same series write the same files.
        raise SystemExit("two configs submit to the same project; submit them one after the other")
    # For the same reason, no job already queued or running may be on the series.
    targets = [(one.cfg, one.lane) for one in lanes]
    for one, taken in zip(lanes, queue_problems(targets), strict=True):
        one.problems += taken

    found = [f"{one.lane.config}: {problem}" for one in lanes for problem in one.problems]
    for problem in found:
        print(f"{'would refuse' if args.dry_run else 'refused'}: {problem}", file=sys.stderr)
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
        init = init_command(cfg, lane)
        try:
            if init is not None:
                print(("would run: " if args.dry_run else "") + shlex.join(init))
                if not args.dry_run:
                    subprocess.run(init, cwd=ROOT, check=True)
            if not lane.series and (init is None or not args.dry_run):
                # Every series: the ones init just wrote manifests for, or the ones
                # the project already has.
                lane = replace(lane, series=project_series(lane.project))
                print(f"  series: {' '.join(lane.series)}")
            ids = submit(lane, stages, args.dry_run, args.overwrite)
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
