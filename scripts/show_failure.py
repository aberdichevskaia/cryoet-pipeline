#!/usr/bin/env python3
"""What went wrong in a lane's failed steps: for each, the lines that matter from
its Slurm log and from the logs of the programs it ran, with their line numbers.

    python3 scripts/show_failure.py configs/lanes/ryr1-bin8.toml
    python3 scripts/show_failure.py configs/lanes/ryr1-bin8.toml --series 161
    python3 scripts/show_failure.py configs/lanes/ryr1-bin8.toml --json

Read-only, and bounded: never a whole log. It shows the step's command, the lines
that look like errors (the first and the last of them, since the real error is
often not at the end), and the last lines before the step stopped. Paths and line
numbers say where to read on.

What counts as failed, among the runs and jobs this lane sees
(scripts/show_project.py says which those are: another lane's job, for the
shared steps only):

* a step whose last run exited non-zero;
* a job that stopped in its preflight, before any step ran, unless one of its
  steps has run for the series since. Another lane's job counts only if every
  step it was to run is shared. That is a `preflight:` line, or a log that
  stops inside lane.sbatch's checks with neither that nor `preflight   ok`, as
  a command failing under set -e leaves it; a job Slurm still runs is left out;
* a job whose log stops in the middle of a step: still running, or stopped by
  Slurm (time limit, memory, node). sacct, where it can be run, says which.

Listed apart, since no step or series is known for them: the lane's jobs that
stopped before lane.sbatch printed its header, so before the preflight and any
step. What lane.sbatch does first, under set -e, can stop a job within a second
and leave only an error in its log: the environment's activation, IMOD's
startup script, the config read by run_series.py. On cryoem.se, 26 November
2026, such a job read here as "0 failed steps". They are the logs named for one
of the lane's lane.sbatch stages (<project>-<stage>) that hold no "project" line,
and nothing run_series.py prints after it, and, where sacct can be run, the
lane's jobs that Slurm ended in a failed state and whose log is missing; a job Slurm still runs is left out, and one that a
later job of the same stage followed is listed as such. The name is all that
ties such a log to the project: another project whose folder has the same name
shares it.

The jobs are found by their Slurm logs, in the repository root and, where sacct
can be run, wherever else a job of the project was submitted from.

Two kinds of record are listed after the failures, without excerpts, so that an
answer of "no failed step" is not read as "nothing ever failed": a step that
failed in a job and passed when run again since, or a preflight a later job got
past, with the jobs it failed in; and a step that ran but whose result failed QC,
with the QC file and its first warning. scripts/show_project.py has every QC
verdict, the warnings included.

A program's log is found by where its step writes and by when: the logs written
while the step ran. That is a stand-in for logs recorded on the run itself, the
second half of roadmap section 6.

Disposable, like scripts/run_series.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_project  # noqa: E402

# Where each step's programs write their logs, relative to the project, with {s}
# for the series: the folders docs/agent/pipeline_map.md lists.
STEP_DIRS: dict[str, tuple[str, ...]] = {
    "prepare-tilt-series": (
        "logs/motioncor3/{s}", "motion/motioncor3/{s}", "stacks/{s}", "qc/{s}/motion",
    ),
    "import-tilt-stack": ("stacks/{s}",),
    "align-tilt-series": ("alignments/{s}",),
    "qc-coarse-alignment": ("qc/{s}/coarse_alignment",),
    "generate-fiducial-seed": ("fiducials/{s}",),
    "track-fiducials": ("fiducials/{s}",),
    "fine-align-tilt-series": ("alignments/{s}/fine",),
    "build-ctf-stack": ("aligned/{s}",),
    "estimate-ctf": ("ctf/{s}", "qc/{s}/ctf", "qc/{s}/defocus_hint"),
    "build-final-aligned-stack": ("aligned/{s}",),
    "correct-ctf": ("ctf/{s}", "qc/{s}/ctf"),
    "reconstruct-tomogram": ("tomograms/{s}", "qc/{s}/reconstruction"),
    "segment-tomogram": ("segmentations/{s}", "qc/{s}/segmentation"),
    "prepare-template": ("templates/{s}", "qc/{s}/template"),
    "pick-particles": ("picks/{s}", "qc/{s}/picking"),
    "extract-particles": ("particles/{s}", "qc/{s}/particle_extraction"),
}

# Lines that look like an error, in any of the programs the pipeline runs.
_ERROR = re.compile(
    r"error|exception|traceback|segmentation fault|core dumped|\bkilled\b|out of memory"
    r"|no kernel image|no such file|not found|permission denied|disk quota|no space left"
    r"|refus|abort|fatal|cannot|failed",
    re.IGNORECASE,
)
_BLOCK = re.compile(r"^# ---- (?P<series>\S+) \(.*\) ----$")

# Bounds, so that an excerpt stays something to read.
MAX_ERRORS = 12
TAIL = 6
MAX_LOGS = 5
MAX_WIDTH = 300
# Clock and file-system skew around a step's run.
SLACK = timedelta(minutes=2)
# A lane job's log, as lane.sbatch's --output names it: slurm-%x-%j.log.
_LOG_NAME = re.compile(r"^slurm-(?P<name>.+)-(?P<id>\d+)\.log$")
# Slurm's states of a job not yet over: no header yet is no failure yet.
_NOT_OVER = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING",
             "SUSPENDED"}
# Slurm's states of a job that ended without a failure.
_ENDED_WELL = {"COMPLETED"}
# What lane.sbatch prints when its checks pass, and the problem given for a log
# that stops in them without saying why.
_CHECKS_PASSED = "preflight   ok"
CHECKS_STOPPED = ("stopped in lane.sbatch's checks, before any step, without a preflight "
                  "line: a command failed under set -e; see the log's last lines")


def _state(job: dict[str, Any]) -> str:
    """Slurm's state of a job, its first word ("CANCELLED by 1234" is CANCELLED);
    empty when Slurm was not asked."""
    return str((job.get("slurm") or {}).get("state") or "").partition(" ")[0]


def _time(text: Any) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(errors="replace").splitlines()
    except OSError:
        return []


def _line(lines: list[str], index: int) -> dict[str, Any]:
    text = lines[index]
    return {"line": index + 1, "text": text if len(text) <= MAX_WIDTH else text[:MAX_WIDTH] + " …"}


def _errors(lines: list[str], indices: range) -> list[int]:
    """The lines that look like errors: the first ones and the last ones."""
    found = [index for index in indices if _ERROR.search(lines[index])]
    if len(found) <= MAX_ERRORS:
        return found
    half = MAX_ERRORS // 2
    return found[:half] + found[-half:]


def slurm_window(lines: list[str], series: str, step: str) -> tuple[int, int] | None:
    """Where a step of a series runs in a lane job's log, as indices: from its
    `# <step>` line to its FAILED line, or to the end of the series' block."""
    blocks = [index for index, line in enumerate(lines)
              if (match := _BLOCK.match(line)) and match["series"] == series]
    if not blocks:
        return None
    begin = blocks[-1]
    end = next((index for index in range(begin + 1, len(lines)) if _BLOCK.match(lines[index])),
               len(lines))
    headers = [index for index in range(begin, end) if lines[index] == f"# {step}"]
    if not headers:
        return None
    start = headers[-1]
    failed = next((index for index in range(start, end)
                   if lines[index].startswith(f"# FAILED at {step} ")), None)
    return start, failed + 1 if failed is not None else end


def slurm_excerpt(lines: list[str], window: tuple[int, int]) -> list[dict[str, Any]]:
    """The step's header and command, its error lines, and its last lines."""
    start, end = window
    keep = {start}
    if start + 1 < end and lines[start + 1].startswith("cryoet "):
        keep.add(start + 1)
    keep.update(_errors(lines, range(start + 2, end)))
    keep.update(range(max(start + 2, end - TAIL), end))
    return [_line(lines, index) for index in sorted(keep)]


def in_progress(lines: list[str]) -> tuple[str, str] | None:
    """The series and step a job's log stops in, when the step never finished:
    a `# <step>` line with neither its time line nor a FAILED line after it."""
    blocks = [index for index, line in enumerate(lines) if _BLOCK.match(line)]
    if not blocks:
        return None
    begin = blocks[-1]
    match = _BLOCK.match(lines[begin])
    assert match is not None
    headers = [index for index in range(begin + 1, len(lines))
               if re.fullmatch(r"# [a-z]+(?:-[a-z]+)+", lines[index])]
    if not headers:
        return None
    step = lines[headers[-1]][2:]
    ended = any(line.startswith((f"# {step}: ", f"# FAILED at {step} "))
                for line in lines[headers[-1] + 1:])
    return None if ended else (match["series"], step)


def _command_runs(lines: list[str]) -> list[tuple[int, int, int | None]]:
    """The commands in a log written by backends/command_log.py: for each, the
    index of its `$ ` line, the index of its exit code, and the code. Empty for a
    log in any other format."""
    runs: list[tuple[int, int, int | None]] = []
    command = None
    for index, line in enumerate(lines):
        if line.startswith("$ "):
            command = index
        elif line == "[exit_code]" and command is not None and index + 1 < len(lines):
            try:
                code: int | None = int(lines[index + 1].strip())
            except ValueError:
                code = None
            runs.append((command, index + 1, code))
            command = None
    return runs


def log_excerpt(lines: list[str]) -> list[dict[str, Any]]:
    """For a command log: each command that exited non-zero, with its error lines,
    its last lines and its exit code. For any other log: its error lines and its
    last lines."""
    failed = [(command, code_line) for command, code_line, code in _command_runs(lines)
              if code not in (0, None)]
    keep: set[int] = set()
    if failed:
        for command, code_line in failed:
            keep.update({command, code_line - 1, code_line})
            keep.update(_errors(lines, range(command + 1, code_line - 1)))
            keep.update(range(max(command + 1, code_line - 1 - TAIL), code_line - 1))
    else:
        keep.update(_errors(lines, range(len(lines))))
        keep.update(range(max(0, len(lines) - TAIL), len(lines)))
    return [_line(lines, index) for index in sorted(keep)]


def step_logs(
    project: Path, series: str, step: str, since: datetime | None, until: datetime | None,
) -> list[tuple[Path, datetime]]:
    """The logs the step's programs last wrote between since and until, with when,
    newest first."""
    found: dict[Path, datetime] = {}
    for folder in STEP_DIRS.get(step, ()):
        directory = project / folder.format(s=series)
        if not directory.is_dir():
            continue
        for path in directory.glob("*.log"):
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            if since is not None and modified < since - SLACK:
                continue
            if until is not None and modified > until + SLACK:
                continue
            found[path] = modified
    return sorted(found.items(), key=lambda item: item[1], reverse=True)


def program_logs(
    root: Path, project: Path, series: str, step: str,
    since: datetime | None, until: datetime | None,
) -> tuple[list[dict[str, Any]], int]:
    """The logs the step's programs wrote while it ran, newest first, and how many
    more there were than are shown."""
    newest = step_logs(project, series, step, since, until)
    shown = [
        {
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "modified": modified.isoformat(timespec="seconds"),
            "lines": log_excerpt(_read_lines(path)),
        }
        for path, modified in newest[:MAX_LOGS]
    ]
    return shown, max(0, len(newest) - MAX_LOGS)


def before_start(
    cfg: dict[str, Any], root: Path, elsewhere: list[Path], jobs: list[dict[str, Any]],
    chain: list[str], step: str | None = None, slurm: bool = True,
) -> list[dict[str, Any]]:
    """The lane's jobs that stopped before lane.sbatch's header, each with its log's
    error lines and last lines, Slurm's state where sacct can be run, and the
    later job of the same stage, if one started after it (`jobs`, the jobs with a
    header). With `step`, only the stages that were to run it."""

    prefix = Path(cfg["run"]["project"]).name
    stages: dict[str, list[str]] = {}
    for stage in cfg["run"].get("stage") or []:
        start, end = stage.get("from"), stage.get("until")
        if start not in chain or end not in chain:
            # The MRA's stage: its step is in no series' chain, and its job,
            # mra_port_run.sbatch's, prints its project line later than lane.sbatch.
            continue
        stages[f"{prefix}-{stage.get('name')}"] = chain[chain.index(start):chain.index(end) + 1]
    if step is not None:
        stages = {name: steps for name, steps in stages.items() if step in steps}

    found: dict[str, dict[str, Any]] = {}
    here = root.glob(f"slurm-{glob.escape(prefix)}-*.log")
    paths = {path.resolve(): path for path in [*elsewhere, *here]}
    for path in sorted(paths.values(), key=lambda path: path.name):
        match = _LOG_NAME.match(path.name)
        if not match or match["name"] not in stages:
            continue
        shown = str(path.relative_to(root) if path.is_relative_to(root) else path)
        entry: dict[str, Any] = {"id": match["id"], "name": match["name"], "log": shown}
        if not path.is_file():
            entry["missing"] = True  # kept only if Slurm says the job failed
        else:
            job = show_project.read_job_log(path, root)
            if job is None or "project" in job:
                continue
            lines = _read_lines(path)
            if job["failed"] or any(_BLOCK.match(line) or show_project._STEP.match(line)
                                    for line in lines):
                # What run_series prints, after the header: the log lost its header
                # (a copy of its last lines, as eval case E01 keeps), the job did not.
                continue
            entry["written"] = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            entry["lines"] = [_line(lines, index) for index in sorted(
                set(_errors(lines, range(len(lines))))
                | set(range(max(0, len(lines) - TAIL), len(lines)))
            )]
        found[entry["id"]] = entry

    if slurm and found:
        for parts in show_project._slurm(["sacct", "-n", "-P", "-X", "-j", ",".join(found),
                                         "-o", "JobID,State,ExitCode,Elapsed"]):
            if len(parts) == 4 and parts[0] in found:
                found[parts[0]]["slurm"] = {
                    "state": parts[1], "exit_code": parts[2], "elapsed": parts[3],
                }
    result = []
    for entry in found.values():
        state = _state(entry)
        if state in _NOT_OVER:
            continue  # still starting: its header may yet come
        if entry.get("missing") and (not state or state in _ENDED_WELL):
            continue  # a log not found says nothing on its own
        written = _time(entry.get("written"))
        later = [
            job for job in jobs
            if job.get("name") == entry["name"] and written is not None
            and (started := _time(job.get("started"))) is not None and started > written
        ]
        if later:
            entry["later"] = _job(later[0])
        result.append(entry)
    return sorted(result, key=lambda entry: int(entry["id"]))


def _job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in ("id", "name", "lane", "log", "started", "outcome", "slurm")
            if key in job}


def failures(
    cfg: dict[str, Any], root: Path, series: list[str] | None = None,
    step: str | None = None, slurm: bool = True,
) -> dict[str, Any]:
    """Every failed step of the lane, each with its excerpts."""
    project_name = cfg["run"]["project"]
    project = Path(project_name) if Path(project_name).is_absolute() else root / project_name
    lane = cfg["lane"]
    chosen = series or show_project.lane_series(cfg, project)
    steps = run_series.build_steps(cfg, project, chosen[0] if chosen else "TS_X")
    chain = [name for name, _ in steps]
    labelled = show_project.step_labels(steps)
    runs = show_project.last_runs(project, set(chosen), labelled, lane["variant"])
    elsewhere = show_project.slurm_logs(project) if slurm else []
    jobs = show_project.job_logs(root, project_name, elsewhere)
    jobs.sort(key=lambda job: int(job["id"]) if str(job.get("id", "")).isdigit() else 0)
    if slurm:
        names = {f"{Path(project_name).name}-{stage.get('name')}"
                 for stage in cfg["run"].get("stage") or []}
        show_project.ask_slurm(jobs, names)

    found: dict[tuple[str, str], dict[str, Any]] = {}
    rerun: dict[tuple[str, str], dict[str, Any]] = {}

    def wanted(one: str, name: str | None) -> bool:
        return one in chosen and (step is None or name == step)

    def sees(job: dict[str, Any], name: str, label: str | None = None) -> bool:
        return show_project.lane_sees(job, name, lane.get("name"), labelled, label)

    def add(one: str, name: str, **fields: Any) -> None:
        if wanted(one, name):
            found.setdefault((one, name), {"series": one, "step": name}).update(fields)

    def ran_since(one: str, job: dict[str, Any]) -> str | None:
        """When one of the job's steps last ran for the series, if that was after
        the job started; None if none did, or the log names no steps."""
        started = _time(job.get("started"))
        if started is None:
            return None
        later = [
            when for name in job.get("steps", [])
            if (when := _time(runs.get((one, name), {}).get("finished_at"))) and when > started
        ]
        return max(later).isoformat() if later else None

    for (one, name), run in runs.items():
        if run.get("exit_code") != 0:
            add(one, name, kind="crashed", exit_code=run.get("exit_code"),
                finished_at=run.get("finished_at"), seconds=run.get("seconds"))

    for job in jobs:
        lines = _read_lines(root / job["log"])
        started = _time(job.get("started"))
        for failure in job["failed"]:
            key = (failure["series"], failure["step"])
            if not sees(job, failure["step"], failure.get("label")):
                continue  # another lane's run of a step each lane labels for itself
            if runs.get(key, {}).get("exit_code") == 0:
                # Rerun since, and the rerun worked: no longer a failure, but on record.
                if wanted(*key):
                    rerun.setdefault(key, {
                        "series": key[0], "step": key[1], "jobs": [],
                        "passed_at": runs[key].get("finished_at"),
                    })["jobs"].append(_job(job))
                continue
            window = slurm_window(lines, *key)
            add(*key, kind="crashed", exit_code=failure["exit_code"], job=_job(job),
                slurm_lines=slurm_excerpt(lines, window) if window else [])
        # A stop before the first step is about every step the job was to run, the
        # whole chain if its log does not say: another lane's, if any is labelled.
        problem = job.get("preflight")
        if (problem is None and not job["failed"] and "finished" not in job
                and _CHECKS_PASSED not in lines
                and not any(_BLOCK.match(line) for line in lines)
                and _state(job) not in _NOT_OVER):
            problem = CHECKS_STOPPED
        if problem is not None and all(sees(job, name) for name in job.get("steps") or chain):
            stop = next((index for index, line in enumerate(lines)
                         if line.startswith("preflight: ")), len(lines) - 1)
            excerpt = [_line(lines, index) for index in sorted(
                set(_errors(lines, range(stop + 1))) | set(range(max(0, stop - TAIL), stop + 1))
            )]
            for one in job.get("series", []):
                passed = ran_since(one, job)
                if passed is None:
                    add(one, "preflight", kind="preflight", problem=problem,
                        job=_job(job), slurm_lines=excerpt)
                elif wanted(one, "preflight"):
                    # A later job got past the preflight: on record, no longer a failure.
                    rerun.setdefault((one, "preflight"), {
                        "series": one, "step": "preflight", "jobs": [], "passed_at": passed,
                    })["jobs"].append(_job(job))
        stopped = in_progress(lines) if not job["failed"] and "finished" not in job else None
        if stopped is not None and sees(job, stopped[1]):
            run = runs.get(stopped)
            finished = _time(run.get("finished_at")) if run else None
            if run is None or started is None or finished is None or finished < started:
                window = slurm_window(lines, *stopped)
                add(*stopped, kind="interrupted", exit_code=None, job=_job(job),
                    started_at=job.get("started"),
                    slurm_lines=slurm_excerpt(lines, window) if window else [])

    order = {name: index for index, name in enumerate(["preflight", *chain])}
    result = sorted(found.values(), key=lambda entry: (
        chosen.index(entry["series"]), order.get(entry["step"], len(order))))
    for entry in result:
        if entry["step"] == "preflight":
            continue
        until = _time(entry.get("finished_at"))
        since = until - timedelta(seconds=float(entry.get("seconds") or 0)) if until else None
        if since is None:
            since = _time(entry.get("started_at") or entry.get("job", {}).get("started"))
        logs, more = program_logs(root, project, entry["series"], entry["step"], since, until)
        entry["program_logs"] = logs
        entry["more_logs"] = more
    def place(entry: dict[str, Any]) -> tuple[int, int]:
        return chosen.index(entry["series"]), order.get(entry["step"], len(order))

    # Steps that ran but whose result failed QC: not crashes, and not excerpted.
    qc_failures = [
        {"series": record["series"], "step": record["step"] or record["qc_type"],
         "path": record["path"], "warnings": record["warnings"][:1]}
        for record in show_project.qc_records(
            root, project, set(chosen), lane["variant"], labelled
        )
        if record["status"] == "fail"
        and wanted(record["series"], record["step"] or record["qc_type"])
    ]
    early = before_start(cfg, root, elsewhere, jobs, chain, step=step, slurm=slurm)
    return {"project": project_name, "lane": lane.get("name"), "variant": lane["variant"],
            "failures": result, "rerun_since": sorted(rerun.values(), key=place),
            "qc_failures": sorted(qc_failures, key=place),
            "before_start": [entry for entry in early if "later" not in entry],
            "before_start_since": [entry for entry in early if "later" in entry]}


def _describe(entry: dict[str, Any]) -> str:
    if entry["kind"] == "preflight":
        return f"preflight: {entry['problem']}"
    if entry["kind"] == "interrupted":
        slurm = entry.get("job", {}).get("slurm")
        state = f"; Slurm: {slurm['state']}" if slurm else "; still running, or stopped by Slurm"
        return f"stopped during the step, no exit line{state}"
    when = entry.get("finished_at")
    return f"crashed, exit {entry.get('exit_code')}" + (f", {when}" if when else "")


def _job_name(job: dict[str, Any], lane: str | None) -> str:
    """A job's name and ID, and its lane if another: lanes of one project share
    job names."""
    other = f" of lane {job['lane']}" if job.get("lane") not in (None, lane) else ""
    return f"{job.get('name', '?')} {job.get('id', '?')}{other}"


def _excerpt(lines: list[dict[str, Any]], indent: str) -> list[str]:
    out: list[str] = []
    previous = None
    for line in lines:
        if previous is not None and line["line"] > previous + 1:
            out.append(f"{indent}{'…':>6}")
        out.append(f"{indent}{line['line']:>6}  {line['text']}")
        previous = line["line"]
    return out


def _slurm_said(entry: dict[str, Any]) -> str:
    slurm = entry.get("slurm")
    if not slurm:
        return ""
    return f"; Slurm: {slurm['state']}, exit {slurm['exit_code']}, after {slurm['elapsed']}"


def render(result: dict[str, Any]) -> list[str]:
    count = len(result["failures"])
    early = result.get("before_start") or []
    head = (f"{result['project']}: lane {result['lane']} (variant {result['variant']}), "
            f"{count} failed step{'' if count == 1 else 's'}")
    if early:
        head += (f", {len(early)} job{'' if len(early) == 1 else 's'} stopped before the "
                 "pipeline started")
    lines = [head]
    if not count and not early:
        lines.append("no failed step on record for this lane")
    if early:
        lines += ["", "stopped before lane.sbatch's header, so before the preflight and any "
                  "step (the environment, IMOD, the config):"]
        for entry in early:
            if entry.get("missing"):
                lines.append(f"  job {entry['name']} {entry['id']}: no log at {entry['log']}"
                             f"{_slurm_said(entry)}")
                continue
            lines.append(f"  {entry['log']} (job {entry['name']} {entry['id']}, written "
                         f"{entry['written']}{_slurm_said(entry)})")
            lines += _excerpt(entry["lines"], "  ") if entry["lines"] else ["      (empty)"]
    for entry in result["failures"]:
        lines += ["", f"{entry['series']}  {entry['step']}  {_describe(entry)}"]
        job = entry.get("job")
        if job:
            lines.append(f"  {job['log']} (job {_job_name(job, result['lane'])})")
            lines += _excerpt(entry.get("slurm_lines", []), "  ")
        for log in entry.get("program_logs", []):
            lines.append(f"  {log['path']} (written {log['modified']})")
            lines += _excerpt(log["lines"], "  ")
        if entry.get("more_logs"):
            lines.append(f"  ({entry['more_logs']} more logs from the same run not shown)")
    if result.get("rerun_since"):
        lines += ["", "failed before, and passed when run again since:"]
        for entry in result["rerun_since"]:
            jobs = ", ".join(_job_name(job, result["lane"]) for job in entry["jobs"])
            passed = f"; passed {entry['passed_at']}" if entry.get("passed_at") else ""
            lines.append(f"  {entry['series']}  {entry['step']}  failed in {jobs}{passed}")
    if result.get("before_start_since"):
        lines += ["", "stopped before lane.sbatch's header, and a later job of the same stage "
                  "started since:"]
        for entry in result["before_start_since"]:
            later = entry["later"]
            lines.append(f"  {entry['name']} {entry['id']}, then {later.get('name')} "
                         f"{later.get('id')} ({later.get('log')})")
    if result.get("qc_failures"):
        lines += [
            "", "ran, but the result failed QC (scripts/show_project.py lists every verdict):"
        ]
        for entry in result["qc_failures"]:
            lines.append(f"  {entry['series']}  {entry['step']}  {entry['path']}")
            lines += [f"      {warning}" for warning in entry["warnings"]]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="the lane config the project was run with")
    parser.add_argument("--series", action="append", default=None,
                        help="only this series; repeat for several")
    parser.add_argument("--step", default=None, help="only this step")
    parser.add_argument("--json", action="store_true", help="print the excerpts as JSON")
    parser.add_argument("--no-slurm", action="store_true",
                        help="do not ask sacct and squeue; read the logs in the repository "
                             "root only")
    args = parser.parse_args()

    _, cfg = run_series.read_config(args.config)
    run_series.check_keys(cfg, args.config)
    if not isinstance(cfg.get("run"), dict) or not cfg["run"].get("project"):
        raise SystemExit(f"{args.config}: no [run] project")
    result = failures(cfg, ROOT, series=args.series, step=args.step, slurm=not args.no_slurm)
    if args.json:
        print(json.dumps({"config": str(args.config), **result}, indent=2))
    else:
        print("\n".join(render(result)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
