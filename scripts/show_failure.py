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

What counts as failed, among the runs this lane sees (scripts/show_project.py
says which those are):

* a step whose last run exited non-zero;
* a job that stopped in its preflight, before any step ran, unless one of its
  steps has run for the series since;
* a job whose log stops in the middle of a step: still running, or stopped by
  Slurm (time limit, memory, node). sacct, where it can be run, says which.

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
    "prepare-tilt-series": ("logs/motioncor3/{s}", "motion/motioncor3/{s}", "stacks/{s}"),
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


def program_logs(
    root: Path, project: Path, series: str, step: str,
    since: datetime | None, until: datetime | None,
) -> tuple[list[dict[str, Any]], int]:
    """The logs the step's programs wrote while it ran, newest first, and how many
    more there were than are shown."""
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
    newest = sorted(found.items(), key=lambda item: item[1], reverse=True)
    shown = [
        {
            "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "modified": modified.isoformat(timespec="seconds"),
            "lines": log_excerpt(_read_lines(path)),
        }
        for path, modified in newest[:MAX_LOGS]
    ]
    return shown, max(0, len(newest) - MAX_LOGS)


def _job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: job[key] for key in ("id", "name", "log", "started", "outcome", "slurm")
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
    labelled = {name for name, command in steps if "--variant" in command}
    runs = show_project.last_runs(project, set(chosen), labelled, lane["variant"])
    jobs = show_project.job_logs(root, project_name,
                                 show_project.slurm_logs(project) if slurm else ())
    jobs.sort(key=lambda job: int(job["id"]) if str(job.get("id", "")).isdigit() else 0)
    if slurm:
        names = {f"{Path(project_name).name}-{stage.get('name')}"
                 for stage in cfg["run"].get("stage") or []}
        show_project.ask_slurm(jobs, names)

    found: dict[tuple[str, str], dict[str, Any]] = {}
    rerun: dict[tuple[str, str], dict[str, Any]] = {}

    def wanted(one: str, name: str | None) -> bool:
        return one in chosen and (step is None or name == step)

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
        if "preflight" in job:
            stop = next((index for index, line in enumerate(lines)
                         if line.startswith("preflight: ")), len(lines) - 1)
            excerpt = [_line(lines, index) for index in sorted(
                set(_errors(lines, range(stop + 1))) | set(range(max(0, stop - TAIL), stop + 1))
            )]
            for one in job.get("series", []):
                passed = ran_since(one, job)
                if passed is None:
                    add(one, "preflight", kind="preflight", problem=job["preflight"],
                        job=_job(job), slurm_lines=excerpt)
                elif wanted(one, "preflight"):
                    # A later job got past the preflight: on record, no longer a failure.
                    rerun.setdefault((one, "preflight"), {
                        "series": one, "step": "preflight", "jobs": [], "passed_at": passed,
                    })["jobs"].append(_job(job))
        stopped = in_progress(lines) if not job["failed"] and "finished" not in job else None
        if stopped is not None:
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
        for record in show_project.qc_records(root, project, set(chosen), lane["variant"])
        if record["status"] == "fail"
        and wanted(record["series"], record["step"] or record["qc_type"])
    ]
    return {"project": project_name, "lane": lane.get("name"), "variant": lane["variant"],
            "failures": result, "rerun_since": sorted(rerun.values(), key=place),
            "qc_failures": sorted(qc_failures, key=place)}


def _describe(entry: dict[str, Any]) -> str:
    if entry["kind"] == "preflight":
        return f"preflight: {entry['problem']}"
    if entry["kind"] == "interrupted":
        slurm = entry.get("job", {}).get("slurm")
        state = f"; Slurm: {slurm['state']}" if slurm else "; still running, or stopped by Slurm"
        return f"stopped during the step, no exit line{state}"
    when = entry.get("finished_at")
    return f"crashed, exit {entry.get('exit_code')}" + (f", {when}" if when else "")


def _excerpt(lines: list[dict[str, Any]], indent: str) -> list[str]:
    out: list[str] = []
    previous = None
    for line in lines:
        if previous is not None and line["line"] > previous + 1:
            out.append(f"{indent}{'…':>6}")
        out.append(f"{indent}{line['line']:>6}  {line['text']}")
        previous = line["line"]
    return out


def render(result: dict[str, Any]) -> list[str]:
    count = len(result["failures"])
    lines = [f"{result['project']}: lane {result['lane']} (variant {result['variant']}), "
             f"{count} failed step{'' if count == 1 else 's'}"]
    if not count:
        lines.append("no failed step on record for this lane")
    for entry in result["failures"]:
        lines += ["", f"{entry['series']}  {entry['step']}  {_describe(entry)}"]
        job = entry.get("job")
        if job:
            lines.append(f"  {job['log']} (job {job.get('name', '?')} {job.get('id', '?')})")
            lines += _excerpt(entry.get("slurm_lines", []), "  ")
        for log in entry.get("program_logs", []):
            lines.append(f"  {log['path']} (written {log['modified']})")
            lines += _excerpt(log["lines"], "  ")
        if entry.get("more_logs"):
            lines.append(f"  ({entry['more_logs']} more logs from the same run not shown)")
    if result.get("rerun_since"):
        lines += ["", "failed before, and passed when run again since:"]
        for entry in result["rerun_since"]:
            jobs = ", ".join(
                f"{job.get('name', '?')} {job.get('id', '?')}" for job in entry["jobs"]
            )
            passed = f"; passed {entry['passed_at']}" if entry.get("passed_at") else ""
            lines.append(f"  {entry['series']}  {entry['step']}  failed in {jobs}{passed}")
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
