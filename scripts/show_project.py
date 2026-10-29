#!/usr/bin/env python3
"""Where a lane stands, for every series at once: each step's last run and QC
verdict, the warnings, and the lane's Slurm jobs.

    python3 scripts/show_project.py configs/lanes/ryr1-bin8.toml
    python3 scripts/show_project.py configs/lanes/ryr1-bin8.toml --series 097 --series 161
    python3 scripts/show_project.py configs/lanes/ryr1-bin8.toml --json

Read-only. It joins what is already written down and decides nothing:

* `step_timings.jsonl`, which run_series.py --execute appends to, says which steps
  ran and how they exited. A step that crashes leaves no artifact, so this is the
  only place a crash shows;
* the registry's QC records say how good the result of a step that did run is;
* the lane's Slurm logs say what its jobs did. lane.sbatch writes a job's log in
  the folder the job was submitted from: the repository root for the jobs
  submit_lanes.py submits, but a job submitted by hand from elsewhere logs
  there. Where sacct can be run, it says which folder each job ran in, and adds
  what only Slurm knows (a time limit, memory, a node failure); squeue adds the
  jobs still waiting.

One project can hold several lanes. Steps before build-final-aligned-stack are
shared, so every lane's runs of them count. The steps after it each lane labels
for itself: their timing rows count only under this lane's variant, and their
failures in a job only if the job ran this lane. A job's log names its lane on
each series' `# ---- <series> (<lane>) ----` line; a job that stopped before
its first step is known by the lane config in its header. A timing row written
before rows named their lane, and a job whose lane cannot be told, count for
every lane.

Disposable, like scripts/run_series.py: a stand-in, for a person and for the
terminal assistant, until the run entity records all of this.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402

# The step each QC type belongs to, so that its verdict lands in that step's
# column. A type not listed here (denoising, pick filters, averages) is reported
# under its own name, after the chain.
QC_STEP = {
    "coarse_alignment_report": "qc-coarse-alignment",
    "fiducial_seed": "generate-fiducial-seed",
    "fiducial_tracking": "track-fiducials",
    "fine_alignment": "fine-align-tilt-series",
    "ctf_estimation": "estimate-ctf",
    "ctf_correction": "correct-ctf",
    "fine_reconstruction": "reconstruct-tomogram",
    "tomogram_segmentation": "segment-tomogram",
    "template_preparation": "prepare-template",
    "particle_picking": "pick-particles",
    "particle_extraction": "extract-particles",
}
# QC records that are pictures rather than verdicts.
_PICTURES = frozenset({"coarse_alignment_preview"})
# Column headers, short enough for a terminal.
SHORT = {
    "prepare-tilt-series": "motion",
    "import-tilt-stack": "import",
    "align-tilt-series": "align",
    "qc-coarse-alignment": "coarse",
    "generate-fiducial-seed": "seed",
    "track-fiducials": "track",
    "fine-align-tilt-series": "fine",
    "build-ctf-stack": "ctfstk",
    "estimate-ctf": "ctf",
    "build-final-aligned-stack": "stack",
    "correct-ctf": "ctfcor",
    "reconstruct-tomogram": "recon",
    "segment-tomogram": "membr",
    "prepare-template": "templ",
    "pick-particles": "pick",
    "visualize-picks": "view",
    "extract-particles": "extract",
}
# How bad a state is, for the worst of several QC records of one step.
_SEVERITY = {"none": 0, "ok": 1, "unknown": 2, "warning": 3, "fail": 4, "crashed": 5}
_CELL = {"none": "·", "ok": "ok", "unknown": "?", "warning": "W", "fail": "F"}

# What lane.sbatch and run_series.py write into a job's log.
_JOB = re.compile(r"^job\s+(?P<name>\S+) \((?P<id>\d+)\)")
_PROJECT = re.compile(r"^project\s+(?P<project>\S+)(?:\s+config\s+(?P<config>\S+))?")
_SERIES = re.compile(r"^series\s+(?P<series>.*)$")
_STEPS = re.compile(r"^steps\s+(?P<steps>.*)$")
_STARTED = re.compile(r"^started\s+(?P<when>\S+)")
_FINISHED = re.compile(r"^finished\s+(?P<when>\S+)")
_PREFLIGHT = re.compile(r"^preflight: (?P<problem>.*)$")
_BLOCK = re.compile(r"^# ---- (?P<series>\S+) \((?P<lane>.*)\) ----$")
_FAILED =re.compile(r"^# FAILED at (?P<step>\S+) \(exit (?P<code>-?\d+)\) for (?P<series>\S+)\.$")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _resolve(root: Path, path: str | Path) -> Path:
    """Paths in a config and in the registry are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else root / path


def _time(text: Any) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _timings(project: Path) -> list[dict[str, Any]]:
    """The rows of the project's step_timings.jsonl."""
    try:
        lines = (project / "step_timings.jsonl").read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut short by a job that was killed while writing it
        if isinstance(row, dict):
            rows.append(row)
    return rows


def lane_series(cfg: dict[str, Any], project: Path) -> list[str]:
    """The series the lane lists, in its order, then any other the project has a
    manifest for."""
    listed = [str(series) for series in cfg.get("run", {}).get("series") or []]
    found = sorted(path.stem for path in (project / "manifests").glob("*.json"))
    return listed + [series for series in found if series not in listed]


def last_runs(
    project: Path, series: set[str], labelled: set[str], variant: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest run of each step of each series, among the runs this lane sees."""
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _timings(project):
        if row.get("series") not in series:
            continue
        step = str(row.get("step"))
        if step in labelled and row.get("variant", variant) != variant:
            continue  # another lane's run of a step each lane labels for itself
        runs[(row["series"], step)] = row
    return runs


def lane_sees(job: dict[str, Any], step: str, lane: str | None, labelled: set[str]) -> bool:
    """Whether a job's run of a step counts for this lane: a shared step's counts
    for every lane, a labelled step's only for the lane the job ran."""
    return step not in labelled or job.get("lane") in (None, lane)


def _config_lane(path: Path) -> str | None:
    """The lane a config names, if the config can still be read."""
    try:
        lane = tomllib.loads(path.read_text()).get("lane")
    except (OSError, ValueError):
        return None
    name = lane.get("name") if isinstance(lane, dict) else None
    return name if isinstance(name, str) else None


def _owner(artifact: dict[str, Any]) -> str | None:
    """The one series a QC record is about, or None for a cross-series one."""
    owner = (artifact.get("parameters") or {}).get("tilt_series_id")
    if isinstance(owner, str):
        return owner
    scope = artifact.get("tilt_series_ids") or []
    if isinstance(scope, list) and len(scope) == 1:
        return str(scope[0])
    return None


def qc_records(
    root: Path, project: Path, series: set[str], variant: str
) -> list[dict[str, Any]]:
    """The QC verdicts this lane sees for these series: the shared ones, and its
    own variant's. An experiment under another label is not the lane's."""
    payload = _load_json(project / "artifacts.json")
    if not isinstance(payload, dict):
        return []
    records: list[dict[str, Any]] = []
    for artifact in payload.get("artifacts") or []:
        parameters = artifact.get("parameters") or {}
        qc_type = parameters.get("qc_type")
        if artifact.get("kind") != "qc" or not isinstance(qc_type, str) or qc_type in _PICTURES:
            continue
        if artifact.get("variant") not in (None, variant):
            continue
        owner = _owner(artifact)
        if owner not in series:
            continue
        path = str(artifact.get("path") or "")
        document = _load_json(_resolve(root, path)) if path.endswith(".json") else None
        document = document if isinstance(document, dict) else {}
        # The document's own status first: the registry's copy is written once,
        # at registration, and is not always the verdict.
        status = str(document.get("status", parameters.get("status", "unknown"))).lower()
        records.append({
            "series": owner,
            "qc_type": qc_type,
            "step": QC_STEP.get(qc_type),
            "status": status if status in ("pass", "warning", "fail") else "unknown",
            "warnings": [str(warning) for warning in document.get("warnings") or []],
            "path": path,
        })
    return records


def step_state(run: dict[str, Any] | None, records: list[dict[str, Any]]) -> dict[str, Any]:
    """One step of one series as one state. A crash in its last run wins over any
    QC, which can only describe an earlier run."""
    worst = max(
        ("ok" if record["status"] == "pass" else record["status"] for record in records),
        key=_SEVERITY.__getitem__,
        default=None,
    )
    if run is not None and run.get("exit_code") != 0:
        state = "crashed"
    elif worst is not None:
        state = worst
    else:
        state = "ok" if run is not None else "none"
    entry: dict[str, Any] = {"state": state, "qc": records}
    if run is not None:
        entry.update(
            exit_code=run.get("exit_code"),
            finished_at=run.get("finished_at"),
            seconds=run.get("seconds"),
        )
    return entry


def slurm_logs(project: Path) -> list[Path]:
    """The logs of this user's jobs named for the project, wherever they were
    submitted from: a lane job writes its log in the folder it ran in, which sacct
    records as WorkDir. Nothing, where sacct cannot be run.

    sacct is asked from a week before the project's first recorded step, for the
    jobs that stopped before running one (a preflight, a kill); from 30 days ago
    for a project with no step on record."""
    started = [
        finished - timedelta(seconds=float(row.get("seconds") or 0))
        for row in _timings(project)
        if (finished := _time(row.get("finished_at"))) is not None
    ]
    since = min(started) - timedelta(days=7) if started else datetime.now(UTC) - timedelta(days=30)
    prefix = f"{project.name}-"
    return [
        Path(parts[2]) / f"slurm-{parts[1]}-{parts[0]}.log"
        for parts in _slurm(["sacct", "-n", "-P", "-X", "-u", os.environ.get("USER", ""),
                             "-S", since.astimezone().strftime("%Y-%m-%dT%H:%M:%S"),
                             "-o", "JobID,JobName,WorkDir"])
        if len(parts) == 3 and parts[1].startswith(prefix) and parts[2]
    ]


def read_job_log(path: Path, root: Path) -> dict[str, Any] | None:
    """One lane job as its log tells it, whatever project it ran; None for a log
    that cannot be read."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    shown = path.relative_to(root) if path.is_relative_to(root) else path
    job: dict[str, Any] = {"log": str(shown), "failed": []}
    for line in text.splitlines():
        if match := _JOB.match(line):
            job.update(name=match["name"], id=match["id"])
        elif match := _PROJECT.match(line):
            job["project"] = match["project"]
            if match["config"]:
                job["config"] = match["config"]
        elif match := _SERIES.match(line):
            job["series"] = match["series"].split()
        elif match := _STEPS.match(line):
            job["steps"] = match["steps"].split()
        elif match := _STARTED.match(line):
            job["started"] = match["when"]
        elif match := _FINISHED.match(line):
            job["finished"] = match["when"]
        elif match := _PREFLIGHT.match(line):
            job["preflight"] = match["problem"]
        elif match := _BLOCK.match(line):
            # One job runs one lane, named as it was when the job ran.
            job.setdefault("lane", match["lane"])
        elif match := _FAILED.match(line):
            job["failed"].append({
                "series": match["series"],
                "step": match["step"],
                "exit_code": int(match["code"]),
            })
    if "lane" not in job and "config" in job:
        # Stopped before its first step: the lane its config names now.
        lane = _config_lane(path.parent / job["config"])
        if lane is not None:
            job["lane"] = lane
    if job["failed"] or "preflight" in job:
        job["outcome"] = "failed"
    elif "finished" in job:
        job["outcome"] = "finished"
    else:
        job["outcome"] = "no end line: still running, or stopped by Slurm"
    return job


def job_logs(root: Path, project: str, elsewhere: Iterable[Path] = ()) -> list[dict[str, Any]]:
    """The lane jobs of this project, as their logs tell it: the logs in the
    repository root, and those found `elsewhere` (slurm_logs)."""
    here = root.glob(f"slurm-{glob.escape(Path(project).name)}-*.log")
    # A log in the root can be found both ways; the root's name for it is kept.
    paths = {path.resolve(): path for path in [*elsewhere, *here]}
    target = _resolve(root, project).resolve()
    jobs: list[dict[str, Any]] = []
    for path in sorted(paths.values(), key=lambda path: path.name):
        job = read_job_log(path, root)
        # Job names are the project's directory name, which another project can
        # share; the header says which project the job really ran, from the
        # folder it ran in, which is where its log is.
        if job is None or (path.parent / job.get("project", "")).resolve() != target:
            continue
        jobs.append(job)
    return jobs


def _slurm(command: list[str]) -> list[list[str]]:
    if shutil.which(command[0]) is None:
        return []
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return [line.split("|") for line in result.stdout.splitlines() if line]


def ask_slurm(jobs: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    """Add Slurm's state to each logged job, and return the lane's jobs still
    waiting, which have no log yet. Nothing, where Slurm cannot be asked."""
    ids = [job["id"] for job in jobs if "id" in job]
    if ids:
        states = {
            parts[0]: {"state": parts[1], "exit_code": parts[2], "elapsed": parts[3]}
            for parts in _slurm(["sacct", "-n", "-P", "-X", "-j", ",".join(ids),
                                 "-o", "JobID,State,ExitCode,Elapsed"])
            if len(parts) == 4
        }
        for job in jobs:
            if job.get("id") in states:
                job["slurm"] = states[job["id"]]
    logged = {job.get("id") for job in jobs}
    return [
        {"id": parts[0], "name": parts[1], "state": parts[2], "reason": parts[3]}
        for parts in _slurm(["squeue", "-h", "-u", os.environ.get("USER", ""),
                             "-o", "%i|%j|%T|%R"])
        if len(parts) == 4 and parts[1] in names and parts[0] not in logged
    ]


def summarize(
    cfg: dict[str, Any], root: Path, series: list[str] | None = None, slurm: bool = True
) -> dict[str, Any]:
    """The lane's state: per series and step, the last run and the QC verdict,
    with the jobs that ran them."""
    project_name = cfg["run"]["project"]
    project = _resolve(root, project_name)
    lane = cfg["lane"]
    chosen = series or lane_series(cfg, project)
    steps = run_series.build_steps(cfg, project, chosen[0] if chosen else "TS_X")
    chain = [name for name, _ in steps]
    labelled = {name for name, command in steps if "--variant" in command}
    runs = last_runs(project, set(chosen), labelled, lane["variant"])
    records = qc_records(root, project, set(chosen), lane["variant"])

    rows = []
    for one in chosen:
        mine = [record for record in records if record["series"] == one]
        rows.append({
            "id": one,
            "steps": {
                step: step_state(
                    runs.get((one, step)), [record for record in mine if record["step"] == step]
                )
                for step in chain
            },
            "other_qc": [record for record in mine if record["step"] not in chain],
        })

    jobs = job_logs(root, project_name, slurm_logs(project) if slurm else ())
    names = {f"{Path(project_name).name}-{stage.get('name')}"
             for stage in cfg["run"].get("stage") or []}
    waiting = ask_slurm(jobs, names) if slurm else []
    return {
        "project": project_name,
        "lane": lane.get("name"),
        "variant": lane["variant"],
        "steps": chain,
        "labelled": [step for step in chain if step in labelled],
        "series": rows,
        "jobs": jobs,
        "waiting": waiting,
    }


def _cell(entry: dict[str, Any]) -> str:
    if entry["state"] == "crashed":
        return f"X{entry.get('exit_code')}"
    return _CELL[entry["state"]]


def render(summary: dict[str, Any]) -> list[str]:
    """The summary for a terminal: the table, then what needs reading."""
    steps = summary["steps"]
    lines = [
        f"{summary['project']}: lane {summary['lane']} (variant {summary['variant']}), "
        f"{len(summary['series'])} series"
    ]
    table = [["series", *(SHORT.get(step, step) for step in steps)]]
    table += [[row["id"], *(_cell(row["steps"][step]) for step in steps)]
              for row in summary["series"]]
    widths = [max(len(text) for text in column) for column in zip(*table, strict=True)]
    for row in table:
        lines.append("  ".join(text.ljust(width) for text, width in zip(row, widths, strict=True))
                     .rstrip())
    lines.append("· no record  ok  W warning  F QC failure  ? QC status unknown  "
                 "X<n> exited with n")

    labelled = set(summary["labelled"])
    failures = {(failure["series"], failure["step"]): job["log"]
                for job in summary["jobs"] for failure in job["failed"]
                if lane_sees(job, failure["step"], summary["lane"], labelled)}
    crashed = [(row["id"], step, entry) for row in summary["series"]
               for step, entry in row["steps"].items() if entry["state"] == "crashed"]
    if crashed:
        lines += ["", "crashed"]
        for series, step, entry in crashed:
            log = failures.get((series, step), "")
            lines.append(f"  {series}  {step}  exit {entry.get('exit_code')}  "
                         f"{entry.get('finished_at') or ''}  {log}".rstrip())

    flagged = [(row["id"], record) for row in summary["series"]
               for entry in row["steps"].values() for record in entry["qc"]]
    flagged += [(row["id"], record) for row in summary["series"] for record in row["other_qc"]]
    flagged = [(series, record) for series, record in flagged if record["status"] != "pass"]
    if flagged:
        lines += ["", "QC not passed"]
        for series, record in flagged:
            lines.append(f"  {series}  {record['step'] or record['qc_type']}  "
                         f"{record['status']}  {record['path']}")
            lines += [f"      {warning}" for warning in record["warnings"]]

    if summary["jobs"] or summary["waiting"]:
        lines += ["", "jobs"]
        for job in summary["jobs"]:
            # Lanes of one project share job names: the lane says whose job it is.
            lane = f"  lane {job['lane']}" if "lane" in job else ""
            text = f"  {job.get('name', '?')}  {job.get('id', '?')}{lane}  {job['outcome']}"
            if "slurm" in job:
                slurm = job["slurm"]
                text += f" (Slurm: {slurm['state']}, exit {slurm['exit_code']}, {slurm['elapsed']})"
            lines.append(f"{text}  {job['log']}")
            if "preflight" in job:
                lines.append(f"      preflight: {job['preflight']}")
            lines += [f"      FAILED at {failure['step']} for {failure['series']} "
                      f"(exit {failure['exit_code']})" for failure in job["failed"]]
        lines += [f"  {job['name']}  {job['id']}  {job['state']}  {job['reason']}"
                  for job in summary["waiting"]]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="the lane config the project was run with")
    parser.add_argument("--series", action="append", default=None,
                        help="only this series; repeat for several")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--no-slurm", action="store_true",
                        help="do not ask sacct and squeue; read the logs in the repository "
                             "root only")
    args = parser.parse_args()

    _, cfg = run_series.read_config(args.config)
    run_series.check_keys(cfg, args.config)
    if not isinstance(cfg.get("run"), dict) or not cfg["run"].get("project"):
        raise SystemExit(f"{args.config}: no [run] project to summarize")
    summary = summarize(cfg, ROOT, series=args.series, slurm=not args.no_slurm)
    if args.json:
        print(json.dumps({"config": str(args.config), **summary}, indent=2))
    else:
        print("\n".join(render(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
