#!/usr/bin/env python3
"""The evidence of the eval cases (docs/agent/eval_cases.md) on this cluster:
what is still there, and a copy of it as a case folder the runner reads.

    python3 scripts/collect_case.py --inventory --cluster mdc
    python3 scripts/collect_case.py --inventory E01 E05
    python3 scripts/collect_case.py --inventory --cluster munich --json
    python3 scripts/collect_case.py --collect E02 E05 E08
    python3 scripts/collect_case.py --restore docs/agent/cases/E04     (what setup.sh runs)

docs/agent/cases/cases.toml says, for each case, its lane, the days it
happened, and its steps, jobs and files.

--inventory is read-only: it copies and writes nothing. For each case it prints:

* its jobs: those the table names, wherever their logs are, with what only Slurm
  knows of them (state, node, the GPU it gave); the project's jobs that started
  in the case's days; and the project's jobs whose log matches the case's
  pattern, at any date, which finds the jobs the table does not name yet;
* each run of the case's steps in its days, with the logs its programs wrote
  while it ran and the runs of the same step since;
* the case's QC records and extra files, and the project's registry, step
  timings, journal and manifests: each as of the case, written later, or
  missing.

A later run of a step writes its files again, and the registry keeps only the
last record: a file written after the case's last day is a later run's, and the
case's own is gone. Slurm logs are named by their job, so they stay. A case
without days is the project as it is now.

--collect writes each case as a folder in the runner's shape
(docs/agent/cases/README.md), under ~/cases unless --out says where:

* tree/: the project as it stood when the case ended (the end of its last day,
  or the time `until` gives), at its paths from the repository root. The lane
  file is its last commit before then. The registry, step timings and journal
  lose what was written after; QC records, program logs, extra files and job
  configs written after are left out, as are binary files and files over 5 MB.
  The jobs the table names and those that started in the case's days give their
  Slurm logs, and the configs their headers name. A stack lane's stack and mdoc
  paths point at empty stand-ins under data/, as in the constructed cases;
* case.toml: the request and the grading lines from eval_cases.md, the table's
  playbook entries, a note on what is missing, and stand-ins for the variables
  the lane's commands name;
* manifest.json: every file's time on the cluster and where it came from, what
  was left out and why, and what only Slurm knows of the named jobs;
* setup.sh, which runs --restore in the runner's sandbox: git keeps no file
  times, and show_failure finds a step's logs by when they were written.

The folders the jobs ran in become {{ROOT}}, the home folder {{HOME}}, the
user's login and the forms of their name {{USER}} (the passwd full name gives
first.last, f.last, first_last and the last name; --name adds others), and
--other names {{OTHER1}} and on. Nothing is written while a known name or a
home folder is left in a file; --keep lets a path through that only looks like
one. The paths kept are listed at the end, to be looked over before the folder
leaves the cluster.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import json
import os
import pwd
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_failure  # noqa: E402
import show_project  # noqa: E402
import submit_lanes  # noqa: E402

TABLE = Path("docs/agent/cases/cases.toml")
EVAL_CASES = Path("docs/agent/eval_cases.md")
CLUSTERS = ("mdc", "munich", "none")
# Every key a case may set. A misspelt one would be skipped without a word, and
# the inventory would look in the wrong place.
CASE_KEYS = frozenset({
    "title", "cluster", "lane", "series", "steps", "since", "until", "jobs",
    "log_pattern", "extras", "elsewhere", "note", "playbook",
})
_LISTS = ("series", "steps", "jobs", "extras", "elsewhere", "playbook")
# What only Slurm knows of a job the table names: where it ran and what it got.
_DETAILS = ("JobID", "JobName", "State", "ExitCode", "Start", "Elapsed", "NodeList",
            "AllocTRES")
# Files named one by one per list in the text; --json has them all.
SHOWN = 5


def read_cases(path: Path) -> dict[str, dict[str, Any]]:
    """The case table, refused whole if any case in it is wrong."""
    try:
        table = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SystemExit(f"{path}: {error}") from None
    problems = []
    for name, case in table.items():
        if not isinstance(case, dict):
            problems.append(f"{name}: not a table")
            continue
        problems += [f"{name}: {problem}" for problem in _case_problems(case)]
    if problems:
        raise SystemExit("\n".join([f"{path}:", *(f"  {problem}" for problem in problems)]))
    return table


def _case_problems(case: dict[str, Any]) -> list[str]:
    found = [f"unknown key {key!r}" for key in sorted(set(case) - CASE_KEYS)]
    if not isinstance(case.get("title"), str):
        found.append("no title")
    if case.get("cluster") not in CLUSTERS:
        found.append(f"cluster must be one of {', '.join(CLUSTERS)}")
    elif case["cluster"] != "none":
        if not isinstance(case.get("lane"), str):
            found.append("no lane")
        if "steps" not in case:
            found.append("no steps; [] for none")
    for key in _LISTS:
        value = case.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            found.append(f"{key} must be a list of strings")
    if "since" in case or "until" in case:
        since, until = _bound(case.get("since"), False), _bound(case.get("until"), True)
        if since is None or until is None:
            found.append("since and until must both be days or times, such as 2026-10-14 "
                         "or 2026-10-14T19:15:00")
        elif since >= until:
            found.append("until is before since")
    if "log_pattern" in case:
        try:
            re.compile(case["log_pattern"])
        except (re.error, TypeError) as error:
            found.append(f"log_pattern: {error}")
    return found


def _bound(value: Any, until: bool) -> datetime | None:
    """A case's since or until as an instant. A day starts at its midnight and, as
    an until, ends at the next one; a time is taken as it is, in this machine's
    time zone, the cluster's, unless it names its own."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.astimezone()
    if isinstance(value, date):
        return datetime.combine(value + timedelta(days=1 if until else 0), time()).astimezone()
    return None


def _time(text: Any) -> datetime | None:
    try:
        value = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _when(moment: datetime) -> str:
    return moment.astimezone().isoformat(timespec="seconds")


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _shown(root: Path, path: Path) -> str:
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def _window(case: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    """When the case begins and ends; None for a case without days."""
    if "since" not in case:
        return None, None
    return _bound(case["since"], False), _bound(case["until"], True)


def _file(root: Path, path: Path, end: datetime | None) -> dict[str, Any]:
    """A file as of the case, written later, or missing."""
    try:
        stat = path.stat()
    except OSError:
        return {"path": _shown(root, path), "state": "missing"}
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    later = end is not None and modified >= end
    return {"path": _shown(root, path), "state": "written later" if later else "as of the case",
            "bytes": stat.st_size, "modified": _when(modified)}


def _rows(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSON-lines file; a line cut short by a killed job is skipped."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _first_match(path: Path, pattern: re.Pattern[str]) -> dict[str, Any] | None:
    try:
        with path.open(errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if pattern.search(line):
                    return {"line": number, "text": line.rstrip()[:show_failure.MAX_WIDTH]}
    except OSError:
        pass
    return None


def _job(root: Path, job: dict[str, Any], pattern: re.Pattern[str] | None) -> dict[str, Any]:
    keys = ("id", "name", "started", "lane", "config", "series", "outcome", "failed",
            "preflight", "log")
    entry = {key: job[key] for key in keys if key in job}
    if pattern is not None:
        entry["match"] = _first_match(_resolve(root, job["log"]), pattern)
    return entry


def _named_logs(root: Path, ids: list[str], slurm: bool) -> dict[str, list[Path]]:
    """Where the logs of the jobs the table names are, whatever project they ran:
    in the repository root, and in the folder sacct says each ran in."""
    found = {job: sorted(root.glob(f"slurm-*-{glob.escape(job)}.log")) for job in ids}
    if slurm and ids:
        for parts in show_project._slurm(["sacct", "-n", "-P", "-X", "-j", ",".join(ids),
                                          "-o", "JobID,JobName,WorkDir"]):
            if len(parts) != 3 or parts[0] not in found or not parts[2]:
                continue
            path = Path(parts[2]) / f"slurm-{parts[1]}-{parts[0]}.log"
            if path.is_file() and path.resolve() not in {log.resolve() for log in found[parts[0]]}:
                found[parts[0]].append(path)
    return found


def _slurm_details(ids: list[str]) -> dict[str, dict[str, str]]:
    rows = show_project._slurm(["sacct", "-n", "-P", "-X", "-j", ",".join(ids),
                                "-o", ",".join(_DETAILS)])
    return {parts[0]: dict(zip(_DETAILS[1:], parts[1:], strict=True))
            for parts in rows if len(parts) == len(_DETAILS)}


def _jobs(
    root: Path, project_name: str, case: dict[str, Any],
    start: datetime | None, end: datetime | None, slurm: bool,
) -> dict[str, Any]:
    """The case's jobs: named, started in its days, and matching its pattern."""
    pattern = re.compile(case["log_pattern"]) if "log_pattern" in case else None
    project = _resolve(root, project_name)
    logs = show_project.job_logs(root, project_name,
                                 show_project.slurm_logs(project) if slurm else ())
    named: dict[str, list[dict[str, Any]]] = {}
    for job_id, paths in _named_logs(root, case.get("jobs", []), slurm).items():
        read = [show_project.read_job_log(path, root) for path in paths]
        named[job_id] = [_job(root, job, pattern) for job in read if job is not None]
    in_days = []
    if start is not None and end is not None:
        in_days = [_job(root, job, None) for job in logs
                   if (when := _time(job.get("started"))) is not None and start <= when < end]
    matching = []
    if pattern is not None:
        matching = [entry for entry in (_job(root, job, pattern) for job in logs)
                    if entry["match"] is not None]
    return {"named": named, "in_days": in_days, "matching": matching}


def _runs(
    root: Path, project: Path, series: list[str], steps: list[str], labelled: set[str],
    variant: str, start: datetime, end: datetime,
) -> list[dict[str, Any]]:
    """Each run of the case's steps in its days, with the logs its programs wrote
    while it ran, and the runs of the same step since, which wrote its files
    again. Another lane's run of a step each lane labels for itself writes
    elsewhere, so it is not one of them."""
    rows = []
    for row in _rows(project / "step_timings.jsonl"):
        finished = _time(row.get("finished_at"))
        step = row.get("step")
        if finished is None or row.get("series") not in series or step not in steps:
            continue
        if step in labelled and row.get("variant", variant) != variant:
            continue
        rows.append((finished, row))
    rows.sort(key=lambda item: item[0])
    found = []
    for finished, row in rows:
        if not start <= finished < end:
            continue
        began = finished - timedelta(seconds=float(row.get("seconds") or 0))
        logs = show_failure.step_logs(project, row["series"], row["step"], began, finished)
        since = [(when, other) for when, other in rows if when > finished
                 and (other["series"], other["step"]) == (row["series"], row["step"])]
        found.append({
            "series": row["series"],
            "step": row["step"],
            "exit_code": row.get("exit_code"),
            "finished_at": _when(finished),
            "lane": row.get("lane"),
            "logs_then": [{"path": _shown(root, path), "modified": _when(modified)}
                          for path, modified in logs],
            "runs_since": [{"finished_at": _when(when), "exit_code": other.get("exit_code")}
                           for when, other in since],
        })
    return found


def _state(root: Path, project: Path, end: datetime | None) -> dict[str, Any]:
    """The project's own files: the registry, which keeps the last record of
    each artifact, and the step timings and journal, which only grow."""
    def grown(name: str, key: str) -> dict[str, Any]:
        entry = _file(root, project / name, end)
        if entry["state"] != "missing":
            times = [when for row in _rows(project / name) if (when := _time(row.get(key)))]
            entry["rows"] = sum(1 for when in times if end is None or when < end)
            entry["rows_later"] = len(times) - entry["rows"]
        return entry

    return {
        "registry": _file(root, project / "artifacts.json", end),
        "step_timings": grown("step_timings.jsonl", "finished_at"),
        "journal": grown("journal.jsonl", "at"),
        "manifests": [_file(root, path, end)
                      for path in sorted((project / "manifests").glob("*.json"))],
    }


def inventory(
    name: str, case: dict[str, Any], root: Path = ROOT, slurm: bool = True
) -> dict[str, Any]:
    """What of one case's evidence is here."""
    entry: dict[str, Any] = {"case": name, "title": case["title"], "cluster": case["cluster"]}
    if case["cluster"] == "none":
        return {**entry, **{key: case[key] for key in ("note", "elsewhere") if key in case}}
    config = root / case["lane"]
    _, cfg = run_series.read_config(config)
    run_series.check_keys(cfg, config)
    project_name = cfg["run"]["project"]
    project = _resolve(root, project_name)
    series = case.get("series") or show_project.lane_series(cfg, project)
    steps = run_series.build_steps(cfg, project, series[0] if series else "TS_X")
    chain = [step for step, _ in steps]
    unknown = [step for step in case["steps"] if step not in chain]
    if unknown:
        raise SystemExit(f"{name}: no step {', '.join(unknown)} in {case['lane']}")
    entry.update(lane=case["lane"], project=project_name, series=series)
    start, end = _window(case)
    if start is not None:
        entry["days"] = [case["since"].isoformat(), case["until"].isoformat()]
    if "log_pattern" in case:
        entry["log_pattern"] = case["log_pattern"]
    entry["here"] = project.is_dir()
    if entry["here"]:
        labelled = {step for step, command in steps if "--variant" in command}
        variant = cfg["lane"]["variant"]
        entry["jobs"] = _jobs(root, project_name, case, start, end, slurm)
        if slurm and case.get("jobs"):
            entry["slurm"] = _slurm_details(case["jobs"])
        if start is not None and end is not None:
            entry["runs"] = _runs(root, project, series, case["steps"], labelled, variant,
                                  start, end)
        entry["qc"] = [
            {"series": record["series"], "qc_type": record["qc_type"],
             **_file(root, _resolve(root, record["path"]), end)}
            for record in show_project.qc_records(root, project, set(series), variant)
            if record["step"] in case["steps"]
        ]
        entry["extras"] = [
            {"pattern": pattern,
             "files": [_file(root, path, end) for path in sorted(project.glob(pattern))
                       if path.is_file()]}
            for pattern in case.get("extras", [])
        ]
        entry["state"] = _state(root, project, end)
    return {**entry, **{key: case[key] for key in ("note", "elsewhere") if key in case}}


def _counts(files: list[dict[str, Any]]) -> str:
    states = ("as of the case", "written later", "missing")
    counted = [(state, sum(1 for one in files if one["state"] == state)) for state in states]
    return ", ".join(f"{count} {state}" for state, count in counted if count) or "none"


def _unlike_the_case(files: list[dict[str, Any]], indent: str) -> list[str]:
    """The files that are not as of the case, a few by name."""
    odd = [one for one in files if one["state"] != "as of the case"]
    lines = [f"{indent}{one['state']}  {one['path']}  {one.get('modified', '')}".rstrip()
             for one in odd[:SHOWN]]
    if len(odd) > SHOWN:
        lines.append(f"{indent}and {len(odd) - SHOWN} more")
    return lines


def _job_line(job: dict[str, Any]) -> str:
    lane = f"  lane {job['lane']}" if "lane" in job else ""
    started = f"  started {job['started']}" if "started" in job else ""
    return (f"{job.get('name', '?')}  {job.get('id', '?')}{lane}{started}  "
            f"{job.get('outcome', '?')}  {job['log']}")


def _job_details(job: dict[str, Any], indent: str) -> list[str]:
    lines = []
    if "config" in job:
        lines.append(f"{indent}config {job['config']}")
    if "preflight" in job:
        lines.append(f"{indent}preflight: {job['preflight']}")
    lines += [f"{indent}FAILED at {failure['step']} for {failure['series']} "
              f"(exit {failure['exit_code']})" for failure in job.get("failed", [])]
    if job.get("match"):
        lines.append(f"{indent}line {job['match']['line']}: {job['match']['text']}")
    return lines


def render(entry: dict[str, Any]) -> list[str]:
    """One case's inventory for a terminal."""
    lines = [f"{entry['case']}  {entry['title']}"]
    if entry["cluster"] == "none":
        return lines + [f"  built by hand: {entry.get('note', '')}".rstrip()]
    days = "..".join(entry["days"]) if "days" in entry else "none, the project as it is now"
    lines.append(f"  {entry['cluster']}  {entry['lane']}  {entry['project']}  days {days}")
    if entry["here"]:
        lines += _found(entry)
    else:
        lines.append(f"  not here: there is no {entry['project']} on this machine")
    lines += [f"  elsewhere: {item}" for item in entry.get("elsewhere", [])]
    if "note" in entry:
        lines.append(f"  note: {entry['note']}")
    return lines


def _found(entry: dict[str, Any]) -> list[str]:
    lines = []
    jobs = entry["jobs"]
    if jobs["named"]:
        lines.append("  jobs the table names")
        for job_id, logs in jobs["named"].items():
            if not logs:
                lines.append(f"    {job_id}  no log found")
            for job in logs:
                lines.append(f"    {_job_line(job)}")
                lines += _job_details(job, "      ")
            slurm = entry.get("slurm", {}).get(job_id)
            if slurm:
                lines.append("      Slurm: " + "  ".join(f"{key} {value}"
                                                     for key, value in slurm.items()))
    if "days" in entry:
        lines.append(f"  jobs that started in its days: {len(jobs['in_days'])}")
        for job in jobs["in_days"]:
            lines.append(f"    {_job_line(job)}")
            lines += _job_details(job, "      ")
    if "log_pattern" in entry:
        lines.append(f"  jobs whose log matches {entry['log_pattern']!r}, at any date: "
                     f"{len(jobs['matching'])}")
        for job in jobs["matching"]:
            lines.append(f"    {_job_line(job)}")
            lines.append(f"      line {job['match']['line']}: {job['match']['text']}")

    if "runs" in entry:
        lines.append(f"  runs of its steps in its days: {len(entry['runs'])}")
        for run in entry["runs"]:
            since = run["runs_since"]
            again = (f"run since {len(since)}, last {since[-1]['finished_at']} "
                     f"exit {since[-1]['exit_code']}" if since else "not run since")
            lines.append(f"    {run['series']}  {run['step']}  exit {run['exit_code']}  "
                         f"{run['finished_at']}  logs written then {len(run['logs_then'])}  "
                         f"{again}")
    lines.append(f"  QC records: {_counts(entry['qc'])}")
    lines += _unlike_the_case(entry["qc"], "    ")
    for extra in entry["extras"]:
        found = _counts(extra["files"]) if extra["files"] else "missing"
        lines.append(f"  {extra['pattern']}: {found}")
        lines += _unlike_the_case(extra["files"], "    ")

    state = entry["state"]
    registry = state["registry"]
    lines.append(f"  registry      {registry['state']}  {registry.get('modified', '')}".rstrip())
    for key, label in (("step_timings", "step timings"), ("journal", "journal")):
        grown = state[key]
        if grown["state"] == "missing":
            text = "missing"
        elif "days" in entry:
            text = f"rows: {grown['rows']} up to its last day, {grown['rows_later']} after"
        else:
            text = f"rows: {grown['rows']}"
        lines.append(f"  {label:<13} {text}")
    lines.append(f"  manifests     {_counts(state['manifests'])}")
    lines += _unlike_the_case(state["manifests"], "    ")
    return lines


# Collecting

OUT = Path.home() / "cases"
# A bigger file is not evidence a case folder can carry. The registry, the step
# timings and the journal are taken whatever their size.
MAX_BYTES = 5 * 1024 * 1024
PLACEHOLDER = re.compile(r"\{\{(ROOT|HOME|USER|OTHER(\d+))\}\}")
# A home folder, the user's or anyone's: none may be left in a case but as a
# placeholder.
_HOME = re.compile(r"(?:/fast/home|/ictstr01/home|/lustre/home|/home|/Users)(?:/[^/\s'\"`]+){1,2}")
# A path's first three folders, for the person to look over.
_PREFIX = re.compile(r"(?<![\w.~{}:/-])/[\w.-]+(?:/[\w.-]+){0,2}")
# What [env] gives a case, as the constructed cases set them: the first three
# always, the others when the lane's commands name them, and /opt/<name> for any
# variable not listed. {bin} is the runner's folder of stand-ins, which holds a
# CTFFIND 5 that prints its banner.
STAND_INS = {
    "CTFFIND5": "{bin}/ctffind5",
    "IMOD_DIR": "/opt/imod_5.1.12",
    "PYTOMO_MP_DIR": "/opt/pytom_tm/bin",
    "MEMBRAIN_CKPT": "/opt/membrain/v10.ckpt",
    "MEMBRAIN_EXE": "/opt/membrain/membrain",
}
_ALWAYS = ("CTFFIND5", "IMOD_DIR", "PYTOMO_MP_DIR")
SETUP = """\
# Puts this sandbox's own folder, home and user where the collector left
# placeholders for the cluster's, and gives every file of the case the time it was
# last written there: git keeps no file times, and show_failure finds a step's
# logs by when they were written.
set -euo pipefail
python3 scripts/collect_case.py --restore "$(dirname "$0")"
"""
_MOVED = (
    "# Evaluation case: this lane's [data] stack and mdoc paths point into the\n"
    "# sandbox, where empty stand-ins replace the cluster's stacks and mdocs.\n"
)
_BULLET = re.compile(r"^- \*\*(?P<label>[^*]+):\*\*\s*(?P<text>.*)$")


@dataclass
class Person:
    """Who collects: their home folders and the forms of their name, other people's
    names, and path prefixes that only look like a home folder."""

    homes: list[str]
    names: list[str]
    others: list[str] = field(default_factory=list)
    keep: list[str] = field(default_factory=list)


def name_forms(full: str) -> list[str]:
    """How a full name shows in folder names: first.last, f.last, first_last,
    "first last" and the last name alone. Nothing from a single word."""
    words = [word.lower() for word in re.split(r"[\s,]+", full) if word]
    if len(words) < 2:
        return []
    first, last = words[0], words[-1]
    return [f"{first}.{last}", f"{first[0]}.{last}", f"{first}_{last}", f"{first} {last}", last]


def this_person(names: list[str], others: list[str], keep: list[str]) -> Person:
    """The user who runs this, as the machine knows them, and the forms given."""
    login = getpass.getuser()
    try:
        full = pwd.getpwnam(login).pw_gecos
    except KeyError:
        full = ""
    home = str(Path.home())
    return Person(homes=sorted({home, os.path.realpath(home)}),
                  names=[login, *name_forms(full), *names], others=others, keep=keep)


class Scrub:
    """Replaces in a text the folders the case's files were written in and the
    person's homes, the longest first, then names in any case. A folder named
    after the user so becomes {{ROOT}}, not a path holding {{USER}}."""

    def __init__(self, roots: list[str], person: Person) -> None:
        short = [name for name in [*person.names, *person.others] if len(name) < 4]
        if short:
            raise SystemExit(f"too short to replace safely: {', '.join(short)}; "
                             "give the longer forms of the name")
        paths = {path.rstrip("/"): "{{HOME}}" for path in person.homes}
        paths.update({path.rstrip("/"): "{{ROOT}}" for path in roots})
        self.paths = sorted(((path, token) for path, token in paths.items() if path),
                            key=lambda item: -len(item[0]))
        self.words = {name.lower(): "{{USER}}" for name in person.names}
        self.words.update({name.lower(): f"{{{{OTHER{number}}}}}"
                           for number, name in enumerate(person.others, 1)})
        # Placeholders first, so that a name is never looked for inside one.
        alternatives = [PLACEHOLDER.pattern] + [
            re.escape(word) for word in sorted(self.words, key=len, reverse=True)
        ]
        self.pattern = re.compile("|".join(alternatives), re.IGNORECASE)
        self.keep = tuple(person.keep)

    def __call__(self, text: str) -> str:
        for path, token in self.paths:
            text = text.replace(path, token)
        return self.pattern.sub(
            lambda match: self.words.get(match.group(0).lower(), match.group(0)), text
        )

    def leftovers(self, text: str) -> list[str]:
        """What of a person a scrubbed text still holds."""
        found = [match.group(0) for match in self.pattern.finditer(text)
                 if not PLACEHOLDER.fullmatch(match.group(0))]
        found += [match.group(0) for match in _HOME.finditer(text)
                  if "{{" not in match.group(0) and not match.group(0).startswith(self.keep)]
        return found


@dataclass
class Evidence:
    """One file of a case folder: its path in tree/, its text, when it was last
    written on the cluster, and what it is."""

    path: str
    text: str
    modified: float
    source: str


def _text(path: Path, limit: int | None = MAX_BYTES) -> tuple[str | None, str]:
    """A file's text, or None and why it cannot be evidence."""
    try:
        size = path.stat().st_size
    except OSError:
        return None, "missing"
    if limit is not None and size > limit:
        return None, f"{size / 2**20:.1f} MB, over the {limit // 2**20} MB a case holds"
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None, "binary"
    try:
        return data.decode(), ""
    except UnicodeDecodeError:
        return None, "binary"


def lane_at(root: Path, lane: str, end: datetime | None) -> tuple[str, float, str]:
    """The lane file as it stood when the case ended: its last commit before then.
    The file as it is now for a case without days, or where git cannot say."""
    if end is not None:
        try:
            found = subprocess.run(
                ["git", "log", "-1", "--format=%H %ct", f"--before={end.isoformat()}", "--", lane],
                cwd=root, check=True, capture_output=True, text=True,
            ).stdout.split()
            if found:
                text = subprocess.run(["git", "show", f"{found[0]}:{lane}"], cwd=root,
                                      check=True, capture_output=True, text=True).stdout
                return (text, float(found[1]),
                        f"commit {found[0][:9]}, the lane file's last before the case ended")
        except (OSError, subprocess.CalledProcessError):
            pass
    path = root / lane
    why = "the lane file as it is now" + (
        ": git has no commit of it from before the case ended" if end is not None else "")
    return path.read_text(), path.stat().st_mtime, why


def moved_data(cfg: dict[str, Any]) -> dict[str, str]:
    """A stack lane's stack and mdoc paths, and where they go in the sandbox: under
    data/, from the folder that holds them on, as the constructed cases have them.
    The dry run looks for those files; nothing looks for a frames lane's."""
    try:
        if run_series.lane_input(cfg) != "stack":
            return {}
    except SystemExit:
        return {}
    data = cfg.get("data") or {}
    paths = [data[key] for key in ("stack", "mdoc")
             if isinstance(data.get(key), str) and data[key].startswith("/")]
    if not paths:
        return {}
    base = os.path.dirname(os.path.commonpath([os.path.dirname(path) for path in paths]))
    return {path: "data/" + os.path.relpath(path, base) for path in paths}


def _trimmed_rows(path: Path, key: str, end: datetime | None) -> tuple[str, int, int] | None:
    """A JSON-lines file without the rows written after the case: they only grow."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    kept = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue  # cut short by a killed job
        when = _time(row.get(key)) if isinstance(row, dict) else None
        if end is None or when is None or when < end:
            kept.append(line + "\n")
    return "".join(kept), len(kept), len(lines)


def _trimmed_registry(
    root: Path, project: Path, end: datetime | None
) -> tuple[str, int, int] | None:
    """The registry without the records of files written after the case: it keeps
    one record per artifact, the last run's."""
    try:
        payload = json.loads((project / "artifacts.json").read_text())
    except (OSError, ValueError):
        return None
    records = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return None

    def later(record: Any) -> bool:
        path = record.get("path") if isinstance(record, dict) else None
        if end is None or not isinstance(path, str) or not path:
            return False
        try:
            return _resolve(root, path).stat().st_mtime >= end.timestamp()
        except OSError:
            return False

    kept = [record for record in records if not later(record)]
    return json.dumps({**payload, "artifacts": kept}, indent=2) + "\n", len(kept), len(records)


def case_text(markdown: str, name: str) -> dict[str, Any]:
    """A case's request and grading lines, as its section of eval_cases.md has
    them: the request without its quotes, and a list item, or else each clause of
    the line, per Pass-if and Must-not line."""
    section = re.search(rf"^### {re.escape(name)}\. .*?$(?P<body>.*?)(?=^##|\Z)", markdown,
                        re.MULTILINE | re.DOTALL)
    if section is None:
        raise SystemExit(f"eval_cases.md has no case {name}")
    bullets: dict[str, list[str]] = {}
    label = None
    for line in section["body"].splitlines():
        if bullet := _BULLET.match(line):
            label = bullet["label"]
            bullets[label] = [bullet["text"]]
        elif label is not None and line.startswith("  - "):
            bullets[label].append(line[4:])
        elif label is not None and line.startswith("  ") and line.strip():
            bullets[label][-1] += " " + line.strip()
        else:
            label = None

    def lines(label: str) -> list[str]:
        inline, *listed = bullets.get(label, [""])
        parts = listed or re.split(r";\s+", inline)
        cleaned = (part.strip().rstrip(";.").replace("`", "").strip() for part in parts)
        return [part for part in cleaned if part]

    request = bullets.get("Request", [""])[0].strip().strip('"')
    return {"request": request, "pass_if": lines("Pass if"), "must_not": lines("Must not")}


def case_toml(name: str, title: str, text: dict[str, Any], note: str, playbook: list[str],
              env: dict[str, str]) -> str:
    """case.toml as scripts/run_cases.py reads it."""
    def value(item: Any) -> str:
        if isinstance(item, list):
            return "[" + ", ".join(value(one) for one in item) + "]"
        return json.dumps(item, ensure_ascii=False)  # a TOML basic string too

    lines = [f"id = {value(name)}", f"title = {value(title)}",
             f"request = {value(text['request'])}", f"note = {value(note)}"]
    if playbook:
        lines.append(f"playbook = {value(playbook)}")
    lines += [f"pass_if = {value(text['pass_if'])}", f"must_not = {value(text['must_not'])}",
              "", "[env]"]
    lines += [f"{key} = {value(env[key])}" for key in sorted(env)]
    return "\n".join(lines) + "\n"


def _inside(path: Path, roots: list[str]) -> str | None:
    """A path from the folder the job ran in, or None outside every such folder."""
    for candidate in (str(path), os.path.realpath(path)):
        for root in sorted(roots, key=len, reverse=True):
            if candidate.startswith(root.rstrip("/") + "/"):
                return candidate[len(root.rstrip("/")) + 1:]
    return None


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(
    name: str, case: dict[str, Any], person: Person, root: Path = ROOT, out: Path = OUT,
    slurm: bool = True, overwrite: bool = False,
) -> tuple[Path, list[str]]:
    """Write one case as a folder the runner reads, and say what went in. Nothing
    is written while a name or a home folder is left in it."""
    if case["cluster"] == "none":
        raise SystemExit(f"{name} is built by hand: {case.get('note', '')}")
    folder = out / name
    if folder.exists() and not overwrite:
        raise SystemExit(f"{folder} exists: remove it, or give --overwrite")
    _, cfg = run_series.read_config(root / case["lane"])
    project = _resolve(root, cfg["run"]["project"])
    if not project.is_dir():
        raise SystemExit(f"{name}: there is no {cfg['run']['project']} on this machine")
    if not project.is_relative_to(root):
        raise SystemExit(f"{name}: the project {project} is not in {root}")
    project_name = _shown(root, project)
    start, end = _window(case)
    series = case.get("series") or show_project.lane_series(cfg, project)
    files: dict[str, Evidence] = {}
    left: list[dict[str, str]] = []

    def take(path: Path, target: str, source: str, *, dated: bool = True,
             limit: int | None = MAX_BYTES) -> None:
        """One file into the case, if it is as of the case and text. A job's log is
        the job's whenever it was last written, and says so."""
        if target in files or any(one["path"] == _shown(root, path) for one in left):
            return  # several jobs can name one config
        try:
            modified = path.stat().st_mtime
        except OSError:
            left.append({"path": _shown(root, path), "why": "missing"})
            return
        if end is not None and modified >= end.timestamp():
            if dated:
                left.append({"path": _shown(root, path), "why": "written later, "
                             + _when(datetime.fromtimestamp(modified, tz=UTC))})
                return
            source += ", written after the case ended"
        text, why = _text(path, limit)
        if text is None:
            left.append({"path": _shown(root, path), "why": why})
            return
        files[target] = Evidence(target, text, modified, source)

    # The jobs the table names, wherever their logs are, and the project's that
    # started in the case's days; the folders they ran in become {{ROOT}}.
    logs: dict[Path, tuple[Path, str]] = {}
    for job_id, paths in _named_logs(root, case.get("jobs", []), slurm).items():
        if not paths:
            left.append({"path": f"the Slurm log of job {job_id}",
                         "why": "not found where the job ran"})
        for path in paths:
            logs.setdefault(path.resolve(),
                            (path, f"Slurm log of job {job_id}, named in the table"))
    if start is not None and end is not None:
        found = show_project.job_logs(root, cfg["run"]["project"],
                                      show_project.slurm_logs(project) if slurm else ())
        for job in found:
            when = _time(job.get("started"))
            if when is not None and start <= when < end:
                path = _resolve(root, job["log"])
                logs.setdefault(path.resolve(), (
                    path, f"Slurm log of job {job.get('id', '?')}, started in the case's days"))
    roots = {str(root), os.path.realpath(root)}
    real = Path(os.path.realpath(project)).parts
    own = Path(project_name).parts
    if real[-len(own):] == own:
        roots.add(str(Path(*real[: -len(own)])))  # the folder a linked project lives in
    named_configs = []
    for path, source in logs.values():
        roots.update({str(path.parent), os.path.realpath(path.parent)})
        take(path, path.name, source, dated=False)
        header = show_project.read_job_log(path, root)
        if header is not None and "config" in header:
            named_configs.append(path.parent / header["config"])

    lane_text, lane_modified, lane_source = lane_at(root, case["lane"], end)
    files[case["lane"]] = Evidence(case["lane"], lane_text, lane_modified, lane_source)
    for path in named_configs:
        target = _inside(path, sorted(roots))
        if target is None:
            left.append({"path": str(path), "why": "outside the folders the jobs ran in"})
        else:
            # One changed after the case may hold its fix: E05's copy was given its
            # CTFFIND between the failed job and the next.
            take(path, target, "a config a job's log names")

    # The project as it stood: what only grows loses its later rows, the registry
    # the records of later files, and each is dated no later than the case.
    def until_case(modified: float) -> float:
        return modified if end is None else min(modified, end.timestamp() - 1)

    for file_name, key in (("step_timings.jsonl", "finished_at"), ("journal.jsonl", "at")):
        rows = _trimmed_rows(project / file_name, key, end)
        if rows is not None:
            files[f"{project_name}/{file_name}"] = Evidence(
                f"{project_name}/{file_name}", rows[0],
                until_case((project / file_name).stat().st_mtime),
                f"{rows[1]} of its {rows[2]} rows")
    registry = _trimmed_registry(root, project, end)
    if registry is not None:
        files[f"{project_name}/artifacts.json"] = Evidence(
            f"{project_name}/artifacts.json", registry[0],
            until_case((project / "artifacts.json").stat().st_mtime),
            f"the registry, {registry[1]} of its {registry[2]} records")
    for path in sorted((project / "manifests").glob("*.json")):
        take(path, _shown(root, path), "a manifest")
    for record in show_project.qc_records(root, project, set(series), cfg["lane"]["variant"]):
        # A project run from another folder has its registry name files by their
        # absolute paths there (Munich's cryoet-runs); in tree/ they go by the
        # path from that folder, as the sandbox will have them.
        path = _resolve(root, record["path"])
        target = _shown(root, path)
        if Path(target).is_absolute():
            target = _inside(path, sorted(roots)) or target
        if Path(target).is_absolute():
            left.append({"path": target, "why": "outside the folders the jobs ran in"})
        else:
            take(path, target, f"the {record['qc_type']} QC record of {record['series']}")
    for step in case["steps"]:
        for one in series:
            for path, _ in show_failure.step_logs(project, one, step, None, None):
                take(path, _shown(root, path), f"a log of {step} for {one}")
    for pattern in case.get("extras", []):
        matched = sorted(path for path in project.glob(pattern) if path.is_file())
        if not matched:
            left.append({"path": f"{project_name}/{pattern}", "why": "nothing matches"})
        for path in matched:
            take(path, _shown(root, path), "a file the table names")

    # A stack lane's inputs move into the sandbox, onto empty stand-ins, so that a
    # dry run finds them; the lane files keep their keys as they were, but the
    # code reads them as it is now.
    moved: dict[str, str] = {}
    refused: list[str] = []
    for evidence in list(files.values()):
        if not evidence.path.endswith(".toml"):
            continue
        try:
            config = tomllib.loads(evidence.text)
        except tomllib.TOMLDecodeError:
            continue
        refused += [f"{evidence.path}: {key}" for key in run_series.unknown_keys(config)]
        if evidence.path == case["lane"] and not (config.get("run") or {}).get("project"):
            # A lane file from before [run] (12 October 2026): the tools need it.
            refused.append(f"{evidence.path}: no [run] project, which the tools need")
        here = moved_data(config)
        if here:
            for old, new in here.items():
                evidence.text = evidence.text.replace(f'"{old}"', f'"{new}"')
            evidence.text = _MOVED + evidence.text
            moved.update(here)
    for template in sorted(set(moved.values())):
        for one in series:
            stand_in = template.replace("{series}", one)
            files.setdefault(stand_in, Evidence(stand_in, "", until_case(lane_modified),
                                                "an empty stand-in for a stack or mdoc"))

    try:
        chain = run_series.build_steps(tomllib.loads(files[case["lane"]].text), project,
                                       series[0] if series else "TS_X")
    except (SystemExit, KeyError, TypeError, tomllib.TOMLDecodeError):
        chain = []
    named = {match for _, command in chain for part in command
             for match in submit_lanes._ENV_REFERENCE.findall(part)}
    env = {key: STAND_INS.get(key, f"/opt/{key.lower()}") for key in {*_ALWAYS, *named}}

    later = sum(1 for one in left if one["why"].startswith("written later"))
    until = case.get("until")
    note = [f"Collected at {case['cluster']} on {date.today().isoformat()}: the project as "
            + ("it was then." if until is None else
               f"it stood at {until.isoformat(sep=' ', timespec='minutes')}."
               if isinstance(until, datetime) else f"it stood at the end of {until.isoformat()}.")]
    if case.get("note"):
        note.append(case["note"])
    if later:
        note.append(f"Left out as written after the case: {later} files, which manifest.json "
                    "lists.")
    lost = [one["path"] for one in left if one["why"].startswith("not found")]
    if lost:
        note.append(f"Not found on the cluster: {', '.join(lost)}.")
    if moved:
        note.append("The lane file's [data] stack and mdoc paths are moved into the sandbox, "
                    "onto empty stand-ins.")
    if refused:
        note.append(f"Keys the code now refuses: {'; '.join(refused)}.")
    note.append("User names and home paths are placeholders, which setup.sh fills in.")
    text = case_text((root / EVAL_CASES).read_text(), name)
    manifest = {
        "case": name,
        "cluster": case["cluster"],
        "days": ([case["since"].isoformat(), case["until"].isoformat()]
                 if end is not None else None),
        "collected": _when(datetime.now(UTC)),
        "commit": _git_head(root),
        "files": [{"path": one.path, "modified": _when(datetime.fromtimestamp(one.modified,
                                                                            tz=UTC)),
                   "source": one.source} for one in files.values()],
        "left_out": left,
        "slurm": _slurm_details(case["jobs"]) if slurm and case.get("jobs") else {},
    }

    scrub = Scrub(sorted(roots), person)
    written = {scrub(one.path): scrub(one.text) for one in files.values()}
    manifest_text = scrub(json.dumps(manifest, indent=2, ensure_ascii=False)) + "\n"
    toml_text = scrub(case_toml(name, case["title"], text, " ".join(note),
                                case.get("playbook", []), env))
    checked = [*written.items(), ("manifest.json", manifest_text), ("case.toml", toml_text)]
    leftovers = sorted({f"{path}: {found}" for path, body in checked
                        for found in scrub.leftovers(f"{path}\n{body}")})
    if leftovers:
        raise SystemExit(
            f"{name}: nothing written, since these are still in it:\n  " + "\n  ".join(leftovers)
            + "\nName someone else with --other, another form of your own name with --name, "
            "or let a path through with --keep."
        )

    if folder.exists():
        shutil.rmtree(folder)
    for relative, body in written.items():
        destination = folder / "tree" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(body)
    (folder / "manifest.json").write_text(manifest_text)
    (folder / "case.toml").write_text(toml_text)
    (folder / "setup.sh").write_text(SETUP)

    kept: dict[str, int] = {}
    for body in written.values():
        for match in _PREFIX.finditer(body):
            kept[match.group(0)] = kept.get(match.group(0), 0) + 1
    reasons: dict[str, int] = {}
    for one in left:
        reason = one["why"].split(",")[0]
        reasons[reason] = reasons.get(reason, 0) + 1
    summary = [
        f"{name}: {len(written)} files in {folder}",
        f"  lane file: {scrub(lane_source)}",
        "  left out: " + (", ".join(f"{count} {reason}"
                                    for reason, count in sorted(reasons.items())) or "nothing"),
    ]
    summary += [f"  not found: {path}" for path in lost]
    summary += [f"  refused by the code now: {one}" for one in refused]
    summary.append("  paths kept as they are; look for anyone's name in them:")
    summary += [f"    {path}  ({count})" for path, count in
                sorted(kept.items(), key=lambda item: (-item[1], item[0]))[:20]]
    return folder, summary


def restore(folder: Path, sandbox: Path) -> None:
    """In a runner's sandbox: the placeholders become this sandbox's folder, home
    and user, and every file of the case gets the time it had on the cluster."""
    manifest = json.loads((folder / "manifest.json").read_text())
    values = {"ROOT": str(sandbox), "HOME": str(Path.home()), "USER": getpass.getuser()}
    for entry in manifest["files"]:
        path = sandbox / entry["path"]
        if not path.is_file():
            continue
        text = path.read_text()
        filled = PLACEHOLDER.sub(lambda match: values.get(match.group(1), f"other{match.group(2)}"),
                                 text)
        if filled != text:
            path.write_text(filled)
        moment = datetime.fromisoformat(entry["modified"]).timestamp()
        os.utime(path, (moment, moment))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*",
                        help="case IDs, such as E01; without them, every case of --cluster")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", action="store_true",
                      help="list what of each case's evidence is here, and copy nothing")
    mode.add_argument("--collect", action="store_true",
                      help="write each case as a folder for scripts/run_cases.py")
    mode.add_argument("--restore", type=Path, metavar="CASE_FOLDER",
                      help="in a runner's sandbox, as setup.sh does: fill in the placeholders "
                           "and give the files their times")
    parser.add_argument("--cluster", choices=CLUSTERS, help="the cases of this cluster")
    parser.add_argument("--json", action="store_true", help="print the inventory as JSON")
    parser.add_argument("--no-slurm", action="store_true",
                        help="do not ask sacct; read the logs in the repository root only")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="where --collect writes the case folders (~/cases)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a case folder --collect wrote before")
    parser.add_argument("--name", action="append", default=[],
                        help="another form of your own name, to become {{USER}}; repeat")
    parser.add_argument("--other", action="append", default=[],
                        help="someone else's name or login, to become {{OTHER1}} and on; repeat")
    parser.add_argument("--keep", action="append", default=[],
                        help="a path that only looks like a home folder, let through; repeat")
    args = parser.parse_args()
    if args.restore is not None:
        restore(args.restore, Path.cwd())
        return 0
    if args.collect and (args.cluster or not args.cases):
        parser.error("--collect takes case IDs, not --cluster")
    if args.inventory and bool(args.cases) == bool(args.cluster):
        parser.error("give case IDs or --cluster, one of the two")
    table = read_cases(ROOT / TABLE)
    unknown = [name for name in args.cases if name not in table]
    if unknown:
        parser.error(f"no case {', '.join(unknown)} in {TABLE}")
    if args.collect:
        person = this_person(args.name, args.other, args.keep)
        for name in args.cases:
            _, summary = collect(name, table[name], person, out=args.out,
                                 slurm=not args.no_slurm, overwrite=args.overwrite)
            print("\n".join(summary) + "\n")
        print(f"To bring them over: tar czf ~/cases.tgz -C {args.out} {' '.join(args.cases)}\n"
              "then, on the other side, from the repository root: "
              "tar xzf cases.tgz -C docs/agent/cases")
        return 0
    names = args.cases or [name for name, case in table.items() if case["cluster"] == args.cluster]
    entries = [inventory(name, table[name], slurm=not args.no_slurm) for name in names]
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        print("\n\n".join("\n".join(render(entry)) for entry in entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
