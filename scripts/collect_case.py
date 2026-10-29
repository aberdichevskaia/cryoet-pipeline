#!/usr/bin/env python3
"""What is still on this cluster of the evidence of each eval case
(docs/agent/eval_cases.md), before any of it is copied.

    python3 scripts/collect_case.py --inventory --cluster mdc
    python3 scripts/collect_case.py --inventory E01 E05
    python3 scripts/collect_case.py --inventory --cluster munich --json

Read-only: it copies and writes nothing. docs/agent/cases/cases.toml says, for
each case, its lane, the days it happened, and its steps, jobs and files. For
each case this prints:

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

Collecting the evidence, with user names and home paths replaced, comes once
the inventory has been read on both clusters.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import tomllib
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_failure  # noqa: E402
import show_project  # noqa: E402

TABLE = Path("docs/agent/cases/cases.toml")
CLUSTERS = ("mdc", "munich", "none")
# Every key a case may set. A misspelt one would be skipped without a word, and
# the inventory would look in the wrong place.
CASE_KEYS = frozenset({
    "title", "cluster", "lane", "series", "steps", "since", "until", "jobs",
    "log_pattern", "extras", "elsewhere", "note",
})
_LISTS = ("series", "steps", "jobs", "extras", "elsewhere")
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
        since, until = _day(case.get("since")), _day(case.get("until"))
        if since is None or until is None:
            found.append("since and until must both be days, such as 2026-10-14")
        elif since > until:
            found.append("until is before since")
    if "log_pattern" in case:
        try:
            re.compile(case["log_pattern"])
        except (re.error, TypeError) as error:
            found.append(f"log_pattern: {error}")
    return found


def _day(value: Any) -> date | None:
    """A TOML day, such as 2026-10-14; a date with a time of day is not one."""
    return value if isinstance(value, date) and not isinstance(value, datetime) else None


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
    """The case's days as instants: from the first day's midnight to the midnight
    after the last, in this machine's time zone, which is the cluster's."""
    if "since" not in case:
        return None, None
    start = datetime.combine(case["since"], time()).astimezone()
    end = datetime.combine(case["until"] + timedelta(days=1), time()).astimezone()
    return start, end


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*",
                        help="case IDs, such as E01; without them, every case of --cluster")
    parser.add_argument("--inventory", action="store_true",
                        help="list what of each case's evidence is here, and copy nothing")
    parser.add_argument("--cluster", choices=CLUSTERS, help="the cases of this cluster")
    parser.add_argument("--json", action="store_true", help="print the inventory as JSON")
    parser.add_argument("--no-slurm", action="store_true",
                        help="do not ask sacct; read the logs in the repository root only")
    args = parser.parse_args()
    if not args.inventory:
        parser.error("collecting is not written yet: --inventory lists what there is to collect")
    if bool(args.cases) == bool(args.cluster):
        parser.error("give case IDs or --cluster, one of the two")
    table = read_cases(ROOT / TABLE)
    unknown = [name for name in args.cases if name not in table]
    if unknown:
        parser.error(f"no case {', '.join(unknown)} in {TABLE}")
    names = args.cases or [name for name, case in table.items() if case["cluster"] == args.cluster]
    entries = [inventory(name, table[name], slurm=not args.no_slurm) for name in names]
    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        print("\n\n".join("\n".join(render(entry)) for entry in entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
