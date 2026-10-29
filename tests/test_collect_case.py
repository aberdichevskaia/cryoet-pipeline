"""The inventory of an eval case's evidence: what of it is still on the cluster,
and what a later run has written again."""

import json
import os
import re
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_case  # noqa: E402
import run_series  # noqa: E402
import show_project  # noqa: E402

# Noon-ish in UTC, so that every moment stays on its day in the time zones the
# clusters and the Mac use.
DAY = datetime(2026, 10, 14, tzinfo=UTC)
LATER = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
CASE = {
    "title": "A crash while seeding",
    "cluster": "mdc",
    "lane": "configs/lanes/l.toml",
    "steps": ["qc-coarse-alignment", "generate-fiducial-seed", "pick-particles"],
    "since": date(2026, 10, 14),
    "until": date(2026, 10, 14),
    "jobs": ["777"],
    "log_pattern": "Segmentation fault|out of memory",
    "extras": ["fiducials/B/*.prexg", "fiducials/*/*.log"],
}


def _at(hour: int, minute: int = 0) -> datetime:
    return DAY.replace(hour=hour, minute=minute)


def _touch(path: Path, moment: datetime, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (moment.timestamp(), moment.timestamp()))
    return path


def _run(series: str, step: str, finished: datetime, exit_code: int = 0, **lane: str) -> dict:
    return {"series": series, "step": step, "seconds": 60.0, "disk_delta_bytes": 0,
            "exit_code": exit_code, "finished_at": finished.isoformat(), **lane}


def _qc(root: Path, series: str, qc_type: str, written: datetime | None) -> dict:
    path = root / "outputs" / "p" / "qc" / series / f"{qc_type}.json"
    if written is not None:
        _touch(path, written, json.dumps({"status": "pass"}))
    return {"id": f"{series}:qc:{qc_type}", "kind": "qc", "path": str(path.relative_to(root)),
            "variant": None, "parent_ids": [],
            "parameters": {"qc_type": qc_type, "tilt_series_id": series, "status": "pass"}}


def _project(root: Path) -> None:
    """B's seeding crashed and passed again the same day; A seeded that day and
    again six days later, which wrote its log and its coarse QC again."""
    text = (ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text()
    text = text.replace('project = "outputs/ribo2"', 'project = "outputs/p"')
    text, replaced = re.subn(r"^series = \[.*\]$", 'series = ["A", "B"]', text, flags=re.M)
    assert replaced == 1 and 'project = "outputs/p"' in text
    (root / "configs" / "lanes").mkdir(parents=True)
    (root / "configs" / "lanes" / "l.toml").write_text(text)

    project = root / "outputs" / "p"
    this_lane = {"lane": "ribo-bin8", "variant": "bin8"}
    rows = [
        _run("A", "generate-fiducial-seed", _at(10), **this_lane),
        _run("B", "generate-fiducial-seed", _at(11), 139, **this_lane),
        # Another lane's pick writes under its own label, and a row from before
        # rows named their lane counts for every lane.
        _run("B", "pick-particles", _at(12), 1, lane="ribo-bin4", variant="bin4"),
        _run("B", "pick-particles", _at(12, 30)),
        _run("B", "generate-fiducial-seed", _at(13), **this_lane),
        _run("A", "generate-fiducial-seed", LATER, **this_lane),
    ]
    _touch(project / "step_timings.jsonl", LATER,
           "".join(json.dumps(row) + "\n" for row in rows))
    _touch(project / "journal.jsonl", LATER,
           json.dumps({"at": "2026-10-13T12:00:00+00:00"}) + "\n"
           + json.dumps({"at": "2026-10-21T12:00:00+00:00"}) + "\n")
    # The passing run of B wrote the seeding log the crash had written before it.
    _touch(project / "fiducials" / "B" / "B_seed.log", _at(12, 59))
    _touch(project / "fiducials" / "A" / "A_seed.log", LATER)
    for series in ("A", "B"):
        _touch(project / "manifests" / f"{series}.json", _at(9), "{}")
    artifacts = [
        _qc(root, "B", "coarse_alignment_report", _at(9)),
        _qc(root, "A", "coarse_alignment_report", LATER),
        _qc(root, "B", "fiducial_seed", None),
    ]
    (project / "artifacts.json").write_text(json.dumps({"artifacts": artifacts}))

    _touch(root / "slurm-p-cpu-123.log", _at(13), (
        "job         p-cpu (123) on node1\n"
        f"started     {_at(10, 58).isoformat()}\n"
        "project     outputs/p   config configs/lanes/l.toml\n"
        "\n# ---- B (ribo-bin8) ----\n"
        "/bin/sh: line 1: 4242 Segmentation fault (core dumped) beadtrack\n"
        "# FAILED at generate-fiducial-seed (exit 139) for B.\n"
    ))
    _touch(root / "slurm-p-cpu-150.log", LATER, (
        "job         p-cpu (150) on node1\n"
        f"started     {LATER.isoformat()}\n"
        "project     outputs/p   config configs/lanes/l.toml\n"
        "Segmentation fault, again\n"
        f"finished    {LATER.isoformat()}\n"
    ))


@pytest.fixture
def named(tmp_path_factory, monkeypatch):
    """Job 777 ran another project, from another folder; sacct says which."""
    runs = tmp_path_factory.mktemp("runs")
    (runs / "slurm-q-gpu-777.log").write_text(
        "job         q-gpu (777) on gpu07\nproject     outputs/q   config q.toml\n"
        "torch.OutOfMemoryError: CUDA out of memory.\n"
    )

    def slurm(command):
        if "-j" not in command:
            return []
        if "JobID,JobName,WorkDir" in command:
            return [["777", "q-gpu", str(runs)]]
        return [["777", "q-gpu", "FAILED", "1:0", "2026-10-14T12:58:00", "00:12:40",
                 "gpu07", "billing=4,gres/gpu:a100_3g.20gb=1"]]

    monkeypatch.setattr(show_project, "_slurm", slurm)
    return runs


def test_a_run_in_the_cases_days_shows_the_logs_it_wrote_then_and_the_runs_since(tmp_path):
    _project(tmp_path)
    runs = collect_case.inventory("E99", CASE, tmp_path, slurm=False)["runs"]
    found = {(run["series"], run["step"], run["exit_code"]): run for run in runs}
    # The later run on another day is not the case's, and another lane's pick
    # writes elsewhere.
    assert list(found) == [
        ("A", "generate-fiducial-seed", 0),
        ("B", "generate-fiducial-seed", 139),
        ("B", "pick-particles", 0),
        ("B", "generate-fiducial-seed", 0),
    ]
    # The crash's log was written again by the run that passed the same day.
    crash = found[("B", "generate-fiducial-seed", 139)]
    assert crash["logs_then"] == []
    assert [run["exit_code"] for run in crash["runs_since"]] == [0]
    passed = found[("B", "generate-fiducial-seed", 0)]
    assert [log["path"] for log in passed["logs_then"]] == ["outputs/p/fiducials/B/B_seed.log"]
    assert passed["runs_since"] == []
    # A's run of that day left nothing: its log is the later run's.
    first = found[("A", "generate-fiducial-seed", 0)]
    assert first["logs_then"] == []
    assert len(first["runs_since"]) == 1


def test_a_file_written_after_the_cases_last_day_is_a_later_runs(tmp_path):
    _project(tmp_path)
    entry = collect_case.inventory("E99", CASE, tmp_path, slurm=False)
    qc = {(record["series"], record["qc_type"]): record["state"] for record in entry["qc"]}
    assert qc == {
        ("B", "coarse_alignment_report"): "as of the case",
        ("A", "coarse_alignment_report"): "written later",
        ("B", "fiducial_seed"): "missing",
    }
    extras = {extra["pattern"]: [(one["path"], one["state"]) for one in extra["files"]]
              for extra in entry["extras"]}
    assert extras == {
        "fiducials/B/*.prexg": [],
        "fiducials/*/*.log": [("outputs/p/fiducials/A/A_seed.log", "written later"),
                              ("outputs/p/fiducials/B/B_seed.log", "as of the case")],
    }
    state = entry["state"]
    assert state["registry"]["state"] == "written later"
    assert (state["step_timings"]["rows"], state["step_timings"]["rows_later"]) == (5, 1)
    assert (state["journal"]["rows"], state["journal"]["rows_later"]) == (1, 1)
    assert {one["state"] for one in state["manifests"]} == {"as of the case"}


def test_the_jobs_are_those_named_those_of_its_days_and_those_its_pattern_finds(
    tmp_path, named
):
    _project(tmp_path)
    entry = collect_case.inventory("E99", CASE, tmp_path)
    jobs = entry["jobs"]
    # Wherever sacct says it ran, whatever project it ran.
    [job] = jobs["named"]["777"]
    assert job["log"] == str(named / "slurm-q-gpu-777.log")
    assert job["match"] == {"line": 3, "text": "torch.OutOfMemoryError: CUDA out of memory."}
    assert "a100_3g.20gb" in entry["slurm"]["777"]["AllocTRES"]
    assert [job["id"] for job in jobs["in_days"]] == ["123"]
    # At any date: the pattern is how the jobs the table does not name are found.
    assert {job["id"]: job["match"]["line"] for job in jobs["matching"]} == {"123": 6, "150": 4}


def test_a_case_without_days_is_the_project_as_it_is_now(tmp_path):
    _project(tmp_path)
    now = {key: value for key, value in CASE.items() if key not in ("since", "until")}
    entry = collect_case.inventory("E99", now, tmp_path, slurm=False)
    assert "runs" not in entry and entry["jobs"]["in_days"] == []
    assert {record["state"] for record in entry["qc"]} == {"as of the case", "missing"}
    assert entry["state"]["step_timings"]["rows_later"] == 0
    text = "\n".join(collect_case.render(entry))
    assert "days none, the project as it is now" in text
    assert "started in its days" not in text


def test_the_text_says_what_is_gone_and_where_the_rest_is(tmp_path, named):
    _project(tmp_path)
    lines = collect_case.render(collect_case.inventory("E99", CASE, tmp_path))
    text = "\n".join(lines)
    assert "  mdc  configs/lanes/l.toml  outputs/p  days 2026-10-14..2026-10-14" in lines
    [crash] = [line for line in lines if "generate-fiducial-seed  exit 139" in line]
    assert "logs written then 0  run since 1" in crash
    assert "      Slurm: JobName q-gpu  State FAILED  ExitCode 1:0" in text
    assert "      line 6: /bin/sh: line 1: 4242 Segmentation fault (core dumped) beadtrack" in lines
    assert "  jobs whose log matches 'Segmentation fault|out of memory', at any date: 2" in lines
    assert "  QC records: 1 as of the case, 1 written later, 1 missing" in lines
    assert "  fiducials/B/*.prexg: missing" in lines
    assert "  step timings  rows: 5 up to its last day, 1 after" in lines


def test_a_project_that_is_not_here_is_said_so(tmp_path):
    _project(tmp_path)
    elsewhere = {**CASE, "lane": "configs/lanes/m.toml"}
    text = (tmp_path / "configs" / "lanes" / "l.toml").read_text()
    (tmp_path / "configs" / "lanes" / "m.toml").write_text(
        text.replace('project = "outputs/p"', 'project = "outputs/gone"')
    )
    entry = collect_case.inventory("E99", elsewhere, tmp_path, slurm=False)
    assert entry["here"] is False
    assert "  not here: there is no outputs/gone on this machine" in collect_case.render(entry)


def test_a_step_the_lane_does_not_have_is_refused(tmp_path):
    _project(tmp_path)
    with pytest.raises(SystemExit, match="E99: no step track-fiducial in configs/lanes/l.toml"):
        collect_case.inventory("E99", {**CASE, "steps": ["track-fiducial"]}, tmp_path,
                               slurm=False)


def test_the_table_refuses_a_case_it_cannot_follow(tmp_path):
    table = tmp_path / "cases.toml"
    table.write_text(
        '[E01]\ntitle = "a"\ncluster = "berlin"\n\n'
        '[E02]\ntitle = "b"\ncluster = "mdc"\nlane = "x.toml"\nstpes = []\n'
        'series = "039"\nsince = 2026-10-14\n\n'
        '[E03]\ntitle = "c"\ncluster = "mdc"\nlane = "x.toml"\nsteps = []\n'
        'since = 2026-10-15\nuntil = 2026-10-14\nlog_pattern = "("\n'
    )
    with pytest.raises(SystemExit) as refused:
        collect_case.read_cases(table)
    message = str(refused.value)
    for problem in (
        "E01: cluster must be one of mdc, munich, none",
        "E02: unknown key 'stpes'",
        "E02: no steps; [] for none",
        "E02: series must be a list of strings",
        "E02: since and until must both be days",
        "E03: until is before since",
        "E03: log_pattern: missing ),",
    ):
        assert problem in message


def test_the_table_has_every_case_of_eval_cases_with_steps_its_lane_has():
    headings = dict(re.findall(r"^### (E\d+)\. (.+)$",
                               (ROOT / "docs" / "agent" / "eval_cases.md").read_text(),
                               flags=re.M))
    table = collect_case.read_cases(ROOT / collect_case.TABLE)
    assert {name: case["title"] for name, case in table.items()} == headings
    for name, case in table.items():
        if case["cluster"] == "none":
            continue
        _, cfg = run_series.read_config(ROOT / case["lane"])
        chain = [step for step, _ in run_series.build_steps(cfg, ROOT / "outputs" / "x", "TS_X")]
        assert set(case["steps"]) <= set(chain), name


def test_the_command_only_takes_inventories_for_now():
    script = [sys.executable, str(ROOT / "scripts" / "collect_case.py")]
    refused = subprocess.run([*script, "E01"], capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "collecting is not written yet" in refused.stderr
    built = subprocess.run([*script, "--inventory", "--cluster", "none"],
                           capture_output=True, text=True, check=True)
    assert "E10  HIV-1 Gag through the safety filter\n  built by hand: " in built.stdout
