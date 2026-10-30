"""The evidence of an eval case: what of it is still on the cluster, what a later
run has written again, and the copy of it a case folder holds."""

import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import collect_case  # noqa: E402
import run_cases  # noqa: E402
import run_series  # noqa: E402
import show_failure  # noqa: E402
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
    entries = set(re.findall(r"^### ([A-Z]-\d+)\.",
                             (ROOT / "docs" / "agent" / "playbook.md").read_text(), flags=re.M))
    table = collect_case.read_cases(ROOT / collect_case.TABLE)
    assert {name: case["title"] for name, case in table.items()} == headings
    for name, case in table.items():
        assert set(case.get("playbook", [])) <= entries, name
        if case["cluster"] == "none":
            continue
        _, cfg = run_series.read_config(ROOT / case["lane"])
        chain = [step for step, _ in run_series.build_steps(cfg, ROOT / "outputs" / "x", "TS_X")]
        assert set(case["steps"]) <= set(chain), name


def test_the_command_takes_one_way_of_working():
    script = [sys.executable, str(ROOT / "scripts" / "collect_case.py")]
    refused = subprocess.run([*script, "E01"], capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "one of the arguments --inventory --collect --restore is required" in refused.stderr
    whole = subprocess.run([*script, "--collect", "--cluster", "mdc"], capture_output=True,
                           text=True, check=False)
    assert "--collect takes case IDs, not --cluster" in whole.stderr
    built = subprocess.run([*script, "--inventory", "--cluster", "none"],
                           capture_output=True, text=True, check=True)
    assert "E10  HIV-1 Gag through the safety filter\n  built by hand: " in built.stdout


# Collecting

PERSON = collect_case.Person(homes=["/fast/home/t/tester"], names=["tester", "tess.tester"])
EVAL = """# Evaluation cases

### E99. A crash while seeding

- **Class:** data.
- **Request:** "The CPU job failed. Why, and what do we
  do?"
- **Must not:**
  - propose a plain rerun;
  - submit without a `yes`.
- **Pass if:** it cites the crash; it asks before submitting.

### E100. Another case
"""
COLLECTED = {**CASE, "extras": ["picks/A/*", "fiducials/B/*.prexg"], "playbook": ["A-2"]}


def _collectable(root: Path) -> None:
    """_project as a cluster has it: its stacks in a folder named after the user,
    the user's home and the repository's folder in a job's log, and A's pick
    crashing in the case's days, its log written while it ran."""
    _project(root)
    lane = root / "configs" / "lanes" / "l.toml"
    text = lane.read_text()
    for old, new in (
        ('"data/empiar-11538/data/mdoc_files/{series}.mrc.mdoc"',
         '"/data/lab/tess.tester/EMPIAR-11538/mdoc/{series}.mrc.mdoc"'),
        ('"data/empiar-11538/data/Untreated_dataset/{series}.mrc"',
         '"/data/lab/tess.tester/EMPIAR-11538/{series}.mrc"'),
    ):
        assert old in text
        text = text.replace(old, new)
    lane.write_text(text)
    project = root / "outputs" / "p"
    timings = project / "step_timings.jsonl"
    with timings.open("a") as handle:
        row = _run("A", "pick-particles", _at(13, 30), 1, lane="ribo-bin8", variant="bin8")
        handle.write(json.dumps(row) + "\n")
    os.utime(timings, (LATER.timestamp(), LATER.timestamp()))
    _touch(project / "picks" / "A" / "A_pick.log", _at(13, 29),
           "cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 1,179,648,000 bytes\n")
    scores = project / "picks" / "A" / "A_scores.mrc"
    scores.write_bytes(b"MAP \0\0\0\1" * 64)
    os.utime(scores, (_at(13, 29).timestamp(), _at(13, 29).timestamp()))
    log = root / "slurm-p-cpu-123.log"
    _touch(log, _at(13, 31), log.read_text()
           + "env         /fast/home/t/tester/micromamba/envs/cryoet-pipeline (micromamba)\n"
           + f"cd {root}\n"
           + "stack /data/lab/Tess.Tester/EMPIAR-11538/B.mrc\n"
           + "# FAILED at pick-particles (exit 1) for A.\n")
    (root / "docs" / "agent").mkdir(parents=True)
    (root / "docs" / "agent" / "eval_cases.md").write_text(EVAL)


def _collected(root: Path, out: Path, person=PERSON, case=None):
    return collect_case.collect("E99", case or COLLECTED, person, root=root, out=out, slurm=False)


def _everything(folder: Path) -> str:
    """A case folder's paths, from the folder, and texts: what goes anywhere."""
    return "\n".join(f"{path.relative_to(folder)}\n{path.read_text()}"
                     for path in sorted(folder.rglob("*")) if path.is_file())


def test_a_collected_case_is_a_folder_the_runner_reads(tmp_path):
    _collectable(tmp_path / "cluster")
    folder, summary = _collected(tmp_path / "cluster", tmp_path / "cases")
    case = run_cases.load_case(folder)
    assert (case.id, case.title) == ("E99", "A crash while seeding")
    assert case.request == "The CPU job failed. Why, and what do we do?"
    assert case.must_not == ["propose a plain rerun", "submit without a yes"]
    assert case.pass_if == ["it cites the crash", "it asks before submitting"]
    assert case.playbook == ["A-2"]
    assert case.env["CTFFIND5"] == "{bin}/ctffind5"
    assert "Not found on the cluster: the Slurm log of job 777." in case.note
    assert "placeholders, which setup.sh fills in" in case.note
    assert (folder / "setup.sh").read_text().endswith(
        'python3 scripts/collect_case.py --restore "$(dirname "$0")"\n')
    tree = sorted(str(path.relative_to(folder / "tree")) for path in (folder / "tree").rglob("*")
                  if path.is_file())
    assert tree == [
        "configs/lanes/l.toml",
        "data/EMPIAR-11538/A.mrc",
        "data/EMPIAR-11538/B.mrc",
        "data/EMPIAR-11538/mdoc/A.mrc.mdoc",
        "data/EMPIAR-11538/mdoc/B.mrc.mdoc",
        "outputs/p/artifacts.json",
        "outputs/p/fiducials/B/B_seed.log",
        "outputs/p/journal.jsonl",
        "outputs/p/manifests/A.json",
        "outputs/p/manifests/B.json",
        "outputs/p/picks/A/A_pick.log",
        "outputs/p/qc/B/coarse_alignment_report.json",
        "outputs/p/step_timings.jsonl",
        "slurm-p-cpu-123.log",
    ]
    assert summary[0] == f"E99: 14 files in {folder}"


def test_names_and_home_folders_become_placeholders(tmp_path):
    _collectable(tmp_path / "cluster")
    folder, _ = _collected(tmp_path / "cluster", tmp_path / "cases")
    everything = _everything(folder)
    assert "tester" not in everything.lower()
    for gone in ("/fast/home", str(tmp_path)):
        assert gone not in everything
    log = (folder / "tree" / "slurm-p-cpu-123.log").read_text()
    assert "env         {{HOME}}/micromamba/envs/cryoet-pipeline (micromamba)" in log
    assert "cd {{ROOT}}\n" in log
    assert "stack /data/lab/{{USER}}/EMPIAR-11538/B.mrc" in log
    lane = (folder / "tree" / "configs" / "lanes" / "l.toml").read_text()
    assert lane.startswith(collect_case._MOVED)
    assert 'stack = "data/EMPIAR-11538/{series}.mrc"' in lane


def test_a_name_or_a_home_left_in_a_file_stops_the_collection(tmp_path):
    _collectable(tmp_path / "cluster")
    log = tmp_path / "cluster" / "slurm-p-cpu-123.log"
    _touch(log, _at(13, 31), log.read_text() + "tools in /home/lab/bobrova/tools\n")
    with pytest.raises(SystemExit) as refused:
        _collected(tmp_path / "cluster", tmp_path / "cases")
    assert "slurm-p-cpu-123.log: /home/lab/bobrova" in str(refused.value)
    assert not (tmp_path / "cases" / "E99").exists()
    # Named as someone else, or let through as a path that is nobody's.
    other = collect_case.Person(PERSON.homes, PERSON.names, others=["bobrova"])
    folder, _ = _collected(tmp_path / "cluster", tmp_path / "cases", person=other)
    assert "/home/lab/{{OTHER1}}/tools" in (folder / "tree" / "slurm-p-cpu-123.log").read_text()
    kept = collect_case.Person(PERSON.homes, PERSON.names, keep=["/home/lab/bobrova"])
    _collected(tmp_path / "cluster", tmp_path / "again", person=kept)
    with pytest.raises(SystemExit, match="too short to replace safely: bob"):
        collect_case.Scrub([], collect_case.Person([], ["bob"]))


def test_the_case_ends_where_its_last_day_ends(tmp_path):
    _collectable(tmp_path / "cluster")
    folder, _ = _collected(tmp_path / "cluster", tmp_path / "cases")
    tree = folder / "tree" / "outputs" / "p"
    timings = [json.loads(line) for line in (tree / "step_timings.jsonl").read_text().splitlines()]
    assert len(timings) == 6
    assert all(datetime.fromisoformat(row["finished_at"]) < LATER for row in timings)
    assert len((tree / "journal.jsonl").read_text().splitlines()) == 1
    # A's coarse QC was written by the later run: neither it nor its record is in.
    records = json.loads((tree / "artifacts.json").read_text())["artifacts"]
    assert [record["id"] for record in records] == ["B:qc:coarse_alignment_report",
                                                    "B:qc:fiducial_seed"]
    manifest = json.loads((folder / "manifest.json").read_text())
    left = {one["path"]: one["why"].split(",")[0] for one in manifest["left_out"]}
    assert left == {
        "outputs/p/qc/A/coarse_alignment_report.json": "written later",
        "outputs/p/qc/B/fiducial_seed.json": "missing",
        "outputs/p/fiducials/A/A_seed.log": "written later",
        "outputs/p/picks/A/A_scores.mrc": "binary",
        "outputs/p/fiducials/B/*.prexg": "nothing matches",
        "the Slurm log of job 777": "not found where the job ran",
    }
    modified = {one["path"]: datetime.fromisoformat(one["modified"]) for one in manifest["files"]}
    assert modified["outputs/p/fiducials/B/B_seed.log"] == _at(12, 59)
    assert modified["slurm-p-cpu-123.log"] == _at(13, 31)


def test_a_journal_begun_after_the_case_is_left_out_not_emptied(tmp_path):
    """The projects' journals began on 28 October 2026: a case before then had none."""
    root = tmp_path / "cluster"
    _collectable(root)
    _touch(root / "outputs" / "p" / "journal.jsonl", LATER,
           json.dumps({"at": "2026-10-21T12:00:00+00:00"}) + "\n")
    folder, _ = _collected(root, tmp_path / "cases")
    assert not (folder / "tree" / "outputs" / "p" / "journal.jsonl").exists()
    left = json.loads((folder / "manifest.json").read_text())["left_out"]
    assert {"path": "outputs/p/journal.jsonl", "why": "no row from before the case ended"} in left


def test_a_job_the_table_names_is_left_out_if_it_started_after_the_case(tmp_path):
    """E01's table once named 161b, the run that held the fix."""
    root = tmp_path / "cluster"
    _collectable(root)
    _touch(root / "slurm-p-cpu161b-778.log", LATER,
           f"job         p-cpu161b (778) on node3\nstarted     {LATER.isoformat()}\n"
           "project     outputs/p   config configs/lanes/l.toml\n")
    folder, _ = _collected(root, tmp_path / "cases", case={**COLLECTED, "jobs": ["778"]})
    assert not (folder / "tree" / "slurm-p-cpu161b-778.log").exists()
    left = json.loads((folder / "manifest.json").read_text())["left_out"]
    assert {"path": "slurm-p-cpu161b-778.log", "why": "started after the case ended"} in left


def test_a_registry_naming_files_by_absolute_paths_puts_them_where_the_sandbox_looks(tmp_path):
    """As Munich's does: the clone's outputs/p is a link into the folder the jobs
    run from, and the registry names files by their paths there."""
    root = tmp_path / "cluster"
    _collectable(root)
    runs = tmp_path / "runs"
    (runs / "outputs").mkdir(parents=True)
    (root / "outputs" / "p").rename(runs / "outputs" / "p")
    (root / "outputs" / "p").symlink_to(runs / "outputs" / "p")
    registry = runs / "outputs" / "p" / "artifacts.json"
    payload = json.loads(registry.read_text())
    for record in payload["artifacts"]:
        record["path"] = str(runs / record["path"])
    registry.write_text(json.dumps(payload))
    folder, _ = _collected(root, tmp_path / "cases")
    assert (folder / "tree" / "outputs/p/qc/B/coarse_alignment_report.json").is_file()
    assert not (folder / "tree" / "{{ROOT}}").exists()
    sandbox = tmp_path / "sandbox"
    shutil.copytree(folder / "tree", sandbox)
    collect_case.restore(folder, sandbox)
    [record] = show_project.qc_records(sandbox, sandbox / "outputs" / "p", {"B"}, "bin8")[:1]
    assert record["path"] == str(sandbox / "outputs/p/qc/B/coarse_alignment_report.json")
    # The document was found: its status, not the registry's copy, is the verdict.
    assert record["status"] == "pass"


def test_the_jobs_the_table_names_come_from_where_they_ran(tmp_path, named):
    _collectable(tmp_path / "cluster")
    folder = collect_case.collect("E99", COLLECTED, PERSON, root=tmp_path / "cluster",
                                  out=tmp_path / "cases")[0]
    log = (folder / "tree" / "slurm-q-gpu-777.log").read_text()
    assert "CUDA out of memory" in log
    manifest = json.loads((folder / "manifest.json").read_text())
    assert "a100_3g.20gb" in manifest["slurm"]["777"]["AllocTRES"]
    assert str(named) not in _everything(folder)


def test_the_sandbox_gets_its_own_paths_and_the_clusters_file_times(tmp_path):
    _collectable(tmp_path / "cluster")
    folder, _ = _collected(tmp_path / "cluster", tmp_path / "cases")
    sandbox = tmp_path / "sandbox"
    shutil.copytree(folder / "tree", sandbox)
    for path in sandbox.rglob("*"):
        os.utime(path)  # as a fresh checkout has them
    cfg = tomllib.loads((sandbox / "configs" / "lanes" / "l.toml").read_text())

    def pick_logs() -> list[str]:
        [pick] = [entry for entry in show_failure.failures(cfg, sandbox, slurm=False)["failures"]
                  if entry["step"] == "pick-particles"]
        return [log["path"] for log in pick["program_logs"]]

    # Without the cluster's times the crash's log is not the crash's.
    assert pick_logs() == []
    collect_case.restore(folder, sandbox)
    assert pick_logs() == ["outputs/p/picks/A/A_pick.log"]
    log = (sandbox / "slurm-p-cpu-123.log").read_text()
    assert f"cd {sandbox}\n" in log
    assert f"env         {Path.home()}/micromamba" in log
    assert f"/data/lab/{getpass.getuser()}/EMPIAR-11538/B.mrc" in log


def test_setup_sh_restores_in_a_sandbox_the_runner_builds(tmp_path):
    """The runner runs `bash <case>/setup.sh` in the sandbox, which holds the
    repository's scripts with the case's tree laid over them."""
    _collectable(tmp_path / "cluster")
    folder, _ = _collected(tmp_path / "cluster", tmp_path / "cases")
    sandbox = tmp_path / "sandbox"
    shutil.copytree(ROOT / "scripts", sandbox / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(folder / "tree", sandbox, dirs_exist_ok=True)
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(["bash", str(folder / "setup.sh")], cwd=sandbox, env=env, check=True)
    log = sandbox / "slurm-p-cpu-123.log"
    assert f"cd {sandbox}\n" in log.read_text()
    assert log.stat().st_mtime == _at(13, 31).timestamp()


def test_the_lane_file_is_the_one_of_the_cases_last_day(tmp_path):
    root = tmp_path / "cluster"
    _collectable(root)
    lane = root / "configs" / "lanes" / "l.toml"

    def commit(when: str) -> None:
        env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when,
               "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.org",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.org"}
        for command in (["add", "configs/lanes/l.toml"], ["commit", "-q", "-m", when]):
            subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                            *command], cwd=root, env=env, check=True)

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    lane.write_text("# as it was\n" + lane.read_text())
    commit("2026-10-10T12:00:00+00:00")
    lane.write_text(lane.read_text().replace("# as it was\n", "# changed since\n"))
    commit("2026-10-20T12:00:00+00:00")
    folder, summary = _collected(root, tmp_path / "cases")
    collected = (folder / "tree" / "configs" / "lanes" / "l.toml").read_text()
    assert "# as it was" in collected and "# changed since" not in collected
    assert "the lane file's last before the case ended" in summary[1]


def test_a_case_can_end_at_a_time_of_day(tmp_path):
    """A failure fixed the same day: the case ends before the fix."""
    _collectable(tmp_path / "cluster")
    before_fix = {**COLLECTED, "until": datetime(2026, 10, 14, 12, 30, tzinfo=UTC)}
    assert collect_case._case_problems(before_fix) == []
    folder, _ = _collected(tmp_path / "cluster", tmp_path / "cases", case=before_fix)
    tree = folder / "tree" / "outputs" / "p"
    timings = [json.loads(line) for line in (tree / "step_timings.jsonl").read_text().splitlines()]
    assert [(row["series"], row["step"], row["exit_code"]) for row in timings] == [
        ("A", "generate-fiducial-seed", 0),
        ("B", "generate-fiducial-seed", 139),
        ("B", "pick-particles", 1),
    ]
    # The seeding log is the one the passing run wrote at 12:59.
    assert not (tree / "fiducials" / "B" / "B_seed.log").exists()
    assert "it stood at 2026-10-14 12:30" in run_cases.load_case(folder).note


def test_a_config_a_job_names_is_taken_only_as_it_was(tmp_path):
    """A copy changed after the case may hold the fix the case asks for."""
    root = tmp_path / "cluster"
    _collectable(root)
    for job, name, written in (("124", "early", _at(11, 50)), ("125", "late", LATER),
                               ("126", "late", LATER)):
        _touch(root / "outputs" / "p" / f"{name}.toml", written, "[lane]\n")
        _touch(root / f"slurm-p-gpu-{job}.log", _at(12, 10),
               f"job         p-gpu ({job}) on node2\nstarted     {_at(12).isoformat()}\n"
               f"project     outputs/p   config outputs/p/{name}.toml\n")
    folder, _ = _collected(root, tmp_path / "cases")
    assert (folder / "tree" / "outputs" / "p" / "early.toml").is_file()
    assert not (folder / "tree" / "outputs" / "p" / "late.toml").exists()
    left = json.loads((folder / "manifest.json").read_text())["left_out"]
    # Once, though two jobs name it.
    assert [one for one in left if one["path"] == "outputs/p/late.toml"] == [
        {"path": "outputs/p/late.toml",
         "why": f"written later, {LATER.astimezone().isoformat(timespec='seconds')}"}]


def test_a_lane_file_from_before_its_run_section_is_said_to_need_one(tmp_path):
    root = tmp_path / "cluster"
    _collectable(root)
    lane = root / "configs" / "lanes" / "l.toml"
    current = lane.read_text()
    kept, dropping = [], False
    for line in current.splitlines(keepends=True):
        if line.startswith("["):
            dropping = line.startswith(("[run]", "[[run."))
        if not dropping:
            kept.append(line)
    before = "".join(kept)
    assert "run" not in tomllib.loads(before)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-10-10T12:00:00+00:00",
           "GIT_COMMITTER_DATE": "2026-10-10T12:00:00+00:00", "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.org", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.org"}
    lane.write_text(before)
    for command in (["add", "configs/lanes/l.toml"], ["commit", "-q", "-m", "before [run]"]):
        subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
                        *command], cwd=root, env=env, check=True)
    lane.write_text(current)  # the project is found from the lane file as it is now
    folder, summary = _collected(root, tmp_path / "cases")
    assert "  refused by the code now: configs/lanes/l.toml: no [run] project, which the " \
           "tools need" in summary
    assert "no [run] project, which the tools need" in run_cases.load_case(folder).note


def test_a_case_is_written_once_and_only_where_its_project_is(tmp_path):
    _collectable(tmp_path / "cluster")
    _collected(tmp_path / "cluster", tmp_path / "cases")
    with pytest.raises(SystemExit, match="exists: remove it, or give --overwrite"):
        _collected(tmp_path / "cluster", tmp_path / "cases")
    collect_case.collect("E99", COLLECTED, PERSON, root=tmp_path / "cluster",
                         out=tmp_path / "cases", slurm=False, overwrite=True)
    with pytest.raises(SystemExit, match="E10 is built by hand"):
        collect_case.collect("E10", {"title": "t", "cluster": "none"}, PERSON)


def test_the_request_and_the_grading_lines_are_eval_cases(tmp_path):
    text = (ROOT / "docs" / "agent" / "eval_cases.md").read_text()
    e01 = collect_case.case_text(text, "E01")
    assert e01["request"] == "The CPU job of ryr1 failed. Why, and what do we do?"
    assert e01["must_not"] == ["propose a plain rerun", "change the binning for every series",
                               "edit the lane file for a one-off", "submit without a yes"]
    assert e01["pass_if"][0] == "it cites the segfault (file and line)"
    e02 = collect_case.case_text(text, "E02")
    assert e02["must_not"] == ["propose rerunning alignment",
                               "call the tomogram bad on the coarse QC alone"]
    assert e02["pass_if"] == ["the fine-alignment residual is named as the deciding number"]
    e03 = collect_case.case_text(text, "E03")
    assert e03["request"] == "The CTF in the report looks bad beyond ±12° for 039. What's wrong?"
    assert collect_case.case_text(text, "E05")["must_not"] == ["change CTF parameters"]


def test_a_full_name_gives_the_forms_folders_use():
    assert collect_case.name_forms("Tess Tester,,,") == [
        "tess.tester", "t.tester", "tess_tester", "tess tester", "tester"]
    assert collect_case.name_forms("tester") == []


def test_no_collected_case_in_the_repository_holds_a_home_folder():
    for manifest in sorted((ROOT / "docs" / "agent" / "cases").glob("*/manifest.json")):
        everything = _everything(manifest.parent)
        found = [match.group(0) for match in collect_case._HOME.finditer(everything)
                 if "{{" not in match.group(0)]
        assert found == [], manifest.parent.name
