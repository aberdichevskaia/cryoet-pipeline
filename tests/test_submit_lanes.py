"""The checks scripts/submit_lanes.py makes before it submits: outputs are
overwritten only with --overwrite, every dry run and submission goes into the
project's journal, and the assistant submits only the dry run it showed."""

import json
import os
import re
import sys
import time
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import submit_lanes  # noqa: E402

# Stands in for sbatch: notes the OVERWRITE it was given and its arguments, fails
# on the call SBATCH_FAIL_AT counts to, and otherwise answers with a job id.
SBATCH = """#!/bin/bash
echo "${OVERWRITE:-}|$*" >> "$SBATCH_CALLS"
n=$(wc -l < "$SBATCH_CALLS" | tr -d ' ')
if [[ "$n" == "${SBATCH_FAIL_AT:-0}" ]]; then
    echo "sbatch: error: Batch job submission failed: Invalid qos specification" >&2
    exit 1
fi
echo $((1000 + n))
"""
# Stands in for squeue: fails with SQUEUE_EXIT, and otherwise lists SQUEUE_LINES.
SQUEUE = """#!/bin/bash
if [[ -n "${SQUEUE_EXIT:-}" ]]; then
    echo "squeue: error: Unable to contact slurm controller" >&2
    exit "$SQUEUE_EXIT"
fi
if [[ -n "${SQUEUE_LINES:-}" ]]; then printf '%s\\n' "$SQUEUE_LINES"; fi
"""
# Stands in for CTFFIND: its start-up banner, with FAKE_CTFFIND_VERSION, then
# its first question; it reads nothing and stops.
CTFFIND = """#!/bin/bash
echo "        **   Welcome to Ctffind   **"
echo ""
echo "            Version : ${FAKE_CTFFIND_VERSION:-5.0.2}"
echo "    Library Version : 2.0.0-alpha-295"
echo ""
echo "Input image file name [input.mrc]                  :"
"""
REASON = ["--reason", "Anna asked: rerun TS_A and TS_B"]


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """ribo-bin8 moved onto two empty stacks and a project in tmp_path, submitted
    by a person, with sbatch and squeue replaced by SBATCH and SQUEUE."""

    data = tmp_path / "data"
    data.mkdir()
    for series in ("TS_A", "TS_B"):
        (data / f"{series}.mrc").write_bytes(b"")
        (data / f"{series}.mrc.mdoc").write_text("")
    text = (ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text()
    for pattern, value in (
        (r'^project = ".*"$', f'project = "{tmp_path / "project"}"'),
        (r"^series = \[.*\]$", 'series = ["TS_A", "TS_B"]'),
        (r'^mdoc = ".*"$', f'mdoc = "{data}/{{series}}.mrc.mdoc"'),
        (r'^stack = ".*"$', f'stack = "{data}/{{series}}.mrc"'),
    ):
        text, count = re.subn(pattern, lambda _, v=value: v, text, count=1, flags=re.MULTILINE)
        assert count == 1, f"ribo-bin8.toml has no line {pattern}"
    config = tmp_path / "lane.toml"
    config.write_text(text)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # A CTFFIND 5 ahead of whatever CTFFIND the machine has (MDC's is 4.1.14).
    for name, script in (("sbatch", SBATCH), ("squeue", SQUEUE), ("ctffind5", CTFFIND)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SBATCH_CALLS", str(tmp_path / "sbatch_calls"))
    monkeypatch.setenv("PYTOMO_MP_DIR", "/opt/pytom")
    # A person, whatever runs the tests, until a test says otherwise.
    for name in (
        *submit_lanes.CLIENT_VARIABLES, "OVERWRITE", "SBATCH_FAIL_AT", "SQUEUE_EXIT",
        "SQUEUE_LINES", "FAKE_CTFFIND_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    return config


def _run(capsys, config: Path, *args: str) -> tuple[int, str, str]:
    code = submit_lanes.main([str(config), *args])
    out, err = capsys.readouterr()
    return code, out, err


def _journal(config: Path) -> list[dict]:
    path = config.parent / "project" / submit_lanes.JOURNAL
    return [json.loads(line) for line in path.read_text().splitlines()]


def _overwrite_per_call(config: Path) -> list[str]:
    """The OVERWRITE each sbatch call saw."""
    path = config.parent / "sbatch_calls"
    if not path.exists():
        return []
    return [line.split("|", 1)[0] for line in path.read_text().splitlines()]


def test_overwrite_reaches_the_jobs_only_from_the_flag(lane, capsys):
    assert _run(capsys, lane)[0] == 0
    assert _run(capsys, lane, "--overwrite")[0] == 0
    assert _overwrite_per_call(lane) == ["", "", "1", "1"]


@pytest.mark.parametrize("dry_run", [[], ["--dry-run"]], ids=["submit", "dry-run"])
def test_overwrite_in_the_environment_is_refused(lane, monkeypatch, dry_run):
    # Left in a shell by one rerun, it would overwrite in every later one.
    monkeypatch.setenv("OVERWRITE", "1")
    with pytest.raises(SystemExit, match="give --overwrite"):
        submit_lanes.main([str(lane), *dry_run])
    assert not _overwrite_per_call(lane)


def test_submit_does_not_pass_on_an_inherited_overwrite_either(lane, monkeypatch):
    # For a caller other than main(), which refuses it before this point.
    monkeypatch.setenv("OVERWRITE", "1")
    plan = submit_lanes.plan(tomllib.loads(lane.read_text()), lane)
    submit_lanes.submit(plan, plan.stages, dry_run=False)
    assert _overwrite_per_call(lane) == ["", ""]


def test_the_assistant_submits_the_dry_run_it_showed_and_only_once(lane, capsys, monkeypatch):
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "no dry run of it is on record" in err

    code, out, _ = _run(capsys, lane, "--dry-run")
    assert code == 0 and "within 30 minutes" in out
    assert _run(capsys, lane, *REASON)[0] == 0
    assert len(_overwrite_per_call(lane)) == 2

    # The same dry run again would queue a second chain on the same series.
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "it was submitted at" in err
    assert len(_overwrite_per_call(lane)) == 2


def test_a_submission_that_differs_from_its_dry_run_is_refused(lane, capsys, monkeypatch):
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    assert _run(capsys, lane, "--dry-run")[0] == 0
    for args, said in (
        (["--overwrite"], "only one of them has --overwrite"),
        (["--from-stage", "gpu"], "the stages differ"),
    ):
        code, _, err = _run(capsys, lane, *args, *REASON)
        assert code == 1 and said in err

    monkeypatch.setenv("PYTOMO_MP_DIR", "/elsewhere/pytom")
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "$PYTOMO_MP_DIR differs" in err
    monkeypatch.setenv("PYTOMO_MP_DIR", "/opt/pytom")

    lane.write_text(lane.read_text() + "# edited after the dry run\n")
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "the config file has changed" in err
    assert not _overwrite_per_call(lane)


def test_a_stale_a_foreign_or_a_failing_dry_run_does_not_count(lane, capsys, monkeypatch):
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    path = lane.parent / "project" / submit_lanes.JOURNAL

    def edit_last_entry(**fields):
        rows = _journal(lane)
        rows[-1].update(fields)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    _run(capsys, lane, "--dry-run")
    edit_last_entry(at=(datetime.now(UTC) - timedelta(minutes=31)).isoformat(timespec="seconds"))
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "31 minutes old" in err

    _run(capsys, lane, "--dry-run")
    edit_last_entry(user="someone-else")
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "was someone-else's" in err

    # A dry run that would have been refused is not one to agree to, even after
    # what it found has been put right.
    stack = lane.parent / "data" / "TS_B.mrc"
    stack.unlink()
    assert "once the problems above are resolved" in _run(capsys, lane, "--dry-run")[1]
    stack.write_bytes(b"")
    code, _, err = _run(capsys, lane, *REASON)
    assert code == 1 and "found problems" in err
    assert not _overwrite_per_call(lane)


def test_the_assistant_says_why(lane, monkeypatch):
    monkeypatch.setenv("AI_AGENT", "some-client_1-0_agent")
    with pytest.raises(SystemExit, match="needs --reason"):
        submit_lanes.main([str(lane)])
    assert not _overwrite_per_call(lane)


def test_the_journal_says_what_was_shown_and_submitted_and_by_whom(lane, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "session-1")
    monkeypatch.setenv("CRYOET_ASSISTANT_MODEL", "model-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-for-the-journal")
    _run(capsys, lane, "--dry-run")
    assert _run(capsys, lane, *REASON)[0] == 0

    shown, submitted, result = _journal(lane)
    assert [shown["event"], submitted["event"], result["event"]] == [
        "dry-run", "submit", "submitted",
    ]
    assert shown["plan"] == submitted["plan"]
    assert submitted["reason"] == REASON[1]
    assert submitted["config_text"] == lane.read_text()
    assert submitted["assistant"] is True
    assert submitted["client"] == {
        "CLAUDECODE": "1",
        "CLAUDE_CODE_SESSION_ID": "session-1",
        "CRYOET_ASSISTANT_MODEL": "model-x",
    }
    assert result["id"] == submitted["id"]
    assert result["jobs"] == {"cpu": "1001", "gpu": "1002"}
    assert result["series"] == ["TS_A", "TS_B"]
    assert "not-for-the-journal" not in (lane.parent / "project" / submit_lanes.JOURNAL).read_text()


def test_a_config_copy_is_journaled_with_its_difference_from_the_lane_file(tmp_path):
    source = ROOT / "configs" / "lanes" / "ribo-bin8.toml"
    text = source.read_text() + "# one series only\n"
    cfg = tomllib.loads(text)
    diff = submit_lanes.config_diff(cfg, tmp_path / "copy.toml", text)
    assert diff["from"] == "configs/lanes/ribo-bin8.toml"
    assert "+# one series only" in diff["diff"]
    # A lane file itself is identified by the commit instead.
    assert submit_lanes.config_diff(cfg, source, text) is None


def test_a_refused_stage_is_journaled_with_the_jobs_already_queued(lane, capsys, monkeypatch):
    monkeypatch.setenv("SBATCH_FAIL_AT", "2")
    code, _, err = _run(capsys, lane)
    assert code == 1
    assert "Invalid qos" in err and "scancel 1001" in err
    failed = _journal(lane)[-1]
    assert failed["event"] == "submit-failed"
    assert failed["jobs"] == {"cpu": "1001"}


def test_overwrite_lists_the_recorded_runs_it_may_replace(lane, capsys):
    project = lane.parent / "project"
    project.mkdir()
    rows = [
        # This lane's own run of a labelled step.
        {"series": "TS_A", "step": "reconstruct-tomogram", "variant": "bin8", "exit_code": 0},
        # A shared step, whichever lane ran it, and whether or not it finished.
        {"series": "TS_B", "step": "align-tilt-series", "variant": "bin4", "exit_code": 1},
        # Another lane's run of a labelled step is not this lane's to replace.
        {"series": "TS_B", "step": "pick-particles", "variant": "bin4", "exit_code": 0},
    ]
    (project / "step_timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    code, out, _ = _run(capsys, lane, "--dry-run", "--overwrite")
    assert code == 0
    assert "OVERWRITE='1'" in out
    listed = out.split("may replace what these recorded runs wrote:\n")[1].splitlines()[:3]
    assert listed[:2] == ["    TS_A: reconstruct-tomogram", "    TS_B: align-tilt-series"]
    assert not listed[2].startswith("    ")


def test_a_series_that_a_queued_job_of_the_project_works_on_is_refused(lane, capsys, monkeypatch):
    assert _run(capsys, lane)[0] == 0  # jobs 1001 and 1002, on TS_A and TS_B
    monkeypatch.setenv("SQUEUE_LINES", "1001|project-cpu|RUNNING\n1002|project-gpu|PENDING")

    code, _, err = _run(capsys, lane, "--dry-run")
    assert code == 0
    assert "would refuse" in err
    assert "job 1001 (project-cpu, running) works on TS_A, TS_B" in err
    assert "job 1002 (project-gpu, pending) works on TS_A, TS_B" in err

    code, _, err = _run(capsys, lane)
    assert code == 1 and "refused" in err
    assert len(_overwrite_per_call(lane)) == 2  # nothing more was submitted


def test_a_jobs_series_come_from_the_journal_then_its_log_then_its_name(lane, tmp_path):
    plan = submit_lanes.plan(tomllib.loads(lane.read_text()), lane)
    project = tmp_path / "project"
    project.mkdir()
    journal = [
        {"event": "submitted", "jobs": {"cpu": "2001"}, "series": ["TS_B"]},
        {"event": "submitted", "jobs": {"cpu": "2002"}, "series": ["TS_C"]},
    ]
    (project / submit_lanes.JOURNAL).write_text("".join(json.dumps(row) + "\n" for row in journal))
    logs = {
        # A recipe's header names series and project on one line.
        "slurm-ctf-window-3001.log": f"job x\nseries    TS_A in {project}\n\n",
        "slurm-pick-ablation-3002_1.log": "series    TS_A in /elsewhere/project\n",
        "slurm-null-4001.log": f"project     {project}   config c\nseries      TS_C\n",
        # The header ends at its blank line; what comes after is not its series.
        "slurm-avg-4002.log": f"project     {project}   variant bin8\n\nseries 039: 12\n",
    }
    for name, text in logs.items():
        (tmp_path / name).write_text(text)
    queued = [
        ("2001", "project-cpu"), ("2002", "project-cpu"), ("3001", "ctf-window"),
        ("3002_1", "pick-ablation"), ("4001", "null"), ("4002", "avg"),
        ("5001", "project-gpu"), ("5002", "other-gpu"),
    ]
    jobs = [{"id": id, "name": name, "state": "RUNNING"} for id, name in queued]

    found = submit_lanes.busy({}, plan, jobs, root=tmp_path)
    assert [problem.split(" (")[0] for problem in found] == [
        "job 2001", "job 3001", "job 4002", "job 5001",
    ]
    assert "works on TS_B," in found[0] and "works on TS_A," in found[1]
    assert all("may be any of them" in problem for problem in found[2:])


def test_a_squeue_that_fails_is_a_problem_and_a_missing_one_a_note(
    lane, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("SQUEUE_EXIT", "1")
    code, _, err = _run(capsys, lane, "--dry-run")
    assert code == 0
    assert "would refuse" in err and "Unable to contact slurm controller" in err
    assert _run(capsys, lane)[0] == 1
    assert not _overwrite_per_call(lane)

    # No Slurm here, as on a laptop: a dry run says what it could not check.
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    code, _, err = _run(capsys, lane, "--dry-run")
    assert code == 0
    assert "no squeue here" in err and "would refuse" not in err


def test_a_ctffind_4_that_estimate_ctf_would_find_is_refused(lane, capsys, monkeypatch):
    # MDC's ctffind on PATH is CTFFIND 4.1.14 (ribo-mdc job 3781480, 2026-10-28).
    monkeypatch.setenv("FAKE_CTFFIND_VERSION", "4.1.14")
    found = lane.parent / "bin" / "ctffind5"
    code, _, err = _run(capsys, lane, "--dry-run")
    assert code == 0
    assert f"would refuse: {lane}: estimate-ctf would run {found}, which is CTFFIND 4.1.14" in err
    code, _, err = _run(capsys, lane)
    assert code == 1 and "which is CTFFIND 4.1.14" in err
    assert not _overwrite_per_call(lane)


def test_a_lane_that_names_its_ctffind_or_fits_no_ctf_is_not_asked(lane, capsys, monkeypatch):
    monkeypatch.setenv("FAKE_CTFFIND_VERSION", "4.1.14")
    # The GPU stage fits no CTF.
    assert "CTFFIND" not in _run(capsys, lane, "--from-stage", "gpu", "--dry-run")[2]
    # A CTFFIND the lane names is taken at its word, as estimate-ctf takes it.
    text = lane.read_text()
    assert "[executables]\n" in text
    lane.write_text(text.replace("[executables]\n", '[executables]\nctffind = "$CTFFIND5"\n', 1))
    monkeypatch.setenv("CTFFIND5", "/opt/ctffind5")
    assert "CTFFIND" not in _run(capsys, lane, "--dry-run")[2]


def test_a_ctffind_that_is_missing_or_silent_is_noted_not_refused(
    lane, capsys, monkeypatch, tmp_path
):
    (tmp_path / "bin" / "ctffind5").unlink()
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "bin"), "/usr/bin", "/bin"]))
    code, _, err = _run(capsys, lane, "--dry-run")
    assert code == 0 and "refuse" not in err
    assert "neither ctffind5 nor ctffind is on PATH here" in err

    # One that prints nothing and waits is stopped after the probe's time.
    silent = tmp_path / "bin" / "ctffind"
    silent.write_text("#!/bin/bash\nexec sleep 30\n")
    silent.chmod(0o755)
    monkeypatch.setattr(submit_lanes, "CTFFIND_PROBE_SECONDS", 0.3)
    started = time.monotonic()
    code, _, err = _run(capsys, lane, "--dry-run")
    assert time.monotonic() - started < 10
    assert code == 0 and "refuse" not in err
    assert f"not checked which CTFFIND {silent} is: it printed no version within 0.3 s" in err


def test_a_ctffind_that_keeps_asking_is_read_and_stopped(tmp_path):
    asking = tmp_path / "ctffind"
    asking.write_text(
        '#!/bin/bash\necho "            Version : 4.1.14"\n'
        'while true; do echo "Input image file name [input.mrc] :"; done\n'
    )
    asking.chmod(0o755)
    started = time.monotonic()
    assert submit_lanes.ctffind_version(str(asking)) == "4.1.14"
    assert time.monotonic() - started < 5


def test_the_banner_is_read_as_the_ctf_backend_reads_it():
    from cryoet_pipeline.backends.ctf import ctffind_version

    for banner in (
        "**   Welcome to Ctffind   **\n    Library Version : 2.0.0-alpha\n"
        "            Version : 5.0.2\n",
        "**   Welcome to Ctffind   **\n            Version : 4.1.14\n",
        "no banner here",
    ):
        mine = submit_lanes._CTFFIND_VERSION.search(banner)
        assert (mine["version"] if mine else None) == ctffind_version(banner)


def test_ctffind5_is_taken_before_ctffind_as_the_backend_takes_it(lane, capsys, tmp_path):
    older = tmp_path / "bin" / "ctffind"
    older.write_text('#!/bin/bash\necho "            Version : 4.1.14"\n')
    older.chmod(0o755)
    assert "CTFFIND" not in _run(capsys, lane, "--dry-run")[2]
