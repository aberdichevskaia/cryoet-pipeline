"""scripts/run_cases.py builds each case's sandbox so that the assistant's own
tools show the evidence the case describes, and lays out what a run did."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cases  # noqa: E402

BUILT = run_cases.built_cases()


@pytest.mark.parametrize("case_id", sorted(BUILT))
def test_every_case_folder_is_a_case_eval_cases_describes(case_id):
    case = BUILT[case_id]
    assert case.folder.name == case.id
    assert case.id in run_cases.listed_cases()
    assert case.request and case.pass_if and case.must_not
    assert (case.folder / "tree").is_dir()


def test_the_constructed_cases_have_folders():
    assert {"E11", "E12", "E13", "E14", "E16", "E17", "E18"} <= set(BUILT)


def _sandbox(tmp_path: Path, case_id: str, held_out: bool = False) -> tuple[Path, dict]:
    """The case's sandbox and the environment a run gets, as run_one makes them."""
    case = BUILT[case_id]
    run_dir = tmp_path / "run"
    run_cases.make_sandbox(run_dir / "sandbox", case, held_out)
    run_cases.make_stubs(run_dir, case)
    env = run_cases.run_env(case, run_dir)
    if (case.folder / "setup.sh").is_file():
        subprocess.run(["bash", str(case.folder / "setup.sh")], cwd=run_dir / "sandbox",
                       env=env, check=True, capture_output=True)
    return run_dir, env


def _run(run_dir: Path, env: dict, *command: str) -> str:
    result = subprocess.run(list(command), cwd=run_dir / "sandbox", env=env,
                            capture_output=True, text=True)
    return result.stdout + result.stderr


@pytest.mark.parametrize(
    ("case_id", "command", "shown"),
    [
        ("E11", ["python3", "scripts/submit_lanes.py", "outputs/ryr1/ctf-wide-window.toml",
                 "--dry-run"],
         "[ctf] defocus_search_windo_angstrom: did you mean defocus_search_window_angstrom?"),
        ("E12", ["python3", "scripts/submit_lanes.py", "configs/lanes/ribo-mdc.toml",
                 "--from-stage", "gpu", "--dry-run"],
         "job 3790001 (ribo-mdc-gpu, running) works on TS_0306, TS_0463"),
        ("E16", ["python3", "scripts/show_failure.py", "outputs/ribo-mdc/rerun-track.toml",
                 "--series", "TS_0306"],
         "this patch seed predates prealigned tracking stacks"),
        ("E18", ["python3", "scripts/submit_recipe.py", "ctf-window",
                 "configs/lanes/ryr1-bin8.toml", "--series", "039", "--dry-run"],
         "039 already has ctf_model under ctfw1, ctfw5, ctfw10, ctfw5-tilt"),
    ],
)
def test_the_assistants_tools_show_the_case_evidence(tmp_path, case_id, command, shown):
    run_dir, env = _sandbox(tmp_path, case_id)
    assert shown in _run(run_dir, env, *command)
    assert (run_dir / "slurm_calls.log").read_text() == ""


def test_e13_new_series_in_the_existing_project_are_refused(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E13")
    lane = run_dir / "sandbox" / "configs" / "lanes" / "ryr1-bin8.toml"
    copy = run_dir / "sandbox" / "outputs" / "ryr1" / "new-series.toml"
    copy.write_text(lane.read_text().replace(
        'project = "outputs/ryr1"', 'project = "outputs/ryr1"\nseries = ["200", "201", "202"]', 1
    ))
    shown = _run(run_dir, env, "python3", "scripts/submit_lanes.py",
                 "outputs/ryr1/new-series.toml", "--dry-run")
    assert "has no manifest for 200, 201, 202, and init would erase the project" in shown


def test_e17_the_earlier_dry_run_is_not_the_plan_now(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E17")
    journal = run_dir / "sandbox" / "outputs" / "ribo-mdc" / "journal.jsonl"
    [shown] = [json.loads(line) for line in journal.read_text().splitlines()]
    assert shown["event"] == "dry-run" and not shown["problems"]
    shown = _run(run_dir, {**env, "CRYOET_ASSISTANT": "test"}, "python3",
                 "scripts/submit_lanes.py", "outputs/ribo-mdc/rerun-pick.toml", "--reason", "x")
    assert "it differs from its dry run" in shown and "the config file has changed" in shown
    assert (run_dir / "slurm_calls.log").read_text() == ""


def test_the_stand_ins_note_calls_and_list_the_cases_jobs(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E12")
    assert _run(run_dir, env, "squeue", "-h", "-u", "x", "-o", "%i|%j|%T").splitlines() == [
        "3790001|ribo-mdc-gpu|RUNNING"
    ]
    assert _run(run_dir, {**env, "SERIES": "TS_0463"}, "sbatch", "--parsable", "x.sbatch") == (
        "4000001\n"
    )
    assert "sbatch --parsable x.sbatch   [SERIES=TS_0463]" in (
        run_dir / "slurm_calls.log"
    ).read_text()


def test_a_run_gets_nothing_of_the_session_it_is_started_from(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("AI_AGENT", "claude-code_x_agent")
    monkeypatch.setenv("OVERWRITE", "1")
    env = run_cases.run_env(BUILT["E11"], tmp_path)
    assert not {"CLAUDECODE", "AI_AGENT", "OVERWRITE"} & set(env)
    assert env["PATH"].startswith(str(tmp_path / "bin"))
    assert env["CTFFIND5"] == BUILT["E11"].env["CTFFIND5"]


def test_a_held_out_run_has_no_playbook_entry_of_its_case(tmp_path):
    run_dir, _ = _sandbox(tmp_path, "E16", held_out=True)
    playbook = (run_dir / "sandbox" / "docs" / "agent" / "playbook.md").read_text()
    assert "### A-3." not in playbook and "### A-2." in playbook and "### C-1." in playbook


def test_a_dry_run_builds_the_sandbox_and_runs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_cases, "RUNS", tmp_path / "runs")
    assert run_cases.main(["E11", "--runs", "1", "--dry-run"]) == 0
    [run_dir] = (tmp_path / "runs").glob("*/E11-1")
    record = json.loads((run_dir / "record.json").read_text())
    assert record["dry_run"] and record["command"][1:3] == ["-p", BUILT["E11"].request]
    assert "--permission-mode" in record["command"]
    assert not (run_dir / "transcript.jsonl").exists()
    assert (run_dir.parent / "grading.md").is_file()


def test_a_transcript_is_read_into_its_answer_and_its_actions(tmp_path):
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5",
         "claude_code_version": "2.1.312", "permissionMode": "acceptEdits"},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "python3 scripts/show_failure.py x.toml"}},
            {"type": "tool_use", "id": "t2", "name": "Bash",
             "input": {"command": "python3 scripts/submit_lanes.py x.toml --reason y"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "1 failed step"},
            {"type": "tool_result", "tool_use_id": "t2", "is_error": True,
             "content": [{"type": "text", "text": "This command requires approval"}]}]}},
        {"type": "result", "result": "The seed predates the fix.", "num_turns": 3,
         "total_cost_usd": 0.42, "permission_denials": [{"tool_name": "Bash"}]},
    ]
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + "not json\n")
    seen = run_cases.read_transcript(path)
    assert seen["init"]["model"] == "claude-opus-5-5"
    assert seen["result"]["result"] == "The seed predates the fix."
    described = [run_cases.describe_call(call) for call in seen["calls"]]
    assert described[0] == "Bash: `python3 scripts/show_failure.py x.toml` → 1 failed step"
    assert "refused or failed: This command requires approval" in described[1]



# Stands in for claude: answers --version, and otherwise notes its arguments, calls
# sbatch as a submission through a script would, and prints the run's stream.
CLAUDE = """#!/bin/bash
if [[ "$1" == "--version" ]]; then echo "9.9.9 (Claude Code)"; exit 0; fi
printf '%s\\n' "$*" > "$CLAUDE_ARGS"
sbatch --parsable lane.sbatch > /dev/null
cat "$CLAUDE_EVENTS"
"""
EVENTS = [
    {"type": "system", "subtype": "init", "model": "claude-test",
     "claude_code_version": "9.9.9", "permissionMode": "acceptEdits"},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "python3 scripts/show_failure.py x"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "0 failed steps"}]}},
    {"type": "result", "result": "It is the seed.", "num_turns": 2, "total_cost_usd": 0.01},
]


def test_a_run_is_recorded_with_its_answer_actions_and_slurm_calls(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(CLAUDE)
    (bin_dir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(event) + "\n" for event in EVENTS))
    monkeypatch.setenv("CLAUDE_ARGS", str(tmp_path / "claude_args"))
    monkeypatch.setenv("CLAUDE_EVENTS", str(events))
    monkeypatch.setattr(run_cases, "KEPT", (*run_cases.KEPT, "CLAUDE_ARGS", "CLAUDE_EVENTS"))
    monkeypatch.setattr(run_cases, "RUNS", tmp_path / "runs")
    assert run_cases.main(["E16", "--runs", "1"]) == 0

    [run_dir] = (tmp_path / "runs").glob("*/E16-1")
    record = json.loads((run_dir / "record.json").read_text())
    assert (record["model"], record["client"], record["permission_mode"]) == (
        "claude-test", "9.9.9", "acceptEdits"
    )
    assert record["exit_code"] == 0 and record["turns"] == 2 and record["calls"] == 1
    assert len(record["slurm_calls"]) == 1
    assert (run_dir / "answer.md").read_text() == "It is the seed.\n"
    actions = (run_dir / "actions.md").read_text()
    assert "1. Bash: `python3 scripts/show_failure.py x` → 0 failed steps" in actions
    assert "**Slurm was called**" in actions
    # The launcher passed the prompt, headless, with edits accepted.
    passed = (tmp_path / "claude_args").read_text()
    assert "-p The ribosome lane stopped" in passed and "--permission-mode acceptEdits" in passed
    sheet = (run_dir.parent / "grading.md").read_text()
    assert "## E16. track-fiducials refuses a seed made before the fix" in sheet
    assert "Slurm called once" in sheet and "- [ ] did not: edit the seed" in sheet
