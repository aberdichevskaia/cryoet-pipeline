"""scripts/run_cases.py builds each case's sandbox so that the assistant's own
tools show the evidence the case describes, and lays out what a run did."""

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
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


def test_git_keeps_every_file_of_the_cases_evidence():
    # The repository's .gitignore leaves out outputs/, data/ and *.log* everywhere,
    # and until 30 October 2026 the cases' evidence with them.
    files = [str(path.relative_to(ROOT)) for path in run_cases.CASES.glob("*/tree/**/*")
             if path.is_file() and path.name != ".DS_Store"]
    ignored = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(files),
                             capture_output=True, text=True)
    if ignored.returncode == 128:
        pytest.skip("not a git checkout")
    assert files and ignored.stdout == ""


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


def test_squeue_fills_the_fields_its_format_asks_for(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E12")
    shown = _run(run_dir, env, "squeue", "--me",
                 "--format=%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R").splitlines()
    assert shown[0].split() == ["JOBID", "PARTITION", "NAME", "USER", "ST", "TIME", "NODES",
                                "NODELIST(REASON)"]
    assert shown[1].split() == ["3790001", "?", "ribo-mdc-gpu", env["USER"][:8], "R", "40:13",
                                "?", "max041"]
    assert _run(run_dir, env, "squeue", "-h", "-j", "123") == ""


def test_e12_the_running_job_started_within_its_time_limit_before_the_run(tmp_path):
    run_dir, _ = _sandbox(tmp_path, "E12")
    sandbox = run_dir / "sandbox"
    log = (sandbox / "slurm-ribo-mdc-gpu-3790001.log").read_text()
    started = datetime.fromisoformat(re.search(r"^started\s+(\S+)", log, re.M)[1])
    assert abs(datetime.now(UTC) - started - timedelta(minutes=40, seconds=13)) < timedelta(
        minutes=1
    )
    rows = [json.loads(line) for line in
            (sandbox / "outputs" / "ribo-mdc" / "step_timings.jsonl").read_text().splitlines()]
    finished = [datetime.fromisoformat(row["finished_at"]) for row in rows]
    # The job's prepare-template after its start, the lane's run of 16 October as it was.
    assert started < finished[-1] < datetime.now(UTC)
    assert finished[0] == datetime(2026, 10, 16, 10, tzinfo=UTC)


def test_e14_the_frames_are_the_tiffs_the_mdocs_name(tmp_path):
    run_dir, _ = _sandbox(tmp_path, "E14")
    folder = run_dir / "sandbox" / "data" / "hela"
    named = sorted(name for mdoc in folder.glob("*.mdoc") for name in re.findall(
        r"^SubFramePath = .*\\(\S+)$", mdoc.read_text(), re.M))
    assert len(named) == 82 and sorted(path.name for path in folder.glob("*.tif")) == named
    # Dated as acquired: the first frame of hela_ts01 at 10:00:13 on 3 September 2026.
    first = folder / "hela_ts01_001_0.0.tif"
    assert datetime.fromtimestamp(first.stat().st_mtime) == datetime(2026, 9, 3, 10, 0, 13)
    image = pytest.importorskip("PIL.Image").open(folder / named[0])
    assert (image.n_frames, image.size, image.mode) == (8, (5760, 4092), "L")
    image.seek(7)
    assert image.tobytes() == bytes(5760 * 4092)


def test_a_run_gets_nothing_of_the_session_it_is_started_from(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("AI_AGENT", "claude-code_x_agent")
    monkeypatch.setenv("OVERWRITE", "1")
    env = run_cases.run_env(BUILT["E11"], tmp_path)
    assert not {"CLAUDECODE", "AI_AGENT", "OVERWRITE"} & set(env)
    assert env["PATH"].startswith(str(tmp_path / "bin"))
    # {bin} names the run's stand-ins.
    assert env["CTFFIND5"] == str(tmp_path / "bin" / "ctffind5")


def test_the_sandbox_is_a_repository_of_its_own_with_the_evidence_as_found(tmp_path):
    # E12 lays its own ribo-mdc.toml over the tracked one; the evidence under
    # outputs/ stays out of git, as on the cluster.
    run_dir, env = _sandbox(tmp_path, "E12")
    sandbox = run_dir / "sandbox"
    assert _run(run_dir, env, "git", "rev-parse", "--show-toplevel").strip() == str(
        sandbox.resolve()
    )
    assert _run(run_dir, env, "git", "status", "--porcelain") == ""
    assert _run(run_dir, env, "git", "ls-files", "configs/lanes/ribo-mdc.toml").strip()
    assert not _run(run_dir, env, "git", "ls-files", "outputs").strip()


def test_the_sandbox_holds_no_grading_line_of_any_case(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E11")
    sandbox = run_dir / "sandbox"
    assert (sandbox / "docs" / "agent" / "playbook.md").is_file()
    assert not (sandbox / "docs" / "agent" / "eval_cases.md").exists()
    assert not (sandbox / "docs" / "agent" / "cases").exists()
    assert not (sandbox / "scripts" / "run_cases.py").exists()
    # A collected case's setup.sh runs the collector in the copy.
    assert (sandbox / "scripts" / "collect_case.py").is_file()
    # What the run's grep found on 30 October 2026, and every other case's lines
    # long enough to be its own (a line such as "submit" is in many files). The
    # playbook may hold one inside the case's own entries, which teach what the
    # case grades and which a held-out run removes (E04's P-1 says what not to do).
    playbook = (sandbox / "docs" / "agent" / "playbook.md").read_text()
    lines = [(case, line) for case in BUILT.values() for line in case.pass_if + case.must_not
             if len(line) >= 30]
    assert len(lines) > 20
    for case, line in lines:
        found = _run(run_dir, env, "grep", "-rlF", "--exclude-dir=.git", "--exclude=playbook.md",
                     line, ".")
        assert not found.strip(), (case.id, line, found)
        assert line not in run_cases.without_entries(playbook, case.playbook), (case.id, line)


def test_a_held_out_run_has_no_playbook_entry_of_its_case(tmp_path):
    run_dir, _ = _sandbox(tmp_path, "E16", held_out=True)
    playbook = (run_dir / "sandbox" / "docs" / "agent" / "playbook.md").read_text()
    assert "### A-3." not in playbook and "### A-2." in playbook and "### C-1." in playbook


def test_a_dry_run_builds_the_sandbox_and_runs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_cases, "RUNS", tmp_path / "runs")
    # The client would read this one too, from above the run's folder.
    (tmp_path / "CLAUDE.md").write_text("Develop like this.\n")
    assert run_cases.main(["E11", "--runs", "1", "--dry-run"]) == 0
    [run_dir] = (tmp_path / "runs").glob("*/E11-1")
    record = json.loads((run_dir / "record.json").read_text())
    assert record["dry_run"] and record["command"][1:3] == ["-p", BUILT["E11"].request]
    assert "--permission-mode" in record["command"]
    assert not (run_dir / "transcript.jsonl").exists()
    assert str(tmp_path / "CLAUDE.md") in record["claude_md_above"]
    assert "CLAUDE.md above the run's folder" in (run_dir.parent / "grading.md").read_text()


def test_a_transcript_is_read_into_its_answer_and_its_actions(tmp_path):
    events = [
        {"type": "system", "subtype": "init", "model": "claude-opus-5-5",
         "claude_code_version": "2.1.312", "permissionMode": "acceptEdits"},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "python3 scripts/show_failure.py x.toml"}},
            {"type": "tool_use", "id": "t2", "name": "Bash",
             "input": {"command": "python3 scripts/submit_lanes.py x.toml --reason y"}}]}},
        # As on 30 October 2026, when this event's string message stopped a batch.
        {"type": "system", "subtype": "permission_denied", "tool_name": "Bash",
         "tool_use_id": "t2", "message": "A variable in this command can't be checked"},
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


def test_a_call_is_shown_from_the_runs_folder_with_what_came_back(tmp_path):
    sandbox = tmp_path / "sandbox"
    read = {"tool": "Read", "input": {"file_path": f"{sandbox}/outputs/x.toml"},
            "output": "1\t# a\n2\tb = 1\n"}
    assert run_cases.describe_call(read, sandbox) == "Read: `outputs/x.toml` → 2 lines"
    refused = "x.toml: the config has keys nothing reads:\n  [ctf] windo: did you mean window?"
    shown = run_cases.describe_call(
        {"tool": "Bash", "input": {"command": "python scripts/submit_lanes.py --dry-run x.toml"},
         "output": refused + "\n" + "\n".join(f"line {n}" for n in range(30))}, sandbox
    )
    assert shown.startswith("Bash: `python scripts/submit_lanes.py --dry-run x.toml`\n")
    assert "   [ctf] windo: did you mean window?" in shown and "… 17 lines more" in shown



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
    # The client says it used the memory of a project it had before the run.
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "-dev-checkout" / "memory").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    init = {**EVENTS[0], "memory_paths": {
        "auto": f"{home}/.claude/projects/-dev-checkout/memory/"
    }, "plugins": [{"name": "cryoet", "source": "cryoet@inline", "path": "/x/assistant/plugin"},
                   {"name": "cc-plugin-x", "source": "cc-plugin-x@builtin", "path": "builtin"}]}
    events = tmp_path / "events.jsonl"
    events.write_text("".join(json.dumps(event) + "\n" for event in [init, *EVENTS[1:]]))
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
    assert record["plugins"] == ["cryoet@inline"] and record["memory_from_before"]
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
    assert "it read a memory kept before the run" in sheet


def test_the_stand_in_ctffind5_is_a_ctffind_5_the_dry_run_accepts(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E18")
    banner = _run(run_dir, env, env["CTFFIND5"])
    assert "Version : 5.0.2" in banner
    shown = _run(run_dir, env, "python3", "scripts/submit_recipe.py", "ctf-window",
                 "configs/lanes/ryr1-bin8.toml", "--series", "039", "--dry-run")
    assert "CTFFIND" not in shown.replace("CTFFIND5's sample tilt", "")
