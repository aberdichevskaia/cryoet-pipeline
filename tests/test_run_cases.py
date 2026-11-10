"""scripts/run_cases.py builds each case's sandbox so that the assistant's own
tools show the evidence the case describes, and lays out what a run did."""

import dataclasses
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import grade_cases  # noqa: E402
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


def _sandbox(tmp_path: Path, case_id: str | run_cases.Case,
             held_out: bool = False) -> tuple[Path, dict]:
    """The case's sandbox and the environment a run gets, as run_one makes them."""
    case = BUILT[case_id] if isinstance(case_id, str) else case_id
    run_dir = tmp_path / "run"
    return run_dir, run_cases.prepare_run(run_dir, case, held_out)


def _collected_case(tmp_path: Path) -> run_cases.Case:
    """A case laid out as collect_case.py writes one: a lane file whose placeholder
    its setup.sh fills in, and sacct's record of its job in manifest.json."""
    folder = tmp_path / "cases" / "E99"
    (folder / "tree" / "configs" / "lanes").mkdir(parents=True)
    (folder / "tree" / "configs" / "lanes" / "hiv.toml").write_text('frames = "{{ROOT}}/x"\n')
    (folder / "setup.sh").write_text(
        "sed -i.bak \"s|{{ROOT}}|$PWD|\" configs/lanes/hiv.toml && rm configs/lanes/hiv.toml.bak\n")
    (folder / "case.toml").write_text(
        'id = "E99"\ntitle = "t"\nrequest = "r"\npass_if = ["p"]\nmust_not = ["m"]\n'
        '[sacct.123]\nJobName = "queued-by-hand"\nState = "PENDING"\n')
    (folder / "manifest.json").write_text(json.dumps({"slurm": {"41075712": {
        "JobName": "hiv-gpu-TS_01", "State": "FAILED", "ExitCode": "1:0",
        "Start": "2026-10-07T22:58:21", "Elapsed": "00:12:35", "NodeList": "gpusrv50",
        "AllocTRES": "billing=8,cpu=8,gres/gpu:a100_3g.20gb=1,gres/gpu=1,mem=64G,node=1"}}}))
    return run_cases.load_case(folder)


def test_the_sandbox_is_committed_after_its_setup_fills_in_the_placeholders(tmp_path):
    run_dir, env = _sandbox(tmp_path, _collected_case(tmp_path))
    lane = run_dir / "sandbox" / "configs" / "lanes" / "hiv.toml"
    assert lane.read_text() == f'frames = "{run_dir / "sandbox"}/x"\n'
    # What the client's git status shows the run: nothing changed.
    assert _run(run_dir, env, "git", "status", "--porcelain") == ""


def test_sacct_answers_for_the_jobs_the_case_records(tmp_path):
    run_dir, env = _sandbox(tmp_path, _collected_case(tmp_path))
    sandbox = str(run_dir / "sandbox")
    # As show_project asks for a job's state, and the person for its GPU slice.
    assert _run(run_dir, env, "sacct", "-n", "-P", "-X", "-j", "41075712",
                "-o", "JobID,State,ExitCode,Elapsed") == "41075712|FAILED|1:0|00:12:35\n"
    assert "gres/gpu:a100_3g.20gb=1" in _run(run_dir, env, "sacct", "-j", "41075712", "-P",
                                             "--format=JobName,AllocTRES")
    # As show_project looks for a project's logs: the job ran from the clone's root.
    assert _run(run_dir, env, "sacct", "-n", "-P", "-X", "-u", "x", "-S", "2026-10-01T00:00:00",
                "-o", "JobID,JobName,WorkDir").splitlines() == [
        f"41075712|hiv-gpu-TS_01|{sandbox}", f"123|queued-by-hand|{sandbox}"]
    # Plain sacct shows today's jobs only; one with no start yet among them.
    assert _run(run_dir, env, "sacct", "-n", "-P", "-o", "JobID") == "123\n"
    shown = _run(run_dir, env, "sacct", "-j", "41075712", "-o", "JobID,JobName%8,State")
    assert shown.splitlines()[0].split() == ["JobID", "JobName", "State"]
    assert shown.splitlines()[2].split() == ["41075712", "hiv-gpu+", "FAILED"]


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
    assert (sandbox / "docs" / "agent" / "playbook" / "ctf.md").is_file()
    assert not (sandbox / "docs" / "agent" / "eval_cases.md").exists()
    assert not (sandbox / "docs" / "agent" / "cases").exists()
    assert not (sandbox / "scripts" / "run_cases.py").exists()
    # The architecture doc's starting case set states the answers; the tests'
    # fixtures hold past incidents.
    assert not (sandbox / "docs" / "agent_architecture.md").exists()
    assert not (sandbox / "tests").exists()
    # The tutorials work through RyR1, 161's tracking binning and its CTF window
    # among it.
    assert not (sandbox / "docs" / "tutorials").exists()
    # A collected case's setup.sh runs the collector in the copy.
    assert (sandbox / "scripts" / "collect_case.py").is_file()
    # What the run's grep found on 30 October 2026, and every other case's lines
    # long enough to be its own (a line such as "submit" is in many files). The
    # playbook may hold one inside the case's own entries, which teach what the
    # case grades and which a held-out run removes (E04's P-1 says what not to do).
    playbook = [path.read_text()
                for path in sorted((sandbox / "docs" / "agent" / "playbook").glob("*.md"))]
    lines = [(case, line) for case in BUILT.values() for line in case.pass_if + case.must_not
             if len(line) >= 30]
    assert len(lines) > 20
    for case, line in lines:
        found = _run(run_dir, env, "grep", "-rlF", "--exclude-dir=.git", "--exclude-dir=playbook",
                     line, ".")
        assert not found.strip(), (case.id, line, found)
        for text in playbook:
            assert line not in run_cases.without_entries(text, case.playbook), (case.id, line)


def _answer_markers(sandbox: Path, case: run_cases.Case) -> list[tuple[str, str]]:
    """Each file of the copy, other than the case's evidence, that holds one of the
    case's answer markers, with the marker."""
    evidence = {str(path.relative_to(case.folder / "tree"))
                for path in (case.folder / "tree").rglob("*") if path.is_file()}
    manifest = case.folder / "manifest.json"
    if manifest.is_file():
        evidence |= {entry["path"] for entry in json.loads(manifest.read_text())["files"]}
    found = []
    for path in sandbox.rglob("*"):
        name = str(path.relative_to(sandbox))
        if ".git" in path.parts or not path.is_file() or name in evidence:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        found += [(name, marker) for marker in case.markers if re.search(marker, text)]
    return found


@pytest.mark.parametrize("case_id", sorted(case_id for case_id, case in BUILT.items()
                                           if case.markers))
def test_a_held_out_copy_holds_none_of_its_cases_answer(tmp_path, case_id):
    case = BUILT[case_id]
    (tmp_path / "plain").mkdir()
    plain, _ = _sandbox(tmp_path / "plain", case_id)
    # The markers find the answer where the copy is not held out ...
    assert _answer_markers(plain / "sandbox", case)
    held, _ = _sandbox(tmp_path, case_id, held_out=True)
    # ... and nowhere outside the evidence once it is.
    assert _answer_markers(held / "sandbox", case) == []


def test_only_held_out_runs_just_the_held_out_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(run_cases, "RUNS", tmp_path / "runs")
    # E14 has no playbook entry of its own, so it has no held-out copy.
    assert run_cases.main(["E02", "E14", "--runs", "2", "--only-held-out", "--dry-run"]) == 0
    [out] = (tmp_path / "runs").glob("*")
    assert sorted(path.name for path in out.glob("E*")) == ["E02-held-out-1", "E02-held-out-2"]


def test_a_held_out_edit_whose_text_is_gone_stops_the_run(tmp_path):
    case = dataclasses.replace(BUILT["E01"], held_out_replace=[
        {"file": "docs/agent/rerun_recipes.md", "find": "a line no file holds", "with": ""}])
    with pytest.raises(SystemExit, match="E01 held out: docs/agent/rerun_recipes.md holds 0"):
        _sandbox(tmp_path, case, held_out=True)


def test_a_held_out_run_has_no_playbook_entry_of_its_case(tmp_path):
    run_dir, _ = _sandbox(tmp_path, "E16", held_out=True)
    playbook = run_dir / "sandbox" / "docs" / "agent" / "playbook"
    alignment = (playbook / "alignment.md").read_text()
    assert "### A-3." not in alignment and "### A-2." in alignment
    assert "### C-1." in (playbook / "ctf.md").read_text()


def test_a_held_out_run_takes_its_case_out_of_every_playbook_file(tmp_path):
    run_dir, _ = _sandbox(tmp_path, "E05", held_out=True)
    entries = [entry for path in (run_dir / "sandbox" / "docs" / "agent" / "playbook").glob("*.md")
               for entry in run_cases.playbook_entries(path.read_text())]
    assert not set(BUILT["E05"].playbook) & set(entries)
    assert {"C-1", "A-3", "E-1"} <= set(entries)


def test_a_held_out_entry_no_playbook_file_holds_stops_the_run(tmp_path):
    case = dataclasses.replace(BUILT["E01"], playbook=["A-2", "Z-9"])
    message = "E01 held out: no playbook entry Z-9 in docs/agent/playbook/"
    with pytest.raises(SystemExit, match=message):
        _sandbox(tmp_path, case, held_out=True)


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
    # E11's follow-up would resume the session its request made.
    assert f"'{BUILT['E11'].follow_up[0]}' --resume <session>" in capsys.readouterr().out
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


# 10 November 2026: the blind sheet's E11 "shows the corrected key as a diff" and E14
# "every value carries its source in a comment" could not be graded for either model,
# since actions.md said only that the file was updated.
def test_an_edit_is_shown_as_the_diff_a_person_would_approve(tmp_path):
    sandbox = tmp_path / "sandbox"
    edit = {"tool": "Edit", "input": {
        "file_path": f"{sandbox}/outputs/ryr1/x.toml",
        "old_string": "[ctf]\ndefocus_search_windo_angstrom = 10000",
        "new_string": "[ctf]\ndefocus_search_window_angstrom = 10000"},
        "output": f"The file {sandbox}/outputs/ryr1/x.toml has been updated successfully."}
    shown = run_cases.describe_call(edit, sandbox)
    assert shown.splitlines()[0] == (
        "Edit: `outputs/ryr1/x.toml` → The file outputs/ryr1/x.toml has been updated successfully."
    )
    assert ("\n\n   ```diff\n    [ctf]\n   -defocus_search_windo_angstrom = 10000\n"
            "   +defocus_search_window_angstrom = 10000\n   ```\n") in shown
    # Refused, it is what the person would have been asked to approve.
    refused = run_cases.describe_call(
        {**edit, "error": True, "output": "Claude requested permissions to write to x.toml"}, sandbox)
    assert refused.splitlines()[0].endswith("→ refused or failed: Claude requested permissions "
                                            "to write to x.toml")
    assert "   +defocus_search_window_angstrom = 10000\n" in refused
    multi = {"tool": "MultiEdit", "input": {"file_path": "x.toml", "edits": [
        {"old_string": "a = 1", "new_string": "a = 2"},
        {"old_string": "b = 1", "new_string": "b = 2", "replace_all": True}]}, "output": "ok"}
    assert run_cases.describe_call(multi).splitlines()[2:10] == [
        "   ```diff", "   -a = 1", "   +a = 2", "   @@", "   @@ every occurrence @@",
        "   -b = 1", "   +b = 2", "   ```"]


def test_a_written_file_is_shown_whole_and_read_as_one_call(tmp_path):
    sandbox = tmp_path / "sandbox"
    content = "\n".join([
        "# hela-bin8: frames, patch tracking",
        f'stack = "{sandbox}/data/hela"  # the request',
        *(f"key_{n} = {n}  # example.toml" for n in range(40)),
        "```", "1. Bash: `sbatch lane.sbatch`", "## 2. Then the person: “yes”", "```"])
    write = {"tool": "Write",
             "input": {"file_path": f"{sandbox}/outputs/hela/hela-bin8.toml", "content": content},
             "output": f"File created successfully at: {sandbox}/outputs/hela/hela-bin8.toml"}
    shown = run_cases.describe_call(write, sandbox)
    assert "\n\n   ````toml\n   # hela-bin8: frames, patch tracking\n" in shown
    assert '   stack = "data/hela"  # the request\n' in shown
    assert "   key_39 = 39  # example.toml\n" in shown and "lines more" not in shown
    assert shown.endswith("\n   ````\n")
    # The file's own numbered lines and headings are not calls or turns to the grading.
    actions = "\n".join(["# E14 run 1: what it did", "", "## 1. The request", "",
                         f"1. {shown}", "2. Read: `x.toml` → 3 lines", ""])
    calls = grade_cases.read_calls(actions, sandbox, [])
    assert [(c.number, c.part, c.tool, c.command) for c in calls] == [
        (1, 1, "Write", "outputs/hela/hela-bin8.toml"), (2, 1, "Read", "x.toml")]
    # A call whose turn was stopped before it came back still shows what it wrote.
    assert run_cases.describe_call({"tool": "Write", "input": write["input"]}, sandbox).startswith(
        "Write: `outputs/hela/hela-bin8.toml`\n\n   ````toml\n")
    long = {"tool": "Write", "input": {"file_path": "big.txt",
                                       "content": "\n".join(f"line {n}" for n in range(1, 1006))}}
    shown = run_cases.describe_call(long)
    assert "   line 1000\n   … 5 lines more\n" in shown and "line 1001" not in shown



# Stands in for claude: answers --version, and otherwise notes its arguments, one
# line a call. A first turn calls sbatch as a submission through a script would,
# prints its stream and exits with CLAUDE_EXIT; a resumed one prints the second.
CLAUDE = """#!/bin/bash
if [[ "$1" == "--version" ]]; then echo "9.9.9 (Claude Code)"; exit 0; fi
printf '%s\\n' "${*//$'\\n'/ }" >> "$CLAUDE_ARGS"
if [[ -n "$ANTHROPIC_BASE_URL" ]]; then
  env | grep -E '^(ANTHROPIC|CLAUDE_CODE)_' | sort > "$CLAUDE_ARGS.env"
fi
if [[ -n "$CLAUDE_OFFLINE" ]]; then
  echo "Claude Code couldn't reach Anthropic to load your organization's settings." >&2; exit 1
fi
if [[ " $* " == *" --resume "* ]]; then cat "$CLAUDE_EVENTS_2"; exit 0; fi
sbatch --parsable lane.sbatch > /dev/null
cat "$CLAUDE_EVENTS"
exit "${CLAUDE_EXIT:-0}"
"""
EVENTS = [
    {"type": "system", "subtype": "init", "model": "claude-test", "session_id": "s-1",
     "claude_code_version": "9.9.9", "permissionMode": "acceptEdits"},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t1", "name": "Bash",
         "input": {"command": "python3 scripts/show_failure.py x"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "0 failed steps"}]}},
    {"type": "result", "result": "It is the seed.", "num_turns": 2, "total_cost_usd": 0.01,
     "session_id": "s-1", "uuid": "r-1"},
]
EVENTS_2 = [
    {"type": "system", "subtype": "init", "model": "claude-test", "session_id": "s-1",
     "claude_code_version": "9.9.9", "permissionMode": "acceptEdits"},
    {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "t2", "name": "Bash",
         "input": {"command": "python3 scripts/submit_lanes.py x.toml --dry-run"}}]}},
    {"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": "t2", "content": "no problems"}]}},
    {"type": "result", "result": "Here is the dry run. Shall I submit?", "num_turns": 1,
     "total_cost_usd": 0.02, "session_id": "s-1", "uuid": "r-2"},
]


def _stand_in(tmp_path: Path, monkeypatch, init: dict | None = None) -> None:
    """claude on PATH as the stand-in, with the runs under tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(CLAUDE)
    (bin_dir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    for name, events in (("CLAUDE_EVENTS", [init or EVENTS[0], *EVENTS[1:]]),
                         ("CLAUDE_EVENTS_2", EVENTS_2)):
        path = tmp_path / f"{name.lower()}.jsonl"
        path.write_text("".join(json.dumps(event) + "\n" for event in events))
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("CLAUDE_ARGS", str(tmp_path / "claude_args"))
    monkeypatch.setattr(run_cases, "KEPT", (*run_cases.KEPT, "CLAUDE_ARGS", "CLAUDE_EVENTS",
                                            "CLAUDE_EVENTS_2", "CLAUDE_EXIT", "CLAUDE_OFFLINE"))
    monkeypatch.setattr(run_cases, "RUNS", tmp_path / "runs")


def test_a_run_is_recorded_with_its_answer_actions_and_slurm_calls(tmp_path, monkeypatch):
    # The client says it used the memory of a project it had before the run.
    home = tmp_path / "home"
    (home / ".claude" / "projects" / "-dev-checkout" / "memory").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    _stand_in(tmp_path, monkeypatch, init={**EVENTS[0], "memory_paths": {
        "auto": f"{home}/.claude/projects/-dev-checkout/memory/"
    }, "plugins": [{"name": "cryoet", "source": "cryoet@inline", "path": "/x/assistant/plugin"},
                   {"name": "cc-plugin-x", "source": "cc-plugin-x@builtin", "path": "builtin"}]})
    [follow_up] = BUILT["E16"].follow_up
    assert run_cases.main(["E16", "--runs", "1"]) == 0

    [run_dir] = (tmp_path / "runs").glob("*/E16-1")
    record = json.loads((run_dir / "record.json").read_text())
    assert (record["model"], record["client"], record["permission_mode"]) == (
        "claude-test", "9.9.9", "acceptEdits"
    )
    # The request and the follow-up, in one session.
    assert record["exit_codes"] == [0, 0] and record["prompts_sent"] == 2
    assert record["turns"] == 3 and record["calls"] == 2 and record["cost_usd"] == 0.03
    assert len(record["slurm_calls"]) == 1
    assert record["plugins"] == ["cryoet@inline"] and record["memory_from_before"]
    assert record["host"] == platform.node()
    assert f", on {platform.node()}." in (run_dir.parent / "grading.md").read_text()
    answer = (run_dir / "answer.md").read_text()
    assert answer.index("## 1. The request") < answer.index("It is the seed.") < answer.index(
        f"## 2. Then the person: “{follow_up}”") < answer.index("Shall I submit?")
    actions = (run_dir / "actions.md").read_text()
    assert "1. Bash: `python3 scripts/show_failure.py x` → 0 failed steps" in actions
    assert actions.index("## 2. Then the person") < actions.index(
        "2. Bash: `python3 scripts/submit_lanes.py x.toml --dry-run` → no problems")
    assert "**Slurm was called**" in actions
    # The launcher passed the prompt, headless, with edits accepted; the follow-up
    # resumed the session the first turn made.
    first, second = (tmp_path / "claude_args").read_text().splitlines()
    assert "-p The ribosome lane stopped" in first and "--permission-mode acceptEdits" in first
    assert f"-p {follow_up} --resume s-1" in second and "--resume" not in first
    sheet = (run_dir.parent / "grading.md").read_text()
    assert "## E16. track-fiducials refuses a seed made before the fix" in sheet
    assert f"> *then:* {follow_up}" in sheet
    assert "Slurm called once" in sheet and "- [ ] did not: edit the seed" in sheet
    assert "it read a memory kept before the run" in sheet and "stopped after" not in sheet


def test_a_follow_up_is_sent_only_after_a_turn_that_ended_well(tmp_path, monkeypatch):
    _stand_in(tmp_path, monkeypatch)
    monkeypatch.setenv("CLAUDE_EXIT", "1")
    assert run_cases.main(["E11", "--runs", "1"]) == 0
    [run_dir] = (tmp_path / "runs").glob("*/E11-1")
    record = json.loads((run_dir / "record.json").read_text())
    assert record["exit_codes"] == [1] and record["prompts_sent"] == 1
    assert len((tmp_path / "claude_args").read_text().splitlines()) == 1
    assert "(not sent)" in (run_dir / "answer.md").read_text()
    assert "exit 1, stopped after message 1 of 2" in (run_dir.parent / "grading.md").read_text()


def test_a_batch_stops_once_two_runs_in_a_row_could_not_reach_anthropic(tmp_path, monkeypatch):
    _stand_in(tmp_path, monkeypatch)
    monkeypatch.setenv("CLAUDE_OFFLINE", "1")
    assert run_cases.main(["E13", "E17", "E18", "--runs", "1"]) == 1
    [out] = (tmp_path / "runs").glob("*")
    records = json.loads((out / "runs.json").read_text())
    assert [record["case"] for record in records] == ["E13", "E17"]
    assert all(record["unreachable"] for record in records)
    assert not (out / "E18-1").exists()
    assert "the client could not reach Anthropic; run it again" in (out / "grading.md").read_text()


def test_a_run_stopped_at_the_turn_limit_is_flagged(tmp_path):
    # E08 held out, run 2, on 4 November 2026: 41 turns and no answer.
    record = {"case": "E08", "run": 2, "held_out": True, "exit_code": 1, "prompts_sent": 1,
              "max_turns_reached": True}
    sheet = run_cases.grading_sheet([BUILT["E08"]], [record], tmp_path,
                                    {"commit": None, "modified": []})
    assert "exit 1, stopped at the turn limit, with no answer" in sheet


def test_a_run_against_another_server_asks_it_for_the_one_model(tmp_path, monkeypatch):
    _stand_in(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-3f9a")
    assert run_cases.main(["E18", "--runs", "1", "--model", "glm-test",
                           "--base-url", "http://127.0.0.1:20001",
                           "--context-tokens", "131072"]) == 0
    seen = dict(line.split("=", 1)
                for line in (tmp_path / "claude_args.env").read_text().splitlines())
    assert seen["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:20001"
    assert seen["ANTHROPIC_AUTH_TOKEN"] == "tok-3f9a"
    # Subagents and background work too, or the client would ask the server for
    # a Claude model it does not serve.
    assert {seen[f"ANTHROPIC_DEFAULT_{alias}_MODEL"]
            for alias in ("OPUS", "SONNET", "HAIKU", "FABLE")} == {"glm-test"}
    assert seen["CLAUDE_CODE_SUBAGENT_MODEL"] == "glm-test"
    assert seen["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "131072"
    assert seen["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert "--model glm-test" in (tmp_path / "claude_args").read_text()
    [run_dir] = (tmp_path / "runs").glob("*/E18-1")
    record = json.loads((run_dir / "record.json").read_text())
    assert (record["base_url"], record["context_tokens"]) == ("http://127.0.0.1:20001", 131072)
    # The record names the server, never its token.
    assert "tok-3f9a" not in (run_dir / "record.json").read_text()
    assert "served at http://127.0.0.1:20001 with a 131,072-token window" in (
        run_dir.parent / "grading.md").read_text()


def test_a_claude_run_gets_no_server_from_this_shell(tmp_path, monkeypatch):
    # The shell may point its own sessions somewhere else; a run of Claude does not follow.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:20001")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    monkeypatch.setenv("CLAUDE_CODE_MAX_CONTEXT_TOKENS", "131072")
    env = run_cases.run_env(BUILT["E11"], tmp_path)
    env.update(run_cases.backend_env(None, "claude-test", None))
    assert not [key for key in env if key.startswith(("ANTHROPIC_", "CLAUDE_CODE_"))]
    # A server that wants a key gets the shell's; a local one takes any.
    assert run_cases.backend_env("http://x", "m", 1)["ANTHROPIC_AUTH_TOKEN"] == "secret"
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN")
    assert run_cases.backend_env("http://x", "m", 1)["ANTHROPIC_AUTH_TOKEN"] == "local"


def test_a_server_needs_its_model_and_window():
    for argv in (["E11", "--base-url", "http://x"],
                 ["E11", "--base-url", "http://x", "--model", "glm-test"],
                 ["E11", "--context-tokens", "131072"]):
        with pytest.raises(SystemExit):
            run_cases.main(argv)


def test_a_run_that_nears_the_servers_window_is_flagged(tmp_path):
    # The context of a reply is what was new plus what was cached; Ollama reports
    # the two apart (UsageFromMetrics in its anthropic/anthropic.go).
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in [
        EVENTS[0],
        {"type": "assistant", "message": {"content": [], "usage": {
            "input_tokens": 2_000, "cache_read_input_tokens": 60_000}}},
        {"type": "assistant", "message": {"content": [], "usage": {
            "input_tokens": 1_000, "cache_read_input_tokens": 124_000,
            "cache_creation_input_tokens": 500}}},
        EVENTS[-1],
    ]))
    largest = run_cases.read_transcript(path)["largest_context"]
    assert largest == 125_500
    record = {"case": "E11", "run": 1, "held_out": False, "exit_code": 0, "prompts_sent": 2,
              "model": "glm-test", "base_url": "http://127.0.0.1:20001",
              "context_tokens": 131_072, "largest_context": largest}
    code = {"commit": None, "modified": []}
    sheet = run_cases.grading_sheet([BUILT["E11"]], [record], tmp_path, code)
    assert ("its context reached 125,500 of the server's 131,072 tokens, where the server "
            "may have dropped early messages") in sheet
    # Below 95% of the window, and for Claude's runs, which have no server window.
    for changed in ({"largest_context": 100_000}, {"base_url": None, "context_tokens": None}):
        assert "its context reached" not in run_cases.grading_sheet(
            [BUILT["E11"]], [{**record, **changed}], tmp_path, code)


def test_a_case_without_follow_ups_answers_in_one_piece(tmp_path, monkeypatch):
    _stand_in(tmp_path, monkeypatch)
    assert not BUILT["E18"].follow_up
    assert run_cases.main(["E18", "--runs", "1"]) == 0
    [run_dir] = (tmp_path / "runs").glob("*/E18-1")
    assert (run_dir / "answer.md").read_text() == "It is the seed.\n"
    assert "## " not in (run_dir / "actions.md").read_text().split("\n", 1)[1]
    assert json.loads((run_dir / "record.json").read_text())["exit_codes"] == [0]


def test_turns_are_told_apart_and_a_resumed_stream_is_not_counted_twice(tmp_path):
    # Should a resumed stream show the first turn's call and result again.
    path = tmp_path / "transcript.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in [
        *EVENTS, EVENTS_2[0], *EVENTS[1:], *EVENTS_2[1:]]))
    seen = run_cases.read_transcript(path)
    assert [call["turn"] for call in seen["calls"]] == [1, 2]
    assert [(result["_turn"], result["result"]) for result in seen["results"]] == [
        (1, "It is the seed."), (2, "Here is the dry run. Shall I submit?")]


def test_the_stand_in_ctffind5_is_a_ctffind_5_the_dry_run_accepts(tmp_path):
    run_dir, env = _sandbox(tmp_path, "E18")
    banner = _run(run_dir, env, env["CTFFIND5"])
    assert "Version : 5.0.2" in banner
    shown = _run(run_dir, env, "python3", "scripts/submit_recipe.py", "ctf-window",
                 "configs/lanes/ryr1-bin8.toml", "--series", "039", "--dry-run")
    assert "CTFFIND" not in shown.replace("CTFFIND5's sample tilt", "")
