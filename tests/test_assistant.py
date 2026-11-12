"""The terminal assistant's launcher, its client rules and its skills
(scripts/assistant and assistant/): what it starts Claude Code with, what the
guard on shell commands blocks, and that the skills name only what exists."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "assistant"
ASSISTANT = ROOT / "assistant"
SKILLS = sorted((ASSISTANT / "plugin" / "skills").glob("*/SKILL.md"))

# Stands in for claude: answers --version, and otherwise notes its arguments, the
# directory it runs in and the variables the launcher sets, one per line.
CLAUDE = """#!/bin/bash
if [[ "$1" == "--version" ]]; then echo "9.9.9 (Claude Code)"; exit 0; fi
{
    echo "cwd=$PWD"
    echo "CRYOET_ASSISTANT=${CRYOET_ASSISTANT:-}"
    echo "CRYOET_ASSISTANT_MODEL=${CRYOET_ASSISTANT_MODEL:-}"
    echo "ENABLE_CLAUDEAI_MCP_SERVERS=${ENABLE_CLAUDEAI_MCP_SERVERS:-}"
    echo "CLAUDE_CODE_AUTO_CONNECT_IDE=${CLAUDE_CODE_AUTO_CONNECT_IDE:-}"
    echo "CRYOET_LIBRARY_ROOT=${CRYOET_LIBRARY_ROOT:-}"
    echo "CRYOET_LIBRARY_SNAPSHOT=${CRYOET_LIBRARY_SNAPSHOT:-}"
    for arg in "$@"; do printf 'arg=%s\\0' "$arg"; done
} > "$CLAUDE_CALLS"
"""


@pytest.fixture
def claude(tmp_path, monkeypatch):
    """A stand-in claude first on PATH; the path of what it noted. No library,
    whatever the host has: a test that wants one names it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "claude").write_text(CLAUDE)
    (bin_dir / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CLAUDE_CALLS", str(tmp_path / "calls"))
    monkeypatch.delenv("CRYOET_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("CRYOET_ASSISTANT", raising=False)
    monkeypatch.delenv("CRYOET_ASSISTANT_BACKEND", raising=False)
    monkeypatch.setenv("CRYOET_LIBRARY", "none")
    monkeypatch.delenv("CRYOET_LIBRARY_ROOT", raising=False)
    monkeypatch.delenv("CRYOET_LIBRARY_SNAPSHOT", raising=False)
    return tmp_path / "calls"


@pytest.fixture
def library(tmp_path):
    """A library folder as scripts/build_library.py leaves it: one snapshot,
    `current` pointing at it; and a link to the snapshot from elsewhere."""
    root = tmp_path / "cryoet-library"
    snapshot = root / "site-2026-11-05"
    (snapshot / "text" / "imod").mkdir(parents=True)
    (snapshot / "text" / "imod" / "newstack.md").write_text("# newstack\n")
    (snapshot / "library.json").write_text("{}\n")
    (root / "current").symlink_to("site-2026-11-05")
    link = tmp_path / "elsewhere"
    link.symlink_to(snapshot)
    return {
        "root": root.resolve(),
        "snapshot": snapshot.resolve(),
        "link": link,
        "tmp": tmp_path.resolve(),
    }


def library_env(library: dict) -> dict[str, str]:
    return {
        "CRYOET_LIBRARY_ROOT": str(library["root"]),
        "CRYOET_LIBRARY_SNAPSHOT": str(library["snapshot"]),
    }


def _launch(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )


def _noted(calls: Path) -> tuple[dict[str, str], list[str]]:
    text = calls.read_text()
    head, _, tail = text.partition("arg=")
    fields = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    args = [arg.removeprefix("arg=") for arg in ("arg=" + tail).split("\0") if arg]
    return fields, args


def test_the_launcher_starts_claude_with_the_assistants_setup(claude, tmp_path, monkeypatch):
    # The person's own choice of editor connection does not carry over.
    monkeypatch.setenv("CLAUDE_CODE_AUTO_CONNECT_IDE", "true")
    # From anywhere: it runs in the repository root, where the scripts' paths start.
    result = _launch("-p", "where do we stand?", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    fields, args = _noted(claude)
    assert fields == {
        "cwd": str(ROOT),
        "CRYOET_ASSISTANT": "cryoet-assistant; claude-code 9.9.9",
        "CRYOET_ASSISTANT_MODEL": "claude-opus-5-5",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
        "CLAUDE_CODE_AUTO_CONNECT_IDE": "false",
        "CRYOET_LIBRARY_ROOT": "",
        "CRYOET_LIBRARY_SNAPSHOT": "",
    }
    # No MCP server from the person's settings, plugins or account.
    assert "--strict-mcp-config" in args and "--mcp-config" not in args
    # No library named, so none to read.
    assert "--add-dir" not in args
    expected = {
        "--model": "claude-opus-5-5",
        "--setting-sources": "user",
        "--settings": str(ASSISTANT / "settings.json"),
        "--plugin-dir": str(ASSISTANT / "plugin"),
        "--append-system-prompt": (ASSISTANT / "instructions.md").read_text().rstrip("\n"),
        "--name": "cryoet-assistant",
    }
    for option, value in expected.items():
        assert args[args.index(option) + 1] == value, option
    # What the person gave comes last, as given.
    assert args[-2:] == ["-p", "where do we stand?"]


@pytest.mark.parametrize(
    ("args", "env", "model"),
    [
        (["--model", "claude-sonnet-5-5"], {}, "claude-sonnet-5-5"),
        (["--model=claude-haiku-5-5"], {}, "claude-haiku-5-5"),
        ([], {"CRYOET_ASSISTANT_MODEL": "claude-fable-5-1"}, "claude-fable-5-1"),
    ],
)
def test_the_journal_names_the_model_the_session_runs(claude, monkeypatch, args, env, model):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _launch(*args, cwd=ROOT).returncode == 0
    fields, passed = _noted(claude)
    assert fields["CRYOET_ASSISTANT_MODEL"] == passed[passed.index("--model") + 1] == model


@pytest.mark.parametrize(
    "args",
    [
        ["--dangerously-skip-permissions"],
        ["--permission-mode", "bypassPermissions"],
        ["--permission-mode=acceptEdits"],
        ["--settings", "other.json"],
        ["--setting-sources", "user,project,local"],
        ["--allowed-tools", "Bash(sbatch *)"],
        ["--system-prompt", "anything"],
        ["--plugin-dir", "elsewhere"],
        ["--safe-mode"],
        ["--bare"],
        ["--mcp-config", "servers.json"],
        ["--mcp-config=servers.json"],
        ["--ide"],
        ["--model", "opus"],
    ],
)
def test_options_that_would_undo_the_setup_are_refused(claude, args):
    result = _launch(*args, cwd=ROOT)
    assert result.returncode == 2 and "scripts/assistant:" in result.stderr
    assert not claude.exists()


def test_accept_edits_passes_only_for_a_headless_run(claude):
    # An evaluation run has nobody to approve an edit; a session for a person does.
    assert _launch("-p", "hello", "--permission-mode", "acceptEdits", cwd=ROOT).returncode == 0
    _, passed = _noted(claude)
    assert passed[passed.index("--permission-mode") + 1] == "acceptEdits"
    claude.unlink()
    for args in (
        ["--permission-mode", "acceptEdits"],
        ["--permission-mode=acceptEdits"],
        ["-p", "hello", "--permission-mode", "bypassPermissions"],
    ):
        result = _launch(*args, cwd=ROOT)
        assert result.returncode == 2 and "scripts/assistant:" in result.stderr
        assert not claude.exists()


def test_without_claude_the_launcher_says_so(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)], cwd=ROOT, capture_output=True, text=True, env=env
    )
    assert result.returncode == 1 and "Claude Code (claude) is not on PATH" in result.stderr
    assert "scripts/assistant --codex" in result.stderr


def test_print_command_runs_nothing(claude):
    result = _launch("--print-command", "-c", cwd=ROOT)
    assert result.returncode == 0 and not claude.exists()
    assert "CRYOET_ASSISTANT_MODEL=claude-opus-5-5" in result.stdout
    assert "CLAUDE_CODE_AUTO_CONNECT_IDE=false" in result.stdout
    assert " --strict-mcp-config" in result.stdout
    assert '--append-system-prompt "$(cat assistant/instructions.md)"' in result.stdout
    assert result.stdout.rstrip().endswith(" -c")


def test_print_command_adds_no_empty_argument(claude):
    result = _launch("--print-command", cwd=ROOT)
    assert result.returncode == 0
    assert result.stdout.rstrip().endswith("--name cryoet-assistant")


# --- the reference library --------------------------------------------------------


def test_a_session_reads_the_snapshot_current_names(claude, library, monkeypatch):
    monkeypatch.setenv("CRYOET_LIBRARY", str(library["tmp"] / "cryoet-library"))
    result = _launch("-p", "what does newstack -bin do?", cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "library snapshot site-2026-11-05, read-only" in result.stderr
    fields, args = _noted(claude)
    assert fields["CRYOET_LIBRARY_ROOT"] == str(library["root"])
    assert fields["CRYOET_LIBRARY_SNAPSHOT"] == str(library["snapshot"])
    assert args[args.index("--add-dir") + 1] == str(library["snapshot"])
    prompt = args[args.index("--append-system-prompt") + 1]
    assert prompt == "\n\n".join(
        [
            (ASSISTANT / "instructions.md").read_text().rstrip("\n"),
            (ASSISTANT / "library.md").read_text().rstrip("\n"),
            f"This session's library is {library['snapshot']} (snapshot site-2026-11-05).",
        ]
    )


def test_none_and_a_folder_without_a_snapshot_start_without_the_library(
    claude, library, monkeypatch
):
    plain = (ASSISTANT / "instructions.md").read_text().rstrip("\n")
    for value, said in (("none", ""), (str(library["tmp"]), "no library snapshot at")):
        monkeypatch.setenv("CRYOET_LIBRARY", value)
        result = _launch("-p", "hello", cwd=ROOT)
        assert result.returncode == 0 and said in result.stderr
        fields, args = _noted(claude)
        assert "--add-dir" not in args and fields["CRYOET_LIBRARY_SNAPSHOT"] == ""
        assert args[args.index("--append-system-prompt") + 1] == plain
        claude.unlink()


@pytest.fixture
def site_launcher(tmp_path, library):
    """The launcher in a copy of the repository whose site list names the test's
    library folder."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    text = LAUNCHER.read_text()
    sites = re.search(r"^LIBRARY_SITES=\(.*\)$", text, re.MULTILINE).group(0)
    (repo / "scripts" / "assistant").write_text(
        text.replace(sites, f"LIBRARY_SITES=({library['root']})")
    )
    (repo / "assistant").symlink_to(ASSISTANT)
    return repo / "scripts" / "assistant"


def test_a_person_gets_the_sites_library_and_a_headless_run_only_a_named_one(
    claude, library, site_launcher, monkeypatch
):
    monkeypatch.delenv("CRYOET_LIBRARY")

    def launch(*args: str) -> list[str]:
        result = subprocess.run(
            ["bash", str(site_launcher), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=dict(os.environ),
        )
        assert result.returncode == 0, result.stderr
        _, passed = _noted(claude)
        claude.unlink()
        return passed

    passed = launch()
    assert passed[passed.index("--add-dir") + 1] == str(library["snapshot"])
    # An evaluation reads the library only when it names one.
    assert "--add-dir" not in launch("-p", "hello")


def test_print_command_shows_the_library(claude, library, monkeypatch):
    monkeypatch.setenv("CRYOET_LIBRARY", str(library["root"]))
    result = _launch("--print-command", cwd=ROOT)
    assert result.returncode == 0 and not claude.exists()
    assert f"CRYOET_LIBRARY_SNAPSHOT={library['snapshot']}" in result.stdout
    assert f"--add-dir {library['snapshot']}" in result.stdout
    assert "$(cat assistant/instructions.md assistant/library.md)" in result.stdout


def test_the_launcher_and_the_guard_keep_the_same_site_folders():
    launcher = re.search(r"^LIBRARY_SITES=\((.*)\)$", LAUNCHER.read_text(), re.MULTILINE)
    guard = re.search(
        r"^LIBRARY_SITES = \((.*)\)$", (ASSISTANT / "guard.py").read_text(), re.MULTILINE
    )
    sites = launcher.group(1).split()
    assert sites and sites == re.findall(r'"([^"]+)"', guard.group(1))
    for site in sites:
        assert f"Edit(/{site}/**)" in _rules()["deny"]


def _rules() -> dict[str, list[str]]:
    return json.loads((ASSISTANT / "settings.json").read_text())["permissions"]


def test_the_client_rules_hold_the_gates():
    rules = _rules()
    for rule in (
        "Bash(sbatch *)",
        "Bash(srun *)",
        "Bash(salloc *)",
        "Bash(git *)",
        "Bash(python3 scripts/run_series.py * --execute*)",
        "Edit(**/journal.jsonl)",
        "Edit(**/*.sbatch)",
        "Edit(scripts/**)",
        "Edit(src/**)",
        "Edit(tests/**)",
        "Edit(docs/**)",
        "Edit(assistant/**)",
        "Edit(.claude/**)",
    ):
        assert rule in rules["deny"], rule
    assert {"Bash(scancel *)", "Edit(configs/lanes/**)"} <= set(rules["ask"])
    # Claude Code 2.1.231 matches file rules by Edit(path) only, for every tool
    # that edits a file; a Write(path) rule is ignored, with a warning.
    every = [rule for kind in ("allow", "ask", "deny") for rule in rules[kind]]
    assert not [rule for rule in every if rule.startswith("Write(")]
    # Real submissions are not allowed outright: only dry runs are.
    submits = [rule for rule in rules["allow"] if "submit_" in rule]
    assert submits and all("--dry-run" in rule or "--list" in rule for rule in submits)
    # Nor is a real export to RELION: it writes the tilt series, and asks first.
    exports = [rule for rule in rules["allow"] if "export_relion" in rule]
    assert exports and all("--dry-run" in rule for rule in exports)
    # The person is asked, not a classifier, and nothing skips the rules.
    assert rules["defaultMode"] == "default"
    assert rules["disableBypassPermissionsMode"] == rules["disableAutoMode"] == "disable"


def test_every_script_the_rules_allow_exists():
    for rule in _rules()["allow"]:
        for script in re.findall(r"scripts/[\w./-]+\.py", rule):
            assert (ROOT / script).is_file(), rule


def test_the_guard_runs_on_every_shell_command_and_every_file_edit():
    hooks = json.loads((ASSISTANT / "settings.json").read_text())["hooks"]["PreToolUse"]
    for matcher in ("Bash", "Edit|Write|MultiEdit|NotebookEdit"):
        (entry,) = [entry for entry in hooks if entry["matcher"] == matcher]
        (command,) = [hook["command"] for hook in entry["hooks"]]
        assert command == 'python3 "$CLAUDE_PROJECT_DIR/assistant/guard.py"'


def test_the_rules_deny_the_library_builder():
    deny = _rules()["deny"]
    assert {
        "Bash(python3 scripts/build_library.py *)",
        "Bash(python scripts/build_library.py *)",
    } <= set(deny)


def _guard(
    command: str = "",
    tool: str = "Bash",
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    tool_input: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    call: dict = {
        "tool_name": tool,
        "tool_input": tool_input if tool_input is not None else {"command": command},
    }
    if cwd is not None:
        call["cwd"] = str(cwd)
    environ = {
        name: value for name, value in os.environ.items() if not name.startswith("CRYOET_LIBRARY")
    }
    environ.update(env or {})
    return subprocess.run(
        [sys.executable, str(ASSISTANT / "guard.py")],
        input=json.dumps(call),
        capture_output=True,
        text=True,
        env=environ,
    )


@pytest.mark.parametrize(
    "command",
    [
        "sbatch scripts/lane.sbatch",
        "cd outputs && sbatch job.sh",
        "echo $(sbatch x)",
        "echo `sbatch x`",
        "env OVERWRITE=1 sbatch x",
        "timeout 10 srun -n1 hostname",
        "nohup salloc &",
        "/usr/bin/sbatch x",
        'bash -c "sbatch x"',
        # A shell's -c among its other options, as Codex's models write it.
        'bash -lc "sbatch x"',
        "sh -ec 'cd outputs && srun -n1 hostname'",
        "bash scripts/lane.sbatch",
        "./scripts/ctf_window_test.sbatch",
        "echo '{}' >> outputs/ryr1/journal.jsonl",
        "sed -i s/a/b/ outputs/ryr1/journal.jsonl",
        "rm outputs/ryr1/journal.jsonl",
        "cp edited.jsonl outputs/ryr1/journal.jsonl",
        "echo 'a quote that does not close; sbatch x",
        "command sbatch x",
        "command -p sbatch x",
        "command -v ls && sbatch x",
        # RELION itself, which would compute on the login node.
        "relion_refine --i particles.star --o Refine3D/run",
        "mpirun -n 3 relion_refine_mpi --ios Extract/job001/optimisation_set.star",
        "mpiexec -n 2 relion_tomo_subtomo --i optimisation_set.star",
        "cd outputs/ryr1/relion/run2-class1 && ./relion_commands.sh",
        "bash outputs/ryr1/relion/run2-class1/relion_commands.sh",
        "nohup relion_tomo_subtomo --i optimisation_set.star &",
        "env THREADS=8 ~/relion/bin/relion_refine_mpi --ios x",
    ],
)
def test_the_guard_blocks_jobs_batch_scripts_and_journal_writes(command):
    result = _guard(command)
    assert result.returncode == 2 and "Blocked by assistant/guard.py" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "python3 scripts/submit_lanes.py configs/lanes/ryr1-bin8.toml --dry-run",
        'python3 scripts/submit_lanes.py c.toml --reason "Anna: not with sbatch by hand"',
        "cat outputs/ryr1/journal.jsonl",
        "cp outputs/ryr1/journal.jsonl /fast/scratch/journal-copy.jsonl",
        "grep sbatch scripts/lane.sbatch",
        "squeue -u $USER",
        "sacct -j 3670792 -o JobID,State",
        # Where sbatch is, which runs nothing; blocked in the first session.
        "command -v sbatch",
        "command -V srun",
        "type sbatch",
        # Where RELION is, and what an export holds, run nothing.
        "command -v relion_tomo_reconstruct_particle",
        "cat outputs/ryr1/relion/run2-class1/relion_commands.sh",
        "python3 scripts/export_relion.py --run outputs/ryr1/mra/run2 --class 1 --dry-run",
        'python3 scripts/export_relion.py --run outputs/ryr1/mra/run2 --class 1 --reason "Anna"',
    ],
)
def test_the_guard_lets_the_rest_through(command):
    assert _guard(command).returncode == 0


def test_the_guard_does_not_judge_what_only_reads():
    assert _guard("sbatch x", tool="Read").returncode == 0


@pytest.mark.parametrize(
    "command",
    [
        "bash",
        "zsh -l",
        "echo 'sbatch x' | bash",
        "cat steps.txt | sh -s",
        "bash <<'EOF'\nsbatch x\nEOF",
        "bash <<< 'sbatch x'",
        "bash < steps.sh",
        "cd outputs && bash",
    ],
)
def test_the_guard_blocks_a_shell_that_runs_its_input(command):
    # Codex can start a shell and type into it later, where the guard never looks.
    result = _guard(command)
    assert result.returncode == 2 and "reads its commands from its input" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "bash --version",
        "bash scripts/check.sh",
        "bash -lc 'ls outputs'",
        "bash -o pipefail scripts/check.sh",
        "command -v bash",
        "which zsh",
    ],
)
def test_the_guard_lets_a_shell_with_something_to_run_through(command):
    assert _guard(command).returncode == 0, command


@pytest.mark.parametrize(
    "command",
    [
        "python3 scripts/run_series.py --project ryr1 --series 097 --execute",
        "python3 scripts/run_series.py --execute --project ryr1",
        # argparse takes the beginning of an option for the option.
        "python3 scripts/run_series.py --project ryr1 --exe",
        "./scripts/run_series.py --project ryr1 --execute",
        "bash -lc 'python3 scripts/run_series.py --project ryr1 --execute'",
    ],
)
def test_the_guard_blocks_run_series_execute(command):
    result = _guard(command)
    assert result.returncode == 2 and "run_series.py --execute" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "python3 scripts/run_series.py --project ryr1 --series 097",
        "python3 scripts/run_series.py --list-steps",
        "grep -n execute scripts/run_series.py",
    ],
)
def test_the_guard_lets_run_series_print(command):
    assert _guard(command).returncode == 0, command


# --- Codex's patches ----------------------------------------------------------------


def _patch(*lines: str) -> str:
    return "*** Begin Patch\n" + "\n".join(lines) + "\n*** End Patch\n"


def _shell_patch(patch: str) -> str:
    """The same patch as Codex also takes it, as a shell command."""
    return f"apply_patch <<'EOF'\n{patch}EOF"


def _edit_rules() -> list[str]:
    return [rule[5:-1] for rule in _rules()["deny"] if rule.startswith("Edit(")]


_DENIED_PATCHES = [
    (_patch("*** Add File: src/cryoet_pipeline/new.py", "+x = 1"), "src/**"),
    (_patch("*** Update File: scripts/submit_lanes.py", "@@", "-a", "+b"), "scripts/**"),
    (_patch("*** Delete File: tests/test_assistant.py"), "tests/**"),
    (_patch("*** Update File: docs/agent/playbook/ctf.md", "@@", "-a", "+b"), "docs/**"),
    (_patch("*** Update File: assistant/guard.py", "@@", "-a", "+b"), "assistant/**"),
    (_patch("*** Add File: assistant/codex/extra.rules", "+x"), "assistant/**"),
    (_patch("*** Add File: .claude/settings.local.json", "+{}"), ".claude/**"),
    (_patch("*** Add File: outputs/ryr1/journal.jsonl", "+{}"), "**/journal.jsonl"),
    (_patch("*** Update File: /fast/x/ryr1/journal.jsonl", "@@", "-a", "+b"), "**/journal.jsonl"),
    (_patch("*** Add File: configs/lanes/run.sbatch", "+#!/bin/bash"), "**/*.sbatch"),
    # A move is checked at its destination too.
    (
        _patch(
            "*** Update File: configs/lanes/a.toml", "*** Move to: scripts/a.toml", "@@", "-a", "+b"
        ),
        "scripts/**",
    ),
    # One file the rules deny among others they do not.
    (
        _patch(
            "*** Update File: configs/lanes/a.toml",
            "@@",
            "-a",
            "+b",
            "*** Add File: src/x.py",
            "+x",
        ),
        "src/**",
    ),
    # A path that leaves the folder it starts from.
    (_patch("*** Add File: configs/../src/x.py", "+x"), "src/**"),
]


@pytest.mark.parametrize(("patch", "rule"), _DENIED_PATCHES)
def test_the_guard_blocks_a_codex_patch_the_edit_rules_deny(patch, rule):
    for result in (
        _guard(tool="apply_patch", tool_input={"command": patch}, cwd=ROOT),
        _guard(_shell_patch(patch), cwd=ROOT),
    ):
        assert result.returncode == 2, result.stderr
        assert f"assistant/settings.json denies Edit({rule})" in result.stderr


def test_every_edit_rule_has_a_denied_patch():
    # The site's library folder is the library's own case, below.
    named = {rule for _, rule in _DENIED_PATCHES}
    assert set(_edit_rules()) - named == {"//fast/AG_Kudryashev/cryoet-library/**"}


@pytest.mark.parametrize(
    "patch",
    [
        _patch("*** Add File: configs/lanes/ryr1-bin4.toml", "+[lane]"),
        _patch("*** Update File: configs/lanes/ryr1-bin8.toml", "@@", "-pixel = 1", "+pixel = 2"),
        _patch("*** Add File: outputs/ryr1/notes.md", "+notes"),
        _patch("*** Add File: /tmp/scratch-notes.md", "+notes"),
        # Words like a rule's inside a path that is not under it.
        _patch("*** Add File: configs/lanes/scripts-notes.toml", "+x"),
    ],
)
def test_the_guard_lets_other_codex_patches_through(patch):
    # The person approves each of them: Codex's sandbox writes nothing by itself.
    assert _guard(tool="apply_patch", tool_input={"command": patch}, cwd=ROOT).returncode == 0
    assert _guard(_shell_patch(patch), cwd=ROOT).returncode == 0


def test_a_codex_patch_in_the_library_is_refused(library):
    env = library_env(library)
    for path in (
        library["snapshot"] / "text" / "imod" / "newstack.md",
        library["link"] / "new.md",
        Path("/fast/AG_Kudryashev/cryoet-library/current/x.md"),
    ):
        patch = _patch(f"*** Update File: {path}", "@@", "-a", "+b")
        result = _guard(tool="apply_patch", tool_input={"command": patch}, cwd=ROOT, env=env)
        assert result.returncode == 2 and "no tool edits it" in result.stderr, path


@pytest.mark.parametrize("command", ["not a patch", "", ["*** Begin Patch"], None])
def test_a_codex_patch_the_guard_cannot_read_is_refused(command):
    result = _guard(tool="apply_patch", tool_input={"command": command}, cwd=ROOT)
    assert result.returncode == 2 and "Blocked by assistant/guard.py" in result.stderr


def test_without_settings_json_a_codex_patch_is_refused(tmp_path):
    # The guard reads the Edit rules for Codex; without them it lets no patch through.
    (tmp_path / "assistant").mkdir()
    guard = tmp_path / "assistant" / "guard.py"
    guard.write_text((ASSISTANT / "guard.py").read_text())
    call = {
        "tool_name": "apply_patch",
        "tool_input": {"command": _patch("*** Add File: notes.md", "+x")},
        "cwd": str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, str(guard)], input=json.dumps(call), capture_output=True, text=True
    )
    assert result.returncode == 2 and "settings.json cannot be read" in result.stderr


def test_the_guard_leaves_claude_codes_edits_to_its_rules():
    # Claude Code applies the Edit rules of settings.json itself.
    assert _guard(tool="Edit", tool_input={"file_path": "src/x.py"}, cwd=ROOT).returncode == 0


@pytest.mark.parametrize(
    "command",
    [
        "python3 scripts/build_library.py --site mdc --dry-run",
        "/opt/env/bin/python3 -I /fast/x/scripts/build_library.py --site mdc",
        "env OVERWRITE=1 python3 scripts/build_library.py --site mdc",
        "bash -c 'python3 scripts/build_library.py --site mdc'",
        "./scripts/build_library.py --site mdc",
    ],
)
def test_the_guard_blocks_the_library_builder(command):
    result = _guard(command)
    assert result.returncode == 2 and "build_library.py writes" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "rm {snapshot}/text/imod/newstack.md",
        "mv {root}/current {root}/old",
        "chmod -R u+w {root}",
        "echo hi > {snapshot}/text/imod/newstack.md",
        "sed -i s/a/b/ {snapshot}/text/imod/newstack.md",
        "cp /etc/hosts {snapshot}/text/",
        "touch {snapshot}/new.md",
        "ln -sfn elsewhere {root}/current",
        "bash -c 'rm {snapshot}/text/imod/newstack.md'",
        "rm $CRYOET_LIBRARY_SNAPSHOT/text/imod/newstack.md",
        "find {snapshot} -name '*.md' -delete",
        "find {snapshot} -name '*.md' -exec rm {{}} +",
        "python3 -c \"open('{snapshot}/x', 'w')\"",
        "cat {snapshot}/text/imod/newstack.md | tee {snapshot}/copy.md",
        "rm {link}/text/imod/newstack.md",
        "rm {tmp}/cryoet-lib*/current",
        "grep -rn newstack {snapshot}/text > {snapshot}/hits.txt",
        "rm /fast/AG_Kudryashev/cryoet-library/current",
        # The search is a reader only as itself: not a copy of the same name, not
        # with code run before it, and not writing into the library.
        "python3 {tmp}/search_library.py x --snapshot {snapshot}",
        "python3 -c 'import os' scripts/search_library.py x --snapshot {snapshot}",
        "python3 -c scripts/search_library.py x --snapshot {snapshot}",
        "python3 scripts/search_library.py x --snapshot {snapshot} > {snapshot}/out.txt",
    ],
)
def test_the_guard_blocks_every_write_to_the_library(library, command):
    result = _guard(command.format(**library), env=library_env(library))
    assert result.returncode == 2, command
    assert "reference library, which is read-only" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "grep -rn newstack {snapshot}/text",
        "ls -la {root}",
        "head -50 {snapshot}/text/imod/newstack.md",
        "find {snapshot} -name '*.md' | wc -l",
        "cat {snapshot}/library.json | head -5",
        "cd {snapshot} && grep -rln newstack text",
        "grep -c newstack {snapshot}/text/imod/newstack.md > /dev/null",
        "rm -f {tmp}/scratch.txt",
        "cat scripts/build_library.py",
        # The library's own search, given the snapshot by its path.
        "python3 scripts/search_library.py -e -bin --snapshot {snapshot}",
        "python3 -I scripts/search_library.py -e -bin --snapshot {snapshot}",
        "python3 scripts/search_library.py newstack --snapshot $CRYOET_LIBRARY_SNAPSHOT | head -20",
    ],
)
def test_the_guard_lets_the_library_be_read(library, command):
    assert _guard(command.format(**library), env=library_env(library)).returncode == 0, command


def test_inside_the_library_every_command_only_reads(library):
    env = library_env(library)
    assert _guard("rm text/imod/newstack.md", cwd=library["snapshot"], env=env).returncode == 2
    assert _guard("grep -rn newstack text", cwd=library["snapshot"], env=env).returncode == 0


@pytest.mark.parametrize(
    ("tool", "key"),
    [
        ("Edit", "file_path"),
        ("Write", "file_path"),
        ("MultiEdit", "file_path"),
        ("NotebookEdit", "notebook_path"),
    ],
)
def test_no_tool_edits_a_file_in_the_library(library, tool, key):
    env = library_env(library)
    for path in (
        library["snapshot"] / "text" / "imod" / "newstack.md",
        library["root"] / "current" / "site.md",
        library["link"] / "new.md",
    ):
        result = _guard(tool=tool, tool_input={key: str(path)}, env=env)
        assert result.returncode == 2 and "no tool edits it" in result.stderr, path
    assert (
        _guard(tool=tool, tool_input={key: "new.md"}, cwd=library["snapshot"], env=env).returncode
        == 2
    )
    assert (
        _guard(tool=tool, tool_input={key: str(library["tmp"] / "notes.md")}, env=env).returncode
        == 0
    )
    # A site's folder is guarded when the session names no library.
    assert (
        _guard(
            tool=tool, tool_input={key: "/fast/AG_Kudryashev/cryoet-library/current/x.md"}
        ).returncode
        == 2
    )


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), path
    block = text.split("---\n", 2)[1]
    return dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)


@pytest.mark.parametrize("skill", SKILLS, ids=lambda path: path.parent.name)
def test_each_skill_has_a_name_and_a_description(skill):
    fields = _frontmatter(skill)
    assert fields["name"] == skill.parent.name
    assert len(fields["description"]) > 80


def test_the_skills_are_there():
    assert [path.parent.name for path in SKILLS] == [
        "emdb-template",
        "empiar-download",
        "lane-config",
        "project-status",
        "relion-export",
    ]


# Repository paths the assistant's files name: scripts, docs, configs, templates.
_PATH = re.compile(r"(?<![\w/.<])((?:scripts|docs|configs|assistant)/[\w./-]*[\w/])")


@pytest.mark.parametrize(
    "document",
    [
        *SKILLS,
        ASSISTANT / "instructions.md",
        ASSISTANT / "library.md",
        ASSISTANT / "README.md",
        ASSISTANT / "codex" / "instructions.md",
    ],
    ids=lambda path: str(path.relative_to(ASSISTANT)),
)
def test_every_repository_path_the_assistant_is_told_about_exists(document):
    named = set(_PATH.findall(document.read_text()))
    assert named, document
    missing = sorted(path for path in named if not (ROOT / path.rstrip("/")).exists())
    assert not missing, f"{document} names paths that do not exist: {missing}"


def test_the_plugin_has_its_manifest():
    manifest = json.loads((ASSISTANT / "plugin" / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "cryoet"


# --- the assistant in Codex -------------------------------------------------------

CODEX_DIR = ASSISTANT / "codex"

# Stands in for codex: answers --version and `login status` (signed in unless
# CODEX_STUB_SIGNED_IN=0), and otherwise notes its arguments, the directory it
# runs in and the variables the launcher sets.
CODEX = """#!/bin/bash
if [[ "$1" == "--version" ]]; then echo "codex-cli 9.9.9"; exit 0; fi
if [[ "$1" == "login" && "$2" == "status" ]]; then
    if [[ "${CODEX_STUB_SIGNED_IN:-1}" == 1 ]]; then echo "Logged in using ChatGPT"; exit 0; fi
    echo "Not logged in"; exit 1
fi
{
    echo "cwd=$PWD"
    echo "CODEX_HOME=${CODEX_HOME:-}"
    echo "CRYOET_ASSISTANT=${CRYOET_ASSISTANT:-}"
    echo "CRYOET_ASSISTANT_MODEL=${CRYOET_ASSISTANT_MODEL:-}"
    echo "CRYOET_ASSISTANT_REPO=${CRYOET_ASSISTANT_REPO:-}"
    echo "CRYOET_LIBRARY_ROOT=${CRYOET_LIBRARY_ROOT:-}"
    echo "CRYOET_LIBRARY_SNAPSHOT=${CRYOET_LIBRARY_SNAPSHOT:-}"
    for arg in "$@"; do printf 'arg=%s\\0' "$arg"; done
} > "$CODEX_CALLS"
"""


@pytest.fixture
def codex(tmp_path, monkeypatch, claude):
    """A stand-in codex first on PATH, next to the stand-in claude; the assistant's
    Codex home in the test's folder. Returns where the stand-in notes its call."""
    (tmp_path / "bin" / "codex").write_text(CODEX)
    (tmp_path / "bin" / "codex").chmod(0o755)
    monkeypatch.setenv("CODEX_CALLS", str(tmp_path / "codex-calls"))
    monkeypatch.setenv("CRYOET_ASSISTANT_CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.delenv("CODEX_STUB_SIGNED_IN", raising=False)
    return {"calls": tmp_path / "codex-calls", "home": tmp_path / "codex-home"}


def _option(args: list[str], name: str) -> str:
    return args[args.index(name) + 1]


def _instructions(args: list[str]) -> str:
    """The developer instructions the launcher gives Codex, read as Codex reads
    its -c values: as TOML."""
    import tomllib

    value = next(arg for arg in args if arg.startswith("developer_instructions="))
    return tomllib.loads(value)["developer_instructions"]


def _text(path: Path) -> str:
    return path.read_text().rstrip("\n")


def test_the_launcher_starts_codex_with_the_assistants_setup(codex, tmp_path):
    result = _launch("--codex", "where do we stand?", cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    fields, args = _noted(codex["calls"])
    assert fields == {
        "cwd": str(ROOT),
        "CODEX_HOME": str(codex["home"]),
        "CRYOET_ASSISTANT": "cryoet-assistant; codex 9.9.9",
        "CRYOET_ASSISTANT_MODEL": "gpt-6.1-sol",
        "CRYOET_ASSISTANT_REPO": str(ROOT),
        "CRYOET_LIBRARY_ROOT": "",
        "CRYOET_LIBRARY_SNAPSHOT": "",
    }
    # A new session, not a resumed one.
    assert args[0] == "--no-daemon"
    for flag in ("--no-daemon", "--strict-config", "--dangerously-bypass-hook-trust"):
        assert flag in args
    assert _option(args, "-C") == str(ROOT)
    assert _option(args, "-m") == "gpt-6.1-sol"
    assert _option(args, "-s") == "read-only"
    assert _option(args, "-a") == "on-request"
    assert _instructions(args) == "\n\n".join(
        [_text(ASSISTANT / "instructions.md"), _text(CODEX_DIR / "instructions.md")]
    )
    # Nothing that would make the library or anything else writable.
    assert "--add-dir" not in args
    assert args[-1] == "where do we stand?"
    # The claude stand-in was not started.
    assert not (tmp_path / "calls").exists()


def test_the_launcher_writes_the_assistants_codex_home(codex):
    import tomllib

    assert _launch("--codex", cwd=ROOT).returncode == 0
    home = codex["home"]
    assert home.stat().st_mode & 0o777 == 0o700
    config = tomllib.loads((home / "config.toml").read_text())
    assert config.pop("projects") == {str(ROOT): {"trust_level": "untrusted"}}
    assert config == tomllib.loads((CODEX_DIR / "config.toml").read_text())
    assert (home / "hooks.json").read_text() == (CODEX_DIR / "hooks.json").read_text()
    rules = home / "rules" / "assistant.rules"
    assert rules.read_text() == (CODEX_DIR / "assistant.rules").read_text()
    assert (home / "skills" / "cryoet").resolve() == (ASSISTANT / "plugin" / "skills").resolve()
    # Written again at the next start, over what was there.
    (home / "hooks.json").write_text("{}")
    assert _launch("--codex", cwd=ROOT).returncode == 0
    assert (home / "hooks.json").read_text() == (CODEX_DIR / "hooks.json").read_text()


def test_the_codex_settings_hold_the_gates():
    import tomllib

    config = tomllib.loads((CODEX_DIR / "config.toml").read_text())
    assert config["sandbox_mode"] == "read-only"
    assert config["approval_policy"] == "on-request"
    # The person approves, not a reviewing model.
    assert config["approvals_reviewer"] == "user"
    assert config["web_search"] == "disabled"
    assert config["features"] == {
        "hooks": True,
        "apps": False,
        "browser_use": False,
        "browser_use_external": False,
        "computer_use": False,
    }
    hooks = json.loads((CODEX_DIR / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {"PreToolUse"}
    for matcher in ("Bash", "apply_patch"):
        (entry,) = [entry for entry in hooks["PreToolUse"] if entry["matcher"] == matcher]
        (command,) = [hook["command"] for hook in entry["hooks"]]
        assert command == 'python3 "$CRYOET_ASSISTANT_REPO/assistant/guard.py"'


@pytest.mark.parametrize(
    ("args", "resume"),
    [
        (["-c"], ["resume", "--last"]),
        (["--continue", "and now?"], ["resume", "--last"]),
        (["-r"], ["resume"]),
        (
            ["--resume", "01a1d033-5bae-7312-95e3-6aa2d3afaf5a"],
            ["resume", "01a1d033-5bae-7312-95e3-6aa2d3afaf5a"],
        ),
    ],
)
def test_continuing_in_codex_is_its_resume(codex, args, resume):
    assert _launch("--codex", *args, cwd=ROOT).returncode == 0
    _, passed = _noted(codex["calls"])
    assert passed[: len(resume)] == resume
    assert passed[len(resume)] == "--no-daemon"
    if args[-1] == "and now?":
        assert passed[-1] == "and now?"


def test_the_client_comes_from_the_variable_and_an_option_wins(codex, monkeypatch, tmp_path):
    monkeypatch.setenv("CRYOET_ASSISTANT_BACKEND", "codex")
    assert _launch(cwd=ROOT).returncode == 0
    assert codex["calls"].exists() and not (tmp_path / "calls").exists()
    codex["calls"].unlink()
    assert _launch("--claude", cwd=ROOT).returncode == 0
    assert (tmp_path / "calls").exists() and not codex["calls"].exists()


@pytest.mark.parametrize(
    "args",
    [
        ["--codex", "--dangerously-bypass-approvals-and-sandbox"],
        ["--codex", "--yolo"],
        ["--codex", "-s", "danger-full-access"],
        ["--codex", "--sandbox=workspace-write"],
        ["--codex", "-a", "never"],
        ["--codex", "--approve-for-me"],
        ["--codex", "--add-dir", "/fast"],
        ["--codex", "-C", "/tmp"],
        ["--codex", "--config", "approval_policy='never'"],
        ["--codex", "--enable", "apps"],
        ["--codex", "-p", "hello"],
        ["--codex", "--model", "claude-opus-5-5"],
        ["--codex", "one", "two"],
        ["--codex", "--device-auth"],
        ["--claude", "--codex"],
    ],
)
def test_options_that_would_undo_the_codex_setup_are_refused(codex, args):
    result = _launch(*args, cwd=ROOT)
    assert result.returncode == 2 and "scripts/assistant:" in result.stderr
    assert not codex["calls"].exists()


def test_an_unknown_client_is_refused(codex, monkeypatch):
    monkeypatch.setenv("CRYOET_ASSISTANT_BACKEND", "chatgpt")
    result = _launch(cwd=ROOT)
    assert result.returncode == 2 and "give claude or codex" in result.stderr


def test_codex_not_signed_in_is_told_how(codex, monkeypatch):
    monkeypatch.setenv("CODEX_STUB_SIGNED_IN", "0")
    result = _launch("--codex", cwd=ROOT)
    assert result.returncode == 1 and "scripts/assistant --codex --login" in result.stderr
    assert not codex["calls"].exists()


@pytest.mark.parametrize(
    ("args", "login"), [([], ["login"]), (["--device-auth"], ["login", "--device-auth"])]
)
def test_the_assistants_codex_signs_in_in_its_own_home(codex, monkeypatch, args, login):
    monkeypatch.setenv("CODEX_STUB_SIGNED_IN", "0")
    assert _launch("--codex", "--login", *args, cwd=ROOT).returncode == 0
    fields, passed = _noted(codex["calls"])
    assert passed == login and fields["CODEX_HOME"] == str(codex["home"])


def test_print_command_for_codex_runs_and_writes_nothing(codex):
    result = _launch("--codex", "--print-command", "-c", "and now?", cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert not codex["calls"].exists() and not codex["home"].exists()
    assert f"CODEX_HOME={codex['home']}" in result.stdout
    assert "CRYOET_ASSISTANT_MODEL=gpt-6.1-sol" in result.stdout
    assert (
        " resume --last --no-daemon --strict-config --dangerously-bypass-hook-trust"
        in result.stdout
    )
    assert " -s read-only -a on-request" in result.stdout
    assert "$(cat assistant/instructions.md assistant/codex/instructions.md)" in result.stdout
    assert result.stdout.rstrip().endswith("and\\ now\\?")


def test_without_codex_the_launcher_says_so(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER), "--codex"], cwd=ROOT, capture_output=True, text=True, env=env
    )
    assert result.returncode == 1 and "Codex (codex) is not on PATH" in result.stderr


def test_a_codex_session_reads_the_library_without_writing_it(codex, library, monkeypatch):
    monkeypatch.setenv("CRYOET_LIBRARY", str(library["tmp"] / "cryoet-library"))
    result = _launch("--codex", cwd=ROOT)
    assert result.returncode == 0, result.stderr
    fields, args = _noted(codex["calls"])
    assert fields["CRYOET_LIBRARY_ROOT"] == str(library["root"])
    assert fields["CRYOET_LIBRARY_SNAPSHOT"] == str(library["snapshot"])
    # Codex's --add-dir makes a folder writable: the sandbox reads everything anyway.
    assert "--add-dir" not in args
    assert _instructions(args) == "\n\n".join(
        [
            _text(ASSISTANT / "instructions.md"),
            _text(ASSISTANT / "library.md"),
            f"This session's library is {library['snapshot']} (snapshot site-2026-11-05).",
            _text(CODEX_DIR / "instructions.md"),
        ]
    )


def test_instructions_with_any_character_reach_codex_as_written(codex, tmp_path):
    # The launcher writes the instructions as a TOML string; read them back as one.
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "assistant").write_text(LAUNCHER.read_text())
    (repo / "assistant" / "codex").mkdir(parents=True)
    for name in ("config.toml", "hooks.json", "assistant.rules", "instructions.md"):
        (repo / "assistant" / "codex" / name).write_text((CODEX_DIR / name).read_text())
    (repo / "assistant" / "plugin" / "skills").mkdir(parents=True)
    text = "Quotes \" and ''' , a backslash \\ and \\n, a $HOME, `ticks`, a\ttab & more\n\nend"
    (repo / "assistant" / "instructions.md").write_text(text + "\n")
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "assistant"), "--codex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr
    _, args = _noted(codex["calls"])
    assert _instructions(args) == text + "\n\n" + _text(CODEX_DIR / "instructions.md")


def test_codex_is_told_the_dry_run_form_its_rules_allow():
    text = (CODEX_DIR / "instructions.md").read_text()
    for script in ("submit_lanes.py", "submit_recipe.py", "export_relion.py"):
        assert f"python3 scripts/{script} --dry-run" in text


# --- Codex itself: its rules and its hooks ------------------------------------------

REAL_CODEX = shutil.which("codex")
needs_codex = pytest.mark.skipif(REAL_CODEX is None, reason="Codex (codex) is not installed here")


def _codex_decision(command: str) -> str | None:
    """What the assistant's Codex rules decide for a command: allow, prompt,
    forbidden, or None for the sandbox."""
    result = subprocess.run(
        [
            REAL_CODEX,
            "execpolicy",
            "check",
            "--rules",
            str(CODEX_DIR / "assistant.rules"),
            "--",
            *shlex.split(command),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout).get("decision")


@needs_codex
def test_the_codex_rules_keep_the_bash_rules_of_settings_json():
    expected = {"allow": "allow", "ask": "prompt", "deny": "forbidden"}
    for kind, decision in expected.items():
        for rule in _rules()[kind]:
            if not rule.startswith("Bash("):
                continue
            pattern = rule[len("Bash(") : -1]
            command = pattern.replace(" *", " x").replace("*", "")
            if "*" not in pattern.rstrip("*"):
                assert _codex_decision(command) == decision, (rule, command)
            elif kind == "allow":
                # A * in the middle: the dry run with --dry-run elsewhere than
                # right after the script asks the person in Codex.
                assert _codex_decision(command) is None, (rule, command)
            else:
                # run_series.py with --execute anywhere: the guard blocks it.
                assert _guard(command).returncode == 2, (rule, command)


@needs_codex
def test_the_codex_rules_let_through_nothing_else_that_writes():
    # Codex alone allows readers, so that a script piped into head runs as itself.
    for command in ("cat x", "head -5 x", "tail -n 3 x", "grep -n a x", "wc -l x", "cut -f1 x"):
        assert _codex_decision(command) == "allow", command
    for command in (
        "rm -rf outputs",
        "mv a b",
        "cp a b",
        "tee x",
        "sort -o x y",
        "find . -delete",
        "python3 -c 'print(1)'",
        "python3 scripts/submit_lanes.py configs/lanes/ryr1-bin8.toml",
        "python3 scripts/export_relion.py --run r --class 1",
        "bash -lc 'ls outputs'",
    ):
        assert _codex_decision(command) is None, command


class _StandIn:
    """A stand-in for OpenAI's Responses API on 127.0.0.1: each request gets the
    next scripted list of output items, as server-sent events, then a plain
    "done"; every request body is kept."""

    def __init__(self, script: list[list[dict]]):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.requests: list[dict] = []
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stand_in.requests.append(body)
                n = len(stand_in.requests) - 1
                items = (
                    script[n]
                    if n < len(script)
                    else [
                        {
                            "type": "message",
                            "role": "assistant",
                            "id": f"m{n}",
                            "content": [{"type": "output_text", "text": "done"}],
                        }
                    ]
                )
                events = [{"type": "response.created", "response": {"id": f"r{n}"}}]
                events += [{"type": "response.output_item.done", "item": item} for item in items]
                events.append(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": f"r{n}",
                            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                        },
                    }
                )
                data = "".join(
                    f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def outputs(self) -> list[str]:
        """What the model was given back for each of its tool calls, in order."""
        found = []
        for body in self.requests[1:]:
            item = [one for one in body["input"] if one.get("type", "").endswith("_output")][-1]
            output = item["output"]
            found.append(
                output
                if isinstance(output, str)
                else "\n".join(part.get("text", "") for part in output)
            )
        return found


def _cell(n: int, code: str) -> list[dict]:
    """A turn in which the model runs code in Codex's JavaScript tool, which is
    how GPT-6.1 calls the shell and apply_patch."""
    return [{"type": "custom_tool_call", "call_id": f"call_{n}", "name": "exec", "input": code}]


def _shell_cell(n: int, command: str) -> list[dict]:
    return _cell(
        n,
        f"const r = await tools.exec_command({{cmd: {json.dumps(command)}, "
        f"login: false}});\ntext(r.output);",
    )


def _patch_cell(n: int, patch: str) -> list[dict]:
    return _cell(n, f"await tools.apply_patch({json.dumps(patch)});\ntext('applied');")


@needs_codex
def test_codex_runs_the_guard_before_its_commands_and_patches(codex, tmp_path):
    # The home and the options are the launcher's own, from a start with the
    # stand-in codex; the real Codex then runs headless, against a stand-in model.
    assert _launch("--codex", cwd=ROOT).returncode == 0
    fields, args = _noted(codex["calls"])
    options = [arg for arg in args if arg not in ("--no-daemon", "-a", "on-request")]
    created = [
        ROOT / "src" / "cryoet_pipeline" / "_codex_guard_check.py",
        ROOT / "tests" / "_codex_guard_check.py",
        ROOT / "configs" / "lanes" / "_codex_guard_check.toml",
    ]
    stand_in = _StandIn(
        [
            _shell_cell(1, "env OVERWRITE=1 sbatch scripts/lane.sbatch"),
            _shell_cell(2, "echo 'sbatch x' | bash"),
            _patch_cell(3, _patch("*** Add File: src/cryoet_pipeline/_codex_guard_check.py", "+x")),
            _shell_cell(4, _shell_patch(_patch("*** Add File: tests/_codex_guard_check.py", "+x"))),
            _shell_cell(5, "env | grep -E '^(CRYOET_ASSISTANT|CODEX_THREAD_ID)'"),
            _patch_cell(6, _patch("*** Add File: configs/lanes/_codex_guard_check.toml", "+x")),
        ]
    )
    env = {name: value for name, value in os.environ.items() if not name.startswith("CRYOET")}
    env.update({name: value for name, value in fields.items() if name != "cwd" and value})
    env["STANDIN_KEY"] = "stand-in"
    provider = (
        f'{{name="stand-in", base_url="http://127.0.0.1:{stand_in.port}/v1", '
        'env_key="STANDIN_KEY", wire_api="responses"}'
    )
    try:
        result = subprocess.run(
            [
                REAL_CODEX,
                "exec",
                *options,
                "-c",
                'model_provider="standin"',
                "-c",
                f"model_providers.standin={provider}",
                "check the guard",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=180,
        )
    finally:
        stand_in.server.shutdown()
        leftovers = [path for path in created if path.exists()]
        for path in leftovers:
            path.unlink()
    assert result.returncode == 0, result.stderr[-2000:]
    assert not leftovers
    # The instructions reached the model, as the launcher wrote them.
    texts = [
        part.get("text", "")
        for item in stand_in.requests[0]["input"]
        if item.get("type") == "message" and item.get("role") == "developer"
        for part in item["content"]
    ]
    assert any(_instructions(args) in text for text in texts)
    sbatch, input_shell, patch, shell_patch, variables, lane = stand_in.outputs()
    for output in (sbatch, input_shell, patch, shell_patch):
        assert "Blocked by assistant/guard.py" in output, output
    assert "denies Edit(src/**)" in patch and "denies Edit(tests/**)" in shell_patch
    # The journal's variables reach the commands, and Codex's own session id too.
    assert "CRYOET_ASSISTANT=cryoet-assistant; codex" in variables
    assert "CRYOET_ASSISTANT_MODEL=gpt-6.1-sol" in variables
    assert "CODEX_THREAD_ID=" in variables
    # A lane config is the person's to approve: the guard lets the patch through,
    # and Codex asks, which a headless run cannot (and refuses).
    assert "Blocked by assistant/guard.py" not in lane and "rejected" in lane
