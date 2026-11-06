"""The terminal assistant's launcher, its client rules and its skills
(scripts/assistant and assistant/): what it starts Claude Code with, what the
guard on shell commands blocks, and that the skills name only what exists."""

import json
import os
import re
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
    return {"root": root.resolve(), "snapshot": snapshot.resolve(), "link": link,
            "tmp": tmp_path.resolve()}


def library_env(library: dict) -> dict[str, str]:
    return {"CRYOET_LIBRARY_ROOT": str(library["root"]),
            "CRYOET_LIBRARY_SNAPSHOT": str(library["snapshot"])}


def _launch(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args], cwd=cwd, capture_output=True, text=True,
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
    for args in (["--permission-mode", "acceptEdits"], ["--permission-mode=acceptEdits"],
                 ["-p", "hello", "--permission-mode", "bypassPermissions"]):
        result = _launch(*args, cwd=ROOT)
        assert result.returncode == 2 and "scripts/assistant:" in result.stderr
        assert not claude.exists()


def test_without_claude_the_launcher_says_so(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(
        ["/bin/bash", str(LAUNCHER)], cwd=ROOT, capture_output=True, text=True, env=env
    )
    assert result.returncode == 1 and "Claude Code (claude) is not on PATH" in result.stderr


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
    assert prompt == "\n\n".join([
        (ASSISTANT / "instructions.md").read_text().rstrip("\n"),
        (ASSISTANT / "library.md").read_text().rstrip("\n"),
        f"This session's library is {library['snapshot']} (snapshot site-2026-11-05).",
    ])


def test_none_and_a_folder_without_a_snapshot_start_without_the_library(
        claude, library, monkeypatch):
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
        text.replace(sites, f"LIBRARY_SITES=({library['root']})"))
    (repo / "assistant").symlink_to(ASSISTANT)
    return repo / "scripts" / "assistant"


def test_a_person_gets_the_sites_library_and_a_headless_run_only_a_named_one(
        claude, library, site_launcher, monkeypatch):
    monkeypatch.delenv("CRYOET_LIBRARY")

    def launch(*args: str) -> list[str]:
        result = subprocess.run(["bash", str(site_launcher), *args], cwd=ROOT,
                                capture_output=True, text=True, env=dict(os.environ))
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
    guard = re.search(r"^LIBRARY_SITES = \((.*)\)$", (ASSISTANT / "guard.py").read_text(),
                      re.MULTILINE)
    sites = launcher.group(1).split()
    assert sites and sites == re.findall(r'"([^"]+)"', guard.group(1))
    for site in sites:
        assert f"Edit(/{site}/**)" in _rules()["deny"]


def _rules() -> dict[str, list[str]]:
    return json.loads((ASSISTANT / "settings.json").read_text())["permissions"]


def test_the_client_rules_hold_the_gates():
    rules = _rules()
    for rule in (
        "Bash(sbatch *)", "Bash(srun *)", "Bash(salloc *)", "Bash(git *)",
        "Bash(python3 scripts/run_series.py * --execute*)",
        "Edit(**/journal.jsonl)", "Edit(**/*.sbatch)", "Edit(scripts/**)", "Edit(src/**)",
        "Edit(tests/**)", "Edit(docs/**)", "Edit(assistant/**)", "Edit(.claude/**)",
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
    assert {"Bash(python3 scripts/build_library.py *)",
            "Bash(python scripts/build_library.py *)"} <= set(deny)


def _guard(command: str = "", tool: str = "Bash", *, cwd: Path | None = None,
           env: dict[str, str] | None = None,
           tool_input: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    call: dict = {"tool_name": tool,
                  "tool_input": tool_input if tool_input is not None else {"command": command}}
    if cwd is not None:
        call["cwd"] = str(cwd)
    environ = {name: value for name, value in os.environ.items()
               if not name.startswith("CRYOET_LIBRARY")}
    environ.update(env or {})
    return subprocess.run(
        [sys.executable, str(ASSISTANT / "guard.py")], input=json.dumps(call),
        capture_output=True, text=True, env=environ,
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
    ],
)
def test_the_guard_lets_the_rest_through(command):
    assert _guard(command).returncode == 0


def test_the_guard_does_not_judge_what_only_reads():
    assert _guard("sbatch x", tool="Read").returncode == 0


@pytest.mark.parametrize("command", [
    "python3 scripts/build_library.py --site mdc --dry-run",
    "/opt/env/bin/python3 -I /fast/x/scripts/build_library.py --site mdc",
    "env OVERWRITE=1 python3 scripts/build_library.py --site mdc",
    "bash -c 'python3 scripts/build_library.py --site mdc'",
    "./scripts/build_library.py --site mdc",
])
def test_the_guard_blocks_the_library_builder(command):
    result = _guard(command)
    assert result.returncode == 2 and "build_library.py writes" in result.stderr


@pytest.mark.parametrize("command", [
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
])
def test_the_guard_blocks_every_write_to_the_library(library, command):
    result = _guard(command.format(**library), env=library_env(library))
    assert result.returncode == 2, command
    assert "reference library, which is read-only" in result.stderr


@pytest.mark.parametrize("command", [
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
])
def test_the_guard_lets_the_library_be_read(library, command):
    assert _guard(command.format(**library), env=library_env(library)).returncode == 0, command


def test_inside_the_library_every_command_only_reads(library):
    env = library_env(library)
    assert _guard("rm text/imod/newstack.md", cwd=library["snapshot"], env=env).returncode == 2
    assert _guard("grep -rn newstack text", cwd=library["snapshot"], env=env).returncode == 0


@pytest.mark.parametrize(("tool", "key"), [
    ("Edit", "file_path"), ("Write", "file_path"), ("MultiEdit", "file_path"),
    ("NotebookEdit", "notebook_path"),
])
def test_no_tool_edits_a_file_in_the_library(library, tool, key):
    env = library_env(library)
    for path in (library["snapshot"] / "text" / "imod" / "newstack.md",
                 library["root"] / "current" / "site.md", library["link"] / "new.md"):
        result = _guard(tool=tool, tool_input={key: str(path)}, env=env)
        assert result.returncode == 2 and "no tool edits it" in result.stderr, path
    assert _guard(tool=tool, tool_input={key: "new.md"}, cwd=library["snapshot"],
                  env=env).returncode == 2
    assert _guard(tool=tool, tool_input={key: str(library["tmp"] / "notes.md")},
                  env=env).returncode == 0
    # A site's folder is guarded when the session names no library.
    assert _guard(tool=tool, tool_input={
        key: "/fast/AG_Kudryashev/cryoet-library/current/x.md"}).returncode == 2


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
        "emdb-template", "empiar-download", "lane-config", "project-status"]


# Repository paths the assistant's files name: scripts, docs, configs, templates.
_PATH = re.compile(r"(?<![\w/.<])((?:scripts|docs|configs|assistant)/[\w./-]*[\w/])")


@pytest.mark.parametrize(
    "document",
    [*SKILLS, ASSISTANT / "instructions.md", ASSISTANT / "library.md", ASSISTANT / "README.md"],
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
