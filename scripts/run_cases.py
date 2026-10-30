#!/usr/bin/env python3
"""Run the assistant's evaluation cases headless, each run in a sandbox of its
own, and lay out what every run did for a person to grade.

    python3 scripts/run_cases.py --list
    python3 scripts/run_cases.py E11 E17 --runs 3
    python3 scripts/run_cases.py --all --runs 3
    python3 scripts/run_cases.py E16 --held-out
    python3 scripts/run_cases.py E12 --dry-run      build the sandbox, show the command, run nothing

A case is a folder docs/agent/cases/<id>/ (see the README there): case.toml with
the request and the grading lines, tree/ with the files its evidence consists
of, and setup.sh when the evidence needs a command run first. A run works in a
copy of this repository's tracked files as they are in the working tree, with
tree/ laid over it, made a git repository of its own: the client keeps auto
memory per repository, and in this checkout a run would read this checkout's.
The copy leaves out the cases and this runner (EVALUATION), so that a search
of the copy cannot find a case's grading lines.
Stand-ins for sbatch, scancel, squeue and sacct come first
on its PATH: no job is ever submitted, a submission shows in slurm_calls.log,
and squeue lists the jobs the case says are queued.

The assistant is started as a person starts it, through the sandbox's own
scripts/assistant, with the case's request as the prompt (-p) and in
--permission-mode acceptEdits, so that it can write a config copy as a person
who approves edits lets it. Every deny and ask rule of assistant/settings.json
still holds, and an ask is refused, since nobody is there to answer it: a
submission is graded on the dry run and the question it shows
(docs/agent/eval_cases.md, "How the cases are run"). The run gets HOME, USER and
the like from this shell, so that the client finds its login, but nothing of a
session this is run from.

Everything goes into outputs/eval-runs/<UTC time>/: per run the transcript
(stream-json), the answer, the actions it took, the Slurm calls and a record
(case, model, client version, commit and the files changed since, the plugins
and the memory the client used, cost, turns); and grading.md, with the case's
Pass-if and Must-not lines as boxes per run, and a flag on a run that read a
memory kept before it or a CLAUDE.md from above its folder.

--held-out removes the case's own playbook entries from the sandbox's playbook,
to test reasoning from the evidence rather than applying what it was told.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "agent" / "cases"
RUNS = ROOT / "outputs" / "eval-runs"
# What a run inherits from this shell: enough for the client to find its login
# and the tools, and nothing of a Claude Code session this may be run from.
KEPT = ("HOME", "USER", "LOGNAME", "PATH", "SHELL", "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        "TMPDIR")
_CASE_HEADING = re.compile(r"^### (?P<id>E\d+)\. (?P<title>.+)$", re.MULTILINE)

# Stand-ins, first on a run's PATH. Each notes its call; squeue lists the case's
# jobs, cut or padded to the fields its -o format asks for.
_SLURM_CALL = '''#!/usr/bin/env python3
import os, sys
with open(os.environ["SLURM_CALLS"], "a") as log:
    shown = " ".join(f"{key}={os.environ[key]}" for key in
                     ("PROJECT", "CONFIG", "SERIES", "FROM", "UNTIL", "OVERWRITE")
                     if os.environ.get(key))
    called = " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]])
    log.write(called + (f"   [{shown}]" if shown else "") + "\\n")
'''
STUBS = {
    "sbatch": _SLURM_CALL + '''with open(os.environ["SLURM_CALLS"]) as log:
    print(4000000 + sum(1 for _ in log))
''',
    "scancel": _SLURM_CALL,
    # squeue fills the -o/--format fields from a case's line, id|name|state|where
    # [|time], with "?" for one the case does not give; -j keeps the jobs named.
    "squeue": '''#!/usr/bin/env python3
import os, re, sys
args = sys.argv[1:]
def option(short, long):
    for i, arg in enumerate(args):
        if arg in (short, long) and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(long + "="):
            return arg[len(long) + 1:]
        if arg.startswith(short) and len(arg) > 2 and not arg.startswith("--"):
            return arg[2:]
shape = option("-o", "--format") or "%i|%j|%T|%R"
jobs = (option("-j", "--jobs") or "").split(",")
HEADERS = {"i": "JOBID", "A": "JOBID", "j": "NAME", "T": "STATE", "t": "ST",
           "R": "NODELIST(REASON)", "N": "NODELIST", "r": "REASON", "M": "TIME", "u": "USER",
           "P": "PARTITION", "D": "NODES", "l": "TIME_LIMIT"}
SHORT = {"RUNNING": "R", "PENDING": "PD", "COMPLETING": "CG", "CONFIGURING": "CF"}
def shown(table):
    def one(match):
        text, width = str(table.get(match[3], "?")), int(match[2] or 0)
        text = text[:width] if width else text
        return text.rjust(width) if match[1] else text.ljust(width)
    return re.sub(r"%(\\.?)(\\d*)([A-Za-z])", one, shape)
path = os.environ.get("CASE_SQUEUE", "")
lines = open(path).read().splitlines() if path and os.path.exists(path) else []
if "-h" not in args and "--noheader" not in args:
    print(shown(HEADERS))
for line in lines:
    job, name, state, where, *rest = line.split("|") + ["?"] * 3
    if jobs != [""] and job not in jobs:
        continue
    running = state == "RUNNING"
    print(shown({"i": job, "A": job, "j": name, "T": state, "t": SHORT.get(state, state[:2]),
                 "R": where, "N": where if running else "", "r": "None" if running else where,
                 "M": rest[0], "u": os.environ.get("USER", "?")}))
''',
    "sacct": "#!/usr/bin/env python3\n",
    # A CTFFIND 5 that prints its start-up banner and stops, for a lane that names
    # CTFFIND 5 as "$CTFFIND5": the case sets CTFFIND5 = "{bin}/ctffind5".
    "ctffind5": """#!/usr/bin/env python3
print("        **   Welcome to Ctffind   **")
print("            Version : 5.0.2")
print("    Library Version : 2.0.0-stand-in")
print("Input image file name [input.mrc]                  :")
""",
}


@dataclass
class Case:
    id: str
    title: str
    request: str
    folder: Path
    pass_if: list[str]
    must_not: list[str]
    playbook: list[str] = field(default_factory=list)
    squeue: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


def load_case(folder: Path) -> Case:
    data = tomllib.loads((folder / "case.toml").read_text())
    return Case(
        id=data["id"], title=data["title"], request=data["request"].strip(), folder=folder,
        pass_if=list(data["pass_if"]), must_not=list(data["must_not"]),
        playbook=list(data.get("playbook", [])), squeue=list(data.get("squeue", [])),
        env={key: str(value) for key, value in data.get("env", {}).items()},
        note=data.get("note", "").strip(),
    )


def built_cases() -> dict[str, Case]:
    return {
        case.id: case
        for case in (load_case(folder.parent) for folder in sorted(CASES.glob("*/case.toml")))
    }


def listed_cases() -> dict[str, str]:
    """Every case eval_cases.md describes, by id, with its title."""
    text = (ROOT / "docs" / "agent" / "eval_cases.md").read_text()
    return {match["id"]: match["title"] for match in _CASE_HEADING.finditer(text)}


# What a copy of the repository leaves out when there is no git to list the
# tracked files, as in an exported tree.
_UNTRACKED = frozenset({".git", ".claude", "outputs", "__pycache__", ".pytest_cache"})
# The evaluation's own files, which no copy holds: the cases with their grading
# lines, this runner and the tests of both. On 30 October 2026 a run's grep for
# the misspelled key brought back E11's Pass-if line. scripts/collect_case.py
# stays: a collected case's setup.sh runs it in the copy, and it names no case.
EVALUATION = ("docs/agent/eval_cases.md", "docs/agent/cases/", "scripts/run_cases.py",
              "tests/test_run_cases.py", "tests/test_collect_case.py")


def tracked_files() -> list[str]:
    """The repository's tracked files; outside a git checkout, every file but
    outputs and caches."""
    try:
        listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True,
                                capture_output=True).stdout.decode().split("\0")
    except (OSError, subprocess.CalledProcessError):
        return sorted(
            str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
            if path.is_file() and not _UNTRACKED & set(path.relative_to(ROOT).parts)
        )
    return [name for name in listed if name]


def make_sandbox(sandbox: Path, case: Case, held_out: bool) -> None:
    """This repository's tracked files as they are now, but the evaluation's, with
    the case's tree on top, as a git repository of its own."""
    for name in tracked_files():
        source = ROOT / name
        if name.startswith(EVALUATION):
            continue
        if source.is_file():  # one deleted in the working tree is left out
            target = sandbox / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if (case.folder / "tree").is_dir():
        shutil.copytree(case.folder / "tree", sandbox, dirs_exist_ok=True)
    if held_out:
        playbook = sandbox / "docs" / "agent" / "playbook.md"
        playbook.write_text(without_entries(playbook.read_text(), case.playbook))
    own_repository(sandbox)


def own_repository(sandbox: Path) -> None:
    """One commit of the copy as the run finds it, the evidence that .gitignore
    leaves out aside, as in a clone on the cluster.

    In this checkout's work tree the client takes this checkout as the run's
    project, and it keeps auto memory per repository: on 30 October 2026 a run
    read the MEMORY.md of the sessions that develop this code. The scripts' own
    git calls would report this checkout too."""
    git = ["git", "-c", "user.name=eval", "-c", "user.email=eval@localhost",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
           "-c", "init.defaultBranch=main"]
    try:
        for args in (["init", "-q"], ["add", "-A"],
                     ["commit", "-q", "-m", "the case as the run finds it"]):
            subprocess.run([*git, *args], cwd=sandbox, check=True, capture_output=True)
    except OSError:  # no git here: the client then keys the run by its folder
        return


def without_entries(text: str, entries: list[str]) -> str:
    """The playbook without these entries (`### A-3. …` to the next heading)."""
    kept: list[str] = []
    skipping = False
    for line in text.splitlines(keepends=True):
        if line.startswith(("### ", "## ")):
            skipping = any(line.startswith(f"### {entry}.") for entry in entries)
        if not skipping:
            kept.append(line)
    return "".join(kept)


def run_env(case: Case, run_dir: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in KEPT if key in os.environ}
    env["PATH"] = os.pathsep.join([str(run_dir / "bin"), env.get("PATH", "")])
    env["SLURM_CALLS"] = str(run_dir / "slurm_calls.log")
    env["CASE_SQUEUE"] = str(run_dir / "squeue.txt")
    # {bin} in a case's value names the run's stand-ins, such as {bin}/ctffind5.
    env.update({key: value.replace("{bin}", str(run_dir / "bin"))
                for key, value in case.env.items()})
    return env


def make_stubs(run_dir: Path, case: Case) -> None:
    bin_dir = run_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, text in STUBS.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (run_dir / "squeue.txt").write_text("".join(f"{line}\n" for line in case.squeue))
    (run_dir / "slurm_calls.log").touch()


def command_for(sandbox: Path, case: Case, args: argparse.Namespace) -> list[str]:
    command = [
        str(sandbox / "scripts" / "assistant"), "-p", case.request,
        "--output-format", "stream-json", "--verbose",
        "--permission-mode", "acceptEdits", "--max-turns", str(args.max_turns),
    ]
    if args.model:
        command += ["--model", args.model]
    return command


def read_transcript(path: Path) -> dict[str, Any]:
    """The run's model and client, its answer, and every tool call with its outcome."""
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"),
                {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        message = event.get("message")  # a string in a permission_denied event
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") == "tool_use":
                calls[block["id"]] = {"tool": block.get("name"), "input": block.get("input", {})}
                order.append(block["id"])
            elif block.get("type") == "tool_result" and block.get("tool_use_id") in calls:
                output = block.get("content")
                if isinstance(output, list):
                    output = " ".join(str(part.get("text", "")) for part in output)
                calls[block["tool_use_id"]].update(
                    error=bool(block.get("is_error")), output=str(output or "")
                )
    return {"init": init, "result": result, "calls": [calls[key] for key in order]}


_SHOWN_LINES = 15


def describe_call(call: dict[str, Any], sandbox: Path | None = None) -> str:
    """A call and what came back: a Read's line count, one line inline, or the first
    lines as a block. Paths are given from the run's folder."""
    def short(text: str) -> str:
        return text.replace(f"{sandbox}/", "").replace(str(sandbox), ".") if sandbox else text

    given = call["input"]
    what = short(str(given.get("command") or given.get("file_path") or given.get("pattern")
                     or given.get("skill") or json.dumps(given)[:200]))
    head = f"{call['tool']}: `{what}`"
    lines = [short(line)[:200] for line in (call.get("output") or "").strip().splitlines()]
    if not lines:
        return head
    if call.get("error"):
        lines[0] = f"refused or failed: {lines[0]}"
    elif call["tool"] == "Read":
        return f"{head} → {len(lines)} lines"
    if len(lines) == 1:
        return f"{head} → {lines[0]}"
    shown = lines[:_SHOWN_LINES]
    if len(lines) > _SHOWN_LINES:
        shown.append(f"… {len(lines) - _SHOWN_LINES} lines more")
    return "\n".join([head, "", "   ```", *(f"   {line}" for line in shown), "   ```", ""])


def code_version() -> dict[str, Any]:
    """The commit checked out and the tracked files changed since; none outside git."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout
    try:
        return {
            "commit": git("rev-parse", "HEAD").strip(),
            "modified": [line[3:] for line in git("status", "--porcelain",
                                                  "--untracked-files=no").splitlines()],
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "modified": []}


def _shown(path: Path) -> str:
    """A path as the person sees it from the repository root."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def run_one(case: Case, number: int, held_out: bool, out: Path, args: argparse.Namespace,
            code: dict[str, Any]) -> dict[str, Any]:
    run_dir = out / f"{case.id}{'-held-out' if held_out else ''}-{number}"
    sandbox = run_dir / "sandbox"
    make_sandbox(sandbox, case, held_out)
    make_stubs(run_dir, case)
    env = run_env(case, run_dir)
    if (case.folder / "setup.sh").is_file():
        subprocess.run(["bash", str(case.folder / "setup.sh")], cwd=sandbox, env=env, check=True,
                       stdout=open(run_dir / "setup.log", "w"), stderr=subprocess.STDOUT)
    command = command_for(sandbox, case, args)
    record: dict[str, Any] = {
        "case": case.id, "title": case.title, "run": number, "held_out": held_out,
        "request": case.request, "command": command, "code": code,
        "user_settings_sha256": _sha256(Path.home() / ".claude" / "settings.json"),
        "user_claude_md": (Path.home() / ".claude" / "CLAUDE.md").is_file(),
        # The client reads every CLAUDE.md from the run's folder up, these included.
        "claude_md_above": [str(folder / name) for folder in sandbox.parents
                            for name in ("CLAUDE.md", "CLAUDE.local.md")
                            if (folder / name).is_file()],
    }
    if args.dry_run:
        print(f"# {_shown(run_dir)}\n  cd {sandbox}\n  " + " ".join(
            f"'{part}'" if " " in part else part for part in command))
        record["dry_run"] = True
        (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        return record

    # The client's projects, with their memories, before the run: the run's own is new.
    projects = Path.home() / ".claude" / "projects"
    before = {path.name for path in projects.iterdir()} if projects.is_dir() else set()
    started = datetime.now(UTC)
    with open(run_dir / "transcript.jsonl", "w") as stdout, \
            open(run_dir / "stderr.log", "w") as stderr:
        try:
            finished = subprocess.run(command, cwd=sandbox, env=env, stdout=stdout, stderr=stderr,
                                      timeout=args.timeout)
            exit_code: int | str = finished.returncode
        except subprocess.TimeoutExpired:
            exit_code = f"stopped after {args.timeout} s"
    seen = read_transcript(run_dir / "transcript.jsonl")
    result, init = seen["result"], seen["init"]
    slurm = (run_dir / "slurm_calls.log").read_text().splitlines()
    memory = (init.get("memory_paths") or {}).get("auto")
    record.update(
        started=started.isoformat(timespec="seconds"),
        seconds=round((datetime.now(UTC) - started).total_seconds()),
        exit_code=exit_code, model=init.get("model"),
        client=init.get("claude_code_version"), permission_mode=init.get("permissionMode"),
        plugins=[plugin.get("source") or plugin.get("name") for plugin in init.get("plugins", [])
                 if plugin.get("path") != "builtin"],
        memory=memory, memory_from_before=bool(memory) and Path(memory).parent.name in before,
        cost_usd=result.get("total_cost_usd"), turns=result.get("num_turns"),
        is_error=result.get("is_error"), permission_denials=result.get("permission_denials"),
        slurm_calls=slurm, calls=len(seen["calls"]),
        untrusted="has not been trusted" in (run_dir / "stderr.log").read_text(errors="replace"),
    )
    (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    (run_dir / "answer.md").write_text(str(result.get("result") or "(no answer)") + "\n")
    lines = [f"# {case.id} run {number}: what it did", ""]
    lines += [f"{index}. {describe_call(call, sandbox)}"
              for index, call in enumerate(seen["calls"], 1)]
    if slurm:
        lines += ["", "**Slurm was called** (the stand-ins noted it):", ""]
        lines += [f"- `{line}`" for line in slurm]
    (run_dir / "actions.md").write_text("\n".join(lines) + "\n")
    print(f"{case.id} run {number}: exit {exit_code}, {record['turns']} turns, "
          f"{len(seen['calls'])} tool calls, {len(slurm)} Slurm calls", file=sys.stderr)
    return record


def grading_sheet(cases: list[Case], records: list[dict[str, Any]], out: Path,
                  code: dict[str, Any]) -> str:
    first = next((record for record in records if record.get("model")), {})
    lines = [
        f"# Evaluation run {out.name}", "",
        f"Model {first.get('model', '?')}, Claude Code {first.get('client', '?')}, "
        f"commit {(code['commit'] or 'unknown')[:12]}"
        + (f" with {len(code['modified'])} tracked files changed" if code["modified"] else "")
        + ".", "",
        "Tick a Pass-if line when it holds, and a Must-not line when it did not happen. A run "
        "passes when every box of it is ticked; report each case as passes out of its runs "
        "(docs/agent/eval_cases.md, \"How the cases are run\").", "",
    ]
    for case in cases:
        mine = [record for record in records if record["case"] == case.id]
        lines += [f"## {case.id}. {case.title}", "", f"> {case.request}", ""]
        if case.note:
            lines += [f"*{case.note}*", ""]
        for record in mine:
            name = f"{case.id}{'-held-out' if record['held_out'] else ''}-{record['run']}"
            flags = []
            if record.get("slurm_calls"):
                count = len(record["slurm_calls"])
                flags.append(f"Slurm called {'once' if count == 1 else f'{count} times'}")
            if record.get("permission_denials"):
                flags.append(f"{len(record['permission_denials'])} permission denials")
            if record.get("exit_code") not in (0, None):
                flags.append(f"exit {record['exit_code']}")
            if record.get("untrusted"):
                flags.append("the client did not trust the folder; see stderr.log")
            if record.get("memory_from_before"):
                flags.append("it read a memory kept before the run; see record.json")
            if record.get("claude_md_above"):
                flags.append("CLAUDE.md above the run's folder; see record.json")
            lines += [
                f"### Run {record['run']}{' (held out)' if record['held_out'] else ''}: "
                f"[answer]({name}/answer.md), [actions]({name}/actions.md)"
                + (f" — {', '.join(flags)}" if flags else ""), "",
            ]
            lines += [f"- [ ] {line}" for line in case.pass_if]
            lines += [f"- [ ] did not: {line}" for line in case.must_not]
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cases", nargs="*", help="case ids, such as E11")
    parser.add_argument("--all", action="store_true", help="every case that has a folder")
    parser.add_argument("--list", action="store_true", help="the cases, built or not")
    parser.add_argument("--runs", type=int, default=3, help="runs per case (3)")
    parser.add_argument("--held-out", action="store_true",
                        help="also run each case with its own playbook entries removed")
    parser.add_argument("--model", default=None, help="a full model ID, passed to the launcher")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=1200, help="seconds a run may take")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the sandboxes and show the commands; run nothing")
    args = parser.parse_args(argv)

    built = built_cases()
    if args.list:
        for case_id, title in listed_cases().items():
            print(f"{case_id}  {'built   ' if case_id in built else 'no folder'}  {title}")
        return 0
    wanted = sorted(built) if args.all else args.cases
    unknown = [case_id for case_id in wanted if case_id not in built]
    if unknown or not wanted:
        raise SystemExit(f"no case folder for {', '.join(unknown)}" if unknown
                         else "name the cases, or give --all or --list")
    if shutil.which("claude") is None and not args.dry_run:
        raise SystemExit("Claude Code (claude) is not on PATH")

    out = RUNS / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    code = code_version()
    records = []
    sheet = out / "grading.md"
    for case_id in wanted:
        case = built[case_id]
        for held_out in (False, True) if args.held_out and case.playbook else (False,):
            for number in range(1, args.runs + 1):
                records.append(run_one(case, number, held_out, out, args, code))
                # After every run, so that a stopped batch leaves the runs it made.
                (out / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
                sheet.write_text(grading_sheet([built[case_id] for case_id in wanted], records,
                                               out, code))
    print(f"{len(records)} runs; grade them in {_shown(sheet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
