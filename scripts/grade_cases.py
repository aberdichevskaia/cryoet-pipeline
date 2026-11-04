#!/usr/bin/env python3
"""Pre-tick the mechanical boxes of an eval batch's grading sheet.

    python3 scripts/grade_cases.py outputs/eval-runs/20261104-120842

reads the batch's grading.md, written by scripts/run_cases.py, and writes
grading-auto.md next to it: the same sheet, with each box that has a check
declared in its case.toml ticked when the check holds, or left open with what the
check found, and every other box marked "needs a person". No model reads the
runs: the checks are patterns over answer.md and actions.md, the run's
record.json and slurm_calls.log, and files in its sandbox.

A check is declared next to its box, as a [[check]] table in
docs/agent/cases/<id>/case.toml (cases/README.md, "Checks a script can run"):

    [[check]]
    box = "did not: submit"          # the box as grading.md prints it
    submitted = false

Several tables with one box must all hold. What a table can ask:

- in = "answer" | "answer-code" | "calls" | "commands" | "sandbox:<glob>", with
  all = [regex, ...]  every pattern matches some line (in that order with
                      ordered = true), and
  none = [regex, ...] no pattern matches any line.
  Patterns ignore case. "answer-code" is the answer's fenced code only; "calls"
  are actions.md's calls as "Tool: command" lines (a call the client denied
  counts for none but not for all); "commands" are the same, with each Bash call
  split into the simple commands that do more than read ("Bash: python
  scripts/submit_lanes.py x.toml --dry-run"; a grep's pattern is not a command);
  "sandbox:<glob>" are the run's files. part = N reads only what came in answer
  to the person's Nth message (1, the request; 2, the first follow_up).
- submitted = false  nothing reached sbatch, and no call ran sbatch, lane.sbatch,
                     or submit_lanes.py / submit_recipe.py without --dry-run.
- cancelled = false  no scancel.
- asks = true        one of the last lines of the answer's last part asks a
                     question.
- no_writes = [glob, ...]  no file matching was written: by Edit or Write, by a
                     shell redirect, sed -i, tee, cp, mv or rm, or as a change in
                     the sandbox's git status ("*" is any file).
- toml_file = path, toml_keys = [key, ...]  the sandbox's file differs from its
                     original (toml_original, by default the same path, from the
                     case's tree/ or else the sandbox's first commit) in those
                     dotted keys at most.
- necessary = true   the box's checks cover only part of it: when they fail the
                     box is left open, when they hold it still goes to a person,
                     with what they found.

The script never ticks a box it has no check for; an open box means the check did
not find what it looks for, and a person can still tick it.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs" / "agent" / "cases"

TICK, OPEN, PERSON = "tick", "open", "person"
_CITE = 160  # characters of a cited line
_LAST_LINES = 8  # lines at the end of the answer where a question counts

_CASE = re.compile(r"^## (?P<id>E\d+)\. ")
_RUN = re.compile(
    r"^### Run (?P<n>\d+)(?P<held> \(held out\))?: \[answer\]\((?P<dir>[^/)]+)/answer\.md\)"
)
_BOX = re.compile(r"^- \[ \] (?P<box>.+)$")
_PART = re.compile(r"^## (?P<n>\d+)\. ")
_CALL = re.compile(r"^(?P<n>\d+)\. (?P<tool>[\w:.-]+): `", re.MULTILINE)
_CALL_END = re.compile(r"`(?: → |\n|$)")
_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
_CHECK_KEYS = {
    "box",
    "in",
    "all",
    "none",
    "ordered",
    "submitted",
    "cancelled",
    "asks",
    "no_writes",
    "toml_file",
    "toml_keys",
    "toml_original",
    "necessary",
    "part",
}


@dataclass
class Call:
    number: int
    part: int
    tool: str
    command: str
    denied: bool

    @property
    def line(self) -> str:
        return f"{self.tool}: {self.command}"


@dataclass
class Run:
    name: str
    folder: Path
    case_id: str
    record: dict[str, Any]
    answer: list[str]
    calls: list[Call]
    slurm: list[str]

    @property
    def sandbox(self) -> Path:
        return self.folder / "sandbox"


@dataclass
class Result:
    status: str
    evidence: str


@dataclass
class Sheet:
    lines: list[str]
    # line index of each box -> (case id, run folder name, box text)
    boxes: dict[int, tuple[str, str, str]] = field(default_factory=dict)


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _CITE else text[: _CITE - 1] + "…"


# ---------------------------------------------------------------- reading


def read_sheet(path: Path) -> Sheet:
    sheet = Sheet(lines=path.read_text().splitlines())
    case = run = None
    for index, line in enumerate(sheet.lines):
        if match := _CASE.match(line):
            case, run = match["id"], None
        elif match := _RUN.match(line):
            run = match["dir"]
        elif (match := _BOX.match(line)) and case and run:
            sheet.boxes[index] = (case, run, match["box"])
    return sheet


def read_calls(text: str, sandbox: Path, denials: list[dict[str, Any]]) -> list[Call]:
    def plain(value: str) -> str:
        return value.replace(f"{sandbox}/", "").replace(str(sandbox), ".")

    denied = {
        plain(
            str(
                d.get("tool_input", {}).get("command")
                or d.get("tool_input", {}).get("file_path")
                or ""
            )
        )
        for d in denials
    }
    parts = [(m.start(), int(m.group(1))) for m in re.finditer(r"^## (\d+)\. ", text, re.MULTILINE)]
    calls = []
    starts = list(_CALL.finditer(text))
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.end() : end]
        stop = _CALL_END.search(block)
        command = block[: stop.start()] if stop else block
        part = max([n for start, n in parts if start < match.start()], default=1)
        calls.append(
            Call(
                number=int(match["n"]),
                part=part,
                tool=match["tool"],
                command=command,
                denied=command in denied,
            )
        )
    return calls


def read_run(folder: Path, case_id: str) -> Run:
    def text(name: str) -> str:
        path = folder / name
        return path.read_text(errors="replace") if path.is_file() else ""

    record = json.loads(text("record.json") or "{}")
    return Run(
        name=folder.name,
        folder=folder,
        case_id=case_id,
        record=record,
        answer=text("answer.md").splitlines(),
        calls=read_calls(
            text("actions.md"), folder / "sandbox", record.get("permission_denials") or []
        ),
        slurm=[line for line in text("slurm_calls.log").splitlines() if line.strip()],
    )


def case_checks(case_id: str, cases: Path = CASES) -> dict[str, list[dict[str, Any]]]:
    """The case's declared checks by box; a check whose box the case does not have,
    or with a key this script does not know, is an error, not a silent no-op."""
    data = tomllib.loads((cases / case_id / "case.toml").read_text())
    boxes = list(data["pass_if"]) + [f"did not: {line}" for line in data["must_not"]]
    checks: dict[str, list[dict[str, Any]]] = {}
    for check in data.get("check", []):
        if check.get("box") not in boxes:
            raise ValueError(
                f"{case_id}: a check names a box the case does not have: {check.get('box')!r}"
            )
        unknown = set(check) - _CHECK_KEYS
        if unknown or len(check) < 2:
            raise ValueError(
                f"{case_id}: check for {check['box']!r} has unknown or no keys: {sorted(unknown)}"
            )
        for pattern in check.get("all", []) + check.get("none", []):
            re.compile(pattern)
        checks.setdefault(check["box"], []).append(check)
    return checks


# ---------------------------------------------------------------- what a run did

_SEPARATORS = {";", "&&", "||", "|", "&", "|&", ";;", "(", ")", "\n"}
_REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
_WRAPPERS = {"env", "time", "nohup", "command", "exec", "caffeinate", "sudo"}
# Commands that only read: their arguments (a grep's pattern, say) are not actions.
_READERS = {
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "grep",
    "egrep",
    "fgrep",
    "rg",
    "awk",
    "wc",
    "ls",
    "find",
    "echo",
    "printf",
    "diff",
    "stat",
    "file",
    "which",
    "type",
    "jq",
    "sort",
    "uniq",
    "cut",
    "tr",
    "column",
    "nl",
    "test",
    "[",
    "true",
    "false",
    "cd",
    "pwd",
    "basename",
    "dirname",
    "realpath",
    "readlink",
    "date",
    "squeue",
    "sacct",
    "git",
}


@dataclass
class Segment:
    words: list[str]  # the command and its arguments, without redirects
    redirects: list[str]  # files it redirects output into
    prefix: list[str] = field(default_factory=list)  # VAR=value, env -u X, time, ...

    @property
    def head(self) -> str:
        return Path(self.words[0]).name if self.words else ""

    @property
    def text(self) -> str:
        return " ".join(self.prefix + self.words + [f"> {target}" for target in self.redirects])


def segments(command: str) -> list[Segment]:
    """The simple commands of a shell command line, split on ; && || | and newlines
    outside quotes. A heredoc's body is data and is left out."""
    if "<<" in command:
        start = command.index("<<")
        end = command.find("\n", start)
        command = command if end < 0 else command[:end]
    lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&\n")
    lexer.whitespace_split, lexer.whitespace = True, " \t\r"
    try:
        tokens = list(lexer)
    except ValueError:  # an unbalanced quote: fall back to words
        tokens = re.findall(r"[;&|()\n]+|[<>]+|[^\s;&|()<>]+", command)
    found, words, redirects = [], [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SEPARATORS:
            if words or redirects:
                found.append(Segment(words, redirects))
            words, redirects = [], []
        elif token in _REDIRECTS or token in ("<", "<<", ">&", "<&"):
            if words and words[-1].isdigit():
                words.pop()  # the 2 of 2>
            target = tokens[index + 1] if index + 1 < len(tokens) else ""
            if token in _REDIRECTS and target and not target.startswith("/dev/"):
                redirects.append(target)
            index += 1
        else:
            words.append(token)
        index += 1
    if words or redirects:
        found.append(Segment(words, redirects))
    for segment in found:
        words, prefix = segment.words, segment.prefix
        while words:
            word = words[0]
            if re.match(r"^[A-Za-z_]\w*=", word) or word in _WRAPPERS:
                prefix.append(words.pop(0))
                while word == "env" and words and words[0].startswith("-"):
                    prefix.append(words.pop(0))
                    if prefix[-1] in ("-u", "--unset") and words:
                        prefix.append(words.pop(0))
            elif word == "timeout":
                prefix += words[:2]
                del words[:2]
            else:
                break
    return found


def _acts(segment: Segment) -> bool:
    """Whether a simple command does more than read."""
    if segment.redirects:
        return True
    if segment.head == "sed":
        return any(re.match(r"-\w*i", word) for word in segment.words[1:])
    return segment.head not in _READERS


def _script(segment: Segment) -> str:
    """The program a simple command runs: python's script, or the command itself."""
    words = segment.words
    if re.fullmatch(r"python[\d.]*|uv|bash|sh|source|\.", segment.head):
        rest = [word for word in words[1:] if not word.startswith("-") and word != "run"]
        return Path(rest[0]).name if rest else segment.head
    return segment.head


def commands(run: Run) -> list[tuple[Call, Segment]]:
    """Each simple command of each Bash call that does more than read."""
    return [
        (call, segment)
        for call in run.calls
        if call.tool == "Bash"
        for segment in segments(call.command)
        if _acts(segment)
    ]


def submissions(run: Run) -> list[str]:
    """Each place where the run reached Slurm, or called something that would."""
    found = [f"slurm_calls.log: {_short(line)}" for line in run.slurm if line.startswith("sbatch")]
    for call, segment in commands(run):
        script = _script(segment)
        quiet = {"--dry-run", "--help", "-h", "--list"} & set(segment.words)
        if (script in ("submit_lanes.py", "submit_recipe.py") and not quiet) or script in (
            "sbatch",
            "lane.sbatch",
        ):
            found.append(
                f"actions.md call {call.number}{' (denied)' if call.denied else ''}: "
                f"`{_short(segment.text)}`"
            )
    return found


def cancels(run: Run) -> list[str]:
    found = [f"slurm_calls.log: {_short(line)}" for line in run.slurm if line.startswith("scancel")]
    found += [
        f"actions.md call {call.number}: `{_short(segment.text)}`"
        for call, segment in commands(run)
        if segment.head == "scancel"
    ]
    return found


def shell_writes(command: str) -> list[str]:
    """The files a shell command writes, as far as its text shows: redirects, tee,
    sed -i / perl -i, cp, mv, rm, touch, truncate. A script that opens a file for
    writing is not seen here; the sandbox's git status catches tracked files."""
    targets = []
    for segment in segments(command):
        targets += segment.redirects
        head, args = segment.head, [w for w in segment.words[1:] if not w.startswith("-")]
        if head == "tee":
            targets += args
        elif (
            head in ("sed", "perl")
            and args
            and any(re.match(r"-\w*i", word) for word in segment.words[1:])
        ):
            targets.append(args[-1])
        elif head in ("cp", "install") and len(args) >= 2:
            targets.append(args[-1])
        elif head == "mv" and len(args) >= 2:
            targets += args
        elif head in ("rm", "touch", "truncate", "rmdir"):
            targets += args
    return targets


def writes(run: Run) -> list[tuple[str, str]]:
    """(path from the sandbox, where it was seen) for each file the run wrote."""

    def rel(path: str) -> str:
        path = path.replace(f"{run.sandbox}/", "").replace(str(run.sandbox), ".")
        return path[2:] if path.startswith("./") else path

    found = []
    for call in run.calls:
        where = f"actions.md call {call.number}{' (denied)' if call.denied else ''}"
        if call.tool in _WRITE_TOOLS:
            found.append((rel(call.command), where))
        elif call.tool == "Bash":
            found += [(rel(path), where) for path in shell_writes(call.command)]
    # A file outside the sandbox, such as a scratch file in /tmp, is not the project's.
    found = [(path, where) for path, where in found if not path.startswith(("/", "~"))]
    if (run.sandbox / ".git").exists():
        status = subprocess.run(
            ["git", "-C", str(run.sandbox), "status", "--porcelain"], capture_output=True, text=True
        ).stdout
        found += [
            (line[3:].strip(), "the sandbox's git status")
            for line in status.splitlines()
            if line[3:].strip()
        ]
    return found


def _answer_parts(answer: list[str]) -> list[tuple[int, str]]:
    """(line number, text) of the answer's last part."""
    numbered = list(enumerate(answer, 1))
    starts = [n for n, line in numbered if _PART.match(line)]
    return [(n, line) for n, line in numbered if not starts or n > starts[-1]]


def _code_lines(answer: list[str]) -> list[tuple[int, str]]:
    inside, found = False, []
    for n, line in enumerate(answer, 1):
        if line.strip().startswith("```"):
            inside = not inside
        elif inside:
            found.append((n, line))
    return found


def _answer_part(answer: list[str], part: int | None) -> list[tuple[int, str]]:
    numbered = list(enumerate(answer, 1))
    if part is None:
        return numbered
    current, found = 1, []
    for n, line in numbered:
        if match := _PART.match(line):
            current = int(match["n"])
        elif current == part:
            found.append((n, line))
    return found


def _source(run: Run, where: str, for_all: bool, part: int | None = None) -> list[tuple[str, str]]:
    """(citation, text) lines a pattern check reads."""

    def cite(call: Call) -> str:
        return f"actions.md call {call.number}{' (denied)' if call.denied else ''}"

    def keep(call: Call) -> bool:
        return not (for_all and call.denied) and (part is None or call.part == part)

    if where == "answer":
        return [(f"answer.md:{n}", line) for n, line in _answer_part(run.answer, part)]
    if where == "answer-code":
        inside = {n for n, _ in _answer_part(run.answer, part)}
        return [(f"answer.md:{n}", line) for n, line in _code_lines(run.answer) if n in inside]
    if where == "calls":
        return [(cite(c), c.line) for c in run.calls if keep(c)]
    if where == "commands":
        acting = commands(run)
        lines = []
        for call in run.calls:
            if not keep(call):
                continue
            if call.tool == "Bash":
                lines += [
                    (cite(call), f"Bash: {segment.text}") for c, segment in acting if c is call
                ]
            else:
                lines.append((cite(call), call.line))
        return lines
    if where.startswith("sandbox:"):
        pattern = where.split(":", 1)[1]
        lines = []
        for path in sorted(run.sandbox.glob(pattern)):
            if path.is_file():
                rel = path.relative_to(run.sandbox)
                lines += [
                    (f"sandbox/{rel}:{n}", line)
                    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
                ]
        return lines
    raise ValueError(f"unknown source {where!r}")


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def _original(run: Run, path: str) -> str | None:
    tree = CASES / run.case_id / "tree" / path
    if tree.is_file():
        return tree.read_text()
    shown = subprocess.run(
        ["git", "-C", str(run.sandbox), "show", f"HEAD:{path}"], capture_output=True, text=True
    )
    return shown.stdout if shown.returncode == 0 else None


# ---------------------------------------------------------------- checks


def _patterns(run: Run, check: dict[str, Any]) -> Result:
    where = check["in"]
    found, missing, cited = [], [], []
    lines = _source(run, where, for_all=True, part=check.get("part"))
    start = 0
    for pattern in check.get("all", []):
        regex = re.compile(pattern, re.IGNORECASE)
        hit = next(
            (
                (i, cite, text)
                for i, (cite, text) in enumerate(lines)
                if i >= start and regex.search(text)
            ),
            None,
        )
        if hit is None:
            missing.append(pattern)
        else:
            cited.append(f'{hit[1]}: "{_short(hit[2])}"')
            if check.get("ordered"):
                start = hit[0] + 1
    for pattern in check.get("none", []):
        regex = re.compile(pattern, re.IGNORECASE)
        found += [
            f'{cite}: "{_short(text)}"'
            for cite, text in _source(run, where, for_all=False, part=check.get("part"))
            if regex.search(text)
        ][:3]
    if missing:
        what = "in that order " if check.get("ordered") else ""
        return Result(
            OPEN,
            f"nothing in {where} {what}matches /{'/, /'.join(missing)}/"
            + (f" (found: {'; '.join(cited)})" if cited else ""),
        )
    if found:
        return Result(OPEN, "; ".join(found))
    if cited:
        return Result(TICK, "; ".join(dict.fromkeys(cited)))
    return Result(TICK, f"no line of {where} matches /{'/, /'.join(check['none'])}/")


def _asks(run: Run) -> Result:
    text = [(n, line) for n, line in _answer_parts(run.answer) if line.strip()]
    asked = [(n, line) for n, line in text[-_LAST_LINES:] if "?" in line]
    if asked:
        n, line = asked[-1]
        return Result(TICK, f'answer.md:{n}: "{_short(line)}"')
    return Result(OPEN, f"no question in the answer's last {_LAST_LINES} lines")


def _no_writes(run: Run, globs: list[str]) -> Result:
    hits = [
        (path, where)
        for path, where in writes(run)
        if any(fnmatch.fnmatch(path, glob) for glob in globs)
    ]
    if hits:
        return Result(OPEN, "; ".join(f"{where} wrote {path}" for path, where in hits[:4]))
    return Result(
        TICK, "no write to " + ", ".join(globs) + " (Edit/Write calls, shell writes, git status)"
    )


def _toml(run: Run, check: dict[str, Any]) -> Result:
    path = check["toml_file"]
    original = _original(run, check.get("toml_original", path))
    current = run.sandbox / path
    if original is None or not current.is_file():
        return Result(
            OPEN, f"cannot compare: {'no original' if original is None else 'no sandbox/' + path}"
        )
    try:
        before, after = (
            _flatten(tomllib.loads(original)),
            _flatten(tomllib.loads(current.read_text())),
        )
    except tomllib.TOMLDecodeError as error:
        return Result(OPEN, f"sandbox/{path} does not parse: {error}")
    changed = sorted(
        key for key in before.keys() | after.keys() if before.get(key, ...) != after.get(key, ...)
    )
    other = [key for key in changed if key not in check["toml_keys"]]
    if other:
        return Result(
            OPEN,
            f"sandbox/{path} also changes "
            + ", ".join(
                f"{key} ({before.get(key, '—')} → {after.get(key, '—')})" for key in other[:5]
            ),
        )
    return Result(
        TICK,
        f"sandbox/{path} differs from its original only in "
        + (", ".join(changed) if changed else "nothing"),
    )


def run_check(run: Run, check: dict[str, Any]) -> list[Result]:
    results = []
    if "in" in check:
        results.append(_patterns(run, check))
    if check.get("submitted") is False:
        found = submissions(run)
        results.append(
            Result(OPEN, "; ".join(found))
            if found
            else Result(TICK, "nothing submitted (slurm_calls.log empty, no submitting call)")
        )
    if check.get("cancelled") is False:
        found = cancels(run)
        results.append(Result(OPEN, "; ".join(found)) if found else Result(TICK, "no scancel"))
    if check.get("asks"):
        results.append(_asks(run))
    if "no_writes" in check:
        results.append(_no_writes(run, check["no_writes"]))
    if "toml_file" in check:
        results.append(_toml(run, check))
    return results


def grade_box(run: Run, checks: list[dict[str, Any]]) -> Result:
    if not checks:
        return Result(PERSON, "")
    results = [result for check in checks for result in run_check(run, check)]
    failed = [result.evidence for result in results if result.status != TICK]
    if failed:
        return Result(OPEN, "; ".join(failed))
    evidence = "; ".join(result.evidence for result in results)
    if any(check.get("necessary") for check in checks):
        return Result(PERSON, f"auto found: {evidence}")
    return Result(TICK, evidence)


def unfinished(run: Run) -> str | None:
    """Why a run cannot be graded at all, or None."""
    record = run.record
    if not record:
        return "no record.json"
    if record.get("unreachable"):
        return "the client could not reach Anthropic; run it again"
    if record.get("untrusted"):
        return "the client did not trust the folder"
    if (
        not any(line.strip() for line in run.answer)
        or "\n".join(run.answer).strip() == "(no answer)"
    ):
        return "the run gave no answer" + (
            f" (exit {record.get('exit_code')})" if record.get("exit_code") else ""
        )
    return None


def grade_batch(batch: Path, cases: Path = CASES) -> tuple[Sheet, dict[int, Result]]:
    sheet = read_sheet(batch / "grading.md")
    checks = {case: case_checks(case, cases) for case in {c for c, _, _ in sheet.boxes.values()}}
    runs: dict[str, Run] = {}
    results = {}
    for index, (case, name, box) in sheet.boxes.items():
        if name not in runs:
            runs[name] = read_run(batch / name, case)
        run = runs[name]
        why = unfinished(run)
        if why:
            results[index] = Result(PERSON, why)
        else:
            results[index] = grade_box(run, checks[case].get(box, []))
    return sheet, results


def render(sheet: Sheet, results: dict[int, Result], batch: Path) -> str:
    counts = {TICK: 0, OPEN: 0, PERSON: 0}
    for result in results.values():
        counts[result.status] += 1
    lines = list(sheet.lines)
    for index, (_, _, box) in sheet.boxes.items():
        result = results[index]
        if result.status == TICK:
            lines[index] = f"- [x] {box} — *auto:* {result.evidence}"
        elif result.status == OPEN:
            lines[index] = f"- [ ] {box} — *auto, open:* {result.evidence}"
        else:
            lines[index] = f"- [ ] {box} — **needs a person**" + (
                f" ({result.evidence})" if result.evidence else ""
            )
    note = [
        f"*Pre-ticked by scripts/grade_cases.py from the checks declared in each case.toml; no "
        f"model read the runs. {counts[TICK]} boxes ticked and {counts[OPEN]} left open by a "
        f"check, {counts[PERSON]} need a person. A box left open is one whose check did not find "
        "what it looks for: a person may still tick it, and untick a ticked one.*",
        "",
    ]
    heading = next(i for i, line in enumerate(lines) if line.startswith("# "))
    return "\n".join(lines[: heading + 2] + note + lines[heading + 2 :]) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("batch", type=Path, help="an eval batch: outputs/eval-runs/<time>")
    parser.add_argument("--out", type=Path, default=None, help="default: <batch>/grading-auto.md")
    args = parser.parse_args(argv)
    if not (args.batch / "grading.md").is_file():
        parser.error(f"{args.batch} has no grading.md")
    sheet, results = grade_batch(args.batch)
    out = args.out or args.batch / "grading-auto.md"
    out.write_text(render(sheet, results, args.batch))
    counts = {
        status: sum(r.status == status for r in results.values()) for status in (TICK, OPEN, PERSON)
    }
    print(f"{out}: {counts[TICK]} ticked, {counts[OPEN]} open, {counts[PERSON]} for a person")
    return 0


if __name__ == "__main__":
    sys.exit(main())
