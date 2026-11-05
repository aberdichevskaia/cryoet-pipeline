#!/usr/bin/env python3
"""A second guard on what the assistant's tools change.

Claude Code runs this, with the call as JSON on stdin, before every Bash call
and every Edit, Write, MultiEdit and NotebookEdit call of a session that
scripts/assistant started (assistant/settings.json, PreToolUse). Exit 2 blocks
the call and shows the assistant the message on stderr. Exit 0 leaves the call
to the session's permission rules.

It blocks what the assistant must never do itself, wherever it sits in a command
line, including after `&&`, `|`, `$(`, `env`, `nohup` or `bash -c`:

* starting a Slurm job (sbatch, srun, salloc), which would skip the checks and
  the journal of scripts/submit_lanes.py and scripts/submit_recipe.py;
* running a batch script (*.sbatch) directly;
* writing a project's journal.jsonl, which only those two scripts write;
* running scripts/build_library.py, which writes the reference library;
* changing the reference library (docs/library.md). A command line that names
  the library, or runs inside it, may run only commands that read. An edit of a
  file in the library is refused, whatever the tool.

Like the permission rules, this is a second layer and not a sandbox: a command
that hides the call (a Python one-liner that runs sbatch, say) gets past it.
Such a command is not on the session's allow list, so the person is asked for
it. The first layer is the scripts' own checks, and for the library its file
permissions, since a snapshot is written read-only.

Written for the python3 of a login node, which may be old: standard library only,
nothing newer than Python 3.6.
"""

import glob
import json
import os
import re
import shlex
import sys

SLURM_JOBS = {"sbatch", "srun", "salloc"}
# Words that run the command after them: the word after one of these is a
# command too.
WRAPPERS = {
    "env", "nohup", "exec", "time", "nice", "timeout", "xargs", "sudo", "command",
    "builtin", "stdbuf", "setsid", "watch", "eval",
}
SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}
# Tokens after which a new command starts.
SEPARATORS = {";", "&&", "||", "|", "&", "(", "{", "!", "then", "do", "else", "elif"}
REDIRECTS = {">", ">>", ">|", "&>", "&>>"}
# Commands that change any file named in their arguments, and commands that
# change only the last one (their destination).
WRITERS = {"tee", "rm", "mv", "truncate", "dd", "chmod"}
COPIERS = {"cp", "ln", "install", "rsync"}
# What a wrapper takes before the command it runs: options, env's VAR=value, a
# duration (timeout) or a count (nice, watch).
_WRAPPER_ARGUMENT = re.compile(r"^(-.*|[A-Za-z_][A-Za-z0-9_]*=.*|\d+(\.\d+)?[smhd]?)$")
JOURNAL = "journal.jsonl"
_SLURM_WORD = re.compile(r"(?:^|[\s;&|(`])(?:\S*/)?(sbatch|srun|salloc)(?=$|[\s;&|)`])")

# The reference library's folders on the sites; scripts/assistant has the same
# list. A session's own library comes in CRYOET_LIBRARY_ROOT.
LIBRARY_SITES = ("/fast/AG_Kudryashev/cryoet-library",)
LIBRARY_VARIABLES = ("CRYOET_LIBRARY", "CRYOET_LIBRARY_ROOT", "CRYOET_LIBRARY_SNAPSHOT")
# What may run in a command line that names the library: commands that only read.
READERS = {
    "cat", "head", "tail", "grep", "egrep", "fgrep", "zgrep", "ls", "find", "wc", "stat",
    "file", "du", "diff", "cmp", "cut", "tr", "nl", "column", "jq", "readlink", "realpath",
    "basename", "dirname", "cd", "pwd", "echo", "printf", "test", "[", "true", "false",
    "sha256sum", "shasum", "md5sum",
}
# find's actions that run or write something.
FIND_ACTIONS = {
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
}
BUILDER = "build_library.py"
FILE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def tokens(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def problem(command, depth=0):
    """Why the command must not run, or None."""
    try:
        words = tokens(command)
    except ValueError:
        # Quotes that do not close: read the raw text for a Slurm command instead.
        match = _SLURM_WORD.search(command)
        if match:
            return slurm_message(match.group(1))
        return None

    starts = True  # whether the next word is in command position
    previous = None
    i = 0
    while i < len(words):
        word = words[i]
        if word.startswith("`"):
            word, starts = word.lstrip("`"), True
        if word in SEPARATORS:
            starts, previous = True, word
            i += 1
            continue
        if previous in REDIRECTS and JOURNAL in word:
            return journal_message()
        if starts:
            name = os.path.basename(word)
            if "=" in word and not word.startswith("-") and previous not in WRAPPERS:
                # VAR=value before the command
                previous = word
                i += 1
                continue
            if name in SLURM_JOBS:
                return slurm_message(name)
            if name.endswith(".sbatch"):
                return batch_message(word)
            if name in WRAPPERS:
                previous = name
                i += 1
                options = []
                while i < len(words) and _WRAPPER_ARGUMENT.match(words[i]):
                    options.append(words[i])
                    i += 1
                # `command -v` and `command -V` only say what a name is: the name
                # after them is not run.
                if name == "command" and any(
                    option.startswith("-") and ("v" in option or "V" in option)
                    for option in options
                ):
                    starts = False
                continue
            if name in SHELLS or name in ("source", "."):
                rest = words[i + 1:]
                if "-c" in rest and depth < 3:
                    script = rest[rest.index("-c") + 1] if rest.index("-c") + 1 < len(rest) else ""
                    found = problem(script, depth + 1)
                    if found:
                        return found
                operands = [one for one in rest if not one.startswith("-")]
                if operands and os.path.basename(operands[0]).endswith(".sbatch"):
                    return batch_message(operands[0])
            end = next(
                (j for j in range(i + 1, len(words)) if words[j] in SEPARATORS), len(words)
            )
            arguments = [one for one in words[i + 1:end] if one not in REDIRECTS]
            in_place = name in ("sed", "perl") and any(one.startswith("-i") for one in arguments)
            if name in WRITERS or in_place:
                if any(JOURNAL in one for one in arguments):
                    return journal_message()
            elif name in COPIERS and arguments and JOURNAL in arguments[-1]:
                return journal_message()
            starts = False
        previous = word
        i += 1
    return None


def slurm_message(name):
    return (
        f"`{name}` would start a Slurm job without the pipeline's checks and journal. "
        "Submit a lane with scripts/submit_lanes.py, or a recipe with "
        "scripts/submit_recipe.py: dry run, show it, and submit after an explicit yes."
    )


def batch_message(path):
    return (
        f"{path} is a batch script; it is never run directly. A lane goes through "
        "scripts/submit_lanes.py, and a recipe through scripts/submit_recipe.py."
    )


def journal_message():
    return (
        "a project's journal.jsonl is written by scripts/submit_lanes.py and "
        "scripts/submit_recipe.py only. Read it, but never change it."
    )


def commands_in(command, depth=0):
    """Every command a line runs, as (its name, the words after it up to the
    next separator), the commands of a `bash -c` script among them. None for a
    line that cannot be read, such as one with a quote that does not close."""
    try:
        words = tokens(command)
    except ValueError:
        return None
    found = []
    starts, previous, i = True, None, 0
    while i < len(words):
        word = words[i]
        if word.startswith("`"):
            word, starts = word.lstrip("`"), True
        if word in SEPARATORS:
            starts, previous = True, word
            i += 1
            continue
        if not starts or ("=" in word and not word.startswith("-") and previous not in WRAPPERS):
            previous = word
            i += 1
            continue
        name = os.path.basename(word)
        if name in WRAPPERS:
            previous = name
            i += 1
            while i < len(words) and _WRAPPER_ARGUMENT.match(words[i]):
                i += 1
            continue
        end = i + 1
        while end < len(words) and words[end] not in SEPARATORS and not words[end].startswith("`"):
            end += 1
        rest = words[i + 1:end]
        found.append((name, rest))
        if name in SHELLS and "-c" in rest and depth < 3:
            at = rest.index("-c") + 1
            inner = commands_in(rest[at], depth + 1) if at < len(rest) else []
            if inner is None:
                return None
            found.extend(inner)
        starts, previous, i = False, word, end
    return found


def builder_problem(command):
    """scripts/build_library.py run in any form: it writes the library."""
    commands = commands_in(command)
    if commands is None:
        return builder_message() if re.search(r"\bbuild_library\.py\b", command) else None
    for name, rest in commands:
        script = name
        if name.startswith("python"):
            script = next((one for one in rest if not one.startswith("-")), "")
        if os.path.basename(script) == BUILDER:
            return builder_message()
    return None


def builder_message():
    return (
        "scripts/build_library.py writes the reference library. The person builds "
        "it; the assistant only reads it."
    )


def library_roots(environ):
    """The folders that hold the library, as named and as resolved."""
    named = [environ.get(name) for name in ("CRYOET_LIBRARY", "CRYOET_LIBRARY_ROOT")]
    roots = set()
    for one in named + list(LIBRARY_SITES):
        if one and one != "none":
            for form in (os.path.normpath(one), os.path.realpath(one)):
                if form != os.sep:
                    roots.add(form)
    return roots


def inside(path, roots):
    return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def resolve(word, cwd):
    return os.path.realpath(os.path.join(cwd, os.path.expanduser(os.path.expandvars(word))))


def names_library(command, cwd, roots):
    """Whether a command line names the library or runs inside it. Read from its
    text, its variables, and each word as the path it resolves to: globs
    expanded, links followed, a relative path taken from the call's folder."""
    if inside(os.path.realpath(cwd), roots):
        return True
    if any(root in command for root in roots):
        return True
    if any("$" + name in command or "${" + name in command for name in LIBRARY_VARIABLES):
        return True
    try:
        words = tokens(command)
    except ValueError:
        return False
    for word in words:
        word = word.strip("`")
        if not word or word in SEPARATORS:
            continue
        if any(char in word for char in "*?["):
            paths = glob.glob(os.path.join(cwd, os.path.expanduser(word))) or [resolve(word, cwd)]
        else:
            paths = [resolve(word, cwd)]
        if any(inside(os.path.realpath(path), roots) for path in paths):
            return True
    return False


def library_problem(command, cwd, environ):
    """A command line that names the library and does more than read it."""
    roots = library_roots(environ)
    if not names_library(command, cwd, roots):
        return None
    commands = commands_in(command)
    if commands is None:
        return library_message("a command line that cannot be read")
    for name, rest in commands:
        if name not in READERS:
            return library_message("`%s`" % name)
        action = next((one for one in rest if one in FIND_ACTIONS), None) if name == "find" else None
        if action:
            return library_message("`find %s`" % action)
        for at, one in enumerate(rest):
            if one in REDIRECTS and at + 1 < len(rest) and inside(resolve(rest[at + 1], cwd), roots):
                return library_message("writing to " + rest[at + 1])
    return None


def library_message(what):
    return (
        what + " in a command line that names the reference library, which is read-only "
        "(docs/library.md). Search it with scripts/search_library.py and read it with "
        "Read, or with cat, head, grep, ls and find; never write, move or delete anything in it."
    )


def file_problem(tool_input, cwd, environ):
    """An edit, by any tool, of a file in the library."""
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if path and inside(resolve(path, cwd), library_roots(environ)):
        return (
            "%s is in the reference library, which is read-only (docs/library.md): "
            "no tool edits it." % path
        )
    return None


def main():
    try:
        call = json.load(sys.stdin)
    except ValueError:
        return 0
    if not isinstance(call, dict):
        return 0
    tool = call.get("tool_name")
    tool_input = call.get("tool_input") or {}
    cwd = call.get("cwd") or os.getcwd()
    if tool == "Bash":
        command = tool_input.get("command") or ""
        found = (problem(command) or builder_problem(command)
                 or library_problem(command, cwd, os.environ))
    elif tool in FILE_TOOLS:
        found = file_problem(tool_input, cwd, os.environ)
    else:
        return 0
    if found:
        print(f"Blocked by assistant/guard.py: {found}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
