#!/usr/bin/env python3
"""A second guard on what the assistant's tools change.

Claude Code runs this, with the call as JSON on stdin, before every Bash call
and every Edit, Write, MultiEdit and NotebookEdit call of a session that
scripts/assistant started (assistant/settings.json, PreToolUse). Codex runs it
before every shell command and every apply_patch of a session that
`scripts/assistant --codex` started (assistant/codex/hooks.json), with the same
fields. Exit 2 blocks the call and shows the assistant the message on stderr.
Exit 0 leaves the call to the session's permission rules.

It blocks what the assistant must never do itself, wherever it sits in a command
line, including after `&&`, `|`, `$(`, `env`, `nohup` or `bash -c` (or -lc):

* starting a Slurm job (sbatch, srun, salloc), which would skip the checks and
  the journal of scripts/submit_lanes.py and scripts/submit_recipe.py;
* running a batch script (*.sbatch) directly;
* writing a project's journal.jsonl, which only those two scripts and
  scripts/export_relion.py write;
* running RELION itself: a relion_* program, an export's relion_commands.sh, or
  mpirun/mpiexec, which would compute on the login node. The person runs RELION,
  in a job on the cluster;
* running scripts/build_library.py, which writes the reference library;
* running scripts/run_series.py with --execute, which computes on the login node;
* a shell with nothing to run (a bare `bash`, or `... | bash`), which reads its
  commands from its input, where this guard never sees them. Codex can start one
  and type into it later;
* changing the reference library (docs/library.md). A command line that names
  the library, or runs inside it, may run only commands that read. An edit of a
  file in the library is refused, whatever the tool;
* a Codex patch (apply_patch, as its own tool or as a shell command) of a file
  the Edit rules of assistant/settings.json deny. Claude Code applies those rules
  itself; Codex has no rules for files, so they are read from there. A patch
  whose files cannot be read is refused.

In a headless Codex run (CRYOET_ASSISTANT_HEADLESS=1, which `scripts/assistant
--codex -p` sets), nobody is there to ask, so it also refuses what a person
would have to approve first: a patch of a file the Edit rules of
assistant/settings.json ask for, and submit_lanes.py, submit_recipe.py or
export_relion.py without --dry-run (submit_recipe.py --list only lists). Claude
Code refuses an ask in a headless run itself; in a Codex session the read-only
sandbox makes the person approve every patch and every submission.

With CRYOET_ASSISTANT_CALL_LOG set, it adds a line for every call it sees to that
file: the call's id, its turn, the tool, its input, and the message it blocked
the call with, if it did. scripts/run_cases.py reads it for a Codex run, whose
own record leaves out the calls Codex refused.

Like the permission rules, this is a second layer and not a sandbox: a command
that hides the call (a Python one-liner that runs sbatch, say) gets past it.
Such a command is not on the session's allow list, so the person is asked for
it; under Codex it runs in its read-only sandbox, without the network. The first
layer is the scripts' own checks, and for the library its file permissions,
since a snapshot is written read-only.

Written for the python3 of a login node, which may be old: standard library only,
nothing newer than Python 3.6.
"""

import fnmatch
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
# RELION's programs and how they are started: never by the assistant.
RELION_PREFIX = "relion_"
MPI_LAUNCHERS = {"mpirun", "mpiexec"}
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
# The library's own search, which only reads: run as `python3 <it>`, it counts
# as a reader even when its command line names the library. Only this file
# counts, not another script that happens to have its name.
SEARCH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "search_library.py")
)
# A shell's -c, alone or among other one-letter options (-lc, -ec): the word
# after it is the script the shell runs.
_SHELL_C = re.compile(r"^-[A-Za-z]*c[A-Za-z]*$")
# What a shell takes that runs nothing: its version or its help.
SHELL_INFO = {"--version", "--help"}
# Redirections that give a shell its input, and those that take its output; the
# word after each is a file, not a script for the shell to run.
INPUTS = {"<", "<<", "<<<", "<<-"}
RUNNER = "run_series.py"
EXECUTE = "--execute"
# Codex's tool for edits: its patch comes in tool_input["command"], each file on
# a line of its own. Codex also takes `apply_patch <<'EOF' ...` as a shell
# command and applies it, and then the guard sees it as a shell command.
PATCH_TOOL = "apply_patch"
_PATCH_FILE = re.compile(
    r"^[ \t]*\*\*\* (?:Add File|Update File|Delete File|Move to):[ \t]*(.*?)[ \t\r]*$",
    re.MULTILINE,
)
_PATCH_COMMAND = re.compile(r"(?:^|[\s;&|(])(?:\S*/)?apply_?patch\b")
REPOSITORY = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# The client rules that Claude Code reads; the guard reads their Edit rules for
# Codex's patches.
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
# A run that asks nobody: `scripts/assistant --codex -p`.
HEADLESS = "CRYOET_ASSISTANT_HEADLESS"
# Where to note every call, for scripts/run_cases.py.
CALL_LOG = "CRYOET_ASSISTANT_CALL_LOG"
# The scripts that submit, or export with the journal, unless given --dry-run.
SUBMITTERS = {"submit_lanes.py", "submit_recipe.py", "export_relion.py"}
DRY_RUN = "--dry-run"


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
            if name.startswith(RELION_PREFIX) or name in MPI_LAUNCHERS:
                return relion_message(name)
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
                script = shell_script(rest)
                if script is not None and depth < 3:
                    found = problem(script, depth + 1)
                    if found:
                        return found
                if name in SHELLS and runs_its_input(words, i):
                    return input_shell_message(name)
                operands = [one for one in rest if not one.startswith("-")]
                if operands and os.path.basename(operands[0]).endswith(".sbatch"):
                    return batch_message(operands[0])
                if operands and os.path.basename(operands[0]).startswith(RELION_PREFIX):
                    return relion_message(os.path.basename(operands[0]))
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


def shell_script(rest):
    """The script a shell runs with -c (or -lc, -ec and the like), or None."""
    for at, word in enumerate(rest):
        if _SHELL_C.match(word):
            return rest[at + 1] if at + 1 < len(rest) else ""
    return None


def runs_its_input(words, at):
    """Whether the shell at words[at] has nothing to run but what comes on its
    input: no -c, no script file, nor only its version or help. Its input may be
    a pipe, a redirection, a here-document or a terminal that Codex types into
    later; the guard sees none of it."""
    j = at + 1
    while j < len(words) and words[j] not in SEPARATORS:
        word = words[j]
        if word in INPUTS:
            return True
        if word in REDIRECTS:
            j += 2
            continue
        if word in SHELL_INFO or _SHELL_C.match(word):
            return False
        if word in ("-o", "-O", "+o", "+O", "--rcfile", "--init-file"):
            j += 2
            continue
        if not word.startswith(("-", "+")):
            return False
        j += 1
    return True


def input_shell_message(name):
    return (
        f"`{name}` with nothing to run reads its commands from its input, which this "
        "guard never sees. Run the command itself, or the script as `bash <file>`."
    )


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
        "a project's journal.jsonl is written by scripts/submit_lanes.py, "
        "scripts/submit_recipe.py and scripts/export_relion.py only. Read it, but never "
        "change it."
    )


def relion_message(name):
    return (
        f"`{name}` would run RELION here, on the login node. RELION is the person's to run, "
        "in a job on the cluster: give them the commands of the export's relion_commands.sh."
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
        script = shell_script(rest) if name in SHELLS else None
        if script is not None and depth < 3:
            inner = commands_in(script, depth + 1)
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


def runner_problem(command):
    """scripts/run_series.py with --execute, in any form. argparse takes any
    beginning of an option for it (--exe, say), and --execute is its only option
    that starts with --e."""
    commands = commands_in(command)
    if commands is None:
        return runner_message() if re.search(r"\brun_series\.py\b.*\s--e", command) else None
    for name, rest in commands:
        script, arguments = name, rest
        if name.startswith("python"):
            at = next((k for k, one in enumerate(rest) if not one.startswith("-")), None)
            if at is None:
                continue
            script, arguments = rest[at], rest[at + 1:]
        if os.path.basename(script) != RUNNER:
            continue
        for one in arguments:
            option = one.split("=", 1)[0]
            if len(option) > 2 and EXECUTE.startswith(option):
                return runner_message()
    return None


def runner_message():
    return (
        "scripts/run_series.py --execute would run pipeline steps here, on the login "
        "node. Without --execute it prints them; a run goes to the cluster through "
        "scripts/submit_lanes.py."
    )


def patch_paths(patch):
    """The files a Codex patch adds, changes, deletes or moves to."""
    return [path for path in _PATCH_FILE.findall(patch) if path]


def edit_rules(kind):
    """The Edit rules assistant/settings.json lists under kind ("deny" or "ask"),
    as written inside Edit(...), or None when the file cannot be read."""
    try:
        with open(SETTINGS) as handle:
            listed = json.load(handle)["permissions"][kind]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return [
        rule[len("Edit("):-1]
        for rule in listed
        if rule.startswith("Edit(") and rule.endswith(")")
    ]


def denied_edits():
    """The Edit rules assistant/settings.json denies, or None."""
    return edit_rules("deny")


def _match(pattern, parts):
    """Whether path parts match pattern parts: ** is any number of folders, and
    each other part a name with * and ? in it."""
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return any(_match(pattern[1:], parts[k:]) for k in range(len(parts) + 1))
    return (
        bool(parts)
        and fnmatch.fnmatchcase(parts[0], pattern[0])
        and _match(pattern[1:], parts[1:])
    )


def rule_matches(rule, path):
    """Whether the Edit rule (as Claude Code reads it) names the file at path, an
    absolute, resolved path. //x is the absolute path /x; ~/x is in the home
    folder; /x is next to settings.json; anything else is in the repository,
    where the assistant runs. A rule that starts with ** names the file wherever
    the rest of it matches."""
    if rule.startswith("//"):
        base, pattern = os.sep, rule[2:]
    elif rule.startswith("~/"):
        base, pattern = os.path.expanduser("~"), rule[2:]
    elif rule.startswith("/"):
        base, pattern = os.path.dirname(SETTINGS), rule[1:]
    else:
        base, pattern = REPOSITORY, rule[2:] if rule.startswith("./") else rule
    parts = [one for one in pattern.split("/") if one]
    if parts and parts[0] == "**":
        return _match(parts, [one for one in path.split(os.sep) if one])
    base = os.path.realpath(base)
    if not inside(path, {base}):
        return False
    return _match(parts, [one for one in os.path.relpath(path, base).split(os.sep) if one])


def library_file_message(path):
    return (
        f"{path} is in the reference library, which is read-only (docs/library.md): "
        "no tool edits it."
    )


def patch_problem(patch, cwd, environ):
    """A Codex patch that changes a file in the library, or a file the Edit rules
    of assistant/settings.json deny; and a patch whose files cannot be read."""
    if not isinstance(patch, str):
        return (
            "this patch came in a form the guard cannot read, so it cannot tell "
            "which files it changes."
        )
    paths = patch_paths(patch)
    if not paths:
        return (
            "the guard cannot tell which files this patch changes, so it does not "
            "let it through."
        )
    rules = denied_edits()
    asked = edit_rules("ask") if environ.get(HEADLESS) == "1" else []
    if rules is None or asked is None:
        return "assistant/settings.json cannot be read, so the guard cannot check this patch."
    roots = library_roots(environ)
    resolved = [os.path.realpath(os.path.join(cwd, os.path.expanduser(path))) for path in paths]
    for path, where in zip(paths, resolved):
        if inside(where, roots):
            return library_file_message(path)
        rule = next((one for one in rules if rule_matches(one, where)), None)
        if rule is not None:
            return (
                f"{path} is not the assistant's to change: assistant/settings.json denies "
                f"Edit({rule})."
            )
    for path, where in zip(paths, resolved):
        rule = next((one for one in asked if rule_matches(one, where)), None)
        if rule is not None:
            return (
                f"the person has not approved this edit of {path} (assistant/settings.json "
                f"asks before Edit({rule}))."
            )
    return None


def submission_problem(command, environ):
    """In a run that asks nobody: a submitting script run without --dry-run, which
    the person approves first. submit_recipe.py --list only lists."""
    if environ.get(HEADLESS) != "1":
        return None
    commands = commands_in(command)
    if commands is None:
        found = re.search(r"\b(?:submit_lanes|submit_recipe|export_relion)\.py\b", command)
        if found and DRY_RUN not in command.split():
            return submission_message(found.group(0))
        return None
    for name, rest in commands:
        script, arguments = name, rest
        if name.startswith("python"):
            at = next((k for k, one in enumerate(rest) if not one.startswith("-")), None)
            if at is None:
                continue
            script, arguments = rest[at], rest[at + 1:]
        base = os.path.basename(script)
        if base not in SUBMITTERS or DRY_RUN in arguments:
            continue
        if base == "submit_recipe.py" and "--list" in arguments:
            continue
        return submission_message(base)
    return None


def submission_message(name):
    return f"the person has not approved running {name} without --dry-run, which submits."


def shell_patch_problem(command, cwd, environ):
    """A patch given to apply_patch as a shell command, which Codex applies itself."""
    if "*** Begin Patch" not in command or not _PATCH_COMMAND.search(command):
        return None
    return patch_problem(command, cwd, environ)


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
        if name not in READERS and not runs_search(name, rest, cwd):
            return library_message(f"`{name}`")
        if name == "find":
            action = next((one for one in rest if one in FIND_ACTIONS), None)
            if action:
                return library_message(f"`find {action}`")
        for at, one in enumerate(rest[:-1]):
            if one in REDIRECTS and inside(resolve(rest[at + 1], cwd), roots):
                return library_message("writing to " + rest[at + 1])
    return None


# Python's options that change nothing about which code runs.
QUIET_FLAGS = {"-I", "-u", "-B", "-E", "-s", "-S", "-O", "-OO", "-q", "-b", "-bb", "-P"}


def runs_search(name, rest, cwd):
    """Whether a command is Python running scripts/search_library.py, this
    repository's own, with no option before it that runs other code (-c, -m)."""
    if not name.startswith("python"):
        return False
    words = list(rest)
    while words and words[0] in QUIET_FLAGS:
        words.pop(0)
    return bool(words) and not words[0].startswith("-") and resolve(words[0], cwd) == SEARCH


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
        return library_file_message(path)
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
    found = None
    if tool == "Bash":
        command = tool_input.get("command") or ""
        if isinstance(command, list):
            command = " ".join(shlex.quote(str(word)) for word in command)
        found = (problem(command) or builder_problem(command) or runner_problem(command)
                 or library_problem(command, cwd, os.environ)
                 or shell_patch_problem(command, cwd, os.environ)
                 or submission_problem(command, os.environ))
    elif tool in FILE_TOOLS:
        found = file_problem(tool_input, cwd, os.environ)
    elif tool == PATCH_TOOL:
        found = patch_problem(tool_input.get("command"), cwd, os.environ)
    note_call(call, found)
    if found:
        print(f"Blocked by assistant/guard.py: {found}", file=sys.stderr)
        return 2
    return 0


def note_call(call, found):
    """The call, and what blocked it, added to CRYOET_ASSISTANT_CALL_LOG if set. A
    log that cannot be written changes nothing about the call."""
    path = os.environ.get(CALL_LOG)
    if not path:
        return
    entry = {
        "tool_use_id": call.get("tool_use_id"),
        "turn_id": call.get("turn_id"),
        "tool": call.get("tool_name"),
        "input": call.get("tool_input"),
        "blocked": found or None,
    }
    try:
        with open(path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
    except (OSError, TypeError, ValueError):
        pass


if __name__ == "__main__":
    sys.exit(main())
