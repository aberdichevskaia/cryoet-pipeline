#!/usr/bin/env python3
"""A second guard on the assistant's shell commands.

Claude Code runs this before every Bash call of a session that scripts/assistant
started (assistant/settings.json, PreToolUse), with the call as JSON on stdin.
Exit 2 blocks the call and shows the assistant the message on stderr; exit 0
leaves it to the session's permission rules.

It blocks three things the assistant must never do itself, wherever they sit in
a command line, including after `&&`, `|`, `$(`, `env`, `nohup` or `bash -c`:

* starting a Slurm job (sbatch, srun, salloc), which would skip the checks and
  the journal of scripts/submit_lanes.py and scripts/submit_recipe.py;
* running a batch script (*.sbatch) directly;
* writing a project's journal.jsonl, which only those two scripts write.

Like the permission rules, this is a second layer and not a sandbox: a command
that hides the call (a Python one-liner that runs sbatch, say) gets past it.
Such a command is not on the session's allow list, so the person is asked for
it. The first layer is the scripts' own checks.

Written for the python3 of a login node, which may be old: standard library only,
nothing newer than Python 3.6.
"""

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


def main():
    try:
        call = json.load(sys.stdin)
    except ValueError:
        return 0
    if not isinstance(call, dict) or call.get("tool_name") != "Bash":
        return 0
    command = (call.get("tool_input") or {}).get("command") or ""
    found = problem(command)
    if found:
        print(f"Blocked by assistant/guard.py: {found}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
