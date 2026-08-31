"""One place that records what an external tool was asked and what it answered.

Every adapter that shells out captures the right material — argv, stdout, stderr,
exit code — and until now each did it with its own private copy of the writer.
Twelve copies, and they had drifted into eight variants: some coerced argv tokens
to `str` and some assumed they already were, some created the parent directory
and some assumed it existed, and the two calling conventions (one command, or a
list of them) had diverged into separate functions rather than one.

Each disagreement is resolved here toward the more defensive of the two, because
the difference only ever showed up as a crash in the adapter that lacked it:
tokens are coerced, and the parent directory is created.

This is the first half of roadmap section 6. It removes the duplication; making
the logs *addressable* — a typed list on the artifact, so that reading a step's
logs does not mean knowing which adapter wrote them — is the second half and
needs the artifact schema to carry them.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

# tqdm redraws its bar by writing a carriage return, and `text=True` translates
# every one of those into a newline — so a progress bar arrives as one line per
# update. A pytom match wrote 41,212 of them into a 2.7 MB log, which is where
# the two lines that mattered were hiding.
_PROGRESS_LINE_RE = re.compile(r"^\s*\d+%\|")

CommandRun = tuple[Sequence[object], "subprocess.CompletedProcess[str]"]
"""One invocation: the argv it was given, and the process that came back."""


def collapse_progress(text: str) -> str:
    """Keep the last state of each progress bar and count what was dropped.

    The intermediate states of a redrawn bar carry nothing the final state does
    not, and they bury everything else in the log. Elided rather than deleted:
    the count stays, so a reader is never left wondering whether output went
    missing.
    """

    lines = text.splitlines()
    kept: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        kept.append(run[-1])
        if len(run) > 1:
            kept.append(f"[... {len(run) - 1} progress updates elided]")
        run.clear()

    for line in lines:
        if _PROGRESS_LINE_RE.match(line):
            run.append(line)
            continue
        flush()
        kept.append(line)
    flush()
    trailing = "\n" if text.endswith("\n") else ""
    return "\n".join(kept) + trailing


def command_log_text(*runs: CommandRun) -> str:
    """The canonical text for one or more external commands.

    Separate from the writer so that a caller assembling a log in memory — or a
    test asserting on its shape — does not have to touch the filesystem.
    """

    sections = [
        f"$ {shlex.join(str(token) for token in command)}\n\n"
        f"[stdout]\n{collapse_progress(result.stdout or '')}\n"
        f"[stderr]\n{collapse_progress(result.stderr or '')}\n"
        f"[exit_code]\n{result.returncode}\n"
        for command, result in runs
    ]
    return "\n\n".join(sections)


def write_command_log(path: Path, *runs: CommandRun) -> Path:
    """Write the log for one or more commands, and return where it went.

    Returning the path is what lets a caller record the log on the artifact it
    belongs to rather than reconstructing the location separately.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(command_log_text(*runs))
    return path
