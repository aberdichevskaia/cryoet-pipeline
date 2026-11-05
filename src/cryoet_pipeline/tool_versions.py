"""The versions of the external tools the pipeline runs, asked of the tools.

`scripts/build_library.py` files each tool's documentation under the version
that is installed (docs/library.md). The backends are to use the same functions
later: today their records say "external" where a version belongs.

Every probe is bounded. A tool gets no input, a time limit, and a cap on what is
read from it, so one that waits for answers or keeps printing is stopped rather
than waited for. A probe runs in its own process group, and stopping it stops
whatever it started.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROBE_SECONDS = 10.0
PROBE_LIMIT = 1 << 18  # 256 KiB


@dataclass(frozen=True)
class Probe:
    """What a command printed, stdout and stderr together, and how it ended.

    `stopped` is None when the command ended on its own. Otherwise it is "time",
    "limit" or "match", and `exit_code` is then None. `error` is set when the
    command could not be started.
    """

    command: tuple[str, ...]
    output: str
    exit_code: int | None
    stopped: str | None = None
    error: str | None = None

    @property
    def ended_well(self) -> bool:
        return self.error is None and self.stopped is None and self.exit_code == 0


@dataclass(frozen=True)
class Version:
    """A tool's version, None when it states none, and how it was asked."""

    value: str | None
    how: str
    output: str = ""


def probe(
    command: Sequence[str | os.PathLike[str]],
    *,
    seconds: float = PROBE_SECONDS,
    limit: int = PROBE_LIMIT,
    stop: Callable[[str], bool] | None = None,
    env: Mapping[str, str] | None = None,
) -> Probe:
    """Run `command` with no input and read what it prints, within bounds.

    Reading ends at end of output, after `seconds`, after `limit` bytes, or as
    soon as `stop` returns true for the text read so far.
    """

    argv = tuple(os.fspath(part) for part in command)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=None if env is None else dict(env),
            start_new_session=True,
        )
    except OSError as error:
        return Probe(argv, "", None, error=str(error))
    assert process.stdout is not None
    read = bytearray()
    stopped: str | None = None
    deadline = time.monotonic() + seconds
    try:
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([process.stdout], [], [], left)[0]:
                stopped = "time"
                break
            chunk = os.read(process.stdout.fileno(), 1 << 16)
            if not chunk:
                break
            read += chunk
            if len(read) >= limit:
                del read[limit:]
                stopped = "limit"
                break
            if stop is not None and stop(read.decode(errors="replace")):
                stopped = "match"
                break
        if stopped is None:
            # The output is closed, which a program can do before it ends.
            try:
                process.wait(timeout=max(deadline - time.monotonic(), 0.1))
            except subprocess.TimeoutExpired:
                stopped = "time"
    finally:
        if stopped is not None or process.poll() is None:
            _stop(process)
        process.wait()
        process.stdout.close()
    return Probe(
        argv,
        read.decode(errors="replace"),
        None if stopped else process.returncode,
        stopped,
    )


def _stop(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def imod_version(imod_dir: Path) -> Version:
    """IMOD's own statement of its version: the VERSION file in its folder."""

    try:
        text = (imod_dir / "VERSION").read_text(errors="replace").strip()
    except OSError:
        return Version(None, f"no VERSION file in {imod_dir}")
    return Version(text.split()[0] if text else None, f"{imod_dir / 'VERSION'}")


# The version line of CTFFIND's start-up banner. The same pattern as
# cryoet_pipeline.backends.ctf.ctffind_version, which is to use this module
# later. It is anchored, so that CTFFIND 5's "Library Version : ..." line is not
# taken for it.
_CTFFIND_VERSION = re.compile(r"^\s*Version\s*:\s*(?P<version>\d+(?:\.\d+)*)", re.MULTILINE)


def ctffind_version(executable: str | os.PathLike[str], seconds: float = 5.0) -> Version:
    """The version CTFFIND's start-up banner states. CTFFIND is given no input
    and stopped once the banner is read, so it fits nothing."""

    result = probe(
        [executable], seconds=seconds, limit=1 << 16,
        stop=lambda text: _CTFFIND_VERSION.search(text) is not None,
    )
    found = _CTFFIND_VERSION.search(result.output)
    return Version(found["version"] if found else None, "its start-up banner", _head(result))


def script_python(executable: Path) -> Path | None:
    """The Python a script runs under, read from its first line.

    This is the interpreter `#!` names. For `#!/usr/bin/env python3`, it is the
    `python3` beside the script (a conda environment's bin folder), or else the
    one on PATH. None means the file names no Python.
    """

    try:
        with open(executable, "rb") as handle:
            first = handle.readline(1024)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    words = first[2:].decode(errors="replace").split()
    if not words:
        return None
    if Path(words[0]).name == "env":
        names = [word for word in words[1:] if not word.startswith("-")]
        if not names or "python" not in names[0]:
            return None
        beside = executable.parent / names[0]
        if beside.is_file():
            return beside
        found = shutil.which(names[0])
        return Path(found) if found else None
    interpreter = Path(words[0])
    return interpreter if "python" in interpreter.name else None


_PACKAGE_VERSION = re.compile(r"^\d[\w.+!-]*$")
_ASK_PACKAGE = (
    "import importlib.metadata as m, sys\n"
    "try:\n"
    "    print(m.version(sys.argv[1]))\n"
    "except m.PackageNotFoundError:\n"
    "    sys.exit(3)\n"
)


def package_version(python: str | os.PathLike[str], distribution: str) -> Version:
    """The installed version of a Python distribution, as the environment of
    `python` records it. `-I` keeps the current folder and the user's site
    packages out of the answer."""

    result = probe([python, "-I", "-c", _ASK_PACKAGE, distribution], seconds=60.0)
    lines = result.output.strip().splitlines()
    value = lines[-1].strip() if result.ended_well and lines else None
    if value is not None and not _PACKAGE_VERSION.match(value):
        value = None
    return Version(value, f"package metadata of {distribution} under {python}", _head(result))


_SLURM_VERSION = re.compile(r"\bslurm[ -](?P<version>\d+(?:\.\d+)+)", re.IGNORECASE)


def slurm_version(sinfo: str | os.PathLike[str]) -> Version:
    result = probe([sinfo, "-V"])
    found = _SLURM_VERSION.search(result.output)
    return Version(found["version"] if found else None, "sinfo -V", _head(result))


_BASH_VERSION = re.compile(r"GNU bash, version (?P<version>\d+(?:\.\d+)+)")


def bash_version(bash: str | os.PathLike[str]) -> Version:
    result = probe([bash, "--version"])
    found = _BASH_VERSION.search(result.output)
    return Version(found["version"] if found else None, "bash --version", _head(result))


def stated_version(
    executable: str | os.PathLike[str],
    name: str,
    attempts: Sequence[Sequence[str]] = (("--version",), ("--help",), ("-h",), ()),
) -> Version:
    """For a program with no known way to ask: the version it states after its
    own name, on one line, when it is run with each of `attempts` in turn.

    The name is required on the line, so that a "CUDA version" line or a
    library's version that the program prints is not taken for its own.
    """

    # Between the name and the number only punctuation, spaces and a word such
    # as "version" may stand: "MotionCor3 version 1.1.2", "MotionCor3: v1.1.2".
    # "MotionCor3 built with CUDA 12.2" names no version of its own.
    pattern = re.compile(
        rf"\b{re.escape(name)}\b[ \t:,(=-]*(?:version|ver\.?|release)?[ \t:=.-]*"
        r"v?(?P<version>\d+\.\d+(?:\.\d+)*)",
        re.IGNORECASE,
    )
    said = ""
    for flags in attempts:
        result = probe([executable, *flags])
        found = pattern.search(result.output)
        how = " ".join([Path(os.fspath(executable)).name, *flags])
        if found:
            return Version(found["version"], how, _head(result))
        said = said or _head(result)
    return Version(None, "asked with " + ", ".join(
        repr(" ".join(flags)) if flags else "no arguments" for flags in attempts
    ), said)


def _head(result: Probe, lines: int = 20) -> str:
    """The start of what a probe printed, for a record of how a version was read."""

    if result.error:
        return f"could not run: {result.error}"
    return "\n".join(result.output.splitlines()[:lines])
