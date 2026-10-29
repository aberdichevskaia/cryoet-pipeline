#!/usr/bin/env python3
"""Does a lock taken on one node keep out a process on another node, here?

    srun -N2 --ntasks-per-node=1 -t 5 python3 scripts/lock_probe.py DIR

Every job commits to its project's registry under a lock (artifacts.py,
registry_lock), and jobs of one project run side by side on different nodes. At
MDC on 2026-10-29 five jobs on five nodes lost three of their registry writes,
on GPFS, which the lock's own check does not judge. Whether a lock reaches other
nodes depends on the filesystem and how it is mounted, and only trying it there
says so.

Run as two Slurm tasks on two nodes, with DIR on the filesystem the projects
live on. Task 0 takes each lock and holds it; task 1 tries each one without
waiting, and says which it got as well, which are the ones that do not exclude:
the registry's own lock, flock(), a POSIX record lock (fcntl/lockf) and a lock
file created with O_EXCL. Task 1 also reads a file that task 0 replaced, as a
commit replaces the registry, to see whether the new one is read at once.

Nothing in the project is touched; DIR holds only this script's small files and
may be reused.
"""

from __future__ import annotations

import errno
import fcntl
import os
import socket
import sys
import time
from contextlib import ExitStack
from pathlib import Path

TIMEOUT_SECONDS = 120.0
HOST = socket.gethostname()


def signal(path: Path, token: str) -> None:
    """Tell the other task a step is done, replacing the file as a commit does."""

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(f"{token} {HOST}\n")
    os.replace(temporary, path)


def wait_for(path: Path, token: str) -> str:
    """Wait for `path` to carry this run's token; return the host that wrote it."""

    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            words = path.read_text().split()
        except FileNotFoundError:
            words = []
        if len(words) == 2 and words[0] == token:
            return words[1]
        time.sleep(0.2)
    raise SystemExit(f"{HOST}: gave up waiting for {path}")


def holder(directory: Path, token: str) -> None:
    excl = directory / "excl.lock"
    # Left by a run that died holding it.
    excl.unlink(missing_ok=True)
    signal(directory / "data", f"old-{token}")
    signal(directory / "ready", token)
    wait_for(directory / "seen", token)
    with ExitStack() as held:
        registry_lock = _registry_lock()
        if registry_lock is not None:
            held.enter_context(registry_lock(directory / "registry.json", timeout=10.0))
        flock_fd = os.open(directory / "flock.lock", os.O_RDWR | os.O_CREAT, 0o666)
        held.callback(os.close, flock_fd)
        fcntl.flock(flock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        posix_fd = os.open(directory / "posix.lock", os.O_RDWR | os.O_CREAT, 0o666)
        held.callback(os.close, posix_fd)
        fcntl.lockf(posix_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        excl_fd = os.open(excl, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        held.callback(excl.unlink)
        held.callback(os.close, excl_fd)
        signal(directory / "data", f"new-{token}")
        signal(directory / "held", token)
        other = wait_for(directory / "done", token)
    print(f"task 0 on {HOST}: held every lock while task 1 on {other} tried them")


def _registry_lock():
    try:
        from cryoet_pipeline.artifacts import registry_lock
    except ImportError:
        return None
    return registry_lock


def try_registry_lock(directory: Path) -> str:
    name = "registry_lock (the registry's own)"
    registry_lock = _registry_lock()
    if registry_lock is None:
        return f"{name}: not tried, cryoet_pipeline is not importable here"
    from cryoet_pipeline.artifacts import RegistryLockError

    try:
        with registry_lock(directory / "registry.json", timeout=1.0):
            pass
    except RegistryLockError as error:
        if "has been held by another command" in str(error):
            return f"{name}: excluded (task 1 was refused while task 0 held it)"
        return f"{name}: refused: {error}"
    return f"{name}: NOT exclusive across nodes (task 1 got it while task 0 held it)"


def try_lock(name: str, path: Path, lock) -> str:
    try:
        descriptor = os.open(path, os.O_RDWR)
    except OSError as error:
        return f"{name}: could not open {path}: {error}"
    try:
        lock(descriptor)
    except OSError as error:
        if error.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return f"{name}: excluded (task 1 was refused while task 0 held it)"
        code = errno.errorcode.get(error.errno or 0, error.errno)
        return f"{name}: failed with {code} ({error.strerror})"
    finally:
        os.close(descriptor)
    return f"{name}: NOT exclusive across nodes (task 1 got it while task 0 held it)"


def try_excl(path: Path) -> str:
    name = "O_EXCL lock file"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return f"{name}: excluded (task 1 could not create it)"
    # Task 0 removes it.
    os.close(descriptor)
    return f"{name}: NOT exclusive across nodes (task 1 created it too)"


def trier(directory: Path, token: str) -> None:
    other = wait_for(directory / "ready", token)
    # Read before task 0 replaces it, so that a node that caches has it cached.
    before = (directory / "data").read_text().split()[0]
    signal(directory / "seen", token)
    wait_for(directory / "held", token)
    after = (directory / "data").read_text().split()[0]
    lines = [
        try_registry_lock(directory),
        try_lock("flock", directory / "flock.lock",
                 lambda fd: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)),
        try_lock("POSIX lock (fcntl/lockf)", directory / "posix.lock",
                 lambda fd: fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)),
        try_excl(directory / "excl.lock"),
        "a file task 0 replaced: read new at once" if after == f"new-{token}"
        else f"a file task 0 replaced: read STALE ({after!r}; before it was {before!r})",
    ]
    signal(directory / "done", token)
    print(f"task 1 on {HOST}, task 0 on {other}:")
    if other == HOST:
        print("  both tasks ran on ONE node, which says nothing about other nodes")
    for line in lines:
        print(f"  {line}")


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit(__doc__)
    directory = Path(args[0])
    directory.mkdir(parents=True, exist_ok=True)
    token = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{os.environ.get('SLURM_STEP_ID', '0')}"
    rank = os.environ.get("SLURM_PROCID")
    if rank == "0":
        holder(directory, token)
    elif rank == "1":
        trier(directory, token)
    else:
        raise SystemExit("run as two Slurm tasks: srun -N2 --ntasks-per-node=1 ...")


if __name__ == "__main__":
    main()
