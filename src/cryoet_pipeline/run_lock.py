"""One job at a time on a run folder: a POSIX lock its job holds from start to end.

scripts/mra_port_run.sbatch runs each job under the lock of its run folder
(scripts/hold_run_lock.py), so a second job of the same run, started while the
first is still on it, stops at once and names the first instead of computing the
same iterations into the same files. On 2026-10-29 two jobs carrying
mra_pad_ssn_g2atan2_port on, submitted a minute apart, did that for two hours on
two nodes; the folder held one run in the end only because both computed the same
numbers. A job that waited in the queue and starts after the first has ended takes
the lock and carries the run on.

A POSIX lock (fcntl), as the registry's (artifacts.registry_lock): on MDC's GPFS it
keeps out a process on another node, where flock does not (scripts/lock_probe.py),
and the system releases it when its process ends, however that ends, so a job its
time limit stopped leaves nothing to clean up. The lock file, beside the run folder,
is never deleted, since a lock on a deleted file excludes nobody; it holds one line
naming its last holder, which a refused job prints.
"""

from __future__ import annotations

import errno
import fcntl
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from cryoet_pipeline.artifacts import node_local_lock

_HELD_ELSEWHERE = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})
"""What a non-blocking POSIX lock fails with while another process holds it."""

_held_here: set[Path] = set()
"""The lock files this process holds: a POSIX lock belongs to the process, so taking
one again would succeed, and closing its descriptor would release the first."""


class RunLockError(RuntimeError):
    """The run is held by another job, or its lock cannot be had."""


def holder() -> str:
    """This process as a refused job names it: its SLURM job, node, pid and start."""

    job = os.environ.get("SLURM_JOB_ID")
    if job:
        name = os.environ.get("SLURM_JOB_NAME")
        who = f"job {job} ({name})" if name else f"job {job}"
    else:
        who = "a process outside SLURM"
    since = datetime.now().astimezone().isoformat(timespec="seconds")
    return f"{who} on {socket.gethostname()}, pid {os.getpid()}, since {since}"


@contextmanager
def run_lock(path: Path) -> Iterator[None]:
    """Hold the lock file `path` while the block runs; refused at once if it is held.

    Refused too where the lock would hold on this node only (`node_local_lock`): a
    job of the same run on another node would not be kept out, which is what the
    lock is for.
    """

    key = path.resolve()
    if key in _held_here:
        raise RunLockError(f"{path} is held by this process already")
    path.parent.mkdir(parents=True, exist_ok=True)
    problem = node_local_lock(path.parent)
    if problem is not None:
        raise RunLockError(
            f"{path}: {problem}, so a job of the same run on another node would not be kept out"
        )
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in _HELD_ELSEWHERE:
                named = os.pread(descriptor, 4096, 0).decode(errors="replace").strip()
                raise RunLockError(
                    f"{path.parent / path.stem} is taken by "
                    f"{named or 'another process, which has not named itself yet'}"
                ) from None
            name = errno.errorcode.get(exc.errno or 0, str(exc.errno))
            raise RunLockError(
                f"a POSIX lock on {path} failed with {name} ({exc.strerror})"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.pwrite(descriptor, (holder() + "\n").encode(), 0)
        _held_here.add(key)
        try:
            yield
        finally:
            _held_here.discard(key)
    finally:
        # This process's only descriptor of the lock file: closing it releases the lock.
        os.close(descriptor)
