#!/usr/bin/env python3
"""Run a command under a run folder's lock, or stop at once naming the job that holds it.

    python scripts/hold_run_lock.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan2_port.lock \
        -- bash scripts/mra_port_run.sbatch

scripts/mra_port_run.sbatch runs the rest of its job so, the setup, the port and the
report (cryoet_pipeline.run_lock). A second job of the same run prints

    REFUSED: outputs/.../mra_pad_ssn_g2atan2_port is taken by job 3790159
    (mra_pad_ssn_g2atan2) on maxg17, pid 81234, since 2026-10-29T19:55:18+01:00;
    this job ran nothing

and ends with 1. Otherwise this script ends as the command did: with its exit status,
or 128 + the number of the signal that ended it, as a shell gives it.

SLURM signals every process of a job that runs out of time or is cancelled. A signal
sent to this script alone (SIGTERM, SIGINT or SIGHUP) is passed on to the command,
and the lock is held until the command has ended.
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType

from cryoet_pipeline.run_lock import RunLockError, run_lock

PASSED_ON = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3 or argv[1] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    try:
        with run_lock(Path(argv[0])):
            return _run(argv[2:])
    except RunLockError as error:
        print(f"REFUSED: {error}; this job ran nothing", file=sys.stderr)
        return 1


def _run(command: list[str]) -> int:
    child: subprocess.Popen[bytes] | None = None
    early: list[int] = []

    def pass_on(number: int, _frame: FrameType | None) -> None:
        if child is None:
            early.append(number)
        else:
            child.send_signal(number)

    before = {number: signal.signal(number, pass_on) for number in PASSED_ON}
    try:
        child = subprocess.Popen(command)
        # A signal that came before the command started stops it as it starts.
        for number in early:
            child.send_signal(number)
        code = child.wait()
    finally:
        for number, handler in before.items():
            signal.signal(number, handler)
    return 128 - code if code < 0 else code


if __name__ == "__main__":
    sys.exit(main())
