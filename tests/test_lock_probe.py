"""scripts/lock_probe.py, run as two processes on this one machine.

On one node every lock excludes, so this checks the probe's own handshake and
reporting, not any filesystem; and that it says one node proves nothing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "lock_probe.py"


def _task(rank: int, directory: Path) -> subprocess.Popen[str]:
    env = {**os.environ, "SLURM_JOB_ID": "7", "SLURM_STEP_ID": "0", "SLURM_PROCID": str(rank)}
    return subprocess.Popen(
        [sys.executable, str(SCRIPT), str(directory)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def test_two_tasks_on_one_node_report_every_lock_excluded(tmp_path: Path) -> None:
    trier = _task(1, tmp_path)
    holder = _task(0, tmp_path)
    held_out, held_err = holder.communicate(timeout=60)
    tried_out, tried_err = trier.communicate(timeout=60)

    assert holder.returncode == 0, held_err
    assert trier.returncode == 0, tried_err
    assert "held every lock" in held_out
    assert "ONE node" in tried_out
    for name in ("registry_lock", "flock:", "POSIX lock", "O_EXCL lock file"):
        line = next(line for line in tried_out.splitlines() if name in line)
        assert "excluded" in line, tried_out
    assert "read new at once" in tried_out
    # The O_EXCL file is removed, so the directory can be used again.
    assert not (tmp_path / "excl.lock").exists()


def test_a_second_run_in_the_same_directory_works(tmp_path: Path) -> None:
    for _ in range(2):
        trier = _task(1, tmp_path)
        holder = _task(0, tmp_path)
        assert holder.wait(timeout=60) == 0
        assert trier.wait(timeout=60) == 0


def test_refuses_to_run_outside_slurm_tasks(tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if key != "SLURM_PROCID"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "two Slurm tasks" in result.stderr
