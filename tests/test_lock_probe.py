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


def _task(rank: int, directory: Path, job_id: str = "7") -> subprocess.Popen[str]:
    env = {
        **os.environ, "SLURM_JOB_ID": job_id, "SLURM_STEP_ID": "0", "SLURM_PROCID": str(rank),
        "SLURM_STEP_NUM_TASKS": "2",
    }
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
    # A new job, so a new token: the first run's markers must not be taken for its.
    for job_id in ("7", "8"):
        trier = _task(1, tmp_path, job_id)
        holder = _task(0, tmp_path, job_id)
        assert holder.wait(timeout=60) == 0
        assert trier.wait(timeout=60) == 0


def test_a_step_of_one_task_stops_at_once_instead_of_waiting(tmp_path: Path) -> None:
    # srun from a shell inside a one-node allocation: one task, which used to wait
    # two minutes for its partner (MDC, 2026-10-29).
    env = {
        **os.environ, "SLURM_JOB_ID": "7", "SLURM_STEP_ID": "0", "SLURM_PROCID": "0",
        "SLURM_STEP_NUM_TASKS": "1",
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "runs 1 task(s)" in result.stderr
    assert not (tmp_path / "ready").exists()


def test_refuses_to_run_outside_slurm_tasks(tmp_path: Path) -> None:
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"SLURM_PROCID", "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS"}
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "two Slurm tasks" in result.stderr
