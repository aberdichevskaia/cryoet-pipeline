"""One job at a time on a run: cryoet_pipeline.run_lock, scripts/hold_run_lock.py, and
scripts/mra_port_run.sbatch running its whole job under the lock.

On 2026-10-29 two jobs carrying mra_pad_ssn_g2atan2_port on, submitted a minute
apart, computed iterations 2-5 into the same files for two hours. Here the jobs
are processes on one machine, where a POSIX lock keeps them out as it keeps out a
job on another node on MDC's GPFS (scripts/lock_probe.py).
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from cryoet_pipeline import run_lock as run_lock_module
from cryoet_pipeline.run_lock import RunLockError, run_lock

ROOT = Path(__file__).resolve().parents[1]
HOLDER = ROOT / "scripts" / "hold_run_lock.py"

# A command that says it started and waits for leave to go.
WAITS = (
    "import pathlib, sys, time\n"
    "here = pathlib.Path(sys.argv[1])\n"
    "(here / 'started').write_text('')\n"
    "while not (here / 'release').exists():\n"
    "    time.sleep(0.05)\n"
)

# A command that, sent SIGTERM, takes its time to end.
ENDS_SLOWLY = (
    "import pathlib, signal, sys, time\n"
    "here = pathlib.Path(sys.argv[1])\n"
    "def stop(number, frame):\n"
    "    (here / 'got term').write_text('')\n"
    "    while not (here / 'release').exists():\n"
    "        time.sleep(0.05)\n"
    "    sys.exit(0)\n"
    "signal.signal(signal.SIGTERM, stop)\n"
    "(here / 'started').write_text('')\n"
    "while True:\n"
    "    time.sleep(0.05)\n"
)


def _env(job: str | None = None, name: str | None = None, **more: str) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    env["PYTHONPATH"] = str(ROOT / "src")
    if job is not None:
        env["SLURM_JOB_ID"] = job
    if name is not None:
        env["SLURM_JOB_NAME"] = name
    return {**env, **more}


def _hold(lock: Path, *command: str) -> list[str]:
    return [sys.executable, str(HOLDER), str(lock), "--", *command]


def _wait_until(condition: Callable[[], bool], process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while not condition():
        if process.poll() is not None:
            out, err = process.communicate()
            raise AssertionError(f"ended with {process.returncode} first:\n{out}\n{err}")
        if time.monotonic() > deadline:
            process.kill()
            raise AssertionError("timed out")
        time.sleep(0.05)


def test_a_second_holder_is_refused_at_once_and_names_the_first(tmp_path: Path) -> None:
    lock = tmp_path / "mra_t_port.lock"
    first = subprocess.Popen(
        _hold(lock, sys.executable, "-c", WAITS, str(tmp_path)),
        env=_env("3790159", "mra_t"), text=True,
    )
    try:
        _wait_until((tmp_path / "started").exists, first)
        second = subprocess.run(
            _hold(lock, sys.executable, "-c", f"open({str(tmp_path / 'ran')!r}, 'w')"),
            env=_env("3790178", "mra_t"), capture_output=True, text=True, timeout=60,
        )
        assert second.returncode == 1
        assert (
            f"REFUSED: {tmp_path / 'mra_t_port'} is taken by job 3790159 (mra_t) on "
            f"{socket.gethostname()}, pid {first.pid}, since "
        ) in second.stderr
        assert second.stderr.rstrip().endswith("; this job ran nothing")
        assert not (tmp_path / "ran").exists()
    finally:
        (tmp_path / "release").write_text("")
    assert first.wait(timeout=60) == 0

    # The first has ended: the lock is free, and its file stays, naming the new holder.
    third = subprocess.run(
        _hold(lock, sys.executable, "-c", f"open({str(tmp_path / 'ran')!r}, 'w')"),
        env=_env("3790200"), capture_output=True, text=True, timeout=60,
    )
    assert third.returncode == 0, third.stderr
    assert (tmp_path / "ran").exists()
    assert lock.read_text().startswith(f"job 3790200 on {socket.gethostname()}, pid ")


@pytest.mark.parametrize(
    ("command", "status"),
    [("sys.exit(3)", 3), ("os.kill(os.getpid(), signal.SIGKILL)", 128 + signal.SIGKILL)],
    ids=["exit-status", "signal"],
)
def test_the_holder_ends_as_its_command_did(tmp_path: Path, command: str, status: int) -> None:
    result = subprocess.run(
        _hold(tmp_path / "r.lock", sys.executable, "-c", f"import os, signal, sys; {command}"),
        env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == status, result.stderr


def test_a_signal_to_the_holder_reaches_its_command_and_the_lock_outlives_it(
    tmp_path: Path,
) -> None:
    lock = tmp_path / "r.lock"
    first = subprocess.Popen(
        _hold(lock, sys.executable, "-c", ENDS_SLOWLY, str(tmp_path)), env=_env("1"), text=True
    )
    try:
        _wait_until((tmp_path / "started").exists, first)
        first.send_signal(signal.SIGTERM)  # to the holder alone
        _wait_until((tmp_path / "got term").exists, first)
        # The command is still ending, so the run is still taken.
        second = subprocess.run(
            _hold(lock, sys.executable, "-c", "pass"),
            env=_env("2"), capture_output=True, text=True, timeout=60,
        )
        assert second.returncode == 1
        assert "is taken by job 1 on " in second.stderr
    finally:
        (tmp_path / "release").write_text("")
    assert first.wait(timeout=60) == 0
    third = subprocess.run(
        _hold(lock, sys.executable, "-c", "pass"), env=_env("3"), capture_output=True,
        text=True, timeout=60,
    )
    assert third.returncode == 0, third.stderr


def test_a_lock_that_would_hold_on_this_node_only_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run_lock_module,
        "node_local_lock",
        lambda directory: "/lustre is lustre mounted with localflock, so a POSIX lock there "
        "holds only on this node (n1)",
    )
    with pytest.raises(RunLockError, match="localflock.*on another node would not be kept out"):
        with run_lock(tmp_path / "r.lock"):
            pass


def test_taking_the_lock_again_in_its_process_is_refused_and_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)  # tests run inside allocations too
    lock = tmp_path / "r.lock"
    with run_lock(lock):
        # Taken again, a POSIX lock would be granted, and closing it would release it.
        with pytest.raises(RunLockError, match="held by this process already"):
            with run_lock(lock):
                pass
        other = subprocess.run(
            _hold(lock, sys.executable, "-c", "pass"), env=_env("2"), capture_output=True,
            text=True, timeout=60,
        )
        assert other.returncode == 1
        assert "a process outside SLURM" in other.stderr
    with run_lock(lock):
        pass


# The job scripts in a stand-in repo, scripts/dynamo_mra_mdc.sbatch with stand-ins
# for Dynamo on PATH. Dynamo's setup, its computation and the port note what they
# were asked, and whether the pipeline's env was active for them, and wait for a
# release file, so that a second job can start while the first is on either.
DYNAMO = r"""#!/bin/bash
# The setup, `dynamo "project = '...'; ...; cores = N; ..."` in the export folder.
project=$(sed -n "s/^project = '\([^']*\)'.*/\1/p" <<< "$1")
cores=$(sed -n 's/.*cores = \([0-9]*\);.*/\1/p' <<< "$1")
echo "dynamo $SLURM_JOB_ID project=$project cores=$cores env=${PIPELINE_ENV:-none}" >> "$CALLS"
while [[ ! -e "$SETUP_RELEASE" ]]; do sleep 0.05; done
mkdir -p "$project/cards/ite_0001"
echo "how_many_processors=$cores;" > "$project/cards/ite_0001/card_ite_ite_0001.card"
"""

DYNAMO_EXECUTE = """#!/bin/bash
echo "compute $SLURM_JOB_ID $1 env=${PIPELINE_ENV:-none}" >> "$CALLS"
while [[ ! -e "$RELEASE" ]]; do sleep 0.05; done
echo "iteration 1 ok"
"""

PORT = """import os, pathlib, sys, time
with open(os.environ["CALLS"], "a") as calls:
    env = os.environ.get("PIPELINE_ENV", "none")
    calls.write(f"port {os.environ['SLURM_JOB_ID']} env={env} {' '.join(sys.argv[1:])}\\n")
while not pathlib.Path(os.environ["RELEASE"]).exists():
    time.sleep(0.05)
"""


def _stand_in_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("mra_port_run.sbatch", "dynamo_mra_mdc.sbatch", "hold_run_lock.py"):
        shutil.copy(ROOT / "scripts" / name, scripts / name)
    (scripts / "activate_env.sh").write_text("ENV_MANAGER=stub\nexport PIPELINE_ENV=active\n")
    (scripts / "mra_port_run.py").write_text(PORT)
    for name in ("dynamo_mra_setup.m", "compare_dynamo_classes.py",
                 "dynamo_classification_report.py"):
        (scripts / name).write_text("")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python").symlink_to(sys.executable)
    stand_ins = (("nvidia-smi", "#!/bin/sh\nexit 0\n"), ("dynamo", DYNAMO),
                 ("dynamo_execute_project", DYNAMO_EXECUTE))
    for name, script in stand_ins:
        (tools / name).write_text(script)
        (tools / name).chmod(0o755)
    (tmp_path / "dynamo_root").mkdir()
    (tmp_path / "dynamo_root" / "dynamo_activate_linux_shipped_MCR.sh").write_text("")
    (tmp_path / "export").mkdir()
    return repo


def _job(
    repo: Path, job: str, *, script: str = "mra_port_run.sbatch", by_hand: bool = False,
    **env: str,
) -> subprocess.Popen[str]:
    """A job script as SLURM runs it, by an absolute path, or by hand from the repo root."""
    here = repo.parent
    path = f"scripts/{script}"
    if not by_hand:
        path = str(repo / path)
    return subprocess.Popen(
        ["bash", path], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={
            "PATH": f"{here / 'tools'}{os.pathsep}{os.environ['PATH']}", "HOME": str(here),
            "PYTHONPATH": str(ROOT / "src"), "SLURM_SUBMIT_DIR": str(repo),
            "SLURM_JOB_NAME": "mra_t", "SLURM_JOB_ID": job,
            "DYNAMO_ROOT": str(here / "dynamo_root"), "SCRATCH_ROOT": str(here / "scratch"),
            "CALLS": str(here / "calls"), "RELEASE": str(here / "release"),
            "SETUP_RELEASE": str(here / "setup_release"), **env,
        },
    )


def _dynamo_job(repo: Path, job: str, **env: str) -> subprocess.Popen[str]:
    return _job(
        repo, job, script="dynamo_mra_mdc.sbatch", EXPORT=str(repo.parent / "export"),
        NAME="mra_t", SEEDS="subset subset noise", SLURM_CPUS_PER_TASK="4", **env,
    )


def _calls(repo: Path) -> list[str]:
    path = repo.parent / "calls"
    return path.read_text().splitlines() if path.exists() else []


def _port_call(job: str, export: Path) -> str:
    return (
        f"port {job} env=active {export}/mra_t --name mra_t_port --device cuda "
        "--dtype float32 --per-gpu 1 --workers 1"
    )


def test_a_second_job_of_a_run_stops_before_the_port_and_names_the_first(
    tmp_path: Path,
) -> None:
    repo = _stand_in_repo(tmp_path)
    project = tmp_path / "export" / "mra_t"
    project.mkdir()
    first = _job(repo, "3790159", PROJECT=str(project))
    try:
        _wait_until(lambda: any(call.startswith("port 3790159") for call in _calls(repo)), first)
        # Another run's lock named in its environment does not let a job past this one.
        second = _job(repo, "3790178", PROJECT=str(project), MRA_PORT_RUN_LOCK="/else.lock")
        out, err = second.communicate(timeout=60)
        assert second.returncode == 1
        assert f"REFUSED: {project}_port is taken by job 3790159 (mra_t) on " in err
        assert [call.split()[1] for call in _calls(repo)] == ["3790159"]
        assert "port run" not in out
    finally:
        (tmp_path / "release").write_text("")
    out, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert f"lock        {project}_port.lock, held by this job to its end" in out

    # Submitted again once the first has ended, it runs.
    third = _job(repo, "3790200", PROJECT=str(project))
    out, err = third.communicate(timeout=60)
    assert third.returncode == 0, err
    assert [call.split()[1] for call in _calls(repo)] == ["3790159", "3790200"]


def test_a_second_job_started_during_the_first_job_s_setup_is_refused_too(
    tmp_path: Path,
) -> None:
    repo = _stand_in_repo(tmp_path)
    export = tmp_path / "export"
    (tmp_path / "release").write_text("")  # the port does not wait here
    first = _job(
        repo, "1", by_hand=True, EXPORT=str(export), NAME="mra_t", SEEDS="subset subset noise"
    )
    try:
        _wait_until(lambda: _calls(repo) == ["dynamo 1 project=mra_t cores=32 env=none"], first)
        second = _job(repo, "2", EXPORT=str(export), NAME="mra_t", SEEDS="subset subset noise")
        _, err = second.communicate(timeout=60)
        assert second.returncode == 1
        assert f"REFUSED: {export}/mra_t_port is taken by job 1 (mra_t) on " in err
        assert _calls(repo) == ["dynamo 1 project=mra_t cores=32 env=none"]
    finally:
        (tmp_path / "setup_release").write_text("")
    _, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    # Dynamo's setup (SETUP_ONLY: nothing computed) ran in the job's own environment,
    # the port in the pipeline's.
    assert _calls(repo) == ["dynamo 1 project=mra_t cores=32 env=none", _port_call("1", export)]


def test_a_second_dynamo_job_of_a_project_is_refused_while_the_first_computes_it(
    tmp_path: Path,
) -> None:
    repo = _stand_in_repo(tmp_path)
    export = tmp_path / "export"
    (tmp_path / "setup_release").write_text("")
    first = _dynamo_job(repo, "11")
    try:
        _wait_until(lambda: "compute 11 mra_t env=none" in _calls(repo), first)
        second = _dynamo_job(repo, "12")
        _, err = second.communicate(timeout=60)
        assert second.returncode == 1
        assert f"REFUSED: {export}/mra_t is taken by job 11 (mra_t) on " in err
        assert _calls(repo) == [
            "dynamo 11 project=mra_t cores=4 env=none", "compute 11 mra_t env=none",
        ]
    finally:
        (tmp_path / "release").write_text("")
    out, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert f"report      {export}/mra_t_classification_report.pdf" in out

    # Submitted again once the first has ended, it carries the project on.
    third = _dynamo_job(repo, "13")
    _, err = third.communicate(timeout=60)
    assert third.returncode == 0, err
    assert _calls(repo)[-1] == "compute 13 mra_t env=none"


def test_a_second_dynamo_job_is_refused_during_the_first_job_s_setup(tmp_path: Path) -> None:
    repo = _stand_in_repo(tmp_path)
    export = tmp_path / "export"
    (tmp_path / "release").write_text("")
    first = _job(
        repo, "11", script="dynamo_mra_mdc.sbatch", by_hand=True, EXPORT=str(export),
        NAME="mra_t", SEEDS="subset subset noise", SLURM_CPUS_PER_TASK="4",
    )
    try:
        _wait_until(lambda: _calls(repo) == ["dynamo 11 project=mra_t cores=4 env=none"], first)
        second = _dynamo_job(repo, "12")
        _, err = second.communicate(timeout=60)
        assert second.returncode == 1
        assert f"REFUSED: {export}/mra_t is taken by job 11 (mra_t) on " in err
        assert _calls(repo) == ["dynamo 11 project=mra_t cores=4 env=none"]
    finally:
        (tmp_path / "setup_release").write_text("")
    _, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert _calls(repo) == [
        "dynamo 11 project=mra_t cores=4 env=none", "compute 11 mra_t env=none",
    ]


def test_a_port_job_does_not_set_up_a_project_a_dynamo_job_is_setting_up(
    tmp_path: Path,
) -> None:
    repo = _stand_in_repo(tmp_path)
    export = tmp_path / "export"
    (tmp_path / "release").write_text("")
    dynamo = _dynamo_job(repo, "21")
    try:
        _wait_until(lambda: _calls(repo) == ["dynamo 21 project=mra_t cores=4 env=none"], dynamo)
        port = _job(repo, "22", EXPORT=str(export), NAME="mra_t", SEEDS="subset subset noise")
        _, err = port.communicate(timeout=60)
        assert port.returncode == 1
        assert f"REFUSED: {export}/mra_t is taken by job 21 (mra_t) on " in err
        assert _calls(repo) == ["dynamo 21 project=mra_t cores=4 env=none"]
    finally:
        (tmp_path / "setup_release").write_text("")
    _, err = dynamo.communicate(timeout=60)
    assert dynamo.returncode == 0, err
