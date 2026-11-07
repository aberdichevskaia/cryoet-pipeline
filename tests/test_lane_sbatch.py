"""scripts/lane.sbatch's job header and preflight, run in a stand-in repo.

The job finds the CTFFIND estimate-ctf will run as the step's backend finds it:
the lane's [executables] ctffind with its $VARs and ~ expanded, relative to the
repo root, or else ctffind5, then ctffind, on PATH. Until 2026-11-08 it looked at
PATH alone, so its header named PATH's CTFFIND for jobs that ran the lane's, and
its preflight would stop a lane that names a CTFFIND on a site without one on PATH.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# run_series.py as lane.sbatch calls it: --list-steps-range lists the stage's
# steps, given here by the test; --execute would run them, and here runs nothing.
RUN_SERIES = """import os
import sys
if "--list-steps-range" in sys.argv:
    print(os.environ["STUB_STEPS"])
"""


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _run_lane(
    tmp_path: Path,
    *,
    config: str = "",
    steps: str = "estimate-ctf",
    on_path: tuple[str, ...] = (),
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run lane.sbatch up to the end of its preflight and its (stand-in) steps.

    PATH holds only the stand-in tools, a python3 that is this interpreter (the
    registry-lock preflight imports cryoet_pipeline), and the system's own bins,
    so that whether a CTFFIND is on PATH is the test's choice.
    """

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "lane.sbatch", repo / "scripts" / "lane.sbatch")
    (repo / "scripts" / "run_series.py").write_text(RUN_SERIES)
    (repo / "scripts" / "activate_env.sh").write_text("CONDA_PREFIX=/env\nENV_MANAGER=stub\n")
    (repo / "lane.toml").write_text(config)
    (repo / "proj").mkdir()
    imod = tmp_path / "imod"
    imod.mkdir()
    (imod / "IMOD-linux.sh").write_text("")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (tools / "python3").chmod(0o755)
    for name in on_path:
        _executable(tools / name)
    return subprocess.run(
        ["bash", str(repo / "scripts" / "lane.sbatch")], cwd=repo, capture_output=True,
        text=True,
        env={
            "PATH": f"{tools}:/usr/bin:/bin", "HOME": str(tmp_path),
            "SLURM_SUBMIT_DIR": str(repo), "SLURM_JOB_NAME": "proj-cpu", "SLURM_JOB_ID": "1",
            "PROJECT": "proj", "CONFIG": "lane.toml", "SERIES": "TS_A",
            "FROM": "build-ctf-stack", "UNTIL": "correct-ctf", "IMOD_DIR": str(imod),
            "STUB_STEPS": steps, **(env or {}),
        },
    )


def _header_ctffind(result: subprocess.CompletedProcess[str]) -> str:
    lines = [line for line in result.stdout.splitlines() if line.startswith("ctffind ")]
    assert len(lines) == 1, result.stdout
    return lines[0].split(None, 1)[1]


def test_a_lane_that_names_its_ctffind_needs_none_on_path(tmp_path):
    ctffind = _executable(tmp_path / "ctffind-5.0.2" / "cisTEM" / "bin" / "ctffind")
    result = _run_lane(
        tmp_path, config='[executables]\nctffind = "$CTFFIND5"\n',
        env={"CTFFIND5": str(ctffind)},
    )
    assert result.returncode == 0, result.stderr
    assert "preflight   ok" in result.stdout
    assert _header_ctffind(result) == f"{ctffind} (the lane's [executables] ctffind)"


def test_the_header_names_the_lanes_ctffind_not_the_one_on_path(tmp_path):
    ctffind = _executable(tmp_path / "ctffind-5.0.2" / "cisTEM" / "bin" / "ctffind")
    result = _run_lane(
        tmp_path, config='[executables]\nctffind = "$CTFFIND5"\n', on_path=("ctffind",),
        env={"CTFFIND5": str(ctffind)},
    )
    assert result.returncode == 0, result.stderr
    assert _header_ctffind(result) == f"{ctffind} (the lane's [executables] ctffind)"


@pytest.mark.parametrize(
    ("value", "made"),
    [("bin/ctffind", "repo/bin/ctffind"), ("~/apps/ctffind", "apps/ctffind")],
    ids=["relative-to-the-repo", "home"],
)
def test_a_named_ctffind_is_found_where_the_step_finds_it(tmp_path, value, made):
    ctffind = _executable(tmp_path / made)
    result = _run_lane(tmp_path, config=f'[executables]\nctffind = "{value}"\n')
    assert result.returncode == 0, result.stderr
    assert _header_ctffind(result).startswith(value.replace("~", str(tmp_path)))
    assert ctffind.is_file()


def test_a_named_ctffind_that_is_a_directory_stops_the_job(tmp_path):
    folder = tmp_path / "ctffind-5.0.2"
    _executable(folder / "cisTEM" / "bin" / "ctffind")
    result = _run_lane(
        tmp_path, config='[executables]\nctffind = "$CTFFIND5"\n', on_path=("ctffind",),
        env={"CTFFIND5": str(folder)},
    )
    assert result.returncode == 1
    assert f"is the directory '{folder}'" in result.stderr
    assert "preflight   ok" not in result.stdout


def test_a_named_ctffind_whose_variable_is_not_set_stops_the_job(tmp_path):
    result = _run_lane(
        tmp_path, config='[executables]\nctffind = "$CTFFIND5"\n', on_path=("ctffind",),
    )
    assert result.returncode == 1
    assert "'$CTFFIND5' here: a variable in it is not set" in result.stderr


def test_a_named_ctffind_that_is_not_executable_stops_the_job(tmp_path):
    ctffind = _executable(tmp_path / "ctffind")
    ctffind.chmod(0o644)
    result = _run_lane(
        tmp_path, config='[executables]\nctffind = "$CTFFIND5"\n',
        env={"CTFFIND5": str(ctffind)},
    )
    assert result.returncode == 1
    assert f"'{ctffind}' here, which is not an executable file" in result.stderr


@pytest.mark.parametrize(
    ("on_path", "found"),
    [(("ctffind5", "ctffind"), "ctffind5"), (("ctffind",), "ctffind")],
    ids=["ctffind5-first", "ctffind"],
)
def test_a_lane_that_names_no_ctffind_takes_the_one_on_path(tmp_path, on_path, found):
    result = _run_lane(tmp_path, on_path=on_path)
    assert result.returncode == 0, result.stderr
    assert _header_ctffind(result) == str(tmp_path / "tools" / found)


def test_no_ctffind_anywhere_stops_the_job(tmp_path):
    result = _run_lane(tmp_path)
    assert result.returncode == 1
    assert "no ctffind5 or ctffind on PATH, and the lane names no [executables] ctffind" in (
        result.stderr
    )
    assert _header_ctffind(result) == "not on PATH"


def test_a_stage_without_estimate_ctf_does_not_check_ctffind(tmp_path):
    result = _run_lane(
        tmp_path, config='[executables]\nctffind = "$CTFFIND5"\n',
        steps="build-final-aligned-stack\ncorrect-ctf",
    )
    assert result.returncode == 0, result.stderr
    assert "preflight   ok" in result.stdout
