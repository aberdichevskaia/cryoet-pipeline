from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_relion.py"


def _script():
    spec = importlib.util.spec_from_file_location("export_relion_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tomogram_binning, raw_pixel, expected",
    [
        (8, 0.675, 2),  # super-resolution sums to the camera's 1.35 A
        (4, 1.68, 1),  # already at the camera's pixel
        (8, 0.4, 4),  # 1.6 A; 3 would be 1.2 A but does not divide 8
        (2, 0.3, 2),  # never coarser than the tomogram
        (6, 0.4, 3),
    ],
)
def test_default_stack_binning_reaches_an_angstrom_dividing_the_tomogram(
    tomogram_binning: int, raw_pixel: float, expected: int
) -> None:
    assert _script().default_stack_binning(tomogram_binning, raw_pixel) == expected


def test_class_spec_reads_run_and_class() -> None:
    script = _script()
    assert script._class_spec("outputs/p/mra/run2:3") == (Path("outputs/p/mra/run2"), 3)
    with pytest.raises(argparse.ArgumentTypeError):
        script._class_spec("outputs/p/mra/run2")


def _record(**changes):
    record = {
        "recipe": "export-relion",
        "config": "/p/mra/run2",
        "series": "class 1 after iteration 5",
        "project": "/p",
        "particles": 300,
        "duplicates_left_out": 2,
        "stacks": {"TS_01": {"binning": 2, "views": 41, "relion_bin": 2, "notes": []}},
        "out": "/p/relion/run2-class1",
        "relion_check": False,
        "env": {},
    }
    return {**record, **changes}


def _dry_run(record, *, user="anna", minutes_ago=5, event="dry-run"):
    from datetime import UTC, datetime, timedelta

    at = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    return {"event": event, "user": user, "at": at, "plan": record, "problems": []}


def _refusal(record, rows, *, assistant=True, reason="Anna asked for class 1"):
    from datetime import UTC, datetime

    who = {"user": "anna", "assistant": assistant}
    return _script().refusal(record, who, reason, rows, datetime.now(UTC))


def test_a_person_exports_without_a_dry_run_on_record() -> None:
    assert _refusal(_record(), [], assistant=False, reason=None) is None


def test_the_assistant_exports_only_the_dry_run_it_showed() -> None:
    record = _record()
    assert _refusal(record, [_dry_run(record)]) is None
    assert "needs --reason" in _refusal(record, [_dry_run(record)], reason=" ")
    assert "no dry run" in _refusal(record, [])
    assert "minutes old" in _refusal(record, [_dry_run(record, minutes_ago=45)])
    assert "anna2's" in _refusal(record, [_dry_run(record, user="anna2")])
    changed = _record(particles=299)
    assert "differs" in _refusal(changed, [_dry_run(record)])
    later = _record(series="class 1 after iteration 6")  # an iteration finished since
    assert "no dry run" in _refusal(later, [_dry_run(record)])
    used = [_dry_run(record, minutes_ago=6), _dry_run(record, minutes_ago=1, event="submit")]
    assert "submitted" in _refusal(record, used)


def test_the_commands_load_relion_check_it_and_use_the_gpus(tmp_path) -> None:
    import subprocess
    from types import SimpleNamespace

    plan = SimpleNamespace(
        chosen=SimpleNamespace(box=48, pixel_angstrom=5.4, channel=1),
        series=[SimpleNamespace(relion_bin=4)],
    )
    text = _script().commands_text(plan, diameter=130.0, ini_high=40.0)
    script = tmp_path / "relion_commands.sh"
    script.write_text(text)
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0
    assert "module load $RELION5_MODULE" in text and 'PATH="$RELION5_HOME/bin:$PATH"' in text
    assert "--b 96 --crop 48 --bin 4" in text and "reference/class_1.mrc" in text
    assert '${GPU_ARGS[@]+"${GPU_ARGS[@]}"}' in text
    assert "--particle_diameter 130 " in text and "--ini_high 40 " in text
    # Without RELION it stops before anything runs, saying what is missing.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "relion_tomo_subtomo not found" in result.stderr
    assert not (tmp_path / "Extract").exists()


def _stubs(folder, *, refine_starts=True):
    """Stand-ins for RELION and mpirun that log their arguments."""

    folder.mkdir(parents=True)
    log = folder / "calls.log"
    for name in ("relion_tomo_subtomo", "relion_refine_mpi", "mpirun"):
        stub = folder / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n')
        stub.chmod(0o755)
    refine = folder / "relion_refine"
    refine.write_text(
        "#!/bin/sh\necho 'RELION version: 5.0.1-commit-test'\n"
        if refine_starts
        else "#!/bin/sh\necho 'relion_refine: error while loading shared libraries: "
        "libucp.so.0' >&2\nexit 127\n"
    )
    refine.chmod(0o755)
    return log


def _commands(tmp_path):
    from types import SimpleNamespace

    plan = SimpleNamespace(
        chosen=SimpleNamespace(box=48, pixel_angstrom=5.4, channel=1),
        series=[SimpleNamespace(relion_bin=4)],
    )
    script = tmp_path / "export" / "relion_commands.sh"
    script.parent.mkdir()
    script.write_text(_script().commands_text(plan, diameter=130.0, ini_high=40.0))
    return script


def test_a_relion_that_does_not_start_stops_the_commands_first(tmp_path) -> None:
    import subprocess

    script = _commands(tmp_path)
    log = _stubs(tmp_path / "bin", refine_starts=False)
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 2
    assert "does not start" in result.stderr and "libucp.so.0" in result.stderr
    assert not log.exists() and not (script.parent / "Extract").exists()


def test_the_commands_run_extract_then_refine(tmp_path) -> None:
    import subprocess

    script = _commands(tmp_path)
    log = _stubs(tmp_path / "bin")
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path), "USE_GPU": "0"}
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "RELION version: 5.0.1" in result.stdout
    calls = log.read_text().splitlines()
    assert calls[0].startswith("relion_tomo_subtomo --i optimisation_set.star")
    assert calls[1].startswith("mpirun -n 3 relion_refine_mpi --ios Extract/job001/")
    assert "--gpu" not in calls[1]
    assert (script.parent / "Extract" / "job001").is_dir()


def test_relion_can_come_from_an_installation_folder(tmp_path) -> None:
    import subprocess

    script = _commands(tmp_path)
    home = tmp_path / "relion-5.0.1"
    log = _stubs(home / "bin")
    (home / "lib").mkdir()
    subtomo = home / "bin" / "relion_tomo_subtomo"
    subtomo.write_text(f'#!/bin/sh\necho "LD=$LD_LIBRARY_PATH" >> "{log}"\n')
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "USE_GPU": "0",
        "RELION5_HOME": str(home),
        "LD_LIBRARY_PATH": "/opt/mpi/lib",
    }
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[0] == f"LD={home / 'lib'}:/opt/mpi/lib"
    assert calls[1].startswith("mpirun -n 3 relion_refine_mpi")


def test_too_few_slurm_tasks_stop_the_commands_first(tmp_path) -> None:
    import subprocess

    script = _commands(tmp_path)
    log = _stubs(tmp_path / "bin")
    env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "HOME": str(tmp_path), "SLURM_NTASKS": "1"}
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 2 and "--ntasks=3" in result.stderr
    assert not log.exists()


def test_mpirun_keeps_its_session_files_where_it_is_told(tmp_path) -> None:
    import subprocess

    script = _commands(tmp_path)
    log = _stubs(tmp_path / "bin")
    mpirun = tmp_path / "bin" / "mpirun"
    mpirun.write_text(f'#!/bin/sh\necho "PRTE=$PRTE_MCA_prte_tmpdir_base" >> "{log}"\n')
    local = tmp_path / "local"
    env = {
        "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "USE_GPU": "0",
        "SLURM_NTASKS": "3",
        "RELION5_MPI_TMPDIR": str(local),
    }
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert log.read_text().splitlines()[-1] == f"PRTE={local}"


def test_the_submission_hint_names_the_export_folder(tmp_path) -> None:
    folder = tmp_path / "relion" / "run2-class1"
    hint = _script().submit_hint(folder)
    assert f"--chdir={folder.resolve()}" in hint
    assert "--ntasks=3" in hint and '--wrap "./relion_commands.sh"' in hint


def test_the_diameter_comes_from_the_plan_or_half_the_box(tmp_path) -> None:
    import json

    script = _script()
    dynamo = tmp_path / "mra_pad_ssn_g2atan"
    dynamo.mkdir()
    assert script.particle_diameter(dynamo, 52, 13.44) == 52 * 13.44 / 2
    run = tmp_path / "run2"
    run.mkdir()
    (run / "plan.json").write_text(
        json.dumps({"setup": {"variables": {"diameter_angstrom": 340.0}}})
    )
    assert script.particle_diameter(run, 52, 13.44) == 340.0
