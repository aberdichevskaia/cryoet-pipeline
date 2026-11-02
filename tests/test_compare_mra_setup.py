"""scripts/compare_mra_setup.py: the variables a Dynamo project was set up with, read
as the setup of its day meant them."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from test_mra_setup import _export

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_mra_setup.py"


@pytest.fixture
def compare(monkeypatch):
    spec = importlib.util.spec_from_file_location("compare_mra_setup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _log(path: Path, command: str, ready: str) -> Path:
    path.write_text(f"Executing command:\n  {command}eval(fileread('/repo/scripts/"
                    f"dynamo_mra_setup.m'))\nseed 1: noise\ndynamo_mra_setup: project p is "
                    f"ready, {ready}\n")
    return path


def test_the_command_s_variables_are_read_as_given(compare, tmp_path: Path) -> None:
    log = _log(tmp_path / "p_setup.log",
               "project = 'p'; seeds = {'subset', 'blob', 'noise'}; cores = 32; schedule = "
               "'coarse_to_fine'; first_lowpass = 53.76; cone_steps = [5 5 3 3 3]; "
               "global_full_box = true; mask_file = '/x/m.mrc'; ",
               "seeds subset blob noise, schedule coarse_to_fine (angstrom steps), 0 global "
               "round(s) in each round's own step, cones [5 5 3 3 3] step(s) wide")
    variables, notes = compare.setup_variables(log)
    assert variables == {
        "project": "p", "seeds": ["subset", "blob", "noise"], "cores": 32.0,
        "schedule": "coarse_to_fine", "first_lowpass": 53.76, "cone_steps": [5, 5, 3, 3, 3],
        "global_full_box": True, "mask_file": "/x/m.mrc", "lowpass_steps": "angstrom"}
    assert notes == []


def test_a_setup_before_2026_10_19_stepped_in_frequency(compare, tmp_path: Path) -> None:
    log = _log(tmp_path / "p_setup.log", "project = 'p'; seeds = {'blob', 'blob', 'noise'}; "
               "cores = 32; schedule = 'coarse_to_fine'; ",
               "seeds blob blob noise, schedule coarse_to_fine, 32 cores")
    variables, notes = compare.setup_variables(log)
    assert variables["lowpass_steps"] == "fourier"
    assert notes == ["set up before 2026-10-19: low-passes in even steps of frequency"]


def test_a_setup_of_2026_10_22_searched_global_rounds_on_its_grid(compare, tmp_path) -> None:
    log = _log(tmp_path / "p_setup.log", "project = 'p'; seeds = {'blob', 'blob', 'noise'}; "
               "cores = 32; schedule = 'coarse_to_fine'; global_rounds = 2; ",
               "seeds blob blob noise, schedule coarse_to_fine (fourier steps), 2 global "
               "round(s) in 45 deg steps, mask sphere 18 px")
    variables, notes = compare.setup_variables(log)
    assert (variables["global_step"], variables["global_full_box"]) == (45.0, True)
    assert notes == ["set up 2026-10-22 to 10-24: global rounds on a 45 deg grid at the "
                     "full box"]
    # Without global rounds that version's line says the same, and means nothing.
    _log(log, "project = 'p'; seeds = {'blob', 'blob', 'noise'}; cores = 32; ",
         "seeds blob blob noise, schedule fixed (angstrom steps), 0 global round(s) in 45 deg "
         "steps")
    variables, notes = compare.setup_variables(log)
    assert "global_step" not in variables and notes == []


def test_a_gpu_setup_binned_the_global_rounds_by_powers_of_2(compare, tmp_path) -> None:
    # 16 px box: the CPU's dim 10 for low-pass 3 does not fit in 8, so the GPU kept 16.
    export = _export(tmp_path)
    options = {"schedule": "coarse_to_fine", "first_lowpass": 50.0, "last_lowpass": 22.0,
               "global_rounds": 1}
    assert compare._gpu_dims(options, export) == [16]
