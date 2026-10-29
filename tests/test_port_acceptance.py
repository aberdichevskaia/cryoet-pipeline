"""scripts/port_acceptance.py: a computed project replayed iteration by iteration, and judged."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_classification_report import write_em
from test_dynamo_mra_project import _project

from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project
from cryoet_pipeline.reports.dynamo_project import read_em, read_table

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "port_acceptance.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-W", "ignore", str(SCRIPT), *arguments],
                          capture_output=True, text=True, check=False)


def _computed(root: Path, **card_changes: object) -> Path:
    """A project the port computed: it holds no starting values after iteration 1 and no
    prepared references, so the replay takes them from the iteration before, as Dynamo does."""

    project, _ = _project(root, **card_changes)
    run_mra_project(plan_from_dynamo(project), root / "computed", log=lambda _: None)
    return root / "computed"


@pytest.mark.parametrize("device", [("numpy",), ("cpu", "--dtype", "float64")],
                         ids=["numpy", "torch"])
def test_a_project_passes_against_the_code_that_computed_it(tmp_path: Path, device) -> None:
    # Dynamo's dim 12 for the 16-voxel box: resized, searched and brought back.
    computed = _computed(tmp_path, dim=12)
    result = _run(str(computed), "--particles", "0", "--workers", "1", "--device", *device)
    assert result.returncode == 0, result.stdout + result.stderr
    for number in (1, 2):
        assert (f"iteration {number}: 10 deg cone in 5 deg steps, refine 1, low 6 px, dim 12 of "
                "16; channels [1, 2]; 12 particles") in result.stdout
    assert result.stdout.count("24 searches: rotation median 0.000") == 2
    assert "sample's classes: 12 of 12 as Dynamo's" in result.stdout
    assert result.stdout.count("averaged from Dynamo's poses") == 4
    assert "correlation 1.0000000, std ratio 1.00000" in result.stdout
    assert result.stdout.rstrip().splitlines()[-1].startswith("PASS: computed, iterations 1, 2")


def test_poses_the_port_does_not_reach_fail_it(tmp_path: Path) -> None:
    # Dynamo's answer turned 3 deg about z in channel 1: the replay ends where the code
    # finds the peak, scores less at the turned pose than the table's CC says, and fails.
    computed = _computed(tmp_path)
    path = computed / "results" / "ite_0001" / "averages" / "refined_table_ref_001_ite_0001.tbl"
    rows = read_table(path)
    rows[:, 6] += 3
    path.write_text("\n".join(" ".join(f"{value:.12g}" for value in row) for row in rows) + "\n")
    result = _run(str(computed), "--particles", "0", "--workers", "1", "--iterations", "1",
                  "--skip-averages")
    assert result.returncode == 1
    assert "12 of 24 searches end more than 0.1 deg or 0.1 px from Dynamo's pose" in result.stdout
    assert "12 do not reproduce Dynamo's CC at its pose" in result.stdout
    assert result.stdout.rstrip().splitlines()[-1].startswith("FAIL: iteration 1:")


def test_an_average_the_port_does_not_make_fails_it(tmp_path: Path) -> None:
    computed = _computed(tmp_path)
    path = computed / "results" / "ite_0002" / "averages" / "average_ref_002_ite_0002.em"
    write_em(path, (1.01 * read_em(path)).astype(np.float32))
    result = _run(str(computed), "--particles", "4", "--workers", "1", "--iterations", "2")
    assert result.returncode == 1
    assert "FAIL: iteration 2: class 2's average from Dynamo's poses: correlation 1.0000000, " \
           "std ratio 0.99010" in result.stdout


def test_three_ccs_tell_a_tie_from_a_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("port_acceptance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Its dataclass looks its own module up while the module loads.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    kind = module._kind
    assert kind(0.50000, 0.50003, 0.49998) == "tie"
    assert kind(0.52, 0.50, 0.50) == "Dynamo short"
    assert kind(0.50, 0.52, 0.48) == "not reproduced"
    assert kind(0.48, 0.50, 0.50) == "missed peak"
    assert kind(0.91, 1.00, 0.80) == "spline at 1"
    assert kind(1.00, 0.91, 0.91) == "spline at 1"
