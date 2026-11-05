"""scripts/mra_port_run.py: a project computed again and compared, or only set up and
computed, or a new run set up from an export without Dynamo."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from test_dynamo_mra_project import _project
from test_mra_setup import _export

from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mra_port_run.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *arguments], capture_output=True,
                          text=True, check=False)


def test_a_project_only_set_up_is_computed_with_nothing_to_compare(tmp_path: Path) -> None:
    # The fake project holds what Dynamo's setup leaves: settings, starting tables and
    # every iteration's card, and no iteration computed.
    project, _ = _project(tmp_path)
    result = _run(str(project), "--device", "numpy")
    assert result.returncode == 0, result.stderr
    assert "mra_t was set up and not computed by Dynamo: nothing to compare" in result.stdout
    assert (tmp_path / "mra_t_port" / "log.txt").read_text() == "iteration 1 ok\niteration 2 ok\n"
    # Each iteration's time, and how much of it was the search.
    for number in (1, 2):
        assert re.search(rf"^iteration {number}: class 1 \d+, class 2 \d+; \d+\.\d min, "
                         r"\d+\.\d of it searching$", result.stdout, re.MULTILINE), result.stdout
    again = _run(str(project), "--compare-only")
    assert again.returncode != 0 and "nothing to compare" in again.stderr
    # Submitted again, as a job out of time is: the run is there, finished.
    again = _run(str(project), "--device", "numpy")
    assert again.returncode == 0, again.stderr
    assert "mra_t_port: all 2 iterations finished before; nothing to compute" in again.stdout


def test_a_new_run_is_set_up_from_an_export_without_dynamo(tmp_path: Path) -> None:
    export = _export(tmp_path)
    arguments = ("--export", str(export), "--name", "mra_n_port", "--seeds", "noise noise",
                 "--mask-radius", "5", "--schedule", "coarse_to_fine", "--first-lowpass", "50",
                 "--iterations", "1", "--device", "numpy")
    result = _run(*arguments)
    assert result.returncode == 0, result.stderr
    assert "seed 1: noise" in result.stdout
    assert ("round 1: 1 iteration(s), low-pass 3 px (53.3 A), cone and in-plane 31.2 deg in "
            "10.4 deg steps") in result.stdout
    assert "mra_n_port was set up from" in result.stdout
    run = export / "mra_n_port"
    assert (run / "log.txt").read_text() == "iteration 1 ok\n"
    setup = json.loads((run / "settings" / "setup.json").read_text())
    assert setup["variables"]["seeds"] == ["noise", "noise"]
    # Submitted again: the plan made again is the run's, and it is finished.
    again = _run(*arguments)
    assert again.returncode == 0, again.stderr
    assert "mra_n_port: all 1 iterations finished before; nothing to compute" in again.stdout
    # The setup's refusals end the script with REFUSED, not a traceback.
    refused = _run("--export", str(export), "--name", "x", "--seeds", "subset subset",
                   "--device", "numpy")
    assert refused.returncode == 1
    assert refused.stderr.strip() == ("REFUSED: the setup: 12 particles cannot give 2 disjoint "
                                      "subsets of 300")
    both = _run(str(export / "mra_t"), "--export", str(export), "--seeds", "noise noise")
    assert both.returncode == 2 and "one of them" in both.stderr


def test_a_computed_project_is_run_again_and_compared(tmp_path: Path) -> None:
    # A computed project, made by the port itself: the same code again agrees everywhere.
    project, _ = _project(tmp_path)
    run_mra_project(plan_from_dynamo(project), tmp_path / "computed", log=lambda _: None)
    result = _run(str(tmp_path / "computed"), "--device", "numpy", "--name", "again")
    assert result.returncode == 0, result.stderr
    assert "again against computed:" in result.stdout
    for number in (1, 2):
        assert (f"iteration {number}: same class 12 of 12 (100.0%); their poses: rotation "
                "median 0.000") in result.stdout
    assert "classes with chance levels" in result.stdout
