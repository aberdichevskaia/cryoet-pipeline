"""scripts/mra_port_run.py: a project computed again and compared, or only set up and
computed, or a new run set up from an export without Dynamo; each written in the
run-record layout, and a run begun in Dynamo's layout carried on in it."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from test_dynamo_mra_project import _project
from test_mra_setup import _export

from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project
from cryoet_pipeline.runs import completed

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
    run = tmp_path / "mra_t_port"
    assert all(completed(run / node) is not None for node in ("setup", "ite_0001", "ite_0002"))
    assert not (run / "log.txt").exists() and not (run / "settings").exists()
    # Each iteration's time, and how much of it was the search.
    for number in (1, 2):
        assert re.search(rf"^ite_{number:04d}: class 1 \d+, class 2 \d+; \d+\.\d min, "
                         r"\d+\.\d of it searching$", result.stdout, re.MULTILINE), result.stdout
    again = _run(str(project), "--compare-only")
    assert again.returncode != 0 and "nothing to compare" in again.stderr
    # Submitted again, as a job out of time is: the run is there, finished.
    again = _run(str(project), "--device", "numpy")
    assert again.returncode == 0, again.stderr
    assert "mra_t_port: all 2 iterations finished before; nothing to compute" in again.stdout


def test_a_new_run_is_set_up_from_an_export_without_dynamo(tmp_path: Path) -> None:
    export = _export(tmp_path)
    arguments = ("--export", str(export), "--name", "mra_n", "--seeds", "noise noise",
                 "--mask-radius", "5", "--schedule", "coarse_to_fine", "--first-lowpass", "50",
                 "--shift-limit", "2", "--iterations", "1", "--device", "numpy")
    result = _run(*arguments)
    assert result.returncode == 0, result.stderr
    assert "seed 1: noise" in result.stdout
    assert ("round 1: 1 iteration(s), low-pass 3 px (53.3 A), cone and in-plane 31.2 deg in "
            "10.4 deg steps") in result.stdout
    assert "mra_n was set up from" in result.stdout
    run = export / "mra_n"
    assert completed(run / "ite_0001") is not None
    asked = json.loads((run / "plan.json").read_text())
    setup = asked["setup"]
    assert setup["variables"]["seeds"] == ["noise", "noise"]
    assert asked["iterations"][0]["search"]["shift_limit"] == 2.0
    assert "shifts: up to 2 px (20.0 A) from the box centre in every round" in result.stdout
    assert completed(run / "ite_0001").execution_params == {"kernel": "numpy", "workers": 1}
    # Submitted again: the plan made again is the run's, and it is finished.
    again = _run(*arguments)
    assert again.returncode == 0, again.stderr
    assert "mra_n: all 1 iterations finished before; nothing to compute" in again.stdout
    # A folder that is not a run of the pipeline's, a Dynamo project say, is refused.
    other = _run(*("mra_t" if argument == "mra_n" else argument for argument in arguments))
    assert other.returncode == 1
    assert other.stderr.startswith(f"REFUSED: {export / 'mra_t'} holds files but no plan.json")
    # Anti-aliased: the plan names it in the global round, and the log says so.
    shrunk = _run("--export", str(export), "--name", "mra_a", "--seeds", "noise noise",
                  "--mask-radius", "5", "--schedule", "coarse_to_fine", "--first-lowpass", "50",
                  "--global-rounds", "1", "--global-step", "60", "--antialias",
                  "--iterations", "1", "--device", "numpy")
    assert shrunk.returncode == 0, shrunk.stderr
    assert "particles at 10 px of 16 (low-passed to its Nyquist first)" in shrunk.stdout
    searches = [item["search"] for item in
                json.loads((export / "mra_a" / "plan.json").read_text())["iterations"]]
    assert [search.get("antialias", False) for search in searches] == [True]
    alone = _run("--export", str(export), "--name", "mra_b", "--seeds", "noise noise",
                 "--antialias", "--device", "numpy")
    assert alone.returncode == 1 and "antialias changes only global rounds" in alone.stderr
    # The setup's refusals end the script with REFUSED, not a traceback.
    refused = _run("--export", str(export), "--name", "x", "--seeds", "subset subset",
                   "--device", "numpy")
    assert refused.returncode == 1
    assert refused.stderr.strip() == ("REFUSED: the setup: 12 particles cannot give 2 disjoint "
                                      "subsets of 300")
    both = _run(str(export / "mra_t"), "--export", str(export), "--seeds", "noise noise")
    assert both.returncode == 2 and "one of them" in both.stderr


def test_a_computed_project_is_run_again_and_compared(tmp_path: Path) -> None:
    # A computed project in Dynamo's layout, made by the port itself: the same code again,
    # in the run-record layout, agrees everywhere.
    project, _ = _project(tmp_path)
    run_mra_project(plan_from_dynamo(project), tmp_path / "computed", log=lambda _: None)
    result = _run(str(tmp_path / "computed"), "--device", "numpy", "--name", "again")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "again" / "plan.json").is_file()
    assert "again against computed:" in result.stdout
    for number in (1, 2):
        assert (f"iteration {number}: same class 12 of 12 (100.0%); their poses: rotation "
                "median 0.000") in result.stdout
    assert "classes with chance levels" in result.stdout


def test_a_run_begun_in_dynamos_layout_is_carried_on_in_it(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    run = tmp_path / "mra_t_port"
    run_mra_project(plan_from_dynamo(project, iterations=1), run, log=lambda _: None)
    result = _run(str(project), "--device", "numpy")
    assert result.returncode == 0, result.stderr
    assert ("mra_t_port was begun in Dynamo's layout (before 2026-11-05) and is carried on in "
            "it") in result.stdout
    assert (run / "log.txt").read_text() == "iteration 1 ok\niteration 2 ok\n"
    assert not (run / "plan.json").exists()
