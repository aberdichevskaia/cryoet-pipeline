"""cryoet_pipeline.mra_runs: an MRA run as a plan of nodes with run records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_dynamo_mra_project import _project
from test_mra_setup import _export, _plan

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project
from cryoet_pipeline.mra_runs import (
    STEP_ITERATION,
    STEP_SETUP,
    Refused,
    plan_record,
    read_state,
    run_plan,
)
from cryoet_pipeline.particle_tables import read_particle_table
from cryoet_pipeline.reports.dynamo_project import read_em, read_table
from cryoet_pipeline.runs import completed, read_record

QUIET = {"log": lambda _: None}


def _short(plan, iterations=2):
    return replace(plan, settings=plan.settings[:iterations], cards=plan.cards[:iterations],
                   mra=plan.mra[:iterations])


def test_a_plan_is_written_as_nodes_each_with_its_record(tmp_path: Path) -> None:
    plan = _short(_plan(_export(tmp_path)))
    run = tmp_path / "mra_x"
    run_plan(plan, run, execution={"device": "numpy"}, **QUIET)

    asked = json.loads((run / "plan.json").read_text())
    assert [item["iteration"] for item in asked["iterations"]] == [1, 2]
    assert asked["setup"]["variables"]["seeds"] == ["subset", "blob", "noise"]
    setup = completed(run / "setup")
    assert setup is not None and setup.step == STEP_SETUP
    assert setup.content_params == plan.setup["variables"]
    assert [item.slot for item in setup.inputs] == ["export"]
    assert sorted(str(output.path) for output in setup.outputs) == [
        "mask.mrc", "particles.parquet", "reference_1.mrc", "reference_2.mrc",
        "reference_3.mrc"]
    columns, metadata = read_particle_table(run / "setup" / "particles.parquet")
    assert columns["particle_id"].tolist() == list(range(1, 13))
    np.testing.assert_array_equal(columns["tdrot_deg"], plan.table[:, 6])
    assert sorted(columns["seed_channel"].tolist()) == [0] * 4 + [1] * 4 + [2] * 4
    assert metadata["box_px"] == 16 and "tdrot, tilt, narot" in metadata["angles"]

    first, second = completed(run / "ite_0001"), completed(run / "ite_0002")
    assert first is not None and second is not None and first.step == STEP_ITERATION
    assert {item.slot: item.fingerprint for item in first.inputs}["previous"] == setup.fingerprint
    assert {item.slot: item.fingerprint for item in second.inputs}["previous"] \
        == first.fingerprint
    # Every setting of the search, the anti-aliasing only when on.
    assert first.content_params["search"] == {
        name: value for name, value in vars(plan.settings[0]).items() if name != "antialias"}
    assert first.execution_params == {"device": "numpy"}
    sizes = {name: value for name, value in second.metrics.items() if name.startswith("class_")}
    assert sum(sizes.values()) == 12
    assert [selection.name for selection in second.selections] == [
        f"class {name.split('_')[1]}" for name in sizes]
    assert sorted(path.name for path in run.iterdir()) == [
        "ite_0001", "ite_0002", "plan.json", "setup"]


def test_the_new_layout_computes_what_the_dynamo_layout_did(tmp_path: Path) -> None:
    plan = _short(_plan(_export(tmp_path)))
    run_plan(plan, tmp_path / "new", **QUIET)
    run_mra_project(plan, tmp_path / "old", **QUIET)
    tags = plan.table[:, 0].astype(np.int64)
    for number, tolerance in ((1, 0.0), (2, 1e-8)):
        state = read_state(tmp_path / "new", number, tags)
        averages = tmp_path / "old" / "results" / f"ite_{number:04d}" / "averages"
        for channel, (angles, shifts) in state.poses.items():
            old = read_table(averages / f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl")
            old = old[np.argsort(old[:, 0])]
            # The old layout wrote 12 significant digits; the new keeps every bit.
            np.testing.assert_allclose(angles, old[:, 6:9], rtol=1e-11, atol=tolerance)
            np.testing.assert_allclose(shifts, old[:, 3:6], rtol=1e-11, atol=1e-11 + tolerance)
            np.testing.assert_allclose(
                state.references[channel],
                read_em(averages / f"average_ref_{channel:03d}_ite_{number:04d}.em"),
                atol=tolerance)


def test_a_stopped_run_carries_on_and_ends_as_the_unstopped_one(tmp_path: Path) -> None:
    plan = _short(_plan(_export(tmp_path)))
    run_plan(plan, tmp_path / "whole", **QUIET)
    calls = []

    def stopping(*args, **kwargs):
        calls.append(1)
        if len(calls) > 3:  # a search per channel: iteration 2's first
            raise KeyboardInterrupt
        return core.search_particles(*args, **kwargs)

    stopped = tmp_path / "stopped"
    with pytest.raises(KeyboardInterrupt):
        run_plan(plan, stopped, search=stopping, **QUIET)
    assert completed(stopped / "ite_0001") is not None and not (stopped / "ite_0002").exists()
    (failed,) = (stopped / "attempts").iterdir()
    assert read_record(failed).status == "failed"
    first_id = completed(stopped / "ite_0001").id

    lines: list[str] = []
    run_plan(plan, stopped, log=lines.append)
    assert [line.split(":")[0] for line in lines] == ["ite_0002"]
    assert completed(stopped / "ite_0001").id == first_id
    tags = plan.table[:, 0].astype(np.int64)
    for number in (1, 2):
        mine, theirs = (read_state(folder, number, tags)
                        for folder in (stopped, tmp_path / "whole"))
        for channel in theirs.poses:
            for ours, unstopped in zip(mine.poses[channel], theirs.poses[channel], strict=True):
                np.testing.assert_array_equal(ours, unstopped)
            np.testing.assert_array_equal(mine.references[channel], theirs.references[channel])


def test_a_plan_carries_on_into_more_iterations_and_never_into_another(tmp_path: Path) -> None:
    export = _export(tmp_path)
    run = tmp_path / "mra_x"
    run_plan(_short(_plan(export), 1), run, **QUIET)
    first = completed(run / "ite_0001")
    run_plan(_short(_plan(export), 2), run, **QUIET)
    assert completed(run / "ite_0001").id == first.id and completed(run / "ite_0002") is not None

    other = _plan(export)
    other = _short(replace(other, settings=[replace(other.settings[0], low=4.0),
                                            *other.settings[1:]]), 2)
    asked = (run / "plan.json").read_text()
    with pytest.raises(ValueError, match="ite_0001 holds a node of another plan"):
        run_plan(other, run, **QUIET)
    with pytest.raises(ValueError, match="setup holds a node of another plan"):
        run_plan(_short(_plan(export, mask_radius=4), 2), run, **QUIET)
    # Refused before plan.json was written: it still says what the run is.
    assert (run / "plan.json").read_text() == asked

    # Carried on with fewer iterations than it once was asked for: nothing to compute, and
    # plan.json keeps the longer request.
    lines: list[str] = []
    run_plan(_short(_plan(export), 1), run, log=lines.append)
    assert lines == ["mra_x: all 1 iterations finished before; nothing to compute"]
    assert (run / "plan.json").read_text() == asked


def test_another_shift_limit_is_another_search_on_the_same_setup(tmp_path: Path) -> None:
    export = _export(tmp_path)
    run = tmp_path / "mra_x"
    run_plan(_short(_plan(export), 1), run, **QUIET)
    with pytest.raises(Refused, match="ite_0001 holds a node of another plan"):
        run_plan(_short(_plan(export, shift_limit=2), 1), run, **QUIET)


def test_an_anti_aliased_search_is_named_only_where_it_is_on(tmp_path: Path) -> None:
    export = _export(tmp_path)
    # A coarse global grid, refined down: a quick global round at dim 10 of the 16 px box.
    usual = _short(_plan(export, global_rounds=1, global_step=60), 1)
    anti_aliased = _short(_plan(export, global_rounds=1, global_step=60, antialias=True), 1)
    # Off, the record names nothing new, so runs made before keep their fingerprints.
    assert "antialias" not in plan_record(usual)["iterations"][0]["search"]
    assert plan_record(anti_aliased)["iterations"][0]["search"]["antialias"] is True
    run = tmp_path / "mra_g"
    run_plan(usual, run, **QUIET)
    with pytest.raises(Refused, match="ite_0001 holds a node of another plan"):
        run_plan(anti_aliased, run, **QUIET)
    with pytest.raises(ValueError, match="not Dynamo's"):
        run_mra_project(anti_aliased, tmp_path / "old", **QUIET)


def test_a_folder_of_another_layout_is_refused(tmp_path: Path) -> None:
    plan = _short(_plan(_export(tmp_path)), 1)
    run_mra_project(plan, tmp_path / "old", **QUIET)
    with pytest.raises(ValueError, match="holds files but no plan.json, so it is not a run"):
        run_plan(plan, tmp_path / "old", **QUIET)
    assert not (tmp_path / "old" / "setup").exists()
    # An empty folder is a new run's.
    (tmp_path / "empty").mkdir()
    run_plan(plan, tmp_path / "empty", **QUIET)
    assert completed(tmp_path / "empty" / "ite_0001") is not None


def test_a_dynamo_projects_starting_state_is_the_setups_input(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    run = tmp_path / "mirror"
    run_plan(_short(plan_from_dynamo(project), 1), run, **QUIET)
    setup = completed(run / "setup")
    assert setup.content_params == {"from": "the starting state of a Dynamo project"}
    assert sorted(item.slot for item in setup.inputs) == ["dynamo_project", "export"]
    assert completed(run / "ite_0001") is not None
