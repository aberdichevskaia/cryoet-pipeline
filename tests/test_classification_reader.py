"""cryoet_pipeline.reports.classification_reader: one reader for a classification in
either layout, a run of the run-record layout or a Dynamo project."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pyarrow")

from test_mra_setup import _export, _plan  # noqa: E402

from cryoet_pipeline.backends import dynamo_search as core  # noqa: E402
from cryoet_pipeline.dynamo_mra_project import run_mra_project  # noqa: E402
from cryoet_pipeline.mra_runs import pixel_size, run_plan  # noqa: E402
from cryoet_pipeline.reports.classification import (  # noqa: E402
    ClassificationRun,
    plan_pages,
    write_classification_report,
)
from cryoet_pipeline.reports.classification_reader import read_classification  # noqa: E402

QUIET = {"log": lambda _: None}


def _short(plan, iterations=2):
    return replace(plan, settings=plan.settings[:iterations], cards=plan.cards[:iterations],
                   mra=plan.mra[:iterations])


@pytest.fixture(scope="module")
def both(tmp_path_factory: pytest.TempPathFactory) -> tuple[ClassificationRun, ClassificationRun]:
    """One plan computed into either layout, each in the export as on the cluster, and
    each read back."""
    export = _export(tmp_path_factory.mktemp("both"))
    plan = _short(_plan(export))
    run_plan(plan, export / "new", **QUIET)
    run_mra_project(plan, export / "old", spacing=pixel_size(plan), **QUIET)
    return read_classification(export / "new"), read_classification(export / "old")


def test_a_run_reads_as_the_dynamo_layout_of_the_same_plan_does(
    both: tuple[ClassificationRun, ClassificationRun],
) -> None:
    new, old = both
    assert new.software == "MRA run of the pipeline's Dynamo port"
    assert old.software == "Dynamo MRA project"
    assert new.problems == old.problems == []
    np.testing.assert_array_equal(new.tags, old.tags)
    assert (new.series, new.scores, new.decoys) == (old.series, old.scores, old.decoys)
    assert new.reference_notes == old.reference_notes
    assert (new.box_px, new.pixel_size_angstrom) == (old.box_px, old.pixel_size_angstrom)
    assert sorted(new.references) == sorted(old.references) == [1, 2, 3]
    for channel, reference in new.references.items():
        np.testing.assert_array_equal(reference.load(), old.references[channel].load())
    assert new.mask is not None and old.mask is not None
    np.testing.assert_array_equal(new.mask.load(), old.mask.load())
    assert new.planned_iterations == 2
    assert [iteration.number for iteration in new.iterations] == [1, 2]
    for ours, theirs in zip(new.iterations, old.iterations, strict=True):
        assert ours.searched == theirs.searched
        assert sorted(ours.classes) == sorted(theirs.classes)
        for channel, result in ours.classes.items():
            other = theirs.classes[channel]
            np.testing.assert_array_equal(result.members, other.members)
            assert result.median_cc == pytest.approx(other.median_cc, abs=1e-11)
            assert result.average is not None and other.average is not None
            np.testing.assert_allclose(result.average.load(), other.average.load(), atol=1e-6)
        assert ours.summary is not None and ours.summary.startswith("low-pass ")
    # The poses, every table: the old layout keeps 12 significant digits, the new every bit.
    assert new.poses is not None and old.poses is not None
    for pose in (None, (1, 1), (1, 2), (1, 3), (2, 1), (2, 2)):
        mine, theirs = new.poses(pose), old.poses(pose)
        np.testing.assert_array_equal(mine.tags, theirs.tags)
        np.testing.assert_allclose(mine.angles, theirs.angles, rtol=1e-11, atol=1e-8)
        np.testing.assert_allclose(mine.shifts, theirs.shifts, rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(mine.cc, theirs.cc, atol=1e-11)
    for tag, pose in ((3, None), (3, (2, 1)), (11, (1, 2))):
        np.testing.assert_allclose(new.aligned_particle(tag, pose),
                                   old.aligned_particle(tag, pose), atol=1e-6)
    assert new.initial_average is not None and old.initial_average is not None
    np.testing.assert_array_equal(new.initial_average.load(), old.initial_average.load())


def test_the_report_draws_every_page_of_a_run(
    both: tuple[ClassificationRun, ClassificationRun], tmp_path: Path,
) -> None:
    run, _ = both
    pages = plan_pages(run, written="now", software="test")
    outcome = write_classification_report(tmp_path / "report.pdf", run, written="now",
                                          software="test")
    assert outcome.problems == []
    assert outcome.pages == len(pages)
    labels = [page.label for page in pages]
    assert "iteration 2" in labels and "class galleries" in labels
    rows = {(row.group, row.name): row.value for row in run.parameters}
    assert rows[("Setup", "seeds")] == "subset blob noise"
    assert rows[("References", "channel 3")] == "white noise"
    assert rows[("References", "particles swap between channels")] == "yes"
    assert rows[("Masks", "alignment")].startswith("sphere 5 px with a 2 px edge: ")
    assert rows[("Search", "iteration 1")].startswith("band 80–53 Å (px 2–3) · cone 31.2°")
    assert rows[("Data", "pixel size")] == "10.000 Å, from the run's particle table"


def test_a_run_is_read_to_its_last_finished_iteration_and_says_why(tmp_path: Path) -> None:
    plan = _short(_plan(_export(tmp_path)))
    run = tmp_path / "mra_x"
    calls = []

    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) > 3:  # a search per channel: iteration 2's first
            raise RuntimeError("the card fell off")
        return core.search_particles(*args, **kwargs)

    with pytest.raises(RuntimeError):
        run_plan(plan, run, search=failing, **QUIET)
    stopped = read_classification(run, compute_initial_average=False)
    assert [iteration.number for iteration in stopped.iterations] == [1]
    assert stopped.planned_iterations == 2
    (line,) = stopped.problems
    assert line.startswith("iteration 2: an attempt failed (")
    assert line.endswith("): RuntimeError: the card fell off")

    # Carried on, then iteration 2's record made to name another iteration 1 as its input.
    run_plan(plan, run, **QUIET)
    assert len(read_classification(run, compute_initial_average=False).iterations) == 2
    path = run / "ite_0002" / "run.json"
    record = json.loads(path.read_text())
    for item in record["inputs"]:
        if item["slot"] == "previous":
            item["fingerprint"] = "0" * 64
    path.write_text(json.dumps(record))
    mixed = read_classification(run, compute_initial_average=False)
    assert [iteration.number for iteration in mixed.iterations] == [1]
    assert mixed.problems == ["iteration 2 was computed from another iteration 1 than the one "
                              "in the run; the report stops before it"]


def test_a_run_whose_setup_has_not_finished_is_refused(tmp_path: Path) -> None:
    run = tmp_path / "mra_x"
    run.mkdir()
    (run / "plan.json").write_text("{}\n")
    with pytest.raises(ValueError, match="its setup has not finished"):
        read_classification(run)
