"""Consensus halves from runs of a registry (docs/halves_from_runs.md): the split as a
run of its own, the halves refined from it, `maps` knowing them by it."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")

from test_mra_from_registry import QUIET, ROOT, SERIES, SETUP, _project, _short  # noqa: E402

from cryoet_pipeline.mra_runs import Refused, run_plan  # noqa: E402
from cryoet_pipeline.mra_setup import plan_from_registry, plan_from_split  # noqa: E402
from cryoet_pipeline.particle_tables import read_particle_table  # noqa: E402
from cryoet_pipeline.reports.classification_reader import read_classification  # noqa: E402
from cryoet_pipeline.runs import completed  # noqa: E402

SCRIPT = ROOT / "scripts" / "consensus_halves.py"
SCHEDULE = {"mask_radius": 3, "mask_edge": 2, "schedule": "coarse_to_fine",
            "first_lowpass": 30.0, "last_lowpass": 22.0}


def _halves_script(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                          capture_output=True, text=True, check=False)


@dataclass(frozen=True)
class Split:
    project: Path
    run: Path
    channel: int
    split: Path
    halves: tuple[Path, Path]


@pytest.fixture(scope="module")
def made(tmp_path_factory: pytest.TempPathFactory) -> Split:
    """A run of the registry split against itself, and both halves refined one iteration."""
    project = _project(tmp_path_factory.mktemp("halves"))
    run = project / "mra" / "mra_a"
    run_plan(_short(plan_from_registry(project, "bin8", SERIES, **SETUP, **QUIET)), run, **QUIET)
    last = read_classification(run, compute_initial_average=False).iterations[-1]
    channel = max(last.classes, key=lambda one: len(last.classes[one].members))
    split = project / "mra" / "split_a"
    result = _halves_script("split", f"{run}:{channel}", "--agree", f"{run}:{channel}",
                            "--start-lowpass", 40, "--out", split)
    assert result.returncode == 0, result.stdout + result.stderr
    halves = []
    for half in (1, 2):
        plan = _short(plan_from_split(split, half, **SCHEDULE, **QUIET), 1)
        halves.append(project / "mra" / f"split_a-half{half}")
        run_plan(plan, halves[-1], **QUIET)
    return Split(project, run, channel, split, (halves[0], halves[1]))


def test_two_runs_of_a_registry_are_split_into_a_run_of_their_own(made: Split) -> None:
    record = completed(made.split)
    assert record is not None and record.step == "consensus-split"
    assert [item.slot for item in record.inputs] == ["poses", "agree"]
    iteration = completed(made.run / "ite_0002")
    assert iteration is not None
    assert {item.fingerprint for item in record.inputs} == {iteration.fingerprint}
    assert [selection.name for selection in record.selections] == ["half 1", "half 2"]
    # Where the project is stays out of the fingerprint: the runs' plans name it.
    assert record.content_params["particles"] == {"label": "bin8", "series": SERIES,
                                                  "decoy_label": None}
    columns, _ = read_particle_table(made.split / "halves.parquet")
    run = read_classification(made.run, compute_initial_average=False)
    members = run.iterations[-1].classes[made.channel].members
    assert columns["particle_id"].tolist() == members.tolist()
    assert set(columns["half"].tolist()) == {1, 2}
    # Each particle starts from its refined pose in the run.
    assert run.poses is not None
    refined = run.poses((2, made.channel))
    rows = {int(tag): row for row, tag in enumerate(refined.tags)}
    for at, tag in enumerate(columns["particle_id"]):
        row = rows[int(tag)]
        assert [columns[name][at] for name in ("tdrot_deg", "tilt_deg", "narot_deg")] == \
            refined.angles[row].tolist()
        assert [columns[name][at] for name in ("shift_x_px", "shift_y_px", "shift_z_px")] == \
            refined.shifts[row].tolist()
    assert (made.split / "start_reference.mrc").is_file()


def test_a_half_is_set_up_from_its_split(made: Split) -> None:
    split = completed(made.split)
    assert split is not None
    columns, _ = read_particle_table(made.split / "halves.parquet")
    fingerprints = set()
    for half, folder in enumerate(made.halves, 1):
        setup = completed(folder / "setup")
        assert setup is not None and setup.selection == f"half {half}"
        named = {item.slot: item for item in setup.inputs}
        assert named["selection"].fingerprint == split.fingerprint
        assert named["selection"].path == str(made.split)
        fingerprints.add(setup.fingerprint)
        table, _ = read_particle_table(folder / "setup" / "particles.parquet")
        chosen = columns["half"] == half
        assert table["particle_id"].tolist() == columns["particle_id"][chosen].tolist()
        np.testing.assert_array_equal(table["tdrot_deg"], columns["tdrot_deg"][chosen])
        np.testing.assert_array_equal(table["shift_z_px"], columns["shift_z_px"][chosen])
        # The tags are the first run's, each the same stack and index.
        ours = read_classification(folder, compute_initial_average=False).identity
        theirs = read_classification(made.run, compute_initial_average=False).identity
        assert ours == {tag: theirs[tag] for tag in ours}
    assert len(fingerprints) == 2
    with pytest.raises(ValueError, match="leave out seeds"):
        plan_from_split(made.split, 1, seeds=["noise", "noise"], **QUIET)
    with pytest.raises(ValueError, match="not a split"):
        plan_from_split(made.run, 1, **QUIET)


def test_maps_knows_the_halves_by_their_split(made: Split, tmp_path: Path) -> None:
    out = tmp_path / "consensus"
    result = _halves_script("maps", made.halves[1], made.halves[0], "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "iteration 1 of both halves" in result.stdout
    for half, folder in enumerate(made.halves, 1):
        written = mrcfile.read(tmp_path / f"consensus_half{half}.mrc")
        np.testing.assert_array_equal(written, mrcfile.read(folder / "ite_0001" / "average_1.mrc"))
    assert "--randomize-beyond-angstrom 40" in result.stdout
    same = _halves_script("maps", made.halves[0], made.halves[0], "--out", tmp_path / "same")
    assert same.returncode != 0 and "not the two halves of one split" in same.stderr


def test_a_split_of_runs_that_read_other_particles_is_refused(made: Split) -> None:
    other = made.project / "mra" / "mra_reversed"
    run_plan(_short(plan_from_registry(made.project, "bin8", SERIES[::-1], **SETUP, **QUIET), 1),
             other, **QUIET)
    refused = _halves_script("split", f"{made.run}:{made.channel}", "--agree", f"{other}:1",
                             "--start-lowpass", 40, "--out", made.project / "mra" / "split_b")
    assert refused.returncode != 0 and "read different particles" in refused.stderr
    elsewhere = _halves_script("split", f"{made.run}:{made.channel}", "--agree",
                               f"{made.run}:{made.channel}", "--start-lowpass", 40,
                               "--out", made.project / "split_c")
    assert elsewhere.returncode != 0 and "lives in" in elsewhere.stderr
    missing = _halves_script("split", f"{made.project / 'mra' / 'mra_none'}:1", "--agree",
                             f"{made.run}:{made.channel}", "--start-lowpass", 40,
                             "--out", made.project / "mra" / "split_e")
    assert missing.returncode != 0
    assert missing.stderr.strip() == f"{made.project / 'mra' / 'mra_none'}: no such run"
    again = _halves_script("split", f"{made.halves[0]}:1", "--agree", f"{made.halves[1]}:1",
                           "--start-lowpass", 40, "--out", made.project / "mra" / "split_d")
    assert again.returncode != 0 and "is not split again" in again.stderr
    # A half run carried on under another split is refused, as any other plan is.
    with pytest.raises(Refused):
        run_plan(_short(plan_from_split(made.split, 2, **SCHEDULE, **QUIET), 1), made.halves[0],
                 **QUIET)


def test_the_job_script_refines_a_half(made: Split) -> None:
    arguments = [sys.executable, str(ROOT / "scripts/mra_port_run.py"), "--split", str(made.split),
                 "--half", "2", "--name", "half_job", "--mask-radius", "3", "--schedule",
                 "coarse_to_fine", "--first-lowpass", "30", "--last-lowpass", "22",
                 "--iterations", "1", "--device", "numpy"]
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    setup = completed(made.project / "mra" / "half_job" / "setup")
    assert setup is not None and setup.selection == "half 2"
    seeded = subprocess.run([*arguments[:-2], "--seeds", "map", "--device", "numpy"],
                            capture_output=True, text=True, check=False)
    assert seeded.returncode == 2 and "leave out --seeds" in seeded.stderr
