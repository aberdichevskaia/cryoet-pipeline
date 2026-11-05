"""An MRA run from a project's registry (docs/mra_from_registry.md): the particle stacks
read where they are, against a run from an export of the same stacks."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import mrcfile
import numpy as np
import pytest

pytest.importorskip("pyarrow")

from test_export_dynamo_script import _series  # noqa: E402

from cryoet_pipeline.artifacts import ArtifactRegistry  # noqa: E402
from cryoet_pipeline.mra_runs import SETUP_VERSION, Refused, run_plan  # noqa: E402
from cryoet_pipeline.mra_setup import plan_from_export, plan_from_registry  # noqa: E402
from cryoet_pipeline.particle_sources import registry_particles  # noqa: E402
from cryoet_pipeline.particle_tables import read_particle_table  # noqa: E402
from cryoet_pipeline.particles import read_particle_star, write_particle_star  # noqa: E402
from cryoet_pipeline.reports.classification_reader import read_classification  # noqa: E402
from cryoet_pipeline.runs import completed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["TS_A", "TS_B"]
SETUP = {"seeds": ["subset", "noise"], "subset": 3, "mask_radius": 3, "mask_edge": 2,
         "schedule": "coarse_to_fine", "first_lowpass": 30.0, "last_lowpass": 22.0}
QUIET = {"log": lambda _: None}


def _project(root: Path) -> Path:
    project = root / "project"
    registry = ArtifactRegistry.empty()
    for name, count, seed in (("TS_A", 7, 1), ("TS_B", 6, 2)):
        artifact, _ = _series(project, name, count, seed)
        registry.add(artifact)
    registry.write(project / "artifacts.json")
    return project


def _export(project: Path, out: Path) -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_dynamo.py"), "--project", str(project),
         "--variant", "bin8", *[word for name in SERIES for word in ("--series", name)],
         "--out", str(out)],
        check=True, capture_output=True,
    )
    return out


def _short(plan, iterations=2):
    return replace(plan, settings=plan.settings[:iterations], cards=plan.cards[:iterations],
                   mra=plan.mra[:iterations])


@dataclass(frozen=True)
class Both:
    project: Path
    export: Path
    registry_run: Path
    export_run: Path


@pytest.fixture(scope="module")
def both(tmp_path_factory: pytest.TempPathFactory) -> Both:
    """One plan from the registry and the same from an export of its stacks, each run."""
    root = tmp_path_factory.mktemp("registry")
    project = _project(root)
    export = _export(project, root / "export")
    from_registry = _short(plan_from_registry(project, "bin8", SERIES, **SETUP, **QUIET))
    from_export = _short(plan_from_export(export, **SETUP, **QUIET))
    run_plan(from_registry, project / "mra" / "mra_r", **QUIET)
    run_plan(from_export, export / "mra_e", **QUIET)
    return Both(project, export, project / "mra" / "mra_r", export / "mra_e")


def test_the_registry_gives_what_the_export_wrote(both: Both) -> None:
    source = registry_particles(both.project, "bin8", SERIES, **QUIET)
    # The export writes the same rows with 12 significant digits, and the same voxels.
    written = (both.export / "initial.tbl").read_text()
    assert written == "\n".join(" ".join(f"{value:.12g}" for value in row)
                                for row in source.table) + "\n"
    manifest = json.loads((both.export / "export_manifest.json").read_text())
    for entry, particle in zip(manifest["particles"], source.particles, strict=True):
        assert (entry["tag"], entry["run"], entry["source_stack_id"],
                entry["source_index_zero_based"]) == (particle.tag, particle.series,
                                                      particle.stack_id, particle.index)
        np.testing.assert_array_equal(mrcfile.read(both.export / entry["file"]),
                                      source.volume(particle.tag))
    assert [particle.tag for particle in source.particles] == list(range(1, 14))


def test_a_run_from_the_registry_is_the_run_from_its_export(both: Both) -> None:
    ours, theirs = (read_classification(run) for run in (both.registry_run, both.export_run))
    np.testing.assert_array_equal(ours.tags, theirs.tags)
    assert ours.identity == theirs.identity and len(ours.identity) == 13
    assert (ours.series, ours.scores, ours.decoys) == (theirs.series, theirs.scores,
                                                        theirs.decoys)
    assert ours.reference_notes == theirs.reference_notes
    # The starting poses differ only by the export table's 12 digits, so the seeds by
    # what that moves in an average, and every iteration after them as little.
    for channel, reference in ours.references.items():
        np.testing.assert_allclose(reference.load(), theirs.references[channel].load(),
                                   atol=1e-6)
    assert [iteration.number for iteration in ours.iterations] == [1, 2]
    for mine, other in zip(ours.iterations, theirs.iterations, strict=True):
        assert sorted(mine.classes) == sorted(other.classes)
        for channel, result in mine.classes.items():
            np.testing.assert_array_equal(result.members, other.classes[channel].members)
    assert ours.poses is not None and theirs.poses is not None
    for pose in (None, (2, 1)):
        np.testing.assert_allclose(ours.poses(pose).angles, theirs.poses(pose).angles,
                                   rtol=1e-10, atol=1e-6)
    # Turned out of its stack, as out of its file.
    np.testing.assert_allclose(ours.aligned_particle(5, (2, 1)),
                               theirs.aligned_particle(5, (2, 1)), atol=1e-4)
    assert ours.initial_average is not None and theirs.initial_average is not None
    np.testing.assert_allclose(ours.initial_average.load(), theirs.initial_average.load(),
                               atol=1e-5)
    rows = {(row.group, row.name): row.value for row in ours.parameters}
    assert rows[("Data", "particles from")] == (
        f"the particle stacks of {both.project}, label bin8")
    assert rows[("Setup", "particles")].startswith("{")


def test_the_setup_table_names_each_particles_stack_and_pick(both: Both) -> None:
    setup = completed(both.registry_run / "setup")
    assert setup is not None and setup.step_version == SETUP_VERSION == 2
    assert [item.slot for item in setup.inputs] == [
        "particle_stack TS_A:particle_stack:cubic-crop_bin8",
        "particle_stack TS_B:particle_stack:cubic-crop_bin8"]
    assert setup.content_params["particles"] == {"label": "bin8", "series": SERIES,
                                                 "decoy_label": None}
    for run in (both.registry_run, both.export_run):
        columns, metadata = read_particle_table(run / "setup" / "particles.parquet")
        assert metadata["project"] == str(both.project)
        assert "zero-based" in metadata["positions"]
        for series in SERIES:
            star = both.project / "particles" / series / "stack.star"
            records = read_particle_star(star)
            rows = np.flatnonzero(columns["tilt_series_id"] == series)
            assert columns["stack_index"][rows].tolist() == [record.index for record in records]
            # The pick as extraction's STAR has it, not the export table's plus 0.5.
            assert columns["position_x_px"][rows].tolist() == [record.x_px
                                                                for record in records]
            assert set(columns["stack_id"][rows]) == {f"{series}:particle_stack:cubic-crop_bin8"}


def test_runs_from_the_registry_and_its_export_compare_tag_by_tag(both: Both,
                                                                  tmp_path: Path) -> None:
    out = tmp_path / "numbers.json"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/compare_dynamo_classes.py"), str(both.registry_run),
         str(both.export_run), "--shuffles", "20", "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["identity"] == ("tags checked in both runs: same "
                                                       "series, stack and index")


def test_a_stack_extracted_again_from_other_picks_is_refused(tmp_path: Path) -> None:
    project = _project(tmp_path)
    plan = _short(plan_from_registry(project, "bin8", SERIES, **SETUP, **QUIET), 1)
    run = project / "mra" / "mra_x"
    run_plan(plan, run, **QUIET)
    star = project / "particles" / "TS_B" / "stack.star"
    records = read_particle_star(star)
    write_particle_star(star, [replace(records[0], score=0.99), *records[1:]])
    again = _short(plan_from_registry(project, "bin8", SERIES, **SETUP, **QUIET), 1)
    with pytest.raises(Refused, match="setup holds a node of another plan"):
        run_plan(again, run, **QUIET)


def test_the_job_script_runs_from_the_registry(tmp_path: Path) -> None:
    project = _project(tmp_path)
    arguments = [sys.executable, str(ROOT / "scripts/mra_port_run.py"), "--source", str(project),
                 "--lane", "bin8", "--series", *SERIES, "--name", "mra_j",
                 "--seeds", "noise noise", "--mask-radius", "3", "--schedule",
                 "coarse_to_fine", "--first-lowpass", "30", "--last-lowpass", "22",
                 "--iterations", "1", "--device", "numpy"]
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert f"mra_j was set up from {project}" in result.stdout
    assert completed(project / "mra" / "mra_j" / "ite_0001") is not None
    again = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert again.returncode == 0, again.stderr
    assert "mra_j: all 1 iterations finished before; nothing to compute" in again.stdout
    wrong = subprocess.run([*arguments[:2], "--source", str(project), "--name", "x",
                            "--seeds", "noise noise"], capture_output=True, text=True,
                           check=False)
    assert wrong.returncode == 2 and "--source needs --series" in wrong.stderr
