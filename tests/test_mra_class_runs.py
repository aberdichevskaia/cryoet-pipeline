"""An MRA run on some classes of an earlier run (docs/mra_in_lanes.md, "Again on a
class"), and a lane's MRA run by scripts/mra_port_run.py --config, on a synthetic
registry."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from test_mra_from_registry import QUIET, SERIES, _project, _short  # noqa: E402

from cryoet_pipeline.mra_lane import config_sha256  # noqa: E402
from cryoet_pipeline.mra_runs import PARTICLES, node_folder, run_plan  # noqa: E402
from cryoet_pipeline.mra_setup import (  # noqa: E402
    class_selection,
    plan_from_classes,
    plan_from_registry,
    selected_particles,
)
from cryoet_pipeline.particle_sources import ANGLES, SHIFTS, registry_particles  # noqa: E402
from cryoet_pipeline.particle_tables import read_particle_table  # noqa: E402
from cryoet_pipeline.particles import read_particle_star, write_particle_star  # noqa: E402
from cryoet_pipeline.runs import completed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SETUP = {"subset": 3, "mask_radius": 3, "mask_edge": 2, "schedule": "coarse_to_fine",
         "first_lowpass": 30.0, "last_lowpass": 22.0}


def _first_run(tmp_path: Path) -> tuple[Path, Path]:
    project = _project(tmp_path)
    plan = _short(plan_from_registry(project, "bin8", SERIES, ["subset", "noise"], **SETUP,
                                     **QUIET))
    run = project / "mra" / "first"
    run_plan(plan, run, **QUIET)
    return project, run


def _fullest_class(run: Path, iteration: int = 2) -> int:
    columns, _ = read_particle_table(node_folder(run, iteration) / PARTICLES)
    assigned = columns["assigned"].astype(bool)
    channels = columns["channel"].astype(int)
    return max(set(channels.tolist()), key=lambda one: int(np.sum(assigned & (channels == one))))


def test_a_class_run_takes_the_class_at_the_poses_its_iteration_gave(tmp_path):
    project, run = _first_run(tmp_path)
    chosen_class = _fullest_class(run)
    chosen = class_selection(run, [chosen_class])
    assert chosen.iteration == 2 and chosen.sizes[chosen_class] == len(chosen.tags) > 0
    columns, _ = read_particle_table(node_folder(run, 2) / PARTICLES)
    rows = (columns["channel"] == chosen_class) & columns["assigned"].astype(bool)
    assert sorted(chosen.tags.tolist()) == sorted(columns["particle_id"][rows].astype(int).tolist())

    plan = plan_from_classes(run, [chosen_class], ["noise", "noise"], **SETUP, **QUIET)
    tags = plan.table[:, 0].astype(int)
    np.testing.assert_array_equal(tags, chosen.tags)
    np.testing.assert_allclose(plan.table[:, ANGLES], chosen.angles)
    np.testing.assert_allclose(plan.table[:, SHIFTS], chosen.shifts)
    origin = plan.setup["variables"]["particles"]
    assert (origin["from_run"], origin["from_iteration"], origin["from_classes"],
            origin["start_poses"]) == ("first", 2, [chosen_class], "run")

    # Each particle at its pick instead.
    picked = plan_from_classes(run, [chosen_class], ["noise", "noise"], start_poses="picked",
                               **SETUP, **QUIET)
    source = registry_particles(project, "bin8", SERIES, **QUIET)
    index = [list(source.tags).index(tag) for tag in tags]
    np.testing.assert_allclose(picked.table[:, ANGLES], source.table[index][:, ANGLES])
    assert picked.setup["variables"] != plan.setup["variables"]

    # Its setup names the iteration it chose from, by that iteration's record.
    again = project / "mra" / "again"
    run_plan(_short(plan, 1), again, **QUIET)
    record = completed(node_folder(again, 0))
    selection = [one for one in record.inputs if one.slot == "selection"]
    assert len(selection) == 1
    assert selection[0].fingerprint == completed(node_folder(run, 2)).fingerprint
    assert completed(node_folder(again, 1)) is not None


def test_a_class_run_refuses_what_it_cannot_take(tmp_path):
    project, run = _first_run(tmp_path)
    with pytest.raises(ValueError, match="iteration 3 is not one of its 2"):
        class_selection(run, [1], iteration=3)
    columns, _ = read_particle_table(node_folder(run, 2) / PARTICLES)
    with pytest.raises(ValueError, match="has no particle in class 9"):
        class_selection(run, [9])
    with pytest.raises(ValueError, match="start_poses"):
        class_selection(run, [1], start_poses="refined")
    with pytest.raises(ValueError, match="no such run"):
        class_selection(project / "mra" / "none", [1])
    # An iteration that has not finished.
    folder = node_folder(run, 2)
    folder.rename(folder.with_name("ite_0002.away"))
    with pytest.raises(ValueError, match=r"iteration 2 has not finished \(1 of its 2 have\)"):
        class_selection(run, [1])


def test_particles_extracted_again_since_the_run_are_refused(tmp_path):
    project, run = _first_run(tmp_path)
    chosen = class_selection(run, [_fullest_class(run)])
    star = project / "particles" / "TS_B" / "stack.star"
    records = read_particle_star(star)
    write_particle_star(star, [replace(records[0], score=0.99), *records[1:]])
    with pytest.raises(ValueError, match="extracted again since"):
        selected_particles(chosen, **QUIET)


def _lane(path: Path, project: Path, mra: str) -> Path:
    """The keys a lane's MRA reads, over the synthetic registry: 9.784 A in a 12 px box."""
    path.write_text(
        f'[lane]\nname = "t"\ninput = "stack"\nvariant = "bin8"\n\n'
        f'[run]\nproject = "{project}"\n\n'
        f'[pick]\nparticle_diameter_angstrom = 20.0\ngpu_ids = "0"\n\n'
        f"[mra]\n{mra}\n"
    )
    return path


def _job(config: Path, project: Path, name: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/mra_port_run.py"), "--config", str(config),
         "--source", str(project), "--series", *SERIES, "--name", name, "--iterations", "1",
         "--device", "numpy", *extra],
        capture_output=True, text=True, check=False,
    )


def test_the_job_runs_a_lanes_mra_as_the_same_numbers_by_hand_would(tmp_path):
    project = _project(tmp_path)
    config = _lane(tmp_path / "lane.toml", project, 'seeds = ["subset", "noise"]\n'
                   "subset_particles = 3\nmask_radius_px = 3")
    text = config.read_text()
    done = _job(config, project, "bin8-sn", "--config-sha256", config_sha256(text))
    assert done.returncode == 0, done.stderr
    assert "subset noise   (lane)" in done.stdout
    run = project / "mra" / "bin8-sn"
    assert completed(node_folder(run, 1)) is not None
    setup = json.loads((run / "plan.json").read_text())["setup"]
    assert setup["request"]["lane_config_sha256"] == config_sha256(text)
    assert setup["request"]["sources"]["first_lowpass"].startswith("rule")
    # 2 and 1.1 x the Nyquist of 9.784 A particles, in A: the numbers by hand give the
    # same plan.
    by_hand = plan_from_registry(
        project, "bin8", SERIES, ["subset", "noise"], subset=3, mask_radius=3, mask_edge=2,
        schedule="coarse_to_fine", first_lowpass=4 * 9.784, last_lowpass=1.1 * 2 * 9.784,
        diameter=20.0, global_rounds=2, cone_steps=[3.0], **QUIET)
    assert setup["variables"] == json.loads(json.dumps(by_hand.setup["variables"]))

    # A lane file changed since its submission is refused, and so is another name.
    assert "has changed since it was submitted" in _job(
        config, project, "bin8-sn", "--config-sha256", "0" * 64).stderr
    assert "is not the run" in _job(config, project, "other").stderr
    assert "leave out --seeds" in _job(config, project, "bin8-sn", "--seeds", "noise noise").stderr

    # Its fullest class again, from the poses it found, with other seeds.
    chosen_class = _fullest_class(run, 1)
    again = _lane(tmp_path / "again.toml", project,
                  f'from_run = "bin8-sn"\nfrom_classes = [{chosen_class}]\n'
                  'seeds = ["noise", "noise"]\nglobal_rounds = 0\nmask_radius_px = 3')
    done = _job(again, project, f"bin8-sn-c{chosen_class}-nn")
    assert done.returncode == 0, done.stderr
    assert f"class {chosen_class} of run bin8-sn" in done.stdout
    record = json.loads((project / "mra" / f"bin8-sn-c{chosen_class}-nn" / "plan.json").read_text())
    selection = record["setup"]["request"]["selection"]
    assert (selection["run"], selection["iteration"], selection["classes"]) == (
        "bin8-sn", 1, [chosen_class])
    assert record["setup"]["variables"]["particles"]["from_run"] == "bin8-sn"
