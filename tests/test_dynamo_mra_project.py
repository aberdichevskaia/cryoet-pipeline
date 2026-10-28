"""A Dynamo MRA project's plan run again with the port, and written as Dynamo lays a project out."""

from pathlib import Path

import mrcfile
import numpy as np
import pytest
from test_classification_report import write_em
from test_wedge import _granular_phantom

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.dynamo_mra_project import plan_from_dynamo, run_mra_project, torch_search
from cryoet_pipeline.reports.dynamo_project import (
    read_dynamo_classification,
    read_em,
    read_map,
    read_table,
)

BOX = 16
TILTS = (-60.0, 60.0)
CARD = {
    "cone_range": 10, "cone_sampling": 5, "inplane_range": 10, "inplane_sampling": 5,
    "refine": 1, "refine_factor": 2, "high": 2, "low": 6, "dim": BOX, "area_search": "3 3 3",
    "area_search_modus": 1, "sym": "c1", "use_CC": 1, "localnc": 1, "threshold_modus": 0,
    "folder_data": "particles", "file_template_initial": "mra_t_seeds",
}


def _project(root: Path, channels: int = 2, **card_changes: object) -> tuple[Path, list[int]]:
    """Two phantoms, 6 particles each at random poses, and a two-channel Dynamo project
    whose channels start from the two phantoms (with `channels` 1, only the first);
    returns it and each tag's phantom."""

    rng = np.random.default_rng(4)
    first = _granular_phantom(BOX).numpy().astype(float)
    second = np.roll(first, 3, axis=2) + np.flip(first, axis=0)
    truth, rows = [], []
    (root / "particles").mkdir(parents=True)
    for tag in range(1, 13):
        kind = 1 + (tag % 2)
        angles = np.array([rng.uniform(0, 360), np.degrees(np.arccos(rng.uniform(-1, 1))),
                           rng.uniform(0, 360)])
        volume, _ = core.resample(first if kind == 1 else second, angles)
        volume = core.apply_filter(volume, core.wedge(BOX, TILTS))
        volume = volume + rng.normal(scale=0.3 * volume.std(), size=volume.shape)
        with mrcfile.new(root / "particles" / f"particle_{tag:05d}.mrc") as handle:
            handle.set_data(volume.astype(np.float32))
        row = np.zeros(35)
        row[[0, 1, 2]] = [tag, 1, 1]
        row[6:9] = angles + rng.uniform(-3, 3, size=3)
        row[12:15] = [1, *TILTS]
        row[19] = 1
        rows.append(row)
        truth.append(kind)
    project = root / "mra_t"
    for channel, reference in ((1, first), (2, second))[:channels]:
        write_em(project / "settings" / f"template_initial_ref_{channel:03d}.em", reference)
        table = np.array(rows)
        table[:, 33] = channel
        path = (project / "results" / "ite_0001" / "starting_values"
                / f"starting_table_ref_{channel:03d}_ite_0001.tbl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(" ".join(f"{v:.12g}" for v in r) for r in table) + "\n")
    z, y, x = np.indices((BOX,) * 3) - BOX / 2
    write_em(project / "settings" / "mask.em",
             np.clip(7.5 - np.sqrt(x * x + y * y + z * z), 0, 1).astype(np.float32))
    for number in (1, 2):
        card = project / "cards" / f"ite_{number:04d}" / f"card_ite_ite_{number:04d}.card"
        card.parent.mkdir(parents=True)
        card.write_text("".join(f"{k}={v};\n" for k, v in {**CARD, **card_changes}.items()))
    (root / "mra_t_seeds").mkdir()
    (root / "mra_t_seeds" / "seed_kinds.txt").write_text(
        "".join(f"{channel} subset 6\n" for channel in range(1, channels + 1)))
    return project, truth


def test_the_plan_is_the_projects_own(tmp_path: Path) -> None:
    project, _ = _project(tmp_path)
    plan = plan_from_dynamo(project)
    assert sorted(plan.references) == [1, 2] and len(plan.settings) == 2
    assert plan.settings[0] == core.SearchSettings(
        cone_aperture=10, cone_sampling=5, inplane_range=10, inplane_sampling=5, refine=1,
        refine_factor=2, high=2, low=6, shift_limit=3, area_search_modus=1)
    assert plan.particles == tmp_path / "particles"
    assert plan.reference_notes == ["1 subset 6", "2 subset 6"]
    assert len(plan_from_dynamo(project, iterations=1).settings) == 1


@pytest.mark.parametrize("change, name", [({"dim": 8}, "dim"), ({"sym": "c4"}, "sym"),
                                          ({"area_search": "3 3 2"}, "area_search"),
                                          ({"mra": 0}, "mra 0 with 2 references")])
def test_what_the_port_does_not_model_is_refused(tmp_path: Path, change, name) -> None:
    project, _ = _project(tmp_path, **change)
    with pytest.raises(ValueError, match=name):
        plan_from_dynamo(project)


def test_a_project_without_mra_averages_as_the_project_does(tmp_path: Path) -> None:
    # A map seed's one channel, mra 0: no plugin, so the average is the project's own
    # (odd and even halves), as Dynamo writes it; the plugin's differs.
    project, _ = _project(tmp_path, channels=1, mra=0)
    plan = plan_from_dynamo(project)
    assert sorted(plan.references) == [1] and plan.mra == [False, False]
    out = tmp_path / "ours"
    run_mra_project(plan, out, log=lambda _: None)
    averages = out / "results" / "ite_0001" / "averages"
    rows = read_table(averages / "refined_table_ref_001_ite_0001.tbl")
    particles = [read_map(plan.particles / f"particle_{int(tag):05d}.mrc") for tag in rows[:, 0]]
    poses = (rows[:, 6:9], rows[:, 3:6], [tuple(row) for row in rows[:, 13:15]])
    written = read_em(averages / "average_ref_001_ite_0001.em")
    own, _, _ = core.average_particles(particles, *poses, plan.soft_mask)
    plugin = core.average_class(particles, *poses, plan.soft_mask).average
    np.testing.assert_allclose(written, own, atol=1e-5 * own.std())
    assert np.abs(plugin - own).max() > 0.1 * own.std()
    card = (out / "cards" / "ite_0001" / "card_ite_ite_0001.card").read_text()
    assert "mra=0;" in card and "nref=1;" in card


def test_a_project_at_a_smaller_dim_runs_resized_on_either_kernel(tmp_path: Path) -> None:
    # Dynamo's dim 12 for the 16-voxel box: the search at 12, the classes at 16.
    project, _ = _project(tmp_path, dim=12)
    plan = plan_from_dynamo(project)
    assert [settings.dim for settings in plan.settings] == [12, 12]
    run_mra_project(plan, tmp_path / "ours_numpy", log=lambda _: None)
    run_mra_project(plan, tmp_path / "ours_torch", search=torch_search("cpu", "float64", 8),
                    log=lambda _: None)
    for number in (1, 2):
        for channel in (1, 2):
            name = f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl"
            numpy_rows, torch_rows = (
                read_table(tmp_path / run / "results" / f"ite_{number:04d}" / "averages" / name)
                for run in ("ours_numpy", "ours_torch"))
            np.testing.assert_allclose(torch_rows, numpy_rows, atol=1e-8)
    card = (tmp_path / "ours_numpy" / "cards" / "ite_0001" / "card_ite_ite_0001.card").read_text()
    assert "dim=12;" in card
    average = read_em(tmp_path / "ours_numpy" / "results" / "ite_0001" / "averages"
                      / "average_ref_001_ite_0001.em")
    assert average.shape == (BOX,) * 3


def test_several_processes_can_share_each_device() -> None:
    # Nothing starts until the first search, so CUDA names can be checked without CUDA.
    search = torch_search("cuda", "float32", 8, devices=2, per_device=2)
    assert search.devices == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    assert torch_search("mps", "float32", 8, per_device=3).devices == ["mps"] * 3
    with pytest.raises(ValueError, match="per_device"):
        torch_search("cuda", "float32", 8, per_device=0)


def test_the_run_is_written_as_dynamo_writes_a_project(tmp_path: Path) -> None:
    project, truth = _project(tmp_path)
    plan = plan_from_dynamo(project)
    run_mra_project(plan, tmp_path / "ours_numpy", log=lambda _: None)
    run_mra_project(plan, tmp_path / "ours_torch", search=torch_search("cpu", "float64", 8),
                    destination="torch cpu float64", log=lambda _: None)

    # The torch kernel in float64 is the numpy one: every table of every iteration.
    for number in (1, 2):
        for channel in (1, 2):
            name = f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl"
            numpy_rows, torch_rows = (
                read_table(tmp_path / run / "results" / f"ite_{number:04d}" / "averages" / name)
                for run in ("ours_numpy", "ours_torch"))
            np.testing.assert_allclose(torch_rows, numpy_rows, atol=1e-8)

    # And the report's reader reads it: two finished iterations, each particle in the
    # class of the phantom it was made from, the seeds' record as the references' notes.
    run = read_dynamo_classification(tmp_path / "ours_numpy", compute_initial_average=False)
    assert [iteration.number for iteration in run.iterations] == [1, 2]
    last = run.iterations[-1]
    classes = {int(tag): channel for channel, result in last.classes.items()
               for tag in result.members}
    assert [classes[tag] for tag in range(1, 13)] == truth
    assert run.reference_notes == {1: "average of 6 random particles at their starting poses",
                                   2: "average of 6 random particles at their starting poses"}
    assert (tmp_path / "ours_numpy" / "log.txt").read_text() == "iteration 1 ok\niteration 2 ok\n"
    with pytest.raises(FileExistsError):
        run_mra_project(plan, tmp_path / "ours_numpy", log=lambda _: None)
