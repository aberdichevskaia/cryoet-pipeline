"""The classification report and its Dynamo reader, on a small synthetic Dynamo project.

The project mimics what Dynamo 1.1.511 left in real MRA projects run on the Mac
(2026-10-16): three channels, two iterations, channel 3 dropped after the first
with its stale table still claiming every particle, and a complex number in a
refined table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from cryoet_pipeline.backends.dynamo_search import average_class  # noqa: E402
from cryoet_pipeline.reports.classification import (  # noqa: E402
    Volume,
    central_views,
    class_changes,
    gallery_tile,
    plan_pages,
    sample_tags,
    write_classification_report,
    z_sections,
)
from cryoet_pipeline.reports.dynamo_project import (  # noqa: E402
    read_dynamo_classification,
    read_em,
    read_table,
)

BOX = 16
TAGS = list(range(1, 13))
FIRST = {**{tag: 1 for tag in range(1, 8)}, **{tag: 2 for tag in range(8, 13)}}
SECOND = {**FIRST, 7: 2}


def write_em(path: Path, volume: np.ndarray) -> None:
    """EM as Dynamo writes it: PC byte order, float32, x fastest."""

    volume = np.asarray(volume, dtype="<f4")
    nz, ny, nx = volume.shape
    header = bytearray(512)
    header[0] = 6
    header[3] = 5
    header[4:16] = np.array([nx, ny, nz], dtype="<i4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + volume.tobytes())


def row(tag: int, channel: int, *, averaged: int = 1, cc: float = 0.2) -> list[float]:
    values = [0.0] * 35
    values[0], values[1], values[2] = tag, 1, averaged
    values[9] = cc
    values[12], values[13], values[14] = 1, -60, 60
    values[19] = 1 if tag <= 6 else 2
    values[21] = 1
    values[33] = channel
    return values


def write_table(path: Path, rows: list[list[float]], complex_entry: bool = False) -> None:
    lines = [" ".join(f"{value:g}" for value in values) for values in rows]
    if complex_entry:
        # MATLAB writes a complex number this way; Dynamo leaves them in column 28.
        tokens = lines[0].split()
        tokens[27] = "0+1.2074e-06i"
        lines[0] = " ".join(tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def sphere(radius: float) -> np.ndarray:
    z, y, x = np.indices((BOX,) * 3) - (BOX - 1) / 2
    return (np.sqrt(x * x + y * y + z * z) <= radius).astype(np.float32)


def refined(classes: dict[int, int], channel: int) -> list[list[float]]:
    """A channel's table as the MRA plugin rewrites it: everyone's class, its own members marked."""

    return [row(tag, classes[tag], averaged=int(classes[tag] == channel),
                cc=0.1 * channel + 0.01 * tag) for tag in TAGS]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    base = tmp_path / "export"
    rng = np.random.default_rng(3)
    blob = sphere(4)
    for tag in TAGS:
        write_em(base / "particles" / f"particle_{tag:05d}.em",
                 -blob + 0.3 * rng.standard_normal((BOX,) * 3))
    folder = base / "mra_t"
    settings = folder / "settings"
    for channel in (1, 2, 3):
        write_em(settings / f"template_initial_ref_{channel:03d}.em",
                 rng.standard_normal((BOX,) * 3))
    for name in ("mask", "mask_classification", "mask_smoothing"):
        write_em(settings / f"{name}.em", sphere(6))
    (base / "mra_t_seeds").mkdir()
    (base / "mra_t_seeds" / "seed_kinds.txt").write_text("1 subset 4\n2 blob 4\n3 noise 0\n")

    scipy_io = pytest.importorskip("scipy.io")
    vpr = {
        "nref_r1": 3, "mra_r1": 1, "ite_r1": 2, "ite_r2": 0,
        "cone_range_r1": 15, "cone_sampling_r1": 5, "inplane_range_r1": 15,
        "inplane_sampling_r1": 5, "refine_r1": 2, "refine_factor_r1": 2, "high_r1": 2,
        "low_r1": 4, "sym_r1": "c1", "dim_r1": BOX, "area_search_r1": np.array([3, 3, 3]),
        "area_search_modus_r1": 1, "threshold_modus_r1": 0, "use_CC_r1": 1, "localnc_r1": 1,
        "folder_data": "particles", "file_template_initial": "mra_t_seeds",
        "file_mask": "mra_mask.em", "file_mask_classification": "mra_mask.em",
        "file_mask_smoothing": "mra_mask.em", "feature_data_sidelength": BOX,
        "feature_extension_data": "em", "destination": "standalone",
        "how_many_processors": 4, "apix": 1,
    }
    scipy_io.savemat(settings / "virtual_project.mat", {"card": vpr})

    results = folder / "results"
    for channel in (1, 2, 3):
        write_table(results / "ite_0001" / "starting_values"
                    / f"starting_table_ref_{channel:03d}_ite_0001.tbl",
                    [row(tag, channel) for tag in TAGS])
    lists = {1: "1 2 3", 2: "1 2", 3: "1 2"}
    for number, text in lists.items():
        name = f"currently_surviving_references_ite_{number:04d}.txt"
        (results / f"ite_{number:04d}").mkdir(parents=True, exist_ok=True)
        (results / f"ite_{number:04d}" / name).write_text(text + "\n")
    for number, classes, searched in ((1, FIRST, (1, 2, 3)), (2, SECOND, (1, 2))):
        averages = results / f"ite_{number:04d}" / "averages"
        for channel in searched:
            name = f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl"
            if channel == 3:
                # Dropped: never rewritten, so it still claims every particle for itself.
                write_table(averages / name, [row(tag, 3) for tag in TAGS])
            else:
                write_table(averages / name, refined(classes, channel),
                            complex_entry=number == 1 and channel == 1)
                write_em(averages / f"average_ref_{channel:03d}_ite_{number:04d}.em",
                         -sphere(3 + channel))
        card = folder / "cards" / f"ite_{number:04d}" / f"card_ite_ite_{number:04d}.card"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text("name_project=mra_t;\n"
                        f"time_start_computation=16-Oct-2026 1{number}:00:00;\n"
                        f"time_end_iteration=16-Oct-2026 1{number}:30:00;\n")
    return folder


def write_manifest(project: Path) -> None:
    particles = [{"tag": tag, "run": "TS_A" if tag <= 6 else "TS_B",
                  "native_record": {"score": 0.1 + 0.01 * tag}} for tag in TAGS]
    (project.parent / "export_manifest.json").write_text(json.dumps({
        "pixel_spacing_angstrom": 10.0,
        "wedge": {"ranges": {"TS_A": [-50.0, 60.0], "TS_B": [-40.0, 55.0]}},
        "particles": particles,
    }))


def test_read_em_keeps_the_axis_order(tmp_path: Path) -> None:
    volume = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    write_em(tmp_path / "v.em", volume)
    assert np.array_equal(read_em(tmp_path / "v.em"), volume)


def test_read_table_keeps_the_real_part_of_a_complex_entry(tmp_path: Path) -> None:
    write_table(tmp_path / "t.tbl", [row(1, 1), row(2, 1)], complex_entry=True)
    table = read_table(tmp_path / "t.tbl")
    assert table.shape == (2, 35)
    assert table[0, 27] == 0.0


def test_classes_come_from_the_channels_the_next_iteration_keeps(project: Path) -> None:
    run = read_dynamo_classification(project, compute_initial_average=False)
    assert run.problems == []
    assert run.planned_iterations == 2
    first, second = run.iterations
    assert first.searched == (1, 2, 3)
    assert sorted(first.classes) == [1, 2]
    assert first.classes[1].members.tolist() == list(range(1, 8))
    assert first.classes[2].members.tolist() == list(range(8, 13))
    assert second.classes[2].members.tolist() == list(range(7, 13))
    assert first.classes[1].median_cc == pytest.approx(0.1 + 0.04)
    assert first.started == "16-Oct-2026 11:00:00"
    assert class_changes(run.iterations) == [None, 1]


def test_an_unfinished_iteration_is_left_out(project: Path) -> None:
    (project / "results" / "ite_0003" / "currently_surviving_references_ite_0003.txt").unlink()
    run = read_dynamo_classification(project, compute_initial_average=False)
    assert [iteration.number for iteration in run.iterations] == [1]
    assert any("iteration 2 has not finished" in problem for problem in run.problems)


def test_channels_that_disagree_on_the_classes_are_reported(project: Path) -> None:
    other = dict(FIRST)
    other[1] = 2
    averages = project / "results" / "ite_0001" / "averages"
    write_table(averages / "refined_table_ref_002_ite_0001.tbl", refined(other, 2))
    run = read_dynamo_classification(project, compute_initial_average=False)
    assert any("disagree on the classes" in problem for problem in run.problems)
    assert run.iterations[0].classes[1].members.tolist() == list(range(1, 8))


def test_parameters_describe_the_project(project: Path) -> None:
    write_manifest(project)
    run = read_dynamo_classification(project, compute_initial_average=False)
    values = {(p.group, p.name): p.value for p in run.parameters}
    assert values[("Data", "pixel size")] == "10.000 Å, from export_manifest.json"
    assert values[("Data", "TS_A")] == "6 particles, tilts -50° to +60°"
    assert values[("References", "channel 2")].startswith(
        "average of 4 random particles at random orientations")
    assert values[("Masks", "classification")] == "the alignment mask"
    assert values[("Search, round 1", "band")] == "Fourier pixels 2 to 4: 80 to 40 Å"
    assert values[("Search, round 1", "shifts")] == "up to 3 3 3 px, from the box centre (modus 1)"
    assert values[("Execution", "iteration 1")].endswith("(30 min)")
    assert run.series[7] == "TS_B"
    assert run.scores[1] == pytest.approx(0.11)
    assert run.pixel_size_angstrom == 10.0


def test_without_the_project_file_the_card_gives_round_one(project: Path) -> None:
    (project / "settings" / "virtual_project.mat").unlink()
    card = project / "cards" / "ite_0001" / "card_ite_ite_0001.card"
    card.write_text(card.read_text() + "nref=3;\nmra=1;\ncone_range=15;\nfolder_data=particles;\n")
    run = read_dynamo_classification(project, compute_initial_average=False)
    assert any("virtual_project.mat is missing" in problem for problem in run.problems)
    assert len(run.iterations) == 2
    values = {(p.group, p.name): p.value for p in run.parameters}
    assert values[("Search, round 1", "cone")].startswith("15° wide")


def test_a_particle_at_the_identity_pose_is_itself(project: Path) -> None:
    run = read_dynamo_classification(project, compute_initial_average=False)
    aligned = run.aligned_particle(3, None)
    original = read_em(project.parent / "particles" / "particle_00003.em")
    inner = (slice(1, BOX - 2),) * 3
    assert np.allclose(aligned[inner], original[inner], atol=1e-6)
    assert np.isnan(aligned[-1]).all()


def test_the_initial_average_is_dynamo_average_of_every_particle(project: Path) -> None:
    run = read_dynamo_classification(project)
    assert run.initial_average is not None
    particles = [read_em(project.parent / "particles" / f"particle_{tag:05d}.em") for tag in TAGS]
    expected = average_class(particles, np.zeros((12, 3)), np.zeros((12, 3)),
                             [(-60.0, 60.0)] * 12, sphere(6), fmin=1).average
    assert np.allclose(run.initial_average.load(), expected, atol=1e-5)


def test_the_report_draws_every_planned_page(project: Path, tmp_path: Path) -> None:
    write_manifest(project)
    run = read_dynamo_classification(project)
    pages = plan_pages(run, written="now", software="test")
    outcome = write_classification_report(tmp_path / "report.pdf", run, written="now",
                                          software="test")
    assert outcome.problems == []
    assert outcome.pages == len(pages)
    labels = [page.label for page in pages]
    assert labels.count("iteration 1") == 1 and "class galleries" in labels
    assert "last classes" in labels
    assert (tmp_path / "report.pdf").read_bytes().startswith(b"%PDF")


def test_a_page_that_fails_is_replaced_and_reported(project: Path, tmp_path: Path) -> None:
    run = read_dynamo_classification(project, compute_initial_average=False)

    def broken() -> np.ndarray:
        raise OSError("unreadable")

    run.references[2] = Volume("broken.em", broken)
    outcome = write_classification_report(tmp_path / "report.pdf", run, written="now",
                                          software="test")
    assert any(problem.startswith("starting references: could not be drawn: OSError")
               for problem in outcome.problems)
    assert (tmp_path / "report.pdf").exists()


def test_z_sections_sit_symmetrically_about_the_centre() -> None:
    indices = [index for index, _ in z_sections(np.zeros((48, 48, 48)))]
    assert indices == list(range(10, 38, 3))
    assert np.mean(indices) == pytest.approx(23.5)


def test_pictures_have_the_expected_shapes() -> None:
    volume = np.random.default_rng(0).standard_normal((16, 12, 10))
    assert gallery_tile(volume).shape == (12, 10)
    assert [view.shape for view in central_views(volume)] == [(12, 10), (16, 10), (16, 12)]


def test_sample_tags_is_stable_and_sorted() -> None:
    tags = np.arange(1, 101)
    first = sample_tags(tags, 8)
    assert np.array_equal(first, sample_tags(tags, 8))
    assert np.array_equal(first, np.sort(first)) and len(first) == 8
    assert np.array_equal(sample_tags(tags[:5], 8), tags[:5])
