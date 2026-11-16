"""import-particles: a Dynamo folder enters as stacks the MRA reads back unchanged.

The particles here are noise with a missing wedge in the pipeline's convention,
over an asymmetric range, and the table gives that range as Dynamo's ytilt (the
range negated, `dynamo_ytilt`). After the import, the MRA's own reader of the
registry (`particle_sources.registry_particles`) has to give back the table it
was given and the voxels as they were.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import euler_matrix
from cryoet_pipeline.models import Provenance, SeriesEntry, TiltSeriesManifest
from cryoet_pipeline.particle_sources import (
    ANGLES,
    SHIFTS,
    reconstructed_tilt_range,
    registry_particles,
)
from cryoet_pipeline.particles_import import import_particles
from cryoet_pipeline.reports.dynamo_project import write_em
from cryoet_pipeline.runs import read_record
from cryoet_pipeline.tomogram_import import ImportRefused
from cryoet_pipeline.wedge import WedgeGeometry, wedge_present_mask_rfft

BOX = 32
RANGE = (-50.0, 60.0)
YTILT = (-60.0, 50.0)  # Dynamo's, for the volumes the pipeline stores
COUNT = 40


@pytest.fixture(scope="module")
def particles() -> list[np.ndarray]:
    mask = wedge_present_mask_rfft(BOX, WedgeGeometry(*RANGE), device=torch.device("cpu")).numpy()
    rng = np.random.default_rng(2)
    made = []
    for _ in range(COUNT):
        spectrum = np.fft.rfftn(rng.standard_normal((BOX, BOX, BOX)))
        spectrum *= mask + 0.2 * (1 - mask)
        made.append(np.fft.irfftn(spectrum, s=(BOX, BOX, BOX), axes=(0, 1, 2)).astype(np.float32))
    return made


def _folder(tmp_path: Path, particles: list[np.ndarray], *, ytilt: tuple[float, float] = YTILT,
            apix: float | None = 5.4, ftype: int = 1, skip: int | None = None
            ) -> tuple[Path, Path, np.ndarray]:
    folder = tmp_path / "data"
    folder.mkdir()
    rng = np.random.default_rng(9)
    rows = np.zeros((COUNT, 42))
    for index, particle in enumerate(particles):
        tag = index + 1
        if tag != skip:
            write_em(folder / f"particle_{tag:06d}.em", particle)
        rows[index, 0:3] = [tag, 1, 1]
        rows[index, 3:6] = rng.uniform(-1.5, 1.5, 3)
        rows[index, 6:9] = rng.uniform([-180, 0, -180], [180, 180, 180])
        rows[index, 9] = rng.uniform(0.1, 0.4)
        rows[index, 12:17] = [ftype, *ytilt, -90, 90]
        rows[index, 19] = 1 if index < 25 else 2
        rows[index, 21] = 1
        rows[index, 23:26] = np.floor(rng.uniform(20, 200, 3)) + 0.5
        rows[index, 35] = apix if apix is not None else 0
    table = tmp_path / "particles.tbl"
    table.write_text("\n".join(" ".join(f"{value:.12g}" for value in row) for row in rows) + "\n")
    return folder, table, rows


def test_the_mra_reads_imported_particles_as_the_table_gave_them(
    tmp_path: Path, particles: list[np.ndarray]
) -> None:
    folder, table, rows = _folder(tmp_path, particles)
    project = tmp_path / "project"

    results = import_particles(folder=folder, table_path=table, out=project,
                               series={1: "TS_01", 2: "TS_02"}, variant="dyn")

    assert [result.wedge["verdict"] for result in results] == ["agrees", "agrees"]
    read = registry_particles(project, "dyn", ["TS_01", "TS_02"], log=lambda _: None)
    assert len(read.particles) == COUNT
    # To the six decimals the particle STAR holds, as for the pipeline's own particles.
    np.testing.assert_allclose(read.table[:, SHIFTS], rows[:, 3:6], atol=1e-5)
    for got, given in zip(read.table[:, ANGLES], rows[:, 6:9], strict=True):
        np.testing.assert_allclose(euler_matrix(got), euler_matrix(given), atol=1e-6)
    np.testing.assert_allclose(read.table[:, 13:15], np.tile(YTILT, (COUNT, 1)))
    np.testing.assert_allclose(read.table[:, 23:26], rows[:, 23:26], atol=1e-5)
    np.testing.assert_array_equal(read.table[:, 19], rows[:, 19])
    assert read.pixel_size == pytest.approx(5.4)
    assert read.box == BOX
    for tag, particle in zip(read.tags, particles, strict=True):
        np.testing.assert_array_equal(read.volume(int(tag)), particle)
    assert set(read.wedge_sources.values()) == {"imported particles' table"}


def test_an_imported_stack_is_a_run_with_its_files_and_wedge(
    tmp_path: Path, particles: list[np.ndarray]
) -> None:
    folder, table, _ = _folder(tmp_path, particles)
    project = tmp_path / "project"
    first, _ = import_particles(folder=folder, table_path=table, out=project,
                                series={1: "TS_01", 2: "TS_02"})
    again, _ = import_particles(folder=folder, table_path=table, out=project,
                                series={1: "TS_01", 2: "TS_02"})

    assert again.already_imported and not first.already_imported
    stack = first.stack
    assert stack.provenance is Provenance.IMPORTED and stack.parent_ids == []
    assert stack.parameters["tilt_range_deg"] == list(RANGE)
    record = read_record(first.folder)
    assert record is not None and {item.slot for item in record.inputs} == {"table", "particles"}
    manifest = TiltSeriesManifest.model_validate_json(
        (project / "manifests" / "TS_01.json").read_text())
    assert manifest.entry is SeriesEntry.PARTICLES and manifest.images == []
    registry = ArtifactRegistry.load(project / "artifacts.json")
    found, source = reconstructed_tilt_range(registry, registry.get(stack.id), manifest)
    assert (found, source) == (RANGE, "imported particles' table")
    qc = json.loads((first.folder / "particles_qc.json").read_text())
    assert qc["dynamo_ytilt"] == list(YTILT)


def test_a_table_whose_wedge_has_the_other_sign_is_refused(
    tmp_path: Path, particles: list[np.ndarray]
) -> None:
    folder, table, _ = _folder(tmp_path, particles, ytilt=RANGE)

    with pytest.raises(ImportRefused, match="other sign"):
        import_particles(folder=folder, table_path=table, out=tmp_path / "project",
                         series={1: "TS_01", 2: "TS_02"})


@pytest.mark.parametrize(
    "change, series, match",
    [
        ({}, {1: "TS_01"}, "tomograms \\[2\\]"),
        ({"skip": 7}, {1: "TS_01", 2: "TS_02"}, "no file for 1"),
        ({"ftype": 3}, {1: "TS_01", 2: "TS_02"}, "ftype 1"),
        ({"apix": None}, {1: "TS_01", 2: "TS_02"}, "--pixel-size"),
    ],
)
def test_what_the_table_leaves_unsaid_or_wrong_is_refused(
    tmp_path: Path, particles: list[np.ndarray], change: dict[str, Any],
    series: dict[int, str], match: str,
) -> None:
    folder, table, _ = _folder(tmp_path, particles, **change)
    with pytest.raises(ImportRefused, match=match):
        import_particles(folder=folder, table_path=table, out=tmp_path / "project",
                         series=series)


def test_the_command_line_imports_a_dynamo_folder(
    tmp_path: Path, particles: list[np.ndarray]
) -> None:
    folder, table, _ = _folder(tmp_path, particles)
    outcome = CliRunner().invoke(cli.app, [
        "import-particles", "--folder", str(folder), "--table", str(table), "--out",
        str(tmp_path / "project"), "--series", "1=TS_01", "--series", "2=TS_02",
        "--variant", "dyn",
    ])
    assert outcome.exit_code == 0, outcome.output
    assert "TS_02: wrote" in outcome.output and "wedge check: agrees" in outcome.output
