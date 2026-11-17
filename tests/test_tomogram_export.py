"""export-tomogram: the import's twin, held to the round trip and to IMOD's own clip.

A volume is entered with import-tomogram, written out in each orientation of the
list, and entered again with the answers its sidecar gives: it has to come back
voxel for voxel, its angles unchanged. Where IMOD is installed, its `clip` lays
out tilt's output in the other orientations, and the export has to match it.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import mrcfile
import numpy as np
import pytest
import zarr
from test_tomogram_import import ANGLES, SHAPE, wedged_noise, write_angles, write_mrc
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, Provenance
from cryoet_pipeline.runs import read_record
from cryoet_pipeline.tomogram_export import ExportRefused, export_tomogram
from cryoet_pipeline.tomogram_import import ORIENTATIONS, import_tomogram

LOW, HIGH = ANGLES[0], ANGLES[-1]


@pytest.fixture(scope="module")
def canonical() -> np.ndarray:
    return wedged_noise(SHAPE, LOW, HIGH, seed=4)


def _imported(tmp_path: Path, canonical: np.ndarray, *, orientation: str = "pipeline",
              layout: Any = None, processing: str = "raw") -> tuple[Path, str]:
    data = canonical if layout is None else layout(canonical)
    volume = write_mrc(tmp_path / "volume.mrc", data)
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"
    result = import_tomogram(volume_path=volume, tilt_angles_path=angles,
                             orientation_name=orientation, processing=processing,
                             tilt_series_id="TS_01", out=project)
    return project, result.tomogram.id


def _command_words(command: str) -> dict[str, str]:
    words = shlex.split(command)
    return {words[index][2:]: words[index + 1] for index in range(2, len(words) - 1, 2)}


@pytest.mark.parametrize("orientation", sorted(ORIENTATIONS))
def test_an_exported_tomogram_imports_back_as_it_was(
    tmp_path: Path, canonical: np.ndarray, orientation: str
) -> None:
    project, tomogram_id = _imported(tmp_path, canonical)
    out = tmp_path / "export"

    result = export_tomogram(project=project, tomogram_id=tomogram_id,
                             orientation_name=orientation, out=out)

    # The sidecar's command, as written, enters the files again.
    words = _command_words(result.sidecar["import_command"])
    assert words["orientation"] == orientation and words["processing"] == "raw"
    again = import_tomogram(
        volume_path=Path(words["volume"]), tilt_angles_path=Path(words["tilt-angles"]),
        orientation_name=words["orientation"], processing=words["processing"],
        tilt_series_id=words["tilt-series-id"], out=tmp_path / "elsewhere",
    )
    stored = np.asarray(cast(Any, zarr.open(again.tomogram.path, mode="r"))[:])
    np.testing.assert_array_equal(stored, canonical)
    assert [float(word) for word in result.tilt_angles.read_text().split()] == \
        pytest.approx(ANGLES)
    assert again.wedge["verdict"] == "agrees"
    with mrcfile.open(result.volume, permissive=True) as mrc:
        assert float(mrc.voxel_size.x) == pytest.approx(10.0)


def test_the_export_is_a_run_and_says_what_the_volume_is(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    project, tomogram_id = _imported(tmp_path, canonical, processing="ctf-corrected")
    out = tmp_path / "export"

    result = export_tomogram(project=project, tomogram_id=tomogram_id,
                             orientation_name="imod-standard", out=out)

    record = read_record(out)
    assert record is not None and record.step == "export-tomogram"
    assert {item.slot for item in record.inputs} == {"tomogram", "tilt_angles"}
    sidecar = json.loads((out / "export.json").read_text())
    assert sidecar == result.sidecar
    assert sidecar["from_the_pipelines_volume"] == "the pipeline's volume reversed in z"
    assert sidecar["shape_sections_rows_columns"] == list(SHAPE)
    assert sidecar["processing"] == ["ctf-corrected"]
    assert sidecar["provenance"] == "imported"
    assert [root["id"] for root in sidecar["not_derived_lineage"]] == [tomogram_id]
    assert sidecar["tilt_range_deg"] == pytest.approx([LOW, HIGH])


def test_a_tomogram_made_here_leaves_with_its_reconstructions_angles(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    """A tomogram that recorded no angles of its own takes its aligned stack's tilt file."""

    project = tmp_path / "project"
    tomogram_path = project / "tomograms" / "TS_01.zarr"
    zarr.open(tomogram_path, mode="w", shape=SHAPE, dtype=np.float32)[:] = canonical
    tilt_file = write_angles(project.parent / "final.tlt")
    stack = Artifact(id="TS_01:aligned_tilt_stack:final", kind=ArtifactKind.ALIGNED_TILT_STACK,
                     path=project / "final.st",
                     parameters={"tilt_series_id": "TS_01", "tilt_file_path": str(tilt_file)})
    tomogram = Artifact(id="TS_01:tomogram:fine", kind=ArtifactKind.TOMOGRAM, path=tomogram_path,
                        parent_ids=[stack.id], axis_order=AxisOrder.ZYX,
                        pixel_spacing_angstrom=5.4,
                        parameters={"tilt_series_id": "TS_01", "ctf_corrected": True})
    registry = ArtifactRegistry.empty()
    registry.extend([stack, tomogram])
    registry.commit(project / "artifacts.json")

    result = export_tomogram(project=project, tomogram_id=tomogram.id,
                             orientation_name="imod-tilt", out=tmp_path / "export")

    assert [float(word) for word in result.tilt_angles.read_text().split()] == \
        pytest.approx(ANGLES)
    assert result.sidecar["processing"] == ["ctf-corrected"]
    assert result.sidecar["provenance"] == Provenance.DERIVED.value
    assert result.sidecar["not_derived_lineage"] == []
    np.testing.assert_array_equal(mrcfile.read(result.volume), canonical.transpose(1, 0, 2))


@pytest.mark.parametrize(
    "orientation, change, match",
    [
        ("aretomo", None, "not on the list"),
        ("imod-standard", "exists", "never written over"),
        ("imod-standard", "not a tomogram", "not a tomogram"),
        ("imod-standard", "no angles", "records no angles"),
    ],
)
def test_what_cannot_leave_is_refused(
    tmp_path: Path, canonical: np.ndarray, orientation: str, change: str | None, match: str
) -> None:
    project, tomogram_id = _imported(tmp_path, canonical)
    out = tmp_path / "export"
    if change == "exists":
        out.mkdir()
    if change == "not a tomogram":
        tomogram_id = "TS_01:qc:tomogram_import"
    if change == "no angles":
        registry = ArtifactRegistry.load(project / "artifacts.json")
        tomogram = registry.get(tomogram_id)
        Path(tomogram.parameters["tilt_angles_path"]).unlink()
    with pytest.raises(ExportRefused, match=match):
        export_tomogram(project=project, tomogram_id=tomogram_id, orientation_name=orientation,
                        out=out)


def _clip() -> str | None:
    found = shutil.which("clip")
    if found:
        return found
    bundled = Path("/Applications/IMOD/bin/clip")
    return str(bundled) if bundled.is_file() else None


@pytest.mark.skipif(_clip() is None, reason="IMOD's clip is not installed")
@pytest.mark.parametrize(
    "orientation, clip_option",
    [("imod-tilt", None), ("imod-parallel", "flipyz"), ("imod-standard", "rotx")],
)
def test_the_export_lays_tilts_output_out_as_imods_clip_does(
    tmp_path: Path, canonical: np.ndarray, orientation: str, clip_option: str | None
) -> None:
    """tilt's raw output, Y as sections, entered as imod-tilt: each IMOD orientation
    of it, as IMOD's own clip makes it, is what the export writes."""

    project, tomogram_id = _imported(tmp_path, canonical, orientation="imod-tilt",
                                     layout=lambda volume: volume.transpose(1, 0, 2))
    raw = tmp_path / "volume.mrc"
    expected = raw
    if clip_option is not None:
        program = _clip()
        assert program is not None
        expected = tmp_path / f"{clip_option}.mrc"
        subprocess.run([program, clip_option, str(raw), str(expected)], check=True,
                       capture_output=True,
                       env={**os.environ, "IMOD_DIR": str(Path(program).parent.parent)})

    result = export_tomogram(project=project, tomogram_id=tomogram_id,
                             orientation_name=orientation, out=tmp_path / "export")

    np.testing.assert_array_equal(mrcfile.read(result.volume), mrcfile.read(expected))


def test_the_command_line_exports_and_says_how_to_import_back(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    project, tomogram_id = _imported(tmp_path, canonical)
    outcome = CliRunner().invoke(cli.app, [
        "export-tomogram", "--project", str(project), "--tomogram-id", tomogram_id,
        "--orientation", "relion5",
    ])
    assert outcome.exit_code == 0, outcome.output
    folder = project / "exports" / "TS_01_tomogram_imported_relion5"
    assert (folder / "TS_01.mrc").is_file() and (folder / "export.json").is_file()
    assert "back in: cryoet import-tomogram" in outcome.output
    refused = CliRunner().invoke(cli.app, [
        "export-tomogram", "--project", str(project), "--tomogram-id", tomogram_id,
        "--orientation", "warp",
    ])
    assert refused.exit_code != 0 and "not on the list" in refused.output
