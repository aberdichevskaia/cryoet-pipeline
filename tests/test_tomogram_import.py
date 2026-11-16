"""import-tomogram, the listed orientations, the checks, and the records it leaves.

The volumes here are noise with a real missing wedge, built in the pipeline's own
convention (`wedge.py`: the view at tilt t integrates along (sin t, 0, cos t) in
x, y, z of the canonical volume). Each orientation's file is made from one
canonical volume the way its program lays it out, and the import has to give
that volume back. That the conventions match IMOD's and RELION's is measured
elsewhere (docs/entry_and_exit.md); here the code is held to them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import mrcfile
import numpy as np
import pytest
import torch
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.bypass_external_inputs import import_external_picks_for_extraction
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    Provenance,
    SeriesEntry,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runs import bytes_fingerprint, read_record
from cryoet_pipeline.tomogram_import import (
    ImportRefused,
    import_tomogram,
    parse_processing,
    read_tilt_angles,
)
from cryoet_pipeline.wedge import WedgeGeometry, wedge_present_mask_rfft
from cryoet_pipeline.wedge_check import measure_wedge

SHAPE = (48, 96, 112)
LOW, HIGH = -60.0, 30.0
ANGLES = [float(angle) for angle in np.arange(LOW, HIGH + 0.1, 3.0)]


def wedged_noise(shape: tuple[int, int, int], low: float, high: float, *, floor: float = 0.2,
                 seed: int = 0) -> np.ndarray:
    """Noise whose transform is weak in the wedge of the range low..high."""

    # Real noise, so that its transform is Hermitian as a tomogram's is.
    spectrum = np.fft.rfftn(np.random.default_rng(seed).standard_normal(shape))
    kz = np.fft.fftfreq(shape[0])[:, None, None]
    kx = np.fft.rfftfreq(shape[2])[None, None, :]
    at_low = kx * np.sin(np.radians(low)) + kz * np.cos(np.radians(low))
    at_high = kx * np.sin(np.radians(high)) + kz * np.cos(np.radians(high))
    spectrum *= np.where(np.broadcast_to(at_low * at_high <= 0, spectrum.shape), 1.0, floor)
    return np.fft.irfftn(spectrum, s=shape, axes=(0, 1, 2)).astype(np.float32)


def write_mrc(path: Path, data: np.ndarray, voxel: float | None = 10.0) -> Path:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data))
        if voxel is not None:
            mrc.voxel_size = voxel
    return path


def write_angles(path: Path, angles: list[float] = ANGLES) -> Path:
    path.write_text("".join(f"{angle:.2f}\n" for angle in angles))
    return path


@pytest.fixture(scope="module")
def canonical() -> np.ndarray:
    return wedged_noise(SHAPE, LOW, HIGH)


# How each program lays the canonical volume out in its MRC.
LAYOUTS = {
    "pipeline": lambda volume: volume,
    "imod-parallel": lambda volume: volume,
    "imod-tilt": lambda volume: volume.transpose(1, 0, 2),
    "imod-standard": lambda volume: volume[::-1],
    "relion5": lambda volume: volume[::-1],
}


@pytest.mark.parametrize("orientation", sorted(LAYOUTS))
def test_each_listed_orientation_gives_back_the_canonical_volume(
    tmp_path: Path, canonical: np.ndarray, orientation: str
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", LAYOUTS[orientation](canonical))
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"

    result = import_tomogram(
        volume_path=volume, tilt_angles_path=angles, orientation_name=orientation,
        processing="raw", tilt_series_id="TS_01", out=project,
    )

    stored = np.asarray(cast(Any, zarr.open(result.tomogram.path, mode="r"))[:])
    np.testing.assert_array_equal(stored, canonical)
    # The angles enter unchanged, whatever the orientation's transform.
    written = [float(word) for word in Path(result.tomogram.parameters["tilt_angles_path"])
               .read_text().split()]
    assert written == pytest.approx(ANGLES)
    assert result.wedge["verdict"] == "agrees"


def test_the_import_is_a_run_with_its_inputs_and_answers(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical[::-1])
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"

    result = import_tomogram(
        volume_path=volume, tilt_angles_path=angles, orientation_name="imod-standard",
        processing="ctf-corrected,denoised", tilt_series_id="TS_01", out=project,
    )

    record = read_record(result.folder)
    assert record is not None and record.status == "succeeded"
    assert record.step == "import-tomogram"
    assert {item.slot: item.fingerprint for item in record.inputs} == {
        "volume": bytes_fingerprint(volume),
        "tilt_angles": bytes_fingerprint(angles),
    }
    assert {item.slot: item.path for item in record.inputs}["volume"] == str(volume.resolve())
    assert record.content_params["orientation"] == "imod-standard"
    assert record.content_params["processing"] == ["ctf-corrected", "denoised"]
    assert record.content_params["pixel_size_angstrom"] == pytest.approx(10.0)

    registry = ArtifactRegistry.load(project / "artifacts.json")
    tomogram = registry.get("TS_01:tomogram:imported")
    assert tomogram.provenance is Provenance.IMPORTED
    assert tomogram.parent_ids == []
    assert tomogram.can_recompute is False
    assert tomogram.parameters["ctf_corrected"] is True
    assert tomogram.parameters["run_fingerprint"] == record.fingerprint
    assert tomogram.parameters["included_z_values"] == list(range(len(ANGLES)))
    assert tomogram.covers("TS_01")
    assert registry.get("TS_01:qc:tomogram_import").parent_ids == [tomogram.id]

    manifest = TiltSeriesManifest.model_validate_json(result.manifest_path.read_text())
    assert manifest.entry is SeriesEntry.TOMOGRAM
    assert manifest.tilt_angles_deg == pytest.approx(ANGLES)
    assert all(image.subframe_path is None and image.binning is None for image in manifest.images)
    report = json.loads((result.folder / "import_qc.json").read_text())
    assert report["canonical_shape_zyx"] == list(SHAPE)


def test_importing_the_same_files_with_the_same_answers_twice_does_nothing(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical)
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"
    arguments: dict[str, Any] = dict(
        volume_path=volume, tilt_angles_path=angles, orientation_name="imod-parallel",
        processing="raw", tilt_series_id="TS_01", out=project,
    )
    first = import_tomogram(**arguments)
    before = (project / "artifacts.json").read_text()

    second = import_tomogram(**arguments)

    assert not first.already_imported and second.already_imported
    assert (project / "artifacts.json").read_text() == before
    assert sorted(path.name for path in (project / "imports" / "TS_01").iterdir()) == [
        "tomogram", "tomogram.lock",
    ]


def test_other_answers_about_the_same_files_need_their_own_label(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical)
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"
    arguments: dict[str, Any] = dict(
        volume_path=volume, tilt_angles_path=angles, orientation_name="imod-parallel",
        tilt_series_id="TS_01", out=project,
    )
    import_tomogram(processing="raw", **arguments)

    with pytest.raises(ImportRefused, match="--variant"):
        import_tomogram(processing="denoised", **arguments)
    labelled = import_tomogram(processing="denoised", variant="denoised", **arguments)

    registry = ArtifactRegistry.load(project / "artifacts.json")
    assert {artifact.id for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)} == {
        "TS_01:tomogram:imported", "TS_01:tomogram:imported:denoised",
    }
    assert labelled.tomogram.variant == "denoised"


def test_an_unmeasured_orientation_is_refused_with_what_measuring_it_needs(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical)
    angles = write_angles(tmp_path / "volume.tlt")

    with pytest.raises(ImportRefused, match="eight axis reversals"):
        import_tomogram(
            volume_path=volume, tilt_angles_path=angles, orientation_name="aretomo",
            processing="raw", tilt_series_id="TS_01", out=tmp_path / "project",
        )
    assert not (tmp_path / "project").exists()


def test_the_wedge_check_refuses_a_volume_answered_as_its_mirror(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    # IMOD's standard orientation answered as -PARALLEL: the import would keep the
    # volume mirrored, and the wedge, with the range asymmetric, shows it.
    volume = write_mrc(tmp_path / "volume.mrc", canonical[::-1])
    angles = write_angles(tmp_path / "volume.tlt")
    arguments: dict[str, Any] = dict(
        volume_path=volume, tilt_angles_path=angles, orientation_name="imod-parallel",
        processing="raw", tilt_series_id="TS_01", out=tmp_path / "project",
    )

    with pytest.raises(ImportRefused, match="mirror"):
        import_tomogram(**arguments)
    assert not (tmp_path / "project" / "imports").exists()

    overruled = import_tomogram(overrule_wedge_check=True, **arguments)
    record = read_record(overruled.folder)
    assert record is not None and record.content_params["wedge_check_overruled"] is True
    assert any("overruled" in warning for warning in overruled.warnings)


def test_the_wedge_check_tests_the_wedge_filled_answer(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    filled = wedged_noise(SHAPE, LOW, HIGH, floor=1.0)
    angles = write_angles(tmp_path / "volume.tlt")
    filled_path = write_mrc(tmp_path / "filled.mrc", filled)
    raw_path = write_mrc(tmp_path / "raw.mrc", canonical)

    with pytest.raises(ImportRefused, match="wedge-filled"):
        import_tomogram(volume_path=filled_path, tilt_angles_path=angles,
                        orientation_name="pipeline", processing="raw", tilt_series_id="TS_01",
                        out=tmp_path / "one")
    result = import_tomogram(volume_path=filled_path, tilt_angles_path=angles,
                             orientation_name="pipeline", processing="wedge-filled",
                             tilt_series_id="TS_01", out=tmp_path / "two")
    assert result.wedge["verdict"] == "no wedge"
    with pytest.raises(ImportRefused, match="shows a wedge"):
        import_tomogram(volume_path=raw_path, tilt_angles_path=angles,
                        orientation_name="pipeline", processing="wedge-filled",
                        tilt_series_id="TS_01", out=tmp_path / "three")


def test_tilts_raw_output_answered_with_sections_as_z_is_refused_by_its_thin_axis(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical.transpose(1, 0, 2))
    angles = write_angles(tmp_path / "volume.tlt")

    with pytest.raises(ImportRefused, match="not the shortest"):
        import_tomogram(volume_path=volume, tilt_angles_path=angles,
                        orientation_name="imod-parallel", processing="raw",
                        tilt_series_id="TS_01", out=tmp_path / "project")


def test_the_header_pixel_size_is_checked_against_the_answer(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    angles = write_angles(tmp_path / "volume.tlt")
    with_size = write_mrc(tmp_path / "sized.mrc", canonical, voxel=10.0)
    without = write_mrc(tmp_path / "unsized.mrc", canonical, voxel=None)
    common: dict[str, Any] = dict(tilt_angles_path=angles, orientation_name="pipeline",
                                  processing="raw", tilt_series_id="TS_01")

    with pytest.raises(ImportRefused, match="header says 10"):
        import_tomogram(volume_path=with_size, declared_pixel_size=5.4,
                        out=tmp_path / "one", **common)
    with pytest.raises(ImportRefused, match="--pixel-size"):
        import_tomogram(volume_path=without, out=tmp_path / "two", **common)
    result = import_tomogram(volume_path=without, declared_pixel_size=5.4,
                             out=tmp_path / "three", **common)
    assert result.tomogram.pixel_spacing_angstrom == pytest.approx(5.4)


def test_a_header_with_another_axis_order_is_refused(tmp_path: Path, canonical: np.ndarray) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical)
    with mrcfile.open(volume, mode="r+") as mrc:
        mrc.header.mapc, mrc.header.maps = 3, 1
    angles = write_angles(tmp_path / "volume.tlt")

    with pytest.raises(ImportRefused, match="axis order"):
        import_tomogram(volume_path=volume, tilt_angles_path=angles,
                        orientation_name="pipeline", processing="raw",
                        tilt_series_id="TS_01", out=tmp_path / "project")


def test_a_series_the_project_has_from_its_frames_is_not_attached(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    project = tmp_path / "project"
    (project / "manifests").mkdir(parents=True)
    frames = TiltSeriesManifest(
        tilt_series_id="TS_01",
        images=[TiltImage(z_value=0, tilt_angle_deg=0.0, subframe_path="a.tif", num_subframes=8,
                          pixel_spacing_angstrom=1.35, binning=1)],
    )
    (project / "manifests" / "TS_01.json").write_text(frames.model_dump_json())
    volume = write_mrc(tmp_path / "volume.mrc", canonical)
    angles = write_angles(tmp_path / "volume.tlt")

    with pytest.raises(ImportRefused, match="entered as frames"):
        import_tomogram(volume_path=volume, tilt_angles_path=angles,
                        orientation_name="pipeline", processing="raw",
                        tilt_series_id="TS_01", out=project)


def test_relion5_angles_come_from_the_tilt_series_star(tmp_path: Path) -> None:
    star = tmp_path / "TS_01.star"
    star.write_text(
        "# version 50001\n\ndata_TS_01\n\nloop_\n_rlnMicrographMovieName #1\n"
        "_rlnTomoXTilt #2\n_rlnTomoYTilt #3\n"
        + "".join(f"view{i}.mrc 1.62 {angle:.3f}\n" for i, angle in enumerate(ANGLES))
    )
    read = read_tilt_angles(star)
    assert read.source == "relion5-star"
    assert read.angles == pytest.approx(ANGLES)

    tomograms = tmp_path / "tomograms.star"
    tomograms.write_text("data_global\n\nloop_\n_rlnTomoName #1\nTS_01\n")
    with pytest.raises(ImportRefused, match="rlnTomoYTilt"):
        read_tilt_angles(tomograms)


def test_processing_is_raw_or_a_set_of_known_steps() -> None:
    assert parse_processing("raw") == ()
    assert parse_processing("denoised, ctf-corrected") == ("ctf-corrected", "denoised")
    for bad in ("", "raw,denoised", "isonet"):
        with pytest.raises(ImportRefused):
            parse_processing(bad)


def test_the_steps_before_a_tomogram_refuse_a_series_entered_as_one(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical)
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"
    result = import_tomogram(volume_path=volume, tilt_angles_path=angles,
                             orientation_name="pipeline", processing="raw",
                             tilt_series_id="TS_01", out=project)

    outcome = CliRunner().invoke(cli.app, [
        "reconstruct-tomogram", "--manifest", str(result.manifest_path),
        "--registry", str(project / "artifacts.json"), "--out", str(project),
    ])

    assert outcome.exit_code != 0
    assert "entered the project as tomogram" in outcome.output


def test_the_command_line_imports_and_reports_the_wedge(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", canonical.transpose(1, 0, 2))
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"

    outcome = CliRunner().invoke(cli.app, [
        "import-tomogram", "--volume", str(volume), "--tilt-angles", str(angles),
        "--orientation", "imod-tilt", "--processing", "raw", "--tilt-series-id", "TS_01",
        "--out", str(project),
    ])

    assert outcome.exit_code == 0, outcome.output
    assert "wedge check: agrees" in outcome.output
    refused = CliRunner().invoke(cli.app, [
        "import-tomogram", "--volume", str(volume), "--tilt-angles", str(angles),
        "--orientation", "warp", "--processing", "raw", "--tilt-series-id", "TS_02",
        "--out", str(project),
    ])
    assert refused.exit_code != 0
    assert "not on the list" in refused.output


def test_imports_made_before_provenance_existed_read_as_imported(tmp_path: Path) -> None:
    legacy = Artifact.model_validate({
        "id": "TS_01:tomogram:external-bypass", "kind": "tomogram", "path": "x.zarr",
        "parameters": {"tilt_series_id": "TS_01", "backend": "external-bypass"},
    })
    made_here = Artifact.model_validate({
        "id": "TS_01:picks", "kind": "picks", "path": "p.star",
        "parent_ids": [legacy.id], "parameters": {"tilt_series_id": "TS_01",
                                                  "backend": "pytom-match-pick"},
    })
    stated = Artifact.model_validate({
        "id": "TS_02:tilt_stack", "kind": "tilt_stack", "path": "s.st", "provenance": "derived",
        "parameters": {"tilt_series_id": "TS_02", "backend": "external_stack_import"},
    })

    assert legacy.provenance is Provenance.IMPORTED
    assert made_here.provenance is Provenance.DERIVED
    assert stated.provenance is Provenance.DERIVED
    registry = ArtifactRegistry.empty()
    registry.extend([legacy, made_here])
    assert [artifact.id for artifact in registry.not_derived_lineage(made_here.id)] == [legacy.id]
    assert registry.not_derived_lineage("unknown") == []


def test_a_series_entered_as_frames_keeps_its_frame_fields_required() -> None:
    with pytest.raises(ValueError, match="entered as frames"):
        TiltSeriesManifest(tilt_series_id="TS_01",
                           images=[TiltImage(z_value=0, tilt_angle_deg=0.0)])
    entered = TiltSeriesManifest(tilt_series_id="TS_01", entry=SeriesEntry.TOMOGRAM,
                                 images=[TiltImage(z_value=0, tilt_angle_deg=0.0)])
    with pytest.raises(ValueError, match="correct-motion needs a series entered as frames"):
        entered.require_entry("correct-motion", SeriesEntry.FRAMES)


@pytest.mark.parametrize("low, high", [(-60.0, 30.0), (-40.0, 58.0), (-53.5, 63.8)])
def test_the_wedge_measurement_finds_the_side_of_an_asymmetric_range(
    low: float, high: float
) -> None:
    # The pipeline's own wedge, wedge.py's, imposed on real noise.
    mask = wedge_present_mask_rfft(96, WedgeGeometry(low, high), device=torch.device("cpu"))
    spectrum = np.fft.rfftn(np.random.default_rng(1).standard_normal((96, 96, 96)))
    spectrum *= mask.numpy() + 0.2 * (1 - mask.numpy())
    volume = np.fft.irfftn(spectrum, s=(96, 96, 96), axes=(0, 1, 2))
    angles = [low, high]

    measured = measure_wedge(volume, angles)
    mirrored = measure_wedge(volume[::-1], angles)
    turned_in_x = measure_wedge(volume[:, :, ::-1], angles)
    turned_in_y = measure_wedge(volume[:, ::-1, :], angles)

    assert measured.verdict == "agrees", measured.reason
    assert (measured.fitted_min_deg + measured.fitted_max_deg) / 2 == pytest.approx(
        (low + high) / 2, abs=1.5
    )
    assert mirrored.verdict == "mirrored"
    assert turned_in_x.verdict == "mirrored"
    assert turned_in_y.verdict == "agrees"


def test_the_wedge_measurement_cannot_tell_the_side_of_a_symmetric_range() -> None:
    volume = wedged_noise((64, 96, 96), -60.0, 60.0)
    assert measure_wedge(volume, [-60.0, 60.0]).verdict == "side not testable"
    assert measure_wedge(volume[::-1], [-60.0, 60.0]).verdict == "side not testable"


def test_picks_in_an_imported_tomogram_extract_and_the_stack_knows_its_import(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    # The chain from picking on is unchanged: extraction finds the imported
    # tomogram as it finds its own, and what it makes says, through its lineage,
    # that a volume made elsewhere is upstream.
    volume = write_mrc(tmp_path / "volume.mrc", canonical[::-1])
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"
    result = import_tomogram(volume_path=volume, tilt_angles_path=angles,
                             orientation_name="relion5", processing="raw",
                             tilt_series_id="TS_01", out=project)
    star = tmp_path / "picks.star"
    star.write_text(
        "data_particles\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
        "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n_rlnLCCmax #7\n"
        "40 40 24 0 0 0 0.5\n60 50 20 10 20 30 0.4\n"
    )
    registry_path = project / "artifacts.json"
    registry = ArtifactRegistry.load(registry_path)
    registry.add(import_external_picks_for_extraction(
        star_path=star, tomogram=result.tomogram, tilt_series_id="TS_01", template_box_px=16,
    ))
    registry.commit(registry_path)

    outcome = CliRunner().invoke(cli.app, [
        "extract-particles", "--manifest", str(result.manifest_path),
        "--registry", str(registry_path), "--out", str(project), "--box-size-px", "16",
    ])

    assert outcome.exit_code == 0, outcome.output
    registry = ArtifactRegistry.load(registry_path)
    (stack,) = registry.by_kind(ArtifactKind.PARTICLE_STACK)
    assert stack.provenance is Provenance.DERIVED
    upstream = {artifact.id for artifact in registry.not_derived_lineage(stack.id)}
    assert result.tomogram.id in upstream
