"""TIFF movies, and tilt metadata read from movie file names instead of mdocs."""

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import tifffile
from typer.testing import CliRunner

from cryoet_pipeline.cli import app
from cryoet_pipeline.ingest.frame_names import manifest_from_frame_names, match_frames
from cryoet_pipeline.models import ProjectConfig
from cryoet_pipeline.movies import MovieFormat, inspect_movie, read_movie
from cryoet_pipeline.project import initialize_project

# The naming of the first local TIFF dataset: tomogram_<series>_<tilt>_<angle>.0.tif
PATTERN = r"tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif"

# Acquisition order of a dose-symmetric scheme: the tilt number is not the angle order.
DOSE_SYMMETRIC = [(0, 0.0), (1, 3.0), (2, -3.0), (3, -6.0), (4, 6.0)]


def _write_tiff(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, frames, photometric="minisblack")


def _write_series(frames_dir: Path, series: str, *, num_frames: int = 4) -> None:
    for tilt, angle in DOSE_SYMMETRIC:
        _write_tiff(
            frames_dir / f"tomogram_{series}_{tilt}_{angle}.tif",
            np.full((num_frames, 6, 8), tilt, dtype=np.uint16),
        )


def test_a_tiff_movie_is_read_as_frames_y_x(tmp_path):
    path = tmp_path / "movie.tif"
    frames = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    _write_tiff(path, frames)

    info = inspect_movie(path)
    assert info.format is MovieFormat.TIFF
    assert info.shape == (3, 4, 5)
    assert info.pixel_spacing_angstrom is None
    np.testing.assert_array_equal(read_movie(path), frames)


def test_a_one_page_tiff_is_not_a_movie(tmp_path):
    path = tmp_path / "image.tif"
    _write_tiff(path, np.zeros((4, 5), dtype=np.uint16))
    with pytest.raises(ValueError, match="multiframe TIFF movie"):
        inspect_movie(path)


def test_eer_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="EER movies are not supported"):
        inspect_movie(tmp_path / "movie.eer")


def test_series_and_tilts_are_read_from_the_names(tmp_path):
    _write_series(tmp_path, "1")
    _write_series(tmp_path, "12")
    (tmp_path / "notes.txt").write_text("not a movie")

    found = match_frames(tmp_path, PATTERN)
    assert sorted(found) == ["1", "12"]

    manifest = manifest_from_frame_names(
        frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="12",
        raw_pixel_spacing_angstrom=1.35,
    )
    assert manifest.source_mdoc is None
    assert manifest.source_frames_pattern == PATTERN
    assert manifest.raw_pixel_spacing_angstrom == pytest.approx(1.35)
    # In tilt-number order, which is the acquisition order, not the angle order.
    assert manifest.tilt_angles_deg == [angle for _, angle in DOSE_SYMMETRIC]
    assert [image.z_value for image in manifest.images] == [0, 1, 2, 3, 4]
    assert [image.metadata["tilt_number"] for image in manifest.images] == [0, 1, 2, 3, 4]
    assert manifest.num_subframes_set == {4}
    assert all(image.local_frame_file.parent == tmp_path for image in manifest.images)
    # Only series 12's movies, each holding its own tilt number as pixel value.
    assert [int(read_movie(image.local_frame_file)[0, 0, 0]) for image in manifest.images] \
        == [0, 1, 2, 3, 4]


def test_tilt_numbers_are_ordered_as_numbers_not_text(tmp_path):
    for tilt, angle in [(10, 30.0), (9, 27.0), (2, 6.0)]:
        _write_tiff(tmp_path / f"tomogram_1_{tilt}_{angle}.tif", np.zeros((2, 2, 2), np.uint8))
    manifest = manifest_from_frame_names(
        frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="1", raw_pixel_spacing_angstrom=1.0
    )
    assert manifest.tilt_angles_deg == [6.0, 27.0, 30.0]


def test_a_tiff_without_resolution_tags_needs_its_pixel_size(tmp_path):
    _write_series(tmp_path, "1")
    with pytest.raises(ValueError, match="no pixel size in its header"):
        manifest_from_frame_names(frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="1")


@pytest.mark.parametrize(
    ("resolution", "unit", "expected"),
    [
        # Pixels per centimetre or per inch, as SerialEM writes them: IMOD 5.1.12's
        # `header` reads both as 0.84 A, as it does the first local dataset.
        (1e8 / 0.84, "CENTIMETER", 0.84),
        (2.54e8 / 0.84, "INCH", 0.84),
        # A print density, and a resolution without a unit: IMOD shows 1.0, its
        # "unknown", for both.
        (72.0, "INCH", None),
        (1e8 / 0.84, "NONE", None),
    ],
)
def test_the_tiff_pixel_size_is_read_from_its_resolution_tags(
    tmp_path, resolution, unit, expected
):
    path = tmp_path / "movie.tif"
    tifffile.imwrite(
        path, np.zeros((3, 4, 5), np.uint8), photometric="minisblack",
        resolution=(resolution, resolution), resolutionunit=unit,
    )
    spacing = inspect_movie(path).pixel_spacing_angstrom
    if expected is None:
        assert spacing is None
    else:
        assert spacing == pytest.approx(expected, rel=1e-6)


def test_a_given_pixel_size_overrides_the_headers_and_is_noted(tmp_path):
    for tilt, angle in DOSE_SYMMETRIC:
        tifffile.imwrite(
            tmp_path / f"tomogram_1_{tilt}_{angle}.tif", np.zeros((2, 4, 5), np.uint8),
            photometric="minisblack", resolution=(1e8 / 0.84, 1e8 / 0.84),
            resolutionunit="CENTIMETER",
        )
    from_headers = manifest_from_frame_names(
        frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="1"
    )
    assert from_headers.raw_pixel_spacing_angstrom == pytest.approx(0.84)
    assert from_headers.notes == []

    overridden = manifest_from_frame_names(
        frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="1",
        raw_pixel_spacing_angstrom=1.68,
    )
    assert overridden.raw_pixel_spacing_angstrom == 1.68
    assert "overriding the 0.84 angstrom in the movie headers" in overridden.notes[0]


def test_mrc_movies_named_by_pattern_take_the_pixel_size_from_their_headers(tmp_path):
    for tilt, angle in DOSE_SYMMETRIC:
        with mrcfile.new(tmp_path / f"TS_01_{tilt:03d}_{angle}.mrc") as movie:
            movie.set_data(np.zeros((3, 2, 2), dtype=np.int8))
            movie.voxel_size = 0.675
    manifest = manifest_from_frame_names(
        frames_dir=tmp_path,
        pattern=r"(?P<series>TS_\d+)_(?P<tilt>\d+)_(?P<angle>-?[\d.]+)\.mrc",
        tilt_series_id="TS_01",
    )
    assert manifest.raw_pixel_spacing_angstrom == pytest.approx(0.675)
    assert manifest.num_subframes_set == {3}


@pytest.mark.parametrize(
    ("pattern", "message"),
    [
        (r"tomogram_(?P<series>\d+)_(\d+)_(?P<angle>.+)\.tif", "missing tilt"),
        (r"tomogram_(?P<series>\d+", "not a regular expression"),
        (r"(?P<series>TS_\d+)_(?P<tilt>\d+)_(?P<angle>.+)\.mrc", "no file .* matches"),
    ],
)
def test_a_pattern_that_cannot_work_is_named(tmp_path, pattern, message):
    _write_series(tmp_path, "1")
    with pytest.raises(ValueError, match=message):
        manifest_from_frame_names(
            frames_dir=tmp_path, pattern=pattern, tilt_series_id="1",
            raw_pixel_spacing_angstrom=1.0,
        )


def test_an_unknown_series_lists_the_ones_found(tmp_path):
    _write_series(tmp_path, "1")
    with pytest.raises(ValueError, match="finds the series 1"):
        manifest_from_frame_names(
            frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="01",
            raw_pixel_spacing_angstrom=1.0,
        )


def test_two_movies_at_one_tilt_number_are_refused(tmp_path):
    _write_series(tmp_path, "1")
    _write_tiff(tmp_path / "tomogram_1_3_9.0.tif", np.zeros((2, 6, 8), np.uint16))
    with pytest.raises(ValueError, match="share tilt number 3"):
        manifest_from_frame_names(
            frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="1",
            raw_pixel_spacing_angstrom=1.0,
        )


def test_init_writes_manifests_from_the_names(tmp_path):
    frames = tmp_path / "frames"
    _write_series(frames, "1")
    _write_series(frames, "2", num_frames=5)

    result = initialize_project(
        ProjectConfig(
            frames_dir=frames,
            frames_pattern=PATTERN,
            raw_pixel_spacing_angstrom=1.35,
            output_dir=tmp_path / "out",
            tilt_series=["1", "2"],
        )
    )
    manifests = [json.loads(path.read_text()) for path in result.manifest_paths]
    assert [m["tilt_series_id"] for m in manifests] == ["1", "2"]
    assert [m["images"][0]["num_subframes"] for m in manifests] == [4, 5]
    assert all(m["raw_pixel_spacing_angstrom"] == 1.35 for m in manifests)


def test_init_needs_exactly_one_metadata_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one of mdocs_dir"):
        ProjectConfig(frames_dir=tmp_path, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="exactly one of mdocs_dir"):
        ProjectConfig(
            frames_dir=tmp_path, mdocs_dir=tmp_path, frames_pattern=PATTERN,
            output_dir=tmp_path / "out",
        )


def test_the_init_command_reports_what_the_pattern_found(tmp_path):
    frames = tmp_path / "frames"
    _write_series(frames, "1")
    result = CliRunner().invoke(
        app,
        ["init", "--frames", str(frames), "--frames-pattern", PATTERN,
         "--pixel-size", "1.35", "--out", str(tmp_path / "out"), "--tilt-series", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "5 tilts, -6 to 6 deg, 4 frames per tilt, 1.35 A/px" in result.output


def _first_dataset_names(series: str) -> list[str]:
    """The folder of the first local dataset: tomogram_039_001_-42.0.tif to
    tomogram_039_029_+42.0.tif. The tilt number follows the angle, the angle is
    signed and zero-padded, and the zero tilt is written -00.0."""
    names = []
    for index in range(29):
        angle = -42.0 + 3.0 * index
        text = "-00.0" if angle == 0 else f"{angle:+05.1f}"
        names.append(f"tomogram_{series}_{index + 1:03d}_{text}.tif")
    return names


def test_the_first_local_dataset_is_read_whole(tmp_path):
    for name in _first_dataset_names("039") + _first_dataset_names("096"):
        _write_tiff(tmp_path / name, np.zeros((2, 2, 2), dtype=np.uint16))
    # The copies macOS leaves on a shared drive are neither frames nor refused.
    (tmp_path / "._tomogram_039_001_-42.0.tif").write_bytes(b"\0")

    assert sorted(match_frames(tmp_path, PATTERN)) == ["039", "096"]
    manifest = manifest_from_frame_names(
        frames_dir=tmp_path, pattern=PATTERN, tilt_series_id="039",
        raw_pixel_spacing_angstrom=1.0,
    )
    assert manifest.num_tilts == 29
    assert manifest.tilt_angles_deg == [-42.0 + 3.0 * index for index in range(29)]
    assert manifest.images[15].subframe_path == "tomogram_039_016_+03.0.tif"


def test_a_pattern_that_misses_some_frames_is_refused(tmp_path):
    for name in _first_dataset_names("039"):
        _write_tiff(tmp_path / name, np.zeros((2, 2, 2), dtype=np.uint16))
    # The first pattern written for this dataset: no "+", so it matched the 15
    # tilts from -42.0 to -00.0 and would have reconstructed from half a series.
    without_plus = r"tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>-?\d+(?:\.\d+)?)\.tif"
    with pytest.raises(ValueError, match=r"14 \.tif files .* tomogram_039_016_\+03\.0\.tif"):
        match_frames(tmp_path, without_plus)


def test_init_without_series_takes_every_series_the_pattern_finds(tmp_path):
    frames = tmp_path / "frames"
    _write_series(frames, "1")
    _write_series(frames, "2")
    result = CliRunner().invoke(
        app,
        ["init", "--frames", str(frames), "--frames-pattern", PATTERN,
         "--pixel-size", "1.35", "--out", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "found 2 series: 1 2" in result.output
    project = json.loads((tmp_path / "out" / "project.json").read_text())
    assert project["tilt_series"] == ["1", "2"]
    # What was asked for stays recorded apart from what was found.
    assert project["config"]["tilt_series"] == []
