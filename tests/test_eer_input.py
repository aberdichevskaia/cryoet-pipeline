"""EER movies: read from their headers, initialised from Tomography mdocs, run through
MotionCor3 with -InEer. The files are laid out as EMPIAR-11830's are (tests/eer_files.py)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import tifffile
from eer_files import write_eer_like

from cryoet_pipeline.backends.alignment import _tilt_axis_angle
from cryoet_pipeline.backends.motion import (
    AverageMotionCorrectionBackend,
    MotionCor3MotionCorrectionBackend,
    _motioncor3_settings,
    motioncor3_eer_fmint,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import ProjectConfig, TiltImage, TiltSeriesManifest
from cryoet_pipeline.movies import MovieFormat, inspect_movie, read_movie
from cryoet_pipeline.project import find_tilt_series, initialize_project
from cryoet_pipeline.runtime import DevicePreference

EER_NAMES = [
    "Position_1_001_-15.01_20230801_175104_EER.eer",
    "Position_1_002_-12.01_20230801_175126_EER.eer",
    "Position_1_003_-18.01_20230801_175158_EER.eer",
]

# The head of EMPIAR-11830's Position_1 mdoc, three tilts of 41, as Tomography wrote it.
TOMOGRAPHY_MDOC = """DataMode = 6
ImageSize = 4096 4096
ImageFile = Position_1.mrc
PixelSpacing = 1.90
Voltage = 300.00

[T = Tomography: KRIOS-BSL    01-Aug-2023  17:51:04]

[T =   TiltAxisAngle = -174.10  Binning = 1  SpotSize = 7]
"""
TILT = """
[ZValue = {z}]
TiltAngle = {angle}
ExposureDose = 1.79
PixelSpacing = 1.90
Defocus = 3.68
RotationAngle = 84.10
ExposureTime = 3.03
Binning = 1
SubFramePath = \\\\192.168.10.2\\OffloadData\\20230801_chlamy_WebUI_data\\{name}
NumSubFrames = 1
"""


def _tomography_series(tmp_path: Path) -> tuple[Path, Path]:
    frames, mdocs = tmp_path / "frames", tmp_path / "mdocs"
    for name in EER_NAMES:
        write_eer_like(frames / name, 728)
    angles = ["-15.02", "-12.02", "-18.02"]
    text = TOMOGRAPHY_MDOC + "".join(
        TILT.format(z=z, angle=angle, name=name)
        for z, (angle, name) in enumerate(zip(angles, EER_NAMES, strict=True))
    )
    mdocs.mkdir()
    (mdocs / "Position_1.mdoc").write_text(text)
    return frames, mdocs


def _context(output_dir: Path, **parameters: object) -> BackendContext:
    return BackendContext(
        output_dir=output_dir, device=DevicePreference.CPU, parameters=dict(parameters)
    )


# --- reading -------------------------------------------------------------------


def test_an_eer_movie_is_read_from_its_tags_and_metadata(tmp_path: Path) -> None:
    movie = tmp_path / EER_NAMES[0]
    write_eer_like(movie, 728)

    info = inspect_movie(movie)

    assert info.format is MovieFormat.EER
    assert info.shape == (728, 16, 16)
    assert info.pixel_spacing_angstrom == pytest.approx(1.90370344)
    assert info.eer is not None
    assert info.eer.compression == 65001
    assert info.eer.orientation_tag == 2
    assert info.eer.camera == "Falcon 4"
    assert info.eer.gain_reference_named is not None
    assert info.eer.gain_reference_named.endswith("20230428_151632_EER_GainReference.gain")


def test_an_eer_movie_whose_metadata_counts_other_frames_is_refused(tmp_path: Path) -> None:
    movie = tmp_path / "short.eer"
    write_eer_like(movie, 7, declared=9)

    with pytest.raises(ValueError, match="7 EER frames on disk, its metadata declares 9"):
        inspect_movie(movie)


def test_an_eer_movie_that_starts_with_an_integrated_image_is_refused(tmp_path: Path) -> None:
    movie = tmp_path / "integrated.eer"
    write_eer_like(movie, 7, integrated_first=True)

    with pytest.raises(ValueError, match="not EER frames alone"):
        inspect_movie(movie)


def test_a_cut_copy_of_an_eer_movie_does_not_open(tmp_path: Path) -> None:
    movie = tmp_path / "cut.eer"
    write_eer_like(movie, 7)
    movie.write_bytes(movie.read_bytes()[:-500])  # its directories sit at the end

    with pytest.raises(ValueError):
        inspect_movie(movie)


def test_a_tiff_named_eer_is_not_taken_for_one(tmp_path: Path) -> None:
    movie = tmp_path / "plain.eer"
    tifffile.imwrite(movie, np.zeros((3, 8, 8), np.uint8), bigtiff=True, photometric="minisblack")

    with pytest.raises(ValueError, match="no EER acquisition metadata"):
        inspect_movie(movie)


def test_eer_frames_are_left_to_motioncor3(tmp_path: Path) -> None:
    movie = tmp_path / EER_NAMES[0]
    write_eer_like(movie, 7)

    with pytest.raises(ValueError, match="decoded by MotionCor3"):
        read_movie(movie)


# --- init ----------------------------------------------------------------------


def test_a_tomography_series_is_found_by_its_mdoc_and_counted_from_its_files(
    tmp_path: Path,
) -> None:
    frames, mdocs = _tomography_series(tmp_path)
    config = ProjectConfig(frames_dir=frames, mdocs_dir=mdocs, output_dir=tmp_path / "out")

    assert find_tilt_series(config) == ["Position_1"]
    result = initialize_project(config)
    manifest = json.loads(result.manifest_paths[0].read_text())

    images = manifest["images"]
    assert [image["num_subframes"] for image in images] == [728, 728, 728]
    assert [image["metadata"]["mdoc_num_subframes"] for image in images] == [1, 1, 1]
    # The EER headers carry 1.9037 A; the mdoc's 1.90 is the scope's nominal value.
    assert manifest["raw_pixel_spacing_angstrom"] == pytest.approx(1.90370344)
    notes = " ".join(manifest["notes"])
    assert "728 EER frames per tilt" in notes
    assert "written by Thermo Fisher Tomography (Tomography: KRIOS-BSL 01-Aug-2023" in notes
    assert "RotationAngle 84.10, TiltAxisAngle -174.10" in notes


def test_a_tomography_mdoc_gives_no_tilt_axis(tmp_path: Path) -> None:
    frames, mdocs = _tomography_series(tmp_path)
    result = initialize_project(
        ProjectConfig(frames_dir=frames, mdocs_dir=mdocs, output_dir=tmp_path / "out")
    )
    manifest = TiltSeriesManifest.model_validate_json(result.manifest_paths[0].read_text())

    assert all(image.rotation_angle_deg is None for image in manifest.images)
    assert manifest.images[0].metadata["RotationAngle"] == "84.10"
    with pytest.raises(ValueError, match=r"\[align\] tilt_axis_angle"):
        _tilt_axis_angle(manifest, _context(tmp_path))
    assert _tilt_axis_angle(manifest, _context(tmp_path, tilt_axis_angle_deg=85.0)) == 85.0


def test_two_mdocs_for_one_series_are_refused(tmp_path: Path) -> None:
    frames, mdocs = _tomography_series(tmp_path)
    (mdocs / "Position_1.mrc.mdoc").write_text((mdocs / "Position_1.mdoc").read_text())

    with pytest.raises(ValueError, match="two mdocs for Position_1"):
        find_tilt_series(
            ProjectConfig(frames_dir=frames, mdocs_dir=mdocs, output_dir=tmp_path / "out")
        )


# --- MotionCor3 ------------------------------------------------------------------


def _eer_manifest(movie: Path, frames: int = 728) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="Position_1",
        source_mdoc=Path("Position_1.mdoc"),
        raw_pixel_spacing_angstrom=1.96,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=-15.02,
                subframe_path=movie.name,
                local_frame_file=movie,
                num_subframes=frames,
                pixel_spacing_angstrom=1.90,
                binning=1,
            )
        ],
    )


def _motioncor3(
    tmp_path: Path,
    *,
    sum_shape: tuple[int, int] = (16, 16),
    stdout: str = "completed",
    **parameters: object,
):
    executable = tmp_path / "MotionCor3"
    executable.touch()
    captured: list[list[str]] = []

    def fake_runner(
        command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
    ) -> subprocess.CompletedProcess[str]:
        captured.append(list(command))
        output = Path(command[command.index("-OutMrc") + 1])
        with mrcfile.new(output, overwrite=True) as mrc:
            mrc.set_data(np.ones(sum_shape, dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    context = BackendContext(
        output_dir=tmp_path / "out",
        device=DevicePreference.CUDA,
        parameters={"motioncor3_executable": executable, **parameters},
    )
    return MotionCor3MotionCorrectionBackend(fake_runner), context, captured


def test_motioncor3_is_given_an_eer_movie_with_its_frame_groups(tmp_path: Path) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)
    backend, context, captured = _motioncor3(tmp_path)

    (artifact,) = backend.correct(_eer_manifest(movie), context)

    command = captured[0]
    assert command[command.index("-InEer") + 1] == str(movie.resolve())
    assert command[command.index("-EerSampling") + 1] == "1"
    fmint = Path(command[command.index("-FmIntFile") + 1])
    assert fmint.read_text() == "728 72 0\n"
    assert command[command.index("-FmDose") + 1] == "0"
    assert artifact.shape == (16, 16)
    assert artifact.pixel_spacing_angstrom == pytest.approx(1.96)
    assert artifact.parameters["movie_format"] == "eer"
    assert artifact.parameters["frame_binning"] == 1
    assert artifact.parameters["eer_frames"] == 728
    assert artifact.parameters["eer_group_size"] == 72
    assert artifact.parameters["fmint_file"] == str(fmint)


def test_eer_rendered_twice_as_fine_is_binned_back_to_the_movie_pixel(tmp_path: Path) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)
    # A 16 x 16 sensor rendered on 32 x 32, Fourier-binned by 2.
    backend, context, captured = _motioncor3(
        tmp_path, motioncor3_eer_sampling=2, motioncor3_ftbin=2
    )

    (artifact,) = backend.correct(_eer_manifest(movie), context)

    assert captured[0][captured[0].index("-EerSampling") + 1] == "2"
    assert captured[0][captured[0].index("-FtBin") + 1] == "2"
    assert artifact.shape == (16, 16)
    assert artifact.pixel_spacing_angstrom == pytest.approx(1.96)
    assert artifact.parameters["frame_binning"] == 1


def test_eer_sums_finer_than_the_movie_pixel_are_refused(tmp_path: Path) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)
    backend, context, captured = _motioncor3(tmp_path, motioncor3_eer_sampling=2)

    with pytest.raises(ValueError, match="set ftbin to a multiple of 2"):
        backend.correct(_eer_manifest(movie), context)
    assert captured == []


def test_a_gain_motioncor3_could_not_load_stops_the_step(tmp_path: Path) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)
    gain = tmp_path / "gain.gain"
    gain.write_bytes(b"not float")
    backend, context, _ = _motioncor3(
        tmp_path,
        stdout="Warning: invalid pixel type of Gain reference: 1\nGain reference not found.\n",
        motioncor3_gain_reference=gain,
    )

    with pytest.raises(RuntimeError, match="Gain reference not found"):
        backend.correct(_eer_manifest(movie), context)


def test_a_loaded_gain_is_recorded(tmp_path: Path) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)
    gain = tmp_path / "gain.gain"
    gain.write_bytes(b"float gain")
    backend, context, _ = _motioncor3(
        tmp_path,
        stdout="Load gain reference from\n   gain.gain\nGain reference has been loaded.\n",
        motioncor3_gain_reference=gain,
        motioncor3_gain_flip=1,
    )

    (artifact,) = backend.correct(_eer_manifest(movie), context)

    assert artifact.parameters["gain_loaded"] is True
    assert artifact.parameters["gain_flip"] == 1


def test_a_falcon_gain_is_turned_upside_down_for_eer_frames_unless_told(
    tmp_path: Path,
) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)
    gain = tmp_path / "20230807_182744_EER_GainReference.gain"
    gain.write_bytes(b"float gain")
    loaded = "Gain reference has been loaded.\n"
    (tmp_path / "default").mkdir()
    (tmp_path / "given").mkdir()

    backend, context, captured = _motioncor3(
        tmp_path / "default", stdout=loaded, motioncor3_gain_reference=gain
    )
    (artifact,) = backend.correct(_eer_manifest(movie), context)

    command = captured[0]
    assert command[command.index("-Gain") + 1] == str(gain)
    assert command[command.index("-FlipGain") + 1] == "1"
    assert artifact.parameters["gain_flip"] == 1
    assert artifact.parameters["gain_flip_given"] is False

    backend, context, captured = _motioncor3(
        tmp_path / "given",
        stdout=loaded,
        motioncor3_gain_reference=gain,
        motioncor3_gain_flip=0,
    )
    (artifact,) = backend.correct(_eer_manifest(movie), context)

    assert "-FlipGain" not in captured[0]
    assert artifact.parameters["gain_flip"] == 0
    assert artifact.parameters["gain_flip_given"] is True


@pytest.mark.parametrize(
    ("gain_name", "movie_format", "flip"),
    [
        ("gain.gain", MovieFormat.EER, 1),
        ("GAIN.GAIN", MovieFormat.EER, 1),
        ("gain.mrc", MovieFormat.EER, 0),
        (None, MovieFormat.EER, 0),
        ("gain.gain", MovieFormat.TIFF, 0),
        ("gain.gain", MovieFormat.MRC, 0),
    ],
)
def test_only_a_falcon_gain_with_eer_movies_is_flipped_by_default(
    tmp_path: Path, gain_name: str | None, movie_format: MovieFormat, flip: int
) -> None:
    parameters: dict[str, object] = {}
    if gain_name is not None:
        gain = tmp_path / gain_name
        gain.write_bytes(b"gain")
        parameters["motioncor3_gain_reference"] = gain
    _, context, _ = _motioncor3(tmp_path, **parameters)

    settings = _motioncor3_settings(
        _eer_manifest(tmp_path / EER_NAMES[0]), context, movie_format=movie_format
    )

    assert settings.gain_flip == flip
    assert settings.gain_flip_given is False


def test_the_other_motion_backends_leave_eer_to_motioncor3(tmp_path: Path) -> None:
    movie = tmp_path / "frames" / EER_NAMES[0]
    write_eer_like(movie, 728)

    with pytest.raises(ValueError, match="decoded by the motioncor3 backend only"):
        AverageMotionCorrectionBackend().correct(
            _eer_manifest(movie), _context(tmp_path / "out")
        )


def test_the_frame_groups_always_outnumber_the_frames_left_over() -> None:
    # MotionCor3 writes past its array otherwise (CFmIntParam::mCalcIntFms).
    for fractions in range(1, 31):
        for frames in range(fractions, 2001):
            n, group, dose = motioncor3_eer_fmint(frames, fractions).split()
            groups, left = frames // int(group), frames % int(group)
            assert (int(n), dose) == (frames, "0")
            assert groups >= fractions
            assert left < groups, (frames, fractions)
