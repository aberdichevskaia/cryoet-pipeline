from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.motion import (
    AverageMotionCorrectionBackend,
    MotionCor3MotionCorrectionBackend,
    PhaseCorrelationMotionCorrectionBackend,
    _estimate_frame_shifts,
    correct_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    ArtifactKind,
    AxisOrder,
    RetentionPolicy,
    StorageRole,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.storage import ArtifactFormat


def test_average_motion_correction_writes_projection_and_registers_artifact(
    tmp_path: Path,
) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    movie_data = np.array(
        [
            [[1, 3], [5, 7]],
            [[3, 5], [7, 9]],
            [[5, 7], [9, 11]],
        ],
        dtype=np.float32,
    )
    _write_mrc(movie_path, movie_data)

    manifest = _manifest(movie_path, num_subframes=3)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"overwrite": False},
    )
    registry = ArtifactRegistry.empty()

    artifacts = correct_and_register(
        AverageMotionCorrectionBackend(),
        manifest,
        context,
        registry,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert registry.get(artifact.id) == artifact
    assert artifact.kind == ArtifactKind.CORRECTED_PROJECTION
    assert artifact.path == tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_avg.mrc"
    assert artifact.shape == (2, 2)
    assert artifact.dtype == "float32"
    assert artifact.axis_order == AxisOrder.YX
    assert artifact.pixel_spacing_angstrom == 1.35
    assert artifact.storage_role == StorageRole.CACHE
    assert artifact.retention_policy == RetentionPolicy.RECOMPUTE
    assert artifact.can_recompute is True
    assert artifact.size_bytes is not None
    assert artifact.size_bytes > 0
    assert artifact.parameters["backend"] == "average"
    assert artifact.parameters["artifact_format"] == "mrc"
    assert artifact.parameters["source_frame_file"] == str(movie_path)

    with mrcfile.open(artifact.path, permissive=True) as corrected:
        np.testing.assert_allclose(corrected.data, movie_data.mean(axis=0))
        assert corrected.data.dtype == np.float32


def test_average_motion_correction_can_write_zarr_cache(tmp_path: Path) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    movie_data = np.array(
        [
            [[1, 2], [3, 4]],
            [[3, 4], [5, 6]],
        ],
        dtype=np.float32,
    )
    _write_mrc(movie_path, movie_data)
    manifest = _manifest(movie_path, num_subframes=2)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "artifact_format": ArtifactFormat.ZARR,
            "storage_role": StorageRole.CACHE,
            "retention_policy": RetentionPolicy.KEEP,
        },
    )

    artifacts = AverageMotionCorrectionBackend().correct(manifest, context)

    assert artifacts[0].path == tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_avg.zarr"
    assert artifacts[0].path.is_dir()
    assert artifacts[0].parameters["artifact_format"] == "zarr"
    assert artifacts[0].retention_policy == RetentionPolicy.KEEP
    assert artifacts[0].size_bytes is not None
    assert artifacts[0].size_bytes > 0
    np.testing.assert_allclose(zarr.open(artifacts[0].path, mode="r")[:], movie_data.mean(axis=0))


def test_average_motion_correction_rejects_missing_local_frame_file(
    tmp_path: Path,
) -> None:
    manifest = _manifest(None)
    context = BackendContext(output_dir=tmp_path, device=DevicePreference.CPU)

    with pytest.raises(ValueError, match="missing local frame file"):
        AverageMotionCorrectionBackend().correct(manifest, context)


def test_average_motion_correction_rejects_single_frame_mrc(tmp_path: Path) -> None:
    movie_path = tmp_path / "single_frame.mrc"
    _write_mrc(movie_path, np.ones((2, 2), dtype=np.float32))
    manifest = _manifest(movie_path)
    context = BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU)

    with pytest.raises(ValueError, match="expected multiframe MRC"):
        AverageMotionCorrectionBackend().correct(manifest, context)


def test_average_motion_correction_preflights_all_movies_before_writing(
    tmp_path: Path,
) -> None:
    complete_path = tmp_path / "frames" / "complete.mrc"
    incomplete_path = tmp_path / "frames" / "incomplete.mrc"
    movie_data = np.ones((2, 2, 2), dtype=np.float32)
    _write_mrc(complete_path, movie_data)
    _write_mrc(incomplete_path, movie_data)
    incomplete_path.write_bytes(incomplete_path.read_bytes()[:-4])
    manifest = TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path=str(complete_path),
                local_frame_file=complete_path,
                num_subframes=2,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
            TiltImage(
                z_value=1,
                tilt_angle_deg=3.0,
                subframe_path=str(incomplete_path),
                local_frame_file=incomplete_path,
                num_subframes=2,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
        ],
    )
    output_dir = tmp_path / "outputs"
    context = BackendContext(output_dir=output_dir, device=DevicePreference.CPU)

    with pytest.raises(ValueError, match="incomplete MRC file"):
        AverageMotionCorrectionBackend().correct(manifest, context)

    assert not output_dir.exists()


def test_average_motion_correction_requires_overwrite_flag(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest = _manifest(movie_path, num_subframes=2)
    context = BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU)
    backend = AverageMotionCorrectionBackend()

    backend.correct(manifest, context)

    with pytest.raises(FileExistsError, match="corrected projection already exists"):
        backend.correct(manifest, context)

    overwrite_context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"overwrite": True},
    )
    artifacts = backend.correct(manifest, overwrite_context)

    assert artifacts[0].path.exists()


def test_average_motion_correction_validates_overwrite_parameter(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest = _manifest(movie_path, num_subframes=2)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"overwrite": "yes"},
    )

    with pytest.raises(TypeError, match="must be a bool"):
        AverageMotionCorrectionBackend().correct(manifest, context)


def test_average_motion_correction_rejects_unknown_artifact_format(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest = _manifest(movie_path, num_subframes=2)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"artifact_format": "jpeg"},
    )

    with pytest.raises(ValueError, match="artifact_format"):
        AverageMotionCorrectionBackend().correct(manifest, context)


def test_motioncor3_backend_runs_patch_correction_and_records_provenance(
    tmp_path: Path,
) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    _write_mrc(movie_path, np.ones((3, 4, 6), dtype=np.float32))
    executable = tmp_path / "bin" / "MotionCor3"
    executable.parent.mkdir()
    executable.touch()
    gain_path = tmp_path / "gain.mrc"
    _write_mrc(gain_path, np.ones((4, 6), dtype=np.float32))
    corrected_data = np.arange(24, dtype=np.float32).reshape(4, 6)
    captured: dict[str, object] = {}

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        captured["cwd"] = cwd
        captured["env"] = env
        output_path = Path(command[command.index("-OutMrc") + 1])
        _write_mrc(output_path, corrected_data)
        log_dir = Path(command[command.index("-LogDir") + 1])
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{output_path.stem}.log").write_text("MotionCor3 log\n")
        alignment_dir = Path(command[command.index("-OutAln") + 1])
        alignment_dir.mkdir(parents=True, exist_ok=True)
        (alignment_dir / f"{output_path.stem}.aln").write_text("alignment\n")
        return subprocess.CompletedProcess(command, 0, stdout="completed", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={
            "motioncor3_executable": executable,
            "motioncor3_gpu_ids": [2, 3],
            "motioncor3_patch_x": 7,
            "motioncor3_patch_y": 5,
            "motioncor3_pixel_size_angstrom": 0.675,
            "motioncor3_gain_reference": gain_path,
            "motioncor3_gain_rotation": 1,
            "motioncor3_gain_flip": 2,
            "motioncor3_version": "1.0.1",
        },
    )

    artifact = MotionCor3MotionCorrectionBackend(fake_runner).correct(
        _manifest(movie_path, num_subframes=3),
        context,
    )[0]

    assert artifact.path == tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_mc3.mrc"
    assert artifact.kind == ArtifactKind.CORRECTED_PROJECTION
    assert artifact.shape == (4, 6)
    assert artifact.dtype == "float32"
    assert artifact.pixel_spacing_angstrom == pytest.approx(0.675)
    assert artifact.parameters["backend"] == "motioncor3"
    assert artifact.parameters["method"] == "motioncor3_patch_correction"
    assert artifact.parameters["patch_grid_xy"] == [7, 5]
    assert artifact.parameters["gpu_ids"] == [2, 3]
    assert artifact.parameters["dose_weighting"] is False
    assert artifact.parameters["aligned_movie_saved"] is False
    assert artifact.parameters["ctf_estimation"] is False
    assert artifact.parameters["motioncor3_executable"] == str(executable.resolve())
    assert artifact.parameters["motioncor3_executable_sha256"] == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )
    assert artifact.software_versions["MotionCor3"] == "1.0.1"
    assert Path(str(artifact.parameters["command_log"])).is_file()
    assert Path(str(artifact.parameters["motioncor3_log"])).is_file()
    assert Path(str(artifact.parameters["alignment_dir"])).is_dir()
    assert captured["cwd"] == tmp_path / "outputs/motion/motioncor3/TS_TEST"

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(executable.resolve())
    assert command[command.index("-InMrc") + 1] == str(movie_path)
    assert command[command.index("-Patch") + 1 : command.index("-Patch") + 3] == ["7", "5"]
    assert command[command.index("-Gpu") + 1 : command.index("-LogDir")] == ["2", "3"]
    assert command[command.index("-PixSize") + 1] == "0.675"
    assert command[command.index("-Align") + 1] == "1"
    assert command[command.index("-OutStack") + 1 : command.index("-OutStack") + 3] == [
        "0",
        "1",
    ]
    assert command[command.index("-FmDose") + 1] == "0"
    assert command[command.index("-Cs") + 1] == "0"
    assert command[command.index("-Gain") + 1] == str(gain_path.resolve())
    assert command[command.index("-RotGain") + 1] == "1"
    assert command[command.index("-FlipGain") + 1] == "2"

    with mrcfile.open(artifact.path, permissive=True) as corrected:
        np.testing.assert_array_equal(corrected.data, corrected_data)
        assert corrected.voxel_size.x == pytest.approx(0.675)


def test_motioncor3_backend_can_canonicalize_output_to_zarr(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 3, 4), dtype=np.float32))
    executable = tmp_path / "MotionCor3"
    executable.touch()
    corrected_data = np.arange(12, dtype=np.float32).reshape(3, 4)

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        _write_mrc(Path(command[command.index("-OutMrc") + 1]), corrected_data[None, ...])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={
            "motioncor3_executable": executable,
            "artifact_format": ArtifactFormat.ZARR,
        },
    )

    artifact = MotionCor3MotionCorrectionBackend(fake_runner).correct(
        _manifest(movie_path, num_subframes=2),
        context,
    )[0]

    assert artifact.path.suffix == ".zarr"
    assert artifact.parameters["artifact_format"] == "zarr"
    np.testing.assert_array_equal(zarr.open(artifact.path, mode="r")[:], corrected_data)


def test_motioncor3_backend_requires_cuda_without_starting_process(
    tmp_path: Path,
) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    called = False

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal called
        del command, cwd, env
        called = True
        raise AssertionError("runner must not be called")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
    )

    with pytest.raises(ValueError, match="requires an NVIDIA CUDA device"):
        MotionCor3MotionCorrectionBackend(fake_runner).correct(
            _manifest(movie_path, num_subframes=2),
            context,
        )

    assert called is False


def test_motioncor3_backend_rejects_missing_executable(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={"motioncor3_executable": tmp_path / "missing-MotionCor3"},
    )

    with pytest.raises(FileNotFoundError, match="MotionCor3 executable not found"):
        MotionCor3MotionCorrectionBackend().correct(
            _manifest(movie_path, num_subframes=2),
            context,
        )


def test_motioncor3_backend_preserves_failure_log_without_output(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    executable = tmp_path / "MotionCor3"
    executable.touch()

    def failing_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="CUDA failed")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={"motioncor3_executable": executable},
    )

    with pytest.raises(RuntimeError, match="CUDA failed"):
        MotionCor3MotionCorrectionBackend(failing_runner).correct(
            _manifest(movie_path, num_subframes=2),
            context,
        )

    final_output = tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_mc3.mrc"
    command_log = tmp_path / "outputs/logs/motioncor3/TS_TEST/TS_TEST_000_mc3.runner.log"
    assert not final_output.exists()
    assert command_log.is_file()
    assert "[exit_code]\n9" in command_log.read_text()


def test_motioncor3_backend_removes_partial_series_after_later_failure(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.mrc"
    second_path = tmp_path / "second.mrc"
    _write_mrc(first_path, np.ones((2, 2, 2), dtype=np.float32))
    _write_mrc(second_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest = TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            _manifest(first_path, num_subframes=2).images[0],
            _manifest(second_path, num_subframes=2).images[0].model_copy(
                update={"z_value": 1}
            ),
        ],
    )
    executable = tmp_path / "MotionCor3"
    executable.touch()
    calls = 0

    def fail_second_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del cwd, env
        calls += 1
        if calls == 1:
            _write_mrc(
                Path(command[command.index("-OutMrc") + 1]),
                np.ones((2, 2), dtype=np.float32),
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 4, stdout="", stderr="second failed")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={"motioncor3_executable": executable},
    )

    with pytest.raises(RuntimeError, match="second failed"):
        MotionCor3MotionCorrectionBackend(fail_second_runner).correct(manifest, context)

    assert calls == 2
    assert not (tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_mc3.mrc").exists()
    assert not (tmp_path / "outputs/corrected/TS_TEST/TS_TEST_001_mc3.mrc").exists()
    assert (
        tmp_path / "outputs/logs/motioncor3/TS_TEST/TS_TEST_001_mc3.runner.log"
    ).is_file()


def test_motioncor3_backend_rejects_wrong_output_shape(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 4, 6), dtype=np.float32))
    executable = tmp_path / "MotionCor3"
    executable.touch()

    def wrong_shape_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        _write_mrc(
            Path(command[command.index("-OutMrc") + 1]),
            np.ones((3, 6), dtype=np.float32),
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={"motioncor3_executable": executable},
    )

    with pytest.raises(ValueError, match="does not match input frame shape"):
        MotionCor3MotionCorrectionBackend(wrong_shape_runner).correct(
            _manifest(movie_path, num_subframes=2),
            context,
        )

    assert not (tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_mc3.mrc").exists()


def test_motioncor3_backend_preflights_all_outputs_before_process(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.mrc"
    second_path = tmp_path / "second.mrc"
    _write_mrc(first_path, np.ones((2, 2, 2), dtype=np.float32))
    _write_mrc(second_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest = TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            _manifest(first_path, num_subframes=2).images[0],
            _manifest(second_path, num_subframes=2).images[0].model_copy(
                update={"z_value": 1}
            ),
        ],
    )
    executable = tmp_path / "MotionCor3"
    executable.touch()
    existing_output = tmp_path / "outputs/corrected/TS_TEST/TS_TEST_001_mc3.mrc"
    _write_mrc(existing_output, np.ones((2, 2), dtype=np.float32))
    calls = 0

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        del command, cwd, env
        calls += 1
        raise AssertionError("runner must not be called")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={"motioncor3_executable": executable},
    )

    with pytest.raises(FileExistsError, match="TS_TEST_001_mc3.mrc"):
        MotionCor3MotionCorrectionBackend(fake_runner).correct(manifest, context)

    assert calls == 0
    assert not (tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_mc3.mrc").exists()


def test_estimate_frame_shifts_returns_corrections_for_known_shifts() -> None:
    rng = np.random.default_rng(42)
    base = rng.standard_normal((64, 64)).astype(np.float32)
    dy, dx = 3, -5
    shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    frames = np.stack([base, shifted])

    shifts = _estimate_frame_shifts(frames)

    assert shifts[0, 0] == pytest.approx(0.0)
    assert shifts[0, 1] == pytest.approx(0.0)
    assert shifts[1, 0] == pytest.approx(-dy, abs=1)
    assert shifts[1, 1] == pytest.approx(-dx, abs=1)


def test_phase_correlation_backend_corrects_known_shift(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((32, 32)).astype(np.float32)
    dy, dx = 2, -3
    shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    movie_data = np.stack([base, shifted, base])

    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    _write_mrc(movie_path, movie_data)
    manifest = _manifest(movie_path, num_subframes=3)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"overwrite": False},
    )

    artifacts = PhaseCorrelationMotionCorrectionBackend().correct(manifest, context)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.kind == ArtifactKind.CORRECTED_PROJECTION
    assert artifact.path == tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_mc.mrc"
    assert artifact.parameters["backend"] == "phase_corr"
    assert artifact.parameters["method"] == "phase_correlation"
    assert "frame_correction_shifts_yx_px" in artifact.parameters
    assert len(artifact.parameters["frame_correction_shifts_yx_px"]) == 3
    assert artifact.shape == (32, 32)
    assert artifact.size_bytes is not None and artifact.size_bytes > 0

    with mrcfile.open(artifact.path, permissive=True) as corrected:
        result = corrected.data
        assert result.shape == (32, 32)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, base, atol=1e-4)


def test_phase_correlation_backend_zero_shift_matches_average(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    frame = rng.standard_normal((32, 32)).astype(np.float32)
    movie_data = np.stack([frame, frame, frame])

    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    _write_mrc(movie_path, movie_data)
    manifest = _manifest(movie_path, num_subframes=3)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
    )

    artifacts = PhaseCorrelationMotionCorrectionBackend().correct(manifest, context)

    with mrcfile.open(artifacts[0].path, permissive=True) as corrected:
        np.testing.assert_allclose(corrected.data, frame, atol=1e-4)


def test_phase_correlation_backend_requires_overwrite_flag(tmp_path: Path) -> None:
    movie_path = tmp_path / "movie.mrc"
    _write_mrc(movie_path, np.ones((2, 16, 16), dtype=np.float32))
    manifest = _manifest(movie_path, num_subframes=2)
    context = BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU)
    backend = PhaseCorrelationMotionCorrectionBackend()

    backend.correct(manifest, context)

    with pytest.raises(FileExistsError):
        backend.correct(manifest, context)


def _manifest(
    movie_path: Path | None,
    *,
    num_subframes: int = 1,
) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frames/TS_TEST_000_0.0.mrc",
                local_frame_file=movie_path,
                num_subframes=num_subframes,
                pixel_spacing_angstrom=5.4,
                binning=4,
            )
        ],
    )


def _write_mrc(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
