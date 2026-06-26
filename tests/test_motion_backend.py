from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.motion import (
    AverageMotionCorrectionBackend,
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
