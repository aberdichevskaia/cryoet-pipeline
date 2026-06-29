from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.stack import SimpleTiltStackBackend, build_stack_and_register
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    RetentionPolicy,
    StorageRole,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.storage import ArtifactFormat


def test_tilt_stack_backend_writes_ordered_st_and_tlt(tmp_path: Path) -> None:
    first_projection = _projection_artifact(tmp_path, z_value=0, value=10)
    second_projection = _projection_artifact(tmp_path, z_value=1, value=20)
    manifest = _manifest(z_values=[1, 0], tilt_angles=[3.0, 0.0])
    registry = ArtifactRegistry.empty()
    registry.extend([first_projection, second_projection])
    context = BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU)

    artifacts = build_stack_and_register(
        SimpleTiltStackBackend(),
        [first_projection, second_projection],
        manifest,
        context,
        registry,
    )

    angles_artifact, stack_artifact = artifacts
    assert angles_artifact.kind == ArtifactKind.TILT_ANGLES
    assert angles_artifact.storage_role == StorageRole.EXPORT
    assert angles_artifact.retention_policy == RetentionPolicy.KEEP
    assert angles_artifact.path.read_text() == "3.000000\n0.000000\n"

    assert stack_artifact.kind == ArtifactKind.TILT_STACK
    assert stack_artifact.path == tmp_path / "outputs/stacks/TS_TEST/TS_TEST.st"
    assert stack_artifact.parent_ids == [second_projection.id, first_projection.id]
    assert stack_artifact.shape == (2, 2, 2)
    assert stack_artifact.dtype == "float32"
    assert stack_artifact.axis_order == AxisOrder.TYX
    assert stack_artifact.parameters["artifact_format"] == "mrc"
    assert stack_artifact.size_bytes is not None
    assert stack_artifact.size_bytes > 0
    assert registry.get(stack_artifact.id) == stack_artifact
    assert registry.get(angles_artifact.id) == angles_artifact

    with mrcfile.open(stack_artifact.path, permissive=True) as stack:
        np.testing.assert_allclose(stack.data[0], np.full((2, 2), 20, dtype=np.float32))
        np.testing.assert_allclose(stack.data[1], np.full((2, 2), 10, dtype=np.float32))


def test_tilt_stack_backend_can_write_zarr_stack(tmp_path: Path) -> None:
    projection = _projection_artifact(tmp_path, z_value=0, value=10)
    manifest = _manifest(z_values=[0], tilt_angles=[0.0])
    registry = ArtifactRegistry.empty()
    registry.add(projection)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "artifact_format": ArtifactFormat.ZARR,
            "storage_role": StorageRole.CACHE,
            "retention_policy": RetentionPolicy.RECOMPUTE,
        },
    )

    artifacts = build_stack_and_register(
        SimpleTiltStackBackend(),
        [projection],
        manifest,
        context,
        registry,
    )
    stack_artifact = artifacts[1]

    assert stack_artifact.path == tmp_path / "outputs/stacks/TS_TEST/TS_TEST.zarr"
    assert stack_artifact.path.is_dir()
    assert stack_artifact.storage_role == StorageRole.CACHE
    assert stack_artifact.retention_policy == RetentionPolicy.RECOMPUTE
    stack = zarr.open(stack_artifact.path, mode="r")
    np.testing.assert_allclose(
        stack[:],
        np.full((1, 2, 2), 10, dtype=np.float32),
    )
    assert stack.chunks[0] == 1


def test_tilt_stack_backend_rejects_missing_projection(tmp_path: Path) -> None:
    projection = _projection_artifact(tmp_path, z_value=0, value=10)
    manifest = _manifest(z_values=[0, 1], tilt_angles=[0.0, 3.0])
    registry = ArtifactRegistry.empty()
    registry.add(projection)
    context = BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU)

    with pytest.raises(ValueError, match="missing corrected projection"):
        build_stack_and_register(
            SimpleTiltStackBackend(),
            [projection],
            manifest,
            context,
            registry,
        )


def test_tilt_stack_backend_rejects_mismatched_projection_shapes(tmp_path: Path) -> None:
    first = _projection_artifact(tmp_path, z_value=0, value=10)
    second_path = tmp_path / "projection_1.mrc"
    _write_mrc(second_path, np.full((3, 2), 20, dtype=np.float32))
    second = Artifact(
        id="TS_TEST:corrected_projection:001",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=second_path,
        parameters={"z_value": 1},
    )
    manifest = _manifest(z_values=[0, 1], tilt_angles=[0.0, 3.0])
    registry = ArtifactRegistry.empty()
    registry.extend([first, second])
    context = BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU)

    with pytest.raises(ValueError, match="must share a shape"):
        build_stack_and_register(
            SimpleTiltStackBackend(),
            [first, second],
            manifest,
            context,
            registry,
        )


def _projection_artifact(tmp_path: Path, *, z_value: int, value: float) -> Artifact:
    path = tmp_path / f"projection_{z_value}.mrc"
    _write_mrc(path, np.full((2, 2), value, dtype=np.float32))
    return Artifact(
        id=f"TS_TEST:corrected_projection:{z_value:03d}",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=path,
        shape=(2, 2),
        dtype="float32",
        axis_order=AxisOrder.YX,
        parameters={"z_value": z_value},
    )


def _manifest(z_values: list[int], tilt_angles: list[float]) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=z_value,
                tilt_angle_deg=tilt_angle,
                subframe_path=f"frames/TS_TEST_{z_value:03d}_{tilt_angle}.mrc",
                num_subframes=2,
                pixel_spacing_angstrom=5.4,
                binning=4,
            )
            for z_value, tilt_angle in zip(z_values, tilt_angles, strict=True)
        ],
    )


def _write_mrc(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
