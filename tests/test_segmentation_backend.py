from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.segmentation import (
    MemBrainSegSegmentationBackend,
    segment_and_register,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_membrain_seg_writes_canonical_zarr_and_qc(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    executable = tmp_path / "membrain"
    executable.touch()
    model = tmp_path / "membrain.ckpt"
    model.touch()
    captured_command: list[str] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert env
        command_list = list(command)
        captured_command.extend(command_list)
        input_path = Path(command_list[command_list.index("--tomogram-path") + 1])
        with mrcfile.open(input_path, permissive=True) as mrc:
            np.testing.assert_allclose(
                mrc.data,
                np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            )
            assert float(mrc.voxel_size.x) == pytest.approx(13.5)

        output_dir = Path(command_list[command_list.index("--out-folder") + 1])
        output_path = output_dir / "TS_TEST_segmentation.mrc"
        segmentation = np.zeros((2, 3, 4), dtype=np.float32)
        segmentation[:, 1, :] = 1.0
        with mrcfile.new(output_path, overwrite=True) as output:
            output.set_data(segmentation)
        return subprocess.CompletedProcess(command, 0, stdout="segmented", stderr="")

    artifacts = segment_and_register(
        MemBrainSegSegmentationBackend(fake_runner),
        tomogram,
        manifest,
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "membrain_executable": executable,
                "membrain_model": model,
                "membrain_out_pixel_size": 10.0,
                "membrain_connected_component_threshold": 100,
                "membrain_segmentation_threshold": 0.25,
                "membrain_sliding_window_size": 80,
            },
        ),
        registry,
    )

    segmentation, qc_artifact = artifacts
    assert registry.get(segmentation.id) == segmentation
    assert registry.get(qc_artifact.id) == qc_artifact
    assert segmentation.kind == ArtifactKind.SEGMENTATION
    assert segmentation.parent_ids == [tomogram.id]
    assert segmentation.shape == (2, 3, 4)
    assert segmentation.axis_order == AxisOrder.ZYX
    assert segmentation.pixel_spacing_angstrom == pytest.approx(13.5)
    assert segmentation.parameters["tomogram_branch"] == "full"
    assert segmentation.parameters["input_tomogram_id"] == tomogram.id
    assert segmentation.parameters["membrain_checkpoint_path"] == str(model.resolve())
    assert segmentation.parameters["membrain_output_mrc_name"] == "TS_TEST_segmentation.mrc"
    assert captured_command[:2] == [str(executable.resolve()), "segment"]
    assert captured_command[captured_command.index("--ckpt-path") + 1] == str(model.resolve())
    assert captured_command[captured_command.index("--in-pixel-size") + 1] == "13.500000"
    assert captured_command[captured_command.index("--out-pixel-size") + 1] == "10.000000"
    assert captured_command[captured_command.index("--segmentation-threshold") + 1] == "0.250000"
    assert captured_command[captured_command.index("--sliding-window-size") + 1] == "80"
    assert (
        captured_command[captured_command.index("--connected-component-thres") + 1]
        == "100"
    )

    segmentation_zarr = zarr.open(segmentation.path, mode="r")
    assert segmentation_zarr.attrs["axis_order"] == "zyx"
    assert segmentation_zarr.attrs["voxel_spacing_angstrom"] == pytest.approx(13.5)
    assert int(np.count_nonzero(segmentation_zarr[:])) == 8

    report = json.loads(qc_artifact.path.read_text())
    assert report["status"] == QcStatus.PASS.value
    assert report["input_tomogram_id"] == tomogram.id
    assert report["segmentation_id"] == segmentation.id
    assert report["shape"] == [2, 3, 4]
    assert report["foreground_voxel_count"] == 8
    assert report["foreground_fraction"] == pytest.approx(8 / 24)
    assert Path(segmentation.parameters["command_log_path"]).is_file()


def test_membrain_seg_requires_voxel_spacing(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path).model_copy(
        update={"pixel_spacing_angstrom": None}
    )

    with pytest.raises(ValueError, match="voxel spacing is required"):
        MemBrainSegSegmentationBackend().segment(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
            ),
        )


def test_membrain_seg_requires_model_path(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    executable = tmp_path / "membrain"
    executable.touch()

    with pytest.raises(TypeError, match="membrain_model"):
        MemBrainSegSegmentationBackend().segment(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "membrain_executable": executable,
                },
            ),
        )


def _tomogram_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "tomogram.zarr"
    zarr.save(path, np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    return Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        shape=(2, 3, 4),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=13.5,
        binning=10,
        parameters={
            "tilt_series_id": "TS_TEST",
            "tomogram_branch": "full",
        },
    )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frame.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            )
        ],
    )
