from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.restoration import IsoNet2RestorationBackend, restore_and_register
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
    TomogramRestorationQc,
)
from cryoet_pipeline.runtime import DevicePreference


def test_isonet2_restoration_writes_canonical_zarr_and_qc(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    executable = tmp_path / "isonet.py"
    executable.touch()
    model = tmp_path / "isonet_model.h5"
    model.touch()
    captured_commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        command_list = list(command)
        captured_commands.append(command_list)
        if command_list[1] == "prepare_star":
            input_dir = Path(command_list[2])
            input_path = input_dir / "TS_TEST.mrc"
            with mrcfile.open(input_path, permissive=True) as mrc:
                np.testing.assert_allclose(
                    mrc.data,
                    np.arange(24, dtype=np.float32).reshape(2, 3, 4),
                )
                assert float(mrc.voxel_size.x) == pytest.approx(13.5)
            Path(command_list[command_list.index("--output_star") + 1]).write_text("star")
            return subprocess.CompletedProcess(command, 0, stdout="star", stderr="")

        output_dir = Path(command_list[command_list.index("--output_dir") + 1])
        output_path = output_dir / "TS_TEST_corrected.mrc"
        with mrcfile.new(output_path, overwrite=True) as output:
            output.set_data(np.full((2, 3, 4), 7.0, dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout="predicted", stderr="")

    artifacts = restore_and_register(
        IsoNet2RestorationBackend(fake_runner),
        tomogram,
        manifest,
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "isonet2_executable": executable,
                "isonet2_model": model,
                "isonet2_gpu_id": "0",
                "isonet2_batch_size": 2,
            },
        ),
        registry,
    )

    restored, qc_artifact = artifacts
    assert registry.get(restored.id) == restored
    assert registry.get(qc_artifact.id) == qc_artifact
    assert restored.kind == ArtifactKind.DENOISED_TOMOGRAM
    assert restored.parent_ids == [tomogram.id]
    assert restored.shape == (2, 3, 4)
    assert restored.axis_order == AxisOrder.ZYX
    assert restored.pixel_spacing_angstrom == pytest.approx(13.5)
    assert restored.parameters["tomogram_branch"] == "full"
    assert restored.parameters["input_tomogram_id"] == tomogram.id
    assert restored.parameters["isonet2_model_path"] == str(model)
    assert restored.parameters["isonet2_output_mrc_name"] == "TS_TEST_corrected.mrc"
    assert captured_commands[0][:3] == [
        str(executable),
        "prepare_star",
        str(captured_commands[0][2]),
    ]
    assert "--pixel_size" in captured_commands[0]
    assert captured_commands[0][captured_commands[0].index("--pixel_size") + 1] == "13.500000"
    assert captured_commands[1][:4] == [
        str(executable),
        "predict",
        captured_commands[1][2],
        str(model),
    ]
    assert captured_commands[1][captured_commands[1].index("--gpuID") + 1] == "0"
    assert captured_commands[1][captured_commands[1].index("--batch_size") + 1] == "2"

    restored_zarr = zarr.open(restored.path, mode="r")
    np.testing.assert_allclose(restored_zarr[:], np.full((2, 3, 4), 7.0, dtype=np.float32))
    assert restored_zarr.attrs["axis_order"] == "zyx"
    assert restored_zarr.attrs["voxel_spacing_angstrom"] == pytest.approx(13.5)

    report = TomogramRestorationQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.PASS
    assert report.input_tomogram_id == tomogram.id
    assert report.restored_tomogram_id == restored.id
    assert report.shape == (2, 3, 4)
    assert report.mean == pytest.approx(7.0)
    assert Path(restored.parameters["command_log_path"]).is_file()


def test_isonet2_restoration_requires_voxel_spacing(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path).model_copy(update={"pixel_spacing_angstrom": None})

    with pytest.raises(ValueError, match="voxel spacing is required"):
        IsoNet2RestorationBackend().denoise(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
            ),
        )


def test_isonet2_restoration_requires_model_path(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    executable = tmp_path / "isonet.py"
    executable.touch()

    with pytest.raises(TypeError, match="isonet2_model"):
        IsoNet2RestorationBackend().denoise(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "isonet2_executable": executable,
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
