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
from cryoet_pipeline.backends.picking import PytomMatchPickBackend, pick_and_register
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_pytom_match_pick_writes_star_and_qc(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    segmentation = _segmentation_artifact(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    registry.add(segmentation)

    match_executable = tmp_path / "pytom_match_template.py"
    match_executable.touch()
    extract_executable = tmp_path / "pytom_extract_candidates.py"
    extract_executable.touch()
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()

    captured_commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert env
        command_list = list(command)
        captured_commands.append(command_list)
        if command_list[0] == str(match_executable.resolve()):
            tomogram_path = Path(command_list[command_list.index("-v") + 1])
            with mrcfile.open(tomogram_path, permissive=True) as mrc:
                np.testing.assert_allclose(
                    mrc.data, np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                )
            mask_path = Path(command_list[command_list.index("--tomogram-mask") + 1])
            assert mask_path.is_file()
            destination = Path(command_list[command_list.index("-d") + 1])
            (destination / "TS_TEST_job.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0, stdout="matched", stderr="")

        job_json = Path(command_list[command_list.index("-j") + 1])
        assert job_json.is_file()
        star_path = job_json.parent / "TS_TEST_particles.star"
        star_path.write_text(
            "data_\n\n"
            "loop_\n"
            "_rlnCoordinateX #1\n"
            "_rlnCoordinateY #2\n"
            "_rlnCoordinateZ #3\n"
            "_rlnLCCmax #4\n"
            "1.000000 2.000000 3.000000 0.500000\n"
            "4.000000 5.000000 6.000000 0.700000\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="extracted", stderr="")

    artifacts = pick_and_register(
        PytomMatchPickBackend(fake_runner),
        tomogram,
        manifest,
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CUDA,
            parameters={
                "pytom_match_template_executable": match_executable,
                "pytom_extract_candidates_executable": extract_executable,
                "pytom_template": template,
                "pytom_template_mask": template_mask,
                "pytom_gpu_ids": "0",
                "pytom_particle_diameter": 100.0,
            },
        ),
        registry,
        supporting_artifacts=[segmentation],
    )

    picks, qc_artifact = artifacts
    assert registry.get(picks.id) == picks
    assert registry.get(qc_artifact.id) == qc_artifact
    assert picks.kind == ArtifactKind.PICKS
    assert picks.parent_ids == [tomogram.id, segmentation.id]
    assert picks.shape == (2,)
    assert picks.pixel_spacing_angstrom == pytest.approx(13.5)
    assert picks.parameters["input_tomogram_id"] == tomogram.id
    assert picks.parameters["segmentation_id"] == segmentation.id
    assert picks.parameters["pytom_template_path"] == str(template.resolve())

    match_command = captured_commands[0]
    assert match_command[:1] == [str(match_executable.resolve())]
    assert match_command[match_command.index("--particle-diameter") + 1] == "100.000000"
    assert match_command[match_command.index("-g") + 1] == "0"

    extract_command = captured_commands[1]
    assert extract_command[:1] == [str(extract_executable.resolve())]
    assert extract_command[extract_command.index("--tomogram-mask") + 1] is not None

    report = json.loads(qc_artifact.path.read_text())
    assert report["status"] == QcStatus.PASS.value
    assert report["particle_count"] == 2
    assert report["mean_score"] == pytest.approx(0.6)
    assert report["input_tomogram_id"] == tomogram.id
    assert report["segmentation_id"] == segmentation.id
    assert Path(picks.parameters["command_log_path"]).is_file()


def test_pytom_match_pick_requires_cuda_device(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)

    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        PytomMatchPickBackend().pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_template": tmp_path / "template.mrc",
                    "pytom_template_mask": tmp_path / "mask.mrc",
                },
            ),
        )


def test_pytom_match_pick_requires_diameter_or_angular_search(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("command runner should not be invoked")

    with pytest.raises(ValueError, match="exactly one of"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
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


def _segmentation_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "segmentation.zarr"
    zarr.save(path, np.ones((2, 3, 4), dtype=np.float32))
    return Artifact(
        id="TS_TEST:segmentation:full:membrain-seg",
        kind=ArtifactKind.SEGMENTATION,
        path=path,
        parent_ids=["TS_TEST:tomogram:fine"],
        shape=(2, 3, 4),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=13.5,
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
            ),
            TiltImage(
                z_value=1,
                tilt_angle_deg=3.0,
                subframe_path="frame1.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
        ],
    )
