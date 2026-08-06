from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.subtomogram import (
    CubicCropExtractionBackend,
    FftRefinementAveragingBackend,
    _read_particle_star,
    average_and_register,
    extract_and_register,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.rotations import relion_euler_to_matrix, resample_volume_zyx
from cryoet_pipeline.runtime import DevicePreference

_TOMOGRAM_SHAPE = (20, 20, 20)
_BOX_SIZE = 6


def _manifest(*, max_tilt_deg: float = 60.0) -> TiltSeriesManifest:
    tilt_angles = [-max_tilt_deg, 0.0, max_tilt_deg]
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=9.504,
        images=[
            TiltImage(
                z_value=index,
                tilt_angle_deg=angle,
                subframe_path=f"frame{index}.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=9.504,
                binning=1,
            )
            for index, angle in enumerate(tilt_angles)
        ],
    )


def _tomogram_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "tomogram.zarr"
    data = np.arange(np.prod(_TOMOGRAM_SHAPE), dtype=np.float32).reshape(_TOMOGRAM_SHAPE)
    zarr.save(path, data)
    return Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        shape=_TOMOGRAM_SHAPE,
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=9.504,
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )


def _picks_artifact(tmp_path: Path, tomogram: Artifact) -> Artifact:
    path = tmp_path / "picks.star"
    path.write_text(
        "data_particles\n\n"
        "loop_\n"
        "_rlnCoordinateX #1\n"
        "_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n"
        "_rlnAngleRot #4\n"
        "_rlnAngleTilt #5\n"
        "_rlnAnglePsi #6\n"
        "_rlnLCCmax #7\n"
        "10 10 10 10.0 20.0 30.0 0.500000\n"
        "2 10 10 0.0 0.0 0.0 0.500000\n"
        "10 10 10 5.0 5.0 5.0 0.050000\n"
    )
    return Artifact(
        id="TS_TEST:picks:full:pytom-match-pick",
        kind=ArtifactKind.PICKS,
        path=path,
        parent_ids=[tomogram.id],
        shape=(3,),
        parameters={
            "tilt_series_id": "TS_TEST",
            "input_tomogram_id": tomogram.id,
        },
    )


def test_cubic_crop_extraction_writes_particle_stack_and_drops_bad_rows(
    tmp_path: Path,
) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    picks = _picks_artifact(tmp_path, tomogram)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    registry.add(picks)

    artifacts = extract_and_register(
        CubicCropExtractionBackend(),
        tomogram,
        picks,
        manifest,
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"box_size_px": _BOX_SIZE, "minimum_score": 0.1},
        ),
        registry,
    )

    particle_stack, qc_artifact = artifacts
    assert registry.get(particle_stack.id) == particle_stack
    assert particle_stack.kind == ArtifactKind.PARTICLE_STACK
    assert particle_stack.shape == (1, _BOX_SIZE, _BOX_SIZE, _BOX_SIZE)
    assert particle_stack.parent_ids == [tomogram.id, picks.id]
    assert particle_stack.axis_order == AxisOrder.NZYX

    stack = zarr.open(particle_stack.path, mode="r")
    tomogram_data = zarr.open(tomogram.path, mode="r")[:]
    np.testing.assert_allclose(stack[0], tomogram_data[7:13, 7:13, 7:13])

    records = _read_particle_star(Path(particle_stack.parameters["metadata_path"]))
    assert len(records) == 1
    assert records[0].rot_deg == pytest.approx(10.0)
    assert records[0].tilt_deg == pytest.approx(20.0)
    assert records[0].psi_deg == pytest.approx(30.0)
    assert records[0].shift_x_px == 0.0
    assert records[0].score == pytest.approx(0.5)

    report = json.loads(qc_artifact.path.read_text())
    assert report["requested_particle_count"] == 3
    assert report["extracted_particle_count"] == 1
    assert report["dropped_edge_count"] == 1
    assert report["dropped_score_count"] == 1
    assert report["status"] == QcStatus.PASS.value


def test_cubic_crop_extraction_fails_when_all_particles_dropped(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    picks = _picks_artifact(tmp_path, tomogram)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    registry.add(picks)

    artifacts = extract_and_register(
        CubicCropExtractionBackend(),
        tomogram,
        picks,
        manifest,
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"box_size_px": _BOX_SIZE, "minimum_score": 0.9},
        ),
        registry,
    )

    particle_stack, qc_artifact = artifacts
    assert particle_stack.shape == (0, _BOX_SIZE, _BOX_SIZE, _BOX_SIZE)
    report = json.loads(qc_artifact.path.read_text())
    assert report["status"] == QcStatus.FAIL.value


def test_cubic_crop_extraction_rejects_mismatched_tomogram(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    other_tomogram = Artifact(
        id="TS_OTHER:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=tomogram.path,
        shape=_TOMOGRAM_SHAPE,
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=9.504,
        parameters={"tilt_series_id": "TS_OTHER"},
    )
    picks = _picks_artifact(tmp_path, tomogram)

    with pytest.raises(ValueError, match="tomogram is for TS_OTHER"):
        CubicCropExtractionBackend().extract(
            other_tomogram,
            picks,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={"box_size_px": _BOX_SIZE},
            ),
        )


def _blob_reference(size: int) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float64) - (size - 1) / 2.0
    z_grid, y_grid, x_grid = torch.meshgrid(axis, axis, axis, indexing="ij")
    blob_a = torch.exp(-(((x_grid - 4) ** 2 + y_grid**2 + z_grid**2) / (2 * 2.0**2)))
    blob_b = torch.exp(-((x_grid**2 + (y_grid - 4) ** 2 + (z_grid + 2) ** 2) / (2 * 1.5**2)))
    return (blob_a + 0.6 * blob_b).to(torch.float32)


def _synthetic_particle(
    reference: torch.Tensor, *, true_angles: tuple[float, float, float]
) -> torch.Tensor:
    true_matrix = relion_euler_to_matrix(*true_angles)
    return resample_volume_zyx(
        reference, true_matrix.T, np.zeros(3), device=torch.device("cpu")
    )


def _particle_stack_artifact(
    tmp_path: Path, volumes: list[torch.Tensor], picked_angles: list[tuple[float, float, float]]
) -> tuple[Artifact, list[str]]:
    box_size = volumes[0].shape[0]
    stack_path = tmp_path / "particles.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(len(volumes), box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    for index, volume in enumerate(volumes):
        stack[index] = volume.numpy()

    from cryoet_pipeline.backends.subtomogram import ParticleRecord, _write_particle_star

    records = [
        ParticleRecord(
            index=index,
            x_px=0.0,
            y_px=0.0,
            z_px=0.0,
            rot_deg=angles[0],
            tilt_deg=angles[1],
            psi_deg=angles[2],
            shift_x_px=0.0,
            shift_y_px=0.0,
            shift_z_px=0.0,
            score=0.5,
            class_number=1,
            random_subset=1 + (index % 2),
        )
        for index, angles in enumerate(picked_angles)
    ]
    metadata_path = tmp_path / "particles.star"
    _write_particle_star(metadata_path, records)

    artifact = Artifact(
        id="TS_TEST:particle_stack:cubic_crop",
        kind=ArtifactKind.PARTICLE_STACK,
        path=stack_path,
        shape=(len(volumes), box_size, box_size, box_size),
        axis_order=AxisOrder.NZYX,
        pixel_spacing_angstrom=9.504,
        parameters={
            "tilt_series_id": "TS_TEST",
            "metadata_path": str(metadata_path),
        },
    )
    return artifact, [str(metadata_path)]


def test_fft_refinement_averaging_recovers_known_orientations(tmp_path: Path) -> None:
    box_size = 24
    reference = _blob_reference(box_size)
    true_angles = [
        (0.0, 0.0, 0.0),
        (30.0, 20.0, -10.0),
        (-45.0, 60.0, 15.0),
    ]
    picked_angles = [
        (5.0, -4.0, 3.0),
        (35.0, 15.0, -14.0),
        (-40.0, 55.0, 20.0),
    ]
    volumes = [
        _synthetic_particle(reference, true_angles=angles) for angles in true_angles
    ]
    particle_stack, _ = _particle_stack_artifact(tmp_path, volumes, picked_angles)

    reference_path = tmp_path / "reference.mrc"
    with mrcfile.new(reference_path, overwrite=True) as mrc:
        mrc.set_data(reference.numpy())
        mrc.voxel_size = 9.504

    registry = ArtifactRegistry.empty()
    registry.add(particle_stack)

    artifacts = average_and_register(
        FftRefinementAveragingBackend(),
        particle_stack,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "num_iterations": 5,
                "initial_angular_step_deg": 6.0,
                "initial_shift_step_px": 2.0,
                "initial_reference_path": reference_path,
            },
        ),
        registry,
    )

    average_artifact, qc_artifact = artifacts
    assert average_artifact.kind == ArtifactKind.SUBTOMOGRAM_AVERAGE
    report = json.loads(qc_artifact.path.read_text())
    assert report["initial_reference_correlation_mean"] > 0.5
    assert report["mean_final_correlation"] > report["initial_reference_correlation_mean"]

    records = _read_particle_star(Path(average_artifact.parameters["metadata_path"]))
    for record, true in zip(records, true_angles, strict=True):
        recovered = relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg)
        expected = relion_euler_to_matrix(*true)
        # Compare rotation matrices (not raw angles) to sidestep Euler gimbal
        # ambiguity; a well-recovered orientation has a matrix close to the
        # true one regardless of which angle triple represents it.
        np.testing.assert_allclose(recovered, expected, atol=0.15)


def test_fft_refinement_averaging_requires_nonempty_stack(tmp_path: Path) -> None:
    box_size = 8
    stack_path = tmp_path / "empty.zarr"
    zarr.open(
        stack_path,
        mode="w",
        shape=(0, box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    metadata_path = tmp_path / "empty.star"
    from cryoet_pipeline.backends.subtomogram import _write_particle_star

    _write_particle_star(metadata_path, [])
    particle_stack = Artifact(
        id="TS_TEST:particle_stack:cubic_crop",
        kind=ArtifactKind.PARTICLE_STACK,
        path=stack_path,
        shape=(0, box_size, box_size, box_size),
        axis_order=AxisOrder.NZYX,
        parameters={"tilt_series_id": "TS_TEST", "metadata_path": str(metadata_path)},
    )

    with pytest.raises(ValueError, match="no particles to average"):
        FftRefinementAveragingBackend().average(
            particle_stack,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={},
            ),
        )
