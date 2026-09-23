from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.subtomogram import (
    ConeSearchRefinementBackend,
    CubicCropExtractionBackend,
    FftRefinementAveragingBackend,
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
from cryoet_pipeline.particles import (
    read_particle_star,
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

    records = read_particle_star(Path(particle_stack.parameters["metadata_path"]))
    assert len(records) == 1
    assert records[0].rot_deg == pytest.approx(10.0)
    assert records[0].tilt_deg == pytest.approx(20.0)
    assert records[0].psi_deg == pytest.approx(30.0)
    assert records[0].shift_x_px == 0.5
    assert records[0].shift_y_px == 0.5
    assert records[0].shift_z_px == 0.5
    assert particle_stack.parameters["extraction_pose_convention"] == "geometric-center-v1"
    assert stack.attrs["extraction_pose_convention"] == "geometric-center-v1"
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


@pytest.mark.parametrize("box_size", [5, 6])
@pytest.mark.parametrize("xyz", [(10.0, 10.0, 10.0), (10.25, 9.625, 10.5)])
def test_extraction_preserves_the_pick_center_in_the_sampling_pose(tmp_path, box_size, xyz):
    """Even/odd boxes and rounded fractional picks use the same physical centre."""
    tomogram = _tomogram_artifact(tmp_path)
    picks = _picks_artifact(tmp_path, tomogram)
    text = picks.path.read_text()
    header = text[:text.index("10 10 10 10.0")]
    picks.path.write_text(header + " ".join(map(str, xyz)) + " 17 31 -23 0.5\n")
    stack, _ = CubicCropExtractionBackend().extract(
        tomogram, picks, _manifest(),
        BackendContext(
            output_dir=tmp_path / "out", device=DevicePreference.CPU,
            parameters={"box_size_px": box_size},
        ),
    )
    record = read_particle_star(Path(stack.parameters["metadata_path"]))[0]
    start_xyz = np.rint(xyz).astype(int) - box_size // 2
    shift = np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px])
    np.testing.assert_allclose(start_xyz + (box_size - 1) / 2 + shift, xyz)
    assert (record.x_px, record.y_px, record.z_px) == xyz
    assert (record.rot_deg, record.tilt_deg, record.psi_deg) == (17, 31, -23)
    x, y, z = start_xyz
    crop = torch.as_tensor(np.asarray(zarr.open(stack.path, mode="r")[0]))
    np.testing.assert_array_equal(
        crop.numpy(), zarr.open(tomogram.path, mode="r")[
            z:z + box_size, y:y + box_size, x:x + box_size
        ],
    )
    # The input is an affine intensity ramp. Linear interpolation is exact in
    # the interior, letting us check the sampling sign/frame independently of
    # the implementation for both identity and a nontrivial rotation.
    axis = np.arange(box_size) - (box_size - 1) / 2
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.stack([xx, yy, zz], axis=-1)
    for angles in [(0, 0, 0), (17, 31, -23)]:
        matrix = relion_euler_to_matrix(*angles)
        moved = resample_volume_zyx(crop, matrix, shift, device=torch.device("cpu"))
        sample_xyz = offsets @ matrix.T + np.asarray(xyz)
        expected = sample_xyz[..., 2] * 400 + sample_xyz[..., 1] * 20 + sample_xyz[..., 0]
        local_xyz = sample_xyz - start_xyz
        interior = np.all((local_xyz >= 0.01) & (local_xyz <= box_size - 1.01), axis=-1)
        assert interior.sum() >= 20
        np.testing.assert_allclose(moved[interior], expected[interior], atol=0.002)


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
    tmp_path: Path,
    volumes: list[torch.Tensor],
    picked_angles: list[tuple[float, float, float]],
    *,
    tilt_series_id: str = "TS_TEST",
) -> tuple[Artifact, list[str]]:
    box_size = volumes[0].shape[0]
    stack_path = tmp_path / f"particles_{tilt_series_id}.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(len(volumes), box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    for index, volume in enumerate(volumes):
        stack[index] = volume.numpy()

    from cryoet_pipeline.particles import ParticleRecord, write_particle_star

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
    metadata_path = tmp_path / f"particles_{tilt_series_id}.star"
    write_particle_star(metadata_path, records)

    artifact = Artifact(
        id=f"{tilt_series_id}:particle_stack:cubic_crop",
        kind=ArtifactKind.PARTICLE_STACK,
        path=stack_path,
        shape=(len(volumes), box_size, box_size, box_size),
        axis_order=AxisOrder.NZYX,
        pixel_spacing_angstrom=9.504,
        parameters={
            "tilt_series_id": tilt_series_id,
            "metadata_path": str(metadata_path),
        },
    )
    return artifact, [str(metadata_path)]


def _manifest_for(tilt_series_id: str) -> TiltSeriesManifest:
    return _manifest().model_copy(update={"tilt_series_id": tilt_series_id})


def test_averaging_combines_particles_from_several_tilt_series(tmp_path: Path) -> None:
    """The point of averaging, and until now impossible to even record.

    Two stacks with different particle counts, so the per-stack counts have to be
    kept apart rather than inferred: an average that is mostly one series is worth
    seeing as such.
    """

    box_size = 24
    reference = _blob_reference(box_size)
    first_angles = [(0.0, 0.0, 0.0), (30.0, 20.0, -10.0)]
    second_angles = [(-45.0, 60.0, 15.0)]

    first, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in first_angles],
        first_angles,
        tilt_series_id="TS_0006",
    )
    second, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in second_angles],
        second_angles,
        tilt_series_id="TS_0005",
    )

    registry = ArtifactRegistry.empty()
    registry.extend([first, second])

    # Given out of order on purpose: the result must not depend on it.
    artifacts = average_and_register(
        FftRefinementAveragingBackend(),
        [first, second],
        [_manifest_for("TS_0006"), _manifest_for("TS_0005")],
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"num_iterations": 2},
        ),
        registry,
    )

    average_artifact, qc_artifact = artifacts
    report = json.loads(qc_artifact.path.read_text())

    # Sorted, so the same two series in either order give the same answer.
    assert report["tilt_series_ids"] == ["TS_0005", "TS_0006"]
    assert report["particle_stack_ids"] == [second.id, first.id]
    assert report["particle_counts"] == [1, 2]
    assert report["particle_count"] == 3

    # Both stacks are parents, because both contributed.
    assert set(average_artifact.parent_ids) == {first.id, second.id}
    # Named for the scope rather than for one series, which would be a lie.
    assert average_artifact.id.startswith("TS_0005_and_1_more:")


def test_averaging_one_series_keeps_the_name_it_always_had(tmp_path: Path) -> None:
    """A list of one must not rename anything, or every existing id moves."""

    box_size = 24
    reference = _blob_reference(box_size)
    angles = [(0.0, 0.0, 0.0), (30.0, 20.0, -10.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )

    registry = ArtifactRegistry.empty()
    registry.add(stack)
    artifacts = average_and_register(
        FftRefinementAveragingBackend(),
        [stack],
        [_manifest()],
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"num_iterations": 1},
        ),
        registry,
    )

    average_artifact, _ = artifacts
    assert average_artifact.id.startswith("TS_TEST:subtomogram_average:")
    assert "and_" not in average_artifact.id


def test_averaging_refuses_stacks_with_different_box_sizes(tmp_path: Path) -> None:
    """Particles can only be averaged together at one box size."""

    small = _blob_reference(24)
    large = _blob_reference(32)
    first, _ = _particle_stack_artifact(
        tmp_path, [_synthetic_particle(small, true_angles=(0.0, 0.0, 0.0))],
        [(0.0, 0.0, 0.0)], tilt_series_id="TS_0005",
    )
    second, _ = _particle_stack_artifact(
        tmp_path, [_synthetic_particle(large, true_angles=(0.0, 0.0, 0.0))],
        [(0.0, 0.0, 0.0)], tilt_series_id="TS_0006",
    )

    with pytest.raises(ValueError, match="single --box-size-px"):
        FftRefinementAveragingBackend().average(
            [first, second],
            [_manifest_for("TS_0005"), _manifest_for("TS_0006")],
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={"num_iterations": 1},
            ),
        )


def test_averaging_refuses_mismatched_stacks_and_manifests(tmp_path: Path) -> None:
    box_size = 24
    reference = _blob_reference(box_size)
    stack, _ = _particle_stack_artifact(
        tmp_path, [_synthetic_particle(reference, true_angles=(0.0, 0.0, 0.0))],
        [(0.0, 0.0, 0.0)],
    )

    with pytest.raises(ValueError, match="each stack needs the manifest"):
        FftRefinementAveragingBackend().average(
            [stack, stack],
            [_manifest()],
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={},
            ),
        )


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
        [particle_stack],
        [_manifest()],
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
    # The baseline is recorded whether or not a reference was supplied, so
    # "did the refinement move anything" is answerable from the report alone.
    assert report["starting_correlation_mean"] is not None

    records = read_particle_star(Path(average_artifact.parameters["metadata_path"]))
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
    from cryoet_pipeline.particles import write_particle_star

    write_particle_star(metadata_path, [])
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
            [particle_stack],
            [_manifest()],
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={},
            ),
        )


def _write_particle_stack(
    tmp_path: Path,
    volumes: list[torch.Tensor],
    picked_angles: list[tuple[float, float, float]],
    *,
    name: str = "cone-search",
    pixel_spacing_angstrom: float = 10.0,
) -> Artifact:
    from cryoet_pipeline.particles import ParticleRecord, write_particle_star

    box_size = int(volumes[0].shape[0])
    stack_path = tmp_path / f"{name}_particles.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(len(volumes), box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    for index, volume in enumerate(volumes):
        stack[index] = volume.numpy()

    metadata_path = tmp_path / f"{name}_particles.star"
    write_particle_star(
        metadata_path,
        [
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
        ],
    )
    return Artifact(
        id=f"TS_TEST:particle_stack:{name}",
        kind=ArtifactKind.PARTICLE_STACK,
        path=stack_path,
        shape=(len(volumes), box_size, box_size, box_size),
        axis_order=AxisOrder.NZYX,
        pixel_spacing_angstrom=pixel_spacing_angstrom,
        parameters={
            "tilt_series_id": "TS_TEST",
            "metadata_path": str(metadata_path),
        },
    )


def _cone_search_context(tmp_path: Path, **overrides: object) -> BackendContext:
    parameters: dict[str, object] = {
        "num_iterations": 2,
        "cone_range_deg": 20.0,
        "cone_sampling_deg": 10.0,
        "inplane_range_deg": 20.0,
        "inplane_sampling_deg": 10.0,
        "shift_range_px": 1.0,
        "particle_diameter_angstrom": 120.0,
        "target_resolution_angstrom": 20.0,
    }
    parameters.update(overrides)
    return BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters=parameters,
    )


def test_crowther_increment_matches_the_published_formula() -> None:
    from cryoet_pipeline.backends.subtomogram import crowther_angular_increment_deg

    # dtheta = 180 / (pi * r_max * D) with r_max = 1/d_res, so a 300 A particle
    # at 20 A resolution allows steps no finer than degrees(20/300) = 3.82 deg.
    assert crowther_angular_increment_deg(300.0, 20.0) == pytest.approx(3.8197, abs=1e-4)
    # Halving the resolution element halves the increment.
    assert crowther_angular_increment_deg(300.0, 10.0) == pytest.approx(1.9099, abs=1e-4)

    with pytest.raises(ValueError, match="diameter"):
        crowther_angular_increment_deg(0.0, 20.0)
    with pytest.raises(ValueError, match="resolution"):
        crowther_angular_increment_deg(300.0, 0.0)


@pytest.mark.parametrize("missing_wedge", [False, True])
@pytest.mark.parametrize("shift_range", [0.0, 1.0])
def test_cone_backend_respects_zero_shift_window(tmp_path, missing_wedge, shift_range):
    """Test the backend schedule, not just the kernel's already-correct zero case."""
    from cryoet_pipeline.particles import write_particle_star

    reference = _blob_reference(20)
    particles = _write_particle_stack(
        tmp_path, [torch.roll(reference, 1, dims=2)] * 2, [(0, 0, 0)] * 2,
    )
    metadata = Path(particles.parameters["metadata_path"])
    # Freezing means preserving the supplied shifts, not resetting them to zero.
    write_particle_star(metadata, [
        replace(record, shift_x_px=0.25)
        for record in read_particle_star(metadata)
    ])
    reference_path = tmp_path / "reference.mrc"
    with mrcfile.new(reference_path) as mrc:
        mrc.set_data(reference.numpy())
        mrc.voxel_size = 10.0
    artifacts = ConeSearchRefinementBackend().average(
        [particles], [_manifest()],
        _cone_search_context(
            tmp_path, num_iterations=2, cone_range_deg=0.0, inplane_range_deg=0.0,
            shift_range_px=shift_range, initial_reference_path=str(reference_path),
            missing_wedge_weighting=missing_wedge,
        ),
    )
    records = read_particle_star(Path(artifacts[0].parameters["metadata_path"]))
    for record in records:
        assert [record.shift_x_px, record.shift_y_px, record.shift_z_px] == [
            0.25 + shift_range, 0, 0,
        ]


def test_cone_search_refinement_improves_alignment_of_a_single_class(tmp_path: Path) -> None:
    box_size = 20
    reference = _blob_reference(box_size)
    true_angles = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0), (-10.0, 6.0, 9.0)]
    picked_angles = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    volumes = [_synthetic_particle(reference, true_angles=angles) for angles in true_angles]
    particles = _write_particle_stack(tmp_path, volumes, picked_angles)
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    artifacts = average_and_register(
        ConeSearchRefinementBackend(),
        [particles],
        [_manifest()],
        _cone_search_context(tmp_path),
        registry,
    )

    *averages, qc_artifact = artifacts
    assert len(averages) == 1
    assert averages[0].kind == ArtifactKind.SUBTOMOGRAM_AVERAGE
    assert averages[0].id == "TS_TEST:subtomogram_average:cone-search"
    assert averages[0].path.is_file()

    report = json.loads(qc_artifact.path.read_text())
    assert report["num_classes"] == 1
    assert report["classes"][0]["particle_count"] == 3
    assert report["mean_final_correlation"] > 0.9
    # Recorded whether or not a reference was supplied: the final correlation
    # alone has no scale, and says nothing about whether refinement moved
    # anything.
    assert report["starting_correlation_mean"] is not None
    assert report["mean_final_correlation"] > report["starting_correlation_mean"]
    assert report["crowther_increment_deg"] == pytest.approx(
        np.degrees(20.0 / 120.0), abs=1e-6
    )


def _wedge_limited_particles(
    box_size: int, poses: list[tuple[float, float, float]]
) -> list[torch.Tensor]:
    """Particles built like `_synthetic_particle`, then stripped of their wedge.

    The wedge is applied in the LAB frame, after the rotation, because it is a
    property of the reconstruction rather than of the particle's pose — which is
    the whole reason it biases a pose search.
    """

    from cryoet_pipeline.wedge import WedgeGeometry, wedge_present_mask_rfft

    present = wedge_present_mask_rfft(
        box_size, WedgeGeometry(-60.0, 60.0), device=torch.device("cpu")
    )
    reference = _blob_reference(box_size)
    volumes = []
    for angles in poses:
        lab = _synthetic_particle(reference, true_angles=angles)
        volumes.append(
            torch.fft.irfftn(
                torch.fft.rfftn(lab) * present, s=(box_size, box_size, box_size)
            )
        )
    return volumes


def test_wedge_weighting_reaches_the_scorer_and_is_recorded(tmp_path: Path) -> None:
    """A wiring test: the unit tests pin the score, this pins that it is used.

    Without it a mis-wired backend would keep scoring the whole box while every
    test of `cryoet_pipeline.wedge` still passed.
    """

    box_size = 20
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0), (-10.0, 6.0, 9.0)]
    particles = _write_particle_stack(
        tmp_path, _wedge_limited_particles(box_size, poses), poses
    )
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    artifacts = average_and_register(
        ConeSearchRefinementBackend(),
        [particles],
        [_manifest()],
        _cone_search_context(tmp_path, missing_wedge=True),
        registry,
        replace_existing=True,
    )

    report = json.loads(artifacts[-1].path.read_text())
    assert report["missing_wedge_weighting"] is True
    assert report["alignment_score"] == "wedge-constrained-ncc"
    # Taken from the manifest's own tilt angles, which span -60 to 60.
    assert report["wedge_tilt_ranges_deg"] == [[-60.0, 60.0]]
    assert report["wedge_tilt_axis_deg"] == 0.0
    assert report["schema_version"] == 6
    assert report["alignment_frame"] == "reference"
    # Wedge weighting alone does not compensate the reference; that is a separate
    # change and the report has to say which of the two it describes.
    assert report["fourier_compensation"] is False
    assert report["wedge_coverage_floor_fraction"] is None
    assert report["wedge_coverage"] is None
    # And the default still says, in the report, that it is the unweighted score.
    assert "missing wedge" not in " ".join(report["warnings"])


def test_the_report_says_when_the_score_ignores_the_missing_wedge(tmp_path: Path) -> None:
    """The default is the measured-worse score, so the report has to admit it."""

    box_size = 20
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0), (-10.0, 6.0, 9.0)]
    particles = _write_particle_stack(
        tmp_path, _wedge_limited_particles(box_size, poses), poses
    )
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    artifacts = average_and_register(
        ConeSearchRefinementBackend(),
        [particles],
        [_manifest()],
        _cone_search_context(tmp_path),
        registry,
        replace_existing=True,
    )

    report = json.loads(artifacts[-1].path.read_text())
    assert report["missing_wedge_weighting"] is False
    assert report["alignment_score"] == "whole-box-ncc"
    assert report["wedge_tilt_ranges_deg"] == []
    assert report["status"] == QcStatus.WARNING.value
    assert any("missing wedge" in warning for warning in report["warnings"])


def test_a_wedge_and_an_alignment_mask_are_refused_together(tmp_path: Path) -> None:
    """Refused rather than approximated: see the backend's own comment."""

    box_size = 20
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0)]
    particles = _write_particle_stack(
        tmp_path, _wedge_limited_particles(box_size, poses), poses
    )
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    with pytest.raises(ValueError, match="cannot be combined"):
        average_and_register(
            ConeSearchRefinementBackend(),
            [particles],
            [_manifest()],
            _cone_search_context(
                tmp_path, missing_wedge=True, alignment_mask_diameter_angstrom=120.0
            ),
            registry,
        )


def test_a_wedge_range_without_the_flag_is_refused(tmp_path: Path) -> None:
    """A parameter that would be read by nothing is a mistake, not a default."""

    box_size = 20
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0)]
    particles = _write_particle_stack(
        tmp_path, _wedge_limited_particles(box_size, poses), poses
    )
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    with pytest.raises(ValueError, match="read by nothing"):
        average_and_register(
            ConeSearchRefinementBackend(),
            [particles],
            [_manifest()],
            _cone_search_context(tmp_path, wedge_min_tilt_deg=-50.0),
            registry,
        )


def test_pose_change_measures_rotations_not_euler_differences() -> None:
    from cryoet_pipeline.backends.subtomogram import pose_change_summary
    from cryoet_pipeline.particles import ParticleRecord

    def record(angles: tuple[float, float, float], shift: tuple[float, float, float]):
        return ParticleRecord(
            index=0,
            x_px=0.0,
            y_px=0.0,
            z_px=0.0,
            rot_deg=angles[0],
            tilt_deg=angles[1],
            psi_deg=angles[2],
            shift_x_px=shift[0],
            shift_y_px=shift[1],
            shift_z_px=shift[2],
            score=None,
            class_number=1,
            random_subset=1,
        )

    start = (12.0, 34.0, -56.0, 0.0, 0.0, 0.0)
    unchanged = pose_change_summary([start], [record((12.0, 34.0, -56.0), (0.0, 0.0, 0.0))])
    assert unchanged["rotation_deg_mean"] == pytest.approx(0.0, abs=1e-9)
    assert unchanged["unmoved_fraction"] == 1.0
    assert unchanged["shift_px_max"] == 0.0

    # A pure in-plane change of 20 degrees is a 20 degree rotation.
    spun = pose_change_summary([start], [record((12.0, 34.0, -36.0), (0.0, 0.0, 0.0))])
    assert spun["rotation_deg_mean"] == pytest.approx(20.0, abs=1e-6)

    # The case that makes Euler differences useless: at tilt zero only rot+psi
    # is defined, so these two triples are the SAME orientation while their
    # numbers differ by 80 degrees apiece. A summary that subtracted Eulers
    # would report a huge move for a particle that did not turn at all.
    degenerate_start = (40.0, 0.0, -40.0, 0.0, 0.0, 0.0)
    same = pose_change_summary(
        [degenerate_start], [record((-40.0, 0.0, 40.0), (0.0, 0.0, 0.0))]
    )
    # Not exactly zero, and the tolerance says why: acos is ill conditioned near
    # one, so an unmoved pose comes back at about 1e-6 degrees. That is also why
    # `unmoved_fraction` thresholds at a millidegree rather than at zero.
    assert same["rotation_deg_mean"] == pytest.approx(0.0, abs=1e-4)
    assert same["unmoved_fraction"] == 1.0

    moved = pose_change_summary(
        [start, start],
        [
            record((12.0, 34.0, -56.0), (0.0, 0.0, 0.0)),
            record((12.0, 34.0, -56.0), (3.0, 4.0, 0.0)),
        ],
    )
    assert moved["shift_px_mean"] == pytest.approx(2.5)
    assert moved["shift_px_max"] == pytest.approx(5.0)
    assert moved["unmoved_fraction"] == 1.0, "a shift is not a rotation"


def test_the_report_warns_when_the_search_saturates_its_own_range(tmp_path: Path) -> None:
    """The check the correlations could not make.

    Particles are handed poses and a cone wide enough that the search can spend
    most of it. What is asserted is not a resolution but that the report says
    how far the poses went and complains when that is most of what was on offer
    — the failure mode that went unreported twice on real data.
    """

    box_size = 20
    reference = _blob_reference(box_size)
    true_angles = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0), (-10.0, 6.0, 9.0)]
    # Seeded far from the truth, so the search has every reason to move them.
    picked = [(70.0, 40.0, -50.0)] * 3
    volumes = [_synthetic_particle(reference, true_angles=angles) for angles in true_angles]
    particles = _write_particle_stack(tmp_path, volumes, picked)
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    artifacts = average_and_register(
        ConeSearchRefinementBackend(),
        [particles],
        [_manifest()],
        _cone_search_context(
            tmp_path, cone_range_deg=60.0, inplane_range_deg=60.0, num_iterations=1
        ),
        registry,
        replace_existing=True,
    )
    report = json.loads(artifacts[-1].path.read_text())
    change = report["pose_change"]
    assert set(change) == {
        "rotation_deg_mean",
        "rotation_deg_median",
        "rotation_deg_max",
        "unmoved_fraction",
        "shift_px_mean",
        "shift_px_max",
    }
    assert change["rotation_deg_max"] >= change["rotation_deg_mean"] >= 0.0
    assert 0.0 <= change["unmoved_fraction"] <= 1.0
    if change["rotation_deg_mean"] > 0.0:
        assert any("saturating its bounds" in one for one in report["warnings"]) or (
            change["rotation_deg_mean"] <= 0.5 * 2 * 30.0
        )


def test_fourier_compensation_is_recorded_with_its_coverage(tmp_path: Path) -> None:
    box_size = 20
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0), (-10.0, 6.0, 9.0), (40.0, 70.0, 20.0)]
    particles = _write_particle_stack(
        tmp_path, _wedge_limited_particles(box_size, poses), poses
    )
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    artifacts = average_and_register(
        ConeSearchRefinementBackend(),
        [particles],
        [_manifest()],
        _cone_search_context(tmp_path, missing_wedge=True, fourier_compensation=True),
        registry,
        replace_existing=True,
    )

    report = json.loads(artifacts[-1].path.read_text())
    assert report["fourier_compensation"] is True
    assert report["missing_wedge_weighting"] is True
    assert report["wedge_coverage_floor_fraction"] == pytest.approx(0.25)
    coverage = report["wedge_coverage"]
    # The coverage is what says whether the denominator was flat; with four
    # particles and a +-60 wedge it cannot be, so the numbers have to be inside
    # the unit interval and the minimum strictly below the median.
    assert 0.0 <= coverage["minimum"] < coverage["median"] <= 1.0
    assert 0.0 <= coverage["fraction_on_the_floor"] <= 1.0


def test_compensation_needs_a_wedge_to_divide_by(tmp_path: Path) -> None:
    box_size = 20
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0)]
    particles = _write_particle_stack(
        tmp_path, _wedge_limited_particles(box_size, poses), poses
    )
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    with pytest.raises(ValueError, match="needs missing_wedge"):
        average_and_register(
            ConeSearchRefinementBackend(),
            [particles],
            [_manifest()],
            _cone_search_context(tmp_path, fourier_compensation=True),
            registry,
        )


def test_fft_refinement_refuses_wedge_weighting(tmp_path: Path) -> None:
    poses = [(0.0, 0.0, 0.0), (12.0, 8.0, -6.0)]
    particles = _write_particle_stack(tmp_path, _wedge_limited_particles(20, poses), poses)
    registry = ArtifactRegistry.empty()
    registry.add(particles)

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"num_iterations": 1, "missing_wedge": True},
    )
    with pytest.raises(ValueError, match="only by cone-search"):
        average_and_register(
            FftRefinementAveragingBackend(), [particles], [_manifest()], context, registry
        )


@pytest.mark.parametrize("perturbed", [False, True], ids=["preserve-correct-pose", "recover-known-rotation"])
def test_cone_angular_search_recovers_analytic_ground_truth(perturbed: bool) -> None:
    """A bounded geometry gate, not a noisy-particle parameter sweep.

    A noiseless asymmetric phantom has an independently specified orientation.
    Test the two contracts missing from a mere correlation-improvement check:
    preserve that orientation, or recover it from one exact on-grid offset.
    Use the CZII first-iteration angular grid and 60 A reference low-pass,
    but freeze shifts to isolate angular search. Passing rules out a basic
    geometry/search failure here; it does not validate noisy CZII refinement.
    """
    from cryoet_pipeline.backends.subtomogram import _search_multi_reference
    from cryoet_pipeline.filters import low_pass_filter_volume
    from cryoet_pipeline.particles import ParticleRecord
    from cryoet_pipeline.rotations import (
        center_particle_volume,
        cone_offsets,
        matrix_to_relion_euler,
    )

    def rodrigues(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
        # Independent active XYZ rotation; no production Euler/perturbation helper.
        vector = np.asarray(axis, dtype=np.float64)
        x, y, z = vector / np.linalg.norm(vector)
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        angle = np.deg2rad(degrees)
        return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew

    def phantom(xyz: np.ndarray) -> torch.Tensor:
        # Evaluate the density in continuous coordinates, without image resampling.
        values = np.zeros(xyz.shape[:-1], dtype=np.float64)
        for amplitude, centre, width in (
            (1.0, (8, 3, -4), 2.5),
            (0.7, (-5, 7, 3), 3.0),
            (0.5, (-2, -7, 5), 2.0),
            (0.4, (-7, -4, -5), 2.5),
        ):
            values -= amplitude * np.exp(-np.sum((xyz - centre) ** 2, axis=-1) / (2 * width**2))
        return center_particle_volume(torch.from_numpy(values.astype(np.float32)))

    axis = np.arange(48, dtype=np.float64) - 23.5
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    xyz = np.stack((x, y, z), axis=-1)
    truth = rodrigues((1, 2, -1), 67.0)
    # q_lab = truth @ q_reference; row-coordinate arrays therefore use xyz @ truth.
    raw = phantom(xyz @ truth)
    reference = low_pass_filter_volume(
        phantom(xyz), cutoff_angstrom=60.0, pixel_spacing_angstrom=10.012
    )
    azimuth = 2 * 360.0 / 7
    correction = (
        rodrigues((0, 0, 1), -azimuth)
        @ rodrigues((0, 1, 0), -10.0)
        @ rodrigues((0, 0, 1), azimuth)
        @ rodrigues((0, 0, 1), -10.0)
    )
    # This 14.13-degree error has an exact inverse in the specified search grid.
    start = truth @ correction.T if perturbed else truth
    rot, tilt, psi = matrix_to_relion_euler(start)
    record = ParticleRecord(
        index=0, x_px=0.0, y_px=0.0, z_px=0.0,
        rot_deg=rot, tilt_deg=tilt, psi_deg=psi,
        shift_x_px=0.0, shift_y_px=0.0, shift_z_px=0.0,
        score=0.5, class_number=1, random_subset=1,
    )
    candidate = _search_multi_reference(
        raw, [reference], record,
        cone=cone_offsets(30.0, 10.0), in_plane_offsets=[0.0, 10.0, -10.0],
        max_shift_px=0.0, device=torch.device("cpu"),
    )
    error_deg = float(np.degrees(np.arccos(np.clip(
        (np.trace(candidate.matrix @ truth.T) - 1.0) / 2.0, -1.0, 1.0
    ))))
    # Tolerance is for matrix floating-point error, not an allowed fraction of the grid step.
    assert error_deg < 1e-4, f"angular search selected a pose {error_deg:.6f} deg from truth"
    np.testing.assert_array_equal(candidate.shift, np.zeros(3))


def test_cone_search_cannot_take_an_out_of_range_spin_even_if_reference_rewards_it() -> None:
    from cryoet_pipeline.backends.subtomogram import _search_multi_reference
    from cryoet_pipeline.particles import ParticleRecord
    from cryoet_pipeline.rotations import cone_offsets

    device = torch.device("cpu")
    raw = _blob_reference(20)
    current = relion_euler_to_matrix(31.0, 64.0, -12.0)
    # This is the exact accidental 164-degree candidate from the CZII grid,
    # constructed without perturbation_matrix so fixing the helper cannot alter it.
    remote = current @ relion_euler_to_matrix(10.0, 10.0, 154.28571428571428)
    misleading_reference = resample_volume_zyx(raw, remote, np.zeros(3), device=device)
    record = ParticleRecord(
        index=0, x_px=0.0, y_px=0.0, z_px=0.0,
        rot_deg=31.0, tilt_deg=64.0, psi_deg=-12.0,
        shift_x_px=0.0, shift_y_px=0.0, shift_z_px=0.0,
        score=0.5, class_number=1, random_subset=1,
    )
    candidate = _search_multi_reference(
        raw, [misleading_reference], record,
        cone=cone_offsets(30.0, 10.0), in_plane_offsets=[0.0, 10.0, -10.0],
        max_shift_px=0.0, device=device,
    )
    cosine = (np.trace(current.T @ candidate.matrix) - 1.0) / 2.0
    full_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    # The sampled grid has at most a 10-degree tilt and a 10-degree spin.
    assert full_angle <= 20.0 + 1e-6


def test_cone_search_refinement_separates_two_distinct_species(tmp_path: Path) -> None:
    box_size = 20
    axis = torch.arange(box_size, dtype=torch.float32) - (box_size - 1) / 2.0
    z_grid, y_grid, x_grid = torch.meshgrid(axis, axis, axis, indexing="ij")

    # Two clearly different shapes: one elongated along x, one along y.
    species_a = torch.exp(-((x_grid / 5.0) ** 2 + (y_grid / 1.6) ** 2 + (z_grid / 1.6) ** 2))
    species_b = torch.exp(-((x_grid / 1.6) ** 2 + (y_grid / 5.0) ** 2 + (z_grid / 1.6) ** 2))
    volumes = [species_a, species_a.clone(), species_b, species_b.clone()]
    picked = [(0.0, 0.0, 0.0)] * 4
    particles = _write_particle_stack(tmp_path, volumes, picked, name="twoclass")

    artifacts = ConeSearchRefinementBackend().average(
        [particles],
        [_manifest()],
        _cone_search_context(
            tmp_path,
            num_classes=2,
            num_iterations=3,
            cone_range_deg=0.0,
            inplane_range_deg=0.0,
            seed=1,
        ),
    )

    *averages, qc_artifact = artifacts
    assert len(averages) == 2
    assert {artifact.id for artifact in averages} == {
        "TS_TEST:subtomogram_average:cone-search:class01",
        "TS_TEST:subtomogram_average:cone-search:class02",
    }

    records = read_particle_star(Path(averages[0].parameters["metadata_path"]))
    classes = {record.index: record.class_number for record in records}
    # The two copies of each species must end up together, whichever label they get.
    assert classes[0] == classes[1]
    assert classes[2] == classes[3]
    assert classes[0] != classes[2]

    report = json.loads(qc_artifact.path.read_text())
    assert report["num_classes"] == 2
    assert sorted(entry["particle_count"] for entry in report["classes"]) == [2, 2]
    assert report["reassignment_fraction_last_iteration"] == pytest.approx(0.0)


def test_cone_search_refinement_reports_when_sampling_hits_the_crowther_floor(
    tmp_path: Path,
) -> None:
    box_size = 16
    reference = _blob_reference(box_size)
    volumes = [_synthetic_particle(reference, true_angles=(0.0, 0.0, 0.0))]
    particles = _write_particle_stack(tmp_path, volumes, [(0.0, 0.0, 0.0)], name="floor")

    artifacts = ConeSearchRefinementBackend().average(
        [particles],
        [_manifest()],
        _cone_search_context(
            tmp_path,
            num_iterations=4,
            cone_sampling_deg=8.0,
            inplane_sampling_deg=8.0,
            particle_diameter_angstrom=100.0,
            target_resolution_angstrom=20.0,
        ),
    )

    report = json.loads(artifacts[-1].path.read_text())
    floor = np.degrees(20.0 / 100.0)
    assert report["crowther_increment_deg"] == pytest.approx(floor, abs=1e-6)
    # 8 deg halved three times is 1 deg, below the 11.5 deg floor, so it clamps.
    assert report["final_cone_sampling_deg"] == pytest.approx(floor, abs=1e-6)
    assert any("Crowther" in warning for warning in report["warnings"])


def test_cone_search_refinement_rejects_more_classes_than_particles(tmp_path: Path) -> None:
    reference = _blob_reference(12)
    volumes = [_synthetic_particle(reference, true_angles=(0.0, 0.0, 0.0))]
    particles = _write_particle_stack(tmp_path, volumes, [(0.0, 0.0, 0.0)], name="toomany")

    with pytest.raises(ValueError, match="cannot refine 3 classes from 1 particles"):
        ConeSearchRefinementBackend().average(
            [particles],
            [_manifest()],
            _cone_search_context(tmp_path, num_classes=3),
        )


def test_one_half_set_cannot_influence_the_other_half_maps(tmp_path: Path) -> None:
    """The guarantee gold-standard rests on, tested directly rather than argued.

    Change a particle that belongs to half 2 and half 1's map must come back
    bit-identical. That is the whole claim: no particle on one side ever reaches a
    reference the other side is aligned against. It is worth testing as an
    identity rather than through a resolution number, because a resolution can
    look right for the wrong reasons — and because on this backend's local search
    the joint scheme's inflation was not measurable on noise, so a test asserting
    that a number improved would be asserting something not demonstrated.
    """

    import numpy as np
    import torch

    from cryoet_pipeline.backends.subtomogram import _refine_half_sets_fft
    from cryoet_pipeline.particles import ParticleRecord
    from cryoet_pipeline.rotations import (
        center_particle_volume,
        relion_euler_to_matrix,
        resample_volume_zyx,
    )

    box_size = 12
    device = torch.device("cpu")

    def build(perturb_half_two: bool, *, num_iterations: int) -> torch.Tensor:
        volumes, records = [], []
        for index in range(8):
            # Half 2 is the odd indices. Perturbing only those must leave half 1 alone.
            seed = index if not (perturb_half_two and index % 2 == 1) else 100 + index
            noise = np.random.default_rng(seed).normal(size=(box_size,) * 3)
            volumes.append(
                center_particle_volume(torch.as_tensor(noise.astype(np.float32)))
            )
            records.append(
                ParticleRecord(
                    index=index,
                    x_px=0.0,
                    y_px=0.0,
                    z_px=0.0,
                    rot_deg=0.0,
                    tilt_deg=0.0,
                    psi_deg=0.0,
                    shift_x_px=0.0,
                    shift_y_px=0.0,
                    shift_z_px=0.0,
                    score=0.5,
                    class_number=1,
                    random_subset=1 + (index % 2),
                )
            )
        aligned = torch.stack(
            [
                resample_volume_zyx(
                    volumes[record.index],
                    relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
                    np.zeros(3),
                    device=device,
                )
                for record in records
            ]
        )
        result = _refine_half_sets_fft(
            records=records,
            raw_volumes=volumes,
            aligned=aligned,
            start_reference=None,
            num_iterations=num_iterations,
            initial_angular_step_deg=20.0,
            initial_shift_step_px=1.0,
            pixel_spacing_angstrom=5.4,
            initial_low_pass_angstrom=60.0,
            finest_low_pass_angstrom=10.8,
            low_pass_margin_shells=2.0,
            device=device,
        )
        return result.half_references[1]

    # One iteration: every particle is aligned to a reference built only from its
    # own half, and nothing has crossed between them yet.
    assert torch.equal(
        build(perturb_half_two=False, num_iterations=1),
        build(perturb_half_two=True, num_iterations=1),
    )


def test_the_only_channel_between_the_halves_is_shut_when_the_fsc_says_nothing(
    tmp_path: Path,
) -> None:
    """Exactly one number can cross between the halves, and it is conditional.

    The resolution limit is derived from the FSC between the two half maps, so
    when that FSC is informative half 1's filter — and therefore half 1's map —
    depends on half 2's content. That coupling is deliberate: it is the feedback
    that stops the refinement reaching into frequencies where only noise agrees,
    and a fully sealed pair would have nothing to set its limit from.

    On noise-dominated data the crossing lands on the first or second shell, which
    carries no information, so the schedule holds and nothing crosses at all —
    which is what this test measures, with three iterations and pure noise. The
    companion test above pins the one-iteration case, and
    `test_an_uninformative_fsc_crossing_never_loosens_the_low_pass` pins the rule
    that decides which regime applies. Together they say: the channel exists, it
    carries one scalar, and it is shut unless the halves actually agree somewhere.
    """

    import numpy as np
    import torch

    from cryoet_pipeline.backends.subtomogram import _refine_half_sets_fft
    from cryoet_pipeline.particles import ParticleRecord
    from cryoet_pipeline.rotations import (
        center_particle_volume,
        relion_euler_to_matrix,
        resample_volume_zyx,
    )

    box_size = 12
    device = torch.device("cpu")

    def build(perturb_half_two: bool) -> tuple[torch.Tensor, list[float]]:
        volumes, records = [], []
        for index in range(8):
            seed = index if not (perturb_half_two and index % 2 == 1) else 100 + index
            noise = np.random.default_rng(seed).normal(size=(box_size,) * 3)
            volumes.append(
                center_particle_volume(torch.as_tensor(noise.astype(np.float32)))
            )
            records.append(
                ParticleRecord(
                    index=index,
                    x_px=0.0,
                    y_px=0.0,
                    z_px=0.0,
                    rot_deg=0.0,
                    tilt_deg=0.0,
                    psi_deg=0.0,
                    shift_x_px=0.0,
                    shift_y_px=0.0,
                    shift_z_px=0.0,
                    score=0.5,
                    class_number=1,
                    random_subset=1 + (index % 2),
                )
            )
        aligned = torch.stack(
            [
                resample_volume_zyx(
                    volumes[record.index],
                    relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
                    np.zeros(3),
                    device=device,
                )
                for record in records
            ]
        )
        result = _refine_half_sets_fft(
            records=records,
            raw_volumes=volumes,
            aligned=aligned,
            start_reference=None,
            num_iterations=3,
            initial_angular_step_deg=20.0,
            initial_shift_step_px=1.0,
            pixel_spacing_angstrom=5.4,
            initial_low_pass_angstrom=60.0,
            finest_low_pass_angstrom=10.8,
            low_pass_margin_shells=2.0,
            device=device,
        )
        return result.half_references[1], result.low_pass_trace

    unperturbed, trace = build(perturb_half_two=False)
    perturbed, perturbed_trace = build(perturb_half_two=True)

    # The schedule never moved, so the channel carried nothing...
    assert trace == [60.0, 60.0, 60.0]
    assert perturbed_trace == trace
    # ...and half 1 is therefore untouched by what happened to half 2.
    assert torch.equal(unperturbed, perturbed)


def test_an_uninformative_fsc_crossing_never_loosens_the_low_pass() -> None:
    """The defect the first version of this rule had, pinned as a test.

    A crossing at the first or second Fourier shell is the median outcome on
    noise-dominated data, and it means the halves agree nowhere — not that the map
    is coarse. Bounded only on the fine side, the rule turned that into a limit
    coarser than the operator's own starting filter and then held it: 60 A to 86 A
    at box 48, to 115 A at box 64. Filtering a refinement more heavily than asked
    is worse than holding the start.
    """

    from cryoet_pipeline.backends.subtomogram import next_low_pass_angstrom
    from cryoet_pipeline.fsc import FscCurve

    pixel_spacing = 5.4
    start = 60.0

    def limit(box_size: int, crossing_shell: int) -> float:
        resolution = box_size * pixel_spacing / crossing_shell
        curve = FscCurve(
            resolution_angstrom=[resolution],
            correlation=[0.0],
            threshold=0.143,
            crossing_resolution_angstrom=resolution,
        )
        return next_low_pass_angstrom(
            curve,
            current_angstrom=start,
            coarsest_angstrom=start,
            finest_angstrom=2.0 * pixel_spacing,
            margin_shells=2.0,
            box_size=box_size,
            pixel_spacing_angstrom=pixel_spacing,
        )

    for box_size in (32, 48, 64):
        for crossing_shell in (1, 2):
            assert limit(box_size, crossing_shell) == start
        for crossing_shell in (1, 2, 3, 6, 12, 20):
            assert 2.0 * pixel_spacing <= limit(box_size, crossing_shell) <= start

    # An informative crossing still tightens it, or the feedback would do nothing.
    assert limit(32, 12) < start


def test_a_curve_that_never_crosses_holds_the_low_pass_where_it_was() -> None:
    """No crossing is an absence of an estimate, not an estimate of infinity."""

    from cryoet_pipeline.backends.subtomogram import next_low_pass_angstrom
    from cryoet_pipeline.fsc import FscCurve

    curve = FscCurve(
        resolution_angstrom=[100.0, 50.0],
        correlation=[0.9, 0.8],
        threshold=0.143,
        crossing_resolution_angstrom=None,
    )

    assert (
        next_low_pass_angstrom(
            curve,
            current_angstrom=37.0,
            coarsest_angstrom=60.0,
            finest_angstrom=10.8,
            margin_shells=2.0,
            box_size=48,
            pixel_spacing_angstrom=5.4,
        )
        == 37.0
    )


def test_the_output_label_is_the_only_thing_variant_does_to_the_backend(
    tmp_path: Path,
) -> None:
    """So two averaging settings over one set of stacks can coexist.

    `--variant` selects inputs and labels outputs with the same string, which makes
    the thing its own docstring names as its purpose — a parameter sweep —
    impossible: a fresh label finds no particle stacks, and reusing the old one
    overwrites the first result. The CLI now resolves inputs by `--variant` and
    labels the output by `--output-variant`, which is only sound because the
    backend uses the label for nothing but naming. That is what this pins: same
    particles, same parameters, two labels, identical content under different ids
    and paths.
    """

    box_size = 24
    reference = _blob_reference(box_size)
    angles = [(0.0, 0.0, 0.0), (30.0, 20.0, -10.0), (-15.0, 40.0, 5.0), (10.0, -25.0, 60.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )

    def run(label: str) -> Artifact:
        registry = ArtifactRegistry.empty()
        registry.add(stack)
        average_artifact, _ = average_and_register(
            FftRefinementAveragingBackend(),
            [stack],
            [_manifest()],
            BackendContext(
                output_dir=tmp_path / label,
                device=DevicePreference.CPU,
                parameters={"num_iterations": 1, "variant": label},
            ),
            registry,
        )
        return average_artifact

    first = run("sweep-a")
    second = run("sweep-b")

    # Different names and different places, so neither replaces the other...
    assert first.id != second.id
    assert first.path != second.path
    assert first.variant == "sweep-a"
    assert second.variant == "sweep-b"
    # ...and identical maps, because the label reached nothing but the naming.
    import mrcfile
    import numpy as np

    with mrcfile.open(first.path) as one, mrcfile.open(second.path) as other:
        np.testing.assert_array_equal(one.data, other.data)


def test_a_zero_search_step_offers_one_candidate_not_three_identical_ones() -> None:
    """Zero means "do not search this axis", and saying so costs 27x less.

    Three offsets of (-0, 0, +0) are the same orientation resampled three times,
    and the rotation grid is the cube of that: 27 identical candidates, then 27
    identical shifts. Collapsing it is both faster and clearer about intent.
    """

    from cryoet_pipeline.backends.subtomogram import _search_deltas

    assert _search_deltas(0.0) == (0.0,)
    assert _search_deltas(6.0) == (-6.0, 0.0, 6.0)


def test_averaging_accepts_a_zero_step_and_keeps_the_orientations_it_was_given(
    tmp_path: Path,
) -> None:
    """The mode this exists for: average the picks as template matching left them.

    It is the cleanest test of whether a set of picks is real, because it gives
    the refinement no opportunity to manufacture agreement — so the parameter
    reader has to accept zero. It did not: the CLI advertised `min=0.0` while the
    backend demanded strictly positive, so the option promised something the code
    refused. Pinned end to end through `average_and_register`, not through the
    search function alone, because that gap lived between the two.
    """

    import mrcfile
    import numpy as np

    from cryoet_pipeline.particles import read_particle_star

    box_size = 24
    reference = _blob_reference(box_size)
    angles = [(0.0, 0.0, 0.0), (30.0, 20.0, -10.0), (-15.0, 40.0, 5.0), (10.0, -25.0, 60.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )

    registry = ArtifactRegistry.empty()
    registry.add(stack)
    average_artifact, qc_artifact = average_and_register(
        FftRefinementAveragingBackend(),
        [stack],
        [_manifest()],
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "num_iterations": 1,
                "initial_angular_step_deg": 0.0,
                "initial_shift_step_px": 0.0,
                "low_pass_margin_shells": 0.0,
            },
        ),
        registry,
    )

    with mrcfile.open(average_artifact.path) as handle:
        assert np.isfinite(handle.data).all()

    refined = read_particle_star(
        Path(str(average_artifact.parameters["metadata_path"]))
    )
    for record, (rot, tilt, psi) in zip(refined, angles, strict=True):
        assert (record.rot_deg, record.tilt_deg, record.psi_deg) == (rot, tilt, psi)
        assert (record.shift_x_px, record.shift_y_px, record.shift_z_px) == (0.0, 0.0, 0.0)

    report = json.loads(qc_artifact.path.read_text())
    assert report["gold_standard"] is True


def test_the_before_and_after_correlations_are_measured_against_the_same_thing(
    tmp_path: Path,
) -> None:
    """A comparison across two different references is not a before and after.

    The reference is low-passed before the search is scored against it, so a
    starting correlation measured on the UNFILTERED reference and a final one
    measured on the filtered reference answer different questions — and the
    difference reads as a refinement that made the alignment worse. An existing
    cone-search test caught this on the multi-class path the moment filtering was
    added there; the half-set path had the same fault and no test saw it.

    Pinned by the property that has to hold rather than by a number: with a
    starting reference the particles already agree with, refinement must not
    report a fall.
    """

    box_size = 24
    reference = _blob_reference(box_size)
    # Particles at their true orientations, so the start is already good and any
    # apparent decline is a measurement artefact rather than a failed search.
    angles = [(0.0, 0.0, 0.0), (25.0, 15.0, -20.0), (-30.0, 45.0, 10.0), (60.0, -20.0, 35.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )

    registry = ArtifactRegistry.empty()
    registry.add(stack)
    _, qc_artifact = average_and_register(
        FftRefinementAveragingBackend(),
        [stack],
        [_manifest()],
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"num_iterations": 2, "variant": "t"},
        ),
        registry,
    )

    report = json.loads(qc_artifact.path.read_text())
    assert report["mean_final_correlation"] >= report["starting_correlation_mean"]
    assert not any("did not improve" in warning for warning in report["warnings"])


def test_the_agreement_trace_moves_where_the_crossing_cannot(tmp_path: Path) -> None:
    """The crossing is quantised to shells; this is the continuous companion.

    At box 48 with 9.784 A voxels the shells near 36 A are 2.6 A apart, so a real
    refinement improving by a fraction of a shell records an identical crossing
    every iteration — which is what a real run showed: five iterations of
    36.12553992638221, byte-identical, next to a low-pass that had tightened from
    60 to 31 A. That reads as "nothing happened" and cannot be told apart from
    "improved by less than a shell". The mean correlation across shells can.
    """

    box_size = 24
    reference = _blob_reference(box_size)
    angles = [(0.0, 0.0, 0.0), (25.0, 15.0, -20.0), (-30.0, 45.0, 10.0), (60.0, -20.0, 35.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )

    registry = ArtifactRegistry.empty()
    registry.add(stack)
    _, qc_artifact = average_and_register(
        FftRefinementAveragingBackend(),
        [stack],
        [_manifest()],
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"num_iterations": 3, "variant": "t"},
        ),
        registry,
    )

    report = json.loads(qc_artifact.path.read_text())
    agreement = report["half_set_agreement_trace"]
    assert len(agreement) == 3
    assert all(-1.0 <= value <= 1.0 for value in agreement)
    # One entry per iteration, alongside the other two traces, so the three can
    # be read together — which is the only way the loop is legible.
    assert len(report["resolution_trace_angstrom"]) == len(agreement)
    assert len(report["low_pass_trace_angstrom"]) == len(agreement)


def test_averaging_can_be_restricted_to_one_class(tmp_path: Path) -> None:
    """The step between classifying and having a structure.

    Classification writes `rlnClassNumber` per particle and, until this, nothing
    read it: the workflow "classify into K, look at the K maps, refine the one
    that is not junk" had no way to express its last step. Per-series counts come
    back correct too, recovered by arithmetic over the original offsets rather
    than from the records — a record cannot say which tilt-series it came from.
    """

    from cryoet_pipeline.backends.subtomogram import (
        _CombinedParticles,
        filter_combined_by_class,
    )
    from cryoet_pipeline.particles import ParticleRecord

    def record(index: int, class_number: int) -> ParticleRecord:
        return ParticleRecord(
            index=index,
            x_px=0.0,
            y_px=0.0,
            z_px=0.0,
            rot_deg=0.0,
            tilt_deg=0.0,
            psi_deg=0.0,
            shift_x_px=0.0,
            shift_y_px=0.0,
            shift_z_px=0.0,
            score=0.5,
            class_number=class_number,
            random_subset=1 + (index % 2),
        )

    # Two series of three particles. Classes alternate so both series contribute.
    classes = [1, 2, 1, 2, 1, 2]
    combined = _CombinedParticles(
        volumes=[f"volume-{index}" for index in range(6)],
        records=[record(index, classes[index]) for index in range(6)],
        box_size=8,
        tilt_series_ids=["TS_A", "TS_B"],
        stack_ids=["a", "b"],
        counts=[3, 3],
    )

    only_one = filter_combined_by_class(combined, 1)

    assert only_one.particle_count == 3
    assert only_one.volumes == ["volume-0", "volume-2", "volume-4"]
    # Re-indexed to positions in the new list, or every later subscript is wrong.
    assert [one.index for one in only_one.records] == [0, 1, 2]
    # Series 0 held indices 0,1,2 and contributed 0 and 2; series 1 held 3,4,5
    # and contributed 4.
    assert only_one.counts == [2, 1]
    assert sum(only_one.counts) == only_one.particle_count
    assert only_one.tilt_series_ids == ["TS_A", "TS_B"]

    # The half-set assignment travels with each particle, unchanged.
    assert [one.random_subset for one in only_one.records] == [1, 1, 1]


def test_asking_for_a_class_nobody_is_in_says_which_classes_exist(tmp_path: Path) -> None:
    """So a typo in --class-number does not read as "this class is empty"."""

    from cryoet_pipeline.backends.subtomogram import (
        _CombinedParticles,
        filter_combined_by_class,
    )
    from cryoet_pipeline.particles import ParticleRecord

    combined = _CombinedParticles(
        volumes=["v"],
        records=[
            ParticleRecord(
                index=0,
                x_px=0.0,
                y_px=0.0,
                z_px=0.0,
                rot_deg=0.0,
                tilt_deg=0.0,
                psi_deg=0.0,
                shift_x_px=0.0,
                shift_y_px=0.0,
                shift_z_px=0.0,
                score=0.5,
                class_number=2,
                random_subset=1,
            )
        ],
        box_size=8,
        tilt_series_ids=["TS_A"],
        stack_ids=["a"],
        counts=[1],
    )

    with pytest.raises(ValueError, match=r"classes \[2\]"):
        filter_combined_by_class(combined, 5)


def test_refining_one_class_reads_the_class_from_a_supplied_star(tmp_path: Path) -> None:
    """End to end, because the component test passed while the path did not.

    `--class-number` was built for "classify into K, then refine the one that is
    not junk" and could not do it: extraction writes `class_number=1` for every
    particle, classification writes the assignment into its own output STAR in
    `averages/`, and the filter read the stacks. So the flag saw one class always.
    The component test set the classes by hand on a `_CombinedParticles` and
    therefore proved nothing about the path.
    """

    from cryoet_pipeline.particles import read_particle_star, write_particle_star

    box_size = 24
    reference = _blob_reference(box_size)
    angles = [(0.0, 0.0, 0.0), (25.0, 15.0, -20.0), (-30.0, 45.0, 10.0), (60.0, -20.0, 35.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )

    # What a classification would have written: the same particles, same indices,
    # with an assignment the stacks know nothing about.
    extracted = read_particle_star(Path(str(stack.parameters["metadata_path"])))
    assert {one.class_number for one in extracted} == {1}, "extraction assigns one class"
    classified = tmp_path / "classified.star"
    # Split by halves of the list, NOT by parity: the half-set assignment is
    # `1 + index % 2`, so a class following the same parity would put an entire
    # class in one half-set and leave the other empty, which gold-standard
    # refinement refuses. That is a real hazard of filtering by class — with a
    # thousand particles per class it is vanishingly unlikely, and the refusal
    # names it clearly if it happens.
    write_particle_star(
        classified,
        [
            replace(one, class_number=1 if index < 2 else 2)
            for index, one in enumerate(extracted)
        ],
    )

    registry = ArtifactRegistry.empty()
    registry.add(stack)
    _, qc_artifact = average_and_register(
        FftRefinementAveragingBackend(),
        [stack],
        [_manifest()],
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "num_iterations": 1,
                "particle_metadata_path": classified,
                "class_number": 2,
                "variant": "class2",
            },
        ),
        registry,
    )

    report = json.loads(qc_artifact.path.read_text())
    # Two of the four particles were put in class 2, and only those were averaged.
    assert report["particle_count"] == 2
    assert report["particle_counts"] == [2]


def test_a_supplied_star_that_does_not_match_the_stacks_is_refused(tmp_path: Path) -> None:
    """A mismatch would address the wrong particle at every subscript, silently."""

    from cryoet_pipeline.backends.subtomogram import (
        _CombinedParticles,
        replace_records_from_star,
    )
    from cryoet_pipeline.particles import read_particle_star, write_particle_star

    box_size = 24
    reference = _blob_reference(box_size)
    angles = [(0.0, 0.0, 0.0), (25.0, 15.0, -20.0)]
    stack, _ = _particle_stack_artifact(
        tmp_path,
        [_synthetic_particle(reference, true_angles=a) for a in angles],
        angles,
    )
    extracted = read_particle_star(Path(str(stack.parameters["metadata_path"])))

    short = tmp_path / "short.star"
    write_particle_star(short, extracted[:1])
    combined = _CombinedParticles(
        volumes=["a", "b"],
        records=extracted,
        box_size=box_size,
        tilt_series_ids=["TS_TEST"],
        stack_ids=[stack.id],
        counts=[2],
    )
    with pytest.raises(ValueError, match="describes 1 particles but the stacks hold 2"):
        replace_records_from_star(combined, short)

    gapped = tmp_path / "gapped.star"
    write_particle_star(gapped, [replace(one, index=one.index + 5) for one in extracted])
    with pytest.raises(ValueError, match="does not index the stacks contiguously"):
        replace_records_from_star(combined, gapped)
