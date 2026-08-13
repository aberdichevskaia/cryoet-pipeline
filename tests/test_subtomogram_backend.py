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
    ConeSearchRefinementBackend,
    CubicCropExtractionBackend,
    FftRefinementAveragingBackend,
    average_and_register,
    extract_and_register,
    read_particle_star,
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

    records = read_particle_star(Path(particle_stack.parameters["metadata_path"]))
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

    from cryoet_pipeline.backends.subtomogram import ParticleRecord, write_particle_star

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
    write_particle_star(metadata_path, records)

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
    from cryoet_pipeline.backends.subtomogram import write_particle_star

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
            particle_stack,
            _manifest(),
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
    from cryoet_pipeline.backends.subtomogram import ParticleRecord, write_particle_star

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
        particles,
        _manifest(),
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
    assert report["crowther_increment_deg"] == pytest.approx(
        np.degrees(20.0 / 120.0), abs=1e-6
    )


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
        particles,
        _manifest(),
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
        particles,
        _manifest(),
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
            particles,
            _manifest(),
            _cone_search_context(tmp_path, num_classes=3),
        )
