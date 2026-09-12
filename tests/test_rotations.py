from __future__ import annotations

import numpy as np
import pytest
import torch

from cryoet_pipeline.rotations import (
    cone_offsets,
    matrix_to_relion_euler,
    perturbation_matrix,
    relion_euler_to_matrix,
    resample_volume_zyx,
)


def test_relion_euler_to_matrix_zero_angles_is_identity() -> None:
    matrix = relion_euler_to_matrix(0.0, 0.0, 0.0)
    np.testing.assert_allclose(matrix, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "rot,tilt,psi",
    [
        (0.0, 0.0, 0.0),
        (37.0, 12.0, -84.0),
        (180.0, 90.0, 45.0),
        (-63.0, 155.0, 170.0),
    ],
)
def test_relion_euler_to_matrix_is_orthogonal_with_unit_determinant(
    rot: float, tilt: float, psi: float
) -> None:
    matrix = relion_euler_to_matrix(rot, tilt, psi)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-10)


def test_relion_euler_to_matrix_pure_rot_is_negative_z_rotation() -> None:
    matrix = relion_euler_to_matrix(90.0, 0.0, 0.0)
    point_along_x = matrix @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(point_along_x, [0.0, -1.0, 0.0], atol=1e-10)


def test_relion_euler_to_matrix_pure_tilt_rotates_x_toward_plus_z() -> None:
    matrix = relion_euler_to_matrix(0.0, 90.0, 0.0)
    point_along_x = matrix @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(point_along_x, [0.0, 0.0, 1.0], atol=1e-10)


def _marker_volume(size: int, marker_xyz: tuple[int, int, int]) -> torch.Tensor:
    volume = torch.zeros((size, size, size), dtype=torch.float32)
    x, y, z = marker_xyz
    volume[z, y, x] = 1.0
    return volume


def test_resample_volume_identity_is_a_no_op() -> None:
    volume = _marker_volume(9, marker_xyz=(6, 4, 5))
    resampled = resample_volume_zyx(
        volume,
        np.eye(3),
        np.zeros(3),
        device=torch.device("cpu"),
    )
    np.testing.assert_allclose(resampled.numpy(), volume.numpy(), atol=1e-5)


def test_resample_volume_pure_shift_moves_marker() -> None:
    size = 11
    center = size // 2
    volume = _marker_volume(size, marker_xyz=(center, center, center))
    resampled = resample_volume_zyx(
        volume,
        np.eye(3),
        np.array([2.0, 0.0, 0.0]),
        device=torch.device("cpu"),
    )
    peak = np.unravel_index(np.argmax(resampled.numpy()), resampled.shape)
    assert peak == (center, center, center - 2)


def test_resample_volume_matches_relion_rot_rotation_direction() -> None:
    # size=11, center=5: marker sits at x=3 (i.e. x_rel=-2 from center), y=z=center.
    # A(90, 0, 0) maps ref +X to lab -Y (verified above), so its inverse A.T maps
    # a lab-frame marker at x_rel=-2 to a ref-frame (output) position at y_rel=-2.
    size = 11
    center = size // 2
    volume = _marker_volume(size, marker_xyz=(center - 2, center, center))

    matrix = relion_euler_to_matrix(90.0, 0.0, 0.0)
    resampled = resample_volume_zyx(
        volume,
        matrix,
        np.zeros(3),
        device=torch.device("cpu"),
    )
    peak = np.unravel_index(np.argmax(resampled.numpy()), resampled.shape)
    assert peak == (center, center - 2, center)


def _smooth_blob_volume(size: int) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    z_grid, y_grid, x_grid = torch.meshgrid(axis, axis, axis, indexing="ij")
    squared_radius = x_grid**2 + y_grid**2 + z_grid**2
    return torch.exp(-squared_radius / (2.0 * (size / 4.0) ** 2))


def test_resample_volume_forward_then_inverse_round_trips() -> None:
    # A random per-voxel signal is a poor round-trip fixture: trilinear
    # interpolation is a lowpass filter, so applying it twice measurably blurs
    # white noise even when the geometric transform is exact. Use a smooth
    # blob instead, matching real (smooth) density data.
    size = 13
    volume = _smooth_blob_volume(size)
    matrix = relion_euler_to_matrix(23.0, 61.0, -104.0)
    shift = np.array([1.3, -0.6, 0.4])
    device = torch.device("cpu")

    forward = resample_volume_zyx(volume, matrix, shift, device=device)
    inverse_matrix = matrix.T
    inverse_shift = -(inverse_matrix @ shift)
    round_tripped = resample_volume_zyx(forward, inverse_matrix, inverse_shift, device=device)

    core = slice(3, size - 3)
    np.testing.assert_allclose(
        round_tripped.numpy()[core, core, core],
        volume.numpy()[core, core, core],
        atol=0.08,
    )


@pytest.mark.parametrize(
    "rot,tilt,psi",
    [
        (0.0, 0.0, 0.0),
        (37.0, 12.0, -84.0),
        (180.0, 90.0, 45.0),
        (-63.0, 155.0, 170.0),
        (12.5, 179.4, -3.0),
    ],
)
def test_matrix_to_relion_euler_round_trips_through_the_matrix(
    rot: float, tilt: float, psi: float
) -> None:
    matrix = relion_euler_to_matrix(rot, tilt, psi)
    recovered = relion_euler_to_matrix(*matrix_to_relion_euler(matrix))
    # Euler triples are not unique, so compare the matrices they produce.
    np.testing.assert_allclose(recovered, matrix, atol=1e-9)


def test_matrix_to_relion_euler_folds_gimbal_lock_into_psi() -> None:
    matrix = relion_euler_to_matrix(40.0, 0.0, 15.0)
    recovered_rot, recovered_tilt, recovered_psi = matrix_to_relion_euler(matrix)

    assert recovered_rot == pytest.approx(0.0)
    assert recovered_tilt == pytest.approx(0.0)
    # At tilt=0 only rot+psi is defined, so it all lands in psi.
    assert recovered_psi == pytest.approx(55.0)


def test_matrix_to_relion_euler_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected a 3x3 rotation matrix"):
        matrix_to_relion_euler(np.eye(4))


def test_cone_offsets_cover_the_cap_without_bunching_at_the_pole() -> None:
    offsets = cone_offsets(60.0, 15.0)

    polars = sorted({polar for polar, _ in offsets})
    assert polars == [0.0, 15.0, 30.0]
    # One sample at the pole, then rings that grow with their circumference.
    counts = {polar: sum(1 for p, _ in offsets if p == polar) for polar in polars}
    assert counts[0.0] == 1
    assert counts[15.0] < counts[30.0]
    assert all(0.0 <= azimuth < 360.0 for _, azimuth in offsets)


def test_cone_offsets_degenerate_to_a_single_sample() -> None:
    assert cone_offsets(0.0, 10.0) == [(0.0, 0.0)]
    # A range finer than one step still yields only the pole.
    assert cone_offsets(5.0, 10.0) == [(0.0, 0.0)]


def test_cone_offsets_reject_bad_sampling() -> None:
    with pytest.raises(ValueError, match="cone_sampling_deg"):
        cone_offsets(30.0, 0.0)
    with pytest.raises(ValueError, match="cone_range_deg"):
        cone_offsets(-1.0, 10.0)


@pytest.mark.parametrize("polar", [0.0, 7.5, 15.0, 45.0])
@pytest.mark.parametrize("azimuth", [0.0, 90.0, 213.0])
@pytest.mark.parametrize("in_plane", [0.0, -30.0, 120.0])
def test_perturbation_polar_deviation_equals_the_requested_angle(
    polar: float, azimuth: float, in_plane: float
) -> None:
    matrix = perturbation_matrix(polar, azimuth, in_plane)
    tilted_axis = matrix @ np.array([0.0, 0.0, 1.0])
    deviation = np.degrees(np.arccos(np.clip(tilted_axis[2], -1.0, 1.0)))

    # This is the property the cone search relies on: the in-plane angle and the
    # azimuth must not change how far the axis has moved.
    assert deviation == pytest.approx(polar, abs=1e-9)


def test_cone_search_composes_onto_an_existing_orientation() -> None:
    current = relion_euler_to_matrix(31.0, 64.0, -12.0)
    candidate = current @ perturbation_matrix(10.0, 45.0, 0.0)

    current_axis = current @ np.array([0.0, 0.0, 1.0])
    candidate_axis = candidate @ np.array([0.0, 0.0, 1.0])
    separation = np.degrees(
        np.arccos(np.clip(float(np.dot(current_axis, candidate_axis)), -1.0, 1.0))
    )

    assert separation == pytest.approx(10.0, abs=1e-9)


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 180.0, 213.0, 359.0])
@pytest.mark.parametrize("in_plane", [0.0, -10.0, 15.0])
def test_zero_polar_offset_has_no_azimuth_spin(azimuth: float, in_plane: float) -> None:
    # At the cone pole the azimuth is undefined, not an extra in-plane rotation.
    np.testing.assert_allclose(
        perturbation_matrix(0.0, azimuth, in_plane),
        relion_euler_to_matrix(in_plane, 0.0, 0.0),
        atol=1e-12,
    )


@pytest.mark.parametrize("polar", [0.001, 10.0, 45.0])
@pytest.mark.parametrize("azimuth", [0.0, 90.0, 154.28571428571428, 213.0])
@pytest.mark.parametrize("in_plane", [0.0, -10.0, 15.0])
def test_cone_perturbation_is_minimal_tilt_then_in_plane_spin(
    polar: float, azimuth: float, in_plane: float
) -> None:
    # Independent axis-angle construction of the shortest tilt. The signs follow
    # RELION's passive rotation convention: azimuth 0 tilts +Z towards -X.
    azimuth_rad = np.radians(azimuth)
    axis = np.array([-np.sin(azimuth_rad), -np.cos(azimuth_rad), 0.0])
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = np.radians(polar)
    tilt = (
        np.cos(angle) * np.eye(3)
        + (1 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    expected = tilt @ relion_euler_to_matrix(in_plane, 0.0, 0.0)
    actual = perturbation_matrix(polar, azimuth, in_plane)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
    assert np.linalg.det(actual) == pytest.approx(1.0, abs=1e-12)


def test_czii_cone_grid_bounds_the_full_rotation_not_only_axis_tilt() -> None:
    from cryoet_pipeline.backends.subtomogram import (
        _symmetric_offsets,
        crowther_angular_increment_deg,
    )

    # Reproduce the five-iteration CZII schedule, including its sampling floor.
    # The old mapping offered a 164.345887-degree move in the first local search.
    current = relion_euler_to_matrix(31.0, 64.0, -12.0)
    floor = crowther_angular_increment_deg(300.0, 35.0)
    for iteration in range(5):
        scale = 2**iteration
        sampling = max(10.0 / scale, floor)
        for polar, azimuth in cone_offsets(30.0 / scale, sampling):
            for in_plane in _symmetric_offsets(30.0 / scale, sampling):
                candidate = current @ perturbation_matrix(polar, azimuth, in_plane)
                relative = current.T @ candidate
                cosine = (np.trace(relative) - 1.0) / 2.0
                # Orthogonal tilt and spin axes give this azimuth-independent angle.
                expected_cosine = (
                    2 * (np.cos(np.radians(polar) / 2) * np.cos(np.radians(in_plane) / 2)) ** 2
                    - 1
                )
                assert cosine == pytest.approx(expected_cosine, abs=1e-12)
                full_angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                assert full_angle <= polar + abs(in_plane) + 2e-6
                assert full_angle <= 30.0 / scale + 2e-6
