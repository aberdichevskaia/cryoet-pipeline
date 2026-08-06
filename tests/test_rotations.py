from __future__ import annotations

import numpy as np
import pytest
import torch

from cryoet_pipeline.rotations import relion_euler_to_matrix, resample_volume_zyx


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
