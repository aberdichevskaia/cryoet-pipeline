from __future__ import annotations

import numpy as np
import torch


def relion_euler_to_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """Build the RELION (rot, tilt, psi) rotation matrix.

    Matches RELION/XMIPP's `Euler_angles2matrix`: intrinsic Z-Y-Z, i.e. rotate by
    `rot` about Z, then by `tilt` about the new Y, then by `psi` about the newest
    Z. The returned matrix `A` maps a reference-frame coordinate to the
    corresponding lab-frame coordinate, `x_lab = A @ x_ref`, with coordinates
    ordered `(x, y, z)`. To bring a lab-frame (extracted, unrotated) particle
    into the reference frame for averaging, resample with `A` itself (output
    indexed by reference-frame coordinates, sampling the lab-frame volume); to
    predict the lab-frame appearance of a reference/template, resample with
    `A.T` (its inverse, since `A` is orthogonal).
    """

    rot, tilt, psi = np.radians([rot_deg, tilt_deg, psi_deg])
    cos_rot, sin_rot = np.cos(rot), np.sin(rot)
    cos_tilt, sin_tilt = np.cos(tilt), np.sin(tilt)
    cos_psi, sin_psi = np.cos(psi), np.sin(psi)
    cc = cos_tilt * cos_rot
    cs = cos_tilt * sin_rot
    sc = sin_tilt * cos_rot
    ss = sin_tilt * sin_rot
    row0 = [cos_psi * cc - sin_psi * sin_rot, cos_psi * cs + sin_psi * cos_rot, -cos_psi * sin_tilt]
    row1 = [
        -sin_psi * cc - cos_psi * sin_rot,
        -sin_psi * cs + cos_psi * cos_rot,
        sin_psi * sin_tilt,
    ]
    row2 = [sc, ss, cos_tilt]
    return np.array([row0, row1, row2], dtype=np.float64)


def resample_volume_zyx(
    volume: torch.Tensor,
    matrix_xyz: np.ndarray | torch.Tensor,
    shift_xyz: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Resample a single `(z, y, x)` volume under a rigid transform.

    For each output voxel at physical `(x, y, z)` offset from the box center,
    samples the input volume at `matrix_xyz @ (x, y, z) + shift_xyz` (also
    relative to center) via trilinear interpolation, zero-padded outside the
    input bounds. Output has the same shape as the input.

    `matrix_xyz`/`shift_xyz` operate on `(x, y, z)`-ordered physical
    coordinates, matching the convention of `relion_euler_to_matrix`; the
    `(z, y, x)` storage order used elsewhere in this pipeline is handled
    internally.
    """

    if volume.ndim != 3:
        raise ValueError(f"expected a 3D (z, y, x) volume, got shape {tuple(volume.shape)}")
    size_z, size_y, size_x = volume.shape
    matrix = torch.as_tensor(matrix_xyz, dtype=torch.float64, device=device)
    shift = torch.as_tensor(shift_xyz, dtype=torch.float64, device=device)
    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got shape {tuple(matrix.shape)}")
    if shift.shape != (3,):
        raise ValueError(f"expected a length-3 shift, got shape {tuple(shift.shape)}")

    center_x = (size_x - 1) / 2.0
    center_y = (size_y - 1) / 2.0
    center_z = (size_z - 1) / 2.0
    z_index, y_index, x_index = torch.meshgrid(
        torch.arange(size_z, dtype=torch.float64, device=device),
        torch.arange(size_y, dtype=torch.float64, device=device),
        torch.arange(size_x, dtype=torch.float64, device=device),
        indexing="ij",
    )
    output_xyz = torch.stack(
        [x_index - center_x, y_index - center_y, z_index - center_z], dim=-1
    )
    sample_xyz = output_xyz @ matrix.T + shift

    sample_x = sample_xyz[..., 0] + center_x
    sample_y = sample_xyz[..., 1] + center_y
    sample_z = sample_xyz[..., 2] + center_z

    def _normalize(coordinate: torch.Tensor, size: int) -> torch.Tensor:
        if size == 1:
            return torch.zeros_like(coordinate)
        return 2.0 * coordinate / (size - 1) - 1.0

    grid = torch.stack(
        [_normalize(sample_x, size_x), _normalize(sample_y, size_y), _normalize(sample_z, size_z)],
        dim=-1,
    ).to(torch.float32)

    input_tensor = volume.to(device=device, dtype=torch.float32)[None, None]
    sampled = torch.nn.functional.grid_sample(
        input_tensor,
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, 0]
