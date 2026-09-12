from __future__ import annotations

import math

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


def matrix_to_relion_euler(matrix: np.ndarray) -> tuple[float, float, float]:
    """Recover `(rot, tilt, psi)` in degrees from a RELION rotation matrix.

    Inverse of `relion_euler_to_matrix`. At `tilt = 0` or `tilt = 180` only the
    sum (or difference) of `rot` and `psi` is defined; `rot` is then reported as
    zero and the whole in-plane rotation is folded into `psi`.
    """

    if matrix.shape != (3, 3):
        raise ValueError(f"expected a 3x3 rotation matrix, got shape {matrix.shape}")

    sin_tilt = math.hypot(float(matrix[2, 0]), float(matrix[2, 1]))
    cos_tilt = float(np.clip(matrix[2, 2], -1.0, 1.0))
    if sin_tilt < 1e-8:
        if cos_tilt >= 0.0:
            tilt = 0.0
            psi = math.atan2(float(matrix[0, 1]), float(matrix[0, 0]))
        else:
            tilt = math.pi
            psi = math.atan2(float(matrix[0, 1]), -float(matrix[0, 0]))
        return 0.0, math.degrees(tilt), math.degrees(psi)

    tilt = math.atan2(sin_tilt, cos_tilt)
    rot = math.atan2(float(matrix[2, 1]), float(matrix[2, 0]))
    psi = math.atan2(float(matrix[1, 2]), -float(matrix[0, 2]))
    return math.degrees(rot), math.degrees(tilt), math.degrees(psi)


def cone_offsets(
    cone_range_deg: float,
    cone_sampling_deg: float,
) -> list[tuple[float, float]]:
    """Sample orientations on a cone, as `(polar_deg, azimuth_deg)` pairs.

    Covers a spherical cap of half-angle `cone_range_deg / 2` at roughly even
    arc spacing: rings are spaced by `cone_sampling_deg`, and each ring carries
    as many azimuths as its circumference supports, so samples do not bunch up
    near the pole the way an independent sweep of Euler angles does.
    """

    if cone_range_deg < 0.0:
        raise ValueError("cone_range_deg must not be negative")
    if cone_sampling_deg <= 0.0:
        raise ValueError("cone_sampling_deg must be greater than zero")

    half_angle = cone_range_deg / 2.0
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    if half_angle <= 0.0:
        return offsets

    ring_count = int(math.floor(half_angle / cone_sampling_deg + 1e-9))
    for ring in range(1, ring_count + 1):
        polar = ring * cone_sampling_deg
        circumference_deg = 360.0 * math.sin(math.radians(polar))
        azimuth_count = max(1, int(math.ceil(circumference_deg / cone_sampling_deg)))
        for index in range(azimuth_count):
            offsets.append((polar, index * 360.0 / azimuth_count))
    return offsets


def perturbation_matrix(polar_deg: float, azimuth_deg: float, in_plane_deg: float) -> np.ndarray:
    """Build a minimal cone tilt plus the requested in-plane spin.

    With RELION's passive convention this is
    ``Rz(-azimuth) @ Ry(-polar) @ Rz(azimuth) @ Rz(-in_plane)``.
    Conjugating the tilt by the azimuth changes its direction without adding
    a spin: at zero polar offset only the requested in-plane rotation remains.

    Passing `in_plane` straight through as `rot` instead would leave an extra
    azimuth rotation. A local cone could then offer near-180-degree moves even
    though its z-axis tilt and explicit in-plane range were both small.
    """

    return relion_euler_to_matrix(in_plane_deg - azimuth_deg, polar_deg, azimuth_deg)


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


def center_particle_volume(volume: torch.Tensor) -> torch.Tensor:
    """Subtract a particle's own mean, so that zero padding means "nothing here".

    `resample_volume_zyx` pads with zeros, and a rotation moves the corners of
    the box outside it, so whatever the padding value is gets written into a
    corner-shaped region that depends on the angle. If the particle's mean is not
    zero, the padding is not neutral: it stamps a step of that size around the
    edge. Averaged over many orientations the steps combine into a smooth,
    spherically symmetric envelope which is nearly IDENTICAL in any two subsets
    of the particles, because it is a function of the box and the offset rather
    than of the specimen.

    That envelope is a correlated low-frequency signal, and an FSC between two
    half-sets sees it as agreement. Measured on this pipeline's own path: pure
    Gaussian noise, disjoint particles, independent random orientations, no
    shared reference at all, offset by the 0.064 sigma measured on real box-48
    crops of TS_01 at bin8 — the first shells correlate at 0.97 and the curve
    crosses 0.143 at 65 A with 20 particles per half, and at 52 A with 100.
    It gets WORSE with more particles, because the envelope adds coherently
    while the noise averages down.

    Centering each particle removes it completely: the same test then falls below
    the threshold in the first shell, which is the correct answer for noise.
    Centering the final map instead changes nothing at all — the envelope is
    already inside every term of the average by then. Hence per particle, before
    resampling, which is what this is for and where it has to be called.
    """

    return volume - volume.mean()
