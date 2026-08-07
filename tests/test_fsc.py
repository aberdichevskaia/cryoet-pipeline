from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
import zarr

from cryoet_pipeline.backends.subtomogram import ParticleRecord, write_particle_star
from cryoet_pipeline.fsc import (
    build_half_set_average,
    compute_fsc_curve,
    load_mrc_volume,
    plot_fsc_curve,
)


def _smooth_blob(size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    z_grid, y_grid, x_grid = torch.meshgrid(axis, axis, axis, indexing="ij")
    centers = torch.rand(3, generator=generator) * size / 4.0
    squared_radius = (x_grid - centers[0]) ** 2 + (y_grid - centers[1]) ** 2 + (
        z_grid - centers[2]
    ) ** 2
    return torch.exp(-squared_radius / (2.0 * (size / 5.0) ** 2))


def test_compute_fsc_curve_identical_volumes_stays_above_threshold() -> None:
    volume = _smooth_blob(16, seed=0)
    result = compute_fsc_curve(volume, volume, pixel_spacing_angstrom=2.0, threshold=0.99)

    assert result.crossing_resolution_angstrom is None
    assert all(value > 0.99 for value in result.correlation)
    assert len(result.resolution_angstrom) == len(result.correlation)
    assert result.resolution_angstrom[0] == pytest.approx(16 * 2.0 / 1)


def test_compute_fsc_curve_uncorrelated_noise_crosses_at_coarsest_shell() -> None:
    torch.manual_seed(0)
    a = torch.randn(16, 16, 16)
    b = torch.randn(16, 16, 16)
    result = compute_fsc_curve(a, b, pixel_spacing_angstrom=2.0, threshold=0.5)

    assert result.crossing_resolution_angstrom == pytest.approx(result.resolution_angstrom[0])


def test_compute_fsc_curve_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must share shape"):
        compute_fsc_curve(
            torch.zeros(8, 8, 8),
            torch.zeros(6, 6, 6),
            pixel_spacing_angstrom=2.0,
            threshold=0.5,
        )


def test_compute_fsc_curve_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match="expected 3D volumes"):
        compute_fsc_curve(
            torch.zeros(8, 8),
            torch.zeros(8, 8),
            pixel_spacing_angstrom=2.0,
            threshold=0.5,
        )


def test_build_half_set_average_filters_and_averages_by_subset(tmp_path: Path) -> None:
    box_size = 6
    volume_a = torch.full((box_size, box_size, box_size), 1.0, dtype=torch.float32)
    volume_b = torch.full((box_size, box_size, box_size), 5.0, dtype=torch.float32)

    stack_path = tmp_path / "particles.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(4, box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    stack[0] = volume_a.numpy()
    stack[1] = volume_a.numpy()
    stack[2] = volume_b.numpy()
    stack[3] = volume_b.numpy()

    metadata_path = tmp_path / "particles.star"
    write_particle_star(
        metadata_path,
        [
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
                random_subset=random_subset,
            )
            for index, random_subset in enumerate([1, 1, 2, 2])
        ],
    )

    device = torch.device("cpu")
    average_subset_1 = build_half_set_average(
        particle_stack_path=stack_path,
        metadata_path=metadata_path,
        random_subset=1,
        device=device,
    )
    average_subset_2 = build_half_set_average(
        particle_stack_path=stack_path,
        metadata_path=metadata_path,
        random_subset=2,
        device=device,
    )

    np.testing.assert_allclose(average_subset_1.numpy(), volume_a.numpy(), atol=1e-5)
    np.testing.assert_allclose(average_subset_2.numpy(), volume_b.numpy(), atol=1e-5)


def test_build_half_set_average_rejects_missing_subset(tmp_path: Path) -> None:
    box_size = 4
    stack_path = tmp_path / "particles.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(1, box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    stack[0] = np.zeros((box_size, box_size, box_size), dtype=np.float32)
    metadata_path = tmp_path / "particles.star"
    write_particle_star(
        metadata_path,
        [
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
                class_number=1,
                random_subset=1,
            )
        ],
    )

    with pytest.raises(ValueError, match="no particles with random_subset=2"):
        build_half_set_average(
            particle_stack_path=stack_path,
            metadata_path=metadata_path,
            random_subset=2,
            device=torch.device("cpu"),
        )


def test_load_mrc_volume_round_trips_shape_and_voxel_size(tmp_path: Path) -> None:
    path = tmp_path / "volume.mrc"
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = 3.21

    volume, voxel_size = load_mrc_volume(path)

    np.testing.assert_allclose(volume.numpy(), data)
    assert voxel_size == pytest.approx(3.21, abs=1e-4)


def test_plot_fsc_curve_writes_a_png(tmp_path: Path) -> None:
    volume = _smooth_blob(16, seed=0)
    curve = compute_fsc_curve(volume, volume, pixel_spacing_angstrom=2.0, threshold=0.5)
    out = tmp_path / "fsc.png"

    plot_fsc_curve(
        curve.resolution_angstrom,
        curve.correlation,
        threshold=curve.threshold,
        crossing_resolution_angstrom=curve.crossing_resolution_angstrom,
        out=out,
        title="test",
    )

    assert out.is_file()
    assert out.stat().st_size > 0
