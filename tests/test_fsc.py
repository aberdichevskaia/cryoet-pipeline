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
    load_mask,
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


def _sphere_mask(size: int, *, radius: float) -> torch.Tensor:
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    z_grid, y_grid, x_grid = torch.meshgrid(axis, axis, axis, indexing="ij")
    squared_radius = x_grid**2 + y_grid**2 + z_grid**2
    return (squared_radius <= radius**2).to(torch.float32)


def test_load_mask_rejects_shape_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "mask.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ones((6, 6, 6), dtype=np.float32))

    with pytest.raises(ValueError, match="does not match volume shape"):
        load_mask(path, expected_shape=(8, 8, 8))


def test_load_mask_round_trips(tmp_path: Path) -> None:
    mask_data = _sphere_mask(8, radius=3.0).numpy()
    path = tmp_path / "mask.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(mask_data)

    loaded = load_mask(path, expected_shape=(8, 8, 8))

    np.testing.assert_allclose(loaded.numpy(), mask_data)


def test_masking_removes_background_noise_from_fsc() -> None:
    size = 16
    torch.manual_seed(0)
    shared_particle = _smooth_blob(size, seed=0)
    mask = _sphere_mask(size, radius=4.0)

    noise_a = torch.randn(size, size, size)
    noise_b = torch.randn(size, size, size)
    volume_a = shared_particle * mask + noise_a * (1 - mask)
    volume_b = shared_particle * mask + noise_b * (1 - mask)

    unmasked = compute_fsc_curve(volume_a, volume_b, pixel_spacing_angstrom=2.0, threshold=0.5)
    masked = compute_fsc_curve(
        volume_a * mask, volume_b * mask, pixel_spacing_angstrom=2.0, threshold=0.5
    )

    assert min(masked.correlation) > min(unmasked.correlation)
    assert all(value > 0.9 for value in masked.correlation)


def test_cli_imports_without_the_optional_fsc_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A package only `compute-fsc` needs must not break unrelated commands.

    The CLI imports every module up front, so a module-scope import of an
    optional dependency makes `estimate-ctf` (and everything else) fail with a
    ModuleNotFoundError that has nothing to do with what was asked.
    """

    import importlib
    import sys

    # Blocking a name with None makes `import name` raise ImportError.
    monkeypatch.setitem(sys.modules, "torch_fourier_shell_correlation", None)
    for name in [n for n in list(sys.modules) if n.startswith("cryoet_pipeline")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module = importlib.import_module("cryoet_pipeline.cli")
    assert module.app is not None


def test_compute_fsc_curve_reports_the_missing_package_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "torch_fourier_shell_correlation", None)
    volume = _smooth_blob(8, seed=0)

    with pytest.raises(ImportError, match="torch-fourier-shell-correlation"):
        compute_fsc_curve(volume, volume, pixel_spacing_angstrom=2.0, threshold=0.5)
