from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
import zarr

from cryoet_pipeline.fsc import (
    build_half_set_average,
    compute_fsc_curve,
    load_mask,
    load_mrc_volume,
    plot_fsc_curve,
)
from cryoet_pipeline.particles import ParticleRecord, write_particle_star


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


def test_uncorrelated_noise_reports_no_resolution_rather_than_the_coarsest_shell() -> None:
    """This test previously asserted the bug, so it is worth saying what changed.

    It required uncorrelated noise to report the COARSEST shell as its
    resolution, on the reasoning that a curve below the threshold everywhere
    crosses immediately. But "469.63 A" is a number and reads as an answer,
    while the truth is that there is no resolution to report at all — and the
    same first-dip scan then misread real data, returning 469.63, 469.63, 234.82
    and 156.54 A for four class averages whose curves in fact peaked together
    near 80 A and died together near 32 A. Those were the positions of
    low-frequency wiggles, reported with `status: pass`.

    The invariant asserted instead cannot be satisfied by that bug: the coarsest
    shell is never a valid answer, because a resolution is the shell BEYOND which
    agreement stops, and something has to have been above the threshold first.
    """

    torch.manual_seed(0)
    a = torch.randn(16, 16, 16)
    b = torch.randn(16, 16, 16)
    result = compute_fsc_curve(a, b, pixel_spacing_angstrom=2.0, threshold=0.5)

    if result.reaches_threshold:
        # A handful of coefficients in the coarsest shells can correlate above
        # any threshold by chance, so this is not asserted away — but even then
        # the reported resolution must lie beyond the shell that was above it.
        assert result.crossing_resolution_angstrom != pytest.approx(
            result.resolution_angstrom[0]
        )
    else:
        assert result.crossing_resolution_angstrom is None


def test_a_curve_that_dips_at_low_frequency_and_recovers_says_so() -> None:
    """Two maps that disagree at coarse scale and agree in the middle.

    The real case is a compact template zero-padded into a larger particle box:
    the padding makes the overall envelopes differ, which lands in the coarsest
    shells, while the density itself still agrees at mid frequencies. Here a
    strong low-frequency ramp of opposite sign does the same job on top of a
    shared blob.

    What must not happen is the dip being reported as the resolution. What should
    happen is the final descent being reported, and the dip being flagged, since
    it says which end of the curve to distrust.
    """

    size = 16
    blob = _smooth_blob(size, seed=3)
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    ramp = axis.view(-1, 1, 1).expand(size, size, size) / axis.abs().max()

    a = blob + 3.0 * ramp
    b = blob - 3.0 * ramp
    result = compute_fsc_curve(a, b, pixel_spacing_angstrom=2.0, threshold=0.5)

    # Precondition, asserted so that a construction that stops producing the
    # intended shape fails here rather than silently passing the real check.
    assert result.correlation[0] < 0.5, result.correlation
    assert any(value >= 0.5 for value in result.correlation[1:]), result.correlation

    assert result.reaches_threshold
    assert result.dips_before_crossing
    assert result.crossing_resolution_angstrom != pytest.approx(
        result.resolution_angstrom[0]
    )


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
    """Two distinguishable half-sets, told apart by structure rather than by level.

    This used to use constant volumes of 1.0 and 5.0, which distinguished the
    halves purely by their mean — the one thing `center_particle_volume` removes,
    because a non-zero mean is what stamps a rotation-shaped envelope into every
    average. Zero-mean patterns test the same thing (that filtering by
    `random_subset` picks up the right particles) without resting on the offset.
    """

    box_size = 6
    half = box_size // 2
    volume_a = torch.full((box_size, box_size, box_size), -1.0, dtype=torch.float32)
    volume_a[:half] = 1.0
    volume_b = torch.full((box_size, box_size, box_size), -1.0, dtype=torch.float32)
    volume_b[:, :half] = 1.0
    assert volume_a.mean() == 0.0 and volume_b.mean() == 0.0
    assert not torch.allclose(volume_a, volume_b)

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


def _stack_and_star(tmp_path: Path, *, stack_count: int, record_count: int) -> tuple[Path, Path]:
    """A stack of one size and a STAR describing another — the mismatch under test."""

    box_size = 4
    stack_path = tmp_path / "particles.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(stack_count, box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    stack[:] = np.zeros((stack_count, box_size, box_size, box_size), dtype=np.float32)
    metadata_path = tmp_path / "refined.star"
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
                random_subset=1 + (index % 2),
            )
            for index in range(record_count)
        ],
    )
    return stack_path, metadata_path


def test_build_half_set_average_refuses_a_cross_series_star_against_one_stack(
    tmp_path: Path,
) -> None:
    """The refined STAR of a cross-tilt-series average indexes all of its stacks.

    `compute-fsc --mode half-set` takes exactly one stack, so those indices run
    past its end. Before the guard this raised zarr's `BoundsCheckError`, which
    named a dimension length and not the actual mistake.
    """

    stack_path, metadata_path = _stack_and_star(tmp_path, stack_count=4, record_count=8)

    with pytest.raises(ValueError, match="describes 8 particles but"):
        build_half_set_average(
            particle_stack_path=stack_path,
            metadata_path=metadata_path,
            random_subset=1,
            device=torch.device("cpu"),
        )


def test_build_half_set_average_refuses_a_stack_larger_than_its_star(
    tmp_path: Path,
) -> None:
    """The dangerous half of the same bug: every index resolves, silently.

    A stack holding more particles than the STAR describes — a re-extraction at a
    smaller box drops fewer particles at the edge, so more survive — leaves every
    combined index in range and pointing at the wrong particle. That path
    returned a map and a resolution number with nothing amiss on the surface,
    which is worse than a crash.
    """

    stack_path, metadata_path = _stack_and_star(tmp_path, stack_count=12, record_count=8)

    with pytest.raises(ValueError, match="cannot .*be paired with a single stack"):
        build_half_set_average(
            particle_stack_path=stack_path,
            metadata_path=metadata_path,
            random_subset=1,
            device=torch.device("cpu"),
        )


def test_build_half_set_average_accepts_a_matched_stack_and_star(tmp_path: Path) -> None:
    """The guard is a count comparison, so the legitimate pair must still pass."""

    stack_path, metadata_path = _stack_and_star(tmp_path, stack_count=8, record_count=8)

    average = build_half_set_average(
        particle_stack_path=stack_path,
        metadata_path=metadata_path,
        random_subset=1,
        device=torch.device("cpu"),
    )

    assert average.shape == (4, 4, 4)


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


def _noise_half_average(
    count: int, *, offset: float, centered: bool, rng: np.random.Generator
) -> torch.Tensor:
    """Average `count` pure-noise particles at independent random orientations."""

    from cryoet_pipeline.rotations import (
        center_particle_volume,
        relion_euler_to_matrix,
        resample_volume_zyx,
    )

    box_size = 32
    volumes = []
    for _ in range(count):
        raw = torch.as_tensor(
            (rng.normal(size=(box_size, box_size, box_size)) + offset).astype(np.float32)
        )
        if centered:
            raw = center_particle_volume(raw)
        matrix = relion_euler_to_matrix(*rng.uniform(0.0, 360.0, 3))
        volumes.append(
            resample_volume_zyx(raw, matrix, np.zeros(3), device=torch.device("cpu"))
        )
    return torch.stack(volumes).mean(dim=0)


def test_an_uncentered_particle_offset_fabricates_correlation_between_noise_halves() -> None:
    """The defect this guards against, stated as the measurement that found it.

    Zero padding under rotation writes the particle's own mean into an
    orientation-shaped region, and averaged over orientations that becomes a
    smooth envelope which is nearly identical in any two subsets. Two half-sets
    of PURE NOISE — disjoint particles, independent random orientations, no
    shared reference of any kind, so no amount of refinement independence could
    help — then correlate almost perfectly at low frequency.

    The offset used is the 0.064 sigma measured on real box-48 crops of TS_01 at
    bin8. Asserted as a floor rather than a point value so the test states the
    effect and not the RNG: without centering the first shell sits above 0.8 at
    this box size, and with it the median across seeds is indistinguishable from
    zero.
    """

    uncentered = [
        compute_fsc_curve(
            _noise_half_average(
                20, offset=0.064, centered=False, rng=np.random.default_rng(seed)
            ),
            _noise_half_average(
                20, offset=0.064, centered=False, rng=np.random.default_rng(seed + 500)
            ),
            pixel_spacing_angstrom=5.4,
            threshold=0.143,
        ).correlation[0]
        for seed in range(4)
    ]

    assert min(uncentered) > 0.8, f"expected the artefact, got {uncentered}"


def test_centering_each_particle_removes_the_fabricated_correlation() -> None:
    """The fix, measured the same way: noise must look like noise.

    Across seeds the first-shell correlation scatters around zero and is
    negative about half the time, which is what an FSC between two independent
    noise averages should do. Any consistently positive floor here would mean a
    second systematic path had been found, so the assertion is on the median
    plus the presence of negative draws rather than on a single run — a lone
    low-frequency shell holds few voxels and is genuinely noisy.
    """

    centered = [
        compute_fsc_curve(
            _noise_half_average(
                20, offset=0.064, centered=True, rng=np.random.default_rng(seed)
            ),
            _noise_half_average(
                20, offset=0.064, centered=True, rng=np.random.default_rng(seed + 500)
            ),
            pixel_spacing_angstrom=5.4,
            threshold=0.143,
        ).correlation[0]
        for seed in range(8)
    ]

    assert abs(float(np.median(centered))) < 0.25, f"still systematic: {centered}"
    assert any(value < 0.0 for value in centered), f"never negative: {centered}"


def test_centering_the_final_map_is_not_a_substitute_for_centering_each_particle() -> None:
    """Recorded because it is the fix one would reach for first, and it does nothing.

    By the time the average exists the envelope is inside every term of it, so
    removing the mean of the result leaves the curve where it was: the crossing
    is unchanged and the shells agree to floating-point noise. Subtracting a
    constant moves only the DC bin in exact arithmetic, and the DC bin is not
    even part of the returned curve.
    """

    average_a = _noise_half_average(
        20, offset=0.064, centered=False, rng=np.random.default_rng(11)
    )
    average_b = _noise_half_average(
        20, offset=0.064, centered=False, rng=np.random.default_rng(22)
    )

    before = compute_fsc_curve(
        average_a, average_b, pixel_spacing_angstrom=5.4, threshold=0.143
    )
    after = compute_fsc_curve(
        average_a - average_a.mean(),
        average_b - average_b.mean(),
        pixel_spacing_angstrom=5.4,
        threshold=0.143,
    )

    assert after.crossing_resolution_angstrom == before.crossing_resolution_angstrom
    np.testing.assert_allclose(after.correlation, before.correlation, atol=1e-6)
    assert before.correlation[0] > 0.8, "the artefact should still be there"
