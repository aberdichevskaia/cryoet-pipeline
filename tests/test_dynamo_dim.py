"""Dynamo's dim: a search at a smaller box, as the CPU project path makes one."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_dynamo_search_torch import BOX, SETTINGS, TILTS, _particle

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.backends import dynamo_search_torch as gpu
from cryoet_pipeline.dynamo_mra import mra_iteration

SMALL = 18  # below BOX (24) and not dividing it: Dynamo interpolates


def _cubic(points):
    return 0.3 + 0.2 * points - 0.01 * points**2 + 0.0004 * points**3


def _soft_sphere(box, radius):
    z, y, x = np.indices((box,) * 3) - box / 2
    return np.clip(radius - np.sqrt(x * x + y * y + z * z), 0, 1)


def test_the_resized_box_is_a_cubic_spline_through_dynamos_sample_points():
    # A not-a-knot cubic spline reproduces a cubic exactly, so a separable cubic
    # resized must be the cubic at 0, N/dim, 2N/dim, ... (0-based), MATLAB's
    # 1:N/dim:N, which puts the box's centre N/2 on the smaller box's dim/2.
    n, dim = 52, 30
    grid = np.arange(n, dtype=float)
    volume = (_cubic(grid)[:, None, None] * _cubic(n - 1 - grid)[None, :, None]
              * _cubic(0.5 * grid)[None, None, :])
    points = np.arange(dim) * (n / dim)
    expected = (_cubic(points)[:, None, None] * _cubic(n - 1 - points)[None, :, None]
                * _cubic(0.5 * points)[None, None, :])
    np.testing.assert_allclose(core.resize_box(volume, dim), expected, rtol=1e-10, atol=1e-12)
    assert points[dim // 2] == n / 2


@pytest.mark.parametrize("dim", [8, 7, 16, 18])
def test_a_dim_dynamo_would_bin_at_or_cannot_take_is_refused(dim):
    with pytest.raises(ValueError, match="dim"):
        core.resize_box(np.zeros((16,) * 3), dim)


def test_the_band_and_the_shift_limit_scale_with_the_box():
    small = core.downsized(core.SearchSettings(high=2, low=6, shift_limit=3, dim=12), 16)
    assert (small.high, small.low, small.shift_limit, small.dim) == (1.5, 4.5, 3.0, 0)
    # ceil(3 * 30 / 52) = ceil(1.73): RyR1's global rounds search 2 px at dim 30.
    assert core.downsized(core.SearchSettings(shift_limit=3, dim=30), 52).shift_limit == 2.0


def test_a_shift_found_at_dim_comes_back_as_dynamo_composes_it():
    # pipeline_align_one_particle: dynamo_rigid_compose(Tbin2real, Tbin, Treal2bin),
    # each 'rotshift', whose shifts compose as R2' * shift1 + shift2.
    box, dim = 52, 30
    b = 0.5 * (1 / (dim / box)) - 0.5

    def compose(first, second):
        return second[0], core.euler_matrix(second[0]).T @ first[1] + second[1]

    rng = np.random.default_rng(5)
    for _ in range(20):
        angles = rng.uniform([0, 0, 0], [360, 180, 360])
        found = rng.uniform(-3, 3, 3)
        binned = (angles, found * (1 / (dim / box)))
        _, expected = compose(compose((np.zeros(3), -b * np.ones(3)), binned),
                              (np.zeros(3), b * np.ones(3)))
        np.testing.assert_allclose(core.expanded_shift(angles, found, box, dim), expected,
                                   atol=1e-12)
    # Unrotated, the correction cancels: only the scale is left.
    np.testing.assert_allclose(core.expanded_shift([0, 0, 0], [1.0, -2.0, 0.5], box, dim),
                               np.array([1.0, -2.0, 0.5]) * box / dim, atol=1e-12)


def test_the_kernels_refuse_maps_of_another_box_than_the_settings_dim():
    with pytest.raises(ValueError, match="even"):
        core.SearchSettings(dim=31)
    particle, phantom = _particle(np.array([20.0, 50.0, 10.0]), [0.0, 0.0, 0.0])
    mask = core.search_mask(BOX, 18.0)
    settings = replace(SETTINGS, dim=SMALL)
    with pytest.raises(ValueError, match=f"dim {SMALL}"):
        core.refine_particle(particle, phantom, mask, [20, 50, 10], [0, 0, 0], TILTS, settings)
    with pytest.raises(ValueError, match=f"dim {SMALL}"):
        gpu.search_particles_torch([particle], phantom, mask, [(np.zeros(3), np.zeros(3))],
                                   [TILTS], settings, dtype=torch.float64)


def test_an_iteration_at_dim_searches_resized_maps_and_writes_full_box_shifts():
    seen = {}

    def search(particles, reference, mask, starts, tilt_ranges, settings, *, workers=1):
        seen.update(shapes={p.shape for p in particles}, reference=reference.shape,
                    mask=set(np.unique(mask).tolist()), settings=settings,
                    starts=[shift for _, shift in starts])
        return [(angles, shift, 0.5) for angles, shift in starts]

    rng = np.random.default_rng(3)
    particles = [rng.normal(size=(BOX,) * 3) for _ in range(3)]
    angles, shifts = rng.uniform(0, 90, (3, 3)), rng.uniform(-2, 2, (3, 3))
    settings = replace(SETTINGS, dim=SMALL)
    step = mra_iteration(particles, {1: rng.normal(size=(BOX,) * 3)}, {1: (angles, shifts)},
                         [TILTS] * 3, _soft_sphere(BOX, 9.0), settings, search=search)
    assert seen["shapes"] == {(SMALL,) * 3} and seen["reference"] == (SMALL,) * 3
    assert seen["mask"] == {0.0, 1.0}
    assert seen["settings"] == core.downsized(settings, BOX)
    np.testing.assert_allclose(seen["starts"], shifts * SMALL / BOX, atol=1e-12)
    expected = [core.expanded_shift(a, s * SMALL / BOX, BOX, SMALL)
                for a, s in zip(angles, shifts, strict=True)]
    np.testing.assert_allclose(step.searches[1].shifts, expected, atol=1e-12)


def test_the_devices_take_the_new_particles_when_the_box_changes():
    mask = core.search_mask(BOX, 18.0)
    particles, starts = [], []
    for seed in range(4):
        angles = np.array([20.0 + 30 * seed, 50.0, 10.0 * seed])
        particle, phantom = _particle(angles, [0.5, -0.5, 0.0], seed=seed)
        particles.append(particle)
        starts.append((angles + 2.0, np.zeros(3)))
    tilts = [TILTS] * len(particles)
    small = [core.resize_box(particle, SMALL) for particle in particles]
    small_inputs = (core.resize_box(phantom, SMALL), core.resize_mask(mask, SMALL), starts, tilts,
                    core.downsized(replace(SETTINGS, dim=SMALL), BOX))
    expected = gpu.search_particles_torch(small, *small_inputs, dtype=torch.float64)
    search = gpu.MultiDeviceSearch(["cpu", "cpu"], dtype="float64")
    try:
        search(particles, phantom, mask, starts, tilts, SETTINGS)
        found = search(small, *small_inputs)
    finally:
        search.close()
    for (angles, shift, cc), (e_angles, e_shift, e_cc) in zip(found, expected, strict=True):
        np.testing.assert_array_equal(angles, e_angles)
        np.testing.assert_array_equal(shift, e_shift)
        assert cc == e_cc
