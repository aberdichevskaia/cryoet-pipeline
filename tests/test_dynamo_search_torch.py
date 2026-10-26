"""The torch search kernel against the numpy one it ports, step by step and whole."""

import numpy as np
import pytest
import torch
from test_wedge import _granular_phantom

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.backends import dynamo_search_torch as gpu

BOX = 24
TILTS = (-55.0, 60.0)
SETTINGS = core.SearchSettings(cone_aperture=15, cone_sampling=5, inplane_range=15,
                               inplane_sampling=5, refine=1, low=8, area_search_modus=1)


def _particle(angles, shift, noise=0.5, seed=7):
    """The granular phantom at a pose, through the wedge, in noise; and the phantom."""

    phantom = _granular_phantom(BOX).numpy().astype(float)
    rotated, _ = core.resample(phantom, angles, shift)
    wedged = core.apply_filter(rotated, core.wedge(BOX, TILTS))
    rng = np.random.default_rng(seed)
    return wedged + rng.normal(scale=noise * wedged.std(), size=wedged.shape), phantom


def test_the_rotation_is_the_numpy_one_bit_for_bit():
    phantom = _granular_phantom(BOX).numpy().astype(float)
    mask = core.search_mask(BOX, 18.0)
    candidates = core.angle_list(np.array([30.0, 60.0, -45.0]), 30, 10, 30, 10)
    values, valid = gpu.resample_batch(
        torch.from_numpy(phantom.astype(np.float32)),
        torch.from_numpy(gpu._matrices(candidates)),
        torch.from_numpy(mask != 0),
    )
    for index, angles in enumerate(candidates):
        expected, expected_valid = core.resample(phantom, angles, mask=mask)
        np.testing.assert_array_equal(values[index].double().numpy(), expected)
        np.testing.assert_array_equal(valid[index].numpy(), expected_valid)


def test_the_spline_is_the_scalar_one_element_by_element():
    rng = np.random.default_rng(3)
    triples = rng.uniform(-1.2, 1.2, size=(4000, 3))
    # The guards: equal ends, a jump of 2 or more, and plain degenerate rows.
    triples[:50, 2] = triples[:50, 0]
    triples[50:100, 1] = triples[50:100, 0] + 2.5
    triples[100:150] = 0.3
    values, offsets = gpu.spline_peaks(triples[:, 0], triples[:, 1], triples[:, 2])
    for row, value, offset in zip(triples, values, offsets, strict=True):
        expected_value, expected_offset = core._spline_peak(*row)
        assert value == pytest.approx(expected_value, abs=1e-12)
        assert offset == pytest.approx(expected_offset, abs=1e-12)


def test_the_local_correlation_is_the_numpy_one():
    particle, phantom = _particle([20.0, 40.0, 10.0], [1.0, -0.5, 0.0])
    mask = core.search_mask(BOX, 18.0)
    candidates = core.angle_list(np.array([20.0, 40.0, 10.0]), 10, 5, 10, 5)
    fp = np.fft.fftn(particle)
    fp2 = np.fft.fftn(particle * particle)
    rotated, valid = gpu.resample_batch(
        torch.from_numpy(phantom.astype(np.float32)),
        torch.from_numpy(gpu._matrices(candidates)),
        torch.from_numpy(mask != 0),
    )
    cc = gpu.local_correlation_batch(rotated.double(), valid, torch.from_numpy(fp),
                                     torch.from_numpy(fp2)).numpy()
    for index in range(len(candidates)):
        expected = core.local_correlation(rotated[index].double().numpy(),
                                          valid[index].numpy(), fp, fp2)
        np.testing.assert_allclose(cc[index], expected, atol=1e-10)


@pytest.mark.parametrize("modus", [1, 2])
def test_in_float64_the_search_is_the_numpy_search(modus):
    truth = np.array([30.0, 60.0, -45.0])
    particle, phantom = _particle(truth, [1.2, -0.7, 0.4])
    mask = core.search_mask(BOX, 18.0)
    settings = core.SearchSettings(**{**vars(SETTINGS), "area_search_modus": modus})
    start = truth + np.array([4.0, -3.0, 5.0])
    expected = core.refine_particle(particle, phantom, mask, start, [0.5, 0, 0], TILTS, settings)
    for chunk in (1, 7, 64):
        found = gpu.refine_particle_torch(particle, phantom, mask, start, [0.5, 0, 0], TILTS,
                                          settings, dtype=torch.float64, chunk=chunk)
        np.testing.assert_allclose(found.angles, expected.angles, atol=1e-9)
        np.testing.assert_allclose(found.shift, expected.shift, atol=1e-9)
        assert found.correlation == pytest.approx(expected.correlation, abs=1e-9)
        assert [level["candidate_count"] for level in found.levels] == [
            level["candidate_count"] for level in expected.levels]


def test_in_float32_a_clear_search_lands_where_the_numpy_one_does():
    truth = np.array([30.0, 60.0, -45.0])
    particle, phantom = _particle(truth, [1.2, -0.7, 0.4], noise=0.3)
    mask = core.search_mask(BOX, 18.0)
    start = truth + np.array([4.0, -3.0, 5.0])
    expected = core.refine_particle(particle, phantom, mask, start, [0, 0, 0], TILTS, SETTINGS)
    found = gpu.refine_particle_torch(particle, phantom, mask, start, [0, 0, 0], TILTS, SETTINGS,
                                      dtype=torch.float32)
    np.testing.assert_allclose(found.angles, expected.angles, atol=1e-6)
    np.testing.assert_allclose(found.shift, expected.shift, atol=1e-3)
    assert found.correlation == pytest.approx(expected.correlation, abs=1e-4)


def test_several_particles_are_searched_as_one_by_one():
    mask = core.search_mask(BOX, 18.0)
    poses = [([30.0, 60.0, -45.0], [1.0, 0.0, 0.0]), ([100.0, 20.0, 10.0], [0.0, -1.0, 0.5])]
    particles, starts = [], []
    for seed, (angles, shift) in enumerate(poses):
        particle, phantom = _particle(angles, shift, seed=seed)
        particles.append(particle)
        starts.append((np.array(angles) + 3.0, np.zeros(3)))
    expected = core.search_particles(particles, phantom, mask, starts, [TILTS, TILTS], SETTINGS)
    found = gpu.search_particles_torch(particles, phantom, mask, starts, [TILTS, TILTS], SETTINGS,
                                       dtype=torch.float64)
    for (angles, shift, cc), (e_angles, e_shift, e_cc) in zip(found, expected, strict=True):
        np.testing.assert_allclose(angles, e_angles, atol=1e-9)
        np.testing.assert_allclose(shift, e_shift, atol=1e-9)
        assert cc == pytest.approx(e_cc, abs=1e-9)
