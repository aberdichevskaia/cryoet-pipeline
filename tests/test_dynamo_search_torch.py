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


def test_only_the_voxels_in_reach_are_computed_and_nothing_else_changes():
    phantom = _granular_phantom(BOX).numpy().astype(float)
    candidates = core.angle_list(np.array([30.0, 60.0, -45.0]), 360, 30, 360, 30)
    for mask in (core.search_mask(BOX, 14.0), core.search_mask(BOX, 22.0)):
        support = torch.from_numpy(mask != 0)
        reach = gpu.support_reach(support)
        assert len(reach) < BOX**3
        arguments = (torch.from_numpy(phantom.astype(np.float32)),
                     torch.from_numpy(gpu._matrices(candidates)), support)
        values, valid = gpu.resample_batch(*arguments)
        reached, reached_valid = gpu.resample_batch(*arguments, reach=reach)
        assert torch.equal(reached, values) and torch.equal(reached_valid, valid)


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
    volume = torch.from_numpy(particle)
    cc = gpu._centred(gpu.local_correlation_batch(
        rotated.double(), valid, torch.fft.rfftn(volume), torch.fft.rfftn(volume * volume))).numpy()
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


def test_the_half_filter_is_what_the_full_fft_keeps_even_on_the_nyquist_planes():
    # An asymmetric wedge with a bandpass reaching Nyquist: the full filter is not
    # point-symmetric there, and the real part of the full-FFT product is the product
    # with its Hermitian part.
    rng = np.random.default_rng(2)
    volume = rng.normal(size=(BOX,) * 3)
    centred = core.bandpass(BOX, 2, BOX / 2) * core.wedge(BOX, (-40.0, 65.0))
    expected = core.apply_filter(volume, centred)
    found = gpu._apply_half_filter(torch.from_numpy(volume),
                                   gpu.half_filter(torch.from_numpy(centred))).numpy()
    np.testing.assert_allclose(found, expected, atol=1e-12)


@pytest.mark.parametrize("shift", [(0.0, 0.0, 0.0), (1.6, -2.4, 0.3)])
def test_the_peak_is_read_from_the_uncentred_map_as_numpy_reads_the_centred_one(shift):
    rng = np.random.default_rng(5)
    raw = rng.uniform(-0.9, 0.9, size=(6, BOX, BOX, BOX))
    raw[3] = -1.0  # a sphere with nothing above -1: the full map's first -1 wins
    shifts, scores = gpu.restricted_peaks(torch.from_numpy(raw), np.array(shift), 3.0)
    for index in range(len(raw)):
        centred = gpu._centred(torch.from_numpy(raw[index])[None])[0].numpy()
        expected_shift, expected_score = core.restricted_peak(centred, np.array(shift), 3.0)
        np.testing.assert_allclose(shifts[index], expected_shift, atol=1e-12)
        assert scores[index] == pytest.approx(expected_score, abs=1e-12)


def test_particles_dealt_among_devices_are_searched_as_on_one():
    mask = core.search_mask(BOX, 18.0)
    particles, starts = [], []
    for seed in range(5):
        angles = np.array([20.0 + 30 * seed, 50.0, 10.0 * seed])
        particle, phantom = _particle(angles, [0.5, -0.5, 0.0], seed=seed)
        particles.append(particle)
        starts.append((angles + 2.0, np.zeros(3)))
    tilts = [TILTS] * len(particles)
    expected = gpu.search_particles_torch(particles, phantom, mask, starts, tilts, SETTINGS,
                                          dtype=torch.float64)
    search = gpu.MultiDeviceSearch(["cpu", "cpu"], dtype="float64")
    try:
        found = search(particles, phantom, mask, starts, tilts, SETTINGS)
        again = search(particles, phantom, mask, starts, tilts, SETTINGS)
    finally:
        search.close()
    for (angles, shift, cc), (e_angles, e_shift, e_cc) in zip(found, expected, strict=True):
        np.testing.assert_array_equal(angles, e_angles)
        np.testing.assert_array_equal(shift, e_shift)
        assert cc == e_cc
    assert [result[2] for result in again] == [result[2] for result in found]
