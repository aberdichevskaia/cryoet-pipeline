"""Dynamo MRA port: reference preparation, class averaging, assignment, iterations.

Behaviour only. Numerical parity with Dynamo is the replay's job, on a real
Dynamo MRA project (benchmark DYNAMO_MRA.md).
"""

import numpy as np
import pytest
from test_wedge import _granular_phantom

from cryoet_pipeline import dynamo_mra as mra
from cryoet_pipeline.backends import dynamo_search as core

BOX = 24
FAST = core.SearchSettings(
    cone_aperture=10, cone_sampling=5, inplane_range=10, inplane_sampling=5, refine=1, low=8
)


def soft_sphere(box=BOX, radius=9.0, width=2.0):
    """A soft mask like pytom's: one inside, a smooth tail through 0.5 at `radius`."""
    r = np.sqrt(np.sum(core._xyz(box).astype(float) ** 2, axis=0))
    return np.clip(0.5 - (r - radius) / (2 * width), 0, 1)


def particle(phantom, angles, rng, noise, tilts=(-60, 60)):
    """In the kernel's own frame: what its search will rotate the reference to match."""
    rotated, _ = core.resample(phantom, angles)
    wedged = core.apply_filter(rotated, core.wedge(len(phantom), tilts))
    return wedged + rng.normal(scale=noise * wedged.std(), size=wedged.shape)


def random_angles(rng, count):
    return np.column_stack(
        [rng.uniform(-180, 180, count), rng.uniform(0, 180, count), rng.uniform(-180, 180, count)]
    )


def test_prepared_reference_is_banded_and_normalized_over_the_thresholded_mask():
    template = _granular_phantom(BOX).numpy().astype(float)
    mask = soft_sphere()
    prepared = mra.prepare_reference(template, mask, FAST)
    inside = mask > 0.5
    assert abs(prepared[inside].mean()) < 1e-12 and prepared[inside].std(ddof=1) == pytest.approx(1)
    # The soft tail is outside the region: over all nonzero voxels the mean is not zero.
    assert abs(prepared[mask != 0].mean()) > 1e-3
    np.testing.assert_allclose(
        mra.prepare_reference(3 * template + 5, mask, FAST), prepared, atol=1e-9
    )
    spectrum = np.abs(np.fft.fftshift(np.fft.fftn(prepared)))
    band = core.bandpass(BOX, FAST.high, FAST.low)
    band[BOX // 2, BOX // 2, BOX // 2] = 1  # the normalization moves the mean
    assert spectrum[band == 0].max() < 1e-9 * spectrum.max()


def test_fweight_divide_divides_what_reaches_the_threshold_and_drops_the_rest():
    k = np.fft.fftfreq(BOX)[:, None, None] * np.ones((BOX, BOX, BOX))
    z = np.arange(BOX)[:, None, None] * np.ones((BOX, BOX, BOX))
    volume = np.cos(2 * np.pi * 2 * z / BOX) + np.cos(2 * np.pi * 5 * z / BOX)
    weight = np.ones((BOX, BOX, BOX))
    shifted_k = np.fft.fftshift(np.abs(k) * BOX)
    weight[np.isclose(shifted_k, 5)] = 0.2
    np.testing.assert_allclose(
        mra.fweight_divide(volume, np.ones_like(volume), 0.5), volume, atol=1e-9
    )
    np.testing.assert_allclose(
        mra.fweight_divide(volume, weight, 0.5), np.cos(2 * np.pi * 2 * z / BOX), atol=1e-9
    )
    np.testing.assert_allclose(
        mra.fweight_divide(volume, weight, 0.1),
        np.cos(2 * np.pi * 2 * z / BOX) + 5 * np.cos(2 * np.pi * 5 * z / BOX),
        atol=1e-9,
    )


def test_class_average_threshold_normalization_and_compensation():
    rng = np.random.default_rng(5)
    phantom = _granular_phantom(BOX).numpy().astype(float)
    angles = random_angles(rng, 8)
    shifts = np.zeros((8, 3))
    particles = [particle(phantom, a, rng, 0.0, tilts=(-45, 45)) for a in angles]
    nmask = soft_sphere()
    tilts = [(-45, 45)] * 8
    result = mra.average_class(particles, angles, shifts, tilts, nmask)
    assert result.members == 8 and result.threshold == 1 / 8
    assert result.fweight.max() == 1
    np.testing.assert_array_equal(result.fmask, result.fweight >= 1 / 8)
    inside = nmask != 0
    assert abs(result.average[inside].mean()) < 1e-12
    assert result.average[inside].std(ddof=1) == pytest.approx(1)
    spectrum = np.abs(np.fft.fftshift(np.fft.fftn(result.average)))
    dropped = result.fweight < result.threshold
    dropped[BOX // 2, BOX // 2, BOX // 2] = False
    # The real part of the inverse FFT, Dynamo's too, hands a dropped component
    # half its kept partner's conjugate; only pairs dropped on both sides are zero.
    partner = np.roll(np.flip(dropped), 1, axis=(0, 1, 2))
    assert (dropped & partner).any()
    assert spectrum[dropped & partner].max() < 1e-9 * spectrum.max()
    band = core.bandpass(BOX, 2, 8)

    def agreement(volume):
        a, b = core.apply_filter(volume, band)[inside], core.apply_filter(phantom, band)[inside]
        return np.corrcoef(a, b)[0, 1]

    # Dividing out the uneven wedge coverage brings the map closer to the truth.
    assert agreement(result.average) > agreement(result.raw) > 0.5
    assert mra.average_class(particles, angles, shifts, tilts, nmask, fmin=3).threshold == 3 / 8
    assert mra.average_class(particles, angles, shifts, tilts, nmask, fmin=9).threshold == 1 / 8
    single = mra.average_class(particles[:1], angles[:1], shifts[:1], tilts[:1], nmask)
    assert single.members == 1 and single.threshold == 1.0
    with pytest.raises(ValueError, match="positive integer"):
        mra.average_class(particles, angles, shifts, tilts, nmask, fmin=0)


def test_assignment_takes_the_highest_cc_and_the_lowest_channel_on_a_tie():
    scores = {
        3: np.array([0.2, 0.5, 0.4]),
        1: np.array([0.3, 0.5, 0.1]),
        2: np.array([0.1, 0.4, 0.4]),
    }
    np.testing.assert_array_equal(mra.assign(scores), [1, 1, 2])
    with pytest.raises(ValueError, match="finite"):
        mra.assign({1: np.array([np.nan])})
    with pytest.raises(ValueError, match="no live channel"):
        mra.assign({})


@pytest.fixture(scope="module")
def two_classes():
    """Six particles each of two different phantoms, a reference for each and a noise trap."""
    rng = np.random.default_rng(9)
    maps = {
        "A": _granular_phantom(BOX, seed=7).numpy().astype(float),
        "B": _granular_phantom(BOX, seed=31).numpy().astype(float),
    }
    truth = np.array(["A"] * 6 + ["B"] * 6)
    angles = random_angles(rng, 12)
    particles = [particle(maps[c], a, rng, 1.0) for c, a in zip(truth, angles, strict=True)]
    references = {1: maps["A"], 2: maps["B"], 3: rng.normal(size=(BOX,) * 3)}
    data = {
        "particles": particles,
        "angles": angles,
        "shifts": np.zeros((12, 3)),
        "tilts": [(-60, 60)] * 12,
        "mask": soft_sphere(),
        "truth": truth,
        "references": references,
    }
    data["first"] = mra.mra_iteration(
        particles,
        references,
        {k: (angles, data["shifts"]) for k in references},
        data["tilts"],
        data["mask"],
        FAST,
    )
    return data


def test_one_iteration_sends_each_particle_to_the_reference_it_was_made_from(two_classes):
    step = two_classes["first"]
    expected = np.where(two_classes["truth"] == "A", 1, 2)
    assert step.computed == [1, 2, 3]
    assert (step.assignment == expected).sum() >= 11, step.assignment
    assert 3 not in step.surviving  # the noise trap caught nothing
    assert step.surviving == sorted(step.averages)
    for k in step.surviving:
        assert step.averages[k].members == int((step.assignment == k).sum())
        assert step.searches[k].angles.shape == (12, 3)
    # Each channel searched every particle, members or not.
    assert all(np.isfinite(step.searches[k].correlations).all() for k in step.computed)


def test_a_dead_channel_is_not_searched_again_and_poses_carry_per_channel(two_classes):
    d = two_classes
    history = mra.run_mra(
        d["particles"],
        d["references"],
        d["angles"],
        d["shifts"],
        d["tilts"],
        d["mask"],
        FAST,
        iterations=2,
    )
    first, second = history
    np.testing.assert_array_equal(first.assignment, d["first"].assignment)
    assert second.computed == first.surviving and 3 not in second.searches


def test_parallel_iteration_is_the_serial_one(two_classes):
    d = two_classes
    parallel = mra.mra_iteration(
        d["particles"],
        d["references"],
        {k: (d["angles"], d["shifts"]) for k in d["references"]},
        d["tilts"],
        d["mask"],
        FAST,
        workers=2,
    )
    serial = d["first"]
    np.testing.assert_array_equal(parallel.assignment, serial.assignment)
    for k in serial.computed:
        for field in ("angles", "shifts", "correlations"):
            np.testing.assert_array_equal(
                getattr(parallel.searches[k], field), getattr(serial.searches[k], field)
            )
    for k in serial.surviving:
        np.testing.assert_array_equal(parallel.averages[k].average, serial.averages[k].average)
