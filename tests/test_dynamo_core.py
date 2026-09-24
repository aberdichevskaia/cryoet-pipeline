"""Small numerical contracts for the opt-in core, not scientific sweeps."""

import numpy as np
import pytest

from cryoet_pipeline import dynamo_core as core


def phantom(n=12):
    z, y, x = np.indices((n,) * 3) - n / 2
    return np.exp(-((x - 2) ** 2 + (y + 1) ** 2 + z * z) / 2) + 0.6 * np.exp(
        -((x + 1) ** 2 + (y - 2) ** 2 + (z - 1) ** 2) / 3
    )


def test_pose_convention_and_mex_boundary_support():
    data = np.arange(12**3, dtype=float).reshape((12,) * 3)
    out, valid = core.resample(data, [0, 0, 0], [1, -2, 0])
    # A positive x shift moves contents towards larger x, in ZYX storage.
    np.testing.assert_array_equal(out[:11, :9, 1:], data[:11, 2:11, :11])
    assert not valid[-1].any()
    assert not valid[:, :, 0].any()
    assert not valid[:, 9:].any()
    angles = [17, 39, -61]
    np.testing.assert_allclose(
        core.euler_matrix(-np.array(angles)[::-1]), core.euler_matrix(angles).T, atol=1e-14
    )
    np.testing.assert_allclose(
        core.euler_matrix(core.matrix_euler(core.euler_matrix(angles))),
        core.euler_matrix(angles),
        atol=1e-14,
    )


def test_soft_mask_is_support_and_all_eight_corners_are_required():
    volume = phantom()
    mask = np.ones_like(volume) * 0.001
    mask[5, 5, 5] = 0
    soft, support = core.resample(volume, [0, 0, 0], mask=mask)
    hard, hard_support = core.resample(volume, [0, 0, 0], mask=mask != 0)
    np.testing.assert_array_equal(soft, hard)
    np.testing.assert_array_equal(support, hard_support)
    assert not support[4:6, 4:6, 4:6].any()
    assert support[3, 3, 3]


def test_local_ncc_matches_direct_moving_roi_and_shift_sign():
    p, t = phantom(), np.rot90(phantom(), axes=(1, 2))
    mask = np.zeros_like(t, dtype=bool)
    mask[3:9, 2:8, 4:10] = True
    cc = core.local_correlation(t, mask, np.fft.fftn(p), np.fft.fftn(p * p))
    for dx, dy, dz in [(0, 0, 0), (2, -1, 1), (-2, 1, -1)]:
        # Particle samples at template coordinate + displacement, with FFT wrap.
        moved = np.roll(p, (-dz, -dy, -dx), (0, 1, 2))
        expected = np.corrcoef(t[mask], moved[mask])[0, 1]
        assert cc[6 + dz, 6 + dy, 6 + dx] == pytest.approx(expected, abs=1e-12)


def test_grid_order_counts_and_legacy_pole_tolerance():
    old = np.array([80, 137.44, -174.38])
    angles = core.angle_list(old, 15, 5, 15, 5)
    assert angles.shape == (21, 3)
    np.testing.assert_allclose(angles[0], old, atol=1e-12)
    np.testing.assert_allclose(angles[:3, 2], [-174.38, -169.38, -179.38])
    assert core.angle_list(old, 10, 2.5, 10, 2.5).shape == (100, 3)
    assert core.matrix_euler(core.euler_matrix([10, 0.001, 20]))[1] == 0


def test_spline_reflection_jump_guard_and_unrestricted_neighbors():
    peak, offset = core._spline_peak(0.4, 0.8, 0.7)
    mirror_peak, mirror_offset = core._spline_peak(0.7, 0.8, 0.4)
    assert peak > 0.8 and 0 < offset < 1
    assert mirror_peak == pytest.approx(peak)
    assert mirror_offset == pytest.approx(-offset)
    assert core._spline_peak(0, 1, 0) == (1.0, 0.0)
    assert core._spline_peak(-1.5, 0.8, 0.7) == (0.8, 0.0)
    cc = np.full((12,) * 3, -0.5)
    cc[6, 6, 6:9] = [0.4, 0.8, 0.7]
    shift, value = core.restricted_peak(cc, [0, 0, 0], 1)
    # The third sample is outside the search sphere, but still enters the fit.
    assert shift[0] > 1
    assert value == pytest.approx(peak)


@pytest.mark.parametrize(
    "triplet,expected_peak,expected_offset",
    [
        pytest.param(
            [0.703947723, 0.710783482, 0.708505571],
            0.71172756049485464,
            0.292932524424224,
            id="czii-tag-53",
        ),
        pytest.param(
            [0.732130229, 0.738079, 0.736096263],
            0.73890043501006974,
            0.29290517464018029,
            id="czii-tag-170",
        ),
    ],
)
def test_measured_single_precision_spline_regression(triplet, expected_peak, expected_offset):
    # Measured Z neighbours from dynamo_peak_audit_53_170.json (2026-09-24).
    # Expected values are Dynamo's double-precision fit of the same rounded
    # single samples, NOT its historical full-search winners. In single,
    # tag 53 produced a spurious offset -1 and peak 0.75; tag 170 is nearby.
    samples = np.asarray(triplet, dtype=np.float32)
    expected = (expected_peak, expected_offset)
    assert core._spline_peak(*samples) == pytest.approx(expected, rel=0, abs=1e-10)

    # Promote AFTER quantizing: both calls must fit exactly the same samples.
    double_samples = samples.astype(np.float64)
    assert core._spline_peak(*double_samples) == pytest.approx(expected, rel=0, abs=1e-10)
    centered_peak, centered_offset = core._spline_peak(*(double_samples - double_samples[1]))
    assert (centered_peak + double_samples[1], centered_offset) == pytest.approx(
        expected, rel=0, abs=1e-10
    )

    # Exercise the public ZYX-grid -> XYZ-shift path with the measured Z line.
    cc = np.full((6, 6, 6), -1, dtype=np.float32)
    cc[2:5, 3, 3] = samples
    shift, peak = core.restricted_peak(cc, [0, 0, 0], 1)
    np.testing.assert_allclose(shift, [0, 0, expected_offset], rtol=0, atol=1e-10)
    assert peak == pytest.approx(expected_peak, rel=0, abs=1e-10)


def test_fourier_filter_radii_wedge_support_and_normalization():
    shell = core.bandpass(48, 2, 16)
    assert shell[24, 24, 24] == pytest.approx(np.exp(-1), abs=1e-7)
    assert shell[24, 24, 40] == 1
    assert shell[24, 24, 42] == pytest.approx(np.exp(-1), abs=1e-7)
    assert shell[24, 24, 43] == 0
    w = core.wedge(48, [-45.04, 44.96])
    assert w[24, 24, 24] == 1
    assert w[25, 24, 24] == 0
    assert w[25, 24, 25] == 0
    assert w[23, 24, 25] == 1
    assert w[0, 0, 0] == 0
    mask = np.zeros((12,) * 3)
    mask[3:9, 3:9, 3:9] = 0.001
    v = core.normalize_roi(phantom(), mask)
    assert v[mask != 0].mean() == pytest.approx(0, abs=1e-14)
    assert v[mask != 0].std(ddof=1) == pytest.approx(1)


def test_average_replays_raw_data_and_preserves_inputs():
    v = phantom()
    original = v.copy()
    mask = np.zeros_like(v)
    mask[3:9, 3:9, 3:9] = 1
    angles, shifts, tilts = [[0, 0, 0]] * 3, [[0, 0, 0]] * 3, [[-60, 60]] * 3
    average, raw, weights = core.average_particles(
        [v, v * 2 + 3, v * 4 - 2], angles, shifts, tilts, mask
    )
    np.testing.assert_array_equal(v, original)
    np.testing.assert_allclose(
        raw[3:9, 3:9, 3:9], core.normalize_roi(v, mask)[3:9, 3:9, 3:9], atol=1e-6
    )
    assert np.isfinite(average).all() and np.isfinite(weights).all()
    assert weights.max() == 1
    expected_fft = np.fft.fftshift(np.fft.fftn(raw))
    expected_fft[weights < 0.1] = 0
    np.divide(expected_fft, weights, out=expected_fft, where=weights >= 0.1)
    np.testing.assert_allclose(average, np.fft.ifftn(np.fft.ifftshift(expected_fft)).real)


@pytest.mark.parametrize(
    "bad", [np.zeros((11,) * 3), np.zeros((12, 12, 10)), np.full((12,) * 3, np.nan)]
)
def test_invalid_volumes_are_refused(bad):
    with pytest.raises(ValueError):
        core.resample(bad, [0, 0, 0])


@pytest.mark.parametrize(
    "kwargs", [{"shift_limit": 0}, {"refine": 1.5}, {"low": float("nan")}, {"refine_factor": 1}]
)
def test_unsupported_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        core.SearchSettings(**kwargs)
