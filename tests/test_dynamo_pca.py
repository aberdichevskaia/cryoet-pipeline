"""Dynamo PCA classification port: rotated wedge, CC matrix, PCA, k-means.

Behaviour only. Numerical parity with Dynamo is the replay's job, on a real
Dynamo ccmatrix project (benchmark DYNAMO_CLEANUP.md, arm P).
"""

import numpy as np
import pytest
from test_wedge import _granular_phantom

from cryoet_pipeline import dynamo_pca as port
from cryoet_pipeline.backends import dynamo_search as core

BOX = 32
TILTS = (-57.0, 60.0)
HIGH, LOW = 2, 8


def phantom(seed=7):
    return _granular_phantom(BOX, seed=seed).numpy().astype(float)


def posed(volume, angles, shift=(0.0, 0.0, 0.0)):
    """At a table pose: the rotated volume, then shifted, so `aligned` undoes it."""
    m = core.euler_matrix(angles)
    moved, _ = core.resample(volume, angles, m @ np.asarray(shift, dtype=float))
    return moved


def particle(volume, angles, rng, noise=0.3):
    """At the pose, with the table wedge of the particle's own frame and noise."""
    wedged = core.apply_filter(posed(volume, angles), core.wedge(BOX, TILTS))
    return wedged + rng.normal(scale=noise * wedged.std(), size=wedged.shape)


def correlation(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def test_rotated_wedge_is_the_kernel_wedge_unrotated_and_follows_a_grid_rotation():
    flat = core.wedge(BOX, TILTS)
    np.testing.assert_array_equal(port.rotated_wedge(BOX, TILTS, [0, 0, 0]), flat)
    # tdrot 90 makes X' = -y and Z' = z: the wedge's x axis becomes -y.
    turned = port.rotated_wedge(BOX, TILTS, [90, 0, 0])
    expected = np.zeros_like(flat)
    for iy in range(1, BOX):
        expected[:, iy, :] = flat[:, BOX // 2, BOX - iy][:, None]
    x, y, z = core._xyz(BOX)
    inside = x * x + y * y + z * z < (BOX / 2 - 1) ** 2
    np.testing.assert_array_equal(turned[inside], expected[inside])


def test_rotated_wedge_has_no_slab_and_keeps_only_the_origin_on_an_exact_axis():
    # tilt 90 maps the tilt axis onto z, where X' == 0 exactly but Z' is a
    # rounding residue: the MEX drops that line except the origin.
    w = port.rotated_wedge(BOX, TILTS, [0, 90, 0])
    c = BOX // 2
    assert w[c, c, c] == 1
    assert not w[c + 1 :, c, c].any() and not w[:c, c, c].any()
    # Off the axis, |Z'| < 0.5 alone samples nothing: k = (-3, 4, -4) under
    # (30, 60, -45) has X' = -0.107, Z' = -0.299, a ratio far outside the range.
    w = port.rotated_wedge(48, (-45.04, 44.96), [30, 60, -45])
    assert w[24 - 4, 24 + 4, 24 - 3] == 0


def test_rotated_wedge_is_centrosymmetric_as_the_half_spectra_need():
    w = port.rotated_wedge(BOX, TILTS, [37.0, 71.0, -122.0])
    interior = w[1:, 1:, 1:]
    np.testing.assert_array_equal(interior, interior[::-1, ::-1, ::-1])
    assert 0.3 < w[w > 0].size / (4 / 3 * np.pi * (BOX / 2) ** 3) < 0.9


def test_aligned_undoes_the_pose_and_fills_undefined_voxels_as_asked():
    v, angles, shift = phantom(), np.array([30.0, 50.0, -70.0]), np.array([1.0, -2.0, 0.5])
    moved = posed(v, angles, shift)
    back = port.aligned(moved, angles, shift, HIGH, LOW)
    band = core.apply_filter(v, core.bandpass(BOX, HIGH, LOW))
    x, y, z = core._xyz(BOX)
    inside = x * x + y * y + z * z < 10**2
    assert correlation(back[inside], band[inside]) > 0.95
    # The two fillings differ by exactly the band-filtered fill of undefined voxels.
    raw, valid = core.resample(moved, -angles[::-1], -shift, sphere=True)
    assert not valid[0, 0, 0] and valid[BOX // 2, BOX // 2, 1]  # a corner, inside the sphere
    fill = np.where(valid, 0.0, raw[valid].min())
    difference = port.aligned(moved, angles, shift, HIGH, LOW, undefined="min") - back
    np.testing.assert_allclose(
        difference, core.apply_filter(fill, core.bandpass(BOX, HIGH, LOW)), atol=1e-12
    )


def test_cc_matrix_is_symmetric_with_unit_diagonal_and_finds_the_copy():
    rng = np.random.default_rng(1)
    v = phantom()
    poses = [np.array([10.0, 40.0, 80.0]), np.array([-60.0, 120.0, 15.0]), np.zeros(3)]
    particles = [particle(v, a, rng) for a in poses[:2]] + [rng.normal(size=(BOX,) * 3)]
    volumes = [
        port.aligned(p, a, np.zeros(3), HIGH, LOW) for p, a in zip(particles, poses, strict=True)
    ]
    wedges = [port.rotated_wedge(BOX, TILTS, -a[::-1]) for a in poses]
    cc = port.cc_matrix(volumes, wedges, core.search_mask(BOX, 24.0))
    assert np.array_equal(cc, cc.T) and np.all(np.diag(cc) == 1)
    assert cc[0, 1] > 0.7
    assert abs(cc[0, 2]) < 0.3 and abs(cc[1, 2]) < 0.3


def test_the_corner_sphere_is_one_inside_radius_n2_minus_1_and_decays_past_it():
    sphere = port.corner_sphere(BOX)
    x, y, z = core._xyz(BOX).astype(float)
    r = np.sqrt(x * x + y * y + z * z)
    edge = BOX / 2 - 1
    assert (sphere[r <= edge] == 1).all() and (sphere[r > edge] < 1).all()
    assert (sphere[r > edge + np.sqrt(-np.log(0.13))] == 0).all()
    soft = sphere[(sphere > 0) & (sphere < 1)]
    assert soft.size and soft.min() > 0.13
    np.testing.assert_array_equal(sphere[1:, 1:, 1:], sphere[:0:-1, :0:-1, :0:-1])


def test_the_cc_matrix_ignores_frequencies_beyond_the_corner_sphere():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(BOX,) * 3)
    x, y, z = core._xyz(BOX).astype(float)
    corners = np.fft.ifftshift(np.sqrt(x * x + y * y + z * z) > BOX / 2 + 0.5)
    noise = np.fft.ifftn(np.fft.fftn(rng.normal(size=base.shape)) * corners).real
    ones = [np.ones(base.shape)] * 2
    cc = port.cc_matrix([base, base + 5 * noise], ones, np.ones(base.shape))
    np.testing.assert_allclose(cc[0, 1], 1.0, atol=1e-12)
    without = port.cc_matrix([base, base + 5 * noise], ones, np.ones(base.shape), sphere=False)
    assert without[0, 1] < 0.9


def test_the_cc_matrix_binarizes_the_samplings_and_the_coefficients_do_not():
    rng = np.random.default_rng(2)
    v, mask = phantom(), core.search_mask(BOX, 24.0)
    poses = [np.array([20.0, 30.0, 40.0]), np.array([-50.0, 100.0, 5.0])]
    volumes = [port.aligned(particle(v, a, rng), a, np.zeros(3), HIGH, LOW) for a in poses]
    sharp = [port.rotated_wedge(BOX, TILTS, -a[::-1]) for a in poses]
    x, y, z = core._xyz(BOX)
    soft = [w * (0.5 + 0.5 * np.cos(np.sqrt(x * x + y * y + z * z) / BOX) ** 8) for w in sharp]
    cc = port.cc_matrix(volumes, sharp, mask)
    np.testing.assert_array_equal(port.cc_matrix(volumes, soft, mask), cc)
    eigen = port.pca(cc, volumes, mask, 2).eigenvolumes
    blurred = port.coefficients(eigen, volumes, soft, mask)
    assert np.abs(blurred - port.coefficients(eigen, volumes, sharp, mask)).max() > 1e-4


def test_project_mode_eigenvolumes_keep_the_mean_particle():
    """Near-copies make the first eigenvector nearly uniform: V' X is then their average."""
    rng = np.random.default_rng(5)
    v, mask = phantom(), core.search_mask(BOX, 24.0)
    poses = rng.uniform(-90, 90, (6, 3))
    volumes = [port.aligned(particle(v, a, rng), a, np.zeros(3), HIGH, LOW) for a in poses]
    wedges = [port.rotated_wedge(BOX, TILTS, -a[::-1]) for a in poses]
    found = port.pca(port.cc_matrix(volumes, wedges, mask), volumes, mask, 3)
    support = mask > 0
    average = np.mean(volumes, axis=0)
    assert abs(correlation(found.eigenvolumes[0][support], average[support])) > 0.99


def test_classify_separates_two_kinds_of_particle():
    rng = np.random.default_rng(3)
    kinds = [phantom(7), phantom(11)]
    truth = np.repeat([0, 1], 6)
    poses = np.column_stack(
        [rng.uniform(-180, 180, 12), rng.uniform(0, 180, 12), rng.uniform(-180, 180, 12)]
    )
    particles = [particle(kinds[t], a, rng) for t, a in zip(truth, poses, strict=True)]
    result = port.classify(
        particles,
        poses,
        np.zeros((12, 3)),
        [TILTS] * 12,
        core.search_mask(BOX, 24.0),
        high=HIGH,
        low=LOW,
        k=2,
        rng=np.random.default_rng(0),
        neigs=4,
        n_coefficients=2,
    )
    assert len(set(zip(result.labels, truth, strict=True))) == 2  # the same partition, up to naming
    assert np.all(np.diff(np.abs(result.pca.eigenvalues)) <= 0)
    vectors = result.pca.eigenvectors
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(4), atol=1e-10)
    assert result.coefficients.shape == (12, 4)


def test_lloyd_reaches_a_fixed_point_and_refills_an_empty_cluster():
    rng = np.random.default_rng(4)
    data = np.concatenate([rng.normal(0, 0.3, (20, 2)), rng.normal(5, 0.3, (20, 2))])
    labels, centroids = port.lloyd(data, [[1.0, 1.0], [4.0, 4.0]])
    assert set(labels[:20]) == {1} and set(labels[20:]) == {2}
    again, _ = port.lloyd(data, centroids)
    np.testing.assert_array_equal(again, labels)
    found, _ = port.kmeans(data, 2, np.random.default_rng(0))
    assert len(set(zip(found, labels, strict=True))) == 2
    # A centroid no point is nearest to takes the farthest point, as MATLAB does.
    labels, _ = port.lloyd(data, [[0.0, 0.0], [5.0, 5.0], [100.0, 100.0]], max_iterations=1)
    assert set(labels) == {1, 2, 3}


@pytest.mark.parametrize(
    "call",
    [
        lambda: port.rotated_wedge(BOX, (10, 60), [1, 2, 3]),
        lambda: port.aligned(np.zeros((BOX,) * 3), [0] * 3, [0] * 3, HIGH, LOW, undefined="mean"),
        lambda: port.cc_matrix(
            [np.random.default_rng(0).normal(size=(BOX - 1,) * 3)] * 2,
            [np.ones((BOX - 1,) * 3)] * 2,
            np.ones((BOX - 1,) * 3),
        ),
        lambda: port.pca(
            np.array([[1.0, 0.2], [0.3, 1.0]]), [np.ones((BOX,) * 3)] * 2, np.ones((BOX,) * 3)
        ),
        lambda: port.kmeans(np.zeros((3, 2)), 4, np.random.default_rng(0)),
    ],
)
def test_outside_the_ported_scope_is_refused(call):
    with pytest.raises(ValueError):
        call()
