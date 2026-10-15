"""The dynamo-search backend: its Dynamo kernel, pose conversion and the backend."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import mrcfile
import numpy as np
import pytest
import torch
import zarr
from test_wedge import _granular_phantom, _manifest, _particle_stack
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.subtomogram import average_and_register
from cryoet_pipeline.models import ArtifactKind
from cryoet_pipeline.particles import read_particle_star
from cryoet_pipeline.rotations import (
    center_particle_volume,
    matrix_to_relion_euler,
    perturbation_matrix,
    relion_euler_to_matrix,
    resample_volume_zyx,
)
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.wedge import (
    WEDGE_BOUNDARY_POLICY,
    WedgeGeometry,
    wedge_present_mask_rfft,
)


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
    "kwargs",
    [
        {"shift_limit": 0},
        {"refine": 1.5},
        {"low": float("nan")},
        {"refine_factor": 1},
        {"area_search_modus": 0},
        {"area_search_modus": 3},
        {"area_search_modus": 2.0},
    ],
)
def test_unsupported_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        core.SearchSettings(**kwargs)


def test_mode_one_keeps_the_shift_sphere_on_the_box_centre():
    # A centred particle recorded 10 voxels off. Mode 2 recentres its 3-voxel
    # sphere on the preceding winner, so two levels move it at most 2 x (3 + 1)
    # voxels and cannot come within 2 of the centre; mode 1 searches about the
    # box centre whatever the record says (Dynamo's limm 1).
    box, angles, tilts = 24, np.array([30.0, 60.0, -45.0]), (-60, 60)
    phantom = _granular_phantom(box).numpy().astype(float)
    rotated, _ = core.resample(phantom, angles)
    wedged = core.apply_filter(rotated, core.wedge(box, tilts))
    noise = np.random.default_rng(7).normal(scale=0.5 * wedged.std(), size=wedged.shape)
    found = {
        modus: core.refine_particle(
            wedged + noise,
            phantom,
            core.search_mask(box, 18.0),
            angles,
            [10, 0, 0],
            tilts,
            core.SearchSettings(
                cone_aperture=10,
                cone_sampling=5,
                inplane_range=10,
                inplane_sampling=5,
                refine=1,
                low=8,
                area_search_modus=modus,
            ),
        ).shift
        for modus in (1, 2)
    }
    assert np.abs(found[1]).max() < 0.5
    assert abs(found[2][0]) >= 2


# ---------------------------------------------------------------- the backend
# Everything below exercises the backend around the kernel: the pose
# conversion between the two centre conventions, and the loop that searches with
# the kernel and averages with this pipeline's compensated reducer.

_CPU = torch.device("cpu")


def _angle_deg(first, second):
    cosine = (float(np.trace(np.asarray(first) @ np.asarray(second).T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


@pytest.mark.parametrize("box", [12, 48])
def test_pose_conversion_keeps_the_pull_map_across_the_two_centres(box):
    generator = np.random.default_rng(5)
    eulers = [*generator.uniform(-180, 180, size=(6, 3)), (0, 0, 0), (17, 180, -61), (41, 0, 71)]
    points = np.array([[0, 0, 0], [box - 1] * 3, [3.1, 7.4, 1.9]], dtype=float)
    for euler in eulers:
        matrix = relion_euler_to_matrix(*euler)
        shift = generator.normal(size=3)
        angles, d = core.native_to_dynamo(matrix, shift, box)
        back, back_shift = core.dynamo_to_native(angles, d, box)
        np.testing.assert_allclose(back, matrix, atol=1e-10)
        np.testing.assert_allclose(back_shift, shift, atol=1e-10)
        # x_in = cN + A (x - cN) + s  ==  cD + M^-1 (x - cD) + d, with M^-1 = M.T
        native_centre, dynamo_centre = (box - 1) / 2, box / 2
        m_inverse = core.euler_matrix(angles).T
        np.testing.assert_allclose(
            native_centre + (points - native_centre) @ matrix.T + shift,
            dynamo_centre + (points - dynamo_centre) @ m_inverse.T + d,
            atol=1e-10,
        )


@pytest.mark.parametrize("tilt", [0.0, 1e-9, 1e-5, 0.5, 90.0, 179.99999, 180.0])
def test_exact_dynamo_angles_hold_at_and_near_the_poles(tilt):
    matrix = core.euler_matrix([33.0, tilt, -71.0])
    np.testing.assert_allclose(
        core.euler_matrix(core.dynamo_angles_from_matrix(matrix)), matrix, atol=1e-9
    )
    with pytest.raises(ValueError, match="proper rotation"):
        core.dynamo_angles_from_matrix(matrix * np.array([1, 1, -1]))


def test_search_mask_and_grid_reach():
    mask = core.search_mask(16, 10.0)
    assert mask[8, 8, 8] == 1 and mask[0, 0, 0] == 0
    # A sphere about the particle's centre (N - 1) / 2, where native poses put
    # it, not the kernel's N / 2: symmetric under the whole reversal i -> N - 1 - i.
    np.testing.assert_array_equal(mask, mask[::-1, ::-1, ::-1])
    assert abs(mask.sum() - 4 / 3 * math.pi * 5**3) / mask.sum() < 0.1
    # Dynamo's refine223 alignment mask, read from that project: 30976 voxels,
    # the same count this sphere has about 23.5 and not about 24 (31103).
    assert core.search_mask(48, 390.468 / 10.012).sum() == 30976
    with pytest.raises(ValueError, match="fit"):
        core.search_mask(16, 17.0)
    # refine223: 7.5 + 7.5, then 5 + 5, then 2.5 + 2.5 degrees.
    assert core.grid_reach_deg(core.DEFAULT_SETTINGS) == pytest.approx(30.0)
    one_level = core.SearchSettings(refine=0)
    assert core.grid_reach_deg(one_level) == pytest.approx(15.0)


def _wedged_particles(tmp_path, poses, box=24, *, truth=None, noise=1.0):
    """Wedge-limited particles of the granular phantom at `truth`, recorded at `poses`.

    Noise at the signal's own standard deviation keeps Dynamo's CC below one: on
    noise-free data its subpixel peak overshoots, (1, 1.05) is clamped to 1 as
    in Dynamo, and last-winner ties then walk the search along its candidate
    list, which measures the clamp rather than the search.
    """
    present = wedge_present_mask_rfft(box, WedgeGeometry(-60.0, 60.0), device=_CPU)
    phantom = _granular_phantom(box)
    generator = np.random.default_rng(3)
    volumes = []
    for pose in truth or poses:
        lab = resample_volume_zyx(
            phantom, relion_euler_to_matrix(*pose).T, np.zeros(3), device=_CPU
        )
        lab = torch.fft.irfftn(torch.fft.rfftn(lab) * present, s=(box,) * 3)
        scale = noise * float(lab.std())
        volumes.append(
            lab + torch.from_numpy(generator.normal(scale=scale, size=lab.shape)).float()
        )
    return _particle_stack(tmp_path, volumes, poses)


def _true_and_perturbed(count, offset_deg):
    generator = np.random.default_rng(11)
    truth = [tuple(float(v) for v in row) for row in generator.uniform(-180, 180, (count, 3))]
    perturbed = [
        tuple(
            matrix_to_relion_euler(
                relion_euler_to_matrix(*pose)
                @ perturbation_matrix(offset_deg, float(generator.uniform(0, 360)), 0.0)
            )
        )
        for pose in truth
    ]
    return truth, perturbed


def _errors_deg(truth, refined):
    return np.array(
        [
            _angle_deg(
                relion_euler_to_matrix(*pose),
                relion_euler_to_matrix(r.rot_deg, r.tilt_deg, r.psi_deg),
            )
            for pose, r in zip(truth, refined, strict=True)
        ]
    )


_FAST = {
    "cone_range_deg": 10.0,
    "cone_sampling_deg": 5.0,
    "inplane_range_deg": 10.0,
    "inplane_sampling_deg": 5.0,
    "refine_levels": 1,
    "search_lowpass_px": 11.0,
    "shift_range_px": 1.5,
    "particle_diameter_angstrom": 220.0,
}


def _run(tmp_path, stack, **parameters):
    registry = ArtifactRegistry.empty()
    registry.add(stack)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={**_FAST, "overwrite": True, **parameters},
    )
    return average_and_register(
        core.DynamoSearchRefinementBackend(), [stack], [_manifest()], context, registry
    )


def test_dynamo_search_brings_poses_twenty_degrees_off_back_to_the_truth(tmp_path):
    """Through the whole backend: stack, both pose conversions, kernel, STAR.

    With the phantom itself as the reference, so only the search is tested.
    Measured on this seed with the kernel alone: 20 deg off comes back to a
    median 3.8 deg (max 9.6). A transposed or mis-centred conversion does not
    come back at all. Box 24 at 10 A with a 220 A particle cannot resolve angles
    much below 5 deg; that floor, not the search, sets the thresholds.
    """
    truth, perturbed = _true_and_perturbed(8, 20.0)
    stack = _wedged_particles(tmp_path, perturbed, truth=truth)
    reference = tmp_path / "phantom.mrc"
    with mrcfile.new(reference) as handle:
        handle.set_data(_granular_phantom(24).numpy())
        handle.voxel_size = 10.0
    average, qc = _run(
        tmp_path,
        stack,
        initial_reference_path=str(reference),
        cone_range_deg=40.0,
        cone_sampling_deg=10.0,
        inplane_range_deg=40.0,
        inplane_sampling_deg=10.0,
        refine_levels=2,
    )
    before = _errors_deg(
        truth, [SimpleNamespace(rot_deg=p[0], tilt_deg=p[1], psi_deg=p[2]) for p in perturbed]
    )
    after = _errors_deg(truth, read_particle_star(Path(average.parameters["metadata_path"])))
    assert np.allclose(before, 20.0, atol=1e-6)
    assert np.median(after) < 8.0 and after.mean() < before.mean() / 2, after
    assert average.kind is ArtifactKind.SUBTOMOGRAM_AVERAGE and qc.kind is ArtifactKind.QC
    assert average.parameters["backend"] == "dynamo-search"
    assert average.parameters["wedge_boundary_policy"] == WEDGE_BOUNDARY_POLICY
    report = json.loads(qc.path.read_text())
    assert report["initial_reference"] == str(reference)
    assert report["wedge_tilt_ranges_deg"] == [[-60.0, 60.0]]


def test_dynamo_search_holds_true_poses_to_its_information_limit(tmp_path):
    """From the true poses, against its own Dynamo average, two iterations.

    Measured on this seed, 12 particles, with Dynamo's averaging: median 4.9 deg,
    max 11.0 deg after one iteration, 5.0 and 13.3 deg after two. With the
    previous reducer one iteration gave 4.9 and 10.8, and every found pose scored
    at least as high as the true one: the landscape is flat to about the Crowther
    limit of this box, 5-6 deg.
    """
    truth, _ = _true_and_perturbed(12, 0.0)
    stack = _wedged_particles(tmp_path, truth)
    average, qc = _run(tmp_path, stack, num_iterations=2)
    refined = read_particle_star(Path(average.parameters["metadata_path"]))
    errors = _errors_deg(truth, refined)
    shifts = np.array([[r.shift_x_px, r.shift_y_px, r.shift_z_px] for r in refined])
    assert np.median(errors) < 7.0, errors
    assert np.abs(shifts).max() < 1.5, shifts
    assert all(r.class_number == 1 for r in refined)
    report = json.loads(qc.path.read_text())
    assert report["backend"] == "dynamo-search" and report["iterations_run"] == 2
    assert report["gold_standard"] is False and len(report["iterations"]) == 2
    assert report["initial_reference"] == core.INITIAL_REFERENCE_DEFAULT
    assert report["grid_reach_deg"] == pytest.approx(
        core.grid_reach_deg(
            core.SearchSettings(
                cone_aperture=10, cone_sampling=5, inplane_range=10, inplane_sampling=5, refine=1
            )
        )
    )


def test_particles_searched_in_parallel_land_exactly_where_one_process_puts_them(
    tmp_path, monkeypatch
):
    """The same STAR bytes, map and QC from three processes as from one."""
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    truth, perturbed = _true_and_perturbed(6, 10.0)
    runs = {}
    for workers in (1, 3):
        folder = tmp_path / f"workers-{workers}"
        folder.mkdir()
        stack = _wedged_particles(folder, perturbed, truth=truth)
        average, qc = _run(folder, stack, workers=workers)
        runs[workers] = (
            Path(average.parameters["metadata_path"]).read_bytes(),
            mrcfile.read(average.path),
            json.loads(qc.path.read_text()),
        )
    (star, volume, report), (parallel_star, parallel_volume, parallel_report) = runs[1], runs[3]
    assert parallel_star == star
    np.testing.assert_array_equal(parallel_volume, volume)
    assert (report["workers"], parallel_report["workers"]) == (1, 3)
    assert parallel_report == {**report, "workers": 3}
    assert report["pose_change"]["rotation_deg_mean"] > 1.0  # the search did move them
    assert os.environ["OMP_NUM_THREADS"] == "7"  # the workers' setting did not leak


def test_the_backend_searches_the_reference_as_dynamo_prepares_it(tmp_path, monkeypatch):
    """Every iteration's reference, the given one and the rebuilt one, is prepared first."""
    truth, _ = _true_and_perturbed(4, 0.0)
    stack = _wedged_particles(tmp_path, truth)
    reference = tmp_path / "reference.mrc"
    with mrcfile.new(reference) as handle:
        handle.set_data(_granular_phantom(24).numpy())
    seen = []
    real = core.search_particles

    def spy(particles, fixed, mask, *args, **kwargs):
        seen.append((fixed.copy(), mask.copy(), kwargs))
        return real(particles, fixed, mask, *args, **kwargs)

    monkeypatch.setattr(core, "search_particles", spy)
    _run(tmp_path, stack, num_iterations=2, initial_reference_path=str(reference), workers=1)
    assert len(seen) == 2
    settings = core.SearchSettings(
        cone_aperture=10,
        cone_sampling=5,
        inplane_range=10,
        inplane_sampling=5,
        refine=1,
        low=11,
        shift_limit=1.5,
    )
    first, mask, _ = seen[0]
    expected = core.prepare_reference(
        _granular_phantom(24).numpy().astype(np.float64), mask, settings
    )
    np.testing.assert_allclose(first, expected, atol=1e-10)
    rebuilt = seen[1][0][mask > 0.5]
    assert abs(rebuilt.mean()) < 1e-10 and rebuilt.std(ddof=1) == pytest.approx(1)


def _loaded_particles(stack):
    """The particles as the backend loads them: centred in float32, then float64."""
    return [
        center_particle_volume(torch.as_tensor(v)).numpy().astype(np.float64)
        for v in zarr.open(stack.path, mode="r")[:]
    ]


def _dynamo_table(records, box):
    poses = [
        core.native_to_dynamo(
            relion_euler_to_matrix(r.rot_deg, r.tilt_deg, r.psi_deg),
            np.array([r.shift_x_px, r.shift_y_px, r.shift_z_px]),
            box,
        )
        for r in records
    ]
    return [angles for angles, _ in poses], [shift for _, shift in poses]


def _relative(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def _spy_searches(monkeypatch):
    seen = []
    real = core.search_particles

    def spy(particles, fixed, mask, *args, **kwargs):
        seen.append((fixed.copy(), mask.copy()))
        return real(particles, fixed, mask, *args, **kwargs)

    monkeypatch.setattr(core, "search_particles", spy)
    return seen


def test_the_backend_averages_as_dynamos_project_does(tmp_path, monkeypatch):
    """It starts from `dynamo_average` of the given poses and writes Dynamo's project
    average at the poses it found, both as replayed against Dynamo."""
    truth, perturbed = _true_and_perturbed(6, 10.0)
    stack = _wedged_particles(tmp_path, perturbed, truth=truth)
    seen = _spy_searches(monkeypatch)
    average, qc = _run(tmp_path, stack, workers=1)
    particles = _loaded_particles(stack)
    mask = core.search_mask(24, 22.0)
    tilts = [(-60.0, 60.0)] * len(particles)
    start = read_particle_star(Path(stack.parameters["metadata_path"]))
    angles, shifts = _dynamo_table(start, 24)
    expected_start = core.average_class(particles, angles, shifts, tilts, mask, fmin=1).average
    np.testing.assert_allclose(
        seen[0][0],
        core.prepare_reference(expected_start, mask, core.SearchSettings(low=11.0)),
        atol=1e-10,
    )
    found = read_particle_star(Path(average.parameters["metadata_path"]))
    angles, shifts = _dynamo_table(found, 24)
    expected = core.average_particles(particles, angles, shifts, tilts, mask)[0]
    assert _relative(mrcfile.read(average.path).astype(np.float64), expected) < 1e-5
    report = json.loads(qc.path.read_text())
    assert report["schema_version"] == 3
    assert report["gold_standard"] is False and report["half_map_fsc"] is None
    assert report["initial_reference"] == core.INITIAL_REFERENCE_DEFAULT
    assert report["fourier_weight_threshold"] == core.FOURIER_WEIGHT_THRESHOLD
    assert set(report["fourier_weight"]) == {"minimum", "median", "fraction_below_threshold"}
    assert report["alignment_mask_diameter_angstrom"] == 220.0
    assert report["alignment_mask_path"] is None
    assert report["search_mask_voxels"] == report["normalization_mask_voxels"] == mask.sum()
    assert report["shift_search_mode"] == 2


def test_the_shift_search_mode_reaches_the_search(tmp_path, monkeypatch):
    """Dynamo's limm 1, the mode the CZII Dynamo projects run in, gets to the kernel."""
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    modes = []
    real = core.search_particles

    def spy(particles, fixed, mask, starts, tilts, settings, **kwargs):
        modes.append(settings.area_search_modus)
        return real(particles, fixed, mask, starts, tilts, settings, **kwargs)

    monkeypatch.setattr(core, "search_particles", spy)
    _, qc = _run(tmp_path, stack, workers=1, shift_search_mode=1)
    assert modes == [1]
    assert json.loads(qc.path.read_text())["shift_search_mode"] == 1


def _soft_mask(box=24, inner=6.0, outer=9.0):
    """One inside `inner`, a cosine to zero at `outer`: above 0.5 inside 7.5."""
    axis = np.arange(box) - (box - 1) / 2
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    edge = np.clip((r - inner) / (outer - inner), 0.0, 1.0)
    return np.where(r < outer, 0.5 + 0.5 * np.cos(np.pi * edge), 0.0)


def test_a_mask_file_is_searched_above_half_and_normalises_on_its_support(tmp_path, monkeypatch):
    """One mask, two uses, as in Dynamo: thresholded for the search, whole for averaging."""
    truth, perturbed = _true_and_perturbed(6, 10.0)
    stack = _wedged_particles(tmp_path, perturbed, truth=truth)
    path = tmp_path / "mask.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(_soft_mask().astype(np.float32))
    soft = mrcfile.read(path).astype(np.float64)
    seen = _spy_searches(monkeypatch)
    average, qc = _run(tmp_path, stack, workers=1, alignment_mask_path=str(path))
    np.testing.assert_array_equal(seen[0][1], (soft > 0.5).astype(float))
    particles = _loaded_particles(stack)
    tilts = [(-60.0, 60.0)] * len(particles)
    found = read_particle_star(Path(average.parameters["metadata_path"]))
    angles, shifts = _dynamo_table(found, 24)
    written = mrcfile.read(average.path).astype(np.float64)
    on_support = core.average_particles(particles, angles, shifts, tilts, soft)[0]
    on_threshold = core.average_particles(particles, angles, shifts, tilts, soft > 0.5)[0]
    assert _relative(written, on_support) < 1e-5
    assert _relative(written, on_threshold) > 1e-3  # the support is what normalises
    report = json.loads(qc.path.read_text())
    assert report["alignment_mask_path"] == str(path.resolve())
    assert report["alignment_mask_diameter_angstrom"] is None
    assert report["search_mask_voxels"] == np.count_nonzero(soft > 0.5)
    assert report["normalization_mask_voxels"] == np.count_nonzero(soft)


@pytest.mark.parametrize(
    ("mask", "extra", "match"),
    [
        (np.ones((20, 20, 20)), {}, "the particles"),
        (-np.ones((24, 24, 24)), {}, "nonnegative"),
        (np.full((24, 24, 24), 0.4), {}, "above 0.5"),
        (np.ones((24, 24, 24)), {"alignment_mask_diameter_angstrom": 200.0}, "not both"),
    ],
)
def test_dynamo_search_refuses_a_mask_it_cannot_use(tmp_path, mask, extra, match):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    path = tmp_path / "mask.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(mask.astype(np.float32))
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, stack, alignment_mask_path=str(path), **extra)


def test_gold_standard_refines_each_half_alone_from_a_filtered_common_start(tmp_path, monkeypatch):
    """Both halves start from the same band-passed `dynamo_average`, each is then
    searched and rebuilt from its own rlnRandomSubset alone, and the map is every
    particle at the pose its own half gave it."""
    truth, perturbed = _true_and_perturbed(8, 10.0)
    stack = _wedged_particles(tmp_path, perturbed, truth=truth)
    seen = _spy_searches(monkeypatch)
    sizes = []
    real = core.search_particles

    def count(particles, *args, **kwargs):
        sizes.append(len(particles))
        return real(particles, *args, **kwargs)

    monkeypatch.setattr(core, "search_particles", count)
    average, qc = _run(
        tmp_path,
        stack,
        workers=1,
        num_iterations=2,
        gold_standard=True,
        gold_standard_start_lowpass_angstrom=40.0,
    )
    assert sizes == [4, 4, 4, 4]  # two iterations of half 1, then two of half 2
    particles = _loaded_particles(stack)
    mask = core.search_mask(24, 22.0)
    tilts = [(-60.0, 60.0)] * len(particles)
    start = read_particle_star(Path(stack.parameters["metadata_path"]))
    angles, shifts = _dynamo_table(start, 24)
    common = core.average_class(particles, angles, shifts, tilts, mask, fmin=1).average
    common = core.apply_filter(common, core.bandpass(24, 1.0, 24 * 10.0 / 40.0, 2.0))
    expected_first = core.prepare_reference(common, mask, core.SearchSettings(low=11.0))
    np.testing.assert_allclose(seen[0][0], expected_first, atol=1e-10)  # half 1, iteration 1
    np.testing.assert_allclose(seen[2][0], expected_first, atol=1e-10)  # half 2, iteration 1
    found = read_particle_star(Path(average.parameters["metadata_path"]))
    assert [r.half_set for r in found] == [r.half_set for r in start]
    angles, shifts = _dynamo_table(found, 24)
    combined = core.average_particles(particles, angles, shifts, tilts, mask)[0]
    assert _relative(mrcfile.read(average.path).astype(np.float64), combined) < 1e-5
    for half, path in average.parameters["half_map_paths"].items():
        members = [i for i, r in enumerate(found) if r.half_set == int(half)]
        expected = core.average_particles(
            [particles[i] for i in members],
            [angles[i] for i in members],
            [shifts[i] for i in members],
            [tilts[i] for i in members],
            mask,
        )[0]
        assert _relative(mrcfile.read(path).astype(np.float64), expected) < 1e-5
    report = json.loads(qc.path.read_text())
    assert report["gold_standard"] is True and average.parameters["gold_standard"] is True
    assert report["half_counts"] == {"1": 4, "2": 4}
    assert report["gold_standard_start_lowpass_angstrom"] == 40.0
    assert [s["half"] for s in report["iterations"]] == [1.0, 1.0, 2.0, 2.0]
    for label in ("masked", "unmasked"):
        assert len(report["half_map_fsc"][label]["correlation"]) == 12
    assert (
        average.parameters["resolution_angstrom"]
        == (report["half_map_fsc"]["masked"]["crossing_0_143_angstrom"])
    )
    assert any("fewer than 10" in w for w in report["warnings"])


def test_the_half_map_fsc_names_the_crossing_a_rise_toward_nyquist_hides():
    """Halves that share a band beyond shell 12: the final descent finds no 0.5
    crossing, and the report gives the one before the dip (shell 7, as test_fsc's
    construction measures it)."""
    from test_fsc import _two_maps

    a, b = _two_maps(shared_outer=True)
    report = core._half_map_fsc({1: a.numpy(), 2: b.numpy()}, np.ones((32, 32, 32)), 2.0)
    for label in ("masked", "unmasked"):
        curve = report[label]
        assert curve["crossing_0_5_angstrom"] is None
        rise = curve["rise_toward_nyquist"]
        assert rise["sigma"] >= 5
        assert rise["crossing_before_dip_0_5_angstrom"] == pytest.approx(64.0 / 7)


def test_gold_standard_halves_never_see_each_other(tmp_path):
    """From a given start, half 2's poses and map do not change by a bit when every
    particle of half 1 is replaced by noise."""
    truth, perturbed = _true_and_perturbed(8, 10.0)
    reference = tmp_path / "start.mrc"
    with mrcfile.new(reference) as handle:
        handle.set_data(_granular_phantom(24).numpy())
    outcomes = []
    for corrupt in (False, True):
        folder = tmp_path / f"corrupt-{corrupt}"
        folder.mkdir()
        stack = _wedged_particles(folder, perturbed, truth=truth)
        if corrupt:
            data = zarr.open(stack.path, mode="r+")
            noise = np.random.default_rng(9).normal(size=data.shape[1:]).astype(np.float32)
            for index in range(0, data.shape[0], 2):  # half_set 1
                data[index] = noise
        average, _ = _run(
            folder,
            stack,
            workers=1,
            gold_standard=True,
            initial_reference_path=str(reference),
        )
        found = read_particle_star(Path(average.parameters["metadata_path"]))
        outcomes.append(
            (
                [r for r in found if r.half_set == 2],
                [r for r in found if r.half_set == 1],
                mrcfile.read(average.parameters["half_map_paths"]["2"]),
            )
        )
    (half2, half1, map2), (half2_again, half1_again, map2_again) = outcomes
    assert half2_again == half2
    np.testing.assert_array_equal(map2_again, map2)
    assert half1_again != half1  # the corruption did reach half 1


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"gold_standard_start_lowpass_angstrom": 60.0}, "only with gold_standard"),
        ({"gold_standard": True, "gold_standard_start_lowpass_angstrom": 300.0}, "between 1"),
        ({"gold_standard": True, "gold_standard_start_lowpass_angstrom": 10.0}, "between 1"),
    ],
)
def test_gold_standard_refuses_what_it_cannot_run(tmp_path, parameters, match):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)] * 2)
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, stack, **parameters)


def test_gold_standard_needs_two_particles_in_each_half(tmp_path):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0), (5.0, 6.0, 7.0)])
    with pytest.raises(ValueError, match="at least two particles in each half"):
        _run(tmp_path, stack, gold_standard=True)


def test_worker_default_is_the_jobs_allocation(monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "3")
    assert core.available_cpus() == 3
    for value in ("", "0", "two"):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", value)
        assert core.available_cpus() >= 1
    monkeypatch.delenv("SLURM_CPUS_PER_TASK")
    assert core.available_cpus() >= 1
    with pytest.raises(ValueError, match="at least 1"):
        core.search_particles([], None, None, [], [], core.DEFAULT_SETTINGS, workers=0)


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"wedge_tilt_axis_deg": 30.0}, "tilt axis"),
        ({"num_classes": 2}, "num_classes"),
        ({"particle_metadata_path": "x.star"}, "particle_metadata_path"),
        ({"class_number": 1}, "class_number"),
        ({"wedge_min_tilt_deg": 5.0, "wedge_max_tilt_deg": 60.0}, "spanning zero"),
        ({"search_lowpass_px": 13.0}, "Nyquist"),
    ],
)
def test_dynamo_search_refuses_what_it_does_not_model(tmp_path, parameters, match):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, stack, **parameters)


def _run_average(tmp_path, stack, **parameters):
    registry = ArtifactRegistry.empty()
    registry.add(stack)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"particle_diameter_angstrom": 220.0, "overwrite": True, **parameters},
    )
    return average_and_register(
        core.DynamoAverageBackend(), [stack], [_manifest()], context, registry
    )


def _poses(records):
    return np.array(
        [[r.rot_deg, r.tilt_deg, r.psi_deg, r.shift_x_px, r.shift_y_px, r.shift_z_px]
         for r in records]
    )


def test_dynamo_average_is_dynamos_console_average_at_the_poses_given(tmp_path):
    """`dynamo_average(..., 'fcompensate', true, 'fmin', 1, 'nmask', mask)` of the
    stack's own poses, as replayed against Dynamo, and the poses go out unchanged."""
    truth, perturbed = _true_and_perturbed(6, 10.0)
    stack = _wedged_particles(tmp_path, perturbed, truth=truth)
    average, qc = _run_average(tmp_path, stack)
    particles = _loaded_particles(stack)
    mask = core.search_mask(24, 22.0)
    tilts = [(-60.0, 60.0)] * len(particles)
    start = read_particle_star(Path(stack.parameters["metadata_path"]))
    angles, shifts = _dynamo_table(start, 24)
    expected = core.average_class(particles, angles, shifts, tilts, mask, fmin=1)
    assert _relative(mrcfile.read(average.path).astype(np.float64), expected.average) < 1e-5
    written = read_particle_star(Path(average.parameters["metadata_path"]))
    np.testing.assert_allclose(_poses(written), _poses(start), atol=1e-5)
    assert all(r.class_number == 1 for r in written)
    assert average.kind is ArtifactKind.SUBTOMOGRAM_AVERAGE and qc.kind is ArtifactKind.QC
    assert average.parameters["backend"] == "dynamo-average"
    assert average.parameters["iterations_run"] == 0
    report = json.loads(qc.path.read_text())
    assert report["backend"] == "dynamo-average" and report["status"] == "pass"
    assert report["iterations_run"] == 0 and report["fmin"] == 1
    assert report["fourier_weight_threshold"] == pytest.approx(expected.threshold)
    assert report["fourier_weight_threshold"] == pytest.approx(1 / 6)
    assert report["alignment_mask_diameter_angstrom"] == 220.0
    assert report["normalization_mask_voxels"] == mask.sum()
    assert report["wedge_tilt_ranges_deg"] == [[-60.0, 60.0]]


def test_dynamo_average_is_the_phantom_only_at_the_right_poses(tmp_path):
    """The phantom, not the port, is the reference: this checks the poses end to end.

    Measured on these seeds, 12 particles: 0.985 within the mask at the true poses,
    0.614 with the poses scrambled. A transposed or mis-centred pose conversion
    lands with the scrambled one.
    """
    truth, _ = _true_and_perturbed(12, 0.0)
    scrambled = [
        tuple(float(v) for v in row)
        for row in np.random.default_rng(5).uniform(-180, 180, (12, 3))
    ]
    phantom = _granular_phantom(24).numpy().astype(np.float64)
    inside = core.search_mask(24, 22.0) > 0
    correlation = {}
    for label, poses in (("true", truth), ("scrambled", scrambled)):
        folder = tmp_path / label
        folder.mkdir()
        average, _ = _run_average(folder, _wedged_particles(folder, poses, truth=truth))
        written = mrcfile.read(average.path).astype(np.float64)
        correlation[label] = np.corrcoef(written[inside], phantom[inside])[0, 1]
    assert correlation["true"] > 0.97, correlation
    assert correlation["scrambled"] < 0.8, correlation


def test_dynamo_average_normalises_on_a_mask_files_support(tmp_path):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    path = tmp_path / "mask.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(_soft_mask().astype(np.float32))
    soft = mrcfile.read(path).astype(np.float64)
    average, qc = _run_average(tmp_path, stack, alignment_mask_path=str(path))
    particles = _loaded_particles(stack)
    start = read_particle_star(Path(stack.parameters["metadata_path"]))
    angles, shifts = _dynamo_table(start, 24)
    expected = core.average_class(particles, angles, shifts, [(-60.0, 60.0)] * 2, soft, fmin=1)
    assert _relative(mrcfile.read(average.path).astype(np.float64), expected.average) < 1e-5
    report = json.loads(qc.path.read_text())
    assert report["alignment_mask_path"] == str(path.resolve())
    assert report["alignment_mask_diameter_angstrom"] is None
    assert report["normalization_mask_voxels"] == np.count_nonzero(soft)


@pytest.mark.parametrize(
    ("parameters", "match"),
    [
        ({"num_iterations": 1}, "does not search"),
        ({"cone_range_deg": 10.0, "shift_range_px": 2.0}, "cone_range_deg, shift_range_px"),
        ({"initial_reference_path": "start.mrc"}, "does not search"),
        ({"gold_standard": True}, "does not search"),
        ({"num_classes": 2}, "num_classes"),
        ({"wedge_tilt_axis_deg": 30.0}, "tilt axis"),
        ({"wedge_min_tilt_deg": 5.0, "wedge_max_tilt_deg": 60.0}, "spanning zero"),
    ],
)
def test_dynamo_average_refuses_what_it_does_not_do(tmp_path, parameters, match):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    with pytest.raises(ValueError, match=match):
        _run_average(tmp_path, stack, **parameters)


def _cli(tmp_path, monkeypatch, *arguments):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    registry = ArtifactRegistry.empty()
    registry.add(stack)
    registry_path = tmp_path / "artifacts.json"
    registry.write(registry_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_manifest().model_dump_json())
    captured = {}

    def capture(backend, particles, manifests, context, registry, **kwargs):
        captured.update(backend=backend, parameters=dict(context.parameters))
        raise SystemExit(0)

    monkeypatch.setattr(cli, "average_and_register", capture)
    result = CliRunner().invoke(
        cli.app,
        [
            "average-particles",
            "--manifest",
            str(manifest),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            *arguments,
        ],
    )
    return result, captured


def test_cli_defaults_to_dynamo_search_and_leaves_grids_to_each_backend(tmp_path, monkeypatch):
    result, captured = _cli(tmp_path / "default", monkeypatch)
    assert isinstance(captured["backend"], core.DynamoSearchRefinementBackend), result.output
    for key in ("num_iterations", "cone_range_deg", "shift_range_px", "refine_levels", "workers"):
        assert key not in captured["parameters"]
    result, captured = _cli(tmp_path / "cone", monkeypatch, "--backend", "cone-search")
    assert "cone_range_deg" not in captured["parameters"], result.output
    result, captured = _cli(
        tmp_path / "given",
        monkeypatch,
        "--cone-range-deg",
        "12",
        "--refine-levels",
        "1",
        "--workers",
        "2",
    )
    assert captured["parameters"]["cone_range_deg"] == 12.0
    assert captured["parameters"]["refine_levels"] == 1
    assert captured["parameters"]["workers"] == 2
    assert "alignment_mask_path" not in captured["parameters"]
    result, captured = _cli(tmp_path / "mask", monkeypatch, "--alignment-mask", "mask.mrc")
    assert captured["parameters"]["alignment_mask_path"] == Path("mask.mrc"), result.output
    result, captured = _cli(tmp_path / "limm", monkeypatch, "--shift-search-mode", "1")
    assert captured["parameters"]["shift_search_mode"] == 1, result.output
    assert "gold_standard" not in captured["parameters"]
    result, captured = _cli(
        tmp_path / "gold",
        monkeypatch,
        "--gold-standard",
        "--gold-standard-start-lowpass-angstrom",
        "50",
    )
    assert captured["parameters"]["gold_standard"] is True, result.output
    assert captured["parameters"]["gold_standard_start_lowpass_angstrom"] == 50.0


@pytest.mark.parametrize(
    "arguments",
    [
        ("--missing-wedge",),
        ("--num-classes", "2"),
        ("--backend", "cone-search", "--refine-levels", "1"),
        ("--backend", "fft-refinement", "--search-lowpass-px", "8"),
        ("--backend", "cone-search", "--workers", "2"),
        ("--workers", "0"),
        ("--backend", "cone-search", "--alignment-mask", "mask.mrc"),
        ("--backend", "cone-search", "--shift-search-mode", "1"),
        ("--backend", "cone-search", "--gold-standard"),
        ("--backend", "fft-refinement", "--gold-standard"),
        ("--gold-standard-start-lowpass-angstrom", "60"),
        ("--shift-search-mode", "3"),
        ("--alignment-mask", "mask.mrc", "--alignment-mask-diameter-angstrom", "300"),
        ("--backend", "dynamo-average", "--num-iterations", "2"),
        ("--backend", "dynamo-average", "--cone-range-deg", "10"),
        ("--backend", "dynamo-average", "--shift-range-px", "1"),
        ("--backend", "dynamo-average", "--initial-reference", "start.mrc"),
        ("--backend", "dynamo-average", "--workers", "2"),
        ("--backend", "dynamo-average", "--gold-standard"),
        ("--backend", "dynamo-average", "--num-classes", "2"),
    ],
)
def test_cli_refuses_options_the_chosen_backend_would_not_read(tmp_path, monkeypatch, arguments):
    result, captured = _cli(tmp_path, monkeypatch, *arguments)
    assert result.exit_code != 0 and not captured, result.output


def test_cli_reaches_dynamo_average_with_its_mask_and_wedge(tmp_path, monkeypatch):
    result, captured = _cli(tmp_path / "plain", monkeypatch, "--backend", "dynamo-average")
    assert isinstance(captured["backend"], core.DynamoAverageBackend), result.output
    assert not any(captured["parameters"].get(key) for key in core.SEARCH_PARAMETERS)
    result, captured = _cli(
        tmp_path / "masked",
        monkeypatch,
        "--backend",
        "dynamo-average",
        "--alignment-mask",
        "mask.mrc",
        "--wedge-min-tilt-deg",
        "-50",
    )
    assert captured["parameters"]["alignment_mask_path"] == Path("mask.mrc"), result.output
    assert captured["parameters"]["wedge_min_tilt_deg"] == -50.0


def test_cli_says_nothing_about_resolution_for_an_average_nothing_refined(tmp_path):
    stack = _wedged_particles(tmp_path, [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)])
    registry = ArtifactRegistry.empty()
    registry.add(stack)
    registry_path = tmp_path / "artifacts.json"
    registry.write(registry_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_manifest().model_dump_json())
    result = CliRunner().invoke(
        cli.app,
        [
            "average-particles",
            "--manifest",
            str(manifest),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--backend",
            "dynamo-average",
            "--alignment-mask-diameter-angstrom",
            "220",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "particles averaged: 2" in result.output
    assert "resolution" not in result.output
    assert "_dynamo-average_average.mrc" in result.output
