from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.subtomogram import (
    ConeSearchRefinementBackend,
    _best_shift_and_correlation,
    average_and_register,
)
from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import ParticleRecord, read_particle_star, write_particle_star
from cryoet_pipeline.rotations import (
    center_particle_volume,
    cone_offsets,
    perturbation_matrix,
    relion_euler_to_matrix,
    resample_volume_zyx,
)
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.wedge import (
    WedgeGeometry,
    constrained_shift_and_correlation,
    hermitian_weights_rfft,
    wedge_present_mask_rfft,
)

_DEVICE = torch.device("cpu")
_APIX = 10.0


def _frequency_grids(size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `(kx, ky, kz)` of every `rfftn` bin, in the repo's physical order."""

    frequency = torch.fft.fftfreq(size, dtype=torch.float64)
    frequency_x = torch.fft.rfftfreq(size, dtype=torch.float64)
    grid_z, grid_y, grid_x = torch.meshgrid(frequency, frequency, frequency_x, indexing="ij")
    return grid_x, grid_y, grid_z


def _granular_phantom(size: int, *, seed: int = 7) -> torch.Tensor:
    """Two unequal lobes packed with small blobs, band-limited at two voxels.

    Small smooth blobs would not do: a nearly spherical phantom has almost no
    orientation for a correlation to find, so a search over it measures the
    phantom's symmetry rather than the score. Band-limited so that nothing in it
    lives above Nyquist and the wedge is the only thing removing information.
    """

    generator = np.random.default_rng(seed)
    axis = (np.arange(size) - (size - 1) / 2) * _APIX
    grid_z, grid_y, grid_x = np.meshgrid(axis, axis, axis, indexing="ij")
    volume = np.zeros((size, size, size))
    lobes = [((0, 0, 0), (60, 50, 54), 1.0), ((24, 20, -16), (40, 34, 36), 0.85)]
    for _ in range(90):
        (center_z, center_y, center_x), (radius_z, radius_y, radius_x), amplitude = (
            lobes[0] if generator.random() < 0.62 else lobes[1]
        )
        direction = generator.normal(size=3)
        direction /= np.linalg.norm(direction)
        depth = generator.random() ** (1 / 3)
        sigma = generator.uniform(8.0, 13.0)
        weight = amplitude * generator.uniform(0.5, 1.5)
        volume += weight * np.exp(
            -(
                (grid_z - center_z - direction[0] * radius_z * depth) ** 2
                + (grid_y - center_y - direction[1] * radius_y * depth) ** 2
                + (grid_x - center_x - direction[2] * radius_x * depth) ** 2
            )
            / (2 * sigma * sigma)
        )
    radius = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    volume *= np.clip((110.0 - radius) / 18.0, 0.0, 1.0)
    return low_pass_filter_volume(
        torch.as_tensor(volume - volume.mean(), dtype=torch.float32),
        cutoff_angstrom=2 * _APIX,
        pixel_spacing_angstrom=_APIX,
    )


def _rotation_angle_deg(matrix: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(matrix) - 1.0) / 2.0))))


def _manifest() -> TiltSeriesManifest:
    """Three views spanning -60 to 60, which is where the wedge range comes from."""

    return TiltSeriesManifest(
        tilt_series_id="TS_WEDGE",
        source_mdoc=Path("TS_WEDGE.mdoc"),
        raw_pixel_spacing_angstrom=_APIX,
        images=[
            TiltImage(
                z_value=index,
                tilt_angle_deg=angle,
                subframe_path=f"frame{index}.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=_APIX,
                binning=1,
            )
            for index, angle in enumerate([-60.0, 0.0, 60.0])
        ],
    )


def _particle_stack(
    tmp_path: Path,
    volumes: list[torch.Tensor],
    angles: list[tuple[float, float, float]],
) -> Artifact:
    box_size = int(volumes[0].shape[0])
    stack_path = tmp_path / "wedge_particles.zarr"
    stack = zarr.open(
        stack_path,
        mode="w",
        shape=(len(volumes), box_size, box_size, box_size),
        chunks=(1, box_size, box_size, box_size),
        dtype=np.float32,
    )
    for index, volume in enumerate(volumes):
        stack[index] = volume.numpy()
    metadata_path = tmp_path / "wedge_particles.star"
    write_particle_star(
        metadata_path,
        [
            ParticleRecord(
                index=index,
                x_px=0.0,
                y_px=0.0,
                z_px=0.0,
                rot_deg=pose[0],
                tilt_deg=pose[1],
                psi_deg=pose[2],
                shift_x_px=0.0,
                shift_y_px=0.0,
                shift_z_px=0.0,
                score=0.5,
                class_number=1,
                half_set=1 + (index % 2),
            )
            for index, pose in enumerate(angles)
        ],
    )
    return Artifact(
        id="TS_WEDGE:particle_stack:cone-search",
        kind=ArtifactKind.PARTICLE_STACK,
        path=stack_path,
        shape=(len(volumes), box_size, box_size, box_size),
        axis_order=AxisOrder.NZYX,
        pixel_spacing_angstrom=_APIX,
        parameters={"tilt_series_id": "TS_WEDGE", "metadata_path": str(metadata_path)},
    )


def _refined_pose_errors(
    tmp_path: Path,
    *,
    box_size: int,
    poses: list[tuple[float, float, float]],
    wedge_the_data: bool,
    missing_wedge: bool,
) -> np.ndarray:
    """Run the real backend from the TRUE poses and report how far each moved."""

    present = wedge_present_mask_rfft(box_size, WedgeGeometry(-60.0, 60.0), device=_DEVICE)
    phantom = _granular_phantom(box_size)
    volumes = []
    for pose in poses:
        lab = resample_volume_zyx(
            phantom, relion_euler_to_matrix(*pose).T, np.zeros(3), device=_DEVICE
        )
        volumes.append(
            torch.fft.irfftn(torch.fft.rfftn(lab) * present, s=(box_size,) * 3)
            if wedge_the_data
            else lab
        )
    particles = _particle_stack(tmp_path, volumes, poses)
    registry = ArtifactRegistry.empty()
    registry.add(particles)
    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "num_iterations": 2,
            "cone_range_deg": 20.0,
            "cone_sampling_deg": 10.0,
            "inplane_range_deg": 20.0,
            "inplane_sampling_deg": 10.0,
            "shift_range_px": 1.0,
            "particle_diameter_angstrom": 220.0,
            "target_resolution_angstrom": 2 * _APIX,
            "missing_wedge": missing_wedge,
            "overwrite": True,
        },
    )
    artifacts = average_and_register(
        ConeSearchRefinementBackend(),
        [particles],
        [_manifest()],
        context,
        registry,
        replace_existing=True,
    )
    refined = read_particle_star(Path(artifacts[0].parameters["metadata_path"]))
    return np.array(
        [
            _rotation_angle_deg(
                relion_euler_to_matrix(*pose).T
                @ relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg)
            )
            for pose, record in zip(poses, refined, strict=True)
        ]
    )


def test_a_symmetric_range_is_the_analytic_wedge_about_the_beam_axis() -> None:
    """The wedge surrounds `kz`, and this is the test that catches a transpose.

    A wedge placed about `kx` instead has the same opening angle and the same
    missing fraction, so it agrees with the correct one on every summary number
    while being rotated ninety degrees from the physical geometry. Only naming
    the axes catches it, so the axes are named.
    """

    size = 16
    present = wedge_present_mask_rfft(size, WedgeGeometry(-60.0, 60.0), device=_DEVICE)
    grid_x, _, grid_z = _frequency_grids(size)
    analytic = (grid_z.abs() <= math.tan(math.radians(60.0)) * grid_x.abs()).to(torch.float32)
    assert torch.equal(present, analytic)

    # A pure beam-direction frequency is never measured: at zero tilt the view
    # fills the kx-ky plane, and no reachable tilt turns far enough to reach kz.
    assert present[1, 0, 0] == 0.0
    # A pure in-plane frequency is measured by the zero-tilt view itself.
    assert present[0, 0, 1] == 1.0
    # The tilt axis is measured at every tilt, since rotating about it does not
    # move it.
    assert present[0, 1, 0] == 1.0


def test_the_mask_is_symmetric_under_negating_the_frequency() -> None:
    """Friedel symmetry holds wherever the grid can express `-k` at all.

    A projection plane through the origin is symmetric under `k -> -k`, so the
    exact wedge is too, and the test condition `f(min) * f(max) <= 0` is exactly
    invariant because both factors change sign together.

    The discrete grid is not, and the exception is worth pinning rather than
    hiding. `fftfreq` puts `-1/2` at index `size/2` and has no `+1/2`, so for an
    even box the Nyquist planes hold a frequency whose negation is not
    representable; negating the other two axes there is not an overall sign flip
    and the product is free to change. Measured: 178 of 4096 bins disagree, and
    every one of them touches a Nyquist index — none in the interior. So the
    asymmetry is the grid's, not the geometry's, and it lives in the outermost
    plane where `irfftn` would symmetrise anything non-Hermitian anyway.
    """

    size = 16
    geometry = WedgeGeometry(-52.0, 61.0, tilt_axis_deg=17.0)
    present = wedge_present_mask_rfft(size, geometry, device=_DEVICE)
    # Build the same mask over the FULL grid so -k is addressable, then compare
    # it against itself with every axis negated.
    frequency = torch.fft.fftfreq(size, dtype=torch.float64)
    grid_z, grid_y, grid_x = torch.meshgrid(frequency, frequency, frequency, indexing="ij")
    azimuth = math.radians(geometry.tilt_axis_deg)
    rotated_x = math.cos(azimuth) * grid_x + math.sin(azimuth) * grid_y
    minimum, maximum = math.radians(geometry.min_tilt_deg), math.radians(geometry.max_tilt_deg)
    full = (
        (rotated_x * math.sin(minimum) + grid_z * math.cos(minimum))
        * (rotated_x * math.sin(maximum) + grid_z * math.cos(maximum))
        <= 0.0
    )
    index = torch.arange(size)
    negated_index = (-index) % size
    disagreeing = full != full[negated_index][:, negated_index][:, :, negated_index]
    nyquist = size // 2
    touches_nyquist = (torch.nonzero(disagreeing) == nyquist).any(dim=1)
    assert bool(disagreeing.any()), "the Nyquist exception is real; if it vanished, say so"
    assert bool(touches_nyquist.all()), (
        f"{int((~touches_nyquist).sum())} bins away from a Nyquist plane are asymmetric, "
        "which would be a defect in the geometry rather than in the grid"
    )
    # And the half the rfft keeps agrees with the full grid on its own bins —
    # excluding the x Nyquist column, where the two grids genuinely disagree:
    # `rfftfreq` ends at +1/2 and `fftfreq` holds -1/2 in the same slot, the
    # same aliasing the interior check above isolates.
    assert torch.equal(
        present[:, :, : size // 2], full[:, :, : size // 2].to(torch.float32)
    )


def test_an_asymmetric_range_measures_only_the_tilts_it_covers() -> None:
    """A lamella's range is not symmetric, so the wedge must not assume it is."""

    size = 16
    symmetric = wedge_present_mask_rfft(size, WedgeGeometry(-60.0, 60.0), device=_DEVICE)
    positive_only = wedge_present_mask_rfft(size, WedgeGeometry(0.0, 60.0), device=_DEVICE)
    # Dropping the negative tilts can only ever remove measured directions.
    assert torch.all(positive_only <= symmetric)
    assert float(positive_only.sum()) < float(symmetric.sum())

    # A frequency annihilated by a tilt of -30 degrees: kx sin(-30) + kz cos(-30) = 0.
    # No bin lands exactly on that plane on a discrete grid — the closest at box
    # 16 is 0.0022 away — so take the closest one rather than demanding a
    # tolerance that the grid cannot meet.
    tilt = math.radians(-30.0)
    grid_x, _, grid_z = _frequency_grids(size)
    residual = (grid_x * math.sin(tilt) + grid_z * math.cos(tilt)).abs()
    # Well away from the origin, where every wedge meets and the test says nothing.
    far_enough = torch.sqrt(grid_x**2 + grid_z**2) > 0.2
    masked_residual = torch.where(far_enough, residual, torch.full_like(residual, math.inf))
    index = tuple(
        int(value) for value in torch.nonzero(masked_residual == masked_residual.min())[0]
    )
    assert symmetric[index] == 1.0, "a tilt inside [-60, 60] must measure its own plane"
    assert positive_only[index] == 0.0, "a tilt outside [0, 60] must not be measured"


def test_rotating_the_tilt_axis_turns_the_wedge_with_it() -> None:
    """`tilt_axis_deg` 90 swaps the roles of x and y, and nothing else."""

    size = 16
    along_y = wedge_present_mask_rfft(size, WedgeGeometry(-60.0, 60.0), device=_DEVICE)
    along_x = wedge_present_mask_rfft(
        size, WedgeGeometry(-60.0, 60.0, tilt_axis_deg=90.0), device=_DEVICE
    )
    grid_x, grid_y, grid_z = _frequency_grids(size)
    limit = math.tan(math.radians(60.0))
    # With the axis along y the condition reads on kx; along x it reads on ky.
    assert torch.equal(along_y, (grid_z.abs() <= limit * grid_x.abs()).to(torch.float32))
    assert torch.allclose(along_x, (grid_z.abs() <= limit * grid_y.abs()).to(torch.float32))


def test_the_reference_frame_mask_is_the_lab_mask_at_the_rotated_frequency() -> None:
    """Scoring in the reference frame needs `A`, not its transpose.

    An aligned particle is `raw(A x_ref)`, so its transform is `raw_hat(A k_ref)`
    and its missing region in reference-frame coordinates is where the LAB
    condition fails at `A k`. Checked by building a wedge-limited lab volume,
    resampling it exactly as the backend does, and asking how much power lands
    outside the predicted region.

    The rotation is chosen to be far from any symmetry of the wedge, because
    near one the transpose is nearly as good and the test would prove nothing:
    measured, `A` leaves 0.053 of the power outside against 0.344 for `A.T`
    here, but only 0.030 against 0.056 at (12, 80, 5), and at the identity both
    are exactly zero since there `A` and `A.T` are the same matrix.
    """

    size = 24
    geometry = WedgeGeometry(-60.0, 60.0)
    phantom = _granular_phantom(size)
    lab_present = wedge_present_mask_rfft(size, geometry, device=_DEVICE)
    matrix = relion_euler_to_matrix(41.0, 33.0, -72.0)

    lab = torch.fft.irfftn(
        torch.fft.rfftn(resample_volume_zyx(phantom, matrix.T, np.zeros(3), device=_DEVICE))
        * lab_present,
        s=(size, size, size),
    )
    aligned = resample_volume_zyx(lab, matrix, np.zeros(3), device=_DEVICE)
    spectrum = torch.fft.rfftn(aligned - aligned.mean())
    power = spectrum.real.square() + spectrum.imag.square()

    for label, candidate in (("A", matrix), ("A.T", matrix.T)):
        present = wedge_present_mask_rfft(size, geometry, matrix=candidate, device=_DEVICE)
        outside = float((power * (1.0 - present)).sum() / power.sum())
        if label == "A":
            # Not zero: rotating in real space with zero padding leaks a little
            # power across the boundary. Small is the claim, not absent.
            assert outside < 0.10, f"A leaves {outside:.3f} of the power outside the wedge"
        else:
            assert outside > 0.25, (
                "the transpose must be clearly wrong, otherwise this test proves nothing; "
                f"it leaves only {outside:.3f} outside"
            )


@pytest.mark.parametrize("size", [16, 15])
def test_hermitian_weights_turn_a_spectral_sum_into_a_real_space_one(size: int) -> None:
    """Both parities, because the self-conjugate planes differ between them."""

    torch.manual_seed(0)
    volume = torch.randn(size, size, size)
    spectrum = torch.fft.rfftn(volume)
    weights = hermitian_weights_rfft(size, _DEVICE)
    spectral = float(
        (weights * (spectrum.real.square() + spectrum.imag.square())).sum()
    ) / size**3
    assert spectral == pytest.approx(float((volume * volume).sum()), rel=1e-5)


def test_the_constrained_score_is_the_correlation_of_the_restricted_volumes() -> None:
    """The Fourier-side shortcut has to equal the thing it is a shortcut for."""

    size = 16
    geometry = WedgeGeometry(-60.0, 60.0)
    present = wedge_present_mask_rfft(size, geometry, device=_DEVICE)
    weights = hermitian_weights_rfft(size, _DEVICE)
    torch.manual_seed(1)
    first, second = torch.randn(size, size, size), torch.randn(size, size, size)
    first_spectrum = torch.fft.rfftn(first - first.mean())
    second_spectrum = torch.fft.rfftn(second - second.mean())

    _, correlation = constrained_shift_and_correlation(
        first_spectrum, second_spectrum, present, weights, size=size, max_shift_px=0
    )
    restricted_first = torch.fft.irfftn(first_spectrum * present, s=(size, size, size))
    restricted_second = torch.fft.irfftn(second_spectrum * present, s=(size, size, size))
    direct = float(
        (restricted_first * restricted_second).sum()
        / torch.sqrt(
            (restricted_first * restricted_first).sum()
            * (restricted_second * restricted_second).sum()
        )
    )
    assert correlation == pytest.approx(direct, abs=1e-6)

    shift, self_correlation = constrained_shift_and_correlation(
        first_spectrum, first_spectrum, present, weights, size=size, max_shift_px=2
    )
    assert self_correlation == pytest.approx(1.0, abs=1e-5)
    assert shift.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "true_delta", [[0, 0, 0], [2, 0, 0], [0, -3, 0], [0, 0, 2], [1, -2, 1]]
)
@pytest.mark.parametrize("angles", [(0.0, 0.0, 0.0), (37.0, 24.0, -61.0)])
def test_the_constrained_score_recovers_a_known_shift(
    true_delta: list[int], angles: tuple[float, float, float]
) -> None:
    """The shift convention has to survive the wedge restriction unchanged.

    Built the way the backend folds a shift in: the target is the base volume
    resampled with `shift + matrix @ delta`, so the scorer must return `delta`
    in reference-frame `(x, y, z)`. Also run at a non-identity rotation, which
    is where a frame or axis-order error hides.
    """

    size = 24
    matrix = relion_euler_to_matrix(*angles)
    delta = np.array(true_delta, dtype=float)
    phantom = _granular_phantom(size)
    base = resample_volume_zyx(phantom, matrix, np.zeros(3), device=_DEVICE)
    target = resample_volume_zyx(phantom, matrix, matrix @ delta, device=_DEVICE)

    present = wedge_present_mask_rfft(
        size, WedgeGeometry(-60.0, 60.0), matrix=matrix, device=_DEVICE
    )
    weights = hermitian_weights_rfft(size, _DEVICE)
    recovered, correlation = constrained_shift_and_correlation(
        torch.fft.rfftn(base - base.mean()),
        torch.fft.rfftn(target - target.mean()),
        present,
        weights,
        size=size,
        max_shift_px=4,
    )
    assert recovered.tolist() == delta.tolist()
    assert correlation > 0.9


def test_a_shift_window_reaching_the_box_edge_still_decodes_the_peak() -> None:
    """`2 * radius + 1` overruns the window at the edge; the span must come from
    the window itself, or the index decoding silently uses the wrong stride."""

    size = 16
    present = wedge_present_mask_rfft(size, WedgeGeometry(-60.0, 60.0), device=_DEVICE)
    weights = hermitian_weights_rfft(size, _DEVICE)
    phantom = _granular_phantom(size)
    delta = np.array([1.0, -2.0, 1.0])
    base = resample_volume_zyx(phantom, np.eye(3), np.zeros(3), device=_DEVICE)
    target = resample_volume_zyx(phantom, np.eye(3), delta, device=_DEVICE)
    recovered, _ = constrained_shift_and_correlation(
        torch.fft.rfftn(base - base.mean()),
        torch.fft.rfftn(target - target.mean()),
        present,
        weights,
        size=size,
        max_shift_px=size,  # deliberately past half the box
    )
    assert recovered.tolist() == delta.tolist()


def test_the_wedge_blind_score_walks_off_correct_poses_and_the_constrained_one_holds() -> None:
    """The regression test for the whole change, and it is a comparison.

    Every particle is handed its EXACTLY correct orientation and there is no
    noise at all, so a sound score has nothing to do and any movement is damage.
    The reference is the mean of the aligned wedge-limited particles, which is
    what a real iteration faces and is the condition that matters: against a
    clean unwedged reference both scores behave identically, so the bias comes
    from the reference's own anisotropic coverage meeting each candidate's
    wedge, not from the particle's wedge alone.

    Measured on this fixed seed: the shipped wedge-blind score keeps 1 of 8
    correct poses at a mean error of 9.27 degrees; the constrained score keeps
    6 of 8 at 3.53 degrees. The thresholds below sit either side of that with
    room, and the assertion is relative, so the test states a comparison rather
    than a magic number.
    """

    size = 24
    particle_count = 8
    geometry = WedgeGeometry(-60.0, 60.0)
    weights = hermitian_weights_rfft(size, _DEVICE)
    lab_present = wedge_present_mask_rfft(size, geometry, device=_DEVICE)
    phantom = _granular_phantom(size)

    generator = np.random.default_rng(11)
    poses = [
        relion_euler_to_matrix(*generator.uniform(-180, 180, size=3))
        for _ in range(particle_count)
    ]
    particles = [
        center_particle_volume(
            torch.fft.irfftn(
                torch.fft.rfftn(resample_volume_zyx(phantom, pose.T, np.zeros(3), device=_DEVICE))
                * lab_present,
                s=(size, size, size),
            )
        )
        for pose in poses
    ]
    reference = low_pass_filter_volume(
        torch.stack(
            [
                resample_volume_zyx(particle, pose, np.zeros(3), device=_DEVICE)
                for particle, pose in zip(particles, poses, strict=True)
            ]
        ).mean(dim=0),
        cutoff_angstrom=60.0,
        pixel_spacing_angstrom=_APIX,
    )
    reference_spectrum = torch.fft.rfftn(reference - reference.mean())

    candidates = [
        perturbation_matrix(polar, azimuth, spin)
        for polar, azimuth in cone_offsets(20.0, 10.0)
        for spin in (0.0, 10.0, -10.0)
    ]
    assert any(_rotation_angle_deg(matrix) < 1e-9 for matrix in candidates), (
        "the search must be able to stay where it is, or nothing below means anything"
    )

    blind_errors: list[float] = []
    constrained_errors: list[float] = []
    for pose, particle in zip(poses, particles, strict=True):
        best_blind = (-math.inf, np.eye(3))
        best_constrained = (-math.inf, np.eye(3))
        for perturbation in candidates:
            matrix = pose @ perturbation
            rotated = resample_volume_zyx(particle, matrix, np.zeros(3), device=_DEVICE)
            _, blind = _best_shift_and_correlation(rotated, reference, max_shift_px=0)
            _, constrained = constrained_shift_and_correlation(
                torch.fft.rfftn(rotated - rotated.mean()),
                reference_spectrum,
                wedge_present_mask_rfft(size, geometry, matrix=matrix, device=_DEVICE),
                weights,
                size=size,
                max_shift_px=0,
            )
            if blind > best_blind[0]:
                best_blind = (blind, perturbation)
            if constrained > best_constrained[0]:
                best_constrained = (constrained, perturbation)
        blind_errors.append(_rotation_angle_deg(best_blind[1]))
        constrained_errors.append(_rotation_angle_deg(best_constrained[1]))

    blind_kept = sum(1 for error in blind_errors if error == 0.0)
    constrained_kept = sum(1 for error in constrained_errors if error == 0.0)
    blind_mean = float(np.mean(blind_errors))
    constrained_mean = float(np.mean(constrained_errors))

    assert blind_kept <= 2, (
        "the wedge-blind score is supposed to walk off correct poses here; if it "
        f"stopped doing so ({blind_kept}/8 kept) this test no longer compares anything"
    )
    assert constrained_kept >= 5, f"constrained score kept only {constrained_kept}/8"
    assert constrained_mean < blind_mean / 2.0, (
        f"constrained mean error {constrained_mean:.2f} deg is not clearly better than "
        f"the wedge-blind {blind_mean:.2f} deg"
    )


def test_compensation_recovers_the_truth_that_a_plain_mean_leaves_attenuated() -> None:
    """The claim behind Fourier compensation, with the control that makes it one.

    Particles at KNOWN poses, wedge-limited, no noise. A plain real-space mean
    leaves each frequency attenuated by how few particles measured it; dividing
    by that count should recover the phantom more closely.

    The control is an exact identity rather than a second comparison, because
    the obvious second comparison does not exist: there is NO tilt range whose
    coverage is flat. Present requires some tilt with
    `kx sin t + kz cos t = 0`, so the whole `kx = 0` plane away from `kz = 0`
    needs `t = +-90` and is never measured however wide the range is — a first
    attempt at this test used +-89 degrees expecting a flat denominator and was
    wrong for exactly that reason.

    What is testable exactly: on a bin every particle measured, the denominator
    IS the particle count, so the compensated spectrum must equal the plain
    mean's there, bin for bin. And it must differ where coverage is partial, or
    the division is doing nothing. Those two together say the code divides out
    coverage and nothing else.
    """

    from cryoet_pipeline.backends.subtomogram import _compensated_reference, _Compensation

    size = 24
    geometry = WedgeGeometry(-45.0, 45.0)
    phantom = _granular_phantom(size)
    generator = np.random.default_rng(5)
    poses = [relion_euler_to_matrix(*generator.uniform(-180, 180, size=3)) for _ in range(24)]

    present = wedge_present_mask_rfft(size, geometry, device=_DEVICE)
    aligned = []
    for pose in poses:
        lab = resample_volume_zyx(phantom, pose.T, np.zeros(3), device=_DEVICE)
        lab = torch.fft.irfftn(torch.fft.rfftn(lab) * present, s=(size,) * 3)
        aligned.append(resample_volume_zyx(lab, pose, np.zeros(3), device=_DEVICE))

    compensation = _Compensation(
        geometries=[geometry] * len(poses), box_size=size, device=_DEVICE
    )
    compensated, coverage = _compensated_reference(
        aligned, poses, compensation, [geometry] * len(poses)
    )
    # Each volume's own mean removed, matching what _compensated_reference does,
    # so the comparison below is about coverage and not about the DC term.
    plain = torch.stack([volume - volume.mean() for volume in aligned]).mean(dim=0)

    truth = phantom - phantom.mean()

    def correlation(volume: torch.Tensor) -> float:
        centred = volume - volume.mean()
        norm = torch.sqrt((centred * centred).sum() * (truth * truth).sum())
        return float((centred * truth).sum() / norm)

    assert 0.0 <= float(coverage.min()) <= float(coverage.max()) <= 1.0
    assert correlation(compensated) > correlation(plain), (
        "compensation did not improve agreement with the truth on wedge-limited "
        f"particles: {correlation(compensated):.4f} against {correlation(plain):.4f}"
    )

    plain_spectrum = torch.fft.rfftn(plain)
    compensated_spectrum = torch.fft.rfftn(compensated)
    fully_covered = coverage >= 1.0 - 1e-6
    partly_covered = (coverage > 0.0) & (coverage < 1.0 - 1e-6)
    assert bool(fully_covered.any()) and bool(partly_covered.any()), (
        "this geometry gives no mix of full and partial coverage, so neither half "
        "of the identity below is being exercised"
    )
    assert torch.allclose(
        compensated_spectrum[fully_covered],
        plain_spectrum[fully_covered],
        atol=1e-4,
        rtol=1e-3,
    ), "on a frequency every particle measured, compensation must change nothing"
    partial_difference = float(
        (compensated_spectrum[partly_covered] - plain_spectrum[partly_covered]).abs().max()
    )
    assert partial_difference > 1e-3, (
        f"compensation changed nothing where coverage is partial (max {partial_difference:.2e}), "
        "so it is not dividing out coverage at all"
    )


def test_the_backend_holds_correct_poses_with_the_wedge_and_loses_them_without(
    tmp_path: Path,
) -> None:
    """The same comparison as the scorer test, but through the whole backend.

    This is the one that would catch a wiring fault: the scorer could be right
    and unused. It also runs the iterated loop with reference updates rather
    than a single ballot, which the scorer test does not.

    The no-wedge arm is the control and it is what makes the wedge arm readable:
    on data with no wedge in it both scores leave the poses alone, so any
    movement in the wedge arm is the wedge's doing and not the phantom's, the
    grid's, or the schedule's. Measured on this fixed seed, box 24, 8 particles:
    control 7/8 held at mean 0.00 deg for both scores; wedge-limited data 1/8
    held at 9.27 deg for the whole-box score against 6/8 at 3.53 deg for the
    constrained one.

    The phantom has to be the granular one. A smooth two-lobe phantom is nearly
    isotropic — it keeps 0.9996 of its power below 60 A — and on it the same
    end-to-end comparison came out 3/6 against 2/6 at box 24, which is no
    separation at all. That is a property of the phantom, not of the fix.
    """

    box_size = 24
    generator = np.random.default_rng(11)
    poses = [
        (float(rot), float(tilt), float(psi))
        for rot, tilt, psi in generator.uniform(-180, 180, size=(8, 3))
    ]

    control_blind = _refined_pose_errors(
        tmp_path / "control-blind",
        box_size=box_size,
        poses=poses,
        wedge_the_data=False,
        missing_wedge=False,
    )
    control_constrained = _refined_pose_errors(
        tmp_path / "control-constrained",
        box_size=box_size,
        poses=poses,
        wedge_the_data=False,
        missing_wedge=True,
    )
    assert control_blind.mean() < 1.0, (
        "with no wedge in the data the whole-box score has nothing to gain by moving; "
        f"it moved by {control_blind.mean():.2f} deg, so this comparison has another cause in it"
    )
    assert control_constrained.mean() < 1.0

    blind = _refined_pose_errors(
        tmp_path / "wedged-blind",
        box_size=box_size,
        poses=poses,
        wedge_the_data=True,
        missing_wedge=False,
    )
    constrained = _refined_pose_errors(
        tmp_path / "wedged-constrained",
        box_size=box_size,
        poses=poses,
        wedge_the_data=True,
        missing_wedge=True,
    )
    blind_held = int((blind == 0.0).sum())
    constrained_held = int((constrained == 0.0).sum())
    assert blind_held <= 2, (
        f"the whole-box score is supposed to lose these poses; it held {blind_held}/8, "
        "so the comparison below no longer compares anything"
    )
    assert constrained_held >= 5, f"the constrained score held only {constrained_held}/8"
    assert constrained.mean() < blind.mean() / 2.0, (
        f"constrained mean error {constrained.mean():.2f} deg against whole-box "
        f"{blind.mean():.2f} deg is not the clear improvement this test claims"
    )


@pytest.mark.parametrize(
    ("minimum", "maximum", "match"),
    [
        (60.0, -60.0, "non-empty"),
        (-60.0, -60.0, "non-empty"),
        (-90.0, 60.0, "strictly inside"),
        (-60.0, 90.0, "strictly inside"),
        (float("nan"), 60.0, "finite"),
    ],
)
def test_a_wedge_geometry_refuses_a_range_no_stage_could_produce(
    minimum: float, maximum: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        WedgeGeometry(minimum, maximum)


def test_a_wedge_geometry_takes_its_range_from_the_views_it_is_given() -> None:
    """Excluding views is expressed by not passing them."""

    geometry = WedgeGeometry.from_tilt_angles([-51.0, -30.0, 0.0, 30.0, 57.0])
    assert (geometry.min_tilt_deg, geometry.max_tilt_deg) == (-51.0, 57.0)
    assert geometry.missing_half_angle_deg == pytest.approx(33.0)
    with pytest.raises(ValueError, match="at least one tilt angle"):
        WedgeGeometry.from_tilt_angles([])


def test_each_series_takes_the_wedge_its_tomogram_was_built_with(tmp_path: Path) -> None:
    """The solved angles of the views the tomogram kept, not the manifest's nominal ones.

    On ryr1 039 fine alignment shifted the angles by +7 deg and correct-ctf left
    out the two most negative views: nominal -42..+42, built from -29..+49.
    """

    from cryoet_pipeline.artifacts import ArtifactRegistry
    from cryoet_pipeline.backends.subtomogram import _CombinedParticles, _wedge_geometry_per_particle
    from cryoet_pipeline.cli import _tomogram_tilt_ranges

    angles = tmp_path / "039.tlt"
    angles.write_text("".join(f"{angle:.2f}\n" for angle in (-29.0, -26.0, 0.0, 46.0, 49.0)))
    registry = ArtifactRegistry.empty()
    registry.add(
        Artifact(
            id="TS_WEDGE:tomogram:fine:bin8",
            kind=ArtifactKind.TOMOGRAM,
            path=tmp_path / "tomogram.zarr",
            parameters={"tilt_series_id": "TS_WEDGE", "tilt_angles_path": str(angles)},
        )
    )
    stack = Artifact(
        id="TS_WEDGE:particle_stack:bin8",
        kind=ArtifactKind.PARTICLE_STACK,
        path=tmp_path / "particles.zarr",
        parameters={
            "tilt_series_id": "TS_WEDGE",
            "input_tomogram_id": "TS_WEDGE:tomogram:fine:bin8",
        },
    )

    ranges = _tomogram_tilt_ranges(registry, [stack])
    assert ranges == {"TS_WEDGE": [-29.0, 49.0]}

    combined = _CombinedParticles(
        volumes=[None, None], records=[], box_size=8, tilt_series_ids=["TS_WEDGE"],
        stack_ids=[stack.id], counts=[2],
    )
    geometries = _wedge_geometry_per_particle(
        [(stack, _manifest())], combined, min_tilt_override=None, max_tilt_override=None,
        tilt_axis_deg=0.0, ranges_by_series={"TS_WEDGE": (-29.0, 49.0)},
    )
    assert [(g.min_tilt_deg, g.max_tilt_deg) for g in geometries] == [(-29.0, 49.0)] * 2

    # Without the tomogram's record, the manifest's -60..60 as before.
    fallback = _wedge_geometry_per_particle(
        [(stack, _manifest())], combined, min_tilt_override=None, max_tilt_override=None,
        tilt_axis_deg=0.0,
    )
    assert (fallback[0].min_tilt_deg, fallback[0].max_tilt_deg) == (-60.0, 60.0)
