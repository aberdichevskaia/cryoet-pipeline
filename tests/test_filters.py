from __future__ import annotations

import numpy as np
import pytest
import torch

from cryoet_pipeline.filters import low_pass_filter_volume

BOX = 32
APIX = 5.4


def _noise(seed: int, size: int = BOX) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.normal(size=(size, size, size)).astype(np.float32))


def test_low_pass_leaves_the_mean_alone() -> None:
    """DC is always passed, so filtering never moves a volume's mean.

    Worth pinning: particles are mean-centred before resampling precisely so that
    zero padding is neutral, and a filter that shifted the mean would put that
    back.
    """

    volume = _noise(0)

    for cutoff in (200.0, 100.0, 30.0):
        filtered = low_pass_filter_volume(
            volume, cutoff_angstrom=cutoff, pixel_spacing_angstrom=APIX
        )
        assert filtered.mean().item() == pytest.approx(volume.mean().item(), abs=1e-6)


def test_a_tighter_low_pass_removes_more_power() -> None:
    volume = _noise(1)

    deviations = [
        low_pass_filter_volume(
            volume, cutoff_angstrom=cutoff, pixel_spacing_angstrom=APIX
        )
        .std()
        .item()
        for cutoff in (200.0, 100.0, 50.0, 30.0, 15.0)
    ]

    assert deviations == sorted(deviations)
    assert deviations[-1] < volume.std().item()


def test_the_pass_band_matches_the_fsc_resolution_convention() -> None:
    """The limit fed in must mean the same thing as the resolution read out.

    A low-pass driven by an FSC curve is a feedback loop, so the two have to agree
    on what "30 angstrom" is. `compute_fsc_curve` calls shell k
    `box * spacing / k`, so a cutoff of `box * spacing / k` must keep shell k and
    drop shells well beyond it.
    """

    volume = _noise(2)
    shell = 6
    cutoff = BOX * APIX / shell

    filtered = low_pass_filter_volume(
        volume, cutoff_angstrom=cutoff, pixel_spacing_angstrom=APIX, edge_width_shells=0.0
    )
    spectrum = torch.fft.rfftn(filtered)

    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.fft.fftfreq(BOX, d=APIX),
        torch.fft.fftfreq(BOX, d=APIX),
        torch.fft.rfftfreq(BOX, d=APIX),
        indexing="ij",
    )
    shell_index = torch.sqrt(grid_z**2 + grid_y**2 + grid_x**2) * BOX * APIX

    assert spectrum[shell_index <= shell - 0.5].abs().max() > 0.0
    assert spectrum[shell_index > shell + 0.5].abs().max() == pytest.approx(0.0, abs=1e-4)


def test_a_soft_edge_rings_less_than_a_hard_cutoff() -> None:
    """The reason the edge is soft, stated as a measurement.

    A brick wall convolves the map with a sinc, putting concentric shells around
    every feature — structure a reader or an FSC can take for density. Measured on
    a point: the hard cutoff leaves a larger extreme excursion away from the
    centre than the raised-cosine edge does.
    """

    point = torch.zeros((BOX, BOX, BOX), dtype=torch.float32)
    point[BOX // 2, BOX // 2, BOX // 2] = 1.0
    cutoff = BOX * APIX / 6

    hard = low_pass_filter_volume(
        point, cutoff_angstrom=cutoff, pixel_spacing_angstrom=APIX, edge_width_shells=0.0
    )
    soft = low_pass_filter_volume(
        point, cutoff_angstrom=cutoff, pixel_spacing_angstrom=APIX, edge_width_shells=2.0
    )

    # The corner is far from the point, so anything there is filter ringing.
    corner = (slice(0, 6), slice(0, 6), slice(0, 6))
    assert soft[corner].abs().max().item() < hard[corner].abs().max().item()


def test_low_pass_rejects_nonsense_arguments() -> None:
    volume = _noise(3)

    with pytest.raises(ValueError, match="expected a 3D volume"):
        low_pass_filter_volume(
            volume[0], cutoff_angstrom=30.0, pixel_spacing_angstrom=APIX
        )
    with pytest.raises(ValueError, match="cutoff must be positive"):
        low_pass_filter_volume(volume, cutoff_angstrom=0.0, pixel_spacing_angstrom=APIX)
    with pytest.raises(ValueError, match="pixel spacing must be positive"):
        low_pass_filter_volume(volume, cutoff_angstrom=30.0, pixel_spacing_angstrom=-1.0)
    with pytest.raises(ValueError, match="edge width cannot be negative"):
        low_pass_filter_volume(
            volume,
            cutoff_angstrom=30.0,
            pixel_spacing_angstrom=APIX,
            edge_width_shells=-1.0,
        )


def test_a_cutoff_beyond_nyquist_is_a_no_op_in_practice() -> None:
    """Nothing to remove above Nyquist, so the volume comes back as it went in."""

    volume = _noise(4)

    filtered = low_pass_filter_volume(
        volume, cutoff_angstrom=APIX, pixel_spacing_angstrom=APIX
    )

    np.testing.assert_allclose(filtered.numpy(), volume.numpy(), atol=1e-4)
