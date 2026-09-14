from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from cryoet_pipeline.alignment import (
    best_masked_shift_and_correlation,
    prepare_masked_ncc,
    soft_spherical_alignment_mask,
)
from cryoet_pipeline.backends.subtomogram import (
    _best_shift_and_correlation,
    _search_multi_reference,
)
from cryoet_pipeline.particles import ParticleRecord
from cryoet_pipeline.rotations import relion_euler_to_matrix, resample_volume_zyx


def direct_ncc(p, q, w):
    p, q, w = (np.asarray(a, dtype=np.float64) for a in (p, q, w))
    p = p - (w * p).sum() / w.sum()
    q = q - (w * q).sum() / w.sum()
    return (w * p * q).sum() / np.sqrt((w * p * p).sum() * (w * q * q).sum())


def test_masked_fft_matches_explicit_shift_dependent_weighted_pearson():
    rng = np.random.default_rng(31)
    q = rng.normal(size=(16, 16, 16)).astype(np.float32)
    # Known shift, with strong irrelevant box-edge density and an affine intensity change.
    p = 2 * np.roll(q, (1, -2, 1), axis=(0, 1, 2)) + 3
    p[:2] = rng.normal(20, 5, size=p[:2].shape)
    weight = soft_spherical_alignment_mask(
        16, pixel_spacing_angstrom=10, diameter_angstrom=80, device=torch.device("cpu")
    )
    padded = np.pad(p, 2)
    expected = []
    for dz, dy, dx in itertools.product(range(-2, 3), repeat=3):
        shifted = padded[2 + dz : 18 + dz, 2 + dy : 18 + dy, 2 + dx : 18 + dx]
        expected.append((direct_ncc(shifted, q, weight), [dx, dy, dz]))
    score, delta = max(expected, key=lambda item: item[0])
    assert delta == [1, -2, 1]
    actual_delta, actual_score = _best_shift_and_correlation(
        torch.from_numpy(p), torch.from_numpy(q), max_shift_px=2, alignment_mask=weight
    )
    np.testing.assert_array_equal(actual_delta, delta)
    assert actual_score == pytest.approx(score, abs=3e-6)
    assert actual_score == pytest.approx(1, abs=3e-6)


def test_all_ones_zero_shift_matches_legacy_and_none_is_unchanged():
    generator = torch.Generator().manual_seed(14)
    p, q = torch.randn(2, 12, 12, 12, generator=generator)
    legacy = _best_shift_and_correlation(p, q, max_shift_px=0)
    default = _best_shift_and_correlation(p, q, max_shift_px=0, alignment_mask=None)
    masked = _best_shift_and_correlation(p, q, max_shift_px=0, alignment_mask=torch.ones_like(q))
    assert legacy[1] == default[1]
    assert masked[1] == pytest.approx(legacy[1], abs=1e-6)
    assert masked[1] == pytest.approx(direct_ncc(p, q, torch.ones_like(q)), abs=1e-6)


def test_masked_search_shift_frame_and_final_resampling_agree():
    raw = torch.randn(16, 16, 16, generator=torch.Generator().manual_seed(7))
    matrix = relion_euler_to_matrix(17, 31, -23)
    start_shift = np.array([0.2, -0.1, 0.3])
    delta = np.array([1, -1, 1])
    final_shift = start_shift + matrix @ delta
    reference = resample_volume_zyx(raw, matrix, final_shift, device=torch.device("cpu"))
    weight = soft_spherical_alignment_mask(
        16, pixel_spacing_angstrom=10, diameter_angstrom=80, device=torch.device("cpu")
    )
    record = ParticleRecord(0, 0, 0, 0, 17, 31, -23, *start_shift, 0.5, 1, 1)
    result = _search_multi_reference(
        raw,
        [reference],
        record,
        cone=[(0, 0)],
        in_plane_offsets=[0],
        max_shift_px=2,
        device=torch.device("cpu"),
        alignment_mask=weight,
    )
    np.testing.assert_allclose(result.shift, final_shift, atol=1e-7)
    assert result.correlation == pytest.approx(
        direct_ncc(result.aligned_volume, reference, weight), abs=3e-6
    )
    assert result.correlation == pytest.approx(1, abs=3e-6)


def test_mask_guards_empty_degenerate_and_wrapping_support():
    q = torch.randn(8, 8, 8, generator=torch.Generator().manual_seed(5))
    with pytest.raises(ValueError, match="empty"):
        prepare_masked_ncc(q, torch.zeros_like(q))
    with pytest.raises(ValueError, match="variance"):
        prepare_masked_ncc(torch.ones_like(q), torch.ones_like(q))
    with pytest.raises(ValueError, match="box edge"):
        best_masked_shift_and_correlation(
            q, prepare_masked_ncc(q, torch.ones_like(q)), max_shift_px=1
        )
    with pytest.raises(ValueError, match="finite NCC"):
        best_masked_shift_and_correlation(
            torch.zeros_like(q), prepare_masked_ncc(q, torch.ones_like(q)), max_shift_px=0
        )
    with pytest.raises(ValueError, match="fit inside"):
        soft_spherical_alignment_mask(
            8, pixel_spacing_angstrom=10, diameter_angstrom=100, device=torch.device("cpu")
        )


def test_native_masked_backend_reports_score_and_keeps_output_unmasked(tmp_path: Path):
    import json

    import mrcfile
    from test_subtomogram_backend import (
        _blob_reference,
        _cone_search_context,
        _manifest,
        _write_particle_stack,
    )

    from cryoet_pipeline.backends.subtomogram import ConeSearchRefinementBackend
    from cryoet_pipeline.particles import read_particle_star

    volume = _blob_reference(20)
    particles = _write_particle_stack(tmp_path, [volume, volume.clone()], [(0, 0, 0)] * 2)
    context = _cone_search_context(tmp_path, num_iterations=1, alignment_mask_diameter_angstrom=120)
    artifacts = ConeSearchRefinementBackend().average([particles], [_manifest()], context)
    report = json.loads(artifacts[-1].path.read_text())
    assert report["alignment_score"] == "weighted-ncc"
    assert report["alignment_mask_diameter_angstrom"] == 120
    assert report["alignment_mask_flat_radius_fraction"] == 0.8
    assert artifacts[0].parameters["alignment_score"] == "weighted-ncc"
    records = read_particle_star(Path(artifacts[0].parameters["metadata_path"]))
    centered = volume - volume.mean()
    expected = torch.stack(
        [
            resample_volume_zyx(
                centered,
                relion_euler_to_matrix(r.rot_deg, r.tilt_deg, r.psi_deg),
                np.array([r.shift_x_px, r.shift_y_px, r.shift_z_px]),
                device=torch.device("cpu"),
            )
            for r in records
        ]
    ).mean(0)
    with mrcfile.open(artifacts[0].path) as handle:
        np.testing.assert_allclose(handle.data, expected, atol=1e-6)
