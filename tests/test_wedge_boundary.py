"""Deterministic boundary regressions; no alignment search or scientific sweep."""

import math

import numpy as np
import pytest
import torch

from cryoet_pipeline.dynamo_core import euler_matrix
from cryoet_pipeline.rotations import relion_euler_to_matrix
from cryoet_pipeline.wedge import (
    WedgeGeometry,
    _wedge_boundary_values_rfft,
    wedge_present_mask_rfft,
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
        ),
    ],
)
def test_tag_222_equivalent_rotations_include_the_entire_tilt_axis(device):
    device = torch.device(device)
    # CZII tag 222 (TS_99_9): its STAR pose, and the same pose read back from
    # the Dynamo export table.
    geometry = WedgeGeometry(-45.02, 44.97)
    native = relion_euler_to_matrix(-120.0, 90.0, 90.0)
    converted = euler_matrix([-6.36110936293e-15, 90.0, -30.0]).T
    np.testing.assert_allclose(native, converted, atol=1e-15, rtol=0)
    masks = [
        wedge_present_mask_rfft(48, geometry, matrix=m, device=device) for m in (native, converted)
    ]
    assert torch.equal(*masks)
    # Reference z maps to the laboratory tilt axis y, including the Nyquist bin.
    assert torch.all(masks[0][:, 0, 0] == 1)
    # Why a strict sign test failed here: on that axis both endpoint values are
    # roundoff, not geometry. On the Mac CPU it excluded all 47 bins for
    # `native` and none for `converted`; the count itself depends on the last
    # bit of sin/cos, so it is not asserted.
    axis = (slice(1, None), 0, 0)
    for matrix in (native, converted):
        lo, hi, tol = _wedge_boundary_values_rfft(48, geometry, matrix=matrix, device=device)
        assert torch.all(tol[axis] > 0)
        assert torch.all(lo[axis].abs() <= tol[axis])
        assert torch.all(hi[axis].abs() <= tol[axis])


def test_endpoint_allowance_does_not_dilate_the_wedge():
    device = torch.device("cpu")
    # k=(1/16, 0, 1/16) is exactly on the -45-degree endpoint plane.
    on = wedge_present_mask_rfft(16, WedgeGeometry(-45.0, 30.0), device=device)
    outside = wedge_present_mask_rfft(16, WedgeGeometry(-45.0 + 1e-8, 30.0), device=device)
    inside = wedge_present_mask_rfft(16, WedgeGeometry(-45.0 - 1e-8, 30.0), device=device)
    assert on[1, 0, 1] == inside[1, 0, 1] == 1
    assert outside[1, 0, 1] == 0
    # The unmeasured pure beam axis must not be admitted by a global epsilon.
    assert torch.all(on[1:, 0, 0] == 0)
    assert torch.all(on[0, :, 0] == 1)


def test_exact_45_degree_wedge_is_the_closed_set():
    # For a symmetric range the condition is |kz| <= tan(a) |kx|, and tan(45)
    # is 1, so every |kx| = |kz| bin sits on an endpoint plane. The strict sign
    # test dropped all of them, because sin and cos of 45 degrees round apart.
    device = torch.device("cpu")
    for size in (15, 16):
        present = wedge_present_mask_rfft(size, WedgeGeometry(-45.0, 45.0), device=device)
        kz = torch.fft.fftfreq(size, dtype=torch.float64).abs()[:, None, None]
        kx = torch.fft.rfftfreq(size, dtype=torch.float64)[None, None, :]
        assert torch.equal(present.bool(), (kz <= kx).expand(present.shape))


def test_roundoff_policy_only_changes_bins_at_a_numerical_boundary():
    device = torch.device("cpu")
    # Exercise the extra tilt-axis rotation and odd/even Fourier grids.
    for size in (15, 16):
        geometry = WedgeGeometry(-53.0, 37.0, tilt_axis_deg=31.0)
        matrix = relion_euler_to_matrix(17.0, 39.0, -61.0)
        lo, hi, tol = _wedge_boundary_values_rfft(size, geometry, matrix=matrix, device=device)
        strict = lo * hi <= 0
        corrected = wedge_present_mask_rfft(size, geometry, matrix=matrix, device=device).bool()
        assert torch.all(corrected[strict])
        away = (lo.abs() > tol) & (hi.abs() > tol)
        assert torch.equal(corrected[away], strict[away])
        # |k|_1 reaches 1.5 only on an even grid, where all three axes sit at Nyquist.
        assert tol.max() <= 16 * np.finfo(float).eps * 1.5
        assert math.isclose(float(tol[0, 0, 0]), 0.0)
