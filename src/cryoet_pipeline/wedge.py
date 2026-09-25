"""The missing wedge of a single-axis tilt series, and a correlation constrained to it.

Why this exists, measured rather than argued. Seeded with *exactly correct*
orientations and no noise at all, `ConeSearchRefinementBackend` moved 100% of
particles off the truth by a mean of 5.90 degrees and dropped the average's
agreement with the known phantom from 0.9717 to 0.9112 — while raising its own
reported correlation. The identical loop on the identical particles with the
wedge removed moved nothing (0.00 degrees, agreement 0.9883 unchanged). The
control is what makes that a measurement: had the wedge been irrelevant, the
control would have degraded too.

The mechanism is not that the wedge adds noise. An unweighted correlation
between a wedge-limited particle and a reference that is not wedge-limited
carries a term that depends on how much *reference* power happens to fall inside
the region the particle actually measured, and that term varies with the
candidate orientation. So the score is maximised partly by rotating a particle
until its own missing region sits where the reference is weakest — which means
the refinement co-aligns the particles' wedges instead of their structure.
Measured: the orientation tensor of the lab tilt axis expressed in the reference
frame climbed 0.496 -> 0.568 -> 0.615 -> 0.638 over three iterations with the
wedge (0.333 is isotropic) and only to 0.511 without it. The damage lands
high-frequency-first, which is what makes a refined map look smoother than the
one it started from: agreement at the 32.6 A shell fell 0.989 -> 0.946 while the
456-114 A shells lost 0.004-0.008.

`docs/roadmap.md` already listed wedge weighting as outstanding, but the reason
given there for tolerating it — "tolerable while orientations are spread,
because each particle's wedge then points somewhere different in the reference
frame and they cancel" — is the exact case measured above, uniformly random
orientations, and it does not hold. The wedges cancel in the *average*. They do
not cancel in the *score*, and the score is what moves the poses.

A soft spatial mask does not fix it (mean error 8.28 vs 6.12 degrees at SNR 0.1;
tensor 0.593 vs 0.638), and neither does a finer low-pass (identical to three
decimals at 19 A). Both were tried before this module was written, which is why
the fix here is in the score's *support* rather than in its weighting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# How `wedge_present_mask_rfft` decides a bin that lies exactly on the wedge
# boundary. Consumers record it, so a map says which rule built it: the strict
# sign test used before this name existed decided such bins by roundoff.
WEDGE_BOUNDARY_POLICY = "float64-relative-roundoff-v1"


@dataclass(frozen=True)
class WedgeGeometry:
    """Which directions a single-axis tilt series actually measured.

    In the `(x, y, z)` physical frame this pipeline uses everywhere else (see
    `rotations.relion_euler_to_matrix`): `z` is the beam direction at zero tilt,
    and the tilt axis lies in the `xy` plane at `tilt_axis_deg` from `y`. An
    IMOD-aligned stack puts the tilt axis along `y`, so zero is the default and
    the value is carried explicitly rather than assumed.

    The range is given as a minimum and a maximum instead of a half-angle
    because a real range is not symmetric: views get excluded, and a FIB lamella
    carries a pre-tilt. A wedge built from a symmetric guess would be wrong on
    exactly the data this pipeline was aimed at.
    """

    min_tilt_deg: float
    max_tilt_deg: float
    tilt_axis_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in ("min_tilt_deg", "max_tilt_deg", "tilt_axis_deg"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.min_tilt_deg >= self.max_tilt_deg:
            raise ValueError(
                f"the tilt range must be non-empty: got min {self.min_tilt_deg} >= "
                f"max {self.max_tilt_deg}"
            )
        # A tilt of exactly +-90 degrees would mean a view along the specimen
        # plane, which no stage reaches and which makes the wedge condition
        # degenerate; refuse it rather than silently producing a full sphere.
        if not -90.0 < self.min_tilt_deg or not self.max_tilt_deg < 90.0:
            raise ValueError(
                "tilt angles must lie strictly inside (-90, 90) degrees; got "
                f"[{self.min_tilt_deg}, {self.max_tilt_deg}]"
            )

    @classmethod
    def from_tilt_angles(
        cls, tilt_angles_deg: Sequence[float], *, tilt_axis_deg: float = 0.0
    ) -> WedgeGeometry:
        """The wedge of the views that were actually used.

        Takes the extremes of the angles it is given, so excluding views is
        expressed by not passing them rather than by a separate flag.
        """

        if not tilt_angles_deg:
            raise ValueError("a wedge needs at least one tilt angle")
        return cls(
            min_tilt_deg=float(min(tilt_angles_deg)),
            max_tilt_deg=float(max(tilt_angles_deg)),
            tilt_axis_deg=float(tilt_axis_deg),
        )

    @property
    def missing_half_angle_deg(self) -> float:
        """Half the opening of the unmeasured cone about the beam axis."""

        return 90.0 - max(abs(self.min_tilt_deg), abs(self.max_tilt_deg))


def wedge_present_mask_rfft(
    size: int,
    geometry: WedgeGeometry,
    *,
    matrix: np.ndarray | torch.Tensor | None = None,
    device: torch.device,
) -> torch.Tensor:
    """Which `rfftn` bins of a cubic volume the tilt series measured.

    Returns a float32 mask, 1.0 where the frequency was measured and 0.0 in the
    missing wedge, on the bin grid of `torch.fft.rfftn` applied to a
    `(z, y, x)`-ordered cube of side `size` — the same layout
    `filters.low_pass_filter_volume` works on.

    The geometry, derived rather than asserted. At tilt `t` the beam points along
    `d(t) = (sin t, 0, cos t)` and the view fills the central plane perpendicular
    to it, so a frequency `k` is measured exactly when some tilt in the range
    annihilates it:

        k . d(t) = kx sin t + kz cos t = 0   for some t in [min, max]

    Written as `f(t) = kx sin t + kz cos t`, this has at most one zero in
    `(-90, 90)` — the interval is shorter than a period — so `f(min) * f(max) <= 0`
    holds precisely when that zero lies inside the range. That is the test used
    here: it needs no division, handles an asymmetric range directly, is exact
    on the `kx = 0` plane (never measured, since no stage reaches +-90 degrees),
    and passes `kx = kz = 0` (measured at every tilt). It is also invariant under
    `k -> -k`, as a mask on the transform of real data has to be.

    The range is closed, and has to stay closed in floating point. A bin on the
    tilt axis is measured at every tilt and a bin on an endpoint plane by the
    extreme view, but both reach the test as residuals of either sign instead of
    zeros, so a strict sign test decided them by roundoff. Measured before the
    allowance existed: two equivalent matrices for one CZII particle (tag 222,
    tilt 90) disagreed on all 47 non-DC bins of its tilt axis, which alone moved
    a 223-particle compensated average by 0.16%; and an exact +-45.0 degree lab
    wedge lost its whole `|kx| = |kz|` boundary, 2256 of 57600 bins at box 48,
    to the rounding of sin and cos of 45 degrees. Endpoint values within a
    float64 roundoff allowance of zero therefore count as zero. The allowance is
    relative to the frequency, not a fitted angle: a bin 1e-8 degrees outside the
    range stays out.

    For a symmetric range this reduces to the familiar `|kz| <= tan(a) |kx|`,
    i.e. a wedge of half-angle `90 - a` about the BEAM axis `kz`. Worth stating
    because the `kx`/`kz` transpose of it — a wedge about `kx` — has the same
    opening angle and the same missing fraction, so it looks right in every
    summary statistic while being rotated 90 degrees from the physical one.

    `matrix` is for scoring in a reference frame. A particle aligned with `A`
    is `raw(A x_ref)`, whose transform is `raw_hat(A k_ref)` because `A` is
    orthogonal; so the aligned particle is missing wherever the LAB condition
    fails at `A k_ref`, and passing `A` evaluates exactly that. Note this is `A`
    itself, not its transpose — the same asymmetry `relion_euler_to_matrix`
    documents for resampling.

    One discretisation caveat, pinned by a test rather than left to be
    rediscovered. The exact wedge is symmetric under `k -> -k`, and so is the
    condition above, but the grid is not: `fftfreq` puts `-1/2` at index
    `size/2` and offers no `+1/2`, so on an even box the Nyquist planes can mark
    a bin present whose conjugate partner is absent. Measured at box 16, that is
    178 of 4096 bins and every one of them lies on such a plane. It is confined
    to the outermost shell, where `irfftn` symmetrises whatever it is handed.
    """

    at_minimum, at_maximum, tolerance = _wedge_boundary_values_rfft(
        size, geometry, matrix=matrix, device=device
    )
    return (
        (torch.minimum(at_minimum, at_maximum) <= tolerance)
        & (torch.maximum(at_minimum, at_maximum) >= -tolerance)
    ).to(torch.float32)


def _wedge_boundary_values_rfft(
    size: int,
    geometry: WedgeGeometry,
    *,
    matrix: np.ndarray | torch.Tensor | None = None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Beam dot products and their frequency-relative roundoff allowance.

    Kept separate so a benchmark can audit the former strict sign rule without
    exposing a production option to restore that numerically unstable policy.
    """
    if size < 2:
        raise ValueError(f"a wedge mask needs a box of at least 2 voxels, got {size}")

    frequency = torch.fft.fftfreq(size, device=device, dtype=torch.float64)
    # rfftn keeps only the non-redundant half of the LAST axis, which is x.
    frequency_x = torch.fft.rfftfreq(size, device=device, dtype=torch.float64)
    grid_z, grid_y, grid_x = torch.meshgrid(
        frequency, frequency, frequency_x, indexing="ij"
    )

    if matrix is not None:
        rotation = torch.as_tensor(matrix, dtype=torch.float64, device=device)
        if rotation.shape != (3, 3):
            raise ValueError(f"expected a 3x3 rotation matrix, got shape {tuple(rotation.shape)}")
        # k' = A @ k, with k ordered (x, y, z) as the rotation matrices are.
        rotated_x = rotation[0, 0] * grid_x + rotation[0, 1] * grid_y + rotation[0, 2] * grid_z
        rotated_y = rotation[1, 0] * grid_x + rotation[1, 1] * grid_y + rotation[1, 2] * grid_z
        rotated_z = rotation[2, 0] * grid_x + rotation[2, 1] * grid_y + rotation[2, 2] * grid_z
    else:
        rotated_x, rotated_y, rotated_z = grid_x, grid_y, grid_z

    if geometry.tilt_axis_deg != 0.0:
        # Bring the tilt axis back onto y, so the condition below can be written
        # in the xz plane. Only the x component is used, but both are rotated so
        # the transform is a rotation rather than a projection.
        azimuth = math.radians(geometry.tilt_axis_deg)
        cos_azimuth, sin_azimuth = math.cos(azimuth), math.sin(azimuth)
        rotated_x, rotated_y = (
            cos_azimuth * rotated_x + sin_azimuth * rotated_y,
            -sin_azimuth * rotated_x + cos_azimuth * rotated_y,
        )

    minimum = math.radians(geometry.min_tilt_deg)
    maximum = math.radians(geometry.max_tilt_deg)
    at_minimum = rotated_x * math.sin(minimum) + rotated_z * math.cos(minimum)
    at_maximum = rotated_x * math.sin(maximum) + rotated_z * math.cos(maximum)
    # A frequency on the tilt axis is measured at every tilt. Its two beam
    # dot products are mathematically zero, but equivalent Euler/matrix paths
    # can leave residuals of either sign (below 1e-16, at most 3.4% of the
    # allowance, in the tag-222 regression).
    # Allow accumulated float64 error in rotation, axis change and endpoint
    # projection. Scale by the ORIGINAL frequency, not its cancelled x/z
    # components. Sixteen eps is a conservative small-operation allowance,
    # not an angular resolution, voxel dilation or data-tuned threshold.
    tolerance = 16 * torch.finfo(torch.float64).eps * (grid_x.abs() + grid_y.abs() + grid_z.abs())
    return at_minimum, at_maximum, tolerance


def hermitian_weights_rfft(size: int, device: torch.device) -> torch.Tensor:
    """Multiplicities that make a sum over `rfftn` bins a sum over the full one.

    `rfftn` stores one of each conjugate pair except on the two self-conjugate
    `x` planes, so `sum(v**2)` over real space is
    `sum(weights * abs(V)**2) / size**3` with these weights and not with ones.
    Needed because the constrained correlation's denominator is a sum over the
    kept bins while its numerator comes back through `irfftn`, which already
    accounts for the pairs — mixing the two conventions inflates the denominator
    by nearly a factor of two and produces "correlations" that cannot reach one.
    """

    weights = torch.full((size, size, size // 2 + 1), 2.0, device=device, dtype=torch.float32)
    weights[..., 0] = 1.0
    if size % 2 == 0:
        weights[..., -1] = 1.0
    return weights


def constrained_shift_and_correlation(
    candidate_spectrum: torch.Tensor,
    reference_spectrum: torch.Tensor,
    present: torch.Tensor,
    hermitian: torch.Tensor,
    *,
    size: int,
    max_shift_px: float,
) -> tuple[np.ndarray, float]:
    """Best shift and correlation, computed only where the particle has data.

    Both operands are restricted to `present` before anything is measured, so
    the score is a Pearson correlation between two volumes that share a support,
    and — the point of the exercise — the support travels with the candidate
    orientation, so a candidate cannot win by moving its missing region onto the
    reference's weak frequencies. Normalising over the shared region rather than
    over the whole box is the half that makes this work: filtering the reference
    alone still leaves each candidate a different amount of reference power to
    be normalised against.

    Restricting the particle too is not redundant. It is already near zero
    outside its own wedge, but only near: rotating in real space with zero
    padding leaks a little power across the boundary, and the leak is
    orientation-dependent, which is the same kind of term this function exists
    to remove.

    The shift search is unchanged in spirit — the whole allowed range comes back
    from one inverse transform — and the returned shift is in reference-frame
    `(x, y, z)`, to be folded into an existing sampling shift as
    `shift + matrix @ delta`, exactly as `_best_shift_and_correlation` is.

    One approximation is left, deliberately, and it belongs to the reference
    rather than to this function: the reference is a plain real-space mean of
    wedge-limited particles, so its Fourier amplitudes already carry whatever
    coverage that set of orientations happened to give. Dividing that out is
    Fourier compensation during averaging, a separate change; until it exists
    the score is constrained but the reference is not equalised.
    """

    if candidate_spectrum.shape != reference_spectrum.shape:
        raise ValueError(
            f"candidate spectrum {tuple(candidate_spectrum.shape)} and reference spectrum "
            f"{tuple(reference_spectrum.shape)} must match"
        )
    if present.shape != candidate_spectrum.shape or hermitian.shape != candidate_spectrum.shape:
        raise ValueError("the wedge mask and Hermitian weights must match the spectra")

    candidate = candidate_spectrum * present
    reference = reference_spectrum * present
    candidate_energy = torch.sum(hermitian * candidate.real.square())
    candidate_energy = candidate_energy + torch.sum(hermitian * candidate.imag.square())
    reference_energy = torch.sum(hermitian * reference.real.square())
    reference_energy = reference_energy + torch.sum(hermitian * reference.imag.square())
    # Both energies are Parseval sums, so each is size**3 times its real-space
    # counterpart; the product is size**6 and the square root size**3.
    norm = torch.sqrt(candidate_energy * reference_energy) / size**3
    if float(norm) <= 1e-12:
        return np.zeros(3), 0.0

    shape = (size, size, size)
    correlation_map = torch.fft.fftshift(
        torch.fft.irfftn(torch.conj(reference) * candidate, s=shape)
    ) / norm

    center = size // 2
    # The window has `2 * radius + 1` samples per axis only while that fits; at
    # the box edge it does not, and deriving the span from the window instead of
    # recomputing it from the radius keeps the index decoding right either way.
    radius = int(min(max(int(math.floor(max_shift_px)), 0), (size - 1) // 2))
    window = correlation_map[
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
        center - radius : center + radius + 1,
    ]
    span = int(window.shape[0])
    flat_index = int(torch.argmax(window).item())
    delta_z = flat_index // (span * span) - radius
    delta_y = (flat_index // span) % span - radius
    delta_x = flat_index % span - radius
    return (
        np.array([float(delta_x), float(delta_y), float(delta_z)]),
        float(window.reshape(-1)[flat_index].item()),
    )
