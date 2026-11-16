"""The missing wedge seen in a tomogram, measured from its Fourier transform.

One of the checks an import makes (docs/entry_and_exit.md, Checks an import
makes). It does not use what the import was told, so it can test that: the
declared tilt angles, whether the volume is mirrored, whether its wedge was
filled.

The geometry is `wedge.py`'s. In the canonical `(z, y, x)` volume the view at
tilt `t` integrates along `d(t) = (sin t, 0, cos t)` in `(x, y, z)`, and fills the
central plane perpendicular to it. Write a frequency's direction in the `kx`-`kz`
plane as the angle `alpha` from `kz` towards `kx`, folded into (-90, 90]. The
range `[a, b]` then measured `alpha` in `[a + 90, b + 90]` (modulo 180), and the
missing sector is `(b - 90, a + 90)`: centred on the range's centre `(a + b) / 2`,
`180 - (b - a)` degrees wide. A mirror of z or x moves that centre to its
negative, so for an asymmetric range the sector's position tells the volume's
side; for a symmetric one it cannot.

`kx` and `kz` alone decide the condition, so every `ky` is pooled. The profile is
the median log power in each `alpha` bin over a shell of frequencies, taken from
a centred crop with its mean removed and a Tukey window on each axis, so that the
crop's edges add no streaks along the axes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

# Frequencies in cycles per voxel. Below the shell, the specimen's large-scale
# variation dominates; above it, the reconstruction's radial filter (tilt's
# -RADIAL 0.35) and the interpolation have taken the power away.
DEFAULT_SHELL = (0.05, 0.30)
DEFAULT_CROP_XY = 256
DEFAULT_CROP_Z = 512
DEFAULT_BIN_DEG = 2.0
FIT_STEP_DEG = 1.0
MIN_RANGE_DEG = 20.0
# No stage collects more than this; a wider fitted range would be a missing sector
# of a few bins, which noise on one axis can fake.
MAX_RANGE_DEG = 160.0

Verdict = Literal["agrees", "mirrored", "inconclusive", "side not testable", "no wedge"]


@dataclass(frozen=True)
class WedgeMeasurement:
    """What the volume's Fourier transform says about its tilt range."""

    bin_centres_deg: list[float]
    profile: list[float]
    """Median log10 power per bin, its median over bins subtracted."""
    fitted_min_deg: float
    fitted_max_deg: float
    fitted_contrast: float
    """Mean log10 power of the measured bins minus the missing ones, at the fit."""
    declared_min_deg: float
    declared_max_deg: float
    declared_contrast: float
    mirrored_contrast: float
    """The same for the declared range negated: what a mirrored volume would show."""
    verdict: Verdict
    reason: str
    shell: tuple[float, float]
    crop_shape: tuple[int, int, int]

    def record(self) -> dict[str, Any]:
        return asdict(self)


def tukey(length: int, alpha: float = 0.5) -> np.ndarray:
    """A Tukey window: flat in the middle, cosine tapers over `alpha` of the length."""

    if length < 2 or alpha <= 0.0:
        return np.ones(length)
    n = np.arange(length, dtype=np.float64)
    width = alpha * (length - 1) / 2.0
    window = np.ones(length)
    left = n < width
    right = n > (length - 1) - width
    window[left] = 0.5 * (1 + np.cos(np.pi * (n[left] / width - 1)))
    window[right] = 0.5 * (1 + np.cos(np.pi * ((n[right] - (length - 1)) / width + 1)))
    return window


def _centred_crop(volume: Any, crop_z: int, crop_xy: int) -> np.ndarray:
    depth, height, width = (int(size) for size in volume.shape)
    sizes = (min(depth, crop_z), min(height, crop_xy), min(width, crop_xy))
    starts = [(full - size) // 2 for full, size in zip((depth, height, width), sizes, strict=True)]
    return np.asarray(
        volume[
            starts[0] : starts[0] + sizes[0],
            starts[1] : starts[1] + sizes[1],
            starts[2] : starts[2] + sizes[2],
        ],
        dtype=np.float64,
    )


def angular_profile(
    volume: Any,
    *,
    shell: tuple[float, float] = DEFAULT_SHELL,
    crop_z: int = DEFAULT_CROP_Z,
    crop_xy: int = DEFAULT_CROP_XY,
    bin_deg: float = DEFAULT_BIN_DEG,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Bin centres in degrees, the median log10 power in each, and the crop's shape.

    `volume` is canonical `(z, y, x)`: a numpy array or anything that slices like
    one (a Zarr array).
    """

    crop = _centred_crop(volume, crop_z, crop_xy)
    if min(crop.shape) < 16:
        raise ValueError(f"a wedge needs a volume of at least 16 voxels a side, got {crop.shape}")
    crop -= crop.mean()
    window = (
        tukey(crop.shape[0])[:, None, None]
        * tukey(crop.shape[1])[None, :, None]
        * tukey(crop.shape[2])[None, None, :]
    )
    power = np.abs(np.fft.rfftn(crop * window)) ** 2

    kz = np.fft.fftfreq(crop.shape[0])[:, None, None]
    ky = np.fft.fftfreq(crop.shape[1])[None, :, None]
    kx = np.fft.rfftfreq(crop.shape[2])[None, None, :]
    in_plane = np.sqrt(kx**2 + kz**2)
    radius = np.sqrt(in_plane**2 + ky**2)
    # An in-plane radius well above zero, so that the angle is defined.
    selected = (radius >= shell[0]) & (radius <= shell[1]) & (in_plane >= shell[0] / 2)
    selected = np.broadcast_to(selected, power.shape)
    alpha = np.degrees(
        np.arctan2(np.broadcast_to(kx, power.shape), np.broadcast_to(kz, power.shape))
    )
    alpha = np.where(alpha > 90.0, alpha - 180.0, alpha)

    values = np.log10(power[selected] + np.finfo(np.float64).tiny)
    angles = alpha[selected]
    edges = np.arange(-90.0, 90.0 + bin_deg / 2, bin_deg)
    index = np.clip(np.digitize(angles, edges) - 1, 0, len(edges) - 2)
    order = np.argsort(index, kind="stable")
    counts = np.bincount(index, minlength=len(edges) - 1)
    groups = np.split(values[order], np.cumsum(counts)[:-1])
    profile = np.array([np.median(group) if group.size else np.nan for group in groups])
    centres = (edges[:-1] + edges[1:]) / 2
    # Next to the kx axis the angle steps by about 1 / (depth * radius), so a thin
    # volume leaves the bins there empty. They are filled from their neighbours,
    # around the circle of lines.
    empty = np.isnan(profile)
    if empty.sum() > len(profile) // 4:
        raise ValueError(
            f"the shell leaves {int(empty.sum())} of {len(profile)} angle bins empty; "
            f"the volume ({crop.shape}) is too small to measure its wedge"
        )
    if empty.any():
        profile[empty] = np.interp(centres[empty], centres[~empty], profile[~empty],
                                   period=180.0)
    shape = (int(crop.shape[0]), int(crop.shape[1]), int(crop.shape[2]))
    return centres, profile - np.median(profile), shape


def measured_bins(centres: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    """Which angle bins a tilt range measured (True) or left in its wedge (False)."""

    shifted_down = centres - 90.0
    shifted_up = centres + 90.0
    return ((shifted_down >= minimum) & (shifted_down <= maximum)) | (
        (shifted_up >= minimum) & (shifted_up <= maximum)
    )


def sector_split(
    centres: np.ndarray, profile: np.ndarray, minimum: float, maximum: float
) -> tuple[float, float]:
    """How well a tilt range's wedge splits the profile: (score, contrast).

    The contrast is the mean of the measured bins minus the mean of the missing
    ones. The score is a two-level least-squares fit's explained variance,
    `n_in n_out / n^2` times the contrast, signed: a plain difference of means
    prefers a narrow sector at the wedge's deepest point, because leakage makes
    the wedge's edges shallower than its middle.
    """

    measured = measured_bins(centres, minimum, maximum)
    inside = profile[measured]
    outside = profile[~measured]
    if not inside.size or not outside.size:
        return 0.0, 0.0
    contrast = float(inside.mean() - outside.mean())
    weight = inside.size * outside.size / profile.size**2
    return float(weight * contrast * abs(contrast)), contrast


def fit_range(centres: np.ndarray, profile: np.ndarray) -> tuple[float, float, float]:
    """The tilt range whose wedge best splits the profile: (min, max, contrast)."""

    candidates = np.arange(-89.0, 89.0 + FIT_STEP_DEG / 2, FIT_STEP_DEG)
    best = (0.0, 0.0, -math.inf, 0.0)
    for minimum in candidates:
        allowed = (candidates >= minimum + MIN_RANGE_DEG) & (candidates <= minimum + MAX_RANGE_DEG)
        for maximum in candidates[allowed]:
            score, contrast = sector_split(centres, profile, float(minimum), float(maximum))
            if score > best[2]:
                best = (float(minimum), float(maximum), score, contrast)
    return best[0], best[1], best[3]


def measure_wedge(
    volume: Any,
    tilt_angles_deg: Sequence[float],
    *,
    shell: tuple[float, float] = DEFAULT_SHELL,
    no_wedge_below: float = 0.15,
    side_tolerance_deg: float = 3.0,
    side_margin_deg: float = 2.0,
) -> WedgeMeasurement:
    """The wedge in a canonical volume, set against the angles it was declared with.

    The fitted range's centre is the measurement of the side: the declared
    range's centre and its negative, a mirrored volume's, are `2 |centre|` apart.

    Verdicts:
    - `no wedge`: the best split's contrast is below `no_wedge_below` (log10
      power): a filled wedge, or a volume too small or too smooth to tell;
    - `side not testable`: the declared range is centred within
      `side_tolerance_deg` of zero, so a mirror shows the same wedge;
    - `agrees` / `mirrored`: the fitted centre is nearer the declared centre, or
      its negative, by at least `side_margin_deg`;
    - `inconclusive`: neither is.
    """

    if len(tilt_angles_deg) < 2:
        raise ValueError("a wedge needs at least two tilt angles")
    declared_min = float(min(tilt_angles_deg))
    declared_max = float(max(tilt_angles_deg))
    centres, profile, crop_shape = angular_profile(volume, shell=shell)
    fitted_min, fitted_max, fitted_contrast = fit_range(centres, profile)
    _, declared = sector_split(centres, profile, declared_min, declared_max)
    _, mirrored = sector_split(centres, profile, -declared_max, -declared_min)

    centre = (declared_min + declared_max) / 2
    fitted_centre = (fitted_min + fitted_max) / 2
    to_declared = abs(fitted_centre - centre)
    to_mirrored = abs(fitted_centre + centre)
    fitted = f"fitted {fitted_min:g}..{fitted_max:g} (centre {fitted_centre:+.1f})"
    verdict: Verdict
    if fitted_contrast < no_wedge_below:
        verdict = "no wedge"
        reason = (
            f"no sector of the transform is weaker than the rest by {no_wedge_below} "
            f"log10 power (best {fitted_contrast:.3f}): the wedge looks filled"
        )
    elif abs(centre) < side_tolerance_deg:
        verdict = "side not testable"
        reason = (
            f"the declared range {declared_min:g}..{declared_max:g} is centred within "
            f"{side_tolerance_deg:g} deg of zero, so its mirror shows the same wedge; {fitted}"
        )
    elif to_declared + side_margin_deg <= to_mirrored:
        verdict = "agrees"
        reason = (
            f"{fitted}, {to_declared:.1f} deg from the declared range's centre "
            f"({centre:+.1f}) and {to_mirrored:.1f} from its mirror's"
        )
    elif to_mirrored + side_margin_deg <= to_declared:
        verdict = "mirrored"
        reason = (
            f"{fitted}, {to_mirrored:.1f} deg from the centre of the declared range "
            f"negated ({-centre:+.1f}) and {to_declared:.1f} from the declared one's "
            f"({centre:+.1f}): the volume is the mirror of what the angles describe"
        )
    else:
        verdict = "inconclusive"
        reason = (
            f"{fitted}, about as far from the declared range's centre ({centre:+.1f}) "
            f"as from its mirror's"
        )
    return WedgeMeasurement(
        bin_centres_deg=[float(value) for value in centres],
        profile=[float(value) for value in profile],
        fitted_min_deg=fitted_min,
        fitted_max_deg=fitted_max,
        fitted_contrast=fitted_contrast,
        declared_min_deg=declared_min,
        declared_max_deg=declared_max,
        declared_contrast=declared,
        mirrored_contrast=mirrored,
        verdict=verdict,
        reason=reason,
        shell=shell,
        crop_shape=crop_shape,
    )


__all__ = [
    "WedgeMeasurement",
    "angular_profile",
    "fit_range",
    "measure_wedge",
    "measured_bins",
    "sector_split",
    "tukey",
]
