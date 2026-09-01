"""What resolution the data can actually support, from what was measured.

The first of the derivations roadmap section 15 groups together: a number a
person currently guesses, derivable from something the pipeline already
measures, and silent when the guess is wrong. Here the guess is the
template-matching low-pass, the measurement is the fine-alignment residual, and
the silence is a flat score volume that returns almost no candidates with
`status: pass` — which is exactly what produced 2 candidates from a requested
2000 on TS_0005.
"""

from __future__ import annotations

import math

ALIGNMENT_BLUR_SAFETY = 1.0
"""How much coarser than the alignment bound to place the limit, as a factor.

One by default, meaning the bound is used as-is. It is a parameter rather than a
constant because the bound is optimistic in a known direction and someone may
want headroom — see `low_pass_from_alignment_residual`.
"""


def low_pass_from_alignment_residual(
    residual_nm: float, *, safety: float = ALIGNMENT_BLUR_SAFETY
) -> float:
    """The finest resolution an alignment with this residual can support, in angstrom.

    An rms misalignment of `delta` blurs the reconstruction, attenuating spatial
    frequency `k` by `exp(-2 pi^2 delta^2 k^2)`. That reaches `1/e` at
    `d = pi * delta * sqrt(2)`, and beyond it the signal the matcher would be
    correlating against has already been smeared away by the alignment.

    **This is a lower bound on the low-pass, not an estimate of it.** It accounts
    for alignment blur alone; dose, lamella thickness and the missing wedge all
    degrade further, so the true usable resolution is coarser than this. Erring
    coarser is safe — the matcher simply uses less of the band — while erring
    finer means correlating against frequencies that hold no signal, which is how
    a score volume goes flat. `safety` multiplies the result for anyone who wants
    that headroom explicit rather than implied.

    Measured on the five EMPIAR-11538 series, this gives 31 to 41 angstrom, and
    the 35 that had been chosen by hand sits in the middle of that — which is the
    argument for deriving it per series rather than configuring one value.
    """

    if not math.isfinite(residual_nm) or residual_nm <= 0.0:
        raise ValueError(f"residual must be finite and positive, got {residual_nm}")
    if not math.isfinite(safety) or safety <= 0.0:
        raise ValueError(f"safety factor must be finite and positive, got {safety}")
    return math.pi * (residual_nm * 10.0) * math.sqrt(2.0) * safety


def alignment_residual_from_low_pass(low_pass_angstrom: float) -> float:
    """The residual a given low-pass corresponds to, in nanometres.

    The inverse of the above, for reading a configured value back into the units
    the alignment reports — so that "is 35 angstrom right for this series?" can be
    answered by comparing one number against one number.
    """

    if not math.isfinite(low_pass_angstrom) or low_pass_angstrom <= 0.0:
        raise ValueError(f"low-pass must be finite and positive, got {low_pass_angstrom}")
    return low_pass_angstrom / (math.pi * math.sqrt(2.0)) / 10.0
