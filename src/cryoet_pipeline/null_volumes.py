"""A tomogram's spectrum without its structure, for controls that need noise of the right kind.

Template matching scores every box of a tomogram, and the best-scoring boxes of
pure noise are not random boxes: each is the noise that happens to look most
like the template, so an average of them looks like the template too. A control
for pytom's false picks has to be selected the same way, which needs a volume
that is this tomogram in every respect except that nothing is in it.

Replacing every Fourier phase with a random one gives that. The amplitude at
every spatial frequency stays as it was, so the power spectrum, the missing
wedge, the CTF zeros, the reconstruction's filtering and the variance are the
original's; with random phases nothing is anywhere. The phases are white
noise's, so they have the conjugate symmetry of a real volume and the result is
real.

What it does not keep is where the variance was. A tomogram's lamella and its
vacuum differ in contrast and in spectrum, and the copy spreads their average
evenly over the whole volume.
"""

from __future__ import annotations

import numpy as np
import scipy.fft  # type: ignore[import-untyped]


def phase_randomised(
    volume: np.ndarray, rng: np.random.Generator, *, workers: int | None = None
) -> np.ndarray:
    """`volume` with every Fourier phase replaced by white noise's; amplitudes and mean kept.

    The zero frequency keeps its own value, so the mean is the original's rather
    than the original's with a random sign. Single precision throughout (scipy
    keeps float32 as complex64), so the peak is about five times the volume:
    under 2 GB for a 512 x 512 x 320 bin8 tomogram.
    """

    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}")
    spectrum = scipy.fft.rfftn(data, workers=workers)
    mean_term = spectrum[0, 0, 0]
    amplitude = np.abs(spectrum)
    del spectrum
    phases = scipy.fft.rfftn(rng.standard_normal(data.shape, dtype=np.float32), workers=workers)
    modulus = np.abs(phases)
    # A coefficient of exactly zero has no phase to give; it keeps its amplitude, real.
    zero = modulus == 0
    modulus[zero] = 1.0
    phases /= modulus
    phases[zero] = 1.0
    phases *= amplitude
    phases[0, 0, 0] = mean_term
    result = scipy.fft.irfftn(phases, s=data.shape, workers=workers)
    return np.asarray(result, dtype=np.float32)
