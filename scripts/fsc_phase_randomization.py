#!/usr/bin/env python3
"""How much of a masked FSC the mask itself makes, measured by phase randomisation.

    python3 scripts/fsc_phase_randomization.py A.mrc B.mrc --mask mask.mrc \\
        --randomize-beyond-angstrom 30.9 21.6

Masking multiplies both maps, which in Fourier space is a convolution: whatever
the two maps share at coarse scale is spread into finer shells, where it reads as
agreement. A sharp edge spreads it far. On the EMPIAR-10164 Gag averages of TS_01
and TS_43, made apart, a binary 250 A sphere held the mean FSC between 15 and
10 A at 0.26, against 0.025 unmasked. A soft edge spreads it less, but by how
much is not known in advance: the same sphere with a 6-voxel cosine edge gave
0.068 there.

The control (Chen et al. 2013, Ultramicroscopy 135:24-35) measures it on the maps
themselves. Beyond a cutoff, each map's Fourier phases are replaced by random
ones, independently for the two maps, and the amplitudes are kept. Past the
cutoff the two maps then share nothing, so whatever the masked FSC of the
randomised pair shows there is the mask's doing. The corrected curve is

    FSC_true = (FSC_masked - FSC_random) / (1 - FSC_random)

on the shells past the cutoff, and the masked curve itself before them.

Shells right next to the cutoff are left uncorrected (`--margin-shells`). There
the randomised pair still borrows real agreement from the intact shells just
inside the cutoff, through the same convolution, which a shell further out
does not; subtracting it there would take away more than the mask added.

Each cutoff is randomised `--seeds` times. The mean over seeds is what is
subtracted. The spread over seeds is the noise of a masked FSC between maps
that share nothing, and so the yardstick for whether a corrected value differs
from zero: under a mask, neighbouring Fourier coefficients are correlated and a
shell holds fewer independent samples than its voxel count.

Try more than one cutoff. The correction only sees what the mask spreads from
shells coarser than the cutoff; agreement the two maps share between the cutoff
and a shell is spread into it as well, and stays. A soft edge spreads only to
neighbouring shells, so that part is small. A sharp edge spreads far: in a
synthetic pair sharing nothing finer than 25 A, under a binary sphere, the mean
FSC finer than 16 A was 0.27 masked, 0.17 corrected from a 30.9 A cutoff and
0.002 from 21.6 A. Under a 6-voxel soft edge the same pair gave 0.006 masked
and 0.003 and -0.003 corrected. Where the corrected curve does not move with
the cutoff, the correction is not what decides the answer.

Curves come from `cryoet_pipeline.fsc.compute_fsc_curve`, the function
`compute-fsc` uses, so its shells and numbers match the reports. Crossings are
read off the curve by interpolating in frequency, as `fsc_table.py` does. When
the curve falls through a level more than once, the first and the last fall are
both printed.

Disposable, like the rest of `scripts/`. Its replacement is the same correction
inside `compute-fsc` and the averaging QC.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cryoet_pipeline.fsc import compute_fsc_curve, load_mask, load_mrc_volume


def shell_radius(shape: tuple[int, ...]) -> np.ndarray:
    """Distance of every `rfftn` coefficient from the origin, in shells."""

    box = shape[0]
    z = np.fft.fftfreq(box) * box
    x = np.fft.rfftfreq(box) * box
    return np.sqrt(z[:, None, None] ** 2 + z[None, :, None] ** 2 + x[None, None, :] ** 2)


def randomize_phases_beyond(
    volume: np.ndarray, radius_shells: float, rng: np.random.Generator
) -> np.ndarray:
    """`volume` with every Fourier phase beyond `radius_shells` replaced by a random one.

    The phases are those of white noise, so they have the conjugate symmetry of a
    real volume and the result stays real. Amplitudes are untouched everywhere.
    """

    spectrum = np.fft.rfftn(volume)
    noise = np.fft.rfftn(rng.standard_normal(volume.shape))
    beyond = shell_radius(volume.shape) > radius_shells
    spectrum[beyond] = np.abs(spectrum[beyond]) * np.exp(1j * np.angle(noise[beyond]))
    return np.fft.irfftn(spectrum, s=volume.shape, axes=(0, 1, 2))


def fsc(a: np.ndarray, b: np.ndarray, voxel: float) -> tuple[list[float], list[float]]:
    curve = compute_fsc_curve(
        torch.as_tensor(a, dtype=torch.float32),
        torch.as_tensor(b, dtype=torch.float32),
        pixel_spacing_angstrom=voxel,
        threshold=0.143,
    )
    return curve.resolution_angstrom, curve.correlation


def first_corrected_shell(radius_shells: float, margin: int) -> int:
    """The first shell (1-based, as the curves count them) to correct.

    `compute_fsc_curve` spreads each coefficient over the two nearest shells, so
    shell k draws on radii k-1 to k+1 and is wholly randomised once k-1 is past
    the cutoff.
    """

    # 64 * 5.4 / 21.6 comes out a hair above 16, and further above it with the
    # float32 voxel size of an MRC header (5.4000001).
    return math.ceil(radius_shells - 1e-6) + 1 + margin


def correct(masked: np.ndarray, random: np.ndarray, start: int) -> np.ndarray:
    """(t - n) / (1 - n) from 1-based shell `start` on; the masked curve before it."""

    corrected = masked.copy()
    tail = slice(start - 1, None)
    corrected[tail] = (masked[tail] - random[tail]) / (1.0 - random[tail])
    return corrected


def falls(resolutions: list[float], values: list[float], level: float) -> list[float]:
    """Every place the curve falls below `level`, interpolated in spatial frequency."""

    found = []
    for index in range(1, len(values)):
        if values[index] < level <= values[index - 1]:
            coarse, fine = 1.0 / resolutions[index - 1], 1.0 / resolutions[index]
            fraction = (values[index - 1] - level) / (values[index - 1] - values[index])
            found.append(1.0 / (coarse + fraction * (fine - coarse)))
    return found


def crossing_text(resolutions: list[float], values: list[float], level: float) -> str:
    found = falls(resolutions, values, level)
    if not found:
        return "none"
    if len(found) == 1 or values[-1] >= level:
        return f"{found[0]:.1f} A"
    return f"{found[0]:.1f}/{found[-1]:.1f} A"


def band_mean(values: np.ndarray, in_band: np.ndarray) -> float:
    return float(values[in_band].mean())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("map_a", type=Path)
    parser.add_argument("map_b", type=Path)
    parser.add_argument("--mask", type=Path, required=True, help="Real-space mask MRC.")
    parser.add_argument(
        "--randomize-beyond-angstrom",
        type=float,
        nargs="+",
        required=True,
        help="Cutoffs: phases are randomised at resolutions finer than each.",
    )
    parser.add_argument("--seeds", type=int, default=20, help="Randomisations per cutoff.")
    parser.add_argument(
        "--margin-shells",
        type=int,
        default=2,
        help="Shells past the first wholly randomised one that stay uncorrected.",
    )
    parser.add_argument("--levels", type=float, nargs="+", default=[0.5, 0.143])
    parser.add_argument(
        "--band-coarse-angstrom",
        type=float,
        default=None,
        help="Coarse end of the band mean; default 1.5 x the Nyquist resolution.",
    )
    parser.add_argument(
        "--band-fine-angstrom", type=float, default=None, help="Fine end; default Nyquist."
    )
    parser.add_argument("--shells", action="store_true", help="Print the curves shell by shell.")
    parser.add_argument("--out", type=Path, default=None, help="Write everything as JSON.")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2, or there is no spread to report")

    a_tensor, voxel = load_mrc_volume(args.map_a)
    b_tensor, voxel_b = load_mrc_volume(args.map_b)
    if a_tensor.shape != b_tensor.shape:
        parser.error(f"maps differ in shape: {tuple(a_tensor.shape)} and {tuple(b_tensor.shape)}")
    if len(set(a_tensor.shape)) != 1:
        parser.error(f"expected cubic maps, got {tuple(a_tensor.shape)}")
    if abs(voxel - voxel_b) > 1e-3:
        parser.error(f"maps differ in voxel size: {voxel} and {voxel_b} A")
    a = a_tensor.numpy().astype(np.float64)
    b = b_tensor.numpy().astype(np.float64)
    mask = load_mask(args.mask, expected_shape=a.shape).numpy().astype(np.float64)
    box = a.shape[0]
    nyquist = 2.0 * voxel

    resolutions, unmasked = fsc(a, b, voxel)
    _, masked_list = fsc(a * mask, b * mask, voxel)
    masked = np.asarray(masked_list)
    shell_resolution = np.asarray(resolutions)
    coarse = args.band_coarse_angstrom or 1.5 * nyquist
    fine = args.band_fine_angstrom or nyquist
    in_band = (shell_resolution <= coarse) & (shell_resolution >= fine - 1e-6)
    if not in_band.any():
        parser.error(f"no shells between {coarse:.1f} and {fine:.1f} A")

    cutoffs = []
    for index, cutoff in enumerate(args.randomize_beyond_angstrom):
        radius = box * voxel / cutoff
        start = first_corrected_shell(radius, args.margin_shells)
        if cutoff <= nyquist or start > len(resolutions):
            parser.error(
                f"cutoff {cutoff} A leaves no shell to correct at {voxel} A/voxel "
                f"(Nyquist {nyquist:.1f} A, margin {args.margin_shells} shells)"
            )
        runs = []
        for seed in range(args.seeds):
            rng = np.random.default_rng([index, seed])
            random_a = randomize_phases_beyond(a, radius, rng)
            random_b = randomize_phases_beyond(b, radius, rng)
            runs.append(fsc(random_a * mask, random_b * mask, voxel)[1])
        curves = np.asarray(runs)
        mean = curves.mean(axis=0)
        cutoffs.append(
            {
                "beyond_angstrom": cutoff,
                "corrected_from_shell": start,
                "corrected_from_angstrom": resolutions[start - 1],
                "random_mean": mean,
                "random_sd": curves.std(axis=0, ddof=1),
                "random_band_means": curves[:, in_band].mean(axis=1),
                "corrected": correct(masked, mean, start),
            }
        )

    print(f"A     {args.map_a}")
    print(f"B     {args.map_b}")
    print(f"mask  {args.mask}  (mean of mask^2 over the box {float((mask**2).mean()):.3f})")
    print(
        f"box {box}, {voxel:g} A/voxel, Nyquist {nyquist:.1f} A; "
        f"{args.seeds} randomisations per cutoff\n"
    )
    level_header = "".join(f" {f'{level:g} at':>13}" for level in args.levels)
    print(f"{'curve':<40}{level_header} {f'band {coarse:.0f}-{fine:.0f} A':>16}")
    print("-" * (57 + 14 * len(args.levels)))

    def row(name: str, values: np.ndarray | list[float], band: str) -> None:
        crossings = "".join(
            f" {crossing_text(resolutions, list(values), level):>13}" for level in args.levels
        )
        print(f"{name:<40}{crossings} {band:>16}")

    row("unmasked", unmasked, f"{band_mean(np.asarray(unmasked), in_band):.3f}")
    row("masked", masked, f"{band_mean(masked, in_band):.3f}")
    for one in cutoffs:
        spread = one["random_band_means"]
        label = f"  random phases beyond {one['beyond_angstrom']:g} A"
        band = f"{spread.mean():.3f} +- {spread.std(ddof=1):.3f}"
        print(f"{label:<40}{'':>{14 * len(args.levels)}} {band:>16}")
        row(
            f"  corrected from {one['corrected_from_angstrom']:.1f} A",
            one["corrected"],
            f"{band_mean(one['corrected'], in_band):.3f}",
        )
    print(
        "\nThe random rows are the masked FSC of maps that share nothing past the cutoff:"
        "\nits mean is what the mask adds, its +- the noise of the band mean itself."
    )

    if args.shells:
        header = f"\n  {'shell':>5} {'A':>7} {'unmasked':>9} {'masked':>8}"
        for one in cutoffs:
            label = f"{one['beyond_angstrom']:g}"
            header += f" {'rnd>' + label:>10} {'sd':>6} {'corr>' + label:>10}"
        print(header)
        for index, resolution in enumerate(resolutions):
            line = (
                f"  {index + 1:>5} {resolution:>7.1f} "
                f"{unmasked[index]:>9.3f} {masked[index]:>8.3f}"
            )
            for one in cutoffs:
                corrected = (
                    f"{one['corrected'][index]:>10.3f}"
                    if index + 1 >= one["corrected_from_shell"]
                    else f"{'':>10}"
                )
                line += (
                    f" {one['random_mean'][index]:>10.3f} {one['random_sd'][index]:>6.3f}"
                    f" {corrected}"
                )
            print(line)

    if args.out is not None:
        payload = {
            "map_a": str(args.map_a),
            "map_b": str(args.map_b),
            "mask": str(args.mask),
            "box_size_px": box,
            "pixel_spacing_angstrom": voxel,
            "seeds": args.seeds,
            "margin_shells": args.margin_shells,
            "resolution_angstrom": resolutions,
            "unmasked": unmasked,
            "masked": masked.tolist(),
            "randomized": [
                {
                    "beyond_angstrom": one["beyond_angstrom"],
                    "corrected_from_shell": one["corrected_from_shell"],
                    "corrected_from_angstrom": one["corrected_from_angstrom"],
                    "random_mean": one["random_mean"].tolist(),
                    "random_sd": one["random_sd"].tolist(),
                    "corrected": one["corrected"].tolist(),
                }
                for one in cutoffs
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
