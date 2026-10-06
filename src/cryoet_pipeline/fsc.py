from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import torch
import zarr

from cryoet_pipeline.models import FscQc, QcStatus
from cryoet_pipeline.particles import read_particle_star
from cryoet_pipeline.rotations import (
    center_particle_volume,
    relion_euler_to_matrix,
    resample_volume_zyx,
)


def _fourier_shell_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Call torch-fourier-shell-correlation, imported only when actually used.

    The CLI imports every module up front, so a package needed by one command
    must not be imported at module scope: otherwise a missing optional
    dependency breaks unrelated commands.

    It gets full transforms. Its `fourier_shell_correlation` sums the `rfftn`
    half grid without Hermitian weights, so the kx = 0 and kx = N/2 planes,
    which hold both members of their conjugate pairs, counted twice against
    every other coefficient. Whatever those planes share then lifts the curve:
    on CZII half-maps it added up to 0.12 near Nyquist unmasked, and 0.09 on
    the masked Nyquist shell of Dynamo's halves.
    """

    try:
        from torch_fourier_shell_correlation import fourier_correlation
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise ImportError(
            "compute-fsc needs the torch-fourier-shell-correlation package; "
            'install it with: pip install -e ".[gpu]"'
        ) from exc
    result: torch.Tensor = fourier_correlation(
        torch.fft.fftn(a), torch.fft.fftn(b), tuple(a.shape), rfft=False
    )
    return result


@dataclass(frozen=True)
class FscCurve:
    """A Fourier shell correlation curve and its threshold-crossing resolution."""

    resolution_angstrom: list[float]
    correlation: list[float]
    threshold: float
    crossing_resolution_angstrom: float | None
    reaches_threshold: bool = True
    """Whether the curve is at or above the threshold on any shell at all.

    False means there is no resolution to report — not a coarse one. The old code
    had no way to say that: a curve entirely below the threshold returned its
    first shell, which is a number and looks like an answer.
    """
    dips_before_crossing: bool = False
    """Whether the curve fell below the threshold and came back before the end.

    Diagnostic of a coarse-scale disagreement between the two volumes rather
    than of a resolution limit.
    """


def compute_fsc_curve(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    pixel_spacing_angstrom: float,
    threshold: float,
) -> FscCurve:
    """Compute the Fourier shell correlation curve between two same-shaped
    real-space volumes, and the shell just beyond the last one at or above
    `threshold` — the curve's final descent through it.

    The DC shell (infinite resolution, trivially correlation 1.0) is omitted
    from the returned curve.

    The previous wording said "the coarsest (lowest-Angstrom) shell at which the
    curve first drops below threshold", which was self-contradictory as well as
    describing the wrong scan: coarsest means HIGHEST angstroms, and the shell
    reported is now the finest one at which agreement ends rather than the first
    place the curve dips.
    """

    if a.shape != b.shape:
        raise ValueError(f"volumes must share shape, got {tuple(a.shape)} and {tuple(b.shape)}")
    if len(a.shape) != 3:
        raise ValueError(f"expected 3D volumes, got shape {tuple(a.shape)}")

    box_size = a.shape[0]
    curve = _fourier_shell_correlation(a.to(torch.float32), b.to(torch.float32))
    all_shells = [float(value) for value in curve.detach().cpu().numpy()]
    if len(all_shells) < 2:
        raise ValueError(f"expected at least one non-DC shell, got box size {box_size}")

    correlation = all_shells[1:]
    resolution_angstrom = [
        (box_size * pixel_spacing_angstrom) / shell_index
        for shell_index in range(1, len(all_shells))
    ]

    # The crossing is the FINAL descent through the threshold, not the first dip
    # through it.
    #
    # The first-dip version read the coarsest shells, which are the least
    # trustworthy part of an FSC: shell 1 of a 48 box holds a handful of Fourier
    # coefficients, so its correlation is estimated from almost nothing and
    # wanders. On four class averages of one real dataset it wandered across the
    # threshold and the reported resolutions came back 469.63, 469.63, 234.82 and
    # 156.54 A — exactly shells 1, 1, 2 and 3 — while all four curves in fact
    # peaked together near 78 A and fell to zero together near 32 A. Every one of
    # those numbers was the position of a low-frequency wiggle, reported as a
    # resolution with `status: pass`.
    #
    # Scanning from the fine end for the last shell at or above the threshold
    # ignores those wiggles and answers the question a resolution is asked for:
    # beyond which shell does agreement stop. A curve that never reaches the
    # threshold at all has no crossing to report, which the caller turns into a
    # warning rather than a number.
    last_above: int | None = None
    for index in range(len(correlation) - 1, -1, -1):
        if correlation[index] >= threshold:
            last_above = index
            break

    crossing_resolution_angstrom = None
    if last_above is not None and last_above + 1 < len(correlation):
        crossing_resolution_angstrom = resolution_angstrom[last_above + 1]

    # Recorded because it is diagnostic rather than merely untidy: a curve that
    # dips below the threshold and comes back has something wrong at coarse
    # scale — most often two maps whose overall envelopes differ, such as a
    # compact template zero-padded into a larger particle box.
    dips_before_crossing = last_above is not None and any(
        value < threshold for value in correlation[:last_above]
    )

    return FscCurve(
        resolution_angstrom=resolution_angstrom,
        correlation=correlation,
        threshold=threshold,
        crossing_resolution_angstrom=crossing_resolution_angstrom,
        reaches_threshold=last_above is not None,
        dips_before_crossing=dips_before_crossing,
    )


# Two independent maps correlate in shell k by about +-sqrt(2 / N_k), N_k being the
# voxels in the shell (van Heel & Schatz 2005, the basis of their 3-sigma curve).
# A rise is flagged when it clears 5 such sigma of the dip and the peak together,
# and 0.1 FSC. Five rather than three because the dip and the peak are picked out
# of a dozen shells after looking: on CZII at box 48 a shuffled-pose null reached
# about 4 by chance, while Dynamo's averaging of real half-maps rose by about 16.
RISE_MIN_SIGMA = 5.0
RISE_MIN_FSC = 0.1


@dataclass(frozen=True)
class NyquistRise:
    """Agreement that comes back after a dip in the outer half of the curve."""

    dip_resolution_angstrom: float
    dip_correlation: float
    peak_resolution_angstrom: float
    peak_correlation: float
    sigma: float
    """The rise in units of the two shells' noise, sqrt(2 / N_k) each."""
    crossing_before_dip_angstrom: float | None
    """Where the curve last fell through the threshold before the dip, if it did:
    the resolution its final descent would report without the rise."""


def _shell_voxel_counts(box_size: int) -> np.ndarray:
    frequency = np.fft.fftfreq(box_size) * box_size
    z, y, x = np.meshgrid(frequency, frequency, frequency, indexing="ij")
    return np.bincount(np.rint(np.sqrt(x * x + y * y + z * z)).astype(int).ravel())


def rise_toward_nyquist(curve: FscCurve, box_size: int) -> NyquistRise | None:
    """The deepest dip in the curve's outer half and the strongest return after it,
    when that return is beyond chance; otherwise None.

    Real agreement fades toward Nyquist. When it comes back there, the two maps
    more often share something other than the structure: Dynamo's averaging does
    this to masked half-maps on CZII (0.27 at 22.9 A back up to 0.69 at 20.0 A),
    and this pipeline's former compensated reducer did it unmasked. It can also be
    signal past an uncorrected CTF zero. Either way the final-descent crossing
    then lands beyond the dip, or nowhere, so the dip is worth reporting.
    """

    correlation = np.asarray(curve.correlation, dtype=float)
    count = len(correlation)
    if count < 4:
        return None
    voxels = _shell_voxel_counts(box_size)[1 : count + 1]
    noise = np.sqrt(2.0 / np.maximum(voxels, 1))
    start = count // 2 - 1  # the outer half: shells count/2 .. count
    dip = start + int(np.argmin(correlation[start:-1]))
    peak = dip + 1 + int(np.argmax(correlation[dip + 1 :]))
    rise = float(correlation[peak] - correlation[dip])
    sigma = rise / float(np.hypot(noise[dip], noise[peak]))
    if rise < RISE_MIN_FSC or sigma < RISE_MIN_SIGMA:
        return None
    crossing = None
    above = np.flatnonzero(correlation[:dip] >= curve.threshold)
    if above.size and correlation[above[-1] + 1] < curve.threshold:
        crossing = curve.resolution_angstrom[above[-1] + 1]
    return NyquistRise(
        dip_resolution_angstrom=curve.resolution_angstrom[dip],
        dip_correlation=float(correlation[dip]),
        peak_resolution_angstrom=curve.resolution_angstrom[peak],
        peak_correlation=float(correlation[peak]),
        sigma=sigma,
        crossing_before_dip_angstrom=crossing,
    )


def build_half_set_average(
    *,
    particle_stack_path: Path,
    metadata_path: Path,
    random_subset: int,
    device: torch.device,
) -> torch.Tensor:
    """Average one half-set's particles using each particle's already-refined
    orientation/shift, as written by `average-particles`.

    `ParticleRecord.index` is a position in the array at hand, never a particle's
    name, so the STAR and the stack have to be the pair that were written for one
    another. Extraction makes that true by construction — one row per slice — and
    cross-tilt-series averaging broke it: its refined STAR carries combined
    indices spanning every stack it averaged, and this command takes exactly one
    stack. The count comparison below is what stands between that and a
    confidently wrong resolution.
    """

    all_records = read_particle_star(metadata_path)
    particle_stack = cast(Any, zarr.open(particle_stack_path, mode="r"))
    stack_count = int(particle_stack.shape[0])
    if len(all_records) != stack_count:
        raise ValueError(
            f"{metadata_path} describes {len(all_records)} particles but "
            f"{particle_stack_path} holds {stack_count}. A refined STAR from a "
            "cross-tilt-series average indexes all of its stacks at once and cannot "
            "be paired with a single stack, so this half-set FSC would silently "
            "average whichever particles those indices happened to land on"
        )
    records = [record for record in all_records if record.random_subset == random_subset]
    if not records:
        raise ValueError(f"no particles with random_subset={random_subset} in {metadata_path}")
    volumes = []
    for record in records:
        raw = center_particle_volume(
            torch.as_tensor(np.asarray(particle_stack[record.index]), device=device)
        )
        matrix = relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg)
        shift = np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px])
        volumes.append(resample_volume_zyx(raw, matrix, shift, device=device))
    return torch.stack(volumes).mean(dim=0)


def compute_and_write_fsc_report(
    volume_a: torch.Tensor,
    volume_b: torch.Tensor,
    *,
    mode: str,
    pixel_spacing_angstrom: float,
    threshold: float,
    out: Path,
) -> FscQc:
    """Compute the FSC curve, decide a QC status, and write the report JSON."""

    curve = compute_fsc_curve(
        volume_a, volume_b, pixel_spacing_angstrom=pixel_spacing_angstrom, threshold=threshold
    )

    warnings: list[str] = []
    status = QcStatus.PASS
    if not curve.reaches_threshold:
        status = QcStatus.WARNING
        warnings.append(
            f"correlation never reached threshold {threshold:.3f} on any shell, so "
            "there is no resolution to report. The two volumes do not agree at any "
            "scale sampled here — check that they are the maps intended, at the same "
            "pixel size and the same handedness, before reading anything else"
        )
    elif curve.crossing_resolution_angstrom is None:
        status = QcStatus.WARNING
        warnings.append(
            f"correlation never dropped below threshold {threshold:.3f} across the "
            "sampled shells"
        )
    if curve.dips_before_crossing:
        # Not a failure: it says which part of the curve to distrust. The
        # coarsest shells hold few Fourier coefficients — shell 1 of a 48 box
        # holds a handful — so they wander, and two maps whose overall envelopes
        # differ disagree there for reasons that are not about resolution.
        if status is QcStatus.PASS:
            status = QcStatus.WARNING
        warnings.append(
            f"the curve fell below {threshold:.3f} and came back before its final "
            "descent. The coarsest shells hold few coefficients and wander, and a "
            "compact map padded into a larger box disagrees there by construction, "
            "so read the reported resolution as the final descent and not the "
            "low-frequency part of the curve"
        )
    rise = rise_toward_nyquist(curve, int(volume_a.shape[0]))
    if rise is not None:
        status = QcStatus.WARNING
        before = (
            f"; before it the curve fell through {threshold:.3f} at "
            f"{rise.crossing_before_dip_angstrom:.1f} A"
            if rise.crossing_before_dip_angstrom is not None
            else ""
        )
        warnings.append(
            f"the curve rises again toward Nyquist, from {rise.dip_correlation:.2f} at "
            f"{rise.dip_resolution_angstrom:.1f} A to {rise.peak_correlation:.2f} at "
            f"{rise.peak_resolution_angstrom:.1f} A ({rise.sigma:.0f} sigma of shell "
            "noise). Agreement that returns this close to Nyquist is more often shared "
            "processing than structure (Dynamo's averaging does it to masked half-maps), "
            "though it can be signal past an uncorrected CTF zero. The final descent "
            f"then lands beyond the dip{before}"
        )

    report = FscQc(
        mode=mode,
        box_size_px=int(volume_a.shape[0]),
        pixel_spacing_angstrom=pixel_spacing_angstrom,
        threshold=threshold,
        crossing_resolution_angstrom=curve.crossing_resolution_angstrom,
        resolution_angstrom=curve.resolution_angstrom,
        correlation=curve.correlation,
        rise_toward_nyquist=None if rise is None else asdict(rise),
        status=status,
        warnings=warnings,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return report


def plot_fsc_curve(
    resolution_angstrom: list[float],
    correlation: list[float],
    *,
    threshold: float,
    crossing_resolution_angstrom: float | None,
    out: Path,
    title: str | None = None,
) -> None:
    """Save a Fourier shell correlation plot: FSC (y) vs spatial frequency,
    1/angstrom (x).
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spatial_frequency = [1.0 / resolution for resolution in resolution_angstrom]

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.plot(spatial_frequency, correlation, marker="o", markersize=3)
    axes.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    if crossing_resolution_angstrom is not None:
        axes.axvline(1.0 / crossing_resolution_angstrom, color="red", linestyle=":", linewidth=1)
    axes.set_xlabel("Spatial frequency (1/A)")
    axes.set_ylabel("Fourier shell correlation")
    axes.set_ylim(-0.05, 1.05)
    if title:
        axes.set_title(title)
    figure.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)


def load_mask(path: Path, *, expected_shape: tuple[int, ...]) -> torch.Tensor:
    """Load a real-space mask MRC and validate it matches the volume shape."""

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
    if data.shape != expected_shape:
        raise ValueError(
            f"mask shape {data.shape} does not match volume shape {expected_shape}: {path}"
        )
    return torch.as_tensor(data)


def load_mrc_volume(path: Path) -> tuple[torch.Tensor, float]:
    """Load an MRC volume and its voxel size (angstrom, assumed isotropic)."""

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}: {path}")
    if voxel_size <= 0.0:
        raise ValueError(f"MRC file has no usable voxel size: {path}")
    return torch.as_tensor(data), voxel_size
