"""scripts/fsc_phase_randomization.py: what a mask adds to an FSC, measured on the maps."""

import json
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fsc_phase_randomization as fpr  # noqa: E402

BOX, VOXEL = 48, 5.4


def _field(rng: np.random.Generator, finest_angstrom: float, scale: float) -> np.ndarray:
    """Random density with nothing finer than `finest_angstrom`."""
    spectrum = np.fft.rfftn(rng.standard_normal((BOX,) * 3))
    spectrum[fpr.shell_radius((BOX,) * 3) > BOX * VOXEL / finest_angstrom] = 0
    volume = np.fft.irfftn(spectrum, s=(BOX,) * 3, axes=(0, 1, 2))
    return scale * volume / volume.std()


def _pair(rng: np.random.Generator, shared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        shared + rng.standard_normal((BOX,) * 3),
        shared + rng.standard_normal((BOX,) * 3),
    )


def _sphere(edge_px: float) -> np.ndarray:
    """A sphere of radius 18 voxels; binary at edge 0, else a cosine edge inward."""
    offset = np.arange(BOX) - (BOX - 1) / 2
    z, y, x = np.meshgrid(offset, offset, offset, indexing="ij")
    r = np.sqrt(x * x + y * y + z * z)
    outer = 18.0
    if edge_px == 0:
        return (r <= outer).astype(float)
    mask = 0.5 * (1 + np.cos(np.pi * (r - (outer - edge_px)) / edge_px))
    mask[r <= outer - edge_px] = 1.0
    mask[r >= outer] = 0.0
    return mask


def _masked_and_corrected(a, b, mask, cutoff_angstrom, seeds=8):
    radius = BOX * VOXEL / cutoff_angstrom
    masked = np.asarray(fpr.fsc(a * mask, b * mask, VOXEL)[1])
    runs = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        random_a = fpr.randomize_phases_beyond(a, radius, rng)
        random_b = fpr.randomize_phases_beyond(b, radius, rng)
        runs.append(fpr.fsc(random_a * mask, random_b * mask, VOXEL)[1])
    start = fpr.first_corrected_shell(radius, 2)
    return masked, fpr.correct(masked, np.mean(runs, axis=0), start)


def _resolutions() -> np.ndarray:
    return BOX * VOXEL / np.arange(1, BOX // 2 + 1)


def test_only_phases_past_the_cutoff_change():
    volume = np.random.default_rng(0).standard_normal((BOX,) * 3)
    randomized = fpr.randomize_phases_beyond(volume, 10.0, np.random.default_rng(1))

    before, after = np.fft.rfftn(volume), np.fft.rfftn(randomized)
    beyond = fpr.shell_radius(volume.shape) > 10.0
    assert randomized.dtype == np.float64
    np.testing.assert_allclose(np.abs(after), np.abs(before), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(after[~beyond], before[~beyond], rtol=1e-6, atol=1e-6)
    turned = np.abs(np.angle(after[beyond] * np.conj(before[beyond]))) > 1e-3
    assert turned.mean() > 0.99


def test_shells_are_intact_inside_the_cutoff_and_random_from_the_first_corrected_one():
    volume = np.random.default_rng(2).standard_normal((BOX,) * 3)
    radius = 10.0
    randomized = fpr.randomize_phases_beyond(volume, radius, np.random.default_rng(3))
    correlation = np.asarray(fpr.fsc(volume, randomized, VOXEL)[1])

    # Shell k draws on radii k-1 to k+1.
    np.testing.assert_allclose(correlation[: int(radius) - 1], 1.0, atol=1e-5)
    wholly_random = correlation[fpr.first_corrected_shell(radius, 0) - 1 :]
    assert np.abs(wholly_random).max() < 0.15
    assert abs(wholly_random.mean()) < 0.03


def test_the_first_corrected_shell_does_not_trip_on_rounding():
    # 64 * 5.4 / 21.6 is a hair above 16; shells 17 on are wholly randomised.
    assert fpr.first_corrected_shell(64 * 5.4 / 21.6, 2) == 19
    # An MRC header holds the voxel size as float32.
    assert fpr.first_corrected_shell(64 * float(np.float32(5.4)) / 21.6, 2) == 19
    assert fpr.first_corrected_shell(64 * 5.4 / 30.9, 0) == 13


def test_the_correction_takes_away_what_a_binary_mask_adds():
    rng = np.random.default_rng(1)
    a, b = _pair(rng, _field(rng, 25.0, 3.0))
    tail = _resolutions() <= 16.2 + 1e-6
    unmasked = np.asarray(fpr.fsc(a, b, VOXEL)[1])

    masked, corrected = _masked_and_corrected(a, b, _sphere(0), 21.6)
    assert abs(unmasked[tail].mean()) < 0.02
    assert masked[tail].mean() > 0.2
    assert abs(corrected[tail].mean()) < 0.05

    # From a cutoff coarser than where the shared part ends, the agreement it
    # spreads from between the cutoff and 25 A is randomised away in the control
    # and stays in the correction. That is why the script takes several cutoffs.
    _, from_coarser = _masked_and_corrected(a, b, _sphere(0), 30.9)
    assert from_coarser[tail].mean() > 0.1


@pytest.mark.parametrize("cutoff", [30.9, 21.6])
def test_the_correction_keeps_agreement_the_maps_really_share(cutoff):
    rng = np.random.default_rng(1)
    a, b = _pair(rng, _field(rng, 25.0, 3.0) + _field(rng, 14.0, 0.6))
    resolutions = list(_resolutions())

    masked, corrected = _masked_and_corrected(a, b, _sphere(6), cutoff)
    band = (_resolutions() <= 20.0) & (_resolutions() >= 14.0)
    assert masked[band].mean() > 0.5
    assert abs(corrected[band].mean() - masked[band].mean()) < 0.03
    assert fpr.falls(resolutions, list(corrected), 0.143)[0] == pytest.approx(
        fpr.falls(resolutions, list(masked), 0.143)[0], abs=0.2
    )


def _write(path: Path, data: np.ndarray) -> Path:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(data.astype(np.float32))
        handle.voxel_size = VOXEL
    return path


def test_the_script_prints_the_table_and_writes_the_curves(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(4)
    a, b = _pair(rng, _field(rng, 25.0, 3.0))
    paths = [
        str(_write(tmp_path / name, data))
        for name, data in (("a.mrc", a), ("b.mrc", b), ("mask.mrc", _sphere(6)))
    ]
    out = tmp_path / "control.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fsc_phase_randomization.py", paths[0], paths[1], "--mask", paths[2],
         "--randomize-beyond-angstrom", "30.9", "21.6", "--seeds", "3", "--shells",
         "--out", str(out)],
    )

    assert fpr.main() == 0

    printed = capsys.readouterr().out
    assert "random phases beyond 21.6 A" in printed
    # 48 * 5.4 / 21.6 = 12: shells 13 on are random, 15 on corrected.
    assert f"corrected from {48 * 5.4 / 15:.1f} A" in printed
    payload = json.loads(out.read_text())
    assert len(payload["resolution_angstrom"]) == BOX // 2
    assert [one["corrected_from_shell"] for one in payload["randomized"]] == [12, 15]
    assert all(len(one["corrected"]) == BOX // 2 for one in payload["randomized"])


@pytest.mark.parametrize(
    "extra",
    [
        ["--randomize-beyond-angstrom", "11"],
        ["--randomize-beyond-angstrom", "21.6", "--seeds", "1"],
    ],
    ids=["no-shell-left", "one-seed"],
)
def test_a_control_that_cannot_say_anything_is_refused(tmp_path, monkeypatch, extra):
    volume = np.random.default_rng(5).standard_normal((BOX,) * 3)
    paths = [
        str(_write(tmp_path / name, data))
        for name, data in (("a.mrc", volume), ("b.mrc", volume), ("mask.mrc", _sphere(6)))
    ]
    monkeypatch.setattr(
        sys, "argv", ["fsc_phase_randomization.py", paths[0], paths[1], "--mask", paths[2], *extra]
    )

    with pytest.raises(SystemExit) as refused:
        fpr.main()
    assert refused.value.code == 2
