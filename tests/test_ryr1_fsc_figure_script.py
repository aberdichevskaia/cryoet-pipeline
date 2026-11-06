"""scripts/ryr1_fsc_figure.py: the consensus halves' FSC drawn from what `maps` wrote."""

import json
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ryr1_fsc_figure  # noqa: E402

PIXEL, BOX = 13.44, 52
RESOLUTIONS = [BOX * PIXEL / k for k in range(1, BOX // 2 + 1)]


def linear(r50: float, r143: float) -> list[float]:
    """Through 0.5 at r50 and 0.143 at r143, straight in spatial frequency there, so
    the interpolated crossings are exactly r50 and r143."""

    slope = (0.5 - 0.143) / (1 / r143 - 1 / r50)
    return [min(1.0, max(-0.2, 0.5 - slope * (1 / r - 1 / r50))) for r in RESOLUTIONS]


def write_maps(folder: Path, name: str, masked, unmasked, resolutions=RESOLUTIONS) -> Path:
    prefix = folder / name
    payload = {"halves": [str(folder / "h1"), str(folder / "h2")], "iteration": 5,
               "pixel_spacing_angstrom": PIXEL,
               "masked": {"resolution_angstrom": resolutions, "correlation": masked,
                          "rise_toward_nyquist": None},
               "unmasked": {"resolution_angstrom": resolutions, "correlation": unmasked,
                            "rise_toward_nyquist": None}}
    prefix.with_name(f"{name}_fsc.json").write_text(json.dumps(payload))
    return prefix


def write_mask(path: Path, volume: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(volume.astype(np.float32))
        handle.voxel_size = PIXEL


def soft_sphere(radius: float = 16.0, edge: float = 2.0) -> np.ndarray:
    z, y, x = np.indices((BOX,) * 3)
    r = np.sqrt((x - BOX / 2) ** 2 + (y - BOX / 2) ** 2 + (z - BOX / 2) ** 2)
    inner = radius - edge / 2
    mask = 0.5 + 0.5 * np.cos(np.pi * (r - inner) / edge)
    mask[r <= inner] = 1.0
    mask[r >= inner + edge] = 0.0
    return mask


def test_a_soft_sphere_is_described_by_its_measured_size_and_edge(tmp_path):
    write_mask(tmp_path / "m.mrc", soft_sphere())

    assert ryr1_fsc_figure.describe_mask(tmp_path / "m.mrc", PIXEL) == (
        "spherical mask, Ø 430 Å (radius 16 px), 2 px soft edge")


def test_a_mask_that_is_not_a_sphere_is_not_called_one(tmp_path):
    cube = np.zeros((BOX,) * 3)
    cube[10:42, 10:42, 10:42] = 1.0
    write_mask(tmp_path / "cube.mrc", cube)

    assert "not a sphere" in ryr1_fsc_figure.describe_mask(tmp_path / "cube.mrc", PIXEL)
    assert ryr1_fsc_figure.describe_mask(tmp_path / "none.mrc", PIXEL) == (
        "the halves' alignment mask")


def test_ticks_start_at_nyquist_and_keep_apart():
    ticks = ryr1_fsc_figure.ticks_for(26.88, RESOLUTIONS[0])

    assert ticks[0] == 26.88
    assert all(t > 26.88 * 1.08 and t <= RESOLUTIONS[0] for t in ticks[1:])
    frequencies = [1 / t for t in ticks]
    gaps = [a - b for a, b in zip(frequencies, frequencies[1:], strict=False)]
    assert all(gap >= 0.07 / 26.88 - 1e-12 for gap in gaps)


def run(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ryr1_fsc_figure.py", *arguments])
    assert ryr1_fsc_figure.main() == 0


def test_the_figure_reads_both_refiners_and_names_the_shared_poses(tmp_path, monkeypatch,
                                                                    capsys):
    port = write_maps(tmp_path, "ryr1-consensus-port", linear(36.8, 32.9),
                      linear(54.4, 39.1))
    write_maps(tmp_path, "ryr1-consensus-dynamo", linear(36.9, 33.0), linear(54.4, 39.1))
    write_mask(tmp_path / "ryr1-consensus-port_mask.mrc", soft_sphere())
    monkeypatch.setattr(ryr1_fsc_figure, "half_facts", lambda report: ([549, 549], 53.76))
    captions: list[str] = []
    original = ryr1_fsc_figure.plt.Figure.text

    def keep(figure, x, y, text, *rest, **options):
        captions.append(text)
        return original(figure, x, y, text, *rest, **options)

    monkeypatch.setattr(ryr1_fsc_figure.plt.Figure, "text", keep)

    run(monkeypatch, "--pipeline", str(port), "--dynamo",
        str(tmp_path / "ryr1-consensus-dynamo"))

    printed = capsys.readouterr().out
    assert "masked         36.8 A       32.9 A      32.9 A" in printed
    assert "unmasked       54.4 A       39.1 A      39.1 A" in printed
    assert "dynamo         36.9 A       33.0 A      33.0 A" in printed
    assert (tmp_path / "ryr1-consensus-port_fsc_figure.png").stat().st_size > 0
    assert (tmp_path / "ryr1-consensus-port_fsc_figure.pdf").stat().st_size > 0
    caption = "\n".join(captions)
    assert "(549 + 549)" in caption and "low-passed to 53.76 Å" in caption
    # bin8 is 13.44 A, so KNOWN supplies the MRA's last low-pass and the control.
    assert "refined with all particles to 29.4 Å" in caption
    assert "Phase randomisation" in caption
    assert "spherical mask, Ø 430 Å (radius 16 px), 2 px soft edge" in caption


def test_another_pixel_size_gets_no_bin8_facts(tmp_path, monkeypatch, capsys):
    resolutions = [102 * 6.72 / k for k in range(1, 52)]
    masked = [max(0.0, 1 - k / 30) for k in range(1, 52)]
    prefix = tmp_path / "bin4"
    prefix.with_name("bin4_fsc.json").write_text(json.dumps({
        "halves": [], "pixel_spacing_angstrom": 6.72,
        "masked": {"resolution_angstrom": resolutions, "correlation": masked},
        "unmasked": {"resolution_angstrom": resolutions, "correlation": masked}}))
    captions: list[str] = []
    original = ryr1_fsc_figure.plt.Figure.text
    monkeypatch.setattr(ryr1_fsc_figure.plt.Figure, "text",
                        lambda figure, x, y, text, *rest, **options:
                        captions.append(text) or original(figure, x, y, text, *rest, **options))

    run(monkeypatch, "--pipeline", str(prefix), "--dynamo", "none")

    caption = "\n".join(captions)
    assert "bin4, 6.72 Å/px, box 102" in caption
    assert "29.4" not in caption and "Phase randomisation" not in caption
    assert "particle counts left out" in capsys.readouterr().out


def test_dynamo_halves_on_other_shells_are_refused(tmp_path, monkeypatch):
    port = write_maps(tmp_path, "port", linear(36.8, 32.9), linear(54.4, 39.1))
    other = [64 * PIXEL / k for k in range(1, 33)]
    write_maps(tmp_path, "dyn", [0.5] * 32, [0.5] * 32, resolutions=other)
    monkeypatch.setattr(ryr1_fsc_figure, "half_facts", lambda report: None)

    with pytest.raises(SystemExit, match="other shells"):
        run(monkeypatch, "--pipeline", str(port), "--dynamo", str(tmp_path / "dyn"))
