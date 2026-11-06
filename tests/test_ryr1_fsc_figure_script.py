"""scripts/ryr1_fsc_figure.py: the consensus halves' FSC drawn from what `maps` wrote."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.text
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


def write_maps(folder: Path, name: str, masked, unmasked, resolutions=RESOLUTIONS,
               pixel=PIXEL) -> Path:
    prefix = folder / name
    payload = {"halves": [str(folder / "h1"), str(folder / "h2")], "iteration": 5,
               "pixel_spacing_angstrom": pixel,
               "masked": {"resolution_angstrom": resolutions, "correlation": masked,
                          "rise_toward_nyquist": None},
               "unmasked": {"resolution_angstrom": resolutions, "correlation": unmasked,
                            "rise_toward_nyquist": None}}
    prefix.with_name(f"{name}_fsc.json").write_text(json.dumps(payload))
    return prefix


def run(monkeypatch, *arguments: str) -> list[str]:
    """Every text on the figure main() drew."""

    ryr1_fsc_figure.plt.close("all")
    monkeypatch.setattr(sys, "argv", ["ryr1_fsc_figure.py", *arguments])
    assert ryr1_fsc_figure.main() == 0
    figure = ryr1_fsc_figure.plt.gcf()
    return [item.get_text() for item in figure.findobj(matplotlib.text.Text)]


def test_ticks_start_at_nyquist_and_keep_apart():
    ticks = ryr1_fsc_figure.ticks_for(26.88, RESOLUTIONS[0])

    assert ticks[0] == 26.88
    assert all(t > 26.88 * 1.08 and t <= RESOLUTIONS[0] for t in ticks[1:])
    frequencies = [1 / t for t in ticks]
    gaps = [a - b for a, b in zip(frequencies, frequencies[1:], strict=False)]
    assert all(gap >= 0.07 / 26.88 - 1e-12 for gap in gaps)


def test_the_font_falls_back_to_arials_metric_twin(monkeypatch):
    def installed(*names):
        monkeypatch.setattr(ryr1_fsc_figure.font_manager.fontManager, "ttflist",
                            [SimpleNamespace(name=name) for name in names])

    installed("DejaVu Sans", "Liberation Sans", "Arial")
    assert ryr1_fsc_figure.font() == "Arial"
    installed("DejaVu Sans", "Liberation Sans")
    assert ryr1_fsc_figure.font() == "Liberation Sans"
    installed("DejaVu Sans")
    assert ryr1_fsc_figure.font() == "DejaVu Sans"


def test_the_figure_shows_the_masked_curves_and_never_says_mask(tmp_path, monkeypatch,
                                                                capsys):
    # The unmasked curves are there to be ignored, and the printed rows show which
    # curve was read.
    port = write_maps(tmp_path, "ryr1-consensus-port", linear(36.8, 32.9),
                      linear(54.4, 39.1))
    write_maps(tmp_path, "ryr1-consensus-dynamo", linear(36.9, 33.0), linear(54.5, 39.2))
    monkeypatch.setattr(ryr1_fsc_figure, "half_facts", lambda halves: ([549, 549], 53.76))

    texts = run(monkeypatch, "--pipeline", str(port), "--dynamo",
                str(tmp_path / "ryr1-consensus-dynamo"))

    printed = capsys.readouterr().out
    assert "pipeline       36.8 A       32.9 A      32.9 A" in printed
    assert "dynamo         36.9 A       33.0 A      33.0 A" in printed
    assert (tmp_path / "ryr1-consensus-port_fsc_figure.png").stat().st_size > 0
    assert (tmp_path / "ryr1-consensus-port_fsc_figure.pdf").stat().st_size > 0
    everything = "\n".join(texts)
    assert "mask" not in everything.lower() and "phase random" not in everything.lower()
    assert "39.1" not in everything and "54.4" not in everything
    assert "32.9 Å" in texts  # the resolution line's own label
    assert "Pipeline: 0.5 at 36.8 Å, 0.143 at 32.9 Å" in texts
    assert "(549 + 549)" in everything and "low-passed to 53.76 Å" in everything
    # bin8 is 13.44 A, so KNOWN supplies the MRA's last low-pass.
    assert "refined with all particles to 29.4 Å" in everything
    assert "bin8, 13.44 Å/px, box 52 (699 Å)." in texts


def test_another_pixel_size_gets_no_bin8_facts(tmp_path, monkeypatch, capsys):
    resolutions = [102 * 6.72 / k for k in range(1, 52)]
    falling = [max(0.0, 1 - k / 30) for k in range(1, 52)]
    prefix = write_maps(tmp_path, "bin4", falling, falling, resolutions=resolutions,
                        pixel=6.72)

    texts = run(monkeypatch, "--pipeline", str(prefix), "--dynamo", "none")

    everything = "\n".join(texts)
    assert "bin4, 6.72 Å/px, box 102" in everything
    assert "29.4" not in everything and "poses shared" not in everything
    assert "particle counts left out" in capsys.readouterr().out


def test_dynamo_halves_on_other_shells_are_refused(tmp_path, monkeypatch):
    port = write_maps(tmp_path, "port", linear(36.8, 32.9), linear(54.4, 39.1))
    other = [64 * PIXEL / k for k in range(1, 33)]
    write_maps(tmp_path, "dyn", [0.5] * 32, [0.5] * 32, resolutions=other)
    monkeypatch.setattr(ryr1_fsc_figure, "half_facts", lambda halves: None)

    with pytest.raises(SystemExit, match="other shells"):
        run(monkeypatch, "--pipeline", str(port), "--dynamo", str(tmp_path / "dyn"))
