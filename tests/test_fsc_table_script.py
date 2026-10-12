"""scripts/fsc_table.py: labelled reports and crossings read off the curve."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fsc_table  # noqa: E402


def test_a_crossing_is_interpolated_in_frequency_between_shells():
    resolutions = [100.0, 50.0, 100.0 / 3.0, 25.0]
    values = [1.0, 0.8, 0.4, 0.1]

    # 0.5 lies three quarters of the way from 0.8 to 0.4, so from 1/50 to 1/33.3.
    assert fsc_table.crossing(resolutions, values, 0.5) == pytest.approx(1 / 0.0275)
    assert fsc_table.crossing(resolutions, values, 0.143) == pytest.approx(
        1 / (0.03 + (0.4 - 0.143) / 0.3 * 0.01)
    )
    assert fsc_table.crossing(resolutions, values, 0.05) is None


def test_a_curve_that_dips_and_rises_is_read_at_its_first_fall():
    resolutions = [100.0, 50.0, 25.0, 12.5]
    values = [1.0, 0.3, 0.6, 0.2]

    assert fsc_table.crossing(resolutions, values, 0.5) == pytest.approx(
        1 / (0.01 + (1.0 - 0.5) / 0.7 * 0.01)
    )


def test_reports_are_named_by_their_labels(tmp_path, monkeypatch, capsys):
    curve = {"resolution_angstrom": [100.0, 50.0, 25.0], "correlation": [1.0, 0.8, 0.1]}
    masked = tmp_path / "fsc_masked.json"
    masked.write_text(json.dumps(curve))
    plain = tmp_path / "fsc_plain.json"
    plain.write_text(json.dumps(curve))
    monkeypatch.setattr(
        sys,
        "argv",
        ["fsc_table.py", "--band-coarse-angstrom", "100", "--band-fine-angstrom", "25",
         "--shells", f"b16 masked={masked}", str(plain)],
    )

    assert fsc_table.main() == 0

    out = capsys.readouterr().out
    assert "b16 masked" in out
    # A bare path keeps its file name.
    assert "fsc_plain.json" in out
    assert f"{1 / (0.02 + (0.8 - 0.5) / 0.7 * 0.02):.1f} A" in out


def test_a_label_without_a_path_is_refused():
    with pytest.raises(Exception, match="LABEL=PATH"):
        fsc_table.labelled("masked=")
