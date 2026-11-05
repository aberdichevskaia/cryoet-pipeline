"""Tests for scripts/search_library.py, the assistant's search of the
reference library (docs/library.md)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import search_library as sl  # noqa: E402


def page(tool: str, version: str, document: str, body: str) -> str:
    return (f'---\ntool: "{tool}"\nversion: "{version}"\nchannel: "installation"\n'
            f'document: "{document}"\n---\n{body}')


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A library folder with one snapshot, `current` pointing at it."""
    snapshot = tmp_path / "library" / "site-2026-11-05"
    (snapshot / "text" / "imod" / "man").mkdir(parents=True)
    (snapshot / "text" / "slurm").mkdir(parents=True)
    (snapshot / "library.json").write_text("{}\n")
    (snapshot / "text" / "imod" / "man" / "newstack.md").write_text(page(
        "imod", "5.1.12", "man/newstack.html", "# newstack\n\n-bin       Binning\nno flag here\n"))
    # More matches than newstack.md, and later in name order.
    (snapshot / "text" / "imod" / "man" / "tilt.md").write_text(page(
        "imod", "5.1.12", "man/tilt.html",
        "# tilt\n\n-LOCALFILE   File name\nsee -bin\nreduced with -bin, -shrink\n"))
    (snapshot / "text" / "slurm" / "sbatch.md").write_text(page(
        "slurm", "26.05.4", "", "# sbatch\n\n--gres=gpu:a40:1\n"))
    (tmp_path / "library" / "current").symlink_to("site-2026-11-05")
    for name in ("CRYOET_LIBRARY", "CRYOET_LIBRARY_SNAPSHOT"):
        monkeypatch.delenv(name, raising=False)
    return snapshot


def test_pages_with_more_matches_come_first_with_their_source(library, monkeypatch, capsys):
    monkeypatch.setenv("CRYOET_LIBRARY_SNAPSHOT", str(library))
    assert sl.main(["-e", "-bin"]) == 0
    printed = capsys.readouterr().out
    assert printed == "\n".join([
        "library site-2026-11-05: 2 of 3 pages match '-bin'",
        "",
        "text/imod/man/tilt.md  (imod 5.1.12, man/tilt.html; 2 lines)",
        "  10: see -bin",
        "  11: reduced with -bin, -shrink",
        "",
        "text/imod/man/newstack.md  (imod 5.1.12, man/newstack.html; 1 lines)",
        "  9: -bin       Binning",
    ]) + "\n"


def test_a_plain_pattern_is_taken_as_written(library, capsys):
    assert sl.main(["a.0", "--snapshot", str(library)]) == 0
    assert "0 of 3 pages" in capsys.readouterr().out
    assert sl.main(["a.0", "--regex", "--snapshot", str(library)]) == 0
    assert "1 of 3 pages" in capsys.readouterr().out


def test_the_header_is_not_searched_and_case_is_ignored(library, monkeypatch, capsys):
    monkeypatch.setenv("CRYOET_LIBRARY", str(library.parent))
    assert sl.main(["IMOD", "--tool", "imod"]) == 0
    assert "0 of 2 pages match" in capsys.readouterr().out
    assert sl.main(["localfile"]) == 0
    assert "text/imod/man/tilt.md" in capsys.readouterr().out


def test_a_regex_and_one_tools_pages(library, capsys):
    assert sl.main(["gres=gpu:\\w+", "--regex", "--tool", "slurm",
                    "--snapshot", str(library)]) == 0
    out = capsys.readouterr().out
    assert "1 of 1 pages" in out and "  9: --gres=gpu:a40:1" in out


def test_the_output_is_bounded(library, capsys):
    assert sl.main(["-e", "-bin", "--pages", "1", "--lines", "1", "--snapshot", str(library)]) == 0
    out = capsys.readouterr().out
    assert "(1 more lines)" in out and "(1 more pages" in out
    assert "newstack.md" not in out


def test_without_a_library_it_says_so(library, capsys):
    assert sl.main(["anything"]) == 2
    assert "no library in this session" in capsys.readouterr().err


@pytest.mark.parametrize("tool", ["../..", "IMOD/../x", "/etc"])
def test_a_tool_is_a_name_not_a_path(library, tool):
    with pytest.raises(SystemExit):
        sl.main(["x", "--tool", tool, "--snapshot", str(library)])


def test_an_unknown_tool_names_the_ones_there(library, capsys):
    assert sl.main(["x", "--tool", "dynamo", "--snapshot", str(library)]) == 2
    assert "the library has imod, slurm" in capsys.readouterr().err
