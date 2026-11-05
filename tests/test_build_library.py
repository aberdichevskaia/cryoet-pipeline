"""Tests for scripts/build_library.py, the reference library's builder
(docs/library.md)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import sys
import textwrap
import zlib
from contextlib import closing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_library as bl  # noqa: E402
import run_series  # noqa: E402

DAY = "2026-11-05"


# --- converters -----------------------------------------------------------------

def test_html_keeps_its_form_and_loses_the_rest():
    html = """<html><head><title>Guide</title><style>p {color: red}</style>
    <script>alert(1)</script></head><body>
    <h2>Fiducial  models</h2>
    <p>Seeds are <b>found</b>&nbsp;by <a href="x.html">autofidseed</a>;<br>see &quot;beads&quot;.</p>
    <ul><li>first</li><li>second <code>-option</code>
      <ol><li>inner</li></ol></li></ul>
    <pre>
  aligned   text
    kept</pre>
    <table><tr><th>Key</th><th>Meaning</th></tr><tr><td>a|b</td><td><p>one</p></td></tr></table>
    <table><tr><td>layout only</td></tr></table>
    <img src="x.png" alt="a picture"><hr>
    </body></html>"""
    assert bl.html_to_markdown(html) == "\n\n".join([
        "# Guide",
        "## Fiducial models",
        'Seeds are found by autofidseed;\nsee "beads".',
        "- first",
        "- second `-option`",
        "  1. inner",
        "```text\n  aligned   text\n    kept\n```",
        "| Key | Meaning |\n|---|---|\n| a\\|b | one |",
        "layout only",
        "[image: a picture]",
        "---",
    ]) + "\n"


def test_a_manual_page_is_fenced_under_its_title():
    html = ('<html><head><title>tilt</title></head><body><a name="TOP"></a><pre>\n'
            "<i>tilt</i>(1)\n\n<b>NAME</b>\n       tilt - calculates &quot;3-D&quot;\n"
            "</pre></body></html>")
    assert bl.html_to_markdown(html) == (
        '# tilt\n\n```text\ntilt(1)\n\nNAME\n       tilt - calculates "3-D"\n```\n'
    )


def test_the_same_html_gives_the_same_markdown():
    html = (ROOT / "README.md").read_text(encoding="utf-8")
    page = f"<html><body><pre>{html[:4000]}</pre><p>{html[:4000]}</p></body></html>"
    assert bl.html_to_markdown(page) == bl.html_to_markdown(page)


def test_a_fence_is_longer_than_any_inside_it():
    assert bl._fenced("a\n```\nb") == "````text\na\n```\nb\n````"


def test_terminal_text_loses_overstrikes_and_colour():
    raw = "N\bNA\bAM\bME\bE\n       _\bf_\bi_\bl_\be  \x1b[1mbold\x1b[0m   \n\n"
    assert bl.terminal_text(raw) == "NAME\n       file  bold"


def make_qch(path: Path, files: dict[str, bytes]) -> None:
    """A Qt help archive as IMOD ships one: SQLite, each file compressed as
    qCompress does (a 4-byte length, then zlib)."""

    with closing(sqlite3.connect(path)) as db:
        db.execute("CREATE TABLE FileNameTable (FolderId INTEGER, Name TEXT, FileId INTEGER, "
                   "Title TEXT)")
        db.execute("CREATE TABLE FileDataTable (Id INTEGER PRIMARY KEY, Data BLOB)")
        for index, (name, data) in enumerate(files.items(), start=1):
            blob = len(data).to_bytes(4, "big") + zlib.compress(data)
            db.execute("INSERT INTO FileDataTable VALUES (?, ?)", (index, blob))
            db.execute("INSERT INTO FileNameTable VALUES (1, ?, ?, '')", (name, index))
        db.commit()


def test_a_qt_help_archive_gives_its_html_pages(tmp_path):
    qch = tmp_path / "IMOD.qch"
    make_qch(qch, {"man/tilt.html": b"<pre>tilt</pre>", "guide.html": b"<p>g</p>",
                   "pic.png": b"\x89PNG"})
    assert bl.qch_pages(qch) == {"guide.html": b"<p>g</p>", "man/tilt.html": b"<pre>tilt</pre>"}


@pytest.mark.parametrize("name, kept", [
    ("man/tilt.html", True), ("../x.html", False), ("/etc/x.html", False), ("a/../../x", False),
])
def test_a_document_cannot_land_outside_the_snapshot(name, kept):
    assert (bl.safe_name(name) is not None) == kept


# --- coverage -----------------------------------------------------------------------

def test_the_imod_programs_are_read_from_the_code_that_runs_them(tmp_path):
    (tmp_path / "a.py").write_text(
        'x = resolve_imod_executable("newstack", context)\n'
        'y = resolve_imod_executable(\n    "tilt", context)\n')
    (tmp_path / "b.py").write_text('# resolve_imod_executable, named and not called\nname = "beadtrack"\n')
    assert bl.imod_programs_called(tmp_path) == {"newstack", "tilt"}


def test_every_executable_a_lane_names_has_a_source():
    """docs/library.md, "Coverage test": an executable key that run_series reads
    and the library does not cover fails here, by name."""

    lane_keys = {("executables", key) for key in run_series.LANE_KEYS["executables"]}
    lane_keys.add(("segment", "executable"))
    assert "executable" in run_series.LANE_KEYS["segment"]
    assert lane_keys == set(bl.LANE_EXECUTABLES)
    sources = bl.read_sources(bl.SOURCES)
    assert set(bl.LANE_EXECUTABLES.values()) <= set(sources)
    assert set(sources) == set(bl.KNOWN)


@pytest.mark.skipif(not Path("/Applications/IMOD/html/IMOD.qch").is_file(),
                    reason="IMOD's manual is not installed on this host")
def test_the_installed_imod_manual_covers_every_program_the_pipeline_runs():
    found = bl.imod_manual(Path("/Applications/IMOD"))
    assert found is not None
    assert bl.missing_imod_pages(found[1]) == []


def test_a_list_of_sources_may_name_only_tools_it_can_find(tmp_path):
    sources = tmp_path / "sources.toml"
    sources.write_text("[dynamo]\nusage = [['--help']]\n")
    with pytest.raises(SystemExit, match="dynamo"):
        bl.read_sources(sources)


# --- a build on a host made of stand-ins ----------------------------------------------

def executable(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if body.startswith("#!") else "#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return path


@pytest.fixture
def host(tmp_path, monkeypatch):
    """IMOD with an HTML manual, CTFFIND, pytom-match-pick in its own
    environment, Slurm with one manual page and one without, and the code to
    cover calling two IMOD programs."""

    bin_ = tmp_path / "bin"
    imod = tmp_path / "IMOD"
    executable(imod / "bin" / "newstack", "echo newstack\n")
    (imod / "VERSION").write_text("5.1.12\n")
    pages = {
        "man/newstack.html": "<html><head><title>newstack</title></head><body><pre>\n"
                             "<b>NAME</b>\n       newstack - stacks</pre></body></html>",
        "man/tilt.html": "<html><head><title>tilt</title></head><body><pre>\ntilt</pre></body></html>",
        "guide.html": "<html><body><h1>Guide</h1><p>Words.</p></body></html>",
    }
    for name, html in pages.items():
        (imod / "html" / name).parent.mkdir(parents=True, exist_ok=True)
        (imod / "html" / name).write_text(html)
    executable(bin_ / "ctffind", """\
        echo '            Version : 5.0.2'
        echo 'Input image file name [input.mrc]   :'
        """)
    pytom = tmp_path / "pytom" / "bin"
    executable(pytom / "python3", """\
        if [ "$1" = "-I" ]; then echo 0.14.0; else echo "usage: $(basename "$1") [-h]"; fi
        """)
    # `env` on the first line, as pip writes it for a script it does not
    # install: macOS will not run a script whose interpreter is itself a script.
    executable(pytom / "pytom_match_template.py", "#!/usr/bin/env python3\n")
    executable(bin_ / "man", r"""
        case "$1" in
          sbatch) printf 'S\bSB\bBA\bAT\bTC\bCH\bH\n       submits a batch script\n' ;;
          *) echo "No manual entry for $1" >&2; exit 16 ;;
        esac
        """)
    executable(bin_ / "sinfo", 'if [ "$1" = "-V" ]; then echo "slurm 26.05.3"; else echo gpu up; fi\n')
    executable(bin_ / "squeue", "echo 'Usage: squeue [OPTIONS]'\n")
    src = tmp_path / "src"
    src.mkdir()
    (src / "step.py").write_text('a = resolve_imod_executable("newstack", c)\n'
                                 'b = resolve_imod_executable("tilt", c)\n')
    sources = tmp_path / "sources.toml"
    sources.write_text(textwrap.dedent("""\
        [imod]
        manual = true
        [ctffind]
        usage = [["--help"]]
        [pytom-match-pick]
        scripts = ["pytom_match_template.py"]
        usage = [["--help"]]
        [slurm]
        man = ["sbatch", "squeue"]
        """))
    monkeypatch.setenv("PATH", f"{bin_}:{pytom}:{imod / 'bin'}:/usr/bin:/bin")
    monkeypatch.setenv("PYTOMO_MP_DIR", str(pytom))
    for variable in ("IMOD_DIR", "CTFFIND5", "MOTIONCOR3", "MEMBRAIN_EXE", "CRYOET_LIBRARY"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(bl, "SRC", src)
    monkeypatch.setattr(bl, "today", lambda: DAY)
    monkeypatch.setattr(bl, "SITE_PROBES", (
        ("uname", ("uname", "-a")),
        ("slurm", ("sinfo", "-V")),
        ("absent", ("no-such-program-here",)),
    ))
    root = tmp_path / "library"
    root.mkdir()
    yield {"root": root, "sources": sources, "tmp": tmp_path}
    # A finished snapshot is read-only; let pytest clean up after it.
    for folder, _, files in os.walk(tmp_path):
        os.chmod(folder, 0o755)
        for name in files:
            path = os.path.join(folder, name)
            if not os.path.islink(path):
                os.chmod(path, 0o644)


def run(host, *more: str) -> int:
    return bl.main(["--site", "test", "--root", str(host["root"]),
                    "--sources", str(host["sources"]), *more])


def tree(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def test_a_dry_run_finds_the_versions_and_writes_nothing(host, capsys):
    assert run(host, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert list(host["root"].iterdir()) == []
    assert "dry run, nothing written" in printed
    for name, version in (("imod", "5.1.12"), ("ctffind", "5.0.2"),
                          ("pytom-match-pick", "0.14.0"), ("slurm", "26.05.3")):
        assert f"{name:<18}{version}" in printed
    assert "IMOD/html (3 pages)" in printed
    assert "Problems: none" in printed


def test_a_build_writes_a_read_only_snapshot_and_points_current_at_it(host, capsys):
    assert run(host) == 0
    snapshot = host["root"] / f"test-{DAY}"
    assert os.readlink(host["root"] / "current") == f"test-{DAY}"
    assert [path.name for path in host["root"].iterdir() if path.name.startswith(".")] == []

    record = json.loads((snapshot / "library.json").read_text())
    versions = {tool["name"]: tool["version"] for tool in record["tools"]}
    assert versions == {"imod": "5.1.12", "ctffind": "5.0.2", "pytom-match-pick": "0.14.0",
                        "slurm": "26.05.3"}
    assert record["problems"] == []
    assert record["documents"] == 7
    assert any("no manual page squeue" in note for note in record["notes"])
    for source in record["sources"]:
        if source.get("raw") and not source["raw"].endswith("/"):
            assert hashlib.sha256((snapshot / source["raw"]).read_bytes()).hexdigest() == source["sha256"]

    tilt = (snapshot / "text" / "imod" / "man" / "tilt.md").read_text()
    assert tilt.startswith('---\ntool: "imod"\nversion: "5.1.12"\nchannel: "installation"\n')
    assert "SBATCH\n       submits a batch script" in (snapshot / "text" / "slurm" / "sbatch.md").read_text()
    assert "Usage: squeue" in (snapshot / "text" / "slurm" / "squeue-help.md").read_text()
    assert "Input image file name" in (snapshot / "text" / "ctffind" / "help.md").read_text()
    assert "usage: pytom_match_template.py" in (
        snapshot / "text" / "pytom-match-pick" / "pytom_match_template-help.md").read_text()
    site = (snapshot / "site.md").read_text()
    assert "slurm 26.05.3" in site
    assert "Could not run" in site

    for folder, folders, files in os.walk(snapshot):
        assert stat.S_IMODE(os.stat(folder).st_mode) == 0o550, folder
        for name in files:
            assert stat.S_IMODE(os.stat(os.path.join(folder, name)).st_mode) == 0o440, name


def test_a_second_build_is_a_new_snapshot_and_the_first_is_untouched(host):
    assert run(host) == 0
    first = host["root"] / f"test-{DAY}"
    before = tree(first)
    assert run(host) == 0
    second = host["root"] / f"test-{DAY}b"
    assert os.readlink(host["root"] / "current") == f"test-{DAY}b"
    assert tree(first) == before
    # The same sources give the same text.
    assert tree(first / "text") == tree(second / "text")


def test_a_failed_build_leaves_nothing_behind(host, monkeypatch):
    assert run(host) == 0

    def broken(*_args, **_kwargs):
        raise RuntimeError("broken on purpose")

    monkeypatch.setattr(bl, "collect_site", broken)
    with pytest.raises(RuntimeError, match="on purpose"):
        run(host)
    assert sorted(path.name for path in host["root"].iterdir()) == ["current", f"test-{DAY}"]
    assert os.readlink(host["root"] / "current") == f"test-{DAY}"


def test_the_library_folder_is_not_made_here(host):
    with pytest.raises(SystemExit, match="make it once"):
        bl.main(["--site", "test", "--root", str(host["tmp"] / "absent"),
                 "--sources", str(host["sources"])])


def test_a_lane_that_names_an_unset_variable_is_a_problem(host, capsys):
    lane = host["tmp"] / "lane.toml"
    lane.write_text('[executables]\nctffind = "$CTFFIND5"\n')
    assert run(host, "--dry-run", "--config", str(lane)) == 0
    assert '$CTFFIND5 is not set in this shell' in capsys.readouterr().out


def test_a_given_version_is_kept_with_what_the_tool_said(host):
    assert run(host, "--version-of", "ctffind=5.0.3") == 0
    record = json.loads((host["root"] / f"test-{DAY}" / "library.json").read_text())
    ctffind = next(tool for tool in record["tools"] if tool["name"] == "ctffind")
    assert ctffind["version"] == "5.0.3"
    assert "states 5.0.2" in ctffind["version_from"]


@pytest.mark.parametrize("argv", [
    ["--site", "Has Spaces", "--dry-run"],
    ["--site", "ok", "--dry-run", "--version-of", "dynamo=1.1"],
])
def test_bad_arguments_are_refused(argv, host):
    with pytest.raises(SystemExit):
        bl.main([*argv, "--sources", str(host["sources"])])
