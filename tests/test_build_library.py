"""Tests for scripts/build_library.py, the reference library's builder
(docs/library.md)."""

from __future__ import annotations

import hashlib
import json
import os
import re
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
    <p>Seeds are <b>found</b>&nbsp;by
    <a href="x.html">autofidseed</a>;<br>see &quot;beads&quot;.</p>
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
    (tmp_path / "b.py").write_text('# resolve_imod_executable, named and not called\n'
                                   'name = "beadtrack"\n')
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


def tiny_pdf(*pages: str) -> bytes:
    """A PDF whose pages each show one line of Helvetica: a converter's fixture,
    written out here so that the test needs nothing to make it."""

    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "",
               "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents "
                       f"{len(objects)} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(len(objects))
    objects[1] = (f"<< /Type /Pages /Kids [{' '.join(f'{kid} 0 R' for kid in kids)}] "
                  f"/Count {len(kids)} >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n"
            "%%EOF\n").encode()
    return out


def test_a_pdf_gives_its_text_page_by_page_with_single_spaces():
    pdf = tiny_pdf("Tilt  axis:   -TiltAxis  initVal  refineFlag", "The second page")
    markdown, pages = bl.pdf_to_markdown(pdf, "Manual.pdf")
    assert pages == 2
    assert markdown == ("# Manual.pdf\n\n## Page 1\n\nTilt axis: -TiltAxis initVal refineFlag\n\n"
                        "## Page 2\n\nThe second page\n")
    assert bl.pdf_to_markdown(pdf, "Manual.pdf") == (markdown, pages)


PIN = {"repository": "https://github.com/owner/tool", "commit": "0123456789" * 4,
       "version": "AreTomo2 1.1.3", "files": {"docs/Manual.pdf": "a" * 64}}


@pytest.mark.parametrize("change, said", [
    ({"commit": "main"}, "full 40-character commit"),
    ({"repository": "https://gitlab.com/owner/tool"}, "https://github.com"),
    ({"version": " "}, "version must be"),
    ({"files": {}}, "files must map"),
    ({"files": {"../Manual.pdf": "a" * 64}}, "inside the repository"),
    ({"files": {"Manual.docx": "a" * 64}}, "ending in"),
    ({"files": {"Manual.pdf": "A" * 64}}, "64 lower-case hex"),
    ({"branch": "main"}, "nothing else"),
])
def test_a_pin_must_name_a_commit_its_version_and_each_files_sha256(tmp_path, change, said):
    sources = tmp_path / "sources.toml"
    pin = {**PIN, **change}
    sources.write_text("[aretomo]\n")
    with pytest.raises(SystemExit, match=re.escape(said)):
        bl.check_pin(sources, "aretomo", pin)


def test_the_pins_of_the_list_of_sources_are_well_formed():
    sources = bl.read_sources(bl.SOURCES)
    assert sources["aretomo"]["pinned"]["version"] == "AreTomo2 1.1.3"


def test_a_fetch_that_fails_stops_the_build():
    # Port 9 (discard) on this host: refused at once, without the network.
    with pytest.raises(SystemExit, match="could not fetch http://127.0.0.1:9/"):
        bl.fetch("http://127.0.0.1:9/Manual.pdf")


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
        "man/tilt.html": "<html><head><title>tilt</title></head><body><pre>\ntilt</pre>"
                         "</body></html>",
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
          -w) if [ "$2" = sbatch ]; then echo /usr/share/man/man1/sbatch.1; else exit 16; fi ;;
          sbatch) printf 'S\bSB\bBA\bAT\bTC\bCH\bH\n       submits a batch script\n' ;;
          *) echo "No manual entry for $1" >&2; exit 16 ;;
        esac
        """)
    executable(bin_ / "sinfo",
               'if [ "$1" = "-V" ]; then echo "slurm 26.05.3"; else echo gpu up; fi\n')
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
    for variable in ("IMOD_DIR", "CTFFIND5", "MOTIONCOR3", "ARETOMO", "MEMBRAIN_EXE",
                     "CRYOET_LIBRARY", "CONDA_EXE", "MAMBA_EXE"):
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
    assert "man sbatch; squeue --help (no manual page here)" in printed
    assert "Problems: none" in printed


def test_with_a_lane_given_a_missing_imod_is_a_problem(host, capsys, monkeypatch):
    monkeypatch.setenv("PATH", os.environ["PATH"].replace(f"{host['tmp'] / 'IMOD' / 'bin'}:", ""))
    monkeypatch.setattr(bl, "MAC_IMOD", host["tmp"] / "no-imod-here")
    assert run(host, "--dry-run") == 0
    alone = capsys.readouterr().out
    assert "Problems: none" in alone and "imod              -         not on this host" in alone
    lane = host["tmp"] / "lane.toml"
    lane.write_text("[lane]\nname = 'any'\n")
    assert run(host, "--dry-run", "--config", str(lane)) == 0
    assert "  - imod: every lane runs IMOD, and no newstack on PATH" in capsys.readouterr().out


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
            raw = (snapshot / source["raw"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == source["sha256"]

    tilt = (snapshot / "text" / "imod" / "man" / "tilt.md").read_text()
    assert tilt.startswith('---\ntool: "imod"\nversion: "5.1.12"\nchannel: "installation"\n')
    sbatch = (snapshot / "text" / "slurm" / "sbatch.md").read_text()
    assert "SBATCH\n       submits a batch script" in sbatch
    assert "Usage: squeue" in (snapshot / "text" / "slurm" / "squeue-help.md").read_text()
    assert "Input image file name" in (snapshot / "text" / "ctffind" / "help.md").read_text()
    assert "usage: pytom_match_template.py" in (
        snapshot / "text" / "pytom-match-pick" / "pytom_match_template-help.md").read_text()
    site = (snapshot / "site.md").read_text()
    assert "slurm 26.05.3" in site
    assert "Could not run" in site

    for folder, _, files in os.walk(snapshot):
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


# What AreTomo2 1.1.3 prints (CAreTomoMain.cpp and CInput.cpp at main
# 01e7219c79, read on 2026-11-11).
ARETOMO_113 = """\
    case "$1" in
      --version) printf 'AreTomo2 version 1.1.3\\nBuilt on Aug 19 2024\\n' ;;
      --help) printf '\\nUsage: AreTomo2 Tags\\n-InMrc    \\n'
              printf '   1. Input MRC file that stores tomo tilt series.\\n\\n' ;;
    esac
    """


def with_aretomo(host, monkeypatch, body: str) -> Path:
    """An AreTomo2 stand-in named by $ARETOMO, and sources that ask it for --help."""
    exe = executable(host["tmp"] / "AreTomo2-cu12.5" / "AreTomo2", body)
    monkeypatch.setenv("ARETOMO", str(exe))
    with host["sources"].open("a") as sources:
        sources.write('[aretomo]\nusage = [["--help"]]\n')
    return exe


def test_aretomo_is_found_by_its_variable_and_documented_under_its_name(host, capsys,
                                                                        monkeypatch):
    exe = with_aretomo(host, monkeypatch, ARETOMO_113)
    assert run(host, "--dry-run") == 0
    printed = capsys.readouterr().out
    # The name is part of the version, and the path does not run into it.
    assert (f"  {'aretomo':<18}AreTomo2 1.1.3  {exe}  ($ARETOMO; version from "
            "AreTomo2 --version)") in printed
    assert f"  {'imod':<18}{'5.1.12':<16}" in printed
    assert "Problems: none" in printed

    assert run(host) == 0
    snapshot = host["root"] / f"test-{DAY}"
    record = json.loads((snapshot / "library.json").read_text())
    assert {tool["name"]: tool["version"] for tool in record["tools"]}["aretomo"] == \
        "AreTomo2 1.1.3"
    assert record["problems"] == []
    page = (snapshot / "text" / "aretomo" / "help.md").read_text()
    assert page.startswith('---\ntool: "aretomo"\nversion: "AreTomo2 1.1.3"\n'
                           'channel: "installation"\n')
    assert "1. Input MRC file that stores tomo tilt series." in page


def test_a_program_that_cannot_start_leaves_a_problem_and_no_page(host, capsys, monkeypatch):
    # How a dynamically linked program ends when a library it links is not on
    # the library path: the loader's message, exit code 127.
    with_aretomo(host, monkeypatch, """\
        said="error while loading shared libraries: libcufft.so.12"
        echo "$0: $said: cannot open shared object file: No such file or directory" >&2
        exit 127
        """)
    assert run(host, "--dry-run") == 0
    assert "aretomo: no version found" in capsys.readouterr().out
    assert run(host) == 0
    snapshot = host["root"] / f"test-{DAY}"
    assert not (snapshot / "text" / "aretomo").exists()
    assert not (snapshot / "raw" / "aretomo").exists()
    problems = json.loads((snapshot / "library.json").read_text())["problems"]
    assert [problem for problem in problems if "AreTomo2 --help" in problem] == [
        f"AreTomo2 --help: could not start (exit code 127): {host['tmp']}/AreTomo2-cu12.5/"
        "AreTomo2: error while loading shared libraries: libcufft.so.12: cannot open shared "
        "object file: No such file or directory"]


def with_pinned(host, monkeypatch, version: str = "AreTomo2 1.1.3",
                sha: dict[str, str] | None = None) -> list[str]:
    """AreTomo2 1.1.3 with a manual and release notes pinned in a stand-in
    repository; returns the list the fetches are counted in."""

    with_aretomo(host, monkeypatch, ARETOMO_113)
    served = {"docs/Manual.pdf": tiny_pdf("13. Volume z height for alignment:  -AlignZ"),
              "Readme.txt": b"AreTomo2 1.1.3 [06-17-2024]\r\n1. Bug fix.\r\n"}
    pins = {path: hashlib.sha256(data).hexdigest() for path, data in served.items()}
    pins.update(sha or {})
    with host["sources"].open("a") as sources:
        sources.write(textwrap.dedent(f"""\
            [aretomo.pinned]
            repository = "https://github.com/owner/AreTomo2"
            commit = "{PIN['commit']}"
            version = "{version}"
            [aretomo.pinned.files]
            """) + "".join(f'"{path}" = "{digest}"\n' for path, digest in pins.items()))
    fetched: list[str] = []

    def fetch(url: str) -> bytes:
        fetched.append(url)
        prefix = f"https://raw.githubusercontent.com/owner/AreTomo2/{PIN['commit']}/"
        assert url.startswith(prefix)
        return served[url.removeprefix(prefix)]

    monkeypatch.setattr(bl, "fetch", fetch)
    return fetched


def test_pinned_files_are_fetched_checked_kept_and_converted(host, capsys, monkeypatch):
    fetched = with_pinned(host, monkeypatch)
    assert run(host, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert ("help; docs/Manual.pdf of github.com/owner/AreTomo2 at 0123456789; Readme.txt of "
            "github.com/owner/AreTomo2 at 0123456789") in printed
    assert "Problems: none" in printed
    assert fetched == []

    assert run(host) == 0
    assert len(fetched) == 2
    snapshot = host["root"] / f"test-{DAY}"
    record = json.loads((snapshot / "library.json").read_text())
    assert record["problems"] == []
    pinned = {source["path"]: source for source in record["sources"] if source["kind"] == "pinned"}
    assert set(pinned) == {"docs/Manual.pdf", "Readme.txt"}
    manual = pinned["docs/Manual.pdf"]
    assert manual["channel"] == "web, pinned" and manual["version"] == "AreTomo2 1.1.3"
    assert manual["commit"] == PIN["commit"] and manual["pages"] == 1
    assert manual["converter"].startswith("pdf 1 (pypdf ")
    raw = (snapshot / manual["raw"]).read_bytes()
    assert raw.startswith(b"%PDF") and hashlib.sha256(raw).hexdigest() == manual["sha256"]

    text = (snapshot / "text" / "aretomo" / "docs" / "Manual.md").read_text()
    assert text.startswith('---\ntool: "aretomo"\nversion: "AreTomo2 1.1.3"\n'
                           'channel: "web, pinned"\nsource: "https://github.com/owner/AreTomo2/'
                           f'blob/{PIN["commit"]}/docs/Manual.pdf"\ndocument: "docs/Manual.pdf"\n')
    assert "13. Volume z height for alignment: -AlignZ" in text
    # Bytes: read_text would turn the "\r\n" it should not hold into "\n" itself.
    notes = (snapshot / "text" / "aretomo" / "Readme.md").read_bytes().decode()
    assert "```text\nAreTomo2 1.1.3 [06-17-2024]\n1. Bug fix.\n```" in notes
    assert (snapshot / "text" / "aretomo" / "help.md").is_file()


def test_a_pinned_file_that_changed_stops_the_build_and_leaves_nothing(host, monkeypatch):
    with_pinned(host, monkeypatch, sha={"Readme.txt": "b" * 64})
    with pytest.raises(SystemExit, match="pins " + "b" * 64):
        run(host)
    assert list(host["root"].iterdir()) == []


def test_pinned_files_of_another_version_are_left_out(host, capsys, monkeypatch):
    fetched = with_pinned(host, monkeypatch, version="AreTomo2 1.1.2")
    assert run(host) == 0
    assert fetched == []
    snapshot = host["root"] / f"test-{DAY}"
    problems = json.loads((snapshot / "library.json").read_text())["problems"]
    assert len(problems) == 2
    assert all("documents AreTomo2 1.1.2, and the aretomo here states AreTomo2 1.1.3" in problem
               for problem in problems)
    # What the installation itself says stays.
    assert (snapshot / "text" / "aretomo" / "help.md").is_file()
    assert not (snapshot / "raw" / "aretomo" / "docs").exists()


def test_a_pdf_without_pypdf_is_a_problem_and_the_rest_is_read(host, monkeypatch):
    fetched = with_pinned(host, monkeypatch)
    monkeypatch.setattr(bl, "pdf_reader", lambda: None)
    assert run(host) == 0
    assert [url.rsplit("/", 1)[1] for url in fetched] == ["Readme.txt"]
    snapshot = host["root"] / f"test-{DAY}"
    problems = json.loads((snapshot / "library.json").read_text())["problems"]
    assert problems == ["aretomo: docs/Manual.pdf is a PDF, and pypdf is not installed in this "
                        "environment; it is left out. Install it (pip install pypdf) and build "
                        "again"]
    assert (snapshot / "text" / "aretomo" / "Readme.md").is_file()


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


# --- the web channel's pages ---------------------------------------------------------

WEB = """\
[web.cuda]
versions = "cuda"
[web.cuda.pages]
release-notes = [
    "https://docs.nvidia.com/cuda/archive/{version}/cuda-toolkit-release-notes/index.html",
    "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html",
]
[web.nvidia.pages]
gpus = "https://developer.nvidia.com/cuda/gpus"
"""


def page(title: str, body: str, frame: str = "<nav>Menu Home Search</nav>") -> bytes:
    """A documentation page: the site's frame around an article."""
    return (f"<html><head><title>{title}</title></head><body>{frame}"
            f'<div itemprop="articleBody">{body}</div><footer>Copyright</footer></body></html>'
            ).encode()


def with_web(host, monkeypatch, served: dict[str, object], web: str = WEB) -> list[str]:
    """Pages served by address: bytes, (bytes, final address), or None for 404.
    The CUDA versions come from stand-ins; returns the addresses asked for."""
    with host["sources"].open("a") as sources:
        sources.write(web)
    asked: list[str] = []

    def fetch_page(url):
        asked.append(url)
        answer = served.get(url)
        if answer is None:
            return None
        return answer if isinstance(answer, tuple) else (answer, url)

    monkeypatch.setattr(bl, "fetch_page", fetch_page)
    monkeypatch.setattr(bl, "module_names", lambda: [])
    monkeypatch.setattr(bl, "torch_cuda", lambda: "")
    monkeypatch.setattr(bl, "cupy_cuda", lambda tool: "")
    return asked


ARCHIVE = "https://docs.nvidia.com/cuda/archive/{}/cuda-toolkit-release-notes/index.html"
CURRENT = "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"
GPUS = "https://developer.nvidia.com/cuda/gpus"


def test_pages_are_read_by_version_and_dated_and_only_their_own_text_kept(host, monkeypatch):
    with_web(host, monkeypatch, {
        ARCHIVE.format("12.5.0"): page(
            "CUDA Toolkit Release Notes",
            '<h1>Release Notes<a class="headerlink" href="#r">#</a></h1>'
            "<table><tr><th>CTK</th><th>Driver</th></tr><tr><td>12.x</td><td>&gt;= 525</td>"
            "</tr></table>"),
        GPUS: b"<html><head><title>GPUs</title></head><body><nav>Menu</nav><main>"
              b"<table><tr><th>CC</th><th>GPU</th></tr><tr><td>8.6</td><td>A40</td></tr></table>"
              b"</main></body></html>",
    })
    assert run(host, "--version-of", "cuda=12.5") == 0
    snapshot = host["root"] / f"test-{DAY}"
    notes = (snapshot / "text" / "cuda" / "12.5.0" / "release-notes.md").read_text()
    assert notes.startswith('---\ntool: "cuda"\nversion: "12.5.0"\nchannel: "web, by version"\n'
                            f'source: "{ARCHIVE.format("12.5.0")}"\n')
    assert "# Release Notes\n" in notes and "| 12.x | >= 525 |" in notes
    assert "Menu" not in notes and "Copyright" not in notes and "Release Notes#" not in notes
    gpus = (snapshot / "text" / "nvidia" / "gpus.md").read_text()
    assert 'version: "as of ' in gpus and 'channel: "web, dated"' in gpus
    assert gpus.split("---\n", 2)[2].startswith("# GPUs\n") and "| 8.6 | A40 |" in gpus
    record = json.loads((snapshot / "library.json").read_text())
    assert record["versions"] == {"cuda": [{"version": "12.5.0", "seen": "given with --version-of"}]}
    pages = {source["name"]: source for source in record["sources"] if source["kind"] == "page"}
    assert pages["release-notes"]["version_from"] == "given with --version-of"
    assert pages["release-notes"]["converter"] == 'html 1, main 1 (itemprop="articleBody")'
    assert pages["gpus"]["converter"] == "html 1, main 1 (<main>)"
    assert pages["gpus"]["version"] is None and pages["gpus"]["channel"] == "web, dated"
    raw = (snapshot / pages["gpus"]["raw"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == pages["gpus"]["sha256"]
    assert record["problems"] == []


def test_the_newest_cuda_comes_from_the_address_without_a_version_if_its_title_says_so(
        host, monkeypatch):
    asked = with_web(host, monkeypatch, {
        CURRENT: page("CUDA Toolkit 13.5 - Release Notes", "<p>Newest.</p>"),
        GPUS: page("GPUs", "<p>Table.</p>"),
    })
    assert run(host, "--version-of", "cuda=13.5") == 0
    assert asked[:2] == [ARCHIVE.format("13.5.0"), CURRENT]
    record = json.loads((host["root"] / f"test-{DAY}" / "library.json").read_text())
    (notes,) = [source for source in record["sources"] if source.get("name") == "release-notes"]
    assert notes["url"] == CURRENT and notes["version"] == "13.5.0"


def test_a_page_of_another_version_is_a_problem_and_the_rest_is_read(host, monkeypatch, capsys):
    with_web(host, monkeypatch, {
        # The address without a version holds a newer release by now.
        CURRENT: page("CUDA Toolkit 13.6 - Release Notes", "<p>Newer.</p>"),
        GPUS: page("GPUs", "<p>Table.</p>"),
    })
    assert run(host, "--version-of", "cuda=13.5") == 0
    snapshot = host["root"] / f"test-{DAY}"
    assert not (snapshot / "text" / "cuda").exists()
    assert (snapshot / "text" / "nvidia" / "gpus.md").is_file()
    (problem,) = json.loads((snapshot / "library.json").read_text())["problems"]
    assert problem.startswith("cuda 13.5.0 release-notes: no page states 13.5 at ")


def test_an_archive_address_redirected_elsewhere_is_not_its_version(host, monkeypatch):
    with_web(host, monkeypatch, {
        ARCHIVE.format("12.5.0"): (page("Release Notes", "<p>Newest.</p>"), CURRENT),
        CURRENT: page("CUDA Toolkit 13.5 - Release Notes", "<p>Newest.</p>"),
        GPUS: page("GPUs", "<p>Table.</p>"),
    })
    assert run(host, "--version-of", "cuda=12.5") == 0
    record = json.loads((host["root"] / f"test-{DAY}" / "library.json").read_text())
    assert record["problems"] == [
        f"cuda 12.5.0 release-notes: no page states 12.5 at {ARCHIVE.format('12.5.0')} or "
        f"{CURRENT}; it is left out"]


def test_without_a_version_found_a_sources_pages_are_left_out_with_a_note(host, monkeypatch,
                                                                        capsys):
    asked = with_web(host, monkeypatch, {GPUS: page("GPUs", "<p>Table.</p>")})
    assert run(host, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert asked == []
    assert "nvidia            as they stand on the day of the build: gpus" in printed
    assert ("cuda: no cuda version found on this host, so its pages are left out; "
            "--version-of cuda=VERSION reads them at a version") in printed
    assert "Problems: none" in printed


def test_the_cuda_versions_are_found_where_this_site_keeps_them(host, monkeypatch, capsys):
    with_web(host, monkeypatch, {})
    executable(host["tmp"] / "bin" / "nvcc",
               'echo "Cuda compilation tools, release 12.5, V12.5.82"\n')
    monkeypatch.setattr(bl, "module_names",
                        lambda: ["cuda/11.7", "CUDA/12.5.1", "cudnn/8.9-cuda12", "relion/v5"])
    monkeypatch.setattr(bl, "torch_cuda", lambda: "13.2")
    monkeypatch.setattr(bl, "cupy_cuda", lambda tool: "12.9")
    assert run(host, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert ("cuda              11.7.0 (module cuda/11.7), 12.5.0 (nvcc on PATH, module "
            "CUDA/12.5.1), 12.9.0 (pytom-match-pick's CuPy), 13.2.0 (the pipeline's PyTorch): "
            "release-notes") in printed


def test_cupy_is_read_at_the_version_pytom_match_picks_environment_has(host, monkeypatch):
    executable(host["tmp"] / "pytom" / "bin" / "python3", """\
        case "$3" in
          *"import cupy"*) echo 12090 ;;
          *) case "$4" in pytom-match-pick) echo 0.14.0 ;; cupy) echo 14.2.0 ;; *) exit 3 ;; esac ;;
        esac
        """)
    tools = bl.find_tools(bl.read_sources(host["sources"]), [], {})
    assert bl.cupy_versions(tools, []) == [("14.2.0", "cupy in pytom-match-pick's environment")]
    assert bl.cupy_cuda(tools["pytom-match-pick"]) == "12.9"
    assert bl.cupy_versions(tools, ["13.6.0"]) == [("13.6.0", "given with --version-of")]


@pytest.mark.parametrize("html, label, kept, left", [
    # The inner div's end is not the article's: C, after it, is the page's own.
    ('<nav>M</nav><div itemprop="articleBody"><p>A</p><div><p>B</p></div><br/><p>C</p></div>'
     '<p>Z</p>', 'itemprop="articleBody"', ["A", "B", "C"], ["M", "Z"]),
    ("<nav>M</nav><main><article><p>A</p></article><aside>S</aside></main>",
     "<article>", ["A"], ["M", "S"]),
    ('<div role="main"><p>A</p></div><footer>F</footer>', 'role="main"', ["A"], ["F"]),
    ("<p>A</p><p>B</p>", "the whole page", ["A", "B"], []),
    ("<nav>M</nav><article><p>A</p>", "<article>", ["A"], ["M"]),
])
def test_a_pages_own_text_is_told_from_the_sites_frame(html, label, kept, left):
    part, found = bl.main_text(f"<html><body>\n{html}\n</body></html>")
    assert found == label
    markdown = bl.html_to_markdown(part)
    assert all(word in markdown for word in kept) and not any(word in markdown for word in left)


@pytest.mark.parametrize("web, said", [
    ('[web.cuda]\nversions = "cuda"\n[web.cuda.pages]\na = "https://docs.nvidia.com/x"\n',
     "its first address must hold {version}"),
    ('[web.x.pages]\na = "https://docs.nvidia.com/{version}/x"\n',
     "{version} in an address needs versions"),
    ('[web.x.pages]\na = "http://docs.nvidia.com/x"\n', "must be an https address"),
    ('[web.x.pages]\na = "https://example.org/x"\n', "must be an https address"),
    ('[web.x]\nversions = "torch"\n[web.x.pages]\na = "https://pytorch.org/{version}"\n',
     "versions must be one of cuda, cupy, conda, mamba"),
    ('[web.conda.pages]\na = "https://docs.conda.io/x"\n', "not a tool's"),
    ('[web.imod.pages]\na = "https://docs.nvidia.com/x"\n', "not a tool's"),
    ('[web.x.pages]\nA_B = "https://docs.nvidia.com/x"\n', "a page's name is"),
    ('[web.x]\nurl = "https://docs.nvidia.com/x"\n', "needs pages"),
])
def test_a_web_source_must_be_well_formed(tmp_path, web, said):
    sources = tmp_path / "sources.toml"
    sources.write_text(web)
    with pytest.raises(SystemExit, match=re.escape(said)):
        bl.read_web(sources)


def test_the_web_pages_of_the_list_of_sources_are_well_formed():
    web = bl.read_web(bl.SOURCES)
    assert set(web) == {"cuda", "nvidia", "cupy", "pytorch", "conda-docs", "mamba-docs"}
    assert web["cuda"]["versions"] == "cuda" and web["cupy"]["versions"] == "cupy"
    assert web["conda-docs"]["versions"] == "conda" and web["mamba-docs"]["versions"] == "mamba"
    # mamba's files at the tag of a 2.x release, then at that of a 1.x one.
    for addresses in web["mamba-docs"]["pages"].values():
        assert [url.split("/")[5] for url in addresses] == ["{version}", "micromamba-{version}"]
    assert "versions" not in web["nvidia"] and "versions" not in web["pytorch"]
    # The tools' part is read as before, without the pages.
    assert "web" not in bl.read_sources(bl.SOURCES)


def test_versions_are_given_to_tools_and_to_pages_apart():
    assert bl.parse_given(["ctffind=5.0.2", "cuda=12.5", "cuda=13.2", "cupy=14.2.0"]) == (
        {"ctffind": "5.0.2"}, {"cuda": ["12.5", "13.2"], "cupy": ["14.2.0"]})
    # conda and mamba are tools; their pages are read at the tool's version.
    assert bl.parse_given(["conda=26.3.2", "mamba=2.5.0"]) == (
        {"conda": "26.3.2", "mamba": "2.5.0"}, {})
    with pytest.raises(SystemExit, match="NAME one of"):
        bl.parse_given(["torch=2.15.0"])


@pytest.fixture
def web_server():
    """A web server on 127.0.0.1: /page answers, /moved redirects to /page,
    /missing is 404, /broken is 500."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/moved":
                self.send_response(301)
                self.send_header("Location", "/page")
                self.end_headers()
                return
            code = {"/page": 200, "/missing": 404}.get(self.path, 500)
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b"<title>Page</title>" if code == 200 else b"no")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_page_the_site_does_not_have_is_none_and_any_other_failure_stops(web_server):
    assert bl.fetch_page(f"{web_server}/page") == (b"<title>Page</title>", f"{web_server}/page")
    assert bl.fetch_page(f"{web_server}/moved")[1] == f"{web_server}/page"
    assert bl.fetch_page(f"{web_server}/missing") is None
    with pytest.raises(SystemExit, match="could not fetch .*/broken"):
        bl.fetch_page(f"{web_server}/broken")


def test_a_gpu_nodes_report_goes_into_the_site_card(host, monkeypatch, capsys):
    report = host["tmp"] / "gpu-node.txt"
    report.write_text("| NVIDIA-SMI 610.57.04   Driver Version: 610.57.04   CUDA Version: 13.3 |\n")
    monkeypatch.setenv("PATH", os.environ["PATH"])
    assert run(host, "--dry-run") == 0
    assert "the site card has no GPU: there is no nvidia-smi on this host" in capsys.readouterr().out
    assert run(host, "--gpu-node-report", str(report)) == 0
    snapshot = host["root"] / f"test-{DAY}"
    card = (snapshot / "site.md").read_text()
    assert "## A GPU node" in card and "Driver Version: 610.57.04" in card
    record = json.loads((snapshot / "library.json").read_text())
    (given,) = [source for source in record["sources"] if source.get("name") == "gpu node"]
    assert given["given"] == str(report) and (snapshot / given["raw"]).is_file()
    assert not any("no nvidia-smi" in note for note in record["notes"])
    with pytest.raises(SystemExit, match="no such file"):
        run(host, "--gpu-node-report", str(host["tmp"] / "absent.txt"))


# --- the environment managers ---------------------------------------------------------

MANAGERS = """\
[conda]
usage = [["--help"]]
[mamba]
usage = [["create", "--help"]]
"""

CONDA_PAGE = "https://docs.conda.io/projects/conda/en/{}/user-guide/tasks/manage-virtual.html"
MAMBA_FILE = ("https://raw.githubusercontent.com/mamba-org/mamba/{}/docs/source/user_guide/"
              "micromamba.rst")
MANAGER_PAGES = f"""\
[web.conda-docs]
versions = "conda"
[web.conda-docs.pages]
virtual-packages = "{CONDA_PAGE.format("{version}")}"
[web.mamba-docs]
versions = "mamba"
[web.mamba-docs.pages]
micromamba = [
    "{MAMBA_FILE.format("{version}")}",
    "{MAMBA_FILE.format("micromamba-{version}")}",
]
"""


def managers(host, monkeypatch, mamba_says: str = "2.5.0") -> dict[str, Path]:
    """conda, micromamba and mamba as stand-ins: conda in a folder its shell
    init would name, the two mambas on PATH."""
    bin_ = host["tmp"] / "bin"
    conda = executable(host["tmp"] / "miniforge" / "bin" / "conda", """\
        if [ "$1" = --version ]; then echo "conda 26.3.2"; else echo "usage: conda [-h]"; fi
        """)
    said = mamba_says.replace("\n", "\\n")
    found = {"conda": conda}
    for name in ("micromamba", "mamba"):
        found[name] = executable(bin_ / name, f"""\
            if [ "$1" = --version ]; then printf '{said}\\n'; else echo "$0 $*"; fi
            """)
    with host["sources"].open("a") as sources:
        sources.write(MANAGERS)
    return found


def test_the_environment_managers_are_found_as_their_shell_init_names_them(host, monkeypatch):
    found = managers(host, monkeypatch)
    tools = bl.find_tools(bl.read_sources(host["sources"]), [], {})
    # No $CONDA_EXE and no conda on PATH; micromamba before mamba, as
    # scripts/activate_env.sh activates with it.
    assert tools["conda"].path is None and "$CONDA_EXE is not set" in tools["conda"].note
    assert (tools["mamba"].path, tools["mamba"].found) == (found["micromamba"],
                                                          "micromamba on PATH")
    assert tools["mamba"].version_text == "2.5.0"
    monkeypatch.setenv("CONDA_EXE", str(found["conda"]))
    monkeypatch.setenv("MAMBA_EXE", str(found["mamba"]))
    tools = bl.find_tools(bl.read_sources(host["sources"]), [], {})
    assert (tools["conda"].path, tools["conda"].found) == (found["conda"], "$CONDA_EXE")
    assert tools["conda"].version_text == "26.3.2"
    assert (tools["mamba"].path, tools["mamba"].found) == (found["mamba"], "$MAMBA_EXE")
    monkeypatch.setenv("MAMBA_EXE", str(host["tmp"] / "gone" / "micromamba"))
    assert "is not a file" in bl.find_tools(bl.read_sources(host["sources"]), [], {})[
        "mamba"].problem


@pytest.mark.parametrize("said, version", [
    ("2.5.0", "2.5.0"),
    ("mamba 1.5.8\nconda 24.1.2", "1.5.8"),
    ("micromamba 1.5.9", "1.5.9"),
    ("critical libmamba Shell not initialized", None),
])
def test_mamba_states_its_version_alone_or_after_its_name(tmp_path, said, version):
    program = executable(tmp_path / "micromamba", f"printf '{said}\\n'\n")
    assert bl.mamba_version(program).value == version


def test_conda_and_mamba_guides_are_read_at_their_versions(host, monkeypatch, capsys):
    found = managers(host, monkeypatch)
    monkeypatch.setenv("CONDA_EXE", str(found["conda"]))
    rst = b"Micromamba User Guide\r\n=====================\r\n\r\n``micromamba`` is tiny.\r\n"
    with_web(host, monkeypatch, {
        CONDA_PAGE.format("26.3.x"): b"<html><head><title>Managing virtual packages</title>"
                                     b"</head><body><nav>Menu</nav><article><h1>Managing "
                                     b"virtual packages</h1><p><code>CONDA_OVERRIDE_CUDA</code>"
                                     b"</p></article></body></html>",
        MAMBA_FILE.format("2.5.0"): rst,
    }, web=MANAGER_PAGES)
    assert run(host, "--dry-run") == 0
    printed = capsys.readouterr().out
    assert re.search(r"conda-docs +26\.3\.x \(conda 26\.3\.2, \$CONDA_EXE\): virtual-packages",
                     printed)
    assert re.search(r"mamba-docs +2\.5\.0 \(micromamba 2\.5\.0, micromamba on PATH\): "
                     r"micromamba\n", printed)
    assert run(host) == 0
    snapshot = host["root"] / f"test-{DAY}"
    virtual = (snapshot / "text" / "conda-docs" / "26.3.x" / "virtual-packages.md").read_text()
    assert 'tool: "conda-docs"\nversion: "26.3.x"\nchannel: "web, by version"' in virtual
    assert "`CONDA_OVERRIDE_CUDA`" in virtual and "Menu" not in virtual
    # read_bytes: read_text would turn a \r\n kept in the text into \n.
    guide = (snapshot / "text" / "mamba-docs" / "2.5.0" / "micromamba.md").read_bytes().decode()
    assert guide.split("---\n", 2)[2] == (
        "# micromamba.rst\n\n```text\nMicromamba User Guide\n=====================\n\n"
        "``micromamba`` is tiny.\n```\n")
    assert (snapshot / "raw" / "mamba-docs" / "2.5.0" / "micromamba.rst").read_bytes() == rst
    assert (snapshot / "text" / "conda" / "help.md").is_file()
    create = (snapshot / "text" / "mamba" / "create-help.md").read_text()
    assert "micromamba create --help" in create
    record = json.loads((snapshot / "library.json").read_text())
    assert record["versions"]["mamba"] == [
        {"version": "2.5.0", "seen": "micromamba 2.5.0, micromamba on PATH"}]
    pages = {source["name"]: source for source in record["sources"] if source["kind"] == "page"}
    assert pages["micromamba"]["converter"] == "text 1"
    assert pages["virtual-packages"]["converter"] == "html 1, main 1 (<article>)"
    assert record["problems"] == []


def test_mamba_1_is_read_at_its_micromamba_tag_and_conda_at_a_given_version(host, monkeypatch):
    managers(host, monkeypatch, mamba_says="mamba 1.5.9\nconda 24.1.2")
    asked = with_web(host, monkeypatch, {
        CONDA_PAGE.format("25.1.x"): b"<html><body><article><p>Virtual.</p></article></body>"
                                     b"</html>",
        MAMBA_FILE.format("micromamba-1.5.9"): b"Micromamba User Guide\n",
    }, web=MANAGER_PAGES)
    # No conda here: the version given is the one its pages are read at.
    assert run(host, "--version-of", "conda=25.1.2") == 0
    assert asked == [CONDA_PAGE.format("25.1.x"), MAMBA_FILE.format("1.5.9"),
                     MAMBA_FILE.format("micromamba-1.5.9")]
    snapshot = host["root"] / f"test-{DAY}"
    assert (snapshot / "text" / "mamba-docs" / "1.5.9" / "micromamba.md").is_file()
    record = json.loads((snapshot / "library.json").read_text())
    assert record["versions"]["conda"] == [
        {"version": "25.1.x", "seen": "given with --version-of"}]
    assert not (snapshot / "text" / "conda").exists()

