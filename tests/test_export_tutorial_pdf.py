"""scripts/export_tutorial_pdf.py: links a PDF can follow, and no command that the
page would wrap. The browser is not run here."""

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import export_tutorial_pdf as export  # noqa: E402

TUTORIALS = sorted((ROOT / "docs" / "tutorials").glob("*.md"))


def _tutorial(tmp_path, monkeypatch, name, text):
    """A Markdown file inside a stand-in repository root, as the script sees one."""
    monkeypatch.setattr(export, "ROOT", tmp_path)
    folder = tmp_path / "docs" / "tutorials"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text)
    return path


def test_links_point_to_what_a_pdf_can_reach(tmp_path, monkeypatch):
    other = _tutorial(tmp_path, monkeypatch, "b.md", "# B\n\n## Step two\n")
    page = _tutorial(tmp_path, monkeypatch, "a.md", (
        "# A tutorial\n\n## Where to go next\n\n"
        "See [below](#where-to-go-next), [step two](b.md#step-two), "
        "[the README](../../README.md#setup), [`configs/lanes/example.toml`]"
        "(../../configs/lanes/example.toml) and [the site](https://example.org).\n"
    ))

    html = export.to_html(page, {page.resolve(), other.resolve()}, "stamp")

    assert '<h2 id="where-to-go-next">' in html
    assert '<a href="#where-to-go-next">below</a>' in html
    # The other tutorial is exported beside this one: its PDF, not its Markdown.
    assert '<a href="b.pdf">step two</a>' in html
    # A repository file is named by its path, once, and is no link.
    assert 'the README</span><span class="path"> (README.md)</span>' in html
    assert "<code>configs/lanes/example.toml</code></span>" in html
    assert "(configs/lanes/example.toml)" not in html
    assert not re.search(r'href="[^"]*\.md', html)
    assert '<a href="https://example.org">the site</a>' in html
    assert '<p class="stamp">stamp</p>' in html


def test_a_link_to_a_file_not_exported_is_its_path(tmp_path, monkeypatch):
    page = _tutorial(tmp_path, monkeypatch, "a.md", "# A\n\n[B](b.md)\n")

    html = export.to_html(page, {page.resolve()}, "stamp")

    assert '<span class="path"> (docs/tutorials/b.md)</span>' in html
    assert "b.pdf" not in html


def test_a_code_line_the_page_would_wrap_is_named():
    text = "x\n```bash\nshort\n" + "y" * (export.MAX_CODE_LINE + 1) + "\n```\n" + "z" * 200 + "\n"

    # Prose may be as long as it likes: the page wraps it harmlessly.
    assert export.long_code_lines(text) == [(4, export.MAX_CODE_LINE + 1)]


def test_main_refuses_a_long_command_before_printing(tmp_path, monkeypatch, capsys):
    page = _tutorial(tmp_path, monkeypatch, "a.md",
                     "# A\n\n```bash\n" + "rsync " + "x" * 100 + "\n```\n")
    monkeypatch.setattr(export, "find_browser", lambda given: (_ for _ in ()).throw(
        AssertionError("no browser before the check")))

    assert export.main([str(page), "--out", str(tmp_path / "out")]) == 2
    assert "docs/tutorials/a.md:4: a code line of 106 characters" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_html_only_writes_a_page_per_file(tmp_path, monkeypatch):
    page = _tutorial(tmp_path, monkeypatch, "a.md", "# A\n\ntext\n")

    assert export.main([str(page), "--out", str(tmp_path / "out"), "--html-only"]) == 0
    written = (tmp_path / "out" / "a.html").read_text()
    assert "<title>A</title>" in written and "docs/tutorials/a.md" in written


def test_the_stamp_says_a_file_is_not_committed(tmp_path, monkeypatch):
    page = _tutorial(tmp_path, monkeypatch, "a.md", "# A\n")

    assert export.stamp_for(page, date(2026, 11, 10)) == (
        "cryoet-pipeline · docs/tutorials/a.md · not committed · exported 10 November 2026"
    )


def test_the_tutorials_in_the_repository_export():
    assert TUTORIALS
    exported = {path.resolve() for path in TUTORIALS}
    for path in TUTORIALS:
        assert export.long_code_lines(path.read_text()) == [], path.name
        html = export.to_html(path, exported, "stamp")
        assert not re.search(r'href="(?!https?:|#)[^"]*\.md', html), path.name
        for anchor in re.findall(r'href="#([^"]+)"', html):
            assert f'id="{anchor}"' in html, (path.name, anchor)
