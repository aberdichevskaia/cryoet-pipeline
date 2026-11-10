#!/usr/bin/env python3
"""Print the tutorials in docs/tutorials/ as A4 PDFs, for handing out.

    python3 scripts/export_tutorial_pdf.py                     every docs/tutorials/*.md
    python3 scripts/export_tutorial_pdf.py docs/tutorials/ryr1-with-assistant.md
    python3 scripts/export_tutorial_pdf.py --out outputs/tutorials --html-only

Each Markdown file becomes <out>/<name>.pdf (default out: outputs/tutorials, which
git leaves out). The Markdown is rendered to HTML with markdown-it-py, which every
pipeline environment has through typer's rich, and printed by a headless Chrome,
Chromium or Edge: the Mac's own, or the one --chrome names. --html-only writes
<out>/<name>.html instead and needs no browser.

A PDF cannot follow a link into the repository, so links are rewritten:
- an anchor in the same file stays a link, to the heading of that name;
- a link to another file being exported points to its PDF, beside this one;
- a link to any other file becomes its path from the repository root, which the
  reader has in the clone.

A line of a code block that is too long to fit the page is refused, naming the file
and the line: the PDF would wrap it, and a command copied from it would arrive in
two pieces, the second run as a command of its own. Split it with a backslash. Each
PDF's first line says which file and commit it was made from, and whether the file
had uncommitted changes.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import html
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"
# The longest code line that fits the page at the code's size, in a list item too.
# Lines of 95 characters wrapped in the first export (10 November 2026).
MAX_CODE_LINE = 88
BROWSERS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
BROWSERS_ON_PATH = ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")

CSS = """
@page { size: A4; margin: 17mm 16mm 18mm 16mm;
  @bottom-center { content: counter(page) " / " counter(pages);
    font: 8.5pt -apple-system, Helvetica, Arial, sans-serif; color: #777; } }
:root { --ink: #1d1f23; --muted: #5d636d; --rule: #d8dbe0; --code-bg: #f4f5f7; --accent: #1f5fa8; }
html { font: 10.2pt/1.5 -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif;
  color: var(--ink); background: #fff; }
body { margin: 0; }
.stamp { color: var(--muted); font-size: 8.5pt; margin: 0 0 14pt;
  border-bottom: 1px solid var(--rule); padding-bottom: 6pt; }
h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 4pt; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt; padding-top: 4pt; border-top: 1px solid var(--rule);
  break-after: avoid; }
h3 { font-size: 11.5pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin: 2pt 0; }
li > p { margin: 0 0 4pt; }
a { color: var(--accent); text-decoration: none; }
.path { color: var(--muted); font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 8.3pt; }
code { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 8.6pt;
  background: var(--code-bg); padding: 0.5pt 2.5pt; border-radius: 2.5pt; }
pre { background: var(--code-bg); border: 1px solid #e3e5e9; border-radius: 4pt; padding: 6pt 8pt;
  margin: 4pt 0 9pt; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.45; }
blockquote { margin: 6pt 0 10pt; padding: 6pt 10pt; border-left: 3pt solid var(--accent);
  background: #f1f5fb; break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 9pt; }
th, td { border: 1px solid var(--rule); padding: 3.5pt 5pt; vertical-align: top; text-align: left; }
th { background: #eef0f3; }
tr { break-inside: avoid; }
td code, th code { font-size: 8.2pt; }
"""


class ExportError(Exception):
    """What stops an export, said to the person."""


def slug(text: str) -> str:
    """A heading's anchor as GitHub makes it: lower case, punctuation dropped,
    spaces as hyphens."""
    text = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return text.replace(" ", "-")


def long_code_lines(text: str, limit: int = MAX_CODE_LINE) -> list[tuple[int, int]]:
    """(line number, length) of every line inside a fenced code block longer than
    limit."""
    found, fenced = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced and len(line) > limit:
            found.append((number, len(line)))
    return found


def _link_text(children, start: int) -> tuple[str, int]:
    """The plain text of the link opened at children[start], and its close's index."""
    depth, text = 0, []
    for index in range(start + 1, len(children)):
        token = children[index]
        if token.type == "link_open":
            depth += 1
        elif token.type == "link_close":
            if depth == 0:
                return "".join(text), index
            depth -= 1
        else:
            text.append(token.content)
    raise ExportError("a link without its end")


def to_html(source: Path, exported: set[Path], stamp: str) -> str:
    """source as one HTML page, its links rewritten as the module says. exported:
    the files whose PDFs are made beside this one."""
    try:
        from markdown_it import MarkdownIt
        from markdown_it.token import Token
    except ImportError as error:
        raise ExportError(
            "markdown-it-py is missing: activate the cryoet-pipeline environment"
        ) from error

    text = source.read_text()
    here = source.resolve().relative_to(ROOT).as_posix()
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = md.parse(text)
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            token.attrSet("id", slug(tokens[index + 1].content))
        if token.type != "inline" or not token.children:
            continue
        children = token.children
        index = 0
        while index < len(children):
            child = children[index]
            href = str(child.attrGet("href") or "") if child.type == "link_open" else ""
            if not href or href.startswith(("http://", "https://", "mailto:", "#")):
                index += 1
                continue
            path, _, _anchor = href.partition("#")
            target = (source.parent / path).resolve()
            if target in exported:
                child.attrSet("href", target.with_suffix(".pdf").name)
                index += 1
                continue
            try:
                shown_path = target.relative_to(ROOT).as_posix()
            except ValueError:
                shown_path = posixpath.normpath(posixpath.join(posixpath.dirname(here), path))
            words, close = _link_text(children, index)
            opened = Token("html_inline", "", 0)
            opened.content = '<span class="repo-link">'
            closed = Token("html_inline", "", 0)
            closed.content = "</span>" + (
                "" if words.strip("` ") == shown_path
                else f'<span class="path"> ({html.escape(shown_path)})</span>'
            )
            children[index], children[close] = opened, closed
            index = close + 1
    body = md.renderer.render(tokens, md.options, {})
    title = re.search(r"^# (.+)$", text, re.M)
    name = html.escape(title.group(1) if title else source.stem)
    body = body.replace("</h1>", f'</h1>\n<p class="stamp">{html.escape(stamp)}</p>', 1)
    return (
        f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        f"<title>{name}</title>\n<style>{CSS}</style></head>\n<body>\n{body}</body></html>\n"
    )


def stamp_for(source: Path, today: date | None = None) -> str:
    """Which file, which commit and which day a PDF was made from."""
    here = source.resolve().relative_to(ROOT).as_posix()

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = git("log", "-1", "--format=%h", "--", here)
    changed = git("status", "--porcelain", "--", here)
    if not commit:
        version = "not committed"
    elif changed:
        version = f"{commit} with uncommitted changes"
    else:
        version = commit
    day = today or date.today()
    return f"cryoet-pipeline · {here} · {version} · exported {day.day} {day:%B %Y}"


def find_browser(given: str | None) -> str:
    if given:
        if not Path(given).is_file() and not shutil.which(given):
            raise ExportError(f"--chrome {given}: no such program")
        return given
    for path in BROWSERS:
        if Path(path).is_file():
            return path
    for name in BROWSERS_ON_PATH:
        if found := shutil.which(name):
            return found
    raise ExportError(
        "no Chrome, Chromium or Edge found to print with: give one with --chrome, or "
        "write the HTML with --html-only and print it from a browser"
    )


def print_pdf(browser: str, page: Path, pdf: Path) -> None:
    """page printed to pdf by a headless browser, in a profile of its own so that
    the person's browser is left alone. An earlier PDF of the same name is removed
    first, so that a failed print cannot pass for a new one."""
    pdf.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="export-tutorial-") as profile:
        result = subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={profile}", f"--print-to-pdf={pdf}", page.resolve().as_uri()],
            capture_output=True, text=True, timeout=120,
        )
    if not pdf.is_file() or pdf.stat().st_size == 0:
        said = (result.stderr or result.stdout).strip().splitlines()[-3:]
        raise ExportError(f"{browser} wrote no PDF for {page.name}: " + " / ".join(said))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="*", type=Path,
                        help="Markdown files; default every docs/tutorials/*.md")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "tutorials",
                        help="folder for the PDFs (default outputs/tutorials)")
    parser.add_argument("--chrome", default=None, help="the browser to print with")
    parser.add_argument("--html-only", action="store_true",
                        help="write the HTML pages and print nothing")
    args = parser.parse_args(argv)

    files = [path.resolve() for path in args.files] or sorted(TUTORIALS.glob("*.md"))
    try:
        if not files:
            raise ExportError(f"no Markdown files in {TUTORIALS}")
        problems = []
        for path in files:
            if not path.is_file():
                problems.append(f"{path}: no such file")
                continue
            try:
                path.relative_to(ROOT)
            except ValueError:
                problems.append(f"{path}: outside the repository")
                continue
            for number, length in long_code_lines(path.read_text()):
                problems.append(
                    f"{path.relative_to(ROOT)}:{number}: a code line of {length} characters, "
                    f"over {MAX_CODE_LINE}; the PDF would wrap it and a copied command would "
                    "break. Split it with a backslash"
                )
        if problems:
            raise ExportError("\n".join(problems))
        browser = None if args.html_only else find_browser(args.chrome)
        args.out.mkdir(parents=True, exist_ok=True)
        exported = set(files)
        for path in files:
            page_text = to_html(path, exported, stamp_for(path))
            if browser is None:
                page = args.out / f"{path.stem}.html"
                page.write_text(page_text)
                print(page)
                continue
            with tempfile.TemporaryDirectory(prefix="export-tutorial-html-") as folder:
                page = Path(folder) / f"{path.stem}.html"
                page.write_text(page_text)
                pdf = args.out / f"{path.stem}.pdf"
                print_pdf(browser, page, pdf)
            print(pdf)
    except ExportError as error:
        print(f"export_tutorial_pdf.py: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
