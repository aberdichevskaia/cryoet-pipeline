#!/usr/bin/env python3
"""Build a snapshot of the assistant's reference library on this site.

    python3 scripts/build_library.py --site mdc --dry-run
    python3 scripts/build_library.py --site mdc --root /fast/AG_Kudryashev/cryoet-library

docs/library.md says what the library is and why it is kept this way. This
script builds its first part, the installation channel: what this site's own
installation says, at the version installed here. Web sources come later.

The sources are listed in configs/library.toml. The script finds each tool the
way the pipeline does:
1. from the lane configs given with --config;
2. otherwise from the variable the lanes name the tool by;
3. otherwise from PATH.

Then it asks each tool for its version. A version it cannot find stays
"unknown", unless --version-of gives it. --dry-run prints what was found and
what would be written, and writes nothing.

A snapshot is the folder <root>/<site>-<date>/. A second build on the same day
gets a letter after the date. It holds:

    library.json   what was read, from where, at which version, with sha256
    site.md        the site card: host, tool versions, partitions, GPUs, modules
    text/<tool>/   each document as markdown, under a header naming its source
    raw/<tool>/    the originals the text was made from

The snapshot is written under a hidden name and then made read-only: owner and
group may read, nobody may write. Only after that does it get its own name. A
failed build therefore leaves nothing behind, and a finished snapshot never
changes. <root>/current then points at it.

The root is not created here. It is the group's folder, made once, with group
read and setgid (docs/library.md, "Where snapshots live").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import sys
import tomllib
import zlib
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

from cryoet_pipeline.tool_versions import (
    Version,
    bash_version,
    ctffind_version,
    imod_version,
    package_version,
    probe,
    script_python,
    slurm_version,
    stated_version,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "configs" / "library.toml"
# Where the IMOD programs the pipeline runs are named, for the coverage check.
SRC = ROOT / "src"
SCHEMA = 1
# A converter's number goes up whenever its output for the same input changes.
CONVERTERS = {"html": 1, "terminal": 1}
SITE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# The tools this script knows how to find and ask. configs/library.toml may name
# only these.
KNOWN = ("imod", "ctffind", "motioncor3", "pytom-match-pick", "membrain-seg", "slurm", "bash")

# The lane config keys that name a tool's executable. The first key of a tool is
# the one its path is taken from.
LANE_EXECUTABLES: dict[tuple[str, str], str] = {
    ("executables", "ctffind"): "ctffind",
    ("executables", "motioncor3"): "motioncor3",
    ("executables", "pytom_match_template"): "pytom-match-pick",
    ("executables", "pytom_create_template"): "pytom-match-pick",
    ("executables", "pytom_create_mask"): "pytom-match-pick",
    ("executables", "pytom_extract_candidates"): "pytom-match-pick",
    ("segment", "executable"): "membrain-seg",
}
# When no lane config names a tool: the variable the lane configs name it by,
# then its names on PATH. pytom-match-pick's variable is the folder its scripts
# are in.
LOOKUP: dict[str, tuple[str, tuple[str, ...]]] = {
    "ctffind": ("CTFFIND5", ("ctffind5", "ctffind")),
    "motioncor3": ("MOTIONCOR3", ("MotionCor3",)),
    "pytom-match-pick": ("PYTOMO_MP_DIR", ("pytom_match_template.py",)),
    "membrain-seg": ("MEMBRAIN_EXE", ("membrain",)),
}
DISTRIBUTIONS = {"pytom-match-pick": "pytom-match-pick", "membrain-seg": "membrain-seg"}

# The site card's questions. Each one's output is kept as it was printed.
SITE_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("uname", ("uname", "-a")),
    ("slurm", ("sinfo", "-V")),
    ("partitions", ("sinfo", "-h", "-o", "%P %a %l %D %G")),
    ("gpus", ("nvidia-smi",)),
    ("nvcc", ("nvcc", "--version")),
    ("modules", ("bash", "-lc", "module -t avail 2>&1")),
)
# The same output whoever runs the build, in whatever terminal.
PROBE_ENV = {
    "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "100",
    "MANPAGER": "cat", "PAGER": "cat", "MANWIDTH": "100",
}
USAGE_SECONDS = 60.0
USAGE_LIMIT = 1 << 20
MAN_LIMIT = 8 << 20


@dataclass
class Lane:
    path: Path
    cfg: dict[str, Any]


@dataclass
class Tool:
    name: str
    path: Path | None = None
    """The executable, or IMOD's folder."""
    found: str = ""
    """How the path was found."""
    version: Version | None = None
    problem: str | None = None
    """Set when the tool is named here but cannot be used."""
    note: str | None = None
    """Set when the tool is simply not on this host."""

    @property
    def version_text(self) -> str:
        return self.version.value if self.version and self.version.value else "unknown"


@dataclass
class Item:
    tool: str
    kind: str
    """"manual", "man" or "usage"."""
    name: str
    """Its file name under text/<tool>/, without .md."""
    command: tuple[str, ...] = ()
    page: str = ""


@dataclass
class Snapshot:
    id: str
    site: str
    built: str
    host: str
    tools: dict[str, Tool]
    records: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    documents: int = 0


# --- finding the tools ------------------------------------------------------

def find_tools(sources: dict[str, Any], lanes: list[Lane],
               given: dict[str, str]) -> dict[str, Tool]:
    """Every tool configs/library.toml names, found and asked for its version."""

    tools: dict[str, Tool] = {}
    for name in sources:
        if name == "imod":
            tool = find_imod()
        elif name == "slurm":
            tool = on_path("slurm", "sinfo")
        elif name == "bash":
            tool = on_path("bash", "bash")
        else:
            tool = find_named(name, lanes)
        if tool.path is not None:
            tool.version = ask_version(tool)
        if name in given:
            said = tool.version.value if tool.version else None
            how = "given with --version-of"
            if said and said != given[name]:
                how += f"; the tool itself states {said}"
            tool.version = Version(given[name], how, tool.version.output if tool.version else "")
        tools[name] = tool
    return tools


def find_imod() -> Tool:
    """IMOD's folder, from where the backends would find its programs
    (cryoet_pipeline.backends.alignment.resolve_imod_executable): newstack on
    PATH, then $IMOD_DIR/bin, then /Applications/IMOD/bin. The first folder
    that has a VERSION file wins."""

    candidates: list[tuple[Path, str]] = []
    if found := shutil.which("newstack"):
        candidates.append((Path(found), "newstack on PATH"))
    if imod_dir := os.environ.get("IMOD_DIR"):
        candidates.append((Path(imod_dir) / "bin" / "newstack", "$IMOD_DIR"))
    candidates.append((Path("/Applications/IMOD/bin/newstack"), "/Applications/IMOD"))
    present = [(path.resolve().parent.parent, how) for path, how in candidates if path.is_file()]
    for folder, how in present:
        if (folder / "VERSION").is_file():
            return Tool("imod", folder, how)
    if present:
        folder, how = present[0]
        return Tool("imod", folder, how, problem=f"{folder} has no VERSION file")
    return Tool("imod", note="not on this host: no newstack on PATH, under $IMOD_DIR/bin "
                             "or under /Applications/IMOD/bin")


def on_path(name: str, program: str) -> Tool:
    if found := shutil.which(program):
        return Tool(name, Path(found), f"{program} on PATH")
    return Tool(name, note=f"not on this host: no {program} on PATH")


def find_named(name: str, lanes: list[Lane]) -> Tool:
    """A tool a lane config names in [executables] or [segment]: as the lanes
    given name it, else as its variable names it, else from PATH."""

    keys = [key for key, tool in LANE_EXECUTABLES.items() if tool == name]
    section, key = keys[0]
    named = [(lane, str(value)) for lane in lanes
             if (value := (lane.cfg.get(section) or {}).get(key)) is not None]
    if named:
        lane, value = named[0]
        tool = path_from(name, value, f'{lane.path.name} [{section}] {key} = "{value}"')
        others = sorted({value for _, value in named[1:]} - {value})
        if others:
            disagree = f"other lane configs name it as {', '.join(others)}; the first is used"
            tool.problem = f"{tool.problem}; {disagree}" if tool.problem else disagree
        return tool
    variable, programs = LOOKUP[name]
    if value := os.environ.get(variable):
        if name == "pytom-match-pick":
            value = os.path.join(value, "pytom_match_template.py")
        return path_from(name, value, f"${variable}")
    for program in programs:
        if found := shutil.which(program):
            return Tool(name, Path(found), f"{program} on PATH")
    return Tool(name, note=f"not on this host: no lane config given names it, ${variable} "
                           f"is not set, and {' or '.join(programs)} is not on PATH")


def path_from(name: str, value: str, how: str) -> Tool:
    expanded = os.path.expandvars(value)
    if unset := re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", expanded):
        return Tool(name, found=how, problem=f"{how}: ${', $'.join(unset)} is not set in this shell")
    path = Path(expanded).expanduser()
    if not path.is_absolute() and (found := shutil.which(expanded)):
        path = Path(found)
    if not path.is_file():
        return Tool(name, found=how, problem=f"{how}: {path} is not a file")
    return Tool(name, path, how)


def ask_version(tool: Tool) -> Version | None:
    assert tool.path is not None
    match tool.name:
        case "imod":
            return imod_version(tool.path)
        case "ctffind":
            return ctffind_version(tool.path)
        case "motioncor3":
            return stated_version(tool.path, "MotionCor3")
        case "pytom-match-pick" | "membrain-seg":
            python = script_python(tool.path)
            if python is None:
                return Version(None, f"{tool.path} names no Python on its first line")
            return package_version(python, DISTRIBUTIONS[tool.name])
        case "slurm":
            return slurm_version(tool.path)
        case "bash":
            return bash_version(tool.path)
    return None


# --- what to read -----------------------------------------------------------

def plan(sources: dict[str, Any], tools: dict[str, Tool]) -> list[Item]:
    items: list[Item] = []
    for name, section in sources.items():
        tool = tools[name]
        if tool.path is None:
            continue
        if section.get("manual"):
            items.append(Item(name, "manual", "manual"))
        for page in section.get("man", []):
            items.append(Item(name, "man", page, page=page))
        usage = section.get("usage", [])
        if name == "pytom-match-pick":
            for script in section.get("scripts", []):
                path = tool.path.parent / script
                for args in usage:
                    items.append(Item(name, "usage", usage_name(args, script),
                                      script_command(path, args)))
        else:
            for args in usage:
                items.append(Item(name, "usage", usage_name(args), (str(tool.path), *args)))
    return items


def usage_name(args: list[str], script: str = "") -> str:
    said = "-".join(arg.lstrip("-") for arg in args) or "no-arguments"
    return f"{Path(script).stem}-{said}" if script else said


def script_command(path: Path, args: list[str]) -> tuple[str, ...]:
    """A script run as it is when it may be executed, else under its Python."""

    if not os.access(path, os.X_OK) and (python := script_python(path)) is not None:
        return (str(python), str(path), *args)
    return (str(path), *args)


def imod_programs_called(src: Path) -> set[str]:
    """The IMOD programs the pipeline's code runs, by the name it resolves them by."""

    pattern = re.compile(r"""resolve_imod_executable\(\s*["']([A-Za-z0-9_]+)["']""")
    return {name for path in src.rglob("*.py")
            for name in pattern.findall(path.read_text(encoding="utf-8"))}


# --- converters ---------------------------------------------------------------

_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_SKIP = frozenset({"script", "style", "head"})
_CODE = frozenset({"code", "tt", "kbd", "samp"})
_BLOCKS = frozenset({
    "p", "div", "center", "blockquote", "section", "article", "header", "footer", "main",
    "nav", "form", "address", "figure", "figcaption", "caption", "dl", "body",
})
_BREAK = "\x00"
_SPACE = re.compile(r"[ \t\r\n\f\v\xa0]+")


def _collapse(text: str) -> str:
    text = _SPACE.sub(" ", text)
    return re.sub(r" ?\x00 ?", "\n", text).strip()


def _fenced(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}"


class _Markdown(HTMLParser):
    """HTML to markdown, the same output for the same input.

    Headings, paragraphs, lists, tables and preformatted text keep their form.
    Emphasis, fonts and links are reduced to their words. Scripts, styles and
    the head are left out, except the title, which heads a page that has no
    h1 of its own.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.inline: list[str] = []
        self.skip = 0
        self.title: list[str] | None = None
        self.title_text = ""
        self.has_h1 = False
        self.heading: int | None = None
        self.pre = False
        self.pre_text: list[str] = []
        self.lists: list[list[int]] = []
        self.prefix = ""
        self.continuation = ""
        self.tables = 0
        self.rows: list[list[str]] = []
        self.cells: list[str] | None = None
        self.cell: list[str] | None = None
        self.codes: list[int] = []

    def markdown(self, html: str) -> str:
        self.feed(html)
        self.close()
        if self.pre:
            self._end_pre()
        if self.tables:
            self.tables = 0
            self._end_table()
        self._flush()
        blocks = self.blocks
        if self.title_text and not self.has_h1:
            blocks = [f"# {self.title_text}", *blocks]
        return "\n\n".join(blocks) + "\n" if blocks else ""

    def _buffer(self) -> list[str]:
        return self.cell if self.cell is not None else self.inline

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self.title = []
            return
        if tag in _SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if self.pre:
            if tag == "br":
                self.pre_text.append("\n")
            return
        if tag in _HEADINGS:
            self._flush()
            self.heading = _HEADINGS[tag]
            self.has_h1 = self.has_h1 or tag == "h1"
        elif tag == "pre":
            self._flush()
            self.pre = True
            self.pre_text = []
        elif tag == "br":
            self._buffer().append(_BREAK)
        elif tag == "hr":
            self._flush()
            if self.cell is None:
                self.blocks.append("---")
        elif tag in ("ul", "ol"):
            self._flush()
            self.lists.append([tag == "ol", 0])
        elif tag == "li":
            self._flush()
            self._start_item()
        elif tag in ("dt", "dd"):
            self._flush()
            self.prefix = self.continuation = "" if tag == "dt" else "  "
        elif tag == "table":
            if self.tables == 0:
                self._flush()
                self.rows = []
            self.tables += 1
        elif tag == "tr" and self.tables == 1:
            self._end_row()
            self.cells = []
        elif tag in ("td", "th") and self.tables == 1:
            self._end_cell()
            if self.cells is None:
                self.cells = []
            self.cell = []
        elif tag in ("td", "th", "tr"):
            self._buffer().append(" ")
        elif tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            if alt:
                self._buffer().append(f" [image: {alt}] ")
        elif tag in _CODE:
            self.codes.append(len(self._buffer()))
        elif tag in _BLOCKS:
            if self.cell is not None:
                self.cell.append(" ")
            else:
                self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in ("br", "hr", "img"):
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            if self.title is not None:
                self.title_text = _collapse("".join(self.title)).replace("\n", " ")
                self.title = None
            return
        if tag in _SKIP:
            self.skip = max(self.skip - 1, 0)
            return
        if self.skip:
            return
        if self.pre:
            if tag == "pre":
                self._end_pre()
            return
        if tag in _HEADINGS:
            self._flush()
            self.heading = None
        elif tag in ("ul", "ol"):
            self._flush()
            if self.lists:
                self.lists.pop()
            self.prefix = self.continuation = ""
        elif tag in ("li", "dt", "dd"):
            self._flush()
            self.prefix = self.continuation = ""
        elif tag == "table" and self.tables:
            self.tables -= 1
            if self.tables == 0:
                self._end_table()
        elif tag == "tr" and self.tables == 1:
            self._end_row()
        elif tag in ("td", "th") and self.tables == 1:
            self._end_cell()
        elif tag in _CODE and self.codes:
            mark = self.codes.pop()
            buffer = self._buffer()
            if mark <= len(buffer):
                text = _collapse("".join(buffer[mark:])).replace("\n", " ")
                del buffer[mark:]
                if text:
                    buffer.append(f"`{text}`" if "`" not in text else text)
        elif tag in _BLOCKS:
            if self.cell is not None:
                self.cell.append(" ")
            else:
                self._flush()

    def handle_data(self, data: str) -> None:
        if self.title is not None:
            self.title.append(data)
        elif self.skip:
            return
        elif self.pre:
            self.pre_text.append(data)
        else:
            self._buffer().append(data)

    def _start_item(self) -> None:
        indent = "  " * max(len(self.lists) - 1, 0)
        if self.lists and self.lists[-1][0]:
            self.lists[-1][1] += 1
            marker = f"{self.lists[-1][1]}. "
        else:
            marker = "- "
        self.prefix = indent + marker
        self.continuation = " " * len(self.prefix)

    def _flush(self) -> None:
        if self.cell is not None:
            return
        text = _collapse("".join(self.inline))
        self.inline = []
        if not text:
            return
        if self.heading is not None:
            self.blocks.append("#" * self.heading + " " + text.replace("\n", " "))
            return
        first, rest = (self.prefix, self.continuation) if self.prefix else (self.continuation,) * 2
        self.prefix = ""
        lines = text.split("\n")
        self.blocks.append("\n".join([first + lines[0], *(rest + line for line in lines[1:])]))

    def _end_pre(self) -> None:
        self.pre = False
        text = "".join(self.pre_text).replace("\r\n", "\n")
        self.pre_text = []
        if text.startswith("\n"):
            text = text[1:]
        text = "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")
        if not text.strip():
            return
        if self.cell is not None:
            self.cell.append(f" {_collapse(text)} ")
        else:
            self.blocks.append(_fenced(text))

    def _end_cell(self) -> None:
        if self.cell is None:
            return
        text = _collapse("".join(self.cell)).replace("\n", " ").replace("|", "\\|")
        self.cell = None
        if self.cells is None:
            self.cells = []
        self.cells.append(text)

    def _end_row(self) -> None:
        self._end_cell()
        if self.cells is not None:
            if any(self.cells):
                self.rows.append(self.cells)
            self.cells = None

    def _end_table(self) -> None:
        self._end_row()
        rows, self.rows = self.rows, []
        if not rows:
            return
        width = max(len(row) for row in rows)
        if width == 1:
            # A one-column table lays text out; its rows are paragraphs.
            self.blocks.extend(row[0] for row in rows if row[0])
            return
        lines = []
        for index, row in enumerate(rows):
            lines.append("| " + " | ".join(row + [""] * (width - len(row))) + " |")
            if index == 0:
                lines.append("|" + "---|" * width)
        self.blocks.append("\n".join(lines))


def html_to_markdown(html: str) -> str:
    return _Markdown().markdown(html)


_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_OVERSTRIKE = re.compile(r"[^\n]\x08")


def terminal_text(text: str) -> str:
    """What a terminal shows of a program's output: colour codes dropped, and
    the overstrikes `man` uses for bold and underline reduced to their letters."""

    text = _CSI.sub("", text)
    while "\x08" in text:
        shorter = _OVERSTRIKE.sub("", text)
        if shorter == text:
            text = text.replace("\x08", "")
            break
        text = shorter
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip("\n")


def qch_pages(qch: Path) -> dict[str, bytes]:
    """The HTML pages of a Qt help archive (.qch), by name, in name order.

    A .qch is an SQLite file. FileNameTable names each file, and FileDataTable
    holds it, compressed as Qt's qCompress does: a 4-byte length, then zlib.
    """

    uri = qch.resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute(
            "SELECT n.Name, d.Data FROM FileNameTable n JOIN FileDataTable d ON n.FileId = d.Id"
        ).fetchall()
    return dict(sorted(
        (name, zlib.decompress(bytes(data)[4:]))
        for name, data in rows if name.lower().endswith((".html", ".htm"))
    ))


def safe_name(name: str) -> PurePosixPath | None:
    """A document's name as a relative path inside the snapshot; None for one
    that would land outside it."""

    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    return path


# --- writing a snapshot -------------------------------------------------------

def front_matter(fields: dict[str, Any]) -> str:
    lines = ["---", *(f"{key}: {json.dumps(str(value))}" for key, value in fields.items()
                      if value is not None), "---", ""]
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        # UTF-8 always: a login shell's locale may be plain ASCII.
        path.write_text(text, encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def imod_manual(folder: Path) -> tuple[str, list[str]] | None:
    """Which form IMOD's installed manual takes in `folder`, and its documents'
    names: "qch" (html/IMOD.qch), "html" (HTML files in html/) or "man" (the
    formatted pages in man/). None when there is none of these."""

    qch = folder / "html" / "IMOD.qch"
    if qch.is_file():
        uri = qch.resolve().as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as db:
            names = [name for (name,) in db.execute("SELECT Name FROM FileNameTable")]
        return "qch", sorted(name for name in names if name.lower().endswith((".html", ".htm")))
    if (folder / "html").is_dir():
        names = sorted(path.relative_to(folder / "html").as_posix()
                       for path in (folder / "html").rglob("*")
                       if path.suffix.lower() in (".html", ".htm") and path.is_file())
        if names:
            return "html", names
    if (folder / "man").is_dir():
        names = sorted(path.relative_to(folder / "man").as_posix()
                       for path in (folder / "man").glob("cat*/*") if path.is_file())
        if names:
            return "man", names
    return None


def missing_imod_pages(names: list[str]) -> list[str]:
    """The IMOD programs the pipeline runs that have no manual page among `names`."""

    present = {PurePosixPath(name).stem for name in names
               if PurePosixPath(name).parts[0] in ("man", "cat1")}
    return sorted(imod_programs_called(SRC) - present)


def collect_manual(snapshot: Snapshot, tool: Tool, out: Path) -> None:
    """IMOD's installed manual: html/IMOD.qch, else the HTML files in html/,
    else the formatted pages in man/. Each page becomes text/imod/<name>.md."""

    assert tool.path is not None
    folder = tool.path
    found = imod_manual(folder)
    if found is None:
        snapshot.problems.append(f"IMOD in {folder} has no manual: no html/IMOD.qch, "
                                 "no HTML files in html/, no pages in man/")
        return
    form, names = found
    base = {"tool": "imod", "version": tool.version_text, "channel": "installation"}
    pages: dict[str, str] = {}
    if form == "qch":
        qch = folder / "html" / "IMOD.qch"
        copy(qch, out / "raw" / "imod" / "IMOD.qch")
        for name, data in qch_pages(qch).items():
            pages[name] = html_to_markdown(data.decode("utf-8", errors="replace"))
        record = {"original": str(qch), "raw": "raw/imod/IMOD.qch", "sha256": sha256_file(qch),
                  "converter": f"html {CONVERTERS['html']}"}
    else:
        within = folder / ("html" if form == "html" else "man")
        lines = []
        for name in names:
            path = within / name
            copy(path, out / "raw" / "imod" / form / name)
            lines.append(f"{sha256_file(path)}  {name}")
            text = path.read_text(encoding="utf-8", errors="replace")
            if form == "html":
                pages[name] = html_to_markdown(text)
            else:
                plain = terminal_text(text)
                pages[name] = f"# {PurePosixPath(name).stem}\n\n{_fenced(plain)}\n" if plain else ""
        record = {"original": str(within), "raw": f"raw/imod/{form}/",
                  "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                  "sha256_of": "the sha256sum lines of its files, in name order",
                  "converter": f"{'html' if form == 'html' else 'terminal'} "
                               f"{CONVERTERS['html' if form == 'html' else 'terminal']}"}
    written = 0
    for name, markdown in pages.items():
        relative = safe_name(name)
        if relative is None:
            snapshot.problems.append(f"IMOD manual page {name!r} would land outside the snapshot; left out")
            continue
        if not markdown.strip():
            continue
        target = out / "text" / "imod" / relative.with_suffix(".md")
        write(target, front_matter({**base, "source": record["original"], "document": name})
              + markdown)
        written += 1
    snapshot.documents += written
    snapshot.records.append({
        "tool": "imod", "kind": "manual", "channel": "installation", "version": tool.version_text,
        **record, "text": "text/imod/", "documents": written,
    })
    for program in missing_imod_pages(names):
        snapshot.problems.append(f"IMOD's manual has no page for {program}, which the pipeline runs")


def collect_man(snapshot: Snapshot, tool: Tool, item: Item, out: Path, env: dict[str, str]) -> None:
    result = probe(["man", item.page], seconds=USAGE_SECONDS, limit=MAN_LIMIT, env=env)
    if not (result.ended_well and result.output.strip()):
        program = shutil.which(item.page)
        if result.error:
            why = f"man could not run: {result.error}"
        elif result.stopped:
            why = f"man was stopped ({result.stopped})"
        else:
            why = f"man ended with exit code {result.exit_code}"
        if program is None:
            snapshot.problems.append(f"no manual page {item.page} ({why}), and no {item.page} on PATH")
            return
        snapshot.notes.append(f"no manual page {item.page} ({why}); its --help is kept instead")
        collect_usage(snapshot, tool, Item(tool.name, "usage", f"{item.page}-help",
                                           (program, "--help")), out, env)
        return
    raw = f"raw/{tool.name}/{item.page}.man.txt"
    write(out / raw, result.output)
    text = terminal_text(result.output)
    write(out / "text" / tool.name / f"{item.page}.md",
          front_matter({"tool": tool.name, "version": tool.version_text, "channel": "installation",
                        "source": f"man {item.page}"})
          + f"# {item.page}\n\n{_fenced(text)}\n")
    snapshot.documents += 1
    snapshot.records.append({
        "tool": tool.name, "kind": "man", "channel": "installation", "version": tool.version_text,
        "command": f"man {item.page}", "raw": raw, "sha256": sha256_file(out / raw),
        "converter": f"terminal {CONVERTERS['terminal']}", "text": f"text/{tool.name}/{item.page}.md",
    })


def collect_usage(snapshot: Snapshot, tool: Tool, item: Item, out: Path, env: dict[str, str]) -> None:
    shown = " ".join([Path(item.command[0]).name, *item.command[1:]])
    if len(item.command) > 1 and Path(item.command[0]).name.startswith("python") \
            and item.command[1].endswith(".py"):
        shown = " ".join([Path(item.command[1]).name, *item.command[2:]])
    result = probe(item.command, seconds=USAGE_SECONDS, limit=USAGE_LIMIT, env=env)
    if result.error:
        snapshot.problems.append(f"{shown}: could not run: {result.error}")
        return
    if not result.output.strip():
        snapshot.problems.append(f"{shown} printed nothing")
        return
    raw = f"raw/{tool.name}/{item.name}.txt"
    write(out / raw, result.output)
    if result.stopped:
        ended = {"time": f"It was stopped after {USAGE_SECONDS:g} s.\n\n",
                 "limit": f"It was stopped after {USAGE_LIMIT} bytes.\n\n"}.get(
            result.stopped, "It was stopped.\n\n")
    elif result.exit_code:
        ended = f"It ended with exit code {result.exit_code}.\n\n"
    else:
        ended = ""
    write(out / "text" / tool.name / f"{item.name}.md",
          front_matter({"tool": tool.name, "version": tool.version_text, "channel": "installation",
                        "source": " ".join(item.command)})
          + f"# {shown}\n\n{ended}{_fenced(terminal_text(result.output))}\n")
    snapshot.documents += 1
    snapshot.records.append({
        "tool": tool.name, "kind": "usage", "channel": "installation", "version": tool.version_text,
        "command": list(item.command), "exit_code": result.exit_code, "stopped": result.stopped,
        "raw": raw, "sha256": sha256_file(out / raw), "converter": f"terminal {CONVERTERS['terminal']}",
        "text": f"text/{tool.name}/{item.name}.md",
    })


def collect_site(snapshot: Snapshot, out: Path, env: dict[str, str]) -> None:
    lines = [
        f"# Site card: {snapshot.site}", "",
        f"Snapshot {snapshot.id}, built {snapshot.built} on {snapshot.host}.", "",
        "## Tools", "",
        "| Tool | Version | Where | Found by | Version read from |",
        "|---|---|---|---|---|",
    ]
    for tool in snapshot.tools.values():
        where = str(tool.path) if tool.path else (tool.problem or tool.note or "")
        how = tool.version.how if tool.version else ""
        version = tool.version_text if tool.path else "-"
        lines.append(f"| {tool.name} | {version} | {where} | {tool.found} | {how} |"
                     .replace("\n", " "))
    for name, command in SITE_PROBES:
        result = probe(command, seconds=USAGE_SECONDS, limit=USAGE_LIMIT, env=env)
        shown = " ".join(command)
        lines += ["", f"## {name}", "", f"`{shown}`", ""]
        record: dict[str, Any] = {"name": name, "command": list(command)}
        if result.error:
            lines.append(f"Could not run: {result.error}.")
            record["error"] = result.error
        else:
            raw = f"raw/site/{name}.txt"
            write(out / raw, result.output)
            if not result.ended_well:
                lines += [f"It ended with exit code {result.exit_code}." if result.stopped is None
                          else f"It was stopped ({result.stopped}).", ""]
            lines.append(_fenced(terminal_text(result.output) or "(nothing)"))
            record.update(exit_code=result.exit_code, stopped=result.stopped, raw=raw,
                          sha256=sha256_file(out / raw))
        snapshot.records.append({"kind": "site", **record})
    write(out / "site.md", "\n".join(lines) + "\n")


def builder_state() -> dict[str, Any]:
    """The commit this script ran from, and which of its files differed from it."""

    files = ["scripts/build_library.py", "src/cryoet_pipeline/tool_versions.py",
             "configs/library.toml"]
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", *files)
    return {"commit": commit,
            "modified": sorted(line[3:] for line in status.splitlines()) if status else []}


def _git(*args: str) -> str | None:
    result = probe(["git", "-C", str(ROOT), *args], seconds=30.0)
    return result.output.strip() if result.ended_well else None


def tool_record(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name, "path": str(tool.path) if tool.path else None, "found": tool.found,
        "version": tool.version.value if tool.version else None,
        "version_from": tool.version.how if tool.version else None,
        "said": tool.version.output if tool.version and not tool.version.value else None,
        "problem": tool.problem, "note": tool.note,
    }


def build(root: Path, snapshot: Snapshot, items: list[Item]) -> Path:
    """Write the snapshot under a hidden name, make it read-only, then name it."""

    partial = root / f".partial-{snapshot.id}-{os.getpid()}"
    partial.mkdir()
    try:
        env = {**os.environ, **PROBE_ENV}
        for item in items:
            tool = snapshot.tools[item.tool]
            if item.kind == "manual":
                collect_manual(snapshot, tool, partial)
            elif item.kind == "man":
                collect_man(snapshot, tool, item, partial, env)
            else:
                collect_usage(snapshot, tool, item, partial, env)
        collect_site(snapshot, partial, env)
        write(partial / "library.json", json.dumps({
            "schema": SCHEMA, "id": snapshot.id, "site": snapshot.site, "built": snapshot.built,
            "host": snapshot.host, "builder": builder_state(),
            "tools": [tool_record(tool) for tool in snapshot.tools.values()],
            "sources": snapshot.records, "documents": snapshot.documents,
            "problems": snapshot.problems, "notes": snapshot.notes,
        }, indent=2) + "\n")
        make_read_only(partial)
        final = root / snapshot.id
        os.rename(partial, final)
    except BaseException:
        remove(partial)
        raise
    point_current(root, snapshot.id)
    return final


def make_read_only(top: Path) -> None:
    """Owner and group may read; nobody may write."""

    for folder, folders, files in os.walk(top, topdown=False):
        for name in files:
            os.chmod(os.path.join(folder, name), 0o440)
        for name in folders:
            os.chmod(os.path.join(folder, name), 0o550)
    os.chmod(top, 0o550)


def remove(path: Path) -> None:
    """Remove a partial snapshot, read-only or not. Only the build's own hidden
    folder is ever passed here."""

    if not path.exists():
        return
    for folder, _, _ in os.walk(path):
        os.chmod(folder, 0o700)
    shutil.rmtree(path)


def point_current(root: Path, name: str) -> None:
    temporary = root / f".current-{os.getpid()}"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    os.symlink(name, temporary)
    os.replace(temporary, root / "current")


def next_id(root: Path | None, site: str, day: str) -> str:
    base = f"{site}-{day}"
    taken = {path.name for path in root.iterdir()} if root is not None and root.is_dir() else set()
    if base not in taken:
        return base
    for letter in "bcdefghijklmnopqrstuvwxyz":
        if base + letter not in taken:
            return base + letter
    raise SystemExit(f"{root} already holds 26 snapshots of {site} from {day}")


def today() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


# --- reading the inputs ---------------------------------------------------------

def read_sources(path: Path) -> dict[str, Any]:
    try:
        sources = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"no list of sources {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{path} is not valid TOML: {error}") from None
    unknown = sorted(set(sources) - set(KNOWN))
    if unknown:
        raise SystemExit(f"{path} names {', '.join(unknown)}, which this script does not know how "
                         f"to find; it knows {', '.join(KNOWN)}")
    return sources


def read_lane(path: Path) -> Lane:
    from run_series import read_config  # scripts/, beside this file

    _, cfg = read_config(path)
    return Lane(path, cfg)


def parse_given(pairs: list[str]) -> dict[str, str]:
    given: dict[str, str] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not value or name not in KNOWN:
            raise SystemExit(f"--version-of {pair!r}: give TOOL=VERSION, TOOL one of {', '.join(KNOWN)}")
        given[name] = value
    return given


# --- reporting --------------------------------------------------------------------

def print_plan(snapshot: Snapshot, root: Path | None, items: list[Item]) -> None:
    where = f" in {root}" if root else ""
    print(f"Library snapshot {snapshot.id}{where}: dry run, nothing written.\n")
    print("Tools")
    width = max(len(name) for name in snapshot.tools) + 2
    for tool in snapshot.tools.values():
        if tool.path is None:
            print(f"  {tool.name:<{width}}{'-':<10}{tool.problem or tool.note}")
            continue
        how = f"; version from {tool.version.how}" if tool.version else ""
        print(f"  {tool.name:<{width}}{tool.version_text:<10}{tool.path}  ({tool.found}{how})")
    print("\nWould read")
    for name in snapshot.tools:
        mine = [item for item in items if item.tool == name]
        if not mine:
            continue
        shown = []
        for item in mine:
            if item.kind == "manual":
                shown.append(manual_summary(snapshot, snapshot.tools[name]))
            elif item.kind == "man":
                shown.append(f"man {item.page}")
            else:
                shown.append(" ".join([Path(part).name if index == 0 else part
                                       for index, part in enumerate(item.command)]))
        print(f"  {name:<{width}}" + "; ".join(shown))
    print(f"  {'site card':<{width}}" + "; ".join(" ".join(command) for _, command in SITE_PROBES))
    report(snapshot, tools_shown=True)


def manual_summary(snapshot: Snapshot, tool: Tool) -> str:
    """The manual a dry run would read, and the pipeline's IMOD programs it lacks."""

    assert tool.path is not None
    try:
        found = imod_manual(tool.path)
    except sqlite3.Error as error:
        snapshot.problems.append(f"IMOD's manual in {tool.path} cannot be read: {error}")
        return f"the manual in {tool.path}"
    if found is None:
        snapshot.problems.append(f"IMOD in {tool.path} has no manual: no html/IMOD.qch, "
                                 "no HTML files in html/, no pages in man/")
        return "no manual"
    form, names = found
    where = {"qch": "html/IMOD.qch", "html": "html/", "man": "man/"}[form]
    for program in missing_imod_pages(names):
        snapshot.problems.append(f"IMOD's manual has no page for {program}, which the pipeline runs")
    return f"{tool.path / where} ({len(names)} pages)"


def report(snapshot: Snapshot, tools_shown: bool = False) -> None:
    """The problems and notes. Tools that are not on this host are named only
    when the tool table was not printed above."""

    problems = [f"{tool.name}: {tool.problem}" for tool in snapshot.tools.values() if tool.problem]
    problems += [f"{tool.name}: no version found ({tool.version.how}); give it with "
                 f"--version-of {tool.name}=VERSION if it is known"
                 for tool in snapshot.tools.values()
                 if tool.path is not None and tool.version and not tool.version.value]
    problems += snapshot.problems
    notes = [] if tools_shown else [f"{tool.name}: {tool.note}"
                                    for tool in snapshot.tools.values() if tool.note]
    notes += snapshot.notes
    print("\nProblems" if problems else "\nProblems: none")
    for line in problems:
        print(f"  - {line}")
    if notes:
        print("Notes")
        for line in notes:
            print(f"  - {line}")


# --- main ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a snapshot of the assistant's reference library on this site "
                    "(docs/library.md).")
    parser.add_argument("--site", required=True,
                        help="the site's short name, such as mdc or munich; it starts the snapshot's name")
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ["CRYOET_LIBRARY"]) if os.environ.get("CRYOET_LIBRARY") else None,
                        help="the library folder that holds the snapshots (default: $CRYOET_LIBRARY)")
    parser.add_argument("--config", type=Path, action="append", default=[],
                        help="a lane config whose executables to document; may be repeated")
    parser.add_argument("--version-of", action="append", default=[], metavar="TOOL=VERSION",
                        help="a tool's version, for one that states none")
    parser.add_argument("--sources", type=Path, default=SOURCES, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true",
                        help="print what was found and what would be read; write nothing")
    args = parser.parse_args(argv)
    if not SITE.match(args.site):
        parser.error(f"--site {args.site!r}: lower-case letters, digits and hyphens")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = read_sources(args.sources)
    lanes = [read_lane(path) for path in args.config]
    tools = find_tools(sources, lanes, parse_given(args.version_of))
    items = plan(sources, tools)
    snapshot = Snapshot(
        id=next_id(args.root, args.site, today()), site=args.site,
        built=datetime.now().astimezone().isoformat(timespec="seconds"),
        host=socket.gethostname(), tools=tools,
    )
    if args.dry_run:
        print_plan(snapshot, args.root, items)
        return 0
    if args.root is None:
        raise SystemExit("name the library folder with --root, or set CRYOET_LIBRARY")
    if not args.root.is_dir():
        raise SystemExit(f"no library folder {args.root}: make it once, with group read and setgid "
                         "(docs/library.md, \"Where snapshots live\")")
    build(args.root, snapshot, items)
    print(f"Built {snapshot.id} in {args.root}: {snapshot.documents} documents; "
          f"{args.root / 'current'} -> {snapshot.id}.")
    report(snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
