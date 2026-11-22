#!/usr/bin/env python3
"""Build a snapshot of the assistant's reference library on this site.

    python3 scripts/build_library.py --site mdc --dry-run
    python3 scripts/build_library.py --site mdc --root /fast/AG_Kudryashev/cryoet-library

docs/library.md says what the library is and why it is kept this way. This
script builds the installation channel: what this site's own installation says,
at the version installed here. Of the web channel it builds two kinds:
- pinned files: files of a tool's GitHub repository at the commit of the version
  installed here, each checked against the sha256 configs/library.toml pins
  (AreTomo2's manual first);
- pages ([web] in configs/library.toml): documentation no installation carries,
  NVIDIA's on CUDA, drivers and GPUs, CuPy's and PyTorch's on their CUDA builds,
  and conda's and mamba's guides. A page is read at each version this site has
  (the CUDA versions, the CuPy version, the conda and mamba versions), which
  this script finds itself, or as it stands on the day, dated. Only its own
  text is converted, not the site's menus; a text file (mamba's guides are
  reStructuredText in its repository, read at the tag of the version) is kept
  as it came.
A file that cannot be fetched, or has changed, stops the build, and the last
snapshot stays; a page the site does not have (404) is a problem, and the rest
is read.

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
import html as _html
import io
import json
import os
import re
import shutil
import socket
import sqlite3
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

from cryoet_pipeline.tool_versions import (
    Version,
    aretomo_version_of,
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
# "main" is how a web page's own text is told from the site's frame around it.
CONVERTERS = {"html": 1, "terminal": 1, "pdf": 1, "text": 1, "main": 1}
# Pinned files come from a GitHub repository at a commit, as its raw file.
GITHUB = re.compile(r"^https://github\.com/(?P<owner>[\w.-]+)/(?P<name>[\w.-]+)$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
# The kinds of pinned file a converter here reads.
PINNED_KINDS = {".pdf": "pdf", ".txt": "text", ".md": "text"}
# One request at a time, under a generic client name (docs/library.md, "Politeness").
USER_AGENT = "cryoet-pipeline-library-builder"
FETCH_SECONDS = 120.0
FETCH_LIMIT = 64 << 20
SITE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# The tools this script knows how to find and ask. configs/library.toml may name
# only these.
KNOWN = ("imod", "ctffind", "motioncor3", "aretomo", "pytom-match-pick", "membrain-seg",
         "slurm", "bash", "conda", "mamba")

# The lane config keys that name a tool's executable. The first key of a tool is
# the one its path is taken from.
LANE_EXECUTABLES: dict[tuple[str, str], str] = {
    ("executables", "ctffind"): "ctffind",
    ("executables", "motioncor3"): "motioncor3",
    ("executables", "aretomo"): "aretomo",
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
    "aretomo": ("ARETOMO", ("AreTomo2", "AreTomo")),
    "pytom-match-pick": ("PYTOMO_MP_DIR", ("pytom_match_template.py",)),
    "membrain-seg": ("MEMBRAIN_EXE", ("membrain",)),
}
DISTRIBUTIONS = {"pytom-match-pick": "pytom-match-pick", "membrain-seg": "membrain-seg"}
# The environment managers: the variable their shell init exports, then their
# names on PATH. micromamba comes before mamba, as scripts/activate_env.sh
# activates with it; the two share their documentation and their versions.
ENV_MANAGERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "conda": ("CONDA_EXE", ("conda",)),
    "mamba": ("MAMBA_EXE", ("micromamba", "mamba")),
}
# The backends' last place to look for IMOD.
MAC_IMOD = Path("/Applications/IMOD")
# Why a tool variable missing here matters: scripts/lane.sbatch reads no shell
# start-up file, and takes the variables from the shell it was submitted from.
SUBMIT_SHELL = " (lane jobs take their tool variables from the shell they are submitted from)"

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

# The web channel's pages ([web] in configs/library.toml). They come from these
# documentation sites only, one request at a time.
WEB_HOSTS = ("docs.nvidia.com", "developer.nvidia.com", "docs.cupy.dev", "pytorch.org",
             "docs.conda.io", "raw.githubusercontent.com")
PAGE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
# The versions a [web.<source>] may read its pages at, each found on this site.
FINDERS = ("cuda", "cupy", "conda", "mamba")
# CuPy's distribution, as conda-forge names it and as its PyPI wheels do.
CUPY_DISTRIBUTIONS = ("cupy", "cupy-cuda13x", "cupy-cuda12x", "cupy-cuda11x")
# A CUDA environment module: cuda/12.5, CUDA/12.1.1, a path before it.
CUDA_MODULE = re.compile(r"(?:^|/)cuda/(\d+)\.(\d+)", re.IGNORECASE)
# Where a page's own text is, tried in order; the rest is the site's menus and
# footers. NVIDIA's versioned CUDA pages mark it as Sphinx does (articleBody),
# its guides, CuPy's and PyTorch's put it in <article>, its GPU tables in <main>.
MAIN_TEXT: tuple[tuple[str, Callable[[str, dict[str, str | None]], bool]], ...] = (
    ('itemprop="articleBody"', lambda tag, attrs: attrs.get("itemprop") == "articleBody"),
    ("<article>", lambda tag, attrs: tag == "article"),
    ("<main>", lambda tag, attrs: tag == "main"),
    ('role="main"', lambda tag, attrs: attrs.get("role") == "main"),
)
# The "#" link Sphinx puts after every heading, which would end each heading's
# words in the text.
HEADERLINK = re.compile(r"""<a\b[^>]*\bclass=["']headerlink["'][^>]*>.*?</a>""",
                        re.IGNORECASE | re.DOTALL)
# A page whose address ends so is a text file, kept as it came in a fence, not
# HTML: mamba's guides, reStructuredText at the tag of a release.
TEXT_PAGES = (".rst", ".md", ".txt")
# mamba states its version alone ("2.5.0"), or after its name (mamba 1.x:
# "mamba 1.5.8", then conda's version on the next line).
_MAMBA_VERSION = re.compile(r"^(?:(?:micro)?mamba\s+)?v?(?P<version>\d+\.\d+(?:\.\d+)*)\b")


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
    """"manual", "man", "usage", "pinned" or "page"."""
    name: str
    """Its file name under text/<tool>/, without .md."""
    command: tuple[str, ...] = ()
    page: str = ""
    """A manual page's name; for a pinned file, its path in the repository."""
    pin: dict[str, Any] | None = None
    """For a pinned file: its tool's [<tool>.pinned] table."""
    version: str = ""
    """For a page read by version: the version, as its folder is named."""
    version_from: str = ""
    """Where that version was seen on this site."""
    states: str = ""
    """What a page's title must state to be taken for that version, when its
    address does not name it."""


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
    versions: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    """The versions the web pages are read at: finder -> (version, where seen)."""
    gpu_node: Path | None = None
    """What nvidia-smi printed on a GPU node, given with --gpu-node-report."""


# --- finding the tools ------------------------------------------------------

def find_tools(sources: dict[str, Any], lanes: list[Lane],
               given: dict[str, str]) -> dict[str, Tool]:
    """Every tool configs/library.toml names, found and asked for its version."""

    tools: dict[str, Tool] = {}
    for name in sources:
        if name == "imod":
            tool = find_imod(needed=bool(lanes))
        elif name == "slurm":
            tool = on_path("slurm", "sinfo")
        elif name == "bash":
            tool = on_path("bash", "bash")
        elif name in ENV_MANAGERS:
            tool = find_env_manager(name)
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


def find_imod(needed: bool = False) -> Tool:
    """IMOD's folder, from where the backends would find its programs
    (cryoet_pipeline.backends.alignment.resolve_imod_executable): newstack on
    PATH, then $IMOD_DIR/bin, then /Applications/IMOD/bin. The first folder
    that has a VERSION file wins. Every lane runs IMOD, so with lane configs
    given (`needed`) a missing IMOD is a problem, not a note."""

    candidates: list[tuple[Path, str]] = []
    if found := shutil.which("newstack"):
        candidates.append((Path(found), "newstack on PATH"))
    if imod_dir := os.environ.get("IMOD_DIR"):
        candidates.append((Path(imod_dir) / "bin" / "newstack", "$IMOD_DIR"))
    candidates.append((MAC_IMOD / "bin" / "newstack", str(MAC_IMOD)))
    present = [(path.resolve().parent.parent, how) for path, how in candidates if path.is_file()]
    for folder, how in present:
        if (folder / "VERSION").is_file():
            return Tool("imod", folder, how)
    if present:
        folder, how = present[0]
        return Tool("imod", folder, how, problem=f"{folder} has no VERSION file")
    where = f"no newstack on PATH, IMOD_DIR is not set, and there is none under {MAC_IMOD}/bin"
    if needed:
        return Tool("imod", problem=f"every lane runs IMOD, and {where}{SUBMIT_SHELL}")
    return Tool("imod", note=f"not on this host: {where}")


def on_path(name: str, program: str) -> Tool:
    if found := shutil.which(program):
        return Tool(name, Path(found), f"{program} on PATH")
    return Tool(name, note=f"not on this host: no {program} on PATH")


def find_env_manager(name: str) -> Tool:
    """conda, or micromamba or mamba: as the variable their shell init exports
    names it, else from PATH. A batch job finds it the same way
    (scripts/activate_env.sh)."""

    variable, programs = ENV_MANAGERS[name]
    if value := os.environ.get(variable):
        return path_from(name, value, f"${variable}")
    for program in programs:
        if found := shutil.which(program):
            return Tool(name, Path(found), f"{program} on PATH")
    return Tool(name, note=f"not on this host: ${variable} is not set, and "
                           f"{' or '.join(programs)} is not on PATH")


def mamba_version(executable: Path) -> Version:
    result = probe([executable, "--version"])
    lines = [line.strip() for line in result.output.splitlines() if line.strip()]
    found = _MAMBA_VERSION.match(lines[0]) if result.ended_well and lines else None
    return Version(found["version"] if found else None, f"{executable.name} --version",
                   "\n".join(lines[:2]))


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
        return Tool(name, found=how,
                    problem=f"{how}: ${', $'.join(unset)} is not set in this shell{SUBMIT_SHELL}")
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
        case "aretomo":
            # The program's name is part of the answer: AreTomo2 and AreTomo3
            # do not share a command line.
            return Version(aretomo_version_of(tool.path), f"{tool.path.name} --version")
        case "pytom-match-pick" | "membrain-seg":
            python = script_python(tool.path)
            if python is None:
                return Version(None, f"{tool.path} names no Python on its first line")
            return package_version(python, DISTRIBUTIONS[tool.name])
        case "slurm":
            return slurm_version(tool.path)
        case "bash":
            return bash_version(tool.path)
        case "conda":
            return stated_version(tool.path, "conda", attempts=(("--version",),))
        case "mamba":
            return mamba_version(tool.path)
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
        pin = section.get("pinned")
        for path in (pin or {}).get("files", {}):
            items.append(Item(name, "pinned", path, (pinned_url(pin, path),), page=path, pin=pin))
    return items


def pinned_url(pin: dict[str, Any], path: str) -> str:
    """Where a pinned file is fetched from: its repository's raw file at the commit."""

    found = GITHUB.match(pin["repository"])
    assert found
    return (f"https://raw.githubusercontent.com/{found['owner']}/{found['name']}/"
            f"{pin['commit']}/{path}")


def pin_problem(tool: Tool, item: Item) -> str | None:
    """Why a pinned file is left out on this host, or None to read it. The files
    document one version; another version installed here would be misread by
    them."""

    assert item.pin is not None
    if tool.version_text != item.pin["version"]:
        return (f"{item.page} at {item.pin['commit'][:10]} documents {item.pin['version']}, "
                f"and the {tool.name} here states {tool.version_text}; it is left out. Pin the "
                f"commit of this version in [{tool.name}.pinned] of configs/library.toml")
    if PINNED_KINDS[PurePosixPath(item.page).suffix.lower()] == "pdf" and pdf_reader() is None:
        return (f"{item.page} is a PDF, and pypdf is not installed in this environment; it is "
                "left out. Install it (pip install pypdf) and build again")
    return None


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
    if result.exit_code == 127:
        # What a program prints when it cannot start: the dynamic loader's
        # complaint (AreTomo2 1.1.3 links cuFFT and the CUDA runtime from the
        # shell's library path, its makefile11), or a wrapper's "command not
        # found". Not its usage, so not a page of a snapshot that never changes.
        said = result.output.strip().splitlines()[0]
        snapshot.problems.append(f"{shown}: could not start (exit code 127): {said}")
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


def fetch(url: str) -> bytes:
    """One file from the web, whole. A failure stops the build: a source that has
    gone or changed fails loudly, and the last snapshot stays (docs/library.md)."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_SECONDS) as response:
            data: bytes = response.read(FETCH_LIMIT + 1)
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(f"could not fetch {url}: {error}; nothing was built") from None
    if len(data) > FETCH_LIMIT:
        raise SystemExit(f"{url} is larger than {FETCH_LIMIT} bytes; nothing was built")
    return data


def pdf_reader() -> Any:
    """pypdf, if this environment has it. Only a PDF source needs it."""

    try:
        import pypdf
    except ImportError:
        return None
    return pypdf


def pdf_to_markdown(data: bytes, title: str) -> tuple[str, int]:
    """A PDF's text, page by page, under the page's number. Each line keeps its
    words with single spaces between them, so that a phrase is found whatever
    spacing the PDF placed between its words."""

    pypdf = pdf_reader()
    assert pypdf is not None
    reader = pypdf.PdfReader(io.BytesIO(data))
    parts = [f"# {title}\n"]
    for number, page in enumerate(reader.pages, start=1):
        lines = (" ".join(line.split()) for line in (page.extract_text() or "").splitlines())
        text = "\n".join(line for line in lines if line)
        parts.append(f"## Page {number}\n\n{text}\n" if text else f"## Page {number}\n")
    return "\n".join(parts), len(reader.pages)


def converter_name(kind: str) -> str:
    if kind == "pdf":
        return f"pdf {CONVERTERS['pdf']} (pypdf {pdf_reader().__version__})"
    return f"{kind} {CONVERTERS[kind]}"


def collect_pinned(snapshot: Snapshot, tool: Tool, item: Item, out: Path) -> None:
    """A file of the tool's repository at the pinned commit: fetched, checked
    against its pinned sha256, kept as it came, and converted to markdown."""

    assert item.pin is not None
    if problem := pin_problem(tool, item):
        snapshot.problems.append(f"{tool.name}: {problem}")
        return
    url = item.command[0]
    data = fetch(url)
    digest = hashlib.sha256(data).hexdigest()
    pinned = item.pin["files"][item.page]
    if digest != pinned:
        raise SystemExit(f"{url} has sha256 {digest}, where [{tool.name}.pinned] in "
                         f"configs/library.toml pins {pinned}; nothing was built")
    relative = safe_name(item.page)
    assert relative is not None
    raw = f"raw/{tool.name}/{relative}"
    write(out / raw, data)
    kind = PINNED_KINDS[relative.suffix.lower()]
    record: dict[str, Any] = {}
    if kind == "pdf":
        markdown, pages = pdf_to_markdown(data, relative.name)
        record["pages"] = pages
    else:
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").strip("\n")
        markdown = f"# {relative.name}\n\n{_fenced(text)}\n"
    source = f"{item.pin['repository']}/blob/{item.pin['commit']}/{item.page}"
    text_path = f"text/{tool.name}/{relative.with_suffix('.md')}"
    write(out / text_path,
          front_matter({"tool": tool.name, "version": item.pin["version"],
                        "channel": "web, pinned", "source": source, "document": item.page})
          + markdown)
    snapshot.documents += 1
    snapshot.records.append({
        "tool": tool.name, "kind": "pinned", "channel": "web, pinned",
        "version": item.pin["version"], "repository": item.pin["repository"],
        "commit": item.pin["commit"], "path": item.page, "url": url, "raw": raw,
        "sha256": digest, "converter": converter_name(kind), "text": text_path, **record,
    })


# --- the web channel's pages ---------------------------------------------------------

def module_names() -> list[str]:
    """The environment modules a login shell offers (`module -t avail`)."""

    result = probe(["bash", "-lc", "module -t avail 2>&1"], seconds=USAGE_SECONDS)
    names = []
    for line in result.output.splitlines():
        token = line.strip()
        if token and not token.endswith(":") and " " not in token and not token.startswith(("/", "-")):
            names.append(re.sub(r"\(.*\)$", "", token))
    return names


def torch_cuda() -> str:
    """The CUDA the pipeline's PyTorch is built for, asked of this script's own
    Python, which is the pipeline's environment; "" for none or a CPU build."""

    result = probe([sys.executable, "-I", "-c", "import torch; print(torch.version.cuda)"],
                   seconds=180.0)
    said = result.output.strip().splitlines()
    return said[-1].strip() if result.ended_well and said and said[-1].strip() != "None" else ""


def cupy_cuda(tool: Tool | None) -> str:
    """The CUDA pytom-match-pick's CuPy is built for, asked of the Python its
    scripts run under; "" when it cannot say. CuPy imports without a GPU."""

    python = script_python(tool.path) if tool is not None and tool.path is not None else None
    if python is None:
        return ""
    result = probe([python, "-I", "-c",
                    "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"], seconds=120.0)
    said = result.output.strip().splitlines()
    if not result.ended_well or not said or not said[-1].strip().isdigit():
        return ""
    number = int(said[-1])
    return f"{number // 1000}.{number % 1000 // 10}"


def cuda_versions(tools: dict[str, Tool], given: list[str]) -> list[tuple[str, str]]:
    """Every CUDA version this site has, as major.minor.0 (the release its pages
    are filed under), with where each was seen: the toolkit on PATH, the CUDA
    modules, and the CUDA the pipeline's PyTorch and pytom-match-pick's CuPy are
    built for, whose drivers matter as much as a toolkit's."""

    seen: dict[tuple[int, int], list[str]] = {}

    def add(text: str, how: str) -> None:
        if found := re.match(r"(\d+)\.(\d+)", text or ""):
            seen.setdefault((int(found[1]), int(found[2])), []).append(how)

    if nvcc := shutil.which("nvcc"):
        if release := re.search(r"release (\d+\.\d+)", probe([nvcc, "--version"]).output):
            add(release[1], "nvcc on PATH")
    for name in module_names():
        if found := CUDA_MODULE.search(name):
            add(f"{found[1]}.{found[2]}", f"module {name}")
    add(torch_cuda(), "the pipeline's PyTorch")
    add(cupy_cuda(tools.get("pytom-match-pick")), "pytom-match-pick's CuPy")
    for version in given:
        add(version, "given with --version-of")
    return [(f"{major}.{minor}.0", ", ".join(hows)) for (major, minor), hows in sorted(seen.items())]


def cupy_versions(tools: dict[str, Tool], given: list[str]) -> list[tuple[str, str]]:
    """The CuPy version of pytom-match-pick's environment."""

    if given:
        return [(given[-1], "given with --version-of")]
    tool = tools.get("pytom-match-pick")
    python = script_python(tool.path) if tool is not None and tool.path is not None else None
    if python is None:
        return []
    for distribution in CUPY_DISTRIBUTIONS:
        if (version := package_version(python, distribution).value) is not None:
            return [(version, f"{distribution} in pytom-match-pick's environment")]
    return []


def manager_version(tools: dict[str, Tool], name: str) -> tuple[str, str] | None:
    """An environment manager's version, and where it was seen: the program and
    how it was found, or --version-of."""

    tool = tools.get(name)
    if tool is None or tool.version is None or not tool.version.value:
        return None
    if tool.path is None or tool.version.how.startswith("given with --version-of"):
        return tool.version.value, tool.version.how
    return tool.version.value, f"{tool.path.name} {tool.version.value}, {tool.found}"


def conda_versions(tools: dict[str, Tool], given: list[str]) -> list[tuple[str, str]]:
    """conda's version as its documentation names a release's folder, by major
    and minor version: 26.3.2 is read under 26.3.x. --version-of conda gives
    the tool's version, and so this one."""

    found = manager_version(tools, "conda")
    number = re.match(r"(\d+)\.(\d+)", found[0]) if found else None
    return [(f"{number[1]}.{number[2]}.x", found[1])] if found and number else []


def mamba_versions(tools: dict[str, Tool], given: list[str]) -> list[tuple[str, str]]:
    """micromamba's or mamba's version, which names the tag of its release in
    mamba's repository."""

    found = manager_version(tools, "mamba")
    return [found] if found else []


def find_versions(web: dict[str, Any], tools: dict[str, Tool],
                  given: dict[str, list[str]]) -> dict[str, list[tuple[str, str]]]:
    """The versions the [web] sources read their pages at; only the finders some
    source names are asked."""

    wanted = {section["versions"] for section in web.values() if "versions" in section}
    finders = {"cuda": cuda_versions, "cupy": cupy_versions, "conda": conda_versions,
               "mamba": mamba_versions}
    return {name: finders[name](tools, given.get(name, [])) for name in FINDERS if name in wanted}


def page_states(finder: str, version: str) -> str:
    """What a page's title states for a version: CUDA names major.minor only."""

    return ".".join(version.split(".")[:2]) if finder == "cuda" else version


def plan_web(web: dict[str, Any], versions: dict[str, list[tuple[str, str]]],
             notes: list[str]) -> list[Item]:
    """One item per page and version; the addresses to try, in order."""

    items: list[Item] = []
    for source, section in web.items():
        finder = section.get("versions")
        found = versions.get(finder, []) if finder else [("", "")]
        if not found:
            notes.append(f"{source}: no {finder} version found on this host, so its pages are "
                         f"left out; --version-of {finder}=VERSION reads them at a version")
            continue
        for version, how in found:
            for name, addresses in section["pages"].items():
                candidates = [addresses] if isinstance(addresses, str) else list(addresses)
                urls = tuple(url.replace("{version}", version) for url in candidates)
                items.append(Item(source, "page", name, urls, version=version, version_from=how,
                                  states=page_states(finder, version) if finder else ""))
    return items


def fetch_page(url: str) -> tuple[bytes, str] | None:
    """A page, and the address it came from after any redirect; None when the
    site has no such page (404, 410). Any other failure stops the build."""

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=FETCH_SECONDS) as response:
            data: bytes = response.read(FETCH_LIMIT + 1)
            final = response.geturl()
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            return None
        raise SystemExit(f"could not fetch {url}: {error}; nothing was built") from None
    except (urllib.error.URLError, OSError) as error:
        raise SystemExit(f"could not fetch {url}: {error}; nothing was built") from None
    if len(data) > FETCH_LIMIT:
        raise SystemExit(f"{url} is larger than {FETCH_LIMIT} bytes; nothing was built")
    return data, final


def page_title(html: str) -> str:
    found = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return _collapse(_html.unescape(found.group(1))).replace("\n", " ") if found else ""


def is_that_version(item: Item, url: str, final: str, html: str) -> bool:
    """Whether a page is the one for the item's version. An address that names
    the version must still name it after any redirect; any other must state it
    in its title (NVIDIA's newest CUDA is at an address without a version)."""

    if not item.version:
        return True
    if item.version in url:
        return item.version in final
    return re.search(rf"(?<![\d.]){re.escape(item.states)}(?![\d])", page_title(html)) is not None


class _Element(HTMLParser):
    """Where the first element a test picks begins and ends in a page."""

    def __init__(self, test: Callable[[str, dict[str, str | None]], bool]) -> None:
        super().__init__(convert_charrefs=True)
        self.test = test
        self.tag: str | None = None
        self.depth = 0
        self.start: tuple[int, int] | None = None
        self.end: tuple[int, int] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.end is not None:
            return
        if self.tag is None:
            if self.test(tag, dict(attrs)):
                self.tag, self.depth, self.start = tag, 1, self.getpos()
        elif tag == self.tag:
            self.depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """A tag closed as it opens, <div/>, holds nothing and leaves the depth."""

    def handle_endtag(self, tag: str) -> None:
        if self.tag is not None and self.end is None and tag == self.tag:
            self.depth -= 1
            if self.depth == 0:
                self.end = self.getpos()


def main_text(html: str) -> tuple[str, str]:
    """A page's own text as HTML, and what found it: the first element of
    MAIN_TEXT, or the whole page when none is there."""

    starts = [0]
    for line in html.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    for label, test in MAIN_TEXT:
        element = _Element(test)
        element.feed(html)
        element.close()
        if element.start is not None:
            begin = starts[element.start[0] - 1] + element.start[1]
            end = (starts[element.end[0] - 1] + element.end[1] if element.end is not None
                   else len(html))
            return html[begin:end], label
    return html, "the whole page"


def collect_page(snapshot: Snapshot, item: Item, out: Path) -> None:
    """A documentation page: the first of its addresses that answers with the
    right version, kept as it came, its own text converted to markdown."""

    found = None
    for url in item.command:
        fetched = fetch_page(url)
        if fetched is None:
            continue
        data, final = fetched
        html = data.decode("utf-8", errors="replace")
        if is_that_version(item, url, final, html):
            found = url, final, data, html
            break
    label = f"{item.tool} {item.version} {item.name}".replace("  ", " ")
    if found is None:
        what = f"states {item.states}" if item.version else "answers"
        snapshot.problems.append(f"{label}: no page {what} at {' or '.join(item.command)}; "
                                 "it is left out")
        return
    url, final, data, html = found
    folder = f"{item.tool}/{item.version}" if item.version else item.tool
    suffix = PurePosixPath(urllib.parse.urlsplit(final).path).suffix.lower()
    if suffix in TEXT_PAGES:
        raw = f"raw/{folder}/{item.name}{suffix}"
        text = html.replace("\r\n", "\n").strip("\n")
        markdown = f"# {PurePosixPath(urllib.parse.urlsplit(final).path).name}\n\n{_fenced(text)}\n"
        converter = f"text {CONVERTERS['text']}"
    else:
        raw = f"raw/{folder}/{item.name}.html"
        part, how = main_text(html)
        markdown = html_to_markdown(HEADERLINK.sub("", part))
        title = page_title(html)
        if title and not markdown.startswith("# "):
            markdown = f"# {title}\n\n{markdown}"
        converter = f"html {CONVERTERS['html']}, main {CONVERTERS['main']} ({how})"
    write(out / raw, data)
    channel = "web, by version" if item.version else "web, dated"
    text_path = f"text/{folder}/{item.name}.md"
    write(out / text_path,
          front_matter({"tool": item.tool,
                        "version": item.version or f"as of {snapshot.built[:10]}",
                        "channel": channel, "source": final, "document": final,
                        "fetched": snapshot.built})
          + markdown)
    snapshot.documents += 1
    snapshot.records.append({
        "tool": item.tool, "kind": "page", "channel": channel, "name": item.name,
        "version": item.version or None, "version_from": item.version_from or None,
        "url": url, "final_url": final, "fetched": snapshot.built, "raw": raw,
        "sha256": hashlib.sha256(data).hexdigest(),
        "converter": converter,
        "text": text_path,
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
    for finder, found in snapshot.versions.items():
        lines += ["", f"## {finder} versions on this site", ""]
        lines += [f"- {version}: {how}" for version, how in found] or ["None found."]
    if snapshot.gpu_node is not None:
        given = snapshot.gpu_node
        raw = "raw/site/gpu-node.txt"
        copy(given, out / raw)
        written = datetime.fromtimestamp(given.stat().st_mtime).astimezone().isoformat(
            timespec="seconds")
        lines += ["", "## A GPU node", "",
                  f"What `nvidia-smi` printed there, given with --gpu-node-report: {given}, "
                  f"written {written}.", "",
                  _fenced(terminal_text(given.read_text(errors="replace")) or "(nothing)")]
        snapshot.records.append({"kind": "site", "name": "gpu node", "given": str(given),
                                 "written": written, "raw": raw, "sha256": sha256_file(out / raw)})
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
            if item.kind == "page":
                collect_page(snapshot, item, partial)
                continue
            tool = snapshot.tools[item.tool]
            if item.kind == "manual":
                collect_manual(snapshot, tool, partial)
            elif item.kind == "man":
                collect_man(snapshot, tool, item, partial, env)
            elif item.kind == "pinned":
                collect_pinned(snapshot, tool, item, partial)
            else:
                collect_usage(snapshot, tool, item, partial, env)
        collect_site(snapshot, partial, env)
        write(partial / "library.json", json.dumps({
            "schema": SCHEMA, "id": snapshot.id, "site": snapshot.site, "built": snapshot.built,
            "host": snapshot.host, "builder": builder_state(),
            "tools": [tool_record(tool) for tool in snapshot.tools.values()],
            "versions": {finder: [{"version": version, "seen": how} for version, how in found]
                         for finder, found in snapshot.versions.items()},
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
    # The web channel's pages are read by read_web; the rest are tools.
    sources.pop("web", None)
    unknown = sorted(set(sources) - set(KNOWN))
    if unknown:
        raise SystemExit(f"{path} names {', '.join(unknown)}, which this script does not know how "
                         f"to find; it knows {', '.join(KNOWN)}")
    for name, section in sources.items():
        if "pinned" in section:
            check_pin(path, name, section["pinned"])
    return sources


def read_web(path: Path) -> dict[str, Any]:
    """The [web] sources of the list, each checked: its pages by name, each at an
    https address on one of WEB_HOSTS, or a list of them to try in order."""

    web = tomllib.loads(path.read_text(encoding="utf-8")).get("web", {})
    if not isinstance(web, dict):
        raise SystemExit(f"{path} [web] must be a table of sources")
    for source, section in web.items():
        where = f"{path} [web.{source}]"
        if not PAGE_NAME.match(source) or source in KNOWN:
            raise SystemExit(f"{where}: a source's name is lower-case letters, digits and "
                             "hyphens, and not a tool's")
        if not isinstance(section, dict) or "pages" not in section or not set(section) <= {
                "versions", "pages"}:
            raise SystemExit(f"{where} needs pages, may have versions, and nothing else")
        finder = section.get("versions")
        if finder is not None and finder not in FINDERS:
            raise SystemExit(f"{where}: versions must be one of {', '.join(FINDERS)}")
        pages = section["pages"]
        if not isinstance(pages, dict) or not pages:
            raise SystemExit(f"{where}: pages must name each page and give its address")
        for name, addresses in pages.items():
            candidates = [addresses] if isinstance(addresses, str) else addresses
            if not PAGE_NAME.match(name) or not isinstance(candidates, list) or not candidates or \
                    not all(isinstance(url, str) for url in candidates):
                raise SystemExit(f"{where} {name}: a page's name is lower-case letters, digits "
                                 "and hyphens, and its address one string or a list of them")
            for url in candidates:
                parts = urllib.parse.urlsplit(url)
                if parts.scheme != "https" or parts.hostname not in WEB_HOSTS:
                    raise SystemExit(f"{where} {name}: {url} must be an https address on one of "
                                     f"{', '.join(WEB_HOSTS)}")
            if finder is not None and "{version}" not in candidates[0]:
                raise SystemExit(f"{where} {name}: its first address must hold {{version}}, as "
                                 "its pages are read by version")
            if finder is None and any("{version}" in url for url in candidates):
                raise SystemExit(f"{where} {name}: {{version}} in an address needs versions = "
                                 "one of " + ", ".join(FINDERS))
    return web


def check_pin(path: Path, name: str, pin: Any) -> None:
    where = f"{path} [{name}.pinned]"
    if not isinstance(pin, dict) or set(pin) != {"repository", "commit", "version", "files"}:
        raise SystemExit(f"{where} needs repository, commit, version and files, and nothing else")
    if not isinstance(pin["repository"], str) or not GITHUB.match(pin["repository"]):
        raise SystemExit(f"{where}: repository must be https://github.com/<owner>/<name>")
    if not isinstance(pin["commit"], str) or not COMMIT.match(pin["commit"]):
        raise SystemExit(f"{where}: commit must be a full 40-character commit, not a branch or tag")
    if not isinstance(pin["version"], str) or not pin["version"].strip():
        raise SystemExit(f"{where}: version must be the version the files document, as the "
                         "tool states it")
    files = pin["files"]
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{where}: files must map each file's path to its sha256")
    for file, digest in files.items():
        relative = safe_name(file)
        if relative is None or relative.suffix.lower() not in PINNED_KINDS:
            raise SystemExit(f"{where}: {file!r} must be a path inside the repository, ending in "
                             f"{', '.join(PINNED_KINDS)}")
        if not isinstance(digest, str) or not SHA256.match(digest):
            raise SystemExit(f"{where}: the sha256 of {file} must be 64 lower-case hex digits")


def read_lane(path: Path) -> Lane:
    from run_series import read_config  # scripts/, beside this file

    _, cfg = read_config(path)
    return Lane(path, cfg)


def parse_given(pairs: list[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """--version-of: a tool's version, and the versions to read a [web] source's
    pages at (cuda may be given more than once). conda and mamba are tools: the
    version given them is the one their pages are read at."""

    given: dict[str, str] = {}
    web: dict[str, list[str]] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if value and name in KNOWN:
            given[name] = value
        elif value and name in FINDERS:
            web.setdefault(name, []).append(value)
        else:
            raise SystemExit(f"--version-of {pair!r}: give NAME=VERSION, NAME one of "
                             f"{', '.join(dict.fromkeys((*KNOWN, *FINDERS)))}")
    return given, web


# --- reporting --------------------------------------------------------------------

def print_plan(snapshot: Snapshot, root: Path | None, items: list[Item]) -> None:
    where = f" in {root}" if root else ""
    print(f"Library snapshot {snapshot.id}{where}: dry run, nothing written.\n")
    print("Tools")
    sources = list(dict.fromkeys(item.tool for item in items if item.kind == "page"))
    width = max(len(name) for name in [*snapshot.tools, *sources, "site card"]) + 2
    # AreTomo's version carries its name ("AreTomo2 1.1.3"), longer than a number.
    said = max([10] + [len(tool.version_text) + 2 for tool in snapshot.tools.values()
                       if tool.path is not None])
    for tool in snapshot.tools.values():
        if tool.path is None:
            print(f"  {tool.name:<{width}}{'-':<{said}}{tool.problem or tool.note}")
            continue
        how = f"; version from {tool.version.how}" if tool.version else ""
        print(f"  {tool.name:<{width}}{tool.version_text:<{said}}{tool.path}  ({tool.found}{how})")
    print("\nWould read")
    for name in snapshot.tools:
        mine = [item for item in items if item.tool == name]
        if not mine:
            continue
        shown = []
        for item in mine:
            if item.kind == "manual":
                shown.append(manual_summary(snapshot, snapshot.tools[name]))
            elif item.kind == "pinned":
                assert item.pin is not None
                if problem := pin_problem(snapshot.tools[name], item):
                    snapshot.problems.append(f"{name}: {problem}")
                shown.append(f"{item.page} of {item.pin['repository'].removeprefix('https://')} "
                             f"at {item.pin['commit'][:10]}")
            elif item.kind == "man":
                # `man -w` names a page's file without showing it: a compute
                # node may have no manual pages where a login node has them.
                page = probe(["man", "-w", item.page], seconds=10.0)
                shown.append(f"man {item.page}" if page.ended_well and page.output.strip()
                             else f"{item.page} --help (no manual page here)")
            else:
                shown.append(" ".join([Path(part).name if index == 0 else part
                                       for index, part in enumerate(item.command)]))
        print(f"  {name:<{width}}" + "; ".join(shown))
    for source in sources:
        mine = [item for item in items if item.tool == source]
        names = list(dict.fromkeys(item.name for item in mine))
        if mine[0].version:
            when = ", ".join(f"{item.version} ({item.version_from})" for item in mine
                             if item.name == names[0])
        else:
            when = "as they stand on the day of the build"
        print(f"  {source:<{width}}{when}: {', '.join(names)}")
    card = "; ".join(" ".join(command) for _, command in SITE_PROBES)
    if snapshot.gpu_node is not None:
        card += f"; a GPU node's nvidia-smi from {snapshot.gpu_node}"
    print(f"  {'site card':<{width}}{card}")
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
    parser.add_argument("--version-of", action="append", default=[], metavar="NAME=VERSION",
                        help="a tool's version, for one that states none; or cuda=12.5 or "
                             "cupy=14.2.0, a version to read those pages at (cuda may be repeated)")
    parser.add_argument("--gpu-node-report", type=Path, metavar="FILE",
                        help="what nvidia-smi printed on a GPU node, for the site card")
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
    web = read_web(args.sources)
    lanes = [read_lane(path) for path in args.config]
    given, given_web = parse_given(args.version_of)
    if args.gpu_node_report is not None and not args.gpu_node_report.is_file():
        raise SystemExit(f"--gpu-node-report {args.gpu_node_report}: no such file")
    tools = find_tools(sources, lanes, given)
    items = plan(sources, tools)
    versions = find_versions(web, tools, given_web)
    snapshot = Snapshot(
        id=next_id(args.root, args.site, today()), site=args.site,
        built=datetime.now().astimezone().isoformat(timespec="seconds"),
        host=socket.gethostname(), tools=tools, versions=versions,
        gpu_node=args.gpu_node_report,
    )
    items += plan_web(web, versions, snapshot.notes)
    if args.gpu_node_report is None and shutil.which("nvidia-smi") is None:
        snapshot.notes.append("the site card has no GPU: there is no nvidia-smi on this host. "
                              "Give --gpu-node-report with what nvidia-smi prints on a GPU node "
                              "(docs/library.md, \"How a snapshot is built\", the site "
                              "card's GPU)")
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
