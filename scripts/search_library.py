#!/usr/bin/env python3
"""Search the reference library this session reads (docs/library.md).

    python3 scripts/search_library.py LOCALFILE
    python3 scripts/search_library.py -e -bin --tool imod
    python3 scripts/search_library.py 'Version :' --tool ctffind
    python3 scripts/search_library.py 'gres.*gpu' --regex --tool slurm

It searches the markdown pages of one snapshot line by line, ignoring case. The
snapshot is the one CRYOET_LIBRARY_SNAPSHOT names (scripts/assistant sets it),
else the one `current` names in the folder CRYOET_LIBRARY names, else the one
given with --snapshot. The pattern is plain text unless --regex is given; one
that starts with "-", such as a flag, goes after -e.

Pages with more matching lines come first. Each page is shown with its tool,
the tool's version and its source, then its matching lines with their numbers
in the page. The output is bounded (--pages, --lines), so read a page in full
with Read, from the line a match names.

It only reads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

TOOL = re.compile(r"^[a-z0-9][a-z0-9-]*$")
LONGEST = 200


def snapshot_from(given: str | None) -> Path | None:
    if given:
        return Path(given)
    if snapshot := os.environ.get("CRYOET_LIBRARY_SNAPSHOT"):
        return Path(snapshot)
    library = os.environ.get("CRYOET_LIBRARY")
    if library and library != "none":
        return Path(library) / "current"
    return None


def front_matter(lines: list[str]) -> tuple[dict[str, str], int]:
    """A page's header fields, and how many lines the header takes."""

    if not lines or lines[0] != "---":
        return {}, 0
    for end in range(1, len(lines)):
        if lines[end] == "---":
            fields = {}
            for line in lines[1:end]:
                key, _, value = line.partition(": ")
                try:
                    fields[key] = str(json.loads(value))
                except ValueError:
                    fields[key] = value
            return fields, end + 1
    return {}, 0


def search(snapshot: Path, pattern: re.Pattern[str], tool: str | None = None):
    """Every page with a matching line: (its path in the snapshot, its header,
    [(line number, line)]), more matches first; and how many pages were read."""

    base = snapshot / "text" / tool if tool else snapshot / "text"
    pages = sorted(base.rglob("*.md")) if base.is_dir() else []
    found = []
    for page in pages:
        lines = page.read_text(encoding="utf-8", errors="replace").splitlines()
        header, skip = front_matter(lines)
        matched = [(number, line) for number, line in enumerate(lines, start=1)
                   if number > skip and pattern.search(line)]
        if matched:
            found.append((page.relative_to(snapshot).as_posix(), header, matched))
    found.sort(key=lambda hit: (-len(hit[2]), hit[0]))
    return found, len(pages)


def report(snapshot: Path, shown: str, found, searched: int, pages: int, lines: int) -> str:
    out = [f"library {snapshot.resolve().name}: {len(found)} of {searched} pages "
           f"match {shown}"]
    for path, header, matched in found[:pages]:
        about = ", ".join(part for part in (
            " ".join(part for part in (header.get("tool"), header.get("version")) if part),
            header.get("document") or header.get("source")) if part)
        out += ["", f"{path}  ({about}; {len(matched)} lines)"]
        for number, line in matched[:lines]:
            line = line.strip()
            out.append(f"  {number}: {line[:LONGEST]}{' ...' if len(line) > LONGEST else ''}")
        if len(matched) > lines:
            out.append(f"  ({len(matched) - lines} more lines)")
    if len(found) > pages:
        out += ["", f"({len(found) - pages} more pages: narrow with --tool, or a longer pattern)"]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search the reference library this session reads (docs/library.md).")
    parser.add_argument("pattern", nargs="?", help="what to find, plain text unless --regex")
    parser.add_argument("-e", "--pattern", dest="dash_pattern", metavar="PATTERN",
                        help="the pattern, for one that starts with '-'")
    parser.add_argument("--regex", action="store_true", help="read the pattern as a regex")
    parser.add_argument("--tool", help="search only this tool's pages, such as imod or slurm")
    parser.add_argument("--pages", type=int, default=8, help="pages to show (default 8)")
    parser.add_argument("--lines", type=int, default=6, help="lines to show per page (default 6)")
    parser.add_argument("--snapshot", help="a snapshot folder, instead of this session's")
    argv = list(sys.argv[1:] if argv is None else argv)
    # argparse takes "-bin" after -e for an option of its own; joined, it is a value.
    for at, arg in enumerate(argv[:-1]):
        if arg in ("-e", "--pattern"):
            argv[at:at + 2] = [f"--pattern={argv[at + 1]}"]
            break
    args = parser.parse_args(argv)

    text = args.dash_pattern if args.dash_pattern is not None else args.pattern
    if not text:
        parser.error("give a pattern, or -e PATTERN for one that starts with '-'")
    if args.tool is not None and not TOOL.match(args.tool):
        parser.error(f"--tool {args.tool!r}: a tool's name, such as imod or slurm")
    snapshot = snapshot_from(args.snapshot)
    if snapshot is None or not (snapshot / "library.json").is_file():
        print("no library in this session: CRYOET_LIBRARY_SNAPSHOT names no snapshot "
              "(scripts/assistant sets it when the site has a library)", file=sys.stderr)
        return 2
    try:
        pattern = re.compile(text if args.regex else re.escape(text), re.IGNORECASE)
    except re.error as error:
        parser.error(f"--regex {text!r}: {error}")
    found, searched = search(snapshot, pattern, args.tool)
    if args.tool is not None and searched == 0:
        tools = sorted(path.name for path in (snapshot / "text").iterdir() if path.is_dir())
        print(f"no pages for {args.tool!r}; the library has {', '.join(tools)}", file=sys.stderr)
        return 2
    print(report(snapshot, repr(text), found, searched, max(args.pages, 1), max(args.lines, 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
