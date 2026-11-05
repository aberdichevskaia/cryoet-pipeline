#!/usr/bin/env python3
"""Say what an EMPIAR entry holds, and check a download layout, before anything is downloaded.

    python3 scripts/show_empiar.py <id>                    # the entry and its folders
    python3 scripts/show_empiar.py <id> <folder>           # one folder of data/, deeper
    python3 scripts/show_empiar.py <id> --layout <yaml> --series A --series B
    python3 scripts/show_empiar.py <id> ... --json

Read-only: it reads the EMPIAR API, EBI's listings and one mdoc per series,
into memory, and writes nothing.

Without --layout it describes the entry. It shows the image sets the EMPIAR API
declares (folder, category, format, number of series, pixel size), then the
folders of data/ to two levels. A folder with many subfolders is shown by three
of them: the first, the middle and the last. A folder of mdocs is shown by the
series ids its names give.

With --layout and --series it checks a layout, written in the form of
configs/empiar-layouts/example.yaml. The files are the ones `cryoet
download-empiar` would take, chosen by the same function
(build_empiar_file_list), and a refusal is printed in its words. The checks
come from elsewhere:
- each series' mdoc is read, and the frames its SubFramePath lines name are
  compared with the frames the layout takes;
- the format the entry declares is compared with the files' suffixes.
Sizes are the listings', which the server rounds.

Exit status: 0 when the entry was read, and, with --layout, every series was
accepted and agrees with its mdoc; 1 otherwise; 2 for arguments that
contradict each other.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin

from cryoet_pipeline.empiar import (
    DEFAULT_EMPIAR_10164_LAYOUT,
    DEFAULT_EMPIAR_ID,
    DatasetLayout,
    DownloadGaveUp,
    _listed_files,
    build_empiar_file_list,
    call_with_retries,
    empiar_base_url,
    extract_listing_hrefs,
    load_dataset_layout,
    read_text_url,
)
from cryoet_pipeline.ingest import parse_mdoc_text

API_URL = "https://www.ebi.ac.uk/empiar/api/entry/{empiar_id}/"
BROWSE_DEPTH = 2
SAMPLE_FOLDERS = 3
MAX_LISTINGS = 30
MOVIE_SUFFIXES = (".mrc", ".mrcs", ".tif", ".tiff", ".eer", ".st")
# The file suffixes each of the EMPIAR API's data formats is written with.
FORMAT_SUFFIXES = {
    "EER": {".eer"},
    "TIFF": {".tif", ".tiff"},
    "MRC": {".mrc", ".mrcs", ".st"},
    "MRCS": {".mrcs"},
}
# One row of an Apache listing, as EBI's server writes it: the link, the date,
# then the size, which is "-" for a folder.
_ROW = re.compile(
    r'<a href="(?P<href>[^"]+)">[^<]*</a></td><td[^>]*>[^<]*</td><td[^>]*>(?P<size>[^<]*)</td>'
)
_SIZE = re.compile(r"(?P<number>\d+(?:\.\d+)?)(?P<unit>[KMGT]?)")
_UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


class Reader:
    """Each URL read once, retried as the downloader retries, and counted."""

    def __init__(self, read: Callable[[str], str] = read_text_url, *, retries: int = 2,
                 retry_wait_s: float = 5.0, sleep: Callable[[float], None] = time.sleep,
                 limit: int = MAX_LISTINGS) -> None:
        self.read, self.retries, self.retry_wait_s, self.sleep = read, retries, retry_wait_s, sleep
        self.limit = limit
        self.pages: dict[str, str] = {}

    @property
    def spent(self) -> bool:
        return len(self.pages) >= self.limit

    def __call__(self, url: str) -> str:
        if url not in self.pages:
            self.pages[url] = call_with_retries(
                lambda attempt: self.read(url), what=url, retries=self.retries,
                retry_wait_s=self.retry_wait_s, progress=None, sleep=self.sleep,
            )
        return self.pages[url]

    def sizes(self) -> dict[str, int | None]:
        """Every file's size as some listing read so far shows it, by its full URL."""

        found: dict[str, int | None] = {}
        for url, html in self.pages.items():
            for row in _ROW.finditer(html):
                found[urljoin(url, row["href"])] = parse_size(row["size"])
        return found


def parse_size(text: str) -> int | None:
    found = _SIZE.fullmatch(text.strip())
    if found is None:
        return None
    return int(float(found["number"]) * _UNITS[found["unit"]])


def suffix_of(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".mdoc"):
        inner = PurePosixPath(lower[: -len(".mdoc")]).suffix
        return f"{inner}.mdoc" if inner in MOVIE_SUFFIXES else ".mdoc"
    return PurePosixPath(lower).suffix or "(none)"


def mdoc_series(name: str) -> str | None:
    for suffix in (".mrc.mdoc", ".mdoc"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def size_text(size: int | None) -> str:
    """In units of 1024, as the listings and the EMPIAR API count them."""

    if size is None:
        return "size not listed"
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def sample(names: list[str], first: int = 3, last: int = 2) -> list[str]:
    if len(names) <= first + last + 1:
        return names
    return [*names[:first], "...", *names[-last:]]


# --- The entry -----------------------------------------------------------------------


def read_entry(reader: Reader, empiar_id: str) -> dict[str, Any] | None:
    """The entry as the EMPIAR API gives it, or None when it has no such entry."""

    try:
        text = reader(API_URL.format(empiar_id=empiar_id))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    entries = json.loads(text)
    if not isinstance(entries, dict) or len(entries) != 1:
        raise ValueError(f"the EMPIAR API gave no single entry for {empiar_id}")
    (entry,) = entries.values()
    keep = ("name", "directory", "category", "data_format", "header_format",
            "num_images_or_tilt_series", "frames_per_image", "pixel_width", "voxel_type",
            "details")
    return {
        "title": entry.get("title"),
        "dataset_size": entry.get("dataset_size"),
        "release_date": entry.get("release_date"),
        "imagesets": [{key: imageset.get(key) for key in keep}
                      for imageset in entry.get("imagesets") or []],
    }


# --- Folders ----------------------------------------------------------------------------


@dataclass
class RemoteFolder:
    path: str  # relative to data/; "" for data/ itself
    url: str
    folders: list[str] = field(default_factory=list)
    files: dict[str, int | None] = field(default_factory=dict)
    not_read: int = 0  # subfolders left out of the description
    error: str | None = None


def folder_url(base_url: str, path: str) -> str:
    path = path.strip("/")
    return urljoin(base_url, f"{quote(path)}/") if path else base_url


def read_folder(reader: Reader, base_url: str, path: str) -> RemoteFolder:
    url = folder_url(base_url, path)
    folder = RemoteFolder(path=path.strip("/"), url=url)
    try:
        html = reader(url)
    except HTTPError as exc:
        folder.error = ("no such folder (HTTP 404)" if exc.code == 404
                        else f"HTTP {exc.code} {exc.reason}")
        return folder
    except (DownloadGaveUp, URLError, OSError) as exc:
        folder.error = str(exc)
        return folder
    # The downloader's own rule for which links are files, and their names.
    names = _listed_files(lambda _url: html, url)
    sizes = {unquote(row["href"]): parse_size(row["size"]) for row in _ROW.finditer(html)}
    folder.files = {name: sizes.get(name) for name in sorted(names)}
    folder.folders = sorted(
        unquote(href[:-1]) for href in extract_listing_hrefs(html)
        if href.endswith("/") and not href.startswith("/") and "://" not in href
    )
    return folder


def browse(reader: Reader, base_url: str, start: str, depth: int) -> list[RemoteFolder]:
    """The folder and its subfolders to `depth` levels; three of many."""

    shown: list[RemoteFolder] = []

    def visit(path: str, level: int) -> None:
        folder = read_folder(reader, base_url, path)
        shown.append(folder)
        if level >= depth or folder.error:
            return
        names = folder.folders
        picks = (names if len(names) <= SAMPLE_FOLDERS + 1
                 else [names[0], names[len(names) // 2], names[-1]])
        for name in picks:
            if reader.spent:
                folder.not_read += 1
                continue
            visit(f"{path.strip('/')}/{name}".strip("/"), level + 1)
        folder.not_read += len(names) - len(picks)

    visit(start, 0)
    return shown


def describe_folder(folder: RemoteFolder) -> dict[str, Any]:
    by_suffix: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    unsized = 0
    for name, size in folder.files.items():
        suffix = suffix_of(name)
        by_suffix[suffix] += 1
        if size is None:
            unsized += 1
        else:
            sizes[suffix] += size
    ids = sorted(filter(None, (mdoc_series(name) for name in folder.files)))
    return {
        "path": folder.path, "error": folder.error, "folders": len(folder.folders),
        "folder_names": sample(folder.folders), "not_read": folder.not_read,
        "files": {suffix: {"count": n, "size_bytes": sizes[suffix]}
                  for suffix, n in by_suffix.most_common()},
        "unsized": unsized, "file_names": sample(list(folder.files)),
        "series_from_mdocs": {"count": len(ids), "ids": sample(ids, 4, 2)} if ids else None,
    }


# --- A layout ---------------------------------------------------------------------------


@dataclass
class SeriesCheck:
    series: str
    verdict: str = "accepted"  # "accepted" or "refused"
    refusal: str | None = None
    folders: dict[str, dict[str, Any]] = field(default_factory=dict)
    size_bytes: int = 0
    unsized: int = 0
    mdoc: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def taken_by_folder(files: list[Any], sizes: dict[str, int | None]) -> dict[str, dict[str, Any]]:
    folders: dict[str, dict[str, Any]] = {}
    for remote in files:
        local = str(PurePosixPath(*remote.relative_path.parts[:-1])) or "."
        entry = folders.setdefault(local, {"files": Counter(), "size_bytes": 0, "unsized": 0})
        entry["files"][suffix_of(remote.relative_path.name)] += 1
        size = sizes.get(remote.url)
        if size is None:
            entry["unsized"] += 1
        else:
            entry["size_bytes"] += size
    return {local: {**entry, "files": dict(entry["files"])} for local, entry in folders.items()}


def compare_with_mdoc(row: SeriesCheck, reader: Reader, files: list[Any]) -> None:
    """The frames the series' mdoc names, against the frames the layout takes."""

    mdocs = [remote for remote in files if remote.relative_path.name.lower().endswith(".mdoc")]
    movies = sorted(remote.relative_path.name for remote in files
                    if suffix_of(remote.relative_path.name) in MOVIE_SUFFIXES)
    if len(mdocs) != 1:
        row.mdoc["note"] = (f"the layout takes {len(mdocs)} mdocs for this series, so the "
                            "frames are not compared with one")
        if mdocs or movies:
            row.verdict = "refused" if len(mdocs) > 1 else row.verdict
        return
    (remote,) = mdocs
    try:
        text = reader(remote.url)
        parsed = parse_mdoc_text(text, source_mdoc=Path(remote.relative_path.name),
                                 tilt_series_id=row.series)
    except KeyError as exc:
        row.verdict = "refused"
        row.mdoc["note"] = (f"{remote.relative_path.name}: a ZValue section has no "
                            f"{exc.args[0]} line, which the pipeline's mdoc parser requires")
        return
    except (ValueError, HTTPError, URLError, OSError) as exc:
        row.verdict = "refused"
        row.mdoc["note"] = f"{remote.relative_path.name}: {exc}"
        return
    named = sorted({PurePosixPath(image.subframe_path.replace("\\", "/")).name
                    for image in parsed.images})
    row.mdoc.update(file=remote.relative_path.name, tilts=len(parsed.images),
                    frames_taken=len(movies))
    stack = [name for name in movies
             if PurePosixPath(name).stem == row.series and len(movies) == 1]
    if stack:
        row.mdoc["note"] = (f"{stack[0]} is one file for the series, an assembled stack: its "
                            "images are counted after the download (show_folder.py)")
        return
    missing = sorted(set(named) - set(movies))
    extra = sorted(set(movies) - set(named))
    row.mdoc.update(missing=missing[:5], missing_count=len(missing), extra=extra[:5],
                    extra_count=len(extra))
    if missing or extra:
        row.verdict = "refused"


def check_layout(reader: Reader, empiar_id: str, layout: DatasetLayout, series: list[str],
                 entry: dict[str, Any] | None) -> dict[str, Any]:
    def files_for(wanted: list[str], groups: DatasetLayout) -> list[Any]:
        return build_empiar_file_list(empiar_id, wanted, groups, listing_reader=reader,
                                      retries=0)

    shared_groups = tuple(group for group in layout.groups if group.shared)
    shared: list[Any] = []
    result: dict[str, Any] = {"series": [], "shared": {}, "warnings": []}
    if shared_groups:
        try:
            shared = files_for(series[:1], DatasetLayout(groups=shared_groups))
        except (ValueError, DownloadGaveUp, HTTPError, URLError) as exc:
            result["shared_refusal"] = str(exc)
    shared_urls = {remote.url for remote in shared}
    taken: list[Any] = []
    for name in series:
        row = SeriesCheck(series=name)
        try:
            files = [remote for remote in files_for([name], layout)
                     if remote.url not in shared_urls]
        except (ValueError, DownloadGaveUp, HTTPError, URLError) as exc:
            row.verdict, row.refusal = "refused", str(exc)
            result["series"].append(asdict(row))
            continue
        sizes = reader.sizes()
        row.folders = taken_by_folder(files, sizes)
        row.size_bytes = sum(entry["size_bytes"] for entry in row.folders.values())
        row.unsized = sum(entry["unsized"] for entry in row.folders.values())
        taken.extend(files)
        compare_with_mdoc(row, reader, files)
        result["series"].append(asdict(row))
    sizes = reader.sizes()
    result["shared"] = taken_by_folder(shared, sizes)
    # Where a shared file comes from, so that a gain of another session shows.
    base_url = empiar_base_url(empiar_id)
    result["shared_from"] = sorted({
        "data/" + unquote(remote.url[len(base_url):]).rsplit("/", 1)[0] for remote in shared
    })
    result["size_bytes"] = (sum(row["size_bytes"] for row in result["series"])
                            + sum(entry["size_bytes"] for entry in result["shared"].values()))

    result["warnings"] = format_warnings(empiar_base_url(empiar_id), taken, entry)
    return result


def format_warnings(base_url: str, files: list[Any], entry: dict[str, Any] | None) -> list[str]:
    """The format an image set declares, against the suffixes of the files taken from it."""

    found = []
    for imageset in (entry or {}).get("imagesets", []):
        directory = str(imageset.get("directory") or "").strip("/")
        directory = directory[len("data/"):] if directory.startswith("data/") else directory
        declared = str(imageset.get("data_format") or "").upper()
        expected = FORMAT_SUFFIXES.get(declared)
        if not directory or not expected:
            continue
        inside = {suffix_of(remote.relative_path.name) for remote in files
                  if unquote(remote.url[len(base_url):]).startswith(directory + "/")}
        movies = inside & set(MOVIE_SUFFIXES)
        if movies and not movies & expected:
            found.append(f"the entry declares {declared} for data/{directory}, and the layout "
                         f"takes {', '.join(sorted(movies))} files from it")
    return found


# --- Output -----------------------------------------------------------------------------


def build(args: argparse.Namespace, reader: Reader) -> dict[str, Any]:
    report: dict[str, Any] = {"empiar_id": args.empiar_id, "exit": 1, "entry": None,
                              "folders": [], "layout": None, "check": None, "command": None}
    base_url = empiar_base_url(args.empiar_id)
    try:
        report["entry"] = read_entry(reader, args.empiar_id)
    except (HTTPError, URLError, DownloadGaveUp, ValueError) as exc:
        report["entry_error"] = str(exc)

    if args.layout is None and not args.series:
        start = args.folder or ""
        folders = browse(reader, base_url, start, args.depth)
        report["folders"] = [describe_folder(folder) for folder in folders]
        report["listings_read"] = len(reader.pages)
        if report["entry"] is not None and not folders[0].error:
            report["exit"] = 0
        return report

    if args.layout is not None:
        layout = load_dataset_layout(args.layout)
        report["layout"] = str(args.layout)
    else:
        layout = DEFAULT_EMPIAR_10164_LAYOUT
        report["layout"] = "the built-in EMPIAR-10164 layout"
    report["groups"] = [asdict(group) for group in layout.groups]
    report["check"] = check_layout(reader, args.empiar_id, layout, args.series,
                                   report["entry"])
    accepted = all(row["verdict"] == "accepted" for row in report["check"]["series"])
    if accepted and "shared_refusal" not in report["check"]:
        report["exit"] = 0
        command = ["cryoet", "download-empiar", "--empiar-id", args.empiar_id,
                   "--out", args.out or f"data/empiar-{args.empiar_id}"]
        if args.layout is not None:
            command += ["--layout", str(args.layout)]
        for name in args.series:
            command += ["--tilt-series", name]
        report["command"] = " ".join(command)
    return report


def entry_lines(report: dict[str, Any]) -> list[str]:
    entry = report["entry"]
    if entry is None:
        why = report.get("entry_error") or "the EMPIAR API has no such entry"
        return [f"EMPIAR-{report['empiar_id']}: {why}"]
    lines = [f"EMPIAR-{report['empiar_id']}: {entry['title']}"]
    if entry.get("dataset_size"):
        lines[0] += f" ({entry['dataset_size']})"
    for imageset in entry["imagesets"]:
        facts = [str(imageset.get("category")), str(imageset.get("data_format"))]
        if imageset.get("num_images_or_tilt_series"):
            facts.append(f"{imageset['num_images_or_tilt_series']} series or images")
        if imageset.get("frames_per_image"):
            facts.append(f"{imageset['frames_per_image']} frames per image")
        if imageset.get("pixel_width"):
            facts.append(f"{imageset['pixel_width']} A/px")
        lines.append(f"  {imageset.get('directory')}: {', '.join(facts)}")
    return lines


def render(report: dict[str, Any]) -> str:
    out = entry_lines(report)
    if report["check"] is None:
        out.append("")
        for folder in report["folders"]:
            name = "data/" + (folder["path"] + "/" if folder["path"] else "")
            if folder["error"]:
                out.append(f"{name}: {folder['error']}")
                continue
            parts = []
            if folder["folders"]:
                parts.append(f"{folder['folders']} folders ({', '.join(folder['folder_names'])})")
            for suffix, files in folder["files"].items():
                size = f", {size_text(files['size_bytes'])}" if files["size_bytes"] else ""
                parts.append(f"{files['count']} {suffix}{size}")
            out.append(f"{name}: {'; '.join(parts) or 'empty'}")
            if folder["files"]:
                out.append(f"    files: {', '.join(folder['file_names'])}")
            if folder["series_from_mdocs"]:
                ids = folder["series_from_mdocs"]
                out.append(f"    series by their mdocs: {ids['count']} ({', '.join(ids['ids'])})")
            if folder["not_read"]:
                out.append(f"    {folder['not_read']} subfolders not read")
        out.append(f"({report.get('listings_read', 0)} listings read; sizes as listed, rounded)")
        return "\n".join(out)

    check = report["check"]
    out += ["", f"Layout: {report['layout']}"]
    for group in report["groups"]:
        where = group["local_dir"] or f"data/{group['remote_dir']}"
        shared = ", once for all series" if group["shared"] else ""
        out.append(f"  {group['remote_dir']}/{group['file_pattern']} -> {where}{shared}")
    out.append("")
    for row in check["series"]:
        if row["refusal"]:
            out.append(f"{row['series']}: REFUSED: {row['refusal']}")
            continue
        taken = "; ".join(
            f"{folder}/: " + ", ".join(f"{n} {suffix}" for suffix, n in entry["files"].items())
            for folder, entry in row["folders"].items()
        )
        out.append(f"{row['series']}: {taken}, {size_text(row['size_bytes'])}, {row['verdict']}")
        mdoc = row["mdoc"]
        if "tilts" in mdoc and "missing_count" in mdoc:
            line = f"    mdoc {mdoc['file']}: {mdoc['tilts']} tilts; the layout takes " \
                   f"{mdoc['frames_taken']} frames"
            if mdoc["missing_count"]:
                line += (f"; {mdoc['missing_count']} it names are not taken "
                         f"({', '.join(mdoc['missing'])})")
            if mdoc["extra_count"]:
                line += (f"; {mdoc['extra_count']} taken are not in it "
                         f"({', '.join(mdoc['extra'])})")
            out.append(line)
        elif "tilts" in mdoc:
            out.append(f"    mdoc {mdoc['file']}: {mdoc['tilts']} tilts. {mdoc.get('note', '')}")
        elif mdoc.get("note"):
            out.append(f"    {mdoc['note']}")
    if check.get("shared_refusal"):
        out.append(f"shared: REFUSED: {check['shared_refusal']}")
    for folder, entry in check["shared"].items():
        files = ", ".join(f"{n} {suffix}" for suffix, n in entry["files"].items())
        out.append(f"once for all series, {folder}/: {files}, {size_text(entry['size_bytes'])}")
    if check.get("shared_from"):
        out.append(f"    from {', '.join(check['shared_from'])}: check that it belongs to "
                   "every series asked for")
    out.append(f"In all: {size_text(check['size_bytes'])}, as the listings show sizes (rounded)")
    for warning in check["warnings"]:
        out.append(f"Warning: {warning}")
    if report["command"]:
        out += ["", "The download, once the person has said yes (it resumes when run again):",
                f"  {report['command']}"]
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("empiar_id", help="the entry's number, such as 11830")
    parser.add_argument("folder", nargs="?", help="a folder of data/ to describe")
    parser.add_argument("--layout", type=Path, help="a download layout to check")
    parser.add_argument("--series", action="append", default=[],
                        help="a series id to check the layout for; repeat for several")
    parser.add_argument("--out", help="the download folder, for the printed command")
    parser.add_argument("--depth", type=int, default=BROWSE_DEPTH,
                        help=f"folder levels to describe (default {BROWSE_DEPTH})")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, reader: Reader | None = None) -> int:
    args = parse_args(argv)
    if not args.empiar_id.isdigit():
        print(f"an EMPIAR id is a number, not {args.empiar_id!r}", file=sys.stderr)
        return 2
    checking = args.layout is not None or bool(args.series)
    if checking and args.folder:
        print("give a folder to describe, or a layout to check, not both", file=sys.stderr)
        return 2
    if checking and not args.series:
        print("give --series: the ids are the names the mdocs give, which the entry's "
              "description shows", file=sys.stderr)
        return 2
    if checking and args.layout is None and args.empiar_id != DEFAULT_EMPIAR_ID:
        print(f"give --layout: only EMPIAR-{DEFAULT_EMPIAR_ID} has a built-in layout",
              file=sys.stderr)
        return 2
    if args.layout is not None and not args.layout.is_file():
        print(f"no such layout file: {args.layout}", file=sys.stderr)
        return 2
    if len(set(args.series)) != len(args.series):
        print("a series is given twice", file=sys.stderr)
        return 2
    try:
        report = build(args, reader or Reader())
    except ValueError as exc:  # a layout the loader refuses
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return int(report["exit"])


if __name__ == "__main__":
    sys.exit(main())
