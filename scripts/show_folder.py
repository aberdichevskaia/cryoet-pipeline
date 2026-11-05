#!/usr/bin/env python3
"""Say what a folder of raw tilt-series data holds, read the way a lane will read it.

    python3 scripts/show_folder.py <folder>
    python3 scripts/show_folder.py <folder> --pattern '<regular expression>'
    python3 scripts/show_folder.py <folder> --mdocs <mdoc folder>
    python3 scripts/show_folder.py <folder> --series 039 --series 096 --json

Read-only: it lists files, reads headers and mdocs, and writes nothing.

It answers the three questions of docs/agent/data_folders.md: how the files are
named, what one file holds, and which lane input they make. The verdict is the
pipeline's own, not a second reading. Series, tilts, frames and pixel size come
from the functions `cryoet init` runs (find_tilt_series,
load_tilt_series_manifest, validate_tilt_series_manifest), and a stack's from
the checks import-tilt-stack makes before it writes anything
(check_external_tilt_stack). A refusal is printed in their words. What this
script adds is description: the folder tree, the fields of the names, a look
inside a few files, and the warnings data_folders.md lists, such as a series
whose angles all have one sign.

Where the tilt angles come from:
- --pattern: the movie file names, as [data] frames_pattern;
- otherwise mdocs: --mdocs, or the one folder of mdocs found in <folder>, under
  it, or beside it;
- neither: the names are described and nothing is decided, so that a pattern
  can be written from the fields shown and checked with --pattern.

Exit status: 0 when every series was opened and accepted, and a config fragment
is printed; 1 when something was refused, the data has no entry, a series was
left unopened, or there is not enough to decide; 2 for a folder that does not
exist or arguments that contradict each other.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cryoet_pipeline.external_stack import check_external_tilt_stack
from cryoet_pipeline.ingest import parse_mdoc_text
from cryoet_pipeline.ingest.frame_names import compile_frame_pattern, match_frames
from cryoet_pipeline.models import ProjectConfig, TiltImage, TiltSeriesManifest
from cryoet_pipeline.movies import inspect_movie
from cryoet_pipeline.mrc_validation import inspect_mrc_file
from cryoet_pipeline.project import (
    find_tilt_series,
    load_tilt_series_manifest,
    validate_tilt_series_manifest,
)

ROOT = Path(__file__).resolve().parents[1]

# Files that can hold tilt data: the movies movies.py reads (MRC, TIFF, EER)
# and assembled stacks, which import-tilt-stack reads as MRC.
MOVIE_SUFFIXES = (".mrc", ".mrcs", ".tif", ".tiff", ".eer")
STACK_SUFFIXES = (".mrc", ".st")
DATA_SUFFIXES = frozenset(MOVIE_SUFFIXES + STACK_SUFFIXES)
# A gain reference is recognised by its name only, and reported as a guess.
GAIN_SUFFIXES = frozenset({".dm4", ".gain"})
GAIN_WORDS = ("gain", "countref", "superref")
# The K3's super-resolution frame, x by y; its physical pixel gives 5760 x 4092.
K3_SUPER_RESOLUTION = frozenset({(11520, 8184), (8184, 11520)})
TREE_DEPTH = 2
TREE_FOLDERS = 30
DEFAULT_MAX_SERIES = 20
MAX_MDOCS = 200
# An mdoc keeps the suffix before .mdoc, since init reads only <id>.mrc.mdoc.
_MDOC_INNER = (".mrc", ".st", ".tif", ".tiff", ".eer")
_SPLIT = re.compile(r"[_\[\]]+")
_KINDS = (
    ("digits", re.compile(r"\d+")),
    ("number with a sign or point", re.compile(r"[-+]\d+(?:\.\d+)?|\d+\.\d+")),
    ("date", re.compile(r"[A-Z][a-z]{2}\d{1,2}")),
    ("time", re.compile(r"\d{1,2}\.\d{2}\.\d{2}")),
)
_NUMERIC = frozenset({"digits", "number with a sign or point"})
_T_LINE = re.compile(r"^\[T\s*=\s*(?P<text>.+?)\]\s*$")
# init never writes here; ProjectConfig only needs a path.
_NO_OUTPUT = Path(os.devnull)


@dataclass
class Folder:
    path: str
    depth: int
    suffixes: dict[str, int] = field(default_factory=dict)
    size_bytes: int = 0
    folders: int = 0
    links_to_folders: list[str] = field(default_factory=list)
    broken_links: int = 0
    gains: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Series:
    series: str
    read_as: str  # "names", "frames" (by mdoc), "stack" (by mdoc), "mdoc" (undecided)
    tilts: int = 0
    angle_min: float | None = None
    angle_max: float | None = None
    steps: dict[str, int] = field(default_factory=dict)
    first_angles: list[float] = field(default_factory=list)
    order: str = ""  # what first_angles are ordered by: "tilt number" or "ZValue"
    monotonic: bool | None = None
    frames: list[int] = field(default_factory=list)
    pixel: float | None = None
    verdict: str = "not opened"  # "accepted", "refused" or "not opened"
    refusal: str | None = None
    notes: list[str] = field(default_factory=list)
    mdoc: dict[str, Any] = field(default_factory=dict)


# --- The tree ------------------------------------------------------------------


def suffix_of(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".mdoc"):
        for inner in _MDOC_INNER:
            if lower.endswith(inner + ".mdoc"):
                return inner + ".mdoc"
        return ".mdoc"
    return Path(lower).suffix or "(none)"


def read_tree(root: Path, depth: int = TREE_DEPTH) -> list[Folder]:
    """The folder and its subfolders to `depth` levels; hidden names are skipped."""

    found: list[Folder] = []

    def visit(path: Path, level: int) -> None:
        folder = Folder(path=str(path), depth=level)
        found.append(folder)
        try:
            entries = sorted(os.scandir(path), key=lambda entry: entry.name)
        except OSError as exc:
            folder.error = str(exc)
            return
        counts: Counter[str] = Counter()
        subfolders: list[Path] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                subfolders.append(Path(entry.path))
            elif entry.is_dir():
                folder.links_to_folders.append(entry.name)
            elif entry.is_file():
                counts[suffix_of(entry.name)] += 1
                try:
                    folder.size_bytes += entry.stat().st_size
                except OSError:
                    pass
                lower = entry.name.lower()
                if Path(lower).suffix in GAIN_SUFFIXES or any(w in lower for w in GAIN_WORDS):
                    folder.gains.append(entry.name)
            elif entry.is_symlink():
                folder.broken_links += 1
        folder.suffixes = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
        folder.folders = len(subfolders)
        if level < depth:
            for sub in subfolders:
                visit(sub, level + 1)

    visit(root, 0)
    return found


def data_count(folder: Folder) -> int:
    return sum(n for suffix, n in folder.suffixes.items() if suffix in DATA_SUFFIXES)


def mdoc_count(folder: Folder) -> int:
    return sum(n for suffix, n in folder.suffixes.items() if suffix.endswith(".mdoc"))


def pick_frames_folder(tree: list[Folder]) -> tuple[Folder | None, list[Folder]]:
    """The folder of data files: the given one, or the only one under it.

    "The only one" holds only when the whole tree was read: a folder at the
    depth limit with folders of its own may hide more data.
    """

    if data_count(tree[0]):
        return tree[0], []
    holders = [folder for folder in tree if data_count(folder)]
    deeper = any(folder.depth == TREE_DEPTH and folder.folders for folder in tree)
    if len(holders) == 1 and not deeper:
        return holders[0], []
    return None, holders


def find_mdocs_folder(frames: Folder, tree: list[Folder]) -> tuple[Path | None, str, list[str]]:
    """The mdoc folder and where it was found, or the candidates when they are several."""

    if mdoc_count(frames):
        return Path(frames.path), "among the data files", []
    under = [folder.path for folder in tree if mdoc_count(folder)]
    if len(under) == 1:
        return Path(under[0]), "under the folder", []
    if under:
        return None, "", under
    parent = Path(frames.path).parent
    beside: list[str] = []
    try:
        for entry in sorted(os.scandir(parent), key=lambda entry: entry.name):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            if Path(entry.path) == Path(frames.path):
                continue
            try:
                names = os.listdir(entry.path)
            except OSError:
                continue
            if any(name.lower().endswith(".mdoc") for name in names):
                beside.append(entry.path)
    except OSError:
        pass
    if len(beside) == 1:
        return Path(beside[0]), "in a folder beside the data", []
    return None, "", beside


# --- Names -----------------------------------------------------------------------


def data_names(frames: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in os.scandir(frames)
        if not entry.name.startswith(".")
        and entry.is_file()
        and suffix_of(entry.name) in DATA_SUFFIXES
    )


def sample(names: list[str], first: int = 5, middle: int = 3, last: int = 3) -> list[str]:
    if len(names) <= first + middle + last + 2:
        return names
    start = len(names) // 2 - middle // 2
    return [*names[:first], "...", *names[start:start + middle], "...", *names[-last:]]


def kind_of(value: str) -> str:
    for name, regex in _KINDS:
        if regex.fullmatch(value):
            return name
    return "text"


def describe_field(values: list[str]) -> dict[str, Any]:
    distinct = sorted(set(values))
    kinds = sorted({kind_of(value) for value in distinct})
    entry: dict[str, Any] = {"distinct": len(distinct), "kind": " / ".join(kinds)}
    if set(kinds) <= _NUMERIC:
        by_value = sorted(distinct, key=float)
        entry["min"], entry["max"] = by_value[0], by_value[-1]
        if kinds == ["digits"]:
            entry["widths"] = sorted({len(value) for value in distinct})
    entry["examples"] = distinct[:3]
    return entry


def field_tables(names: list[str]) -> list[dict[str, Any]]:
    """The names split at _ [ and ], each field described, by number of fields."""

    groups: dict[int, list[list[str]]] = defaultdict(list)
    for name in names:
        stem = name[: len(name) - len(Path(name).suffix)]
        parts = [part for part in _SPLIT.split(stem) if part]
        groups[len(parts)].append(parts)
    tables = []
    for count, rows in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:3]:
        tables.append({
            "fields": count,
            "names": len(rows),
            "table": [describe_field([row[i] for row in rows]) for i in range(count)],
        })
    return tables


def match_for_display(
    frames: Path, compiled: re.Pattern[str]
) -> tuple[dict[str, list[tuple[int, float]]], dict[str, list[str]]]:
    """What the pattern matches, by series, and what it misses, by suffix.

    The same whole-name match as match_frames, for showing what it found where
    match_frames refuses the folder; the verdict is always match_frames' own.
    """

    by_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    missed: dict[str, list[str]] = defaultdict(list)
    names = sorted(
        entry.name for entry in os.scandir(frames)
        if entry.is_file() and not entry.name.startswith(".")
    )
    for name in names:
        found = compiled.fullmatch(name)
        if found is None:
            missed[Path(name).suffix or "(none)"].append(name)
            continue
        try:
            by_series[found["series"]].append((int(found["tilt"]), float(found["angle"])))
        except ValueError:
            missed["(tilt or angle is not a number)"].append(name)
    return dict(by_series), dict(missed)


# --- Series ------------------------------------------------------------------------


def _num(value: float) -> str:
    return f"{value + 0.0:g}"


def fill_angles(row: Series, angles: list[float], order: str) -> None:
    if not angles:
        return
    ordered = sorted(angles)
    row.tilts = len(angles)
    row.angle_min, row.angle_max = ordered[0], ordered[-1]
    # Stage readings in mdocs wander by a few hundredths of a degree.
    steps = Counter(_num(round(b - a, 1)) for a, b in zip(ordered, ordered[1:], strict=False))
    row.steps = dict(steps.most_common())
    row.first_angles = angles[:6]
    row.order = order
    if len(angles) > 2:
        pairs = list(zip(angles, angles[1:], strict=False))
        row.monotonic = all(b > a for a, b in pairs) or all(b < a for a, b in pairs)


def accept(row: Series, manifest: TiltSeriesManifest) -> None:
    row.verdict = "accepted"
    row.frames = sorted(manifest.num_subframes_set)
    row.pixel = manifest.raw_pixel_spacing_angstrom
    row.notes.extend(manifest.notes)


def read_by_pattern(
    frames: Path, pattern: str, wanted: list[str], max_series: int, report: dict[str, Any]
) -> list[Series]:
    try:
        compiled = compile_frame_pattern(pattern)
    except ValueError as exc:
        report["refusal"] = str(exc)
        return []
    shown, missed = match_for_display(frames, compiled)
    report["unmatched"] = {suffix: names[:10] for suffix, names in sorted(missed.items())}
    report["unmatched_counts"] = {suffix: len(names) for suffix, names in sorted(missed.items())}
    try:
        by_series = match_frames(frames, pattern)
    except (ValueError, FileNotFoundError) as exc:
        # init refuses the whole folder; show what the pattern did match.
        report["refusal"] = str(exc)
        rows = []
        for series, pairs in sorted(shown.items()):
            row = Series(series, "names", verdict="refused",
                         refusal="init refuses the whole folder: see above")
            fill_angles(row, [angle for _, angle in sorted(pairs)], "tilt number")
            rows.append(row)
        return rows
    ids = wanted or sorted(by_series)
    try:
        config = ProjectConfig(frames_dir=frames, frames_pattern=pattern,
                               output_dir=_NO_OUTPUT, tilt_series=ids)
    except ValidationError as exc:
        report["refusal"] = str(exc)
        return []
    rows = []
    for index, series in enumerate(ids):
        row = Series(series, "names")
        fill_angles(row, [frame.angle_deg for frame in by_series.get(series, [])], "tilt number")
        rows.append(row)
        if index < max_series:
            init_check(row, config)
    return rows


def init_check(row: Series, config: ProjectConfig) -> None:
    """The verdict of init's own functions on one series, without writing anything."""

    try:
        manifest = load_tilt_series_manifest(config, row.series)
        validate_tilt_series_manifest(manifest)
    except (ValueError, FileNotFoundError, OSError) as exc:
        row.verdict, row.refusal = "refused", str(exc)
        return
    accept(row, manifest)
    if not row.tilts:
        images = sorted(manifest.images, key=lambda image: image.z_value)
        fill_angles(row, [image.tilt_angle_deg for image in images], "ZValue")


def mdoc_files(mdocs: Path) -> dict[str, Path]:
    """The mdocs by series id, for showing them: <id>.mrc.mdoc, else <id>.mdoc.

    Which of them init reads, and as which series, is init's to say: the rows
    take their ids from find_tilt_series and their verdicts from
    load_tilt_series_manifest. A stack's mdoc is any file [data] mdoc names.
    """

    found: dict[str, Path] = {}
    for entry in sorted(os.scandir(mdocs), key=lambda entry: entry.name):
        if entry.name.startswith(".") or not entry.is_file():
            continue
        for suffix in (".mrc.mdoc", ".mdoc"):
            if entry.name.endswith(suffix) and len(entry.name) > len(suffix):
                series = entry.name[: -len(suffix)]
                if series not in found or suffix == ".mrc.mdoc":
                    found[series] = Path(entry.path)
                break
    return found


def mdoc_header(text: str) -> dict[str, str]:
    """The key = value lines before the first [section], as parse_mdoc_text reads them."""

    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            break
        if "=" in line:
            key, value = (part.strip() for part in line.split("=", 1))
            fields[key] = value
    return fields


def _metadata_numbers(images: list[TiltImage], key: str) -> list[float]:
    numbers = []
    for image in images:
        try:
            numbers.append(float(image.metadata[key]))
        except (KeyError, TypeError, ValueError):
            pass
    return numbers


def mdoc_facts(text: str, images: list[TiltImage]) -> dict[str, Any]:
    """What an mdoc says that a lane config needs, as the parser read it."""

    header = mdoc_header(text)
    facts: dict[str, Any] = {key: header[key] for key in ("Voltage", "PixelSpacing")
                             if key in header}
    for raw_line in text.splitlines():
        found = _T_LINE.match(raw_line.strip())
        if found:
            facts["written_by"] = " ".join(found["text"].split())
            break
    facts["NumSubFrames"] = sorted({image.num_subframes for image in images})
    axes = sorted({round(image.rotation_angle_deg, 2) for image in images
                   if image.rotation_angle_deg is not None})
    facts["tilt_axis_from_mdoc"] = bool(axes)
    if axes:
        facts["RotationAngle"] = axes
    else:
        kept = sorted({round(value, 2) for value in _metadata_numbers(images, "RotationAngle")})
        if kept:
            facts["RotationAngle"] = kept
    doses = [image.exposure_dose for image in images if image.exposure_dose is not None]
    if doses:
        facts["ExposureDose"] = {"per_tilt": [round(min(doses), 3), round(max(doses), 3)],
                                 "total": round(sum(doses), 2)}
    targets = _metadata_numbers(images, "TargetDefocus")
    if targets:
        facts["TargetDefocus"] = [min(targets), max(targets)]
    return facts


def read_one_mdoc(frames: Path, config: ProjectConfig, series: str, mdoc_path: Path | None,
                  opened: bool, init_refusal: str | None) -> Series:
    row = Series(series, "mdoc")
    if mdoc_path is None:
        # init found a series this listing did not; its own verdict is all there is.
        row.read_as = "frames"
        if opened:
            init_check(row, config)
        return row
    row.mdoc["file"] = mdoc_path.name
    text = mdoc_path.read_text(errors="replace")
    try:
        parsed = parse_mdoc_text(text, source_mdoc=mdoc_path, tilt_series_id=series,
                                 frames_dir=frames)
    except KeyError as exc:
        row.verdict = "refused"
        row.refusal = (f"{mdoc_path.name}: a ZValue section has no {exc.args[0]} line, "
                       "which the pipeline's mdoc parser requires")
        return row
    except ValueError as exc:
        row.verdict, row.refusal = "refused", f"{mdoc_path.name}: {exc}"
        return row
    images = sorted(parsed.images, key=lambda image: image.z_value)
    fill_angles(row, [image.tilt_angle_deg for image in images], "ZValue")
    row.mdoc.update(mdoc_facts(text, images))
    present = sum(1 for image in images
                  if image.local_frame_file is not None and image.local_frame_file.is_file())
    row.mdoc["frames_found"] = present
    stack = next((frames / f"{series}{suffix}" for suffix in STACK_SUFFIXES
                  if (frames / f"{series}{suffix}").is_file()), None)

    if present and (present == len(images) or stack is None):
        row.read_as = "frames"
        if stack is not None:
            row.notes.append(f"{stack.name} is there too: an assembled stack of this series?")
        if init_refusal is not None:
            row.verdict, row.refusal = "refused", "init refuses the folder of mdocs: see above"
        elif opened:
            init_check(row, config)
        return row

    if stack is not None:
        row.read_as = "stack"
        row.mdoc["stack"] = stack.name
        if present:
            row.notes.append(f"{present} of the {len(images)} frames its SubFramePath lines "
                             "name are there too")
        if not opened:
            return row
        try:
            manifest, _ = check_external_tilt_stack(
                mdoc_path=mdoc_path, stack_path=stack, tilt_series_id=series
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            row.verdict, row.refusal = "refused", str(exc)
        else:
            row.verdict = "accepted"
            row.pixel = manifest.raw_pixel_spacing_angstrom
            row.notes.extend(manifest.notes)
        return row

    row.verdict = "refused"
    row.refusal = (f"none of the {len(images)} frames its SubFramePath lines name is in "
                   f"{frames}, and there is no {series}.mrc or {series}.st there")
    return row


def read_by_mdocs(
    frames: Path, mdocs: Path, wanted: list[str], max_series: int, report: dict[str, Any]
) -> list[Series]:
    files = mdoc_files(mdocs)
    try:
        config = ProjectConfig(frames_dir=frames, mdocs_dir=mdocs, output_dir=_NO_OUTPUT,
                               tilt_series=wanted)
    except ValidationError as exc:
        report["refusal"] = str(exc)
        return []
    report["mdoc_names"] = {
        "count": len(files),
        "suffixes": dict(Counter(suffix_of(path.name) for path in files.values())),
    }
    ids, init_refusal = list(wanted), None
    if not ids:
        try:
            ids = find_tilt_series(config)
        except (ValueError, FileNotFoundError) as exc:
            # A frames lane cannot start; a stack lane does not run init.
            init_refusal = str(exc)
            report["init_mdocs"] = init_refusal
            ids = sorted(files)
        else:
            skipped = sorted(set(files) - set(ids))
            if skipped:
                report["mdocs_init_skips"] = [files[series].name for series in skipped][:10]
    if len(ids) > MAX_MDOCS:
        report["mdocs_left_out"] = len(ids) - MAX_MDOCS
    return [
        read_one_mdoc(frames, config, series, files.get(series),
                      opened=index < max_series, init_refusal=init_refusal)
        for index, series in enumerate(ids[:MAX_MDOCS])
    ]


# --- Inside the files -------------------------------------------------------------------


def look_inside(path: Path) -> dict[str, Any]:
    """Images per file, size and dtype from the header, then the movie reader's verdict.

    The header is read raw, since inspect_movie refuses what is not a movie (a
    one-page TIFF, a 2D MRC) and the folder still has to be described. Its
    verdict, and the pixel size it reads, are the pipeline's own.
    """

    look: dict[str, Any] = {"file": path.name, "format": path.suffix.lstrip(".").upper()}
    try:
        look["size_bytes"] = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(path) as tiff:
                first = tiff.pages[0]
                look.update(format="TIFF", images=len(tiff.pages), y=int(first.shape[0]),
                            x=int(first.shape[1]), dtype=str(first.dtype))
        elif suffix in (".mrc", ".mrcs", ".st"):
            info = inspect_mrc_file(path)
            look.update(format="MRC", images=info.shape[0] if len(info.shape) == 3 else 1,
                        y=info.shape[-2], x=info.shape[-1], dtype=info.dtype,
                        pixel=info.pixel_spacing_angstrom,
                        complete=info.actual_size_bytes >= info.expected_size_bytes)
    except Exception as exc:  # an unreadable file is reported, not raised
        look["error"] = str(exc)
    try:
        movie = inspect_movie(path)
    except Exception as exc:  # the reader's refusal is part of the description
        look["movie_refusal"] = str(exc)
    else:
        frames, y, x = movie.shape
        look.setdefault("images", frames)
        look.setdefault("y", y)
        look.setdefault("x", x)
        if movie.pixel_spacing_angstrom is not None:
            look["pixel"] = movie.pixel_spacing_angstrom
    return look


# --- Conclusions ------------------------------------------------------------------------


def ids_text(ids: list[str], limit: int = 6) -> str:
    shown = ", ".join(ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def warnings_for(rows: list[Series], looks: list[dict[str, Any]],
                 report: dict[str, Any]) -> list[str]:
    found: list[str] = []
    with_angles = [row for row in rows if row.tilts > 1 and row.angle_min is not None]
    for sign, rows_of_sign in (
        ("<= 0", [r for r in with_angles if r.angle_max is not None and r.angle_max <= 0]),
        (">= 0", [r for r in with_angles if r.angle_min is not None and r.angle_min >= 0]),
    ):
        if rows_of_sign:
            found.append(
                f"every angle is {sign} in {ids_text([r.series for r in rows_of_sign])}: a "
                "tilt series spans both signs, and a pattern that misses '+' reads one side"
            )
    counts = [row.tilts for row in with_angles]
    if counts and max(counts) >= 1.5 * min(counts):
        longest = max(with_angles, key=lambda row: row.tilts)
        shortest = min(with_angles, key=lambda row: row.tilts)
        found.append(
            f"series differ in length, {shortest.series} {shortest.tilts} tilts to "
            f"{longest.series} {longest.tilts}: twice the others can be two series under "
            "one id, and a short one a wrong field split"
        )
    for row in with_angles:
        if "0" in row.steps:
            found.append(f"{row.series}: the same angle twice")
        elif len(row.steps) > 1:
            steps = ", ".join(f"{step} x{n}" for step, n in row.steps.items())
            found.append(f"{row.series}: uneven steps between neighbouring angles ({steps})")
    by_names = [row for row in with_angles if row.order == "tilt number"
                and row.monotonic is not None]
    if by_names and all(row.monotonic for row in by_names):
        found.append("the tilt number follows the angle order in every series: the "
                     "acquisition order is not in the names")
    elif by_names:
        other = [row for row in by_names if not row.monotonic]
        first = ", ".join(_num(angle) for angle in other[0].first_angles)
        found.append(
            f"the tilt number is not the angle order in {ids_text([r.series for r in other])} "
            f"(first angles by tilt number in {other[0].series}: {first}): it may be the "
            "acquisition order"
        )
    zeros = [row.series for row in rows
             if len(row.series) > 1 and row.series.isdigit() and row.series.startswith("0")]
    if zeros:
        found.append(f"series ids with leading zeros ({ids_text(zeros)}) are kept verbatim: "
                     "[run] series must spell them so")
    pixels = sorted({round(row.pixel, 4) for row in rows if row.pixel is not None})
    if len(pixels) > 1:
        found.append(f"the series have different pixel sizes: {', '.join(map(_num, pixels))} A")
    uneven = [row.series for row in rows if len(row.frames) > 1]
    if uneven:
        found.append(f"tilts with different frame counts within {ids_text(uneven)}")
    for row in rows:
        found.extend(f"{row.series}: {note}" for note in row.notes)
    if any((look.get("x"), look.get("y")) in K3_SUPER_RESOLUTION for look in looks):
        found.append("the frames are 11520 x 8184, the K3's super-resolution size: "
                     "[motion] ftbin = 2 brings the sums to the physical pixel (example.toml)")
    skipped = report.get("mdocs_init_skips")
    if skipped:
        found.append(f"init does not take these mdocs as series: {ids_text(skipped, 3)}")
    if any(row.read_as == "frames" and row.mdoc and not row.mdoc.get("tilt_axis_from_mdoc")
           for row in rows):
        found.append("the mdoc parser takes no tilt axis from these mdocs: the lane needs "
                     "[align] tilt_axis_angle")
    if report.get("mdocs_left_out"):
        found.append(f"{report['mdocs_left_out']} more mdocs were not read: use --series")
    no_data = [row.series for row in rows if row.read_as == "mdoc" and row.tilts]
    if no_data:
        found.append(f"mdocs whose frames or stack are not here: {ids_text(no_data)}; "
                     "--series reads only the others")
    unopened = [row.series for row in rows if row.verdict == "not opened"]
    if unopened:
        found.append(f"not opened, so not checked: {ids_text(unopened)}; use --series or "
                     "--max-series")
    return found


def no_entry(looks: list[dict[str, Any]]) -> str | None:
    read = [look for look in looks if "error" not in look]
    if read and all(look.get("images") == 1 for look in read):
        return ("one image per file: the tilts are already summed. A lane would need a "
                "stack assembled without motion correction, which does not exist yet "
                "(docs/roadmap.md section 22)")
    refusals = {look.get("movie_refusal", "") for look in looks}
    if looks and all("not supported" in refusal for refusal in refusals):
        return f"the pipeline's movie reader refuses these files: {sorted(refusals)[0]}"
    return None


def lane_path(path: Path) -> str:
    """Relative to the repository root when inside it, as lane paths are; else absolute."""

    absolute = path.absolute()
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def toml_string(text: str) -> str:
    return json.dumps(text)


def toml_pattern(pattern: str) -> str:
    # A literal string keeps the backslashes as they are, unless it holds a quote.
    return toml_string(pattern) if "'" in pattern else f"'{pattern}'"


def span(values: list[int]) -> str:
    return str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}"


def fragment(report: dict[str, Any], rows: list[Series], frames: Path,
             mdocs: Path | None, pattern: str | None) -> str | None:
    accepted = [row for row in rows if row.verdict == "accepted"]
    if not accepted or any(row.verdict == "refused" for row in rows):
        return None
    lane_input = report["input"]
    tilts = sorted({row.tilts for row in accepted})
    low = min(row.angle_min for row in accepted if row.angle_min is not None)
    high = max(row.angle_max for row in accepted if row.angle_max is not None)
    summary = (f"{len(accepted)} series, {span(tilts)} tilts, {_num(round(low, 1))} to "
               f"{_num(round(high, 1))} deg")
    frame_counts = sorted({n for row in accepted for n in row.frames})
    if lane_input == "frames" and frame_counts:
        summary += f", {span(frame_counts)} frames per tilt"
    pixels = sorted({round(row.pixel, 4) for row in accepted if row.pixel is not None})
    if pixels:
        summary += f", {'/'.join(map(_num, pixels))} A/px from the headers"
    lines = [*textwrap.wrap(
        f"Read by scripts/show_folder.py on {date.today().isoformat()}: {summary}.",
        width=86, initial_indent="# ", subsequent_indent="# ",
    ), "[lane]", f'input = "{lane_input}"', "", "[data]"]
    if lane_input == "frames":
        lines.append(f"frames = {toml_string(lane_path(frames))}")
        if pattern is not None:
            lines.append(f"frames_pattern = {toml_pattern(pattern)}")
        else:
            assert mdocs is not None
            lines.append(f"mdocs = {toml_string(lane_path(mdocs))}")
    else:
        assert mdocs is not None
        stack_suffixes = {Path(row.mdoc["stack"]).suffix for row in accepted}
        mdoc_suffixes = {".mrc.mdoc" if row.mdoc["file"].endswith(".mrc.mdoc") else ".mdoc"
                         for row in accepted}
        if len(stack_suffixes) != 1 or len(mdoc_suffixes) != 1:
            report["warnings"].append("the stacks or their mdocs are named in more than one "
                                      "way, so one [data] path cannot name them all")
            return None
        (stack_suffix,), (mdoc_suffix,) = stack_suffixes, mdoc_suffixes
        lines.append(f"stack = {toml_string(lane_path(frames) + '/{series}' + stack_suffix)}")
        lines.append(f"mdoc = {toml_string(lane_path(mdocs) + '/{series}' + mdoc_suffix)}")
    no_axis = [row for row in accepted
               if row.read_as == "frames" and not row.mdoc.get("tilt_axis_from_mdoc")]
    if pattern is not None:
        lines += ["", "[align]",
                  "# EDIT: required with frames_pattern, as nothing in the names gives the",
                  "# tilt axis; run_series.py refuses the lane without it.",
                  "# tilt_axis_angle ="]
    elif no_axis:
        lines += ["", "[align]",
                  "# EDIT: the mdoc parser takes no tilt axis from these mdocs, so it has",
                  "# to be given here.",
                  "# tilt_axis_angle ="]
    lines += ["", "[run]", f"series = [{', '.join(toml_string(row.series) for row in rows)}]"]
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    folder = args.folder.absolute()
    tree = read_tree(folder)
    report: dict[str, Any] = {
        "folder": str(folder), "tree": [asdict(f) for f in tree[:TREE_FOLDERS]],
        "tree_folders": len(tree), "frames": None, "frames_candidates": [],
        "mdocs": None, "mdocs_found": None, "mdocs_candidates": [], "names": None,
        "angles_from": None, "pattern": args.pattern, "refusal": None, "series": [],
        "inside": [], "gains": [], "warnings": [], "input": None, "no_entry": None,
        "fragment": None, "next": None, "exit": 1,
    }
    for item in tree:
        relative = Path(item.path).relative_to(folder)
        report["gains"].extend(str(relative / name) for name in item.gains)
    del report["gains"][10:]

    frames_folder, candidates = pick_frames_folder(tree)
    if frames_folder is None:
        report["frames_candidates"] = [c.path for c in candidates]
        deeper = any(item.depth == TREE_DEPTH and item.folders for item in tree)
        if len(candidates) > 1:
            why = "several folders hold data files, and a lane reads one"
        elif candidates:
            why = "data files are here, but the folders go deeper than read"
        else:
            why = (f"no data files ({' '.join(sorted(DATA_SUFFIXES))}) in the folders read"
                   if deeper else
                   f"no data files ({' '.join(sorted(DATA_SUFFIXES))}) in or under this folder")
        report["next"] = (f"{why}: run this again on the folder that holds the files "
                          "the lane is to read" if candidates or deeper else why)
        return report
    frames = Path(frames_folder.path)
    report["frames"] = str(frames)
    names = data_names(frames)
    report["names"] = {"count": len(names), "sample": sample(names),
                       "fields": field_tables(names)}
    picks = sorted({0, len(names) // 2, len(names) - 1}) if names else []
    report["inside"] = [look_inside(frames / names[i]) for i in picks]

    mdocs: Path | None = None
    if args.pattern is not None:
        report["angles_from"] = "pattern"
        rows = read_by_pattern(frames, args.pattern, args.series, args.max_series, report)
    else:
        if args.mdocs is not None:
            mdocs, where, others = args.mdocs.absolute(), "given", []
        else:
            mdocs, where, others = find_mdocs_folder(frames_folder, tree)
        report["mdocs"], report["mdocs_found"] = (str(mdocs) if mdocs else None), where or None
        report["mdocs_candidates"] = others
        if mdocs is not None:
            report["angles_from"] = "mdocs"
            rows = read_by_mdocs(frames, mdocs, args.series, args.max_series, report)
        else:
            rows = []
            report["next"] = (
                "several folders hold mdocs: run this again with --mdocs <one of them>"
                if others else
                "no mdocs were found and no --pattern was given, so the angles must come "
                "from the names: write a frames_pattern from the fields above "
                "(docs/agent/data_folders.md) and run this again with --pattern"
            )
    report["series"] = [asdict(row) for row in rows]
    report["warnings"] = warnings_for(rows, report["inside"], report)
    report["no_entry"] = no_entry(report["inside"])
    if report["no_entry"]:
        report["next"] = None
    reads = {row.read_as for row in rows if row.read_as != "mdoc"}
    if report["no_entry"] is None and reads:
        if reads <= {"names", "frames"}:
            report["input"] = "frames"
        elif reads == {"stack"}:
            report["input"] = "stack"
        else:
            report["warnings"].append(f"the series are read in more than one way: "
                                      f"{', '.join(sorted(reads))}")
    if report["input"] is not None:
        report["fragment"] = fragment(report, rows, frames, mdocs, args.pattern)
    if report["fragment"] is not None and all(row.verdict == "accepted" for row in rows):
        report["exit"] = 0
    return report


# --- Text -----------------------------------------------------------------------------------


def size_text(size: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if size >= scale:
            return f"{size / scale:.1f} {unit}"
    return f"{size} B"


def counts_text(suffixes: dict[str, int]) -> str:
    return ", ".join(f"{n} {suffix}" for suffix, n in suffixes.items()) or "no files"


def field_text(position: int, entry: dict[str, Any]) -> str:
    if entry["distinct"] == 1:
        return f"  {position:>2}  {entry['examples'][0]:<26} constant"
    if entry["distinct"] <= 3:
        shown = " | ".join(entry["examples"])
    elif "min" in entry:
        shown = f"{entry['min']} .. {entry['max']}"
    else:
        shown = " | ".join(entry["examples"]) + " ..."
    detail = f"{entry['distinct']} values, {entry['kind']}"
    if len(entry.get("widths", [])) == 1:
        detail += f", {entry['widths'][0]} wide"
    return f"  {position:>2}  {shown:<26} {detail}"


def facts_text(facts: dict[str, Any]) -> str:
    def values(numbers: list[float]) -> str:
        return " .. ".join(_num(n) for n in sorted({numbers[0], numbers[-1]}))

    parts = [f"{key} {facts[key]}" for key in ("Voltage", "PixelSpacing") if key in facts]
    parts.append(f"NumSubFrames {'/'.join(map(str, facts['NumSubFrames']))}")
    if "RotationAngle" in facts:
        rotation = f"RotationAngle {'/'.join(map(_num, facts['RotationAngle']))}"
        if not facts["tilt_axis_from_mdoc"]:
            rotation += " (not taken as the tilt axis)"
        parts.append(rotation)
    if "ExposureDose" in facts:
        dose = facts["ExposureDose"]
        parts.append(f"ExposureDose {values(dose['per_tilt'])} per tilt, "
                     f"{_num(dose['total'])} in all")
    if "TargetDefocus" in facts:
        parts.append(f"TargetDefocus {values(facts['TargetDefocus'])}")
    if "written_by" in facts:
        parts.append(f"[T = {facts['written_by']}]")
    return "; ".join(parts)


def row_text(row: dict[str, Any]) -> str:
    angles = (f"{_num(round(row['angle_min'], 1))} .. {_num(round(row['angle_max'], 1))}"
              if row["angle_min"] is not None else "-")
    steps = "/".join(row["steps"]) or "-"
    frames = "/".join(map(str, row["frames"])) or ("stack" if row["read_as"] == "stack" else "-")
    pixel = _num(round(row["pixel"], 4)) if row["pixel"] is not None else "-"
    return (f"  {row['series']:<14} {row['read_as']:<7} {row['tilts']:>5}  {angles:<16} "
            f"{steps:<8} {frames:<7} {pixel:<7} {row['verdict']}")


def render(report: dict[str, Any]) -> str:
    out: list[str] = []
    root, *below = report["tree"]
    out.append(f"Folder {report['folder']}: {counts_text(root['suffixes'])}"
               + (f", {size_text(root['size_bytes'])}" if root["size_bytes"] else "")
               + (f", {root['folders']} subfolders" if root["folders"] else ""))
    base = Path(report["folder"])
    for item in below:
        name = Path(item["path"]).relative_to(base).as_posix() + "/"
        line = f"  {name:<28} {counts_text(item['suffixes'])}"
        if item["size_bytes"]:
            line += f", {size_text(item['size_bytes'])}"
        if item["folders"] and item["depth"] == TREE_DEPTH:
            line += f", {item['folders']} subfolders not listed"
        out.append(line)
    if report["tree_folders"] > TREE_FOLDERS:
        out.append(f"  ... {report['tree_folders'] - TREE_FOLDERS} more folders")
    links = [f"{Path(item['path']).name}/{link}" for item in report["tree"]
             for link in item["links_to_folders"]]
    if links:
        out.append(f"  links to folders, not followed: {', '.join(links[:5])}")
    if report["gains"]:
        out.append(f"Maybe a gain reference, by the name only: {', '.join(report['gains'])}")
    if report["frames"] is None:
        out.append("")
        out.append(f"Next: {report['next']}")
        out.extend(f"  {path}" for path in report["frames_candidates"])
        return "\n".join(out)

    names = report["names"]
    out += ["", f"Data files: {names['count']} in {report['frames']}",
            f"Names: {', '.join(names['sample'])}"]
    for table in names["fields"]:
        out.append(f"Fields (split at _ [ ]), {table['names']} names with {table['fields']}:")
        out.extend(field_text(i + 1, entry) for i, entry in enumerate(table["table"]))
    out.append("Inside:")
    for look in report["inside"]:
        if "images" not in look:
            why = look.get("error") or look.get("movie_refusal") or "not read"
            size = f", {size_text(look['size_bytes'])}" if "size_bytes" in look else ""
            out.append(f"  {look['file']}: {look['format']}{size}; not read: {why}")
            continue
        images = f"{look['images']} image{'s' if look['images'] != 1 else ''}"
        dtype = f" {look['dtype']}" if "dtype" in look else ""
        pixel = f"{_num(look['pixel'])} A/px" if look.get("pixel") else "no pixel size"
        truncated = "" if look.get("complete", True) else ", SHORTER than its header says"
        out.append(f"  {look['file']}: {look['format']}, {images} of {look['x']} x "
                   f"{look['y']}{dtype}, {pixel}, {size_text(look['size_bytes'])}{truncated}")

    out.append("")
    if report["angles_from"] == "pattern":
        out.append(f"Angles from the names: frames_pattern {toml_pattern(report['pattern'])}")
        skipped = {s: n for s, n in report.get("unmatched_counts", {}).items()}
        if skipped and report["refusal"] is None:
            out.append(f"  not frames, skipped as init skips them: "
                       f"{', '.join(f'{n} {s}' for s, n in skipped.items())}")
    elif report["angles_from"] == "mdocs":
        names_seen = report.get("mdoc_names", {})
        out.append(f"Angles from the mdocs in {report['mdocs']} ({report['mdocs_found']}): "
                   f"{counts_text(names_seen.get('suffixes', {}))}")
        if report.get("init_mdocs"):
            out.append(f"  init refuses the folder of mdocs: {report['init_mdocs']}")
    if report["refusal"]:
        out.append(f"REFUSED: {report['refusal']}")
        for suffix, missed in report.get("unmatched", {}).items():
            count = report["unmatched_counts"][suffix]
            out.append(f"  not matched, {count} {suffix}: {', '.join(missed[:5])}"
                       + (" ..." if count > 5 else ""))
    if report["series"]:
        out.append(f"  {'series':<14} {'read as':<7} {'tilts':>5}  {'angles':<16} "
                   f"{'step':<8} {'frames':<7} {'A/px':<7} verdict")
        out.extend(row_text(row) for row in report["series"])
        mdoc_rows = [row for row in report["series"] if row["mdoc"].get("NumSubFrames")]
        for row in mdoc_rows[:3]:
            out.append(f"  {row['mdoc']['file']}: {facts_text(row['mdoc'])}")
        refused = [row for row in report["series"]
                   if row["verdict"] == "refused" and "whole folder" not in (row["refusal"] or "")]
        for row in refused:
            out.append(f"  REFUSED {row['series']}: {row['refusal']}")
    if report["warnings"]:
        out.append("Warnings:")
        out.extend(f"  - {warning}" for warning in report["warnings"])
    if report["no_entry"]:
        out.append(f"No entry: {report['no_entry']}.")
    if report["next"]:
        out.append(f"Next: {report['next']}")
        out.extend(f"  {path}" for path in report["mdocs_candidates"])
    if report["fragment"]:
        what = ("movies, one file per tilt" if report["input"] == "frames"
                else "one assembled stack per series, with its mdoc")
        out += ["", f"Lane input: {report['input']} ({what}). Config fragment, once the "
                "person has confirmed the series and tilts above:", ""]
        out.extend(f"  {line}" if line else "" for line in report["fragment"].splitlines())
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("folder", type=Path, help="the data folder, or the folder of movies")
    parser.add_argument("--pattern", help="a [data] frames_pattern to read the names with")
    parser.add_argument("--mdocs", type=Path, help="the mdoc folder, if not found by itself")
    parser.add_argument("--series", action="append", default=[],
                        help="read only this series; repeat for several")
    parser.add_argument("--max-series", type=int, default=DEFAULT_MAX_SERIES,
                        help=f"open the files of at most this many series "
                             f"(default {DEFAULT_MAX_SERIES})")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    if args.pattern is not None and args.mdocs is not None:
        print("give --pattern or --mdocs, not both: init takes the angles from one of them",
              file=sys.stderr)
        return 2
    if args.mdocs is not None and not args.mdocs.is_dir():
        print(f"not a folder: {args.mdocs}", file=sys.stderr)
        return 2
    if len(set(args.series)) != len(args.series):
        print("a series is given twice", file=sys.stderr)
        return 2
    report = build(args)
    print(json.dumps(report, indent=2, default=str) if args.json else render(report))
    return int(report["exit"])


if __name__ == "__main__":
    sys.exit(main())
