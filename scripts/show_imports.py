#!/usr/bin/env python3
"""Say what files made elsewhere would be entered as, read the way the imports read them.

    python3 scripts/show_imports.py tomograms <folder or volumes>...
    python3 scripts/show_imports.py picks <file> [--tomogram <the volume picked in>]
    python3 scripts/show_imports.py particles <Dynamo data folder> [--table <.tbl>]

Read-only: it reads headers, tables and the files beside them, measures the wedge
in a central crop of each volume and in a sample of particles, and writes nothing.

It answers the questions of docs/entry_and_exit.md ("Same questions, listed
answers") from the files, as show_folder.py does for frames. For each question it
gives the answer the files give and where that comes from, or leaves the question
for the person with the answers on the list. The lists, the readers and the
measurements are the imports' own (cryoet_pipeline.tomogram_import, picks_import,
particles_import, wedge_check), so what this script cannot read is what an import
would refuse to guess. It ends with the [lane] input and the [data] lines of a lane
config (scripts/run_series.py), each question left as a commented line to answer.

Where the answers come from:
- tomograms. The orientation from the header's history and the thinnest axis.
  IMOD's tilt writes "TILT: Tomographic reconstruction", and clip (trimvol calls
  it) adds "clip: rotx" or "clip: flipyz"; RELION 5 writes "Relion" (measured on
  IMOD 5.1.12 and RELION 5.0, 17 November 2026). tilt -PARALLEL and -RotateBy90
  write the same label, so the wedge measured in the volume decides between
  imod-parallel and imod-standard, when the range is asymmetric. The wedge also
  checks every other answer. The angles from tilt.com's TILTFILE, a .tlt named
  after the series, or RELION's tomograms.star; the pixel size from the header.
  What was done to the volume is asked; a wedge too faint to see is said.
- picks. The format from the file's columns; pytom's template box from the job
  file it wrote beside the picks; a Dynamo table's tomogram numbers; and the extent
  of the coordinates against the volume picked in.
- particles. The files and their box; from the table, the tomograms, the pixel
  size and each tomogram's wedge, which the particles' own power is set against.

Exit status: 0 when a fragment is printed, with questions left in it or not; 1 when
something an import refuses was found, or nothing could be read; 2 for a path that
does not exist or arguments that contradict each other.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import textwrap
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

from cryoet_pipeline.particles import pytom_template_shape, read_star_blocks
from cryoet_pipeline.particles_import import (
    APIX,
    FTYPE,
    TAG,
    TOMO,
    WEDGE_SAMPLE,
    YMAX,
    YMIN,
    _particle_files,
    _read_particle,
    _table,
)
from cryoet_pipeline.picks_import import (
    CSV_NAMES,
    FORMATS,
    UNMEASURED_FORMATS,
    read_csv,
    read_dynamo_table,
    read_imod_model,
    read_pytom_star,
)
from cryoet_pipeline.tomogram_import import (
    ORIENTATIONS,
    PROCESSING_STEPS,
    ImportRefused,
    canonical_shape,
    inspect_volume,
    read_tilt_angles,
    wedge_in_volume,
)
from cryoet_pipeline.wedge_check import measure_particles_wedge

ROOT = Path(__file__).resolve().parents[1]
VOLUME_SUFFIXES = (".mrc", ".rec")
DEFAULT_MAX_VOLUMES = 20
# The labels the programs on the list write, measured on IMOD 5.1.12 and RELION 5.0.
TILT_LABEL = "TILT: Tomographic reconstruction"
RELION_LABEL = "Relion"
CLIP_TURNS = {"rotx": "imod-standard", "flipyz": "imod-parallel"}
# How each answer of the list turns the voxels; the two of a pair hold the same ones.
PAIRS = {"none": ("imod-parallel", "pipeline"), "reverse_z": ("imod-standard", "relion5")}
_NAME_AFFIXES = re.compile(r"^(?:rec_)?(?P<series>.+?)(?:_full_rec|_rec)?$")


@dataclass
class Answer:
    key: str
    value: Any = None
    source: str = ""
    question: str | None = None
    """Set when the files do not answer it: what to ask, with the listed answers."""

    @property
    def asked(self) -> bool:
        return self.question is not None


@dataclass
class Reading:
    kind: str
    path: str
    items: list[dict[str, Any]] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    series: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def lane_path(path: Path | str) -> str:
    """Relative to the repository root when inside it, as lane paths are; else absolute."""

    absolute = Path(path).absolute()
    try:
        return absolute.relative_to(ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def _num(value: float) -> str:
    return f"{value:g}"


def _range(values: Any) -> str:
    array = np.asarray(values, dtype=float)
    return f"{_num(round(float(array.min()), 2))}..{_num(round(float(array.max()), 2))}"


# --- Tomograms ---------------------------------------------------------------------


def labels_of(path: Path) -> list[str]:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        header = mrc.header
        # The programs pad a label with spaces to its date; one space is kept.
        return [re.sub(r"\s{2,}", "  ", bytes(label).decode(errors="replace").strip())
                for label in header.label[: int(header.nlabl)]]


def series_of(path: Path) -> str:
    """The series a volume's name gives: the folder it is in when the name begins with
    it, as in per-series folders; else the name with RELION's rec_ prefix and IMOD's
    _rec or _full_rec suffix dropped."""

    folder = path.parent.name
    if folder and path.stem.startswith((f"{folder}_", f"{folder}.")):
        return folder
    match = _NAME_AFFIXES.match(path.stem)
    return match["series"] if match else path.stem


def template_of(path: Path, series: str) -> str:
    """The volume's lane path with its series written as {series}, in its name and in
    a folder named after it, as etomo's per-series folders are."""

    shown = lane_path(path)
    parts = shown.split("/")
    parts[-1] = parts[-1].replace(series, "{series}", 1)
    if len(parts) > 1 and parts[-2] == series:
        parts[-2] = "{series}"
    return "/".join(parts)


def history(labels: list[str], thin: str) -> tuple[str | None, str]:
    """The orientation the header's history gives, or None, and why.

    `thin` is the axis the volume is thinnest along as stored: "z" (sections) or
    "y" (rows). A name ending in "?" is a pair the history cannot split.
    """

    tilts = [index for index, label in enumerate(labels) if label.startswith(TILT_LABEL)]
    relion = [index for index, label in enumerate(labels) if label.startswith(RELION_LABEL)]
    if not tilts:
        if relion:
            return "relion5", "RELION's label, which RELION 5 writes on its reconstructions"
        return None, "the header names no program of the list"
    after = labels[tilts[-1] + 1:]
    if any(label.startswith(RELION_LABEL) for label in after):
        return None, "IMOD's tilt, then a RELION program the header does not say more of"
    turns = [label[len("clip:"):].strip().split()[0] for label in after
             if label.startswith("clip:")]
    if not turns:
        if thin == "y":
            return "imod-tilt", "IMOD's tilt, its output as written (Y as sections)"
        return "imod-parallel|imod-standard?", (
            "IMOD's tilt with z as sections: -PARALLEL or -RotateBy90, which write the same "
            "label")
    if len(turns) == 1 and turns[0] in CLIP_TURNS and thin == "z":
        return CLIP_TURNS[turns[0]], f"IMOD's tilt, then clip {turns[0]}"
    return None, f"IMOD's tilt, then clip {', '.join(turns)}, which the list does not hold"


def angles_for(volume: Path, series: str) -> tuple[Path | None, str]:
    """The angle file the volume was reconstructed with, and how it was found."""

    folder = volume.parent
    if volume.name.startswith("rec_") and folder.name == "tomograms":
        table = folder.parent / "tomograms.star"
        if table.is_file():
            for columns, rows in read_star_blocks(table).values():
                if {"rlnTomoName", "rlnTomoTiltSeriesStarFile"} <= set(columns):
                    for row in rows:
                        if row[columns.index("rlnTomoName")] != series:
                            continue
                        named = Path(row[columns.index("rlnTomoTiltSeriesStarFile")])
                        for base in (table.parent, *table.parents[:4]):
                            if (base / named).is_file():
                                return base / named, f"RELION's {lane_path(table)}"
            return None, f"{lane_path(table)} names no tilt series of {series} that is there"
        return None, f"no RELION tomograms.star in {lane_path(folder.parent)}"
    command = folder / "tilt.com"
    if command.is_file():
        for line in command.read_text(errors="replace").splitlines():
            words = line.split()
            if len(words) == 2 and words[0] == "TILTFILE":
                named = folder / words[1]
                if named.is_file():
                    return named, "the TILTFILE of tilt.com beside it"
    for own, said in ((volume.with_suffix(".tlt"), "the .tlt of the volume's name beside it"),
                      (folder / f"{series}.tlt", "the .tlt of the series' name beside it")):
        if own.is_file():
            return own, said
    others = sorted(folder.glob("*.tlt"))
    if len(others) == 1:
        return others[0], "the one .tlt beside it, not named after the series: check it"
    return None, (f"{len(others)} .tlt files beside it" if others else "no .tlt beside it")


def read_volume(path: Path, *, wedge: bool) -> dict[str, Any]:
    """What one volume says: header, history, angles, and the wedge in it."""

    item: dict[str, Any] = {"file": lane_path(path), "series": series_of(path)}
    try:
        header = inspect_volume(path)
    except (ImportRefused, ValueError, OSError) as error:
        item["refusal"] = str(error)
        return item
    sections, rows, columns = header.shape
    item.update(shape_xyz=[columns, rows, sections], dtype=header.dtype,
                voxel=header.voxel_size[0], axes=list(header.axis_map))
    if header.axis_map != (1, 2, 3):
        item["refusal"] = (f"the header's axis order is {header.axis_map}; every orientation "
                           "on the list has columns X, rows Y, sections Z")
        return item
    thin = "z" if sections < min(rows, columns) else "y" if rows < min(sections, columns) else ""
    item["thin"] = thin or "none"
    item["labels"] = labels_of(path)
    name, why = history(item["labels"], thin)
    item["history"] = {"orientation": name, "why": why}
    angle_file, found = angles_for(path, item["series"])
    item["angles_found"] = found
    if angle_file is not None:
        item["angles"] = lane_path(angle_file)
        try:
            values = read_tilt_angles(angle_file).angles
        except (ImportRefused, OSError) as error:
            item["angles_refusal"] = str(error)
            values = None
        if values is not None:
            item["angle_count"], item["angle_range"] = len(values), [min(values), max(values)]
            if wedge and thin:
                item["wedge"] = wedge_reading(path, thin, name, values)
    item["orientation"], item["orientation_from"] = decide(item)
    return item


def wedge_reading(path: Path, thin: str, name: str | None,
                  angles: list[float]) -> dict[str, Any]:
    """The wedge measured as the history's answer would turn the volume, or, with no
    single answer, as the voxels of a pair are kept."""

    if name in ORIENTATIONS:
        transform = ORIENTATIONS[name].transform
    else:
        transform = "transpose_yz" if thin == "y" else "none"
    shape = canonical_shape(_stored_shape(path), transform)
    if not shape[0] < min(shape[1], shape[2]):
        return {"transform": transform, "verdict": "not measured",
                "reason": "z does not come out the thinnest axis this way"}
    measured = wedge_in_volume(path, transform, angles)
    return {"transform": transform, "verdict": measured.verdict, "reason": measured.reason,
            "fitted": [measured.fitted_min_deg, measured.fitted_max_deg]}


def _stored_shape(path: Path) -> tuple[int, int, int]:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        header = mrc.header
        return int(header.nz), int(header.ny), int(header.nx)


def decide(item: dict[str, Any]) -> tuple[str | None, str]:
    """The orientation the history and the wedge give together, or None and why."""

    name, why = item["history"]["orientation"], item["history"]["why"]
    measured = item.get("wedge")
    verdict = measured["verdict"] if measured else None
    if name is not None and name.endswith("?"):
        if verdict == "agrees":
            return "imod-parallel", f"{why}; the wedge has the side of the voxels as kept"
        if verdict == "mirrored":
            return "imod-standard", f"{why}; the wedge has the side of the voxels reversed in z"
        return None, f"{why}; the wedge cannot split them ({verdict or 'not measured'})"
    if name is not None:
        if verdict == "mirrored":
            return None, (f"the history says {name}, and the wedge has the other side: the "
                          "volume was mirrored after, or the angles are not its own")
        checked = {"agrees": "the wedge agrees",
                   "side not testable": "the range is symmetric, so the wedge cannot test the side",
                   None: "the wedge was not measured"}.get(verdict, f"the wedge: {verdict}")
        return name, f"{why}; {checked}"
    if verdict in ("agrees", "mirrored") and item.get("thin") == "z":
        transform = "none" if verdict == "agrees" else "reverse_z"
        pair = " or ".join(PAIRS[transform])
        return None, (f"{why}; the wedge has the side of {pair}, which hold the same voxels: "
                      "which program wrote it is asked")
    if verdict == "agrees" and item.get("thin") == "y":
        return "imod-tilt", f"{why}; Y as sections, and the wedge agrees as tilt's output"
    return None, why


def read_tomograms(paths: list[Path], *, wedge: bool, max_volumes: int) -> Reading:
    volumes: list[Path] = []
    for path in paths:
        if path.is_dir():
            volumes += sorted(p for p in path.iterdir()
                              if p.suffix.lower() in VOLUME_SUFFIXES and not p.name.startswith("."))
        else:
            volumes.append(path)
    reading = Reading("tomograms", ", ".join(lane_path(path) for path in paths))
    if not volumes:
        reading.refusals.append("no .mrc or .rec volume here")
        return reading
    if len(volumes) > max_volumes:
        reading.notes.append(f"{len(volumes)} volumes; the first {max_volumes} are read "
                             "(--max-volumes)")
    for path in volumes[:max_volumes]:
        reading.items.append(read_volume(path, wedge=wedge))
    read = [item for item in reading.items if "refusal" not in item]
    reading.refusals += [f"{item['file']}: {item['refusal']}" for item in reading.items
                         if "refusal" in item]
    reading.series = [item["series"] for item in read]
    if len(set(reading.series)) != len(reading.series):
        reading.refusals.append("two volumes give one series by their names: "
                                + ", ".join(sorted(s for s, n in Counter(reading.series).items()
                                                   if n > 1)))
    if not read:
        return reading
    reading.answers += [
        _shared(read, "tomogram", lambda item: template_of(ROOT / item["file"], item["series"]),
                "the volumes' names, the series as {series}"),
        _shared(read, "tilt_angles",
                lambda item: (template_of(ROOT / item["angles"], item["series"])
                              if "angles" in item else None),
                "each volume's angle file, the series as {series}",
                question="the angle file each volume was reconstructed with (IMOD's .tlt, "
                         "or RELION 5's tilt-series STAR)"),
        _shared(read, "orientation", lambda item: item["orientation"],
                "the history and the wedge",
                question="the orientation, from the list: " + ", ".join(ORIENTATIONS)),
        Answer("processing", question="what was done to the volumes after reconstruction: "
               f"raw, or a set of {', '.join(PROCESSING_STEPS)}"),
    ]
    voxels = sorted({round(item["voxel"], 4) for item in read})
    if voxels == [0.0]:
        reading.answers.append(Answer("tomogram_pixel_size_angstrom",
                                      question="the voxel size in A: the headers have none"))
    elif len(voxels) > 1:
        reading.notes.append(f"the voxel sizes differ: {', '.join(map(_num, voxels))} A")
    named = {item["orientation"] for item in read}
    if None not in named and len(named) > 1:
        reading.notes.append("the volumes are in different orientations ("
                             + ", ".join(sorted(named)) + "): a lane takes one, so each "
                             "orientation goes in a lane of its own")
    faint = [item["file"] for item in read if item.get("wedge", {}).get("verdict") == "no wedge"]
    if faint:
        reading.notes.append("no wedge seen in " + ", ".join(faint) + ": filled (wedge-filled), "
                             "or too small or smooth to tell")
    return reading


def _shared(items: list[dict[str, Any]], key: str, of: Any, source: str,
            question: str | None = None) -> Answer:
    """One answer for every volume, or the question when they differ or give none."""

    values = [of(item) for item in items]
    if None in values or len(set(values)) != 1:
        given = sorted({str(value) for value in values if value is not None})
        detail = f" (the files give {', '.join(given)})" if given else ""
        return Answer(key, question=(question or f"[data] {key}") + detail)
    return Answer(key, values[0], source)


# --- Picks ---------------------------------------------------------------------------


def star_format(columns: list[str], rows: list[list[str]]) -> tuple[str | None, str]:
    """The format a STAR table's columns say, or None, and why."""

    names = set(columns)
    if {"rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst"} <= names:
        return "relion5-star", "rlnCenteredCoordinate*Angst, RELION 5's centred coordinates"
    if "rlnMicrographName" in names and any(
            row[columns.index("rlnMicrographName")].endswith(".tomostar") for row in rows):
        return "warp-star", "rlnMicrographName naming .tomostar files, as Warp writes them"
    if names & {"rlnImageName", "rlnCtfImage", "rlnOriginX", "rlnOriginXAngst"}:
        return "relion3-star", "subtomogram columns (rlnImageName, rlnOrigin*) of RELION 3/4"
    if {"rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"} <= names:
        if {"rlnLCCmax", "rlnCutOff"} <= names:
            return "pytom-star", "rlnLCCmax and rlnCutOff, the columns pytom writes"
        return None, ("voxel coordinates without the columns of a program on the list: "
                      "pytom-star reads them as voxels of the volume picked")
    return None, "no coordinate columns this script knows"


def picks_format(path: Path) -> tuple[str | None, str, dict[str, Any]]:
    """What the file is, why, and what reading it gave on the way."""

    found: dict[str, Any] = {}
    head = path.read_bytes()[:8]
    if head.startswith(b"IMOD"):
        return "imod-model", "the IMOD model's own header", found
    suffix = path.suffix.lower()
    if suffix == ".star":
        blocks = read_star_blocks(path)
        for columns, rows in blocks.values():
            name, why = star_format(columns, rows)
            if name is not None or "rlnCoordinateX" in columns:
                found.update(columns=columns, rows=len(rows))
                return name, why, found
        return None, "no table of coordinates in the STAR file", found
    if suffix == ".tbl":
        return "dynamo-table", "a .tbl, Dynamo's table", found
    if suffix in (".csv", ".txt", ".tsv"):
        return "csv", "a table of named or numbered columns", found
    return None, f"a {suffix or 'nameless'} file this script does not know", found


def _csv_columns(path: Path) -> list[str] | None:
    """The CSV's column names, when its first line names them all from CSV_NAMES."""

    first = next((line for line in path.read_text().splitlines() if line.strip()), "")
    words = [word.strip().lower() for word in re.split(r"[,\t;]|\s+", first) if word.strip()]
    aliases = {"x": "x", "y": "y", "z": "z", "rot": "rot", "tilt": "tilt", "psi": "psi",
               "score": "score", "coordinatex": "x", "coordinatey": "y", "coordinatez": "z"}
    named = [aliases.get(word.replace("_", "").replace("rln", ""), "skip") for word in words]
    if {"x", "y", "z"} <= set(named) and all(name in CSV_NAMES for name in named):
        return named
    return None


def read_picks(path: Path, tomogram: Path | None) -> Reading:
    reading = Reading("picks", lane_path(path))
    name, why, found = picks_format(path)
    item: dict[str, Any] = {"file": lane_path(path), "format": name, "why": why, **found}
    reading.items.append(item)
    if name in UNMEASURED_FORMATS:
        reading.refusals.append(f"{name} is not on the list yet: it needs "
                                f"{UNMEASURED_FORMATS[name]}")
        return reading
    series = series_of(tomogram) if tomogram is not None else None
    if tomogram is not None and series is not None and series in path.name:
        picks = Answer("picks", template_of(path, series),
                       f"the picks' name, with the series of {lane_path(tomogram)} as {{series}}")
    else:
        picks = Answer("picks", lane_path(path), "given: one file for every series reads the "
                       "series from its rows (relion5-star, dynamo-table); for a file per "
                       "series, write {series} where its name holds the series")
    reading.answers += [
        picks,
        Answer("picks_format", name, why) if name in FORMATS else
        Answer("picks_format", question=f"the format, from the list: {', '.join(FORMATS)} ({why})"),
    ]
    positions = None
    if name == "pytom-star":
        jobs = sorted(path.parent.glob("*_job.json"))
        stem = path.name.split("_particles")[0]
        job = next((one for one in jobs if one.name.startswith(stem)), jobs[0] if len(jobs) == 1
                   else None)
        shape = pytom_template_shape(job) if job is not None else None
        if job is not None and shape is not None:
            reading.answers.append(Answer("template_box_px", int(shape[0]),
                                          f"template_shape in {lane_path(job)}"))
        else:
            reading.answers.append(Answer(
                "template_box_px", question="the template box pytom matched with (no pytom job "
                "file beside the picks gives it)"))
        try:
            source = read_pytom_star(path, int(shape[0]) if shape else 1)
        except ImportRefused as error:
            reading.refusals.append(str(error))
        else:
            positions, item["angles"] = source.positions_xyz, source.angles_deg is not None
        columns = found.get("columns", [])
        if "rlnDetectorPixelSize" in columns:
            _, rows = next(block for block in read_star_blocks(path).values()
                           if "rlnDetectorPixelSize" in block[0])
            index = columns.index("rlnDetectorPixelSize")
            item["pixel"] = sorted({float(row[index]) for row in rows})
    elif name == "dynamo-table":
        table = _table(path)
        numbers = Counter(int(round(value)) for value in table[:, TOMO])
        item["tomograms"] = dict(sorted(numbers.items()))
        if len(numbers) > 1:
            counts = ", ".join(f"{number} ({count} rows)"
                               for number, count in sorted(numbers.items()))
            reading.answers.append(Answer(
                "dynamo_tomograms",
                question=f"the series of each tomogram number (column 20): {counts}"))
        poses = table[:, [7, 8, 9]]
        item["angles"] = bool(np.any(np.abs(poses) > 1e-9))
        if not item["angles"]:
            reading.notes.append("every pose in the table is zero: these picks have no "
                                 "orientations, and the import keeps them as given")
        source = read_dynamo_table(path, sorted(numbers)[0] if len(numbers) > 1 else None)
        positions = source.positions_xyz
        if len(numbers) > 1:
            item["positions_of"] = f"tomogram {sorted(numbers)[0]}"
    elif name == "csv":
        columns = _csv_columns(path)
        if columns is not None:
            reading.answers.append(Answer("csv_columns", columns, "the CSV's header line"))
            positions = read_csv(path, columns).positions_xyz
            item["angles"] = "rot" in columns
        else:
            reading.answers.append(Answer(
                "csv_columns",
                question=f"each column's name, in order, from {', '.join(CSV_NAMES)}"))
        reading.answers += [
            Answer("coordinate_units", question="voxels or angstrom"),
            Answer("coordinate_origin", question="where zero is: voxel-0, voxel-1, corner, centre"),
        ]
    elif name == "imod-model":
        with tempfile.TemporaryDirectory(prefix=".show-imports-") as folder:
            try:
                positions = read_imod_model(path, None, Path(folder)).positions_xyz
            except ImportRefused as error:
                item["model"] = str(error)
                reading.answers.append(Answer("imod_object", question=f"which object holds the "
                                              f"picks ({error})"))
        item["angles"] = False
    elif name == "relion5-star":
        blocks = read_star_blocks(path)
        for columns, rows in blocks.values():
            if "rlnTomoName" in columns:
                names = Counter(row[columns.index("rlnTomoName")] for row in rows)
                item["tomo_names"] = dict(sorted(names.items()))
                reading.series = sorted(names)
            if "rlnTomoTiltSeriesPixelSize" in columns:
                item["tilt_series_pixel"] = sorted({float(row[columns.index(
                    "rlnTomoTiltSeriesPixelSize")]) for row in rows})
        beside = [folder / "tomograms.star" for folder in (path.parent, path.parent.parent)]
        tables = [table for table in beside if table.is_file() and any(
            "rlnTomoTiltSeriesPixelSize" in columns
            for columns, _ in read_star_blocks(table).values())]
        if tables:
            reading.answers.append(Answer(
                "relion_tomograms_star", lane_path(tables[0]),
                "RELION's tomograms.star near the picks: the tilt series' pixel, and the "
                "tomograms' size checked against the one imported"))
        elif "tilt_series_pixel" not in item:
            reading.answers.append(Answer(
                "relion_tilt_series_pixel_size_angstrom", question="the tilt series' pixel size "
                "in A (no optics table here gives it), or [data] relion_tomograms_star"))
    if name in FORMATS and name != "relion5-star":
        reading.notes.append("[data] picked_in is left out: the picks are read as taken in the "
                             "volume as it is imported. Set it when they were picked in another "
                             "orientation of it (" + ", ".join(ORIENTATIONS) + ")")
    if positions is not None and len(positions):
        item["count"] = len(positions)
        item["extent_xyz"] = [[float(positions[:, axis].min()), float(positions[:, axis].max())]
                              for axis in range(3)]
        if tomogram is not None:
            reading.notes += extent_notes(positions, tomogram, item.get("pixel"))
    return reading


def extent_notes(positions: np.ndarray, tomogram: Path, pixels: list[float] | None) -> list[str]:
    """Where the picks lie against the volume they were taken in, and what a wrong
    scale would look like."""

    header = inspect_volume(tomogram)
    sections, rows, columns = header.shape
    size = np.array([columns, rows, sections], dtype=float)
    top = positions.max(axis=0)
    notes = []
    if np.all(positions.min(axis=0) >= -0.5) and np.all(top <= size - 0.5):
        notes.append(f"the picks lie inside {lane_path(tomogram)}, {columns} x {rows} x "
                     f"{sections} (x, y, z)")
    else:
        ratio = float(np.max(top / size))
        notes.append(f"the picks reach {ratio:.2f} times {lane_path(tomogram)}'s size along "
                     "an axis: another binning or orientation than this volume's")
    voxel = header.voxel_size[0]
    if pixels and voxel and all(abs(pixel - voxel) > 0.005 * voxel for pixel in pixels):
        notes.append(f"the picks' pixel ({', '.join(map(_num, pixels))} A) is not the volume's "
                     f"({_num(voxel)} A)")
    return notes


# --- Particles -----------------------------------------------------------------------


def read_particles(folder: Path, table_path: Path | None) -> Reading:
    reading = Reading("particles", lane_path(folder))
    try:
        files = _particle_files(folder)
    except (ImportRefused, FileNotFoundError) as error:
        reading.refusals.append(str(error))
        return reading
    if table_path is None:
        tables = sorted(folder.glob("*.tbl")) + sorted(folder.parent.glob("*.tbl"))
        if len(tables) != 1:
            reading.answers.append(Answer("particles_table", question="the table of these "
                                          f"particles ({len(tables)} .tbl files near)"))
            return reading
        table_path = tables[0]
        reading.answers.append(Answer("particles_table", lane_path(table_path),
                                      "the one .tbl in or beside the folder"))
    else:
        reading.answers.append(Answer("particles_table", lane_path(table_path), "given"))
    reading.answers.insert(0, Answer("particles_folder", lane_path(folder), "given"))
    first = _read_particle(files[min(files)])
    box = int(first.shape[0])
    item: dict[str, Any] = {"files": len(files), "box": list(first.shape)}
    reading.items.append(item)
    if len(set(first.shape)) != 1 or box % 2:
        reading.refusals.append(f"the particles are {first.shape}; the MRA takes even cubes")
    try:
        table = _table(table_path)
    except (ImportRefused, FileNotFoundError) as error:
        reading.refusals.append(str(error))
        return reading
    tags = [int(round(value)) for value in table[:, TAG]]
    item["rows"] = len(tags)
    missing = [tag for tag in tags if tag not in files]
    if missing:
        reading.refusals.append(f"{len(missing)} rows have no particle file, tags {missing[:5]}")
    unused = len(set(files) - set(tags))
    if unused:
        reading.notes.append(f"{unused} particle files have no row; the import leaves them out")
    types = sorted({int(round(value)) for value in table[:, FTYPE] if np.isfinite(value)})
    if types != [1]:
        reading.refusals.append(f"wedge types {types} (column 13); the import takes 1, a "
                                "single tilt axis")
    pixels = sorted({float(value) for value in table[:, APIX] if np.isfinite(value)})
    item["pixel"] = pixels
    if not pixels or pixels == [0.0]:
        reading.answers.append(Answer("particles_pixel_size_angstrom", question="the pixel size "
                                      "in A: column 36 (apix) has none"))
    elif len(pixels) > 1:
        reading.refusals.append(f"column 36 holds several pixel sizes: {pixels}")
    tomograms = {}
    for number in sorted({int(round(value)) for value in table[:, TOMO]}):
        rows = table[np.rint(table[:, TOMO]) == number]
        ytilt = sorted({(float(low), float(high)) for low, high in rows[:, [YMIN, YMAX]]})
        entry: dict[str, Any] = {"particles": len(rows), "dynamo_ytilt": ytilt}
        if len(ytilt) == 1:
            low, high = -ytilt[0][1], -ytilt[0][0]
            entry["tilt_range"] = [low, high]
            present = [int(round(tag)) for tag in rows[:, TAG] if int(round(tag)) in files]
            step = max(1, len(present) // WEDGE_SAMPLE)
            if present and len(set(first.shape)) == 1:
                measured = measure_particles_wedge(
                    (_read_particle(files[tag]) for tag in present[::step]), [low, high])
                entry["wedge"] = {"verdict": measured.verdict, "reason": measured.reason}
                if measured.verdict == "mirrored":
                    reading.refusals.append(f"tomogram {number}: the particles' wedge has the "
                                            "other side of the table's")
        else:
            reading.refusals.append(f"tomogram {number} has several wedges in its rows")
        tomograms[number] = entry
    item["tomograms"] = tomograms
    reading.answers.append(Answer(
        "particles_series", question="the series of each tomogram number (column 20): "
        + ", ".join(f"{number} ({entry['particles']} particles)"
                    for number, entry in tomograms.items())))
    return reading


# --- Output --------------------------------------------------------------------------


def fragment(reading: Reading) -> str:
    lines = [f"# Read by scripts/show_imports.py on {date.today().isoformat()}. Each "
             "commented line is", "# a question for the person.", "[lane]",
             f'input = "{reading.kind}"']
    if reading.series:
        lines += ["", "[run]", "series = " + json.dumps(reading.series)]
    lines += ["", "[data]"]
    for answer in reading.answers:
        if answer.asked:
            lines += textwrap.wrap(f"{answer.key} = ?  {answer.question}", width=86,
                                   initial_indent="# ", subsequent_indent="#   ",
                                   break_long_words=False, break_on_hyphens=False)
        else:
            lines.append(f"{answer.key} = {json.dumps(answer.value)}")
    if reading.kind == "picks":
        lines.append("# the tomogram keys too: run show_imports.py tomograms on the volume")
    if reading.kind in ("picks", "particles"):
        lines += ["", "[pick]", "# particle_diameter_angstrom = ?  the particle where it is "
                  "widest, for", "#   the import's check and the MRA"]
    return "\n".join(lines)


def render(reading: Reading) -> str:
    out = [f"{reading.kind}: {reading.path}"]
    for item in reading.items:
        out += item_lines(reading.kind, item)
    if reading.answers:
        out += ["", "The answers"]
        for answer in reading.answers:
            if answer.asked:
                out.append(f"  {answer.key:<26} ASK: {answer.question}")
            else:
                out.append(f"  {answer.key:<26} {json.dumps(answer.value)}   ({answer.source})")
    if reading.notes:
        out += ["", "Notes:", *[f"  - {note}" for note in reading.notes]]
    if reading.refusals:
        out += ["", "Refused by the import:", *[f"  - {line}" for line in reading.refusals]]
    else:
        out += ["", f"Lane input: {reading.kind}. Config fragment:", ""]
        out += [f"  {line}" if line else "" for line in fragment(reading).splitlines()]
    return "\n".join(out)


def item_lines(kind: str, item: dict[str, Any]) -> list[str]:
    if kind == "tomograms":
        if "refusal" in item:
            return [f"  {item['file']}: REFUSED {item['refusal']}"]
        x, y, z = item["shape_xyz"]
        lines = [f"  {item['file']}  series {item['series']}: {x} x {y} x {z} (x, y, z), "
                 f"{item['dtype']}, {_num(round(item['voxel'], 4))} A, thinnest {item['thin']}",
                 f"    history      {'; '.join(item['labels']) or 'no labels'}"]
        if "angles" in item:
            span = (f", {item['angle_count']} angles {_range(item['angle_range'])}"
                    if "angle_count" in item else f": {item.get('angles_refusal')}")
            lines.append(f"    angles       {item['angles']}{span}   ({item['angles_found']})")
        else:
            lines.append(f"    angles       none found: {item['angles_found']}")
        if "wedge" in item:
            lines.append(f"    wedge        {item['wedge']['verdict']}: {item['wedge']['reason']}")
        lines.append(f"    orientation  {item['orientation'] or 'ASK'}   "
                     f"({item['orientation_from']})")
        return lines
    if kind == "picks":
        lines = [f"  format       {item['format'] or 'ASK'}   ({item['why']})"]
        if "count" in item:
            extent = ", ".join(f"{axis} {_range(span)}"
                               for axis, span in zip("xyz", item["extent_xyz"], strict=True))
            angles = "with angles" if item.get("angles") else "no angles"
            whose = f" ({item['positions_of']})" if "positions_of" in item else ""
            lines.append(f"  picks        {item['count']}, {angles}; {extent}{whose}")
        for key, label in (("tomograms", "tomograms"), ("tomo_names", "rlnTomoName"),
                           ("pixel", "pixel"), ("tilt_series_pixel", "tilt series pixel")):
            if key in item:
                lines.append(f"  {label:<12} {item[key]}")
        return lines
    lines = [f"  files        {item['files']} particle files, box {item['box']}"]
    if "rows" in item:
        lines.append(f"  table        {item['rows']} rows, pixel {item.get('pixel')} A "
                     "(column 36)")
    for number, entry in item.get("tomograms", {}).items():
        wedge = entry.get("wedge", {})
        lines.append(f"  tomogram {number:<4} {entry['particles']} particles, Dynamo ytilt "
                     f"{entry['dynamo_ytilt']}, wedge {entry.get('tilt_range')} in the "
                     f"pipeline's sign: {wedge.get('verdict', 'not measured')}")
    return lines


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    kinds = parser.add_subparsers(dest="kind", required=True)
    tomograms = kinds.add_parser("tomograms", help="volumes reconstructed elsewhere")
    tomograms.add_argument("paths", type=Path, nargs="+", help="volumes, or folders of them")
    tomograms.add_argument("--no-wedge", action="store_true",
                           help="do not measure the wedge in the volumes")
    tomograms.add_argument("--max-volumes", type=int, default=DEFAULT_MAX_VOLUMES)
    picks = kinds.add_parser("picks", help="picks made elsewhere")
    picks.add_argument("path", type=Path, help="the picks file")
    picks.add_argument("--tomogram", type=Path, help="the volume the picks were taken in")
    particles = kinds.add_parser("particles", help="a Dynamo data folder and its table")
    particles.add_argument("path", type=Path, help="the folder of particle_<tag> files")
    particles.add_argument("--table", type=Path, help="the table, if not the one .tbl near")
    for sub in (tomograms, picks, particles):
        sub.add_argument("--json", action="store_true", help="print the reading as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    given = args.paths if args.kind == "tomograms" else [args.path]
    given += [path for path in (getattr(args, "tomogram", None), getattr(args, "table", None))
              if path is not None]
    absent = [path for path in given if not path.exists()]
    if absent:
        print(f"no such file or folder: {', '.join(map(str, absent))}", file=sys.stderr)
        return 2
    if args.kind == "tomograms":
        reading = read_tomograms(args.paths, wedge=not args.no_wedge,
                                 max_volumes=args.max_volumes)
    elif args.kind == "picks":
        reading = read_picks(args.path, args.tomogram)
    else:
        reading = read_particles(args.path, args.table)
    if args.json:
        record = asdict(reading)
        record["fragment"] = None if reading.refusals else fragment(reading)
        print(json.dumps(record, indent=2, default=str))
    else:
        print(render(reading))
    return 1 if reading.refusals or not reading.answers else 0


if __name__ == "__main__":
    sys.exit(main())
