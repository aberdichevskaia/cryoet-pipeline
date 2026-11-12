#!/usr/bin/env python3
"""Find EMDB entries to pick with, and say how one would serve as a lane's template.

    python3 scripts/show_emdb.py search '<query>' [--method subtomogramAveraging] [--empiar <id>]
    python3 scripts/show_emdb.py entry EMD-<id> [--pixel <tomogram A/px>] [--diameter <A>]
                                 [--empiar <id>]
    python3 scripts/show_emdb.py measure EMD-<id> | <map file> [--level <contour>] [--cache <dir>]
    ... --json

`search` and `entry` only read the EMDB and EMPIAR APIs. `measure` downloads
the map, once, into a cache: `~/.cache/cryoet-pipeline/emdb` unless --cache
names another folder. Given a lane's `<project>/emdb-cache`, prepare-template
finds the map there and does not download it again.

`search` runs an EMDB search (Solr syntax, such as `ryanodine AND receptor`)
and counts what it finds by method. It lists the candidates with subtomogram
averages first, then whole maps before maps of a region, and newer entries
before older ones. Flags come from the titles, and say so: a map of a region
("local", "focused", "masked"), a membrane or vesicle in the sample, a
tomogram rather than an average. With --empiar, an entry that EMPIAR entry
cross-references is marked as made from the same data.

`entry` reads one entry: method, resolution, symmetry, sample and organism,
the map's box and pixel, the authors' contour, the template the authors picked
with, the microscope and the citation. With --pixel and --diameter it compares
the map with what prepare-template will make of it:
- the template box is twice the diameter in tomogram pixels, rounded up to
  even, and pytom pads a smaller map up to it;
- the map is resampled to the tomogram pixel;
- the template is low-passed to about twice the tomogram pixel.

`measure` gives the size of the particle in a map, for a lane's
`[pick] particle_diameter_angstrom` (src/cryoet_pipeline/map_size.py says how):
its widest extent above the authors' contour, or above --level, and the
diameter a mask around the box centre needs to hold it, which it proposes.
prepare-template draws the mask, and turns the template, about the box centre.
It reads the size again at 1.5 times the level, to show how much the size
depends on it. Density that reaches the map's edge is more than one particle,
and then no diameter is proposed. The map's outermost voxels say what reached
it: noise, when at least half as many of them are as far below their median as
the level is above it as are above the level, so that the level is in the
noise; otherwise a membrane, a neighbour or a lattice. A map file has no
authors' contour, so it needs --level.

The entries are for the person to choose from. The weighing, in the order of
docs/roadmap.md section 23, is the assistant's emdb-template skill.

Exit status: 0 when the search found entries, the entry is released, or the
map was measured; 1 when nothing was found, the entry does not exist, it is
not released, or its map could not be had or measured; 2 for arguments that
contradict each other.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote

from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.template import _box_size_from_diameter
from cryoet_pipeline.emdb import fetch_emdb_map, normalize_emdb_id
from cryoet_pipeline.empiar import DownloadGaveUp, call_with_retries, read_text_url
from cryoet_pipeline.runtime import DevicePreference

SEARCH_URL = "https://www.ebi.ac.uk/emdb/api/search/{query}?rows={rows}&wt=csv&fl={fields}"
ENTRY_URL = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{number}"
EMPIAR_URL = "https://www.ebi.ac.uk/empiar/api/entry/{empiar_id}/"
FIELDS = "emdb_id,title,resolution,structure_determination_method,release_date"
MAX_ROWS = 5000
METHODS = {
    "subtomogramAveraging": "STA",
    "singleParticle": "SPA",
    "helical": "helical",
    "tomography": "tomogram",
    "electronCrystallography": "crystal",
    "twoDArray": "2D array",
}
# Words in a title that mark a map of a region, not a whole particle.
_PART = re.compile(r"\b(local(ly)?|focused|focus|masked|partial|sub-?regions?)\b", re.I)
# Words in a title or sample that put a membrane, or a cell, into the map.
_MEMBRANE = re.compile(
    r"\b(membranes?|nanodiscs?|liposomes?|vesicles?|in situ|lamellae?|bilayers?|detergent)\b",
    re.I,
)
_METHOD_ORDER = ("subtomogramAveraging", "singleParticle", "helical")


def fetch(read: Callable[[str], str], url: str) -> str:
    return call_with_retries(lambda attempt: read(url), what=url, retries=2,
                             retry_wait_s=5.0, sleep=time.sleep)


# --- Search ---------------------------------------------------------------------------


def search(read: Callable[[str], str], query: str, method: str | None) -> list[dict[str, Any]]:
    full = f"({query}) AND structure_determination_method:{method}" if method else query
    text = fetch(read, SEARCH_URL.format(query=quote(full), rows=MAX_ROWS, fields=FIELDS))
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        title = row.get("title") or ""
        try:
            resolution = float(row["resolution"]) if row.get("resolution") else None
        except ValueError:
            resolution = None
        rows.append({
            "emdb_id": row.get("emdb_id"),
            "title": title,
            "resolution": resolution,
            "method": row.get("structure_determination_method") or "",
            "released": (row.get("release_date") or "")[:10],
            "part": bool(_PART.search(title)),
            "membrane": bool(_MEMBRANE.search(title)),
        })
    return rows


def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subtomogram averages first, whole maps before parts, newer before older."""

    def method_rank(row: dict[str, Any]) -> int:
        method = row["method"]
        return _METHOD_ORDER.index(method) if method in _METHOD_ORDER else len(_METHOD_ORDER)

    newest_first = sorted(rows, key=lambda row: row["released"], reverse=True)
    return sorted(newest_first, key=lambda row: (method_rank(row), row["part"]))


# --- One entry ------------------------------------------------------------------------


def walk(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, inner in value.items():
            yield from walk(inner, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from walk(inner, f"{path}[{index}]")
    else:
        yield path, value


def get(value: Any, path: str) -> Any:
    """`a.b[0].c` from nested dicts and lists, or None."""

    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            index = int(part[1:-1])
            value = value[index] if isinstance(value, list) and len(value) > index else None
        else:
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            return None
    return value


def number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("valueOf_")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_entry(read: Callable[[str], str], emdb_id: str) -> dict[str, Any] | None:
    number_part = normalize_emdb_id(emdb_id)
    try:
        text = fetch(read, ENTRY_URL.format(number=number_part))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    raw = json.loads(text)
    determination = "structure_determination_list.structure_determination[0]"
    processing = f"{determination}.image_processing[0]"
    reconstruction = f"{processing}.final_reconstruction"
    microscopy = f"{determination}.microscopy_list.microscopy[0]"
    leaves = list(walk(raw))
    organisms = sorted({str(value) for path, value in leaves
                        if path.startswith("sample.") and path.endswith("organism.valueOf_")})
    detectors = sorted({str(value) for path, value in leaves
                        if path.endswith("film_or_detector_model.valueOf_")})
    paper = re.compile(r"crossreferences\.citation_list\.primary_citation\.citation_type"
                       r"\.author\[\d+\]\.valueOf_")
    depositors = re.compile(r"admin\.authors_list\.author\[\d+\]\.valueOf_")
    # The paper's authors; the depositors only when the entry cites no paper.
    authors = ([str(value) for path, value in leaves if paper.fullmatch(path)]
               or [str(value) for path, value in leaves if depositors.fullmatch(path)])
    contours = get(raw, "map.contour_list.contour") or []
    primary = next((c for c in contours if c.get("primary")), contours[0] if contours else None)
    dimensions = [number(get(raw, f"map.dimensions.{axis}")) for axis in ("col", "row", "sec")]
    pixels = [number(get(raw, f"map.pixel_spacing.{axis}")) for axis in ("x", "y", "z")]
    citation = "crossreferences.citation_list.primary_citation.citation_type"
    defocus = [number(get(raw, f"{microscopy}.{kind}_defocus_{end}"))
               for kind in ("calibrated", "nominal") for end in ("min", "max")]
    return {
        "emdb_id": f"EMD-{number_part}",
        "title": get(raw, "admin.title"),
        "status": get(raw, "admin.current_status.code.valueOf_"),
        "method": get(raw, f"{determination}.method"),
        "resolution": number(get(raw, f"{reconstruction}.resolution")),
        "symmetry": get(raw, f"{reconstruction}.applied_symmetry.point_group"),
        "subtomograms": number(get(raw, f"{reconstruction}.number_subtomograms_used")),
        "tomograms": number(get(raw, f"{processing}.extraction.number_tomograms")),
        "picked_with": get(raw, f"{processing}.extraction.reference_model"),
        "picking": get(raw, f"{processing}.extraction.method"),
        "sample": get(raw, "sample.name.valueOf_"),
        "organisms": organisms,
        "map_box": [int(d) for d in dimensions if d is not None],
        "map_pixel": [p for p in pixels if p is not None],
        "contour": number(primary.get("level")) if primary else None,
        "microscope": get(raw, f"{microscopy}.microscope"),
        "detectors": detectors,
        "defocus_um": next(([lo, hi] for lo, hi in (defocus[:2], defocus[2:])
                            if lo is not None and hi is not None), None),
        "authors": authors,
        "citation": {key: get(raw, f"{citation}.{key}") for key in ("title", "journal", "year")},
    }


def same_data(read: Callable[[str], str], empiar_id: str) -> set[str]:
    """The EMDB entries an EMPIAR entry cross-references: maps made from its data."""

    entries = json.loads(fetch(read, EMPIAR_URL.format(empiar_id=empiar_id)))
    (entry,) = entries.values()
    found = set()
    for reference in entry.get("cross_references") or []:
        name = reference.get("name") if isinstance(reference, dict) else reference
        try:
            found.add(f"EMD-{normalize_emdb_id(str(name))}")
        except ValueError:
            continue
    return found


def template_box(diameter: float, pixel: float) -> int:
    """The box prepare-template makes for this diameter, by its own rule (twice the diameter
    in tomogram pixels, rounded up to even)."""

    context = BackendContext(output_dir=Path(os.devnull), device=DevicePreference.CPU,
                             parameters={"template_particle_diameter_angstrom": diameter})
    box = _box_size_from_diameter(context, voxel_size_angstrom=pixel)
    assert box is not None
    return box


def lane_checks(entry: dict[str, Any], pixel: float | None, diameter: float | None) -> list[str]:
    found: list[str] = []
    if entry["status"] not in (None, "REL"):
        found.append(f"its status is {entry['status']}, not released")
    if entry["method"] == "tomography":
        found.append("it is a tomogram, not an average: no template")
    title = entry["title"] or ""
    if _PART.search(title):
        found.append("its title names a map of a region, not of the whole particle")
    if _MEMBRANE.search(f"{title} {entry['sample'] or ''}"):
        found.append(
            "its title or sample names a membrane or vesicle: if the map holds the membrane, "
            "the template can match every membrane unless the mask leaves it out; look at it "
            "in ChimeraX"
        )
    if entry["symmetry"] and entry["symmetry"] != "C1":
        found.append(f"it was averaged with {entry['symmetry']} symmetry")
    box, map_pixel = entry["map_box"], entry["map_pixel"]
    if len(set(box)) > 1 or len(set(map_pixel)) > 1:
        found.append(f"the map is not a cube of equal pixels: {box} at {map_pixel} A")
    if pixel is None or diameter is None or not box or not map_pixel:
        return found
    across = max(box) * max(map_pixel)
    wanted = template_box(diameter, pixel)
    if across < diameter:
        found.append(f"the map is {across:.0f} A across, less than the {diameter:g} A particle: "
                     "it cannot hold it")
    elif across < wanted * pixel:
        found.append(f"the template box is {wanted} px = {wanted * pixel:.0f} A, twice the "
                     f"diameter; the map is {across:.0f} A across, so pytom pads it up to the box")
    else:
        found.append(f"the map is {across:.0f} A across, more than the {wanted} px = "
                     f"{wanted * pixel:.0f} A template box; pytom keeps the map's box and warns")
    factor = pixel / max(map_pixel)
    if factor < 1:
        found.append(f"the map's pixel, {max(map_pixel):g} A, is coarser than the tomogram's, "
                     f"{pixel:g} A: it would be resampled finer")
    else:
        found.append(f"the map is resampled from {max(map_pixel):g} to {pixel:g} A/px "
                     f"({factor:.1f} times coarser)")
    if entry["resolution"] is not None:
        low_pass = 2 * pixel
        relation = "finer" if entry["resolution"] < low_pass else "coarser"
        found.append(f"{entry['resolution']:g} A is {relation} than the template's low-pass, about "
                     f"2 x {pixel:g} = {low_pass:.0f} A" + (
                         ": the map limits the template" if relation == "coarser" else ""))
    return found


# --- Measure a map --------------------------------------------------------------------


def default_cache() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "cryoet-pipeline" / "emdb"


def measure(args: argparse.Namespace, read: Callable[[str], str]) -> dict[str, Any]:
    # Here, not at the top: only measuring needs scipy and mrcfile.
    from cryoet_pipeline import map_size

    report: dict[str, Any] = {
        "command": "measure", "source": args.source, "emdb_id": None, "map": None,
        "level": args.level, "level_from": None if args.level is None else "given",
        "first": None, "second_level": None, "second": None, "diameter_angstrom": None,
        "problem": None, "exit": 1,
    }
    if Path(args.source).is_file():
        path = Path(args.source)
    else:
        report["emdb_id"] = f"EMD-{normalize_emdb_id(args.source)}"
        entry = read_entry(read, args.source)
        if entry is None:
            report["problem"] = "EMDB has no such entry"
            return report
        if entry["status"] not in (None, "REL"):
            report["problem"] = f"its status is {entry['status']}, not released: it has no map"
            return report
        if args.level is None:
            if entry["contour"] is None:
                report["problem"] = "EMDB gives no contour for it: give one with --level"
                return report
            report["level"], report["level_from"] = entry["contour"], "authors"
        path = fetch_emdb_map(report["emdb_id"], cache_dir=args.cache or default_cache(),
                              progress=lambda message: print(message, file=sys.stderr))
    report["map"] = str(path)
    data, voxel = map_size.read_map(path)
    first = map_size.measure_density(data, voxel, report["level"])
    report["first"] = size_report(first)
    if report["level"] > 0:
        report["second_level"] = report["level"] * map_size.LEVEL_STEP
        try:
            report["second"] = size_report(
                map_size.measure_density(data, voxel, report["second_level"]))
        except ValueError:
            report["second"] = None
    if first.faces_reached == 0 and first.kept <= map_size.MOST_PIECES:
        report["diameter_angstrom"] = math.ceil(first.mask_diameter_angstrom)
    report["exit"] = 0
    return report


def size_report(size: Any) -> dict[str, Any]:
    return {**dataclasses.asdict(size), "mask_diameter_angstrom": size.mask_diameter_angstrom,
            "in_noise": size.in_noise}


# --- Output ---------------------------------------------------------------------------


def render_search(report: dict[str, Any]) -> str:
    counts = ", ".join(f"{n} {METHODS.get(m, m or 'no method')}"
                       for m, n in report["by_method"].items())
    out = [f"{report['found']} entries for {report['query']!r}" + (
        f" by {report['method']}" if report["method"] else "") + (f": {counts}" if counts else "")]
    if report["found"] >= MAX_ROWS:
        out[0] += f" (the first {MAX_ROWS} read)"
    for row in report["rows"]:
        flags = [flag for flag, on in (("part", row["part"]), ("membrane?", row["membrane"]),
                                       ("same data", row.get("same_data"))) if on]
        resolution = f"{row['resolution']:g} A" if row["resolution"] is not None else "-"
        out.append(f"  {row['emdb_id']:<10} {METHODS.get(row['method'], row['method']):<8} "
                   f"{resolution:>7}  {row['released'][:4]}  "
                   f"{('[' + ', '.join(flags) + '] ') if flags else ''}{row['title'][:90]}")
    if report["found"] > len(report["rows"]):
        out.append(f"  ... {report['found'] - len(report['rows'])} more (--rows to show them)")
    out.append("Flags are read from the titles. Read an entry before proposing it: "
               "show_emdb.py entry EMD-<id>")
    return "\n".join(out)


def render_entry(report: dict[str, Any]) -> str:
    entry = report["entry"]
    if entry is None:
        return f"{report['emdb_id']}: EMDB has no such entry"
    method = METHODS.get(entry["method"] or "", entry["method"])
    facts = [method or "method not given"]
    if entry["resolution"] is not None:
        facts.append(f"{entry['resolution']:g} A")
    if entry["symmetry"]:
        facts.append(f"{entry['symmetry']} symmetry")
    if entry["subtomograms"]:
        facts.append(f"{entry['subtomograms']:.0f} subtomograms"
                     + (f" from {entry['tomograms']:.0f} tomograms" if entry["tomograms"] else ""))
    out = [f"{entry['emdb_id']}: {entry['title']}", f"  method: {', '.join(facts)}"]
    if entry["sample"] or entry["organisms"]:
        out.append(f"  sample: {entry['sample'] or '-'}"
                   + (f"; {', '.join(entry['organisms'])}" if entry["organisms"] else ""))
    if entry["map_box"]:
        box = " x ".join(map(str, entry["map_box"]))
        pixel = "/".join(f"{p:g}" for p in sorted(set(entry["map_pixel"])))
        across = max(entry["map_box"]) * max(entry["map_pixel"]) if entry["map_pixel"] else None
        line = f"  map: {box} at {pixel} A/px"
        if across:
            line += f" = {across:.0f} A across"
        if entry["contour"] is not None:
            line += f"; authors' contour {entry['contour']:g}"
        out.append(line)
    if entry["picked_with"]:
        out.append(f"  picked with: {entry['picked_with']}"
                   + (f" ({entry['picking']})" if entry["picking"] else ""))
    if entry["microscope"]:
        line = f"  microscope: {entry['microscope']}"
        if entry["detectors"]:
            line += f", {', '.join(entry['detectors'])}"
        if entry["defocus_um"]:
            line += f", defocus {entry['defocus_um'][0]:g} to {entry['defocus_um'][1]:g} um"
        out.append(line)
    citation = entry["citation"]
    if citation.get("title"):
        authors = ", ".join(entry["authors"][:3]) + (" et al." if len(entry["authors"]) > 3 else "")
        out.append(f"  citation: {citation['title']} {citation.get('journal') or ''} "
                   f"{citation.get('year') or ''}" + (f" ({authors})" if authors else ""))
    if report["checks"]:
        lane = (f" for a lane at {report['pixel']:g} A/px and a {report['diameter']:g} A particle"
                if report["pixel"] and report["diameter"] else "")
        out.append(f"Checks{lane}:")
        out.extend(f"  - {check}" for check in report["checks"])
    if report.get("empiar"):
        out.append(f"  - EMPIAR-{report['empiar']} "
                   + ("cross-references it: made from the same data, a fine template but no "
                      "independent check of the result" if report["same_data"]
                      else "does not cross-reference it"))
    return "\n".join(out)


def render_measure(report: dict[str, Any]) -> str:
    from cryoet_pipeline.map_size import KEEP_SHARE, LEVEL_STEP, MOST_PIECES

    name = report["emdb_id"] or report["source"]
    first = report["first"]
    if first is None:
        return f"{name}: {report['problem']}"
    level = "the authors' contour" if report["level_from"] == "authors" else "the level"
    box = " x ".join(map(str, first["box_px"]))
    out = [f"{name} above {level}, {report['level']:g} (a {box} box at "
           f"{first['voxel_size_angstrom']:g} A/px; {first['share_of_map']:.1%} of the map "
           "is above it):"]
    if first["kept"] == 1:
        line = f"  one piece holds {first['kept_share']:.0%} of the volume above it"
    elif first["kept"] <= MOST_PIECES:
        line = (f"  {first['kept']} pieces of at least {KEEP_SHARE:.0%} of the largest hold "
                f"{first['kept_share']:.0%} of the volume above it and are measured together: "
                "look at the map to see whether they are one particle")
    else:
        line = (f"  {first['kept']} pieces of at least {KEEP_SHARE:.0%} of the largest hold "
                f"{first['kept_share']:.0%} of the volume above it: too many to be one "
                "particle, so the level is above its density or the map is noise")
    left = first["pieces"] - first["kept"]
    if left:
        line += (f"; {left} smaller {'one is' if left == 1 else 'ones are'} left out, the "
                 f"largest {first['largest_left_out']:.1%} of the largest piece")
    out.append(line)
    out.append(f"  widest across: {first['widest_angstrom']:.0f} A")
    out.append("  around the box centre, where prepare-template draws the mask and turns the "
               f"template, it reaches {first['from_centre_angstrom']:.0f} A: a mask holds it at "
               f"{first['mask_diameter_angstrom']:.0f} A across")
    if first["faces_reached"]:
        edge = (f"of the map's outermost voxels, {first['edge_above']:.1%} are above the level "
                f"and {first['edge_below']:.1%} as far below their median, "
                f"{first['edge_median']:.3g}, as the level is above it")
        if first["in_noise"]:
            out.append(f"  it reaches the map's edge on {first['faces_reached']} of 6 faces, and "
                       f"what reaches them is noise: {edge}. The level is in the noise: noise "
                       "above it is measured with the particle, so these sizes are the noise's")
        else:
            out.append(f"  it reaches the map's edge on {first['faces_reached']} of 6 faces, and "
                       f"not as noise does: {edge}. A membrane, a neighbour or a lattice is in "
                       "the map, so these are the map's sizes, not one particle's; look at it in "
                       "ChimeraX")
    else:
        out.append(f"  it stays {first['inside_angstrom']:.0f} A or more inside the map's edges")
    if report["second_level"] is not None:
        second = report["second"]
        if second is None:
            out.append(f"  at {LEVEL_STEP:g} x the level, {report['second_level']:g}: nothing "
                       "is above it")
        else:
            out.append(f"  at {LEVEL_STEP:g} x the level, {second['level']:g}: "
                       f"{second['widest_angstrom']:.0f} A across, "
                       f"{second['mask_diameter_angstrom']:.0f} A for the mask"
                       + ("; in the noise there too" if second["in_noise"] else ""))
    if report["diameter_angstrom"] is None:
        if first["in_noise"]:
            out.append("No diameter to propose: the level is in the noise. Give a higher "
                       "--level, or measure one particle in ChimeraX")
        else:
            why = ("the density reaches the map's edge" if first["faces_reached"]
                   else f"more than {MOST_PIECES} pieces are measured")
            out.append(f"No diameter to propose: {why}. Measure one particle in ChimeraX, or "
                       "give another --level")
    else:
        out.append(f"For [pick] particle_diameter_angstrom: {report['diameter_angstrom']} A, "
                   "the mask's diameter")
        if report["emdb_id"]:
            out.append(f"Check it against the lane: show_emdb.py entry {report['emdb_id']} "
                       f"--pixel <tomogram A/px> --diameter {report['diameter_angstrom']}")
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    find = commands.add_parser("search", help="search EMDB")
    find.add_argument("query", help="Solr syntax, such as 'ryanodine AND receptor'")
    find.add_argument("--method", choices=sorted(METHODS),
                      help="only entries of this method")
    find.add_argument("--rows", type=int, default=25, help="candidates to list (25)")
    one = commands.add_parser("entry", help="read one entry")
    one.add_argument("emdb_id", help="EMD-1234, emd_1234 or 1234")
    one.add_argument("--pixel", type=float, help="the tomogram's pixel size, A")
    one.add_argument("--diameter", type=float, help="the particle's widest diameter, A")
    for command in (find, one):
        command.add_argument("--empiar", help="mark entries made from this EMPIAR entry's data")
    size = commands.add_parser("measure", help="measure the particle in a map")
    size.add_argument("source", help="EMD-1234, emd_1234, 1234, or a map file")
    size.add_argument("--level", type=float,
                      help="the contour level; for an entry, the authors' unless given")
    size.add_argument("--cache", type=Path,
                      help="where downloaded maps are kept (~/.cache/cryoet-pipeline/emdb)")
    for command in (find, one, size):
        command.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def build(args: argparse.Namespace, read: Callable[[str], str]) -> dict[str, Any]:
    same = same_data(read, args.empiar) if args.empiar else set()
    if args.command == "search":
        rows = search(read, args.query, args.method)
        for row in rows:
            row["same_data"] = row["emdb_id"] in same
        return {"command": "search", "query": args.query, "method": args.method,
                "found": len(rows),
                "by_method": dict(Counter(row["method"] for row in rows).most_common()),
                "rows": ranked(rows)[: args.rows], "exit": 0 if rows else 1}
    entry = read_entry(read, args.emdb_id)
    report: dict[str, Any] = {
        "command": "entry", "emdb_id": f"EMD-{normalize_emdb_id(args.emdb_id)}",
        "entry": entry, "pixel": args.pixel, "diameter": args.diameter,
        "empiar": args.empiar, "checks": [], "exit": 1,
    }
    if entry is None:
        return report
    report["checks"] = lane_checks(entry, args.pixel, args.diameter)
    report["same_data"] = entry["emdb_id"] in same
    report["exit"] = 0 if entry["status"] in (None, "REL") else 1
    return report


def main(argv: list[str] | None = None, read: Callable[[str], str] = read_text_url) -> int:
    args = parse_args(argv)
    if args.command == "measure":
        return main_measure(args, read)
    if args.command == "entry":
        try:
            normalize_emdb_id(args.emdb_id)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if (args.pixel is None) != (args.diameter is None):
            print("give --pixel and --diameter together: the checks need both", file=sys.stderr)
            return 2
        if (args.pixel is not None and args.pixel <= 0) or (
                args.diameter is not None and args.diameter <= 0):
            print("--pixel and --diameter must be positive", file=sys.stderr)
            return 2
    if args.empiar is not None and not args.empiar.isdigit():
        print(f"an EMPIAR id is a number, not {args.empiar!r}", file=sys.stderr)
        return 2
    try:
        report = build(args, read)
    except (HTTPError, URLError, DownloadGaveUp, ValueError) as exc:
        print(f"EMDB or EMPIAR could not be read: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_search(report) if args.command == "search" else render_entry(report))
    return int(report["exit"])


def main_measure(args: argparse.Namespace, read: Callable[[str], str]) -> int:
    if not Path(args.source).is_file():
        try:
            normalize_emdb_id(args.source)
        except ValueError:
            print(f"{args.source!r} is neither a map file nor an EMDB id", file=sys.stderr)
            return 2
    elif args.level is None:
        print("a map file has no authors' contour: give its level with --level",
              file=sys.stderr)
        return 2
    try:
        report = measure(args, read)
    except (OSError, DownloadGaveUp) as exc:
        # URLError and HTTPError are OSErrors too.
        print(f"the entry or its map could not be had: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, default=str) if args.json else render_measure(report))
    return int(report["exit"])


if __name__ == "__main__":
    sys.exit(main())
