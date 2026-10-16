"""Read a Dynamo project's multireference classification into a `ClassificationRun`.

What Dynamo 1.1.511's CPU project path leaves for every finished iteration N
(read from MRA projects run on the Mac on 2026-10-16):

- `results/ite_000N/currently_surviving_references_ite_000N.txt`: the channels
  that searched in iteration N.
- `results/ite_000N/averages/refined_table_ref_00K_ite_000N.tbl`: channel K's
  pose and CC (column 10) for every particle of its table.
- `results/ite_000N/averages/average_ref_00K_ite_000N.em`: channel K's class
  average, for every channel that kept particles.
- `cards/ite_000N/card_ite_ite_000N.card`: the iteration's parameters, and its
  times once it has finished.

With `mra = 1` the MRA plugin rewrites the tables of the channels that kept
particles: column 34 is each particle's class, column 3 marks the class's own
members. A channel left with no particle is dropped for good. Its table is not
rewritten and still claims every particle (column 34 its own number, column 3
all ones), and the next iteration's list of channels leaves it out. So classes
are read from a channel the next iteration still lists, and an iteration counts
as finished once that list exists: Dynamo writes it at the end of every
iteration, the last one included. A project still running is read up to its
last finished iteration.

The parameters of every round are in `settings/virtual_project.mat`; without
scipy the first iteration's card stands in for round 1. Paths in them are
relative to the folder the project was made in, which holds the project folder.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.backends.dynamo_search import average_class, resample
from cryoet_pipeline.reports.classification import (
    ClassificationRun,
    ClassResult,
    IterationResult,
    Parameter,
    Pose,
    Volume,
)

TAG = 0
AVERAGED = 2
SHIFTS = slice(3, 6)
ANGLES = slice(6, 9)
CC = 9
FTYPE = 12
TILT_RANGE = slice(13, 15)
TOMOGRAM = 19
CLASS = 33

ROUNDS = range(1, 9)

AREA_SEARCH_MODES = {
    0: "unlimited",
    1: "from the box centre",
    2: "from the previous pose",
    3: "from the round's first pose",
    4: "from the project's first pose",
}
THRESHOLD_MODES = {
    0: "none",
    1: "absolute CC",
    2: "times the mean CC",
    3: "mean CC plus σ times the value",
    4: "the best N particles",
    5: "the best fraction",
}
SEED_KINDS = {
    "subset": "average of {count} random particles at their starting poses",
    "blob": "average of {count} random particles at random orientations",
    "noise": "white noise",
}
DYNAMO_TIME = "%d-%b-%Y %H:%M:%S"


def read_em(path: Path) -> np.ndarray:
    """An EM-format map, Dynamo's own, as a (z, y, x) float32 array."""

    raw = path.read_bytes()
    if len(raw) < 512:
        raise ValueError(f"{path}: shorter than an EM header")
    machine, code = raw[0], raw[3]
    if machine == 6:
        order = "<"
    elif machine in (3, 5):
        order = ">"
    else:
        raise ValueError(f"{path}: unknown EM machine code {machine}")
    types = {1: "i1", 2: "i2", 4: "i4", 5: "f4", 9: "f8"}
    if code not in types:
        raise ValueError(f"{path}: EM data type {code} is not read here")
    nx, ny, nz = (int(value) for value in np.frombuffer(raw[4:16], dtype=f"{order}i4"))
    dtype = np.dtype(order + types[code])
    count = nx * ny * nz
    if len(raw) < 512 + count * dtype.itemsize:
        raise ValueError(f"{path}: holds less data than its header says")
    data = np.frombuffer(raw, dtype=dtype, count=count, offset=512)
    return data.reshape(nz, ny, nx).astype(np.float32)


def read_map(path: Path) -> np.ndarray:
    """An EM or MRC map as (z, y, x) float32."""

    if path.suffix.lower() == ".em":
        return read_em(path)
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float32)


def read_table(path: Path) -> np.ndarray:
    """A Dynamo table. MATLAB writes a complex entry as `0+1.2074e-06i`; its real part is kept."""

    rows = [[_number(token) for token in line.split()]
            for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path}: an empty table")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: rows of {sorted(widths)} columns")
    return np.array(rows, dtype=float)


def _number(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        if token.endswith("i"):
            return complex(token[:-1] + "j").real
        raise


def read_card(path: Path) -> dict[str, str]:
    """A Dynamo card: `name=value;` lines."""

    values = {}
    for line in path.read_text().splitlines():
        name, sign, value = line.strip().partition("=")
        if sign:
            values[name.strip()] = value.strip().rstrip(";").strip()
    return values


def _channel_list(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    return [int(token) for token in path.read_text().split()]


def _read_vpr(path: Path, problems: list[str]) -> dict[str, Any]:
    if not path.exists():
        problems.append(f"{path} is missing; parameters come from the first iteration's card")
        return {}
    try:
        from scipy.io import loadmat  # type: ignore[import-untyped]

        card = loadmat(path, squeeze_me=True, struct_as_record=False)["card"]
        return {name: getattr(card, name) for name in card._fieldnames}
    except Exception as error:  # noqa: BLE001 - the card still gives round 1
        problems.append(f"{path.name} could not be read ({type(error).__name__}: {error}); "
                        "parameters come from the first iteration's card")
        return {}


def read_dynamo_classification(
    project: Path,
    *,
    manifest: Path | None = None,
    pixel_size_angstrom: float | None = None,
    initial_average: Path | None = None,
    compute_initial_average: bool = True,
) -> ClassificationRun:
    """Everything the classification report shows about one Dynamo project folder.

    `manifest` is an export manifest of `scripts/export_dynamo.py` (pixel size,
    tilt-series and picking score of every tag); by default the one beside the
    project, if any. `initial_average` is a map to show as the average of all
    particles at their starting poses; without one it is computed when its page
    is drawn, as Dynamo's console `dynamo_average` would (fcompensate, fmin 1,
    normalised under the alignment mask), through the pipeline's port of it.
    """

    project = project.resolve()
    base = project.parent
    if not (project / "results").is_dir():
        raise ValueError(f"{project} is not a Dynamo project with results: no results/ folder")
    problems: list[str] = []
    vpr = _read_vpr(project / "settings" / "virtual_project.mat", problems)
    card_path = project / "cards" / "ite_0001" / "card_ite_ite_0001.card"
    card = read_card(card_path) if card_path.exists() else {}

    def setting(name: str, round_: int = 1) -> Any:
        if vpr:
            return vpr.get(f"{name}_r{round_}", vpr.get(name))
        return card.get(name) if round_ == 1 else None

    starting = _starting_table(project, problems)
    tags = np.sort(starting[:, TAG].astype(int))
    nref = int(_as_float(setting("nref")) or len(_channels_in(project / "settings")))
    mra = bool(_as_float(setting("mra")))

    folder = vpr.get("folder_data") or card.get("folder_data") or "particles"
    particles = base / str(folder)
    extension = str(vpr.get("feature_extension_data") or card.get("feature_extension_data") or "")
    if not particles.is_dir():
        problems.append(f"particle folder {particles} is missing: no galleries and no "
                        "computed initial average")

    manifest_path = manifest or (base / "export_manifest.json")
    series: dict[int, str] = {}
    scores: dict[int, float] = {}
    manifest_spacing = None
    manifest_wedges: dict[str, Any] = {}
    if manifest_path.exists():
        series, scores, manifest_spacing, manifest_wedges = _read_manifest(manifest_path)
    elif manifest is not None:
        problems.append(f"manifest {manifest} is missing")
    apix_setting = _as_float(vpr.get("apix") or card.get("apix"))
    if pixel_size_angstrom:
        spacing, spacing_source = float(pixel_size_angstrom), "given to the report"
    elif manifest_spacing:
        spacing, spacing_source = manifest_spacing, manifest_path.name
    elif apix_setting and apix_setting != 1:
        spacing, spacing_source = apix_setting, "the project's apix"
    else:
        spacing, spacing_source = None, ""

    references = {}
    for channel in range(1, nref + 1):
        path = project / "settings" / f"template_initial_ref_{channel:03d}.em"
        if not path.exists():
            path = (project / "results" / "ite_0001" / "starting_values"
                    / f"starting_reference_ref_{channel:03d}_ite_0001.em")
        if path.exists():
            references[channel] = Volume(_relative(path, base), partial(read_em, path))
        else:
            problems.append(f"channel {channel}: no starting reference found")
    seeds_folder = base / str(vpr.get("file_template_initial") or card.get("file_template_initial")
                              or "")
    notes = _seed_notes(seeds_folder / "seed_kinds.txt")

    iterations, refined = _iterations(project, tags, mra, problems)
    box = int(_as_float(vpr.get("feature_data_sidelength") or card.get("feature_data_sidelength"))
              or 0)
    if not box and references:
        box = next(iter(references.values())).load().shape[0]

    rows: dict[Pose, dict[int, np.ndarray]] = {}

    def pose_row(tag: int, pose: Pose) -> np.ndarray:
        if pose not in rows:
            table = starting if pose is None else refined[pose]
            rows[pose] = {int(row[TAG]): row for row in table}
        return rows[pose][tag]

    def aligned(tag: int, pose: Pose) -> np.ndarray:
        row = pose_row(tag, pose)
        particle = read_map(_particle_path(particles, tag, extension))
        volume, valid = resample(  # type: ignore[no-untyped-call]
            particle, -row[ANGLES][::-1], -row[SHIFTS]
        )
        result = np.asarray(volume, dtype=np.float32)
        result[~np.asarray(valid, dtype=bool)] = np.nan
        return result

    mask_path = project / "settings" / "mask.em"
    if initial_average is not None:
        average: Volume | None = Volume(str(initial_average), partial(read_map, initial_average))
    elif compute_initial_average and particles.is_dir() and mask_path.exists():
        average = Volume(
            "computed for this report",
            _cached(partial(_initial_average, starting, particles, extension, mask_path)),
        )
    else:
        average = None

    parameters = _parameters(
        setting=setting, vpr=vpr, card=card, project=project, base=base, tags=tags,
        starting=starting, box=box, spacing=spacing, spacing_source=spacing_source,
        series=series, wedges=manifest_wedges, nref=nref, mra=mra, notes=notes,
        references=references, particles=particles, iterations=iterations,
    )
    planned = sum(int(_as_float(vpr.get(f"ite_r{r}")) or 0) for r in ROUNDS) if vpr else None
    return ClassificationRun(
        name=project.name,
        source=project,
        software="Dynamo MRA project" if mra else "Dynamo project",
        box_px=box,
        pixel_size_angstrom=spacing,
        tags=tags,
        references=references,
        reference_notes=notes,
        parameters=parameters,
        iterations=iterations,
        aligned_particle=aligned,
        initial_average=average,
        planned_iterations=planned or None,
        series=series,
        scores=scores,
        score_label="picking score (LCCmax, from the export manifest)",
        problems=problems,
    )


def _iterations(
    project: Path, tags: np.ndarray, mra: bool, problems: list[str]
) -> tuple[list[IterationResult], dict[tuple[int, int], np.ndarray]]:
    results = project / "results"
    iterations: list[IterationResult] = []
    refined: dict[tuple[int, int], np.ndarray] = {}
    number = 1
    while True:
        here = results / f"ite_{number:04d}"
        searched = _channel_list(here / f"currently_surviving_references_ite_{number:04d}.txt")
        if searched is None:
            break
        following = results / f"ite_{number + 1:04d}"
        kept = _channel_list(following / f"currently_surviving_references_ite_{number + 1:04d}.txt")
        averages = here / "averages"
        if kept is None:
            if averages.exists():
                problems.append(f"iteration {number} has not finished (iteration {number + 1} "
                                "has no list of channels yet); the report stops before it")
            break
        tables = {}
        for channel in searched:
            path = averages / f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl"
            if path.exists():
                tables[channel] = read_table(path)
        missing = [channel for channel in searched if channel not in tables]
        if missing:
            problems.append(f"iteration {number}: no refined table for channel(s) {missing}; the "
                            "report stops before it")
            break
        if mra:
            members = _mra_members(tables, kept, number, problems)
        else:
            members = {channel: np.sort(table[table[:, AVERAGED] == 1, TAG].astype(int))
                       for channel, table in tables.items()}
            members = {channel: tags_ for channel, tags_ in members.items() if len(tags_)}
        classes = {}
        for channel, chosen in members.items():
            table = tables[channel]
            index = {int(tag): i for i, tag in enumerate(table[:, TAG].astype(int))}
            correlations = np.array([table[index[int(tag)], CC] for tag in chosen])
            path = averages / f"average_ref_{channel:03d}_ite_{number:04d}.em"
            classes[channel] = ClassResult(
                channel,
                chosen,
                Volume(_relative(path, project.parent), partial(read_em, path))
                if path.exists() else None,
                float(np.median(correlations)) if correlations.size else None,
            )
            if not path.exists():
                problems.append(f"iteration {number}: channel {channel} kept particles but wrote "
                                "no average")
        unclaimed = len(set(tags.tolist()) - {int(tag) for chosen in members.values()
                                              for tag in chosen})
        if unclaimed:
            problems.append(f"iteration {number}: {unclaimed} particles of the starting table are "
                            "in no class")
        card_path = project / "cards" / f"ite_{number:04d}" / f"card_ite_ite_{number:04d}.card"
        card = read_card(card_path) if card_path.exists() else {}
        iterations.append(IterationResult(
            number,
            tuple(searched),
            classes,
            card.get("time_start_computation"),
            card.get("time_end_iteration"),
        ))
        for channel, table in tables.items():
            refined[(number, channel)] = table
        number += 1
    return iterations, refined


def _mra_members(tables: Mapping[int, np.ndarray], kept: list[int], number: int,
                 problems: list[str]) -> dict[int, np.ndarray]:
    """Each kept channel's members, read from column 34 of a kept channel's table."""

    live = [channel for channel in kept if channel in tables]
    if not live:
        problems.append(f"iteration {number}: none of the channels kept for iteration "
                        f"{number + 1} searched in it")
        return {}
    first = tables[live[0]]
    tags = first[:, TAG].astype(int)
    classes = np.rint(first[:, CLASS]).astype(int)
    for channel in live[1:]:
        other = tables[channel]
        same = np.array_equal(other[:, TAG].astype(int), tags) and np.array_equal(
            np.rint(other[:, CLASS]).astype(int), classes)
        if not same:
            problems.append(f"iteration {number}: channels {live[0]} and {channel} disagree on "
                            f"the classes (column 34); the report reads channel {live[0]}'s")
    stray = sorted(set(classes.tolist()) - set(live))
    if stray:
        problems.append(f"iteration {number}: column 34 puts particles in channel(s) {stray}, "
                        f"which iteration {number + 1} does not list")
    members = {}
    for channel in live:
        chosen = np.sort(tags[classes == channel])
        if not len(chosen):
            problems.append(f"iteration {number}: channel {channel} goes on to iteration "
                            f"{number + 1} but column 34 gives it no particle")
            continue
        table = tables[channel]
        index = {int(tag): i for i, tag in enumerate(table[:, TAG].astype(int))}
        unmarked = sum(1 for tag in chosen if int(tag) not in index
                       or table[index[int(tag)], AVERAGED] != 1)
        if unmarked:
            problems.append(f"iteration {number}: {unmarked} of channel {channel}'s particles are "
                            "not marked as averaged in its table (column 3)")
        members[channel] = chosen
    return members


def _starting_table(project: Path, problems: list[str]) -> np.ndarray:
    folder = project / "results" / "ite_0001" / "starting_values"
    paths = sorted(folder.glob("starting_table_ref_*_ite_0001.tbl"))
    if not paths:
        paths = sorted((project / "settings").glob("table_initial_ref_*.tbl"))
    if not paths:
        raise ValueError(f"{project}: no starting table (results/ite_0001/starting_values or "
                         "settings/table_initial_ref_*.tbl)")
    first = read_table(paths[0])
    for path in paths[1:]:
        # Every channel's copy carries its own number in column 34; compare the rest.
        if not np.array_equal(_without_class(read_table(path)), _without_class(first)):
            problems.append(f"the channels start from different tables; the starting gallery and "
                            f"average use {paths[0].name}")
            break
    return first


def _without_class(table: np.ndarray) -> np.ndarray:
    return np.delete(table, CLASS, axis=1) if table.shape[1] > CLASS else table


def _channels_in(settings: Path) -> list[Path]:
    return sorted(settings.glob("template_initial_ref_*.em"))


def _particle_path(folder: Path, tag: int, extension: str) -> Path:
    names = [f"particle_{tag:05d}.{extension}"] if extension else []
    names += [f"particle_{tag:05d}.em", f"particle_{tag:05d}.mrc"]
    for name in names:
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"no file for particle {tag} in {folder}")


def _initial_average(starting: np.ndarray, particles: Path, extension: str,
                     mask_path: Path) -> np.ndarray:
    if not np.all(starting[:, FTYPE] == 1):
        raise ValueError("only single-axis wedges (ftype 1) are averaged here")
    order = np.argsort(starting[:, TAG])
    rows = starting[order]
    volumes = (read_map(_particle_path(particles, int(tag), extension)) for tag in rows[:, TAG])
    result = average_class(
        volumes,
        rows[:, ANGLES],
        rows[:, SHIFTS],
        [tuple(pair) for pair in rows[:, TILT_RANGE]],
        read_em(mask_path),
        fmin=1,
    )
    return np.asarray(result.average, dtype=np.float32)


def _cached(build: Callable[[], np.ndarray]) -> Callable[[], np.ndarray]:
    cache: dict[str, np.ndarray] = {}

    def load() -> np.ndarray:
        if "volume" not in cache:
            cache["volume"] = build()
        return cache["volume"]

    return load


def _read_manifest(
    path: Path,
) -> tuple[dict[int, str], dict[int, float], float | None, dict[str, Any]]:
    data = json.loads(path.read_text())
    series: dict[int, str] = {}
    scores: dict[int, float] = {}
    for entry in data.get("particles", []):
        tag = int(entry["tag"])
        if entry.get("run"):
            series[tag] = str(entry["run"])
        score = (entry.get("native_record") or {}).get("score")
        if score is not None and math.isfinite(float(score)):
            scores[tag] = float(score)
    spacing = data.get("pixel_spacing_angstrom")
    wedges = (data.get("wedge") or {}).get("ranges") or {}
    return series, scores, float(spacing) if spacing else None, wedges


def _seed_notes(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    notes = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            kind = parts[1]
            count = parts[2] if len(parts) > 2 else "?"
            notes[int(parts[0])] = SEED_KINDS.get(kind, kind).format(count=count)
    return notes


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return " ".join(_text(item) for item in value.ravel())
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, np.generic):
        return _text(value.item())
    return str(value).strip()


def _sphere_radius(mask: np.ndarray) -> float:
    return float((3 * np.count_nonzero(mask > 0.5) / (4 * math.pi)) ** (1 / 3))


def _parameters(
    *, setting: Callable[..., Any], vpr: Mapping[str, Any], card: Mapping[str, str],
    project: Path, base: Path, tags: np.ndarray, starting: np.ndarray, box: int,
    spacing: float | None, spacing_source: str, series: Mapping[int, str],
    wedges: Mapping[str, Any], nref: int, mra: bool, notes: Mapping[int, str],
    references: Mapping[int, Volume], particles: Path, iterations: list[IterationResult],
) -> list[Parameter]:
    rows: list[Parameter] = []

    def add(group: str, name: str, value: str) -> None:
        rows.append(Parameter(group, name, value))

    def nm(px: float) -> str:
        return f" ({px * spacing / 10:.1f} nm)" if spacing else ""

    add("Data", "particles", f"{len(tags):,}")
    add("Data", "box", f"{box} px" + nm(box))
    add("Data", "pixel size",
        f"{spacing:.3f} Å, from {spacing_source}" if spacing else "not recorded")
    add("Data", "particle folder", _relative(particles, base))
    # One row per tomogram: its particles and the tilt range of its wedge, by tilt-series name
    # when the manifest gives them, by the table's tomogram number otherwise.
    counts: dict[str, int] = {}
    tilts: dict[str, tuple[float, float]] = {}
    for row in starting:
        name = series.get(int(row[TAG])) or f"tomogram {int(row[TOMOGRAM])}"
        counts[name] = counts.get(name, 0) + 1
        tilts.setdefault(name, (float(row[TILT_RANGE][0]), float(row[TILT_RANGE][1])))
    for name in sorted(counts):
        lo, hi = (float(wedges[name][0]), float(wedges[name][1])) if name in wedges else tilts[name]
        add("Data", name, f"{counts[name]:,} particles, tilts {lo:+.0f}° to {hi:+.0f}°")

    add("References", "channels", str(nref))
    add("References", "particles swap between channels", "yes (mra 1)" if mra else "no")
    for channel in range(1, nref + 1):
        source = references[channel].source if channel in references else "missing"
        add("References", f"channel {channel}", notes.get(channel) or source)

    masks = {
        "alignment": (project / "settings" / "mask.em",
                      vpr.get("file_mask") or card.get("file_mask")),
        "classification": (project / "settings" / "mask_classification.em",
                           vpr.get("file_mask_classification")
                           or card.get("file_mask_classification")),
        "smoothing": (project / "settings" / "mask_smoothing.em",
                      vpr.get("file_mask_smoothing") or card.get("file_mask_smoothing")),
    }
    alignment = read_em(masks["alignment"][0]) if masks["alignment"][0].exists() else None
    for label, (path, file_name) in masks.items():
        if not path.exists():
            add("Masks", label, _text(file_name) or "none")
            continue
        mask = read_em(path)
        if label != "alignment" and alignment is not None and np.array_equal(mask, alignment):
            add("Masks", label, "the alignment mask")
            continue
        radius = _sphere_radius(mask)
        add("Masks", label, f"{_text(file_name)}: its voxels above 0.5 hold a sphere of radius "
            f"{radius:.1f} px{nm(radius)}")

    for round_ in ROUNDS:
        count = _as_float(setting("ite", round_))
        if vpr and not count:
            continue
        if not vpr and round_ > 1:
            break
        group = f"Search, round {round_}"
        if count:
            add(group, "iterations", _text(count))
        add(group, "cone", f"{_text(setting('cone_range', round_))}° wide, "
            f"{_text(setting('cone_sampling', round_))}° steps")
        add(group, "in-plane", f"{_text(setting('inplane_range', round_))}° wide, "
            f"{_text(setting('inplane_sampling', round_))}° steps")
        add(group, "refinement", f"{_text(setting('refine', round_))} levels, factor "
            f"{_text(setting('refine_factor', round_))}")
        high, low = _as_float(setting("high", round_)), _as_float(setting("low", round_))
        band = f"Fourier pixels {_text(setting('high', round_))} to {_text(setting('low', round_))}"
        if spacing and box and high and low:
            band += f": {box * spacing / high:.0f} to {box * spacing / low:.0f} Å"
        add(group, "band", band)
        modus = _as_float(setting("area_search_modus", round_))
        mode = AREA_SEARCH_MODES.get(int(modus), "?") if modus is not None else "?"
        add(group, "shifts", f"up to {_text(setting('area_search', round_))} px, {mode} "
            f"(modus {_text(setting('area_search_modus', round_))})")
        add(group, "symmetry, search size",
            f"{_text(setting('sym', round_))}, {_text(setting('dim', round_))} px")
        threshold = _as_float(setting("threshold_modus", round_))
        add(group, "thresholding",
            THRESHOLD_MODES.get(int(threshold), "?") if threshold is not None else "?")
        add(group, "CC", f"use_CC {_text(setting('use_CC', round_))}, localnc "
            f"{_text(setting('localnc', round_))}")

    add("Execution", "destination, cores",
        f"{_text(vpr.get('destination') or card.get('destination'))}, "
        f"{_text(vpr.get('how_many_processors') or card.get('how_many_processors'))}")
    for iteration in iterations:
        add("Execution", f"iteration {iteration.number}", _duration(iteration))
    return rows


def _duration(iteration: IterationResult) -> str:
    if not iteration.started or not iteration.finished:
        return "times not recorded"
    try:
        start = datetime.strptime(iteration.started, DYNAMO_TIME)
        end = datetime.strptime(iteration.finished, DYNAMO_TIME)
    except ValueError:
        return f"{iteration.started} to {iteration.finished}"
    minutes = (end - start).total_seconds() / 60
    return f"{iteration.started} to {iteration.finished} ({minutes:.0f} min)"
