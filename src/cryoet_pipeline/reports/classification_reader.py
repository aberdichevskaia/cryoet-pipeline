"""One reader for a classification in either layout it is written in (docs/run_records.md).

    read_classification(<folder>)

reads a run of the run-record layout (`cryoet_pipeline.mra_runs`: plan.json, setup/,
ite_NNNN/) when the folder holds a plan.json, and a Dynamo project otherwise: Dynamo's
own, or a run of the port written in Dynamo's layout, as they were until 2026-11-05
(`reports.dynamo_project`). Both come out as the `ClassificationRun` the report, the
comparisons of two runs and the consensus halves read. Reading changes nothing.

A run of the run-record layout is read from its records. An iteration counts once its
run.json says it succeeded and names as its input the record of the node before it.
Its classes are the selections that record lists ("class 2": the rows of its particle
table whose channel is 2 and that are assigned), its search the record's content
parameters, its times, host and execution the record's. plan.json says what was asked
for: the setup's variables and how many iterations.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.mra_runs import PARTICLES, PLAN, STEP_SETUP, node_folder
from cryoet_pipeline.particle_tables import read_particle_table, select_rows
from cryoet_pipeline.reports.classification import (
    DEFAULT_DECOY_KIND,
    ClassificationRun,
    ClassResult,
    IterationResult,
    Parameter,
    Pose,
    PoseTable,
    Volume,
)
from cryoet_pipeline.reports.dynamo_project import (
    Pick,
    _area_mode,
    _cached,
    _particle_locations,
    _particle_path,
    _read_manifest,
    _relative,
    _sphere_radius,
    _template,
    aligned_volume,
    average_at_poses,
    manifest_identity,
    particle_locations,
    read_dynamo_classification,
    read_map,
    seed_notes,
)
from cryoet_pipeline.runs import ATTEMPTS, RunRecord, completed, read_record

ANGLE_COLUMNS = ("tdrot_deg", "tilt_deg", "narot_deg")
SHIFT_COLUMNS = ("shift_x_px", "shift_y_px", "shift_z_px")


def read_classification(
    folder: Path,
    *,
    manifest: Path | None = None,
    pixel_size_angstrom: float | None = None,
    initial_average: Path | None = None,
    compute_initial_average: bool = True,
    template: Path | None = None,
    workers: int = 1,
) -> ClassificationRun:
    """A run of the run-record layout when `folder` holds a plan.json, else a Dynamo
    project; the options as `read_dynamo_classification` takes them."""

    read = read_mra_run if (Path(folder) / PLAN).is_file() else read_dynamo_classification
    return read(folder, manifest=manifest, pixel_size_angstrom=pixel_size_angstrom,
                initial_average=initial_average,
                compute_initial_average=compute_initial_average, template=template,
                workers=workers)


def read_mra_run(
    run: Path,
    *,
    manifest: Path | None = None,
    pixel_size_angstrom: float | None = None,
    initial_average: Path | None = None,
    compute_initial_average: bool = True,
    template: Path | None = None,
    workers: int = 1,
) -> ClassificationRun:
    """Everything the classification report shows about one run of `mra_runs.run_plan`.

    The run sits in its export folder (<export>/<NAME>): the export manifest is the one
    beside it unless `manifest` names another, and the particles are the plan's folder,
    or the one of that name beside the run when the run was moved away from it. The
    other options are `read_dynamo_classification`'s.
    """

    run = Path(run).resolve()
    if not (run / PLAN).is_file():
        raise ValueError(f"{run}: no {PLAN}, so not a run of the run-record layout")
    plan = json.loads((run / PLAN).read_text())
    base = run.parent
    problems: list[str] = []
    setup_folder = node_folder(run, 0)
    setup = completed(setup_folder)
    if setup is None:
        raise ValueError(f"{run}: its setup has not finished (no setup/run.json that "
                         "succeeded), so there is nothing to read yet")
    starting, metadata = read_particle_table(setup_folder / PARTICLES)
    starting = _by_tag(starting)
    tags = starting["particle_id"].astype(int)
    box = int(metadata.get("box_px") or 0)

    load, particles = _particle_loader(plan["particles"], base, starting, metadata, problems)

    manifest_path = manifest or base / "export_manifest.json"
    manifest_spacing, decoy_variant, decoy_kind = None, None, DEFAULT_DECOY_KIND
    if manifest_path.exists():
        _, _, manifest_spacing, _, _, decoy_variant, decoy_kind = _read_manifest(manifest_path)
    elif manifest is not None:
        problems.append(f"manifest {manifest} is missing")
    if isinstance(plan["particles"], dict):
        decoy_variant = plan["particles"].get("decoy_label")
    if metadata.get("decoy_kind"):
        decoy_kind = str(metadata["decoy_kind"])
    table_spacing = metadata.get("pixel_size_angstrom")
    if pixel_size_angstrom:
        spacing, spacing_source = float(pixel_size_angstrom), "given to the report"
    elif table_spacing:
        spacing, spacing_source = float(table_spacing), "the run's particle table"
    elif manifest_spacing:
        spacing, spacing_source = manifest_spacing, manifest_path.name
    else:
        spacing, spacing_source = None, ""

    # Each particle's tilt-series, picking score and decoy mark, as the setup copied them
    # from the export manifest into the run's own table.
    series = {int(tag): str(name) for tag, name in zip(tags, starting["tilt_series_id"],
                                                       strict=True) if name}
    scores = {int(tag): float(score) for tag, score in zip(tags, starting["score"], strict=True)
              if math.isfinite(float(score))}
    decoys = frozenset(int(tag) for tag, decoy in zip(tags, starting["decoy"], strict=True)
                       if decoy)

    channels = [int(channel) for channel in plan["channels"]]
    references = {}
    for channel in channels:
        path = setup_folder / f"reference_{channel}.mrc"
        if path.exists():
            references[channel] = Volume(_relative(path, base), partial(read_map, path))
        else:
            problems.append(f"channel {channel}: no starting reference found")
    notes = seed_notes([str(line) for line in plan.get("reference_notes") or []])

    iterations, records, tables = _iterations(run, tags, setup, box, spacing, problems)

    index: dict[Pose, tuple[PoseTable, dict[int, int]]] = {}

    def poses(pose: Pose) -> PoseTable:
        if pose is None:
            return PoseTable(tags, _stack(starting, ANGLE_COLUMNS), _stack(starting, SHIFT_COLUMNS),
                             np.full(len(tags), np.nan))
        number, channel = pose
        if number not in tables:
            raise KeyError(f"iteration {number} has not finished")
        columns = tables[number]
        rows = np.flatnonzero(columns["channel"] == channel)
        if not len(rows):
            raise KeyError(f"iteration {number} did not search channel {channel}")
        rows = rows[np.argsort(columns["particle_id"][rows], kind="stable")]
        return PoseTable(columns["particle_id"][rows].astype(int),
                         _stack(columns, ANGLE_COLUMNS)[rows], _stack(columns, SHIFT_COLUMNS)[rows],
                         columns["cc"][rows].astype(float))

    def aligned(tag: int, pose: Pose) -> np.ndarray:
        if pose not in index:
            table = poses(pose)
            index[pose] = (table, {int(one): row for row, one in enumerate(table.tags)})
        table, rows = index[pose]
        row = rows[int(tag)]
        if load is None:
            raise FileNotFoundError(f"the particles of {run} are not at hand")
        return aligned_volume(load(int(tag)), table.angles[row], table.shifts[row])

    template_volume, template_note = (_template(template, box, spacing, problems)
                                      if template is not None else (None, None))
    mask_path = setup_folder / "mask.mrc"
    identity = {int(tag): (str(series), str(stack), int(position)) for tag, series, stack, position
                in zip(tags, starting.get("tilt_series_id", []), starting.get("stack_id", []),
                       starting.get("stack_index", []), strict=False) if stack}
    if identity and metadata.get("project"):
        picks: list[Pick] = [
            (int(tag), str(series), str(stack), (float(x), float(y), float(z)))
            for tag, series, stack, x, y, z, decoy in zip(
                tags, starting["tilt_series_id"], starting["stack_id"],
                starting["position_x_px"], starting["position_y_px"], starting["position_z_px"],
                starting["decoy"], strict=True)
            if stack and not decoy]
        locations = particle_locations(metadata["project"], picks, mask_path, spacing, box,
                                       problems)
    elif manifest_path.exists():
        locations = _particle_locations(manifest_path, mask_path, spacing, box, problems)
    else:
        locations = None
    if not identity and manifest_path.exists():
        identity = manifest_identity(manifest_path)
    if initial_average is not None:
        average: Volume | None = Volume(str(initial_average), partial(read_map, initial_average))
    elif compute_initial_average and load is not None and mask_path.exists():
        average = Volume("computed for this report",
                         _cached(partial(_starting_average, load, starting, mask_path,
                                         workers=workers)))
    else:
        average = None

    swapping = [bool(item.get("mra")) for item in plan.get("iterations", [])]
    parameters = _parameters(
        plan=plan, base=base, starting=starting, box=box, spacing=spacing,
        spacing_source=spacing_source, series=series, particles=particles, channels=channels,
        references=references, notes=notes, records=records, decoys=decoys,
        decoy_variant=decoy_variant, decoy_kind=decoy_kind, template=template_note,
        mask_path=mask_path,
    )
    return ClassificationRun(
        name=run.name,
        source=run,
        software=("MRA" if any(swapping) else "Refinement") + " run of the pipeline's Dynamo port",
        box_px=box,
        pixel_size_angstrom=spacing,
        tags=tags,
        references=references,
        reference_notes=notes,
        parameters=parameters,
        iterations=iterations,
        aligned_particle=aligned,
        initial_average=average,
        planned_iterations=len(swapping) or None,
        series=series,
        scores=scores,
        score_label="picking score (LCCmax, from the export manifest)",
        decoys=decoys,
        decoy_kind=decoy_kind,
        template=template_volume,
        problems=problems,
        locations=locations,
        poses=poses,
        mask=Volume(_relative(mask_path, base), partial(read_map, mask_path))
        if mask_path.exists() else None,
        identity=identity,
    )


def _particle_loader(
    named: Any, base: Path, starting: Mapping[str, np.ndarray], metadata: Mapping[str, Any],
    problems: list[str],
) -> tuple[Callable[[int], np.ndarray] | None, str]:
    """How the run's particles are read, and their source in words: from the particle
    folder the plan names (or the one of that name beside the run, when the run was
    moved away from it), or from the particle stacks the setup's table names."""

    if isinstance(named, str):
        folder = Path(named)
        if not folder.is_dir() and (base / folder.name).is_dir():
            folder = base / folder.name
        if not folder.is_dir():
            problems.append(f"particle folder {folder} is missing: no galleries and no "
                            "computed initial average")
            return None, _relative(folder, base)
        return (lambda tag: read_map(_particle_path(folder, int(tag), "mrc"))), \
            _relative(folder, base)
    note = (f"the particle stacks of {named.get('project')}, label {named.get('label')}"
            + (f", decoys {named['decoy_label']}" if named.get("decoy_label") else ""))
    stacks = {str(stack): Path(path) for stack, path in (metadata.get("stacks") or {}).items()}
    missing = sorted(str(path) for path in stacks.values() if not path.exists())
    if missing or not stacks or "stack_id" not in starting:
        problems.append(f"particle stacks missing ({', '.join(missing) or 'none named'}): no "
                        "galleries and no computed initial average")
        return None, note
    where = {int(tag): (str(stack), int(position)) for tag, stack, position in zip(
        starting["particle_id"], starting["stack_id"], starting["stack_index"], strict=True)}
    opened: dict[str, Any] = {}

    def load(tag: int) -> np.ndarray:
        import zarr

        stack, position = where[int(tag)]
        if stack not in opened:
            opened[stack] = zarr.open_array(stacks[stack], mode="r")
        return np.asarray(opened[stack][position], dtype=np.float32)

    return load, note


def _by_tag(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(columns["particle_id"], kind="stable")
    return {name: np.asarray(values)[order] for name, values in columns.items()}


def _stack(columns: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray:
    return np.column_stack([np.asarray(columns[name], dtype=float) for name in names])


def _iterations(
    run: Path, tags: np.ndarray, setup: RunRecord, box: int, spacing: float | None,
    problems: list[str],
) -> tuple[list[IterationResult], list[RunRecord], dict[int, dict[str, np.ndarray]]]:
    """The finished iterations, every finished node's record (the setup's first), and
    each finished iteration's particle table."""

    iterations: list[IterationResult] = []
    records = [setup]
    tables: dict[int, dict[str, np.ndarray]] = {}
    number = 1
    while True:
        folder = node_folder(run, number)
        record = completed(folder)
        if record is None:
            if folder.exists() or any(run.glob(f"{folder.name}.tmp-*")):
                problems.append(f"iteration {number} has not finished (no run.json that "
                                "succeeded); the report stops before it")
            problems.extend(_failed_attempts(run, number))
            break
        previous = {item.slot: item.fingerprint for item in record.inputs}.get("previous")
        if previous != records[-1].fingerprint:
            before = "setup" if number == 1 else f"iteration {number - 1}"
            problems.append(f"iteration {number} was computed from another {before} than the "
                            "one in the run; the report stops before it")
            break
        columns, _ = read_particle_table(folder / PARTICLES)
        searched = tuple(sorted(int(channel) for channel in np.unique(columns["channel"])))
        classes: dict[int, ClassResult] = {}
        for selection in record.selections:
            definition = selection.definition
            try:
                if definition.get("column") != "channel":
                    raise ValueError("it does not name a channel")
                rows = select_rows(columns, definition)
            except ValueError as error:
                problems.append(f"iteration {number}: {selection.name} is not read as a class: "
                                f"{error}")
                continue
            channel = int(definition["equals"])
            path = folder / f"average_{channel}.mrc"
            classes[channel] = ClassResult(
                channel,
                np.sort(columns["particle_id"][rows].astype(int)),
                Volume(_relative(path, run.parent), partial(read_map, path))
                if path.exists() else None,
                float(np.median(columns["cc"][rows])) if rows.any() else None,
            )
            if not path.exists():
                problems.append(f"iteration {number}: class {channel} has no average")
        claimed = Counter(int(tag) for result in classes.values() for tag in result.members)
        unclaimed = len(set(tags.tolist()) - set(claimed))
        if unclaimed:
            problems.append(f"iteration {number}: {unclaimed} particles of the setup are in no "
                            "class")
        twice = sum(1 for count in claimed.values() if count > 1)
        if twice:
            problems.append(f"iteration {number}: {twice} particles are in more than one class")
        iterations.append(IterationResult(
            number, searched, dict(sorted(classes.items())), record.started_at, record.ended_at,
            _summary(record.content_params.get("search") or {}, record, box, spacing),
        ))
        records.append(record)
        tables[number] = columns
        number += 1
    return iterations, records, tables


def _failed_attempts(run: Path, number: int) -> list[str]:
    """Why iteration `number` (the first not finished) failed, when an attempt said so."""

    lines = []
    for path in sorted((run / ATTEMPTS).glob("*/run.json")):
        record = read_record(path.parent)
        if (record is None or record.status != "failed" or record.step == STEP_SETUP
                or record.content_params.get("iteration") != number):
            continue
        lines.append(f"iteration {number}: an attempt failed ({record.ended_at}, "
                     f"{record.host or '?'}, job {record.slurm_job_id or '?'}): {record.error}")
    return lines


def _minutes(record: RunRecord) -> float | None:
    try:
        start = datetime.fromisoformat(record.started_at)
        end = datetime.fromisoformat(record.ended_at)
    except ValueError:
        return None
    return (end - start).total_seconds() / 60


def _summary(search: Mapping[str, Any], record: RunRecord, box: int,
             spacing: float | None) -> str | None:
    """What an iteration searched with, from its own record, and how long it took."""

    low = search.get("low")
    if not low:
        return None
    band = f"low-pass {box * spacing / low:.0f} Å" if spacing and box else f"low-pass px {low:g}"
    summary = f"{band}, cone {search['cone_aperture']:g}° in {search['cone_sampling']:g}° steps"
    minutes = _minutes(record)
    return summary + (f" · {minutes:.0f} min" if minutes is not None else "")


def _starting_average(load: Callable[[int], np.ndarray], starting: Mapping[str, np.ndarray],
                      mask_path: Path, *, workers: int = 1) -> np.ndarray:
    tags = starting["particle_id"].astype(int)
    return average_at_poses(
        (load(int(tag)) for tag in tags),
        _stack(starting, ANGLE_COLUMNS), _stack(starting, SHIFT_COLUMNS),
        [(float(low), float(high)) for low, high in zip(starting["tilt_min_deg"],
                                                         starting["tilt_max_deg"], strict=True)],
        read_map(mask_path), workers=workers,
    )


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return " ".join(_value(item) for item in value)
    return str(value)


def _search_line(search: Mapping[str, Any], box: int, spacing: float | None) -> str:
    """One search on one line: band, cone, in-plane, refinement, shifts, box."""

    high, low = float(search["high"]), float(search["low"])
    pixels = f"px {high:g}–{low:g}"
    band = (f"band {box * spacing / high:.0f}–{box * spacing / low:.0f} Å ({pixels})"
            if spacing and box else f"band {pixels}")
    line = (f"{band} · cone {search['cone_aperture']:g}°, step {search['cone_sampling']:g}° · "
            f"in-plane {search['inplane_range']:g}°, step {search['inplane_sampling']:g}° · "
            f"refine {search['refine']}×{search['refine_factor']:g} · shifts up to "
            f"{search['shift_limit']:g} px, {_area_mode(search['area_search_modus'])}")
    if search.get("dim"):
        line += f" · searched at {search['dim']} px"
    return line


def _spans(numbers: Sequence[int]) -> str:
    """1, 2, 3, 5 as "iterations 1–3, 5"."""

    spans: list[list[int]] = []
    for number in numbers:
        if spans and number == spans[-1][-1] + 1:
            spans[-1].append(number)
        else:
            spans.append([number])
    text = ", ".join(f"{span[0]}–{span[-1]}" if len(span) > 1 else str(span[0])
                     for span in spans)
    return ("iterations " if len(numbers) > 1 else "iteration ") + text


def _parameters(
    *, plan: Mapping[str, Any], base: Path, starting: Mapping[str, np.ndarray], box: int,
    spacing: float | None, spacing_source: str, series: Mapping[int, str], particles: str,
    channels: Sequence[int], references: Mapping[int, Volume], notes: Mapping[int, str],
    records: Sequence[RunRecord], decoys: frozenset[int], decoy_variant: str | None,
    decoy_kind: str, template: str | None, mask_path: Path,
) -> list[Parameter]:
    rows: list[Parameter] = []

    def add(group: str, name: str, value: str) -> None:
        rows.append(Parameter(group, name, value))

    def nm(px: float) -> str:
        return f" ({px * spacing / 10:.1f} nm)" if spacing else ""

    tags = starting["particle_id"].astype(int)
    add("Data", "particles", f"{len(tags):,}")
    add("Data", "box", f"{box} px" + nm(box))
    add("Data", "pixel size",
        f"{spacing:.3f} Å, from {spacing_source}" if spacing else "not recorded")
    add("Data", "particles from", particles)
    if template:
        add("Data", "template", template)
    if decoys:
        add("Data", "decoys", f"{len(decoys):,} {decoy_kind}"
            + (f" ({decoy_variant})" if decoy_variant else "") + ", counted with the series")
    counts: dict[str, int] = {}
    tilts: dict[str, tuple[float, float]] = {}
    for row, tag in enumerate(tags):
        name = series.get(int(tag)) or f"tomogram {int(starting['tomogram'][row])}"
        counts[name] = counts.get(name, 0) + 1
        tilts.setdefault(name, (float(starting["tilt_min_deg"][row]),
                                float(starting["tilt_max_deg"][row])))
    for name in sorted(counts):
        lo, hi = tilts[name]
        add("Data", name, f"{counts[name]:,} particles, tilts {lo:+.0f}° to {hi:+.0f}°")

    setup = plan.get("setup")
    if setup:
        add("Setup", "made by", str(setup.get("made_by", "?")))
        for name, value in setup.get("variables", {}).items():
            if value is not None:
                add("Setup", name.replace("_", " "), _value(value))
        for slot, value in setup.get("files", {}).items():
            if value and slot != "export":
                add("Setup", slot.replace("_", " "), str(value))
    else:
        add("Setup", "from", f"the starting state of the Dynamo project {plan.get('source')}")

    asked = list(plan.get("iterations", []))
    add("References", "channels", str(len(channels)))
    swapping = [int(item["iteration"]) for item in asked if item.get("mra")]
    add("References", "particles swap between channels",
        "yes" if asked and len(swapping) == len(asked)
        else "no" if not swapping else f"in {_spans(swapping)}")
    for channel in channels:
        source = references[channel].source if channel in references else "missing"
        add("References", f"channel {channel}", notes.get(channel) or source)

    if mask_path.exists():
        note = ((setup or {}).get("derived") or {}).get("mask") or _relative(mask_path, base)
        radius = _sphere_radius(read_map(mask_path))
        add("Masks", "alignment", f"{note}: its voxels above 0.5 fill as much as a sphere of "
            f"radius {radius:.1f} px{nm(radius)}")
    else:
        add("Masks", "alignment", "missing")

    # Consecutive iterations that search alike share a row.
    groups: list[tuple[list[int], dict[str, Any]]] = []
    for item in asked:
        if groups and groups[-1][1] == item["search"]:
            groups[-1][0].append(int(item["iteration"]))
        else:
            groups.append(([int(item["iteration"])], item["search"]))
    for numbers, search in groups:
        add("Search", _spans(numbers), _search_line(search, box, spacing))

    commits = {record.code.get("commit", "?") + (" with uncommitted changes"
                                                 if record.code.get("dirty") else "")
               for record in records}
    if len(commits) == 1:
        add("Execution", "code", f"commit {commits.pop()}")
    for record in records:
        name = ("setup" if record.step == STEP_SETUP
                else f"iteration {record.content_params.get('iteration')}")
        minutes = _minutes(record)
        parts = [f"{minutes:.0f} min" if minutes is not None else "time not recorded"]
        if record.execution_params:
            parts.append(", ".join(f"{key} {_value(value)}"
                                   for key, value in record.execution_params.items()))
        parts.append(f"{record.host or '?'}" + (f", job {record.slurm_job_id}"
                                                 if record.slurm_job_id else ""))
        if len(commits) > 1:
            parts.append(f"commit {record.code.get('commit', '?')}")
        add("Execution", name, " · ".join(parts))
    return rows


__all__ = ["read_classification", "read_mra_run"]
