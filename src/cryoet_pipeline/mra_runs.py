"""An MRA run in the run-record layout (docs/run_records.md): a plan of nodes.

    <run>/plan.json          what the run was asked for: the setup's record, every
                             iteration's search
    <run>/setup/             node mra-setup: particles.parquet (one row per particle),
                             reference_<k>.mrc (each channel's seed), mask.mrc
    <run>/ite_0001/          node mra-iteration: particles.parquet (one row per particle
                             and channel searched), average_<k>.mrc (each surviving class)
    <run>/ite_0002/ ...

Each node is a run with its own run.json (`cryoet_pipeline.runs`), and starts from what
the node before it wrote: the setup's seeds and starting poses, then each iteration's
averages and poses, the maps as float32 and the poses as float64, as written. So a run
carried on after a stop ends as the unstopped one does. A node whose record is there,
succeeded, with the fingerprint this plan gives it, is not computed again; one there with
another fingerprint is refused: a job submitted again carries its own plan on and no
other.

The computation is `dynamo_mra.mra_iteration`'s, as `dynamo_mra_project.run_mra_project`
runs it into Dynamo's layout; only what is written, and where, differs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.backends.dynamo_search import search_particles
from cryoet_pipeline.dynamo_mra import mra_iteration
from cryoet_pipeline.dynamo_mra_project import MraPlan, _particle
from cryoet_pipeline.models import ArtifactKind
from cryoet_pipeline.particle_tables import DYNAMO_POSES, read_particle_table, write_particle_table
from cryoet_pipeline.runs import (
    RunInput,
    RunRecord,
    Selection,
    canonical_json,
    clear_abandoned,
    code_record,
    completed,
    files_fingerprint,
    fingerprint,
    writing,
)

STEP_SETUP, STEP_ITERATION = "mra-setup", "mra-iteration"
BACKEND = "dynamo-port"
SETUP_VERSION = ITERATION_VERSION = BACKEND_VERSION = 1
"""Declared: raised by hand when a change moves what the step computes."""
PLAN = "plan.json"
PARTICLES = "particles.parquet"
TAG, SHIFTS, ANGLES, TILTS, TOMOGRAM, POSITION, PIXEL = (
    0, slice(3, 6), slice(6, 9), slice(13, 15), 19, slice(23, 26), 35)
"""The columns of the plan's starting table (a Dynamo table) the setup names."""


class Refused(ValueError):
    """A run not written where it was asked to be: that folder holds another plan's
    nodes, or files of another kind."""


def node_folder(run: Path, number: int) -> Path:
    """The setup for 0, iteration `number` otherwise."""

    return run / ("setup" if number == 0 else f"ite_{number:04d}")


def _export(plan: MraPlan) -> Path:
    return plan.particles.parent


def export_fingerprint(export: Path, particles: Path) -> str:
    """What an export says it is: its table and its manifest, or as much of them as it
    has; with neither, as a Dynamo project's particles folder may sit, the names and
    sizes of its particle files."""

    files = [path for path in (export / "initial.tbl", export / "export_manifest.json")
             if path.exists()]
    if files:
        return files_fingerprint(files)
    listing = sorted((path.name, path.stat().st_size) for path in particles.glob("particle_*"))
    if not listing:
        raise ValueError(f"{export}: no initial.tbl, export_manifest.json or particle files "
                         "to know the export by")
    return hashlib.sha256(canonical_json(listing).encode()).hexdigest()


def _setup_identity(plan: MraPlan) -> tuple[dict[str, Any], list[RunInput]]:
    """The setup node's content parameters and inputs.

    A plan the pipeline's setup made is its variables and the files it read; one read
    from a Dynamo project is that project's starting state (seeds, mask, starting
    tables), which Dynamo's setup made from variables the plan does not hold.
    """

    export = _export(plan)
    inputs = [RunInput(slot="export", path=str(export),
                     fingerprint=export_fingerprint(export, plan.particles))]
    if plan.setup is not None:
        files = plan.setup["files"]
        for slot in ("reference_file", "mask_file"):
            if files.get(slot):
                inputs.append(RunInput(slot=slot, path=files[slot],
                                       fingerprint=files_fingerprint([Path(files[slot])])))
        return dict(plan.setup["variables"]), inputs
    project = plan.source
    state = sorted((project / "settings").glob("template_initial_ref_*.em"))
    state += [project / "settings" / "mask.em"]
    state += sorted((project / "results" / "ite_0001" / "starting_values")
                    .glob("starting_table_ref_*_ite_0001.tbl"))
    inputs.append(RunInput(slot="dynamo_project", path=str(project),
                           fingerprint=files_fingerprint(state)))
    return {"from": "the starting state of a Dynamo project"}, inputs


def _iteration_content(plan: MraPlan, number: int) -> dict[str, Any]:
    return {"iteration": number, "search": asdict(plan.settings[number - 1]),
            "mra": plan.mra[number - 1], "fmin": 1}


def _identity(plan: MraPlan, run: Path, number: int,
              previous: RunRecord | None) -> tuple[str, int, dict[str, Any], list[RunInput]]:
    """Step, its version, content parameters and inputs of node `number`."""

    if number == 0:
        content, inputs = _setup_identity(plan)
        return STEP_SETUP, SETUP_VERSION, content, inputs
    assert previous is not None
    inputs = [
        RunInput(slot="previous", path=node_folder(run, number - 1).name,
                 fingerprint=previous.fingerprint),
        RunInput(slot="particles", path=str(plan.particles),
                 fingerprint=export_fingerprint(_export(plan), plan.particles)),
    ]
    return STEP_ITERATION, ITERATION_VERSION, _iteration_content(plan, number), inputs


def plan_record(plan: MraPlan) -> dict[str, Any]:
    """plan.json: what the run was asked for, for people and tools to read."""

    return {
        "made_by": "cryoet_pipeline.mra_runs", "source": str(plan.source),
        "particles": str(plan.particles), "setup": plan.setup,
        "channels": sorted(plan.references), "reference_notes": plan.reference_notes,
        "iterations": [_iteration_content(plan, number)
                       for number in range(1, len(plan.settings) + 1)],
    }


def _write_plan(path: Path, plan: MraPlan) -> None:
    """plan.json, unless the one there asks for the same and more: a run carried on with
    fewer iterations than it was once asked for keeps the longer request."""

    record = json.loads(canonical_json(plan_record(plan)))
    if path.exists():
        before = json.loads(path.read_text())
        asked = record["iterations"]
        if ({key: value for key, value in before.items() if key != "iterations"}
                == {key: value for key, value in record.items() if key != "iterations"}
                and before.get("iterations", [])[:len(asked)] == asked):
            return
    _write_json(path, record)


def _finished(plan: MraPlan, run: Path) -> list[RunRecord]:
    """The records of the nodes finished before, from the setup on as far as they go
    unbroken. A node there of another plan is refused before anything is written."""

    done: list[RunRecord] = []
    for number in range(len(plan.settings) + 1):
        record = completed(node_folder(run, number))
        if record is None:
            break
        step, step_version, content, inputs = _identity(plan, run, number,
                                                        done[-1] if done else None)
        if record.fingerprint != fingerprint(step, step_version, BACKEND, BACKEND_VERSION,
                                             content, inputs):
            raise Refused(f"{node_folder(run, number)} holds a node of another plan; give "
                          "the run a new name")
        done.append(record)
    return done


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(json.loads(canonical_json(value)), indent=1) + "\n")
    os.replace(temporary, path)


def _write_mrc(path: Path, volume: np.ndarray, spacing: float) -> None:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = spacing


def _read_mrc(path: Path) -> np.ndarray:
    with mrcfile.open(path, permissive=True) as handle:
        return np.asarray(handle.data, dtype=np.float32).astype(float)


def _manifest(export: Path) -> dict[int, dict[str, Any]]:
    path = export / "export_manifest.json"
    if not path.exists():
        return {}
    return {int(entry["tag"]): entry
            for entry in json.loads(path.read_text()).get("particles", [])}


def pixel_size(plan: MraPlan) -> float | None:
    """The table's column 36, else the export manifest's, else unknown: a Dynamo table
    may stop at column 35."""

    if plan.table.shape[1] > PIXEL and plan.table[0, PIXEL] > 0:
        return float(plan.table[0, PIXEL])
    manifest = _export(plan) / "export_manifest.json"
    value = json.loads(manifest.read_text()).get("pixel_spacing_angstrom") \
        if manifest.exists() else None
    return float(value) if value else None


def _table_metadata(plan: MraPlan, what: str) -> dict[str, Any]:
    return {"set": what, **DYNAMO_POSES, "pixel_size_angstrom": pixel_size(plan),
            "box_px": int(plan.soft_mask.shape[0]), "export": str(_export(plan))}


def _write_setup(plan: MraPlan, add: Callable[..., Path]) -> None:
    """The seeds, the mask and every particle's row at its starting pose."""

    table, spacing = plan.table, pixel_size(plan) or 0.0
    tags = table[:, TAG].astype(np.int64)
    entries = _manifest(_export(plan))
    groups = ({int(tag): int(channel) for tag, channel in plan.seed_groups}
              if plan.seed_groups is not None else {})

    def score(tag: int) -> float:
        value = (entries.get(tag, {}).get("native_record") or {}).get("score")
        return float(value) if value is not None else float("nan")

    columns: dict[str, Any] = {
        "particle_id": tags,
        "tilt_series_id": [str(entries.get(int(tag), {}).get("run", "")) for tag in tags],
        "tomogram": table[:, TOMOGRAM].astype(np.int64),
        "position_x_px": table[:, POSITION][:, 0], "position_y_px": table[:, POSITION][:, 1],
        "position_z_px": table[:, POSITION][:, 2],
        "tilt_min_deg": table[:, TILTS][:, 0], "tilt_max_deg": table[:, TILTS][:, 1],
        "decoy": np.array([entries.get(int(tag), {}).get("decoy") is True for tag in tags]),
        "score": np.array([score(int(tag)) for tag in tags]),
        "tdrot_deg": table[:, ANGLES][:, 0], "tilt_deg": table[:, ANGLES][:, 1],
        "narot_deg": table[:, ANGLES][:, 2],
        "shift_x_px": table[:, SHIFTS][:, 0], "shift_y_px": table[:, SHIFTS][:, 1],
        "shift_z_px": table[:, SHIFTS][:, 2],
        "seed_channel": np.array([groups.get(int(tag), 0) for tag in tags], dtype=np.int16),
    }
    write_particle_table(add(ArtifactKind.PARTICLE_TABLE, PARTICLES),
                         columns, _table_metadata(plan, "the particles at their starting poses"))
    for channel, reference in sorted(plan.references.items()):
        _write_mrc(add(ArtifactKind.CLASS_REFERENCE, f"reference_{channel}.mrc"),
                   reference, spacing)
    _write_mrc(add(ArtifactKind.ALIGNMENT_MASK, "mask.mrc"), plan.soft_mask, spacing)


@dataclass(frozen=True)
class State:
    """What a node left the next one: each channel's reference and every particle's pose
    in it, in the plan's particle order."""

    references: dict[int, np.ndarray]
    poses: dict[int, tuple[np.ndarray, np.ndarray]]
    """Channel: (angles, shifts), each (particles, 3)."""


def read_state(run: Path, number: int, tags: np.ndarray) -> State:
    """The state node `number` wrote, read back from its files."""

    folder = node_folder(run, number)
    columns, _ = read_particle_table(folder / PARTICLES)
    angles = np.column_stack([columns[name] for name in ("tdrot_deg", "tilt_deg", "narot_deg")])
    shifts = np.column_stack([columns[name] for name in ("shift_x_px", "shift_y_px", "shift_z_px")])
    ids = columns["particle_id"].astype(np.int64)
    if number == 0:
        channels = sorted(int(path.stem.split("_")[1]) for path in folder.glob("reference_*.mrc"))
        order = _order(ids, tags)
        references = {channel: _read_mrc(folder / f"reference_{channel}.mrc")
                      for channel in channels}
        return State(references, {channel: (angles[order], shifts[order])
                                  for channel in channels})
    surviving = sorted(int(path.stem.split("_")[1]) for path in folder.glob("average_*.mrc"))
    poses = {}
    for channel in surviving:
        rows = np.flatnonzero(columns["channel"] == channel)
        order = rows[_order(ids[rows], tags)]
        poses[channel] = (angles[order], shifts[order])
    return State({channel: _read_mrc(folder / f"average_{channel}.mrc") for channel in surviving},
                 poses)


def _order(ids: np.ndarray, tags: np.ndarray) -> np.ndarray:
    """The rows of `ids` in the order of `tags`, refusing a table of other particles."""

    position = {int(tag): row for row, tag in enumerate(ids)}
    if len(position) != len(ids) or sorted(position) != sorted(int(tag) for tag in tags):
        raise ValueError("a node's table holds other particles than the plan's")
    return np.array([position[int(tag)] for tag in tags])


def _write_iteration(plan: MraPlan, number: int, state: State, particles: Sequence[np.ndarray],
                     search: Callable[..., list[Any]], workers: int, add: Callable[..., Path],
                     selections: list[Selection], metrics: dict[str, float]) -> str:
    """One iteration from `state`; returns its line for the log."""

    tags = plan.table[:, TAG].astype(np.int64)
    tilt_ranges = [(float(row[TILTS][0]), float(row[TILTS][1])) for row in plan.table]
    searching = 0.0

    def timed_search(*args: Any, **kwargs: Any) -> list[Any]:
        nonlocal searching
        clock = time.perf_counter()
        try:
            return search(*args, **kwargs)
        finally:
            searching += time.perf_counter() - clock

    clock = time.perf_counter()
    step = mra_iteration(particles, state.references, state.poses, tilt_ranges, plan.soft_mask,
                         plan.settings[number - 1], fmin=1, workers=workers, search=timed_search,
                         assembly="mra" if plan.mra[number - 1] else "project")
    seconds = time.perf_counter() - clock
    rows = [(channel, step.searches[channel]) for channel in step.computed]
    columns = {
        "particle_id": np.concatenate([tags for _ in rows]),
        "channel": np.concatenate([np.full(len(tags), channel, dtype=np.int16)
                                   for channel, _ in rows]),
        **{name: np.concatenate([found.angles[:, axis] for _, found in rows])
           for axis, name in enumerate(("tdrot_deg", "tilt_deg", "narot_deg"))},
        **{name: np.concatenate([found.shifts[:, axis] for _, found in rows])
           for axis, name in enumerate(("shift_x_px", "shift_y_px", "shift_z_px"))},
        "cc": np.concatenate([found.correlations for _, found in rows]),
        "assigned": np.concatenate([step.assignment == channel for channel, _ in rows]),
    }
    write_particle_table(add(ArtifactKind.PARTICLE_TABLE, PARTICLES), columns,
                         _table_metadata(plan, f"each particle's pose in each channel after "
                                               f"iteration {number}, and its class"))
    spacing = pixel_size(plan) or 0.0
    for channel in step.surviving:
        _write_mrc(add(ArtifactKind.CLASS_AVERAGE, f"average_{channel}.mrc"),
                   step.averages[channel].average, spacing)
        selections.append(Selection(
            name=f"class {channel}", kind="particles",
            definition={"column": "channel", "equals": channel, "where": "assigned"},
            snapshot=PARTICLES))
        metrics[f"class_{channel}_particles"] = float(np.sum(step.assignment == channel))
    metrics.update({"seconds": seconds, "search_seconds": searching})
    sizes = ", ".join(f"class {channel} {int(np.sum(step.assignment == channel)):,}"
                      for channel in step.surviving)
    return f"{sizes}; {seconds / 60:.1f} min, {searching / 60:.1f} of it searching"


def run_plan(
    plan: MraPlan,
    run: Path,
    *,
    search: Callable[..., list[Any]] = search_particles,
    execution: Mapping[str, Any] | None = None,
    workers: int = 1,
    log_path: str | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Every node of the plan not done before, into `run`, in order.

    `search` is the kernel, as `dynamo_mra.mra_iteration` takes it; `execution` the
    execution parameters each record keeps (device, dtype, GPUs, ...), which never
    change a fingerprint.
    """

    run = run.resolve()
    if run.is_dir() and not (run / PLAN).is_file() and any(run.iterdir()):
        raise Refused(f"{run} holds files but no {PLAN}, so it is not a run of this layout "
                      "(a Dynamo project, or a run of the port in Dynamo's layout); give the "
                      "run a new name")
    run.mkdir(parents=True, exist_ok=True)
    execution = dict(execution or {})
    commit = code_record().get("commit")
    tags = plan.table[:, TAG].astype(np.int64)
    particles: list[np.ndarray] | None = None
    previous: RunRecord | None = None
    computed = 0
    try:
        _finished(plan, run)
        _write_plan(run / PLAN, plan)
        for number in range(len(plan.settings) + 1):
            folder = node_folder(run, number)
            step, step_version, content, inputs = _identity(plan, run, number, previous)
            expected = fingerprint(step, step_version, BACKEND, BACKEND_VERSION, content, inputs)
            record = completed(folder)
            if record is not None:
                if record.fingerprint != expected:
                    raise Refused(f"{folder} holds a node of another plan; give the run a "
                                  "new name")
                if record.code.get("commit") != commit:
                    log(f"{folder.name}: done before under commit "
                        f"{record.code.get('commit', '?')}, carrying on under {commit or '?'}")
                previous = record
                continue
            for moved in clear_abandoned(folder):
                log(f"{folder.name}: what a stopped attempt left is now {moved}")
            state = read_state(run, number - 1, tags) if number else None
            if number and particles is None:
                particles = [_particle(plan.particles, int(tag)) for tag in tags]
            with writing(folder, step=step, step_version=step_version, backend=BACKEND,
                         backend_version=BACKEND_VERSION, content_params=content, inputs=inputs,
                         execution_params=execution, log_path=log_path) as attempt:
                if number == 0:
                    _write_setup(plan, attempt.add)
                    line = (f"{len(tags):,} particles, channels {sorted(plan.references)}, "
                            f"{len(plan.settings)} iterations planned")
                else:
                    assert state is not None and particles is not None
                    line = _write_iteration(plan, number, state, particles, search, workers,
                                            attempt.add, attempt.selections, attempt.metrics)
            log(f"{folder.name}: {line}")
            previous = completed(folder)
            computed += 1
        if not computed:
            log(f"{run.name}: all {len(plan.settings)} iterations finished before; nothing to "
                "compute")
    finally:
        close = getattr(search, "close", None)
        if callable(close):
            close()


__all__ = ["Refused", "State", "export_fingerprint", "node_folder", "plan_record",
           "read_state", "run_plan"]
