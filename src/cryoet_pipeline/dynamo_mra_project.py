"""A Dynamo MRA run done again with the pipeline's port, written as Dynamo writes a project.

`plan_from_dynamo` takes from a Dynamo MRA project everything that defines its run:
the starting table, the references its channels start from (its seeds, as
`settings/template_initial_ref_*.em`), the soft mask (`settings/mask.em`), the
particles (its `folder_data`), and each iteration's search from that iteration's
own card. `run_mra_project` then runs those iterations with `dynamo_mra`
(reference preparation, every channel's search, assignment by the highest CC,
class averages as the MRA plugin makes them, dead channels dropped) and writes a
project folder laid out as Dynamo 1.1.511 lays out its own:

- `results/ite_N/currently_surviving_references_ite_N.txt`: the channels that
  searched in iteration N, and after the last iteration the ones kept;
- `results/ite_N/averages/refined_table_ref_K_ite_N.tbl`: channel K's pose and CC
  (column 10) for every particle, with column 34 each particle's class and column
  3 marking the class's own members, as the MRA plugin rewrites them;
- `results/ite_N/averages/average_ref_K_ite_N.em`: every surviving class's average;
- `results/ite_0001/starting_values/starting_table_ref_K_ite_0001.tbl`,
  `settings/` (the seeds, the mask, `virtual_project.mat` when scipy is at hand),
  `cards/ite_N/card_ite_ite_N.card` (the iteration's search and its times) and
  `log.txt` with `iteration N ok` after each one.

So the classification report, `compare_dynamo_classes.py` and
`class_orientations.py` read it as they read Dynamo's, and the two runs differ
only in the code that searched, assigned and averaged: Dynamo's, or ours on the
CPU (numpy, float64, the replayed port) or on a torch device.

What the port does not model is refused when the plan is read, iteration by
iteration: a search at a dim other than the box (Dynamo's CPU downsampling,
which the GPU path replaces by searching at the full box), symmetry, CC modes
other than `use_CC 1`, `localnc 1`, thresholding, a shift area other than a sphere.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.backends.dynamo_search import SearchSettings, search_particles
from cryoet_pipeline.dynamo_mra import mra_iteration
from cryoet_pipeline.reports.dynamo_project import (
    DYNAMO_TIME,
    _starting_table,
    read_card,
    read_em,
    read_map,
    write_em,
)

TAG, AVERAGED, CC, CLASS = 0, 2, 9, 33
TABLE_COLUMNS = 35
"""Dynamo's table width up to column 34, the class; a wider table keeps its own."""


@dataclass(frozen=True)
class MraPlan:
    """Everything that defines an MRA run, as a Dynamo project holds it."""

    source: Path
    """The project the plan was read from."""
    particles: Path
    """The folder of particle_NNNNN.mrc or .em files."""
    table: np.ndarray
    """The starting table every channel starts from, one row per particle, by tag."""
    references: dict[int, np.ndarray]
    """Each channel's starting reference, as Dynamo's iteration 1 takes it."""
    reference_notes: list[str]
    """The seeds' own record (seed_kinds.txt lines), copied for the report."""
    soft_mask: np.ndarray
    settings: list[SearchSettings]
    """One per iteration."""
    cards: list[dict[str, str]]
    """Each iteration's card as Dynamo wrote it, for the record."""


_REFUSED = {
    "sym": ("c1",),
    "use_CC": ("1",),
    "localnc": ("1",),
    "threshold_modus": ("0",),
    "cone_flip": ("0",),
    "inplane_flip": ("0",),
}


def _card_settings(card: dict[str, str], box: int, number: int) -> SearchSettings:
    """An iteration's search from its card, refusing what the port does not model."""

    wrong = {key: card[key] for key, allowed in _REFUSED.items()
             if key in card and card[key].strip() not in allowed}
    dim = card.get("dim")
    if dim is not None and int(float(dim)) != box:
        wrong["dim"] = dim
    limits = [float(value) for value in card.get("area_search", "").replace(",", " ")
              .strip("[] ").split()]
    if len(limits) != 3 or len(set(limits)) != 1:
        wrong["area_search"] = card.get("area_search", "")
    if wrong:
        raise ValueError(f"iteration {number}: outside what the port models: {wrong}")
    return SearchSettings(
        cone_aperture=float(card["cone_range"]), cone_sampling=float(card["cone_sampling"]),
        inplane_range=float(card["inplane_range"]),
        inplane_sampling=float(card["inplane_sampling"]),
        refine=int(float(card["refine"])), refine_factor=float(card["refine_factor"]),
        high=float(card["high"]), low=float(card["low"]), shift_limit=limits[0],
        area_search_modus=int(float(card["area_search_modus"])),
    )


def plan_from_dynamo(project: Path, *, iterations: int | None = None) -> MraPlan:
    """The plan of a Dynamo MRA project, from the iterations it has cards for.

    A finished project has a card for every iteration; a running one, for those
    it has begun. `iterations` takes only the first so many.
    """

    project = project.resolve()
    settings = project / "settings"
    references = {int(path.stem.rsplit("_", 1)[1]): read_em(path).astype(float)
                  for path in sorted(settings.glob("template_initial_ref_*.em"))}
    if not references:
        raise ValueError(f"{project}: no settings/template_initial_ref_*.em")
    problems: list[str] = []
    table = _starting_table(project, problems)
    if problems:
        raise ValueError(f"{project}: {'; '.join(problems)}")
    table = table[np.argsort(table[:, TAG], kind="stable")]
    soft_mask = read_em(settings / "mask.em").astype(float)
    box = soft_mask.shape[0]
    cards = []
    number = 1
    while iterations is None or number <= iterations:
        path = project / "cards" / f"ite_{number:04d}" / f"card_ite_ite_{number:04d}.card"
        if not path.exists():
            break
        cards.append(read_card(path))
        number += 1
    if not cards:
        raise ValueError(f"{project}: no iteration card, cards/ite_0001/card_ite_ite_0001.card")
    first = cards[0]
    particles = project.parent / first.get("folder_data", "particles")
    seeds = project.parent / first.get("file_template_initial", "")
    if seeds.is_file():
        seeds = seeds.parent
    notes_path = seeds / "seed_kinds.txt"
    notes = notes_path.read_text().splitlines() if notes_path.exists() else []
    return MraPlan(
        source=project, particles=particles, table=table, references=references,
        reference_notes=notes, soft_mask=soft_mask,
        settings=[_card_settings(card, box, index) for index, card in enumerate(cards, 1)],
        cards=cards,
    )


def _particle(folder: Path, tag: int) -> np.ndarray:
    for extension in ("mrc", "em"):
        path = folder / f"particle_{tag:05d}.{extension}"
        if path.exists():
            return read_map(path).astype(float)
    raise FileNotFoundError(f"no particle {tag} in {folder}")


def _write_table(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(f"{value:.12g}" for value in row) for row in rows)
                    + "\n")


def _card_text(values: dict[str, Any]) -> str:
    return "".join(f"{name}={value};\n" for name, value in values.items())


def _search_fields(settings: SearchSettings, box: int) -> dict[str, Any]:
    return {
        "cone_range": f"{settings.cone_aperture:g}", "cone_sampling": f"{settings.cone_sampling:g}",
        "inplane_range": f"{settings.inplane_range:g}",
        "inplane_sampling": f"{settings.inplane_sampling:g}",
        "refine": settings.refine, "refine_factor": f"{settings.refine_factor:g}",
        "high": f"{settings.high:g}", "low": f"{settings.low:g}", "dim": box,
        "area_search": " ".join([f"{settings.shift_limit:g}"] * 3),
        "area_search_modus": settings.area_search_modus, "sym": "c1", "use_CC": 1,
        "localnc": 1, "threshold_modus": 0,
    }


def _write_vpr(path: Path, plan: MraPlan, name: str, destination: str, box: int,
               spacing: float | None) -> bool:
    """`settings/virtual_project.mat` with every round's search, when scipy can write it."""

    try:
        from scipy import io as scipy_io  # type: ignore[import-untyped]
    except ImportError:
        return False
    nref = len(plan.references)
    card: dict[str, Any] = {
        "name_project": name, "folder_data": plan.particles.name,
        "file_template_initial": f"{name}_seeds", "file_mask": f"{name}/settings/mask.em",
        "file_mask_classification": f"{name}/settings/mask.em",
        "file_mask_smoothing": f"{name}/settings/mask.em", "feature_data_sidelength": box,
        "feature_extension_data": "mrc" if any(plan.particles.glob("*.mrc")) else "em",
        "destination": destination, "how_many_processors": 1, "apix": spacing or 1,
    }
    for round_, settings in enumerate(plan.settings, 1):
        fields = _search_fields(settings, box)
        fields["area_search"] = np.array([settings.shift_limit] * 3)
        card.update({f"{key}_r{round_}": value for key, value in fields.items()})
        card.update({f"ite_r{round_}": 1, f"nref_r{round_}": nref,
                     f"mra_r{round_}": int(nref > 1)})
    for round_ in range(len(plan.settings) + 1, 9):
        card[f"ite_r{round_}"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy_io.savemat(path, {"card": card})
    return True


def run_mra_project(
    plan: MraPlan,
    out: Path,
    *,
    search: Callable[..., list] = search_particles,
    destination: str = "cryoet-pipeline, numpy",
    workers: int = 1,
    spacing: float | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Run the plan's iterations with the port and write them as a Dynamo project at `out`.

    `search` is the kernel, as `dynamo_mra.mra_iteration` takes it; `destination`
    says in the project's cards which one ran. Each iteration is written when it
    finishes, so a report can read a run that is still going.
    """

    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"{out} exists; give the run a new name")
    name, base = out.name, out.parent
    box = plan.soft_mask.shape[0]
    tags = plan.table[:, TAG].astype(int)
    log(f"{name}: {len(tags):,} particles from {plan.particles}, channels "
        f"{sorted(plan.references)}, {len(plan.settings)} iterations, {destination}")
    particles = [_particle(plan.particles, int(tag)) for tag in tags]
    tilt_ranges = [(float(row[13]), float(row[14])) for row in plan.table]

    seeds = base / f"{name}_seeds"
    seeds.mkdir(parents=True)
    for channel, reference in plan.references.items():
        write_em(seeds / f"template_initial_ref_{channel:03d}.em", reference)
        write_em(out / "settings" / f"template_initial_ref_{channel:03d}.em", reference)
    (seeds / "seed_kinds.txt").write_text("".join(f"{line}\n" for line in plan.reference_notes))
    write_em(out / "settings" / "mask.em", plan.soft_mask)
    (out / "settings" / "mirrors.json").write_text(json.dumps(
        {"source": str(plan.source), "iterations": len(plan.settings)}, indent=1) + "\n")
    wrote_vpr = _write_vpr(out / "settings" / "virtual_project.mat", plan, name, destination,
                           box, spacing)
    starting = out / "results" / "ite_0001" / "starting_values"
    for channel in plan.references:
        rows = plan.table.copy()
        if rows.shape[1] > CLASS:
            rows[:, CLASS] = channel
        _write_table(starting / f"starting_table_ref_{channel:03d}_ite_0001.tbl", rows)
    if not wrote_vpr:
        log("no scipy: settings/virtual_project.mat not written; the report shows round 1 only")

    try:
        _iterate(plan, out, particles, tilt_ranges, search, destination, workers, log)
    finally:
        close = getattr(search, "close", None)
        if callable(close):
            close()


def _iterate(plan: MraPlan, out: Path, particles: list[np.ndarray],
             tilt_ranges: list[tuple[float, float]], search: Callable[..., list],
             destination: str, workers: int, log: Callable[[str], None]) -> None:
    name = out.name
    box = plan.soft_mask.shape[0]
    references = dict(plan.references)
    tables = {channel: plan.table.copy() for channel in references}
    for number, settings in enumerate(plan.settings, 1):
        here = out / "results" / f"ite_{number:04d}"
        here.mkdir(parents=True, exist_ok=True)
        (here / f"currently_surviving_references_ite_{number:04d}.txt").write_text(
            " ".join(str(channel) for channel in sorted(references)) + "\n")
        started = datetime.now()
        clock = time.perf_counter()
        starts = {channel: (tables[channel][:, 6:9], tables[channel][:, 3:6])
                  for channel in references}
        step = mra_iteration(particles, references, starts, tilt_ranges, plan.soft_mask,
                             settings, fmin=1, workers=workers, search=search)
        seconds = time.perf_counter() - clock
        finished = datetime.now()
        averages = here / "averages"
        for channel in step.computed:
            found = step.searches[channel]
            rows = tables[channel].copy()
            if rows.shape[1] < TABLE_COLUMNS:
                rows = np.pad(rows, ((0, 0), (0, TABLE_COLUMNS - rows.shape[1])))
            rows[:, 3:6], rows[:, 6:9], rows[:, CC] = found.shifts, found.angles, found.correlations
            rows[:, AVERAGED] = (step.assignment == channel).astype(float)
            rows[:, CLASS] = step.assignment
            tables[channel] = rows
            _write_table(averages / f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl",
                         rows)
        for channel in step.surviving:
            write_em(averages / f"average_ref_{channel:03d}_ite_{number:04d}.em",
                     step.averages[channel].average)
        following = out / "results" / f"ite_{number + 1:04d}"
        following.mkdir(parents=True, exist_ok=True)
        (following / f"currently_surviving_references_ite_{number + 1:04d}.txt").write_text(
            " ".join(str(channel) for channel in step.surviving) + "\n")
        card = {
            "name_project": name, "database_ite": number, "nref": len(step.computed),
            "mra": int(len(plan.references) > 1), **_search_fields(settings, box),
            "folder_data": plan.particles.name, "file_template_initial": f"{name}_seeds",
            "feature_data_sidelength": box, "destination": destination,
            "time_start_computation": started.strftime(DYNAMO_TIME),
            "time_end_iteration": finished.strftime(DYNAMO_TIME),
        }
        card_path = out / "cards" / f"ite_{number:04d}" / f"card_ite_ite_{number:04d}.card"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(_card_text(card))
        with (out / "log.txt").open("a") as handle:
            handle.write(f"iteration {number} ok\n")
        sizes = ", ".join(f"class {channel} {int(np.sum(step.assignment == channel)):,}"
                          for channel in step.surviving)
        log(f"iteration {number}: {sizes}; {seconds / 60:.1f} min")
        references = {channel: step.averages[channel].average for channel in step.surviving}
        tables = {channel: tables[channel] for channel in step.surviving}


def torch_search(device: str, dtype: str, chunk: int, devices: int = 1) -> Callable[..., list]:
    """The torch kernel bound to a device, called as `search_particles` is.

    With `devices` above 1 the particles are dealt among that many devices of the
    kind, cuda:0, cuda:1, ... (or as many processes on one CPU or MPS device), a
    process each (`dynamo_search_torch.MultiDeviceSearch`); the run closes them.
    """

    import torch

    from cryoet_pipeline.backends.dynamo_search_torch import (
        MultiDeviceSearch,
        search_particles_torch,
    )

    if devices > 1:
        names = ([f"{device}:{index}" for index in range(devices)] if device == "cuda"
                 else [device] * devices)
        return MultiDeviceSearch(names, dtype=dtype, chunk=chunk)
    kind = {"float32": torch.float32, "float64": torch.float64}[dtype]

    def search(particles: Sequence[np.ndarray], reference: np.ndarray, mask: np.ndarray,
               starts: Sequence[tuple[np.ndarray, np.ndarray]],
               tilt_ranges: Sequence[tuple[float, float]], settings: SearchSettings, *,
               workers: int = 1) -> list[tuple[np.ndarray, np.ndarray, float]]:
        return search_particles_torch(particles, reference, mask, starts, tilt_ranges, settings,
                                      device=device, dtype=kind, chunk=chunk)

    return search


__all__ = ["MraPlan", "plan_from_dynamo", "run_mra_project", "torch_search"]
