"""The MRA setup without Dynamo: a run's plan made from an export folder, or from the
particle stacks of a project's registry.

`plan_from_export` makes, from an export folder of scripts/export_dynamo.py
(particles/, initial.tbl), and `plan_from_registry` from the particle stacks that
export would copy (`particle_sources.registry_particles`, the same conversion and the
same tags in the same order), the plan scripts/dynamo_mra_setup.m made a Dynamo project
for and `dynamo_mra_project.plan_from_dynamo` read back from it: every channel's
seed, the soft mask, the starting table and every iteration's search, from the same
variables (SEEDS, SCHEDULE, ... of scripts/mra_port_run.sbatch). `run_mra_project`
runs it as it runs a Dynamo project's plan.

What it keeps of Dynamo's setup, and what not:

- The random draws are Dynamo's. MATLAB's `rand` after `rng(20261016)` is numpy's
  `RandomState(20261016)` draw for draw, and `randperm(n)` is the order that sorts n
  of them (measured in Dynamo 1.1.511's runtime on 2026-11-02, at n = 12 and 2,384,
  with the blob orientations drawn after it equal to 17 digits): a 'subset' seed
  averages the particles Dynamo's setup took, and a 'blob' seed turns them as it did.
  A 'noise' seed is new noise: MATLAB's `randn` is not numpy's, so it differs from a
  Dynamo-made project's, and is the same in every plan of the same particle count
  and box.
- The seeds are averaged as `dynamo_average` averages (`average_class`); the mask is
  `dynamo_sphere`'s (`soft_sphere`).
- The starting table keeps the export's digits. Dynamo wrote its project's tables
  rounded to 5 significant digits (`%.5g`), so the run of a Dynamo-made project
  started from angles rounded by up to 0.005 deg; a comparison with one rounds on
  its side (scripts/compare_mra_setup.py).
- Each round's search is the setup's arithmetic, its angles kept to 1e-6 deg: three
  4.9 deg steps are 14.7 deg, as Dynamo's card held them, not 14.700000000000001.

Symmetry, the axis offset that moved particles onto Dynamo's symmetry axis, and
Dynamo's GPU options are left out: the port does not model them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.backends.dynamo_search import SearchSettings, average_class
from cryoet_pipeline.dynamo_mra_project import (
    MraPlan,
    _particle,
    _search_fields,
    dim_is_ported,
)
from cryoet_pipeline.mra_runs import HALVES, PLAN, START, STEP_SPLIT
from cryoet_pipeline.particle_sources import (
    RegistryParticles,
    SelectionInput,
    registry_particles,
)
from cryoet_pipeline.particle_tables import read_particle_table
from cryoet_pipeline.reports.dynamo_project import read_map, read_table
from cryoet_pipeline.runs import completed

SHIFT_LIMIT = 3.0
"""Pixels of the particles' box: how far a particle may move from its box centre, in
every round (Dynamo's area_search 3 with area_search_modus 1, dynamo_mra_setup.m's).
The integer peak is sought within that sphere; the subpixel step may leave it by
about half a pixel, as in Dynamo."""
SUBSET = 300
"""Particles averaged into each 'subset' or 'blob' seed."""
RNG_SEED = 20261016
"""The one seed of every plan's random order, orientations and noise."""
SEED_KINDS = ("subset", "blob", "noise")
ROUNDS = {"fixed": 1, "coarse_to_fine": 5}
LOWPASS_STEPS = ("angstrom", "fourier")
TAG, SHIFTS, ANGLES, FTYPE, TILTS, PIXEL = 0, slice(3, 6), slice(6, 9), 12, slice(13, 15), 35
"""The export table's columns the setup reads (Dynamo's 1, 4-6, 7-9, 13, 14-15, 36)."""


@dataclass(frozen=True)
class ExportParticles:
    """What the setup reads from an export folder, by name rather than by column."""

    folder: Path
    """The particle files, particle_NNNNN.mrc or .em."""
    table: np.ndarray
    """initial.tbl's rows in the file's order, which the random draws follow."""
    box: int
    pixel_size: float
    """A per voxel: the table's column 36, as the setup read it from the first row."""

    @property
    def tags(self) -> np.ndarray:
        return self.table[:, TAG].astype(int)

    @property
    def angles(self) -> np.ndarray:
        return self.table[:, ANGLES]

    @property
    def shifts(self) -> np.ndarray:
        return self.table[:, SHIFTS]

    @property
    def tilt_ranges(self) -> list[tuple[float, float]]:
        return [(float(low), float(high)) for low, high in self.table[:, TILTS]]

    def load(self, tag: int) -> np.ndarray:
        return _particle(self.folder, int(tag))

    def __str__(self) -> str:
        return str(self.folder)


Particles = ExportParticles | RegistryParticles
"""What the setup reads: a table of Dynamo rows by tag, the box, the pixel size, and
each particle's voxels."""


def read_export(export: Path, data_folder: str = "particles") -> ExportParticles:
    """The export's particles and starting table, as the setup reads them."""

    export = Path(export).resolve()
    table_path = export / "initial.tbl"
    if not table_path.exists():
        raise ValueError(f"{export}: no initial.tbl; give an export folder of "
                         "scripts/export_dynamo.py")
    table = read_table(table_path)
    files = sorted((export / data_folder).glob("particle_*"))
    if not files:
        raise ValueError(f"{export}: no {data_folder}/particle_*")
    shape = read_map(files[0]).shape
    if len(set(shape)) != 1:
        raise ValueError(f"{files[0]}: not a cubic box, {shape}")
    if table.shape[1] <= PIXEL or not table[0, PIXEL] > 0:
        raise ValueError(f"{table_path}: column 36 holds no pixel size")
    if np.any(table[:, FTYPE] != 1):
        raise ValueError(f"{table_path}: only single-axis wedges (ftype 1) are modelled")
    if len(np.unique(table[:, TAG])) != len(table):
        raise ValueError(f"{table_path}: a tag is there twice")
    return ExportParticles(export / data_folder, table, shape[0], float(table[0, PIXEL]))


def soft_sphere(radius: float, box: int, edge: float) -> np.ndarray:
    """`dynamo_sphere(radius, box, (box + 1) / 2 * [1 1 1], edge)` as a (z, y, x) float32 map.

    As `volume_noncubic::fillSphere` computes it (Dynamo 1.1.591's C++ source): with
    each coordinate's distance from the box's centre, (box - 1) / 2 zero-based, over
    the radius, 1 where their squares sum to at most 1; past that, with the root d of
    that sum, exp(-((d - 1) / (edge / radius))^2), set to 0 where it is not above
    0.13; without an edge, 1 and 0 only.
    """

    if not radius > 0 or edge < 0:
        raise ValueError(f"a sphere needs a positive radius and no negative edge, got "
                         f"{radius} and {edge}")
    scaled = (np.arange(box, dtype=float) - (box - 1) / 2) / radius
    # Summed in Dynamo's order, x + y + z, so the last bits are its own.
    inside = scaled[None, None, :] ** 2 + scaled[None, :, None] ** 2 + scaled[:, None, None] ** 2
    sphere = (inside <= 1).astype(float)
    if edge > 0:
        outside = inside > 1
        weighted = (np.sqrt(inside[outside]) - 1) / (edge / radius)
        values = np.exp(-(weighted * weighted))
        sphere[outside] = np.where(values > 0.13, values, 0.0)
    return sphere.astype(np.float32)


def seed_draws(count: int, box: int, channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The setup's random draws, in the order dynamo_mra_setup.m drew them after
    `rng(20261016)`: the particles' order (`randperm(count)`, zero-based here), every
    particle's random orientation (tdrot, tilt, narot) and max(channels, 3) noise maps.

    MATLAB's `rand` is `RandomState.random_sample` for the same seed, and `randperm(n)`
    is the order that sorts n of its draws, so the order and the orientations are
    Dynamo's own. MATLAB's `randn` is not numpy's: the noise is the setup's own.
    """

    state = np.random.RandomState(RNG_SEED)
    order = np.argsort(state.random_sample(count), kind="stable")
    orientations = np.column_stack([
        360 * state.random_sample(count),
        np.degrees(np.arccos(2 * state.random_sample(count) - 1)),
        360 * state.random_sample(count),
    ])
    noise = state.standard_normal((max(channels, 3), box, box, box))
    return order, orientations, noise


@dataclass(frozen=True)
class Round:
    """One round of the search, as the setup gave it to Dynamo."""

    iterations: int
    low: int
    """The low-pass, in Fourier pixels of the box."""
    step: float
    """atan(resolution / diameter) in deg, to 0.1 deg."""
    cone_range: float
    """The cone's and the in-plane range's full opening in deg; 360 in a global round."""
    sampling: float
    refine: int
    dim: int
    """The box the round searches at."""
    is_global: bool

    def settings(self, box: int, shift_limit: float = SHIFT_LIMIT) -> SearchSettings:
        return SearchSettings(
            cone_aperture=self.cone_range, cone_sampling=self.sampling,
            inplane_range=self.cone_range, inplane_sampling=self.sampling, refine=self.refine,
            refine_factor=2.0, high=2.0, low=float(self.low), shift_limit=float(shift_limit),
            area_search_modus=1, dim=0 if self.dim == box else self.dim,
        )

    def describe(self, number: int, box: int, pixel_size: float) -> str:
        """The round as dynamo_mra_setup.m printed it."""

        if self.is_global:
            search = (f"every orientation, 360 deg in {self.sampling:.1f} deg steps refined "
                      f"{self.refine} times by 2 to {self.sampling / 2 ** self.refine:.1f} deg, "
                      f"particles at {self.dim} px of {box}")
        else:
            search = f"cone and in-plane {self.cone_range:.1f} deg in {self.step:.1f} deg steps"
        return (f"round {number}: {self.iterations} iteration(s), low-pass {self.low} px "
                f"({box * pixel_size / self.low:.1f} A), {search}")


def _matlab_round(value: float) -> int:
    """MATLAB's round: halves away from zero, where Python's go to the even neighbour."""

    whole = math.floor(abs(value))
    return int(math.copysign(whole + (abs(value) - whole >= 0.5), value))


def _linspace(first: float, last: float, count: int) -> list[float]:
    """MATLAB's linspace: first + (k * (last - first)) / (count - 1), the ends as given."""

    values = [first + k * (last - first) / (count - 1) for k in range(count)]
    values[0], values[-1] = first, last
    return values


def search_rounds(
    box: int,
    pixel_size: float,
    *,
    schedule: str = "fixed",
    first_lowpass: float = 40.0,
    last_lowpass: float = 22.0,
    diameter: float = 290.0,
    lowpass_steps: str = "angstrom",
    global_rounds: int = 0,
    global_step: float | None = None,
    global_full_box: bool = False,
    cone_steps: Sequence[float] = (3,),
) -> list[Round]:
    """Every round's search, as dynamo_mra_setup.m set it up (see its header).

    'fixed' is one round of 5 iterations, band 2..16 Fourier px, 5 deg steps;
    'coarse_to_fine' 5 rounds of one iteration from `first_lowpass` to `last_lowpass`
    (A), in equal steps of resolution or of frequency (`lowpass_steps`), each round's
    angular step atan(resolution / diameter). The first `global_rounds` search every
    orientation, downsampled to dim 2 * (low-pass + 2) unless `global_full_box`, in
    their own step or refined down to it from `global_step`.
    """

    size = box * pixel_size
    if schedule == "fixed":
        iterations, lows, steps = [5], [16], [5.0]
    else:
        if lowpass_steps == "angstrom":
            lows = [_matlab_round(size / value)
                    for value in _linspace(first_lowpass, last_lowpass, 5)]
        else:
            lows = [_matlab_round(value)
                    for value in _linspace(size / first_lowpass, size / last_lowpass, 5)]
        iterations = [1] * 5
        steps = [_matlab_round(10 * math.degrees(math.atan(size / low / diameter))) / 10
                 for low in lows]
    count = len(lows)
    widths = list(cone_steps) * count if len(cone_steps) == 1 else list(cone_steps)
    rounds = []
    for index in range(count):
        is_global = index < global_rounds
        cone_range, sampling, refine, dim = widths[index] * steps[index], steps[index], 2, box
        if is_global:
            cone_range = 360.0
            dim = box if global_full_box else min(box, 2 * (lows[index] + 2))
            if global_step is not None:
                sampling = global_step
                refine = max(2, math.ceil(math.log2(global_step / steps[index])))
        if not 2 < lows[index] <= box / 2:
            raise ValueError(f"round {index + 1}'s low-pass, {lows[index]} Fourier px, is not "
                             f"between the high-pass (2 px) and the {box} px box's Nyquist "
                             f"({box / 2:g} px)")
        if not dim_is_ported(dim, box):
            raise ValueError(f"round {index + 1} would search at dim {dim} of the {box} px box, "
                             "which divides it: Dynamo bins then, and the port does not")
        rounds.append(Round(iterations[index], lows[index], steps[index], round(cone_range, 6),
                            round(sampling, 6), refine, dim, is_global))
    return rounds


def _seed_note(channel: int, kind: str, members: int, reference_file: Path | None) -> str:
    """A line of seed_kinds.txt, as the report reads it."""

    return f"{channel} {kind} {reference_file if kind == 'map' else members}"


def plan_from_export(
    export: Path,
    seeds: Sequence[str],
    *,
    data_folder: str = "particles",
    **options: Any,
) -> MraPlan:
    """The plan dynamo_mra_setup.m made a project for, made from `export` without Dynamo;
    the options as `make_plan` takes them."""

    export = Path(export).resolve()
    return make_plan(lambda: read_export(export, data_folder), seeds, source=export,
                     made_by="cryoet_pipeline.mra_setup.plan_from_export",
                     origin={"data_folder": data_folder}, files={"export": str(export)},
                     **options)


def plan_from_registry(
    project: Path,
    label: str | None,
    series: Sequence[str],
    seeds: Sequence[str],
    *,
    decoy_label: str | None = None,
    log: Callable[[str], None] = print,
    **options: Any,
) -> MraPlan:
    """The same plan made from the particle stacks of `series` (in that order) with
    label `label` in `project`'s registry, the decoys of `decoy_label` after them: the
    particles an export of the same arguments would copy, read where they are. The
    project, the label, the series and the decoys' label are the plan's variables."""

    project = Path(project).resolve()
    series = list(series)
    return make_plan(
        lambda: registry_particles(project, label, series, decoy_label, log=log), seeds,
        source=project, made_by="cryoet_pipeline.mra_setup.plan_from_registry",
        origin={"particles": {"label": label, "series": series, "decoy_label": decoy_label}},
        files={"project": str(project)}, log=log, **options,
    )


def plan_from_split(
    split: Path,
    half: int,
    *,
    log: Callable[[str], None] = print,
    **options: Any,
) -> MraPlan:
    """A consensus half's plan (docs/halves_from_runs.md): the particles of half `half`
    of `split`, a split scripts/consensus_halves.py wrote of two runs of a registry, read
    from that registry as the runs read them, each at the split's starting pose, and
    refined from the split's start in one channel. The seeds and the reference are the
    split's; the other options as `make_plan` takes them."""

    split = Path(split).resolve()
    record = completed(split)
    if record is None or record.step != STEP_SPLIT:
        raise ValueError(f"{split}: not a split of scripts/consensus_halves.py (no run.json "
                         f"of step {STEP_SPLIT} that succeeded)")
    if half not in (1, 2):
        raise ValueError(f"half must be 1 or 2, got {half}")
    given = sorted(name for name in ("seeds", "reference_file") if name in options)
    if given:
        raise ValueError(f"a half's seed is its split's {START}; leave out {', '.join(given)}")
    columns, _ = read_particle_table(split / HALVES)
    rows = np.flatnonzero(columns["half"] == half)
    if not len(rows):
        raise ValueError(f"{split}: half {half} holds no particle")
    tags = columns["particle_id"][rows].astype(int)
    angles = np.column_stack([columns[name] for name in ("tdrot_deg", "tilt_deg",
                                                         "narot_deg")])[rows]
    shifts = np.column_stack([columns[name] for name in ("shift_x_px", "shift_y_px",
                                                         "shift_z_px")])[rows]
    # The project is the one the split's first run read, as that run's plan names it.
    poses = [item for item in record.inputs if item.slot == "poses"]
    node = completed(Path(poses[0].path)) if len(poses) == 1 else None
    if node is None or node.fingerprint != poses[0].fingerprint:
        raise ValueError(f"{split}: the run it was split from is not there as it was "
                         f"({poses[0].path if poses else 'no poses input'})")
    named = json.loads((Path(poses[0].path).parent / PLAN).read_text())["particles"]
    particles = record.content_params["particles"]
    if {key: value for key, value in named.items() if key != "project"} != particles:
        raise ValueError(f"{split}: its run's plan names other particles than the split")
    project = Path(named["project"])
    chosen = SelectionInput(name=f"half {half}", run=split, fingerprint=record.fingerprint)

    def read() -> Particles:
        return registry_particles(project, particles["label"], particles["series"],
                                  particles["decoy_label"], log=log).selected(
            tags, angles, shifts, chosen)

    return make_plan(
        read, ["map"], source=project.resolve(),
        made_by="cryoet_pipeline.mra_setup.plan_from_split",
        origin={"particles": {"label": particles["label"], "series": particles["series"],
                              "decoy_label": particles["decoy_label"]}},
        files={"project": str(project.resolve())}, reference_file=split / START, log=log,
        **options,
    )


def make_plan(
    read: Callable[[], Particles],
    seeds: Sequence[str],
    *,
    source: Path,
    made_by: str,
    origin: dict[str, Any],
    files: dict[str, str],
    reference_file: Path | None = None,
    schedule: str = "fixed",
    first_lowpass: float = 40.0,
    last_lowpass: float = 22.0,
    diameter: float = 290.0,
    lowpass_steps: str = "angstrom",
    mask_radius: float | None = None,
    mask_edge: float = 2.0,
    mask_file: Path | None = None,
    global_rounds: int = 0,
    global_step: float | None = None,
    global_full_box: bool = False,
    cone_steps: Sequence[float] = (3,),
    shift_limit: float = SHIFT_LIMIT,
    subset: int = SUBSET,
    log: Callable[[str], None] = print,
) -> MraPlan:
    """The plan dynamo_mra_setup.m made a project for, from the particles `read` gives.

    `origin` says where the particles come from among the plan's variables, `files`
    names the files the setup reads besides the reference and the mask.

    `seeds` holds one kind per channel: 'subset' (the average of `subset` particles at
    their exported poses; channel k takes the k-th block of one random order), 'blob'
    (such a block at random orientations), 'noise'; or ['map'] alone, which refines
    from `reference_file` in one channel without MRA. The mask is a sphere of
    `mask_radius` px (the particle's radius from `diameter` plus 3 px unless given)
    with a soft edge of `mask_edge` px, or `mask_file`; it aligns, classifies and
    averages. The search is `search_rounds`', every particle kept within `shift_limit`
    px of its box centre in every round (`SHIFT_LIMIT`). The limit changes only what
    the iterations search, which each iteration's record holds whole; it is not one of
    the setup's variables, whose results it does not touch.
    """

    seeds = list(seeds)
    one_map = seeds == ["map"]
    if not one_map and (len(seeds) < 2 or any(kind not in SEED_KINDS for kind in seeds)):
        raise ValueError("seeds must be ['map'] alone or list at least two of 'subset', "
                         f"'blob' and 'noise', got {seeds}")
    if one_map != (reference_file is not None):
        raise ValueError("give reference_file with seeds ['map'], and only then")
    if schedule not in ROUNDS:
        raise ValueError(f"schedule must be 'fixed' or 'coarse_to_fine', got {schedule!r}")
    if lowpass_steps not in LOWPASS_STEPS:
        raise ValueError(f"lowpass_steps must be 'angstrom' or 'fourier', got {lowpass_steps!r}")
    if type(global_rounds) is not int or not 0 <= global_rounds <= ROUNDS[schedule]:
        raise ValueError(f"global_rounds must be a whole number from 0 to the schedule's "
                         f"rounds ({ROUNDS[schedule]} for {schedule}), got {global_rounds!r}")
    if global_step is not None and not 0 < global_step <= 180:
        raise ValueError(f"global_step must be above 0 and at most 180 deg, got {global_step}")
    if global_full_box and not global_rounds:
        raise ValueError("global_full_box changes only global rounds; give global_rounds too")
    cone_steps = list(cone_steps)
    if len(cone_steps) not in (1, ROUNDS[schedule]) or not all(step > 0 for step in cone_steps):
        raise ValueError(f"cone_steps must be one positive number of steps or one per round "
                         f"({ROUNDS[schedule]} rounds), got {cone_steps}")

    particles = read()
    box, pixel_size, count = particles.box, particles.pixel_size, len(particles.table)
    if not 0 < shift_limit < box / 2:
        raise ValueError(f"shift_limit must be above 0 and below half the {box} px box, got "
                         f"{shift_limit:g}")
    if schedule == "coarse_to_fine" and not 2 * pixel_size <= last_lowpass < first_lowpass:
        raise ValueError(f"last_lowpass must lie between Nyquist ({2 * pixel_size:.1f} A) and "
                         f"first_lowpass ({first_lowpass:g} A)")
    nref = len(seeds)
    # Channel k averages the k-th block of the order. Dynamo's setup asked for a block
    # per channel, a noise channel's too; only the blocks up to the last averaged one
    # are needed.
    blocks = max((channel for channel, kind in enumerate(seeds, 1)
                  if kind in ("subset", "blob")), default=0)
    if blocks * subset > count:
        raise ValueError(f"{count} particles cannot give {blocks} disjoint subsets of {subset}")
    reference = None
    if reference_file is not None:
        reference_file = Path(reference_file).resolve()
        if not reference_file.exists():
            raise ValueError(f"no reference_file {reference_file}")
        reference = read_map(reference_file)
        if reference.shape != (box,) * 3:
            raise ValueError(f"reference_file {reference_file} is not a {box} px box like "
                             "the particles")
    if mask_file is not None:
        mask_file = Path(mask_file).resolve()
        if not mask_file.exists():
            raise ValueError(f"no mask_file {mask_file}")
        mask = read_map(mask_file)
        if mask.shape != (box,) * 3:
            raise ValueError(f"mask_file {mask_file} is not a {box} px box like the particles")
        mask_note = str(mask_file)
    else:
        # The rule for every mask (Anna, 2026-10-16): the particle's widest radius plus
        # 3 px, then a soft edge.
        if mask_radius is None:
            mask_radius = _matlab_round(diameter / 2 / pixel_size + 3)
        if mask_radius + mask_edge > box / 2:
            raise ValueError(f"a mask of {mask_radius:g} px with a {mask_edge:g} px edge does "
                             f"not fit the {box} px box; give a smaller mask_radius or diameter")
        mask = soft_sphere(mask_radius, box, mask_edge)
        mask_note = f"sphere {mask_radius:g} px with a {mask_edge:g} px edge"
    mask = mask.astype(np.float32).astype(float)
    rounds = search_rounds(
        box, pixel_size, schedule=schedule, first_lowpass=first_lowpass,
        last_lowpass=last_lowpass, diameter=diameter, lowpass_steps=lowpass_steps,
        global_rounds=global_rounds, global_step=global_step, global_full_box=global_full_box,
        cone_steps=cone_steps,
    )

    order, orientations, noise = seed_draws(count, box, nref)
    group = np.zeros(count, dtype=int)
    for channel, kind in enumerate(seeds, 1):
        if kind in ("subset", "blob"):
            group[order[(channel - 1) * subset:channel * subset]] = channel
    references, notes = {}, []
    for channel, kind in enumerate(seeds, 1):
        if kind == "map":
            seed = reference
            log(f"seed {channel}: the map {reference_file}")
        elif kind == "noise":
            seed = noise[channel - 1]
            log(f"seed {channel}: noise")
        else:
            # Averaged in ascending tags, as dynamo_average sums a class.
            members = np.flatnonzero(group == channel)
            members = members[np.argsort(particles.tags[members], kind="stable")]
            angles = orientations if kind == "blob" else particles.angles
            seed = average_class(
                (particles.load(int(tag)) for tag in particles.tags[members]),
                angles[members], particles.shifts[members],
                [particles.tilt_ranges[index] for index in members], mask, fmin=1,
            ).average
            log(f"seed {channel}: {kind} of {len(members)} particles")
        references[channel] = np.asarray(seed, dtype=np.float32).astype(float)
        notes.append(_seed_note(channel, kind, int(np.sum(group == channel)), reference_file))
    for number, round_ in enumerate(rounds, 1):
        log(round_.describe(number, box, pixel_size))
    log(f"shifts: up to {shift_limit:g} px ({shift_limit * pixel_size:.1f} A) from the box "
        "centre in every round")

    settings = [round_.settings(box, shift_limit) for round_ in rounds
                for _ in range(round_.iterations)]
    mra = [nref > 1] * len(settings)
    cards = [{key: str(value) for key, value in
              {**_search_fields(searched, box), "mra": int(mra[0])}.items()}
             for searched in settings]
    table = particles.table[np.argsort(particles.tags, kind="stable")]
    sphere = mask_file is None
    setup: dict[str, Any] = {
        "made_by": made_by,
        # What the plan is made from besides the files below: a run's content
        # parameters (docs/run_records.md). Numbers as floats, so that 17 and 17.0 given
        # by two callers declare the same plan.
        "variables": {
            "seeds": seeds, "subset": int(subset), "rng_seed": RNG_SEED, **origin,
            "schedule": schedule,
            "first_lowpass_angstrom": float(first_lowpass),
            "last_lowpass_angstrom": float(last_lowpass), "diameter_angstrom": float(diameter),
            "lowpass_steps": lowpass_steps, "global_rounds": int(global_rounds),
            "global_step_deg": None if global_step is None else float(global_step),
            "global_full_box": bool(global_full_box),
            "cone_steps": [float(step) for step in cone_steps],
            "mask_radius_px": float(mask_radius) if sphere and mask_radius is not None else None,
            "mask_edge_px": float(mask_edge) if sphere else None,
        },
        # The files it read: its inputs, known by their contents rather than their paths.
        "files": {**files,
                  "reference_file": str(reference_file) if reference_file else None,
                  "mask_file": str(mask_file) if mask_file else None},
        # What follows from both, for the record.
        "derived": {
            "particles": count, "box_px": box, "pixel_size_angstrom": pixel_size,
            "mask": mask_note,
            "rounds": [{"iterations": r.iterations, "low_px": r.low, "step_deg": r.step,
                        "cone_range_deg": r.cone_range, "sampling_deg": r.sampling,
                        "refine": r.refine, "dim_px": r.dim, "global": r.is_global}
                       for r in rounds],
        },
    }
    log(f"mra_setup: {len(settings)} iterations of {count:,} particles from "
        f"{particles}, seeds {' '.join(seeds)}, schedule {schedule} ({lowpass_steps} "
        f"steps), {global_rounds} global round(s), mask {mask_note}")
    return MraPlan(
        source=source,
        particles=particles.folder if isinstance(particles, ExportParticles) else particles,
        table=table,
        references=references, reference_notes=notes, soft_mask=mask, settings=settings,
        cards=cards, mra=mra, seed_groups=np.column_stack([particles.tags, group]),
        setup=setup,
    )


__all__ = ["SHIFT_LIMIT", "ExportParticles", "Particles", "Round", "make_plan",
           "plan_from_export",
           "plan_from_registry", "plan_from_split", "read_export", "search_rounds",
           "seed_draws", "soft_sphere"]
