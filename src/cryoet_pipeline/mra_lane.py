"""A lane's MRA (docs/mra_in_lanes.md): the [mra] section of a lane config, the
defaults the lab's rules give it, and the run it comes to once the particles are known.

`MraSettings` is the step's parameter schema, typed, with its limits, as
docs/database_design.md asks of every step: a key left out of [mra] is None here and
takes its default from `lane_mra`, which says for every value whether the lane gave it
or which rule did. scripts/submit_lanes.py checks a lane with it and shows what it
comes to; scripts/mra_port_run.py --config runs it. Where the job runs is the lane's
[[run.stage]] of the step `mra`, not a key here.

The low-pass is given relative to the particles' Nyquist by default, so its numbers in
A are known only with the particles' pixel size: `LaneMra.options` takes it, and gives
`mra_setup.make_plan`'s arguments, the same numbers a run asked for by hand would give.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cryoet_pipeline.mra_setup import ROUNDS, SHIFT_LIMIT, START_POSES, SUBSET, Round, search_rounds

SECTION = "mra"
STEP = "mra"
"""The step a [[run.stage]] runs to run the MRA; the last of every lane's chain."""

SEED_LETTERS = {"subset": "s", "blob": "b", "noise": "n", "map": "m"}
DEFAULT_SEEDS = ("subset", "subset", "noise")
FIRST_NYQUIST = 2.0
LAST_NYQUIST = 1.1
GLOBAL_ROUNDS = 2
CONE_STEPS = 3.0
MASK_EDGE = 2.0
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MraSettings(BaseModel):
    """[mra]: what differs from the defaults. A key left out is None, its default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = None
    """The run, <project>/mra/<name>; default <[lane] variant>-<one letter per seed>."""
    seeds: list[Literal["subset", "blob", "noise", "map"]] | None = None
    """One kind per channel; default subset, subset, noise. ['map'] alone with reference."""
    reference: str | None = None
    """The map a 'map' seed refines from, in the particles' box, from the repo root."""
    schedule: Literal["fixed", "coarse_to_fine"] | None = None
    first_lowpass_nyquist: float | None = Field(default=None, ge=1.0)
    """The first round's low-pass in multiples of the particles' Nyquist; default 2."""
    last_lowpass_nyquist: float | None = Field(default=None, ge=1.0)
    """The last round's; default 1.1."""
    first_lowpass_angstrom: float | None = Field(default=None, gt=0)
    last_lowpass_angstrom: float | None = Field(default=None, gt=0)
    lowpass_steps: Literal["angstrom", "fourier"] | None = None
    diameter_angstrom: float | None = Field(default=None, gt=0)
    """The particle where it is widest; default [pick] particle_diameter_angstrom."""
    global_rounds: int | None = Field(default=None, ge=0)
    """The first rounds, over every orientation; default 2 (0 for 'fixed')."""
    global_step_deg: float | None = Field(default=None, gt=0, le=180)
    global_full_box: bool | None = None
    cone_steps: float | list[float] | None = None
    """Each cone's width in angular steps, one number or one per round; default 3."""
    antialias: bool | None = None
    shift_limit_px: float | None = Field(default=None, gt=0)
    mask_radius_px: float | None = Field(default=None, gt=0)
    mask_edge_px: float | None = Field(default=None, ge=0)
    mask_file: str | None = None
    decoy_label: str | None = None
    subset_particles: int | None = Field(default=None, ge=1)
    """The particles a 'subset' or 'blob' seed averages; default 300."""
    from_run: str | None = None
    """An earlier run of the project, <project>/mra/<from_run>: run again on part of it."""
    from_classes: int | list[int] | None = None
    """Its classes whose particles are run again."""
    from_iteration: int | None = Field(default=None, ge=1)
    """Its iteration whose classes they are; default its last."""
    start_poses: Literal["run", "picked"] | None = None
    """Where they start: the poses that iteration gave them (default), or their picks'."""

    @model_validator(mode="after")
    def _consistent(self) -> MraSettings:
        if self.name is not None and not _NAME.match(self.name):
            raise ValueError(f"name {self.name!r} must be a folder name: letters, digits, "
                             "'.', '_' and '-', not starting with a dot or a dash")
        if (self.from_run is None) != (self.from_classes is None):
            raise ValueError("from_run and from_classes go together: the earlier run, and "
                             "which of its classes")
        if self.from_run is None and (self.from_iteration is not None
                                      or self.start_poses is not None):
            raise ValueError("from_iteration and start_poses choose part of an earlier run; "
                             "they need from_run and from_classes")
        if self.from_run is not None:
            if not _NAME.match(self.from_run):
                raise ValueError(f"from_run {self.from_run!r} must be the name of a run in "
                                 "<project>/mra")
            if self.from_run == self.name:
                raise ValueError("from_run is an earlier run; name the new one otherwise")
            if self.decoy_label is not None:
                raise ValueError("the particles of from_run's classes are that run's, decoys "
                                 "among them as it read them; leave out decoy_label")
            classes: list[int] = (self.from_classes if isinstance(self.from_classes, list)
                                  else [] if self.from_classes is None
                                  else [self.from_classes])
            if not classes or any(channel < 1 for channel in classes) \
                    or len(set(classes)) != len(classes):
                raise ValueError(f"from_classes names classes by their number, each once; "
                                 f"got {self.from_classes}")
        for end in ("first", "last"):
            if (getattr(self, f"{end}_lowpass_nyquist") is not None
                    and getattr(self, f"{end}_lowpass_angstrom") is not None):
                raise ValueError(f"give {end}_lowpass_nyquist or {end}_lowpass_angstrom, "
                                 "not both")
        seeds = self.seeds
        if seeds is not None:
            if seeds == ["map"]:
                if self.reference is None:
                    raise ValueError("a 'map' seed refines from reference: give it")
            elif "map" in seeds or len(seeds) < 2:
                raise ValueError("seeds must be ['map'] alone, or at least two of 'subset', "
                                 f"'blob' and 'noise'; got {seeds}")
        if self.reference is not None and seeds != ["map"]:
            raise ValueError("reference is the map of seeds = ['map'], and only of that")
        if self.mask_file is not None and (self.mask_radius_px is not None
                                           or self.mask_edge_px is not None):
            raise ValueError("mask_file replaces the sphere; leave out mask_radius_px and "
                             "mask_edge_px")
        if self.schedule == "fixed":
            given = [key for key in ("first_lowpass_nyquist", "last_lowpass_nyquist",
                                     "first_lowpass_angstrom", "last_lowpass_angstrom",
                                     "lowpass_steps")
                     if getattr(self, key) is not None]
            if given:
                raise ValueError(f"the 'fixed' schedule searches one band (2..16 Fourier px) "
                                 f"in 5 deg steps; {', '.join(given)} is for 'coarse_to_fine'")
        schedule = self.schedule or "coarse_to_fine"
        if self.global_rounds is not None and self.global_rounds > ROUNDS[schedule]:
            raise ValueError(f"global_rounds is at most the schedule's rounds, "
                             f"{ROUNDS[schedule]} for {schedule}")
        steps = self.cone_steps
        if isinstance(steps, list) and (len(steps) not in (1, ROUNDS[schedule])
                                        or not all(step > 0 for step in steps)):
            raise ValueError(f"cone_steps is one positive number or one per round "
                             f"({ROUNDS[schedule]} for {schedule}); got {steps}")
        if isinstance(steps, (int, float)) and not steps > 0:
            raise ValueError(f"cone_steps must be positive; got {steps}")
        return self


@dataclass(frozen=True)
class Value:
    """One setting of the run, and where it came from: 'lane', or the rule's words."""

    value: Any
    source: str


@dataclass(frozen=True)
class LaneMra:
    """A lane's MRA as its config asks for it, before the particles are known."""

    settings: MraSettings
    project: str
    label: str
    values: dict[str, Value]

    @property
    def name(self) -> str:
        return str(self.values["name"].value)

    @property
    def seeds(self) -> list[str]:
        return list(self.values["seeds"].value)

    @property
    def decoy_label(self) -> str | None:
        value: str | None = self.values["decoy_label"].value
        return value

    @property
    def reference(self) -> str | None:
        value: str | None = self.values["reference"].value
        return value

    @property
    def from_run(self) -> str | None:
        value: str | None = self.values["from_run"].value
        return value

    @property
    def from_classes(self) -> list[int] | None:
        value: list[int] | None = self.values["from_classes"].value
        return value

    @property
    def from_iteration(self) -> int | None:
        value: int | None = self.values["from_iteration"].value
        return value

    @property
    def start_poses(self) -> str | None:
        value: str | None = self.values["start_poses"].value
        return value

    def run(self, root: Path = Path()) -> Path:
        """The run's folder."""
        return root / self.project / "mra" / self.name

    def lowpass_angstrom(self, pixel_size: float) -> tuple[float, float]:
        """The first and last round's low-pass in A, for particles of this pixel size."""
        nyquist = 2 * pixel_size
        found = []
        for end in ("first", "last"):
            kind, number = self.values[f"{end}_lowpass"].value
            found.append(float(number) if kind == "angstrom" else float(number) * nyquist)
        return found[0], found[1]

    def options(self, pixel_size: float) -> dict[str, Any]:
        """`mra_setup.make_plan`'s arguments besides the particles, the seeds and `log`."""
        values = {key: value.value for key, value in self.values.items()}
        options: dict[str, Any] = {
            "schedule": values["schedule"], "diameter": float(values["diameter"]),
            "global_rounds": int(values["global_rounds"]),
            "global_step": values["global_step"],
            "global_full_box": bool(values["global_full_box"]),
            "cone_steps": [float(step) for step in values["cone_steps"]],
            "shift_limit": float(values["shift_limit"]),
            "antialias": bool(values["antialias"]),
            "mask_radius": values["mask_radius"], "mask_edge": float(values["mask_edge"]),
            "mask_file": Path(values["mask_file"]) if values["mask_file"] else None,
            "reference_file": Path(values["reference"]) if values["reference"] else None,
            "subset": int(values["subset"]),
        }
        if values["schedule"] == "coarse_to_fine":
            first, last = self.lowpass_angstrom(pixel_size)
            options |= {"first_lowpass": first, "last_lowpass": last,
                        "lowpass_steps": values["lowpass_steps"]}
        return options

    def setup_problems(self, box: int, pixel_size: float, count: int) -> list[str]:
        """What `mra_setup.make_plan` would refuse for this many particles of this box and
        pixel size, found without reading their voxels: the dry run's check."""
        found = []
        options = self.options(pixel_size)
        if not 0 < options["shift_limit"] < box / 2:
            found.append(f"a shift limit of {options['shift_limit']:g} px is not below half "
                         f"the {box} px box")
        if options["schedule"] == "coarse_to_fine":
            first, last = options["first_lowpass"], options["last_lowpass"]
            if not 2 * pixel_size <= last < first:
                found.append(f"the low-pass must go from coarser to finer, no finer than the "
                             f"particles' Nyquist ({2 * pixel_size:g} A); it goes {first:.2f} "
                             f"-> {last:.2f} A")
        blocks = max((channel for channel, kind in enumerate(self.seeds, 1)
                      if kind in ("subset", "blob")), default=0)
        if blocks * options["subset"] > count:
            found.append(f"{count:,} particles cannot give {blocks} disjoint subsets of "
                         f"{options['subset']:,}; give [mra] subset_particles, or other seeds")
        if options["mask_file"] is None:
            radius = options["mask_radius"]
            if radius is None:
                radius = math.floor(options["diameter"] / 2 / pixel_size + 3 + 0.5)
            if radius + options["mask_edge"] > box / 2:
                found.append(f"a mask of {radius:g} px with a {options['mask_edge']:g} px edge "
                             f"does not fit the {box} px box; give a smaller "
                             "diameter_angstrom or mask_radius_px")
        try:
            self.rounds(box, pixel_size)
        except ValueError as error:
            found.append(str(error))
        return found

    def rounds(self, box: int, pixel_size: float) -> list[Round]:
        """The search of every round, as the setup will make it."""
        options = self.options(pixel_size)
        keys = ("schedule", "first_lowpass", "last_lowpass", "diameter", "lowpass_steps",
                "global_rounds", "global_step", "global_full_box", "cone_steps", "antialias")
        return search_rounds(box, pixel_size, **{key: options[key] for key in keys
                                                 if key in options})

    def resolved(self, pixel_size: float | None) -> dict[str, Any]:
        """Every setting as the run will take it: the low-pass in A when the pixel size
        is known, else as asked. What the dry run compares with an existing run."""
        out: dict[str, Any] = {}
        for key, value in self.values.items():
            if key in ("first_lowpass", "last_lowpass"):
                kind, number = value.value
                out[f"{key}_{kind}"] = float(number)
            else:
                out[key] = value.value
        if pixel_size is not None and self.values["schedule"].value == "coarse_to_fine":
            first, last = self.lowpass_angstrom(pixel_size)
            out["first_lowpass_resolved_angstrom"] = first
            out["last_lowpass_resolved_angstrom"] = last
        return out

    def request(self, config: Path, config_text: str, pixel_size: float,
                series: list[str] | tuple[str, ...],
                selection: dict[str, Any] | None = None) -> dict[str, Any]:
        """How the run was asked for, for plan.json's setup.request: the lane file, its
        sha256, [mra] as written, the particles (and, for a run on an earlier run's
        classes, the iteration and particles it took), and every value with its source."""
        return {
            "made_by": "cryoet_pipeline.mra_lane",
            "lane_config": str(config),
            "lane_config_sha256": config_sha256(config_text),
            "mra_section": self.settings.model_dump(exclude_none=True),
            "label": self.label,
            "series": list(series),
            **({"selection": selection} if selection is not None else {}),
            "resolved": self.resolved(pixel_size),
            "sources": {key: value.source for key, value in self.values.items()},
        }

    def describe(self, box: int | None = None, pixel_size: float | None = None,
                 count: int | None = None) -> list[str]:
        """The run in lines for a dry run, each value with its source."""
        values = self.values

        def line(label: str, text: str, key: str) -> str:
            return f"{label:<12} {text}   ({values[key].source})"

        seeds = " ".join(self.seeds)
        lines = []
        if self.from_run is not None:
            classes = " + ".join(f"class {channel}" for channel in self.from_classes or [])
            iteration = (f"iteration {self.from_iteration}" if self.from_iteration is not None
                         else "its last iteration")
            poses = ("the poses that iteration gave them" if self.start_poses == "run"
                     else "their picks' poses")
            lines.append(line("particles", f"{classes} of run {self.from_run}, {iteration}; "
                              f"they start at {poses}", "from_run"))
        lines.append(line("seeds", seeds + (f", from {self.reference}" if self.reference
                                            else ""), "seeds"))
        if any(kind in ("subset", "blob") for kind in self.seeds):
            lines.append(line("subsets", f"{values['subset'].value:,} particles per subset or "
                              "blob seed", "subset"))
        schedule = values["schedule"].value
        rounds_total = ROUNDS[schedule]
        lines.append(line("schedule", f"{schedule}, {rounds_total} round(s)", "schedule"))
        if schedule == "coarse_to_fine":
            ends = []
            for end in ("first", "last"):
                kind, number = values[f"{end}_lowpass"].value
                ends.append(f"{number:g} x Nyquist" if kind == "nyquist" else f"{number:g} A")
            steps = "resolution" if values["lowpass_steps"].value == "angstrom" else "frequency"
            sources = [values[f"{end}_lowpass"].source for end in ("first", "last")]
            source = (sources[0] if sources[0] == sources[1] else
                      "rule" if all(one.startswith("rule") for one in sources) else
                      f"first: {sources[0]}; last: {sources[1]}")
            lines.append(f"{'low-pass':<12} {' -> '.join(ends)}, in equal steps of {steps}   "
                         f"({source})")
        lines.append(line("diameter", f"{values['diameter'].value:g} A", "diameter"))
        rounds = values["global_rounds"].value
        lines.append(line("global", (f"the first {rounds} round(s) over every orientation"
                                     if rounds else "none: every round searches a cone")
                          + (f", from a {values['global_step'].value:g} deg grid"
                             if rounds and values["global_step"].value else "")
                          + (", at the full box" if rounds and values["global_full_box"].value
                             else ""),
                          "global_rounds"))
        cone = values["cone_steps"].value
        lines.append(line("cones", f"{' '.join(f'{step:g}' for step in cone)} angular step(s) "
                          "wide; every step atan(resolution / diameter)", "cone_steps"))
        if values["mask_file"].value:
            lines.append(line("mask", f"{values['mask_file'].value}", "mask_file"))
        else:
            radius = values["mask_radius"].value
            lines.append(line("mask", "sphere of "
                              + ("the particle's radius + 3 px" if radius is None
                                 else f"{radius:g} px")
                              + f", {values['mask_edge'].value:g} px soft edge", "mask_radius"))
        lines.append(line("shifts", f"up to {values['shift_limit'].value:g} px from the box "
                          "centre", "shift_limit"))
        lines.append(line("antialias", "on" if values["antialias"].value else "off",
                          "antialias"))
        if self.from_run is None:
            lines.append(line("decoys", self.decoy_label or "none", "decoy_label"))
        else:
            lines.append(f"{'decoys':<12} those among the classes, as run {self.from_run} "
                         "read them")
        if box is not None and pixel_size is not None:
            how_many = f"{count:,} " if count is not None else ""
            lines.append(f"{'particles':<12} {how_many}at {pixel_size:g} A/px in a {box} px "
                         f"box (Nyquist {2 * pixel_size:g} A)")
            if schedule == "coarse_to_fine":
                first, last = self.lowpass_angstrom(pixel_size)
                lines.append(f"{'':<12} low-pass {first:.2f} -> {last:.2f} A")
            try:
                rounds = self.rounds(box, pixel_size)
            except ValueError as error:
                lines.append(f"{'':<12} the setup would refuse it: {error}")
            else:
                lines += [f"{'':<12} {one.describe(number, box, pixel_size)}"
                          for number, one in enumerate(rounds, 1)]
        elif self.from_run is None:
            lines.append(f"{'particles':<12} not extracted yet: the job reads their pixel size "
                         "and box, and the low-pass in A follows from them")
        return lines


class LaneMraError(ValueError):
    """[mra] that cannot be run, in words for the person."""


def config_sha256(text: str) -> str:
    """The lane file's digest, as scripts/submit_lanes.py journals it."""
    return hashlib.sha256(text.encode()).hexdigest()


def settings_of(cfg: dict[str, Any]) -> MraSettings:
    """[mra] of a lane config, checked; an empty one when it has none."""
    section = cfg.get(SECTION, {})
    if not isinstance(section, dict):
        raise LaneMraError(f"[{SECTION}] must be a table")
    try:
        return MraSettings.model_validate(section)
    except ValidationError as error:
        problems = []
        for item in error.errors():
            where = ".".join(str(part) for part in item["loc"])
            message = str(item["msg"]).removeprefix("Value error, ")
            problems.append(f"[{SECTION}] {where}: {message}" if where
                            else f"[{SECTION}] {message}")
        raise LaneMraError("; ".join(problems)) from None


def lane_mra(cfg: dict[str, Any]) -> LaneMra:
    """The lane's MRA: [mra] over the defaults, each value with its source."""

    settings = settings_of(cfg)
    lane = cfg.get("lane", {})
    label = lane.get("variant")
    if not isinstance(label, str) or not label:
        raise LaneMraError("[lane] variant is the particles' label, which the MRA reads")
    project = cfg.get("run", {}).get("project")
    if not isinstance(project, str) or not project:
        raise LaneMraError("[run] project is where the particles are and the run goes")
    pick = cfg.get("pick", {})

    def given(key: str, default: Any, rule: str) -> Value:
        value = getattr(settings, key)
        return Value(default, rule) if value is None else Value(value, "lane")

    schedule = given("schedule", "coarse_to_fine", "rule: coarse to fine")
    seeds = given("seeds", list(DEFAULT_SEEDS), "rule: subset, subset, noise")
    letters = "".join(SEED_LETTERS[kind] for kind in seeds.value)
    classes = settings.from_classes
    if classes is not None and not isinstance(classes, list):
        classes = [classes]
    if settings.from_run is None:
        name = given("name", f"{label}-{letters}", "rule: <[lane] variant>-<one letter per seed>")
        start = Value(None, "rule: the picks' poses")
    else:
        name = given("name", f"{settings.from_run}-c{'_'.join(map(str, classes or []))}-{letters}",
                     "rule: <from_run>-c<classes>-<one letter per seed>")
        start = given("start_poses", START_POSES[0], "rule: the poses that iteration gave them")
    values = {
        "name": name,
        "from_run": Value(settings.from_run,
                          "lane" if settings.from_run else "rule: the lane's particles"),
        "from_classes": Value(classes, "lane" if classes else "rule: all of them"),
        "from_iteration": given("from_iteration", None, "rule: the run's last iteration"),
        "start_poses": start,
        "seeds": seeds,
        "subset": given("subset_particles", SUBSET, f"rule: {SUBSET} per seed"),
        "reference": given("reference", None, "rule: none"),
        "schedule": schedule,
    }
    for end, factor in (("first", FIRST_NYQUIST), ("last", LAST_NYQUIST)):
        nyquist = getattr(settings, f"{end}_lowpass_nyquist")
        angstrom = getattr(settings, f"{end}_lowpass_angstrom")
        if angstrom is not None:
            values[f"{end}_lowpass"] = Value(("angstrom", angstrom), "lane")
        elif nyquist is not None:
            values[f"{end}_lowpass"] = Value(("nyquist", nyquist), "lane")
        else:
            values[f"{end}_lowpass"] = Value(("nyquist", factor),
                                             f"rule: {factor:g} x the particles' Nyquist")
    diameter = settings.diameter_angstrom
    if diameter is None:
        diameter = pick.get("particle_diameter_angstrom")
        if not isinstance(diameter, (int, float)) or isinstance(diameter, bool) or diameter <= 0:
            raise LaneMraError("the MRA's diameter comes from [pick] particle_diameter_angstrom, "
                               "which is not a positive number; or give [mra] diameter_angstrom")
        values["diameter"] = Value(float(diameter), "[pick] particle_diameter_angstrom")
    else:
        values["diameter"] = Value(float(diameter), "lane")
    cone = settings.cone_steps
    values |= {
        "lowpass_steps": given("lowpass_steps", "angstrom", "rule: equal steps in A"),
        "global_rounds": given("global_rounds", GLOBAL_ROUNDS if schedule.value ==
                               "coarse_to_fine" else 0,
                               "rule: 2 global rounds, then cones" if schedule.value ==
                               "coarse_to_fine" else "rule: none for 'fixed'"),
        "global_step": given("global_step_deg", None, "rule: each round's own atan step"),
        "global_full_box": given("global_full_box", False, "rule: shrunk to the round"),
        "cone_steps": (Value([float(CONE_STEPS)], "rule: 3 steps") if cone is None else
                       Value([float(step) for step in (cone if isinstance(cone, list)
                                                       else [cone])], "lane")),
        "antialias": given("antialias", False, "rule: off, as Dynamo"),
        "shift_limit": given("shift_limit_px", SHIFT_LIMIT, f"rule: {SHIFT_LIMIT:g} px"),
        "mask_radius": given("mask_radius_px", None, "rule: the particle's radius + 3 px"),
        "mask_edge": given("mask_edge_px", MASK_EDGE, f"rule: {MASK_EDGE:g} px"),
        "mask_file": given("mask_file", None, "rule: the sphere"),
        "decoy_label": given("decoy_label", None, "rule: none"),
    }
    if values["decoy_label"].value == label:
        raise LaneMraError(f"[mra] decoy_label must differ from the particles' label, {label!r}")
    if values["global_full_box"].value and not values["global_rounds"].value:
        raise LaneMraError("[mra] global_full_box changes only global rounds; it needs "
                           "global_rounds")
    if values["antialias"].value and (not values["global_rounds"].value
                                      or values["global_full_box"].value):
        raise LaneMraError("[mra] antialias changes only global rounds searched at a smaller "
                           "box: it needs global_rounds, without global_full_box")
    if schedule.value == "coarse_to_fine":
        first, last = values["first_lowpass"].value, values["last_lowpass"].value
        if first[0] == last[0] and not first[1] > last[1]:
            raise LaneMraError(f"[mra] the first round's low-pass ({first[1]:g}) must be "
                               f"coarser than the last's ({last[1]:g})")
    return LaneMra(settings=settings, project=project, label=label, values=values)


def existing_request(run: Path) -> dict[str, Any] | None:
    """What a run's plan.json says it was asked for by a lane, or None: no plan.json, or
    a run asked for some other way. Raises OSError/ValueError for an unreadable one."""
    plan = run / "plan.json"
    if not plan.is_file():
        return None
    record = json.loads(plan.read_text())
    request = (record.get("setup") or {}).get("request")
    return request if isinstance(request, dict) else None


def lane_template(project: Path, label: str, series: list[str] | tuple[str, ...]) -> Path | None:
    """The template the lane's picks were matched with, from its registry: the first
    series' with one under the label, for the report to draw. None when there is none."""
    from cryoet_pipeline.artifacts import ArtifactRegistry
    from cryoet_pipeline.models import ArtifactKind

    registry = ArtifactRegistry.load(project / "artifacts.json")
    templates = [artifact for artifact in registry.by_kind(ArtifactKind.TEMPLATE)
                 if artifact.variant == label]
    for name in series:
        for artifact in templates:
            if artifact.covers(name):
                return Path(artifact.path)
    return None


__all__ = ["CONE_STEPS", "DEFAULT_SEEDS", "FIRST_NYQUIST", "GLOBAL_ROUNDS", "LAST_NYQUIST",
           "MASK_EDGE", "SECTION", "STEP", "LaneMra", "LaneMraError", "MraSettings", "Value",
           "config_sha256", "existing_request", "lane_mra", "lane_template", "settings_of"]
