#!/usr/bin/env python3
"""The pipeline's MRA setup against Dynamo's: the same plan from the same variables?

    python scripts/compare_mra_setup.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan2 ...

For each Dynamo project scripts/dynamo_mra_setup.m made, reads the variables it was
made with from the command in <export>/<name>_setup.log (scripts/dynamo_mra_mdc.sbatch
writes it there), makes the plan from the same export with
`mra_setup.plan_from_export`, and compares it with the project's own
(`plan_from_dynamo`):

- the particles' folder, the starting table, every iteration's search, the mask and
  which particles each seed averaged (the project's seed_manifest.txt) must be the
  same. The table only after rounding ours to 5 significant digits, as Dynamo wrote
  its own; the run keeps the export's digits.
- a map seed must be the same map; a subset or blob seed, Dynamo's average to 1e-5
  over the whole box (the norm of the difference over the norm of Dynamo's), what
  float32 leaves being 2e-7 to 8e-7: five projects made in Dynamo 1.1.511 on the Mac
  from the CZII export, 2026-11-02, where the largest difference anywhere was 1.1e-6
  of the largest value. Before `average_class` computed its coordinates as the MEX
  does (`resample(..., coordinates_in_double=True)`), one voxel at a box face, which
  one particle in 300 reached in one but not in the other, had them differ by up to
  1.6e-4 over the box there and up to 4.6e-4 of the sd inside the mask on max;
- a noise seed is not compared: MATLAB's randn is not numpy's, so it is new noise.

A variable the command left out takes the setup's default of today, but where an
older dynamo_mra_setup.m meant something else, read from the log's last line, as
the setup of today can make it again:

- before 2026-10-19 (c2f442a) the low-passes stepped evenly in frequency, and that
  line named no steps: lowpass_steps 'fourier';
- from 2026-10-22 to 10-24 (22bdc7d) global rounds searched a grid of global_step
  deg, 45 unless given, at the full box ("global round(s) in 45 deg steps"):
  global_step and global_full_box;
- a GPU project (gpus, from 2026-10-26) binned its global rounds by powers of 2:
  global_full_box where that left the full box, as for RyR1's 52 px; another dim
  is not compared, the port does not bin.

Projects with symmetry or an axis offset are not compared: the port models neither.
On max (2026-11-02, before the older setups were read so and before the seeds were
averaged as the MEX turns), of 41 projects 39 were compared: the tables, masks,
seeds' particles and map seeds of all matched, the searches of all but the older
setups' and one GPU project's, which differed as above, and the subset and blob
seeds of most differed as above. Prints one block per project and PASS or FAIL;
exits 1 if any failed. Disposable.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.dynamo_mra_project import CLASS, MraPlan, plan_from_dynamo
from cryoet_pipeline.mra_setup import plan_from_export, read_export, search_rounds

SEED_DIFFERENCE = 1e-5
"""A subset or blob seed's difference from Dynamo's over the whole box: the norm of the
difference over the norm of Dynamo's."""


def _split(command: str) -> list[str]:
    """MATLAB statements separated by ';' outside quotes, braces and brackets."""

    statements, depth, quoted, current = [], 0, False, ""
    for character in command:
        if character == "'":
            quoted = not quoted
        elif not quoted and character in "{[":
            depth += 1
        elif not quoted and character in "}]":
            depth -= 1
        if character == ";" and not quoted and depth == 0:
            statements.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        statements.append(current.strip())
    return statements


def _value(text: str) -> Any:
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith("{"):
        return re.findall(r"'([^']*)'", text)
    if text.startswith("["):
        return [float(token) for token in text.strip("[]").replace(",", " ").split()]
    if text in ("true", "false"):
        return text == "true"
    return float(text)


def setup_variables(log: Path) -> tuple[dict[str, Any], list[str]]:
    """The variables the setup was evaluated with, from the command its log repeats, and
    what an older setup meant by the ones it left out (see the header), in words."""

    lines = log.read_text(errors="replace").splitlines()
    command = next((line for line in lines if "project = '" in line and "fileread(" in line),
                   None)
    if command is None:
        raise ValueError(f"{log}: no setup command in it")
    variables = {}
    for statement in _split(command):
        name, sign, value = statement.partition("=")
        if sign and re.fullmatch(r"\s*\w+\s*", name):
            variables[name.strip()] = _value(value)
    notes = []
    ready = next((line for line in lines if " is ready, " in line), "")
    steps = re.search(r"\((angstrom|fourier) steps\)", ready)
    if "lowpass_steps" not in variables and steps:
        variables["lowpass_steps"] = steps.group(1)
    elif "lowpass_steps" not in variables and ready:
        variables["lowpass_steps"] = "fourier"
        if variables.get("schedule") == "coarse_to_fine":
            notes.append("set up before 2026-10-19: low-passes in even steps of frequency")
    grid = re.search(r"global round\(s\) in (\S+) deg steps", ready)
    if grid and variables.get("global_rounds"):
        variables.setdefault("global_step", float(grid.group(1)))
        variables["global_full_box"] = True
        notes.append(f"set up 2026-10-22 to 10-24: global rounds on a {grid.group(1)} deg grid "
                     "at the full box")
    return variables, notes


def _gpu_dims(options: dict[str, Any], export: Path) -> list[int]:
    """The dims Dynamo's GPU setup gave the global rounds: the box halved while the
    half still holds the dim the CPU's would have had (it resizes only by binning)."""

    particles = read_export(export, options.get("data_folder", "particles"))
    keys = ("schedule", "first_lowpass", "last_lowpass", "diameter", "lowpass_steps",
            "global_rounds", "global_step", "cone_steps")
    rounds = search_rounds(particles.box, particles.pixel_size,
                           **{key: options[key] for key in keys if key in options})
    dims = []
    for searched in (r for r in rounds if r.is_global):
        dim = particles.box
        while dim % 2 == 0 and dim / 2 >= searched.dim:
            dim //= 2
        dims.append(dim)
    return dims


def _options(variables: dict[str, Any], export: Path) -> dict[str, Any]:
    """plan_from_export's keywords for the setup's variables."""

    def path(name: str) -> Path | None:
        if name not in variables or not variables[name]:
            return None
        given = Path(variables[name])
        return given if given.is_absolute() else export / given

    options: dict[str, Any] = {"seeds": variables["seeds"]}
    for name in ("schedule", "lowpass_steps", "data_folder"):
        if name in variables:
            options[name] = variables[name]
    for name in ("first_lowpass", "last_lowpass", "diameter", "mask_radius", "mask_edge",
                 "global_step"):
        if name in variables and variables[name] != []:
            options[name] = float(variables[name])
    if "global_rounds" in variables:
        options["global_rounds"] = int(variables["global_rounds"])
    if "global_full_box" in variables:
        options["global_full_box"] = bool(variables["global_full_box"])
    if "cone_steps" in variables:
        steps = variables["cone_steps"]
        options["cone_steps"] = steps if isinstance(steps, list) else [steps]
    options["reference_file"] = path("reference_file")
    options["mask_file"] = path("mask_file")
    return options


def _rounded(table: np.ndarray) -> np.ndarray:
    """A table as Dynamo writes it: every value to 5 significant digits."""

    return np.vectorize(lambda value: float(f"{value:.5g}"))(table)


def _largest(difference: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(difference).max() / (np.abs(reference).max() or 1.0))


def compare(ours: MraPlan, theirs: MraPlan) -> tuple[list[str], bool]:
    """Lines saying what agrees and what does not, and whether everything that must does."""

    lines, passed = [], True

    def check(label: str, same: bool, detail: str) -> None:
        nonlocal passed
        passed &= same
        lines.append(f"  {label:<15}{detail}{'' if same else '   <- DIFFERS'}")

    check("particles", ours.particles.resolve() == theirs.particles.resolve(),
          f"{theirs.particles}")
    mine, dynamo = _rounded(ours.table), theirs.table
    same_shape = mine.shape == dynamo.shape
    if same_shape:
        differing = ~np.all(np.delete(mine, CLASS, axis=1) == np.delete(dynamo, CLASS, axis=1),
                            axis=1)
        check("starting table", not differing.any(),
              f"{len(dynamo):,} rows; {int(differing.sum())} differ after rounding ours to "
              "5 digits as Dynamo wrote its own")
    else:
        check("starting table", False, f"ours {mine.shape}, Dynamo's {dynamo.shape}")
    mask = _largest(ours.soft_mask - theirs.soft_mask, theirs.soft_mask) \
        if ours.soft_mask.shape == theirs.soft_mask.shape else np.inf
    check("mask", mask == 0, "identical" if mask == 0 else f"largest difference {mask:.2g}")
    pairs = zip(ours.settings, theirs.settings, strict=False)  # the counts are checked below
    searches = [number for number, (a, b) in enumerate(pairs, 1) if a != b]
    check("searches", len(ours.settings) == len(theirs.settings) and not searches
          and ours.mra == theirs.mra,
          f"{len(theirs.settings)} iterations" + (f"; iteration(s) {searches} differ: ours "
          f"{ours.settings[searches[0] - 1]}, Dynamo's {theirs.settings[searches[0] - 1]}"
          if searches else (", the same" if len(ours.settings) == len(theirs.settings)
                            else f", ours {len(ours.settings)}")))
    if theirs.seed_groups is None:
        lines.append("  seed groups    no seed_manifest.txt in the project: not compared")
    else:
        same = (ours.seed_groups is not None and ours.seed_groups.shape == theirs.seed_groups.shape
                and np.array_equal(ours.seed_groups, theirs.seed_groups))
        check("seed groups", same, f"{len(theirs.seed_groups):,} tags, "
              f"{int(np.sum(theirs.seed_groups[:, 1] > 0)):,} of them in a seed")
    kinds = [note.split()[1] if len(note.split()) > 1 else "?" for note in theirs.reference_notes]
    for channel in sorted(theirs.references):
        kind = kinds[channel - 1] if channel <= len(kinds) else "?"
        label = f"seed {channel} {kind}"
        if channel not in ours.references:
            check(label, False, "not in ours")
            continue
        mine_seed, dynamo_seed = ours.references[channel], theirs.references[channel]
        if kind == "noise":
            lines.append(f"  {label:<15}new noise (MATLAB's randn is not numpy's): not compared")
            continue
        difference = mine_seed - dynamo_seed
        if kind == "map":
            largest = _largest(difference, dynamo_seed)
            check(label, largest == 0, "identical" if largest == 0
                  else f"largest difference {largest:.2g} of Dynamo's largest value")
            continue
        inside = theirs.soft_mask > 0.5
        masked = float(np.sqrt(np.mean(difference[inside] ** 2)) / dynamo_seed[inside].std())
        whole = float(np.linalg.norm(difference) / np.linalg.norm(dynamo_seed))
        correlation = float(np.corrcoef(mine_seed[inside], dynamo_seed[inside])[0, 1])
        check(label, whole <= SEED_DIFFERENCE,
              f"the whole map {whole:.1e}, the largest "
              f"{_largest(difference, dynamo_seed):.1e} of its largest value; inside the mask "
              f"rms difference {masked:.1e} of its sd, correlation {correlation:.10f}")
    check("seed notes", [_note(note) for note in ours.reference_notes]
          == [_note(note) for note in theirs.reference_notes], "; ".join(theirs.reference_notes))
    return lines, passed


def _note(line: str) -> list[str]:
    """A seed_kinds.txt line, a map seed's file by its name: Dynamo's setup kept the
    path as given, ours resolves it."""

    parts = line.split(maxsplit=2)
    if len(parts) == 3 and parts[1] == "map":
        parts[2] = Path(parts[2]).name
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("projects", type=Path, nargs="+",
                        help="Dynamo projects scripts/dynamo_mra_setup.m made")
    args = parser.parse_args()
    failed = 0
    for project in args.projects:
        project = project.resolve()
        export = project.parent
        log = export / f"{project.name}_setup.log"
        print(project.name)
        try:
            variables, notes = setup_variables(log)
        except (OSError, ValueError) as error:
            print(f"  not compared: {error}")
            failed += 1
            continue
        shown = {name: value for name, value in variables.items()
                 if name not in ("project", "cores")}
        print(f"  variables      {shown}")
        if str(variables.get("symmetry", "c1")) != "c1" or any(variables.get("axis_offset", [])):
            print("  not compared: symmetry or an axis offset, which the port does not model")
            continue
        try:
            options = _options(variables, export)
            if variables.get("gpus") and options.get("global_rounds"):
                box = read_export(export, options.get("data_folder", "particles")).box
                dims = _gpu_dims(options, export)
                if any(dim != box for dim in dims):
                    print(f"  not compared: Dynamo's GPU setup binned the global rounds to "
                          f"{dims} px, which the port does not")
                    continue
                options["global_full_box"] = True
                notes.append("set up for Dynamo's GPUs: global rounds at the full box, the "
                             "only power-of-2 binning that held their dim")
            for note in notes:
                print(f"  read as        {note}")
            theirs = plan_from_dynamo(project)
            ours = plan_from_export(export, log=lambda _: None, **options)
        except (OSError, ValueError) as error:
            print(f"  not compared: {error}")
            failed += 1
            continue
        lines, passed = compare(ours, theirs)
        print("\n".join(lines))
        print("  PASS" if passed else "  FAIL")
        failed += not passed
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
