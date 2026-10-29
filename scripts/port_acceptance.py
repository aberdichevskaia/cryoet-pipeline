#!/usr/bin/env python3
"""The port's acceptance check: a project Dynamo computed, replayed iteration by iteration.

    python scripts/port_acceptance.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan \\
        --iterations 1 2 3 --particles 32 --device cuda

Each named iteration of a Dynamo MRA project (several channels, or a map seed's
one) is run again from Dynamo's own inputs to it, so each is checked on its own and
no difference compounds, as one does through the class averages of a free run
(mra_port_run.py). The inputs are every searched channel's starting table and
starting reference (results/ite_N/starting_values; where missing, the iteration
before's refined table and average, which they are), the soft mask and the
iteration's card. They go through `dynamo_mra.mra_iteration` as in a pipeline run:
the box resized to the card's dim, each reference prepared, every channel's search,
the shifts brought back to the box, the assignment by the highest CC. The particles
are a seeded sample, the same at every iteration and on every device. The kernel is
--device's: numpy (float64 on --workers processes, every CPU of the job unless
given: the CPU path) or torch on cuda (the job's GPUs), cpu or mps, float32 unless
--dtype float64. Dynamo's answer is the iteration's refined tables.

For each iteration it prints

- each reference and the search mask as we prepare them, against Dynamo's own
  (starting_reference_transformed, mask_transformed), where Dynamo wrote them;
- how far the searches end from Dynamo's poses: rotation (deg), shift (px), CC;
- every search ending more than 0.1 deg or 0.1 px from Dynamo's pose, with three
  CCs: ours, Dynamo's, and ours at Dynamo's orientation with its shift searched
  again. All three alike (within 1e-4) is a near-tie, two poses that score the
  same, as a C1 search of a near-symmetric particle such as RyR1 has one per
  symmetry copy. Ours above the other two: Dynamo's search stopped short of the
  higher peak. Dynamo's above ours at its orientation: a CC we do not reproduce
  there. Ours below ours at Dynamo's orientation: our search missed a peak it
  scores, the port's fault. A CC of exactly 1 on either side is the subpixel
  spline overshooting and set to 1, as Dynamo sets it: its formula is unstable
  and the overshoot does not repeat across machines;
- the sample's classes against Dynamo's;
- unless --skip-averages, the averaging on every particle: each class averaged
  from Dynamo's own poses and members, against Dynamo's average inside r = n/2 - 2,
  and the classes assigned from Dynamo's own CCs, against its classes;
- the search's time, per search and per candidate orientation, beside the
  project's own for the whole iteration (its averaging included) on the cores its
  card names. The numpy kernel starts its processes once per channel, which a
  whole run spreads over every particle and a sample does not: on a short cone
  search that start dominates.

An iteration passes when
- the searches' medians are within 0.01 deg, 0.01 px and 1e-4 of Dynamo's (the free
  run's iteration 1 on g2atan, 2026-10-29, gave 0.002 deg, 0.000 px and 2.5e-6);
- at most 5% of them end elsewhere, each a near-tie, Dynamo stopping short or a
  spline at 1;
- every sampled particle outside Dynamo's class ties between the two channels in
  Dynamo's own CCs;
- the references and the search mask are prepared as Dynamo prepares them;
- the averaging, where run, matches Dynamo's.
The check passes when every iteration does; it prints PASS or FAIL last and exits 1
on FAIL.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.dynamo_mra import assign, mra_iteration
from cryoet_pipeline.dynamo_mra_project import MraPlan, plan_from_dynamo, torch_search
from cryoet_pipeline.reports.dynamo_project import DYNAMO_TIME, read_em, read_map, read_table

TAG, AVERAGED, CC, CLASS = 0, 2, 9, 33
SHIFT, ANGLES, TILTS = slice(3, 6), slice(6, 9), slice(13, 15)

# A search ending this close to Dynamo's pose ended where Dynamo's did.
SAME_DEG = SAME_PX = 0.1
# CCs this close are alike: float32 against Dynamo's tables, which keep 5 digits.
TIE = 1e-4
MEDIAN_DEG, MEDIAN_PX, MEDIAN_CC = 0.01, 0.01, 1e-4
# The share of searches that may end elsewhere, every one of them explained.
APART = 0.05
# A prepared reference against Dynamo's, as a fraction of its std (Dynamo's toys,
# 2026-10-29: 2e-7, its float32 files).
PREPARED = 1e-4
# An average from Dynamo's poses against Dynamo's: correlation, std ratio off 1
# (the toys: 1 - 5e-8 and 1.00000).
AVERAGE_CORRELATION, AVERAGE_STD = 0.99999, 1e-4


@dataclass(frozen=True)
class Iteration:
    """Dynamo's inputs to one iteration and its answer."""

    number: int
    channels: list[int]
    starts: dict[int, dict[int, np.ndarray]]
    """Each channel's starting rows by tag."""
    references: dict[int, np.ndarray]
    """Each channel's starting reference, at the particles' box, before preparation."""
    refined: dict[int, np.ndarray]
    """Each channel's refined table, by tag: Dynamo's answer."""
    transformed: dict[int, np.ndarray]
    """Each channel's reference as Dynamo prepared it, where it wrote it."""
    mask: np.ndarray | None
    """The search mask as Dynamo prepared it, where it wrote it."""


def _angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = core.euler_matrix(first) @ core.euler_matrix(second).T
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1) / 2))))


def _particle(folder: Path, tag: int) -> np.ndarray:
    for extension in ("mrc", "em"):
        path = folder / f"particle_{tag:05d}.{extension}"
        if path.exists():
            return read_map(path).astype(float)
    raise SystemExit(f"no particle {tag} in {folder}")


def _by_tag(table: np.ndarray) -> dict[int, np.ndarray]:
    return {int(row[TAG]): row for row in table}


def _tilts(row: np.ndarray) -> tuple[float, float]:
    low, high = row[TILTS]
    return float(low), float(high)


def _read_iteration(project: Path, plan: MraPlan, number: int) -> Iteration:
    here = project / "results" / f"ite_{number:04d}"
    listed = here / f"currently_surviving_references_ite_{number:04d}.txt"
    if not listed.exists() or number > len(plan.settings):
        raise SystemExit(f"{project.name}: no iteration {number}")
    channels = [int(value) for value in listed.read_text().split()]
    starting = here / "starting_values"
    before = project / "results" / f"ite_{number - 1:04d}" / "averages"
    starts, references, refined, transformed = {}, {}, {}, {}
    for channel in channels:
        name = f"ref_{channel:03d}_ite_{number:04d}"
        answer = here / "averages" / f"refined_table_{name}.tbl"
        if not answer.exists():
            raise SystemExit(f"{project.name}: Dynamo has not finished iteration {number} "
                             f"(no {answer.name})")
        table = read_table(answer)
        refined[channel] = table[np.argsort(table[:, TAG], kind="stable")]
        # What Dynamo starts the iteration from, or where it took it: the run's own
        # starting table and seeds, then the iteration before's results.
        earlier = f"ref_{channel:03d}_ite_{number - 1:04d}"
        path = starting / f"starting_table_{name}.tbl"
        if path.exists():
            starts[channel] = _by_tag(read_table(path))
        elif number == 1:
            starts[channel] = _by_tag(plan.table)
        else:
            starts[channel] = _by_tag(read_table(before / f"refined_table_{earlier}.tbl"))
        path = starting / f"starting_reference_{name}.em"
        if path.exists():
            references[channel] = read_em(path).astype(float)
        elif number == 1:
            references[channel] = plan.references[channel]
        else:
            references[channel] = read_em(before / f"average_{earlier}.em").astype(float)
        path = starting / f"starting_reference_transformed_{name}.em"
        if path.exists():
            transformed[channel] = read_em(path).astype(float)
    path = starting / f"mask_transformed_ite_{number:04d}.em"
    mask = read_em(path).astype(float) if path.exists() else None
    return Iteration(number, channels, starts, references, refined, transformed, mask)


def _candidates(settings: core.SearchSettings, start: np.ndarray, found: np.ndarray) -> int:
    """Orientations one search scores: Dynamo's grid about the start, then each
    refinement's about the winner (the final one stands in for each level's; the
    count barely depends on where a grid sits)."""

    cone, cone_step = settings.cone_aperture, settings.cone_sampling
    inplane, inplane_step = settings.inplane_range, settings.inplane_sampling
    factor = settings.refine_factor
    total = 0
    for level in range(settings.refine + 1):
        total += len(core.angle_list(start if level == 0 else found, cone, cone_step,
                                     inplane, inplane_step))
        cone, inplane = factor * cone_step, factor * inplane_step
        cone_step, inplane_step = cone_step / factor, inplane_step / factor
    return total


def _kind(ours: float, theirs: float, at_theirs: float) -> str:
    """What three CCs make of a search that ended away from Dynamo's pose."""

    if max(ours, theirs) >= 1:
        # The subpixel spline overshot past 1 and was set to 1, as Dynamo sets it
        # (restricted_peak): an unstable formula whose overshoot does not repeat
        # across machines (Dynamo's own 1.00000 on the Mac toy, 2026-10-27).
        return "spline at 1"
    if at_theirs > ours + TIE:
        return "missed peak"
    if theirs > at_theirs + TIE:
        return "not reproduced"
    if ours > max(theirs, at_theirs) + TIE:
        return "Dynamo short"
    return "tie"


def _prepared(iteration: Iteration, plan: MraPlan) -> tuple[list[str], list[str]]:
    """The references and the search mask as mra_iteration prepares them, against Dynamo's."""

    settings = plan.settings[iteration.number - 1]
    box = plan.soft_mask.shape[0]
    small = settings.dim if settings.dim not in (0, box) else 0
    inner = core.downsized(settings, box) if small else settings
    binary = (core.resize_mask(plan.soft_mask, small) if small
              else (plan.soft_mask > 0.5).astype(float))
    lines, failures = [], []
    if iteration.mask is not None:
        differ = int(np.sum(binary != (iteration.mask > 0.5)))
        lines.append(f"search mask {binary.shape[0]}^3: {differ} voxels differ from Dynamo's")
        if differ:
            failures.append(f"the search mask differs from Dynamo's in {differ} voxels")
    worst = {}
    for channel, theirs in iteration.transformed.items():
        reference = iteration.references[channel]
        ours = core.prepare_reference(core.resize_box(reference, small) if small else reference,
                                      binary, inner)
        if ours.shape != theirs.shape:
            failures.append(f"channel {channel}'s reference is {ours.shape[0]}^3, Dynamo's "
                            f"{theirs.shape[0]}^3")
            continue
        worst[channel] = float(np.abs(ours - theirs).max() / theirs.std())
    if worst:
        lines.append("references prepared: largest difference from Dynamo's "
                     + ", ".join(f"{value:.1e}" for value in worst.values())
                     + f" of its std (channels {', '.join(map(str, worst))})")
        failures += [f"channel {channel}'s reference is prepared {value:.1e} std off Dynamo's"
                     for channel, value in worst.items() if value > PREPARED]
    return lines, failures


def _averages(iteration: Iteration, plan: MraPlan, project: Path) -> tuple[list[str], list[str]]:
    """Each class averaged from Dynamo's own poses and members, against Dynamo's average,
    and the assignment from Dynamo's own CCs, against its classes."""

    lines, failures = [], []
    mra = plan.mra[iteration.number - 1]
    tables = iteration.refined
    channels = iteration.channels
    if mra and len(channels) > 1:
        theirs = tables[channels[0]][:, CLASS].astype(int)
        ours = assign({channel: tables[channel][:, CC] for channel in channels})
        differ = int(np.sum(ours != theirs))
        lines.append(f"classes from Dynamo's CCs: {len(theirs) - differ:,} of {len(theirs):,} "
                     "as Dynamo's")
        if differ:
            failures.append(f"{differ} particles assigned from Dynamo's own CCs differ from its "
                            "classes")
    box = plan.soft_mask.shape[0]
    z, y, x = np.indices((box,) * 3) - box / 2
    inside = np.sqrt(x * x + y * y + z * z) <= box / 2 - 2
    folder = plan.particles
    for channel in channels:
        path = (project / "results" / f"ite_{iteration.number:04d}" / "averages"
                / f"average_ref_{channel:03d}_ite_{iteration.number:04d}.em")
        if not path.exists():
            continue
        table = tables[channel]
        rows = table[table[:, CLASS] == channel] if mra else table[table[:, AVERAGED] == 1]
        particles = (_particle(folder, int(tag)) for tag in rows[:, TAG])
        tilts = [_tilts(row) for row in rows]
        if mra:
            ours = core.average_class(particles, rows[:, ANGLES], rows[:, SHIFT], tilts,
                                      plan.soft_mask, fmin=1).average
        else:
            ours = core.average_particles(particles, rows[:, ANGLES], rows[:, SHIFT], tilts,
                                          plan.soft_mask)[0]
        theirs = read_em(path).astype(float)
        correlation = float(np.corrcoef(ours[inside], theirs[inside])[0, 1])
        ratio = float(ours[inside].std() / theirs[inside].std())
        worst = float(np.abs(ours - theirs)[inside].max() / theirs[inside].std())
        lines.append(f"class {channel} averaged from Dynamo's poses ({len(rows):,} particles): "
                     f"correlation {correlation:.7f}, std ratio {ratio:.5f}, largest difference "
                     f"{worst:.1e} std")
        if correlation < AVERAGE_CORRELATION or abs(ratio - 1) > AVERAGE_STD:
            failures.append(f"class {channel}'s average from Dynamo's poses: correlation "
                            f"{correlation:.7f}, std ratio {ratio:.5f}")
    return lines, failures


def _dynamo_seconds(card: dict[str, str]) -> float | None:
    try:
        start = datetime.strptime(card.get("time_start_computation", ""), DYNAMO_TIME)
        end = datetime.strptime(card.get("time_end_iteration", ""), DYNAMO_TIME)
    except ValueError:
        return None
    return (end - start).total_seconds()


def check(iteration: Iteration, plan: MraPlan, project: Path, sample: np.ndarray,
          volumes: dict[int, np.ndarray], search: Callable[..., list], workers: int,
          kernel: str, averages: bool) -> list[str]:
    """One iteration replayed, printed; what fails it."""

    number = iteration.number
    settings = plan.settings[number - 1]
    box = plan.soft_mask.shape[0]
    channels = iteration.channels
    tags = [int(tag) for tag in sample]
    print(f"iteration {number}: {settings.cone_aperture:g} deg cone in {settings.cone_sampling:g} "
          f"deg steps, refine {settings.refine}, low {settings.low:g} px, dim "
          f"{settings.dim or box} of {box}; channels {channels}; {len(tags)} particles")
    lines, failures = _prepared(iteration, plan)

    spent = 0.0

    def timed(*arguments: Any, **options: Any) -> list:
        nonlocal spent
        clock = time.perf_counter()
        try:
            return search(*arguments, **options)
        finally:
            spent += time.perf_counter() - clock

    first = iteration.starts[channels[0]]
    tilts = [_tilts(first[tag]) for tag in tags]
    starts = {channel: (np.array([iteration.starts[channel][tag][ANGLES] for tag in tags]),
                        np.array([iteration.starts[channel][tag][SHIFT] for tag in tags]))
              for channel in channels}
    step = mra_iteration([volumes[tag] for tag in tags], iteration.references, starts, tilts,
                         plan.soft_mask, settings, fmin=1, workers=workers, search=timed,
                         assembly="mra" if plan.mra[number - 1] else "project")
    searching = spent

    truth = {channel: _by_tag(iteration.refined[channel]) for channel in channels}
    distances, apart = [], []
    for channel in channels:
        found = step.searches[channel]
        for index, tag in enumerate(tags):
            row = truth[channel][tag]
            turn = _angle(found.angles[index], row[ANGLES])
            offset = float(np.linalg.norm(found.shifts[index] - row[SHIFT]))
            distances.append((turn, offset, float(found.correlations[index] - row[CC])))
            if turn > SAME_DEG or offset > SAME_PX:
                apart.append((channel, tag, turn, offset, float(found.correlations[index]),
                              float(row[CC])))
    degrees, pixels, ccs = (np.array(column) for column in zip(*distances, strict=True))
    lines.append(f"{len(distances)} searches: rotation median {np.median(degrees):.3f}, max "
                 f"{degrees.max():.3f} deg; shift median {np.median(pixels):.3f}, max "
                 f"{pixels.max():.3f} px; CC median {np.median(np.abs(ccs)):.1e}, max "
                 f"{np.abs(ccs).max():.1e}")
    for name, value, limit, unit in (("rotation", np.median(degrees), MEDIAN_DEG, " deg"),
                                     ("shift", np.median(pixels), MEDIAN_PX, " px"),
                                     ("CC difference", np.median(np.abs(ccs)), MEDIAN_CC, "")):
        if value > limit:
            failures.append(f"median {name} {value:.3g}{unit}, over {limit:g}")

    # Each search away from Dynamo's pose, scored again at Dynamo's orientation (one
    # candidate, the shift searched again) to tell a tie from a fault.
    kinds: Counter[str] = Counter()
    listed = []
    fixed = replace(settings, cone_aperture=0, inplane_range=0, refine=0)
    for channel in channels:
        away = [entry for entry in apart if entry[0] == channel]
        if not away:
            continue
        rows = [truth[channel][entry[1]] for entry in away]
        again = mra_iteration(
            [volumes[entry[1]] for entry in away], {channel: iteration.references[channel]},
            {channel: (np.array([row[ANGLES] for row in rows]),
                       np.array([row[SHIFT] for row in rows]))},
            [_tilts(row) for row in rows], plan.soft_mask, fixed, workers=workers,
            search=search, assembly="mra")
        for entry, at_theirs in zip(away, again.searches[channel].correlations, strict=True):
            _, tag, apart_degrees, apart_pixels, ours, theirs = entry
            kind = _kind(ours, theirs, float(at_theirs))
            kinds[kind] += 1
            listed.append(f"  channel {channel}, tag {tag}: {apart_degrees:.2f} deg and "
                          f"{apart_pixels:.2f} px from Dynamo's pose; CC ours {ours:.5f}, "
                          f"Dynamo's {theirs:.5f}, ours at its orientation {at_theirs:.5f}: "
                          f"{kind}")
    if apart:
        lines.append(f"{len(apart)} of {len(distances)} searches end more than {SAME_DEG:g} deg "
                     f"or {SAME_PX:g} px from Dynamo's pose: "
                     + ", ".join(f"{count} {kind}" for kind, count in sorted(kinds.items())))
        lines += listed[:20] + ([f"  and {len(listed) - 20} more"] if len(listed) > 20 else [])
    if kinds["missed peak"] or kinds["not reproduced"]:
        failures.append(f"{kinds['missed peak']} searches missed a peak they score, "
                        f"{kinds['not reproduced']} do not reproduce Dynamo's CC at its pose")
    if len(apart) > APART * len(distances):
        failures.append(f"{len(apart)} of {len(distances)} searches end elsewhere, over "
                        f"{APART:.0%}")

    if len(channels) > 1:
        # A particle may land in another class only where Dynamo's own CCs for the two
        # channels tie.
        dynamos = {tag: int(truth[channels[0]][tag][CLASS]) for tag in tags}
        moved = [(tag, int(mine)) for tag, mine in zip(tags, step.assignment, strict=True)
                 if int(mine) != dynamos[tag]]
        untied = [tag for tag, mine in moved
                  if abs(float(truth[mine][tag][CC]) - float(truth[dynamos[tag]][tag][CC])) > TIE]
        lines.append(f"sample's classes: {len(tags) - len(moved)} of {len(tags)} as Dynamo's"
                     + (f"; the others tie in Dynamo's CCs: {len(moved) - len(untied)} of "
                        f"{len(moved)}" if moved else ""))
        if untied:
            failures.append(f"{len(untied)} sampled particles leave Dynamo's class without a tie "
                            "in its CCs")

    if averages:
        more, wrong = _averages(iteration, plan, project)
        lines += more
        failures += wrong

    searches = len(channels) * len(tags)
    count = _candidates(settings, starts[channels[0]][0][0], step.searches[channels[0]].angles[0])
    timing = (f"time: ours {searching / searches:.2f} s a search, "
              f"{1000 * searching / (searches * count):.3f} ms a candidate ({count:,} a search), "
              f"{kernel}")
    card = plan.cards[number - 1]
    seconds = _dynamo_seconds(card)
    if seconds is not None:
        total = len(plan.table) * len(channels)
        cores = card.get("how_many_processors")
        timing += (f"; the project's own run {seconds / total:.2f} s a search"
                   + (f" on {cores} cores" if cores else "")
                   + f" ({card.get('destination', '?')}; {seconds / 60:.1f} min for {total:,} "
                   "searches, its averaging included)")
    lines.append(timing)
    for line in lines:
        print(f"  {line}")
    print(f"  {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="a Dynamo MRA project Dynamo computed")
    parser.add_argument("--iterations", type=int, nargs="+", default=None,
                        help="the iterations to replay; default every one Dynamo finished")
    parser.add_argument("--particles", type=int, default=32,
                        help="sample size, at least 2; 0 for every particle")
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", default="numpy",
                        help="numpy for the numpy kernel; cuda, cpu or mps for the torch one")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--chunk", type=int, default=256, help="torch: candidates per batch")
    parser.add_argument("--workers", type=int, default=core.available_cpus(),
                        help="numpy: processes; default the job's CPUs")
    parser.add_argument("--gpus", type=int, default=0,
                        help="torch on cuda: how many GPUs; 0 (default) for all the job sees")
    parser.add_argument("--per-gpu", type=int, default=1, help="torch: processes on each device")
    parser.add_argument("--skip-averages", action="store_true",
                        help="leave out the averaging, which runs on the CPU whatever the device")
    args = parser.parse_args()
    if args.particles == 1 or args.particles < 0:
        parser.error("--particles is at least 2, or 0 for every particle")

    project = args.project.resolve()
    plan = plan_from_dynamo(project)
    numbers = args.iterations or [
        number for number in range(1, len(plan.settings) + 1)
        if any((project / "results" / f"ite_{number:04d}" / "averages").glob("refined_table_*"))]
    tags = plan.table[:, TAG].astype(int)
    rng = np.random.default_rng(args.seed)
    size = len(tags) if args.particles == 0 else min(args.particles, len(tags))
    sample = np.sort(rng.choice(tags, size=size, replace=False))
    volumes = {int(tag): _particle(plan.particles, int(tag)) for tag in sample}
    if args.device == "numpy":
        search: Callable[..., list] = core.search_particles
        kernel = f"numpy float64, {args.workers} workers"
    else:
        devices = max(1, args.gpus)
        if args.device == "cuda" and args.gpus == 0:
            import torch

            devices = max(1, torch.cuda.device_count())
        search = torch_search(args.device, args.dtype, args.chunk, devices=devices,
                              per_device=args.per_gpu)
        kernel = f"torch {args.device} {args.dtype} x{devices}"
        if args.per_gpu > 1:
            kernel += f", {args.per_gpu} processes each"
    print(f"{project.name}: iterations {', '.join(map(str, numbers))}; {size} of {len(tags):,} "
          f"particles (seed {args.seed}); {kernel}")
    failed: dict[int, list[str]] = {}
    try:
        for number in numbers:
            iteration = _read_iteration(project, plan, number)
            failures = check(iteration, plan, project, sample, volumes, search, args.workers,
                             kernel, not args.skip_averages)
            if failures:
                failed[number] = failures
    finally:
        close = getattr(search, "close", None)
        if callable(close):
            close()
    if failed:
        print("FAIL: " + "; ".join(f"iteration {number}: {', '.join(failures)}"
                                   for number, failures in failed.items()))
        return 1
    print(f"PASS: {project.name}, iterations {', '.join(map(str, numbers))}, {kernel}, "
          f"{size} particles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
