#!/usr/bin/env python3
"""Pick a lane's tomograms again under a new label, as its picks were made, and compare.

    python scripts/repick.py --project outputs/ribo-mdc \\
        --config configs/lanes/ribo-mdc.toml --series TS_0306 --label bin8-solved

prints, per series, the two commands it would run; `--execute` runs them and then
the comparison, stopping at the first failure:

1. the lane's pick-particles on the lane's tomogram, repeating the search the
   lane's picks record: their template and mask files, their count, their
   membrane mask or none, pytom's seed, and the low-pass, per-tilt weighting and
   split the lane must still give (scripts/null_decoys.py builds the command the
   same way for the null decoys). Its output is labelled `--label`
   (--output-variant), so the old picks stay as they are;
2. the lane's extract-particles on the new picks, under the new label;
3. per series: what the new search's pytom job differs in from the old one's and
   from the null decoys' search (`<lane>-null`); how many of the old picks are
   found again within 5 px and how they scored then and now; and how many picks of
   each list score above the null's best. At the end, the same summed.

`--compare-only` prints the comparison of picks already made, for any series.

Why. Until 6b8e3be (2026-10-18) pick-particles gave pytom the tilt angles of the
tracking stack, the stage's, while every tomogram is reconstructed from the angles
fine alignment solved: on ribo-mdc 9-11 deg away (the lamella's tilt), with steps
of 1.4-3.0 deg where the stage's are 2. pick-particles takes the solved ones now,
so a search repeated with nothing else changed tells what the old angles cost.
The null decoys were picked with the solved angles, so these picks are the ones
they control exactly.

The reading, written before the run. The new search's job should differ from the
old one's in the tilt angles alone, and from the null search's in nothing but the
tomogram (a random seed apart, which pytom uses for random-phase correction only).
Then, summed over the series, the picks above the null's best: more than a tenth
more than the old lists had, and the stage angles cost real picks, so averaging
and classification move to the new lists; within a tenth, they did not matter at
this binning; more than a tenth fewer, and the solved angles search worse than the
stage's, which points at the solved angles themselves.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import null_decoys
import numpy as np
import run_series
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.subtomogram import _read_pytom_star
from cryoet_pipeline.models import Artifact, ArtifactKind

Command = list[str]

REFOUND_PX = 5.0
"""How near a new pick has to be to an old one to count as the same particle found again:
a few voxels of peak wander, far inside the 30 px the search keeps picks apart by."""


def picks_labelled(registry: ArtifactRegistry, series: str, label: str) -> Artifact | None:
    """The one pick list of `series` labelled `label` that is not a random-position decoy list."""

    found = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PICKS)
        if artifact.covers(series)
        and artifact.variant == label
        and artifact.parameters.get("backend") != "decoy"
    ]
    if len(found) > 1:
        raise SystemExit(f"{series}: {len(found)} pick lists labelled {label!r}: "
                         f"{[artifact.id for artifact in found]}")
    return found[0] if found else None


def repick_commands(cfg: dict[str, Any], project: Path, series: str, *,
                    registry: ArtifactRegistry, old: Artifact, label: str
                    ) -> tuple[Command, Command]:
    """The lane's pick and extract for `series`, repeating `old`'s search into `label`."""

    lane = cfg["lane"]["variant"]
    if "number_of_false_positives" in cfg.get("pick", {}):
        # The builder keeps the best peaks with no floor, which is not this lane's rule.
        raise SystemExit("this lane cuts its picks at a false-positive height; repeating "
                         "that is not done here")
    template, mask, spherical = null_decoys.template_of(registry, old)
    count = old.parameters.get("pytom_number_of_particles")
    if not isinstance(count, int):
        count = cfg["pick"].get("number_of_particles")
    if not isinstance(count, int):
        raise SystemExit(f"{old.id} does not record how many peaks it kept, nor does the lane")
    # With the lane's own label the null decoys' builder searches the lane's tomogram.
    pick, extract = null_decoys.lane_commands(
        cfg, project, series, real=old, variant=lane, count=count, template=template,
        mask=mask, spherical_mask=spherical)
    pick = [*pick, "--output-variant", label]
    extract[extract.index("--variant") + 1] = label
    return pick, extract


def search_differences(first: Artifact, second: Artifact,
                       names: tuple[str, str] = ("new", "old")) -> list[str] | None:
    """What two pick lists' pytom jobs differ in besides the tomogram; None without both."""

    jobs = [first.parameters.get("pytom_job_json_path"),
            second.parameters.get("pytom_job_json_path")]
    if not all(isinstance(job, str) and Path(job).is_file() for job in jobs):
        return None
    paths = [Path(str(job)) for job in jobs]
    lines = null_decoys.job_differences(*paths, names=names)
    settings = [json.loads(path.read_text()) for path in paths]
    if not any(setting.get("random_phase_correction") for setting in settings):
        # pytom draws a seed for every search and uses it for random-phase correction only.
        lines = [line for line in lines if not line.startswith("rng_seed:")]
    return lines


@dataclass
class Comparison:
    old: int
    new: int
    refound: int
    null_best: float | None = None
    old_above: int | None = None
    new_above: int | None = None
    refound_above: int | None = None


def _positions_and_scores(picks: Artifact) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_pytom_star(picks.path)
    positions = np.array([[row.x_px, row.y_px, row.z_px] for row in rows], dtype=float)
    scores = np.array([row.lccmax if row.lccmax is not None else np.nan for row in rows])
    return positions.reshape(-1, 3), scores


def compare(old: Artifact, new: Artifact, null: Artifact | None) -> Comparison:
    """Print and return how the new picks stand against the old and against the null."""

    old_at, old_scores = _positions_and_scores(old)
    new_at, new_scores = _positions_and_scores(new)
    if len(new_at):
        distance, nearest = cKDTree(new_at).query(old_at)
        refound = distance <= REFOUND_PX
    else:
        refound, nearest = np.zeros(len(old_at), dtype=bool), np.zeros(len(old_at), dtype=int)
    result = Comparison(len(old_at), len(new_at), int(refound.sum()))
    line = (f"# old picks found again within {REFOUND_PX:g} px: {result.refound} of "
            f"{result.old}")
    if result.refound:
        line += (f"; their scores, median {np.nanmedian(old_scores[refound]):.3f} then and "
                 f"{np.nanmedian(new_scores[nearest[refound]]):.3f} now")
    print(line)
    if null is not None:
        _, null_scores = _positions_and_scores(null)
        if np.isfinite(null_scores).any():
            best = float(np.nanmax(null_scores))
            result.null_best = best
            result.old_above = int(np.count_nonzero(old_scores > best))
            result.new_above = int(np.count_nonzero(new_scores > best))
            result.refound_above = int(np.count_nonzero(refound & (old_scores > best)))
            print(f"# above the null's best ({best:.3f}): {result.old_above} of {result.old} "
                  f"old picks, {result.new_above} of {result.new} new; "
                  f"{result.refound_above} of those old ones found again")
    for name, short, other in (("the old search", "old", old),
                               ("the null search", "null", null)):
        if other is None:
            print(f"# against {name}: no pick list to compare")
            continue
        differing = search_differences(new, other, ("new", short))
        if differing is None:
            print(f"# against {name}: a pytom job file is missing")
        elif not differing:
            print(f"# against {name}: the same search, but for the tomogram")
        else:
            print(f"# against {name}, it differs in:")
            for difference in differing:
                print(f"#   {difference}")
    return result


def summarise(results: dict[str, Comparison]) -> None:
    if not results:
        return
    old = sum(one.old for one in results.values())
    refound = sum(one.refound for one in results.values())
    print(f"\n# all {len(results)} series: {refound} of {old} old picks found again")
    above = [one for one in results.values() if one.old_above is not None]
    if len(above) != len(results):
        print("# not every series has null picks, so the rule below is not read")
        return
    old_above = sum(one.old_above or 0 for one in above)
    new_above = sum(one.new_above or 0 for one in above)
    change = (new_above - old_above) / old_above if old_above else float("nan")
    print(f"# above the null's best: {old_above} old, {new_above} new ({change:+.0%})")
    verdict = ("the stage angles cost real picks" if change > 0.1
               else "the solved angles search worse than the stage's" if change < -0.1
               else "the angles did not matter at this binning")
    print(f"# the rule written before the run (a tenth either way): {verdict}")


def run(command: Command) -> int:
    print(f"\n{run_series.render(command)}", flush=True)
    # No shell, so $VAR in a config value would reach the tool literally.
    return subprocess.run(["cryoet", *(os.path.expandvars(part) for part in command)],
                          check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path,
                        help="the lane config the old picks were made with")
    parser.add_argument("--series", action="append", required=True, dest="series")
    parser.add_argument("--label", required=True, help="label of the new picks, e.g. bin8-solved")
    parser.add_argument("--null-label", default=None,
                        help="label of the null decoys' picks (default: the lane's plus -null)")
    parser.add_argument("--overwrite", action="store_true",
                        help="pass --overwrite to both commands, to pick again under the label")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="run, then compare")
    action.add_argument("--compare-only", action="store_true",
                        help="compare picks made before; run nothing")
    args = parser.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    lane = cfg["lane"]["variant"]
    null_label = args.null_label or f"{lane}-null"
    if args.label in (lane, null_label):
        parser.error(f"--label must differ from {lane!r} and {null_label!r}")
    registry_path = args.project / "artifacts.json"

    results: dict[str, Comparison] = {}
    for series in args.series:
        # Loaded again for every series: the commands of the one before registered theirs.
        registry = ArtifactRegistry.load(registry_path)
        old = picks_labelled(registry, series, lane)
        if old is None:
            raise SystemExit(f"{series}: no pick list labelled {lane!r} to repeat")
        print(f"\n# ---- {series}: {old.id} picked again as {args.label} ----")
        if not args.compare_only:
            pick, extract = repick_commands(cfg, args.project, series, registry=registry,
                                            old=old, label=args.label)
            tomogram = registry.get(str(old.parameters["input_tomogram_id"]))
            print(f"# searched {null_decoys.search_mask(registry, old, tomogram, pick)}")
            if args.overwrite:
                pick, extract = [*pick, "--overwrite"], [*extract, "--overwrite"]
            if not args.execute:
                print(f"\n{run_series.render(pick)}\n\n{run_series.render(extract)}")
                continue
            for step, command in (("pick-particles", pick), ("extract-particles", extract)):
                code = run(command)
                if code != 0:
                    print(f"\n# FAILED at {step} (exit {code}) for {series}. Fix it and run "
                          f"again from --series {series}"
                          + ("" if step == "pick-particles" else " with --overwrite"),
                          file=sys.stderr)
                    summarise(results)
                    return code
            registry = ArtifactRegistry.load(registry_path)
        new = picks_labelled(registry, series, args.label)
        if new is None:
            print(f"# no picks labelled {args.label!r} yet")
            continue
        results[series] = compare(old, new, picks_labelled(registry, series, null_label))
    summarise(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
