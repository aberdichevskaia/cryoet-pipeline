#!/usr/bin/env python3
"""Decoys chosen by template matching: pytom's picks in a phase-randomised tomogram.

    python scripts/null_decoys.py --project outputs/ribo-mdc \\
        --config configs/lanes/ribo-mdc.toml --series TS_0306 --series TS_0463

prints, per series, what it would make and the two commands it would run;
`--execute` makes and runs them, stopping at the first failure:

1. a copy of the tomogram the lane's picks were found in, with every Fourier
   phase replaced by white noise's (cryoet_pipeline.null_volumes), written
   beside it and registered as a full tomogram labelled `<lane>-null`: a child
   of the real one that keeps its tilt angles and views;
2. the lane's pick-particles, as scripts/run_series.py renders it from the
   config, on that copy, with three changes: the label; the template and mask
   the real picks were matched with, named rather than looked up, since no
   template carries the new label; and the best `--count` peaks with no score
   floor, 120 unless told otherwise, as many as the random decoys per tomogram.
   It searches where the real picks were searched: the whole volume when they
   were (the ribosome lanes), and inside the same membrane segmentation when they
   were matched inside one (RyR1), named by its id, since the copy has no
   membranes to segment. That mask is built before anything is made, and must
   admit the fraction of the volume the real picks record admitting, or the
   lane's dilation has changed since they were made;
3. the lane's extract-particles on those picks, under the same label.

`scripts/export_dynamo.py --variant <lane> --decoy-variant <lane>-null` then puts
them after every real particle of an export, as it did the random decoys.

Why. Boxes at random positions (scripts/decoy_picks.py) went to the
noise-seeded channel 86% of the time against 0.5% of the real picks
(2026-10-17): Dynamo tells nothing from a ribosome. pytom's false picks are not
nothing: each is noise that matched the template better than everything around
it, and an average of such boxes looks like the template. These decoys are
chosen the same way, from noise with the tomogram's own spectrum. If the
classification still sends them to the junk channels, it can be trusted to
clean pytom's list; if they stay with the real picks, it cannot, and the
cleaning belongs to picking.

After each pick it prints two comparisons with the real picks. Their scores:
the null's best is what noise of this spectrum reaches against this template
over a volume of the same size. The two pytom job files: every setting the
searches differ in besides the tomogram, which should be none. And the two
search masks, by segmentation and the fraction they admitted.

What the copy does not reproduce is where the variance was. The lamella's
contrast and the vacuum's are spread evenly over the whole volume, so the null
picks come from any depth, and sample the tomogram's average texture rather
than the lamella's. Should Dynamo separate them from the real picks, check that
a difference in spectrum alone could not, before reading it as structure.

Inside a membrane mask the copy is searched where the real tomogram has
membranes, but has none there itself: every real box holds a bilayer and no
decoy box does. A classification mask that takes the membrane in, as a sphere
about a membrane protein does, tells them apart by that alone; read these decoys
with one that leaves the membrane out (scripts/make_mra_mask.py).

A null tomogram registered before from the same tomogram with the same seed is
used again, so a job resubmitted after a failed pick does not make it twice;
`--overwrite` makes it again and passes --overwrite to both commands.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tomllib
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import run_series
import scipy  # type: ignore[import-untyped]
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.picking import tomogram_search_mask
from cryoet_pipeline.backends.subtomogram import _read_pytom_star
from cryoet_pipeline.cli import _template_from_picks
from cryoet_pipeline.models import Artifact, ArtifactKind, RetentionPolicy, StorageRole
from cryoet_pipeline.null_volumes import phase_randomised

Command = list[str]

# What the copy must not inherit: it was not made by IMOD, and these name the real volume.
_REAL_VOLUME_PARAMETERS = ("imod_rec_path", "imod_rec_axis_order", "log_path")

# pytom job settings that differ between the two searches by construction, and
# the whitening filter, which pytom names after the tomogram inside output_dir.
_EXPECTED_JOB_DIFFERENCES = frozenset({"tomogram", "tomo_id", "output_dir", "job_key", "job_stats",
                                       "whitening_filter"})
# The tomogram mask pytom read: pick-particles writes it into a folder of its own
# for every search, so only whether there was one can be compared. That it was the
# same mask is what the two pick lists' segmentation_id and coverage say.
_COMPARED_BY_PRESENCE = frozenset({"tomogram_mask"})


def real_picks(registry: ArtifactRegistry, series: str, lane: str) -> Artifact:
    """The lane's own pick list for `series`: labelled `lane`, and not a decoy list."""

    found = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PICKS)
        if artifact.covers(series)
        and artifact.variant == lane
        and artifact.parameters.get("backend") != "decoy"
    ]
    if len(found) != 1:
        raise SystemExit(f"{series}: expected one pick list labelled {lane!r}, found "
                         f"{[artifact.id for artifact in found] or 'none'}")
    return found[0]


def series_seed(seed: int, series: str) -> list[int]:
    """Each series its own noise, the same on every run: the seed and a checksum of the name."""

    return [seed, zlib.crc32(series.encode())]


def null_tomogram_id(series: str, variant: str) -> str:
    return f"{series}:tomogram:phase-randomised:{variant}"


def null_tomogram_path(real: Artifact, series: str, variant: str) -> Path:
    return real.path.parent / f"{series}_phase_randomised_{variant}.zarr"


def _size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _statistics(source: np.ndarray, null: np.ndarray) -> dict[str, float]:
    """Mean and spread of both volumes and how alike they are: the same, the same, none."""

    means = [float(volume.mean(dtype=np.float64)) for volume in (source, null)]
    spreads = [float(volume.std(dtype=np.float64)) for volume in (source, null)]
    covariance = float(np.mean((source - np.float32(means[0])) * (null - np.float32(means[1])),
                               dtype=np.float64))
    return {
        "source_mean": means[0],
        "source_standard_deviation": spreads[0],
        "mean": means[1],
        "standard_deviation": spreads[1],
        "correlation_with_source": (covariance / (spreads[0] * spreads[1])
                                    if spreads[0] and spreads[1] else float("nan")),
    }


def make_null_tomogram(
    registry: ArtifactRegistry,
    real: Artifact,
    *,
    variant: str,
    seed: int,
    overwrite: bool = False,
    workers: int | None = None,
) -> tuple[Artifact, bool]:
    """Write and register the phase-randomised copy of `real`; say whether it was made now.

    One registered from the same tomogram with the same seed is used again: the
    copy is a function of those two, so making it twice gives the same volume.
    """

    series = str(real.parameters["tilt_series_id"])
    artifact_id = null_tomogram_id(series, variant)
    seeds = series_seed(seed, series)
    try:
        existing: Artifact | None = registry.get(artifact_id)
    except KeyError:
        existing = None
    if existing is not None and not overwrite:
        made_from = (existing.parameters.get("phase_randomised_from"),
                     existing.parameters.get("phase_seed"))
        if made_from != (real.id, seeds):
            raise SystemExit(f"{artifact_id} was made from {made_from[0]} with seed "
                             f"{made_from[1]}, not from {real.id} with seed {seeds}; pass "
                             "--overwrite to make it again")
        if not existing.path.is_dir():
            raise SystemExit(f"{artifact_id} is registered but {existing.path} is gone; pass "
                             "--overwrite to make it again")
        return existing, False

    source: Any = zarr.open(real.path, mode="r")
    volume = np.asarray(source[:], dtype=np.float32)
    null = phase_randomised(volume, np.random.default_rng(seeds), workers=workers)
    statistics = _statistics(volume, null)
    del volume
    path = null_tomogram_path(real, series, variant)
    # Written aside and moved into place, so a job killed while writing leaves no
    # half volume under the name a later run would use again.
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    output: Any = zarr.open(partial, mode="w", shape=null.shape, chunks=source.chunks,
                            dtype=np.float32)
    output.attrs.update({**dict(source.attrs), "phase_randomised_from": str(real.path),
                         "phase_seed": seeds})
    output[:] = null
    if path.exists():
        shutil.rmtree(path)
    partial.rename(path)

    parameters = {key: value for key, value in real.parameters.items()
                  if key not in _REAL_VOLUME_PARAMETERS}
    parameters.update({
        "backend": "phase-randomisation",
        "phase_randomised_from": real.id,
        "phase_seed": seeds,
        "null_statistics": statistics,
    })
    artifact = Artifact(
        id=artifact_id,
        variant=variant,
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        parent_ids=[real.id],
        shape=tuple(int(size) for size in null.shape),
        dtype="float32",
        axis_order=real.axis_order,
        pixel_spacing_angstrom=real.pixel_spacing_angstrom,
        binning=real.binning,
        parameters=parameters,
        software_versions={"numpy": np.__version__, "scipy": scipy.__version__,
                           "zarr": zarr.__version__},
        storage_role=StorageRole.CACHE,
        retention_policy=RetentionPolicy.RECOMPUTE,
        can_recompute=True,
        size_bytes=_size_bytes(path),
        tilt_series_ids=list(real.tilt_series_ids),
    )
    registry.add(artifact, replace=existing is not None)
    return artifact, True


def _set(command: Command, flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def _drop(command: Command, flag: str) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + 2]


def lane_commands(
    cfg: dict[str, Any],
    project: Path,
    series: str,
    *,
    real: Artifact,
    variant: str,
    count: int,
    template: Path,
    mask: Path,
    spherical_mask: bool,
) -> tuple[Command, Command]:
    """The lane's pick and extract commands for `series`, turned onto the null tomogram.

    Refused when the lane no longer describes how the real picks were matched:
    the copy is a control only if it was searched as they were.
    """

    steps = dict(run_series.build_steps(cfg, project, series))
    pick, extract = list(steps["pick-particles"]), list(steps["extract-particles"])
    recorded = real.parameters.get("pytom_low_pass_angstrom")
    lane_low_pass = (float(pick[pick.index("--pytom-low-pass") + 1])
                     if "--pytom-low-pass" in pick else None)
    if lane_low_pass is None:
        # pick-particles would derive one from an alignment carrying the new label,
        # which there is none of; the real picks recorded the one they derived.
        if recorded is None:
            raise SystemExit(f"neither the lane nor {real.id} says which low-pass the real "
                             "picks were matched at")
        _set(pick, "--pytom-low-pass", f"{float(recorded):g}")
    elif recorded is not None and float(recorded) != lane_low_pass:
        raise SystemExit(f"{real.id} was matched at a {recorded} A low-pass and the lane now "
                         f"says {lane_low_pass}; the config has changed since those picks")
    if real.parameters.get("per_tilt_weighting") is not None and (
            "--pytom-per-tilt-weighting" in pick) != bool(real.parameters["per_tilt_weighting"]):
        raise SystemExit(f"{real.id} was matched with per_tilt_weighting "
                         f"{real.parameters['per_tilt_weighting']} and the lane now says "
                         "otherwise; the config has changed since those picks")
    split = real.parameters.get("pytom_volume_split")
    lane_split = (pick[pick.index("--pytom-volume-split") + 1]
                  if "--pytom-volume-split" in pick else "1,1,1")
    if split is not None and lane_split != ",".join(str(part) for part in split):
        raise SystemExit(f"{real.id} was matched in pieces {split} and the lane now says "
                         f"{lane_split}; the config has changed since those picks")

    _set(pick, "--variant", variant)
    _set(pick, "--pytom-number-of-particles", str(count))
    _drop(pick, "--pytom-number-of-false-positives")
    if "--pytom-cut-off" not in pick or float(pick[pick.index("--pytom-cut-off") + 1]) != 0:
        _set(pick, "--pytom-cut-off", "0")
    _set(pick, "--pytom-template", str(template))
    _set(pick, "--pytom-template-mask", str(mask))
    pick.append("--no-pytom-non-spherical-mask" if spherical_mask
                else "--pytom-non-spherical-mask")
    # Where the real search was, by what the real picks record rather than what the
    # lane renders: unmasked, or inside the segmentation they name. Said outright
    # either way: left to its default, pick-particles would look for a segmentation
    # under the new label, which none carries, and refuse when the series has one
    # under another. The dilation stays the lane's; search_mask checks it.
    segmentation_id = real.parameters.get("segmentation_id")
    for flag in ("--use-segmentation-mask", "--no-use-segmentation-mask"):
        while flag in pick:
            pick.remove(flag)
    _drop(pick, "--segmentation-id")
    if segmentation_id is None:
        _drop(pick, "--segmentation-mask-dilation-px")
        pick.append("--no-use-segmentation-mask")
    else:
        pick.extend(["--use-segmentation-mask", "--segmentation-id", str(segmentation_id)])
    _set(extract, "--variant", variant)
    return pick, extract


def search_mask(registry: ArtifactRegistry, real: Artifact, tomogram: Artifact,
                pick: Command) -> str:
    """Where the null search will look, refused unless it is where the real one did.

    The real picks record the segmentation they were searched inside and the
    fraction of the tomogram its mask admitted, but not the margin it was grown
    by, which comes from the config. So the mask `pick` will make is made here,
    from the real tomogram, whose shape the copy has, and its fraction compared:
    the same segmentation grown by the same margin admits the same fraction,
    another margin practically never does.
    """

    segmentation_id = real.parameters.get("segmentation_id")
    if segmentation_id is None:
        return "the whole volume, as the real picks were"
    try:
        segmentation = registry.get(str(segmentation_id))
    except KeyError:
        raise SystemExit(f"{real.id} was matched inside {segmentation_id}, which is no "
                         "longer registered") from None
    recorded = real.parameters.get("tomogram_mask_coverage")
    if not isinstance(recorded, int | float):
        raise SystemExit(f"{real.id} does not record how much of the tomogram its mask "
                         "admitted, so the null search's mask cannot be shown to be the same")
    dilation = (int(pick[pick.index("--segmentation-mask-dilation-px") + 1])
                if "--segmentation-mask-dilation-px" in pick else None)
    coverage = float(tomogram_search_mask(segmentation, tomogram=tomogram,
                                          dilation_px=dilation).mean())
    grown = f"grown by {dilation} px" if dilation else "not grown"
    if not math.isclose(coverage, float(recorded), rel_tol=1e-6):
        raise SystemExit(f"{real.id} was matched inside a mask admitting {float(recorded):.4%} "
                         f"of the tomogram, and {segmentation_id} {grown}, as the lane says "
                         f"now, admits {coverage:.4%}; the config has changed since those picks")
    return (f"inside {segmentation_id} {grown}, {coverage:.2%} of the volume, as the real "
            "picks were")


def template_of(registry: ArtifactRegistry, real: Artifact) -> tuple[Path, Path, bool]:
    """The template and mask files the real picks were matched with, and the mask's shape."""

    pair = _template_from_picks(registry, real)
    if pair is None:
        raise SystemExit(f"{real.id}: no registered template and mask in its lineage or "
                         "under the files it records, so the search cannot be repeated")
    template, mask = pair
    for artifact, key in ((template, "pytom_template_path"), (mask, "pytom_template_mask_path")):
        used = real.parameters.get(key)
        if isinstance(used, str) and Path(used).resolve() != artifact.path.resolve():
            raise SystemExit(f"{real.id} records {key} {used}, not {artifact.path} "
                             f"({artifact.id})")
    spherical = mask.parameters.get("mask_is_spherical")
    if not isinstance(spherical, bool):
        raise SystemExit(f"{mask.id} does not record whether it is spherical")
    return template.path, mask.path, spherical


def job_differences(real_job: Path, null_job: Path) -> list[str]:
    """Every pytom setting the two searches differ in, besides those that must."""

    first, second = (json.loads(path.read_text()) for path in (real_job, null_job))
    differing = []
    for key in sorted(set(first) | set(second)):
        if key in _EXPECTED_JOB_DIFFERENCES or first.get(key) == second.get(key):
            continue
        if key in _COMPARED_BY_PRESENCE and (first.get(key) is None) == (second.get(key) is None):
            continue
        shown = [json.dumps(job.get(key)) for job in (first, second)]
        shown = [text if len(text) <= 60 else text[:57] + "..." for text in shown]
        differing.append(f"{key}: {shown[0]} (real) vs {shown[1]} (null)")
    return differing


def compare_with_real(real: Artifact, null: Artifact) -> None:
    real_scores = np.array([row.lccmax for row in _read_pytom_star(real.path)
                            if row.lccmax is not None])
    null_scores = np.array([row.lccmax for row in _read_pytom_star(null.path)
                            if row.lccmax is not None])
    if real_scores.size and null_scores.size:
        best = float(null_scores.max())
        above = int(np.count_nonzero(real_scores > best))
        print(f"# scores: {null_scores.size} null peaks, best {best:.3f}, median "
              f"{float(np.median(null_scores)):.3f}; real picks median "
              f"{float(np.median(real_scores)):.3f}, {above} of {real_scores.size} above the "
              "null's best")
    jobs = [artifact.parameters.get("pytom_job_json_path") for artifact in (real, null)]
    if all(isinstance(job, str) and Path(job).is_file() for job in jobs):
        differing = job_differences(Path(str(jobs[0])), Path(str(jobs[1])))
        if differing:
            print("# WARNING: the null search differs from the real one besides the tomogram, "
                  "so it does not control for it as it stands:")
            for line in differing:
                print(f"#   {line}")
        else:
            print("# pytom jobs: the same search as the real picks', on the null tomogram")
    else:
        print(f"# pytom jobs not compared: {jobs[0]} or {jobs[1]} is missing")
    masks = [(artifact.parameters.get("segmentation_id"),
              artifact.parameters.get("tomogram_mask_coverage")) for artifact in (real, null)]
    (real_mask, real_coverage), (null_mask, null_coverage) = masks
    same_coverage = real_coverage == null_coverage or (
        isinstance(real_coverage, int | float) and isinstance(null_coverage, int | float)
        and math.isclose(real_coverage, null_coverage, rel_tol=1e-6))
    if real_mask == null_mask and same_coverage:
        print("# search mask: " + ("none, as the real picks'" if real_mask is None else
                                   f"{real_mask}, {float(real_coverage):.2%} of the volume, as "
                                   "the real picks'"))
    else:
        print(f"# WARNING: the null search's mask, {null_mask} admitting {null_coverage}, is "
              f"not the real picks', {real_mask} admitting {real_coverage}")


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
                        help="the lane config the real picks were made with")
    parser.add_argument("--series", action="append", required=True, dest="series")
    parser.add_argument("--count", type=int, default=120,
                        help="best peaks to keep per null tomogram (default 120)")
    parser.add_argument("--variant", default=None, help="label of everything made here "
                        "(default: the lane's label plus -null)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true", help="make and run, not print")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least 1")

    cfg = tomllib.loads(args.config.read_text())
    lane = cfg["lane"]["variant"]
    variant = args.variant or f"{lane}-null"
    if variant == lane:
        parser.error("--variant must differ from the lane's label")
    registry_path = args.project / "artifacts.json"

    for series in args.series:
        # Loaded again for every series: the commands of the one before registered theirs.
        registry = ArtifactRegistry.load(registry_path)
        real = real_picks(registry, series, lane)
        tomogram = registry.get(str(real.parameters["input_tomogram_id"]))
        template, mask, spherical = template_of(registry, real)
        pick, extract = lane_commands(cfg, args.project, series, real=real, variant=variant,
                                      count=args.count, template=template, mask=mask,
                                      spherical_mask=spherical)
        # Before the copy is made: a mask that is not the real search's stops it here.
        where = search_mask(registry, real, tomogram, pick)
        if args.overwrite:
            pick, extract = [*pick, "--overwrite"], [*extract, "--overwrite"]
        print(f"\n# ---- {series}: null decoys beside {real.id} ----")
        print(f"# phase-randomised copy of {tomogram.id} (seed {series_seed(args.seed, series)})"
              f"\n#   {null_tomogram_path(tomogram, series, variant)}"
              f"\n#   registered as {null_tomogram_id(series, variant)}"
              f"\n# searched {where}")
        if not args.execute:
            print(f"\n{run_series.render(pick)}\n\n{run_series.render(extract)}")
            continue

        null, made = make_null_tomogram(registry, tomogram, variant=variant, seed=args.seed,
                                        overwrite=args.overwrite)
        if made:
            registry.commit(registry_path)
            numbers = null.parameters["null_statistics"]
            print(f"# made it: standard deviation {numbers['standard_deviation']:.4g} (the "
                  f"real one's {numbers['source_standard_deviation']:.4g}), correlation with "
                  f"the real one {numbers['correlation_with_source']:+.4f}")
        else:
            print("# registered before from the same tomogram and seed: used again")
        for step, command in (("pick-particles", pick), ("extract-particles", extract)):
            code = run(command)
            if code != 0:
                print(f"\n# FAILED at {step} (exit {code}) for {series}. Fix it and run again "
                      f"with --series {series}; the null tomogram is used again"
                      + ("" if step == "pick-particles" else
                         ", and --overwrite to pick again"), file=sys.stderr)
                return code
            if step == "pick-particles":
                registry = ArtifactRegistry.load(registry_path)
                found = [artifact for artifact in registry.by_kind(ArtifactKind.PICKS)
                         if artifact.covers(series) and artifact.variant == variant]
                if len(found) == 1:
                    compare_with_real(real, found[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
