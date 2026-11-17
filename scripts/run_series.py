#!/usr/bin/env python3
"""Emit the pipeline commands for one or more tilt-series, from a lane config.

Prints by default; `--execute` runs them and stops at the first failure, naming
the command to resume from.

    python3 scripts/run_series.py --project outputs/ribo \
        --config configs/lanes/ribo-bin8.toml \
        --series TS_0005 --series TS_0006

    # just the tail, after fixing something upstream
    python3 scripts/run_series.py ... --from prepare-template

What is durable here and what is not:

* the **config** is durable — it is the parameter set for a lane, and it becomes
  the parameter set a plan node carries once the data model grows one;
* the **step order** is code, on purpose, so that this file cannot quietly turn
  into a workflow engine;
* this **script** is disposable. It is a command generator, not a scheduler: no
  resume, no dependency resolution, no state inspection. Skipping work you have
  already done is done by typing fewer steps, because every command resolves its
  inputs from the registry.

**Delete this file when `cryoet execute-run` exists.** At that point the step
list below becomes the canned chain in the step registry, driven by run records,
and this generator has nothing left to do. It is a draft of that deliverable, not
a permanent layer.

**It must never gain**, because each of these is the first commit of a second
workflow engine competing with the one in docs/pipeline_engine.md:

* skip-if-output-exists — that is fingerprinting, and a half-version of it now
  means wrong cache semantics later;
* retries or failure policy of any kind;
* running series in parallel;
* the step order moving into the config — order is the plan, and the plan belongs
  in the data model.

If one of those is genuinely needed, that is the signal to build `execute-run`
rather than to extend this.

The `--project` layout it assumes is the one every command already uses:

    <project>/artifacts.json          the registry
    <project>/manifests/<series>.json the manifest

Several invocations may share a project, one per series — the parallelism is the
scheduler's, not this script's: each command merges what it registered into the
registry under a lock when it finishes, instead of writing back the copy it
loaded. Two invocations on the same series must not overlap, because they write
the same files.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Command = list[str]
# Where the lane files are, for the hint when a config path names no file.
LANES = Path(__file__).resolve().parents[1] / "configs" / "lanes"


def flag(name: str, value: Any) -> Command:
    """Render one option, dropping it when the value is absent."""
    if value is None:
        return []
    if isinstance(value, bool):
        return [f"--{name}"] if value else []
    return [f"--{name}", str(value)]


def project_size_bytes(project: Path) -> int:
    """How much the project occupies, for the per-step delta.

    Walked rather than asked of the filesystem because the interesting number is
    what this project holds, not what the volume has free — and on a shared
    quota the latter moves for reasons that have nothing to do with this run.
    Other jobs writing into the same project still land in it, so a step's
    delta is its own only while it runs alone.
    """

    total = 0
    for path in project.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            # A temporary file a step is still writing may vanish mid-walk.
            continue
    return total


def record_timing(
    project: Path,
    *,
    series: str,
    step: str,
    lane: str,
    variant: str,
    seconds: float,
    disk_delta_bytes: int,
    exit_code: int,
    finished_at: datetime | None = None,
) -> None:
    """Append one measured step to the project's timing table.

    Appended to a file rather than printed only, because the numbers are needed
    to size a sandbox and a quota request, and scrollback is not a table. One
    JSON object per line so an interrupted run leaves every step it did finish.
    `finished_at` is when the step's command ended, now if not given; the
    report reads the step's span as `finished_at` less `seconds`, to tell which
    run wrote the files it shows.

    This is a stand-in for what the `run` entity will record properly, and it is
    deliberately not a substitute: no fingerprint, no parameters, no attribution.
    """

    row = {
        "series": series,
        "step": step,
        # Which lane ran it. One project can hold several, and a labelled step
        # run by one of them is not that step of another (scripts/show_project.py).
        "lane": lane,
        "variant": variant,
        "seconds": round(seconds, 1),
        "disk_delta_bytes": disk_delta_bytes,
        "exit_code": exit_code,
        "finished_at": (finished_at or datetime.now(UTC)).isoformat(timespec="seconds"),
    }
    destination = project / "step_timings.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


INPUTS = ("frames", "stack", "tomograms", "picks", "particles")
# Inputs made elsewhere and entered by an import (docs/entry_and_exit.md): tomograms,
# picked here; tomograms with their picks; or particles with their table. Their
# series begin at the import, so the sections of the steps before it have nothing to
# act on.
IMPORTED_INPUTS = ("tomograms", "picks", "particles")
# The steps that enter such files. An import is never written over: the same files
# with the same answers are imported once, and other answers go under another
# [lane] variant. So --overwrite, which every other step takes, is not given to them.
IMPORT_STEPS = frozenset({"import-tomogram", "import-picks", "import-particles"})
# What each input enters as, said in the refusal of a section it does not read.
ENTERS_AS = {
    "tomograms": "tomograms reconstructed elsewhere, and no step before picking runs",
    "picks": "tomograms with their picks, and no step before extraction runs",
    "particles": "particles with their table, and no step before the MRA runs",
}
# The sections a lane of an imported input reads, besides [lane], [run], [data] and
# [mra]. A section outside them is refused, as an unread key is: a [ctf] left in a
# tomograms lane would look as if it applied.
INPUT_SECTIONS = {
    "tomograms": frozenset({"segment", "template", "pick", "extract", "executables"}),
    "picks": frozenset({"pick", "extract"}),
    "particles": frozenset({"pick"}),
}
# [pick] and [executables] in those lanes: picks and particles are imported with
# the particle's diameter only (the check that picks land on particles, and the
# MRA's default), and a tomograms lane runs pytom alone.
INPUT_KEYS = {
    ("picks", "pick"): frozenset({"particle_diameter_angstrom"}),
    ("particles", "pick"): frozenset({"particle_diameter_angstrom"}),
    ("tomograms", "executables"): frozenset({"pytom_create_template", "pytom_create_mask",
                                             "pytom_match_template",
                                             "pytom_extract_candidates"}),
}
# [data] by input. The tomogram keys are import-tomogram's questions; the picks keys
# import-picks', of which the ones after picks_format belong to one format each; the
# particles keys import-particles'.
TOMOGRAM_DATA = frozenset({"tomogram", "tilt_angles", "orientation", "processing",
                           "tomogram_pixel_size_angstrom"})
PICKS_DATA = frozenset({"picks", "picks_format", "picked_in", "template_box_px", "csv_columns",
                        "coordinate_units", "coordinate_origin", "relion_tomo_name",
                        "relion_tomograms_star", "relion_tilt_series_pixel_size_angstrom",
                        "imod_object", "dynamo_tomograms"})
PARTICLES_DATA = frozenset({"particles_folder", "particles_table", "particles_series",
                            "particles_pixel_size_angstrom"})
DATA_KEYS = {
    "frames": frozenset({"frames", "mdocs", "frames_pattern", "pixel_size_angstrom"}),
    "stack": frozenset({"mdoc", "stack"}),
    "tomograms": TOMOGRAM_DATA,
    "picks": TOMOGRAM_DATA | PICKS_DATA,
    "particles": PARTICLES_DATA,
}
# The picks keys each format reads (picks_import.import_picks); it passes over the
# others without a word, so a lane refuses them. picked_in is the orientation of
# the volume picked, which RELION 5's centred coordinates do not have.
FORMAT_KEYS = {
    "pytom-star": frozenset({"template_box_px", "picked_in"}),
    "csv": frozenset({"csv_columns", "coordinate_units", "coordinate_origin", "picked_in"}),
    "relion5-star": frozenset({"relion_tomo_name", "relion_tomograms_star",
                               "relion_tilt_series_pixel_size_angstrom"}),
    "imod-model": frozenset({"imod_object", "picked_in"}),
    "dynamo-table": frozenset({"dynamo_tomograms", "picked_in"}),
}
# The seed and the tracking backend come in pairs: beads are seeded and tracked
# as beads, patches as patches. A mixed pair has no defined meaning.
FIDUCIAL_PAIRS = {
    "imod-autofidseed": "imod-beadtrack",
    "imod-patch-seed": "imod-patch-track",
}
ALIGNMENTS = {"imod-autofidseed": "beads", "imod-patch-seed": "patches"}
# [fine_align] backend. AreTomo2 aligns from the tilt stack itself, so its lanes
# have no [fiducials] and skip the two fiducial steps.
FINE_BACKENDS = ("imod-tiltalign", "aretomo")
ARETOMO_FINE_KEYS = frozenset({"aretomo_align_z_nm", "aretomo_tilt_cor",
                               "aretomo_dark_tol", "aretomo_gpu"})
TILTALIGN_FINE_KEYS = frozenset({"robust_fitting", "auto_prune_outliers",
                                 "max_pruned_fraction", "surfaces_to_analyze"})

# The combinations of input, alignment and membranes that have run on real data,
# as docs/agent/pipeline_map.md ("Which combinations have run on real data") and
# the README list them; tests/test_submit_lanes.py holds pipeline_map.md to this.
# submit_lanes.py warns a lane of any other combination, on its dry run and its
# submission, that it is the first. The frames here were MRC and TIFF movies: a
# lane of EER movies is "EER frames" (lane_combination), which has run in none.
RUN_ON_REAL_DATA = frozenset({
    ("stack", "patches", "no membranes"),  # ribo-bin8.toml, EMPIAR-11538
    ("frames", "beads", "membranes"),  # gag-bin8.toml, EMPIAR-10164; the RyR1 lanes at MDC
})

# Every key a lane config may set, by section. Keys are read by name, so a
# misspelt one used to be skipped without a word: its flag was not rendered and
# the step ran with its default. Nothing outside this table is accepted, so a key
# a step starts reading goes in here in the same change.
LANE_KEYS: dict[str, frozenset[str]] = {
    # [run] and its [[run.stage]] tables are read by scripts/submit_lanes.py, and
    # the frames half of [data] by `cryoet init`; the rest by build_steps below.
    "lane": frozenset({"name", "input", "variant"}),
    "run": frozenset({"project", "series", "stage"}),
    "data": frozenset().union(*DATA_KEYS.values()),
    "motion": frozenset({"backend", "storage_policy", "gpu", "patch_x", "patch_y",
                         "pixel_size_angstrom", "gain_reference", "gain_rotation",
                         "gain_flip", "ftbin", "eer_sampling", "eer_fractions"}),
    "binning": frozenset({"ctf_fit", "tomogram"}),
    "align": frozenset({"binning", "min_std_ratio", "tilt_axis_angle",
                        "exclude_tilt_angles"}),
    "qc_coarse": frozenset({"preview_binning", "residual_warning_px"}),
    "fiducials": frozenset({
        "seed_backend", "track_backend",
        "tracking_binning", "fiducial_diameter_nm", "raw_pixel_spacing_angstrom",
        "target_beads", "rounds",
        "pre_binning", "num_patches_x", "num_patches_y", "patch_overlap",
        "patch_margin_px", "patch_filter_sigma1", "patch_filter_radius2",
        "patch_filter_sigma2",
        "min_tracked_fiducials", "coverage_warning", "coverage_failure",
    }),
    "fine_align": frozenset({"backend"}) | TILTALIGN_FINE_KEYS | ARETOMO_FINE_KEYS,
    "reconstruct": frozenset({"thickness"}),
    "ctf": frozenset({"voltage_kv", "spherical_aberration_mm", "amplitude_contrast",
                      "min_defocus_angstrom", "max_defocus_angstrom",
                      "defocus_search_window_angstrom", "defocus_step_angstrom",
                      "max_excluded_views", "determine_sample_tilt"}),
    "segment": frozenset({"executable", "model", "sliding_window_size",
                          "test_time_augmentation"}),
    "template": frozenset({"emdb_id", "source_map", "particle_diameter_angstrom",
                           "low_pass_angstrom", "mask_radius_px", "mask_sigma_px",
                           "box_size_px"}),
    "pick": frozenset({
        "particle_diameter_angstrom", "extract_particle_diameter_angstrom",
        "z_axis_symmetry", "number_of_particles", "expected_particles_per_tomogram",
        "number_of_false_positives",
        "cut_off", "tophat_filter", "low_pass_angstrom", "random_phase_correction",
        "ctf_weighting", "per_tilt_weighting", "spectral_whitening", "gpu_ids",
        "volume_split", "use_segmentation_mask", "segmentation_mask_dilation_px",
    }),
    "extract": frozenset({"box_size_px", "pad_edges"}),
    # Read by scripts/submit_lanes.py and the MRA job through
    # cryoet_pipeline.mra_lane.MraSettings, whose fields these are
    # (tests/test_mra_lane.py holds the two together); this file imports no package.
    "mra": frozenset({
        "name", "seeds", "reference", "schedule",
        "first_lowpass_nyquist", "last_lowpass_nyquist",
        "first_lowpass_angstrom", "last_lowpass_angstrom", "lowpass_steps",
        "diameter_angstrom", "global_rounds", "global_step_deg", "global_full_box",
        "cone_steps", "antialias", "shift_limit_px",
        "mask_radius_px", "mask_edge_px", "mask_file", "decoy_label", "subset_particles",
        "from_run", "from_classes", "from_iteration", "start_poses",
    }),
    "executables": frozenset({"motioncor3", "ctffind", "aretomo", "pytom_create_template",
                              "pytom_create_mask", "pytom_match_template",
                              "pytom_extract_candidates"}),
}
# The MRA: the last step of every lane, after extract-particles, on all the lane's
# series together (docs/mra_in_lanes.md). It is no step of build_steps, which renders
# one series; a [[run.stage]] that runs it runs nothing else, as a job of
# scripts/mra_port_run.sbatch that scripts/submit_lanes.py submits.
MRA_STEP = "mra"
STAGE_KEYS = frozenset({"name", "from", "until", "time", "sbatch"})
# Keys that were renamed, with what replaced them.
RENAMED = {
    ("lane", "chain"): (
        "is now [lane] input: 'frames' or 'stack'. Alignment and segmentation are "
        "chosen by [fiducials] and [segment], not by the chain"
    ),
}


def _did_you_mean(name: str, choices: frozenset[str], brackets: bool = False) -> str:
    close = difflib.get_close_matches(name, sorted(choices), n=1)
    if not close:
        return ""
    return f": did you mean [{close[0]}]?" if brackets else f": did you mean {close[0]}?"


def _belongs_in(key: str) -> str:
    homes = [f"[{section}]" for section, keys in LANE_KEYS.items() if key in keys]
    return f": it belongs in {' or '.join(homes)}" if homes else ""


def unknown_keys(cfg: dict[str, Any]) -> list[str]:
    """Every section and key nothing reads, each with what it was likely meant to
    be. Empty when every key in the config reaches a step."""

    found: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            found.append(f"{section} is set above every section{_belongs_in(section)}")
            continue
        if section not in LANE_KEYS:
            found.append(
                f"[{section}] is not a section of a lane config"
                + _did_you_mean(section, frozenset(LANE_KEYS), brackets=True)
            )
            continue
        for key, value in values.items():
            if (section, key) in RENAMED:
                found.append(f"[{section}] {key} {RENAMED[section, key]}")
            elif key not in LANE_KEYS[section]:
                found.append(
                    f"[{section}] {key}"
                    + (_belongs_in(key) or _did_you_mean(key, LANE_KEYS[section]))
                )
            elif (section, key) == ("run", "stage") and isinstance(value, list):
                for stage in value:
                    if not isinstance(stage, dict):
                        continue
                    for stage_key in sorted(set(stage) - STAGE_KEYS):
                        found.append(
                            f"[[run.stage]] {stage.get('name', '?')!r}: {stage_key}"
                            + _did_you_mean(stage_key, STAGE_KEYS)
                        )
    return found


def read_config(path: Path) -> tuple[str, dict[str, Any]]:
    """A lane config's text and its TOML. A path that names no file is refused
    with the lane file it was probably meant to be, rather than a traceback."""
    try:
        text = path.read_text()
    except IsADirectoryError:
        inside = ", ".join(sorted(other.name for other in path.glob("*.toml"))) or "none"
        raise SystemExit(
            f"{path} is a folder, not a lane config; the configs in it: {inside}"
        ) from None
    except FileNotFoundError:
        raise SystemExit(f"no lane config {path}{_lane_file_hint(path)}") from None
    try:
        return text, tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise SystemExit(f"{path} is not valid TOML: {error}") from None


def _lane_file_hint(path: Path) -> str:
    """Where a config path that names no file probably meant: a file with a close
    name in its folder, or, for a bare name such as `ribo-mdc`, in configs/lanes."""
    stem = path.stem if path.suffix == ".toml" else path.name
    bare = len(path.parts) == 1
    folder = LANES if bare else path.parent
    if not folder.is_dir():
        return f": there is no folder {folder}"
    shown = Path("configs", "lanes") if bare else folder
    stems = sorted(other.stem for other in folder.glob("*.toml"))
    # Stems, not file names: a shared ".toml" would make unrelated names look close.
    close = [stem] if stem in stems else difflib.get_close_matches(stem, stems, n=3, cutoff=0.75)
    if close:
        return "; did you mean " + " or ".join(str(shown / f"{one}.toml") for one in close) + "?"
    if bare:
        return "; the lane files are " + ", ".join(str(shown / f"{one}.toml") for one in stems)
    return ""


def check_keys(cfg: dict[str, Any], config: Path | None = None) -> None:
    """Refuse a config with keys nothing reads, naming all of them at once."""
    found = unknown_keys(cfg)
    if found:
        where = f"{config}: " if config is not None else ""
        raise SystemExit(
            f"{where}the config has keys nothing reads, so they would be ignored:\n  "
            + "\n  ".join(found)
        )


def _require(cfg: dict[str, Any], section: str, keys: tuple[str, ...], why: str) -> dict[str, Any]:
    """One config section with the keys the chain reads from it, or a refusal naming them."""
    values = cfg.get(section)
    if not isinstance(values, dict):
        raise SystemExit(f"the config has no [{section}] section, which {why} needs")
    missing = [key for key in keys if key not in values]
    if missing:
        raise SystemExit(f"[{section}] is missing {', '.join(missing)}, which {why} needs")
    return values


def lane_combination(
    cfg: dict[str, Any], movies: frozenset[str] = frozenset()
) -> tuple[str, str, str]:
    """The lane's input, alignment and membranes, named as pipeline_map.md names them.

    `movies` are the formats of a frames lane's movies (submit_lanes.movie_formats).
    EER movies make the input "EER frames", an input of its own: the frames lanes
    that have run had MRC and TIFF movies, and no lane of EER movies has run yet.
    An imported input has no alignment here, and picks and particles no membranes:
    those parts are empty.
    """
    source = lane_input(cfg)
    if source == "tomograms":
        return source, "", "membranes" if "segment" in cfg else "no membranes"
    if source in IMPORTED_INPUTS:
        return source, "", ""
    seed = cfg.get("fiducials", {}).get("seed_backend")
    alignment = ALIGNMENTS.get(seed, str(seed))
    if cfg.get("fine_align", {}).get("backend") == "aretomo":
        alignment = "AreTomo"
    if source == "frames" and "eer" in movies:
        source = "EER frames"
    return source, alignment, "membranes" if "segment" in cfg else "no membranes"


def combination_text(combination: tuple[str, str, str]) -> str:
    """`frames + patches, no membranes`, `frames + beads + membranes`, `tomograms, no
    membranes`, `picks`."""
    source, alignment, membranes = combination
    text = f"{source} + {alignment}" if alignment else source
    if membranes == "membranes":
        return f"{text} + membranes"
    return f"{text}, {membranes}" if membranes else text


def lane_input(cfg: dict[str, Any]) -> str:
    """How the series enter the project: `frames` (movies, through init), `stack`, or
    one of the IMPORTED_INPUTS."""
    lane = cfg.get("lane", {})
    if "chain" in lane:
        raise SystemExit(f"[lane] chain {RENAMED['lane', 'chain']}")
    value = lane.get("input")
    if value not in INPUTS:
        raise SystemExit(f"[lane] input must be one of {', '.join(INPUTS)}, not {value!r}")
    return value


def input_problems(cfg: dict[str, Any], source: str) -> list[str]:
    """The sections and keys that a lane of input `source` does not read."""

    found = []
    data = cfg.get("data")
    if isinstance(data, dict):
        for key in sorted(set(data) - DATA_KEYS[source]):
            homes = " or ".join(repr(name) for name, keys in DATA_KEYS.items() if key in keys)
            found.append(f"[data] {key}, read with input {homes}")
        chosen = data.get("picks_format")
        if source == "picks" and chosen in FORMAT_KEYS:
            for key in sorted(set(data) & (PICKS_DATA - FORMAT_KEYS[chosen]
                                           - {"picks", "picks_format"})):
                homes = " or ".join(name for name, keys in FORMAT_KEYS.items() if key in keys)
                found.append(f"[data] {key}, read with picks_format {homes}")
    if source in INPUT_SECTIONS:
        read = INPUT_SECTIONS[source] | {"lane", "run", "data", "mra"}
        found += [f"[{section}]" for section in sorted(set(cfg) - read)]
        for (name, section), keys in INPUT_KEYS.items():
            values = cfg.get(section)
            if name == source and isinstance(values, dict):
                found += [f"[{section}] {key}" for key in sorted(set(values) - keys)]
    return found


def check_input_keys(cfg: dict[str, Any], source: str) -> None:
    """Refuse what the lane's input does not read, naming all of it at once."""

    found = input_problems(cfg, source)
    if found:
        why = f"; its series enter as {ENTERS_AS[source]}" if source in ENTERS_AS else ""
        raise SystemExit(
            f"a lane of input {source!r} does not read these, so they would be ignored{why}:"
            "\n  " + "\n  ".join(found)
        )


def build_steps(cfg: dict[str, Any], project: Path, series: str) -> list[tuple[str, Command]]:
    """The lane's chain, in the one order this file fixes.

    The config chooses three things and nothing about the order:

    * `[lane] input` — `frames`: movies, motion-corrected here, after `init`
      wrote the manifests (project-level, so not a step); `stack`: one assembled
      stack per series, imported here; `tomograms`, `picks` or `particles`: files
      made elsewhere, entered by their import, and the chain from there
      (_imported_steps);
    * `[fiducials]` backends — gold beads or patch tracking;
    * `[segment]` — present: membranes are segmented after reconstruction, and
      picking may use them as a mask; absent: the step is left out.

    A flag is rendered only when its value is in the config, so one builder
    serves every combination: bead-only options never reach a patch lane. A key
    that no step reads is refused rather than skipped.
    """

    check_keys(cfg)
    source = lane_input(cfg)
    check_input_keys(cfg, source)
    manifest = project / "manifests" / f"{series}.json"
    registry = project / "artifacts.json"
    common = ["--manifest", str(manifest), "--registry", str(registry), "--out", str(project)]

    lane = _require(cfg, "lane", ("variant",), "every lane")["variant"]
    if source in IMPORTED_INPUTS:
        return _imported_steps(cfg, project, series, source, common, lane)
    binning = _require(cfg, "binning", ("ctf_fit", "tomogram"), "every lane that reconstructs")
    align = _require(cfg, "align", ("binning", "min_std_ratio"), "every lane that reconstructs")
    qc = _require(cfg, "qc_coarse", ("preview_binning", "residual_warning_px"),
                  "every lane that reconstructs")
    fine = cfg.get("fine_align", {})
    fine_backend = fine.get("backend", "imod-tiltalign")
    if fine_backend not in FINE_BACKENDS:
        raise SystemExit(
            f"[fine_align] backend must be one of {', '.join(FINE_BACKENDS)}, "
            f"not {fine_backend!r}"
        )
    aretomo = fine_backend == "aretomo"
    # Keys of the other backend would reach no step, so they are refused rather
    # than skipped, as every unread key is.
    stray = sorted(set(fine) & (TILTALIGN_FINE_KEYS if aretomo else ARETOMO_FINE_KEYS))
    if stray:
        raise SystemExit(
            f"[fine_align] {', '.join(stray)} {'do' if len(stray) > 1 else 'does'} not "
            f"apply to backend {fine_backend!r}"
        )
    if aretomo:
        if "fiducials" in cfg:
            raise SystemExit(
                "[fiducials] is not read by a lane whose [fine_align] backend is "
                "'aretomo': AreTomo2 aligns from the tilt stack, with no fiducial steps"
            )
        fid: dict[str, Any] = {}
    else:
        fid = _require(cfg, "fiducials", ("seed_backend", "track_backend"),
                       "every lane that reconstructs")
    ctf = _require(cfg, "ctf", ("voltage_kv", "spherical_aberration_mm"),
                   "every lane that reconstructs")
    tmpl, pick, extract, exe = _picking_sections(cfg)
    recon = cfg.get("reconstruct", {})
    seg = cfg.get("segment")

    if not aretomo and FIDUCIAL_PAIRS.get(fid["seed_backend"]) != fid["track_backend"]:
        pairs = "; ".join(f"{seed} with {track}" for seed, track in FIDUCIAL_PAIRS.items())
        raise SystemExit(
            f"[fiducials] seed_backend {fid['seed_backend']!r} and track_backend "
            f"{fid['track_backend']!r} do not belong together; use {pairs}"
        )
    if aretomo and "thickness" not in recon:
        # tiltalign's thickness comes from the fiducials' spread in depth;
        # AreTomo2 reports none, so reconstruction would refuse after hours of
        # the chain. The authors of EMPIAR-11830 gave tilt THICKNESS themselves.
        raise SystemExit(
            "[reconstruct] thickness must be set with [fine_align] backend 'aretomo': "
            "AreTomo2 does not measure the sample depth"
        )
    if fid.get("seed_backend") == "imod-patch-seed" and "thickness" not in recon:
        # The derived thickness measures specimen depth only when the fiducials
        # sample it. Coplanar patches do not: on EMPIAR-11538 TS_0002 it gave a
        # 24 nm box for a lamella, and reconstruction still reported `pass`.
        raise SystemExit(
            "[reconstruct] thickness must be set with patch tracking: patches cannot "
            "measure the sample depth (see configs/lanes/ribo-bin8.toml)"
        )
    _check_picking(tmpl, pick, seg)

    def sub(template: str) -> str:
        return template.format(series=series)

    if source == "frames":
        motion = _require(cfg, "motion", ("backend",), "input = 'frames'")
        if "frames_pattern" in cfg.get("data", {}) and align.get("tilt_axis_angle") is None:
            # Without mdocs nothing records the tilt-axis rotation, and
            # align-tilt-series would refuse only after motion correction ran.
            raise SystemExit(
                "[align] tilt_axis_angle must be set with [data] frames_pattern: "
                "without mdocs nothing else gives the tilt axis"
            )
        entry: list[tuple[str, Command]] = [
            # Motion correction, then the stack from the corrected sums.
            (
                "prepare-tilt-series",
                ["prepare-tilt-series", *common,
                 *flag("motion-backend", motion["backend"]),
                 # MotionCor3 finds the GPU itself; the flag only tells the
                 # backend there is one. Left to auto, the device is looked up
                 # through torch, which refused a run on an MDC node whose GPU
                 # nvidia-smi saw but torch did not (job 3609935, 2026-10-14).
                 *(flag("device", "cuda") if motion["backend"] == "motioncor3" else []),
                 *flag("storage-policy", motion.get("storage_policy")),
                 *flag("motioncor3-executable", exe.get("motioncor3")),
                 *flag("motioncor3-gpu", motion.get("gpu")),
                 *flag("motioncor3-patch-x", motion.get("patch_x")),
                 *flag("motioncor3-patch-y", motion.get("patch_y")),
                 *flag("motioncor3-pixel-size", motion.get("pixel_size_angstrom")),
                 *flag("motioncor3-gain", motion.get("gain_reference")),
                 *flag("motioncor3-gain-rotation", motion.get("gain_rotation")),
                 *flag("motioncor3-gain-flip", motion.get("gain_flip")),
                 *flag("motioncor3-ftbin", motion.get("ftbin")),
                 *flag("motioncor3-eer-sampling", motion.get("eer_sampling")),
                 *flag("motioncor3-eer-fractions", motion.get("eer_fractions"))],
            ),
        ]
    else:
        data = _require(cfg, "data", ("mdoc", "stack"), "input = 'stack'")
        entry = [
            # One assembled stack per series, so init and prepare-tilt-series do
            # not apply. import-tilt-stack takes no --registry; it writes the
            # manifest and registry under --out itself.
            (
                "import-tilt-stack",
                [
                    "import-tilt-stack",
                    "--mdoc", sub(data["mdoc"]),
                    "--stack", sub(data["stack"]),
                    "--tilt-series-id", series,
                    "--out", str(project),
                ],
            ),
        ]

    excluded_angles = align.get("exclude_tilt_angles")
    if excluded_angles is not None and (
        not isinstance(excluded_angles, list)
        or not all(isinstance(a, (int, float)) and not isinstance(a, bool)
                   for a in excluded_angles)
    ):
        raise SystemExit("[align] exclude_tilt_angles must be a list of angles, as [-75, 6]")
    coarse: list[tuple[str, Command]] = [
        # Everything up to fine alignment is shared and unlabelled.
        (
            "align-tilt-series",
            ["align-tilt-series", *common,
             *flag("binning", align["binning"]),
             *flag("min-std-ratio", align["min_std_ratio"]),
             *flag("tilt-axis-angle", align.get("tilt_axis_angle")),
             *flag("exclude-tilt-angles",
                   ",".join(f"{a:g}" for a in excluded_angles) if excluded_angles else None)],
        ),
        (
            "qc-coarse-alignment",
            ["qc-coarse-alignment", *common,
             *flag("preview-binning", qc["preview_binning"]),
             *flag("residual-warning-px", qc["residual_warning_px"])],
        ),
    ]
    tilt_cor = fine.get("aretomo_tilt_cor")
    fine_aretomo: list[tuple[str, Command]] = [
        (
            "fine-align-tilt-series",
            ["fine-align-tilt-series", *common,
             *flag("backend", "aretomo"),
             *flag("aretomo-executable", exe.get("aretomo")),
             *flag("aretomo-align-z-nm", fine.get("aretomo_align_z_nm")),
             *([] if tilt_cor is None else [
                 "--aretomo-tilt-cor" if tilt_cor else "--no-aretomo-tilt-cor"
             ]),
             *flag("aretomo-dark-tol", fine.get("aretomo_dark_tol")),
             *flag("aretomo-gpu", fine.get("aretomo_gpu"))],
        ),
    ]
    fiducial_and_fine: list[tuple[str, Command]] = fine_aretomo if aretomo else [
        (
            "generate-fiducial-seed",
            ["generate-fiducial-seed", *common,
             *flag("backend", fid["seed_backend"]),
             # Bead path.
             *flag("tracking-binning", fid.get("tracking_binning")),
             *flag("fiducial-diameter-nm", fid.get("fiducial_diameter_nm")),
             *flag("raw-pixel-spacing", fid.get("raw_pixel_spacing_angstrom")),
             *flag("target-beads", fid.get("target_beads")),
             # Patch path.
             *flag("pre-binning", fid.get("pre_binning")),
             *flag("num-patches-x", fid.get("num_patches_x")),
             *flag("num-patches-y", fid.get("num_patches_y")),
             *flag("patch-overlap", fid.get("patch_overlap")),
             *flag("patch-margin-px", fid.get("patch_margin_px"))],
        ),
        (
            "track-fiducials",
            ["track-fiducials", *common,
             *flag("backend", fid["track_backend"]),
             *flag("min-tracked-fiducials", fid.get("min_tracked_fiducials")),
             *flag("coverage-warning", fid.get("coverage_warning")),
             *flag("coverage-failure", fid.get("coverage_failure")),
             # Bead path only. Patch tracking is a single correlation pass, and
             # track-fiducials refuses this flag for it rather than ignoring it.
             *flag("rounds", fid.get("rounds")),
             # Patch path only, and likewise refused for the bead backend.
             *flag("patch-filter-sigma1", fid.get("patch_filter_sigma1")),
             *flag("patch-filter-radius2", fid.get("patch_filter_radius2")),
             *flag("patch-filter-sigma2", fid.get("patch_filter_sigma2"))],
        ),
        (
            "fine-align-tilt-series",
            ["fine-align-tilt-series", *common,
             *flag("robust-fitting", fine.get("robust_fitting")),
             *flag("auto-prune-outliers", fine.get("auto_prune_outliers")),
             *flag("max-pruned-fraction", fine.get("max_pruned_fraction")),
             # One plane for patch tracking, two for beads. Left to the config
             # rather than derived from the tracking backend, because the two are
             # separate choices and a lane comparing backends needs to say so.
             *flag("surfaces-to-analyze", fine.get("surfaces_to_analyze"))],
        ),
    ]
    alignment: list[tuple[str, Command]] = [
        *coarse,
        *fiducial_and_fine,
        # The CTF fit stack carries NO variant label, so that correct-ctf in any
        # lane can fall back to the model built from it.
        (
            "build-ctf-stack",
            ["build-final-aligned-stack", *common,
             *flag("output-binning", binning["ctf_fit"])],
        ),
        (
            "estimate-ctf",
            ["estimate-ctf", *common,
             *flag("ctf-voltage-kv", ctf["voltage_kv"]),
             *flag("ctf-spherical-aberration-mm", ctf["spherical_aberration_mm"]),
             *flag("ctf-amplitude-contrast", ctf.get("amplitude_contrast")),
             *flag("ctffind-min-defocus-angstrom", ctf.get("min_defocus_angstrom")),
             *flag("ctffind-max-defocus-angstrom", ctf.get("max_defocus_angstrom")),
             # The per-view pass searches this far either side of the 0-degree fit.
             *flag("ctffind-defocus-search-window-angstrom",
                   ctf.get("defocus_search_window_angstrom")),
             *flag("ctffind-defocus-step-angstrom", ctf.get("defocus_step_angstrom")),
             *flag("ctf-max-excluded-views", ctf.get("max_excluded_views")),
             # Otherwise ctffind5, then ctffind, from PATH: at MDC the latter can be
             # CTFFIND 4.1.14, which this backend refuses.
             *flag("ctffind-executable", exe.get("ctffind")),
             *([] if ctf.get("determine_sample_tilt") is None else [
                 "--ctffind-determine-sample-tilt" if ctf["determine_sample_tilt"]
                 else "--no-ctffind-determine-sample-tilt"
             ])],
        ),
        # The lane parts here.
        (
            "build-final-aligned-stack",
            ["build-final-aligned-stack", *common,
             *flag("output-binning", binning["tomogram"]),
             *flag("variant", lane)],
        ),
        (
            "correct-ctf",
            ["correct-ctf", *common, *flag("variant", lane)],
        ),
        (
            "reconstruct-tomogram",
            ["reconstruct-tomogram", *common,
             *flag("thickness", recon.get("thickness")),
             *flag("variant", lane)],
        ),
    ]

    return [*entry, *alignment, *_segmentation_steps(seg, common, lane),
            *_picking_steps(tmpl, pick, exe, common, lane),
            *_extraction_steps(extract, common, lane)]


def _picking_sections(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """[template], [pick], [extract] and [executables] of a lane that picks."""
    tmpl = _require(cfg, "template", ("low_pass_angstrom",), "every lane that picks")
    pick = _require(
        cfg, "pick", ("particle_diameter_angstrom", "gpu_ids"),
        "every lane that picks",
    )
    if "number_of_particles" in pick and "expected_particles_per_tomogram" in pick:
        raise SystemExit(
            "[pick] sets both number_of_particles (the count to take) and "
            "expected_particles_per_tomogram (twice it is taken); keep one"
        )
    extract = _require(cfg, "extract", ("box_size_px",), "every lane that picks")
    exe = _require(
        cfg, "executables",
        ("pytom_create_template", "pytom_create_mask",
         "pytom_match_template", "pytom_extract_candidates"),
        "every lane that picks",
    )
    return tmpl, pick, extract, exe


def _check_picking(
    tmpl: dict[str, Any], pick: dict[str, Any], seg: dict[str, Any] | None
) -> None:
    """The choices of a picking lane that contradict each other."""
    if ("emdb_id" in tmpl) == ("source_map" in tmpl):
        raise SystemExit(
            "[template] needs exactly one of emdb_id (fetched from EMDB) and "
            "source_map (a local MRC map)"
        )
    if "cut_off" in pick and "number_of_false_positives" in pick:
        # pick-particles refuses the pair too, but only once its job has started.
        raise SystemExit(
            "[pick] sets both cut_off and number_of_false_positives, which are "
            "alternatives: cut_off = 0 keeps the best number_of_particles peaks with "
            "no score floor, number_of_false_positives sets a floor from the noise"
        )
    if pick.get("use_segmentation_mask") and seg is None:
        raise SystemExit(
            "[pick] use_segmentation_mask needs membranes, but there is no [segment] section"
        )


def _segmentation_steps(
    seg: dict[str, Any] | None, common: Command, lane: str
) -> list[tuple[str, Command]]:
    """Membrane segmentation, when the lane has [segment]."""
    if seg is None:
        return []
    if "model" not in seg:
        raise SystemExit("[segment] is missing model, the MemBrain-seg checkpoint")
    tta = seg.get("test_time_augmentation")
    return [
        (
            "segment-tomogram",
            ["segment-tomogram", *common,
             *flag("membrain-executable", seg.get("executable")),
             *flag("membrain-model", seg["model"]),
             *flag("membrain-sliding-window-size", seg.get("sliding_window_size")),
             *([] if tta is None else [
                 "--membrain-test-time-augmentation" if tta
                 else "--no-membrain-test-time-augmentation"
             ]),
             *flag("variant", lane)],
        ),
    ]


def _picking_steps(
    tmpl: dict[str, Any], pick: dict[str, Any], exe: dict[str, Any], common: Command, lane: str
) -> list[tuple[str, Command]]:
    """The template, then template matching."""
    return [
        (
            "prepare-template",
            ["prepare-template", *common,
             *flag("emdb-id", tmpl.get("emdb_id")),
             *flag("source-map", tmpl.get("source_map")),
             # The particle's widest diameter sizes the mask (radius + 3 px) and
             # the box (2x); [pick] already names it, so [template] need not.
             *flag("template-particle-diameter-angstrom",
                   tmpl.get("particle_diameter_angstrom", pick["particle_diameter_angstrom"])),
             *flag("template-low-pass-angstrom", tmpl["low_pass_angstrom"]),
             *flag("template-mask-radius-px", tmpl.get("mask_radius_px")),
             *flag("template-mask-sigma-px", tmpl.get("mask_sigma_px")),
             *flag("template-box-size-px", tmpl.get("box_size_px")),
             *flag("pytom-create-template-executable", exe["pytom_create_template"]),
             *flag("pytom-create-mask-executable", exe["pytom_create_mask"]),
             *flag("variant", lane)],
        ),
        (
            "pick-particles",
            ["pick-particles", *common,
             # pytom finds the GPU itself, as MotionCor3 does; see prepare-tilt-series.
             *flag("device", "cuda"),
             *flag("pytom-particle-diameter", pick["particle_diameter_angstrom"]),
             *flag("pytom-extract-particle-diameter",
                   pick.get("extract_particle_diameter_angstrom")),
             *flag("pytom-z-axis-symmetry", pick.get("z_axis_symmetry")),
             # Left out: twice what pytom recommends (pick-particles' default).
             *flag("pytom-number-of-particles", pick.get("number_of_particles")),
             # Twice this is taken: the person's expectation per tomogram.
             *flag("pytom-expected-particles", pick.get("expected_particles_per_tomogram")),
             *flag("pytom-number-of-false-positives", pick.get("number_of_false_positives")),
             # 0 keeps the best number_of_particles peaks with no score floor.
             *flag("pytom-cut-off", pick.get("cut_off")),
             *flag("pytom-tophat-filter", pick.get("tophat_filter")),
             *flag("pytom-low-pass", pick.get("low_pass_angstrom")),
             *flag("pytom-random-phase-correction", pick.get("random_phase_correction")),
             # pick-particles weights by |CTF| unless told not to, so off is a flag too.
             *([] if pick.get("ctf_weighting") is None else [
                 "--pytom-ctf-weighting" if pick["ctf_weighting"]
                 else "--no-pytom-ctf-weighting"
             ]),
             *flag("pytom-per-tilt-weighting", pick.get("per_tilt_weighting")),
             *flag("pytom-spectral-whitening", pick.get("spectral_whitening")),
             *flag("pytom-gpu-ids", pick["gpu_ids"]),
             *flag("pytom-volume-split", pick.get("volume_split")),
             *([] if pick.get("use_segmentation_mask") is None else [
                 "--use-segmentation-mask" if pick["use_segmentation_mask"]
                 else "--no-use-segmentation-mask"
             ]),
             *flag("segmentation-mask-dilation-px", pick.get("segmentation_mask_dilation_px")),
             *flag("pytom-match-template-executable", exe["pytom_match_template"]),
             *flag("pytom-extract-candidates-executable", exe["pytom_extract_candidates"]),
             *flag("variant", lane)],
        ),
    ]


def _extraction_steps(
    extract: dict[str, Any], common: Command, lane: str
) -> list[tuple[str, Command]]:
    """The picks' preview, then the particles cut from the tomogram."""
    return [
        # Look at the picks before spending the extraction on them: this writes
        # the tomogram with its picks for both ChimeraX and 3dmod. It registers
        # nothing, so it is the one step here that can be skipped or re-run at
        # any time without changing what the rest of the chain sees.
        (
            "visualize-picks",
            ["visualize-picks", *common, *flag("variant", lane)],
        ),
        (
            "extract-particles",
            ["extract-particles", *common,
             *flag("box-size-px", extract["box_size_px"]),
             *flag("pad-edges", extract.get("pad_edges")),
             *flag("variant", lane)],
        ),
    ]


def particles_series(cfg: dict[str, Any]) -> dict[str, int]:
    """[data] particles_series of a particles lane: each series with the tomogram
    number (Dynamo's column 20) of its rows in the table."""
    value = cfg.get("data", {}).get("particles_series")
    if (not isinstance(value, dict) or not value
            or not all(isinstance(number, int) and not isinstance(number, bool)
                       for number in value.values())):
        raise SystemExit(
            "[data] particles_series must map each series to the tomogram number of its "
            "rows in the table (column 20), as {TS_01 = 1, TS_02 = 2}"
        )
    if len(set(value.values())) != len(value):
        raise SystemExit("[data] particles_series gives two series the same tomogram number")
    return dict(value)


def _listed(value: Any, where: str) -> str:
    """A list of words as the commands take it, comma-separated; a string as it is."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and value and all(
            isinstance(item, str) and item for item in value):
        return ",".join(value)
    raise SystemExit(f"{where} must be a word or a list of words, as [\"x\", \"y\"]")


def _imported_steps(
    cfg: dict[str, Any], project: Path, series: str, source: str, common: Command, lane: str
) -> list[tuple[str, Command]]:
    """The chain of a lane whose series enter as files made elsewhere
    (docs/entry_and_exit.md): their import, then the steps from there.

    [data] answers the questions the import asks, each key one option of its
    command, and nothing is assumed for a key left out: the import refuses what it
    needs and was not told. Every import takes the lane's label, as every step after
    it does, so they find what it entered by that label.
    """

    def sub(template: str) -> str:
        return template.format(series=series)

    if source == "particles":
        data = _require(cfg, "data", ("particles_folder", "particles_table", "particles_series"),
                        "input = 'particles'")
        named = particles_series(cfg)
        return [(
            "import-particles",
            ["import-particles",
             "--folder", str(data["particles_folder"]),
             "--table", str(data["particles_table"]),
             "--out", str(project),
             # The whole table's series are named, so that none of its tomograms is
             # left out unnoticed; this series' rows are imported.
             *[part for name, number in named.items() for part in ("--series", f"{number}={name}")],
             "--only-series", series,
             *flag("pixel-size", data.get("particles_pixel_size_angstrom")),
             *flag("variant", lane)],
        )]

    data = _require(cfg, "data", ("tomogram", "tilt_angles", "orientation", "processing"),
                    f"input = {source!r}")
    entry: list[tuple[str, Command]] = [(
        "import-tomogram",
        ["import-tomogram",
         "--volume", sub(str(data["tomogram"])),
         "--tilt-angles", sub(str(data["tilt_angles"])),
         "--orientation", str(data["orientation"]),
         "--processing", _listed(data["processing"], "[data] processing"),
         "--tilt-series-id", series,
         "--out", str(project),
         *flag("pixel-size", data.get("tomogram_pixel_size_angstrom")),
         *flag("variant", lane)],
    )]
    if source == "tomograms":
        tmpl, pick, extract, exe = _picking_sections(cfg)
        if "low_pass_angstrom" not in pick:
            # pick-particles would refuse it in the GPU job, after the queue.
            raise SystemExit(
                "[pick] low_pass_angstrom must be set with input = 'tomograms': "
                "pick-particles derives it from a fine alignment, and these tomograms "
                "were aligned elsewhere"
            )
        if pick.get("ctf_weighting"):
            raise SystemExit(
                "[pick] ctf_weighting = true cannot apply with input = 'tomograms': the "
                "tomograms enter without a CTF fit, so picking has none to weight by"
            )
        seg = cfg.get("segment")
        _check_picking(tmpl, pick, seg)
        return [*entry, *_segmentation_steps(seg, common, lane),
                *_picking_steps(tmpl, pick, exe, common, lane),
                *_extraction_steps(extract, common, lane)]

    pick = _require(cfg, "pick", ("particle_diameter_angstrom",),
                    "input = 'picks' (the check that the picks land on particles)")
    extract = _require(cfg, "extract", ("box_size_px",), "every lane that extracts")
    data = _require(cfg, "data", ("picks", "picks_format"), "input = 'picks'")
    number = None
    if "dynamo_tomograms" in data:
        numbers = data["dynamo_tomograms"]
        number = numbers.get(series) if isinstance(numbers, dict) else None
        if not isinstance(number, int) or isinstance(number, bool):
            raise SystemExit(
                f"[data] dynamo_tomograms gives no tomogram number for {series}: it maps "
                "each series to the number of its rows in the table (column 20), as "
                "{TS_01 = 1, TS_02 = 2}"
            )
    columns = data.get("csv_columns")
    tomo_name = data.get("relion_tomo_name")
    return [
        *entry,
        (
            "import-picks",
            ["import-picks",
             "--picks", sub(str(data["picks"])),
             "--format", str(data["picks_format"]),
             "--tilt-series-id", series,
             "--out", str(project),
             *flag("particle-diameter", pick["particle_diameter_angstrom"]),
             # The coordinates are in the tomogram the step before imported.
             *flag("tomogram-variant", lane),
             *flag("picked-in", data.get("picked_in")),
             *flag("template-box-px", data.get("template_box_px")),
             *flag("csv-columns",
                   None if columns is None else _listed(columns, "[data] csv_columns")),
             *flag("coordinate-units", data.get("coordinate_units")),
             *flag("coordinate-origin", data.get("coordinate_origin")),
             *flag("relion-tomo-name", None if tomo_name is None else sub(str(tomo_name))),
             *flag("relion-tomograms-star", data.get("relion_tomograms_star")),
             *flag("relion-tilt-series-pixel-size",
                   data.get("relion_tilt_series_pixel_size_angstrom")),
             *flag("imod-object", data.get("imod_object")),
             *flag("dynamo-tomogram", number),
             *flag("variant", lane)],
        ),
        *_extraction_steps(extract, common, lane),
    ]


def slice_steps(
    steps: list[tuple[str, Command]], start: str | None, end: str | None
) -> list[tuple[str, Command]]:
    """Plain list slicing by step name. Not resume: no state is inspected."""
    names = [name for name, _ in steps]
    for label, value in (("--from", start), ("--until", end)):
        if value is not None and value not in names:
            raise SystemExit(
                f"{label}: unknown step {value!r}\nknown steps: {', '.join(names)}"
            )
    lo = names.index(start) if start else 0
    hi = names.index(end) + 1 if end else len(steps)
    return steps[lo:hi]


def render(command: Command) -> str:
    return "cryoet " + " ".join(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--series", action="append", required=True, dest="series")
    parser.add_argument("--from", dest="start", default=None, help="first step to emit")
    parser.add_argument("--until", dest="end", default=None, help="last step to emit")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="add --overwrite to every step but the imports; blanket by design, since a "
        "re-run of a range is what it is for. An import is never written over: the same "
        "files with the same answers are imported once",
    )
    parser.add_argument("--execute", action="store_true", help="run instead of printing")
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument(
        "--list-steps-range",
        action="store_true",
        help="print the step names --from..--until would run, one per line",
    )
    args = parser.parse_args()

    _, cfg = read_config(args.config)

    if args.list_steps:
        for name, _ in build_steps(cfg, args.project, args.series[0]):
            print(name)
        return 0
    if args.list_steps_range:
        steps = build_steps(cfg, args.project, args.series[0])
        for name, _ in slice_steps(steps, args.start, args.end):
            print(name)
        return 0

    for series in args.series:
        steps = slice_steps(build_steps(cfg, args.project, series), args.start, args.end)
        print(f"\n# ---- {series} ({cfg['lane']['name']}) ----")
        for name, command in steps:
            if args.overwrite and name not in IMPORT_STEPS:
                command = [*command, "--overwrite"]
            if not args.execute:
                print(f"\n# {name}")
                print(render(command))
                continue

            print(f"\n# {name}\n{render(command)}", flush=True)
            # No shell here, so $VAR in a config value would reach the tool
            # literally. Expand it ourselves; printing keeps the readable form.
            expanded = [os.path.expandvars(part) for part in command]
            disk_before = project_size_bytes(args.project)
            # The clock runs around the step alone. Until 2026-10-28 it also ran
            # through the size walk above, which is not the step's work and grows
            # with every file the project holds.
            started = time.monotonic()
            result = subprocess.run(["cryoet", *expanded], check=False)
            elapsed = time.monotonic() - started
            finished = datetime.now(UTC)
            disk_after = project_size_bytes(args.project)
            record_timing(
                args.project,
                series=series,
                step=name,
                lane=cfg["lane"]["name"],
                variant=cfg["lane"]["variant"],
                seconds=elapsed,
                disk_delta_bytes=disk_after - disk_before,
                exit_code=result.returncode,
                finished_at=finished,
            )
            print(
                f"# {name}: {elapsed / 60:.1f} min, "
                f"{(disk_after - disk_before) / 1e9:+.2f} GB",
                flush=True,
            )
            if result.returncode != 0:
                print(
                    f"\n# FAILED at {name} (exit {result.returncode}) for {series}."
                    f"\n# Fix it, then resume with:"
                    f"\n#   --series {series} --from {name}",
                    file=sys.stderr,
                )
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
