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
    seconds: float,
    disk_delta_bytes: int,
    exit_code: int,
) -> None:
    """Append one measured step to the project's timing table.

    Appended to a file rather than printed only, because the numbers are needed
    to size a sandbox and a quota request, and scrollback is not a table. One
    JSON object per line so an interrupted run leaves every step it did finish.

    This is a stand-in for what the `run` entity will record properly, and it is
    deliberately not a substitute: no fingerprint, no parameters, no attribution.
    """

    row = {
        "series": series,
        "step": step,
        "seconds": round(seconds, 1),
        "disk_delta_bytes": disk_delta_bytes,
        "exit_code": exit_code,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    destination = project / "step_timings.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


INPUTS = ("frames", "stack")
# The seed and the tracking backend come in pairs: beads are seeded and tracked
# as beads, patches as patches. A mixed pair has no defined meaning.
FIDUCIAL_PAIRS = {
    "imod-autofidseed": "imod-beadtrack",
    "imod-patch-seed": "imod-patch-track",
}

# Every key a lane config may set, by section. Keys are read by name, so a
# misspelt one used to be skipped without a word: its flag was not rendered and
# the step ran with its default. Nothing outside this table is accepted, so a key
# a step starts reading goes in here in the same change.
LANE_KEYS: dict[str, frozenset[str]] = {
    # [run] and its [[run.stage]] tables are read by scripts/submit_lanes.py, and
    # the frames half of [data] by `cryoet init`; the rest by build_steps below.
    "lane": frozenset({"name", "input", "variant"}),
    "run": frozenset({"project", "series", "stage"}),
    "data": frozenset({"frames", "mdocs", "frames_pattern", "pixel_size_angstrom",
                       "mdoc", "stack"}),
    "motion": frozenset({"backend", "storage_policy", "gpu", "patch_x", "patch_y",
                         "pixel_size_angstrom", "gain_reference", "gain_rotation",
                         "gain_flip", "ftbin"}),
    "binning": frozenset({"ctf_fit", "tomogram"}),
    "align": frozenset({"binning", "min_std_ratio", "tilt_axis_angle"}),
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
    "fine_align": frozenset({"robust_fitting", "auto_prune_outliers",
                             "max_pruned_fraction", "surfaces_to_analyze"}),
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
        "z_axis_symmetry", "number_of_particles", "number_of_false_positives",
        "cut_off", "tophat_filter", "low_pass_angstrom", "random_phase_correction",
        "ctf_weighting", "per_tilt_weighting", "spectral_whitening", "gpu_ids",
        "volume_split", "use_segmentation_mask", "segmentation_mask_dilation_px",
    }),
    "extract": frozenset({"box_size_px", "pad_edges"}),
    "executables": frozenset({"motioncor3", "ctffind", "pytom_create_template",
                              "pytom_create_mask", "pytom_match_template",
                              "pytom_extract_candidates"}),
}
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


def lane_input(cfg: dict[str, Any]) -> str:
    """How the series enter the project: `frames` (movies, through init) or `stack`."""
    lane = cfg.get("lane", {})
    if "chain" in lane:
        raise SystemExit(f"[lane] chain {RENAMED['lane', 'chain']}")
    value = lane.get("input")
    if value not in INPUTS:
        raise SystemExit(f"[lane] input must be one of {', '.join(INPUTS)}, not {value!r}")
    return value


def build_steps(cfg: dict[str, Any], project: Path, series: str) -> list[tuple[str, Command]]:
    """The lane's chain, in the one order this file fixes.

    The config chooses three things and nothing about the order:

    * `[lane] input` — `frames`: movies, motion-corrected here, after `init`
      wrote the manifests (project-level, so not a step); `stack`: one assembled
      stack per series, imported here;
    * `[fiducials]` backends — gold beads or patch tracking;
    * `[segment]` — present: membranes are segmented after reconstruction, and
      picking may use them as a mask; absent: the step is left out.

    A flag is rendered only when its value is in the config, so one builder
    serves every combination: bead-only options never reach a patch lane. A key
    that no step reads is refused rather than skipped.
    """

    check_keys(cfg)
    source = lane_input(cfg)
    manifest = project / "manifests" / f"{series}.json"
    registry = project / "artifacts.json"
    common = ["--manifest", str(manifest), "--registry", str(registry), "--out", str(project)]

    lane = _require(cfg, "lane", ("variant",), "every lane")["variant"]
    binning = _require(cfg, "binning", ("ctf_fit", "tomogram"), "every lane")
    align = _require(cfg, "align", ("binning", "min_std_ratio"), "every lane")
    qc = _require(cfg, "qc_coarse", ("preview_binning", "residual_warning_px"), "every lane")
    fid = _require(cfg, "fiducials", ("seed_backend", "track_backend"), "every lane")
    fine = cfg.get("fine_align", {})
    ctf = _require(cfg, "ctf", ("voltage_kv", "spherical_aberration_mm"), "every lane")
    tmpl = _require(cfg, "template", ("low_pass_angstrom",), "every lane")
    pick = _require(
        cfg, "pick", ("particle_diameter_angstrom", "gpu_ids"),
        "every lane",
    )
    extract = _require(cfg, "extract", ("box_size_px",), "every lane")
    exe = _require(
        cfg, "executables",
        ("pytom_create_template", "pytom_create_mask",
         "pytom_match_template", "pytom_extract_candidates"),
        "every lane",
    )
    recon = cfg.get("reconstruct", {})
    seg = cfg.get("segment")

    if FIDUCIAL_PAIRS.get(fid["seed_backend"]) != fid["track_backend"]:
        pairs = "; ".join(f"{seed} with {track}" for seed, track in FIDUCIAL_PAIRS.items())
        raise SystemExit(
            f"[fiducials] seed_backend {fid['seed_backend']!r} and track_backend "
            f"{fid['track_backend']!r} do not belong together; use {pairs}"
        )
    if fid["seed_backend"] == "imod-patch-seed" and "thickness" not in recon:
        # The derived thickness measures specimen depth only when the fiducials
        # sample it. Coplanar patches do not: on EMPIAR-11538 TS_0002 it gave a
        # 24 nm box for a lamella, and reconstruction still reported `pass`.
        raise SystemExit(
            "[reconstruct] thickness must be set with patch tracking: patches cannot "
            "measure the sample depth (see configs/lanes/ribo-bin8.toml)"
        )
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
                 *flag("motioncor3-ftbin", motion.get("ftbin"))],
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

    alignment: list[tuple[str, Command]] = [
        # Everything up to fine alignment is shared and unlabelled.
        (
            "align-tilt-series",
            ["align-tilt-series", *common,
             *flag("binning", align["binning"]),
             *flag("min-std-ratio", align["min_std_ratio"]),
             *flag("tilt-axis-angle", align.get("tilt_axis_angle"))],
        ),
        (
            "qc-coarse-alignment",
            ["qc-coarse-alignment", *common,
             *flag("preview-binning", qc["preview_binning"]),
             *flag("residual-warning-px", qc["residual_warning_px"])],
        ),
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

    segmentation: list[tuple[str, Command]] = []
    if seg is not None:
        if "model" not in seg:
            raise SystemExit("[segment] is missing model, the MemBrain-seg checkpoint")
        tta = seg.get("test_time_augmentation")
        segmentation = [
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

    picking: list[tuple[str, Command]] = [
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
    return [*entry, *alignment, *segmentation, *picking]


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
        help="add --overwrite to every step; blanket by design, since a re-run "
        "of a range is what it is for, and every command accepts the flag",
    )
    parser.add_argument("--execute", action="store_true", help="run instead of printing")
    parser.add_argument("--list-steps", action="store_true")
    parser.add_argument(
        "--list-steps-range",
        action="store_true",
        help="print the step names --from..--until would run, one per line",
    )
    args = parser.parse_args()

    cfg = tomllib.loads(args.config.read_text())

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
            if args.overwrite:
                command = [*command, "--overwrite"]
            if not args.execute:
                print(f"\n# {name}")
                print(render(command))
                continue

            print(f"\n# {name}\n{render(command)}", flush=True)
            # No shell here, so $VAR in a config value would reach the tool
            # literally. Expand it ourselves; printing keeps the readable form.
            expanded = [os.path.expandvars(part) for part in command]
            started = time.monotonic()
            disk_before = project_size_bytes(args.project)
            result = subprocess.run(["cryoet", *expanded], check=False)
            elapsed = time.monotonic() - started
            disk_after = project_size_bytes(args.project)
            record_timing(
                args.project,
                series=series,
                step=name,
                seconds=elapsed,
                disk_delta_bytes=disk_after - disk_before,
                exit_code=result.returncode,
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
