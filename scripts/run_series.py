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
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
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


def build_steps(cfg: dict[str, Any], project: Path, series: str) -> list[tuple[str, Command]]:
    """The canned lane-B chain. Order is fixed here, not in the config."""

    manifest = project / "manifests" / f"{series}.json"
    registry = project / "artifacts.json"
    common = ["--manifest", str(manifest), "--registry", str(registry), "--out", str(project)]

    lane = cfg["lane"]["variant"]
    data, binning = cfg["data"], cfg["binning"]
    align, qc = cfg["align"], cfg["qc_coarse"]
    fid, fine = cfg["fiducials"], cfg["fine_align"]
    ctf, tmpl, pick = cfg["ctf"], cfg["template"], cfg["pick"]
    extract, exe = cfg["extract"], cfg["executables"]
    recon = cfg.get("reconstruct", {})

    def sub(template: str) -> str:
        return template.format(series=series)

    steps: list[tuple[str, Command]] = [
        # No frames in this deposition: one assembled stack per series, so init
        # and prepare-tilt-series do not apply. import-tilt-stack takes no
        # --registry; it writes the manifest and registry under --out itself.
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
             # Bead path only; ignored by imod-patch-seed.
             *flag("fiducial-diameter-nm", fid.get("fiducial_diameter_nm")),
             # Patch path only; ignored by imod-autofidseed.
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
             # track-fiducials now refuses this flag for it rather than ignoring it.
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
             *flag("auto-prune-outliers", fine.get("auto_prune_outliers"))],
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
             *flag("ctf-amplitude-contrast", ctf.get("amplitude_contrast"))],
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
        (
            "prepare-template",
            ["prepare-template", *common,
             *flag("emdb-id", tmpl["emdb_id"]),
             *flag("template-low-pass-angstrom", tmpl["low_pass_angstrom"]),
             *flag("pytom-create-template-executable", exe["pytom_create_template"]),
             *flag("pytom-create-mask-executable", exe["pytom_create_mask"]),
             *flag("variant", lane)],
        ),
        (
            "pick-particles",
            ["pick-particles", *common,
             *flag("pytom-particle-diameter", pick["particle_diameter_angstrom"]),
             *flag("pytom-number-of-particles", pick["number_of_particles"]),
             *flag("pytom-number-of-false-positives", pick.get("number_of_false_positives")),
             *flag("pytom-tophat-filter", pick.get("tophat_filter")),
             *flag("pytom-gpu-ids", pick["gpu_ids"]),
             *flag("pytom-match-template-executable", exe["pytom_match_template"]),
             *flag("pytom-extract-candidates-executable", exe["pytom_extract_candidates"]),
             *flag("variant", lane)],
        ),
        (
            "export-picks",
            ["export-picks", *common, *flag("variant", lane)],
        ),
        (
            "extract-particles",
            ["extract-particles", *common,
             *flag("box-size-px", extract["box_size_px"]),
             *flag("variant", lane)],
        ),
    ]
    return steps


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
    parser.add_argument("--execute", action="store_true", help="run instead of printing")
    parser.add_argument("--list-steps", action="store_true")
    args = parser.parse_args()

    cfg = tomllib.loads(args.config.read_text())

    if args.list_steps:
        for name, _ in build_steps(cfg, args.project, args.series[0]):
            print(name)
        return 0

    for series in args.series:
        steps = slice_steps(build_steps(cfg, args.project, series), args.start, args.end)
        print(f"\n# ---- {series} ({cfg['lane']['name']}) ----")
        for name, command in steps:
            if not args.execute:
                print(f"\n# {name}")
                print(render(command))
                continue

            print(f"\n# {name}\n{render(command)}", flush=True)
            # No shell here, so $VAR in a config value would reach the tool
            # literally. Expand it ourselves; printing keeps the readable form.
            expanded = [os.path.expandvars(part) for part in command]
            result = subprocess.run(["cryoet", *expanded], check=False)
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
