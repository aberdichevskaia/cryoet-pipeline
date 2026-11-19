#!/usr/bin/env python3
"""A lane of input tomograms, tested on a series a lane of this site has picked.

    python3 scripts/tomograms_lane_test.py prepare --lane configs/lanes/ryr1-bin8.toml --series 039
    python3 scripts/show_imports.py tomograms <out>/data
    python3 scripts/submit_lanes.py <out>/lane.toml --dry-run
    python3 scripts/submit_lanes.py <out>/lane.toml
    python3 scripts/tomograms_lane_test.py check --test <out>

The first run of a lane whose series enter as tomograms made elsewhere
(docs/entry_and_exit.md, Order of work item 6), on real data, where the answer is
known. `prepare` hands the lane a volume as an etomo user brings one: the series'
tomogram as IMOD's tilt wrote it, turned by IMOD's own `clip rotx` into IMOD's
standard orientation, the mirror of the pipeline's, with the angles it was
reconstructed with. It writes a lane config of input tomograms, into a project of
its own beside the source project, on the storage the source lane already used,
that picks and extracts with the source lane's template, picking and extraction
settings, on the source lane's partitions. The source project is only read.

What is known, and what `check` sets against it:
- the volume: entered as imod-standard, it has to be the source project's
  tomogram voxel for voxel. The same tomogram reversed in z, which the other
  answer of the mirror pair would give, is the control. Measured on EMPIAR-10164
  TS_01 at bin8 on the Mac, 17 November 2026: a largest difference of 0, against
  35 for the reversed one; in Munich, 19 November, on the October project's
  TS_01, 0 against 354;
- the import's wedge check, which has to agree, or to say that a symmetric range
  leaves it blind to the side;
- the picks, matched with the source lane's within a radius both ways, and set
  against the same picks mirrored in z and against random positions in the
  picks' region. The two searches need not be the same: the import carries no
  CTF weighting and no dose, and the source picks were made with the lane as it
  stood then. So the picks are reported, with what the two pytom jobs differ in
  besides the tomogram, read from their job files, the template and its mask
  compared as data, since each project keeps its own.

The volume and the wedge decide the exit status: 0 when both hold, 1 when either
fails, 2 when the lane has not run far enough to tell. The picks are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repick  # noqa: E402
import run_series  # noqa: E402
import show_imports  # noqa: E402

from cryoet_pipeline.artifacts import ArtifactRegistry  # noqa: E402
from cryoet_pipeline.backends.subtomogram import pick_centre_offset  # noqa: E402
from cryoet_pipeline.emdb import normalize_emdb_id  # noqa: E402
from cryoet_pipeline.models import Artifact, ArtifactKind  # noqa: E402
from cryoet_pipeline.particles import read_star_blocks  # noqa: E402
from cryoet_pipeline.tomogram_export import processing_of, tilt_angles_of  # noqa: E402

ORIENTATION = "imod-standard"
RECORD = "test.json"
LANE = "lane.toml"
MATCH_RADIUS_PX = 2.0
PYTOM_KEYS = ("pytom_create_template", "pytom_create_mask", "pytom_match_template",
              "pytom_extract_candidates")
BLOCK = 32
# What the picks record of their search and pytom's job file does not hold.
RECORDED_SEARCH = ("pytom_number_of_particles", "pytom_volume_split", "tomogram_mask_coverage")


class TestRefused(SystemExit):
    """What makes the test impossible here, in words for the person."""

    def __init__(self, message: str) -> None:
        super().__init__(f"refused: {message}")


# --- The source lane -----------------------------------------------------------------


def _one(found: list[Artifact], what: str) -> Artifact:
    if len(found) != 1:
        named = ", ".join(artifact.id for artifact in found) or "none"
        raise TestRefused(f"the source project has {len(found)} {what} ({named}); one is needed")
    return found[0]


def source_tomogram(registry: ArtifactRegistry, series: str, variant: str) -> Artifact:
    return _one([artifact for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)
                 if artifact.covers(series) and artifact.variant == variant
                 and artifact.parameters.get("tomogram_branch", "full") == "full"],
                f"full tomograms of {series} under {variant!r}")


def source_picks(registry: ArtifactRegistry, tomogram: Artifact, variant: str,
                 named: str | None = None) -> Artifact:
    found = [artifact for artifact in registry.by_kind(ArtifactKind.PICKS)
             if artifact.variant == variant
             and artifact.parameters.get("input_tomogram_id") == tomogram.id]
    if named is not None:
        found = [artifact for artifact in found if artifact.id == named]
    return _one(found, f"picks in {tomogram.id} under {variant!r}"
                + (f" named {named}" if named else "; name one with --picks"))


def _stage_with(cfg: dict[str, Any], project: Path, series: str, step: str) -> dict[str, Any]:
    steps = run_series.build_steps(cfg, project, series)
    for stage in cfg["run"]["stage"]:
        if stage["from"] == run_series.MRA_STEP:
            continue
        names = [name for name, _ in run_series.slice_steps(steps, stage["from"], stage["until"])]
        if step in names:
            return dict(stage)
    raise TestRefused(f"no [[run.stage]] of the source lane runs {step}")


# --- prepare -------------------------------------------------------------------------


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {toml_value(item)}" for key, item in value.items()) + "}"
    raise TestRefused(f"cannot write {value!r} into a lane config")


def lane_text(header: list[str], cfg: dict[str, Any]) -> str:
    lines = [f"# {line}" if line else "#" for line in header]
    for section, values in cfg.items():
        if section == "run":
            continue
        lines += ["", f"[{section}]"]
        lines += [f"{key} = {toml_value(value)}" for key, value in values.items()]
    run = cfg["run"]
    lines += ["", "[run]", f"project = {toml_value(run['project'])}",
              f"series = {toml_value(run['series'])}"]
    for stage in run["stage"]:
        lines += ["", "[[run.stage]]"]
        lines += [f"{key} = {toml_value(value)}" for key, value in stage.items()]
    return "\n".join(lines) + "\n"


def _clip() -> str:
    found = shutil.which("clip")
    if found is None:
        raise TestRefused("IMOD's clip is not on PATH: load IMOD first, as the lane jobs do")
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(lane: Path, series: str, out: Path | None, picks_id: str | None = None) -> Path:
    _, source = run_series.read_config(lane)
    if run_series.lane_input(source) not in ("frames", "stack"):
        raise TestRefused(f"{lane} is a lane of input {source['lane']['input']!r}; the test "
                          "takes a series a lane reconstructed and picked here")
    variant = str(source["lane"]["variant"])
    project = ROOT / source["run"]["project"]
    registry = ArtifactRegistry.load(project / "artifacts.json")
    tomogram = source_tomogram(registry, series, variant)
    picks = source_picks(registry, tomogram, variant, picks_id)
    rec = Path(str(tomogram.parameters.get("imod_rec_path", "")))
    if tomogram.parameters.get("imod_rec_axis_order") != "yzx" or not rec.is_file():
        raise TestRefused(f"{tomogram.id} keeps no tilt output with Y as sections (its "
                          f"imod_rec_path is {rec}); the test turns that file with clip rotx")
    angles = tilt_angles_of(registry, tomogram, series)
    if angles is None:
        raise TestRefused(f"{tomogram.id} records no tilt angles")
    low_pass = source["pick"].get("low_pass_angstrom", picks.parameters.get("pytom_low_pass"))
    if not isinstance(low_pass, (int, float)) or isinstance(low_pass, bool):
        raise TestRefused(f"neither the source lane nor {picks.id} gives the low-pass its "
                          "picking used")

    name = f"{source['lane'].get('name', lane.stem)}-tomograms-test"
    # Beside the source project, where it really is: outputs/ can be a link to
    # other storage, as in Munich, where the clone's own share filled up.
    out = out if out is not None else project.resolve().parent / name
    if out.exists():
        raise TestRefused(f"{out} is there already; a test is never written over")
    data = out / "data"
    data.mkdir(parents=True)
    volume, tilts = data / f"{series}_rec.mrc", data / f"{series}.tlt"
    command = [_clip(), "rotx", str(rec), str(volume)]
    turned = subprocess.run(command, capture_output=True, text=True)
    if turned.returncode != 0 or not volume.is_file():
        raise TestRefused(f"clip rotx failed: {turned.stderr.strip() or turned.stdout.strip()}")
    shutil.copyfile(angles, tilts)
    # The template's map, as the source lane's prepare-template cached it, so that
    # the GPU job does not have to reach EMDB from a compute node.
    cached = None
    if "emdb_id" in source["template"]:
        name = f"emd_{normalize_emdb_id(str(source['template']['emdb_id']))}.map"
        if (project / "emdb-cache" / name).is_file():
            cached = out / "project" / "emdb-cache" / name
            cached.parent.mkdir(parents=True)
            shutil.copyfile(project / "emdb-cache" / name, cached)

    picking = _stage_with(source, project, series, "pick-particles")
    importing = _stage_with(source, project, series, "reconstruct-tomogram")
    steps = [step for step in ("segment-tomogram", "prepare-template")
             if step != "segment-tomogram" or "segment" in source]
    processing = processing_of(registry, tomogram) or ["raw"]
    cfg: dict[str, Any] = {
        "lane": {"name": name, "input": "tomograms", "variant": variant},
        "data": {"tomogram": show_imports.lane_path(data) + "/{series}_rec.mrc",
                 "tilt_angles": show_imports.lane_path(data) + "/{series}.tlt",
                 "orientation": ORIENTATION, "processing": processing},
        **({"segment": dict(source["segment"])} if "segment" in source else {}),
        "template": dict(source["template"]),
        "pick": {**{key: value for key, value in source["pick"].items()
                    if key != "ctf_weighting"}, "low_pass_angstrom": low_pass},
        "extract": dict(source["extract"]),
        "executables": {key: source["executables"][key] for key in PYTOM_KEYS},
        "run": {"project": show_imports.lane_path(out / "project"), "series": [series],
                "stage": [
                    {"name": "import", "from": "import-tomogram", "until": "import-tomogram",
                     "time": "1:00:00", "sbatch": list(importing["sbatch"])},
                    {"name": "gpu", "from": steps[0], "until": "extract-particles",
                     "time": picking["time"], "sbatch": list(picking["sbatch"])},
                ]},
    }
    header = [
        f"A lane of input tomograms, written by scripts/tomograms_lane_test.py prepare on "
        f"{date.today().isoformat()}",
        f"from {show_imports.lane_path(lane)}, series {series} (docs/entry_and_exit.md, item 6).",
        "The volume is that series' tomogram as IMOD's tilt wrote it, turned by IMOD's",
        "clip rotx into IMOD's standard orientation, as etomo hands one over, with the",
        "angles it was reconstructed with. Template, picking, extraction and partitions",
        "are the source lane's; [pick] ctf_weighting is left out, since the import has no",
        f"CTF fit, and low_pass_angstrom is the {low_pass:g} A the source picks used.",
        "Check it with: python3 scripts/tomograms_lane_test.py check --test "
        + show_imports.lane_path(out),
    ]
    text = lane_text(header, cfg)
    (out / LANE).write_text(text)
    import tomllib

    run_series.build_steps(tomllib.loads(text), out / "project", series)  # refuses here, if at all
    reading = show_imports.read_volume(volume, wedge=True)
    record = {
        "written": date.today().isoformat(),
        "source_lane": show_imports.lane_path(lane),
        "source_lane_sha256": _sha256(lane),
        "source_project": show_imports.lane_path(project),
        "series": series,
        "variant": variant,
        "source_tomogram": tomogram.id,
        "source_tomogram_path": str(tomogram.path),
        "source_picks": picks.id,
        "rec": str(rec),
        "command": ["clip", *command[1:]],
        "volume_sha256": _sha256(volume),
        "template_map": None if cached is None else show_imports.lane_path(cached),
        "tilt_angles": str(angles),
        "processing": processing,
        "reader": {"orientation": reading.get("orientation"),
                   "why": reading.get("orientation_from"),
                   "wedge": reading.get("wedge", {}).get("verdict")},
    }
    (out / RECORD).write_text(json.dumps(record, indent=2) + "\n")
    return out


# --- check ---------------------------------------------------------------------------


def compare_volumes(test: Any, source: Any) -> tuple[float, float]:
    """The largest difference from the source, and from the source reversed in z."""

    if tuple(test.shape) != tuple(source.shape):
        return math.inf, math.inf
    depth = int(source.shape[0])
    same = mirrored = 0.0
    for start in range(0, depth, BLOCK):
        stop = min(start + BLOCK, depth)
        block = np.asarray(test[start:stop], dtype=np.float64)
        same = max(same, float(np.max(np.abs(block - np.asarray(source[start:stop])))))
        reverse = np.asarray(source[depth - stop:depth - start])[::-1]
        mirrored = max(mirrored, float(np.max(np.abs(block - reverse))))
    return same, mirrored


def centres(picks: Artifact) -> np.ndarray:
    """The picks as particle centres, x y z in the tomogram's voxels."""

    names = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")
    for columns, rows in read_star_blocks(Path(picks.path)).values():
        if all(name in columns for name in names):
            table = np.array([[float(row[columns.index(name)]) for name in names]
                              for row in rows]).reshape(-1, 3)
            return table - float(pick_centre_offset(picks))
    raise TestRefused(f"{picks.path} holds no rlnCoordinateX/Y/Z")


def matched(points: np.ndarray, others: np.ndarray, radius: float) -> int:
    """How many of `points` have one of `others` within `radius`."""

    if not len(points) or not len(others):
        return 0
    count = 0
    for start in range(0, len(points), 512):
        chunk = points[start:start + 512]
        nearest = np.min(np.linalg.norm(chunk[:, None, :] - others[None, :, :], axis=2), axis=1)
        count += int(np.sum(nearest <= radius))
    return count


def chance(others: np.ndarray, radius: float) -> float:
    """The share of random positions in `others`' bounding box with one of them near."""

    if len(others) < 2:
        return math.nan
    extent = np.maximum(others.max(axis=0) - others.min(axis=0), 2 * radius)
    density = len(others) / float(np.prod(extent))
    return 1.0 - math.exp(-density * 4.0 / 3.0 * math.pi * radius**3)


def same_data(first: str, second: str) -> bool:
    """Whether two files a search read hold the same data: MRC files by their arrays,
    so that a header alone does not count, other files by their bytes."""

    one, other = Path(first), Path(second)
    if not (one.is_file() and other.is_file()):
        return False
    if one.suffix == other.suffix == ".mrc":
        return bool(np.array_equal(mrcfile.read(one), mrcfile.read(other)))
    return one.read_bytes() == other.read_bytes()


def search_differences(here: Artifact, there: Artifact) -> list[str] | None:
    """What the two pytom searches differ in besides the tomogram; None without both
    job files.

    The template and its mask are files of each project, so two paths that hold the
    same data are the same template. What the picks record and the job file does
    not, such as the volume split and the search mask's coverage, is set side by
    side too.
    """

    lines = repick.search_differences(here, there, names=("here", "source"))
    if lines is None:
        return None
    jobs = [json.loads(Path(str(picks.parameters["pytom_job_json_path"])).read_text())
            for picks in (here, there)]
    same = {key for key in set(jobs[0]) & set(jobs[1])
            if isinstance(jobs[0][key], str) and isinstance(jobs[1][key], str)
            and same_data(jobs[0][key], jobs[1][key])}
    lines = [line for line in lines if line.split(":", 1)[0] not in same]
    for key in RECORDED_SEARCH:
        ours, theirs = here.parameters.get(key), there.parameters.get(key)
        if ours != theirs:
            lines.append(f"{key}: {json.dumps(ours)} (here) vs {json.dumps(theirs)} (source)")
    return lines


def check(test: Path, radius: float) -> int:
    record = json.loads((test / RECORD).read_text())
    _, cfg = run_series.read_config(test / LANE)
    series, variant = record["series"], record["variant"]
    project = ROOT / cfg["run"]["project"]
    print(f"tomograms lane test {show_imports.lane_path(test)}: {record['source_lane']}, "
          f"series {series}")
    print(f"  file        {record['command'][0]} {' '.join(record['command'][1:3])}; the reader "
          f"read it as {record['reader']['orientation']} ({record['reader']['why']})")
    registry_path = project / "artifacts.json"
    if not registry_path.is_file():
        print("  the lane has not run: no registry in its project yet")
        return 2
    registry = ArtifactRegistry.load(registry_path)
    imported = [artifact for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)
                if artifact.covers(series) and artifact.variant == variant]
    if not imported:
        print("  the import has not run")
        return 2
    tomogram = imported[0]
    folder = Path(str(tomogram.parameters.get("run_path", tomogram.path.parent)))
    qc = json.loads((folder / "import_qc.json").read_text()) if (
        folder / "import_qc.json").is_file() else {}
    wedge = qc.get("wedge", {})
    print(f"  import      {tomogram.id}, {tomogram.provenance.value}; wedge check: "
          f"{wedge.get('verdict', 'not recorded')}: {wedge.get('reason', '')}")

    source_registry = ArtifactRegistry.load(ROOT / record["source_project"] / "artifacts.json")
    source = source_registry.get(record["source_tomogram"])
    source_volume = zarr.open_array(str(source.path), mode="r")
    same, mirrored = compare_volumes(zarr.open(str(tomogram.path), mode="r"), source_volume)
    if same == 0.0:
        volume = "identical"
    elif mirrored == 0.0:
        volume = "the source reversed in z"
    else:
        volume = "different"
    print(f"  volume      {volume}: largest difference from the source's tomogram {same:g}, "
          f"from it reversed in z {mirrored:g}")
    # A symmetric range leaves the wedge blind to a mirror; the volume decides then.
    holds = volume == "identical" and wedge.get("verdict") in ("agrees", "side not testable")

    found = [artifact for artifact in registry.by_kind(ArtifactKind.PICKS)
             if artifact.variant == variant
             and artifact.parameters.get("input_tomogram_id") == tomogram.id]
    if not found:
        print("  picks       none yet: the GPU stage has not run")
        print(f"verdict: {'the import holds' if holds else 'the import FAILS'}; picks to come")
        return 0 if holds else 1
    source_picks = source_registry.get(record["source_picks"])
    here = centres(found[0])
    there = centres(source_picks)
    depth = int(source_volume.shape[0])
    mirror = here * np.array([1.0, 1.0, -1.0]) + np.array([0.0, 0.0, depth - 1.0])
    ours, theirs = matched(here, there, radius), matched(there, here, radius)
    print(f"  picks       {len(here)} here, {len(there)} in the source lane; within "
          f"{radius:g} voxels: {ours} of these ({ours / len(here):.1%}), {theirs} of the "
          f"source's ({theirs / max(len(there), 1):.1%})")
    mirrored_share = matched(mirror, there, radius) / len(here)
    print(f"              against: these mirrored in z {mirrored_share:.1%}, random positions "
          f"in the source picks' region {chance(there, radius):.1%}")
    differing = search_differences(found[0], source_picks)
    if differing is None:
        print("              the two pytom searches not compared: a job file is missing")
    elif not differing:
        print("              the two pytom searches differ in nothing but the tomogram's file")
    else:
        print("              the two pytom searches differ, besides the tomogram, in:")
        for line in differing:
            print(f"                {line}")
    stacks = [artifact for artifact in registry.by_kind(ArtifactKind.PARTICLE_STACK)
              if artifact.variant == variant and artifact.covers(series)]
    if stacks:
        shape = stacks[0].shape
        print(f"  particles   {shape[0] if shape else '?'} extracted, {stacks[0].id}")
    print(f"verdict: {'the import holds' if holds else 'the import FAILS'}: volume {volume}, "
          f"wedge {wedge.get('verdict', 'not recorded')}")
    return 0 if holds else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    making = steps.add_parser("prepare", help="write the test's volume, angles and lane")
    making.add_argument("--lane", type=Path, required=True,
                        help="the lane that reconstructed and picked the series here")
    making.add_argument("--series", required=True)
    making.add_argument("--picks", help="the source picks to compare with, when the series "
                        "has several under the lane's label")
    making.add_argument("--out", type=Path,
                        help="the test's folder (default <lane name>-tomograms-test beside "
                             "the source project)")
    checking = steps.add_parser("check", help="set the lane's results against the known answer")
    checking.add_argument("--test", type=Path, required=True, help="the folder prepare wrote")
    checking.add_argument("--radius", type=float, default=MATCH_RADIUS_PX,
                          help=f"voxels within which two picks are one (default "
                               f"{MATCH_RADIUS_PX:g})")
    args = parser.parse_args(argv)
    if args.step == "prepare":
        out = prepare(args.lane, args.series, args.out, args.picks)
        record = json.loads((out / RECORD).read_text())
        shown = show_imports.lane_path(out)
        print(f"wrote {shown}: {LANE}, {RECORD}, data/{args.series}_rec.mrc and .tlt")
        print(f"the reader reads the file as {record['reader']['orientation']}: "
              f"{record['reader']['why']}")
        print("next, from the repository root:")
        for line in (f"python3 scripts/show_imports.py tomograms {shown}/data",
                     f"python3 scripts/submit_lanes.py {shown}/{LANE} --dry-run",
                     f"python3 scripts/submit_lanes.py {shown}/{LANE}",
                     f"python3 scripts/tomograms_lane_test.py check --test {shown}"):
            print(f"  {line}")
        return 0
    return check(args.test, args.radius)


if __name__ == "__main__":
    sys.exit(main())
