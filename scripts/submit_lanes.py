#!/usr/bin/env python3
"""Submit a lane as chained Slurm jobs, from the [run] section of its config.

Run from the repo root on the submit node:

    python3 scripts/submit_lanes.py configs/lanes/ribo-bin8.toml
    python3 scripts/submit_lanes.py configs/lanes/ribo-bin8.toml configs/lanes/gag-bin8.toml
    python3 scripts/submit_lanes.py configs/lanes/gag-bin8.toml --from-stage gpu
    python3 scripts/submit_lanes.py configs/lanes/gag-bin8.toml --dry-run

[run] names the project, its series, and how the chain in scripts/run_series.py
is cut into jobs. Each stage is one scripts/lane.sbatch job over every series,
steps `from`..`until`, with its own sbatch resources and time limit; stages are
joined by afterok, so one starts only when the previous finished without a
failed series. The cut is not the order: stages must be consecutive slices of
the fixed chain, and a gap or an overlap is refused here, before anything is
submitted. The step order stays in code (see scripts/run_series.py).

Two lanes given together run side by side; every config is checked before any
job is submitted. The environment goes to the jobs with --export=ALL, so set
what the config's `$VAR` values name (PYTOMO_MP_DIR, MOTIONCOR3, MEMBRAIN_EXE,
...) before running; a missing one is refused here rather than in the job.
OVERWRITE=1 in the environment reaches lane.sbatch the same way.

[run] series may be left out, to take every series: a frames lane then gets the
ones `init` finds, or, for a project that exists, the ones it has manifests for;
a stack lane gets every series its [data] stack path matches, with {series} as
the wildcard.

For a frames lane whose project does not exist yet, `cryoet init` runs here
first, from [data] frames and either mdocs or frames_pattern (tilt metadata
read from the movie file names). It is never run on an existing project:
it starts the registry empty. Not done here: downloading data and caching EMDB
maps — see the README.

Disposable, like scripts/run_series.py.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402

LANE_SBATCH = ROOT / "scripts" / "lane.sbatch"
_ENV_REFERENCE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


@dataclass(frozen=True)
class Stage:
    name: str
    start: str
    end: str
    time: str
    sbatch: tuple[str, ...]
    env: frozenset[str]  # the $VARs the stage's commands name


@dataclass(frozen=True)
class Plan:
    config: Path
    project: str
    series: tuple[str, ...]
    stages: tuple[Stage, ...]


def _string_list(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SystemExit(f"{where} must be a list of strings")
    return tuple(value)


def plan(cfg: dict[str, Any], config: Path) -> Plan:
    """The lane's stages, checked against its chain. Refuses rather than guesses."""

    run = cfg.get("run")
    if not isinstance(run, dict):
        raise SystemExit(f"{config}: no [run] section")
    project = run.get("project")
    if not isinstance(project, str) or not project:
        raise SystemExit(f"{config}: [run] project must be a path")
    if run.get("series") is None:
        # Every series. A stack lane finds them now, in its stack paths; a frames
        # lane only once init has run (see project_series).
        series = stack_series(cfg, config) if run_series.lane_input(cfg) == "stack" else ()
    else:
        series = _string_list(run.get("series"), f"{config}: [run] series")
        if not series:
            raise SystemExit(f"{config}: [run] series is empty; leave it out for every series")
    if len(set(series)) != len(series):
        raise SystemExit(f"{config}: [run] series lists a series twice")
    raw_stages = run.get("stage")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise SystemExit(f"{config}: [run] has no [[run.stage]]")

    # The chain is the same for every series, and a frames lane may not know its
    # series yet: any id renders it.
    example = series[0] if series else "TS_X"
    chain = [name for name, _ in run_series.build_steps(cfg, Path(project), example)]
    stages: list[Stage] = []
    for raw in raw_stages:
        where = f"{config}: [[run.stage]] {raw.get('name', '?')!r}"
        missing = [key for key in ("name", "from", "until", "time", "sbatch") if key not in raw]
        if missing:
            raise SystemExit(f"{where}: missing {', '.join(missing)}")
        for key in ("from", "until"):
            if raw[key] not in chain:
                raise SystemExit(
                    f"{where}: unknown step {raw[key]!r} in {key}\n"
                    f"known steps: {', '.join(chain)}"
                )
        lo, hi = chain.index(raw["from"]), chain.index(raw["until"])
        if lo > hi:
            raise SystemExit(f"{where}: {raw['from']!r} comes after {raw['until']!r}")
        if stages:
            expected = chain.index(stages[-1].end) + 1
            if lo != expected:
                gap = "skips" if lo > expected else "repeats"
                between = ", ".join(chain[min(lo, expected) : max(lo, expected)])
                raise SystemExit(
                    f"{where}: starts at {raw['from']!r}, but the previous stage ends at "
                    f"{stages[-1].end!r}, so it {gap} {between}"
                )
        if raw["name"] in {stage.name for stage in stages}:
            raise SystemExit(f"{where}: two stages share the name")

        # Every series renders the same $VARs; the paths differ, the variables do not.
        sliced = run_series.slice_steps(
            run_series.build_steps(cfg, Path(project), example), raw["from"], raw["until"]
        )
        env = frozenset(
            match
            for _, command in sliced
            for part in command
            for match in _ENV_REFERENCE.findall(part)
        )
        stages.append(
            Stage(
                name=raw["name"],
                start=raw["from"],
                end=raw["until"],
                time=str(raw["time"]),
                sbatch=_string_list(raw["sbatch"], f"{where}: sbatch"),
                env=env,
            )
        )
    return Plan(config=config, project=project, series=series, stages=tuple(stages))


def stack_series(cfg: dict[str, Any], config: Path) -> tuple[str, ...]:
    """Every series a stack lane's [data] stack path matches, {series} its wildcard."""

    template = cfg.get("data", {}).get("stack")
    if not isinstance(template, str) or template.count("{series}") != 1:
        raise SystemExit(
            f"{config}: without [run] series, [data] stack must contain {{series}} once, "
            "so that the series can be found"
        )
    prefix, suffix = os.path.join(ROOT, template).split("{series}")
    found = sorted(
        path[len(prefix) : len(path) - len(suffix)]
        for path in glob.glob(glob.escape(prefix) + "*" + glob.escape(suffix))
    )
    if not found:
        raise SystemExit(f"{config}: no file matches [data] stack {template!r}")
    return tuple(found)


def project_series(project: str) -> tuple[str, ...]:
    """The series a project has manifests for: what init wrote, or import-tilt-stack."""

    return tuple(sorted(path.stem for path in (ROOT / project / "manifests").glob("*.json")))


def problems(cfg: dict[str, Any], lane: Plan, stages: tuple[Stage, ...]) -> list[str]:
    """What would make the submitted jobs fail at once, found on the submit node."""

    found = [
        f"${name} is not set (used by stage {stage.name!r})"
        for stage in stages
        for name in sorted(stage.env)
        if not os.environ.get(name)
    ]
    if run_series.lane_input(cfg) != "frames":
        # A stack lane writes its own manifests in import-tilt-stack; only its
        # inputs can be checked here, which saves a queue wait on a typo.
        data = cfg.get("data", {})
        for series in lane.series:
            for key in ("mdoc", "stack"):
                if isinstance(data.get(key), str):
                    path = ROOT / data[key].format(series=series)
                    if not path.is_file():
                        found.append(f"[data] {key}: no file {path}")
        return found
    data = cfg.get("data", {})
    if init_command(cfg, lane) is not None:
        if ("mdocs" in data) == ("frames_pattern" in data):
            found.append(
                "[data] needs exactly one of mdocs and frames_pattern; init reads "
                "the tilt metadata from one of them"
            )
        for key in ("frames", "mdocs"):
            if key == "mdocs" and "mdocs" not in data:
                continue
            if not isinstance(data.get(key), str):
                found.append(f"[data] {key} is not set; init needs it for a new project")
            elif not (ROOT / data[key]).is_dir():
                found.append(f"[data] {key}: no directory {ROOT / data[key]}")
        return found
    if not lane.series and not project_series(lane.project):
        found.append(f"{lane.project} exists but has no manifests to take every series from")
    manifests = ROOT / lane.project / "manifests"
    absent = [s for s in lane.series if not (manifests / f"{s}.json").is_file()]
    if absent:
        # init writes a new, empty registry, so it is never run on a project that
        # already has one: that would throw away every series processed so far.
        found.append(
            f"{lane.project} already exists but has no manifest for {', '.join(absent)}, "
            "and init would erase the project to add them; use a new project for them"
        )
    return found


def init_command(cfg: dict[str, Any], lane: Plan) -> list[str] | None:
    """`cryoet init` for a frames lane whose project does not exist yet, else None.

    init is project-level and quick (it reads the mdocs and checks the movies are
    there), so it runs here on the submit node rather than in a job. It is never
    run on an existing project, because it starts the registry empty.
    """
    if run_series.lane_input(cfg) != "frames":
        return None
    if (ROOT / lane.project / "artifacts.json").exists():
        return None
    data = cfg.get("data", {})
    command = ["cryoet", "init", "--frames", str(data.get("frames"))]
    if "frames_pattern" in data:
        command += ["--frames-pattern", str(data["frames_pattern"])]
    else:
        command += ["--mdocs", str(data.get("mdocs"))]
    if "pixel_size_angstrom" in data:
        command += ["--pixel-size", str(data["pixel_size_angstrom"])]
    command += ["--out", lane.project]
    for series in lane.series:
        command += ["--tilt-series", series]
    return command


def _repo_relative(path: Path) -> str:
    """lane.sbatch cds to the repo, so a relative path has to be from there."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def submit(lane: Plan, stages: tuple[Stage, ...], dry_run: bool) -> list[str]:
    job_ids: list[str] = []
    prefix = Path(lane.project).name
    for stage in stages:
        after = (
            [f"--dependency=afterok:{job_ids[-1]}", "--kill-on-invalid-dep=yes"]
            if job_ids
            else []
        )
        command = [
            "sbatch", "--parsable", "--export=ALL",
            *stage.sbatch, *after,
            f"--time={stage.time}", f"--job-name={prefix}-{stage.name}",
            str(LANE_SBATCH.relative_to(ROOT)),
        ]
        # Through the environment rather than --export=VAR=value, whose list is
        # comma-separated: SERIES holds spaces, and nothing has to survive a parser.
        if not lane.series and not dry_run:
            raise SystemExit(f"{lane.config}: no series to submit")
        env = {
            "PROJECT": lane.project,
            "CONFIG": _repo_relative(lane.config),
            # Unknown in a dry run of a new project without [run] series: init,
            # which finds them, has not run.
            "SERIES": " ".join(lane.series) or "<every series init finds>",
            "FROM": stage.start,
            "UNTIL": stage.end,
        }
        if dry_run:
            shown = " ".join(f"{key}={value!r}" for key, value in env.items())
            print(f"{shown} {' '.join(command)}")
            job_ids.append(f"<{stage.name}>")
            continue
        result = subprocess.run(
            command, cwd=ROOT, env={**os.environ, **env},
            check=True, capture_output=True, text=True,
        )
        # --parsable prints "id;cluster" on a multi-cluster setup.
        job_ids.append(result.stdout.strip().split(";")[0])
    return job_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("configs", nargs="+", type=Path, help="lane configs with a [run] section")
    parser.add_argument(
        "--from-stage",
        default=None,
        help="submit this stage and the ones after it, e.g. to resume after a failed "
        "job; with several configs, every one of them must have it",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the sbatch commands and any problems; submit nothing",
    )
    args = parser.parse_args()

    # Everything is checked before the first sbatch, so a broken second config
    # does not leave the first lane submitted alone.
    lanes = []
    for config in args.configs:
        cfg = tomllib.loads(config.read_text())
        lane = plan(cfg, config)
        names = [stage.name for stage in lane.stages]
        if args.from_stage is not None and args.from_stage not in names:
            raise SystemExit(f"{config}: no stage {args.from_stage!r}; stages: {', '.join(names)}")
        first = names.index(args.from_stage) if args.from_stage else 0
        stages = lane.stages[first:]
        lanes.append((cfg, lane, stages, problems(cfg, lane, stages)))

    projects = [lane.project for _, lane, _, _ in lanes]
    if len(set(projects)) != len(projects):
        # Two jobs on the same series write the same files.
        raise SystemExit("two configs submit to the same project; submit them one after the other")

    found = [f"{lane.config}: {problem}" for _, lane, _, issues in lanes for problem in issues]
    for problem in found:
        print(f"{'would refuse' if args.dry_run else 'refused'}: {problem}", file=sys.stderr)
    if found and not args.dry_run:
        return 1

    for cfg, lane, stages, _ in lanes:
        print(f"# {lane.config}: {lane.project}, "
              + (f"{len(lane.series)} series" if lane.series else "every series"))
        init = init_command(cfg, lane)
        if init is not None:
            print(("would run: " if args.dry_run else "") + shlex.join(init))
            if not args.dry_run:
                subprocess.run(init, cwd=ROOT, check=True)
        if not lane.series and (init is None or not args.dry_run):
            # Every series: the ones init just wrote manifests for, or the ones
            # the project already has.
            lane = replace(lane, series=project_series(lane.project))
            print(f"  series: {' '.join(lane.series)}")
        ids = submit(lane, stages, args.dry_run)
        chained = zip(stages, ids, strict=True)
        print("  " + " -> ".join(f"{stage.name} {job}" for stage, job in chained))
    if not args.dry_run:
        print("watch: squeue -u $USER   logs: slurm-<job name>-<id>.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
