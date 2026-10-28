#!/usr/bin/env python3
"""Submit a recipe, one of the experiments written in advance for a lane's series,
with the checks and the journal of scripts/submit_lanes.py.

Run from the repo root on the submit node:

    python3 scripts/submit_recipe.py --list
    python3 scripts/submit_recipe.py ctf-window configs/lanes/ryr1-bin8.toml --series 039 --dry-run
    python3 scripts/submit_recipe.py repick configs/lanes/ribo-mdc.toml --set LABEL=bin8-solved
    python3 scripts/submit_recipe.py average configs/lanes/ribo-bin8.toml --overwrite \\
        --set THRESHOLDS=outputs/ribo2/dynamo/pick-qc/thresholds.json --set MASK=381.6

A recipe is a batch script in scripts/ that runs the lane's own steps, as
scripts/run_series.py renders them from the config, in an experiment written in
advance. RECIPES below lists the ones submitted this way: what each runs, which
variables it needs, and what it writes. One recipe is one job, named
<project>-<recipe>, over the series --series names; a recipe that takes several
gets the lane's when none are named. Its parameters are given as --set NAME=VALUE,
and reach the job from there alone.

Where the job runs (partition, QOS, account, GPU) is the lane's choice for the
step the recipe repeats: the placement options of the [[run.stage]] that runs it,
so a recipe needs no site of its own. What the job needs (CPUs, memory, time) is
the recipe's, in its #SBATCH lines; --time asks for longer.

The checks are submit_lanes.py's. --overwrite is the only way to let the job
replace what is there: OVERWRITE set in the environment is refused, and so is
one of the recipe's parameters, since --export=ALL would hand either to the job
unseen. A series that a queued or running job of the project works on is
refused. So is what would stop the job at once: a variable unset, a series
without a manifest, or outputs already under the recipe's labels without
--overwrite. Every dry run and submission goes into the project's journal.jsonl,
and a submission from the assistant needs --reason and must be the same person's
last dry run of the same recipe on the same series, unchanged, at most 30
minutes old.

The batch scripts still run when given to sbatch by hand, but then none of this
happens, and the check for busy series sees such a job only once its log names
the project.

Disposable, like scripts/submit_lanes.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import submit_lanes  # noqa: E402

# The sbatch options that say where a job runs, as opposed to how much it needs.
_PLACEMENT = re.compile(r"^--(partition|qos|account|gres|gpus|constraint|reservation|clusters)=")
_SCRIPT_TIME = re.compile(r"^#SBATCH\s+--time=(\S+)", re.MULTILINE)


@dataclass(frozen=True)
class Context:
    """What a recipe's own check looks at."""

    cfg: dict[str, Any]
    project: str
    series: tuple[str, ...]
    parameters: dict[str, str]
    commands: dict[str, str]  # the lane's steps the recipe runs, rendered for one series


@dataclass(frozen=True)
class Recipe:
    script: str  # in scripts/
    what: str
    one_series: bool
    steps: tuple[str, ...]  # the lane's steps it runs, as run_series.py renders them
    where: str  # the step whose [[run.stage]] says where the job runs
    needs: tuple[str, ...] = ()  # environment variables its job stops without
    required: tuple[str, ...] = ()  # parameters
    optional: tuple[str, ...] = ()
    # What it writes under its labels, as (artifact kind, label), from the lane's
    # label and the parameters. Already there, it stops the job unless given
    # --overwrite, apart from the kinds the recipe reuses.
    outputs: Callable[[str, dict[str, str]], list[tuple[str, str]]] = lambda lane, given: []
    reuses: frozenset[str] = frozenset()
    # What --overwrite replaces outside its labels, in words.
    shared: Callable[[dict[str, str]], str | None] = lambda given: None
    check: Callable[[Context], list[str]] = lambda context: []

    @property
    def parameters(self) -> tuple[str, ...]:
        return (*self.required, *self.optional)


def _export_free(context: Context) -> list[str]:
    export = context.parameters.get("EXPORT")
    if export and (ROOT / export).is_dir() and any((ROOT / export).iterdir()):
        return [f"EXPORT {export} already holds files; name a new EXPORT"]
    return []


def _null_decoys_problems(context: Context) -> list[str]:
    lane = context.cfg["lane"]["variant"]
    found = _export_free(context)
    if context.parameters.get("VARIANT") == lane:
        found.append(f"VARIANT must differ from the lane's label, {lane!r}")
    return found


def _repick_problems(context: Context) -> list[str]:
    lane = context.cfg["lane"]["variant"]
    null = context.parameters.get("NULL_LABEL") or f"{lane}-null"
    found = _export_free(context)
    if context.parameters["LABEL"] in (lane, null):
        found.append(f"LABEL must differ from the lane's label {lane!r} and the decoys' {null!r}")
    return found


# What scripts/pick_ablation.sbatch edits in the lane's commands. The job refuses a
# lane that no longer renders one of them, and so does its dry run here.
ABLATION_FLAGS = (
    " --pytom-random-phase-correction", " --pytom-low-pass ", " --pytom-cut-off 0",
    " --template-box-size-px ", " --template-mask-sigma-px ",
)


def _ablation_problems(context: Context) -> list[str]:
    rendered = " ".join(context.commands.values())
    missing = [flag.strip() for flag in ABLATION_FLAGS if flag not in rendered]
    if not missing:
        return []
    return [
        f"the lane no longer renders {', '.join(missing)}, which the recipe edits; it was "
        "written for the ribo lanes as they were before 2026-10-15"
    ]


def _average_problems(context: Context) -> list[str]:
    found = []
    thresholds = context.parameters["THRESHOLDS"]
    try:
        values = json.loads((ROOT / thresholds).read_text())
    except (OSError, ValueError) as error:
        found.append(f"THRESHOLDS {thresholds} cannot be read: {error}")
    else:
        # As the job reads them: a number per series, and no null placeholder.
        absent = [
            one for one in context.series
            if not isinstance(values, dict)
            or isinstance(values.get(one), bool)
            or not isinstance(values.get(one), int | float)
        ]
        if absent:
            found.append(
                f"THRESHOLDS {thresholds} has no numeric threshold for {', '.join(absent)}"
            )
    mask = context.parameters["MASK"]
    if not (ROOT / mask).is_file():
        try:
            float(mask)
        except ValueError:
            found.append(f"MASK {mask!r} is neither a diameter in A nor a file")
    return found


CTF_WINDOWS = ("ctfw1", "ctfw5", "ctfw10", "ctfw5-tilt")
ABLATIONS = (
    "bin8-abl-lane", "bin8-abl-new25", "bin8-abl-new25-nr", "bin8-abl-new25-pl",
    "bin8-abl-new35-pl",
)

# Every recipe that may be submitted, by the name it is submitted under. A batch
# script that is not here is not a recipe: it is run by hand, or not at all.
RECIPES = {
    "ctf-window": Recipe(
        script="ctf_window_test.sbatch",
        what="estimate-ctf four ways on one series: a search window of +-1, 5 and 10 kA, "
        "and 5 kA with CTFFIND5's sample tilt, labelled ctfw1, ctfw5, ctfw10, ctfw5-tilt; "
        "then scripts/ctf_fit_table.py",
        one_series=True,
        steps=("build-ctf-stack", "estimate-ctf"),
        where="estimate-ctf",
        needs=("IMOD_DIR",),
        optional=("KEEP_STACK",),
        outputs=lambda lane, given: [("ctf_model", label) for label in CTF_WINDOWS],
        shared=lambda given: None if given.get("KEEP_STACK") == "1" else (
            "the series' unlabelled CTF stack, which every lane of the project uses, "
            "built again (build-ctf-stack)"
        ),
    ),
    "pick-ablation": Recipe(
        script="pick_ablation.sbatch",
        what="pick-particles five ways on one series (the template, the low-pass, pytom's "
        "two corrections), labelled bin8-abl-*, one task at a time; each ends with "
        "scripts/score_band_readout.py",
        one_series=True,
        steps=("prepare-template", "pick-particles"),
        where="pick-particles",
        needs=("PYTOMO_MP_DIR",),
        # Its two templates are made when missing and used again otherwise.
        outputs=lambda lane, given: [("picks", label) for label in ABLATIONS],
        check=_ablation_problems,
    ),
    "null-decoys": Recipe(
        script="null_decoys.sbatch",
        what="the lane's pick and extraction on phase-randomised copies of its tomograms, "
        "labelled <lane>-null unless VARIANT says otherwise; with EXPORT, a Dynamo export "
        "of the real particles followed by these decoys",
        one_series=False,
        steps=("pick-particles", "extract-particles"),
        where="pick-particles",
        needs=("PYTOMO_MP_DIR",),
        optional=("EXPORT", "COUNT", "SEED", "VARIANT"),
        outputs=lambda lane, given: [
            (kind, given.get("VARIANT") or f"{lane}-null")
            for kind in ("tomogram", "picks", "particle_stack")
        ],
        # A copy made before from the same tomogram and seed is used again.
        reuses=frozenset({"tomogram"}),
        check=_null_decoys_problems,
    ),
    "repick": Recipe(
        script="repick.sbatch",
        what="the lane's pick and extraction again under LABEL, compared with the lane's "
        "picks and the null decoys'; with EXPORT, a Dynamo export of the new picks "
        "followed by the null decoys",
        one_series=False,
        steps=("pick-particles", "extract-particles"),
        where="pick-particles",
        needs=("PYTOMO_MP_DIR",),
        required=("LABEL",),
        optional=("EXPORT", "NULL_LABEL"),
        outputs=lambda lane, given: [
            (kind, given["LABEL"]) for kind in ("picks", "particle_stack")
        ],
        check=_repick_problems,
    ),
    "average": Recipe(
        script="average_lane.sbatch",
        what="every series extracted again at its threshold from scripts/qc_dynamo_picks.py, "
        "then averaged together with independent halves (average-particles --gold-standard)",
        one_series=False,
        steps=("extract-particles",),
        # Averaging runs on CPUs, where the lane reconstructs.
        where="reconstruct-tomogram",
        required=("THRESHOLDS", "MASK"),
        optional=("ITERATIONS", "START_LOWPASS_ANGSTROM", "VARIANT"),
        outputs=lambda lane, given: [
            (kind, given.get("VARIANT") or lane)
            for kind in ("particle_stack", "subtomogram_average")
        ],
        check=_average_problems,
    ),
}


def parse_parameters(name: str, recipe: Recipe, given: list[str]) -> dict[str, str]:
    """The --set values, each one the recipe reads, and every one it needs."""
    parameters: dict[str, str] = {}
    for item in given:
        key, sign, value = item.partition("=")
        if not sign or not value:
            raise SystemExit(f"--set {item!r}: give NAME=VALUE")
        if key not in recipe.parameters:
            known = ", ".join(recipe.parameters) or "none"
            raise SystemExit(f"{name} reads no {key}; its parameters: {known}")
        if key in parameters:
            raise SystemExit(f"--set {key} is given twice")
        parameters[key] = value
    missing = [key for key in recipe.required if key not in parameters]
    if missing:
        raise SystemExit(f"{name} needs {', '.join(f'--set {key}=...' for key in missing)}")
    return parameters


def lane_stage(
    lane: submit_lanes.Plan, chain: list[str], step: str
) -> submit_lanes.Stage | None:
    """The lane's stage that runs this step, if any does."""
    if step not in chain:
        return None
    at = chain.index(step)
    for stage in lane.stages:
        if chain.index(stage.start) <= at <= chain.index(stage.end):
            return stage
    return None


def _scope(artifact: dict[str, Any]) -> list[str]:
    """The series a registry entry is about, as the registry's model reads it."""
    listed = artifact.get("tilt_series_ids")
    if isinstance(listed, list) and listed:
        return [str(one) for one in listed]
    parameters = artifact.get("parameters") or {}
    listed = parameters.get("tilt_series_ids")
    if isinstance(listed, list):
        return [str(one) for one in listed]
    single = parameters.get("tilt_series_id")
    return [single] if isinstance(single, str) else []


def registered(
    project: str, series: tuple[str, ...], wanted: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    """Per (kind, label) wanted, the series the project's registry already holds
    one for."""
    try:
        payload = json.loads((ROOT / project / "artifacts.json").read_text())
    except (OSError, ValueError):
        return []
    found: dict[tuple[str, str], set[str]] = {}
    artifacts = payload.get("artifacts") if isinstance(payload, dict) else None
    for artifact in artifacts or []:
        key = (artifact.get("kind"), artifact.get("variant"))
        if key in wanted:
            covered = set(_scope(artifact)) & set(series)
            if covered:
                found.setdefault(key, set()).update(covered)
    return [
        {"kind": kind, "label": label, "series": sorted(found[(kind, label)])}
        for kind, label in wanted
        if (kind, label) in found
    ]


def script_time(recipe: Recipe) -> str | None:
    """The time limit the recipe's #SBATCH lines ask for."""
    match = _SCRIPT_TIME.search((ROOT / "scripts" / recipe.script).read_text())
    return match[1] if match else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recipe", nargs="?", choices=sorted(RECIPES), help="what to run")
    parser.add_argument(
        "config", nargs="?", type=Path, help="the lane config whose steps the recipe runs"
    )
    parser.add_argument(
        "--series", action="append", default=[],
        help="a series to run it on; once for a recipe that takes one, and for the "
        "others the lane's series when left out",
    )
    parser.add_argument(
        "--set", dest="parameters", action="append", default=[], metavar="NAME=VALUE",
        help="one of the recipe's parameters (--list names them)",
    )
    parser.add_argument("--time", default=None, help="a longer time limit than the recipe's")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the sbatch command and any problems; submit nothing",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="let the job replace what it or the lane wrote before, for this submission "
        "only; the dry run lists what may be replaced",
    )
    parser.add_argument(
        "--reason", default=None,
        help="what was asked and by whom, for the project's journal; required from the "
        "assistant",
    )
    parser.add_argument("--list", action="store_true", help="list the recipes")
    args = parser.parse_args(argv)
    command_line = ["scripts/submit_recipe.py", *(sys.argv[1:] if argv is None else argv)]

    if args.list:
        for name, recipe in RECIPES.items():
            settable = [*recipe.required, *(f"[{key}]" for key in recipe.optional)]
            needs = ", ".join(f"${key}" for key in recipe.needs)
            print(
                f"{name}: scripts/{recipe.script}, "
                f"{'one series' if recipe.one_series else 'series of the lane'}"
                + (f"; needs {needs}" if needs else "")
                + (f"; --set {' '.join(settable)}" if settable else "")
            )
            print(f"    {recipe.what}")
        return 0
    if args.recipe is None or args.config is None:
        parser.error("name a recipe and a lane config, or give --list")
    recipe = RECIPES[args.recipe]

    if os.environ.get("OVERWRITE"):
        raise SystemExit(
            "OVERWRITE is set in the environment, and submit_recipe.py does not pass it on: "
            "give --overwrite to replace outputs in this submission only, or unset OVERWRITE"
        )
    inherited = [key for key in recipe.parameters if key in os.environ]
    if inherited:
        raise SystemExit(
            f"{', '.join(inherited)} set in the environment, which the job would read "
            f"unseen: give each as --set NAME=VALUE, or unset {' '.join(inherited)}"
        )
    parameters = parse_parameters(args.recipe, recipe, args.parameters)
    who = submit_lanes._who()
    if who["assistant"] and not args.dry_run and not (args.reason or "").strip():
        raise SystemExit(
            "a submission from the assistant needs --reason: what was asked, and by whom"
        )

    text, cfg = run_series.read_config(args.config)
    lane = submit_lanes.plan(cfg, args.config)  # the config's keys and stages checked
    project = lane.project
    if recipe.one_series and len(args.series) != 1:
        raise SystemExit(f"{args.recipe} runs on one series: name it with --series, once")
    series = (
        tuple(args.series)
        or tuple(cfg["run"].get("series") or ())
        or submit_lanes.project_series(project)
    )
    if not series:
        raise SystemExit(
            f"{project} has no series to run {args.recipe} on: name them with --series"
        )
    if len(set(series)) != len(series):
        raise SystemExit("--series names a series twice")
    time = args.time or script_time(recipe)
    if time is None:
        raise SystemExit(f"scripts/{recipe.script} sets no time limit: give --time")

    steps = run_series.build_steps(cfg, Path(project), series[0])
    chain = [step for step, _ in steps]
    commands = {step: run_series.render(command) for step, command in steps if step in recipe.steps}
    # The variables the job reads from the environment, with what reads them: its
    # own, and the ones the lane's commands name, which --export=ALL hands it.
    variables = dict.fromkeys(recipe.needs, "the recipe's job")
    for step, command in commands.items():
        for key in submit_lanes._ENV_REFERENCE.findall(command):
            variables.setdefault(key, step)

    problems = [f"the lane has no step {step}, which the recipe runs"
                for step in recipe.steps if step not in chain]
    stage = lane_stage(lane, chain, recipe.where)
    if stage is None:
        problems.append(
            f"no [[run.stage]] of the lane runs {recipe.where}, so where the recipe would "
            "run is unknown"
        )
    manifests = ROOT / project / "manifests"
    absent = [one for one in series if not (manifests / f"{one}.json").is_file()]
    if absent:
        problems.append(
            f"{project} has no manifest for {', '.join(absent)}; a recipe runs on series "
            "the lane has processed"
        )
    problems += [
        f"${key} is not set (used by {reader})"
        for key, reader in variables.items()
        if not os.environ.get(key)
    ]
    problems += recipe.check(Context(cfg, project, series, parameters, commands))
    there = registered(project, series, recipe.outputs(cfg["lane"]["variant"], parameters))
    if args.overwrite:
        replaces = {"registered": there, "outside_labels": recipe.shared(parameters)}
    else:
        replaces = {}
        # One line per kind and set of series, however many of the labels hold it.
        held: dict[tuple[str, tuple[str, ...]], list[str]] = {}
        for one in there:
            if one["kind"] not in recipe.reuses:
                held.setdefault((one["kind"], tuple(one["series"])), []).append(one["label"])
        problems += [
            f"{', '.join(owners)} already {'has' if len(owners) == 1 else 'have'} {kind} "
            f"under {', '.join(labels)}, and the job would stop at "
            f"{'it' if len(labels) == 1 else 'them'}: give --overwrite to replace "
            f"{'it' if len(labels) == 1 else 'them'}"
            for (kind, owners), labels in held.items()
        ]
    problems += submit_lanes.queue_problems(
        [(cfg, replace(lane, series=series))], "this recipe"
    )[0]

    options = [
        *(option for option in (stage.sbatch if stage else ()) if _PLACEMENT.match(option)),
        f"--time={time}",
        f"--job-name={Path(project).name}-{args.recipe}",
    ]
    files = {
        key: _sha256((ROOT / value).read_bytes())
        for key, value in sorted(parameters.items())
        if (ROOT / value).is_file()
    }
    record = {
        "recipe": args.recipe,
        "script": f"scripts/{recipe.script}",
        "script_sha256": _sha256((ROOT / "scripts" / recipe.script).read_bytes()),
        "config": submit_lanes._repo_relative(args.config),
        "config_sha256": _sha256(text.encode()),
        "project": project,
        "series": list(series),
        "parameters": dict(sorted(parameters.items())),
        "files": files,
        "sbatch": options,
        "overwrite": args.overwrite,
        "env": {key: os.environ.get(key, "") for key in sorted(variables)},
    }

    for problem in problems:
        print(
            f"{'would refuse' if args.dry_run else 'refused'}: {args.config}: {problem}",
            file=sys.stderr,
        )
    if problems and not args.dry_run:
        return 1
    if who["assistant"] and not args.dry_run:
        # The person agreed to the dry run they were shown, and to nothing else.
        rows = submit_lanes.read_journal(project)
        why = submit_lanes.unshown(rows, record, who["user"], datetime.now(UTC))
        if why:
            print(
                f"refused: {args.config}: {args.recipe}: {why}; dry-run it, show the dry run, "
                "and submit after a yes",
                file=sys.stderr,
            )
            return 1

    # Journaled before sbatch, as a lane is: one dry run stands for one submission.
    entry = {
        **who, "at": submit_lanes._now(), "recipe": args.recipe, "config": record["config"],
        "argv": command_line, "plan": record, "plan_sha256": submit_lanes._digest(record),
        "reason": args.reason, "replaces": replaces,
    }
    if args.dry_run:
        entry |= {"event": "dry-run", "problems": problems}
    else:
        entry |= {
            "event": "submit", "id": uuid.uuid4().hex[:12], "config_text": text,
            "config_diff": submit_lanes.config_diff(cfg, args.config, text),
            "code": submit_lanes.code_version(),
        }
    try:
        journal: Path | None = submit_lanes.write_journal(project, entry)
    except OSError as error:
        if not args.dry_run:
            print(f"refused: {args.config}: cannot write its journal: {error}", file=sys.stderr)
            return 1
        print(f"not journaled: {error}", file=sys.stderr)
        journal = None

    # Through the environment, as for a lane: SERIES holds spaces.
    env = {"PROJECT": project, "CONFIG": record["config"], "SERIES": " ".join(series), **parameters}
    if args.overwrite:
        env["OVERWRITE"] = "1"
    command = ["sbatch", "--parsable", "--export=ALL", *options, f"scripts/{recipe.script}"]
    print(f"# {args.recipe} on {project}: {' '.join(series)}")
    print(f"  {recipe.what}")
    if args.dry_run:
        print(" ".join(f"{key}={value!r}" for key, value in env.items()) + " " + " ".join(command))
    else:
        closing = {"config": record["config"], "recipe": args.recipe, "id": entry["id"],
                   "user": who["user"]}
        try:
            job = submit(recipe, command, env)
        except submit_lanes.SubmitError as error:
            _close(project, closing, "submit-failed", jobs={}, error=str(error))
            print(f"refused: {args.config}: {error}", file=sys.stderr)
            return 1
        _close(project, closing, "submitted", jobs={args.recipe: job}, series=list(series))
        print(f"  {args.recipe} {job}")
    if args.overwrite:
        if there:
            print("  --overwrite may replace what is registered under its labels:")
            for one in there:
                print(f"    {one['kind']} under {one['label']}: {', '.join(one['series'])}")
        else:
            print("  --overwrite: nothing is registered under its labels yet")
        if replaces["outside_labels"]:
            print(f"  and outside its labels: {replaces['outside_labels']}")

    if args.dry_run:
        if journal is not None:
            print(f"recorded in {submit_lanes._repo_relative(journal)}")
        if who["assistant"]:
            print(submit_lanes.next_step(bool(problems)))
    else:
        print(f"journal: {submit_lanes._repo_relative(journal) if journal else '?'}")
        print(f"watch: squeue -u $USER   log: slurm-{Path(project).name}-{args.recipe}-{job}*.log")
    return 0


def submit(recipe: Recipe, command: list[str], env: dict[str, str]) -> str:
    """Run sbatch, and return the job's ID.

    The job gets this environment with env on top, but without OVERWRITE or the
    recipe's parameters: those come from --overwrite and --set alone. main()
    refuses a shell that sets them, and they are dropped here as well.
    """
    inherited = {
        key: value for key, value in os.environ.items()
        if key != "OVERWRITE" and key not in recipe.parameters
    }
    try:
        result = subprocess.run(
            command, cwd=ROOT, env={**inherited, **env},
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as error:
        said = (error.stderr or "").strip() or f"exit {error.returncode}"
        raise submit_lanes.SubmitError(f"sbatch refused it: {said}", []) from error
    except OSError as error:
        raise submit_lanes.SubmitError(f"could not run sbatch: {error}", []) from error
    # --parsable prints "id;cluster" on a multi-cluster setup.
    return result.stdout.strip().split(";")[0]


def _close(project: str, closing: dict[str, Any], event: str, **fields: Any) -> None:
    """The entry that says how a journaled submission ended."""
    try:
        submit_lanes.write_journal(
            project, {"at": submit_lanes._now(), "event": event, **closing, **fields}
        )
    except OSError as error:
        # The job is in the queue whether or not this is written.
        print(f"not journaled: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
