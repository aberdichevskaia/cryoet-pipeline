"""scripts/submit_recipe.py submits a recipe with the checks and the journal of
scripts/submit_lanes.py, and the recipes replace outputs only when told to."""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import submit_lanes  # noqa: E402
import submit_recipe  # noqa: E402

# Stands in for sbatch: notes what the job would find in its environment and the
# arguments, fails on the call SBATCH_FAIL_AT counts to, and otherwise answers
# with a job id.
SBATCH = """#!/bin/bash
echo "${OVERWRITE:-}|${SERIES:-}|${KEEP_STACK:-}|${LABEL:-}|$*" >> "$SBATCH_CALLS"
n=$(wc -l < "$SBATCH_CALLS" | tr -d ' ')
if [[ "$n" == "${SBATCH_FAIL_AT:-0}" ]]; then
    echo "sbatch: error: Batch job submission failed: Invalid qos specification" >&2
    exit 1
fi
echo $((1000 + n))
"""
# Stands in for squeue: fails with SQUEUE_EXIT, and otherwise lists SQUEUE_LINES.
SQUEUE = """#!/bin/bash
if [[ -n "${SQUEUE_EXIT:-}" ]]; then
    echo "squeue: error: Unable to contact slurm controller" >&2
    exit "$SQUEUE_EXIT"
fi
if [[ -n "${SQUEUE_LINES:-}" ]]; then printf '%s\\n' "$SQUEUE_LINES"; fi
"""
REASON = ["--reason", "Anna asked: the CTF window test on TS_A"]
CTF_A = ["ctf-window", "--series", "TS_A"]


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """ribo-bin8 moved onto two empty stacks and a project in tmp_path whose series,
    TS_A and TS_B, have manifests; run by a person, with sbatch and squeue
    replaced by SBATCH and SQUEUE."""

    data = tmp_path / "data"
    data.mkdir()
    project = tmp_path / "project"
    (project / "manifests").mkdir(parents=True)
    for series in ("TS_A", "TS_B"):
        (data / f"{series}.mrc").write_bytes(b"")
        (data / f"{series}.mrc.mdoc").write_text("")
        (project / "manifests" / f"{series}.json").write_text("{}")
    text = (ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text()
    for pattern, value in (
        (r'^project = ".*"$', f'project = "{project}"'),
        (r"^series = \[.*\]$", 'series = ["TS_A", "TS_B"]'),
        (r'^mdoc = ".*"$', f'mdoc = "{data}/{{series}}.mrc.mdoc"'),
        (r'^stack = ".*"$', f'stack = "{data}/{{series}}.mrc"'),
    ):
        text, count = re.subn(pattern, lambda _, v=value: v, text, count=1, flags=re.MULTILINE)
        assert count == 1, f"ribo-bin8.toml has no line {pattern}"
    config = tmp_path / "lane.toml"
    config.write_text(text)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("sbatch", SBATCH), ("squeue", SQUEUE)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SBATCH_CALLS", str(tmp_path / "sbatch_calls"))
    monkeypatch.setenv("PYTOMO_MP_DIR", "/opt/pytom")
    monkeypatch.setenv("IMOD_DIR", "/opt/imod")
    # A person, whatever runs the tests, with no recipe's parameter in the shell.
    parameters = {key for recipe in submit_recipe.RECIPES.values() for key in recipe.parameters}
    for name in (
        *submit_lanes.CLIENT_VARIABLES, *parameters, "OVERWRITE", "SBATCH_FAIL_AT",
        "SQUEUE_EXIT", "SQUEUE_LINES",
    ):
        monkeypatch.delenv(name, raising=False)
    return config


def _run(capsys, config: Path, recipe: str, *args: str) -> tuple[int, str, str]:
    code = submit_recipe.main([recipe, str(config), *args])
    out, err = capsys.readouterr()
    return code, out, err


def _calls(config: Path) -> list[dict]:
    """What each sbatch call was given."""
    path = config.parent / "sbatch_calls"
    if not path.exists():
        return []
    calls = []
    for line in path.read_text().splitlines():
        overwrite, series, keep_stack, label, args = line.split("|", 4)
        calls.append({
            "overwrite": overwrite, "series": series, "keep_stack": keep_stack,
            "label": label, "args": args.split(),
        })
    return calls


def _journal(config: Path) -> list[dict]:
    path = config.parent / "project" / submit_lanes.JOURNAL
    return [json.loads(line) for line in path.read_text().splitlines()]


def _register(config: Path, *artifacts: tuple[str, str, list[str]]) -> None:
    """A registry holding these (kind, label, series) entries."""
    entries = [
        {"id": f"{kind}:{label}:{n}", "kind": kind, "path": f"{kind}/{n}", "variant": label,
         "tilt_series_ids": series}
        for n, (kind, label, series) in enumerate(artifacts)
    ]
    path = config.parent / "project" / "artifacts.json"
    path.write_text(json.dumps({"schema_version": 1, "artifacts": entries}))


def test_a_recipe_runs_where_the_lane_runs_its_step_and_on_its_own_resources(lane, capsys):
    assert _run(capsys, lane, *CTF_A)[0] == 0
    assert _run(capsys, lane, "repick", "--set", "LABEL=bin8-again", "--time", "20:00:00")[0] == 0
    ctf, repick = _calls(lane)
    # The partition and QOS of the lane's stage that runs estimate-ctf; the CPUs,
    # memory and time of the recipe's own #SBATCH lines.
    assert ctf["args"] == [
        "--parsable", "--export=ALL", "--partition=cpu_p", "--qos=cpu_normal",
        "--time=03:00:00", "--job-name=project-ctf-window", "scripts/ctf_window_test.sbatch",
    ]
    assert repick["args"][2:6] == [
        "--partition=gpu_p", "--qos=gpu_normal", "--gres=gpu:a100_3g.20gb:1", "--time=20:00:00",
    ]
    # A recipe that takes several series gets the lane's.
    assert (ctf["series"], repick["series"], repick["label"]) == ("TS_A", "TS_A TS_B", "bin8-again")


def test_overwrite_reaches_the_job_only_from_the_flag(lane, capsys):
    assert _run(capsys, lane, *CTF_A)[0] == 0
    assert _run(capsys, lane, "ctf-window", "--series", "TS_B", "--overwrite")[0] == 0
    assert [call["overwrite"] for call in _calls(lane)] == ["", "1"]


@pytest.mark.parametrize("dry_run", [[], ["--dry-run"]], ids=["submit", "dry-run"])
@pytest.mark.parametrize(
    ("name", "words"),
    [("OVERWRITE", "give --overwrite"), ("KEEP_STACK", "give each as --set NAME=VALUE")],
)
def test_overwrite_or_a_parameter_left_in_the_shell_is_refused(
    lane, monkeypatch, dry_run, name, words
):
    # --export=ALL would hand either to the job, and the dry run would not show it.
    monkeypatch.setenv(name, "1")
    with pytest.raises(SystemExit, match=words):
        submit_recipe.main(["ctf-window", str(lane), "--series", "TS_A", *dry_run])
    assert not _calls(lane)


def test_submit_does_not_pass_on_an_inherited_overwrite_or_parameter_either(lane, monkeypatch):
    # For a caller other than main(), which refuses both before this point.
    for name, value in (("OVERWRITE", "1"), ("KEEP_STACK", "1"), ("LABEL", "not-its-own")):
        monkeypatch.setenv(name, value)
    command = ["sbatch", "--parsable", "scripts/ctf_window_test.sbatch"]
    assert submit_recipe.submit(submit_recipe.RECIPES["ctf-window"], command, {}) == "1001"
    (call,) = _calls(lane)
    # LABEL is another recipe's parameter, so it passes, as anything else in the shell does.
    assert (call["overwrite"], call["keep_stack"], call["label"]) == ("", "", "not-its-own")


@pytest.mark.parametrize(
    ("args", "words"),
    [
        ([*CTF_A, "--set", "LABEL=x"], "ctf-window reads no LABEL"),
        ([*CTF_A, "--set", "KEEP_STACK"], "give NAME=VALUE"),
        ([*CTF_A, "--set", "KEEP_STACK=1", "--set", "KEEP_STACK=0"], "given twice"),
        (["repick"], "repick needs --set LABEL=..."),
        (["ctf-window"], "runs on one series"),
        ([*CTF_A, "--series", "TS_B"], "runs on one series"),
    ],
)
def test_a_recipe_takes_its_own_parameters_and_series_only(lane, args, words):
    with pytest.raises(SystemExit, match=re.escape(words)):
        submit_recipe.main([args[0], str(lane), *args[1:], "--dry-run"])


def test_outputs_already_under_its_labels_are_replaced_only_with_overwrite(lane, capsys):
    _register(lane, ("ctf_model", "ctfw5", ["TS_A"]), ("ctf_model", "bin8", ["TS_A"]))
    code, _, err = _run(capsys, lane, *CTF_A, "--dry-run")
    assert code == 0 and "TS_A already has ctf_model under ctfw5" in err
    assert "under bin8" not in err  # the lane's own model is not the recipe's
    assert _run(capsys, lane, *CTF_A)[0] == 1
    assert not _calls(lane)

    code, out, err = _run(capsys, lane, *CTF_A, "--overwrite")
    assert code == 0 and "refuse" not in err
    assert "ctf_model under ctfw5: TS_A" in out and "unlabelled CTF stack" in out
    # KEEP_STACK=1 keeps the stack, and TS_B has nothing under the labels.
    code, out, _ = _run(
        capsys, lane, "ctf-window", "--series", "TS_B", "--overwrite", "--set", "KEEP_STACK=1",
        "--dry-run",
    )
    assert "nothing is registered under its labels yet" in out
    assert "unlabelled CTF stack" not in out


def test_a_copy_the_recipe_reuses_does_not_stop_it(lane, capsys):
    _register(lane, ("tomogram", "bin8-null", ["TS_A"]))
    assert "refuse" not in _run(capsys, lane, "null-decoys", "--dry-run")[2]
    _register(lane, ("tomogram", "bin8-null", ["TS_A"]), ("picks", "bin8-null", ["TS_A"]))
    _, _, err = _run(capsys, lane, "null-decoys", "--dry-run")
    assert "TS_A already has picks under bin8-null" in err


def test_what_would_stop_the_job_at_once_is_refused_in_its_dry_run(
    lane, capsys, monkeypatch, tmp_path
):
    export = tmp_path / "export"
    export.mkdir()
    (export / "old.tbl").write_text("")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"TS_A": 0.2, "TS_B": None}))
    average = ["average", "--set", f"THRESHOLDS={thresholds}", "--set", "MASK=wide"]
    for args, words in (
        (["ctf-window", "--series", "TS_C"], "has no manifest for TS_C"),
        (["repick", "--set", "LABEL=bin8"], "LABEL must differ from the lane's label 'bin8'"),
        (["repick", "--set", "LABEL=x", "--set", f"EXPORT={export}"], "already holds files"),
        (["null-decoys", "--set", "VARIANT=bin8"], "VARIANT must differ"),
        (average, "has no numeric threshold for TS_B"),
        (average, "MASK 'wide' is neither a diameter in A nor a file"),
        # The ribo lanes have run without pytom's corrections since 2026-10-15.
        (
            ["pick-ablation", "--series", "TS_A"],
            "no longer renders --pytom-random-phase-correction",
        ),
    ):
        code, _, err = _run(capsys, lane, *args, "--dry-run")
        assert code == 0 and f"would refuse: {lane}: " in err and words in err, (args, err)

    monkeypatch.delenv("IMOD_DIR")
    code, _, err = _run(capsys, lane, *CTF_A)
    assert code == 1 and "$IMOD_DIR is not set (used by the recipe's job)" in err
    assert not _calls(lane)


def test_the_assistant_submits_the_recipe_dry_run_it_showed_and_only_once(
    lane, capsys, monkeypatch
):
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    with pytest.raises(SystemExit, match="needs --reason"):
        submit_recipe.main(["ctf-window", str(lane), "--series", "TS_A"])
    code, _, err = _run(capsys, lane, *CTF_A, *REASON)
    assert code == 1 and "no dry run of it is on record" in err

    code, out, _ = _run(capsys, lane, *CTF_A, "--dry-run")
    assert code == 0 and "within 30 minutes" in out
    assert _run(capsys, lane, *CTF_A, *REASON)[0] == 0
    code, _, err = _run(capsys, lane, *CTF_A, *REASON)
    assert code == 1 and "it was submitted at" in err
    assert len(_calls(lane)) == 1


@pytest.mark.parametrize(
    ("change", "words"),
    [
        (["--set", "KEEP_STACK=1"], "the recipe's parameters differ"),
        (["--time", "05:00:00"], "the sbatch options differ"),
        (["--overwrite"], "only one of them has --overwrite"),
    ],
)
def test_a_recipe_submission_that_differs_from_its_dry_run_is_refused(
    lane, capsys, monkeypatch, change, words
):
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    assert _run(capsys, lane, *CTF_A, "--dry-run")[0] == 0
    code, _, err = _run(capsys, lane, *CTF_A, *change, *REASON)
    assert code == 1 and words in err and not _calls(lane)


def test_a_changed_script_or_input_file_is_not_what_was_shown(lane, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"TS_A": 0.2, "TS_B": 0.3}))
    average = ["average", "--set", f"THRESHOLDS={thresholds}", "--set", "MASK=380"]
    assert _run(capsys, lane, *average, "--dry-run")[0] == 0
    thresholds.write_text(json.dumps({"TS_A": 0.1, "TS_B": 0.3}))
    code, _, err = _run(capsys, lane, *average, *REASON)
    assert code == 1 and "a file the recipe reads has changed" in err

    script = tmp_path / "stand_in.sbatch"
    script.write_text("#!/bin/bash\n#SBATCH --time=00:10:00\necho one\n")
    monkeypatch.setitem(submit_recipe.RECIPES, "stand-in", submit_recipe.Recipe(
        script=str(script), what="a stand-in", one_series=True, steps=("estimate-ctf",),
        where="estimate-ctf",
    ))
    assert _run(capsys, lane, "stand-in", "--series", "TS_A", "--dry-run")[0] == 0
    script.write_text("#!/bin/bash\n#SBATCH --time=00:10:00\necho two\n")
    code, _, err = _run(capsys, lane, "stand-in", "--series", "TS_A", *REASON)
    assert code == 1 and "the recipe's script has changed" in err
    assert not _calls(lane)


def test_a_dry_run_counts_for_its_lane_or_its_recipe_on_its_series(lane, capsys, monkeypatch):
    # Dry runs of a lane and of recipes on one config interleave; each submission
    # finds its own.
    monkeypatch.setenv("CRYOET_ASSISTANT", "test-client 1.0")
    ctf_b = ["ctf-window", "--series", "TS_B"]
    assert _run(capsys, lane, *CTF_A, "--dry-run")[0] == 0
    assert submit_lanes.main([str(lane), "--dry-run"]) == 0
    assert _run(capsys, lane, *ctf_b, "--dry-run")[0] == 0
    capsys.readouterr()
    assert submit_lanes.main([str(lane), *REASON]) == 0
    assert _run(capsys, lane, *CTF_A, *REASON)[0] == 0
    assert _run(capsys, lane, *ctf_b, *REASON)[0] == 0
    # The lane's two stages, then the two recipes.
    assert [call["series"] for call in _calls(lane)] == ["TS_A TS_B"] * 2 + ["TS_A", "TS_B"]


def test_a_recipe_is_journaled_as_a_lane_is(lane, capsys, monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("AI_AGENT", "claude-code_test_agent")
    repick = ["repick", "--set", "LABEL=bin8-again"]
    assert _run(capsys, lane, *repick, "--dry-run")[0] == 0
    assert _run(capsys, lane, *repick, *REASON)[0] == 0
    dry, submit, submitted = _journal(lane)
    assert [row["event"] for row in (dry, submit, submitted)] == ["dry-run", "submit", "submitted"]
    assert dry["recipe"] == submit["recipe"] == submitted["recipe"] == "repick"
    plan = submit["plan"]
    assert plan == dry["plan"] and plan["series"] == ["TS_A", "TS_B"]
    assert plan["parameters"] == {"LABEL": "bin8-again"}
    assert plan["env"] == {"PYTOMO_MP_DIR": "/opt/pytom"}
    assert submit["client"] == {"AI_AGENT": "claude-code_test_agent", "CLAUDECODE": "1"}
    assert submit["reason"] == REASON[1] and submit["config_text"] == lane.read_text()
    assert submitted["id"] == submit["id"]
    assert submitted["jobs"] == {"repick": "1001"} and submitted["series"] == ["TS_A", "TS_B"]


def test_a_job_sbatch_refuses_is_journaled_as_failed(lane, capsys, monkeypatch):
    monkeypatch.setenv("SBATCH_FAIL_AT", "1")
    code, _, err = _run(capsys, lane, *CTF_A)
    assert code == 1 and "Invalid qos specification" in err
    submit, failed = _journal(lane)
    assert failed["event"] == "submit-failed" and failed["id"] == submit["id"]
    assert failed["jobs"] == {}


def test_a_queued_recipe_job_holds_its_series_against_lanes_and_recipes(
    lane, capsys, monkeypatch
):
    # Waiting, it has no log yet: the journal is what tells its series.
    assert _run(capsys, lane, *CTF_A)[0] == 0
    monkeypatch.setenv("SQUEUE_LINES", "1001|project-ctf-window|PENDING")
    _, _, err = _run(capsys, lane, *CTF_A, "--dry-run")
    assert "job 1001 (project-ctf-window, pending) works on TS_A, which this recipe" in err
    assert "refuse" not in _run(capsys, lane, "ctf-window", "--series", "TS_B", "--dry-run")[2]
    assert submit_lanes.main([str(lane), "--dry-run"]) == 0
    _, err = capsys.readouterr()
    assert "job 1001 (project-ctf-window, pending) works on TS_A, which this lane" in err


def test_the_tasks_of_an_array_job_are_known_by_its_journaled_id(lane, capsys, monkeypatch):
    # squeue names an array's tasks <id>_<task>, where sbatch printed <id>.
    assert _run(capsys, lane, *CTF_A)[0] == 0
    monkeypatch.setenv(
        "SQUEUE_LINES",
        "1001_0|project-ctf-window|RUNNING\n1001_[1-4%1]|project-ctf-window|PENDING",
    )
    _, _, err = _run(capsys, lane, *CTF_A, "--dry-run")
    assert "job 1001_0 (project-ctf-window, running) works on TS_A" in err
    assert "job 1001_[1-4%1] (project-ctf-window, pending) works on TS_A" in err


@pytest.mark.parametrize("name", sorted(submit_recipe.RECIPES))
def test_every_recipe_names_what_its_script_reads(name):
    recipe = submit_recipe.RECIPES[name]
    text = (ROOT / "scripts" / recipe.script).read_text()
    for key in ("PROJECT", "CONFIG", "SERIES", "OVERWRITE", *recipe.needs, *recipe.parameters):
        assert re.search(rf"\$\{{?{key}\b", text), f"{recipe.script} never reads ${key}"
    assert submit_recipe.script_time(recipe), f"{recipe.script} sets no time limit"
    chain = [
        step for step, _ in submit_recipe.run_series.build_steps(
            tomllib.loads((ROOT / "configs/lanes/ryr1-bin8.toml").read_text()),
            Path("outputs/ryr1"), "TS_X",
        )
    ]
    assert set(recipe.steps) | {recipe.where} <= set(chain)


# The lane's commands as scripts/run_series.py prints them, for the stand-in repo.
RUN_SERIES = """import sys
step = sys.argv[sys.argv.index("--from") + 1]
print("# " + step)
print({
    "build-ctf-stack": "cryoet build-ctf-stack --series TS_A",
    "estimate-ctf": "cryoet estimate-ctf --series TS_A"
                    " --ctffind-defocus-search-window-angstrom 1000",
    "prepare-template": "cryoet prepare-template --series TS_A --template-box-size-px 60"
                        " --template-mask-sigma-px 1 --template-mask-radius-px 18",
    "pick-particles": "cryoet pick-particles --series TS_A --pytom-random-phase-correction"
                      " --pytom-low-pass 35 --pytom-cut-off 0",
}[step])
"""


def _run_script(tmp_path: Path, script: str, env: dict[str, str]) -> list[str]:
    """Run a recipe's batch script in a stand-in repo, where `cryoet` notes its
    arguments and does nothing; return what it was asked to run."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "scripts" / script, repo / "scripts" / script)
    (repo / "scripts" / "run_series.py").write_text(RUN_SERIES)
    (repo / "scripts" / "activate_env.sh").write_text("CONDA_PREFIX=/env\nENV_MANAGER=stub\n")
    for name in ("ctf_fit_table.py", "score_band_readout.py"):
        (repo / "scripts" / name).write_text("")
    imod = tmp_path / "imod"
    imod.mkdir(exist_ok=True)
    (imod / "IMOD-linux.sh").write_text("")
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "cryoet").write_text('#!/bin/bash\necho "$*" >> "$CRYOET_CALLS"\n')
    (tools / "cryoet").chmod(0o755)
    calls = tmp_path / "cryoet_calls"
    calls.unlink(missing_ok=True)
    result = subprocess.run(
        ["bash", str(repo / "scripts" / script)], cwd=repo, capture_output=True, text=True,
        env={
            "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path),
            "SLURM_SUBMIT_DIR": str(repo), "SLURM_JOB_NAME": "test", "SLURM_JOB_ID": "1",
            "IMOD_DIR": str(imod), "CRYOET_CALLS": str(calls), "SERIES": "TS_A", **env,
        },
    )
    assert result.returncode == 0, result.stderr
    return calls.read_text().splitlines() if calls.exists() else []


@pytest.mark.parametrize(
    ("env", "stack", "fits"),
    [({}, False, False), ({"OVERWRITE": "1"}, True, True),
     ({"OVERWRITE": "1", "KEEP_STACK": "1"}, False, True)],
    ids=["neither", "overwrite", "overwrite-keep-stack"],
)
def test_the_ctf_window_test_replaces_the_stack_and_its_fits_only_when_told(
    tmp_path, env, stack, fits
):
    calls = _run_script(
        tmp_path, "ctf_window_test.sbatch", {"PROJECT": "proj", "CONFIG": "lane.toml", **env}
    )
    assert [call for call in calls if call.startswith("build-ctf-stack")] == (
        ["build-ctf-stack --series TS_A --overwrite"] if stack else []
    )
    labels = [call.split(" --variant ")[1] for call in calls if call.startswith("estimate-ctf")]
    assert [label.endswith(" --overwrite") for label in labels] == [fits] * 4
    assert [label.split()[0] for label in labels] == list(submit_recipe.CTF_WINDOWS)


@pytest.mark.parametrize("overwrite", [False, True])
def test_the_average_replaces_an_extraction_or_an_average_only_when_told(tmp_path, overwrite):
    config = tmp_path / "lane.toml"
    config.write_text(
        '[lane]\nvariant = "bin8"\n[run]\nproject = "proj"\nseries = ["TS_A"]\n'
        "[extract]\nbox_size_px = 60\n"
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"TS_A": 0.2}))
    calls = _run_script(tmp_path, "average_lane.sbatch", {
        "CONFIG": str(config), "THRESHOLDS": str(thresholds), "MASK": "380",
        **({"OVERWRITE": "1"} if overwrite else {}),
    })
    assert [call.split()[0] for call in calls] == ["extract-particles", "average-particles"]
    assert [call.endswith(" --overwrite") for call in calls] == [overwrite, overwrite]


@pytest.mark.parametrize("overwrite", [False, True])
def test_the_pick_ablation_replaces_its_picks_only_when_told(tmp_path, overwrite):
    templates = tmp_path / "repo" / "proj" / "templates" / "TS_A"
    templates.mkdir(parents=True)
    for part in ("template", "mask"):
        (templates / f"TS_A_pytom_bin8-abl-tpl39_{part}.mrc").write_bytes(b"")
    (tmp_path / "repo" / "proj" / "tomograms" / "TS_A" / "TS_A_bin8.zarr").mkdir(parents=True)
    calls = _run_script(tmp_path, "pick_ablation.sbatch", {
        "PROJECT": "proj", "CONFIG": "lane.toml", "PYTOMO_MP_DIR": "/opt/pytom",
        "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "1",
        **({"OVERWRITE": "1"} if overwrite else {}),
    })
    (pick,) = calls  # task 0's template was there already
    assert pick.startswith("pick-particles") and " --output-variant bin8-abl-lane " in pick
    assert pick.endswith(" --overwrite") == overwrite
