"""A lane's MRA stage, as scripts/submit_lanes.py submits it (docs/mra_in_lanes.md):
chained after the extraction as a job of scripts/mra_port_run.sbatch, held back with
--until-stage, submitted alone with --from-stage, and an MRA job counted as reading
its series."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from test_submit_lanes import SBATCH, _journal, _run, make_lane

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import submit_lanes  # noqa: E402

pytest.importorskip("pyarrow")

# Notes, besides SBATCH's line, the variables a job would get from the environment.
SBATCH_ENV = SBATCH.replace(
    'echo "${OVERWRITE:-}|$*" >> "$SBATCH_CALLS"',
    'echo "${OVERWRITE:-}|$*" >> "$SBATCH_CALLS"\n'
    'echo "NAME=${NAME:-} SOURCE=${SOURCE:-} SEEDS=${SEEDS:-} PROJECT=${PROJECT:-} '
    'SHA=${CONFIG_SHA256:-}" >> "$SBATCH_CALLS.env"',
)
MRA_STAGE = """
[[run.stage]]
name = "mra"
from = "mra"
until = "mra"
time = "1-00:00:00"
sbatch = ["--partition=gpu", "--gres=gpu:a40:4", "--cpus-per-task=8", "--mem=64G"]
"""
# What lets the synthetic particles (12 px at 9.784 A) through the MRA's setup.
SMALL = '\n[mra]\ndiameter_angstrom = 20.0\nsubset_particles = 3\n'


@pytest.fixture
def lane(tmp_path, monkeypatch):
    return make_lane(tmp_path, monkeypatch)


def _with_mra(config: Path, extra: str = "") -> Path:
    config.write_text(config.read_text() + MRA_STAGE + extra)
    return config


def _calls(config: Path) -> tuple[list[str], list[str]]:
    calls = config.parent / "sbatch_calls"
    env = config.parent / "sbatch_calls.env"
    return (calls.read_text().splitlines() if calls.exists() else [],
            env.read_text().splitlines() if env.exists() else [])


def _particles(config: Path) -> Path:
    """The synthetic registry of tests/test_mra_from_registry.py as the lane's project:
    TS_A and TS_B extracted under bin8."""
    from test_mra_from_registry import _project

    return _project(config.parent)


def test_the_mra_waits_for_the_extraction_in_a_job_of_its_own(lane, capsys, monkeypatch):
    _with_mra(lane)
    (lane.parent / "bin" / "sbatch").write_text(SBATCH_ENV)
    # Left in the shell by an MRA typed by hand: neither reaches the lane's MRA job.
    monkeypatch.setenv("SEEDS", "blob blob")
    monkeypatch.setenv("PROJECT", "outputs/elsewhere")
    code, out, err = _run(capsys, lane, "--dry-run")
    assert code == 0, err
    assert "scripts/mra_port_run.sbatch" in out and "--dependency=afterok:<gpu>" in out
    assert "after stage 'gpu', on the particles it extracts" in out
    assert "not passed to the MRA job, which takes its settings from [mra]: PROJECT, SEEDS" in out
    assert "low-pass     2 x Nyquist -> 1.1 x Nyquist" in out

    code, out, err = _run(capsys, lane)
    assert code == 0, err
    calls, env = _calls(lane)
    assert [call.split()[-1] for call in calls] == [
        "scripts/lane.sbatch", "scripts/lane.sbatch", "scripts/mra_port_run.sbatch"]
    assert "--dependency=afterok:1002" in calls[2] and "--job-name=project-mra" in calls[2]
    sha = hashlib.sha256(lane.read_text().encode()).hexdigest()
    project = str((lane.parent / "project").resolve())
    assert env[2] == f"NAME=bin8-ssn SOURCE={lane.parent / 'project'} SEEDS= PROJECT= SHA={sha}"
    assert "SEEDS=blob blob" in env[0]  # a lane stage keeps the environment it is given
    submitted = [row for row in _journal(lane) if row["event"] == "submitted"][-1]
    assert submitted["jobs"] == {"cpu": "1001", "gpu": "1002", "mra": "1003"}
    plan = [row for row in _journal(lane) if row["event"] == "submit"][-1]["plan"]
    assert plan["stages"][-1]["kind"] == "mra"
    assert plan["mra"]["run"] == f"{project}/mra/bin8-ssn"
    assert plan["mra"]["first_lowpass_nyquist"] == 2.0


def test_until_stage_holds_the_mra_back_and_from_stage_submits_it_alone(lane, capsys):
    _with_mra(lane, SMALL)
    code, out, err = _run(capsys, lane, "--until-stage", "gpu")
    assert code == 0, err
    assert all("mra_port_run" not in call for call in _calls(lane)[0])
    # Nothing extracted yet: the MRA alone is refused.
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "would refuse" in err and "the MRA reads the particles extracted under 'bin8'" in err
    assert "extract them first" in err

    _particles(lane)
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert code == 0 and "would refuse" not in err, err
    assert "particles    13 at 9.784 A/px in a 12 px box (Nyquist 19.568 A)" in out
    assert "round 1: 1 iteration(s), low-pass 3 px" in out
    code, out, err = _run(capsys, lane, "--from-stage", "mra")
    assert code == 0, err
    assert _calls(lane)[0][-1].split()[-1] == "scripts/mra_port_run.sbatch"
    assert "--dependency" not in _calls(lane)[0][-1]


def test_what_the_setup_would_refuse_is_found_in_the_dry_run(lane, capsys):
    _with_mra(lane)  # ribo-bin8's 290 A particle in the synthetic 12 px box
    _particles(lane)
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "the MRA's setup would refuse it" in err and "does not fit the 12 px box" in err
    assert "cannot give 2 disjoint subsets of 300" in err


def test_a_misplaced_or_unplaced_mra_is_refused(lane, capsys):
    text = lane.read_text()
    lane.write_text(text + SMALL)
    with pytest.raises(SystemExit, match=r"\[mra\] is set, but no \[\[run.stage\]\] runs mra"):
        _run(capsys, lane, "--dry-run")
    lane.write_text(text.replace('until = "extract-particles"', 'until = "mra"'))
    with pytest.raises(SystemExit, match="runs mra together with other steps"):
        _run(capsys, lane, "--dry-run")
    lane.write_text(text + MRA_STAGE.replace('"--gres=gpu:a40:4", ', "") + SMALL)
    code, out, err = _run(capsys, lane, "--dry-run")
    assert "runs the MRA, which searches on GPUs" in err
    lane.write_text(text + MRA_STAGE + '\n[mra]\nseeds = ["subset"]\n')
    with pytest.raises(SystemExit, match="at least two"):
        _run(capsys, lane, "--dry-run")
    lane.write_text(text + MRA_STAGE + SMALL)
    with pytest.raises(SystemExit, match="--until-stage: no stage 'review'"):
        _run(capsys, lane, "--until-stage", "review", "--dry-run")


def test_a_run_of_the_name_is_carried_on_or_refused(lane, capsys):
    _with_mra(lane, SMALL)
    project = _particles(lane)
    run = project / "mra" / "bin8-ssn"
    run.mkdir(parents=True)
    # Extracting again under a run that exists would leave it particles it never read.
    code, out, err = _run(capsys, lane, "--dry-run")
    assert "exists, and this submission extracts the particles again" in err
    # Alone, a run asked for some other way is not taken for this one...
    (run / "plan.json").write_text(json.dumps({"setup": {"variables": {}}}))
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "was not asked for by a lane's [mra]" in err
    # ... nor one of other settings.
    (run / "plan.json").write_text(json.dumps({"setup": {"request": {
        "label": "bin8", "series": ["TS_A", "TS_B"], "resolved": {"seeds": ["blob", "noise"]}}}}))
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "exists with other settings" in err and "seeds" in err


def test_an_mra_job_reads_its_series(lane, capsys, monkeypatch):
    _with_mra(lane, SMALL)
    project = _particles(lane)
    other = submit_lanes._repo_relative(project / "mra" / "other")
    submit_lanes.write_journal(str(project), {
        "event": "submit", "id": "a1", "plan": {
            "series": ["TS_A", "TS_B"], "stages": [{"name": "mra", "kind": "mra"}],
            "mra": {"run": other}}})
    submit_lanes.write_journal(str(project), {
        "event": "submitted", "id": "a1", "jobs": {"mra": "777"}, "series": ["TS_A", "TS_B"]})
    monkeypatch.setenv("SQUEUE_LINES", "777|project-mra|RUNNING")
    # The lane's steps would change the particles under it.
    code, out, err = _run(capsys, lane, "--dry-run")
    assert "job 777" in err and "which reads the particles of TS_A, TS_B" in err
    # Another MRA of them may run beside it...
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "job 777" not in err, err
    # ... but not the same run.
    lane.write_text(lane.read_text() + 'name = "other"\n')
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "job 777" in err and "which this lane would run too; its lock" in err


def test_a_class_of_an_earlier_run_is_run_again_alone(lane, capsys):
    from test_mra_class_runs import _fullest_class
    from test_mra_from_registry import QUIET, SERIES, _short

    from cryoet_pipeline.mra_runs import run_plan
    from cryoet_pipeline.mra_setup import plan_from_registry

    project = _particles(lane)
    first = project / "mra" / "bin8-ssn"
    run_plan(_short(plan_from_registry(project, "bin8", SERIES, ["subset", "noise"], subset=3,
                                       mask_radius=3, schedule="coarse_to_fine",
                                       first_lowpass=30.0, last_lowpass=22.0, **QUIET)),
             first, **QUIET)
    chosen = _fullest_class(first)
    _with_mra(lane, SMALL + f'from_run = "bin8-ssn"\nfrom_classes = [{chosen}]\n'
              'seeds = ["noise", "noise"]\nglobal_rounds = 0\n')
    code, out, err = _run(capsys, lane, "--dry-run")
    assert "[mra] from_run runs on the classes of bin8-ssn" in err
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert code == 0 and "would refuse" not in err, err
    assert f"class {chosen} of bin8-ssn, iteration 2:" in out
    assert f"mra {submit_lanes._repo_relative(project / 'mra')}/bin8-ssn-c{chosen}-nn" in out
    os.rename(first / "ite_0002", first / "ite_0002.away")
    code, out, err = _run(capsys, lane, "--from-stage", "mra", "--dry-run")
    assert "iteration 2 has not finished" in err
