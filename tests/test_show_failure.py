"""Failed-step excerpts: the step's lines from its Slurm log and from its programs'
logs, bounded, for the failures the lane still has."""

import copy
import json
import os
import re
import sys
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import show_failure  # noqa: E402
import show_project  # noqa: E402

LANE = tomllib.loads((ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text())
CRASHED = datetime(2026, 10, 27, 10, 0, tzinfo=UTC)


def _config() -> dict:
    config = copy.deepcopy(LANE)
    config["run"]["project"] = "outputs/p"
    config["run"]["series"] = ["A", "B"]
    return config


def _row(series: str, step: str, exit_code: int, finished: datetime) -> dict:
    return {"series": series, "step": step, "lane": "ribo-bin8", "variant": "bin8",
            "seconds": 60.0, "disk_delta_bytes": 0, "exit_code": exit_code,
            "finished_at": finished.isoformat()}


def _write(path: Path, text: str, when: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (when.timestamp(), when.timestamp()))


def _project(root: Path) -> None:
    """B crashed while seeding (job 123). A crashed while tracking (job 120) and
    was rerun successfully since. Job 124 stopped in its preflight, and job 125
    stopped in the middle of picking A."""
    project = root / "outputs" / "p"
    project.mkdir(parents=True)
    rows = [
        _row("A", "track-fiducials", 1, datetime(2026, 10, 26, 9, 0, tzinfo=UTC)),
        _row("A", "track-fiducials", 0, datetime(2026, 10, 26, 12, 0, tzinfo=UTC)),
        _row("B", "generate-fiducial-seed", 139, CRASHED),
    ]
    (project / "step_timings.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    # A command log whose second command crashed, written while the step ran; and
    # an older log in the same folder, from another run.
    _write(project / "fiducials" / "B" / "B_autofidseed.log",
           "$ newstack -bin 2 B.st B_bin2.st\n\n[stdout]\ndone\n\n[stderr]\n\n[exit_code]\n0\n"
           "\n\n$ autofidseed -StandardInput\n\n[stdout]\nRunning beadtrack\n\n[stderr]\n"
           "beadtrack: Segmentation fault (core dumped)\n\n[exit_code]\n-11\n",
           CRASHED - timedelta(seconds=10))
    _write(project / "fiducials" / "B" / "B_older.log", "ERROR: from an earlier run\n",
           CRASHED - timedelta(days=3))
    header = "project     outputs/p   config configs/lanes/ribo-bin8.toml\n"
    (root / "slurm-p-cpu-120.log").write_text(
        "job         p-cpu (120) on node1\nstarted     2026-10-26T08:00:00+00:00\n" + header
        + "series      A\npreflight   ok\n\n# ---- A (ribo-bin8) ----\n\n# track-fiducials\n"
        "cryoet track-fiducials --manifest outputs/p/manifests/A.json\n"
        "ERROR: too few points\n# track-fiducials: 1.0 min, +0.00 GB\n\n"
        "# FAILED at track-fiducials (exit 1) for A.\n"
    )
    (root / "slurm-p-cpu-123.log").write_text(
        "job         p-cpu (123) on node1\nstarted     2026-10-27T09:00:00+00:00\n" + header
        + "series      B\npreflight   ok\n\n# ---- B (ribo-bin8) ----\n\n# align-tilt-series\n"
        "cryoet align-tilt-series --manifest outputs/p/manifests/B.json\n"
        "# align-tilt-series: 2.0 min, +0.10 GB\n\n# generate-fiducial-seed\n"
        "cryoet generate-fiducial-seed --manifest outputs/p/manifests/B.json\n"
        "seeding at binning 2\nsubprocess failed: autofidseed returned -11\n"
        "# generate-fiducial-seed: 1.0 min, +0.00 GB\n\n"
        "# FAILED at generate-fiducial-seed (exit 139) for B.\n"
        "# Fix it, then resume with:\n#   --series B --from generate-fiducial-seed\n"
    )
    (root / "slurm-p-gpu-124.log").write_text(
        "job         p-gpu (124) on node2\nstarted     2026-10-27T11:00:00+00:00\n" + header
        + "series      A\nNVIDIA A100 80GB PCIe, 8.0, 81920 MiB\n"
        "preflight: MotionCor3 has nothing this GPU (compute 8.0) can run.\n"
    )
    (root / "slurm-p-gpu-125.log").write_text(
        "job         p-gpu (125) on node2\nstarted     2026-10-27T12:00:00+00:00\n" + header
        + "series      A\npreflight   ok\n\n# ---- A (ribo-bin8) ----\n\n# pick-particles\n"
        "cryoet pick-particles --manifest outputs/p/manifests/A.json\nmatching rotations 1/3\n"
    )


def _found(root: Path, **options: object) -> dict[tuple[str, str], dict]:
    result = show_failure.failures(_config(), root, slurm=False, **options)
    return {(entry["series"], entry["step"]): entry for entry in result["failures"]}


def test_only_the_failures_the_lane_still_has_are_reported(tmp_path):
    _project(tmp_path)
    # A's tracking crash was fixed by a later run.
    assert set(_found(tmp_path)) == {
        ("A", "preflight"), ("A", "pick-particles"), ("B", "generate-fiducial-seed")
    }


def test_a_crash_shows_its_command_its_error_lines_and_its_programs_log(tmp_path):
    _project(tmp_path)
    seed = _found(tmp_path)["B", "generate-fiducial-seed"]
    assert (seed["kind"], seed["exit_code"], seed["job"]["id"]) == ("crashed", 139, "123")

    texts = [line["text"] for line in seed["slurm_lines"]]
    assert texts[0] == "# generate-fiducial-seed"
    assert texts[1].startswith("cryoet generate-fiducial-seed")
    assert "subprocess failed: autofidseed returned -11" in texts
    assert texts[-1] == "# FAILED at generate-fiducial-seed (exit 139) for B."
    assert not any("align-tilt-series" in text for text in texts)

    # Only the log written while the step ran, and in it only the command that failed.
    [log] = seed["program_logs"]
    assert log["path"] == "outputs/p/fiducials/B/B_autofidseed.log"
    shown = [line["text"] for line in log["lines"]]
    assert "$ autofidseed -StandardInput" in shown
    assert "beadtrack: Segmentation fault (core dumped)" in shown
    assert shown[-2:] == ["[exit_code]", "-11"]
    assert "$ newstack -bin 2 B.st B_bin2.st" not in shown


def test_a_preflight_stop_and_a_job_stopped_mid_step_are_failures(tmp_path):
    _project(tmp_path)
    found = _found(tmp_path)
    preflight = found["A", "preflight"]
    assert preflight["problem"] == "MotionCor3 has nothing this GPU (compute 8.0) can run."
    assert any("NVIDIA A100" in line["text"] for line in preflight["slurm_lines"])

    stopped = found["A", "pick-particles"]
    assert (stopped["kind"], stopped["exit_code"]) == ("interrupted", None)
    assert stopped["slurm_lines"][-1]["text"] == "matching rotations 1/3"


def test_the_text_gives_line_numbers_and_leaves_other_runs_out(tmp_path):
    _project(tmp_path)
    result = show_failure.failures(_config(), tmp_path, step="generate-fiducial-seed",
                                   slurm=False)
    text = "\n".join(show_failure.render(result))
    assert "B  generate-fiducial-seed  crashed, exit 139" in text
    assert "slurm-p-cpu-123.log (job p-cpu 123)" in text
    assert re.search(r"^\s+\d+  beadtrack: Segmentation fault", text, re.MULTILINE)
    assert "B_older.log" not in text


def test_error_lines_are_capped_at_the_first_and_the_last():
    lines = [f"ERROR {index}" for index in range(30)]
    picked = show_failure._errors(lines, range(len(lines)))
    half = show_failure.MAX_ERRORS // 2
    assert picked == [*range(half), *range(30 - half, 30)]


def test_a_failure_rerun_since_is_listed_apart_with_the_jobs_it_failed_in(tmp_path):
    _project(tmp_path)
    result = show_failure.failures(_config(), tmp_path, slurm=False)
    [rerun] = result["rerun_since"]
    assert (rerun["series"], rerun["step"]) == ("A", "track-fiducials")
    assert [job["id"] for job in rerun["jobs"]] == ["120"]
    assert rerun["passed_at"] == "2026-10-26T12:00:00+00:00"
    text = "\n".join(show_failure.render(result))
    assert "A  track-fiducials  failed in p-cpu 120; passed 2026-10-26T12:00:00+00:00" in text


def _submitted_elsewhere(
    root: Path, runs: Path, monkeypatch, logs: dict[tuple[str, str], str]
) -> None:
    """`runs`, a folder outside the repository that jobs were submitted from by
    hand, reaching the project through a link as cryoet-runs does in Munich,
    gets the logs of these jobs (by ID and name); sacct names it as their WorkDir."""
    (runs / "outputs").mkdir()
    (runs / "outputs" / "p").symlink_to(root / "outputs" / "p")
    for (job_id, name), text in logs.items():
        (runs / f"slurm-{name}-{job_id}.log").write_text(text)
    rows = [[job_id, name, str(runs)] for job_id, name in logs]
    monkeypatch.setattr(show_project, "_slurm",
                        lambda command: rows if "JobID,JobName,WorkDir" in command else [])


HEADER = "project     outputs/p   config p.toml\n"
# A job that stopped in its preflight, because a path in its environment was wrong.
STOPPED = (
    "job         p-cpu-B (118) on node1\nstarted     2026-10-27T07:00:00+00:00\n" + HEADER
    + "series      B\nsteps       align-tilt-series generate-fiducial-seed\n"
    "preflight: MEMBRAIN_CKPT does not exist: fast/membrain.ckpt\n"
)


def test_a_job_submitted_from_another_folder_is_read_where_it_ran(
    tmp_path, tmp_path_factory, monkeypatch
):
    _project(tmp_path)
    with (tmp_path / "outputs" / "p" / "step_timings.jsonl").open("a") as handle:
        row = _row("A", "pick-particles", 1, datetime(2026, 10, 27, 13, 5, tzinfo=UTC))
        handle.write(json.dumps(row) + "\n")
    runs = tmp_path_factory.mktemp("runs")
    _submitted_elsewhere(tmp_path, runs, monkeypatch, {("131", "p-gpu-A"): (
        "job         p-gpu-A (131) on node2\nstarted     2026-10-27T13:00:00+00:00\n" + HEADER
        + "series      A\nsteps       pick-particles\npreflight   ok\n\n"
        "# ---- A (ribo-bin8) ----\n\n# pick-particles\n"
        "cryoet pick-particles --manifest outputs/p/manifests/A.json\n"
        "cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 1,179,648,000 bytes\n"
        "# pick-particles: 1.8 min, +0.00 GB\n\n# FAILED at pick-particles (exit 1) for A.\n"
    )})
    result = show_failure.failures(_config(), tmp_path)
    [pick] = [entry for entry in result["failures"]
              if (entry["series"], entry["step"]) == ("A", "pick-particles")]
    assert (pick["kind"], pick["exit_code"], pick["job"]["id"]) == ("crashed", 1, "131")
    assert pick["job"]["log"] == str(runs / "slurm-p-gpu-A-131.log")
    assert any("OutOfMemoryError" in line["text"] for line in pick["slurm_lines"])


def test_a_preflight_stop_is_history_once_its_steps_ran_for_the_series(
    tmp_path, tmp_path_factory, monkeypatch
):
    _project(tmp_path)
    runs = tmp_path_factory.mktemp("runs")
    _submitted_elsewhere(tmp_path, runs, monkeypatch, {("118", "p-cpu-B"): STOPPED})
    result = show_failure.failures(_config(), tmp_path)
    found = {(entry["series"], entry["step"]) for entry in result["failures"]}
    # B's seeding ran at 10:00, after the stop, and crashed: that crash is B's
    # failure. Job 124 names no steps, so its stop stays a failure.
    assert ("B", "preflight") not in found
    assert {("B", "generate-fiducial-seed"), ("A", "preflight")} <= found
    [stop] = [entry for entry in result["rerun_since"] if entry["step"] == "preflight"]
    assert [job["id"] for job in stop["jobs"]] == ["118"]
    assert stop["passed_at"] == CRASHED.isoformat()
    text = "\n".join(show_failure.render(result))
    assert "B  preflight  failed in p-cpu-B 118; passed 2026-10-27T10:00:00+00:00" in text


def _other_lane(root: Path) -> None:
    """Jobs of ribo-bin4, another lane of the project, which labels its steps
    bin4 and names its jobs as ribo-bin8 does. On C its pick crashed, and a later
    job stopped before reconstructing and picking; on D its alignment crashed, and
    a later job stopped before aligning; on E its pick is still running."""
    (root / "configs" / "lanes").mkdir(parents=True)
    (root / "configs" / "lanes" / "ribo-bin4.toml").write_text(
        '[lane]\nname = "ribo-bin4"\nvariant = "bin4"\n'
    )
    with (root / "outputs" / "p" / "step_timings.jsonl").open("a") as handle:
        for series, step, exit_code, hour in (("D", "align-tilt-series", 2, 12),
                                              ("C", "pick-particles", 1, 13)):
            row = _row(series, step, exit_code, datetime(2026, 10, 27, hour, 1, tzinfo=UTC))
            handle.write(json.dumps({**row, "lane": "ribo-bin4", "variant": "bin4"}) + "\n")

    def log(job_id: str, hour: int, series: str, steps: str, rest: str) -> None:
        (root / f"slurm-p-gpu-{job_id}.log").write_text(
            f"job         p-gpu ({job_id}) on node3\nstarted     2026-10-27T{hour}:00:00+00:00\n"
            "project     outputs/p   config configs/lanes/ribo-bin4.toml\n"
            f"series      {series}\nsteps       {steps}\n" + rest
        )

    def ran(series: str, step: str, *lines: str) -> str:
        return (f"preflight   ok\n\n# ---- {series} (ribo-bin4) ----\n\n# {step}\n"
                f"cryoet {step} --manifest outputs/p/manifests/{series}.json\n" + "".join(lines))

    log("201", 12, "D", "align-tilt-series", ran(
        "D", "align-tilt-series", "ERROR: no tilt angles\n",
        "# align-tilt-series: 1.0 min, +0.00 GB\n\n# FAILED at align-tilt-series (exit 2) for D.\n",
    ))
    log("200", 13, "C", "pick-particles", ran(
        "C", "pick-particles", "out of memory\n",
        "# pick-particles: 1.0 min, +0.00 GB\n\n# FAILED at pick-particles (exit 1) for C.\n",
    ))
    log("205", 14, "C", "reconstruct-tomogram pick-particles",
        "preflight: PYTOMO_MP_DIR is not set (note the O)\n")
    log("203", 15, "D", "align-tilt-series generate-fiducial-seed",
        "preflight: cannot lock the registry from this node (above)\n")
    log("206", 16, "E", "pick-particles", ran("E", "pick-particles", "matching rotations 1/3\n"))


def _lane(name: str, variant: str) -> dict:
    config = _config()
    config["lane"].update(name=name, variant=variant)
    config["run"]["series"] = ["A", "B", "C", "D", "E"]
    return config


def test_another_lanes_job_fails_only_its_shared_steps_here(tmp_path):
    _project(tmp_path)
    _other_lane(tmp_path)
    result = show_failure.failures(_lane("ribo-bin8", "bin8"), tmp_path, slurm=False)
    found = {(entry["series"], entry["step"]): entry for entry in result["failures"]}
    # Its pick, its stop before picking and its pick still running are its own.
    assert set(found) == {
        ("A", "preflight"), ("A", "pick-particles"), ("B", "generate-fiducial-seed"),
        ("D", "align-tilt-series"), ("D", "preflight"),
    }
    assert found["D", "align-tilt-series"]["job"]["id"] == "201"
    assert found["D", "preflight"]["job"]["id"] == "203"
    text = "\n".join(show_failure.render(result))
    assert "slurm-p-gpu-201.log (job p-gpu 201 of lane ribo-bin4)" in text
    assert "slurm-p-cpu-123.log (job p-cpu 123)" in text

    # That lane sees all of them, but not this lane's pick still running on A.
    # Job 124's config is gone, so its stop counts for every lane.
    other = show_failure.failures(_lane("ribo-bin4", "bin4"), tmp_path, slurm=False)
    found = {(entry["series"], entry["step"]): entry for entry in other["failures"]}
    assert set(found) == {
        ("A", "preflight"), ("B", "generate-fiducial-seed"), ("C", "pick-particles"),
        ("C", "preflight"), ("D", "align-tilt-series"), ("D", "preflight"),
        ("E", "pick-particles"),
    }
    assert (found["C", "pick-particles"]["kind"], found["C", "pick-particles"]["job"]["id"]) \
        == ("crashed", "200")
    assert found["C", "preflight"]["job"]["id"] == "205"
    assert found["E", "pick-particles"]["kind"] == "interrupted"
    text = "\n".join(show_failure.render(other))
    assert "slurm-p-cpu-123.log (job p-cpu 123 of lane ribo-bin8)" in text


def _qc(root: Path, series: str, qc_type: str, status: str, warning: str) -> dict:
    path = f"outputs/p/qc/{series}/{qc_type}.json"
    _write(root / path, json.dumps({"status": status, "warnings": [warning]}), CRASHED)
    return {"id": f"{series}:qc:{qc_type}", "kind": "qc", "path": path,
            "parameters": {"qc_type": qc_type, "tilt_series_id": series}}


def test_a_step_whose_result_failed_qc_is_named_with_its_first_warning(tmp_path):
    _project(tmp_path)
    shift = "residual maximum shift 143.773 px exceeds failure threshold 5.000 px"
    artifacts = [
        _qc(tmp_path, "B", "coarse_alignment_report", "fail", shift),
        _qc(tmp_path, "A", "ctf_estimation", "warning", "3 of 29 views stopped at 30 A"),
    ]
    registry = tmp_path / "outputs" / "p" / "artifacts.json"
    registry.write_text(json.dumps({"artifacts": artifacts}))
    result = show_failure.failures(_config(), tmp_path, slurm=False)
    assert result["qc_failures"] == [{
        "series": "B", "step": "qc-coarse-alignment",
        "path": "outputs/p/qc/B/coarse_alignment_report.json", "warnings": [shift],
    }]
    text = "\n".join(show_failure.render(result))
    assert "ran, but the result failed QC" in text
    assert "B  qc-coarse-alignment  outputs/p/qc/B/coarse_alignment_report.json" in text
    # Asked about one step, it keeps to that step.
    other = show_failure.failures(_config(), tmp_path, step="generate-fiducial-seed", slurm=False)
    assert other["qc_failures"] == []


def test_no_failed_step_still_says_what_failed_before(tmp_path):
    _project(tmp_path)
    result = show_failure.failures(_config(), tmp_path, series=["A"], step="track-fiducials",
                                   slurm=False)
    text = show_failure.render(result)
    assert text[:2] == [
        "outputs/p: lane ribo-bin8 (variant bin8), 0 failed steps",
        "no failed step on record for this lane",
    ]
    assert "failed before, and passed when run again since:" in text


# What lane.sbatch leaves when it stops before its header, as a job did on
# cryoem.se on 26 November 2026, one second after it started.
NO_HEADER = (
    "scripts/activate_env.sh: line 12: micromamba: command not found\n"
    "no environment cryoet-pipeline: run session.sh first\n"
)


def test_a_job_that_stopped_before_its_header_is_counted_and_shown(tmp_path):
    _project(tmp_path)
    _write(tmp_path / "slurm-p-gpu-130.log", NO_HEADER, datetime(2026, 10, 28, 9, tzinfo=UTC))
    result = show_failure.failures(_config(), tmp_path, slurm=False)
    [early] = result["before_start"]
    assert (early["id"], early["name"], early["log"]) == ("130", "p-gpu", "slurm-p-gpu-130.log")
    assert [line["text"] for line in early["lines"]] == NO_HEADER.splitlines()
    assert result["before_start_since"] == []
    # Not a failed step: no step or series is known for it.
    assert all(entry.get("job", {}).get("id") != "130" for entry in result["failures"])

    text = show_failure.render(result)
    assert text[0].endswith("3 failed steps, 1 job stopped before the pipeline started")
    joined = "\n".join(text)
    assert "slurm-p-gpu-130.log (job p-gpu 130, written 2026-10-28T09:00:00+00:00)" in joined
    assert re.search(r"^\s+1  scripts/activate_env.sh: line 12: micromamba", joined, re.M)


def test_with_no_failed_step_the_job_is_not_hidden_behind_none_on_record(tmp_path):
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    _write(tmp_path / "slurm-p-cpu-131.log", NO_HEADER, datetime(2026, 10, 28, 9, tzinfo=UTC))
    text = show_failure.render(show_failure.failures(_config(), tmp_path, slurm=False))
    assert text[0].endswith("0 failed steps, 1 job stopped before the pipeline started")
    assert "no failed step on record for this lane" not in text


def test_an_empty_log_is_shown_as_empty(tmp_path):
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    _write(tmp_path / "slurm-p-cpu-132.log", "", datetime(2026, 10, 28, 9, tzinfo=UTC))
    result = show_failure.failures(_config(), tmp_path, slurm=False)
    assert [entry["id"] for entry in result["before_start"]] == ["132"]
    assert "      (empty)" in show_failure.render(result)


def test_one_followed_by_a_job_of_the_same_stage_is_listed_apart(tmp_path):
    _project(tmp_path)
    # Before jobs 124 and 125 of the same stage started.
    _write(tmp_path / "slurm-p-gpu-121.log", NO_HEADER, datetime(2026, 10, 27, 10, tzinfo=UTC))
    result = show_failure.failures(_config(), tmp_path, slurm=False)
    assert result["before_start"] == []
    [since] = result["before_start_since"]
    assert (since["id"], since["later"]["id"]) == ("121", "124")
    text = show_failure.render(result)
    assert text[0].endswith("3 failed steps")
    assert "  p-gpu 121, then p-gpu 124 (slurm-p-gpu-124.log)" in text


def test_only_the_lanes_own_lane_sbatch_stages_and_the_step_asked_about(tmp_path):
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    config = _config()
    config["run"]["stage"].append({"name": "mra", "from": "mra", "until": "mra",
                                   "time": "1:00:00", "sbatch": []})
    when = datetime(2026, 10, 28, 9, tzinfo=UTC)
    # The MRA's job prints its project line later than lane.sbatch does.
    _write(tmp_path / "slurm-p-mra-140.log", "started     2026-10-28T09:00:00\n", when)
    # Another job of the project's name, not a stage of this lane.
    _write(tmp_path / "slurm-p-ctf_window-141.log", NO_HEADER, when)
    _write(tmp_path / "slurm-p-gpu-142.log", NO_HEADER, when)
    _write(tmp_path / "slurm-p-cpu-143.log", NO_HEADER, when)

    def ids(**options: object) -> list[str]:
        result = show_failure.failures(config, tmp_path, slurm=False, **options)
        return [entry["id"] for entry in result["before_start"]]

    assert ids() == ["142", "143"]
    assert ids(step="track-fiducials") == ["143"]
    assert ids(step="pick-particles") == ["142"]


def _sacct(monkeypatch, workdir: Path, rows: dict[str, tuple[str, str]]) -> None:
    """sacct as it answers show_project and show_failure: the jobs' folders, and
    their states. `rows` is {job id: (job name, state)}."""
    def slurm(command: list[str]) -> list[list[str]]:
        if "JobID,JobName,WorkDir" in command:
            return [[job, name, str(workdir)] for job, (name, _) in rows.items()]
        if "JobID,State,ExitCode,Elapsed" in command:
            asked = command[command.index("-j") + 1].split(",")
            return [[job, state, "1:0", "00:00:01"] for job, (_, state) in rows.items()
                    if job in asked]
        return []
    monkeypatch.setattr(show_project, "_slurm", slurm)


def test_slurm_says_how_it_ended_and_a_job_still_starting_is_left_out(tmp_path, monkeypatch):
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    when = datetime(2026, 10, 28, 9, tzinfo=UTC)
    _write(tmp_path / "slurm-p-gpu-150.log", NO_HEADER, when)
    _write(tmp_path / "slurm-p-cpu-151.log", "", when)
    _sacct(monkeypatch, tmp_path, {"150": ("p-gpu", "FAILED"), "151": ("p-cpu", "RUNNING")})
    result = show_failure.failures(_config(), tmp_path)
    [early] = result["before_start"]
    assert early["id"] == "150"
    assert early["slurm"] == {"state": "FAILED", "exit_code": "1:0", "elapsed": "00:00:01"}
    assert ("(job p-gpu 150, written 2026-10-28T09:00:00+00:00; Slurm: FAILED, exit 1:0, "
            "after 00:00:01)") in "\n".join(show_failure.render(result))


def test_a_missing_log_counts_only_when_slurm_says_the_job_failed(
    tmp_path, tmp_path_factory, monkeypatch
):
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    away = tmp_path_factory.mktemp("away")
    _sacct(monkeypatch, away, {"160": ("p-gpu", "NODE_FAIL"), "161": ("p-cpu", "COMPLETED"),
                                "162": ("p-cpu", "CANCELLED by 501")})
    result = show_failure.failures(_config(), tmp_path)
    assert [(entry["id"], entry.get("missing")) for entry in result["before_start"]] == [
        ("160", True), ("162", True)]
    text = "\n".join(show_failure.render(result))
    assert f"  job p-gpu 160: no log at {away}/slurm-p-gpu-160.log; Slurm: NODE_FAIL" in text


# A header, then a command of lane.sbatch's checks failing under set -e: no
# "preflight:" line and no "preflight   ok".
IN_CHECKS = (
    "job         p-gpu (170) on node2\nstarted     2026-10-28T10:00:00+00:00\n" + HEADER
    + "series      A B\nsteps       prepare-template pick-particles extract-particles\n"
    "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n"
    "IndexError: list index out of range\n"
)


def test_a_job_stopped_inside_the_checks_without_a_preflight_line_is_a_preflight_stop(tmp_path):
    _project(tmp_path)
    (tmp_path / "slurm-p-gpu-170.log").write_text(IN_CHECKS)
    found = _found(tmp_path)
    # A's entry is the later of its two stops, job 124's preflight line and this.
    for series in ("A", "B"):
        entry = found[series, "preflight"]
        assert entry["problem"] == show_failure.CHECKS_STOPPED
        assert entry["job"]["id"] == "170"
        assert "IndexError: list index out of range" in [
            line["text"] for line in entry["slurm_lines"]]
    text = "\n".join(show_failure.render(show_failure.failures(_config(), tmp_path, slurm=False)))
    assert "B  preflight  preflight: stopped in lane.sbatch's checks" in text


def test_a_job_slurm_still_runs_in_its_checks_is_not_a_failure(tmp_path, monkeypatch):
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    (tmp_path / "slurm-p-gpu-170.log").write_text(IN_CHECKS)
    _sacct(monkeypatch, tmp_path, {"170": ("p-gpu", "RUNNING")})
    assert show_failure.failures(_config(), tmp_path)["failures"] == []
    _sacct(monkeypatch, tmp_path, {"170": ("p-gpu", "FAILED")})
    [entry] = [e for e in show_failure.failures(_config(), tmp_path)["failures"]
               if e["series"] == "B"]
    assert entry["problem"] == show_failure.CHECKS_STOPPED


def test_a_log_without_the_checks_line_that_ran_its_steps_is_not_a_preflight_stop(tmp_path):
    # As a job of a lane.sbatch older than its "preflight   ok" line would read,
    # stopped between two steps: no end line, and no step in progress.
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    (tmp_path / "slurm-p-cpu-171.log").write_text(
        "job         p-cpu (171) on node1\nstarted     2026-10-28T10:00:00+00:00\n" + HEADER
        + "series      B\n\n# ---- B (ribo-bin8) ----\n\n# align-tilt-series\n"
        "cryoet align-tilt-series --manifest outputs/p/manifests/B.json\n"
        "# align-tilt-series: 2.0 min, +0.10 GB\n"
    )
    assert show_failure.failures(_config(), tmp_path, slurm=False)["failures"] == []


def test_a_log_that_lost_its_header_but_shows_steps_is_not_a_job_stopped_before_it(tmp_path):
    """Eval case E01 keeps the last 50 lines of a job's log, without its header,
    so that the evidence tools do not list that job."""
    (tmp_path / "outputs" / "p").mkdir(parents=True)
    when = datetime(2026, 10, 28, 9, tzinfo=UTC)
    _write(tmp_path / "slurm-p-cpu-180.log",
           "RuntimeError: IMOD autofidseed failed with exit code 1\n"
           "# generate-fiducial-seed: 0.4 min, +0.00 GB\n\n"
           "# FAILED at generate-fiducial-seed (exit 1) for B.\n", when)
    _write(tmp_path / "slurm-p-cpu-181.log",
           "matching rotations 1/3\n# pick-particles\n", when)
    _write(tmp_path / "slurm-p-cpu-182.log", "\n# ---- B (ribo-bin8) ----\n", when)
    result = show_failure.failures(_config(), tmp_path, slurm=False)
    assert result["before_start"] == [] and result["before_start_since"] == []


def test_no_eval_case_gains_a_job_stopped_before_its_header():
    """The cases' evidence stays what it was: E01's headerless copy of a log
    in particular, which is there so that the tools do not list that job."""
    checked = 0
    for tree in sorted((ROOT / "docs" / "agent" / "cases").glob("*/tree")):
        for config in sorted((tree / "configs" / "lanes").glob("*.toml")):
            result = show_failure.failures(tomllib.loads(config.read_text()), tree, slurm=False)
            assert result["before_start"] == [] and result["before_start_since"] == [], config
            checked += 1
    assert checked >= 9
