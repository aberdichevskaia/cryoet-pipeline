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
