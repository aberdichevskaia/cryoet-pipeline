"""The project summary: per series and step, the last run and the QC verdict this
lane sees, and the lane's jobs as their logs tell it."""

import copy
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_project  # noqa: E402

LANE = tomllib.loads((ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text())
FINISHED = "2026-10-27T10:00:00+00:00"


def _config() -> dict:
    config = copy.deepcopy(LANE)
    config["run"]["project"] = "outputs/p"
    config["run"]["series"] = ["A", "B", "C"]
    return config


def _run(series: str, step: str, exit_code: int = 0, **lane: str) -> dict:
    # A row as run_series.py writes it; without `lane`, one written before rows
    # named their lane.
    return {"series": series, "step": step, "seconds": 1.0, "disk_delta_bytes": 0,
            "exit_code": exit_code, "finished_at": FINISHED, **lane}


def _qc(root: Path, series: str, qc_type: str, status: str, warnings: tuple[str, ...] = (),
        variant: str | None = None) -> dict:
    path = root / "outputs" / "p" / "qc" / series / f"{qc_type}_{variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "warnings": list(warnings)}))
    # The registry's copy of the status says pass: the document is the verdict.
    return {"id": f"{series}:qc:{qc_type}:{variant}", "kind": "qc",
            "path": str(path.relative_to(root)), "variant": variant, "parent_ids": [],
            "parameters": {"qc_type": qc_type, "tilt_series_id": series, "status": "pass"}}


def _project(root: Path) -> list[str]:
    """Series A ran the whole chain, B crashed while seeding, C never ran, and D
    is in the project but not in the lane's list."""
    project = root / "outputs" / "p"
    chain = [name for name, _ in run_series.build_steps(_config(), project, "A")]
    this_lane = {"lane": "ribo-bin8", "variant": "bin8"}
    seed = chain.index("generate-fiducial-seed")
    rows = [_run("A", step, **this_lane) for step in chain]
    rows += [_run("B", step, **this_lane) for step in chain[:seed]]
    rows += [
        _run("B", "generate-fiducial-seed", 139, **this_lane),
        # Another lane of the project picked A under its own label, and failed.
        _run("A", "pick-particles", 1, lane="ribo-bin4", variant="bin4"),
        # A run from before rows named their lane counts for every lane.
        _run("D", "pick-particles"),
    ]
    project.mkdir(parents=True)
    (project / "step_timings.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + '{"series": "A", "st'
    )
    artifacts = [
        _qc(root, "A", "coarse_alignment_report", "pass"),
        _qc(root, "A", "fine_alignment", "warning",
            ("mean residual 0.900 nm exceeds warning threshold 0.800 nm",)),
        _qc(root, "A", "ctf_estimation", "pass"),
        # A CTF experiment under its own label is not the lane's CTF.
        _qc(root, "A", "ctf_estimation", "fail", ("every view unfitted",), variant="ctfw5"),
        _qc(root, "A", "particle_picking", "pass", variant="bin8"),
        _qc(root, "A", "particle_picking", "fail", variant="bin4"),
        _qc(root, "A", "picks_score_selection", "warning",
            ("the threshold keeps every pick",), variant="bin8"),
        _qc(root, "B", "coarse_alignment_report", "fail",
            ("residual maximum shift 144.000 px exceeds failure threshold 5.000 px",)),
    ]
    (project / "artifacts.json").write_text(
        json.dumps({"schema_version": 1, "artifacts": artifacts})
    )
    (project / "manifests").mkdir()
    for series in ("A", "B", "C", "D"):
        (project / "manifests" / f"{series}.json").write_text("{}")
    (root / "slurm-p-cpu-123.log").write_text(
        "job         p-cpu (123) on node1\n"
        "started     2026-10-27T09:00:00+00:00\n"
        "project     outputs/p   config configs/lanes/ribo-bin8.toml\n"
        "series      A B C\n"
        "preflight   ok\n"
        "# FAILED at generate-fiducial-seed (exit 139) for B.\n"
    )
    # The same job name from another project with the same directory name.
    (root / "slurm-p-cpu-99.log").write_text(
        "job         p-cpu (99) on node1\n"
        "project     outputs/elsewhere/p   config x.toml\n"
        "finished    2026-10-20T09:00:00+00:00\n"
    )
    return chain


def test_each_step_shows_its_last_run_and_the_verdict_this_lane_sees(tmp_path):
    chain = _project(tmp_path)
    summary = show_project.summarize(_config(), tmp_path, slurm=False)
    assert summary["steps"] == chain
    assert [row["id"] for row in summary["series"]] == ["A", "B", "C", "D"]
    steps = {row["id"]: row["steps"] for row in summary["series"]}

    a = steps["A"]
    assert a["qc-coarse-alignment"]["state"] == "ok"
    # The document's status, not the registry's copy of it.
    assert a["fine-align-tilt-series"]["state"] == "warning"
    # Neither the CTF experiment nor the other lane's picking is this lane's.
    assert a["estimate-ctf"]["state"] == "ok"
    assert a["pick-particles"]["state"] == "ok"
    # Ran, and has no QC of its own.
    assert a["extract-particles"]["state"] == "ok"

    b = steps["B"]
    assert b["qc-coarse-alignment"]["state"] == "fail"
    assert b["generate-fiducial-seed"]["state"] == "crashed"
    assert b["generate-fiducial-seed"]["exit_code"] == 139
    assert b["track-fiducials"]["state"] == "none"

    assert {entry["state"] for entry in steps["C"].values()} == {"none"}
    assert steps["D"]["pick-particles"]["state"] == "ok"

    other = {row["id"]: row["other_qc"] for row in summary["series"]}
    assert [record["qc_type"] for record in other["A"]] == ["picks_score_selection"]


def test_the_jobs_are_this_projects_as_their_logs_tell_it(tmp_path):
    _project(tmp_path)
    jobs = show_project.summarize(_config(), tmp_path, slurm=False)["jobs"]
    assert [job["id"] for job in jobs] == ["123"]
    assert jobs[0]["outcome"] == "failed"
    assert jobs[0]["series"] == ["A", "B", "C"]
    assert jobs[0]["failed"] == [
        {"series": "B", "step": "generate-fiducial-seed", "exit_code": 139}
    ]


def test_the_table_marks_a_crash_and_points_at_its_log(tmp_path):
    _project(tmp_path)
    summary = show_project.summarize(_config(), tmp_path, series=["A", "B"], slurm=False)
    lines = show_project.render(summary)
    row_b = next(line for line in lines if line.startswith("B "))
    assert "X139" in row_b.split()
    assert "F" in row_b.split()
    text = "\n".join(lines)
    assert "B  generate-fiducial-seed  exit 139" in text
    assert "slurm-p-cpu-123.log" in text
    assert "residual maximum shift 144.000 px exceeds failure threshold" in text
    assert "the threshold keeps every pick" in text
    assert "every view unfitted" not in text


def test_a_timing_row_names_the_lane_that_ran_it(tmp_path):
    run_series.record_timing(tmp_path, series="A", step="pick-particles", lane="ribo-bin8",
                             variant="bin8", seconds=1.0, disk_delta_bytes=0, exit_code=0)
    row = json.loads((tmp_path / "step_timings.jsonl").read_text())
    assert (row["lane"], row["variant"]) == ("ribo-bin8", "bin8")
