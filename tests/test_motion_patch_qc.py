"""MotionCor3's patch alignment, judged: the patch log read as MotionCor3 writes it,
the rule that fails a tilt, the record the motion step keeps per tilt, and the
series report prepare-tilt-series registers under qc/<TS>/motion/."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.motion import (
    PATCH_SHIFT_TOLERANCE_NM,
    MotionCor3MotionCorrectionBackend,
    MotionCor3PatchLog,
    judge_patch_shifts,
    motion_qc_and_register,
    read_motioncor3_patch_log,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    MotionCorrectionQc,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference

ROOT = Path(__file__).resolve().parents[1]

# The first two patches of the -75 deg tilt of EMPIAR-11830 Position_1 as the
# lane's MotionCor3 (5 x 5 patches, 3 frames) wrote them on 7 November 2026: the
# first with overflowing shifts, the second with shifts of hundreds of pixels.
REAL_START = """\
# Patch based alignment
# Number of patches: 25
# Shifts are listed per patch.
# Movie size: 4096  4096  3

#Patch 001 raw and fit shifts
   1   408.00   560.00   167.86 969176258291246825472.00    1
   2   408.00   560.00    83.93 484587953223762968576.00    1
   3   408.00   560.00     0.00 281474976710656.00    1

#Patch 002 raw and fit shifts
   1  1264.00   432.00  -532.74   397.77    0
   2  1264.00   432.00  -271.58   426.96    0
   3  1264.00   432.00     0.00   456.16    0
"""


def _patch_log_text(
    shifts: Sequence[Sequence[tuple[float, float]]],
    *,
    width: int = 4096,
    height: int = 4096,
) -> str:
    """A patch log in MotionCor3's layout: per patch, one line per frame."""

    frames = len(shifts[0])
    lines = [
        "# Patch based alignment",
        f"# Number of patches: {len(shifts)}",
        "# Shifts are listed per patch.",
        f"# Movie size: {width}  {height}  {frames}",
        "",
    ]
    for index, patch in enumerate(shifts, start=1):
        lines.append(f"#Patch {index:03d} raw and fit shifts")
        centre = (100.0 * index, 200.0)
        for frame, (dx, dy) in enumerate(patch, start=1):
            lines.append(
                f"{frame:4d} {centre[0]:8.2f} {centre[1]:8.2f} {dx:8.2f} {dy:8.2f}    0"
            )
        lines.append("")
    return "\n".join(lines)


def _log(**patches: float) -> MotionCor3PatchLog:
    """A three-frame log; each patch moves by `shift` px along x in the last frame."""

    shifts = tuple(((0.0, 0.0), (0.2, 0.1), (shift, 0.0)) for shift in patches.values())
    return MotionCor3PatchLog(
        width=4096,
        height=4096,
        frames=3,
        centres=tuple((0.0, 0.0) for _ in shifts),
        shifts=shifts,
    )


# --- the patch log ---------------------------------------------------------------


def test_the_log_reads_as_motioncor3_wrote_it(tmp_path: Path) -> None:
    path = tmp_path / "Position_1_040_-75.01_20230801_181126_EER-Patch-Patch.log"
    path.write_text(REAL_START)

    log = read_motioncor3_patch_log(path)

    assert (log.width, log.height, log.frames) == (4096, 4096, 3)
    assert log.centres == ((408.0, 560.0), (1264.0, 432.0))
    assert log.shifts[0][0] == (167.86, 969176258291246825472.0)
    assert log.shifts[1] == ((-532.74, 397.77), (-271.58, 426.96), (0.0, 456.16))


@pytest.mark.parametrize(
    ("text", "says"),
    [
        (REAL_START.replace("# Movie size: 4096  4096  3\n", ""), "no '# Movie size' line"),
        (REAL_START.replace("   3  1264.00   432.00     0.00   456.16    0\n", ""),
         "patch 2 has 2 shifts for 3 frames"),
        (REAL_START.replace("-271.58", "-271.5x"), ":13: could not convert"),
        (REAL_START.replace("#Patch 001 raw and fit shifts\n", ""),
         ":6: a shift before the first #Patch line"),
        ("\n".join(REAL_START.splitlines()[:5]) + "\n", "no patches"),
    ],
)
def test_a_log_that_is_not_whole_is_refused(tmp_path: Path, text: str, says: str) -> None:
    path = tmp_path / "x-Patch-Patch.log"
    path.write_text(text)

    with pytest.raises(ValueError, match=says):
        read_motioncor3_patch_log(path)


# --- the rule ---------------------------------------------------------------------


def test_a_tilt_fails_when_at_least_half_its_patches_moved_beyond_the_tolerance() -> None:
    half = judge_patch_shifts(_log(a=30.0, b=30.0, c=1.0, d=1.0), tolerance_px=10.0)
    under_half = judge_patch_shifts(_log(a=30.0, b=1.0, c=1.0, d=1.0), tolerance_px=10.0)

    assert half["failed"] is True
    assert half["patches_beyond"] == 2
    assert under_half["failed"] is False
    assert under_half["patches_beyond"] == 1


def test_a_shift_at_the_tolerance_is_within_it() -> None:
    verdict = judge_patch_shifts(_log(a=10.0, b=10.0), tolerance_px=10.0)

    assert verdict["patches_beyond"] == 0
    assert verdict["failed"] is False


def test_the_largest_shift_of_each_patch_counts_not_its_last() -> None:
    log = MotionCor3PatchLog(
        width=4096, height=4096, frames=3, centres=((0.0, 0.0),),
        shifts=(((0.0, 0.0), (0.0, -40.0), (0.0, 0.0)),),
    )

    verdict = judge_patch_shifts(log, tolerance_px=10.0)

    assert verdict["patches_beyond"] == 1
    assert verdict["median_patch_shift_px"] == 40.0


@pytest.mark.parametrize("lost", [float("nan"), float("inf")])
def test_a_patch_with_a_shift_that_is_not_a_number_is_beyond(lost: float) -> None:
    # NaN placed first: max() would otherwise keep or drop it by its position.
    log = MotionCor3PatchLog(
        width=4096, height=4096, frames=2, centres=((0.0, 0.0),) * 3,
        shifts=(((lost, 0.0), (0.0, 0.0)), ((lost, 0.0), (0.0, 0.0)), ((0.1, 0.0), (0.0, 0.0))),
    )

    verdict = judge_patch_shifts(log, tolerance_px=10.0)

    assert verdict["patches_beyond"] == 2
    assert verdict["failed"] is True
    # Most patches have no finite shift, so neither has the median.
    assert verdict["median_patch_shift_px"] is None
    json.dumps(verdict, allow_nan=False)


def test_the_real_tilt_fails_and_its_record_is_json(tmp_path: Path) -> None:
    path = tmp_path / "x-Patch-Patch.log"
    path.write_text(REAL_START)

    verdict = judge_patch_shifts(read_motioncor3_patch_log(path), tolerance_px=7.879)

    assert verdict == {
        "frames": 3,
        "patches": 2,
        "tolerance_px": 7.879,
        "patches_beyond": 2,
        "median_patch_shift_px": pytest.approx(4.8458812914562e20),
        "failed": True,
    }
    json.dumps(verdict, allow_nan=False)


# --- the record the motion step keeps -----------------------------------------------


def _run_motioncor3(
    tmp_path: Path,
    *,
    patches: tuple[int, int],
    write_log: Callable[[Path], None] | None,
) -> dict[str, object] | None:
    """Correct one 4 x 6 movie at 0.675 A with a stand-in MotionCor3 that writes
    the sum and, through `write_log`, its patch log; the tilt's record."""

    movie = tmp_path / "frames" / "TS_000_-75.0_EER.mrc"
    _write_mrc(movie, np.ones((3, 4, 6), dtype=np.float32))
    executable = tmp_path / "bin" / "MotionCor3"
    executable.parent.mkdir(exist_ok=True)
    executable.touch()

    def stand_in(
        command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
    ) -> subprocess.CompletedProcess[str]:
        _write_mrc(
            Path(command[command.index("-OutMrc") + 1]),
            np.arange(24, dtype=np.float32).reshape(4, 6),
        )
        log_dir = Path(command[command.index("-LogDir") + 1])
        log_dir.mkdir(parents=True, exist_ok=True)
        if write_log is not None:
            write_log(log_dir / f"{movie.stem}-Patch-Patch.log")
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CUDA,
        parameters={
            "motioncor3_executable": executable,
            "motioncor3_patch_x": patches[0],
            "motioncor3_patch_y": patches[1],
            "motioncor3_pixel_size_angstrom": 0.675,
            "motioncor3_version": "1.0.1",
            "overwrite": True,
        },
    )
    artifact = MotionCor3MotionCorrectionBackend(stand_in).correct(
        _manifest([(0, -75.0, movie)]), context
    )[0]
    return artifact.parameters["patch_alignment"]


def _two_far_patches(width: int) -> Callable[[Path], None]:
    """Four patches, two of them 30 px off in the last frame."""

    def write(path: Path) -> None:
        shifts = [[(0.0, 0.0), (0.1, 0.0), (shift, 0.0)] for shift in (30.0, 30.0, 0.5, 0.5)]
        path.write_text(_patch_log_text(shifts, width=width, height=4))

    return write


def test_the_step_judges_each_tilt_from_its_patch_log(tmp_path: Path) -> None:
    record = _run_motioncor3(tmp_path, patches=(2, 2), write_log=_two_far_patches(6))

    assert record is not None
    # The log's frames are as wide as the sum, so its pixel is the sum's, 0.675 A:
    # 1.5 nm is 22.2 px.
    assert record["tolerance_px"] == pytest.approx(15.0 / 0.675, abs=1e-3)
    assert record["patches_beyond"] == 2
    assert record["failed"] is True
    assert record["frames"] == 3
    assert Path(str(record["log"])).name == "TS_000_-75.0_EER-Patch-Patch.log"


def test_the_tolerance_follows_the_pixel_of_the_frames_motioncor3_aligned(
    tmp_path: Path,
) -> None:
    # Frames twice as wide as the sum: their pixel is half the sum's, so 1.5 nm
    # is twice as many of them, and 30 px is within it.
    record = _run_motioncor3(tmp_path, patches=(2, 2), write_log=_two_far_patches(12))

    assert record is not None
    assert record["tolerance_px"] == pytest.approx(2 * 15.0 / 0.675, abs=1e-3)
    assert record["patches_beyond"] == 0
    assert record["failed"] is False


def test_whole_frames_only_have_nothing_to_judge(tmp_path: Path) -> None:
    # Even with a patch log of an earlier run beside it.
    record = _run_motioncor3(tmp_path, patches=(1, 1), write_log=_two_far_patches(6))

    assert record is None


def test_a_tilt_without_its_patch_log_is_not_judged(tmp_path: Path) -> None:
    record = _run_motioncor3(tmp_path, patches=(2, 2), write_log=None)

    assert record is not None
    assert "failed" not in record
    assert record["log"] is None
    assert "wrote no TS_000_-75.0_EER-Patch-Patch.log" in str(record["error"])


def test_a_patch_log_older_than_the_run_is_not_judged(tmp_path: Path) -> None:
    old = tmp_path / "outputs" / "logs" / "motioncor3" / "TS" / "TS_000_-75.0_EER-Patch-Patch.log"
    old.parent.mkdir(parents=True)
    _two_far_patches(6)(old)
    an_hour_ago = time.time() - 3600
    os.utime(old, (an_hour_ago, an_hour_ago))

    record = _run_motioncor3(tmp_path, patches=(2, 2), write_log=None)

    assert record is not None
    assert "older than this run" in str(record["error"])
    assert "failed" not in record


def test_a_patch_log_of_another_grid_is_not_judged(tmp_path: Path) -> None:
    record = _run_motioncor3(tmp_path, patches=(3, 2), write_log=_two_far_patches(6))

    assert record is not None
    assert "has 4 patches, the grid 3 x 2 = 6" in str(record["error"])


def test_a_patch_log_that_does_not_read_is_not_judged(tmp_path: Path) -> None:
    record = _run_motioncor3(
        tmp_path, patches=(2, 2), write_log=lambda path: path.write_text("# nothing\n")
    )

    assert record is not None
    assert "does not read" in str(record["error"])
    assert "no '# Movie size' line" in str(record["error"])


# --- the series report ----------------------------------------------------------------


def _tilt(z: int, angle: float, record: dict[str, object] | None, tmp_path: Path) -> Artifact:
    return Artifact(
        id=f"TS:corrected_projection:{z:03d}:motioncor3",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=tmp_path / f"{z}.mrc",
        backend="motioncor3",
        parameters={
            "z_value": z,
            "tilt_angle_deg": angle,
            "patch_grid_xy": [5, 5],
            "patch_alignment": record,
        },
    )


def _judged(failed: bool, frames: int = 8, median: float | None = 0.5) -> dict[str, object]:
    return {
        "log": "/logs/x-Patch-Patch.log",
        "frames": frames,
        "patches": 25,
        "tolerance_px": 7.879,
        "patches_beyond": 25 if failed else 1,
        "median_patch_shift_px": median,
        "failed": failed,
    }


def _qc(
    tmp_path: Path,
    tilts: Sequence[tuple[int, float, dict[str, object] | None]],
    registry: ArtifactRegistry | None = None,
) -> tuple[Artifact | None, ArtifactRegistry]:
    registry = registry if registry is not None else ArtifactRegistry.empty()
    corrected = [_tilt(z, angle, record, tmp_path) for z, angle, record in tilts]
    for artifact in corrected:
        if not any(existing.id == artifact.id for existing in registry.artifacts):
            registry.add(artifact)
    context = BackendContext(
        output_dir=tmp_path / "outputs", device=DevicePreference.CPU, parameters={}
    )
    manifest = _manifest([(z, angle, None) for z, angle, _ in tilts])
    return motion_qc_and_register(corrected, manifest, context, registry), registry


def test_the_series_fails_and_names_the_tilts_whose_patch_alignment_failed(
    tmp_path: Path,
) -> None:
    artifact, registry = _qc(
        tmp_path,
        [
            (39, -75.0, _judged(True, frames=3, median=472.3)),
            (0, -15.0, _judged(False)),
            (40, 45.0, _judged(True, frames=4, median=695.2)),
        ],
    )

    assert artifact is not None
    assert artifact.path == tmp_path / "outputs/qc/TS/motion/TS_motion_qc.json"
    assert artifact.parameters == {
        "qc_type": "motion_correction", "tilt_series_id": "TS", "status": "fail",
    }
    assert sorted(artifact.parent_ids) == [
        "TS:corrected_projection:000:motioncor3",
        "TS:corrected_projection:039:motioncor3",
        "TS:corrected_projection:040:motioncor3",
    ]
    assert registry.get("TS:qc:motion:report") == artifact
    report = MotionCorrectionQc.model_validate_json(artifact.path.read_text())
    assert report.status == "fail"
    assert report.failed_z_values == [39, 40]
    assert report.tolerance_nm == PATCH_SHIFT_TOLERANCE_NM
    assert report.patch_grid_xy == [5, 5]
    assert [tilt.tilt_angle_deg for tilt in report.tilts] == [-75.0, -15.0, 45.0]
    assert report.warnings == [
        "patch alignment failed on 2 of 3 tilts (-75.0, 45.0 deg; 3-4 frames each): at "
        "least half of each one's patches moved more than 1.5 nm (7.9 px) against the "
        "whole frame (median patch 472-695 px), so those sums are cut into displaced "
        "tiles. [motion] patch_x = patch_y = 1 aligns whole frames only"
    ]


def test_a_tilt_not_judged_makes_a_warning(tmp_path: Path) -> None:
    artifact, _ = _qc(
        tmp_path,
        [
            (0, -15.0, _judged(False)),
            (1, -12.0, {"log": None, "error": "MotionCor3 wrote no x-Patch-Patch.log in /l"}),
        ],
    )

    assert artifact is not None
    report = MotionCorrectionQc.model_validate_json(artifact.path.read_text())
    assert report.status == "warning"
    assert report.failed_z_values == []
    assert report.warnings == [
        "patch alignment not judged on 1 of 2 tilts (-12.0 deg): MotionCor3 wrote no "
        "x-Patch-Patch.log in /l"
    ]


def test_a_failed_tilt_outranks_one_not_judged(tmp_path: Path) -> None:
    artifact, _ = _qc(
        tmp_path,
        [(0, -15.0, {"log": None, "error": "no log"}), (39, -75.0, _judged(True, frames=3))],
    )

    assert artifact is not None
    report = MotionCorrectionQc.model_validate_json(artifact.path.read_text())
    assert report.status == "fail"
    assert len(report.warnings) == 2


def test_every_tilt_within_the_tolerance_passes(tmp_path: Path) -> None:
    artifact, _ = _qc(tmp_path, [(0, -15.0, _judged(False)), (1, -12.0, _judged(False))])

    assert artifact is not None
    assert artifact.parameters["status"] == "pass"
    report = MotionCorrectionQc.model_validate_json(artifact.path.read_text())
    assert report.warnings == []


def test_whole_frames_remove_the_report_of_an_earlier_patch_run(tmp_path: Path) -> None:
    earlier, registry = _qc(tmp_path, [(39, -75.0, _judged(True, frames=3))])
    assert earlier is not None and earlier.path.is_file()

    # The same tilt corrected again with -Patch 1 1: no patch log, nothing judged.
    again, registry = _qc(tmp_path, [(39, -75.0, None)], registry)

    assert again is None
    assert not earlier.path.exists()
    assert all(artifact.kind != ArtifactKind.QC for artifact in registry.artifacts)


def test_a_rerun_replaces_the_report(tmp_path: Path) -> None:
    _, registry = _qc(tmp_path, [(39, -75.0, _judged(True, frames=3))])

    again, registry = _qc(tmp_path, [(39, -75.0, _judged(False))], registry)

    assert again is not None
    reports = [artifact for artifact in registry.artifacts if artifact.kind == ArtifactKind.QC]
    assert [report.parameters["status"] for report in reports] == ["pass"]


# --- where people and the assistant read it --------------------------------------------


def test_prepare_tilt_series_prints_the_verdict_and_registers_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from cryoet_pipeline import cli
    from cryoet_pipeline.models import AxisOrder

    tilts = [(0, -15.0, _judged(False)), (39, -75.0, _judged(True, frames=3, median=472.3))]

    class StandInMotionCor3:
        name = "motioncor3"

        def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
            sums = []
            for z, angle, record in tilts:
                path = context.output_dir / f"{z}.mrc"
                _write_mrc(path, np.full((2, 2), float(z), dtype=np.float32))
                artifact = _tilt(z, angle, record, tmp_path)
                sums.append(artifact.model_copy(update={
                    "path": path, "shape": (2, 2), "dtype": "float32",
                    "axis_order": AxisOrder.YX, "pixel_spacing_angstrom": 1.9,
                }))
            return sums

    monkeypatch.setattr(cli, "_motion_backend", lambda name: StandInMotionCor3())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_manifest([(z, angle, None) for z, angle, _ in tilts]).model_dump_json())
    registry_path = tmp_path / "artifacts.json"
    ArtifactRegistry.empty().write(registry_path)

    result = CliRunner().invoke(cli.app, [
        "prepare-tilt-series", "--manifest", str(manifest), "--registry", str(registry_path),
        "--out", str(tmp_path / "outputs"), "--motion-backend", "motioncor3",
        "--device", "cuda", "--storage-policy", "debug",
    ])

    assert result.exit_code == 0, result.output
    assert "motion QC: fail (" in result.output
    assert "  patch alignment failed on 1 of 2 tilts (-75.0 deg; 3 frames each)" in result.output
    registry = ArtifactRegistry.load(registry_path)
    assert registry.get("TS:qc:motion:report").parameters["status"] == "fail"
    # The stack is built all the same: the verdict stops nothing.
    assert len(registry.by_kind(ArtifactKind.TILT_STACK)) == 1


def test_show_project_reads_the_verdict_as_the_motion_steps(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import show_project

    config = tomllib.loads((ROOT / "configs" / "lanes" / "ryr1-bin8.toml").read_text())
    config["run"]["project"] = "outputs/p"
    config["run"]["series"] = ["A"]
    project = tmp_path / "outputs" / "p"
    report = project / "qc" / "A" / "motion" / "A_motion_qc.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "fail", "warnings": ["patch alignment failed"]}))
    (project / "artifacts.json").write_text(json.dumps({"schema_version": 1, "artifacts": [{
        "id": "A:qc:motion:report", "kind": "qc", "path": str(report.relative_to(tmp_path)),
        "parent_ids": [],
        "parameters": {"qc_type": "motion_correction", "tilt_series_id": "A", "status": "fail"},
    }]}))
    (project / "step_timings.jsonl").write_text(json.dumps({
        "series": "A", "step": "prepare-tilt-series", "seconds": 1.0, "disk_delta_bytes": 0,
        "exit_code": 0, "finished_at": "2026-11-07T17:08:00+00:00",
    }) + "\n")
    (project / "manifests").mkdir()
    (project / "manifests" / "A.json").write_text("{}")

    summary = show_project.summarize(config, tmp_path, slurm=False)

    motion = summary["series"][0]["steps"]["prepare-tilt-series"]
    assert motion["state"] == "fail"


def _manifest(images: Sequence[tuple[int, float, Path | None]]) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS",
        source_mdoc=Path("TS.mdoc"),
        raw_pixel_spacing_angstrom=0.675,
        images=[
            TiltImage(
                z_value=z,
                tilt_angle_deg=angle,
                subframe_path=f"frames/{z}.mrc",
                local_frame_file=movie,
                num_subframes=3,
                pixel_spacing_angstrom=0.675,
                binning=1,
            )
            for z, angle, movie in images
        ],
    )


def _write_mrc(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
