"""scripts/mc3_patch_check.py: the commands it reruns, and its tile measure."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.motion import MotionCor3MotionCorrectionBackend
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import TiltImage, TiltSeriesManifest
from cryoet_pipeline.runtime import DevicePreference

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mc3_patch_check.py"

_spec = importlib.util.spec_from_file_location("mc3_patch_check", SCRIPT)
assert _spec is not None and _spec.loader is not None
mc3_patch_check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mc3_patch_check
_spec.loader.exec_module(mc3_patch_check)

# Shared with the fake MotionCor3 below, which runs as its own process.
_SYNTHETIC = '''
import numpy as np
from scipy.ndimage import gaussian_filter


def texture(size, seed):
    rng = np.random.default_rng(seed)
    fine = gaussian_filter(rng.normal(size=(size, size)), 2.0) * 10.0
    coarse = gaussian_filter(rng.normal(size=(size, size)), 20.0) * 30.0
    return (fine + coarse + 100.0).astype(np.float32)


def tile_pattern(amplitude, grid=5, phase=0.0):
    """Whole-pixel (dy, dx) for each tile, up to `amplitude` px."""
    shifts = []
    for j in range(grid):
        row = []
        for i in range(grid):
            angle = 2.4 * (j * grid + i) + phase
            dy = int(round(amplitude * np.sin(angle)))
            dx = int(round(amplitude * np.cos(angle)))
            row.append((dy, dx))
        shifts.append(row)
    return shifts


def tiled(base, shifts):
    """Each tile shows the base moved by its (dy, dx): out[y, x] = base[y+dy, x+dx]."""
    out = np.empty_like(base)
    ny, nx = len(shifts), len(shifts[0])
    h, w = base.shape
    for j in range(ny):
        for i in range(nx):
            y0, y1 = int(round(j * h / ny)), int(round((j + 1) * h / ny))
            x0, x1 = int(round(i * w / nx)), int(round((i + 1) * w / nx))
            dy, dx = shifts[j][i]
            out[y0:y1, x0:x1] = np.roll(base, (-dy, -dx), axis=(0, 1))[y0:y1, x0:x1]
    return out
'''
_namespace: dict[str, object] = {}
exec(_SYNTHETIC, _namespace)
texture = _namespace["texture"]
tile_pattern = _namespace["tile_pattern"]
tiled = _namespace["tiled"]

_FAKE_MOTIONCOR3 = '''#!{python}
"""Stands in for MotionCor3. The movie's first frame says how far -Patch 5 5
moves tiles; in its second, pixel (0, 0) asks for another tile pattern in a
rerun than in the pipeline's run (a patch result that changes from run to run)
and pixel (0, 1) for fresh noise in every sum (MotionCor3 never gives the same
pixels twice)."""
import os
import sys

import mrcfile
import numpy as np
{synthetic}
args = sys.argv[1:]
if "-InMrc" not in args:
    sys.exit(0)
movie = args[args.index("-InMrc") + 1]
out = args[args.index("-OutMrc") + 1]
patch = (int(args[args.index("-Patch") + 1]), int(args[args.index("-Patch") + 2]))
with mrcfile.open(movie, permissive=True) as mrc:
    amplitude = float(mrc.data[0, 0, 0])
    changing = float(mrc.data[1, 0, 0]) > 0
    jitter = float(mrc.data[1, 0, 1]) > 0
    size = mrc.data.shape[-1]
image = texture(size, seed=7)
if patch != (1, 1) and amplitude > 0:
    phase = 1.7 if changing and out.endswith("_p5.mrc") else 0.0
    image = tiled(image, tile_pattern(amplitude, phase=phase))
if jitter:
    rng = np.random.default_rng(int.from_bytes(os.urandom(4), "little"))
    image = image + rng.normal(scale=0.15 * image.std(), size=image.shape).astype(np.float32)
with mrcfile.new(out, overwrite=True) as mrc:
    mrc.set_data(image)
print("MotionCor3 stand-in: done")
'''


def test_tile_shifts_find_each_tiles_own_displacement() -> None:
    base = texture(1024, seed=1)
    shifts = tile_pattern(30)
    measured = mc3_patch_check.tile_shifts(tiled(base, shifts), base, (5, 5))

    assert np.allclose(measured[..., :2], np.array(shifts, dtype=float), atol=0.3)
    assert measured[..., 2].min() > 0.9


def test_one_shift_of_the_whole_image_is_no_displacement() -> None:
    base = texture(1024, seed=2)
    moved = np.roll(base, (-5, 7), axis=(0, 1))  # moved[y, x] = base[y+5, x-7]
    summary = mc3_patch_check.displacement_summary(
        mc3_patch_check.tile_shifts(moved, base, (5, 5)), scale=4
    )

    assert summary["matched"] == 25
    assert summary["common_px"] == pytest.approx([20.0, -28.0], abs=1.0)
    assert summary["max_px"] < 1.0


def test_identical_sums_show_nothing_and_displacements_scale_to_raw_pixels() -> None:
    base = texture(1024, seed=3)
    same = mc3_patch_check.displacement_summary(
        mc3_patch_check.tile_shifts(base, base, (5, 5)), scale=4
    )
    assert same["max_px"] < 0.5

    shifts = tile_pattern(10)
    moved = mc3_patch_check.displacement_summary(
        mc3_patch_check.tile_shifts(tiled(base, shifts), base, (5, 5)), scale=4
    )
    vectors = np.array(shifts, dtype=float) * 4
    expected = np.hypot(*(vectors - np.median(vectors.reshape(-1, 2), axis=0)).transpose(2, 0, 1))
    assert moved["max_px"] == pytest.approx(expected.max(), abs=1.5)


def test_a_tile_without_content_is_reported_not_measured() -> None:
    base = texture(1024, seed=4)
    flat = base.copy()
    flat[:204, :204] = 100.0
    summary = mc3_patch_check.displacement_summary(
        mc3_patch_check.tile_shifts(flat, base, (5, 5)), scale=1
    )
    assert summary["matched"] == 24
    assert not summary["matched_mask"][0, 0]


def test_rewrite_moves_the_outputs_and_sets_the_grid_and_keeps_the_rest() -> None:
    command = [
        "/sw/MotionCor3", "-InEer", "/data/a.eer", "-OutMrc", "/tmp/x/a.mrc",
        "-Patch", "5", "5", "-FtBin", "1", "-Align", "1", "-OutStack", "0", "1",
        "-FmDose", "0", "-PixSize", "1.9037", "-Cs", "0", "-Gpu", "2", "3",
        "-LogDir", "/p/logs", "-OutAln", "/p/aln", "-EerSampling", "1",
        "-FmIntFile", "/p/a.fmint", "-Gain", "/g/gain.gain", "-FlipGain", "1",
    ]
    new = mc3_patch_check.rewrite_command(
        command,
        patches=(1, 1),
        out_mrc=Path("/o/a_p1.mrc"),
        log_dir=Path("/o/logs/a_p1"),
        aln_dir=Path("/o/aln/a_p1"),
        fmint=None,
    )
    assert new == [
        "/sw/MotionCor3", "-InEer", "/data/a.eer", "-OutMrc", "/o/a_p1.mrc",
        "-Patch", "1", "1", "-FtBin", "1", "-Align", "1", "-OutStack", "0", "1",
        "-FmDose", "0", "-PixSize", "1.9037", "-Cs", "0", "-Gpu", "0",
        "-LogDir", "/o/logs/a_p1", "-OutAln", "/o/aln/a_p1", "-EerSampling", "1",
        "-FmIntFile", "/p/a.fmint", "-Gain", "/g/gain.gain", "-FlipGain", "1",
    ]
    as_run = mc3_patch_check.rewrite_command(
        command, patches=None, out_mrc=Path("/o/a_p5.mrc"), log_dir=Path("/o/l"),
        aln_dir=Path("/o/a"), fmint=Path("/o/a.fmint"),
    )
    assert as_run[as_run.index("-Patch") + 1 : as_run.index("-Patch") + 3] == ["5", "5"]
    assert as_run[as_run.index("-FmIntFile") + 1] == "/o/a.fmint"

    with pytest.raises(ValueError, match="has no -LogDir"):
        mc3_patch_check.rewrite_command(
            command[:-12], patches=None, out_mrc=Path("/o/a.mrc"), log_dir=Path("/o/l"),
            aln_dir=Path("/o/a"), fmint=None,
        )


def _pipeline_project(
    tmp_path: Path,
    amplitudes: dict[float, float],
    *,
    changing: frozenset[float] = frozenset(),
    jitter: bool = False,
) -> tuple[Path, Path]:
    """Run the pipeline's MotionCor3 step with the stand-in, as on the cluster."""

    executable = tmp_path / "bin" / "MotionCor3"
    executable.parent.mkdir()
    executable.write_text(
        _FAKE_MOTIONCOR3.format(python=sys.executable, synthetic=_SYNTHETIC)
    )
    executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
    images = []
    for z, (angle, amplitude) in enumerate(amplitudes.items()):
        movie = tmp_path / "frames" / f"TS_TEST_{z:03d}_{angle:.1f}.mrc"
        movie.parent.mkdir(exist_ok=True)
        frames = np.zeros((2, 1024, 1024), dtype=np.float32)
        frames[0] = amplitude
        frames[1, 0, 0] = 1.0 if angle in changing else 0.0
        frames[1, 0, 1] = 1.0 if jitter else 0.0
        with mrcfile.new(movie, overwrite=True) as mrc:
            mrc.set_data(frames)
        images.append(
            TiltImage(
                z_value=z, tilt_angle_deg=angle, subframe_path=f"frames/{movie.name}",
                local_frame_file=movie, num_subframes=2, pixel_spacing_angstrom=1.9037,
                binning=1,
            )
        )
    manifest = TiltSeriesManifest(
        tilt_series_id="TS_TEST", source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.9037, images=images,
    )
    project = tmp_path / "outputs"
    context = BackendContext(
        output_dir=project,
        device=DevicePreference.CUDA,
        parameters={
            "motioncor3_executable": executable,
            "motioncor3_pixel_size_angstrom": 1.9037,
            "motioncor3_version": "stand-in",
        },
    )
    artifacts = MotionCor3MotionCorrectionBackend().correct(manifest, context)
    registry = ArtifactRegistry.empty()
    registry.extend(artifacts)
    registry.write(project / "artifacts.json")
    return project, executable


def _check(project: Path, out: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    assert mc3_patch_check.main(
        ["plan", "--project", str(project), "--series", "TS_TEST",
         "--angles", "-75", "45", "-15", "--out", str(out)]
    ) == 0
    assert mc3_patch_check.main(["run", "--out", str(out)]) == 0
    capsys.readouterr()
    code = mc3_patch_check.main(["compare", "--out", str(out)])
    return code, capsys.readouterr().out


def test_moved_tiles_at_the_ends_are_found_and_the_prediction_holds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, _ = _pipeline_project(tmp_path, {-75.0: 30.0, 45.0: 20.0, -15.0: 0.0})
    out = tmp_path / "check"

    code, text = _check(project, out, capsys)

    assert code == 0
    assert "control: at every tilt the rerun of the recorded command puts each tile" in text
    assert (
        " -75.0 deg: the 5x5 sums' tiles moved by more than 1.5 nm "
        "(pipeline sum: moved; rerun: moved)" in text
    )
    assert "  45.0 deg: the 5x5 sums' tiles moved by more than 1.5 nm" in text
    assert (
        " -15.0 deg: no tile of the 5x5 sums moved by more than 1.5 nm "
        "(pipeline sum: unchanged; rerun: unchanged)" in text
    )
    assert "prediction (ends moved, -15 not): HELD" in text
    assert (out / "compare.png").is_file()
    plan = json.loads((out / "plan.json").read_text())
    p1 = plan["tilts"][0]["runs"]["p1"]["command"]
    assert p1[p1.index("-Patch") + 1 : p1.index("-Patch") + 3] == ["1", "1"]
    assert plan["tilts"][0]["pipeline_patches"] == [5, 5]


def test_no_moved_tiles_and_the_prediction_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, _ = _pipeline_project(tmp_path, {-75.0: 0.0, 45.0: 0.0, -15.0: 0.0})

    code, text = _check(project, tmp_path / "check", capsys)

    assert code == 0
    assert "prediction (ends moved, -15 not): DID NOT HOLD" in text


def test_patch_results_that_change_from_run_to_run_are_named_and_still_judged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # As job 3901454 on max: at the ends a rerun of the same command gives
    # another patchwork; no sum has the same pixels twice.
    project, _ = _pipeline_project(
        tmp_path,
        {-75.0: 30.0, 45.0: 20.0, -15.0: 0.0},
        changing=frozenset({-75.0, 45.0}),
        jitter=True,
    )

    code, text = _check(project, tmp_path / "check", capsys)

    assert code == 0
    assert (
        "control: at -75, 45 deg the rerun of the recorded command (same executable, "
        "sha256 checked) does not put the tiles where the pipeline's sum has them, "
        "while at the other tilts it does" in text
    )
    assert "binary, the flags" not in text
    assert " -75.0 deg: the 5x5 sums' tiles moved" in text
    assert "  45.0 deg: the 5x5 sums' tiles moved" in text
    assert " -15.0 deg: no tile of the 5x5 sums moved" in text
    assert "prediction (ends moved, -15 not): HELD" in text
    # The noise keeps r below 0.99 at -15, where the tiles stay in place.
    row = next(line for line in text.splitlines() if line.startswith(" -15.0  "))
    assert float(row.split()[1]) < 0.99
    assert " same " in row


def test_a_pipeline_sum_unlike_its_rerun_is_named_by_tilt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, _ = _pipeline_project(tmp_path, {-75.0: 30.0, 45.0: 20.0, -15.0: 0.0})
    pipeline_sum = project / "corrected" / "TS_TEST" / "TS_TEST_002_mc3.mrc"
    with mrcfile.open(pipeline_sum, mode="r+") as mrc:
        mrc.data[:] = tiled(np.array(mrc.data), tile_pattern(12))

    code, text = _check(project, tmp_path / "check", capsys)

    assert code == 0
    assert "control: at -15 deg the rerun of the recorded command" in text
    assert (
        " -15.0 deg: the two 5x5 sums disagree (pipeline sum: moved; rerun: unchanged)"
        in text
    )
    assert "prediction (ends moved, -15 not): DID NOT HOLD" in text


def test_no_tilt_like_its_rerun_points_at_the_sums_not_at_motioncor3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, _ = _pipeline_project(tmp_path, {-75.0: 30.0, 45.0: 20.0, -15.0: 0.0})
    for z in (0, 1, 2):
        pipeline_sum = project / "corrected" / "TS_TEST" / f"TS_TEST_{z:03d}_mc3.mrc"
        with mrcfile.open(pipeline_sum, mode="r+") as mrc:
            mrc.data[:] = tiled(np.array(mrc.data), tile_pattern(12, phase=0.9))

    code, text = _check(project, tmp_path / "check", capsys)

    assert code == 0
    assert "control: at no tilt does the rerun of the recorded command" in text
    assert "Check that these sums are the ones this command made" in text


def test_a_sum_moves_on_two_far_tiles_and_stays_on_most_found_near() -> None:
    def summary(found: dict[tuple[int, int], tuple[float, float]]) -> dict:
        shifts = np.zeros((5, 5, 3))
        for (j, i), (dy, dx) in found.items():
            shifts[j, i] = (dy, dx, 0.9)
        return mc3_patch_check.displacement_summary(shifts, scale=1)

    verdict = mc3_patch_check.sum_verdict
    far_six = {(0, 0): (50, 0), (0, 1): (-50, 0), (1, 0): (0, 50), (1, 1): (0, -50),
               (2, 2): (40, 40), (3, 3): (-40, -40)}
    assert verdict(summary(far_six), 7.9) == "moved"  # +45 of job 3901454: 6 found
    assert verdict(summary({(0, 0): (50, 0)}), 7.9) == "unclear"  # one alone
    every = {(j, i): (0.0, 0.0) for j in range(5) for i in range(5)}
    assert verdict(summary(every), 7.9) == "unchanged"
    one_far = dict(every)
    one_far[(2, 2)] = (30.0, 0.0)
    assert verdict(summary(one_far), 7.9) == "unclear"  # could be a false match
    ten = {(j, i): (0.0, 0.0) for j in range(2) for i in range(5)}
    assert verdict(summary(ten), 7.9) == "unclear"  # fewer than half found
    # The same far shift for every found tile is one shift of the image.
    same_far = {(j, i): (50.0, 0.0) for j in range(5) for i in range(5)}
    assert verdict(summary(same_far), 7.9) == "unchanged"


def test_plan_refuses_a_motioncor3_that_changed_since_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, executable = _pipeline_project(tmp_path, {-75.0: 0.0})
    executable.write_text(executable.read_text() + "\n# rebuilt\n")

    code = mc3_patch_check.main(
        ["plan", "--project", str(project), "--series", "TS_TEST",
         "--angles", "-75", "--out", str(tmp_path / "check")]
    )

    assert code == 1
    assert "changed since the pipeline ran it" in capsys.readouterr().err


def test_the_control_needs_most_tiles_found_in_the_same_place() -> None:
    def summary(found: dict[tuple[int, int], tuple[float, float]]) -> dict:
        shifts = np.zeros((5, 5, 3))
        for (j, i), (dy, dx) in found.items():
            shifts[j, i] = (dy, dx, 0.9)
        return mc3_patch_check.displacement_summary(shifts, scale=1)

    every = {(j, i): (0.4, -0.3) for j in range(5) for i in range(5)}
    assert mc3_patch_check.reproduced(summary(every))
    # +45 of job 3901454: one tile found, 0.0 px away, the other 24 nowhere.
    assert not mc3_patch_check.reproduced(summary({(2, 2): (0.0, 0.0)}))
    one_off = dict(every)
    one_off[(4, 4)] = (3.0, 0.0)
    assert not mc3_patch_check.reproduced(summary(one_off))
