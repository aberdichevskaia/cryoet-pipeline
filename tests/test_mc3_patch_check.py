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


def tile_pattern(amplitude, grid=5):
    """Whole-pixel (dy, dx) for each tile, up to `amplitude` px."""
    shifts = []
    for j in range(grid):
        row = []
        for i in range(grid):
            angle = 2.4 * (j * grid + i)
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
"""Stands in for MotionCor3: the movie's first pixel is how far -Patch 5 5 moves tiles."""
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
    size = mrc.data.shape[-1]
image = texture(size, seed=7)
if patch != (1, 1) and amplitude > 0:
    image = tiled(image, tile_pattern(amplitude))
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


def _pipeline_project(tmp_path: Path, amplitudes: dict[float, float]) -> tuple[Path, Path]:
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
        with mrcfile.new(movie, overwrite=True) as mrc:
            mrc.set_data(np.full((2, 1024, 1024), amplitude, dtype=np.float32))
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
    assert "control: each rerun as the pipeline ran it reproduces" in text
    assert " -75.0 deg: tiles moved by more than 1.5 nm" in text
    assert "  45.0 deg: tiles moved by more than 1.5 nm" in text
    assert " -15.0 deg: no tile moved by more than 1.5 nm" in text
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


def test_a_sum_the_rerun_does_not_reproduce_fails_the_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project, _ = _pipeline_project(tmp_path, {-75.0: 30.0, 45.0: 20.0, -15.0: 0.0})
    pipeline_sum = project / "corrected" / "TS_TEST" / "TS_TEST_002_mc3.mrc"
    with mrcfile.open(pipeline_sum, mode="r+") as mrc:
        mrc.data[:] = tiled(np.array(mrc.data), tile_pattern(12))

    code, text = _check(project, tmp_path / "check", capsys)

    assert code == 2
    assert "CONTROL FAILED" in text
    assert "prediction" not in text


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
