"""scripts/tomograms_lane_test.py: the test of a tomograms lane on a series a lane
has picked, held to its known answer on a synthetic source project.

The source project is made as a lane leaves it: tilt's output with Y as sections,
its Zarr, its angles and pytom's picks. prepare turns the output with IMOD's own
clip, the lane's import runs through run_series.py, and check finds the volume
identical; entered with the other answer of the mirror pair, it is told apart.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import pytest
import zarr
from test_picks_import import _pytom_star, scene  # noqa: F401  (scene is a fixture)

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder
from cryoet_pipeline.tomogram_import import import_tomogram

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import tomograms_lane_test as lane_test  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("clip") is None,
                                reason="the test turns the volume with IMOD's clip")
LOW, HIGH = -60.0, 50.0


@pytest.fixture
def source(tmp_path: Path, scene: dict[str, Any]) -> dict[str, Any]:  # noqa: F811
    """A project as gag-bin8 leaves one: tomogram, tilt's output, angles, picks."""

    project = tmp_path / "source"
    folder = project / "tomograms" / "TS_01"
    folder.mkdir(parents=True)
    canonical = scene["volume"]
    rec = folder / "TS_01_bin8.rec"
    with mrcfile.new(rec) as mrc:
        mrc.set_data(np.ascontiguousarray(canonical.transpose(1, 0, 2)))
        mrc.voxel_size = 10.0
        # The label IMOD's tilt writes (5.1.12), which clip keeps and adds to.
        mrc.header.label[0] = b"TILT: Tomographic reconstruction".ljust(80)
        mrc.header.nlabl = 1
    zarr.open(str(folder / "TS_01_bin8.zarr"), mode="w", shape=canonical.shape,
              dtype=np.float32)[:] = canonical
    angles = folder / "TS_01_bin8.tlt"
    angles.write_text("".join(f"{a:.1f}\n" for a in np.arange(LOW, HIGH + 0.1, 2.0)))
    picks_star = _pytom_star(project / "TS_01_picks.star", scene["centres"] + 0.5,
                             scene["angles"])
    tomogram = Artifact(id="TS_01:tomogram:bin8", kind=ArtifactKind.TOMOGRAM, variant="bin8",
                        path=folder / "TS_01_bin8.zarr", axis_order=AxisOrder.ZYX,
                        pixel_spacing_angstrom=10.0, shape=canonical.shape,
                        parameters={"tilt_series_id": "TS_01", "imod_rec_path": str(rec),
                                    "imod_rec_axis_order": "yzx",
                                    "tilt_angles_path": str(angles), "ctf_corrected": True})
    picks = Artifact(id="TS_01:picks:bin8", kind=ArtifactKind.PICKS, variant="bin8",
                     path=picks_star, parent_ids=[tomogram.id],
                     parameters={"tilt_series_id": "TS_01", "input_tomogram_id": tomogram.id,
                                 "pick_template_box_px": 16, "pytom_low_pass": 25.0})
    registry = ArtifactRegistry.empty()
    registry.extend([tomogram, picks])
    registry.commit(project / "artifacts.json")
    # The map prepare-template cached for the lane (gag-bin8 names EMD-2638).
    (project / "emdb-cache").mkdir()
    (project / "emdb-cache" / "emd_2638.map").write_bytes(b"map")
    text = (ROOT / "configs" / "lanes" / "gag-bin8.toml").read_text()
    text = text.replace('project = "outputs/hiv"', f'project = "{project}"', 1)
    lane = tmp_path / "gag-bin8.toml"
    lane.write_text(text)
    return {"lane": lane, "project": project, "picks": picks_star, "canonical": canonical,
            "count": len(scene["centres"])}


def _import_step(test: Path) -> subprocess.CompletedProcess[str]:
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    cfg = tomllib.loads((test / "lane.toml").read_text())
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_series.py"), "--project",
         str(ROOT / cfg["run"]["project"]), "--config", str(test / "lane.toml"), "--series",
         "TS_01", "--until", "import-tomogram", "--execute"],
        capture_output=True, text=True, env={**os.environ, "PATH": path}, cwd=ROOT)


def test_prepare_writes_a_lane_of_the_source_settings_and_an_imod_turned_volume(
    tmp_path: Path, source: dict[str, Any]
) -> None:
    test = lane_test.prepare(source["lane"], "TS_01", tmp_path / "test")

    cfg = tomllib.loads((test / "lane.toml").read_text())
    gag = tomllib.loads(source["lane"].read_text())
    assert cfg["lane"]["input"] == "tomograms" and cfg["lane"]["variant"] == "bin8"
    assert cfg["data"]["orientation"] == "imod-standard"
    assert cfg["data"]["processing"] == ["ctf-corrected"]
    assert cfg["template"] == gag["template"] and cfg["extract"] == gag["extract"]
    assert cfg["segment"] == gag["segment"]
    assert cfg["pick"] == {key: value for key, value in gag["pick"].items()
                           if key != "ctf_weighting"}
    assert set(cfg["executables"]) == set(lane_test.PYTOM_KEYS)
    stages = {stage["name"]: stage for stage in cfg["run"]["stage"]}
    assert stages["gpu"]["sbatch"] == next(stage["sbatch"] for stage in gag["run"]["stage"]
                                           if stage["name"] == "gpu")
    assert [name for name, _ in run_series.build_steps(cfg, Path("p"), "TS_01")] == [
        "import-tomogram", "segment-tomogram", "prepare-template", "pick-particles",
        "visualize-picks", "extract-particles"]
    record = json.loads((test / "test.json").read_text())
    assert (test / "project" / "emdb-cache" / "emd_2638.map").read_bytes() == b"map"
    assert record["reader"]["orientation"] == "imod-standard"
    assert "clip rotx" in record["reader"]["why"] and record["reader"]["wedge"] == "agrees"
    # IMOD's clip turned tilt's output: z as sections, the mirror of the pipeline's.
    turned = mrcfile.read(test / "data" / "TS_01_rec.mrc")
    np.testing.assert_array_equal(turned, source["canonical"][::-1])
    with pytest.raises(SystemExit, match="never written over"):
        lane_test.prepare(source["lane"], "TS_01", test)
    # Without --out, beside the source project, wherever its folder really is.
    linked = tmp_path / "linked"
    linked.symlink_to(source["project"])
    text = source["lane"].read_text().replace(str(source["project"]), str(linked))
    source["lane"].write_text(text)
    beside = lane_test.prepare(source["lane"], "TS_01", None)
    assert beside == source["project"].resolve().parent / "gag-bin8-tomograms-test"


def test_the_check_finds_the_imported_volume_identical_and_the_picks_again(
    tmp_path: Path, source: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    test = lane_test.prepare(source["lane"], "TS_01", tmp_path / "test")
    assert lane_test.check(test, 2.0) == 2  # nothing has run

    ran = _import_step(test)
    assert ran.returncode == 0, ran.stdout[-2000:] + ran.stderr[-2000:]
    capsys.readouterr()
    assert lane_test.check(test, 2.0) == 0
    shown = capsys.readouterr().out
    assert "volume      identical: largest difference from the source's tomogram 0" in shown
    assert "wedge check: agrees" in shown and "picks       none yet" in shown
    # With a symmetric range the wedge cannot test the side, and the volume decides.
    project = ROOT / tomllib.loads((test / "lane.toml").read_text())["run"]["project"]
    qc_path = project / "imports" / "TS_01" / "tomogram_bin8" / "import_qc.json"
    qc = json.loads(qc_path.read_text())
    qc_path.write_text(json.dumps({**qc, "wedge": {**qc["wedge"],
                                                   "verdict": "side not testable"}}))
    assert lane_test.check(test, 2.0) == 0
    qc_path.write_text(json.dumps(qc))
    capsys.readouterr()

    # What the GPU stage leaves: pytom's picks in the imported tomogram.
    registry = ArtifactRegistry.load(project / "artifacts.json")
    imported = registry.get("TS_01:tomogram:imported:bin8")
    registry.add(Artifact(id="TS_01:picks:pytom:bin8", kind=ArtifactKind.PICKS, variant="bin8",
                          path=source["picks"], parent_ids=[imported.id],
                          parameters={"tilt_series_id": "TS_01",
                                      "input_tomogram_id": imported.id,
                                      "pick_template_box_px": 16}))
    registry.commit(project / "artifacts.json")
    assert lane_test.check(test, 2.0) == 0
    shown = capsys.readouterr().out
    count = source["count"]
    assert f"{count} here, {count} in the source lane; within 2 voxels: {count} of these (100.0%)" \
        in shown


def test_the_other_answer_of_the_mirror_pair_is_told_apart(
    tmp_path: Path, source: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Entered as imod-parallel, past the wedge check, the volume is the source
    reversed in z, and the check fails."""

    test = lane_test.prepare(source["lane"], "TS_01", tmp_path / "test")
    cfg = tomllib.loads((test / "lane.toml").read_text())
    import_tomogram(volume_path=test / "data" / "TS_01_rec.mrc",
                    tilt_angles_path=test / "data" / "TS_01.tlt",
                    orientation_name="imod-parallel", processing="ctf-corrected",
                    tilt_series_id="TS_01", out=ROOT / cfg["run"]["project"], variant="bin8",
                    overrule_wedge_check=True)
    capsys.readouterr()

    assert lane_test.check(test, 2.0) == 1
    shown = capsys.readouterr().out
    assert "volume      the source reversed in z" in shown
    assert "wedge check: mirrored" in shown


def test_a_series_the_source_lane_has_not_picked_is_refused(
    tmp_path: Path, source: dict[str, Any]
) -> None:
    registry = ArtifactRegistry.load(source["project"] / "artifacts.json")
    registry.remove("TS_01:picks:bin8")
    registry.commit(source["project"] / "artifacts.json")
    with pytest.raises(SystemExit, match="0 picks"):
        lane_test.prepare(source["lane"], "TS_01", tmp_path / "test")


def test_a_series_the_source_lane_has_not_reconstructed_is_refused(
    tmp_path: Path, source: dict[str, Any]
) -> None:
    with pytest.raises(SystemExit, match="0 full tomograms of TS_02"):
        lane_test.prepare(source["lane"], "TS_02", tmp_path / "test")
