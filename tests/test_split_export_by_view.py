"""scripts/split_export_by_view.py: an export split by whether the membrane was measured."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "split_export_by_view.py"

# Tilt ranges of the two series, and per tag: series, Dynamo angles (tdrot, tilt, narot),
# decoy, and the group the axis belongs to. The axis is (sin tdrot sin tilt,
# cos tdrot sin tilt, cos tilt); its membrane is seen edge-on at the tilt t where
# ax sin t + az cos t = 0.
RANGES = {"S1": [-40.0, 50.0], "S2": [-29.0, 49.0]}
PARTICLES = {
    1: ("S1", (90.0, 90.0, 0.0), False, "side"),   # axis x: edge-on at 0
    2: ("S1", (0.0, 0.0, 0.0), False, "top"),      # axis along the beam: never
    3: ("S1", (0.0, 90.0, 0.0), False, "side"),    # axis y, the tilt axis: every view
    4: ("S2", (0.0, 20.0, 0.0), False, "top"),     # no x component: never
    5: ("S1", (90.0, 60.0, 0.0), True, "side"),    # edge-on at -30, inside -40..50
    6: ("S2", (90.0, 60.0, 0.0), False, "top"),    # edge-on at -30, outside -29..49
}


def _export(folder: Path) -> list[str]:
    (folder / "particles").mkdir(parents=True)
    lines = []
    entries = []
    for tag, (series, angles, decoy, _) in PARTICLES.items():
        name = f"particles/particle_{tag:05d}.mrc"
        with mrcfile.new(folder / name) as handle:
            handle.set_data(np.full((4, 4, 4), tag, dtype=np.float32))
        row = np.zeros(42)
        row[0], row[2], row[12] = tag, 1, 1
        row[6:9] = angles
        row[13:15] = [-high for high in RANGES[series][::-1]]
        row[19] = 1 if series == "S1" else 2
        row[35] = 13.44
        lines.append(" ".join(f"{value:g}" for value in row))
        entries.append({"tag": tag, "run": series, "file": name, "decoy": decoy})
    (folder / "initial.tbl").write_text("\n".join(lines) + "\n")
    (folder / "export_manifest.json").write_text(json.dumps({
        "selected_particles": len(entries), "decoy_particles": 1, "box_size_px": 4,
        "wedge": {"ranges": RANGES, "axis": "y", "ftype": 1}, "particles": entries,
    }))
    (folder / "export_complete.json").write_text(json.dumps(
        {"particles": len(entries), "decoys": 1, "series": ["S1", "S2"]}))
    return lines


def test_each_particle_goes_where_its_membrane_was_or_was_not_measured(tmp_path: Path) -> None:
    source = tmp_path / "export"
    lines = _export(source)
    result = subprocess.run([sys.executable, str(SCRIPT), str(source)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "side: 2 particles and 1 decoys" in result.stdout
    assert "top: 3 particles in" in result.stdout and "S1 1, S2 2" in result.stdout

    for view in ("side", "top"):
        out = tmp_path / f"export-{view}"
        tags = [tag for tag, (*_, group) in PARTICLES.items() if group == view]
        # The source's own table rows, in its order, and its own voxels through links.
        assert (out / "initial.tbl").read_text().splitlines() == [lines[tag - 1] for tag in tags]
        for tag in tags:
            link = out / "particles" / f"particle_{tag:05d}.mrc"
            assert link.is_symlink()
            assert link.resolve() == (source / link.relative_to(out)).resolve()
            assert float(mrcfile.read(link).mean()) == tag
        assert sorted(path.name for path in (out / "particles").iterdir()) == [
            f"particle_{tag:05d}.mrc" for tag in tags]
        manifest = json.loads((out / "export_manifest.json").read_text())
        assert [entry["tag"] for entry in manifest["particles"]] == tags
        assert manifest["selected_particles"] == len(tags)
        assert manifest["decoy_particles"] == (1 if view == "side" else 0)
        assert manifest["wedge"]["ranges"] == RANGES and manifest["split"]["view"] == view
        complete = json.loads((out / "export_complete.json").read_text())
        assert complete["particles"] == len(tags)
        for name in ("initial.tbl", "export_manifest.json"):
            assert complete[name] == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_a_folder_that_already_holds_files_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "export"
    _export(source)
    (tmp_path / "export-top").mkdir()
    (tmp_path / "export-top" / "old.txt").write_text("an earlier split\n")
    result = subprocess.run([sys.executable, str(SCRIPT), str(source)],
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "exists and is not empty" in result.stderr
    assert not (tmp_path / "export-side").exists()
