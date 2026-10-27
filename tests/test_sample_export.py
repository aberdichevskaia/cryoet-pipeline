"""scripts/sample_export.py: a seeded sample of an export, as an export of its own."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sample_export.py"

# Tag: (series, decoy).
PARTICLES = {tag: ("S1" if tag <= 6 else "S2", tag == 4) for tag in range(1, 11)}


def _export(folder: Path) -> list[str]:
    (folder / "particles").mkdir(parents=True)
    lines, entries = [], []
    for tag, (series, decoy) in PARTICLES.items():
        name = f"particles/particle_{tag:05d}.mrc"
        with mrcfile.new(folder / name) as handle:
            handle.set_data(np.full((4, 4, 4), tag, dtype=np.float32))
        row = np.zeros(42)
        row[0], row[2], row[12], row[35] = tag, 1, 1, 13.44
        row[13:15] = [-45, 50]
        lines.append(" ".join(f"{value:g}" for value in row))
        entries.append({"tag": tag, "run": series, "file": name, "decoy": decoy})
    (folder / "initial.tbl").write_text("\n".join(lines) + "\n")
    (folder / "export_manifest.json").write_text(json.dumps({
        "selected_particles": len(entries), "decoy_particles": 1, "box_size_px": 4,
        "wedge": {"ranges": {"S1": [-50.0, 45.0], "S2": [-50.0, 45.0]}}, "particles": entries,
    }))
    (folder / "export_complete.json").write_text(json.dumps(
        {"particles": len(entries), "decoys": 1, "series": ["S1", "S2"]}))
    return lines


def _sample(source: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), str(source), *options],
                          capture_output=True, text=True, check=False)


def test_a_sample_is_an_export_of_the_sources_own_lines_and_files(tmp_path: Path) -> None:
    source = tmp_path / "export"
    lines = _export(source)
    result = _sample(source, "--particles", "5", "--seed", "3")
    assert result.returncode == 0, result.stderr
    out = tmp_path / "export-sample5"
    assert f"5 of 9 particles in {out}" in result.stdout

    manifest = json.loads((out / "export_manifest.json").read_text())
    tags = [entry["tag"] for entry in manifest["particles"]]
    assert len(tags) == 5 and tags == sorted(tags) and 4 not in tags
    assert manifest["selected_particles"] == 5 and manifest["decoy_particles"] == 0
    assert manifest["sample"] == {"from": str(source.resolve()), "particles": 5, "of": 9,
                                  "seed": 3}
    # The source's own table lines, in its order, and its own voxels through links.
    assert (out / "initial.tbl").read_text().splitlines() == [lines[tag - 1] for tag in tags]
    for tag in tags:
        link = out / "particles" / f"particle_{tag:05d}.mrc"
        assert link.is_symlink()
        assert float(mrcfile.read(link).mean()) == tag
    complete = json.loads((out / "export_complete.json").read_text())
    assert complete["particles"] == 5 and complete["split_from"] == str(source.resolve())


def test_the_same_seed_samples_the_same_particles(tmp_path: Path) -> None:
    source = tmp_path / "export"
    _export(source)
    picks = []
    for name, seed in (("a", "11"), ("b", "11"), ("c", "12")):
        result = _sample(source, "--particles", "4", "--seed", seed, "--out", str(tmp_path / name))
        assert result.returncode == 0, result.stderr
        manifest = json.loads((tmp_path / name / "export_manifest.json").read_text())
        picks.append([entry["tag"] for entry in manifest["particles"]])
    assert picks[0] == picks[1]
    assert picks[0] != picks[2]


def test_it_refuses_a_used_folder_and_more_particles_than_there_are(tmp_path: Path) -> None:
    source = tmp_path / "export"
    _export(source)
    used = tmp_path / "used"
    used.mkdir()
    (used / "something").write_text("x")
    result = _sample(source, "--particles", "2", "--out", str(used))
    assert result.returncode != 0 and "not empty" in result.stderr
    result = _sample(source, "--particles", "10")
    assert result.returncode != 0 and "from 1 to 9" in result.stderr
