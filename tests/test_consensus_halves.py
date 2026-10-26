"""scripts/consensus_halves.py: two halves of a consensus class, their start and their FSC."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from test_classification_report import write_em

from cryoet_pipeline.filters import low_pass_filter_volume
from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "consensus_halves.py"
BOX = 16
SPACING = 13.44
SERIES = {tag: "S1" if tag <= 6 else "S2" for tag in range(1, 13)}
DECOYS = {12}
# Class 1 of both runs holds 2-5 and 7-9, and the decoy 12, which is left out.
POSES_RUN = {**{tag: 1 for tag in (1, 2, 3, 4, 5, 7, 8, 9, 12)}, **{tag: 2 for tag in (6, 10, 11)}}
AGREE_RUN = {**{tag: 1 for tag in (2, 3, 4, 5, 7, 8, 9, 10, 12)}, **{tag: 2 for tag in (1, 6, 11)}}
CONSENSUS = [2, 3, 4, 5, 7, 8, 9]


def _row(tag: int) -> list[float]:
    row = [0.0] * 42
    row[0], row[1], row[2] = tag, 1, 1
    row[3:9] = [0.5, -0.5, 1.0, 11.0 * tag, 7.0 * tag, 3.0 * tag]
    row[9] = 0.3
    row[12:15] = [1, -60, 60]
    row[19] = 1 if SERIES[tag] == "S1" else 2
    row[23:26] = [100.0 + tag, 200.0, 300.0]
    row[35] = SPACING
    return row


def _line(row: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in row)


def _refined_pose(tag: int, channel: int) -> list[float]:
    """The pose a run's channel gave the particle: unlike the export's and the other channel's."""

    return [0.25 * channel, -0.125 * tag, 0.5, 40.0 + tag, 30.0 + 2 * tag + channel, 15.0 * channel]


def _export(folder: Path) -> list[str]:
    (folder / "particles").mkdir(parents=True)
    entries, lines = [], []
    for tag in SERIES:
        name = f"particles/particle_{tag:05d}.mrc"
        with mrcfile.new(folder / name) as handle:
            handle.set_data(np.full((BOX,) * 3, tag, dtype=np.float32))
        lines.append(_line(_row(tag)))
        entries.append({"tag": tag, "run": SERIES[tag], "file": name, "decoy": tag in DECOYS})
    (folder / "initial.tbl").write_text("\n".join(lines) + "\n")
    (folder / "export_manifest.json").write_text(json.dumps({
        "selected_particles": 12, "decoy_particles": 1, "box_size_px": BOX,
        "pixel_spacing_angstrom": SPACING,
        "wedge": {"ranges": {"S1": [-60.0, 60.0], "S2": [-60.0, 60.0]}}, "particles": entries,
    }))
    (folder / "export_complete.json").write_text(json.dumps(
        {"particles": 12, "decoys": 1, "series": ["S1", "S2"]}))
    return lines


def _write_table(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_line(row) for row in rows) + "\n")


def _project(folder: Path, starting: list[list[float]], refined: dict[int, list[list[float]]],
             averages: dict[int, np.ndarray], *, mra: bool, mask: np.ndarray | None = None,
             template: str | None = None) -> None:
    """One finished iteration of a Dynamo project, as the report's reader reads it; `template`
    is its file_template_initial, relative to the folder that holds the project."""

    channels = sorted(refined)
    for channel in channels:
        write_em(folder / "settings" / f"template_initial_ref_{channel:03d}.em",
                 np.zeros((BOX,) * 3, dtype=np.float32))
        _write_table(folder / "results" / "ite_0001" / "starting_values"
                     / f"starting_table_ref_{channel:03d}_ite_0001.tbl", starting)
        _write_table(folder / "results" / "ite_0001" / "averages"
                     / f"refined_table_ref_{channel:03d}_ite_0001.tbl", refined[channel])
        write_em(folder / "results" / "ite_0001" / "averages"
                 / f"average_ref_{channel:03d}_ite_0001.em", averages[channel])
    write_em(folder / "settings" / "mask.em",
             np.ones((BOX,) * 3, dtype=np.float32) if mask is None else mask)
    for number in (1, 2):
        (folder / "results" / f"ite_{number:04d}").mkdir(parents=True, exist_ok=True)
        (folder / "results" / f"ite_{number:04d}"
         / f"currently_surviving_references_ite_{number:04d}.txt").write_text(
            " ".join(map(str, channels)) + "\n")
    card = folder / "cards" / "ite_0001" / "card_ite_ite_0001.card"
    card.parent.mkdir(parents=True)
    card.write_text(f"nref={len(channels)};\nmra={int(mra)};\n"
                    + (f"file_template_initial={template};\n" if template else ""))


def _mra(export: Path, name: str, classes: dict[int, int], average: np.ndarray) -> Path:
    starting = [_row(tag) for tag in SERIES]
    refined = {}
    for channel in (1, 2):
        rows = []
        for tag in SERIES:
            row = _row(tag)
            row[3:9] = _refined_pose(tag, channel)
            row[2] = int(classes[tag] == channel)
            row[9] = 0.1 * channel
            row[33] = classes[tag]
            rows.append(row)
        refined[channel] = rows
    _project(export / name, starting, refined, {1: average, 2: -average}, mra=True)
    return export / name


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)],
                          capture_output=True, text=True, check=False)


def _split(tmp_path: Path, out_name: str = "consensus") -> tuple[Path, list[str], np.ndarray,
                                                                 subprocess.CompletedProcess[str]]:
    export = tmp_path / "export"
    lines = _export(export)
    rng = np.random.default_rng(5)
    average = rng.standard_normal((BOX,) * 3).astype(np.float32)
    _mra(export, "mra_p", POSES_RUN, average)
    _mra(export, "mra_a", AGREE_RUN, rng.standard_normal((BOX,) * 3).astype(np.float32))
    result = _run("split", f"{export / 'mra_p'}:1", "--agree", f"{export / 'mra_a'}:1",
                  "--start-lowpass", 60, "--out", tmp_path / out_name)
    return export, lines, average, result


def test_the_halves_split_the_consensus_with_the_first_runs_poses(tmp_path: Path) -> None:
    export, lines, average, result = _split(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "mra_p class 1 after iteration 1: 8 particles (decoys left out)" in result.stdout
    assert "in both: 7" in result.stdout

    halves = {}
    for half in (1, 2):
        folder = tmp_path / f"consensus-half{half}"
        table = (folder / "initial.tbl").read_text().splitlines()
        tags = [int(float(line.split()[0])) for line in table]
        halves[half] = tags
        for line, tag in zip(table, tags, strict=True):
            tokens, source = line.split(), lines[tag - 1].split()
            # The poses run's channel 1 pose; every other column the export's own.
            assert [float(token) for token in tokens[3:9]] == _refined_pose(tag, 1)
            assert tokens[:3] + tokens[9:] == source[:3] + source[9:]
            link = folder / "particles" / f"particle_{tag:05d}.mrc"
            assert link.is_symlink() and float(mrcfile.read(link).mean()) == tag
        manifest = json.loads((folder / "export_manifest.json").read_text())
        assert [entry["tag"] for entry in manifest["particles"]] == tags
        record = manifest["consensus_half"]
        assert record["half"] == half and record["start_lowpass_angstrom"] == 60
        assert record["other_half"] == str((tmp_path / f"consensus-half{3 - half}").resolve())
        complete = json.loads((folder / "export_complete.json").read_text())
        for name in ("initial.tbl", "export_manifest.json"):
            assert complete[name] == hashlib.sha256((folder / name).read_bytes()).hexdigest()
    assert sorted(halves[1] + halves[2]) == CONSENSUS
    for name in ("S1", "S2"):
        counts = [sum(SERIES[tag] == name for tag in halves[half]) for half in (1, 2)]
        assert abs(counts[0] - counts[1]) <= 1
    assert abs(len(halves[1]) - len(halves[2])) <= 1

    # One start for both: the class average low-passed, in the particles' axis order.
    expected = low_pass_filter_volume(torch.from_numpy(average), cutoff_angstrom=60,
                                      pixel_spacing_angstrom=SPACING).numpy()
    starts = [tmp_path / f"consensus-half{half}" / "start_reference.mrc" for half in (1, 2)]
    assert starts[0].read_bytes() == starts[1].read_bytes()
    with mrcfile.open(starts[0]) as handle:
        assert np.allclose(handle.data, expected, atol=1e-6)
        assert float(handle.voxel_size.x) == pytest.approx(SPACING)

    # The same seed deals the same halves.
    again = _run("split", f"{export / 'mra_p'}:1", "--agree", f"{export / 'mra_a'}:1",
                 "--start-lowpass", 60, "--out", tmp_path / "again")
    assert again.returncode == 0, again.stderr
    for half in (1, 2):
        assert ((tmp_path / f"again-half{half}" / "initial.tbl").read_text()
                == (tmp_path / f"consensus-half{half}" / "initial.tbl").read_text())


def test_split_refuses_a_used_folder_another_export_and_a_start_past_nyquist(
        tmp_path: Path) -> None:
    export, _, _, result = _split(tmp_path)
    assert result.returncode == 0, result.stderr
    used = _run("split", f"{export / 'mra_p'}:1", "--agree", f"{export / 'mra_a'}:1",
                "--start-lowpass", 60, "--out", tmp_path / "consensus")
    assert used.returncode != 0 and "exists and is not empty" in used.stderr
    other = tmp_path / "other"
    _export(other)
    _mra(other, "mra_o", AGREE_RUN, np.zeros((BOX,) * 3, dtype=np.float32))
    elsewhere = _run("split", f"{export / 'mra_p'}:1", "--agree", f"{other / 'mra_o'}:1",
                     "--start-lowpass", 60, "--out", tmp_path / "x")
    assert elsewhere.returncode != 0 and "both runs must be on the export" in elsewhere.stderr
    fine = _run("split", f"{export / 'mra_p'}:1", "--agree", f"{export / 'mra_a'}:1",
                "--start-lowpass", 20, "--out", tmp_path / "y")
    assert fine.returncode != 0 and "not coarser than Nyquist" in fine.stderr
    assert not (tmp_path / "y-half1").exists()


def _half_project(folder: Path, name: str, turn: float, average: np.ndarray,
                  mask: np.ndarray) -> Path:
    """A finished one-channel refinement of a half: every particle turned by `turn` about
    its own z axis (tdrot) and shifted by (0.3, 0.4, 0) px from its start."""

    starting = [[float(token) for token in line.split()]
                for line in (folder / "initial.tbl").read_text().splitlines()]
    refined = []
    for row in starting:
        moved = list(row)
        moved[3:6] = [row[3] + 0.3, row[4] + 0.4, row[5]]
        moved[6] = row[6] + turn
        refined.append(moved)
    # As dynamo_mra_setup.m leaves a project of one map: the seed file named, not its folder.
    seed = f"{name}_seeds/template_initial_ref_001.em"
    write_em(folder / seed, average)
    (folder / f"{name}_seeds" / "seed_kinds.txt").write_text("1 map start_reference.mrc\n")
    _project(folder / name, starting, {1: refined}, {1: average}, mra=False, mask=mask,
             template=seed)
    return folder / name


def test_maps_writes_the_halves_and_their_fsc(tmp_path: Path) -> None:
    _, _, _, result = _split(tmp_path)
    assert result.returncode == 0, result.stderr
    rng = np.random.default_rng(9)
    shared = rng.standard_normal((BOX,) * 3).astype(np.float32)
    maps = {half: (shared + 0.3 * rng.standard_normal((BOX,) * 3)).astype(np.float32)
            for half in (1, 2)}
    z, y, x = np.indices((BOX,) * 3) - BOX / 2
    mask = (np.sqrt(x * x + y * y + z * z) < 6).astype(np.float32)
    projects = [_half_project(tmp_path / f"consensus-half{half}", f"refine_h{half}", 5.0,
                              maps[half], mask) for half in (1, 2)]

    # Given in the other order, the halves still come out by their own numbers.
    out = tmp_path / "maps" / "consensus"
    result = _run("maps", projects[1], projects[0], "--out", out)
    assert result.returncode == 0, result.stderr
    for half in (1, 2):
        with mrcfile.open(tmp_path / "maps" / f"consensus_half{half}.mrc") as handle:
            assert np.array_equal(handle.data, maps[half])
            assert float(handle.voxel_size.x) == pytest.approx(SPACING)
    assert np.allclose(mrcfile.read(tmp_path / "maps" / "consensus_mean.mrc"),
                       (maps[1] + maps[2]) / 2)
    assert np.array_equal(mrcfile.read(tmp_path / "maps" / "consensus_mask.mrc"), mask)
    assert "turned median 5.0 deg" in result.stdout and "shifted median 0.50 px" in result.stdout
    assert "FSC unmasked : 0.5 never crossed, 0.143 never crossed" in result.stdout
    assert "FSC masked   : 0.5 " in result.stdout
    assert "fsc_phase_randomization.py" in result.stdout
    assert "--randomize-beyond-angstrom 60 40.32" in result.stdout
    report = json.loads((tmp_path / "maps" / "consensus_fsc.json").read_text())
    # Two noisy copies of one map agree at every shell, well above 0.5.
    assert min(report["unmasked"]["correlation"]) > 0.5

    run = read_dynamo_classification(projects[0], compute_initial_average=False)
    assert run.software == "Dynamo project" and run.reference_notes == {
        1: "the map start_reference.mrc"}

    same = _run("maps", projects[0], projects[0], "--out", tmp_path / "maps" / "same")
    assert same.returncode != 0 and "not the two halves of one split" in same.stderr
