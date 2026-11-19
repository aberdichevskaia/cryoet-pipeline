"""scripts/show_imports.py proposes an import's answers from the files, and says where
each comes from (docs/entry_and_exit.md, Order of work item 6).

The rules for a volume's orientation are held to the labels IMOD's own programs
write, where IMOD is installed; the wedge settles the pair those labels cannot, and
contradicts a history the voxels do not bear out. Picks and particles are read with
the imports' readers. A fragment, once its questions are answered, renders as a lane.
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
from test_particles_import import _folder, particles  # noqa: F401  (a fixture)
from test_tomogram_import import SHAPE, wedged_noise

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import show_imports  # noqa: E402

LOW, HIGH = -60.0, 30.0
ANGLES = [float(angle) for angle in np.arange(LOW, HIGH + 0.1, 3.0)]
# As IMOD 5.1.12 and RELION 5.0 wrote them on 17 November 2026, padding and dates kept.
TILT = "TILT: Tomographic reconstruction                        17-Nov-26  11:36:39"
ROTX = "clip: rotx - rotation by -90 around X                   17-Nov-26  11:36:39"
FLIPYZ = "clip: flipyz                                            17-Nov-26  11:36:39"
NEWSTACK = "NEWSTACK: Images copied                                 17-Nov-26  11:36:40"
RELION = "Relion    17-Nov-26  11:37:25"


@pytest.fixture(scope="module")
def canonical() -> np.ndarray:
    return wedged_noise(SHAPE, LOW, HIGH, seed=4)


def _volume(path: Path, data: np.ndarray, labels: list[str], voxel: float = 10.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data, dtype=np.float32))
        mrc.voxel_size = voxel
        for index, label in enumerate(labels):
            mrc.header.label[index] = label.encode().ljust(80)[:80]
        mrc.header.nlabl = len(labels)
    return path


def _angles(path: Path, angles: list[float] = ANGLES) -> Path:
    path.write_text("".join(f"{angle:.2f}\n" for angle in angles))
    return path


def _answers(reading: show_imports.Reading) -> dict[str, Any]:
    return {answer.key: ("ASK" if answer.asked else answer.value) for answer in reading.answers}


# --- A volume's orientation -----------------------------------------------------------


@pytest.mark.parametrize(("labels", "thin", "answer"), [
    ([TILT], "y", "imod-tilt"),
    ([TILT, ROTX], "z", "imod-standard"),
    ([TILT, FLIPYZ], "z", "imod-parallel"),
    ([TILT, NEWSTACK, ROTX], "z", "imod-standard"),  # trimvol -rx
    ([TILT, NEWSTACK, FLIPYZ], "z", "imod-parallel"),  # trimvol -yz
    ([TILT], "z", "imod-parallel|imod-standard?"),  # -PARALLEL or -RotateBy90
    ([RELION], "z", "relion5"),
    ([TILT, "clip: flipz"], "z", None),
    ([TILT, RELION], "z", None),
    (["Created by mrcfile.py"], "z", None),
])
def test_the_history_gives_the_orientation_its_programs_wrote(labels, thin, answer):
    clean = [" ".join(label.split()) for label in labels]
    assert show_imports.history(clean, thin)[0] == answer


def _imod(name: str) -> str | None:
    return shutil.which(name)


@pytest.mark.skipif(_imod("tilt") is None or _imod("clip") is None or _imod("trimvol") is None,
                    reason="IMOD is not installed")
def test_the_rules_hold_for_the_volumes_imod_writes(tmp_path: Path) -> None:
    """IMOD's tilt, clip and trimvol on one small series: every way it writes the
    volume is read as the answer on the list for it."""

    rng = np.random.default_rng(1)
    stack = tmp_path / "ts.st"
    with mrcfile.new(stack) as mrc:
        mrc.set_data(rng.normal(size=(11, 64, 64)).astype(np.float32))
        mrc.voxel_size = 10.0
    _angles(tmp_path / "ts.tlt", [float(a) for a in range(-50, 51, 10)])
    base = ["tilt", "-InputProjections", "ts.st", "-TILTFILE", "ts.tlt", "-THICKNESS", "24"]
    runs = {
        "raw.rec": [*base, "-OutputFile", "raw.rec"],
        "par.rec": [*base, "-OutputFile", "par.rec", "-PARALLEL"],
        "r90.rec": [*base, "-OutputFile", "r90.rec", "-RotateBy90"],
        "rotx.rec": ["clip", "rotx", "raw.rec", "rotx.rec"],
        "flipyz.rec": ["clip", "flipyz", "raw.rec", "flipyz.rec"],
        "trimrx.rec": ["trimvol", "-rx", "raw.rec", "trimrx.rec"],
    }
    for command in runs.values():
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)

    read = {name: show_imports.read_volume(tmp_path / name, wedge=False) for name in runs}

    assert {name: item["history"]["orientation"] for name, item in read.items()} == {
        "raw.rec": "imod-tilt", "par.rec": "imod-parallel|imod-standard?",
        "r90.rec": "imod-parallel|imod-standard?", "rotx.rec": "imod-standard",
        "flipyz.rec": "imod-parallel", "trimrx.rec": "imod-standard"}
    assert read["raw.rec"]["thin"] == "y" and read["rotx.rec"]["thin"] == "z"


def test_the_wedge_splits_the_pair_the_history_cannot(tmp_path: Path, canonical: np.ndarray):
    # tilt -PARALLEL keeps the pipeline's voxels; -RotateBy90 reverses them in z.
    _volume(tmp_path / "TS_01_rec.mrc", canonical, [TILT])
    _volume(tmp_path / "TS_02_rec.mrc", canonical[::-1], [TILT])
    for series in ("TS_01", "TS_02"):
        _angles(tmp_path / f"{series}.tlt")

    reading = show_imports.read_tomograms([tmp_path], wedge=True, max_volumes=20)

    assert [item["orientation"] for item in reading.items] == ["imod-parallel", "imod-standard"]
    assert [item["wedge"]["verdict"] for item in reading.items] == ["agrees", "mirrored"]
    answers = _answers(reading)
    assert answers["orientation"] == "ASK"
    assert any("a lane takes one" in note for note in reading.notes)
    assert answers["tomogram"] == show_imports.lane_path(tmp_path) + "/{series}_rec.mrc"
    assert answers["tilt_angles"] == show_imports.lane_path(tmp_path) + "/{series}.tlt"
    assert reading.series == ["TS_01", "TS_02"]


def test_a_history_the_wedge_contradicts_is_not_taken(tmp_path: Path, canonical: np.ndarray):
    # Labelled clip rotx, but holding the voxels as the pipeline keeps them.
    _volume(tmp_path / "TS_01_rec.mrc", canonical, [TILT, ROTX])
    _angles(tmp_path / "TS_01.tlt")

    (item,) = show_imports.read_tomograms([tmp_path], wedge=True, max_volumes=20).items

    assert item["history"]["orientation"] == "imod-standard"
    assert item["orientation"] is None
    assert "the wedge has the other side" in item["orientation_from"]


def test_a_symmetric_range_leaves_the_pair_asked_and_says_why(tmp_path: Path) -> None:
    volume = wedged_noise(SHAPE, -45.0, 45.0, seed=5)
    _volume(tmp_path / "TS_01_rec.mrc", volume, [TILT])
    _angles(tmp_path / "TS_01.tlt", [float(a) for a in np.arange(-45, 45.1, 3)])

    (item,) = show_imports.read_tomograms([tmp_path], wedge=True, max_volumes=20).items

    assert item["orientation"] is None
    assert "side not testable" in item["orientation_from"]


def test_the_angles_come_from_tilt_com_and_relions_tomograms_star(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    etomo = tmp_path / "etomo" / "TS_01"
    _volume(etomo / "TS_01_rec.mrc", canonical, [TILT, ROTX])
    _angles(etomo / "final.tlt")
    _angles(etomo / "TS_01.tlt", [0.0, 1.0])
    (etomo / "tilt.com").write_text("$tilt -StandardInput\nInputProjections TS_01.ali\n"
                                    "TILTFILE final.tlt\nTHICKNESS 48\n")
    item = show_imports.read_volume(etomo / "TS_01_rec.mrc", wedge=False)
    assert item["angles"].endswith("etomo/TS_01/final.tlt")
    assert "TILTFILE of tilt.com" in item["angles_found"]
    assert show_imports.template_of(etomo / "TS_01_rec.mrc", "TS_01").endswith(
        "etomo/{series}/{series}_rec.mrc")

    project = tmp_path / "relion"
    job = project / "Tomograms" / "job007"
    _volume(job / "tomograms" / "rec_TS_01.mrc", canonical[::-1], [RELION])
    series_star = project / "AlignTiltSeries" / "job005" / "tilt_series" / "TS_01.star"
    series_star.parent.mkdir(parents=True)
    series_star.write_text("data_TS_01\n\nloop_\n_rlnMicrographName #1\n_rlnTomoYTilt #2\n"
                           + "".join(f"view{i}.mrc {a:.2f}\n" for i, a in enumerate(ANGLES)))
    (job / "tomograms.star").write_text(
        "data_global\n\nloop_\n_rlnTomoName #1\n_rlnTomoTiltSeriesStarFile #2\n"
        "TS_01 AlignTiltSeries/job005/tilt_series/TS_01.star\n")

    item = show_imports.read_volume(job / "tomograms" / "rec_TS_01.mrc", wedge=True)

    assert item["series"] == "TS_01" and item["angle_count"] == len(ANGLES)
    assert item["orientation"] == "relion5"
    assert item["wedge"]["verdict"] == "agrees"


def test_a_fragment_once_answered_renders_as_a_lane(tmp_path: Path, canonical: np.ndarray):
    for series in ("TS_01", "TS_02"):
        _volume(tmp_path / f"{series}_rec.mrc", canonical, [TILT, FLIPYZ])
        _angles(tmp_path / f"{series}.tlt")
    reading = show_imports.read_tomograms([tmp_path], wedge=True, max_volumes=20)
    text = show_imports.fragment(reading)
    assert '# processing = ?' in text and 'orientation = "imod-parallel"' in text

    config = tomllib.loads(text.replace("# processing = ?", 'processing = "raw" #'))
    config.update(template={"emdb_id": "EMD-1234", "low_pass_angstrom": 20.0},
                  pick={"particle_diameter_angstrom": 250.0, "low_pass_angstrom": 30.0,
                        "gpu_ids": "0"},
                  extract={"box_size_px": 48},
                  executables={key: "x" for key in ("pytom_create_template", "pytom_create_mask",
                                                    "pytom_match_template",
                                                    "pytom_extract_candidates")})
    config["lane"]["variant"] = "bin8"
    command = dict(run_series.build_steps(config, Path("p"), "TS_02"))["import-tomogram"]
    assert command[command.index("--volume") + 1] == str(
        show_imports.lane_path(tmp_path / "TS_02_rec.mrc"))
    assert command[command.index("--orientation") + 1] == "imod-parallel"


# As IMOD 5.1.12's ctfphaseflip wrote it on 19 November 2026, here and in Munich.
CTF = "ctfPhaseFlip: CTF correction with phase flipping only   19-Nov-26  12:20:17"


@pytest.mark.parametrize(("labels", "steps"), [
    ([NEWSTACK, CTF, TILT, ROTX], ["ctf-corrected"]),
    ([NEWSTACK, TILT, ROTX], None),
    ([TILT, CTF], None),  # after the reconstruction: not on the views tilt read
    ([CTF], None),  # a stack's history, no reconstruction in it
])
def test_the_history_shows_ctfphaseflip_only_before_tilt(labels, steps):
    clean = [" ".join(label.split()) for label in labels]
    assert show_imports.processing_seen(clean)[0] == steps


def test_ctf_corrected_is_proposed_from_the_history_and_still_asked(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    for series in ("TS_01", "TS_02"):
        _volume(tmp_path / f"{series}_rec.mrc", canonical[::-1], [NEWSTACK, CTF, TILT, ROTX])
        _angles(tmp_path / f"{series}.tlt")

    reading = show_imports.read_tomograms([tmp_path], wedge=False, max_volumes=20)

    (answer,) = [one for one in reading.answers if one.key == "processing"]
    assert answer.asked and answer.value == ["ctf-corrected"]
    assert "a denoising or a wedge filling" in answer.question
    # Kept by uncommenting the line; the question stays a comment on it.
    line = next(text for text in show_imports.fragment(reading).splitlines()
                if text.startswith("# processing = "))
    assert tomllib.loads(line[2:])["processing"] == ["ctf-corrected"]

    # One volume without the label: nothing is proposed, and the count is said.
    _volume(tmp_path / "TS_02_rec.mrc", canonical[::-1], [NEWSTACK, TILT, ROTX])
    reading = show_imports.read_tomograms([tmp_path], wedge=False, max_volumes=20)
    (answer,) = [one for one in reading.answers if one.key == "processing"]
    assert answer.value is None and "in the history of 1 of 2 volumes" in answer.question


@pytest.mark.skipif(any(_imod(name) is None for name in ("ctfphaseflip", "tilt", "clip")),
                    reason="IMOD is not installed")
def test_imods_ctfphaseflip_then_tilt_leave_the_history_the_reader_reads(tmp_path: Path):
    """As etomo runs them: the stack phase-flipped, reconstructed, turned by clip rotx."""

    rng = np.random.default_rng(1)
    with mrcfile.new(tmp_path / "ts.st") as mrc:
        mrc.set_data(rng.normal(size=(11, 64, 64)).astype(np.float32))
        mrc.voxel_size = 10.0
    angles = [float(a) for a in range(-50, 51, 10)]
    _angles(tmp_path / "ts.tlt", angles)
    # IMOD's defocus file, version 2: views, angles and defocus in nm per line.
    (tmp_path / "ts.defocus").write_text("".join(
        f"{view} {view} {angle:.1f} {angle:.1f} 3000.0" + (" 2" if view == 1 else "") + "\n"
        for view, angle in enumerate(angles, start=1)))
    for command in (
        ["ctfphaseflip", "-InputStack", "ts.st", "-OutputFileName", "ctf.st", "-AngleFile",
         "ts.tlt", "-DefocusFile", "ts.defocus", "-Voltage", "300", "-SphericalAberration",
         "2.7", "-AmplitudeContrast", "0.07", "-PixelSize", "1.0", "-DefocusTol", "200",
         "-InterpolationWidth", "15"],
        ["tilt", "-InputProjections", "ctf.st", "-OutputFile", "ctf.rec", "-TILTFILE", "ts.tlt",
         "-THICKNESS", "24"],
        ["clip", "rotx", "ctf.rec", "TS_01_rec.mrc"],
    ):
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)

    item = show_imports.read_volume(tmp_path / "TS_01_rec.mrc", wedge=False)

    assert item["processing_seen"]["steps"] == ["ctf-corrected"]
    assert item["history"]["orientation"] == "imod-standard"


# --- Picks ---------------------------------------------------------------------------


def _star(path: Path, columns: list[str], rows: list[list[Any]], block: str = "") -> Path:
    path.write_text(f"data_{block}\n\nloop_\n"
                    + "".join(f"_{name} #{i}\n" for i, name in enumerate(columns, start=1))
                    + "".join(" ".join(str(value) for value in row) + "\n" for row in rows))
    return path


PYTOM = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt",
         "rlnAnglePsi", "rlnLCCmax", "rlnCutOff", "rlnSearchStd", "rlnDetectorPixelSize",
         "rlnMicrographName"]


@pytest.mark.parametrize(("columns", "row", "format_name"), [
    (PYTOM, [50, 40, 20, 1, 2, 3, 0.4, 0.2, 0.01, 10.0, "TS_01.mrc"], "pytom-star"),
    (["rlnTomoName", "rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst",
      "rlnCenteredCoordinateZAngst"], ["TS_01", 1.0, 2.0, 3.0], "relion5-star"),
    (["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"],
     ["TS_01.tomostar", 1, 2, 3], "warp-star"),
    (["rlnMicrographName", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ",
      "rlnImageName"], ["TS_01.mrc", 1, 2, 3, "sub/1.mrc"], "relion3-star"),
    (["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"], [1, 2, 3], None),
])
def test_a_star_files_columns_say_its_format(tmp_path, columns, row, format_name):
    star = _star(tmp_path / "picks.star", columns, [row])
    reading = show_imports.read_picks(star, None)
    assert reading.items[0]["format"] == format_name
    if format_name in ("warp-star", "relion3-star"):
        assert reading.refusals and "is not on the list yet" in reading.refusals[0]
    elif format_name is None:
        assert _answers(reading)["picks_format"] == "ASK"


def test_pytoms_box_comes_from_its_job_file_and_the_picks_meet_their_volume(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    volume = _volume(tmp_path / "TS_01_rec.mrc", canonical, [TILT, FLIPYZ])
    rows = [[50, 40, 20, 1, 2, 3, 0.4, 0.2, 0.01, 10.0, "TS_01_rec.mrc"],
            [60, 50, 25, 4, 5, 6, 0.3, 0.2, 0.01, 10.0, "TS_01_rec.mrc"]]
    star = _star(tmp_path / "TS_01_rec_particles.star", PYTOM, rows)
    (tmp_path / "TS_01_rec_job.json").write_text(json.dumps({"template_shape": [36, 36, 36]}))

    reading = show_imports.read_picks(star, volume)

    answers = _answers(reading)
    assert answers["picks_format"] == "pytom-star" and answers["template_box_px"] == 36
    assert answers["picks"] == show_imports.lane_path(tmp_path) + "/{series}_rec_particles.star"
    assert any("lie inside" in note for note in reading.notes)
    # The same picks, said to be at 4 times the volume's pixel, reach past it.
    far = _star(tmp_path / "far.star", PYTOM, [[row[0] * 4, *row[1:]] for row in rows])
    (tmp_path / "TS_01_rec_job.json").unlink()
    reading = show_imports.read_picks(far, volume)
    assert any(f"reach {240 / SHAPE[2]:.2f} times" in note for note in reading.notes), reading.notes
    assert _answers(reading)["template_box_px"] == "ASK"


def test_a_dynamo_table_of_several_tomograms_asks_each_ones_series(tmp_path: Path) -> None:
    rows = np.zeros((5, 35))
    rows[:, 0] = np.arange(1, 6)
    rows[:, 19] = [1, 1, 1, 4, 4]
    rows[:, 23:26] = 30.5
    table = tmp_path / "picks.tbl"
    table.write_text("\n".join(" ".join(f"{v:g}" for v in row) for row in rows) + "\n")

    reading = show_imports.read_picks(table, None)

    assert reading.items[0]["tomograms"] == {1: 3, 4: 2}
    assert _answers(reading)["dynamo_tomograms"] == "ASK"
    assert any("every pose in the table is zero" in note for note in reading.notes)


def test_a_csvs_header_names_its_columns_and_its_units_are_asked(tmp_path: Path) -> None:
    table = tmp_path / "TS_01.csv"
    table.write_text("x,y,z,score\n10,20,30,0.5\n11,21,31,0.4\n")
    answers = _answers(show_imports.read_picks(table, None))
    assert answers["csv_columns"] == ["x", "y", "z", "score"]
    assert answers["coordinate_units"] == "ASK" and answers["coordinate_origin"] == "ASK"


def test_relions_tomograms_star_near_the_picks_answers_the_tilt_series_pixel(tmp_path: Path):
    star = _star(tmp_path / "particles.star",
                 ["rlnTomoName", "rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst",
                  "rlnCenteredCoordinateZAngst"], [["TS_01", 1.0, 2.0, 3.0],
                                                    ["TS_02", 4.0, 5.0, 6.0]], block="particles")
    answers = _answers(show_imports.read_picks(star, None))
    assert answers["relion_tilt_series_pixel_size_angstrom"] == "ASK"
    _star(tmp_path / "tomograms.star", ["rlnTomoName", "rlnTomoTiltSeriesPixelSize"],
          [["TS_01", 1.35], ["TS_02", 1.35]], block="global")
    reading = show_imports.read_picks(star, None)
    assert _answers(reading)["relion_tomograms_star"].endswith("tomograms.star")
    assert reading.series == ["TS_01", "TS_02"]


# --- Particles -----------------------------------------------------------------------


def test_a_dynamo_folder_gives_its_box_pixel_and_wedges_and_asks_the_series(
    tmp_path: Path, particles: list[np.ndarray]  # noqa: F811
) -> None:
    folder, table, _ = _folder(tmp_path, particles)

    reading = show_imports.read_particles(folder, None)

    assert not reading.refusals
    answers = _answers(reading)
    assert answers["particles_table"] == show_imports.lane_path(table)
    assert answers["particles_series"] == "ASK"
    item = reading.items[0]
    assert item["box"] == [32, 32, 32] and item["pixel"] == [5.4]
    assert {number: entry["wedge"]["verdict"] for number, entry in item["tomograms"].items()} \
        == {1: "agrees", 2: "agrees"}


def test_a_table_whose_wedge_has_the_other_side_is_refused(
    tmp_path: Path, particles: list[np.ndarray]  # noqa: F811
) -> None:
    folder, table, _ = _folder(tmp_path, particles, ytilt=(-50.0, 60.0))
    reading = show_imports.read_particles(folder, table)
    assert any("the other side of the table's" in line for line in reading.refusals)


def test_the_command_line_exits_as_its_docstring_says(
    tmp_path: Path, particles: list[np.ndarray]  # noqa: F811
) -> None:
    folder, _, _ = _folder(tmp_path, particles)
    script = [sys.executable, str(ROOT / "scripts" / "show_imports.py")]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    shown = subprocess.run([*script, "particles", str(folder)], capture_output=True, text=True,
                           env=env)
    assert shown.returncode == 0, shown.stderr
    assert "Lane input: particles. Config fragment:" in shown.stdout
    as_json = json.loads(subprocess.run([*script, "particles", str(folder), "--json"],
                                        capture_output=True, text=True, env=env).stdout)
    assert as_json["kind"] == "particles" and "[data]" in as_json["fragment"]
    star = _star(tmp_path / "warp.star", ["rlnMicrographName", "rlnCoordinateX",
                                          "rlnCoordinateY", "rlnCoordinateZ"],
                 [["a.tomostar", 1, 2, 3]])
    refused = subprocess.run([*script, "picks", str(star)], capture_output=True, text=True,
                             env=env)
    assert refused.returncode == 1 and "Refused by the import" in refused.stdout
    missing = subprocess.run([*script, "picks", str(tmp_path / "none.star")],
                             capture_output=True, text=True, env=env)
    assert missing.returncode == 2


def test_a_per_series_folder_names_the_series_and_the_volumes_own_tlt_comes_first(
    tmp_path: Path, canonical: np.ndarray
) -> None:
    # As reconstruct-tomogram keeps them: tomograms/<TS>/<TS>_fine_bin8.rec and its .tlt.
    folder = tmp_path / "tomograms" / "TS_01"
    _volume(folder / "TS_01_fine_bin8.rec", canonical.transpose(1, 0, 2), [TILT])
    _angles(folder / "TS_01_fine_bin8.tlt")
    _angles(folder / "TS_01_coarse_bin16.tlt", [0.0, 1.0])

    reading = show_imports.read_tomograms([folder / "TS_01_fine_bin8.rec"], wedge=True,
                                          max_volumes=20)

    (item,) = reading.items
    assert item["series"] == "TS_01" and item["orientation"] == "imod-tilt"
    assert item["angles_found"] == "the .tlt of the volume's name beside it"
    answers = _answers(reading)
    assert answers["tomogram"].endswith("tomograms/{series}/{series}_fine_bin8.rec")
    assert answers["tilt_angles"].endswith("tomograms/{series}/{series}_fine_bin8.tlt")
