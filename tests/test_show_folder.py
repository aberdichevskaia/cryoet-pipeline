"""scripts/show_folder.py: a data folder read the way a lane will read it, on synthetic folders."""

import json
import os
import sys
import tomllib
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import tifffile

from cryoet_pipeline.movies import inspect_movie

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import show_folder  # noqa: E402

# The first local TIFF dataset's naming: signed angles with "+", zero as "-00.0",
# series ids with leading zeros, the tilt number in angle order.
PATTERN = r"tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif"
NO_PLUS = r"tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>-?\d+(?:\.\d+)?)\.tif"
SIGNED = ["-06.0", "-03.0", "-00.0", "+03.0", "+06.0"]
# Acquisition order of a dose-symmetric scheme, as an mdoc lists it.
DOSE_SYMMETRIC = [0.0, 3.0, -3.0, -6.0, 6.0]


def _tiff(path: Path, *, frames: int = 3, pixel: float | None = 0.84) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((frames, 6, 8), np.uint16) if frames > 1 else np.zeros((6, 8), np.uint16)
    extra = {}
    if pixel is not None:
        extra = {"resolution": (1e8 / pixel, 1e8 / pixel), "resolutionunit": "CENTIMETER"}
    tifffile.imwrite(path, data, photometric="minisblack", **extra)


def _mrc(path: Path, shape: tuple[int, ...], pixel: float, dtype=np.int8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.zeros(shape, dtype))
        mrc.voxel_size = pixel


def _ryr_folder(folder: Path, series=("039", "096"), angles=SIGNED) -> Path:
    for name in series:
        for tilt, angle in enumerate(angles, start=1):
            _tiff(folder / f"tomogram_{name}_{tilt:03d}_{angle}.tif")
    return folder


def _mdoc(path: Path, series: str, angles, *, num_subframes: int = 4, rotation=85.3,
          frame_names=None) -> None:
    lines = ["PixelSpacing = 1.35", "Voltage = 300", f"ImageFile = {series}.mrc", "",
             "[T = SerialEM: Digitized on a test scope]", ""]
    for z, angle in enumerate(angles):
        name = frame_names[z] if frame_names else f"{series}_{z:03d}_{angle}.mrc"
        lines += [f"[ZValue = {z}]", f"TiltAngle = {angle}", "PixelSpacing = 1.35",
                  "Binning = 1", f"NumSubFrames = {num_subframes}", "ExposureDose = 3.0",
                  "TargetDefocus = -3"]
        if rotation is not None:
            lines.append(f"RotationAngle = {rotation}")
        lines += [f"SubFramePath = D:\\DATA\\{name}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _mdoc_folder(root: Path, series=("TS_01", "TS_03"), *, frames: int = 4, declared: int = 4,
                 rotation=85.3) -> Path:
    for name in series:
        for z, angle in enumerate(DOSE_SYMMETRIC):
            _mrc(root / "frames" / f"{name}_{z:03d}_{angle}.mrc", (frames, 6, 8), 1.35)
        _mdoc(root / "mdocs" / f"{name}.mrc.mdoc", name, DOSE_SYMMETRIC,
              num_subframes=declared, rotation=rotation)
    return root


def _stack_folder(root: Path, series=("TS_0002", "TS_0005"), *, images: int = 5) -> Path:
    for name in series:
        _mrc(root / "Untreated_dataset" / f"{name}.mrc", (images, 6, 8), 1.223, np.float32)
        _mdoc(root / "mdoc_files" / f"{name}.mrc.mdoc", name, DOSE_SYMMETRIC,
              num_subframes=10, frame_names=[f"{name}_{z:03d}.tif" for z in range(5)])
    return root


def read(*argv) -> dict:
    return show_folder.build(show_folder.parse_args([str(arg) for arg in argv]))


def rows(report: dict) -> dict[str, dict]:
    return {row["series"]: row for row in report["series"]}


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
    }


# --- Names ---------------------------------------------------------------------------


def test_signed_angles_are_read_by_inits_own_functions(tmp_path):
    frames = _ryr_folder(tmp_path / "frames")
    report = read(frames, "--pattern", PATTERN)

    assert report["exit"] == 0
    assert report["refusal"] is None
    for row in rows(report).values():
        assert row["verdict"] == "accepted"
        assert (row["tilts"], row["angle_min"], row["angle_max"]) == (5, -6.0, 6.0)
        assert row["steps"] == {"3": 4}
        assert (row["frames"], row["pixel"]) == ([3], pytest.approx(0.84))
    warnings = " | ".join(report["warnings"])
    assert "the tilt number follows the angle order" in warnings
    assert "leading zeros (039, 096)" in warnings

    fragment = tomllib.loads(report["fragment"])
    assert fragment["lane"]["input"] == "frames"
    # A TOML literal string: the pattern comes back exactly as given.
    assert fragment["data"]["frames_pattern"] == PATTERN
    assert fragment["data"]["frames"] == str(frames)
    assert fragment["run"]["series"] == ["039", "096"]
    # Nothing in the names gives the tilt axis, so the fragment asks for it.
    assert "# tilt_axis_angle =" in report["fragment"]


def test_a_pattern_without_the_plus_is_refused_and_shows_the_half_it_read(tmp_path):
    frames = _ryr_folder(tmp_path / "frames")
    report = read(frames, "--pattern", NO_PLUS)

    assert report["exit"] == 1
    assert report["fragment"] is None
    assert "do not match the frames pattern" in report["refusal"]
    assert report["unmatched_counts"] == {".tif": 4}
    assert all("+" in name for name in report["unmatched"][".tif"])
    for row in rows(report).values():
        assert row["verdict"] == "refused"
        assert (row["tilts"], row["angle_max"]) == (3, 0.0)
    assert any(w.startswith("every angle is <= 0 in 039, 096") for w in report["warnings"])


def test_without_mdocs_or_a_pattern_the_names_are_described_and_nothing_decided(tmp_path):
    frames = _ryr_folder(tmp_path / "frames")
    report = read(frames)

    assert report["exit"] == 1
    assert report["angles_from"] is None and report["series"] == []
    assert "--pattern" in report["next"]
    (table,) = report["names"]["fields"]
    assert (table["names"], table["fields"]) == (10, 4)
    prefix, series, tilt, angle = table["table"]
    assert (prefix["distinct"], prefix["examples"]) == (1, ["tomogram"])
    assert (series["kind"], series["widths"], series["examples"]) == ("digits", [3],
                                                                      ["039", "096"])
    assert (tilt["min"], tilt["max"]) == ("001", "005")
    assert angle["kind"] == "number with a sign or point"
    assert (angle["min"], angle["max"]) == ("-06.0", "+06.0")


def test_the_tilt_number_out_of_angle_order_is_described(tmp_path):
    for tilt, angle in enumerate(DOSE_SYMMETRIC):
        _tiff(tmp_path / f"tomogram_1_{tilt}_{angle}.tif")
    report = read(tmp_path, "--pattern", PATTERN)

    assert report["exit"] == 0
    assert any("is not the angle order in 1" in w and "0, 3, -3, -6, 6" in w
               for w in report["warnings"])


def test_series_of_different_lengths_are_flagged(tmp_path):
    _ryr_folder(tmp_path, series=("001",))
    _ryr_folder(tmp_path, series=("002",),
                angles=[f"{a:+05.1f}" for a in range(-27, 30, 3)])
    report = read(tmp_path, "--pattern", PATTERN)

    assert any("series differ in length, 001 5 tilts to 002 19" in w
               for w in report["warnings"])


# --- Mdocs -----------------------------------------------------------------------------


def test_mdocs_under_the_folder_are_found_and_init_accepts_the_movies(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar")
    report = read(root)

    assert report["exit"] == 0
    assert report["frames"] == str(root / "frames")
    assert (report["mdocs"], report["mdocs_found"]) == (str(root / "mdocs"), "under the folder")
    for row in rows(report).values():
        assert (row["read_as"], row["verdict"], row["tilts"]) == ("frames", "accepted", 5)
        assert row["order"] == "ZValue" and row["first_angles"] == DOSE_SYMMETRIC
        assert row["mdoc"]["Voltage"] == "300"
        assert row["mdoc"]["tilt_axis_from_mdoc"] is True
        assert row["mdoc"]["frames_found"] == 5
    fragment = tomllib.loads(report["fragment"])
    assert fragment["data"] == {"frames": str(root / "frames"), "mdocs": str(root / "mdocs")}
    assert "[align]" not in report["fragment"]


def test_the_mdoc_folder_beside_the_movies_is_found(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar")
    report = read(root / "frames")

    assert report["mdocs_found"] == "in a folder beside the data"
    assert report["exit"] == 0


def test_a_frame_count_the_mdoc_does_not_declare_is_refused_in_inits_words(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar", frames=4, declared=5)
    report = read(root)

    assert report["exit"] == 1 and report["fragment"] is None
    for row in rows(report).values():
        assert row["verdict"] == "refused"
        assert "another frame count than declared" in row["refusal"]


def test_an_mdoc_that_gives_no_tilt_axis_asks_for_one(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar", rotation=None)
    report = read(root)

    assert report["exit"] == 0
    assert "# tilt_axis_angle =" in report["fragment"]
    assert any("takes no tilt axis" in w for w in report["warnings"])


def test_only_the_series_asked_for_are_read(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar")
    report = read(root, "--series", "TS_03")

    assert list(rows(report)) == ["TS_03"]
    assert tomllib.loads(report["fragment"])["run"]["series"] == ["TS_03"]


def test_a_series_left_unopened_is_not_called_checked(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar")
    report = read(root, "--max-series", "1")

    assert [row["verdict"] for row in report["series"]] == ["accepted", "not opened"]
    assert report["exit"] == 1
    assert any(w.startswith("not opened, so not checked: TS_03") for w in report["warnings"])


def test_stage_readings_a_hundredth_off_are_one_step(tmp_path):
    # EMPIAR-11538's mdocs read -51.99, -50.01, -47.98 ... for a 2-degree scheme.
    angles = [-4.01, -1.99, 0.02, 1.98, 4.01]
    for z, angle in enumerate(angles):
        _mrc(tmp_path / "frames" / f"TS_01_{z:03d}_{angle}.mrc", (4, 6, 8), 1.35)
    _mdoc(tmp_path / "mdocs" / "TS_01.mrc.mdoc", "TS_01", angles)
    report = read(tmp_path)

    (row,) = report["series"]
    assert row["steps"] == {"2": 4}
    assert not any("uneven" in w for w in report["warnings"])


def test_mdocs_without_their_data_are_named(tmp_path):
    root = _stack_folder(tmp_path / "11538")
    _mdoc(root / "mdoc_files" / "TS_1002.mrc.mdoc", "TS_1002", DOSE_SYMMETRIC)
    report = read(root)

    assert rows(report)["TS_1002"]["verdict"] == "refused"
    assert any(w.startswith("mdocs whose frames or stack are not here: TS_1002")
               for w in report["warnings"])
    # An mdoc with nothing beside it is no second way of reading the folder.
    assert report["input"] == "stack"
    assert not any("more than one way" in w for w in report["warnings"])
    assert report["exit"] == 1 and report["fragment"] is None
    assert read(root, "--series", "TS_0002", "--series", "TS_0005")["exit"] == 0


# --- Stacks and what has no entry --------------------------------------------------------


def test_assembled_stacks_with_their_mdocs_make_a_stack_lane(tmp_path):
    root = _stack_folder(tmp_path / "11538")
    report = read(root)

    assert report["exit"] == 0
    assert report["input"] == "stack"
    for row in rows(report).values():
        assert (row["read_as"], row["verdict"], row["tilts"]) == ("stack", "accepted", 5)
        assert row["pixel"] == pytest.approx(1.223)
    fragment = tomllib.loads(report["fragment"])
    assert fragment["data"] == {
        "stack": f"{root / 'Untreated_dataset'}/{{series}}.mrc",
        "mdoc": f"{root / 'mdoc_files'}/{{series}}.mrc.mdoc",
    }


def test_a_stack_with_another_image_count_is_refused_in_the_imports_words(tmp_path):
    root = _stack_folder(tmp_path / "11538", images=4)
    report = read(root)

    assert report["exit"] == 1 and report["fragment"] is None
    for row in rows(report).values():
        assert "external stack has 4 images, mdoc has 5 ZValue tilts" in row["refusal"]


def test_one_image_per_file_has_no_entry(tmp_path):
    for tilt, angle in enumerate(DOSE_SYMMETRIC):
        _tiff(tmp_path / f"TS_01_{tilt:03d}_{angle}.tif", frames=1)
    report = read(tmp_path)

    assert report["exit"] == 1 and report["fragment"] is None
    assert "already summed" in report["no_entry"]
    assert report["next"] is None


def test_an_eer_folders_verdict_is_the_movie_readers_own(tmp_path):
    # Not real EER files: whatever the movie reader says of them is the verdict.
    names = [f"Position_02_{z + 1:03d}[{angle:.2f}]_EER.eer"
             for z, angle in enumerate(DOSE_SYMMETRIC)]
    for name in names:
        (tmp_path / name).write_bytes(b"\0" * 64)
    with pytest.raises(Exception) as reader:
        inspect_movie(tmp_path / names[0])
    report = read(tmp_path, "--pattern",
                  r"Position_(?P<series>\d+)_(?P<tilt>\d+)\[(?P<angle>[-+]?\d+(?:\.\d+)?)\]"
                  r"_EER\.eer")

    (row,) = report["series"]
    assert (row["series"], row["tilts"], row["verdict"]) == ("02", 5, "refused")
    assert row["refusal"] == str(reader.value)
    assert report["exit"] == 1


# --- The folder and the arguments ------------------------------------------------------


def test_several_data_folders_are_listed_not_chosen(tmp_path):
    _ryr_folder(tmp_path / "a")
    _stack_folder(tmp_path / "b")
    report = read(tmp_path)

    assert report["frames"] is None and report["exit"] == 1
    assert report["frames_candidates"] == [str(tmp_path / "a"),
                                           str(tmp_path / "b" / "Untreated_dataset")]


def test_a_tree_deeper_than_read_has_no_only_folder(tmp_path):
    # The data of one condition one level below what is read, as on the Mac's
    # EMPIAR-11538 copy, where etomo folders hold the untreated stacks.
    _ryr_folder(tmp_path / "data" / "treated")
    _ryr_folder(tmp_path / "data" / "untreated" / "TS_0002")
    report = read(tmp_path)

    assert report["frames"] is None and report["exit"] == 1
    assert "deeper than read" in report["next"]
    assert read(tmp_path / "data" / "treated")["frames"] == str(tmp_path / "data" / "treated")


def test_the_folder_is_only_read(tmp_path):
    root = _mdoc_folder(tmp_path / "empiar")
    _ryr_folder(root / "frames")
    (root / "gain").mkdir()
    (root / "gain" / "CountRef_x.dm4").write_bytes(b"\0" * 16)
    before = snapshot(tmp_path)

    report = read(root)
    read(root / "frames", "--pattern", PATTERN)
    show_folder.main([str(root), "--json"])

    assert snapshot(tmp_path) == before
    assert report["gains"] == [os.path.join("gain", "CountRef_x.dm4")]


def test_the_text_and_the_json_say_the_same(tmp_path, capsys):
    root = _mdoc_folder(tmp_path / "empiar")
    assert show_folder.main([str(root), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert show_folder.main([str(root)]) == 0
    text = capsys.readouterr().out

    assert report["fragment"] and all(line in text for line in report["fragment"].splitlines())
    for row in report["series"]:
        assert f"  {row['series']:<14} frames" in text


@pytest.mark.parametrize("argv", [
    ["{missing}"],
    ["{root}", "--pattern", PATTERN, "--mdocs", "{root}"],
    ["{root}", "--mdocs", "{missing}"],
    ["{root}", "--series", "TS_01", "--series", "TS_01"],
])
def test_contradicting_arguments_stop_with_status_2(tmp_path, argv):
    root = _mdoc_folder(tmp_path / "empiar")
    filled = [arg.format(root=root, missing=tmp_path / "missing") for arg in argv]
    assert show_folder.main(filled) == 2


def test_super_resolution_k3_frames_are_named():
    warnings = show_folder.warnings_for([], [{"x": 11520, "y": 8184}], {})
    assert any("super-resolution" in w and "ftbin" in w for w in warnings)


def test_lane_paths_are_relative_to_the_repository_inside_it(tmp_path):
    assert show_folder.lane_path(ROOT / "data" / "x" / "frames") == "data/x/frames"
    assert show_folder.lane_path(tmp_path) == str(tmp_path)
