"""scripts/show_tilts.py and cryoet_pipeline.tilt_preview, on synthetic tilt series.

A synthetic series is a slab of faint blobs and gold beads, projected about an
axis at a known IMOD angle: the axis the tilts give must come back, the beads
must be counted and a slab without them must give none.
"""

import math
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import tifffile

from cryoet_pipeline import tilt_preview
from cryoet_pipeline.tilt_preview import (
    TiltPicture,
    axis_difference,
    circular_median,
    estimate_axis,
    find_dark_spots,
    orientation_word,
    stretch_across,
    sum_movie,
    typical_diameter,
    views_to_read,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import show_tilts  # noqa: E402

PIXEL = 10.0
SIZE = 192
ANGLES = [float(angle) for angle in range(-60, 61, 3)]
PATTERN = r"tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif"


def _specimen(seed: int, *, beads: int):
    """Points of a slab 60 px thick: (x, y, z, sigma px, depth), x and y about the centre."""

    rng = np.random.default_rng(seed)
    blobs = []
    for _ in range(900):
        blobs.append((rng.uniform(-130, 130), rng.uniform(-130, 130), rng.uniform(-30, 30),
                      rng.uniform(1.5, 4.0), rng.uniform(0.02, 0.06)))
    # 10 nm gold at 10 Å per pixel: discs of radius 5 px, far darker than the rest, on a
    # jittered grid so that none touch and none sit at an edge, on both faces of the slab.
    grid = [(x, y) for y in np.linspace(-60, 60, 5) for x in np.linspace(-62, 62, 6)]
    for x, y in grid[:beads]:
        blobs.append((x + rng.uniform(-4, 4), y + rng.uniform(-4, 4),
                      rng.choice([-28.0, 28.0]), 0.0, 0.55))
    return blobs


def _project(blobs, angle_deg: float, axis_deg: float, seed: int, *, dose: float = 400.0,
             blank: bool = False) -> np.ndarray:
    """One tilt in the stack's orientation (row 0 at the bottom), counts with Poisson noise."""

    theta, phi = math.radians(angle_deg), math.radians(axis_deg)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    centre = (SIZE - 1) / 2.0
    transmission = np.ones((SIZE, SIZE))
    for x, y, z, sigma, depth in blobs:
        # Tilted about the specimen's y axis, then turned so that axis runs at IMOD's phi.
        along_x = x * math.cos(theta) + z * math.sin(theta)
        image_x = along_x * math.cos(phi) - y * math.sin(phi) + centre
        image_y = along_x * math.sin(phi) + y * math.cos(phi) + centre
        distance2 = (xx - image_x) ** 2 + (yy - image_y) ** 2
        if sigma == 0.0:
            transmission *= 1.0 - depth * (distance2 <= 25.0)
        else:
            transmission *= 1.0 - depth * np.exp(-distance2 / (2.0 * sigma**2))
    mean = dose * math.cos(theta) * (0.004 if blank else 1.0)
    rng = np.random.default_rng(seed)
    return rng.poisson(mean * transmission).astype(np.float32)


def _pictures(axis_deg: float, angles, *, beads: int = 0, seed: int = 0, blank=()):
    blobs = _specimen(seed, beads=beads)
    return [
        TiltPicture(z_value=z, angle_deg=angle, source=f"view {z}",
                    image=_project(blobs, angle, axis_deg, seed + z,
                                   blank=angle in blank),
                    pixel_angstrom=PIXEL, binning=1)
        for z, angle in enumerate(angles)
    ]


def _pairs(pictures):
    by_angle = {picture.angle_deg: picture for picture in pictures}
    angles = [picture.angle_deg for picture in pictures]
    _, pairs = views_to_read(angles)
    return [(by_angle[angles[lower]], by_angle[angles[higher]]) for lower, higher in pairs]


# --- The axis --------------------------------------------------------------------------


@pytest.mark.parametrize("axis_deg", [84.0, 96.0, 6.0, 40.0, 179.5])
def test_the_axis_comes_back_from_the_tilts(axis_deg):
    # 84 and 96 differ only by an upside-down flip, 84 and 6 by a transpose:
    # an estimate that ignored either would fail one of them. At 179.5 the
    # pairs' bests fall on both sides of 0/180, which a plain median averages
    # to a horizontal axis.
    angles = [-60.0, -57.0, -54.0, -51.0, 0.0, 51.0, 54.0, 57.0, 60.0]
    estimate = estimate_axis(_pairs(_pictures(axis_deg, angles, seed=3)))

    assert estimate.angle_deg is not None
    assert abs(axis_difference(estimate.angle_deg, axis_deg)) <= 2.0
    assert estimate.agreeing == len(estimate.pairs) == 6


def test_a_stretch_across_the_axis_undoes_the_tilt():
    blobs = _specimen(1, beads=0)
    flat = [(x, y, 0.0, sigma, depth) for x, y, _, sigma, depth in blobs]
    upright = _project(flat, 0.0, 30.0, 0, dose=1e6)
    tilted = _project(flat, 60.0, 30.0, 0, dose=1e6)

    def match(first, second):
        first = first - first.mean()
        second = second - second.mean()
        return float((first * second).sum() / math.sqrt((first**2).sum() * (second**2).sum()))

    right = stretch_across(tilted, 30.0, 0.5)
    wrong = stretch_across(tilted, 120.0, 0.5)
    window = (slice(60, 132), slice(60, 132))
    assert match(right[window], upright[window]) > 0.9
    assert match(wrong[window], upright[window]) < 0.6


def test_views_to_read_takes_the_four_highest_on_each_side():
    zero, pairs = views_to_read(ANGLES)
    assert ANGLES[zero] == 0.0
    assert sorted((ANGLES[lower], ANGLES[higher]) for lower, higher in pairs) == [
        (-57.0, -60.0), (-54.0, -57.0), (-51.0, -54.0), (51.0, 54.0), (54.0, 57.0),
        (57.0, 60.0),
    ]
    # A lamella's range, -52 to +66 in 2 degree steps: each side still gives three.
    lamella = [float(angle) for angle in range(-52, 67, 2)]
    assert len(views_to_read(lamella)[1]) == 6
    # Nothing reaches 20 degrees: no pair.
    assert views_to_read([-15.0, -12.0, 0.0, 12.0, 15.0])[1] == []


def test_an_empty_tilt_is_left_out_of_the_axis(tmp_path):
    folder = _tiff_folder(tmp_path, axis_deg=84.0, blank=(-60.0,))
    manifest = _manifest(folder)
    preview = tilt_preview.preview_series(manifest)

    assert len(preview.axis.pairs) == 5
    assert all(-60.0 not in (pair.lower_deg, pair.higher_deg) for pair in preview.axis.pairs)
    assert any("-60.0 deg tilt is left out" in note for note in preview.notes)
    assert abs(axis_difference(preview.axis.angle_deg, 84.0)) <= 2.0


def test_lines_without_a_sign():
    assert axis_difference(179.0, 1.0) == pytest.approx(-2.0)
    assert axis_difference(-95.0, 85.0) == pytest.approx(0.0)
    assert circular_median([179.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert circular_median([84.0, 86.0, 85.0, 30.0]) == pytest.approx(84.5)
    assert orientation_word(85.3) == "horizontal"
    assert orientation_word(-96.0) == "horizontal"
    assert orientation_word(175.0) == "vertical"
    assert orientation_word(45.0) == "diagonal"


# --- The beads -------------------------------------------------------------------------


def test_gold_is_counted_and_a_slab_without_gold_gives_none():
    with_gold = _project(_specimen(5, beads=30), 0.0, 84.0, 1)
    without = _project(_specimen(5, beads=0), 0.0, 84.0, 1)

    found = find_dark_spots(with_gold, PIXEL)
    counted = [spot for spot in found if spot.counts]
    assert len(counted) == 30
    # The filter ladder steps 8, 10, 12 nm: a hard-edged 10 nm disc may answer one step off.
    assert typical_diameter(found) in (8.0, 10.0, 12.0)
    assert not [spot for spot in find_dark_spots(without, PIXEL) if spot.counts]


# --- Reading ---------------------------------------------------------------------------


def test_a_tiff_is_turned_upside_down_and_an_mrc_is_not(tmp_path):
    frames = np.zeros((2, 4, 3), np.uint8)
    frames[:, 0, :] = 5  # row 0 as stored
    tifffile.imwrite(tmp_path / "movie.tif", frames, photometric="minisblack")
    with mrcfile.new(tmp_path / "movie.mrc") as mrc:
        mrc.set_data(frames.astype(np.int8))

    tiff_sum = sum_movie(tmp_path / "movie.tif")
    mrc_sum = sum_movie(tmp_path / "movie.mrc")
    assert tiff_sum[-1].tolist() == [10.0, 10.0, 10.0] and tiff_sum[0].sum() == 0
    assert mrc_sum[0].tolist() == [10.0, 10.0, 10.0] and mrc_sum[-1].sum() == 0


def test_a_compression_tifffile_cannot_read_names_the_package(tmp_path, monkeypatch):
    tifffile.imwrite(tmp_path / "movie.tif", np.zeros((2, 4, 3), np.uint8),
                     photometric="minisblack")

    def refuse(self, *args, **kwargs):
        raise KeyError("<COMPRESSION.LZW: 5> requires the 'imagecodecs' package")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", refuse)
    with pytest.raises(ValueError, match="pip install imagecodecs"):
        sum_movie(tmp_path / "movie.tif")


def test_eer_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="EER movies are not read"):
        sum_movie(tmp_path / "movie.eer")


# --- The script ------------------------------------------------------------------------


def _tiff_folder(tmp_path: Path, *, axis_deg: float, beads: int = 0, blank=()) -> Path:
    """A RyR1-like folder: signed angles in the names, TIFFs as SerialEM leaves them.

    MotionCor3 reads a TIFF bottom row first, so the file holds each tilt upside
    down from the stack it will make.
    """

    folder = tmp_path / "frames"
    folder.mkdir(parents=True, exist_ok=True)
    blobs = _specimen(7, beads=beads)
    for tilt, angle in enumerate(ANGLES, start=1):
        needed = angle == 0.0 or abs(angle) >= 51.0
        image = (_project(blobs, angle, axis_deg, tilt, blank=angle in blank)
                 if needed else np.zeros((SIZE, SIZE), np.float32))
        half = np.round(image[::-1] / 2.0).astype(np.uint16)
        tifffile.imwrite(folder / f"tomogram_039_{tilt:03d}_{angle:+05.1f}.tif",
                         np.stack([half, half]), photometric="minisblack",
                         resolution=(1e8 / PIXEL, 1e8 / PIXEL), resolutionunit="CENTIMETER")
    return folder


def _manifest(folder: Path):
    from cryoet_pipeline.models import ProjectConfig
    from cryoet_pipeline.project import load_tilt_series_manifest

    config = ProjectConfig(frames_dir=folder, frames_pattern=PATTERN,
                           output_dir=Path("/dev/null"))
    return load_tilt_series_manifest(config, "039")


def _mrc_folder(tmp_path: Path, *, axis_deg: float, rotation: float) -> Path:
    blobs = _specimen(9, beads=20)
    frames = tmp_path / "frames"
    frames.mkdir(parents=True)
    lines = ["PixelSpacing = 10", "ImageFile = TS_01.mrc", "",
             "[T = SerialEM: a test scope]", ""]
    for z, angle in enumerate(ANGLES):
        name = f"TS_01_{z:03d}_{angle}.mrc"
        needed = angle == 0.0 or abs(angle) >= 51.0
        image = _project(blobs, angle, axis_deg, z) if needed else np.zeros((SIZE, SIZE))
        with mrcfile.new(frames / name) as mrc:
            mrc.set_data(np.stack([image, image]).astype(np.float32) / 2.0)
            mrc.voxel_size = PIXEL
        lines += [f"[ZValue = {z}]", f"TiltAngle = {angle}", "PixelSpacing = 10",
                  "Binning = 1", "NumSubFrames = 2", f"RotationAngle = {rotation}",
                  f"SubFramePath = D:\\DATA\\{name}", ""]
    (tmp_path / "mdocs").mkdir()
    (tmp_path / "mdocs" / "TS_01.mrc.mdoc").write_text("\n".join(lines))
    return tmp_path


def test_a_tiff_folder_by_pattern_gives_the_axis_and_the_beads(tmp_path, capsys):
    folder = _tiff_folder(tmp_path, axis_deg=84.0, beads=25)
    out = tmp_path / "tilts.pdf"

    status = show_tilts.main([str(folder), "--pattern", PATTERN, "--out", str(out)])

    printed = capsys.readouterr().out
    assert status == 0, printed
    assert out.read_bytes().startswith(b"%PDF")
    # Read wrongly upside down, the axis would come back as 96 degrees.
    found = float(printed.split("tilt axis from the tilts:")[1].split("°")[0])
    assert abs(found - 84.0) <= 3.0
    assert "horizontal" in printed
    assert "dark round spots, most about" in printed
    assert "none: the angles come from file names" in printed


def test_mdocs_give_the_recorded_axis_and_the_tilts_are_compared(tmp_path, capsys):
    root = _mrc_folder(tmp_path, axis_deg=85.3, rotation=175.3)

    status = show_tilts.main([str(root / "frames"), "--mdocs", str(root / "mdocs"),
                              "--out", str(tmp_path / "t.pdf")])

    printed = capsys.readouterr().out
    assert status == 0, printed
    assert "85.3° (TS_01.mrc.mdoc: RotationAngle - 90): the tilts agree within" in printed


def test_an_mdoc_axis_the_tilts_do_not_show_is_called_out(tmp_path, capsys):
    root = _mrc_folder(tmp_path, axis_deg=85.3, rotation=85.3)  # axis -4.7: vertical

    show_tilts.main([str(root / "frames"), "--mdocs", str(root / "mdocs"),
                     "--out", str(tmp_path / "t.pdf")])

    assert "the tilts differ by" in capsys.readouterr().out


def test_a_stack_is_read_by_section(tmp_path, capsys):
    blobs = _specimen(11, beads=0)
    angles = [-60.0, -57.0, -54.0, -51.0, 0.0, 51.0, 54.0, 57.0, 60.0]
    order = [4, 5, 3, 6, 2, 7, 1, 8, 0]  # dose-symmetric: ZValue is not angle order
    stack = np.stack([_project(blobs, angles[index], 6.0, index) for index in order])
    with mrcfile.new(tmp_path / "TS_9.mrc") as mrc:
        mrc.set_data(stack.astype(np.float32))
        mrc.voxel_size = PIXEL
    lines = ["PixelSpacing = 10", ""]
    for z, index in enumerate(order):
        lines += [f"[ZValue = {z}]", f"TiltAngle = {angles[index]}", "PixelSpacing = 10",
                  "Binning = 1", "NumSubFrames = 10", "RotationAngle = 96.0",
                  f"SubFramePath = D:\\DATA\\TS_9_{z:03d}.tif", ""]
    (tmp_path / "TS_9.mrc.mdoc").write_text("\n".join(lines))

    status = show_tilts.main(["--stack", str(tmp_path / "TS_9.mrc"), "--mdoc",
                              str(tmp_path / "TS_9.mrc.mdoc"), "--out", str(tmp_path / "s.pdf")])

    printed = capsys.readouterr().out
    assert status == 0, printed
    assert "vertical" in printed
    assert "6.0° (TS_9.mrc.mdoc: RotationAngle - 90): the tilts agree" in printed
    assert "no spot at most 0.8" in printed


def test_only_the_first_series_is_read_unless_asked(tmp_path, capsys):
    folder = _tiff_folder(tmp_path, axis_deg=84.0)
    for path in list(folder.iterdir()):
        (folder / path.name.replace("_039_", "_096_")).write_bytes(path.read_bytes())

    show_tilts.main([str(folder), "--pattern", PATTERN, "--out", str(tmp_path / "a.pdf")])
    printed = capsys.readouterr().out
    assert "reading 039, the first of 2 series (039, 096)" in printed
    assert "096:" not in printed


@pytest.mark.parametrize("argv", [
    [],
    ["frames"],
    ["frames", "--mdocs", "m", "--pattern", "p"],
    ["--stack", "s.mrc"],
    ["frames", "--pattern", "p", "--stack", "s.mrc", "--mdoc", "m"],
    ["frames", "--pattern", "p", "--all", "--series", "039"],
])
def test_arguments_that_cannot_work_are_refused(argv, capsys):
    assert show_tilts.main(argv) == 2


def test_a_folder_init_would_refuse_is_refused_in_its_words(tmp_path, capsys):
    status = show_tilts.main([str(tmp_path), "--mdocs", str(tmp_path / "nowhere"),
                              "--out", str(tmp_path / "x.pdf")])
    assert status == 1
    assert "mdocs directory not found" in capsys.readouterr().out
    assert not (tmp_path / "x.pdf").exists()


def test_the_default_pdf_is_named_after_the_folder_and_its_series():
    tilt_previews = Path("outputs") / "tilt-previews"
    assert show_tilts.default_out("frames", ["TS_01"]) == tilt_previews / "frames_TS_01.pdf"
    assert show_tilts.default_out("frames", ["TS_01", "TS_43"]) == (
        tilt_previews / "frames_TS_01-TS_43.pdf")
    assert show_tilts.default_out("frames", ["1", "2", "3", "4", "5"]) == (
        tilt_previews / "frames_5-series.pdf")
    assert show_tilts.default_out("TS_1002", ["TS_1002"]) == tilt_previews / "TS_1002.pdf"
