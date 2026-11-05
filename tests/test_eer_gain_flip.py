"""scripts/eer_gain_flip.py: the FmIntFile it writes, and its reading of the sums."""

from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eer_gain_flip.py"

_spec = importlib.util.spec_from_file_location("eer_gain_flip", SCRIPT)
assert _spec is not None and _spec.loader is not None
eer_gain_flip = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eer_gain_flip
_spec.loader.exec_module(eer_gain_flip)

# The acquisition metadata of an EMPIAR-11830 movie, shortened (Position_1_010).
METADATA = (
    '<metadata><item name="commercialName">Falcon 4</item>'
    '<item name="eerGainReference">ImagesForProcessing/EF-Falcon/300kV/'
    "20230428_151632_EER_GainReference.gain</item>"
    '<item name="exposureTime" unit="s">3.0278130000000001</item>'
    '<item name="numberOfFrames">{declared}</item>'
    '<item name="sensorPixelSize.width" unit="m">1.90370344e-10</item>'
    '<item name="totalDose" unit="e/pixel">6.3804134669928416</item></metadata>'
)


def _write_eer_like(
    path: Path,
    frames: int,
    *,
    declared: int | None = None,
    integrated_first: bool = False,
    size: int = 16,
) -> None:
    """A BigTIFF laid out as the 11830 EER files are: one strip and one IFD per
    frame, compression 65001, TIFF Orientation 2, and the acquisition metadata in
    tag 65001 of the first page. The strips are not real EER; nothing decodes them."""

    xml = METADATA.format(declared=frames if declared is None else declared).encode()
    body = bytearray(xml)
    strips = []
    pages = [1] * integrated_first + [65001] * frames
    for index in range(len(pages)):
        strips.append((16 + len(body), 10))
        body += bytes([index % 256]) * 10
    ifd_at = 16 + len(body)
    ifds = bytearray()
    for index, compression in enumerate(pages):
        entries = [
            (256, 4, 1, size),
            (257, 4, 1, size),
            (259, 3, 1, compression),
            (262, 3, 1, 1),
            (273, 16, 1, strips[index][0]),
            (274, 3, 1, 2),
            (278, 4, 1, size),
            (279, 16, 1, strips[index][1]),
        ]
        if index == 0:
            entries.append((65001, 7, len(xml), 16))
        here = ifd_at + len(ifds)
        last = index == len(pages) - 1
        following = 0 if last else here + 8 + 20 * len(entries) + 8
        ifds += struct.pack("<Q", len(entries))
        for tag, kind, count, value in entries:
            ifds += struct.pack("<HHQQ", tag, kind, count, value)
        ifds += struct.pack("<Q", following)
    path.write_bytes(b"II" + struct.pack("<HHHQ", 43, 8, 0, ifd_at) + bytes(body) + bytes(ifds))


def test_fmint_reads_the_frames_from_the_tags_and_writes_one_line(tmp_path: Path) -> None:
    movie = tmp_path / "Position_1_010_-0.01_20230801_175515_EER.eer"
    _write_eer_like(movie, 728)
    out = tmp_path / "movie.fmint"
    gain = tmp_path / "20230807_182744_EER_GainReference.gain"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "fmint", str(movie), "--fractions", "10",
         "--out", str(out), "--gain", str(gain)],
        capture_output=True, text=True, check=True,
    )

    assert out.read_text() == "728 72 0\n"
    assert "frames           728" in result.stdout
    assert "pixel_angstrom   1.9037" in result.stdout
    assert "orientation_tag  2" in result.stdout
    assert "10 fractions of 72 frames, the last 8 with one more" in result.stdout
    # 11830's movies name an older gain than the one deposited with them.
    assert "(ANOTHER name)" in result.stdout


def test_the_groups_always_outnumber_the_frames_left_over() -> None:
    # MotionCor3 writes past its array otherwise (CFmIntParam::mCalcIntFms).
    for fractions in range(1, 31):
        for frames in range(fractions, 2001):
            n, group, dose = eer_gain_flip.fmint_line(frames, fractions).split()
            groups, left = frames // int(group), frames % int(group)
            assert (int(n), dose) == (frames, "0")
            assert groups >= fractions
            assert left < groups, (frames, fractions)


def test_fmint_refuses_fewer_frames_than_fractions() -> None:
    with pytest.raises(ValueError, match="9 EER frames cannot make 10"):
        eer_gain_flip.fmint_line(9, 10)


def test_a_movie_whose_metadata_counts_other_frames_is_refused(tmp_path: Path) -> None:
    movie = tmp_path / "short.eer"
    _write_eer_like(movie, 7, declared=9)

    with pytest.raises(ValueError, match="7 frames on disk, its metadata says 9"):
        eer_gain_flip.eer_facts(movie)


def test_a_movie_that_starts_with_an_integrated_image_is_refused(tmp_path: Path) -> None:
    movie = tmp_path / "integrated.eer"
    _write_eer_like(movie, 7, integrated_first=True)

    with pytest.raises(ValueError, match="counts every page as a frame"):
        eer_gain_flip.eer_facts(movie)


# --- compare ---------------------------------------------------------------------


def _write(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))


def _sums(
    directory: Path,
    stem: str,
    *,
    needs_flip: int,
    gain: np.ndarray,
    rng: np.random.Generator,
    dose: float = 10.0,
) -> None:
    """MotionCor3's -Align 0 sums of one movie, as the sources say it makes them.

    `gain` is the gain that flattens the frames, in their orientation. MotionCor3
    holds it upside down when `needs_flip` is 1, and turns it over with
    -FlipGain 1 (np.flipud)."""

    shape = gain.shape
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    specimen = 1.0 + 0.3 * np.sin(2 * np.pi * xx / 64) * np.cos(2 * np.pi * yy / 50)
    # A pixel the gain marks 0 is dead: it counts nothing.
    rate = np.where(gain > 0, dose * specimen / np.where(gain > 0, gain, 1.0), 0.0)
    raw = rng.poisson(rate).astype(np.float64)
    held = gain if needs_flip == 0 else np.flipud(gain)
    _write(directory / f"{stem}_nogain.mrc", raw)
    _write(directory / f"{stem}_flip0.mrc", raw * held)
    _write(directory / f"{stem}_flip1.mrc", raw * np.flipud(held))


def _compare(directory: Path) -> str:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "compare", str(directory)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def _pixel_gain(rng: np.random.Generator, spread: float, shape=(256, 256)) -> np.ndarray:
    sensitivity = 1.0 + spread * rng.standard_normal(shape)
    return sensitivity.mean() / sensitivity


@pytest.mark.parametrize("needs_flip", [0, 1])
def test_the_flip_that_flattens_the_frames_is_found(tmp_path: Path, needs_flip: int) -> None:
    rng = np.random.default_rng(11830 + needs_flip)
    gain = _pixel_gain(rng, 0.05)
    for stem in ("movie_001", "movie_002"):
        _sums(tmp_path, stem, needs_flip=needs_flip, gain=gain, rng=rng)

    out = _compare(tmp_path)

    assert f"result:               FlipGain {needs_flip}; per movie 2/2 agree" in out
    assert "A_1 vs A_0 flipud  +1.0000" in out


def test_a_gain_that_looks_the_same_upside_down_cannot_decide(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    gain = _pixel_gain(rng, 0.05)
    gain = (gain + np.flipud(gain)) / 2
    _sums(tmp_path, "movie_001", needs_flip=1, gain=gain, rng=rng)

    out = _compare(tmp_path)

    assert "inconclusive: flips [0, 1] anticorrelate about equally" in out
    assert "A_0 vs A_1         +1.0000" in out


def test_a_gain_with_no_pattern_above_the_shot_noise_cannot_decide(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    gain = _pixel_gain(rng, 0.0005)
    _sums(tmp_path, "movie_001", needs_flip=1, gain=gain, rng=rng)

    out = _compare(tmp_path)

    assert "inconclusive: no flip's gain anticorrelates with the counts" in out


def test_defects_and_hot_pixels_are_left_out(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    gain = _pixel_gain(rng, 0.05)
    gain[100, 100] = 0.0  # a defect, as a Falcon gain marks one
    _sums(tmp_path, "movie_001", needs_flip=1, gain=gain, rng=rng)
    raw_path = tmp_path / "movie_001_nogain.mrc"
    with mrcfile.open(raw_path, mode="r+") as mrc:
        mrc.data[50, 60] = 5000.0  # a hot pixel

    judgement = eer_gain_flip.judge(
        "movie_001",
        eer_gain_flip._read_mrc(raw_path),
        {k: eer_gain_flip._read_mrc(tmp_path / f"movie_001_flip{k}.mrc") for k in (0, 1)},
    )

    assert judgement.flip == 1
    assert judgement.left_out >= 3  # the defect, where it lands when flipped, the hot pixel


def test_sums_without_a_flip0_are_refused(tmp_path: Path) -> None:
    _write(tmp_path / "movie_001_nogain.mrc", np.ones((64, 64)))
    _write(tmp_path / "movie_001_flip1.mrc", np.ones((64, 64)))

    with pytest.raises(ValueError, match="needs movie_001_flip0.mrc"):
        eer_gain_flip.read_sums(tmp_path)
