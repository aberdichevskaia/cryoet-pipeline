#!/usr/bin/env python3
"""Which MotionCor3 -FlipGain a Falcon `.gain` needs for EER movies.

    python3 scripts/eer_gain_flip.py fmint MOVIE.eer --fractions 10 --out MOVIE.fmint
    python3 scripts/eer_gain_flip.py compare OUT_DIR

scripts/eer_gain_flip.sbatch runs both around MotionCor3; the job is described there.

`fmint` reads one EER file's tags and metadata without decoding a frame (tifffile
reads them from the page tags alone, imagecodecs is not needed), prints what they
say and writes MotionCor3's -FmIntFile: one line "N G 0", the N EER frames summed G
at a time, at dose 0 so that nothing is dose-weighted, as the pipeline runs
MotionCor3. G is N // fractions. MotionCor3 adds the N % G frames left over to the
last groups, one each, and writes past the end of its array when they outnumber
the groups (DataUtil/CFmIntParam.cpp, mCalcIntFms: 728 frames in groups of 73
leave 71 for 9 groups). N // fractions never leaves more than there are groups.

`compare` reads MotionCor3's sums written with -Align 0, so that no alignment
smears the gain's pattern: per movie <stem>_nogain.mrc, and <stem>_flip<k>.mrc for
each -FlipGain k. Without alignment a sum with the gain is the raw sum times the
gain MotionCor3 applied, so A_k = flip<k> / nogain is that gain as MotionCor3 lays
it on the frames: read off MotionCor3 itself, not off our own reading of the TIFF.
For each k:

  r_k     correlation of the raw sum with A_k, both high-passed (minus their 9x9
          box mean). A pixel that counts more electrons needs a smaller gain, so
          the orientation that matches is the one with r clearly below 0.
  flat_k  the corrected sum's high-passed standard deviation over its mean. The
          matching orientation leaves the flatter sum, as on the RyR1 TIFFs.

Once per movie, also: how alike A_0 and A_1 are (a gain that looks the same upside
down cannot be told apart from its flip, and then the flag hardly matters), and
whether A_1 is A_0 turned upside down, which is what -FlipGain 1 is documented to
do. Pixels where an A_k is below 0.2 or above 5 times its median (gain defects),
or where the raw sum is above 5 times its median plus 10 (hot pixels), and a
16-pixel border are left out. The movies are then summed and judged together.

Verdict: FlipGain k when A_k anticorrelates with the counts beyond 10/sqrt(n), by
5/sqrt(n) more than any other flip, and its sum is also the flattest; otherwise it
says which of these failed.

Prediction, from the sources (MotionCor3 main, read 4 November 2026): -FlipGain 1.
MotionCor3 reads a TIFF, a `.gain` among them, bottom row first
(TiffUtil/CLoadTiffImage.cpp), and decodes EER frames top row first
(EerUtil/CDecodeEerFrame.cpp). RELION turns a `.gain` back over before using it
with EER ("Revert Y flip in TIFF reader", src/renderEER.h). On EMPIAR-11830 the
gain and the EER files carry the same TIFF Orientation tag, 2.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
from numpy.typing import NDArray

BOX = 9
BORDER = 16
DEFECT_RANGE = (0.2, 5.0)
HOT_FACTOR = 5.0
HOT_OFFSET = 10.0
STRUCTURE_SE = 10.0
MARGIN_SE = 5.0
SAME_FLIPPED = 0.95
PREDICTION = 1
# What MotionCor3 does to the gain for each -FlipGain, in its frames' orientation
# ("1 - flip upside down, 2 - flip left right", CInput.cpp).
FLIP_OF = {1: np.flipud, 2: np.fliplr}
NOGAIN = "_nogain.mrc"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fmint = commands.add_parser("fmint", help="print an EER file's header, write its FmIntFile")
    fmint.add_argument("movie", type=Path)
    fmint.add_argument("--fractions", type=int, default=10)
    fmint.add_argument("--out", type=Path, required=True)
    fmint.add_argument("--gain", type=Path, help="the gain given, to compare with the one named")
    compare = commands.add_parser("compare", help="judge MotionCor3's -Align 0 sums")
    compare.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "fmint":
            write_fmint(args.movie, args.fractions, args.out, args.gain)
        else:
            print_comparison(args.directory)
    except ValueError as exc:
        print(f"eer_gain_flip.py: {exc}", file=sys.stderr)
        return 1
    return 0


# --- fmint -----------------------------------------------------------------------


def eer_facts(path: Path) -> dict[str, Any]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        if not tif.is_eer:
            raise ValueError(f"{path}: no EER metadata in its first page; not an EER file")
        series = tif.series
        frames = series[0]
        if len(series) > 1:
            # tifffile puts an integrated image found before the frames in a series
            # of its own; MotionCor3 would count that page as one more EER frame.
            raise ValueError(
                f"{path}: holds {', '.join(s.name for s in series)}, not EER frames alone; "
                "MotionCor3 counts every page as a frame"
            )
        keyframe = frames.keyframe
        metadata = tif.eer_metadata or {}
        n_frames = int(frames.shape[0]) if len(frames.shape) == 3 else 1
        facts: dict[str, Any] = {
            "frames": n_frames,
            "size": f"{int(keyframe.shape[1])}x{int(keyframe.shape[0])}",
            "compression": int(keyframe.compression),
            "orientation_tag": keyframe.tags.valueof(274),
            "camera": metadata.get("commercialName"),
            "exposure_s": metadata.get("exposureTime"),
            "dose_e_per_pixel": metadata.get("totalDose"),
            "gain_named": metadata.get("eerGainReference"),
        }
    declared = metadata.get("numberOfFrames")
    if declared is not None and int(declared) != n_frames:
        raise ValueError(
            f"{path}: {n_frames} frames on disk, its metadata says {declared}; "
            "an incomplete download?"
        )
    width = metadata.get("sensorPixelSize.width")
    unit = metadata.get("sensorPixelSize.width.unit")
    facts["pixel_angstrom"] = float(width) * 1e10 if width is not None and unit == "m" else None
    return facts


def fmint_line(n_frames: int, fractions: int) -> str:
    if fractions < 1:
        raise ValueError(f"fractions must be at least 1, not {fractions}")
    if n_frames < fractions:
        raise ValueError(f"{n_frames} EER frames cannot make {fractions} fractions")
    return f"{n_frames} {n_frames // fractions} 0\n"


def write_fmint(movie: Path, fractions: int, out: Path, gain: Path | None) -> None:
    facts = eer_facts(movie)
    line = fmint_line(facts["frames"], fractions)
    out.write_text(line)
    n, group = facts["frames"], facts["frames"] // fractions
    print(f"movie            {movie}")
    for key, value in facts.items():
        shown = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key:<16} {shown}")
    print(
        f"fmint            {line.strip()}: {n // group} fractions of {group} frames, "
        f"the last {n % group} with one more"
    )
    if gain is not None:
        named = Path(str(facts["gain_named"] or "").replace("\\", "/")).name
        same = "the same name" if named == gain.name else "ANOTHER name"
        print(
            f"gain             given {gain.name}; the EER metadata names "
            f"{named or 'none'} ({same})"
        )


# --- compare ---------------------------------------------------------------------


@dataclass(frozen=True)
class Judgement:
    label: str
    pixels: int
    left_out: int
    r: dict[int, float]
    flat: dict[int, float]
    alike: float
    as_documented: dict[int, float]
    gain_spread: float
    verdict: str
    flip: int | None


def high_pass(image: NDArray[Any], box: int = BOX) -> NDArray[np.float64]:
    """The image minus its mean over a box x box neighbourhood."""

    data = np.asarray(image, dtype=np.float64)
    half = box // 2
    padded = np.pad(data, half, mode="reflect")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = padded.cumsum(axis=0).cumsum(axis=1)
    sums = (
        integral[box:, box:]
        - integral[:-box, box:]
        - integral[box:, :-box]
        + integral[:-box, :-box]
    )
    return data - sums / (box * box)


def correlation(a: NDArray[Any], b: NDArray[Any]) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum()) / denominator if denominator > 0 else math.nan


def _filled(image: NDArray[Any], keep: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Pixels outside `keep` set to the median of those inside, before high-passing,
    so that a defect or a hot pixel does not reach its neighbours through the box mean."""

    data = np.asarray(image, dtype=np.float64)
    return np.where(keep, data, float(np.median(data[keep])))


def judge(label: str, raw: NDArray[Any], corrected: dict[int, NDArray[Any]]) -> Judgement:
    flips = sorted(corrected)
    raw = np.asarray(raw, dtype=np.float64)
    for k in flips:
        if corrected[k].shape != raw.shape:
            raise ValueError(f"{label}: flip {k} is {corrected[k].shape}, the raw sum {raw.shape}")
    interior = np.zeros(raw.shape, dtype=bool)
    interior[BORDER:-BORDER, BORDER:-BORDER] = True
    counted = interior & (raw > 0)
    if not counted.any():
        raise ValueError(f"{label}: the raw sum has no counts")
    keep = counted & (raw <= HOT_FACTOR * float(np.median(raw[counted])) + HOT_OFFSET)
    applied: dict[int, NDArray[np.float64]] = {}
    for k in flips:
        a = np.divide(
            np.asarray(corrected[k], dtype=np.float64), raw, out=np.zeros(raw.shape), where=raw > 0
        )
        applied[k] = a
        median = float(np.median(a[counted]))
        keep &= (a > DEFECT_RANGE[0] * median) & (a < DEFECT_RANGE[1] * median)
    pixels = int(keep.sum())
    if pixels < 1000:
        raise ValueError(f"{label}: only {pixels} usable pixels")

    hp_raw = high_pass(_filled(raw, keep))[keep]
    hp_applied = {k: high_pass(_filled(applied[k], keep))[keep] for k in flips}
    r = {k: correlation(hp_raw, hp_applied[k]) for k in flips}
    flat = {
        k: float(np.std(high_pass(_filled(corrected[k], keep))[keep]))
        / float(np.mean(np.asarray(corrected[k], dtype=np.float64)[keep]))
        for k in flips
    }
    alike = correlation(hp_applied[0], hp_applied[1]) if {0, 1} <= set(flips) else math.nan
    as_documented: dict[int, float] = {}
    if 0 in flips:
        for k in flips:
            if k in FLIP_OF:
                both = keep & FLIP_OF[k](keep)
                as_documented[k] = correlation(applied[k][both], FLIP_OF[k](applied[0])[both])
    first = applied[flips[0]][keep]
    gain_spread = float(np.std(hp_applied[flips[0]])) / float(np.mean(first))
    verdict, flip = _verdict(r, flat, pixels, alike)
    return Judgement(
        label=label,
        pixels=pixels,
        left_out=int(interior.sum()) - pixels,
        r=r,
        flat=flat,
        alike=alike,
        as_documented=as_documented,
        gain_spread=gain_spread,
        verdict=verdict,
        flip=flip,
    )


def _verdict(
    r: dict[int, float], flat: dict[int, float], pixels: int, alike: float
) -> tuple[str, int | None]:
    se = 1.0 / math.sqrt(pixels)
    best = min(r, key=lambda k: r[k])
    flattest = min(flat, key=lambda k: flat[k])
    if not r[best] < -STRUCTURE_SE * se:
        return (
            f"inconclusive: no flip's gain anticorrelates with the counts "
            f"(strongest r {r[best]:+.4f}, needs < {-STRUCTURE_SE * se:+.4f})",
            None,
        )
    others = [r[k] - r[best] for k in r if k != best]
    if others and min(others) < MARGIN_SE * se:
        why = f"; A_0 and A_1 correlate {alike:.3f}" if not math.isnan(alike) else ""
        return (
            f"inconclusive: flips {sorted(r)} anticorrelate about equally "
            f"(margin {min(others):.4f}, needs {MARGIN_SE * se:.4f}){why}",
            None,
        )
    if best != flattest:
        return (
            f"split: the counts point to FlipGain {best}, the flattest sum is FlipGain {flattest}",
            None,
        )
    return f"FlipGain {best}", best


def read_sums(directory: Path) -> dict[str, tuple[NDArray[Any], dict[int, NDArray[Any]]]]:
    stems = sorted(path.name[: -len(NOGAIN)] for path in directory.glob(f"*{NOGAIN}"))
    if not stems:
        raise ValueError(f"no *{NOGAIN} sums in {directory}")
    sums: dict[str, tuple[NDArray[Any], dict[int, NDArray[Any]]]] = {}
    flip_sets = set()
    for stem in stems:
        raw = _read_mrc(directory / f"{stem}{NOGAIN}")
        corrected: dict[int, NDArray[Any]] = {}
        for path in sorted(directory.glob(f"{stem}_flip*.mrc")):
            tail = path.name[len(stem) + len("_flip") : -len(".mrc")]
            if tail.isdigit():
                corrected[int(tail)] = _read_mrc(path)
        if 0 not in corrected or len(corrected) < 2:
            raise ValueError(f"{stem}: needs {stem}_flip0.mrc and at least one other flip")
        flip_sets.add(tuple(sorted(corrected)))
        sums[stem] = (raw, corrected)
    if len(flip_sets) > 1:
        raise ValueError(f"the movies were summed at different flips: {sorted(flip_sets)}")
    return sums


def _read_mrc(path: Path) -> NDArray[np.float32]:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"{path}: expected one 2D sum, got shape {data.shape}")
    return data.copy()


def print_comparison(directory: Path) -> list[Judgement]:
    sums = read_sums(directory)
    judgements = [judge(stem, raw, corrected) for stem, (raw, corrected) in sums.items()]
    if len(sums) > 1:
        raws = [raw for raw, _ in sums.values()]
        flips = sorted(next(iter(sums.values()))[1])
        total = {
            k: np.sum([corrected[k] for _, corrected in sums.values()], axis=0, dtype=np.float64)
            for k in flips
        }
        judgements.append(
            judge(f"all {len(sums)} summed", np.sum(raws, axis=0, dtype=np.float64), total)
        )

    flips = sorted(judgements[0].r)
    print(f"sums in {directory}; high-pass {BOX}x{BOX}, border {BORDER} px")
    for j in judgements:
        print()
        print(f"{j.label}")
        se = 1 / math.sqrt(j.pixels)
        print(f"  pixels used        {j.pixels} (left out: {j.left_out}); 1/sqrt(n) = {se:.5f}")
        print(f"  gain spread        {j.gain_spread:.4f} (high-passed sd / mean of applied gain)")
        for k in flips:
            print(f"  FlipGain {k}         r {j.r[k]:+.5f}   flat {j.flat[k]:.6f}")
        if not math.isnan(j.alike):
            print(f"  A_0 vs A_1         {j.alike:+.4f} (1 = the gain looks the same upside down)")
        for k, value in j.as_documented.items():
            name = FLIP_OF[k].__name__
            print(f"  A_{k} vs A_0 {name:<7} {value:+.4f} (1 = -FlipGain {k} does what it says)")
        print(f"  verdict            {j.verdict}")
    print()
    final = judgements[-1]
    per_movie = [j for j in judgements if not j.label.startswith("all ")]
    agreeing = sum(j.flip == final.flip for j in per_movie) if final.flip is not None else 0
    print(f"prediction (sources): FlipGain {PREDICTION}")
    print(f"result:               {final.verdict}; per movie {agreeing}/{len(per_movie)} agree")
    return judgements


if __name__ == "__main__":
    raise SystemExit(main())
