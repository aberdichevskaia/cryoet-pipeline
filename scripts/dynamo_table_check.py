#!/usr/bin/env python3
"""Measure where a Dynamo table puts a particle in the tomogram Dynamo read.

docs/entry_and_exit.md lists Dynamo's table among the pick formats: positions in
columns 24-26 and shifts in 4-6, angles tdrot, tilt, narot. As a particle
folder, the pipeline's port of the MRA matches Dynamo. As positions in a
tomogram, the table is to be measured against Dynamo's own cropping before
import-picks offers it. This script is that measurement, with Dynamo itself.

Read first from Dynamo's source (dynamo_table_crop.m, 1.1.591): dtcrop crops
at the position alone, a box of side L whose 1-based indices run from
c - L/2 to c + L/2 - 1, c = ceil(position) ("positions are supposed to be
expressed in semiintegers"). Measured on the installed 1.1.511, 16 November
2026: c = floor(position) + 1, which differs from ceil only for whole positions
(20.0 is cropped as 20.5 is). So Dynamo's box centre (L/2 + 1, 1-based) is our
0-based voxel floor(p), the voxel that holds p when voxel i spans [i, i + 1):
for the semi-integers Dynamo documents, p - 0.5. Its averaging puts a particle
at that centre plus the shift (the sign measured below), so a table row is our
voxel floor(p) + shift.

Two parts, run with the installed Dynamo:

1. **Positions.** A volume of noise, every voxel distinct, written as an MRC from
   a (z, y, x) array; a table of integer, semi-integer and other fractional
   positions; `dtcrop` with L = 16. Each crop is found in the volume (exactly, up to
   dtcrop's normalisation of each particle to mean 0 and sd 1), and
   its first voxel set against floor(p) - L/2 (0-based), the voxel holding p,
   and against the source's ceil(p) - L/2 - 1 and round(p) - L/2.
2. **Shifts.** One Gaussian blob, cropped at positions moved off it by integer
   amounts d, and averaged by `daverage` with shifts +d, then -d. The sign under
   which the average is the blob, sharp, at the box centre, is the convention.

    python scripts/dynamo_table_check.py --work <scratch folder> \\
        [--dynamo ~/Downloads/dynamo-v-1.1.511_MCR-9.8.0_MACI64_withMCR]

Exit status 0 when both predictions hold, 1 when either does not.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np

from cryoet_pipeline.reports.dynamo_project import read_em

SIDE = 16
SHAPE = (40, 56, 72)  # z, y, x
POSITIONS = [  # x, y, z as Dynamo reads them
    (20.5, 25.5, 18.5), (20.0, 25.0, 18.0), (20.25, 25.75, 18.49), (33.5, 30.5, 21.5),
    (40.51, 28.99, 19.01), (50.5, 40.5, 25.5), (13.5, 12.5, 11.5), (59.0, 43.0, 28.0),
]
BLOB_XYZ = (36, 28, 20)  # 0-based voxel of the blob's centre
OFFSETS = [(0, 0, 0), (2, 0, 0), (0, -3, 0), (0, 0, 2), (-2, 1, -1), (3, -2, 1)]


def table_rows(positions: list[tuple[float, float, float]],
               shifts: list[tuple[float, float, float]]) -> str:
    lines = []
    for tag, (position, shift) in enumerate(zip(positions, shifts, strict=True), start=1):
        row = [0.0] * 35
        row[0], row[1], row[2] = tag, 1, 1
        row[3:6] = shift
        row[12] = 0  # ftype 0: no missing wedge
        row[13:17] = [-90, 90, -90, 90]
        row[19] = 1  # tomogram
        row[21] = 1  # class
        row[23:26] = position
        row[34] = 1.0  # pixel size
        lines.append(" ".join(f"{value:g}" for value in row))
    return "\n".join(lines) + "\n"


def write_mrc(path: Path, data: np.ndarray) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data, dtype=np.float32))
        mrc.voxel_size = 1.0


def run_dynamo(dynamo: Path, work: Path, script: str) -> str:
    (work / "check.m").write_text(script)
    command = (f"source '{dynamo}/dynamo_activate_mac_shipped_MCR.sh' > /dev/null 2>&1; "
               f"cd '{work}'; dynamo")
    result = subprocess.run(["bash", "-c", command], input="eval(fileread('check.m'))\nexit\n",
                            capture_output=True, text=True, timeout=1800)
    (work / "dynamo.log").write_text(result.stdout + result.stderr)
    return result.stdout + result.stderr


def particle_file(folder: Path, tag: int) -> Path:
    """Dynamo 1.1.511 names particles with six digits; its documentation says five."""

    for digits in (6, 5):
        path = folder / f"particle_{tag:0{digits}d}.em"
        if path.is_file():
            return path
    raise FileNotFoundError(f"{folder} has no particle {tag}")


def find(volume: np.ndarray, crop: np.ndarray) -> tuple[int, int, int] | None:
    """The first voxel (z, y, x) where `crop` is a part of `volume`, up to dtcrop's
    normalisation of each particle to mean 0 and standard deviation 1."""

    padded = np.zeros_like(volume)
    padded[: crop.shape[0], : crop.shape[1], : crop.shape[2]] = crop - crop.mean()
    centred = volume - volume.mean()
    score = np.real(np.fft.ifftn(np.fft.fftn(centred) * np.conj(np.fft.fftn(padded))))
    z, y, x = np.unravel_index(np.argmax(score), score.shape)
    part = volume[z:z + crop.shape[0], y:y + crop.shape[1], x:x + crop.shape[2]]
    if part.shape != crop.shape:
        return None
    correlation = float(np.corrcoef(part.ravel(), crop.ravel())[0, 1])
    return (int(z), int(y), int(x)) if correlation > 1 - 1e-6 else None


def positions_part(work: Path, dynamo: Path) -> list[str]:
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(SHAPE).astype(np.float32)
    write_mrc(work / "noise.mrc", noise)
    (work / "positions.tbl").write_text(table_rows(POSITIONS, [(0.0, 0.0, 0.0)] * len(POSITIONS)))
    output = run_dynamo(dynamo, work, (
        "try\n"
        f"  dtcrop('noise.mrc', 'positions.tbl', 'crops_noise', {SIDE});\n"
        "  disp('CROPPED');\n"
        "catch e\n  disp(getReport(e, 'extended'));\nend\n"
    ))
    if "CROPPED" not in output:
        return [f"dtcrop failed; see {work / 'dynamo.log'}"]
    failures = []
    rules = {
        "floor(p) - L/2, the voxel holding p": lambda p: math.floor(p) - SIDE // 2,
        "ceil(p) - L/2 - 1, from the 1.1.591 source": lambda p: math.ceil(p) - SIDE // 2 - 1,
        "round(p) - L/2": lambda p: round(p) - SIDE // 2,
    }
    hits = {rule: 0 for rule in rules}
    print(f"\npositions: dtcrop of {len(POSITIONS)} boxes of {SIDE} from a {SHAPE} volume")
    for tag, position in enumerate(POSITIONS, start=1):
        crop = read_em(particle_file(work / "crops_noise", tag))
        found = find(noise, crop)
        if found is None:
            failures.append(f"crop {tag} is not a part of the volume")
            continue
        start_xyz = (found[2], found[1], found[0])
        marks = []
        for rule, rounding in rules.items():
            predicted = tuple(int(rounding(p)) for p in position)
            if predicted == start_xyz:
                hits[rule] += 1
                marks.append(rule.split(",")[0])
        print(f"  p = {position}: first voxel (x, y, z) {start_xyz}, matches: "
              f"{', '.join(marks) or 'none'}")
    for rule, count in hits.items():
        print(f"  {rule}: {count} of {len(POSITIONS)}")
    if hits["floor(p) - L/2, the voxel holding p"] != len(POSITIONS):
        failures.append("dtcrop does not centre its box on the voxel holding p")
    return failures


def blob_centre(volume: np.ndarray) -> np.ndarray:
    weights = np.clip(volume - np.median(volume), 0, None)
    grid = np.indices(volume.shape).astype(float)
    centre = [(grid[axis] * weights).sum() / weights.sum() for axis in range(3)]
    return np.array([centre[2], centre[1], centre[0]])  # x, y, z


def shifts_part(work: Path, dynamo: Path) -> list[str]:
    grid = np.indices(SHAPE).astype(float)
    x0, y0, z0 = BLOB_XYZ
    blob = np.exp(-((grid[0] - z0) ** 2 + (grid[1] - y0) ** 2 + (grid[2] - x0) ** 2)
                  / (2 * 1.5**2))
    write_mrc(work / "blob.mrc", blob)
    # A crop at p - d holds the blob d off its centre.
    positions = [(x0 + 0.5 - dx, y0 + 0.5 - dy, z0 + 0.5 - dz) for dx, dy, dz in OFFSETS]
    plus = [tuple(float(value) for value in offset) for offset in OFFSETS]
    minus = [tuple(-float(value) for value in offset) for offset in OFFSETS]
    zero = [(0.0, 0.0, 0.0)] * len(OFFSETS)
    (work / "blob_positions.tbl").write_text(table_rows(positions, zero))
    (work / "blob_plus.tbl").write_text(table_rows(positions, plus))
    (work / "blob_minus.tbl").write_text(table_rows(positions, minus))
    output = run_dynamo(dynamo, work, (
        "try\n"
        f"  dtcrop('blob.mrc', 'blob_positions.tbl', 'crops_blob', {SIDE});\n"
        "  o = daverage('crops_blob', 't', 'blob_plus.tbl');\n"
        "  dwrite(o.average, 'average_plus.em');\n"
        "  o = daverage('crops_blob', 't', 'blob_minus.tbl');\n"
        "  dwrite(o.average, 'average_minus.em');\n"
        "  disp('AVERAGED');\n"
        "catch e\n  disp(getReport(e, 'extended'));\nend\n"
    ))
    if "AVERAGED" not in output:
        return [f"daverage failed; see {work / 'dynamo.log'}"]
    centre = np.full(3, SIDE // 2, dtype=float)  # Dynamo's box centre, 0-based
    print(f"\nshifts: {len(OFFSETS)} crops of one blob, moved off it by d, averaged by Dynamo")
    results = {}
    for name in ("plus", "minus"):
        average = read_em(work / f"average_{name}.em")
        found = blob_centre(average)
        peak = float(average.max())
        results[name] = (found, peak)
        print(f"  shifts {'+d' if name == 'plus' else '-d'}: blob at (x, y, z) "
              f"{np.round(found, 2)}, box centre {centre}, peak {peak:.3f}")
    off_plus = float(np.linalg.norm(results["plus"][0] - centre))
    off_minus = float(np.linalg.norm(results["minus"][0] - centre))
    if not (off_plus < 0.25 and off_plus < off_minus and results["plus"][1] > results["minus"][1]):
        return ["the +d average is not the sharp blob at the box centre"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, help="Scratch folder; its contents are replaced.")
    parser.add_argument("--dynamo", default=str(Path.home() / "Downloads" /
                                                "dynamo-v-1.1.511_MCR-9.8.0_MACI64_withMCR"))
    args = parser.parse_args(argv)
    work, dynamo = Path(args.work).resolve(), Path(args.dynamo).expanduser().resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    failures = positions_part(work, dynamo) + shifts_part(work, dynamo)
    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASSED: dtcrop centres its box on our voxel floor(p), and Dynamo averages a particle "
          "at that centre plus the shift, so a table row is our voxel floor(p) + shift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
