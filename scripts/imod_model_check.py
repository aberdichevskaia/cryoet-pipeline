#!/usr/bin/env python3
"""Measure how an IMOD model's points sit on the volume they were made on.

docs/entry_and_exit.md lists IMOD models among the pick formats as "model2point:
pixels of the volume it was made on, in that volume's orientation", to be
measured before it is offered. The prediction, from IMOD's model format as
`picks_visualization` states it (a pixel's centre at index + 0.5 in X and Y, a
section's at its index in Z): a point (X, Y, Z) of `model2point -float`'s default
output is voxel (X - 0.5, Y - 0.5, Z) of the volume the model was made on.

Three parts.

1. **Known centres** (pass/fail). Dark Gaussian beads at exact, partly fractional
   centres, in one canonical volume written in IMOD's three layouts: z as
   sections (-PARALLEL's), z reversed (the standard orientation), and Y as
   sections (tilt's own output; `clip rotx` of it must equal the second). IMOD's
   `findbeads3d` models them (`-ylong` for the last), `model2point` reads the
   models, and the points, mapped through each layout to the canonical volume,
   are set against the known centres, at two bead sizes. The prediction holds when
   each axis's mean is within `TOLERANCE_PX`; the readings without the half pixel,
   or with it on Z too, must miss by about half a voxel on those axes.
2. **Round trip** (pass/fail). `point2model` and then `model2point` give back the
   numbers they were given, so the model2point output is what point2model takes,
   the form whose placement `imodmop` confirmed for `picks_visualization`.
3. **A tomogram's own beads** (informational, with `--rec`). findbeads3d on the
   series' reconstruction in the three layouts, set against each bead's density
   minimum. On EMPIAR-10164 TS_01 at bin8 on 16 November, findbeads3d's
   centres there differed from the minima by 0.4 to 0.9 voxel, depending on the
   layout. That is findbeads3d's own centring on real beads, not the format: the
   part shows it, and does not judge.

    python scripts/imod_model_check.py --work <scratch folder> \\
        [--rec outputs/dev-ts01-registry/tomograms/TS_01/TS_01_fine_bin8.rec \\
         --tilt-file outputs/dev-ts01-registry/tomograms/TS_01/TS_01_fine_bin8.tlt \\
         --bead-px 18.5]

Exit status 0 when parts 1 and 2 hold, 1 when either does not.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import mrcfile
import numpy as np

MATCH_PX = 3.0
TOLERANCE_PX = 0.2
READINGS = {"(X-0.5, Y-0.5, Z), the prediction": (0.5, 0.0), "no half pixel": (0.0, 0.0),
            "half a voxel on Z too": (0.5, 0.5)}


def run(command: list[str], imod: Path) -> str:
    environment = dict(os.environ)
    environment["IMOD_DIR"] = str(imod)
    environment["PATH"] = f"{imod / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def write_mrc(path: Path, data: np.ndarray, voxel: float = 5.4) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data, dtype=np.float32))
        mrc.voxel_size = voxel


def find_beads(volume: Path, size: float, imod: Path, extra: list[str]) -> np.ndarray:
    model = volume.with_suffix(".mod")
    run([str(imod / "bin" / "findbeads3d"), "-input", str(volume), "-output", str(model),
         "-size", f"{size:g}", *extra], imod)
    return model_points(model, imod)


def model_points(model: Path, imod: Path) -> np.ndarray:
    text = model.with_suffix(".txt")
    run([str(imod / "bin" / "model2point"), "-float", "-input", str(model), "-output",
         str(text)], imod)
    return np.loadtxt(text, ndmin=2)[:, :3]


LAYOUTS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], list[str]]] = {
    "imod-parallel": (lambda canonical: canonical, []),
    "imod-standard": (lambda canonical: canonical[::-1], []),
    "imod-tilt": (lambda canonical: canonical.transpose(1, 0, 2), ["-ylong"]),
}


def to_canonical(points: np.ndarray, layout: str, depth: int, xy: float, z: float) -> np.ndarray:
    """model2point's (X, Y, Z) of a volume in `layout`, as canonical (x, y, z) voxels."""

    column, row, section = points[:, 0] - xy, points[:, 1] - xy, points[:, 2] - z
    if layout == "imod-parallel":
        return np.column_stack([column, row, section])
    if layout == "imod-standard":
        return np.column_stack([column, row, (depth - 1) - section])
    # tilt's output: rows are the thickness (canonical z), sections canonical y.
    return np.column_stack([column, section, row])


def offsets(found: np.ndarray, truth: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    differences = []
    for point in truth:
        if not np.all(np.isfinite(point)):
            continue
        distance = np.linalg.norm(found - point, axis=1)
        if distance.min() < MATCH_PX:
            differences.append(found[np.argmin(distance)] - point)
    values = np.array(differences).reshape(-1, 3)
    if not len(values):
        return 0, np.full(3, np.nan), np.full(3, np.nan)
    return len(values), values.mean(axis=0), values.std(axis=0)


def beads(size: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A canonical volume of dark beads, and their exact centres (x, y, z)."""

    step = int(3 * size)
    depth, height, width = 96, 5 * step, 6 * step
    fractions = (0.0, 0.25, 0.5, 0.75)
    centres = np.array([
        [step * (ix + 1) - step // 2 + fractions[ix % 4],
         step * (iy + 1) - step // 2 + fractions[iy % 4],
         (32, 48, 64)[(ix + iy) % 3] + fractions[(ix + 2 * iy) % 4]]
        for iy in range(4) for ix in range(5)
    ])
    grid = np.indices((depth, height, width)).astype(float)
    volume = rng.normal(0.0, 0.05, (depth, height, width))
    sigma = size / 4
    for x, y, z in centres:
        volume -= np.exp(-((grid[0] - z) ** 2 + (grid[1] - y) ** 2 + (grid[2] - x) ** 2)
                         / (2 * sigma**2))
    return volume.astype(np.float32), centres


def known_centres(work: Path, imod: Path) -> list[str]:
    failures = []
    rng = np.random.default_rng(3)
    for size in (10.0, 18.5):
        canonical, truth = beads(size, rng)
        depth = canonical.shape[0]
        print(f"\nbeads of {size:g} px at known centres, {len(truth)} of them")
        for layout, (lay_out, extra) in LAYOUTS.items():
            path = work / f"known_{layout}_{size:g}.mrc"
            write_mrc(path, lay_out(canonical))
            points = find_beads(path, size, imod, ["-store", "0.3", *extra])
            for reading, (xy, z) in READINGS.items():
                count, mean, spread = offsets(to_canonical(points, layout, depth, xy, z), truth)
                print(f"  {layout:14s} {reading:36s} {count:2d} matched, model minus truth "
                      f"x {mean[0]:+.3f} y {mean[1]:+.3f} z {mean[2]:+.3f} (sd {spread[0]:.2f} "
                      f"{spread[1]:.2f} {spread[2]:.2f})")
                if reading.endswith("prediction"):
                    if count < 15:
                        failures.append(f"{layout} at {size:g} px: {count} beads matched")
                    elif np.any(np.abs(mean) > TOLERANCE_PX):
                        failures.append(f"{layout} at {size:g} px: off by {mean.round(3)}")
        tilt_layout = work / f"known_imod-tilt_{size:g}.mrc"
        rotated = work / f"known_rotx_{size:g}.mrc"
        run([str(imod / "bin" / "clip"), "rotx", str(tilt_layout), str(rotated)], imod)
        with mrcfile.open(rotated, permissive=True) as mrc:
            same = np.array_equal(np.asarray(mrc.data), canonical[::-1])
        print(f"  clip rotx of tilt's layout is the canonical volume reversed in z: {same}")
        if not same:
            failures.append(f"clip rotx at {size:g} px is not the standard layout")
    return failures


def round_trip(work: Path, imod: Path) -> list[str]:
    given = np.array([[10.25, 20.5, 3.0], [0.5, 0.5, 0.0], [99.75, 63.125, 41.5]])
    points = work / "round_trip_given.txt"
    points.write_text("".join(f"{x} {y} {z}\n" for x, y, z in given))
    model = work / "round_trip.mod"
    run([str(imod / "bin" / "point2model"), "-input", str(points), "-output", str(model)], imod)
    back = model_points(model, imod)
    worst = float(np.max(np.abs(back - given)))
    # model2point prints two decimals, so 63.125 comes back as 63.12.
    print(f"\npoint2model then model2point: largest change {worst:.4g} (two decimals printed)")
    return [] if worst <= 0.006 else [f"the round trip changes the points by {worst:.4g}"]


def gaussian_smooth(volume: np.ndarray, sigma: float) -> np.ndarray:
    smoothed = volume.astype(np.float64)
    for axis in range(3):
        size = smoothed.shape[axis]
        frequencies = np.fft.rfftfreq(size)
        shape = [1, 1, 1]
        shape[axis] = len(frequencies)
        kernel = np.exp(-2 * (np.pi * sigma * frequencies) ** 2).reshape(shape)
        smoothed = np.fft.irfft(np.fft.rfft(smoothed, axis=axis) * kernel, n=size, axis=axis)
    return smoothed


def density_minima(volume: np.ndarray, guesses: np.ndarray, bead_px: float) -> np.ndarray:
    """Each bead's smoothed minimum (x, y, z) within half a diameter of a guess."""

    half = int(np.ceil(1.5 * bead_px))
    grid = np.indices((2 * half + 1,) * 3) - half
    near = np.sqrt(np.square(grid).sum(axis=0)) <= np.ceil(bead_px / 2)
    found = []
    for x, y, z in guesses:
        start = np.rint([z, y, x]).astype(int) - half
        stop = start + 2 * half + 1
        if np.any(start < 0) or np.any(stop > np.asarray(volume.shape)):
            found.append([np.nan] * 3)
            continue
        box = gaussian_smooth(volume[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]],
                              bead_px / 4)
        k, j, i = np.unravel_index(np.argmin(np.where(near, box, np.inf)), box.shape)
        refined = []
        for axis, index in enumerate((k, j, i)):
            step = np.eye(3, dtype=int)[axis]
            minus = box[k - step[0], j - step[1], i - step[2]]
            plus = box[k + step[0], j + step[1], i + step[2]]
            curvature = minus - 2 * box[k, j, i] + plus
            refined.append(index + (0.5 * (minus - plus) / curvature if curvature > 0 else 0.0))
        found.append([start[2] + refined[2], start[1] + refined[1], start[0] + refined[0]])
    return np.array(found)


def tomogram_beads(args: argparse.Namespace, work: Path, imod: Path) -> None:
    rec = Path(args.rec).resolve()
    with mrcfile.open(rec, permissive=True) as mrc:
        canonical = np.asarray(mrc.data, dtype=np.float32).transpose(1, 0, 2)
    depth = canonical.shape[0]
    volumes = {"imod-parallel": work / "tomogram_parallel.mrc",
               "imod-standard": work / "tomogram_standard.mrc", "imod-tilt": rec}
    write_mrc(volumes["imod-parallel"], canonical)
    run([str(imod / "bin" / "clip"), "rotx", str(rec), str(volumes["imod-standard"])], imod)
    print(f"\n{rec.name}: findbeads3d's beads against their density minima (informational)")
    for layout, volume in volumes.items():
        model = work / f"tomogram_{layout}.mod"
        command = [str(imod / "bin" / "findbeads3d"), "-input", str(volume), "-output",
                   str(model), "-size", f"{args.bead_px:g}", "-tilt",
                   str(Path(args.tilt_file).resolve())]
        if layout == "imod-tilt":
            command.append("-ylong")
        run(command, imod)
        mapped = to_canonical(model_points(model, imod), layout, depth, 0.5, 0.0)
        minima = density_minima(canonical, mapped, args.bead_px)
        count, mean, spread = offsets(mapped, minima)
        print(f"  {layout:14s} {count:2d} beads, findbeads3d minus minimum x {mean[0]:+.3f} "
              f"y {mean[1]:+.3f} z {mean[2]:+.3f} (sd {spread[0]:.2f} {spread[1]:.2f} "
              f"{spread[2]:.2f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, help="Scratch folder; its files are replaced.")
    parser.add_argument("--rec", help="Optional: tilt's output of a series with gold beads.")
    parser.add_argument("--tilt-file", help="With --rec: the .tlt it was reconstructed with.")
    parser.add_argument("--bead-px", type=float, default=18.5, help="With --rec: bead size.")
    parser.add_argument("--imod-dir", default=os.environ.get("IMOD_DIR", "/Applications/IMOD"))
    args = parser.parse_args(argv)
    imod, work = Path(args.imod_dir), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    version = imod / "VERSION"
    print(f"IMOD {version.read_text().strip() if version.exists() else '(version unknown)'} "
          f"at {imod}")
    failures = known_centres(work, imod) + round_trip(work, imod)
    if args.rec:
        if not args.tilt_file:
            parser.error("--rec needs --tilt-file")
        tomogram_beads(args, work, imod)
    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"PASSED: model2point's (X - 0.5, Y - 0.5, Z) is the voxel of the volume the model "
          f"was made on, in each of IMOD's three layouts, to within {TOLERANCE_PX} voxel; "
          "point2model and model2point are each other's inverse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
