#!/usr/bin/env python
"""Check that RELION 5 sees a particle where the pipeline's pose puts it.

The RELION export turns a particle the pipeline aligned (a box cut from the
canonical tomogram, Dynamo's angles and shift about N/2, as MRA records them)
into rlnCenteredCoordinate*Angst and rlnAngleRot/Tilt/Psi
(cryoet_pipeline.relion_export.relion_pose), and the class average into RELION's
frame (standard_frame_reference). RELION's own relion_tomo_reconstruct_particle
reconstructs the particle from the tilt images at the exported pose; the
pipeline's average_class, which makes the MRA class averages, averages the same
box at the same pose. The two must be the same map.

The check needs no class: it places `--count` particles on structured parts of
a tomogram (no beads), gives each a random orientation and a shift of up to 2
voxels, and compares each particle alone and all of them together, so a wrong
position, angle convention, mirror or box centre shows in every particle.

Prediction: for every particle and for all together, RELION's map is the
pipeline's average mirrored in z about int(N/2): of the eight axis reversals of
that map (each about int(N/2)), none (z0 y0 x0) correlates best, at least twice
the next, and their offset is at most 0.5 voxel. Anything else fails.

Usage, from the repository root:

    python scripts/relion_particle_check.py --project outputs/dev-ts01-registry \\
        --series TS_01 --out /tmp/relion-particles \\
        --relion ~/Developer/relion5/build/bin/relion_tomo_reconstruct_particle

Exit status 0 when every comparison passes, 1 when one fails, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import zarr
from scipy import ndimage

from cryoet_pipeline.backends.dynamo_search import (
    average_class,
    dynamo_angles_from_matrix,
    euler_matrix,
)
from cryoet_pipeline.relion_export import (
    RelionParticle,
    export_series,
    relion_pose,
    series_sources,
    standard_frame_reference,
    write_particles,
    write_tomogram_set,
)

MARGIN = 2.0
SHIFT_LIMIT = 0.5


def _band(volume: np.ndarray) -> np.ndarray:
    volume = volume - volume.mean()
    return ndimage.gaussian_filter(volume, 0.7) - ndimage.gaussian_filter(volume, 5.0)


def _sphere(box: int, radius: float) -> np.ndarray:
    centre = box // 2
    grid = np.indices((box, box, box)) - centre
    return (np.sqrt((grid**2).sum(axis=0)) <= radius).astype(float)


def _mirror(volume: np.ndarray, axis: int) -> np.ndarray:
    """Reverse one axis about int(N/2), as standard_frame_reference reverses z."""

    flipped = np.flip(volume, axis=axis)
    return np.roll(flipped, 1, axis=axis) if volume.shape[axis] % 2 == 0 else flipped


def choose_boxes(tomogram: np.ndarray, box: int, count: int) -> list[tuple[int, int, int]]:
    """Box starts (x, y, z) on the most structured parts of the tomogram, beads left out,
    at least a box apart."""

    shrunk = ndimage.zoom(tomogram, 0.5, order=1)
    band = _band(shrunk)
    sd = float(band.std())
    half = box // 2
    nz, ny, nx = tomogram.shape
    candidates = []
    for z in range(half + 2, nz - half - 2, max(box // 4, 1)):
        for y in range(half + 2, ny - half - 2, half):
            for x in range(half + 2, nx - half - 2, half):
                window = band[
                    (z - half) // 2 : (z + half) // 2,
                    (y - half) // 2 : (y + half) // 2,
                    (x - half) // 2 : (x + half) // 2,
                ]
                if window.size == 0 or np.abs(window).max() > 8 * sd:
                    continue
                candidates.append((float(window.std()), (x - half, y - half, z - half)))
    candidates.sort(reverse=True)
    chosen: list[tuple[int, int, int]] = []
    for _, start in candidates:
        if all(np.linalg.norm(np.subtract(start, other)) >= box for other in chosen):
            chosen.append(start)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        sys.exit(f"found room for {len(chosen)} particles, not {count}")
    return chosen


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    return q if np.linalg.det(q) > 0 else -q


def relion_reconstruct(
    relion: str, out: Path, star: Path, tag: str, box: int, threads: int, binning: int
) -> np.ndarray:
    folder = out / f"rec_{tag}"
    command = [
        relion,
        "--p",
        str(star.relative_to(out)),
        "--t",
        "tomograms.star",
        "--o",
        f"{folder.name}/",
        "--b",
        str(box),
        "--bin",
        str(binning),
        "--no_ctf",
        "--j",
        str(threads),
    ]
    result = subprocess.run(command, cwd=out, capture_output=True, text=True)
    (out / f"relion_{tag}.log").write_text(
        " ".join(command) + "\n\n" + result.stdout + result.stderr
    )
    # The backprojected data, not merged.mrc: merged is the data divided by its Fourier
    # weights, and a few particles' weights are a few planes, so the division is mostly
    # noise (seen on a gold bead of TS_01: data_merged holds the bead at the centre,
    # merged only streaks). Negated: reconstruct_particle inverts the contrast
    # (flip_value is true in RELION 5.0).
    target = folder / "data_merged.mrc"
    if result.returncode != 0 or not target.is_file():
        sys.exit(
            f"RELION did not reconstruct {tag} (exit {result.returncode}); "
            f"see {out / f'relion_{tag}.log'}"
        )
    return -mrcfile.read(target).astype(np.float64)


def peak_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cc = np.fft.fftshift(
        np.fft.irfftn(np.fft.rfftn(b) * np.conj(np.fft.rfftn(a)), s=a.shape, axes=(0, 1, 2))
    )
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    shift = []
    for axis, position in enumerate(peak):
        below, above = list(peak), list(peak)
        below[axis] = (position - 1) % cc.shape[axis]
        above[axis] = (position + 1) % cc.shape[axis]
        y0, y1, y2 = cc[tuple(below)], cc[peak], cc[tuple(above)]
        denominator = y0 - 2 * y1 + y2
        shift.append(
            position - a.shape[axis] // 2 + (0.5 * (y0 - y2) / denominator if denominator else 0.0)
        )
    return np.array(shift)


def compare(ours_standard: np.ndarray, relion: np.ndarray, mask: np.ndarray) -> dict:
    target = _band(relion) * mask
    scores = {}
    for flips in itertools.product((False, True), repeat=3):
        candidate = ours_standard
        for axis, flip in enumerate(flips):
            if flip:
                candidate = _mirror(candidate, axis)
        candidate = _band(candidate) * mask
        name = "z{} y{} x{}".format(*(int(flip) for flip in flips))
        scores[name] = float(
            (candidate * target).sum() / np.sqrt((candidate**2).sum() * (target**2).sum())
        )
    shift = peak_shift(_band(ours_standard) * mask, target)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    reasons = []
    if ranked[0][0] != "z0 y0 x0":
        reasons.append(f"best is {ranked[0][0]}")
    if ranked[0][1] < MARGIN * max(ranked[1][1], 1e-6):
        reasons.append(f"best NCC {ranked[0][1]:.3f} not {MARGIN:g}x the next {ranked[1][1]:.3f}")
    if np.abs(shift).max() > SHIFT_LIMIT:
        reasons.append(f"offset {np.round(shift, 2).tolist()} voxel")
    return {
        "ncc": scores,
        "shift_zyx_voxel": [round(float(v), 3) for v in shift],
        "passed": not reasons,
        "reasons": reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--tomogram-id", help="default <series>:tomogram:fine")
    parser.add_argument("--out", type=Path, required=True, help="a new folder")
    parser.add_argument(
        "--relion",
        default=os.environ.get("RELION_RECONSTRUCT_PARTICLE")
        or shutil.which("relion_tomo_reconstruct_particle"),
        help="relion_tomo_reconstruct_particle (RELION 5)",
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--box", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    )
    parser.add_argument(
        "--stack-binning",
        type=int,
        help="bin the tilt images RELION reads this much (default: as much as the tomogram)",
    )
    args = parser.parse_args(argv)
    if not args.relion:
        parser.error("no relion_tomo_reconstruct_particle: give --relion")
    if args.box < 8 or args.box % 2:
        parser.error("--box must be even, as MRA's boxes are")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    try:
        sources = series_sources(args.project, args.series, args.tomogram_id)
        exported = export_series(sources, args.out, stack_binning=args.stack_binning)
    except (KeyError, ValueError) as error:
        sys.exit(f"cannot export {args.series}: {error}")
    write_tomogram_set(
        args.out,
        {
            args.series: (
                exported.views,
                exported.placement.size_xyz(),
                exported.placement.stack_pixel_angstrom,
            )
        },
        optics=exported.optics,
        hand=-1,
    )
    tomogram = np.asarray(zarr.open(str(sources.tomogram.path), mode="r"), dtype=np.float32)
    tilts = [view.tilt.ytilt_deg for view in exported.views]
    tilt_range = (min(tilts), max(tilts))

    rng = np.random.default_rng(args.seed)
    box = args.box
    starts = choose_boxes(tomogram, box, args.count)
    mask = _sphere(box, box / 2 - 2)
    particles, boxes, angles, shifts = [], [], [], []
    for number, (x, y, z) in enumerate(starts, start=1):
        matrix = random_rotation(rng)
        dynamo = dynamo_angles_from_matrix(matrix.T)
        assert np.allclose(euler_matrix(dynamo).T, matrix, atol=1e-9)
        shift = rng.uniform(-2.0, 2.0, size=3)
        centred, relion_angles = relion_pose(
            box_start_xyz=(x, y, z),
            box=box,
            dynamo_angles_deg=dynamo,
            dynamo_shift_px=shift,
            placement=exported.placement,
        )
        particles.append(
            RelionParticle(
                tomogram=args.series,
                number=number,
                centred_angstrom=tuple(centred),
                angles_deg=relion_angles,
            )
        )
        boxes.append(np.asarray(tomogram[z : z + box, y : y + box, x : x + box], dtype=np.float64))
        angles.append(dynamo)
        shifts.append(shift)

    results = {}
    runs = [(f"p{k + 1}", [k]) for k in range(len(particles))] + [
        ("all", list(range(len(particles))))
    ]
    for tag, members in runs:
        star = write_particles(args.out / f"particles_{tag}.star", [particles[k] for k in members])
        ours = average_class(
            [boxes[k] for k in members],
            [angles[k] for k in members],
            [shifts[k] for k in members],
            [tilt_range] * len(members),
            mask,
        )
        relion = relion_reconstruct(
            args.relion, args.out, star, tag, box, args.threads, exported.placement.binning
        )
        result = compare(standard_frame_reference(ours.raw), relion, mask)
        result["box_starts_xyz"] = [list(starts[k]) for k in members]
        results[tag] = result

    passed = all(result["passed"] for result in results.values())
    report = {
        "series": args.series,
        "tomogram_id": sources.tomogram.id,
        "box": box,
        "seed": args.seed,
        "notes": list(exported.notes),
        "results": results,
        "passed": passed,
    }
    (args.out / "particle_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.series} ({sources.tomogram.id}): {len(particles)} particles, box {box}, "
        f"{exported.placement.stack_pixel_angstrom:.3f} A"
    )
    for tag, result in results.items():
        ranked = sorted(result["ncc"].items(), key=lambda item: item[1], reverse=True)
        shift = result["shift_zyx_voxel"]
        print(
            f"  {tag:4s} best {ranked[0][0]} {ranked[0][1]:+.3f}, next {ranked[1][0]} "
            f"{ranked[1][1]:+.3f}; offset (z, y, x) {shift[0]:+.2f} {shift[1]:+.2f} "
            f"{shift[2]:+.2f}; "
            + ("pass" if result["passed"] else "FAIL: " + "; ".join(result["reasons"]))
        )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
