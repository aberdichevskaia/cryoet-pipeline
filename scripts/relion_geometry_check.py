#!/usr/bin/env python
"""Check that RELION 5 rebuilds a pipeline tomogram from the exported geometry.

The tilt geometry the RELION export writes (cryoet_pipeline.relion_export) is
the base of everything RELION will do with the particles, so it is checked by
RELION itself, not by the code that wrote it: RELION's own
relion_tomo_reconstruct_tomogram reconstructs the series from the exported
tomograms.star at the tomogram's binning, and the result is compared with the
pipeline's tomogram.

Prediction: RELION's volume is the pipeline's tomogram reversed in z (the
pipeline's canonical volume is IMOD's standard orientation mirrored, and the
export gives RELION the standard one). It passes when that reversal is the best
of the eight, by at least three times the next correlation, and the shift
between the two in 128-voxel boxes is at most 0.25 voxel at the centre and 0.5
in the four quarter positions. Anything else fails, and the table says how.

On 9 November 2026, EMPIAR-10164 TS_01 (outputs/dev-ts01-registry, bin8) with
RELION 5.0 (7f45a71) built on a Mac: NCC 0.68 for the predicted reversal, 0.03
for the next; shift 0.00 voxel at the centre, at most 0.07 in the quarters.

Usage, from the repository root:

    python scripts/relion_geometry_check.py --project outputs/dev-ts01-registry \\
        --series TS_01 --out /tmp/relion-check \\
        --relion ~/Developer/relion5/build/bin/relion_tomo_reconstruct_tomogram

The folder it writes is a RELION project of one tilt series (tomograms.star,
tilt_series/), with the raw tilt images binned as the final stack binned them.
Exit status 0 when the check passes, 1 when it fails, 2 on a usage error.
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

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import write_binned_mrc_stack
from cryoet_pipeline.models import Artifact, ArtifactKind, TiltAlignment, TiltSeriesManifest
from cryoet_pipeline.relion_export import (
    ImodReconstruction,
    RelionOptics,
    RelionView,
    relion_tilt,
    write_tomogram_set,
)

BOX = 128
QUARTERS = ("centre", "y- x-", "y- x+", "y+ x-", "y+ x+")
CENTRE_LIMIT = 0.25
QUARTER_LIMIT = 0.5
MARGIN = 3.0


def _ancestors(registry: ArtifactRegistry, artifact: Artifact) -> list[Artifact]:
    seen: dict[str, Artifact] = {}
    queue = list(artifact.parent_ids)
    while queue:
        parent_id = queue.pop(0)
        if parent_id in seen:
            continue
        try:
            parent = registry.get(parent_id)
        except KeyError:
            continue
        seen[parent_id] = parent
        queue.extend(parent.parent_ids)
    return list(seen.values())


def _one(found: list[Artifact], what: str) -> Artifact:
    if len(found) != 1:
        names = ", ".join(item.id for item in found) or "none"
        sys.exit(f"expected one {what} among the tomogram's ancestors, found {names}")
    return found[0]


def _fine(artifact: Artifact) -> bool:
    if artifact.kind is not ArtifactKind.ALIGNMENT:
        return False
    try:
        return TiltAlignment.model_validate_json(artifact.path.read_text()).stage == "fine"
    except (OSError, ValueError):
        return False


def series_inputs(project: Path, series: str, tomogram_id: str | None):
    registry = ArtifactRegistry.load(project / "artifacts.json")
    tomogram = registry.get(tomogram_id or f"{series}:tomogram:fine")
    ancestors = _ancestors(registry, tomogram)
    alignment = _one([a for a in ancestors if _fine(a)], "fine alignment")
    raw = _one([a for a in ancestors if a.kind is ArtifactKind.TILT_STACK], "raw tilt stack")
    final = [
        a
        for a in ancestors
        if a.kind is ArtifactKind.ALIGNED_TILT_STACK
        and a.parameters.get("purpose") == "final_alignment"
    ]
    final_stack = _one(final, "final aligned stack")
    manifest = TiltSeriesManifest.model_validate_json(
        (project / "manifests" / f"{series}.json").read_text()
    )
    return tomogram, alignment, raw, final_stack, manifest


def build(project: Path, series: str, tomogram_id: str | None, out: Path) -> dict:
    tomogram, alignment_artifact, raw, final_stack, manifest = series_inputs(
        project, series, tomogram_id
    )
    solved = TiltAlignment.model_validate_json(alignment_artifact.path.read_text())
    binning = int(tomogram.binning or final_stack.binning)
    raw_px = float(final_stack.parameters["raw_pixel_spacing_angstrom"])
    pixel = raw_px * binning
    included = [
        int(z)
        for z in tomogram.parameters.get("included_z_values")
        or [t.z_value for t in solved.transforms]
    ]
    transforms = [t for t in solved.transforms if t.z_value in set(included)]
    index_by_z = {image.z_value: i for i, image in enumerate(manifest.images)}
    tilt_series = out / "tilt_series"
    tilt_series.mkdir(parents=True, exist_ok=True)
    stack = tilt_series / f"{series}.mrcs"
    write_binned_mrc_stack(
        raw.path,
        stack,
        binning=binning,
        pixel_spacing_angstrom=pixel,
        section_indices=[index_by_z[t.z_value] for t in transforms],
    )
    with mrcfile.open(stack, permissive=True, header_only=True) as header:
        image_xy = (int(header.header.nx), int(header.header.ny))
    nz, ny, nx = (int(s) for s in tomogram.shape)
    reconstruction = ImodReconstruction(
        size_xyz=(nx, ny, nz),
        x_axis_tilt_deg=float(tomogram.parameters.get("x_axis_tilt_deg", 0.0)),
        z_shift_px=float(tomogram.parameters.get("z_shift_px", 0.0)),
    )
    image_by_z = {image.z_value: image for image in manifest.images}
    views = []
    dropped = []
    for section, t in enumerate(transforms, start=1):
        binned = t.model_copy(
            update={"shift_x_px": t.shift_x_px / binning, "shift_y_px": t.shift_y_px / binning}
        )
        tilt = relion_tilt(
            binned, image_size_xy=image_xy, reconstruction=reconstruction, pixel_size_angstrom=pixel
        )
        dropped.append(tilt.dropped_px)
        image = image_by_z[t.z_value]
        nominal = 10000.0 * (image.defocus_um or 3.0)
        # Geometry only: no CTF correction or dose weighting is asked of RELION here.
        views.append(
            RelionView(
                image=f"{section}@{stack.relative_to(out)}",
                nominal_tilt_deg=image.tilt_angle_deg,
                pre_exposure=0.0,
                defocus_u_angstrom=nominal,
                defocus_v_angstrom=nominal,
                defocus_angle_deg=0.0,
                tilt=tilt,
            )
        )
    optics = RelionOptics(
        voltage_kv=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.1,
        original_pixel_size_angstrom=raw_px,
    )
    write_tomogram_set(
        out, {series: (views, reconstruction.size_xyz, pixel)}, optics=optics, hand=-1
    )
    return {
        "tomogram": tomogram,
        "size_xyz": reconstruction.size_xyz,
        "binning": binning,
        "pixel_angstrom": pixel,
        "views": len(views),
        "image_xy": image_xy,
        "x_axis_tilt_deg": reconstruction.x_axis_tilt_deg,
        "z_shift_px": reconstruction.z_shift_px,
        "dropped_px_max": max(dropped),
        "dropped_angstrom_max": max(dropped) * pixel,
    }


def reconstruct(relion: str, out: Path, series: str, size_xyz, threads: int) -> np.ndarray:
    target = out / "rec" / "tomograms" / f"rec_{series}.mrc"
    command = [
        relion,
        "--t",
        "tomograms.star",
        "--tn",
        series,
        "--o",
        "rec/",
        "--w",
        str(size_xyz[0]),
        "--h",
        str(size_xyz[1]),
        "--d",
        str(size_xyz[2]),
        "--bin",
        "1",
        "--j",
        str(threads),
    ]
    result = subprocess.run(command, cwd=out, capture_output=True, text=True)
    (out / "relion_reconstruct.log").write_text(
        " ".join(command) + "\n\n" + result.stdout + result.stderr
    )
    if result.returncode != 0 or not target.is_file():
        sys.exit(
            f"RELION did not reconstruct {series} (exit {result.returncode}); "
            f"see {out / 'relion_reconstruct.log'}"
        )
    return mrcfile.read(target).astype(np.float32)


def _band(volume: np.ndarray) -> np.ndarray:
    volume = volume - volume.mean()
    return ndimage.gaussian_filter(volume, 0.7) - ndimage.gaussian_filter(volume, 6.0)


def _shrunk_band(volume: np.ndarray) -> np.ndarray:
    z, y, x = (s // 2 * 2 for s in volume.shape)
    shrunk = volume[:z, :y, :x].reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))
    return _band(shrunk)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def peak_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shift (z, y, x) of `b` against `a`, from the correlation peak and a parabola
    through it and its neighbours on each axis."""

    cc = np.fft.irfftn(np.fft.rfftn(b) * np.conj(np.fft.rfftn(a)), s=a.shape, axes=(0, 1, 2))
    cc = np.fft.fftshift(cc)
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


def compare(pipeline: np.ndarray, relion: np.ndarray) -> dict:
    if pipeline.shape != relion.shape:
        sys.exit(f"RELION's volume is {relion.shape}, the pipeline's {pipeline.shape} (z, y, x)")
    reference = _shrunk_band(pipeline)
    candidate = _shrunk_band(relion)
    reversals = {}
    for flip_z, flip_y, flip_x in itertools.product((False, True), repeat=3):
        view = candidate[:: -1 if flip_z else 1, :: -1 if flip_y else 1, :: -1 if flip_x else 1]
        reversals[f"z{int(flip_z)} y{int(flip_y)} x{int(flip_x)}"] = _ncc(view, reference)
    standard = np.ascontiguousarray(pipeline[::-1])
    nz, ny, nx = standard.shape
    centres = {
        "centre": (ny // 2, nx // 2),
        "y- x-": (ny // 4, nx // 4),
        "y- x+": (ny // 4, 3 * nx // 4),
        "y+ x-": (3 * ny // 4, nx // 4),
        "y+ x+": (3 * ny // 4, 3 * nx // 4),
    }
    shifts = {}
    for name, (y, x) in centres.items():
        box = (
            slice(max(nz // 2 - BOX // 2, 0), min(nz // 2 + BOX // 2, nz)),
            slice(max(y - BOX // 2, 0), y + BOX // 2),
            slice(max(x - BOX // 2, 0), x + BOX // 2),
        )
        shifts[name] = [
            round(float(v), 3) for v in peak_shift(_band(standard[box]), _band(relion[box]))
        ]
    return {"reversals": reversals, "shifts_zyx_voxel": shifts}


def verdict(result: dict) -> tuple[bool, list[str]]:
    ranked = sorted(result["reversals"].items(), key=lambda item: item[1], reverse=True)
    (best, best_ncc), (_, second_ncc) = ranked[0], ranked[1]
    reasons = []
    if best != "z1 y0 x0":
        reasons.append(f"the best match is the reversal {best}, not z1 y0 x0 (z alone)")
    if best_ncc < MARGIN * max(second_ncc, 1e-6):
        reasons.append(
            f"the best NCC {best_ncc:.3f} is not {MARGIN:g} times the next ({second_ncc:.3f})"
        )
    for name, shift in result["shifts_zyx_voxel"].items():
        limit = CENTRE_LIMIT if name == "centre" else QUARTER_LIMIT
        if max(abs(v) for v in shift) > limit:
            reasons.append(f"the {name} box is shifted {shift} voxel, over {limit}")
    return not reasons, reasons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--series", required=True)
    parser.add_argument("--tomogram-id", help="default <series>:tomogram:fine")
    parser.add_argument("--out", type=Path, required=True, help="a new folder")
    parser.add_argument(
        "--relion",
        default=os.environ.get("RELION_RECONSTRUCT_TOMOGRAM")
        or shutil.which("relion_tomo_reconstruct_tomogram"),
        help="relion_tomo_reconstruct_tomogram (RELION 5)",
    )
    parser.add_argument(
        "--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    )
    args = parser.parse_args(argv)
    if not args.relion:
        parser.error("no relion_tomo_reconstruct_tomogram: give --relion")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    built = build(args.project, args.series, args.tomogram_id, args.out)
    tomogram: Artifact = built.pop("tomogram")
    relion = reconstruct(args.relion, args.out, args.series, built["size_xyz"], args.threads)
    pipeline = np.asarray(zarr.open(str(tomogram.path), mode="r"), dtype=np.float32)
    result = compare(pipeline, relion)
    passed, reasons = verdict(result)
    report = {
        "series": args.series,
        "tomogram_id": tomogram.id,
        **built,
        **result,
        "passed": passed,
        "reasons": reasons,
    }
    (args.out / "geometry_check.json").write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"{args.series} ({tomogram.id}): {built['views']} views at bin{built['binning']} "
        f"({built['pixel_angstrom']:.3f} A), tomogram {built['size_xyz']}, "
        f"x-axis tilt {built['x_axis_tilt_deg']:+.2f} deg, z shift {built['z_shift_px']:+.1f} px"
    )
    print(
        f"dropped by RELION's model (magnification/stretch/skew): up to "
        f"{built['dropped_px_max']:.2f} px = {built['dropped_angstrom_max']:.1f} A at the corners"
    )
    for name, value in sorted(result["reversals"].items(), key=lambda item: -item[1]):
        print(f"  reversal {name}: NCC {value:+.3f}")
    for name, shift in result["shifts_zyx_voxel"].items():
        print(f"  {name:7s} shift (z, y, x) {shift[0]:+.2f} {shift[1]:+.2f} {shift[2]:+.2f} voxel")
    print("PASS" if passed else "FAIL: " + "; ".join(reasons))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
