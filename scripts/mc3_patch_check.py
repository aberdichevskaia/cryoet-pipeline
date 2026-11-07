"""Does MotionCor3's patch grid cut a tilt's sum into displaced tiles?

The question (EMPIAR-11830 Position_1, 7 November 2026): the five views that
broke coarse alignment (-75, -72, -69, +42, +45 degrees) show straight seams on
the 5 x 5 grid the pipeline gives MotionCor3 (`-Patch 5 5`, the [motion]
default), with content moved from tile to tile. If MotionCor3's patch
alignment makes them, the same movie summed with `-Patch 1 1` (one shift per
frame for the whole image, which a lane sets as [motion] patch_x = patch_y = 1)
has none, and the tiles of the 5 x 5 sum sit at other places than in it.

    plan     reads the series' MotionCor3 sums from <project>/artifacts.json,
             takes the tilts nearest the requested angles, and writes
             <out>/plan.json: for each, the command the pipeline ran, to be run
             again as it was ("p5", outputs moved to <out>) and with
             `-Patch 1 1` ("p1"). The executable, movie, gain, flip, FmIntFile
             and every other flag are the pipeline's; the executable's sha256
             is checked against the one the run recorded.
    run      runs them, one after another, on GPU 0 of the job.
    compare  bins every sum to about 1024 px and finds, for the centre of each
             tile of the pipeline's grid, where that content sits in another sum
             (normalized cross-correlation after a band-pass, +-112 binned px;
             a tile is found when the best correlation is at least 0.3):
               control: the pipeline's own sum against p5. The tilt reproduces
                        when at least half the tiles are found and none is
                        more than 2 raw px away. Not bit for bit: MotionCor3
                        does not give the same pixels twice (job 3901454: r
                        0.976-0.989 at -15/-45/+15, every tile within 0.6 px).
               effect:  the pipeline's sum and p5, each against p1.
             The common shift of the found tiles is removed; what is left, in
             raw pixels, is how far the patch alignment moved each tile. A sum
             "moved" when at least two found tiles sit beyond the threshold,
             is "unchanged" when at least half are found and none does, and is
             "unclear" otherwise. Prints a table and the verdicts, writes
             compare.png, and exits 0 whatever they are.

Prediction, written before the run: at -75 and +45 some tile of the p5 sum sits
more than 7.9 raw px from where it sits in the p1 sum (7.9 px = 1.5 nm at
1.9037 A, the fine-alignment failure threshold); at -15 none does.

Changed after the first run (job 3901454): the control first also asked for a
correlation of the two images above 0.99, which MotionCor3's reruns never reach,
and told the reader to suspect the binary; it now judges tile positions, tilt by
tilt. A tilt with fewer than half its tiles found was "not measurable", which
the prediction never said: at +45 the 5 x 5 tiles are mostly not found in the
-Patch 1 1 sum at all, and the six found sat 105-469 px away.

Frames or darkness (`plan --fractions`, EER movies only). On Position_1 the five
tilts whose patch alignment broke are the darkest and also the only ones with 3
or 4 frames: the pipeline sums an EER movie's frames in as many groups as its
dose allows, and MotionCor3 aligns patches on groups of 4 frames (`-Group 1 4`,
its default). `--fractions 4 10` adds, for each chosen tilt, the recorded command
again with the same patch grid but the frames summed in 4 and in 10 groups. Each
5 x 5 sum (the rerun as recorded and these) is judged twice: by MotionCor3's
patch log, with the motion QC's rule (at least half the patches beyond the
threshold), and by its tiles against the -Patch 1 1 sum, as above. Then:
  the frame count   every sum with no more frames than the patch group broke,
                    and none with more, on tilts that broke as recorded and on
                    tilts that did not;
  the tilt          every sum broke as its tilt's recorded run did, whatever
                    its frames (darkness, or anything else of the tilt);
  not told apart    no run put a tilt's frames on the other side of the group
                    size from its recorded run, so both rules fit;
  neither alone     some run fits neither.

Prediction, written before that run: the frame count. -75 and -72 with 10
groups do not break, and -66 and -15 with 4 do. The step on Position_1 is sharp
(every tilt with 5 frames had at most one patch beyond the threshold, every one
with 4 had all 25 there), and the shifts at -75 overflow to 1e20, which looks
like arithmetic on a single group rather than on a weak signal. Darkness would
instead keep -75 and -72 broken with 10 groups and -66 and -15 whole with 4.

Disposable; scripts/mc3_patch_check.sbatch runs the three steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

# Displacement that counts: the fine-alignment failure threshold (1.5 nm) in
# raw pixels of EMPIAR-11830 (1.9037 A). compare prints it in nm too.
DEFAULT_THRESHOLD_NM = 1.5
# A tile whose best correlation is lower than this is reported, not measured.
MIN_PEAK = 0.3
# Tolerance for "the same place" in the control, raw px.
REPRODUCIBLE_PX = 2.0
# Found tiles beyond the threshold that make a sum "moved" (one could be a
# false match).
MIN_MOVED_TILES = 2


# ---------------------------------------------------------------- plan

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _flag_values(command: list[str], flag: str) -> tuple[int, int]:
    """Index of `flag` and of the first token after its values."""

    index = command.index(flag)
    end = index + 1
    while end < len(command) and not command[end].startswith("-"):
        end += 1
    # A negative number is a value, not a flag.
    while end < len(command):
        try:
            float(command[end])
        except ValueError:
            break
        end += 1
    return index, end


def rewrite_command(
    command: list[str],
    *,
    patches: tuple[int, int] | None,
    out_mrc: Path,
    log_dir: Path,
    aln_dir: Path,
    fmint: Path | None,
) -> list[str]:
    """The pipeline's MotionCor3 command with outputs moved and, if given,
    another patch grid. Everything else is kept."""

    new = list(command)
    replacements: dict[str, list[str]] = {
        "-OutMrc": [str(out_mrc)],
        "-LogDir": [str(log_dir)],
        "-OutAln": [str(aln_dir)],
        "-Gpu": ["0"],
    }
    if patches is not None:
        replacements["-Patch"] = [str(patches[0]), str(patches[1])]
    if fmint is not None:
        replacements["-FmIntFile"] = [str(fmint)]
    for flag, values in replacements.items():
        if flag not in new:
            raise ValueError(f"the recorded command has no {flag}: {command}")
        index, end = _flag_values(new, flag)
        new[index + 1 : end] = values
    return new


def _groups(eer_frames: int, fractions: int) -> int:
    """The frames MotionCor3 makes of an EER movie asked for `fractions` groups:
    N // G with G = N // fractions (docs/motioncor3_backend.md, "EER movies")."""

    return eer_frames // (eer_frames // fractions)


def _patch_grid(command: list[str]) -> tuple[int, int]:
    index, end = _flag_values(command, "-Patch")
    values = [int(value) for value in command[index + 1 : end]]
    if len(values) < 2:
        raise ValueError(f"cannot read -Patch from {command}")
    return values[0], values[1]


def plan(args: argparse.Namespace) -> int:
    from cryoet_pipeline.artifacts import ArtifactRegistry
    from cryoet_pipeline.backends.motion import motioncor3_eer_fmint
    from cryoet_pipeline.models import ArtifactKind

    project = Path(args.project)
    registry_path = project / "artifacts.json"
    if not registry_path.is_file():
        print(f"no registry at {registry_path}", file=sys.stderr)
        return 1
    registry = ArtifactRegistry.load(registry_path)
    prefix = f"{args.series}:corrected_projection:"
    sums = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.CORRECTED_PROJECTION)
        if artifact.id.startswith(prefix) and artifact.id.endswith(":motioncor3")
    ]
    if not sums:
        series = sorted(
            {
                artifact.id.split(":")[0]
                for artifact in registry.by_kind(ArtifactKind.CORRECTED_PROJECTION)
                if artifact.id.endswith(":motioncor3")
            }
        )
        print(
            f"{registry_path} has no MotionCor3 sums of {args.series}; "
            f"it has some of: {', '.join(series) or 'no series'}",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    chosen: list[dict[str, Any]] = []
    for angle in args.angles:
        artifact = min(
            sums, key=lambda item: abs(float(item.parameters["tilt_angle_deg"]) - angle)
        )
        tilt = float(artifact.parameters["tilt_angle_deg"])
        if abs(tilt - angle) > 1.5:
            print(f"no tilt within 1.5 deg of {angle:g} (nearest {tilt:g})", file=sys.stderr)
            return 1
        if any(entry["id"] == artifact.id for entry in chosen):
            continue
        params = artifact.parameters
        command = [str(token) for token in params["command"]]
        executable = Path(command[0])
        if not executable.is_file():
            print(f"the recorded MotionCor3 is gone: {executable}", file=sys.stderr)
            return 1
        recorded_sha = params.get("motioncor3_executable_sha256")
        actual_sha = _sha256(executable)
        if recorded_sha and recorded_sha != actual_sha:
            print(
                f"{executable} changed since the pipeline ran it "
                f"(sha256 {actual_sha[:16]}, recorded {str(recorded_sha)[:16]})",
                file=sys.stderr,
            )
            return 1
        movie = Path(str(params["source_frame_file"]))
        if not movie.is_file():
            print(f"the movie is gone: {movie}", file=sys.stderr)
            return 1
        stem = f"{args.series}_{int(params['z_value']):03d}"
        fmint: Path | None = None
        if "-FmIntFile" in command:
            recorded = Path(command[_flag_values(command, "-FmIntFile")[0] + 1])
            if recorded.is_file():
                fmint = recorded
            else:
                fmint = out / f"{stem}.fmint"
                fmint.write_text(
                    motioncor3_eer_fmint(
                        int(params["eer_frames"]), int(params["eer_fractions"])
                    )
                )
        sum_path = Path(str(artifact.path))
        if not sum_path.is_absolute() and not sum_path.exists():
            sum_path = project / sum_path
        eer_frames = params.get("eer_frames")
        recorded_frames = params.get("eer_groups")
        if recorded_frames is None and eer_frames and params.get("eer_fractions"):
            recorded_frames = _groups(int(eer_frames), int(params["eer_fractions"]))
        groups: list[tuple[str, tuple[int, int] | None, Path | None, int | None]] = [
            ("p5", None, fmint, recorded_frames),
            ("p1", (1, 1), fmint, recorded_frames),
        ]
        for fractions in args.fractions:
            if fmint is None or not eer_frames:
                print(
                    "--fractions needs EER movies: the recorded command has no -FmIntFile",
                    file=sys.stderr,
                )
                return 1
            if not 2 <= fractions <= int(eer_frames):
                print(f"--fractions {fractions}: not between 2 and {eer_frames}", file=sys.stderr)
                return 1
            group_file = out / f"{stem}_g{fractions}.fmint"
            group_file.write_text(motioncor3_eer_fmint(int(eer_frames), fractions))
            groups.append(
                (f"g{fractions}", None, group_file, _groups(int(eer_frames), fractions))
            )
        runs = {}
        for label, patches, group_file, frames in groups:
            runs[label] = {
                "out_mrc": str(out / f"{stem}_{label}.mrc"),
                "stdout": str(out / "logs" / f"{stem}_{label}.out"),
                "frames": frames,
                "command": rewrite_command(
                    command,
                    patches=patches,
                    out_mrc=out / f"{stem}_{label}.mrc",
                    log_dir=out / "logs" / f"{stem}_{label}",
                    aln_dir=out / "aln" / f"{stem}_{label}",
                    fmint=group_file,
                ),
            }
        chosen.append(
            {
                "id": artifact.id,
                "tilt_angle_deg": tilt,
                "z_value": int(params["z_value"]),
                "pipeline_sum": str(sum_path.resolve()),
                "pipeline_patches": list(_patch_grid(command)),
                "pixel_angstrom": float(params["pixel_spacing_angstrom"]),
                "gain_reference": params.get("gain_reference"),
                "executable_sha256": actual_sha,
                "tilt_dose_e_per_a2": params.get("eer_tilt_dose_e_per_a2"),
                "runs": runs,
            }
        )
        print(
            f"{tilt:7.2f} deg  z {int(params['z_value']):3d}  {movie.name}  "
            f"pipeline -Patch {' '.join(map(str, _patch_grid(command)))}"
        )
    (out / "plan.json").write_text(json.dumps({"series": args.series, "tilts": chosen}, indent=2))
    first = chosen[0]
    print(
        f"executable {first['runs']['p5']['command'][0]} "
        f"(sha256 {first['executable_sha256'][:16]})"
    )
    print(f"wrote {out / 'plan.json'}")
    return 0


# ---------------------------------------------------------------- run

def run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    document = json.loads((out / "plan.json").read_text())
    for tilt in document["tilts"]:
        needs_gain = tilt.get("gain_reference") is not None
        for label, spec in tilt["runs"].items():
            command = spec["command"]
            for key in ("-LogDir", "-OutAln"):
                Path(command[command.index(key) + 1]).mkdir(parents=True, exist_ok=True)
            stdout = Path(spec["stdout"])
            stdout.parent.mkdir(parents=True, exist_ok=True)
            print(f"{tilt['tilt_angle_deg']:7.2f} deg {label}: MotionCor3 ...", flush=True)
            with stdout.open("w") as handle:
                result = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, cwd=out, check=False
                )
            text = stdout.read_text(errors="replace")
            if result.returncode != 0 or not Path(spec["out_mrc"]).is_file():
                print(
                    f"MotionCor3 failed (exit {result.returncode}); the end of {stdout}:",
                    file=sys.stderr,
                )
                print("\n".join(text.splitlines()[-20:]), file=sys.stderr)
                return 1
            if needs_gain and "Gain reference has been loaded" not in text:
                print(f"MotionCor3 did not load the gain; see {stdout}", file=sys.stderr)
                return 1
    print("all runs done")
    return 0


# ---------------------------------------------------------------- compare

def read_sum(path: Path) -> np.ndarray:
    if path.suffix == ".zarr" or path.is_dir():
        import zarr

        data = np.asarray(zarr.open(str(path), mode="r")[...])
    else:
        import mrcfile

        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f"{path}: expected one 2D image, got shape {data.shape}")
    return data.astype(np.float32)


def bin_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    h, w = (size // factor * factor for size in image.shape)
    return image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def band_pass(image: np.ndarray) -> np.ndarray:
    return gaussian_filter(image, 1.0) - gaussian_filter(image, 12.0)


def _ncc_peak(patch: np.ndarray, window: np.ndarray) -> tuple[float, float, float]:
    """Where `patch` fits in `window` (offset of its top-left corner from the
    window's centre position), with sub-pixel refinement, and the NCC there."""

    ph, pw = patch.shape
    wh, ww = window.shape
    p = patch - patch.mean()
    p_norm = float(np.sqrt((p * p).sum()))
    if p_norm == 0.0:
        return 0.0, 0.0, 0.0
    shape = (wh, ww)
    f_window = np.fft.rfft2(window, shape)
    numerator = np.fft.irfft2(f_window * np.conj(np.fft.rfft2(p, shape)), shape)
    ones = np.zeros(shape)
    ones[:ph, :pw] = 1.0
    f_ones = np.conj(np.fft.rfft2(ones, shape))
    local_sum = np.fft.irfft2(f_window * f_ones, shape)
    local_sq = np.fft.irfft2(np.fft.rfft2(window * window, shape) * f_ones, shape)
    local_var = np.maximum(local_sq - local_sum**2 / (ph * pw), 1e-12)
    ncc = numerator / (p_norm * np.sqrt(local_var))
    valid = ncc[: wh - ph + 1, : ww - pw + 1]
    iy, ix = np.unravel_index(int(np.argmax(valid)), valid.shape)
    peak = float(valid[iy, ix])

    def refine(values: np.ndarray, index: int) -> float:
        if 0 < index < len(values) - 1:
            left, mid, right = values[index - 1], values[index], values[index + 1]
            denominator = left - 2 * mid + right
            if denominator < 0:
                return 0.5 * (left - right) / denominator
        return 0.0

    dy = iy + refine(valid[:, ix], iy) - (wh - ph) / 2
    dx = ix + refine(valid[iy, :], ix) - (ww - pw) / 2
    return float(dy), float(dx), peak


def tile_shifts(
    test: np.ndarray,
    reference: np.ndarray,
    grid: tuple[int, int],
    *,
    core_fraction: float = 0.55,
    search: int = 112,
) -> np.ndarray:
    """For each tile of `grid` (x, y) over `test`: (dy, dx, peak), where dy, dx
    is where the tile's central content sits in `reference` relative to where
    it sits in `test`, in pixels of the images given."""

    if test.shape != reference.shape:
        raise ValueError(f"shapes differ: {test.shape} vs {reference.shape}")
    nx, ny = grid
    h, w = test.shape
    t_band = band_pass(test)
    r_band = np.pad(band_pass(reference), search, mode="constant")
    core_y = max(16, int(core_fraction * h / ny))
    core_x = max(16, int(core_fraction * w / nx))
    out = np.zeros((ny, nx, 3))
    for j in range(ny):
        for i in range(nx):
            cy = (j + 0.5) * h / ny
            cx = (i + 0.5) * w / nx
            y0 = int(round(cy - core_y / 2))
            x0 = int(round(cx - core_x / 2))
            patch = t_band[y0 : y0 + core_y, x0 : x0 + core_x]
            window = r_band[y0 : y0 + core_y + 2 * search, x0 : x0 + core_x + 2 * search]
            out[j, i] = _ncc_peak(patch, window)
    return out


def displacement_summary(shifts: np.ndarray, scale: float) -> dict[str, Any]:
    """Per-tile displacement in raw px after the median of matched tiles is
    removed (the sums can differ by one shift for the whole image)."""

    matched = shifts[..., 2] >= MIN_PEAK
    if matched.sum() == 0:
        return {"matched": 0, "tiles": int(matched.size), "max_px": None, "median_px": None,
                "common_px": None, "residual": None, "magnitude": None,
                "matched_mask": matched}
    vectors = shifts[..., :2] * scale
    common = np.median(vectors[matched], axis=0)
    residual = vectors - common
    magnitude = np.hypot(residual[..., 0], residual[..., 1])
    return {
        "matched": int(matched.sum()),
        "tiles": int(matched.size),
        "max_px": float(magnitude[matched].max()),
        "median_px": float(np.median(magnitude[matched])),
        "common_px": [float(common[0]), float(common[1])],
        "residual": residual,
        "magnitude": magnitude,
        "matched_mask": matched,
    }


def beyond(summary: dict[str, Any], threshold_px: float) -> int:
    """Found tiles that sit more than threshold_px away."""

    if summary["magnitude"] is None:
        return 0
    return int((summary["magnitude"][summary["matched_mask"]] > threshold_px).sum())


def sum_verdict(summary: dict[str, Any], threshold_px: float) -> str:
    """"moved", "unchanged" or "unclear" for one 5 x 5 sum against p1."""

    far = beyond(summary, threshold_px)
    if far >= MIN_MOVED_TILES:
        return "moved"
    if far == 0 and 2 * summary["matched"] >= summary["tiles"]:
        return "unchanged"
    return "unclear"


def reproduced(control: dict[str, Any]) -> bool:
    """The rerun puts the tiles where the pipeline's sum has them."""

    return (
        control["max_px"] is not None
        and 2 * control["matched"] >= control["tiles"]
        and control["max_px"] <= REPRODUCIBLE_PX
    )


def compare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    document = json.loads((out / "plan.json").read_text())
    rows: list[dict[str, Any]] = []
    for tilt in document["tilts"]:
        pipeline = read_sum(Path(tilt["pipeline_sum"]))
        p5 = read_sum(Path(tilt["runs"]["p5"]["out_mrc"]))
        p1 = read_sum(Path(tilt["runs"]["p1"]["out_mrc"]))
        if not (pipeline.shape == p5.shape == p1.shape):
            print(f"shapes differ: {pipeline.shape} {p5.shape} {p1.shape}", file=sys.stderr)
            return 1
        factor = max(1, int(round(pipeline.shape[1] / 1024)))
        b_pipe, b_p5, b_p1 = (bin_image(image, factor) for image in (pipeline, p5, p1))
        grid = tuple(tilt["pipeline_patches"])
        rows.append(
            {
                "tilt": tilt,
                "factor": factor,
                "r": float(np.corrcoef(b_pipe.ravel(), b_p5.ravel())[0, 1]),
                "control": displacement_summary(tile_shifts(b_pipe, b_p5, grid), factor),
                "pipeline": displacement_summary(tile_shifts(b_pipe, b_p1, grid), factor),
                "rerun": displacement_summary(tile_shifts(b_p5, b_p1, grid), factor),
                "images": (b_pipe, b_p5, b_p1),
            }
        )

    # The sums' pixel: "raw px" below are pixels of the sums, before binning.
    pixel = rows[0]["tilt"]["pixel_angstrom"] if rows else 1.0
    threshold_px = args.threshold_nm * 10.0 / pixel
    print(
        f"sums' pixel {pixel:.4f} A; threshold {args.threshold_nm} nm = {threshold_px:.1f} px; "
        f"a tile is found at a correlation of {MIN_PEAK} or more"
    )
    print(
        "        control: rerun vs pipeline sum   | each 5x5 sum vs -Patch 1 1: "
        "found, beyond threshold, max px"
    )
    print(
        "angle    r      found  max px  place     | pipeline sum           "
        "| rerun"
    )

    def fmt(value: float | None) -> str:
        return "     -" if value is None else f"{value:6.1f}"

    def cell(summary: dict[str, Any]) -> str:
        return (
            f"{summary['matched']:2d}/{summary['tiles']:2d} {beyond(summary, threshold_px):3d} "
            f"{fmt(summary['max_px'])}"
        )

    for row in rows:
        control = row["control"]
        place = "same" if reproduced(control) else "differs"
        print(
            f"{row['tilt']['tilt_angle_deg']:6.1f}  {row['r']:6.3f}  "
            f"{control['matched']:2d}/{control['tiles']:2d} {fmt(control['max_px'])}  "
            f"{place:8s}  | {cell(row['pipeline'])}       | {cell(row['rerun'])}"
        )
    print()

    differ = [row["tilt"]["tilt_angle_deg"] for row in rows if not reproduced(row["control"])]
    if not differ:
        print(
            "control: at every tilt the rerun of the recorded command puts each tile where "
            "the pipeline's sum has it."
        )
    elif len(differ) < len(rows):
        print(
            "control: at " + ", ".join(f"{angle:g}" for angle in differ) + " deg the rerun of "
            "the recorded command (same executable, sha256 checked) does not put the tiles "
            "where the pipeline's sum has them, while at the other tilts it does: there "
            "MotionCor3's result with this patch grid changes from run to run."
        )
    else:
        print(
            "control: at no tilt does the rerun of the recorded command put the tiles where "
            "the pipeline's sum has them. Check that these sums are the ones this command "
            "made (their dates; a later motion run) before reading the rest."
        )
    print("(r is for information: MotionCor3 does not give the same pixels twice.)")
    print()

    verdicts: dict[float, str] = {}
    for row in rows:
        angle = row["tilt"]["tilt_angle_deg"]
        pipeline_says = sum_verdict(row["pipeline"], threshold_px)
        rerun_says = sum_verdict(row["rerun"], threshold_px)
        both = pipeline_says if pipeline_says == rerun_says else "mixed"
        verdicts[angle] = both
        text = {
            "moved": f"the 5x5 sums' tiles moved by more than {args.threshold_nm} nm",
            "unchanged": f"no tile of the 5x5 sums moved by more than {args.threshold_nm} nm",
            "unclear": "unclear (too few tiles found)",
            "mixed": "the two 5x5 sums disagree",
        }[both]
        print(
            f"{angle:6.1f} deg: {text} (pipeline sum: {pipeline_says}; rerun: {rerun_says})"
        )
    ends = [angle for angle in verdicts if abs(angle - (-75)) < 1.6 or abs(angle - 45) < 1.6]
    mid = [angle for angle in verdicts if abs(angle - (-15)) < 1.6]
    if ends and mid:
        held = all(verdicts[a] == "moved" for a in ends) and all(
            verdicts[a] == "unchanged" for a in mid
        )
        print()
        print("prediction (ends moved, -15 not): " + ("HELD" if held else "DID NOT HOLD"))
    _figure(rows, out / "compare.png", threshold_px)
    print(f"wrote {out / 'compare.png'}")
    tested = [row for row in rows if any(label.startswith("g") for label in row["tilt"]["runs"])]
    if tested:
        print()
        frames_or_darkness(tested, out, threshold_px, args.threshold_nm)
    return 0


# ---------------------------------------------------------------- frames or darkness

# MotionCor3's default: patches are aligned on groups of this many frames
# (`-Group 1 4`). A run's own output gives the value when it prints it.
DEFAULT_PATCH_GROUP = 4


def patch_group(stdout: Path) -> int:
    """The second value of the -Group MotionCor3 printed, else its default."""

    if stdout.is_file():
        match = re.search(
            r"^-Group\s+\d+\s+(\d+)", stdout.read_text(errors="replace"), re.MULTILINE
        )
        if match:
            return int(match.group(1))
    return DEFAULT_PATCH_GROUP


def patch_log_verdict(
    spec: dict[str, Any], pixel: float, width: int, threshold_nm: float
) -> dict[str, Any] | None:
    """The motion QC's judgement of one run, from the patch log MotionCor3 wrote
    for it; None when there is none that reads."""

    from cryoet_pipeline.backends.motion import (
        PATCH_LOG_SUFFIX,
        judge_patch_shifts,
        read_motioncor3_patch_log,
    )

    command = spec["command"]
    movie = next(
        (
            Path(command[command.index(flag) + 1])
            for flag in ("-InEer", "-InMrc", "-InTiff")
            if flag in command
        ),
        None,
    )
    if movie is None:
        return None
    log = Path(command[command.index("-LogDir") + 1]) / f"{movie.stem}{PATCH_LOG_SUFFIX}"
    try:
        patch_log = read_motioncor3_patch_log(log)
    except (OSError, ValueError):
        return None
    shift_pixel = pixel * width / patch_log.width
    return judge_patch_shifts(patch_log, tolerance_px=threshold_nm * 10.0 / shift_pixel)


def broken(log_says: str | None, tiles_say: str) -> bool | None:
    """Whether a 5 x 5 sum broke: the patch log decides unless the tiles
    contradict it; without a log, the tiles. None when it cannot be said."""

    if log_says is None:
        return {"moved": True, "unchanged": False}.get(tiles_say)
    if log_says == "failed":
        return None if tiles_say == "unchanged" else True
    return None if tiles_say == "moved" else False


def judge_frames_or_darkness(runs: list[dict[str, Any]]) -> tuple[str, str]:
    """(verdict, explanation) over the 5 x 5 runs: each with `angle`, `label`,
    `frames`, `group`, `broken` and `tilt_broken` (its tilt's recorded run)."""

    known = [
        run for run in runs
        if run["broken"] is not None and run["tilt_broken"] is not None
        and run["frames"] is not None
    ]
    if not known:
        return "nothing to judge", "no 5x5 sum could be called broken or whole"
    group = known[0]["group"]

    def by_frames(run: dict[str, Any]) -> bool:
        return run["broken"] == (run["frames"] <= run["group"])

    def by_tilt(run: dict[str, Any]) -> bool:
        return run["broken"] == run["tilt_broken"]

    frames_fit = all(by_frames(run) for run in known)
    tilt_fits = all(by_tilt(run) for run in known)
    if frames_fit and not tilt_fits:
        return "the frame count", (
            f"every 5x5 sum with {group} frames or fewer broke and none with more did, "
            "on tilts that broke as recorded and on tilts that did not"
        )
    if tilt_fits and not frames_fit:
        return "the tilt", (
            "every 5x5 sum broke as its tilt's recorded run did, whatever its frames "
            "(darkness, or anything else of the tilt), not by the frame count"
        )
    if frames_fit and tilt_fits:
        return "not told apart", (
            f"no run put a tilt's frames on the other side of {group} from its recorded "
            f"run; give --fractions that do (more than {group} groups for a tilt that "
            f"broke, {group} or fewer for one that did not)"
        )
    neither = [run for run in known if not by_frames(run) and not by_tilt(run)]
    return "neither alone", "these fit neither rule: " + "; ".join(
        f"{run['angle']:.1f} deg {_run_name(run['label'])}, {run['frames']} frames, "
        f"{'broke' if run['broken'] else 'whole'}"
        for run in neither
    )


def _run_name(label: str) -> str:
    return "as recorded" if label == "p5" else f"{label[1:]} groups"


def frames_or_darkness(
    rows: list[dict[str, Any]], out: Path, threshold_px: float, threshold_nm: float
) -> str:
    """Judge every 5 x 5 sum of the tested tilts, print the table and the
    verdict on frames against darkness, write frames.png; the verdict."""

    runs: list[dict[str, Any]] = []
    for row in rows:
        tilt = row["tilt"]
        group = patch_group(Path(tilt["runs"]["p5"]["stdout"]))
        b_p1 = row["images"][2]
        for label, spec in tilt["runs"].items():
            if label == "p1":
                continue
            raw = read_sum(Path(spec["out_mrc"]))
            image = bin_image(raw, row["factor"])
            tiles = displacement_summary(
                tile_shifts(image, b_p1, tuple(tilt["pipeline_patches"])), row["factor"]
            )
            log = patch_log_verdict(spec, tilt["pixel_angstrom"], raw.shape[1], threshold_nm)
            log_says = None if log is None else ("failed" if log["failed"] else "ok")
            tiles_say = sum_verdict(tiles, threshold_px)
            runs.append(
                {
                    "angle": tilt["tilt_angle_deg"],
                    "label": label,
                    "frames": spec.get("frames"),
                    "dose": tilt.get("tilt_dose_e_per_a2"),
                    "group": group,
                    "log": log,
                    "log_says": log_says,
                    "tiles": tiles,
                    "tiles_say": tiles_say,
                    "broken": broken(log_says, tiles_say),
                    "image": image,
                    "factor": row["factor"],
                    "grid": tuple(tilt["pipeline_patches"]),
                }
            )
    for run in runs:
        recorded = next(
            other for other in runs if other["angle"] == run["angle"] and other["label"] == "p5"
        )
        run["tilt_broken"] = recorded["broken"]

    print(
        "frames or darkness: each 5x5 sum judged by its patch log (the motion QC's rule) and "
        "by its tiles against -Patch 1 1"
    )
    print(
        "angle   e/A2  run           frames  e/A2/frame | patch log: beyond, median px "
        "| tiles vs 1x1: found, beyond, max px | broke"
    )
    for run in runs:
        dose = run["dose"]
        frames = run["frames"]
        per_frame = (
            f"{dose / frames:6.3f}" if dose is not None and frames else "     -"
        )
        if run["log"] is None:
            log_cell = "no patch log        "
        else:
            median = run["log"]["median_patch_shift_px"]
            log_cell = (
                f"{run['log']['patches_beyond']:2d}/{run['log']['patches']:2d} "
                f"{'   lost' if median is None else f'{median:7.1f}'} {run['log_says']:6s}"
            )
        tiles = run["tiles"]
        max_px = "     -" if tiles["max_px"] is None else f"{tiles['max_px']:6.1f}"
        says = {True: "yes", False: "no", None: "?"}[run["broken"]]
        print(
            f"{run['angle']:6.1f}  {'    -' if dose is None else f'{dose:5.2f}'}  "
            f"{_run_name(run['label']):12s}  {'-' if frames is None else frames:>6}  "
            f"{per_frame:>10} | {log_cell} | {tiles['matched']:2d}/{tiles['tiles']:2d} "
            f"{beyond(tiles, threshold_px):3d} {max_px} {run['tiles_say']:9s} | {says}"
        )
    print()
    groups = sorted({run["group"] for run in runs})
    print(
        "MotionCor3 aligned the patches on groups of "
        + "/".join(str(value) for value in groups)
        + " frames (-Group, as its output printed it, else its default 4)."
    )
    left_out = [run for run in runs if run["broken"] is None]
    if left_out:
        print(
            "left out, neither broken nor whole by their patch log and tiles: "
            + "; ".join(
                f"{run['angle']:.1f} deg {_run_name(run['label'])} "
                f"(log {run['log_says'] or 'none'}, tiles {run['tiles_say']})"
                for run in left_out
            )
        )
    verdict, why = judge_frames_or_darkness(runs)
    print(f"verdict: {verdict}: {why}.")
    outcome = {
        "the frame count": "HELD",
        "the tilt": "DID NOT HOLD",
        "neither alone": "DID NOT HOLD",
    }.get(verdict, "NOT TESTED")
    print(f"prediction (the frame count): {outcome}")
    _frames_figure(runs, out / "frames.png", threshold_px)
    print(f"wrote {out / 'frames.png'}")
    return verdict


def _frames_figure(runs: list[dict[str, Any]], path: Path, threshold_px: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    angles = sorted({run["angle"] for run in runs})
    columns = max(sum(1 for run in runs if run["angle"] == angle) for angle in angles)
    fig, axes = plt.subplots(
        len(angles), columns, figsize=(5 * columns, 5.2 * len(angles)), squeeze=False
    )
    for axes_row, angle in zip(axes, angles, strict=True):
        tilt_runs = [run for run in runs if run["angle"] == angle]
        for ax in axes_row[len(tilt_runs):]:
            ax.axis("off")
        for ax, run in zip(axes_row, tilt_runs, strict=False):
            image = run["image"]
            shown = gaussian_filter(image, 1.0)
            lo, hi = np.percentile(shown, [1, 99])
            h, w = image.shape
            ax.imshow(shown, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=(0, w, 0, h))
            nx, ny = run["grid"]
            for i in range(1, nx):
                ax.axvline(w * i / nx, color="red", lw=0.6, alpha=0.5)
            for j in range(1, ny):
                ax.axhline(h * j / ny, color="red", lw=0.6, alpha=0.5)
            _annotate(ax, run["tiles"], run["factor"], threshold_px, (nx, ny), (h, w))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"{angle:.1f} deg, {_run_name(run['label'])}: {run['frames']} frames\n"
                f"patch log {run['log_says'] or 'none'}, tiles {run['tiles_say']} "
                f"(raw px from the -Patch 1 1 sum)",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def _annotate(ax: Any, summary: dict[str, Any], factor: int, threshold_px: float,
              grid: tuple[int, int], size: tuple[int, int]) -> None:
    """Each tile's distance from the -Patch 1 1 sum, raw px; x = not found."""

    nx, ny = grid
    h, w = size
    if summary["residual"] is None:
        return
    residual = summary["residual"] / factor
    for j in range(ny):
        for i in range(nx):
            cx, cy = (i + 0.5) * w / nx, (j + 0.5) * h / ny
            if not summary["matched_mask"][j, i]:
                ax.plot(cx, cy, "x", color="orange", ms=8)
                continue
            dy, dx = residual[j, i]
            magnitude = float(summary["magnitude"][j, i])
            colour = "yellow" if magnitude > threshold_px else "cyan"
            # Arrows are true length in the binned image shown.
            ax.annotate(
                "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.5),
            )
            ax.text(cx, cy - 0.12 * h / ny, f"{magnitude:.0f}", color=colour,
                    ha="center", va="top", fontsize=9, fontweight="bold")


def _figure(rows: list[dict[str, Any]], path: Path, threshold_px: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5.2 * len(rows)), squeeze=False)
    titles = (
        "pipeline sum (as run)",
        "rerun of the recorded command",
        "rerun with -Patch 1 1",
    )
    for axes_row, row in zip(axes, rows, strict=True):
        tilt = row["tilt"]
        nx, ny = tilt["pipeline_patches"]
        for k, (ax, image, title) in enumerate(zip(axes_row, row["images"], titles, strict=True)):
            shown = gaussian_filter(image, 1.0)
            lo, hi = np.percentile(shown, [1, 99])
            h, w = image.shape
            ax.imshow(shown, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=(0, w, 0, h))
            for i in range(1, nx):
                ax.axvline(w * i / nx, color="red", lw=0.6, alpha=0.5)
            for j in range(1, ny):
                ax.axhline(h * j / ny, color="red", lw=0.6, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            label = f"{tilt['tilt_angle_deg']:.1f} deg: {title}"
            if k < 2:
                summary = row["pipeline"] if k == 0 else row["rerun"]
                _annotate(ax, summary, row["factor"], threshold_px, (nx, ny), (h, w))
                label += (
                    f"\nraw px from the -Patch 1 1 sum "
                    f"(yellow > {threshold_px:.1f}, x not found)"
                )
            ax.set_title(label, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    p_plan = sub.add_parser("plan", help="choose tilts and write the commands")
    p_plan.add_argument(
        "--project", required=True, help="the lane's project folder (holds artifacts.json)"
    )
    p_plan.add_argument("--series", required=True, help="tilt-series id")
    p_plan.add_argument(
        "--angles", type=float, nargs="+", default=[-75.0, 45.0, -15.0, -45.0, 15.0]
    )
    p_plan.add_argument(
        "--fractions", type=int, nargs="+", default=[],
        help="EER movies only: also run each tilt's recorded command with its frames "
        "summed in these group counts, same patch grid (frames or darkness)",
    )
    p_plan.add_argument("--out", required=True)
    p_run = sub.add_parser("run", help="run the planned MotionCor3 commands")
    p_run.add_argument("--out", required=True)
    p_cmp = sub.add_parser("compare", help="measure tile displacements and judge")
    p_cmp.add_argument("--out", required=True)
    p_cmp.add_argument("--threshold-nm", type=float, default=DEFAULT_THRESHOLD_NM)
    args = parser.parse_args(argv)
    return {"plan": plan, "run": run, "compare": compare}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
