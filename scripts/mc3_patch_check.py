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
             (normalized cross-correlation after a band-pass, +-112 binned px):
               control: the pipeline's own sum against p5, which should be the
                        same image (the run is reproducible);
               effect:  p5 against p1.
             The common shift of all tiles is removed; what is left, in raw
             pixels, is how far the patch alignment moved each tile. Prints a
             table and the verdict, and writes compare.png.

Prediction, written before the run: at -75 and +45 some tile of the p5 sum sits
more than 7.9 raw px from where it sits in the p1 sum (7.9 px = 1.5 nm at
1.9037 A, the fine-alignment failure threshold); at -15 none does.

Disposable; scripts/mc3_patch_check.sbatch runs the three steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
# Tolerance for "the same image" in the control, raw px.
REPRODUCIBLE_PX = 2.0


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
        runs = {}
        for label, patches in (("p5", None), ("p1", (1, 1))):
            runs[label] = {
                "out_mrc": str(out / f"{stem}_{label}.mrc"),
                "stdout": str(out / "logs" / f"{stem}_{label}.out"),
                "command": rewrite_command(
                    command,
                    patches=patches,
                    out_mrc=out / f"{stem}_{label}.mrc",
                    log_dir=out / "logs" / f"{stem}_{label}",
                    aln_dir=out / "aln" / f"{stem}_{label}",
                    fmint=fmint,
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
                "common_px": None, "residual": None, "matched_mask": matched}
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
        "matched_mask": matched,
    }


def compare(args: argparse.Namespace) -> int:
    out = Path(args.out)
    document = json.loads((out / "plan.json").read_text())
    rows = []
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
        control = displacement_summary(tile_shifts(b_pipe, b_p5, grid), factor)
        effect = displacement_summary(tile_shifts(b_p5, b_p1, grid), factor)
        same = float(np.corrcoef(b_pipe.ravel(), b_p5.ravel())[0, 1])
        rows.append((tilt, factor, control, effect, same, (b_pipe, b_p5, b_p1)))

    # The sums' pixel: "raw px" below are pixels of the sums, before binning.
    pixel = rows[0][0]["pixel_angstrom"] if rows else 1.0
    threshold_px = args.threshold_nm * 10.0 / pixel
    print(f"sums' pixel {pixel:.4f} A; threshold {args.threshold_nm} nm = {threshold_px:.1f} px")
    print("angle    control: pipeline sum vs p5 rerun       effect: p5 vs p1 (-Patch 1 1)")
    print(
        "         r      matched  max px              "
        "matched  max px  median px  common shift px"
    )
    reproducible = True
    verdicts: dict[float, bool | None] = {}
    for tilt, _factor, control, effect, same, _images in rows:
        angle = tilt["tilt_angle_deg"]
        control_ok = (
            control["max_px"] is not None
            and control["max_px"] <= REPRODUCIBLE_PX
            and same > 0.99
        )
        reproducible &= control_ok
        if effect["max_px"] is None or effect["matched"] < effect["tiles"] // 2:
            verdicts[angle] = None
        else:
            verdicts[angle] = effect["max_px"] > threshold_px

        def fmt(value: float | None) -> str:
            return "   -  " if value is None else f"{value:6.1f}"

        common = effect["common_px"]
        common_text = "-" if common is None else f"({common[0]:+.1f}, {common[1]:+.1f})"
        print(
            f"{angle:6.1f}   {same:6.4f}  {control['matched']:2d}/{control['tiles']:2d}  "
            f"{fmt(control['max_px'])}              {effect['matched']:2d}/{effect['tiles']:2d}  "
            f"{fmt(effect['max_px'])}  {fmt(effect['median_px'])}     {common_text}"
        )
    print()
    if not reproducible:
        print("CONTROL FAILED: a rerun as the pipeline ran it is not the pipeline's sum.")
        print("Do not read the effect: compare the binary, the flags and the inputs first.")
    else:
        print("control: each rerun as the pipeline ran it reproduces the pipeline's sum.")
    for angle, moved in verdicts.items():
        text = {
            None: "not measurable (fewer than half the tiles matched)",
            True: f"tiles moved by more than {args.threshold_nm} nm",
            False: f"no tile moved by more than {args.threshold_nm} nm",
        }[moved]
        print(f"{angle:6.1f} deg: {text}")
    ends = [angle for angle in verdicts if abs(angle - (-75)) < 1.6 or abs(angle - 45) < 1.6]
    mid = [angle for angle in verdicts if abs(angle - (-15)) < 1.6]
    if reproducible and ends and mid:
        held = all(verdicts[a] is True for a in ends) and all(verdicts[a] is False for a in mid)
        print()
        print("prediction (ends moved, -15 not): " + ("HELD" if held else "DID NOT HOLD"))
    _figure(rows, out / "compare.png", threshold_px)
    print(f"wrote {out / 'compare.png'}")
    return 0 if reproducible else 2


def _figure(rows: list[Any], path: Path, threshold_px: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 5.2 * len(rows)), squeeze=False)
    titles = ("pipeline sum (as run)", "rerun as the pipeline ran it", "rerun with -Patch 1 1")
    for row, (tilt, factor, _control, effect, _same, images) in zip(axes, rows, strict=True):
        nx, ny = tilt["pipeline_patches"]
        for k, (ax, image, title) in enumerate(zip(row, images, titles, strict=True)):
            shown = gaussian_filter(image, 1.0)
            lo, hi = np.percentile(shown, [1, 99])
            h, w = image.shape
            ax.imshow(shown, cmap="gray", vmin=lo, vmax=hi, origin="lower", extent=(0, w, 0, h))
            for i in range(1, nx):
                ax.axvline(w * i / nx, color="red", lw=0.6, alpha=0.5)
            for j in range(1, ny):
                ax.axhline(h * j / ny, color="red", lw=0.6, alpha=0.5)
            ax.set_title(f"{tilt['tilt_angle_deg']:.1f} deg: {title}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 1 and effect["residual"] is not None:
                residual = effect["residual"] / factor
                matched = effect["matched_mask"]
                for j in range(ny):
                    for i in range(nx):
                        cx, cy = (i + 0.5) * w / nx, (j + 0.5) * h / ny
                        if not matched[j, i]:
                            ax.plot(cx, cy, "x", color="orange", ms=8)
                            continue
                        dy, dx = residual[j, i]
                        magnitude = float(np.hypot(dy, dx) * factor)
                        colour = "yellow" if magnitude > threshold_px else "cyan"
                        # Arrows are true length in the binned image shown.
                        ax.annotate(
                            "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color=colour, lw=1.5),
                        )
                        ax.text(cx, cy - 0.12 * h / ny, f"{magnitude:.0f}", color=colour,
                                ha="center", va="top", fontsize=9, fontweight="bold")
                ax.set_title(
                    f"{tilt['tilt_angle_deg']:.1f} deg: rerun as run; numbers = raw px each "
                    f"tile sits from the -Patch 1 1 sum (yellow > {threshold_px:.1f})",
                    fontsize=9,
                )
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
