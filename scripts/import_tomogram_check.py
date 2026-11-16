#!/usr/bin/env python3
"""Measure IMOD's answers on import-tomogram's orientation list, with IMOD itself.

docs/entry_and_exit.md keeps a list of orientations, each with the measurement
that showed it right. This script is that measurement for IMOD's three answers,
made on a tilt series the pipeline has. It reconstructs a slab of the series with
`tilt` in each of the ways IMOD writes a volume:

- `tilt` as it is (Y as sections): imod-tilt;
- `tilt -PARALLEL`, and `clip flipyz` of the first: imod-parallel;
- `tilt -RotateBy90`, `clip rotx` and `trimvol -rx` of the first: imod-standard;

imports each with `import_tomogram`, giving the `.tlt` `tilt` was given, and
checks two things:

1. every import gives the same canonical volume, the one reconstruct-tomogram
   makes from `tilt`'s output (`transpose(1, 0, 2)`);
2. the wedge seen in that volume agrees with the `.tlt` as it is (the angles enter
   unchanged), and the import refuses a volume answered as its mirror (a
   standard-orientation volume answered imod-parallel).

It runs twice: on the views as they are, and without the views above an angle,
which makes the range asymmetric enough for the wedge to tell a mirror. On
EMPIAR-10164 TS_01 at bin8 (outputs/dev-ts01-registry):

    scripts/import_tomogram_check.py \\
        --stack outputs/dev-ts01-registry/aligned/TS_01/TS_01_final_bin8.st \\
        --tilt-file outputs/dev-ts01-registry/aligned/TS_01/TS_01_final_bin8.tlt \\
        --x-axis-tilt -1.62 --thickness 296 --work <scratch folder>

It prints what it found and exits non-zero when a prediction fails.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from cryoet_pipeline.tomogram_import import ImportRefused, import_tomogram


def run(command: list[str], imod: Path, cwd: Path) -> None:
    environment = dict(os.environ)
    environment["IMOD_DIR"] = str(imod)
    environment["PATH"] = f"{imod / 'bin'}{os.pathsep}{environment.get('PATH', '')}"
    result = subprocess.run(command, cwd=cwd, env=environment, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")


def reconstruct(args: argparse.Namespace, folder: Path, exclude: list[int]) -> dict[str, Path]:
    """The slab, written each way IMOD writes it; keys are the orientations' names."""

    imod = Path(args.imod_dir)
    tilt = str(imod / "bin" / "tilt")
    base = [
        tilt, "-input", str(Path(args.stack).resolve()),
        "-TILTFILE", str(Path(args.tilt_file).resolve()),
        "-THICKNESS", str(args.thickness), "-XAXISTILT", f"{args.x_axis_tilt:.6f}",
        "-SLICE", f"{args.slices[0]},{args.slices[1]}",
        "-RADIAL", "0.35,0.05", "-FalloffIsTrueSigma", "-MODE", "2",
    ]
    if exclude:
        base += ["-EXCLUDELIST", ",".join(str(view) for view in exclude)]
    volumes = {
        "imod-tilt": folder / "tilt.rec",
        "imod-parallel": folder / "parallel.rec",
        "imod-standard": folder / "rotate_by_90.rec",
        "imod-parallel (clip flipyz)": folder / "flipyz.rec",
        "imod-standard (clip rotx)": folder / "rotx.rec",
        "imod-standard (trimvol -rx)": folder / "trimvol_rx.rec",
    }
    run([*base, "-output", str(volumes["imod-tilt"])], imod, folder)
    run([*base, "-PARALLEL", "-output", str(volumes["imod-parallel"])], imod, folder)
    run([*base, "-RotateBy90", "-output", str(volumes["imod-standard"])], imod, folder)
    clip = str(imod / "bin" / "clip")
    run([clip, "flipyz", str(volumes["imod-tilt"]),
         str(volumes["imod-parallel (clip flipyz)"])], imod, folder)
    run([clip, "rotx", str(volumes["imod-tilt"]), str(volumes["imod-standard (clip rotx)"])],
        imod, folder)
    run([str(imod / "bin" / "trimvol"), "-rx", str(volumes["imod-tilt"]),
         str(volumes["imod-standard (trimvol -rx)"])], imod, folder)
    return volumes


def check(args: argparse.Namespace, label: str, exclude: list[int]) -> list[str]:
    folder = Path(args.work) / label
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    all_angles = [float(word) for word in Path(args.tilt_file).read_text().split()]
    kept = [angle for view, angle in enumerate(all_angles, start=1) if view not in exclude]
    angles = folder / "kept.tlt"
    angles.write_text("".join(f"{angle:.6f}\n" for angle in kept))
    volumes = reconstruct(args, folder, exclude)

    failures: list[str] = []
    canonical: dict[str, Any] = {}
    print(f"\n{label}: {len(kept)} views, {min(kept):g} to {max(kept):g} deg")
    for index, (name, path) in enumerate(volumes.items()):
        answer = name.split(" ")[0]
        result = import_tomogram(
            volume_path=path, tilt_angles_path=angles, orientation_name=answer,
            processing="raw", tilt_series_id="TS", out=folder / "project",
            declared_pixel_size=None, variant=f"v{index}",
        )
        canonical[name] = np.asarray(zarr.open(result.tomogram.path, mode="r")[:])
        print(f"  {name:30s} -> {canonical[name].shape}, wedge {result.wedge['verdict']}: "
              f"{result.wedge['reason']}")
        if result.wedge["verdict"] not in {"agrees", "side not testable"}:
            failures.append(f"{label}: {name}'s wedge says {result.wedge['verdict']}")

    reference = canonical["imod-tilt"]
    for name, volume in canonical.items():
        if volume.shape != reference.shape:
            failures.append(
                f"{label}: {name} has shape {volume.shape}, imod-tilt {reference.shape}"
            )
            continue
        difference = float(np.max(np.abs(volume - reference)))
        scale = float(np.max(np.abs(reference)))
        print(f"  {name:30s} max |difference| from imod-tilt's: {difference:.3g} "
              f"(of {scale:.3g})")
        if difference > 1e-5 * scale:
            failures.append(f"{label}: {name} differs from imod-tilt's canonical volume")

    try:
        import_tomogram(
            volume_path=volumes["imod-standard (clip rotx)"], tilt_angles_path=angles,
            orientation_name="imod-parallel", processing="raw", tilt_series_id="TS",
            out=folder / "project", variant="wrong",
        )
    except ImportRefused as refusal:
        print(f"  clip rotx answered imod-parallel: refused ({str(refusal)[:160]}...)")
        mirrored = "mirror" in str(refusal)
    else:
        print("  clip rotx answered imod-parallel: imported")
        mirrored = False
    centre = (min(kept) + max(kept)) / 2
    if abs(centre) >= 3.0 and not mirrored:
        failures.append(f"{label}: a mirrored answer was not refused (range centre {centre:+.1f})")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stack", required=True, help="The aligned stack tilt reads.")
    parser.add_argument("--tilt-file", required=True, help="The .tlt tilt is given.")
    parser.add_argument("--x-axis-tilt", type=float, default=0.0)
    parser.add_argument("--thickness", type=int, required=True)
    parser.add_argument("--slices", type=int, nargs=2, default=(200, 727),
                        help="The slab: first and last slice, 0-based (default 200 727); "
                        "more slices than the thickness, so that z stays the thin axis.")
    parser.add_argument("--above", type=float, default=16.0,
                        help="The second run leaves out the views above this angle.")
    parser.add_argument("--work", required=True, help="Scratch folder; its contents are replaced.")
    parser.add_argument("--imod-dir", default=os.environ.get("IMOD_DIR", "/Applications/IMOD"))
    args = parser.parse_args(argv)

    angles = [float(word) for word in Path(args.tilt_file).read_text().split()]
    above = [view for view, angle in enumerate(angles, start=1) if angle > args.above]
    failures = check(args, "all-views", [])
    failures += check(args, f"views-up-to-{args.above:g}", above)
    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASSED: every IMOD orientation gives imod-tilt's canonical volume, the wedge agrees "
          "with the .tlt as it is, and a mirrored answer is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
