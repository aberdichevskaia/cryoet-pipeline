#!/usr/bin/env python3
"""Re-extract picks from a saved score volume, at several exclusion distances.

    python3 scripts/reextract_candidates.py \\
        --job outputs/ribo/picks/TS_0006/TS_0006_full_pytom-match-pick_bin8-mask16_job.json \\
        --executable $PYTOMO_MP_DIR/pytom_extract_candidates.py \\
        --particle-diameter 290 580

Template matching costs about half an hour of A100 and produces a correlation
volume; extracting candidates from that volume takes seconds. `pick-particles`
does both together, so until now every question about a cut-off or an exclusion
distance meant paying for the match again. With `--pytom-keep-score-volume` the
volume survives, and this reads it.

The question it exists for: pytom removes further peaks within HALF the diameter
it is given, which this repository's own docstring states. So
`--particle-diameter 290` permits two picks 145 A apart, and across five series
19% of picks have a neighbour closer than 290 A — closer than two ribosome
centres can physically be. Passing 580 makes the exclusion one true diameter.

Disposable, like the rest of `scripts/`. Nothing here is registered: it writes
STAR files into a scratch directory and prints numbers. The proper version is an
`extract-picks` command that takes a picks artifact's job and registers a new
one, which is worth building once the answer here says the parameter matters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _read_star_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Scores and XYZ coordinates from a pytom particle STAR."""

    lines = [line.rstrip("\n") for line in path.read_text().splitlines()]
    columns = [line.split()[0].lstrip("_") for line in lines if line.startswith("_")]
    rows = [
        line.split()
        for line in lines
        if line.strip() and not line.startswith(("data_", "loop_", "_", "#"))
    ]
    if not rows:
        return np.zeros(0), np.zeros((0, 3))
    index = {name: position for position, name in enumerate(columns)}
    scores = np.array([float(row[index["rlnLCCmax"]]) for row in rows])
    xyz = np.array(
        [
            [
                float(row[index["rlnCoordinateX"]]),
                float(row[index["rlnCoordinateY"]]),
                float(row[index["rlnCoordinateZ"]]),
            ]
            for row in rows
        ]
    )
    return scores, xyz


def _one_false_positive_z(search_space: float) -> float:
    low, high = 0.0, 40.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if search_space * 0.5 * math.erfc(middle / math.sqrt(2.0)) > 1.0:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _rebased_job(job: Path, work: Path, args) -> dict:
    """The job with its temporary-directory paths pointed at what exists now.

    The match ran in a temporary directory, so the job records three paths inside
    it: the MRC copy of the tomogram, the output directory, and the whitening
    filter. Moving the job out of that directory does not move them, which is why
    re-extraction failed with `FileNotFoundError` on the tomogram.

    The tomogram is rebuilt from the canonical Zarr rather than kept as a third
    gigabyte. The output directory becomes this scratch. The whitening filter has
    no fallback — if it was not kept, its absence is reported rather than worked
    around, because dropping it silently would change the score scale and make
    the comparison this script exists for meaningless.
    """

    payload = json.loads(job.read_text())
    payload["output_dir"] = str(work)

    recorded_tomogram = payload.get("tomogram")
    if isinstance(recorded_tomogram, str) and not Path(recorded_tomogram).exists():
        if args.tomogram is None:
            raise SystemExit(
                "the job points at an MRC copy of the tomogram that lived in the "
                f"match's temporary directory and is gone:\n  {recorded_tomogram}\n"
                "Pass --tomogram with the canonical Zarr and it will be rebuilt here."
            )
        from cryoet_pipeline.backends.picking import _write_zarr_to_mrc

        rebuilt = work / Path(recorded_tomogram).name
        print(f"  rebuilding {rebuilt.name} from {args.tomogram} ...", flush=True)
        _write_zarr_to_mrc(
            args.tomogram, rebuilt, voxel_spacing_angstrom=args.apix or 1.0
        )
        payload["tomogram"] = str(rebuilt)

    whitening = payload.get("whitening_filter")
    if isinstance(whitening, str) and not Path(whitening).exists():
        print(
            "  NOTE: the whitening filter was not kept. Extraction may recompute or "
            "refuse it; if the scores come back on a different scale, this run is "
            "not comparable with the original and needs one match with it retained."
        )
    return payload


def _report_job_paths(job: Path) -> None:
    """Print the paths the job JSON holds, and whether they still exist.

    The first suspect when re-extraction fails: the match ran in a temporary
    directory, so a job JSON written there records paths inside it, and moving
    the job to a project directory does not move what it points at. If these
    read as gone, the saved volumes are not re-readable without rewriting the
    job, and that is a defect in how the match is persisted rather than in this
    script.
    """

    try:
        payload = json.loads(job.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"    (could not read the job JSON: {exc})")
        return
    print("    paths recorded inside the job JSON:")
    found = False
    for key, value in sorted(payload.items()):
        if not isinstance(value, str) or "/" not in value:
            continue
        found = True
        state = "exists" if Path(value).exists() else "GONE"
        print(f"      {key} = {value}  [{state}]")
    if not found:
        print("      (none — the job holds no path-like values at the top level)")
        print(f"      top-level keys: {', '.join(sorted(payload))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=Path, help="Saved *_job.json.")
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument(
        "--particle-diameter",
        required=True,
        type=float,
        nargs="+",
        help="One or more diameters to sweep. pytom excludes within HALF of each.",
    )
    parser.add_argument("--number-of-particles", type=int, default=2000)
    parser.add_argument("--number-of-false-positives", type=float, default=100.0)
    parser.add_argument("--apix", type=float, default=None, help="For neighbour distances.")
    parser.add_argument(
        "--tomogram",
        type=Path,
        default=None,
        help="Canonical tomogram Zarr. The job JSON points at an MRC copy that "
        "lived in the match's temporary directory, so it is rebuilt here rather "
        "than kept: the tomogram is already stored as Zarr and converting it is "
        "cheap and deterministic, which is a gigabyte per series not spent.",
    )
    args = parser.parse_args()

    job = args.job.resolve()
    if not job.is_file():
        print(f"no job JSON at {job}")
        return 1
    stem = job.name.removesuffix("_job.json")
    volumes = [job.parent / f"{stem}_scores.mrc", job.parent / f"{stem}_angles.mrc"]
    missing = [one for one in volumes if not one.is_file()]
    if missing:
        print("the score volumes are not beside the job:")
        for one in missing:
            print(f"  {one}")
        print("\nRe-run pick-particles with --pytom-keep-score-volume to produce them.")
        return 1

    stats = json.loads(job.read_text()).get("job_stats") or {}
    sigma, search_space = stats.get("std"), stats.get("search_space")
    floor = _one_false_positive_z(float(search_space)) if search_space else None

    print(f"{'diameter':>9} {'exclusion':>10} {'picks':>6} {'z_cut':>7} {'z_max':>7} "
          f"{'>floor':>7} {'min NN':>8}")
    print("-" * 60)

    for diameter in args.particle_diameter:
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            # Symlinked, not copied: these are about a gigabyte each.
            for source in volumes:
                (work / source.name).symlink_to(source)
            job_copy = work / job.name
            job_copy.write_text(json.dumps(_rebased_job(job, work, args)))
            command = [
                str(args.executable),
                "-j", str(job_copy),
                "-n", str(args.number_of_particles),
                "--number-of-false-positives", f"{args.number_of_false_positives:.6f}",
                "--particle-diameter", f"{diameter:.6f}",
            ]
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no output").strip()
                print(f"\n{diameter:>9.0f}  extraction FAILED. Command:")
                print("    " + " ".join(command))
                # The exception is on the LAST lines of a traceback, so showing
                # the first 120 characters hides it — which is what the first
                # version of this did, costing a round trip.
                tail = detail.splitlines()[-12:]
                for line in tail:
                    print(f"    {line}")
                _report_job_paths(job)
                print()
                continue
            stars = sorted(work.glob("*_particles.star"))
            if not stars:
                print(f"{diameter:>9.0f}  no STAR written")
                continue
            scores, xyz = _read_star_scores(stars[0])
            if scores.size == 0:
                print(f"{diameter:>9.0f} {diameter/2:>10.0f} {0:>6}")
                continue
            z_cut = scores.min() / sigma if sigma else float("nan")
            z_max = scores.max() / sigma if sigma else float("nan")
            above = int((scores / sigma > floor).sum()) if (sigma and floor) else -1
            nn = ""
            if args.apix and xyz.shape[0] > 1:
                points = xyz * args.apix
                distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
                np.fill_diagonal(distances, np.inf)
                nn = f"{distances.min():8.0f}"
            print(f"{diameter:>9.0f} {diameter/2:>10.0f} {scores.size:>6} {z_cut:>7.2f} "
                  f"{z_max:>7.2f} {above:>7} {nn}")
            shutil.copy(stars[0], job.parent / f"{stem}_d{int(diameter)}_particles.star")

    if floor is not None:
        print(f"\none-false-positive floor: {floor:.2f} sigma")
    print("STAR files written beside the job, suffixed with the diameter used.")
    print("Counts are what to compare: the null predicts about "
          f"{args.number_of_false_positives:.0f} picks at the cut-off regardless of "
          "the exclusion distance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
