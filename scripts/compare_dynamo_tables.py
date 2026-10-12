#!/usr/bin/env python3
"""Per-particle differences between two Dynamo projects, iteration by iteration.

    python scripts/compare_dynamo_tables.py --first gag_half_eo --second gag_gpu_em_eo

Run from the folder that holds both projects. For every iteration and reference
present in both, reads results/ite_NNNN/averages/refined_table_ref_RRR_ite_NNNN.tbl,
joins the rows on the tag (column 1) and prints, over the particles the two share:
the rotation between the two poses (degrees, from the Euler angles in columns 7-9),
the distance between the two shifts (pixels, columns 4-6) and the CC difference
(column 10, second minus first). Also the largest |shift| per axis in each, since
a different shift limit would show there before anywhere else.

Built for Dynamo GPU against Dynamo CPU on the same particles and cards: their
spread is the tolerance any GPU port of our search can be held to. Disposable,
like the rest of scripts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cryoet_pipeline.backends.dynamo_search import euler_matrix


def read_table(path: Path) -> dict[int, np.ndarray]:
    rows = np.atleast_2d(np.loadtxt(path))
    return {int(row[0]): row for row in rows}


def rotation_deg(first_angles, second_angles) -> float:
    first, second = euler_matrix(first_angles), euler_matrix(second_angles)
    cosine = (np.trace(first @ second.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def tables(project: Path) -> dict[tuple[int, int], Path]:
    found = {}
    for path in project.glob("results/ite_*/averages/refined_table_ref_*_ite_*.tbl"):
        parts = path.stem.split("_")  # refined_table_ref_RRR_ite_NNNN
        found[(int(parts[-1]), int(parts[-3]))] = path
    return found


def spread(values: np.ndarray) -> str:
    return (f"median {np.median(values):7.3f}  p95 {np.percentile(values, 95):7.3f}  "
            f"max {np.max(values):7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=Path, required=True, help="reference project, e.g. the CPU one")
    parser.add_argument("--second", type=Path, required=True, help="project compared against it")
    parser.add_argument("--outlier-deg", type=float, default=10.0, help="rotation counted as an outlier")
    parser.add_argument("--worst", type=int, default=5, help="particles listed, largest rotation first")
    args = parser.parse_args()

    first, second = tables(args.first), tables(args.second)
    common = sorted(set(first) & set(second))
    if not common:
        raise SystemExit(f"no iteration/reference refined in both {args.first} and {args.second}")

    for ite, ref in common:
        a, b = read_table(first[(ite, ref)]), read_table(second[(ite, ref)])
        tags = sorted(set(a) & set(b))
        only = len(set(a) ^ set(b))
        rotation = np.array([rotation_deg(a[t][6:9], b[t][6:9]) for t in tags])
        shift = np.array([np.linalg.norm(b[t][3:6] - a[t][3:6]) for t in tags])
        cc = np.array([b[t][9] - a[t][9] for t in tags])
        identical = sum(np.array_equal(a[t][3:10], b[t][3:10]) for t in tags)
        print(f"ite {ite} ref {ref:03d}: {len(tags)} shared particles, {only} in one table only, "
              f"{identical} with identical shifts, angles and CC")
        print(f"  rotation, deg  {spread(rotation)}")
        print(f"  shift, px      {spread(shift)}")
        print(f"  CC 2nd - 1st   median {np.median(cc):+7.4f}  min {np.min(cc):+7.4f}  max {np.max(cc):+7.4f}")
        for name, table in ((args.first.name, a), (args.second.name, b)):
            largest = np.max(np.abs(np.array([table[t][3:6] for t in tags])), axis=0)
            print(f"  max |shift| x,y,z in {name}: {largest[0]:.3f} {largest[1]:.3f} {largest[2]:.3f}")
        print(f"  rotation > {args.outlier_deg:g} deg: {int(np.sum(rotation > args.outlier_deg))} particles")
        for i in np.argsort(rotation)[::-1][:args.worst]:
            t = tags[i]
            print(f"    tag {t:5d}  rotation {rotation[i]:6.2f}  shift {shift[i]:5.2f}  "
                  f"CC {a[t][9]:.3f} / {b[t][9]:.3f}")


if __name__ == "__main__":
    main()
