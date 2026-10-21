#!/usr/bin/env python3
"""Whether each MRA class's particles could show the membrane they sit on.

    python scripts/class_orientations.py outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_c4

The membrane under a RyR1 is perpendicular to its fourfold axis, so the axis is the
membrane's normal. A plane's Fourier power lies along its normal, and a tilt series
measures a direction only if some view's beam is perpendicular to it: here, only if
some view saw the membrane edge-on. For every class of the last finished iteration
(or --iteration) this turns each member's refined angles into its axis in the
tomogram (cryoet_pipeline.backends.dynamo_search.dynamo_to_native: the reference's
z axis, A @ z) and asks the pipeline's own wedge rule (cryoet_pipeline.wedge: the
beam at tilt t is (sin t, 0, cos t), and a direction k was measured when
kx sin t + kz cos t changes sign over the series' tilt range, read from the export
manifest) whether that axis was measured. It prints, per class, how many axes were,
how far the axes lie from the beam, and the members' median CC with their class's
reference, decoys apart from real particles when the export has them.

Why (2026-10-21). In all four RyR1 runs on bin8-2x-pad the classes split by
membrane: one holds a RyR1-shaped average with no membrane under it, one RyR1 on a
curved membrane, one membrane alone. A class without a membrane is either particles
that have none there (picks beside a membrane, noise the search shaped into the
template) or RyR1 whose membrane the missing wedge removed. The picking scores cannot
tell these apart, since the template carries a membrane band that only a measured
membrane matches. This can: if the membrane-less class's axes are mostly unmeasured
and the membrane classes' mostly measured, the split is the wedge's; if all classes
look alike, it is not. In a run with decoys (pytom's picks in phase-randomised
copies, which have no membrane) the decoys go to a membrane-less class either way;
what tells is their CC there against the real members': as high if the class is
noise itself, lower if its particles carry a structure the decoys lack.

The axis taken is the reference's z axis, the C4 axis of a run with c4 imposed. In a
C1 run a class average can turn a few degrees (4.7 in mra_bbn), which moves a few
axes across the boundary and changes no verdict. Disposable, like the rest of
scripts/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cryoet_pipeline.backends.dynamo_search import dynamo_to_native
from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification, read_table

_BEAM_BANDS = ((0, 30), (30, 60), (60, 90))


def measured(axis: np.ndarray, tilt_range: tuple[float, float]) -> bool:
    """Whether a series over `tilt_range` (deg) measured the Fourier direction `axis` (x, y, z)."""

    low, high = np.deg2rad(tilt_range)
    at_low = axis[0] * np.sin(low) + axis[2] * np.cos(low)
    at_high = axis[0] * np.sin(high) + axis[2] * np.cos(high)
    return bool(at_low * at_high <= 1e-12)


def axis_of(angles: np.ndarray, box: int) -> np.ndarray:
    """The reference's z axis in the particle's (tomogram's) frame, (x, y, z)."""

    matrix, _ = dynamo_to_native(angles, np.zeros(3), box)
    return matrix @ np.array([0.0, 0.0, 1.0])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("project", type=Path, help="the Dynamo MRA project folder")
    parser.add_argument("--iteration", type=int, default=None,
                        help="which finished iteration's classes (default the last)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="export_manifest.json; default the one beside the project")
    args = parser.parse_args()

    project = args.project.resolve()
    manifest_path = args.manifest or project.parent / "export_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no {manifest_path}: the tilt ranges come from the export manifest")
    ranges = {series: tuple(map(float, pair)) for series, pair in
              json.loads(manifest_path.read_text())["wedge"]["ranges"].items()}
    run = read_dynamo_classification(project, manifest=manifest_path)
    finished = {iteration.number: iteration for iteration in run.iterations}
    if not finished:
        raise SystemExit(f"{project}: no finished iteration")
    number = args.iteration if args.iteration is not None else max(finished)
    if number not in finished:
        raise SystemExit(f"{project}: iteration {number} is not finished; finished: "
                         f"{sorted(finished)}")

    print(f"{run.name}, iteration {number}: each member's axis against its series' tilt "
          f"range ({manifest_path.name})")
    print(f"  {'class':15s} {'particles':>9s}   {'membrane measured':>17s}   {'median CC':>9s}   "
          "axis from the beam: " + "  ".join(f"{low}-{high}" for low, high in _BEAM_BANDS) + " deg")
    rows: dict[str, list[tuple[bool, float, float]]] = {}
    for channel, result in sorted(finished[number].classes.items()):
        table_path = (project / "results" / f"ite_{number:04d}" / "averages"
                      / f"refined_table_ref_{channel:03d}_ite_{number:04d}.tbl")
        table = read_table(table_path)
        by_tag = {int(row[0]): row for row in table}
        for kind in ("particles", "decoys"):
            tags = [int(tag) for tag in result.members
                    if (int(tag) in run.decoys) == (kind == "decoys")]
            if not tags:
                continue
            found = []
            for tag in tags:
                series = run.series.get(tag)
                if series not in ranges:
                    raise SystemExit(f"tag {tag}: no tilt range for series {series!r}")
                axis = axis_of(by_tag[tag][6:9], run.box_px)
                found.append((measured(axis, ranges[series]),
                              float(np.degrees(np.arccos(min(1.0, abs(axis[2]))))),
                              float(by_tag[tag][9])))
            name = f"class {channel}" + ("" if kind == "particles" else " decoys")
            rows[name] = found
    totals = [item for name, found in rows.items() if "decoys" not in name for item in found]
    rows["all"] = totals
    for name, found in rows.items():
        count = len(found)
        seen = sum(1 for visible, _, _ in found if visible)
        angles = np.array([angle for _, angle, _ in found])
        shares = [np.mean((angles >= low) & ((angles < high) | (high == 90)))
                  for low, high in _BEAM_BANDS]
        # Column 10: each particle's CC with its own class's reference, as the report's median.
        correlation = float(np.median([cc for _, _, cc in found]))
        print(f"  {name:15s} {count:9,d}   {seen:9,d} ({seen / count:4.0%})   {correlation:9.3f}   "
              f"{'':19s}" + "  ".join(f"{share:5.0%}" for share in shares))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
