#!/usr/bin/env python3
"""Split an export into side views and top views: membrane measured or not.

    python scripts/split_export_by_view.py outputs/ryr1/dynamo/bin8-2x-pad

writes outputs/ryr1/dynamo/bin8-2x-pad-side and outputs/ryr1/dynamo/bin8-2x-pad-top,
each an export of scripts/export_dynamo.py in its own right: particles/ (links to the
source's files, tags unchanged), initial.tbl and export_manifest.json holding that
group alone, and export_complete.json. scripts/dynamo_mra_mdc.sbatch and the
classification report run on either as on the source. --side-out and --top-out name
other folders; neither may hold anything yet.

The rule is scripts/class_orientations.py's, at the picks' own poses, the source's
initial.tbl: a particle's fourfold axis, the reference's z axis turned by its angles,
is the normal of the membrane under a RyR1, and that membrane was measured when some
view of the particle's series saw it edge-on (cryoet_pipeline.wedge's rule, with the
series' tilt range from the manifest). Side: measured. Top: not; the membrane lies
too close to the lamella's plane, and the missing wedge removed it. Decoys are split
by the same rule on their own poses, so each group's decoys point as its particles
do.

Why (2026-10-21). Every MRA run on bin8-2x-pad sorted the particles by this before
anything else: the class without a membrane held 10-15% measurable axes against
74-82% in the membrane classes, at the picks' poses as at the refined ones, and
among top views the classification could not tell RyR1 from junk at all. Apart,
each group varies in less: side views in whether a RyR1 sits on the membrane, top
views in whether a RyR1 is there. On bin8-2x-pad the split should give the 1,364
measurable particles class_orientations.py --starting counted, and 1,020 others.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from class_orientations import axis_of, measured

from cryoet_pipeline.reports.dynamo_project import read_table

RULE = ("side when the particle's axis (the reference's z axis turned by its initial.tbl "
        "angles) is a direction its series' tilt range measured, by cryoet_pipeline.wedge's "
        "rule: the membrane under it was seen edge-on; top otherwise")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_subexport(source: Path, out: Path, chosen: list[tuple[int, str]],
                    manifest: dict[str, Any], record: dict[str, Any]) -> int:
    """An export of `chosen`, (tag, initial.tbl line) pairs of `source`'s particles, in `out`.

    Links to the source's particle files, those lines as initial.tbl, the source's
    manifest cut to them with `record` added, and export_complete.json. Returns how
    many of them are decoys.
    """

    entries = {int(entry["tag"]): entry for entry in manifest["particles"]}
    complete = json.loads((source / "export_complete.json").read_text())
    (out / "particles").mkdir(parents=True, exist_ok=True)
    kept = [entries[tag] for tag, _ in chosen]
    for entry in kept:
        (out / entry["file"]).symlink_to((source / entry["file"]).resolve())
    (out / "initial.tbl").write_text("\n".join(line for _, line in chosen) + "\n")
    decoys = sum(1 for entry in kept if entry.get("decoy"))
    (out / "export_manifest.json").write_text(json.dumps({
        **manifest,
        "selected_particles": len(kept),
        "decoy_particles": decoys,
        "particles": kept,
        **record,
    }, indent=1) + "\n")
    (out / "export_complete.json").write_text(json.dumps({
        "particles": len(kept),
        "decoys": decoys,
        "series": complete.get("series"),
        "initial.tbl": _sha256(out / "initial.tbl"),
        "export_manifest.json": _sha256(out / "export_manifest.json"),
        "split_from": str(source),
    }, indent=1) + "\n")
    return decoys


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export", type=Path, help="an export folder of scripts/export_dynamo.py")
    parser.add_argument("--side-out", type=Path, default=None, help="default <export>-side")
    parser.add_argument("--top-out", type=Path, default=None, help="default <export>-top")
    args = parser.parse_args()

    source = args.export.resolve()
    outs = {"side": (args.side_out or source.with_name(source.name + "-side")).resolve(),
            "top": (args.top_out or source.with_name(source.name + "-top")).resolve()}
    if not (source / "export_complete.json").is_file():
        raise SystemExit(f"{source}: no export_complete.json, so no finished export to split")
    for out in outs.values():
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not empty; split into new folders")

    manifest = json.loads((source / "export_manifest.json").read_text())
    ranges = {series: (float(low), float(high))
              for series, (low, high) in manifest["wedge"]["ranges"].items()}
    box = int(manifest["box_size_px"])
    entries = {int(entry["tag"]): entry for entry in manifest["particles"]}
    lines = [line for line in (source / "initial.tbl").read_text().splitlines() if line.strip()]
    table = read_table(source / "initial.tbl")
    if len(lines) != len(table) or {int(row[0]) for row in table} != set(entries):
        raise SystemExit(f"{source}: initial.tbl and export_manifest.json list different particles")

    groups: dict[str, list[tuple[int, str]]] = {"side": [], "top": []}
    for line, row in zip(lines, table, strict=True):
        tag = int(row[0])
        series = str(entries[tag]["run"])
        if series not in ranges:
            raise SystemExit(f"tag {tag}: the manifest gives no tilt range for series {series!r}")
        view = "side" if measured(axis_of(row[6:9], box), ranges[series]) else "top"
        groups[view].append((tag, line))

    for view, out in outs.items():
        chosen = groups[view]
        other = outs["top" if view == "side" else "side"]
        decoys = write_subexport(source, out, chosen, manifest, {
            "split": {"from": str(source), "view": view, "rule": RULE,
                      "particles": len(chosen), "other_group": str(other)},
        })
        kept = [entries[tag] for tag, _ in chosen]
        by_series = Counter(str(entry["run"]) for entry in kept if not entry.get("decoy"))
        print(f"{view}: {len(kept) - decoys:,} particles"
              + (f" and {decoys:,} decoys" if decoys else "") + f" in {out}; by series: "
              + ", ".join(f"{series} {count}" for series, count in sorted(by_series.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
