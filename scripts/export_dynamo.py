#!/usr/bin/env python3
"""Export a project's extracted particles in the layout Dynamo and the pick review read.

    python scripts/export_dynamo.py --project <project> --variant bin8 \\
        --series TS_0201 --series TS_0388 --out <export-dir>

Writes, for the series given (numbered 1..N in that order, Dynamo's column 20):

    particles/particle_00001.mrc   each extracted subvolume as written by extraction
    initial.tbl                    Dynamo table, 42 columns, one row per particle
    export_manifest.json           per particle: tag, series, source stack and STAR
                                   row, including the PyTom score
    export_complete.json           written last; its absence means the export broke

`scripts/qc_dynamo_picks.py --export <export-dir>` then plots the score
distributions and writes a score-filtered `selected.tbl`.

`--decoy-variant <label>` adds, after every real particle, the particle stacks of
that label for the same series (random positions from `scripts/decoy_picks.py`, or
pytom's picks in phase-randomised tomograms from `scripts/null_decoys.py`, extracted
like the real picks), marks them `"decoy": true` in the manifest and says which
kind they are in its `decoy_kind`. Dynamo sees them as particles like any other;
the classification report counts where they went. The real particles keep the
tags an export without decoys gives them.

A format and pose adapter, nothing more: the voxels are copied, not processed. The
poses go through `native_to_dynamo`, the conversion the dynamo-search backend uses
and the CZII export was checked against Dynamo with. The table keeps the native
pick position plus 0.5 in columns 24-26, as that export did, and the missing wedge
as a single y-axis range per series, from the views its tomogram was reconstructed
from (the manifest's tilt angles only when the tomogram does not record them).
Since 2026-11-05 the reading and the conversion are cryoet_pipeline.particle_sources,
which an MRA run reads the registry through too; this script writes its files from
it, numbers with 12 significant digits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mrcfile

from cryoet_pipeline.backends.dynamo_search import dynamo_ytilt
from cryoet_pipeline.particle_sources import TABLE_COLUMNS, registry_particles


def format_row(row) -> str:
    return " ".join(f"{value:.12g}" for value in row)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True, type=Path,
                        help="Project with artifacts.json and manifests/")
    parser.add_argument("--series", required=True, action="append", dest="series",
                        help="Tilt-series to export, in the order to number them")
    parser.add_argument("--variant", default=None,
                        help="Variant label of the particle stacks (e.g. bin8); "
                        "leave out for unlabelled stacks")
    parser.add_argument("--out", required=True, type=Path, help="Export directory (new)")
    parser.add_argument("--decoy-variant", default=None,
                        help="Also export the particle stacks of this label (decoys), "
                        "after all real particles, marked as decoys in the manifest")
    args = parser.parse_args()
    if args.decoy_variant is not None and args.decoy_variant == args.variant:
        raise SystemExit("--decoy-variant must differ from --variant")

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} exists and is not empty; export into a new directory")
    if len(set(args.series)) != len(args.series):
        raise SystemExit("a series is given twice")
    try:
        source = registry_particles(args.project, args.variant, args.series,
                                    args.decoy_variant)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    (args.out / "particles").mkdir(parents=True, exist_ok=True)

    particles = []
    for particle in source.particles:
        name = f"particles/particle_{particle.tag:05d}.mrc"
        with mrcfile.new(args.out / name) as handle:
            handle.set_data(source.volume(particle.tag))
            handle.voxel_size = source.pixel_size
        stack = source.stacks[particle.stack_id]
        particles.append({
            "tag": particle.tag,
            "tomogram_number": particle.tomogram,
            "run": particle.series,
            "source_stack_id": stack.id,
            "source_stack": str(stack.path),
            "source_metadata": str(source.metadata_paths[stack.id]),
            "source_index_zero_based": particle.index,
            "native_record": asdict(particle.record),
            "file": name,
            "decoy": particle.decoy,
        })

    table = args.out / "initial.tbl"
    table.write_text("\n".join(format_row(row) for row in source.table) + "\n")
    decoys = sum(1 for particle in particles if particle["decoy"])
    manifest_out = {
        "schema_version": 1,
        "source_project": str(args.project.resolve()),
        "variant": args.variant,
        "decoy_variant": args.decoy_variant,
        "decoy_kind": "; ".join(source.decoy_kinds) if source.decoy_kinds else None,
        "selected_particles": len(particles),
        "decoy_particles": decoys,
        "box_size_px": source.box,
        "pixel_spacing_angstrom": source.pixel_size,
        "pose_convention": {
            "native_center_zero_based": (source.box - 1) / 2,
            "dynamo_center_zero_based": source.box / 2,
            "native_map": "x_in = cN + A @ (x_out - cN) + s",
            "dynamo_table_matrix": "M = A.T = Rz(narot) @ Rx(tilt) @ Rz(tdrot)",
            "dynamo_table_shift": "d = s + (A - I) @ (cD - cN)",
            "conversion": "cryoet_pipeline.backends.dynamo_search.native_to_dynamo",
        },
        "table_columns": TABLE_COLUMNS,
        "wedge": {
            "ranges": source.wedges,
            "range_source": source.wedge_sources,
            # What the table's ymintilt/ymaxtilt hold: each range negated.
            "dynamo_ytilt": {series: list(dynamo_ytilt(tuple(r)))
                             for series, r in source.wedges.items()},
            "axis": "y",
            "ftype": 1,
        },
        "particles": particles,
        "limitations": [
            "Voxels are copied from extraction unchanged: no normalisation, no contrast "
            "inversion, no masking.",
            "Scores are PyTom's LCCmax from the extraction STAR, kept in the manifest; "
            "the table's CC column is zero.",
            "Columns 24-26 are the native pick position plus 0.5, for reference only; "
            "the particles are already cropped.",
        ],
    }
    (args.out / "export_manifest.json").write_text(json.dumps(manifest_out, indent=1) + "\n")
    (args.out / "export_complete.json").write_text(json.dumps({
        "particles": len(particles),
        "decoys": decoys,
        "series": args.series,
        "initial.tbl": sha256(table),
        "export_manifest.json": sha256(args.out / "export_manifest.json"),
    }, indent=1) + "\n")
    print(f"exported {len(particles)} particles of {len(args.series)} series to {args.out}"
          + (f", {decoys} of them decoys ({'; '.join(source.decoy_kinds)})"
             if args.decoy_variant is not None else ""))


if __name__ == "__main__":
    main()
