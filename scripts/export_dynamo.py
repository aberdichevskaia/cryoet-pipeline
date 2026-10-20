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
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import mrcfile
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import dynamo_ytilt, native_to_dynamo
from cryoet_pipeline.models import ArtifactKind, TiltSeriesManifest
from cryoet_pipeline.particles import read_particle_star
from cryoet_pipeline.rotations import relion_euler_to_matrix

TABLE_COLUMNS = 42


def particle_stack(registry: ArtifactRegistry, series: str, variant: str | None):
    """The one particle stack of `series` in `variant` (None: the unlabelled one)."""
    found = [
        artifact
        for artifact in registry.by_kind(ArtifactKind.PARTICLE_STACK)
        if artifact.covers(series) and artifact.variant == variant
    ]
    if len(found) != 1:
        raise SystemExit(
            f"{series}: expected one particle stack with variant {variant!r}, found "
            f"{[a.id for a in found] or 'none'}"
        )
    return found[0]


def reconstructed_tilt_range(
    registry: ArtifactRegistry, stack, manifest: TiltSeriesManifest
) -> tuple[tuple[float, float], str]:
    """The tilt range of the views the particles' tomogram was built from, and its source.

    The manifest lists every view collected, and alignment drops some: on the MDC
    EMPIAR-11538 series min_std_ratio cut the dark extremes, so TS_0306's tomogram
    holds -40..+58 of the manifest's -52..+68. The manifest's extremes would give
    Dynamo a wedge with data where the tomogram has none. Best are the solved
    angles reconstruction recorded (`tilt_angles_path`), which fine alignment has
    also shifted to make the specimen flat (+7 deg on ryr1 039); then the nominal
    angles of the views the tomogram kept (`included_z_values`); then the manifest.
    The export says which it used. The range is in the tomogram's own angles;
    the table gets it as Dynamo's ytilt, through `dynamo_ytilt`.
    """
    angle_by_z = {image.z_value: image.tilt_angle_deg for image in manifest.images}
    tomogram_id = stack.parameters.get("input_tomogram_id")
    if isinstance(tomogram_id, str):
        try:
            tomogram = registry.get(tomogram_id)
        except KeyError:
            tomogram = None
        solved = tomogram.parameters.get("tilt_angles_path") if tomogram is not None else None
        if isinstance(solved, str) and Path(solved).is_file():
            angles = [float(word) for word in Path(solved).read_text().split()]
            if angles:
                return (min(angles), max(angles)), "reconstruction's solved angles"
        included = tomogram.parameters.get("included_z_values") if tomogram is not None else None
        if isinstance(included, list):
            angles = [angle_by_z[z] for z in included if z in angle_by_z]
            if angles:
                return (float(min(angles)), float(max(angles))), "reconstruction"
    angles = manifest.tilt_angles_deg
    return (float(min(angles)), float(max(angles))), "manifest"


def decoy_kind(registry: ArtifactRegistry, stack) -> str:
    """What a decoy stack's boxes are, in words for the report, from its picks and tomogram."""

    def recorded(key: str):
        value = stack.parameters.get(key)
        try:
            return registry.get(value) if isinstance(value, str) else None
        except KeyError:
            return None

    tomogram, picks = recorded("input_tomogram_id"), recorded("picks_id")
    if tomogram is not None and tomogram.parameters.get("phase_randomised_from"):
        return "pytom picks in phase-randomised tomograms"
    if picks is not None and picks.parameters.get("picking_method") == "random_positions":
        return "boxes at random positions"
    return "decoy boxes"


def table_row(tag: int, tomogram: int, record, box: int, tilt_range, spacing: float) -> list:
    """One Dynamo table row: pose from the native one, wedge, tomogram, position."""
    angles, shift = native_to_dynamo(
        relion_euler_to_matrix(record.rot_deg, record.tilt_deg, record.psi_deg),
        np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px]),
        box,
    )
    row = [0.0] * TABLE_COLUMNS
    row[0], row[1], row[2] = tag, 1, 1  # tag, aligned, averaged
    row[3:6] = shift.tolist()  # dx dy dz
    row[6:9] = angles.tolist()  # tdrot tilt narot
    row[12] = 1  # ftype: single-axis wedge about y
    # ymintilt ymaxtilt: the tomogram's range negated, which is where Dynamo's
    # wedge has to put it for the volumes this pipeline stores (see dynamo_ytilt).
    row[13], row[14] = dynamo_ytilt(tilt_range)
    row[19] = tomogram
    row[21] = 1  # class
    row[23:26] = [record.x_px + 0.5, record.y_px + 0.5, record.z_px + 0.5]
    row[30] = tag  # original tag
    row[31] = 1
    row[33] = 1  # reference
    row[35] = spacing
    return row


def format_row(row: list) -> str:
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
    registry = ArtifactRegistry.load(args.project / "artifacts.json")
    (args.out / "particles").mkdir(parents=True, exist_ok=True)

    rows, particles, wedges, wedge_sources, decoy_kinds = [], [], {}, {}, set()
    box = spacing = None
    tag = 0
    # Every real particle first, so they keep the tags an export without decoys gives.
    passes = [(args.variant, False)]
    if args.decoy_variant is not None:
        passes.append((args.decoy_variant, True))
    jobs = [(variant, decoy, tomogram, series) for variant, decoy in passes
            for tomogram, series in enumerate(args.series, start=1)]
    for variant, decoy, tomogram, series in jobs:
        stack = particle_stack(registry, series, variant)
        manifest = TiltSeriesManifest.model_validate_json(
            (args.project / "manifests" / f"{series}.json").read_text()
        )
        tilt_range, source = reconstructed_tilt_range(registry, stack, manifest)
        if not tilt_range[0] < 0 < tilt_range[1]:
            raise SystemExit(f"{series}: tilt range {tilt_range} does not span zero")
        if decoy:
            # Decoys are cut from the same tomogram, or from a phase-randomised copy of
            # it that keeps its tilt angles, so they share its wedge.
            if list(tilt_range) != wedges[series]:
                raise SystemExit(f"{series}: the decoys' wedge {tilt_range} differs from the "
                                 f"real particles' {wedges[series]}; not the same tomogram?")
            decoy_kinds.add(decoy_kind(registry, stack))
        else:
            wedges[series] = list(tilt_range)
            wedge_sources[series] = source
        low, high = tilt_range
        kind = "decoys" if decoy else "particles"
        print(f"{series} {kind}: wedge {low:+.1f} to {high:+.1f} deg, from the {source}")
        star = Path(str(stack.parameters["metadata_path"]))
        records = read_particle_star(star)
        data = zarr.open(stack.path, mode="r")
        if data.shape[0] != len(records):
            raise SystemExit(f"{series}: {data.shape[0]} particles in the stack, "
                             f"{len(records)} rows in {star}")
        this_box, this_spacing = int(data.shape[1]), float(stack.pixel_spacing_angstrom)
        if box is None:
            box, spacing = this_box, this_spacing
        elif (this_box, this_spacing) != (box, spacing):
            raise SystemExit(f"{series}: box {this_box} at {this_spacing} A differs from "
                             f"{box} at {spacing} A of the first series")
        for record in records:
            tag += 1
            name = f"particles/particle_{tag:05d}.mrc"
            with mrcfile.new(args.out / name) as handle:
                handle.set_data(np.asarray(data[record.index], dtype=np.float32))
                handle.voxel_size = spacing
            rows.append(table_row(tag, tomogram, record, box, tilt_range, spacing))
            particles.append({
                "tag": tag,
                "tomogram_number": tomogram,
                "run": series,
                "source_stack_id": stack.id,
                "source_stack": str(stack.path),
                "source_metadata": str(star),
                "source_index_zero_based": record.index,
                "native_record": asdict(record),
                "file": name,
                "decoy": decoy,
            })
    if not rows:
        raise SystemExit("no particles to export")

    table = args.out / "initial.tbl"
    table.write_text("\n".join(format_row(row) for row in rows) + "\n")
    manifest_out = {
        "schema_version": 1,
        "source_project": str(args.project.resolve()),
        "variant": args.variant,
        "decoy_variant": args.decoy_variant,
        "decoy_kind": "; ".join(sorted(decoy_kinds)) if decoy_kinds else None,
        "selected_particles": len(rows),
        "decoy_particles": sum(1 for particle in particles if particle["decoy"]),
        "box_size_px": box,
        "pixel_spacing_angstrom": spacing,
        "pose_convention": {
            "native_center_zero_based": (box - 1) / 2,
            "dynamo_center_zero_based": box / 2,
            "native_map": "x_in = cN + A @ (x_out - cN) + s",
            "dynamo_table_matrix": "M = A.T = Rz(narot) @ Rx(tilt) @ Rz(tdrot)",
            "dynamo_table_shift": "d = s + (A - I) @ (cD - cN)",
            "conversion": "cryoet_pipeline.backends.dynamo_search.native_to_dynamo",
        },
        "table_columns": TABLE_COLUMNS,
        "wedge": {
            "ranges": wedges,
            "range_source": wedge_sources,
            # What the table's ymintilt/ymaxtilt hold: each range negated.
            "dynamo_ytilt": {series: list(dynamo_ytilt(tuple(r))) for series, r in wedges.items()},
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
        "particles": len(rows),
        "decoys": sum(1 for particle in particles if particle["decoy"]),
        "series": args.series,
        "initial.tbl": sha256(table),
        "export_manifest.json": sha256(args.out / "export_manifest.json"),
    }, indent=1) + "\n")
    decoys = sum(1 for particle in particles if particle["decoy"])
    print(f"exported {len(rows)} particles of {len(args.series)} series to {args.out}"
          + (f", {decoys} of them decoys ({'; '.join(sorted(decoy_kinds))})"
             if args.decoy_variant is not None else ""))


if __name__ == "__main__":
    main()
