#!/usr/bin/env python3
"""Does an export still hold what its project's registry gives? It writes nothing.

    python scripts/compare_export_registry.py outputs/ryr1/dynamo/bin8-2x-pad

reads the export's manifest (its project, its stacks' label, the series in the order
it numbered them, the decoys' label), reads the same particles from the project's
registry through cryoet_pipeline.particle_sources, the conversion
scripts/export_dynamo.py writes from (docs/mra_from_registry.md), and compares:

- initial.tbl with the registry's rows written as the export writes them, 12
  significant digits, line by line;
- each tag's series, stack and index in it with the manifest's;
- every particle file with its slice of its stack, voxel by voxel.

When all three agree, an MRA run from the registry reads the particles a run from
this export reads, and the script prints the variables that make
scripts/mra_port_run.sbatch read them (SOURCE, LANE, SERIES, DECOY_LANE). When the
voxels or the identities differ, the stacks were extracted again after the export;
when only the table differs, the conversion or what the tomogram records (its solved
angles, the views it kept) changed since. Exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np

from cryoet_pipeline.particle_sources import registry_particles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path, help="an export folder of scripts/export_dynamo.py")
    parser.add_argument("--project", type=Path, default=None,
                        help="the pipeline project; default the one the manifest names")
    args = parser.parse_args()

    export = args.export.resolve()
    manifest = json.loads((export / "export_manifest.json").read_text())
    project = args.project or Path(manifest["source_project"])
    entries = manifest["particles"]
    numbers = {str(entry["run"]): int(entry["tomogram_number"]) for entry in entries}
    series = sorted(numbers, key=numbers.__getitem__)
    label, decoys = manifest.get("variant"), manifest.get("decoy_variant")
    print(f"{export}: {len(entries):,} particles of {project}, label {label}, series "
          f"{' '.join(series)}" + (f", decoys {decoys}" if decoys else ""))
    try:
        source = registry_particles(project, label, series, decoys, log=lambda _: None)
    except (OSError, ValueError, KeyError) as error:
        print(f"the registry cannot give them: {error}")
        return 1

    differences = []
    written = (export / "initial.tbl").read_text().splitlines()
    rows = [" ".join(f"{value:.12g}" for value in row) for row in source.table]
    if len(written) != len(rows):
        differences.append(f"table: initial.tbl has {len(written):,} rows, the registry gives "
                           f"{len(rows):,}")
    else:
        unequal = [tag for tag, (ours, theirs) in enumerate(zip(rows, written, strict=True), 1)
                   if ours != theirs]
        if unequal:
            differences.append(f"table: {len(unequal):,} rows differ, the first tag {unequal[0]}")
    print(f"table       {len(rows):,} rows from the registry"
          + ("" if differences else ", every one as initial.tbl writes it"))

    known = {particle.tag: (particle.series, particle.stack_id, particle.index)
             for particle in source.particles}
    named = {int(entry["tag"]): (str(entry["run"]), str(entry["source_stack_id"]),
                                 int(entry["source_index_zero_based"])) for entry in entries}
    other = sorted(tag for tag in set(known) | set(named) if known.get(tag) != named.get(tag))
    if other:
        differences.append(f"identity: {len(other):,} tags name another stack or index, the "
                           f"first tag {other[0]}: {named.get(other[0])} in the manifest, "
                           f"{known.get(other[0])} in the registry")
    print(f"identity    {len(named):,} tags" + ("" if other else ", each the manifest's series, "
                                                "stack and index"))

    unequal_voxels, largest = [], 0.0
    for entry in entries:
        tag = int(entry["tag"])
        if tag not in known:
            continue
        theirs = mrcfile.read(export / entry["file"])
        ours = source.volume(tag)
        if theirs.shape != ours.shape or not np.array_equal(theirs, ours):
            unequal_voxels.append(tag)
            if theirs.shape == ours.shape:
                largest = max(largest, float(np.abs(theirs.astype(float) - ours).max()))
    if unequal_voxels:
        differences.append(f"voxels: {len(unequal_voxels):,} particle file(s) differ from "
                           f"their stack slice, the first tag {unequal_voxels[0]}, by up to "
                           f"{largest:.3g}")
    print(f"voxels      {len(entries):,} particle files"
          + ("" if unequal_voxels else ", each equal to its stack slice"))

    if differences:
        print("NOT the registry's particles:")
        for line in differences:
            print(f"  {line}")
        return 1
    print("the export holds what the registry gives; a run from the registry reads the same "
          "particles with")
    words = [f"SOURCE={project}"]
    if label is not None:
        words.append(f"LANE={label}")
    words.append(f'SERIES="{" ".join(series)}"')
    if decoys:
        words.append(f"DECOY_LANE={decoys}")
    print("  " + " ".join(words))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
