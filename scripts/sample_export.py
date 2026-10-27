#!/usr/bin/env python3
"""A small seeded sample of an export, as an export of its own.

    python scripts/sample_export.py outputs/ryr1/dynamo/ryr1-consensus-half1 --particles 64

writes outputs/ryr1/dynamo/ryr1-consensus-half1-sample64: particles/ (links to the
source's files, tags unchanged), the sampled lines of initial.tbl in the source's
order, export_manifest.json cut to them with the sample recorded under "sample", and
export_complete.json, as scripts/split_export_by_view.py writes its groups. Decoys
are never sampled. --out names another folder, which may not hold anything yet;
--seed (default 20261027) picks another sample, and the same seed the same one.

What it is for: a Dynamo CPU run that would take days on a whole export, such as a
global round at the full box (dynamo_mra_setup.m's global_full_box), made on few
enough particles to finish in an hour, so the port's search can be checked against
it (scripts/torch_search_check.py). Tags stay the source's, so a sampled particle
is the same particle in every comparison.

Disposable, like the rest of scripts/.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from split_export_by_view import write_subexport

from cryoet_pipeline.reports.dynamo_project import read_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("export", type=Path, help="an export folder of scripts/export_dynamo.py")
    parser.add_argument("--particles", type=int, required=True, help="how many to sample")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", type=Path, default=None, help="default <export>-sample<N>")
    args = parser.parse_args()

    source = args.export.resolve()
    out = (args.out or source.with_name(f"{source.name}-sample{args.particles}")).resolve()
    if not (source / "export_complete.json").is_file():
        raise SystemExit(f"{source}: no export_complete.json, so no finished export to sample")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; sample into a new folder")

    manifest = json.loads((source / "export_manifest.json").read_text())
    entries = {int(entry["tag"]): entry for entry in manifest["particles"]}
    lines = [line for line in (source / "initial.tbl").read_text().splitlines() if line.strip()]
    table = read_table(source / "initial.tbl")
    if len(lines) != len(table) or {int(row[0]) for row in table} != set(entries):
        raise SystemExit(f"{source}: initial.tbl and export_manifest.json list different particles")
    candidates = [int(row[0]) for row in table if not entries[int(row[0])].get("decoy")]
    if not 0 < args.particles <= len(candidates):
        raise SystemExit(f"--particles must be from 1 to {len(candidates):,}, the export's "
                         "particles that are not decoys")

    rng = np.random.default_rng(args.seed)
    picked = {int(tag) for tag in rng.choice(candidates, size=args.particles, replace=False)}
    chosen = [(int(row[0]), line) for row, line in zip(table, lines, strict=True)
              if int(row[0]) in picked]
    write_subexport(source, out, chosen, manifest, {
        "sample": {"from": str(source), "particles": len(chosen), "of": len(candidates),
                   "seed": args.seed},
    })
    by_series = Counter(str(entries[tag]["run"]) for tag, _ in chosen)
    print(f"{len(chosen):,} of {len(candidates):,} particles in {out}; by series: "
          + ", ".join(f"{series} {count}" for series, count in sorted(by_series.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
