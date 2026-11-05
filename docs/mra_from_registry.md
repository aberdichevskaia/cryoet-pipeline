# MRA from the registry

## Purpose

[run_records.md](run_records.md) left one thing for the step after it: an MRA run
reads its particles from an export folder of `scripts/export_dynamo.py`, a copy of
the extraction step's particle stacks in Dynamo's layout, not from the project's
registry. This document is how the MRA reads the registry instead. Agreed on
2026-11-05 (the five decisions at the end) and built the same day (Status).

## What the export does now

`scripts/export_dynamo.py --project P --variant L --series S1 --series S2 ...
[--decoy-variant D]` reads, per series, the one PARTICLE_STACK artifact of label L
in `P/artifacts.json`: a Zarr array (one chunk per particle) and its STAR file of
picks (index in the stack, pick position, pose, shift, score). It writes:

- every particle again as `particles/particle_NNNNN.mrc`, the voxels unchanged;
- `initial.tbl`: each pick's pose through `native_to_dynamo`, the wedge of the
  views its tomogram was built from (the reconstruction's solved angles, else the
  views it kept, else the manifest's) negated as Dynamo's ytilt, the series' number
  in the given order, the pick position plus 0.5, the pixel size; numbers with 12
  significant digits;
- `export_manifest.json`: per particle its series, stack id, stack index, STAR row
  and decoy mark, and the decoys' kind.

Tags run 1..N in the order: real particles series by series in the given order,
each stack in its index order, then the decoys in the same order. The MRA setup
reads only `initial.tbl`, the particle files and the manifest; its random draws
follow the table's order.

## The design

**One conversion in the package.** The export's reading and conversion move from the
script into `cryoet_pipeline` as a particle source: given the project, the series in
order, the label and the decoy label, it gives one row per particle (tag, series,
stack id, stack index, series number, pick position, pose and shift in Dynamo's
convention, wedge, score, decoy) and each particle's voxels, read from its stack on
demand. `export_dynamo.py` keeps its command line and writes its files from that
source, so the export and the MRA cannot convert differently; a test checks the
export's table against the source's rows to the table's 12 digits.

**The MRA takes the source, not a folder.** `MraPlan.particles`, a folder of
particle files now, becomes the source. The setup makes the plan from it as it does
from an export, the same seeds from the same draws in the same order, so a run from
the registry and a run from an export of the same stacks differ only by the export
table's 12 digits. An export folder stays a source too, read as now: Dynamo's
controls, the consensus halves and the runs made so far keep working unchanged.

**The setup's inputs are the stacks.** Each particle stack is an input of the setup
node, known by what it says it is: its registry record (shape, parameters, padded
indices) and its STAR file, not its voxels, as an export is known by its table and
manifest. The series, the labels and their order are content parameters. A stack
extracted again from other picks or with other parameters makes the setup another
node, and the run refuses to carry on into it. A stack extracted again with
identical picks and parameters is not seen; extraction writing its own run records
(with declared versions) is what closes that.

**No new artifact for the particle set.** The setup node already writes the run's
particle table (`setup/particles.parquet`): membership, starting poses, wedges,
scores, decoys. That is the snapshot the database design's `selection` asks for, so
no export-like particle set is registered in between. A later run that starts from
part of an earlier one (a class, a score cut, a consensus half) names a selection of
that run's table.

**Where a run lives.** `<project>/mra/<NAME>/`, beside the project's other outputs,
locked by `<project>/mra/<NAME>.lock`, its report `<project>/mra/<NAME>_classification_report.pdf`.
Runs from an export stay in the export folder. Runs are not registered in
`artifacts.json` in this step: their records are the truth, and the index over them
is the database's step.

**One position convention.** The setup's table holds the pick position in the
tomogram's voxels, zero-based, as extraction's STAR has it. Today it copies the
export table's columns 24-26, which add 0.5; that changes now, before any run of the
new layout exists on the cluster, with the setup's declared version raised.

**What the report reads.** Series, scores and decoys from the setup's table, as now;
the decoys' kind and the project for the where-picked pages from the table's
metadata instead of an export manifest. Tags mean the same particle in two runs when
their tables give each tag the same stack and index: `compare_dynamo_classes.py`
checks that from the tables, and from the manifests for Dynamo projects.

**The job.** `scripts/mra_port_run.sbatch` gains a third way in beside EXPORT and
PROJECT: `SOURCE=<pipeline project> LANE=<label> SERIES="TS_1 TS_2 ..."`
(`DECOY_LANE` for decoys), with the setup's variables as now.

## Not in this step

Score cuts and classes of earlier runs as inputs (selections), the consensus halves
from a run instead of from sub-exports, registering runs, extraction's own run
records, and a STAR or RELION export of a run.

## Decided on 2026-11-05

1. Runs from the registry live in `<project>/mra/<NAME>/`.
2. No separate particle-set artifact: the setup's table is the snapshot.
3. A stack is known by its registry record and STAR, not by its voxels.
4. The position columns become extraction's zero-based pick position.
5. Tags in the export's order, so a registry run and an export run compare tag by tag.

## Status

2026-11-05: `cryoet_pipeline.particle_sources.registry_particles` is the conversion,
and `scripts/export_dynamo.py` writes its files from it (its tests unchanged).
`mra_setup.plan_from_registry` makes the plan from it, `MraPlan.particles` is a
folder or the stacks, and the setup node (declared version 2) names each stack as
an input (`particle_stack <id>`), its table each particle's stack, index and
zero-based pick; the old writer refuses a plan from the registry. The report reads
such a run's particles from their stacks, its where-picked pages and each tag's
identity from the setup's table; `compare_dynamo_classes.py` checks identity from
the runs. `mra_port_run.py --source/--lane/--series/--decoy-lane` and
`mra_port_run.sbatch` with `SOURCE/LANE/SERIES/DECOY_LANE` run it.
tests/test_mra_from_registry.py runs one plan from a synthetic registry and from an
export of it: the export's table is the source's rows to its 12 digits and its
voxels the stacks'; the two runs keep the same classes and members, with seeds
within 1e-6 and poses within 1e-6 deg of each other (the tests' bounds).
`scripts/compare_export_registry.py <export>` checks an export against its project's
registry (its table to the 12 digits it wrote, each tag's stack and index, every
voxel) and prints the SOURCE, LANE and SERIES that read the same particles.

On max, 2026-11-05: one plan (subset subset noise, coarse_to_fine 53.76 to 29.4 A,
diameter 340 A, no global rounds) run from the registry of outputs/ryr1 (label
bin8-2x, series 039 096 097 125 161, 2,384 particles; job 3891692,
outputs/ryr1/mra/regcheck_registry) and from the export bin8-2x-pad made of the same
stacks (job 3891678, regcheck_export), on the same four A40s of maxg21. Every one of
the 5 iterations put all 2,384 particles in the same class, with CC differences
exactly 0 and poses equal as far as the comparison prints them (0.001 deg, 0.001 px);
each tag's series, stack and index agreed in both runs. The export's 12-digit table
moved nothing the float32 search sees. The run from the registry took 21 minutes,
the one from the export 24.5, the difference before the first iteration (the GPUs
idle about 2 minutes against 5, from their samples; not measured further).
