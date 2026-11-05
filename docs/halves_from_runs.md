# Consensus halves from runs

## Purpose

An MRA run reads its particles from the project's registry since 2026-11-05
([mra_from_registry.md](mra_from_registry.md)), but `scripts/consensus_halves.py`
still needs an export: `split` takes two runs inside one export folder and writes
each half as an export of its own. At bin4 that export would be a copy about eight
times the 1.3 GB of bin8. This document is how the halves are made from two runs
of the registry, with no export. Agreed on 2026-11-05 and built the same day (Status).

## What `split` and `maps` do now

`split RUN:CHANNEL --agree RUN2:CHANNEL2 --start-lowpass A --out OUT` takes the
members of one class after the last finished iteration of its run that are also
members of the other run's class, decoys left out. It deals them into two halves at
random, series by series (`--seed`), so both halves hold every series in the same
proportion. Each half's starting pose is the first run's refined pose of that
particle in that channel. Both halves get the same start, that run's class average
of that iteration low-passed to A. Each half becomes an export folder,
`OUT-half1` and `OUT-half2`: links to the parent export's particle files, the
parent's table rows with those poses, the manifest with a `consensus_half` record.
The half runs are then set up from those folders with `SEEDS=map` and the start as
`REFERENCE`.

`maps` reads the two half runs, checks through their manifests that they are the
two halves of one split, and writes the half maps, their FSC and how far each half
moved its particles.

## The design

**The split is a run.** `split` on two runs of the registry writes one folder,
`<project>/mra/<OUT>/`, in the run-record layout ([run_records.md](run_records.md)):

```text
<project>/mra/<OUT>/
  run.json             step consensus-split; inputs: the two runs' iteration nodes
                       it read, by their records' fingerprints; content: the
                       channels, the seed, the start's low-pass, and the label,
                       series and decoys' label the runs read (the project's path
                       stays out of the fingerprint; the runs' plans name it)
  halves.parquet       one row per particle kept: tag, half (1 or 2), its starting
                       pose (the first run's refined pose)
  start_reference.mrc  the start both halves refine from
```

Its record lists two selections, `half 1` and `half 2`, the rows of `halves.parquet`
with that half, as an MRA iteration lists its classes. How the particles are chosen,
dealt and posed, and how the start is made, is what `split` does now.

**A half run reads the registry and the split.** `mra_setup.plan_from_split(split,
half)` reads the particles the first run read (its project, label, series and decoys'
label, from its `plan.json`) through the same `registry_particles`, keeps the half's
tags, and gives each the split's starting pose; the wedge, the series' number and the
pick stay the registry's. The seed is the split's `start_reference.mrc` in one
channel (`SEEDS=map`). The setup node's inputs are the stacks, as for any run from
the registry, and the split's record; its `selection` names the half. The tags are
the first run's, so the identity checks keep working across the three runs.

**The job.** `scripts/mra_port_run.sbatch` takes `SPLIT=<project>/mra/<OUT> HALF=1
NAME=...` with the schedule's variables as now; the project, the label, the series,
the seeds and the reference come from the split, so they cannot disagree with it.
The run is `<project>/mra/<NAME>`.

**`maps` checks the halves by their split.** Two half runs of the run-record layout
must name the same split (by its record's fingerprint) and halves 1 and 2; the seed
and the start's low-pass come from the split's record. The rest is as now: the same
finished iteration, the same mask, the maps, the FSC, the moves.

**Both runs read the same particles.** `split` refuses two runs whose particles
differ: other project, label, series or decoys, or a tag naming another stack or
index. A run from an export and a run from the registry are not split together.

**Exports stay as they are.** Two runs in one export folder are split into
sub-exports as now, and `maps` reads those halves through their manifests.

## Not in this step

Selections as the input of an MRA run other than halves (a class of an earlier run,
a score cut), the halves at another binning than their runs', and registering the
split in `artifacts.json`.

## Decided on 2026-11-05

1. The split is a run of its own, `<project>/mra/<OUT>/`, its halves two selections.
2. A half run is set up from the split alone: `SPLIT=... HALF=1`, the rest from it.
3. `split` refuses two runs that read different particles, an export run with a
   registry run included.
4. `maps` knows the two halves by the split they name, not by a manifest.
5. Splitting into sub-exports stays for runs in an export folder.

## Status

2026-11-05: `consensus_halves.py split` writes the split of two runs of a registry as
a run of step `consensus-split` (`SPLIT_VERSION` 1) in `<project>/mra/<OUT>` and
refuses runs that read other particles, a run of a half, and an `--out` outside
`<project>/mra`; the export route keeps its sub-exports and shares the dealing and the
start with it. `RegistryParticles.selected` keeps part of a set at given poses, and
the setup's record names the selection (`selection`, and the split as its input
`selection`). `mra_setup.plan_from_split` makes a half's plan, and refuses seeds or a
reference of its own and a split whose first run is no longer as the split read it.
`mra_port_run.py --split/--half` and `mra_port_run.sbatch` with `SPLIT/HALF` refine a
half beside its split; `maps` reads the halves through their split's record.
tests/test_halves_from_runs.py runs a registry run, splits it, refines both halves and
reads their maps. Not run on max yet.
