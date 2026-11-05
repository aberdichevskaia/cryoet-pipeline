# Run records, MRA first

## Purpose

[database_design.md](database_design.md) decides that every run writes a small
record next to its outputs, that per-particle data are columnar files, and that a
database is only an index over those files. [pipeline_engine.md](pipeline_engine.md)
decides the write order. Neither says what the record holds field by field, and no
step writes one yet.

This document fixes that record for every step, and the MRA classification as the
first step to write it. The record is general from the start: MRA is its first
user, not its owner, so the other steps adopt the same record later without it
being written again. Draft of 2026-11-05, to be agreed before code.

## The record

One file, `run.json`, per run, written last (below). A pydantic model,
`cryoet_pipeline.runs.RunRecord`, writes and reads it.

```text
schema_version     1
id                 opaque, uuid4
step               "mra-setup", "mra-iteration", ...
step_version       declared integer, raised by hand when the step's results change
backend            the implementation, e.g. "dynamo-port"
backend_version    declared integer, the same rule
content_params     what defines the result; decided per step
execution_params   device, dtype, GPUs, workers, chunk: never in the fingerprint
inputs             slot, path relative to the run, fingerprint of what it read
selection          null for the whole input; a selection's name otherwise
lane               the variant label, when runs are meant to coexist
fingerprint        sha256 of the canonical JSON of step, step_version, backend,
                   backend_version, content_params, the inputs' fingerprints and
                   the selection
status             "succeeded" or "failed"
started_at         ISO time
ended_at           ISO time
host, slurm_job_id, log_path
code               package version and git commit: recorded, not fingerprinted
outputs            Artifact records (models.Artifact), on success only
selections         what the run defines as particle sets (below)
error              on failure, the message
```

**Versions are declared, not read from git.** The design's fingerprint holds
`backend_version`; taken from the git commit, every commit would make every result
stale, and a job submitted again after an unrelated commit would compute all over.
A declared version is raised when a change moves the results, as the averaging
change of cc1933b did. The commit is still recorded, and a run carried on under
another commit says so.

**Parameters are split per step.** For MRA, the content parameters are the setup's
variables and each iteration's search; the device, dtype, GPU count and workers
are execution parameters, as float32 on a GPU and float64 on the CPU were
accepted as the same search (scripts/port_acceptance.py).

## Write order

As [pipeline_engine.md](pipeline_engine.md) has it, with one change of order:
outputs go to `<node>.tmp-<run id>/`, `run.json` is written last into that folder,
and the folder is then renamed to `<node>/`. The engine's order, rename first and
record after, leaves a moment in which a killed job leaves a whole folder without
its record; renaming last, the folder never exists without it. A failed attempt
keeps its `run.json` (status `failed`, the error) in `attempts/<run id>/`, and the
next attempt starts a new temporary folder; one killed outright, with no record,
is moved there by the next attempt.

## Per-particle tables

Parquet, one file per particle set, as [database_design.md](database_design.md)
asks, with named, typed columns and the conventions in the file's metadata:
angles as Dynamo's (tdrot, tilt, narot, degrees, about the box centre N/2), shifts
in pixels of the particles' box, the pixel size and the box. STAR and Dynamo
tables are written only for export. Values are float64; nothing is rounded.

## MRA as runs

One MRA run is a plan of nodes, each a run with its own record: the setup, then one
node per iteration, as the design unrolls iterative refinement. Each node reads
what the one before it wrote, which is what lets a job that ran out of time carry
on, and what decides whether a node is done: a node whose fingerprint is already
there with `succeeded` is not computed again.

```text
<export>/<NAME>/
  plan.json                     the setup's variables and every iteration's search
  setup/run.json                step mra-setup; input: the export
  setup/particles.parquet       one row per particle: tag, tilt-series, tomogram,
                                position, wedge, pick score, decoy, starting pose,
                                and the channel whose seed averaged it (0 for none)
  setup/reference_<k>.mrc       each channel's seed
  setup/mask.mrc
  ite_0001/run.json             step mra-iteration; inputs: setup's outputs
  ite_0001/particles.parquet    one row per particle and channel: pose, shift, CC,
                                and whether the particle is in that channel's class
  ite_0001/average_<k>.mrc      each surviving class's average
  ite_0002/ ...
```

The export's fingerprint, the setup's input, is the hash of its `initial.tbl` and
`export_manifest.json`: what the export says it is. The particle files are not
hashed; reading gigabytes to answer "is this the same export" costs more than
the manifest, which names each particle's source, already answers.

**Classes are selections.** Each iteration's record lists its classes as
selections, the design's `definition` and `snapshot`:

```text
{"name": "class 2", "kind": "particles",
 "definition": {"column": "channel", "equals": 2, "where": "assigned"},
 "snapshot": "ite_0005/particles.parquet"}
```

so "carry on with class 2 of the last iteration" names a selection, not "column 34
of a kept channel's table", which is how the report reads Dynamo's projects now.

## Reading

One module reads a classification from either layout into the in-memory model the
report already has (`reports.classification.ClassificationRun`): a Dynamo project
(cards, settings, `refined_table_ref_*`), or a run of this layout (`plan.json`).
Dynamo's layout is read, never written: Dynamo's projects stay as controls, and the
port's runs made until now (`*_port`, in Dynamo's layout) stay readable through the
same loader. The report, the port's comparison with Dynamo and
`compare_dynamo_classes.py` move to that module first; the other scripts that read
port runs follow when next touched.

## Not in this step

The database index, the controller and `execute-run`, plans that span steps, the
other steps' records, and the export as a registered artifact (the MRA reads the
export folder as now; reading the registry's particle stacks is the step after).

## Decided on 2026-11-05

A run's folder is `<NAME>`, without the `_port` that runs in Dynamo's layout carry;
a node carried on under another commit than the one before it is noted in the
log, and each record keeps its own commit.

## Status

2026-11-05: the record (`cryoet_pipeline.runs`), Parquet particle tables
(`cryoet_pipeline.particle_tables`, pyarrow) and the MRA in this layout with its
carrying on (`cryoet_pipeline.mra_runs.run_plan`) exist and are tested, the first
iteration equal to the last bit to what the Dynamo layout's writer computes from the
same plan. Still in Dynamo's layout: what `scripts/mra_port_run.py` writes, and
what the report and the comparisons read. Next: the reader for both layouts, then
the job's switch to this one.
