# Data model and database

## Purpose

This document decides how project state is stored once the pipeline stops being
a single linear run over one tilt-series. It covers the shape of the provenance
graph, what belongs in a database and what stays a file, how repeated and
branched runs coexist, and which engine to use.

It supersedes nothing; it fills the gap left open by
[architecture_decisions.md](architecture_decisions.md), which fixes *what* the
internal truth is, and by [roadmap.md](roadmap.md) section 5, which describes the
symptoms of not having this model.

## Core rule

```text
The provenance graph is relational.
Per-particle and per-pixel tables are columnar files, queried in place.
The database is a rebuildable index over per-run records on disk,
never the only copy of project state.
```

## Requirements this has to satisfy

Collected from the August 2026 trip notes (biologist requests, killing features)
and from follow-up answers by the intended users.

- **Steps are skipped.** CTF estimation and correction may be absent entirely.
- **Binning happens at arbitrary points with arbitrary values**, more than once.
- **A step is repeated with different backends or parameters**, both lanes are
  carried to the end, and the winner is chosen later — sometimes at the very end.
- **Variants never overwrite each other**, remember where they came from, and are
  not recomputed if already done.
- **Re-entry from an earlier step** happens both to compare and to discard: after
  reaching step N, going back to step N-3, redoing it differently, and either
  comparing both or abandoning the first attempt.
- **Entry and exit at any step.** Data arrives as frames or as tilt-series;
  results must be exportable at any stage.
- **The decisive metric is far downstream.** "Which aligner was better" is
  usually answered by the final resolution after subtomogram averaging, not by
  a metric produced at the alignment step.
- **Selection by predicate across a lineage hop**: "carry to the end only
  particles with property X", and "only particles whose *tomogram* has
  property Y".
- **One tomogram contains several species.** Ribosomes and something else, with
  different templates, independent downstream branches, and the option of
  returning to the same tomograms later with a third template.
- **Manual curation costs hours** and must survive reruns.
- **Several people share a project; the database is lab-wide.**
- **Reproduction months later** is a real request, typically "rerun this one step
  with a different backend".
- **Failure recovery must be local.** A failure in the last step must not
  invalidate the preceding ones.

Scale, from the users' own numbers:

```text
tilt-series per project      hundreds to a few thousand
steps per tilt-series        ~20
runs per project             ~10^4 to 10^5
artifact records             ~10^5 to 10^6
particles per project        ~10^7
```

Online (during-acquisition) processing is explicitly *not* required: next-morning
results are acceptable. That removes the main argument for a long-lived watcher
service and keeps the design simpler.

## The shape of the pipeline is a DAG

Branching and merging do not make a graph non-DAG; only cycles do. Every case
above stays acyclic:

- two backends for one step, merged later — a node with two parents;
- returning to an earlier step — a new node whose parent is an older node, as in
  branching from an old commit;
- iterative refinement — unrolled, one node per iteration.

This matters because it means the standard machinery applies unchanged:
topological ordering, transitive closure, invalidation by reachability,
memoization. The plan a user draws may look cyclic; the recorded history never
is. **Store the history.**

What a bare artifact DAG does *not* capture is human intent. Three relations must
be recorded explicitly alongside it:

- `supersedes` — this run replaces that one, without deleting it;
- `lane` / `variant` — these nodes belong to one comparison;
- `comparison` — these results are being weighed against each other.

These are annotations on the DAG, not a different structure.

## Decision: relational graph, columnar payload

Three layers, each doing what it is good at:

```text
provenance graph, runs, QC summaries   -> relational database
per-particle and per-tilt-image tables -> Parquet files, registered as artifacts
arrays (stacks, tomograms, averages)   -> Zarr, as today
analytic queries across all of them    -> DuckDB
```

DuckDB reads Parquet natively and can `ATTACH` a SQLite or Postgres database, so
a query that joins tomogram-level metadata against per-particle rows is a single
SQL statement, not application-side stitching. This is what makes "row per
particle stays a file" cost nothing in query power.

The single most important query the design must serve is the deferred comparison:

```text
for every final average, walk the provenance upward, find the fine-alignment run
that produced its lineage, and report that run's backend next to the average's
final resolution
```

That is a recursive ancestor traversal joined against a metric. It is a few lines
of SQL with a recursive CTE, and it is not expressible over filenames or over
labels embedded in artifact ids. It, rather than write concurrency, is the
strongest reason to have a real database at all.

### Why not a document store

The attraction is per-step schema freedom: every step has different parameters
and different QC fields. But a JSON column with an index on extracted paths gives
the same freedom, and the schema is enforced by pydantic in code regardless — so
what is needed from the database is tolerance, not flexibility.

The cost is decisive: "particles whose tomogram has property Y" requires either
denormalizing tomogram properties into every particle document, where they go
stale the moment tomogram QC is recomputed, or joining in application code. Both
are worse than a join.

### Why not a graph database

Lineage traversals are prettier in Cypher. But the traversals needed here are
shallow (~20 levels), of two fixed shapes (ancestors, descendants), and always
end in a metric aggregation, which is not a graph database's strength. Adding a
second service and a second query language is not justified, especially against
the two-year support obligation.

If a graph-native store were adopted anyway, the honest candidate is an embedded
one such as KùzuDB — Cypher, columnar, reads Parquet, no server. It is rejected
here mainly because it has no multi-writer story, and the project is lab-shared.

### What is genuinely not relational

Per-particle rows. At ~10^7 particles per project, before classification and
before multiple picking variants, these must not be database rows. They are
columnar files:

- one particle set = one Parquet file = one artifact;
- columns include `tomogram_id` and `tilt_series_id`, so the lineage back to the
  source volume is a foreign key at row level;
- STAR remains the interchange format; Parquet is the internal one.

The same applies to per-tilt-image data if it ever grows: for now
`tilt_image` rows are worth keeping relational, because excluded views, dose, and
per-tilt defocus are queried directly.

The other genuinely non-relational idea worth adopting is content addressing for
payload paths — see [Identity](#identity-what-makes-two-results-the-same). Nix is
the reference implementation of exactly this combination: a content-addressed
store with a relational (SQLite) metadata index.

## Binning is not a step

Two questions had to be answered before this could be modelled.

**Does the position of the binning in the chain matter, or only its value?**
The position matters, decisively:

- an alignment computed at bin8 has bin8 accuracy even when applied at bin4;
- binning before CTF estimation moves the Nyquist limit and therefore the
  achievable fit resolution — a fixed default fit limit becomes physically
  meaningless, as already noted in [roadmap.md](roadmap.md) section 1;
- phase flipping after binning cannot restore contrast in frequencies the
  binning already removed;
- every real-space resampling with interpolation blurs, so
  bin-then-align differs from align-then-bin.

**Is bin2 twice equivalent to bin4 once?** It depends on the operator:

- box averaging (integer binning as in IMOD `newstack -bin`): exactly
  equivalent, edges included, because `floor(floor(N/2)/2) == floor(N/4)`; the
  only discrepancy is double rounding when intermediates are integer;
- Fourier cropping: equivalent, the same central region of the transform;
- antialiased/filtered resampling: **not** equivalent — two filter applications
  are not one wider filter.

Because the identity holds only for some operators, no rewriting rule of the form
`bin2 ∘ bin2 -> bin4` should be implemented. A wrong rule of this kind produces a
silently incorrect cache hit.

The design consequence is to split binning into two different things:

- **Binning as representation.** "Give me this tomogram at bin8." Not a new
  entity — a *resolution level* of the same one. `Artifact.binning` and
  `Artifact.pixel_spacing_angstrom` already exist for this. Each step declares
  the input binning it requires, and the framework materializes that level if it
  is absent. In the vocabulary of the surrounding ecosystem this is OME-NGFF
  multiscales; in database vocabulary it is a materialized view; in compiler
  vocabulary the automatic insertion is coercion insertion.
- **Binning as a parameter of a measurement.** "At which binning was this
  alignment or CTF fit computed." A content-defining run parameter, explicit,
  part of the fingerprint, never inserted automatically.
  `TiltAlignment.input_binning` is already this.

```text
binning as representation -> artifact property, coerced automatically
binning as measurement    -> run parameter, explicit and content-defining
```

## Identity: what makes two results the same

Every run carries a `fingerprint`:

```text
fingerprint = hash(step, step_version, backend, backend_version,
                   content_params, input_fingerprints, selection_snapshot)
```

Rules:

- **Parameters are split per step into `content_params` and `execution_params`.**
  Binning, defocus tolerance, angular search, thresholds are content. Device,
  thread count, GPU ids, executable paths are execution. There is no default
  split; it is decided per step and reviewed when a step changes. Getting this
  wrong makes the cache either useless (everything re-runs) or wrong (a stale
  result is reused).
- **The fingerprint expresses declared intent, not output bytes.** Program-internal
  randomness and GPU non-associativity are deliberately ignored, per the users'
  own decision. An output content hash may be recorded separately for auditing,
  but it does not participate in identity.
- **Identity is separate from naming and from paths.** The current scheme, where
  a single human-readable id string encodes everything and `--variant` is
  appended to it, is what makes ids un-migratable today. An artifact should carry
  an opaque key, a fingerprint, a human display name, and a path, as four
  different fields.

Fingerprints give the caching and re-entry behaviour for free: a step whose
fingerprint already exists is not re-run, and a partially failed pipeline resumes
at the first missing fingerprint.

## Runs: history is append-only

A `run` row exists for every attempt, including failures. Artifacts are created
only on success; partial outputs are marked `temporary`. This restores something
TomoBear as-is had and the current registry lost: a failed step currently leaves
no trace at all, so "not attempted" and "attempted and failed" are
indistinguishable.

Nothing is ever updated destructively:

- redoing a step creates a new run and marks the previous artifact
  `superseded_by`; the old result stays visible and comparable;
- "undo" at the user level is not a rollback, it is a new event;
- deletion is a separate garbage-collection concern, not an edit.

This is the append-only/event-sourced shape, and it is what makes "go back three
steps, try again, keep both" cheap instead of destructive.

Lineage becomes run-mediated:

```text
artifact -> run (produced by) -> run_input -> artifact (consumed)
```

This is the standard bipartite provenance shape (W3C PROV entity/activity), and
it is what `parent_ids` is a lossy projection of: today one can see that a
tomogram came from a stack and an alignment, but not *by which step with which
parameters*. `parent_ids` should be retained as a derived view for compatibility.

## Selection as a first-class entity

A run does not operate on one tilt-series; it operates on a **subset of the
project**, which in the ordinary case happens to be everything.

```text
selection(id, project_id, name, kind,        -- tilt_series | tomograms | particles
          definition_json,                    -- the predicate, if any
          snapshot_artifact_id,               -- the frozen membership
          created_by_run_id)
```

This one entity resolves four separate requirements:

- **Run granularity.** The normal run covers the whole dataset; an experiment on
  two tilt-series is the same thing with a different selection. No special case.
- **Particle selection criteria.** Two averaging runs that used different score
  thresholds or different classification cut-offs differ by their input
  selection, which is visible, named, and comparable — rather than being buried
  in parameters or in filenames.
- **Cross-lineage predicates.** "Particles whose tomogram has property Y" is a
  selection defined by a predicate over a join.
- **Membership overlap.** A particle belonging to several selections is a
  many-to-many relation.

A selection must record **both** its predicate and a snapshot of its membership.
A predicate-defined selection is not reproducible over time, because upstream QC
changes; a snapshot without a predicate cannot be re-evaluated on new data. Both
are needed, and the distinction must be explicit in the schema rather than
implied.

## Object type

Several species in one tomogram is not a special case; it is ordinary branching
plus one missing dimension.

```text
object_type(id, project_id, name, taxonomy_ref)
```

Every `template`, `picks`, and `particle_set` carries `object_type_id`. Then:

- ribosomes and a second species are two independent branches from the same
  tomogram, each with its own template;
- returning later with a third template is another branch from the same
  tomograms, with nothing recomputed;
- "how many ribosomes per tomogram in this dataset" is an ordinary query.

The vocabulary should follow copick's pickable objects and the CZ cryoET Data
Portal's annotation object names, so that export stays cheap. `object_type` must
**not** be encoded into id strings — that is the same mistake `--variant` makes
today.

## Transactions, claims, and locking

Four distinct concerns, usually conflated.

**Atomicity.** One short transaction per finished run: its artifacts, QC records,
and terminal status commit together. Half-registered runs are the worst kind of
corruption because they look valid.

**Write concurrency.** There is no parallel processing of the *same* data;
parallelism is across tilt-series and across selections. But hundreds of runs
finish at once and all append to the index. That is contention on the index, not
on the data: roughly 10^4–10^5 short write transactions over a project's life,
arriving in bursts. Trivial for a server database.

**Mutual exclusion.** The lock that actually matters is on output paths, not on
rows: two runs must not write the same directory. This is obtained from the
schema rather than from a lock manager:

```text
UNIQUE (step, fingerprint, selection_id)
```

A run inserts itself with status `claimed` before starting. The winner computes;
a loser gets a constraint violation and reuses the existing result. One
constraint provides both mutual exclusion and memoization. Output paths are
derived from the fingerprint, so identical paths only ever mean identical work.

**Rollback.** Transaction rollback on a failed write is automatic and needs no
design. User-level "undo" is not rollback at all — it is a new run plus
`superseded_by`, as above. Keeping these two meanings apart is what prevents a
destructive-update design from creeping in.

## Engine: Postgres now, file-first always

Given a lab-wide database, several people per project, thousands of tilt-series,
and a browser UI as the stated direction:

- **Postgres** for the transactional store: real multi-user concurrency, JSONB
  with GIN indexes for parameters, recursive CTEs for lineage, one location per
  lab, and DuckDB can attach to it.
- **SQLite** for project *snapshots*: hand a project to a colleague, attach it to
  a paper, archive it. This is SQLite's strongest role and does not compete with
  the server.
- **DuckDB** as the query engine for analytic and cross-file queries.

An earlier draft of this analysis recommended SQLite as the primary store. That
was based on a single-user assumption; the users' answers changed it. SQLite also
has documented locking problems on network filesystems, which rules it out as a
primary store on Lustre or NFS regardless of user count.

The **file-first** principle is what makes this choice reversible and is more
important than the choice itself:

- each run writes a small `run.json` next to its outputs, recording step,
  parameters, versions, status, and the artifacts produced;
- the database is rebuildable from a scan of the project tree (`reindex`);
- writes go to the file first, then to the index.

Consequences: `rsync` of a project still works; a lost database is an
inconvenience, not a lost project; schema evolution is an upcaster plus a
reindex, not a migration of user data; and starting on SQLite and moving to
Postgres later loses nothing. Under the two-year support obligation — and with
the real possibility of no maintainer after that — a design that survives the
loss of its service layer is worth the duplication.

The exception is ephemeral state with no natural file home: queue position,
claims, "what is running right now". That may live in the database only and be
lost on reindex, because it is not history.

## Schema sketch

```text
project(id, name, root_path, schema_version, created_at)
dataset(id, project_id, name, notes)
tilt_series(id, dataset_id, name, source_mdoc, num_tilts, raw_apix, ...)
tilt_image(tilt_series_id, z, tilt_angle_deg, dose, defocus_hint, excluded)

object_type(id, project_id, name, taxonomy_ref)

selection(id, project_id, name, kind, definition_json,
          snapshot_artifact_id, created_by_run_id)

run(id, project_id, plan_node_id, step, step_version,
    backend, backend_version,
    content_params_json, execution_params_json, fingerprint,
    selection_id, lane,
    status, claimed_at, started_at, ended_at, exit_code,
    host, slurm_job_id, log_path,
    cloned_from_run_id, supersedes_run_id, actor)
run_input(run_id, slot_name, artifact_id)

artifact(id, run_id, slot_name, kind, display_name, path, fingerprint,
         object_type_id, storage_role, retention_policy, presence,
         size_bytes, shape_json, dtype, axis_order, apix, binning,
         superseded_by, meta_json)
artifact_scope(artifact_id, tilt_series_id)

qc(id, run_id, artifact_id, kind, status, report_path)
qc_metric(qc_id, name, value, unit)
metric_series(run_id, iteration, name, value)

plan(id, project_id, name)
plan_node(plan_id, node_id, step, content_params_json, lane)
plan_edge(plan_id, from_node, from_slot, to_node, to_slot)

shared_asset(content_hash, natural_key, path, kind)
```

Notes on the shape:

- **One `run` table, not one table per step.** Step-specific parameters live in a
  JSON column; step-specific metrics live in `qc_metric` in long format. A table
  per step would mean thirty tables and permanent migrations, and would be the
  point at which "relational is wrong for this" became true. The cost of the long
  format is awkward pivots, mitigated by materialized views for the few hot ones.
- **Step types are code, not rows.** A step declares its input slots
  (name to `ArtifactKind`, cardinality), its output slots, and its parameter
  schema in pydantic. That declaration is the typed socket that makes
  drag-and-drop and pre-built plans possible; it does not belong in the database.
- **`presence` is three-valued**, not a boolean `can_recompute`: present,
  absent-but-recomputable, gone. A node stays in the graph forever; only its
  payload comes and goes.
- **`plan_*` is what allows a pipeline to be assembled before its inputs
  exist.** A plan edge binds a future output slot to a future input slot; a run
  materializes a plan node. Without this, inputs can only be resolved against
  artifacts that already exist, and a pipeline cannot be built in advance.
- **Transitive closure is the one thing worth denormalizing.** A
  `provenance_closure(descendant_artifact_id, ancestor_run_id)` table maintained
  on insert makes the deferred-comparison query instant. At ~10^5 runs it is
  small.

## Interactive and manual steps

Manual curation costs hours and must not be lost. It therefore has to be a
first-class run, with a human as the backend:

- status `awaiting_human`, not a database lock — the "lock" is held for hours;
- the edit is recorded as an artifact with its own provenance, so a
  hand-cleaned particle set is a node in the graph rather than a file that
  appeared from nowhere;
- draft state is checkpointed periodically, so a crashed workstation loses
  minutes rather than the session;
- when an upstream artifact is recomputed, the manual edit is *not* silently
  discarded: it becomes an edit whose parent is superseded, and resolving that
  is an explicit decision.

If manual steps are not modelled, the graph lies and reruns destroy hand-work.

## Shared assets across projects

EMDB maps, trained models, and gain references do not belong to a project and
must not be duplicated per project.

```text
shared_asset(content_hash, natural_key, path, kind)
```

`natural_key` is the human handle (`emdb:EMD-1234@apix=2.10`) used to answer
"already downloaded?"; `content_hash` is the real identity, used to answer "the
same thing?". Assets live once in a lab-shared path and are referenced by
artifacts. **Shared assets are immutable** — editing one in place would silently
change every project referencing it.

## Deletion and garbage collection

Terabytes of recomputable intermediates cannot be curated by hand. The model is
reachability from roots, as in a tracing collector:

- some artifacts are **roots**: an export, a published figure, a final average,
  anything explicitly pinned;
- an artifact is collectable if it is not reachable from any root and its
  `retention_policy` allows recomputation;
- collection nulls the payload and sets `presence = absent`; the graph node and
  its fingerprint remain, so the artifact can be regenerated and so history stays
  readable.

This is the missing half of [storage_policy.md](storage_policy.md): the roles and
policies exist, the collector does not. Note that on a cluster with a purged
scratch filesystem, "recompute" is not a policy but a recurring fact of life, so
the collector's behaviour is exercised constantly rather than rarely.

## Consequences for the current code

What this design breaks, and what it does not:

- **Kept as-is.** [`Artifact`](../src/cryoet_pipeline/models.py) storage fields,
  the per-stage QC models, backend protocols, Zarr as array storage, and the
  principle that registry state is internal truth. The per-stage QC models are
  the most valuable existing asset: they are the metric vocabulary the deferred
  comparison needs, and no surveyed tool has an equivalent.
- **Replaced.** `Artifact.parent_ids` becomes a derived view over run-mediated
  lineage. `parameters["tilt_series_id"]` becomes `artifact_scope`, which
  became required the moment subtomogram averaging spanned several tilt-series;
  it now exists as `Artifact.tilt_series_ids`, and the parameter it replaces is
  still written alongside it. `--variant` encoded in id strings becomes a
  `lane` column plus fingerprints.
- **Corrected.** [`ArtifactRegistry`](../src/cryoet_pipeline/artifacts.py) does
  read-modify-write of an entire JSON document with no locking, so two runs
  finishing simultaneously lose one another's entries, and the rewrite is O(n)
  per step. Per-run JSON records plus an index fix both, and the per-run records
  are wanted anyway.
- **Rescaled.** One artifact per corrected projection means ~120k records and
  ~120k small files at 2000 tilt-series, which is unpleasant for the index and
  worse for a parallel filesystem. A corrected stack should be one artifact with
  a T axis, with per-tilt information in `tilt_image`. This also settles run
  granularity: one run per tilt-series, which maps directly onto Slurm job
  arrays.
- **Filled in.** `run`, `export`, and `qc` in the CLI are currently stubs; they
  are the natural home for the plan runner, the converters, and the QC roll-up.

Order of work, so that storage is not migrated under a schema that is about to
change twice:

1. introduce `run` and declared input/output slots, in pydantic, still on JSON;
2. split identity from naming: add fingerprints alongside the semantic ids;
3. make scope a relation (`artifact_scope`), before multi-tilt-series averaging
   — **half done**, see below;
4. move to a database as a rebuildable index, with `reindex`;
5. add `plan_*` and a thin Slurm runner.

Step 3 arrived early, because cross-tilt-series averaging landed first and an
artifact about two tomograms cannot state that in a field called
`tilt_series_id`. `Artifact.tilt_series_ids` is now a list field with
`covers(tilt_series_id)` as the way to ask, and every resolver and guard in the
CLI and the backends goes through it — twenty-nine call sites that used to read
`parameters["tilt_series_id"]` directly.

What is done is the file-first shape: a list on the artifact, which is one row
per element of `artifact_scope` once there are rows. What is not done is the
removal — forty-two producers still write the scope into `parameters` as well,
and the field is adopted from it, exactly as `backend` and `variant` were
promoted without rewriting their producers. Construction and deserialization
refuse a field that disagrees with the parameters, so the duplication cannot
drift silently; it is duplication, not ambiguity.

The removal belongs with step 4 rather than before it, for a reason worth
writing down: `parameters` becomes a JSON column beside real columns, and that
is the point at which a duplicated scope stops being harmless redundancy and
starts being a second source of truth the index would have to choose between.
Removing it earlier would mean touching forty-two call sites twice.

## Format conversion

Adjacent decision, since it affects what the database has to describe.

A small set of canonical converters should be hand-written and covered by
fixtures: MRC/Zarr, RELION STAR in both the 3.1 and RELION-5 conventions, IMOD
`.tlt`/`.xf`/`.defocus`, and possibly Dynamo tables. Coordinate conventions are
where silent corruption lives — origin versus centre, handedness, pixels versus
angstrom, axis order — and a plausible-looking wrong converter is not detected
until a structure fails to refine. The two `.star` readers ArtiaX registers for
the same suffix, already documented in [roadmap.md](roadmap.md) section 3, are
the canonical example.

The long tail — exotic formats, one-off exports, a CSV for a plot — can be
generated on demand by an agent, under two conditions: it composes the tested
primitives rather than writing raw I/O, and the generated script is recorded as a
run so that a wrong conversion is traceable. The rule of thumb: generated
converters are acceptable for data leaving the system for good, and not for data
that re-enters the pipeline or informs a decision.

## Open questions

1. **The `content_params` / `execution_params` split per step.** No default
   exists; each step needs a decision and a review point.
2. **Where the relational/columnar boundary sits for tilt-image data.** Rows for
   now; if per-tilt records grow, the same Parquet treatment applies.
3. **How a superseded upstream interacts with a manual edit downstream.** The
   policy is not obvious and should not be silent.
4. **Predicate selections and drift.** Storing both predicate and snapshot is
   decided; which one a rerun uses by default is not.
5. **Display names.** The mechanism is needed now — a name field separate from
   identity, with a decided uniqueness scope. The names themselves come later.
   Note that ids currently leak into paths, so this must be untangled before
   aliases exist.
6. **Who runs the Postgres instance**, and the fallback if nobody will.
7. **What an agent is allowed to write.** Proposed rule: only through the same
   run-creating API as humans, so that every agent action is a `run` row with an
   `actor`, and is auditable and revertible like any other.

## Prior art worth reading

Ranked by usefulness per hour.

- **"Build Systems à la Carte"** (Mokhov, Mitchell, Peyton Jones) — formalizes
  exactly this problem: what counts as the same result, what to rebuild, where
  the cache lives.
- **Git internals** (Pro Git ch. 10) — content-addressed objects plus a DAG plus
  refs. The described workflow of branching, returning, comparing and discarding
  is git's, and the model can be borrowed wholesale.
- **OME-NGFF multiscales** — binning as a resolution level rather than a step,
  already standardized in this ecosystem.
- **Nix** — a content-addressed store with a relational metadata index, plus the
  cleanest available model of garbage collection by reachability from roots.
- **Scipion / pyworkflow** — the closest existing analogue: Python, SQLite, a
  protocol graph, a GUI, and cryo-ET support. `SqliteMapper` and the `Protocol`
  class show typed slots in working code. Worth running on a small dataset and
  inspecting its project database directly.
- **cryosparc-tools** — cryoSPARC's data model exposed as a Python API:
  `add_input`/`add_output` slot declarations and `connect()` edges are the
  formalization of drag-and-drop.
- **CZ cryoET Data Portal and copick** — the domain vocabulary for datasets,
  runs, tomograms, annotations, and pickable objects. Matching it makes
  interoperability nearly free.
- **Temporal.io** durable execution and signals — the right abstraction for a run
  that waits hours for a human and survives a process restart.
- **`sqlite.org`**: `appfileformat.html`, `wal.html`, and the network-filesystem
  section of `lockingv3.html` — the last of these is why SQLite is not the
  primary store here.
- **DVC** `dvc.lock` and `dvc gc`, **Snakemake** `--rerun-triggers`, **Nextflow**
  task hashing and `-resume` — three implementations of staleness; read what goes
  into the hash, not the tutorials.
- **RELION's pipeline model** (`default_pipeline.star`: processes, nodes, input
  and output edges) — proof that the graph is what matters and that a flat file
  can carry it. Check the state of CCP-EM's `ccpem-pipeliner`, which is a Python
  implementation of that graph and may be adoptable rather than reimplemented.

Airflow is deliberately absent: fixed DAGs defined at parse time, a heavy service
footprint, and a design that wants to own run state. Nextflow is a better
conceptual fit and has Slurm executors, but it owns its work-directory layout and
carries its own provenance model, which would mean two registries — the situation
[architecture_decisions.md](architecture_decisions.md) exists to prevent. If an
external engine is wanted later, the plan tables above can generate it: the plan
is data, engines are backends, exactly as tool wrappers are backends today.
