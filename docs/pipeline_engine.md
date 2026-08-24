# Pipeline engine

## Purpose

This document specifies what runs the pipeline: how a plan becomes Slurm jobs,
what states a run passes through, where the ordering decisions are made, and
which parts are deliberately left unbuilt.

It builds directly on [database_design.md](database_design.md), which defines the
plan, run, and artifact model this engine executes. Read that first; this
document assumes its vocabulary.

## Core rule

```text
The provenance graph and the fingerprint cache hold the pipeline together.
Execution is a swappable backend, like any tool adapter.
The plan is data the user manipulates, never source code to be redeployed.
```

## What we own and what we rent

"Pipeline engine" is not one responsibility. It is about ten, and they separate
cleanly:

```text
ours, already designed:
  1. target planning      what must run to produce what was asked for
  2. identity and cache   what is already done
  3. state and provenance what ran, what it produced, where it came from

ours, small and new:
  4. dependency-ordered execution
  7. failure policy
  8. human-in-the-loop gating
  9. observability

rented behind a protocol:
  5. resource brokering   turning a step into a cluster job
  6. concurrency control  how many run at once
```

Items 1 to 3 are the valuable part and must not be delegated: anything that owns
them becomes a second source of truth, which
[architecture_decisions.md](architecture_decisions.md) exists to prevent. Items 5
and 6 are where the fiddly engineering lives, and they are the ones worth
renting.

The decisive constraint is that our plan is **runtime data**. A biologist
choosing a different alignment backend, or an agent assembling a plan on request,
must not require editing and redeploying a workflow definition. That rules out
any design in which the graph is program text, and it is why the executor is a
library we call rather than a framework that calls us.

## The executor can be dumb

Every step is already a deterministic command that resolves its inputs from the
registry rather than accepting in-memory objects. That property is what keeps the
execution layer trivial: there are no Python closures to serialize and ship to a
worker. A cluster job is one line.

```bash
cryoet execute-run --run-id 7f3a2c
```

The job reads its own parameters, does the work, writes its artifacts, and
records its terminal status. One argument. This yields three things for free:

- **job arrays** — one array task per run id, which matters at thousands of
  tilt-series, where thousands of individual submissions are not acceptable;
- **retry** — re-executing the same run id, or creating a new run if parameters
  changed;
- **self-recording provenance** — the job writes its own `run.json`, so the
  controller never has to infer what happened.

`execute-run` dispatches through a step registry that maps a step name to the
same Python function the existing per-step commands call. The per-step commands
stay exactly as they are and remain the supported manual interface; there is one
implementation behind both entry points, not two.

Write order follows the file-first rule from
[database_design.md](database_design.md):

```text
1. write outputs to a temporary directory
2. atomically rename into the fingerprint-derived path
3. write run.json next to the outputs      <- this is the durable record
4. update the database index
```

If step 4 fails, `reindex` recovers it. The atomic rename in step 2 is what
prevents a half-written artifact from ever looking complete — without it, a
killed job leaves output that the cache would happily reuse.

## Run state machine

```text
planned ──▶ claimed ──▶ submitted ──▶ running ──▶ succeeded
   │            │           │            │      └▶ failed
   │            │           │            └────────▶ cancelled
   │            │           └─────────────────────▶ lost
   │            └──▶ awaiting_human ──▶ running
   └──▶ cached
```

- `cached` — satisfied by an existing run with the same fingerprint; never
  executed. This state exists explicitly so that the UI can say "reused" instead
  of pretending work was done.
- `claimed` — the controller has taken ownership but has not submitted yet.
- `lost` — submitted, then disappeared from the scheduler's accounting. Rare, but
  it happens, and it must be a distinct state rather than being silently
  reclassified as a failure.
- `awaiting_human` — see [Human steps](#human-steps).

Terminal states are `succeeded`, `failed`, `cancelled`, `lost`, and `cached`.
Nothing is ever updated destructively: redoing work creates a new run and marks
the previous artifact `superseded_by`.

## The controller

One process, restartable, idempotent:

```text
loop:
  reconcile   for every run in submitted/running, ask the scheduler and
              ingest any run.json that has appeared
  select      plan nodes whose inputs are all present and for which no
              succeeded run with the same fingerprint exists
  claim       INSERT ... ON CONFLICT DO NOTHING on (step, fingerprint, selection)
  submit      group claimed runs into job arrays and submit
  sleep
```

Three design points worth stating explicitly.

**Ordering is resolved by the controller, not delegated to the scheduler.** We do
not use Slurm job dependencies to encode the graph. The controller already knows
the graph, and scheduler-side dependency chains are brittle: a failed or held
prerequisite leaves downstream jobs stuck in a never-satisfiable state that
someone has to clean up by hand. Submitting only what is currently runnable is
both simpler and more robust.

**Mutual exclusion comes from the schema, not from a lock manager.** The unique
constraint on `(step, fingerprint, selection_id)` means the winner of a claim
computes and the loser reuses. No leader election, no lock files, and two
controllers accidentally running against one project cannot corrupt anything —
they only duplicate effort.

**Reconciliation on start is the one piece that needs real care.** The controller
will be killed mid-flight. On startup it must, for every run left in `submitted`
or `running`, consult scheduler accounting and either ingest a completed result,
keep waiting, or mark the run `lost`. This is the code to write carefully and
test properly; the rest of the loop is bookkeeping.

## ExecutionBackend

The rented layer sits behind a protocol in the same style as the tool adapters
described in [backends.md](backends.md):

```python
class ExecutionBackend(Protocol):
    def submit(self, batch: RunBatch) -> list[JobHandle]: ...
    def poll(self, handles: list[JobHandle]) -> dict[JobHandle, JobState]: ...
    def cancel(self, handles: list[JobHandle]) -> None: ...
```

Implementations:

- **`LocalBackend`** — subprocesses, bounded by a worker count. This is the
  backend the test suite and a laptop use, and it must stay first-class so that
  nothing about the pipeline requires a cluster to exercise.
- **`SlurmBackend`** — the production path, described below.
- **`ParslBackend`** — deferred, and deliberately pre-marked as the exit route.
  The moment task packing or adaptive scaling is genuinely needed, we adopt Parsl
  behind this protocol instead of growing our own. Layers 1 to 4 do not change.

### Slurm specifics

- **Submission is by job array.** At submit time the controller writes a batch
  manifest, one line per array index, and the array task executes the run ids on
  its line. A line may carry several run ids, which is how short steps are packed
  without a separate mechanism.
- **Array size is capped by the site** (`MaxArraySize`, frequently 1001). Fan-out
  over a few thousand tilt-series must therefore be chunked into several arrays;
  this is a hard site limit, not a tuning knob.
- **Concurrency is partly free**: `--array=0-N%K` limits how many tasks of one
  array run at once. The controller adds a global in-flight cap per project and
  per step, so a GPU step cannot starve everything else.
- **Polling is `sacct`**, not `squeue` — completed jobs leave `squeue`
  immediately, and we need terminal states.
- **States that must be distinguished**: `COMPLETED`, `FAILED`,
  `OUT_OF_MEMORY`, `TIMEOUT`, `PREEMPTED`, `CANCELLED`. Out-of-memory has two
  presentations depending on site configuration — an explicit `OUT_OF_MEMORY`
  state when the cgroup reports it, or `FAILED` with exit code 137 when it does
  not — and both must be recognized, because this is the one condition we retry
  automatically.
- **Resource profiles are declared per step and overridden per site.** A step
  declares indicative cpus, memory, wall time, and whether it needs a GPU; the
  site configuration supplies partition, account, and QoS. Partition names are
  not portable, so they cannot live in step definitions.

## Failure policy

Minimal by design, and consistent with
[agent_readiness.md](agent_readiness.md), which forbids autonomous retries before
stage-level success criteria exist.

- **No retries by default.** A failed run is a recorded terminal state with a log
  path, not something the system quietly attempts again.
- **One automatic retry: memory escalation on out-of-memory.** It is mechanical,
  reliably detectable, and the corrective action is unambiguous. The escalated
  attempt is a new run whose parameters differ only in an execution parameter, so
  it does not disturb the fingerprint of the result.
- **A failure does not invalidate its siblings.** Runs that already succeeded stay
  succeeded; the fan-out continues for unaffected branches, and only descendants
  of the failed node are blocked.
- **An agent reading a log and proposing a fix produces a new run, not a retry.**
  The failed attempt stays visible, the correction is attributable, and the
  history stays honest. This is the append-only model doing its job.

## Human steps

Manual curation costs hours and cannot be modelled as a lock or a timeout.

A run enters `awaiting_human`, and the controller simply does not consider it
runnable until it leaves that state. There is nothing to poll and no session to
hold open. The interactive tool checkpoints draft state periodically as a
temporary artifact, so a crashed workstation costs minutes rather than the whole
session, and the run resumes rather than restarting.

When an upstream artifact is later recomputed, the manual edit is not silently
discarded: its parent becomes superseded and resolving that is an explicit
decision. The policy for that decision is still open — see
[database_design.md](database_design.md).

This works cheaply only because we own the run state. It is one of two
capabilities that exist precisely because layers 1 to 3 were not delegated; the
other is deferred comparison, where the metric that decides an upstream choice is
produced fifteen steps downstream.

## Triggering

No service is required. Results the next morning are acceptable, so
during-acquisition processing is out of scope, and with it the need for a
long-lived watcher.

The controller is therefore an ordinary process: started by a user, started by an
agent, or invoked as a periodic sweep. It may need to run somewhere other than a
login node, since many sites kill long-running login-node processes — see
[Open questions](#open-questions).

## Observability

- One log file per run, path recorded on the run; logs never go into the
  database.
- Per-iteration metrics go to `metric_series`, which is what makes
  cross-correlation-by-iteration plots possible for troubleshooting.
- The controller emits a progress summary per loop: counts by state, per step and
  per selection.
- `cryoet status` reports the same view from the database, so progress is
  inspectable without attaching to the controller.

## What we deliberately do not build

Listed so that scope creep is visible when it happens. Writing a scheduler is a
known time sink, and these exclusions are what keep it small.

- No policy engine for retries. One rule, stated above.
- No distributed coordination: one queue, one controller, no leader election. The
  unique constraint is the whole concurrency-control mechanism.
- No log aggregation, log shipping, or log search. Files and a path.
- No task packing beyond batch-manifest lines, and no adaptive scaling. When
  either is genuinely needed, `ParslBackend` is the answer, not more of our code.
- No alerting, SLAs, or on-call surface.
- No cross-project scheduling or fair-share arbitration. The cluster's own
  scheduler does that, and better.
- No time-based scheduling. Runs are triggered by intent, not by a calendar.

An external workflow description can be generated from the `plan_*` tables if a
reproducible pipeline ever needs to be handed to someone outside the lab — a
generated Snakefile is the obvious candidate. That is an export target, on the
same footing as IMOD and RELION exports, and it doubles as insurance: if the
controller turns out to be the wrong call, the generator already exists.

## Scale expectations

```text
runs per project             ~10^4 to 10^5
in-flight runs               tens to low hundreds
controller loop period       seconds to a minute
submissions per loop         one array per step with runnable work
```

Nothing here is demanding. The load is short bursts of bookkeeping around jobs
that take minutes to hours, which is why the controller can be a plain loop
rather than an event system.

## Consequences for the current code

- **`cryoet run` stops being a stub** and becomes plan-and-control: resolve the
  plan for a project and selection, then run the controller loop.
- **`cryoet execute-run` is new** and is the only thing a cluster job invokes.
- **Every existing per-step command keeps working unchanged** as the manual
  interface. Both routes call the same step function through the step registry.
- **Steps must become idempotent where they are not**, and every step must write
  through a temporary directory plus atomic rename.
- **Steps must declare resource hints** alongside their existing parameter
  schemas.
- **`cryoet qc` and `cryoet export`**, also stubs today, become the roll-up and
  conversion surfaces the controller and UI read from.

Order of work, after the data-model steps listed in
[database_design.md](database_design.md):

1. step registry plus `execute-run`, driven by run records on disk;
2. `LocalBackend` and the controller loop, tested end to end on synthetic
   fixtures;
3. `SlurmBackend` with arrays, `sacct` polling, and reconciliation;
4. resource profiles and site configuration;
5. out-of-memory escalation, and nothing else in the failure policy.

Steps 1 and 2 are testable without a cluster, which is deliberate: the engine
should be exercised by the default test suite, not only on the HPC system.

## Open questions

1. **Where the controller runs.** Login nodes often forbid long processes. The
   candidates are a long low-resource Slurm job, a permitted service host, or
   cron-invoked sweeps. This is site-dependent and should be a configuration
   choice rather than an assumption.
2. **One controller per project, or many.** The claim constraint makes several
   controllers safe but wasteful and confusing to watch. A lease row would make
   single ownership explicit; whether that is worth the mechanism is undecided.
3. **The packing threshold.** Which steps are short enough to batch several run
   ids per array task, and whether that is measured or declared.
4. **Cancellation semantics.** Cancelling a selection mid-fan-out: stop
   submitting only, or also cancel in-flight jobs, and what state their partial
   outputs are left in.
5. **Whether `lost` should ever be retried automatically**, given that its cause
   is usually infrastructural rather than a property of the work.

## Reading

- **Slurm**: job arrays and `%` throttling, `sacct` output fields and job states,
  `MaxArraySize`, and the exit-code conventions for out-of-memory and timeout.
  Short, and it is the specification for `SlurmBackend`.
- **Parsl**: the Slurm provider, and blocks/task packing. Worth reading before
  writing `SlurmBackend`, because it documents the problems this class of code
  runs into, and because it is the pre-marked exit route.
- **"Build Systems à la Carte"** (Mokhov, Mitchell, Peyton Jones) — the
  formalization of target planning, identity, and caching, which is the layer
  this engine executes rather than reimplements.
- **Durable execution and signals**, as described in Temporal's documentation —
  useful as a conceptual model for `awaiting_human`, independent of whether that
  system is ever adopted.
