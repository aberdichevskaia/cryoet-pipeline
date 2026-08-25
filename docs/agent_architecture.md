# Agent architecture

## Status: forward design

Nothing in this document is MVP work. The stance in
[agent_readiness.md](agent_readiness.md) is unchanged — this project is not
building agents now — and its "What not to do now" section remains in force in
full.

What this document is for: the interfaces that make a copilot attachable later
are cheaper to keep open than to reopen. Writing the design down means the small
number of decisions that are expensive to reverse can be made deliberately, and
everything else can be deferred with confidence. The two items that genuinely
need to be right early are already recorded as near-term work in
[roadmap.md](roadmap.md) sections 6 and 7; they are referenced rather than
repeated here.

## Core rule

```text
An agent is a client of the same API a human uses, never a component inside it.
Every agent action materializes as the same records a human action would.
The database is the only channel between an agent and execution.
```

The second line is what makes the first enforceable. The third is what makes an
agent bug survivable: an agent writes intent, the controller reads it and acts,
so a looping agent produces surplus rows rather than surplus GPU jobs. Claiming,
quotas, resource profiles, and concurrency limits described in
[pipeline_engine.md](pipeline_engine.md) apply to agent-created work automatically
because there is no second path to the scheduler.

This is a blackboard architecture: participants coordinate through shared
structured state rather than by messaging each other. That applies to
agent-to-agent coordination too — see [Multiple roles](#multiple-roles).

## Topology

```text
        human
       ↕      ↕
    agent    GUI / CLI
       ↕      ↕
   ┌──────────────────┐
   │   project API    │  typed mutations; read-only SQL for queries
   └────────┬─────────┘
            ↓
        database
            ↓
       controller          ← the only thing that talks to the scheduler
            ↓
      execute-run → steps → backends (IMOD, pytom, …)
```

The forbidden edges matter more than the permitted ones:

- **agent to scheduler** — never;
- **agent to controller** — never, only through the database;
- **agent inside a step** — never. Suggesting a parameter that a human accepts and
  that is then recorded on a run is one thing; choosing a parameter at runtime
  inside numerical processing is another, and it stays prohibited;
- **agent to agent** — no direct messaging; coordinate through stored state;
- **agent to filesystem** — artifacts are read through the API with a size bound;
  writes go only to scratch and export locations.

## Action classes and gates

Gates should be derived from risk, not attached ad hoc.

```text
class                     example                          gate
─────────────────────────────────────────────────────────────────────────────
read, verifiable          "tomograms with residual < 1.5"  none
read, not verifiable      "why did this fail"              evidence required
mutate, mechanical        create run / plan / selection    cost estimate + confirm
mutate, generative        write a script or converter      sandbox + recorded as a run
choose inside processing   pick a defocus range at runtime  prohibited
```

For the second class, the rule is that **advice is not issued without evidence**.
A tool returns its answer together with what the answer rests on — specific
metric rows, specific log lines — and the answer must cite them. This is the only
available defence against the characteristic failure of these systems in
scientific work, which is confident interpolation where no data exists.

The same tools therefore return **coverage**: "no comparable runs found" is a
structured result, not something the model has to volunteer. Without this, an
absence of evidence is indistinguishable from a weak prior.

## Tool surface

The discipline is the same one that applies to the schema: write the queries
first, derive the tools from them. Roughly eight tools suffice.

Reading is free-form SQL against a read-only role, with a statement timeout and a
row limit:

```text
query(sql)             read-only role, timeout, row cap
read_artifact(id, …)   preview, slice, header, or first N rows — always bounded
read_log(run_id, …)    tail or filtered lines, bounded output
```

The asymmetry is deliberate: **reads are free-form SQL, writes are typed calls
only.** Language models write SQL well, the schema is documented and stable, and
the alternative is endless proliferation of narrow read tools. Writes through SQL
would discard every invariant at once.

### `read_artifact` returns interpretation, not a header dump

A raw MRC header dump is the wrong output, for a person and for an agent alike.
The answer to a question as simple as "what is the box size" is not a field in the
file:

- **Voxel size is derived, not stored.** It is `cella / mx`, and `mx` is not
  always `nx`; the naive `cella / nx` is silently wrong when they differ.
  [`inspect_mrc_file`](../src/cryoet_pipeline/mrc_validation.py) already gets this
  right by going through `mrcfile`'s `voxel_size`, and the interpreted view should
  reuse it rather than re-derive it.
- **Box size only means something against a target.** A template's physical extent
  is fixed, so its box in pixels depends on the voxel size it will be resampled
  to — which lives on the tomogram artifact, not in the map. The useful answer
  joins the two and then checks the box against the particle diameter, which is
  the rule `TemplatePreparationQc` already enforces. This is precisely why the
  question belongs to the pipeline and not to a header utility.

Fields a raw dump hides and an interpreted view must surface:

- box in voxels and in angstrom, per axis, with the derived voxel size;
- dtype by name, not by mode number;
- **axis order as stored**, from `mapc`/`mapr`/`maps`, with a flag when permuted —
  IMOD writes `.rec` in `YZX`, as [storage_policy.md](storage_policy.md) records,
  so `nx` is not reliably the X of the array as indexed;
- **one effective origin**, reconciling the MRC2014 `origin` against
  `nxstart`/`nystart`/`nzstart`, and flagging disagreement between them — different
  programs write different ones, and a silent disagreement becomes a coordinate
  shift downstream;
- whether the file declares MRC2014 conformance (`nversion`), since that bounds
  which conventions can be trusted;
- a contrast hint from `dmin`/`dmax`/`dmean`, labelled as a hint: excursion around
  the *median* is the reliable signal, and that check already exists in template
  QC;
- file completeness, which `validate_complete_mrc` already provides.

The same interpretation should be reachable two ways from one implementation: a
`cryoet describe` command for a person, and this tool for an agent. The value is
not agent-specific — the same wall of fields is unreadable for whoever is looking
at it — so the human-facing half is worth having whether or not the agent surface
is ever built.

This depends on the hard queries existing as views, rather than on the model
composing recursive CTEs correctly each time:

```text
v_run_summary          run, step, backend, status, duration, logs
v_artifact_ancestry    the transitive closure, already materialized
v_metric_wide          qc_metric pivoted for the common metrics
v_variant_comparison   deferred comparison: final metric against upstream choice
```

The last one is the query behind "which aligner was better", where the deciding
metric is produced fifteen steps below the decision. If it exists as a view, an
agent writes a trivial SELECT. If it does not, the agent reinvents an ancestry
traversal each time and will occasionally get it wrong silently.

Mutations are typed and, where they change the plan, return a diff rather than
applying it:

```text
create_selection(predicate | ids)        returns predicate + snapshot for approval
propose_plan(...) / add_plan_node(...)   returns a diff, does not apply
create_run(step, params, inputs, selection)
cancel_run(id)
write_export(artifact_ids, format, dest) export storage role only
run_script(code, inputs, outputs)        sandboxed, recorded as a run
```

`create_run` is the only execution primitive; everything else composes from it.
The propose-then-approve shape is borrowed from agentic coding tools, where it is
the pattern that makes shared authorship over one artifact workable.

**Capability boundaries are enforced by the tool set, not by the prompt.** A
read-only analyst role is a server configuration without mutation tools, not an
instruction telling a model to behave.

## Parameter advice

Three tiers, in this order, with the tier reported alongside the answer.

**Formulas first.** Nyquist, Crowther, thickness estimation, box size relative to
particle diameter. These are already implemented in the codebase, and the agent
should call them rather than reproduce them from memory. The trip notes record
this as a design preference in its own right — defaults should come from formulas
— which means a good share of "parameter advice" stops being a question for a
model at all.

**History second.** "What defocus range worked on datasets with similar pixel
size and thickness" is a query over past runs and their QC outcomes. This is
where a lab-wide database earns its keep: an accumulating record of what worked is
something no external tool can supply, and it is the only route to the biologists'
request for the system to indicate which backend suits a given dataset without
hand-labelled ground truth.

**Model priors last**, and explicitly labelled as such.

Whether the advice was any good becomes measurable for free through the
attribution fields in [roadmap.md](roadmap.md) section 7: a run records who
created it and who approved it, so the acceptance rate of suggestions and the QC
outcome of accepted ones are both queryable. No separate feedback mechanism is
needed.

## Human, external, and imported steps

Three cases that are easy to conflate and that differ in reproducibility.

**Interactive.** The step is ours, the tool is integrated, the parameters are
known, and a person does the work — cleaning picks, excluding views. Modelled as
`awaiting_human`, described in [pipeline_engine.md](pipeline_engine.md).

**External.** A person did something entirely outside the system, with their own
script or a tool we do not wrap, and wants to re-enter. The step type, parameters,
and software version are unknown. What can be recorded: the inputs if they were
ours, the outputs, a description, and who attested to it.

**Imported.** Data entered mid-pipeline with no known upstream at all — the
`import-artifact` path in [roadmap.md](roadmap.md) section 5.

Two consequences:

- The provenance class must be recorded at entry and must propagate downward.
  This is in [roadmap.md](roadmap.md) section 7, because it cannot be backfilled.
  The practical effect is that a subtree descended from attested data is honestly
  marked as not reproducible from scratch, instead of appearing reproducible
  because its fingerprints are well-formed.
- External inputs must pass validation on entry: shape, dtype, pixel size, axis
  order, coordinate convention. A person can easily bring back a STAR file in the
  wrong convention or an MRC with a stale header, and this is the error class that
  costs months rather than hours.

Entry is a good agent task and a safe one: given a directory of files, inspect
headers, propose what each artifact is and how it should be registered, and let a
person confirm. It also serves the request for the pipeline to accept datasets
that start at different stages and carry different metadata formats.

## Plots and ad-hoc analysis

**Templates first.** With `qc_metric` in long format and `metric_series` per
iteration, a small set of templates — histogram, scatter, per-iteration line, FSC
curve, variant comparison — covers most requests. The agent picks a template and
writes a query. No code execution, reproducible by construction.

**Generated code as the fallback.** Sandboxed, reading artifacts through the API
and writing only to scratch or export; the result registers as an artifact with
`storage_role=export`, and the script is stored alongside it as part of that
artifact's provenance. A plot without its code is unauditable a month later, and
the code costs kilobytes.

Three details worth fixing in advance:

- **The fingerprint of a generated-script run is the hash of the code**, not a
  step version. Same script on the same inputs is a cache hit; a different script
  is a different fingerprint and cannot collide.
- **Stored is not trusted.** Re-running a stored script — "reproduce figure 3" —
  goes through the same sandbox. Living in the project tree confers nothing.
- **A script without its environment is reproducible only nominally.** The
  resolved versions of what it ran against belong in `Artifact.software_versions`,
  which already exists; no new mechanism is needed.

Generated code stays out of the codebase. Promotion to a maintained template is a
separate decision with a real cost: once code is in the codebase its version
participates in the fingerprints of every run that uses it, so changing it
invalidates cached results, whereas a one-off script living in a project
invalidates nothing. Promotion is also measurable rather than a matter of taste —
since each generated script is a run with a code hash, recurring hashes are a
`GROUP BY`.

## Failure triage

The application worth designing most carefully, because the naive version — hand
the log to a model — will not work in this ecosystem.

### Failures arrive in cohorts

At hundreds to thousands of tilt-series, the population of failures is the
evidence:

- one series of four hundred failed — something about that series' data;
- four hundred of four hundred failed — configuration or environment;
- forty-seven failed and all have unusually high defocus — a correlation with a
  data property, which is a query rather than an inference.

Diagnosis should therefore be per cohort, not per incident. Otherwise a person
gets forty-seven separate conversations about one problem.

### Why raw logs are not enough

- **The real error is not at the end.** IMOD prints an error mid-run, continues,
  and fails later somewhere else.
- **Messages mislead systematically.** `tiltalign` reports too few points when the
  cause is a thin seed model produced by an upstream step that *succeeded*, with a
  warning in its QC.
- **Many causes are not in the log at all.** Out-of-memory, timeout, and
  preemption live in scheduler accounting; the log simply stops.
- **Some failures are obscure by design.** A box-size disagreement between
  template and mask makes pytom fail in a way that means nothing to a reader and
  everything to a rule — this one is already documented in
  [roadmap.md](roadmap.md) section 2.

### Deterministic first, model for the residual

Classification happens in code before a model sees anything:

```text
infra    OUT_OF_MEMORY, TIMEOUT, PREEMPTED, NODE_FAIL, exit 137, quota,
         CUDA init failure, missing module or executable
config   incompatible parameter combination, missing input, size mismatch
data     the step ran and rejected the data: too few fiducials, no CTF fit,
         no peaks above threshold
code     our own exception and traceback
unknown  everything else
```

Most real cluster failures are `infra` or `config`, where a rule gives a certain
answer instantly and a model gives a probabilistic one slowly. The agent handles
the residual.

This yields a ratchet worth stating explicitly: **the agent's job is to turn
unknowns into rules.** A confirmed diagnosis becomes a signature, and the next
occurrence is deterministic. The system becomes less agentic over time, in the
good sense.

The prerequisite is uniform, addressable logs plus a per-backend
`parse_failure` — [roadmap.md](roadmap.md) section 6. Until logs are addressable,
triage quality is bounded by whether the evidence can be found at all.

### The evidence bundle

```text
1. scheduler terminal state: state, exit code, MaxRSS vs requested,
   elapsed vs limit
2. rule-based class and failure signature
3. log: head, tail, and every line matching the error grammar, with line
   numbers — never the whole file
4. the run: step, backend and version, content and execution parameters,
   fingerprint
5. inputs: kind, shape, dtype, pixel size, binning, provenance class
6. upstream QC, including warnings
7. sibling outcomes: the same step across other series — one failure or four
   hundred
8. prior diagnoses with a similar signature, in this project and lab-wide
9. the command line actually executed
```

Items 6 and 7 are what no general-purpose log analyser has. Every QC model in the
codebase already carries `warnings`, and an upstream warning is the best single
predictor of a downstream failure. Sibling outcomes make a hypothesis
*checkable*: the agent can confirm or refute by query rather than by reasoning,
which moves the whole activity from the unverifiable read class toward the
verifiable one.

### The output is a record

```text
diagnosis(id, run_id, created_at, origin, created_by,
          class,            infra | config | data | code | unknown
          signature,        normalized fingerprint of the failure
          cause,            one sentence
          evidence_json,    log line numbers, qc ids, sibling counts used
          proposed_action,  none | escalate_memory | change_param |
                            rerun_upstream | exclude_views | ...
          proposed_run_id,  a concrete unapplied run
          confidence,
          confirmed_by, outcome)
```

- `signature` collapses four hundred identical failures into one diagnosis and
  makes future occurrences match instantly. Without it the feature does not scale.
- `proposed_run_id` presents the fix as a concrete unapplied run rather than as
  prose. Approval turns it into `planned`.
- `outcome` records whether the fix was applied and whether it worked, which makes
  triage quality measurable by the same mechanism as parameter advice.

A cohort is repaired through a selection: one diagnosis per signature, the member
list as a selection, and the proposed action applied to it once. The entity
designed for choosing particles turns out to fit here unchanged, which is some
evidence the data model is right.

### Steps that succeed and produce garbage

The more expensive class: a CTF fit returned at 40 Å with exit code zero,
template matching finding nothing above threshold, an average that is a blob.
These are QC failures rather than run failures, and they surface a week later.

The machinery is identical — same bundle, same diagnosis record — with the
trigger being `QcStatus.FAIL` or `WARNING` instead of an exit code. Worth building
in from the start, since it is easy to build triage for crashes only and miss the
costlier problem.

### What triage must not do

- Not apply fixes autonomously, with the single exception of memory escalation on
  out-of-memory, which [pipeline_engine.md](pipeline_engine.md) already permits.
- **Not suggest simply rerunning.** It is the most common wrong answer and it
  burns cluster time; acceptable only for `infra` with a transient signature.
- Not diagnose without evidence. "Insufficient evidence, and here is what is
  missing" is itself actionable — usually it means an adapter failed to capture
  stderr, which is our bug.
- Not touch the failed run. The diagnosis attaches to it; the run stays failed.

## Candidate applications

Ordered roughly by value over cost. Note how many are mostly deterministic work
with a model at the edges rather than model-driven: that is the intended shape.

1. **Physical-consistency auditor.** Scan a project for what is physically
   suspicious: pixel size in a header disagreeing with the manifest, a template at
   a different voxel size than its tomogram, a defocus range that excludes the
   fitted values, an alignment computed at one binning and applied at another
   without scaling, an inverted-contrast template. These do not crash; they
   quietly produce nothing or noise. Rules do the detection, the agent explains
   and handles the ambiguous cases. Probably the highest-value item here, because
   it targets the error class that costs months.
2. **Failure triage with accumulation.** As above. The accumulating part —
   diagnoses becoming signatures becoming rules — is what distinguishes it from a
   chat transcript.
3. **QC triage at scale.** Two thousand series produce two thousand QC records.
   "These forty-seven look bad, grouped by cause, with previews" is a query plus
   clustering. This serves the request to inspect results after each step without
   switching to IMOD for every volume.
4. **Plan authoring from intent.** "Process this dataset the way we did the HIV
   one, but with a different aligner" — read the old plan, diff the step list,
   produce a new plan for approval. This is the drag-and-drop capability expressed
   as text, and it is safe because the output is a diff.
5. **Selection authoring.** Natural language to predicate to selection, with the
   predicate shown before anything runs. A direct implementation of "select every
   tomogram where *predicate* and do *action* to them".
6. **Variant comparison reports.** `v_variant_comparison` plus a written summary.
   Already wanted as [roadmap.md](roadmap.md) section 5.3. The agent ranks and
   explains; it never picks the winner.
7. **Cost and feasibility estimation.** Before launching: how many GPU-hours this
   plan implies against the available quota, computed from recorded durations of
   past runs. Prevents the most likely agent-caused accident, and improves by
   itself as the database accumulates timings.
8. **Reproduction assistant.** A reviewer asks how a figure was produced; walk the
   provenance closure and emit a methods paragraph with software versions and
   parameters. Nearly free given the graph, and a confirmed real need.
9. **Ingestion of unfamiliar data layouts.** Given a directory, work out what is in
   it and propose an import plan. A generalization of the layout-description
   approach already used for public-archive downloads.
10. **Documentation assistant** over `docs/` and the database. Cheap, useful, not
    urgent.

## What an agent never does

- Pick the winner when comparing variants. The users were explicit that this is
  their decision, not an automatic one; the agent may rank, present, and explain.
- Write to canonical artifacts. Scratch and export only.
- Submit jobs.
- Modify a completed run, or influence a fingerprint.
- Delete anything. Collection is a separate operation driven by roots, described
  in [database_design.md](database_design.md).
- Choose parameters inside numerical processing.
- Retry autonomously. A proposed fix is a new run.

## Multiple roles

Not several long-lived services. Several **tool-restricted configurations** of one
surface: a read-only analyst, a planner with mutation tools, a triage role with
read access plus the ability to write diagnoses. The restriction lives in which
tools the role is given.

If roles ever need to hand work to each other, they do it through stored state —
a diagnosis record, a proposed plan — not by talking. Same reason as everywhere
else in this document: what passes between participants should be inspectable
afterwards.

## Trust escalation

Start with everything gated. Loosen per action class, based on recorded outcomes
rather than on impression. Because every agent action is a run with attribution,
the questions "how often was advice accepted" and "did accepted advice lead to
passing QC" are queries. That is the honest path to autonomy, and it is the same
condition [agent_readiness.md](agent_readiness.md) already sets: no autonomy
before stage-level success criteria exist.

## Delivery shape

Build the surface, not the agent: expose the tools as an MCP server over the
database and the project API.

- It separates what the pipeline exposes from which assistant drives it — the same
  adapter discipline applied to tooling. Models and products change; the surface
  does not.
- It gives an immediate working loop. The surface can be driven from an existing
  agentic CLI on day one, with no UI, which is the fastest way to learn which
  tools and views are actually needed. The eight tools and four views above are a
  hypothesis; two weeks of real use will revise them.
- The same server later serves a web cockpit, so the surface is not built twice.
- Role restriction falls out naturally as server configuration.

A reasonable first increment is three read tools and four views, with no
mutations at all.

## What is expensive to defer, and what is not

Almost nothing here needs to exist early. Two things do, and both are already
recorded as near-term work:

- uniform, addressable external-tool logs — [roadmap.md](roadmap.md) section 6,
  which is plain hygiene and needed by a person reading logs regardless;
- attribution and provenance-class fields — [roadmap.md](roadmap.md) section 7,
  because they cannot be backfilled.

Everything else — the tool surface, diagnosis records, views, templates, any model
call — is additive and can wait.

Two disciplines, however, protect the option and cost nothing to maintain:

1. **Every mutation creates a record.** Not "agents must go through the API" but
   nobody writes artifacts outside it — no helper script dropping a file into a
   project, no UI editing something on disk. If part of the state can appear
   without provenance, an agent becomes a permanent hazard no audit can fix. While
   the human-facing CLI itself goes through the typed path, an agent attaches
   later for free.
2. **No step reads anything not declared as an input.** A hidden config file, an
   environment variable, a file found next to another one — any of these makes
   fingerprints lie, and "same parameters, same result" is the assumption
   underneath the cache, reproducibility, and every inference an agent draws.
   Cheap to hold now, unpleasant to discover later.

## Prior art worth looking at

- **Agentic coding tools.** The closest analogue in interaction pattern: a person
  and an assistant working on shared state, with proposed diffs and explicit
  approval. Worth borrowing: propose-then-approve, and permission modes (ask,
  auto for reads, deny) that map onto the action classes above.
- **Text-to-SQL and analytics assistants.** For the read surface: schema in
  context, curated views, worked example queries, read-only roles, showing the
  query before the answer. The semantic-layer idea — metrics defined once so an
  assistant does not reinvent them — is what `qc_metric` plus views amounts to.
- **LLM-assisted bioimage analysis** (Robert Haase's `bia-bob`,
  `napari-chatgpt`). Same community, same task shape of generating a small
  analysis over image data, and published experience about what works.
- **Autonomous laboratories** — Coscientist (Boiko et al., *Nature* 2023),
  ChemCrow, A-Lab (Szymanski et al., *Nature* 2023). The nearest domain analogue:
  a model proposes an experiment, a deterministic system executes it, results
  return. Worth borrowing: the proposer/executor split and the failure-analysis
  loop.
- **Asset-graph assistants** in data tooling, for the UX of explaining why a step
  failed — for interface ideas, not architecture.

There appears to be no production agentic assistant in cryo-EM or cryo-ET tooling
yet. That is both a risk, since there is nothing to copy, and the reason this is
worth designing carefully.
