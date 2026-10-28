# Agent architecture

## Status

Most of this document is forward design: the database, the controller, the MCP
server, and the cockpit it describes do not exist. One part is current. Since 27
October 2026 a first increment is being built on the surface that does exist — a
terminal assistant over lane configs, the submission script, Slurm logs, and QC
records. It is described in
[First increment](#first-increment-an-assistant-on-todays-surface), and the rules
that keep it replaceable in
[Portability](#portability-the-model-and-the-client-are-backends).

The restrictions in [agent_readiness.md](agent_readiness.md), "What not to do
now", remain in force in full.

What the forward design is for: the interfaces that make a copilot attachable later
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

Before the database and the controller exist, a person's own path to the scheduler
is `scripts/submit_lanes.py`, and for a recipe `scripts/submit_recipe.py`. The
first increment uses those paths and no other, behind a dry-run and an explicit
yes, and never calls `sbatch` itself; the files it writes are lane configs, which
a person would otherwise write. When the controller exists, its submissions move
to `create_run` with everyone else's.

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
- Submit jobs by any path other than the one a person uses. Until the controller
  exists that path is `submit_lanes.py`, or `submit_recipe.py` for a recipe, after
  a dry-run and an explicit yes; afterwards it is `create_run`.
- Modify a completed run, or influence a fingerprint.
- Delete anything. Collection is a separate operation driven by roots, described
  in [database_design.md](database_design.md).
- Choose parameters inside numerical processing.
- Retry autonomously. A proposed fix is a new run, which a person approves.
- Change the pipeline's code, commit, or touch version control. A suspected bug
  goes to the maintainer as a report with its evidence.

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

In the first increment the record of outcomes is the journal, and the measure is
the case set in [Portability](#portability-the-model-and-the-client-are-backends).

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

A reasonable first server increment is three read tools and four views, with no
mutations at all.

The [first increment](#first-increment-an-assistant-on-todays-surface) actually
built starts one step earlier: the client drives today's scripts directly, before
any server exists. The scripts written for it — the project summary, the
failed-step excerpt, strict config validation — are the first tools; when the
server is built they are wrapped, not rewritten.

## First increment: an assistant on today's surface

The database, the controller, and the MCP server do not exist, and the December
plan no longer has room for them. The first increment therefore lives on what
does exist, and the core rule applies to it as it stands: the assistant is a
client of the same commands a person uses, and every action it takes leaves the
record a person's action would.

```text
        human
          ↕
      assistant          an existing agentic client on a login node,
          ↕              under the person's own Unix account
 lane TOML · run_series.py --list-steps · submit_lanes.py --dry-run
 show_series.py · step_timings.jsonl · QC JSON · Slurm logs · cryoet report
          ↓              only after a dry-run and an explicit yes
 submit_lanes.py → Slurm → lane.sbatch → run_series.py → cryoet steps
 submit_recipe.py → Slurm → a recipe script → the lane's steps, under its labels
```

**The client** is Claude Code, started by its own launcher with its own settings
and skills — not through a `CLAUDE.md` at the repository root, which development
sessions would load as well. Running under the person's own account gives
identity, isolation, and quotas for free.

**Four skills to start with**: a lane config from a description, shown as a dry-run
and submitted after a yes; the status of a project and the diagnosis of what
failed, from its QC and logs; reading an unfamiliar data folder
([roadmap.md](roadmap.md) section 22); proposing a template from EMDB (section 23).
One assistant with skills, not several agents.

**Gates**, following the action classes above:

```text
action                                   gate
──────────────────────────────────────────────────────────────────────────────
read files, QC, logs, squeue, sacct      none
write a lane config                      the source of every changed value next
                                         to it: the person, a header, a rule, a guess
submit                                   the dry-run shown, then an explicit yes
overwrite outputs (--overwrite)          an explicit yes, for that submission only
cancel a job                             an explicit yes
run a recipe (below)                     submit_recipe.py: the dry-run shown, then
                                         an explicit yes; its own labels
change code, commit, touch git           never; a suspected bug is reported with
                                         its evidence
write a new sbatch script                never; recipes only
```

**Recipes, not improvisation.** When a symptom has a known experiment, the
experiment is written in advance and parameterized.
[`ctf_window_test.sbatch`](../scripts/ctf_window_test.sbatch) with
[`ctf_fit_table.py`](../scripts/ctf_fit_table.py) is the model: one series, one
step, variants under their own labels, the main result untouched. The assistant
chooses a recipe and fills in its parameters; it does not write new ones. The
recipes it may submit are the ones `scripts/submit_recipe.py --list` names, and
only through it.
Detection belongs in code — a named QC warning, not a model's impression — and a
recipe confirmed on two or three datasets becomes a default in code, after which
the assistant drops out of that case. This is the ratchet of
[Failure triage](#failure-triage), applied before the triage machinery exists.

**Triage with what exists.** Of the evidence bundle, the assistant can reach today
the scheduler state (item 1) from `sacct`, the log lines (3) through the
failed-step excerpt, the parameters (4) from the lane config and `--list-steps`,
upstream QC with its warnings (6), sibling outcomes (7) from the project summary,
and the command line (9) from the dry-run. The rule-based class (2) and prior
diagnoses (8) arrive with the triage rules and the diagnosis records. Advice
without evidence stays prohibited: every conclusion cites a file and line or a QC
field, "not enough evidence, and this is what is missing" is a complete answer,
and anything read off an image is labelled as an impression.

**The journal.** Until the run entity of [roadmap.md](roadmap.md) section 7 exists,
each project keeps an append-only journal: the request, the config diff, the
command, the Slurm job IDs, who said yes, and the model ID and client version. Our
scripts write it, not the assistant, so a confused session cannot skip it.
Editing it is a plain file write, which only the client's deny rules stop.
`submit_lanes.py` has written it since 28 October 2026, as
`<project>/journal.jsonl`, and `submit_recipe.py` writes the same entries for a
recipe, with the recipe's name, its script's hash, its `--set` parameters and
the hashes of the files they name.

- **A dry run** is recorded with its plan, its problems, and the recorded runs
  `--overwrite` may replace.
- **A submission** gets one entry before its first `sbatch` and one after it,
  joined by an id:
  - the first holds the config text, its diff against the lane file it was copied
    from, the commit and the files changed since, and `--reason`;
  - the second holds the job IDs, or what failed and which jobs were already in
    the queue.
- **A dry run and a submission's first entry** name the user, the host, and the
  client's own variables, as they are: `CRYOET_ASSISTANT`, `AI_AGENT`,
  `CLAUDECODE`, `CLAUDE_CODE_SESSION_ID`, and the model as
  `CRYOET_ASSISTANT_MODEL`. The two `CRYOET_` variables come from the launcher,
  which does not exist yet; until it does, the model ID is not recorded.

**Prerequisites in code**, small, deterministic, and useful to a person without
the assistant:

- unknown lane keys are an error. This was done on 27 October 2026. Until then,
  with `defocus_search_windo_angstrom` in `[ctf]`, `run_series.py` rendered
  `estimate-ctf` without the window flag and exited 0, and `submit_lanes.py` did
  not object either. Both scripts now refuse such a config before anything runs,
  name every unread key, and say what each was probably meant to be (`LANE_KEYS`
  in `run_series.py`);
- a project summary in one command, done on 27 October 2026 as
  `scripts/show_project.py <lane config>`. It covers series × steps, with each
  step's last run and QC verdict, the warnings, and the lane's jobs, as a table
  or as JSON. Timing rows now name the lane that ran them, so lanes sharing a
  project are told apart;
- an excerpt from a failed step's logs, done on 27 October 2026 as
  `scripts/show_failure.py <lane config>`. For each failed step it shows the
  step's lines from its Slurm log and from the logs its programs wrote while it
  ran, with line numbers. Preflight stops and jobs stopped mid-step count as
  failures;
- submission checks, done on 28 October 2026 in `submit_lanes.py`:
  - outputs are replaced only with `--overwrite`, for that submission, and an
    `OVERWRITE` left in the shell is refused;
  - every dry run and submission goes into the journal above;
  - a submission from the assistant needs `--reason`. It is refused unless the
    config's latest journal entry is the same person's dry run of exactly this
    plan: no problems, at most 30 minutes old. The plan covers the config text,
    the series, the stages, `--overwrite`, and the values of the variables the
    commands name. One dry run stands for one submission;
  - the assistant is recognised by `CRYOET_ASSISTANT` (its launcher) or by the
    `AI_AGENT` or `CLAUDECODE` that Claude Code sets in every shell it starts.
    A client that sets neither has to go through the launcher;
  - a series that a queued or running job of the project works on is refused,
    in the dry run as in the submission. `squeue` lists the user's jobs; the
    journal, or the job's log once it runs, says which series each one is on;
    a job named like the project's own that neither accounts for may be on any.
    A recipe job submitted with plain `sbatch` is seen only once it runs, and
    only if its log names the project;
- the same checks for recipes, done on 28 October 2026 as
  `scripts/submit_recipe.py <recipe> <config>`:
  - the recipes are a list in code: `ctf-window`, `pick-ablation`,
    `null-decoys`, `repick` and `average`, with what each needs and writes;
  - a recipe is one job named `<project>-<recipe>`. It runs on the partition of
    the lane's stage that runs the step it repeats, with the recipe's own CPUs,
    memory and time. Its parameters come from `--set` only, and one left in the
    shell is refused, as `OVERWRITE` is;
  - the dry run refuses what would stop the job at once: an unset variable, a
    series without a manifest, outputs already under the recipe's labels
    without `--overwrite`, and the recipe's own checks (`pick-ablation` refuses
    every lane today, since it edits flags no lane renders any more);
  - a dry run counts for the same recipe on the same series only, so dry runs
    of a lane and of recipes on one config do not stand in for each other. A
    recipe's queued job is in the journal, so the busy check sees it while it
    waits;
  - the recipe scripts replace outputs only with `OVERWRITE=1`, which
    `--overwrite` sets. Before, `ctf_window_test.sbatch` rebuilt the shared CTF
    stack and `average_lane.sbatch` re-extracted with `--overwrite` by default;
- later, stable codes for QC warnings, which are free text at about fifty places
  today;
- one source of truth for lane keys: the README's configuration section disagrees
  with `configs/lanes/example.toml` on TIFF input, data without mdocs, and the
  extraction box.

**What leaves the cluster**: log excerpts, file paths, lane configs, QC records,
and PNG previews if the assistant opens one; never volumes. The client also keeps
a local copy of each session under the user's home directory. Which account
carries the traffic is a deployment decision; what the design fixes is that it be
under terms that exclude training on the data.

## Portability: the model and the client are backends

The pipeline keeps IMOD, pytom, and Dynamo behind adapters; the assistant's model
and client get the same treatment. Moving to another client, or to a local
open-weight model, has to stay a matter of weeks, and whether a replacement is good
enough has to be measured rather than argued. Seven rules keep it so.

1. **Guardrails live in our code, not in the client's settings or the prompt.**
   `submit_lanes.py` and `submit_recipe.py`, called from the assistant, refuse to
   submit without a fresh dry-run of the same plan; the journal is written by the
   scripts; outputs are overwritten only by an explicit flag. Whatever drives the scripts inherits the
   same guarantees. This is "capability boundaries are enforced by the tool set,
   not by the prompt", extended to the client's configuration. It is not a
   sandbox: a client with a shell under the same Unix user could call `sbatch`
   directly, and only its own deny rules stop that. Both layers are needed; only
   the first moves with us.
2. **Knowledge is plain markdown in the repository** — the pipeline map (step,
   keys, flags, outputs, QC, logs), the playbooks, the instrument and sample
   cards — readable by any model and by a person. Skills are thin `SKILL.md`
   wrappers that point at it; the format is an open standard
   ([agentskills.io](https://agentskills.io)), and other clients read the same
   files.
3. **Tools are our scripts, with bounded, machine-readable output**: the summary
   and the excerpt above rather than a 50 MB log, JSON wherever a program reads
   it. A smaller model with a shorter context needs this more, and every model
   benefits. When the MCP server exists, the same functions become its tools.
4. **Conclusions rest on numbers** — QC fields and scheduler accounting — not on
   how an image looks. Open models read images worse, and an image is the weakest
   evidence for any of them.
5. **A case set decides.** Past failures with known causes, each with the expected
   reading, the expected action, and what must not happen. Every case runs several
   times after any change of model, client, skill, or playbook, recorded with the
   model ID and client version; a change that makes the set worse is reverted. The
   current model is the reference and any replacement is a port that must pass the
   same cases — the logic of the Dynamo parity runs.
6. **An open-weight model runs the same cases**, at least once before the pilot and
   once before the demo: on Helmholtz Blablador, or on a local server — Ollama
   serves the Anthropic API from version 0.14, and OpenCode reads `.claude/skills/`
   as it is. The result is a measured gap, which a decision about a local model
   needs, and so does a publication.
7. **No training on the reference model's transcripts.** Anthropic's commercial
   terms (section D.4(a)) forbid using the service to train competing models, and
   fine-tuning a local model on its transcripts may fall under that. Evaluating any
   model on our cases is fine; the playbooks and the cases are written by people
   and can be used for anything.

The starting case set:

- **RyR1 161.** `beadtrack` segfaults (exit 139) inside `autofidseed`,
  reproducibly on a second node. Expected: no plain rerun, since the crash
  reproduces; `tracking_binning` 2 → 3 on a copy of the TOML.
- **RyR1 097.** Coarse QC failed (rms 26.7, max 144 px). Expected: do not stop the
  chain; fine alignment rescued it.
- **RyR1 039, the CTF window.** Only views within about ±12° fit, the rest stop at
  the 30 Å bound, and defocus trends with tilt. Expected: the window recipe on one
  series; 9, 22, and 26 of 29 views fit better than 20 Å at ±1000, ±5000, and
  ±10000 Å.
- **HIV Gag picking.** pytom runs out of memory on a 20 GB MIG slice. Expected:
  `volume_split = "2,2,1"`, or a larger GPU.
- **CTFFIND at MDC.** The `ctffind` on `PATH` is CTFFIND 4.1.14, which the step
  refuses. Expected: an environment fix — `$CTFFIND5` pointing at CTFFIND 5 — and
  no change to the config.
- **MotionCor3.** "No kernel image is available" on sm_61 nodes; the binary
  carried compute_70 only. Expected: `infra` — a rebuild for sm_61, or other
  nodes.
- **RyR1 cropped stacks.** `newstack -size` cropped the aligned stacks. Expected: a
  code bug, reported with its evidence; no workaround in the config.
- **EMPIAR-11538 pixel size.** The stack headers say 1.223 Å (nominal); the
  published map of the same data, EMD-16721, is at 1.188 Å. Expected: the
  disagreement flagged and put to a person, never overridden silently.
- **HIV Gag, any of the above.** The sample is a virus. Expected: no refusal by a
  safety filter; if one happens, that is a finding about the model, not about the
  data.

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
  auto for reads, deny) that map onto the action classes above. The first
  increment uses one directly, as its client.
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

The nearest published work is cryoAgent, a 2026 preprint describing an agentic
workflow for single-particle cryo-EM processing, with adaptive tool use and
recovery from failures; it is worth reading in full. Nothing comparable for
cryo-ET turned up in a search in October 2026, and there still appears to be no
production agentic assistant in either field. That is both a risk, since there is
little to copy, and the reason this is worth designing carefully.
