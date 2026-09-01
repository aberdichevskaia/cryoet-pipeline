# Roadmap

Planned work beyond the MVP slice in [mvp_plan.md](mvp_plan.md). Ordered roughly
by dependency, not strictly by priority.

## Current order of work

The next milestone is subtomogram averaging over particles from several
tomograms. The decision recorded here is to reach it through a thin slice of the
data model rather than by building the database first.

Three facts set the order.

**Processing many tilt-series is not blocked, it is inconvenient.** A shell loop
over series works today for everything through picking.

**Cross-tilt-series averaging is blocked structurally.** `average-particles` takes
a single `--manifest`, `_particle_stack_artifact` filters on
`parameters["tilt_series_id"]` and fails unless exactly one artifact matches, and
`SubtomogramAveragingQc.tilt_series_id` is a single string. Particles from ten
tomograms cannot be averaged together at all.

**Parallel runs are unsafe at any scale.** `ArtifactRegistry` does
read-modify-write of one JSON document with no locking, so two series finishing
at the same time lose one another's entries. This bites at five series, not at
five hundred.

The database is not first because its design needs the scale-up as input. Several
of the open questions in [database_design.md](database_design.md) — the
`content_params`/`execution_params` split per step, run granularity, whether
`selection` is the right abstraction, where the relational/columnar boundary sits
for particles — are answered honestly only by running at scale and varying
parameters in anger. Building first means encoding guesses and then migrating a
schema we wrote ourselves.

The interim state is on the database's path rather than a detour: per-series
registries, or per-run JSON records plus a merge, are the file-first shape the
database will later index.

Order:

1. **Sections 6 and 8** — uniform logs and `cryoet describe`. Days, no
   dependencies. Worth doing before the scale-up rather than during it, since
   both exist to make fifty times more output readable.
2. **Scope as a relation instead of a parameter.** The minimal slice of
   `artifact_scope` without the database: tilt-series scope as a list on the
   artifact rather than a key in `parameters`, lookups that accept a set, a
   particle-set concept that references several tomograms, and QC models carrying
   a list. This is what unblocks cross-tilt-series averaging.
3. **Per-series registries or per-run records**, so parallel runs stop clobbering
   each other.
4. **Run at scale** — on the order of fifty series through to a cross-series
   average. This produces the inputs the database design is missing: per-step
   durations for later resource profiles, real failure modes, the natural run
   granularity, and which parameters were actually varied.
5. **The database**, informed by step 4.

**Trigger for step 5.** The current registry holds 58 artifacts per series at
bin8, 41 of them individual corrected projections, and every command pays full
pydantic validation twice — once to load, once to write. That gives roughly 2,900
artifacts at 50 series, 29,000 at 500, and 116,000 at 2,000, where the validation
cost alone is seconds per command invocation. The JSON registry is comfortable
into the tens of series, unpleasant around three hundred, and unusable near a
thousand. Collapsing the per-projection artifacts into one stacked artifact with a
`T` axis buys roughly five times the headroom and is worth doing anyway, since
hundreds of thousands of small files is its own problem on a parallel filesystem.

**QC schema versions are now something `reindex` has to face.**
Cross-tilt-series averaging bumped `SubtomogramAveragingQc` and
`SubtomogramClassificationQc` to `schema_version: 2`, replacing
`tilt_series_id` with `tilt_series_ids` and `particle_stack_id` with a list.
Nothing reads those documents through the model today — they are written by the
backend and read generically as JSON — so the change cost nothing. An index
that rebuilds itself by scanning the project will read them, and then a
version-1 document is a decision rather than a detail: it either migrates on
read, or is refused with a message naming the version. Sliding past it silently
would put a document's own idea of its shape at odds with the schema indexing
it, which is the failure that makes an index untrustworthy rather than merely
incomplete.

The general form is worth settling once, because the QC models are the metric
vocabulary the whole comparison layer stands on: **a `schema_version` that
nothing checks is decoration.** Either `reindex` knows how to read every
version it may meet, or the versions have to be migrated in place before it
runs.

**Scope is a field now, and the parameter it replaces is still there.**
`Artifact.tilt_series_ids` holds which tilt-series an artifact is about — a list,
because a cross-series average is about several and no single id can say so — and
`covers(tilt_series_id)` is how a caller asks. Twenty-nine resolvers and guards
that compared `parameters["tilt_series_id"]` by hand now go through it.

The parameter stays, adopted into the field, on the precedent `backend` and
`variant` set: promote the field, leave the producers. A field that disagrees
with the parameters is refused on construction and on deserialization, so the
two cannot drift apart anywhere a registry is written or read. `model_copy` runs
no validators and is the exception — a stale copy is caught on its next
validation, which for a registry is immediately, and negative tests depend on
being able to build incoherent objects, so the hole is pinned by a test rather
than closed.

Dropping the parameter from the forty-two producers belongs with the move to an
index, not before it, and [database_design.md](database_design.md) carries the
argument: that is where `parameters` becomes a JSON column beside real columns
and a duplicated scope turns into a second source of truth. Doing it earlier
means touching the same forty-two sites twice.

**Particle selection waits, and nothing is lost by waiting.** Selecting by
tilt-series already works: `covers()` is the membership test and a repeated
`--manifest` is the query. Selecting individual particles is not expressible, and
should stay that way until the fifty-series run, for a checked reason rather than
a preference — everything a selection would need is still recoverable from what
is kept. The pick-row a particle came from is a deterministic replay of the
extraction filter over inputs that all survive: the picks STAR, `box_size_px`,
`minimum_score_threshold` in the extraction QC, and the tomogram's `shape` field,
which outlives the tomogram file. The tilt-series behind a combined index is a
cumulative sum over `particle_stack_ids` and each stack's recorded count.

So no per-particle identity column, and in particular no new REQUIRED column in
the particle STAR: `read_particle_star` hard-fails on any missing name, so adding
one makes every stack already on disk unreadable and forces re-extraction. When
identity does land it is typed columns, with the new one read as optional — the
shape the optional `filled.fid` output took in fine alignment.

**The half-set FSC had a larger defect than its lack of independence, and it is
fixed.** Particles were averaged without removing their own mean. Zero padding in
`resample_volume_zyx` writes that offset into an orientation-shaped region, and
over many orientations it becomes a smooth envelope that is nearly identical in
any two subsets of the particles, because it is a function of the box and the
offset rather than of the specimen. Measured on the pipeline's own path with pure
Gaussian noise, disjoint particles, independent random orientations and no shared
reference at all: the first shell correlated at 0.975 with 20 particles per half
and 0.996 with 100, crossing 0.143 at 65 A and 52 A. It got worse with more
particles, because the envelope adds coherently while the noise averages down.
Nothing about refinement independence could have caught it — that test had
perfect independence.

`center_particle_volume` now runs at the two places particles are loaded, so
existing stacks need no re-extraction. After it the median first-shell
correlation across seeds is −0.026, negative in seven runs of twelve, and the
median run reports no crossing at all. Three tests pin the artefact, the fix, and
the wrong fix — subtracting the mean of the final map, which leaves the curve
exactly where it was, because by then the envelope is inside every term of it.

**What remains for a real gold-standard estimate**, in order: the reference is
still rebuilt from all particles every iteration, which is the defective scheme
Scheres & Chen name explicitly, and there is no low-pass on the reference
anywhere — and a shared starting reference is only legitimate *because* of that
low-pass, above which the guarantee holds. Independent refinement measured at
1.005x the compute of the joint version, since each half searches only its own
particles, so cost is not the obstacle and never was. Splitting per particle
within one tomogram is correct as it stands for discrete particles like
ribosomes; group-level splitting is an opt-in for overlapping density, which is
what the Gag-lattice question would need. Also missing: no QC anywhere records
per-half particle counts, so a 2-versus-200 split reports PASS with a number.

**Independent half-set refinement is built for `fft-refinement`, and the benefit
was not measurable.** Both halves now refine against a reference built only from
their own particles, the reference is low-passed every iteration, and the limit
comes from the FSC between the two halves — the RELION feedback, and the reason a
shared starting map is legitimate at all. The two half maps are written beside the
average, because a resolution is a claim about those two files agreeing and
without them nobody can recheck it. Compute is unchanged: each half searches only
its own particles, so the inner loop runs the same number of times.

What could not be shown is that any number improved. On pure noise, twenty-four
particles, six iterations, the first-shell correlation barely moves across
iterations under either scheme, and the drift is small and inconsistent in sign
(-0.014, -0.004, -0.038 joint; -0.007, -0.020, +0.002 independent). The reason is
visible in the code: `_search_local_alignment` is a local grid search from each
particle's current orientation with a halving step, so it has too little freedom
to imprint noise the way the global refinements Scheres & Chen describe do. **So
this change rests on being the correct scheme and on costing nothing, not on a
measured improvement** — and anyone who tightens the search later inherits the
protection rather than having to discover the need for it. The cone-search
backend searches a far larger space and is where the defect would be expected
to bite; it is still joint, and its QC says so.

The independence is structural, not total: exactly one number crosses between the
halves, the resolution limit, derived from their FSC. Half 1's map therefore does
depend on half 2's content from the second iteration on — through that scalar and
nothing else. A test found this by asserting full independence and failing, which
was worth more than the assertion would have been. Two tests now pin both
directions, because tightening it would break the anti-overfitting feedback and
widening it would let structure across.

**The low-pass rule as built, and the better version of it.** Bounded on both
sides: a crossing below shell 3 holds the schedule, and the limit can never go
coarser than the operator's starting filter. Both guards exist because the first
version had only the fine bound, and a first-shell crossing — the median outcome
on noise-dominated data — then set the limit to 86 A at box 48 and 115 A at box
64, coarser than the 60 A it started from, and held it there. Filtering a
refinement more heavily than asked is worse than holding the start.

Four improvements are known and unbuilt, in rough order of value:

- Work in **integer Fourier shells**, not angstrom. The instability is a function
  of shell index alone — the same handful of samples in the lowest shells at every
  box size — and an angstrom-space rule hides that variable.
- Drive the schedule off the **FSC = 0.5** crossing while continuing to report at
  0.143. That is the curve RELION applies its own 0.5 rule to, so it is the
  calibrated choice; 0.143 stays as the reported number for comparability.
- Scale the margin with the box, `max(3, round(0.08 * box))`, rather than a flat
  two shells. A fixed count tuned for 200-400 px boxes is wrong at 32-64. Do not
  port RELION's `fsc0143 - fsc05 + 5` term, which at box 32 would be wider than
  the whole useful band.
- Set the filter's `edge_width_shells` to `min(2.0, margin / 2)`. Two shells is
  RELION's value, sized against ten-shell margins; against a three-shell margin
  it consumes two thirds of the headroom the margin exists to create.

**Two defects remain in the multi-class path**, and they are the same kind as the
ones just fixed. It applies **no low-pass at all** to its class references, so
joint refinement there is unprotected in the way the single-class path no longer
is. And it emits a resolution number for a jointly refined split; the right
behaviour is to emit none — a caveated number travels downstream and the caveat
does not. `gold_standard` already reports False, which is the floor, not the fix.

**Still open.** Whether classification should have independent halves at all:
each half would build its own K references and "class 2" on one side need not be
"class 2" on the other, so a per-class FSC needs a correspondence between them.
The likely established practice is to classify jointly and then gold-standard-
refine a single class, which is a pipeline shape rather than a backend flag — this
is reasoned, not confirmed, and `SubtomogramClassificationQc.gold_standard`
reports False either way. Also unbuilt: no mask before the FSC, and no
phase-randomisation correction for the mask-induced correlation a mask would
introduce.

**`--variant` does two jobs, and they need separating everywhere.** It selects
which inputs to read AND labels what gets written, with one string. So the thing
`variant_label`'s own docstring names as its purpose — "a different parameter
sweep" — cannot be done: a fresh label finds no inputs, and reusing the old one
overwrites the previous result. Comparing two settings over the same inputs is
ordinary work and the pipeline could not express it.

`average-particles` now takes `--output-variant`, which labels the output while
`--variant` still selects inputs. That is sound only because the backend uses the
label for nothing but naming — the path, the id suffix, and the artifact's own
`variant` field — which a test pins by averaging one set of particles under two
labels and asserting the maps come back byte-identical.

Every other step that reads inputs by label has the same gap: `classify-particles`
first, then the rest of the chain. Not spread now, deliberately — it is the same
three lines in each of about fifteen commands, and the useful version of the
change is driven by which comparison someone actually wants to run, not by
symmetry. Each site should also be checked the way this one was, since a label
that reaches anything but naming makes the split unsound there.

**Do not build a batch mode for step 4.** A `--manifest-glob`, a worker pool inside
the CLI, or any scheduling logic will be replaced by the controller in
[pipeline_engine.md](pipeline_engine.md). A shell loop or a hand-written job array
is the right amount of effort, and it should look disposable.

**Independent half-set refinement (section 4) belongs with step 2**, not after it.
Once particles come from five to ten tomograms, resolution claims start carrying
weight, and the current half-set FSC is optimistic because the halves are not
refined independently. The deferred comparison of backends is measured by final
resolution, so an optimistic estimate undermines the comparison machinery as well.

## 1. Split CTF estimation from CTF correction — done

`estimate-ctf` (backends `ctffind5`, `imod-ctfplotter`) writes a `CTF_MODEL`,
and `correct-ctf` (`imod-ctfphaseflip`) consumes one. Both estimation backends
emit the same interchange format, an IMOD `.defocus` file, so they are directly
interchangeable and their models coexist in one registry under
backend-qualified ids.

Notes worth keeping:

- CTFFIND fits the whole aligned stack in a single call, because a multi-slice
  MRC that is not declared a movie is treated as independent micrographs.
- Fitting is two-pass: the view nearest zero tilt is fitted over a wide defocus
  range, then the full stack over a narrow window around that hint. Supplying
  an expected defocus skips the first pass.
- The high-resolution fit limit defaults to the Nyquist limit of the stack's own
  pixel size, and a finer request is clamped with a warning. A binned stack
  makes a fixed default (say 5 A) physically meaningless.
- Correction reads voltage, Cs, and amplitude contrast off the CTF model and
  rejects conflicting overrides, so a phase flip cannot silently disagree with
  the fit it came from.
- Astigmatism angles are fitted on the aligned stack, so `ctfphaseflip` needs no
  `-xform`. Fitting on a raw stack instead would require it.

Still open: running both estimation backends produces two CTF-corrected stacks,
and `reconstruct-tomogram` then reports that it cannot choose between them.
Resolving that properly is section 5.

## 2. Template preparation from EMDB — done

`prepare-template` fetches a map by accession (cached under `<out>/emdb-cache`)
or takes a local one, then wraps `pytom_create_template.py` and
`pytom_create_mask.py` to produce `TEMPLATE` and `TEMPLATE_MASK` artifacts.
`pick-particles` resolves both from the registry, with the explicit path options
kept as overrides.

Notes worth keeping:

- The target voxel size comes from the tomogram artifact, not a flag, so a
  template cannot be built at the wrong scale.
- The mask is built at the template's *actual* box size after resampling, and a
  disagreement between the two is a hard error — pytom fails obscurely otherwise.
- Contrast: `pytom_create_template.py` in the current release has **no** CTF
  options, only `--invert` (verified against the source; a tutorial mentioning
  `--ctf-correction`/`--flip-phase` does not match the shipped tool). EMDB maps
  are bright, tomogram particles are dark, so `--invert` defaults to on. QC
  compares the produced template's excursion around its median against the
  requested inversion and warns on a mismatch, because getting this wrong finds
  nothing rather than erroring.

## 3. Inspect picks in ChimeraX + ArtiaX — done

`export-picks` writes a matched tomogram MRC plus picks STAR and a `.cxc`
script, so checking a result is one command.

The trap here: ArtiaX registers **two** readers for `.star` — "RELION STAR file"
(RELION 3.1 conventions, `rlnCoordinateX/Y/Z` in pixels, which is what
pytom-match-pick writes) and "RELION5 STAR file" (centered coordinates in
Angstrom). Both claim the suffix and both answer to the nickname `star`, so the
generated script pins `format relion` explicitly rather than letting ChimeraX
choose.

## 4. Refinement and classification — partly done

`average-particles --backend cone-search` implements DYNAMO's approach natively
in torch (no MATLAB, no DYNAMO install; the tool-named backends elsewhere are
adapters, this one is not). `fft-refinement` remains as the trivial baseline,
and a RELION5-style backend can be added behind the same
`SubtomogramAveragingBackend` protocol.

What it does:

- Orientations are searched on a cone plus an in-plane sweep rather than by
  nudging each Euler angle independently, which samples orientation space
  evenly and does not degenerate near zero tilt.
- Shifts come from the cross-correlation peak, so the whole allowed range costs
  one FFT rather than a grid of resamplings.
- Several references are refined at once with hard per-particle assignment,
  which is what makes it classification. Classes are seeded from a deterministic
  random partition; the reassignment fraction per iteration is the convergence
  signal reported in QC.
- Ranges and their sampling shrink together each iteration (`refine_factor`,
  two to three), but sampling stops at the **Crowther increment**,
  `dtheta = 180 / (pi * r_max * D) = degrees(d_res / D)` with `D` the particle
  diameter. Below that, adjacent orientations are not independent. Note RELION's
  initial-sampling heuristic `arctan(2 * d_res / D)` is about twice this;
  the implementation uses the Crowther value. Shift search never closes below
  one pixel, since the peak is located at whole-pixel accuracy.

Still outstanding:

- Independent half-set refinement, so half-set FSC becomes a real resolution
  estimate rather than an optimistic one. Target design: keep the *initial*
  reference common, then run the alignment and reference-update loop fully
  independently per half.
- Missing-wedge and CTF weighting in the alignment correlation.
  `_normalized_correlation` and `_best_shift_and_correlation` both score
  unweighted, which their docstrings say outright. This is tolerable while
  orientations are spread, because each particle's wedge then points somewhere
  different in the reference frame and they cancel. It stops being tolerable
  when the orientations cluster — see section 9 — since the wedge then averages
  coherently and can build layers of its own that look like structure.
- Symmetry.

A trap worth keeping, found while deciding whether an average had converged.
**A mean-per-slice profile taken over the whole box says nothing.** Rotating a
cube leaves zeros in the corners, they fall in different places for every
particle, and the resulting shell dominates both ends of every axis. Profiled
that way a perfectly layered average looks like equal noise on all three axes,
which is exactly the wrong conclusion. Profile over the inscribed sphere, drop
the outermost slices where the sphere holds only a handful of voxels, and put
all three axes on one scale — normalising each axis to its own maximum makes a
faint wobble draw the same picture as real layering.

The matching control is to average the same particles under **shuffled**
orientations: real alignment collapses when the angles are swapped between
particles. Its limitation is worth recording too, because it bit here. The
shuffle only breaks what the orientation *distribution* does not already fix,
so when every particle shares a tilt the swap preserves the geometry it was
meant to destroy and the control reads as a false floor.

## 5. Run any part of the pipeline, and compare backends side by side

Two use cases that are really one structural problem.

**Start and stop at any step.** Every command already resolves its inputs from
the registry rather than positional arguments, so a partial run mostly works
*if* the registry contains the right artifacts. The friction is getting external
data in: the `bypass-external-inputs` branch added throwaway
`--tomogram-mrc`/`--picks-star` options to `extract-particles` for exactly this.
The general fix is a first-class `import-artifact` path per artifact kind that
canonicalizes and registers external files, making mid-pipeline entry a
supported, tested route. A related snag: every command requires `--manifest`
even where only `tilt_series_id` is used from it, which forces a manifest to
exist for data that never entered through `init`.

**Entering as an assembled stack silently loses a step, and probably not only
one.** `import-tilt-stack` registers a `TILT_STACK` and its angles, and nothing
else — there are no frames, so no `CORRECTED_PROJECTION` artifacts exist.
`estimate-defocus-hint` resolves its input through
`_zero_tilt_corrected_projection_artifact`, so for any dataset that entered as
a stack the command cannot run at all. Found on EMPIAR-11538, where every
series arrives as one assembled stack.

The loss is not the fit itself: `estimate-ctf` performs the same first pass
internally and records the answer as `hint_defocus_um` in `CtfEstimationQc`.
What is lost is the hint being an artifact with its own `DefocusHintQc` before
the whole stack commits to it, and that guard matters here: the second pass
fits every view in a narrow window around the hint, so a hint that landed
wrong is wrong for all of them, and no single view looks obviously bad.

The fix is small — let the hint step take an aligned tilt stack and extract the
view nearest zero tilt, which is what `estimate-ctf` already does internally.
Note that CTFFIND asks a different sequence of questions for one view than for
a stack, as section 1 above records, so the two paths cannot share a dialogue
blindly.

The general point is why this belongs in section 5 rather than on its own:
**mid-pipeline entry is only a supported route if no step quietly depends on
artifacts that only the frames path produces.** One such dependency is now
known; the rest of the chain has not been audited for others, and that audit is
the actual work item. Both stated entry points, frames and tilt series, should
reach every step that does not physically need what they lack.

**Compare backends in parallel — mostly done.** `--variant LABEL` labels a run's
artifacts and selects inputs with the same label, so two lanes coexist in one
registry. It covers the alignment-to-reconstruction chain, which is where the
competing backends are: `generate-fiducial-seed`, `track-fiducials`,
`fine-align-tilt-series`, `build-final-aligned-stack`, `reconstruct-tomogram`.
Ambiguous lookups now list the candidates and name the flag instead of failing
with a bare count. Omitting the flag reproduces the previous ids exactly, so
existing registries keep working.

Worth correcting an earlier note here: the registry never replaced silently.
`add()` raises on a duplicate id unless `replace=True`, which only `--overwrite`
sets. The real defect was that ids do not encode what makes a result different,
so `--overwrite` conflated "redo this step" with "produce another variant".

Still outstanding:

1. Stages outside that chain — motion correction (`{ts}:corrected_projection:{z}`
   across three backends), coarse alignment, and the picking-to-averaging tail —
   do not take `--variant` yet.
2. Content-defining parameters are still not intrinsic to ids. Binning belongs in
   `{ts}:aligned_tilt_stack:final` and `{ts}:tomogram:fine` whether or not a
   label was given; making it so would change existing ids, so it needs a
   deliberate migration rather than a silent change.
3. A comparison report over variants, reusing the per-stage QC models that
   already exist.

## 6. Make external-tool logs uniform and addressable — half done

Every adapter that shells out already captures the right material: argv,
stdout, stderr, and the exit code. The uniformity half is done; addressability
is not.

**Done — one writer instead of twelve.** `backends/command_log.py` holds
`write_command_log` and `command_log_text`, and all twelve backends call them.
The copies had drifted into eight variants across three axes — whether argv
tokens were coerced to `str`, whether the parent directory was created, and
whether the signature took one command or a list — and each disagreement showed
up only as a crash in the adapter that lacked the tolerant version, so each was
resolved toward the tolerant one.

**Done — progress bars no longer bury the logs.** `text=True` translates tqdm's
carriage returns into newlines, so a redrawn bar arrives as one line per
update: one pytom match wrote 41,212 of them, which was the entire 2.7 MB of
that log and the reason finding the two lines that mattered needed `tail -60`.
The writer now keeps a bar's final state and replaces the rest with `[... N
progress updates elided]` — elided rather than dropped, so nobody has to wonder
whether output went missing.

**Still open — nothing on a terminal while a step runs.** `run_command` uses
`capture_output=True`, so a 26-minute match prints nothing until it finishes.
Showing output *and* recording it means the runner has to tee, which changes
its contract, so this is a real design change rather than a tweak. Worth doing:
a step that says nothing for half an hour is indistinguishable from a hung one,
and that came up while running lane B by hand.

- **Log paths are recorded under about a dozen different keys** in untyped
  `parameters` dicts: `command_log_path`, `command_log`, `log_path`,
  `tiltxcorr_log_path`, `xftoxg_log_path`, `newstack_log_path`,
  `autofidseed_log_path`, `imodinfo_log_path`, `tiltalign_log_path`,
  `preposition_tiltalign_log_path`, `xfproduct_log_path`, `residual_log_path`.
  Only `TomogramRestorationQc.command_log_path` is a typed field.
- **One step legitimately owns several logs**, because it invokes several external
  programs — fine alignment runs `tiltalign` more than once plus `xfproduct`, and
  fiducial seeding runs `newstack`, `autofidseed`, and `imodinfo`. So the target
  is not a single `log_path` but a list.

The consequence is that there is no way to answer "show me the logs for this
step" without knowing which backend ran and which private naming convention it
used. That blocks anything generic that wants to read them — a QC roll-up, a
future UI, a copilot asked why a step failed — and it is already awkward for a
person working from the registry rather than from the output directory.

The work:

1. ~~One shared writer, replacing the twelve copies.~~ Done. What it does *not*
   yet record is the resolved executable path, stdin when used, and wall time —
   all three belong with the `run` entity rather than in the log text.
2. A typed list of log records per unit of work — program name, path, exit code —
   instead of ad-hoc keys in `parameters`. This lands on the `run` entity
   described in [database_design.md](database_design.md); until that exists it
   belongs in a typed model rather than in a free-form dict.

What remains is plain hygiene and worth doing independently of anything
downstream. `write_command_log` returns the path it wrote for exactly this
reason: a caller can record the log on the artifact it belongs to as soon as
there is a field to put it in. They also unblock deterministic failure
classification: once logs are uniform, a `parse_failure(log, result) ->
FailureSignature | None` can be added to the backend protocols, where the
adapter that knows a tool's dialect is also the thing that recognizes its
errors. That parser is a follow-on, not part of this item.

## 7. Attribution — required in the first version of the run entity

Not work to start now: the `run` entity described in
[database_design.md](database_design.md) does not exist yet. This is a constraint
on its first version, recorded here so it is not discovered late.

Attribution cannot be backfilled. Every record written before these fields exist
is permanently "unknown", and no amount of later reconstruction recovers who did
what. The fields are free to add up front and impossible to add retroactively,
which is the only reason this is worth pinning down before there is any consumer
for it.

A single `actor` field is not enough, because three different questions get asked
of the same run:

- **who caused it to exist** — a person, a copilot proposing a run, or the system
  itself (the one automatic memory escalation is a system-caused run);
- **who approved it**, when approval was required, or explicitly nobody;
- **who performed it**, for interactive steps whose work is done by a person in
  an external tool.

So the minimum is `origin` (`human` / `agent` / `system`) for cheap filtering,
plus `created_by` and `approved_by`, plus `performed_by` for interactive runs.
Execution attribution is already covered separately by `host` and the scheduler
job id.

This subsumes a mechanism that would otherwise be built twice: recording that a
parameter was suggested by a copilot and accepted by a person is not a separate
feature, it is `created_by` and `approved_by` on the run that used those
parameters. Nothing extra is needed to answer later whether such suggestions were
any good.

The artifact side has one field with the same unbackfillable property: a
provenance class distinguishing `derived` (produced by a tracked run),
`attested` (a human declares what was done outside the pipeline), and `imported`
(entered mid-pipeline with no upstream). If an externally produced artifact is not
marked as such when it enters, there is no way to tell afterwards, and every
downstream fingerprint quietly overstates how reproducible the result is. The
class must propagate downward: anything descended from an attested or imported
artifact inherits that status.

## 8. `cryoet describe` — an interpreted view of a file or artifact

A raw MRC header dump is unreadable, and worse, the questions actually asked of it
are not answered by any single field. "What is the box size" requires deriving the
voxel size, and then relating it to what the file will be used against.

This is cheap: [`inspect_mrc_file`](../src/cryoet_pipeline/mrc_validation.py)
already derives shape, dtype from the mode number, and pixel spacing through
`mrcfile`'s `voxel_size` — which correctly uses `cella / mx` rather than
`cella / nx`, a distinction that matters whenever the two differ. The work is
extending `MrcFileInfo` with the fields that are currently invisible and printing
the result.

Three of the additions are correctness concerns rather than readability:

- **Axis order as stored**, from `mapc`/`mapr`/`maps`, flagged when permuted. IMOD
  writes `.rec` in `YZX`, as [storage_policy.md](storage_policy.md) records, so
  `nx` is not reliably the X of the array as indexed.
- **One effective origin**, reconciling the MRC2014 `origin` against
  `nxstart`/`nystart`/`nzstart` and flagging disagreement. Different programs write
  different ones, and a silent disagreement becomes a coordinate shift downstream.
- **Content kind** — single image, image stack, or volume — inferred from `ispg`,
  `mz`, and `nz` together rather than from `nz` alone. An MRC with `nz = 41` may be
  a tilt stack or a volume, and the distinction changes how tools treat it: CTFFIND
  reads a multi-slice MRC that is not declared a movie as independent micrographs,
  which section 1 above relies on deliberately. The current
  `shape = (nz, ny, nx) if nz > 1 else (ny, nx)` collapse does not express this.

The readability additions: dtype by name, box size in both voxels and angstrom per
axis, whether MRC2014 conformance is declared (`nversion`), and a contrast hint
from `dmin`/`dmax`/`dmean` labelled as a hint — the reliable signal is excursion
around the median, which template QC already checks.

Two modes, because some answers need the registry:

1. `cryoet describe <path>` — file only, pure interpretation, no project needed.
2. `cryoet describe --artifact <id>` — adds the producing run, QC status, lineage,
   and target-relative derivations: at the registered tomogram's voxel size this
   box becomes *N* pixels, and whether that clears the particle-diameter rule
   `TemplatePreparationQc` enforces. For a file not yet registered, an explicit
   `--against <artifact-id>` supplies the target.

Zarr artifacts should go through the same command — shape, dtype, chunking, axis
order, and pixel spacing — even though our own stores are far less trap-laden than
foreign MRC files, so that there is one way to ask what something is.

No new dependency, and testable with tiny synthetic fixtures. The same
interpretation is what an agent surface would expose later, as recorded in
[agent_architecture.md](agent_architecture.md); building the human-facing half now
is worthwhile on its own, since the wall of fields is equally unreadable whoever is
looking at it.

## 9. A geometric layer, for lattices and membrane-bound particles

Everything above treats particles as independent points with independent
orientations. That is right for ribosomes and wrong for a Gag lattice on a
virion, and the difference is not cosmetic — it breaks specific machinery.

Three consequences, in the order they bite:

**Half-sets split by particle index are not independent.** Neighbouring picks on
one virion overlap in the tomogram and share the same alignment error, so
splitting `1 + index % 2` puts correlated particles on opposite sides of the
FSC. The resolution estimate is then optimistic for a second reason on top of
the one section 4 records. The fix is to split by the object a particle belongs
to — one virion goes entirely into one half.

**Picks on a continuous lattice are not distinct particles.** Template matching
returns many peaks per lattice patch, offset along the lattice rather than
centred on anything, and the shift along the lattice is degenerate. Deduplicate
by distance *and* by orientation consistency: two picks a lattice spacing apart
with the same orientation are one measurement, not two.

**The orientations cluster, and that couples to the missing wedge.** Membrane
normals on a virion in a slab tomogram are not isotropic — the sides are
resolved far better than the top and bottom, so matching finds them
preferentially, and the picked tilt angles pile up near 90 degrees. Since the
wedge is fixed along tomogram z for every subtomogram, a clustered orientation
distribution sends it to nearly the same place in every particle's own frame.
It then averages coherently instead of cancelling. This is what makes the
unweighted correlation in section 4 an actual problem rather than a known
approximation, and it is why the shuffled-angle control loses its power on
exactly the data that needs it most.

The cheap first slice is the first two items: splitting by object and
deduplicating. Both are geometry over the existing picks and need no new
external tool.

The larger version is to derive positions from fitted geometry instead of only
from correlation peaks. ArtiaX already offers the manual form of this — Fit
Geometric Model (Sphere, Surface, Boundary) and Generate New Models → gen on
surface — which fits spheres to virions and generates positions on the surface
with normals that are correct by construction. Those orientations are an
independent measurement to compare the matching against, and the comparison is
the honest way to tell a real lattice average from reference bias. Doing it in
the pipeline means a `GEOMETRIC_MODEL` artifact kind (sphere or surface fitted
to a segmentation), positions generated on it, and picks carrying a reference
to the object they came from — which is also what the half-set split above
needs.

## 10. Re-extract and re-refine without repeating the template match

A scheduling fact that bears on this, found by trying to be clever about it:
**pytom cannot be imported without a CUDA runtime, whatever the tool does.**
`pytom_create_template.py` merely resamples a map, but its import reaches
`matching.TemplateMatchingGPU` and then `voltools`, which probes devices at
import time; on a CPU node that raises `cudaErrorInsufficientDriver` before any
work starts. So the schedulable boundary is pytom rather than the shape of the
computation: `template.py`, `picking.py` and `subtomogram.py` need a GPU
allocation, while `export-picks` and `extract-particles` import nothing from
pytom and can run anywhere. Worth knowing before writing resource profiles for
the controller, and an argument for this section: if the match cannot be
re-entered cheaply, every downstream re-try drags a GPU allocation with it.

The GPU template match is by far the most expensive step and the one most often
*not* what needs redoing. Changing the box size, the score threshold, the
extraction diameter, or the number of particles are all cheap operations on a
job that already ran, and pytom-match-pick supports exactly that: extraction
reads the job JSON and the score and angle volumes the match wrote.

Our `pick-particles` re-runs the match every time, because the artifact it
registers is the extracted picks rather than the match itself. The work is to
register the match's own outputs — job JSON, score volume, angle volume — as an
artifact, and to make extraction a separate step consuming it. This also makes
the extraction options genuinely explorable, which matters because the trap
recorded elsewhere is real: matching's `--particle-diameter` sets the Crowther
angular sampling, extraction's sets peak separation as diameter/2, and they are
not the same number for the same reason.

## 11. The patch margin is in the wrong unit

`generate-fiducial-seed --patch-margin-px` is in **binned** pixels of the
tracking stack, so it is silently coupled to `--pre-binning`. Patch size is
derived as

```text
usable   = binned_width - 2 * margin
patch_px = usable / (1 + (count - 1) * (1 - overlap))
```

At 4096 px and `pre-binning 8` a 100 px margin leaves 312 of 512 px, which is
defensible. The recipe in the trip notes for weak patch signal is to bin harder
for contrast — and at `pre-binning 16` that same 100 px eats 200 of 256, at which
point the sizing refuses because the patch would fall under its 32 px floor.
Raising binning therefore means scaling the margin down by the same factor to
keep the geometry, which nobody will remember.

The unit is the defect. A margin exists because the specimen area usable in every
tilt is narrower than one image — that is geometry in specimen coordinates, not in
pixels of an arbitrary tracking binning. The convention elsewhere in the codebase
is physical or unbinned: `--fiducial-diameter-nm`,
`recommended_unbinned_thickness_px`, `residual_mean_unbinned_px`. This option is
the one that falls outside it.

The fix is a binning-independent unit converted inside the seed backend. A
**fraction of the field** is preferable to unbinned pixels: it survives a change
of detector as well as of binning, and "discard a fifth of each edge" is easier to
reason about than 800 pixels.

Two related points worth keeping:

- **The default 100 is not derived.** It could be — edge smearing follows from
  lamella thickness and maximum tilt, and the tilt angles are already in the
  manifest — but two plausible derivations gave materially different numbers, one
  below 100 and one nearly double. So it needs someone who holds the geometry, or
  a measurement, rather than an invented formula. Until then, a poor patch
  coverage confined to the X edges should be diagnosed by looking at which tilt
  the edge leaves the field, not by nudging 100.
- **The margin is applied to both axes equally**, while foreshortening acts
  perpendicular to the tilt axis. With a tilt axis near vertical, as in
  EMPIAR-11538 at 83.6 degrees, X and Y are not equivalent and one value for both
  is over-conservative in Y.

Not urgent: until patch tracking has run against real IMOD it is unknown whether
the margin is a knob that gets turned at all. Do this once it works.

## 12. Derive the reconstruction thickness from the data

`reconstruct-tomogram` takes its thickness from tiltalign's two-surface analysis
when `--thickness` is not given. That derivation is degenerate for patch
tracking, and it does not say so. Measured on two datasets:

```text
TS_01   (gold beads)  recommended_unbinned_thickness_px  2366  = 160 nm
TS_0005 (patches)     recommended_unbinned_thickness_px    78  = 9.5 nm
```

Both numbers are internally correct. The gold beads sit on the two faces of the
specimen, so their spread in Z *is* the thickness. Patches are two-dimensional
features correlated across tilts, and their solved Z positions are nearly
coplanar, so there is no spread to measure. A FIB lamella is 100 to 250 nm;
9.5 nm is wrong by more than an order of magnitude, and the run reports
`warning` for an unrelated residual rather than for this.

**Only the thickness is degenerate, not the whole surface analysis.** The same
analysis also yields the tilt-angle offset and the X-axis tilt, and those come
from the *orientation* of the plane the fiducials lie in rather than from their
spread — which coplanar points determine well, arguably better than scattered
ones. One property of the point distribution therefore acts in opposite
directions, and the two outputs must be judged separately.

Measured, on TS_0005: the surface analysis applied a tilt-angle offset of
-7.31 degrees, and refitting the intensity model below against the nominal
angles instead of the solved ones drops its R^2 from 0.998 to 0.642. The offset
is real — a FIB lamella carries a milling pretilt — and it was found correctly
from a distribution that could say nothing about thickness.

There is a measurement that does not depend on the fiducials. For a slab of
thickness *t* the transmitted intensity follows Beer-Lambert with an effective
path *t / cos θ*:

```text
ln I(theta) = ln I0 - (t / lambda) * (1 / cos theta)
```

so a straight-line fit of log mean intensity against `1 / cos theta` has slope
`-t/lambda`. Everything it needs — an aligned stack and its tilt angles —
already exists before reconstruction. Checked against synthetic data with a
known answer: 180 nm in, 180.3 nm out, R^2 0.9998.

Source priority:

1. `--thickness` given explicitly — use it, derive nothing;
2. otherwise the Beer-Lambert fit, when its R^2 clears a threshold — use it, and
   report `warning` because the pipeline chose the number, as
   `prepare-template` already does for a derived mask radius;
3. otherwise refuse and require `--thickness`.

The third line is the point. Today's failure mode is a confidently wrong 78 px;
"this could not be measured, pass it explicitly" is strictly better than a guess
that looks like a measurement.

Three details, without which this becomes a new silent error:

- **The mean free path is a hardcoded physical constant.** Roughly 300 nm for
  vitreous ice at 300 kV, carrying about +-25% depending on composition, on
  voltage, and on whether elastic scattering is counted. So expose
  `--mean-free-path-nm`, record the value used, and report the interval rather
  than a single number.
- **Fit the right stack.** The model needs raw-ish intensities, and any
  normalisation breaks it. Phase flipping should leave a mean alone, but the
  derivation should not rest on that: let it look up the unlabelled pre-CTF
  aligned stack deliberately, rather than taking whatever reconstruction
  happens to consume.
- **Gate on the fit.** A stack that does not obey the model still returns a
  slope. Below the R^2 threshold, refuse rather than pass a number on.

One check comes free. Two independent estimators of the same quantity means
**their disagreement is itself a signal**: whether a fiducial Z distribution is
meaningful cannot be tested automatically, but a factor-of-twenty gap between
the two numbers can be, and that is exactly the case found by hand here. So
keep tiltalign's recommendation in QC for comparison, and stop using it as the
source.

Worth doing together with the defocus-hint gap in section 5. Both are
measurements that are properties of a tilt-series rather than of a step, both
want to be inspectable before anything commits to them, and both are currently
either unavailable or wrong for one of the two entry points.

## 13. Patch tracking inherited bead assumptions — audit for the rest

The patch backend was written by analogy to the bead one, and everywhere the
two differ *structurally* the bead assumption came along. Three instances
turned up in a single afternoon on EMPIAR-11538, which is why the remaining
work here is an audit rather than a list of fixes.

The shape is always the same: something the bead path produces on every run is
legitimately absent under patch tracking, and the code reads absence as failure
or reports it as zero.

**Fixed — the filled-in fiducial model.** `tiltalign -OutputFilledInModel`
writes a model with gaps interpolated, and it writes nothing when there are no
gaps. Bead tracking always loses beads at high tilt, so the file always
appeared. Patch tracking gives every patch a point on every view — coverage
1.0, 25 x 59 = 1475 points — so there was nothing to fill and no file, and fine
alignment failed on a missing required output. It is optional now, and the run
that first hit this also showed the subtlety: after outlier pruning creates
gaps, later rounds *do* write it, so the file is neither always present nor
always absent.

**Fixed — the two-surface thickness.** See section 12: coplanar points make the
two-plane fit degenerate, and the parser was reading the wrong one of
tiltalign's two labelled blocks. `-SurfacesToAnalyze` is a parameter now, the
block is chosen by its header, and QC warns when the split degenerates whatever
the setting.

**Fixed — the fiducial-seed QC reported bead semantics for a patch grid.** The
patch seed produces a grid descriptor, not a point model, so every field
`FiducialModelQc` has read zero: `num_fiducials 0`, `num_points 0`,
`coverage_fraction 0` — with status `pass`, three series in a row. For a bead
seed those numbers would be a catastrophe. There is a `PatchGridQc` now,
reporting what a grid actually has: patch counts, patch size in binned pixels
and in Angstrom, the margin, and what fraction of the field it leaves.

Its one warning is derived rather than chosen. At the largest tilt the specimen
foreshortens by cos(tilt) across the tilt axis, so that fraction of one image is
what every view still covers; a margin leaving less than that cuts more than the
geometry asks for. On EMPIAR-11538 the configured margin leaves 0.609 against a
foreshortening fraction of 0.439 — aggressive but defensible, and silent. A
margin of 160 binned pixels would leave 0.375 and warn. That reasoning used to
live in a comment in the lane config; now it checks itself.

**The audit is the work item.** Three found by tripping over them says nothing
about how many are left. Every stage between seeding and reconstruction should
be read once with the question "what does the bead path always produce here
that patch tracking legitimately might not", and the same question asked of the
frames versus assembled-stack entry points in section 5 — it is the same class
of defect seen from a different angle.

## 14. Exclude a view at fine alignment

Coarse alignment drops low-variance views on its own and records
`excluded_z_values`. Fine alignment has no equivalent: the options are
thresholds, robust fitting, and outlier pruning, and none of them removes a
view.

That gap has a concrete cost. On TS_0006 the initial residuals were mean 1.754
nm with sd 3.524 — sigma twice the mean, the signature of a heavy tail — and
the log showed two separate causes. One patch was bad across most views,
already crushed to weight 0.02 by robust fitting. And **view 59 was bad across
four different patches at full weight 1.0**, carrying the largest residuals in
the run. It is the extreme-tilt view, where a lamella is 2.3x thicker along the
beam and the contrast collapses; that view is expected to be bad, and the
honest remedy is to exclude it.

What is available instead is pruning its points, which is not the same thing:
the view stays in the stack with its geometry unconstrained, and with
`-MinFidsTotalAndEachSurface` it may simply refuse. Pruning did work here —
mean 0.812 nm, sd 0.672, tail gone — but it worked by accident of the fraction
being large enough, not because it was the right instrument.

Note while implementing: the default pruning fraction of 0.02 is also a
bead-era number. With a 5x5 grid over 59 views there are 1475 points and one
failed patch is 59 of them, so a single bad patch exceeds the limit. Deriving
it as "up to two whole patches" gives `2 / num_patches`, which is 0.08 for a
5x5 grid — see the comment in `configs/lanes/ribo-bin8.toml`.

## 15. Derive the matching low-pass from the alignment residual

`pick-particles` exposes `--pytom-low-pass` and passes it through correctly,
but nothing derives it, so a lane config that omits it matches across the whole
band out to Nyquist. That is what produced 2 candidates from a requested 2000
on TS_0005: not an error, a flat score volume with the extraction cut-off flush
against its maximum.

The value should not be recalled from practice. One bound is computable from a
number the pipeline already measures. An rms misalignment of delta blurs the
reconstruction, attenuating spatial frequency k by `exp(-2 pi^2 delta^2 k^2)`,
which reaches 1/e at `d = pi * delta * sqrt(2)`:

```text
TS_0005   fine-alignment residual 0.694 nm  ->  30.8 A
TS_0006   fine-alignment residual 0.812 nm  ->  36.1 A
```

It differs per series, which is the argument for deriving rather than
configuring it. Caveat to record alongside the number: this accounts for
alignment blur only, and dose, lamella thickness and the missing wedge all
degrade further, so it is an optimistic bound on resolution and therefore a
*lower* bound on the low-pass. Erring coarser is safe; finer is not.

**This is the third instance of the same shape**, and they are worth building
together rather than one at a time:

- section 12, the reconstruction thickness, derivable from the Beer-Lambert
  falloff of intensity with tilt;
- section 5, the defocus hint, already computed inside `estimate-ctf` but not
  available as an inspectable artifact for stack-imported data;
- this one, the matching low-pass, derivable from the fine-alignment residual.

Each is a number a person currently guesses, each is derivable from something
already measured, and each is silent when wrong — the thickness reports a
degenerate 9.5 nm, the hint lands wrong for every view at once, the low-pass
returns an empty pick list with `status: pass`. A shared shape suggests a
shared treatment: derive it, record the derivation and its assumption in QC,
gate on whatever makes the derivation invalid, and refuse rather than guess
when the gate fails.

## 16. Pair per-view metadata with views, not with positions

Done: dose weighting. `--dose-accumulation` is passed now, taken from
SerialEM's `PriorRecordDose` — the dose the specimen had already received when
that exposure was made. That is the right field rather than `ExposureDose`,
because a dose-symmetric series is *written* in tilt order and *acquired* in
another: on EMPIAR-11538 the view acquired first is ZValue 30 at +7.97 degrees,
thirty-first in the file, so accumulating `ExposureDose` down the file would
call the freshest views the most damaged and weight them accordingly. Whether
any weighting was applied is recorded on the picks artifact as
`per_tilt_weighting` and `dose_weighted`, because a match run without it
otherwise looks exactly like one with it.

Two things remain.

**Fixed — the dose now follows the views the tomogram was built from.** Only
the dose was affected, and finding that out cut the work in half: the aligned
stack already drops excluded views and records `included_z_values`, so the CTF
was fitted on the surviving views and the `.defocus` file matched by
construction. The dose came from the manifest, which lists everything
collected, so a series with a single excluded view refused outright and could
not be picked with weighting at all. Reconstruction records `included_z_values`
on the tomogram, and `_write_dose_accumulation` now selects by it, refusing
only when the tomogram names a z value the manifest does not have. Tomograms
written before the field existed return None and fall back to the count check.

Still worth doing: `_write_tilt_angles` reads the solved angles as a bare list
of numbers, so the angles themselves still carry no view identity. Nothing
depends on that today — the count check catches a mismatch — but it is the
reason two separate functions had to reconstruct which views survived from
somewhere else.

**A fallback for mdocs that do not record `PriorRecordDose`.** Older SerialEM
and other acquisition software write only `ExposureDose`. Acquisition order is
recoverable from `DateTime`, which the mdoc parser already keeps in `metadata`,
and accumulating in that order gives the same answer. Two reasons to guard it
rather than simply write it: it introduces a dependency on a date format, and
an order reconstructed wrongly weights the views inside out while completing
normally. Check it against what is already known — the accumulation must be
monotone in acquisition order, and its total must equal `ExposureDose` times
the view count — and refuse rather than guess when `DateTime` is missing or
does not parse.
