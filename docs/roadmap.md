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
five hundred. Observed since, and worse than this paragraph says: parallel
picking does not quietly lose entries, it brings the runs down — section 20.

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
   each other. Promoted by section 20: parallel picking crashes rather than only
   clobbering, so this gates step 4 instead of tidying it.
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

**The multi-class path applied no low-pass at all to its class references, and
now does** — at a limit held constant, since with several classes there is no
defined pairing between half 1's class *k* and half 2's class *k* to derive one
from. Joint refinement is not exempt from a reference imprinting its own noise on
the particles that built it.

Adding it broke an existing cone-search test, which had been earning its keep
without anyone noticing: the starting correlation was measured against the
UNFILTERED reference while the search is scored against the filtered one, so the
before-and-after compared two different questions and read as a refinement that
made things worse. The half-set path had the same fault and no test saw it. Both
now measure the start against what the search actually faces.

A second defect recorded here — that this path emits a resolution number it has
not earned — **was wrong, and is withdrawn.** `SubtomogramClassificationQc` emits
no resolution at all; its only resolution-shaped field is
`target_resolution_angstrom`, which is an input. The `gold_standard` flag stays
useful for the reader, but there was no number for it to qualify.

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

**Averaging picks at their template-matched orientations cannot test whether
the picks are real.** Measured, because it was about to be relied on: 100 pure
noise volumes, each oriented by a global search over 300 orientations against a
template — which is what template matching does — average to a map correlating
**+0.321** with that template. The same volumes at random orientations correlate
**+0.035**. Nearly tenfold, on nothing at all, with a grid far coarser than
pytom's.

So the obvious test — average the picks as matching left them and look for the
expected shape — is confounded, and confounded toward a false yes. It gets worse
with more particles, since the template-like part adds coherently. Refining
afterwards does not help: the orientations already carry the bias, and the
refinement search here is local, so it cannot leave the basin the template put
each particle in.

This also rules out reading a half-set FSC as evidence of the picks being real.
Both halves were oriented against the same template, so both averages resemble
it, so they correlate — independent refinement of the halves does nothing about
a bias that entered before the split. `--random-phase-correction` corrects the
score statistics used to place the cut-off; it does not touch orientations.

**A control has to break the link between the template and the orientations**,
not merely re-refine downstream of it. Three candidates were on the list:
matching with a mirrored or phase-randomised template, averaging the population
just below the cut-off, and boxes at random coordinates.

**Settled: boxes at random coordinates.** It was run on EMPIAR-11538 and it
answered the question the others cannot, so the choice is empirical rather than
argued.

Why it beats the other two. The shuffled-orientation control's failure mode is
already recorded in section 4 below: the shuffle only breaks what the
orientation *distribution* does not already fix, so a population sharing a tilt
keeps the geometry the swap was meant to destroy. A phase-randomised or mirrored
template breaks the template, but the picks' **positions** are still the ones
template matching chose — and that selection is itself enough to put
ribosome-sized central density in every box, which is exactly the confound that
made a radial-concentration measurement unreadable until this control existed.
Random coordinates break both links at once.

What it established, on 4000 picks across five series against 4000 random-position
boxes put through the identical chain with identical settings:

| | class 1 | class 2 | class 3 | class 4 | shuffle ceiling |
|---|---|---|---|---|---|
| real, seed-to-seed agreement | 25% | **36%** | 27% | **33%** | 31% |
| control | 27% | 26% | 27% | 30% | 30% |
| real, RMS inside/outside the particle radius | 2.32 | 2.43 | 2.29 | 2.55 | — |
| control, same measure | 1.00 | 1.04 | 0.99 | 1.00 | — |

So two independent statistics separate the real picks from nothing, and both are
properties of the pick positions rather than of the processing. The adjusted Rand
index tells the same story: +0.010 real against +0.001 for the control, on a
shuffled null of ±0.001. The control's radial ratio also landed on the
independently computed geometric null for this box and this rotation scheme
(1.001–1.006), which validates the measurement rather than only its result.

Two limitations, both real. It establishes that something reproducible and
spatially concentrated is there, **not what it is** — other large complexes,
membrane fragments and ice or film edges all qualify. And it is post-hoc: it
needs a finished real pick list to copy, so it cannot gate picking.

**A control's parameters are copied from the real list, never chosen.** This is
the transferable lesson and it cost most of a day. Every parameter picked by
reasoning was wrong: an exclusion distance set to the box size, on the sound
argument that two boxes closer than that are largely the same voxels, was past
the jamming limit of random sequential addition and placed 425 of 800 positions
— while also being *more* spread out than the list it controlled, since pytom
excludes only within half the particle diameter. Every parameter measured off
the real list was right: exclusion = its own minimum nearest-neighbour distance,
depths = a permutation of its own depths, count = its own count. A control step
should therefore have almost no knobs, and the fairness check in section 17 is
what makes that safe rather than merely convenient.

What this becomes as a pipeline step is section 17.

**Two things bit in one session for the same missing reason, and both are
fingerprints.** Step 2 of the order in
[database_design.md](database_design.md) — "split identity from naming: add
fingerprints alongside the semantic ids" — stopped being an abstraction today.

First: a particle stack was silently built from a different version of its picks.
Picking was re-run after its parameters were fixed and found 179 particles where
it had found 2; the extraction was not re-run, so the stack still held one
particle. The extraction QC recorded `picks_id`, and that id was unchanged,
because re-running picking overwrites in place. So the artifact chain said the
stack came from those picks and it was true of the name and false of the content.
Nothing in the registry could have revealed it. It surfaced only because an
average printed its per-series counts and one series contributed 1 of 179 — which
is to say, by luck and by a log line added an hour earlier.

Second: `--output-variant` exists at all. `--variant` selects inputs and labels
outputs with one string, so varying a parameter over fixed inputs either finds
nothing or overwrites what it is being compared against. The flag splits the two
jobs, and it is a workaround rather than a design: with fingerprints, two runs
differing in a mask radius get different identities from their own parameters, and
neither `--output-variant` nor `--variant` is needed to keep them apart.

So the fingerprint is not only for the index. It is what makes "this artifact was
built from that one" a checkable statement rather than a naming convention, and
what lets a parameter sweep exist without an operator inventing labels for it.
Both of today's failures were invisible in exactly the way a content hash is
designed to prevent.

**The score volume is the expensive artefact and it is thrown away.** Template
matching costs about 29 minutes of A100 per tilt-series and produces a
correlation volume; extracting candidates from it takes seconds. This pipeline
keeps the extracted candidates, the job JSON, the QC and the log, and discards
the volume they came from. So every change to a cut-off, a false-positive budget,
or a peak-exclusion distance costs a full re-match instead of seconds.

At bin8 that volume is roughly a gigabyte per series, so not keeping it at
fifty-series scale is right. During parameter tuning it is exactly backwards.
The shape this wants is a retention policy on the artifact rather than a
decision baked into the backend — `RetentionPolicy` already exists and the score
volume is the clearest case for a `WORKING` role that a later sweep can delete.

**Two models of mine were tested against the real data and both failed.** Worth
recording as measurements, since each was about to be built on.

The first: that the template mask being larger than the particle was costing
significance, worth about 1.23x. Measured by rebuilding the template with a mask
of 293 A instead of 333 A and re-matching one series: scores rose by a factor of
1.16 to 1.22, and **sigma rose by 1.22 as well**, so z_max went from 7.54 to
7.20 — slightly down. The normalised cross-correlation is already divided by the
mask, so shrinking it lifts the signal and the noise scale together. The mask is
not a lever on significance at all, and that term must come out of any
decomposition of the sigma gap.

The second, and this one cost more: that the excess of picks over the Gaussian
null was the real evidence for the picks being genuine. With the over-large mask
TS_0006 gave 192 picks against the ~100 the null predicts, an excess of +92. With
the mask matched to the particle it gives 73, a **deficit of -27**. So the excess
was being manufactured by the solvent shell inside the too-large mask correlating
with something outside the particle, and it disappears when the mask is corrected.

**What survives, measured on the 706-particle average**: at a threshold matched in
units of each map's own spread, 87% of the above-threshold volume sits in one
connected component, against 10% for an average of boxes cropped at peaks of
band-limited noise, and 100% for the ribosome template. So the average is not
noise. Its radial profile is a core plus a shell at 81-110 A rather than the
template's solid body out to 137 A, and that profile reproduces between the two
independent half-sets and survives with no filtering at all. Three mechanisms
were guessed for it and all three were refuted by measurement.

**A real defect found along the way.** pytom excludes further peaks within half
the diameter it is given, which this repository's own docstring states. So
`--pytom-particle-diameter 290` permits picks 145 A apart; the measured minimum
across the five series is 147 A and 19% of picks have a neighbour closer than
290 A — closer than two ribosome centres can physically be. The fix is the
already-existing `--pytom-extract-particle-diameter`, set to twice the diameter.
Testing it needs a re-match, which is the first paragraph above.

**Lane B: the pipeline is alignment-limited, and that is the finding.** Five
tilt-series of EMPIAR-11538 ran end to end and template matching found no
ribosome detections: in pytom's own sigma the best pick reaches 7.2 against a
one-false-positive floor of 7.24, where pytom's own benchmark reaches ~20 for
80S in thin ice. The cause is quantitative and it is upstream of everything this
project can tune.

The fine-alignment residual is 0.69 to 0.92 nm. An rms misalignment attenuates
frequency `k` by `exp(-2 pi^2 delta^2 k^2)`, so at 35 A only 35% of the signal
survives, at 25 A 12%, at 20 A 4%. Matching at 35 A is therefore correct and
matching finer would add noise. Significance grows as the square root of the
independent Fourier samples used and that count as `d^-3`, so a 20 A band would
multiply sigma by 2.3 — turning 7.2 into 16.7, essentially the benchmark. Getting
there needs a residual of 0.45 nm against the 0.79 median: a 1.8x improvement in
the tilt-series alignment, which is the only lever.

**Everything else was checked and cleared, and the list is the useful part.** The
Zarr canonicalisation is voxel-exact against the IMOD `.rec` — verified on eight
random voxels, `.rec` (928, 296, 960) as `[y, z, x]` to Zarr (296, 928, 960) as
`[z, y, x]`. The tilt angles fed to pytom match the reconstruction, 59 of 59 with
none dropped, so the missing wedge is modelled from the right set. The template's
density extends 294/313/303 A, right for an 80S, and its contrast is inverted as
the tomogram needs. The template mask and the peak-exclusion distance were both
wrong and are both fixed. CTF is ruled out by arithmetic: the first zero sits at
14 to 31 A across the fitted defocus range, so a 35 A band is inside the first
pass band. Binning is ruled out too — bin8's Nyquist is 19.6 A against a 36 A
ceiling, nearly a factor of two of headroom already unused, so finer sampling
buys frequencies that hold nothing.

**A supervisor's read, worth recording because it beat every metric here.** Shown
the 706-particle average he said it looks like a ribosome shell and something is
not right, and gave five checks. Three were already possible, two were not and
were built (`--class-number`, and a particle gallery). One of his points
reframed the whole question: significance tests ask "is any single pick a
detection" and cannot see a minority, while classification asks "is there a
subpopulation" — which is the better question for this data and the one still
unanswered.

**What still has no answer.** Whether a minority of real ribosomes hides among
mostly-noise picks, which needs classification and a look at the classes. And
what the published work on this dataset did differently — most likely
per-particle per-tilt geometry refinement, which is a different architecture
rather than a missing step, but that is inference and not established.

**A 2.9% pixel-size discrepancy with the published work, unresolved and cheap
to bound.** The mdoc records `PixelSpacing = 1.223` and the stack header agrees,
which is where this pipeline gets its number; Xing et al. state 1.188 for the
same data. Two instrument records against one figure in a paper, so metadata
reading is not the fault — most likely a calibrated value against SerialEM's
calibration table.

It is not free either. The template is resampled to the voxel size this pipeline
attributes to the tomogram, so if 1.188 is the true scale a 290 A ribosome spans
30.5 voxels in the data and 29.6 in the template. That displaces features at the
particle's edge by 4.3 A, which combines with the 8.1 A alignment residual in
quadrature to 9.2 A — a 13% worse effective error, a band of 41 A instead of 36,
and **a factor of 1.20 in sigma**. Real, and not the factor of three that is
missing.

Settling it does not need to know which figure is right: rescale the template and
see whether matching improves, one series, 29 minutes. Worth doing after the
decisive test rather than instead of it. If 1.188 turns out to be correct then
every derived number here shifts by 3% — voxel size, thickness, the residual in
nm, the 36 A ceiling — which is a systematic worth knowing about even though
nothing in the argument turns on it.

**The particle loop is serial and threads do not help it.** Measured on the
classification search at box 48 with four references and a 30-degree cone: 161 ms
per particle on one torch thread, 158 on two, 168 on four, 187 on eight. Eight
threads are 14% *slower* than one. Each call resamples a 48-cubed volume — about
110 thousand voxels — which is too small an operation for the thread pool to pay
for itself, and torch parallelises within an operation rather than across the
many small ones this loop issues.

What would parallelise perfectly is the loop itself: `for record in records` in
both averaging backends treats every particle independently, reading shared
references and writing its own result, so there are no races to resolve. Four
thousand particles at five iterations take about two hours serially and would
take a quarter of an hour across sixteen cores.

The one real care needed is that `records[record.index]` is reassigned inside the
loop, so a worker must return its record rather than mutate the shared list —
which is a small restructuring, not a redesign. Until then `--cpus-per-task` above
about four is wasted on these jobs, and the practical speedup is to run
independent invocations concurrently: two classification seeds are two jobs and
exactly twice as fast.

Worth doing before any classification at the scale the published work used —
286,400 candidates rather than our 4,000 — where two hours becomes a week.

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

## 3. Inspect picks in ChimeraX + ArtiaX, and in 3dmod — done

`visualize-picks` writes a matched tomogram MRC plus picks STAR and a `.cxc`
script, and converts the same picks into an IMOD model with `point2model`, so
checking a result is one command for either viewer. Both halves share the one
tomogram conversion, which is the largest file involved.

Two viewers because they answer different questions. ChimeraX with ArtiaX shows
the whole set at once, each pick drawn as the template at the orientation found
for it, over the segmentation surface: that is where a systematically wrong
orientation, or a lattice picked as a cloud, shows up. 3dmod shows one slice
with the picks drawn on it, which is the only way to answer whether a particular
pick sits on density. Neither substitutes for the other, and the earlier
`export-chimerax` only offered the first.

Both viewers disagree with the pipeline about what a coordinate means, and both
disagreements are silent — nothing errors, the picks are simply in the wrong
places.

ArtiaX registers **two** readers for `.star` — "RELION STAR file" (RELION 3.1
conventions, `rlnCoordinateX/Y/Z` in pixels, which is what pytom-match-pick
writes) and "RELION5 STAR file" (centered coordinates in Angstrom). Both claim
the suffix and both answer to the nickname `star`, so the generated script pins
`format relion` explicitly rather than letting ChimeraX choose.

IMOD model coordinates run from 0 to NX in X and Y but from **-0.5 to NZ - 0.5
in Z**, so the centre of voxel `i` is at `i + 0.5` in X and Y and at `i` in Z,
while a pick is a voxel index in all three. The conversion therefore adds half a
voxel in X and Y and nothing in Z. Stated by IMOD's own binary model spec and
checked against the library rather than the prose: painting the written model
onto the tomogram with `imodmop` puts the paint on the voxels the picks name,
and dropping the offset moves every sphere half a voxel in X and Y while leaving
Z alone — which is what makes it worth a line here, since half a voxel is
invisible on a screenshot.

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
- CTF weighting in the alignment correlation. Still unweighted, and a per-particle
  CTF weight needs a defocus per particle that this pipeline does not carry yet;
  `estimate-ctf` fits per tilt-series.
- The per-voxel count of real real-space contributions. `_recompute_references`
  still divides by the member count although a rotated box contributes zeros
  where the rotation reached outside it. Measured on this geometry — a 300 A
  particle in a box 48 at 10 A/voxel — the real-contribution fraction is exactly
  1.0000 out to r = 22 voxels while the density lives inside r = 15, so this
  touches only the corners and cannot move a resolution. It makes the written
  map honest at the periphery, and it is one extra `grid_sample` per particle.
- Combining wedge weighting with an alignment mask. Refused rather than
  approximated today: one restricts the Fourier support and the other weights
  real space, and applying both is not their exact combination.
- Symmetry.

**Missing-wedge weighting in the alignment correlation is done for
`cone-search`** — `average-particles --missing-wedge` — and the reason given here
for tolerating its absence **was wrong, so it is withdrawn rather than edited.**
It read: "tolerable while orientations are spread, because each particle's wedge
then points somewhere different in the reference frame and they cancel." That is
the exact case that was then measured, uniformly random orientations, and it does
not hold. The wedges cancel in the *average*; they do not cancel in the *score*,
and the score is what moves the poses.

What was measured, with the control that makes it a measurement. Seeded with
exactly correct orientations and no noise, the whole-box score moved 100% of
particles by a mean of 5.90 degrees and dropped the average's agreement with the
known phantom from 0.9717 to 0.9112; the identical loop on the identical
particles with the wedge removed moved nothing (0.00 degrees, 0.9883 unchanged).
The loss is high-frequency-first — agreement at the 32.6 A shell fell 0.989 to
0.946 while the 456-114 A shells lost 0.004-0.008 — which is why a refined map
looks smoother than the one it started from. The mechanism is visible directly:
the orientation tensor of the lab tilt axis expressed in the reference frame
climbed 0.496 to 0.568 to 0.615 to 0.638 over three iterations with the wedge
(0.333 is isotropic) and only to 0.511 without, so the refinement was
co-aligning wedges instead of structure.

Two things that were tried first and did not fix it, both worth keeping so they
are not tried again: a soft alignment mask (8.28 vs 6.12 degrees mean error at
SNR 0.1) and a finer reference low-pass (identical to three decimals at 19 A).
And one sharpening of the mechanism, because it changes what else to fix: against
a clean unwedged reference the whole-box and constrained scores behave
identically (2.21 vs 2.21 degrees, 80% of poses held either way). The bias needs
the reference to be a wedge-limited average, which is why Fourier compensation
above attacks the same defect from the other side.

**How much the fix buys depends strongly on the tilt range, and the numbers above
are for +-60 degrees.** Measured at box 24 with 40 particles and two seeds,
seeded at the true poses with no noise, counting poses held exactly:

| range | reference | whole-box | wedge-constrained |
|---|---|---|---|
| +-60 (70% of Fourier space measured) | plain mean | 9/40 | 22-28/40 |
| +-60 | Fourier-compensated | 12/40 | 31/40 |
| +-45 (48% measured) | plain mean | 2-3/40 | 10/40 |
| +-45 | Fourier-compensated | 1/40 | 14-15/40 |

So the constraint helps at both ranges — 3 to 5 times more poses held at +-45 —
but the absolute level at +-45 is about a quarter against 55-70% at +-60, and
compensation adds roughly half again on top. At the narrower range compensation
stops being a refinement of the fix and becomes part of it. Sample size matters
here: at 8 particles the same measurement put +-45 below the WIDER +-30 wedge,
which is scatter and not a property of the range, so do not read small-N runs of
this comparison.

**Fourier compensation is built** — `average-particles --missing-wedge
--fourier-compensation` — and is the "compensated" row above: each reference is
`sum(W_i * F_i) / max(sum(W_i), floor)` over the particles' own wedges, with the
floor at 0.25 of the contributing particles and the realised coverage reported in
QC. Default off, like the wedge score.

**What the wedge score did on real CZII data, and the line that now governs
this section.** At the CZII tilt range of +-45 degrees, on 223 GT-matched
ribosome picks, the map-to-template FSC crossing went 45.05 A (whole box) to
41.59 A (constrained) against 28.46 A for the fixed initial poses — the
constrained curve is above the unweighted one at every one of the 20 informative
shells, recovering a median 24% of the lost agreement, but the recovery decays
from 68% at the coarsest shells to 2-6% below 33 A. Refinement therefore still
makes this map worse than not refining at all. A stopping criterion was agreed
before the compensated control was run: the compensated map must cross finer than
28.46 A, or this line closes.

**It was not met, and the line is closed.** Compensation helped, and by more
than the score did — 41.59 to 36.04 A, two Fourier shells against one, closing
30% of the gap the wedge arm still had, and above the wedge curve at every shell
finer than 67 A. Poses moved less (9.95 against 11.08 degrees mean) and drifted
less (1.68 against 1.80 px). The coverage it divided by was far from flat —
minimum 0.350, median 0.502, nothing on the floor — so the negative result is
not "there was nothing to compensate". It still lands 7.6 A short of not
refining at all.

Do not sweep the coverage floor, the tilt range, the axis, the low-pass or the
iteration count against this. The arithmetic says why: by the Crowther criterion
a 300 A particle needs 5.4 degrees of pose accuracy to support 28.46 A, and the
mean pose CHANGE the search introduces is 9.95 degrees — error, since the map
gets worse for it. The search is running about twice as coarse as the resolution
it would have to beat, and three separate measurements point the same way: the
fraction of particles left where they started is 5.8% against 4.2% for a blind
draw from the 24-candidate grid, the shift drift is anisotropic along the beam
axis in both arms (2.3x and 2.7x the in-plane spread), and the schedule offers a
single candidate after iteration 2. That is an information budget, not a list of
unfinished features.

What remains is a choice of a different line rather than a continuation of this
one: a finer voxel spacing, which is this benchmark's own parameter rather than a
property of the data (`fetch_czii.py --voxel-spacing`, default 10.012, and the
raw tilt-series spacing is recorded alongside); more than 223 particles, noting
that `gt-initial`'s half-sets never even crossed 0.143 so none of these maps has
an independent resolution; or per-tilt geometry refinement, which scores against
the 2D projections with per-tilt CTF and dose instead of correlating inside a
wedge-limited volume. The last is a hypothesis about what the published work on
similar data did, not an established fact.

Two other things the real run made measurable. The angular schedule is dead
after iteration 2: with the Crowther clamp at 6.68 degrees for a 300 A particle
at 35 A, the per-iteration candidate counts are 24, 24, 1, 1, 1, so three of
five iterations can only move shifts.

**Fixing that schedule was tried and is measured to make things worse, so do not
try it again.** `cone_offsets` places rings at integer multiples of the sampling,
which both empties the cap when the range falls below one step and never reaches
the requested cap; rounding the ring count instead, never letting it fall below
one, and spreading the rings to land on the cap gives a live schedule at every
iteration — 48, 27, 5, 3, 2 candidates. It also raises the total from 51 to 85,
and on this repository's own regression test the constrained score then held 2 of
8 exactly-correct poses instead of 6 of 8. That is the order-statistic effect
measured at the start of this work — 24 candidates held the truth in 62% of
trials against 19% at 200 — reappearing: a search whose scoring differences are
smaller than its noise gets worse the more candidates it is offered. A denser
cone is not a correctness fix here, it is a second way of losing poses, and it
would have to be paired with a move-acceptance margin to be worth anything.

One thing that does NOT need to change with the voxel spacing: the Crowther clamp
depends on the particle diameter and the target resolution, not on the grid, so a
finer reconstruction leaves the angular schedule identical. The information a
finer grid adds is frequencies in the particle and the reference, not orientation
sampling. And the shift drift is
anisotropic along the beam axis in BOTH arms — per-axis standard deviations
0.643/0.852/1.716 unweighted and 0.664/0.770/1.906 constrained, i.e. 2.3x and
2.7x the in-plane spread — reaching 93 A at the worst particle, with nothing
bounding it. The constraint loosens it slightly, which is expected: restricting
the score to the measured support leaves even less to say about the one direction
the wedge does not measure. An anisotropic shift bound is the companion this
change needs.

One normalisation alternative was tested and rejected: dropping the
reference-energy term from the denominator, or replacing it with a
model-energy penalty, is worse at every wedge size tried (+-60, +-45, +-30). The
implemented form — normalising both operands over the shared support — was the
best of the three, reaching 8/8 against an oracle reference at +-60.

The default is still the unweighted score, so that runs already on disk stay
comparable; `SubtomogramClassificationQc` records which score was used and the
report carries a warning whenever it was the unweighted one.

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
allocation, while `visualize-picks` and `extract-particles` import nothing from
pytom and can run anywhere (`visualize-picks` does need IMOD on the node, for
`point2model`). Worth knowing before writing resource profiles for
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

## 17. A decoy control as a step, not a script

`scripts/decoy_picks.py` settled which control to use (section on averaging
above). It should become a first-class step, because the question it answers —
*would this result have appeared with no particles in the boxes?* — is asked of
every subtomogram average, not only of a pipeline under suspicion. It is the
same category as gold-standard FSC, which is mandatory precisely because people
got resolution from noise.

The argument that it is general rather than debugging scaffolding is that
nothing else answered it. Adjusted Rand index, per-class agreement, radial
concentration, correlation before and after refinement, and box density all
came back equally consistent with "weak but real" and with "nothing, processed".
The control separated them in fourteen minutes of one A100.

**`cryoet decoy-picks`** takes a real picks artifact and registers a sibling
whose positions are random and whose everything else is copied — orientations
row by row so the refinement starts from the same distribution, scores so any
score-dependent behaviour is identical, depths as a permutation of the real
depths so no estimate of where the specimen sits has to be trusted, count and
exclusion measured off the real list per the principle above. It registers as
`picking_method: random_positions` with `mirrors_picks_id`, storage role
temporary and retention recompute: the whole control is a seed plus a source
list, so keeping its particle stacks past the comparison stores a gigabyte a
run that one command regenerates.

**The fairness check is part of the step, not advice.** A control that samples
different material is worse than no control, because it still looks like an
answer. Comparing the density content of real and control boxes is what makes
it credible — measured at 0.986–0.999 with equal medians across five series —
and it belongs in QC with a status, so that an unfair control refuses to be
used as one. Note which tail to read: box standard deviation saturates, since a
48-voxel box estimates the local value closely and at bin8 that is set by
reconstruction noise rather than by biology, so a particle moves it less than
boxes differ from each other. The left tail is what carries information — real
picks avoid the emptiest boxes slightly more than random positions do, which is
matching preferring material over vacuum.

**Record that no control was run, rather than being silent about it.** The
pattern is already here in `gold_standard: bool = False` and
`resolution_is_interpretable`: a field that says a check did not happen, so
that its absence cannot read as its success. `decoy_controlled: bool = False`
on the classification QC, and the seed-to-seed comparison recorded against the
control's when one exists.

Not proposed as a default. Running a decoy for every classification triples the
averaging branch — real times two seeds, control times two seeds — and that is
the user's call, not a default worth asserting. Cheap, available, and recorded
when run is the whole of it.

Two smaller items this produced, worth not losing:

- `scripts/radial_profile.py` takes the box boundary as `box / 2`, but an even
  box centres at `(box - 1) / 2`, so the true inscribed radius is 23.5 for a
  48 box and 22% of the annulus it measures sits inside the attenuated zone.
  Worth about half a percent on the null, so it changed no conclusion here.
  Its docstring also claims the rotation artefact reaches "roughly a third of
  the core value at the corners"; the corners are about 2%, and a third is
  reached near the edge-midpoint radius of 33. Both are one-line fixes.
- A step that runs for minutes to hours prints nothing until it finishes: there
  are no `typer.echo` or `print` calls in `backends/subtomogram.py` at all. A
  silent step cannot be told from a hung one, which cost a wasted allocation
  here, and there are no checkpoints either, so a job that hits its wall clock
  loses everything.

## 18. A step that runs for hours must be able to say how far it got

The question to answer is "how much longer?", asked of an agent rather than of a
log. That framing settles most of the design, and it rules out the obvious
answer.

**Not `print`.** Standard output goes to a SLURM log whose name carries a job id
the agent was never told, and a step run interactively writes somewhere else
again. A progress record has to live at a path that can be found without being
told: `progress.jsonl` beside `step_timings.jsonl` at the project root, appended
as whole lines so a killed job still leaves valid content. Appended rather than
overwritten for the same reason — the moment you most want to know how far
something got is when it died.

**Backends stay printless.** There are zero `print` or `typer.echo` calls in
`backends/`, and that is layering rather than an oversight: a backend returns
artifacts and the CLI echoes them. So the mechanism is a callback on
`BackendContext` — alongside `output_dir`, `device` and `parameters` — that the
CLI wires to the file and to a throttled human line. A backend that ignores it
keeps working, which is what makes it addable one step at a time.

**The unit must be the atom of work, not the loop variable.** This is the part
that is not obvious and that today's runs taught. Counting particles would give
a wrong estimate for a cone-search run, because the cost per particle RISES
across iterations: `refine_factor` halves the cone range and the step together,
so the ring count in `cone_offsets` stays put while each ring's circumference is
sampled more finely. For the template-null run that is 2650, 3200 and 3325
orientations per particle over three iterations. A "particles done" counter would
read as a steady slowdown and its extrapolation would be wrong by a third. The
atom is the orientation evaluation.

**The total has to be declared before the loop, and here it can be.** "Iteration
3 of ?" answers nothing. `cone_offsets` and `_symmetric_offsets` are pure
functions of the parameters, so the exact total — 9175 per particle, 36.7
million for 4000 particles — is computable in three lines before any work
starts. It was in fact computed by hand, after the fact, to find out whether a
running job was stuck.

**Throttle by time, not by iteration**, so a fast loop does not spend itself on
I/O, and so a slow one still reports.

### The half that makes the answer good rather than merely available

Record the measured throughput in QC: units, elapsed, and the rate. Then "how
long will this take" is answerable BEFORE starting, from what a comparable run
actually did rather than from arithmetic about the inputs.

That is worth stating because the alternative was tried twice in one day and
failed twice. The same template-null run was estimated at three hours and then
at 7.2 hours, and the GPU memory at 5.3 GB against a measured 27983 MiB. Both
first figures were derived from the input sizes; both were wrong by between two
and five times, because **the width of the orientation search sets the cost, not
the size of the data**. The correction was only possible because one comparable
run had been timed — a classification of the same 4000 particles at 296
orientations each, which took 14 minutes on an A100 and gave the 31x ratio.

`step_timings.jsonl` already records per-step wall time and disk delta, so half
of this exists. What it cannot do is give a rate, because it records neither the
unit nor how many of them there were. Adding those two fields to what it already
writes is most of the work.

### Adjacent, and worse

A step that reports progress still loses everything at the wall clock: nothing
is written until a run finishes and there are no checkpoints, so a 7-hour job
that is stopped at 8 hours has produced nothing. Progress reporting makes that
visible without fixing it. Writing the reference and the per-particle metadata
at the end of every iteration would fix it, at the cost of one map and one STAR
per iteration, and would also make a run resumable — which matters more here
than the disk does.

## 19. Membrane-relative geometry as an identity test — closed, with the reason

Rejected rather than deferred, so that it is not proposed again.

When every shape-based statistic in lane B turned out to be printed by the
orientation search, positions were the only thing left that owes nothing to the
template. The obvious test: a translating ER-bound ribosome sits with its exit
tunnel at the translocon, so its centre is about one ribosome radius from the
membrane — 100 to 180 A. A real population should show a bump there on top of a
broad background, and random-position controls, which already exist, cannot
produce one. `scripts/membrane_distance.py` implements it.

**It will not be run, because its negative result would be uninterpretable.**
The test's power depends on a substantial membrane-bound subpopulation existing.
On these FIB-milled HEK293T lamellae few membranes are visible, and if most
ribosomes here are free cytosolic then "no bump" follows whether or not the
picks are ribosomes at all. A measurement whose negative outcome says nothing
about the question is not worth the GPU time, however cheap.

Two facts worth separating, because they were conflated when this was proposed:
MemBrain-seg was never run on lane B's tomograms — there is no
`segment-membranes` step in `scripts/run_series.py` and no segmentation section
in `configs/lanes/ribo-bin8.toml`, and picking took its segmentation as
optional (`_optional_segmentation_artifact`) and never received one. So this is
untried rather than failed. And separately, membranes are not obvious by eye in
these reconstructions — which is weak evidence about what MemBrain-seg would
find, since it is trained to find what raw slices hide, but it is what makes the
subpopulation prior low.

The script stays: it is correct, its refusals are in place, and on a specimen
with visible ER it is the right test. It is simply not a test for this one.

**What this left on this dataset, and what it found.** One live measurement: the
shell-resolved map-to-model FSC. The cone-search reference is low-passed at 60 A
and `filters.py` gives it a raised-cosine edge two shells wide, which for a 48
box at 9.784 A/voxel reaches exactly zero at 47.79 A — shells 10 and finer have
zero response, and the correlation numerator uses the FILTERED reference, so
orientation ranking is blind there and the printing mechanism cannot reach. A
band mean over 200-35 A averaged those shells together with six the search
fully controls, which is how the discriminating part came to be hidden.

Read out with `scripts/fsc_table.py --shells`, against the same template, real
against the random-position control:

| shells | A | mean difference | note |
|---|---|---|---|
| 10-15 | 47.0-31.3 | **+0.031** | search blind, alignment permits signal |
| 16-24 | 29.4-19.6 | **-0.0005** | search blind, alignment permits nothing |

The second row is the calibration and it came out at zero, sd 0.0224, 3 of 9
positive — which is what makes the first row readable: **+0.031 is 3.4 sigma**
against that scatter. Both band edges were fixed before the readout existed, one
from the filter arithmetic and one from the alignment residuals, so the band was
predicted rather than chosen. Narrowing to the four coarsest shells gives
+0.052 at 4.6 sigma and is the best of all twelve sliding windows in the blind
region, but that narrowing is post-hoc and 3.4 sigma is the number to quote.

**The structural argument is stronger than the sigma.** A printed excess would
have to STOP at 47.8 A, where the reference's response becomes zero. It does
not: it persists four shells further and dies at 31-36 A. The sliding window
shows the decay directly — 47.0-36.1 gives +0.052, 42.7-33.5 gives +0.041,
39.1-31.3 gives +0.020, then flat near +0.010. The excess ends where the DATA
runs out, not where the SEARCH runs out, and those are different frequencies.

So lane B has one result that survives its own null. Three things it is not:
one run per side, so 0.0224 is shell-to-shell scatter within a curve and not
run-to-run variation; agreement with THIS template rather than ribosome
identity, which a same-procedure run against an unrelated complex's map would
test; and small in absolute terms, consistent with a minority population among
mostly-junk picks — a reading that must not be propped up by fitting the
published 13.8%, which is a different pipeline's purity figure.

Beyond that the choice is between writing lane B up as a methodological result —
a pipeline whose controls caught 0.914 FSC and 2.277 radial concentration
produced from boxes containing no centred particle, and then found the one band
where neither could reach — and moving validation to a dataset with deposited
particle coordinates. EMPIAR-11538 holds only raw tilt-series and mdocs,
verified against the EMPIAR API, so it cannot close a validation lane by itself:
a published map is not ground truth for picking, coordinates are.

## 20. Picking several tilt-series in parallel crashes

Observed on the cluster and recorded as observed: launching `pick-particles`
concurrently for several tilt-series — one process per GPU, which is the whole
reason to ask for a multi-GPU node — brings the runs down. `pytom_gpu_ids` is a
per-run parameter, so one process per GPU is expressible and was the obvious way
to run; it is the concurrency that is new, not the placement.

**The first work item is to capture a failure, not to fix one.** What the report
does not carry is the traceback, how many processes were in flight, whether all
of them die or one survives, and whether the crash lands inside the match or in
the registry write at the end of it. Two processes on different GPUs against the
same project, with both stderrs kept, is minutes of GPU time and it separates
mechanisms that need different fixes: a crash inside pytom or the CUDA runtime is
another repository's problem, while a crash in the registry is step 3 of the
order of work above.

**What is verified in the code, offered as a candidate and not as a diagnosis.**
Every command loads the whole `ArtifactRegistry`, mutates it, and writes it back
through `path.write_text` — no lock, and no temp-file-and-rename, so the write is
not atomic and a concurrent reader can see a partial document. `load` goes
through `model_validate_json`, which raises on malformed JSON. `add` raises on a
duplicate id, and `_validate_parent_ids` raises when a parent is absent from the
document that was just loaded — which is what a lost write looks like to the next
process. So there are at least three routes from concurrent writers to an
exception rather than to silent loss, which is why the order of work above
understates this. Whether any of them is what actually fired here is unknown
until there is a traceback.

**Why this is not cosmetic.** The workaround is the serial shell loop, and at the
measured 29 minutes of A100 per series a fifty-series run is about 24 hours
serially against roughly 6 on four GPUs. Step 4 of the order of work is a
fifty-series run, so the crash sits directly in front of the milestone the whole
order was built to reach.

**The fix shape is already chosen** — per-series registries or per-run records
plus a merge, as step 3 records, which is also the file-first shape the database
will index. An atomic write plus a lock is the smaller intermediate step and
would be tempting to do first; it is worth doing anyway for the non-atomic write,
but it should not be sold as the fix for this until the traceback says the
registry is where the crash is.

## 21. Every average is rotated about a point half a voxel off its own centre

**Confirmed, not yet fixed, and deliberately so.** Fixing it moves the 10.012 Å
do-nothing baseline, and that baseline is currently the only fixed point the
voxel-control investigation is measured against. Fix it as its own change, with
its own control, once that investigation has a number.

Extraction puts the pick at local index `box_size // 2` —
`subtomogram.py:111` sets `half = box_size // 2` and `_crop_start` returns
`int(round(row.z_px)) - half`, so the pick lands at index `half`: 24 in a box of
48, 49 in a box of 98. The resampler rotates about `(size - 1) / 2`
(`rotations.py:147-149`): 23.5 and 48.5. Every particle's density is therefore
offset from the rotation centre by half a voxel on each axis before its pose is
applied, and the pose scatters that 0.866-voxel offset into a direction that
differs per particle.

Verified on real arrays through the pipeline's own code: a blob cropped by the
`_crop_start` rule sits at local index [49, 49, 49] in a 98 box, and a 180°
turn through `resample_volume_zyx` moves it to [48, 49, 48] — a full one-voxel
move that a correct centre would not produce.

**Size, and which way it cuts.** 0.5·√3 = 0.866 voxels, which is 8.67 Å at
10.012 Å/voxel and 4.32 Å at 4.99. It is in *every* subtomogram average this
pipeline has written, and it penalises the coarse grid twice as hard as the
fine one. Monte-Carlo over 600 random orientations puts the coherent signal
retention at 45 Å at 0.758 for the coarse arm against 0.918 for the fine one —
so the 28.46 Å the coarse do-nothing average reached is itself depressed by
roughly a factor 0.76 at that shell and would improve if this were fixed.

Odd boxes are unaffected: 109 gives 54 either way, which is why the template
side of the FSC has never shown it.

**Two smaller relatives in the same chain**, worth fixing in the same pass:

- `cli.py:4014` pads the smaller map to the larger with `before = pad // 2`,
  which cannot centre across a parity change. 48 → 54 (difference 6) lands
  exactly; 98 → 109 (difference 11) puts the average's centre at 53.5 against
  the template's 54.0. Measured cost through `compute_fsc_curve` on a noiseless
  matched pair: 0.947 at 45 Å, 1.000 with the half voxel put back. No integer
  padding can centre this, so the fix is to resample or to compare in a common
  box, not to round the other way.
- `subtomogram.py:161-163` writes `shift_x_px=0.0` into every extraction record
  while the crop was taken at `round(x_px)`. The sub-voxel part of the pick
  position is information extraction already has and throws away, worth ±0.5
  voxel of avoidable spread.

**How this was found.** Not by a test — by asking why a fixed-pose average on a
4.99 Å grid disagreed with its template past 60 Å when the same particles at
10.012 Å reached 28.46 Å. See `VOXEL_GT_CONTROL.md` section 7 in the
benchmarking repository. None of these three is the main cause there; they were
turned up while looking for it, and the search is what makes them measured
rather than suspected.
