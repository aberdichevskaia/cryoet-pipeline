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
- Symmetry.

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

## 6. Make external-tool logs uniform and addressable — do this soon

Every adapter that shells out already captures the right material: argv, stdout,
stderr, and the exit code, written by a private `_write_command_log`. The defect
is that none of it is addressable from outside the adapter that wrote it.

- **`_write_command_log` exists in twelve private copies**, one per backend
  module (`alignment`, `alignment_qc`, `ctf`, `fiducials`, `final_stack`,
  `fine_alignment`, `motion`, `picking`, `reconstruction`, `restoration`,
  `segmentation`, `template`). There is no shared module. The bodies have drifted:
  some take a single `CompletedProcess`, some take a list of them, one appends a
  rendered `<<stdin>>` block because CTFFIND is driven through stdin.
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

1. One shared runner for external processes, replacing the twelve copies. It
   should record argv, the resolved executable path, stdin when used, stdout,
   stderr, exit code, and wall time, and it should be the only place that decides
   the log file's format.
2. A typed list of log records per unit of work — program name, path, exit code —
   instead of ad-hoc keys in `parameters`. This lands on the `run` entity
   described in [database_design.md](database_design.md); until that exists it
   belongs in a typed model rather than in a free-form dict.

Both are plain hygiene and are worth doing independently of anything downstream.
They also unblock deterministic failure classification: once logs are uniform, a
`parse_failure(log, result) -> FailureSignature | None` can be added to the
backend protocols, where the adapter that knows a tool's dialect is also the
thing that recognizes its errors. That parser is a follow-on, not part of this
item.

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
