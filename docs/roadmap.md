# Roadmap

Planned work beyond the MVP slice in [mvp_plan.md](mvp_plan.md). Ordered roughly
by dependency, not strictly by priority.

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

## 2. Template preparation from EMDB

`pick-particles` currently requires `--pytom-template` and
`--pytom-template-mask` as already-prepared MRC files. Preparing them is a
manual, error-prone step that belongs in the pipeline.

- Fetch by accession: `https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-<n>/map/emd_<n>.map.gz`
  (verified to resolve). Cache locally and skip the download when the file is
  already present.
- Wrap `pytom_create_template.py` (resample to the tomogram voxel size, optional
  low-pass, box size, centering) and `pytom_create_mask.py` (radius, sigma) as a
  `prepare-template` stage producing `TEMPLATE` and `TEMPLATE_MASK` artifacts.
- Teach `pick-particles` to resolve those from the registry, keeping the explicit
  path options as overrides.

**Contrast inversion is the sharp edge here.** EMDB maps have bright (positive)
density; particles in a tomogram are dark. `pytom_create_template.py --invert`
multiplies the template by -1 to match, and its own help warns it is "not needed
if ctf with defocus is already applied". Newer pytom-match-pick also has CTF
options (`--ctf-correction`, `--flip-phase`, `--defocus`) that produce the right
contrast without `--invert`, and `--flip-phase` specifically matches a
phase-flipped tomogram — which is what this pipeline's `correct-ctf` produces.
Getting this wrong does not error, it silently finds nothing. So: expose the
choice explicitly, record it in the artifact parameters, and verify the
installed pytom version's actual flags rather than assuming.

## 3. Inspect picks in ChimeraX + ArtiaX

Checking picks visually currently means hand-converting Zarr to MRC and pairing
it with the right STAR file. Add an export that writes a matched tomogram MRC
plus particle STAR, and a ChimeraX `.cxc` script that opens both with the
correct pixel size, so verification is one command. Confirm ArtiaX's expected
STAR flavour against a real file before committing to a format.

## 4. Refinement and classification

`FftRefinementAveragingBackend` already covers DYNAMO-style 3D cropping plus 3D
alignment, behind the `SubtomogramAveragingBackend` protocol, so a RELION5-style
alternative can be added later without disturbing it. Outstanding work inside
that seam:

- Multi-class classification. `rlnClassNumber` is already carried per particle
  and averaging is structured as "average the particles assigned to class k",
  currently always called with a single class.
- Independent half-set refinement, so half-set FSC becomes a real resolution
  estimate rather than an optimistic one. Target design: keep the *initial*
  reference common, then run the alignment and reference-update loop fully
  independently per half.
- Missing-wedge and CTF weighting in the alignment correlation, deliberately
  absent from the baseline.

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

**Compare backends in parallel.** This is currently blocked by artifact
identity, and the blocker is already real rather than hypothetical. Some
artifact ids encode the backend and can coexist:

    {ts}:picks:{branch}:{backend}
    {ts}:particle_stack:{backend}
    {ts}:subtomogram_average:{backend}

but many do not, and re-running with a different backend or binning silently
replaces the previous entry instead of adding a second one:

    {ts}:alignment:coarse          {ts}:alignment:fine
    {ts}:fiducial_seed             {ts}:fiducial_model
    {ts}:aligned_tilt_stack:final  {ts}:tomogram:fine
    {ts}:corrected_projection:{z}

Concretely, the IMOD bead-tracking and patch-tracking backends write identical
ids, so their results cannot coexist in one registry — the two things most worth
comparing are the two that collide. `{ts}:aligned_tilt_stack:final` omitting
binning has already caused a silently replaced entry once.

Needed:

1. Artifact ids that carry the distinguishing parameters (backend, binning,
   variant label).
2. Registry lookups that can disambiguate. Ten call sites currently raise on
   "expected exactly one ...", which is what enforces the single-variant world.
3. A comparison report over variants, reusing the per-stage QC models that
   already exist.
