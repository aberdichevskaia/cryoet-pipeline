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
