# Backends

Every pipeline stage is a `Protocol` in
[`backends/protocols.py`](../src/cryoet_pipeline/backends/protocols.py) with one
or more concrete implementations, chosen per run with `--backend`. Backends for
the same stage produce the same artifact kind, so they are interchangeable and
can be compared against each other.

Names follow one rule: **a backend named after an external tool is an adapter to
that tool**, and requires it to be installed. Backends with descriptive names
(`average`, `phase-corr`, `cubic-crop`, `fft-refinement`, `cone-search`) are
implemented in this repository and need nothing beyond its Python dependencies.

## Stage reference

| Command | Backend | Default | Needs | Notes |
| --- | --- | --- | --- | --- |
| `correct-motion`, `prepare-tilt-series` | `average` | ✓ | — | Sums frames without alignment. Debug baseline. |
| | `phase-corr` | | — | Global phase-correlation frame alignment, pure Python. |
| | `motioncor3` | | MotionCor3, CUDA | Patch-based; see [motioncor3_backend.md](motioncor3_backend.md). |
| `align-tilt-series` | `imod-xcorr` | ✓ | IMOD | `tiltxcorr` + `xftoxg` whole-frame coarse alignment. |
| `qc-coarse-alignment` | `imod` | ✓ | IMOD | Binned preview plus residual-shift metrics. |
| `generate-fiducial-seed` | `imod-autofidseed` | ✓ | IMOD | Finds gold beads automatically. |
| | `imod-patch-seed` | | IMOD | For specimens without beads: lays out a patch grid instead. |
| `track-fiducials` | `imod-beadtrack` | ✓ | IMOD | Tracks the bead seed model. |
| | `imod-patch-track` | | IMOD | Patch tracking; pair with `imod-patch-seed`. |
| `fine-align-tilt-series` | `imod-tiltalign` | ✓ | IMOD | Solves the alignment and positions the specimen. |
| `build-final-aligned-stack` | `imod-newstack` | ✓ | IMOD | Applies the solved transforms at a chosen binning. |
| `estimate-defocus-hint` | `ctffind5` | ✓ | CTFFIND5 | One near-zero-tilt image, to seed a full estimation. |
| `estimate-ctf` | `ctffind5` | ✓ | CTFFIND, IMOD | Two-pass, whole stack in one call. See below. |
| | `imod-ctfplotter` | | IMOD | Tilt-aware fitting in IMOD's own framework. |
| `correct-ctf` | `imod-ctfphaseflip` | ✓ | IMOD | Consumes any `CTF_MODEL`, whichever backend fitted it. |
| `reconstruct-tomogram` | `imod-tilt` | ✓ | IMOD | Weighted back-projection. |
| `restore-tomogram` | `isonet2` | ✓ | IsoNet2 | Needs a trained model; optional stage. |
| `segment-tomogram` | `membrain-seg` | ✓ | MemBrain-seg, GPU | Membrane segmentation. |
| `prepare-template` | `pytom` | ✓ | pytom-match-pick | `pytom_create_template.py` + `pytom_create_mask.py`. |
| `pick-particles` | `pytom-match-pick` | ✓ | pytom-match-pick, **CUDA** | Template matching. No CPU path. |
| `extract-particles` | `cubic-crop` | ✓ | — | Crops cubic subvolumes at pick locations. |
| `average-particles` | `dynamo-search` | ✓ | — | Dynamo's local search + Dynamo's averaging, K=1; `--gold-standard` for independent halves. See [dynamo_search.md](dynamo_search.md). |
| | `dynamo-average` | | — | Dynamo's `dynamo_average` at the stacks' own poses, no search: the map before refinement. Takes the mask and wedge options only. |
| | `cone-search` | | — | Cone search + multi-reference classification. |
| | `fft-refinement` | | — | Simple per-Euler-angle baseline. |

`compute-fsc`, `visualize-picks`, `import-tilt-stack` and `download-empiar`
have no backend choice. That is not the same as needing nothing installed:
`visualize-picks` calls IMOD's `point2model` to turn the picks into a model
3dmod can draw on the tomogram, and finds it with the same
`resolve_imod_executable` helper and the same `BackendContext` the IMOD
backends use, so `--imod-dir` and `--point2model-executable` behave there as
their equivalents do elsewhere. What is absent is the stage `Protocol`: the
conversion is fixed, there is no second way to do it, and so there is no
`--backend` flag and nothing to compare.

## Choosing between backends

### Fiducial versus patch tracking

Gold beads give the most reliable fine alignment, so `imod-autofidseed` plus
`imod-beadtrack` is the default. When the specimen has no beads, switch **both**
steps to the patch pair — `imod-patch-seed` then `imod-patch-track` — which
correlates image patches instead. Patch size is derived from the image size, the
patch count, and the margin rather than entered directly.

### CTF: which estimator

Both estimators write the same IMOD `.defocus` file, so the choice is about
fitting, not about what follows.

- `ctffind5` drives CTFFIND interactively: the program has no command-line
  parameters, it asks questions and reads answers from stdin, positionally.
  Answering no to its movie question makes it fit each slice of the stack
  separately, so one call covers the series. Fitting runs in two passes: the view
  nearest zero tilt over a wide defocus range for a hint, then the stack over a
  narrow window around it. `--ctf-expected-defocus-um` skips the first pass.

  The window (`--ctffind-defocus-search-window-angstrom`, ±1000 Å by default)
  has to hold every view's defocus, and the defocus moves with the tilt: on the
  ryr1 data it fell ~0.03 µm per degree near 0, so ±1000 Å fitted 9 of 29 views
  and ±10000 Å 26 (`scripts/ctf_window_test.sbatch`, 2026-10-16). A view whose
  defocus lies outside it is not fitted, and CTFFIND says so only through the
  fit limit, which stops at the low end of the range (29.7 Å of 30). Such
  views are *unfitted*: their defocus in the model is interpolated from the
  fitted views (linear between neighbours, along the outermost views' line past
  the last), and with at most `--ctf-max-excluded-views` of them (5) `correct-ctf`
  leaves them out of the corrected stack, and so of the tomogram, while with
  more they stay in. The QC warns either way. The model records each row's z
  value, so picking pairs its rows with the views the tomogram kept.

  `--ctffind-determine-sample-tilt` lets CTFFIND5 fit each view's tilt and the
  defocus gradient across it: sharper fits (6.4-7.4 Å against 7-13 Å on ryr1),
  no more views, and about two hours a series. The backend refuses a CTFFIND 4
  by its banner, since that one asks neither the tilt nor the thickness question,
  and records the version that fitted each model.

  The question sequence is not fixed. CTFFIND opens the input file before asking
  anything and asks about movies only when the file holds more than one slice, so
  the hint pass and the full pass answer **different** sequences — that one extra
  question is enough to displace every answer after it. `_ctffind_dialogue`
  builds questions and answers together, mirroring the conditionals in
  `DoInteractiveUserInput`, and `_verify_dialogue` compares them against
  CTFFIND's own echo, because landing an answer on the wrong question is usually
  quiet rather than loud: an answer meant for one question is often a valid value
  for its neighbour.

  `--old-school-input-ctffind4` offers a fixed record format and looks like the
  obvious way out, but it is unusable here: the same view fits in about two
  seconds interactively and had not finished after two minutes in old-school
  mode, on identical parameters. That mode also pins itself to one thread and
  refuses stacks unless it is doing movie processing.
- `imod-ctfplotter` stays inside IMOD and fits with explicit tilt-axis geometry.

The high-resolution fit limit defaults to the Nyquist limit of the stack's own
pixel size, and a finer request is clamped with a warning: on a binned stack a
fixed default such as 5 Å is not physically meaningful.

`correct-ctf` reads voltage, spherical aberration, and amplitude contrast from
the CTF model rather than its own flags, and rejects conflicting overrides, so
a phase flip cannot disagree with the fit it came from. When several models are
registered, pick one with `--ctf-model-backend`.

### Reusing one fit across binnings

Defocus is physical and the IMOD file records it against tilt angles, not
pixels, so a model fitted on one stack corrects the same views at another
binning: what has to match is the pixel size handed to ctfphaseflip, and that is
read from the stack being corrected. This matters because coarse binning is
where fitting is least reliable — at 5.4 Å/pixel the fit window closes at the
10.8 Å Nyquist limit, leaving about six Thon rings for a 4 µm defocus against
a couple of dozen at 2.7 Å/pixel. Fit once at the finest binning and apply the
result to the rest.

The transfer is checked rather than assumed. The model's per-view angles must be
the stack's own, so a model covering different views is refused instead of
phase-flipping each image against the wrong defocus; a stack that records no
tilt-angle file is refused too, because unverifiable is not the same as fine.
The QC report carries a warning naming the stack the fit came from and both
pixel sizes. A model that claims *this* stack but a different pixel size stays a
hard error: one of the two records is wrong and guessing which would put a wrong
pixel size into the flip.

### Averaging: `cone-search` versus `fft-refinement`

`cone-search` provides cone sampling and multi-reference classification.
`fft-refinement` remains a minimal baseline and supports fixed-pose controls.
Neither backend's internal correlation alone establishes an improvement in
structural quality; the CZII GT-positive control deteriorated after cone
refinement even though that correlation increased.

`cone-search` differs in three ways that matter:

- Orientations are searched on a cone plus an in-plane sweep. Nudging each Euler
  angle independently, as `fft-refinement` does, samples orientation space
  unevenly and degenerates near zero tilt.
- Shifts come from FFT cross-correlations instead of a grid of resamplings;
  masked scoring also evaluates the local normalization over that range.
- Several references are refined at once with hard per-particle assignment,
  which is what makes it classification rather than plain averaging. Set
  `--num-classes`.
- `--missing-wedge` restricts the alignment score to the frequencies each
  particle actually measured, evaluated at the candidate orientation, and
  normalises over that shared support rather than over the whole box. The tilt
  range comes from each contributing series' own manifest — so particles averaged
  from series with different ranges get different wedges — and
  `--wedge-min-tilt-deg` / `--wedge-max-tilt-deg` override it, which matters
  because a manifest still lists views that alignment later excluded.
  `--wedge-tilt-axis-deg` says where the tilt axis lies in the tomogram `xy`
  plane; zero (along `y`) is the IMOD-aligned convention and is a flag rather
  than an assumption.

  Default off, and that default is the score measured to be worse: seeded with
  exactly correct orientations and no noise, the whole-box score moved 100% of
  wedge-limited particles by a mean of 5.9 degrees while the same loop with the
  wedge removed from the data moved nothing. Off only so that runs already on
  disk stay comparable — the QC report records which score was used and warns
  when it was the unweighted one. See `cryoet_pipeline.wedge` and
  [roadmap.md](roadmap.md) section 4 for the numbers and the control.

  Cannot be combined with `--alignment-mask-diameter-angstrom`: one restricts the
  Fourier support and the other weights real space, and applying both is not
  their exact combination, so the pair is refused rather than approximated. The
  mask was separately measured not to address the wedge bias.
- `--fourier-compensation` builds each reference as
  `sum(W_i * F_i) / max(sum(W_i), floor)` over the particles' own wedges instead
  of a plain real-space mean, so a frequency measured by few particles is not
  left attenuated relative to one measured by many. Requires `--missing-wedge`,
  since without a wedge model nothing says which frequencies a particle
  measured. Default off.

  The coverage floor is 0.25 of the contributing particles, hardcoded: dividing
  by a near-empty denominator amplifies whatever noise reached that frequency.
  It comes from the synthetic measurements in [roadmap.md](roadmap.md) section 4
  and has not been tuned against real data. `wedge_coverage` in the QC reports
  the minimum, median and floored fraction of the coverage actually seen, which
  is the number that says whether compensation had anything to do: an
  orientation set that happens to cover Fourier space evenly gives a flat
  denominator, and then this is the plain mean by another route.

  The written map is the compensated reference — the change is to how the
  average is built, so the average changes. Note what is NOT included: the
  per-voxel count of real real-space contributions is still not divided out, and
  on a 300 A particle in a box 48 at 10 A/voxel that touches only the box
  corners, where the density is not.

The cone azimuth chooses the direction of a minimal tilt; it is not an extra
in-plane spin. For a sampled tilt `p` and in-plane offset `s`, the full relative
rotation is at most `p + abs(s)`. Both range flags specify full widths, so this
bound is half the cone aperture plus half the in-plane range **per iteration**,
not a cap on the accumulated change from the original pick. The former
azimuth-to-Euler mapping omitted the compensating rotation and could offer
164-degree moves on the CZII 30-degree local-search grid. Results made with
that mapping need a new refinement run; changing the code does not repair
stored poses or averages. Preserve old results under their existing label.

Ranges and their sampling shrink together each iteration by `--refine-factor`
(two to three is usual), but the angular step stops at the Crowther increment,

    dtheta = 180 / (pi * r_max * D) = degrees(d_res / D)

with `D` the particle diameter and `d_res` the target resolution. Below that,
adjacent orientations are not independent. Pass `--particle-diameter-angstrom`
and `--target-resolution-angstrom`; both default to the box size and the Nyquist
limit respectively, which is conservative. RELION's initial-sampling heuristic
`arctan(2 * d_res / D)` is roughly twice this value; the implementation uses the
Crowther figure. Shift search never closes below one pixel, because the peak is
located at whole-pixel accuracy.

Despite the name, `cone-search` is a native torch implementation modelled on
DYNAMO's approach. It does not call DYNAMO or MATLAB.

#### Optional soft alignment mask

`--alignment-mask-diameter-angstrom 300` enables weighted Pearson NCC in
`cone-search`. The default remains the original unmasked scorer, so historical
runs and fixed-pose `fft-refinement` controls retain their behavior. In
`dynamo-search` the same option sets the diameter of the binary search mask,
which also normalises the particles before averaging; `--alignment-mask` takes a
mask file instead, searched above 0.5 and normalising on its nonzero voxels.
Supplying it to `fft-refinement` is rejected rather than silently ignored.

The reference-frame spherical weight is flat to 80% of its radius and has a
cosine falloff to zero at the requested radius (120/150 A for a 300 A diameter).
Each orientation and class uses the same weight. For each allowed shift,
candidate and reference means, covariance and variances are evaluated with
this weight **once**, not by treating zeros outside a mask as observations or
by inadvertently squaring the soft weights. Candidate mean and variance are
shift-dependent; FFT correlations evaluate these moments over the shift window.

The positive mask support plus the integer shift window must fit inside the
rotated box. This is checked explicitly: FFT periodic wraparound must not
introduce density from the opposite edge, and the scored samples must agree
with the final resampling. An empty mask, blank reference, or undefined masked
score fails rather than producing a spurious high correlation.

Mask diameter, fixed taper fraction and `alignment_score` are recorded in the
average artifact and classification QC, alongside `missing_wedge_weighting`,
`wedge_tilt_ranges_deg`, `wedge_tilt_axis_deg`, `fourier_compensation`,
`wedge_coverage_floor_fraction` and `wedge_coverage` (schema 5). Old reports remain
readable: every added field carries a default, and `False`/empty is what those
reports in fact describe.

Starting and final internal scores use the same objective within a run — a
wedge-constrained run measures its start the same way, as a masked run already
did — but do not compare magnitudes ACROSS objectives. Whole-box NCC, masked NCC
and wedge-constrained NCC are three different quantities, and the last is
computed over a support that changes with the candidate orientation, so it is not
on the scale of the other two.

Only pose/class selection changes. The original crop volumes, reference
low-pass schedule, reference averaging, particle counts, and full output maps
are not masked or otherwise changed. This is not particle rejection, wedge
compensation, training, or proof of improved structural quality. In particular,
a geometric mask may exclude real particle density if the crop is off-center.

#### Bounded angular-search geometry check

`test_cone_angular_search_recovers_analytic_ground_truth` checks exactly two
contracts: preserve an already correct pose and recover a known 14.13-degree
on-grid error. A nontrivially oriented, asymmetric noiseless density is evaluated
analytically in continuous XYZ coordinates using independent Rodrigues matrices;
the production resampler does not generate the input particle. Search uses the
CZII first-iteration cone/in-plane grid, a 48-pixel box, 10.012 A pixels and a
60 A low-pass reference. Shifts are frozen in the search kernel.

The acceptance criterion is a full rotation-matrix error below 0.0001 degrees,
not merely a better correlation. Both cases passed on CPU on 2026-09-14.
Temporarily reinstating the former azimuth-spin bug in memory made recovery
fail with a 10-degree error; no production file was reverted. The related
rotation/subtomogram suite passed all 140 cases.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_subtomogram_backend.py -k analytic_ground_truth
```

This closes the basic on-grid geometry hypothesis for this controlled case,
not the CZII quality problem. It does not test whether the score selects useful
poses in noisy, incompletely sampled data or during joint reference updates.
Do not expand this into parameter sweeps without a specific falsifiable failure
hypothesis. No additional cluster run is required for this check.

## Comparing runs side by side

Artifact ids do not encode every choice that changes a result, so two runs of
one stage would otherwise share an id: the second either fails as a duplicate,
or replaces the first when `--overwrite` is given. That is why re-running
`build-final-aligned-stack` at a different `--output-binning` loses the previous
entry — `--overwrite` cannot tell "redo this" from "make a different variant".

`--variant LABEL` resolves it. The label is appended to the ids that run
produces, and selects inputs carrying the same label, so a whole lane runs in
parallel with another:

```bash
cryoet generate-fiducial-seed ... --backend imod-autofidseed --variant bead
cryoet track-fiducials        ... --backend imod-beadtrack   --variant bead
cryoet fine-align-tilt-series ...                            --variant bead

cryoet generate-fiducial-seed ... --backend imod-patch-seed  --variant patch
cryoet track-fiducials        ... --backend imod-patch-track --variant patch
cryoet fine-align-tilt-series ...                            --variant patch
```

Both lanes then coexist in one registry and their QC reports can be compared
directly. The flag is available on `generate-fiducial-seed`, `track-fiducials`,
`fine-align-tilt-series`, `build-final-aligned-stack`, and
`reconstruct-tomogram`. It is also the way to keep two binnings:
`--output-binning 8 --variant bin8` alongside `--output-binning 4 --variant bin4`.

Omitting `--variant` keeps the previous ids exactly, so existing registries are
unaffected. When a lookup finds several candidates it lists them and tells you
to pass `--variant`, rather than guessing.

A label marks where a lane **parts** from the others, so at the parting point
the inputs are still shared and carry no label yet: `build-final-aligned-stack
--output-binning 8 --variant bin8` reads the same unlabelled fine alignment as
every other lane. A lookup falls back to that shared artifact only, never to
another lane's — asking for `--variant patch` when only a `bead` artifact exists
is an error, not a silent substitution.

`estimate-ctf` and `correct-ctf` take the label too, so a whole CTF lane runs
beside another. A binning lane parts at the stack rebuild and carries the label
from there on:

```bash
cryoet build-final-aligned-stack ... --output-binning 8 --variant bin8
cryoet estimate-ctf              ...                    --variant bin8
cryoet correct-ctf               ...                    --variant bin8
cryoet reconstruct-tomogram      ...                    --variant bin8
```

The CTF stages write under a fixed per-backend filename prefix, so the label
goes into the filename as well as the id — otherwise the second lane would
overwrite the first lane's defocus file while both registry entries still
pointed at it.

The label is recorded on the artifact as a field, not only rendered into its id,
so a lookup asks a question about the data rather than parsing strings. Same for
the backend. Artifacts written before those fields existed carry the label only
in their id; those are matched by the id suffix and otherwise counted as shared.

Whatever tells two runs apart is also in the output filenames — the backend, the
binning, the label. Omitting any of them aims the second run at the first run's
files: it either refuses to start or overwrites them while both registry entries
keep pointing at one path.

## Reconstructing with and without CTF

`reconstruct-tomogram --from-stack` names the route:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | Use a CTF-corrected stack when one is registered, otherwise the plain final aligned stack. |
| `ctf-corrected` | Require a corrected stack; say what to run when there is none. |
| `aligned` | Use the plain final aligned stack even when a corrected one exists. |

`auto` was previously the only behaviour and could not be overridden, so once
`correct-ctf` had run there was no way to ask for an uncorrected reconstruction —
not even to see what the correction changed. Pair `--from-stack aligned` with
`--variant` to keep both results:

```bash
cryoet reconstruct-tomogram ...
cryoet reconstruct-tomogram ... --from-stack aligned --variant noctf
```

The route itself is not part of the id: the tomogram records it as
`ctf_corrected`, and `--variant` is what makes two runs coexist. Adding a third
mechanism for the same thing would only mean a second way to ask the same
question.

## Contrast conventions

The wrong contrast convention silently produces nothing rather than failing, so
the one place it is decided is explicit and checked:

- **Template contrast.** EMDB maps have bright density; particles in a tomogram
  are dark. `prepare-template --template-invert` (on by default) multiplies the
  template by -1 to match. QC compares the produced template's excursion around
  its median against the requested inversion and warns on a mismatch.

## Templates that are not compact particles

`--template-particle-diameter-angstrom` describes a sphere, which fits a
globular complex and nothing else. A lattice, a membrane patch, or a slice of a
tube is extended in two directions and thin along the third; a sphere drawn
around it takes in mostly background, and a diameter large enough to span it
does not fit the box at all.

Such a reference is masked with an ellipsoid instead. `pytom_create_mask.py`
reads `-r` as the x semi-axis and takes y and z separately, so:

```bash
cryoet prepare-template ... \
  --template-mask-radius-px 12 \
  --template-mask-radius-y-px 12 \
  --template-mask-radius-z-px 3
```

Radii are in **tomogram** pixels. Giving only `--template-mask-radius-px` keeps
the sphere and adds no flags.

Whether a deposited map is compact is worth checking before choosing: resample
it to the tomogram's voxel size and look at whether the density falls away from
the box walls. Density that holds a substantial fraction of its peak all the way
to the edges in two directions and varies only along the third is a slab, not a
particle — no diameter describes it.

For picking, `--pytom-angular-search` replaces the Crowther sampling that
`--pytom-particle-diameter` would compute, since there is no diameter to compute
it from. `--pytom-z-axis-symmetry` divides the orientation search by the
template's rotational symmetry about z, which is only correct when the box is
centred on that axis — being centred on the density's centre of mass, as
`--template-center` does, is not the same thing.

An ellipsoidal mask also changes what pytom may assume. Its fast path
precomputes the mask's normalisation once and reuses it for every orientation,
which holds only while the mask is rotation-invariant; for anything else the
score ends up ranking orientations rather than measuring fit. The mask records
its own shape when `prepare-template` builds it, and `pick-particles` reads that
to pass `--non-spherical-mask` on its own. `--pytom-non-spherical-mask`
overrides the detection, which is what a mask from outside the pipeline needs.

### Two different diameters

pytom uses a diameter for two unrelated things, and for an extended reference
they are different numbers:

- matching, where it sets the angular sampling through the Crowther criterion —
  `--pytom-particle-diameter`, or bypass it with `--pytom-angular-search`;
- extraction, where half of it is the minimum distance kept between accepted
  peaks — `--pytom-extract-particle-diameter`, defaulting to the matching one.

A lattice makes the split necessary: matching wants the sampling that the
template's longest axis implies, extraction wants peaks no closer than the
lattice repeat. Driving matching with `--pytom-angular-search` leaves no
diameter at all, and extraction requires one, so that combination must set the
extraction diameter explicitly.

### CTF weighting, and why the defocus is restated

Phase flipping corrects the sign of the transfer function but not its amplitude,
so a corrected tomogram still carries |CTF| while a deposited reference map does
not. Correlating them is a mismatched filter. `pick-particles` therefore weights
the template the same way, taking the values from the CTF model that actually
corrected this tomogram — found through the artifact lineage, so they cannot
disagree with the correction. `--no-pytom-ctf-weighting` turns it off; nothing is
declared when nothing corrected the tomogram.

The defocus is handed over as a plain list of micrometres, one per view, rather
than as the IMOD `.defocus` file it came from. pytom accepts that file, but its
version-3 reader averages columns five and six as the two astigmatic axes, and
this pipeline stamps version 3 whether or not astigmatism was fitted. Without
astigmatism column six is either absent — pytom raises — or the phase shift,
which it averages with the defocus and so **halves** it silently. A list of
micrometres has no columns to misinterpret.

The number of defocus values must match the number of tilt angles, since the two
reach pytom as parallel lists with nothing tying them together.

`--pytom-number-of-particles` caps how many candidates extraction returns.
pytom requires it. Everything extraction will refuse is checked before the GPU
match starts rather than after it.

## Coordinate conventions in the two viewers

`visualize-picks` writes one directory for two viewers, and neither of them
reads a coordinate the way the pipeline writes one. Both disagreements are
silent — nothing errors, the picks are simply drawn somewhere other than where
they were found — so the command states the convention instead of letting the
viewer pick it:

- **STAR flavour in ChimeraX.** ArtiaX registers two readers for `.star`:
  RELION 3.1 (`rlnCoordinateX/Y/Z` in pixels, which is what pytom-match-pick
  writes) and RELION 5 (centered coordinates in Ångström). Both claim the suffix
  and both answer to the nickname `star`, so the script written by
  `visualize-picks` pins `format relion` rather than letting ChimeraX choose.
- **Half a voxel in the IMOD model.** IMOD model coordinates run from 0 to NX in
  X and 0 to NY in Y, but from -0.5 to NZ - 0.5 in Z, so the centre of voxel `i`
  is at `i + 0.5` in X and Y and at `i` in Z. pytom-match-pick writes picks as
  voxel indices, so the point list handed to `point2model` adds half a voxel in
  X and Y and adds nothing in Z. This is the kind of error that never announces
  itself: get it wrong and the model still loads onto the tomogram, every sphere
  still lands on the slice it belongs to, and the whole set is displaced by half
  a voxel in XY — a fraction of the radius the picks are drawn at, and the same
  fraction for all of them, so nothing looks inconsistent. It surfaces only as
  picks sitting slightly off their density, which is the very judgement 3dmod
  was opened to make. The offset is stated by IMOD's own binary model spec and
  was confirmed against the library rather than the prose, by painting a model
  sphere onto a volume with `imodmop` and checking which voxels took the paint.

## Adding a backend

1. Implement the stage's `Protocol` in a module under `backends/`.
2. Emit the same artifact kind as the existing backends for that stage, and put
   the backend name in the artifact id so results can coexist with others.
3. Write a QC model dump alongside, following the surrounding stages.
4. Register the name in the matching `_..._backend` dispatcher in
   [`cli.py`](../src/cryoet_pipeline/cli.py).

External-tool backends take an injectable `CommandRunner`, which is how their
tests assert on the exact command line without the tool installed.
