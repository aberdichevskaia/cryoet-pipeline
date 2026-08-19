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
| `estimate-ctf` | `ctffind5` | ✓ | CTFFIND, IMOD | Two-pass, one call per view. See below. |
| | `imod-ctfplotter` | | IMOD | Tilt-aware fitting in IMOD's own framework. |
| `correct-ctf` | `imod-ctfphaseflip` | ✓ | IMOD | Consumes any `CTF_MODEL`, whichever backend fitted it. |
| `reconstruct-tomogram` | `imod-tilt` | ✓ | IMOD | Weighted back-projection. |
| `restore-tomogram` | `isonet2` | ✓ | IsoNet2 | Needs a trained model; optional stage. |
| `segment-tomogram` | `membrain-seg` | ✓ | MemBrain-seg, GPU | Membrane segmentation. |
| `prepare-template` | `pytom` | ✓ | pytom-match-pick | `pytom_create_template.py` + `pytom_create_mask.py`. |
| `pick-particles` | `pytom-match-pick` | ✓ | pytom-match-pick, **CUDA** | Template matching. No CPU path. |
| `extract-particles` | `cubic-crop` | ✓ | — | Crops cubic subvolumes at pick locations. |
| `average-particles` | `cone-search` | ✓ | — | Cone search + multi-reference classification. |
| | `fft-refinement` | | — | Simple per-Euler-angle baseline. |

`compute-fsc`, `export-picks`, `import-tilt-stack` and `download-empiar` have no
backend choice.

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

- `ctffind5` drives CTFFIND through `--old-school-input-ctffind4`, a fixed
  six-line stdin format kept for backwards compatibility. The interactive
  prompt sequence was tried first and does not work: it differs between CTFFIND
  builds, so a hardcoded list of answers lands on the wrong questions. That
  format takes one 2D image per call and rejects a stack unless it is declared
  a movie (which averages frames instead of fitting each), so every view is
  extracted with `newstack` and fitted separately. Fitting runs in two passes:
  the view nearest zero tilt over a wide defocus range for a hint, then every
  view over a narrow window around it. Supplying `--ctf-expected-defocus-um`
  skips the first pass.

  Old-school input does not expose CTFFIND5's tilt and thickness estimation, or
  a separate exhaustive-search toggle; astigmatism restraint is the tolerance
  value in `--ctffind-astigmatism-tolerance-angstrom`, where zero means no
  restraint.
- `imod-ctfplotter` stays inside IMOD and fits with explicit tilt-axis geometry.

The high-resolution fit limit defaults to the Nyquist limit of the stack's own
pixel size, and a finer request is clamped with a warning: on a binned stack a
fixed default such as 5 Å is not physically meaningful.

`correct-ctf` reads voltage, spherical aberration, and amplitude contrast from
the CTF model rather than its own flags, and rejects conflicting overrides, so
a phase flip cannot disagree with the fit it came from. When several models are
registered, pick one with `--ctf-model-backend`.

### Averaging: `cone-search` versus `fft-refinement`

`cone-search` is the one to use. `fft-refinement` remains as a minimal baseline.

`cone-search` differs in three ways that matter:

- Orientations are searched on a cone plus an in-plane sweep. Nudging each Euler
  angle independently, as `fft-refinement` does, samples orientation space
  unevenly and degenerates near zero tilt.
- Shifts come from the cross-correlation peak, so the whole allowed range costs
  one FFT instead of a grid of resamplings.
- Several references are refined at once with hard per-particle assignment,
  which is what makes it classification rather than plain averaging. Set
  `--num-classes`.

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

## Contrast conventions

Two places silently produce nothing rather than failing when the convention is
wrong, so both are explicit and checked:

- **Template contrast.** EMDB maps have bright density; particles in a tomogram
  are dark. `prepare-template --template-invert` (on by default) multiplies the
  template by -1 to match. QC compares the produced template's excursion around
  its median against the requested inversion and warns on a mismatch.
- **STAR flavour in ChimeraX.** ArtiaX registers two readers for `.star`:
  RELION 3.1 (`rlnCoordinateX/Y/Z` in pixels, which is what pytom-match-pick
  writes) and RELION 5 (centered coordinates in Ångström). Both claim the suffix
  and both answer to the nickname `star`, so the script written by `export-picks`
  pins `format relion` rather than letting ChimeraX choose.

## Adding a backend

1. Implement the stage's `Protocol` in a module under `backends/`.
2. Emit the same artifact kind as the existing backends for that stage, and put
   the backend name in the artifact id so results can coexist with others.
3. Write a QC model dump alongside, following the surrounding stages.
4. Register the name in the matching `_..._backend` dispatcher in
   [`cli.py`](../src/cryoet_pipeline/cli.py).

External-tool backends take an injectable `CommandRunner`, which is how their
tests assert on the exact command line without the tool installed.
