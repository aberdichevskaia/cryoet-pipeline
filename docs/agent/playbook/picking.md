# Picking: `pick-particles`, `visualize-picks`

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `P-3 ✓` if it is right, `P-4 ✗` if it is wrong
>   (say why), `P-5 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
steps, their keys and their QC are in [pipeline_map.md](../pipeline_map.md),
steps 14 and 15. The template is in [template.md](template.md).

- **What the steps do.**
  - `pick-particles` runs template matching with pytom-match-pick (0.14.0 at
    MDC) on a GPU, then extracts the best peaks.
  - `visualize-picks` writes the picks over the tomogram for ChimeraX and
    3dmod.
- **The rules every lane follows:**
  - the angular step is atan(resolution / particle diameter), for full 360°
    searches too;
  - the picks' low-pass is about 1.3 × the tomogram's Nyquist;
  - without `[pick] number_of_particles`, the step keeps twice the number pytom
    recommends.
- **Scores in σ.** pytom also reports the noise of the search, so the QC reads
  scores in σ against the height where one false positive is expected.

## How a healthy run looks

- **Run time per series.**
  - HIV-1 Gag at bin8: 37 min for one series.
  - RyR1 at bin4: 5.0–8.6 h for 147,600 rotations (2.949° step, C4).
- **Significant peaks.** An EMPIAR-11538 series at bin8 (TS_0306): 72 picks
  above the one-false-positive height of 7.36σ, the best at 16.1σ.
- **Picks on density.** Picks sit on particle-sized density inside the specimen
  when opened over the tomogram (`visualize-picks`).
- **The z spread** of the picks is narrower than a uniform fill of the volume
  when the specimen fills only part of it (P-5).

## Where to look first

1. **The picking QC** in `qc/<TS>/picking/`: the number of picks, their
   significance, and the z spread.
2. **The report's picking page**, with the preview of the score map.
3. **The picks over the tomogram:** `picks/<TS>/<TS>_open_in_chimerax.cxc` for
   ChimeraX, or `<TS>_picks.mod` with the `.rec` in 3dmod.
4. **pytom's own logs** in `picks/<TS>/`, and the job header's GPU lines
   (P-1).

## Already checked by the code

- **Matching views.** The views of the tomogram, the CTF model and the mdoc
  must match, or the step stops (P-7).
- **The membrane mask.** A segmentation mask that admits nothing, or has another
  shape than the tomogram, is refused (P-6).
- **Score floors.** `cut_off` and `number_of_false_positives` together are
  refused. A threshold that cannot be a σ, or a raw score that cannot be one,
  is refused too (P-8).

## Checks before trusting the result

1. **Are the picks real?** See P-2.
2. **Do the picks sit inside the specimen in z** (P-5, and R-1 in
   [tomogram.md](tomogram.md))?
3. **Template contrast, pixel size and hand** ([template.md](template.md)).
4. **Gold beads** among the best picks. *General:* high-contrast beads can
   score high against many templates.

## Problems

### P-1. Template matching runs out of GPU memory

**Kind:** crash · **Evidence:** measured

**Where it shows:** `pick-particles` fails with a CUDA out-of-memory error.

**How to confirm:**
- In the job header, `nvidia-smi -L` shows a MIG slice, such as `3g.20gb`.
- The `--query-gpu` line above it shows the whole card (40960 MiB on a 3g.20gb
  slice), so read the slice, not the card.
- `sacct -j <job> -o AllocTRES` shows what was allocated.

**Causes, common to rare:**
1. The tomogram and the search do not fit in the slice's memory.

**What to try, cheap to expensive:**
1. `[pick] volume_split = "2,2,1"`.
2. A finer split.
3. A full GPU in the stage's `sbatch` list.

**What not to do:** shrink the angular search or the box to fit the memory.
That changes the result.

**When to stop and ask someone:** when no split fits.

**Applies to:** every version.

**Source:** HIV-1 Gag (EMPIAR-10164) at bin8, 8 October 2026: `2,2,1` fitted on
a 20 GB slice.

### P-2. The score maps do not see the specimen

**Kind:** silent · **Evidence:** measured

**Where it shows:** the picks do not sit on density. Of the best 100 peaks, no
more fall inside the specimen than chance would put there (compare with noise),
and few or no peaks rise above the height where one false positive is expected.

**How to confirm:**
- Measure the share of the best 100 peaks inside the specimen band against the
  share noise gives.
- Run a plain correlation with the same template on the same tomogram.
- Check the template's pixel size and contrast against the tomogram.

**Causes, common to rare**, as found on one dataset:
1. Random-phase correction and |CTF| weighting together made the maps blind.
   Each alone hurt less; with both off, the maps saw the lamella.
2. Template set-up. A 60 px box with a soft 18 px mask was better than a hard
   17 px mask in a 39 px box.
3. Low-pass: 25 Å gave about a seventh more significant peaks than 35 Å, at
   about 2.7 times the orientations.
4. Before 18 October 2026, a code bug gave pytom the stage angles, not the solved
   ones, so the wedge was 8.9–11.5° off. This is fixed.

**What to try, cheap to expensive:**
1. Check the template's pixel size and contrast.
2. Run the ablation (recipe C, `python3 scripts/submit_recipe.py pick-ablation
   <config> --series TS --dry-run`): the corrections on and off, and the
   low-pass. It switches off corrections the lane must still render. On a lane
   that already runs without them, as the ribo lanes have since 15 October 2026,
   its dry run refuses; a different ablation is a script to ask for, not to
   write.
3. Set the lane to what the ablation measured.

**What not to do:** trust the pick count; change many settings at once. Our
tomograms have the opposite hand to the manual etomo reconstruction, but
flipping one did not change the matcher's peaks, so handedness is not the cause
here.

**When to stop and ask someone:** if the share stays at noise level after the
ablation. The mid-band of these tomograms was weaker than in a manual etomo
reconstruction, and alignment is suspected; that question is open (R-4 in
[tomogram.md](tomogram.md)).

**Applies to:** every version.

**Source:** EMPIAR-11538 ribosome lanes at MDC, 15–16 October 2026 (jobs 3640899
and 3643100). The ribosome lanes now use low-pass 25 Å with
`random_phase_correction = false` and `ctf_weighting = false`.

### P-3. pytom finds no peak worth counting

**Kind:** crash · **Evidence:** seen

**Error text:** `pytom found no peak above its false-positive cut-off, so there
is no count to double; pass --pytom-number-of-particles to take a fixed number of
the best peaks`

**Where it shows:** `pick-particles` stops when the lane gives no
`number_of_particles`.

**Causes, common to rare:**
1. The score map sees nothing above noise (P-2).
2. The template does not match the tomogram: contrast, pixel size or hand
   ([template.md](template.md)).
3. A setting that removes the signal. On EMPIAR-11538, one re-pick with other
   search settings gave no candidates at all.

**What to try:**
1. Set `[pick] number_of_particles` to a fixed number (a few hundred) on one
   series, to see where the best peaks sit.
2. Then work through P-2.

**What not to do:** lower the false-positive setting until something comes
out. The peaks are then noise by construction.

**Applies to:** every version.

**Source:** EMPIAR-11538, TS_0201, Munich, 8 October 2026: a re-pick gave 0
candidates. The message is read in `backends/picking.py`.

### P-4. The best pick is no detection

**Kind:** warning · **Evidence:** code

**Error text** (picking QC): `the best pick is <z> sigma, at or below the <f>
sigma a peak-free search over this space reaches on its own, so no single pick
here is a detection. An excess of picks over the noise prediction can still be
real signal; the count is what to read, not the peak height`

**What it means:** no single peak stands out from what noise gives over this many
positions and orientations. Many weak particles can still be there; the excess
of picks over the noise prediction says so.

**What to try:** read the count above the floor and the excess in the QC. If
there is no excess, work through P-2.

**Source:** read in `backends/picking.py`.

### P-5. The picks fill the volume in z

**Kind:** warning · **Evidence:** measured

**Error text** (picking QC):
- `… standard errors wider than uniform: more picks sit near the top and
  bottom of the volume than a uniform fill puts there. A histogram of the
  picks' Z shows where`
- `… standard errors below uniform, so not distinguishable from it. Three
  readings, and this cannot tell them apart: the picks are noise; or the
  reconstruction does not place the specimen in a band of Z; or the thickness is
  snug around the specimen, which fills the volume and makes uniform the correct
  answer. …`

**What it means:** the QC compares the picks' spread in z with a uniform fill of
the depth pytom searched. Picks in a lamella should be confined to the
lamella's band.

**How to confirm:** the XZ slice of the tomogram, and `scripts/find_slab.py`
(R-1 in [tomogram.md](tomogram.md)).

**What to try:**
- If the volume is mostly vacuum, restrict the search in z with
  `find_slab.py`'s `--pytom-search-z` value, or reconstruct thinner (R-1).
- If the picks crowd at the top and bottom faces, look at what is there: often
  the volume's edges, not particles.

**What not to do:** read "uniform" as "noise" without looking at the
thickness.

**Source:** RyR1 at bin8 at MDC, with the search restricted to membranes,
October 2026. Two series of five were confined (4.4 and 3.3 standard errors
narrower than uniform); three were not distinguishable from uniform.

### P-6. The membrane mask admits nothing, or does not fit

**Kind:** crash · **Evidence:** code

**Error text:**
- `segmentation <id> admits no voxels, so every candidate would be discarded`
- `segmentation <id> has shape <a> but <tomogram> has <b>; pytom requires them
  to match. Segment the tomogram you are picking in, passing the same --variant`

**Causes, common to rare:**
1. An empty segmentation (G-3 in [membranes.md](membranes.md)).
2. A segmentation of another lane's tomogram, at another binning.

**What to try:** look at the segmentation, and segment the tomogram being
picked in, in the same lane.

**Source:** read in `backends/picking.py`.

### P-7. The tomogram's views and the CTF model's do not match

**Kind:** crash · **Evidence:** code

**Error text:**
- `the CTF model covers <n> views but the tomogram was reconstructed from <m>;
  pytom pairs defocus with tilt angles by position, so the two must describe the
  same views`
- `the mdoc covers <n> views but the tomogram was reconstructed from <m>; pytom
  pairs dose with tilt angles by position, so the two must describe the same
  views`
- `the tomogram was built from z values <z> that the CTF model has no defocus
  for`

**Causes, common to rare:**
1. The CTF fit or the alignment was redone after this tomogram was made (C-11,
   C-10 in [ctf.md](ctf.md)).

**What to try:** rerun the lane from `correct-ctf`, or from the step that
changed, so the tomogram and its models describe the same views.

**Source:** read in `backends/picking.py`.

### P-8. A score threshold given in the wrong unit

**Kind:** crash · **Evidence:** code

**Error text** (choosing picks by score):
- `a raw pytom score of <x> is impossible (LCCmax is at most 1); a value like
  this is in σ, so give it as σ`
- `a threshold of <x>σ would keep every peak; a value like this is a raw pytom
  score, so give it as a score`
- `no pick lies above its one-false-positive height (<h>σ), so there is nothing
  to take a multiple of`

**What it means:** pytom's raw scores are correlations of at most 1. The QC's
scores are in σ of the search's noise. A threshold has to say which it is.

**What to try:** give the threshold in the unit the message names. Read the
one-false-positive height from the picking QC first.

**Source:** read in `pick_selection.py`.

### P-9. Picking takes many hours

**Kind:** slow · **Evidence:** measured

**Where it shows:** `pick-particles` runs for hours and can reach the stage's
time limit.

**Causes, common to rare:**
1. The angular step from the rule, atan(resolution / diameter), is fine at fine
   binning. RyR1 at bin4 needed 147,600 rotations even with C4 symmetry.
2. A large volume, or one split into many pieces.

**What to try:**
1. Time one series before the lane, and give the stage time to spare.
2. Check the site's rules for long GPU jobs. At MDC, H100 jobs must stay under
   6 hours.

**What not to do:** coarsen the angular step to save time without saying so in
the lane. It changes what is found.

**When to stop and ask someone:** when one series needs more than the
partition's longest time.

**Applies to:** pytom-match-pick 0.14.0.

**Source:** RyR1 at bin4 at MDC, 30 October 2026: 5.0–8.6 h per series, only
1.1 times the time of a search with a third of the rotations on 14 October. Our
time model was wrong, and why is not known.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-P1.** How do you judge that picks are real before averaging: by eye, by
  score, by a decoy?
- **Q-P2.** Do gold beads, carbon edges or ice crystals show up among your
  picks? Do you erase beads before picking?
- **Q-P3.** How many picks per tomogram do you keep for a first average, and how
  do you choose the number?
- **Q-P4.** Which pytom settings do you use for lamellae: random-phase
  correction, CTF weighting, spectral whitening?
- **Q-P5.** Have you met a picking problem that is not here? In a line each:
  what you saw, what it turned out to be, what helped.
