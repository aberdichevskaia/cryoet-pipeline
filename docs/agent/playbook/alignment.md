# Alignment: coarse alignment, fiducials, fine alignment

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `A-4 ✓` if it is right, `A-5 ✗` if it is wrong
>   (say why), `A-6 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
steps, their keys and their QC are in [pipeline_map.md](../pipeline_map.md),
steps 2 to 6.

- **Tools.** IMOD at MDC on 5 November 2026 was 5.1.12.
  - Coarse alignment: `tiltxcorr` and `xftoxg`.
  - Beads: `autofidseed` and `beadtrack`.
  - Patches: `tiltxcorr` patch tracking.
  - Fine alignment: `tiltalign`.
- **The steps.**
  1. `align-tilt-series` (coarse alignment).
  2. `qc-coarse-alignment`.
  3. `generate-fiducial-seed`.
  4. `track-fiducials`.
  5. `fine-align-tilt-series`.
- **Two routes.** Gold beads (`imod-autofidseed` with `imod-beadtrack`) or
  patches (`imod-patch-seed` with `imod-patch-tracking`). A lamella without
  beads takes patches.
- **Shared by every lane.** All these steps are shared by every lane of a
  project. Redoing them for a series changes what every lane sees for it.
- **Not covered.** An AreTomo route is being built (November 2026) and is not
  covered here.

## How a healthy run looks

- **Fine alignment on beads.** RyR1 at MDC: a mean residual of 0.25–0.29 nm on
  15–24 beads per series.
- **Fine alignment on patches.** EMPIAR-11538 at MDC: a mean residual of
  1.39 nm on TS_0306, where the report lists 50 of 61 tilts aligned.
- **Coarse alignment QC.** On EMPIAR-11830 Position_1 with whole-frame motion
  correction: rms 0.30 px, max 1.0 px.
- **Run times per series** (RyR1 and HIV-1 Gag):
  - coarse alignment 11–179 s;
  - seed 25–79 s;
  - tracking 5–23 s;
  - fine alignment 6–139 s.
- **QC thresholds** (pipeline_map, steps 3, 5 and 6):
  - coarse: the residual's 95th percentile above 2 preview px warns, a maximum
    above 5 fails;
  - tracking: fewer than 10 fiducials fails; coverage below 0.8 warns, below
    0.5 fails;
  - fine: a mean residual above 0.8 nm warns and above 1.5 nm fails; a largest
    point residual above 5 tracking px warns and above 20 fails.

## Where to look first

1. **`python3 scripts/show_project.py <lane config>`** shows the QC verdict of
   each step per series.
2. **`python3 scripts/show_failure.py <lane config>`** shows the failing command
   and the steps whose result failed QC.
3. **IMOD's own logs**, next to the outputs. The real error is often there:
   - `alignments/<TS>/` for `tiltxcorr` and `xftoxg`;
   - `fiducials/<TS>/` for `autofidseed` and `beadtrack`;
   - `alignments/<TS>/fine/` for `tiltalign` and the outlier pruning.
4. **The QC records:** `qc/<TS>/coarse_alignment/` (with a coarse-aligned
   preview stack), `fiducials/<TS>/<TS>_tracking_qc.json` and
   `alignments/<TS>/fine/<TS>_fine_alignment_qc.json`.
5. **On a frames lane**, the motion QC first ([motion.md](motion.md), M-2). A
   tilt cut into tiles breaks alignment, and the alignment QC then points at
   the wrong step.

## Already checked by the code

- **Matching pairs.** A seed and a tracker of different routes (beads with
  patches) are refused before the job runs.
- **Old patch seeds.** A patch seed made before 27 October 2026 is refused (A-3).
- **The tilt axis.** No tilt axis is taken from Tomography mdocs; the lane must
  give it (D-8 in [data.md](data.md)).
- **Outlier pruning.** Pruning that would remove more than `max_pruned_fraction`
  of the points is refused (A-4).
- **The thickness on patch lanes.** A patch lane without
  `[reconstruct] thickness` is refused, since patches do not measure the
  specimen's depth ([tomogram.md](tomogram.md)).

## Checks before trusting the result

1. **Low residuals are not proof.** On the patch route before 27 October 2026
   they were low while every view was off by its coarse shift (A-3).
2. **Excluded views.** Which views did coarse alignment leave out, and why?
   (A-7)
3. **The tilt axis angle** of the fine alignment against the lane's or the
   authors' value. A drift of several degrees along the series means a failed
   fit. *General.*

## Problems

### A-1. Coarse alignment QC fails, the rest is fine

**Kind:** warning · **Evidence:** measured

**Error text** (coarse QC): `residual maximum shift <x> px exceeds failure
threshold <t> px` (5 preview px by default)

**Where it shows:** `qc/<TS>/coarse_alignment/` has status `fail`: the largest
residual is above 5 preview px.

**How to confirm:** check whether bead tracking and fine alignment of the same
series passed, and with what residual.

**Causes, common to rare:**
1. Cross-correlation struggles with this series, because of large shifts or low
   contrast. The fine alignment on beads corrects it.

**What to try:** nothing, if fine alignment passes. If it fails too, open the
coarse preview in 3dmod and check the tilt-axis angle.

**What not to do:** stop the chain, or retune coarse alignment, before seeing
the fine alignment.

**When to stop and ask someone:** when fine alignment fails as well.

**Applies to:** every version.

**Source:** RyR1 series 097, 14 October 2026. Coarse QC failed (rms 26.7, max
144 px); fine alignment passed, like the other series' (0.25–0.29 nm on 15–24
beads).

### A-2. `beadtrack` crashes while seeding

**Kind:** crash · **Evidence:** measured

**Where it shows:** the lane job fails at `generate-fiducial-seed`. `beadtrack`
crashed with a segmentation fault (exit 139) inside `autofidseed`. The
`autofidseed` log is in `fiducials/<TS>/`.

**How to confirm:**
- The same crash happens on a second node (with `--overwrite`).
- The series' coarse QC passed.
- The other series passed the same step.
- The series stands out in some way. Here it had the largest coarse shift in the
  set, 0.150 of the image.

**Causes, common to rare:**
1. A crash in IMOD 5.1.12's `beadtrack` at this tracking binning on this series.
   The root cause is unknown.

**What to try:** `tracking_binning` 2 → 3 for that series only, on a config copy
(recipe B). Record the exception in the lane file, and split the stages so that
a later run does not redo the alignment with the lane's value.

**What not to do:**
- Rerun unchanged, since the crash reproduces.
- Change the binning for every series.

**When to stop and ask someone:** if it crashes at binning 3 too. Then it is a
report for IMOD, with the model and the command.

**Applies to:** IMOD 5.1.12.

**Source:** RyR1 series 161, 14 October 2026. With binning 3 the whole CPU stage
passed, with every QC at pass.

### A-3. Patch tracking left every view off by its coarse shift

**Kind:** silent · **Evidence:** measured

**Error text** (`track-fiducials`, since 27 October 2026): `this patch seed
predates prealigned tracking stacks, so tracking it would apply the coarse shifts
twice; rerun generate-fiducial-seed`

**Where it shows:** nowhere in the QC.
- The fine-alignment residuals look normal, because they are computed on the
  tracked model, before the transforms are composed.
- In the reconstruction, every view sits displaced by its own coarse shift.
- This affects patch lanes only. Bead lanes prealign their tracking stack and
  are not affected.

**How to confirm:**
- Read the spread of `fiducials/<TS>/<TS>_patch.prexg`, the coarse shifts the
  patch seed used. The error is as large as that spread.
- Compare the final transforms with the ones tracking alone measured. They
  differ by the coarse shift.
- Check the date. A seed made before 27 October 2026 has the error, and since
  then `track-fiducials` refuses such a seed (the message above).

**Causes, common to rare:**
1. A bug in our code, fixed on 27 October 2026. The patch seed tracked on a stack
   that was binned but not prealigned. Fine alignment then composed the coarse
   shift with the tracked one, so it was applied twice.

**What to try:** rerun the patch lanes from `generate-fiducial-seed` with
`--overwrite`, for every series: a config copy whose first stage starts there
([rerun_recipes.md](../rerun_recipes.md)). Compare the tomograms and the picks
with the ones from before.

**What not to do:**
- Tune alignment parameters.
- Take the low residuals as proof that the alignment is right.
- Look for the cause in the data.
- Call this the cause of weak picking before a rerun shows it.

**When to stop and ask someone:** whenever the transforms the reconstruction
uses disagree with what tracking measured. That is a code problem, and it goes
to the maintainer with the numbers.

**Applies to:** patch lanes with a seed made before 27 October 2026.

**Source:** EMPIAR-11538, ribo-mdc series TS_0306, 27 October 2026. It turned up
while checking how much of the image the patch margin cut: 100 px removed 63% of
the area, and the default is now etomo's 5% of each edge.
- Measured with local IMOD 5.1.12: the final transforms were off from the etomo
  route by a median of 3.4 px, and at most 9 px, at bin8 (33 and 88 Å).
- The spread of the cluster's `.prexg` for TS_0306 was the same size.

### A-4. Fine alignment refuses to prune that many points

**Kind:** crash · **Evidence:** measured

**Error text:** `refusing to prune <n>/<m> fiducial points; configured maximum
fraction is <f>`

**Where it shows:** `fine-align-tilt-series` stops, and the chain with it.

**How to confirm:** the outlier-pruning log and the `tiltalign` log in
`alignments/<TS>/fine/`. They show which points and views carry the large
residuals.

**Causes, common to rare:**
1. More points look like outliers than the limit allows. The default
   `max_pruned_fraction`, 0.02, is sized for beads; a 5 × 5 patch grid needs
   about 0.08.
2. A real alignment problem that pruning would hide: a broken view, or a wrong
   tilt axis.

**What to try:**
1. Look at where the outliers sit. If they cluster on a few views, look at those
   views first (M-2 in [motion.md](motion.md), A-7).
2. If they are spread thinly, raise `[fine_align] max_pruned_fraction` for that
   series on a config copy (recipe B).

**What not to do:** raise the limit for the whole lane without looking at the
points.

**When to stop and ask someone:** when the points to prune cluster in views
that look fine in 3dmod.

**Applies to:** every version.

**Source:** EMPIAR-11538, series TS_0388, Munich, October 2026. Pruning would
have removed 251 of 1500 points against a limit of 0.08. The limit was raised for
the rerun.

### A-5. Fine alignment of a lamella fails on patches

**Kind:** warning · **Evidence:** measured

**Error text** (fine-alignment QC): `mean residual <x> nm exceeds failure
threshold <t> nm` (1.5 nm by default)

**Where it shows:** the fine-alignment QC fails far above the threshold, while
motion and coarse alignment passed.

**How to confirm:**
- The motion QC passes (M-2), and the coarse QC too.
- Fixing the tilt angles in `tiltalign` changes little. Then the recorded angles
  fit, and the angle corrections tiltalign reports are noise.
- The large residuals are spread over many views, not a few.

**Causes, common to rare:**
1. IMOD patch tracking does not hold on this series. Its patches follow
   different features from view to view on thick, low-contrast lamellae.
   *Seen on one series.*
2. Views that should be left out. The authors of EMPIAR-11830 removed seven
   views of this series, four of them not dark: −66°, −12°, −9° and +6°.

**What to try:**
1. Read the motion QC and the coarse QC first.
2. Ask the maintainer. Splitting the patch contours into chunks (IMOD's
   `imodchopconts`, an etomo option that is not its default) brought the
   residual down from 17.7 to 8.2 nm locally, still above the threshold. An
   AreTomo route is being built.

**What not to do:**
- Drop views by darkness alone (`[align] min_std_ratio`). No darkness rule
  catches the views the authors removed.
- Take a patch lane's failure as proof that the data cannot be aligned. The
  authors aligned this series with AreTomo.

**When to stop and ask someone:** always, at present.

**Applies to:** the IMOD patch route, as of 8 November 2026.

**Source:** EMPIAR-11830 Position_1, 7–8 November 2026. After motion correction
with whole frames, the coarse QC passed (rms 0.30, max 1.0 px), and fine
alignment failed with a mean of 18.5 nm. Locally, with the five end views left
out, it gave 17.7 nm, and 16.4 nm on the authors' 34 views.

### A-6. Too few fiducials, or tracked in too few views

**Kind:** warning · **Evidence:** code; the causes are *general*

**Error text** (tracking QC):
- `found <n> fiducials; minimum is <m>`
- `fiducial coverage <c> is below failure threshold <t>` (or `warning
  threshold`)

**Causes, common to rare:**
1. Few beads in the field, or none on a lamella.
2. Beads lost at high tilt, or hidden behind thick material.
3. A tracking binning so coarse that small beads vanish.

**What to try:**
1. Count the beads in the view nearest 0° in 3dmod.
2. If there are few, use the patch route for that data.
3. Otherwise try another `tracking_binning` or `target_beads` on one series
   (recipe B).

**What not to do:** lower `min_tracked_fiducials` to pass the QC without
looking.

**Applies to:** every version.

**Source:** read in `backends/fiducials.py`.

### A-7. Coarse alignment leaves views out

**Kind:** reading results · **Evidence:** seen

**Error text** (coarse QC): `coarse alignment excluded low-variance tilt images:
<views>`; or `coarse alignment excludes every tilt image`.

**What it means:** views with too little variance, by `[align] min_std_ratio`,
are left out from here on: dark or empty views, usually at the highest tilts.
The QC lists them.

**When it is not fine:**
- When views in the middle of the range are left out. Open them.
- On a frames lane, when the dark views are dark because motion correction
  failed on them. Then the defect is in the sums (M-2).
- When every view is left out: the ratio is wrong for this data.

**What to try:** change `min_std_ratio` only after looking at the left-out
views. The EMPIAR-11538 lanes, which are stack lanes, use 0.84.

**Source:**
- EMPIAR-11538 lanes, October 2026.
- On EMPIAR-11830, dropping the dark end views by this ratio was proposed, but
  the defect was in the sums (M-2).

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-A1.** What fine-alignment residual do you accept, in nm, with beads and
  with patches? Are 0.8 and 1.5 nm sensible thresholds?
- **Q-A2.** When do you choose patches over beads, and how many beads make a
  bead series workable?
- **Q-A3.** Which views do you remove before alignment, and by what criterion?
- **Q-A4.** What do you do when `tiltalign` reports a large tilt-angle or Z
  correction after positioning? The QC warns above 0.1° and 1 px.
- **Q-A5.** For lamellae, which aligner works for you: etomo patch tracking,
  AreTomo, something else? With which settings?
- **Q-A6.** Have you met an alignment problem that is not here? In a line each:
  what you saw, what it turned out to be, what helped.
