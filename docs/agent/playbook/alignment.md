# Alignment: coarse alignment, fiducials, fine alignment

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `A-4 ✓` if it is right, `A-5 ✗` if it is wrong
>   (say why), `A-6 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
> - A-8 to A-13, the AreTomo route, were added on 11 November 2026 from one
>   series (EMPIAR-11830 Position_1) and AreTomo2's manual and source.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained in [playbook.md](../playbook.md). The
steps, their keys and their QC are in [pipeline_map.md](../pipeline_map.md),
steps 2 to 6.

- **Tools.** IMOD at MDC on 5 November 2026 was 5.1.12.
  - Coarse alignment: `tiltxcorr` and `xftoxg`.
  - Beads: `autofidseed` and `beadtrack`.
  - Patches: `tiltxcorr` patch tracking.
  - Fine alignment: `tiltalign`, or AreTomo2. At MDC AreTomo2 was 1.1.3
    (module `aretomo/v2-cu12.5`) on 9 November 2026.
- **The steps.**
  1. `align-tilt-series` (coarse alignment).
  2. `qc-coarse-alignment`.
  3. `generate-fiducial-seed`.
  4. `track-fiducials`.
  5. `fine-align-tilt-series`.

  On the AreTomo route steps 3 and 4 do not run.
- **Three routes.**
  - Gold beads: `imod-autofidseed` with `imod-beadtrack`.
  - Patches: `imod-patch-seed` with `imod-patch-tracking`.
  - AreTomo2: `[fine_align] backend = "aretomo"`, with no `[fiducials]`
    section. It aligns from the tilt stack itself by projection matching, on a
    GPU ([aretomo_backend.md](../../aretomo_backend.md)).

  A lamella without beads takes patches or AreTomo2.
- **Shared by every lane.** All these steps are shared by every lane of a
  project. Redoing them for a series changes what every lane sees for it.

## How a healthy run looks

- **Fine alignment on beads.** RyR1 at MDC: a mean residual of 0.25–0.29 nm on
  15–24 beads per series.
- **Fine alignment on patches.** EMPIAR-11538 at MDC: a mean residual of
  1.39 nm on TS_0306, where the report lists 50 of 61 tilts aligned.
- **Fine alignment by AreTomo2.** EMPIAR-11830 Position_1 at MDC, 10 November
  2026, the only series so far: QC pass, 34 of 41 tilts aligned, 6
  projection-matching passes in 3 rounds, the last pass's mean 0.07 nm (not a
  residual, A-8). The tilt axis was solved at 83.2° from the lane's 84.1°, and
  the tilt offset was +12° (A-9, A-10).
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
   - `alignments/<TS>/fine/` for `tiltalign` and the outlier pruning, or for
     AreTomo2: `<TS>_fine_aretomo.log` (its whole output) and
     `<TS>_fine_aretomo.aln`.
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
- **The AreTomo route.** Before the job runs, the dry run refuses:
  - an AreTomo lane without `[reconstruct] thickness` (AreTomo2 measures no
    depth either);
  - a `[fiducials]` section, or tiltalign's keys in `[fine_align]`;
  - a stage that runs `fine-align-tilt-series` without asking for a GPU.

  The step itself refuses AreTomo3, whose command line is not AreTomo2's
  (A-13).

## Checks before trusting the result

1. **Low residuals are not proof.** On the patch route before 27 October 2026
   they were low while every view was off by its coarse shift (A-3).
2. **Excluded views.** Which views did coarse alignment leave out, and why?
   (A-7)
3. **The tilt axis angle** of the fine alignment against the lane's or the
   authors' value. A drift of several degrees along the series means a failed
   fit. *General.*
4. **On the AreTomo route**, an xz slice of the bin8 tomogram: the lamella
   should lie flat. AreTomo2's QC number does not show this (A-8), and a wrong
   tilt offset does (A-9).

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
   residual down from 17.7 to 8.2 nm locally, still above the threshold.
3. The AreTomo route (`[fine_align] backend = "aretomo"`), which aligned this
   series on 10 November 2026 (A-8 to A-10).

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

### A-8. AreTomo: the QC number is not a residual

**Kind:** reading results · **Evidence:** measured; how it compares with
tiltalign's residual is not

**Where it shows:** `last_pass_correction_mean_nm` in the fine-alignment QC, the
tilt-alignment page of `cryoet report`, and the residual column of the report's
summary table, marked `*`.

**What it means:** AreTomo2 has no fiducials. Its projection matching compares
each view with a reprojection of the others and shifts the view to match. The
QC reports the shifts of the last pass: what AreTomo2 still moved, not what is
left after fitting a model, as tiltalign's residual is. The two do not compare.
On Position_1 the last pass moved the views by 0.07 nm on average, where
tiltalign on patches had left 18.5 nm (A-5); the alignment is not 250 times
better. The 0.8 and 1.5 nm thresholds are tiltalign's, reused.

**How the passes read** (AreTomo2's source, and Position_1's log agrees):
- It works in rounds. With the axis given and refined, as the pipeline runs it,
  there are three, and the axis is searched again before the second and third.
- Within a round it makes passes until the largest shift is under 2 px. A pass
  that moves a view further than the last kept one is undone, although its table
  is printed; the report draws it hollow.
- The first pass of a round can be large again after the axis moved. That is not
  a fault.
- The reference view, the one nearest 0° after the tilt offset, is never
  printed. One aligned tilt without a point on the report's page is expected.

**What to check instead:** the lamella in an xz slice of the bin8 tomogram, the
tilt offset (A-9), the tilt axis (A-10), and whether picking finds the specimen.

**What not to do:** put it beside bead or patch residuals in a table or on a
slide; tighten the thresholds on it.

**Applies to:** AreTomo2 1.1.3; the report's page since 10 November 2026.

**Source:** EMPIAR-11830 Position_1, MDC, 10 November 2026: 6 passes in 3
rounds of 2, the rounds starting from axes 86.73, 83.70 and 83.21°; in the last
pass the largest shift was 0.17 nm (0.9 px). The rounds and the undo rule were
read in AreTomo2's source the same day (`CProcessThread::mAlign` and
`mProjAlign`).

### A-9. AreTomo: every tilt angle moves by a whole-degree offset

**Kind:** reading results · **Evidence:** measured on one series (the log and
AreTomo2's source agree); the advice on averaging is *manual*

**Where it shows:** `applied_tilt_angle_offset_deg` in the fine-alignment QC,
and every page after it: the tilt range in the report's summary, the
aligned-tilts page, the CTF page. On Position_1 the view taken at +45° shows as
+57°.

**What it means:**
- With `[fine_align] aretomo_tilt_cor` unset or true, AreTomo2 measures the
  offset between the stage's angles and the specimen's, which milling or
  mounting gives, and the pipeline applies it from then on. The final stack, the
  CTF fit and the reconstruction all take the nominal angle plus the offset, so
  that the lamella lies flat in the tomogram.
- The offset is the best of 31 whole degrees from −15 to +15, with no finer step
  (`CTiltOffsetMain::mSearch`). It is good to half a degree at best, and an
  offset beyond 15° cannot be found.

**How to confirm a wrong offset:** an xz slice of the bin8 tomogram. A lamella
lying at an angle about the tilt axis has the wrong offset, by about that angle.

**Causes, common to rare:**
1. A failed measurement. AreTomo2's manual (1.1, section 6) says it is
   occasionally wrong.
2. A true offset beyond ±15°.

**What to try:**
1. `[fine_align] aretomo_tilt_cor = false` on one series, on a config copy
   (recipe B). AreTomo2 then measures the offset for its own alignment only, and
   the tomogram keeps the nominal angles. Compare the xz slices.
2. Not built yet: AreTomo2 takes a known offset as the second value of
   `-TiltCor`, and the pipeline does not pass one. Ask the maintainer.

**What not to do:** read an angle on a later page as the nominal one; take the
offset as right without looking at the slice.

**Before averaging:** AreTomo2's manual (1.1, sections 16.1 and 16.4) advises
against applying the offset before subtomogram averaging: it rotates the
tomogram, and the missing wedge then differs between tomograms. The pipeline
carries the corrected angles into every later step, as the authors of
EMPIAR-11830 did (their deposited `.tlt` has +8° applied). *For the
maintainer:* whether the RELION export carries the same angles has not been
checked.

**Applies to:** AreTomo2 1.1.3, with the pipeline's default
`aretomo_tilt_cor = true` (`-TiltCor 1`).

**Source:** EMPIAR-11830 Position_1, MDC, 10 November 2026: +12.00°, where the
authors' files have +8°. The difference has not been looked into.

### A-10. AreTomo: the tilt axis it solves is not the lane's

**Kind:** reading results · **Evidence:** measured on one series; the search
window is *manual* and AreTomo2's source

**Where it shows:** `tilt_axis_angle_deg` against `initial_tilt_axis_angle_deg`
in the fine-alignment QC, with `tilt_axis_spread_deg`; the round labels on the
report's tilt-alignment page.

**What it means:** the pipeline gives AreTomo2 the coarse alignment's axis, the
lane's `[align] tilt_axis_angle`, as `-TiltAxis <axis> 1`. AreTomo2 searches
near it, again before each later round, and then solves an axis for every view
(manual 1.1, section 15). The manual gives the window as ±3°; the source
searches within 6° and then 3°. A few degrees from the lane's value is normal.
The spread is the range of the per-view axes.

**When it is not fine:**
- The solved axis sits about 6° from the lane's, at the edge of the window. Then
  the lane's axis is probably wrong: check it on the coarse preview.
- The lane took its axis from someone else's AreTomo files. Theirs may be of a
  mirrored stack: the authors' `.xf` for Position_1 has −84.6°, on a stack
  mirrored in Y, where ours solves +83.2°.

**What not to do:** copy an axis from deposited files without checking how
their stack is oriented.

**Applies to:** AreTomo2 1.1.3.

**Source:** EMPIAR-11830 Position_1, MDC, 10 November 2026: the lane's 84.1°,
the rounds starting from 86.73, 83.70 and 83.21°.

### A-11. AreTomo: the alignment height is 600 pixels, whatever the pixel

**Kind:** silent · **Evidence:** manual (AreTomo2 1.1, section 13); its effect on
our data is not measured

**Where it shows:** `align_z_px` in the fine-alignment QC. Nothing warns.

**What it means:** AreTomo2's `-AlignZ` is the height of the volume it
reconstructs for projection matching, in pixels of the stack it reads. The
manual counts it among the settings that matter most for accuracy, and wants it
close to the specimen's thickness. Unset, the pipeline passes AreTomo2's own
default, 600 px of the full-resolution stack: 120 nm at 2 Å per pixel, 60 nm at
1 Å. `[fine_align] aretomo_align_z_nm` sets it in nm.

**How to confirm:** `align_z_px` times the stack's pixel size, against the
lamella's thickness measured in an xz slice.

**What to try:** `aretomo_align_z_nm` near the measured thickness on one series,
on a config copy (recipe B). Compare the xz slices and the picks.

**What not to do:** set it to a padded `[reconstruct] thickness`. The manual
wants the alignment height smaller than the reconstruction's.

**Applies to:** AreTomo2 1.1.3. Its `--help` says 256; its code and manual say
600.

**Source:** AreTomo2's manual 1.1 (revised 19 February 2024) and `CInput.cpp`,
read on 11 November 2026. Not met on our data.

### A-12. AreTomo leaves out views it finds dark

**Kind:** warning · **Evidence:** code; the threshold is *manual*

**Error text** (fine-alignment QC): `AreTomo left out views it found dark: z
<z values>`

**What it means:** AreTomo2 drops views darker than its `-DarkTol`, 0.7 by
default; a lower value drops fewer (manual 1.1, section 17). Since 1.1.3 it also
drops nearly flat views. The pipeline leaves them out of the final stack, the
CTF fit and the reconstruction, as it does the views coarse alignment left out
(A-7).

**When it is not fine:**
- Views in the middle of the tilt range, or many views.
- On a frames lane, a tilt that motion correction cut into tiles can look dark.
  Read the motion QC of those tilts first (M-2).

**What to try:** look at the views; then a lower `[fine_align] aretomo_dark_tol`
on one series, on a config copy (recipe B).

**Applies to:** AreTomo2 1.1.3.

**Source:** read in `backends/aretomo_alignment.py`, AreTomo2's manual 1.1 and
its release notes for 1.1.3 (`Readme.txt`). Not met on our data yet.

### A-13. AreTomo2 does not start in the lane job

**Kind:** crash · **Evidence:** code for the messages; cause 1 is *general*, not
met

**Error text:**
- `AreTomo exited with code <n>; see <path>/<TS>_fine_aretomo.log`
- `AreTomo executable not found: <path>`
- `no AreTomo executable: pass --aretomo-executable (a lane's [executables]
  aretomo), or put AreTomo2 on PATH`
- `<path> is AreTomo3 <version>; this backend drives AreTomo2 (or AreTomo 1.x),
  whose command line AreTomo3 does not share`

**Causes, common to rare:**
1. The CUDA libraries are not found. AreTomo2 1.1.3 links cuFFT and the CUDA
   runtime when it starts, with no path to them built in (its `makefile11`). A
   lane job takes its environment from the shell it was submitted from (E-9). At
   MDC, load `aretomo/v2-cu12.5` in that shell, after unloading the other
   AreTomo modules (`v1-cu11.7`, `v3-cu12.5`).
2. `$ARETOMO` is unset in the submit shell, or names another program.
3. AreTomo3 is named. It reads movies and mdocs instead of one stack.

**How to confirm:** in a new shell, set up as for submitting:
- `"$ARETOMO" --version` prints `AreTomo2 version 1.1.3`;
- `ldd "$ARETOMO" | grep 'not found'` prints nothing.

**What not to do:** resubmit, or move to another node, before those two checks.

**Applies to:** AreTomo2 1.1.3 at MDC, November 2026.

**Source:** setting up the first AreTomo lane at MDC, 9 November 2026. Its run
on 10 November started; this failure has not been met.

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
- **Q-A7.** If you align lamellae with AreTomo or AreTomo2: which `-AlignZ`,
  `-TiltCor`, `-DarkTol` and `-Patch` do you use, and why?
- **Q-A8.** How do you judge an AreTomo alignment good: by its log, the xz
  slice, the tomogram, the picks?
- **Q-A6.** Have you met an alignment problem that is not here? In a line each:
  what you saw, what it turned out to be, what helped.
