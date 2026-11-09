# CTF: `build-ctf-stack`, `estimate-ctf`, `correct-ctf`

> **Draft for review, 7 November 2026.**
>
> - Mark each entry by its ID: `C-4 ✓` if it is right, `C-5 ✗` if it is wrong
>   (say why), `C-6 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

The steps, their keys and their outputs are in
[pipeline_map.md](../pipeline_map.md), steps 7, 8 and 10. Safe reruns are in
[rerun_recipes.md](../rerun_recipes.md).

- **Tools.** CTFFIND 5 fits the defocus of every view; IMOD's `ctfphaseflip`
  corrects the views. At MDC on 5 November 2026 these were CTFFIND 5.0.2 and
  IMOD 5.1.12.
- **How the fit runs.**
  1. The view nearest 0° is fitted over the lane's `min_defocus_angstrom` to
     `max_defocus_angstrom` (5000–50000 Å if unset).
  2. Every view is then fitted within `defocus_search_window_angstrom` of that
     first fit (±1000 Å if unset).
  3. A fit uses spacings from 30 Å down to the stack's Nyquist.
- **Shared by every lane.** The CTF fit stack and the fit carry no lane label,
  so every lane of a project uses them. Changing them needs `--overwrite`, and
  then every lane reruns from `correct-ctf`. A one-series experiment goes under
  a variant label instead (recipe C).

Each entry states its **kind** and its **evidence**:

- **Kinds:**
  - crash: the job stops;
  - refused: the dry run stops the submission;
  - warning: the QC or the report warns;
  - silent: nothing warns;
  - false alarm;
  - reading results;
  - slow.
- **Evidence:**
  - measured: seen on our data and checked;
  - seen: met on our data, not checked further;
  - code: read in our code, not met in a run;
  - manual: in the tool's documentation, at the version named;
  - general: general knowledge or a colleague's practice, not checked here.

## How a healthy run looks

| Data | Fitted at (Nyquist) | Views | Defocus: mean, range | Views without rings | `estimate-ctf` |
|---|---|---|---|---|---|
| HIV-1 Gag, EMPIAR-10164, TS_01 (Munich) | 2.7 Å/px (5.4 Å) | 40 | 3.93 µm, 3.77–4.38 | 0 | 47 s |
| EMPIAR-11538 ribosomes at MDC, 5 series | 2.45 Å/px (4.9 Å) | 50–60 | 2.66–4.49 µm by series; 0.3–0.4 µm within one | 0 or 1 per series | — |
| RyR1 at MDC, 5 series | 3.36 Å/px (6.7 Å) | 29 | 4.0–5.6 µm on one series | — | 29–35 s |

- **Agreement between the passes.** On all six records with a QC file above,
  the first fit (`hint_defocus_um` in the estimation QC) lies within 0.1 µm of
  the series' mean defocus (`mean_defocus_um`).
- **Astigmatism.** The series' mean astigmatism is 0.03–0.07 µm.
- **How far fits reach.** On RyR1, fitted views reached 7–9 Å near 0° and
  13–17 Å at high tilt. A view without Thon rings reports 29.7–29.8 Å, at the
  30 Å end of the range.
- **Run times per series.** `build-ctf-stack` 15–40 s, `estimate-ctf` 30–50 s,
  `correct-ctf` 10–25 s.
- **The report's CTF page.**
  - One point per tilt: its defocus, and how far its fit reached.
  - The fit limit worsens with tilt and with dose.
  - Points are coloured by dose when the mdoc records it. Yellow tilts kept an
    interpolated defocus; red tilts were left out of the tomogram.
  - Below the plots are CTFFIND's own diagnostic images for three tilts.
- **`correct-ctf` QC.** It always ends in `warning` in our lanes; C-13 says
  why that is fine.

## Where to look first

1. `python3 scripts/show_failure.py <lane config> --step estimate-ctf` (or
   `--step correct-ctf`) prints the failing command, its error lines and the end
   of the log.
2. The step's own log, `ctf/<TS>/<TS>_ctffind5.log`, holds every program the
   step ran, the answers it sent CTFFIND, and what CTFFIND printed. A generic
   `CTFFIND failed with exit code …` is explained there. The first CTFFIND line
   names the program that ran:
   `grep -m1 '^\$ [^ ]*ctffind' ctf/<TS>/<TS>_ctffind5.log`.
3. After a failed fit, `ctf/<TS>/.<TS>-ctffind-*/` stays behind. It holds the
   extracted view nearest 0° and the exact answers sent.
4. The QC records are in `qc/<TS>/ctf/`:
   - `<TS>_ctffind5_estimation_qc.json` holds the first fit, the mean defocus
     and the views without rings by z;
   - `<TS>_imod-ctfphaseflip_<lane>_correction_qc.json` holds the views left
     out.
5. The report's CTF page and its warnings page.

## Already checked by the code

Nobody needs to check these by hand:

- **An old CTFFIND.** The dry run refuses a CTFFIND older than 5, whether the
  lane names it or it is first on `PATH`. The step refuses one too.
- **A broken named path.** The dry run refuses a named CTFFIND that is a folder,
  a missing file or not executable (since 30 October 2026). The job's preflight
  checks the same again, for the CTFFIND the step will run, whether the lane
  names it or it is on `PATH` (since 8 November 2026; C-5).
- **The answers given to CTFFIND.** Each answer is checked against the question
  CTFFIND echoes (C-12).
- **The optics at correction.** `correct-ctf` uses the voltage, Cs and amplitude
  contrast the fit used, and refuses others. IMOD's manual asks for exactly
  that: "The value should be the same as was used when detecting the defocus"
  (`ctfphaseflip`, IMOD 5.1.12).
- **The views.** `correct-ctf` refuses a CTF model that covers other views than
  the stack (C-10).
- **The pixel size.** `ctfphaseflip` is given the pixel size of the stack it
  corrects.
- **Views without Thon rings.** Such a view never keeps its own CTFFIND defocus
  (since 18 October 2026; C-7).

## Checks before trusting the result

The QC does not show problems of the silent kind, so look at these once per
series:

1. **The fit's Nyquist.** The subtitle of the CTF page gives the pixel size and
   Nyquist of the fit. A fit reaches no finer than Nyquist, and the lane rule is
   finer than about 6 Å.
2. **The first fit agrees with the rest.** `hint_defocus_um` should lie within
   about 0.1 µm of `mean_defocus_um`. If not, see C-6.
3. **The defocus is smooth.** Defocus moves smoothly from view to view. One view
   far from its neighbours is a bad fit, not the sample. *General.*
4. **Small, even astigmatism.** It should be small and similar across views. One
   long bar on the CTF page means C-8.
5. **The left-out views are the highest tilts.** See C-7.
6. **The pixel size is right.** If the pixel size is off by a few percent, every
   defocus is off by about twice that, because the fitted defocus scales with
   the square of the pixel size. This is derived, not measured. See D-1 in the
   data file.
7. **No later fit.** In a project with several lanes, the report's warnings page
   should not say that the CTF fit was written again after use (C-11).

## Problems

### C-1. Only the views near zero tilt fit

**Kind:** warning · **Evidence:** measured

**Error text** (estimation QC, since 18 October 2026):
`<n> of <m> views stopped at the low end of the fit range (27 A or coarser of
30), so CTFFIND found no Thon rings on them: <angles>`

**Where it shows:** the CTF page of the report. Beyond about ±12° the fit limit
sits at the 30 Å end, and those views are yellow or red. On the views that do
fit, defocus changes steadily with tilt angle.

**How to confirm:**
- Count the views fitted better than 20 Å.
- Plot defocus against tilt for the fitted views. A steady trend (here
  0.025–0.035 µm per degree) means the defocus at high tilt lies outside the
  range searched per view.
- A fit can land a little past its window, so a few such views fitting does not
  rule this out.

**Causes, common to rare:**
1. The per-view window (`defocus_search_window_angstrom`, ±1000 Å by default)
   is narrower than the change in defocus across the series. A defocus that
   moves with sin(tilt) is what an image centre off the tilt axis gives.
2. Thickness or accumulated dose at high tilt. In that case a wider window does
   not help.

**What to try, cheap to expensive:**
1. Run the window test on one series (recipe C):
   `python3 scripts/submit_recipe.py ctf-window <lane config> --series <TS> --dry-run`.
   - It fits the series four ways under labels of its own: ±1000, ±5000 and
     ±10000 Å, and ±5000 Å with CTFFIND5's sample tilt. It touches nothing of
     the lane's.
   - Compare the fits with
     `python3 scripts/ctf_fit_table.py outputs/<project> <TS> ctfw1 ctfw5 ctfw10 ctfw5-tilt`.
   - If cause 1 is right, more views fit, and their defocus continues the same
     trend.
2. Set `[ctf] defocus_search_window_angstrom` in the lane to the window that
   worked, and rerun from `estimate-ctf`. The fit is shared by every lane, so
   this needs `--overwrite`, and every lane reruns from `correct-ctf` (C-11).
   Ask first.

**What not to do:**
- Rerun the whole lane before the one-series test.
- Widen `min_defocus_angstrom` and `max_defocus_angstrom` instead of the window.
  They bound only the first fit.
- Count views stopped at the 30 Å end as fitted.

**When to stop and ask someone:** when fewer than half the views fit even at
±10000 Å.

**Applies to:** every version. The QC warning exists since 18 October 2026, the
window key since 16 October.

**Source:** RyR1 series 039 at MDC, October 2026. With windows of ±1000, ±5000
and ±10000 Å, 9, 22 and 26 of 29 views fitted better than 20 Å. ryr1-bin8 now
searches ±10000 Å. Seen on one dataset so far; 161 had the weakest trend and the
widest fit.

### C-2. The step refuses the CTFFIND it found

**Kind:** crash, or refused · **Evidence:** measured

**Error text:**
- in the job: `<path> is CTFFIND <version>, and this backend answers CTFFIND
  5's questions; put ctffind5 on PATH ahead of it, or pass --ctffind-executable`
- in the dry run, since 28 October 2026: `estimate-ctf would run <path>, which
  is CTFFIND <version>, and its backend answers CTFFIND 5's questions: name
  CTFFIND 5 in [executables] ctffind ("$CTFFIND5", as ryr1-bin8 does), or put
  ctffind5 on PATH ahead of it`

**Where it shows:** `estimate-ctf` fails. Since 28 October 2026 the dry run
refuses the lane before it is queued.

**How to confirm:** the version in the message. The job header's `ctffind` line
names the program the job found: the lane's own, marked "(the lane's
[executables] ctffind)", or else the one on `PATH`. In logs from before
8 November 2026 it always named the one on `PATH` (C-14).

**Causes, common to rare:**
1. The lane names no CTFFIND, and the `ctffind` first on `PATH` is CTFFIND 4.
   The step needs CTFFIND 5, which asks different questions.

**What to try:**
1. Name CTFFIND 5 in the lane: `[executables] ctffind = "$CTFFIND5"`.
2. Export `CTFFIND5` with the path to the CTFFIND 5 program before submitting
   (C-3, C-4).
3. Resume the lane (recipe A).

**What not to do:** change CTF parameters.

**When to stop and ask someone:** when CTFFIND 5 is not installed on the site.

**Applies to:** every version. Refused at submission since 28 October 2026.

**Source:** MDC, job 3670792, October 2026: `ctffind` on `PATH` was CTFFIND
4.1.14. It happened again on ribo-mdc, job 3781480, 28 October 2026.

### C-3. The lane's CTFFIND is a folder, a missing file, or not a program

**Kind:** refused · **Evidence:** measured

**Error text** (dry run, since 30 October 2026):
- `[executables] ctffind "<value>": <path> is a directory, and estimate-ctf
  would stop on it; name the CTFFIND program in it instead, such as <program>`
- `[executables] ctffind "<value>": no file <path>, and estimate-ctf would stop
  on it`
- `[executables] ctffind "<value>": <path> is not executable, and estimate-ctf
  would stop on it`
- in a job from before that date: `CTFFIND executable not found: <path> (from
  --ctffind-executable)`

**Where it shows:** the dry run of `submit_lanes.py` or `submit_recipe.py`.

**How to confirm:** run `echo "$CTFFIND5"`, then `ls -l` on what it prints.

**Causes, common to rare:**
1. The variable names the folder CTFFIND was installed in, not the program. At
   MDC the program is `cisTEM/bin/ctffind` inside the `ctffind-5.0.2` folder.
   It is called `ctffind`, not `ctffind5`.

**What to try:** set the variable to the program, as the message suggests, and
run the dry run again.

**What not to do:** write a site path into the lane file. The lane names a
variable so that each site can set its own.

**When to stop and ask someone:** when nobody knows where the site's CTFFIND 5
is.

**Applies to:** refused at submission since 30 October 2026. Before that, the
job stopped about 11 minutes in.

**Source:** ribo-mdc at MDC, job 3789938, 29 October 2026: `CTFFIND5` named the
install folder.

### C-4. A tool variable the lane names is not set

**Kind:** refused · **Evidence:** measured

**Error text** (dry run): `$CTFFIND5 is not set (used by stage '<stage>')`.
The same message comes for `$IMOD_DIR`, `$MOTIONCOR3`, `$MEMBRAIN_EXE`,
`$MEMBRAIN_CKPT` and `$PYTOMO_MP_DIR`.

**Where it shows:** the dry run.

**How to confirm:** run `echo "$CTFFIND5"` in the shell you submit from.

**Causes, common to rare:**
1. The shell you submit from does not have the variables. A job takes the
   environment of the shell that submits it and reads no `.bashrc` of its own.
   A `.bashrc` that sets the tool variables only on compute hosts leaves a
   login shell without them.

**What to try:** set the variables in that shell and submit from it. One way is
to source the group's tool file, as the README's one-time setup describes.

**What not to do:** write site paths into the lane file.

**When to stop and ask someone:** when nobody knows where the site's tools are.

**Applies to:** every version.

**Source:** MDC, 29 October and 5 November 2026. In a login shell `IMOD_DIR`,
`CTFFIND5`, `PYTOMO_MP_DIR`, `MOTIONCOR3` and `MEMBRAIN_EXE` were all unset.

### C-5. The job stops in its preflight over CTFFIND

**Kind:** crash · **Evidence:** code

**Error text** (the job's preflight, since 8 November 2026):
- `preflight: no ctffind5 or ctffind on PATH, and the lane names no
  [executables] ctffind`
- `preflight: the lane's [executables] ctffind is '<path>' here: a variable in
  it is not set in the environment the job was submitted with`
- `preflight: the lane's [executables] ctffind is the directory '<path>'; name
  the CTFFIND program in it instead`
- `preflight: the lane's [executables] ctffind is '<path>' here, which is not an
  executable file`
- before 8 November 2026: `preflight: no ctffind on PATH`

**Where it shows:** the stage that runs `estimate-ctf` stops before its first
step.

**How to confirm:** the job header's `ctffind` line names the program the job
looked for.

**Causes, common to rare:**
1. The job was submitted by hand, or from another shell than the one the dry
   run checked. `submit_lanes.py` refuses all of these before the queue (C-2,
   C-3, C-4); the preflight catches the rest.
2. Before 8 November 2026 the preflight looked at `PATH` only. A lane that names
   its CTFFIND stopped there on a site with no CTFFIND on `PATH`, although the
   step would have run the lane's.

**What to try:** fix the CTFFIND as C-2, C-3 or C-4 says, then submit through
`submit_lanes.py`, whose dry run checks all of it.

**Applies to:** the messages above since 8 November 2026.

**Source:** read in `scripts/lane.sbatch` and tested in
`tests/test_lane_sbatch.py`; not met in a job.

### C-6. The views near zero tilt do not fit either, or no view fits

**Kind:** warning, or crash · **Evidence:** code

**Error text** (crash, when no view fits): `CTFFIND found Thon rings on none of
the views: every fit stopped at the low end of its range, so there is no
defocus to build a model from`

**Where it shows:** the QC warning lists views near 0° too, unlike C-1. Or
`estimate-ctf` stops with the message above.

**How to confirm:**
- Compare `hint_defocus_um` in the estimation QC with `mean_defocus_um` and
  with the defocus in the mdoc. `hint_defocus_um` is the first fit, of the view
  nearest 0°. On good series they agree within about 0.1 µm (see the table at
  the top).
- Look at CTFFIND's diagnostic images on the CTF page, and at the view nearest
  0° in 3dmod.

**Causes, common to rare:**
1. The first fit settled on a wrong defocus, and every view was then searched
   around it. The code does not check how well that first fit went.
2. The lane's `min_defocus_angstrom`…`max_defocus_angstrom` does not contain
   the real defocus, so the first fit could not find it.
3. The view nearest 0° is bad: dark, contaminated, or spoiled in motion
   correction.
4. A wrong pixel size, voltage or Cs.
5. The sample shows no Thon rings at all.

**What to try, cheap to expensive:**
1. Set `min_defocus_angstrom` and `max_defocus_angstrom` around the defocus the
   microscope recorded, 0.5–1 µm to spare on each side. Do it on a config copy
   for one series, from `estimate-ctf`, with `--overwrite` (recipe B). The fit
   is shared by every lane, but a fit that found nothing is no loss.
2. The step can skip the first fit when it is given the defocus
   (`--ctf-expected-defocus-um`), but lanes have no key for it. Ask the
   maintainer.

**What not to do:** widen the per-view window for this. The window is centred
on the first fit.

**When to stop and ask someone:** when the view nearest 0° shows no Thon rings
in its power spectrum.

**Applies to:** every version.

**Source:** read in `backends/ctf.py`, not met in a run. The causes are
*general*.

### C-7. Some views are left out of the tomogram

**Kind:** reading results · **Evidence:** measured

**Error text:**
- estimation QC: `… With no more than <n>, correct-ctf leaves them out of the
  corrected stack and so of the tomogram; the model gives them a defocus
  interpolated from the fitted views all the same`
- correction QC: `left out <n> of <m> views that the CTF model marks as
  unfitted (z <z>); the corrected stack, and the tomogram built from it, has <k>`

**Where it shows:** both CTF QC records, and red points on the report's CTF
page.

**What it means:**
- A view counts as unfitted when its fit stopped at 27 Å or coarser, the 30 Å
  end of the range. Its own CTFFIND defocus means nothing.
- Such a view gets a defocus interpolated from the fitted views: linear between
  fitted neighbours, and past the last fitted view along a line through the
  four outermost ones.
- With at most `max_excluded_views` such views (5 by default), `correct-ctf`
  leaves them out. With more, they all stay in with the interpolated defocus.
- The QC warns either way.

**How to confirm it is harmless:** the left-out views are the highest tilts. A
left-out view in the middle of the range is worth opening in 3dmod (C-6,
cause 3).

**What to try:**
- To keep all the views, set `[ctf] max_excluded_views = 0`.
- To leave out more views, raise it.
- Either way `estimate-ctf` decides it, so rerun from there with
  `--overwrite`. Every lane reruns from `correct-ctf`.

**What not to do:** count an interpolated view as fitted.

**When to stop and ask someone:** when more than a quarter of the views have no
rings. That is C-1 or C-6.

**Applies to:** since 18 October 2026. Before that, unfitted views were
corrected with their own CTFFIND defocus.

**Source:**
- EMPIAR-11538 ribosomes at MDC, October 2026: one view in each of three of the
  five series, at +44°, −51° and −62°, of 50–60 views.
- On RyR1, an unfitted view's own fit said 4.10 µm where the fitted views'
  trend puts about 5.8.

### C-8. One view's astigmatism runs away

**Kind:** warning, seen on the report · **Evidence:** seen

**Where it shows:** on the report's CTF page, one view's bar is much longer than
its neighbours'. The bar spans the two defocus values of an astigmatic fit.

**How to confirm:** compare that view's two defocus values with its neighbours'.
They are in CTFFIND's summary, `ctf/<TS>/<TS>_ctffind5_ctffind.txt`. Our good
series have a mean astigmatism of 0.03–0.07 µm.

**Causes, common to rare:**
1. With little signal, the fit trades defocus for astigmatism. *General.*

**What to try:** fit the series again with astigmatism restrained.
`estimate-ctf` takes `--ctffind-restrain-astigmatism-angstrom` (1000 Å was
used), but lanes have no key for it. Ask the maintainer.

**When to stop and ask someone:** when several views run away.

**Applies to:** every version.

**Source:** EMPIAR-11538, series TS_0487, Munich, October 2026. View 5's
astigmatism diverged, and the series was fitted again with a 1000 Å restraint.
Whether the restraint changed the tomogram was not measured.

### C-9. The sample-tilt fit takes hours

**Kind:** slow · **Evidence:** measured, with the time inferred

**Where it shows:** with `[ctf] determine_sample_tilt = true`, `estimate-ctf`
runs for hours and can reach the stage's time limit.

**How to confirm:** check the step's time with `scripts/show_timings.py`.
Without the tilt fit, a series of 29–40 views takes 30–50 s.

**Causes, common to rare:**
1. CTFFIND5 fits each view's tilt from the spectra of its tiles, which is far
   slower.

**What to try:** leave it off unless the fits need it. On RyR1 it sharpened the
fits (6.4–7.4 Å against 7–13 Å) but made no more views fit than the same search
without it. If it is needed, give the stage more time.

**What not to do:** switch it on for a whole lane before timing it on one
series.

**Applies to:** CTFFIND 5.0.2.

**Source:** RyR1 at MDC, 16 October 2026. One series took about two hours with
it, against a few seconds per view without it. The time is inferred from the
job's total time.

### C-10. `correct-ctf` refuses: the CTF model covers other views

**Kind:** crash · **Evidence:** code; cause 2 measured

**Error text:**
- `CTF model covers <n> views but <stack> has <m>`
- `CTF model and <stack> disagree on view <k>: <a> versus <b> degrees. The model
  covers different views, so its defocus values do not line up with this stack`
- `aligned stack artifact shape <recorded> does not match <found>`

**Where it shows:** `correct-ctf` fails.

**Causes, common to rare:**
1. The alignment was redone after the fit, with views excluded or the series
   aligned again, while the CTF fit stack and the fit were not redone.
2. The registry lost a record because several jobs of one project wrote it at
   once. This gives the third message. It was fixed at MDC on 29 October 2026.

**What to try:**
1. For cause 1, rerun from `build-ctf-stack` with `--overwrite`. It is shared,
   so every lane reruns from `correct-ctf`.
2. If the third message comes back, report it: the registry lock is not holding.

**What not to do:** rerun `correct-ctf` alone.

**Applies to:** every version. Cause 2 only before 29 October 2026 at MDC.

**Source:** cause 2 hit ryr1-bin4 series 096, 097 and 161 at MDC on 29 October
2026.

### C-11. The report says the CTF fit was written again after use

**Kind:** warning, on the report · **Evidence:** measured

**Error text** (the report's warnings page, since 29 October 2026):
`Written again after use: the CTF fit, dated <date>, is newer than <what was
made from it> made from it, so that was made from an earlier version than the
one this report reads.`

**Causes, common to rare:**
1. The CTF fit is shared by the lanes of a project. Another lane, or a rerun,
   fitted it again after this lane had corrected its stacks. This lane's
   tomograms were made with the earlier fit, and the report draws the new one.

**What to try:** decide which fit is wanted. To bring this lane in line with
the current fit, rerun it from `correct-ctf`. Use a config copy whose first
stage starts there, with `--overwrite`.

**What not to do:** read this lane's tomograms against the report's CTF page as
if the two matched.

**Applies to:** every version. The report has warned since 29 October 2026.

**Source:** RyR1 at MDC. ryr1-bin8 fitted the shared CTF again on 18 October
2026. ryr1-bin4 had corrected its stacks with the earlier fit on 14 October, and
its report drew the new fit beside the old tomograms.

### C-12. CTFFIND asked a question the step did not expect

**Kind:** crash, *for the maintainer* · **Evidence:** code

**Error text:**
- `CTFFIND asked '<question>' as question <n> and got '<answer>', which this
  adapter meant for '<expected>': the question sequence differs from the
  expected one, so every later answer is displaced too`
- `CTFFIND read '<value>' for '<question>' but was sent '<answer>'`

**Causes, common to rare:**
1. A CTFFIND whose questions differ from those of 5.0.2: another 5.x release,
   or a build with other options. CTFFIND has no command-line parameters. The
   step answers its questions in order and checks each answer against what
   CTFFIND echoes.

**What to try:** run the CTFFIND the lanes have used so far (5.0.2). Report the
message and the version to the maintainer.

**What not to do:** change CTF parameters to get past it.

**Source:** read in `backends/ctf.py`, not met in a run.

## Looks wrong, is fine

### C-13. `correct-ctf` always ends in "warning"

**Kind:** false alarm · **Evidence:** measured

**Error text** (correction QC):
- `CTF model was fitted on '<TS>:aligned_tilt_stack:final', not on
  '<TS>:aligned_tilt_stack:final:<lane>'; defocus transfers between binnings
  but the fit was not made on this stack`
- `fitted at <a> angstrom/pixel and applied at <b> angstrom/pixel`

**Why it is fine:**
- The fit is made on the finer CTF stack (`[binning] ctf_fit`) and applied to
  the lane's stack (`[binning] tomogram`).
- Defocus is a physical quantity, so it transfers between binnings.
- `ctfphaseflip` is given the pixel size of the stack it corrects.
- The code refuses a model that covers other views (C-10).
- Every lane so far fits at a finer binning than its tomogram, so every
  correction QC says this.

**When it is not fine:** this warning alone never is. Read the other warnings
in the same record.

**Source:** every correction QC of the EMPIAR-10164 and EMPIAR-11538 lanes,
October 2026.

### C-14. An older job's header names another CTFFIND than the one that ran

**Kind:** false alarm · **Evidence:** measured

**Where it shows:** in a Slurm log from before 8 November 2026, the header line
`ctffind <path>` names an older CTFFIND. Yet the lane names CTFFIND 5, and
`estimate-ctf` passed.

**Why it is fine:** until 8 November 2026 the header line looked at `PATH`
only, not at the lane's `[executables] ctffind`. Since then it names the lane's,
marked "(the lane's [executables] ctffind)". The program that ran is in two
places:
- the first CTFFIND line of the step's own log:
  `grep -m1 '^\$ [^ ]*ctffind' ctf/<TS>/<TS>_ctffind5.log`;
- the model's record in `artifacts.json`, whose `software_versions` give its
  version.

**When it is not fine:** when the lane names no CTFFIND. The header line is
then the program the step found (C-2).

**Source:** MDC, October 2026. Jobs that ran CTFFIND 5 through `$CTFFIND5` showed
an older `ctffind` from `PATH` in their header.

## Reading results

### C-15. Checking that phase flipping worked

**Kind:** reading results · **Evidence:** measured

**The idea:** two tomograms at different defocus have CTFs of opposite sign in
some resolution bands. Without phase flipping, an FSC between averages from the
two would be negative in those bands. With it, the FSC is positive there.

**How:**
1. Find the bands where the two series' CTFs have opposite signs, from their
   mean defocus in the estimation QC.
2. Read the FSC between the two per-tomogram averages in those bands.

**Source:** HIV-1 Gag (EMPIAR-10164), October 2026. TS_01 was at 3.93 µm and
TS_43 at 2.56 µm.
- In 28–23, 20–16 and 16–14 Å their CTFs have opposite signs.
- The FSC there was +0.6, +0.3–0.4 and +0.1–0.16.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-C1.** At what tilt do you stop expecting per-view fits on your samples:
  lamellae, whole cells, purified particles on carbon? How far should a fit
  reach at 0° on a good series?
- **Q-C2.** C-1 names thickness or dose as the other cause of failed fits at
  high tilt. How do you tell it from a search that is too narrow, and what do
  you do then?
- **Q-C3.** When some views show no Thon rings, do you leave them out of the
  tomogram, keep them with an interpolated defocus, or do something else? Is
  "leave out at most 5" a sensible default (C-7)?
- **Q-C4.** How do you choose the defocus range to search? How far is the
  measured defocus from the target in your sessions? On HIV-1 Gag the target
  was −4 µm, the mdoc read about 2.7 µm, and the fits gave 2.6–3.9 µm.
- **Q-C5.** Do you restrain astigmatism when fitting tilt series, and to what
  value (C-8)?
- **Q-C6.** Do you have phase-plate data? The lanes have no setting for it.
- **Q-C7.** How do you check that CTF correction worked on your data? Do you use
  the check in C-15?
- **Q-C8.** Have you met a CTF problem that is not here? In a line each: what
  you saw, what it turned out to be, what helped.
