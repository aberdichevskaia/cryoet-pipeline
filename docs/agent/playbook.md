# Starter playbook

Problems we have met, each in the format of
[templates/playbook_entry.md](templates/playbook_entry.md), grouped by step. Each
entry names its source. An entry seen on one dataset says so; it is a lead, not a
rule.

The biologists' entries will join these. The step names, keys, and output folders
are in [pipeline_map.md](pipeline_map.md), and the recipes the entries point to
are in [rerun_recipes.md](rerun_recipes.md).

## Data and ingest

### D-1. The pixel size in the headers is not the published one

**Where it shows:** the manifest's raw pixel size (first line of `show_series.py`)
differs from the pixel size of a published map of the same data. Templates and
averages come out at a slightly wrong scale.

**How to confirm:** compare the manifest with the deposition, the paper, and the
EMDB map of the same data. A map deposited at another pixel size is the strongest
evidence.

**Causes, common to rare:**
1. The headers carry the nominal pixel size, not the calibrated one.

**What to try:** ask a person which value is right before changing anything. For
a frames lane the pixel size is read at `cryoet init`
(`[data] pixel_size_angstrom`), so a different value means a new project.

**What not to do:** override the value silently; combine particles from projects
processed at different pixel sizes.

**When to stop and ask someone:** always — choosing the value is a decision about
the data.

**Source:** EMPIAR-11538. The stack headers say 1.223 Å (nominal); EMD-16721,
the published map of the same data, is at 1.188 Å. The manual etomo processing
used 1.188 and the lanes 1.223. Found on 15 October 2026.

### D-2. Angles read wrongly from file names

**Where it shows:** for each series, `cryoet init` prints the number of tilts and
the angle range it found. The numbers do not match the acquisition: tilts are
missing, the range is shifted, or two series are merged.

**How to confirm:** compare the printout with the tilt scheme on the instrument
card, and with the file list.

**Causes, common to rare** — the traps known from the RyR1 folder (roadmap
section 22):
1. Signed angles such as `+03.0`.
2. Zero tilt written as `-00.0`.
3. Leading zeros in the series id.
4. A tilt number that follows the angle rather than the acquisition order.

**What to try:** test the `frames_pattern` regex against the real file names
before `init` — `re.fullmatch` over the list, then the count and range per
series. Start from the main naming schemes and their patterns in
[data_folders.md](data_folders.md).

**What not to do:** start the lane before the `init` printout has been checked.

**When to stop and ask someone:** when the names are ambiguous, for example when
it is unclear which number is the series and which the tilt. List the readings
rather than pick one.

**Source:** the RyR1 dataset at MDC, named like `tomogram_039_001_-42.0.tif`:
29 tilts, −42° to +42° in 3° steps.

## Motion correction

### M-1. MotionCor3 has no code for the job's GPU

**Where it shows:** the lane job stops in preflight with "MotionCor3 has nothing
this GPU (compute X) can run", listing the binary's cubins and PTX. Jobs from
before that check failed inside MotionCor3 with "no kernel image is available".

**How to confirm:** the job header in the Slurm log shows the GPU and its compute
capability. `cuobjdump -lelf "$MOTIONCOR3"` and `cuobjdump -lptx "$MOTIONCOR3"`
list what the binary carries.

**Causes, common to rare:**
1. The binary was built for other GPU generations. Once it carried compute_70
   only, which the sm_61 nodes (Quadro P600/P6000) cannot run; later it was
   rebuilt for sm_61 alone, which an A100 cannot run.

**What to try, cheap to expensive:**
1. Give the motion stage a GPU the binary supports, in the stage's `sbatch`
   list.
2. Rebuild MotionCor3 with code for every GPU generation the partition has. That
   is a person's task.

**What not to do:** change the motion parameters; rerun on the same partition.

**When to stop and ask someone:** when no partition has a supported GPU.

**Source:** Munich cluster setup; the preflight in `scripts/lane.sbatch`.

## Alignment

### A-1. Coarse alignment QC fails, the rest is fine

**Where it shows:** `qc/<TS>/coarse_alignment/` has status `fail` (largest
residual above 5 preview px).

**How to confirm:** check whether bead tracking and fine alignment of the same
series passed, and with what residual.

**Causes, common to rare:**
1. Cross-correlation struggles with this series (large shifts, low contrast).
   The fine alignment on beads corrects it.

**What to try:** nothing, if fine alignment passes. If it fails too, open the
coarse preview in 3dmod and check the tilt-axis angle.

**What not to do:** stop the chain, or retune coarse alignment, before seeing the
fine alignment.

**When to stop and ask someone:** when fine alignment fails as well.

**Source:** RyR1 series 097, 14 October 2026. Coarse QC failed (rms 26.7, max
144 px); fine alignment passed, like the other series' (0.25–0.29 nm on 15–24
beads).

### A-2. `beadtrack` segfaults while seeding

**Where it shows:** the lane job fails at `generate-fiducial-seed`.
`beadtrack` crashed with a segmentation fault (exit 139) inside `autofidseed`;
the `autofidseed` log is in `fiducials/<TS>/`.

**How to confirm:**
- The same crash happens on a second node (with `--overwrite`).
- The series' coarse QC passed.
- The other series passed the same step.
- The series stands out in some way — here it had the largest coarse shift in
  the set, 0.150 of the image.

**Causes, common to rare:**
1. A crash in IMOD 5.1.12's `beadtrack` at this tracking binning on this series.
   The root cause is unknown.

**What to try:** `tracking_binning` 2 → 3 for that series only, on a config copy
(recipe B). Record the exception in the lane file, and split the stages so that
a later run does not redo the alignment with the lane's value.

**What not to do:** rerun unchanged, since the crash reproduces; change the
binning for every series.

**When to stop and ask someone:** if it crashes at binning 3 too. Then it is a
report for IMOD, with the model and the command.

**Source:** RyR1 series 161, 14 October 2026. With binning 3 the whole CPU stage
passed, with every QC at pass.

### A-3. Patch tracking left every view off by its coarse shift

**Where it shows:** nowhere in the QC. The fine-alignment residuals look normal,
because they are computed on the tracked model, before the transforms are
composed. In the reconstruction, every view sits displaced by its own coarse
shift. This affects patch lanes only. Bead lanes prealign their tracking stack and
are not affected.

**How to confirm:**
- Read the spread of `fiducials/<TS>/<TS>_patch.prexg`, the coarse shifts the
  patch seed used. The error is as large as that spread.
- Compare the final transforms with the ones tracking alone measured; they differ
  by the coarse shift.
- Check the date. A seed made before 27 October 2026 has the error. Since then,
  `track-fiducials` refuses such a seed with "rerun generate-fiducial-seed".

**Causes, common to rare:**
1. A bug in our code, fixed on 27 October 2026. The patch seed tracked on a stack
   that was binned but not prealigned. Fine alignment then composed the coarse
   shift with the tracked one, so it was applied twice.

**What to try:** rerun the patch lanes from `generate-fiducial-seed` with
`--overwrite`, for every series: a config copy whose first stage starts there
([rerun_recipes.md](rerun_recipes.md)). Compare the tomograms and the picks with
the ones from before.

**What not to do:**
- tune alignment parameters;
- take the low residuals as proof that the alignment is right;
- look for the cause in the data;
- call this the cause of weak picking before a rerun shows it.

**When to stop and ask someone:** whenever the transforms the reconstruction uses
disagree with what tracking measured. That is a code problem, and it goes to the
maintainer with the numbers.

**Source:** EMPIAR-11538, ribo-mdc series TS_0306, 27 October 2026. It turned up
while checking how much of the image the patch margin cut: 100 px removed 63% of
the area, and the default is now etomo's 5% of each edge.
- Measured with local IMOD 5.1.12: the final transforms were off from the etomo
  route by a median of 3.4 px, and at most 9 px, at bin8 (33 and 88 Å).
- The spread of the cluster's `.prexg` for TS_0306 was the same size.

## CTF

### C-1. Only the views near zero tilt fit

**Where it shows:** the CTF page of the report. Beyond about ±12° the fitted
resolution sits at the 30 Å end of the fit range, and those views are listed as
unfitted in the QC warning. On the views that do fit, defocus changes steadily
with tilt angle.

**How to confirm:** count the views fitted better than 20 Å. Plot defocus against
tilt for the fitted views; a steady trend (here 0.025–0.035 µm per degree) means
the defocus at high tilt lies outside the range searched per view.

**Causes, common to rare:**
1. The per-view search window (±1000 Å by default) is narrower than the change
   in defocus across the series.
2. Thickness or accumulated dose at high tilt. Then a wider window does not help.

**What to try, cheap to expensive:**
1. The window test on one series (recipe C). If cause 1 is right, more views
   fit, and their defocus continues the same trend.
2. Set the lane's `[ctf] defocus_search_window_angstrom` to the window that
   worked, and rerun from `estimate-ctf`. The CTF model is shared and
   unlabelled, so this needs `--overwrite`, and every step after it reruns. Ask
   first.

**What not to do:** rerun the whole lane before the one-series test; count views
stopped at the 30 Å bound as fitted.

**When to stop and ask someone:** when fewer than half the views fit even at
±10000 Å.

**Source:** RyR1 series 039, October 2026. With windows of ±1000, ±5000 and
±10000 Å, 9, 22 and 26 of 29 views fitted better than 20 Å. Seen on one dataset
so far; 161 had the weakest trend and the widest fit.

### C-2. The CTF step refuses the CTFFIND it found

**Where it shows:** `estimate-ctf` fails. The job header's `ctffind` line names
the program the job found.

**How to confirm:** the header line points at CTFFIND 4 (4.1.14 at MDC).

**Causes, common to rare:**
1. `ctffind` on `PATH` is an old CTFFIND, and the step needs CTFFIND 5.

**What to try:** in the lane, `[executables] ctffind = "$CTFFIND5"`; export
`CTFFIND5` with the path to CTFFIND 5 before submitting. `submit_lanes.py`
refuses a `$VAR` that is not set. Since 28 October 2026 its dry run also asks a
CTFFIND found on `PATH` for its version, and refuses CTFFIND 4 before the queue.

**What not to do:** change CTF parameters.

**When to stop and ask someone:** when CTFFIND 5 is not installed on the site.

**Source:** MDC, job 3670792, October 2026; again on ribo-mdc, job 3781480,
28 October 2026, in a shell that no longer had `ctffind5` on `PATH`.

## Stacks

### S-1. Aligned stacks with blank borders

**Where it shows:** an aligned stack (`aligned/<TS>/`) opened in 3dmod shows part
of every view blank — about 30% on RyR1 at bin4. The CTF fit stack can be
affected too, and fits run on less of the image.

**How to confirm:** compare the view size and content with the raw stack, and
check whether the stack was built before 16 October 2026.

**Causes, common to rare:**
1. `newstack -size` cropped the output. This was a bug in our code, fixed on
   16 October 2026.

**What to try:** rebuild the affected stack with `--overwrite`, then rerun what was
computed from it.

**What not to do:** work around wrong geometry with config values.

**When to stop and ask someone:** whenever an output is geometrically wrong while
its inputs look right. Suspect the code, and report it with the command, the
input and output shapes, and a view of the slices.

**Source:** RyR1 bin4 lane; fixed 16 October 2026.

## Picking

### P-1. Template matching runs out of GPU memory

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

**What not to do:** shrink the angular search or the box to fit the memory. That
changes the result.

**When to stop and ask someone:** when no split fits.

**Source:** HIV-1 Gag (EMPIAR-10164) at bin8, 8 October 2026: `2,2,1` fitted on a
20 GB slice.

### P-2. The score maps do not see the specimen

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
tomograms have the opposite hand to the manual etomo reconstruction, but flipping
one did not change the matcher's peaks, so handedness is not the cause here.

**When to stop and ask someone:** if the share stays at noise level after the
ablation. The mid-band of these tomograms was weaker than in a manual etomo
reconstruction, and alignment is suspected; that question is open.

**Source:** EMPIAR-11538 ribosome lanes at MDC, 15–16 October 2026 (jobs 3640899
and 3643100). The ribosome lanes now use low-pass 25 Å with
`random_phase_correction = false` and `ctf_weighting = false`.

## Environment

### E-1. An install dies with "No space left on device"

**Where it shows:** `pip` or `micromamba` fails with Errno 28 while unpacking a
large package such as torch.

**How to confirm:** `df -h /tmp`. At MDC it is 100 MB.

**Causes, common to rare:**
1. The temporary directory is too small.

**What to try:** set `TMPDIR` to a directory on the large filesystem, then
install again.

**What not to do:** read it as a network failure. It was first misread as one.

**When to stop and ask someone:** when no writable large filesystem is known.

**Source:** MDC install, 13 October 2026.
