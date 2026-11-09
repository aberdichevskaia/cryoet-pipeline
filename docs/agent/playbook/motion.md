# Motion correction: `prepare-tilt-series`

> **Draft for review, 8 November 2026.**
>
> - Mark each entry by its ID: `M-4 ✓` if it is right, `M-5 ✗` if it is wrong
>   (say why), `M-6 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained at the top of [ctf.md](ctf.md). The
step, its keys and its outputs are in [pipeline_map.md](../pipeline_map.md),
step 1. The backend is described in
[motioncor3_backend.md](../../motioncor3_backend.md).

- **What the step does.** It runs only in frames lanes. MotionCor3 aligns and
  sums the frames of each tilt's movie, on a GPU, then the step builds the tilt
  stack from the sums.
- **Versions.** At MDC on 5 November 2026 this was MotionCor3 1.2.4. The run
  records the version MotionCor3 states for `--version`.
- **Patches.** `[motion] patch_x` and `patch_y` set MotionCor3's patch grid
  (`-Patch`). With more than one patch, the step judges how far the patches
  moved against the whole frame (M-2).
- **EER movies.** MotionCor3 decodes them. The frames are summed in groups
  before alignment (M-7), and a `.gain` reference is turned upside down (M-3).

## How a healthy run looks

- **Run time per series.** 131–160 s for RyR1 (29 K3 TIFF movies) and
  288–328 s for HIV-1 Gag (EMPIAR-10164).
- **The motion QC** (`qc/<TS>/motion/<TS>_motion_qc.json`) is `pass`. Most
  tilts have no patch beyond the 1.5 nm tolerance. On EMPIAR-11830, 36 of 41
  tilts had at most one such patch, and their median patch shifts were at most
  1.25 px.
- **The sums** look like images of the specimen, with no straight seams and no
  pattern that stays fixed from tilt to tilt.
- **The job header** names the GPU. The preflight line reads `preflight   ok`.

## Where to look first

1. **The job's preflight**, in the Slurm log. It checks that `$MOTIONCOR3` is
   executable, that its libraries load, and that it has code for the job's GPU.
2. **`python3 scripts/show_failure.py <lane config> --step prepare-tilt-series`.**
3. **MotionCor3's own logs**, in `logs/motioncor3/<TS>/`, and its working files
   in `motion/motioncor3/<TS>/`. A failure reads `MotionCor3 failed for <movie>
   with exit code <n>: …; see <log>`.
4. **The motion QC** and the corrected sums in `corrected/<TS>/`. Open the
   darkest and the highest tilts in 3dmod.

## Already checked by the code

- **Before the job runs anything.** The preflight checks that MotionCor3 is
  executable, loads all its libraries, and has code for the job's GPU (M-1,
  M-5).
- **The gain is loaded.** MotionCor3's "Gain reference not found" stops the
  step (M-4).
- **The output.** A sum that is not one 2D image of the frame's shape, or that
  holds non-finite pixels, stops the step.
- **The frame count.** A movie whose frame count differs from the manifest is
  refused before MotionCor3 runs.
- **EER groups.** The step never asks MotionCor3 for EER groups that would make
  it write past its own array (M-7).

## Checks before trusting the result

1. **The motion QC**, before alignment results are read. A tilt cut into tiles
   breaks alignment later, and the alignment QC then points at the wrong step
   (M-2).
2. **The darkest and the highest tilts**, in 3dmod: no straight seams, no
   repeated content.
3. **A fixed pattern across tilts**, the same spots or stripes in every sum,
   means the gain or its orientation (M-3, D-9 in [data.md](data.md)).
   *General.*

## Problems

### M-1. MotionCor3 has no code for the job's GPU

**Kind:** crash · **Evidence:** measured

**Error text:**
- preflight: `MotionCor3 has nothing this GPU (compute <cap>) can run. cubins:
  <list>ptx: <list> Rebuild it, or give the motion stage another GPU in the
  config's [[run.stage]].`
- MotionCor3 itself, in jobs from before that check: `no kernel image is
  available`

**Where it shows:** the lane job stops in its preflight.

**How to confirm:** the job header in the Slurm log shows the GPU and its
compute capability. `cuobjdump -lelf "$MOTIONCOR3"` and `cuobjdump -lptx
"$MOTIONCOR3"` list what the binary carries.

**Causes, common to rare:**
1. The binary was built for other GPU generations. Once it carried compute_70
   only, which the sm_61 nodes (Quadro P600/P6000) cannot run. Later it was
   rebuilt for sm_61 alone, which an A100 cannot run.

**What to try, cheap to expensive:**
1. Give the motion stage a GPU the binary supports, in the stage's `sbatch`
   list.
2. Rebuild MotionCor3 with code for every GPU generation the partition has.
   That is a person's task.

**What not to do:**
- Change the motion parameters.
- Rerun on the same partition.

**When to stop and ask someone:** when no partition has a supported GPU.

**Applies to:** every version. The preflight check exists since 7 October 2026.

**Source:** Munich cluster setup; the preflight in `scripts/lane.sbatch`.

### M-2. Patch alignment cut the dark tilts into displaced tiles

**Kind:** warning · **Evidence:** measured

**Error text** (motion QC): `patch alignment failed on <n> of <m> tilts (<angles>
deg; <frames> frames each): at least half of each one's patches moved more
than 1.5 nm (<px> px) against the whole frame (<medians>), so those sums are
cut into displaced tiles. [motion] patch_x = patch_y = 1 aligns whole frames
only`

**Where it shows:**
- `scripts/show_project.py` marks `prepare-tilt-series` as `fail`, and the
  motion QC names the tilts.
- Further on, coarse alignment jumps between neighbouring views at the ends of
  the series, the tilt axis drifts, and fine alignment fails.
- The views look dark. Their sums show straight seams on the patch grid, every
  819 px of a 4096 px frame at 5 × 5, with content cut and repeated across
  them.

**How to confirm:**
- The QC file. A failed tilt has most of its patches beyond the tolerance, by
  hundreds of pixels; the others have none, one or two.
- To see it in the sums themselves, `scripts/mc3_patch_check.sbatch` reruns
  chosen tilts with the recorded command and with `-Patch 1 1`, and measures
  how far each tile moved. It is one GPU job of a few minutes, and a person
  submits it.

**Causes, common to rare:**
1. Too little in a patch to align: the darkest tilts (high tilt, thick
   lamella) at the dose of one patch and one frame. On EMPIAR-11830 these tilts
   also had the fewest frames (3–4, since EER groups follow the dose), and
   MotionCor3 aligns patches on groups of 4 frames. Which of the two matters is
   not known ([motioncor3_backend.md](../../motioncor3_backend.md), "Patch
   alignment QC").

**What to try, cheap to expensive:**
1. Set `[motion] patch_x = 1` and `patch_y = 1`, and rerun from the motion
   stage with `--overwrite`. If this works:
   - there is no motion QC, since with whole frames there is nothing to judge;
   - the sums have no seams;
   - coarse alignment no longer jumps at the ends of the series.

**What not to do:** raise `[align] min_std_ratio` to drop the dark views before
reading the motion QC. On a frames lane the defect can sit in the sums, not in
the darkness. The 0.84 in `ribo-mdc.toml` was a stack lane, with no motion
correction of ours.

**When to stop and ask someone:**
- when alignment still fails after the rerun with whole frames;
- when a tilt fails the motion QC with fewer patches.

**Applies to:** the motion QC exists since 7 November 2026.

**Source:** EMPIAR-11830 Position_1, 7 November 2026.
- Of 41 tilts, five (−75/−72/−69, +42/+45) had all 25 patches beyond the
  tolerance, each by 125 px or more. Of the others, none had more than one, by
  at most 11.9 px.
- Job 3901454 found the tiles of the −75 and +45 sums 150–480 px from where a
  `-Patch 1 1` sum has them.
- Seen on one dataset.

### M-3. A `.gain` reference upside down against EER frames

**Kind:** silent · **Evidence:** measured

**Where it shows:** nowhere by itself. A gain applied upside down adds its
pixel pattern to every sum instead of removing it.

**How to confirm:** the corrected sum's record says how the flip was chosen:
`gain_flip_given: false` means the step chose it.

**Causes, common to rare:**
1. MotionCor3 reads a TIFF bottom row first and decodes EER frames top row
   first, so a Falcon `.gain` (a TIFF) lies upside down against the frames.

**What to try:** nothing for EER movies with a `.gain`. Since 5 November 2026
the step passes `-FlipGain 1` for them unless `[motion] gain_flip` is set. An
MRC gain keeps MotionCor3's default, 0.

**What not to do:** set `gain_flip` for EER without measuring it.
`scripts/eer_gain_flip.sbatch` measures it from sums with no gain and with each
flip.

**When to stop and ask someone:** when the gain comes from another camera or
was written by another program.

**Applies to:** EER movies, since 5 November 2026.

**Source:** EMPIAR-11830, job 3890116, 5 November 2026. Summed over three
movies, only the flipped gain anticorrelated with the counts (r −0.016 against
−0.0005), and every sum was flatter with it.

### M-4. MotionCor3 did not load the gain

**Kind:** crash · **Evidence:** code

**Error text:** `MotionCor3 did not apply the gain reference <path>: it says
'Gain reference not found' (it loads 32-bit float gains only); see <log>`

**Causes, common to rare:**
1. The gain is not 32-bit float. MotionCor3 skips any other mode with a warning
   and corrects nothing.
2. The path is wrong in the job's environment.

**What to try:** convert the gain to 32-bit float, for example with IMOD's
`newstack -mode 2`, or point to the right file. Then rerun from the motion
stage.

**Applies to:** since 5 November 2026. Before that a skipped gain went
unnoticed.

**Source:** read in `backends/motion.py` and in MotionCor3's source. MotionCor3
at MDC prints "Gain reference has been loaded" when it works.

### M-5. MotionCor3 cannot load a library

**Kind:** crash · **Evidence:** seen

**Error text:**
- preflight: `MotionCor3 cannot load: <libraries>- put their directory on
  LD_LIBRARY_PATH before submitting`
- before that check: exit code 127, naming the library, for example
  `libnvToolsExt.so.1` or `libtiff.so.6`

**Causes, common to rare:**
1. The binary was built against libraries the GPU nodes do not have on their
   default path. `libtiff.so.6` comes from the pipeline's environment, which the
   job puts on `LD_LIBRARY_PATH` itself.
2. A CUDA build that needs a library no CUDA install on the node has. In Munich
   `libnvToolsExt.so.1` was in none of `/usr/local/cuda-*`.

**What to try:** put the library's directory on `LD_LIBRARY_PATH` in the shell
you submit from, then resubmit (recipe A). Find it with `ldd "$MOTIONCOR3"` on
a GPU node.

**When to stop and ask someone:** when the library exists nowhere on the
cluster. Then MotionCor3 needs another build.

**Applies to:** every version. The preflight check exists since 9 October 2026.

**Source:** Munich, 7 October 2026, jobs 41065633 and 41074288 (exit code 127
on `libnvToolsExt.so.1`).

### M-6. MotionCor3 could not open the frames

**Kind:** crash, *for the maintainer* · **Evidence:** seen

**Where it shows:** MotionCor3 stops at once (exit code −8), and its log says
the frames file was not opened.

**Causes, common to rare:**
1. MotionCor3 was given paths relative to a folder it did not run in. The HIV-1
   Gag project had been set up with relative paths. `backends/motion.py` has
   passed absolute paths since 9 October 2026, and the project was set up again.
   The link between this crash and the relative paths is likely, not shown.
2. The frames moved after `init`, which records their paths in the manifests.

**What to try:** check that the paths in `manifests/<TS>.json` exist from the
repository root. If the frames moved, the project needs `init` again, as a new
project.

**Applies to:** before 9 October 2026 for cause 1.

**Source:** Munich, 7 October 2026, job 41074613.

### M-7. EER frames grouped too finely or too coarsely

**Kind:** reading results, or crash · **Evidence:** general for the rule; code
for the messages

**Error text** (the step):
- `<n> EER frames cannot make <m> fractions`
- `EER sampling <s> renders the frames on a grid <s> times finer than the movie
  pixel, and -FtBin <b> would leave the sums at a fraction of it; set ftbin to a
  multiple of <s>, e.g. <s> for sums at the movie pixel`

**What it means:**
- EER frames are summed in groups before alignment (`[motion]
  eer_fractions`).
- Unset, each movie gets as many groups as keep each at 0.2 e/Å² or more, and
  at least 2. An EMPIAR-11830 tilt of 1.80 e/Å² gets 9.
- The rule of thumb behind it is at least 0.1–0.15 e/Å² per aligned frame,
  below which frames no longer align. More groups follow the drift more finely;
  fewer align more surely.
- `[motion] eer_sampling = 2` renders on a grid twice as fine, and `ftbin = 2`
  must then bin back.

**What to try:** leave both unset first. Change the groups only on one series,
and compare the motion QC and the sums.

**When to stop and ask someone:** when a tilt has fewer than 3 groups, since
MotionCor3 aligns patches on groups of 4 frames (M-2).

**Applies to:** EER movies, since 5 November 2026; dose-based groups since
7 November 2026.

**Source:** the rule is a colleague's practice, not measured here. The messages
are read in `backends/motion.py`.

### M-8. The disk fills during motion correction

**Kind:** crash · **Evidence:** code

**Where it shows:** the step or a later one fails with "No space left on
device", or the project's folder grows by gigabytes per series.

**Causes, common to rare:**
1. `[motion] storage_policy = "debug"`, which is the command's default, writes
   one float32 MRC per tilt besides the stack. `"working"` keeps Zarr.

**What to try:** set `storage_policy = "working"` in the lane. Check the space
with `df -h` on the project's filesystem, and see what each step wrote with
`scripts/show_timings.py` (disk delta per step).

**Applies to:** every version.

**Source:** [pipeline_map.md](../pipeline_map.md), step 1.

## Looks wrong, is fine

### M-9. A rerun of MotionCor3 does not give the same pixels

**Kind:** false alarm · **Evidence:** measured

**Where it shows:** the sums of a rerun with the same command and the same
binary differ slightly from the first run's.

**Why it is fine:** MotionCor3 does not reproduce its sums bit for bit. On
tilts that aligned well, a rerun correlated at r 0.976–0.989 with the first sum,
with every tile within 0.6 px.

**When it is not fine:** when the tiles of a rerun sit hundreds of pixels from
the first run's. That is a patchwork (M-2), and the two runs patched it
differently.

**Source:** EMPIAR-11830 Position_1, job 3901454, 7 November 2026, at −15°,
−45° and +15°.

### M-10. The motion QC says "not judged"

**Kind:** reading results · **Evidence:** code

**Error text:** `patch alignment not judged on <n> of <m> tilts (<angles> deg):
<reasons>`

**What it means:** the QC found no patch log to judge a tilt by: no log, a log
older than the run, one that does not read, or a patch count other than the
grid's. The QC status is `warning`, not `fail`.

**When it is not fine:** when every tilt is "not judged" on a lane that uses
patches. The QC then judged nothing. The log's name was checked for EER movies
only, so MRC and TIFF movies may hit this; tell the maintainer.

**Source:** read in `backends/motion.py` and
[motioncor3_backend.md](../../motioncor3_backend.md).

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-M1.** Which patch grid do you use for tilt movies, and on what data does
  it fail? Do you use patches on lamellae at all?
- **Q-M2.** How many frames, or how much dose per frame, do your tilt movies
  have? Is 0.2 e/Å² per group a sensible floor for EER (M-7)?
- **Q-M3.** How do you see that motion correction went wrong, before
  alignment? Which tilts do you look at?
- **Q-M4.** Do you dose-weight the sums for tomography? The lanes do not.
- **Q-M5.** Have you met a motion-correction problem that is not here? In a
  line each: what you saw, what it turned out to be, what helped.
