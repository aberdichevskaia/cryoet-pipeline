# RyR1 from frames to MRA classes on cryoem.se, without the assistant

This tutorial takes you from a course account on cryoem.se, the cluster of the
Stockholm workshop, to class averages: start a session in the pipeline your
account already has installed, look at the raw tilts, and run a real dataset
through the whole pipeline with your own commands. The data are RyR1 in native
membranes, five tilt-series of SerialEM TIFF frames with gold beads.

```text
start a session ─▶ check the installation ─▶ check the frames
   ─▶ look at the raw tilts: gold beads and the tilt axis   (a PDF)
   ─▶ create the project
   ─▶ submission 1: frames → tomograms → picks → particles  (4 Slurm jobs)
   ─▶ look at the report and the picks
   ─▶ optional: a branch, 161 aligned by AreTomo2           (2 Slurm jobs)
   ─▶ submission 2: MRA classification                      (1 Slurm job, 1 GPU)
   ─▶ read the classification report
   ─▶ optional: export a class to RELION 5
```

You will end with tomograms of the five series at 13.44 Å/px, RyR1 picks on
their membranes, and an MRA classification of the picks: a few class averages
and a noise class that collects the false picks. A resolution is not part of the
pipeline's own steps: step 13 hands a class to RELION 5 for one.

> **Status, 25 November 2026.** On cryoem.se, this tutorial's lane ran on series
> 161 alone on 24 November 2026, from the frames to the tomogram, with its CPU
> stages on cryo5000 rather than cs-cpu, and so did the branch of step 10. Not
> run there yet: the CPU stages on cs-cpu, the `gpu` stage (membranes, template
> matching, particles), the MRA on one A5000, and the tilt preview of step 4 on
> these TIFF frames. The accounts' installation and the session file of step 1
> were written on 25 November and had not run when this was; a course account
> was installed with the assistant on 22–23 November, with the same programs.
> The course's results of step 8, for going on after a failure,
> were still to be made, and going on from a copy of them had not been tried. At
> MDC the same lane ran through the MRA on four of the series on 19–20 November,
> and through fine alignment on all five on 23 November. If something fails, the
> log says why; tell Anna what happened.

## What you need

- **A course account on cryoem.se** and an SSH connection to
  `login2.cryoem.se`: a terminal, or VS Code with Remote SSH. Every command
  below runs there unless it says it runs on your laptop. The heavy work runs
  in Slurm jobs, which the pipeline's scripts submit for you.
- **Room in your home folder**, which has no quota on course accounts. The
  installation there takes about 27 GB (one course account, 24 November 2026).
  The outputs at bin8 add the motion-corrected sums and stacks and the
  tomograms: 8.4 GB for 161 alone up to the tomogram, with both alignments of
  step 10, so plan for some 40 GB for the five series.
- **On your laptop**: `scp` or another SFTP client, to copy PDFs and folders
  from the cluster; and ChimeraX with ArtiaX, or IMOD's `3dmod`, to see the
  picks in 3D.

## 1. Your account, and how every session starts

Your account has the pipeline installed, in one folder of your home, `~/cryoet`:
the environments, the downloads pip and micromamba keep, and the clone,
`~/cryoet/cryoet-pipeline`, which the outputs go into. The programs the steps
run are installed once for the whole cluster, under `/opt/cryoem`; the course's
module, `tomobear/2026`, names the files the course shares: the model
MemBrain-seg segments membranes with, and the assistant's reference library.

None of it is in `~/.bashrc`: the same account serves another part of the
workshop, which should see none of it. So **every new terminal** starts with
two lines:

```bash
module load tomobear/2026
source ~/cryoet/session.sh
```

The module sets `MEMBRAIN_CKPT`, MemBrain-seg's model, and `CRYOET_LIBRARY`, the
reference library, both in the course folder `/mnt/cephfs/home/tomobear`.
`session.sh` is [`docs/tutorials/ryr1/stockholm-session.sh`](ryr1/stockholm-session.sh),
copied there by the installation; it refuses to run before the module, and its
comments say what each line does:

- `CRYOET_WORK`, `MAMBA_ROOT_PREFIX` and `PIP_CACHE_DIR` name `~/cryoet` and the
  places in it where micromamba keeps environments and downloads and pip keeps
  its downloads.
- `ENV_MANAGER` makes the jobs activate the pipeline's environment with
  micromamba, and micromamba's shell hook makes `micromamba activate` work in
  your shell.
- `IMOD_DIR`, `CTFFIND5`, `MOTIONCOR3` and `ARETOMO` are the cluster's IMOD
  5.2.1, CTFFIND 5.0.2, MotionCor3 1.1.1 and AreTomo2 (its folder says 1.1.2,
  the program reports 1.1.3). MotionCor3 and AreTomo2 need CUDA 12's libraries,
  which it adds to `LD_LIBRARY_PATH`. CTFFIND is named by its full path because
  the cluster also has CTFFIND 4, which the pipeline refuses.
- `PYTOMO_MP_DIR` and `MEMBRAIN_EXE` are pytom-match-pick and MemBrain-seg,
  each in an environment of its own.
- Last, it goes to the clone's root and activates the pipeline's environment,
  and says so: `pipeline session: …/cryoet/cryoet-pipeline, environment
  cryoet-pipeline`. Every command below runs from there.

Jobs take all of this from the shell you submit from and read no `~/.bashrc` of
their own: submit from a terminal where both lines ran.

## 2. Check the installation

`show_setup.py` writes nothing. It shows the room where the environments,
downloads and temporary files go, the pipeline's environment and how its
PyTorch was built, every tool the lane needs with its version, and which of the
cluster's GPUs each program can run on:

```bash
python3 scripts/show_setup.py --config docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml
```

Give it a few minutes: it asks every tool for its version. Read its problems
first: there should be none (`Problems (0): none`). Each problem and each note
names an entry of the playbook, `docs/agent/playbook/`, which says what to do.
A tool shown as missing means its variable is not set in this shell: run the
two lines of step 1 in it.

The installation ran the same check on your account in a short job on one of
the course's A5000 cards, where it also compares the driver with the CUDA of
each program. To run that part again:

```bash
srun --partition=cryo5000 --reservation=tomo --gres=gpu:a5000:1 \
    --cpus-per-task=4 --mem=16G --time=00:15:00 \
    python3 scripts/show_setup.py \
    --config docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml
```

What the installation made, for doing the same on another cluster: three
environments, each with micromamba from conda-forge. `cryoet-pipeline` holds
Python 3.13, the clone installed with `pip install -e .`, and `imagecodecs`,
which lets step 4 read the TIFF frames if they are compressed; its PyTorch is a
CUDA build. `pytom_tm` holds Python 3.12, CuPy for CUDA 12 and
pytom-match-pick 0.14.0; `membrain` holds Python 3.9 and MemBrain-seg 0.0.10.
The README's [one-time setup](../../README.md#0-one-time-setup) says how to
install the pipeline on a cluster of your own.

## 3. The data: read in place, checked before anything runs

The frames are in the course folder, which every course account can read:
`/mnt/cephfs/home/tomobear/data/ryr1`. Your home has a link to the course's
data folder, `~/tomobear`, so the same folder is also `~/tomobear/ryr1`. Do not
copy them: every group would hold its own copy of the same frames, and the
pipeline only reads them. The tutorial's lane names this folder.

```bash
ls /mnt/cephfs/home/tomobear/data/ryr1/*.tif | wc -l
```

```bash
ls -l /mnt/cephfs/home/tomobear/data/ryr1/gain.mrc
```

You should see 145 frames (5 series × 29 tilts) and `gain.mrc`, the gain
reference.

There are no mdocs with these frames: the series, the tilt number and the angle
are in each file name, `tomogram_039_001_-42.0.tif`. The pipeline reads them
with a regular expression. `show_folder.py` reads the folder the way the
pipeline will, and writes nothing:

```bash
python3 scripts/show_folder.py /mnt/cephfs/home/tomobear/data/ryr1 --pattern \
    'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'
```

Check its table against what was collected: five series, 039, 096, 097, 125 and
161; 29 tilts each, from −42° to +42°; 10 frames per tilt; 0.84 Å per pixel (K3
super-resolution). It also says that `gain.mrc` is not a frame. With your own
data, this is the moment to catch a wrong pattern or a missing tilt.

## 4. Look at the raw tilts: gold beads and the tilt axis

Two things the lane needs are not in the file names: whether there are gold
beads to align on, and which way the tilt axis runs on the camera. Without
mdocs and without a viewer on the cluster, `show_tilts.py` answers both from the
frames themselves, in a PDF. It reads nine tilts of a series: the one nearest
0°, the lowest, the highest, and the four highest on each side, and adds up each
tilt's frames, without gain or motion correction. It writes only the PDF.

```bash
python3 scripts/show_tilts.py /mnt/cephfs/home/tomobear/data/ryr1 \
    --series 039 --pattern \
    'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'
```

It reads about nine gigabytes of frames, so give it some minutes. It prints, for
039, the tilts it read, the dark round spots it found and their size, the tilt
axis it measured, and a note for anything it left out; the last line names the
PDF, `outputs/tilt-previews/ryr1_039.pdf`. If it stops saying that tifffile
needs the imagecodecs package, the installation missed it: run
`python -m pip install imagecodecs` in the pipeline's environment, which step 1
activated, then the same command again.

**Open the PDF.** In VS Code connected over SSH, open it from the file tree
(VS Code needs a PDF viewer extension for that). Or copy it to your laptop. In a
terminal on the laptop, name your clone on the cluster once, with your account
in place of `<account>`:

```bash
CLONE=<account>@login2.cryoem.se:cryoet/cryoet-pipeline
```

and copy the PDF from it:

```bash
scp "$CLONE/outputs/tilt-previews/ryr1_039.pdf" .
```

A graphical SFTP client does the same: the PDF is in your home on the cluster,
under `cryoet/cryoet-pipeline/outputs/tilt-previews/`.

**Gold beads.** The pages show the tilt nearest 0°, the lowest and the highest,
with the dark round spots circled, and the darkest of them close up. Gold beads
are the darkest, roundest spots, the same size everywhere. The lane aligns on
them (`[fiducials]`) and assumes 10 nm beads: do the spots look like beads, and
is the size the PDF gives close to 10 nm? With no beads, a lane would align by
patch tracking or by AreTomo2 instead.

**The tilt axis.** The tilts on either side of 0° are compared in pairs, the
more tilted one stretched across each candidate axis, and the best axis is the
one where they match. The PDF gives it as a line on the image and as an angle
from the Y axis, the way IMOD counts it, in two forms, such as 84° or −96°. The
two are the same line: the measurement has no sign, and the two signs give
tomograms of opposite hand. The lane says 84.0 (`[align] tilt_axis_angle`),
which IMOD's tiltalign solved on four of these series at MDC, starting from an
axis read by eye; on 161, AreTomo2 refined it to 83.9 on cryoem.se. So 039
should read about 84, give or take a degree or two. If it reads about 96 (or
−84), the preview has read the TIFF rows in the other order from MotionCor3:
keep 84 in the lane, which was solved on MotionCor3's sums, and tell Anna.

> **Status, 24 November 2026.** `show_tilts.py` is new. On MRC stacks of two
> EMPIAR datasets it found the axis within one degree of their mdocs, and 62
> gold beads of 10 nm on one of them. It has not read TIFF frames on the
> cluster yet: this step is its first run on them. Tell Anna what it read.

To look at another series, give its name to `--series`, for example
`--series 161`; each is another nine gigabytes to read. `--all` reads nine
tilts of every series, about 45 GB.

## 5. Create the project, with all five series

The project is the folder that holds everything the pipeline writes about these
data, `outputs/ryr1-tutorial`, in your clone. Create it now, with every series
the folder holds:

```bash
cryoet init --frames /mnt/cephfs/home/tomobear/data/ryr1 \
    --out outputs/ryr1-tutorial --frames-pattern \
    'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'
```

It prints each series with its tilt count, angle range, frames per tilt and
pixel size: the same numbers as step 3.

Why by hand: the submission script would create the project itself, with the
series its lane file lists, but this way you see what goes into it before
anything runs. A project is never created a second time: `cryoet init` refuses
a folder that holds a project, since it would start its registry empty and lose
everything in it. A series left out can join later, with the same command,
`--tilt-series <series>` and `--add-series`, which writes only the new series'
manifests and keeps the rest of the project as it is. A dry run of a lane
naming a series the project does not have prints that command.

## 6. Read the lane file

A *lane* is one TOML file that says how a dataset is processed: where the data
are, every setting of every step, and how the fixed chain of steps is cut into
Slurm jobs. The tutorial's lane is
[`docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml`](ryr1/ryr1-tutorial-stockholm.toml):
the five series from the frames to the particles, and an MRA of all of them.

Open it and read it top to bottom; each value has a comment saying where it
came from. The parts to understand before you submit:

- **`[run] series`** lists all five.
- **`[[run.stage]]`** cuts the chain into five jobs: `motion` (MotionCor3, a
  GPU), `align` (beads, IMOD), `cpu` (CTF and the tomogram), `gpu` (membranes,
  template, picking, particles) and `mra`. Each job runs its steps for every
  series in turn; the next starts only if the one before it succeeded. To list
  the steps:

  ```bash
  python3 scripts/run_series.py --project outputs/ryr1-tutorial \
      --config docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml \
      --series 039 --list-steps
  ```

- **Each stage's `sbatch` options** are what the course was given on
  cryoem.se. Every job runs in the reservation `tomo`. GPU stages go to the
  partition `cryo5000`, two nodes of eight RTX A5000 cards (24 GB each), with
  one card and at most 40 GB of memory a job, so that twelve groups fit on the
  two nodes. CPU stages go to `cs-cpu`, whose jobs must ask for 8 CPUs and
  31 GB. The CPU stages peaked under 4 GB on 161 (24 November 2026).
- **`[data]`** names the course folder of step 3, and **`[motion]`** its gain
  reference. These frames are not gain-normalised, and MotionCor3 sums them
  without complaint when the gain is missing.
- **`[align] tilt_axis_angle = 84.0`** stands in for the mdoc this dataset does
  not have: the axis of step 4. Its sign is not known: 84 and −96 give
  mirror-image tomograms.
- **`[fiducials]`** tracks the gold beads of step 4, as 10 nm beads.
- **`[pick]`** searches for RyR1 only near membranes (`use_segmentation_mask`),
  with a 340 Å particle, and keeps twice as many picks as pytom recommends, so
  that the MRA's noise class has false picks to take.
- **There is no `[mra]` section.** The MRA then follows the lab's rules, which
  the dry run in step 11 prints with the source of every value.

For your own data, start from
[`configs/lanes/example.toml`](../../configs/lanes/example.toml) instead; the
[README](../../README.md#processing-your-own-data-on-the-cluster) says how.

## 7. Submission 1: frames to particles

First a dry run. It submits nothing, and checks everything that can be checked
from your shell: the stages, the input folders, the tool variables, whether
one of your jobs is already working on these series, and each stage's options,
which it puts to Slurm with `sbatch --test-only`.

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml \
    --until-stage gpu --dry-run
```

`--until-stage gpu` stops the submission before the MRA, so that you can look
at the picks first. The dry run should show:

- `outputs/ryr1-tutorial, 5 series`, and no `would run: cryoet init` line (the
  project exists since step 5);
- four `sbatch` lines, each job waiting for the one before it
  (`--dependency=afterok`), and their chain: `motion -> align -> cpu -> gpu`;
- what Slurm said of each stage: when and where it would start, each stage
  judged alone, without waiting for the one before;
- no line that starts with `would refuse:`. If there is one, it names what to
  fix; a stage Slurm refuses comes with Slurm's own reason, for example a
  reservation or partition your account may not use.

Then the same command without `--dry-run`:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml \
    --until-stage gpu
```

It prints the four job IDs. Every dry run and every submission is written to
`outputs/ryr1-tutorial/journal.jsonl`, a submission with the lane file's text,
so that the project keeps a record of what was asked, by whom and when.

## 8. Follow the jobs

```bash
squeue --me
```

```bash
python3 scripts/show_project.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml
```

`show_project.py` shows, per series and step, the last run and its QC verdict
(`pass`, `warning` or `fail`), the warnings, and the lane's jobs. Each job
writes its log to the clone's root, as `slurm-<job name>-<job id>.log`, for
example `slurm-ryr1-tutorial-align-<id>.log`. The `motion` job's log warns that
the node has no `cuobjdump`, so MotionCor3's GPU code goes unchecked: on
cryoem.se that is expected, and not a failure.

**When a job fails**, the jobs after it are cancelled by Slurm. Find out why:

```bash
python3 scripts/show_failure.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml
```

It gives the failed step's lines from the log, the error and the logs of the
programs the step ran. Fix the cause, then submit again from the stage that
failed, with a dry run first. Steps of that stage that had finished run again,
which is why `--overwrite` is needed:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml \
    --from-stage cpu --until-stage gpu --overwrite --dry-run
```

`--overwrite` holds for that submission only, and its dry run lists the
recorded results it may replace. Do not resubmit a failure that will happen
again unchanged: find the cause first.

**If the cause cannot be fixed in the course's time**, go on from the course's
results. The same lane has run through every stage in the course folder, in
`/mnt/cephfs/home/tomobear/cryoet-pipeline/outputs/ryr1-tutorial`. Copied into
your clone at the same place, it is a project like your own: its records name
its files from the clone's root, and it reads the frames from the course folder
as yours does. Set your project aside, so that nothing of it is lost, and copy
the course's in its place (some 40 GB, a few minutes):

```bash
mv outputs/ryr1-tutorial outputs/ryr1-tutorial.mine
cp -r /mnt/cephfs/home/tomobear/cryoet-pipeline/outputs/ryr1-tutorial outputs/
```

It holds the results of every stage, the MRA's too, so you can read them right
away, from step 9 on. To run the rest yourself, submit from the stage after the
one that failed; its results and those after it are replaced by yours, which
is why `--overwrite` is needed. After a failed `cpu` job, for example:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml \
    --from-stage gpu --until-stage gpu --overwrite --dry-run
```

## 9. Look at the results before the MRA

Once the `gpu` job has finished, write the project's report. It reads the
project, runs nothing, and writes one PDF with a page per step and series:

```bash
cryoet report --project outputs/ryr1-tutorial --variant bin8 \
    --config docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml
```

It goes to `outputs/ryr1-tutorial/reports/report-bin8.pdf`. Open it in VS Code,
or copy it to your laptop as in step 4, with `CLONE` named there:

```bash
scp "$CLONE/outputs/ryr1-tutorial/reports/report-bin8.pdf" .
```

Look at:

- **the summary table**: each step's verdict per series. A `warning` does not
  stop the chain, so read each one;
- **tilt alignment**: the residual and the beads per tilt, and the tilt axis the
  alignment solved. Is it the axis step 4 measured from the raw tilts?
- **CTF**: which views were fitted. On these data a few of the highest tilts
  may find no rings. If a series has five such views or fewer, they are left
  out of the tomogram; with more, they stay in with an interpolated defocus,
  and the QC warns;
- **the tomogram and its membranes**: is the membrane segmentation on the
  membranes?
- **the picks**: on the tomogram, their scores against noise, and the galleries
  of picks by score, each turned to the template's orientation. Good picks look
  like the template; the low-score end is where the false ones are.

**The picks in 3D.** `visualize-picks` has written, per series,
`outputs/ryr1-tutorial/picks_view/<series>/`: the tomogram as `<series>.mrc`, a
`<series>_open_in_chimerax.cxc` for ChimeraX with ArtiaX, and the picks as
`<series>_picks.mod` for 3dmod. The folder holds everything it opens, so copy
it whole to your laptop, some 300–400 MB a series (measured at MDC):

```bash
scp -r "$CLONE/outputs/ryr1-tutorial/picks_view/039" .
```

Then, on the laptop, open `039/039_open_in_chimerax.cxc` in ChimeraX, or the
tomogram with its picks in 3dmod:

```bash
3dmod 039/039.mrc 039/039_picks.mod
```

ArtiaX did not start in ChimeraX 1.12 for one of the organizers (a NumPy
conflict, ArtiaX issue #56); 3dmod shows the same picks. Open one or two series:
do the picks sit on density at the membranes?

## 10. A branch: 161 aligned by AreTomo2 (optional)

> **Status, 24 November 2026.** The branch ran once on cryoem.se, on 24
> November 2026, in a test project up to the tomogram: AreTomo2 aligned 161 in
> a minute and a half on an A5000, and each lane's status and report showed its
> own results.

A lane can part from another earlier than at its own tomogram. Here 161 is
aligned a second way, by AreTomo2's projection matching instead of the gold
beads, in the same project. The branch's lane is
[`ryr1-tutorial-stockholm-161-aretomo.toml`](ryr1/ryr1-tutorial-stockholm-161-aretomo.toml):

- **Shared:** motion correction and the coarse alignment of 161, which the main
  lane made. They are not done again.
- **Its own:** the fine alignment and the CTF fit, labelled `aretomo`
  (`[lane] alignment_variant`), and the aligned stack, the tomogram and the
  picks, labelled `bin8-aretomo` (`[lane] variant`). Nothing of the main lane's
  is replaced: its bead alignment and its `bin8` results stay as they are.
- **Set for AreTomo2:** no `[fiducials]`, `[fine_align] backend = "aretomo"`,
  and `[reconstruct] thickness`, since AreTomo2 measures no sample depth. The
  file's comments say how each value was chosen.

AreTomo2 runs on a GPU and is found through `$ARETOMO`, which `session.sh` of
step 1 sets.

Submit once the main lane's jobs are done with 161: a series that one of your
jobs works on is refused to another. The branch starts at its own alignment and
stops at the tomogram:

```bash
python3 scripts/submit_lanes.py \
    docs/tutorials/ryr1/ryr1-tutorial-stockholm-161-aretomo.toml \
    --from-stage align-aretomo --until-stage cpu-aretomo --dry-run
```

The dry run should show two jobs on 161, `align-aretomo` and then
`cpu-aretomo`, and no `would refuse:` line. Then the same command without
`--dry-run`. `show_project.py` given the branch's lane file shows its own runs
of the steps it labels; the main lane's table does not change.

When `cpu-aretomo` has finished, write the branch's report:

```bash
cryoet report --project outputs/ryr1-tutorial --variant bin8-aretomo \
    --config docs/tutorials/ryr1/ryr1-tutorial-stockholm-161-aretomo.toml --series 161
```

It goes to `outputs/ryr1-tutorial/reports/report-bin8-aretomo.pdf`. Put its
pages for 161 beside those of step 9:

- **tilt alignment:** the beads' page shows tiltalign's residual per tilt,
  AreTomo2's the shifts of its last projection-matching pass. The two measure
  different things, so their numbers are not to be compared;
- **tilt angles:** each page says by how much the alignment shifted the angles.
  In the test on cryoem.se the beads shifted them by +0.2°, AreTomo2, which
  tries whole degrees only, by −6°: the two tomograms are turned against each
  other by that difference, about the tilt axis;
- **CTF:** the branch fits the CTF on its own aligned stack, and centres the
  search of every tilt on the fit of its 0° tilt, which after AreTomo2's −6° is
  the one collected at +6°. In the test that left out two more of the most
  tilted views than the beads did, 24 of 29 against 26: their defocus lay past
  the ±1 µm searched (playbook entry C-1);
- **the tomograms:** the same membranes and beads in both? Is one sharper, or
  tilted against the other?

To pick and extract the branch's particles too, submit `gpu-aretomo` the same
way. They stay out of the MRA of step 11, which reads the `bin8` particles.

## 11. Submission 2: the MRA

The MRA (multi-reference alignment) classifies the particles of all the series
together: it aligns every particle to a few references and sorts them into
classes. With the lab's rules it starts from three references: two averages of
300 random particles each and one of pure noise, which collects the particles
that resemble nothing.

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml \
    --from-stage mra --dry-run
```

The dry run prints the run's folder, `outputs/ryr1-tutorial/mra/bin8-ssn`, and
every setting with where it came from: the seeds, the five rounds and their
low-pass in Å, the diameter from `[pick]`, the mask and the shifts. It refuses
an MRA while a series has no particles yet, and while one of your jobs still
works on the series. When it shows no `would refuse:` line, submit the same
command without `--dry-run`.

The job runs on one A5000, and takes hours. At MDC, on four A40s and four of
the series, its first two iterations took 40 and 83 minutes and the last three
about 2 minutes together; one A5000 should take several times as long, which
has not been measured. Its time limit is two days. If it stops on that limit,
submitting it again carries the run on from its last finished iteration.

## 12. Read the classification report

The job writes its report beside the run:
`outputs/ryr1-tutorial/mra/bin8-ssn_classification_report.pdf`. Its pages, in
order:

1. the parameters: the data per series, the starting references, the masks,
   every round's search and how long it took;
2. particles at their starting poses, and their average;
3. the starting references;
4. the class sizes after every iteration, and how many particles changed class;
5. every iteration's class averages, as central sections and z slices;
6. the last classes by series and by picking score;
7. particles of each last class.

What to look for: does a class show RyR1's square shape, with the membrane? Is
the noise class taking mostly low-score picks? Do the classes differ in the
particles or only in how they are seen? On these data at bin8, earlier MRA runs
split the particles by viewing direction, top views against side views, which
the missing wedge makes look different: two such classes are one particle.

## 13. A class to RELION 5 (optional)

The MRA sorts the particles; it does not give a resolution. For a gold-standard
refinement, a class that looks like RyR1 can go to RELION 5.
`scripts/export_relion.py` writes one folder that RELION runs as it is. It
submits nothing.

> **Status, 24 November 2026.** The export is new. A RyR1 class was exported
> at MDC on 10 November 2026; RELION's Extract and Refine3D have not run on its
> exports yet, at MDC or on cryoem.se. Tell Anna what happens.

**The dry run first.** It names the class's particles after the run's last
finished iteration, the duplicates it leaves out, the two halves and a note per
series:

```bash
python3 scripts/export_relion.py --run outputs/ryr1-tutorial/mra/bin8-ssn \
    --class <C> --dry-run
```

`<C>` is the class number in the classification report. For these data the
notes will say that the acquisition order comes from the tilt order and that
there is no dose. Here the tilt order is the angle order, not the order the
tilts were taken in. **Do not give `--dose-per-tilt` for these data:** with the
order unknown, a dose would weight the wrong tilts if they were taken
dose-symmetrically. Without a dose, RELION weights no tilt by dose. Look at the
two halves' sizes too: neighbours closer than a box stay in one half and the
halves are balanced over the whole class, so they come out nearly equal unless
one group of neighbours holds a large part of the class; the dry run names the
largest group.

**Then the export**, the same command without `--dry-run`. It writes
`outputs/ryr1-tutorial/relion/bin8-ssn-class<C>/`:

- the tilt series, tomograms and particles as RELION 5 reads them;
- the class average and the MRA's mask, in RELION's frame;
- `relion_commands.sh`: an Extract, then a gold-standard Refine3D with local
  searches from the MRA's poses, without symmetry;
- `README.md`, which says why each value is what it is.

Both run at the MRA's pixel, 13.44 Å here, so this first refinement checks the
export rather than reaching ~20 Å. Finer pixels come after, by extracting again
at a lower `--bin` (the README says how). At the end, the export prints the
`sbatch` line that runs the commands, with the folder filled in.

**RELION on cryoem.se.** `relion_commands.sh` takes RELION from your `PATH`,
from the modules `RELION5_MODULE` names, and from an installation
`RELION5_HOME` names. Which module gives RELION 5 on cryoem.se is the
organizers' to say; set it in the shell you submit from, since the job takes it
from there:

```bash
export RELION5_MODULE=<the RELION 5 module>
```

Then run the line the export printed, with the course's GPU options in place of
`<GPU partition and --gres>`: one A5000, which the two Refine3D workers share,
and the course's 40 GB.

```bash
sbatch --nodes=1 --ntasks=3 --cpus-per-task=8 --time=1-00:00:00 \
    --partition=cryo5000 --reservation=tomo --gres=gpu:a5000:1 --mem=40G \
    --chdir=<the folder it printed> --wrap "./relion_commands.sh"
```

`relion_commands.sh` first checks that RELION starts, and stops, naming what is
missing, if it does not. The job's log, `slurm-<job id>.out`, is written in the
export's folder.

## Where to go next

- **Run one class again.** If a class looks like RyR1, its particles can be
  classified again, with other settings if you like: a copy of the lane whose
  `[mra]` names the run and its classes (`from_run`, `from_classes`), submitted
  with `--from-stage mra`. [docs/mra_in_lanes.md](../mra_in_lanes.md#again-on-a-class)
  has the keys.
- **A finer tomogram.** bin4 (6.72 Å/px) is the next step towards ~20 Å:
  `configs/lanes/ryr1-bin4.toml` and its comments, with this lane's `[data]`
  and `sbatch` options.
- **A resolution.** A class in RELION 5, [step 13](#13-a-class-to-relion-5-optional).
  Halves split after a joint classification give the map's consistency, not its
  resolution: [docs/agent/playbook/averaging.md](../agent/playbook/averaging.md),
  entries V-3 and V-4.
- **Your own data.** The README's
  [Processing your own data on the cluster](../../README.md#processing-your-own-data-on-the-cluster),
  starting from `configs/lanes/example.toml`, with step 4's PDF for the beads
  and the axis.

## When something refuses

| It says | What it means |
|---|---|
| `$MOTIONCOR3 is not set` (or another `$…`) | the shell lacks the tool variables: run the two lines of step 1 in it, and submit from there |
| `session.sh: load the course module first` | `module load tomobear/2026` comes before `source ~/cryoet/session.sh` |
| a stage refused by Slurm, with Slurm's reason | the stage's `sbatch` options do not fit what your account may use: compare them with step 6, and ask the organizers if the reservation has ended |
| `… already exists but has no manifest for …; add them to it …` | the project was created without that series (step 5 was skipped): run the `cryoet init … --add-series` it prints, then the dry run again |
| `… is a project already …` | `cryoet init` was run on an existing project; add series with `--add-series` |
| a series is busy, naming a job | one of your jobs works on that series; wait until it has ended |
| `OVERWRITE is set in the environment` | `unset OVERWRITE`; give `--overwrite` to the one submission that needs it |
| `the MRA reads the particles extracted under 'bin8', and cannot` | a series has no particles yet: wait for its `gpu` job, or check why it failed |
| an existing MRA run with other settings | that name is taken: give the copy its own `[mra] name` |
| CTFFIND 4 refused | `$CTFFIND5` is not set: the two lines of step 1 |
| `show_tilts.py`: tifffile needs the imagecodecs package | `python -m pip install imagecodecs` in the pipeline's environment |
| `micromamba: command not found`, or `cryoet: command not found` | the two lines of step 1 were not run in this terminal, or one of them failed: run them and read what they print |
| a stage of yours failed and cannot be fixed in time | go on from the course's results: step 8 |
