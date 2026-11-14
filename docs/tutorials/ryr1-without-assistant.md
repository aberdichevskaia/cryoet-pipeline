# RyR1 from frames to MRA classes on max, without the assistant

This tutorial takes you from an empty folder to class averages, on MDC's cluster
max: clone the repository, install it, copy a real dataset, and run it through
the whole pipeline with your own commands. The data are RyR1 in native
membranes, five tilt-series of SerialEM TIFF frames with gold beads.

```text
clone + install ─▶ copy frames ─▶ check them ─▶ create the project
   ─▶ submission 1: frames → tomograms → picks → particles  (4 Slurm jobs)
   ─▶ look at the report
   ─▶ submission 2: MRA classification                      (1 Slurm job, 4 GPUs)
   ─▶ read the classification report
   ─▶ optional: export a class to RELION 5 and refine it there
```

You will end with tomograms of all five series at 13.44 Å/px, RyR1 picks on
their membranes, and an MRA classification of the picks: a few class averages
and a noise class that collects the false picks. A resolution is not part of the
pipeline's own steps: step 12 hands a class to RELION 5 for one.

There is a second tutorial for the same data with the assistant:
[ryr1-with-assistant.md](ryr1-with-assistant.md).

> **Status, 10 November 2026.** Every step up to the particles has run on these
> data on max. The MRA as a stage of a lane is new, built on 10 November 2026,
> and has not run on max yet. If it fails, the log says why; tell Anna what
> happened.

## What you need

- An account on max, and a folder on `/fast` with room for the environments
  (step 1), the data and the outputs. Measure the frames with `du -sh` before
  copying them (step 2); the outputs at bin8 add about 30 GB of
  motion-corrected sums and stacks plus the tomograms.
- A terminal on a login node. Every command below runs there. The heavy work
  runs in Slurm jobs, which the pipeline's scripts submit for you.
- About an hour of your time for the setup, then waiting for the queue.

## 1. Clone and install (once)

**Your folder, first.** Your home on max is almost always too small for what
this tutorial installs: two environments, one with PyTorch and its CUDA
libraries, and the downloads pip and micromamba keep. Choose one folder of
yours on `/fast` with room, and name it once in `~/.bashrc`; the environments,
the downloads and the clone with its outputs all go there. Add these lines at
the end of `~/.bashrc`, with your folder in place of `<your folder>`:

```bash
export CRYOET_WORK=/fast/AG_Kudryashev/<your folder>
export MAMBA_ROOT_PREFIX=$CRYOET_WORK/micromamba
export PIP_CACHE_DIR=$CRYOET_WORK/cache/pip
export ENV_MANAGER=micromamba
source /fast/AG_Kudryashev/software/cryoet-tools.sh
export PYTOMO_MP_DIR=$MAMBA_ROOT_PREFIX/envs/pytom_tm/bin
```

- `CRYOET_WORK` is your folder; the commands below use it.
- `MAMBA_ROOT_PREFIX` is where micromamba makes environments and keeps its
  downloads. micromamba's own block in `~/.bashrc` sets it to your home, so
  these lines must come after that block. Environments you made before stay
  where they were, and `micromamba activate <their path>` still reaches them.
- `PIP_CACHE_DIR` is where pip keeps what it downloads.
- `ENV_MANAGER` makes the jobs activate the environment with micromamba. Some
  of max's nodes have a system conda, which a job would otherwise try first and
  fail with.
- The group's tools (IMOD, MotionCor3, CTFFIND 5, MemBrain-seg) are installed
  once for everyone, and `/fast/AG_Kudryashev/software/cryoet-tools.sh` names
  them.
- `PYTOMO_MP_DIR` is pytom-match-pick, in the environment made below.

Keep them outside any check for the host name: jobs take these variables from
the shell you submit from, and that shell is on a login node. Open a new
terminal so that they take effect, and check:

```bash
echo "$MAMBA_ROOT_PREFIX"
```

It must print your folder followed by `/micromamba`. If it prints a folder in
your home, micromamba's block comes after these lines: move them below it.

Then clone into your folder:

```bash
cd "$CRYOET_WORK"
git clone https://github.com/aberdichevskaia/cryoet-pipeline.git
cd cryoet-pipeline
```

**The pipeline's environment.** On max, environments are made with micromamba.
The `gpu` extra brings PyTorch, which the MRA runs on; `viz` brings matplotlib,
which the reports need.

```bash
micromamba create -n cryoet-pipeline -c conda-forge python=3.13
micromamba activate cryoet-pipeline
python -m pip install -e ".[io,viz,gpu]"
```

If pip stops on a broken download (`error: incomplete-download` or another
network error), the network dropped, not the pipeline: run the same command
again. What pip has already downloaded is cached. With `--resume-retries 10`
pip resumes a broken download itself, which matters most for PyTorch and its
CUDA libraries, the largest downloads here:

```bash
python -m pip install --resume-retries 10 -e ".[io,viz,gpu]"
```

Check that PyTorch is a CUDA build. The second number must be a CUDA version,
not `None` (the login node has no GPU, which is fine: the jobs run on GPU
nodes).

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**pytom-match-pick**, the template matching, in an environment of its own, at
version 0.14.0, the one the group's runs use:

```bash
micromamba create -n pytom_tm -c conda-forge python=3.12 cupy "cuda-version=12.*"
micromamba run -n pytom_tm python -m pip install "pytom-match-pick[plotting]==0.14.0"
```

**Check what your shell finds.** This writes nothing:

```bash
python3 scripts/build_library.py --site mdc --dry-run \
    --config docs/tutorials/ryr1/ryr1-tutorial.toml
```

Every tool should show a version, pytom-match-pick 0.14.0 among them, and the
last lines should say `Problems: none`. If a tool is missing, its variable is
not set in this shell: check the `.bashrc` lines at the top of this step.

**Every time you come back**, start the same way: a login node, the clone's
root, the environment active.

```bash
cd "$CRYOET_WORK/cryoet-pipeline"
micromamba activate cryoet-pipeline
```

## 2. Copy the data

The frames are in Anna's folder. Copy the TIFF frames and the gain reference
into your clone, under `data/ryr1/frames`, where the tutorial's lane files
expect them. `data/` is ignored by git. First see how much it is:

```bash
du -sh /fast/AG_Kudryashev/anna.berdichevskaia/frames.renamed.top1to5
```

```bash
mkdir -p data/ryr1/frames
rsync -a --info=progress2 \
    --include='tomogram_*.tif' --include='gain.mrc' --exclude='*' \
    /fast/AG_Kudryashev/anna.berdichevskaia/frames.renamed.top1to5/ \
    data/ryr1/frames/
```

You should now have 145 frames (5 series × 29 tilts) and `gain.mrc`:

```bash
ls data/ryr1/frames/*.tif | wc -l
```

```bash
ls -l data/ryr1/frames/gain.mrc
```

## 3. Check the data before anything runs

There are no mdocs with these frames: the series, the tilt number and the angle
are in each file name, `tomogram_039_001_-42.0.tif`. The pipeline reads them with
a regular expression. `show_folder.py` reads the folder the way the pipeline
will, and writes nothing:

```bash
python3 scripts/show_folder.py data/ryr1/frames --pattern \
    'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'
```

Check its table against what was collected: five series, 039, 096, 097, 125 and
161; 29 tilts each, from −42° to +42°; 10 frames per tilt; 0.84 Å per pixel (K3
super-resolution). It also says that `gain.mrc` is not a frame. With your own
data, this is the moment to catch a wrong pattern or a missing tilt.

## 4. Create the project, with all five series

The project is the folder that holds everything the pipeline writes about these
data, `outputs/ryr1-tutorial`. Create it now, with every series the folder
holds:

```bash
cryoet init --frames data/ryr1/frames --out outputs/ryr1-tutorial --frames-pattern \
    'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'
```

It prints each series with its tilt count, angle range, frames per tilt and
pixel size: the same numbers as step 3.

Why by hand: the submission script would create the project itself, but only
with the series its lane file lists, and the main lane lists four of the five
(step 5 says why). A project is never created a second time, because that
would start its registry empty and lose everything in it. So a series left out
here could not join later; it would need a new project.

## 5. Read the lane file

A *lane* is one TOML file that says how a dataset is processed: where the data
are, every setting of every step, and how the fixed chain of steps is cut into
Slurm jobs. The tutorial's lanes are in `docs/tutorials/ryr1/`:

| File | Series | What for |
|---|---|---|
| [`ryr1-tutorial.toml`](ryr1/ryr1-tutorial.toml) | 039 096 097 125 | the main lane, frames to particles, and an MRA of these four |
| [`ryr1-tutorial-161.toml`](ryr1/ryr1-tutorial-161.toml) | 161 | the same, with `tracking_binning = 3` (optional, step 7) |
| [`ryr1-tutorial-all.toml`](ryr1/ryr1-tutorial-all.toml) | all five | only the MRA of all five (step 10) |

Open `ryr1-tutorial.toml` and read it top to bottom; each value has a comment
saying where it came from. The parts to understand before you submit:

- **`[run] series`** lists four series. On 161, IMOD's `beadtrack` segfaults at
  the tracking binning that suits the other four, and a lane cannot set a value
  per series. 161 gets its own copy of the lane.
- **`[[run.stage]]`** cuts the chain into five jobs: `motion` (MotionCor3, a
  GPU), `align` (beads, IMOD), `cpu` (CTF and the tomogram), `gpu` (membranes,
  template, picking, particles) and `mra`. Each job runs its steps for every
  series in turn; the next starts only if the one before it succeeded. To list
  the steps:

  ```bash
  python3 scripts/run_series.py --project outputs/ryr1-tutorial \
      --config docs/tutorials/ryr1/ryr1-tutorial.toml --series 039 --list-steps
  ```

- **`[motion]`** names the gain reference. These frames are not
  gain-normalised, and MotionCor3 sums them without complaint when it is
  missing.
- **`[align] tilt_axis_angle = 84.0`** stands in for the mdoc this dataset does
  not have. Its sign is not known: 84 and −96 give mirror-image tomograms.
- **`[pick]`** searches for RyR1 only near membranes (`use_segmentation_mask`),
  with a 340 Å particle, and keeps twice as many picks as pytom recommends, so
  that the MRA's noise class has false picks to take.
- **There is no `[mra]` section.** The MRA then follows the lab's rules, which
  the dry run in step 10 prints with the source of every value.

For your own data, start from
[`configs/lanes/example.toml`](../../configs/lanes/example.toml) instead; the
[README](../../README.md#processing-your-own-data-on-the-cluster) says how.

## 6. Submission 1: frames to particles

First a dry run. It submits nothing, and checks everything that can be checked
on the login node: the stages, the input folders, the tool variables, whether
one of your jobs is already working on these series.

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial.toml \
    --until-stage gpu --dry-run
```

`--until-stage gpu` stops the submission before the MRA, so that you can look
at the picks first. The dry run should show:

- `outputs/ryr1-tutorial, 4 series`, and no `would run: cryoet init` line (the
  project exists since step 4);
- four `sbatch` lines, each job waiting for the one before it
  (`--dependency=afterok`), and their chain: `motion -> align -> cpu -> gpu`;
- no line that starts with `would refuse:`. If there is one, it names what to
  fix.

Then the same command without `--dry-run`:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial.toml \
    --until-stage gpu
```

It prints the four job IDs. Every dry run and every submission is written to
`outputs/ryr1-tutorial/journal.jsonl`, a submission with the lane file's text,
so that the project keeps a record of what was asked, by whom and when.

## 7. Series 161 (optional, recommended)

`ryr1-tutorial-161.toml` is the main lane with one series, `tracking_binning =
3`, and stages named `motion-161` … `gpu-161`. Different series of one project
may run at the same time, so submit it now:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-161.toml --dry-run
```

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-161.toml
```

It uses the same project and the same label (`bin8`), so 161's particles join
the others' in the MRA of step 10.

## 8. Follow the jobs

```bash
squeue --me
```

```bash
python3 scripts/show_project.py docs/tutorials/ryr1/ryr1-tutorial.toml
```

`show_project.py` shows, per series and step, the last run and its QC verdict
(`pass`, `warning` or `fail`), the warnings, and the lane's jobs. For 161, give
it `ryr1-tutorial-161.toml`. Each job writes its log to the clone's root, as
`slurm-<job name>-<job id>.log`, for example `slurm-ryr1-tutorial-align-<id>.log`.

**When a job fails**, the jobs after it are cancelled by Slurm. Find out why:

```bash
python3 scripts/show_failure.py docs/tutorials/ryr1/ryr1-tutorial.toml
```

It gives the failed step's lines from the log, the error and the logs of the
programs the step ran. Fix the cause, then submit again from the stage that
failed, with a dry run first. Steps of that stage that had finished run again,
which is why `--overwrite` is needed:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial.toml \
    --from-stage cpu --until-stage gpu --overwrite --dry-run
```

`--overwrite` holds for that submission only, and its dry run lists the
recorded results it may replace. For 161, the stages are `cpu-161` and so on,
in `ryr1-tutorial-161.toml`. Do not resubmit a failure that will happen again
unchanged: find the cause first.

## 9. Look at the results before the MRA

Once the `gpu` jobs have finished, write the project's report. It reads the
project, runs nothing, and writes one PDF with a page per step and series:

```bash
cryoet report --project outputs/ryr1-tutorial --variant bin8 \
    --config docs/tutorials/ryr1/ryr1-tutorial.toml
```

It goes to `outputs/ryr1-tutorial/reports/report-bin8.pdf`. If you skipped 161,
add `--series 039 --series 096 --series 097 --series 125`, since the report
otherwise covers every series in the project. Copy it to your computer, or open
it on max. Look at:

- **the summary table**: each step's verdict per series. A `warning` does not
  stop the chain, so read each one;
- **tilt alignment**: the residual and the beads per tilt;
- **CTF**: which views were fitted. On these data a few of the highest tilts
  may find no rings. If a series has five such views or fewer, they are left
  out of the tomogram; with more, they stay in with an interpolated defocus,
  and the QC warns;
- **the tomogram and its membranes**: is the membrane segmentation on the
  membranes?
- **the picks**: on the tomogram, their scores against noise, and the galleries
  of picks by score, each turned to the template's orientation. Good picks look
  like the template; the low-score end is where the false ones are.

To see the picks in 3D, `visualize-picks` has written, per series,
`outputs/ryr1-tutorial/picks_view/<series>/`: a `.cxc` for ChimeraX (with
ArtiaX), and the tomogram `.mrc` with the picks as a `.mod` for 3dmod. Open one
or two series: do the picks sit on density at the membranes?

## 10. Submission 2: the MRA

The MRA (multi-reference alignment) classifies the particles of all the series
together: it aligns every particle to a few references and sorts them into
classes. With the lab's rules it starts from three references: two averages of
300 random particles each and one of pure noise, which collects the particles
that resemble nothing.

**With 161**, once both lanes have extracted their particles, the MRA of all
five comes from `ryr1-tutorial-all.toml`:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial-all.toml \
    --from-stage mra --dry-run
```

**Without 161**, from the main lane:

```bash
python3 scripts/submit_lanes.py docs/tutorials/ryr1/ryr1-tutorial.toml \
    --from-stage mra --dry-run
```

The dry run prints the run's folder, `outputs/ryr1-tutorial/mra/bin8-ssn-all`
(or `bin8-ssn`), and every setting with where it came from: the seeds, the five
rounds and their low-pass in Å, the diameter from `[pick]`, the mask and the
shifts. It refuses an MRA while a series has no particles yet, and while one of
your jobs still works on the series. When it shows no `would refuse:` line,
submit the same command without `--dry-run`.

The job runs on four A40s, and takes a few hours (2 h 48 min for 2,384 RyR1
particles at bin8, before the MRA was a lane stage). If it stops on its time
limit, submitting it again carries the run on from its last finished
iteration.

## 11. Read the classification report

The job writes its report beside the run:
`outputs/ryr1-tutorial/mra/bin8-ssn-all_classification_report.pdf` (or
`bin8-ssn_classification_report.pdf`). Its pages, in order:

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

## 12. A class to RELION 5 (optional)

The MRA sorts the particles; it does not give a resolution. For a gold-standard
refinement, a class that looks like RyR1 can go to RELION 5.
`scripts/export_relion.py` writes one folder that RELION runs as it is. It
submits nothing.

**The dry run first.** It names the class's particles after the run's last
finished iteration, the duplicates it leaves out, the two halves and a note per
series:

```bash
python3 scripts/export_relion.py --run outputs/ryr1-tutorial/mra/bin8-ssn-all \
    --class <C> --dry-run
```

`<C>` is the class number in the classification report (without 161, the run
is `bin8-ssn`). For these data the notes will say that the acquisition order
comes from the tilt order and that there is no dose. Here the tilt order is the
angle order, not the order the tilts were taken in. **Do not give
`--dose-per-tilt` for these data:** with the order unknown, a dose would weight
the wrong tilts if they were taken dose-symmetrically. Without a dose, RELION
weights no tilt by dose. Look at the two halves' sizes too: neighbours closer
than a box stay in one half and the halves are balanced over the whole class,
so they come out nearly equal unless one group of neighbours holds a large part
of the class; the dry run names the largest group.

**Then the export**, the same command without `--dry-run`. It writes
`outputs/ryr1-tutorial/relion/bin8-ssn-all-class<C>/`:

- the tilt series, tomograms and particles as RELION 5 reads them;
- the class average and the MRA's mask, in RELION's frame;
- `relion_commands.sh`: an Extract, then a gold-standard Refine3D with local
  searches from the MRA's poses, without symmetry;
- `README.md`, which says why each value is what it is.

Both run at the MRA's pixel, 13.44 Å here, so this first refinement checks the
export rather than reaching ~20 Å. Finer pixels come after, by extracting again
at a lower `--bin` (the README says how). At the end, the export prints the
`sbatch` line that runs the commands, with the folder filled in.

**RELION on max.** The module `relion/v5/gpu/sm86-cu11.7` does not start RELION
(its Open MPI lacks UCX). The group's RELION 5.0.1 build does, with another Open
MPI. Set both in the shell you submit from; the job takes them from there:

```bash
export RELION5_MODULE=openmpi/v5.0.5-xchu
export RELION5_HOME=/fast/AG_Kudryashev/software/relion-5.0.1/build-gpu_sm86_cu11.7
```

Then run the line the export printed, with max's GPU options in place of
`<GPU partition and --gres>`:

```bash
sbatch --nodes=1 --ntasks=3 --cpus-per-task=8 --time=1-00:00:00 \
    --partition=gpu --gres=gpu:a40:2 \
    --chdir=<the folder it printed> --wrap "./relion_commands.sh"
```

`relion_commands.sh` first checks that RELION starts, and stops, naming what is
missing, if it does not. The job's log, `slurm-<job id>.out`, is written in the
export's folder.

> **Status, 10 November 2026.** The export is new. A RyR1 class was exported on
> max that day; RELION's Extract and Refine3D have not run on max yet. Tell
> Anna what happens.

## Where to go next

- **Run one class again.** If a class looks like RyR1, its particles can be
  classified again, with other settings if you like: a copy of the lane whose
  `[mra]` names the run and its classes (`from_run`, `from_classes`), submitted
  with `--from-stage mra`. [docs/mra_in_lanes.md](../mra_in_lanes.md#again-on-a-class)
  has the keys.
- **A finer tomogram.** bin4 (6.72 Å/px) is the next step towards ~20 Å:
  `configs/lanes/ryr1-bin4.toml` and its comments.
- **A resolution.** A class in RELION 5, [step 12](#12-a-class-to-relion-5-optional).
  Halves split after a joint classification give the map's consistency, not its
  resolution: [docs/agent/playbook/averaging.md](../agent/playbook/averaging.md),
  entries V-3 and V-4.
- **Your own data.** The README's
  [Processing your own data on the cluster](../../README.md#processing-your-own-data-on-the-cluster),
  starting from `configs/lanes/example.toml`.

## When something refuses

| It says | What it means |
|---|---|
| `$MOTIONCOR3 is not set` (or another `$…`) | the shell lacks the group's tools: the `.bashrc` lines of step 1, then a new terminal |
| `… already exists but has no manifest for …` | the project was created without that series (step 4 was skipped); use a new project |
| a series is busy, naming a job | one of your jobs works on that series; wait until it has ended |
| `OVERWRITE is set in the environment` | `unset OVERWRITE`; give `--overwrite` to the one submission that needs it |
| `the MRA reads the particles extracted under 'bin8', and cannot` | a series has no particles yet: wait for its `gpu` job, or check why it failed |
| an existing MRA run with other settings | that name is taken: give the copy its own `[mra] name` |
| a job log stops at the GPU preflight | the job landed on a GPU too old for MemBrain-seg; the stages ask for A40s for this reason |
| CTFFIND 4 refused | `$CTFFIND5` is not set: on max, `ctffind` on PATH is version 4 |
