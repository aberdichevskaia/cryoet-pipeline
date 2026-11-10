# RyR1 from frames to MRA classes on max, with the assistant

This tutorial runs the same data to the same end as
[ryr1-without-assistant.md](ryr1-without-assistant.md): five tilt-series of RyR1
in native membranes, from SerialEM TIFF frames to tomograms, picks on their
membranes and an MRA classification. The difference is who types the commands.
Here the pipeline's terminal assistant reads the data, writes the lane files,
checks them and submits them; you answer its questions, read what it shows you,
and say yes or no.

The assistant is Claude Code, started by `scripts/assistant` with the
pipeline's own instructions, skills and permission rules
([assistant/README.md](../../assistant/README.md)). What it may do:

| It | without asking | after your explicit yes | never |
|---|---|---|---|
| read files, QC, logs, `squeue`, `sacct` | ✓ | | |
| read a data folder (`show_folder.py`), dry runs | ✓ | | |
| write a lane file | | ✓ (Claude Code asks you to approve the file) | |
| submit, with `--overwrite`, cancel a job | | ✓ each its own yes | |
| `sbatch` directly, git, change code, docs or the journal | | | ✓ |

The submission scripts hold it to its yes themselves: a submission from the
assistant must repeat the dry run you were shown, unchanged, within 30
minutes, with the reason you gave, and the project's journal records it.

> **Status, 10 November 2026.** Every pipeline step up to the particles has run
> on these data on max, and the assistant has run on max's login nodes. The MRA
> as a stage of a lane is new (10 November 2026) and has not run on max yet.
> Tell Anna what happened.

## What you need

- **An account on max.**
- **A folder on `/fast` with room for the data and the outputs.** The clone
  lives there, and the outputs are written inside it. The frames are copied
  into it too: measure them with `du -sh` before copying (step 1 says how). At
  bin8 the outputs add about 30 GB of motion-corrected sums and stacks, plus
  the tomograms.
- **A terminal on a login node.** You, and the assistant, run everything there.
  The heavy work runs in Slurm jobs, which the pipeline's scripts submit.
- **A Claude subscription that includes Claude Code**: Pro, Max, Team or
  Enterprise.
- **Terms that exclude training on your data.** A Team or Enterprise seat has
  them by default. On a personal Pro or Max plan, turn off the setting that
  lets Anthropic use your chats to improve its models (claude.ai → Settings →
  Privacy) before the first session.
- **To know what leaves the cluster.** In a session, the text the assistant
  reads goes to Anthropic: log excerpts, file paths, lane files, QC records,
  and any picture it opens. The frames and the tomograms do not.
- **About an hour for the setup**, then waiting for the queue. While the jobs
  run, the session can be closed.

## 1. Set up the pipeline and the data (once)

Do steps 1, 2 and 4 of the other tutorial, then come back:

1. [Clone and install](ryr1-without-assistant.md#1-clone-and-install-once),
   including the two `.bashrc` lines and the `build_library.py` check;
2. [Copy the data](ryr1-without-assistant.md#2-copy-the-data) to
   `data/ryr1/frames`;
4. [Create the project, with all five series](ryr1-without-assistant.md#4-create-the-project-with-all-five-series),
   `outputs/ryr1-tutorial`, with `cryoet init`.

Step 4 is yours, not the assistant's, for a reason worth knowing: a project is
created once, with the series it will ever hold. Series 161 will be processed
apart from the other four (step 6 below), so all five must be in the project
from the start.

## 2. Install Claude Code and sign in (once)

If `claude --version` already prints a version on the login node, skip to the
sign-in. Otherwise install it into your home folder:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

It installs `claude` in `~/.local/bin`; open a new terminal if `claude
--version` does not find it. Then sign in once, from any folder:

```bash
claude
```

Claude Code prints a link: open it in a browser on your computer, sign in with
your Claude account, and paste the code it gives you back into the terminal.
Type `/exit` to leave. From now on, start Claude Code only through the
launcher below: it is what gives it the pipeline's rules.

## 3. Start the assistant

On a login node, from the root of your clone, with the environment active (the
assistant runs the pipeline's scripts in this shell, and your jobs take this
shell's tool variables):

```bash
cd /fast/AG_Kudryashev/$USER/cryoet-pipeline
micromamba activate cryoet-pipeline
scripts/assistant
```

The launcher prints the reference library it gives the session
(`library snapshot …, read-only`): manuals of IMOD, pytom-match-pick and the
other tools, at the versions installed on max. If Claude Code asks whether you
trust the folder, it is your clone: yes.

- `scripts/assistant -c` continues your last session, after a lost connection
  or the next day. The jobs do not depend on the session: they run whether it
  is open or not.
- `scripts/assistant --model claude-sonnet-5-5` starts a cheaper model. The
  default is `claude-opus-5-5`. Give a full model ID, and to change the model
  start a new session rather than using `/model`, so that the journal names the
  right one.
- Write to it in English or in Russian; it answers in the language you use.

## 4. Have it read the data and write the lane

Ask in your own words. A request that gives it what it cannot read for itself:

> I copied RyR1 frames to data/ryr1/frames: SerialEM TIFF, no mdocs, gain.mrc
> beside them. I created the project outputs/ryr1-tutorial with cryoet init,
> all five series. I want a bin8 lane up to the MRA for 039, 096, 097 and 125;
> 161 comes later in its own copy. I want to look at the picks before the MRA.
> Read the folder and ask me what you cannot read.

What it does, following its `lane-config` skill:

1. **Reads the folder** with `scripts/show_folder.py`, writes the file-name
   pattern, and shows you a table: series, tilts, angle range, frames per tilt,
   pixel size. Check it against what was collected: five series of 29 tilts,
   −42° to +42°, 10 frames, 0.84 Å/px. Say so, or say what is wrong.
2. **Reads the closest lane at the same site.** Here that is likely
   `configs/lanes/ryr1-bin8.toml`, which ran on these very frames: its comments
   say where each value came from.
3. **Asks what is left**, in one message, each question with its default. Your
   answers for these data:

   | It may ask | Answer |
   |---|---|
   | what to pick, from which map | RyR1, EMD-54612 |
   | its widest diameter | 340 Å, read off the map; the map is C4 about z, so `z_axis_symmetry = 4` for picking (only shortens the search) |
   | gold beads, size | yes, 10 nm |
   | membrane, how far | yes, RyR1 spans the SR membrane; search within ~17.5 nm of it |
   | gain-corrected? | no: `data/ryr1/frames/gain.mrc`, rotation 0, flip 1 |
   | Cs | 2.7 mm; most likely 300 kV and 2-7 µm nominal defocus too (EMD-54612's record, not confirmed for these files) |
   | tilt-axis angle | 84.0°; its sign is not known |
   | one series first, or all | the four: 039 096 097 125 |
   | where the outputs go | `outputs/ryr1-tutorial`, which exists with all five series |
   | MRA right after picking, or look first | look first |
   | particles per tomogram | not known: keep the default, twice pytom's recommendation |

   One thing it will not ask, and these data need: the CTF search window. Its
   rule is the code's default, ±1000 Å, which fits only the views near zero
   tilt on these data, because the defocus moves with the tilt. Tell it:
   `[ctf] defocus_search_window_angstrom = 10000`, as in `ryr1-bin8.toml`, and
   a first-fit defocus range of 1-8 µm. The frames' super-resolution binning,
   `[motion] ftbin = 2`, it sets itself from the frame size.
4. **Writes the lane** into `configs/lanes/`, with the source of every value in
   a comment (you, a header, a rule, or a guess, labelled as one). Claude Code
   shows you the file and asks before writing it. Read it.
5. **Checks it** with `run_series.py --list-steps`, which refuses any key the
   pipeline does not read.

Compare what it wrote with the tutorial's own lane,
[`ryr1/ryr1-tutorial.toml`](ryr1/ryr1-tutorial.toml). Differences are not
errors by themselves: ask it to explain any you do not understand.

## 5. Dry run, yes, submission

It then runs the dry run and shows you what it printed: four jobs (`motion`,
`align`, `cpu`, `gpu`), each waiting for the one before, the series, and any
`would refuse:` line with what it means. It submits with
`--until-stage` set to the stage that ends at particle extraction, because you
said you would look at the picks first.

It asks whether to submit. Answer with a clear yes. It then runs the same
command without `--dry-run` and with `--reason`, and reports the job IDs. If
anything changed between the dry run and your yes, it has to run the dry run
again and ask again; the script refuses otherwise.

## 6. Series 161

Ask for it once the main lane is submitted:

> Process 161 in the same project, in its own copy of the lane: IMOD's
> beadtrack segfaults on 161 at tracking binning 2, and at 3 it aligned. Name
> its stages with -161 so I can tell its jobs apart. No MRA in that copy.

It writes the copy, shows you the difference from the main lane (`series`,
`tracking_binning`, the stage names), dry-runs it and asks. Different series of
one project may run at the same time, so the dry run should not refuse it as
busy. The tutorial's version of this copy is
[`ryr1/ryr1-tutorial-161.toml`](ryr1/ryr1-tutorial-161.toml).

## 7. While the jobs run

Ask it how things stand, as often as you like:

> How is ryr1-tutorial doing?

It runs `scripts/show_project.py` (its `project-status` skill) and summarises
which series are done, which are running, which failed and where, and the QC
warnings. It can also answer what a QC warning means, citing the file and the
line it read.

**If a job fails**, ask why:

> The cpu job of ryr1-tutorial failed. Why, and what do we do?

It reads the failure with `scripts/show_failure.py`, checks the playbook
(`docs/agent/playbook/`) for a problem met before, and proposes a fix from the
recipes known to be safe, usually a resubmission from the failed stage with
`--overwrite`. That needs its dry run, then your yes to the submission and to
`--overwrite`. It will not rerun a failure that reproduces without changing
something first.

## 8. Look at the picks

When the `gpu` jobs have finished, ask it for the project's report:

> Write the report for ryr1-tutorial.

`cryoet report` is not on its list of commands that run without asking, so
Claude Code asks you to approve it; it only reads the project and writes a PDF,
`outputs/ryr1-tutorial/reports/report-bin8.pdf`. Or run it yourself, in another
terminal, as in
[step 9 of the other tutorial](ryr1-without-assistant.md#9-look-at-the-results-before-the-mra),
which also says what to look at.

Read the PDF yourself: the alignment, the CTF fits, the membranes, the picks on
the tomogram and their galleries. Ask the assistant about anything in it. What
it says about a picture is an impression, and it says so.

## 9. The MRA

When you are satisfied with the picks, and 161's particles are extracted:

> Run the MRA on the particles of all five series.

The main lane lists four series, so it writes a copy for the MRA with all five,
and a run name of its own, and submits it with `--from-stage mra`. The dry run
shows the run's folder and every setting with its source; with no `[mra]`
section they are the lab's rules: three references (two averages of 300
particles, one of noise), five rounds from coarse to fine, the 340 Å diameter
from `[pick]`, the mask from it. It refuses while a series has no particles,
or while one of your jobs still works on the series. The tutorial's version is
[`ryr1/ryr1-tutorial-all.toml`](ryr1/ryr1-tutorial-all.toml).

The job runs on four A40s, for a few hours. If it stops on its time limit, the
same submission carries the run on from its last finished iteration.

## 10. Read the classification report

The job writes `outputs/ryr1-tutorial/mra/<run name>_classification_report.pdf`.
[Step 11 of the other tutorial](ryr1-without-assistant.md#11-read-the-classification-report)
lists its pages and what to look for. Then, for instance:

> Class 2 looks like RyR1. Run only its particles again.

It writes a copy of the lane whose `[mra]` names the run and the class
(`from_run`, `from_classes`), with any settings you change, and submits it with
`--from-stage mra` after the dry run and your yes.

## Good to know

- **It submits only through `submit_lanes.py` and `submit_recipe.py`.** It
  cannot call `sbatch`, and it will not write a batch script for an experiment
  the recipes do not cover: it stops and asks a person instead.
- **Do not get round its rules.** In particular, do not run a submission,
  `scancel` or `--overwrite` with Claude Code's `!` prefix: `!` runs a command
  outside the rules and the guard. Run such a command in your own terminal, or
  through the assistant with its dry run and your yes.
- **It does not change the pipeline.** If it suspects a bug, it says so with its
  evidence, for a person to fix.
- **Downloads are yours.** For an EMPIAR entry it writes and checks the layout
  and gives you the `cryoet download-empiar` command, which can run for hours,
  to run in your own shell.
- **Every submission it makes is in the journal**,
  `outputs/ryr1-tutorial/journal.jsonl`, with the model, your reason and the
  lane file's text.
- **When something refuses**, the table at the end of the
  [other tutorial](ryr1-without-assistant.md#when-something-refuses) says what
  it means; the assistant shows you the refusal and explains it.
