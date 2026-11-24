# RyR1 from frames to MRA classes on max, with the assistant

This tutorial runs the same data to the same end as
[ryr1-without-assistant.md](ryr1-without-assistant.md): five tilt-series of RyR1
in native membranes, from SerialEM TIFF frames to tomograms, picks on their
membranes and an MRA classification, and optionally a class exported to RELION 5. The difference is who types the commands.
Here the pipeline's terminal assistant reads the data, writes the lane files,
checks them and submits them; you answer its questions, read what it shows you,
and say yes or no.

The assistant runs in one of two clients. Both are started by
`scripts/assistant`, with the pipeline's own instructions, skills, guard and
permission rules ([assistant/README.md](../../assistant/README.md)):

- **Claude Code**, Anthropic's, for a Claude account: `scripts/assistant`;
- **Codex**, OpenAI's, for a ChatGPT account: `scripts/assistant --codex`.

Use the one your account is for. The steps are the same in both, and where the
two differ, this tutorial says so. On the pipeline's 13 evaluation cases,
graded blind on 14 November 2026, they did about equally well. What the
assistant may do, in either:

| It | without asking | after your explicit yes | never |
|---|---|---|---|
| read files, QC, logs, `squeue`, `sacct` | ✓ | | |
| read a data folder (`show_folder.py`), dry runs | ✓ | | |
| write a lane file | | ✓ (the client shows you the file and asks) | |
| submit, with `--overwrite`, cancel a job | | ✓ each its own yes | |
| export a class of an MRA run to RELION 5 | | ✓ | |
| run RELION, `relion_commands.sh` or `mpirun` (you run them, in a job) | | | ✓ |
| `sbatch` directly, git, change code, docs or the journal | | | ✓ |

The submission scripts hold it to its yes themselves: a submission from the
assistant must repeat the dry run you were shown, unchanged, within 30
minutes, with the reason you gave, and the project's journal records it.

> **Status, 24 November 2026.** In tests of the other tutorial on max, its lane
> ran through the MRA on four of these series on 19–20 November 2026, and
> through fine alignment on all five on 23 November; 161 had been left out until
> then, for a crash in IMOD's `beadtrack` that the pipeline now avoids (playbook
> entry A-2). The assistant has run on max in Claude Code since 5 November, and
> in Codex on 14 November, where Codex's sandbox works. Tell Anna what
> happened.

## What you need

- **An account on max.**
- **A folder on `/fast` with room for the data and the outputs.** The clone
  lives there, and the outputs are written inside it. The frames are copied
  into it too: measure them with `du -sh` before copying (step 1 says how). At
  bin8 the outputs add about 30 GB of motion-corrected sums and stacks, plus
  the tomograms.
- **A terminal on max**, on a login node or in an interactive job such as
  `slogin`. You, and the assistant, run everything there. The heavy work runs
  in Slurm jobs, which the pipeline's scripts submit.
- **An account for one of the two clients.**
  - Claude Code: a Claude subscription that includes it, that is Pro, Max,
    Team or Enterprise.
  - Codex: a ChatGPT plan; Codex is part of every plan, from Free to
    Enterprise. Signing in from max uses a device code, which must be enabled
    in ChatGPT's security settings, or by your workspace's admin. Step 2 says
    what to do if it is not.
- **Terms that exclude training on your data.** Set them before the first
  session.
  - Claude: a Team or Enterprise seat has them by default. On a personal Pro
    or Max plan, turn off the setting that lets Anthropic use your chats to
    improve its models (claude.ai → Settings → Privacy).
  - ChatGPT: a Business, Enterprise or Edu workspace has them by default. On a
    personal plan, turn off **Improve the model for everyone** (ChatGPT →
    Settings → Data controls). Codex has a separate setting, **Include
    environments**, in Codex's settings, which that one does not change: turn
    it off too.
- **To know what leaves the cluster.** In a session, the text the assistant
  reads goes to Anthropic, or to OpenAI in Codex: log excerpts, file paths,
  lane files, QC records, and any picture it opens. The frames and the
  tomograms do not.
- **About an hour for the setup**, then waiting for the queue. While the jobs
  run, the session can be closed.

## 1. Set up the pipeline and the data (once)

Do steps 1, 2 and 4 of the other tutorial, then come back:

1. [Clone and install](ryr1-without-assistant.md#1-clone-and-install-once),
   including the `.bashrc` lines that name your folder on `/fast` and the
   `build_library.py` check;
2. [Copy the data](ryr1-without-assistant.md#2-copy-the-data) to
   `data/ryr1/frames`;
4. [Create the project, with all five series](ryr1-without-assistant.md#4-create-the-project-with-all-five-series),
   `outputs/ryr1-tutorial`, with `cryoet init`.

Step 4 is yours, not the assistant's. A series left out can join later, with
the `cryoet init … --add-series` command the dry run prints, which keeps the
rest of the project as it is.

**The assistant can help with this setup.** It needs only the clone and its
client: clone, do steps 2 and 3 below, and ask it to check your setup or to take
you through it. It checks with `scripts/show_setup.py`, which writes nothing:
the room where the environments, downloads and temporary files go, the
versions of the tools, and the GPU, its driver and CUDA. You run each install
command yourself, in your own terminal, and it checks after each.

## 2. Install the client and sign in (once)

Install the client your account is for, Claude Code or Codex.

**Claude Code.** If `claude --version` already prints a version on max, skip
to the sign-in. Otherwise install it into your home folder:

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

**Codex.** If `codex --version` already prints a version on max, skip to the
sign-in. Otherwise install it into your home folder with OpenAI's
installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Open a new terminal if `codex --version` does not find it afterwards. Then
sign in once, through the launcher, from your clone:

```bash
cd "$CRYOET_WORK/cryoet-pipeline"
scripts/assistant --codex --login
```

The assistant's Codex keeps its own sign-in and settings in
`~/.cryoet-assistant/codex`, apart from any Codex of your own, so sign in this
way and not with `codex login`. Over SSH the launcher asks for a device code:
open the link it prints in a browser on your computer, sign in with your
ChatGPT account (your workspace's, if you have one), and enter the one-time
code it shows. If device code login is not enabled for you, the sign-in fails
and the launcher names the other way, a browser login through an SSH tunnel;
[assistant/README.md](../../assistant/README.md#signing-in-to-codex-on-a-cluster)
says how. Treat `~/.cryoet-assistant/codex/auth.json` like a password.

If Codex is the only client you will use, this line in `~/.bashrc` makes
`scripts/assistant` start Codex without `--codex`:

```bash
export CRYOET_ASSISTANT_BACKEND=codex
```

From now on, start Codex for the pipeline only through the launcher: it is
what gives it the pipeline's rules.

**Only if your home is really short of space.** The clients keep their
programs and sessions in your home: roughly half a gigabyte for Claude Code
and a gigabyte for Codex, plus the sessions. A couple of gigabytes free is
enough. If you do not have that, Codex can live in your folder on `/fast`
instead. This has not been tried on max yet: tell Anna how it went.

- Install it with this command instead of the one above. Its program goes to
  your folder, and only a link stays in `~/.local/bin`:

  ```bash
  curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_HOME="$CRYOET_WORK/codex" sh
  ```

  Give `CODEX_HOME` to this command only, not in `~/.bashrc`, where it would
  move any Codex of your own as well.
- Before you sign in, add this line to `~/.bashrc` and open a new terminal.
  The assistant's sign-in and sessions then live in your folder, which the
  launcher keeps readable by you alone. Keep the line: a start without it
  looks for the sign-in in your home.

  ```bash
  export CRYOET_ASSISTANT_CODEX_HOME=$CRYOET_WORK/cryoet-assistant/codex
  ```

For Claude Code, this tutorial has no such way yet.

## 3. Start the assistant

On a login node or in an `slogin` job, from the root of your clone, with the
environment active (the assistant runs the pipeline's scripts in this shell,
and your jobs take this shell's tool variables):

```bash
cd "$CRYOET_WORK/cryoet-pipeline"
micromamba activate cryoet-pipeline
scripts/assistant
```

For Codex, the last line is `scripts/assistant --codex`.

The launcher prints the reference library it gives the session
(`library snapshot …, read-only`): manuals of IMOD, pytom-match-pick and the
other tools, at the versions installed on max.

- **Claude Code** may ask whether you trust the folder. It is your clone: yes.
- **Codex** opens on a **Folder access** screen at every start: choose
  **1. Open restricted** (Enter). The launcher marks the clone untrusted on
  purpose, so that no settings inside the clone apply; the assistant's own
  rules and guard come from its Codex home and act all the same. If you are
  ever offered to trust the folder, do not. If Codex warns about
  `--dangerously-bypass-hook-trust`, that is expected: it is how the launcher
  makes sure the guard runs.

Then:

- `scripts/assistant -c` (or `scripts/assistant --codex -c`) continues your
  last session, after a lost connection or the next day. The jobs do not
  depend on the session: they run whether it is open or not.
- The model. In Claude Code the default is `claude-opus-5-5`, and
  `scripts/assistant --model claude-sonnet-5-5` starts a cheaper one. In Codex
  it is `gpt-6.1-sol`, reasoning at `high`; `scripts/assistant --codex --effort
  medium` makes it reason less. Give a full model ID, and to change the model
  start a new session rather than using `/model`, so that the journal names the
  right one.
- Write to it in English or in Russian; it answers in the language you use.

## 4. Have it read the data and write the lane

Ask in your own words. A request that gives it what it cannot read for itself:

> I copied RyR1 frames to data/ryr1/frames: SerialEM TIFF, no mdocs, gain.mrc
> beside them. I created the project outputs/ryr1-tutorial with cryoet init,
> all five series. I want a bin8 lane up to the MRA for all five. I want to look
> at the picks before the MRA.
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
   | one series first, or all | all five |
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
   a comment (you, a header, a rule, or a guess, labelled as one). The client
   shows you the file and asks before writing it. Read it, then approve it
   once; in Codex the question is *Would you like to make the following
   edits?*, and the answer **Yes, proceed**.
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

In Codex there is one more question. A submission runs outside Codex's
sandbox, so Codex shows the command and asks *Would you like to run the
following command?* Check that it is the dry run's command without
`--dry-run`, with `--reason`, and answer **Yes, proceed**. Never the answer
that says *don't ask again*: it would let every command that starts the same
way run without asking, for the rest of the session.

## 6. While the jobs run

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

## 7. Look at the picks

When the `gpu` jobs have finished, ask it for the project's report:

> Write the report for ryr1-tutorial.

`cryoet report` is not on its list of commands that run without asking, so
the client asks you before running it (Codex, to run it outside its sandbox,
which writes nothing); it only reads the project and writes a PDF,
`outputs/ryr1-tutorial/reports/report-bin8.pdf`. Or run it yourself, in another
terminal, as in
[step 8 of the other tutorial](ryr1-without-assistant.md#8-look-at-the-results-before-the-mra),
which also says what to look at.

Read the PDF yourself: the alignment, the CTF fits, the membranes, the picks on
the tomogram and their galleries. Ask the assistant about anything in it. What
it says about a picture is an impression, and it says so.

## 8. The MRA

When you are satisfied with the picks:

> Run the MRA on the particles of the five series.

It submits the lane's `mra` stage with `--from-stage mra`. The dry run shows
the run's folder and every setting with its source; with no `[mra]` section
they are the lab's rules: three references (two averages of 300 particles, one
of noise), five rounds from coarse to fine, the 340 Å diameter from `[pick]`,
the mask from it. It refuses while a series has no particles, or while one of
your jobs still works on the series.

The job runs on four A40s, for a few hours: in a test of the other tutorial on
max, on four of the series, its first two iterations took 40 and 83 minutes and
the last three about 2 minutes together. If it stops on its time limit, the
same submission carries the run on from its last finished iteration.

## 9. Read the classification report

The job writes `outputs/ryr1-tutorial/mra/<run name>_classification_report.pdf`.
[Step 10 of the other tutorial](ryr1-without-assistant.md#10-read-the-classification-report)
lists its pages and what to look for. Then, for instance:

> Class 2 looks like RyR1. Run only its particles again.

It writes a copy of the lane whose `[mra]` names the run and the class
(`from_run`, `from_classes`), with any settings you change, and submits it with
`--from-stage mra` after the dry run and your yes.

## 10. A class to RELION 5 (optional)

The MRA sorts the particles; it does not give a resolution. For a gold-standard
refinement, a class that looks like RyR1 can go to RELION 5:

> Export class 2 of bin8-ssn to RELION.

Its `relion-export` skill runs `scripts/export_relion.py` as a dry run and
shows you what it printed:

- the class's particles after the run's last finished iteration;
- the duplicates it leaves out;
- the two halves. Neighbours closer than a box stay in one half and the halves
  are balanced over the whole class, so they come out nearly equal unless one
  group of neighbours holds a large part of the class; the dry run names the
  largest group;
- a note per series. For these frames, which have no mdocs, the notes say that
  the acquisition order comes from the tilt order, here the angle order, and
  that there is no dose.

**For these data, no dose.** If it offers `--dose-per-tilt`, say no: with the
order unknown, a dose would weight the wrong tilts if they were taken
dose-symmetrically. Without a dose, RELION weights no tilt by dose.

After your yes it exports, to
`outputs/ryr1-tutorial/relion/bin8-ssn-class2/`:

- `relion_commands.sh`: an Extract, then a gold-standard Refine3D with local
  searches from the MRA's poses, without symmetry, both at the MRA's pixel
  (13.44 Å here). This first refinement checks the export rather than reaching
  ~20 Å; finer pixels come after, by extracting again at a lower `--bin`;
- `README.md`, which says why each value is what it is.

It then gives you the `sbatch` line the export printed. RELION is yours to run:
the assistant never runs RELION, `relion_commands.sh` or `mpirun`, and cannot
call `sbatch`. In your own terminal (not with `!`), set RELION for the job. On
max the module `relion/v5/gpu/sm86-cu11.7` does not start RELION (its Open MPI
lacks UCX); the group's RELION 5.0.1 build does, with another Open MPI:

```bash
export RELION5_MODULE=openmpi/v5.0.5-xchu
export RELION5_HOME=/fast/AG_Kudryashev/software/relion-5.0.1/build-gpu_sm86_cu11.7
```

Then run that line, with max's GPU options in place of
`<GPU partition and --gres>`:

```bash
sbatch --nodes=1 --ntasks=3 --cpus-per-task=8 --time=1-00:00:00 \
    --partition=gpu --gres=gpu:a40:2 \
    --chdir=<the folder it gave you> --wrap "./relion_commands.sh"
```

`relion_commands.sh` first checks that RELION starts, and stops, naming what is
missing, if it does not. The job's log, `slurm-<job id>.out`, is written in the
export's folder; the assistant can read it for you.

> **Status, 10 November 2026.** The export is new. A RyR1 class was exported on
> max that day; RELION's Extract and Refine3D have not run on max yet. Tell
> Anna what happens.

## Good to know

- **It submits only through `submit_lanes.py` and `submit_recipe.py`.** It
  cannot call `sbatch`, and it will not write a batch script for an experiment
  the recipes do not cover: it stops and asks a person instead.
- **Do not get round its rules.** In particular, do not run a submission,
  `scancel` or `--overwrite` with the `!` prefix, which both clients have: in
  Claude Code `!` runs a command outside the rules and the guard, and in Codex
  that has not been checked. Run such a command in your own terminal, or
  through the assistant with its dry run and your yes. In Codex, do not answer
  a question with *Yes, and don't ask again …*: the launcher removes that
  answer at the next start, but until then it lets every command that starts
  the same way run without asking.
- **It does not change the pipeline.** If it suspects a bug, it says so with its
  evidence, for a person to fix.
- **Downloads are yours.** For an EMPIAR entry it writes and checks the layout
  and gives you the `cryoet download-empiar` command, which can run for hours,
  to run in your own shell.
- **Every submission and every export it makes is in the journal**,
  `outputs/ryr1-tutorial/journal.jsonl`, with the client, the model and your
  reason; a submission also with the lane file's text.
- **When something refuses**, the table at the end of the
  [other tutorial](ryr1-without-assistant.md#when-something-refuses) says what
  it means; the assistant shows you the refusal and explains it.
