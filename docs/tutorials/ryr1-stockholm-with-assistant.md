# RyR1 from frames to MRA classes on cryoem.se, with the assistant

This tutorial takes you from an empty home folder to class averages on
cryoem.se, the cluster of the Stockholm workshop, with the pipeline's terminal
assistant. You clone the repository and install the assistant's client. From
there the assistant works beside you: it checks what the cluster has, finds its
programs and gives you the install commands one at a time; it reads the frames,
draws the raw tilts into a PDF that shows the gold beads and the tilt axis,
writes the lane file, checks it and submits it. You run what it hands you,
answer its questions, read what it shows you, and say yes or no. The data are
RyR1 in native membranes, five tilt-series of SerialEM TIFF frames with gold
beads.

```text
clone + client ─▶ the assistant checks the cluster, you install
   ─▶ it reads the frames and draws the raw tilts: beads and axis   (a PDF)
   ─▶ you create the project
   ─▶ it writes the lane; dry run, your yes, submission 1
        frames → tomograms → picks → particles                     (4 Slurm jobs)
   ─▶ you look at the report and the picks
   ─▶ optional: a branch, 161 aligned by AreTomo2                  (2 Slurm jobs)
   ─▶ submission 2: MRA classification                             (1 Slurm job, 1 GPU)
   ─▶ you read the classification report
   ─▶ optional: a class exported to RELION 5
```

You will end with tomograms of the five series at 13.44 Å/px, RyR1 picks on
their membranes, and an MRA classification of the picks: a few class averages
and a noise class that collects the false picks.

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
| read files, QC, logs, `squeue`, `sacct`, `sinfo` | ✓ | | |
| check the installation (`show_setup.py`) | ✓ | | |
| read a data folder, draw its raw tilts (`show_folder.py`, `show_tilts.py`), dry runs | ✓ | | |
| write a lane file | | ✓ (the client shows you the file and asks) | |
| submit, with `--overwrite`, cancel a job | | ✓ each its own yes | |
| export a class of an MRA run to RELION 5 | | ✓ | |
| install: pip, micromamba, `module load`, an installer; edit `~/.bashrc` (you do, with its commands) | | | ✓ |
| `srun`, `sbatch` directly, git, change code, docs or the journal | | | ✓ |
| run RELION, `relion_commands.sh` or `mpirun` (you run them, in a job) | | | ✓ |

The submission scripts hold it to its yes themselves: a submission from the
assistant must repeat the dry run you were shown, unchanged, within 30
minutes, with the reason you gave, and the project's journal records it.

> **Status, 24 November 2026.** On cryoem.se the assistant ran in Claude Code
> under a course account from 19 to 24 November 2026: it took that account
> through the installation on 22–23 November, with fixes to the pipeline in
> between, and wrote a lane for these data. That lane ran on series 161 alone
> on 24 November, from the frames to the tomogram, with its CPU stages on
> cryo5000 rather than cs-cpu, and so did the branch of step 12. Not run there
> yet: Codex, the CPU stages on cs-cpu, the `gpu` stage (membranes, template
> matching, particles), the MRA on one A5000, and the tilt preview of step 6 on
> these TIFF frames. At MDC the same lane ran through the MRA on four of the
> series on 19–20 November, and through fine alignment on all five on 23
> November. Tell Anna what happened.

## What you need

- **A course account on cryoem.se** and an SSH connection to
  `login2.cryoem.se`: a terminal, or VS Code with Remote SSH. You, and the
  assistant, run everything there. The heavy work runs in Slurm jobs, which the
  pipeline's scripts submit.
- **Room in your home folder**, which has no quota on course accounts. The
  environments and the downloads take about 27 GB (one course account, 24
  November 2026). The outputs at bin8 add the motion-corrected sums and stacks
  and the tomograms: 8.4 GB for 161 alone up to the tomogram, with both
  alignments of step 12, so plan for some 40 GB for the five series. The
  clients add about half a gigabyte (Claude Code) or a gigabyte (Codex), plus
  their sessions.
- **An account for one of the two clients.**
  - Claude Code: a Claude subscription that includes it, that is Pro, Max,
    Team or Enterprise.
  - Codex: a ChatGPT plan; Codex is part of every plan, from Free to
    Enterprise. Signing in from the cluster uses a device code, which must be
    enabled in ChatGPT's security settings, or by your workspace's admin.
    Step 2 says what to do if it is not.
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
  lane files, QC records, the output of the checks, and any picture it opens.
  The frames and the tomograms do not.
- **On your laptop**: `scp` or another SFTP client, to copy PDFs and folders
  from the cluster; and ChimeraX with ArtiaX, or IMOD's `3dmod`, to see the
  picks in 3D.
- **One to two hours for the setup**, then waiting for the queue. While the jobs
  run, the session can be closed.

## 1. Clone, and connect the course's shared files (once)

Everything this tutorial installs and writes goes into one folder in your home,
`~/cryoet`: the environments, the downloads, and the clone with its outputs.
Clone into it:

```bash
mkdir -p ~/cryoet
cd ~/cryoet
git clone https://github.com/aberdichevskaia/cryoet-pipeline.git
```

The assistant never clones; everything else in the installation it takes you
through.

The course shares two files through a module, `tomobear/2026`: MemBrain-seg's
model, which segments the membranes, and the assistant's reference library, the
manuals of IMOD, pytom-match-pick and the other tools at the versions installed
here. Add this line at the end of `~/.bashrc`, outside any check for the host
name:

```bash
module load tomobear/2026
```

Open a new terminal and check that it set both:

```bash
echo "$MEMBRAIN_CKPT" "$CRYOET_LIBRARY"
```

It should print `/mnt/cephfs/home/tomobear/models/MemBrain_seg_v10_alpha.ckpt`
and `/mnt/cephfs/home/tomobear/cryoet-library`. The library has to be there
before the assistant starts: the launcher hands it to the session at the start.

## 2. Install the client and sign in (once)

Install the client your account is for, Claude Code or Codex, into your home
folder.

**Claude Code.** If `claude --version` already prints a version, skip to the
sign-in. Otherwise:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

It installs `claude` in `~/.local/bin`; open a new terminal if `claude
--version` does not find it. Then sign in once, from any folder:

```bash
claude
```

Claude Code prints a link: open it in a browser on your laptop, sign in with
your Claude account, and paste the code it gives you back into the terminal.
Type `/exit` to leave. From now on, start Claude Code only through the
launcher of step 3: it is what gives it the pipeline's rules.

**Codex.** If `codex --version` already prints a version, skip to the sign-in.
Otherwise install it with OpenAI's installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Open a new terminal if `codex --version` does not find it afterwards. Then
sign in once, through the launcher, from your clone:

```bash
cd ~/cryoet/cryoet-pipeline
scripts/assistant --codex --login
```

The assistant's Codex keeps its own sign-in and settings in
`~/.cryoet-assistant/codex`, apart from any Codex of your own, so sign in this
way and not with `codex login`. Over SSH the launcher asks for a device code:
open the link it prints in a browser on your laptop, sign in with your ChatGPT
account (your workspace's, if you have one), and enter the one-time code it
shows. If device code login is not enabled for you, the sign-in fails and the
launcher names the other way, a browser login through an SSH tunnel;
[assistant/README.md](../../assistant/README.md#signing-in-to-codex-on-a-cluster)
says how. Treat `~/.cryoet-assistant/codex/auth.json` like a password.

If Codex is the only client you will use, this line in `~/.bashrc` makes
`scripts/assistant` start Codex without `--codex`:

```bash
export CRYOET_ASSISTANT_BACKEND=codex
```

From now on, start Codex for the pipeline only through the launcher: it is
what gives it the pipeline's rules.

## 3. Start the assistant

From the root of your clone, in a new terminal:

```bash
cd ~/cryoet/cryoet-pipeline
scripts/assistant
```

For Codex, the last line is `scripts/assistant --codex`. There is no
pipeline environment yet: until step 5, the assistant's checks run on the
system's Python.

The launcher prints the reference library it gives the session
(`library snapshot …, read-only`). If it prints no such line, `CRYOET_LIBRARY`
is not set in this terminal: the module of step 1 is not loaded.

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
  last session: after a lost connection, the next day, or a new terminal with
  new lines in `~/.bashrc`. The jobs do not depend on the session: they run
  whether it is open or not.
- The model. In Claude Code the default is `claude-opus-5-5`, and
  `scripts/assistant --model claude-sonnet-5-5` starts a cheaper one. In Codex
  it is `gpt-6.1-sol`, reasoning at `high`; `scripts/assistant --codex --effort
  medium` makes it reason less. Give a full model ID, and to change the model
  start a new session rather than using `/model`, so that the journal names the
  right one.
- Write to it in English or in Russian; it answers in the language you use.

## 4. Install with the assistant

Ask in your own words, with what it cannot know:

> I am on cryoem.se for the Stockholm course. My clone is ~/cryoet/cryoet-pipeline,
> and everything I install should go into ~/cryoet. I want to run the lane
> docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml. The module tomobear/2026 is
> loaded. Check what is there and take me through the installation, one
> command at a time.

What it does, following its `install` skill:

1. **Checks** with `scripts/show_setup.py`, which writes nothing: the clone,
   the room where the environments, downloads and temporary files go, the
   pipeline's environment, every tool the lane needs with its version, and
   which of the cluster's GPUs each program can run on. Your home has no
   quota; if the check says it cannot read one, that is right.
2. **Finds the programs** the cluster has installed (`show_setup.py --find`):
   IMOD, CTFFIND 5, MotionCor3 and AreTomo2 are installed once for everyone,
   under `/opt/cryoem`, with modules. It proposes the lines that name them for
   your `~/.bashrc`.
3. **Gives you the install commands**, one at a time: micromamba's own lines,
   and three environments, one for the pipeline, one for pytom-match-pick and
   one for MemBrain-seg. You run each in your own terminal, not in the session,
   and tell it how it ended; it checks again after each. Installs run for
   minutes and download gigabytes; if pip stops on a broken download, run the
   same command again, since what it has downloaded is cached.
4. **Has you start again after a change to `~/.bashrc`.** The session's shell
   is the one it started in, and a line added since is not in it. Open a new
   terminal, `cd ~/cryoet/cryoet-pipeline`, and continue the session with
   `scripts/assistant -c`.
5. **Has the setup checked on a GPU node.** It gives you an `srun` line that
   runs the same check for a few minutes on one of the course's cards, which
   compares the driver with the CUDA of each program. Check that the line
   carries `--partition=cryo5000 --reservation=tomo --gres=gpu:a5000:1`: the
   course's cards are reserved for `tomo`. You run it; it reads what it
   printed.

It never installs anything itself and never edits `~/.bashrc`: an install
writes gigabytes outside the clone, an environment it activated would be gone
at its next command, and `~/.bashrc` is yours. If it ever offers to, say no.

**What the setup should end with.** These are the values that work on cryoem.se,
for checking what the assistant proposes. If it proposes something else, ask it
why.

| What | On cryoem.se |
|---|---|
| `CRYOET_WORK`, `MAMBA_ROOT_PREFIX`, `PIP_CACHE_DIR` | in `~/cryoet`: `~/cryoet`, `~/cryoet/micromamba`, `~/cryoet/cache/pip` |
| micromamba | `module load micromamba/2.5.0`, then its shell hook, `eval "$(micromamba shell hook --shell bash)"` |
| `ENV_MANAGER` | `micromamba`: the jobs activate the pipeline's environment with it |
| `IMOD_DIR` | `/opt/cryoem/imod/5.2.1` |
| `CTFFIND5` | `/opt/cryoem/ctffind/5.0.2/bin/ctffind`; the cluster's CTFFIND 4 is refused |
| `MOTIONCOR3` | `/opt/cryoem/motioncor3/1.1.1/bin/MotionCor3` |
| `ARETOMO` | `/opt/cryoem/aretomo2/1.1.2/bin/AreTomo2`, which reports itself as 1.1.3 |
| `LD_LIBRARY_PATH` | begins with `/opt/tcbsys/cuda/12.6.2/lib64`, CUDA 12's libraries, which MotionCor3 and AreTomo2 need |
| the `cryoet-pipeline` environment | Python 3.13, your clone installed with `pip install -e .`, and `imagecodecs`; its PyTorch a CUDA build |
| the `pytom_tm` environment | Python 3.12, CuPy for CUDA 12 from conda-forge, pytom-match-pick 0.14.0; `PYTOMO_MP_DIR` is its `bin` folder |
| the `membrain` environment | Python 3.9, MemBrain-seg 0.0.10; `MEMBRAIN_EXE` is its `bin/membrain` |
| `MEMBRAIN_CKPT`, `CRYOET_LIBRARY` | from `module load tomobear/2026` (step 1) |

`imagecodecs` lets step 6 read the TIFF frames if they are compressed: tifffile
decodes compressed TIFF only with it. If the assistant leaves it out, step 6
says so, and the assistant gives you the command then.

The setup is done when the check, run in a new terminal and on the GPU node,
shows no problems (`Problems (0): none`). Each problem and each note names an
entry of the playbook, `docs/agent/playbook/`, which the assistant reads and
explains.

## 5. Start again, in the environment

From now on, start the assistant from a terminal where the pipeline's
environment is active: it runs the pipeline's scripts in this shell, and your
jobs take this shell's tool variables.

```bash
cd ~/cryoet/cryoet-pipeline
micromamba activate cryoet-pipeline
scripts/assistant -c
```

`-c` continues the installation's session; without it a new session starts.
For Codex, `scripts/assistant --codex -c`. **Every time you come back**, start
the same way.

## 6. Have it read the frames and look at the raw tilts

The frames are in the course folder, which every course account can read:
`/mnt/cephfs/home/tomobear/data/ryr1`, also `~/tomobear/ryr1` through the link
your home has to the course's data folder. Do not copy them: every group would
hold its own copy of the same frames, and the pipeline only reads them.

> The RyR1 frames are in /mnt/cephfs/home/tomobear/data/ryr1: SerialEM TIFF,
> no mdocs, gain.mrc beside them. Read the folder, then look at the raw tilts
> of 039: are there gold beads, and which way does the tilt axis run?

What it does, following its `lane-config` skill:

1. **Reads the folder** with `scripts/show_folder.py`, writes the file-name
   pattern, and shows you a table: series, tilts, angle range, frames per tilt,
   pixel size. The series, the tilt number and the angle are in each file name,
   `tomogram_039_001_-42.0.tif`, since there are no mdocs. Check the table
   against what was collected: five series (039, 096, 097, 125, 161) of 29
   tilts, −42° to +42°, 10 frames, 0.84 Å/px. Say so, or say what is wrong.
2. **Draws the raw tilts** of 039 with `scripts/show_tilts.py`, which writes
   only a PDF, `outputs/tilt-previews/ryr1_039.pdf`. It reads nine tilts: the
   one nearest 0°, the lowest, the highest, and the four highest on each side,
   and adds up each tilt's frames without gain or motion correction. That is
   about nine gigabytes to read, so it takes some minutes.
3. **Tells you what it printed**: the dark round spots and their size, and the
   tilt axis measured from the tilts. It reads these numbers, not the
   pictures: the PDF is for you to look at.

If `show_tilts.py` stops saying that tifffile needs the imagecodecs package,
the assistant gives you the install command for the pipeline's environment; run
it in your own terminal, then ask it to try again.

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
are the darkest, roundest spots, the same size everywhere. Do the spots look
like beads, and is the size close to 10 nm, the usual gold? With beads, the
lane aligns on them; without, it would align by patch tracking or by AreTomo2.

**The tilt axis.** The tilts on either side of 0° are compared in pairs, the
more tilted one stretched across each candidate axis, and the best axis is the
one where they match. The PDF gives it as a line on the image and as an angle
from the Y axis, the way IMOD counts it, in two forms, such as 84° or −96°. The
two are the same line: the measurement has no sign, and the two signs give
tomograms of opposite hand. At MDC, IMOD's tiltalign solved 84.0 on four of
these series, starting from an axis read by eye; on 161, AreTomo2 refined it to
83.9 on cryoem.se. So 039 should read about 84, give or take a degree or two.
If it reads about 96 (or −84), the preview has read the TIFF rows in the other
order from MotionCor3: keep 84, which was solved on MotionCor3's sums, and tell
Anna.

> **Status, 24 November 2026.** `show_tilts.py` is new. On MRC stacks of two
> EMPIAR datasets it found the axis within one degree of their mdocs, and 62
> gold beads of 10 nm on one of them. It has not read TIFF frames on the
> cluster yet: this step is its first run on them. Tell Anna what it read.

To look at another series, ask for it; each is another nine gigabytes to read.

## 7. Create the project, with all five series

The project is the folder that holds everything the pipeline writes about these
data, `outputs/ryr1-tutorial`, in your clone. This step is yours, not the
assistant's. In a terminal of your own, in the clone with the environment
active:

```bash
cryoet init --frames /mnt/cephfs/home/tomobear/data/ryr1 \
    --out outputs/ryr1-tutorial --frames-pattern \
    'tomogram_(?P<series>\d+)_(?P<tilt>\d+)_(?P<angle>[-+]?\d+(?:\.\d+)?)\.tif'
```

It prints each series with its tilt count, angle range, frames per tilt and
pixel size: the numbers of step 6. A project is never created a second time:
`cryoet init` refuses a folder that holds one, since it would start its
registry empty and lose everything in it. A series left out can join later,
with the `cryoet init … --add-series` command a dry run prints, which keeps the
rest of the project as it is.

## 8. Have it write the lane

A *lane* is one TOML file that says how a dataset is processed: where the data
are, every setting of every step, and how the fixed chain of steps is cut into
Slurm jobs. Ask for one:

> I created the project outputs/ryr1-tutorial with cryoet init, all five
> series. I want a bin8 lane up to the MRA for all five, with this course's
> Slurm options. I want to look at the picks before the MRA.
> Ask me what you cannot read.

What it does, following its `lane-config` skill:

1. **Reads the closest lane at this site.** That is likely
   `docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml`, made for these frames
   on cryoem.se, or `configs/lanes/ryr1-bin8.toml`, which ran on them at MDC:
   their comments say where each value came from.
2. **Asks what is left**, in one message, each question with its default. Your
   answers for these data:

   | It may ask | Answer |
   |---|---|
   | what to pick, from which map | RyR1, EMD-54612 |
   | its widest diameter | 340 Å, read off the map; the map is C4 about z, so `z_axis_symmetry = 4` for picking (only shortens the search) |
   | gold beads, size | yes, 10 nm (step 6) |
   | membrane, how far | yes, RyR1 spans the SR membrane; search within ~17.5 nm of it |
   | gain-corrected? | no: `gain.mrc` beside the frames, rotation 0, flip 1 |
   | Cs | 2.7 mm; most likely 300 kV and 2-7 µm nominal defocus too (EMD-54612's record, not confirmed for these files) |
   | tilt-axis angle | 84.0°, as step 6 measured; its sign is not known |
   | one series first, or all | all five |
   | where the outputs go | `outputs/ryr1-tutorial`, which exists with all five series |
   | MRA right after picking, or look first | look first |
   | particles per tomogram | not known: keep the default, twice pytom's recommendation |
   | partitions, GPUs, memory | GPU stages and the MRA: `--partition=cryo5000 --reservation=tomo --gres=gpu:a5000:1 --cpus-per-task=8 --mem=40G`; CPU stages: `--partition=cs-cpu --reservation=tomo --cpus-per-task=8 --mem=31G` |

   The partitions are what the course was given: every job runs in the
   reservation `tomo`; the GPU nodes are two nodes of eight RTX A5000 cards (24
   GB each), shared by twelve groups, so one card and at most 40 GB a job; jobs
   on `cs-cpu` must ask for 8 CPUs and 31 GB. The MRA too runs on one card.

   One thing it will not ask, and these data need: the CTF search window. Its
   rule is the code's default, ±1000 Å, which fits only the views near zero
   tilt on these data, because the defocus moves with the tilt. Tell it:
   `[ctf] defocus_search_window_angstrom = 10000`, as in the tutorial's lane,
   and a first-fit defocus range of 1-8 µm. The frames' super-resolution
   binning, `[motion] ftbin = 2`, it sets itself from the frame size.
3. **Writes the lane** into `configs/lanes/`, with the source of every value in
   a comment (you, a header, a rule, or a guess, labelled as one). The client
   shows you the file and asks before writing it. Read it, then approve it
   once; in Codex the question is *Would you like to make the following
   edits?*, and the answer **Yes, proceed**.
4. **Checks it** with `run_series.py --list-steps`, which refuses any key the
   pipeline does not read.

Compare what it wrote with the tutorial's own lane,
[`ryr1/ryr1-tutorial-stockholm.toml`](ryr1/ryr1-tutorial-stockholm.toml), whose
time limits are 3 hours for motion correction, 12 for the alignment, 8 for the
CTF and the tomogram, a day for the `gpu` stage and two for the MRA.
Differences are not errors by themselves: ask it to explain any you do not
understand.

## 9. Dry run, yes, submission

It then runs the dry run and shows you what it printed: four jobs (`motion`,
`align`, `cpu`, `gpu`), each waiting for the one before, the series, what Slurm
said of each stage when asked with `sbatch --test-only` (when and where it
would start, each stage judged alone), and any `would refuse:` line with what it
means. A stage Slurm refuses comes with Slurm's own reason, for example a
reservation or partition your account may not use. It submits with
`--until-stage` set to the stage that ends at particle extraction, because you
said you would look at the picks first.

It asks whether to submit. Answer with a clear yes. It then runs the same
command without `--dry-run` and with `--reason`, and reports the job IDs. If
anything changed between the dry run and your yes, it has to run the dry run
again and ask again; the script refuses otherwise. Every dry run and every
submission is written to the project's journal,
`outputs/ryr1-tutorial/journal.jsonl`.

In Codex there is one more question. A submission runs outside Codex's
sandbox, so Codex shows the command and asks *Would you like to run the
following command?* Check that it is the dry run's command without
`--dry-run`, with `--reason`, and answer **Yes, proceed**. Never the answer
that says *don't ask again*: it would let every command that starts the same
way run without asking, for the rest of the session.

## 10. While the jobs run

Ask it how things stand, as often as you like:

> How is ryr1-tutorial doing?

It runs `scripts/show_project.py` (its `project-status` skill) and summarises
which series are done, which are running, which failed and where, and the QC
warnings. It can also answer what a QC warning means, citing the file and the
line it read. The `motion` job's log warns that the node has no `cuobjdump`, so
MotionCor3's GPU code goes unchecked: on cryoem.se that is expected, and not a
failure.

**If a job fails**, ask why:

> The cpu job of ryr1-tutorial failed. Why, and what do we do?

It reads the failure with `scripts/show_failure.py`, checks the playbook
(`docs/agent/playbook/`) for a problem met before, and proposes a fix from the
recipes known to be safe, usually a resubmission from the failed stage with
`--overwrite`. That needs its dry run, then your yes to the submission and to
`--overwrite`. It will not rerun a failure that reproduces without changing
something first.

## 11. Look at the picks

When the `gpu` job has finished, ask it for the project's report:

> Write the report for ryr1-tutorial.

`cryoet report` is not on its list of commands that run without asking, so
the client asks you before running it (Codex, to run it outside its sandbox);
it only reads the project and writes a PDF,
`outputs/ryr1-tutorial/reports/report-bin8.pdf`. Open it in VS Code, or copy it
to your laptop as in step 6, with `CLONE` named there:

```bash
scp "$CLONE/outputs/ryr1-tutorial/reports/report-bin8.pdf" .
```

Read it yourself:

- **the summary table**: each step's verdict per series. A `warning` does not
  stop the chain, so read each one;
- **tilt alignment**: the residual and the beads per tilt, and the tilt axis the
  alignment solved. Is it the axis step 6 measured from the raw tilts?
- **CTF**: which views were fitted. On these data a few of the highest tilts
  may find no rings. If a series has five such views or fewer, they are left
  out of the tomogram; with more, they stay in with an interpolated defocus,
  and the QC warns;
- **the tomogram and its membranes**: is the membrane segmentation on the
  membranes?
- **the picks**: on the tomogram, their scores against noise, and the galleries
  of picks by score, each turned to the template's orientation. Good picks look
  like the template; the low-score end is where the false ones are.

Ask the assistant about anything in it. What it says about a picture is an
impression, and it says so.

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

## 12. A branch: 161 aligned by AreTomo2 (optional)

> **Status, 24 November 2026.** The branch ran once on cryoem.se, on 24
> November 2026, in a test project up to the tomogram: AreTomo2 aligned 161 in
> a minute and a half on an A5000, and each lane's status and report showed its
> own results.

A lane can part from another earlier than at its own tomogram. Here 161 is
aligned a second way, by AreTomo2's projection matching instead of the gold
beads, in the same project. The branch shares the main lane's motion
correction and coarse alignment of 161; it has its own fine alignment and CTF
fit, labelled with `[lane] alignment_variant`, and its own aligned stack,
tomogram and picks, under its own `[lane] variant`. Nothing of the main lane's
is replaced. AreTomo2 is found through `$ARETOMO`, which the installation set.
Ask:

> Align 161 a second way, with AreTomo2, beside the bead alignment, in the same
> project, up to the tomogram. Then write its report.

It writes a copy of the lane that aligns on its own and starts at the fine
alignment, without `[fiducials]` and with a `[reconstruct] thickness`, since
AreTomo2 measures no sample depth; it shows it, dry-runs it and asks. The
tutorial's version is
[`ryr1/ryr1-tutorial-stockholm-161-aretomo.toml`](ryr1/ryr1-tutorial-stockholm-161-aretomo.toml),
whose comments say how each value was chosen. The branch has to wait until the
main lane's jobs are done with 161, and the MRA until the branch's are.

Put the branch report's pages for 161 beside the main report's:

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

## 13. The MRA

The MRA (multi-reference alignment) classifies the particles of all the series
together: it aligns every particle to a few references and sorts them into
classes. When you are satisfied with the picks:

> Run the MRA on the particles of the five series.

It submits the lane's `mra` stage with `--from-stage mra`. The dry run shows
the run's folder and every setting with its source; with no `[mra]` section
they are the lab's rules: three references (two averages of 300 particles, one
of noise, which collects the particles that resemble nothing), five rounds from
coarse to fine, the 340 Å diameter from `[pick]`, the mask from it. It refuses
while a series has no particles, or while one of your jobs still works on the
series.

The job runs on one A5000, for hours. At MDC, on four A40s and four of the
series, its first two iterations took 40 and 83 minutes and the last three
about 2 minutes together; one A5000 should take several times as long, which
has not been measured. If it stops on its time limit, the same submission
carries the run on from its last finished iteration.

## 14. Read the classification report

The job writes `outputs/ryr1-tutorial/mra/<run name>_classification_report.pdf`,
by the lab's rules `bin8-ssn_classification_report.pdf`. Its pages, in order:

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

Then, for instance:

> Class 2 looks like RyR1. Run only its particles again.

It writes a copy of the lane whose `[mra]` names the run and the class
(`from_run`, `from_classes`), with any settings you change, and submits it with
`--from-stage mra` after the dry run and your yes.

## 15. A class to RELION 5 (optional)

The MRA sorts the particles; it does not give a resolution. For a gold-standard
refinement, a class that looks like RyR1 can go to RELION 5:

> Export class 2 of bin8-ssn to RELION.

> **Status, 24 November 2026.** The export is new. A RyR1 class was exported
> at MDC on 10 November 2026; RELION's Extract and Refine3D have not run on its
> exports yet, at MDC or on cryoem.se. Tell Anna what happens.

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

After your yes it exports, to `outputs/ryr1-tutorial/relion/bin8-ssn-class2/`:

- `relion_commands.sh`: an Extract, then a gold-standard Refine3D with local
  searches from the MRA's poses, without symmetry, both at the MRA's pixel
  (13.44 Å here). This first refinement checks the export rather than reaching
  ~20 Å; finer pixels come after, by extracting again at a lower `--bin`;
- `README.md`, which says why each value is what it is.

It then gives you the `sbatch` line the export printed. RELION is yours to run:
the assistant never runs RELION, `relion_commands.sh` or `mpirun`, and cannot
call `sbatch`. `relion_commands.sh` takes RELION from your `PATH`, from the
modules `RELION5_MODULE` names, and from an installation `RELION5_HOME` names.
Which module gives RELION 5 on cryoem.se is the organizers' to say. In your own
terminal (not with `!`), set it for the job, which takes it from your shell:

```bash
export RELION5_MODULE=<the RELION 5 module>
```

Then run that line, with the course's GPU options in place of
`<GPU partition and --gres>`: one A5000, which the two Refine3D workers share,
and the course's 40 GB.

```bash
sbatch --nodes=1 --ntasks=3 --cpus-per-task=8 --time=1-00:00:00 \
    --partition=cryo5000 --reservation=tomo --gres=gpu:a5000:1 --mem=40G \
    --chdir=<the folder it gave you> --wrap "./relion_commands.sh"
```

`relion_commands.sh` first checks that RELION starts, and stops, naming what is
missing, if it does not. The job's log, `slurm-<job id>.out`, is written in the
export's folder; the assistant can read it for you.

## Good to know

- **It submits only through `submit_lanes.py` and `submit_recipe.py`.** It
  cannot call `sbatch` or `srun`, and it will not write a batch script for an
  experiment the recipes do not cover: it stops and asks a person instead.
- **Do not get round its rules.** In particular, do not run an install, a
  submission, `scancel` or `--overwrite` with the `!` prefix, which both
  clients have: in Claude Code `!` runs a command outside the rules and the
  guard, and in Codex that has not been checked. Run such a command in your own
  terminal, or through the assistant with its dry run and your yes. In Codex,
  do not answer a question with *Yes, and don't ask again …*: the launcher
  removes that answer at the next start, but until then it lets every command
  that starts the same way run without asking.
- **It does not change the pipeline.** If it suspects a bug, it says so with its
  evidence, for a person to fix.
- **Every submission and every export it makes is in the journal**,
  `outputs/ryr1-tutorial/journal.jsonl`, with the client, the model and your
  reason; a submission also with the lane file's text.
- **When something refuses**, it shows you the refusal and explains it. The
  ones met most often:

  | It says | What it means |
  |---|---|
  | `$MOTIONCOR3 is not set` (or another `$…`) | the session's shell lacks the tool variables: the `.bashrc` lines of step 4, a new terminal, and `scripts/assistant -c` |
  | a stage refused by Slurm, with Slurm's reason | the stage's `sbatch` options do not fit what your account may use: compare them with step 8, and ask the organizers if the reservation has ended |
  | `… already exists but has no manifest for …` | the project lacks that series: run the `cryoet init … --add-series` it prints, yourself |
  | a series is busy, naming a job | one of your jobs works on that series; wait until it has ended |
  | `the MRA reads the particles extracted under 'bin8', and cannot` | a series has no particles yet: wait for its `gpu` job, or ask why it failed |
  | CTFFIND 4 refused | `$CTFFIND5` is not set: the `.bashrc` lines of step 4 |
