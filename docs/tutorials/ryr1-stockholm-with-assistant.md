# RyR1 from frames to MRA classes on cryoem.se, with the assistant

This tutorial takes you from a course account on cryoem.se, the cluster of the
Stockholm workshop, to class averages, with the pipeline's terminal assistant.
Your account already has the pipeline installed; you install the assistant's
client. From there the assistant works beside you: it checks the installation,
reads the frames, draws the raw tilts into a PDF that shows the gold beads and
the tilt axis, writes the lane file, checks it and submits it. You run what it
hands you, answer its questions, read what it shows you, and say yes or no. The
data are RyR1 in native membranes, five tilt-series of SerialEM TIFF frames
with gold beads.

```text
start a session + client ─▶ the assistant checks the installation
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

> **Status, 26 November 2026.** On cryoem.se the assistant ran in Claude Code
> under a course account from 19 to 25 November 2026: it took that account
> through the installation on 22–23 November, with fixes to the pipeline in
> between, and wrote a lane for these data. The course's lane then ran through
> the MRA on all five series on 25–26 November: motion correction, alignment
> and the tomograms in 30 minutes, picking and particles in 1 hour 28 minutes,
> and the MRA in 8 hours on one A5000. Picking had first failed at its first
> angle: pytom-match-pick's CuPy cannot compile one of its kernels there. The
> assistant found it in the playbook, entry I-9, and narrowed it down with
> tests on a GPU node; `session.sh` now turns that part of CuPy off, and the
> run above picked with it off. The branch of step 12 ran on 161 on 24
> November. That run's results are the course's results of step 10. A course
> account was installed on 26 November as yours was, in about ten minutes, and
> its check on an A5000 found no problem; `session.sh` ran there. Not run there
> yet: Codex, going on from a copy of the course's results (a copy was made on
> one account on 26 November, and nothing has been submitted from it), and the
> tilt preview of step 6 on these TIFF frames. At MDC the same lane ran
> through the MRA on four of the series on 19–20 November, and through fine
> alignment on all five on 23 November. Tell Anna what happened.

## What you need

- **A course account on cryoem.se** and an SSH connection to
  `login2.cryoem.se`: a terminal, or VS Code with Remote SSH. You, and the
  assistant, run everything there. The heavy work runs in Slurm jobs, which the
  pipeline's scripts submit.
- **Room in your home folder**, which has no quota on course accounts. The
  installation there takes about 27 GB (one course account, 24 November 2026).
  The outputs at bin8 add the motion-corrected sums and stacks
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
  from the cluster; and ChimeraX with ArtiaX, installed as the course's
  [setup guide](https://cryoem-sweden.github.io/sta-workshop/2612/setup-guide.html) says, or IMOD's
  `3dmod`, to see the picks in 3D.
- **Time for the client's installation and sign-in**, then for the queue. While
  the jobs run, the session can be closed.

## 1. Your account, and how every session starts

Your account has the pipeline installed, in one folder of your home, `~/cryoet`:
the environments, the downloads, and the clone, `~/cryoet/cryoet-pipeline`,
which the outputs go into. The programs are installed once for the whole
cluster, under `/opt/cryoem`. The course's module, `tomobear/2026`, names the
two files the course shares: MemBrain-seg's model, which segments the
membranes, and the assistant's reference library, the manuals of IMOD,
pytom-match-pick and the other tools at the versions installed here.

None of it is in `~/.bashrc`: the same account serves another part of the
workshop, which should see none of it. So **every new terminal** starts with
two lines:

```bash
module load tomobear/2026
source ~/cryoet/session.sh
```

Loading the module prints a warning that `MAMBA_ROOT_PREFIX` is not set. It
comes from the micromamba module that the course's module loads, and is
expected: `session.sh`, the second line, sets it.

`session.sh` is [`docs/tutorials/ryr1/stockholm-session.sh`](ryr1/stockholm-session.sh),
copied there by the installation. It names the cluster's programs and the
environments in `~/cryoet`, goes to the clone's root, activates the pipeline's
environment, and says so: `pipeline session: …/cryoet/cryoet-pipeline,
environment cryoet-pipeline`. It refuses to run before the module. Check that
the module set its two files:

```bash
echo "$MEMBRAIN_CKPT" "$CRYOET_LIBRARY"
```

It should print `/mnt/cephfs/home/tomobear/models/MemBrain_seg_v10_alpha.ckpt`
and `/mnt/cephfs/home/tomobear/cryoet-library`. The library has to be there
before the assistant starts: the launcher hands it to the session at the start.
The assistant runs the pipeline's scripts in the shell it was started from, and
your jobs take that shell's variables: start it only from a terminal where both
lines ran.

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

If Codex is the only client you will use, this line makes `scripts/assistant`
start Codex without `--codex`. Type it after the two lines of step 1, in each
new terminal, since `~/.bashrc` stays as it is:

```bash
export CRYOET_ASSISTANT_BACKEND=codex
```

From now on, start Codex for the pipeline only through the launcher: it is
what gives it the pipeline's rules.

## 3. Start the assistant

In a terminal where the two lines of step 1 ran, which leaves you at the
clone's root with the pipeline's environment active:

```bash
scripts/assistant
```

For Codex, `scripts/assistant --codex`.

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
  last session: after a lost connection, or the next day, in a new terminal
  after the two lines of step 1 (step 5). The jobs do not depend on the
  session: they run whether it is open or not.
- The model. In Claude Code the default is `claude-opus-5-5`, and
  `scripts/assistant --model claude-sonnet-5-5` starts a cheaper one. In Codex
  it is `gpt-6.1-sol`, reasoning at `high`; `scripts/assistant --codex --effort
  medium` makes it reason less. Give a full model ID, and to change the model
  start a new session rather than using `/model`, so that the journal names the
  right one.
- Write to it in English or in Russian; it answers in the language you use.

## 4. Have it check the installation

Ask in your own words, with what it cannot know:

> I am on cryoem.se for the Stockholm course. The pipeline is installed in
> ~/cryoet, my clone is ~/cryoet/cryoet-pipeline, and this shell has the module
> tomobear/2026 and ~/cryoet/session.sh loaded. I want to run the lane
> docs/tutorials/ryr1/ryr1-tutorial-stockholm.toml. Check the installation.

It checks with `scripts/show_setup.py`, following its `install` skill. The
check writes nothing. It shows the clone, the room where the environments,
downloads and temporary files go, the pipeline's environment, every tool the
lane needs with its version, and which of the cluster's GPUs each program can
run on. Your home has no quota; if the check says it cannot read one, that is
right. It should show no problems (`Problems (0): none`). Notes that IMOD
5.2.1 or MotionCor3 1.1.2 has not run in the pipeline are expected: the check
knows one tested version of each, and both ran in the course's run of 25
November. Each problem and each
note names an entry of the playbook, `docs/agent/playbook/`, which the
assistant reads and explains.

The installation ran the same check on your account in a short job on one of
the course's A5000 cards, where it also compares the driver with the CUDA of
each program. If the assistant proposes to run that part again, it gives you an
`srun` line: check that it carries `--partition=cryo5000 --reservation=tomo
--gres=gpu:a5000:1`, since the course's cards are reserved for `tomo`. You run
it; it reads what it printed. On the card the check also compiles one of
pytom-match-pick's kernels, with CuPy's CUB and without: a note that the kernel
fails with CUB and that the shell turns CUB off is expected on this cluster,
where `session.sh` turns it off (playbook entry I-9).

It never installs anything itself and never edits `~/.bashrc`. If it finds
something missing, it gives you the command, and you run it in your own
terminal; on these accounts nothing should be missing, so tell Anna too. If it
proposes lines for `~/.bashrc`, say no: on this account the two lines of step 1
do that job, and a tool it calls missing most often means they were not run in
the terminal the assistant was started from.

**What your account has.** These are the installation's values on cryoem.se,
for checking what the assistant reports, and for doing the same on another
cluster.

| What | On cryoem.se |
|---|---|
| `CRYOET_WORK`, `MAMBA_ROOT_PREFIX`, `PIP_CACHE_DIR` | in `~/cryoet`: `~/cryoet`, `~/cryoet/micromamba`, `~/cryoet/cache/pip` |
| micromamba | `module load micromamba/2.5.0`, then its shell hook, `eval "$(micromamba shell hook --shell bash)"` |
| `ENV_MANAGER` | `micromamba`: the jobs activate the pipeline's environment with it |
| `IMOD_DIR` | `/opt/cryoem/imod/5.2.1` |
| `CTFFIND5` | `/opt/cryoem/ctffind/5.0.2/bin/ctffind`; the cluster's CTFFIND 4 is refused |
| `MOTIONCOR3` | `/opt/cryoem/motioncor3/1.1.1/bin/MotionCor3`, which reports itself as 1.1.2 |
| `ARETOMO` | `/opt/cryoem/aretomo2/1.1.2/bin/AreTomo2`, which reports itself as 1.1.3 |
| `LD_LIBRARY_PATH` | begins with `/opt/tcbsys/cuda/12.6.2/lib64`, CUDA 12's libraries, which MotionCor3 and AreTomo2 need |
| the `cryoet-pipeline` environment | Python 3.13, your clone installed with `pip install -e .`, and `imagecodecs`; its PyTorch a CUDA build |
| the `pytom_tm` environment | Python 3.12, CuPy for CUDA 12 from conda-forge, pytom-match-pick 0.14.0; `PYTOMO_MP_DIR` is its `bin` folder |
| `CUPY_ACCELERATORS` | set to nothing: CuPy's CUB off, which pytom-match-pick's CuPy cannot compile here (playbook entry I-9) |
| the `membrain` environment | Python 3.9, MemBrain-seg 0.0.10; `MEMBRAIN_EXE` is its `bin/membrain` |
| `MEMBRAIN_CKPT`, `CRYOET_LIBRARY` | from `module load tomobear/2026`; everything above, from `session.sh` |

`imagecodecs` lets step 6 read the TIFF frames if they are compressed: tifffile
decodes compressed TIFF only with it. If it is missing, step 6 says so, and the
assistant gives you the command then.

## 5. Every time you come back

Start the same way: a new terminal, the two lines of step 1, and the assistant,
continuing your last session with `-c`:

```bash
module load tomobear/2026
source ~/cryoet/session.sh
scripts/assistant -c
```

For Codex, the last line is `scripts/assistant --codex -c`. Without `-c` a new
session starts.

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

**If the cause cannot be fixed in the course's time**, go on from the course's
results. The same lane has run through every stage in the course folder, in
`/mnt/cephfs/home/tomobear/cryoet-pipeline/outputs/ryr1-tutorial`. Copied into
your clone at the same place, it is a project like your own: its records name
its files from the clone's root, and it reads the frames from the course folder
as yours does. Make the copy yourself, in your own terminal: set your project
aside, so that nothing of it is lost, and copy the course's in its place (some
40 GB, a few minutes):

```bash
mv outputs/ryr1-tutorial outputs/ryr1-tutorial.mine
mkdir -p outputs
cp -r /mnt/cephfs/home/tomobear/cryoet-pipeline/outputs/ryr1-tutorial outputs/
```

Without a project of your own, skip the `mv`, but not `mkdir -p outputs`: in a
clone without an `outputs` folder, `cp` would make `outputs` itself the copy,
and the pipeline would not find the project (one course account, 26 November).

It holds the results of every stage, the MRA's too, so you can read them right
away, from step 11 on. To run the rest yourself, tell the assistant which stage
failed and that the project is now the course's copy: it submits from the stage
after, with its dry run and your yes, and with `--overwrite`, since that stage's
results and those after it are replaced by yours.

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
conflict, ArtiaX issue #56). Their fix, in the course's
[setup guide](https://cryoem-sweden.github.io/sta-workshop/2612/setup-guide.html): after
`toolshed install ArtiaX`, run `pip install pandas==3.0.5` in ChimeraX's
command line. 3dmod shows the same picks. Open one or two series: do the picks
sit on density at the membranes?

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
is replaced. AreTomo2 is found through `$ARETOMO`, which `session.sh` of step 1
sets.
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
about 2 minutes together. On one A5000 here, the course's run of the five
series took 8 hours, on 25–26 November. If it stops on its time limit, the
same submission
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

> **Status, 26 November 2026.** The export is new. On cryoem.se on 26 November
> 2026, with the module `relion/5.1`, RELION's Extract ran on a class of this
> lane and Refine3D stopped at its first iteration ("subtomo multiplicity
> correction only applies to ctf_premultiplied data"): RELION 5.1.1 refuses its
> own 3D pseudo-subtomograms. The export now has Extract write 2D stacks, which
> has not run yet. An export whose `relion_commands.sh` lacks `--stack2d` has
> to be written again. Tell Anna what happens.

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

**From a copy of the course's results** (step 10), tell it so: the export
needs `--project outputs/ryr1-tutorial`. The MRA run names the folder it ran
in, the course's, where you cannot write, and the export writes into that
project unless told otherwise.

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
On cryoem.se the module is `relion/5.1`. In your own terminal (not with `!`),
set it for the job, which takes it from your shell:

```bash
export RELION5_MODULE=relion/5.1
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
  | `$MOTIONCOR3 is not set` (or another `$…`) | the session's shell lacks the tool variables: leave the session, run the two lines of step 1, and `scripts/assistant -c` |
  | `session.sh: load the course module first` | `module load tomobear/2026` comes before `source ~/cryoet/session.sh` |
  | `Cannot activate, prefix does not exist`, naming `~/.local/share/mamba`, or `Shell not initialized` | `micromamba activate` ran in a terminal without `session.sh`, which says where the environments are: run the two lines of step 1 instead; `session.sh` activates the environment itself |
  | a stage refused by Slurm, with Slurm's reason | the stage's `sbatch` options do not fit what your account may use: compare them with step 8, and ask the organizers if the reservation has ended |
  | `… already exists but has no manifest for …` | the project lacks that series: run the `cryoet init … --add-series` it prints, yourself |
  | a series is busy, naming a job | one of your jobs works on that series; wait until it has ended |
  | `the MRA reads the particles extracted under 'bin8', and cannot` | a series has no particles yet: wait for its `gpu` job, or ask why it failed |
  | CTFFIND 4 refused | `$CTFFIND5` is not set: the two lines of step 1, as above |
  | a stage of yours failed and cannot be fixed in time | go on from the course's results: step 10 |
