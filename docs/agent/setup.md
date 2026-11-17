# Installing the pipeline and its tools

What an installation needs: where each part goes and whether there is room,
which versions fit together, and how a GPU, its driver and CUDA decide what can
run. The assistant's `install` skill works from this page; a person can read it
on its own. The problems met so far are in
[playbook/install.md](playbook/install.md) (the I- entries), with E-1, E-9 and
E-10 in [playbook/cluster.md](playbook/cluster.md).

One command checks all of it on the host it runs on. It writes nothing, and it
runs before anything is installed, on the system's Python (3.9 or later):

```bash
python3 scripts/show_setup.py
```

| Option | What it adds |
|---|---|
| `--env <name or folder>` | ask that environment instead of the one named `cryoet-pipeline` |
| `--config <lane config>` | find the tools as that lane names them, and need what it needs (Python 3.11 or later) |
| `--folder <path>` | show the room of another folder, such as where the data will go |
| `--sizes` | also measure what each place holds, with `du` (up to a minute each) |
| `--find` | also look for each tool where this shell does not name it: other environments, environment modules, software folders (below, "Tools that are installed but not found") |
| `--search <folder>` | another folder for `--find`, such as a group's software folder |
| `--setup-file` | print a proposed `pipeline_setup.sh` instead of the report |
| `--json` | the same report as JSON |

Its parts: the clone and whether its remote has newer commits; 1. room; 2. the
pipeline's environment; 3. the tools; 4. the GPU and CUDA; 5. what can run here.
Then the problems and the notes, each with its playbook entry.

## What gets installed, and where

| Part | What it is | Where it goes | Who installs it |
|---|---|---|---|
| the clone | this repository; the lanes write their outputs inside it | a folder of yours on a large filesystem (`$CRYOET_WORK` at MDC) | you, with `git clone` |
| the `cryoet-pipeline` environment | Python and the pipeline with its extras: `io`, `gpu` (PyTorch with its CUDA libraries), `viz` (the reports), `library` (the reference library) | the environments' folder, `$MAMBA_ROOT_PREFIX/envs` with micromamba | you |
| the `pytom_tm` environment | Python 3.12, CuPy for CUDA 12 from conda-forge, and pytom-match-pick 0.14.0 | beside it | you |
| IMOD, CTFFIND 5, MotionCor3, AreTomo2, MemBrain-seg | the programs the steps run | a shared install: at MDC the group's software folder | the site, once |
| the reference library | the tools' manuals at the versions the site installed, and NVIDIA's, CuPy's and PyTorch's pages for its CUDA versions, as read-only snapshots of some 20 MB (19 MB on the Mac, 17 November 2026) | a folder the group can read: at MDC `/fast/AG_Kudryashev/cryoet-library` | one person of the group, once per site, and again when the tools change |
| Claude Code or Codex | the assistant's client | your home (`~/.local/bin` and the clients' own folders) | you |

The lanes name each tool by a variable: `IMOD_DIR`, `CTFFIND5`, `MOTIONCOR3`,
`ARETOMO`, `PYTOMO_MP_DIR` (the bin folder of the `pytom_tm` environment; note
the O), `MEMBRAIN_EXE` and `MEMBRAIN_CKPT`. A job takes them from the shell it
is submitted from and reads no `~/.bashrc` of its own (E-9). At MDC,
`/fast/AG_Kudryashev/software/cryoet-tools.sh` names the group's tools: IMOD,
MotionCor3, CTFFIND 5 and MemBrain-seg.

Every `cryoet` command imports PyTorch, so the `gpu` extra is not optional: the
CLI imports the averaging backend, which imports PyTorch (I-1).

## Order of installation

The commands are in the README, "0. One-time setup" (at MDC its list "At MDC
(max)"), and in
[the tutorial without the assistant](../tutorials/ryr1-without-assistant.md),
step 1. The clone comes first there. Whoever reads this through the assistant
has one already: the assistant runs from it. In order:

1. **Keep the clone current.** The check's line about the clone says whether its
   remote branch has commits the clone lacks; it reads that with
   `git ls-remote` and fetches nothing. If it does, run `git pull` in the clone,
   unless it was cloned minutes ago. After a pull that changes `pyproject.toml`,
   run the pip install of step 3 again.
2. **Room.** Choose one folder of yours on a large filesystem, and point the
   environments, the downloads and the temporary files at it in `~/.bashrc`
   (below, "Room"). Open a new terminal, and check before installing anything.
3. **The pipeline's environment**, with `pip install -e ".[io,viz,gpu,library]"`
   from the clone. `--resume-retries 10` lets pip resume a broken download
   itself (I-2).
4. **The `pytom_tm` environment**: CuPy from conda-forge first, then
   pytom-match-pick with pip (I-5).
5. **The tools.** Many are installed already somewhere on a cluster: look with
   `--find` before installing one (below, "Tools that are installed but not
   found"). Name each by its variable, in `~/.bashrc` or in a setup file it
   sources. At MDC that is the line that sources `cryoet-tools.sh`, and
   `PYTOMO_MP_DIR`.
6. **Check**, in a new terminal on a login node: `python3 scripts/show_setup.py`
   shows no problems. Then on a GPU node of the kind the lanes use (below, "GPU
   and CUDA"). Then the lane's dry run, which checks the lane's keys, folders,
   tool variables and named programs.
7. **The site's reference library**, once per site rather than per person
   ([library.md](../library.md)). The assistant reads the tools' manuals at the
   versions installed here from it, and NVIDIA's and PyTorch's pages for the
   site's CUDA versions; without one it answers from this page and the
   playbook. At MDC it is built, and `scripts/assistant` finds it. On another
   site, after step 6, one person of the group builds it on a login node, in
   the pipeline's environment and from a shell that names the tools, since the
   builder finds them as the lanes do:

   ```bash
   python3 scripts/build_library.py --site <site> --dry-run
   python3 scripts/build_library.py --site <site> --root <folder>
   ```

   The folder is one the group can read (library.md, "Where snapshots live"),
   and the host has to reach NVIDIA's, CuPy's and PyTorch's sites and GitHub.
   `--gpu-node-report` adds what `nvidia-smi` printed on a GPU node (library.md,
   "How a snapshot is built"). Each person then names the folder with
   `export CRYOET_LIBRARY=<folder>` in `~/.bashrc` or the setup file: by itself
   `scripts/assistant` looks only in MDC's. Build again when a tool, its version
   or the site's CUDA changes. The new snapshot becomes `current`, and each
   session reads the one that was current when it started.

## What can run where

Part 5 of the check answers "can I run the pipeline here?", step group by step
group. Where Slurm answers (`sinfo`), the lanes' jobs run on its nodes, so it
judges the GPU steps against the GPU types the cluster lists. Where it does not,
it judges them against this machine's own GPUs.

| Steps | Needs | GPU |
|---|---|---|
| motion correction (`prepare-tilt-series`), lanes of frames | MotionCor3 | required: MotionCor3 runs only on an NVIDIA GPU |
| stack import (`import-tilt-stack`), lanes of stacks | the pipeline | none |
| alignment: coarse, fiducials or patches, fine with tiltalign | IMOD | none |
| fine alignment with AreTomo2 | AreTomo2 | required (`submit_lanes.py` refuses a stage without one) |
| CTF: `build-ctf-stack`, `estimate-ctf`, `correct-ctf` | IMOD, CTFFIND 5 | none |
| tomogram: `build-final-aligned-stack`, `reconstruct-tomogram` | IMOD | none; IMOD's `tilt` takes one with `--device cuda` |
| membranes (`segment-tomogram`) | MemBrain-seg and its model | advised: the preflight takes no GPU below compute capability 7.5, and a run on the CPU has not been tried |
| template (`prepare-template`) | pytom-match-pick | none: its template and mask run on the CPU |
| picking (`pick-particles`) | pytom-match-pick | required: matching takes `--gpu-ids` |
| extraction (`visualize-picks`, `extract-particles`) | the pipeline | none |
| the MRA | the pipeline's PyTorch | optional: it runs on the CPU too, slower |

Read in the code on 16 November 2026, and in pytom-match-pick 0.14.0's own
source: its template and mask scripts work on the CPU, and only the matching
takes GPU ids. Every pytom-match-pick script imports CuPy, so CuPy must be
installed even where no GPU is used.

So a machine with no NVIDIA GPU can run a lane of stacks up to the tomogram,
the template, and from the picks to the MRA (on the CPU), but not motion
correction or picking. Without Slurm, a lane's steps run one after another with
`scripts/run_series.py --execute`, which the person runs; the lanes'
submissions need Slurm.

## Tools that are installed but not found

On a cluster most tools are installed once for everyone, and a shell finds
none of them until its variables name them (E-9). Before installing one, look:

```bash
python3 scripts/show_setup.py --find
```

`--find` looks, for each tool this shell does not name:

- in the other environments of the usual roots, for pytom-match-pick and
  MemBrain-seg;
- in the environment modules (`module -t avail`, then `module show` of each
  module whose name says which tool it holds);
- in `/opt`, `/usr/local`, `~/software`, `~/apps`, `~/opt`, `~/bin`,
  `~/.local/bin`, MDC's group folder `/fast/AG_Kudryashev/software`,
  `$CRYOET_WORK`, and every `--search` folder: five folders deep, and for at
  most 20 seconds each.

It asks each copy found for its version, ranks the ones a lane can use first
(CTFFIND 5 before CTFFIND 4, the version the pipeline has run with first), and
reads MotionCor3's and AreTomo2's GPU code where `cuobjdump` is there.

What it finds goes into the shell as variables, in one of two places:

- **`~/.bashrc`**, a line per tool;
- **a setup file that `~/.bashrc` sources**, which keeps them in one place, as
  MDC's `cryoet-tools.sh` does for the group's tools. The check proposes one,
  and the person writes it to a folder of theirs, reads it, and adds the
  `source` line to `~/.bashrc`:

  ```bash
  python3 scripts/show_setup.py --setup-file > <your folder>/pipeline_setup.sh
  ```

  It sets only what the shell does not set already, so it does not repeat a
  group file: a variable already set appears as a comment.

Either way the lines go after micromamba's own block and outside any check for
the host name, since the jobs take the variables from the shell that submits
them, on a login node. Then open a new terminal and run the check again.

A program found as a module may need its module loaded to run (its libraries),
not only its path: `module show <name>` says what loading it sets.

## Room

### What writes where

| Place | What goes there | Set by |
|---|---|---|
| the temporary folder | pip unpacks each download there, and writes it twice while installing it (MDC, 13 October 2026) | `TMPDIR`, else `/tmp` |
| the environments' folder | each environment, and micromamba's cache of downloaded packages (`pkgs`) | `MAMBA_ROOT_PREFIX` for micromamba; conda's base for conda |
| pip's cache | pip's downloads, kept for the next install | `PIP_CACHE_DIR`, else `~/.cache/pip` |
| home | the clients and their sessions, and everything above that is not moved | |
| the clone | the lanes' outputs | where it is cloned |

Sizes known so far:

- **PyTorch with its CUDA libraries** downloads 18 wheels, 3.1 GB in all; the
  largest, PyTorch itself, is 0.54 GB (PyPI, torch 2.15.0 for Python 3.13 on
  Linux x86_64, read 16 November 2026). Unpacked, it takes more.
- **The clients**: about half a gigabyte for Claude Code and a gigabyte for
  Codex, plus their sessions (measured on the Mac).
- **The outputs** of the RyR1 tutorial at bin8: about 30 GB of motion-corrected
  sums and stacks, plus the tomograms. Measure data with `du -sh` before
  copying it.
- **The two environments installed on max** have not been measured yet;
  `show_setup.py --sizes` measures them on any host.

Three rules follow:

- **The temporary folder needs room for PyTorch's downloads**: pip writes the
  largest, 0.54 GB, twice, so 2 GB free at least. At MDC `/tmp` holds 100 MB, so
  `TMPDIR` must name a folder on `/fast` (E-1).
- **The folder `TMPDIR` names must exist.** Python passes over one that does not
  and uses `/tmp`, without a word (Python's `tempfile.gettempdir`).
- **`MAMBA_ROOT_PREFIX` is set after micromamba's own block in `~/.bashrc`**,
  which sets it to the home (I-4).

At MDC the README's `.bashrc` block sets `CRYOET_WORK`, `MAMBA_ROOT_PREFIX`,
`PIP_CACHE_DIR`, `ENV_MANAGER`, the group's tools and `PYTOMO_MP_DIR`.

### How to look

- **`df -h <folder>`**: what the filesystem that holds the folder has free.
- **`du -sh <folder>`**: what the folder itself holds.
- **The quota**: your share, which can be smaller than what `df` shows. Each
  filesystem has its own tool: `quota -s` (local disks and NFS), `mmlsquota`
  (GPFS, which `/fast` at MDC is), `lfs quota -h -u <user> <mount>` (Lustre).
  `show_setup.py` runs the ones it finds and prints what they say.
- **Which error it is.** `Errno 28`, "No space left on device", is a full
  filesystem, the temporary folder included (E-1). `Errno 122`, "Disk quota
  exceeded", is a full quota (I-3).

## Versions

What the pipeline needs, checked by its code, and what it has run with on real
data. A version not in the table has not run in the pipeline. It may well
work: run one series first, and tell the maintainers what happened.

| Part | The pipeline needs | Has run with |
|---|---|---|
| Python, pipeline's environment | 3.11 or later (`pyproject.toml`) | 3.13 (MDC) |
| PyTorch | 2.2 or later (`pyproject.toml`), a CUDA build that the GPU nodes' driver supports (below) | 2.14.1 for CUDA 13.0 (MDC, October 2026). pip takes the newest: 2.15.0 for CUDA 13.2 on 16 November 2026 |
| Python, `pytom_tm` | | 3.12. A 3.15 environment got no CuPy wheel, and pip tried to build CuPy (I-5) |
| CuPy | from conda-forge with `cuda-version=12.*` (README) | 14.2.0 (MDC) |
| pytom-match-pick | | 0.14.0, the version of the library's manuals |
| IMOD | | 5.1.12 |
| CTFFIND | 5; `estimate-ctf` refuses CTFFIND 4 (C-2) | 5.0.2 |
| MotionCor3 | GPU code for the job's GPU; the job's preflight checks it (M-1) | 1.2.4 (MDC) |
| AreTomo2 | only for `[fine_align] backend = "aretomo"` | 1.1.3 (MDC) |
| MemBrain-seg | a GPU of compute capability 7.5 or later; the preflight checks it (G-1) | 0.0.10 (MDC) |
| Slurm | | 26.05.4 (MDC) |

The versions at MDC were read by `scripts/build_library.py` on 5 November 2026,
and AreTomo2's on 9 November ([library.md](../library.md), "Versions the
pipeline records today"). Every step records the versions it ran with in its
record's `software_versions`.

## GPU and CUDA

### In plain words

- **The driver** is NVIDIA's software on a GPU node. It decides the newest CUDA
  a program may be built for. `nvidia-smi` prints both in its header, as
  "Driver Version" and "CUDA Version", or as "KMD Version" and "CUDA UMD
  Version" (driver 610.57.04 on max's A40 nodes, 17 November 2026). A login
  node usually has no GPU and no driver, so there `nvidia-smi` is missing and
  PyTorch sees no GPU, which is fine (I-11).
- **A program's CUDA version** is the one it was built for:
  `torch.version.cuda` for PyTorch, CuPy's runtime version, and the
  `libcudart.so.<major>` MotionCor3 is linked to.
- **The rule.** A driver runs programs built for its CUDA version or older. A
  program built for a newer minor version of the same major version runs too
  (minor version compatibility), except code compiled at run time. A newer
  major version needs a newer driver: CUDA 13 needs driver 580 or later, CUDA 12
  driver 525 or later (NVIDIA, CUDA Toolkit release notes, read 16 November
  2026).
- **A GPU's compute capability** is its generation as a number: V100 7.0, T4
  7.5, A100 8.0, A40 8.6, H100 and H200 9.0, B300 10.3 (NVIDIA's table of CUDA
  GPUs, read 16 November 2026).
  `nvidia-smi --query-gpu=name,compute_cap --format=csv` prints it.
- **A program's GPU code.** A program carries cubins, each compiled for one
  compute capability (`sm_86` for 8.6), and may carry PTX, which the driver
  compiles when the program starts. A cubin runs on its own major version at an
  equal or higher minor: `sm_80` runs on 8.0 and 8.6, not on 9.0. PTX runs on
  any GPU at least as new as its target. A program with nothing for its GPU
  stops with "no kernel image is available for execution on the device".
- **MIG.** A large GPU can be split into slices. `nvidia-smi --query-gpu`
  then reports the whole card, and `nvidia-smi -L` the slices (P-1).

### In NVIDIA's own words

The reference library ([library.md](../library.md)) holds the pages these
rules come from: for each CUDA version the site has, its release notes, its
`nvcc` manual and its installation guide; NVIDIA's guide to driver
compatibility, the compute capability tables, MIG and installing the driver;
CuPy's installation and environment variables; PyTorch's install command for
each CUDA version. Search them with `python3 scripts/search_library.py`, as
`--tool cuda`, `nvidia`, `cupy` or `pytorch`. The site card lists the CUDA
versions found on the site, and a GPU node's driver when the library was built
with one.

### What runs on a GPU here

| Program | Its GPU code is | Read it with |
|---|---|---|
| MotionCor3 | its own cubins and PTX | `cuobjdump -lelf` and `-lptx` on the program |
| MemBrain-seg | the PyTorch of its environment | the architectures that PyTorch was built for |
| pytom-match-pick | CuPy, which compiles its kernels when they are first used, with NVRTC | CuPy's runtime and NVRTC versions (I-9) |
| the MRA (`mra_port_run.py`) | the pipeline's PyTorch | as for MemBrain-seg |

`show_setup.py` reads all of these, on a login node too, and says which of the
cluster's GPU types (from `sinfo`) each can run on. Whether a program sees a
GPU, and whether the driver suits it, shows only on a GPU node.

### On a GPU node

From a login node, in the shell the jobs are submitted from, with the
environment active, the person runs the check on one GPU of the kind the lanes
use. At MDC:

```bash
srun --partition=gpu --gres=gpu:a40:1 --time=00:15:00 --mem=16G python3 scripts/show_setup.py
```

Elsewhere, take the partition and the GPU type from the lane's GPU stage, or
from `sinfo -o "%P %G"`. `srun` gives the check this shell's variables. A shell
that is itself inside a job (`echo $SLURM_JOB_ID` prints a number) runs `srun`
as a step of that job, with that job's resources only: if the job has a GPU,
run the check there directly.

### Site facts

- **MDC (max).** The GPU node max023 had driver 610.57.04, supporting CUDA 13.3,
  and A40s (13 October 2026). The `gpu` partition had A40, H100, H200 and B300
  nodes in early November 2026 ([playbook/cluster.md](playbook/cluster.md)). A
  MotionCor3 or AreTomo2 built from the published makefiles has no code for the
  B300s (I-14). Which code the group's builds carry has not been read: the check
  reads it on a node with `cuobjdump`.
- **Munich.** The GPU nodes had driver 575, supporting CUDA 12.9 (7 October
  2026), so by the rule above a PyTorch for CUDA 13 does not run there; that
  has not been tried. GPUs there can be MIG slices (P-1).

## The NVIDIA driver

The check lists every NVIDIA card the kernel sees on the PCI bus, with the
driver bound to it, so a card without its driver shows too (I-12):

- **`nvidia`**: NVIDIA's driver; `nvidia-smi` should answer.
- **`nouveau`**: the open-source driver, which CUDA cannot use.
- **none bound**: no driver at all.

Installing a driver needs root and changes the system.

- **On a cluster** it is for the administrators. Tell them the node and the
  check's PCI line; never try it on a cluster node.
- **On a machine of your own**, follow NVIDIA's Driver Installation Guide for
  your Linux (docs.nvidia.com/datacenter/tesla/driver-installation-guide, read
  16 November 2026). In outline:
  1. the kernel headers of the running kernel (Ubuntu:
     `apt install linux-headers-$(uname -r)`; Rocky Linux 9:
     `dnf install kernel-devel-matched kernel-headers`, with EPEL enabled);
  2. NVIDIA's repository;
  3. the driver, which the guide gives as `nvidia-open`. Its package
     `nvidia-driver-assistant` looks at the card and names the right package
     for it;
  4. a reboot, then `nvidia-smi`, which should list the card.

The CUDA toolkit is not the driver. PyTorch and CuPy bring their own CUDA
libraries and need only the driver; the toolkit (`nvcc`) is needed only to
build a program such as MotionCor3.

## Building MotionCor3 and AreTomo2

Both come as source from their GitHub repositories. Read on 16 November 2026
(MotionCor3's `main` at commit dd8b683, 3 October 2025, tag v1.0.1; AreTomo2's
`main`), both build the same way:

```bash
git clone https://github.com/czimaginginstitute/MotionCor3.git
cd MotionCor3
make exe -f makefile11 CUDAHOME=<the CUDA toolkit's folder> CONDA=<an environment with libtiff>
```

AreTomo2 is the same with its own repository, and needs no libtiff. A GPU of
compute capability 5.x takes `makefile` instead of `makefile11`. What to choose:

- **The CUDA toolkit: a CUDA 12 one.** The makefiles ask for code for Volta
  (`compute_70`), and AreTomo2's for older GPUs too. CUDA 13 can no longer build
  any of these (CUDA 13.0 release notes), so a CUDA 13 toolkit fails. On a
  cluster, take the site's (`module avail cuda`; `--find` lists them) and give
  its folder, the one above `bin/nvcc`. Without one and without root, NVIDIA's
  runfile installs the toolkit alone into a folder of yours:
  `sh cuda_<version>_linux.run --silent --toolkit --toolkitpath=<folder>` (CUDA
  installation guide, "Advanced options"). `CUDAHOME` left out falls back to the
  makefile's own guess, such as `~/nvidia/cuda-12.1`, which is seldom there.
- **libtiff, for MotionCor3**: an environment of its own,
  `micromamba create -n mc3-build -c conda-forge libtiff`, given as
  `CONDA=<that environment's folder>`. The makefile's default is
  `~/miniconda3`. At run time MotionCor3 needs that `lib/` folder on
  `LD_LIBRARY_PATH` too, in the shell that submits (M-5).
- **Code for every GPU the lanes will use.** The makefiles compile cubins for a
  fixed list and no PTX. MotionCor3's covers compute capability 7.0 to 9.0, so a
  B300 (10.3) cannot run it. AreTomo2's covers 5.2 to 8.6, so H100, H200 and
  B300 cannot run it (I-14). Before building, add to `CUFLAG` in `makefile11`:
  - `-gencode arch=compute_90,code=sm_90` for H100 and H200 (AreTomo2);
  - `-gencode arch=compute_100,code=sm_100` for B300; CUDA 12.8 is the first
    to compile for it (CUDA 12.8 release notes);
  - or, for any newer GPU, PTX that the driver compiles when the program starts:
    `-gencode arch=compute_90,code=compute_90`.

After building, check the program before naming it in a lane:

- `./MotionCor3 --version` answers;
- `ldd ./MotionCor3 | grep "not found"` prints nothing, with the libtiff folder
  on `LD_LIBRARY_PATH`;
- `python3 scripts/show_setup.py`, with `MOTIONCOR3` naming it, shows its GPU
  code and the cluster's GPU types it runs on.

The job's preflight checks the same on the node (M-1, M-5). Problems met while
building are in I-13.

## What `show_setup.py` cannot tell

- **Your quota, read.** It prints what the site's tool says, and does not read
  its columns, which differ from site to site.
- **Another node.** Its GPU part speaks for the node it runs on, and for the
  cluster's GPU types by their names.
- **Variables set after the shell started.** A session, or a terminal, opened
  before `~/.bashrc` changed has the old variables. Run it in a new terminal.
- **Whether a step runs.** A version and a list of GPU code are not a run. The
  lane's dry run checks the lane, and a job's preflight checks the tools its
  stages need, on its node.
- **The nodes' view.** On a cluster it takes the login node's paths for the
  nodes', which holds where the software is on a shared filesystem.
- **What a module needs loaded.** `--find` takes a program's path from a module;
  whether the program also needs the module's libraries shows when it runs.
