# Installing: environments, tools, GPU and CUDA

> **Draft for review, 16 November 2026.**
>
> - Mark each entry by its ID: `I-2 ✓` if it is right, `I-3 ✗` if it is wrong
>   (say why), `I-4 ✎` if it is right but needs a change (say which).
> - Answer the questions at the end, a line each, and add what is missing.
> - Any language and any form: edit this file, comment on it, or reply by email.
> - Entries marked *for the maintainer* are about our code; skip them if you like.
>
> This box and the questions go once the review is in.

Kinds and evidence levels are explained in [playbook.md](../playbook.md). What
an installation needs, where each part goes, the versions and the GPU and CUDA
rules are in [setup.md](../setup.md). Three entries about the environment are
in [cluster.md](cluster.md): a full temporary folder (E-1), a program not found
(E-9), and another person's environment that cannot be read (E-10).

- **What is installed.** The clone, two environments (`cryoet-pipeline` and
  `pytom_tm`), the tools' variables, and the assistant's client. The tools
  themselves are shared installs.
- **Who installs.** The person, in their own shell. The assistant checks and
  explains, and gives the commands.

## How a healthy install looks

- **`python3 scripts/show_setup.py`**, in a new terminal where you submit from
  (a login node, or an interactive job such as `slogin` at MDC), ends in
  "Problems (0): none". Every tool has a version, the environment imports the
  pipeline from this clone, and PyTorch is built for a CUDA version.
- **The same check on a GPU node** of the kind the lanes use, through `srun`
  or directly in a job that has such a GPU, also shows PyTorch seeing the GPU,
  and each program's GPU code able to run on it.
- **The lane's dry run** lists its jobs and no problems.

## Where to look first

1. **`python3 scripts/show_setup.py`**: room, versions, GPU and CUDA, with the
   entry for each problem it finds.
2. **The error itself**, word for word, from pip, micromamba or the program.
3. **The room where it was writing**: `df -h` of the temporary folder and of the
   environments' folder, and the quota.

## Already checked by the code

- **`show_setup.py`**, when asked: the room, the environment and its packages,
  the tools and their versions, the GPU code against the node's GPU and the
  cluster's GPU types, and the CUDA builds against the node's driver.
- **The dry run**: the lane's keys, folders, tool variables and named programs,
  and that its CTFFIND is CTFFIND 5.
- **The job's preflight**, on its node: the tools each stage needs,
  MotionCor3's libraries and GPU code (M-1, M-5), MemBrain-seg's GPU (G-1), and
  pytom-match-pick's scripts.

## Problems

### I-1. Every `cryoet` command stops: no module named 'torch'

**Kind:** crash · **Evidence:** measured

**Error text:** `ModuleNotFoundError: No module named 'torch'`

**Where it shows:** every `cryoet` command, `cryoet init` included, in the
terminal and in every step of a job.

**How to confirm:** `python3 scripts/show_setup.py` lists torch among the
environment's missing packages, and names this entry.

**Causes, common to rare:**
1. The pipeline was installed from a clone older than 22 November 2026
   without the `gpu` extra, which then brought PyTorch: for example with
   `pip install -e ".[io]"`. Since then PyTorch comes with the pipeline
   itself.
2. The command ran in another environment than the pipeline's, such as the
   base environment.

**What to try:** from the clone, with the environment active,
`python -m pip install -e ".[io,viz,gpu,library]"`. Then
`python -c "import torch; print(torch.__version__, torch.version.cuda)"` prints
a version and a CUDA version.

**What not to do:** install a CPU-only PyTorch to save room. The MRA runs on
the GPU (I-6).

**Applies to:** since the averaging backend became part of the CLI; checked in
the code on 16 November 2026.

**Source:** MDC, 13 October 2026: with PyTorch blocked, no `cryoet` command
started. In the code, `cli.py` imports `backends/dynamo_search.py`, which
imports PyTorch.

### I-2. pip stops with `incomplete-download`

**Kind:** crash · **Evidence:** measured

**Error text:** `error: incomplete-download`

**Where it shows:** pip's output, often during PyTorch or one of its CUDA
libraries, the largest downloads of the install.

**How to confirm:** look at the temporary folder's room first
(`python3 scripts/show_setup.py`, or `df -h` of it). Then run the same command
again: a full temporary folder stops it at about the same size every time.

**Causes, common to rare:**
1. The network dropped.
2. The temporary folder is full, and pip reports the failed write as a broken
   download (E-1).

**What to try:**
1. If the temporary folder is small, E-1.
2. Otherwise run the same command again, with `--resume-retries 10`: pip then
   resumes a broken download itself. What it has downloaded stays in its cache.

**What not to do:** look for a proxy, or download the packages elsewhere, before
the temporary folder is checked. At MDC this was first read as the network.

**Applies to:** pip on any site.

**Source:** MDC, 13 October 2026: PyTorch's download stopped at 40.6 MB twice,
then pip said `No space left on device` at 40.4 MB. `/tmp` there holds 100 MB.

### I-3. "Disk quota exceeded", or a home too small for the environments

**Kind:** crash · **Evidence:** seen

**Error text:** `OSError: [Errno 122] Disk quota exceeded`, or `Disk quota
exceeded` from a program.

**Where it shows:** pip or micromamba part way through an install, or a program
part way through writing its output.

**How to confirm:** the site's quota tool; `python3 scripts/show_setup.py` runs
the ones it finds and prints what they say. `show_setup.py --sizes` measures
the environments' folder, micromamba's package cache and pip's cache.

**Causes, common to rare:**
1. The environments, their package cache and pip's cache are in the home,
   which has a small quota.
2. A run writes its outputs or scratch files into the home.

**What to try:**
1. Move them to a folder on a large filesystem: `MAMBA_ROOT_PREFIX` and
   `PIP_CACHE_DIR` in `~/.bashrc`, after micromamba's own block (I-4), as
   [setup.md](../setup.md) says under "Room". Then make the environments again
   there.
2. Empty the download caches, which hold downloads only: `pip cache purge` and
   `micromamba clean --all`.

**What not to do:** delete files inside an environment by hand.

**When to stop and ask someone:** when no folder on a large filesystem is
yours.

**Applies to:** any site with a home quota.

**Source:** Munich, autumn 2026: a `tilt` run filled the home quota part way
through a write. MDC, November 2026: home is almost always too small for the
environments there, and a pytom-match-pick environment in a home failed for lack
of space.

### I-4. New environments still go into the home

**Kind:** silent · **Evidence:** measured

**Where it shows:** `micromamba env list` names folders in the home, although
`~/.bashrc` sets `MAMBA_ROOT_PREFIX` to another folder.

**How to confirm:** in a new terminal, `echo "$MAMBA_ROOT_PREFIX"` prints a
folder in the home.

**Causes, common to rare:**
1. The lines come before micromamba's own block in `~/.bashrc`, which sets
   `MAMBA_ROOT_PREFIX` to the home again.
2. The terminal was opened before `~/.bashrc` changed.

**What to try:** move the lines below micromamba's block, open a new terminal,
and check again.

**Applies to:** micromamba. Measured with mamba 2.5.0 on a Mac; not yet on max.

**Source:** the setup at MDC, 12 November 2026.

### I-5. CuPy is missing, or pip builds it from source and fails

**Kind:** crash · **Evidence:** seen

**Error text:** pip downloads `cupy-<version>.tar.gz` and builds it, and the
build fails; or `No module named 'cupy'` from pytom-match-pick.

**Where it shows:** installing pytom-match-pick with pip, or the log of
`prepare-template` or `pick-particles`.

**How to confirm:** `python3 scripts/show_setup.py`: the line under
pytom-match-pick names its CuPy, or says that it does not import.

**Causes, common to rare:**
1. pytom-match-pick was installed with pip before CuPy came from conda-forge.
   On PyPI, `cupy` is source only; its built packages there are named
   `cupy-cuda12x` and `cupy-cuda13x`.
2. The environment's Python is newer than any CuPy build.

**What to try:** make the environment again as the README does: Python 3.12 and
CuPy with `cuda-version=12.*` from conda-forge in `micromamba create`, then
pytom-match-pick 0.14.0 with pip.

**What not to do:** let pip build CuPy. It needs a CUDA toolkit, and room in
the temporary folder (E-1).

**Source:** MDC, 14 October 2026: the environment's Python came out 3.15, and
pip took `cupy-14.2.0.tar.gz`. Its build stopped with `No space left on
device`, as `TMPDIR` was not set in that shell.

### I-6. PyTorch has no GPU to work on

**Kind:** crash · **Evidence:** seen; the driver rule general

**Error text** (PyTorch's own words):
- `Torch not compiled with CUDA enabled`: a CPU-only build;
- `The NVIDIA driver on your system is too old`: a build for a newer CUDA than
  the driver supports;
- `No CUDA GPUs are available`: no GPU was given to the process.

**Where it shows:** a step that runs PyTorch on the GPU, such as the MRA.

**How to confirm:** `python3 scripts/show_setup.py` on a GPU node of the kind
the job had ([setup.md](../setup.md), "GPU and CUDA"). It gives the CUDA the
build is for, the one the driver supports, and whether PyTorch sees the GPU.

**Causes, common to rare:**
1. The job had no GPU, or the step ran on a login node.
2. A CPU-only PyTorch: `torch.version.cuda` is `None`.
3. A PyTorch for a newer CUDA major version than the node's driver supports.
   CUDA 13 needs driver 580 or later; Munich's GPU nodes had driver 575 (CUDA
   12.9) on 7 October 2026.

**What to try:**
1. For cause 1, give the stage a GPU in its `sbatch` list.
2. For causes 2 and 3, install a PyTorch built for a CUDA version the driver
   supports; pytorch.org gives the command for each.

**When to stop and ask someone:** when the driver is too old for every PyTorch
the pipeline can use.

**Source:** MDC, 14 October 2026: PyTorch did not see the GPU of a node that had
one, in job 3609935; the cause was not established. The drivers: NVIDIA's CUDA
Toolkit release notes, read 16 November 2026.

### I-7. A job cannot activate the environment on some nodes

**Kind:** crash · **Evidence:** measured

**Error text:** `EnvironmentNameNotFound`, after a Python error from the
system's conda naming `X509_V_FLAG_NOTIFY_POLICY`.

**Where it shows:** the first lines of the job's log, before its header.

**How to confirm:** `sacct -j <job> -o NodeList` names the node, and that node
has `/usr/bin/conda`. Jobs of the same lane on other nodes started.
`ENV_MANAGER` is not set in the shell the job was submitted from.

**Causes, common to rare:**
1. `scripts/activate_env.sh` takes conda whenever one is on `PATH`. Some nodes
   have a system conda, which fails, and the environment, which is
   micromamba's, is not found.

**What to try:** `export ENV_MANAGER=micromamba` in `~/.bashrc` (the README's
block for MDC has it), a new terminal, then resume (recipe A).

**Applies to:** MDC, node by node: max164 has a system conda; max017 and max025
do not (9 November 2026).

**Source:** MDC, 9 November 2026: job 3907183 on max164 failed this way, while
job 3901834 on max017 took micromamba.

### I-8. The environment runs another clone's code

**Kind:** silent · **Evidence:** measured

**Where it shows:** nowhere. The jobs of this clone run the code of the clone
the environment was installed from.

**How to confirm:** `python3 scripts/show_setup.py`: its "pipeline" line says
"NOT this clone". Or
`python -c "import cryoet_pipeline; print(cryoet_pipeline.__file__)"`.

**Causes, common to rare:**
1. `pip install -e` from another clone, or a worktree, into the same
   environment. An editable install points the environment at the clone it was
   made from.

**What to try:** one environment per clone. To point an environment at this
clone, `python -m pip install -e ".[io,viz,gpu,library]"` from this clone; the
other clone then runs this one's code.

**Source:** 9 October 2026, in a worktree: tests passed that had imported the
main clone's code.

### I-9. CuPy's kernels do not compile when picking

**Kind:** crash · **Evidence:** measured

**Error text:** `NVRTC_ERROR_COMPILATION`, with a compile error in one of
CuPy's kernels such as `incomplete type "__nv_fp8_e8m0"`, and `42 errors
detected in the compilation of`.

**Where it shows:** the log of `pick-particles`, in pytom-match-pick's output,
at its first angle: the GPU stage's job fails within minutes, and the stages
after it are cancelled.

**How to confirm:** `python3 scripts/show_setup.py` on a GPU node, through
`srun`; a login node compiles nothing. Under pytom-match-pick it compiles the
reduction pytom-match-pick runs on its score map, once with CuPy's CUB and once
without: `its reduction kernel, with CuPy's CUB: fails`, and `without CUB:
compiles and sums right`. The versions the check shows above it can all agree:
on cryoem.se CuPy built for CUDA 12.9 compiled with NVRTC 12.9, and the check
without this compile found no problem.

**Causes, common to rare:**
1. CuPy's CUB reduction does not compile with the CUDA headers it is given; why
   is not known. On cryoem.se every version agreed: CuPy 14.2.0 built for CUDA
   12.9, from conda-forge, NVRTC 12.9, and the environment's own headers of 12.9
   (`cuda-cccl` 12.9.27, `cuda-nvrtc` 12.9.86). At MDC the same CuPy compiled
   with NVRTC 13.3 from the system's `/usr/local/cuda-13.3`, with the headers of
   CUDA 12.5, whose `nvcc` a loaded module had put on `PATH`.

**What to try:** turn CuPy's CUB off for picking, `export CUPY_ACCELERATORS=`
(set to nothing) in the shell the lane is submitted from, then resume with
recipe A of [rerun_recipes.md](../rerun_recipes.md). Picking passes the
shell's variables to pytom-match-pick, and a lane's jobs get the shell's. CuPy
then sums with kernels of its own; whether that slows picking was not measured.
The check above says, in a note, that this shell turns CUB off. The Stockholm
course accounts set it in `session.sh`.

**What not to do:** set `CUDA_PATH` to the older CUDA: at MDC that changed
nothing. Resubmit picking unchanged: it fails at the same kernel.

**When to stop and ask someone:** before the first lane on a site is submitted
with CUB off, and whenever the check says the kernel fails without CUB too.
Tell the maintainer how it ended.

**Applies to:** pytom-match-pick 0.14.0 with CuPy 14.2.0 built for CUDA 12.9,
at MDC and on cryoem.se.

**Source:** MDC, 14 October 2026, `pick-particles` of a lane with a z-axis
symmetry and a tomogram mask. Another lane, with neither, picked on the same
day. How it ended was not recorded. cryoem.se, 25 November 2026, the
Stockholm RyR1 lane on an RTX A5000: `pick-particles` of 039 at bin8 failed so, and
`show_setup.py` on the same node found no problem. A probe of the same kernel,
`4edd3527….cubin.cu` as in the job's log, failed with CUB and gave the right
sum with `CUPY_ACCELERATORS` empty; CuPy's own `sum` compiled with CUB on. A
lane with CUB off had not run when this was written.

### I-10. Wheels for Linux downloaded on a Mac lack the CUDA libraries

**Kind:** silent · **Evidence:** measured

**Where it shows:** an install from packages downloaded on a Mac with
`pip download --platform …`, for a cluster without internet access: PyTorch's
CUDA libraries are not among them.

**Causes, common to rare:**
1. pip judges `platform_system == "Linux"` by the Mac it runs on, so every
   dependency marked for Linux is dropped, PyTorch's CUDA libraries among them.

**What to try:** download on a Linux machine of the same architecture, or on
the cluster itself.

**Source:** a resolve without installing, on a Mac, 13 October 2026.

### I-12. An NVIDIA GPU is here, but no program can use it

**Kind:** crash · **Evidence:** general

**Error text:**
- `NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA
  driver`, from `nvidia-smi`;
- `Found no NVIDIA driver on your system`, from PyTorch.

**Where it shows:** `nvidia-smi` is missing or fails, while the machine has an
NVIDIA card.

**How to confirm:** `python3 scripts/show_setup.py`: its GPU part lists the card
on the PCI bus with the driver bound to it, `nouveau` or none.

**Causes, common to rare:**
1. No NVIDIA driver is installed, as on a new workstation.
2. The open-source `nouveau` driver holds the card.
3. The driver is installed, and its kernel module is not loaded until a reboot.

**What to try:**
1. On a cluster, tell its administrators, with the node and the check's PCI
   line.
2. On a machine of your own, install the driver as NVIDIA's guide for your Linux
   says ([setup.md](../setup.md), "The NVIDIA driver"), reboot, and check that
   `nvidia-smi` lists the card.

**What not to do:** install a CUDA toolkit to fix it: the toolkit is not the
driver. Never try to install a driver on a cluster node.

**When to stop and ask someone:** on any machine where you have no root.

**Applies to:** any NVIDIA GPU on Linux.

**Source:** NVIDIA's Driver Installation Guide, read 16 November 2026. Not met
on our data yet.

### I-13. MotionCor3 or AreTomo2 does not build

**Kind:** crash · **Evidence:** seen; manual for CUDA 13

**Error text:**
- `nvcc` not found under the makefile's `CUDAHOME`;
- `Unsupported gpu architecture 'compute_70'`, from a CUDA 13 `nvcc`;
- `tiffio.h: No such file or directory`, or `cannot find -ltiff` (MotionCor3);
- `cannot find -lcuda`.

**Where it shows:** the output of `make exe -f makefile11`.

**How to confirm:** the first error in the output; `<CUDAHOME>/bin/nvcc --version`
names the toolkit.

**Causes, common to rare:**
1. No `CUDAHOME` on the `make` line, and the makefile's own path does not
   exist (`~/nvidia/cuda-12.1` in MotionCor3's at commit dd8b683).
2. A CUDA 13 toolkit. The makefiles ask for Volta's `compute_70` (AreTomo2's
   for older GPUs too), which CUDA 13 no longer compiles.
3. libtiff is not in the folder `CONDA` names; the makefile's default is
   `~/miniconda3`.
4. The node has no NVIDIA driver, whose library the link step names (`-lcuda`).
   The toolkit has a stand-in in `lib64/stubs`.

**What to try:**
1. `CUDAHOME=<a CUDA 12 toolkit's folder>` on the `make` line: the site's module,
   or a runfile install into a folder of yours ([setup.md](../setup.md),
   "Building MotionCor3 and AreTomo2").
2. `CONDA=<an environment with libtiff>`, made from conda-forge.
3. With CUDA 13 only, take the `compute_70` line, and any older, out of
   `CUFLAG`. GPUs older than Turing then cannot run the build.
4. For `-lcuda`, build on a GPU node, or add `-L$(CUDAHOME)/lib64/stubs` to the
   link line.

**When to stop and ask someone:** when no CUDA 12 toolkit can be had.

**Applies to:** MotionCor3 `main` at dd8b683 (3 October 2025) and AreTomo2's
`makefile11`, read 16 November 2026.

**Source:** Munich, October 2026: the makefile named a CUDA 10.1 that the
cluster did not have, and had to point at CUDA 11.8. CUDA 13.0's release notes,
for CUDA 13.

### I-14. A build has no code for the newest GPUs

**Kind:** crash · **Evidence:** code

**Error text:**
- `MotionCor3 has nothing this GPU (compute <cap>) can run`, from the job's
  preflight (M-1);
- `no kernel image is available for execution on the device`, from AreTomo2.

**Where it shows:** a job on a newer GPU than the build covers.

**How to confirm:** `python3 scripts/show_setup.py`: the program's "GPU code"
line, and the cluster's GPU types it runs on.

**Causes, common to rare:**
1. The published `makefile11`s compile cubins for a fixed list, and no PTX.
   MotionCor3's covers compute capability 7.0 to 9.0, so a B300 (10.3) cannot
   run it. AreTomo2's covers 5.2 to 8.6, so H100, H200 and B300 cannot.

**What to try:**
1. Keep the stage on GPUs the build covers, with the lane's `--gres`.
2. Build again with the newer GPUs in `CUFLAG` ([setup.md](../setup.md),
   "Building MotionCor3 and AreTomo2"): `sm_90` for H100 and H200, `sm_100`
   for B300 (CUDA 12.8 or later), or PTX (`code=compute_90`) for anything newer.

**Applies to:** the makefiles as read on 16 November 2026, and the builds
installed at MDC as the check read them on 19 November: no PTX and nothing for
the B300s either, while their AreTomo2 has `sm_90` for H100 and H200
([setup.md](../setup.md), "Site facts").

**Source:** MotionCor3's and AreTomo2's `makefile11`; NVIDIA's table of compute
capabilities; CUDA 12.8's release notes. Not met in a job yet.

## Looks wrong, is fine

### I-11. PyTorch sees no GPU on a login node

**Kind:** false alarm · **Evidence:** seen

**Where it shows:** `torch.cuda.is_available()` is `False`, and `nvidia-smi`
is missing, on a login node.

**What it means:** login nodes have no GPU. What counts is the GPU nodes: check
there ([setup.md](../setup.md), "GPU and CUDA"). An interactive job without a
GPU looks the same; one with a GPU sees it, as `slogin` sessions on max's A40
nodes did on 19 November 2026.

**Source:** max's login nodes, October 2026.

## Questions for reviewers

Please answer in a line each; "don't know" is a useful answer.

- **Q-I1.** Where do you keep your environments, and how much room do they
  take? (`du -sh` of the environments' folder.)
- **Q-I2.** Which install problem have you met that is not here? In a line
  each: what you saw, what it turned out to be, what helped.
- **Q-I3.** Do you install anything else for your work with the pipeline, such
  as napari or ChimeraX, that others should know about?
