---
name: install
description: Help a person install the pipeline and its tools, or check an installation - whether the pipeline can run on this machine or cluster and which steps, where the environments, downloads and temporary files go and whether there is room (disk space, quotas), which versions are needed and whether the installed ones fit together, tools that are installed but not found, and whether the GPUs, their driver and CUDA suit each program, building MotionCor3 included. Use when the person asks whether they can run the pipeline here, asks to install or set up the pipeline, an environment or a tool; asks about disk space, a quota, versions, a GPU, CUDA or a driver; or shows an error from pip, micromamba, conda, make, or a program that will not start.
---

# Installing and checking the tools

Paths are from the repository root. The knowledge behind this skill is in
`docs/agent/setup.md`: what can run where, what goes where and the room it
needs, finding installed tools, the versions, the GPU and CUDA rules, the
driver, building MotionCor3 and AreTomo2, and the order of installation.
Problems met so far are in `docs/agent/playbook/install.md` (the I- entries),
with E-1, E-9 and E-10 in `docs/agent/playbook/cluster.md`.

The person installs; you check, explain and give the commands. Never run pip,
micromamba, conda, an installer piped from `curl`, `make`, `sudo`, `module
load`, `git`, or anything else that writes outside the repository, and never
edit `~/.bashrc` or another start-up file:
- an install runs for minutes and writes gigabytes outside the repository;
- your shell is not theirs: an environment you activate, or a variable you
  export, is gone at your next command;
- `~/.bashrc` is theirs, and a driver is the machine's.

## 1. Check what is there

```bash
python3 scripts/show_setup.py
```

It writes nothing and needs no environment: it runs on the system's Python. It
asks every tool for its version and imports PyTorch in the environment, so give
it up to five minutes.

- `--env <name or folder>`: ask another environment than the one named
  `cryoet-pipeline`.
- `--config <lane config>`: find the tools as that lane names them, and need
  only what it needs.
- `--folder <path> --sizes`: the room where the data will go, and what each
  place already holds.
- `--find` (and `--search <folder>`): look for each tool where this shell does
  not name it (section 5). Up to a few minutes.

Read its problems first, then its notes. Each names a playbook entry: do what
the entry's "How to confirm" says before you name a cause.

Your shell is the one the session started in. A line added to `~/.bashrc` since
then, or an environment activated in the person's own terminal, is not in it.
To check the person's shell as it is now, either ask them to run the check
there and show you its problems and notes, or to start the session again from a
new terminal (`scripts/assistant -c` continues this one).

## 2. The clone

You run from the clone, so never help to clone it. The check's line about the
clone says whether its remote branch has commits the clone lacks. If it does,
and the person did not clone minutes ago, recommend `git pull` in the clone,
for them to run. After a pull that changes `pyproject.toml`, the pip install of
the pipeline runs again. If the clone has commits of its own, or changed files,
say so before they pull.

## 3. "Can I run the pipeline here?"

Answer from part 5 of the check, in plain words, step group by step group:
which run here, which run only on the CPU and slower, which do not and why. For
example: "No NVIDIA GPU here: a lane of stacks up to the tomogram, the template,
and from the picks to the MRA, on the CPU; not motion correction or picking,
which need an NVIDIA GPU."

- Where Slurm answers, the jobs run on the cluster's nodes: the answer is about
  the cluster's GPU types (part 4 lists them), not this login node's.
- Without Slurm, it is about this machine. The lanes' submissions need Slurm;
  here a lane's steps run with `scripts/run_series.py --execute`, which the
  person runs. Never run it yourself.
- With the lane's config (`--config`), part 5 shows only the steps that lane
  runs.

`docs/agent/setup.md`, "What can run where", says which steps need what.

## 4. Room first

Before anything is installed, check what `setup.md` lists under "Room":

- the temporary folder: 2 GB free at least, and the folder `TMPDIR` names
  exists (E-1);
- the environments, micromamba's package cache and pip's cache: on a large
  filesystem, not in a small home (I-3); `MAMBA_ROOT_PREFIX` set after
  micromamba's own block in `~/.bashrc` (I-4), and `PIP_CACHE_DIR`;
- the quota: what is free on a filesystem is not the person's share. Read the
  quota lines of the check. If no tool there prints a quota, say that you cannot
  see it;
- the data and the outputs: `--folder <where they will go> --sizes`.

Give the `~/.bashrc` lines with the person's folder filled in, say where in the
file they go, and ask them to open a new terminal and run the check again.

## 5. A tool that is installed but not found

Before anything is installed, look for it:
`python3 scripts/show_setup.py --find`, with `--search <folder>` for a group's
software folder the person knows. It lists each copy it found, its version, and
where (an environment, a module, a folder), best first.

- If a usable copy is there, propose naming it rather than installing another.
  Show the line, `export <VARIABLE>=<path>`.
- For several tools, propose a setup file:
  `python3 scripts/show_setup.py --setup-file` prints one with every variable
  the jobs need that this shell does not set yet. Show it. The person writes it
  to a folder of theirs, outside the clone, with
  `python3 scripts/show_setup.py --setup-file > <folder>/pipeline_setup.sh`,
  reads it, and adds `source <folder>/pipeline_setup.sh` to `~/.bashrc`.
- On a server the lines go into `~/.bashrc`, or a file it sources, after
  micromamba's block and outside any check for the host name: the jobs take
  them from the shell that submits.
- A copy found through a module may need the module loaded to run. Say so when
  `--find` names a module.

## 6. The install, in order

Follow `setup.md`, "Order of installation". The commands are in the README
("0. One-time setup", and at MDC its list "At MDC (max)") and in
`docs/tutorials/ryr1-without-assistant.md`, step 1, from after its clone. For
each step:

1. Say what it installs, where it writes, and how big it is, where `setup.md`
   knows.
2. Give the command in a code block, for the person to run in their own shell,
   and the folder to run it from.
3. Ask how it ended: the last lines, or the error word for word.
4. Run the check again before the next step.

Give only commands you have read in those files, in `setup.md`, or in a tool's
own documentation (`scripts/search_library.py`). If the person's site is not
covered, say so, and ask who installs the site's software.

## 7. Versions

Use the table in `setup.md`, "Versions", and say which kind each answer is:

- **needed by the code**: Python 3.11 or later, CTFFIND 5, PyTorch built for a
  CUDA version the GPU nodes' driver supports, MemBrain-seg on a GPU of compute
  capability 7.5 or later;
- **has run with**: IMOD 5.1.12, pytom-match-pick 0.14.0 and the rest of the
  table;
- **not in the table**: it has not run in the pipeline. It may work: run one
  series first, and tell the maintainers what happened.

## 8. GPU, driver and CUDA

- On a login node nothing sees a GPU, and that is fine (I-11). The check still
  reads each program's GPU code there, and says which of the cluster's GPU
  types it can run on.
- Whether the driver suits each CUDA build, and whether PyTorch and CuPy see the
  GPU, shows only on a GPU node. Give the person the `srun` command of
  `setup.md`, "On a GPU node", with the partition and GPU type of the lane's GPU
  stage. Never run `srun` yourself.
- **A card without its driver** (I-12): on a cluster, the administrators install
  drivers, so the person tells them, with the node and the check's PCI line. On
  the person's own machine, explain `setup.md`, "The NVIDIA driver", and point
  to NVIDIA's guide for their Linux; it needs root, and they run it. A CUDA
  toolkit does not replace the driver.
- **Building MotionCor3 or AreTomo2**: `setup.md`, "Building MotionCor3 and
  AreTomo2", step by step. Choose a CUDA 12 toolkit, give `CUDAHOME` and, for
  MotionCor3, `CONDA` on the `make` line. Add GPU code for every GPU type the
  lanes will use (I-14). After the build, check the program, then name it.
- Explain with `setup.md`, "In plain words": the driver and the CUDA version a
  program was built for; a GPU's compute capability and a program's cubins and
  PTX.
- For NVIDIA's, CuPy's or PyTorch's own words, where the session has the
  reference library, search it: `python3 scripts/search_library.py <words>`
  with `--tool cuda` (each CUDA version's release notes, `nvcc` manual and
  installation guide), `--tool nvidia` (driver compatibility, compute
  capabilities, MIG, installing the driver), `--tool cupy` or `--tool pytorch`
  (which build for which CUDA). The site card, `site.md` in the snapshot,
  lists the CUDA versions the site has, and a GPU node's driver when the
  library was built with one. Cite the page and its version or date.

## 9. When something fails

1. Get the error word for word, and what printed it.
2. Look it up: `docs/agent/playbook/install.md`; E-1, E-9 and E-10 in
   `cluster.md`; C-2 to C-4 in `ctf.md` (CTFFIND); M-1 and M-5 in `motion.md`
   (MotionCor3); G-1 and G-2 in `membranes.md` (MemBrain-seg); P-1 in
   `picking.md` (a GPU split into MIG slices).
3. Do what the entry's "How to confirm" says; the check often does it.
4. If no entry fits, say what the evidence shows and what is missing, and stop
   there.

## 10. Done

The installation is ready when:

- the check shows no problems, in a new terminal on a login node;
- and on a GPU node of the kind the lanes use;
- and the lane's dry run shows none (the `lane-config` skill).
