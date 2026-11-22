## The reference library

This session can read the reference library. It holds the documentation of the
tools the pipeline runs, at the versions installed on this site:

- IMOD's manual;
- the manual pages of Slurm and bash;
- what CTFFIND, MotionCor3, pytom-match-pick and MemBrain-seg print for `--help`;
- NVIDIA's pages on CUDA and GPUs (`--tool cuda`, `--tool nvidia`): for each
  CUDA version the site has, its release notes (the driver it needs, the GPUs
  it compiles for or dropped), its `nvcc` manual and its installation guide;
  the driver compatibility guide, the compute capability tables, MIG, and
  installing the driver;
- CuPy's pages at the version pytom-match-pick runs on (`--tool cupy`), and
  PyTorch's install commands for each CUDA version (`--tool pytorch`). The
  pages NVIDIA and PyTorch keep without versions say the date they were read;
- conda and micromamba (or mamba) at the site's versions: what their commands
  print for `--help` (`--tool conda`, `--tool mamba`), conda's user guide
  (`--tool conda-docs`: environments, channels, virtual packages and
  `CONDA_OVERRIDE_CUDA`, `.condarc` and its settings, where environments and
  the package cache go) and mamba's guides (`--tool mamba-docs`: installing
  micromamba, `MAMBA_ROOT_PREFIX`, configuration, troubleshooting), the last
  as reStructuredText from mamba's repository at the release's tag;
- the site card, `site.md`: tool versions, the CUDA versions found, partitions,
  GPU types, modules, and a GPU node's `nvidia-smi` when one was given.

`library.json` says what each page was read from. docs/library.md says how the
library is built.

- **Searching.** Search it with `python3 scripts/search_library.py <word>`, by
  the flag, option or message you want explained. Use `-e <flag>` for a word
  that starts with "-", and `--tool <name>` for one tool's pages. It prints
  each matching page with its tool, the tool's version and its source, and the
  matching lines with their numbers. Then read the page with Read, from that
  line.
- **What it is for.** It tells what a tool's flags, options and messages mean.
  What happened in a run still comes from that run's logs and QC records.
- **Citing.** Name what you used: the tool, its version and the page, such as
  "IMOD 5.1.12, man/newstack.html", or for a web page its address and version
  or date, such as "CUDA 12.5.0 release notes,
  docs.nvidia.com/cuda/archive/12.5.0/…" or "NVIDIA's compute capability table,
  as of 2026-11-16".
- **Versions.** If a run's log shows a version of a tool other than the
  library's, say so before relying on the page.
- **Read-only.** It is read-only for you, as for everyone. Never write, move or
  delete anything in it, and never ask the person to.
