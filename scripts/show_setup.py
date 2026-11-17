#!/usr/bin/env python3
"""Check this host for what installing and running the pipeline needs: room,
versions, the GPU and CUDA.

    python3 scripts/show_setup.py
    python3 scripts/show_setup.py --env cryoet-pipeline
    python3 scripts/show_setup.py --config configs/lanes/ryr1-bin8.toml
    python3 scripts/show_setup.py --folder data/ryr1 --sizes
    python3 scripts/show_setup.py --json

Read-only. It reads files and variables, asks programs for their versions, and
asks nvidia-smi, sinfo and the site's quota tool what they report. It installs
nothing, changes nothing and writes nothing.

It runs on any Python from 3.9 on (the system Python of RHEL and Rocky 9), with
nothing installed, so that it works before the pipeline's environment exists.
What only an environment can answer (its packages, PyTorch, CuPy) is asked of
that environment's own Python.

What it reports, part by part (docs/agent/setup.md says what each means):

1. Room: where installs and runs write (home, this clone, $CRYOET_WORK, the
   temporary folder, the environments, the package caches, any --folder), what
   is free there, and the quota as the site's own tool prints it.
2. The pipeline's environment: its Python, whether the pipeline is installed
   from this clone, the packages of its extras, and PyTorch: its version, the
   CUDA it was built for, the GPU architectures it has code for, and whether it
   sees a GPU.
3. The tools, found as the lanes find them, with their versions against the
   ones the pipeline has run with, and what each needs: CTFFIND 5, MotionCor3's
   libraries and GPU code, pytom-match-pick's CuPy, MemBrain-seg's PyTorch.
4. The GPU and CUDA: this node's GPUs and driver, the GPU types the cluster
   lists, and which of them each program's GPU code can run on.

Then the problems, which stop an install or a run, and the notes, each with the
playbook entry that covers it.

A shell sees only its own variables. Run it in the shell the jobs will be
submitted from, in a new one if ~/.bashrc has changed; and on a GPU node,
through srun, for what only a GPU can show.

Exit status: 0 when nothing is a problem, 1 when something is, 2 for arguments
it cannot use.
"""

from __future__ import annotations

import argparse
import ctypes
import getpass
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_tool_versions() -> Any:
    """cryoet_pipeline.tool_versions, loaded from its file: importing the package
    would import pydantic and the rest, which a fresh host does not have. The
    module itself needs only the standard library."""

    path = ROOT / "src" / "cryoet_pipeline" / "tool_versions.py"
    spec = importlib.util.spec_from_file_location("_show_setup_tool_versions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tv = _load_tool_versions()

GB = 10**9
# PyTorch with its CUDA libraries is 18 wheels, 3.1 GB in all; the largest, torch
# itself, is 0.54 GB (PyPI, torch 2.15.0 for Python 3.13 on Linux x86_64, read
# 2026-11-16). pip writes each download into the temporary folder twice while it
# installs it (MDC, 2026-10-13), so a temporary folder with less free than this
# stops the install (E-1).
TEMP_NEEDS = 2 * GB
# Below this, a place installs or runs write to is nearly full.
NEARLY_FULL = 1 * GB
# Environments and download caches in a home with less free than this: PyTorch's
# downloads alone are 3.1 GB, and it takes more once unpacked (I-3).
HOME_NEEDS = 10 * GB

# The pipeline's own distribution and those of its extras (pyproject.toml), each
# with the extra it comes in. "core" are its dependencies.
PACKAGES = (
    ("cryoet-pipeline", "the pipeline"),
    ("numpy", "core"),
    ("pydantic", "core"),
    ("typer", "core"),
    ("PyYAML", "core"),
    ("mrcfile", "io"),
    ("pyarrow", "io"),
    ("tifffile", "io"),
    ("zarr", "io"),
    ("torch", "gpu"),
    ("torch-fourier-shell-correlation", "gpu"),
    ("matplotlib", "viz"),
    ("pypdf", "library"),
)
# Missing from these, the pipeline does not start or read its files. matplotlib
# only draws the reports, and pypdf only reads a manual into the library.
REQUIRED_EXTRAS = ("the pipeline", "core", "io", "gpu")
PYTHON_NEEDED = (3, 11)  # pyproject.toml, requires-python
ENV_NAME = "cryoet-pipeline"

# How the lanes name each tool, as scripts/build_library.py finds them: a lane
# config's key first, then the variable the lane configs name it by, then its
# names on PATH. tests/test_show_setup.py keeps the two scripts the same.
LANE_EXECUTABLES: dict[tuple[str, str], str] = {
    ("executables", "ctffind"): "ctffind",
    ("executables", "motioncor3"): "motioncor3",
    ("executables", "aretomo"): "aretomo",
    ("executables", "pytom_match_template"): "pytom-match-pick",
    ("executables", "pytom_create_template"): "pytom-match-pick",
    ("executables", "pytom_create_mask"): "pytom-match-pick",
    ("executables", "pytom_extract_candidates"): "pytom-match-pick",
    ("segment", "executable"): "membrain-seg",
}
LOOKUP: dict[str, tuple[str, tuple[str, ...]]] = {
    "ctffind": ("CTFFIND5", ("ctffind5", "ctffind")),
    "motioncor3": ("MOTIONCOR3", ("MotionCor3",)),
    "aretomo": ("ARETOMO", ("AreTomo2", "AreTomo")),
    "pytom-match-pick": ("PYTOMO_MP_DIR", ("pytom_match_template.py",)),
    "membrain-seg": ("MEMBRAIN_EXE", ("membrain",)),
}
# The backends' last place to look for IMOD.
MAC_IMOD = Path("/Applications/IMOD")
# The scripts a job's preflight requires in $PYTOMO_MP_DIR (scripts/lane.sbatch).
PYTOM_SCRIPTS = ("pytom_create_template", "pytom_create_mask", "pytom_match_template",
                 "pytom_extract_candidates")
TOOLS = ("imod", "ctffind", "motioncor3", "aretomo", "pytom-match-pick", "membrain-seg")
NAMES = {"imod": "IMOD", "ctffind": "CTFFIND", "motioncor3": "MotionCor3", "aretomo": "AreTomo",
         "pytom-match-pick": "pytom-match-pick", "membrain-seg": "MemBrain-seg"}
# The version each ran with on real data: at MDC on 2026-11-05, AreTomo2 on
# 2026-11-09 (docs/library.md, "Versions the pipeline records today").
TESTED = {"imod": "5.1.12", "ctffind": "5.0.2", "motioncor3": "1.2.4",
          "aretomo": "AreTomo2 1.1.3", "pytom-match-pick": "0.14.0", "membrain-seg": "0.0.10"}
# Which lanes need each tool, when no lane config is given.
EVERY_LANE = ("imod", "ctffind", "pytom-match-pick")
NEEDED_BY = {
    "imod": "every lane",
    "ctffind": "every lane",
    "pytom-match-pick": "every lane",
    "motioncor3": "lanes of frames ([lane] input = \"frames\")",
    "membrain-seg": "lanes that segment membranes ([segment])",
    "aretomo": "lanes aligned with AreTomo ([fine_align] backend = \"aretomo\")",
}
# MemBrain-seg's lowest compute capability (scripts/lane.sbatch, G-1), and the
# name its PyTorch goes by in the report.
MEMBRAIN_CAPABILITY = (7, 5)
MEMBRAIN = "MemBrain-seg's PyTorch"

# Compute capability by GPU model, for the GRES types sinfo lists: NVIDIA's tables
# (developer.nvidia.com/cuda/gpus and its legacy page, read 2026-11-16), for the
# GPUs met at MDC and in Munich. A model is matched as a word of the type's name,
# so that p600 is not taken for p6000.
GPU_MODELS = (
    ("h200", "9.0"), ("h100", "9.0"), ("b300", "10.3"), ("a100", "8.0"), ("a40", "8.6"),
    ("v100", "7.0"), ("t4", "7.5"), ("rtx8000", "7.5"), ("rtx_8000", "7.5"),
    ("rtx-8000", "7.5"), ("p6000", "6.1"), ("p600", "6.1"),
)
# The lowest driver for each CUDA major version on Linux (NVIDIA, CUDA Toolkit
# release notes, minor version compatibility; read 2026-11-16).
DRIVER_FOR_CUDA = {"13": "580", "12": "525", "11": "450"}

# NVIDIA GPUs on the PCI bus, which the kernel lists whether or not a driver is
# bound to them: vendor 0x10de, class display (0x0300) or 3D controller (0x0302).
SYS_PCI = Path("/sys/bus/pci/devices")
PCI_IDS = (Path("/usr/share/hwdata/pci.ids"), Path("/usr/share/misc/pci.ids"),
           Path("/usr/share/pci.ids"))

# The variable each tool is named by in a shell, for the lines of a setup file.
VARIABLES = {"imod": "IMOD_DIR", "ctffind": "CTFFIND5", "motioncor3": "MOTIONCOR3",
             "aretomo": "ARETOMO", "pytom-match-pick": "PYTOMO_MP_DIR",
             "membrain-seg": "MEMBRAIN_EXE"}
# Where --find looks besides the environments: the usual software folders, and
# the group folder of a site the pipeline runs at (MDC's, from the README). Only
# the folders that exist are searched, to SEARCH_DEPTH, for SEARCH_SECONDS each.
SEARCH_ROOTS = ("/opt", "/usr/local", "~/software", "~/apps", "~/opt", "~/bin", "~/.local/bin",
                "/fast/AG_Kudryashev/software")
SEARCH_DEPTH = 5
SEARCH_SECONDS = 20.0
# Folders --find does not enter: Python packages and package caches hold no
# program a lane runs, and are large.
SEARCH_SKIP = frozenset({"site-packages", "pkgs", "node_modules", "__pycache__", "proc", "sys",
                         "dev", "man", "include", "doc", "docs", "test", "tests"})
# Environment modules whose name says which tool they hold.
MODULE_NAMES = {
    "imod": r"imod", "ctffind": r"ctffind", "motioncor3": r"motioncor|mcor", "aretomo": r"aretomo",
    "pytom-match-pick": r"pytom", "membrain-seg": r"membrain", "cuda": r"(^|/)cuda\b",
}


@dataclass(frozen=True)
class StepGroup:
    """Lane steps that need the same things: their tools, and whether a GPU is
    required (MotionCor3, pytom-match-pick's matching, AreTomo2), optional (the
    MRA, which runs on the CPU too), advised (MemBrain-seg: the preflight takes
    no GPU below 7.5, and a run on the CPU has not been tried) or not used."""

    name: str
    steps: str
    tools: tuple[str, ...]
    gpu: str
    code: str = ""
    """Whose GPU code it runs: a GpuCode holder."""
    lanes: str = ""
    """Which lanes run it, when not every lane."""


# In the order a lane runs them. Read in the code on 2026-11-16: the backends and
# scripts/submit_lanes.py for the GPU, and pytom-match-pick 0.14.0's source, whose
# template and mask run on the CPU and whose matching takes --gpu-ids.
STEP_GROUPS = (
    StepGroup("motion correction", "prepare-tilt-series", ("motioncor3",), "required",
              "MotionCor3", 'input = "frames"'),
    StepGroup("stack import", "import-tilt-stack", (), "none", "", 'input = "stack"'),
    StepGroup("alignment", "align-tilt-series, fiducials or patches, fine alignment (tiltalign)",
              ("imod",), "none"),
    StepGroup("fine alignment with AreTomo2", "fine-align-tilt-series", ("aretomo",), "required",
              "AreTomo2", '[fine_align] backend = "aretomo"'),
    StepGroup("CTF", "build-ctf-stack, estimate-ctf, correct-ctf", ("imod", "ctffind"), "none"),
    StepGroup("tomogram", "build-final-aligned-stack, reconstruct-tomogram", ("imod",), "none"),
    StepGroup("membranes", "segment-tomogram", ("membrain-seg",), "advised",
              "MemBrain-seg's PyTorch", "[segment]"),
    StepGroup("template", "prepare-template", ("pytom-match-pick",), "none"),
    StepGroup("picking", "pick-particles", ("pytom-match-pick",), "required"),
    StepGroup("extraction", "visualize-picks, extract-particles", (), "none"),
    StepGroup("MRA", "the mra stage", (), "optional", "the pipeline's PyTorch"),
)

PROBE_TAG = "@@show_setup@@"
# Asked of an environment's Python, which may be any version: the packages, where
# the pipeline is imported from, and PyTorch. find_spec does not import the
# pipeline, which would import PyTorch before it is asked about.
ENV_PROBE = r"""
import json, sys
names = json.loads(sys.argv[1])
out = {"python": "%d.%d.%d" % tuple(sys.version_info[:3]), "executable": sys.executable}
try:
    import importlib.metadata as metadata
except ImportError:
    metadata = None
packages = {}
for name in names:
    try:
        packages[name] = metadata.version(name) if metadata else None
    except Exception:
        packages[name] = None
out["packages"] = packages
try:
    import importlib.util
    spec = importlib.util.find_spec("cryoet_pipeline")
    out["pipeline"] = spec.origin if spec else None
except Exception as error:
    out["pipeline_error"] = repr(error)[:300]
# Said once before PyTorch is imported, which can take minutes, so that what is
# known stays known if the import does not end in time.
print("@@show_setup@@" + json.dumps(out), flush=True)
if packages.get("torch"):
    torch_facts = {}
    try:
        import torch
        torch_facts["version"] = torch.__version__
        torch_facts["cuda"] = torch.version.cuda
        try:
            torch_facts["archs"] = (torch._C._cuda_getArchFlags() or "").split()
        except Exception as error:
            torch_facts["archs_error"] = repr(error)[:300]
        torch_facts["available"] = bool(torch.cuda.is_available())
        if torch_facts["available"]:
            torch_facts["devices"] = [
                [torch.cuda.get_device_name(i), "%d.%d" % torch.cuda.get_device_capability(i)]
                for i in range(torch.cuda.device_count())]
    except Exception as error:
        torch_facts["error"] = repr(error)[:300]
    out["torch"] = torch_facts
print("@@show_setup@@" + json.dumps(out))
"""
# Asked of pytom-match-pick's Python: its CuPy, the CUDA it was built for, the
# NVRTC it compiles kernels with, and whether it sees a GPU.
CUPY_PROBE = r"""
import json, sys
out = {"python": "%d.%d.%d" % tuple(sys.version_info[:3]), "executable": sys.executable}
facts = {}
try:
    import cupy
    facts["version"] = cupy.__version__
    for key, ask in (
        ("runtime", lambda: cupy.cuda.runtime.runtimeGetVersion()),
        ("driver", lambda: cupy.cuda.runtime.driverGetVersion()),
        ("nvrtc", lambda: "%d.%d" % tuple(cupy.cuda.nvrtc.getVersion())),
        ("cuda_path", lambda: cupy.cuda.get_cuda_path()),
        ("devices", lambda: cupy.cuda.runtime.getDeviceCount()),
    ):
        try:
            facts[key] = ask()
        except Exception as error:
            facts[key + "_error"] = repr(error)[:300]
except Exception as error:
    facts["error"] = repr(error)[:300]
out["cupy"] = facts
print("@@show_setup@@" + json.dumps(out))
"""


# --- what the report is made of ----------------------------------------------


@dataclass
class Finding:
    problem: bool
    text: str
    entry: str = ""
    """The playbook entry that covers it, such as "E-1"."""


@dataclass
class Place:
    label: str
    path: str
    why: str
    exists: bool = False
    mount: str = ""
    fstype: str = ""
    free: int | None = None
    total: int | None = None
    in_home: bool = False
    size: int | None = None
    size_note: str = ""


@dataclass
class Gpu:
    index: str
    name: str
    capability: str
    memory: str
    driver: str


@dataclass
class ToolReport:
    name: str
    needed_by: str
    needed: bool
    path: str = ""
    found: str = ""
    version: str = ""
    version_how: str = ""
    tested: str = ""
    lines: list[str] = field(default_factory=list)
    usable: bool = False
    """Found, readable, and for CTFFIND a CTFFIND 5: what a step can run."""
    elsewhere: list[dict[str, str]] = field(default_factory=list)
    """Copies --find found outside what this shell names, best first."""


@dataclass
class GpuCode:
    """A program's GPU code, from the architectures it was built for."""

    holder: str
    cubins: list[str] = field(default_factory=list)
    ptx: list[str] = field(default_factory=list)
    cuda: str = ""
    """The CUDA version it was built for, where known."""
    cpu_only: bool = False
    """A PyTorch built without CUDA."""


@dataclass
class Report:
    host: dict[str, Any] = field(default_factory=dict)
    places: list[Place] = field(default_factory=list)
    quota: list[dict[str, str]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    tools: list[ToolReport] = field(default_factory=list)
    clients: list[str] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)
    clone: dict[str, Any] = field(default_factory=dict)
    modules: dict[str, list[str]] = field(default_factory=dict)
    """--find: the environment modules whose names say which tool they hold."""
    steps: list[dict[str, str]] = field(default_factory=list)
    """What can run here, step group by step group."""
    findings: list[Finding] = field(default_factory=list)

    def problem(self, text: str, entry: str = "") -> None:
        self.findings.append(Finding(True, text, entry))

    def note(self, text: str, entry: str = "") -> None:
        self.findings.append(Finding(False, text, entry))


# --- small helpers -------------------------------------------------------------


def size_text(value: int | None) -> str:
    if value is None:
        return "?"
    for unit, scale in (("TB", 10**12), ("GB", GB), ("MB", 10**6)):
        if value >= scale:
            return f"{value / scale:.1f} {unit}"
    return f"{value / 1000:.0f} kB"


def run(command: list[str], seconds: float = 15.0) -> Any:
    """A bounded probe (cryoet_pipeline.tool_versions.probe): no input, a time limit."""

    return tv.probe(command, seconds=seconds)


def head(text: str, lines: int = 20) -> list[str]:
    return [line.rstrip() for line in text.strip().splitlines()[:lines]]


def under(path: Path, folder: Path) -> bool:
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(folder))
    except ValueError:
        return False
    return True


def version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


# --- 1. room -------------------------------------------------------------------


def existing(path: Path) -> Path:
    """The path, or the nearest folder above it that exists."""

    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def mount_of(path: Path) -> str:
    path = Path(os.path.realpath(path))
    while not os.path.ismount(path) and path != path.parent:
        path = path.parent
    return str(path)


def mounts() -> dict[str, str]:
    """Mount point -> filesystem type, from /proc/self/mounts (Linux)."""

    found: dict[str, str] = {}
    try:
        text = Path("/proc/self/mounts").read_text()
    except OSError:
        return found
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), parts[1])
            found[point] = parts[2]
    return found


def free_space(path: Path) -> tuple[int, int] | None:
    """Bytes free to this user and in all, on the filesystem that holds `path`."""

    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize, stat.f_blocks * stat.f_frsize


def temp_folder() -> tuple[str, str]:
    """The folder Python, and with it pip, keeps temporary files in, and why.

    Python takes the first of $TMPDIR, $TEMP, $TMP, /tmp, /var/tmp and /usr/tmp
    that it can write to, and says nothing when it passes one over (tempfile,
    gettempdir). It finds out by writing a file; this asks os.access instead, so
    that nothing is written."""

    candidates = [(os.environ.get(name), f"${name}") for name in ("TMPDIR", "TEMP", "TMP")]
    candidates += [(folder, "the system's default") for folder in ("/tmp", "/var/tmp", "/usr/tmp")]
    for folder, how in candidates:
        if folder and os.path.isdir(folder) and os.access(folder, os.W_OK | os.X_OK):
            return folder, how
    return os.getcwd(), "the current folder, since no other is writable"


def env_root() -> tuple[Path | None, str]:
    """Where new environments go: $MAMBA_ROOT_PREFIX, else the root of the active
    environment."""

    if value := os.environ.get("MAMBA_ROOT_PREFIX"):
        return Path(value), "$MAMBA_ROOT_PREFIX"
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        path = Path(prefix)
        if path.parent.name == "envs":
            return path.parent.parent, "the root of the active environment"
        return path, "the active environment's root"
    return None, ""


def pip_cache() -> tuple[Path | None, str]:
    if os.environ.get("PIP_NO_CACHE_DIR", "").lower() in ("1", "true", "yes", "on"):
        return None, "off ($PIP_NO_CACHE_DIR)"
    if value := os.environ.get("PIP_CACHE_DIR"):
        return Path(value), "$PIP_CACHE_DIR"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "pip", "pip's default"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "pip", "pip's default"


def places(report: Report, folders: list[Path], sizes: bool) -> None:
    home = Path.home()
    temp, temp_how = temp_folder()
    root, root_how = env_root()
    wanted: list[tuple[str, Path | None, str]] = [
        ("home", home, "your home folder"),
        ("this clone", ROOT, "the outputs of its lanes, unless a lane names another folder"),
    ]
    if work := os.environ.get("CRYOET_WORK"):
        wanted.append(("$CRYOET_WORK", Path(work), "your folder for the clone and the environments"))
    wanted.append(("temporary files", Path(temp), f"{temp_how}: pip unpacks downloads here"))
    if root is not None:
        wanted.append(("environments", root / "envs" if (root / "envs").is_dir() else root,
                       f"{root_how}: new environments go here"))
        pkgs = os.environ.get("CONDA_PKGS_DIRS", "").split(",")[0] or str(root / "pkgs")
        wanted.append(("package cache", Path(pkgs),
                       "micromamba's and conda's downloads" +
                       (" ($CONDA_PKGS_DIRS)" if os.environ.get("CONDA_PKGS_DIRS") else "")))
    cache, cache_how = pip_cache()
    if cache is not None:
        wanted.append(("pip's cache", cache, f"{cache_how}: pip's downloads"))
    if prefix := os.environ.get("CONDA_PREFIX"):
        wanted.append(("active environment", Path(prefix), "the environment active in this shell"))
    for folder in folders:
        wanted.append(("--folder", folder, "named on the command line"))

    types = mounts()
    for label, path, why in wanted:
        assert path is not None
        place = Place(label, str(path), why, exists=path.exists(), in_home=under(path, home))
        anchor = existing(path)
        place.mount = mount_of(anchor)
        place.fstype = types.get(place.mount, "")
        if (space := free_space(anchor)) is not None:
            place.free, place.total = space
        if sizes and place.exists:
            place.size, place.size_note = folder_size(path)
        report.places.append(place)

    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and os.path.realpath(tmpdir) != os.path.realpath(temp):
        report.problem(f"TMPDIR is {tmpdir}, which is not a folder you can write to, so Python and "
                       f"pip use {temp} instead, without a word: create the folder, or name another",
                       "E-1")
    full: dict[str, list[Place]] = {}
    for place in report.places:
        if place.free is None:
            continue
        if place.label == "temporary files" and place.free < TEMP_NEEDS:
            report.problem(
                f"the temporary folder {place.path} has {size_text(place.free)} free: pip writes "
                f"each download there twice while it installs it, and PyTorch's largest are about "
                f"0.5 GB each, so its install would stop with 'No space left on device'. Set "
                f"TMPDIR to a folder of yours on a large filesystem, and create that folder",
                "E-1")
        elif place.free < NEARLY_FULL:
            full.setdefault(place.mount, []).append(place)
    for mount, held in full.items():
        report.problem(f"the filesystem at {mount} has only {size_text(held[0].free)} free, and "
                       f"{', '.join(place.label for place in held)} write there", "I-3")
    in_home = [place for place in report.places
               if place.in_home and place.label in ("environments", "package cache", "pip's cache")]
    home_place = report.places[0]
    if in_home and home_place.free is not None and home_place.free < HOME_NEEDS:
        report.note(
            f"{', '.join(place.label for place in in_home)} go into your home, which has "
            f"{size_text(home_place.free)} free on its filesystem: PyTorch alone downloads 3.1 GB "
            f"and takes more once installed. Point MAMBA_ROOT_PREFIX and PIP_CACHE_DIR at a "
            f"folder on a large filesystem (docs/agent/setup.md, 'Room')", "I-3")
    if root is None:
        report.note("this shell names no place for environments (no $MAMBA_ROOT_PREFIX and no "
                    "active environment): micromamba then uses its own default, which "
                    "'micromamba info' names", "I-4")


def folder_size(path: Path, seconds: float = 60.0) -> tuple[int | None, str]:
    """What a folder holds, as `du -sk` counts it, within a time limit."""

    result = run(["du", "-sk", str(path)], seconds=seconds)
    if result.stopped == "time":
        return None, f"not measured: du took over {seconds:.0f} s"
    found = re.match(r"\s*(\d+)", result.output)
    if not found:
        return None, "not measured: " + (head(result.output, 1) or [result.error or "no output"])[0]
    return int(found.group(1)) * 1024, ""


QUOTA_TOOLS = (
    ("quota", ["quota", "-s"]),
    ("mmlsquota", ["mmlsquota", "--block-size", "auto"]),
)


def quotas(report: Report) -> None:
    """The quota, as the site's own tools print it, unread: the tools and their
    columns differ from site to site. Linux only: the BSD quota of a Mac takes
    other options."""

    if not sys.platform.startswith("linux"):
        return
    for name, command in QUOTA_TOOLS:
        program = shutil.which(name)
        if program is None and name == "mmlsquota" and os.path.isfile("/usr/lpp/mmfs/bin/mmlsquota"):
            program = "/usr/lpp/mmfs/bin/mmlsquota"
        if program is None:
            continue
        result = run([program, *command[1:]])
        report.quota.append({"command": " ".join(command),
                             "output": "\n".join(head(result.output, 25)) or (result.error or "")})
    user = getpass.getuser()
    lustre = sorted({place.mount for place in report.places if place.fstype == "lustre"})
    if lustre and (lfs := shutil.which("lfs")):
        for mount in lustre:
            result = run([lfs, "quota", "-h", "-u", user, mount])
            report.quota.append({"command": f"lfs quota -h -u {user} {mount}",
                                 "output": "\n".join(head(result.output, 25))})


# --- 2. the pipeline's environment -----------------------------------------------


def conda_base() -> Path | None:
    """conda's base folder, from the conda on PATH (bin/conda, or condabin/conda
    in a shell conda's init has set up), without running it: a system conda can
    crash when asked (I-7)."""

    found = shutil.which("conda") or os.environ.get("CONDA_EXE")
    return Path(found).resolve().parent.parent if found else None


def env_folders() -> list[tuple[Path, str]]:
    """Folders environments live in, as envs/<name> under each, in the order a
    job would look: the micromamba root a job is given, conda's base, then the
    usual places."""

    folders: list[tuple[Path, str]] = []
    if value := os.environ.get("MAMBA_ROOT_PREFIX"):
        folders.append((Path(value) / "envs", "$MAMBA_ROOT_PREFIX"))
    if base := conda_base():
        folders.append((base / "envs", "conda's base"))
    for usual in ("~/micromamba", "~/.local/share/mamba", "~/miniforge3", "~/mambaforge",
                  "~/miniconda3", "~/anaconda3"):
        folders.append((Path(usual).expanduser() / "envs", "a usual place"))
    folders.append((Path("~/.conda/envs").expanduser(), "a usual place"))
    return folders


def env_python(env: Path) -> Path:
    return env / "bin" / "python"


def choose_env(given: str | None) -> tuple[Path | None, str]:
    """The environment to ask: --env; else the one named cryoet-pipeline, which
    a job activates by that name (scripts/activate_env.sh); else the active one."""

    if given:
        path = Path(given).expanduser()
        if env_python(path).exists():
            return path, "named with --env"
        for folder, how in env_folders():
            if env_python(folder / given).exists():
                return folder / given, f"named with --env, found under {folder} ({how})"
        return None, f"--env {given}: no environment of that name or folder"
    active = os.environ.get("CONDA_PREFIX")
    for folder, how in env_folders():
        if env_python(folder / ENV_NAME).exists():
            env = folder / ENV_NAME
            if active and os.path.realpath(active) == os.path.realpath(env):
                return env, "active in this shell"
            return env, (f"found under {folder} ({how}); not active in this shell"
                         + (f", where {active} is" if active else ""))
    if active:
        return Path(active), f"active in this shell; none is named {ENV_NAME}"
    return None, ""


def env_manager(report: Report, facts: dict[str, Any]) -> None:
    """How a job will activate the environment: scripts/activate_env.sh takes
    $ENV_MANAGER, else conda when there is one on PATH, else micromamba."""

    chosen = os.environ.get("ENV_MANAGER")
    conda = shutil.which("conda")
    micromamba = shutil.which("micromamba") or (
        os.environ.get("MAMBA_EXE") if os.access(os.environ.get("MAMBA_EXE", ""), os.X_OK) else None)
    if chosen:
        facts["manager"] = f"{chosen} ($ENV_MANAGER)"
        if chosen not in ("conda", "micromamba"):
            report.problem(f"ENV_MANAGER is '{chosen}': scripts/activate_env.sh takes conda or "
                           "micromamba, and a job stops on anything else", "I-7")
        return
    if conda:
        facts["manager"] = f"conda ({conda}), since ENV_MANAGER is not set"
        root = os.environ.get("MAMBA_ROOT_PREFIX")
        base = conda_base()
        if micromamba and root and base and os.path.realpath(root) != os.path.realpath(base):
            report.note("conda and micromamba are both here, with different roots, and ENV_MANAGER "
                        "is not set: a job activates with conda. If your environments are "
                        "micromamba's, export ENV_MANAGER=micromamba", "I-7")
        return
    if micromamba:
        facts["manager"] = f"micromamba ({micromamba}), since there is no conda on PATH"
        report.note("ENV_MANAGER is not set: a job on a node that has a system conda would take "
                    "that conda and not find the environment. Export ENV_MANAGER=micromamba in "
                    "~/.bashrc", "I-7")
        return
    facts["manager"] = "neither conda nor micromamba found"
    report.problem("neither conda nor micromamba is on PATH (or in $MAMBA_EXE): a job cannot "
                   "activate the environment", "I-7")


def ask_python(python: Path, script: str, argument: str = "[]", seconds: float = 180.0) -> dict:
    """Run a probe script under another Python, isolated (-I), and read its line.
    The time limit allows for importing PyTorch from a cold network filesystem."""

    result = tv.probe([str(python), "-I", "-c", script, argument], seconds=seconds, limit=1 << 20)
    lines = result.output.splitlines()
    for line in reversed(lines):
        if line.startswith(PROBE_TAG):
            try:
                found = json.loads(line[len(PROBE_TAG):])
            except ValueError:
                continue
            said = [text for text in lines if not text.startswith(PROBE_TAG)]
            found["said"] = head("\n".join(said), 8)
            if result.stopped == "time":
                found["stopped"] = f"it did not finish within {seconds:.0f} s"
            return found
    if result.error:
        why = result.error
    elif result.stopped == "time":
        why = f"no answer within {seconds:.0f} s"
    else:
        why = "it printed no answer"
    return {"error": why, "said": head(result.output, 12)}


def environment(report: Report, given: str | None) -> dict[str, Any]:
    env, how = choose_env(given)
    facts: dict[str, Any] = {"path": str(env) if env else "", "how": how}
    report.environment = facts
    env_manager(report, facts)
    if env is not None and "not active in this shell" in how:
        report.note(f"the pipeline's environment is not active in this shell: its scripts run "
                    f"with the active Python. Activate it first (micromamba activate {ENV_NAME}, "
                    f"or conda activate {ENV_NAME})")
    if env is None:
        if how:
            report.problem(how)
        else:
            report.problem(
                f"no environment with the pipeline: none is active in this shell, and none named "
                f"{ENV_NAME} is in the usual places. Create it (docs/agent/setup.md, 'Order of "
                f"installation'), or name it with --env")
        return facts
    names = [name for name, _ in PACKAGES]
    found = ask_python(env_python(env), ENV_PROBE, json.dumps(names))
    facts.update(found)
    if "error" in found:
        report.problem(f"the environment's Python ({env_python(env)}) did not answer: {found['error']}")
        return facts
    if version_tuple(found["python"]) < PYTHON_NEEDED:
        report.problem(f"the environment's Python is {found['python']}; the pipeline needs "
                       f"{'.'.join(map(str, PYTHON_NEEDED))} or later (pyproject.toml)")
    packages = found.get("packages", {})
    missing = [(name, extra) for name, extra in PACKAGES if not packages.get(name)]
    hard = [name for name, extra in missing if extra in REQUIRED_EXTRAS]
    if "cryoet-pipeline" in hard:
        report.problem(f"the pipeline is not installed in {env}: from this clone, "
                       "python -m pip install -e \".[io,viz,gpu,library]\"", "I-1")
    elif hard:
        report.problem(
            f"{env} lacks {', '.join(hard)}, which the pipeline needs to start"
            + (" (every cryoet command imports PyTorch)" if "torch" in hard else "")
            + ": install the extras again, python -m pip install -e \".[io,viz,gpu,library]\"",
            "I-1" if "torch" in hard else "")
    for name, extra in missing:
        if extra not in REQUIRED_EXTRAS:
            report.note(f"{name} is missing ({extra} extra): "
                        + ("the reports need it" if extra == "viz"
                           else "only building the reference library needs it"))
    origin = found.get("pipeline")
    if origin:
        package = Path(origin).parent
        facts["from_this_clone"] = under(package, ROOT / "src")
        if not facts["from_this_clone"]:
            report.problem(f"the environment imports the pipeline from {package}, not from this "
                           f"clone ({ROOT}): its cryoet commands run that code, the lanes of this "
                           f"clone included", "I-8")
    torch_facts = found.get("torch") or {}
    if packages.get("torch") and not torch_facts and found.get("stopped"):
        report.note(f"importing PyTorch in {env}: {found['stopped']}, so it was not asked about "
                    "its CUDA (a first import from a network filesystem can be that slow)")
    if torch_facts.get("error"):
        report.problem(f"PyTorch is installed but does not import: {torch_facts['error']}", "I-1")
    elif torch_facts and not torch_facts.get("cuda"):
        text = ("PyTorch in the environment is a CPU-only build (torch.version.cuda is None): the "
                "MRA and the other GPU steps cannot run")
        if sys.platform == "darwin":
            report.note(text + "; on a Mac that is expected, and the GPU steps run on the cluster")
        else:
            report.problem(text, "I-6")
    return facts


# --- 3. the tools ----------------------------------------------------------------


@dataclass
class Lane:
    path: Path
    cfg: dict[str, Any]


def read_lanes(paths: list[Path], report: Report) -> list[Lane]:
    if not paths:
        return []
    try:
        import tomllib  # type: ignore[import-not-found]  # Python 3.11 and later
    except ImportError:
        report.note("--config is read with tomllib, which this Python "
                    f"({platform.python_version()}) lacks: run this script with the "
                    "environment's Python to check the tools as the lanes name them")
        return []
    lanes = []
    for path in paths:
        with open(path, "rb") as handle:
            lanes.append(Lane(path, tomllib.load(handle)))
    return lanes


def lane_needs(lanes: list[Lane]) -> set[str]:
    needs = set(EVERY_LANE)
    for lane in lanes:
        cfg = lane.cfg
        if (cfg.get("lane") or {}).get("input") == "frames" and \
                (cfg.get("motion") or {}).get("backend") == "motioncor3":
            needs.add("motioncor3")
        if "segment" in cfg:
            needs.add("membrain-seg")
        if (cfg.get("fine_align") or {}).get("backend") == "aretomo":
            needs.add("aretomo")
    return needs


def path_from(value: str, how: str) -> tuple[Path | None, str, str]:
    """A tool named by a value with $VARs, as run_series.py expands it: the path,
    how it was found, and what is wrong with it."""

    expanded = os.path.expandvars(value)
    if unset := re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)", expanded):
        return None, how, f"{how}: ${', $'.join(unset)} is not set in this shell"
    path = Path(expanded).expanduser()
    if not path.is_absolute() and (found := shutil.which(expanded)):
        path = Path(found)
    if path.is_dir():
        return path, how, f"{how}: {path} is a folder, not a program"
    if not path.exists():
        return path, how, f"{how}: {path} does not exist"
    if not os.access(path, os.R_OK):
        return path, how, f"{how}: {path} cannot be read by you (another person's folder?)"
    return path, how, ""


def find_tool(name: str, lanes: list[Lane]) -> tuple[Path | None, str, str]:
    keys = [key for key, tool in LANE_EXECUTABLES.items() if tool == name]
    for lane in lanes:
        for section, key in keys:
            value = (lane.cfg.get(section) or {}).get(key)
            if value is not None:
                if name == "pytom-match-pick" and key != "pytom_match_template":
                    continue
                return path_from(str(value), f'{lane.path.name} [{section}] {key} = "{value}"')
    variable, programs = LOOKUP[name]
    if value := os.environ.get(variable):
        if name == "pytom-match-pick":
            value = os.path.join(value, "pytom_match_template.py")
        return path_from(value, f"${variable}")
    for program in programs:
        if found := shutil.which(program):
            return Path(found), f"{program} on PATH", ""
    if len(programs) == 1:
        missing = f"no {programs[0]} is on PATH"
    else:
        missing = f"neither {' nor '.join(programs)} is on PATH"
    return None, "", f"${variable} is not set, and {missing}"


def find_imod() -> tuple[Path | None, str, str]:
    """IMOD's folder, as the backends find its programs: newstack on PATH, then
    $IMOD_DIR/bin, then /Applications/IMOD/bin."""

    candidates: list[tuple[Path, str]] = []
    if found := shutil.which("newstack"):
        candidates.append((Path(found), "newstack on PATH"))
    if imod_dir := os.environ.get("IMOD_DIR"):
        candidates.append((Path(imod_dir) / "bin" / "newstack", "$IMOD_DIR"))
    candidates.append((MAC_IMOD / "bin" / "newstack", str(MAC_IMOD)))
    for program, how in candidates:
        if program.is_file():
            return program.resolve().parent.parent, how, ""
    if os.environ.get("IMOD_DIR"):
        return None, "$IMOD_DIR", f"$IMOD_DIR is {os.environ['IMOD_DIR']}, which has no bin/newstack"
    return None, "", "$IMOD_DIR is not set, and no newstack is on PATH"


def tools(report: Report, lanes: list[Lane]) -> list[GpuCode]:
    """Each tool found and asked; returns the GPU code of the programs that have
    some, for part 4."""

    needs = lane_needs(lanes) if lanes else set(EVERY_LANE)
    gpu_code: list[GpuCode] = []
    for name in TOOLS:
        needed = name in needs
        tool = ToolReport(NAMES[name], NEEDED_BY[name], needed, tested=TESTED[name])
        report.tools.append(tool)
        path, how, wrong = find_imod() if name == "imod" else find_tool(name, lanes)
        tool.found = how
        tool.path = str(path) if path else ""
        if wrong:
            tool.lines.append(wrong)
            if "cannot be read" in wrong:
                entry = "E-10"
            elif name == "ctffind":
                # A named path that is no program, or no CTFFIND named at all.
                entry = "C-3" if path is not None else "C-4"
            else:
                entry = "E-9"
            text = f"{NAMES[name]} (needed by {NEEDED_BY[name]}): {wrong}"
            (report.problem if needed else report.note)(text, entry)
            continue
        if path is None:
            continue
        version = ask_version(name, path)
        tool.version = version.value or ""
        tool.version_how = version.how
        tool.usable = True
        if not version.value:
            tool.lines.append(f"states no version ({version.how})")
        elif version.value != TESTED[name]:
            report.note(f"{NAMES[name]} {version.value} has not run in the pipeline; "
                        f"{TESTED[name]} has (docs/agent/setup.md, 'Versions')")
        if name == "ctffind" and version.value and version_tuple(version.value)[0] < 5:
            tool.usable = False
            report.problem(f"{path} is CTFFIND {version.value}; estimate-ctf needs CTFFIND 5. "
                           "Name CTFFIND 5 in $CTFFIND5 or the lane's [executables] ctffind", "C-2")
        if name == "motioncor3":
            gpu_code.append(gpu_binary(path, "MotionCor3", tool, report, "M-5"))
        if name == "aretomo":
            gpu_code.append(gpu_binary(path, "AreTomo2", tool, report, ""))
        if name == "pytom-match-pick":
            pytom(path, tool, report)
        if name == "membrain-seg":
            code = membrain(path, tool, report)
            if code is not None:
                gpu_code.append(code)
    return gpu_code


def ask_version(name: str, path: Path) -> Any:
    """A tool's version, asked as the backends ask it (tool_versions)."""

    if name == "imod":
        return tv.imod_version(path)
    if name == "ctffind":
        return tv.ctffind_version(path)
    if name == "motioncor3":
        return tv.stated_version(path, "MotionCor3")
    if name == "aretomo":
        return tv.Version(tv.aretomo_version_of(path), f"{path.name} --version")
    python = tv.script_python(path)
    return (tv.package_version(python, name) if python
            else tv.Version(None, f"{path} names no Python on its first line"))


def gpu_code_of(path: Path, holder: str) -> GpuCode:
    """A CUDA program's GPU code, as cuobjdump lists it: cubins (-lelf) and PTX
    (-lptx), each named by its sm_ architecture. Empty without cuobjdump."""

    code = GpuCode(holder)
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
    if os.access(cuobjdump, os.X_OK):
        for flag, into in (("-lelf", code.cubins), ("-lptx", code.ptx)):
            result = run([cuobjdump, flag, str(path)], seconds=60.0)
            into.extend(sorted(set(re.findall(r"sm_\d+[af]?", result.output)), key=arch_number))
    return code


def gpu_binary(path: Path, holder: str, tool: ToolReport, report: Report, entry: str) -> GpuCode:
    """MotionCor3 or AreTomo2: the libraries it cannot load, the CUDA it is
    linked against, and its GPU code."""

    code = gpu_code_of(path, holder)
    if ldd := shutil.which("ldd"):
        result = run([ldd, str(path)])
        missing = sorted({line.split()[0] for line in result.output.splitlines()
                          if "not found" in line})
        if missing:
            tool.lines.append("libraries not found: " + " ".join(missing))
            tool.usable = False
            report.problem(f"{holder} cannot load {' '.join(missing)}: put the folder that holds "
                           "them on LD_LIBRARY_PATH before submitting", entry)
        if runtime := re.search(r"libcudart\.so\.(\d+)", result.output):
            code.cuda = runtime.group(1)
            tool.lines.append(f"linked against CUDA {code.cuda} (libcudart.so.{code.cuda})")
    if code.cubins or code.ptx:
        tool.lines.append(f"GPU code: cubins {' '.join(code.cubins) or 'none'}; "
                          f"PTX {' '.join(code.ptx) or 'none'} ({coverage(code)})")
    elif not (shutil.which("cuobjdump") or os.access("/usr/local/cuda/bin/cuobjdump", os.X_OK)):
        tool.lines.append("GPU code not read: no cuobjdump here (it comes with the CUDA toolkit)")
    return code


def pytom(path: Path, tool: ToolReport, report: Report) -> None:
    folder = path.parent
    absent = [script for script in PYTOM_SCRIPTS if not (folder / f"{script}.py").is_file()]
    if absent:
        tool.usable = False
        report.problem(f"{folder} lacks {', '.join(f'{s}.py' for s in absent)}, which a job's "
                       "preflight requires: is PYTOMO_MP_DIR the bin folder of pytom-match-pick's "
                       "environment?", "E-9")
    python = tv.script_python(path)
    if python is None:
        return
    tool.lines.append(f"runs under {python}")
    found = ask_python(python, CUPY_PROBE, seconds=120.0)
    cupy = found.get("cupy") or {}
    if "error" in found or cupy.get("error"):
        tool.lines.append(f"CuPy: {found.get('error') or cupy.get('error')}")
        # Every pytom-match-pick script imports CuPy, the template's too.
        tool.usable = False
        report.problem(f"pytom-match-pick's CuPy does not import under {python}", "I-5")
        return
    parts = [f"CuPy {cupy.get('version')}"]
    if "runtime" in cupy:
        parts.append(f"built for CUDA {cuda_text(cupy['runtime'])}")
    if "nvrtc" in cupy:
        parts.append(f"compiles with NVRTC {cupy['nvrtc']}")
    if cupy.get("cuda_path"):
        parts.append(f"from {cupy['cuda_path']}")
    parts.append(f"sees {cupy['devices']} GPU(s)" if "devices" in cupy else "sees no GPU here")
    tool.lines.append("; ".join(parts))
    report.gpu.setdefault("cupy", cupy)
    runtime = cuda_text(cupy.get("runtime"))
    nvrtc = cupy.get("nvrtc", "")
    if runtime and nvrtc and runtime.split(".")[0] != nvrtc.split(".")[0]:
        report.note(f"pytom-match-pick's CuPy is built for CUDA {runtime} but compiles its kernels "
                    f"with NVRTC {nvrtc}: a mix like this broke picking at MDC", "I-9")


def membrain(path: Path, tool: ToolReport, report: Report) -> GpuCode | None:
    checkpoint = os.environ.get("MEMBRAIN_CKPT")
    if not checkpoint:
        tool.lines.append("MEMBRAIN_CKPT is not set")
        tool.usable = False
        report.note("MEMBRAIN_CKPT is not set: a lane that segments membranes needs the model's "
                    "checkpoint", "G-2")
    elif not os.path.isfile(checkpoint):
        tool.lines.append(f"MEMBRAIN_CKPT: {checkpoint} is not a file")
        tool.usable = False
        report.problem(f"MEMBRAIN_CKPT is {checkpoint}, which is not a file", "G-2")
    python = tv.script_python(path)
    if python is None:
        return None
    found = ask_python(python, ENV_PROBE, json.dumps(["membrain-seg", "torch"]))
    torch_facts = found.get("torch") or {}
    if not torch_facts or torch_facts.get("error"):
        tool.lines.append(f"its PyTorch: {torch_facts.get('error') or found.get('error') or 'none'}")
        return None
    code = GpuCode(MEMBRAIN, cuda=torch_facts.get("cuda") or "",
                   cpu_only=not torch_facts.get("cuda"))
    split_archs(torch_facts.get("archs", []), code)
    tool.lines.append(f"runs under {python}; its PyTorch {torch_line(torch_facts, code)}")
    return code


# --- 4. the GPU and CUDA -----------------------------------------------------------


def arch_number(arch: str) -> int:
    return int(re.sub(r"\D", "", arch.split("_")[1]))


def split_archs(flags: list[str], code: GpuCode) -> None:
    """PyTorch's arch flags: sm_XY is a cubin, compute_XY is PTX."""

    for flag in flags:
        if flag.startswith("sm_"):
            code.cubins.append(flag)
        elif flag.startswith("compute_"):
            code.ptx.append("sm_" + flag.split("_", 1)[1])


def cap_number(capability: str) -> int:
    major, minor = capability.split(".")
    return int(major) * 10 + int(minor)


def runs_on(capability: str, code: GpuCode) -> bool:
    """Whether a GPU of this compute capability can run the code: a cubin runs on
    its own major version at an equal or higher minor ("a" cubins only on their
    own), and PTX is compiled at load time for any GPU at least as new as its
    target. The same rule as scripts/lane.sbatch's preflight."""

    cap = cap_number(capability)
    for arch in code.cubins:
        number = arch_number(arch)
        if arch.endswith("a"):
            if number == cap:
                return True
        elif number // 10 == cap // 10 and number <= cap:
            return True
    return any(arch_number(arch) <= cap for arch in code.ptx)


def usable(capability: str, code: GpuCode) -> bool:
    """Whether the program can run on a GPU of this compute capability: it has
    code for it, and for MemBrain-seg the GPU is one a job's preflight accepts."""

    if code.holder == MEMBRAIN and version_tuple(capability) < MEMBRAIN_CAPABILITY:
        return False
    return runs_on(capability, code)


def coverage(code: GpuCode) -> str:
    """The compute capabilities the code runs on, in words."""

    parts = []
    for arch in code.cubins:
        number = arch_number(arch)
        major, minor = divmod(number, 10)
        parts.append(f"{major}.{minor} only" if arch.endswith("a") else f"{major}.{minor} to {major}.9")
    if code.ptx:
        lowest = min(arch_number(arch) for arch in code.ptx)
        parts.append(f"{lowest // 10}.{lowest % 10} and newer through PTX")
    return "runs on compute capability " + ", ".join(parts) if parts else "no GPU code found"


def cuda_text(value: Any) -> str:
    """A CUDA version as text: 12090 -> 12.9."""

    if isinstance(value, int) and value > 0:
        return f"{value // 1000}.{value % 1000 // 10}"
    return str(value) if value else ""


def torch_line(facts: dict[str, Any], code: GpuCode) -> str:
    parts = [str(facts.get("version"))]
    parts.append(f"built for CUDA {facts['cuda']}" if facts.get("cuda") else "CPU-only build")
    if code.cubins or code.ptx:
        parts.append(f"cubins {' '.join(code.cubins) or 'none'}, PTX {' '.join(code.ptx) or 'none'}")
    if facts.get("available"):
        devices = ", ".join(f"{name} ({cap})" for name, cap in facts.get("devices", []))
        parts.append(f"sees {devices}")
    else:
        parts.append("sees no GPU here")
    return "; ".join(parts)


def driver_cuda() -> str:
    """The highest CUDA version this node's driver supports, asked of the driver
    itself (cuDriverGetVersion in libcuda); empty without a driver. A toolkit's
    stub libcuda answers with an error, which counts as none."""

    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        value = ctypes.c_int(0)
        try:
            if library.cuDriverGetVersion(ctypes.byref(value)) == 0 and value.value > 0:
                return cuda_text(value.value)
        except AttributeError:
            continue
    return ""


def gpus_here(report: Report) -> list[Gpu]:
    """The GPUs a working NVIDIA driver serves on this node; and the cards on the
    PCI bus that none serves (driverless)."""

    found = working_gpus(report)
    driverless(report, found)
    return found


def working_gpus(report: Report) -> list[Gpu]:
    facts = report.gpu
    smi = shutil.which("nvidia-smi")
    if smi is None:
        facts["here"] = "no nvidia-smi: no NVIDIA driver on this node (a login node usually has none)"
        return []
    result = run([smi, "--query-gpu=index,name,compute_cap,memory.total,driver_version",
                  "--format=csv,noheader"])
    found = []
    for line in result.output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5 and re.fullmatch(r"\d+\.\d+", parts[2]):
            found.append(Gpu(*parts))
    facts["gpus"] = [asdict(gpu) for gpu in found]
    if not found:
        facts["here"] = "nvidia-smi lists no GPU: " + " ".join(head(result.output, 2))
    listing = run([smi, "-L"])
    facts["listing"] = head(listing.output, 16)
    if any("MIG" in line for line in facts["listing"]):
        report.note("a GPU here is split into MIG slices: nvidia-smi's memory is the whole card's, "
                    "and a job gets its slice's share; -L lists the slices", "P-1")
    cuda = driver_cuda()
    if not cuda:
        banner = run([smi])
        if match := re.search(r"CUDA(?: UMD)? Version:\s*(\d+\.\d+)", banner.output):
            cuda = match.group(1)
    facts["driver_cuda"] = cuda
    return found


def cluster_gpus(report: Report) -> dict[str, str]:
    """GPU type -> compute capability, for the GRES types sinfo lists."""

    sinfo = shutil.which("sinfo")
    if sinfo is None:
        return {}
    result = run([sinfo, "-h", "-o", "%P|%G"])
    # Slurm answers: the lanes' jobs run on its nodes, not on this one.
    report.gpu["slurm"] = result.ended_well and bool(result.output.strip())
    types: dict[str, str] = {}
    for line in result.output.splitlines():
        for name in re.findall(r"gpu:([A-Za-z0-9_.\-]+):\d+", line):
            types[name] = model_capability(name) or ""
    report.gpu["cluster"] = types
    return types


def model_capability(gres: str) -> str:
    name = gres.lower()
    for model, capability in GPU_MODELS:
        if re.search(rf"(?<![a-z0-9]){re.escape(model)}(?![0-9])", name):
            return capability
    return ""


def cuda_fit(built: str, driver: str) -> str:
    """'ok', 'minor' or 'no', for a program built for CUDA `built` on a driver
    that supports CUDA `driver`. A newer driver runs older builds. Within one
    major version a newer build runs too (minor version compatibility), except
    code compiled at run time. A newer major version needs a newer driver."""

    built_v, driver_v = version_tuple(built)[:2], version_tuple(driver)[:2]
    if built_v <= driver_v:
        return "ok"
    return "minor" if built_v[0] == driver_v[0] else "no"


def gpu_and_cuda(report: Report, code: list[GpuCode]) -> list[GpuCode]:
    """Part 4. Returns every program's GPU code, the pipeline's PyTorch among
    them, for part 5."""

    facts = report.gpu
    here = gpus_here(report)
    cluster = cluster_gpus(report)
    facts["visible"] = {name: os.environ[name] for name in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS")
                        if name in os.environ}
    if nvcc := shutil.which("nvcc"):
        result = run([nvcc, "--version"])
        release = re.search(r"release (\d+\.\d+)", result.output)
        facts["nvcc"] = f"{nvcc} (CUDA {release.group(1)})" if release else nvcc
    facts["cuda_vars"] = {name: os.environ[name] for name in ("CUDA_PATH", "CUDA_HOME")
                          if name in os.environ}
    facts["cuda_libraries"] = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                               if "cuda" in part.lower()]

    pipeline_torch = report.environment.get("torch") or {}
    if pipeline_torch.get("version") and not pipeline_torch.get("error"):
        pipeline_code = GpuCode("the pipeline's PyTorch", cuda=pipeline_torch.get("cuda") or "")
        split_archs(pipeline_torch.get("archs", []), pipeline_code)
        report.environment["torch_line"] = torch_line(pipeline_torch, pipeline_code)
        if pipeline_code.cuda:
            code = [pipeline_code, *code]

    targets = [(f"{gpu.name} here", gpu.capability) for gpu in here]
    targets += [(name, capability) for name, capability in sorted(cluster.items()) if capability]
    unknown = sorted(name for name, capability in cluster.items() if not capability)
    if unknown:
        facts["unknown_types"] = unknown
    fits: list[str] = []
    for holder in code:
        if not (holder.cubins or holder.ptx):
            continue
        if not targets:
            fits.append(f"{holder.holder}: {coverage(holder)}")
            continue
        yes = [name for name, capability in targets if usable(capability, holder)]
        no = [name for name, capability in targets if not usable(capability, holder)]
        line = f"{holder.holder}: runs on {', '.join(yes) or 'none of them'}"
        if no:
            line += f"; not on {', '.join(no)}"
        fits.append(line)
        for gpu in here:
            if usable(gpu.capability, holder):
                continue
            entry = {"MotionCor3": "M-1", MEMBRAIN: "G-1"}.get(holder.holder, "I-6")
            if runs_on(gpu.capability, holder):
                report.problem(f"MemBrain-seg needs a GPU of compute capability "
                               f"{'.'.join(map(str, MEMBRAIN_CAPABILITY))} or later, and this "
                               f"node's {gpu.name} is {gpu.capability}: a job's preflight refuses "
                               "it", entry)
            else:
                report.problem(f"{holder.holder} has no code this node's {gpu.name} (compute "
                               f"capability {gpu.capability}) can run: 'no kernel image is "
                               f"available'", entry)
    facts["fits"] = fits

    driver = facts.get("driver_cuda", "")
    builds = [(holder.holder, holder.cuda) for holder in code if holder.cuda]
    cupy = facts.get("cupy") or {}
    if cuda_text(cupy.get("runtime")):
        builds.append(("pytom-match-pick's CuPy", cuda_text(cupy["runtime"])))
    facts["builds"] = [f"{holder}: CUDA {cuda}" for holder, cuda in builds]
    if not driver:
        # Only where Slurm has GPU nodes to run the check on.
        if builds and facts.get("slurm"):
            report.note("no NVIDIA driver here, so whether it suits "
                        + ", ".join(f"{holder} (CUDA {cuda})" for holder, cuda in builds)
                        + " shows only on a GPU node: run this check there, through srun "
                        "(docs/agent/setup.md, 'GPU and CUDA')", "I-11")
        return code
    for holder, cuda in builds:
        fit = cuda_fit(cuda, driver)
        major = cuda.split(".")[0]
        if fit == "no":
            report.problem(f"{holder} is built for CUDA {cuda}, and this node's driver supports "
                           f"CUDA {driver} at most: CUDA {major} needs a driver "
                           f"{DRIVER_FOR_CUDA.get(major, 'newer than this one')} or later. Install "
                           f"a build for CUDA {driver.split('.')[0]}, or use nodes with a newer "
                           "driver", "I-6")
        elif fit == "minor":
            report.note(f"{holder} is built for CUDA {cuda} and the driver supports {driver}: it "
                        "runs by minor version compatibility, except code compiled at run time")
    if here and pipeline_torch.get("cuda") and not pipeline_torch.get("available"):
        report.problem("this node has a GPU and the pipeline's PyTorch does not see it", "I-6")
    return code


# --- an NVIDIA GPU without its driver ------------------------------------------------


def pci_name(device_id: str) -> str:
    """An NVIDIA device's name from the PCI ID database that lspci reads."""

    ident = device_id.lower().replace("0x", "")
    for path in PCI_IDS:
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        inside = False
        for line in lines:
            if not line.startswith("\t"):
                inside = line.startswith("10de ")
            elif inside and not line.startswith("\t\t") and line[1:5].lower() == ident:
                return line[5:].strip()
        return ""
    return ""


def pci_gpus() -> list[dict[str, str]]:
    """The NVIDIA GPUs the kernel lists on the PCI bus, with the driver bound to
    each: "nvidia" for NVIDIA's, "nouveau" for the open-source one, "" for none.
    The kernel lists a card whether or not a driver is bound to it."""

    found: list[dict[str, str]] = []
    try:
        devices = sorted(SYS_PCI.iterdir())
    except OSError:
        return found
    for device in devices:
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            klass = (device / "class").read_text().strip().lower()
            ident = (device / "device").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x10de" or not klass.startswith(("0x0300", "0x0302")):
            continue
        driver = device / "driver"
        bound = os.path.basename(os.path.realpath(driver)) if os.path.lexists(driver) else ""
        found.append({"slot": device.name, "id": ident,
                      "name": pci_name(ident) or f"device {ident}", "driver": bound})
    return found


def driverless(report: Report, working: list[Gpu]) -> None:
    """The cards on the PCI bus that no working NVIDIA driver serves."""

    pci = pci_gpus()
    report.gpu["pci"] = pci
    if not pci or working:
        return
    names = ", ".join(sorted({gpu["name"] for gpu in pci}))
    bound = {gpu["driver"] for gpu in pci}
    if bound == {"nvidia"}:
        report.note(f"NVIDIA's driver holds this node's GPU ({names}), but nvidia-smi does not answer "
                    "here: is it on PATH?", "I-12")
        return
    held = " (the open-source nouveau driver holds it)" if "nouveau" in bound else ""
    report.problem(f"this node has an NVIDIA GPU ({names}), but NVIDIA's driver is not loaded{held}: "
                   "no program can use it. On a cluster that is for its administrators; on your own "
                   "machine, install the driver (docs/agent/setup.md, 'The NVIDIA driver')", "I-12")


# --- the clone --------------------------------------------------------------------------


def git_command(root: Path, *args: str, seconds: float = 15.0) -> Any:
    """A git command that only reads, in the clone. --no-optional-locks keeps
    `git status` from writing the index, and git asks nobody for a password."""

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="", SSH_ASKPASS="",
               GIT_SSH_COMMAND="ssh -o BatchMode=yes")
    return tv.probe([shutil.which("git") or "git", "-C", str(root), "--no-optional-locks", *args],
                    seconds=seconds, env=env)


def clone_state(report: Report, root: Path = ROOT) -> None:
    """The clone's commit, its changes, and whether its remote branch has commits
    it lacks, read with git ls-remote: nothing is fetched or written."""

    facts: dict[str, Any] = {"path": str(root)}
    report.clone = facts
    if shutil.which("git") is None or not (root / ".git").exists():
        facts["state"] = "not a git clone, or no git here"
        return
    head = git_command(root, "rev-parse", "HEAD").output.strip()
    facts["head"] = head[:7]
    facts["branch"] = git_command(root, "rev-parse", "--abbrev-ref", "HEAD").output.strip()
    facts["date"] = git_command(root, "log", "-1", "--format=%cs").output.strip()
    status = git_command(root, "status", "--porcelain", "--untracked-files=no")
    facts["changed"] = len([line for line in status.output.splitlines() if line.strip()])
    upstream = git_command(root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    name = upstream.output.strip()
    if not upstream.ended_well or "/" not in name:
        facts["state"] = "its branch follows no branch of a remote"
        return
    remote, branch = name.split("/", 1)
    facts["upstream"] = name
    listed = git_command(root, "ls-remote", remote, f"refs/heads/{branch}", seconds=30.0)
    words = listed.output.split()
    if not listed.ended_well or not words or not re.fullmatch(r"[0-9a-f]{40}", words[0]):
        facts["state"] = f"{remote} did not answer, so whether it has newer commits is unknown"
        return
    remote_head = words[0]
    changed = facts["changed"]
    if remote_head == head:
        facts["state"] = f"up to date with {name}"
    elif not git_command(root, "cat-file", "-e", f"{remote_head}^{{commit}}").ended_well:
        facts["state"] = f"{name} has commits this clone does not"
        report.note(f"{name} has commits this clone does not: run git pull in {root}. After a pull "
                    "that changes pyproject.toml, run the pip install of the pipeline again"
                    + (f". {changed} files are changed in this clone, and a pull may stop on them"
                       if changed else ""))
    elif git_command(root, "merge-base", "--is-ancestor", remote_head, "HEAD").ended_well:
        facts["state"] = f"ahead of {name}: it has commits {name} lacks"
    else:
        facts["state"] = f"apart from {name}: each has commits the other lacks"
        report.note(f"this clone and {name} each have commits the other lacks: a git pull merges "
                    "them. If you did not commit here yourself, ask the maintainers")


# --- tools on this host that this shell does not name -----------------------------------


def is_program(tool: str, name: str) -> bool:
    """Whether a file name is the program --find looks for, for a tool."""

    if tool == "imod":
        return name == "newstack"
    if tool == "ctffind":
        return name in ("ctffind", "ctffind5")
    if tool == "motioncor3":
        return name.startswith("MotionCor3") and not name.endswith(
            (".cu", ".cpp", ".h", ".o", ".a", ".txt", ".md", ".log", ".png", ".pdf"))
    if tool == "aretomo":
        return re.fullmatch(r"AreTomo[23]?([_-][\w.\-]*)?", name) is not None
    if tool == "pytom-match-pick":
        return name == "pytom_match_template.py"
    if tool == "membrain-seg":
        return name == "membrain"
    return False


def is_checkpoint(name: str) -> bool:
    return name.endswith(".ckpt") and "membrain" in name.lower()


def search_folders(extra: list[Path]) -> list[Path]:
    """The folders --find searches: SEARCH_ROOTS, $CRYOET_WORK and --search,
    those that exist, each once."""

    roots = [Path(os.path.expanduser(root)) for root in SEARCH_ROOTS] + list(extra)
    if work := os.environ.get("CRYOET_WORK"):
        roots.append(Path(work))
    chosen: list[Path] = []
    seen: list[str] = []
    for root in roots:
        real = os.path.realpath(root)
        if os.path.isdir(real) and not any(real == s or real.startswith(s + os.sep) for s in seen):
            seen.append(real)
            chosen.append(root)
    return chosen


def walk_programs(root: Path, seconds: float = SEARCH_SECONDS,
                  depth: int = SEARCH_DEPTH) -> tuple[list[tuple[str, Path]], bool]:
    """(tool, path) of every program of a tool under root, and whether the time
    ran out first. Goes into no hidden folder and none of SEARCH_SKIP; follows
    no link to a folder."""

    deadline = time.monotonic() + seconds
    base = len(root.parts)
    found: list[tuple[str, Path]] = []
    for folder, dirs, files in os.walk(root):
        if time.monotonic() > deadline:
            return found, True
        here = Path(folder)
        if len(here.parts) - base >= depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SEARCH_SKIP]
        for name in files:
            path = here / name
            if is_checkpoint(name):
                found.append(("membrain-ckpt", path))
                continue
            for tool in TOOLS:
                if is_program(tool, name) and os.access(path, os.X_OK):
                    found.append((tool, path))
    return found, False


def env_programs() -> list[tuple[str, Path, str]]:
    """pytom-match-pick and MemBrain-seg in any environment of the usual roots."""

    found: list[tuple[str, Path, str]] = []
    for folder, how in env_folders():
        try:
            envs = sorted(path for path in folder.iterdir() if path.is_dir())
        except OSError:
            continue
        for env in envs:
            for tool, program in (("pytom-match-pick", "pytom_match_template.py"),
                                  ("membrain-seg", "membrain")):
                path = env / "bin" / program
                if path.is_file() and os.access(path, os.X_OK):
                    found.append((tool, path, f"environment {env.name}, under {folder} ({how})"))
    return found


def module_names() -> list[str]:
    """The environment modules a login shell offers (`module -t avail`)."""

    result = run(["bash", "-lc", "module -t avail 2>&1"], seconds=30.0)
    names = []
    for line in result.output.splitlines():
        token = line.strip()
        if not token or token.endswith(":") or " " in token or token.startswith(("/", "-")):
            continue
        names.append(re.sub(r"\(.*\)$", "", token))
    return names


def module_paths(text: str) -> list[str]:
    """The folders a module puts on PATH and the paths it sets, from what
    `module show` prints: Lmod's Lua form or the Tcl form."""

    paths = re.findall(r'(?:prepend|append)_path\(\s*"PATH"\s*,\s*"([^"]+)"', text)
    paths += re.findall(r"^\s*(?:prepend|append)-path\s+PATH\s+(\S+)", text, re.M)
    paths += re.findall(r'setenv\(\s*"\w+"\s*,\s*"(/[^"]+)"', text)
    paths += re.findall(r"^\s*setenv\s+\w+\s+(/\S+)", text, re.M)
    return paths


def in_module(name: str) -> list[str]:
    return module_paths(run(["bash", "-lc", f"module show {shlex.quote(name)} 2>&1"],
                            seconds=20.0).output)


def tool_value(tool: str, path: Path) -> Path:
    """What a tool's variable names: IMOD's folder, pytom-match-pick's bin
    folder, or the program itself."""

    if tool == "imod":
        return path.parent.parent if path.name == "newstack" else path
    if tool == "pytom-match-pick":
        return path.parent
    return path


def candidate(tool: str, path: Path, how: str) -> dict[str, str]:
    """A copy of a tool found elsewhere: its version, and whether a lane can use
    it as it is."""

    asked = tool_value(tool, path) if tool == "imod" else path
    version = ask_version(tool, asked)
    good, why = "yes", ""
    if tool == "imod" and not (asked / "VERSION").is_file():
        good, why = "no", "no VERSION file: not an IMOD folder"
    elif tool == "ctffind" and version.value and version_tuple(version.value)[0] < 5:
        good, why = "no", "CTFFIND 4; estimate-ctf needs CTFFIND 5 (C-2)"
    found = {"path": str(tool_value(tool, path)), "how": how, "version": version.value or "",
             "good": good, "why": why}
    if tool in ("motioncor3", "aretomo"):
        code = gpu_code_of(path, NAMES[tool])
        if code.cubins or code.ptx:
            found["gpu"] = coverage(code)
    return found


def rank(found: dict[str, str], tool: str) -> tuple:
    return (found["good"] == "yes", found["version"] == TESTED[tool],
            version_tuple(found["version"]) if found["version"] else ())


def find_elsewhere(report: Report, extra: list[Path]) -> None:
    """--find: each tool's copies in the environments, the environment modules and
    the software folders, apart from the one this shell names."""

    places_of: dict[str, dict[str, tuple[Path, str]]] = {
        tool: {} for tool in (*TOOLS, "membrain-ckpt")}

    def add(tool: str, path: Path, how: str) -> None:
        places_of[tool].setdefault(os.path.realpath(path), (path, how))

    for tool, path, how in env_programs():
        add(tool, path, how)
    names = module_names()
    for tool, pattern in MODULE_NAMES.items():
        matching = [name for name in names if re.search(pattern, name, re.I)]
        if matching:
            report.modules[tool] = matching
        if tool == "cuda":
            continue
        for name in matching[:3]:
            for value in in_module(name):
                folder = Path(value)
                if folder.is_file() and is_program(tool, folder.name):
                    add(tool, folder, f"module {name}")
                    continue
                if tool == "imod" and (folder / "VERSION").is_file():
                    add(tool, folder / "bin" / "newstack", f"module {name}")
                try:
                    entries = sorted(os.listdir(folder))
                except OSError:
                    continue
                for entry in entries:
                    if is_program(tool, entry) and os.access(folder / entry, os.X_OK):
                        add(tool, folder / entry, f"module {name}")
    searched, stopped = [], []
    for root in search_folders(extra):
        found, ran_out = walk_programs(root)
        searched.append(str(root))
        if ran_out:
            stopped.append(str(root))
        for tool, path in found:
            add(tool, path, f"under {root}")
    report.gpu["searched"] = {"folders": searched, "stopped": stopped}

    for tool, tool_report in zip(TOOLS, report.tools):
        current = os.path.realpath(tool_report.path) if tool_report.path else ""
        copies = [candidate(tool, path, how) for real, (path, how) in places_of[tool].items()
                  if real != current and os.path.realpath(tool_value(tool, path)) != current]
        tool_report.elsewhere = sorted(copies, key=lambda found: rank(found, tool), reverse=True)[:5]
        best = tool_report.elsewhere[0] if tool_report.elsewhere else None
        if not tool_report.usable and best and best["good"] == "yes":
            report.note(f"{NAMES[tool]} {best['version']} is at {best['path']} ({best['how']}): name "
                        f"it with export {VARIABLES[tool]}={best['path']} in ~/.bashrc, or in a setup "
                        "file that ~/.bashrc sources (--setup-file proposes one)", "E-9")
    checkpoints = [str(path) for path, _ in places_of["membrain-ckpt"].values()]
    report.gpu["checkpoints"] = sorted(checkpoints)[:5]


def setup_file(report: Report) -> str:
    """A proposed pipeline_setup.sh: the variables the jobs need, from what this
    check found. Variables this shell sets already are left to whatever sets them."""

    host = report.host
    lines = [
        "# pipeline_setup.sh: what the pipeline and its jobs need from your shell.",
        f"# Proposed by scripts/show_setup.py --setup-file on {host['name']}, {host['date'][:10]}.",
        "# Check every path. Then source it from ~/.bashrc, after micromamba's own",
        "# block and outside any check for the host name:",
        "#     source /path/to/pipeline_setup.sh",
        "# and open a new terminal. Jobs take these variables from the shell that",
        "# submits them.",
        "",
        "# Room: the environments, the downloads and the temporary files go to your",
        "# folder on a large filesystem (docs/agent/setup.md, 'Room').",
    ]
    work = os.environ.get("CRYOET_WORK")
    if work:
        lines.append(f"# CRYOET_WORK is set in your shell already: {work}")
    else:
        lines.append("# export CRYOET_WORK=/path/to/your/folder    # name it, then uncomment the lines below")
    prefix = "" if work else "# "
    for variable, value, why in (
        ("MAMBA_ROOT_PREFIX", "$CRYOET_WORK/micromamba", "environments and their downloads"),
        ("PIP_CACHE_DIR", "$CRYOET_WORK/cache/pip", "pip's downloads"),
        ("TMPDIR", "$CRYOET_WORK/tmp", "pip unpacks its downloads here"),
    ):
        current = os.environ.get(variable)
        if current and not (variable == "TMPDIR" and any(
                f.entry == "E-1" for f in report.findings)):
            lines.append(f"# {variable} is set in your shell already: {current}")
        else:
            lines.append(f"{prefix}export {variable}={value}    # {why}")
    lines.append(f'{prefix}mkdir -p "$TMPDIR"    # Python passes over a TMPDIR that does not exist')
    manager = report.environment.get("manager", "")
    if not os.environ.get("ENV_MANAGER") and manager.startswith("micromamba"):
        lines.append("export ENV_MANAGER=micromamba    # jobs then never take a system conda (I-7)")
    lines += ["", "# The tools, by the variables the lanes name them by."]
    for tool, tool_report in zip(TOOLS, report.tools):
        variable = VARIABLES[tool]
        current = os.environ.get(variable)
        label = f"{NAMES[tool]} {tool_report.version}".strip()
        if current and tool_report.usable:
            lines.append(f"# {variable} is set in your shell already: {current} ({label})")
        elif tool_report.usable and tool_report.path:
            value = str(tool_value(tool, Path(tool_report.path)))
            lines.append(f"export {variable}={shlex.quote(value)}    # {label}, {tool_report.found}")
        elif tool_report.elsewhere and tool_report.elsewhere[0]["good"] == "yes":
            best = tool_report.elsewhere[0]
            lines.append(f"export {variable}={shlex.quote(best['path'])}    "
                         f"# {NAMES[tool]} {best['version']}, {best['how']}")
        else:
            lines.append(f"# export {variable}=    # {NAMES[tool]}: not found here; "
                         f"needed by {NEEDED_BY[tool]}")
        for line in tool_report.lines:
            if line.startswith("libraries not found"):
                lines.append(f"#   {NAMES[tool]}: {line}; add their folder:")
                lines.append('#   export LD_LIBRARY_PATH=/folder/with/them:"$LD_LIBRARY_PATH"')
    if os.environ.get("MEMBRAIN_CKPT"):
        lines.append(f"# MEMBRAIN_CKPT is set in your shell already: {os.environ['MEMBRAIN_CKPT']}")
    elif report.gpu.get("checkpoints"):
        lines.append(f"export MEMBRAIN_CKPT={shlex.quote(report.gpu['checkpoints'][0])}    "
                     "# MemBrain-seg's model")
    else:
        lines.append("# export MEMBRAIN_CKPT=    # MemBrain-seg's model checkpoint: not found here")
    if report.modules.get("cuda"):
        lines += ["", "# CUDA toolkits offered as modules here, for building MotionCor3 or AreTomo2:",
                  f"#   {', '.join(report.modules['cuda'])}"]
    return "\n".join(lines) + "\n"


# --- what can run here ------------------------------------------------------------------


def lane_groups(lanes: list[Lane]) -> set[str]:
    """The step groups the given lanes run."""

    names: set[str] = set()
    for lane in lanes:
        cfg = lane.cfg
        frames = (cfg.get("lane") or {}).get("input", "frames") == "frames"
        names.add("motion correction" if frames else "stack import")
        names.update({"alignment", "CTF", "tomogram", "template", "picking", "extraction"})
        if (cfg.get("fine_align") or {}).get("backend") == "aretomo":
            names.add("fine alignment with AreTomo2")
        if "segment" in cfg:
            names.add("membranes")
        if "mra" in cfg:
            names.add("MRA")
    return names


def judge(group: StepGroup, ready: bool, tools_ready: dict[str, bool], code: GpuCode | None,
          here: list[Gpu], cluster: dict[str, str], on_cluster: bool, driver: str,
          pci: list[dict[str, str]]) -> tuple[str, str]:
    """Whether a step group can run: "yes", "cpu" (on the CPU only), "no" or "?",
    and why. GPU code that is not known (CuPy compiles its kernels when run)
    counts as fitting every GPU."""

    if not ready:
        return "no", "the pipeline's environment is not ready (part 2)"
    missing = [NAMES[tool] for tool in group.tools if not tools_ready.get(tool)]
    if missing:
        return "no", f"{' and '.join(missing)} not ready (part 3)"
    if group.gpu == "none":
        return "yes", ""

    def fits(capability: str) -> bool:
        return code is None or not (code.cubins or code.ptx) or usable(capability, code)

    if code is not None and code.cpu_only:
        reason = f"{code.holder} is a CPU-only build"
    elif on_cluster:
        if not cluster:
            return "?", "sinfo lists no GPU type to check against"
        yes = [name for name, capability in sorted(cluster.items()) if fits(capability)]
        if yes:
            return "yes", "on the GPU nodes: " + ", ".join(yes)
        reason = "its GPU code fits none of the cluster's GPUs (" + ", ".join(sorted(cluster)) + ")"
    elif here:
        if code is not None and code.cuda and driver and cuda_fit(code.cuda, driver) == "no":
            reason = f"it is built for CUDA {code.cuda}, and the driver here supports {driver}"
        else:
            yes = sorted({gpu.name for gpu in here if fits(gpu.capability)})
            if yes:
                return "yes", "on the GPU here: " + ", ".join(yes)
            reason = "its GPU code fits none of the GPUs here"
    else:
        reason = ("the NVIDIA GPU here has no driver" if pci
                  else "this machine has no NVIDIA GPU")
    if group.gpu == "optional":
        return "cpu", f"on the CPU, slower: {reason}"
    if group.gpu == "advised":
        return "cpu", f"on the CPU only, untested and slow: {reason}"
    return "no", f"needs an NVIDIA GPU, and {reason}"


def steps_map(report: Report, lanes: list[Lane], codes: list[GpuCode]) -> None:
    env = report.environment
    packages = env.get("packages") or {}
    ready = bool(packages.get("cryoet-pipeline") and packages.get("torch")) and not (
        env.get("torch") or {}).get("error")
    tools_ready = {tool: tool_report.usable for tool, tool_report in zip(TOOLS, report.tools)}
    holders = {code.holder: code for code in codes}
    pipeline_torch = env.get("torch") or {}
    if pipeline_torch.get("version") and not pipeline_torch.get("cuda"):
        holders["the pipeline's PyTorch"] = GpuCode("the pipeline's PyTorch", cpu_only=True)
    cupy = report.gpu.get("cupy") or {}
    holders["pytom-match-pick's CuPy"] = GpuCode("pytom-match-pick's CuPy",
                                                 cuda=cuda_text(cupy.get("runtime")))
    here = [Gpu(**gpu) for gpu in report.gpu.get("gpus", [])]
    cluster = {name: capability for name, capability in (report.gpu.get("cluster") or {}).items()
               if capability}
    on_cluster = bool(report.gpu.get("slurm"))
    report.gpu["mode"] = ("on this cluster's nodes, through Slurm" if on_cluster
                          else "on this machine itself (no Slurm here)")
    wanted = lane_groups(lanes) if lanes else None
    for group in STEP_GROUPS:
        code = holders.get(group.code or ("pytom-match-pick's CuPy" if group.name == "picking"
                                          else ""))
        status, why = judge(group, ready, tools_ready, code, here, cluster, on_cluster,
                            report.gpu.get("driver_cuda", ""), report.gpu.get("pci", []))
        report.steps.append({"name": group.name, "steps": group.steps, "lanes": group.lanes,
                             "status": status, "why": why,
                             "in_lane": "" if wanted is None else
                             ("yes" if group.name in wanted else "no")})


# --- the host and the clients --------------------------------------------------------


def host(report: Report) -> None:
    name = socket.gethostname()
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        where = f"inside Slurm job {job}"
    elif "login" in name:
        where = "a login node, by its name"
    else:
        where = "not inside a Slurm job"
    report.host = {
        "name": name,
        "system": f"{platform.system()} {platform.release()}",
        "where": where,
        "user": getpass.getuser(),
        "python": f"{sys.executable} {platform.python_version()}",
        "date": datetime.now().isoformat(timespec="seconds"),
        "clone": str(ROOT),
    }


def clients(report: Report) -> None:
    for program, name in (("claude", "Claude Code"), ("codex", "Codex")):
        found = shutil.which(program)
        if found is None:
            report.clients.append(f"{name}: not on PATH")
            continue
        result = run([found, "--version"], seconds=30.0)
        version = re.search(r"\d+\.\d+\.\d+", result.output)
        report.clients.append(f"{name}: {version.group(0) if version else 'version unknown'} "
                              f"({found})")


# --- the report --------------------------------------------------------------------


def build(args: argparse.Namespace) -> Report:
    report = Report()
    host(report)
    clone_state(report)
    places(report, [Path(folder) for folder in args.folder], args.sizes)
    quotas(report)
    environment(report, args.env)
    lanes = read_lanes([Path(path) for path in args.config], report)
    code = tools(report, lanes)
    if args.find or args.setup_file:
        find_elsewhere(report, [Path(folder) for folder in args.search])
    clients(report)
    code = gpu_and_cuda(report, code)
    steps_map(report, lanes, code)
    return report


def render(report: Report) -> str:
    out: list[str] = []
    h = report.host
    out.append(f"{h['name']} ({h['system']}), {h['where']}, user {h['user']}, {h['date']}")
    out.append(f"this check runs under {h['python']}, from {h['clone']}")
    clone = report.clone
    if clone.get("head"):
        changed = clone.get("changed", 0)
        out.append(f"the clone: {clone.get('branch')} at {clone['head']} ({clone.get('date')})"
                   + (f", {changed} files changed" if changed else "")
                   + f"; {clone.get('state')}")
    elif clone.get("state"):
        out.append(f"the clone: {clone['state']}")

    out.append("\n1. Room")
    out.append(f"  {'place':<20} {'free':>9} {'of':>9}  {'filesystem':<24} path")
    for place in report.places:
        filesystem = place.mount + (f" ({place.fstype})" if place.fstype else "")
        path = place.path + ("" if place.exists else "  (not there yet)")
        out.append(f"  {place.label:<20} {size_text(place.free):>9} {size_text(place.total):>9}  "
                   f"{filesystem:<24} {path}")
        out.append(f"  {'':<20} {place.why}"
                   + (f"; holds {size_text(place.size)}" if place.size is not None else "")
                   + (f"; {place.size_note}" if place.size_note else ""))
    out.append("  'free' is the filesystem's. Your quota can be smaller; the site's tool says:")
    if not report.quota:
        out.append("    no quota tool here (quota, mmlsquota, lfs): ask the site how quotas are shown")
    for entry in report.quota:
        out.append(f"    $ {entry['command']}")
        out.extend(f"      {line}" for line in entry["output"].splitlines())

    env = report.environment
    out.append("\n2. The pipeline's environment")
    if env.get("path"):
        out.append(f"  environment  {env['path']} ({env['how']})")
    if env.get("manager"):
        out.append(f"  activation   a job activates it with {env['manager']} "
                   "(scripts/activate_env.sh)")
    if env.get("python"):
        origin = env.get("pipeline")
        out.append(f"  Python       {env['python']}")
        out.append("  pipeline     " + ("not installed" if not origin else
                                       f"{Path(origin).parent}"
                                       + (" (this clone)" if env.get("from_this_clone") else
                                          " (NOT this clone)")))
        packages = env.get("packages", {})
        have = [f"{name} {packages[name]}" for name, _ in PACKAGES if packages.get(name)]
        lack = [f"{name} ({extra})" for name, extra in PACKAGES if not packages.get(name)]
        out.append("  packages     " + (", ".join(have) or "none"))
        if lack:
            out.append("  missing      " + ", ".join(lack))
        if env.get("torch_line"):
            out.append(f"  PyTorch      {env['torch_line']}")
    elif env.get("error"):
        out.append(f"  {env['error']}")
    else:
        out.append("  none found")

    out.append("\n3. Tools")
    for tool in report.tools:
        version = tool.version or "?"
        out.append(f"  {tool.name:<17} {version:<16} tested {tool.tested:<16} "
                   f"{'needed' if tool.needed else 'optional':<8} {tool.found}")
        if tool.path:
            out.append(f"  {'':<17} {tool.path}")
        out.extend(f"  {'':<17} {line}" for line in tool.lines)
        for copy in tool.elsewhere:
            out.append(f"  {'':<17} elsewhere: {copy['path']}, {copy['version'] or 'version ?'}, "
                       f"{copy['how']}" + (f"; {copy['why']}" if copy["why"] else "")
                       + (f"; {copy['gpu']}" if copy.get("gpu") else ""))
    out.extend(f"  {line}" for line in report.clients)
    for tool, names in report.modules.items():
        out.append(f"  modules for {NAMES.get(tool, tool.upper())}: {', '.join(names)}")
    searched = report.gpu.get("searched")
    if searched:
        out.append("  searched: " + (", ".join(searched["folders"]) or "no folder")
                   + (f" (stopped after {SEARCH_SECONDS:.0f} s in "
                      f"{', '.join(searched['stopped'])})" if searched["stopped"] else ""))
        if report.gpu.get("checkpoints"):
            out.append("  MemBrain-seg models: " + ", ".join(report.gpu["checkpoints"]))

    gpu = report.gpu
    out.append("\n4. GPU and CUDA")
    if gpu.get("here"):
        out.append(f"  this node    {gpu['here']}")
    for card in gpu.get("pci", []):
        out.append(f"  PCI {card['slot']}  {card['name']}, driver "
                   + (card["driver"] or "none bound"))
    for item in gpu.get("gpus", []):
        out.append(f"  GPU {item['index']}        {item['name']}, compute capability "
                   f"{item['capability']}, {item['memory']}, driver {item['driver']}")
    if any("MIG" in line for line in gpu.get("listing", [])):
        out.extend(f"               {line}" for line in gpu["listing"])
    if gpu.get("driver_cuda"):
        out.append(f"  driver       supports CUDA {gpu['driver_cuda']} at most")
    for name, value in gpu.get("visible", {}).items():
        out.append(f"  {name}={value}")
    if gpu.get("nvcc"):
        out.append(f"  nvcc         {gpu['nvcc']}")
    for name, value in gpu.get("cuda_vars", {}).items():
        out.append(f"  {name:<12} {value}")
    if gpu.get("cuda_libraries"):
        out.append("  LD_LIBRARY_PATH, CUDA folders: " + " ".join(gpu["cuda_libraries"]))
    if gpu.get("cluster"):
        out.append("  the cluster's GPU types (sinfo): " + ", ".join(
            f"{name} {capability or '?'}" for name, capability in sorted(gpu["cluster"].items())))
    for line in gpu.get("builds", []):
        out.append(f"  built for    {line}")
    for line in gpu.get("fits", []):
        out.append(f"  GPU code     {line}")

    out.append(f"\n5. What can run here: {gpu.get('mode', '')}")
    marks = {"yes": "yes", "cpu": "CPU", "no": "no", "?": "?"}
    for step in report.steps:
        if step["in_lane"] == "no":
            continue
        which = f"only lanes with {step['lanes']}" if step["lanes"] and not step["in_lane"] else ""
        why = "; ".join(part for part in (step["why"], which) if part)
        out.append(f"  {step['name']:<29} {marks[step['status']]:<4} {why}")
    runs = [step["name"] for step in report.steps
            if step["status"] in ("yes", "cpu") and step["in_lane"] != "no"]
    not_here = [step["name"] for step in report.steps
                if step["status"] == "no" and step["in_lane"] != "no"]
    out.append("  runs here: " + (", ".join(runs) or "nothing"))
    if not_here:
        out.append("  does not run here: " + ", ".join(not_here))

    problems = [finding for finding in report.findings if finding.problem]
    notes = [finding for finding in report.findings if not finding.problem]
    out.append(f"\nProblems ({len(problems)})" + ("" if problems else ": none"))
    out.extend(f"- {finding.text}" + (f" ({finding.entry})" if finding.entry else "")
               for finding in problems)
    out.append(f"\nNotes ({len(notes)})" + ("" if notes else ": none"))
    out.extend(f"- {finding.text}" + (f" ({finding.entry})" if finding.entry else "")
               for finding in notes)
    out.append("\nEntries are in docs/agent/playbook/: E- in cluster.md, I- in install.md, "
               "C- in ctf.md, M- in motion.md, G- in membranes.md, P- in picking.md.")
    return "\n".join(out)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", help="the environment to ask: a name or a folder")
    parser.add_argument("--config", action="append", default=[],
                        help="a lane config: find the tools as it names them, and need what it needs")
    parser.add_argument("--folder", action="append", default=[],
                        help="another folder to show the room of, such as a data folder")
    parser.add_argument("--sizes", action="store_true",
                        help="also measure what each place holds (du; up to a minute each)")
    parser.add_argument("--find", action="store_true",
                        help="also look for each tool where this shell does not name it: other "
                             "environments, environment modules, software folders (a minute or two)")
    parser.add_argument("--search", action="append", default=[],
                        help="another folder for --find to search, such as a group's software folder")
    parser.add_argument("--setup-file", action="store_true",
                        help="print a proposed pipeline_setup.sh instead of the report (implies --find)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in args.config:
        if not Path(path).is_file():
            print(f"--config {path}: no such file", file=sys.stderr)
            return 2
    for folder in args.search:
        if not Path(folder).is_dir():
            print(f"--search {folder}: no such folder", file=sys.stderr)
            return 2
    report = build(args)
    if args.setup_file:
        print(setup_file(report), end="")
        return 0
    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        print(render(report))
    return 1 if any(finding.problem for finding in report.findings) else 0


if __name__ == "__main__":
    sys.exit(main())
