"""scripts/show_setup.py: room, versions, GPU and CUDA, on a site made of stand-ins.

The stand-ins are small bash scripts: CTFFIND, MotionCor3, ldd, cuobjdump,
nvidia-smi and sinfo print what the real ones would, and an environment's
"python" answers the check's questions from files beside it.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_library  # noqa: E402
import show_setup  # noqa: E402

TAG = show_setup.PROBE_TAG


def _program(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\n" + text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# An environment's Python: it answers the check's probe scripts from the files
# beside it, and a package-version question from version-<distribution>.
PYTHON = """\
here=$(dirname "$0")
case "$3" in
  *"import cupy"*) cat "$here/cupy.json" ;;
  *"find_spec"*) cat "$here/env.json" ;;
  *"m.version(sys.argv[1])"*) cat "$here/version-$4" 2>/dev/null || exit 3 ;;
  *) exit 9 ;;
esac
"""


def _python(folder: Path, answers: dict[str, object], versions: dict[str, str] = {}) -> Path:
    python = _program(folder / "python", PYTHON)
    for name, answer in answers.items():
        (folder / f"{name}.json").write_text(f"some warning on the way\n{TAG}{json.dumps(answer)}\n")
    for name, version in versions.items():
        (folder / f"version-{name}").write_text(version + "\n")
    return python


A40 = "0, NVIDIA A40, 8.6, 46068 MiB, 610.57.04"
V100 = "0, Tesla V100-SXM2-32GB, 7.0, 32768 MiB, 610.57.04"


# nvidia-smi's header: driver 610.57.04 names the driver KMD and its CUDA UMD (an A40
# node of max, 17 November 2026); older drivers, such as Munich's 575, say Driver and CUDA.
KMD_HEADER = "| NVIDIA-SMI {driver}              KMD Version: {driver}     CUDA UMD Version: {cuda}     |"
DRIVER_HEADER = "| NVIDIA-SMI {driver}   Driver Version: {driver}   CUDA Version: {cuda} |"


def _nvidia_smi(folder: Path, gpu: str = A40, cuda: str = "13.3", mig: bool = False,
                header: str = KMD_HEADER, driver: str = "610.57.04") -> None:
    listing = "GPU 0: NVIDIA A40 (UUID: GPU-1)"
    if mig:
        listing += "\n  MIG 3g.20gb     Device  0: (UUID: MIG-1)"
    _program(folder / "nvidia-smi", f"""\
case "$1" in
  --query-gpu=*) echo "{gpu}" ;;
  -L) printf '%s\\n' "{listing}" ;;
  *) echo "{header.format(driver=driver, cuda=cuda)}" ;;
esac
""")


PIPELINE_TORCH = {
    "version": "2.15.0", "cuda": "13.2",
    "archs": ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"],
    "available": True, "devices": [["NVIDIA A40", "8.6"]],
}


def _env_answer(**changes) -> dict:
    answer = {
        "python": "3.13.5", "executable": "python",
        "packages": {name: "1.0" for name, _ in show_setup.PACKAGES},
        "pipeline": str(ROOT / "src" / "cryoet_pipeline" / "__init__.py"),
        "torch": dict(PIPELINE_TORCH),
    }
    answer.update(changes)
    return answer


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A GPU node with the pipeline's environment, the group's tools and Slurm,
    each a stand-in. Returns its folders, to change before the check runs."""

    bin_ = tmp_path / "bin"
    home = tmp_path / "home"
    home.mkdir()
    temp = tmp_path / "tmp"
    temp.mkdir()
    root = tmp_path / "micromamba"
    env_bin = root / "envs" / "cryoet-pipeline" / "bin"
    env_bin.mkdir(parents=True)
    _python(env_bin, {"env": _env_answer()})

    _program(bin_ / "ctffind", 'echo " Version : 4.1.14"\n')
    motioncor3 = _program(bin_ / "MotionCor3", 'echo "MotionCor3 version 1.2.4"\n')
    _program(bin_ / "ldd", 'printf "\\tlibcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12\\n'
                           '\\tlibtiff.so.6 => not found\\n"\n')
    _program(bin_ / "cuobjdump", '[[ "$1" == -lelf ]] && echo "ELF file    1: MotionCor3.1.sm_86.cubin"\n'
                                 'exit 0\n')
    _nvidia_smi(bin_)
    _program(bin_ / "sinfo", 'if [[ "$1" == -V ]]; then echo "slurm 26.05.4"; else\n'
                             'echo "gpu|gpu:a40:8(S:0-1),gpu:tesla-v100:4(S:0)"\n'
                             'echo "normal|(null)"; fi\n')

    imod = tmp_path / "imod"
    (imod / "bin").mkdir(parents=True)
    (imod / "bin" / "newstack").write_text("")
    (imod / "VERSION").write_text("5.1.12\n")

    pytom_bin = tmp_path / "pytom_tm" / "bin"
    python = _python(pytom_bin, {"cupy": {"python": "3.12.8", "cupy": {
        "version": "14.2.0", "runtime": 12090, "nvrtc": "12.9", "cuda_path": "/env/targets",
        "devices": 1, "square_sum": {"cub": 10500000.0, "plain": 10500000.0}}}},
        {"pytom-match-pick": "0.14.0"})
    for script in show_setup.PYTOM_SCRIPTS:
        (pytom_bin / f"{script}.py").write_text(f"#!{python}\n")

    membrain_bin = tmp_path / "membrain" / "bin"
    membrain_python = _python(membrain_bin, {"env": {"python": "3.10.0", "packages": {
        "membrain-seg": "0.0.10", "torch": "2.1.0"}, "torch": {
        "version": "2.1.0", "cuda": "12.1", "archs": ["sm_75", "sm_80", "sm_86", "sm_90"],
        "available": True, "devices": [["NVIDIA A40", "8.6"]]}}}, {"membrain-seg": "0.0.10"})
    (membrain_bin / "membrain").write_text(f"#!{membrain_python}\n")
    (membrain_bin / "membrain").chmod(0o755)
    checkpoint = tmp_path / "membrain.ckpt"
    checkpoint.write_text("")

    for name in list(os.environ):
        if name.startswith(("CONDA", "MAMBA", "CUDA", "SLURM", "PIP_", "XDG_")) or name in (
                "CTFFIND5", "MOTIONCOR3", "ARETOMO", "PYTOMO_MP_DIR", "MEMBRAIN_EXE",
                "MEMBRAIN_CKPT", "IMOD_DIR", "ENV_MANAGER", "CRYOET_WORK", "TEMP", "TMP",
                "LD_LIBRARY_PATH"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PATH", f"{bin_}:/usr/bin:/bin")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("TMPDIR", str(temp))
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(root))
    monkeypatch.setenv("ENV_MANAGER", "micromamba")
    monkeypatch.setenv("MOTIONCOR3", str(motioncor3))
    monkeypatch.setenv("IMOD_DIR", str(imod))
    monkeypatch.setenv("PYTOMO_MP_DIR", str(pytom_bin))
    monkeypatch.setenv("MEMBRAIN_EXE", str(membrain_bin / "membrain"))
    monkeypatch.setenv("MEMBRAIN_CKPT", str(checkpoint))
    monkeypatch.setattr(show_setup, "MAC_IMOD", tmp_path / "no-imod")
    # No libcuda on the test host, or on a login node: the banner then says.
    monkeypatch.setattr(show_setup, "driver_cuda", lambda: "")
    # The clone's own state has tests of its own, on repositories made for them;
    # here it would ask GitHub every time.
    monkeypatch.setattr(show_setup, "clone_state", lambda report, root=None: None)
    monkeypatch.setattr(show_setup, "SYS_PCI", tmp_path / "no-pci")
    monkeypatch.setattr(show_setup, "SEARCH_ROOTS", ())
    monkeypatch.setattr(show_setup, "module_names", lambda: [])
    return {"tmp": tmp_path, "bin": bin_, "env_bin": env_bin, "temp": temp, "pytom": pytom_bin,
            "membrain": membrain_bin, "root": root}


def _check(*argv: str) -> show_setup.Report:
    return show_setup.build(show_setup.parse_args(list(argv)))


def _findings(report: show_setup.Report, problem: bool) -> list[tuple[str, str]]:
    return [(finding.entry, finding.text) for finding in report.findings
            if finding.problem is problem]


def _entries(report: show_setup.Report, problem: bool = True) -> set[str]:
    return {entry for entry, _ in _findings(report, problem)}


# --- the whole check on the stand-in site ----------------------------------------


def test_a_gpu_node_with_ctffind_4_and_a_library_missing(site):
    report = _check()
    assert _entries(report) == {"C-2", "M-5"}, _findings(report, True)
    tools = {tool.name: tool for tool in report.tools}
    assert tools["CTFFIND"].version == "4.1.14" and tools["CTFFIND"].found == "ctffind on PATH"
    assert tools["IMOD"].version == "5.1.12" and tools["IMOD"].found == "$IMOD_DIR"
    assert tools["pytom-match-pick"].version == "0.14.0"
    assert tools["MemBrain-seg"].version == "0.0.10"
    motioncor3 = "\n".join(tools["MotionCor3"].lines)
    assert "libtiff.so.6" in motioncor3 and "linked against CUDA 12" in motioncor3
    assert "cubins sm_86; PTX none" in motioncor3
    gpu = report.gpu
    assert gpu["driver_cuda"] == "13.3"
    assert gpu["cluster"] == {"a40": "8.6", "tesla-v100": "7.0"}
    fits = "\n".join(gpu["fits"])
    assert "MotionCor3: runs on NVIDIA A40 here, a40; not on tesla-v100" in fits
    assert "the pipeline's PyTorch: runs on NVIDIA A40 here, a40; not on tesla-v100" in fits
    assert report.environment["from_this_clone"] is True
    text = show_setup.render(report)
    for heading in ("1. Room", "2. The pipeline's environment", "3. Tools", "4. GPU and CUDA",
                    "Problems (2)"):
        assert heading in text


def test_the_check_writes_nothing(site):
    before = sorted((path, path.stat().st_mtime_ns) for path in site["tmp"].rglob("*"))
    assert show_setup.main([]) == 1
    assert sorted((path, path.stat().st_mtime_ns) for path in site["tmp"].rglob("*")) == before


def test_json_holds_the_same_report(site, capsys):
    show_setup.main(["--json"])
    report = json.loads(capsys.readouterr().out)
    assert {finding["entry"] for finding in report["findings"] if finding["problem"]} == {"C-2", "M-5"}
    assert [tool["name"] for tool in report["tools"]] == [
        "IMOD", "CTFFIND", "MotionCor3", "AreTomo", "pytom-match-pick", "MemBrain-seg"]


def test_a_build_for_a_newer_cuda_than_the_driver_is_a_problem(site):
    _nvidia_smi(site["bin"], cuda="12.9", header=DRIVER_HEADER, driver="575.57.08")
    problems = _findings(_check(), True)
    (text,) = [text for entry, text in problems if entry == "I-6"]
    assert "the pipeline's PyTorch is built for CUDA 13.2" in text
    assert "CUDA 13 needs a driver 580 or later" in text


def test_a_newer_minor_version_than_the_driver_is_a_note(site):
    _nvidia_smi(site["bin"], cuda="13.0")
    report = _check()
    assert "I-6" not in _entries(report)
    assert any("minor version compatibility" in text for _, text in _findings(report, False))


def test_a_gpu_the_code_has_no_kernel_for(site):
    # A V100 here: MotionCor3 has sm_86 only, and both PyTorch builds start at sm_75.
    _nvidia_smi(site["bin"], gpu=V100)
    entries = _entries(_check())
    assert {"M-1", "G-1", "I-6"} <= entries


def test_membrain_seg_below_compute_capability_7_5_whatever_its_pytorch_has(site):
    # A T4-class card would do; a 7.0 card with code for it in PyTorch is refused
    # by the job's preflight all the same.
    _nvidia_smi(site["bin"], gpu=V100)
    _python(site["membrain"], {"env": {"python": "3.10.0", "packages": {
        "membrain-seg": "0.0.10", "torch": "2.1.0"}, "torch": {
        "version": "2.1.0", "cuda": "12.1", "archs": ["sm_70", "sm_75", "sm_80", "sm_86"],
        "available": True, "devices": [["Tesla V100-SXM2-32GB", "7.0"]]}}}, {"membrain-seg": "0.0.10"})
    report = _check()
    (text,) = [text for entry, text in _findings(report, True) if entry == "G-1"]
    assert "needs a GPU of compute capability 7.5 or later" in text
    assert "MemBrain-seg's PyTorch: runs on a40; not on Tesla V100-SXM2-32GB here, tesla-v100" in (
        report.gpu["fits"])


def test_pytorch_that_does_not_see_the_gpu_of_its_node(site):
    torch = dict(PIPELINE_TORCH, available=False)
    torch.pop("devices")
    _python(site["env_bin"], {"env": _env_answer(torch=torch)})
    problems = _findings(_check(), True)
    assert ("I-6", "this node has a GPU and the pipeline's PyTorch does not see it") in problems


def test_a_login_node_leaves_the_driver_question_to_a_gpu_node(site):
    (site["bin"] / "nvidia-smi").unlink()
    torch = dict(PIPELINE_TORCH, available=False)
    torch.pop("devices")
    _python(site["env_bin"], {"env": _env_answer(torch=torch)})
    report = _check()
    assert "I-6" not in _entries(report)
    assert "no NVIDIA driver on this node" in report.gpu["here"]
    (text,) = [text for entry, text in _findings(report, False) if entry == "I-11"]
    assert "the pipeline's PyTorch (CUDA 13.2)" in text and "through srun" in text
    # The cluster's GPU types are still checked against the code.
    assert any("not on tesla-v100" in line for line in report.gpu["fits"])


def test_mig_slices_are_named(site):
    _nvidia_smi(site["bin"], mig=True)
    assert "P-1" in _entries(_check(), problem=False)


def test_cupy_compiling_with_another_cuda_than_it_was_built_for(site):
    _python(site["pytom"], {"cupy": {"python": "3.12.8", "cupy": {
        "version": "14.2.0", "runtime": 12090, "nvrtc": "13.3", "devices": 1}}},
        {"pytom-match-pick": "0.14.0"})
    notes = _findings(_check(), False)
    assert any(entry == "I-9" and "built for CUDA 12.9 but compiles its kernels with NVRTC 13.3"
               in text for entry, text in notes)


# CuPy's compile error on cryoem.se, 25 November 2026, as the probe shortens it.
CUB_FAILS = ("CompileException: /env/include/cuda/std/__floating_point/nvfp_types.h(30): "
             "error: incomplete type \"__nv_fp8_e8m0\" is not allowed ... 42 errors detected "
             "in the compilation of \"/tmp/tmppjbq0mc6/4edd3527.cubin.cu\".")


def _square_sum(site, square_sum: dict, setting: str | None = None, devices: int = 1) -> list:
    facts = {"version": "14.2.0", "runtime": 12090, "nvrtc": "12.9", "devices": devices}
    if square_sum is not None:
        facts["square_sum"] = square_sum
        facts["accelerators_env"] = setting
    _python(site["pytom"], {"cupy": {"python": "3.12.8", "cupy": facts}},
            {"pytom-match-pick": "0.14.0"})
    report = _check()
    lines = {tool.name: tool for tool in report.tools}["pytom-match-pick"].lines
    return [report, "\n".join(lines)]


def test_a_reduction_that_compiles_is_said_and_raises_nothing(site):
    report, lines = _square_sum(site, {"cub": 10500000.0, "plain": 10500000.0})
    assert "with CuPy's CUB: compiles and sums right" in lines
    assert "without CUB: compiles and sums right" in lines
    assert "I-9" not in _entries(report) | _entries(report, problem=False)


def test_a_reduction_that_fails_with_cub_is_a_problem_while_cub_is_on(site):
    # The versions agree, as on gpu54: only the compile shows it.
    for setting in (None, "cub"):
        report, lines = _square_sum(site, {"cub_error": CUB_FAILS, "plain": 10500000.0}, setting)
        assert "__nv_fp8_e8m0" in lines and "without CUB: compiles and sums right" in lines
        (text,) = [text for entry, text in _findings(report, True) if entry == "I-9"]
        assert "export CUPY_ACCELERATORS=" in text


def test_a_shell_that_turns_cub_off_makes_it_a_note(site):
    for setting in ("", "cutensor"):
        report, _ = _square_sum(site, {"cub_error": CUB_FAILS, "plain": 10500000.0}, setting)
        assert "I-9" not in _entries(report)
        (text,) = [text for entry, text in _findings(report, False) if entry == "I-9"]
        assert f"CUPY_ACCELERATORS is '{setting}' in this shell" in text


def test_a_reduction_that_fails_or_sums_wrong_either_way_is_a_problem(site):
    for square_sum in ({"cub_error": CUB_FAILS, "plain_error": "CompileException: error"},
                       {"cub_error": CUB_FAILS, "plain": 4.0}):
        report, lines = _square_sum(site, square_sum, "")
        (text,) = [text for entry, text in _findings(report, True) if entry == "I-9"]
        assert "with CUB and without" in text
    assert "gives 4.0, not 10500000" in lines


def test_a_reduction_is_compiled_only_where_a_gpu_is_seen(site):
    _, lines = _square_sum(site, None, devices=0)
    assert "not compiled here: no GPU" in lines
    report, lines = _square_sum(site, {"error": "AttributeError: no _accelerator"})
    assert "its reduction kernel not compiled: AttributeError" in lines
    assert "I-9" not in _entries(report)


def test_cupy_that_does_not_import(site):
    _python(site["pytom"], {"cupy": {"python": "3.12.8", "cupy": {
        "error": "ModuleNotFoundError(\"No module named 'cupy'\")"}}}, {"pytom-match-pick": "0.14.0"})
    assert "I-5" in _entries(_check())


def test_a_pytom_folder_without_its_scripts(site):
    (site["pytom"] / "pytom_extract_candidates.py").unlink()
    problems = _findings(_check(), True)
    assert any(entry == "E-9" and "pytom_extract_candidates.py" in text for entry, text in problems)


# --- room ------------------------------------------------------------------------


def test_a_small_temporary_folder_stops_pytorchs_install(site, monkeypatch):
    real = show_setup.free_space

    def free_space(path):
        free, total = real(path)
        return (100 * 10**6, 100 * 10**6) if path == site["temp"] else (free, total)

    monkeypatch.setattr(show_setup, "free_space", free_space)
    problems = _findings(_check(), True)
    (text,) = [text for entry, text in problems if entry == "E-1"]
    assert "100.0 MB free" in text and "Set TMPDIR" in text


def test_tmpdir_naming_no_folder_is_passed_over_silently(site, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(site["tmp"] / "not-made"))
    report = _check()
    (text,) = [text for entry, text in _findings(report, True) if entry == "E-1"]
    assert "not-made" in text and "without a word" in text
    temp = next(place for place in report.places if place.label == "temporary files")
    assert temp.path != str(site["tmp"] / "not-made")


def test_environments_in_a_small_home(site, monkeypatch):
    home_root = site["tmp"] / "home" / "micromamba"
    (home_root / "envs").mkdir(parents=True)
    shutil.move(str(site["root"] / "envs" / "cryoet-pipeline"), str(home_root / "envs"))
    monkeypatch.setenv("MAMBA_ROOT_PREFIX", str(home_root))
    home = str(site["tmp"] / "home")
    real = show_setup.free_space
    monkeypatch.setattr(show_setup, "free_space", lambda path: (
        (5 * 10**9, 50 * 10**9) if str(path).startswith(home) else real(path)))
    notes = _findings(_check(), False)
    (text,) = [text for entry, text in notes if entry == "I-3"]
    # pip's cache is in the home by default, PIP_CACHE_DIR being unset.
    assert text.startswith(
        "environments, package cache, pip's cache go into your home, which has 5.0 GB free")


def test_a_nearly_full_filesystem_is_named_once(site, monkeypatch):
    monkeypatch.setattr(show_setup, "free_space", lambda path: (10**8, 10**12))
    problems = [text for entry, text in _findings(_check(), True) if entry == "I-3"]
    assert len(problems) == len({place.mount for place in _check().places}) == 1


def test_sizes_measures_each_place(site):
    data = site["tmp"] / "data"
    data.mkdir()
    (data / "frames.tif").write_bytes(b"x" * 300_000)
    report = _check("--folder", str(data), "--sizes")
    folder = next(place for place in report.places if place.label == "--folder")
    assert folder.size is not None and folder.size >= 300_000


# --- the environment ------------------------------------------------------------------


def test_the_named_environment_is_asked_though_another_is_active(site, monkeypatch):
    monkeypatch.setenv("CONDA_PREFIX", str(site["tmp"] / "base"))
    report = _check()
    assert report.environment["path"] == str(site["root"] / "envs" / "cryoet-pipeline")
    assert "not active in this shell" in report.environment["how"]
    assert any("Activate it first" in text for _, text in _findings(report, False))


def test_no_environment_at_all(site, monkeypatch):
    shutil.rmtree(site["root"] / "envs")
    problems = _findings(_check(), True)
    assert any("no environment with the pipeline" in text for _, text in problems)


def test_an_environment_without_pytorch(site):
    packages = {name: "1.0" for name, _ in show_setup.PACKAGES}
    packages["torch"] = None
    _python(site["env_bin"], {"env": _env_answer(packages=packages, torch={})})
    (text,) = [text for entry, text in _findings(_check(), True) if entry == "I-1"]
    assert "lacks torch" in text and "every cryoet command imports PyTorch" in text


def test_the_pipeline_from_another_clone(site):
    _python(site["env_bin"], {"env": _env_answer(
        pipeline="/elsewhere/cryoet-pipeline/src/cryoet_pipeline/__init__.py")})
    (text,) = [text for entry, text in _findings(_check(), True) if entry == "I-8"]
    assert "/elsewhere/cryoet-pipeline/src/cryoet_pipeline" in text


def test_a_cpu_only_pytorch_on_linux_is_a_problem(site, monkeypatch):
    _python(site["env_bin"], {"env": _env_answer(torch={"version": "2.15.0+cpu", "cuda": None,
                                                        "available": False})})
    monkeypatch.setattr(show_setup.sys, "platform", "linux")
    assert "I-6" in _entries(_check())


def test_without_env_manager_a_micromamba_user_is_told(site, monkeypatch):
    monkeypatch.delenv("ENV_MANAGER")
    _program(site["bin"] / "micromamba", "exit 0\n")
    assert "I-7" in _entries(_check(), problem=False)


def test_an_unknown_env_manager_stops_the_jobs(site, monkeypatch):
    monkeypatch.setenv("ENV_MANAGER", "mamba")
    assert "I-7" in _entries(_check())


def test_the_probe_runs_in_a_real_environment():
    # This test's own environment: the probe script itself, not a stand-in.
    found = show_setup.ask_python(Path(sys.executable), show_setup.ENV_PROBE,
                                  json.dumps([name for name, _ in show_setup.PACKAGES]))
    assert "error" not in found, found
    assert found["packages"]["cryoet-pipeline"] and found["packages"]["numpy"]
    assert found["pipeline"].endswith(os.path.join("cryoet_pipeline", "__init__.py"))
    assert found["torch"]["version"] and "available" in found["torch"]


# --- the tools as a lane names them ---------------------------------------------------------


def test_a_lane_of_frames_needs_motioncor3_and_names_its_ctffind(site, monkeypatch, tmp_path):
    monkeypatch.delenv("MOTIONCOR3")
    (site["bin"] / "MotionCor3").unlink()
    lane = tmp_path / "lane.toml"
    lane.write_text('[lane]\ninput = "frames"\n[motion]\nbackend = "motioncor3"\n'
                    '[executables]\nctffind = "$CTFFIND5"\n')
    problems = _findings(_check("--config", str(lane)), True)
    assert ("C-4", 'CTFFIND (needed by every lane): lane.toml [executables] ctffind = "$CTFFIND5": '
                   "$CTFFIND5 is not set in this shell") in problems
    assert any(entry == "E-9" and text.startswith("MotionCor3 (needed by lanes of frames")
               for entry, text in problems)


def test_a_ctffind_that_is_a_folder(site, monkeypatch):
    monkeypatch.setenv("CTFFIND5", str(site["tmp"]))
    assert "C-3" in _entries(_check())


def test_a_missing_config_is_refused(site, capsys):
    assert show_setup.main(["--config", "no-such.toml"]) == 2


# --- the rules -------------------------------------------------------------------------------


@pytest.mark.parametrize("capability, cubins, ptx, runs", [
    ("8.6", ["sm_86"], [], True),
    ("8.9", ["sm_86"], [], True),
    ("8.0", ["sm_86"], [], False),
    ("9.0", ["sm_86"], [], False),
    ("9.0", ["sm_86"], ["sm_86"], True),
    ("7.0", ["sm_75"], ["sm_120"], False),
    ("9.0", ["sm_90a"], [], True),
    ("10.3", ["sm_100"], [], True),
    ("10.3", ["sm_90a"], [], False),
    ("6.1", ["sm_61"], [], True),
])
def test_which_gpu_runs_which_code(capability, cubins, ptx, runs):
    assert show_setup.runs_on(capability, show_setup.GpuCode("x", cubins, ptx)) is runs


@pytest.mark.parametrize("gres, capability", [
    ("a40", "8.6"), ("tesla-v100", "7.0"), ("tesla-t4", "7.5"), ("h100", "9.0"), ("h200", "9.0"),
    ("b300", "10.3"), ("a100_3g.20gb", "8.0"), ("quadro_rtx_8000", "7.5"), ("p600", "6.1"),
    ("p6000", "6.1"), ("gpu", ""),
    # An RTX A1000 is no A100: a model counts only as a word of the type's name.
    ("rtx_a1000", ""),
    # A number that names cards of different generations stays unknown: 4000 is
    # a Quadro RTX (7.5), an RTX Ada (8.9) and an RTX PRO Blackwell (12.0).
    ("rtx4000", ""), ("rtx_6000", ""),
    # Stockholm's cards, however the maker's and the line's words are written,
    # and NVIDIA's other current ones (its table in the library of 2026-11-17).
    ("a5000", "8.6"), ("rtx_a5000", "8.6"), ("rtxa5000", "8.6"), ("nvidia_rtx_a5000", "8.6"),
    ("3080", "8.6"), ("rtx3070", "8.6"), ("geforce_rtx_3080_ti", "8.6"), ("4070ti", "8.9"),
    ("rtx_4070_ti", "8.9"), ("gtx2080", "7.5"), ("rtx_2080ti", "7.5"), ("l40s", "8.9"),
    ("l4", "8.9"), ("a10", "8.6"), ("a2", "8.6"), ("a2000", "8.6"), ("a30", "8.0"),
    ("b200", "10.0"), ("gb200", "10.0"), ("rtx_5090", "12.0"),
])
def test_gres_types_to_compute_capability(gres, capability):
    assert show_setup.model_capability(gres) == capability


@pytest.mark.parametrize("built, driver, fit", [
    ("13.2", "13.3", "ok"), ("13.2", "13.0", "minor"), ("13.0", "12.9", "no"),
    ("12.1", "13.3", "ok"), ("11.8", "12.9", "ok"),
])
def test_cuda_builds_and_drivers(built, driver, fit):
    assert show_setup.cuda_fit(built, driver) == fit


def test_cuda_versions_as_the_runtime_states_them():
    assert show_setup.cuda_text(12090) == "12.9"
    assert show_setup.cuda_text(13030) == "13.3"
    assert show_setup.cuda_text(None) == ""


# --- kept in step with the rest of the repository --------------------------------------------


def test_the_tools_are_found_as_build_library_finds_them():
    assert show_setup.LOOKUP == build_library.LOOKUP
    assert show_setup.LANE_EXECUTABLES == build_library.LANE_EXECUTABLES
    assert show_setup.MAC_IMOD == build_library.MAC_IMOD


def test_the_packages_are_pyprojects():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]

    def names(requirements):
        return {requirement.split(">")[0].split("=")[0].split("<")[0].strip()
                for requirement in requirements}

    listed = {}
    for name, extra in show_setup.PACKAGES:
        listed.setdefault(extra, set()).add(name)
    assert listed["core"] == names(project["dependencies"])
    assert listed["library"] == names(extras["library"])
    # Empty, their packages core: kept for the install commands that name them.
    assert extras["io"] == extras["gpu"] == extras["viz"] == []
    assert show_setup.PYTHON_NEEDED == tuple(
        int(part) for part in project["requires-python"].lstrip(">=").split("."))


def test_the_pytom_scripts_are_the_ones_the_preflight_requires():
    text = (ROOT / "scripts" / "lane.sbatch").read_text()
    for script in show_setup.PYTOM_SCRIPTS:
        assert script in text


SYSTEM_PYTHON = Path("/usr/bin/python3")


def _old_python() -> bool:
    if not SYSTEM_PYTHON.exists():
        return False
    result = subprocess.run([SYSTEM_PYTHON, "-c", "import sys; print(sys.version_info >= (3, 11))"],
                            capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip() == "False"


@pytest.mark.skipif(not _old_python(), reason="no Python older than 3.11 at /usr/bin/python3")
def test_it_runs_on_a_system_python_without_the_environment(tmp_path):
    # Before the pipeline's environment exists, the system's Python is all there
    # is: on RHEL and Rocky 9 that is 3.9.
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(tmp_path)}
    result = subprocess.run([SYSTEM_PYTHON, str(ROOT / "scripts" / "show_setup.py"), "--json"],
                            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)
    assert result.returncode in (0, 1), result.stderr
    report = json.loads(result.stdout)
    assert report["host"]["python"].split()[-1].startswith("3.")
    assert any("no environment with the pipeline" in finding["text"]
               for finding in report["findings"] if finding["problem"])


# --- what can run here -------------------------------------------------------------------


def _healthy(site):
    """The stand-in site without its two faults: CTFFIND 5, and every library found."""
    _program(site["bin"] / "ctffind", 'echo " Version : 5.0.2"\n')
    _program(site["bin"] / "ldd",
             'printf "\\tlibcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12\\n"\n')


def _steps(report):
    return {step["name"]: (step["status"], step["why"]) for step in report.steps}


def _no_gpu_here(site):
    torch = dict(PIPELINE_TORCH, available=False)
    torch.pop("devices")
    _python(site["env_bin"], {"env": _env_answer(torch=torch)})
    (site["bin"] / "nvidia-smi").unlink()


def test_on_a_cluster_each_gpu_step_runs_on_the_types_its_code_fits(site):
    _healthy(site)
    report = _check()
    assert _entries(report) == set(), _findings(report, True)
    steps = _steps(report)
    assert report.gpu["mode"] == "on this cluster's nodes, through Slurm"
    # MotionCor3 has sm_86 cubins; both PyTorch builds start at sm_75, and
    # MemBrain-seg's preflight takes no GPU below 7.5; CuPy compiles for any.
    assert steps["motion correction"] == ("yes", "on the GPU nodes: a40")
    assert steps["MRA"] == ("yes", "on the GPU nodes: a40")
    assert steps["membranes"] == ("yes", "on the GPU nodes: a40")
    assert steps["picking"] == ("yes", "on the GPU nodes: a40, tesla-v100")
    for name in ("alignment", "CTF", "tomogram", "template", "extraction"):
        assert steps[name] == ("yes", ""), name
    assert steps["fine alignment with AreTomo2"][0] == "no"


def _sinfo(site, gres):
    _program(site["bin"] / "sinfo", 'if [[ "$1" == -V ]]; then echo "slurm 26.05.4"; else\n'
                                    f'echo "gpu|{gres}"\n'
                                    'echo "normal|(null)"; fi\n')


def test_a_gpu_type_the_check_does_not_know_is_named_not_dropped(site):
    # In Stockholm on 23 November 2026 part 5 said 'a40' alone: the reserved
    # A5000s had no compute capability in the check and fell out of the list.
    _healthy(site)
    _sinfo(site, "gpu:a40:2(S:0-1),gpu:rtx_a5000:8(S:0-1),gpu:titan_v:2(S:0)")
    report = _check()
    steps = _steps(report)
    assert report.gpu["cluster"] == {"a40": "8.6", "rtx_a5000": "8.6", "titan_v": ""}
    assert steps["motion correction"] == (
        "yes", "on the GPU nodes: a40, rtx_a5000; not known for titan_v (part 4)")
    assert steps["membranes"] == (
        "yes", "on the GPU nodes: a40, rtx_a5000; not known for titan_v (part 4)")
    # CuPy compiles its kernels when it runs, so it fits a type of any capability.
    assert steps["picking"] == ("yes", "on the GPU nodes: a40, rtx_a5000, titan_v")
    (note,) = [text for _, text in _findings(report, False) if "titan_v" in text]
    assert "compute capability" in note and "developer.nvidia.com/cuda/gpus" in note
    assert ("MotionCor3: runs on NVIDIA A40 here, a40, rtx_a5000; not known for titan_v"
            in report.gpu["fits"]), report.gpu["fits"]
    assert "titan_v ?" in show_setup.render(report)


def test_on_gpus_the_check_does_not_know_a_gpu_step_is_open(site):
    _healthy(site)
    _sinfo(site, "gpu:titan_v:2(S:0)")
    steps = _steps(_check())
    assert steps["motion correction"] == ("?", "not known for titan_v (part 4)")
    assert steps["picking"] == ("yes", "on the GPU nodes: titan_v")


def test_a_machine_without_a_gpu_runs_the_cpu_steps(site):
    _healthy(site)
    _no_gpu_here(site)
    (site["bin"] / "sinfo").unlink()
    report = _check()
    steps = _steps(report)
    assert report.gpu["mode"] == "on this machine itself (no Slurm here)"
    assert steps["motion correction"] == (
        "no", "needs an NVIDIA GPU, and this machine has no NVIDIA GPU")
    assert steps["picking"][0] == "no"
    for name in ("stack import", "alignment", "CTF", "tomogram", "template", "extraction"):
        assert steps[name] == ("yes", ""), name
    assert steps["MRA"] == ("cpu", "on the CPU, slower: this machine has no NVIDIA GPU")
    assert steps["membranes"][0] == "cpu"
    text = show_setup.render(report)
    assert "does not run here: motion correction, fine alignment with AreTomo2, picking" in text


def test_a_gpu_without_its_driver(site, monkeypatch):
    _healthy(site)
    _no_gpu_here(site)
    (site["bin"] / "sinfo").unlink()
    pci = site["tmp"] / "pci"
    card = pci / "0000:41:00.0"
    card.mkdir(parents=True)
    for name, value in (("vendor", "0x10de"), ("class", "0x030200"), ("device", "0x2235")):
        (card / name).write_text(value + "\n")
    nouveau = site["tmp"] / "drivers" / "nouveau"
    nouveau.mkdir(parents=True)
    (card / "driver").symlink_to(nouveau)
    intel = pci / "0000:00:02.0"
    intel.mkdir()
    for name, value in (("vendor", "0x8086"), ("class", "0x030000"), ("device", "0x4680")):
        (intel / name).write_text(value + "\n")
    ids = site["tmp"] / "pci.ids"
    ids.write_text("10de  NVIDIA Corporation\n\t2235  GA102GL [A40]\n\t\t10de 145a  A40\n"
                   "8086  Intel Corporation\n\t4680  AlderLake-S GT1\n")
    monkeypatch.setattr(show_setup, "SYS_PCI", pci)
    monkeypatch.setattr(show_setup, "PCI_IDS", (ids,))
    report = _check()
    assert [card["name"] for card in report.gpu["pci"]] == ["GA102GL [A40]"]
    (text,) = [text for entry, text in _findings(report, True) if entry == "I-12"]
    assert "GA102GL [A40]" in text and "nouveau" in text
    assert _steps(report)["motion correction"] == (
        "no", "needs an NVIDIA GPU, and the NVIDIA GPU here has no driver")


def test_a_stack_lane_leaves_out_what_it_does_not_run(site, tmp_path):
    _healthy(site)
    lane = tmp_path / "stack.toml"
    lane.write_text('[lane]\ninput = "stack"\n')
    report = _check("--config", str(lane))
    in_lane = {step["name"]: step["in_lane"] for step in report.steps}
    assert in_lane["motion correction"] == "no" and in_lane["stack import"] == "yes"
    assert in_lane["MRA"] == "no" and in_lane["picking"] == "yes"
    table = show_setup.render(report).split("5. What can run here")[1].split("Problems")[0]
    assert "motion correction" not in table and "stack import" in table


def test_a_cpu_only_pytorch_runs_the_mra_on_the_cpu_only(site, monkeypatch):
    _healthy(site)
    _python(site["env_bin"], {"env": _env_answer(torch={"version": "2.15.0+cpu", "cuda": None,
                                                        "available": False})})
    assert _steps(_check())["MRA"] == (
        "cpu", "on the CPU, slower: the pipeline's PyTorch is a CPU-only build")


# --- the clone --------------------------------------------------------------------------------

_GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.org",
                GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.org",
                GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=_GIT_ENV).stdout.strip()


@pytest.fixture
def clones(tmp_path, monkeypatch):
    """A remote, a clone that pushes to it, and the clone the check looks at."""
    for name, value in _GIT_ENV.items():
        monkeypatch.setenv(name, value)
    remote = tmp_path / "remote.git"
    _git("init", "--bare", "-b", "main", str(remote), cwd=tmp_path)
    work = tmp_path / "work"
    _git("clone", str(remote), str(work), cwd=tmp_path)
    (work / "a.txt").write_text("1")
    _git("add", "a.txt", cwd=work)
    _git("commit", "-m", "one", cwd=work)
    _git("push", "origin", "HEAD:main", cwd=work)
    mine = tmp_path / "mine"
    _git("clone", str(remote), str(mine), cwd=tmp_path)
    return remote, work, mine


def _clone(root):
    report = show_setup.Report()
    show_setup.clone_state(report, root)
    return report


def test_a_clone_up_to_date(clones):
    report = _clone(clones[2])
    assert report.clone["state"] == "up to date with origin/main" and report.findings == []


def test_a_clone_behind_its_remote_is_told_to_pull_and_nothing_is_fetched(clones):
    remote, work, mine = clones
    (work / "a.txt").write_text("2")
    _git("commit", "-am", "two", cwd=work)
    _git("push", "origin", "HEAD:main", cwd=work)
    before = _git("rev-parse", "origin/main", cwd=mine)
    report = _clone(mine)
    assert report.clone["state"] == "origin/main has commits this clone does not"
    (finding,) = report.findings
    assert f"run git pull in {mine}" in finding.text and not finding.problem
    assert _git("rev-parse", "origin/main", cwd=mine) == before


def test_a_clone_with_its_own_commits(clones):
    mine = clones[2]
    (mine / "b.txt").write_text("mine")
    _git("add", "b.txt", cwd=mine)
    _git("commit", "-m", "mine", cwd=mine)
    assert _clone(mine).clone["state"] == "ahead of origin/main: it has commits origin/main lacks"


def test_a_remote_that_does_not_answer(clones):
    shutil.rmtree(clones[0])
    state = _clone(clones[2]).clone["state"]
    assert state == "origin did not answer, so whether it has newer commits is unknown"


def test_a_folder_that_is_no_clone(tmp_path):
    assert _clone(tmp_path).clone["state"] == "not a git clone, or no git here"


# --- tools this shell does not name -----------------------------------------------------------


def test_find_names_a_ctffind_5_that_the_shell_does_not(site, monkeypatch):
    software = site["tmp"] / "software"
    ctffind5 = _program(software / "ctffind-5.0.2" / "cisTEM" / "bin" / "ctffind",
                        'echo " Version : 5.0.2"\n')
    monkeypatch.setattr(show_setup, "SEARCH_ROOTS", (str(software),))
    report = _check("--find")
    tool = next(tool for tool in report.tools if tool.name == "CTFFIND")
    assert tool.elsewhere[0]["path"] == str(ctffind5) and tool.elsewhere[0]["version"] == "5.0.2"
    assert ("E-9", f"CTFFIND 5.0.2 is at {ctffind5} (under {software}): name it with export "
                   f"CTFFIND5={ctffind5} in ~/.bashrc, or in a setup file that ~/.bashrc sources "
                   "(--setup-file proposes one)") in _findings(report, False)
    assert f"export CTFFIND5={ctffind5}" in show_setup.setup_file(report)


def test_find_reads_the_environment_modules(site, monkeypatch):
    motioncor3 = _program(site["tmp"] / "sw" / "mcor" / "bin" / "MotionCor3",
                          'echo "MotionCor3 version 1.2.4"\n')
    monkeypatch.delenv("MOTIONCOR3")
    (site["bin"] / "MotionCor3").unlink()
    monkeypatch.setattr(show_setup, "module_names",
                        lambda: ["mcor/v3-cu12.5", "cuda/12.5", "relion/v5"])
    monkeypatch.setattr(show_setup, "in_module", lambda name: (
        [str(motioncor3.parent)] if name.startswith("mcor") else []))
    report = _check("--find")
    assert report.modules == {"motioncor3": ["mcor/v3-cu12.5"], "cuda": ["cuda/12.5"]}
    tool = next(tool for tool in report.tools if tool.name == "MotionCor3")
    assert (tool.elsewhere[0]["path"], tool.elsewhere[0]["how"]) == (
        str(motioncor3), "module mcor/v3-cu12.5")
    text = show_setup.setup_file(report)
    assert f"export MOTIONCOR3={motioncor3}" in text and "cuda/12.5" in text


def test_find_finds_tools_in_other_environments(site, monkeypatch):
    other = site["root"] / "envs" / "membrain-v2" / "bin"
    python = _python(other, {"env": {"python": "3.10.0", "packages": {}}}, {"membrain-seg": "0.0.10"})
    (other / "membrain").write_text(f"#!{python}\n")
    (other / "membrain").chmod(0o755)
    monkeypatch.delenv("MEMBRAIN_EXE")
    report = _check("--find")
    tool = next(tool for tool in report.tools if tool.name == "MemBrain-seg")
    assert tool.elsewhere[0]["path"] == str(other / "membrain")
    assert tool.elsewhere[0]["how"].startswith("environment membrain-v2")


def test_module_show_in_lmod_and_tcl_forms():
    lua = 'prepend_path("PATH","/sw/mcor/bin")\nsetenv("MOTIONCOR3","/sw/mcor/bin/MotionCor3")\n'
    tcl = "prepend-path\tPATH /sw/ctffind/bin\nsetenv CTFFIND5 /sw/ctffind/bin/ctffind\n"
    assert show_setup.module_paths(lua) == ["/sw/mcor/bin", "/sw/mcor/bin/MotionCor3"]
    assert show_setup.module_paths(tcl) == ["/sw/ctffind/bin", "/sw/ctffind/bin/ctffind"]


@pytest.mark.parametrize("tool, name, found", [
    ("motioncor3", "MotionCor3", True),
    ("motioncor3", "MotionCor3_1.0.1_Cuda118_06-14-2023", True),
    ("motioncor3", "MotionCor3.cpp", False),
    ("aretomo", "AreTomo2", True),
    ("aretomo", "AreTomo3", True),
    ("aretomo", "AreTomo2_1.1.3_Cuda12", True),
    ("aretomo", "AreTomoSomething", False),
    ("ctffind", "ctffind5", True),
    ("ctffind", "ctffind_plot_results.sh", False),
    ("imod", "newstack", True),
    ("pytom-match-pick", "pytom_match_template.py", True),
    ("membrain-seg", "membrain", True),
])
def test_which_files_find_takes_for_a_tool(tool, name, found):
    assert show_setup.is_program(tool, name) is found


def test_a_search_that_runs_out_of_time_says_so(tmp_path):
    (tmp_path / "a").mkdir()
    assert show_setup.walk_programs(tmp_path, seconds=-1.0)[1] is True


def test_the_setup_file_proposes_room_and_leaves_what_is_set(site, monkeypatch):
    work = site["tmp"] / "work"
    work.mkdir()
    monkeypatch.setenv("CRYOET_WORK", str(work))
    monkeypatch.delenv("TMPDIR")
    text = show_setup.setup_file(_check("--find"))
    assert "export TMPDIR=$CRYOET_WORK/tmp" in text and 'mkdir -p "$TMPDIR"' in text
    assert "export PIP_CACHE_DIR=$CRYOET_WORK/cache/pip" in text
    assert f"# MAMBA_ROOT_PREFIX is set in your shell already: {site['root']}" in text
    assert f"# PYTOMO_MP_DIR is set in your shell already: {site['pytom']}" in text
    assert "# export ARETOMO=" in text
    assert subprocess.run(["bash", "-n"], input=text, text=True).returncode == 0


def test_setup_file_prints_the_file_alone(site, capsys):
    assert show_setup.main(["--setup-file"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# pipeline_setup.sh") and "1. Room" not in out


# What the check read on max on 19 November 2026 (an A40 node), and the ranges it
# then listed one cubin at a time.
MDC_MOTIONCOR3 = ["sm_52", "sm_70", "sm_75", "sm_80", "sm_86", "sm_87", "sm_89", "sm_90"]
MDC_ARETOMO2 = ["sm_52", "sm_53", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
MDC_TORCH = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.parametrize("cubins, ptx, words", [
    (MDC_MOTIONCOR3, [], "5.2 to 5.9, 7.0 to 9.9"),
    (MDC_ARETOMO2, [], "5.2 to 9.9"),
    (MDC_TORCH, [], "7.5 to 10.9, 12.0 to 12.9"),
    (["sm_86"], [], "8.6 to 8.9"),
    (["sm_90a"], [], "9.0 only"),
    (["sm_90", "sm_90a"], [], "9.0 to 9.9"),
    (["sm_80", "sm_100a"], ["compute_90"], "8.0 to 8.9, 10.0 only, 9.0 and newer through PTX"),
])
def test_the_ranges_a_program_runs_on_are_joined(cubins, ptx, words):
    code = show_setup.GpuCode("a program", cubins=cubins, ptx=ptx)
    assert show_setup.coverage(code) == "runs on compute capability " + words
    # The words say what runs_on decides, capability by capability.
    for major in range(5, 13):
        for minor in range(10):
            capability = f"{major}.{minor}"
            said = any(
                part.endswith(" only") and part == f"{capability} only"
                or " to " in part and show_setup.version_tuple(part.split(" to ")[0])
                <= (major, minor) <= show_setup.version_tuple(part.split(" to ")[1])
                or part.endswith("through PTX")
                and show_setup.version_tuple(part.split(" and ")[0]) <= (major, minor)
                for part in words.split(", "))
            assert said == show_setup.runs_on(capability, code), capability
