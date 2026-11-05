"""Tests for cryoet_pipeline.tool_versions: bounded probes, and each tool's
version read the way the tool states it."""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cryoet_pipeline import tool_versions as tv


def stand_in(folder: Path, name: str, body: str) -> Path:
    path = folder / name
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(0o755)
    return path


def test_a_probe_reads_a_command_that_ends(tmp_path):
    exe = stand_in(tmp_path, "says", "echo out; echo err >&2; exit 3\n")
    result = tv.probe([exe])
    assert result.output == "out\nerr\n"
    assert (result.exit_code, result.stopped, result.error) == (3, None, None)
    assert not result.ended_well


def test_a_probe_stops_a_command_that_waits(tmp_path):
    exe = stand_in(tmp_path, "waits", "echo started; sleep 30\n")
    started = time.monotonic()
    result = tv.probe([exe], seconds=0.5)
    assert (result.stopped, result.exit_code, result.output) == ("time", None, "started\n")
    assert time.monotonic() - started < 5


def test_a_probe_stops_a_command_that_keeps_printing(tmp_path):
    exe = stand_in(tmp_path, "floods", "yes flood\n")
    result = tv.probe([exe], limit=1000)
    assert result.stopped == "limit"
    assert len(result.output) == 1000


def test_a_probe_stops_once_the_answer_is_read(tmp_path):
    exe = stand_in(tmp_path, "asks", "echo 'Version : 5.0.2'; sleep 30\n")
    started = time.monotonic()
    result = tv.probe([exe], seconds=20, stop=lambda text: "Version" in text)
    assert result.stopped == "match"
    assert time.monotonic() - started < 5


def test_a_probe_stops_what_the_command_started(tmp_path):
    marker = tmp_path / "child.pid"
    exe = stand_in(tmp_path, "parent", f"sleep 30 & echo $! > {marker}; echo ok; wait\n")
    assert tv.probe([exe], seconds=0.5).stopped == "time"
    child = int(marker.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"the probe's child {child} is still running")


def test_a_probe_names_a_command_that_cannot_start(tmp_path):
    result = tv.probe([tmp_path / "absent"])
    assert result.error
    assert result.exit_code is None
    assert not result.ended_well


def test_imod_states_its_version_in_a_file(tmp_path):
    (tmp_path / "VERSION").write_text("5.1.12\n")
    assert tv.imod_version(tmp_path).value == "5.1.12"
    assert tv.imod_version(tmp_path / "elsewhere").value is None


def test_ctffind_is_read_from_its_banner_and_not_waited_for(tmp_path):
    exe = stand_in(tmp_path, "ctffind", """\
        echo '        **   Welcome to Ctffind   **'
        echo '            Version : 5.0.2'
        echo '    Library Version : 2.0.0-alpha'
        echo 'Input image file name [input.mrc]   :'
        sleep 30
        """)
    started = time.monotonic()
    assert tv.ctffind_version(exe).value == "5.0.2"
    assert time.monotonic() - started < 5


def test_ctffind_library_version_is_not_its_own(tmp_path):
    exe = stand_in(tmp_path, "ctffind", "echo '    Library Version : 2.0.0'\n")
    assert tv.ctffind_version(exe, seconds=2).value is None


def test_a_script_runs_under_the_python_its_first_line_names(tmp_path):
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    python = env / "python3"
    python.write_text("")
    direct = env / "direct"
    direct.write_text(f"#!{python}\nprint()\n")
    via_env = env / "via_env"
    via_env.write_text("#!/usr/bin/env python3\nprint()\n")
    shell = env / "shell"
    shell.write_text("#!/bin/sh\necho\n")
    assert tv.script_python(direct) == python
    assert tv.script_python(via_env) == python
    assert tv.script_python(shell) is None
    assert tv.script_python(tmp_path / "absent") is None


def test_a_package_version_comes_from_its_environment():
    assert tv.package_version(sys.executable, "pytest").value == pytest.__version__
    assert tv.package_version(sys.executable, "no-such-distribution-here").value is None


def test_slurm_and_bash_state_theirs(tmp_path):
    sinfo = stand_in(tmp_path, "sinfo", "echo 'slurm 26.05.3'\n")
    assert tv.slurm_version(sinfo).value == "26.05.3"
    bash = shutil.which("bash")
    assert bash is not None
    assert re.fullmatch(r"\d+(\.\d+)+", tv.bash_version(bash).value or "")


def test_a_stated_version_must_follow_the_programs_own_name(tmp_path):
    exe = stand_in(tmp_path, "MotionCor3", """\
        echo 'CUDA version 12.2'
        echo 'MotionCor3 built with CUDA 12.2'
        echo 'MotionCor3 version 1.1.2'
        """)
    found = tv.stated_version(exe, "MotionCor3")
    assert (found.value, found.how) == ("1.1.2", "MotionCor3 --version")


def test_a_record_names_the_imod_install_its_program_belongs_to(tmp_path):
    install = tmp_path / "imod_5.1.12"
    program = install / "bin" / "newstack"
    program.parent.mkdir(parents=True)
    program.touch()
    (install / "VERSION").write_text("5.1.12\n")
    link = tmp_path / "on-path" / "newstack"
    link.parent.mkdir()
    link.symlink_to(program)
    assert tv.imod_version_of(program) == "5.1.12"
    # A program on PATH that links into the install, as a module may put it.
    assert tv.imod_version_of(link) == "5.1.12"
    bare = tmp_path / "bare" / "bin" / "tilt"
    bare.parent.mkdir(parents=True)
    bare.touch()
    assert tv.imod_version_of(bare) == tv.EXTERNAL


def test_a_record_names_the_package_version_of_the_scripts_environment(tmp_path):
    script = tmp_path / "pytom_match_template.py"
    script.write_text(f"#!{sys.executable}\n")
    assert tv.package_version_of(script, "pytest") == pytest.__version__
    assert tv.package_version_of(script, "no-such-distribution-here") == tv.EXTERNAL
    plain = tmp_path / "plain"
    plain.write_text("no interpreter named here\n")
    assert tv.package_version_of(plain, "pytest") == tv.EXTERNAL


def test_motioncor3_states_its_version_for_a_record(tmp_path):
    named = stand_in(tmp_path, "MotionCor3",
                     'if [ "$1" = "--version" ]; then echo "MotionCor3 version 1.2.4"; fi\n')
    assert tv.motioncor3_version_of(named) == "1.2.4"
    silent = stand_in(tmp_path, "MotionCor3-silent", "echo 'CUDA version 12.5'\n")
    assert tv.motioncor3_version_of(silent) is None


def test_a_runs_own_output_gives_ctffinds_version():
    said = "   **   Welcome to Ctffind   **\n   Version : 5.0.2\n   Library Version : 2.0.0\n"
    assert tv.ctffind_banner_version(said) == "5.0.2"
    assert tv.ctffind_banner_version("   Library Version : 2.0.0\n") is None


def test_no_stated_version_is_unknown_rather_than_a_guess(tmp_path):
    exe = stand_in(tmp_path, "MotionCor3",
                   "echo 'CUDA version 12.2'; echo 'MotionCor3 built with CUDA 12.2'\n")
    found = tv.stated_version(exe, "MotionCor3")
    assert found.value is None
    assert "CUDA version 12.2" in found.output
