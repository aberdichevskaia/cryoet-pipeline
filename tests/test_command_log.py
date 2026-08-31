from __future__ import annotations

import subprocess
from pathlib import Path

from cryoet_pipeline.backends.command_log import (
    collapse_progress,
    command_log_text,
    write_command_log,
)


def _result(returncode: int = 0, stdout: str = "out", stderr: str = "err"):
    return subprocess.CompletedProcess(
        args=["tool"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_one_command_is_recorded_with_argv_output_and_exit_code() -> None:
    text = command_log_text((["tiltalign", "-ModelFile", "a.fid"], _result()))

    assert text.startswith("$ tiltalign -ModelFile a.fid\n")
    assert "[stdout]\nout\n" in text
    assert "[stderr]\nerr\n" in text
    assert "[exit_code]\n0\n" in text


def test_several_commands_are_recorded_in_order() -> None:
    text = command_log_text(
        (["first"], _result(stdout="one")),
        (["second"], _result(stdout="two")),
    )

    assert text.index("$ first") < text.index("$ second")
    assert "one" in text and "two" in text


def test_argv_tokens_that_are_not_strings_are_coerced() -> None:
    """One of the twelve copies did this and the others crashed on it.

    Paths and numbers reach these calls routinely, so the tolerant behaviour is
    the one worth keeping — the strict variants only ever differed by failing.
    """

    text = command_log_text((["tilt", Path("/tmp/x.rec"), 320], _result()))

    assert "$ tilt /tmp/x.rec 320\n" in text


def test_a_failing_command_records_its_exit_code() -> None:
    """A log that only describes successes is no use for the failures."""

    text = command_log_text((["ctffind"], _result(returncode=1, stderr="boom")))

    assert "[exit_code]\n1\n" in text
    assert "boom" in text


def test_empty_output_is_recorded_as_empty_rather_than_as_none() -> None:
    missing = subprocess.CompletedProcess(args=["t"], returncode=0, stdout=None, stderr=None)

    text = command_log_text((["t"], missing))

    assert "[stdout]\n\n" in text
    assert "None" not in text


def test_writing_creates_the_parent_directory_and_returns_the_path(
    tmp_path: Path,
) -> None:
    """Also one of the drifted behaviours: half the copies assumed the directory.

    The path comes back so a caller can record the log on the artifact it belongs
    to instead of reconstructing where it went.
    """

    destination = tmp_path / "logs" / "nested" / "run.log"

    written = write_command_log(destination, (["tool"], _result()))

    assert written == destination
    assert destination.is_file()
    assert destination.read_text() == command_log_text((["tool"], _result()))


def test_progress_bars_are_collapsed_to_their_last_state() -> None:
    """A redrawn bar arrives as one line per update and buries everything else.

    `text=True` turns tqdm's carriage returns into newlines, so one pytom match
    put 41,212 progress lines into a 2.7 MB log. The final state says everything
    the intermediate ones do.
    """

    text = "\n".join(
        ["starting"]
        + [f"{p}%|{'#' * (p // 10)}| {p}/100 [00:01<00:09, 9.9it/s]" for p in range(0, 101)]
        + ["cut off 0.10530"]
    )

    collapsed = collapse_progress(text)

    lines = collapsed.splitlines()
    assert lines[0] == "starting"
    assert lines[1].startswith("100%|")
    assert lines[2] == "[... 100 progress updates elided]"
    assert lines[3] == "cut off 0.10530"


def test_a_single_progress_line_is_left_alone() -> None:
    """Nothing was hidden, so there is nothing to say about it."""

    collapsed = collapse_progress("50%|#####| 1/2 [00:01<00:01]")

    assert collapsed == "50%|#####| 1/2 [00:01<00:01]"
    assert "elided" not in collapsed


def test_two_separate_bars_keep_their_own_last_state() -> None:
    """Two commands in one log each reach 100% and both endings matter."""

    text = "\n".join(
        ["first tool", "10%|#| 1/10", "100%|##########| 10/10",
         "second tool", "50%|#####| 1/2", "100%|##########| 2/2", "done"]
    )

    lines = collapse_progress(text).splitlines()

    assert lines == [
        "first tool",
        "100%|##########| 10/10",
        "[... 1 progress updates elided]",
        "second tool",
        "100%|##########| 2/2",
        "[... 1 progress updates elided]",
        "done",
    ]


def test_output_without_progress_is_untouched() -> None:
    text = "INFO: estimating\nINFO: done\n"

    assert collapse_progress(text) == text
