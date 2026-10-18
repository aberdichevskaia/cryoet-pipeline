from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.ctf import (
    Ctffind5CtfEstimationBackend,
    ImodCtfPhaseFlipBackend,
    ImodCtfPlotterEstimationBackend,
    _questions_asked,
    _verify_dialogue,
    correct_ctf_and_register,
    estimate_ctf_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.imod_defocus import read_imod_defocus_file
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    CtfCorrectionQc,
    CtfEstimationQc,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference

_TILT_ANGLES = [-3.1, 0.1]
_CORRECTED_DATA = np.arange(48, dtype=np.float32).reshape(2, 4, 6)


def test_ctfplotter_estimation_writes_model_and_qc(tmp_path: Path) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    captured: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        command_list = list(command)
        captured.append(command_list)
        defocus_path = Path(command_list[command_list.index("-DefocusFile") + 1])
        # Real ctfplotter writes a version-2 file: data on the first line with
        # the version appended, exactly as the IMOD man page documents.
        defocus_path.write_text(
            "1 1 -3.100000 -3.100000 3000.000000 2\n2 2 0.100000 0.100000 2800.000000\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="plotted", stderr="")

    artifacts = estimate_ctf_and_register(
        ImodCtfPlotterEstimationBackend(fake_runner),
        aligned_stack,
        alignment,
        _manifest(),
        _context(tmp_path, {"imod_dir": _fake_imod_dir(tmp_path)}),
        registry,
    )

    ctf_model, qc_artifact = artifacts
    assert registry.get(ctf_model.id) == ctf_model
    assert ctf_model.kind == ArtifactKind.CTF_MODEL
    assert ctf_model.id == "TS_TEST:ctf_model:imod-ctfplotter"
    assert ctf_model.parent_ids == [aligned_stack.id, alignment.id]
    assert ctf_model.parameters["axis_angle_deg"] == pytest.approx(85.3)
    assert ctf_model.parameters["input_stack_id"] == aligned_stack.id
    assert ctf_model.parameters["voltage_kv"] == pytest.approx(300.0)

    command = captured[0]
    assert command[command.index("-AxisAngle") + 1] == "85.300000"
    assert command[command.index("-Voltage") + 1] == "300"
    assert command[-1] == "-SaveAndExit"

    report = CtfEstimationQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.PASS
    assert report.mean_defocus_um == pytest.approx(2.9)
    assert report.minimum_defocus_um == pytest.approx(2.8)
    assert report.maximum_defocus_um == pytest.approx(3.0)


# The questions CTFFIND asks for a multi-slice input with the answers this
# adapter sends, spelled out here rather than imported so that a change to the
# adapter's sequence has to be acknowledged in the test too. A single-image input
# draws the same list without the movie question, which is the whole difficulty:
# the hint pass and the full pass answer different sequences.
_STACK_QUESTIONS: tuple[str, ...] = (
    "Input image file name",
    "Input is a movie (stack of frames)",
    "Output diagnostic image file name",
    "Pixel size",
    "Acceleration voltage",
    "Spherical aberration",
    "Amplitude contrast",
    "Size of amplitude spectrum to compute",
    "Minimum resolution",
    "Maximum resolution",
    "Minimum defocus",
    "Maximum defocus",
    "Defocus search step",
    "Do you know what astigmatism is present?",
    "Slower, more exhaustive search?",
    "Use a restraint on astigmatism?",
    "Find additional phase shift?",
    "Determine sample tilt?",
    "Determine samnple thickness?",
    "Do you want to set expert options?",
)
_SINGLE_IMAGE_QUESTIONS: tuple[str, ...] = tuple(
    question for question in _STACK_QUESTIONS if question != "Input is a movie (stack of frames)"
)
_YES_NO_QUESTIONS = frozenset(
    {
        "Input is a movie (stack of frames)",
        "Do you know what astigmatism is present?",
        "Slower, more exhaustive search?",
        "Use a restraint on astigmatism?",
        "Find additional phase shift?",
        "Determine sample tilt?",
        "Determine samnple thickness?",
        "Do you want to set expert options?",
    }
)


def _answer_to(question: str, stdin: str, *, single_image: bool = False) -> str:
    """The answer CTFFIND would read for one question, by its position."""

    questions = _SINGLE_IMAGE_QUESTIONS if single_image else _STACK_QUESTIONS
    return stdin.splitlines()[questions.index(question)]


def _ctffind_runner(
    *,
    stdins: list[str] | None = None,
    newstack_commands: list[list[str]] | None = None,
    defocus_angstrom: float = 30000.0,
    rows: int | None = None,
    echo_prompts: bool = True,
    prompt_shift: int = 0,
    always_ask_about_movies: bool = False,
    version: str = "5.0.2",
    fit_limits: Sequence[float] | None = None,
    defocus_per_view: Sequence[float] | None = None,
):
    """Fake CTFFIND driven interactively, plus newstack.

    The real program opens the input file, then asks a question, echoes it with
    the answer it read, and moves on, aborting as soon as an answer does not fit
    the question. Two ways of getting out of step are reproduced here:
    `always_ask_about_movies` asks a question the adapter did not expect, as the
    real one does for a stack, and `prompt_shift` keeps the questions but reads
    the answers displaced.
    """

    def runner(
        command,
        *,
        cwd: Path,
        env,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        command_list = list(command)
        if "newstack" in command_list[0]:
            if newstack_commands is not None:
                newstack_commands.append(command_list)
            with mrcfile.new(Path(command_list[-1]), overwrite=True) as view:
                view.set_data(_CORRECTED_DATA[0])
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        assert stdin is not None, "CTFFIND must be driven through stdin"
        if stdins is not None:
            stdins.append(stdin)
        answers = stdin.splitlines()

        # Exactly as CTFFIND does it: open the input, then decide what to ask.
        with mrcfile.open(Path(answers[0]), header_only=True, permissive=True) as image:
            input_is_a_stack = int(image.header.nz) > 1
        asks_about_movies = input_is_a_stack or always_ask_about_movies
        questions = _STACK_QUESTIONS if asks_about_movies else _SINGLE_IMAGE_QUESTIONS

        echo = ["**   Welcome to Ctffind   **", f"            Version : {version}"]
        for index, question in enumerate(questions):
            source = index + prompt_shift
            answer = answers[source] if 0 <= source < len(answers) else ""
            if question in _YES_NO_QUESTIONS and answer.lower() not in {"yes", "no"}:
                # The real abort: the question is echoed without a value.
                echo.append(f"{question:<50} : ")
                echo.append(" Error: Running as script, and answer is not recognized...")
                return subprocess.CompletedProcess(
                    command, 255, stdout="\n".join(echo), stderr=""
                )
            echo.append(f"{question:<50} : {answer}")

        diagnostic = Path(answers[questions.index("Output diagnostic image file name")])
        diagnostic.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.write_bytes(b"diagnostic")

        fitted = _CORRECTED_DATA.shape[0] if input_is_a_stack else 1
        row_count = rows if rows is not None else fitted
        def view_defocus(index: int) -> float:
            if defocus_per_view is not None and input_is_a_stack:
                return defocus_per_view[index]
            return defocus_angstrom

        def view_fit(index: int) -> float:
            if fit_limits is not None and input_is_a_stack:
                return fit_limits[index]
            return 18.5

        diagnostic.with_suffix(".txt").write_text(
            "# CTFFIND fake output\n"
            + "".join(
                f"{index + 1}.0 {view_defocus(index)} {view_defocus(index) - 500.0} "
                f"-43.2 0.0 0.10 {view_fit(index)}\n"
                for index in range(row_count)
            )
        )
        return subprocess.CompletedProcess(
            command, 0, stdout="\n".join(echo) if echo_prompts else "", stderr=""
        )

    return runner


def test_ctffind5_estimation_fits_the_whole_stack_in_one_call(tmp_path: Path) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    ctffind = _fake_executable(tmp_path / "ctffind")
    stdins: list[str] = []
    newstack_commands: list[list[str]] = []

    artifacts = estimate_ctf_and_register(
        Ctffind5CtfEstimationBackend(
            _ctffind_runner(stdins=stdins, newstack_commands=newstack_commands)
        ),
        aligned_stack,
        alignment,
        _manifest(),
        _context(tmp_path, {"imod_dir": _fake_imod_dir(tmp_path), "ctffind_executable": ctffind}),
        registry,
    )

    ctf_model, qc_artifact = artifacts
    assert ctf_model.id == "TS_TEST:ctf_model:ctffind5"

    # One hint pass on the extracted zero-tilt view, then one call for the stack.
    assert len(stdins) == 2
    assert len(newstack_commands) == 1
    assert newstack_commands[0][newstack_commands[0].index("-secs") + 1] == "1"

    hint, full = stdins
    assert "view_" in hint.splitlines()[0], "the hint pass fits a single extracted view"
    assert full.splitlines()[0] == str(aligned_stack.path.resolve()), "then the whole stack"

    # The hint pass answers one question fewer, because a single extracted view
    # is not a stack and CTFFIND does not ask about movies.
    assert len(hint.splitlines()) == len(_SINGLE_IMAGE_QUESTIONS)
    assert len(full.splitlines()) == len(_STACK_QUESTIONS)

    assert _answer_to("Input is a movie (stack of frames)", full) == "no"
    assert float(_answer_to("Pixel size", full)) == pytest.approx(2.7)
    assert float(_answer_to("Acceleration voltage", full)) == pytest.approx(300.0)
    assert float(_answer_to("Spherical aberration", full)) == pytest.approx(2.7)
    assert int(_answer_to("Size of amplitude spectrum to compute", full)) == 512
    # Clamped to the Nyquist limit of the stack's own 2.7 A pixel.
    assert float(_answer_to("Maximum resolution", full)) == pytest.approx(5.4)
    assert _answer_to("Do you know what astigmatism is present?", full) == "no"
    assert _answer_to("Do you want to set expert options?", full) == "no"

    entries = read_imod_defocus_file(ctf_model.path)
    assert len(entries) == 2
    assert entries[0].angle_start_deg == pytest.approx(-3.1)

    report = CtfEstimationQc.model_validate_json(qc_artifact.path.read_text())
    assert report.num_tilts == 2
    # Which CTFFIND fitted it, from its banner.
    assert ctf_model.software_versions == {"ctffind": "5.0.2"}


def test_ctffind5_estimation_names_a_ctffind4_on_path(tmp_path: Path) -> None:
    """CTFFIND 4 skips the tilt and thickness questions; say that, not "out of step".

    MDC job 3670792 (2026-10-16) found CTFFIND 4.1.14 as `ctffind` and stopped at
    question 17 with a message about the question sequence.
    """

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")

    with pytest.raises(RuntimeError, match=r"is CTFFIND 4\.1\.14"):
        Ctffind5CtfEstimationBackend(_ctffind_runner(version="4.1.14")).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(
                tmp_path,
                {"imod_dir": _fake_imod_dir(tmp_path), "ctffind_executable": ctffind},
            ),
        )


def test_ctffind_version_skips_the_library_line() -> None:
    from cryoet_pipeline.backends.ctf import ctffind_version

    banner = (
        "        **   Welcome to Ctffind   **\n\n"
        "    Library Version : 2.0.0-alpha-295-b21db55-dirty\n"
        "            Version : 5.0.2\n"
    )
    assert ctffind_version(banner) == "5.0.2"
    assert ctffind_version("no banner here") is None


def test_two_ctf_lanes_coexist_in_one_registry(tmp_path: Path) -> None:
    """A labelled CTF run must not overwrite the unlabelled one, on disk or in the registry.

    Estimation and correction write under a fixed per-backend prefix, so without
    the label in the filename the second lane would silently replace the first
    lane's defocus file while both registry entries pointed at it.
    """

    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    ctffind = _fake_executable(tmp_path / "ctffind")
    imod_dir = _fake_imod_dir(tmp_path)

    def run(variant: str | None) -> list[Artifact]:
        parameters: dict[str, object] = {
            "imod_dir": imod_dir,
            "ctffind_executable": ctffind,
            "ctf_expected_defocus_um": 3.0,
        }
        if variant is not None:
            parameters["variant"] = variant
        return estimate_ctf_and_register(
            Ctffind5CtfEstimationBackend(_ctffind_runner()),
            aligned_stack,
            alignment,
            _manifest(),
            _context(tmp_path, parameters),
            registry,
        )

    plain_model, plain_qc = run(None)
    labelled_model, labelled_qc = run("bin8")

    assert plain_model.id == "TS_TEST:ctf_model:ctffind5"
    assert labelled_model.id == "TS_TEST:ctf_model:ctffind5:bin8"
    assert plain_qc.id != labelled_qc.id
    assert plain_model.path != labelled_model.path, "the defocus files must not collide"
    assert plain_model.path.is_file() and labelled_model.path.is_file()
    assert registry.get(plain_model.id).path == plain_model.path
    assert registry.get(labelled_model.id).path == labelled_model.path


def test_ctffind5_estimation_detects_an_unexpected_question(tmp_path: Path) -> None:
    """The failure this adapter actually hit, reported as itself.

    A CTFFIND that asks one question more than the answer list allows for
    consumes every later answer in the wrong slot. The exit code says only
    "answer is not recognized", so the useful diagnosis has to come from
    comparing the questions echoed against the ones answered.
    """

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")

    with pytest.raises(RuntimeError, match="question sequence differs"):
        Ctffind5CtfEstimationBackend(
            _ctffind_runner(always_ask_about_movies=True)
        ).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(
                tmp_path,
                {
                    "imod_dir": _fake_imod_dir(tmp_path),
                    "ctffind_executable": ctffind,
                },
            ),
        )


def test_ctffind5_estimation_detects_a_displaced_answer(tmp_path: Path) -> None:
    """Right questions, wrong answers: caught by the values CTFFIND echoes."""

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")

    with pytest.raises(RuntimeError, match="but was sent"):
        Ctffind5CtfEstimationBackend(_ctffind_runner(prompt_shift=1)).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(
                tmp_path,
                {
                    "imod_dir": _fake_imod_dir(tmp_path),
                    "ctffind_executable": ctffind,
                    "ctf_expected_defocus_um": 3.0,
                },
            ),
        )


# Captured from CTFFIND 5.0.2 refusing the sequence this adapter used to send,
# so the parsing is checked against the program's real output and not only
# against the fake above.
_REAL_CTFFIND_ABORT = """**   Welcome to Ctffind   **

            Version : 5.0.2
           Compiled : Mar 28 2024
    Library Version : 2.0.0-alpha-295-b21db55-dirty
        From Branch : ctffind5_merge
               Mode : Scripted

Input image file name                              : /data/TS_01/TS_01_final_bin4.st
Input is a movie (stack of frames)                 :
 Error: Running as script, and answer is not recognized...
"""


def test_questions_asked_ignores_the_banner() -> None:
    asked = _questions_asked(_REAL_CTFFIND_ABORT)

    # "Version : 5.0.2" and friends read like prompts but are not questions.
    assert asked == [
        ("Input image file name", "/data/TS_01/TS_01_final_bin4.st"),
        ("Input is a movie (stack of frames)", ""),
    ]


def test_verify_dialogue_names_both_questions_on_a_mismatch() -> None:
    """The message has to say which question was asked and which was answered."""

    stale = (
        ("Input image file name", "/data/TS_01/TS_01_final_bin4.st"),
        ("Output diagnostic image file name", "/tmp/diagnostic.mrc"),
        ("Pixel size", "2.700000"),
    )

    with pytest.raises(RuntimeError) as failure:
        _verify_dialogue(_REAL_CTFFIND_ABORT, stale)

    message = str(failure.value)
    assert "Input is a movie (stack of frames)" in message
    assert "Output diagnostic image file name" in message


def test_ctffind5_estimation_clamps_resolution_beyond_nyquist(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")
    stdins: list[str] = []

    artifacts = Ctffind5CtfEstimationBackend(_ctffind_runner(stdins=stdins)).estimate(
        aligned_stack,
        alignment,
        _manifest(),
        _context(
            tmp_path,
            {
                "imod_dir": _fake_imod_dir(tmp_path),
                "ctffind_executable": ctffind,
                "ctf_expected_defocus_um": 3.0,
                "ctffind_max_resolution_angstrom": 3.0,
            },
        ),
    )

    # 2.7 A/pixel gives a 5.4 A Nyquist limit, so a 3 A request is not physical.
    assert float(_answer_to("Maximum resolution", stdins[0])) == pytest.approx(5.4)
    report = CtfEstimationQc.model_validate_json(artifacts[1].path.read_text())
    assert report.max_resolution_angstrom == pytest.approx(5.4)
    assert any("Nyquist" in warning for warning in report.warnings)


def test_ctffind5_estimation_skips_first_pass_with_explicit_defocus(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")
    stdins: list[str] = []

    Ctffind5CtfEstimationBackend(_ctffind_runner(stdins=stdins)).estimate(
        aligned_stack,
        alignment,
        _manifest(),
        _context(
            tmp_path,
            {
                "imod_dir": _fake_imod_dir(tmp_path),
                "ctffind_executable": ctffind,
                "ctf_expected_defocus_um": 4.0,
            },
        ),
    )

    assert len(stdins) == 1, "no hint pass when the defocus is supplied"
    assert float(_answer_to("Minimum defocus", stdins[0])) == pytest.approx(39000.0)
    assert float(_answer_to("Maximum defocus", stdins[0])) == pytest.approx(41000.0)


def test_ctffind5_estimation_can_fit_the_sample_tilt(tmp_path: Path) -> None:
    """A yes to the tilt question draws no further questions, so the list is unchanged."""

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")
    stdins: list[str] = []

    artifacts = Ctffind5CtfEstimationBackend(_ctffind_runner(stdins=stdins)).estimate(
        aligned_stack,
        alignment,
        _manifest(),
        _context(
            tmp_path,
            {
                "imod_dir": _fake_imod_dir(tmp_path),
                "ctffind_executable": ctffind,
                "ctf_expected_defocus_um": 4.0,
                "ctffind_determine_sample_tilt": True,
                "ctffind_defocus_search_window_angstrom": 5000.0,
            },
        ),
    )

    (full,) = stdins
    assert len(full.splitlines()) == len(_STACK_QUESTIONS)
    assert _answer_to("Determine sample tilt?", full) == "yes"
    assert _answer_to("Determine samnple thickness?", full) == "no"
    # The per-view window around the 4 um hint.
    assert float(_answer_to("Minimum defocus", full)) == pytest.approx(35000.0)
    assert float(_answer_to("Maximum defocus", full)) == pytest.approx(45000.0)
    ctf_model = artifacts[0]
    assert ctf_model.parameters["determines_sample_tilt"] is True
    assert ctf_model.parameters["defocus_search_window_angstrom"] == pytest.approx(5000.0)


def test_ctffind5_refuses_sample_tilt_with_a_phase_shift(tmp_path: Path) -> None:
    """CTFFIND5 asks the tilt question only without a phase-shift search."""

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")

    with pytest.raises(ValueError, match="phase shift"):
        Ctffind5CtfEstimationBackend(_ctffind_runner()).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(
                tmp_path,
                {
                    "imod_dir": _fake_imod_dir(tmp_path),
                    "ctffind_executable": ctffind,
                    "ctffind_determine_sample_tilt": True,
                    "ctffind_search_phase_shift": True,
                },
            ),
        )


def _estimate_with_fits(
    tmp_path: Path,
    fit_limits: Sequence[float],
    extra: dict[str, object] | None = None,
) -> tuple[Artifact, CtfEstimationQc]:
    """Fit the two-view test stack (z 2 at -3.1 deg, z 0 at 0.1) with given fit limits."""

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")
    ctf_model, qc_artifact = Ctffind5CtfEstimationBackend(
        _ctffind_runner(fit_limits=fit_limits, defocus_per_view=[41000.0, 30000.0])
    ).estimate(
        aligned_stack,
        alignment,
        _manifest(),
        _context(
            tmp_path,
            {
                "imod_dir": _fake_imod_dir(tmp_path),
                "ctffind_executable": ctffind,
                "ctf_expected_defocus_um": 3.0,
                **(extra or {}),
            },
        ),
    )
    return ctf_model, CtfEstimationQc.model_validate_json(qc_artifact.path.read_text())


def test_an_unfitted_view_is_left_out_when_there_are_few(tmp_path: Path) -> None:
    """At most five views with no Thon rings: correct-ctf drops them, QC warns."""

    ctf_model, report = _estimate_with_fits(tmp_path, [29.8, 7.5])

    assert ctf_model.parameters["z_values"] == [2, 0]
    assert ctf_model.parameters["unfitted_z_values"] == [2]
    assert ctf_model.parameters["excluded_z_values"] == [2]
    assert report.status == QcStatus.WARNING
    assert report.unfitted_z_values == [2]
    assert report.excluded_z_values == [2]
    warning = next(w for w in report.warnings if "no Thon rings" in w)
    assert "-3.1 deg (z 2)" in warning
    assert "leaves them out" in warning

    # The model never carries the failed fit: the view gets the fitted one's
    # defocus (the mean of its 30000 and 29500 A), without astigmatism.
    entries = read_imod_defocus_file(ctf_model.path)
    assert entries[0].defocus_nm == pytest.approx(2975.0)
    assert entries[0].second_defocus_nm == pytest.approx(2975.0)
    assert entries[1].defocus_nm == pytest.approx(3000.0)


def test_unfitted_views_are_interpolated_when_there_are_many(tmp_path: Path) -> None:
    ctf_model, report = _estimate_with_fits(
        tmp_path, [29.8, 7.5], {"ctf_max_excluded_views": 0}
    )

    assert ctf_model.parameters["unfitted_z_values"] == [2]
    assert ctf_model.parameters["interpolated_z_values"] == [2]
    assert ctf_model.parameters["excluded_z_values"] == []
    assert report.status == QcStatus.WARNING
    warning = next(w for w in report.warnings if "no Thon rings" in w)
    assert "they stay in" in warning
    assert read_imod_defocus_file(ctf_model.path)[0].defocus_nm == pytest.approx(2975.0)


def test_a_fit_that_found_rings_somewhere_is_not_unfitted(tmp_path: Path) -> None:
    """A poor fit is still a fit: ryr1 161 reached 16.7 A at +36 deg."""

    ctf_model, report = _estimate_with_fits(tmp_path, [16.7, 7.5])

    assert ctf_model.parameters["unfitted_z_values"] == []
    assert report.status == QcStatus.PASS
    assert read_imod_defocus_file(ctf_model.path)[0].defocus_nm == pytest.approx(4100.0)


def test_estimation_refuses_when_no_view_was_fitted(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="none of the views"):
        _estimate_with_fits(tmp_path, [29.8, 29.7])


def test_interpolation_follows_the_trend_past_the_last_fitted_view() -> None:
    """Linear between fitted views, along the outermost ones' line past them."""

    from cryoet_pipeline.backends.ctf import _interpolate_unfitted
    from cryoet_pipeline.imod_defocus import CtffindRow

    angles = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]

    def truth(angle: float) -> float:
        return 50000.0 - 300.0 * angle  # falling with the tilt, as on ryr1 039

    rows = [
        CtffindRow(
            view=index + 1,
            defocus1_angstrom=truth(angle) + 100.0,
            defocus2_angstrom=truth(angle) - 100.0,
            astigmatism_azimuth_deg=30.0,
            phase_shift_rad=0.0,
            cross_correlation=0.1,
            fit_resolution_angstrom=8.0,
        )
        for index, angle in enumerate(angles)
    ]
    # Garbage where the fit failed: the two outermost views of one side and one
    # in the middle.
    unfitted = [0, 1, 4]
    for index in unfitted:
        rows[index] = CtffindRow(index + 1, 41000.0, 40000.0, 12.0, 0.0, 0.01, 29.8)

    result = _interpolate_unfitted(rows, angles, unfitted)

    for index in unfitted:
        assert result[index].mean_defocus_angstrom == pytest.approx(truth(angles[index]))
        assert result[index].astigmatism_angstrom == pytest.approx(0.0)
    for index in set(range(len(rows))) - set(unfitted):
        assert result[index] == rows[index]


def test_ctfphaseflip_leaves_out_the_views_the_model_excludes(tmp_path: Path) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)
    ctf_model = ctf_model.model_copy(
        update={
            "parameters": {
                **ctf_model.parameters,
                "z_values": [2, 0],
                "unfitted_z_values": [2],
                "excluded_z_values": [2],
            }
        }
    )
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack, ctf_model])
    captured: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        command_list = list(command)
        captured.append(command_list)
        if "newstack" in command_list[0]:
            sections = [
                int(section)
                for section in command_list[command_list.index("-secs") + 1].split(",")
            ]
            source = Path(command_list[command_list.index("-input") + 1])
            with mrcfile.open(source, permissive=True) as full:
                kept = np.array(full.data[sections])
            with mrcfile.new(
                Path(command_list[command_list.index("-output") + 1]), overwrite=True
            ) as output:
                output.set_data(kept)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        output_path = Path(command_list[command_list.index("-OutputFileName") + 1])
        with mrcfile.new(output_path, overwrite=True) as output:
            output.set_data(_CORRECTED_DATA)
        return subprocess.CompletedProcess(command, 0, stdout="flipped", stderr="")

    corrected_stack, qc_artifact = correct_ctf_and_register(
        ImodCtfPhaseFlipBackend(fake_runner),
        aligned_stack,
        ctf_model,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"imod_dir": _fake_imod_dir(tmp_path)},
        ),
        registry,
    )

    # Every view is corrected, then z 2 (section 0) is taken out.
    phaseflip, keep = captured
    assert "-DefocusFile" in phaseflip
    assert keep[keep.index("-secs") + 1] == "1"
    assert corrected_stack.shape == (1, 4, 6)
    with mrcfile.open(corrected_stack.path, permissive=True) as mrc:
        np.testing.assert_allclose(mrc.data, _CORRECTED_DATA[1:2])

    # What reconstruction and picking read the views from.
    assert corrected_stack.parameters["included_z_values"] == [0]
    assert corrected_stack.parameters["excluded_z_values"] == [1, 2]
    assert corrected_stack.parameters["ctf_excluded_z_values"] == [2]
    angles = Path(corrected_stack.parameters["tilt_file_path"]).read_text().split()
    assert [float(angle) for angle in angles] == pytest.approx([0.1])

    report = CtfCorrectionQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.WARNING
    assert report.num_tilts == 1
    assert report.excluded_z_values == [2]
    assert qc_artifact.parameters["status"] == QcStatus.WARNING.value


def test_ctffind5_estimation_rejects_view_count_mismatch(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")

    with pytest.raises(RuntimeError, match="fitted 1 views but the aligned stack has 2"):
        Ctffind5CtfEstimationBackend(_ctffind_runner(rows=1)).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(
                tmp_path,
                {
                    "imod_dir": _fake_imod_dir(tmp_path),
                    "ctffind_executable": ctffind,
                    "ctf_expected_defocus_um": 3.0,
                },
            ),
        )


def test_ctfphaseflip_corrects_from_a_registered_model(tmp_path: Path) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack, ctf_model])
    captured: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        command_list = list(command)
        captured.append(command_list)
        output_path = Path(command_list[command_list.index("-OutputFileName") + 1])
        with mrcfile.new(output_path, overwrite=True) as output:
            output.set_data(_CORRECTED_DATA)
        return subprocess.CompletedProcess(command, 0, stdout="flipped", stderr="")

    artifacts = correct_ctf_and_register(
        ImodCtfPhaseFlipBackend(fake_runner),
        aligned_stack,
        ctf_model,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"imod_dir": _fake_imod_dir(tmp_path)},
        ),
        registry,
    )

    corrected_stack, qc_artifact = artifacts
    assert corrected_stack.kind == ArtifactKind.ALIGNED_TILT_STACK
    assert corrected_stack.id.endswith(":ctf_corrected:imod-ctfphaseflip")
    assert corrected_stack.parent_ids == [aligned_stack.id, ctf_model.id]
    assert corrected_stack.parameters["purpose"] == "ctf_corrected"
    assert corrected_stack.parameters["ctf_estimation_backend"] == "ctffind5"
    assert corrected_stack.parameters["included_z_values"] == [2, 0]
    with mrcfile.open(corrected_stack.path, permissive=True) as mrc:
        np.testing.assert_allclose(mrc.data, _CORRECTED_DATA)

    command = captured[0]
    # Optics must come from the model, and pixel size is passed in nanometres.
    assert command[command.index("-Voltage") + 1] == "300"
    assert command[command.index("-SphericalAberration") + 1] == "2.700000"
    assert command[command.index("-PixelSize") + 1] == "0.270000"

    report = CtfCorrectionQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.PASS
    assert report.ctf_model_id == ctf_model.id
    assert report.mean_defocus_um == pytest.approx(2.9)


def test_ctfphaseflip_rejects_optics_conflicting_with_the_model(tmp_path: Path) -> None:
    _, _, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)

    with pytest.raises(ValueError, match="conflicts with the CTF model"):
        ImodCtfPhaseFlipBackend().correct(
            aligned_stack,
            ctf_model,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "imod_dir": _fake_imod_dir(tmp_path),
                    "ctf_voltage_kv": 200.0,
                },
            ),
        )


def test_ctfphaseflip_applies_a_model_fitted_at_another_binning(tmp_path: Path) -> None:
    """Defocus is physical, so a fit from a finer stack corrects a coarser one.

    Coarse binning is where CTF fitting is least reliable — few Thon rings fall
    below Nyquist — so being able to fit once at the finest binning and apply the
    result elsewhere is the point. The transfer is recorded as a warning, not
    silently accepted, and ctfphaseflip still gets this stack's pixel size.
    """

    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)
    finer = ctf_model.model_copy(
        update={
            "parameters": {
                **ctf_model.parameters,
                "input_stack_id": "TS_TEST:aligned_tilt_stack:final",
                "pixel_spacing_angstrom": 1.35,
            }
        }
    )
    coarse = aligned_stack.model_copy(update={"id": "TS_TEST:aligned_tilt_stack:final:bin8"})
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack, coarse, finer])
    captured: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        command_list = list(command)
        captured.append(command_list)
        with mrcfile.new(
            Path(command_list[command_list.index("-OutputFileName") + 1]), overwrite=True
        ) as output:
            output.set_data(_CORRECTED_DATA)
        return subprocess.CompletedProcess(command, 0, stdout="flipped", stderr="")

    _, qc_artifact = correct_ctf_and_register(
        ImodCtfPhaseFlipBackend(fake_runner),
        coarse,
        finer,
        _manifest(),
        _context(tmp_path, {"imod_dir": _fake_imod_dir(tmp_path)}),
        registry,
    )

    # The stack's own pixel size, not the one the model was fitted at.
    assert captured[0][captured[0].index("-PixelSize") + 1] == "0.270000"

    report = CtfCorrectionQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.WARNING
    assert any("not on" in warning for warning in report.warnings)
    assert any("1.35 angstrom/pixel" in warning for warning in report.warnings)


def test_ctfphaseflip_rejects_a_model_covering_different_views(tmp_path: Path) -> None:
    """Angles are what ties model to stack once the stacks may differ."""

    _, _, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)
    shifted = tmp_path / "shifted.defocus"
    shifted.write_text(
        "0 0 0.0 0.0 0.0 3\n"
        "1 1 -30.000000 -30.000000 3000.000\n"
        "2 2 0.100000 0.100000 2800.000\n"
    )
    foreign = ctf_model.model_copy(
        update={
            "path": shifted,
            "parameters": {**ctf_model.parameters, "input_stack_id": "TS_TEST:other"},
        }
    )

    with pytest.raises(ValueError, match="disagree on view 1"):
        ImodCtfPhaseFlipBackend().correct(
            aligned_stack,
            foreign,
            _manifest(),
            BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU),
        )


def test_ctfphaseflip_rejects_one_stack_with_two_pixel_sizes(tmp_path: Path) -> None:
    """Claiming this stack but another pixel size means one record is wrong."""

    _, _, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)
    inconsistent = ctf_model.model_copy(
        update={"parameters": {**ctf_model.parameters, "pixel_spacing_angstrom": 1.35}}
    )

    with pytest.raises(ValueError, match="angstrom/pixel"):
        ImodCtfPhaseFlipBackend().correct(
            aligned_stack,
            inconsistent,
            _manifest(),
            BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU),
        )


def test_estimation_rejects_ctf_corrected_input(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    already_corrected = aligned_stack.model_copy(
        update={"parameters": {**aligned_stack.parameters, "purpose": "ctf_corrected"}}
    )

    with pytest.raises(ValueError, match="requires the plain final fine-aligned stack"):
        ImodCtfPlotterEstimationBackend().estimate(
            already_corrected,
            alignment,
            _manifest(),
            BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU),
        )


def test_estimation_requires_voltage(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)

    with pytest.raises(TypeError, match="ctf_voltage_kv"):
        ImodCtfPlotterEstimationBackend().estimate(
            aligned_stack,
            alignment,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={"ctf_spherical_aberration_mm": 2.7},
            ),
        )


def _context(tmp_path: Path, extra: dict[str, object]) -> BackendContext:
    return BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "ctf_voltage_kv": 300.0,
            "ctf_spherical_aberration_mm": 2.7,
            **extra,
        },
    )


def _ctf_model_artifact(tmp_path: Path, aligned_stack: Artifact) -> Artifact:
    path = tmp_path / "TS_TEST.defocus"
    path.write_text(
        "0 0 0.0 0.0 0.0 3\n"
        "1 1 -3.100000 -3.100000 3000.000\n"
        "2 2 0.100000 0.100000 2800.000\n"
    )
    return Artifact(
        id="TS_TEST:ctf_model:ctffind5",
        kind=ArtifactKind.CTF_MODEL,
        path=path,
        parent_ids=[aligned_stack.id],
        parameters={
            "backend": "ctffind5",
            "tilt_series_id": "TS_TEST",
            "input_stack_id": aligned_stack.id,
            "voltage_kv": 300.0,
            "spherical_aberration_mm": 2.7,
            "amplitude_contrast": 0.07,
            "pixel_spacing_angstrom": 2.7,
            "axis_angle_deg": 85.3,
        },
    )


def _input_artifacts(tmp_path: Path) -> tuple[Artifact, Artifact, Artifact]:
    source_stack = Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=tmp_path / "source.zarr",
        parameters={"tilt_series_id": "TS_TEST"},
    )
    alignment_result = TiltAlignment(
        tilt_series_id="TS_TEST",
        backend="test",
        stage="fine",
        input_stack_id=source_stack.id,
        input_binning=2,
        tilt_axis_angle_deg=85.3,
        transform_semantics="global",
        transforms=[
            AlignmentTransform(
                z_value=z_value,
                tilt_angle_deg=angle,
                a11=1.0,
                a12=0.0,
                a21=0.0,
                a22=1.0,
                shift_x_px=0.0,
                shift_y_px=0.0,
            )
            for z_value, angle in [(2, _TILT_ANGLES[0]), (0, _TILT_ANGLES[1])]
        ],
        excluded_z_values=[1],
    )
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(alignment_result.model_dump_json())
    alignment = Artifact(
        id="TS_TEST:alignment:fine",
        kind=ArtifactKind.ALIGNMENT,
        path=alignment_path,
        parent_ids=[source_stack.id],
        parameters={"tilt_series_id": "TS_TEST", "stage": "fine"},
    )
    aligned_path = tmp_path / "aligned.st"
    with mrcfile.new(aligned_path, overwrite=True) as aligned:
        aligned.set_data(_CORRECTED_DATA)
        aligned.voxel_size = 2.7
    solved_tilts = tmp_path / "final.tlt"
    solved_tilts.write_text("".join(f"{angle}\n" for angle in _TILT_ANGLES))
    aligned_stack = Artifact(
        id="TS_TEST:aligned_tilt_stack:final",
        kind=ArtifactKind.ALIGNED_TILT_STACK,
        path=aligned_path,
        parent_ids=[source_stack.id, alignment.id],
        shape=(2, 4, 6),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=2.7,
        binning=2,
        parameters={
            "purpose": "final_alignment",
            "alignment_stage": "fine",
            "tilt_series_id": "TS_TEST",
            "included_z_values": [2, 0],
            "excluded_z_values": [1],
            "order": "tilt_angle_ascending",
            "tilt_file_path": str(solved_tilts),
        },
    )
    return source_stack, alignment, aligned_stack


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=z_value,
                tilt_angle_deg=angle,
                subframe_path=f"frame_{z_value}.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
                defocus_um=3.0,
            )
            for z_value, angle in [(0, 0.0), (1, 3.0), (2, -3.0)]
        ],
    )


def _fake_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def _fake_imod_dir(tmp_path: Path) -> Path:
    imod_dir = tmp_path / "imod"
    for name in ("ctfplotter", "ctfphaseflip", "newstack"):
        executable = imod_dir / "bin" / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    return imod_dir


def test_ctffind5_estimation_keeps_evidence_when_a_view_fails(tmp_path: Path) -> None:
    """A failed run must leave the extracted view and the exact stdin behind.

    Cleaning up on failure destroys the only two things worth looking at.
    """

    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind")

    def failing_runner(command, *, cwd: Path, env, stdin: str | None = None):
        command_list = list(command)
        if "newstack" in command_list[0]:
            Path(command_list[-1]).write_bytes(b"extracted view")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command, 1, stdout="", stderr="Error: answer is not recognized"
        )

    out = tmp_path / "outputs"
    with pytest.raises(RuntimeError, match="CTFFIND failed with exit code 1"):
        Ctffind5CtfEstimationBackend(failing_runner).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(
                tmp_path,
                {
                    "imod_dir": _fake_imod_dir(tmp_path),
                    "ctffind_executable": ctffind,
                    "ctf_expected_defocus_um": 3.0,
                },
            ),
        )

    leftovers = list((out / "ctf" / "TS_TEST").glob(".TS_TEST-ctffind-*"))
    assert leftovers, "the working directory must survive a failure"

    log = (out / "ctf" / "TS_TEST" / "TS_TEST_ctffind5.log").read_text()
    assert "<<stdin>>" in log, "the exact stdin belongs in the log"
    assert "answer is not recognized" in log
