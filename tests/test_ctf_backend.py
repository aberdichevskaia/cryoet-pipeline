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


def test_ctffind5_estimation_runs_two_passes_and_writes_defocus(tmp_path: Path) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    ctffind = _fake_executable(tmp_path / "ctffind5")
    imod_dir = _fake_imod_dir(tmp_path)
    stdins: list[str] = []
    newstack_commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        command_list = list(command)
        if "newstack" in command_list[0]:
            newstack_commands.append(command_list)
            Path(command_list[-1]).write_bytes(b"")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        assert stdin is not None
        stdins.append(stdin)
        answers = stdin.splitlines()
        diagnostic = Path(answers[2])
        summary = diagnostic.with_suffix(".txt")
        # A single-view first pass, then one row per view for the full stack.
        rows = 1 if "zero_tilt" in diagnostic.name else 2
        summary.write_text(
            "# CTFFIND fake output\n"
            + "".join(
                f"{index + 1}.0 {30000.0 + index * 500} {29500.0 + index * 500} "
                f"-43.2 0.0 0.10 18.5\n"
                for index in range(rows)
            )
        )
        diagnostic.write_bytes(b"diagnostic")
        return subprocess.CompletedProcess(command, 0, stdout="fitted", stderr="")

    artifacts = estimate_ctf_and_register(
        Ctffind5CtfEstimationBackend(fake_runner),
        aligned_stack,
        alignment,
        _manifest(),
        _context(tmp_path, {"imod_dir": imod_dir, "ctffind_executable": ctffind}),
        registry,
        replace_existing=False,
    )

    ctf_model, qc_artifact = artifacts
    assert ctf_model.id == "TS_TEST:ctf_model:ctffind5"
    assert ctf_model.parameters["astigmatism_frame"] == "aligned_stack"

    # Pass one extracts the view nearest zero tilt; angles are [-3.1, 0.1].
    assert len(newstack_commands) == 1
    assert newstack_commands[0][newstack_commands[0].index("-secs") + 1] == "1"
    assert len(stdins) == 2

    # Pass two searches a narrow window around the hint of 29750 A.
    second = stdins[1].splitlines()
    assert float(second[10]) == pytest.approx(28750.0)
    assert float(second[11]) == pytest.approx(30750.0)
    assert second[1] == "no"  # not a movie, so each slice is fitted separately

    entries = read_imod_defocus_file(ctf_model.path)
    assert len(entries) == 2
    assert entries[0].angle_start_deg == pytest.approx(-3.1)
    assert entries[1].angle_start_deg == pytest.approx(0.1)
    assert entries[0].defocus_nm == pytest.approx(3000.0)
    assert entries[0].second_defocus_nm == pytest.approx(2950.0)
    assert Path(ctf_model.parameters["diagnostic_path"]).is_file()

    report = CtfEstimationQc.model_validate_json(qc_artifact.path.read_text())
    assert report.hint_defocus_um == pytest.approx(2.975)
    assert report.nyquist_angstrom == pytest.approx(5.4)
    assert report.mean_defocus_um == pytest.approx(3.0)


def test_ctffind5_estimation_clamps_resolution_beyond_nyquist(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind5")
    stdins: list[str] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        assert stdin is not None
        stdins.append(stdin)
        diagnostic = Path(stdin.splitlines()[2])
        diagnostic.with_suffix(".txt").write_text(
            "1.0 30000.0 29500.0 -43.2 0.0 0.10 18.5\n2.0 30000.0 29500.0 -43.2 0.0 0.1 18.5\n"
        )
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    artifacts = Ctffind5CtfEstimationBackend(fake_runner).estimate(
        aligned_stack,
        alignment,
        _manifest(),
        _context(
            tmp_path,
            {
                "ctffind_executable": ctffind,
                "ctf_expected_defocus_um": 3.0,
                "ctffind_max_resolution_angstrom": 3.0,
            },
        ),
    )

    # 2.7 A/pixel gives a 5.4 A Nyquist limit, so a 3 A request is not physical.
    assert float(stdins[0].splitlines()[9]) == pytest.approx(5.4)
    report = CtfEstimationQc.model_validate_json(artifacts[1].path.read_text())
    assert report.max_resolution_angstrom == pytest.approx(5.4)
    assert any("Nyquist" in warning for warning in report.warnings)


def test_ctffind5_estimation_skips_first_pass_with_explicit_defocus(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind5")
    calls: list[str] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        assert stdin is not None
        calls.append(stdin)
        diagnostic = Path(stdin.splitlines()[2])
        diagnostic.with_suffix(".txt").write_text(
            "1.0 30000.0 29500.0 -43.2 0.0 0.1 18.5\n2.0 30000.0 29500.0 -43.2 0.0 0.1 18.5\n"
        )
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    Ctffind5CtfEstimationBackend(fake_runner).estimate(
        aligned_stack,
        alignment,
        _manifest(),
        _context(tmp_path, {"ctffind_executable": ctffind, "ctf_expected_defocus_um": 4.0}),
    )

    assert len(calls) == 1
    answers = calls[0].splitlines()
    assert float(answers[10]) == pytest.approx(39000.0)
    assert float(answers[11]) == pytest.approx(41000.0)


def test_ctffind5_estimation_rejects_view_count_mismatch(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    ctffind = _fake_executable(tmp_path / "ctffind5")

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        assert stdin is not None
        diagnostic = Path(stdin.splitlines()[2])
        diagnostic.with_suffix(".txt").write_text("1.0 30000.0 29500.0 -43.2 0.0 0.1 18.5\n")
        return subprocess.CompletedProcess(list(command), 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="fitted 1 views but the aligned stack has 2"):
        Ctffind5CtfEstimationBackend(fake_runner).estimate(
            aligned_stack,
            alignment,
            _manifest(),
            _context(tmp_path, {"ctffind_executable": ctffind, "ctf_expected_defocus_um": 3.0}),
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


def test_ctfphaseflip_rejects_a_model_from_another_stack(tmp_path: Path) -> None:
    _, _, aligned_stack = _input_artifacts(tmp_path)
    ctf_model = _ctf_model_artifact(tmp_path, aligned_stack)
    foreign = ctf_model.model_copy(
        update={"parameters": {**ctf_model.parameters, "input_stack_id": "TS_TEST:other"}}
    )

    with pytest.raises(ValueError, match="was fitted on"):
        ImodCtfPhaseFlipBackend().correct(
            aligned_stack,
            foreign,
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
