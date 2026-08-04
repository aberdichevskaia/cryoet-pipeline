from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.ctf import (
    ImodCtfPlotterBackend,
    _parse_defocus_file,
    correct_ctf_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    CtfEstimationQc,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_imod_ctf_correction_writes_model_stack_and_qc(tmp_path: Path) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    imod_dir = _fake_imod_dir(tmp_path)
    corrected_data = np.arange(48, dtype=np.float32).reshape(2, 4, 6)
    captured_commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        command_list = list(command)
        captured_commands.append(command_list)
        if "ctfplotter" in command_list[0]:
            defocus_path = Path(command_list[command_list.index("-DefocusFile") + 1])
            defocus_path.write_text(
                "1 1 -3.100000 -3.100000 3000.000000 3\n2 2 0.100000 0.100000 2800.000000\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout="plotted", stderr="")

        output_path = Path(command_list[command_list.index("-OutputFileName") + 1])
        with mrcfile.new(output_path, overwrite=True) as output:
            output.set_data(corrected_data)
        return subprocess.CompletedProcess(command, 0, stdout="flipped", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "imod_dir": imod_dir,
            "ctf_voltage_kv": 300.0,
            "ctf_spherical_aberration_mm": 2.7,
        },
    )

    artifacts = correct_ctf_and_register(
        ImodCtfPlotterBackend(fake_runner),
        aligned_stack,
        alignment,
        manifest,
        context,
        registry,
    )

    ctf_model, corrected_stack, qc_artifact = artifacts
    assert registry.get(ctf_model.id) == ctf_model
    assert registry.get(corrected_stack.id) == corrected_stack
    assert registry.get(qc_artifact.id) == qc_artifact

    assert ctf_model.kind == ArtifactKind.CTF_MODEL
    assert ctf_model.parent_ids == [aligned_stack.id, alignment.id]
    assert ctf_model.parameters["axis_angle_deg"] == pytest.approx(85.3)

    assert corrected_stack.kind == ArtifactKind.ALIGNED_TILT_STACK
    assert corrected_stack.parent_ids == [aligned_stack.id, ctf_model.id]
    assert corrected_stack.parameters["purpose"] == "ctf_corrected"
    assert corrected_stack.parameters["alignment_stage"] == "fine"
    assert corrected_stack.parameters["included_z_values"] == [2, 0]
    assert corrected_stack.shape == (2, 4, 6)
    with mrcfile.open(corrected_stack.path, permissive=True) as mrc:
        np.testing.assert_allclose(mrc.data, corrected_data)

    plot_command = captured_commands[0]
    assert plot_command[plot_command.index("-AxisAngle") + 1] == "85.300000"
    assert plot_command[plot_command.index("-Voltage") + 1] == "300"
    assert plot_command[plot_command.index("-SphericalAberration") + 1] == "2.700000"
    assert plot_command[-1] == "-SaveAndExit"

    flip_command = captured_commands[1]
    assert flip_command[flip_command.index("-Voltage") + 1] == "300"

    report = CtfEstimationQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.PASS
    assert report.input_stack_id == aligned_stack.id
    assert report.mean_defocus_um == pytest.approx(2.9)
    assert report.minimum_defocus_um == pytest.approx(2.8)
    assert report.maximum_defocus_um == pytest.approx(3.0)


def test_imod_ctf_correction_rejects_ctf_corrected_input(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    already_corrected = aligned_stack.model_copy(
        update={
            "parameters": {**aligned_stack.parameters, "purpose": "ctf_corrected"},
        }
    )

    with pytest.raises(ValueError, match="requires the plain final fine-aligned stack"):
        ImodCtfPlotterBackend().correct(
            already_corrected,
            alignment,
            _manifest(),
            BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU),
        )


def test_imod_ctf_correction_requires_voltage(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)

    with pytest.raises(TypeError, match="ctf_voltage_kv"):
        ImodCtfPlotterBackend().correct(
            aligned_stack,
            alignment,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={"ctf_spherical_aberration_mm": 2.7},
            ),
        )


def test_parse_defocus_file_plain_format(tmp_path: Path) -> None:
    path = tmp_path / "TS_TEST.defocus"
    path.write_text("1 1 -3.100000 -3.100000 3000.000000 3\n2 2 0.100000 0.100000 2800.000000\n")

    assert _parse_defocus_file(path) == pytest.approx([3000.0, 2800.0])


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
            for z_value, angle in [(2, -3.1), (0, 0.1)]
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
        parameters={
            "tilt_series_id": "TS_TEST",
            "stage": "fine",
        },
    )
    aligned_path = tmp_path / "aligned.st"
    with mrcfile.new(aligned_path, overwrite=True) as aligned:
        aligned.set_data(np.arange(48, dtype=np.float32).reshape(2, 4, 6))
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


def _fake_imod_dir(tmp_path: Path) -> Path:
    imod_dir = tmp_path / "imod"
    for name in ("ctfplotter", "ctfphaseflip"):
        executable = imod_dir / "bin" / name
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    return imod_dir
