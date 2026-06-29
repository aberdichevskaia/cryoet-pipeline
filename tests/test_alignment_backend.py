from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    ImodTiltXcorrAlignmentBackend,
    align_and_register,
    parse_imod_xf,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    StorageRole,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_imod_tiltxcorr_backend_normalizes_and_registers_transforms(
    tmp_path: Path,
) -> None:
    stack_data = np.arange(48, dtype=np.float32).reshape(2, 4, 6)
    stack_artifact = _zarr_stack_artifact(tmp_path, stack_data)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(stack_artifact)
    executable = _fake_imod_executable(tmp_path)
    captured: dict[str, object] = {}

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = list(command)
        captured["env"] = dict(env)
        input_path = Path(command[command.index("-input") + 1])
        output_path = Path(command[command.index("-output") + 1])
        tilt_path = Path(command[command.index("-tiltfile") + 1])
        rotation = command[command.index("-rotation") + 1]

        with mrcfile.open(input_path, permissive=True) as binned:
            expected = stack_data[[1, 0]].reshape(2, 2, 2, 3, 2).mean(
                axis=(2, 4),
                dtype=np.float32,
            )
            np.testing.assert_allclose(binned.data, expected)
            assert float(binned.voxel_size.x) == pytest.approx(2.7)
        assert tilt_path.read_text() == "-3.000000\n3.000000\n"
        assert rotation == "85.300000"
        assert cwd == input_path.parent

        output_path.write_text(
            "1.0 0.0 0.0 1.0 1.5 -2.0\n"
            "0.99 0.01 -0.01 1.01 -3.0 4.0\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="aligned", stderr="")

    context = _context(
        tmp_path,
        executable,
        parameters={"binning": 2},
    )
    backend = ImodTiltXcorrAlignmentBackend(fake_runner)

    artifact = align_and_register(
        backend,
        stack_artifact,
        manifest,
        context,
        registry,
    )

    assert registry.get(artifact.id) == artifact
    assert artifact.kind == ArtifactKind.ALIGNMENT
    assert artifact.parent_ids == [stack_artifact.id]
    assert artifact.storage_role == StorageRole.CANONICAL
    assert artifact.shape == (2, 6)
    assert artifact.parameters["input_binning"] == 2
    assert artifact.parameters["tilt_axis_angle_deg"] == pytest.approx(85.3)

    alignment = TiltAlignment.model_validate_json(artifact.path.read_text())
    assert alignment.input_stack_id == stack_artifact.id
    assert alignment.stage == "coarse"
    assert alignment.transforms[0].shift_x_px == pytest.approx(-6.0)
    assert alignment.transforms[0].shift_y_px == pytest.approx(8.0)
    assert alignment.transforms[1].shift_x_px == pytest.approx(3.0)
    assert alignment.transforms[1].shift_y_px == pytest.approx(-4.0)

    imod_xf_path = Path(artifact.parameters["imod_xf_path"])
    exported = parse_imod_xf(imod_xf_path)
    assert exported[0][4:] == pytest.approx((-6.0, 8.0))
    assert exported[1][4:] == pytest.approx((3.0, -4.0))
    assert "aligned" in Path(artifact.parameters["log_path"]).read_text()
    assert captured["command"]
    assert captured["env"]["IMOD_DIR"] == str(executable.parent.parent)
    assert not list(artifact.path.parent.glob(".TS_TEST-tiltxcorr-*"))


def test_imod_tiltxcorr_backend_rejects_wrong_transform_count(
    tmp_path: Path,
) -> None:
    stack_artifact = _zarr_stack_artifact(
        tmp_path,
        np.arange(32, dtype=np.float32).reshape(2, 4, 4),
    )
    executable = _fake_imod_executable(tmp_path)

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        output_path = Path(command[command.index("-output") + 1])
        output_path.write_text("1 0 0 1 0 0\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = _context(tmp_path, executable, parameters={"binning": 2})

    with pytest.raises(ValueError, match="1 transforms for 2 tilts"):
        ImodTiltXcorrAlignmentBackend(fake_runner).align(
            stack_artifact,
            _manifest(),
            context,
        )

    assert not (tmp_path / "outputs/alignments/TS_TEST/TS_TEST_coarse_alignment.json").exists()


def test_imod_tiltxcorr_backend_reports_external_failure(tmp_path: Path) -> None:
    stack_artifact = _zarr_stack_artifact(
        tmp_path,
        np.arange(32, dtype=np.float32).reshape(2, 4, 4),
    )
    executable = _fake_imod_executable(tmp_path)

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="bad input")

    context = _context(tmp_path, executable, parameters={"binning": 2})

    with pytest.raises(RuntimeError, match="exit code 2"):
        ImodTiltXcorrAlignmentBackend(fake_runner).align(
            stack_artifact,
            _manifest(),
            context,
        )

    log_path = tmp_path / "outputs/alignments/TS_TEST/TS_TEST_coarse_tiltxcorr.log"
    assert "bad input" in log_path.read_text()


def test_imod_tiltxcorr_backend_skips_low_variance_tilt(tmp_path: Path) -> None:
    stack_data = np.stack(
        [
            np.arange(16, dtype=np.float32).reshape(4, 4),
            np.zeros((4, 4), dtype=np.float32),
        ]
    )
    stack_artifact = _zarr_stack_artifact(tmp_path, stack_data)
    executable = _fake_imod_executable(tmp_path)
    captured_command: list[str] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        captured_command.extend(command)
        output_path = Path(command[command.index("-output") + 1])
        output_path.write_text("1 0 0 1 0 0\n1 0 0 1 1 -1\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = _context(tmp_path, executable, parameters={"binning": 2})
    artifact = ImodTiltXcorrAlignmentBackend(fake_runner).align(
        stack_artifact,
        _manifest(),
        context,
    )

    assert captured_command[captured_command.index("-skip") + 1] == "1"
    alignment = TiltAlignment.model_validate_json(artifact.path.read_text())
    assert alignment.excluded_z_values == [1]
    assert alignment.input_projection_std[0] > 0
    assert alignment.input_projection_std[1] == 0


def test_parse_imod_xf_rejects_malformed_or_nonfinite_rows(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.xf"
    malformed.write_text("1 0 0 1 2\n")
    nonfinite = tmp_path / "nonfinite.xf"
    nonfinite.write_text("1 0 0 1 nan 2\n")

    with pytest.raises(ValueError, match="expected 6"):
        parse_imod_xf(malformed)
    with pytest.raises(ValueError, match="must be finite"):
        parse_imod_xf(nonfinite)


def _zarr_stack_artifact(tmp_path: Path, data: np.ndarray) -> Artifact:
    path = tmp_path / "TS_TEST.zarr"
    zarr.save(path, data)
    return Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=path,
        shape=tuple(int(axis_size) for axis_size in data.shape),
        dtype=str(data.dtype),
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=1.35,
        parameters={"tilt_series_id": "TS_TEST"},
    )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=3.0,
                subframe_path="TS_TEST_000.mrc",
                num_subframes=2,
                pixel_spacing_angstrom=1.35,
                binning=1,
                rotation_angle_deg=175.3,
            ),
            TiltImage(
                z_value=1,
                tilt_angle_deg=-3.0,
                subframe_path="TS_TEST_001.mrc",
                num_subframes=2,
                pixel_spacing_angstrom=1.35,
                binning=1,
                rotation_angle_deg=175.3,
            ),
        ],
    )


def _fake_imod_executable(tmp_path: Path) -> Path:
    executable = tmp_path / "imod" / "bin" / "tiltxcorr"
    executable.parent.mkdir(parents=True)
    executable.touch()
    return executable


def _context(
    tmp_path: Path,
    executable: Path,
    *,
    parameters: dict[str, object],
) -> BackendContext:
    return BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            **parameters,
            "tiltxcorr_executable": executable,
            "imod_dir": executable.parent.parent,
        },
    )
