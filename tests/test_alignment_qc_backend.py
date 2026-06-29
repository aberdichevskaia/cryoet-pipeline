from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import parse_imod_xf
from cryoet_pipeline.backends.alignment_qc import (
    ImodCoarseAlignmentQcBackend,
    evaluate_coarse_alignment_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    CoarseAlignmentQc,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_coarse_alignment_qc_writes_ordered_preview_and_report(
    tmp_path: Path,
) -> None:
    stack_data = np.arange(192, dtype=np.float32).reshape(3, 8, 8)
    stack_artifact = _stack_artifact(tmp_path, stack_data)
    alignment_artifact = _alignment_artifact(tmp_path, stack_artifact)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([stack_artifact, alignment_artifact])
    imod_dir = _fake_imod_dir(tmp_path)
    commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        commands.append(list(command))
        program = Path(command[0]).name
        if program == "newstack":
            input_path = Path(command[command.index("-input") + 1])
            output_path = Path(command[command.index("-output") + 1])
            transform_path = Path(command[command.index("-xform") + 1])
            with mrcfile.open(input_path, permissive=True) as input_stack:
                assert input_stack.data.shape == (2, 4, 4)
                expected = stack_data[[2, 0]].reshape(2, 4, 2, 4, 2).mean(
                    axis=(2, 4),
                    dtype=np.float32,
                )
                np.testing.assert_allclose(input_stack.data, expected)
            transforms = parse_imod_xf(transform_path)
            assert transforms[0][4:] == pytest.approx((-1.0, 1.0))
            assert transforms[1][4:] == pytest.approx((2.0, -1.0))
            shutil.copyfile(input_path, output_path)
            return subprocess.CompletedProcess(command, 0, stdout="preview", stderr="")

        assert program == "tiltxcorr"
        output_path = Path(command[command.index("-output") + 1])
        tilt_path = Path(command[command.index("-tiltfile") + 1])
        assert tilt_path.read_text() == "-3.000000\n0.000000\n"
        output_path.write_text("1 0 0 1 0 0\n1 0 0 1 0.5 -0.25\n")
        return subprocess.CompletedProcess(command, 0, stdout="residual", stderr="")

    context = _context(tmp_path, imod_dir, parameters={"preview_binning": 2})
    artifacts = evaluate_coarse_alignment_and_register(
        ImodCoarseAlignmentQcBackend(fake_runner),
        stack_artifact,
        alignment_artifact,
        manifest,
        context,
        registry,
    )

    preview_artifact, report_artifact = artifacts
    assert registry.get(preview_artifact.id) == preview_artifact
    assert registry.get(report_artifact.id) == report_artifact
    assert preview_artifact.kind == ArtifactKind.QC
    assert preview_artifact.shape == (2, 4, 4)
    assert preview_artifact.axis_order == AxisOrder.TYX
    assert preview_artifact.binning == 2
    assert preview_artifact.parameters["included_z_values"] == [2, 0]
    assert report_artifact.parent_ids == [
        stack_artifact.id,
        alignment_artifact.id,
        preview_artifact.id,
    ]

    report = CoarseAlignmentQc.model_validate_json(report_artifact.path.read_text())
    assert report.status == QcStatus.WARNING
    assert report.included_z_values == [2, 0]
    assert report.excluded_z_values == [1]
    assert report.residual_shift_x_px == [0.0, 0.5]
    assert report.residual_shift_y_px == [0.0, -0.25]
    assert report.residual_max_px == pytest.approx(np.hypot(0.5, -0.25))
    assert report.preview_path == preview_artifact.path
    assert len(commands) == 2
    assert not list(preview_artifact.path.parent.glob(".TS_TEST-coarse-qc-*"))


def test_coarse_alignment_qc_does_not_register_failed_newstack(
    tmp_path: Path,
) -> None:
    stack_artifact = _stack_artifact(
        tmp_path,
        np.arange(192, dtype=np.float32).reshape(3, 8, 8),
    )
    alignment_artifact = _alignment_artifact(tmp_path, stack_artifact)
    registry = ArtifactRegistry.empty()
    registry.extend([stack_artifact, alignment_artifact])
    imod_dir = _fake_imod_dir(tmp_path)

    def failing_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="failed")

    context = _context(tmp_path, imod_dir, parameters={"preview_binning": 2})

    with pytest.raises(RuntimeError, match="newstack failed"):
        evaluate_coarse_alignment_and_register(
            ImodCoarseAlignmentQcBackend(failing_runner),
            stack_artifact,
            alignment_artifact,
            _manifest(),
            context,
            registry,
        )

    assert len(registry.by_kind(ArtifactKind.QC)) == 0
    assert not (
        tmp_path
        / "outputs/qc/TS_TEST/coarse_alignment/TS_TEST_coarse_aligned_bin2.st"
    ).exists()


def _stack_artifact(tmp_path: Path, data: np.ndarray) -> Artifact:
    path = tmp_path / "TS_TEST.zarr"
    zarr.save(path, data)
    return Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=path,
        shape=tuple(int(axis_size) for axis_size in data.shape),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=1.35,
        parameters={"tilt_series_id": "TS_TEST"},
    )


def _alignment_artifact(tmp_path: Path, stack: Artifact) -> Artifact:
    result = TiltAlignment(
        tilt_series_id="TS_TEST",
        backend="test",
        stage="coarse",
        input_stack_id=stack.id,
        input_binning=2,
        tilt_axis_angle_deg=85.3,
        transforms=[
            AlignmentTransform(
                z_value=0,
                tilt_angle_deg=0.0,
                a11=1.0,
                a12=0.0,
                a21=0.0,
                a22=1.0,
                shift_x_px=4.0,
                shift_y_px=-2.0,
            ),
            AlignmentTransform(
                z_value=1,
                tilt_angle_deg=3.0,
                a11=1.0,
                a12=0.0,
                a21=0.0,
                a22=1.0,
                shift_x_px=100.0,
                shift_y_px=100.0,
            ),
            AlignmentTransform(
                z_value=2,
                tilt_angle_deg=-3.0,
                a11=1.0,
                a12=0.0,
                a21=0.0,
                a22=1.0,
                shift_x_px=-2.0,
                shift_y_px=2.0,
            ),
        ],
        excluded_z_values=[1],
    )
    path = tmp_path / "alignment.json"
    path.write_text(result.model_dump_json())
    return Artifact(
        id="TS_TEST:alignment:coarse",
        kind=ArtifactKind.ALIGNMENT,
        path=path,
        parent_ids=[stack.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "stage": "coarse",
        },
    )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frame_0.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
            TiltImage(
                z_value=1,
                tilt_angle_deg=3.0,
                subframe_path="frame_1.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
            TiltImage(
                z_value=2,
                tilt_angle_deg=-3.0,
                subframe_path="frame_2.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
        ],
    )


def _fake_imod_dir(tmp_path: Path) -> Path:
    imod_dir = tmp_path / "imod"
    bin_dir = imod_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "newstack").touch()
    (bin_dir / "tiltxcorr").touch()
    return imod_dir


def _context(
    tmp_path: Path,
    imod_dir: Path,
    *,
    parameters: dict[str, object],
) -> BackendContext:
    return BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            **parameters,
            "imod_dir": imod_dir,
        },
    )
