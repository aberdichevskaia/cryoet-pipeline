from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.reconstruction import (
    ImodTiltReconstructionBackend,
    reconstruct_and_register,
)
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
    TomogramQc,
)
from cryoet_pipeline.runtime import DevicePreference


def test_imod_reconstruction_writes_canonical_zarr_rec_and_qc(
    tmp_path: Path,
) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    imod_dir = _fake_imod_dir(tmp_path)
    imod_yzx = np.arange(48, dtype=np.float32).reshape(4, 2, 6)
    captured_command: list[str] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        captured_command.extend(command)
        tilt_path = Path(command[command.index("-TILTFILE") + 1])
        output_path = Path(command[command.index("-output") + 1])
        assert tilt_path.read_text() == "-3.100000\n0.100000\n"
        assert command[command.index("-THICKNESS") + 1] == "2"
        assert command[command.index("-XAXISTILT") + 1] == "-1.600000"
        assert command[command.index("-SHIFT") + 1] == "0.000000,0.000000"
        assert command[command.index("-RADIAL") + 1] == "0.350000,0.050000"
        with mrcfile.new(output_path, overwrite=True) as rec:
            rec.set_data(imod_yzx)
        return subprocess.CompletedProcess(command, 0, stdout="reconstructed", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "imod_dir": imod_dir,
        },
    )

    artifacts = reconstruct_and_register(
        ImodTiltReconstructionBackend(fake_runner),
        aligned_stack,
        alignment,
        manifest,
        context,
        registry,
    )

    tomogram, qc_artifact = artifacts
    assert registry.get(tomogram.id) == tomogram
    assert registry.get(qc_artifact.id) == qc_artifact
    assert tomogram.kind == ArtifactKind.TOMOGRAM
    assert tomogram.parent_ids == [aligned_stack.id, alignment.id]
    assert tomogram.shape == (2, 4, 6)
    assert tomogram.axis_order == AxisOrder.ZYX
    assert tomogram.pixel_spacing_angstrom == pytest.approx(2.7)
    assert tomogram.parameters["ctf_corrected"] is False
    assert tomogram.parameters["included_z_values"] == [2, 0]
    assert tomogram.parameters["alignment_stage"] == "fine"
    assert tomogram.parameters["thickness"] == 2
    assert tomogram.parameters["x_axis_tilt_deg"] == pytest.approx(-1.6)
    assert tomogram.parameters["z_shift_px"] == pytest.approx(0.0)
    assert tomogram.parameters["imod_rec_axis_order"] == "yzx"
    assert captured_command

    canonical = zarr.open(tomogram.path, mode="r")
    np.testing.assert_allclose(canonical[:], imod_yzx.transpose(1, 0, 2))
    assert canonical.attrs["axis_order"] == "zyx"
    assert canonical.attrs["source_axis_order"] == "yzx"

    imod_rec_path = Path(tomogram.parameters["imod_rec_path"])
    with mrcfile.open(imod_rec_path, permissive=True) as rec:
        np.testing.assert_allclose(rec.data, imod_yzx)
        assert float(rec.voxel_size.x) == pytest.approx(2.7)

    report = TomogramQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.WARNING
    assert report.shape == (2, 4, 6)
    assert report.minimum == pytest.approx(float(imod_yzx.min()))
    assert report.maximum == pytest.approx(float(imod_yzx.max()))
    assert report.mean == pytest.approx(float(imod_yzx.mean()))
    assert report.standard_deviation == pytest.approx(float(imod_yzx.std()))
    assert report.ctf_corrected is False
    assert report.alignment_stage == "fine"
    assert report.warnings == ["CTF correction was not applied"]
    assert all(path.is_file() for path in report.central_slice_paths.values())
    assert not list(tomogram.path.parent.glob(".TS_TEST-reconstruct-*"))


def test_imod_reconstruction_does_not_register_external_failure(
    tmp_path: Path,
) -> None:
    source_stack, alignment, aligned_stack = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    imod_dir = _fake_imod_dir(tmp_path)

    def failing_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        return subprocess.CompletedProcess(command, 4, stdout="", stderr="failed")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "thickness": 2,
            "imod_dir": imod_dir,
        },
    )

    with pytest.raises(RuntimeError, match="IMOD tilt failed with exit code 4"):
        reconstruct_and_register(
            ImodTiltReconstructionBackend(failing_runner),
            aligned_stack,
            alignment,
            _manifest(),
            context,
            registry,
        )

    assert len(registry.by_kind(ArtifactKind.TOMOGRAM)) == 0
    assert not (tmp_path / "outputs/tomograms/TS_TEST/TS_TEST_fine_bin2.rec").exists()


def test_imod_reconstruction_rejects_coarse_preview(tmp_path: Path) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    coarse_stack = aligned_stack.model_copy(
        update={
            "parameters": {
                **aligned_stack.parameters,
                "purpose": "coarse_alignment_preview",
            }
        }
    )

    with pytest.raises(ValueError, match="requires a final fine-aligned stack"):
        ImodTiltReconstructionBackend().reconstruct(
            coarse_stack,
            alignment,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
            ),
        )


def test_imod_reconstruction_requires_positioned_fine_alignment(
    tmp_path: Path,
) -> None:
    _, alignment, aligned_stack = _input_artifacts(tmp_path)
    unpositioned = alignment.model_copy(
        update={
            "parameters": {
                key: value
                for key, value in alignment.parameters.items()
                if key != "axis_z_shift_applied_in_alignment"
            }
        }
    )

    with pytest.raises(ValueError, match="must apply AxisZShift"):
        ImodTiltReconstructionBackend().reconstruct(
            aligned_stack,
            unpositioned,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
            ),
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
            "positioning_source": "tiltalign_two_surface_analysis",
            "recommended_x_axis_tilt_deg": -1.6,
            "recommended_unbinned_thickness_px": 4.0,
            "recommended_unbinned_z_shift_px": 8.0,
            "axis_z_shift_applied_in_alignment": True,
        },
    )
    solved_tilts = tmp_path / "fine.tlt"
    solved_tilts.write_text("-3.1\n0.1\n")
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
            )
            for z_value, angle in [(0, 0.0), (1, 3.0), (2, -3.0)]
        ],
    )


def _fake_imod_dir(tmp_path: Path) -> Path:
    imod_dir = tmp_path / "imod"
    executable = imod_dir / "bin" / "tilt"
    executable.parent.mkdir(parents=True)
    executable.touch()
    return imod_dir
