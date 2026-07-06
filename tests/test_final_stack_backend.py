from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import parse_imod_xf
from cryoet_pipeline.backends.final_stack import (
    ImodFinalAlignedStackBackend,
    build_final_stack_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_final_stack_backend_applies_fine_transforms_in_solved_order(
    tmp_path: Path,
) -> None:
    source_data = np.arange(192, dtype=np.float32).reshape(4, 8, 6)
    source, fine_alignment = _input_artifacts(tmp_path, source_data)
    registry = ArtifactRegistry.empty()
    registry.extend([source, fine_alignment])
    imod_dir = _fake_imod_dir(tmp_path)

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert env["IMOD_DIR"] == str(imod_dir)
        input_path = Path(command[command.index("-input") + 1])
        output_path = Path(command[command.index("-output") + 1])
        transform_path = Path(command[command.index("-xform") + 1])
        assert command[command.index("-size") + 1] == "4,3"
        assert command[command.index("-taper") + 1] == "1,0"
        with mrcfile.open(input_path, permissive=True) as binned:
            expected = source_data[[3, 2, 0]].reshape(3, 4, 2, 3, 2).mean(
                axis=(2, 4),
                dtype=np.float32,
            )
            np.testing.assert_allclose(binned.data, expected)
            assert float(binned.voxel_size.x) == pytest.approx(1.4)
        transforms = parse_imod_xf(transform_path)
        assert transforms[0][0:4] == pytest.approx((0.087, 0.996, -0.996, 0.087))
        assert transforms[0][4:] == pytest.approx((-2.0, 3.0))
        with mrcfile.new(output_path) as output:
            output.set_data(np.zeros((3, 3, 4), dtype=np.float32))
            output.voxel_size = 1.4
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="final stack",
            stderr="",
        )

    artifact = build_final_stack_and_register(
        ImodFinalAlignedStackBackend(fake_runner),
        source,
        fine_alignment,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"imod_dir": imod_dir, "output_binning": 2},
        ),
        registry,
    )

    assert artifact.kind == ArtifactKind.ALIGNED_TILT_STACK
    assert artifact.shape == (3, 3, 4)
    assert artifact.axis_order == AxisOrder.TYX
    assert artifact.pixel_spacing_angstrom == pytest.approx(1.4)
    assert artifact.parameters["purpose"] == "final_alignment"
    assert artifact.parameters["raw_pixel_spacing_angstrom"] == pytest.approx(0.7)
    assert artifact.parameters["output_pixel_spacing_angstrom"] == pytest.approx(1.4)
    assert artifact.parameters["included_z_values"] == [3, 2, 0]
    assert artifact.parameters["excluded_z_values"] == [1]
    assert Path(artifact.parameters["imod_xf_path"]).is_file()
    assert Path(artifact.parameters["tilt_file_path"]).read_text() == (
        "-6.100000\n-3.000000\n0.100000\n"
    )
    assert registry.get(artifact.id) == artifact


def test_final_stack_backend_rejects_coarse_alignment(tmp_path: Path) -> None:
    source_data = np.arange(192, dtype=np.float32).reshape(4, 8, 6)
    source, fine_alignment = _input_artifacts(tmp_path, source_data)
    coarse = fine_alignment.model_copy(
        update={"parameters": {"tilt_series_id": "TS_TEST", "stage": "coarse"}}
    )

    with pytest.raises(ValueError, match="requires fine alignment"):
        ImodFinalAlignedStackBackend().build(
            source,
            coarse,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
            ),
        )


def test_final_stack_backend_rounds_canvas_up_for_partial_bins(
    tmp_path: Path,
) -> None:
    source_data = np.arange(252, dtype=np.float32).reshape(4, 9, 7)
    source, fine_alignment = _input_artifacts(tmp_path, source_data)
    imod_dir = _fake_imod_dir(tmp_path)

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        assert command[command.index("-size") + 1] == "5,4"
        output_path = Path(command[command.index("-output") + 1])
        with mrcfile.new(output_path) as output:
            output.set_data(np.zeros((3, 4, 5), dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    artifact = ImodFinalAlignedStackBackend(fake_runner).build(
        source,
        fine_alignment,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={"imod_dir": imod_dir, "output_binning": 2},
        ),
    )

    assert artifact.shape == (3, 4, 5)


def _input_artifacts(
    tmp_path: Path,
    source_data: np.ndarray,
) -> tuple[Artifact, Artifact]:
    source_path = tmp_path / "source.zarr"
    zarr.save(source_path, source_data)
    source = Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=source_path,
        shape=tuple(int(axis_size) for axis_size in source_data.shape),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=1.35,
        parameters={"tilt_series_id": "TS_TEST"},
    )
    result = TiltAlignment(
        tilt_series_id="TS_TEST",
        backend="test",
        stage="fine",
        input_stack_id="TS_TEST:aligned_tilt_stack:fiducial",
        input_binning=2,
        tilt_axis_angle_deg=85.1,
        transform_semantics="global",
        transforms=[
            AlignmentTransform(
                z_value=3,
                tilt_angle_deg=-6.1,
                a11=0.087,
                a12=0.996,
                a21=-0.996,
                a22=0.087,
                shift_x_px=-4.0,
                shift_y_px=6.0,
            ),
            AlignmentTransform(
                z_value=2,
                tilt_angle_deg=-3.0,
                a11=0.087,
                a12=0.996,
                a21=-0.996,
                a22=0.087,
                shift_x_px=-2.0,
                shift_y_px=2.0,
            ),
            AlignmentTransform(
                z_value=0,
                tilt_angle_deg=0.1,
                a11=0.087,
                a12=0.996,
                a21=-0.996,
                a22=0.087,
                shift_x_px=4.0,
                shift_y_px=-2.0,
            ),
        ],
        excluded_z_values=[1],
    )
    alignment_path = tmp_path / "fine_alignment.json"
    alignment_path.write_text(result.model_dump_json())
    fine_alignment = Artifact(
        id="TS_TEST:alignment:fine",
        kind=ArtifactKind.ALIGNMENT,
        path=alignment_path,
        parent_ids=[source.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "stage": "fine",
            "raw_pixel_spacing_angstrom": 0.7,
        },
    )
    return source, fine_alignment


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=0.7,
        images=[
            TiltImage(
                z_value=index,
                tilt_angle_deg=angle,
                subframe_path=f"frame_{index}.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=0.7,
                binning=1,
            )
            for index, angle in enumerate((0.0, 3.0, -3.0, -6.0))
        ],
    )


def _fake_imod_dir(tmp_path: Path) -> Path:
    imod_dir = tmp_path / "imod"
    bin_dir = imod_dir / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "newstack").touch()
    return imod_dir
