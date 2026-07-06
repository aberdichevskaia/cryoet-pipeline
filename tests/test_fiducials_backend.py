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
from cryoet_pipeline.backends.fiducials import (
    ImodAutofidseedBackend,
    ImodBeadtrackBackend,
    generate_seed_and_register,
    parse_imod_model_ascii,
    track_fiducials_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    FiducialModelQc,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_imod_fiducial_backends_generate_and_track_models(tmp_path: Path) -> None:
    stack_data = np.arange(256, dtype=np.float32).reshape(4, 8, 8)
    stack = _stack_artifact(tmp_path, stack_data)
    alignment = _alignment_artifact(tmp_path, stack)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([stack, alignment])
    imod_dir = _fake_imod_dir(tmp_path)
    commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert env["IMOD_DIR"] == str(imod_dir)
        commands.append(list(command))
        program = Path(command[0]).name
        if program == "newstack":
            input_path = Path(command[command.index("-input") + 1])
            output_path = Path(command[command.index("-output") + 1])
            transform_path = Path(command[command.index("-xform") + 1])
            with mrcfile.open(input_path, permissive=True) as binned:
                assert binned.data.shape == (3, 4, 4)
                expected = stack_data[[3, 2, 0]].reshape(3, 4, 2, 4, 2).mean(
                    axis=(2, 4),
                    dtype=np.float32,
                )
                np.testing.assert_allclose(binned.data, expected)
                assert float(binned.voxel_size.x) == pytest.approx(1.35)
            assert transform_path.read_text().splitlines()[0].endswith(
                "-2.000 3.000"
            )
            shutil.copyfile(input_path, output_path)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="aligned tracking stack",
                stderr="",
            )
        if program == "autofidseed":
            track_path = Path(command[command.index("-track") + 1])
            track_text = track_path.read_text()
            assert "ImagesAreBinned\t2" in track_text
            assert "PixelSize\t0.06750000" in track_text
            assert "BeadDiameter\t148.148148" in track_text
            assert "BoxSizeXandY\t490,490" in track_text
            assert command[command.index("-number") + 1] == "20"
            Path(command[command.index("-output") + 1]).write_bytes(b"seed")
            Path(command[command.index("-info") + 1]).write_text("seed info\n")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="generated seed",
                stderr="",
            )
        if program == "beadtrack":
            assert command[command.index("-PixelSize") + 1] == "0.06750000"
            assert command[command.index("-BeadDiameter") + 1] == "148.148148"
            assert command[command.index("-ImagesAreBinned") + 1] == "2"
            box_index = command.index("-BoxSizeXandY")
            assert command[box_index + 1] == "490,490"
            Path(command[command.index("-OutputModel") + 1]).write_bytes(b"fid")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="tracked beads",
                stderr="",
            )
        if program == "imodinfo":
            model_path = Path(command[-1])
            points_per_contour = 1 if model_path.suffix == ".seed" else 3
            output = (
                "imod 1\n"
                "max 4 4 3\n"
                f"contour 0 0 {points_per_contour} 0\n"
                f"contour 1 0 {points_per_contour} 0\n"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=output,
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    seed_context = _context(
        tmp_path,
        imod_dir,
        {
            "tracking_binning": 2,
            "target_beads": 20,
            "min_seed_fiducials": 2,
        },
    )
    seed_artifacts = generate_seed_and_register(
        ImodAutofidseedBackend(fake_runner),
        stack,
        alignment,
        manifest,
        seed_context,
        registry,
    )

    tracking_stack, seed_model, seed_qc_artifact = seed_artifacts
    assert tracking_stack.kind == ArtifactKind.ALIGNED_TILT_STACK
    assert tracking_stack.shape == (3, 4, 4)
    assert tracking_stack.parameters["included_z_values"] == [3, 2, 0]
    assert tracking_stack.parameters["excluded_z_values"] == [1]
    assert seed_model.kind == ArtifactKind.FIDUCIAL_SEED_MODEL
    assert seed_model.parent_ids == [tracking_stack.id]
    assert seed_model.parameters["num_fiducials"] == 2
    assert Path(seed_model.parameters["track_command_path"]).is_file()
    seed_qc = FiducialModelQc.model_validate_json(
        seed_qc_artifact.path.read_text()
    )
    assert seed_qc.stage == "seed"
    assert seed_qc.status == QcStatus.PASS
    assert seed_qc.coverage_fraction == pytest.approx(1 / 3)

    tracking_context = _context(
        tmp_path,
        imod_dir,
        {
            "min_tracked_fiducials": 2,
            "coverage_warning": 0.8,
            "coverage_failure": 0.5,
        },
    )
    tracking_artifacts = track_fiducials_and_register(
        ImodBeadtrackBackend(fake_runner),
        tracking_stack,
        seed_model,
        manifest,
        tracking_context,
        registry,
    )

    fiducial_model, tracking_qc_artifact = tracking_artifacts
    assert fiducial_model.kind == ArtifactKind.FIDUCIAL_MODEL
    assert fiducial_model.parent_ids == [tracking_stack.id, seed_model.id]
    assert fiducial_model.parameters["coverage_fraction"] == pytest.approx(1.0)
    tracking_qc = FiducialModelQc.model_validate_json(
        tracking_qc_artifact.path.read_text()
    )
    assert tracking_qc.stage == "tracked"
    assert tracking_qc.status == QcStatus.PASS
    assert tracking_qc.num_points == 6
    assert registry.get(fiducial_model.id) == fiducial_model
    assert [Path(command[0]).name for command in commands] == [
        "newstack",
        "autofidseed",
        "imodinfo",
        "beadtrack",
        "imodinfo",
    ]


def test_autofidseed_backend_accepts_explicit_calibration_overrides(
    tmp_path: Path,
) -> None:
    stack = _stack_artifact(
        tmp_path,
        np.arange(256, dtype=np.float32).reshape(4, 8, 8),
    )
    alignment = _alignment_artifact(tmp_path, stack)
    imod_dir = _fake_imod_dir(tmp_path)

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        program = Path(command[0]).name
        if program == "newstack":
            shutil.copyfile(
                Path(command[command.index("-input") + 1]),
                Path(command[command.index("-output") + 1]),
            )
        elif program == "autofidseed":
            track_text = Path(command[command.index("-track") + 1]).read_text()
            assert "PixelSize\t0.10000000" in track_text
            assert "BeadDiameter\t80.000000" in track_text
            assert "BoxSizeXandY\t264,264" in track_text
            Path(command[command.index("-output") + 1]).write_bytes(b"seed")
            Path(command[command.index("-info") + 1]).write_text("info")
        elif program == "imodinfo":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="imod 1\nmax 4 4 3\ncontour 0 0 1 0\n",
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = _context(
        tmp_path,
        imod_dir,
        {
            "tracking_binning": 2,
            "raw_pixel_spacing_angstrom": 1.0,
            "fiducial_diameter_unbinned_px": 80.0,
            "min_seed_fiducials": 1,
        },
    )
    artifacts = ImodAutofidseedBackend(fake_runner).generate(
        stack,
        alignment,
        _manifest(),
        context,
    )

    seed = artifacts[1]
    assert seed.parameters["fiducial_diameter_unbinned_px"] == 80.0
    assert seed.parameters["fiducial_diameter_px_source"] == "user_override"
    assert artifacts[0].pixel_spacing_angstrom == pytest.approx(2.0)


def test_autofidseed_failure_does_not_register_partial_artifacts(
    tmp_path: Path,
) -> None:
    stack = _stack_artifact(
        tmp_path,
        np.arange(256, dtype=np.float32).reshape(4, 8, 8),
    )
    alignment = _alignment_artifact(tmp_path, stack)
    registry = ArtifactRegistry.empty()
    registry.extend([stack, alignment])
    imod_dir = _fake_imod_dir(tmp_path)

    def failing_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        return subprocess.CompletedProcess(
            command,
            4,
            stdout="",
            stderr="cannot align",
        )

    with pytest.raises(RuntimeError, match="newstack failed with exit code 4"):
        generate_seed_and_register(
            ImodAutofidseedBackend(failing_runner),
            stack,
            alignment,
            _manifest(),
            _context(tmp_path, imod_dir, {"tracking_binning": 2}),
            registry,
        )

    assert registry.artifact_ids == {stack.id, alignment.id}
    assert len(registry.by_kind(ArtifactKind.FIDUCIAL_SEED_MODEL)) == 0


def test_autofidseed_rejects_stale_stack_pixel_spacing(tmp_path: Path) -> None:
    stack = _stack_artifact(
        tmp_path,
        np.arange(256, dtype=np.float32).reshape(4, 8, 8),
    ).model_copy(update={"pixel_spacing_angstrom": 1.35})
    alignment = _alignment_artifact(tmp_path, stack)
    imod_dir = _fake_imod_dir(tmp_path)

    with pytest.raises(ValueError, match="rerun tilt-series preparation"):
        ImodAutofidseedBackend().generate(
            stack,
            alignment,
            _manifest(),
            _context(tmp_path, imod_dir, {"tracking_binning": 2}),
        )


def test_parse_imod_model_ascii_rejects_missing_dimensions() -> None:
    with pytest.raises(ValueError, match="no valid max dimensions"):
        parse_imod_model_ascii("imod 1\ncontour 0 0 1 0\n")


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
        pixel_spacing_angstrom=0.675,
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
        transform_semantics="global",
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
            AlignmentTransform(
                z_value=3,
                tilt_angle_deg=-6.0,
                a11=1.0,
                a12=0.0,
                a21=0.0,
                a22=1.0,
                shift_x_px=-4.0,
                shift_y_px=6.0,
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
        raw_pixel_spacing_angstrom=0.675,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frame_0.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=0.675,
                binning=1,
            ),
            TiltImage(
                z_value=1,
                tilt_angle_deg=3.0,
                subframe_path="frame_1.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=0.675,
                binning=1,
            ),
            TiltImage(
                z_value=2,
                tilt_angle_deg=-3.0,
                subframe_path="frame_2.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=0.675,
                binning=1,
            ),
            TiltImage(
                z_value=3,
                tilt_angle_deg=-6.0,
                subframe_path="frame_3.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=0.675,
                binning=1,
            ),
        ],
    )


def _fake_imod_dir(tmp_path: Path) -> Path:
    imod_dir = tmp_path / "imod"
    bin_dir = imod_dir / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("newstack", "autofidseed", "beadtrack", "imodinfo"):
        (bin_dir / name).touch()
    return imod_dir


def _context(
    tmp_path: Path,
    imod_dir: Path,
    parameters: dict[str, object],
) -> BackendContext:
    return BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={**parameters, "imod_dir": imod_dir},
    )
