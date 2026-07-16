from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.fine_alignment import (
    ImodTiltalignBackend,
    fine_align_and_register,
    parse_tiltalign_log,
    parse_tiltalign_surface_analysis,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    FineAlignmentQc,
    QcStatus,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_imod_tiltalign_backend_writes_fine_alignment_and_qc(
    tmp_path: Path,
) -> None:
    tracking_stack, fiducial_model = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([tracking_stack, fiducial_model])
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
        if program == "tiltalign":
            angle_offset = command[command.index("-AngleOffset") + 1]
            positioned = angle_offset != "0.0"
            assert command[command.index("-ImagesAreBinned") + 1] == "2"
            assert command[command.index("-UnbinnedPixelSize") + 1] == "0.06750000"
            assert command[command.index("-MinFidsTotalAndEachSurface") + 1] == "10,4"
            assert "-RobustFitting" in command
            if angle_offset == "3.720000":
                assert command[command.index("-AxisZShift") + 1] == "512.000000"
                incremental_angle = 0.16
                total_angle = 3.88
                incremental_z_shift = -20.0
                total_z_shift = 492.0
            elif angle_offset == "3.880000":
                assert command[command.index("-AxisZShift") + 1] == "492.000000"
                incremental_angle = 0.0
                total_angle = 3.88
                incremental_z_shift = 0.0
                total_z_shift = 492.0
            else:
                assert not positioned
                incremental_angle = 3.72
                total_angle = 3.72
                incremental_z_shift = 512.0
                total_z_shift = 512.0
            _path_after(command, "-OutputModelFile").write_bytes(b"3d model")
            _path_after(command, "-OutputResidualFile").write_text(
                "6 residuals\n"
                "0 0 0 0.1 0.2\n"
                "0 0 1 -0.2 0.1\n"
                "0 0 2 0.0 -0.1\n"
                "0 0 0 0.3 0.2\n"
                "0 0 1 -0.1 -0.2\n"
                "0 0 2 0.2 0.2\n"
            )
            _path_after(command, "-OutputFilledInModel").write_bytes(b"filled")
            _path_after(command, "-OutputFidXYZFile").write_text("xyz\n")
            _path_after(command, "-OutputTiltFile").write_text(
                "-6.100000\n-3.000000\n0.100000\n"
            )
            _path_after(command, "-OutputXAxisTiltFile").write_text("0\n0\n0\n")
            _path_after(command, "-OutputTransformFile").write_text(
                "1 0 0 1 0 0\n"
                "1 0 0 1 1 -1\n"
                "1 0 0 1 2 -2\n"
            )
            stdout = (
                "3 views, 10 geometric variables, 2 3-D points, "
                "6 projection points\n"
                "At minimum tilt, rotation angle is   85.10\n"
                "Residual error mean and sd:     0.100   0.020 nm\n"
                "Global leave-out error (6 pts): 0.120 nm\n"
                "X axis tilt needed = -1.20\n"
                "X axis tilt needed = -1.60\n"
                "Unbinned thickness needed to contain centers of all "
                "fiducials = 2400\n"
                "Incremental tilt angle change = "
                f"{incremental_angle}\n"
                f"Total tilt angle change = {total_angle}\n"
                "Incremental unbinned shift needed to center range of "
                "fiducials in Z = "
                f"{incremental_z_shift}\n"
                "Total unbinned shift needed to center range of fiducials "
                f"in Z = {total_z_shift}\n"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=stdout,
                stderr="",
            )
        if program == "xfproduct":
            assert _path_after(command, "-in1").name == "tracking.prexg"
            assert _path_after(command, "-in2").is_file()
            assert command[command.index("-ScaleShifts") + 1] == "1.0,2.000000"
            _path_after(command, "-output").write_text(
                "0.087 0.996 -0.996 0.087 -4 6\n"
                "0.087 0.996 -0.996 0.087 -2 2\n"
                "0.087 0.996 -0.996 0.087 4 -2\n"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="combined transforms",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"imod_dir": imod_dir},
    )
    artifacts = fine_align_and_register(
        ImodTiltalignBackend(fake_runner),
        tracking_stack,
        fiducial_model,
        _manifest(),
        context,
        registry,
    )

    alignment_artifact, report_artifact = artifacts
    assert alignment_artifact.kind == ArtifactKind.ALIGNMENT
    assert alignment_artifact.parent_ids == [
        tracking_stack.id,
        fiducial_model.id,
    ]
    assert alignment_artifact.parameters["stage"] == "fine"
    assert alignment_artifact.parameters["raw_pixel_spacing_angstrom"] == 0.675
    assert alignment_artifact.parameters["recommended_x_axis_tilt_deg"] == -1.6
    assert alignment_artifact.parameters["recommended_unbinned_thickness_px"] == 2400
    assert alignment_artifact.parameters["recommended_unbinned_z_shift_px"] == 492
    assert alignment_artifact.parameters["applied_tilt_angle_offset_deg"] == 3.88
    assert (
        alignment_artifact.parameters["applied_axis_z_shift_unbinned_px"]
        == 492
    )
    assert alignment_artifact.parameters["axis_z_shift_applied_in_alignment"] is True
    assert Path(alignment_artifact.parameters["imod_xf_path"]).is_file()
    result = TiltAlignment.model_validate_json(alignment_artifact.path.read_text())
    assert result.stage == "fine"
    assert result.input_stack_id == tracking_stack.id
    assert [transform.z_value for transform in result.transforms] == [3, 2, 0]
    assert [transform.tilt_angle_deg for transform in result.transforms] == pytest.approx(
        [-6.1, -3.0, 0.1]
    )
    assert result.transforms[0].a12 == pytest.approx(0.996)
    assert result.excluded_z_values == [1]

    report = FineAlignmentQc.model_validate_json(report_artifact.path.read_text())
    assert report.status == QcStatus.PASS
    assert report.fiducial_count == 2
    assert report.projection_point_count == 6
    assert report.residual_mean_nm == pytest.approx(0.1)
    assert report.residual_mean_unbinned_px == pytest.approx(0.1 / 0.0675)
    assert report.residual_max_tracking_px == pytest.approx(np.hypot(0.3, 0.2))
    assert report.residual_outlier_count == 0
    assert report.global_leave_out_error_nm == pytest.approx(0.12)
    assert report.recommended_x_axis_tilt_deg == pytest.approx(-1.6)
    assert report.recommended_unbinned_thickness_px == pytest.approx(2400)
    assert report.recommended_unbinned_z_shift_px == pytest.approx(492)
    assert report.applied_tilt_angle_offset_deg == pytest.approx(3.88)
    assert report.applied_axis_z_shift_unbinned_px == pytest.approx(492)
    assert report.positioning_incremental_tilt_angle_deg == pytest.approx(0.0)
    assert report.positioning_incremental_z_shift_unbinned_px == pytest.approx(0.0)
    assert report.alignment_rounds == 3
    assert registry.get(alignment_artifact.id) == alignment_artifact
    assert [Path(command[0]).name for command in commands] == [
        "tiltalign",
        "tiltalign",
        "tiltalign",
        "xfproduct",
    ]


def test_imod_tiltalign_failure_does_not_register_artifacts(
    tmp_path: Path,
) -> None:
    tracking_stack, fiducial_model = _input_artifacts(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.extend([tracking_stack, fiducial_model])
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
            7,
            stdout="",
            stderr="bad model",
        )

    with pytest.raises(RuntimeError, match="tiltalign failed with exit code 7"):
        fine_align_and_register(
            ImodTiltalignBackend(failing_runner),
            tracking_stack,
            fiducial_model,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={"imod_dir": imod_dir},
            ),
            registry,
        )

    assert len(registry.by_kind(ArtifactKind.ALIGNMENT)) == 0
    assert len(registry.by_kind(ArtifactKind.QC)) == 0


def test_imod_tiltalign_prunes_failure_level_point_and_reruns(
    tmp_path: Path,
) -> None:
    tracking_stack, fiducial_model = _input_artifacts(tmp_path)
    imod_dir = _fake_imod_dir(tmp_path)
    programs: list[str] = []
    tiltalign_round = 0

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        nonlocal tiltalign_round
        del cwd, env
        program = Path(command[0]).name
        programs.append(program)
        if program == "tiltalign":
            tiltalign_round += 1
            _path_after(command, "-OutputModelFile").write_bytes(b"3d model")
            residuals = [
                (25.0, 0.0) if tiltalign_round == 1 else (0.2, 0.1),
                (0.2, 0.1),
                (0.1, 0.2),
                (0.3, 0.1),
                (0.1, 0.1),
                (0.2, 0.2),
            ]
            _path_after(command, "-OutputResidualFile").write_text(
                "6 residuals\n"
                + "".join(
                    f"0 0 {index % 3} {x} {y}\n"
                    for index, (x, y) in enumerate(residuals)
                )
            )
            _path_after(command, "-OutputFilledInModel").write_bytes(b"filled")
            _path_after(command, "-OutputFidXYZFile").write_text("xyz\n")
            _path_after(command, "-OutputTiltFile").write_text("-6\n-3\n0\n")
            _path_after(command, "-OutputXAxisTiltFile").write_text("0\n0\n0\n")
            _path_after(command, "-OutputTransformFile").write_text(
                "1 0 0 1 0 0\n1 0 0 1 0 0\n1 0 0 1 0 0\n"
            )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "3 views, 10 geometric variables, 2 3-D points, "
                    "6 projection points\n"
                    "At minimum tilt, rotation angle is 85.1\n"
                    "Residual error mean and sd: 0.1 0.02 nm\n"
                ),
                stderr="",
            )
        if program == "model2point":
            Path(command[-1]).write_text(
                "".join(
                    f"0 0 {index}.0 {index}.0 {index % 3 + 0.5}\n"
                    for index in range(6)
                )
            )
            return subprocess.CompletedProcess(command, 0, stdout="6 points", stderr="")
        if program == "point2model":
            Path(command[-1]).write_bytes(b"cleaned")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="cleaned model",
                stderr="",
            )
        if program == "xfproduct":
            assert command[command.index("-ScaleShifts") + 1] == "1.0,2.000000"
            _path_after(command, "-output").write_text(
                "0.087 0.996 -0.996 0.087 0 0\n"
                "0.087 0.996 -0.996 0.087 0 0\n"
                "0.087 0.996 -0.996 0.087 0 0\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    artifacts = ImodTiltalignBackend(fake_runner).align(
        tracking_stack,
        fiducial_model,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "imod_dir": imod_dir,
                "residual_max_failure_tracking_px": 20.0,
                "max_pruned_fraction": 0.2,
            },
        ),
    )

    report = FineAlignmentQc.model_validate_json(artifacts[1].path.read_text())
    assert report.status == QcStatus.PASS
    assert report.pruned_point_count == 1
    assert report.alignment_rounds == 2
    assert report.residual_max_tracking_px < 1.0
    assert programs == [
        "tiltalign",
        "model2point",
        "point2model",
        "tiltalign",
        "xfproduct",
    ]


def test_parse_tiltalign_log_rejects_incomplete_summary() -> None:
    with pytest.raises(ValueError, match="no residual mean"):
        parse_tiltalign_log("At minimum tilt, rotation angle is 85.0\n")


def test_parse_tiltalign_surface_analysis_uses_two_surface_recommendation() -> None:
    summary = parse_tiltalign_surface_analysis(
        "X axis tilt needed = -1.34\n"
        "X axis tilt needed = -1.61\n"
        "Unbinned thickness needed to contain centers of all fiducials = 2366\n"
        "Incremental tilt angle change = 3.72\n"
        "Total tilt angle change = 3.72\n"
        "Incremental unbinned shift needed to center range of fiducials "
        "in Z = 513.8\n"
        "Total unbinned shift needed to center range of fiducials in Z = 513.8\n"
    )

    assert summary is not None
    assert summary.x_axis_tilt_deg == pytest.approx(-1.61)
    assert summary.unbinned_thickness_px == pytest.approx(2366)
    assert summary.unbinned_z_shift_px == pytest.approx(513.8)
    assert summary.incremental_unbinned_z_shift_px == pytest.approx(513.8)
    assert summary.total_tilt_angle_deg == pytest.approx(3.72)
    assert summary.incremental_tilt_angle_deg == pytest.approx(3.72)


def test_parse_tiltalign_surface_analysis_rejects_partial_summary() -> None:
    with pytest.raises(ValueError, match="surface analysis is incomplete"):
        parse_tiltalign_surface_analysis("X axis tilt needed = -1.61\n")


def _input_artifacts(tmp_path: Path) -> tuple[Artifact, Artifact]:
    stack_path = tmp_path / "tracking.mrc"
    with mrcfile.new(stack_path) as stack:
        stack.set_data(np.zeros((3, 4, 4), dtype=np.float32))
        stack.voxel_size = 1.35
    prealign_path = tmp_path / "tracking.prexg"
    prealign_path.write_text(
        "1 0 0 1 -4 6\n"
        "1 0 0 1 -2 2\n"
        "1 0 0 1 4 -2\n"
    )
    tilt_path = tmp_path / "tracking.rawtlt"
    tilt_path.write_text("-6\n-3\n0\n")
    tracking_stack = Artifact(
        id="TS_TEST:aligned_tilt_stack:fiducial",
        kind=ArtifactKind.ALIGNED_TILT_STACK,
        path=stack_path,
        shape=(3, 4, 4),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=1.35,
        binning=2,
        parameters={
            "purpose": "fiducial_tracking",
            "tilt_series_id": "TS_TEST",
            "included_z_values": [3, 2, 0],
            "excluded_z_values": [1],
            "prealign_transform_path": str(prealign_path),
            "tilt_file_path": str(tilt_path),
            "tilt_axis_angle_deg": 85.3,
            "raw_pixel_spacing_angstrom": 0.675,
        },
    )
    model_path = tmp_path / "tracking.fid"
    model_path.write_bytes(b"fid")
    fiducial_model = Artifact(
        id="TS_TEST:fiducial_model",
        kind=ArtifactKind.FIDUCIAL_MODEL,
        path=model_path,
        parent_ids=[tracking_stack.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "num_fiducials": 2,
            "num_points": 6,
        },
    )
    return tracking_stack, fiducial_model


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
    for name in ("tiltalign", "xfproduct", "model2point", "point2model"):
        (bin_dir / name).touch()
    return imod_dir


def _path_after(command: Sequence[str], option: str) -> Path:
    return Path(command[command.index(option) + 1])
