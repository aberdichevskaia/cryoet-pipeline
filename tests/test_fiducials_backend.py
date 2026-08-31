from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr
from pydantic import ValidationError

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.fiducials import (
    ImodAutofidseedBackend,
    ImodBeadtrackBackend,
    ImodPatchSeedBackend,
    ImodPatchTrackBackend,
    _compute_patch_size_px,
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
    PatchGridQc,
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
    # The bead and patch backends both produce a seed, so both ids name theirs.
    assert seed_model.id == "TS_TEST:fiducial_seed:imod-autofidseed"
    assert seed_model.backend == "imod-autofidseed"
    assert tracking_stack.id == "TS_TEST:aligned_tilt_stack:fiducial:imod-autofidseed"
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
    assert fiducial_model.id == "TS_TEST:fiducial_model:imod-beadtrack"
    assert fiducial_model.path.name == "TS_TEST_imod-beadtrack.fid"
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


def test_compute_patch_size_px_matches_user_formula() -> None:
    assert (
        _compute_patch_size_px(1000, 5, margin_px=100, overlap_fraction=0.33) == 217
    )
    assert _compute_patch_size_px(500, 1, margin_px=50, overlap_fraction=0.5) == 400


def test_compute_patch_size_px_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _compute_patch_size_px(1000, 0, margin_px=100, overlap_fraction=0.33)
    with pytest.raises(ValueError, match="not be negative"):
        _compute_patch_size_px(1000, 5, margin_px=-1, overlap_fraction=0.33)
    with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\)"):
        _compute_patch_size_px(1000, 5, margin_px=100, overlap_fraction=1.0)
    with pytest.raises(ValueError, match="no usable area"):
        _compute_patch_size_px(100, 5, margin_px=60, overlap_fraction=0.33)
    with pytest.raises(ValueError, match="too small"):
        _compute_patch_size_px(100, 10, margin_px=40, overlap_fraction=0.0)


def test_patch_grid_qc_warns_when_the_margin_beats_foreshortening() -> None:
    """The margin is judged against geometry, not against a round number.

    At the largest tilt the specimen foreshortens by cos(tilt) along the axis
    across the tilt axis, so that fraction of one image is what every view still
    covers. A margin leaving less than that is cutting more than the geometry
    asks for — which is worth saying, because the margin is in binned pixels and
    so does not scale with pre_binning.
    """

    common = dict(
        tilt_series_id="TS_TEST",
        backend="imod-patch-seed",
        model_id="TS_TEST:fiducial_seed:imod-patch-seed",
        image_count=59,
        num_patches_x=5,
        num_patches_y=5,
        num_patches=25,
        patch_overlap=0.33,
        pre_binning=8,
        patch_size_x_px=84,
        patch_size_y_px=84,
        patch_size_x_angstrom=822.0,
        patch_size_y_angstrom=822.0,
        binned_width_px=512,
        binned_height_px=512,
        # cos(64 degrees), the largest tilt in EMPIAR-11538
        foreshortening_fraction=0.438,
    )

    # 100 px of margin leaves 312 of 512, or 0.609 — more than foreshortening
    # needs, so aggressive but defensible and no warning.
    generous = PatchGridQc(
        **common,
        patch_margin_px=100,
        usable_width_fraction=312 / 512,
        usable_height_fraction=312 / 512,
        status=QcStatus.PASS,
    )
    assert generous.usable_width_fraction > generous.foreshortening_fraction

    # 160 px leaves 192 of 512, or 0.375, below what every view covers.
    tight = PatchGridQc(
        **common,
        patch_margin_px=160,
        usable_width_fraction=192 / 512,
        usable_height_fraction=192 / 512,
        status=QcStatus.WARNING,
        warnings=["the margin is cutting more than foreshortening requires"],
    )
    assert tight.usable_width_fraction < tight.foreshortening_fraction


def test_patch_grid_qc_rejects_an_inconsistent_patch_count() -> None:
    with pytest.raises(ValidationError, match="product of the two axis counts"):
        PatchGridQc(
            tilt_series_id="TS_TEST",
            backend="imod-patch-seed",
            model_id="TS_TEST:fiducial_seed:imod-patch-seed",
            image_count=59,
            num_patches_x=5,
            num_patches_y=5,
            num_patches=24,
            patch_overlap=0.33,
            pre_binning=8,
            patch_size_x_px=84,
            patch_size_y_px=84,
            patch_size_x_angstrom=822.0,
            patch_size_y_angstrom=822.0,
            patch_margin_px=100,
            binned_width_px=512,
            binned_height_px=512,
            usable_width_fraction=0.609,
            usable_height_fraction=0.609,
            foreshortening_fraction=0.438,
            status=QcStatus.PASS,
        )


def test_imod_patch_tracking_backends_generate_and_track_models(
    tmp_path: Path,
) -> None:
    stack_data = np.arange(4 * 96 * 96, dtype=np.float32).reshape(4, 96, 96)
    stack = _stack_artifact(tmp_path, stack_data)
    alignment = _alignment_artifact(tmp_path, stack)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([stack, alignment])
    imod_dir = _fake_imod_dir(tmp_path)

    seed_context = _context(
        tmp_path,
        imod_dir,
        {
            "pre_binning": 1,
            "num_patches_x": 2,
            "num_patches_y": 2,
            "patch_overlap": 0.0,
            "patch_margin_px": 0,
        },
    )
    seed_artifacts = generate_seed_and_register(
        ImodPatchSeedBackend(),
        stack,
        alignment,
        manifest,
        seed_context,
        registry,
    )

    tracking_stack, seed_model, seed_qc_artifact = seed_artifacts
    assert tracking_stack.kind == ArtifactKind.ALIGNED_TILT_STACK
    assert tracking_stack.parameters["purpose"] == "fiducial_tracking"
    assert tracking_stack.shape == (3, 96, 96)
    assert tracking_stack.parameters["included_z_values"] == [3, 2, 0]
    assert tracking_stack.parameters["excluded_z_values"] == [1]
    with mrcfile.open(tracking_stack.path, permissive=True) as binned:
        np.testing.assert_allclose(binned.data, stack_data[[3, 2, 0]])
    transform_path = Path(tracking_stack.parameters["prealign_transform_path"])
    assert transform_path.read_text().splitlines()[0].endswith("-4.000 6.000")

    assert seed_model.kind == ArtifactKind.FIDUCIAL_SEED_MODEL
    assert seed_model.id == "TS_TEST:fiducial_seed:imod-patch-seed"
    assert seed_model.backend == "imod-patch-seed"
    assert tracking_stack.id == "TS_TEST:aligned_tilt_stack:fiducial:imod-patch-seed"
    assert seed_model.parent_ids == [tracking_stack.id]
    assert seed_model.parameters["num_patches_x"] == 2
    assert seed_model.parameters["num_patches_y"] == 2
    assert seed_model.parameters["patch_size_x_px"] == 48
    assert seed_model.parameters["patch_size_y_px"] == 48
    # A patch seed reports its grid, not point counts: through FiducialModelQc
    # every field read zero, which for a bead seed would mean total failure.
    seed_qc = PatchGridQc.model_validate_json(seed_qc_artifact.path.read_text())
    assert seed_qc.stage == "seed"
    assert seed_qc.status == QcStatus.PASS
    assert seed_qc.warnings == []
    assert (seed_qc.num_patches_x, seed_qc.num_patches_y) == (2, 2)
    assert seed_qc.num_patches == 4
    assert seed_qc.patch_size_x_px == 48
    assert seed_qc.patch_size_y_px == 48
    assert seed_qc.pre_binning == 1
    # Reported in Angstrom too, because only a physical size can be judged
    # against a specimen.
    scale = seed_qc.pre_binning * tracking_stack.pixel_spacing_angstrom
    assert seed_qc.patch_size_x_angstrom == pytest.approx(48 * scale)
    # Margin 0 leaves the whole field, which is never below foreshortening.
    assert seed_qc.patch_margin_px == 0
    assert seed_qc.usable_width_fraction == pytest.approx(1.0)
    assert seed_qc.usable_height_fraction == pytest.approx(1.0)
    assert 0.0 < seed_qc.foreshortening_fraction <= 1.0

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
        if program == "tiltxcorr":
            assert "-overlap" not in command
            assert "-length" not in command
            assert command[command.index("-size") + 1] == "48,48"
            assert command[command.index("-number") + 1] == "2,2"
            assert command[command.index("-border") + 1] == "0,0"
            assert command[command.index("-imagebinned") + 1] == "1"
            Path(command[command.index("-output") + 1]).write_bytes(b"patchmodel")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="tracked patches",
                stderr="",
            )
        if program == "imodinfo":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "imod 1\nmax 4 4 3\n"
                    "contour 0 0 3 0\ncontour 1 0 3 0\n"
                    "contour 2 0 3 0\ncontour 3 0 3 0\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

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
        ImodPatchTrackBackend(fake_runner),
        tracking_stack,
        seed_model,
        manifest,
        tracking_context,
        registry,
    )

    fiducial_model, tracking_qc_artifact = tracking_artifacts
    assert fiducial_model.kind == ArtifactKind.FIDUCIAL_MODEL
    assert fiducial_model.parent_ids == [tracking_stack.id, seed_model.id]
    assert fiducial_model.parameters["num_fiducials"] == 4
    assert fiducial_model.parameters["coverage_fraction"] == pytest.approx(1.0)
    tracking_qc = FiducialModelQc.model_validate_json(
        tracking_qc_artifact.path.read_text()
    )
    assert tracking_qc.stage == "tracked"
    assert tracking_qc.status == QcStatus.PASS
    assert registry.get(fiducial_model.id) == fiducial_model
    assert fiducial_model.id == "TS_TEST:fiducial_model:imod-patch-track"
    assert fiducial_model.path.name == "TS_TEST_imod-patch-track.fid"
    assert [Path(command[0]).name for command in commands] == [
        "tiltxcorr",
        "imodinfo",
    ]


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
    for name in ("newstack", "autofidseed", "beadtrack", "tiltxcorr", "imodinfo"):
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
