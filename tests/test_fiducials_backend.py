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
    _etomo_patch_border_px,
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


def test_etomo_patch_border_is_five_percent_of_the_binned_size() -> None:
    # IMOD's makecomfile: int(0.05 * size + 0.5), per axis.
    assert _etomo_patch_border_px(512) == 26  # EMPIAR-11538, 4096 px at bin 8
    assert _etomo_patch_border_px(720) == 36  # a K3 frame's width at bin 8
    assert _etomo_patch_border_px(511) == 26  # and its height
    assert _etomo_patch_border_px(256) == 13  # 4096 px at bin 16


def test_patch_seed_defaults_to_etomos_border_on_each_axis(tmp_path: Path) -> None:
    """With no margin configured, X and Y each get 5% of their own binned size."""

    stack_data = np.arange(4 * 160 * 224, dtype=np.float32).reshape(4, 160, 224)
    stack = _stack_artifact(tmp_path, stack_data)
    alignment = _alignment_artifact(tmp_path, stack)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([stack, alignment])
    imod_dir = _fake_imod_dir(tmp_path)
    tiltxcorr_commands: list[list[str]] = []

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
                command[command.index("-input") + 1],
                command[command.index("-output") + 1],
            )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if program == "tiltxcorr":
            tiltxcorr_commands.append(list(command))
            Path(command[command.index("-output") + 1]).write_bytes(b"patchmodel")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
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

    tracking_stack, seed_model, seed_qc_artifact = generate_seed_and_register(
        ImodPatchSeedBackend(fake_runner),
        stack,
        alignment,
        manifest,
        _context(
            tmp_path,
            imod_dir,
            {
                "pre_binning": 2,
                "num_patches_x": 2,
                "num_patches_y": 2,
                "patch_overlap": 0.0,
            },
        ),
        registry,
    )

    # 224 x 160 binned 2 is 112 x 80: borders of int(5.6 + 0.5) = 6 in X and
    # int(4.0 + 0.5) = 4 in Y, leaving 100 and 72 px for two patches each.
    assert seed_model.parameters["patch_margin_x_px"] == 6
    assert seed_model.parameters["patch_margin_y_px"] == 4
    assert seed_model.parameters["patch_margin_source"] == "five_percent_of_binned_size"
    assert seed_model.parameters["patch_size_x_px"] == 50
    assert seed_model.parameters["patch_size_y_px"] == 36
    seed_qc = PatchGridQc.model_validate_json(seed_qc_artifact.path.read_text())
    assert (seed_qc.patch_margin_x_px, seed_qc.patch_margin_y_px) == (6, 4)
    assert seed_qc.usable_width_fraction == pytest.approx(100 / 112)
    assert seed_qc.usable_height_fraction == pytest.approx(72 / 80)

    tracking_context = _context(tmp_path, imod_dir, {"min_tracked_fiducials": 2})
    track_fiducials_and_register(
        ImodPatchTrackBackend(fake_runner),
        tracking_stack,
        seed_model,
        manifest,
        tracking_context,
        registry,
    )
    (command,) = tiltxcorr_commands
    assert command[command.index("-border") + 1] == "6,4"
    assert command[command.index("-size") + 1] == "50,36"

    # A seed from before the tracking stack was prealigned recorded one margin;
    # tracking it would apply the coarse shifts twice, so it is refused.
    stale_parameters = {
        key: value
        for key, value in seed_model.parameters.items()
        if not key.startswith("patch_margin")
    }
    stale_seed = seed_model.model_copy(
        update={"parameters": {**stale_parameters, "patch_margin_px": 100}}
    )
    with pytest.raises(ValueError, match="rerun generate-fiducial-seed"):
        ImodPatchTrackBackend(fake_runner).track(
            tracking_stack,
            stale_seed,
            manifest,
            tracking_context,
        )
    assert len(tiltxcorr_commands) == 1


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
            patch_margin_x_px=100,
            patch_margin_y_px=100,
            patch_margin_source="user_override",
            binned_width_px=512,
            binned_height_px=512,
            usable_width_fraction=0.609,
            usable_height_fraction=0.609,
            status=QcStatus.PASS,
        )


def test_imod_patch_tracking_backends_generate_and_track_models(
    tmp_path: Path,
) -> None:
    stack_data = np.arange(4 * 128 * 128, dtype=np.float32).reshape(4, 128, 128)
    stack = _stack_artifact(tmp_path, stack_data)
    alignment = _alignment_artifact(tmp_path, stack)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.extend([stack, alignment])
    imod_dir = _fake_imod_dir(tmp_path)
    binned_views = stack_data[[3, 2, 0]].reshape(3, 64, 2, 64, 2).mean(
        axis=(2, 4),
        dtype=np.float32,
    )
    seed_commands: list[list[str]] = []

    # Tiltxcorr reports positions in the frame of the stack it is given, and fine
    # alignment composes them with the coarse transforms, so the tracking stack
    # has to come out of newstack with those transforms applied. A plain binned
    # stack had the coarse shifts applied twice.
    def seed_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert env["IMOD_DIR"] == str(imod_dir)
        seed_commands.append(list(command))
        assert Path(command[0]).name == "newstack"
        input_path = Path(command[command.index("-input") + 1])
        transform_path = Path(command[command.index("-xform") + 1])
        assert command[command.index("-mode") + 1] == "2"
        with mrcfile.open(input_path, permissive=True) as binned:
            np.testing.assert_allclose(binned.data, binned_views)
        # Shifts are scaled to the binned pixels newstack works in.
        assert transform_path.read_text().splitlines()[0].endswith("-2.000 3.000")
        with mrcfile.new(Path(command[command.index("-output") + 1])) as aligned:
            aligned.set_data(binned_views + 1000.0)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="prealigned tracking stack",
            stderr="",
        )

    seed_context = _context(
        tmp_path,
        imod_dir,
        {
            "pre_binning": 2,
            "num_patches_x": 2,
            "num_patches_y": 2,
            "patch_overlap": 0.0,
            "patch_margin_px": 0,
        },
    )
    seed_artifacts = generate_seed_and_register(
        ImodPatchSeedBackend(seed_runner),
        stack,
        alignment,
        manifest,
        seed_context,
        registry,
    )

    assert [Path(command[0]).name for command in seed_commands] == ["newstack"]
    tracking_stack, seed_model, seed_qc_artifact = seed_artifacts
    assert tracking_stack.kind == ArtifactKind.ALIGNED_TILT_STACK
    assert tracking_stack.parameters["purpose"] == "fiducial_tracking"
    assert tracking_stack.shape == (3, 64, 64)
    assert tracking_stack.binning == 2
    assert tracking_stack.parameters["included_z_values"] == [3, 2, 0]
    assert tracking_stack.parameters["excluded_z_values"] == [1]
    with mrcfile.open(tracking_stack.path, permissive=True) as prealigned:
        np.testing.assert_allclose(prealigned.data, binned_views + 1000.0)
    assert Path(tracking_stack.parameters["newstack_log_path"]).is_file()
    # The transforms handed on to tiltxcorr and fine alignment stay unbinned.
    transform_path = Path(tracking_stack.parameters["prealign_transform_path"])
    assert transform_path.read_text().splitlines()[0].endswith("-4.000 6.000")

    assert seed_model.kind == ArtifactKind.FIDUCIAL_SEED_MODEL
    assert seed_model.id == "TS_TEST:fiducial_seed:imod-patch-seed"
    assert seed_model.backend == "imod-patch-seed"
    assert tracking_stack.id == "TS_TEST:aligned_tilt_stack:fiducial:imod-patch-seed"
    assert seed_model.parent_ids == [tracking_stack.id]
    assert seed_model.parameters["num_patches_x"] == 2
    assert seed_model.parameters["num_patches_y"] == 2
    assert seed_model.parameters["patch_size_x_px"] == 32
    assert seed_model.parameters["patch_size_y_px"] == 32
    # A patch seed reports its grid, not point counts: through FiducialModelQc
    # every field read zero, which for a bead seed would mean total failure.
    seed_qc = PatchGridQc.model_validate_json(seed_qc_artifact.path.read_text())
    assert seed_qc.stage == "seed"
    assert seed_qc.status == QcStatus.PASS
    assert seed_qc.warnings == []
    assert (seed_qc.num_patches_x, seed_qc.num_patches_y) == (2, 2)
    assert seed_qc.num_patches == 4
    assert seed_qc.patch_size_x_px == 32
    assert seed_qc.patch_size_y_px == 32
    assert seed_qc.pre_binning == 2
    # Reported in Angstrom too, because only a physical size can be judged
    # against a specimen. The tracking stack's pixel already includes the binning.
    scale = tracking_stack.pixel_spacing_angstrom
    assert scale == pytest.approx(2 * 0.675)
    assert seed_qc.patch_size_x_angstrom == pytest.approx(32 * scale)
    # A configured margin is the same on both axes.
    assert (seed_qc.patch_margin_x_px, seed_qc.patch_margin_y_px) == (0, 0)
    assert seed_qc.patch_margin_source == "user_override"
    assert seed_qc.usable_width_fraction == pytest.approx(1.0)
    assert seed_qc.usable_height_fraction == pytest.approx(1.0)

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
            assert Path(command[command.index("-input") + 1]) == (
                tracking_stack.path.resolve()
            )
            assert command[command.index("-size") + 1] == "32,32"
            assert command[command.index("-number") + 1] == "2,2"
            assert command[command.index("-border") + 1] == "0,0"
            assert command[command.index("-imagebinned") + 1] == "2"
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


def test_the_raw_pixel_is_the_stacks_when_motion_correction_binned_it(
    tmp_path: Path,
) -> None:
    from cryoet_pipeline.backends.fiducials import _raw_pixel_spacing

    context = BackendContext(output_dir=tmp_path, device=DevicePreference.CPU)
    stack = _stack_artifact(tmp_path, np.zeros((4, 8, 8), dtype=np.float32))
    # The manifest's movies are at 0.675 A; MotionCor3 -FtBin 2 sums are at 1.35.
    binned = stack.model_copy(
        update={"pixel_spacing_angstrom": 1.35,
                "parameters": {**stack.parameters, "frame_binning": 2}}
    )
    assert _raw_pixel_spacing(_manifest(), binned, context) == 1.35
    # The same 1.35 A without the recorded binning is a stale stack, as before.
    stale = stack.model_copy(update={"pixel_spacing_angstrom": 1.35})
    with pytest.raises(ValueError, match="binned 1x; rerun tilt-series preparation"):
        _raw_pixel_spacing(_manifest(), stale, context)
