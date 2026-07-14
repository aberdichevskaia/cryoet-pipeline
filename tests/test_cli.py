from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import ImodTiltXcorrAlignmentBackend
from cryoet_pipeline.backends.motion import MotionCor3MotionCorrectionBackend
from cryoet_pipeline.backends.restoration import IsoNet2RestorationBackend
from cryoet_pipeline.backends.segmentation import MemBrainSegSegmentationBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    ProjectConfig,
    RetentionPolicy,
    StorageRole,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.project import InitResult
from cryoet_pipeline.runtime import DevicePreference


def test_init_command_passes_normalized_project_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, ProjectConfig] = {}

    def fake_initialize_project(config: ProjectConfig) -> InitResult:
        captured["config"] = config
        return InitResult(
            project_path=tmp_path / "project.json",
            manifest_paths=[],
            artifact_registry_path=tmp_path / "artifacts.json",
        )

    monkeypatch.setattr(cli, "initialize_project", fake_initialize_project)

    result = CliRunner().invoke(
        cli.app,
        [
            "init",
            "--frames",
            str(tmp_path / "frames"),
            "--mdocs",
            str(tmp_path / "mdocs"),
            "--out",
            str(tmp_path / "out"),
            "--tilt-series",
            "TS_TEST",
            "--device",
            "CPU",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].tilt_series == ["TS_TEST"]
    assert captured["config"].device == DevicePreference.CPU


def test_correct_motion_command_writes_outputs_and_updates_registry(
    tmp_path: Path,
) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    movie_data = np.array(
        [
            [[1, 2], [3, 4]],
            [[3, 4], [5, 6]],
        ],
        dtype=np.float32,
    )
    _write_mrc(movie_path, movie_data)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(movie_path).model_dump_json())
    registry_path = tmp_path / "artifacts.json"
    ArtifactRegistry.empty().write(registry_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "correct-motion",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "wrote 1 corrected projections" in result.output

    registry = ArtifactRegistry.load(registry_path)
    artifacts = registry.by_kind(ArtifactKind.CORRECTED_PROJECTION)
    assert len(artifacts) == 1
    assert artifacts[0].path == tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_avg.mrc"
    assert artifacts[0].storage_role == StorageRole.CACHE
    assert artifacts[0].retention_policy == RetentionPolicy.KEEP
    assert artifacts[0].size_bytes is not None
    assert artifacts[0].size_bytes > 0

    with mrcfile.open(artifacts[0].path, permissive=True) as corrected:
        np.testing.assert_allclose(corrected.data, movie_data.mean(axis=0))


def test_correct_motion_command_requires_overwrite_for_rerun(tmp_path: Path) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(movie_path).model_dump_json())
    registry_path = tmp_path / "artifacts.json"
    ArtifactRegistry.empty().write(registry_path)
    runner = CliRunner()
    args = [
        "correct-motion",
        "--manifest",
        str(manifest_path),
        "--registry",
        str(registry_path),
        "--out",
        str(tmp_path / "outputs"),
        "--device",
        "cpu",
    ]

    first_result = runner.invoke(cli.app, args)
    second_result = runner.invoke(cli.app, args)
    overwrite_result = runner.invoke(cli.app, [*args, "--overwrite"])

    assert first_result.exit_code == 0
    assert second_result.exit_code != 0
    assert overwrite_result.exit_code == 0

    registry = ArtifactRegistry.load(registry_path)
    assert len(registry.by_kind(ArtifactKind.CORRECTED_PROJECTION)) == 1


def test_correct_motion_command_supports_working_zarr_storage_policy(
    tmp_path: Path,
) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(movie_path).model_dump_json())
    registry_path = tmp_path / "artifacts.json"
    ArtifactRegistry.empty().write(registry_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "correct-motion",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
            "--storage-policy",
            "working",
        ],
    )

    assert result.exit_code == 0
    assert "storage policy: working" in result.output

    artifact = ArtifactRegistry.load(registry_path).by_kind(ArtifactKind.CORRECTED_PROJECTION)[0]
    assert artifact.path == tmp_path / "outputs/corrected/TS_TEST/TS_TEST_000_avg.zarr"
    assert artifact.path.is_dir()
    assert artifact.storage_role == StorageRole.CACHE
    assert artifact.retention_policy == RetentionPolicy.RECOMPUTE


def test_correct_motion_command_rejects_unknown_backend(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "correct-motion",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--registry",
            str(tmp_path / "artifacts.json"),
            "--out",
            str(tmp_path / "outputs"),
            "--backend",
            "missing",
        ],
    )

    assert result.exit_code != 0
    assert "unsupported motion-correction backend" in result.output


def test_correct_motion_command_rejects_unknown_storage_policy(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli.app,
        [
            "correct-motion",
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--registry",
            str(tmp_path / "artifacts.json"),
            "--out",
            str(tmp_path / "outputs"),
            "--storage-policy",
            "forever",
        ],
    )

    assert result.exit_code != 0
    assert "unsupported storage policy" in result.output


def test_correct_motion_command_passes_motioncor3_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    movie_path = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    _write_mrc(movie_path, np.ones((2, 2, 2), dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(movie_path).model_dump_json())
    registry_path = tmp_path / "artifacts.json"
    ArtifactRegistry.empty().write(registry_path)
    executable = tmp_path / "MotionCor3"
    executable.touch()
    gain_path = tmp_path / "gain.mrc"
    _write_mrc(gain_path, np.ones((2, 2), dtype=np.float32))
    captured: dict[str, object] = {}

    class FakeMotionCor3Backend:
        name = "motioncor3"

        def correct(
            self,
            manifest: TiltSeriesManifest,
            context,
        ) -> list[Artifact]:
            captured["manifest"] = manifest
            captured["context"] = context
            output_path = context.output_dir / "fake-motioncor3.mrc"
            _write_mrc(output_path, np.ones((2, 2), dtype=np.float32))
            return [
                Artifact(
                    id="TS_TEST:corrected_projection:000",
                    kind=ArtifactKind.CORRECTED_PROJECTION,
                    path=output_path,
                    shape=(2, 2),
                    dtype="float32",
                    axis_order=AxisOrder.YX,
                    pixel_spacing_angstrom=0.7,
                )
            ]

    monkeypatch.setattr(cli, "_motion_backend", lambda name: FakeMotionCor3Backend())

    result = CliRunner().invoke(
        cli.app,
        [
            "correct-motion",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--backend",
            "motioncor3",
            "--device",
            "cuda",
            "--motioncor3-executable",
            str(executable),
            "--motioncor3-gpu",
            "2",
            "--motioncor3-patch-x",
            "7",
            "--motioncor3-patch-y",
            "5",
            "--motioncor3-pixel-size",
            "0.7",
            "--motioncor3-gain",
            str(gain_path),
            "--motioncor3-gain-rotation",
            "1",
            "--motioncor3-gain-flip",
            "2",
        ],
    )

    assert result.exit_code == 0
    context = captured["context"]
    assert context.device == DevicePreference.CUDA
    assert context.parameters["motioncor3_executable"] == executable
    assert context.parameters["motioncor3_gpu_ids"] == [2]
    assert context.parameters["motioncor3_patch_x"] == 7
    assert context.parameters["motioncor3_patch_y"] == 5
    assert context.parameters["motioncor3_pixel_size_angstrom"] == 0.7
    assert context.parameters["motioncor3_gain_reference"] == gain_path
    assert context.parameters["motioncor3_gain_rotation"] == 1
    assert context.parameters["motioncor3_gain_flip"] == 2


def test_motion_backend_selector_supports_motioncor3() -> None:
    assert isinstance(cli._motion_backend("motioncor3"), MotionCor3MotionCorrectionBackend)


def test_restoration_backend_selector_supports_isonet2() -> None:
    assert isinstance(cli._restoration_backend("isonet2"), IsoNet2RestorationBackend)


def test_segmentation_backend_selector_supports_membrain_seg() -> None:
    assert isinstance(cli._segmentation_backend("membrain-seg"), MemBrainSegSegmentationBackend)


def test_prepare_tilt_series_command_corrects_movies_and_builds_stack(
    tmp_path: Path,
) -> None:
    first_movie = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    second_movie = tmp_path / "frames" / "TS_TEST_001_3.0.mrc"
    _write_mrc(first_movie, np.full((2, 2, 2), 10, dtype=np.float32))
    _write_mrc(second_movie, np.full((2, 2, 2), 20, dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        _manifest_from_movies([(first_movie, 0, 0.0), (second_movie, 1, 3.0)]).model_dump_json()
    )
    registry_path = tmp_path / "artifacts.json"
    ArtifactRegistry.empty().write(registry_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "prepare-tilt-series",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
            "--storage-policy",
            "debug",
        ],
    )

    assert result.exit_code == 0
    assert "wrote 2 corrected projections" in result.output
    assert "wrote 2 tilt-series preparation artifacts" in result.output

    registry = ArtifactRegistry.load(registry_path)
    corrected = registry.by_kind(ArtifactKind.CORRECTED_PROJECTION)
    stacks = registry.by_kind(ArtifactKind.TILT_STACK)
    angles = registry.by_kind(ArtifactKind.TILT_ANGLES)
    assert len(corrected) == 2
    assert len(stacks) == 1
    assert len(angles) == 1
    assert angles[0].path.read_text() == "0.000000\n3.000000\n"

    with mrcfile.open(stacks[0].path, permissive=True) as stack:
        assert stack.data.shape == (2, 2, 2)
        np.testing.assert_allclose(stack.data[0], np.full((2, 2), 10, dtype=np.float32))
        np.testing.assert_allclose(stack.data[1], np.full((2, 2), 20, dtype=np.float32))


def test_align_tilt_series_command_updates_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stack_path = tmp_path / "TS_TEST.zarr"
    zarr.save(stack_path, np.arange(32, dtype=np.float32).reshape(2, 4, 4))
    stack_artifact = Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=stack_path,
        shape=(2, 4, 4),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=1.35,
        parameters={"tilt_series_id": "TS_TEST"},
    )
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.add(stack_artifact)
    registry.write(registry_path)
    manifest = _manifest_from_movies(
        [
            (tmp_path / "movie_0.mrc", 0, 0.0),
            (tmp_path / "movie_1.mrc", 1, 3.0),
        ]
    )
    for image in manifest.images:
        image.rotation_angle_deg = 175.3
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    imod_dir = tmp_path / "imod"
    executable = imod_dir / "bin" / "tiltxcorr"
    executable.parent.mkdir(parents=True)
    executable.touch()
    (executable.parent / "xftoxg").touch()

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        if Path(command[0]).name == "tiltxcorr":
            output_path = Path(command[command.index("-output") + 1])
            output_path.write_text("1 0 0 1 0 0\n1 0 0 1 1 -1\n")
        else:
            output_path = Path(command[command.index("-goutput") + 1])
            output_path.write_text("1 0 0 1 0 0\n1 0 0 1 2 -2\n")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    backend = ImodTiltXcorrAlignmentBackend(fake_runner)
    monkeypatch.setattr(cli, "_alignment_backend", lambda name: backend)

    result = CliRunner().invoke(
        cli.app,
        [
            "align-tilt-series",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--binning",
            "2",
            "--imod-dir",
            str(imod_dir),
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "wrote coarse alignment" in result.output
    alignments = ArtifactRegistry.load(registry_path).by_kind(ArtifactKind.ALIGNMENT)
    assert len(alignments) == 1
    assert alignments[0].parent_ids == [stack_artifact.id]
    assert Path(alignments[0].parameters["imod_xf_path"]).is_file()


def test_qc_coarse_alignment_command_updates_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stack_path = tmp_path / "TS_TEST.zarr"
    zarr.save(stack_path, np.arange(16, dtype=np.float32).reshape(1, 4, 4))
    stack_artifact = Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=stack_path,
        shape=(1, 4, 4),
        dtype="float32",
        axis_order=AxisOrder.TYX,
        parameters={"tilt_series_id": "TS_TEST"},
    )
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text("{}")
    alignment_artifact = Artifact(
        id="TS_TEST:alignment:coarse",
        kind=ArtifactKind.ALIGNMENT,
        path=alignment_path,
        parent_ids=[stack_artifact.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "stage": "coarse",
        },
    )
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.extend([stack_artifact, alignment_artifact])
    registry.write(registry_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(tmp_path / "movie.mrc").model_dump_json())

    class FakeQcBackend:
        name = "fake"

        def evaluate(
            self,
            tilt_stack: Artifact,
            alignment: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> list[Artifact]:
            preview_path = context.output_dir / "preview.st"
            report_path = context.output_dir / "qc.json"
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview_path.write_bytes(b"preview")
            report_path.write_text("{}")
            preview = Artifact(
                id=f"{manifest.tilt_series_id}:qc:coarse_alignment:preview",
                kind=ArtifactKind.QC,
                path=preview_path,
                parent_ids=[tilt_stack.id, alignment.id],
            )
            report = Artifact(
                id=f"{manifest.tilt_series_id}:qc:coarse_alignment:report",
                kind=ArtifactKind.QC,
                path=report_path,
                parent_ids=[tilt_stack.id, alignment.id, preview.id],
                parameters={"status": "pass"},
            )
            return [preview, report]

    monkeypatch.setattr(
        cli,
        "_coarse_alignment_qc_backend",
        lambda name: FakeQcBackend(),
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "qc-coarse-alignment",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--preview-binning",
            "2",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "QC status: pass" in result.output
    assert len(ArtifactRegistry.load(registry_path).by_kind(ArtifactKind.QC)) == 2


def test_fiducial_seed_and_tracking_commands_update_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_stack = Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=tmp_path / "source.zarr",
        parameters={"tilt_series_id": "TS_TEST"},
    )
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text("{}")
    alignment = Artifact(
        id="TS_TEST:alignment:coarse",
        kind=ArtifactKind.ALIGNMENT,
        path=alignment_path,
        parent_ids=[source_stack.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "stage": "coarse",
        },
    )
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment])
    registry.write(registry_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(tmp_path / "movie.mrc").model_dump_json())

    class FakeSeedBackend:
        name = "fake-seed"

        def generate(
            self,
            tilt_stack: Artifact,
            alignment_artifact: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> list[Artifact]:
            assert context.parameters["raw_pixel_spacing_angstrom"] == 0.7
            assert context.parameters["fiducial_diameter_unbinned_px"] == 140.0
            output_dir = context.output_dir / "fiducials"
            output_dir.mkdir(parents=True)
            tracking_path = output_dir / "tracking.st"
            seed_path = output_dir / "seed.mod"
            report_path = output_dir / "seed_qc.json"
            tracking_path.write_bytes(b"tracking")
            seed_path.write_bytes(b"seed")
            report_path.write_text("{}")
            tracking = Artifact(
                id=f"{manifest.tilt_series_id}:aligned_tilt_stack:fiducial",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=tracking_path,
                parent_ids=[tilt_stack.id, alignment_artifact.id],
                parameters={
                    "tilt_series_id": manifest.tilt_series_id,
                    "purpose": "fiducial_tracking",
                },
            )
            seed = Artifact(
                id=f"{manifest.tilt_series_id}:fiducial_seed",
                kind=ArtifactKind.FIDUCIAL_SEED_MODEL,
                path=seed_path,
                parent_ids=[tracking.id],
                parameters={"tilt_series_id": manifest.tilt_series_id},
            )
            report = Artifact(
                id=f"{manifest.tilt_series_id}:qc:fiducial_seed",
                kind=ArtifactKind.QC,
                path=report_path,
                parent_ids=[seed.id],
                parameters={"status": "pass"},
            )
            return [tracking, seed, report]

    class FakeTrackingBackend:
        name = "fake-tracking"

        def track(
            self,
            tracking_stack: Artifact,
            seed_model: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> list[Artifact]:
            output_dir = context.output_dir / "fiducials"
            model_path = output_dir / "tracked.fid"
            report_path = output_dir / "tracking_qc.json"
            model_path.write_bytes(b"fid")
            report_path.write_text("{}")
            model = Artifact(
                id=f"{manifest.tilt_series_id}:fiducial_model",
                kind=ArtifactKind.FIDUCIAL_MODEL,
                path=model_path,
                parent_ids=[tracking_stack.id, seed_model.id],
                parameters={"tilt_series_id": manifest.tilt_series_id},
            )
            report = Artifact(
                id=f"{manifest.tilt_series_id}:qc:fiducial_tracking",
                kind=ArtifactKind.QC,
                path=report_path,
                parent_ids=[model.id],
                parameters={"status": "warning"},
            )
            return [model, report]

    class FakeFineAlignmentBackend:
        name = "fake-fine"

        def align(
            self,
            tracking_stack: Artifact,
            fiducial_model: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> list[Artifact]:
            output_dir = context.output_dir / "alignments"
            output_dir.mkdir(parents=True)
            alignment_path = output_dir / "fine.json"
            transform_path = output_dir / "fine.xf"
            report_path = output_dir / "fine_qc.json"
            alignment_path.write_text("{}")
            transform_path.write_text("1 0 0 1 0 0\n")
            report_path.write_text("{}")
            alignment = Artifact(
                id=f"{manifest.tilt_series_id}:alignment:fine",
                kind=ArtifactKind.ALIGNMENT,
                path=alignment_path,
                parent_ids=[tracking_stack.id, fiducial_model.id],
                parameters={
                    "tilt_series_id": manifest.tilt_series_id,
                    "stage": "fine",
                    "imod_xf_path": str(transform_path),
                },
            )
            report = Artifact(
                id=f"{manifest.tilt_series_id}:qc:alignment:fine",
                kind=ArtifactKind.QC,
                path=report_path,
                parent_ids=[alignment.id],
                parameters={"status": "pass"},
            )
            return [alignment, report]

    class FakeFinalStackBackend:
        name = "fake-final-stack"

        def build(
            self,
            tilt_stack: Artifact,
            fine_alignment: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> Artifact:
            output_dir = context.output_dir / "aligned"
            output_dir.mkdir(parents=True)
            stack_path = output_dir / "final.st"
            tilt_path = output_dir / "final.tlt"
            stack_path.write_bytes(b"aligned")
            tilt_path.write_text("0\n")
            return Artifact(
                id=f"{manifest.tilt_series_id}:aligned_tilt_stack:final",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=stack_path,
                parent_ids=[tilt_stack.id, fine_alignment.id],
                parameters={
                    "tilt_series_id": manifest.tilt_series_id,
                    "purpose": "final_alignment",
                    "tilt_file_path": str(tilt_path),
                },
            )

    monkeypatch.setattr(
        cli,
        "_fiducial_seed_backend",
        lambda name: FakeSeedBackend(),
    )
    monkeypatch.setattr(
        cli,
        "_fiducial_tracking_backend",
        lambda name: FakeTrackingBackend(),
    )
    monkeypatch.setattr(
        cli,
        "_fine_alignment_backend",
        lambda name: FakeFineAlignmentBackend(),
    )
    monkeypatch.setattr(
        cli,
        "_final_aligned_stack_backend",
        lambda name: FakeFinalStackBackend(),
    )
    runner = CliRunner()
    seed_result = runner.invoke(
        cli.app,
        [
            "generate-fiducial-seed",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--raw-pixel-spacing",
            "0.7",
            "--fiducial-diameter-px",
            "140",
            "--device",
            "cpu",
        ],
    )
    tracking_result = runner.invoke(
        cli.app,
        [
            "track-fiducials",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
        ],
    )
    fine_result = runner.invoke(
        cli.app,
        [
            "fine-align-tilt-series",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--device",
            "cpu",
        ],
    )
    final_stack_result = runner.invoke(
        cli.app,
        [
            "build-final-aligned-stack",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--output-binning",
            "8",
            "--device",
            "cpu",
        ],
    )

    assert seed_result.exit_code == 0
    assert "wrote fiducial seed model" in seed_result.output
    assert "QC status: pass" in seed_result.output
    assert tracking_result.exit_code == 0
    assert "wrote tracked fiducial model" in tracking_result.output
    assert "QC status: warning" in tracking_result.output
    assert fine_result.exit_code == 0
    assert "wrote fine alignment" in fine_result.output
    assert "QC status: pass" in fine_result.output
    assert final_stack_result.exit_code == 0
    assert "wrote final aligned stack" in final_stack_result.output
    updated = ArtifactRegistry.load(registry_path)
    assert len(updated.by_kind(ArtifactKind.FIDUCIAL_SEED_MODEL)) == 1
    assert len(updated.by_kind(ArtifactKind.FIDUCIAL_MODEL)) == 1
    fine_alignments = [
        artifact
        for artifact in updated.by_kind(ArtifactKind.ALIGNMENT)
        if artifact.parameters.get("stage") == "fine"
    ]
    assert len(fine_alignments) == 1
    final_stacks = [
        artifact
        for artifact in updated.by_kind(ArtifactKind.ALIGNED_TILT_STACK)
        if artifact.parameters.get("purpose") == "final_alignment"
    ]
    assert len(final_stacks) == 1


def test_reconstruct_tomogram_command_updates_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_stack = Artifact(
        id="TS_TEST:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=tmp_path / "source.zarr",
        parameters={"tilt_series_id": "TS_TEST"},
    )
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text("{}")
    alignment = Artifact(
        id="TS_TEST:alignment:fine",
        kind=ArtifactKind.ALIGNMENT,
        path=alignment_path,
        parent_ids=[source_stack.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "stage": "fine",
            "recommended_x_axis_tilt_deg": -1.0,
            "recommended_unbinned_thickness_px": 4.0,
            "recommended_unbinned_z_shift_px": 0.0,
        },
    )
    aligned_path = tmp_path / "aligned.st"
    aligned_path.write_bytes(b"aligned")
    aligned_stack = Artifact(
        id="TS_TEST:aligned_tilt_stack:final",
        kind=ArtifactKind.ALIGNED_TILT_STACK,
        path=aligned_path,
        parent_ids=[source_stack.id, alignment.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "purpose": "final_alignment",
            "alignment_stage": "fine",
        },
    )
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.extend([source_stack, alignment, aligned_stack])
    registry.write(registry_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(tmp_path / "movie.mrc").model_dump_json())

    class FakeReconstructionBackend:
        name = "fake"

        def reconstruct(
            self,
            tilt_stack: Artifact,
            alignment_artifact: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> list[Artifact]:
            tomogram_path = context.output_dir / "tomogram.zarr"
            rec_path = context.output_dir / "tomogram.rec"
            report_path = context.output_dir / "reconstruction_qc.json"
            tomogram_path.mkdir(parents=True)
            rec_path.write_bytes(b"rec")
            report_path.write_text("{}")
            tomogram = Artifact(
                id=f"{manifest.tilt_series_id}:tomogram:fine",
                kind=ArtifactKind.TOMOGRAM,
                path=tomogram_path,
                parent_ids=[tilt_stack.id, alignment_artifact.id],
                parameters={"imod_rec_path": str(rec_path)},
            )
            report = Artifact(
                id=f"{manifest.tilt_series_id}:qc:reconstruction:fine",
                kind=ArtifactKind.QC,
                path=report_path,
                parent_ids=[tomogram.id],
                parameters={"status": "warning"},
            )
            return [tomogram, report]

    monkeypatch.setattr(
        cli,
        "_reconstruction_backend",
        lambda name: FakeReconstructionBackend(),
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "reconstruct-tomogram",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--thickness",
            "2",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "wrote canonical tomogram" in result.output
    assert "QC status: warning" in result.output
    updated = ArtifactRegistry.load(registry_path)
    assert len(updated.by_kind(ArtifactKind.TOMOGRAM)) == 1


def test_restore_tomogram_command_updates_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tomogram_path = tmp_path / "tomogram.zarr"
    zarr.save(tomogram_path, np.ones((2, 3, 4), dtype=np.float32))
    tomogram = Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=tomogram_path,
        shape=(2, 3, 4),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=13.5,
        parameters={
            "tilt_series_id": "TS_TEST",
            "tomogram_branch": "full",
        },
    )
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    registry.write(registry_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(tmp_path / "movie.mrc").model_dump_json())
    executable = tmp_path / "isonet.py"
    executable.touch()
    model = tmp_path / "isonet_model.h5"
    model.touch()

    class FakeRestorationBackend:
        name = "isonet2"

        def denoise(
            self,
            input_tomogram: Artifact,
            manifest: TiltSeriesManifest,
            context,
        ) -> Artifact:
            assert input_tomogram == tomogram
            assert context.parameters["isonet2_executable"] == executable
            assert context.parameters["isonet2_model"] == model
            assert context.parameters["isonet2_gpu_id"] == "0"
            assert context.parameters["isonet2_cube_size"] == 80
            assert context.parameters["isonet2_crop_size"] == 112
            assert context.parameters["isonet2_batch_size"] == 2
            assert context.parameters["isonet2_number_subtomos"] == 25
            assert context.parameters["isonet2_normalize_percentile"] is False
            output_path = context.output_dir / "restored.zarr"
            qc_path = context.output_dir / "restoration_qc.json"
            output_path.mkdir(parents=True)
            qc_path.write_text("{}")
            return Artifact(
                id=f"{manifest.tilt_series_id}:tomogram:full:isonet2",
                kind=ArtifactKind.DENOISED_TOMOGRAM,
                path=output_path,
                parent_ids=[input_tomogram.id],
                parameters={
                    "backend": "isonet2",
                    "tilt_series_id": manifest.tilt_series_id,
                    "tomogram_branch": "full",
                    "qc_path": str(qc_path),
                },
            )

    monkeypatch.setattr(
        cli,
        "_restoration_backend",
        lambda name: FakeRestorationBackend(),
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "restore-tomogram",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--isonet2-executable",
            str(executable),
            "--isonet2-model",
            str(model),
            "--isonet2-gpu-id",
            "0",
            "--isonet2-cube-size",
            "80",
            "--isonet2-crop-size",
            "112",
            "--isonet2-batch-size",
            "2",
            "--isonet2-number-subtomos",
            "25",
            "--no-isonet2-normalize-percentile",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "wrote restored tomogram" in result.output
    assert "QC status: pass" in result.output
    updated = ArtifactRegistry.load(registry_path)
    assert len(updated.by_kind(ArtifactKind.DENOISED_TOMOGRAM)) == 1
    assert len(updated.by_kind(ArtifactKind.QC)) == 1


def test_segment_tomogram_command_updates_registry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tomogram_path = tmp_path / "tomogram.zarr"
    zarr.save(tomogram_path, np.ones((2, 3, 4), dtype=np.float32))
    tomogram = Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=tomogram_path,
        shape=(2, 3, 4),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=13.5,
        parameters={
            "tilt_series_id": "TS_TEST",
            "tomogram_branch": "full",
        },
    )
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    registry.write(registry_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest(tmp_path / "movie.mrc").model_dump_json())
    executable = tmp_path / "membrain"
    executable.touch()
    model = tmp_path / "membrain.ckpt"
    model.touch()

    class FakeSegmentationBackend:
        name = "membrain-seg"

        def segment(
            self,
            input_tomogram: Artifact,
            manifest: TiltSeriesManifest,
            context,
            supporting_artifacts: Sequence[Artifact] = (),
        ) -> Artifact:
            assert input_tomogram == tomogram
            assert supporting_artifacts == ()
            assert context.parameters["membrain_executable"] == executable
            assert context.parameters["membrain_model"] == model
            assert context.parameters["membrain_rescale_patches"] is False
            assert context.parameters["membrain_out_pixel_size"] == 12.5
            assert context.parameters["membrain_store_probabilities"] is True
            assert context.parameters["membrain_store_connected_components"] is True
            assert context.parameters["membrain_connected_component_threshold"] == 200
            assert context.parameters["membrain_test_time_augmentation"] is False
            assert context.parameters["membrain_store_uncertainty_map"] is True
            assert context.parameters["membrain_segmentation_threshold"] == 0.4
            assert context.parameters["membrain_sliding_window_size"] == 96
            assert context.parameters["membrain_output_name"] == "segmentation.mrc"
            output_path = context.output_dir / "segmentation.zarr"
            qc_path = context.output_dir / "segmentation_qc.json"
            output_path.mkdir(parents=True)
            qc_path.write_text("{}")
            return Artifact(
                id=f"{manifest.tilt_series_id}:segmentation:full:membrain-seg",
                kind=ArtifactKind.SEGMENTATION,
                path=output_path,
                parent_ids=[input_tomogram.id],
                parameters={
                    "backend": "membrain-seg",
                    "tilt_series_id": manifest.tilt_series_id,
                    "tomogram_branch": "full",
                    "qc_path": str(qc_path),
                },
            )

    monkeypatch.setattr(
        cli,
        "_segmentation_backend",
        lambda name: FakeSegmentationBackend(),
    )
    result = CliRunner().invoke(
        cli.app,
        [
            "segment-tomogram",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "outputs"),
            "--membrain-executable",
            str(executable),
            "--membrain-model",
            str(model),
            "--no-membrain-rescale-patches",
            "--membrain-out-pixel-size",
            "12.5",
            "--membrain-store-probabilities",
            "--membrain-store-connected-components",
            "--membrain-connected-component-threshold",
            "200",
            "--no-membrain-test-time-augmentation",
            "--membrain-store-uncertainty-map",
            "--membrain-segmentation-threshold",
            "0.4",
            "--membrain-sliding-window-size",
            "96",
            "--membrain-output-name",
            "segmentation.mrc",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "wrote segmentation" in result.output
    assert "QC status: pass" in result.output
    updated = ArtifactRegistry.load(registry_path)
    assert len(updated.by_kind(ArtifactKind.SEGMENTATION)) == 1
    assert len(updated.by_kind(ArtifactKind.QC)) == 1


def _manifest(movie_path: Path) -> TiltSeriesManifest:
    return _manifest_from_movies([(movie_path, 0, 0.0)])


def _manifest_from_movies(
    movies: list[tuple[Path, int, float]],
) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            _tilt_image(movie_path, z_value=z_value, tilt_angle=tilt_angle)
            for movie_path, z_value, tilt_angle in movies
        ],
    )


def _tilt_image(movie_path: Path, *, z_value: int, tilt_angle: float) -> TiltImage:
    return TiltImage(
        z_value=z_value,
        tilt_angle_deg=tilt_angle,
        subframe_path=f"frames/TS_TEST_{z_value:03d}_{tilt_angle}.mrc",
        local_frame_file=movie_path,
        num_subframes=2,
        pixel_spacing_angstrom=5.4,
        binning=4,
    )


def _write_mrc(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
