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

    artifact = ArtifactRegistry.load(registry_path).by_kind(
        ArtifactKind.CORRECTED_PROJECTION
    )[0]
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


def test_prepare_tilt_series_command_corrects_movies_and_builds_stack(
    tmp_path: Path,
) -> None:
    first_movie = tmp_path / "frames" / "TS_TEST_000_0.0.mrc"
    second_movie = tmp_path / "frames" / "TS_TEST_001_3.0.mrc"
    _write_mrc(first_movie, np.full((2, 2, 2), 10, dtype=np.float32))
    _write_mrc(second_movie, np.full((2, 2, 2), 20, dtype=np.float32))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        _manifest_from_movies(
            [(first_movie, 0, 0.0), (second_movie, 1, 3.0)]
        ).model_dump_json()
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

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        output_path = Path(command[command.index("-output") + 1])
        output_path.write_text("1 0 0 1 0 0\n1 0 0 1 1 -1\n")
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
    manifest_path.write_text(
        _manifest(tmp_path / "movie.mrc").model_dump_json()
    )

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
