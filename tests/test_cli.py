from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import (
    ArtifactKind,
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


def _manifest(movie_path: Path) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frames/TS_TEST_000_0.0.mrc",
                local_frame_file=movie_path,
                num_subframes=2,
                pixel_spacing_angstrom=5.4,
                binning=4,
            )
        ],
    )


def _write_mrc(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
