import json
from pathlib import Path

import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import ProjectConfig
from cryoet_pipeline.project import initialize_project


def _mdoc_text(series: str, num_subframes: int, count: int = 41) -> str:
    lines = ["PixelSpacing = 5.4"]
    for index in range(count):
        lines.extend(
            [
                "",
                f"[ZValue = {index}]",
                f"TiltAngle = {float(index)}",
                "PixelSpacing = 5.4",
                "Binning = 4",
                f"SubFramePath = D:\\DATA\\frames\\{series}_{index:03d}_{float(index)}.mrc",
                f"NumSubFrames = {num_subframes}",
            ]
        )
    return "\n".join(lines)


def test_initialize_project_writes_manifests(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    mdocs = tmp_path / "mdocs"
    out = tmp_path / "out"
    frames.mkdir()
    mdocs.mkdir()
    (mdocs / "TS_01.mrc.mdoc").write_text(_mdoc_text("TS_01", 8))
    (mdocs / "TS_43.mrc.mdoc").write_text(_mdoc_text("TS_43", 10))
    for series in ["TS_01", "TS_43"]:
        for index in range(41):
            (frames / f"{series}_{index:03d}_{float(index)}.mrc").write_bytes(b"mrc")

    result = initialize_project(
        ProjectConfig(frames_dir=frames, mdocs_dir=mdocs, output_dir=out)
    )

    assert result.project_path.exists()
    assert result.artifact_registry_path.exists()
    assert len(result.manifest_paths) == 2
    assert all(path.exists() for path in result.manifest_paths)

    project_payload = json.loads(result.project_path.read_text())
    assert project_payload["artifact_registry"] == str(result.artifact_registry_path)

    registry = ArtifactRegistry.load(result.artifact_registry_path)
    assert registry.artifacts == []


def test_initialize_project_rejects_missing_frames(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    mdocs = tmp_path / "mdocs"
    out = tmp_path / "out"
    frames.mkdir()
    mdocs.mkdir()
    (mdocs / "TS_01.mrc.mdoc").write_text(_mdoc_text("TS_01", 8))

    with pytest.raises(FileNotFoundError):
        initialize_project(
            ProjectConfig(
                frames_dir=frames,
                mdocs_dir=mdocs,
                output_dir=out,
                tilt_series=["TS_01"],
            )
        )
