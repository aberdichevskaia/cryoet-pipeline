from pathlib import Path

from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, ProjectConfig


def test_project_config_defaults_to_mvp_tilt_series() -> None:
    config = ProjectConfig(
        frames_dir=Path("frames"),
        mdocs_dir=Path("mdocs"),
        output_dir=Path("out"),
    )

    assert config.tilt_series == ["TS_01", "TS_43"]
    assert config.device == "auto"


def test_artifact_serializes_lineage() -> None:
    artifact = Artifact(
        kind=ArtifactKind.TOMOGRAM,
        path=Path("outputs/TS_01/TS_01.rec"),
        parent_ids=["stack", "alignment"],
        shape=(256, 512, 512),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
    )

    payload = artifact.model_dump(mode="json")

    assert payload["kind"] == "tomogram"
    assert payload["axis_order"] == "zyx"
    assert payload["parent_ids"] == ["stack", "alignment"]
