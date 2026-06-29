from pathlib import Path

import pytest
from pydantic import ValidationError

from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    ProjectConfig,
    RetentionPolicy,
    StorageRole,
)


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


def test_new_artifact_kinds_serialize_as_stable_values() -> None:
    assert AxisOrder.YX.value == "yx"
    assert ArtifactKind.DENOISED_TOMOGRAM.value == "denoised_tomogram"
    assert ArtifactKind.SEGMENTATION.value == "segmentation"
    assert ArtifactKind.PICKS.value == "picks"
    assert ArtifactKind.TILT_ANGLES.value == "tilt_angles"
    assert ArtifactKind.DATASET_EXPORT.value == "dataset_export"


def test_artifact_storage_metadata_serializes_as_stable_values() -> None:
    artifact = Artifact(
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=Path("outputs/corrected.mrc"),
        storage_role=StorageRole.EXPORT,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        size_bytes=128,
    )

    payload = artifact.model_dump(mode="json")

    assert payload["storage_role"] == "export"
    assert payload["retention_policy"] == "keep"
    assert payload["can_recompute"] is False
    assert payload["size_bytes"] == 128


def test_alignment_transform_rejects_nonfinite_values() -> None:
    with pytest.raises(ValidationError, match="must be finite"):
        AlignmentTransform(
            z_value=0,
            tilt_angle_deg=0.0,
            a11=1.0,
            a12=0.0,
            a21=0.0,
            a22=1.0,
            shift_x_px=float("nan"),
            shift_y_px=0.0,
        )
