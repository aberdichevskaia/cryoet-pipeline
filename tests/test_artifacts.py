from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, StorageRole


def test_artifact_registry_adds_and_queries_lineage(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()
    raw = Artifact(
        id="raw-0",
        kind=ArtifactKind.RAW_TILT_MOVIE,
        path=tmp_path / "raw.mrc",
        shape=(8, 16, 16),
        dtype="int8",
        axis_order=AxisOrder.FYX,
        storage_role=StorageRole.SOURCE,
        can_recompute=False,
        size_bytes=2048,
    )
    corrected = Artifact(
        id="corrected-0",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=tmp_path / "corrected.zarr",
        parent_ids=["raw-0"],
        shape=(16, 16),
        dtype="float32",
        storage_role=StorageRole.CACHE,
        size_bytes=1024,
    )

    registry.extend([raw, corrected])

    assert registry.get("raw-0") == raw
    assert registry.by_kind(ArtifactKind.CORRECTED_PROJECTION) == [corrected]
    assert registry.by_storage_role(StorageRole.SOURCE) == [raw]
    assert registry.children_of("raw-0") == [corrected]
    assert registry.artifact_ids == {"raw-0", "corrected-0"}
    assert registry.total_size_bytes == 3072
    assert registry.size_bytes_by_storage_role[StorageRole.SOURCE] == 2048
    assert registry.size_bytes_by_storage_role[StorageRole.CACHE] == 1024


def test_artifact_registry_roundtrips_json(tmp_path: Path) -> None:
    path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.add(
        Artifact(
            id="tomogram",
            kind=ArtifactKind.TOMOGRAM,
            path=Path("outputs/TS_01/TS_01.rec"),
            shape=(32, 64, 64),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            parameters={"backend": "fake"},
        )
    )

    registry.write(path)
    loaded = ArtifactRegistry.load(path)

    assert loaded == registry
    assert loaded.get("tomogram").path == Path("outputs/TS_01/TS_01.rec")
    assert loaded.get("tomogram").parameters == {"backend": "fake"}


def test_artifact_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()
    artifact = Artifact(
        id="duplicate",
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "a.rec",
    )
    registry.add(artifact)

    with pytest.raises(ValueError, match="duplicate artifact id"):
        registry.add(
            Artifact(
                id="duplicate",
                kind=ArtifactKind.QC,
                path=tmp_path / "qc.json",
            )
        )


def test_artifact_registry_can_replace_existing_artifact(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()
    first = Artifact(
        id="projection",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=tmp_path / "first.mrc",
        parameters={"run": "first"},
    )
    second = Artifact(
        id="projection",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=tmp_path / "second.mrc",
        parameters={"run": "second"},
    )

    registry.add(first)
    registry.add(second, replace=True)

    assert len(registry.artifacts) == 1
    assert registry.get("projection").path == tmp_path / "second.mrc"
    assert registry.get("projection").parameters == {"run": "second"}


def test_artifact_registry_rejects_unknown_parent(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()

    with pytest.raises(ValueError, match="parent artifacts must be registered first"):
        registry.add(
            Artifact(
                id="child",
                kind=ArtifactKind.TOMOGRAM,
                path=tmp_path / "child.rec",
                parent_ids=["missing-parent"],
            )
        )


def test_artifact_registry_validates_loaded_lineage_order(tmp_path: Path) -> None:
    parent = Artifact(
        id="parent",
        kind=ArtifactKind.TILT_STACK,
        path=tmp_path / "stack.st",
    )
    child = Artifact(
        id="child",
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "child.rec",
        parent_ids=["parent"],
    )

    with pytest.raises(ValidationError, match="parent artifacts must be registered first"):
        ArtifactRegistry(artifacts=[child, parent])


def test_artifact_registry_raises_key_error_for_unknown_id() -> None:
    registry = ArtifactRegistry.empty()

    with pytest.raises(KeyError, match="unknown artifact id"):
        registry.get("missing")
