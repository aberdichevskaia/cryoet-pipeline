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


def test_descends_from_walks_past_direct_parents(tmp_path: Path) -> None:
    """Inserting a stage must not sever the lineage the later stage checks."""

    registry = ArtifactRegistry.empty()
    registry.extend(
        [
            Artifact(id="raw", kind=ArtifactKind.RAW_TILT_MOVIE, path=tmp_path / "raw.mrc"),
            Artifact(
                id="aligned",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=tmp_path / "aligned.st",
                parent_ids=["raw"],
            ),
            Artifact(
                id="corrected",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=tmp_path / "corrected.st",
                parent_ids=["aligned"],
            ),
            Artifact(
                id="sibling",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=tmp_path / "sibling.st",
                parent_ids=["raw"],
            ),
        ]
    )

    assert registry.descends_from("corrected", "raw")
    assert registry.descends_from("corrected", "aligned")
    assert not registry.descends_from("corrected", "sibling")
    assert not registry.descends_from("raw", "corrected"), "lineage has a direction"
    assert not registry.descends_from("corrected", "corrected"), "nothing descends from itself"
    assert not registry.descends_from("absent", "raw")


def test_replacing_an_entry_moves_it_below_newly_inserted_parents(tmp_path: Path) -> None:
    """Re-running a stage after a new stage appeared upstream must not be rejected.

    The tomogram was registered before CTF correction existed. Rebuilding it
    from the corrected stack gives it a parent that sits later in the registry,
    so keeping its old position would break parents-before-children.
    """

    registry = ArtifactRegistry.empty()
    aligned = Artifact(
        id="aligned",
        kind=ArtifactKind.ALIGNED_TILT_STACK,
        path=tmp_path / "aligned.st",
    )
    tomogram = Artifact(
        id="tomogram",
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "tomogram.zarr",
        parent_ids=["aligned"],
    )
    tomogram_qc = Artifact(
        id="tomogram-qc",
        kind=ArtifactKind.QC,
        path=tmp_path / "tomogram_qc.json",
        parent_ids=["tomogram"],
    )
    corrected = Artifact(
        id="corrected",
        kind=ArtifactKind.ALIGNED_TILT_STACK,
        path=tmp_path / "corrected.st",
        parent_ids=["aligned"],
    )
    registry.extend([aligned, tomogram, tomogram_qc, corrected])

    registry.add(tomogram.model_copy(update={"parent_ids": ["corrected"]}), replace=True)

    order = [artifact.id for artifact in registry.artifacts]
    assert order == ["aligned", "corrected", "tomogram", "tomogram-qc"]
    # The invariant the registry enforces on load must still hold.
    ArtifactRegistry.model_validate(registry.model_dump())


def test_replacing_an_entry_leaves_a_valid_order_untouched(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()
    first = Artifact(id="a", kind=ArtifactKind.TOMOGRAM, path=tmp_path / "a.zarr")
    second = Artifact(
        id="b", kind=ArtifactKind.TOMOGRAM, path=tmp_path / "b.zarr", parent_ids=["a"]
    )
    third = Artifact(id="c", kind=ArtifactKind.TOMOGRAM, path=tmp_path / "c.zarr")
    registry.extend([first, second, third])

    registry.add(second.model_copy(update={"dtype": "float32"}), replace=True)

    assert [artifact.id for artifact in registry.artifacts] == ["a", "b", "c"]


def test_replacing_an_entry_rejects_an_unknown_parent(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()
    tomogram = Artifact(id="tomogram", kind=ArtifactKind.TOMOGRAM, path=tmp_path / "t.zarr")
    registry.add(tomogram)

    with pytest.raises(ValueError, match="parent artifacts must be registered first"):
        registry.add(tomogram.model_copy(update={"parent_ids": ["absent"]}), replace=True)


def test_replacing_an_entry_rejects_a_cycle(tmp_path: Path) -> None:
    registry = ArtifactRegistry.empty()
    parent = Artifact(id="parent", kind=ArtifactKind.TOMOGRAM, path=tmp_path / "p.zarr")
    child = Artifact(
        id="child", kind=ArtifactKind.TOMOGRAM, path=tmp_path / "c.zarr", parent_ids=["parent"]
    )
    registry.extend([parent, child])

    with pytest.raises(ValueError, match="cycle"):
        registry.add(parent.model_copy(update={"parent_ids": ["child"]}), replace=True)


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


def test_scope_is_adopted_from_the_single_parameter_every_backend_writes(
    tmp_path: Path,
) -> None:
    """Every registry written before the field existed carries only this."""

    artifact = Artifact(
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "t.mrc",
        parameters={"tilt_series_id": "TS_0005"},
    )

    assert artifact.tilt_series_ids == ["TS_0005"]
    assert artifact.covers("TS_0005")
    assert not artifact.covers("TS_0006")


def test_scope_is_adopted_from_the_list_a_cross_series_average_writes(
    tmp_path: Path,
) -> None:
    """The case the field exists for: one artifact about several tilt-series."""

    average = Artifact(
        kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
        path=tmp_path / "avg.mrc",
        parameters={"tilt_series_ids": ["TS_0005", "TS_0006"]},
    )

    assert average.tilt_series_ids == ["TS_0005", "TS_0006"]
    assert average.covers("TS_0005")
    assert average.covers("TS_0006")


def test_an_artifact_about_no_particular_series_has_an_empty_scope(
    tmp_path: Path,
) -> None:
    """A shared template belongs to the project, not to one tilt-series."""

    template = Artifact(kind=ArtifactKind.TEMPLATE, path=tmp_path / "ref.mrc")

    assert template.tilt_series_ids == []
    assert not template.covers("TS_0005")


def test_an_explicit_scope_is_not_overwritten_by_adoption(tmp_path: Path) -> None:
    artifact = Artifact(
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "t.mrc",
        tilt_series_ids=["TS_0005"],
        parameters={"tilt_series_id": "TS_0005"},
    )

    assert artifact.tilt_series_ids == ["TS_0005"]


def test_a_scope_that_disagrees_with_its_parameters_is_refused(tmp_path: Path) -> None:
    """The promotion is only worth having if the two homes cannot diverge.

    Nothing writes both today, so a disagreement means a bug — and a silent one,
    since a resolver would read the field while a human read the parameter.
    """

    with pytest.raises(ValidationError, match="scope disagrees with its parameters"):
        Artifact(
            kind=ArtifactKind.TOMOGRAM,
            path=tmp_path / "t.mrc",
            tilt_series_ids=["TS_0009"],
            parameters={"tilt_series_id": "TS_0005"},
        )


def test_scope_survives_a_round_trip_through_json(tmp_path: Path) -> None:
    """What a rebuilt index will actually read, and the column it will fill."""

    average = Artifact(
        kind=ArtifactKind.SUBTOMOGRAM_AVERAGE,
        path=tmp_path / "avg.mrc",
        parameters={"tilt_series_ids": ["TS_0005", "TS_0006"]},
    )

    restored = Artifact.model_validate_json(average.model_dump_json())

    assert restored.tilt_series_ids == ["TS_0005", "TS_0006"]


def test_model_copy_does_not_re_derive_the_scope(tmp_path: Path) -> None:
    """Pinned because it is the one way the two homes can still diverge.

    `model_copy` runs no validators, so updating `parameters` alone leaves the
    field saying what it said before. The invariant holds where a registry
    actually crosses it — construction and deserialization — and not here.
    Negative tests rely on this to build artifacts that violate cross-field
    rules, which is why the hole is documented rather than closed by overriding
    `model_copy`: a caller updating scope must update both, or construct anew.
    """

    artifact = Artifact(
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "t.mrc",
        parameters={"tilt_series_id": "TS_0005"},
    )

    stale = artifact.model_copy(update={"parameters": {"tilt_series_id": "TS_OTHER"}})

    assert stale.tilt_series_ids == ["TS_0005"]
    with pytest.raises(ValidationError, match="scope disagrees with its parameters"):
        Artifact.model_validate(stale.model_dump())
