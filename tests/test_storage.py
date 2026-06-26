from __future__ import annotations

import pytest

from cryoet_pipeline.models import RetentionPolicy, StorageRole
from cryoet_pipeline.storage import ArtifactFormat, StoragePolicyName, resolve_storage_policy


def test_debug_storage_policy_keeps_mrc_cache_for_visual_inspection() -> None:
    policy = resolve_storage_policy("debug")

    assert policy.name == StoragePolicyName.DEBUG
    assert policy.artifact_format == ArtifactFormat.MRC
    assert policy.storage_role == StorageRole.CACHE
    assert policy.retention_policy == RetentionPolicy.KEEP
    assert policy.can_recompute is True


def test_working_storage_policy_uses_recomputable_zarr_cache() -> None:
    policy = resolve_storage_policy("working")

    assert policy.name == StoragePolicyName.WORKING
    assert policy.artifact_format == ArtifactFormat.ZARR
    assert policy.storage_role == StorageRole.CACHE
    assert policy.retention_policy == RetentionPolicy.RECOMPUTE
    assert policy.can_recompute is True


def test_minimal_storage_policy_marks_zarr_as_temporary() -> None:
    policy = resolve_storage_policy("minimal")

    assert policy.name == StoragePolicyName.MINIMAL
    assert policy.artifact_format == ArtifactFormat.ZARR
    assert policy.storage_role == StorageRole.TEMPORARY
    assert policy.retention_policy == RetentionPolicy.DELETE_AFTER_EXPORT
    assert policy.can_recompute is True


def test_storage_policy_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported storage policy"):
        resolve_storage_policy("forever")
