from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from cryoet_pipeline.models import RetentionPolicy, StorageRole


class ArtifactFormat(StrEnum):
    """On-disk representation for array artifacts."""

    MRC = "mrc"
    ZARR = "zarr"


class StoragePolicyName(StrEnum):
    """Named storage policies for balancing debuggability and disk usage."""

    DEBUG = "debug"
    WORKING = "working"
    MINIMAL = "minimal"


@dataclass(frozen=True)
class StoragePolicy:
    name: StoragePolicyName
    artifact_format: ArtifactFormat
    storage_role: StorageRole
    retention_policy: RetentionPolicy
    can_recompute: bool


def resolve_storage_policy(value: str | StoragePolicyName) -> StoragePolicy:
    """Resolve a user-facing storage policy name into concrete artifact defaults."""

    try:
        name = StoragePolicyName(str(value).lower())
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in StoragePolicyName)
        raise ValueError(
            f"unsupported storage policy {value!r}; expected one of: {allowed}"
        ) from exc

    if name is StoragePolicyName.DEBUG:
        return StoragePolicy(
            name=name,
            artifact_format=ArtifactFormat.MRC,
            storage_role=StorageRole.CACHE,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
        )

    if name is StoragePolicyName.WORKING:
        return StoragePolicy(
            name=name,
            artifact_format=ArtifactFormat.ZARR,
            storage_role=StorageRole.CACHE,
            retention_policy=RetentionPolicy.RECOMPUTE,
            can_recompute=True,
        )

    return StoragePolicy(
        name=name,
        artifact_format=ArtifactFormat.ZARR,
        storage_role=StorageRole.TEMPORARY,
        retention_policy=RetentionPolicy.DELETE_AFTER_EXPORT,
        can_recompute=True,
    )
