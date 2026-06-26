from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cryoet_pipeline.models import Artifact, ArtifactKind


class ArtifactRegistry(BaseModel):
    """Ordered registry of artifacts produced or exported by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    artifacts: list[Artifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ids_and_lineage_order(self) -> Self:
        """Require unique ids and parent artifacts to appear before children."""

        seen: set[str] = set()
        for artifact in self.artifacts:
            if artifact.id in seen:
                raise ValueError(f"duplicate artifact id: {artifact.id}")

            missing_parent_ids = [
                parent_id for parent_id in artifact.parent_ids if parent_id not in seen
            ]
            if missing_parent_ids:
                raise ValueError(
                    f"{artifact.id}: parent artifacts must be registered first; "
                    f"missing {missing_parent_ids}"
                )

            seen.add(artifact.id)

        return self

    @classmethod
    def empty(cls) -> ArtifactRegistry:
        return cls()

    @classmethod
    def load(cls, path: Path) -> ArtifactRegistry:
        return cls.model_validate_json(path.read_text())

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def add(self, artifact: Artifact) -> None:
        existing_ids = self.artifact_ids
        if artifact.id in existing_ids:
            raise ValueError(f"duplicate artifact id: {artifact.id}")

        missing_parent_ids = [
            parent_id for parent_id in artifact.parent_ids if parent_id not in existing_ids
        ]
        if missing_parent_ids:
            raise ValueError(
                f"{artifact.id}: parent artifacts must be registered first; "
                f"missing {missing_parent_ids}"
            )

        self.artifacts.append(artifact)

    def extend(self, artifacts: Iterable[Artifact]) -> None:
        for artifact in artifacts:
            self.add(artifact)

    def get(self, artifact_id: str) -> Artifact:
        for artifact in self.artifacts:
            if artifact.id == artifact_id:
                return artifact
        raise KeyError(f"unknown artifact id: {artifact_id}")

    def by_kind(self, kind: ArtifactKind) -> list[Artifact]:
        return [artifact for artifact in self.artifacts if artifact.kind == kind]

    def children_of(self, artifact_id: str) -> list[Artifact]:
        return [
            artifact for artifact in self.artifacts if artifact_id in artifact.parent_ids
        ]

    @property
    def artifact_ids(self) -> set[str]:
        return {artifact.id for artifact in self.artifacts}
