"""Run records: what every run of a step writes beside its outputs (docs/run_records.md).

A run writes into a temporary folder beside its own, writes `run.json` there last,
and then renames the folder into place: the folder never exists without its record,
and a record that says `succeeded` stands for a whole folder
(docs/pipeline_engine.md). A failed attempt keeps its record,
and whatever it wrote, under `attempts/<run id>/` beside it; one killed outright,
which wrote no record, is moved there by the next attempt (`clear_abandoned`).

The record's fingerprint decides whether a run's work is already done: the same
step, declared versions, content parameters, inputs and selection make the same
fingerprint whatever the device, the commit or the day. Paths are not part of it,
only what the inputs are: identity is not naming (docs/database_design.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cryoet_pipeline.models import Artifact, ArtifactKind

SCHEMA_VERSION = 1
RECORD = "run.json"
ATTEMPTS = "attempts"


class RunInput(BaseModel):
    """What a run read: its slot, where it was, and the fingerprint of what it was."""

    model_config = ConfigDict(extra="forbid")

    slot: str
    path: str
    """Relative to the run's folder when inside the plan, absolute otherwise."""
    fingerprint: str


class Selection(BaseModel):
    """A particle set a run defines: its rule and the file that holds its members."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["particles", "tilt_series", "tomograms"]
    definition: dict[str, Any]
    snapshot: str
    """The file holding the membership, relative to the run's folder."""


class RunRecord(BaseModel):
    """`run.json`, field by field as docs/run_records.md lists them."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    id: str
    step: str
    step_version: int
    backend: str
    backend_version: int
    content_params: dict[str, Any]
    execution_params: dict[str, Any] = Field(default_factory=dict)
    inputs: list[RunInput] = Field(default_factory=list)
    selection: str | None = None
    lane: str | None = None
    fingerprint: str
    status: Literal["succeeded", "failed"]
    started_at: str
    ended_at: str
    host: str | None = None
    slurm_job_id: str | None = None
    log_path: str | None = None
    code: dict[str, str] = Field(default_factory=dict)
    outputs: list[Artifact] = Field(default_factory=list)
    """On success only; each path relative to the run's folder."""
    selections: list[Selection] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    """Numbers the run measured on its way: its times, the sizes of what it made. The
    file-first form of the design's `qc_metric` and `metric_series` rows."""
    error: str | None = None


def _canonical(value: Any) -> Any:
    """What JSON cannot hold, as it can: paths as text, numpy scalars and arrays as numbers."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} has no canonical form in a fingerprint")


def canonical_json(value: Any) -> str:
    """Sorted keys, no spaces, the shortest repr of every float: the same text for the
    same values on every machine."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=_canonical)


def fingerprint(step: str, step_version: int, backend: str, backend_version: int,
                content_params: Mapping[str, Any], inputs: Sequence[RunInput],
                selection: str | None = None) -> str:
    """sha256 of what declares a run's result; never its execution, paths or time."""

    payload = {
        "step": step, "step_version": step_version, "backend": backend,
        "backend_version": backend_version, "content_params": dict(content_params),
        "inputs": sorted(({"slot": item.slot, "fingerprint": item.fingerprint}
                          for item in inputs), key=lambda item: item["slot"]),
        "selection": selection,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def files_fingerprint(paths: Sequence[Path]) -> str:
    """sha256 over each file's name and bytes, in the order given."""

    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).name.encode() + b"\0")
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def code_record() -> dict[str, str]:
    """The package's version and its source's commit, with `dirty` when the source has
    uncommitted changes: recorded with every run, never fingerprinted."""

    try:
        package = version("cryoet-pipeline")
    except PackageNotFoundError:
        package = "unknown"
    record = {"package": package}
    source = Path(__file__).resolve().parent
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, timeout=10, check=False)
    if commit.returncode == 0 and commit.stdout.strip():
        record["commit"] = commit.stdout.strip()
        status = subprocess.run(["git", "-C", str(source), "status", "--porcelain",
                                 "--untracked-files=no"],
                                capture_output=True, text=True, timeout=10, check=False)
        if status.returncode == 0 and status.stdout.strip():
            record["dirty"] = "true"
    return record


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_record(folder: Path) -> RunRecord | None:
    path = Path(folder) / RECORD
    return RunRecord.model_validate_json(path.read_text()) if path.exists() else None


def completed(folder: Path) -> RunRecord | None:
    """The record of a run whose folder is whole, else None."""

    record = read_record(folder)
    return record if record is not None and record.status == "succeeded" else None


def clear_abandoned(folder: Path) -> list[Path]:
    """Move the temporary folders a killed attempt at `folder` left to `attempts/`.

    Called before a new attempt, under the run's lock, when no other attempt can be
    writing: whatever temporary folder is there belongs to a dead one.
    """

    folder = Path(folder)
    moved = []
    for leftover in sorted(folder.parent.glob(f"{folder.name}.tmp-*")):
        target = folder.parent / ATTEMPTS / leftover.name.split(".tmp-", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(leftover, target)
        moved.append(target)
    return moved


@dataclass
class Attempt:
    """What a run writes while it runs: files in `folder`, and what they are."""

    folder: Path
    outputs: list[Artifact] = field(default_factory=list)
    selections: list[Selection] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def add(self, kind: ArtifactKind, name: str, **fields: Any) -> Path:
        """Record the output written as `name` in this run's folder; returns its path."""

        self.outputs.append(Artifact(kind=kind, path=Path(name), **fields))
        return self.folder / name


def _write_record(folder: Path, record: RunRecord) -> None:
    temporary = folder / f".{RECORD}.tmp"
    temporary.write_text(record.model_dump_json(indent=1) + "\n")
    os.replace(temporary, folder / RECORD)


@contextmanager
def writing(
    folder: Path,
    *,
    step: str,
    step_version: int,
    backend: str,
    backend_version: int,
    content_params: Mapping[str, Any],
    inputs: Sequence[RunInput] = (),
    execution_params: Mapping[str, Any] | None = None,
    selection: str | None = None,
    lane: str | None = None,
    log_path: str | None = None,
) -> Iterator[Attempt]:
    """One run of a step into `folder`, written as docs/pipeline_engine.md orders it.

    The caller writes into `attempt.folder` and records each output with
    `attempt.add`. On success `run.json` is written last, into that folder, which is
    then renamed to `folder`; on any failure, Ctrl-C included, the attempt moves to
    `attempts/<run id>/` with a record saying it failed, and the error goes on.
    `folder` must not exist: a run is never written over.
    """

    folder = Path(folder)
    if folder.exists():
        raise FileExistsError(f"{folder} is there already; a run is never written over")
    run_id = str(uuid.uuid4())
    inputs = list(inputs)
    shared: dict[str, Any] = {
        "id": run_id, "step": step, "step_version": step_version, "backend": backend,
        "backend_version": backend_version, "content_params": dict(content_params),
        "execution_params": dict(execution_params or {}), "inputs": inputs,
        "selection": selection, "lane": lane,
        "fingerprint": fingerprint(step, step_version, backend, backend_version,
                                   content_params, inputs, selection),
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "log_path": log_path, "code": code_record(), "started_at": _now(),
    }
    temporary = folder.with_name(f"{folder.name}.tmp-{run_id}")
    temporary.mkdir(parents=True)
    attempt = Attempt(temporary)
    try:
        yield attempt
    except BaseException as error:
        failed = folder.parent / ATTEMPTS / run_id
        failed.parent.mkdir(parents=True, exist_ok=True)
        os.rename(temporary, failed)
        _write_record(failed, RunRecord(**shared, status="failed", ended_at=_now(),
                                        error=f"{type(error).__name__}: {error}"))
        raise
    _write_record(temporary, RunRecord(**shared, status="succeeded", ended_at=_now(),
                                       outputs=attempt.outputs, selections=attempt.selections,
                                       metrics=attempt.metrics))
    os.rename(temporary, folder)


__all__ = ["Attempt", "RunInput", "RunRecord", "Selection", "canonical_json", "clear_abandoned",
           "code_record", "completed", "files_fingerprint", "fingerprint", "read_record",
           "writing"]
