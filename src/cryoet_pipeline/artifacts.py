from __future__ import annotations

import errno
import fcntl
import json
import os
import random
import shutil
import socket
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from cryoet_pipeline.models import Artifact, ArtifactKind, StorageRole

LOCK_TIMEOUT_SECONDS = 600.0
"""How long a commit waits for the registry lock before giving up.

A commit holds the lock for one read, merge and write of the document, well under
a second, so ten minutes of waiting means the holder is stuck, not busy.
"""


class RegistryLockError(RuntimeError):
    """The registry lock was not obtained, or would not exclude other nodes."""


class RegistryConflictError(RuntimeError):
    """Another command changed an entry that this command's changes rely on."""


@dataclass(frozen=True)
class _Change:
    """One change a command made, replayed onto the registry on disk at commit."""

    action: Literal["add", "replace", "remove"]
    artifact_id: str
    artifact: Artifact | None = None


class ArtifactRegistry(BaseModel):
    """Ordered registry of artifacts produced or exported by the pipeline.

    One JSON document per project, which several commands may be using at once,
    from different jobs. A command therefore never writes back the copy it loaded:
    `add` and `remove` change that copy and also record the change, and `commit`
    replays the recorded changes onto the document as it is on disk at that moment.
    Writing back the loaded copy is what lost TS_0487's bin2 stack and TS_0572's
    import on 2026-10-07: a cancelled job's fine-align wrote back a copy it had
    loaded 1.5 minutes earlier.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    artifacts: list[Artifact] = Field(default_factory=list)

    _changes: list[_Change] = PrivateAttr(default_factory=list)
    _loaded: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    """Each entry as of the last load, commit or write: what `commit` compares the file to."""
    _source: Path | None = PrivateAttr(default=None)

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

    def model_post_init(self, context: Any, /) -> None:
        self._loaded = self._snapshot()

    @classmethod
    def empty(cls) -> ArtifactRegistry:
        return cls()

    @classmethod
    def load(cls, path: Path) -> ArtifactRegistry:
        # No lock: commits replace the file atomically, so a reader gets one
        # whole document or the next one, never part of either.
        return cls._parse(path.read_text(), path)

    @classmethod
    def _parse(cls, document: str | bytes, path: Path) -> ArtifactRegistry:
        registry = cls.model_validate_json(document)
        registry._source = path.resolve()
        return registry

    def commit(self, path: Path, *, lock_timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        """Apply this registry's recorded changes to the document at `path`.

        Under the registry lock the document is read again, the changes are
        replayed onto it with the same rules `add` and `remove` apply in memory,
        and the result replaces the file atomically. Entries other commands
        registered in the meantime are kept.

        Refused, with nothing written, when another command has since changed an
        entry these changes rely on: one they add, replace or remove, or a parent
        of one they register. Replacing such an entry anyway would silently drop
        the other command's version, and registering a child of a parent that
        changed would record lineage that is no longer true. Without concurrent
        writers this cannot happen, so `replace=True` (`--overwrite`) keeps its
        meaning: the entry is replaced whatever it held when the command started.

        Refused as well when the document is no longer the one the merge started
        from by the time the result is renamed over it: another command wrote it
        while this one held the lock, so the lock does not keep out other nodes on
        this filesystem, and renaming would erase that command's entries. That is
        how three of five jobs lost their entries on MDC's GPFS on 2026-10-29,
        when the lock was still flock. Two commits that reach this check within
        the same few milliseconds both pass it, so it shows that a lock fails; it
        does not stand in for one.

        Afterwards this registry holds the merged document, with nothing pending.
        """

        with registry_lock(path, timeout=lock_timeout):
            # The document the merge starts from, to be found unchanged before the
            # result replaces it.
            based_on = path.read_bytes() if path.exists() else None
            current = (
                ArtifactRegistry._parse(based_on, path)
                if based_on is not None
                else ArtifactRegistry.empty()
            )
            conflicts = self._changed_since_loaded(current)
            if conflicts:
                raise RegistryConflictError(_conflict_message(path, conflicts))
            try:
                for change in self._changes:
                    if change.action == "remove":
                        current._apply_remove(change.artifact_id)
                    else:
                        assert change.artifact is not None
                        current._apply_add(change.artifact, replace=change.action == "replace")
            except (KeyError, ValueError) as exc:
                # The checks above cover every entry the changes name; what is
                # left is, for instance, a child another command registered under
                # an entry this one removes.
                raise RegistryConflictError(
                    f"{path}: this command's changes no longer apply to the registry "
                    f"as it is now ({exc}); nothing was registered"
                ) from exc
            _replace_atomically(
                path, current._document(), before_rename=lambda: _unchanged(path, based_on)
            )
        self._settle(current.artifacts, path)

    def write(self, path: Path, *, lock_timeout: float = LOCK_TIMEOUT_SECONDS) -> None:
        """Replace the whole document at `path` with this registry.

        For starting a registry (`init`) and for test fixtures. A command that
        loaded the registry must `commit` instead, because writing its copy back
        would drop whatever other commands registered since; that is refused.
        """

        if self._changes and self._source is not None and self._source == path.resolve():
            raise RuntimeError(
                f"{path}: this registry was loaded from that file and has uncommitted "
                "changes; write() would replace entries other commands registered since "
                "it was loaded, so use commit()"
            )
        with registry_lock(path, timeout=lock_timeout):
            _replace_atomically(path, self._document())
        self._settle(self.artifacts, path)

    def add(self, artifact: Artifact, *, replace: bool = False) -> None:
        self._apply_add(artifact, replace=replace)
        self._changes.append(_Change("replace" if replace else "add", artifact.id, artifact))

    def extend(self, artifacts: Iterable[Artifact], *, replace: bool = False) -> None:
        for artifact in artifacts:
            self.add(artifact, replace=replace)

    def remove(self, artifact_id: str) -> None:
        """Drop an entry. Refused while another entry still names it as a parent."""

        self._apply_remove(artifact_id)
        self._changes.append(_Change("remove", artifact_id))

    def get(self, artifact_id: str) -> Artifact:
        for artifact in self.artifacts:
            if artifact.id == artifact_id:
                return artifact
        raise KeyError(f"unknown artifact id: {artifact_id}")

    def by_kind(self, kind: ArtifactKind) -> list[Artifact]:
        return [artifact for artifact in self.artifacts if artifact.kind == kind]

    def by_storage_role(self, role: StorageRole) -> list[Artifact]:
        return [artifact for artifact in self.artifacts if artifact.storage_role == role]

    def children_of(self, artifact_id: str) -> list[Artifact]:
        return [
            artifact for artifact in self.artifacts if artifact_id in artifact.parent_ids
        ]

    def descends_from(self, artifact_id: str, ancestor_id: str) -> bool:
        """Whether one artifact was derived from another, at any depth.

        Direct parentage is too strict a test for provenance: inserting a stage
        between two others, as CTF correction sits between the final aligned
        stack and reconstruction, pushes the earlier artifact from parent to
        grandparent without changing what it contributed.
        """

        seen: set[str] = set()
        pending = [artifact_id]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            index = self._find_index(current)
            if index is None:
                continue
            for parent_id in self.artifacts[index].parent_ids:
                if parent_id == ancestor_id:
                    return True
                pending.append(parent_id)
        return False

    def ancestors_of_kind(self, artifact_id: str, kind: ArtifactKind) -> list[Artifact]:
        """Every registered ancestor of one kind, each once, nearest generation first."""

        found: list[Artifact] = []
        seen = {artifact_id}
        index = self._find_index(artifact_id)
        pending = list(self.artifacts[index].parent_ids) if index is not None else []
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            index = self._find_index(current)
            if index is None:
                continue
            parent = self.artifacts[index]
            if parent.kind == kind:
                found.append(parent)
            pending.extend(parent.parent_ids)
        return found

    @property
    def total_size_bytes(self) -> int:
        return sum(artifact.size_bytes or 0 for artifact in self.artifacts)

    @property
    def size_bytes_by_storage_role(self) -> dict[StorageRole, int]:
        sizes = {role: 0 for role in StorageRole}
        for artifact in self.artifacts:
            sizes[artifact.storage_role] += artifact.size_bytes or 0
        return sizes

    @property
    def artifact_ids(self) -> set[str]:
        return {artifact.id for artifact in self.artifacts}

    def _apply_add(self, artifact: Artifact, *, replace: bool) -> None:
        existing_index = self._find_index(artifact.id)
        if existing_index is not None and not replace:
            raise ValueError(f"duplicate artifact id: {artifact.id}")

        if existing_index is not None:
            # Parents must exist, but not necessarily before the position the
            # old entry held. Re-running a stage after a new stage was inserted
            # upstream deepens its lineage — reconstructing from a stack that
            # CTF correction produced is exactly that — and the entry has to
            # move rather than be rejected for sitting where it used to belong.
            self._validate_parent_ids(artifact, self.artifact_ids - {artifact.id})
            self.artifacts[existing_index] = artifact
            self._restore_lineage_order()
            return

        existing_ids = self.artifact_ids
        self._validate_parent_ids(artifact, existing_ids)
        self.artifacts.append(artifact)

    def _apply_remove(self, artifact_id: str) -> None:
        index = self._find_index(artifact_id)
        if index is None:
            raise KeyError(f"unknown artifact id: {artifact_id}")
        children = [artifact.id for artifact in self.children_of(artifact_id)]
        if children:
            raise ValueError(f"{artifact_id} is still a parent of {children}; remove them first")
        del self.artifacts[index]

    def _changed_since_loaded(self, current: ArtifactRegistry) -> dict[str, str]:
        """Entries the pending changes rely on that differ on disk from when loaded.

        Other entries may differ freely: they are what other commands registered,
        and keeping them is the point of merging. Compared as dumped JSON, the
        form both sides were read from.
        """

        touched = {change.artifact_id for change in self._changes}
        relied_on: dict[str, str] = {}
        for change in self._changes:
            relied_on.setdefault(change.artifact_id, f"this command {change.action}s it")
            if change.artifact is None:
                continue
            for parent_id in change.artifact.parent_ids:
                if parent_id not in touched:
                    relied_on.setdefault(parent_id, f"a parent of {change.artifact_id}")

        on_disk = current._snapshot()
        conflicts: dict[str, str] = {}
        for artifact_id, reason in relied_on.items():
            then = self._loaded.get(artifact_id)
            now = on_disk.get(artifact_id)
            if then == now:
                continue
            if then is None:
                what = "registered"
            elif now is None:
                what = "removed"
            else:
                what = "replaced"
            conflicts[artifact_id] = f"{what} by another command; {reason}"
        return conflicts

    def _settle(self, artifacts: list[Artifact], path: Path) -> None:
        """Take `artifacts` as what is on disk at `path`, with nothing pending."""

        self.artifacts = artifacts
        self._changes = []
        self._loaded = self._snapshot()
        self._source = path.resolve()

    def _snapshot(self) -> dict[str, dict[str, Any]]:
        return {artifact.id: artifact.model_dump(mode="json") for artifact in self.artifacts}

    def _document(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"

    def _restore_lineage_order(self) -> None:
        """Reorder so parents precede children again, moving as little as possible.

        Repeatedly takes the earliest entry whose parents are all placed, so a
        registry that already satisfies the invariant is left exactly as it was
        and only the entries whose lineage changed actually move.
        """

        known_ids = self.artifact_ids
        remaining = list(self.artifacts)
        ordered: list[Artifact] = []
        placed: set[str] = set()
        while remaining:
            for index, artifact in enumerate(remaining):
                if all(
                    parent_id in placed
                    for parent_id in artifact.parent_ids
                    if parent_id in known_ids
                ):
                    ordered.append(artifact)
                    placed.add(artifact.id)
                    del remaining[index]
                    break
            else:
                unplaceable = ", ".join(sorted(artifact.id for artifact in remaining))
                raise ValueError(f"artifact lineage contains a cycle among: {unplaceable}")
        self.artifacts = ordered

    def _find_index(self, artifact_id: str) -> int | None:
        for index, artifact in enumerate(self.artifacts):
            if artifact.id == artifact_id:
                return index
        return None

    @staticmethod
    def _validate_parent_ids(artifact: Artifact, existing_ids: set[str]) -> None:
        missing_parent_ids = [
            parent_id for parent_id in artifact.parent_ids if parent_id not in existing_ids
        ]
        if missing_parent_ids:
            raise ValueError(
                f"{artifact.id}: parent artifacts must be registered first; "
                f"missing {missing_parent_ids}"
            )


_in_process = threading.Lock()
"""Taken before the POSIX lock, which another thread of this process would get too."""

_held_here = threading.local()


@contextmanager
def registry_lock(path: Path, *, timeout: float = LOCK_TIMEOUT_SECONDS) -> Iterator[None]:
    """Hold the exclusive lock of the registry at `path`: a POSIX lock on `<path>.lock`.

    A separate file, because every commit replaces the registry itself, and a
    lock held on a replaced file excludes nobody. The lock file is never deleted
    for the same reason.

    A POSIX record lock (fcntl), not flock: on MDC's GPFS, flock locks only within
    one node, and five jobs on five nodes lost three of their entries there on
    2026-10-29, while a POSIX lock kept the other node out (scripts/lock_probe.py).
    A POSIX lock belongs to the process rather than to the open file: another
    thread of the process gets it too, and closing any descriptor of the lock file
    releases it. So the threads of a process take turns on a lock of their own
    first, and asking for this lock while holding it is refused.

    Refused rather than skipped when the lock cannot be had or would not reach
    other nodes: proceeding unlocked would reopen the lost update this lock
    exists to prevent, and nothing would show it until entries went missing.
    """

    if getattr(_held_here, "lock_path", None) is not None:
        raise RuntimeError(
            f"registry_lock({path}) asked for while this thread holds "
            f"{_held_here.lock_path}: closing the inner lock file would release the "
            "outer lock, so it is not re-entrant"
        )
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    problem = lock_scope_problem(lock_path.parent)
    if problem is not None:
        raise RegistryLockError(f"{lock_path}: {problem}")
    if not _in_process.acquire(timeout=timeout):
        raise RegistryLockError(
            f"another thread of this process has held a registry lock for over {timeout:g} s; "
            "a commit holds it for well under a second. The registry was left as it was."
        )
    _held_here.lock_path = lock_path
    try:
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            _acquire(descriptor, lock_path, timeout)
            yield
        finally:
            # This process's only descriptor of the lock file: closing it releases
            # the lock.
            os.close(descriptor)
    finally:
        _held_here.lock_path = None
        _in_process.release()


_HELD_ELSEWHERE = frozenset({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK})
"""What a non-blocking POSIX lock fails with while another process holds it."""


def _acquire(descriptor: int, lock_path: Path, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    delay = 0.01
    while True:
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError as exc:
            if exc.errno not in _HELD_ELSEWHERE:
                name = errno.errorcode.get(exc.errno or 0, str(exc.errno))
                raise RegistryLockError(
                    f"a POSIX lock on {lock_path} failed with {name} ({exc.strerror}). "
                    "Without the lock, commands committing at the same time can lose each "
                    "other's entries, so the registry was left as it was. On Lustre, ENOSYS "
                    "means the client is mounted without -o flock."
                ) from exc
            if time.monotonic() >= deadline:
                raise RegistryLockError(
                    f"{lock_path} has been held by another command for over {timeout:g} s; "
                    "a commit holds it for well under a second, so look for a hung cryoet "
                    "process. The registry was left as it was."
                ) from None
            time.sleep(delay * random.uniform(0.5, 1.5))
            delay = min(delay * 2, 0.5)


def lock_scope_problem(directory: Path) -> str | None:
    """Why a POSIX lock in `directory` would lock only this node, or None.

    A mount can grant every lock locally while saying nothing: Lustre mounted with
    `localflock`, which covers fcntl locks as well as flock, and NFS mounted with
    `nolock` or `local_lock=posix|all`. Jobs on two nodes would then both hold the
    lock. (Lustre with `noflock` makes the call itself fail with ENOSYS, which
    `_acquire` reports.) Read from /proc/self/mountinfo, so Linux only; elsewhere
    there is nothing to read.

    Other filesystems are not judged from their mount options. GPFS mounts at MDC
    as plain `rw,relatime` and keeps POSIX locks across nodes but flock within one,
    which only trying it on two nodes showed (scripts/lock_probe.py). For a
    filesystem that misbehaves unseen, a commit that finds the registry replaced
    under its lock refuses (see `ArtifactRegistry.commit`).
    """

    try:
        mountinfo = Path("/proc/self/mountinfo").read_text()
    except OSError:
        return None
    return _mount_lock_problem(mountinfo, os.stat(directory).st_dev)


def _mount_lock_problem(mountinfo: str, device: int) -> str | None:
    """The mountinfo line of `device`, judged for node-local POSIX locks.

    Lines look like `36 35 0:52 / /lustre rw,relatime shared:1 - lustre mgs@o2ib:/fs
    rw,flock,lazystatfs`: the per-mount options in field 6, then optional fields,
    a lone `-`, the filesystem type, the source and the filesystem's own options.
    """

    wanted = f"{os.major(device)}:{os.minor(device)}"
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 9 or fields[2] != wanted:
            continue
        try:
            separator = fields.index("-", 6)
        except ValueError:
            continue
        mount_point = fields[4]
        fstype = fields[separator + 1] if len(fields) > separator + 1 else ""
        options = set(fields[5].split(","))
        if len(fields) > separator + 3:
            options |= set(fields[separator + 3].split(","))

        if fstype == "lustre":
            local = sorted(options & {"localflock"})
        elif fstype in {"nfs", "nfs4"}:
            local = sorted(options & {"nolock", "local_lock=posix", "local_lock=all"})
        else:
            local = []
        if local:
            return (
                f"{mount_point} is {fstype} mounted with {', '.join(local)}, so a POSIX lock "
                f"there holds only on this node ({socket.gethostname()}) and a job on another "
                "node could commit to the same registry at the same time. The registry needs "
                "locks across nodes (Lustre: -o flock)."
            )
        return None
    return None


def _unchanged(path: Path, based_on: bytes | None) -> None:
    """Refuse, before the rename, if `path` is not the document a merge started from."""

    now = path.read_bytes() if path.exists() else None
    if now != based_on:
        raise RegistryLockError(
            f"{path} was written by another command while this one held the registry "
            "lock, so on this filesystem the lock does not keep out commands on other "
            "nodes. Nothing was registered: renaming this command's result over it "
            "would have erased the other command's entries. Run the step again; "
            "scripts/lock_probe.py shows which locks reach other nodes here."
        )


def _replace_atomically(
    path: Path, text: str, *, before_rename: Callable[[], None] | None = None
) -> None:
    """Write `text` to `path` so that readers see the old document or the new one.

    The temporary file sits beside the target because `os.replace` is atomic
    only within one filesystem, and the target's own directory is the one place
    certain to be on it. `before_rename` runs once the new document is written
    and synced, as near the rename as it can be; if it raises, nothing replaces
    `path`.
    """

    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            # Data before name: otherwise a crash just after the rename can
            # leave the registry's name on an empty file.
            os.fsync(handle.fileno())
        if path.exists():
            # The new file was created with the umask's mode; keep the old one's.
            shutil.copymode(path, temporary)
        if before_rename is not None:
            before_rename()
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _conflict_message(path: Path, conflicts: dict[str, str]) -> str:
    lines = [
        f"{path} changed under this command since it was loaded; nothing was registered:",
        *(f"  {artifact_id}: {reason}" for artifact_id, reason in conflicts.items()),
        "Another job on the same tilt-series? This command's output files are written. "
        "Once the other job has finished or is gone, re-run this step with --overwrite.",
    ]
    return "\n".join(lines)
