"""Several commands committing to one registry, the way jobs on one project do.

Every command loads `artifacts.json` when it starts and commits when it ends,
which for picking is hours later. These tests interleave commands at those two
points.
"""

from __future__ import annotations

import errno
import multiprocessing
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from cryoet_pipeline import artifacts as artifacts_module
from cryoet_pipeline.artifacts import (
    ArtifactRegistry,
    RegistryConflictError,
    RegistryLockError,
    _mount_lock_problem,
    registry_lock,
)
from cryoet_pipeline.models import Artifact, ArtifactKind


def _artifact(artifact_id: str, *parent_ids: str, run: str = "initial") -> Artifact:
    return Artifact(
        id=artifact_id,
        kind=ArtifactKind.TOMOGRAM,
        path=Path("/project") / f"{artifact_id.replace(':', '_')}.mrc",
        parent_ids=list(parent_ids),
        parameters={"run": run},
    )


def _registry_file(tmp_path: Path, *artifacts: Artifact) -> Path:
    path = tmp_path / "artifacts.json"
    ArtifactRegistry(artifacts=list(artifacts)).write(path)
    return path


def _on_disk(path: Path) -> dict[str, Any]:
    """Which run registered each entry, as the next command would read it."""

    return {
        artifact.id: artifact.parameters["run"]
        for artifact in ArtifactRegistry.load(path).artifacts
    }


@pytest.mark.parametrize("replace", [False, True], ids=["add", "overwrite"])
@pytest.mark.parametrize("first", ["A", "B"])
def test_two_interleaved_commands_keep_each_others_entries(
    tmp_path: Path, first: str, replace: bool
) -> None:
    """Load A, load B, A commits, B commits: neither loses the other's entry."""

    path = _registry_file(tmp_path, _artifact("stack"))
    command_a = ArtifactRegistry.load(path)
    command_b = ArtifactRegistry.load(path)
    command_a.add(_artifact("from_a", "stack", run="A"), replace=replace)
    command_b.add(_artifact("from_b", "stack", run="B"), replace=replace)

    order = [command_a, command_b] if first == "A" else [command_b, command_a]
    for command in order:
        command.commit(path)

    assert _on_disk(path) == {"stack": "initial", "from_a": "A", "from_b": "B"}
    # The later command now holds the merged registry, not its stale copy.
    assert order[-1].artifact_ids == {"stack", "from_a", "from_b"}


def test_a_late_stale_commit_keeps_what_other_jobs_registered_meanwhile(
    tmp_path: Path,
) -> None:
    """The loss of 2026-10-07, replayed.

    A cancelled duplicate job's fine-align for TS_0388 finished after scancel
    and wrote back a registry it had loaded 1.5 minutes earlier. Between the
    two, other jobs had registered TS_0487's bin2 final aligned stack and
    TS_0572's import, and the next steps found neither.
    """

    path = _registry_file(
        tmp_path, _artifact("TS_0388:tilt_stack"), _artifact("TS_0487:alignment:fine")
    )
    duplicate = ArtifactRegistry.load(path)

    ts_0487 = ArtifactRegistry.load(path)
    ts_0487.add(
        _artifact("TS_0487:aligned_tilt_stack:final", "TS_0487:alignment:fine", run="TS_0487 job")
    )
    ts_0487.commit(path)
    ts_0572 = ArtifactRegistry.load(path)
    ts_0572.extend(
        [
            _artifact("TS_0572:tilt_angles", run="TS_0572 job"),
            _artifact("TS_0572:tilt_stack", run="TS_0572 job"),
        ]
    )
    ts_0572.commit(path)

    duplicate.add(
        _artifact("TS_0388:alignment:fine", "TS_0388:tilt_stack", run="duplicate"), replace=True
    )
    duplicate.commit(path)

    assert _on_disk(path) == {
        "TS_0388:tilt_stack": "initial",
        "TS_0487:alignment:fine": "initial",
        "TS_0487:aligned_tilt_stack:final": "TS_0487 job",
        "TS_0572:tilt_angles": "TS_0572 job",
        "TS_0572:tilt_stack": "TS_0572 job",
        "TS_0388:alignment:fine": "duplicate",
    }


def test_a_command_started_before_the_registry_existed_merges_into_it(
    tmp_path: Path,
) -> None:
    """Two imports into a project that has no registry yet."""

    path = tmp_path / "artifacts.json"
    first_import = ArtifactRegistry.empty()
    second_import = ArtifactRegistry.empty()
    first_import.add(_artifact("TS_0201:tilt_stack", run="first"))
    second_import.add(_artifact("TS_0388:tilt_stack", run="second"))

    first_import.commit(path)
    second_import.commit(path)

    assert _on_disk(path) == {"TS_0201:tilt_stack": "first", "TS_0388:tilt_stack": "second"}


def test_overwrite_without_a_concurrent_writer_replaces_and_reorders_as_before(
    tmp_path: Path,
) -> None:
    """Replaying `replace=True` applies what `add` does in memory, moves included.

    The rebuilt tomogram gains a parent that sits after it in the file, so it
    has to move there, in the merged document as in the loaded copy. The
    other job's entry, about another series, is no reason to refuse.
    """

    path = _registry_file(
        tmp_path,
        _artifact("aligned"),
        _artifact("tomogram", "aligned"),
        _artifact("corrected", "aligned"),
    )
    rebuild = ArtifactRegistry.load(path)
    other_series = ArtifactRegistry.load(path)
    rebuild.add(_artifact("tomogram", "corrected", run="rebuilt"), replace=True)
    other_series.add(_artifact("other_series", run="other job"))
    other_series.commit(path)

    rebuild.commit(path)

    order = [artifact.id for artifact in ArtifactRegistry.load(path).artifacts]
    assert order == ["aligned", "corrected", "tomogram", "other_series"]
    assert _on_disk(path)["tomogram"] == "rebuilt"


@pytest.mark.parametrize("replace", [False, True], ids=["add", "overwrite"])
def test_registering_an_id_another_command_registered_meanwhile_is_refused(
    tmp_path: Path, replace: bool
) -> None:
    """Two jobs on one series: the second commit is refused even with --overwrite."""

    path = _registry_file(tmp_path, _artifact("stack"))
    first_job = ArtifactRegistry.load(path)
    second_job = ArtifactRegistry.load(path)
    second_job.add(_artifact("tomogram", "stack", run="second job"), replace=replace)
    second_job.commit(path)
    first_job.add(_artifact("tomogram", "stack", run="first job"), replace=replace)

    with pytest.raises(
        RegistryConflictError, match="tomogram: registered by another command"
    ):
        first_job.commit(path)
    assert _on_disk(path) == {"stack": "initial", "tomogram": "second job"}


def test_replacing_an_entry_another_command_replaced_meanwhile_is_refused(
    tmp_path: Path,
) -> None:
    """Last-writer-wins would drop one of the two versions without a word."""

    path = _registry_file(tmp_path, _artifact("alignment"))
    first_job = ArtifactRegistry.load(path)
    second_job = ArtifactRegistry.load(path)
    second_job.add(_artifact("alignment", run="second job"), replace=True)
    second_job.commit(path)
    first_job.add(_artifact("alignment", run="first job"), replace=True)

    with pytest.raises(
        RegistryConflictError,
        match="alignment: replaced by another command; this command replaces it",
    ):
        first_job.commit(path)
    assert _on_disk(path) == {"alignment": "second job"}


@pytest.mark.parametrize("change", ["replaced", "removed"])
def test_registering_a_child_of_a_parent_that_changed_meanwhile_is_refused(
    tmp_path: Path, change: str
) -> None:
    """The picks came from the tomogram as it was; the registry must not say otherwise."""

    path = _registry_file(tmp_path, _artifact("tomogram"))
    picking = ArtifactRegistry.load(path)
    rerun = ArtifactRegistry.load(path)
    if change == "replaced":
        rerun.add(_artifact("tomogram", run="rerun"), replace=True)
    else:
        rerun.remove("tomogram")
    rerun.commit(path)
    picking.add(_artifact("picks", "tomogram", run="picking"))

    with pytest.raises(
        RegistryConflictError, match=f"tomogram: {change} by another command; a parent of picks"
    ):
        picking.commit(path)
    assert "picks" not in _on_disk(path)


def test_a_removal_is_replayed_and_refused_for_an_entry_with_children(
    tmp_path: Path,
) -> None:
    path = _registry_file(tmp_path, _artifact("stale"), _artifact("kept"))
    cleanup = ArtifactRegistry.load(path)
    cleanup.remove("stale")
    cleanup.commit(path)
    assert _on_disk(path) == {"kept": "initial"}

    with_child = _registry_file(
        tmp_path / "with_child", _artifact("tomogram"), _artifact("picks", "tomogram")
    )
    with pytest.raises(ValueError, match=r"tomogram is still a parent of \['picks'\]"):
        ArtifactRegistry.load(with_child).remove("tomogram")


def test_removing_an_entry_that_gained_a_child_meanwhile_is_refused(tmp_path: Path) -> None:
    path = _registry_file(tmp_path, _artifact("tomogram"))
    cleanup = ArtifactRegistry.load(path)
    picking = ArtifactRegistry.load(path)
    cleanup.remove("tomogram")
    picking.add(_artifact("picks", "tomogram"))
    picking.commit(path)

    with pytest.raises(RegistryConflictError, match="still a parent of"):
        cleanup.commit(path)
    assert set(_on_disk(path)) == {"tomogram", "picks"}


def test_writing_back_a_loaded_copy_with_changes_is_refused(tmp_path: Path) -> None:
    """`write` is the old lost update itself, so a command cannot reach it by habit."""

    path = _registry_file(tmp_path, _artifact("stack"))
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))
    other = ArtifactRegistry.load(path)
    other.add(_artifact("other_series"))
    other.commit(path)

    with pytest.raises(RuntimeError, match="use commit"):
        command.write(path)
    assert set(_on_disk(path)) == {"stack", "other_series"}


def test_the_new_document_is_complete_before_it_takes_the_registry_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Written beside the registry, then renamed over it; never written in place."""

    path = _registry_file(tmp_path, _artifact("stack"))
    renames: list[tuple[bool, set[str]]] = []
    real_replace = os.replace

    def watched_replace(source: Any, target: Any) -> None:
        source, target = Path(source), Path(target)
        renames.append(
            (source.parent == target.parent, ArtifactRegistry.load(source).artifact_ids)
        )
        real_replace(source, target)

    monkeypatch.setattr(artifacts_module.os, "replace", watched_replace)
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))
    command.commit(path)

    assert renames == [(True, {"stack", "tomogram"})]


def test_a_failed_replace_leaves_the_previous_document_and_no_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry_file(tmp_path, _artifact("stack"))
    before = path.read_bytes()
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))

    def failing_replace(source: Any, target: Any) -> None:
        raise OSError(errno.EIO, "simulated I/O error")

    monkeypatch.setattr(artifacts_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated I/O error"):
        command.commit(path)

    assert path.read_bytes() == before
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "artifacts.json",
        "artifacts.json.lock",
    ]


def test_a_commit_keeps_the_registry_file_mode(tmp_path: Path) -> None:
    """A replaced file would otherwise take the umask's mode, not the one set on it."""

    path = _registry_file(tmp_path, _artifact("stack"))
    path.chmod(0o640)
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))
    command.commit(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


_HOLD_LOCK = """
import fcntl, os, sys
descriptor = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
fcntl.lockf(descriptor, fcntl.LOCK_EX)
print("held", flush=True)
sys.stdin.read()
"""


def test_a_commit_waits_for_the_lock_and_gives_up_loudly(tmp_path: Path) -> None:
    path = _registry_file(tmp_path, _artifact("stack"))
    before = path.read_bytes()
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))

    # Another process: a POSIX lock never keeps out the process that holds it.
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(tmp_path / "artifacts.json.lock")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(RegistryLockError, match="held by another command"):
            command.commit(path, lock_timeout=0.2)
    finally:
        holder.communicate(timeout=30)
    assert path.read_bytes() == before

    command.commit(path)
    assert set(_on_disk(path)) == {"stack", "tomogram"}


@pytest.mark.parametrize("code", [errno.ENOSYS, errno.ENOLCK], ids=["ENOSYS", "ENOLCK"])
def test_a_lock_that_fails_is_refused_not_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """ENOSYS is Lustre mounted without flock; proceeding unlocked is the bug itself."""

    path = _registry_file(tmp_path, _artifact("stack"))
    before = path.read_bytes()
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))

    def failing_lockf(descriptor: int, operation: int, *args: int) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(artifacts_module.fcntl, "lockf", failing_lockf)
    with pytest.raises(RegistryLockError, match=f"failed with {errno.errorcode[code]}"):
        command.commit(path)
    assert path.read_bytes() == before


def test_a_commit_on_a_mount_with_node_local_locks_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry_file(tmp_path, _artifact("stack"))
    before = path.read_bytes()
    command = ArtifactRegistry.load(path)
    command.add(_artifact("tomogram", "stack"))

    monkeypatch.setattr(
        artifacts_module,
        "lock_scope_problem",
        lambda directory: "/lustre is lustre mounted with localflock",
    )
    with pytest.raises(RegistryLockError, match="mounted with localflock"):
        command.commit(path)
    assert path.read_bytes() == before


def test_a_commit_whose_registry_was_written_under_its_lock_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MDC, 2026-10-29: flock on GPFS let a job on another node commit at the same
    time, and the later rename erased the earlier job's entry. Whatever the lock
    does, the rename must not erase what it finds written in the meantime."""

    path = _registry_file(tmp_path, _artifact("stack"))
    command = ArtifactRegistry.load(path)
    command.add(_artifact("from_here", "stack", run="here"))

    merge = ArtifactRegistry._changed_since_loaded

    def another_node_commits_meanwhile(
        self: ArtifactRegistry, current: ArtifactRegistry
    ) -> dict[str, str]:
        other = ArtifactRegistry.load(path)
        other.artifacts.append(_artifact("from_another_node", "stack", run="there"))
        # Straight to the file, as a commit the lock did not keep out would.
        artifacts_module._replace_atomically(path, other._document())
        return merge(self, current)

    monkeypatch.setattr(
        ArtifactRegistry, "_changed_since_loaded", another_node_commits_meanwhile
    )
    with pytest.raises(RegistryLockError, match="written by another command"):
        command.commit(path)

    assert _on_disk(path) == {"stack": "initial", "from_another_node": "there"}
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "artifacts.json",
        "artifacts.json.lock",
    ]


def test_threads_of_one_process_take_turns(tmp_path: Path) -> None:
    """A POSIX lock does not keep out the threads of the process that holds it."""

    path = _registry_file(tmp_path, _artifact("stack"))
    count = 8
    barrier = threading.Barrier(count)
    errors: list[BaseException] = []

    def load_wait_add_commit(index: int) -> None:
        try:
            registry = ArtifactRegistry.load(path)
            barrier.wait(timeout=60)
            registry.add(_artifact(f"series_{index}", "stack", run=f"thread {index}"))
            registry.commit(path)
        except BaseException as error:  # noqa: BLE001 - reported below
            errors.append(error)

    threads = [threading.Thread(target=load_wait_add_commit, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert _on_disk(path) == {
        "stack": "initial",
        **{f"series_{index}": f"thread {index}" for index in range(count)},
    }


def test_the_lock_is_not_taken_again_by_the_thread_that_holds_it(tmp_path: Path) -> None:
    """Closing the inner lock file would release the outer POSIX lock."""

    path = _registry_file(tmp_path, _artifact("stack"))
    with registry_lock(path):
        with pytest.raises(RuntimeError, match="not re-entrant"):
            with registry_lock(path):
                pass
    # Released properly afterwards.
    with registry_lock(path, timeout=1.0):
        pass


_MOUNTINFO = (
    "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw,errors=remount-ro\n"
    "640 22 0:52 / /project rw,relatime shared:300 - {fstype} server:/export rw,{options}\n"
)


@pytest.mark.parametrize(
    ("fstype", "options", "named"),
    [
        # /ictstr01 on the cluster, as findmnt printed it on 2026-10-09.
        ("lustre", "seclabel,flock,lazystatfs,encrypt", None),
        ("lustre", "lazystatfs", None),
        ("lustre", "localflock,lazystatfs", "localflock"),
        # An old Lustre showed both after a localflock mount; refused, not guessed at.
        ("lustre", "flock,localflock", "localflock"),
        ("nfs4", "vers=4.2,local_lock=none", None),
        # Only flock stays local; POSIX locks still go to the server.
        ("nfs4", "vers=4.2,local_lock=flock", None),
        ("nfs4", "vers=4.2,local_lock=posix", "local_lock=posix"),
        ("nfs", "vers=3,nolock,local_lock=all", "local_lock=all, nolock"),
        # /fast at MDC, as findmnt printed it on 2026-10-29. Its flock stays on one
        # node, its POSIX locks do not (scripts/lock_probe.py); nothing here says so.
        ("gpfs", "relatime", None),
        ("xfs", "attr2,inode64", None),
    ],
)
def test_a_mount_whose_locks_stay_on_this_node_is_named(
    fstype: str, options: str, named: str | None
) -> None:
    mountinfo = _MOUNTINFO.format(fstype=fstype, options=options)

    problem = _mount_lock_problem(mountinfo, os.makedev(0, 52))

    if named is None:
        assert problem is None
    else:
        assert problem is not None
        assert f"/project is {fstype} mounted with {named}" in problem


def test_a_device_missing_from_the_mount_table_is_not_judged() -> None:
    mountinfo = _MOUNTINFO.format(fstype="lustre", options="localflock")

    assert _mount_lock_problem(mountinfo, os.makedev(0, 53)) is None
    assert _mount_lock_problem(mountinfo, os.makedev(8, 1)) is None


def _load_wait_add_commit(path_text: str, index: int, barrier: Any) -> None:
    path = Path(path_text)
    registry = ArtifactRegistry.load(path)
    # Every process holds a stale copy before any of them commits.
    barrier.wait(timeout=60)
    registry.add(_artifact(f"series_{index}", "stack", run=f"process {index}"))
    registry.commit(path)


def test_processes_committing_at_once_lose_nothing(tmp_path: Path) -> None:
    """Real processes and the real lock, all loading before anyone commits."""

    path = _registry_file(tmp_path, _artifact("stack"))
    count = 8
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(count)
    processes = [
        context.Process(target=_load_wait_add_commit, args=(str(path), index, barrier))
        for index in range(count)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=120)

    assert [process.exitcode for process in processes] == [0] * count
    assert _on_disk(path) == {
        "stack": "initial",
        **{f"series_{index}": f"process {index}" for index in range(count)},
    }
    assert sorted(entry.name for entry in tmp_path.iterdir()) == [
        "artifacts.json",
        "artifacts.json.lock",
    ]
