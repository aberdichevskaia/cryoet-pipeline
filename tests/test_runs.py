"""cryoet_pipeline.runs: a run's record, its fingerprint, and how it is written."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cryoet_pipeline.models import ArtifactKind
from cryoet_pipeline.runs import (
    RunInput,
    Selection,
    canonical_json,
    clear_abandoned,
    completed,
    files_fingerprint,
    fingerprint,
    read_record,
    writing,
)

DECLARED = {"step": "mra-iteration", "step_version": 1, "backend": "dynamo-port",
            "backend_version": 1}


def _print(**changes):
    values = {**DECLARED, "content_params": {"low": 13, "cone": 27.0},
              "inputs": [RunInput(slot="previous", path="setup", fingerprint="a"),
                         RunInput(slot="export", path="/x", fingerprint="b")],
              "selection": None, **changes}
    return fingerprint(**values)


def test_the_fingerprint_is_what_declares_the_result_and_nothing_else() -> None:
    reference = _print()
    # Where the inputs were, and the order they are named in, are not what they are.
    assert _print(inputs=[RunInput(slot="export", path="/elsewhere", fingerprint="b"),
                          RunInput(slot="previous", path="ite_0001", fingerprint="a")]) \
        == reference
    assert _print(content_params={"cone": 27.0, "low": 13}) == reference
    for changed in ({"content_params": {"low": 14, "cone": 27.0}}, {"step_version": 2},
                    {"backend_version": 2}, {"selection": "class 2"},
                    {"inputs": [RunInput(slot="previous", path="setup", fingerprint="c"),
                                RunInput(slot="export", path="/x", fingerprint="b")]}):
        assert _print(**changed) != reference, changed


def test_canonical_json_is_the_same_text_for_the_same_values() -> None:
    assert canonical_json({"b": np.float32(0.5), "a": np.arange(2), "p": Path("x/y")}) \
        == '{"a":[0,1],"b":0.5,"p":"x/y"}'
    assert canonical_json({"x": 3 * 4.9}) == '{"x":14.700000000000001}'
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_files_fingerprint_reads_names_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / "b.txt").write_text("1")
    first = files_fingerprint([tmp_path / "a.txt"])
    assert files_fingerprint([tmp_path / "a.txt"]) == first
    assert files_fingerprint([tmp_path / "b.txt"]) != first
    (tmp_path / "a.txt").write_text("2")
    assert files_fingerprint([tmp_path / "a.txt"]) != first


def test_a_run_is_renamed_into_place_and_its_record_written_last(tmp_path: Path) -> None:
    node = tmp_path / "plan" / "ite_0001"
    with writing(node, **DECLARED, content_params={"low": 13},
                 inputs=[RunInput(slot="previous", path="setup", fingerprint="a")],
                 execution_params={"device": "cuda"}) as attempt:
        assert attempt.folder.name.startswith("ite_0001.tmp-") and not node.exists()
        attempt.add(ArtifactKind.PARTICLE_TABLE, "particles.parquet").write_text("rows")
        attempt.selections.append(Selection(name="class 1", kind="particles",
                                            definition={"column": "channel", "equals": 1},
                                            snapshot="particles.parquet"))
    record = completed(node)
    assert record is not None and record.status == "succeeded"
    assert (node / "particles.parquet").read_text() == "rows"
    assert [str(output.path) for output in record.outputs] == ["particles.parquet"]
    assert record.execution_params == {"device": "cuda"} and record.code["package"]
    assert record.fingerprint == fingerprint(**DECLARED, content_params={"low": 13},
                                             inputs=record.inputs)
    assert record.selections[0].name == "class 1"
    assert sorted(path.name for path in node.parent.iterdir()) == ["ite_0001"]
    with pytest.raises(FileExistsError, match="never written over"):
        with writing(node, **DECLARED, content_params={"low": 13}):
            pass


def test_a_failed_run_keeps_its_record_and_files_under_attempts(tmp_path: Path) -> None:
    node = tmp_path / "plan" / "ite_0002"
    with pytest.raises(RuntimeError, match="out of memory"):
        with writing(node, **DECLARED, content_params={"low": 14}) as attempt:
            (attempt.folder / "partial.txt").write_text("half")
            raise RuntimeError("out of memory")
    assert not node.exists() and completed(node) is None
    (failed,) = (tmp_path / "plan" / "attempts").iterdir()
    record = read_record(failed)
    assert record is not None and record.status == "failed" and record.outputs == []
    assert record.error == "RuntimeError: out of memory"
    assert (failed / "partial.txt").read_text() == "half"


def test_what_a_killed_attempt_left_is_moved_aside_before_the_next(tmp_path: Path) -> None:
    node = tmp_path / "plan" / "ite_0003"
    leftover = node.with_name("ite_0003.tmp-1234")
    leftover.mkdir(parents=True)
    (leftover / "half.mrc").write_text("x")
    assert clear_abandoned(node) == [tmp_path / "plan" / "attempts" / "1234"]
    assert not leftover.exists() and (tmp_path / "plan" / "attempts" / "1234" / "half.mrc").exists()
    assert clear_abandoned(node) == []
