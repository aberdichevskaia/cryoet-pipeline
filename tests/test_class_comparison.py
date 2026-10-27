"""Comparing two classifications of the same particles, and two Dynamo projects by tag."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_classification_report import (
    SECOND,
    TAGS,
    make_project,
    refined,
    write_manifest,
    write_table,
)

from cryoet_pipeline.class_comparison import (
    ShuffledNull,
    adjusted_rand_index,
    contingency,
    group_overlap,
    paired_share,
    shuffled_null,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dynamo_project(tmp_path: Path) -> Path:
    return make_project(tmp_path)


def test_relabelled_copies_agree_fully_and_beat_every_shuffle() -> None:
    rng = np.random.default_rng(1)
    first = rng.integers(1, 4, size=600)
    second = np.choose(first - 1, [7, 5, 9])  # the same partition under other names
    table, left, right = contingency(first, second)
    assert left == [1, 2, 3] and right == [5, 7, 9]
    assert adjusted_rand_index(table) == pytest.approx(1.0)
    assert paired_share(table) == pytest.approx(1.0)
    null = shuffled_null(first, second, rounds=200)
    assert np.max(null.rand_index) < 0.05
    assert ShuffledNull.p_value(null.rand_index, 1.0) == pytest.approx(1 / 201)


def test_independent_labels_sit_at_chance() -> None:
    rng = np.random.default_rng(2)
    first, second = rng.integers(1, 4, size=3000), rng.integers(1, 4, size=3000)
    table, _, _ = contingency(first, second)
    assert abs(adjusted_rand_index(table)) < 0.01
    null = shuffled_null(first, second, rounds=200)
    assert ShuffledNull.p_value(null.rand_index, adjusted_rand_index(table)) > 0.05
    # The best pairing lifts the share above a third even at chance; the null is lifted alike.
    assert paired_share(table) > 1 / 3
    assert ShuffledNull.p_value(null.paired_share, paired_share(table)) > 0.05


def test_group_overlap_against_the_hypergeometric() -> None:
    overlap = group_overlap([True] * 30 + [False] * 70, [True] * 25 + [False] * 75)
    assert (overlap.both, overlap.first_only, overlap.second_only, overlap.neither) == (
        25, 5, 0, 70)
    assert overlap.expected_both == pytest.approx(7.5)
    assert overlap.ratio == pytest.approx(25 / 7.5)
    assert overlap.p_more < 1e-15
    rng = np.random.default_rng(3)
    independent = group_overlap(rng.random(2000) < 0.2, rng.random(2000) < 0.3)
    assert 0.01 < min(independent.p_more, independent.p_less)


def _second_project(project: Path, classes: dict[int, int]) -> Path:
    """A copy of the fixture's export whose second iteration ends in `classes`."""

    copy = project.parent.parent / "export_b"
    shutil.copytree(project.parent, copy)
    averages = copy / project.name / "results" / "ite_0002" / "averages"
    for channel in (1, 2):
        write_table(averages / f"refined_table_ref_{channel:03d}_ite_0002.tbl",
                    refined(classes, channel))
    return copy / project.name


def _run(*arguments: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "scripts/compare_dynamo_classes.py"),
                           *map(str, arguments)], capture_output=True, text=True, check=False)


def test_two_dynamo_projects_are_compared_by_tag(dynamo_project: Path, tmp_path: Path) -> None:
    write_manifest(dynamo_project, decoys=(12,))
    other = {tag: 1 if tag <= 4 else 2 for tag in TAGS}
    second = _second_project(dynamo_project, other)
    out = tmp_path / "comparison.json"
    result = _run(dynamo_project, second, "--group-first", "2", "--group-second", "2",
                  "--shuffles", "100", "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    numbers = json.loads(out.read_text())
    # Tag 12 is a decoy in both manifests and is left out.
    assert numbers["particles"] == 11 and numbers["decoys_left_out"] == 1
    expected = [[sum(1 for t in range(1, 12) if SECOND[t] == a and other[t] == b)
                 for b in (1, 2)] for a in (1, 2)]
    assert numbers["table"] == expected
    assert numbers["groups"]["both"] == sum(
        1 for t in range(1, 12) if SECOND[t] == 2 and other[t] == 2)
    assert "same series, stack and index" in numbers["identity"]


def test_one_class_is_not_tested_against_chance(dynamo_project: Path, tmp_path: Path) -> None:
    # A one-channel run (SEEDS=map, or a port mirror of one) has nothing to shuffle:
    # its index is 0/0, and NaN once read here as beating every shuffle.
    write_manifest(dynamo_project)
    second = _second_project(dynamo_project, {tag: 1 for tag in TAGS})
    out = tmp_path / "comparison.json"
    result = _run(dynamo_project, second, "--shuffles", "50", "--out", out)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "one class in the second: nothing to compare with chance" in result.stdout
    assert "shuffles reached it" not in result.stdout and "nan" not in result.stdout
    numbers = json.loads(out.read_text())
    assert numbers["adjusted_rand_index"] is None and numbers["paired_share_p"] is None
    assert numbers["table"] == [[sum(1 for t in TAGS if SECOND[t] == k)] for k in (1, 2)]
    both = _run(second, second, "--shuffles", "50")
    assert both.returncode == 0, both.stdout + both.stderr
    assert "one class in the first run and the second: nothing to compare" in both.stdout
    assert "nan" not in both.stdout


def test_tags_naming_other_particles_are_refused(dynamo_project: Path) -> None:
    write_manifest(dynamo_project)
    second = _second_project(dynamo_project, SECOND)
    manifest = second.parent / "export_manifest.json"
    data = json.loads(manifest.read_text())
    data["particles"][2]["run"] = "TS_OTHER"
    manifest.write_text(json.dumps(data))
    result = _run(dynamo_project, second)
    assert result.returncode != 0
    assert "name different particles" in result.stdout + result.stderr
