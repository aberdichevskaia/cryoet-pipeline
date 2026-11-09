"""docs/agent/lane_questions.md names every key the pipeline reads, and only those."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402

DOC = ROOT / "docs" / "agent" / "lane_questions.md"
SOURCES = {"data", "card", "person", "site", "rule"}
ROW = re.compile(
    r"^\| `\[(?P<section>[a-z_.]+)\] (?P<keys>[a-z0-9_, ]+)` \| (?P<sources>[a-z, ]+) \| .+ \|$"
)
# A key named in the text: `[section] key`, followed by anything.
MENTION = re.compile(r"`\[(?P<section>[a-z_.]+)\] (?P<key>[a-z0-9_]+)")
GOAL_ROW = re.compile(r"^\| [a-z ]+ \| `(?P<step>[a-z-]+)`")


def section(title: str) -> str:
    text = DOC.read_text()
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def rows() -> list[re.Match[str]]:
    found = []
    for line in section("Every key").splitlines():
        if line.startswith("| `["):
            row = ROW.match(line)
            assert row, f"a row this test cannot read: {line[:90]}"
            found.append(row)
    return found


def pipeline_keys() -> set[tuple[str, str]]:
    keys = {(name, key) for name, keys in run_series.LANE_KEYS.items() for key in keys}
    return keys | {("run.stage", key) for key in run_series.STAGE_KEYS}


def test_the_table_names_every_key_the_pipeline_reads_and_no_other():
    named = [(row["section"], key.strip()) for row in rows() for key in row["keys"].split(",")]
    assert len(named) == len(set(named)), "a key has two rows"
    missing = sorted(pipeline_keys() - set(named))
    unread = sorted(set(named) - pipeline_keys())
    assert not missing, f"keys the pipeline reads with no row: {missing}"
    assert not unread, f"rows for keys the pipeline does not read: {unread}"


def test_every_row_says_where_its_value_comes_from():
    for row in rows():
        sources = {source.strip() for source in row["sources"].split(",")}
        assert sources and sources <= SOURCES, row.group(0)[:90]


def test_every_key_the_text_names_is_one_the_pipeline_reads():
    named = {(m["section"], m["key"]) for m in MENTION.finditer(DOC.read_text())}
    assert named, "the pattern found no key at all"
    assert not sorted(named - pipeline_keys())


def test_the_goals_end_at_steps_a_lane_has():
    listed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_series.py"), "--project", "x",
         "--config", str(ROOT / "configs" / "lanes" / "example.toml"), "--series", "X",
         "--list-steps"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout.split()
    goals = [GOAL_ROW.match(line)["step"] for line in section("What the goal sets").splitlines()
             if GOAL_ROW.match(line)]
    assert goals == ["reconstruct-tomogram", "segment-tomogram", "visualize-picks",
                     "extract-particles", run_series.MRA_STEP]
    # The MRA follows the chain each series runs, as the stage of all of them at once.
    assert set(goals) - {run_series.MRA_STEP} <= set(listed)
    assert run_series.MRA_STEP not in listed
