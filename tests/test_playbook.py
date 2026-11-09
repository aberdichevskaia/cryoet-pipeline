"""docs/agent/playbook/: the playbook's files, one per step, checked against each
other, the lane keys, and the evaluation's case lists and held-out markers.

The playbook goes into every eval sandbox. A held-out run takes a case's own
entries out (scripts/run_cases.py), so a case's answer may sit only there.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cases  # noqa: E402
import run_series  # noqa: E402

AGENT = ROOT / "docs" / "agent"
FILES = sorted((AGENT / "playbook").glob("*.md"))
INDEX = AGENT / "playbook.md"
TABLE = tomllib.loads((AGENT / "cases" / "cases.toml").read_text())
HELD_OUT = tomllib.loads((AGENT / "cases" / "held_out.toml").read_text())
ENTRY = re.compile(r"^### ([A-Z]-\d+)\.", re.M)
# An entry's "Error text" quotes messages as the code prints them, a misspelt key
# among them, so lane keys are checked everywhere else.
ERROR_TEXT = re.compile(r"\*\*Error text\b.*?(?=\n\n\*\*|\n\n#|\Z)", re.S)


def _entries() -> dict[str, str]:
    """Each entry's ID and the file that holds it, refusing an ID held twice."""
    found: dict[str, str] = {}
    for path in FILES:
        for entry in ENTRY.findall(path.read_text()):
            assert entry not in found, f"{entry} is in {path.name} and in {found[entry]}"
            found[entry] = path.name
    return found


def test_the_index_lists_every_file_and_holds_no_entry_of_its_own():
    names = {path.name for path in FILES}
    assert {"data.md", "motion.md", "alignment.md", "ctf.md", "tomogram.md", "membranes.md",
            "template.md", "picking.md", "averaging.md", "cluster.md"} <= names
    index = INDEX.read_text()
    assert not ENTRY.findall(index)
    assert set(re.findall(r"\(playbook/([a-z]+\.md)\)", index)) == names


def test_every_entry_id_is_unique_and_every_reference_names_an_entry():
    entries = _entries()
    for path in [*FILES, INDEX]:
        for reference in sorted(set(re.findall(r"\b([A-Z]-\d+)\b", path.read_text()))):
            assert reference in entries, f"{path.name} names {reference}, which no file holds"


def test_every_entry_states_its_kind_its_evidence_and_its_source():
    for path in FILES:
        for block in re.split(r"(?=^### )", path.read_text(), flags=re.M)[1:]:
            title = block.splitlines()[0]
            for field in ("**Kind:**", "**Evidence:**", "**Source:**"):
                assert field in block, (path.name, title, field)


def test_every_lane_key_named_outside_a_quoted_message_is_one():
    for path in FILES:
        text = ERROR_TEXT.sub("", path.read_text())
        for section, key in re.findall(r"\[([a-z_]+)\] ([a-z_0-9]+)", text):
            if section in run_series.LANE_KEYS:
                assert key in run_series.LANE_KEYS[section], (path.name, f"[{section}] {key}")


def test_every_link_leads_to_a_file():
    for path in [*FILES, INDEX]:
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", path.read_text()):
            if not target.startswith(("http://", "https://")):
                assert (path.parent / target).exists(), (path.name, target)


def test_no_file_names_an_evaluation_case():
    # The case IDs belong to the evaluation's own files, which no sandbox holds.
    for path in [*FILES, INDEX]:
        assert not re.findall(r"\bE\d{2}\b", path.read_text()), path.name


def test_every_entry_a_case_names_is_in_the_playbook_and_its_two_lists_agree():
    entries = set(_entries())
    for name, case in TABLE.items():
        assert set(case.get("playbook", [])) <= entries, name
    for case in run_cases.built_cases().values():
        assert case.playbook == TABLE[case.id].get("playbook", []), case.id


@pytest.mark.parametrize(
    "case_id", sorted(name for name, spec in HELD_OUT.items() if spec.get("markers")))
def test_a_cases_answer_markers_are_only_in_its_own_entries(case_id):
    own = TABLE[case_id].get("playbook", [])
    for path in FILES:
        rest = run_cases.without_entries(path.read_text(), own)
        for marker in HELD_OUT[case_id]["markers"]:
            assert not re.search(marker, rest), (case_id, path.name, marker)
