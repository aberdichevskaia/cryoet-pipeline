"""scripts/grade_cases.py: its checks are well formed, its readers of a run's files
work, and its ticks agree with the hand-checked drafts of two batches of 4 November
2026 (grading-draft.md), except where listed below with the reason."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import grade_cases  # noqa: E402

BATCHES = ROOT / "outputs" / "eval-runs"
CASE_IDS = sorted(p.name for p in grade_cases.CASES.iterdir() if (p / "case.toml").is_file())

# Where the script and a draft disagree: (batch, run, box) -> why. A new
# disagreement fails the test, and so does one listed here that no longer occurs.
KNOWN_DISAGREEMENTS: dict[tuple[str, str, str], str] = {}

_DRAFT_RUN = re.compile(r"^### Run (?P<n>\d+)(?P<held> \(held out\))? — ")
_DRAFT_BOX = re.compile(r"^- \[(?P<mark>[ x])\] (?P<rest>.+)$")
_DRAFT_CASE = re.compile(r"^## (?P<id>E\d+)\. ")


def case_boxes(case_id: str) -> list[str]:
    data = tomllib.loads((grade_cases.CASES / case_id / "case.toml").read_text())
    return list(data["pass_if"]) + [f"did not: {line}" for line in data["must_not"]]


def draft_ticks(path: Path) -> dict[tuple[str, str], bool]:
    """(run folder, box) -> ticked, as the draft has it."""
    ticks: dict[tuple[str, str], bool] = {}
    case = run = None
    for line in path.read_text().splitlines():
        if match := _DRAFT_CASE.match(line):
            case, run, boxes = match["id"], None, case_boxes(match["id"])
        elif match := _DRAFT_RUN.match(line):
            run = f"{case}-{'held-out-' if match['held'] else ''}{match['n']}"
        elif (match := _DRAFT_BOX.match(line)) and case and run:
            box = max((b for b in boxes if match["rest"].startswith(b)), key=len, default=None)
            if box:
                ticks[(run, box)] = match["mark"] == "x"
    return ticks


def disagreements(batch: str) -> dict[tuple[str, str, str], str]:
    folder = BATCHES / batch
    sheet, results = grade_cases.grade_batch(folder)
    drafted = draft_ticks(folder / "grading-draft.md")
    found = {}
    for index, (_, run, box) in sheet.boxes.items():
        result = results[index]
        if result.status == grade_cases.PERSON:
            continue
        if (run, box) not in drafted:
            found[(batch, run, box)] = f"not in the draft; script: {result.status}"
        elif drafted[(run, box)] != (result.status == grade_cases.TICK):
            found[(batch, run, box)] = (
                f"draft {'ticks' if drafted[(run, box)] else 'leaves open'}; "
                f"script {result.status}: {result.evidence}"
            )
    return found


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_every_check_names_a_box_of_its_case_and_parses(case_id):
    checks = grade_cases.case_checks(case_id)  # raises on an unknown box, key or pattern
    assert set(checks) <= set(case_boxes(case_id))


def test_a_check_for_a_box_the_case_does_not_have_is_an_error(tmp_path):
    (tmp_path / "E99").mkdir()
    (tmp_path / "E99" / "case.toml").write_text(
        'pass_if = ["it says x"]\nmust_not = ["submit"]\n'
        '[[check]]\nbox = "did not: submit!"\nsubmitted = false\n'
    )
    with pytest.raises(ValueError, match="does not have"):
        grade_cases.case_checks("E99", tmp_path)


@pytest.mark.parametrize(
    "command, written",
    [
        ("python3 scripts/show_project.py x.toml 2>&1 | head -40", []),
        ("cat a.toml > /dev/null; echo hi >&2", []),
        ("sed -i 's/a/b/' configs/lanes/ryr1-bin8.toml", ["configs/lanes/ryr1-bin8.toml"]),
        (
            "cp configs/lanes/ribo-mdc.toml outputs/ribo-mdc/copy.toml",
            ["outputs/ribo-mdc/copy.toml"],
        ),
        ("echo '{}' >> outputs/ribo-mdc/journal.jsonl", ["outputs/ribo-mdc/journal.jsonl"]),
        ("rm -f outputs/ryr1/registry.json", ["outputs/ryr1/registry.json"]),
    ],
)
def test_shell_writes(command, written):
    assert grade_cases.shell_writes(command) == written


def _run(tmp_path, calls: str, slurm: str = "", answer: str = "Done.\n") -> grade_cases.Run:
    folder = tmp_path / "E99-1"
    (folder / "sandbox").mkdir(parents=True)
    (folder / "actions.md").write_text(f"# E99 run 1: what it did\n\n## 1. The request\n\n{calls}")
    (folder / "slurm_calls.log").write_text(slurm)
    (folder / "answer.md").write_text(answer)
    (folder / "record.json").write_text(json.dumps({"permission_denials": []}))
    return grade_cases.read_run(folder, "E99")


def test_a_dry_run_is_not_a_submission_and_a_real_one_is(tmp_path):
    dry = _run(
        tmp_path / "a", "1. Bash: `python scripts/submit_lanes.py x.toml --dry-run 2>&1 | tail`\n"
    )
    assert grade_cases.submissions(dry) == []
    real = _run(
        tmp_path / "b",
        "1. Bash: `grep -n sbatch scripts/x.py; python scripts/submit_lanes.py x.toml`\n",
    )
    assert [s.split(":")[0] for s in grade_cases.submissions(real)] == ["actions.md call 1"]
    stub = _run(tmp_path / "c", "", slurm="sbatch --parsable lane.sbatch\n")
    assert grade_cases.submissions(stub)


def test_calls_span_lines_and_follow_ups_start_a_part(tmp_path):
    run = _run(
        tmp_path,
        "1. Bash: `cat <<EOF > a.txt\nx\nEOF` → done\n\n"
        "## 2. Then the person: “Go on.”\n\n2. Edit: `outputs/a.toml` → ok\n",
    )
    assert [(c.number, c.part, c.tool) for c in run.calls] == [(1, 1, "Bash"), (2, 2, "Edit")]
    assert run.calls[0].command == "cat <<EOF > a.txt\nx\nEOF"


@pytest.mark.parametrize("batch", ["20261104-112736", "20261104-120842"])
def test_the_script_agrees_with_the_hand_checked_drafts(batch):
    if not (BATCHES / batch / "grading-draft.md").is_file():
        pytest.skip(f"{batch} is not in this checkout's outputs/eval-runs")
    found = disagreements(batch)
    known = {key: why for key, why in KNOWN_DISAGREEMENTS.items() if key[0] == batch}
    new = {key: found[key] for key in found.keys() - known.keys()}
    gone = sorted(known.keys() - found.keys())
    assert not new and not gone, (
        "new disagreements:\n"
        + "\n".join(f"  {k[1]} | {k[2]} | {v}" for k, v in sorted(new.items()))
        + "\nlisted but no longer found:\n"
        + "\n".join(f"  {k[1]} | {k[2]}" for k in gone)
    )
