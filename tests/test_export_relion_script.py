from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_relion.py"


def _script():
    spec = importlib.util.spec_from_file_location("export_relion_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "tomogram_binning, raw_pixel, expected",
    [
        (8, 0.675, 2),  # super-resolution sums to the camera's 1.35 A
        (4, 1.68, 1),  # already at the camera's pixel
        (8, 0.4, 4),  # 1.6 A; 3 would be 1.2 A but does not divide 8
        (2, 0.3, 2),  # never coarser than the tomogram
        (6, 0.4, 3),
    ],
)
def test_default_stack_binning_reaches_an_angstrom_dividing_the_tomogram(
    tomogram_binning: int, raw_pixel: float, expected: int
) -> None:
    assert _script().default_stack_binning(tomogram_binning, raw_pixel) == expected


def test_class_spec_reads_run_and_class() -> None:
    script = _script()
    assert script._class_spec("outputs/p/mra/run2:3") == (Path("outputs/p/mra/run2"), 3)
    with pytest.raises(argparse.ArgumentTypeError):
        script._class_spec("outputs/p/mra/run2")


def _record(**changes):
    record = {
        "recipe": "export-relion",
        "config": "/p/mra/run2",
        "series": "class 1 after iteration 5",
        "project": "/p",
        "particles": 300,
        "duplicates_left_out": 2,
        "stacks": {"TS_01": {"binning": 2, "views": 41, "relion_bin": 2, "notes": []}},
        "out": "/p/relion/run2-class1",
        "relion_check": False,
        "env": {},
    }
    return {**record, **changes}


def _dry_run(record, *, user="anna", minutes_ago=5, event="dry-run"):
    from datetime import UTC, datetime, timedelta

    at = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
    return {"event": event, "user": user, "at": at, "plan": record, "problems": []}


def _refusal(record, rows, *, assistant=True, reason="Anna asked for class 1"):
    from datetime import UTC, datetime

    who = {"user": "anna", "assistant": assistant}
    return _script().refusal(record, who, reason, rows, datetime.now(UTC))


def test_a_person_exports_without_a_dry_run_on_record() -> None:
    assert _refusal(_record(), [], assistant=False, reason=None) is None


def test_the_assistant_exports_only_the_dry_run_it_showed() -> None:
    record = _record()
    assert _refusal(record, [_dry_run(record)]) is None
    assert "needs --reason" in _refusal(record, [_dry_run(record)], reason=" ")
    assert "no dry run" in _refusal(record, [])
    assert "minutes old" in _refusal(record, [_dry_run(record, minutes_ago=45)])
    assert "anna2's" in _refusal(record, [_dry_run(record, user="anna2")])
    changed = _record(particles=299)
    assert "differs" in _refusal(changed, [_dry_run(record)])
    later = _record(series="class 1 after iteration 6")  # an iteration finished since
    assert "no dry run" in _refusal(later, [_dry_run(record)])
    used = [_dry_run(record, minutes_ago=6), _dry_run(record, minutes_ago=1, event="submit")]
    assert "submitted" in _refusal(record, used)
