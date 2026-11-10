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
