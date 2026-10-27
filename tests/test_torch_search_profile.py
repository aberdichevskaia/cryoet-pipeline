"""scripts/torch_search_profile.py: the kernel timed step by step, its poses unchanged."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from test_classification_report import write_em
from test_dynamo_mra_project import CARD, _project

from cryoet_pipeline.backends import dynamo_search as core
from cryoet_pipeline.reports.dynamo_project import read_em

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "torch_search_profile.py"


def _started_iteration(root: Path) -> Path:
    """`_project` with iteration 1 as Dynamo starts it: the search mask, and each
    channel's prepared reference and card, in the iteration's own files."""

    project, _ = _project(root)
    here = project / "results" / "ite_0001"
    (here / "currently_surviving_references_ite_0001.txt").write_text("1 2\n")
    mask = read_em(project / "settings" / "mask.em")
    write_em(here / "starting_values" / "mask_transformed_ite_0001.em",
             (mask > 0.5).astype(np.float32))
    settings = core.SearchSettings(cone_aperture=10, cone_sampling=5, inplane_range=10,
                                   inplane_sampling=5, refine=1, high=2, low=6)
    for channel in (1, 2):
        template = read_em(project / "settings" / f"template_initial_ref_{channel:03d}.em")
        write_em(here / "starting_values"
                 / f"starting_reference_transformed_ref_{channel:03d}_ite_0001.em",
                 core.prepare_reference(template, mask, settings).astype(np.float32))
        (project / "cards" / "ite_0001" / f"card_iteref_ref_{channel:03d}_ite_0001.card"
         ).write_text("".join(f"{key}={value};\n" for key, value in CARD.items()))
    return project


def test_the_steps_and_the_processes_give_the_kernels_own_poses(tmp_path: Path) -> None:
    project = _started_iteration(tmp_path)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(project), "--iterations", "1", "--particles", "4",
         "--device", "cpu", "--processes", "2"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "mra_t, iteration 1: channels [1, 2], 4 particles each (8 searches)" in out
    assert "1. kernel, one process:" in out
    for step in ("grid (host)", "filters (once per tilt range)", "reference and mask (once a call)",
                 "rotate", "filter candidates", "correlate", "peak", "other"):
        assert f"       {step} " in out
    assert "poses and CCs equal to 1 bit for bit: yes" in out
    assert "4. processes sharing cpu: 2:" in out and "same bit for bit" in out
