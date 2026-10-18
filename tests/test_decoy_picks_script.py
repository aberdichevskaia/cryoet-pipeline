"""scripts/decoy_picks.py: random positions that carry the real picks' orientations."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot",
           "rlnAngleTilt", "rlnAnglePsi", "rlnLCCmax")


def _project(tmp_path: Path, template_box: int | None) -> Path:
    generator = np.random.default_rng(0)
    rows = [
        (generator.uniform(40, 260), generator.uniform(40, 260), generator.uniform(25, 35),
         float(index), 90.0, -float(index), 0.1 + 0.01 * index)
        for index in range(20)
    ]
    star = tmp_path / "TS_A_picks.star"
    lines = ["data_particles", "", "loop_", *(f"_{name} #{i}" for i, name in enumerate(COLUMNS, 1))]
    lines += [" ".join(f"{value:.6f}" for value in row) for row in rows]
    star.write_text("\n".join(lines) + "\n")
    registry = ArtifactRegistry.empty()
    tomogram = Artifact(
        id="TS_A:tomogram:bin8", variant="bin8", kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "TS_A.zarr", shape=(60, 300, 300), pixel_spacing_angstrom=9.784,
        parameters={"tilt_series_id": "TS_A"},
    )
    registry.add(tomogram)
    registry.add(Artifact(
        id="TS_A:picks:full:pytom-match-pick:bin8", variant="bin8", kind=ArtifactKind.PICKS,
        path=star, parent_ids=[tomogram.id],
        parameters={"tilt_series_id": "TS_A", "input_tomogram_id": tomogram.id,
                    "tomogram_branch": "full"}
        | ({"pick_template_box_px": template_box} if template_box else {}),
    ))
    path = tmp_path / "artifacts.json"
    registry.write(path)
    return path


def _run(registry: Path, tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/decoy_picks.py"), "--registry", str(registry),
         "--series", "TS_A", "--real-variant", "bin8", "--variant", "bin8-decoy",
         "--out", str(tmp_path / "decoys"), "--check-variance", "0", "--seed", "1", *extra],
        capture_output=True, text=True, check=False,
    )


def test_fewer_decoys_carry_real_orientations_and_the_template_box(tmp_path):
    registry_path = _project(tmp_path, template_box=60)
    result = _run(registry_path, tmp_path, "--count", "5")
    assert result.returncode == 0, result.stdout + result.stderr

    registry = ArtifactRegistry.load(registry_path)
    decoy = registry.get("TS_A:picks:full:decoy:bin8-decoy")
    assert decoy.variant == "bin8-decoy"
    assert decoy.parameters["pick_template_box_px"] == 60
    assert decoy.parameters["particle_count"] == 5
    assert decoy.parameters["real_particle_count"] == 20

    rows = [line.split() for line in decoy.path.read_text().splitlines()
            if line and line[0].isdigit()]
    assert len(rows) == 5
    # Orientation and score of a real pick each; positions inside the half-box margin.
    for row in rows:
        x, y, z, rot, tilt, psi, score = (float(value) for value in row)
        assert psi == -rot and tilt == 90.0 and score == round(0.1 + 0.01 * rot, 6)
        assert 24 <= x <= 276 and 24 <= y <= 276 and 24 <= z <= 36


def test_picks_that_do_not_say_their_template_box_are_refused(tmp_path):
    registry_path = _project(tmp_path, template_box=None)
    refused = _run(registry_path, tmp_path, "--count", "5")
    assert refused.returncode != 0
    assert "--template-box-px" in refused.stdout + refused.stderr
    given = _run(registry_path, tmp_path, "--count", "5", "--template-box-px", "60")
    assert given.returncode == 0, given.stdout + given.stderr
