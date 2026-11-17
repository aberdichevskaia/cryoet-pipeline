"""Picks and particle stacks to RELION 5 without an MRA run (docs/entry_and_exit.md, Exits).

The class export's conversion was checked against RELION's own reconstructions; a
particle of picks or of a stack enters it at the pose an MRA run of the stack would
start from. So these tests hold the particle set to that start: a stack's members
are the rows `registry_particles` gives a run, and picks' members are their
extracted stack's. Where RELION then puts each particle does not depend on the box.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import euler_matrix
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, Provenance
from cryoet_pipeline.particle_sources import ANGLES, SHIFTS, registry_particles
from cryoet_pipeline.relion_export import (
    RelionParticle,
    particle_centre,
    read_particle_set,
    write_particles,
)
from cryoet_pipeline.tomogram_import import tomogram_manifest

SHAPE = (40, 72, 80)
VOXEL = 10.0
TILTS = [float(angle) for angle in np.arange(-54.0, 60.1, 3.0)]
TEMPLATE_BOX = 16


@pytest.fixture
def project(tmp_path: Path) -> dict[str, Any]:
    """A tomogram made here (derived) with pytom-like picks at known centres."""

    rng = np.random.default_rng(3)
    root = tmp_path / "project"
    volume = root / "tomograms" / "TS_01.zarr"
    zarr.open(volume, mode="w", shape=SHAPE, dtype=np.float32)[:] = rng.normal(
        size=SHAPE).astype(np.float32)
    angles_file = root / "tomograms" / "TS_01.tlt"
    angles_file.write_text("".join(f"{angle:.2f}\n" for angle in TILTS))
    # Far enough from the faces for a box of 24 around every pick.
    centres = np.column_stack([rng.uniform(13, SHAPE[2] - 13, 12),
                               rng.uniform(13, SHAPE[1] - 13, 12),
                               rng.uniform(13, SHAPE[0] - 13, 12)])
    poses = rng.uniform([-180, 0, -180], [180, 180, 180], size=(12, 3))
    # pytom's pick lies half a voxel above the centre for an even template box.
    picks_star = root / "picks" / "TS_01_picks.star"
    picks_star.parent.mkdir(parents=True)
    labels = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot",
              "rlnAngleTilt", "rlnAnglePsi", "rlnLCCmax"]
    rows = np.hstack([centres + 0.5, poses, rng.uniform(0.2, 0.5, (12, 1))])
    picks_star.write_text("data_\n\nloop_\n" + "".join(f"_{label} #{index}\n" for index, label
                                                       in enumerate(labels, start=1))
                          + "".join(" ".join(f"{value:.6f}" for value in row) + "\n"
                                    for row in rows))
    tomogram = Artifact(id="TS_01:tomogram:fine", kind=ArtifactKind.TOMOGRAM, path=volume,
                        axis_order=AxisOrder.ZYX, pixel_spacing_angstrom=VOXEL,
                        shape=SHAPE, parameters={"tilt_series_id": "TS_01",
                                                 "tilt_angles_path": str(angles_file)})
    picks = Artifact(id="TS_01:picks:pytom", kind=ArtifactKind.PICKS, path=picks_star,
                     parent_ids=[tomogram.id],
                     parameters={"tilt_series_id": "TS_01", "input_tomogram_id": tomogram.id,
                                 "pick_template_box_px": TEMPLATE_BOX})
    registry = ArtifactRegistry.empty()
    registry.extend([tomogram, picks])
    registry.commit(root / "artifacts.json")
    manifest = root / "manifests" / "TS_01.json"
    manifest.parent.mkdir()
    manifest.write_text(tomogram_manifest("TS_01", TILTS, "test").model_dump_json(indent=2))
    return {"root": root, "centres": centres, "poses": poses, "manifest": manifest}


def _extract(project: dict[str, Any], box: int) -> Artifact:
    root = project["root"]
    outcome = CliRunner().invoke(cli.app, [
        "extract-particles", "--manifest", str(project["manifest"]), "--registry",
        str(root / "artifacts.json"), "--out", str(root), "--box-size-px", str(box),
    ])
    assert outcome.exit_code == 0, outcome.output
    (stack,) = ArtifactRegistry.load(root / "artifacts.json").by_kind(ArtifactKind.PARTICLE_STACK)
    return stack


def _poses(members: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (np.array([member.pick_xyz for member in members]),
            np.array([member.angles_deg for member in members]),
            np.array([member.shifts_px for member in members]))


def test_a_stack_leaves_from_where_an_mra_run_of_it_starts(project: dict[str, Any]) -> None:
    stack = _extract(project, 24)

    particles = read_particle_set(project["root"], [stack.id])
    run = registry_particles(project["root"], None, ["TS_01"], log=lambda _: None)

    picks, angles, shifts = _poses(particles.members)
    np.testing.assert_allclose(angles, run.table[:, ANGLES], atol=1e-9)
    np.testing.assert_allclose(shifts, run.table[:, SHIFTS], atol=1e-9)
    # Dynamo's column 24 holds the pick plus 0.5.
    np.testing.assert_allclose(picks, run.table[:, 23:26] - 0.5, atol=1e-9)
    assert particles.box == 24 and particles.angles == "given"
    assert particles.tomograms == {"TS_01": "TS_01:tomogram:fine"}


def test_picks_leave_as_their_extracted_stack_does(project: dict[str, Any]) -> None:
    stack = _extract(project, 24)

    from_stack = read_particle_set(project["root"], [stack.id])
    from_picks = read_particle_set(project["root"], ["TS_01:picks:pytom"], box=24)

    stack_picks, stack_angles, stack_shifts = _poses(from_stack.members)
    picks, angles, shifts = _poses(from_picks.members)
    # The stack's STAR keeps a pick to three decimals and its shift to six; the pick
    # places the particle only through its rounding, so the places agree to the shifts'.
    np.testing.assert_allclose(stack_picks, picks, atol=5e-4)
    np.testing.assert_allclose(stack_angles, angles, atol=1e-5)
    np.testing.assert_allclose(stack_shifts, shifts, atol=1e-5)
    for one, two in zip(from_stack.members, from_picks.members, strict=True):
        np.testing.assert_allclose(particle_centre(one, 24), particle_centre(two, 24), atol=1e-5)


@pytest.mark.parametrize("box", [16, 24, 48])
def test_where_relion_puts_a_particle_does_not_depend_on_the_box(
    project: dict[str, Any], box: int
) -> None:
    """The reference's voxel N/2, half a voxel from its geometric centre along each of
    its axes, lands at the particle's centre plus that half voxel turned by the pose."""

    particles = read_particle_set(project["root"], ["TS_01:picks:pytom"], box=box)

    for member, centre in zip(particles.members, project["centres"], strict=True):
        turned = euler_matrix(np.array(member.angles_deg)).T @ np.full(3, 0.5)
        np.testing.assert_allclose(particle_centre(member, box), centre + turned, atol=1e-5)


def test_picks_without_poses_leave_as_positions_only(
    tmp_path: Path, project: dict[str, Any]
) -> None:
    root = project["root"]
    registry = ArtifactRegistry.load(root / "artifacts.json")
    picks = registry.get("TS_01:picks:pytom")
    registry.add(picks.model_copy(update={"parameters": {**picks.parameters, "angles": "none"}}),
                 replace=True)
    registry.commit(root / "artifacts.json")

    particles = read_particle_set(root, [picks.id], box=24)

    assert particles.angles == "none"
    assert any("positions only" in note for note in particles.notes)
    for member, centre in zip(particles.members, project["centres"], strict=True):
        np.testing.assert_allclose(member.angles_deg, 0.0, atol=1e-9)
        np.testing.assert_allclose(particle_centre(member, 24), centre + 0.5, atol=1e-5)
    star = write_particles(tmp_path / "particles.star",
                           [RelionParticle("TS_01", 1, (0.0, 1.0, 2.0), (10.0, 20.0, 30.0))],
                           angles=False)
    assert "rlnAngle" not in star.read_text() and "rlnCenteredCoordinateZAngst" in star.read_text()


def _with(root: Path, artifact: Artifact) -> None:
    registry = ArtifactRegistry.load(root / "artifacts.json")
    registry.add(artifact, replace=True)
    registry.commit(root / "artifacts.json")


@pytest.mark.parametrize(
    "case, match",
    [
        ("no box", "give --box"),
        ("twice", "given twice"),
        ("a tomogram", "give picks or particle stacks"),
        ("imported tomogram", "entered as a tomogram"),
        ("imported particles", "names no tomogram"),
        ("odd box", "even integer"),
    ],
)
def test_what_relion_cannot_be_given_is_refused(
    project: dict[str, Any], case: str, match: str
) -> None:
    root = project["root"]
    ids, box = ["TS_01:picks:pytom"], 24
    registry = ArtifactRegistry.load(root / "artifacts.json")
    if case == "no box":
        box = cast(Any, None)
    if case == "twice":
        picks = registry.get("TS_01:picks:pytom")
        _with(root, picks.model_copy(update={"id": "TS_01:picks:pytom:other"}))
        ids.append("TS_01:picks:pytom:other")
    if case == "a tomogram":
        ids = ["TS_01:tomogram:fine"]
    if case == "imported tomogram":
        tomogram = registry.get("TS_01:tomogram:fine")
        _with(root, tomogram.model_copy(update={"provenance": Provenance.IMPORTED}))
    if case == "imported particles":
        picks = registry.get("TS_01:picks:pytom")
        parameters = {key: value for key, value in picks.parameters.items()
                      if key != "input_tomogram_id"}
        _with(root, picks.model_copy(update={"parameters": parameters}))
    if case == "odd box":
        box = 25
    with pytest.raises(ValueError, match=match):
        read_particle_set(root, ids, box=box)


def _script() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "export_relion.py"
    spec = importlib.util.spec_from_file_location("export_relion_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("angles, reconstructs", [("given", True), ("none", False)])
def test_the_commands_extract_and_with_poses_reconstruct(
    tmp_path: Path, project: dict[str, Any], angles: str, reconstructs: bool
) -> None:
    from types import SimpleNamespace

    script = _script()
    particles = read_particle_set(project["root"], ["TS_01:picks:pytom"], box=24)
    chosen = particles.__class__(**{**particles.__dict__, "angles": angles})
    plan = SimpleNamespace(chosen=chosen, series=[SimpleNamespace(relion_bin=2)])
    text = script.particles_commands_text(plan)
    commands = tmp_path / "export" / "relion_commands.sh"
    commands.parent.mkdir()
    commands.write_text(text)
    assert subprocess.run(["bash", "-n", str(commands)], capture_output=True).returncode == 0
    assert "--b 48 --crop 24 --bin 2" in text and "mpirun" not in text

    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = stubs / "calls.log"
    for name in ("relion_tomo_subtomo", "relion_tomo_reconstruct_particle"):
        stub = stubs / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n')
        stub.chmod(0o755)
    refine = stubs / "relion_refine"
    refine.write_text("#!/bin/sh\necho 'RELION version: 5.0.1-commit-test'\n")
    refine.chmod(0o755)
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path)}
    result = subprocess.run(["bash", str(commands)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[0].startswith("relion_tomo_subtomo --i optimisation_set.star")
    assert (len(calls) == 2 and calls[1].startswith(
        "relion_tomo_reconstruct_particle --i optimisation_set.star")) is reconstructs
