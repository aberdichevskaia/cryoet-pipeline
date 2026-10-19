"""scripts/export_dynamo.py: a project's extracted particles in Dynamo's layout."""

import json
import subprocess
import sys
from pathlib import Path

import mrcfile
import numpy as np
import zarr
from test_wedge import _manifest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import dynamo_to_native
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder
from cryoet_pipeline.particles import ParticleRecord, write_particle_star
from cryoet_pipeline.rotations import relion_euler_to_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qc_dynamo_picks  # noqa: E402

BOX = 12


def _series(project: Path, name: str, count: int, seed: int) -> tuple[Artifact, list]:
    generator = np.random.default_rng(seed)
    folder = project / "particles" / name
    folder.mkdir(parents=True)
    data = generator.normal(size=(count, BOX, BOX, BOX)).astype(np.float32)
    stack = zarr.open(folder / "stack.zarr", mode="w", shape=data.shape, dtype=np.float32)
    stack[:] = data
    records = [
        ParticleRecord(
            index=i,
            x_px=float(generator.integers(20, 200)),
            y_px=float(generator.integers(20, 200)),
            z_px=float(generator.integers(20, 80)),
            rot_deg=float(generator.uniform(-180, 180)),
            tilt_deg=float(generator.uniform(0, 180)),
            psi_deg=float(generator.uniform(-180, 180)),
            shift_x_px=float(generator.uniform(-1, 1)),
            shift_y_px=float(generator.uniform(-1, 1)),
            shift_z_px=float(generator.uniform(-1, 1)),
            score=float(generator.uniform(0.05, 0.3)),
            class_number=1,
            half_set=1 + i % 2,
        )
        for i in range(count)
    ]
    star = folder / "stack.star"
    write_particle_star(star, records)
    artifact = Artifact(
        id=f"{name}:particle_stack:cubic-crop_bin8",
        variant="bin8",
        kind=ArtifactKind.PARTICLE_STACK,
        path=folder / "stack.zarr",
        shape=data.shape,
        axis_order=AxisOrder.NZYX,
        pixel_spacing_angstrom=9.784,
        parameters={"tilt_series_id": name, "metadata_path": str(star)},
    )
    (project / "manifests").mkdir(exist_ok=True)
    manifest = _manifest().model_copy(update={"tilt_series_id": name})
    (project / "manifests" / f"{name}.json").write_text(manifest.model_dump_json())
    return artifact, data


def test_the_export_is_read_by_the_pick_review_and_keeps_poses_and_voxels(tmp_path):
    project = tmp_path / "project"
    registry = ArtifactRegistry.empty()
    stacks = {}
    for name, count, seed in (("TS_A", 5, 1), ("TS_B", 4, 2)):
        artifact, data = _series(project, name, count, seed)
        registry.add(artifact)
        stacks[name] = data
    registry.write(project / "artifacts.json")
    out = tmp_path / "export"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_dynamo.py"), "--project", str(project),
         "--variant", "bin8", "--series", "TS_A", "--series", "TS_B", "--out", str(out)],
        check=True,
    )
    picks = qc_dynamo_picks.load_picks(out)  # validates tags, tomograms and positions
    assert {run: len(p) for run, p in picks.items()} == {"TS_A": 5, "TS_B": 4}

    manifest = json.loads((out / "export_manifest.json").read_text())
    table = np.loadtxt(out / "initial.tbl")
    assert table.shape == (9, 42)
    assert manifest["wedge"]["ranges"] == {"TS_A": [-60.0, 60.0], "TS_B": [-60.0, 60.0]}
    for row, record in zip(table, manifest["particles"], strict=True):
        native = record["native_record"]
        assert row[0] == record["tag"] and row[19] == record["tomogram_number"]
        assert (row[13], row[14]) == (-60.0, 60.0) and row[12] == 1
        matrix, shift = dynamo_to_native(row[6:9], row[3:6], BOX)
        expected = relion_euler_to_matrix(native["rot_deg"], native["tilt_deg"], native["psi_deg"])
        np.testing.assert_allclose(matrix, expected, atol=1e-9)
        np.testing.assert_allclose(
            shift, [native["shift_x_px"], native["shift_y_px"], native["shift_z_px"]], atol=1e-9
        )
        particle = mrcfile.read(out / record["file"])
        np.testing.assert_array_equal(particle, stacks[record["run"]][native["index"]])
    assert json.loads((out / "export_complete.json").read_text())["particles"] == 9
    assert manifest["wedge"]["range_source"] == {"TS_A": "manifest", "TS_B": "manifest"}


def test_the_wedge_is_the_reconstructions_not_the_manifests(tmp_path):
    """Views dropped in alignment are still in the manifest, not in the tomogram.

    TS_A's five views run -60..+60 and its tomogram records views 1-3 only
    (-30..+30), so Dynamo must not be told it holds data out to 60 degrees; TS_B's
    tomogram records nothing and keeps the manifest's range.
    """
    project = tmp_path / "project"
    registry = ArtifactRegistry.empty()
    for name, count, seed, included in (("TS_A", 3, 1, [1, 2, 3]), ("TS_B", 2, 2, None)):
        artifact, _ = _series(project, name, count, seed)
        if name == "TS_A":
            base = _manifest()
            five = base.model_copy(
                update={
                    "tilt_series_id": name,
                    "images": [
                        base.images[0].model_copy(update={"z_value": z, "tilt_angle_deg": angle})
                        for z, angle in enumerate([-60.0, -30.0, 0.0, 30.0, 60.0])
                    ],
                }
            )
            (project / "manifests" / f"{name}.json").write_text(five.model_dump_json())
        tomogram = Artifact(
            id=f"{name}:tomogram:bin8",
            variant="bin8",
            kind=ArtifactKind.TOMOGRAM,
            path=project / f"{name}.zarr",
            parameters={"tilt_series_id": name}
            | ({"included_z_values": included} if included is not None else {}),
        )
        registry.add(tomogram)
        registry.add(
            artifact.model_copy(
                update={"parameters": artifact.parameters | {"input_tomogram_id": tomogram.id}}
            )
        )
    registry.write(project / "artifacts.json")
    out = tmp_path / "export"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_dynamo.py"), "--project", str(project),
         "--variant", "bin8", "--series", "TS_A", "--series", "TS_B", "--out", str(out)],
        check=True,
    )
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["wedge"]["ranges"] == {"TS_A": [-30.0, 30.0], "TS_B": [-60.0, 60.0]}
    assert manifest["wedge"]["range_source"] == {"TS_A": "reconstruction", "TS_B": "manifest"}
    table = np.loadtxt(out / "initial.tbl")
    assert [tuple(row[13:15]) for row in table] == [(-30.0, 30.0)] * 3 + [(-60.0, 60.0)] * 2


def test_decoys_follow_every_real_particle_and_are_marked(tmp_path):
    """Real particles keep the tags an export without decoys gives them."""

    project = tmp_path / "project"
    registry = ArtifactRegistry.empty()
    for name, count, seed in (("TS_A", 3, 1), ("TS_B", 2, 2)):
        artifact, _ = _series(project, name, count, seed)
        registry.add(artifact)
        decoy, _ = _series(project / "decoys", name, 2, seed + 10)
        registry.add(decoy.model_copy(update={
            "id": f"{name}:particle_stack:cubic-crop_bin8-decoy", "variant": "bin8-decoy",
        }))
    registry.write(project / "artifacts.json")
    out = tmp_path / "export"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_dynamo.py"), "--project", str(project),
         "--variant", "bin8", "--decoy-variant", "bin8-decoy", "--series", "TS_A",
         "--series", "TS_B", "--out", str(out)],
        check=True,
    )
    manifest = json.loads((out / "export_manifest.json").read_text())
    particles = manifest["particles"]
    assert [(p["tag"], p["run"], p["decoy"]) for p in particles] == [
        (1, "TS_A", False), (2, "TS_A", False), (3, "TS_A", False),
        (4, "TS_B", False), (5, "TS_B", False),
        (6, "TS_A", True), (7, "TS_A", True), (8, "TS_B", True), (9, "TS_B", True),
    ]
    assert manifest["decoy_variant"] == "bin8-decoy" and manifest["decoy_particles"] == 4
    table = np.loadtxt(out / "initial.tbl")
    assert table[:, 19].tolist() == [1, 1, 1, 2, 2, 1, 1, 2, 2]
    assert json.loads((out / "export_complete.json").read_text())["decoys"] == 4
    assert {run: len(p) for run, p in qc_dynamo_picks.load_picks(out).items()} == {
        "TS_A": 5, "TS_B": 4}


def test_dynamo_gets_the_solved_range_of_the_tomogram_negated(tmp_path):
    """ryr1 039: nominal -42..+42, built from -29..+49 after the +7 deg offset and
    correct-ctf; Dynamo's ytilt for that is -49..+29 (see dynamo_ytilt)."""

    project = tmp_path / "project"
    registry = ArtifactRegistry.empty()
    artifact, _ = _series(project, "TS_A", 3, 1)
    solved = project / "TS_A.tlt"
    solved.write_text("".join(f"{angle:.2f}\n" for angle in (-29.0, 0.0, 49.0)))
    tomogram = Artifact(
        id="TS_A:tomogram:bin8",
        variant="bin8",
        kind=ArtifactKind.TOMOGRAM,
        path=project / "TS_A.zarr",
        parameters={"tilt_series_id": "TS_A", "tilt_angles_path": str(solved)},
    )
    registry.add(tomogram)
    registry.add(
        artifact.model_copy(
            update={"parameters": artifact.parameters | {"input_tomogram_id": tomogram.id}}
        )
    )
    registry.write(project / "artifacts.json")
    out = tmp_path / "export"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_dynamo.py"), "--project", str(project),
         "--variant", "bin8", "--series", "TS_A", "--out", str(out)],
        check=True,
    )
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["wedge"]["ranges"] == {"TS_A": [-29.0, 49.0]}
    assert manifest["wedge"]["range_source"] == {"TS_A": "reconstruction's solved angles"}
    assert manifest["wedge"]["dynamo_ytilt"] == {"TS_A": [-49.0, 29.0]}
    table = np.loadtxt(out / "initial.tbl", ndmin=2)
    assert [tuple(row[13:15]) for row in table] == [(-49.0, 29.0)] * 3
