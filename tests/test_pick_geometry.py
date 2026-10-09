from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.pick_geometry import (
    MembraneRule,
    PickGeometry,
    measure_picks,
    write_star_subset,
)
from cryoet_pipeline.rotations import matrix_to_relion_euler, relion_euler_to_matrix

_SHAPE = (48, 48, 48)
_CENTRE = np.array([24.0, 24.0, 24.0])  # x, y, z
_RADIUS = 15.0


def _membrane() -> np.ndarray:
    """A spherical shell about 1.5 voxels thick, as (z, y, x)."""

    z, y, x = np.indices(_SHAPE)
    radius = np.sqrt((x - _CENTRE[0]) ** 2 + (y - _CENTRE[1]) ** 2 + (z - _CENTRE[2]) ** 2)
    return np.abs(radius - _RADIUS) < 0.75


def _angles_with_z_along(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """RELION angles whose template z axis lands on `direction` in the tomogram."""

    z_axis = np.asarray(direction, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrix = np.column_stack([x_axis, y_axis, z_axis])
    angles = matrix_to_relion_euler(matrix)
    # The pipeline's convention, x_tomogram = A @ x_template: the column is the axis.
    np.testing.assert_allclose(relion_euler_to_matrix(*angles)[:, 2], z_axis, atol=1e-9)
    return angles


# Each: a position (x, y, z) and the direction its template z axis points.
_PICKS = {
    # Under the membrane, axis outward: toward the membrane, along its normal.
    "inside_outward": ((33.0, 24.0, 24.0), (1.0, 0.0, 0.0)),
    # Same place, axis tangential: right distance, wrong orientation.
    "inside_tangential": ((33.0, 24.0, 24.0), (0.0, 1.0, 0.0)),
    # Outside the shell, axis outward: along the normal but pointing away.
    "outside_outward": ((24.0, 45.0, 24.0), (0.0, 1.0, 0.0)),
    # Far from any membrane.
    "centre": ((24.0, 24.0, 24.0), (0.0, 0.0, 1.0)),
    # On the membrane itself, where there is no direction to it.
    "on_membrane": ((9.0, 24.0, 24.0), (1.0, 0.0, 0.0)),
}


def _write_picks_star(path: Path) -> list[str]:
    rows = []
    for index, ((x, y, z), direction) in enumerate(_PICKS.values()):
        rot, tilt, psi = _angles_with_z_along(direction)
        rows.append(
            f"{x:.6f} {y:.6f} {z:.6f} {rot:.6f} {tilt:.6f} {psi:.6f} {0.5 - 0.01 * index:.6f}"
        )
    path.write_text(
        "# Created by the starfile Python package\n\n"
        "data_particles\n\n"
        "loop_\n"
        "_rlnCoordinateX #1\n"
        "_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n"
        "_rlnAngleRot #4\n"
        "_rlnAngleTilt #5\n"
        "_rlnAnglePsi #6\n"
        "_rlnLCCmax #7\n" + "\n".join(rows) + "\n"
    )
    return list(_PICKS)


def _measured(tmp_path: Path) -> dict[str, PickGeometry]:
    star = tmp_path / "picks.star"
    names = _write_picks_star(star)
    return dict(zip(names, measure_picks(star, _membrane()), strict=True))


def test_distance_and_axis_angle_follow_the_membrane(tmp_path: Path) -> None:
    picks = _measured(tmp_path)

    inside = picks["inside_outward"]
    # Radius 9 against a shell whose inner voxels sit at about 14.25 to 15.
    assert 5.0 <= inside.distance_px <= 6.5
    assert inside.axis_angle_deg < 15.0, "outward, toward the membrane above it"
    assert inside.normal_angle_deg < 15.0

    tangential = picks["inside_tangential"]
    assert tangential.distance_px == inside.distance_px
    assert tangential.normal_angle_deg > 75.0

    outside = picks["outside_outward"]
    assert 4.5 <= outside.distance_px <= 6.5
    # Along the normal as a line, but pointing away from the membrane below it.
    assert outside.axis_angle_deg > 165.0
    assert outside.normal_angle_deg < 15.0

    assert 13.5 <= picks["centre"].distance_px <= 15.0

    on = picks["on_membrane"]
    assert on.distance_px == 0.0
    assert math.isnan(on.axis_angle_deg)


def test_the_rule_keeps_what_sits_and_points_as_the_target_does(tmp_path: Path) -> None:
    picks = _measured(tmp_path)

    def kept(rule: MembraneRule) -> set[str]:
        return {name for name, pick in picks.items() if rule.keeps(pick, 1.0)}

    lattice = MembraneRule(
        max_distance_angstrom=8.0,
        normal_angle_deg=0.0,
        angle_tolerance_deg=20.0,
        axis_direction="toward",
    )
    assert kept(lattice) == {"inside_outward"}
    # Either side of the membrane once the direction is not asked for.
    either = MembraneRule(max_distance_angstrom=8.0, normal_angle_deg=0.0, angle_tolerance_deg=20.0)
    assert kept(either) == {"inside_outward", "outside_outward"}
    # Distance alone, for a target with no fixed orientation: a pick on the
    # membrane has no angle, but no angle is asked of it.
    distance_only = MembraneRule(max_distance_angstrom=8.0)
    assert kept(distance_only) == {
        "inside_outward",
        "inside_tangential",
        "outside_outward",
        "on_membrane",
    }
    assert kept(MembraneRule(min_distance_angstrom=1.0, max_distance_angstrom=8.0)) == {
        "inside_outward",
        "inside_tangential",
        "outside_outward",
    }
    # A target meeting the membrane at its own angle, here a right angle.
    assert kept(
        MembraneRule(max_distance_angstrom=8.0, normal_angle_deg=90.0, angle_tolerance_deg=20.0)
    ) == {"inside_tangential"}


def test_a_rule_refuses_what_it_could_not_apply() -> None:
    with pytest.raises(ValueError, match="go together"):
        MembraneRule(normal_angle_deg=0.0)
    with pytest.raises(ValueError, match="larger than the minimum"):
        MembraneRule(min_distance_angstrom=50.0, max_distance_angstrom=20.0)
    with pytest.raises(ValueError, match="axis direction"):
        MembraneRule(axis_direction="inward")
    with pytest.raises(ValueError, match="0 to 90"):
        MembraneRule(normal_angle_deg=120.0, angle_tolerance_deg=10.0)


def test_the_subset_keeps_the_header_and_the_rows_asked_for(tmp_path: Path) -> None:
    source = tmp_path / "picks.star"
    _write_picks_star(source)
    destination = tmp_path / "kept.star"

    assert write_star_subset(source, destination, [0, 2]) == 2

    columns, rows = read_star_loop(source)
    kept_columns, kept_rows = read_star_loop(destination)
    assert kept_columns == columns
    # Row numbers mean what read_star_loop means by them.
    assert kept_rows == [rows[0], rows[2]]
    header = source.read_text().split("_rlnLCCmax #7\n")[0]
    assert destination.read_text().startswith(header)


def _series_registry(tmp_path: Path, *, with_segmentation: bool = True) -> tuple[Path, Path]:
    tomogram_path = tmp_path / "tomogram.zarr"
    zarr.save(tomogram_path, np.zeros(_SHAPE, dtype=np.float32))
    tomogram = Artifact(
        id="TS_TEST:tomogram:fine:bin8",
        kind=ArtifactKind.TOMOGRAM,
        path=tomogram_path,
        shape=_SHAPE,
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=1.0,
        variant="bin8",
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )
    segmentation_path = tmp_path / "segmentation.zarr"
    zarr.save(segmentation_path, _membrane().astype(np.float32))
    segmentation = Artifact(
        id="TS_TEST:segmentation:full:membrain-seg:bin8",
        kind=ArtifactKind.SEGMENTATION,
        path=segmentation_path,
        parent_ids=[tomogram.id],
        shape=_SHAPE,
        variant="bin8",
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )
    star = tmp_path / "picks" / "TS_TEST_full_pytom-match-pick_bin8_particles.star"
    star.parent.mkdir()
    _write_picks_star(star)
    picks = Artifact(
        id="TS_TEST:picks:full:pytom-match-pick:bin8",
        kind=ArtifactKind.PICKS,
        path=star,
        parent_ids=[tomogram.id],
        shape=(len(_PICKS),),
        pixel_spacing_angstrom=1.0,
        variant="bin8",
        parameters={
            "backend": "pytom-match-pick",
            "tilt_series_id": "TS_TEST",
            "tomogram_branch": "full",
            "input_tomogram_id": tomogram.id,
            "segmentation_id": segmentation.id if with_segmentation else None,
            "pytom_job_json_path": str(tmp_path / "TS_TEST_job.json"),
            "qc_path": str(tmp_path / "picking_qc.json"),
        },
    )
    registry = ArtifactRegistry.empty()
    for artifact in (tomogram, segmentation, picks):
        registry.add(artifact)
    registry_path = tmp_path / "artifacts.json"
    registry.write(registry_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        TiltSeriesManifest(
            tilt_series_id="TS_TEST",
            source_mdoc=Path("TS_TEST.mdoc"),
            raw_pixel_spacing_angstrom=1.0,
            images=[
                TiltImage(
                    z_value=0,
                    tilt_angle_deg=0.0,
                    subframe_path="frame.mrc",
                    num_subframes=1,
                    pixel_spacing_angstrom=1.0,
                    binning=1,
                )
            ],
        ).model_dump_json()
    )
    return registry_path, manifest_path


def _filter(tmp_path: Path, registry: Path, manifest: Path, *extra: str) -> object:
    return CliRunner().invoke(
        cli.app,
        [
            "filter-picks",
            "--manifest",
            str(manifest),
            "--registry",
            str(registry),
            "--out",
            str(tmp_path / "outputs"),
            "--variant",
            "bin8",
            "--output-variant",
            "bin8-membrane",
            "--max-membrane-distance-angstrom",
            "8",
            "--normal-angle-deg",
            "0",
            "--angle-tolerance-deg",
            "20",
            "--axis-direction",
            "toward",
            *extra,
        ],
    )


def test_the_kept_picks_are_a_pick_list_extraction_can_select(tmp_path: Path) -> None:
    registry_path, manifest_path = _series_registry(tmp_path)

    result = _filter(tmp_path, registry_path, manifest_path)

    assert result.exit_code == 0, result.output
    assert "kept 1 of 5" in result.output
    registry = ArtifactRegistry.load(registry_path)
    manifest = TiltSeriesManifest.model_validate_json(manifest_path.read_text())
    # Found by its label, the way extract-particles --variant finds picks.
    filtered = cli._picks_artifact(registry, manifest, variant="bin8-membrane")
    original = registry.get("TS_TEST:picks:full:pytom-match-pick:bin8")
    assert filtered.id == "TS_TEST:picks:full:pytom-match-pick:bin8-membrane"
    assert filtered.parent_ids == [original.id, "TS_TEST:segmentation:full:membrain-seg:bin8"]
    assert filtered.shape == (1,)
    # What extraction reads from picks is carried over unchanged.
    for key in ("input_tomogram_id", "pytom_job_json_path", "backend", "tomogram_branch"):
        assert filtered.parameters[key] == original.parameters[key]
    assert filtered.parameters["membrane_filter"]["kept"] == 1
    _, rows = read_star_loop(filtered.path)
    _, original_rows = read_star_loop(original.path)
    assert rows == [original_rows[0]]
    # The original is untouched and still selected by its own label.
    assert cli._picks_artifact(registry, manifest, variant="bin8").id == original.id
    report = json.loads(Path(filtered.parameters["qc_path"]).read_text())
    assert report["kept"] == 1
    # The pick at the sphere's centre has the membrane evenly around it.
    assert report["no_direction"] == 2


def test_a_dry_run_reports_and_registers_nothing(tmp_path: Path) -> None:
    registry_path, manifest_path = _series_registry(tmp_path)
    before = registry_path.read_text()

    result = _filter(tmp_path, registry_path, manifest_path, "--dry-run")

    assert result.exit_code == 0, result.output
    assert "dry run: nothing registered" in result.output
    assert "distance to membrane" in result.output
    assert registry_path.read_text() == before
    tables = list((tmp_path / "outputs").rglob("*_membrane_filter.tsv"))
    assert len(tables) == 1
    assert not list((tmp_path / "outputs").rglob("*bin8-membrane_particles.star"))


def test_picks_searched_without_a_membrane_are_refused(tmp_path: Path) -> None:
    registry_path, manifest_path = _series_registry(tmp_path, with_segmentation=False)

    result = _filter(tmp_path, registry_path, manifest_path)

    assert result.exit_code != 0
    assert "no membrane" in result.output
