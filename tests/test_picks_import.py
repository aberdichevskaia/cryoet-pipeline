"""import-picks: each format and each orientation picked in give back the particles.

The tomogram here holds bright blobs at known centres, in noise, with a missing
wedge so that it imports as a raw reconstruction would. Picks are written from
those centres the way each program writes them, in the orientation it saw, and
the import has to give the centres back, with poses that are the same rotations.
The RELION 5 picks are written with relion_export's own functions, so that
importing them is checked as the inverse of the export RELION confirmed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import pytest
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import ArtifactKind, Provenance
from cryoet_pipeline.particles import read_star_blocks
from cryoet_pipeline.picks_import import import_picks, relion_geometry
from cryoet_pipeline.relion_export import TomogramPlacement, standard_frame_matrix
from cryoet_pipeline.rotations import matrix_to_relion_euler, relion_euler_to_matrix
from cryoet_pipeline.runs import read_record
from cryoet_pipeline.tomogram_import import ImportRefused, import_tomogram

SHAPE = (64, 160, 176)  # z, y, x
VOXEL = 10.0
DIAMETER = 80.0
LOW, HIGH = -60.0, 50.0
F = np.diag([1.0, 1.0, -1.0])


def _wedged(volume: np.ndarray, low: float, high: float, floor: float = 0.2) -> np.ndarray:
    spectrum = np.fft.rfftn(volume)
    kz = np.fft.fftfreq(volume.shape[0])[:, None, None]
    kx = np.fft.rfftfreq(volume.shape[2])[None, None, :]
    at_low = kx * np.sin(np.radians(low)) + kz * np.cos(np.radians(low))
    at_high = kx * np.sin(np.radians(high)) + kz * np.cos(np.radians(high))
    spectrum *= np.where(np.broadcast_to(at_low * at_high <= 0, spectrum.shape), 1.0, floor)
    return np.fft.irfftn(spectrum, s=volume.shape, axes=(0, 1, 2)).astype(np.float32)


@pytest.fixture(scope="module")
def scene() -> dict[str, Any]:
    """A tomogram with blobs at integral centres, and their poses."""

    rng = np.random.default_rng(7)
    centres: list[np.ndarray] = []
    while len(centres) < 60:
        candidate = np.array([rng.integers(14, SHAPE[2] - 14), rng.integers(14, SHAPE[1] - 14),
                              rng.integers(14, SHAPE[0] - 14)], dtype=float)  # x, y, z
        if all(np.linalg.norm(candidate - other) > 12 for other in centres):
            centres.append(candidate)
    grid = np.indices(SHAPE).astype(float)
    volume = rng.normal(0.0, 1.0, SHAPE)
    for x, y, z in centres:
        volume += 3.0 * np.exp(-((grid[0] - z) ** 2 + (grid[1] - y) ** 2 + (grid[2] - x) ** 2)
                               / (2 * 2.5**2))
    angles = rng.uniform([-180, 0, -180], [180, 180, 180], size=(len(centres), 3))
    return {"volume": _wedged(volume, LOW, HIGH), "centres": np.array(centres),
            "angles": angles}


@pytest.fixture
def project(tmp_path: Path, scene: dict[str, Any]) -> Path:
    volume = tmp_path / "volume.mrc"
    with mrcfile.new(volume) as mrc:
        mrc.set_data(scene["volume"])
        mrc.voxel_size = VOXEL
    angles = tmp_path / "volume.tlt"
    angles.write_text("".join(f"{a:.1f}\n" for a in np.arange(LOW, HIGH + 0.1, 2.0)))
    out = tmp_path / "project"
    import_tomogram(volume_path=volume, tilt_angles_path=angles, orientation_name="pipeline",
                    processing="raw", tilt_series_id="TS_01", out=out)
    return out


def _pytom_star(path: Path, positions: np.ndarray, angles: np.ndarray | None = None) -> Path:
    labels = ["rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"]
    columns = [positions]
    if angles is not None:
        labels += ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]
        columns.append(angles)
    labels.append("rlnLCCmax")
    columns.append(np.full((len(positions), 1), 0.3))
    table = np.hstack(columns)
    path.write_text("data_\n\nloop_\n" + "".join(f"_{name} #{i}\n" for i, name in
                                                 enumerate(labels, start=1))
                    + "".join(" ".join(f"{v:.6f}" for v in row) + "\n" for row in table))
    return path


def _imported(result_path: Path) -> tuple[np.ndarray, np.ndarray]:
    columns, rows = next(iter(read_star_blocks(result_path).values()))
    table = np.array([[float(value) for value in row] for row in rows])
    positions = table[:, [columns.index(n) for n in ("rlnCoordinateX", "rlnCoordinateY",
                                                     "rlnCoordinateZ")]]
    angles = table[:, [columns.index(n) for n in ("rlnAngleRot", "rlnAngleTilt",
                                                  "rlnAnglePsi")]]
    return positions, angles


def _same_rotations(first: np.ndarray, second: np.ndarray) -> None:
    for one, two in zip(first, second, strict=True):
        np.testing.assert_allclose(relion_euler_to_matrix(*one), relion_euler_to_matrix(*two),
                                   atol=1e-6)


def _source_angles(angles: np.ndarray, linear: np.ndarray) -> np.ndarray:
    return np.array([matrix_to_relion_euler(linear @ relion_euler_to_matrix(*a) @ linear)
                     for a in angles])


# How a picker that saw the volume in each orientation writes a centre (x, y, z).
PICKED = {
    "pipeline": (lambda c: c, np.eye(3)),
    "imod-standard": (lambda c: np.column_stack([c[:, 0], c[:, 1], (SHAPE[0] - 1) - c[:, 2]]), F),
    "imod-tilt": (lambda c: np.column_stack([c[:, 0], c[:, 2], c[:, 1]]),
                  np.array([[1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])),
}


@pytest.mark.parametrize("picked_in", sorted(PICKED))
def test_pytom_picks_from_any_orientation_give_back_the_centres_and_poses(
    tmp_path: Path, project: Path, scene: dict[str, Any], picked_in: str
) -> None:
    to_source, linear = PICKED[picked_in]
    # pytom's pick is the paste point T // 2: half a voxel above the centre for an even box.
    star = _pytom_star(tmp_path / "picks.star", to_source(scene["centres"]) + 0.5,
                       _source_angles(scene["angles"], linear))

    result = import_picks(picks_path=star, format_name="pytom-star", tilt_series_id="TS_01",
                          out=project, particle_diameter_angstrom=DIAMETER,
                          picked_in=picked_in, template_box_px=16)

    positions, angles = _imported(result.picks.path)
    np.testing.assert_allclose(positions, scene["centres"], atol=1e-5)
    _same_rotations(angles, scene["angles"])
    assert result.check["verdict"] == "lands", result.check["reason"]
    assert result.picks.parameters["reference_mirrored"] is (picked_in != "pipeline")


def test_relion5_picks_import_as_the_inverse_of_the_export(
    tmp_path: Path, project: Path, scene: dict[str, Any]
) -> None:
    placement = TomogramPlacement(shape_zyx=SHAPE, binning=4, stack_pixel_angstrom=VOXEL / 4)
    rows = []
    for centre, angle in zip(scene["centres"], scene["angles"], strict=True):
        centred = placement.centred_angstrom(centre)
        standard = matrix_to_relion_euler(standard_frame_matrix(relion_euler_to_matrix(*angle)))
        rows.append(("TS_01", *centred, *standard))
    star = tmp_path / "particles.star"
    star.write_text(
        "# version 50001\n\ndata_optics\n\nloop_\n_rlnOpticsGroup #1\n"
        "_rlnTomoTiltSeriesPixelSize #2\n1 2.5\n\ndata_particles\n\nloop_\n_rlnTomoName #1\n"
        "_rlnCenteredCoordinateXAngst #2\n_rlnCenteredCoordinateYAngst #3\n"
        "_rlnCenteredCoordinateZAngst #4\n_rlnAngleRot #5\n_rlnAngleTilt #6\n_rlnAnglePsi #7\n"
        + "".join(" ".join(str(value) for value in row) + "\n" for row in rows)
    )

    result = import_picks(picks_path=star, format_name="relion5-star", tilt_series_id="TS_01",
                          out=project, particle_diameter_angstrom=DIAMETER)

    positions, angles = _imported(result.picks.path)
    np.testing.assert_allclose(positions, scene["centres"], atol=1e-5)
    _same_rotations(angles, scene["angles"])
    assert result.check["verdict"] == "lands"


def test_relion5_offsets_and_fractional_binnings_are_refused(
    tmp_path: Path, project: Path
) -> None:
    star = tmp_path / "particles.star"
    star.write_text(
        "data_particles\n\nloop_\n_rlnTomoName #1\n_rlnCenteredCoordinateXAngst #2\n"
        "_rlnCenteredCoordinateYAngst #3\n_rlnCenteredCoordinateZAngst #4\n"
        "_rlnOriginXAngst #5\nTS_01 0 0 0 4.0\n"
    )
    with pytest.raises(ImportRefused, match="rlnOriginXAngst"):
        import_picks(picks_path=star, format_name="relion5-star", tilt_series_id="TS_01",
                     out=project, particle_diameter_angstrom=DIAMETER,
                     relion_tilt_series_pixel=2.5)
    tomogram = ArtifactRegistry.load(project / "artifacts.json").get("TS_01:tomogram:imported")
    with pytest.raises(ImportRefused, match="whole number"):
        relion_geometry(tomogram, 3.0)


@pytest.mark.parametrize(
    "units, origin, convert",
    [
        ("voxels", "voxel-0", lambda c: c),
        ("voxels", "voxel-1", lambda c: c + 1),
        ("voxels", "corner", lambda c: c + 0.5),
        ("angstrom", "centre",
         lambda c: (c - (np.array([SHAPE[2], SHAPE[1], SHAPE[0]]) - 1) / 2) * VOXEL),
    ],
)
def test_a_csv_says_its_units_and_origin(
    tmp_path: Path, project: Path, scene: dict[str, Any], units: str, origin: str,
    convert: Any,
) -> None:
    values = convert(scene["centres"])
    path = tmp_path / "picks.csv"
    path.write_text("id,x,y,z\n" + "".join(f"{i},{x},{y},{z}\n" for i, (x, y, z) in
                                           enumerate(values)))

    result = import_picks(picks_path=path, format_name="csv", tilt_series_id="TS_01",
                          out=project, particle_diameter_angstrom=DIAMETER,
                          csv_columns=["skip", "x", "y", "z"], coordinate_units=units,
                          coordinate_origin=origin)

    positions, angles = _imported(result.picks.path)
    np.testing.assert_allclose(positions, scene["centres"], atol=1e-5)
    assert result.picks.parameters["angles"] == "none"
    assert not angles.any()


def test_a_csv_without_its_origin_is_refused(tmp_path: Path, project: Path) -> None:
    path = tmp_path / "picks.csv"
    path.write_text("1 2 3\n")
    with pytest.raises(ImportRefused, match="--coordinate-origin"):
        import_picks(picks_path=path, format_name="csv", tilt_series_id="TS_01", out=project,
                     particle_diameter_angstrom=DIAMETER, csv_columns=["x", "y", "z"],
                     coordinate_units="voxels")


def test_picks_with_their_axes_exchanged_are_refused_by_the_check(
    tmp_path: Path, project: Path, scene: dict[str, Any]
) -> None:
    swapped = scene["centres"][:, [1, 0, 2]]
    swapped = swapped[(swapped[:, 1] < SHAPE[1] - 14)]
    star = _pytom_star(tmp_path / "picks.star", swapped + 0.5)
    arguments: dict[str, Any] = dict(picks_path=star, format_name="pytom-star",
                                     tilt_series_id="TS_01", out=project,
                                     particle_diameter_angstrom=DIAMETER, template_box_px=16)

    with pytest.raises(ImportRefused, match="random"):
        import_picks(**arguments)
    assert not (project / "imports" / "TS_01" / "picks").exists()

    overruled = import_picks(overrule_picks_check=True, **arguments)
    record = read_record(overruled.folder)
    assert record is not None and record.content_params["picks_check_overruled"] is True


def test_picks_at_the_wrong_scale_fall_outside_and_are_refused(
    tmp_path: Path, project: Path, scene: dict[str, Any]
) -> None:
    star = _pytom_star(tmp_path / "picks.star", 2 * scene["centres"] + 0.5)
    with pytest.raises(ImportRefused, match="outside"):
        import_picks(picks_path=star, format_name="pytom-star", tilt_series_id="TS_01",
                     out=project, particle_diameter_angstrom=DIAMETER, template_box_px=16)


def test_formats_not_measured_yet_are_refused_with_what_they_need(
    tmp_path: Path, project: Path
) -> None:
    path = tmp_path / "picks.tbl"
    path.write_text("1 1 1\n")
    for name, needs in (("dynamo-table", "Dynamo's own cropping"),
                        ("relion3-star", "measured as RELION 5's was")):
        with pytest.raises(ImportRefused, match=needs):
            import_picks(picks_path=path, format_name=name, tilt_series_id="TS_01",
                         out=project, particle_diameter_angstrom=DIAMETER)


def test_imported_picks_are_a_run_and_extraction_reads_them_as_centres(
    tmp_path: Path, project: Path, scene: dict[str, Any]
) -> None:
    star = _pytom_star(tmp_path / "picks.star", scene["centres"] + 0.5, scene["angles"])
    result = import_picks(picks_path=star, format_name="pytom-star", tilt_series_id="TS_01",
                          out=project, particle_diameter_angstrom=DIAMETER, template_box_px=16)
    again = import_picks(picks_path=star, format_name="pytom-star", tilt_series_id="TS_01",
                         out=project, particle_diameter_angstrom=DIAMETER, template_box_px=16)
    assert again.already_imported

    registry = ArtifactRegistry.load(project / "artifacts.json")
    picks = registry.get("TS_01:picks:imported")
    assert picks.provenance is Provenance.IMPORTED
    assert picks.parent_ids == ["TS_01:tomogram:imported"]
    record = read_record(result.folder)
    assert record is not None
    assert {item.slot for item in record.inputs} == {"picks", "tomogram"}
    qc = json.loads((result.folder / "picks_qc.json").read_text())
    assert qc["check"]["verdict"] == "lands"

    manifest = project / "manifests" / "TS_01.json"
    outcome = CliRunner().invoke(cli.app, [
        "extract-particles", "--manifest", str(manifest), "--registry",
        str(project / "artifacts.json"), "--out", str(project), "--box-size-px", "16",
    ])
    assert outcome.exit_code == 0, outcome.output
    registry = ArtifactRegistry.load(project / "artifacts.json")
    (stack,) = registry.by_kind(ArtifactKind.PARTICLE_STACK)
    assert stack.parameters["pick_center_offset_px"] == 0.0
    upstream = {artifact.id for artifact in registry.not_derived_lineage(stack.id)}
    assert upstream == {"TS_01:picks:imported", "TS_01:tomogram:imported"}


def test_the_command_line_imports_picks(tmp_path: Path, project: Path,
                                        scene: dict[str, Any]) -> None:
    star = _pytom_star(tmp_path / "picks.star", scene["centres"] + 0.5)
    outcome = CliRunner().invoke(cli.app, [
        "import-picks", "--picks", str(star), "--format", "pytom-star", "--tilt-series-id",
        "TS_01", "--out", str(project), "--particle-diameter", str(DIAMETER),
        "--template-box-px", "16",
    ])
    assert outcome.exit_code == 0, outcome.output
    assert "picks check: lands" in outcome.output


def test_star_blocks_are_read_by_name(tmp_path: Path) -> None:
    path = tmp_path / "particles.star"
    path.write_text("data_optics\n\nloop_\n_rlnOpticsGroup #1\n1\n\n"
                    "data_particles\n\nloop_\n_rlnTomoName #1\n_rlnAngleRot #2\nTS_01 10.0\n")
    blocks = read_star_blocks(path)
    assert blocks["optics"] == (["rlnOpticsGroup"], [["1"]])
    assert blocks["particles"] == (["rlnTomoName", "rlnAngleRot"], [["TS_01", "10.0"]])


def test_picks_taken_in_the_mirror_orientation_are_refused_by_the_check(
    tmp_path: Path, project: Path, scene: dict[str, Any]
) -> None:
    # Taken in IMOD's standard orientation, answered as the pipeline's own: the
    # picks land mirrored in z, and their mirror lands on the particles.
    to_source, _ = PICKED["imod-standard"]
    star = _pytom_star(tmp_path / "picks.star", to_source(scene["centres"]) + 0.5)

    with pytest.raises(ImportRefused, match="mirrored in z"):
        import_picks(picks_path=star, format_name="pytom-star", tilt_series_id="TS_01",
                     out=project, particle_diameter_angstrom=DIAMETER, template_box_px=16,
                     picked_in="pipeline")


def _imod(program: str) -> str | None:
    import shutil

    found = shutil.which(program)
    if found:
        return found
    bundled = Path("/Applications/IMOD/bin") / program
    return str(bundled) if bundled.is_file() else None


needs_imod = pytest.mark.skipif(_imod("point2model") is None or _imod("model2point") is None,
                                reason="IMOD's point2model and model2point are not installed")


def _imod_model(path: Path, rows: list[str]) -> Path:
    """A scattered-point model written by IMOD's own point2model."""

    import os
    import subprocess

    points = path.with_suffix(".points")
    points.write_text("".join(f"{row}\n" for row in rows))
    program = _imod("point2model")
    assert program is not None
    imod_dir = str(Path(program).parent.parent)
    subprocess.run([program, "-scat", "-input", str(points), "-output", str(path)], check=True,
                   capture_output=True, env={**os.environ, "IMOD_DIR": imod_dir})
    return path


@needs_imod
@pytest.mark.parametrize("picked_in", ["pipeline", "imod-standard"])
def test_an_imod_model_gives_back_the_centres_it_was_made_on(
    tmp_path: Path, project: Path, scene: dict[str, Any], picked_in: str
) -> None:
    # In IMOD's model a pixel's centre is at index + 0.5 in X and Y, a section's at
    # its index in Z (scripts/imod_model_check.py measured it).
    to_source, _ = PICKED[picked_in]
    source = to_source(scene["centres"])
    model = _imod_model(tmp_path / "picks.mod",
                        [f"{x + 0.5} {y + 0.5} {z}" for x, y, z in source])

    result = import_picks(picks_path=model, format_name="imod-model", tilt_series_id="TS_01",
                          out=project, particle_diameter_angstrom=DIAMETER,
                          picked_in=picked_in)

    positions, _ = _imported(result.picks.path)
    np.testing.assert_allclose(positions, scene["centres"], atol=0.006)
    assert result.check["verdict"] == "lands"
    assert result.picks.parameters["angles"] == "none"


@needs_imod
def test_a_model_with_several_objects_needs_the_picks_object_named(
    tmp_path: Path, project: Path, scene: dict[str, Any]
) -> None:
    centres = scene["centres"]
    rows = [f"1 1 {x + 0.5} {y + 0.5} {z}" for x, y, z in centres]
    rows += [f"2 1 {x + 0.5} {y + 0.5} {z}" for x, y, z in centres[:5] + [3, 3, 0]]
    model = _imod_model(tmp_path / "picks.mod", rows)
    arguments: dict[str, Any] = dict(picks_path=model, format_name="imod-model",
                                     tilt_series_id="TS_01", out=project,
                                     particle_diameter_angstrom=DIAMETER)

    with pytest.raises(ImportRefused, match="--imod-object"):
        import_picks(**arguments)
    result = import_picks(imod_object=1, **arguments)
    positions, _ = _imported(result.picks.path)
    np.testing.assert_allclose(positions, centres, atol=0.006)
