"""import-picks: coordinates made elsewhere, entered as picks in a named tomogram.

docs/entry_and_exit.md (What an import is, 7). Picks are coordinates in a
tomogram, so an import of picks names the registered tomogram they refer to and
answers what the file leaves open. Each format is one reader, and the answer on
the list says what its columns mean:

- `pytom-star`: `rlnCoordinateX/Y/Z` in voxels of the volume picked, angles
  `rlnAngleRot/Tilt/Psi`. pytom puts a pick half a voxel above the particle's
  centre for an even template box (`particles.pick_center_offset_px`), so the box
  is asked for.
- `relion5-star`: `rlnCenteredCoordinateX/Y/ZAngst`, angles in IMOD's standard
  orientation of the tilt series' full-size tomogram: the exact inverse of
  `relion_export` (`TomogramPlacement.centred_angstrom`, `standard_frame_matrix`),
  whose particles RELION placed where the pipeline did (10 November 2026).
- `csv`: the columns the person names, coordinates of particle centres.
- `imod-model`: points of a model made on the volume, read by IMOD's
  `model2point -float`. Its (X, Y, Z) is voxel (X - 0.5, Y - 0.5, Z) of the volume
  the model was made on, in that volume's orientation: measured 16 November with
  findbeads3d on beads of known centre in IMOD's three layouts
  (scripts/imod_model_check.py). model2point prints two decimals.

`dynamo-table` is refused until measured against Dynamo's own cropping.

Coordinates taken in a volume of another orientation (`picked_in`: by default
the one the tomogram was imported in, or `pipeline` for the project's own) are
mapped by that orientation's transform, and their poses by `M A M`, `M` its linear
part. A mirror in `M` makes the poses relative to the picker's reference mirrored,
as `relion_export` does for the other direction; the picks record it.

The import writes the pipeline's own picks STAR, particle centres in the
tomogram's voxels, the only form extraction reads, with the centre offset 0.
Picks without angles carry zeros and say so. Before anything is registered the
picks are set against boxes at random positions (`picks_check`): a wrong scale,
axis or orientation leaves their average as flat as the random ones'.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    imod_environment,
    resolve_imod_executable,
    run_command,
)
from cryoet_pipeline.backends.protocols import _VARIANT_PATTERN, BackendContext
from cryoet_pipeline.backends.subtomogram import pick_centre_offset
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    Provenance,
    RetentionPolicy,
    StorageRole,
)
from cryoet_pipeline.particles import pick_center_offset_px, read_star_blocks
from cryoet_pipeline.picks_check import PicksCheck, check_picks
from cryoet_pipeline.picks_visualization import EXPORT_RECORD
from cryoet_pipeline.rotations import matrix_to_relion_euler, relion_euler_to_matrix
from cryoet_pipeline.run_lock import run_lock
from cryoet_pipeline.runs import (
    RunInput,
    bytes_fingerprint,
    clear_abandoned,
    completed,
    fingerprint,
    writing,
)
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.tomogram_import import ImportRefused, Transform, orientation

STEP = "import-picks"
STEP_VERSION = 1
BACKEND = "picks-import"
BACKEND_VERSION = 1

FORMATS = ("pytom-star", "relion5-star", "csv", "imod-model")
UNMEASURED_FORMATS = {
    "dynamo-table": "Dynamo's table, measured against Dynamo's own cropping of the same "
    "positions",
    "relion3-star": "RELION 3/4's STAR, measured as RELION 5's was",
    "warp-star": "Warp's STAR, measured as RELION 5's was",
    "copick": "copick points, measured against the CZ portal's annotations",
}
CSV_NAMES = ("x", "y", "z", "rot", "tilt", "psi", "score", "skip")
ORIGINS = ("voxel-0", "voxel-1", "corner", "centre")
"""Where a CSV's zero is: the first voxel's centre, the first voxel's centre called
1 (MATLAB, Dynamo), the first voxel's outer edge, or the volume's centre."""

_LINEAR: dict[Transform, np.ndarray] = {
    "none": np.eye(3),
    "reverse_z": np.diag([1.0, 1.0, -1.0]),
    "transpose_yz": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
}


@dataclass(frozen=True)
class SourcePicks:
    """Picks as read, before they are placed in the canonical tomogram."""

    positions_xyz: np.ndarray
    angles_deg: np.ndarray | None
    scores: np.ndarray | None
    frame: Literal["picked", "relion5"]
    """`picked`: voxels of the volume picked, in its orientation; `relion5`:
    already canonical, from RELION's centred angstroms."""


@dataclass
class PicksImportResult:
    folder: Path
    picks: Artifact
    check: dict[str, Any]
    already_imported: bool
    warnings: list[str] = field(default_factory=list)


def check_format(name: str) -> str:
    if name in FORMATS:
        return name
    if name in UNMEASURED_FORMATS:
        raise ImportRefused(
            f"{name!r} is not on the list yet: it needs {UNMEASURED_FORMATS[name]}. Picks "
            f"in it can be converted to a CSV with stated columns, which is; the list holds "
            f"{', '.join(FORMATS)}"
        )
    raise ImportRefused(f"unknown picks format {name!r}; the list holds {', '.join(FORMATS)}")


def _numbers(rows: Sequence[Sequence[str]], index: int, what: str, path: Path) -> np.ndarray:
    try:
        values = np.array([float(row[index]) for row in rows])
    except (ValueError, IndexError) as exc:
        raise ImportRefused(f"{path}: {what} is not a number on every row ({exc})") from exc
    if not np.isfinite(values).all():
        raise ImportRefused(f"{path}: {what} is not finite on every row")
    return values


def _star_table(path: Path, required: Sequence[str]) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"picks file not found: {path}")
    for columns, rows in read_star_blocks(path).values():
        if all(name in columns for name in required):
            return columns, rows
    raise ImportRefused(f"{path} has no table with the columns {', '.join(required)}")


def read_pytom_star(path: Path, template_box_px: int) -> SourcePicks:
    """pytom's picks, moved from its paste point to the particle's centre."""

    names = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")
    columns, rows = _star_table(path, names)
    positions = np.column_stack([_numbers(rows, columns.index(n), n, path) for n in names])
    angle_names = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    angles = None
    if all(name in columns for name in angle_names):
        angles = np.column_stack(
            [_numbers(rows, columns.index(n), n, path) for n in angle_names]
        )
    scores = (_numbers(rows, columns.index("rlnLCCmax"), "rlnLCCmax", path)
              if "rlnLCCmax" in columns else None)
    # The pick is pytom's paste point T // 2; the particle's centre is (T - 1) / 2.
    centres = positions - pick_center_offset_px(template_box_px)
    return SourcePicks(centres, angles, scores, "picked")


def read_imod_model(path: Path, imod_object: int | None, out: Path) -> SourcePicks:
    """The points of an IMOD model, as voxels of the volume it was made on."""

    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    context = BackendContext(output_dir=out, device=DevicePreference.CPU, parameters={})
    try:
        model2point = resolve_imod_executable("model2point", context)
    except FileNotFoundError as exc:
        raise ImportRefused(f"reading an IMOD model needs IMOD's model2point: {exc}") from exc
    with tempfile.TemporaryDirectory(prefix=".imod-model-") as folder:
        text = Path(folder) / "points.txt"
        result = run_command([str(model2point), "-float", "-object", "-input", str(path),
                              "-output", str(text)], cwd=Path(folder),
                             env=imod_environment(model2point, context))
        if result.returncode != 0 or not text.is_file():
            raise ImportRefused(f"model2point could not read {path}: {result.stderr.strip()}")
        rows = [line.split() for line in text.read_text().splitlines() if line.strip()]
    if not rows:
        raise ImportRefused(f"{path} holds no points")
    table = np.array([[float(value) for value in row[:5]] for row in rows])
    objects = sorted({int(number) for number in table[:, 0]})
    if imod_object is None and len(objects) > 1:
        counts = ", ".join(f"{number} ({int((table[:, 0] == number).sum())} points)"
                           for number in objects)
        raise ImportRefused(f"{path} has objects {counts}; name the picks' one with --imod-object")
    if imod_object is not None:
        table = table[table[:, 0] == imod_object]
        if not len(table):
            raise ImportRefused(f"{path} has no points in object {imod_object}; it has {objects}")
    # A pixel's centre is at index + 0.5 in X and Y, a section's at its index in Z.
    positions = table[:, 2:5] - np.array([0.5, 0.5, 0.0])
    return SourcePicks(positions, None, None, "picked")


def read_csv(path: Path, column_names: Sequence[str]) -> SourcePicks:
    """A table of particle centres whose columns the person named, in that order."""

    if not path.is_file():
        raise FileNotFoundError(f"picks file not found: {path}")
    unknown = [name for name in column_names if name not in CSV_NAMES]
    named = [name for name in column_names if name != "skip"]
    if unknown or len(set(named)) != len(named) or not {"x", "y", "z"} <= set(named):
        raise ImportRefused(
            f"--csv-columns names each column, in order, from {', '.join(CSV_NAMES)}, with x, "
            f"y and z among them and no name twice; got {', '.join(column_names)}"
        )
    angle_names = [name for name in ("rot", "tilt", "psi") if name in named]
    if angle_names and len(angle_names) != 3:
        raise ImportRefused("angles are rot, tilt and psi together, or none of them")
    text = path.read_text()
    delimiter = next((mark for mark in (",", "\t", ";") if mark in text), None)
    if delimiter is None:
        rows = [line.split() for line in text.splitlines() if line.strip()]
    else:
        rows = [[cell.strip() for cell in row]
                for row in csv.reader(text.splitlines(), delimiter=delimiter) if any(row)]
    if rows:
        try:
            float(rows[0][column_names.index("x")])
        except (ValueError, IndexError):
            rows = rows[1:]  # a header line
    if not rows:
        raise ImportRefused(f"{path} holds no picks")
    if any(len(row) != len(column_names) for row in rows):
        raise ImportRefused(
            f"{path}: --csv-columns names {len(column_names)} columns, but rows have "
            f"{sorted({len(row) for row in rows})}"
        )

    def column(name: str) -> np.ndarray:
        return _numbers(rows, column_names.index(name), name, path)

    positions = np.column_stack([column("x"), column("y"), column("z")])
    angles = np.column_stack([column(name) for name in ("rot", "tilt", "psi")]) \
        if angle_names else None
    scores = column("score") if "score" in named else None
    return SourcePicks(positions, angles, scores, "picked")


@dataclass(frozen=True)
class RelionGeometry:
    tilt_series_pixel_angstrom: float
    binning: int
    size_xyz: tuple[int, int, int]


def relion_geometry(tomogram: Artifact, tilt_series_pixel_angstrom: float) -> RelionGeometry:
    """Where the tomogram sits in RELION's: as `relion_export.TomogramPlacement` has it."""

    voxel = tomogram.pixel_spacing_angstrom
    shape = tomogram.shape
    if voxel is None or shape is None or len(shape) != 3:
        raise ImportRefused(f"{tomogram.id} records no voxel size or shape")
    ratio = voxel / tilt_series_pixel_angstrom
    binning = round(ratio)
    if binning < 1 or abs(ratio - binning) > 1e-3 * ratio:
        raise ImportRefused(
            f"the tomogram's voxel ({voxel:g} A) is not a whole number of the tilt series' "
            f"pixels ({tilt_series_pixel_angstrom:g} A); only whole binnings are measured"
        )
    nz, ny, nx = (int(size) for size in shape)
    return RelionGeometry(tilt_series_pixel_angstrom, binning,
                          (binning * nx, binning * ny, binning * nz))


def relion5_to_canonical(centred_angstrom: np.ndarray, angles_deg: np.ndarray | None,
                         geometry: RelionGeometry, depth: int
                         ) -> tuple[np.ndarray, np.ndarray | None]:
    """The inverse of `TomogramPlacement.centred_angstrom` and `standard_frame_matrix`."""

    b = geometry.binning
    stack_px = np.asarray(centred_angstrom, dtype=float) / geometry.tilt_series_pixel_angstrom \
        + np.asarray(geometry.size_xyz, dtype=float) / 2.0
    standard = (stack_px - (b - 1) / 2.0) / b
    canonical = standard.copy()
    canonical[:, 2] = (depth - 1) - standard[:, 2]
    if angles_deg is None:
        return canonical, None
    return canonical, transform_angles(angles_deg, _LINEAR["reverse_z"])


def read_relion5_star(path: Path, tomo_name: str, geometry: RelionGeometry,
                      depth: int) -> SourcePicks:
    names = ("rlnCenteredCoordinateXAngst", "rlnCenteredCoordinateYAngst",
             "rlnCenteredCoordinateZAngst")
    columns, rows = _star_table(path, names)
    if "rlnTomoName" in columns:
        rows = [row for row in rows if row[columns.index("rlnTomoName")] == tomo_name]
        if not rows:
            raise ImportRefused(f"{path} has no particles of rlnTomoName {tomo_name!r}")
    for prefix in ("rlnOrigin", "rlnTomoSubtomogram"):
        offsets = [name for name in columns if name.startswith(prefix)]
        for name in offsets:
            if np.any(_numbers(rows, columns.index(name), name, path) != 0):
                raise ImportRefused(
                    f"{path}: {name} is not zero. How RELION 5 combines it with the "
                    "coordinates and angles has not been measured here; apply it to them in "
                    "RELION first"
                )
    centred = np.column_stack([_numbers(rows, columns.index(n), n, path) for n in names])
    angle_names = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    angles = None
    if all(name in columns for name in angle_names):
        angles = np.column_stack(
            [_numbers(rows, columns.index(n), n, path) for n in angle_names]
        )
    positions, canonical_angles = relion5_to_canonical(centred, angles, geometry, depth)
    return SourcePicks(positions, canonical_angles, None, "relion5")


def relion5_tilt_series_pixel(path: Path, tomo_name: str) -> float | None:
    """rlnTomoTiltSeriesPixelSize from an optics table, or tomograms.star's row."""

    for columns, rows in read_star_blocks(path).values():
        if "rlnTomoTiltSeriesPixelSize" not in columns:
            continue
        if "rlnTomoName" in columns:
            rows = [row for row in rows if row[columns.index("rlnTomoName")] == tomo_name]
        values = {float(row[columns.index("rlnTomoTiltSeriesPixelSize")]) for row in rows}
        if len(values) == 1:
            return values.pop()
    return None


def _check_relion_size(path: Path, tomo_name: str, geometry: RelionGeometry) -> None:
    """RELION's own tomogram size must be the named tomogram's, binning times over."""

    for columns, rows in read_star_blocks(path).values():
        names = ("rlnTomoName", "rlnTomoSizeX", "rlnTomoSizeY", "rlnTomoSizeZ")
        if not all(name in columns for name in names):
            continue
        for row in rows:
            if row[columns.index("rlnTomoName")] != tomo_name:
                continue
            size = tuple(int(round(float(row[columns.index(name)]))) for name in names[1:])
            if size != geometry.size_xyz:
                raise ImportRefused(
                    f"RELION's tomogram {tomo_name} is {size} pixels (x, y, z), but the named "
                    f"tomogram at binning {geometry.binning} covers {geometry.size_xyz}: they "
                    "are not one reconstruction's geometry"
                )
            return
    raise ImportRefused(f"{path} has no row for rlnTomoName {tomo_name!r} with its size")


def picked_shape_xyz(shape_zyx: Sequence[int] | None, transform: Transform
                     ) -> tuple[int, int, int]:
    """The volume picked, as (x, y, z) lengths of its own columns, rows and sections."""

    if shape_zyx is None or len(shape_zyx) != 3:
        raise ImportRefused("the tomogram records no shape")
    nz, ny, nx = (int(size) for size in shape_zyx)
    return (nx, nz, ny) if transform == "transpose_yz" else (nx, ny, nz)


def from_origin(values: np.ndarray, origin: str, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    """Coordinates in voxels from their stated zero, to the first voxel's centre as 0."""

    if origin == "voxel-0":
        return values
    if origin == "voxel-1":
        return values - 1.0
    if origin == "corner":
        return values - 0.5
    if origin == "centre":
        return values + (np.asarray(shape_xyz, dtype=float) - 1.0) / 2.0
    raise ImportRefused(f"--coordinate-origin is one of {', '.join(ORIGINS)}; got {origin!r}")


def transform_angles(angles_deg: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """Poses of the source frame in the canonical one: `M A M`, RELION's convention."""

    result = []
    for rot, tilt, psi in np.asarray(angles_deg, dtype=float):
        matrix = linear @ relion_euler_to_matrix(rot, tilt, psi) @ linear
        result.append(matrix_to_relion_euler(matrix))
    return np.array(result, dtype=float).reshape(-1, 3)


def to_canonical(source: SourcePicks, transform: Transform, depth: int
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """Positions and poses of picks taken in a volume of `transform`'s orientation."""

    if source.frame == "relion5":
        return source.positions_xyz, source.angles_deg
    x, y, z = (source.positions_xyz[:, axis] for axis in range(3))
    if transform == "transpose_yz":
        positions = np.column_stack([x, z, y])
    elif transform == "reverse_z":
        positions = np.column_stack([x, y, (depth - 1) - z])
    else:
        positions = np.column_stack([x, y, z])
    angles = None
    if source.angles_deg is not None:
        angles = transform_angles(source.angles_deg, _LINEAR[transform])
    return positions, angles


def picks_star_text(positions_xyz: np.ndarray, angles_deg: np.ndarray | None,
                    scores: np.ndarray | None) -> str:
    """The pipeline's picks STAR: particle centres in the tomogram's voxels."""

    count = len(positions_xyz)
    angles = angles_deg if angles_deg is not None else np.zeros((count, 3))
    score_column = scores if scores is not None else np.zeros(count)
    labels = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot",
              "rlnAngleTilt", "rlnAnglePsi", "rlnLCCmax")
    lines = ["data_particles", "", "loop_"]
    lines += [f"_{label} #{number}" for number, label in enumerate(labels, start=1)]
    for position, angle, score in zip(positions_xyz, angles, score_column, strict=True):
        values = [*position, *angle, score]
        lines.append(" ".join(f"{float(value):.6f}" for value in values))
    return "\n".join(lines) + "\n"


def _tomogram(registry: ArtifactRegistry, tilt_series_id: str, tomogram_id: str | None,
              variant: str | None) -> Artifact:
    if tomogram_id is not None:
        try:
            found = registry.get(tomogram_id)
        except KeyError as exc:
            raise ImportRefused(f"no tomogram {tomogram_id!r} in the registry") from exc
        if found.kind is not ArtifactKind.TOMOGRAM or not found.covers(tilt_series_id):
            raise ImportRefused(f"{tomogram_id} is not a tomogram of {tilt_series_id}")
        return found
    matches = [
        artifact for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)
        if artifact.covers(tilt_series_id)
        and artifact.parameters.get("tomogram_branch", "full") == "full"
        and (variant is None or artifact.variant == variant)
    ]
    if len(matches) != 1:
        raise ImportRefused(
            f"{tilt_series_id}: expected one tomogram{f' labelled {variant!r}' if variant else ''}"
            f", found {[artifact.id for artifact in matches] or 'none'}; name it with "
            "--tomogram-id"
        )
    return matches[0]


def import_picks(
    *,
    picks_path: Path,
    format_name: str,
    tilt_series_id: str,
    out: Path,
    particle_diameter_angstrom: float,
    tomogram_id: str | None = None,
    tomogram_variant: str | None = None,
    picked_in: str | None = None,
    template_box_px: int | None = None,
    csv_columns: Sequence[str] | None = None,
    coordinate_units: str | None = None,
    coordinate_origin: str | None = None,
    relion_tomo_name: str | None = None,
    relion_tilt_series_pixel: float | None = None,
    relion_tomograms_star: Path | None = None,
    imod_object: int | None = None,
    variant: str | None = None,
    overrule_picks_check: bool = False,
) -> PicksImportResult:
    """Enter picks made elsewhere as this project's picks in one of its tomograms."""

    if variant is not None and not _VARIANT_PATTERN.fullmatch(variant):
        raise ImportRefused(f"invalid variant {variant!r}")
    check_format(format_name)
    if particle_diameter_angstrom <= 0:
        raise ImportRefused("--particle-diameter must be positive")
    registry_path = out / "artifacts.json"
    if not registry_path.exists():
        raise ImportRefused(f"{out} has no registry; import or make the tomogram first")
    registry = ArtifactRegistry.load(registry_path)
    tomogram = _tomogram(registry, tilt_series_id, tomogram_id, tomogram_variant)
    voxel = tomogram.pixel_spacing_angstrom
    if voxel is None or tomogram.shape is None:
        raise ImportRefused(f"{tomogram.id} records no voxel size or shape")
    depth = int(tomogram.shape[0])

    content: dict[str, Any] = {"tilt_series_id": tilt_series_id, "format": format_name,
                               "particle_diameter_angstrom": particle_diameter_angstrom}
    inputs = [
        RunInput(slot="picks", path=str(picks_path.resolve()),
                 fingerprint=bytes_fingerprint(picks_path)),
        # A tomogram of the registry is known by its import's record, or else by its id.
        RunInput(slot="tomogram", path=str(tomogram.path),
                 fingerprint=str(tomogram.parameters.get("run_fingerprint",
                                                          f"artifact:{tomogram.id}"))),
    ]
    transform: Transform = "none"
    if format_name == "relion5-star":
        name = relion_tomo_name or tilt_series_id
        pixel = relion_tilt_series_pixel or relion5_tilt_series_pixel(picks_path, name)
        if relion_tomograms_star is not None:
            inputs.append(RunInput(slot="relion_tomograms",
                                   path=str(relion_tomograms_star.resolve()),
                                   fingerprint=bytes_fingerprint(relion_tomograms_star)))
            if pixel is None:
                pixel = relion5_tilt_series_pixel(relion_tomograms_star, name)
        if pixel is None:
            raise ImportRefused(
                f"{picks_path} does not give the tilt series' pixel size "
                "(rlnTomoTiltSeriesPixelSize); give RELION's tomograms.star with "
                "--relion-tomograms-star, or --relion-tilt-series-pixel-size"
            )
        geometry = relion_geometry(tomogram, pixel)
        if relion_tomograms_star is not None:
            _check_relion_size(relion_tomograms_star, name, geometry)
        source = read_relion5_star(picks_path, name, geometry, depth)
        content.update({"relion_tomo_name": name, "relion_tilt_series_pixel_angstrom": pixel,
                        "relion_binning": geometry.binning})
    else:
        picked = picked_in or str(tomogram.parameters.get("orientation", "pipeline"))
        transform = orientation(picked).transform
        content["picked_in"] = picked
        if format_name == "pytom-star":
            if template_box_px is None:
                raise ImportRefused(
                    "pytom's picks lie half a voxel off their particles for an even template "
                    "box; give the box pytom matched with as --template-box-px"
                )
            source = read_pytom_star(picks_path, template_box_px)
            content["template_box_px"] = template_box_px
        elif format_name == "imod-model":
            source = read_imod_model(picks_path, imod_object, out)
            if imod_object is not None:
                content["imod_object"] = imod_object
        else:
            if not csv_columns:
                raise ImportRefused("a CSV needs --csv-columns, naming each column in order")
            if coordinate_units not in ("voxels", "angstrom"):
                raise ImportRefused("a CSV needs --coordinate-units: voxels or angstrom")
            if coordinate_origin not in ORIGINS:
                raise ImportRefused(
                    f"a CSV needs --coordinate-origin, one of {', '.join(ORIGINS)}: where its "
                    "zero is, stated rather than assumed"
                )
            source = read_csv(picks_path, csv_columns)
            positions_in_voxels = from_origin(
                source.positions_xyz / voxel if coordinate_units == "angstrom"
                else source.positions_xyz,
                coordinate_origin, picked_shape_xyz(tomogram.shape, transform),
            )
            source = SourcePicks(positions_in_voxels, source.angles_deg, source.scores, "picked")
            content.update({"csv_columns": list(csv_columns),
                            "coordinate_units": coordinate_units,
                            "coordinate_origin": coordinate_origin})
    positions, angles = to_canonical(source, transform, depth)
    mirrored = format_name == "relion5-star" or float(np.linalg.det(_LINEAR[transform])) < 0
    return _check_write_register(
        registry=registry, registry_path=registry_path, tomogram=tomogram,
        tilt_series_id=tilt_series_id, out=out, positions=positions, angles=angles,
        scores=source.scores, content=content, inputs=inputs,
        particle_diameter_angstrom=particle_diameter_angstrom,
        overrule_picks_check=overrule_picks_check, variant=variant, edit=None,
        source_path=picks_path, format_name=format_name, mirrored=mirrored,
    )


@dataclass(frozen=True)
class _Edit:
    """What makes picks an attested edit rather than an import."""

    derived_from: Artifact
    counts: dict[str, int]
    angles_mark: str
    qc: dict[str, Any]


def _check_write_register(
    *,
    registry: ArtifactRegistry,
    registry_path: Path,
    tomogram: Artifact,
    tilt_series_id: str,
    out: Path,
    positions: np.ndarray,
    angles: np.ndarray | None,
    scores: np.ndarray | None,
    content: dict[str, Any],
    inputs: list[RunInput],
    particle_diameter_angstrom: float,
    overrule_picks_check: bool,
    variant: str | None,
    edit: _Edit | None,
    source_path: Path,
    format_name: str,
    mirrored: bool,
) -> PicksImportResult:
    """Check the picks, write them as a run, and register them: imported, or an edit."""

    voxel = cast(float, tomogram.pixel_spacing_angstrom)
    if overrule_picks_check:
        content["picks_check_overruled"] = True
    # One and a half diameters: the statistic is read inside the box's sphere.
    box = min(96, max(16, 2 * math.ceil(0.75 * particle_diameter_angstrom / voxel)))
    tomogram_array = cast(Any, zarr.open(tomogram.path, mode="r"))
    check = check_picks(tomogram_array, positions, box)
    warnings = _judge_check(check, overrule_picks_check)

    run_fingerprint = fingerprint(STEP, STEP_VERSION, BACKEND, BACKEND_VERSION, content, inputs)
    label = f"_{variant}" if variant else ""
    stem = "picks_edited" if edit is not None else "picks"
    folder = out / "imports" / tilt_series_id / f"{stem}{label}"
    artifact_id = (f"{tilt_series_id}:picks:{'edited' if edit is not None else 'imported'}"
                   + (f":{variant}" if variant else ""))
    provenance = Provenance.ATTESTED if edit is not None else Provenance.IMPORTED
    angles_mark = edit.angles_mark if edit is not None else (
        "given" if angles is not None else "none")
    if artifact_id in registry.artifact_ids and (
        registry.get(artifact_id).parameters.get("run_fingerprint") != run_fingerprint
    ):
        raise ImportRefused(
            f"{artifact_id} is in the registry already, from another file or other answers. "
            "Give --variant to keep both"
        )
    with run_lock(folder.with_name(folder.name + ".lock")):
        record = completed(folder)
        already = record is not None
        if record is not None:
            if record.fingerprint != run_fingerprint:
                raise ImportRefused(
                    f"{folder} holds an import of another file or other answers. Give "
                    "--variant to keep both"
                )
        elif folder.exists():
            raise ImportRefused(f"{folder} exists without a completed run.json; look at it first")
        else:
            clear_abandoned(folder)
            with writing(folder, step=STEP, step_version=STEP_VERSION, backend=BACKEND,
                         backend_version=BACKEND_VERSION, content_params=content,
                         inputs=inputs) as attempt:
                star = f"{tilt_series_id}_picks.star"
                (attempt.folder / star).write_text(picks_star_text(positions, angles, scores))
                report: dict[str, Any] = {
                    "tomogram_id": tomogram.id, "format": format_name, "count": len(positions),
                    "angles": angles_mark,
                    "reference_mirrored": mirrored and angles is not None,
                    "check": check.record(), "warnings": warnings,
                }
                if edit is not None:
                    report["edit"] = edit.qc
                (attempt.folder / "picks_qc.json").write_text(json.dumps(report, indent=2) + "\n")
                attempt.add(ArtifactKind.PICKS, star, provenance=provenance)
                attempt.add(ArtifactKind.QC, "picks_qc.json", provenance=provenance)
                attempt.metrics.update({"picks": float(len(positions)),
                                        "picks_inside": float(check.inside),
                                        "check_beaten": float(check.beaten)})
                if edit is not None:
                    attempt.metrics.update({key: float(value)
                                            for key, value in edit.counts.items()})
            record = completed(folder)
            if record is None:
                raise RuntimeError(f"{folder} has no completed run.json after the import")

    star_path = folder / f"{tilt_series_id}_picks.star"
    parameters: dict[str, Any] = {
        "backend": BACKEND,
        "tilt_series_id": tilt_series_id,
        "input_tomogram_id": tomogram.id,
        "format": format_name,
        "source_path": str(source_path.resolve()),
        "pick_center_offset_px": 0.0,
        "angles": angles_mark,
        "reference_mirrored": mirrored and angles is not None,
        "picks_check_verdict": check.verdict,
        "count": len(positions),
        "run_path": str(folder),
        "run_fingerprint": record.fingerprint,
    }
    if edit is not None:
        parameters.update({"derived_from": edit.derived_from.id, "edited_with": "3dmod",
                           **{f"edit_{key}": value for key, value in edit.counts.items()}})
    picks = Artifact(
        id=artifact_id,
        variant=variant,
        kind=ArtifactKind.PICKS,
        path=star_path,
        parent_ids=[edit.derived_from.id] if edit is not None else [tomogram.id],
        parameters=parameters,
        storage_role=StorageRole.SOURCE,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        provenance=provenance,
        size_bytes=star_path.stat().st_size,
    )
    if artifact_id not in registry.artifact_ids:
        registry.add(picks)
        registry.commit(registry_path)
    return PicksImportResult(folder=folder, picks=picks, check=check.record(),
                             already_imported=already, warnings=warnings)


EDIT_KEPT_PX = 0.02
"""A point this close to an exported pick is that pick, kept: the export writes three
decimals and model2point prints two."""


@dataclass(frozen=True)
class EditMatch:
    """Each exported pick paired with at most one point of the edit."""

    kept: list[tuple[int, int]]
    """(exported, edited) indices of picks left where they were."""
    moved: list[tuple[int, int, float]]
    """(exported, edited, distance in voxels) of picks a person moved."""
    added: list[int]
    """Edited indices with no exported pick near them."""
    removed: list[int]
    """Exported indices with no point of the edit near them."""


def match_edit(exported: np.ndarray, edited: np.ndarray, *, move_radius_px: float) -> EditMatch:
    """Pair an export's picks with the points of its edit, the nearest pairs first.

    Within `EDIT_KEPT_PX` a pick is kept; within `move_radius_px` it was moved; a pick
    with no point that near was removed, and a point with no pick that near added.
    """

    exported = np.asarray(exported, dtype=float).reshape(-1, 3)
    edited = np.asarray(edited, dtype=float).reshape(-1, 3)
    candidates: list[tuple[float, int, int]] = []
    for start in range(0, len(exported), 512):
        block = exported[start:start + 512]
        distances = np.linalg.norm(block[:, None, :] - edited[None, :, :], axis=2)
        rows, columns = np.nonzero(distances <= move_radius_px)
        candidates += [(float(distances[r, c]), int(r) + start, int(c))
                       for r, c in zip(rows, columns, strict=True)]
    candidates.sort()
    taken_exported: set[int] = set()
    taken_edited: set[int] = set()
    kept: list[tuple[int, int]] = []
    moved: list[tuple[int, int, float]] = []
    for distance, i, j in candidates:
        if i in taken_exported or j in taken_edited:
            continue
        taken_exported.add(i)
        taken_edited.add(j)
        if distance <= EDIT_KEPT_PX:
            kept.append((i, j))
        else:
            moved.append((i, j, distance))
    return EditMatch(
        kept=sorted(kept),
        moved=sorted(moved),
        added=[j for j in range(len(edited)) if j not in taken_edited],
        removed=[i for i in range(len(exported)) if i not in taken_exported],
    )


def import_edited_picks(
    *,
    picks_path: Path,
    derived_from: str,
    tilt_series_id: str,
    out: Path,
    particle_diameter_angstrom: float,
    imod_object: int | None = None,
    template_box_px: int | None = None,
    variant: str | None = None,
    overrule_picks_check: bool = False,
) -> PicksImportResult:
    """Enter a person's edit of exported picks, made in 3dmod, as attested picks.

    docs/entry_and_exit.md (What an import is, 9). The edit's format and
    orientation are the export's: visualize-picks writes an IMOD model on the
    canonical tomogram, so nothing about them is asked. Picks left in place go
    back to their particles' centres as the exported ones would (a pytom pick
    is half a voxel off for an even template box); a point a person moved or
    added is taken as the centre they placed it at. Kept and moved picks keep
    their poses and scores; added ones have none.
    """

    if variant is not None and not _VARIANT_PATTERN.fullmatch(variant):
        raise ImportRefused(f"invalid variant {variant!r}")
    if particle_diameter_angstrom <= 0:
        raise ImportRefused("--particle-diameter must be positive")
    if picks_path.suffix.lower() != ".mod":
        raise ImportRefused(
            f"{picks_path}: an edit comes back in its export's format, and visualize-picks "
            "exports an IMOD model for 3dmod (<series>_picks.mod). ArtiaX's saved STAR is not "
            "measured yet"
        )
    registry_path = out / "artifacts.json"
    if not registry_path.exists():
        raise ImportRefused(f"{out} has no registry")
    registry = ArtifactRegistry.load(registry_path)
    try:
        exported = registry.get(derived_from)
    except KeyError as exc:
        raise ImportRefused(f"no artifact {derived_from!r} in the registry") from exc
    if exported.kind is not ArtifactKind.PICKS or not exported.covers(tilt_series_id):
        raise ImportRefused(f"{derived_from} is not picks of {tilt_series_id}")
    record_path = picks_path.parent / EXPORT_RECORD
    export_record = None
    if record_path.is_file():
        export_record = json.loads(record_path.read_text())
        if export_record.get("picks_id") != derived_from:
            raise ImportRefused(
                f"{record_path} says this model was exported from "
                f"{export_record.get('picks_id')!r}, not {derived_from!r}"
            )
    tomogram_id = exported.parameters.get("input_tomogram_id")
    try:
        tomogram = registry.get(str(tomogram_id))
    except KeyError as exc:
        raise ImportRefused(f"{derived_from} names no tomogram of the registry") from exc
    voxel = tomogram.pixel_spacing_angstrom
    if voxel is None or tomogram.shape is None:
        raise ImportRefused(f"{tomogram.id} records no voxel size or shape")
    try:
        offset = pick_centre_offset(exported, template_box_px)
    except ValueError as exc:
        raise ImportRefused(f"{derived_from}: {exc}") from exc

    names = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")
    columns, rows = _star_table(exported.path, names)
    coordinates = np.column_stack(
        [_numbers(rows, columns.index(n), n, exported.path) for n in names])
    angle_names = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    exported_angles = None
    if all(name in columns for name in angle_names):
        exported_angles = np.column_stack(
            [_numbers(rows, columns.index(n), n, exported.path) for n in angle_names])
    exported_scores = (_numbers(rows, columns.index("rlnLCCmax"), "rlnLCCmax", exported.path)
                       if "rlnLCCmax" in columns else None)
    # The model's points are the export's coordinates: voxels of the canonical tomogram.
    edited = read_imod_model(picks_path, imod_object, out).positions_xyz
    match = match_edit(coordinates, edited, move_radius_px=particle_diameter_angstrom / 2 / voxel)

    order = sorted([(i, j, "kept") for i, j in match.kept]
                   + [(i, j, "moved") for i, j, _ in match.moved])
    positions = np.array(
        [coordinates[i] - offset if how == "kept" else edited[j] for i, j, how in order]
        + [edited[j] for j in match.added]
    ).reshape(-1, 3)
    angles = None
    angles_mark = "none"
    if exported_angles is not None:
        angles = np.vstack([exported_angles[[i for i, _, _ in order]],
                            np.zeros((len(match.added), 3))])
        angles_mark = "partial" if match.added else "given"
    scores = None
    if exported_scores is not None:
        scores = np.concatenate([exported_scores[[i for i, _, _ in order]],
                                 np.zeros(len(match.added))])
    counts = {"kept": len(match.kept), "moved": len(match.moved), "added": len(match.added),
              "removed": len(match.removed)}
    qc = {
        **counts,
        "derived_from": derived_from,
        "exported_centre_offset_px": offset,
        "moved_px": [round(distance, 3) for _, _, distance in match.moved],
        "removed_rows": match.removed,
        "export_record": str(record_path) if export_record is not None else None,
    }
    content: dict[str, Any] = {"tilt_series_id": tilt_series_id, "format": "imod-model",
                               "derived_from": derived_from,
                               "particle_diameter_angstrom": particle_diameter_angstrom}
    if imod_object is not None:
        content["imod_object"] = imod_object
    if template_box_px is not None:
        content["template_box_px"] = template_box_px
    inputs = [
        RunInput(slot="picks", path=str(picks_path.resolve()),
                 fingerprint=bytes_fingerprint(picks_path)),
        RunInput(slot="exported_picks", path=str(exported.path),
                 fingerprint=bytes_fingerprint(exported.path)),
        RunInput(slot="tomogram", path=str(tomogram.path),
                 fingerprint=str(tomogram.parameters.get("run_fingerprint",
                                                          f"artifact:{tomogram.id}"))),
    ]
    return _check_write_register(
        registry=registry, registry_path=registry_path, tomogram=tomogram,
        tilt_series_id=tilt_series_id, out=out, positions=positions, angles=angles,
        scores=scores, content=content, inputs=inputs,
        particle_diameter_angstrom=particle_diameter_angstrom,
        overrule_picks_check=overrule_picks_check, variant=variant,
        edit=_Edit(derived_from=exported, counts=counts, angles_mark=angles_mark, qc=qc),
        source_path=picks_path, format_name="imod-model",
        mirrored=bool(exported.parameters.get("reference_mirrored", False)),
    )


def _judge_check(check: PicksCheck, overrule: bool) -> list[str]:
    warnings = []
    if check.verdict == "weak":
        warnings.append(f"the picks barely stand out from random positions: {check.reason}")
    if check.verdict == "not testable":
        warnings.append(f"the picks could not be checked: {check.reason}")
    if check.verdict in {"flat", "outside", "mirrored"}:
        if not overrule:
            raise ImportRefused(
                f"the picks check refuses this import: {check.reason}. If the check itself is "
                "wrong here, --overrule-picks-check imports them and records that you did"
            )
        warnings.append(f"picks check overruled: {check.reason}")
    return warnings


__all__ = [
    "FORMATS",
    "EditMatch",
    "PicksImportResult",
    "check_format",
    "import_edited_picks",
    "import_picks",
    "match_edit",
    "picks_star_text",
    "read_csv",
    "read_pytom_star",
    "read_relion5_star",
    "relion5_to_canonical",
    "relion_geometry",
    "to_canonical",
    "transform_angles",
]
