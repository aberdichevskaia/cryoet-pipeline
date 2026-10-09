"""Where each pick sits relative to a membrane, and which picks a geometric rule keeps.

Template matching places and orients every pick from the template alone, so
nothing in its output says whether a pick is where the target can be. For a
target with a fixed geometry relative to a membrane, the membrane segmentation
says so independently of the template: how far the pick sits from the
membrane, and at what angle a chosen template axis meets the membrane normal.

Neither number is universal. The HIV-1 Gag lattice lies parallel to the
membrane at a fixed depth under it, so the template axis normal to the lattice
runs along the membrane normal (0 deg). Another membrane protein meets the
membrane at its own angle, and a particle that merely sits near a membrane has
no fixed angle at all, so for it only the distance applies. Both are therefore
parameters of the rule, never assumptions of the code.

The membrane normal at a pick is the direction from the pick to the centre of
the membrane patch nearest it: every membrane voxel within the nearest
distance plus a few voxels. The nearest voxel alone does not do: on a voxelised
surface it sits off the normal by up to a voxel sideways, which on a sphere of
radius 15 with the pick 6 voxels under it turned a pick lying exactly along
the normal into 29.5 deg. The patch is symmetric about the normal, so its
centre is on it. On the membrane itself, or where the membrane surrounds the
pick evenly, there is no direction, and a rule that needs an angle drops such a
pick rather than guessing one.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.rotations import relion_euler_to_matrix

TEMPLATE_AXES = ("x", "y", "z")
AXIS_DIRECTIONS = ("either", "toward", "away")

_DISTANCE_BIN_ANGSTROM = 20.0
_DISTANCE_BINS = 30
_ANGLE_BIN_DEG = 10.0
_LOW_KEPT_FRACTION = 0.1
_PATCH_MARGIN_PX = 3.0
"""How far past the nearest membrane voxel the patch that sets the normal reaches."""
_NO_DIRECTION_PX = 0.5


@dataclass(frozen=True)
class PickGeometry:
    """One pick against the membrane: how far, and how its template axis points."""

    row: int
    """Position among the STAR file's data rows, as `read_star_loop` counts them."""
    x_px: float
    y_px: float
    z_px: float
    score: float | None
    distance_px: float
    """To the nearest membrane voxel, centre to centre."""
    axis_angle_deg: float
    """Between the template axis and the direction to the nearest membrane patch.

    0 to 180: below 90 the axis points toward the membrane, above 90 away from
    it. NaN where there is no direction: on the membrane, or where it surrounds
    the pick evenly.
    """

    @property
    def normal_angle_deg(self) -> float:
        """The same angle against the normal as a line, 0 to 90, ignoring sign."""

        return min(self.axis_angle_deg, 180.0 - self.axis_angle_deg)


@dataclass(frozen=True)
class MembraneRule:
    """Which picks to keep, in angstrom and degrees."""

    min_distance_angstrom: float = 0.0
    max_distance_angstrom: float | None = None
    normal_angle_deg: float | None = None
    """Expected angle between the template axis and the membrane normal, 0 to 90."""
    angle_tolerance_deg: float | None = None
    axis_direction: str = "either"
    """`toward` or `away` from the membrane, or `either`."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_distance_angstrom) or self.min_distance_angstrom < 0:
            raise ValueError("the minimum distance must be finite and not negative")
        if self.max_distance_angstrom is not None and not (
            self.max_distance_angstrom > self.min_distance_angstrom
        ):
            raise ValueError("the maximum distance must be larger than the minimum")
        if (self.normal_angle_deg is None) != (self.angle_tolerance_deg is None):
            raise ValueError("an expected angle and its tolerance go together")
        if self.normal_angle_deg is not None and not 0.0 <= self.normal_angle_deg <= 90.0:
            raise ValueError("the expected angle to the membrane normal must be 0 to 90 deg")
        if self.angle_tolerance_deg is not None and not 0.0 < self.angle_tolerance_deg <= 90.0:
            raise ValueError("the angle tolerance must be above 0 and at most 90 deg")
        if self.axis_direction not in AXIS_DIRECTIONS:
            raise ValueError(f"axis direction must be one of {', '.join(AXIS_DIRECTIONS)}")

    @property
    def uses_angle(self) -> bool:
        return self.normal_angle_deg is not None or self.axis_direction != "either"

    def keeps_orientation(self, pick: PickGeometry) -> bool:
        """The angular half of the rule alone, so distances can be read under it."""

        if not self.uses_angle:
            return True
        angle = pick.axis_angle_deg
        if math.isnan(angle):
            return False
        if self.axis_direction == "toward" and not angle < 90.0:
            return False
        if self.axis_direction == "away" and not angle > 90.0:
            return False
        if self.normal_angle_deg is not None:
            tolerance = cast(float, self.angle_tolerance_deg)
            if abs(pick.normal_angle_deg - self.normal_angle_deg) > tolerance:
                return False
        return True

    def keeps(self, pick: PickGeometry, voxel_size_angstrom: float) -> bool:
        distance = pick.distance_px * voxel_size_angstrom
        if distance < self.min_distance_angstrom:
            return False
        if self.max_distance_angstrom is not None and distance > self.max_distance_angstrom:
            return False
        return self.keeps_orientation(pick)

    def as_parameters(self) -> dict[str, object]:
        return {
            "min_distance_angstrom": self.min_distance_angstrom,
            "max_distance_angstrom": self.max_distance_angstrom,
            "normal_angle_deg": self.normal_angle_deg,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "axis_direction": self.axis_direction,
        }


def measure_picks(
    star_path: Path,
    membrane: np.ndarray,
    *,
    template_axis: str = "z",
) -> list[PickGeometry]:
    """Distance and template-axis angle to the membrane for every pick in a STAR file.

    `membrane` is a boolean (z, y, x) mask on the tomogram's voxel grid, the grid
    pytom's coordinates refer to. The axis is read off the pipeline's own
    rotation, `x_tomogram = A @ x_template`, so it is the column of `A` — the
    same axis ArtiaX draws for that pick.
    """

    if template_axis not in TEMPLATE_AXES:
        raise ValueError(f"template axis must be one of {', '.join(TEMPLATE_AXES)}")
    columns, rows = read_star_loop(star_path)
    required = (
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnCoordinateZ",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
    )
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"{star_path} is missing required STAR columns: {missing}")
    index = {name: columns.index(name) for name in required}
    score_index = columns.index("rlnLCCmax") if "rlnLCCmax" in columns else None

    if membrane.ndim != 3:
        raise ValueError(f"expected a (z, y, x) membrane mask, got shape {membrane.shape}")
    membrane_xyz = np.argwhere(membrane)[:, ::-1].astype(np.float64)
    if len(membrane_xyz) == 0:
        raise ValueError("the membrane segmentation has no voxels to measure against")
    tree = cKDTree(membrane_xyz)

    picks_xyz = np.array(
        [
            [
                float(fields[index["rlnCoordinateX"]]),
                float(fields[index["rlnCoordinateY"]]),
                float(fields[index["rlnCoordinateZ"]]),
            ]
            for fields in rows
        ],
        dtype=np.float64,
    )
    distances, _ = tree.query(picks_xyz)
    axis_column = TEMPLATE_AXES.index(template_axis)

    geometries = []
    for row, fields in enumerate(rows):
        matrix = relion_euler_to_matrix(
            float(fields[index["rlnAngleRot"]]),
            float(fields[index["rlnAngleTilt"]]),
            float(fields[index["rlnAnglePsi"]]),
        )
        axis = matrix[:, axis_column]
        distance = float(distances[row])
        toward = np.zeros(3)
        length = 0.0
        if distance >= _NO_DIRECTION_PX:
            patch = tree.query_ball_point(picks_xyz[row], r=distance + _PATCH_MARGIN_PX)
            toward = membrane_xyz[patch].mean(axis=0) - picks_xyz[row]
            length = float(np.linalg.norm(toward))
        if length < _NO_DIRECTION_PX:
            angle = math.nan
        else:
            cosine = float(np.dot(axis, toward) / length)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        score = float(fields[score_index]) if score_index is not None else None
        geometries.append(
            PickGeometry(
                row=row,
                x_px=float(picks_xyz[row, 0]),
                y_px=float(picks_xyz[row, 1]),
                z_px=float(picks_xyz[row, 2]),
                score=score,
                distance_px=float(distances[row]),
                axis_angle_deg=angle,
            )
        )
    return geometries


def write_star_subset(source: Path, destination: Path, keep: Sequence[int]) -> int:
    """Copy a STAR file keeping only the data rows at the given positions.

    Positions count data rows the way `read_star_loop` does — every non-empty
    line after the first block's column names — so a row from `measure_picks`
    names the same line here. Everything before the rows is copied as it is, so
    whatever reads pytom's own layout still can. Returns the rows written.
    """

    lines = source.read_text().splitlines()
    data_start = _star_data_start(lines)
    if data_start is None:
        raise ValueError(f"no STAR data rows found in {source}")
    wanted = set(keep)
    written = lines[:data_start]
    row = 0
    kept = 0
    for line in lines[data_start:]:
        if not line.strip():
            continue
        if row in wanted:
            written.append(line)
            kept += 1
        row += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(written) + "\n")
    return kept


def _star_data_start(lines: Sequence[str]) -> int | None:
    """The first data row of the first `loop_` block, found as `read_star_loop` finds it."""

    in_loop = False
    has_columns = False
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line == "loop_":
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            has_columns = True
            continue
        if in_loop and has_columns and line:
            return line_index
    return None


def load_membrane_mask(segmentation: Artifact) -> np.ndarray:
    """The segmentation as pick-particles used it for its search mask: nonzero is membrane."""

    data = np.asarray(cast(Any, zarr.open(segmentation.path, mode="r"))[:])
    if data.ndim != 3:
        raise ValueError(f"segmentation {segmentation.id} is not a volume: {data.shape}")
    return data > 0


def summarize(
    geometries: Sequence[PickGeometry],
    kept: Sequence[bool],
    *,
    voxel_size_angstrom: float,
    rule: MembraneRule,
) -> dict[str, object]:
    """Counts and histograms to choose a rule from, and to see what one did."""

    distances = np.array([pick.distance_px for pick in geometries]) * voxel_size_angstrom
    oriented = np.array([rule.keeps_orientation(pick) for pick in geometries], dtype=bool)
    edges = [
        index * _DISTANCE_BIN_ANGSTROM for index in range(_DISTANCE_BINS + 1)
    ]
    histogram = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        inside = (distances >= low) & (distances < high)
        histogram.append(
            {
                "from_angstrom": low,
                "to_angstrom": high,
                "picks": int(inside.sum()),
                "picks_oriented": int((inside & oriented).sum()),
            }
        )
    beyond = distances >= edges[-1]
    angles = np.array([pick.normal_angle_deg for pick in geometries])
    angle_histogram = []
    for low in np.arange(0.0, 90.0, _ANGLE_BIN_DEG):
        high = low + _ANGLE_BIN_DEG
        upper = (angles <= high) if high >= 90.0 else (angles < high)
        angle_histogram.append(
            {
                "from_deg": float(low),
                "to_deg": float(high),
                "picks": int(((angles >= low) & upper).sum()),
            }
        )
    raw = np.array([pick.axis_angle_deg for pick in geometries])
    return {
        "picks": len(geometries),
        "kept": int(sum(bool(value) for value in kept)),
        "rule": rule.as_parameters(),
        "distance_histogram": histogram,
        "beyond_last_bin": {
            "from_angstrom": edges[-1],
            "picks": int(beyond.sum()),
            "picks_oriented": int((beyond & oriented).sum()),
        },
        "normal_angle_histogram": angle_histogram,
        "axis_toward_membrane": int(np.sum(raw < 90.0)),
        "axis_away_from_membrane": int(np.sum(raw > 90.0)),
        "no_direction": int(np.sum(np.isnan(raw))),
        "distance_median_angstrom": float(np.median(distances)) if len(distances) else None,
    }


def summary_lines(summary: dict[str, object]) -> list[str]:
    """The summary as text, for a job log read on the cluster."""

    lines = []
    histogram = cast(list[dict[str, Any]], summary["distance_histogram"])
    beyond = cast(dict[str, Any], summary["beyond_last_bin"])
    lines.append("distance to membrane (A)   all picks   with the orientation rule")
    for entry in histogram:
        if entry["picks"] == 0:
            continue
        lines.append(
            f"  {entry['from_angstrom']:5.0f} - {entry['to_angstrom']:5.0f}"
            f"   {entry['picks']:9d}   {entry['picks_oriented']:9d}"
        )
    if beyond["picks"]:
        lines.append(
            f"  {beyond['from_angstrom']:5.0f} -      "
            f"   {beyond['picks']:9d}   {beyond['picks_oriented']:9d}"
        )
    lines.append("template axis to membrane normal (deg, as a line)")
    for entry in cast(list[dict[str, Any]], summary["normal_angle_histogram"]):
        lines.append(f"  {entry['from_deg']:4.0f} - {entry['to_deg']:3.0f}   {entry['picks']:9d}")
    lines.append(
        f"axis toward the membrane {summary['axis_toward_membrane']}, "
        f"away {summary['axis_away_from_membrane']}, "
        f"no direction (on it, or surrounded by it) {summary['no_direction']}"
    )
    lines.append(f"kept {summary['kept']} of {summary['picks']}")
    return lines


@dataclass(frozen=True)
class MembraneFilterResult:
    artifacts: list[Artifact]
    """The filtered picks and their QC; empty for a dry run."""
    summary: dict[str, object]
    table_path: Path
    report_path: Path
    star_path: Path | None


def filter_picks_by_membrane(
    picks: Artifact,
    segmentation: Artifact,
    manifest: TiltSeriesManifest,
    *,
    output_root: Path,
    label: str,
    rule: MembraneRule,
    template_axis: str = "z",
    overwrite: bool = False,
    dry_run: bool = False,
) -> MembraneFilterResult:
    """Measure the picks against the membrane and, unless a dry run, write the kept ones.

    The kept picks become a pick list of their own under `label`, a copy of the
    original's record with its STAR file narrowed: everything downstream that
    reads picks — the template box behind the centre convention, the tomogram
    they were found in — reads it from there unchanged, and selects them by the
    label like any other lane's.
    """

    if picks.kind is not ArtifactKind.PICKS:
        raise ValueError(f"expected picks, got {picks.kind}")
    if segmentation.kind is not ArtifactKind.SEGMENTATION:
        raise ValueError(f"expected a segmentation, got {segmentation.kind}")
    if picks.variant == label:
        raise ValueError(
            f"the filtered picks need a label of their own; {picks.id} already carries {label!r}"
        )
    voxel_size = picks.pixel_spacing_angstrom
    if voxel_size is None or not voxel_size > 0:
        raise ValueError(f"{picks.id} records no voxel size")

    series = manifest.tilt_series_id
    branch = str(picks.parameters.get("tomogram_branch", "full"))
    backend = str(picks.parameters.get("backend", "picks"))
    prefix = f"{series}_{branch}_{backend}_{label}"
    picks_dir = output_root / "picks" / series
    qc_dir = output_root / "qc" / series / "picking"
    star_path = picks_dir / f"{prefix}_particles.star"
    table_path = qc_dir / f"{prefix}_membrane_filter.tsv"
    report_path = qc_dir / f"{prefix}_membrane_filter_qc.json"
    outputs = [table_path, report_path] if dry_run else [star_path, table_path, report_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"membrane filter outputs already exist: {joined}")

    geometries = measure_picks(
        picks.path, load_membrane_mask(segmentation), template_axis=template_axis
    )
    kept = [rule.keeps(pick, voxel_size) for pick in geometries]
    summary = summarize(geometries, kept, voxel_size_angstrom=voxel_size, rule=rule)
    summary.update(
        {
            "picks_id": picks.id,
            "segmentation_id": segmentation.id,
            "template_axis": template_axis,
            "voxel_size_angstrom": voxel_size,
            "dry_run": dry_run,
        }
    )
    kept_count = cast(int, summary["kept"])
    warnings = []
    if geometries and kept_count / len(geometries) < _LOW_KEPT_FRACTION:
        warnings.append(
            f"the rule keeps {kept_count} of {len(geometries)} picks; check it against "
            "the histograms before using them"
        )
    status = QcStatus.WARNING if warnings else QcStatus.PASS
    summary["status"] = status.value
    summary["warnings"] = warnings

    _write_table(table_path, geometries, kept, voxel_size)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if dry_run:
        return MembraneFilterResult([], summary, table_path, report_path, None)
    if kept_count == 0:
        raise ValueError("the rule keeps no picks; nothing to register")

    keep_rows = [pick.row for pick, keep in zip(geometries, kept, strict=True) if keep]
    written = write_star_subset(picks.path, star_path, keep_rows)
    filtered_id = f"{series}:picks:{branch}:{backend}:{label}"
    filtered = Artifact.model_validate(
        {
            **picks.model_dump(mode="json"),
            "id": filtered_id,
            "variant": label,
            "path": str(star_path),
            "parent_ids": [picks.id, segmentation.id],
            "shape": [written],
            "size_bytes": star_path.stat().st_size,
            "parameters": {
                **picks.parameters,
                "filtered_from": picks.id,
                "membrane_filter": {
                    **rule.as_parameters(),
                    "template_axis": template_axis,
                    "segmentation_id": segmentation.id,
                    "picks": len(geometries),
                    "kept": written,
                },
                "membrane_filter_table_path": str(table_path),
                "picking_qc_path": picks.parameters.get("qc_path"),
                "qc_path": str(report_path),
            },
        }
    )
    qc = Artifact(
        id=f"{series}:qc:picks_membrane_filter:{backend}:{label}",
        variant=label,
        kind=ArtifactKind.QC,
        path=report_path,
        parent_ids=[filtered_id],
        parameters={
            "qc_type": "picks_membrane_filter",
            "tilt_series_id": series,
            "tomogram_branch": branch,
            "status": status.value,
        },
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=report_path.stat().st_size,
    )
    return MembraneFilterResult([filtered, qc], summary, table_path, report_path, star_path)


def _write_table(
    path: Path,
    geometries: Sequence[PickGeometry],
    kept: Sequence[bool],
    voxel_size_angstrom: float,
) -> None:
    lines = [
        "\t".join(
            (
                "row",
                "x_px",
                "y_px",
                "z_px",
                "score",
                "distance_angstrom",
                "axis_angle_deg",
                "normal_angle_deg",
                "kept",
            )
        )
    ]
    for pick, keep in zip(geometries, kept, strict=True):
        lines.append(
            "\t".join(
                (
                    str(pick.row),
                    f"{pick.x_px:.2f}",
                    f"{pick.y_px:.2f}",
                    f"{pick.z_px:.2f}",
                    "" if pick.score is None else f"{pick.score:.6f}",
                    f"{pick.distance_px * voxel_size_angstrom:.1f}",
                    f"{pick.axis_angle_deg:.1f}",
                    f"{pick.normal_angle_deg:.1f}",
                    "1" if keep else "0",
                )
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
