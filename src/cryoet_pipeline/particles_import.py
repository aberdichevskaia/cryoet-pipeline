"""import-particles: a Dynamo data folder and its table, entered as particle stacks.

docs/entry_and_exit.md (What an import is, 8). Particles cropped elsewhere enter
as the project's own: per series, a `PARTICLE_STACK` of the voxels as they are
(Zarr, one chunk per particle) with the pipeline's particle STAR, so that the MRA
reads them from the registry (docs/mra_from_registry.md) as it reads extracted
ones, with its usual records, and knows them by the bytes of their files rather
than by the table alone.

What a row means is Dynamo's own, read from its columns (dynamo_columninfo.m):

- the pose and shift (tdrot, tilt, narot; dx, dy, dz) go through
  `dynamo_to_native`, the conversion the port's MRA matches Dynamo with, so the
  registry gives the MRA back the table it was given;
- the wedge is the table's single-axis ytilt (ftype 1, columns 14 and 15) of each
  series, held by the stack in the pipeline's sign: negated, as `dynamo_ytilt`
  negates it on the way to Dynamo (measured 2026-10-19). There is no tomogram
  here to read it from, so the stack carries it (`tilt_range_deg`);
- the pixel size is column 36 (apix);
- the position (columns 24-26) is kept as the pipeline writes it for Dynamo, p -
  0.5, for the record only: these particles are in no tomogram of the project.

What the import measures, and refuses on: the files' shape (one even cubic box),
every row's file, one pixel size, one wedge per series, and the wedge in the
particles themselves. Boxes cut from one tomogram and never turned all hold its
wedge in the same place, so the mean of their power spectra shows it, and the
side of an asymmetric range (`wedge_check.measure_particles_wedge`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import dynamo_to_native
from cryoet_pipeline.backends.protocols import _VARIANT_PATTERN
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    Provenance,
    RetentionPolicy,
    SeriesEntry,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import CROP_POSE_CONVENTION, ParticleRecord, write_particle_star
from cryoet_pipeline.picks_import import _dynamo_number
from cryoet_pipeline.reports.dynamo_project import read_em
from cryoet_pipeline.rotations import matrix_to_relion_euler
from cryoet_pipeline.run_lock import run_lock
from cryoet_pipeline.runs import (
    RunInput,
    bytes_fingerprint,
    clear_abandoned,
    completed,
    files_fingerprint,
    fingerprint,
    writing,
)
from cryoet_pipeline.tomogram_import import ImportRefused
from cryoet_pipeline.wedge_check import WedgeMeasurement, measure_particles_wedge

STEP = "import-particles"
STEP_VERSION = 1
BACKEND = "particles-import"
BACKEND_VERSION = 1
WEDGE_SAMPLE = 300
"""Particles whose power spectra are averaged for the wedge check, at most."""

# Dynamo's columns, 1-based as dynamo_columninfo.m numbers them.
TAG, DX, DY, DZ, TDROT, TILT, NAROT, CC = 1, 4, 5, 6, 7, 8, 9, 10
FTYPE, YMIN, YMAX, TOMO, CLASS, X, Y, Z, APIX = 13, 14, 15, 20, 22, 24, 25, 26, 36
_PARTICLE_FILE = re.compile(r"particle_(\d+)\.(em|mrc)$")


@dataclass
class ParticlesImportResult:
    tilt_series_id: str
    folder: Path
    stack: Artifact
    wedge: dict[str, Any]
    already_imported: bool
    warnings: list[str] = field(default_factory=list)


def _table(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"table not found: {path}")
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ImportRefused(f"{path} holds no rows")
    width = min(len(row) for row in rows)
    if width < Z:
        raise ImportRefused(f"{path}: a Dynamo table has at least {Z} columns")
    wanted = [column for column in (TAG, DX, DY, DZ, TDROT, TILT, NAROT, CC, FTYPE, YMIN,
                                    YMAX, TOMO, CLASS, X, Y, Z, APIX) if column <= width]
    table = np.full((len(rows), APIX + 1), np.nan)
    for row_index, row in enumerate(rows):
        for column in wanted:
            table[row_index, column] = _dynamo_number(row[column - 1], column, path)
    return table


def _particle_files(folder: Path) -> dict[int, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"data folder not found: {folder}")
    files: dict[int, Path] = {}
    for path in folder.iterdir():
        match = _PARTICLE_FILE.match(path.name)
        if match:
            tag = int(match.group(1))
            if tag in files:
                raise ImportRefused(f"{folder} holds two files of particle {tag}")
            files[tag] = path
    if not files:
        raise ImportRefused(f"{folder} holds no particle_<tag>.em or .mrc files")
    return files


def _read_particle(path: Path) -> np.ndarray:
    if path.suffix == ".em":
        return read_em(path)
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data, dtype=np.float32)


def _pixel_size(table: np.ndarray, declared: float | None, path: Path) -> float:
    column = table[:, APIX]
    known = column[np.isfinite(column) & (column > 0)]
    measured = None
    if len(known):
        if np.ptp(known) > 1e-6 * known.max():
            raise ImportRefused(f"{path}: column 36 (apix) holds several pixel sizes")
        measured = float(known[0])
    if measured is None and declared is None:
        raise ImportRefused(f"{path}: column 36 (apix) holds no pixel size; give --pixel-size")
    if measured is not None and declared is not None and (
        abs(declared - measured) > 0.005 * measured
    ):
        raise ImportRefused(f"{path}: --pixel-size {declared:g} A, but column 36 says {measured:g}")
    return declared if declared is not None else cast(float, measured)


def _wedge(rows: np.ndarray, series: str, path: Path) -> tuple[float, float]:
    """The series' range in the pipeline's sign: Dynamo's ytilt negated."""

    if not np.all(rows[:, FTYPE] == 1):
        types = sorted({float(value) for value in rows[:, FTYPE]})
        raise ImportRefused(f"{path}: {series}'s rows have wedge types {types}; only "
                            "single-axis wedges (ftype 1) are modelled")
    lows, highs = rows[:, YMIN], rows[:, YMAX]
    if np.ptp(lows) > 1e-6 or np.ptp(highs) > 1e-6:
        raise ImportRefused(f"{path}: {series}'s rows carry more than one wedge")
    low, high = -float(highs[0]), -float(lows[0])
    if not low < 0 < high:
        raise ImportRefused(f"{path}: {series}'s wedge {low:g}..{high:g} does not span zero")
    return low, high


def _records(rows: np.ndarray, box: int) -> list[ParticleRecord]:
    records = []
    for index, row in enumerate(rows):
        matrix, shift = dynamo_to_native(row[[TDROT, TILT, NAROT]], row[[DX, DY, DZ]], box)
        rot, tilt, psi = matrix_to_relion_euler(matrix)
        klass = int(row[CLASS]) if np.isfinite(row[CLASS]) and row[CLASS] >= 1 else 1
        records.append(ParticleRecord(
            index=index,
            # As the pipeline writes them for Dynamo, p - 0.5; nothing reads them here.
            x_px=float(row[X]) - 0.5, y_px=float(row[Y]) - 0.5, z_px=float(row[Z]) - 0.5,
            rot_deg=rot, tilt_deg=tilt, psi_deg=psi,
            shift_x_px=float(shift[0]), shift_y_px=float(shift[1]), shift_z_px=float(shift[2]),
            score=float(row[CC]) if np.isfinite(row[CC]) else None,
            class_number=klass,
            half_set=1 + index % 2,
        ))
    return records


def _judge(measurement: WedgeMeasurement, overrule: bool) -> list[str]:
    problem = None
    if measurement.verdict == "no wedge":
        problem = f"the particles show no wedge: {measurement.reason}"
    elif measurement.verdict == "mirrored":
        problem = (f"{measurement.reason}. The table's ytilt has the other sign from the "
                   "particles' wedge")
    warnings = []
    if measurement.verdict == "inconclusive":
        warnings.append(f"the wedge check could not tell the side: {measurement.reason}")
    if problem is not None:
        if not overrule:
            raise ImportRefused(f"the wedge check refuses this import: {problem}. If the check "
                                "is wrong here, --overrule-wedge-check imports them and records "
                                "that you did")
        warnings.append(f"wedge check overruled: {problem}")
    return warnings


def import_particles(
    *,
    folder: Path,
    table_path: Path,
    out: Path,
    series: Mapping[int, str],
    declared_pixel_size: float | None = None,
    variant: str | None = None,
    overrule_wedge_check: bool = False,
) -> list[ParticlesImportResult]:
    """Enter a Dynamo data folder and its table as one particle stack per series.

    `series` maps the table's tomogram numbers (column 20) to the project's series.
    """

    if variant is not None and not _VARIANT_PATTERN.fullmatch(variant):
        raise ImportRefused(f"invalid variant {variant!r}")
    if not series:
        raise ImportRefused("name the series of each tomogram number, as --series 1=TS_01")
    table = _table(table_path)
    files = _particle_files(folder)
    numbers = sorted({int(round(value)) for value in table[:, TOMO]})
    unmapped = [number for number in numbers if number not in series]
    if unmapped:
        raise ImportRefused(f"{table_path}: rows of tomograms {unmapped} (column 20) have no "
                            "series; name each with --series <number>=<series>")
    tags = [int(round(value)) for value in table[:, TAG]]
    if len(set(tags)) != len(tags):
        raise ImportRefused(f"{table_path}: a tag (column 1) is used twice")
    missing = [tag for tag in tags if tag not in files]
    if missing:
        raise ImportRefused(f"{folder} has no file for {len(missing)} of the table's particles, "
                            f"tags {missing[:5]}")
    pixel = _pixel_size(table, declared_pixel_size, table_path)
    shapes = {_read_particle(files[tag]).shape for tag in tags[:1]}
    box = int(next(iter(shapes))[0])
    registry_path = out / "artifacts.json"
    registry = (ArtifactRegistry.load(registry_path) if registry_path.exists()
                else ArtifactRegistry.empty())

    results = []
    for number in numbers:
        name = series[number]
        rows = table[np.rint(table[:, TOMO]) == number]
        series_tags = [int(round(value)) for value in rows[:, TAG]]
        low, high = _wedge(rows, name, table_path)
        particles = [_read_particle(files[tag]) for tag in series_tags]
        wrong = [tag for tag, particle in zip(series_tags, particles, strict=True)
                 if particle.shape != (box, box, box)]
        if wrong:
            raise ImportRefused(f"particles {wrong[:5]} are not boxes of {box}, as the first is")
        if box % 2:
            raise ImportRefused(f"the boxes are {box} a side; the MRA works on even boxes")
        step = max(1, len(particles) // WEDGE_SAMPLE)
        wedge = measure_particles_wedge(particles[::step], [low, high])
        warnings = _judge(wedge, overrule_wedge_check)
        results.append(_write(registry, registry_path, out, name, number, rows, particles,
                              [files[tag] for tag in series_tags], table_path, folder, box,
                              pixel, (low, high), wedge, warnings, variant,
                              overrule_wedge_check))
    registry.commit(registry_path)
    return results


def _write(registry: ArtifactRegistry, registry_path: Path, out: Path, name: str, number: int,
           rows: np.ndarray, particles: list[np.ndarray], paths: list[Path], table_path: Path,
           folder: Path, box: int, pixel: float, tilt_range: tuple[float, float],
           wedge: WedgeMeasurement, warnings: list[str], variant: str | None,
           overruled: bool) -> ParticlesImportResult:
    content: dict[str, Any] = {"tilt_series_id": name, "dynamo_tomogram": number,
                               "pixel_size_angstrom": pixel, "tilt_range_deg": list(tilt_range),
                               "box": box}
    if overruled:
        content["wedge_check_overruled"] = True
    inputs = [
        RunInput(slot="table", path=str(table_path.resolve()),
                 fingerprint=bytes_fingerprint(table_path)),
        # The particles are known by their voxels' bytes, in tag order.
        RunInput(slot="particles", path=str(folder.resolve()),
                 fingerprint=files_fingerprint(paths)),
    ]
    run_fingerprint = fingerprint(STEP, STEP_VERSION, BACKEND, BACKEND_VERSION, content, inputs)
    label = f"_{variant}" if variant else ""
    run_folder = out / "imports" / name / f"particles{label}"
    artifact_id = f"{name}:particle_stack:imported" + (f":{variant}" if variant else "")
    if artifact_id in registry.artifact_ids and (
        registry.get(artifact_id).parameters.get("run_fingerprint") != run_fingerprint
    ):
        raise ImportRefused(f"{artifact_id} is in the registry already, from other files or "
                            "answers. Give --variant to keep both")
    zarr_name, star_name = f"{name}_particles.zarr", f"{name}_particles.star"
    with run_lock(run_folder.with_name(run_folder.name + ".lock")):
        record = completed(run_folder)
        already = record is not None
        if record is not None:
            if record.fingerprint != run_fingerprint:
                raise ImportRefused(f"{run_folder} holds an import of other files or answers. "
                                    "Give --variant to keep both")
        elif run_folder.exists():
            raise ImportRefused(f"{run_folder} exists without a completed run.json")
        else:
            clear_abandoned(run_folder)
            with writing(run_folder, step=STEP, step_version=STEP_VERSION, backend=BACKEND,
                         backend_version=BACKEND_VERSION, content_params=content,
                         inputs=inputs) as attempt:
                stack = cast(Any, zarr.open(attempt.folder / zarr_name, mode="w",
                                            shape=(len(particles), box, box, box),
                                            chunks=(1, box, box, box), dtype=np.float32))
                stack.attrs.update({"axis_order": AxisOrder.NZYX.value,
                                    "extraction_pose_convention": CROP_POSE_CONVENTION})
                for index, particle in enumerate(particles):
                    stack[index] = particle
                write_particle_star(attempt.folder / star_name, _records(rows, box))
                (attempt.folder / "particles_qc.json").write_text(json.dumps({
                    "tilt_series_id": name, "dynamo_tomogram": number, "particles": len(rows),
                    "box": box, "pixel_size_angstrom": pixel, "tilt_range_deg": list(tilt_range),
                    "dynamo_ytilt": [-tilt_range[1], -tilt_range[0]], "wedge": wedge.record(),
                    "warnings": warnings,
                }, indent=2) + "\n")
                attempt.add(ArtifactKind.PARTICLE_STACK, zarr_name,
                            shape=(len(particles), box, box, box), dtype="float32",
                            axis_order=AxisOrder.NZYX, pixel_spacing_angstrom=pixel,
                            provenance=Provenance.IMPORTED)
                attempt.add(ArtifactKind.QC, "particles_qc.json", provenance=Provenance.IMPORTED)
                attempt.metrics.update({"particles": float(len(rows)),
                                        "wedge_fitted_min_deg": wedge.fitted_min_deg,
                                        "wedge_fitted_max_deg": wedge.fitted_max_deg})
            record = completed(run_folder)
            if record is None:
                raise RuntimeError(f"{run_folder} has no completed run.json after the import")

    stack_artifact = Artifact(
        id=artifact_id,
        variant=variant,
        kind=ArtifactKind.PARTICLE_STACK,
        path=run_folder / zarr_name,
        shape=(len(particles), box, box, box),
        dtype="float32",
        axis_order=AxisOrder.NZYX,
        pixel_spacing_angstrom=pixel,
        parameters={
            "backend": BACKEND,
            "tilt_series_id": name,
            "metadata_path": str(run_folder / star_name),
            "box_size_px": box,
            "tilt_range_deg": list(tilt_range),
            "dynamo_tomogram": number,
            "source_table": str(table_path.resolve()),
            "source_folder": str(folder.resolve()),
            "extraction_pose_convention": CROP_POSE_CONVENTION,
            "wedge_verdict": wedge.verdict,
            "run_path": str(run_folder),
            "run_fingerprint": record.fingerprint,
        },
        storage_role=StorageRole.SOURCE,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        provenance=Provenance.IMPORTED,
        size_bytes=sum(path.stat().st_size for path in (run_folder / zarr_name).rglob("*")
                       if path.is_file()),
    )
    if artifact_id not in registry.artifact_ids:
        registry.add(stack_artifact)
    manifest_path = out / "manifests" / f"{name}.json"
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = TiltSeriesManifest(
            tilt_series_id=name, images=[], entry=SeriesEntry.PARTICLES,
            notes=[f"entered as particles by import-particles from {folder} and {table_path}"],
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return ParticlesImportResult(tilt_series_id=name, folder=run_folder, stack=stack_artifact,
                                 wedge=wedge.record(), already_imported=already,
                                 warnings=warnings)


__all__ = ["BACKEND", "ParticlesImportResult", "STEP", "import_particles"]
