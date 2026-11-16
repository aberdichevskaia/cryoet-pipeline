"""import-tomogram: a volume reconstructed elsewhere, entered as a run of its own.

docs/entry_and_exit.md decides what an import is. In short:

- **The same questions for every volume.** Its orientation, its pixel size, the
  angles it was reconstructed with, what was done to it after. The person
  answers them; this module asks nothing else.
- **The orientation is answered from a list** (`ORIENTATIONS`). Each answer says
  which files it covers, what the import does to their voxels, and the
  measurement that showed it right. An answer with no measurement is refused,
  with what measuring it needs (`UNMEASURED`).
- **The angles enter unchanged.** IMOD's `.tlt` and RELION 5's `rlnTomoYTilt`
  are the angles the volume was reconstructed with, and every listed transform
  gives the volume the pipeline itself reconstructs with those same angles
  (docs/entry_and_exit.md, What an import is, 5).
- **Measured, then checked against what was declared.** The header's pixel size
  and axis map, which axis is thin, and the wedge in the volume's Fourier
  transform (`wedge_check`). A disagreement refuses the import before anything
  is copied.
- **An import is a run.** `run.json` names the external files by path and sha256
  and records the declarations as its content parameters, so the same files with
  the same answers are imported once. The canonical volume is a copy in the
  project, a Zarr in `(z, y, x)` as reconstruction writes it, registered with
  provenance `imported`, and its series gets a manifest that says it entered as
  a tomogram, with no frame fields.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import _VARIANT_PATTERN
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    Provenance,
    RetentionPolicy,
    SeriesEntry,
    StorageRole,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.run_lock import run_lock
from cryoet_pipeline.runs import (
    Attempt,
    RunInput,
    RunRecord,
    bytes_fingerprint,
    clear_abandoned,
    completed,
    fingerprint,
    writing,
)
from cryoet_pipeline.wedge_check import WedgeMeasurement, measure_wedge

STEP = "import-tomogram"
STEP_VERSION = 1
BACKEND = "tomogram-import"
BACKEND_VERSION = 1

Transform = Literal["none", "transpose_yz", "reverse_z"]


@dataclass(frozen=True)
class Orientation:
    """One answer on the list: the files it covers, the transform, how we know."""

    name: str
    volume: str
    transform: Transform
    how_we_know: str


ORIENTATIONS: dict[str, Orientation] = {
    orientation.name: orientation
    for orientation in (
        Orientation(
            "pipeline",
            "the pipeline's own canonical volume, written as MRC with z as sections",
            "none",
            "ours",
        ),
        Orientation(
            "imod-tilt",
            "IMOD tilt's output with Y as sections (the .rec reconstruct-tomogram keeps)",
            "transpose_yz",
            "what reconstruct-tomogram does with it",
        ),
        Orientation(
            "imod-parallel",
            "IMOD tilt -PARALLEL, or clip flipyz of tilt's output",
            "none",
            "measured 15 October 2026 (IMOD 5.1.12)",
        ),
        Orientation(
            "imod-standard",
            "IMOD's standard orientation: clip rotx, trimvol -rx, tilt -RotateBy90",
            "reverse_z",
            "measured 15 October 2026 (IMOD 5.1.12); an etomo reconstruction of "
            "EMPIAR-11538 TS_0002 matches it",
        ),
        Orientation(
            "relion5",
            "RELION 5's own tomogram reconstruction",
            "reverse_z",
            "measured 9 November 2026 on EMPIAR-10164 TS_01 "
            "(scripts/relion_geometry_check.py: NCC 0.68, the other seven reversals 0.03 "
            "at most)",
        ),
    )
}

# Orientations people ask for that no measurement has placed yet.
UNMEASURED: dict[str, str] = {
    "aretomo": "AreTomo",
    "warp": "Warp",
    "cz-portal": "the CZ Imaging Institute's data portal",
}

PROCESSING_STEPS = ("ctf-corrected", "denoised", "wedge-filled")


class ImportRefused(ValueError):
    """An import that a declaration, a measurement or the project refuses."""


@dataclass(frozen=True)
class TiltAngles:
    angles: list[float]
    source: Literal["imod-tlt", "relion5-star"]


@dataclass(frozen=True)
class VolumeHeader:
    shape: tuple[int, int, int]
    """As the MRC stores it: sections, rows, columns."""
    dtype: str
    voxel_size: tuple[float, float, float]
    """x, y, z in angstrom; 0 where the header leaves it unset."""
    axis_map: tuple[int, int, int]
    """mapc, mapr, maps."""


@dataclass
class ImportResult:
    folder: Path
    tomogram: Artifact
    manifest_path: Path
    wedge: dict[str, Any]
    already_imported: bool
    warnings: list[str] = field(default_factory=list)


def orientation(name: str) -> Orientation:
    """The listed answer, or a refusal saying what would put it on the list."""

    if name in ORIENTATIONS:
        return ORIENTATIONS[name]
    listed = ", ".join(ORIENTATIONS)
    if name in UNMEASURED:
        raise ImportRefused(
            f"{name!r} is not on the list: no measurement shows how volumes from "
            f"{UNMEASURED[name]} relate to the pipeline's. To put it there, reconstruct a "
            "tilt series the pipeline has (EMPIAR-10164 TS_01, say) with it and compare "
            "the volume with the pipeline's over the eight axis reversals, as "
            "scripts/relion_geometry_check.py does for RELION 5. A volume known to be in "
            f"one of these orientations can be imported with it: {listed}"
        )
    raise ImportRefused(f"unknown orientation {name!r}; the list holds {listed}")


def parse_processing(text: str) -> tuple[str, ...]:
    """`raw`, or a comma-separated set of `PROCESSING_STEPS`, in a fixed order."""

    words = [word.strip() for word in text.split(",") if word.strip()]
    if words == ["raw"]:
        return ()
    unknown = sorted(set(words) - set(PROCESSING_STEPS))
    if not words or unknown or "raw" in words:
        raise ImportRefused(
            f"--processing takes 'raw' (nothing done after the reconstruction) or a "
            f"comma-separated set of {', '.join(PROCESSING_STEPS)}; got {text!r}"
        )
    return tuple(step for step in PROCESSING_STEPS if step in words)


def read_tilt_angles(path: Path) -> TiltAngles:
    """The angles a volume was reconstructed with, in the order of its views.

    IMOD's `.tlt` (one number a line, or any whitespace between them), or RELION
    5's tilt-series STAR, from its `rlnTomoYTilt` column.
    """

    if not path.is_file():
        raise FileNotFoundError(f"tilt-angle file not found: {path}")
    source: Literal["imod-tlt", "relion5-star"]
    if path.suffix.lower() == ".star":
        columns, rows = read_star_loop(path)
        if "rlnTomoYTilt" not in columns:
            raise ImportRefused(
                f"{path} has no rlnTomoYTilt column: give RELION 5's tilt-series STAR "
                "(one per tomogram, under tilt_series/), not tomograms.star"
            )
        column = columns.index("rlnTomoYTilt")
        words = [row[column] for row in rows]
        source = "relion5-star"
    else:
        words = path.read_text().split()
        source = "imod-tlt"
    try:
        angles = [float(word) for word in words]
    except ValueError as exc:
        raise ImportRefused(f"{path}: a tilt angle is not a number ({exc})") from exc
    if len(angles) < 2:
        raise ImportRefused(f"{path}: a tilt series needs at least two angles")
    bad = [angle for angle in angles if not math.isfinite(angle) or not -90 < angle < 90]
    if bad:
        raise ImportRefused(f"{path}: tilt angles must lie inside (-90, 90) degrees, got {bad}")
    return TiltAngles(angles=angles, source=source)


def inspect_volume(path: Path) -> VolumeHeader:
    info = validate_complete_mrc(path)
    if len(info.shape) != 3:
        raise ImportRefused(f"{path} is not a volume: shape {info.shape}")
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        header = mrc.header
        axis_map = (int(header.mapc), int(header.mapr), int(header.maps))
        voxel = (float(mrc.voxel_size.x), float(mrc.voxel_size.y), float(mrc.voxel_size.z))
    return VolumeHeader(
        shape=cast(tuple[int, int, int], tuple(int(size) for size in info.shape)),
        dtype=info.dtype,
        voxel_size=voxel,
        axis_map=axis_map,
    )


def pixel_size(header: VolumeHeader, declared: float | None, path: Path) -> float:
    """The voxel size the import uses: the header's, checked against a declared one."""

    sizes = header.voxel_size
    if all(size == 0.0 for size in sizes):
        measured = None
    elif all(size > 0.0 for size in sizes) and max(sizes) - min(sizes) <= 1e-3 * max(sizes):
        measured = sizes[0]
    else:
        raise ImportRefused(
            f"{path}: the header's voxel size is not one number (x, y, z = "
            f"{sizes[0]:g}, {sizes[1]:g}, {sizes[2]:g} A); fix it with IMOD's alterheader "
            "and import again"
        )
    if measured is None and declared is None:
        raise ImportRefused(f"{path}: the header has no voxel size; give --pixel-size")
    disagree = measured is not None and declared is not None and (
        abs(declared - measured) > 0.005 * measured
    )
    if disagree:
        raise ImportRefused(
            f"{path}: --pixel-size {declared:g} A, but the header says {measured:g} A. If the "
            "header is wrong, fix it with IMOD's alterheader -PixelSize and import again"
        )
    return declared if declared is not None else cast(float, measured)


def canonical_shape(shape: tuple[int, int, int], transform: Transform) -> tuple[int, int, int]:
    if transform == "transpose_yz":
        return (shape[1], shape[0], shape[2])
    return shape


class _CanonicalView:
    """The source volume as the canonical `(z, y, x)` array, sliced lazily."""

    def __init__(self, data: np.ndarray, transform: Transform) -> None:
        self._data = data
        self._transform = transform
        self.shape = canonical_shape(cast(tuple[int, int, int], data.shape), transform)

    def __getitem__(self, key: tuple[slice, slice, slice]) -> np.ndarray:
        z, y, x = key
        if self._transform == "transpose_yz":
            return np.asarray(self._data[y, z, x]).transpose(1, 0, 2)
        if self._transform == "reverse_z":
            depth = self.shape[0]
            start, stop, _ = z.indices(depth)
            block = np.asarray(self._data[depth - stop : depth - start, y, x])
            return block[::-1]
        return np.asarray(self._data[z, y, x])


@dataclass(frozen=True)
class _Statistics:
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float


def _write_canonical_zarr(view: _CanonicalView, destination: Path, voxel: float,
                          orientation_name: str) -> _Statistics:
    """Copy the volume into the project as reconstruction writes its own, float32."""

    depth, height, width = view.shape
    chunks = (min(depth, 16), min(height, 64), min(width, 256))
    output = cast(Any, zarr.open(destination, mode="w", shape=view.shape, chunks=chunks,
                                 dtype=np.float32))
    output.attrs.update({"axis_order": "zyx", "voxel_spacing_angstrom": voxel,
                         "source_orientation": orientation_name})
    count = 0
    total = 0.0
    squares = 0.0
    minimum = math.inf
    maximum = -math.inf
    full = slice(None)
    if view._transform == "transpose_yz":
        # The source's sections are canonical y: read them in order.
        blocks = [(full, slice(start, min(start + chunks[1], height)))
                  for start in range(0, height, chunks[1])]
    else:
        blocks = [(slice(start, min(start + chunks[0], depth)), full)
                  for start in range(0, depth, chunks[0])]
    for z, y in blocks:
        block = np.asarray(view[z, y, full], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ImportRefused("the volume holds values that are not finite")
        output[z, y, full] = block
        wide = block.astype(np.float64, copy=False)
        count += wide.size
        total += float(wide.sum())
        squares += float(np.square(wide).sum())
        minimum = min(minimum, float(wide.min()))
        maximum = max(maximum, float(wide.max()))
    mean = total / count
    return _Statistics(minimum, maximum, mean, math.sqrt(max(squares / count - mean * mean, 0.0)))


def _write_central_slices(volume: Any, folder: Path, voxel: float) -> dict[str, str]:
    depth, height, width = (int(size) for size in volume.shape)
    slices = {
        "xy": np.asarray(volume[depth // 2, :, :], dtype=np.float32),
        "xz": np.asarray(volume[:, height // 2, :], dtype=np.float32),
        "yz": np.asarray(volume[:, :, width // 2], dtype=np.float32),
    }
    names = {}
    for plane, data in slices.items():
        name = f"central_{plane}.mrc"
        with mrcfile.new(folder / name, overwrite=True) as output:
            output.set_data(data)
            output.voxel_size = voxel
        names[plane] = name
    return names


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def tomogram_manifest(
    tilt_series_id: str, angles: Sequence[float], note: str
) -> TiltSeriesManifest:
    """A series entered as a tomogram: its views' angles, and nothing of frames."""

    return TiltSeriesManifest(
        tilt_series_id=tilt_series_id,
        images=[
            TiltImage(z_value=index, tilt_angle_deg=angle) for index, angle in enumerate(angles)
        ],
        entry=SeriesEntry.TOMOGRAM,
        notes=[note],
    )


def _check_manifest(path: Path, angles: Sequence[float]) -> TiltSeriesManifest | None:
    """The series' manifest if it can take this import as it is, None if there is none."""

    if not path.exists():
        return None
    manifest = TiltSeriesManifest.model_validate_json(path.read_text())
    if manifest.entry is not SeriesEntry.TOMOGRAM:
        raise ImportRefused(
            f"{manifest.tilt_series_id} is in this project already, entered as "
            f"{manifest.entry.value} ({path}). Attaching an imported tomogram to a series "
            "the project has is not built yet (docs/entry_and_exit.md, What an import is, "
            "6); import it into another project"
        )
    if len(manifest.images) != len(angles) or any(
        abs(image.tilt_angle_deg - angle) > 1e-6
        for image, angle in zip(manifest.images, angles, strict=False)
    ):
        raise ImportRefused(
            f"{manifest.tilt_series_id} was imported before with other tilt angles ({path}); "
            "two volumes of one series must come from the same views"
        )
    return manifest


def _judge_wedge(measurement: WedgeMeasurement, processing: Sequence[str],
                 overrule: bool) -> list[str]:
    """Refuse an import its wedge contradicts; the warnings of one it lets through."""

    filled = "wedge-filled" in processing
    problem: str | None = None
    if measurement.verdict == "no wedge" and not filled:
        problem = (
            f"{measurement.reason}, but the volume is not declared wedge-filled. A volume "
            "IsoNet or DeepDeWedge filled is declared with --processing wedge-filled"
        )
    elif filled and measurement.verdict != "no wedge":
        problem = (
            "the volume is declared wedge-filled, but its transform shows a wedge: "
            f"{measurement.reason}"
        )
    elif measurement.verdict == "mirrored":
        problem = (
            f"{measurement.reason}. The orientation answered is likely the wrong one of a "
            "mirror pair: imod-parallel and imod-standard, or pipeline and relion5"
        )
    warnings = []
    if measurement.verdict == "inconclusive":
        warnings.append(f"the wedge check could not tell the volume's side: {measurement.reason}")
    if problem is not None:
        if not overrule:
            raise ImportRefused(
                f"the wedge check refuses this import: {problem}. If the check itself is "
                "wrong here, --overrule-wedge-check imports it and records that you did"
            )
        warnings.append(f"wedge check overruled: {problem}")
    return warnings


def import_tomogram(
    *,
    volume_path: Path,
    tilt_angles_path: Path,
    orientation_name: str,
    processing: str,
    tilt_series_id: str,
    out: Path,
    declared_pixel_size: float | None = None,
    variant: str | None = None,
    overrule_wedge_check: bool = False,
) -> ImportResult:
    """Enter a volume reconstructed elsewhere as this project's tomogram of a series."""

    if variant is not None and not _VARIANT_PATTERN.fullmatch(variant):
        raise ImportRefused(
            f"invalid variant {variant!r}; use letters, digits, dot, dash, or underscore, "
            "starting with a letter or digit"
        )
    chosen = orientation(orientation_name)
    steps = parse_processing(processing)
    angles = read_tilt_angles(tilt_angles_path)
    header = inspect_volume(volume_path)
    if header.axis_map != (1, 2, 3):
        raise ImportRefused(
            f"{volume_path}: the header's axis order is {header.axis_map}, not columns X, "
            "rows Y, sections Z (1, 2, 3), which every orientation on the list assumes"
        )
    if header.dtype not in {"float32", "float16", "int8", "int16", "uint16"}:
        raise ImportRefused(f"{volume_path}: data of type {header.dtype} is not a density")
    voxel = pixel_size(header, declared_pixel_size, volume_path)
    shape = canonical_shape(header.shape, chosen.transform)
    if not shape[0] < min(shape[1], shape[2]):
        raise ImportRefused(
            f"{volume_path}: as {chosen.name}, the volume's axes come out (z, y, x) = {shape}, "
            "and z, the thickness, is not the shortest. tilt's own output, Y as sections, "
            "is imod-tilt; a volume with z as sections is one of the others"
        )

    manifest_path = out / "manifests" / f"{tilt_series_id}.json"
    existing_manifest = _check_manifest(manifest_path, angles.angles)
    label = f"_{variant}" if variant else ""
    folder = out / "imports" / tilt_series_id / f"tomogram{label}"
    artifact_id = f"{tilt_series_id}:tomogram:imported" + (f":{variant}" if variant else "")

    with mrcfile.mmap(volume_path, mode="r", permissive=True) as mrc:
        view = _CanonicalView(mrc.data, chosen.transform)
        wedge = measure_wedge(view, angles.angles)
    warnings = _judge_wedge(wedge, steps, overrule_wedge_check)

    inputs = [
        RunInput(slot="volume", path=str(volume_path.resolve()),
                 fingerprint=bytes_fingerprint(volume_path)),
        RunInput(slot="tilt_angles", path=str(tilt_angles_path.resolve()),
                 fingerprint=bytes_fingerprint(tilt_angles_path)),
    ]
    content: dict[str, Any] = {
        "tilt_series_id": tilt_series_id,
        "orientation": chosen.name,
        "pixel_size_angstrom": voxel,
        "pixel_size_source": "declared" if declared_pixel_size is not None else "header",
        "processing": list(steps),
        "tilt_angles_source": angles.source,
        "hand": "the reconstruction's, carried by the orientation",
    }
    if overrule_wedge_check:
        content["wedge_check_overruled"] = True
    run_fingerprint = fingerprint(STEP, STEP_VERSION, BACKEND, BACKEND_VERSION, content, inputs)

    registry_path = out / "artifacts.json"
    registry = (
        ArtifactRegistry.load(registry_path) if registry_path.exists() else ArtifactRegistry.empty()
    )
    if artifact_id in registry.artifact_ids:
        known = registry.get(artifact_id)
        if known.parameters.get("run_fingerprint") != run_fingerprint:
            raise ImportRefused(
                f"{artifact_id} is in the registry already, from other files or other "
                "answers. Give --variant to keep both"
            )

    with run_lock(folder.with_name(folder.name + ".lock")):
        record = completed(folder)
        already = record is not None
        if record is not None:
            if record.fingerprint != run_fingerprint:
                raise ImportRefused(
                    f"{folder} holds an import of other files or other answers "
                    f"(fingerprint {record.fingerprint[:12]}). Give --variant to keep both"
                )
        elif folder.exists():
            raise ImportRefused(f"{folder} exists without a completed run.json; look at it first")
        else:
            clear_abandoned(folder)
            with writing(folder, step=STEP, step_version=STEP_VERSION, backend=BACKEND,
                         backend_version=BACKEND_VERSION, content_params=content,
                         inputs=inputs) as attempt:
                _write_import(attempt.folder, attempt, volume_path, header, chosen, voxel,
                              tilt_series_id, angles, wedge, warnings)
            record = completed(folder)
            if record is None:
                raise RuntimeError(f"{folder} has no completed run.json after the import")

    tomogram = _tomogram_artifact(
        record, folder, artifact_id=artifact_id, variant=variant, tilt_series_id=tilt_series_id,
        chosen=chosen, steps=steps, angles=angles, voxel=voxel, volume_path=volume_path,
        wedge=wedge,
    )
    qc = Artifact(
        id=f"{tilt_series_id}:qc:tomogram_import" + (f":{variant}" if variant else ""),
        variant=variant,
        kind=ArtifactKind.QC,
        path=folder / "import_qc.json",
        parent_ids=[tomogram.id],
        parameters={"qc_type": "tomogram_import", "tilt_series_id": tilt_series_id,
                    "wedge_verdict": wedge.verdict},
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        provenance=Provenance.IMPORTED,
        size_bytes=_path_size_bytes(folder / "import_qc.json"),
    )
    if artifact_id not in registry.artifact_ids:
        registry.extend([tomogram, qc])
        registry.commit(registry_path)

    if existing_manifest is None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = tomogram_manifest(
            tilt_series_id, angles.angles,
            f"entered as a tomogram by import-tomogram ({chosen.name}) from {volume_path}",
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")

    return ImportResult(folder=folder, tomogram=tomogram, manifest_path=manifest_path,
                        wedge=wedge.record(), already_imported=already, warnings=warnings)


def _write_import(
    folder: Path,
    attempt: Attempt,
    volume_path: Path,
    header: VolumeHeader,
    chosen: Orientation,
    voxel: float,
    tilt_series_id: str,
    angles: TiltAngles,
    wedge: WedgeMeasurement,
    warnings: list[str],
) -> None:
    with mrcfile.mmap(volume_path, mode="r", permissive=True) as mrc:
        view = _CanonicalView(mrc.data, chosen.transform)
        statistics = _write_canonical_zarr(view, folder / f"{tilt_series_id}.zarr", voxel,
                                           chosen.name)
    canonical = cast(Any, zarr.open(folder / f"{tilt_series_id}.zarr", mode="r"))
    slices = _write_central_slices(canonical, folder, voxel)
    (folder / f"{tilt_series_id}.tlt").write_text("".join(f"{a:.6f}\n" for a in angles.angles))
    report = {
        "tilt_series_id": tilt_series_id,
        "orientation": chosen.name,
        "orientation_how_we_know": chosen.how_we_know,
        "source_shape_sections_rows_columns": list(header.shape),
        "source_dtype": header.dtype,
        "source_voxel_size_xyz_angstrom": list(header.voxel_size),
        "source_axis_map": list(header.axis_map),
        "canonical_shape_zyx": list(canonical.shape),
        "voxel_spacing_angstrom": voxel,
        "minimum": statistics.minimum,
        "maximum": statistics.maximum,
        "mean": statistics.mean,
        "standard_deviation": statistics.standard_deviation,
        "tilt_angles_source": angles.source,
        "tilt_range_deg": [min(angles.angles), max(angles.angles)],
        "wedge": wedge.record(),
        "central_slices": slices,
        "warnings": warnings,
    }
    (folder / "import_qc.json").write_text(json.dumps(report, indent=2) + "\n")
    attempt.add(ArtifactKind.TOMOGRAM, f"{tilt_series_id}.zarr",
                shape=tuple(canonical.shape), dtype="float32", axis_order=AxisOrder.ZYX,
                pixel_spacing_angstrom=voxel, provenance=Provenance.IMPORTED)
    attempt.add(ArtifactKind.QC, "import_qc.json", provenance=Provenance.IMPORTED)
    attempt.metrics.update({
        "wedge_fitted_min_deg": wedge.fitted_min_deg,
        "wedge_fitted_max_deg": wedge.fitted_max_deg,
        "wedge_contrast_log10": wedge.fitted_contrast,
        "mean": statistics.mean,
        "standard_deviation": statistics.standard_deviation,
    })


def _tomogram_artifact(
    record: RunRecord,
    folder: Path,
    *,
    artifact_id: str,
    variant: str | None,
    tilt_series_id: str,
    chosen: Orientation,
    steps: Sequence[str],
    angles: TiltAngles,
    voxel: float,
    volume_path: Path,
    wedge: WedgeMeasurement,
) -> Artifact:
    zarr_path = folder / f"{tilt_series_id}.zarr"
    canonical = cast(Any, zarr.open(zarr_path, mode="r"))
    count = len(angles.angles)
    return Artifact(
        id=artifact_id,
        variant=variant,
        kind=ArtifactKind.TOMOGRAM,
        path=zarr_path,
        shape=tuple(int(size) for size in canonical.shape),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=voxel,
        parameters={
            "backend": BACKEND,
            "tilt_series_id": tilt_series_id,
            "tomogram_branch": "full",
            "orientation": chosen.name,
            "processing": list(steps),
            "ctf_corrected": "ctf-corrected" in steps,
            "source_path": str(volume_path.resolve()),
            "tilt_angles_path": str(folder / f"{tilt_series_id}.tlt"),
            "tilt_angles_source": angles.source,
            "included_z_values": list(range(count)),
            "excluded_z_values": [],
            "wedge_verdict": wedge.verdict,
            "run_path": str(folder),
            "run_fingerprint": record.fingerprint,
        },
        software_versions={
            "mrcfile": _package_version("mrcfile"),
            "zarr": _package_version("zarr"),
        },
        storage_role=StorageRole.SOURCE,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        provenance=Provenance.IMPORTED,
        size_bytes=_path_size_bytes(zarr_path) + (folder / f"{tilt_series_id}.tlt").stat().st_size,
    )


__all__ = [
    "BACKEND",
    "ImportRefused",
    "ImportResult",
    "ORIENTATIONS",
    "Orientation",
    "PROCESSING_STEPS",
    "STEP",
    "UNMEASURED",
    "import_tomogram",
    "inspect_volume",
    "orientation",
    "parse_processing",
    "pixel_size",
    "read_tilt_angles",
    "tomogram_manifest",
]
