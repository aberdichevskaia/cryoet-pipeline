"""export-tomogram: a tomogram of the project as MRC, in an orientation from the list.

docs/entry_and_exit.md, Exits: a tomogram leaves for napari, ChimeraX or another
picker in an orientation other programs know, and says in a sidecar what it is.
It is import-tomogram's twin. One list answers both (`tomogram_import.ORIENTATIONS`):
the export undoes the transform the import applies, so a volume exported in an
orientation and imported with that answer is the volume it was, voxel for voxel.
An orientation the list does not hold is refused here as there. The angles leave
unchanged, as they enter: for every orientation on the list they are the angles
the volume was reconstructed with (What an import is, 5).

What is written, into a folder of its own that is never written over:

- `<series>.mrc`: the volume, float32, with its sections, rows and columns as the
  orientation lays them out and the voxel size in the header;
- `<series>.tlt`: the angles it was reconstructed with, unchanged;
- `export.json`: the orientation and how we know it, what was done to the volume,
  the tomogram and the imported or attested artifacts in its lineage, and the
  import-tomogram command that brings the files back;
- `run.json`, last, as for every run.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder
from cryoet_pipeline.runs import RunInput, bytes_fingerprint, canonical_json, writing
from cryoet_pipeline.tomogram_import import (
    ORIENTATIONS,
    UNMEASURED,
    Orientation,
    Transform,
    read_tilt_angles,
)

STEP = "export-tomogram"
STEP_VERSION = 1
BACKEND = "tomogram-export"
BACKEND_VERSION = 1
SIDECAR = "export.json"

# What each transform of the list does to the pipeline's volume on the way out.
UNDONE: dict[Transform, str] = {
    "none": "as the pipeline keeps it: z as sections",
    "reverse_z": "the pipeline's volume reversed in z",
    "transpose_yz": "y as sections, z as rows: the pipeline's y and z exchanged",
}


class ExportRefused(ValueError):
    """An export the list, the registry or the destination refuses."""


@dataclass(frozen=True)
class TomogramExport:
    folder: Path
    volume: Path
    tilt_angles: Path
    sidecar: dict[str, Any]


def listed(name: str) -> Orientation:
    """The listed orientation, or a refusal saying the list does not hold it."""

    if name in ORIENTATIONS:
        return ORIENTATIONS[name]
    holds = ", ".join(ORIENTATIONS)
    if name in UNMEASURED:
        raise ExportRefused(
            f"{name!r} is not on the list: no measurement shows how {UNMEASURED[name]} lays "
            f"out a volume, so nothing here can write one its way. The list holds {holds}"
        )
    raise ExportRefused(f"unknown orientation {name!r}; the list holds {holds}")


def tilt_angles_of(registry: ArtifactRegistry, tomogram: Artifact,
                   tilt_series_id: str) -> Path | None:
    """The angles this tomogram was reconstructed from, as a file, or None.

    The tomogram's own record first: reconstruction and import write the angles
    they used. Then the tilt file of an aligned stack it was made from, for
    tomograms that recorded none; an ancestor stack may hold more views than the
    tomogram, since correct-ctf can leave out views CTFFIND did not fit.
    """

    own = tomogram.parameters.get("tilt_angles_path")
    if isinstance(own, (str, Path)) and Path(own).is_file():
        return Path(own)
    for stack in registry.by_kind(ArtifactKind.ALIGNED_TILT_STACK):
        if not (stack.covers(tilt_series_id) and registry.descends_from(tomogram.id, stack.id)):
            continue
        value = stack.parameters.get("tilt_file_path")
        if isinstance(value, (str, Path)) and Path(value).is_file():
            return Path(value)
    return None


def processing_of(registry: ArtifactRegistry, tomogram: Artifact) -> list[str]:
    """What was done to the volume after its reconstruction, in import-tomogram's words."""

    declared = tomogram.parameters.get("processing")
    if isinstance(declared, list):
        return [str(step) for step in declared]
    sources = [tomogram, *registry.ancestors_of_kind(tomogram.id, ArtifactKind.TOMOGRAM)]
    steps = []
    if any(source.parameters.get("ctf_corrected") is True for source in sources):
        steps.append("ctf-corrected")
    if tomogram.kind is ArtifactKind.DENOISED_TOMOGRAM:
        steps.append("denoised")
    return steps


def _series(tomogram: Artifact) -> str:
    if len(tomogram.tilt_series_ids) != 1:
        raise ExportRefused(
            f"{tomogram.id} is about {len(tomogram.tilt_series_ids)} tilt series, not one"
        )
    return tomogram.tilt_series_ids[0]


def _record_fingerprint(artifact: Artifact) -> str:
    return hashlib.sha256(canonical_json(artifact.model_dump(mode="json")).encode()).hexdigest()


def oriented_shape(shape_zyx: tuple[int, int, int], transform: Transform) -> tuple[int, int, int]:
    """The MRC's sections, rows and columns for the pipeline's (z, y, x) volume."""

    if transform == "transpose_yz":
        return (shape_zyx[1], shape_zyx[0], shape_zyx[2])
    return shape_zyx


def write_oriented_mrc(volume: Any, path: Path, transform: Transform, voxel: float) -> None:
    """The pipeline's (z, y, x) volume as an MRC laid out as `transform` undoes it.

    The inverse of `tomogram_import._CanonicalView`, block by block so that a large
    volume never sits in memory whole.
    """

    depth, height, width = (int(size) for size in volume.shape)
    shape = oriented_shape((depth, height, width), transform)
    with mrcfile.new_mmap(path, shape=shape, mrc_mode=2) as mrc:
        mrc.voxel_size = voxel
        data = mrc.data
        if transform == "transpose_yz":
            step = 64
            for start in range(0, height, step):
                stop = min(start + step, height)
                block = np.asarray(volume[:, start:stop, :], dtype=np.float32)
                data[start:stop] = block.transpose(1, 0, 2)
        else:
            step = 16
            for start in range(0, depth, step):
                stop = min(start + step, depth)
                block = np.asarray(volume[start:stop], dtype=np.float32)
                if transform == "reverse_z":
                    data[depth - stop : depth - start] = block[::-1]
                else:
                    data[start:stop] = block
        mrc.update_header_stats()


def import_command(*, volume: Path, tilt_angles: Path, orientation: str,
                   processing: Sequence[str], tilt_series_id: str) -> str:
    """The import-tomogram call that enters the exported files again, as they are."""

    words = [
        "cryoet", "import-tomogram", "--volume", str(volume), "--tilt-angles",
        str(tilt_angles), "--orientation", orientation, "--processing",
        ",".join(processing) or "raw", "--tilt-series-id", tilt_series_id,
        "--out", "<project>",
    ]
    return " ".join(shlex.quote(word) if word != "<project>" else word for word in words)


def export_tomogram(*, project: Path, tomogram_id: str, orientation_name: str,
                    out: Path) -> TomogramExport:
    """Write the project's tomogram `tomogram_id` into `out` in a listed orientation."""

    chosen = listed(orientation_name)
    project = Path(project)
    out = Path(out)
    if out.exists():
        raise ExportRefused(f"{out} is there already; an export is never written over")
    registry = ArtifactRegistry.load(project / "artifacts.json")
    try:
        tomogram = registry.get(tomogram_id)
    except KeyError as exc:
        raise ExportRefused(f"{project} has no artifact {tomogram_id!r}") from exc
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ExportRefused(f"{tomogram_id} is a {tomogram.kind.value}, not a tomogram")
    if tomogram.axis_order not in {None, AxisOrder.ZYX}:
        raise ExportRefused(f"{tomogram_id} is stored {tomogram.axis_order}, not as (z, y, x)")
    series = _series(tomogram)
    angles_path = tilt_angles_of(registry, tomogram, series)
    if angles_path is None:
        raise ExportRefused(
            f"{tomogram_id} records no angles it was reconstructed with, and none of the "
            "stacks it was made from has a tilt file. A tomogram does not leave without "
            "them: the side of an asymmetric wedge depends on them"
        )
    angles = read_tilt_angles(angles_path).angles
    voxel = tomogram.pixel_spacing_angstrom
    if voxel is None or not voxel > 0:
        raise ExportRefused(f"{tomogram_id} records no pixel size")
    source = cast(Any, zarr.open(tomogram.path, mode="r"))
    if len(source.shape) != 3:
        raise ExportRefused(f"{tomogram.path} is not a volume: shape {source.shape}")
    steps = processing_of(registry, tomogram)
    roots = registry.not_derived_lineage(tomogram.id)

    content = {"tomogram_id": tomogram.id, "orientation": chosen.name}
    inputs = [
        RunInput(slot="tomogram", path=str(Path(tomogram.path).resolve()),
                 fingerprint=_record_fingerprint(tomogram)),
        RunInput(slot="tilt_angles", path=str(angles_path.resolve()),
                 fingerprint=bytes_fingerprint(angles_path)),
    ]
    volume_name, angles_name = f"{series}.mrc", f"{series}.tlt"
    shape = oriented_shape(cast(tuple[int, int, int], tuple(source.shape)), chosen.transform)
    sidecar: dict[str, Any] = {
        "tomogram_id": tomogram.id,
        "tilt_series_id": series,
        "volume": volume_name,
        "tilt_angles": angles_name,
        "orientation": chosen.name,
        "orientation_volume": chosen.volume,
        "orientation_how_we_know": chosen.how_we_know,
        "from_the_pipelines_volume": UNDONE[chosen.transform],
        "shape_sections_rows_columns": list(shape),
        "pixel_size_angstrom": voxel,
        "angles": "the angles the volume was reconstructed with, unchanged; for every "
                  "orientation on the list they are read as they are",
        "tilt_range_deg": [min(angles), max(angles)],
        "processing": steps or ["raw"],
        "provenance": tomogram.provenance.value,
        "not_derived_lineage": [
            {"id": root.id, "kind": root.kind.value, "provenance": root.provenance.value}
            for root in roots
        ],
        "hand": "the reconstruction's; the orientation carries it unchanged",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with writing(out, step=STEP, step_version=STEP_VERSION, backend=BACKEND,
                 backend_version=BACKEND_VERSION, content_params=content,
                 inputs=inputs) as attempt:
        folder = attempt.folder
        write_oriented_mrc(source, folder / volume_name, chosen.transform, voxel)
        (folder / angles_name).write_text("".join(f"{angle:.6f}\n" for angle in angles))
        sidecar["import_command"] = import_command(
            volume=out / volume_name, tilt_angles=out / angles_name, orientation=chosen.name,
            processing=steps, tilt_series_id=series,
        )
        (folder / SIDECAR).write_text(json.dumps(sidecar, indent=2) + "\n")
        attempt.add(ArtifactKind.TOMOGRAM, volume_name, shape=shape, dtype="float32",
                    pixel_spacing_angstrom=voxel, provenance=tomogram.provenance)
        attempt.add(ArtifactKind.TILT_ANGLES, angles_name)
        attempt.add(ArtifactKind.DATASET_EXPORT, SIDECAR)
    return TomogramExport(folder=out, volume=out / volume_name, tilt_angles=out / angles_name,
                          sidecar=sidecar)


__all__ = [
    "ExportRefused",
    "SIDECAR",
    "TomogramExport",
    "export_tomogram",
    "import_command",
    "listed",
    "oriented_shape",
    "processing_of",
    "tilt_angles_of",
    "write_oriented_mrc",
]
