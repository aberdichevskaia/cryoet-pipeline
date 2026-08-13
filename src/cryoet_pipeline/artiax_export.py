"""Export a tomogram and its picks as a pair ChimeraX/ArtiaX can open directly.

ArtiaX registers two readers for `.star`: "RELION STAR file" (RELION 3.1
conventions, `rlnCoordinateX/Y/Z` in pixels) and "RELION5 STAR file" (centered
coordinates in Angstrom). Both claim the `.star` suffix and both answer to the
nickname `star`, so the generated ChimeraX script names the format explicitly.
pytom-match-pick writes the RELION 3.1 flavour, and letting ChimeraX guess can
put every particle in the wrong place without any error.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.models import Artifact, ArtifactKind

CHIMERAX_STAR_FORMAT = "relion"
"""ArtiaX nickname for the RELION 3.1 star reader, as opposed to `relion5`."""


@dataclass(frozen=True)
class ArtiaxExport:
    """Files written for one ChimeraX/ArtiaX inspection session."""

    directory: Path
    tomogram_path: Path
    picks_path: Path
    script_path: Path


def export_picks_for_artiax(
    *,
    tomogram: Artifact,
    picks: Artifact,
    out: Path,
    overwrite: bool = False,
) -> ArtiaxExport:
    """Write a tomogram MRC, its picks STAR, and a ChimeraX script beside them."""

    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected a tomogram artifact, got {tomogram.kind}")
    if picks.kind is not ArtifactKind.PICKS:
        raise ValueError(f"expected a picks artifact, got {picks.kind}")
    if not picks.path.is_file():
        raise FileNotFoundError(f"picks file not found: {picks.path}")
    voxel_size = tomogram.pixel_spacing_angstrom
    if voxel_size is None or voxel_size <= 0.0:
        raise ValueError("tomogram voxel spacing is required so ArtiaX scales picks correctly")

    tilt_series_id = str(tomogram.parameters.get("tilt_series_id", "tomogram"))
    directory = out / "artiax" / tilt_series_id
    tomogram_path = directory / f"{tilt_series_id}.mrc"
    picks_path = directory / f"{tilt_series_id}_particles.star"
    script_path = directory / f"{tilt_series_id}_open_in_chimerax.cxc"

    existing = [path for path in (tomogram_path, picks_path, script_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"ArtiaX export already exists: {joined}")

    directory.mkdir(parents=True, exist_ok=True)
    _write_tomogram_mrc(tomogram.path, tomogram_path, voxel_size_angstrom=voxel_size)
    shutil.copyfile(picks.path, picks_path)
    script_path.write_text(_chimerax_script(tomogram_path.name, picks_path.name))

    return ArtiaxExport(
        directory=directory,
        tomogram_path=tomogram_path,
        picks_path=picks_path,
        script_path=script_path,
    )


def _chimerax_script(tomogram_name: str, picks_name: str) -> str:
    return "\n".join(
        [
            "# Open with: chimerax <this file>",
            "# Requires the ArtiaX bundle (toolshed install ArtiaX).",
            f"open {tomogram_name}",
            "# ArtiaX ships two readers for .star; name the RELION 3.1 one explicitly,",
            "# because the RELION5 reader expects centered coordinates in Angstrom.",
            f"open {picks_name} format {CHIMERAX_STAR_FORMAT}",
            "",
        ]
    )


def _write_tomogram_mrc(
    source: Path,
    destination: Path,
    *,
    voxel_size_angstrom: float,
) -> None:
    if source.is_dir():
        array = cast(Any, zarr.open(source, mode="r"))
        data = np.asarray(array[:], dtype=np.float32)
    else:
        with mrcfile.open(source, permissive=True) as mrc:
            data = np.array(mrc.data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D tomogram, got shape {data.shape}: {source}")
    with mrcfile.new(destination, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size_angstrom
