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
    template_path: Path | None = None
    segmentation_path: Path | None = None


def export_picks_for_artiax(
    *,
    tomogram: Artifact,
    picks: Artifact,
    out: Path,
    template: Artifact | None = None,
    segmentation: Artifact | None = None,
    overwrite: bool = False,
) -> ArtiaxExport:
    """Write everything one inspection session needs, into one directory.

    Points alone answer little. Attaching the template shows each pick with the
    orientation that was found for it, and a segmentation surface gives the
    picks something to sit on or miss — a tomogram slice shows one plane at a
    time and says nothing about the set as a whole. Both are optional, and both
    are converted here because ChimeraX reads MRC, not the canonical Zarr.
    """

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

    template_path = directory / f"{tilt_series_id}_template.mrc" if template else None
    segmentation_path = (
        directory / f"{tilt_series_id}_segmentation.mrc" if segmentation else None
    )

    existing = [
        path
        for path in (tomogram_path, picks_path, script_path, template_path, segmentation_path)
        if path is not None and path.exists()
    ]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"ArtiaX export already exists: {joined}")

    directory.mkdir(parents=True, exist_ok=True)
    _write_tomogram_mrc(tomogram.path, tomogram_path, voxel_size_angstrom=voxel_size)
    shutil.copyfile(picks.path, picks_path)
    if template is not None and template_path is not None:
        _write_tomogram_mrc(template.path, template_path, voxel_size_angstrom=voxel_size)
    if segmentation is not None and segmentation_path is not None:
        _write_tomogram_mrc(
            segmentation.path, segmentation_path, voxel_size_angstrom=voxel_size
        )
    script_path.write_text(
        _chimerax_script(
            tomogram_path.name,
            picks_path.name,
            voxel_size_angstrom=voxel_size,
            template_name=template_path.name if template_path else None,
            segmentation_name=segmentation_path.name if segmentation_path else None,
        )
    )

    return ArtiaxExport(
        directory=directory,
        template_path=template_path,
        segmentation_path=segmentation_path,
        tomogram_path=tomogram_path,
        picks_path=picks_path,
        script_path=script_path,
    )


def _chimerax_script(
    tomogram_name: str,
    picks_name: str,
    *,
    voxel_size_angstrom: float,
    template_name: str | None = None,
    segmentation_name: str | None = None,
) -> str:
    """Open everything at one scale, with each pick shown as the template.

    Three things have to be said explicitly or the result is quietly wrong.
    ArtiaX registers two readers for `.star` and both answer to the suffix, so
    the RELION 3.1 one is named — the RELION5 reader expects centred coordinates
    in Angstrom. ArtiaX places a particle at `coordinate * originScaleFactor`
    with the factor defaulting to one, while the coordinates are in tomogram
    voxels, so without it every pick lands at a fraction of its true position.
    And the template is inverted to match a tomogram's dark particles, so its
    surface has to be contoured at a negative level or nothing appears.

    Model numbering follows from the order below: ArtiaX is #1, its tomogram
    #1.1.1, the particle list #1.2.1, and anything opened afterwards continues
    from #2.
    """

    lines = [
        "# Open with: chimerax <this file>",
        "# Requires the ArtiaX bundle (toolshed install ArtiaX).",
        "artiax start",
        "# Through ArtiaX rather than as a plain volume, so the tomogram is under",
        "# its control and appears in the Tomograms list.",
        f"artiax open tomo {tomogram_name}",
        "# Two readers claim .star; the RELION5 one expects centred coordinates",
        "# in Angstrom, which these are not.",
        f"open {picks_name} format {CHIMERAX_STAR_FORMAT}",
        "# Coordinates are in tomogram voxels, and ArtiaX scales them by this",
        "# factor, which defaults to one.",
        f"artiax particles #1.2.1 originScaleFactor {voxel_size_angstrom:.6f}",
    ]
    next_model = 2
    if template_name is not None:
        lines += [
            "",
            "# Each pick drawn as the template at the orientation found for it,",
            "# which is what makes a wrong orientation visible. The template is",
            "# inverted to match dark particles, hence the negative level.",
            f"open {template_name}",
            f"artiax attach #{next_model} toParticleList #1.2.1",
            f"volume #{next_model} level -1 style surface",
            "# Adjust the level until the shape is recognisable.",
        ]
        next_model += 1
    if segmentation_name is not None:
        lines += [
            "",
            "# The membranes as a surface, so the picks have something to sit on",
            "# or miss. A tomogram slice shows one plane; this shows the set.",
            f"open {segmentation_name}",
            f"volume #{next_model} level 0.5 style surface transparency 0.5",
        ]
    lines.append("")
    return "\n".join(lines)



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
