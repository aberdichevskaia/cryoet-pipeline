from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, RetentionPolicy, StorageRole
from cryoet_pipeline.mrc_validation import validate_complete_mrc


def import_external_tomogram_for_extraction(
    *, mrc_path: Path, tilt_series_id: str, out: Path
) -> Artifact:
    """Wrap an externally-produced MRC tomogram as an ad-hoc TOMOGRAM artifact.

    Temporary bypass for testing `extract-particles` against a tomogram that
    was reconstructed outside this pipeline (no registry entry exists yet).
    Writes a scratch canonical Zarr under `out/bypass/`; not intended to
    persist alongside pipeline-produced tomograms.
    """

    info = validate_complete_mrc(mrc_path)
    if len(info.shape) != 3:
        raise ValueError(f"expected a 3D (z, y, x) tomogram, got shape {info.shape}")
    if info.pixel_spacing_angstrom is None:
        raise ValueError(f"tomogram MRC header has no pixel spacing: {mrc_path}")

    zarr_path = out / "bypass" / f"{tilt_series_id}_external_tomogram.zarr"
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"external tomogram contains nonfinite values: {mrc_path}")
    canonical = cast(
        Any,
        zarr.open(zarr_path, mode="w", shape=data.shape, dtype=np.float32),
    )
    canonical[:] = data

    return Artifact(
        id=f"{tilt_series_id}:tomogram:external-bypass",
        kind=ArtifactKind.TOMOGRAM,
        path=zarr_path,
        shape=info.shape,
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=info.pixel_spacing_angstrom,
        parameters={
            "tilt_series_id": tilt_series_id,
            "tomogram_branch": "full",
            "backend": "external-bypass",
            "source_mrc_path": str(mrc_path),
        },
        storage_role=StorageRole.SOURCE,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        size_bytes=_path_size_bytes(zarr_path),
    )


def import_external_picks_for_extraction(
    *,
    star_path: Path,
    tomogram: Artifact,
    tilt_series_id: str,
    template_box_px: int | None = None,
) -> Artifact:
    """Wrap an externally-produced pytom-match-pick STAR file as an ad-hoc
    PICKS artifact, referencing it directly with no format conversion.

    Temporary bypass for testing `extract-particles` against picks made
    outside this pipeline (no registry entry exists yet). An external STAR
    carries no pytom job, so the template box it was matched with, which fixes
    where each pick's particle centre is, is recorded when the caller states it.
    """

    if not star_path.is_file():
        raise FileNotFoundError(f"picks STAR file not found: {star_path}")
    parameters: dict[str, object] = {
        "tilt_series_id": tilt_series_id,
        "input_tomogram_id": tomogram.id,
        "backend": "external-bypass",
    }
    if template_box_px is not None:
        parameters["pick_template_box_px"] = template_box_px
    return Artifact(
        id=f"{tilt_series_id}:picks:external-bypass",
        kind=ArtifactKind.PICKS,
        path=star_path,
        parent_ids=[tomogram.id],
        parameters=parameters,
        storage_role=StorageRole.SOURCE,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        size_bytes=star_path.stat().st_size,
    )


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
