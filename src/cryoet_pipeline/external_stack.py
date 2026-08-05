from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.stack import write_tilt_angles
from cryoet_pipeline.ingest import parse_mdoc_text, with_raw_pixel_spacing
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc
from cryoet_pipeline.runtime import DevicePreference


@dataclass(frozen=True)
class ExternalStackImportResult:
    """Artifacts produced by importing one externally pre-assembled tilt stack."""

    manifest: TiltSeriesManifest
    manifest_path: Path
    artifact_registry_path: Path
    tilt_stack: Artifact
    tilt_angles: Artifact


def import_external_tilt_stack(
    *,
    mdoc_path: Path,
    stack_path: Path,
    tilt_series_id: str,
    out: Path,
    overwrite: bool = False,
) -> ExternalStackImportResult:
    """Wrap an externally pre-assembled tilt-series stack as canonical artifacts.

    Some depositions (unlike EMPIAR-10164) do not provide individually
    addressable raw per-tilt movies; instead each tilt-series is a single
    already-combined MRC stack with one already-corrected image per tilt.
    This bypasses `init` and motion correction entirely: it parses the
    tilt-series' own `.mdoc` for per-tilt metadata (angles, defocus, pixel
    spacing) and registers the given stack file directly as the canonical
    `TILT_STACK` artifact that `align-tilt-series` already expects.

    This assumes stack slice order matches the `.mdoc` `ZValue` order (the
    standard SerialEM/IMOD convention) — that ordering is not independently
    verified here, only that the slice *count* matches the tilt count.
    """

    if not mdoc_path.is_file():
        raise FileNotFoundError(f"mdoc file not found: {mdoc_path}")
    manifest = parse_mdoc_text(
        mdoc_path.read_text(),
        source_mdoc=mdoc_path,
        tilt_series_id=tilt_series_id,
        frames_dir=None,
    )

    stack_info = validate_complete_mrc(stack_path)
    if len(stack_info.shape) != 3:
        raise ValueError(
            f"expected external tilt stack shape (tilts, y, x), got {stack_info.shape}"
        )
    if stack_info.shape[0] != manifest.num_tilts:
        raise ValueError(
            f"{tilt_series_id}: external stack has {stack_info.shape[0]} images, "
            f"mdoc has {manifest.num_tilts} ZValue tilts"
        )
    if stack_info.pixel_spacing_angstrom is None:
        raise ValueError(f"external stack MRC header has no pixel spacing: {stack_path}")
    manifest = with_raw_pixel_spacing(manifest, stack_info.pixel_spacing_angstrom)

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest(out, manifest, overwrite=overwrite)

    stack_dir = out / "stacks" / tilt_series_id
    canonical_stack_path = stack_dir / f"{tilt_series_id}.st"
    if canonical_stack_path.exists() and not overwrite:
        raise FileExistsError(f"tilt stack already exists: {canonical_stack_path}")
    _write_canonical_stack(
        stack_path,
        canonical_stack_path,
        pixel_spacing_angstrom=stack_info.pixel_spacing_angstrom,
    )

    context = BackendContext(
        output_dir=out,
        device=DevicePreference.CPU,
        parameters={"overwrite": overwrite},
    )
    tilt_angles_artifact = write_tilt_angles(manifest, context)

    tilt_stack_artifact = Artifact(
        id=f"{tilt_series_id}:tilt_stack",
        kind=ArtifactKind.TILT_STACK,
        path=canonical_stack_path,
        parent_ids=[],
        shape=stack_info.shape,
        dtype="float32",
        axis_order=AxisOrder.TYX,
        pixel_spacing_angstrom=stack_info.pixel_spacing_angstrom,
        parameters={
            "backend": "external_stack_import",
            "artifact_format": "mrc",
            "tilt_series_id": tilt_series_id,
            "num_tilts": manifest.num_tilts,
            "source_stack_path": str(stack_path),
            "source_mdoc_path": str(mdoc_path),
        },
        software_versions={"mrcfile": _package_version("mrcfile")},
        storage_role=StorageRole.SOURCE,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=False,
        size_bytes=canonical_stack_path.stat().st_size,
    )

    registry_path = out / "artifacts.json"
    registry = (
        ArtifactRegistry.load(registry_path) if registry_path.exists() else ArtifactRegistry.empty()
    )
    registry.extend([tilt_angles_artifact, tilt_stack_artifact], replace=overwrite)
    registry.write(registry_path)

    return ExternalStackImportResult(
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_registry_path=registry_path,
        tilt_stack=tilt_stack_artifact,
        tilt_angles=tilt_angles_artifact,
    )


def _write_manifest(
    out: Path,
    manifest: TiltSeriesManifest,
    *,
    overwrite: bool,
) -> Path:
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{manifest.tilt_series_id}.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def _write_canonical_stack(
    source_path: Path,
    destination_path: Path,
    *,
    pixel_spacing_angstrom: float,
) -> None:
    with mrcfile.open(source_path, permissive=True) as source:
        data = np.asarray(cast(np.ndarray, source.data), dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"external tilt stack contains nonfinite values: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(destination_path, overwrite=True) as destination:
        destination.set_data(data)
        destination.voxel_size = pixel_spacing_angstrom


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
