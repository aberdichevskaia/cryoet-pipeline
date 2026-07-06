from __future__ import annotations

import json
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.ingest import parse_mdoc_text, with_raw_pixel_spacing
from cryoet_pipeline.models import ProjectConfig, TiltSeriesManifest
from cryoet_pipeline.mrc_validation import validate_complete_mrc

EXPECTED_TILTS_PER_SERIES = 41
EXPECTED_SUBFRAMES_BY_SERIES = {
    "TS_01": 8,
    "TS_43": 10,
}


@dataclass(frozen=True)
class InitResult:
    project_path: Path
    manifest_paths: list[Path]
    artifact_registry_path: Path


def initialize_project(config: ProjectConfig) -> InitResult:
    """Validate local input data and write project manifests."""

    manifests = [
        load_tilt_series_manifest(config, tilt_series_id)
        for tilt_series_id in config.tilt_series
    ]
    for manifest in manifests:
        validate_tilt_series_manifest(manifest)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = config.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    for manifest in manifests:
        path = manifest_dir / f"{manifest.tilt_series_id}.json"
        write_json(path, manifest.model_dump(mode="json"))
        manifest_paths.append(path)

    artifact_registry_path = config.output_dir / "artifacts.json"
    ArtifactRegistry.empty().write(artifact_registry_path)

    project_path = config.output_dir / "project.json"
    write_json(
        project_path,
        {
            "config": config.model_dump(mode="json"),
            "tilt_series": [manifest.tilt_series_id for manifest in manifests],
            "manifests": [str(path) for path in manifest_paths],
            "artifact_registry": str(artifact_registry_path),
        },
    )

    return InitResult(
        project_path=project_path,
        manifest_paths=manifest_paths,
        artifact_registry_path=artifact_registry_path,
    )


def load_tilt_series_manifest(config: ProjectConfig, tilt_series_id: str) -> TiltSeriesManifest:
    mdoc_path = config.mdocs_dir / f"{tilt_series_id}.mrc.mdoc"
    if not mdoc_path.exists():
        raise FileNotFoundError(f"missing mdoc file for {tilt_series_id}: {mdoc_path}")

    manifest = parse_mdoc_text(
        mdoc_path.read_text(),
        source_mdoc=mdoc_path,
        tilt_series_id=tilt_series_id,
        frames_dir=config.frames_dir,
    )
    return with_raw_pixel_spacing(
        manifest,
        _raw_frame_pixel_spacing_angstrom(manifest),
    )


def validate_tilt_series_manifest(manifest: TiltSeriesManifest) -> None:
    if manifest.num_tilts != EXPECTED_TILTS_PER_SERIES:
        raise ValueError(
            f"{manifest.tilt_series_id}: expected {EXPECTED_TILTS_PER_SERIES} tilts, "
            f"found {manifest.num_tilts}"
        )

    expected_subframes = EXPECTED_SUBFRAMES_BY_SERIES.get(manifest.tilt_series_id)
    if expected_subframes is not None and manifest.num_subframes_set != {expected_subframes}:
        raise ValueError(
            f"{manifest.tilt_series_id}: expected {expected_subframes} frames per tilt, "
            f"found {sorted(manifest.num_subframes_set)}"
        )

    local_frame_files: list[Path] = []
    for image in manifest.images:
        if image.local_frame_file is None:
            raise ValueError(
                f"{manifest.tilt_series_id}: some mdoc entries did not resolve to local files"
            )
        local_frame_files.append(image.local_frame_file)

    missing_files = [path for path in local_frame_files if not path.exists()]
    if missing_files:
        preview = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"{manifest.tilt_series_id}: {len(missing_files)} frame files are missing; "
            f"first: {preview}"
        )


def _raw_frame_pixel_spacing_angstrom(manifest: TiltSeriesManifest) -> float:
    spacings: list[tuple[Path, float]] = []
    for image in manifest.images:
        path = image.local_frame_file
        if path is None:
            raise ValueError(
                f"{manifest.tilt_series_id} z={image.z_value}: "
                "missing local frame file"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"{manifest.tilt_series_id} z={image.z_value}: "
                f"frame file not found: {path}"
            )
        info = validate_complete_mrc(path)
        if info.pixel_spacing_angstrom is None:
            raise ValueError(f"raw MRC header has no pixel spacing: {path}")
        spacings.append((path, info.pixel_spacing_angstrom))

    reference_path, reference_spacing = spacings[0]
    inconsistent = [
        (path, spacing)
        for path, spacing in spacings[1:]
        if not isclose(spacing, reference_spacing, rel_tol=1e-5, abs_tol=1e-6)
    ]
    if inconsistent:
        path, spacing = inconsistent[0]
        raise ValueError(
            f"{manifest.tilt_series_id}: inconsistent raw MRC pixel spacing; "
            f"{reference_path} has {reference_spacing} angstrom, "
            f"{path} has {spacing} angstrom"
        )
    return reference_spacing


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
