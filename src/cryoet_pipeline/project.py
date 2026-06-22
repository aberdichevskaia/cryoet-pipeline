from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryoet_pipeline.ingest import parse_mdoc_text
from cryoet_pipeline.models import ProjectConfig, TiltSeriesManifest

EMPIAR_10164_RAW_PIXEL_SPACING_ANGSTROM = 1.35
EXPECTED_TILTS_PER_SERIES = 41
EXPECTED_SUBFRAMES_BY_SERIES = {
    "TS_01": 8,
    "TS_43": 10,
}


@dataclass(frozen=True)
class InitResult:
    project_path: Path
    manifest_paths: list[Path]


def initialize_project(config: ProjectConfig) -> InitResult:
    """Validate local input data and write project manifests."""

    manifests = [load_tilt_series_manifest(config, tilt_series_id) for tilt_series_id in config.tilt_series]
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

    project_path = config.output_dir / "project.json"
    write_json(
        project_path,
        {
            "config": config.model_dump(mode="json"),
            "tilt_series": [manifest.tilt_series_id for manifest in manifests],
            "manifests": [str(path) for path in manifest_paths],
        },
    )

    return InitResult(project_path=project_path, manifest_paths=manifest_paths)


def load_tilt_series_manifest(config: ProjectConfig, tilt_series_id: str) -> TiltSeriesManifest:
    mdoc_path = config.mdocs_dir / f"{tilt_series_id}.mrc.mdoc"
    if not mdoc_path.exists():
        raise FileNotFoundError(f"missing mdoc file for {tilt_series_id}: {mdoc_path}")

    return parse_mdoc_text(
        mdoc_path.read_text(),
        source_mdoc=mdoc_path,
        tilt_series_id=tilt_series_id,
        frames_dir=config.frames_dir,
        raw_pixel_spacing_angstrom=EMPIAR_10164_RAW_PIXEL_SPACING_ANGSTROM,
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

    missing = [image.local_frame_file for image in manifest.images if not image.local_frame_file]
    if missing:
        raise ValueError(f"{manifest.tilt_series_id}: some mdoc entries did not resolve to local files")

    missing_files = [image.local_frame_file for image in manifest.images if not image.local_frame_file.exists()]
    if missing_files:
        preview = ", ".join(str(path) for path in missing_files[:5])
        raise FileNotFoundError(
            f"{manifest.tilt_series_id}: {len(missing_files)} frame files are missing; first: {preview}"
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
