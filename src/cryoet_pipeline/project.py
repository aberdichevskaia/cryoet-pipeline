from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.ingest import parse_mdoc_text, with_raw_pixel_spacing
from cryoet_pipeline.ingest.frame_names import (
    compile_frame_pattern,
    manifest_from_frame_names,
    match_frames,
)
from cryoet_pipeline.models import ProjectConfig, TiltSeriesManifest
from cryoet_pipeline.movies import common_pixel_spacing, inspect_movie


@dataclass(frozen=True)
class InitResult:
    project_path: Path
    manifest_paths: list[Path]
    artifact_registry_path: Path
    # The series included: the ones asked for, or every one found.
    tilt_series: tuple[str, ...] = ()


def initialize_project(config: ProjectConfig) -> InitResult:
    """Validate local input data and write project manifests."""

    if config.frames_pattern is not None:
        # A malformed pattern is refused before any folder is listed.
        compile_frame_pattern(config.frames_pattern)
    tilt_series = config.tilt_series or find_tilt_series(config)
    manifests = [
        load_tilt_series_manifest(config, tilt_series_id)
        for tilt_series_id in tilt_series
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
        tilt_series=tuple(tilt_series),
    )


def find_tilt_series(config: ProjectConfig) -> list[str]:
    """Every series the inputs hold: by the frames pattern, or one per mdoc."""

    if config.frames_pattern is not None:
        return sorted(match_frames(config.frames_dir, config.frames_pattern))
    assert config.mdocs_dir is not None
    if not config.mdocs_dir.is_dir():
        raise FileNotFoundError(f"mdocs directory not found: {config.mdocs_dir}")
    # The same name load_tilt_series_manifest looks for.
    suffix = ".mrc.mdoc"
    found = sorted(
        path.name[: -len(suffix)]
        for path in config.mdocs_dir.iterdir()
        if path.is_file() and path.name.endswith(suffix) and not path.name.startswith(".")
    )
    if not found:
        raise ValueError(f"no *{suffix} files in {config.mdocs_dir}")
    return found


def load_tilt_series_manifest(config: ProjectConfig, tilt_series_id: str) -> TiltSeriesManifest:
    if config.frames_pattern is not None:
        return manifest_from_frame_names(
            frames_dir=config.frames_dir,
            pattern=config.frames_pattern,
            tilt_series_id=tilt_series_id,
            raw_pixel_spacing_angstrom=config.raw_pixel_spacing_angstrom,
        )

    assert config.mdocs_dir is not None
    mdoc_path = config.mdocs_dir / f"{tilt_series_id}.mrc.mdoc"
    if not mdoc_path.exists():
        raise FileNotFoundError(f"missing mdoc file for {tilt_series_id}: {mdoc_path}")

    manifest = parse_mdoc_text(
        mdoc_path.read_text(),
        source_mdoc=mdoc_path,
        tilt_series_id=tilt_series_id,
        frames_dir=config.frames_dir,
    )
    _require_local_frame_files(manifest)
    pixel_spacing = config.raw_pixel_spacing_angstrom
    if pixel_spacing is None:
        pixel_spacing = common_pixel_spacing(
            tilt_series_id,
            [image.local_frame_file for image in manifest.images
             if image.local_frame_file is not None],
        )
    return with_raw_pixel_spacing(manifest, pixel_spacing)


def validate_tilt_series_manifest(manifest: TiltSeriesManifest) -> None:
    """Every movie is there and has the frame count the metadata declares.

    Nothing about the dataset is expected beyond that: the number of tilts and
    of frames per tilt are the data's own.
    """

    _require_local_frame_files(manifest)
    mismatched = []
    for image in manifest.images:
        assert image.local_frame_file is not None
        frames = inspect_movie(image.local_frame_file).shape[0]
        if frames != image.num_subframes:
            mismatched.append(
                f"{image.local_frame_file.name}: {frames} frames, "
                f"metadata declares {image.num_subframes}"
            )
    if mismatched:
        raise ValueError(
            f"{manifest.tilt_series_id}: {len(mismatched)} movies have another frame "
            f"count than declared; first: {mismatched[0]}"
        )


def _require_local_frame_files(manifest: TiltSeriesManifest) -> None:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
