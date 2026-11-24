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
    # The series whose manifests were written: the ones asked for, or every one
    # found; when adding, those of them the project had no manifest for.
    tilt_series: tuple[str, ...] = ()
    # When adding: the series asked for or found that the project has already,
    # whose manifests were left as they are.
    kept: tuple[str, ...] = ()


class ProjectExistsError(ValueError):
    """init asked to make a project where there is one, or to add to one that is not there."""


@dataclass(frozen=True)
class ProjectInputs:
    """What init read and checked, before it writes anything."""

    # The manifests init would write: of the series asked for, or of every one
    # found; when adding, of those the project has no manifest for.
    manifests: tuple[TiltSeriesManifest, ...]
    # When adding: the series asked for or found that the project has already.
    kept: tuple[str, ...] = ()


def read_inputs(config: ProjectConfig, *, add_series: bool = False) -> ProjectInputs:
    """Everything init checks, with nothing written: the project folder, the
    series found or asked for, and every movie each manifest names.

    initialize_project writes what this returns. scripts/submit_lanes.py's dry run
    calls it too, so that the dry run refuses what init would, by the same reading.
    """

    artifact_registry_path = config.output_dir / "artifacts.json"
    exists = artifact_registry_path.exists()
    if exists and not add_series:
        raise ProjectExistsError(
            f"{config.output_dir} is a project already: its registry, "
            f"{artifact_registry_path}, records what has run on its series, and init "
            "would start it empty and lose that. To add series to the project, give "
            "--add-series; for a project of their own, another --out."
        )
    if add_series and not exists:
        raise ProjectExistsError(
            f"--add-series adds series to an existing project, and {config.output_dir} "
            f"has no registry ({artifact_registry_path}); without --add-series, init "
            "makes a new project there"
        )
    if config.frames_pattern is not None:
        # A malformed pattern is refused before any folder is listed.
        compile_frame_pattern(config.frames_pattern)
    tilt_series = config.tilt_series or find_tilt_series(config)
    manifest_dir = config.output_dir / "manifests"
    kept = tuple(
        series for series in tilt_series
        if add_series and (manifest_dir / f"{series}.json").is_file()
    )
    manifests = tuple(
        load_tilt_series_manifest(config, tilt_series_id)
        for tilt_series_id in tilt_series
        if tilt_series_id not in kept
    )
    for manifest in manifests:
        validate_tilt_series_manifest(manifest)
    return ProjectInputs(manifests=manifests, kept=kept)


def describe_manifest(manifest: TiltSeriesManifest) -> str:
    """One line on what a manifest holds, as init prints it: a pattern that matched
    the wrong part of the names shows up here, as a strange angle range or tilt
    count, before any job."""

    angles = manifest.tilt_angles_deg
    frames_per_tilt = ", ".join(str(n) for n in sorted(manifest.num_subframes_set))
    return (
        f"{manifest.num_tilts} tilts, {min(angles):g} to {max(angles):g} deg, "
        f"{frames_per_tilt} frames per tilt, {manifest.raw_pixel_spacing_angstrom:g} A/px"
    )


def initialize_project(config: ProjectConfig, *, add_series: bool = False) -> InitResult:
    """Validate local input data and write project manifests.

    A new project gets an empty registry. A folder whose registry exists is a
    project with results in it, which an empty registry would lose: init refuses
    it, unless `add_series`, which writes manifests only for the series the
    project has none for and leaves the registry and every other manifest as
    they are. Jobs on the project's other series may run meanwhile, since
    nothing they read changes.
    """

    inputs = read_inputs(config, add_series=add_series)
    manifests, kept = list(inputs.manifests), inputs.kept
    added = [manifest.tilt_series_id for manifest in manifests]
    manifest_dir = config.output_dir / "manifests"
    artifact_registry_path = config.output_dir / "artifacts.json"

    project_path = config.output_dir / "project.json"
    if add_series and not added:
        return InitResult(
            project_path=project_path,
            manifest_paths=[],
            artifact_registry_path=artifact_registry_path,
            kept=kept,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: list[Path] = []
    for manifest in manifests:
        path = manifest_dir / f"{manifest.tilt_series_id}.json"
        write_json(path, manifest.model_dump(mode="json"))
        manifest_paths.append(path)

    if add_series:
        write_json(project_path, _with_added_series(project_path, manifest_dir, config, added))
    else:
        ArtifactRegistry.empty().write(artifact_registry_path)
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
        tilt_series=tuple(added),
        kept=kept,
    )


def _with_added_series(
    project_path: Path, manifest_dir: Path, config: ProjectConfig, added: list[str]
) -> dict[str, Any]:
    """project.json after an addition: every series the project has manifests for,
    its first config kept, and each addition's config and series under "added".

    A project that came from import-tilt-stack has no project.json until then.
    """

    before = json.loads(project_path.read_text()) if project_path.is_file() else {}
    paths = sorted(manifest_dir.glob("*.json"))
    return {
        **before,
        "config": before.get("config"),
        "tilt_series": [path.stem for path in paths],
        "manifests": [str(path) for path in paths],
        "artifact_registry": str(config.output_dir / "artifacts.json"),
        "added": [
            *before.get("added", []),
            {"config": config.model_dump(mode="json"), "tilt_series": added},
        ],
    }


def find_tilt_series(config: ProjectConfig) -> list[str]:
    """Every series the inputs hold: by the frames pattern, or one per mdoc."""

    if config.frames_pattern is not None:
        return sorted(match_frames(config.frames_dir, config.frames_pattern))
    assert config.mdocs_dir is not None
    if not config.mdocs_dir.is_dir():
        raise FileNotFoundError(f"mdocs directory not found: {config.mdocs_dir}")
    found: dict[str, Path] = {}
    for path in sorted(config.mdocs_dir.iterdir()):
        series = _mdoc_series_id(path.name)
        if series is None or not path.is_file() or path.name.startswith("."):
            continue
        if series in found:
            raise ValueError(f"two mdocs for {series}: {found[series].name} and {path.name}")
        found[series] = path
    if not found:
        raise ValueError(f"no *{' or *'.join(_MDOC_SUFFIXES)} files in {config.mdocs_dir}")
    return sorted(found)


# SerialEM names an mdoc after its stack, <id>.mrc.mdoc; Thermo Fisher's
# Tomography after the series, <id>.mdoc (EMPIAR-11830).
_MDOC_SUFFIXES = (".mrc.mdoc", ".mdoc")


def _mdoc_series_id(name: str) -> str | None:
    for suffix in _MDOC_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def _mdoc_path(mdocs_dir: Path, tilt_series_id: str) -> Path:
    paths = [mdocs_dir / f"{tilt_series_id}{suffix}" for suffix in _MDOC_SUFFIXES]
    existing = [path for path in paths if path.exists()]
    if len(existing) > 1:
        raise ValueError(
            f"two mdocs for {tilt_series_id}: {existing[0].name} and {existing[1].name}"
        )
    if not existing:
        raise FileNotFoundError(
            f"missing mdoc file for {tilt_series_id}: {paths[0]} or {paths[1].name}"
        )
    return existing[0]


def load_tilt_series_manifest(config: ProjectConfig, tilt_series_id: str) -> TiltSeriesManifest:
    if config.frames_pattern is not None:
        return manifest_from_frame_names(
            frames_dir=config.frames_dir,
            pattern=config.frames_pattern,
            tilt_series_id=tilt_series_id,
            raw_pixel_spacing_angstrom=config.raw_pixel_spacing_angstrom,
        )

    assert config.mdocs_dir is not None
    mdoc_path = _mdoc_path(config.mdocs_dir, tilt_series_id)

    manifest = parse_mdoc_text(
        mdoc_path.read_text(),
        source_mdoc=mdoc_path,
        tilt_series_id=tilt_series_id,
        frames_dir=config.frames_dir,
    )
    _require_local_frame_files(manifest)
    manifest = _with_eer_frame_counts(manifest)
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


def _with_eer_frame_counts(manifest: TiltSeriesManifest) -> TiltSeriesManifest:
    """The frame count of each EER movie as its file holds it.

    An mdoc counts an EER movie's frames its own way: Tomography writes
    NumSubFrames = 1 for movies of 728 EER frames (EMPIAR-11830). The file is
    what MotionCor3 sums, so its count replaces the mdoc's, which each image
    keeps in its metadata as mdoc_num_subframes.
    """

    images = []
    counts: set[int] = set()
    declared: set[int] = set()
    for image in manifest.images:
        path = image.local_frame_file
        if path is None or path.suffix.lower() != ".eer":
            images.append(image)
            continue
        frames = inspect_movie(path).shape[0]
        counts.add(frames)
        if frames == image.num_subframes:
            images.append(image)
            continue
        if image.num_subframes is not None:
            declared.add(image.num_subframes)
        images.append(
            image.model_copy(
                update={
                    "num_subframes": frames,
                    "metadata": {**image.metadata, "mdoc_num_subframes": image.num_subframes},
                }
            )
        )
    if not declared:
        return manifest
    held = (
        str(min(counts)) if len(counts) == 1 else f"{min(counts)} to {max(counts)}"
    )
    note = (
        f"EER movies: frame counts read from the files ({held} EER frames per tilt); the "
        f"mdoc's NumSubFrames ({', '.join(str(n) for n in sorted(declared))}) is kept in "
        "each image's metadata as mdoc_num_subframes"
    )
    return manifest.model_copy(update={"images": images, "notes": [*manifest.notes, note]})


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
