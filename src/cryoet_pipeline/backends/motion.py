from __future__ import annotations

import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr
from numpy.typing import NDArray

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext, MotionCorrectionBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    RetentionPolicy,
    StorageRole,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc
from cryoet_pipeline.storage import ArtifactFormat


class AverageMotionCorrectionBackend:
    """Baseline motion correction that averages frames in each multiframe MRC."""

    name = "average"

    def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
        """Average every local multiframe movie listed in the tilt-series manifest."""

        _validate_manifest_movies(manifest)
        return [self.correct_image(image, manifest, context) for image in manifest.images]

    def correct_image(
        self,
        image: TiltImage,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Average one multiframe movie and write a corrected projection MRC."""

        if image.local_frame_file is None:
            raise ValueError(
                f"{manifest.tilt_series_id} z={image.z_value}: missing local frame file"
            )

        projection = _average_movie_frames(image.local_frame_file)
        artifact_format = _artifact_format_parameter(context)
        output_path = _corrected_projection_path(
            context.output_dir,
            manifest,
            image,
            artifact_format=artifact_format,
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        _write_projection(
            output_path,
            projection,
            artifact_format=artifact_format,
            pixel_spacing_angstrom=manifest.raw_pixel_spacing_angstrom,
            overwrite=overwrite,
        )

        return Artifact(
            id=f"{manifest.tilt_series_id}:corrected_projection:{image.z_value:03d}",
            kind=ArtifactKind.CORRECTED_PROJECTION,
            path=output_path,
            shape=tuple(int(axis_size) for axis_size in projection.shape),
            dtype=str(projection.dtype),
            axis_order=AxisOrder.YX,
            pixel_spacing_angstrom=manifest.raw_pixel_spacing_angstrom,
            binning=image.binning,
            parameters={
                "backend": self.name,
                "method": "frame_mean",
                "artifact_format": artifact_format.value,
                "source_frame_file": str(image.local_frame_file),
                "z_value": image.z_value,
                "tilt_angle_deg": image.tilt_angle_deg,
                "num_subframes": image.num_subframes,
            },
            software_versions=_software_versions(),
            storage_role=_storage_role_parameter(context),
            retention_policy=_retention_policy_parameter(context),
            can_recompute=_bool_parameter(context, "can_recompute", default=True),
            size_bytes=_path_size_bytes(output_path),
        )


def correct_and_register(
    backend: MotionCorrectionBackend,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Run motion correction and add the returned artifacts to the registry."""

    artifacts = backend.correct(manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def _average_movie_frames(path: Path) -> NDArray[np.float32]:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
        if data.ndim != 3:
            raise ValueError(
                f"expected multiframe MRC with shape (frames, y, x), got {data.shape}"
            )

        projection = np.zeros(data.shape[1:], dtype=np.float32)
        for frame in data:
            np.add(projection, frame, out=projection, casting="unsafe")
        projection /= data.shape[0]
    return cast(NDArray[np.float32], projection)


def _validate_manifest_movies(manifest: TiltSeriesManifest) -> None:
    issues: list[str] = []
    for image in manifest.images:
        path = image.local_frame_file
        if path is None:
            issues.append(
                f"{manifest.tilt_series_id} z={image.z_value}: missing local frame file"
            )
            continue
        if not path.is_file():
            issues.append(f"{manifest.tilt_series_id} z={image.z_value}: file not found: {path}")
            continue

        try:
            info = validate_complete_mrc(path)
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
            continue

        if len(info.shape) != 3:
            issues.append(
                f"{path}: expected multiframe MRC with shape (frames, y, x), "
                f"got {info.shape}"
            )
        elif image.num_subframes is not None and info.shape[0] != image.num_subframes:
            issues.append(
                f"{path}: manifest declares {image.num_subframes} frames, "
                f"MRC header declares {info.shape[0]}"
            )

    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"input movie validation failed:\n{details}")


def _write_projection(
    path: Path,
    projection: NDArray[np.float32],
    *,
    artifact_format: ArtifactFormat,
    pixel_spacing_angstrom: float | None,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"corrected projection already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_format is ArtifactFormat.ZARR:
        if path.exists():
            _remove_existing_path(path)
        zarr.save(path, cast(Any, projection))
        return

    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(projection)
        if pixel_spacing_angstrom is not None:
            mrc.voxel_size = pixel_spacing_angstrom


def _corrected_projection_path(
    output_dir: Path,
    manifest: TiltSeriesManifest,
    image: TiltImage,
    *,
    artifact_format: ArtifactFormat,
) -> Path:
    extension = "zarr" if artifact_format is ArtifactFormat.ZARR else "mrc"
    return (
        output_dir
        / "corrected"
        / manifest.tilt_series_id
        / f"{manifest.tilt_series_id}_{image.z_value:03d}_avg.{extension}"
    )


def _artifact_format_parameter(context: BackendContext) -> ArtifactFormat:
    value = context.parameters.get("artifact_format", ArtifactFormat.MRC)
    try:
        return ArtifactFormat(str(value))
    except ValueError as exc:
        allowed = ", ".join(artifact_format.value for artifact_format in ArtifactFormat)
        raise ValueError(
            f"context parameter 'artifact_format' must be one of: {allowed}"
        ) from exc


def _storage_role_parameter(context: BackendContext) -> StorageRole:
    value = context.parameters.get("storage_role", StorageRole.CACHE)
    try:
        return StorageRole(str(value))
    except ValueError as exc:
        allowed = ", ".join(role.value for role in StorageRole)
        raise ValueError(f"context parameter 'storage_role' must be one of: {allowed}") from exc


def _retention_policy_parameter(context: BackendContext) -> RetentionPolicy:
    value = context.parameters.get("retention_policy", RetentionPolicy.RECOMPUTE)
    try:
        return RetentionPolicy(str(value))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in RetentionPolicy)
        raise ValueError(
            f"context parameter 'retention_policy' must be one of: {allowed}"
        ) from exc


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _remove_existing_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _software_versions() -> dict[str, str]:
    try:
        mrcfile_version = version("mrcfile")
    except PackageNotFoundError:
        mrcfile_version = "unknown"
    try:
        zarr_version = version("zarr")
    except PackageNotFoundError:
        zarr_version = "unknown"
    return {"mrcfile": mrcfile_version, "zarr": zarr_version}
