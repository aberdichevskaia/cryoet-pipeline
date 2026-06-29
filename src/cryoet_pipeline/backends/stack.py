from __future__ import annotations

import shutil
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr
from numpy.typing import NDArray

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext, TiltStackBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc
from cryoet_pipeline.storage import ArtifactFormat


class SimpleTiltStackBackend:
    """Assemble corrected projections into an alignment-ready tilt stack."""

    name = "simple"

    def build_stack(
        self,
        corrected_projections: Sequence[Artifact],
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Write an ordered tilt stack from corrected projection artifacts."""

        ordered_artifacts = _ordered_projection_artifacts(corrected_projections, manifest)
        projection_shape = _validate_projection_shapes(ordered_artifacts)
        stack_shape = (len(ordered_artifacts), *projection_shape)
        artifact_format = _artifact_format_parameter(context)
        output_path = _tilt_stack_path(
            context.output_dir,
            manifest,
            artifact_format=artifact_format,
        )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        _write_stack(
            output_path,
            ordered_artifacts,
            stack_shape=stack_shape,
            artifact_format=artifact_format,
            pixel_spacing_angstrom=manifest.raw_pixel_spacing_angstrom,
            overwrite=overwrite,
        )

        return Artifact(
            id=f"{manifest.tilt_series_id}:tilt_stack",
            kind=ArtifactKind.TILT_STACK,
            path=output_path,
            parent_ids=[artifact.id for artifact in ordered_artifacts],
            shape=stack_shape,
            dtype="float32",
            axis_order=AxisOrder.TYX,
            pixel_spacing_angstrom=manifest.raw_pixel_spacing_angstrom,
            parameters={
                "backend": self.name,
                "artifact_format": artifact_format.value,
                "tilt_series_id": manifest.tilt_series_id,
                "num_tilts": manifest.num_tilts,
            },
            software_versions=_software_versions(),
            storage_role=_storage_role_parameter(context),
            retention_policy=_retention_policy_parameter(context),
            can_recompute=_bool_parameter(context, "can_recompute", default=True),
            size_bytes=_path_size_bytes(output_path),
        )


def build_stack_and_register(
    backend: TiltStackBackend,
    corrected_projections: Sequence[Artifact],
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Build a tilt stack, write a `.tlt` file, and register both artifacts."""

    tilt_angles_artifact = write_tilt_angles(manifest, context)
    stack_artifact = backend.build_stack(corrected_projections, manifest, context)
    artifacts = [tilt_angles_artifact, stack_artifact]
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def write_tilt_angles(manifest: TiltSeriesManifest, context: BackendContext) -> Artifact:
    """Write manifest tilt angles as an IMOD-compatible `.tlt` file."""

    output_path = _tilt_angles_path(context.output_dir, manifest)
    overwrite = _bool_parameter(context, "overwrite", default=False)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"tilt angles file already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{tilt_angle:.6f}\n" for tilt_angle in manifest.tilt_angles_deg)
    )

    return Artifact(
        id=f"{manifest.tilt_series_id}:tilt_angles",
        kind=ArtifactKind.TILT_ANGLES,
        path=output_path,
        shape=(manifest.num_tilts,),
        dtype="float64",
        axis_order=None,
        parameters={
            "format": "imod_tlt",
            "tilt_series_id": manifest.tilt_series_id,
            "num_tilts": manifest.num_tilts,
            "source_mdoc": str(manifest.source_mdoc),
        },
        storage_role=StorageRole.EXPORT,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=_path_size_bytes(output_path),
    )


def _ordered_projection_artifacts(
    corrected_projections: Sequence[Artifact],
    manifest: TiltSeriesManifest,
) -> list[Artifact]:
    by_z_value = {
        _projection_z_value(artifact): artifact for artifact in corrected_projections
    }

    ordered: list[Artifact] = []
    for image in manifest.images:
        artifact = by_z_value.get(image.z_value)
        if artifact is None:
            raise ValueError(
                f"{manifest.tilt_series_id}: missing corrected projection for "
                f"z={image.z_value}"
            )
        ordered.append(artifact)

    return ordered


def _projection_z_value(artifact: Artifact) -> int:
    if artifact.kind is not ArtifactKind.CORRECTED_PROJECTION:
        raise ValueError(f"expected corrected projection artifact, got {artifact.kind}")

    z_value = artifact.parameters.get("z_value")
    if not isinstance(z_value, int):
        raise ValueError(f"{artifact.id}: missing integer z_value parameter")
    return z_value


def _load_projection(path: Path) -> NDArray[np.float32]:
    if path.suffix == ".zarr" or path.is_dir():
        array = cast(Any, zarr.open(path, mode="r"))
        data = np.asarray(array[:])
    else:
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data)

    if data.ndim != 2:
        raise ValueError(f"expected 2D corrected projection, got {data.shape} from {path}")

    return cast(NDArray[np.float32], data.astype(np.float32, copy=False))


def _validate_projection_shapes(
    corrected_projections: Sequence[Artifact],
) -> tuple[int, int]:
    if not corrected_projections:
        raise ValueError("at least one corrected projection is required")

    shapes = [_projection_shape(artifact.path) for artifact in corrected_projections]
    expected_shape = shapes[0]
    for shape in shapes[1:]:
        if shape != expected_shape:
            raise ValueError(
                f"corrected projections must share a shape; expected {expected_shape}, "
                f"got {shape}"
            )
    return expected_shape


def _projection_shape(path: Path) -> tuple[int, int]:
    if path.suffix == ".zarr" or path.is_dir():
        array = cast(Any, zarr.open(path, mode="r"))
        shape = tuple(int(axis_size) for axis_size in array.shape)
    else:
        shape = validate_complete_mrc(path).shape

    if len(shape) != 2:
        raise ValueError(f"expected 2D corrected projection, got {shape} from {path}")
    return shape


def _write_stack(
    path: Path,
    corrected_projections: Sequence[Artifact],
    *,
    stack_shape: tuple[int, int, int],
    artifact_format: ArtifactFormat,
    pixel_spacing_angstrom: float | None,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"tilt stack already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    if temporary_path.exists():
        _remove_existing_path(temporary_path)

    try:
        if artifact_format is ArtifactFormat.ZARR:
            _write_zarr_stack(temporary_path, corrected_projections, stack_shape)
        else:
            _write_mrc_stack(
                temporary_path,
                corrected_projections,
                stack_shape,
                pixel_spacing_angstrom=pixel_spacing_angstrom,
            )

        if path.exists():
            _remove_existing_path(path)
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            _remove_existing_path(temporary_path)
        raise


def _write_zarr_stack(
    path: Path,
    corrected_projections: Sequence[Artifact],
    stack_shape: tuple[int, int, int],
) -> None:
    chunks = (1, min(stack_shape[1], 1024), min(stack_shape[2], 1024))
    stack = cast(
        Any,
        zarr.open(
            path,
            mode="w",
            shape=stack_shape,
            chunks=chunks,
            dtype=np.float32,
        ),
    )
    for index, artifact in enumerate(corrected_projections):
        stack[index, :, :] = _load_projection(artifact.path)


def _write_mrc_stack(
    path: Path,
    corrected_projections: Sequence[Artifact],
    stack_shape: tuple[int, int, int],
    *,
    pixel_spacing_angstrom: float | None,
) -> None:
    with mrcfile.new_mmap(path, shape=stack_shape, mrc_mode=2, overwrite=True) as mrc:
        for index, artifact in enumerate(corrected_projections):
            mrc.data[index, :, :] = _load_projection(artifact.path)
        if pixel_spacing_angstrom is not None:
            mrc.voxel_size = pixel_spacing_angstrom


def _tilt_stack_path(
    output_dir: Path,
    manifest: TiltSeriesManifest,
    *,
    artifact_format: ArtifactFormat,
) -> Path:
    filename = (
        f"{manifest.tilt_series_id}.zarr"
        if artifact_format is ArtifactFormat.ZARR
        else f"{manifest.tilt_series_id}.st"
    )
    return output_dir / "stacks" / manifest.tilt_series_id / filename


def _tilt_angles_path(output_dir: Path, manifest: TiltSeriesManifest) -> Path:
    return output_dir / "stacks" / manifest.tilt_series_id / f"{manifest.tilt_series_id}.tlt"


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
