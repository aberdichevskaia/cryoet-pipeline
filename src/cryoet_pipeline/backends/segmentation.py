from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
from cryoet_pipeline.backends.protocols import BackendContext, SegmentationBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
    TomogramBranch,
)


class MemBrainSegSegmentationBackend:
    """Run an external MemBrain-seg model on a reconstructed tomogram."""

    name = "membrain-seg"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def segment(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        supporting_artifacts: Sequence[Artifact] = (),
    ) -> Artifact:
        """Return a canonical membrane segmentation Zarr artifact."""

        del supporting_artifacts
        shape = _validate_input_tomogram(tomogram, manifest)
        voxel_spacing = _voxel_spacing(tomogram)
        branch = _tomogram_branch(tomogram)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _segmentation_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            branch=branch,
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        executable = _resolve_membrain_executable(context)
        checkpoint = _resolve_membrain_checkpoint(context)

        paths.segmentation_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-membrain-seg-",
            dir=paths.segmentation_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_input = temporary_root / f"{manifest.tilt_series_id}.mrc"
            temporary_output_dir = temporary_root / "predictions"
            temporary_zarr = temporary_root / f"{manifest.tilt_series_id}_segmentation.zarr"
            temporary_report = temporary_root / "membrain_seg_qc.json"

            temporary_output_dir.mkdir()
            _write_zarr_to_mrc(
                tomogram.path,
                temporary_input,
                voxel_spacing_angstrom=voxel_spacing,
            )
            command = _membrain_segment_command(
                executable,
                context,
                tomogram_path=temporary_input,
                checkpoint_path=checkpoint,
                output_dir=temporary_output_dir,
                voxel_spacing_angstrom=voxel_spacing,
            )
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=os.environ.copy(),
            )
            _write_command_log(paths.log, command, result)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"MemBrain-seg failed with exit code {result.returncode}: "
                    f"{detail}; see {paths.log}"
                )

            segmentation_mrc = _find_membrain_segmentation_mrc(
                temporary_output_dir,
                expected_name=_optional_string_parameter(context, "membrain_output_name"),
            )
            statistics = _write_mrc_to_canonical_zarr(
                segmentation_mrc,
                temporary_zarr,
                expected_shape=shape,
                voxel_spacing_angstrom=voxel_spacing,
            )
            segmentation_id = f"{manifest.tilt_series_id}:segmentation:{branch.value}:membrain-seg"
            report = {
                "schema_version": 1,
                "tilt_series_id": manifest.tilt_series_id,
                "backend": self.name,
                "input_tomogram_id": tomogram.id,
                "segmentation_id": segmentation_id,
                "tomogram_branch": branch.value,
                "shape": shape,
                "voxel_spacing_angstrom": voxel_spacing,
                "dtype": statistics.dtype,
                "minimum": statistics.minimum,
                "maximum": statistics.maximum,
                "mean": statistics.mean,
                "standard_deviation": statistics.standard_deviation,
                "foreground_voxel_count": statistics.foreground_voxel_count,
                "foreground_fraction": statistics.foreground_fraction,
                "finite": True,
                "command_log_path": str(paths.log),
                "status": QcStatus.PASS.value,
            }
            temporary_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n"
            )

            _replace_output(temporary_zarr, paths.segmentation_zarr)
            _replace_output(temporary_report, paths.report)

        return Artifact(
            id=f"{manifest.tilt_series_id}:segmentation:{branch.value}:membrain-seg",
            kind=ArtifactKind.SEGMENTATION,
            path=paths.segmentation_zarr,
            parent_ids=[tomogram.id],
            shape=shape,
            dtype=statistics.dtype,
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=voxel_spacing,
            binning=tomogram.binning,
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "tomogram_branch": branch.value,
                "input_tomogram_id": tomogram.id,
                "segmentation_method": "membrain-seg",
                "membrain_checkpoint_path": str(checkpoint),
                "membrain_output_mrc_name": segmentation_mrc.name,
                "command_log_path": str(paths.log),
                "qc_path": str(paths.report),
            },
            software_versions={
                "membrain-seg": "external",
                "zarr": _package_version("zarr"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(paths.segmentation_zarr) + paths.report.stat().st_size,
        )


def segment_and_register(
    backend: SegmentationBackend,
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Segment a tomogram and register segmentation plus QC artifacts."""

    segmentation = backend.segment(tomogram, manifest, context)
    qc_path = segmentation.parameters.get("qc_path")
    if not isinstance(qc_path, str):
        raise ValueError("segmentation backend must record qc_path")
    qc = Artifact(
        id=f"{manifest.tilt_series_id}:qc:segmentation:{segmentation.parameters['backend']}",
        kind=ArtifactKind.QC,
        path=Path(qc_path),
        parent_ids=[segmentation.id],
        parameters={
            "qc_type": "tomogram_segmentation",
            "backend": segmentation.parameters["backend"],
            "tilt_series_id": manifest.tilt_series_id,
            "tomogram_branch": segmentation.parameters.get("tomogram_branch", "full"),
            "status": QcStatus.PASS.value,
        },
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=Path(qc_path).stat().st_size,
    )
    artifacts = [segmentation, qc]
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _SegmentationPaths:
    def __init__(
        self,
        *,
        segmentation_dir: Path,
        qc_dir: Path,
        segmentation_zarr: Path,
        report: Path,
        log: Path,
    ) -> None:
        self.segmentation_dir = segmentation_dir
        self.qc_dir = qc_dir
        self.segmentation_zarr = segmentation_zarr
        self.report = report
        self.log = log

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.segmentation_zarr, self.report, self.log)


@dataclass(frozen=True)
class _SegmentationStatistics:
    dtype: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    foreground_voxel_count: int
    foreground_fraction: float


def _segmentation_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    branch: TomogramBranch,
) -> _SegmentationPaths:
    segmentation_dir = output_root / "segmentations" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "segmentation"
    prefix = f"{manifest.tilt_series_id}_{branch.value}_{backend_name}"
    return _SegmentationPaths(
        segmentation_dir=segmentation_dir,
        qc_dir=qc_dir,
        segmentation_zarr=segmentation_dir / f"{prefix}.zarr",
        report=qc_dir / f"{prefix}_qc.json",
        log=segmentation_dir / f"{prefix}.log",
    )


def _validate_input_tomogram(
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[int, int, int]:
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if tomogram.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError(
            f"tomogram is for {tomogram.parameters.get('tilt_series_id')}, "
            f"expected {manifest.tilt_series_id}"
        )
    if tomogram.axis_order is not AxisOrder.ZYX:
        raise ValueError("segmentation requires a canonical ZYX tomogram")
    if not tomogram.path.is_dir():
        raise FileNotFoundError(f"canonical tomogram Zarr not found: {tomogram.path}")
    zarr_array = cast(Any, zarr.open(tomogram.path, mode="r"))
    shape = tuple(int(axis_size) for axis_size in zarr_array.shape)
    if len(shape) != 3:
        raise ValueError(f"expected tomogram shape (z, y, x), got {shape}")
    if tomogram.shape is not None and tuple(tomogram.shape) != shape:
        raise ValueError(f"tomogram artifact shape {tomogram.shape} does not match {shape}")
    return shape


def _voxel_spacing(tomogram: Artifact) -> float:
    if tomogram.pixel_spacing_angstrom is None:
        raise ValueError("tomogram voxel spacing is required for MemBrain-seg")
    if not math.isfinite(tomogram.pixel_spacing_angstrom) or tomogram.pixel_spacing_angstrom <= 0:
        raise ValueError("tomogram voxel spacing must be finite and positive")
    return tomogram.pixel_spacing_angstrom


def _tomogram_branch(tomogram: Artifact) -> TomogramBranch:
    value = tomogram.parameters.get("tomogram_branch", TomogramBranch.FULL.value)
    return TomogramBranch(value)


def _resolve_membrain_executable(context: BackendContext) -> Path:
    configured = context.parameters.get("membrain_executable")
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError("context parameter 'membrain_executable' must be a path string")
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"MemBrain-seg executable not found: {candidate}")
        return candidate.resolve()
    resolved = shutil.which("membrain")
    if resolved is not None:
        return Path(resolved)
    raise FileNotFoundError(
        "MemBrain-seg executable not found; pass --membrain-executable "
        "or add membrain to PATH"
    )


def _resolve_membrain_checkpoint(context: BackendContext) -> Path:
    configured = context.parameters.get("membrain_model")
    if not isinstance(configured, (str, Path)):
        raise TypeError("context parameter 'membrain_model' must be a path string")
    path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"MemBrain-seg checkpoint not found: {path}")
    return path.resolve()


def _membrain_segment_command(
    executable: Path,
    context: BackendContext,
    *,
    tomogram_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    voxel_spacing_angstrom: float,
) -> list[str]:
    command = [
        str(executable),
        "segment",
        "--tomogram-path",
        str(tomogram_path),
        "--ckpt-path",
        str(checkpoint_path),
        "--out-folder",
        str(output_dir),
    ]
    if _bool_parameter(context, "membrain_rescale_patches", default=True):
        command.append("--rescale-patches")
        output_pixel_size = _positive_float_parameter(
            context,
            "membrain_out_pixel_size",
            default=10.0,
        )
        command.extend(["--in-pixel-size", f"{voxel_spacing_angstrom:.6f}"])
        command.extend(["--out-pixel-size", f"{output_pixel_size:.6f}"])
    else:
        command.append("--no-rescale-patches")
    _extend_boolean_flag(
        command,
        context,
        key="membrain_store_probabilities",
        default=False,
        true_flag="--store-probabilities",
        false_flag="--no-store-probabilities",
    )
    _extend_boolean_flag(
        command,
        context,
        key="membrain_store_connected_components",
        default=False,
        true_flag="--store-connected-components",
        false_flag="--no-store-connected-components",
    )
    connected_component_threshold = _optional_positive_int_parameter(
        context,
        "membrain_connected_component_threshold",
    )
    if connected_component_threshold is not None:
        command.extend(["--connected-component-thres", str(connected_component_threshold)])
    _extend_boolean_flag(
        command,
        context,
        key="membrain_test_time_augmentation",
        default=True,
        true_flag="--test-time-augmentation",
        false_flag="--no-test-time-augmentation",
    )
    _extend_boolean_flag(
        command,
        context,
        key="membrain_store_uncertainty_map",
        default=False,
        true_flag="--store-uncertainty-map",
        false_flag="--no-store-uncertainty-map",
    )
    threshold = _float_parameter(context, "membrain_segmentation_threshold", default=0.0)
    command.extend(["--segmentation-threshold", f"{threshold:.6f}"])
    window_size = _positive_int_parameter(
        context,
        "membrain_sliding_window_size",
        default=160,
    )
    command.extend(["--sliding-window-size", str(window_size)])
    return command


def _find_membrain_segmentation_mrc(
    output_dir: Path,
    *,
    expected_name: str | None,
) -> Path:
    if expected_name is not None:
        expected_path = output_dir / expected_name
        if not expected_path.is_file():
            raise RuntimeError(f"MemBrain-seg output not found: {expected_path}")
        return expected_path
    candidates = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mrc", ".rec"}
    )
    if not candidates:
        raise RuntimeError(f"MemBrain-seg did not write a segmentation .mrc/.rec in {output_dir}")
    if len(candidates) == 1:
        return candidates[0]
    preferred = [
        path
        for path in candidates
        if "seg" in path.stem.lower()
        and not any(
            token in path.stem.lower()
            for token in ("prob", "score", "uncert", "component", "skeleton")
        )
    ]
    if len(preferred) == 1:
        return preferred[0]
    joined = ", ".join(str(path.relative_to(output_dir)) for path in candidates)
    raise RuntimeError(
        "MemBrain-seg wrote multiple .mrc/.rec outputs; pass "
        f"'membrain_output_name' to select one: {joined}"
    )


def _write_zarr_to_mrc(
    zarr_path: Path,
    output_path: Path,
    *,
    voxel_spacing_angstrom: float,
) -> None:
    source = cast(Any, zarr.open(zarr_path, mode="r"))
    data = np.asarray(source[:], dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("input tomogram contains nonfinite values")
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_spacing_angstrom


def _write_mrc_to_canonical_zarr(
    mrc_path: Path,
    output_path: Path,
    *,
    expected_shape: tuple[int, int, int],
    voxel_spacing_angstrom: float,
) -> _SegmentationStatistics:
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
    if data.shape != expected_shape:
        raise ValueError(f"segmentation shape {data.shape} does not match {expected_shape}")
    if not np.isfinite(data).all():
        raise ValueError("segmentation contains nonfinite values")
    chunks = (
        min(expected_shape[0], 16),
        min(expected_shape[1], 64),
        min(expected_shape[2], 256),
    )
    output = cast(
        Any,
        zarr.open(
            output_path,
            mode="w",
            shape=expected_shape,
            chunks=chunks,
            dtype=data.dtype,
        ),
    )
    output[:] = data
    output.attrs.update(
        {
            "axis_order": "zyx",
            "voxel_spacing_angstrom": voxel_spacing_angstrom,
            "segmentation_backend": "membrain-seg",
        }
    )
    data64 = data.astype(np.float64, copy=False)
    foreground = int(np.count_nonzero(data))
    return _SegmentationStatistics(
        dtype=str(data.dtype),
        minimum=float(data64.min()),
        maximum=float(data64.max()),
        mean=float(data64.mean()),
        standard_deviation=float(data64.std()),
        foreground_voxel_count=foreground,
        foreground_fraction=foreground / data.size,
    )


def _write_command_log(
    path: Path,
    command: Sequence[str],
    result: subprocess.CompletedProcess[str],
) -> None:
    path.write_text(
        f"$ {shlex.join(command)}\n\n"
        f"[stdout]\n{result.stdout or ''}\n"
        f"[stderr]\n{result.stderr or ''}\n"
        f"[exit_code]\n{result.returncode}\n"
    )


def _replace_output(source: Path, destination: Path) -> None:
    if destination.exists():
        _remove_path(destination)
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"segmentation outputs already exist: {joined}")


def _extend_boolean_flag(
    command: list[str],
    context: BackendContext,
    *,
    key: str,
    default: bool,
    true_flag: str,
    false_flag: str,
) -> None:
    command.append(true_flag if _bool_parameter(context, key, default=default) else false_flag)


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _positive_int_parameter(
    context: BackendContext,
    key: str,
    *,
    default: int,
) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < 1:
        raise ValueError(f"context parameter {key!r} must be at least 1")
    return value


def _optional_positive_int_parameter(context: BackendContext, key: str) -> int | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < 1:
        raise ValueError(f"context parameter {key!r} must be at least 1")
    return value


def _float_parameter(context: BackendContext, key: str, *, default: float) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"context parameter {key!r} must be finite")
    return normalized


def _positive_float_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
) -> float:
    normalized = _float_parameter(context, key, default=default)
    if normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be greater than zero")
    return normalized


def _optional_string_parameter(context: BackendContext, key: str) -> str | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"context parameter {key!r} must be a string")
    if not value:
        raise ValueError(f"context parameter {key!r} must not be empty")
    return value


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"
