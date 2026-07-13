from __future__ import annotations

import json
import math
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
from cryoet_pipeline.backends.protocols import BackendContext, DenoisingBackend
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
    TomogramBranch,
    TomogramRestorationQc,
)


class IsoNet2RestorationBackend:
    """Restore reconstructed tomograms with an external IsoNet2 installation."""

    name = "isonet2"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def denoise(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Return a canonical restored Zarr tomogram artifact."""

        shape = _validate_input_tomogram(tomogram, manifest)
        voxel_spacing = _voxel_spacing(tomogram)
        branch = _tomogram_branch(tomogram)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _restoration_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            branch=branch,
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        executable = _resolve_isonet2_executable(context)
        model = _resolve_isonet2_model(context)
        number_subtomos = _positive_int_parameter(
            context,
            "isonet2_number_subtomos",
            default=100,
        )

        paths.restoration_dir.mkdir(parents=True, exist_ok=True)
        paths.qc_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-isonet2-",
            dir=paths.restoration_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            isonet_input_dir = temporary_root / "isonet_input"
            isonet_output_dir = temporary_root / "isonet_output"
            temporary_input = isonet_input_dir / f"{manifest.tilt_series_id}.mrc"
            temporary_star = temporary_root / "tomograms.star"
            temporary_zarr = temporary_root / f"{manifest.tilt_series_id}_restored.zarr"
            temporary_report = temporary_root / "isonet2_qc.json"

            isonet_input_dir.mkdir()
            isonet_output_dir.mkdir()
            _write_zarr_to_mrc(
                tomogram.path,
                temporary_input,
                voxel_spacing_angstrom=voxel_spacing,
            )
            prepare_command = _isonet2_prepare_star_command(
                executable,
                input_dir=isonet_input_dir,
                star_path=temporary_star,
                voxel_spacing_angstrom=voxel_spacing,
                number_subtomos=number_subtomos,
            )
            prepare_result = self._command_runner(prepare_command, cwd=temporary_root, env={})
            _write_command_log(paths.log, [(prepare_command, prepare_result)])
            if prepare_result.returncode != 0:
                raise RuntimeError(
                    f"IsoNet2 prepare_star failed with exit code "
                    f"{prepare_result.returncode}; see {paths.log}"
                )

            predict_command = _isonet2_predict_command(
                executable,
                context,
                star_path=temporary_star,
                model_path=model,
                output_dir=isonet_output_dir,
            )
            predict_result = self._command_runner(predict_command, cwd=temporary_root, env={})
            _write_command_log(
                paths.log,
                [(prepare_command, prepare_result), (predict_command, predict_result)],
            )
            if predict_result.returncode != 0:
                raise RuntimeError(
                    f"IsoNet2 predict failed with exit code {predict_result.returncode}; "
                    f"see {paths.log}"
                )

            restored_mrc = _find_isonet2_restored_mrc(isonet_output_dir)

            statistics = _write_mrc_to_canonical_zarr(
                restored_mrc,
                temporary_zarr,
                expected_shape=shape,
                voxel_spacing_angstrom=voxel_spacing,
            )
            restored_id = f"{manifest.tilt_series_id}:tomogram:{branch.value}:isonet2"
            report = TomogramRestorationQc(
                tilt_series_id=manifest.tilt_series_id,
                backend=self.name,
                input_tomogram_id=tomogram.id,
                restored_tomogram_id=restored_id,
                tomogram_branch=branch,
                shape=shape,
                voxel_spacing_angstrom=voxel_spacing,
                minimum=statistics.minimum,
                maximum=statistics.maximum,
                mean=statistics.mean,
                standard_deviation=statistics.standard_deviation,
                finite=True,
                command_log_path=paths.log,
                status=QcStatus.PASS,
            )
            temporary_report.write_text(
                json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
            )

            _replace_output(temporary_zarr, paths.restored_zarr)
            _replace_output(temporary_report, paths.report)

        return Artifact(
            id=f"{manifest.tilt_series_id}:tomogram:{branch.value}:isonet2",
            kind=ArtifactKind.DENOISED_TOMOGRAM,
            path=paths.restored_zarr,
            parent_ids=[tomogram.id],
            shape=shape,
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=voxel_spacing,
            binning=tomogram.binning,
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "tomogram_branch": branch.value,
                "input_tomogram_id": tomogram.id,
                "restoration_method": "isonet2",
                "isonet2_model_path": str(model),
                "isonet2_number_subtomos": number_subtomos,
                "isonet2_output_mrc_name": restored_mrc.name,
                "command_log_path": str(paths.log),
                "qc_path": str(paths.report),
            },
            software_versions={
                "isonet2": "external",
                "zarr": _package_version("zarr"),
            },
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_path_size_bytes(paths.restored_zarr) + paths.report.stat().st_size,
        )


def restore_and_register(
    backend: DenoisingBackend,
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Restore a tomogram and register restored tomogram plus QC artifacts."""

    restored = backend.denoise(tomogram, manifest, context)
    qc_path = restored.parameters.get("qc_path")
    if not isinstance(qc_path, str):
        raise ValueError("restoration backend must record qc_path")
    qc = Artifact(
        id=f"{manifest.tilt_series_id}:qc:restoration:{restored.parameters['backend']}",
        kind=ArtifactKind.QC,
        path=Path(qc_path),
        parent_ids=[restored.id],
        parameters={
            "qc_type": "tomogram_restoration",
            "backend": restored.parameters["backend"],
            "tilt_series_id": manifest.tilt_series_id,
            "tomogram_branch": restored.parameters.get("tomogram_branch", "full"),
            "status": QcStatus.PASS.value,
        },
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=Path(qc_path).stat().st_size,
    )
    artifacts = [restored, qc]
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


class _RestorationPaths:
    def __init__(
        self,
        *,
        restoration_dir: Path,
        qc_dir: Path,
        restored_zarr: Path,
        report: Path,
        log: Path,
    ) -> None:
        self.restoration_dir = restoration_dir
        self.qc_dir = qc_dir
        self.restored_zarr = restored_zarr
        self.report = report
        self.log = log

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.restored_zarr, self.report, self.log)


@dataclass(frozen=True)
class _VolumeStatistics:
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float


def _restoration_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    branch: TomogramBranch,
) -> _RestorationPaths:
    restoration_dir = output_root / "tomograms" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "restoration"
    prefix = f"{manifest.tilt_series_id}_{branch.value}_{backend_name}"
    return _RestorationPaths(
        restoration_dir=restoration_dir,
        qc_dir=qc_dir,
        restored_zarr=restoration_dir / f"{prefix}.zarr",
        report=qc_dir / f"{prefix}_qc.json",
        log=restoration_dir / f"{prefix}.log",
    )


def _validate_input_tomogram(
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[int, int, int]:
    if tomogram.kind is not ArtifactKind.TOMOGRAM:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if tomogram.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError(
            f"tomogram is for {tomogram.parameters.get('tilt_series_id')}, "
            f"expected {manifest.tilt_series_id}"
        )
    if tomogram.axis_order is not AxisOrder.ZYX:
        raise ValueError("restoration requires a canonical ZYX tomogram")
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
        raise ValueError("tomogram voxel spacing is required for IsoNet2 restoration")
    if not math.isfinite(tomogram.pixel_spacing_angstrom) or tomogram.pixel_spacing_angstrom <= 0:
        raise ValueError("tomogram voxel spacing must be finite and positive")
    return tomogram.pixel_spacing_angstrom


def _tomogram_branch(tomogram: Artifact) -> TomogramBranch:
    value = tomogram.parameters.get("tomogram_branch", TomogramBranch.FULL.value)
    return TomogramBranch(value)


def _resolve_isonet2_executable(context: BackendContext) -> Path:
    configured = context.parameters.get("isonet2_executable")
    if configured is not None:
        if not isinstance(configured, Path):
            raise TypeError("context parameter 'isonet2_executable' must be a Path")
        if not configured.is_file():
            raise FileNotFoundError(f"IsoNet2 executable not found: {configured}")
        return configured
    for name in ("isonet.py", "isonet2"):
        resolved = shutil.which(name)
        if resolved is not None:
            return Path(resolved)
    raise FileNotFoundError(
        "IsoNet2 executable not found; pass --isonet2-executable or add it to PATH"
    )


def _resolve_isonet2_model(context: BackendContext) -> Path:
    configured = context.parameters.get("isonet2_model")
    if not isinstance(configured, Path):
        raise TypeError("context parameter 'isonet2_model' must be a Path")
    if not configured.is_file():
        raise FileNotFoundError(f"IsoNet2 model not found: {configured}")
    return configured


def _isonet2_prepare_star_command(
    executable: Path,
    *,
    input_dir: Path,
    star_path: Path,
    voxel_spacing_angstrom: float,
    number_subtomos: int,
) -> list[str]:
    return [
        str(executable),
        "prepare_star",
        str(input_dir),
        "--output_star",
        str(star_path),
        "--pixel_size",
        f"{voxel_spacing_angstrom:.6f}",
        "--number_subtomos",
        str(number_subtomos),
    ]


def _isonet2_predict_command(
    executable: Path,
    context: BackendContext,
    *,
    star_path: Path,
    model_path: Path,
    output_dir: Path,
) -> list[str]:
    command = [
        str(executable),
        "predict",
        str(star_path),
        str(model_path),
        "--output_dir",
        str(output_dir),
    ]
    gpu_id = _optional_string_parameter(context, "isonet2_gpu_id")
    if gpu_id is not None:
        command.extend(["--gpuID", gpu_id])
    cube_size = _positive_int_parameter(context, "isonet2_cube_size", default=64)
    crop_size = _positive_int_parameter(context, "isonet2_crop_size", default=96)
    command.extend(["--cube_size", str(cube_size), "--crop_size", str(crop_size)])
    batch_size = _optional_positive_int_parameter(context, "isonet2_batch_size")
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    normalize_percentile = _bool_parameter(context, "isonet2_normalize_percentile", default=True)
    command.extend(["--normalize_percentile", str(normalize_percentile)])
    return command


def _find_isonet2_restored_mrc(output_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mrc", ".rec"}
    )
    if not candidates:
        raise RuntimeError(f"IsoNet2 did not write a restored .mrc/.rec in {output_dir}")
    if len(candidates) > 1:
        joined = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"IsoNet2 wrote multiple restored volumes in {output_dir}: {joined}")
    return candidates[0]


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
) -> _VolumeStatistics:
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
    if data.shape != expected_shape:
        raise ValueError(f"restored tomogram shape {data.shape} does not match {expected_shape}")
    if not np.isfinite(data).all():
        raise ValueError("restored tomogram contains nonfinite values")
    output = cast(
        Any,
        zarr.open(
            output_path,
            mode="w",
            shape=expected_shape,
            chunks=(
                min(expected_shape[0], 16),
                min(expected_shape[1], 64),
                min(expected_shape[2], 256),
            ),
            dtype=np.float32,
        ),
    )
    output[:] = data
    output.attrs.update(
        {
            "axis_order": "zyx",
            "voxel_spacing_angstrom": voxel_spacing_angstrom,
            "restoration_backend": "isonet2",
        }
    )
    data64 = data.astype(np.float64, copy=False)
    return _VolumeStatistics(
        minimum=float(data64.min()),
        maximum=float(data64.max()),
        mean=float(data64.mean()),
        standard_deviation=float(data64.std()),
    )


def _write_command_log(
    path: Path,
    command_results: Sequence[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
) -> None:
    sections = []
    for command, result in command_results:
        sections.append(
            f"$ {shlex.join(command)}\n\n"
            f"[stdout]\n{result.stdout or ''}\n"
            f"[stderr]\n{result.stderr or ''}\n"
            f"[exit_code]\n{result.returncode}\n"
        )
    path.write_text("\n\n".join(sections))


def _replace_output(source: Path, destination: Path) -> None:
    if destination.exists():
        _remove_path(destination)
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"restoration outputs already exist: {joined}")


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
