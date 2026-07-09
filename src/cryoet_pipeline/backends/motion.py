from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr
from numpy.typing import NDArray

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
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
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.storage import ArtifactFormat


@dataclass(frozen=True)
class _MotionCor3Settings:
    executable: Path
    executable_sha256: str
    gpu_ids: tuple[int, ...]
    patch_x: int
    patch_y: int
    pixel_spacing_angstrom: float
    gain_reference: Path | None
    gain_rotation: int
    gain_flip: int
    software_version: str


class MotionCor3MotionCorrectionBackend:
    """Run MotionCor3 patch-based correction for one tilt movie at a time."""

    name = "motioncor3"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
        """Correct every movie with MotionCor3 and canonicalize its frame sum."""

        _validate_manifest_movies(manifest)
        settings = _motioncor3_settings(manifest, context)
        artifact_format = _artifact_format_parameter(context)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        output_paths = [
            _corrected_projection_path(
                context.output_dir,
                manifest,
                image,
                artifact_format=artifact_format,
                suffix="mc3",
            )
            for image in manifest.images
        ]
        _require_available_paths(output_paths, overwrite=overwrite)

        _storage_role_parameter(context)
        _retention_policy_parameter(context)
        _bool_parameter(context, "can_recompute", default=True)
        try:
            return [
                self._correct_image(
                    image,
                    manifest,
                    context,
                    settings=settings,
                    artifact_format=artifact_format,
                    overwrite=overwrite,
                )
                for image in manifest.images
            ]
        except Exception:
            if not overwrite:
                for path in output_paths:
                    if path.exists():
                        _remove_existing_path(path)
            raise

    def _correct_image(
        self,
        image: TiltImage,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        *,
        settings: _MotionCor3Settings,
        artifact_format: ArtifactFormat,
        overwrite: bool,
    ) -> Artifact:
        if image.local_frame_file is None:
            raise ValueError(
                f"{manifest.tilt_series_id} z={image.z_value}: missing local frame file"
            )

        output_path = _corrected_projection_path(
            context.output_dir,
            manifest,
            image,
            artifact_format=artifact_format,
            suffix="mc3",
        )
        work_dir = context.output_dir / "motion" / "motioncor3" / manifest.tilt_series_id
        log_dir = context.output_dir / "logs" / "motioncor3" / manifest.tilt_series_id
        alignment_dir = work_dir / "alignment"
        work_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        alignment_dir.mkdir(parents=True, exist_ok=True)

        output_stem = f"{manifest.tilt_series_id}_{image.z_value:03d}_mc3"
        command_log_path = log_dir / f"{output_stem}.runner.log"
        motioncor3_log_path = log_dir / f"{output_stem}.log"
        with tempfile.TemporaryDirectory(
            prefix=f".{output_stem}-",
            dir=work_dir.resolve(),
        ) as temporary_directory:
            temporary_output = Path(temporary_directory) / f"{output_stem}.mrc"
            command = _motioncor3_command(
                settings,
                input_path=image.local_frame_file,
                output_path=temporary_output,
                log_dir=log_dir,
                alignment_dir=alignment_dir,
            )
            result = self._command_runner(
                command,
                cwd=work_dir,
                env=os.environ.copy(),
            )
            _write_command_log(command_log_path, command, result)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"MotionCor3 failed for {image.local_frame_file} with exit code "
                    f"{result.returncode}: {detail}; see {command_log_path}"
                )

            projection = _read_motioncor3_projection(
                temporary_output,
                input_path=image.local_frame_file,
            )
            _write_projection(
                output_path,
                projection,
                artifact_format=artifact_format,
                pixel_spacing_angstrom=settings.pixel_spacing_angstrom,
                overwrite=overwrite,
            )

        software_versions = _software_versions()
        software_versions["MotionCor3"] = settings.software_version
        return Artifact(
            id=f"{manifest.tilt_series_id}:corrected_projection:{image.z_value:03d}",
            kind=ArtifactKind.CORRECTED_PROJECTION,
            path=output_path,
            shape=tuple(int(axis_size) for axis_size in projection.shape),
            dtype=str(projection.dtype),
            axis_order=AxisOrder.YX,
            pixel_spacing_angstrom=settings.pixel_spacing_angstrom,
            binning=image.binning,
            parameters={
                "backend": self.name,
                "method": "motioncor3_patch_correction",
                "artifact_format": artifact_format.value,
                "source_frame_file": str(image.local_frame_file),
                "z_value": image.z_value,
                "tilt_angle_deg": image.tilt_angle_deg,
                "num_subframes": image.num_subframes,
                "patch_grid_xy": [settings.patch_x, settings.patch_y],
                "gpu_ids": list(settings.gpu_ids),
                "pixel_spacing_angstrom": settings.pixel_spacing_angstrom,
                "gain_reference": (
                    str(settings.gain_reference)
                    if settings.gain_reference is not None
                    else None
                ),
                "gain_rotation": settings.gain_rotation,
                "gain_flip": settings.gain_flip,
                "dose_weighting": False,
                "aligned_movie_saved": False,
                "ctf_estimation": False,
                "motioncor3_executable": str(settings.executable),
                "motioncor3_executable_sha256": settings.executable_sha256,
                "command": command,
                "command_log": str(command_log_path),
                "motioncor3_log": str(motioncor3_log_path),
                "alignment_dir": str(alignment_dir),
            },
            software_versions=software_versions,
            storage_role=_storage_role_parameter(context),
            retention_policy=_retention_policy_parameter(context),
            can_recompute=_bool_parameter(context, "can_recompute", default=True),
            size_bytes=_path_size_bytes(output_path),
        )


class PhaseCorrelationMotionCorrectionBackend:
    """Motion correction using per-frame phase-correlation shift estimation.

    Each frame is registered to the first frame via normalized cross-power
    spectrum (phase correlation). Shifts are applied with subpixel accuracy
    using the Fourier shift theorem. Corrected frames are then averaged.
    """

    name = "phase_corr"

    def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
        """Correct every movie in the manifest and return projection artifacts."""

        _validate_manifest_movies(manifest)
        return [self.correct_image(image, manifest, context) for image in manifest.images]

    def correct_image(
        self,
        image: TiltImage,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Phase-correlate and average one multiframe movie."""

        if image.local_frame_file is None:
            raise ValueError(
                f"{manifest.tilt_series_id} z={image.z_value}: missing local frame file"
            )

        frames = _read_movie_frames(image.local_frame_file)
        shifts = _estimate_frame_shifts(frames)
        projection = _apply_shifts_and_average(frames, shifts)

        artifact_format = _artifact_format_parameter(context)
        output_path = _corrected_projection_path(
            context.output_dir,
            manifest,
            image,
            artifact_format=artifact_format,
            suffix="mc",
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
            shape=tuple(int(s) for s in projection.shape),
            dtype=str(projection.dtype),
            axis_order=AxisOrder.YX,
            pixel_spacing_angstrom=manifest.raw_pixel_spacing_angstrom,
            binning=image.binning,
            parameters={
                "backend": self.name,
                "method": "phase_correlation",
                "artifact_format": artifact_format.value,
                "source_frame_file": str(image.local_frame_file),
                "z_value": image.z_value,
                "tilt_angle_deg": image.tilt_angle_deg,
                "num_subframes": image.num_subframes,
                "frame_correction_shifts_yx_px": shifts.tolist(),
            },
            software_versions=_software_versions(),
            storage_role=_storage_role_parameter(context),
            retention_policy=_retention_policy_parameter(context),
            can_recompute=_bool_parameter(context, "can_recompute", default=True),
            size_bytes=_path_size_bytes(output_path),
        )


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


def _motioncor3_settings(
    manifest: TiltSeriesManifest,
    context: BackendContext,
) -> _MotionCor3Settings:
    if context.device is not DevicePreference.CUDA:
        raise ValueError(
            "MotionCor3 requires an NVIDIA CUDA device; run it on a Linux/CUDA host "
            "with --device cuda"
        )

    configured_pixel_spacing = context.parameters.get("motioncor3_pixel_size_angstrom")
    pixel_spacing_value = (
        manifest.raw_pixel_spacing_angstrom
        if configured_pixel_spacing is None
        else configured_pixel_spacing
    )
    pixel_spacing = _positive_float_value(
        pixel_spacing_value,
        name="motioncor3_pixel_size_angstrom",
    )
    software_version = context.parameters.get("motioncor3_version", "unknown")
    if not isinstance(software_version, str) or not software_version.strip():
        raise TypeError("context parameter 'motioncor3_version' must be a non-empty string")

    executable = _resolve_motioncor3_executable(context)
    return _MotionCor3Settings(
        executable=executable,
        executable_sha256=_sha256_file(executable),
        gpu_ids=_motioncor3_gpu_ids(context),
        patch_x=_bounded_int_parameter(
            context,
            "motioncor3_patch_x",
            default=5,
            minimum=1,
        ),
        patch_y=_bounded_int_parameter(
            context,
            "motioncor3_patch_y",
            default=5,
            minimum=1,
        ),
        pixel_spacing_angstrom=pixel_spacing,
        gain_reference=_optional_existing_file_parameter(
            context,
            "motioncor3_gain_reference",
        ),
        gain_rotation=_bounded_int_parameter(
            context,
            "motioncor3_gain_rotation",
            default=0,
            minimum=0,
            maximum=3,
        ),
        gain_flip=_bounded_int_parameter(
            context,
            "motioncor3_gain_flip",
            default=0,
            minimum=0,
            maximum=2,
        ),
        software_version=software_version.strip(),
    )


def _motioncor3_command(
    settings: _MotionCor3Settings,
    *,
    input_path: Path,
    output_path: Path,
    log_dir: Path,
    alignment_dir: Path,
) -> list[str]:
    command = [
        str(settings.executable),
        "-InMrc",
        str(input_path),
        "-OutMrc",
        str(output_path),
        "-Patch",
        str(settings.patch_x),
        str(settings.patch_y),
        "-FtBin",
        "1.0",
        "-Align",
        "1",
        "-OutStack",
        "0",
        "1",
        "-FmDose",
        "0",
        "-PixSize",
        f"{settings.pixel_spacing_angstrom:.8g}",
        "-Cs",
        "0",
        "-Gpu",
        *(str(gpu_id) for gpu_id in settings.gpu_ids),
        "-LogDir",
        str(log_dir),
        "-OutAln",
        str(alignment_dir),
    ]
    if settings.gain_reference is not None:
        command.extend(["-Gain", str(settings.gain_reference)])
    if settings.gain_rotation:
        command.extend(["-RotGain", str(settings.gain_rotation)])
    if settings.gain_flip:
        command.extend(["-FlipGain", str(settings.gain_flip)])
    return command


def _read_motioncor3_projection(
    output_path: Path,
    *,
    input_path: Path,
) -> NDArray[np.float32]:
    try:
        output_info = validate_complete_mrc(output_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"MotionCor3 did not produce a valid MRC output: {exc}") from exc

    input_info = validate_complete_mrc(input_path)
    expected_shape = tuple(input_info.shape[-2:])
    with mrcfile.open(output_path, permissive=True) as mrc:
        data = np.asarray(mrc.data)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        if data.ndim != 2:
            raise ValueError(
                "MotionCor3 output must contain one 2D projection, "
                f"got shape {output_info.shape}"
            )
        if data.shape != expected_shape:
            raise ValueError(
                f"MotionCor3 output shape {data.shape} does not match input frame "
                f"shape {expected_shape}"
            )
        projection = np.asarray(data, dtype=np.float32).copy()
    if not np.isfinite(projection).all():
        raise ValueError(f"MotionCor3 output contains non-finite pixels: {output_path}")
    return projection


def _resolve_motioncor3_executable(context: BackendContext) -> Path:
    configured = context.parameters.get("motioncor3_executable")
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError(
                "context parameter 'motioncor3_executable' must be a path string"
            )
        candidates = [Path(configured).expanduser()]
    else:
        discovered = shutil.which("MotionCor3")
        candidates = [Path(discovered)] if discovered is not None else []

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "MotionCor3 executable not found; put MotionCor3 on PATH or configure "
        "'motioncor3_executable'"
    )


def _motioncor3_gpu_ids(context: BackendContext) -> tuple[int, ...]:
    value = context.parameters.get("motioncor3_gpu_ids", (0,))
    if isinstance(value, bool):
        raise TypeError("context parameter 'motioncor3_gpu_ids' must contain integers")
    if isinstance(value, int):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    else:
        raise TypeError("context parameter 'motioncor3_gpu_ids' must contain integers")

    gpu_ids: list[int] = []
    for gpu_id in values:
        if isinstance(gpu_id, bool) or not isinstance(gpu_id, int):
            raise TypeError("context parameter 'motioncor3_gpu_ids' must contain integers")
        if gpu_id < 0:
            raise ValueError("context parameter 'motioncor3_gpu_ids' cannot be negative")
        gpu_ids.append(gpu_id)
    if not gpu_ids:
        raise ValueError("context parameter 'motioncor3_gpu_ids' cannot be empty")
    return tuple(gpu_ids)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_existing_file_parameter(
    context: BackendContext,
    key: str,
) -> Path | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError(f"context parameter {key!r} must be a path string")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"context parameter {key!r} does not exist: {path}")
    return path.resolve()


def _bounded_int_parameter(
    context: BackendContext,
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and {maximum}" if maximum is not None else ""
        raise ValueError(
            f"context parameter {key!r} must be between {minimum}{suffix}"
        )
    return value


def _positive_float_value(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {name!r} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"context parameter {name!r} must be greater than zero")
    return normalized


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            f"corrected projection output already exists:\n{formatted}"
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


def _read_movie_frames(path: Path) -> NDArray[np.float32]:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(
            f"expected multiframe MRC with shape (frames, y, x), got {data.shape}"
        )
    return data


def _estimate_frame_shifts(frames: NDArray[np.float32]) -> NDArray[np.float64]:
    """Estimate (dy, dx) corrections that register each frame to frame zero."""
    n_frames, height, width = frames.shape
    shifts = np.zeros((n_frames, 2), dtype=np.float64)
    ref_fft = np.fft.rfft2(frames[0])
    for i in range(1, n_frames):
        frame_fft = np.fft.rfft2(frames[i])
        cross_power = ref_fft * np.conj(frame_fft)
        norm = np.abs(cross_power)
        normalized = np.zeros_like(cross_power)
        np.divide(cross_power, norm, out=normalized, where=norm > 1e-10)
        cross_power = normalized
        cc = np.fft.irfft2(cross_power, s=(height, width))
        peak = np.unravel_index(np.argmax(cc), cc.shape)
        dy = int(peak[0])
        dx = int(peak[1])
        if dy > height // 2:
            dy -= height
        if dx > width // 2:
            dx -= width
        shifts[i] = [dy, dx]
    return shifts


def _apply_shifts_and_average(
    frames: NDArray[np.float32],
    shifts: NDArray[np.float64],
) -> NDArray[np.float32]:
    """Apply Fourier-domain shifts to each frame and return their mean."""
    n_frames, height, width = frames.shape
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.rfftfreq(width)[None, :]
    accumulator = np.zeros((height, width), dtype=np.float64)
    for i in range(n_frames):
        dy, dx = shifts[i]
        fft = np.fft.rfft2(frames[i])
        phase = np.exp(-2j * np.pi * (dy * ky + dx * kx))
        shifted = np.fft.irfft2(fft * phase, s=(height, width))
        accumulator += shifted
    result = (accumulator / n_frames).astype(np.float32)
    return cast(NDArray[np.float32], result)


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
    suffix: str = "avg",
) -> Path:
    extension = "zarr" if artifact_format is ArtifactFormat.ZARR else "mrc"
    return (
        output_dir
        / "corrected"
        / manifest.tilt_series_id
        / f"{manifest.tilt_series_id}_{image.z_value:03d}_{suffix}.{extension}"
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
