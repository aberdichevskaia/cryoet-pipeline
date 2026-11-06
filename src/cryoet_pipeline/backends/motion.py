from __future__ import annotations

import hashlib
import os
import shutil
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
from cryoet_pipeline.backends.command_log import write_command_log
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
from cryoet_pipeline.movies import MovieFormat, inspect_movie, movie_format, read_movie
from cryoet_pipeline.mrc_validation import validate_complete_mrc
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.storage import ArtifactFormat
from cryoet_pipeline.tool_versions import motioncor3_version_of


@dataclass(frozen=True)
class _MotionCor3Settings:
    executable: Path
    executable_sha256: str
    gpu_ids: tuple[int, ...]
    patch_x: int
    patch_y: int
    # The movies' pixel, which MotionCor3 takes as -PixSize; the sums it writes
    # are binned by ftbin, so their pixel is this times ftbin.
    pixel_spacing_angstrom: float
    ftbin: int
    gain_reference: Path | None
    gain_rotation: int
    gain_flip: int
    software_version: str
    # Every movie of the series is in this format; the next two apply to EER.
    movie_format: MovieFormat = MovieFormat.MRC
    # False when gain_flip is the pipeline's default (_default_gain_flip), not
    # one the lane or the command line gave.
    gain_flip_given: bool = True
    # -EerSampling: 1 renders the events on the sensor's pixels, 2 on a grid
    # twice as fine, which -FtBin then has to bin back.
    eer_sampling: int = 1
    # The groups the EER frames are summed in before alignment (-FmIntFile);
    # None: from each movie's dose (eer_fractions_for_dose).
    eer_fractions: int | None = None

    @property
    def render_factor(self) -> int:
        """How much finer than the movie's pixel the frames are rendered."""

        return self.eer_sampling if self.movie_format is MovieFormat.EER else 1

    @property
    def frame_binning(self) -> int:
        """The sums' pixel over the movie's: ftbin, less the EER rendering."""

        return self.ftbin // self.render_factor

    @property
    def output_pixel_spacing_angstrom(self) -> float:
        return self.pixel_spacing_angstrom * self.frame_binning


class MotionCor3MotionCorrectionBackend:
    """Run MotionCor3 patch-based correction for one tilt movie at a time."""

    name = "motioncor3"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
        """Correct every movie with MotionCor3 and canonicalize its frame sum."""

        movie_format = _validate_manifest_movies(manifest, reads_eer=True)
        settings = _motioncor3_settings(manifest, context, movie_format=movie_format)
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
        eer_parameters: dict[str, object] = {}
        fmint_path: Path | None = None
        if settings.movie_format is MovieFormat.EER:
            tilt_dose = _eer_tilt_dose_e_per_a2(
                image.local_frame_file, settings.pixel_spacing_angstrom
            )
            fractions = (
                settings.eer_fractions
                if settings.eer_fractions is not None
                else eer_fractions_for_dose(image.num_subframes, tilt_dose)
            )
            fmint_line = motioncor3_eer_fmint(image.num_subframes, fractions)
            fmint_path = work_dir / f"{output_stem}.fmint"
            fmint_path.write_text(fmint_line)
            group_size = int(fmint_line.split()[1])
            # MotionCor3 makes N // G groups, the last N % G of them one frame longer.
            groups = image.num_subframes // group_size
            eer_parameters = {
                "eer_sampling": settings.eer_sampling,
                "eer_fractions": fractions,
                "eer_fractions_given": settings.eer_fractions is not None,
                "eer_frames": image.num_subframes,
                "eer_group_size": group_size,
                "eer_groups": groups,
                # From the movie's metadata and pixel; None when it states no dose.
                "eer_tilt_dose_e_per_a2": tilt_dose,
                "eer_group_dose_e_per_a2": (
                    None if tilt_dose is None else tilt_dose / groups
                ),
                "fmint_file": str(fmint_path),
            }
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
                fmint_path=fmint_path,
            )
            result = self._command_runner(
                command,
                cwd=work_dir,
                env=os.environ.copy(),
            )
            write_command_log(command_log_path, (command, result))
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"MotionCor3 failed for {image.local_frame_file} with exit code "
                    f"{result.returncode}: {detail}; see {command_log_path}"
                )
            gain_loaded = _gain_loaded(settings, result.stdout, command_log_path)

            projection = _read_motioncor3_projection(
                temporary_output,
                input_path=image.local_frame_file,
                ftbin=settings.ftbin,
                render_factor=settings.render_factor,
            )
            _write_projection(
                output_path,
                projection,
                artifact_format=artifact_format,
                pixel_spacing_angstrom=settings.output_pixel_spacing_angstrom,
                overwrite=overwrite,
            )

        software_versions = _software_versions()
        software_versions["MotionCor3"] = settings.software_version
        return Artifact(
            id=(
                f"{manifest.tilt_series_id}:corrected_projection:"
                f"{image.z_value:03d}:{self.name}"
            ),
            backend=self.name,
            kind=ArtifactKind.CORRECTED_PROJECTION,
            path=output_path,
            shape=tuple(int(axis_size) for axis_size in projection.shape),
            dtype=str(projection.dtype),
            axis_order=AxisOrder.YX,
            pixel_spacing_angstrom=settings.output_pixel_spacing_angstrom,
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
                "pixel_spacing_angstrom": settings.output_pixel_spacing_angstrom,
                "input_pixel_spacing_angstrom": settings.pixel_spacing_angstrom,
                "movie_format": settings.movie_format.value,
                "ftbin": settings.ftbin,
                # The sums' pixel over the movie's, which the stack records.
                "frame_binning": settings.frame_binning,
                **eer_parameters,
                "gain_reference": (
                    str(settings.gain_reference)
                    if settings.gain_reference is not None
                    else None
                ),
                "gain_rotation": settings.gain_rotation,
                "gain_flip": settings.gain_flip,
                "gain_flip_given": settings.gain_flip_given,
                "gain_loaded": gain_loaded,
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

    name = "phase-corr"

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
            id=(
                f"{manifest.tilt_series_id}:corrected_projection:"
                f"{image.z_value:03d}:{self.name}"
            ),
            backend=self.name,
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
            id=(
                f"{manifest.tilt_series_id}:corrected_projection:"
                f"{image.z_value:03d}:{self.name}"
            ),
            backend=self.name,
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
    *,
    movie_format: MovieFormat = MovieFormat.MRC,
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
    # A version given is the one recorded; otherwise MotionCor3 is asked for its
    # own once the executable is known (tool_versions.motioncor3_version_of).
    configured_version = context.parameters.get("motioncor3_version")
    if configured_version is not None and (
        not isinstance(configured_version, str) or not configured_version.strip()
    ):
        raise TypeError("context parameter 'motioncor3_version' must be a non-empty string")

    ftbin = _bounded_int_parameter(context, "motioncor3_ftbin", default=1, minimum=1)
    # 16K rendering (-EerSampling 3) is left out: MotionCor3 caps it at 2 for
    # compression 65002 without changing the pixel size it reports, and
    # tomography has no use for it.
    eer_sampling = _bounded_int_parameter(
        context, "motioncor3_eer_sampling", default=1, minimum=1, maximum=2
    )
    eer_fractions = (
        _bounded_int_parameter(context, "motioncor3_eer_fractions", default=10, minimum=1)
        if context.parameters.get("motioncor3_eer_fractions") is not None
        else None
    )
    if movie_format is MovieFormat.EER and ftbin % eer_sampling:
        raise ValueError(
            f"EER sampling {eer_sampling} renders the frames on a grid {eer_sampling} times "
            f"finer than the movie pixel, and -FtBin {ftbin} would leave the sums at a "
            f"fraction of it; set ftbin to a multiple of {eer_sampling}, e.g. "
            f"{eer_sampling} for sums at the movie pixel"
        )

    executable = _resolve_motioncor3_executable(context)
    software_version = (
        configured_version.strip()
        if configured_version is not None
        else motioncor3_version_of(executable) or "unknown"
    )
    gain_reference = _optional_existing_file_parameter(context, "motioncor3_gain_reference")
    gain_flip_given = context.parameters.get("motioncor3_gain_flip") is not None
    gain_flip = (
        _bounded_int_parameter(context, "motioncor3_gain_flip", default=0, minimum=0, maximum=2)
        if gain_flip_given
        else _default_gain_flip(gain_reference, movie_format)
    )
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
        # Fourier binning of the sums: 2 brings super-resolution frames to the
        # physical pixel. Whole numbers only, so that the stack's pixel stays a
        # whole multiple of the movies' and the fiducial step can check it.
        ftbin=ftbin,
        gain_reference=gain_reference,
        gain_rotation=_bounded_int_parameter(
            context,
            "motioncor3_gain_rotation",
            default=0,
            minimum=0,
            maximum=3,
        ),
        gain_flip=gain_flip,
        software_version=software_version,
        movie_format=movie_format,
        gain_flip_given=gain_flip_given,
        eer_sampling=eer_sampling,
        eer_fractions=eer_fractions,
    )


def _default_gain_flip(gain_reference: Path | None, movie_format: MovieFormat) -> int:
    """-FlipGain when none is given: 1 for a Falcon `.gain` with EER movies, else 0.

    MotionCor3 reads a `.gain`, a TIFF, bottom row first (TiffUtil/CLoadTiffImage.cpp)
    and decodes EER frames top row first (EerUtil/CDecodeEerFrame.cpp), so the gain
    has to be turned upside down to meet the frames. Measured on 5 November 2026 with
    the MotionCor3 the MDC lanes run, on three EMPIAR-11830 movies
    (scripts/eer_gain_flip.sbatch, job 3890116): over the three, only the flipped gain
    anticorrelates with the counts (r -0.016 against -0.0005; 1/sqrt(n) 0.00025), and
    each movie's sum is flatter with it. TIFF movies go through the same reader as the
    gain, so by the sources they need no flip (not measured); an MRC gain lies the way
    whatever wrote it laid it, so it keeps MotionCor3's default.
    """

    if (
        movie_format is MovieFormat.EER
        and gain_reference is not None
        and gain_reference.suffix.lower() == ".gain"
    ):
        return 1
    return 0


_INPUT_FLAGS = {MovieFormat.MRC: "-InMrc", MovieFormat.TIFF: "-InTiff", MovieFormat.EER: "-InEer"}


def motioncor3_eer_fmint(eer_frames: int, fractions: int) -> str:
    """MotionCor3's -FmIntFile for one EER movie: its frames summed in `fractions` groups.

    One line, "N G 0": the N EER frames summed G at a time, at dose 0, so that
    nothing is dose-weighted (the pipeline passes -FmDose 0 too). G is
    N // fractions. MotionCor3 adds the N % G frames left over to the last
    groups, one each, and writes past the end of its array when they outnumber
    the groups (DataUtil/CFmIntParam.cpp, mCalcIntFms: 728 frames in groups
    of 73 leave 71 for 9 groups). N // fractions never leaves more than there
    are groups.
    """

    if fractions < 1:
        raise ValueError(f"EER fractions must be at least 1, not {fractions}")
    if eer_frames < fractions:
        raise ValueError(f"{eer_frames} EER frames cannot make {fractions} fractions")
    return f"{eer_frames} {eer_frames // fractions} 0\n"


# Each group MotionCor3 aligns should carry about this dose. The rule of thumb for
# tomography is at least 0.1-0.15 e/A^2 per aligned frame, below which the
# frames no longer align, and tilt movies of physical frames carry 0.2-0.3. More
# groups follow the drift more finely; fewer align more surely.
EER_GROUP_DOSE_E_PER_A2 = 0.2
# Groups for a movie that states no dose per pixel: the count used before the
# dose was read.
EER_FRACTIONS_WITHOUT_DOSE = 10
# Never fewer, so that MotionCor3 always has frames to align (what it does
# with a single frame was not read).
EER_MIN_FRACTIONS = 2


def eer_fractions_for_dose(eer_frames: int, tilt_dose_e_per_a2: float | None) -> int:
    """How many groups to sum one EER movie's frames in, from its dose.

    As many as keep each group at EER_GROUP_DOSE_E_PER_A2 or more: 9 for an
    EMPIAR-11830 tilt (6.53 e/pixel at 1.9037 A, 1.80 e/A^2). At least
    EER_MIN_FRACTIONS, at most one group per EER frame, and
    EER_FRACTIONS_WITHOUT_DOSE when the dose is not known.
    """

    if tilt_dose_e_per_a2 is None:
        fractions = EER_FRACTIONS_WITHOUT_DOSE
    else:
        # The small addition keeps a dose of exactly n groups' worth at n.
        fractions = int(tilt_dose_e_per_a2 / EER_GROUP_DOSE_E_PER_A2 + 1e-9)
    return max(1, min(max(fractions, EER_MIN_FRACTIONS), eer_frames))


def _eer_tilt_dose_e_per_a2(movie: Path, pixel_spacing_angstrom: float) -> float | None:
    """The dose an EER movie's metadata states, per square angstrom of the movie pixel."""

    info = inspect_movie(movie)
    dose = info.eer.total_dose_e_per_pixel if info.eer is not None else None
    return None if dose is None else dose / pixel_spacing_angstrom**2


def _gain_loaded(
    settings: _MotionCor3Settings,
    output: str | None,
    command_log_path: Path,
) -> bool | None:
    """Whether MotionCor3 said it applied the gain: True, or None when it did not say.

    MotionCor3 only warns when it cannot use a gain (any mode but 32-bit
    float) and sums without one; "Gain reference not found" in its output then
    stops the step here. The success line is the one MotionCor3's main branch
    prints (CMain.cpp); a build that words it otherwise leaves None.
    """

    if settings.gain_reference is None:
        return None
    text = output or ""
    if "Gain reference not found" in text:
        raise RuntimeError(
            f"MotionCor3 did not apply the gain reference {settings.gain_reference}: it "
            "says 'Gain reference not found' (it loads 32-bit float gains only); see "
            f"{command_log_path}"
        )
    return True if "Gain reference has been loaded" in text else None


def _motioncor3_command(
    settings: _MotionCor3Settings,
    *,
    input_path: Path,
    output_path: Path,
    log_dir: Path,
    alignment_dir: Path,
    fmint_path: Path | None = None,
) -> list[str]:
    # MotionCor3 runs in its work directory, not ours, so a relative path here
    # (a manifest written by `init --frames data/...`, a relative --out) would
    # be looked up from the wrong place: "cannot open stack file".
    input_format = movie_format(input_path)
    if input_format is MovieFormat.EER and fmint_path is None:
        raise ValueError(f"{input_path}: an EER movie needs its -FmIntFile")
    command = [
        str(settings.executable),
        _INPUT_FLAGS[input_format],
        str(input_path.resolve()),
        "-OutMrc",
        str(output_path.resolve()),
        "-Patch",
        str(settings.patch_x),
        str(settings.patch_y),
        "-FtBin",
        str(settings.ftbin),
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
        str(log_dir.resolve()),
        "-OutAln",
        str(alignment_dir.resolve()),
    ]
    if input_format is MovieFormat.EER:
        assert fmint_path is not None
        command.extend(
            [
                "-EerSampling",
                str(settings.eer_sampling),
                "-FmIntFile",
                str(fmint_path.resolve()),
            ]
        )
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
    ftbin: int = 1,
    render_factor: int = 1,
) -> NDArray[np.float32]:
    try:
        output_info = validate_complete_mrc(output_path)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"MotionCor3 did not produce a valid MRC output: {exc}") from exc

    expected_shape = tuple(
        size * render_factor // ftbin for size in inspect_movie(input_path).shape[1:]
    )
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



def _read_movie_frames(path: Path) -> NDArray[np.float32]:
    return np.asarray(read_movie(path), dtype=np.float32)


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
    data = read_movie(path)
    projection = np.zeros(data.shape[1:], dtype=np.float32)
    for frame in data:
        np.add(projection, frame, out=projection, casting="unsafe")
    projection /= data.shape[0]
    return cast(NDArray[np.float32], projection)


def _validate_manifest_movies(
    manifest: TiltSeriesManifest,
    *,
    reads_eer: bool = False,
) -> MovieFormat:
    """Every movie is there, in one format, with the frames the manifest declares.

    Returns that format. EER movies pass only for a backend that decodes them
    (`reads_eer`), which is MotionCor3 alone.
    """

    issues: list[str] = []
    formats: set[MovieFormat] = set()
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
            info = inspect_movie(path)
        except (OSError, ValueError) as exc:
            issues.append(str(exc))
            continue
        formats.add(info.format)

        if image.num_subframes is not None and info.shape[0] != image.num_subframes:
            issues.append(
                f"{path}: manifest declares {image.num_subframes} frames, "
                f"the movie has {info.shape[0]}"
            )

    if len(formats) > 1:
        issues.append(
            f"{manifest.tilt_series_id}: movies in more than one format "
            f"({', '.join(sorted(fmt.value for fmt in formats))})"
        )
    if MovieFormat.EER in formats and not reads_eer:
        issues.append(
            f"{manifest.tilt_series_id}: EER movies are decoded by the motioncor3 "
            "backend only"
        )
    if issues:
        details = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"input movie validation failed:\n{details}")
    (movie_format,) = formats
    return movie_format


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
