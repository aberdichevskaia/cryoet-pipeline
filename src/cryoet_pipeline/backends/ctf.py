from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    resolve_imod_executable,
    run_command,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    CtfCorrectionBackend,
    CtfEstimationBackend,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.imod_defocus import (
    parse_ctffind_summary,
    read_imod_defocus_file,
    write_imod_defocus_file,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    CtfCorrectionQc,
    CtfEstimationQc,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import inspect_mrc_file, validate_complete_mrc

DEFAULT_DEFOCUS_SEARCH_WINDOW_ANGSTROM = 1000.0
"""Half-width of the narrow second-pass defocus search around the hint."""


class Ctffind5CtfEstimationBackend:
    """Estimate per-tilt CTF with CTFFIND, emitting an IMOD `.defocus` file.

    CTFFIND has no command-line parameters: it asks questions and reads answers
    from stdin, positionally, so the adapter has to know the exact question
    sequence. The sequence depends on the input: CTFFIND opens the file before
    asking anything and asks its movie question only when the file holds more
    than one slice, so a tilt series draws one question more than a single
    extracted view. `_ctffind_dialogue` builds questions and answers together and
    `_verify_dialogue` checks them against CTFFIND's own echo, because landing an
    answer on the wrong question is usually quiet rather than loud.

    `--old-school-input-ctffind4` offers a fixed record format instead and was
    tried, but it is pathologically slow here: the same view fits in about two
    seconds interactively and had not finished after two minutes in old-school
    mode, on identical parameters. That mode also pins itself to one thread.

    Answering no to the movie question makes CTFFIND fit each slice separately,
    so one call covers the series. Fitting runs in two passes, which is markedly
    more robust than one wide search: the view nearest zero tilt is fitted first
    over a wide defocus range to get a hint, then the stack is fitted over a
    narrow window around it. Pass one is skipped when a defocus is supplied.
    """

    name = "ctffind5"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def estimate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a CTF model artifact and its QC artifact."""

        stack_shape = _validate_final_aligned_stack(tilt_stack)
        alignment_result = _load_alignment(alignment, tilt_stack, manifest)
        pixel_spacing_angstrom = _voxel_spacing(tilt_stack)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        optics = _optics_parameters(context)
        settings = _ctffind_settings(context, pixel_spacing_angstrom=pixel_spacing_angstrom)

        tilt_angles_deg = [transform.tilt_angle_deg for transform in alignment_result.transforms]
        if len(tilt_angles_deg) != stack_shape[0]:
            raise ValueError(
                f"alignment has {len(tilt_angles_deg)} transforms but the aligned stack has "
                f"{stack_shape[0]} views"
            )

        paths = _estimation_paths(
            context.output_dir, manifest, backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)
        executable = _resolve_ctffind_executable(context)

        command_log: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]] = []
        # Not a TemporaryDirectory: on failure the extracted view and the exact
        # stdin that produced it are the only evidence, and cleaning them up
        # leaves nothing to diagnose.
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".{manifest.tilt_series_id}-ctffind-",
                dir=paths.output_dir.resolve(),
            )
        )
        succeeded = False
        try:
            hint_defocus_angstrom = _explicit_expected_defocus_angstrom(context)
            if hint_defocus_angstrom is None:
                hint_defocus_angstrom = self._fit_zero_tilt_hint(
                    tilt_stack,
                    tilt_angles_deg,
                    context=context,
                    executable=executable,
                    temporary_root=temporary_root,
                    pixel_spacing_angstrom=pixel_spacing_angstrom,
                    optics=optics,
                    settings=settings,
                    command_log=command_log,
                )

            window = _positive_float_parameter(
                context,
                "ctffind_defocus_search_window_angstrom",
                default=DEFAULT_DEFOCUS_SEARCH_WINDOW_ANGSTROM,
            )
            minimum_defocus = max(hint_defocus_angstrom - window, 1.0)
            maximum_defocus = hint_defocus_angstrom + window

            # Interactive mode treats each slice of the stack as its own
            # micrograph, so the whole series is fitted in one call.
            summary_path, diagnostic_path = self._run_ctffind(
                input_path=tilt_stack.path.resolve(),
                output_stem=temporary_root / f"{manifest.tilt_series_id}_ctffind",
                executable=executable,
                temporary_root=temporary_root,
                pixel_spacing_angstrom=pixel_spacing_angstrom,
                optics=optics,
                settings=settings,
                minimum_defocus_angstrom=minimum_defocus,
                maximum_defocus_angstrom=maximum_defocus,
                command_log=command_log,
            )
            rows = parse_ctffind_summary(summary_path)
            if len(rows) != stack_shape[0]:
                raise RuntimeError(
                    f"CTFFIND fitted {len(rows)} views but the aligned stack has "
                    f"{stack_shape[0]}"
                )

            temporary_defocus = temporary_root / f"{manifest.tilt_series_id}.defocus"
            write_imod_defocus_file(
                temporary_defocus,
                rows,
                tilt_angles_deg,
                include_astigmatism=settings.include_astigmatism,
                include_phase_shift=settings.search_phase_shift,
            )

            _replace_output(temporary_defocus, paths.defocus)
            _replace_output(summary_path, paths.summary)
            if diagnostic_path.is_file():
                _replace_output(diagnostic_path, paths.diagnostic)
            for extra in temporary_root.glob("*_avrot.txt"):
                _replace_output(extra, paths.rotational_average)
            succeeded = True
        finally:
            _write_command_log(paths.log, command_log)
            if succeeded:
                shutil.rmtree(temporary_root, ignore_errors=True)

        defocus_um = [row.mean_defocus_angstrom / 10000.0 for row in rows]
        astigmatism_um = [row.astigmatism_angstrom / 10000.0 for row in rows]
        ctf_model_id = f"{manifest.tilt_series_id}:ctf_model:{self.name}{variant_suffix(context)}"

        warnings = list(settings.warnings)
        status = QcStatus.PASS
        report = CtfEstimationQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            input_stack_id=tilt_stack.id,
            ctf_model_id=ctf_model_id,
            num_tilts=stack_shape[0],
            voltage_kv=optics.voltage_kv,
            spherical_aberration_mm=optics.spherical_aberration_mm,
            amplitude_contrast=optics.amplitude_contrast,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            nyquist_angstrom=settings.nyquist_angstrom,
            max_resolution_angstrom=settings.max_resolution_angstrom,
            hint_defocus_um=hint_defocus_angstrom / 10000.0,
            mean_defocus_um=sum(defocus_um) / len(defocus_um),
            minimum_defocus_um=min(defocus_um),
            maximum_defocus_um=max(defocus_um),
            mean_astigmatism_um=sum(astigmatism_um) / len(astigmatism_um),
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        ctf_model_artifact = Artifact(
            id=ctf_model_id,
            kind=ArtifactKind.CTF_MODEL,
            path=paths.defocus,
            parent_ids=[tilt_stack.id, alignment.id],
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "input_stack_id": tilt_stack.id,
                "num_tilts": stack_shape[0],
                "voltage_kv": optics.voltage_kv,
                "spherical_aberration_mm": optics.spherical_aberration_mm,
                "amplitude_contrast": optics.amplitude_contrast,
                "pixel_spacing_angstrom": pixel_spacing_angstrom,
                "axis_angle_deg": alignment_result.tilt_axis_angle_deg,
                "astigmatism_frame": "aligned_stack",
                "includes_astigmatism": settings.include_astigmatism,
                "includes_phase_shift": settings.search_phase_shift,
                "hint_defocus_um": hint_defocus_angstrom / 10000.0,
                "summary_path": str(paths.summary),
                "diagnostic_path": str(paths.diagnostic),
                "command_log_path": str(paths.log),
                "qc_path": str(paths.report),
            },
            software_versions={"ctffind": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.defocus.stat().st_size,
        )
        qc_artifact = _qc_artifact(
            artifact_id=(
                f"{manifest.tilt_series_id}:qc:ctf_estimation:{self.name}"
                f"{variant_suffix(context)}"
            ),
            path=paths.report,
            parent_id=ctf_model_artifact.id,
            qc_type="ctf_estimation",
            backend=self.name,
            manifest=manifest,
            status=status,
        )
        return [ctf_model_artifact, qc_artifact]

    def _fit_zero_tilt_hint(
        self,
        tilt_stack: Artifact,
        tilt_angles_deg: Sequence[float],
        *,
        context: BackendContext,
        executable: Path,
        temporary_root: Path,
        pixel_spacing_angstrom: float,
        optics: _Optics,
        settings: _CtffindSettings,
        command_log: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
    ) -> float:
        """Fit the view nearest zero tilt to seed the narrow second-pass search."""

        zero_index = min(
            range(len(tilt_angles_deg)),
            key=lambda index: abs(tilt_angles_deg[index]),
        )
        view_path = self._extract_view(
            tilt_stack,
            zero_index,
            context=context,
            temporary_root=temporary_root,
            command_log=command_log,
        )
        summary_path, _ = self._run_ctffind(
            input_path=view_path,
            output_stem=temporary_root / "zero_tilt_ctffind",
            executable=executable,
            temporary_root=temporary_root,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            optics=optics,
            settings=settings,
            minimum_defocus_angstrom=settings.wide_minimum_defocus_angstrom,
            maximum_defocus_angstrom=settings.wide_maximum_defocus_angstrom,
            command_log=command_log,
        )
        return parse_ctffind_summary(summary_path)[0].mean_defocus_angstrom

    def _extract_view(
        self,
        tilt_stack: Artifact,
        view_index: int,
        *,
        context: BackendContext,
        temporary_root: Path,
        command_log: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
    ) -> Path:
        """Write one view of the stack as a standalone 2D MRC."""

        view_path = temporary_root / f"view_{view_index + 1:03d}.mrc"
        if view_path.is_file():
            return view_path
        newstack = resolve_imod_executable("newstack", context)
        command = [
            str(newstack),
            "-secs",
            str(view_index),
            str(tilt_stack.path.resolve()),
            str(view_path),
        ]
        result = self._command_runner(
            command,
            cwd=temporary_root,
            env=imod_environment(newstack, context),
        )
        command_log.append((command, result))
        if result.returncode != 0:
            raise RuntimeError(
                f"IMOD newstack failed to extract view {view_index} with exit code "
                f"{result.returncode}"
            )
        if not view_path.is_file():
            raise RuntimeError(f"IMOD newstack did not write {view_path}")
        return view_path

    def _run_ctffind(
        self,
        *,
        input_path: Path,
        output_stem: Path,
        executable: Path,
        temporary_root: Path,
        pixel_spacing_angstrom: float,
        optics: _Optics,
        settings: _CtffindSettings,
        minimum_defocus_angstrom: float,
        maximum_defocus_angstrom: float,
        command_log: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
    ) -> tuple[Path, Path]:
        diagnostic_path = output_stem.with_suffix(".mrc")
        summary_path = output_stem.with_suffix(".txt")
        dialogue = _ctffind_dialogue(
            input_path=input_path,
            input_is_a_stack=_ctffind_asks_about_movies(input_path),
            diagnostic_path=diagnostic_path,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            optics=optics,
            settings=settings,
            minimum_defocus_angstrom=minimum_defocus_angstrom,
            maximum_defocus_angstrom=maximum_defocus_angstrom,
        )
        stdin_text = _ctffind_stdin(dialogue)
        command = [str(executable)]
        result = self._command_runner(
            command,
            cwd=temporary_root,
            env=os.environ.copy(),
            stdin=stdin_text,
        )
        command_log.append(([*command, f"<<stdin>>\n{stdin_text}"], result))
        # Before the exit code, because a misaligned answer list aborts CTFFIND
        # with a generic "answer is not recognized" and the mismatch it echoed
        # just above is the part worth reporting.
        _verify_dialogue(result.stdout or "", dialogue)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "no process output").strip()
            raise RuntimeError(f"CTFFIND failed with exit code {result.returncode}: {detail}")
        if not summary_path.is_file():
            raise RuntimeError(f"CTFFIND did not write a summary file: {summary_path}")
        return summary_path, diagnostic_path


class ImodCtfPlotterEstimationBackend:
    """Estimate per-tilt CTF with IMOD ctfplotter."""

    name = "imod-ctfplotter"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def estimate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a CTF model artifact and its QC artifact."""

        stack_shape = _validate_final_aligned_stack(tilt_stack)
        alignment_result = _load_alignment(alignment, tilt_stack, manifest)
        pixel_spacing_angstrom = _voxel_spacing(tilt_stack)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        optics = _optics_parameters(context)

        expected_defocus_um = _expected_defocus_um(context, manifest)
        defocus_tol_nm = _positive_float_parameter(context, "ctf_defocus_tol_nm", default=200.0)
        left_defocus_tol_nm = _positive_float_parameter(
            context, "ctf_left_defocus_tol_nm", default=2000.0
        )
        right_defocus_tol_nm = _positive_float_parameter(
            context, "ctf_right_defocus_tol_nm", default=2000.0
        )
        tile_size = _positive_int_parameter(context, "ctf_tile_size", default=256)
        autofit_range_deg = _positive_float_parameter(
            context, "ctf_autofit_range_deg", default=16.0
        )
        autofit_step_deg = _positive_float_parameter(context, "ctf_autofit_step_deg", default=2.0)
        ps_resolution = _positive_int_parameter(context, "ctf_ps_resolution", default=101)

        paths = _estimation_paths(
            context.output_dir, manifest, backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)
        ctfplotter = resolve_imod_executable("ctfplotter", context)

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-ctfplotter-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_angles = temporary_root / f"{manifest.tilt_series_id}.tlt"
            temporary_defocus = temporary_root / f"{manifest.tilt_series_id}.defocus"

            write_tilt_angles_file(
                temporary_angles,
                [transform.tilt_angle_deg for transform in alignment_result.transforms],
            )

            command = [
                str(ctfplotter),
                "-InputStack",
                str(tilt_stack.path.resolve()),
                "-AngleFile",
                str(temporary_angles),
                "-DefocusFile",
                str(temporary_defocus),
                "-AxisAngle",
                f"{alignment_result.tilt_axis_angle_deg:.6f}",
                "-PixelSize",
                f"{pixel_spacing_angstrom / 10.0:.6f}",
                "-ExpectedDefocus",
                f"{expected_defocus_um * 1000.0:.6f}",
                "-AutoFitRangeAndStep",
                f"{autofit_range_deg:.6f},{autofit_step_deg:.6f}",
                "-Voltage",
                str(int(round(optics.voltage_kv))),
                "-SphericalAberration",
                f"{optics.spherical_aberration_mm:.6f}",
                "-AmplitudeContrast",
                f"{optics.amplitude_contrast:.6f}",
                "-DefocusTol",
                str(int(round(defocus_tol_nm))),
                "-PSResolution",
                str(ps_resolution),
                "-LeftDefTol",
                f"{left_defocus_tol_nm:.6f}",
                "-RightDefTol",
                f"{right_defocus_tol_nm:.6f}",
                "-tileSize",
                str(tile_size),
                "-SaveAndExit",
            ]
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(ctfplotter, context),
            )
            _write_command_log(paths.log, [(command, result)])
            if result.returncode != 0:
                raise RuntimeError(
                    f"IMOD ctfplotter failed with exit code {result.returncode}; see {paths.log}"
                )
            if not temporary_defocus.is_file():
                raise RuntimeError(
                    f"IMOD ctfplotter did not write a defocus file: {temporary_defocus}"
                )
            entries = read_imod_defocus_file(temporary_defocus)

            _replace_output(temporary_defocus, paths.defocus)

        defocus_um = [entry.mean_defocus_nm / 1000.0 for entry in entries]
        ctf_model_id = f"{manifest.tilt_series_id}:ctf_model:{self.name}{variant_suffix(context)}"
        nyquist_angstrom = 2.0 * pixel_spacing_angstrom
        report = CtfEstimationQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            input_stack_id=tilt_stack.id,
            ctf_model_id=ctf_model_id,
            num_tilts=stack_shape[0],
            voltage_kv=optics.voltage_kv,
            spherical_aberration_mm=optics.spherical_aberration_mm,
            amplitude_contrast=optics.amplitude_contrast,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            nyquist_angstrom=nyquist_angstrom,
            max_resolution_angstrom=nyquist_angstrom,
            axis_angle_deg=alignment_result.tilt_axis_angle_deg,
            mean_defocus_um=sum(defocus_um) / len(defocus_um),
            minimum_defocus_um=min(defocus_um),
            maximum_defocus_um=max(defocus_um),
            status=QcStatus.PASS,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        ctf_model_artifact = Artifact(
            id=ctf_model_id,
            kind=ArtifactKind.CTF_MODEL,
            path=paths.defocus,
            parent_ids=[tilt_stack.id, alignment.id],
            parameters={
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "input_stack_id": tilt_stack.id,
                "num_tilts": stack_shape[0],
                "voltage_kv": optics.voltage_kv,
                "spherical_aberration_mm": optics.spherical_aberration_mm,
                "amplitude_contrast": optics.amplitude_contrast,
                "pixel_spacing_angstrom": pixel_spacing_angstrom,
                "axis_angle_deg": alignment_result.tilt_axis_angle_deg,
                "astigmatism_frame": "aligned_stack",
                "command_log_path": str(paths.log),
                "qc_path": str(paths.report),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.defocus.stat().st_size,
        )
        qc_artifact = _qc_artifact(
            artifact_id=(
                f"{manifest.tilt_series_id}:qc:ctf_estimation:{self.name}"
                f"{variant_suffix(context)}"
            ),
            path=paths.report,
            parent_id=ctf_model_artifact.id,
            qc_type="ctf_estimation",
            backend=self.name,
            manifest=manifest,
            status=QcStatus.PASS,
        )
        return [ctf_model_artifact, qc_artifact]


class ImodCtfPhaseFlipBackend:
    """Phase-flip a tilt series with IMOD ctfphaseflip from any CTF model."""

    name = "imod-ctfphaseflip"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def correct(
        self,
        tilt_stack: Artifact,
        ctf_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a CTF-corrected aligned stack and its QC artifact."""

        stack_shape = _validate_final_aligned_stack(tilt_stack)
        _validate_ctf_model(ctf_model, tilt_stack, manifest)
        pixel_spacing_angstrom = _voxel_spacing(tilt_stack)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        optics = _consistent_optics(context, ctf_model)
        axis_angle_deg = _model_float_parameter(ctf_model, "axis_angle_deg")

        defocus_tol_nm = _positive_float_parameter(context, "ctf_defocus_tol_nm", default=200.0)
        interpolation_width = _positive_int_parameter(
            context, "ctf_interpolation_width", default=15
        )

        entries = read_imod_defocus_file(ctf_model.path)
        if len(entries) != stack_shape[0]:
            raise ValueError(
                f"CTF model covers {len(entries)} views but the aligned stack has "
                f"{stack_shape[0]}"
            )

        paths = _correction_paths(
            context.output_dir, manifest, backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)
        ctfphaseflip = resolve_imod_executable("ctfphaseflip", context)

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-ctfphaseflip-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_angles = temporary_root / f"{manifest.tilt_series_id}.tlt"
            temporary_corrected = temporary_root / f"{manifest.tilt_series_id}_ctf.st"

            write_tilt_angles_file(
                temporary_angles,
                [entry.angle_start_deg for entry in entries],
            )

            command = [
                str(ctfphaseflip),
                "-InputStack",
                str(tilt_stack.path.resolve()),
                "-AngleFile",
                str(temporary_angles),
                "-OutputFileName",
                str(temporary_corrected),
                "-DefocusFile",
                str(ctf_model.path.resolve()),
                "-Voltage",
                str(int(round(optics.voltage_kv))),
                "-SphericalAberration",
                f"{optics.spherical_aberration_mm:.6f}",
                "-DefocusTol",
                str(int(round(defocus_tol_nm))),
                "-PixelSize",
                f"{pixel_spacing_angstrom / 10.0:.6f}",
                "-AmplitudeContrast",
                f"{optics.amplitude_contrast:.6f}",
                "-InterpolationWidth",
                str(interpolation_width),
            ]
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(ctfphaseflip, context),
            )
            _write_command_log(paths.log, [(command, result)])
            if result.returncode != 0:
                raise RuntimeError(
                    f"IMOD ctfphaseflip failed with exit code {result.returncode}; "
                    f"see {paths.log}"
                )
            if not temporary_corrected.is_file():
                raise RuntimeError(
                    f"IMOD ctfphaseflip did not write a corrected stack: {temporary_corrected}"
                )

            info = validate_complete_mrc(temporary_corrected)
            if info.shape != stack_shape:
                raise ValueError(
                    f"CTF-corrected stack has shape {info.shape}, expected {stack_shape}"
                )

            _replace_output(temporary_corrected, paths.corrected_stack)
            _replace_output(temporary_angles, paths.tilt_angles)

        defocus_um = [entry.mean_defocus_nm / 1000.0 for entry in entries]
        corrected_stack_id = (
            f"{manifest.tilt_series_id}:aligned_tilt_stack:ctf_corrected:{self.name}"
            f"{variant_suffix(context)}"
        )
        report = CtfCorrectionQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            input_stack_id=tilt_stack.id,
            ctf_model_id=ctf_model.id,
            corrected_stack_id=corrected_stack_id,
            num_tilts=stack_shape[0],
            voltage_kv=optics.voltage_kv,
            spherical_aberration_mm=optics.spherical_aberration_mm,
            amplitude_contrast=optics.amplitude_contrast,
            pixel_spacing_angstrom=pixel_spacing_angstrom,
            axis_angle_deg=axis_angle_deg,
            mean_defocus_um=sum(defocus_um) / len(defocus_um),
            minimum_defocus_um=min(defocus_um),
            maximum_defocus_um=max(defocus_um),
            status=QcStatus.PASS,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        corrected_stack_artifact = Artifact(
            id=corrected_stack_id,
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=paths.corrected_stack,
            parent_ids=[tilt_stack.id, ctf_model.id],
            shape=info.shape,
            dtype=info.dtype,
            axis_order=AxisOrder.TYX,
            pixel_spacing_angstrom=tilt_stack.pixel_spacing_angstrom,
            binning=tilt_stack.binning,
            parameters={
                **{
                    key: value
                    for key, value in tilt_stack.parameters.items()
                    if key not in {"purpose", "backend"}
                },
                "purpose": "ctf_corrected",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "ctf_model_id": ctf_model.id,
                "ctf_estimation_backend": ctf_model.parameters.get("backend"),
                "command_log_path": str(paths.log),
                "qc_path": str(paths.report),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CACHE,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(paths.corrected_stack, paths.tilt_angles, paths.log),
        )
        qc_artifact = _qc_artifact(
            artifact_id=(
                f"{manifest.tilt_series_id}:qc:ctf_correction:{self.name}"
                f"{variant_suffix(context)}"
            ),
            path=paths.report,
            parent_id=corrected_stack_artifact.id,
            qc_type="ctf_correction",
            backend=self.name,
            manifest=manifest,
            status=QcStatus.PASS,
        )
        return [corrected_stack_artifact, qc_artifact]


def estimate_ctf_and_register(
    backend: CtfEstimationBackend,
    tilt_stack: Artifact,
    alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Estimate CTF and register the model plus QC artifacts."""

    artifacts = backend.estimate(tilt_stack, alignment, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


def correct_ctf_and_register(
    backend: CtfCorrectionBackend,
    tilt_stack: Artifact,
    ctf_model: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Phase-flip a tilt series and register the corrected stack plus QC artifacts."""

    artifacts = backend.correct(tilt_stack, ctf_model, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


@dataclass(frozen=True)
class _Optics:
    voltage_kv: float
    spherical_aberration_mm: float
    amplitude_contrast: float


@dataclass(frozen=True)
class _CtffindSettings:
    box_size: int
    min_resolution_angstrom: float
    max_resolution_angstrom: float
    nyquist_angstrom: float
    defocus_step_angstrom: float
    wide_minimum_defocus_angstrom: float
    wide_maximum_defocus_angstrom: float
    include_astigmatism: bool
    exhaustive_search: bool
    restrain_astigmatism_angstrom: float | None
    search_phase_shift: bool
    minimum_phase_shift_rad: float
    maximum_phase_shift_rad: float
    phase_shift_step_rad: float
    warnings: tuple[str, ...]


def _optics_parameters(context: BackendContext) -> _Optics:
    return _Optics(
        voltage_kv=_required_positive_float_parameter(context, "ctf_voltage_kv"),
        spherical_aberration_mm=_required_positive_float_parameter(
            context, "ctf_spherical_aberration_mm"
        ),
        amplitude_contrast=_bounded_float_parameter(
            context, "ctf_amplitude_contrast", default=0.07, minimum=0.0, maximum=1.0
        ),
    )


def _consistent_optics(context: BackendContext, ctf_model: Artifact) -> _Optics:
    """Use the optics recorded on the CTF model, rejecting conflicting overrides.

    ctfphaseflip must be given the same voltage, Cs, and amplitude contrast that
    were used to fit the CTF; silently correcting with different values produces
    a plausible-looking but wrong result.
    """

    recorded = _Optics(
        voltage_kv=_model_float_parameter(ctf_model, "voltage_kv"),
        spherical_aberration_mm=_model_float_parameter(ctf_model, "spherical_aberration_mm"),
        amplitude_contrast=_model_float_parameter(ctf_model, "amplitude_contrast"),
    )
    overrides = {
        "ctf_voltage_kv": recorded.voltage_kv,
        "ctf_spherical_aberration_mm": recorded.spherical_aberration_mm,
        "ctf_amplitude_contrast": recorded.amplitude_contrast,
    }
    for key, expected in overrides.items():
        supplied = context.parameters.get(key)
        if supplied is None:
            continue
        if isinstance(supplied, bool) or not isinstance(supplied, (int, float)):
            raise TypeError(f"context parameter {key!r} must be numeric")
        if not math.isclose(float(supplied), expected, rel_tol=1e-6):
            raise ValueError(
                f"{key}={float(supplied)!r} conflicts with the CTF model's recorded value "
                f"{expected!r}; correction must reuse the estimation optics"
            )
    return recorded


def _validate_ctf_model(
    ctf_model: Artifact,
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
) -> None:
    if ctf_model.kind is not ArtifactKind.CTF_MODEL:
        raise ValueError(f"expected CTF model artifact, got {ctf_model.kind}")
    if ctf_model.parameters.get("tilt_series_id") != manifest.tilt_series_id:
        raise ValueError("CTF model is for a different tilt series")
    if ctf_model.parameters.get("input_stack_id") != tilt_stack.id:
        raise ValueError(
            f"CTF model was fitted on {ctf_model.parameters.get('input_stack_id')!r}, "
            f"not on {tilt_stack.id!r}"
        )
    if not ctf_model.path.is_file():
        raise FileNotFoundError(f"CTF model file not found: {ctf_model.path}")
    recorded_spacing = ctf_model.parameters.get("pixel_spacing_angstrom")
    stack_spacing = tilt_stack.pixel_spacing_angstrom
    if (
        isinstance(recorded_spacing, (int, float))
        and stack_spacing is not None
        and not math.isclose(float(recorded_spacing), stack_spacing, rel_tol=1e-6)
    ):
        raise ValueError(
            f"CTF model was fitted at {float(recorded_spacing)} angstrom/pixel but the stack "
            f"is {stack_spacing} angstrom/pixel"
        )


def _ctffind_settings(
    context: BackendContext,
    *,
    pixel_spacing_angstrom: float,
) -> _CtffindSettings:
    """Resolve CTFFIND search settings, keeping them physically meaningful.

    The finest fittable spacing is the Nyquist limit, twice the pixel size. A
    binned tilt series can easily have a Nyquist limit coarser than a
    naive default, so the default tracks the actual pixel size and an explicit
    request finer than Nyquist is clamped with a warning rather than passed on.
    """

    warnings: list[str] = []
    nyquist_angstrom = 2.0 * pixel_spacing_angstrom

    requested_max = _optional_positive_float_parameter(
        context, "ctffind_max_resolution_angstrom"
    )
    max_resolution_angstrom = nyquist_angstrom if requested_max is None else requested_max
    if max_resolution_angstrom < nyquist_angstrom:
        warnings.append(
            f"requested max resolution {max_resolution_angstrom:.2f} A is finer than the "
            f"Nyquist limit {nyquist_angstrom:.2f} A at {pixel_spacing_angstrom:.3f} A/pixel; "
            "clamped to Nyquist"
        )
        max_resolution_angstrom = nyquist_angstrom

    requested_min = _optional_positive_float_parameter(
        context, "ctffind_min_resolution_angstrom"
    )
    min_resolution_angstrom = 30.0 if requested_min is None else requested_min
    if min_resolution_angstrom <= max_resolution_angstrom:
        adjusted = max_resolution_angstrom * 2.0
        warnings.append(
            f"min resolution {min_resolution_angstrom:.2f} A is not coarser than the max "
            f"resolution {max_resolution_angstrom:.2f} A; widened to {adjusted:.2f} A"
        )
        min_resolution_angstrom = adjusted

    if _bool_parameter(context, "ctffind_determine_sample_tilt", default=False):
        raise NotImplementedError(
            "CTFFIND5 sample-tilt determination adds follow-up prompts that this adapter "
            "does not answer yet; leave it disabled"
        )
    if _bool_parameter(context, "ctffind_determine_sample_thickness", default=False):
        raise NotImplementedError(
            "CTFFIND5 sample-thickness determination adds follow-up prompts that this "
            "adapter does not answer yet; leave it disabled"
        )

    return _CtffindSettings(
        box_size=_positive_int_parameter(context, "ctffind_box_size", default=512),
        min_resolution_angstrom=min_resolution_angstrom,
        max_resolution_angstrom=max_resolution_angstrom,
        nyquist_angstrom=nyquist_angstrom,
        defocus_step_angstrom=_positive_float_parameter(
            context, "ctffind_defocus_step_angstrom", default=100.0
        ),
        wide_minimum_defocus_angstrom=_positive_float_parameter(
            context, "ctffind_min_defocus_angstrom", default=5000.0
        ),
        wide_maximum_defocus_angstrom=_positive_float_parameter(
            context, "ctffind_max_defocus_angstrom", default=50000.0
        ),
        include_astigmatism=_bool_parameter(
            context, "ctffind_include_astigmatism", default=True
        ),
        exhaustive_search=_bool_parameter(
            context, "ctffind_exhaustive_search", default=False
        ),
        restrain_astigmatism_angstrom=_optional_positive_float_parameter(
            context, "ctffind_restrain_astigmatism_angstrom"
        ),
        search_phase_shift=_bool_parameter(context, "ctffind_search_phase_shift", default=False),
        minimum_phase_shift_rad=_float_parameter(
            context, "ctffind_min_phase_shift_rad", default=0.0
        ),
        maximum_phase_shift_rad=_float_parameter(
            context, "ctffind_max_phase_shift_rad", default=3.15
        ),
        phase_shift_step_rad=_positive_float_parameter(
            context, "ctffind_phase_shift_step_rad", default=0.2
        ),
        warnings=tuple(warnings),
    )


# Every question CTFFIND can ask interactively, taken from
# `DoInteractiveUserInput` in ctffind.cpp. Used only to tell prompt echoes apart
# from the rest of CTFFIND's output, so banner lines such as "Version : 5.0.2"
# are not mistaken for questions. "samnple" is CTFFIND's spelling, not a typo.
CTFFIND_QUESTIONS: frozenset[str] = frozenset(
    {
        "Input image file name",
        "Input is a movie (stack of frames)",
        "Number of frames to average together",
        "Output diagnostic image file name",
        "Pixel size",
        "Acceleration voltage",
        "Spherical aberration",
        "Amplitude contrast",
        "Size of amplitude spectrum to compute",
        "Minimum resolution",
        "Maximum resolution",
        "Minimum defocus",
        "Maximum defocus",
        "Defocus search step",
        "Do you know what astigmatism is present?",
        "Slower, more exhaustive search?",
        "Known astigmatism",
        "Known astigmatism angle",
        "Use a restraint on astigmatism?",
        "Expected (tolerated) astigmatism",
        "Find additional phase shift?",
        "Minimum phase shift (rad)",
        "Maximum phase shift (rad)",
        "Phase shift search step",
        "Determine sample tilt?",
        "Determine samnple thickness?",
        "Use brute force 1D search?",
        "Use 2D refinement?",
        "Low resolution limit for nodes",
        "High resolution limit for nodes",
        "Use rounded square for nodes?",
        "Downweight nodes?",
        "Do you want to set expert options?",
        "Resample micrograph if pixel size too small?",
        "Target pixel size after resampling",
        "Movie is dark-subtracted?",
        "Dark image filename",
        "Movie is gain-corrected?",
        "Gain image filename",
        "Correct Movie Mag. Distortion?",
        "Mag. distortion angle",
        "Mag. distortion major scale",
        "Mag. distortion minor scale",
        "Number of EER frames per image",
        "EER super resolution factor",
        "Do you already know the defocus?",
        "Known defocus 1",
        "Known defocus 2",
        "Known phase shift (radians)",
        "Desired number of parallel threads",
    }
)


def _ctffind_asks_about_movies(input_path: Path) -> bool:
    """Whether CTFFIND will ask its movie question for this input.

    It asks only when the file holds more than one slice, mirroring
    `input_file.ReturnZSize() > 1` in `DoInteractiveUserInput`. The answer
    changes the question sequence, so it is read from the file rather than
    assumed from which pass is running.
    """

    if input_path.suffix.lower() == ".eer":
        raise ValueError(
            f"CTFFIND treats EER input as a movie and asks a different set of "
            f"questions than this adapter answers: {input_path}"
        )
    return len(inspect_mrc_file(input_path).shape) == 3


def _ctffind_dialogue(
    *,
    input_path: Path,
    input_is_a_stack: bool,
    diagnostic_path: Path,
    pixel_spacing_angstrom: float,
    optics: _Optics,
    settings: _CtffindSettings,
    minimum_defocus_angstrom: float,
    maximum_defocus_angstrom: float,
) -> tuple[tuple[str, str], ...]:
    """Pair every question CTFFIND will ask with the answer to send it.

    Questions and answers are built together so the verification list cannot
    drift from the answer list. Each branch mirrors a conditional in
    `DoInteractiveUserInput`.
    """

    dialogue: list[tuple[str, str]] = [("Input image file name", str(input_path))]
    if input_is_a_stack:
        # No means "process each slice separately", which is what a tilt series
        # needs. Yes would average the views together into one micrograph.
        dialogue.append(("Input is a movie (stack of frames)", "no"))
    dialogue += [
        ("Output diagnostic image file name", str(diagnostic_path)),
        ("Pixel size", f"{pixel_spacing_angstrom:.6f}"),
        ("Acceleration voltage", f"{optics.voltage_kv:.6f}"),
        ("Spherical aberration", f"{optics.spherical_aberration_mm:.6f}"),
        ("Amplitude contrast", f"{optics.amplitude_contrast:.6f}"),
        ("Size of amplitude spectrum to compute", str(settings.box_size)),
        ("Minimum resolution", f"{settings.min_resolution_angstrom:.6f}"),
        ("Maximum resolution", f"{settings.max_resolution_angstrom:.6f}"),
        ("Minimum defocus", f"{minimum_defocus_angstrom:.6f}"),
        ("Maximum defocus", f"{maximum_defocus_angstrom:.6f}"),
        ("Defocus search step", f"{settings.defocus_step_angstrom:.6f}"),
        ("Do you know what astigmatism is present?", "no"),
        ("Slower, more exhaustive search?", "yes" if settings.exhaustive_search else "no"),
    ]
    if settings.restrain_astigmatism_angstrom is not None:
        dialogue.append(("Use a restraint on astigmatism?", "yes"))
        dialogue.append(
            (
                "Expected (tolerated) astigmatism",
                f"{settings.restrain_astigmatism_angstrom:.6f}",
            )
        )
    else:
        dialogue.append(("Use a restraint on astigmatism?", "no"))
    if settings.search_phase_shift:
        dialogue += [
            ("Find additional phase shift?", "yes"),
            ("Minimum phase shift (rad)", f"{settings.minimum_phase_shift_rad:.6f}"),
            ("Maximum phase shift (rad)", f"{settings.maximum_phase_shift_rad:.6f}"),
            ("Phase shift search step", f"{settings.phase_shift_step_rad:.6f}"),
        ]
    else:
        dialogue.append(("Find additional phase shift?", "no"))
        # Tilt determination is offered only without an additional phase shift.
        dialogue.append(("Determine sample tilt?", "no"))
    dialogue.append(("Determine samnple thickness?", "no"))
    dialogue.append(("Do you want to set expert options?", "no"))
    return tuple(dialogue)


def _ctffind_stdin(dialogue: Sequence[tuple[str, str]]) -> str:
    """Render a dialogue as the answer-per-line text CTFFIND reads from stdin."""

    return "".join(f"{answer}\n" for _, answer in dialogue)


_PROMPT_ECHO = re.compile(r"^(?P<label>\S.*?)\s{2,}:[ \t]*(?P<value>.*?)\s*$", re.MULTILINE)
_PROMPT_DEFAULT = re.compile(r"\s*\[[^\]]*\]\s*$")


def _questions_asked(output: str) -> list[tuple[str, str]]:
    """Return the questions CTFFIND echoed, in order, with the values it read."""

    asked: list[tuple[str, str]] = []
    for match in _PROMPT_ECHO.finditer(output):
        prompt = _PROMPT_DEFAULT.sub("", match.group("label")).strip()
        if prompt in CTFFIND_QUESTIONS:
            asked.append((prompt, match.group("value")))
    return asked


def _verify_dialogue(output: str, dialogue: Sequence[tuple[str, str]]) -> None:
    """Fail with the mismatch when the questions asked are not the ones answered.

    A positional answer list is only as good as its picture of the question
    sequence, and getting that wrong is usually quiet: an answer meant for one
    question is often a valid value for its neighbour. CTFFIND echoes each
    question as it reads the answer, so the divergence is recoverable.
    """

    asked = _questions_asked(output)
    for index, (expected, intended) in enumerate(dialogue):
        if index >= len(asked):
            return
        prompt, received = asked[index]
        if prompt != expected:
            raise RuntimeError(
                f"CTFFIND asked {prompt!r} as question {index + 1} and got "
                f"{intended!r}, which this adapter meant for {expected!r}: the "
                "question sequence differs from the expected one, so every "
                "later answer is displaced too"
            )
        if received and received != intended and not _numerically_equal(received, intended):
            raise RuntimeError(
                f"CTFFIND read {received!r} for {expected!r} but was sent {intended!r}"
            )


def _numerically_equal(left: str, right: str) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
    except ValueError:
        return False


@dataclass(frozen=True)
class _EstimationPaths:
    output_dir: Path
    defocus: Path
    summary: Path
    diagnostic: Path
    rotational_average: Path
    log: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.defocus, self.summary, self.diagnostic, self.log, self.report)


def _estimation_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    label: str | None = None,
) -> _EstimationPaths:
    ctf_dir = output_root / "ctf" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "ctf"
    prefix = f"{manifest.tilt_series_id}_{backend_name}" + (f"_{label}" if label else "")
    return _EstimationPaths(
        output_dir=ctf_dir,
        defocus=ctf_dir / f"{prefix}.defocus",
        summary=ctf_dir / f"{prefix}_ctffind.txt",
        diagnostic=ctf_dir / f"{prefix}_diagnostic.mrc",
        rotational_average=ctf_dir / f"{prefix}_avrot.txt",
        log=ctf_dir / f"{prefix}.log",
        report=qc_dir / f"{prefix}_estimation_qc.json",
    )


@dataclass(frozen=True)
class _CorrectionPaths:
    output_dir: Path
    corrected_stack: Path
    tilt_angles: Path
    log: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.corrected_stack, self.tilt_angles, self.log, self.report)


def _correction_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    label: str | None = None,
) -> _CorrectionPaths:
    ctf_dir = output_root / "ctf" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "ctf"
    prefix = f"{manifest.tilt_series_id}_{backend_name}" + (f"_{label}" if label else "")
    return _CorrectionPaths(
        output_dir=ctf_dir,
        corrected_stack=ctf_dir / f"{prefix}_ctf.st",
        tilt_angles=ctf_dir / f"{prefix}.tlt",
        log=ctf_dir / f"{prefix}.log",
        report=qc_dir / f"{prefix}_correction_qc.json",
    )


def _qc_artifact(
    *,
    artifact_id: str,
    path: Path,
    parent_id: str,
    qc_type: str,
    backend: str,
    manifest: TiltSeriesManifest,
    status: QcStatus,
) -> Artifact:
    return Artifact(
        id=artifact_id,
        kind=ArtifactKind.QC,
        path=path,
        parent_ids=[parent_id],
        parameters={
            "qc_type": qc_type,
            "backend": backend,
            "tilt_series_id": manifest.tilt_series_id,
            "status": status.value,
        },
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=path.stat().st_size,
    )


def _validate_final_aligned_stack(tilt_stack: Artifact) -> tuple[int, int, int]:
    if tilt_stack.kind is not ArtifactKind.ALIGNED_TILT_STACK:
        raise ValueError(f"expected aligned tilt stack artifact, got {tilt_stack.kind}")
    if tilt_stack.parameters.get("purpose") != "final_alignment":
        raise ValueError("CTF work requires the plain final fine-aligned stack")
    if tilt_stack.parameters.get("alignment_stage") != "fine":
        raise ValueError("final aligned stack must record alignment_stage='fine'")
    if not tilt_stack.path.is_file():
        raise FileNotFoundError(f"aligned tilt stack not found: {tilt_stack.path}")
    info = validate_complete_mrc(tilt_stack.path)
    if len(info.shape) != 3:
        raise ValueError(f"expected aligned tilt stack shape (tilts, y, x), got {info.shape}")
    shape = info.shape
    if tilt_stack.shape is not None and tuple(tilt_stack.shape) != shape:
        raise ValueError(f"aligned stack artifact shape {tilt_stack.shape} does not match {shape}")
    return shape


def _load_alignment(
    alignment: Artifact,
    tilt_stack: Artifact,
    manifest: TiltSeriesManifest,
) -> TiltAlignment:
    if alignment.kind is not ArtifactKind.ALIGNMENT:
        raise ValueError(f"expected alignment artifact, got {alignment.kind}")
    result = TiltAlignment.model_validate_json(alignment.path.read_text())
    if result.tilt_series_id != manifest.tilt_series_id:
        raise ValueError(
            f"alignment is for {result.tilt_series_id}, expected {manifest.tilt_series_id}"
        )
    if alignment.id not in tilt_stack.parent_ids:
        raise ValueError(f"aligned stack {tilt_stack.id} is not derived from {alignment.id}")
    if result.stage != "fine" or alignment.parameters.get("stage") != "fine":
        raise ValueError("CTF work requires fine alignment")
    return result


def _voxel_spacing(tilt_stack: Artifact) -> float:
    pixel_spacing = tilt_stack.pixel_spacing_angstrom
    if pixel_spacing is None:
        raise ValueError("aligned stack voxel spacing is required for CTF work")
    if not math.isfinite(pixel_spacing) or pixel_spacing <= 0:
        raise ValueError("aligned stack voxel spacing must be finite and positive")
    return pixel_spacing


def _expected_defocus_um(context: BackendContext, manifest: TiltSeriesManifest) -> float:
    override = context.parameters.get("ctf_expected_defocus_um")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, (int, float)):
            raise TypeError("context parameter 'ctf_expected_defocus_um' must be numeric")
        value = float(override)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("context parameter 'ctf_expected_defocus_um' must be positive")
        return value
    candidates = [image for image in manifest.images if image.defocus_um is not None]
    if not candidates:
        raise ValueError(
            "no nominal defocus recorded in the manifest; "
            "pass ctf_expected_defocus_um explicitly"
        )
    closest = min(candidates, key=lambda image: abs(image.tilt_angle_deg))
    nominal = closest.defocus_um
    if nominal is None or not math.isfinite(nominal) or nominal <= 0.0:
        raise ValueError("manifest nominal defocus must be finite and positive")
    return nominal


def _explicit_expected_defocus_angstrom(context: BackendContext) -> float | None:
    value = context.parameters.get("ctf_expected_defocus_um")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("context parameter 'ctf_expected_defocus_um' must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("context parameter 'ctf_expected_defocus_um' must be positive")
    return normalized * 10000.0


def _model_float_parameter(artifact: Artifact, key: str) -> float:
    value = artifact.parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"CTF model artifact is missing a numeric {key!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"CTF model artifact {key!r} must be finite")
    return normalized


def _resolve_ctffind_executable(context: BackendContext) -> Path:
    configured = context.parameters.get("ctffind_executable")
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError("context parameter 'ctffind_executable' must be a path string")
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"CTFFIND executable not found: {candidate}")
        return candidate.resolve()
    for name in ("ctffind5", "ctffind"):
        resolved = shutil.which(name)
        if resolved is not None:
            return Path(resolved)
    raise FileNotFoundError(
        "CTFFIND executable not found; pass --ctffind-executable or add it to PATH"
    )


def _write_command_log(
    path: Path,
    command_results: Sequence[tuple[Sequence[str], subprocess.CompletedProcess[str]]],
) -> None:
    sections = []
    for command, result in command_results:
        sections.append(
            f"$ {shlex.join(str(token) for token in command)}\n\n"
            f"[stdout]\n{result.stdout or ''}\n"
            f"[stderr]\n{result.stderr or ''}\n"
            f"[exit_code]\n{result.returncode}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _replace_output(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _remove_path(destination)
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"CTF outputs already exist: {joined}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _positive_int_parameter(context: BackendContext, key: str, *, default: int) -> int:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"context parameter {key!r} must be an int")
    if value < 1:
        raise ValueError(f"context parameter {key!r} must be at least 1")
    return value


def _nonnegative_float_parameter(
    context: BackendContext, key: str, *, default: float
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and not negative")
    return normalized


def _positive_float_parameter(context: BackendContext, key: str, *, default: float) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and positive")
    return normalized


def _optional_positive_float_parameter(context: BackendContext, key: str) -> float | None:
    if context.parameters.get(key) is None:
        return None
    return _positive_float_parameter(context, key, default=1.0)


def _float_parameter(context: BackendContext, key: str, *, default: float) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"context parameter {key!r} must be finite")
    return normalized


def _required_positive_float_parameter(context: BackendContext, key: str) -> float:
    value = context.parameters.get(key)
    if value is None:
        raise TypeError(f"context parameter {key!r} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and positive")
    return normalized


def _bounded_float_parameter(
    context: BackendContext,
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = context.parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(f"context parameter {key!r} must be between {minimum} and {maximum}")
    return normalized


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths if path.exists())
