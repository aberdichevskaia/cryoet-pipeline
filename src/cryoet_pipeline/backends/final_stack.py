from __future__ import annotations

import shlex
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import (
    CommandRunner,
    imod_environment,
    resolve_imod_executable,
    run_command,
    write_binned_mrc_stack,
    write_tilt_angles_file,
)
from cryoet_pipeline.backends.protocols import BackendContext, FinalAlignedStackBackend
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    RetentionPolicy,
    StorageRole,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.mrc_validation import validate_complete_mrc


class ImodFinalAlignedStackBackend:
    """Apply canonical fine transforms with IMOD Newstack."""

    name = "imod_newstack_final"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def build(
        self,
        tilt_stack: Artifact,
        fine_alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Create a binned, tilt-axis-vertical final aligned stack."""

        stack_shape, alignment = _validate_inputs(
            tilt_stack,
            fine_alignment,
            manifest,
        )
        output_binning = _positive_int_parameter(
            context,
            "output_binning",
            default=8,
        )
        if output_binning > min(stack_shape[1:]):
            raise ValueError(
                f"output binning {output_binning} exceeds image shape "
                f"{stack_shape[1:]}"
            )
        overwrite = _bool_parameter(context, "overwrite", default=False)
        paths = _final_stack_paths(
            context.output_dir,
            manifest,
            output_binning=output_binning,
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        newstack = resolve_imod_executable("newstack", context)

        transform_by_z = {
            transform.z_value: transform for transform in alignment.transforms
        }
        source_index_by_z = {
            image.z_value: index for index, image in enumerate(manifest.images)
        }
        ordered_z_values = [transform.z_value for transform in alignment.transforms]
        try:
            source_indices = [
                source_index_by_z[z_value] for z_value in ordered_z_values
            ]
        except KeyError as exc:
            raise ValueError(
                f"fine alignment refers to unknown manifest z={exc.args[0]}"
            ) from exc

        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-final-stack-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            binned_input = temporary_root / "input_bin.mrc"
            scaled_transform = temporary_root / "final_bin.xf"
            final_stack = temporary_root / "final.st"
            pixel_spacing = _fine_alignment_pixel_spacing(
                fine_alignment,
                tilt_stack,
            )
            write_binned_mrc_stack(
                tilt_stack.path,
                binned_input,
                binning=output_binning,
                pixel_spacing_angstrom=pixel_spacing,
                section_indices=source_indices,
            )
            _write_scaled_transforms(
                scaled_transform,
                ordered_z_values,
                transform_by_z,
                binning=output_binning,
            )
            output_y = _ceil_div(stack_shape[1], output_binning)
            output_x = _ceil_div(stack_shape[2], output_binning)
            command = [
                str(newstack),
                "-input",
                str(binned_input),
                "-output",
                str(final_stack),
                "-xform",
                str(scaled_transform),
                "-size",
                f"{output_x},{output_y}",
                "-mode",
                "2",
                "-taper",
                "1,0",
                "-origin",
            ]
            result = self._command_runner(
                command,
                cwd=temporary_root,
                env=imod_environment(newstack, context),
            )
            _write_command_log(paths.newstack_log, command, result)
            if result.returncode != 0:
                raise RuntimeError(
                    f"IMOD newstack failed with exit code {result.returncode}; "
                    f"see {paths.newstack_log}"
                )
            if not final_stack.is_file():
                raise RuntimeError(
                    f"IMOD newstack did not write final aligned stack: "
                    f"{final_stack}"
                )
            _replace_file(final_stack, paths.stack)

        paths.transform.write_text(
            "".join(
                _format_transform(transform, shift_scale=1.0 / output_binning)
                for transform in alignment.transforms
            )
        )
        write_tilt_angles_file(
            paths.tilt_file,
            [transform.tilt_angle_deg for transform in alignment.transforms],
        )
        info = validate_complete_mrc(paths.stack)
        expected_shape = (
            len(alignment.transforms),
            _ceil_div(stack_shape[1], output_binning),
            _ceil_div(stack_shape[2], output_binning),
        )
        if info.shape != expected_shape:
            raise ValueError(
                f"final aligned stack has shape {info.shape}, "
                f"expected {expected_shape}"
            )
        return Artifact(
            id=f"{manifest.tilt_series_id}:aligned_tilt_stack:final",
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=paths.stack,
            parent_ids=[tilt_stack.id, fine_alignment.id],
            shape=info.shape,
            dtype=info.dtype,
            axis_order=AxisOrder.TYX,
            pixel_spacing_angstrom=(
                pixel_spacing * output_binning
                if pixel_spacing is not None
                else None
            ),
            binning=output_binning,
            parameters={
                "purpose": "final_alignment",
                "tilt_series_id": manifest.tilt_series_id,
                "alignment_stage": "fine",
                "raw_pixel_spacing_angstrom": pixel_spacing,
                "output_pixel_spacing_angstrom": (
                    pixel_spacing * output_binning
                    if pixel_spacing is not None
                    else None
                ),
                "included_z_values": ordered_z_values,
                "excluded_z_values": alignment.excluded_z_values,
                "order": "tilt_angle_ascending",
                "imod_xf_path": str(paths.transform),
                "tilt_file_path": str(paths.tilt_file),
                "newstack_log_path": str(paths.newstack_log),
            },
            software_versions={"imod": "external"},
            storage_role=StorageRole.CACHE,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=_paths_size_bytes(
                paths.stack,
                paths.transform,
                paths.tilt_file,
                paths.newstack_log,
            ),
        )


def build_final_stack_and_register(
    backend: FinalAlignedStackBackend,
    tilt_stack: Artifact,
    fine_alignment: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> Artifact:
    """Build and register a final aligned tilt stack."""

    artifact = backend.build(tilt_stack, fine_alignment, manifest, context)
    registry.add(artifact, replace=replace_existing)
    return artifact


class _FinalStackPaths:
    def __init__(
        self,
        *,
        output_dir: Path,
        stack: Path,
        transform: Path,
        tilt_file: Path,
        newstack_log: Path,
    ) -> None:
        self.output_dir = output_dir
        self.stack = stack
        self.transform = transform
        self.tilt_file = tilt_file
        self.newstack_log = newstack_log

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (
            self.stack,
            self.transform,
            self.tilt_file,
            self.newstack_log,
        )


def _final_stack_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    output_binning: int,
) -> _FinalStackPaths:
    output_dir = output_root / "aligned" / manifest.tilt_series_id
    prefix = f"{manifest.tilt_series_id}_final_bin{output_binning}"
    return _FinalStackPaths(
        output_dir=output_dir,
        stack=output_dir / f"{prefix}.st",
        transform=output_dir / f"{prefix}.xf",
        tilt_file=output_dir / f"{prefix}.tlt",
        newstack_log=output_dir / f"{prefix}_newstack.log",
    )


def _validate_inputs(
    tilt_stack: Artifact,
    fine_alignment: Artifact,
    manifest: TiltSeriesManifest,
) -> tuple[tuple[int, int, int], TiltAlignment]:
    if tilt_stack.kind is not ArtifactKind.TILT_STACK:
        raise ValueError(f"expected tilt stack artifact, got {tilt_stack.kind}")
    if not tilt_stack.path.exists():
        raise FileNotFoundError(f"tilt stack not found: {tilt_stack.path}")
    if tilt_stack.shape is None or len(tilt_stack.shape) != 3:
        raise ValueError("tilt stack artifact must record shape (tilts, y, x)")
    stack_shape = tuple(int(axis_size) for axis_size in tilt_stack.shape)
    if stack_shape[0] != manifest.num_tilts:
        raise ValueError(
            f"tilt stack has {stack_shape[0]} images, "
            f"manifest has {manifest.num_tilts}"
        )
    if fine_alignment.kind is not ArtifactKind.ALIGNMENT:
        raise ValueError(f"expected alignment artifact, got {fine_alignment.kind}")
    if fine_alignment.parameters.get("stage") != "fine":
        raise ValueError("final aligned stack requires fine alignment")
    if not fine_alignment.path.is_file():
        raise FileNotFoundError(
            f"fine alignment result not found: {fine_alignment.path}"
        )
    result = TiltAlignment.model_validate_json(fine_alignment.path.read_text())
    if result.stage != "fine":
        raise ValueError("canonical alignment stage must be fine")
    if result.tilt_series_id != manifest.tilt_series_id:
        raise ValueError("fine alignment is for a different tilt series")
    if not result.transforms:
        raise ValueError("fine alignment contains no transforms")
    return (
        (stack_shape[0], stack_shape[1], stack_shape[2]),
        result,
    )


def _write_scaled_transforms(
    path: Path,
    ordered_z_values: Sequence[int],
    transforms_by_z: dict[int, AlignmentTransform],
    *,
    binning: int,
) -> None:
    path.write_text(
        "".join(
            _format_transform(
                transforms_by_z[z_value],
                shift_scale=1.0 / binning,
            )
            for z_value in ordered_z_values
        )
    )


def _fine_alignment_pixel_spacing(
    fine_alignment: Artifact,
    tilt_stack: Artifact,
) -> float | None:
    value = fine_alignment.parameters.get("raw_pixel_spacing_angstrom")
    if value is None:
        return tilt_stack.pixel_spacing_angstrom
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("fine alignment raw pixel spacing must be numeric")
    normalized = float(value)
    if normalized <= 0.0:
        raise ValueError("fine alignment raw pixel spacing must be positive")
    return normalized


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _format_transform(
    transform: AlignmentTransform,
    *,
    shift_scale: float,
) -> str:
    return (
        f"{transform.a11:.7f} {transform.a12:.7f} "
        f"{transform.a21:.7f} {transform.a22:.7f} "
        f"{transform.shift_x_px * shift_scale:.3f} "
        f"{transform.shift_y_px * shift_scale:.3f}\n"
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


def _replace_file(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"final aligned-stack outputs already exist: {joined}")


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


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _paths_size_bytes(*paths: Path) -> int:
    return sum(path.stat().st_size for path in paths)
