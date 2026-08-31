from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import CommandRunner, run_command
from cryoet_pipeline.backends.command_log import write_command_log
from cryoet_pipeline.backends.protocols import (
    BackendContext,
    TemplatePreparationBackend,
    variant_label,
    variant_suffix,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TemplatePreparationQc,
    TiltSeriesManifest,
)


class PytomTemplateBackend:
    """Build a matching template and mask with pytom-match-pick's helper tools.

    `pytom_create_template.py` resamples a reference map to the tomogram's voxel
    size; `pytom_create_mask.py` builds the matching spherical mask. The two
    must share a box size, which is checked here rather than left to fail
    obscurely inside template matching.
    """

    name = "pytom"

    def __init__(self, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner or run_command

    def prepare(
        self,
        source_map: Path,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a template, a template mask, and a QC artifact."""

        if not source_map.is_file():
            raise FileNotFoundError(f"reference map not found: {source_map}")
        _validate_tomogram(tomogram, manifest)
        output_voxel_size = _tomogram_voxel_size(tomogram)
        overwrite = _bool_parameter(context, "overwrite", default=False)
        invert = _bool_parameter(context, "template_invert", default=True)
        center = _bool_parameter(context, "template_center", default=False)
        mirror = _bool_parameter(context, "template_mirror", default=False)
        low_pass = _optional_positive_float_parameter(context, "template_low_pass_angstrom")
        box_size = _optional_positive_int_parameter(context, "template_box_size_px")
        input_voxel_size = _optional_positive_float_parameter(
            context, "template_input_voxel_size_angstrom"
        )
        mask_sigma = _optional_positive_float_parameter(context, "template_mask_sigma_px")
        emdb_id = _optional_string_parameter(context, "template_emdb_id")

        paths = _template_paths(
            context.output_dir,
            manifest,
            backend_name=self.name,
            label=variant_label(context),
        )
        _require_available_paths(paths.outputs, overwrite=overwrite)
        paths.output_dir.mkdir(parents=True, exist_ok=True)
        paths.report.parent.mkdir(parents=True, exist_ok=True)

        create_template = _resolve_pytom_executable(
            context, "pytom_create_template_executable", default_name="pytom_create_template.py"
        )
        create_mask = _resolve_pytom_executable(
            context, "pytom_create_mask_executable", default_name="pytom_create_mask.py"
        )

        command_log: list[tuple[Sequence[str], subprocess.CompletedProcess[str]]] = []
        with tempfile.TemporaryDirectory(
            prefix=f".{manifest.tilt_series_id}-template-",
            dir=paths.output_dir.resolve(),
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_template = temporary_root / "template.mrc"
            temporary_mask = temporary_root / "mask.mrc"

            template_command = [
                str(create_template),
                "-i",
                str(source_map.resolve()),
                "-o",
                str(temporary_template),
                "--output-voxel-size-angstrom",
                f"{output_voxel_size:.6f}",
            ]
            if input_voxel_size is not None:
                template_command.extend(
                    ["--input-voxel-size-angstrom", f"{input_voxel_size:.6f}"]
                )
            if low_pass is not None:
                template_command.extend(["--low-pass", f"{low_pass:.6f}"])
            if box_size is not None:
                template_command.extend(["-b", str(box_size)])
            if center:
                template_command.append("--center")
            if invert:
                template_command.append("--invert")
            if mirror:
                template_command.append("--mirror")

            result = self._command_runner(
                template_command, cwd=temporary_root, env=os.environ.copy()
            )
            command_log.append((template_command, result))
            if result.returncode != 0:
                write_command_log(paths.log, *command_log)
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"pytom_create_template.py failed with exit code {result.returncode}: "
                    f"{detail}; see {paths.log}"
                )
            if not temporary_template.is_file():
                write_command_log(paths.log, *command_log)
                raise RuntimeError(
                    f"pytom_create_template.py did not write {temporary_template}"
                )

            template_statistics = _inspect_volume(temporary_template)
            actual_box_size = template_statistics.box_size
            mask_radii, mask_radius_derived = _mask_radii_px(
                context,
                voxel_size_angstrom=output_voxel_size,
                box_size=actual_box_size,
                measured_extent_px=template_statistics.density_extent_px,
            )

            mask_command = [
                str(create_mask),
                "-b",
                str(actual_box_size),
                "-o",
                str(temporary_mask),
                "-r",
                f"{mask_radii.x:.6f}",
                "--voxel-size",
                f"{output_voxel_size:.6f}",
            ]
            if not mask_radii.is_sphere:
                # pytom reads -r as the x semi-axis and these as y and z.
                mask_command.extend(
                    [
                        "--radius-minor1",
                        f"{mask_radii.y:.6f}",
                        "--radius-minor2",
                        f"{mask_radii.z:.6f}",
                    ]
                )
            if mask_sigma is not None:
                mask_command.extend(["-s", f"{mask_sigma:.6f}"])

            result = self._command_runner(mask_command, cwd=temporary_root, env=os.environ.copy())
            command_log.append((mask_command, result))
            write_command_log(paths.log, *command_log)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no process output").strip()
                raise RuntimeError(
                    f"pytom_create_mask.py failed with exit code {result.returncode}: "
                    f"{detail}; see {paths.log}"
                )
            if not temporary_mask.is_file():
                raise RuntimeError(f"pytom_create_mask.py did not write {temporary_mask}")

            mask_statistics = _inspect_volume(temporary_mask)
            if mask_statistics.box_size != actual_box_size:
                raise RuntimeError(
                    f"mask box size {mask_statistics.box_size} does not match the template's "
                    f"{actual_box_size}; template matching requires them to agree"
                )

            _replace_output(temporary_template, paths.template)
            _replace_output(temporary_mask, paths.mask)

        warnings: list[str] = []
        status = QcStatus.PASS
        if not _contrast_matches_inversion(template_statistics, inverted=invert):
            status = QcStatus.WARNING
            expected = "dark (negative)" if invert else "bright (positive)"
            warnings.append(
                f"template density does not look {expected} after applying invert={invert}; "
                "check that the contrast matches the tomogram, since a mismatch makes "
                "template matching quietly find nothing"
            )
        if mask_radius_derived:
            # A size the pipeline chose is worth a look even when it looks fine,
            # and a warning nobody sees in the status is a warning nobody reads.
            # Quantified rather than merely announced, because the derivation
            # applies two margins and only one of them is visible: the extent is
            # measured off a low-passed map, whose density reaches past the
            # particle before the factor below is applied on top.
            status = QcStatus.WARNING
            widest = max(template_statistics.density_extent_px)
            warnings.append(
                f"no mask size was given, so it was derived from the map: the "
                f"density reaches {widest:.1f} px at its widest, and a "
                f"{_DERIVED_MARGIN} margin on that gives a radius of "
                f"{mask_radii.x:.1f} px — a mask "
                f"{mask_radii.x * 2.0 * output_voxel_size:.0f} angstrom across. "
                "Divide that by the particle's known diameter for the overhang; a "
                "tight mask wants 10-20%, and the two margins compound because "
                "the measured extent already exceeds the particle"
            )
        shape_warnings = _mask_versus_density_warnings(
            mask_radii,
            template_statistics,
            voxel_size_angstrom=output_voxel_size,
        )
        if shape_warnings:
            status = QcStatus.WARNING
            warnings.extend(shape_warnings)

        template_id = (
            f"{manifest.tilt_series_id}:template:{self.name}"
            f"{variant_suffix(context)}"

        )
        mask_id = (
            f"{manifest.tilt_series_id}:template_mask:{self.name}"
            f"{variant_suffix(context)}"
        )
        report = TemplatePreparationQc(
            tilt_series_id=manifest.tilt_series_id,
            backend=self.name,
            source_map_path=str(source_map.resolve()),
            emdb_id=emdb_id,
            template_id=template_id,
            template_mask_id=mask_id,
            input_voxel_size_angstrom=(
                input_voxel_size
                if input_voxel_size is not None
                else _source_voxel_size(source_map)
            ),
            output_voxel_size_angstrom=output_voxel_size,
            box_size_px=actual_box_size,
            inverted=invert,
            low_pass_angstrom=low_pass,
            mask_radius_px=mask_radii.x,
            mask_radius_derived_from_density=mask_radius_derived,
            mask_derived_margin=_DERIVED_MARGIN if mask_radius_derived else None,
            mask_sigma_px=mask_sigma,
            template_minimum=template_statistics.minimum,
            template_maximum=template_statistics.maximum,
            template_median=template_statistics.median,
            density_extent_x_px=template_statistics.density_extent_px[0],
            density_extent_y_px=template_statistics.density_extent_px[1],
            density_extent_z_px=template_statistics.density_extent_px[2],
            status=status,
            warnings=warnings,
        )
        _write_json(paths.report, report.model_dump(mode="json"))

        shared_parameters = {
            "backend": self.name,
            "tilt_series_id": manifest.tilt_series_id,
            "tomogram_id": tomogram.id,
            "source_map_path": str(source_map.resolve()),
            "emdb_id": emdb_id,
            "box_size_px": actual_box_size,
            "inverted": invert,
            # Recorded on the artifact, not only in QC, so a later step can ask
            # how big the reference actually is without reopening the map.
            "density_extent_px": list(template_statistics.density_extent_px),
            "command_log_path": str(paths.log),
            "qc_path": str(paths.report),
        }
        template_artifact = Artifact(
            id=template_id,
            variant=variant_label(context),
            kind=ArtifactKind.TEMPLATE,
            path=paths.template,
            parent_ids=[tomogram.id],
            shape=(actual_box_size, actual_box_size, actual_box_size),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=output_voxel_size,
            parameters={
                **shared_parameters,
                "low_pass_angstrom": low_pass,
                "centered": center,
                "mirrored": mirror,
            },
            software_versions={"pytom-match-pick": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.template.stat().st_size,
        )
        mask_artifact = Artifact(
            id=mask_id,
            variant=variant_label(context),
            kind=ArtifactKind.TEMPLATE_MASK,
            path=paths.mask,
            parent_ids=[template_artifact.id],
            shape=(actual_box_size, actual_box_size, actual_box_size),
            dtype="float32",
            axis_order=AxisOrder.ZYX,
            pixel_spacing_angstrom=output_voxel_size,
            parameters={
                **shared_parameters,
                "template_id": template_artifact.id,
                "mask_radius_px": mask_radii.x,
                "mask_radius_y_px": mask_radii.y,
                "mask_radius_z_px": mask_radii.z,
                "mask_is_spherical": mask_radii.is_sphere,
                "mask_sigma_px": mask_sigma,
            },
            software_versions={"pytom-match-pick": "external"},
            storage_role=StorageRole.CANONICAL,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.mask.stat().st_size,
        )
        qc_artifact = Artifact(
            id=(
                f"{manifest.tilt_series_id}:qc:template_preparation:{self.name}"
                f"{variant_suffix(context)}"
            ),
            variant=variant_label(context),
            kind=ArtifactKind.QC,
            path=paths.report,
            parent_ids=[template_artifact.id],
            parameters={
                "qc_type": "template_preparation",
                "backend": self.name,
                "tilt_series_id": manifest.tilt_series_id,
                "status": status.value,
            },
            storage_role=StorageRole.QC,
            retention_policy=RetentionPolicy.KEEP,
            can_recompute=True,
            size_bytes=paths.report.stat().st_size,
        )
        return [template_artifact, mask_artifact, qc_artifact]


def prepare_template_and_register(
    backend: TemplatePreparationBackend,
    source_map: Path,
    tomogram: Artifact,
    manifest: TiltSeriesManifest,
    context: BackendContext,
    registry: ArtifactRegistry,
    *,
    replace_existing: bool = False,
) -> list[Artifact]:
    """Prepare a template and register it, its mask, and the QC artifact."""

    artifacts = backend.prepare(source_map, tomogram, manifest, context)
    registry.extend(artifacts, replace=replace_existing)
    return artifacts


@dataclass(frozen=True)
class _VolumeStatistics:
    box_size: int
    minimum: float
    maximum: float
    median: float
    voxel_size_angstrom: float
    density_extent_px: tuple[float, float, float]
    """How far the density actually reaches, as `(x, y, z)` in pixels."""


def _inspect_volume(path: Path) -> _VolumeStatistics:
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.array(mrc.data, dtype=np.float32)
        voxel_size = float(mrc.voxel_size.x)
    if data.ndim != 3:
        raise RuntimeError(f"expected a 3D volume, got shape {data.shape}: {path}")
    if len(set(data.shape)) != 1:
        raise RuntimeError(f"expected a cubic volume, got shape {data.shape}: {path}")
    if not np.isfinite(data).all():
        raise RuntimeError(f"volume contains nonfinite values: {path}")
    median = float(np.median(data))
    return _VolumeStatistics(
        box_size=int(data.shape[0]),
        minimum=float(data.min()),
        maximum=float(data.max()),
        median=median,
        voxel_size_angstrom=voxel_size,
        density_extent_px=_density_extent_px(data, median=median),
    )


def _density_extent_px(data: np.ndarray, *, median: float) -> tuple[float, float, float]:
    """Measure how far the density reaches along each axis, as `(x, y, z)`.

    Per axis rather than as one diameter, because a reference map is not always
    a compact blob: EMD-2638 is a helical reconstruction whose density repeats
    along z and fills the box in that direction, so a single number describes it
    only by discarding what makes it different. Three numbers show the shape,
    and their disagreement is exactly the signal that a spherical mask is wrong.

    The width reported is where the profile falls to a tenth of its peak, which
    reads a hard-edged sphere about 8% under its true diameter and is the only
    tried measure that holds up in noise. Two others were tried and rejected on
    a sphere buried in 30% noise, where the truth is 24 px: the width holding
    95% of the mass drifted to 31, and the width at a twentieth of the peak —
    exact on clean shapes — blew up to 64, the whole box. Being slightly under
    is worth being stable, and the margin added when a mask is derived from this
    is set accordingly.

    Background has to be excluded first or the measurement is worthless. Mass
    spread thinly over the whole box is still mass, so on a map with ordinary
    noise every axis otherwise reads as the full box whatever the particle looks
    like.

    The background is removed from each axis's own profile, as the level it
    falls to, rather than from the volume as a whole. Estimating a noise level
    over the volume is what the obvious version does — a median absolute
    deviation, three sigma — and it is wrong for exactly the maps this function
    exists to describe. That estimator assumes most of the box is background,
    which is true of a lone particle and false of a lattice, so on EMD-2638 it
    measured the signal rather than the noise and cut away everything but the
    single strongest band: 8 px of a 28 px box along one axis, for a map whose
    density is spread across all three. Subtracting a per-axis floor makes no
    such assumption, and reads 28/27/24 there.

    Density is taken from whichever side of the median holds the signal, the
    same test `_contrast_matches_inversion` makes, so this reads an inverted
    template correctly.
    """

    signal = data - median
    below = median - float(data.min())
    above = float(data.max()) - median
    signal = -np.minimum(signal, 0.0) if below > above else np.maximum(signal, 0.0)

    extents: list[float] = []
    # The array is indexed (z, y, x); summing over the other two axes leaves the
    # profile along this one, so these yield x, then y, then z.
    for other_axes in ((0, 1), (0, 2), (1, 2)):
        profile = signal.sum(axis=other_axes).astype(np.float64)
        # A low percentile rather than the minimum, so one empty slice at the
        # edge of the box does not set the level for the whole axis.
        profile = np.clip(profile - np.percentile(profile, 10.0), 0.0, None)
        peak = profile.max()
        if peak <= 0.0:
            extents.append(0.0)
            continue
        occupied = np.nonzero(profile > 0.1 * peak)[0]
        extents.append(float(occupied.max() - occupied.min() + 1))
    return extents[0], extents[1], extents[2]


_MASK_TOO_TIGHT = 0.6
_MASK_TOO_LOOSE = 1.8
_ANISOTROPIC = 1.5


def _mask_versus_density_warnings(
    radii: _MaskRadii,
    statistics: _VolumeStatistics,
    *,
    voxel_size_angstrom: float,
) -> list[str]:
    """Compare the mask that was asked for against the density that is there.

    The only existing guard on the requested size is that the mask fits in the
    box, which catches a diameter larger than the whole map and nothing else.
    That was enough for EMD-2638, whose box is 153 angstrom, only because the
    proposed 250 angstrom exceeded it; on a map with a roomy box any wrong value
    up to the box size passes silently, and a mask that misses the density finds
    nothing without ever failing.

    These are warnings, not errors. The measurement rests on a definition of
    "extent" rather than on a fact about the map, a deliberately tight mask is a
    legitimate choice for a lattice, and refusing to run on a judgement call
    would be worse than saying what was noticed.
    """

    warnings: list[str] = []
    axes = zip(
        ("x", "y", "z"), (radii.x, radii.y, radii.z), statistics.density_extent_px, strict=True
    )
    for name, radius, extent in axes:
        if extent <= 0.0:
            continue
        ratio = (radius * 2.0) / extent
        if ratio < _MASK_TOO_TIGHT:
            warnings.append(
                f"mask is {radius * 2.0:.1f} px across along {name} but 95% of the density spans "
                f"{extent:.1f} px ({extent * voxel_size_angstrom:.0f} angstrom), and that is a "
                "floor; the mask cuts the particle, which lowers every score without failing"
            )
        elif ratio > _MASK_TOO_LOOSE:
            warnings.append(
                f"mask is {radius * 2.0:.1f} px across along {name} but 95% of the density "
                f"spans only {extent:.1f} px ({extent * voxel_size_angstrom:.0f} angstrom); "
                "the mask is "
                "mostly background, which dilutes the correlation"
            )

    extents = statistics.density_extent_px
    smallest, largest = min(extents), max(extents)
    if radii.is_sphere and smallest > 0.0 and largest / smallest > _ANISOTROPIC:
        warnings.append(
            f"density is not isotropic — x/y/z span {extents[0]:.1f}/{extents[1]:.1f}/"
            f"{extents[2]:.1f} px — but the mask is a sphere; consider "
            "--template-mask-radius-y-px and --template-mask-radius-z-px, and note that a "
            "non-spherical mask also changes how pytom normalises the score"
        )
    return warnings


def _contrast_matches_inversion(
    statistics: _VolumeStatistics,
    *,
    inverted: bool,
) -> bool:
    """Check the template's density points the way the invert flag implies.

    A reference map is a compact blob against a flat background, so the signal
    sits on one side of the median. After `--invert` the blob should be the
    darker excursion; without it, the brighter one.
    """

    below = statistics.median - statistics.minimum
    above = statistics.maximum - statistics.median
    if math.isclose(below, above, rel_tol=1e-6, abs_tol=1e-12):
        return True
    return below > above if inverted else above > below


def _source_voxel_size(path: Path) -> float:
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError(
            f"reference map has no usable voxel size: {path}; "
            "pass template_input_voxel_size_angstrom explicitly"
        )
    return voxel_size


@dataclass(frozen=True)
class _MaskRadii:
    """Mask semi-axes in template pixels, along x, y and z.

    A sphere covers a compact particle, but not every reference is one: a
    lattice or a membrane patch is extended in two directions and thin along the
    third, and a sphere drawn around it takes in mostly background. Leaving the
    y and z radii unset keeps the sphere.
    """

    x: float
    y: float
    z: float

    @property
    def is_sphere(self) -> bool:
        return self.x == self.y == self.z


_DERIVED_MARGIN = 1.2
"""Overhang on the measured extent, following pytom-match-pick's own advice.

Its documentation recommends a mask 10-20% wider than the particle radius, since
a tight mask removes noise from around the particle. The measurement this
multiplies reads a hard edge about 8% under true, so 1.2 lands near the bottom of
that band — which is the side the documentation leans to.
"""


def _mask_radii_px(
    context: BackendContext,
    *,
    voxel_size_angstrom: float,
    box_size: int,
    measured_extent_px: tuple[float, float, float] | None = None,
) -> tuple[_MaskRadii, bool]:
    """Size the mask, and say whether the size had to be derived.

    Deriving it is safe to add because asking for neither parameter used to be
    an error, so no command line that works today changes behaviour. The derived
    value is a sphere on the largest measured axis, never an ellipsoid: a
    non-spherical mask changes how pytom normalises the score, which is too
    consequential to start doing without being asked. When the map really is
    anisotropic the QC warning says so and names the per-axis options.

    A map whose density fills its box — a lattice, say — derives a radius that
    does not fit, and the existing box check refuses it. That refusal is the
    right answer: nothing about such a map says where one particle ends.
    """

    derived = False
    explicit = _optional_positive_float_parameter(context, "template_mask_radius_px")
    if explicit is not None:
        radius = explicit
    else:
        diameter = _optional_positive_float_parameter(
            context, "template_particle_diameter_angstrom"
        )
        if diameter is not None:
            radius = diameter / 2.0 / voxel_size_angstrom
        elif measured_extent_px is not None and max(measured_extent_px) > 0.0:
            radius = max(measured_extent_px) * _DERIVED_MARGIN / 2.0
            derived = True
        else:
            raise TypeError(
                "provide either 'template_mask_radius_px' or "
                "'template_particle_diameter_angstrom' to size the mask"
            )
    radii = _MaskRadii(
        x=radius,
        y=_optional_positive_float_parameter(context, "template_mask_radius_y_px") or radius,
        z=_optional_positive_float_parameter(context, "template_mask_radius_z_px") or radius,
    )

    longest = max(radii.x, radii.y, radii.z)
    if longest * 2.0 > box_size:
        if derived:
            # `derived` is only set when the measurement exists.
            extents = measured_extent_px or (0.0, 0.0, 0.0)
            raise ValueError(
                f"the density fills this map — measured extents "
                f"{extents[0]:.0f}/{extents[1]:.0f}/{extents[2]:.0f} px in a "
                f"{box_size} px box — so there is "
                "no particle size to derive. This is what a lattice or a helical "
                "reconstruction looks like; pass --template-particle-diameter-angstrom "
                "or --template-mask-radius-px to say what one particle is"
            )
        # Without an explicit box the size comes from the deposited map's own
        # extent after resampling, so the two numbers being far apart usually
        # means the requested diameter does not describe this map.
        raise ValueError(
            f"mask diameter {longest * 2.0:.1f} px does not fit in the template box of "
            f"{box_size} px ({box_size * voxel_size_angstrom:.0f} angstrom at "
            f"{voxel_size_angstrom:.2f} angstrom/pixel). Pass --template-box-size-px "
            f"with at least {math.ceil(longest * 2.0)}, or check that the particle is "
            "really that large"
        )
    return radii, derived


@dataclass(frozen=True)
class _TemplatePaths:
    output_dir: Path
    template: Path
    mask: Path
    log: Path
    report: Path

    @property
    def outputs(self) -> tuple[Path, ...]:
        return (self.template, self.mask, self.log, self.report)


def _template_paths(
    output_root: Path,
    manifest: TiltSeriesManifest,
    *,
    backend_name: str,
    label: str | None = None,
) -> _TemplatePaths:
    template_dir = output_root / "templates" / manifest.tilt_series_id
    qc_dir = output_root / "qc" / manifest.tilt_series_id / "template"
    prefix = f"{manifest.tilt_series_id}_{backend_name}" + (
        f"_{label}" if label else ""
    )
    return _TemplatePaths(
        output_dir=template_dir,
        template=template_dir / f"{prefix}_template.mrc",
        mask=template_dir / f"{prefix}_mask.mrc",
        log=template_dir / f"{prefix}.log",
        report=qc_dir / f"{prefix}_qc.json",
    )


def _validate_tomogram(tomogram: Artifact, manifest: TiltSeriesManifest) -> None:
    if tomogram.kind not in {ArtifactKind.TOMOGRAM, ArtifactKind.DENOISED_TOMOGRAM}:
        raise ValueError(f"expected tomogram artifact, got {tomogram.kind}")
    if not tomogram.covers(manifest.tilt_series_id):
        raise ValueError(
            f"tomogram is for {tomogram.parameters.get('tilt_series_id')}, "
            f"expected {manifest.tilt_series_id}"
        )


def _tomogram_voxel_size(tomogram: Artifact) -> float:
    value = tomogram.pixel_spacing_angstrom
    if value is None:
        raise ValueError("tomogram voxel spacing is required to scale the template")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("tomogram voxel spacing must be finite and positive")
    return value


def _resolve_pytom_executable(
    context: BackendContext,
    key: str,
    *,
    default_name: str,
) -> Path:
    configured = context.parameters.get(key)
    if configured is not None:
        if not isinstance(configured, (str, Path)):
            raise TypeError(f"context parameter {key!r} must be a path string")
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"pytom-match-pick executable not found: {candidate} "
                f"(from --{key.replace('_', '-')})"
            )
        return candidate.resolve()
    resolved = shutil.which(default_name)
    if resolved is not None:
        return Path(resolved)
    raise FileNotFoundError(
        f"{default_name} not found; pass --{key.replace('_', '-')} or add it to PATH"
    )



def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _replace_output(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    source.replace(destination)


def _require_available_paths(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"template outputs already exist: {joined}")


def _bool_parameter(context: BackendContext, key: str, *, default: bool) -> bool:
    value = context.parameters.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"context parameter {key!r} must be a bool")
    return value


def _optional_positive_float_parameter(context: BackendContext, key: str) -> float | None:
    value = context.parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"context parameter {key!r} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"context parameter {key!r} must be finite and greater than zero")
    return normalized


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
    return value
