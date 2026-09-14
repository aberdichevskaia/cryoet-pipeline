from __future__ import annotations

from enum import StrEnum
from math import isfinite
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cryoet_pipeline.runtime import DevicePreference, normalize_device


class AxisOrder(StrEnum):
    """Known array axis conventions used by this pipeline."""

    YX = "yx"
    ZYX = "zyx"
    TYX = "tyx"
    FYX = "fyx"
    NZYX = "nzyx"


class ArtifactKind(StrEnum):
    """High-level artifact categories produced by pipeline steps."""

    RAW_TILT_MOVIE = "raw_tilt_movie"
    CORRECTED_PROJECTION = "corrected_projection"
    TILT_STACK = "tilt_stack"
    ALIGNED_TILT_STACK = "aligned_tilt_stack"
    TILT_ANGLES = "tilt_angles"
    ALIGNMENT = "alignment"
    FIDUCIAL_SEED_MODEL = "fiducial_seed_model"
    FIDUCIAL_MODEL = "fiducial_model"
    DEFOCUS_HINT = "defocus_hint"
    CTF_MODEL = "ctf_model"
    TOMOGRAM = "tomogram"
    DENOISED_TOMOGRAM = "denoised_tomogram"
    SEGMENTATION = "segmentation"
    TEMPLATE = "template"
    TEMPLATE_MASK = "template_mask"
    TEMPLATE_MATCH = "template_match"
    """The correlation search itself, separate from the picks read out of it.

    The match is the expensive thing — half an hour of GPU — and the picks are one
    cheap reading of it, at one cut-off with one peak separation. Registering the
    match makes those choices explorable without paying for the search again, and
    makes a pick list say which search it came from rather than restating the
    search's parameters.
    """
    PICKS = "picks"
    PARTICLE_STACK = "particle_stack"
    SUBTOMOGRAM_AVERAGE = "subtomogram_average"
    DATASET_EXPORT = "dataset_export"
    QC = "qc"


class TomogramBranch(StrEnum):
    """Logical tomogram branches used by post-reconstruction steps."""

    FULL = "full"
    EVEN = "even"
    ODD = "odd"


class StorageRole(StrEnum):
    """How an artifact should be treated by storage and cleanup policies."""

    SOURCE = "source"
    CANONICAL = "canonical"
    CACHE = "cache"
    EXPORT = "export"
    TEMPORARY = "temporary"
    QC = "qc"


class RetentionPolicy(StrEnum):
    """Whether an artifact should be kept or can be regenerated/deleted."""

    KEEP = "keep"
    RECOMPUTE = "recompute"
    DELETE_AFTER_EXPORT = "delete_after_export"


class QcStatus(StrEnum):
    """Machine-readable outcome of a pipeline quality-control check."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class ProjectConfig(BaseModel):
    """User-facing project configuration."""

    model_config = ConfigDict(extra="forbid")

    frames_dir: Path
    mdocs_dir: Path
    output_dir: Path
    tilt_series: list[str] = Field(default_factory=lambda: ["TS_01", "TS_43"])
    device: DevicePreference = DevicePreference.AUTO

    @field_validator("tilt_series")
    @classmethod
    def require_tilt_series(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one tilt-series id is required")
        return value

    @field_validator("device", mode="before")
    @classmethod
    def normalize_device_preference(cls, value: str | DevicePreference) -> DevicePreference:
        return normalize_device(value)


class TiltImage(BaseModel):
    """One tilt image entry from a SerialEM `.mdoc` file."""

    model_config = ConfigDict(extra="forbid")

    z_value: int
    tilt_angle_deg: float
    subframe_path: str
    local_frame_file: Path | None = None
    num_subframes: int
    pixel_spacing_angstrom: float
    binning: int
    exposure_time_s: float | None = None
    exposure_dose: float | None = None
    defocus_um: float | None = None
    rotation_angle_deg: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TiltSeriesManifest(BaseModel):
    """Normalized metadata for a tilt-series before processing."""

    model_config = ConfigDict(extra="forbid")

    tilt_series_id: str
    source_mdoc: Path
    images: list[TiltImage]
    mdoc_pixel_spacing_angstrom: float | None = None
    raw_pixel_spacing_angstrom: float | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("images")
    @classmethod
    def require_images(cls, value: list[TiltImage]) -> list[TiltImage]:
        if not value:
            raise ValueError("tilt-series manifest must contain at least one image")
        return value

    @property
    def tilt_angles_deg(self) -> list[float]:
        return [image.tilt_angle_deg for image in self.images]

    @property
    def num_tilts(self) -> int:
        return len(self.images)

    @property
    def num_subframes_set(self) -> set[int]:
        return {image.num_subframes for image in self.images}


class AlignmentTransform(BaseModel):
    """One normalized 2D affine transform for a tilt image."""

    model_config = ConfigDict(extra="forbid")

    z_value: int
    tilt_angle_deg: float
    a11: float
    a12: float
    a21: float
    a22: float
    shift_x_px: float
    shift_y_px: float

    @model_validator(mode="after")
    def require_finite_values(self) -> AlignmentTransform:
        values = (
            self.tilt_angle_deg,
            self.a11,
            self.a12,
            self.a21,
            self.a22,
            self.shift_x_px,
            self.shift_y_px,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("alignment transform values must be finite")
        return self


class TiltAlignment(BaseModel):
    """Canonical alignment result independent of an external backend format."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    tilt_series_id: str
    backend: str
    stage: str
    input_stack_id: str
    input_binning: int = Field(ge=1)
    tilt_axis_angle_deg: float
    transform_semantics: str
    transforms: list[AlignmentTransform]
    input_projection_std: list[float] = Field(default_factory=list)
    excluded_z_values: list[int] = Field(default_factory=list)

    @field_validator("tilt_axis_angle_deg")
    @classmethod
    def require_finite_tilt_axis_angle(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("tilt-axis angle must be finite")
        return value

    @field_validator("transform_semantics")
    @classmethod
    def require_global_transforms(cls, value: str) -> str:
        if value != "global":
            raise ValueError("canonical alignment transforms must be global")
        return value

    @field_validator("transforms")
    @classmethod
    def require_unique_transforms(
        cls,
        value: list[AlignmentTransform],
    ) -> list[AlignmentTransform]:
        if not value:
            raise ValueError("alignment must contain at least one transform")
        z_values = [transform.z_value for transform in value]
        if len(z_values) != len(set(z_values)):
            raise ValueError("alignment transform z_value entries must be unique")
        return value

    @model_validator(mode="after")
    def validate_input_quality(self) -> TiltAlignment:
        if self.input_projection_std:
            if len(self.input_projection_std) != len(self.transforms):
                raise ValueError("input projection standard deviations must match transform count")
            if not all(isfinite(value) and value >= 0.0 for value in self.input_projection_std):
                raise ValueError(
                    "input projection standard deviations must be finite and nonnegative"
                )

        transform_z_values = {transform.z_value for transform in self.transforms}
        if len(self.excluded_z_values) != len(set(self.excluded_z_values)):
            raise ValueError("excluded z_value entries must be unique")
        excluded = set(self.excluded_z_values)
        if self.stage == "coarse" and not excluded.issubset(transform_z_values):
            raise ValueError("coarse excluded z_value entries must refer to transforms")
        if self.stage == "fine" and excluded & transform_z_values:
            raise ValueError("fine transforms must not contain excluded z_value entries")
        return self


class CoarseAlignmentQc(BaseModel):
    """QC summary for a coarsely aligned tilt-series preview."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    input_stack_id: str
    alignment_id: str
    preview_path: Path
    preview_binning: int = Field(ge=1)
    included_z_values: list[int]
    excluded_z_values: list[int]
    residual_shift_x_px: list[float]
    residual_shift_y_px: list[float]
    residual_rms_px: float = Field(ge=0.0)
    residual_p95_px: float = Field(ge=0.0)
    residual_max_px: float = Field(ge=0.0)
    input_max_abs_shift_fraction: float = Field(ge=0.0)
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_qc_vectors(self) -> CoarseAlignmentQc:
        if not self.included_z_values:
            raise ValueError("coarse-alignment QC requires included tilt images")
        if set(self.included_z_values) & set(self.excluded_z_values):
            raise ValueError("included and excluded z_value entries must be disjoint")
        expected = len(self.included_z_values)
        if len(self.residual_shift_x_px) != expected:
            raise ValueError("residual X shifts must match included tilt count")
        if len(self.residual_shift_y_px) != expected:
            raise ValueError("residual Y shifts must match included tilt count")
        residual_values = [
            *self.residual_shift_x_px,
            *self.residual_shift_y_px,
        ]
        if not all(isfinite(value) for value in residual_values):
            raise ValueError("residual shifts must be finite")
        return self


class FiducialModelQc(BaseModel):
    """QC summary for an IMOD seed or fully tracked fiducial model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    stage: str
    model_id: str
    image_count: int = Field(ge=1)
    num_fiducials: int = Field(ge=0)
    num_points: int = Field(ge=0)
    min_points_per_fiducial: int = Field(ge=0)
    median_points_per_fiducial: float = Field(ge=0.0)
    max_points_per_fiducial: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @field_validator("stage")
    @classmethod
    def require_known_stage(cls, value: str) -> str:
        if value not in {"seed", "tracked"}:
            raise ValueError("fiducial model QC stage must be seed or tracked")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> FiducialModelQc:
        if self.num_fiducials == 0:
            if self.num_points != 0:
                raise ValueError("fiducial model with no contours cannot contain points")
            return self
        if self.num_points < self.num_fiducials:
            raise ValueError("each fiducial contour must contain at least one point")
        if not (
            self.min_points_per_fiducial
            <= self.median_points_per_fiducial
            <= self.max_points_per_fiducial
        ):
            raise ValueError("fiducial point-count statistics are inconsistent")
        return self


class PatchGridQc(BaseModel):
    """QC summary for a patch-tracking seed, which is a grid and not a model.

    A patch seed produces a description of where correlation patches will be
    placed, not a set of located points, so none of `FiducialModelQc` applies to
    it: reported through that model every field reads zero, which for a bead seed
    would mean the seeding had failed completely. These are the quantities a grid
    actually has.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    stage: str = "seed"
    model_id: str
    image_count: int = Field(ge=1)
    num_patches_x: int = Field(ge=1)
    num_patches_y: int = Field(ge=1)
    num_patches: int = Field(ge=1)
    patch_overlap: float = Field(ge=0.0, lt=1.0)
    # Sizes in the binned pixels the tracking stack is in, and in Angstrom,
    # because only the physical size can be judged against a specimen.
    pre_binning: int = Field(ge=1)
    patch_size_x_px: int = Field(ge=1)
    patch_size_y_px: int = Field(ge=1)
    patch_size_x_angstrom: float = Field(gt=0.0)
    patch_size_y_angstrom: float = Field(gt=0.0)
    # The margin is in binned pixels and does not scale with pre_binning, which
    # is the trap; recording what it left makes that visible.
    patch_margin_px: int = Field(ge=0)
    binned_width_px: int = Field(ge=1)
    binned_height_px: int = Field(ge=1)
    usable_width_fraction: float = Field(gt=0.0, le=1.0)
    usable_height_fraction: float = Field(gt=0.0, le=1.0)
    # cos of the largest tilt: the share of one image that every view still
    # covers once foreshortening is accounted for. A margin cutting more than
    # this is cutting more than the geometry asks for.
    foreshortening_fraction: float = Field(gt=0.0, le=1.0)
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @field_validator("stage")
    @classmethod
    def require_seed_stage(cls, value: str) -> str:
        if value != "seed":
            raise ValueError("patch grid QC describes the seed stage only")
        return value

    @model_validator(mode="after")
    def validate_grid(self) -> PatchGridQc:
        if self.num_patches != self.num_patches_x * self.num_patches_y:
            raise ValueError("patch count must be the product of the two axis counts")
        for value in (
            self.patch_size_x_angstrom,
            self.patch_size_y_angstrom,
            self.usable_width_fraction,
            self.usable_height_fraction,
            self.foreshortening_fraction,
        ):
            if not isfinite(value):
                raise ValueError("patch grid QC values must be finite")
        return self


class FineAlignmentQc(BaseModel):
    """Residual and geometry summary for fiducial-based fine alignment."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    alignment_id: str
    image_count: int = Field(ge=1)
    fiducial_count: int = Field(ge=1)
    projection_point_count: int = Field(ge=1)
    residual_mean_nm: float = Field(ge=0.0)
    residual_sd_nm: float = Field(ge=0.0)
    residual_mean_unbinned_px: float = Field(ge=0.0)
    residual_rms_tracking_px: float = Field(ge=0.0)
    residual_p95_tracking_px: float = Field(ge=0.0)
    residual_max_tracking_px: float = Field(ge=0.0)
    residual_outlier_count: int = Field(ge=0)
    pruned_point_count: int = Field(ge=0)
    alignment_rounds: int = Field(ge=1)
    global_leave_out_error_nm: float | None = Field(default=None, ge=0.0)
    minimum_tilt_rotation_deg: float
    recommended_x_axis_tilt_deg: float | None = None
    recommended_unbinned_thickness_px: float | None = Field(default=None, gt=0.0)
    # Points Tiltalign sorted onto each surface, when it fitted two. A surface
    # holding almost nothing means the two-plane fit degenerated and its
    # thickness is the offset of a stray point, not a specimen dimension.
    surface_point_counts: list[int] | None = None
    surfaces_analyzed: int | None = Field(default=None, ge=1, le=2)
    recommended_unbinned_z_shift_px: float | None = None
    applied_tilt_angle_offset_deg: float | None = None
    applied_axis_z_shift_unbinned_px: float | None = None
    positioning_incremental_tilt_angle_deg: float | None = None
    positioning_incremental_z_shift_unbinned_px: float | None = None
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_finite_values(self) -> FineAlignmentQc:
        values = [
            self.residual_mean_nm,
            self.residual_sd_nm,
            self.residual_mean_unbinned_px,
            self.residual_rms_tracking_px,
            self.residual_p95_tracking_px,
            self.residual_max_tracking_px,
            self.minimum_tilt_rotation_deg,
        ]
        if self.global_leave_out_error_nm is not None:
            values.append(self.global_leave_out_error_nm)
        optional_geometry = (
            self.recommended_x_axis_tilt_deg,
            self.recommended_unbinned_thickness_px,
            self.recommended_unbinned_z_shift_px,
            self.applied_tilt_angle_offset_deg,
            self.applied_axis_z_shift_unbinned_px,
            self.positioning_incremental_tilt_angle_deg,
            self.positioning_incremental_z_shift_unbinned_px,
        )
        values.extend(value for value in optional_geometry if value is not None)
        if not all(isfinite(value) for value in values):
            raise ValueError("fine-alignment QC values must be finite")
        return self


class TomogramQc(BaseModel):
    """Machine-readable statistics and preview paths for a reconstructed tomogram."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    tomogram_id: str
    shape: tuple[int, int, int]
    voxel_spacing_angstrom: float = Field(gt=0.0)
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float = Field(ge=0.0)
    finite: bool
    ctf_corrected: bool
    alignment_stage: str
    central_slice_paths: dict[str, Path]
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> TomogramQc:
        values = (
            self.voxel_spacing_angstrom,
            self.minimum,
            self.maximum,
            self.mean,
            self.standard_deviation,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("tomogram QC statistics must be finite")
        required_slices = {"xy", "xz", "yz"}
        if set(self.central_slice_paths) != required_slices:
            raise ValueError("central_slice_paths must contain exactly: xy, xz, yz")
        return self


class DefocusHintQc(BaseModel):
    """Machine-readable statistics for a single-image CTFFIND defocus hint."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    input_projection_id: str
    defocus_hint_id: str
    tilt_angle_deg: float
    defocus_um: float = Field(gt=0.0)
    astigmatism_um: float = Field(ge=0.0)
    azimuth_deg: float
    cross_correlation: float
    fit_resolution_angstrom: float = Field(gt=0.0)
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> DefocusHintQc:
        values = (
            self.tilt_angle_deg,
            self.defocus_um,
            self.astigmatism_um,
            self.azimuth_deg,
            self.cross_correlation,
            self.fit_resolution_angstrom,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("defocus hint QC statistics must be finite")
        return self


class CtfEstimationQc(BaseModel):
    """Machine-readable statistics for tilt-series CTF estimation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    tilt_series_id: str
    backend: str
    input_stack_id: str
    ctf_model_id: str
    num_tilts: int = Field(ge=1)
    voltage_kv: float = Field(gt=0.0)
    spherical_aberration_mm: float = Field(gt=0.0)
    amplitude_contrast: float = Field(ge=0.0, le=1.0)
    pixel_spacing_angstrom: float = Field(gt=0.0)
    nyquist_angstrom: float = Field(gt=0.0)
    max_resolution_angstrom: float = Field(gt=0.0)
    axis_angle_deg: float | None = None
    hint_defocus_um: float | None = None
    mean_defocus_um: float
    minimum_defocus_um: float
    maximum_defocus_um: float
    mean_astigmatism_um: float | None = None
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> CtfEstimationQc:
        values = [
            self.mean_defocus_um,
            self.minimum_defocus_um,
            self.maximum_defocus_um,
        ]
        if self.axis_angle_deg is not None:
            values.append(self.axis_angle_deg)
        if not all(isfinite(value) for value in values):
            raise ValueError("CTF estimation QC statistics must be finite")
        if not (self.minimum_defocus_um <= self.mean_defocus_um <= self.maximum_defocus_um):
            raise ValueError("CTF defocus statistics are inconsistent")
        if self.max_resolution_angstrom < self.nyquist_angstrom:
            raise ValueError(
                "max_resolution_angstrom cannot be finer than the Nyquist limit; "
                "it must be clamped before building the QC report"
            )
        return self


class CtfCorrectionQc(BaseModel):
    """Machine-readable statistics for tilt-series CTF phase-flip correction."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    input_stack_id: str
    ctf_model_id: str
    corrected_stack_id: str
    num_tilts: int = Field(ge=1)
    voltage_kv: float = Field(gt=0.0)
    spherical_aberration_mm: float = Field(gt=0.0)
    amplitude_contrast: float = Field(ge=0.0, le=1.0)
    pixel_spacing_angstrom: float = Field(gt=0.0)
    axis_angle_deg: float
    mean_defocus_um: float
    minimum_defocus_um: float
    maximum_defocus_um: float
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> CtfCorrectionQc:
        values = (
            self.axis_angle_deg,
            self.mean_defocus_um,
            self.minimum_defocus_um,
            self.maximum_defocus_um,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("CTF correction QC statistics must be finite")
        if not (self.minimum_defocus_um <= self.mean_defocus_um <= self.maximum_defocus_um):
            raise ValueError("CTF defocus statistics are inconsistent")
        return self


class TomogramRestorationQc(BaseModel):
    """Machine-readable statistics for a restored tomogram."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    input_tomogram_id: str
    restored_tomogram_id: str
    tomogram_branch: TomogramBranch = TomogramBranch.FULL
    shape: tuple[int, int, int]
    voxel_spacing_angstrom: float = Field(gt=0.0)
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float = Field(ge=0.0)
    finite: bool
    command_log_path: Path
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> TomogramRestorationQc:
        values = (
            self.voxel_spacing_angstrom,
            self.minimum,
            self.maximum,
            self.mean,
            self.standard_deviation,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("restoration QC statistics must be finite")
        if any(axis_size < 1 for axis_size in self.shape):
            raise ValueError("restored tomogram shape must be positive")
        return self


class PicksQc(BaseModel):
    """Machine-readable statistics for a particle-picking result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    input_tomogram_id: str
    segmentation_id: str | None = None
    picks_id: str
    particle_count: int = Field(ge=0)
    minimum_score: float | None = None
    maximum_score: float | None = None
    mean_score: float | None = None
    # Where the picks sit through the tomogram's thickness. A confined specimen
    # confines its particles; noise peaks are spread through the whole
    # reconstructed volume, and the count alone cannot tell the two apart.
    coordinate_z_median: float | None = None
    coordinate_z_sd: float | None = None
    uniform_z_sd: float | None = Field(default=None, gt=0.0)
    """The spread picks would have if they were uniform through the region where
    a template box can be centred: (thickness - box) / sqrt(12). Recorded so the
    observed spread can be read against it without knowing either size."""
    search_sigma: float | None = Field(default=None, gt=0.0)
    """The noise scale of the search, as the matcher measured it.

    pytom accumulates the standard deviation of the locally-normalised
    cross-correlation across its rotation loop. Without it a score is a number
    with no scale, and every question about whether a pick is real has to be
    re-asked later from files.
    """
    search_space: float | None = Field(default=None, gt=0.0)
    """Positions times orientations searched — how many chances noise had."""
    score_max_sigma: float | None = None
    """The best pick, in sigma. The number that says whether anything was found."""
    score_cut_off_sigma: float | None = None
    """Where the cut-off sat, in sigma. Set by the false-positive budget asked for."""
    one_false_positive_sigma: float | None = None
    """The highest peak a peak-free search over this space gives.

    The floor a detection has to clear. Comparing `score_max_sigma` against it is
    scale-free: it needs no external benchmark, and no assumption about how any
    other package defines its sigma — a comparison that has already been got
    wrong once here, between packages that normalise differently.
    """
    picks_above_one_false_positive: int | None = Field(default=None, ge=0)
    """How many picks clear that floor. One or two is what noise alone gives."""
    excess_over_noise_at_cut_off: float | None = None
    """Observed picks minus what the noise alone predicts, at the cut-off.

    For in-cell matching this is often the whole of the evidence: a real
    population whose members each sit near the noise floor still shows up as a
    count the null cannot account for. A value near zero means the picks ARE the
    tail.
    """
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> PicksQc:
        score_values = [
            value
            for value in (self.minimum_score, self.maximum_score, self.mean_score)
            if value is not None
        ]
        if not all(isfinite(value) for value in score_values):
            raise ValueError("picks QC score statistics must be finite")
        if self.particle_count == 0 and score_values:
            raise ValueError("picks QC cannot report score statistics with zero particles")
        return self


class TemplatePreparationQc(BaseModel):
    """Machine-readable statistics for a prepared matching template and mask."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    source_map_path: str
    emdb_id: str | None = None
    template_id: str
    template_mask_id: str
    input_voxel_size_angstrom: float = Field(gt=0)
    output_voxel_size_angstrom: float = Field(gt=0)
    box_size_px: int = Field(gt=0)
    inverted: bool
    low_pass_angstrom: float | None = None
    mask_radius_px: float = Field(gt=0)
    mask_radius_derived_from_density: bool = False
    """True when no mask size was given and it was measured off the map instead."""
    mask_derived_margin: float | None = Field(default=None, gt=0.0)
    """The factor applied to the measured extent when the radius was derived, so
    that the derivation can be reproduced from this document alone. None when a
    size was given rather than measured."""
    mask_sigma_px: float | None = None
    template_minimum: float
    template_maximum: float
    template_median: float
    density_extent_x_px: float = Field(ge=0.0)
    density_extent_y_px: float = Field(ge=0.0)
    density_extent_z_px: float = Field(ge=0.0)
    """How far the density reaches along each axis, measured from the produced
    template. Recorded per axis because a reference map is not always a compact
    blob, and the three disagreeing is what says a spherical mask is wrong."""
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> TemplatePreparationQc:
        values = (self.template_minimum, self.template_maximum, self.template_median)
        if not all(isfinite(value) for value in values):
            raise ValueError("template statistics must be finite")
        if self.template_minimum > self.template_maximum:
            raise ValueError("template minimum cannot exceed its maximum")
        if self.mask_radius_px * 2.0 > self.box_size_px:
            raise ValueError("mask diameter cannot exceed the template box size")
        extents = (self.density_extent_x_px, self.density_extent_y_px, self.density_extent_z_px)
        if any(extent > self.box_size_px for extent in extents):
            raise ValueError("measured density extent cannot exceed the template box size")
        return self


class ParticleExtractionQc(BaseModel):
    """Machine-readable statistics for a subvolume-extraction result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    input_tomogram_id: str
    picks_id: str
    particle_stack_id: str
    box_size_px: int = Field(gt=0)
    requested_particle_count: int = Field(ge=0)
    extracted_particle_count: int = Field(ge=0)
    dropped_edge_count: int = Field(ge=0)
    dropped_score_count: int = Field(ge=0)
    minimum_score_threshold: float | None = None
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_counts(self) -> ParticleExtractionQc:
        accounted = (
            self.extracted_particle_count + self.dropped_edge_count + self.dropped_score_count
        )
        if accounted != self.requested_particle_count:
            raise ValueError(
                "extracted + dropped_edge + dropped_score must equal requested_particle_count"
            )
        return self


class SubtomogramAveragingQc(BaseModel):
    """Machine-readable statistics for an iterative subtomogram-averaging result."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 2
    tilt_series_ids: list[str]
    """Every tilt-series that contributed particles, in the order given.

    A list rather than one id because an average over particles from ten
    tomograms is the point of averaging, and a field that can hold only one
    made that impossible to record — which is what kept it impossible to do.
    """
    backend: str
    particle_stack_ids: list[str]
    average_id: str
    particle_count: int = Field(ge=0)
    particle_counts: list[int] = Field(default_factory=list)
    """Particles taken from each stack, positionally matching
    `particle_stack_ids`. One series contributing almost everything is worth
    seeing, because an average is then mostly that series."""
    iterations_run: int = Field(ge=0)
    mean_final_correlation: float | None = None
    minimum_final_correlation: float | None = None
    starting_correlation_mean: float | None = None
    """Mean correlation before refinement, against whatever it started from.

    The final correlation alone has no scale. This is the other end of the
    comparison, and it is recorded whether or not a reference was supplied.
    """
    initial_reference_correlation_mean: float | None = None
    initial_reference_correlation_fraction_above_threshold: float | None = None
    half_set_counts: dict[int, int] = Field(default_factory=dict)
    """Particles in each half-set, which is what makes the resolution readable.

    Nothing recorded the split before, so a two-versus-two-hundred division
    reported PASS with a number beside it. The counts are the first thing a
    reader needs in order to decide whether the FSC below is an estimate or a
    coincidence.
    """
    gold_standard: bool = False
    """Whether the two halves were refined without ever seeing each other.

    False means the halves were split after a joint refinement against one
    shared reference, which inflates the FSC — the scheme Scheres & Chen name as
    defective. The flag exists so that a resolution can never be read as
    gold-standard by default: it has to be claimed, by the code that earned it.
    """
    resolution_angstrom: float | None = None
    """The inter-half FSC crossing at 0.143 after the final iteration.

    None when the curve never crossed, which is not a resolution of infinity but
    an absence of one.
    """
    resolution_trace_angstrom: list[float | None] = Field(default_factory=list)
    """The crossing after each iteration, in order.

    A refinement that is working shows this improving and then flattening. One
    that is overfitting shows it improving without pause, which is the signature
    worth being able to see rather than infer.
    """
    half_set_agreement_trace: list[float] = Field(default_factory=list)
    """Mean inter-half FSC across shells, per iteration — a continuous trajectory.

    The crossing above is quantised to Fourier shells: at box 48 the steps near
    36 angstrom are 2.6 angstrom wide, so a refinement that improves by less than
    a shell records as perfectly flat. That is not a plateau and it is not a
    failure, and the crossing alone cannot say which. This moves with agreement at
    any frequency, so a refinement that is working shows it rising and then
    levelling even when the crossing never budges.
    """
    low_pass_trace_angstrom: list[float] = Field(default_factory=list)
    """The resolution limit the reference was filtered to, per iteration.

    Recorded next to the trace above because the two are a feedback loop: the
    limit comes from the previous iteration's agreement, and the agreement is
    measured on a reference cut at that limit. Reading either alone hides that.
    """
    resolution_is_interpretable: bool = True
    """False when a half holds too few particles for its FSC to mean anything.

    Separate from `status` so that a small-particle run is not indistinguishable
    from a broken one: the average may be perfectly good while the resolution
    beside it is a draw from a wide distribution.
    """
    status: QcStatus

    @model_validator(mode="after")
    def check_half_set_counts_against_the_total(self) -> SubtomogramAveragingQc:
        """The halves have to be a partition of the particles, or they are not halves."""

        if not self.half_set_counts:
            return self
        if sorted(self.half_set_counts) != [1, 2]:
            raise ValueError(
                f"half-set counts must be keyed 1 and 2, got {sorted(self.half_set_counts)}"
            )
        total = sum(self.half_set_counts.values())
        if total != self.particle_count:
            raise ValueError(
                f"half-set counts sum to {total} but the average used "
                f"{self.particle_count} particles"
            )
        return self
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_statistics(self) -> SubtomogramAveragingQc:
        correlation_values = [
            value
            for value in (
                self.starting_correlation_mean,
                self.mean_final_correlation,
                self.minimum_final_correlation,
                self.initial_reference_correlation_mean,
            )
            if value is not None
        ]
        if not all(isfinite(value) for value in correlation_values):
            raise ValueError("subtomogram averaging QC correlations must be finite")
        fraction = self.initial_reference_correlation_fraction_above_threshold
        if fraction is not None and not (0.0 <= fraction <= 1.0):
            raise ValueError(
                "initial_reference_correlation_fraction_above_threshold must be in [0, 1]"
            )
        if not self.tilt_series_ids:
            raise ValueError("an average must record at least one tilt-series")
        if len(self.tilt_series_ids) != len(set(self.tilt_series_ids)):
            raise ValueError("tilt-series ids must be unique")
        if len(self.particle_stack_ids) != len(self.tilt_series_ids):
            raise ValueError(
                "every tilt-series must contribute exactly one particle stack"
            )
        if self.particle_counts:
            if len(self.particle_counts) != len(self.particle_stack_ids):
                raise ValueError(
                    "per-stack particle counts must match the stacks they describe"
                )
            if sum(self.particle_counts) != self.particle_count:
                raise ValueError(
                    "per-stack particle counts must sum to the total particle count"
                )
        return self


class ClassAverageSummary(BaseModel):
    """One reference produced by multi-reference alignment."""

    model_config = ConfigDict(extra="forbid")

    class_number: int = Field(ge=1)
    average_id: str
    particle_count: int = Field(ge=0)
    mean_correlation: float | None = None


class SubtomogramClassificationQc(BaseModel):
    """Machine-readable statistics for multi-reference subtomogram refinement."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 3
    alignment_mask_diameter_angstrom: float | None = Field(default=None, gt=0.0)
    alignment_mask_flat_radius_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    alignment_score: str = "whole-box-ncc"
    tilt_series_ids: list[str]
    """Every tilt-series that contributed particles, for the same reason
    `SubtomogramAveragingQc` carries a list: classifying particles from several
    tomograms is ordinary, and a single-id field made it unrecordable."""
    backend: str
    particle_stack_ids: list[str]
    particle_count: int = Field(ge=1)
    particle_counts: list[int] = Field(default_factory=list)
    """Particles taken from each stack, positionally matching
    `particle_stack_ids` — the same field `SubtomogramAveragingQc` carries.

    Missing here until a class had to be checked for tracking a tilt-series
    rather than a structure. A class drawn mostly from one series is separating
    microscope conditions, which reproduces across classification seeds exactly
    as faithfully as a real population does and is invisible in the class map.
    Nothing recorded the composition, so the question could not be asked of a
    finished run: `particle_count` gives the total and `tilt_series_ids` gives
    the names, and neither says how the total divides.

    Defaulted rather than required, so reports already written stay readable —
    they answer the question with an empty list, which is honest.
    """
    num_classes: int = Field(ge=1)
    seed: int | None = None
    """The seed that produced this partition.

    Recorded because a classification is only meaningful next to another one, and
    the comparison is between seeds. Reproducing a run from its report was
    impossible without this: every other parameter was here, so a report looked
    complete while being one integer short of reproducible — which surfaced when
    a control had to be built to match an existing run and the command could not
    be rebuilt from the report at all.

    Optional so that reports already written stay readable; None means the run
    predates the field, not that it had no seed.
    """
    refine_factor: float | None = None
    """Divisor applied to every range and step each iteration.

    The other half of what reproduction needs. `final_cone_sampling_deg` and
    `cone_sampling_deg` imply it across `iterations_run`, so it was recoverable
    by arithmetic — but a value that has to be back-solved from two others is not
    recorded, it is inferable, and the two are not the same thing when the
    formula changes.
    """
    iterations_run: int = Field(ge=0)
    cone_range_deg: float = Field(ge=0.0)
    cone_sampling_deg: float = Field(gt=0.0)
    inplane_range_deg: float = Field(ge=0.0)
    inplane_sampling_deg: float = Field(gt=0.0)
    shift_range_px: float = Field(ge=0.0)
    particle_diameter_angstrom: float = Field(gt=0.0)
    target_resolution_angstrom: float = Field(gt=0.0)
    crowther_increment_deg: float = Field(gt=0.0)
    final_cone_sampling_deg: float = Field(gt=0.0)
    classes: list[ClassAverageSummary]
    starting_correlation_mean: float | None = None
    """Mean correlation before refinement, against the references it started from.

    The final correlation alone has no scale, so on its own it cannot say
    whether the refinement moved anything.
    """
    mean_final_correlation: float | None = None
    minimum_final_correlation: float | None = None
    reassignment_fraction_last_iteration: float | None = None
    gold_standard: bool = False
    """Whether the half-sets were refined without ever seeing each other.

    False, and not as an oversight: this backend refines every particle against a
    shared set of class references, rebuilt from all of them each iteration.
    Making the halves independent here is not simply the same change as for
    single-class refinement, because each half would build its own K references
    and "class 2" on one side need not be "class 2" on the other — a per-class FSC
    would then need a correspondence between them. Whether the established
    practice is to classify jointly and gold-standard-refine one class afterwards
    is an open question recorded in the roadmap, not something this flag decides.

    It is here so that a reader of this report never has to know any of that to
    see that the resolution is not a gold-standard number.
    """
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_classes(self) -> SubtomogramClassificationQc:
        if len(self.classes) != self.num_classes:
            raise ValueError("classes must have one entry per class")
        if sum(entry.particle_count for entry in self.classes) != self.particle_count:
            raise ValueError("class particle counts must sum to particle_count")
        correlations = [
            value
            for value in (self.mean_final_correlation, self.minimum_final_correlation)
            if value is not None
        ]
        if not all(isfinite(value) for value in correlations):
            raise ValueError("classification QC correlations must be finite")
        fraction = self.reassignment_fraction_last_iteration
        if fraction is not None and not (0.0 <= fraction <= 1.0):
            raise ValueError("reassignment_fraction_last_iteration must be in [0, 1]")
        return self


class FscQc(BaseModel):
    """Machine-readable Fourier shell correlation report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    mode: str
    box_size_px: int = Field(gt=0)
    pixel_spacing_angstrom: float = Field(gt=0)
    threshold: float
    crossing_resolution_angstrom: float | None = None
    resolution_angstrom: list[float]
    correlation: list[float]
    status: QcStatus
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_curve(self) -> FscQc:
        if len(self.resolution_angstrom) != len(self.correlation):
            raise ValueError("resolution_angstrom and correlation must be the same length")
        if not self.resolution_angstrom:
            raise ValueError("FSC curve must have at least one shell")
        if self.crossing_resolution_angstrom is not None and not isfinite(
            self.crossing_resolution_angstrom
        ):
            raise ValueError("crossing_resolution_angstrom must be finite")
        return self


class Artifact(BaseModel):
    """Serializable record for any artifact written by the pipeline."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: ArtifactKind
    path: Path
    parent_ids: list[str] = Field(default_factory=list)
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    axis_order: AxisOrder | None = None
    pixel_spacing_angstrom: float | None = None
    binning: int | None = None
    backend: str | None = None
    """Which implementation produced this, when the stage has more than one."""
    variant: str | None = None
    """The lane label this run was given, when runs are meant to coexist."""
    parameters: dict[str, Any] = Field(default_factory=dict)
    software_versions: dict[str, str] = Field(default_factory=dict)
    storage_role: StorageRole = StorageRole.CACHE
    retention_policy: RetentionPolicy = RetentionPolicy.RECOMPUTE
    can_recompute: bool = True
    size_bytes: int | None = Field(default=None, ge=0)
    tilt_series_ids: list[str] = Field(default_factory=list)
    """Which tilt-series this artifact is about — its scope.

    A list because the moment particles from several tomograms are averaged
    together, the result is about all of them, and a single id cannot say so.
    Ordinary artifacts have exactly one; a cross-series average has several; a
    project-wide artifact such as a shared template has none.

    This is the file-first shape of `artifact_scope(artifact_id, tilt_series_id)`
    in [database_design.md](../../docs/database_design.md), which is a relation
    there and a list here. Promoted out of `parameters` for the same reason
    `backend` and `variant` were: a scope buried in an untyped dict cannot be
    indexed, joined, or checked.
    """

    @model_validator(mode="after")
    def adopt_scope_from_parameters(self) -> Artifact:
        """Fill the scope from `parameters` when only the parameter was given.

        Every backend has long recorded `tilt_series_id` as a parameter, and some
        now record `tilt_series_ids`. Adopting both keeps every existing registry
        readable and every existing producer correct, while making the field the
        one place a reader has to look.

        A field that disagrees with the parameters is refused rather than
        reconciled: nothing writes both, so a disagreement is a bug, and a
        silent one — a resolver reads the field while a human reads the
        parameter. This runs on construction and on deserialization, which is
        every path a registry takes; `model_copy` runs no validators, so a stale
        copy is caught on its next validation rather than at the copy.
        """

        recorded = self._recorded_scope()
        if not self.tilt_series_ids:
            if recorded is not None:
                object.__setattr__(self, "tilt_series_ids", recorded)
            return self
        if recorded is not None and recorded != self.tilt_series_ids:
            raise ValueError(
                f"artifact scope disagrees with its parameters: field says "
                f"{self.tilt_series_ids}, parameters say {recorded}"
            )
        return self

    def _recorded_scope(self) -> list[str] | None:
        """The scope as `parameters` states it, or None if it states none."""

        listed = self.parameters.get("tilt_series_ids")
        if isinstance(listed, list) and all(isinstance(item, str) for item in listed):
            return list(listed)
        single = self.parameters.get("tilt_series_id")
        if isinstance(single, str):
            return [single]
        return None

    def covers(self, tilt_series_id: str) -> bool:
        """Whether this artifact is about the given tilt-series.

        The question a resolver actually asks, and the reason it is a method: a
        caller comparing against `tilt_series_ids[0]` would be right for every
        artifact until the first cross-series one, and wrong from then on.
        """

        return tilt_series_id in self.tilt_series_ids

    @model_validator(mode="after")
    def adopt_backend_from_parameters(self) -> Artifact:
        """Fill the field from `parameters` when only the parameter was given.

        Backends have long recorded which implementation ran as a parameter.
        Promoting it to a field gives it one canonical home — the shape a
        relational schema will want as a column — without rewriting the thirty
        odd places that already set the parameter.

        Adoption only fires when the field is unset, so it cannot silently
        contradict a field a caller stated. It does not check the two against
        each other the way `adopt_scope_from_parameters` does; nothing writes
        both, and if that changes this should gain the same check.
        """

        if self.backend is None:
            recorded = self.parameters.get("backend")
            if isinstance(recorded, str):
                object.__setattr__(self, "backend", recorded)
        return self
