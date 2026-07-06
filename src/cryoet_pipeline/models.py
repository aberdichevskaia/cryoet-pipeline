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
    TOMOGRAM = "tomogram"
    DENOISED_TOMOGRAM = "denoised_tomogram"
    SEGMENTATION = "segmentation"
    PICKS = "picks"
    DATASET_EXPORT = "dataset_export"
    QC = "qc"


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
                raise ValueError(
                    "input projection standard deviations must match transform count"
                )
            if not all(
                isfinite(value) and value >= 0.0
                for value in self.input_projection_std
            ):
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
            raise ValueError(
                "central_slice_paths must contain exactly: xy, xz, yz"
            )
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
    parameters: dict[str, Any] = Field(default_factory=dict)
    software_versions: dict[str, str] = Field(default_factory=dict)
    storage_role: StorageRole = StorageRole.CACHE
    retention_policy: RetentionPolicy = RetentionPolicy.RECOMPUTE
    can_recompute: bool = True
    size_bytes: int | None = Field(default=None, ge=0)
