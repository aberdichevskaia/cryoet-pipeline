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
    TILT_ANGLES = "tilt_angles"
    ALIGNMENT = "alignment"
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

    schema_version: int = 1
    tilt_series_id: str
    backend: str
    stage: str
    input_stack_id: str
    input_binning: int = Field(ge=1)
    tilt_axis_angle_deg: float
    transforms: list[AlignmentTransform]
    input_projection_std: list[float] = Field(default_factory=list)
    excluded_z_values: list[int] = Field(default_factory=list)

    @field_validator("tilt_axis_angle_deg")
    @classmethod
    def require_finite_tilt_axis_angle(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("tilt-axis angle must be finite")
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
        if not set(self.excluded_z_values).issubset(transform_z_values):
            raise ValueError("excluded z_value entries must refer to transforms")
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
