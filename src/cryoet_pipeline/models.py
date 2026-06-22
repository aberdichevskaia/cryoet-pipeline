from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cryoet_pipeline.runtime import DevicePreference, normalize_device


class AxisOrder(StrEnum):
    """Known array axis conventions used by this pipeline."""

    ZYX = "zyx"
    TYX = "tyx"
    FYX = "fyx"


class ArtifactKind(StrEnum):
    """High-level artifact categories produced by pipeline steps."""

    RAW_TILT_MOVIE = "raw_tilt_movie"
    CORRECTED_PROJECTION = "corrected_projection"
    TILT_STACK = "tilt_stack"
    ALIGNMENT = "alignment"
    TOMOGRAM = "tomogram"
    QC = "qc"


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
