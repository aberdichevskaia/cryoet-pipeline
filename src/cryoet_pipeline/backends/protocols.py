from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from cryoet_pipeline.models import Artifact, TiltSeriesManifest
from cryoet_pipeline.runtime import DevicePreference


@dataclass(frozen=True)
class BackendContext:
    """Runtime information shared with replaceable backend implementations."""

    output_dir: Path
    device: DevicePreference
    parameters: Mapping[str, Any] = field(default_factory=dict)


class MotionCorrectionBackend(Protocol):
    """Replaceable backend for multiframe movie correction."""

    name: str

    def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
        """Return corrected projection artifacts for one tilt-series."""
        ...


class TiltStackBackend(Protocol):
    """Replaceable backend for assembling corrected projections into a tilt stack."""

    name: str

    def build_stack(
        self,
        corrected_projections: Sequence[Artifact],
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Return a tilt-stack artifact, usually compatible with IMOD `.st` semantics."""
        ...


class TiltAlignmentBackend(Protocol):
    """Replaceable backend for tilt-series alignment."""

    name: str

    def align(
        self,
        tilt_stack: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Return an alignment artifact, usually compatible with IMOD `.xf` semantics."""
        ...


class CoarseAlignmentQcBackend(Protocol):
    """Replaceable backend for coarse-alignment previews and QC metrics."""

    name: str

    def evaluate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return preview and machine-readable QC artifacts."""
        ...


class FiducialSeedBackend(Protocol):
    """Replaceable backend for generating an initial fiducial seed model."""

    name: str

    def generate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a tracking stack, seed model, and seed QC artifacts."""
        ...


class FiducialTrackingBackend(Protocol):
    """Replaceable backend for tracking seed fiducials across tilt images."""

    name: str

    def track(
        self,
        tracking_stack: Artifact,
        seed_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a fully tracked fiducial model and QC artifact."""
        ...


class FineAlignmentBackend(Protocol):
    """Replaceable backend for fiducial-based fine alignment."""

    name: str

    def align(
        self,
        tracking_stack: Artifact,
        fiducial_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return canonical fine-alignment and QC artifacts."""
        ...


class FinalAlignedStackBackend(Protocol):
    """Replaceable backend for applying final alignment to a tilt stack."""

    name: str

    def build(
        self,
        tilt_stack: Artifact,
        fine_alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Return a final aligned tilt stack ready for reconstruction."""
        ...


class ReconstructionBackend(Protocol):
    """Replaceable backend for tomogram reconstruction."""

    name: str

    def reconstruct(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a reconstructed tomogram and its QC artifacts."""
        ...


class DenoisingBackend(Protocol):
    """Replaceable backend for denoising or restoring reconstructed tomograms."""

    name: str

    def denoise(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        """Return a denoised or restored tomogram artifact."""
        ...


class SegmentationBackend(Protocol):
    """Replaceable backend for producing masks or probability maps from tomograms."""

    name: str

    def segment(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        supporting_artifacts: Sequence[Artifact] = (),
    ) -> Artifact:
        """Return a segmentation, mask, or probability-map artifact."""
        ...


class PickingBackend(Protocol):
    """Replaceable backend for particle or object picking."""

    name: str

    def pick(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        supporting_artifacts: Sequence[Artifact] = (),
    ) -> Artifact:
        """Return a picks artifact, such as coordinates or a backend-native table."""
        ...


class DatasetExportBackend(Protocol):
    """Replaceable backend for exporting project data to external dataset formats."""

    name: str

    def export(
        self,
        manifests: Sequence[TiltSeriesManifest],
        artifacts: Sequence[Artifact],
        context: BackendContext,
    ) -> list[Artifact]:
        """Return export artifacts such as Croissant, copick, STAR, or Dynamo outputs."""
        ...
