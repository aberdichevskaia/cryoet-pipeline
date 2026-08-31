from __future__ import annotations

import re
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


_VARIANT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def variant_label(context: BackendContext) -> str | None:
    """Return the run's variant label, if one was requested.

    A variant names a parallel run of the same stage that should coexist with
    the others rather than replace them: a different backend, a different
    binning, a different parameter sweep. Artifact ids carry it, so two runs
    that differ in something the id would not otherwise capture stay
    distinguishable in one registry.
    """

    value = context.parameters.get("variant")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("context parameter 'variant' must be a string")
    if not _VARIANT_PATTERN.fullmatch(value):
        raise ValueError(
            f"invalid variant {value!r}; use letters, digits, dot, dash, or underscore, "
            "starting with a letter or digit"
        )
    return value


def variant_suffix(context: BackendContext) -> str:
    """Return the id suffix for the run's variant, or an empty string."""

    label = variant_label(context)
    return f":{label}" if label is not None else ""


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


class DefocusEstimationBackend(Protocol):
    """Replaceable backend for estimating a single-image defocus hint."""

    name: str

    def estimate(
        self,
        corrected_projection: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a defocus-hint artifact and its QC artifact."""
        ...


class CtfEstimationBackend(Protocol):
    """Replaceable backend for per-tilt CTF estimation.

    Implementations differ in the fitting program (IMOD ctfplotter, CTFFIND5,
    ...) but all emit the same `CTF_MODEL` artifact: an IMOD-format `.defocus`
    file, so any estimation backend can feed any correction backend.
    """

    name: str

    def estimate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a CTF model artifact and its QC artifact."""
        ...


class CtfCorrectionBackend(Protocol):
    """Replaceable backend for tilt-series CTF phase correction."""

    name: str

    def correct(
        self,
        tilt_stack: Artifact,
        ctf_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a CTF-corrected aligned stack and its QC artifact."""
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


class TemplatePreparationBackend(Protocol):
    """Replaceable backend for turning a reference density map into a template.

    The target voxel size is taken from the tomogram that will be searched, so
    a template cannot silently be prepared at the wrong scale.
    """

    name: str

    def prepare(
        self,
        source_map: Path,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a template, a template mask, and a QC artifact."""
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


class ParticleExtractionBackend(Protocol):
    """Replaceable backend for cropping subvolumes from a tomogram at pick locations."""

    name: str

    def extract(
        self,
        tomogram: Artifact,
        picks: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        """Return a particle-stack artifact and its QC artifact."""
        ...


class SubtomogramAveragingBackend(Protocol):
    """Replaceable backend for iterative subtomogram alignment and averaging."""

    name: str

    def average(
        self,
        particles: Sequence[Artifact],
        manifests: Sequence[TiltSeriesManifest],
        context: BackendContext,
    ) -> list[Artifact]:
        """Average particles from one or more tilt-series into one map.

        Sequences rather than single values because averaging across tomograms is
        the ordinary case, not an extension of it. One stack is a list of one.
        """
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
