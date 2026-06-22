from __future__ import annotations

from typing import Protocol

from cryoet_pipeline.models import Artifact, TiltSeriesManifest


class MotionCorrectionBackend(Protocol):
    """Replaceable backend for multiframe movie correction."""

    name: str

    def correct(self, manifest: TiltSeriesManifest) -> list[Artifact]:
        """Return corrected projection artifacts for one tilt-series."""
        ...


class TiltAlignmentBackend(Protocol):
    """Replaceable backend for tilt-series alignment."""

    name: str

    def align(self, tilt_stack: Artifact, manifest: TiltSeriesManifest) -> Artifact:
        """Return an alignment artifact, usually compatible with IMOD `.xf` semantics."""
        ...


class ReconstructionBackend(Protocol):
    """Replaceable backend for tomogram reconstruction."""

    name: str

    def reconstruct(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
    ) -> Artifact:
        """Return a reconstructed tomogram artifact."""
        ...
