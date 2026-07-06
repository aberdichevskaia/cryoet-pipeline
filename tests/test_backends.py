from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from cryoet_pipeline.backends.protocols import (
    BackendContext,
    CoarseAlignmentQcBackend,
    DatasetExportBackend,
    DenoisingBackend,
    FiducialSeedBackend,
    FiducialTrackingBackend,
    FinalAlignedStackBackend,
    FineAlignmentBackend,
    MotionCorrectionBackend,
    PickingBackend,
    ReconstructionBackend,
    SegmentationBackend,
    TiltAlignmentBackend,
    TiltStackBackend,
)
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, TiltImage, TiltSeriesManifest
from cryoet_pipeline.runtime import DevicePreference


class _FakeBackends:
    name = "fake"

    def correct(self, manifest: TiltSeriesManifest, context: BackendContext) -> list[Artifact]:
        return [
            Artifact(
                id="corrected-0",
                kind=ArtifactKind.CORRECTED_PROJECTION,
                path=context.output_dir / f"{manifest.tilt_series_id}_000.zarr",
                axis_order=AxisOrder.FYX,
                parameters={"device": context.device},
            )
        ]

    def build_stack(
        self,
        corrected_projections: Sequence[Artifact],
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        return Artifact(
            id="stack",
            kind=ArtifactKind.TILT_STACK,
            path=context.output_dir / f"{manifest.tilt_series_id}.st",
            parent_ids=[artifact.id for artifact in corrected_projections],
            axis_order=AxisOrder.TYX,
        )

    def align(
        self,
        tilt_stack: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        return Artifact(
            id="alignment",
            kind=ArtifactKind.ALIGNMENT,
            path=context.output_dir / f"{manifest.tilt_series_id}.xf",
            parent_ids=[tilt_stack.id],
        )

    def reconstruct(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        return [
            Artifact(
                id="tomogram",
                kind=ArtifactKind.TOMOGRAM,
                path=context.output_dir / f"{manifest.tilt_series_id}.rec",
                parent_ids=[tilt_stack.id, alignment.id],
                axis_order=AxisOrder.ZYX,
            )
        ]

    def evaluate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        return [
            Artifact(
                id="alignment-qc",
                kind=ArtifactKind.QC,
                path=context.output_dir / f"{manifest.tilt_series_id}_alignment_qc.json",
                parent_ids=[tilt_stack.id, alignment.id],
            )
        ]

    def generate(
        self,
        tilt_stack: Artifact,
        alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        tracking_stack = Artifact(
            id="tracking-stack",
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=context.output_dir / f"{manifest.tilt_series_id}_tracking.st",
            parent_ids=[tilt_stack.id, alignment.id],
        )
        seed = Artifact(
            id="seed",
            kind=ArtifactKind.FIDUCIAL_SEED_MODEL,
            path=context.output_dir / f"{manifest.tilt_series_id}.seed",
            parent_ids=[tracking_stack.id],
        )
        return [tracking_stack, seed]

    def track(
        self,
        tracking_stack: Artifact,
        seed_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        return [
            Artifact(
                id="fiducial-model",
                kind=ArtifactKind.FIDUCIAL_MODEL,
                path=context.output_dir / f"{manifest.tilt_series_id}.fid",
                parent_ids=[tracking_stack.id, seed_model.id],
            )
        ]

    def denoise(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        return Artifact(
            id="denoised",
            kind=ArtifactKind.DENOISED_TOMOGRAM,
            path=context.output_dir / f"{manifest.tilt_series_id}_denoised.zarr",
            parent_ids=[tomogram.id],
            parameters={"method": context.parameters["denoising_method"]},
        )

    def segment(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        supporting_artifacts: Sequence[Artifact] = (),
    ) -> Artifact:
        return Artifact(
            id="segmentation",
            kind=ArtifactKind.SEGMENTATION,
            path=context.output_dir / f"{manifest.tilt_series_id}_segmentation.zarr",
            parent_ids=[tomogram.id, *(artifact.id for artifact in supporting_artifacts)],
        )

    def pick(
        self,
        tomogram: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
        supporting_artifacts: Sequence[Artifact] = (),
    ) -> Artifact:
        return Artifact(
            id="picks",
            kind=ArtifactKind.PICKS,
            path=context.output_dir / f"{manifest.tilt_series_id}_picks.json",
            parent_ids=[tomogram.id, *(artifact.id for artifact in supporting_artifacts)],
        )

    def export(
        self,
        manifests: Sequence[TiltSeriesManifest],
        artifacts: Sequence[Artifact],
        context: BackendContext,
    ) -> list[Artifact]:
        return [
            Artifact(
                id="croissant-export",
                kind=ArtifactKind.DATASET_EXPORT,
                path=context.output_dir / "croissant.json",
                parent_ids=[artifact.id for artifact in artifacts],
                parameters={
                    "format": "croissant",
                    "tilt_series": [manifest.tilt_series_id for manifest in manifests],
                },
            )
        ]


class _FakeFineAlignmentBackend:
    name = "fake-fine"

    def align(
        self,
        tracking_stack: Artifact,
        fiducial_model: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> list[Artifact]:
        return [
            Artifact(
                id="fine-alignment",
                kind=ArtifactKind.ALIGNMENT,
                path=context.output_dir / f"{manifest.tilt_series_id}_fine.json",
                parent_ids=[tracking_stack.id, fiducial_model.id],
                parameters={"stage": "fine"},
            )
        ]


class _FakeFinalAlignedStackBackend:
    name = "fake-final-stack"

    def build(
        self,
        tilt_stack: Artifact,
        fine_alignment: Artifact,
        manifest: TiltSeriesManifest,
        context: BackendContext,
    ) -> Artifact:
        return Artifact(
            id="final-stack",
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=context.output_dir / f"{manifest.tilt_series_id}_final.st",
            parent_ids=[tilt_stack.id, fine_alignment.id],
            parameters={"purpose": "final_alignment"},
        )


def test_backend_protocols_share_pipeline_artifacts(tmp_path: Path) -> None:
    manifest = _manifest()
    context = BackendContext(
        output_dir=tmp_path,
        device=DevicePreference.CPU,
        parameters={"denoising_method": "average"},
    )
    backend = _FakeBackends()

    motion: MotionCorrectionBackend = backend
    stacker: TiltStackBackend = backend
    aligner: TiltAlignmentBackend = backend
    alignment_qc_backend: CoarseAlignmentQcBackend = backend
    seed_backend: FiducialSeedBackend = backend
    tracking_backend: FiducialTrackingBackend = backend
    fine_backend: FineAlignmentBackend = _FakeFineAlignmentBackend()
    final_stack_backend: FinalAlignedStackBackend = _FakeFinalAlignedStackBackend()
    reconstructor: ReconstructionBackend = backend
    denoiser: DenoisingBackend = backend
    segmenter: SegmentationBackend = backend
    picker: PickingBackend = backend
    exporter: DatasetExportBackend = backend

    corrected = motion.correct(manifest, context)
    tilt_stack = stacker.build_stack(corrected, manifest, context)
    alignment = aligner.align(tilt_stack, manifest, context)
    alignment_qc = alignment_qc_backend.evaluate(
        tilt_stack,
        alignment,
        manifest,
        context,
    )
    tracking_stack, seed_model = seed_backend.generate(
        tilt_stack,
        alignment,
        manifest,
        context,
    )
    fiducial_model = tracking_backend.track(
        tracking_stack,
        seed_model,
        manifest,
        context,
    )[0]
    fine_alignment = fine_backend.align(
        tracking_stack,
        fiducial_model,
        manifest,
        context,
    )[0]
    final_stack = final_stack_backend.build(
        tilt_stack,
        fine_alignment,
        manifest,
        context,
    )
    tomogram = reconstructor.reconstruct(tilt_stack, alignment, manifest, context)[0]
    denoised = denoiser.denoise(tomogram, manifest, context)
    segmentation = segmenter.segment(denoised, manifest, context)
    picks = picker.pick(denoised, manifest, context, supporting_artifacts=[segmentation])
    exports = exporter.export([manifest], [picks], context)

    assert [artifact.kind for artifact in corrected] == [ArtifactKind.CORRECTED_PROJECTION]
    assert tilt_stack.kind == ArtifactKind.TILT_STACK
    assert alignment.kind == ArtifactKind.ALIGNMENT
    assert alignment_qc[0].kind == ArtifactKind.QC
    assert seed_model.kind == ArtifactKind.FIDUCIAL_SEED_MODEL
    assert fiducial_model.kind == ArtifactKind.FIDUCIAL_MODEL
    assert fine_alignment.parameters["stage"] == "fine"
    assert final_stack.parameters["purpose"] == "final_alignment"
    assert tomogram.kind == ArtifactKind.TOMOGRAM
    assert denoised.kind == ArtifactKind.DENOISED_TOMOGRAM
    assert denoised.parameters == {"method": "average"}
    assert segmentation.kind == ArtifactKind.SEGMENTATION
    assert picks.kind == ArtifactKind.PICKS
    assert picks.parent_ids == ["denoised", "segmentation"]
    assert exports[0].kind == ArtifactKind.DATASET_EXPORT
    assert exports[0].parameters["format"] == "croissant"


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mrc.mdoc"),
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frames/TS_TEST_000_0.0.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            )
        ],
    )
