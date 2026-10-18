from __future__ import annotations

from pathlib import Path

import pytest
import typer

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext, variant_label, variant_suffix
from cryoet_pipeline.cli import (
    _ctf_weighting_for_tomogram,
    _tilt_angles_for_tomogram,
    _fiducial_seed_artifact,
    _particle_stack_artifact,
    _reconstruction_input_stack_artifact,
    _select_variant,
    _tomogram_artifact,
)
from cryoet_pipeline.models import Artifact, ArtifactKind, TiltImage, TiltSeriesManifest
from cryoet_pipeline.runtime import DevicePreference


def _context(**parameters: object) -> BackendContext:
    return BackendContext(
        output_dir=Path("."),
        device=DevicePreference.CPU,
        parameters=parameters,
    )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frame.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            )
        ],
    )


def _seed(variant: str | None) -> Artifact:
    """A seed model as the backends now write one: the label is a field."""

    suffix = f":{variant}" if variant else ""
    return Artifact(
        id=f"TS_TEST:fiducial_seed{suffix}",
        kind=ArtifactKind.FIDUCIAL_SEED_MODEL,
        path=Path(f"seed{suffix}.mod"),
        variant=variant,
        parameters={"tilt_series_id": "TS_TEST"},
    )


def _legacy_seed(variant: str) -> Artifact:
    """A seed model from before the field existed: the label is only in the id."""

    return _seed(variant).model_copy(update={"variant": None})


def test_absent_variant_leaves_ids_unchanged() -> None:
    assert variant_suffix(_context()) == ""
    assert variant_label(_context()) is None


def test_variant_is_appended_and_validated() -> None:
    assert variant_suffix(_context(variant="patch")) == ":patch"
    assert variant_label(_context(variant="bin8")) == "bin8"

    with pytest.raises(ValueError, match="invalid variant"):
        variant_label(_context(variant="two words"))
    with pytest.raises(ValueError, match="invalid variant"):
        # A colon would forge an extra id segment.
        variant_label(_context(variant="a:b"))
    with pytest.raises(TypeError, match="must be a string"):
        variant_label(_context(variant=8))


def test_two_variants_of_one_stage_coexist_in_the_registry() -> None:
    registry = ArtifactRegistry.empty()
    registry.add(_seed("bead"))
    registry.add(_seed("patch"))

    assert len(registry.by_kind(ArtifactKind.FIDUCIAL_SEED_MODEL)) == 2
    assert _fiducial_seed_artifact(registry, _manifest(), variant="patch").id == (
        "TS_TEST:fiducial_seed:patch"
    )
    assert _fiducial_seed_artifact(registry, _manifest(), variant="bead").id == (
        "TS_TEST:fiducial_seed:bead"
    )


def test_unlabelled_runs_of_one_stage_still_collide() -> None:
    # Without a variant the ids are identical, which is what makes an
    # accidental re-run a duplicate rather than a silent second result.
    registry = ArtifactRegistry.empty()
    registry.add(_seed(None))

    with pytest.raises(ValueError, match="duplicate artifact id"):
        registry.add(_seed(None))


def test_ambiguous_lookup_lists_the_candidates() -> None:
    registry = ArtifactRegistry.empty()
    registry.add(_seed("bead"))
    registry.add(_seed("patch"))

    with pytest.raises(typer.BadParameter) as excinfo:
        _fiducial_seed_artifact(registry, _manifest())

    message = str(excinfo.value)
    assert "found 2 candidates" in message
    assert "--variant" in message
    assert "TS_TEST:fiducial_seed:bead" in message
    assert "TS_TEST:fiducial_seed:patch" in message


def test_a_lane_can_diverge_from_shared_unlabelled_inputs() -> None:
    """Labelling a rebuild must not require a labelled input that cannot exist.

    Rebuilding the final stack at another binning is where the lane parts, so
    its input — the fine alignment — is still the shared one. Demanding that it
    carry the label too would mean only a labelled rebuild could enable a
    labelled rebuild.
    """

    registry = ArtifactRegistry.empty()
    registry.add(_seed(None))

    selected = _fiducial_seed_artifact(registry, _manifest(), variant="bin8")

    assert selected.id == "TS_TEST:fiducial_seed"


def test_a_lane_never_falls_back_to_another_lane() -> None:
    """Falling back is only ever to the shared artifact, never to a rival lane."""

    registry = ArtifactRegistry.empty()
    registry.add(_seed("bead"))

    with pytest.raises(typer.BadParameter, match="no fiducial seed model"):
        _fiducial_seed_artifact(registry, _manifest(), variant="patch")


def test_a_labelled_input_wins_over_the_shared_one() -> None:
    registry = ArtifactRegistry.empty()
    registry.add(_seed(None))
    registry.add(_seed("patch"))

    assert _fiducial_seed_artifact(registry, _manifest(), variant="patch").id == (
        "TS_TEST:fiducial_seed:patch"
    )


def _corrected_stack(variant: str | None) -> Artifact:
    suffix = f":{variant}" if variant else ""
    return Artifact(
        id=f"TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip{suffix}",
        kind=ArtifactKind.ALIGNED_TILT_STACK,
        path=Path(f"corrected{suffix}.st"),
        variant=variant,
        parameters={
            "tilt_series_id": "TS_TEST",
            "purpose": "ctf_corrected",
            "backend": "imod-ctfphaseflip",
        },
    )


def test_ctf_lanes_are_selectable_although_the_id_carries_a_backend_name() -> None:
    """The unlabelled form is one id per backend, not a single fixed string."""

    registry = ArtifactRegistry.empty()
    registry.add(_corrected_stack(None))
    registry.add(_corrected_stack("bin8"))

    assert _reconstruction_input_stack_artifact(
        registry, _manifest(), variant="bin8"
    ).id == "TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip:bin8"
    assert _reconstruction_input_stack_artifact(
        registry, _manifest(), variant="other"
    ).id == "TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip", (
        "an unlabelled corrected stack is what a lane without its own CTF run shares"
    )


def _tomogram(variant: str | None) -> Artifact:
    suffix = f":{variant}" if variant else ""
    return Artifact(
        id=f"TS_TEST:tomogram:fine{suffix}",
        kind=ArtifactKind.TOMOGRAM,
        path=Path(f"tomogram{suffix}.zarr"),
        variant=variant,
        parameters={"tilt_series_id": "TS_TEST"},
    )


def test_stages_after_reconstruction_can_pick_a_lane() -> None:
    """Two reconstructions must not block everything downstream of them.

    Template preparation, picking, extraction and the rest all look the tomogram
    up, and until they could name a lane a second reconstruction stopped the
    pipeline dead with "expected exactly one full tomogram".
    """

    registry = ArtifactRegistry.empty()
    registry.add(_tomogram(None))
    registry.add(_tomogram("bin8"))

    assert _tomogram_artifact(registry, _manifest(), variant="bin8").id == (
        "TS_TEST:tomogram:fine:bin8"
    )
    with pytest.raises(typer.BadParameter, match="found 2 candidates"):
        _tomogram_artifact(registry, _manifest())


def test_select_variant_returns_the_only_match_without_a_label() -> None:
    only = _seed(None)
    assert (
        _select_variant(
            [only], what="fiducial seed model", tilt_series_id="TS_TEST", variant=None
        )
        is only
    )


def test_shared_fallback_selects_the_artifact_every_lane_shares() -> None:
    """A lane labelling its rebuild must not hide the artifact every lane shares.

    The CTF fit stack carries no label on purpose, so that `correct-ctf --variant
    bin8` can fall back to the model fitted from it. `estimate-ctf` therefore looks
    it up with no label, and once `--variant bin8` had registered a stack beside
    the shared one it saw two candidates and refused -- so the lane could not be
    re-run from fine alignment, when both stacks exist by then.
    """

    shared = _seed(None)
    lane = _seed("bin8")

    assert (
        _select_variant(
            [shared, lane],
            what="fiducial seed model",
            tilt_series_id="TS_TEST",
            variant=None,
            shared_fallback=True,
        )
        is shared
    )


def test_shared_fallback_is_off_for_every_other_lookup() -> None:
    """The refusal is what lets a second lane's tomogram coexist."""

    with pytest.raises(typer.BadParameter, match="found 2 candidates"):
        _select_variant(
            [_seed(None), _seed("bin8")],
            what="fiducial seed model",
            tilt_series_id="TS_TEST",
            variant=None,
        )


def test_shared_fallback_still_reports_two_lanes_and_no_baseline() -> None:
    """Nothing shared to fall back to is a real ambiguity, fallback or not."""

    with pytest.raises(typer.BadParameter, match="found 2 candidates"):
        _select_variant(
            [_seed("bin4"), _seed("bin8")],
            what="fiducial seed model",
            tilt_series_id="TS_TEST",
            variant=None,
            shared_fallback=True,
        )


def test_a_label_recorded_only_in_an_old_id_still_selects() -> None:
    """Registries written before the field must keep working.

    The label of such an artifact cannot be recovered as data — the canonical
    form is unknown — so it is matched by its id suffix and otherwise counted as
    shared. Asking for a label nothing carries then reports the ambiguity rather
    than picking a lane, which is the honest degradation.
    """

    registry = ArtifactRegistry.empty()
    registry.add(_seed(None))
    registry.add(_legacy_seed("bin8"))

    assert _fiducial_seed_artifact(registry, _manifest(), variant="bin8").id == (
        "TS_TEST:fiducial_seed:bin8"
    )
    with pytest.raises(typer.BadParameter, match="found 2 candidates"):
        _fiducial_seed_artifact(registry, _manifest(), variant="other")


def test_ctf_weighting_comes_from_the_model_that_corrected_this_tomogram(
    tmp_path: Path,
) -> None:
    """Two lanes can hold two CTF models; the right one is a matter of lineage.

    The corrected stack records which model corrected it, so the weighting uses
    the same optics as the correction rather than a second set that could
    disagree with it.
    """

    defocus = tmp_path / "TS_TEST.defocus"
    defocus.write_text("1 0 0.0 0.0 0.0 3\n")
    registry = ArtifactRegistry.empty()
    registry.extend(
        [
            Artifact(
                id="TS_TEST:aligned_tilt_stack:final",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=tmp_path / "aligned.st",
                parameters={"tilt_series_id": "TS_TEST", "purpose": "final_alignment"},
            ),
            Artifact(
                id="TS_TEST:ctf_model:ctffind5",
                kind=ArtifactKind.CTF_MODEL,
                path=defocus,
                parent_ids=["TS_TEST:aligned_tilt_stack:final"],
                parameters={
                    "tilt_series_id": "TS_TEST",
                    "backend": "ctffind5",
                    "voltage_kv": 300.0,
                    "spherical_aberration_mm": 2.7,
                    "amplitude_contrast": 0.07,
                },
            ),
            Artifact(
                id="TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip",
                kind=ArtifactKind.ALIGNED_TILT_STACK,
                path=tmp_path / "corrected.st",
                parent_ids=[
                    "TS_TEST:aligned_tilt_stack:final",
                    "TS_TEST:ctf_model:ctffind5",
                ],
                parameters={
                    "tilt_series_id": "TS_TEST",
                    "purpose": "ctf_corrected",
                    "backend": "imod-ctfphaseflip",
                    "ctf_model_id": "TS_TEST:ctf_model:ctffind5",
                },
            ),
        ]
    )
    corrected_tomogram = Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "tomogram.zarr",
        parent_ids=["TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip"],
        parameters={"tilt_series_id": "TS_TEST"},
    )
    plain_tomogram = Artifact(
        id="TS_TEST:tomogram:fine:noctf",
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "tomogram_noctf.zarr",
        parent_ids=["TS_TEST:aligned_tilt_stack:final"],
        parameters={"tilt_series_id": "TS_TEST"},
        variant="noctf",
    )
    registry.extend([corrected_tomogram, plain_tomogram])

    weighting = _ctf_weighting_for_tomogram(registry, _manifest(), corrected_tomogram)
    assert weighting is not None
    assert weighting.defocus_path == defocus
    # A model that records no z values pairs its rows with views by count alone.
    assert weighting.z_values is None
    assert weighting.voltage_kv == 300.0
    assert weighting.spherical_aberration_mm == 2.7
    assert weighting.amplitude_contrast == 0.07
    assert weighting.model == "phase-flip"

    # Nothing corrected this one, so there is no weighting to declare.
    assert _ctf_weighting_for_tomogram(registry, _manifest(), plain_tomogram) is None


def test_averaging_can_name_the_lane_it_averages(tmp_path: Path) -> None:
    """Extraction labels its stack, so averaging has to be able to choose.

    Two lanes could run all the way through extraction and then dead-end: the
    lookup demanded exactly one particle stack and there was no flag to resolve
    which. Not a wrong answer, but no way to proceed.
    """

    def stack(variant: str | None) -> Artifact:
        suffix = f":{variant}" if variant else ""
        return Artifact(
            id=f"TS_TEST:particle_stack:cubic-crop{suffix}",
            kind=ArtifactKind.PARTICLE_STACK,
            path=tmp_path / f"particles{suffix}.zarr",
            variant=variant,
            parameters={"tilt_series_id": "TS_TEST"},
        )

    registry = ArtifactRegistry.empty()
    registry.add(stack(None))
    registry.add(stack("bin8"))

    assert _particle_stack_artifact(registry, _manifest(), variant="bin8").id == (
        "TS_TEST:particle_stack:cubic-crop:bin8"
    )
    with pytest.raises(typer.BadParameter, match="found 2 candidates"):
        _particle_stack_artifact(registry, _manifest())


def test_pytom_takes_the_angles_the_tomogram_was_built_from(tmp_path: Path) -> None:
    """Not an ancestor stack's: correct-ctf can leave out views it still holds."""

    all_views = tmp_path / "final.tlt"
    all_views.write_text("-3.0\n0.0\n3.0\n")
    kept_views = tmp_path / "tomogram.tlt"
    kept_views.write_text("0.0\n3.0\n")
    registry = ArtifactRegistry.empty()
    registry.add(
        Artifact(
            id="TS_TEST:aligned_tilt_stack:final",
            kind=ArtifactKind.ALIGNED_TILT_STACK,
            path=tmp_path / "aligned.st",
            parameters={
                "tilt_series_id": "TS_TEST",
                "purpose": "final_alignment",
                "tilt_file_path": str(all_views),
            },
        )
    )
    tomogram = Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=tmp_path / "tomogram.zarr",
        parent_ids=["TS_TEST:aligned_tilt_stack:final"],
        parameters={"tilt_series_id": "TS_TEST", "tilt_angles_path": str(kept_views)},
    )
    registry.add(tomogram)

    assert _tilt_angles_for_tomogram(registry, _manifest(), tomogram) == kept_views

    # A tomogram from before it recorded its angles falls back to its stacks.
    older = tomogram.model_copy(
        update={"id": "TS_TEST:tomogram:fine:old", "parameters": {"tilt_series_id": "TS_TEST"}}
    )
    registry.add(older)
    assert _tilt_angles_for_tomogram(registry, _manifest(), older) == all_views
