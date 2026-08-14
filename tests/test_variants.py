from __future__ import annotations

from pathlib import Path

import pytest
import typer

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext, variant_label, variant_suffix
from cryoet_pipeline.cli import _fiducial_seed_artifact, _select_variant
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
    suffix = f":{variant}" if variant else ""
    return Artifact(
        id=f"TS_TEST:fiducial_seed{suffix}",
        kind=ArtifactKind.FIDUCIAL_SEED_MODEL,
        path=Path(f"seed{suffix}.mod"),
        parameters={"tilt_series_id": "TS_TEST"},
    )


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


def test_missing_variant_is_reported_clearly() -> None:
    registry = ArtifactRegistry.empty()
    registry.add(_seed("bead"))

    with pytest.raises(typer.BadParameter, match="no fiducial seed model"):
        _fiducial_seed_artifact(registry, _manifest(), variant="patch")


def test_select_variant_returns_the_only_match_without_a_label() -> None:
    only = _seed(None)
    assert (
        _select_variant(
            [only], what="fiducial seed model", tilt_series_id="TS_TEST", variant=None
        )
        is only
    )
