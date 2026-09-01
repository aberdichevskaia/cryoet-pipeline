from __future__ import annotations

import pytest

from cryoet_pipeline.resolution import (
    alignment_residual_from_low_pass,
    low_pass_from_alignment_residual,
)


def test_the_derivation_reproduces_the_numbers_recorded_for_the_real_series() -> None:
    """Pinned against measurements, not against the formula restated.

    These five residuals were measured by IMOD on EMPIAR-11538, and the first two
    values were written into the roadmap by hand before this function existed. If
    the derivation changes, these are what it has to keep agreeing with.
    """

    measured = {
        0.694: 30.8,
        0.812: 36.1,
        0.786: 34.9,
        0.922: 41.0,
        0.785: 34.9,
    }

    for residual_nm, expected_angstrom in measured.items():
        assert low_pass_from_alignment_residual(residual_nm) == pytest.approx(
            expected_angstrom, abs=0.05
        )


def test_the_hand_chosen_value_sits_inside_the_derived_range() -> None:
    """35 A was picked by hand and used for every series; the range is 31 to 41.

    Worth stating as a test because it is the argument for deriving per series
    rather than configuring once: one value was not wrong, it was wrong by a
    different amount for each series.
    """

    derived = [low_pass_from_alignment_residual(r) for r in (0.694, 0.812, 0.786, 0.922, 0.785)]

    assert min(derived) < 35.0 < max(derived)
    assert max(derived) / min(derived) == pytest.approx(0.922 / 0.694, rel=1e-6)


def test_a_coarser_safety_factor_only_ever_loosens_the_limit() -> None:
    """The bound is optimistic in a known direction, so headroom must go coarser.

    Alignment blur is the only term it accounts for; dose, thickness and the
    missing wedge all degrade further. Erring coarser costs some of the band;
    erring finer correlates against frequencies that hold nothing, which is how a
    score volume goes flat.
    """

    base = low_pass_from_alignment_residual(0.8)

    assert low_pass_from_alignment_residual(0.8, safety=1.5) > base
    assert low_pass_from_alignment_residual(0.8, safety=1.5) == pytest.approx(base * 1.5)


def test_the_inverse_reads_a_configured_value_back_into_residual_units() -> None:
    """So "is 35 A right for this series?" is one number against one number."""

    assert alignment_residual_from_low_pass(35.0) == pytest.approx(0.788, abs=0.001)
    assert low_pass_from_alignment_residual(
        alignment_residual_from_low_pass(35.0)
    ) == pytest.approx(35.0)


def test_nonsense_inputs_are_refused_rather_than_returning_a_number() -> None:
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="residual must be finite and positive"):
            low_pass_from_alignment_residual(bad)
    with pytest.raises(ValueError, match="safety factor must be finite and positive"):
        low_pass_from_alignment_residual(0.8, safety=0.0)
    with pytest.raises(ValueError, match="low-pass must be finite and positive"):
        alignment_residual_from_low_pass(-1.0)


def _registry_with_alignment(
    tmp_path, *, residual_nm: float | None, stage: str = "fine", series: str = "TS_TEST"
):
    """A registry shaped the way a real project's is, for the resolver to walk."""

    import json

    from cryoet_pipeline.artifacts import ArtifactRegistry
    from cryoet_pipeline.models import Artifact, ArtifactKind

    registry = ArtifactRegistry.empty()
    registry.add(
        Artifact(
            id=f"{series}:alignment:{stage}",
            kind=ArtifactKind.ALIGNMENT,
            path=tmp_path / "align.json",
            parameters={"tilt_series_id": series, "stage": stage},
        )
    )
    if residual_nm is not None:
        report = tmp_path / "align_qc.json"
        report.write_text(json.dumps({"residual_mean_nm": residual_nm}))
        registry.add(
            Artifact(
                id=f"{series}:qc:fine_alignment",
                kind=ArtifactKind.QC,
                path=report,
                parameters={"tilt_series_id": series, "qc_type": "fine_alignment"},
            )
        )
    return registry


def _manifest():
    from tests.test_subtomogram_backend import _manifest as build

    return build()


def test_the_resolver_finds_the_residual_through_the_qc_artifact(tmp_path) -> None:
    """The path no test covered, and the one the first version got wrong.

    The alignment artifact carries no `qc_path`, so reading one off it silently
    yielded nothing and the derivation would have refused on every real project.
    It is found as its own artifact instead, by kind and qc_type — which also
    means it works on registries already written, without re-running an
    alignment to add a field.
    """

    from cryoet_pipeline.cli import _derived_low_pass_angstrom

    registry = _registry_with_alignment(tmp_path, residual_nm=0.812)

    low_pass, source = _derived_low_pass_angstrom(registry, _manifest())

    assert low_pass == pytest.approx(36.1, abs=0.05)
    assert source == "0.812 nm"


def test_no_fine_alignment_is_refused_rather_than_falling_back_to_nyquist(
    tmp_path,
) -> None:
    """A default that happens to be wrong looks exactly like one that is right.

    Matching the whole band to Nyquist is what returned 2 candidates from a
    requested 2000, with `status: pass`. Half an hour of GPU is spent before
    anyone counts the picks, so this refuses at the argument instead.
    """

    import typer

    from cryoet_pipeline.cli import _derived_low_pass_angstrom

    registry = _registry_with_alignment(tmp_path, residual_nm=0.8, stage="coarse")

    with pytest.raises(typer.BadParameter, match="no fine alignment is registered"):
        _derived_low_pass_angstrom(registry, _manifest())


def test_an_alignment_without_a_usable_residual_is_refused_too(tmp_path) -> None:
    import typer

    from cryoet_pipeline.cli import _derived_low_pass_angstrom

    registry = _registry_with_alignment(tmp_path, residual_nm=None)

    with pytest.raises(typer.BadParameter, match="records no usable residual_mean_nm"):
        _derived_low_pass_angstrom(registry, _manifest())
