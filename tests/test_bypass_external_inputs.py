from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import mrcfile
import numpy as np
import pytest
import zarr
from typer.testing import CliRunner, Result

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.bypass_external_inputs import (
    import_external_picks_for_extraction,
    import_external_tomogram_for_extraction,
)
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    TiltImage,
    TiltSeriesManifest,
)

REGISTRY_TOMOGRAM_ID = "TS_TEST:tomogram:fine"
REGISTRY_PICKS_ID = "TS_TEST:picks:pytom-match-pick"


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


def _write_manifest(tmp_path: Path) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_manifest().model_dump_json())
    return manifest_path


def _registry_tomogram(tmp_path: Path, *, variant: str | None = None) -> Artifact:
    """A tomogram as reconstruction registers one, with a canonical Zarr."""

    suffix = f":{variant}" if variant else ""
    path = tmp_path / f"tomogram{suffix}.zarr"
    zarr.save(path, np.zeros((4, 6, 6), dtype=np.float32))
    return Artifact(
        id=f"{REGISTRY_TOMOGRAM_ID}{suffix}",
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        shape=(4, 6, 6),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=1.35,
        variant=variant,
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )


def _registry_picks(tmp_path: Path, *, found_in: Artifact) -> Artifact:
    """Picks as picking registers them: naming the tomogram they were found in."""

    path = tmp_path / "picks.star"
    path.write_text("data_particles\nloop_\n_rlnCoordinateX #1\n1.0\n")
    return Artifact(
        id=REGISTRY_PICKS_ID,
        kind=ArtifactKind.PICKS,
        path=path,
        parent_ids=[found_in.id],
        parameters={
            "tilt_series_id": "TS_TEST",
            "input_tomogram_id": found_in.id,
            "backend": "pytom-match-pick",
        },
    )


def _write_registry(tmp_path: Path, artifacts: Sequence[Artifact]) -> Path:
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry.empty()
    registry.extend(artifacts)
    registry.write(registry_path)
    return registry_path


def _write_external_tomogram(
    path: Path,
    *,
    shape: tuple[int, ...] = (4, 6, 6),
    pixel_spacing_angstrom: float = 13.48,
    nonfinite: bool = False,
) -> Path:
    """An MRC as some other pipeline would have reconstructed it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    if nonfinite:
        data.flat[0] = np.nan
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = pixel_spacing_angstrom
    return path


def _write_external_picks(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "data_particles\nloop_\n_rlnCoordinateX #1\n_rlnLCCmax #2\n2.0 0.31\n"
    )
    return path


def _capture_extraction_inputs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Artifact]:
    """Stand in for extraction and record which two artifacts it was handed.

    The question these tests ask is not what `cubic-crop` computes but which
    tomogram and which picks the command decided to hand it, which is exactly
    what the bypass changed and what no test covered.
    """

    captured: dict[str, Artifact] = {}

    def fake_extract_and_register(
        backend: object,
        tomogram: Artifact,
        picks: Artifact,
        manifest: TiltSeriesManifest,
        context: object,
        registry: ArtifactRegistry,
        *,
        replace_existing: bool = False,
    ) -> list[Artifact]:
        del backend, manifest, context, replace_existing
        captured["tomogram"] = tomogram
        captured["picks"] = picks

        out = Path(str(tomogram.path)).parent
        qc_path = out / "extraction_qc.json"
        qc_path.parent.mkdir(parents=True, exist_ok=True)
        qc_path.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "requested_particle_count": 1,
                    "extracted_particle_count": 1,
                    "warnings": [],
                }
            )
        )
        artifacts = [
            Artifact(
                id="TS_TEST:particle_stack:cubic-crop",
                kind=ArtifactKind.PARTICLE_STACK,
                path=out / "particles.zarr",
                parent_ids=[tomogram.id, picks.id],
                parameters={"tilt_series_id": "TS_TEST"},
            ),
            Artifact(
                id="TS_TEST:qc:particle_extraction",
                kind=ArtifactKind.QC,
                path=qc_path,
                parameters={"tilt_series_id": "TS_TEST", "status": "pass"},
            ),
        ]
        registry.extend(artifacts, replace=True)
        return artifacts

    monkeypatch.setattr(cli, "extract_and_register", fake_extract_and_register)
    return captured


def _invoke(
    tmp_path: Path,
    manifest_path: Path,
    registry_path: Path,
    *extra: str,
) -> Result:
    return CliRunner().invoke(
        cli.app,
        [
            "extract-particles",
            "--manifest",
            str(manifest_path),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "out"),
            "--box-size-px",
            "4",
            *extra,
        ],
    )


def test_without_bypass_flags_the_tomogram_still_comes_from_the_picks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lineage path must survive having a bypass branch added beside it.

    Two tomograms are registered so a label-based lookup could not choose, and
    the picks name the second. Reaching it proves the command still follows what
    the picks recorded rather than falling into the bypass branch's older
    tomogram-first lookup.
    """

    first = _registry_tomogram(tmp_path)
    second = _registry_tomogram(tmp_path, variant="bin8")
    picks = _registry_picks(tmp_path, found_in=second)
    registry_path = _write_registry(tmp_path, [first, second, picks])
    captured = _capture_extraction_inputs(monkeypatch)

    result = _invoke(tmp_path, _write_manifest(tmp_path), registry_path)

    assert result.exit_code == 0, result.output
    assert captured["tomogram"].id == f"{REGISTRY_TOMOGRAM_ID}:bin8"
    assert captured["picks"].id == REGISTRY_PICKS_ID


def test_external_tomogram_and_picks_bypass_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both flags together: neither registered artifact is consulted.

    This is the case the bypass exists for — a tomogram reconstructed elsewhere
    and picks made on it, with no lineage in this registry to follow.
    """

    tomogram = _registry_tomogram(tmp_path)
    picks = _registry_picks(tmp_path, found_in=tomogram)
    registry_path = _write_registry(tmp_path, [tomogram, picks])
    captured = _capture_extraction_inputs(monkeypatch)
    external_mrc = _write_external_tomogram(tmp_path / "external" / "tomo.mrc")
    external_star = _write_external_picks(tmp_path / "external" / "picks.star")

    result = _invoke(
        tmp_path,
        _write_manifest(tmp_path),
        registry_path,
        "--tomogram-mrc",
        str(external_mrc),
        "--picks-star",
        str(external_star),
    )

    assert result.exit_code == 0, result.output
    assert captured["tomogram"].id == "TS_TEST:tomogram:external-bypass"
    assert captured["tomogram"].pixel_spacing_angstrom == pytest.approx(13.48)
    assert captured["picks"].id == "TS_TEST:picks:external-bypass"
    assert captured["picks"].path == external_star
    # The external picks refer to the external tomogram and say so, which is
    # what keeps `_tomogram_from_picks` correct for anything reading them later.
    assert captured["picks"].parent_ids == [captured["tomogram"].id]
    assert (
        captured["picks"].parameters["input_tomogram_id"] == captured["tomogram"].id
    )


def test_external_picks_alone_take_the_tomogram_from_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picks from elsewhere, cropped from a tomogram this pipeline built.

    The imported picks must then name that registered tomogram, not invent a
    parent, or their lineage would dead-end the moment anything followed it.
    """

    tomogram = _registry_tomogram(tmp_path)
    registry_path = _write_registry(tmp_path, [tomogram])
    captured = _capture_extraction_inputs(monkeypatch)
    external_star = _write_external_picks(tmp_path / "external" / "picks.star")

    result = _invoke(
        tmp_path,
        _write_manifest(tmp_path),
        registry_path,
        "--picks-star",
        str(external_star),
    )

    assert result.exit_code == 0, result.output
    assert captured["tomogram"].id == REGISTRY_TOMOGRAM_ID
    assert captured["picks"].id == "TS_TEST:picks:external-bypass"
    assert captured["picks"].parent_ids == [REGISTRY_TOMOGRAM_ID]


def test_a_bypassed_run_can_still_name_the_lane_it_reads_from(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--variant` has to keep working on the side the bypass does not replace.

    Only one of the two inputs is external here, so the other is still a registry
    lookup and still has to be able to choose between lanes. The bypass predates
    `--variant` reaching these lookups, and without passing it through the flag
    goes quiet exactly when one external input is supplied: two lanes, and the
    run refuses with a count instead of reading the one that was asked for.
    """

    registry_path = _write_registry(
        tmp_path,
        [_registry_tomogram(tmp_path), _registry_tomogram(tmp_path, variant="bin8")],
    )
    captured = _capture_extraction_inputs(monkeypatch)
    external_star = _write_external_picks(tmp_path / "external" / "picks.star")

    result = _invoke(
        tmp_path,
        _write_manifest(tmp_path),
        registry_path,
        "--picks-star",
        str(external_star),
        "--variant",
        "bin8",
    )

    assert result.exit_code == 0, result.output
    assert captured["tomogram"].id == f"{REGISTRY_TOMOGRAM_ID}:bin8"


def test_an_external_tomogram_alone_overrides_what_the_picks_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hazardous combination, pinned so it stays deliberate.

    Registered picks carry the id of the volume their coordinates refer to, and
    `--tomogram-mrc` substitutes a different one. Honouring the flag is the
    point of a bypass, but it is exactly the mismatch `_tomogram_from_picks`
    exists to prevent: unless the external MRC is that same volume, every box
    lands in the wrong place and nothing says so. This test asserts the
    divergence rather than the safety, so that if the behaviour is ever tightened
    — a warning, a refusal — it is a decision someone makes, not a silent drift.
    """

    tomogram = _registry_tomogram(tmp_path)
    picks = _registry_picks(tmp_path, found_in=tomogram)
    registry_path = _write_registry(tmp_path, [tomogram, picks])
    captured = _capture_extraction_inputs(monkeypatch)
    external_mrc = _write_external_tomogram(tmp_path / "external" / "tomo.mrc")

    result = _invoke(
        tmp_path,
        _write_manifest(tmp_path),
        registry_path,
        "--tomogram-mrc",
        str(external_mrc),
    )

    assert result.exit_code == 0, result.output
    assert captured["tomogram"].id == "TS_TEST:tomogram:external-bypass"
    assert captured["picks"].id == REGISTRY_PICKS_ID
    assert captured["picks"].parameters["input_tomogram_id"] == REGISTRY_TOMOGRAM_ID
    assert captured["picks"].parameters["input_tomogram_id"] != captured["tomogram"].id


def test_the_bypass_artifacts_land_in_the_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the run cropped from has to be recoverable after it.

    The imported artifacts are written to the registry file, so a stack made
    this way still records which volume and which picks produced it.
    """

    registry_path = _write_registry(tmp_path, [])
    _capture_extraction_inputs(monkeypatch)
    external_mrc = _write_external_tomogram(tmp_path / "external" / "tomo.mrc")
    external_star = _write_external_picks(tmp_path / "external" / "picks.star")

    result = _invoke(
        tmp_path,
        _write_manifest(tmp_path),
        registry_path,
        "--tomogram-mrc",
        str(external_mrc),
        "--picks-star",
        str(external_star),
    )

    assert result.exit_code == 0, result.output
    written = ArtifactRegistry.load(registry_path)
    assert written.get("TS_TEST:tomogram:external-bypass").backend == "external-bypass"
    assert written.get("TS_TEST:picks:external-bypass").backend == "external-bypass"


def test_an_external_tomogram_is_read_into_a_canonical_zarr(tmp_path: Path) -> None:
    """The MRC is copied into the Zarr the rest of the pipeline expects."""

    external_mrc = _write_external_tomogram(tmp_path / "tomo.mrc")

    artifact = import_external_tomogram_for_extraction(
        mrc_path=external_mrc,
        tilt_series_id="TS_TEST",
        out=tmp_path / "out",
    )

    assert artifact.kind is ArtifactKind.TOMOGRAM
    assert artifact.axis_order is AxisOrder.ZYX
    assert artifact.shape == (4, 6, 6)
    assert artifact.covers("TS_TEST")
    stored = cast(Any, zarr.open(str(artifact.path), mode="r"))
    assert stored.shape == (4, 6, 6)
    assert np.array_equal(
        np.asarray(stored[:]), np.arange(144, dtype=np.float32).reshape(4, 6, 6)
    )


def test_a_two_dimensional_mrc_is_not_a_tomogram(tmp_path: Path) -> None:
    external_mrc = _write_external_tomogram(tmp_path / "flat.mrc", shape=(6, 6))

    with pytest.raises(ValueError, match="3D"):
        import_external_tomogram_for_extraction(
            mrc_path=external_mrc,
            tilt_series_id="TS_TEST",
            out=tmp_path / "out",
        )


def test_an_external_tomogram_without_a_pixel_size_is_refused(tmp_path: Path) -> None:
    """Extraction converts angstrom to voxels, so an unscaled MRC is unusable."""

    external_mrc = _write_external_tomogram(
        tmp_path / "tomo.mrc", pixel_spacing_angstrom=0.0
    )

    with pytest.raises(ValueError, match="pixel spacing"):
        import_external_tomogram_for_extraction(
            mrc_path=external_mrc,
            tilt_series_id="TS_TEST",
            out=tmp_path / "out",
        )


def test_a_nonfinite_external_tomogram_is_refused(tmp_path: Path) -> None:
    external_mrc = _write_external_tomogram(tmp_path / "tomo.mrc", nonfinite=True)

    with pytest.raises(ValueError, match="nonfinite"):
        import_external_tomogram_for_extraction(
            mrc_path=external_mrc,
            tilt_series_id="TS_TEST",
            out=tmp_path / "out",
        )


def test_a_missing_picks_star_is_refused(tmp_path: Path) -> None:
    tomogram = _registry_tomogram(tmp_path)

    with pytest.raises(FileNotFoundError, match="picks STAR file not found"):
        import_external_picks_for_extraction(
            star_path=tmp_path / "absent.star",
            tomogram=tomogram,
            tilt_series_id="TS_TEST",
        )


def test_imported_picks_record_the_template_box_stated_for_them(tmp_path: Path) -> None:
    # An external STAR has no pytom job; the box stated at import is what tells
    # a later extraction of these same picks where their centres are.
    tomogram = _registry_tomogram(tmp_path)
    star = tmp_path / "external.star"
    star.write_text("data_particles\n")
    stated = import_external_picks_for_extraction(
        star_path=star, tomogram=tomogram, tilt_series_id="TS_TEST", template_box_px=48
    )
    assert stated.parameters["pick_template_box_px"] == 48
    unstated = import_external_picks_for_extraction(
        star_path=star, tomogram=tomogram, tilt_series_id="TS_TEST"
    )
    assert "pick_template_box_px" not in unstated.parameters
