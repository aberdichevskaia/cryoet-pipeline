"""The post-picking report, on a synthetic project that has every kind of page.

Three series: TS_A has gone through picking with membranes, a CTF fit and a
saved correlation map; TS_B stopped at its tomogram; TS_C has a manifest and
nothing else. The registry, ids, lineage and files are shaped as the real
steps write them, so the lookups the report shares with the other commands
are exercised as they are in a lane.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.reports.collect import (
    PickTable,
    SeriesArtifacts,
    SummaryRow,
    lineage_qc,
    load_ctf,
    load_series,
    map_agreement,
)
from cryoet_pipeline.reports.imaging import (
    choose_depths,
    crop_centered,
    random_positions,
    rank_bands,
)
from cryoet_pipeline.score_preview import score_preview_from_volume, write_score_preview

pytest.importorskip("matplotlib")

from cryoet_pipeline.reports.sections import front  # noqa: E402

_ANGLES = [-40.0, -20.0, 0.0, 20.0, 40.0]
# Dose-symmetric: 0 deg first, then +20, -20, +40, -40. PriorRecordDose is the
# dose before each exposure, so it follows acquisition, not the file order.
_PRIOR_DOSE = {2: 0.0, 3: 3.0, 1: 6.0, 4: 9.0, 0: 12.0}
_SHAPE = (40, 96, 128)
_VOXEL = 10.0
_SIGMA = 0.01


def test_the_report_has_a_page_for_everything_each_series_has(tmp_path: Path) -> None:
    project = _project(tmp_path)

    result = CliRunner().invoke(
        cli.app, ["report", "--project", str(project), "--variant", "bin8"]
    )

    assert result.exit_code == 0, result.output
    pdf = project / "reports" / "report-bin8.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    # Front: title, contents, table, warnings; no picks-per-tomogram chart, since
    # only one series has picks. TS_A: tomogram, membranes, CTF, picks,
    # correlation map, two gallery pages. TS_B: tomogram. TS_C: a page saying
    # there is nothing yet. Appendix: one table.
    assert "(14 pages)" in result.output
    assert "could not be shown" not in result.output


def test_picks_per_tomogram_is_a_bar_per_series_with_picks() -> None:
    rows = [
        SummaryRow(series, None, 41, None, None, None, None, None, picks, above, None, {})
        for series, picks, above in (("TS_1", 700, 40), ("TS_2", 650, None), ("TS_3", None, None))
    ]

    figure = front.picks_per_tomogram_page(rows)

    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["TS_1", "TS_2"]


def test_an_existing_report_is_kept_unless_asked(tmp_path: Path) -> None:
    project = _project(tmp_path)
    runner = CliRunner()
    first = runner.invoke(cli.app, ["report", "--project", str(project), "--series", "TS_B"])
    second = runner.invoke(cli.app, ["report", "--project", str(project), "--series", "TS_B"])

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert "--overwrite" in second.output


def test_an_ambiguous_series_is_reported_rather_than_fatal(tmp_path: Path) -> None:
    """Two pick lists and no label: that series says why, the others are drawn."""

    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    original = registry.get("TS_A:picks:full:pytom-match-pick:bin8")
    registry.add(
        Artifact.model_validate(
            {
                **original.model_dump(mode="json"),
                "id": "TS_A:picks:full:pytom-match-pick:sweep",
                "variant": "sweep",
            }
        )
    )
    registry.commit(project / "artifacts.json")

    result = CliRunner().invoke(cli.app, ["report", "--project", str(project)])

    assert result.exit_code == 0, result.output
    assert re.search(r"TS_A: picks: found 2 candidates", result.output)


def test_ctf_rows_pair_with_views_by_z_value_and_take_their_dose(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    manifest = _manifest("TS_A")
    ctf_model = registry.get("TS_A:ctf_model:ctffind5")

    table = load_ctf(registry, ctf_model, manifest, lineage_qc(registry, [ctf_model]))

    # The fitted stack excludes nothing here but is ordered by the alignment,
    # which lists z values 4..0: row i is the alignment's transform i.
    assert table.z_value.tolist() == [4, 3, 2, 1, 0]
    assert table.tilt_angle_deg.tolist() == [40.0, 20.0, 0.0, -20.0, -40.0]
    assert table.dose is not None
    assert table.dose.tolist() == [_PRIOR_DOSE[z] for z in (4, 3, 2, 1, 0)]
    assert table.fit_resolution_angstrom is not None
    assert table.fit_resolution_angstrom[table.nearest_view(0.0)] == pytest.approx(8.0)
    assert table.nyquist_angstrom == pytest.approx(5.4)


def test_qc_documents_come_from_this_result_and_not_another_lane(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    # A second lane's reconstruction of the same series, which failed.
    other = _write_json(tmp_path / "other_qc.json", {"status": "fail", "warnings": ["bin4"]})
    registry.add(
        Artifact(
            id="TS_A:tomogram:fine:bin4",
            kind=ArtifactKind.TOMOGRAM,
            path=tmp_path / "missing.zarr",
            parent_ids=["TS_A:alignment:fine"],
            variant="bin4",
            parameters={"tilt_series_id": "TS_A"},
        )
    )
    registry.add(
        Artifact(
            id="TS_A:qc:reconstruction:fine:bin4",
            kind=ArtifactKind.QC,
            path=other,
            parent_ids=["TS_A:tomogram:fine:bin4"],
            parameters={"qc_type": "fine_reconstruction", "tilt_series_id": "TS_A"},
        )
    )

    records = lineage_qc(registry, [registry.get("TS_A:picks:full:pytom-match-pick:bin8")])

    assert records["fine_reconstruction"].status == "pass"
    # The picking QC's status is read from its document, not the registry entry.
    assert records["particle_picking"].status == "warning"
    assert records["fine_alignment"].warnings == ("one view has a large residual",)
    assert "ctf_estimation" in records


def test_a_map_read_with_its_axes_swapped_fails_the_check(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    volume = rng.normal(0.0, 0.002, size=(20, 30, 30)).astype(np.float32)
    x = rng.integers(2, 28, size=12).astype(float)
    y = rng.integers(2, 28, size=12).astype(float)
    z = rng.integers(2, 18, size=12).astype(float)
    score = rng.uniform(0.08, 0.15, size=12)
    volume[z.astype(int), y.astype(int), x.astype(int)] = score
    table = PickTable(x_px=x, y_px=y, z_px=z, score=score)

    right = score_preview_from_volume(volume, slab_thickness_px=5)
    swapped = score_preview_from_volume(np.swapaxes(volume, 1, 2), slab_thickness_px=5)

    assert map_agreement(right, table) == pytest.approx(1.0)
    assert map_agreement(swapped, table) < 0.5


def test_bands_are_thirds_of_the_ranking_shown_at_evenly_spaced_ranks() -> None:
    bands = rank_bands(30, 4)

    assert [band.name for band in bands] == ["high", "middle", "low"]
    assert [(band.first_rank, band.last_rank) for band in bands] == [(1, 10), (11, 20), (21, 30)]
    assert bands[0].positions.tolist() == [0, 3, 6, 9]
    assert bands[2].positions.tolist() == [20, 23, 26, 29]
    # Fewer picks than tiles: every pick is shown once, and empty bands are dropped.
    assert [band.positions.tolist() for band in rank_bands(2, 4)] == [[0], [1]]


def test_random_positions_sit_at_pick_depths_inside_the_volume() -> None:
    depths = np.asarray([12.0, 14.0, 19.0])

    positions = random_positions((40, 96, 128), 8, depths, 50, seed=7)

    assert set(positions[:, 0].tolist()) <= {12, 14, 19}
    assert positions[:, 1].min() >= 8 and positions[:, 1].max() <= 96 - 1 - 8
    assert positions[:, 2].min() >= 8 and positions[:, 2].max() <= 128 - 1 - 8
    assert np.array_equal(positions, random_positions((40, 96, 128), 8, depths, 50, seed=7))


def test_depths_follow_the_map_slabs_and_spread_over_a_thin_specimen() -> None:
    preview = score_preview_from_volume(np.zeros((40, 4, 4), np.float32), slab_thickness_px=10)

    # Picks spread through the depth: one slab per quartile.
    # Quartiles 12.5, 20 and 27.5 fall in two slabs; the middle one's neighbour fills in.
    spread = choose_depths(40, np.asarray([5.0, 15.0, 25.0, 35.0]), preview, 2)
    assert [(depth.start, depth.stop) for depth in spread] == [(10, 20), (20, 30), (30, 40)]
    assert [depth.center for depth in spread] == [14, 24, 34]
    # A thin specimen: all quartiles in one slab, so its neighbours are shown too.
    thin = choose_depths(40, np.asarray([21.0, 22.0, 23.0, 24.0]), preview, 2)
    assert [(depth.start, depth.stop) for depth in thin] == [(10, 20), (20, 30), (30, 40)]
    # Without a map, slabs of the asked half-width around the quartiles.
    plain = choose_depths(40, np.asarray([21.0, 22.0, 23.0, 24.0]), None, 2)
    assert all(depth.stop - depth.start == 5 for depth in plain)


def test_a_crop_past_the_edge_is_padded_rather_than_shifted() -> None:
    volume = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)

    box = crop_centered(volume, (0, 0, 0), 2)

    assert box.shape == (5, 5, 5)
    assert box[2, 2, 2] == volume[0, 0, 0]
    assert np.isnan(box[0, 0, 0])


def test_loading_keeps_going_past_a_part_it_cannot_read(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    summary = registry.get("TS_A:ctf_model:ctffind5").parameters["summary_path"]
    Path(summary).write_text("1 not-a-number\n")

    report = load_series(
        registry,
        SeriesArtifacts(
            manifest=_manifest("TS_A"),
            picks=registry.get("TS_A:picks:full:pytom-match-pick:bin8"),
            tomogram=registry.get("TS_A:tomogram:fine:bin8"),
            ctf_model=registry.get("TS_A:ctf_model:ctffind5"),
        ),
    )

    assert report.ctf is None
    assert any(problem.startswith("CTF:") for problem in report.problems)
    assert report.picks is not None and report.picks.count == 36
    assert report.correlation is not None
    assert report.correlation.agreement == pytest.approx(1.0)


# --- the synthetic project -------------------------------------------------------------


def _project(root: Path) -> Path:
    project = root / "project"
    (project / "manifests").mkdir(parents=True)
    for series in ("TS_A", "TS_B", "TS_C"):
        (project / "manifests" / f"{series}.json").write_text(
            _manifest(series).model_dump_json(indent=1)
        )
    registry = ArtifactRegistry.empty()
    _full_series(project, registry)
    _tomogram_only(project, registry)
    registry.write(project / "artifacts.json")
    return project


def _manifest(series: str) -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id=series,
        source_mdoc=Path(f"{series}.mdoc"),
        raw_pixel_spacing_angstrom=1.25,
        images=[
            TiltImage(
                z_value=z,
                tilt_angle_deg=angle,
                subframe_path=f"{series}_{z}.tif",
                num_subframes=8,
                pixel_spacing_angstrom=1.25,
                binning=1,
                metadata={"PriorRecordDose": str(_PRIOR_DOSE[z])},
            )
            for z, angle in enumerate(_ANGLES)
        ],
    )


def _full_series(project: Path, registry: ArtifactRegistry) -> None:
    series = "TS_A"
    rng = np.random.default_rng(11)
    out = project / series
    out.mkdir()

    alignment_path = out / "fine_alignment.json"
    alignment_path.write_text(
        TiltAlignment(
            tilt_series_id=series,
            backend="imod",
            stage="fine",
            input_stack_id=f"{series}:tilt_stack",
            input_binning=4,
            tilt_axis_angle_deg=85.0,
            transform_semantics="global",
            transforms=[
                AlignmentTransform(
                    z_value=z, tilt_angle_deg=_ANGLES[z], a11=1.0, a12=0.0, a21=0.0, a22=1.0,
                    shift_x_px=0.0, shift_y_px=0.0,
                )
                for z in (4, 3, 2, 1, 0)
            ],
        ).model_dump_json()
    )
    alignment = _artifact(f"{series}:alignment:fine", ArtifactKind.ALIGNMENT, alignment_path,
                          series, parameters={"stage": "fine"})
    registry.add(alignment)
    _qc(registry, out, series, "fine_alignment", alignment.id,
        {"status": "warning", "warnings": ["one view has a large residual"],
         "residual_mean_nm": 1.4})

    ctf_stack = _artifact(f"{series}:aligned_tilt_stack:bin2", ArtifactKind.ALIGNED_TILT_STACK,
                          out / "ctf_stack.mrc", series, parents=[alignment.id])
    registry.add(ctf_stack)
    summary = out / "ctffind.txt"
    fits = [14.0, 10.0, 8.0, 11.0, 15.0]
    summary.write_text(
        "# view defocus1 defocus2 azimuth phase cc fit\n"
        + "".join(
            f"{view} {30000 + 400 * view:.1f} {29000 + 400 * view:.1f} 30.0 0.0 0.12 {fit}\n"
            for view, fit in enumerate(fits, 1)
        )
    )
    diagnostic = out / "diagnostic.mrc"
    with mrcfile.new(diagnostic) as handle:
        handle.set_data(rng.normal(size=(5, 32, 32)).astype(np.float32))
    ctf_model = _artifact(
        f"{series}:ctf_model:ctffind5", ArtifactKind.CTF_MODEL, out / "ctf.defocus", series,
        parents=[ctf_stack.id, alignment.id],
        parameters={
            "backend": "ctffind5", "summary_path": str(summary),
            "diagnostic_path": str(diagnostic), "voltage_kv": 300.0,
            "spherical_aberration_mm": 2.7, "amplitude_contrast": 0.07,
            "pixel_spacing_angstrom": 2.7, "hint_defocus_um": 3.05,
        },
    )
    registry.add(ctf_model)
    _qc(registry, out, series, "ctf_estimation", ctf_model.id,
        {"status": "pass", "warnings": [], "nyquist_angstrom": 5.4, "mean_defocus_um": 3.1})

    corrected = _artifact(f"{series}:aligned_tilt_stack:ctf_corrected:bin8",
                          ArtifactKind.ALIGNED_TILT_STACK, out / "corrected.mrc", series,
                          parents=[ctf_model.id, alignment.id], variant="bin8")
    registry.add(corrected)

    volume = rng.normal(0.0, 1.0, size=_SHAPE).astype(np.float32)
    picks_xyz = _particles(rng)
    zz, yy, xx = np.mgrid[0 : _SHAPE[0], 0 : _SHAPE[1], 0 : _SHAPE[2]]
    for x, y, z in picks_xyz[:24]:
        volume -= 3.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * 2.5**2))
    tomogram_path = out / "tomogram.zarr"
    zarr.save(tomogram_path, volume)
    tomogram = _artifact(
        f"{series}:tomogram:fine:bin8", ArtifactKind.TOMOGRAM, tomogram_path, series,
        parents=[corrected.id, alignment.id], variant="bin8",
        parameters={"tomogram_branch": "full", "included_z_values": [0, 1, 2, 3, 4],
                    "excluded_z_values": [], "ctf_corrected": True},
        shape=_SHAPE, axis_order=AxisOrder.ZYX, pixel_spacing_angstrom=_VOXEL,
    )
    registry.add(tomogram)
    _qc(registry, out, series, "fine_reconstruction", tomogram.id,
        {"status": "pass", "warnings": []})

    membrane = np.zeros(_SHAPE, dtype=np.uint8)
    membrane[11:13, 10:80, 15:110] = 1
    segmentation_path = out / "segmentation.zarr"
    zarr.save(segmentation_path, membrane)
    segmentation = _artifact(
        f"{series}:segmentation:full:membrain-seg:bin8", ArtifactKind.SEGMENTATION,
        segmentation_path, series, parents=[tomogram.id], variant="bin8",
        shape=_SHAPE, axis_order=AxisOrder.ZYX, pixel_spacing_angstrom=_VOXEL,
    )
    registry.add(segmentation)
    _qc(registry, out, series, "tomogram_segmentation", segmentation.id,
        {"status": "pass", "warnings": [], "foreground_fraction": float(membrane.mean())})

    box = 16
    grid = np.mgrid[0:box, 0:box, 0:box] - (box - 1) / 2.0
    radius_sq = (grid**2).sum(axis=0)
    template_path = out / "template.mrc"
    with mrcfile.new(template_path) as handle:
        handle.set_data((-np.exp(-radius_sq / (2 * 3.0**2))).astype(np.float32))
        handle.voxel_size = _VOXEL
    mask_path = out / "mask.mrc"
    with mrcfile.new(mask_path) as handle:
        handle.set_data((radius_sq <= 36).astype(np.float32))
        handle.voxel_size = _VOXEL
    template = _artifact(f"{series}:template:bin8", ArtifactKind.TEMPLATE, template_path,
                         series, parents=[tomogram.id], variant="bin8")
    registry.add(template)
    template_mask = _artifact(f"{series}:template_mask:bin8", ArtifactKind.TEMPLATE_MASK,
                              mask_path, series, parents=[template.id], variant="bin8",
                              parameters={"template_id": template.id})
    registry.add(template_mask)
    _qc(registry, out, series, "template_preparation", template.id,
        {"status": "pass", "warnings": [], "emdb_id": "EMD-0000", "low_pass_angstrom": 20.0,
         "mask_radius_px": 6.0, "inverted": True})

    match = _artifact(
        f"{series}:template_match:pytom-match-pick:bin8", ArtifactKind.TEMPLATE_MATCH,
        out / "job.json", series,
        parents=[tomogram.id, segmentation.id, template.id, template_mask.id], variant="bin8",
    )
    registry.add(match)

    scores = np.concatenate([rng.uniform(0.09, 0.16, 24), rng.uniform(0.05, 0.065, 12)])
    star = out / "particles.star"
    star.write_text(
        "data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
        "_rlnLCCmax #4\n"
        + "".join(f"{x} {y} {z} {s:.6f}\n" for (x, y, z), s in zip(picks_xyz, scores,
                                                                  strict=True))
    )
    score_volume = np.abs(rng.normal(0.0, 0.015, size=_SHAPE)).astype(np.float32)
    for (x, y, z), s in zip(picks_xyz, scores, strict=True):
        score_volume[z, y, x] = s
    preview_path = out / "scores_preview.npz"
    write_score_preview(score_preview_from_volume(score_volume, slab_thickness_px=12),
                        preview_path)
    sigma = scores / _SIGMA
    picking_qc = _write_json(
        out / "picking_qc.json",
        {
            "status": "warning", "warnings": ["the picks fill the whole thickness"],
            "particle_count": len(scores), "search_sigma": _SIGMA, "search_space": 1.0e7,
            "one_false_positive_sigma": 5.33, "score_cut_off_sigma": float(sigma.min()),
            "excess_over_noise_at_cut_off": 30.0, "coordinate_z_sd": 5.0,
            "uniform_z_sd": 7.0,
        },
    )
    picks = _artifact(
        f"{series}:picks:full:pytom-match-pick:bin8", ArtifactKind.PICKS, star, series,
        parents=[match.id], variant="bin8",
        parameters={
            "backend": "pytom-match-pick", "input_tomogram_id": tomogram.id,
            "segmentation_id": segmentation.id, "qc_path": str(picking_qc),
            "pytom_scores_preview_path": str(preview_path),
            "pytom_template_path": str(template_path),
            "pytom_template_mask_path": str(mask_path),
        },
    )
    registry.add(picks)
    registry.add(
        Artifact(
            id=f"{series}:qc:picking:pytom-match-pick:bin8",
            kind=ArtifactKind.QC,
            path=picking_qc,
            parent_ids=[picks.id],
            variant="bin8",
            parameters={"qc_type": "particle_picking", "tilt_series_id": series,
                        "status": "pass"},
        )
    )


def _tomogram_only(project: Path, registry: ArtifactRegistry) -> None:
    series = "TS_B"
    path = project / series / "tomogram.zarr"
    path.parent.mkdir()
    zarr.save(path, np.random.default_rng(5).normal(size=(24, 48, 64)).astype(np.float32))
    registry.add(
        _artifact(
            f"{series}:tomogram:fine", ArtifactKind.TOMOGRAM, path, series,
            parameters={"tomogram_branch": "full", "included_z_values": [1, 2, 3]},
            shape=(24, 48, 64), axis_order=AxisOrder.ZYX, pixel_spacing_angstrom=13.5,
        )
    )


def _particles(rng: np.random.Generator) -> list[tuple[int, int, int]]:
    """36 well separated positions inside the specimen layer, z 12..28."""

    chosen: list[tuple[int, int, int]] = []
    while len(chosen) < 36:
        candidate = (int(rng.integers(10, 118)), int(rng.integers(10, 86)),
                     int(rng.integers(12, 28)))
        if all(sum((a - b) ** 2 for a, b in zip(candidate, other, strict=True)) > 64
               for other in chosen):
            chosen.append(candidate)
    return chosen


def _artifact(
    artifact_id: str,
    kind: ArtifactKind,
    path: Path,
    series: str,
    *,
    parents: list[str] | None = None,
    variant: str | None = None,
    parameters: dict[str, object] | None = None,
    **fields: object,
) -> Artifact:
    if not path.exists() and path.suffix in {".mrc", ".defocus", ".json"}:
        path.write_text("")
    return Artifact(
        id=artifact_id,
        kind=kind,
        path=path,
        parent_ids=parents or [],
        variant=variant,
        parameters={"tilt_series_id": series, **(parameters or {})},
        **fields,  # type: ignore[arg-type]
    )


def _qc(
    registry: ArtifactRegistry,
    out: Path,
    series: str,
    qc_type: str,
    parent: str,
    document: dict[str, object],
) -> None:
    path = _write_json(out / f"{qc_type}_qc.json", document)
    registry.add(
        Artifact(
            id=f"{series}:qc:{qc_type}:{parent.split(':', 1)[1]}",
            kind=ArtifactKind.QC,
            path=path,
            parent_ids=[parent],
            parameters={"qc_type": qc_type, "tilt_series_id": series,
                        "status": document.get("status")},
        )
    )


def _write_json(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(document))
    return path
