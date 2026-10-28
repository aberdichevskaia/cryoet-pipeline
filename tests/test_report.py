"""The post-picking report, on a synthetic project that has every kind of page.

Three series: TS_A has gone through picking with membranes, a CTF fit and a
saved correlation map; TS_B stopped at its tomogram; TS_C has a manifest and
nothing else. The registry, ids, lineage and files are shaped as the real
steps write them, so the lookups the report shares with the other commands
are exercised as they are in a lane.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
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
    QC_STEPS,
    PickTable,
    QcRecord,
    SeriesArtifacts,
    SeriesReport,
    SummaryRow,
    _per_view_residuals,
    lineage_qc,
    load_alignment,
    load_ctf,
    load_picks,
    load_series,
    map_agreement,
    per_view_table,
    summary_row,
)
from cryoet_pipeline.reports.imaging import (
    choose_depths,
    crop_centered,
    random_positions,
    rank_bands,
    sphere_sections,
)
from cryoet_pipeline.reports.timing import (
    STEPS,
    StepTime,
    Timing,
    duration,
    expected_steps,
    load_step_runs,
    panel_lines,
    series_timing,
    short_duration,
    table_cell,
    took,
)
from cryoet_pipeline.rotations import relion_euler_to_matrix, resample_volume_zyx
from cryoet_pipeline.score_preview import score_preview_from_volume, write_score_preview

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure  # noqa: E402

from cryoet_pipeline.reports import imaging, style  # noqa: E402
from cryoet_pipeline.reports.sections import alignment as alignment_pages  # noqa: E402
from cryoet_pipeline.reports.sections import ctf, front, gallery, picking  # noqa: E402
from cryoet_pipeline.reports.sections import tomogram as tomogram_pages  # noqa: E402

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
    # only one series has picks. TS_A: tilt alignment, aligned tilts, CTF,
    # tomogram, membranes, picks, pick scores, correlation map, two gallery
    # pages. TS_B: tomogram. TS_C: a page saying there is nothing yet.
    # Appendix: one table.
    assert "(17 pages)" in result.output
    assert "could not be shown" not in result.output


def test_picks_per_tomogram_is_a_bar_per_series_with_picks() -> None:
    rows = [
        SummaryRow(series, None, 41, None, None, None, None, None, picks, above, None, {})
        for series, picks, above in (("TS_1", 700, 40), ("TS_2", 650, None), ("TS_3", None, None))
    ]

    figure = front.picks_per_tomogram_page(rows)

    axes = figure.axes[0]
    assert [label.get_text() for label in axes.get_yticklabels()] == ["TS_1", "TS_2"]


def test_alignment_pairs_each_tilt_with_its_residual_and_its_beads(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    tomogram = registry.get("TS_A:tomogram:fine:bin8")

    info = load_alignment(registry, tomogram, lineage_qc(registry, [tomogram]))

    assert info is not None
    assert info.method == "gold beads"
    assert info.z_value.tolist() == [4, 3, 2, 1, 0]
    # The last table tiltalign printed, not the earlier fit's.
    assert info.residual_nm is not None
    assert info.residual_nm.tolist() == [0.6, 0.4, 0.3, 0.5, 1.1]
    # One bead was lost on the last tilt.
    assert info.points_per_view is not None
    assert info.points_per_view.tolist() == [6, 6, 6, 6, 5]
    assert info.aligned_z_values == (4, 3, 2, 1, 0)
    assert info.fiducial_count == 6
    # A tomogram that was never aligned here has no alignment to show, and that is fine.
    plain = registry.get("TS_B:tomogram:fine")
    assert load_alignment(registry, plain, {}) is None


def test_a_per_tilt_table_for_other_tilts_is_not_paired() -> None:
    table = (
        " view   rotation    tilt    deltilt     mag      dmag      skew    resid-nm\n"
        "    1      85.0     -20.0     0.00    1.0000    0.0000      0.00      0.30\n"
        "    2      85.0      20.0     0.00    1.0000    0.0000      0.00      0.40\n"
    )

    assert [row["resid-nm"] for row in per_view_table(table)] == [0.3, 0.4]
    assert _per_view_residuals(table, np.asarray([-20.0, 20.0])) is not None
    assert _per_view_residuals(table, np.asarray([-25.0, 25.0])) is None
    assert _per_view_residuals(table, np.asarray([-20.0, 0.0, 20.0])) is None


def test_a_slab_shows_each_pick_at_the_size_of_its_sphere_there() -> None:
    centres = np.asarray([20.0, 23.0, 26.0, 40.0])

    radii = sphere_sections(centres, plane=20.0, half_width=2.0, radius=5.0)

    # Inside the slab: full size. One voxel beyond its edge: smaller. Far away: absent.
    assert radii[0] == pytest.approx(5.0)
    assert radii[1] == pytest.approx(np.sqrt(25.0 - 1.0))
    assert radii[2] == pytest.approx(np.sqrt(25.0 - 16.0))
    assert np.isnan(radii[3])


def test_the_mask_radius_is_converted_to_tomogram_voxels(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    report = load_series(
        registry,
        SeriesArtifacts(
            manifest=_manifest("TS_A"),
            picks=registry.get("TS_A:picks:full:pytom-match-pick:bin8"),
            tomogram=registry.get("TS_A:tomogram:fine:bin8"),
            template=registry.get("TS_A:template:bin8"),
            template_mask=registry.get("TS_A:template_mask:bin8"),
        ),
    )
    assert report.template is not None and report.tomogram is not None
    # Same voxel size: the mask radius as recorded.
    assert report.particle_radius_px == pytest.approx(6.0)
    # A template at half the tomogram's voxel size has a radius half as many tomogram voxels.
    report.template = replace(report.template, voxel_spacing_angstrom=_VOXEL / 2.0)
    assert report.particle_radius_px == pytest.approx(3.0)
    assert "mask" in report.particle_radius()[1]


def test_the_ctf_page_knows_the_binning_it_was_fitted_at(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    ctf_model = registry.get("TS_A:ctf_model:ctffind5")

    table = load_ctf(registry, ctf_model, _manifest("TS_A"), lineage_qc(registry, [ctf_model]))

    assert table.fit_binning == 2
    assert ctf.fit_sampling(table) == "fitted at bin 2 (2.70 Å per pixel, Nyquist 5.4 Å)"


def test_a_gallery_tile_shows_its_particle_in_the_template_frame() -> None:
    """The tile turns a particle back by pytom's angles, as averaging does.

    `relion_euler_to_matrix` documents both directions: resampled with `A.T`, a
    template shows how it appears in the tomogram at that orientation; resampled
    with `A`, a particle cut from the tomogram comes back into the template's
    frame, which is what averaging does with pytom's angles. So a template put
    into a volume one way must come back out of the gallery looking like the
    template, and not when the angles are ignored or applied the other way. The
    object has no part that every turn leaves in place, so only the right turn
    can match it.
    """

    box = 25
    grid = np.mgrid[0:box, 0:box, 0:box] - (box - 1) / 2.0
    template = np.zeros((box, box, box), np.float32)
    for x, y, z, width in ((7, 0, 0, 1.5), (0, 4, 0, 1.5), (0, 0, -5, 1.5), (-3, -3, 2, 1.5)):
        squared = (grid[2] - x) ** 2 + (grid[1] - y) ** 2 + (grid[0] - z) ** 2
        template += np.exp(-squared / (2 * width**2)).astype(np.float32)
    matrix = relion_euler_to_matrix(40.0, 65.0, 110.0)
    in_tomogram = resample_volume_zyx(torch.from_numpy(template), matrix.T, np.zeros(3),
                                      device=torch.device("cpu")).numpy()
    volume = np.zeros((60, 60, 60), np.float32)
    volume[18:43, 18:43, 18:43] = in_tomogram

    def tile(rotation: np.ndarray | None) -> list[np.ndarray]:
        return gallery.oriented_views(volume, (30.0, 30.0, 30.0), rotation, offset_px=0.0,
                                      half=12, low_pass_angstrom=None,
                                      voxel_spacing_angstrom=10.0, slab_px=1)

    reference = imaging.central_views(template, 1)

    def agreement(views: list[np.ndarray]) -> float:
        return min(float(np.corrcoef(np.nan_to_num(view).ravel(), expected.ravel())[0, 1])
                   for view, expected in zip(views, reference, strict=True))

    assert agreement(tile(matrix)) > 0.95
    # Ignoring the angles, or applying them the other way round, does not.
    assert agreement(tile(None)) < 0.5
    assert agreement(tile(matrix.T)) < 0.5


def test_picks_carry_the_angles_and_low_pass_of_their_search(tmp_path: Path) -> None:
    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")

    table = load_picks(registry.get("TS_A:picks:full:pytom-match-pick:bin8"))

    assert table.has_orientations
    assert table.low_pass_angstrom == pytest.approx(25.0)
    assert table.template_box_px == 16


def test_the_score_overview_puts_every_tomogram_on_one_sigma_axis() -> None:
    reports = [_scored_report(f"TS_{number}", strong) for number, strong in
               ((1, 30), (2, 90), (3, 5), (4, 0))]
    contents = {"TS_1": 7, "TS_2": 14, "TS_3": 21, "TS_4": 28}

    builders = picking.score_overview_pages(reports, contents)
    figure = builders[0]()

    # Three tomograms to a page, one to a row.
    assert len(builders) == 2
    titled = [axes for axes in figure.axes if axes.get_title(loc="left")]
    assert [axes.get_title(loc="left").split(" · ")[0] for axes in titled] == [
        "TS_1", "TS_2", "TS_3"
    ]
    assert titled[1].get_title(loc="left").endswith("p. 14")
    # Each row is the same curve twice: on a log rank axis and on a linear one.
    assert [axes.get_xscale() for axes in figure.axes] == ["log", "linear"] * 3
    # Shared, and wide enough for the best pick of all and the floor of all.
    assert len({axes.get_ylim() for axes in figure.axes}) == 1
    low, high = figure.axes[0].get_ylim()
    assert low < 5.33 and high > max(float(report.picks.sigma.max()) for report in reports)
    assert len(picking.score_overview_pages((reports * 3)[:10], contents)) == 4


def test_the_score_overview_gives_each_its_own_axis_without_sigma() -> None:
    reports = [_scored_report("TS_1", 30), _scored_report("TS_2", 10, sigma=None)]

    figure = picking.score_overview_pages(reports, {})[0]()

    panels = [axes for axes in figure.axes if axes.get_title(loc="left")]
    assert panels[1].get_ylabel() == "Score (pytom LCCmax)"


def test_report_text_shows_at_most_three_decimals() -> None:
    """Floats printed to all their digits, as a QC warning on the cluster wrote them."""

    assert style.tidy_numbers(
        "fitted at 3.360000086016002 angstrom/pixel and applied at 13.440000344064009"
    ) == "fitted at 3.36 angstrom/pixel and applied at 13.44"
    # Three decimals or fewer stay as written, and so do dotted names.
    assert style.tidy_numbers("0.2679 nm, sd 0.209, 5.000 px, Dynamo 1.1.591") == (
        "0.268 nm, sd 0.209, 5.000 px, Dynamo 1.1.591"
    )
    assert style.tidy_numbers("0.30000000000000004, 2.0000, -53.50001°") == "0.3, 2, -53.5°"
    # A small number keeps its digits rather than becoming a misleading zero.
    assert style.tidy_numbers("chance 0.0000123") == "chance 1.23e-05"


def test_a_page_is_tidied_but_not_its_verbatim_text() -> None:
    figure = Figure()
    figure.text(0.0, 1.0, "residual 5.420793299877795 px")
    figure.text(0.0, 0.5, "pixel_size_angstrom = 1.1883", gid=style.VERBATIM)
    table = figure.add_subplot().table(cellText=[["0.30000000000000004"]])

    style.tidy_figure_text(figure)

    texts = {text.get_text() for text in figure.texts}
    assert texts == {"residual 5.421 px", "pixel_size_angstrom = 1.1883"}
    assert [cell.get_text().get_text() for cell in table.get_celld().values()] == ["0.3"]


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


def _scored_report(series: str, strong: int, *, sigma: float | None = _SIGMA) -> SeriesReport:
    """Picks shaped like a real search: `strong` high peaks, then a long flat tail."""

    rng = np.random.default_rng(len(series) + strong)
    scores = np.concatenate(
        [rng.uniform(0.09, 0.17, strong), rng.uniform(0.066, 0.075, 240)]
    )
    zeros = np.zeros(scores.size)
    return SeriesReport(
        artifacts=SeriesArtifacts(manifest=_manifest(series)),
        picks=PickTable(
            x_px=zeros, y_px=zeros, z_px=zeros, score=scores, search_sigma=sigma,
            floor_sigma=5.33 if sigma is not None else None,
        ),
    )


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
    # Beads tracked on a bin4 stack, in the alignment's view order; one bead is lost
    # on the steepest tilt.
    tracking_path = out / "tracking.mrc"
    tracking_images = rng.normal(0.0, 1.0, size=(5, 64, 72)).astype(np.float32)
    beads = [(10.0, 12.0), (30.0, 40.0), (50.0, 20.0), (60.0, 50.0), (20.0, 55.0), (40.0, 8.0)]
    residual_rows = []
    for view in range(5):
        for number, (x, y) in enumerate(beads):
            if view == 4 and number == 5:
                continue
            tracking_images[view, int(y) - 1 : int(y) + 2, int(x) - 1 : int(x) + 2] -= 4.0
            residual_rows.append(f"{x:.2f} {y:.2f} {view} 0.10 -0.20")
    with mrcfile.new(tracking_path) as handle:
        handle.set_data(tracking_images)
        handle.voxel_size = 5.0
    tracking = _artifact(f"{series}:aligned_tilt_stack:fiducial", ArtifactKind.ALIGNED_TILT_STACK,
                         tracking_path, series, pixel_spacing_angstrom=5.0, binning=4,
                         parameters={"purpose": "fiducial_tracking",
                                     "included_z_values": [4, 3, 2, 1, 0]})
    registry.add(tracking)
    model = _artifact(f"{series}:fiducial_model", ArtifactKind.FIDUCIAL_MODEL, out / "beads.json",
                      series, parents=[tracking.id], parameters={"backend": "imod-beadtrack"})
    registry.add(model)
    residual_file = out / "fine.resid"
    residual_file.write_text(f"{len(residual_rows)} residuals\n" + "\n".join(residual_rows) + "\n")
    log = out / "tiltalign.log"
    table = "".join(
        f"  {view:3d}  85.0  {_ANGLES[z]:7.1f}  0.00  1.0000  0.0000  0.00  {residual:6.2f}\n"
        for view, (z, residual) in enumerate(zip((4, 3, 2, 1, 0), (0.6, 0.4, 0.3, 0.5, 1.1),
                                                 strict=True), 1)
    )
    header = " view   rotation    tilt    deltilt     mag      dmag      skew    resid-nm\n"
    # An earlier fit's table comes first, as tiltalign prints one per fit.
    log.write_text(header + table.replace("1.10", "9.99") + "\n\n" + header + table + "\n")
    alignment = _artifact(f"{series}:alignment:fine", ArtifactKind.ALIGNMENT, alignment_path,
                          series, parents=[tracking.id, model.id],
                          parameters={"stage": "fine", "tiltalign_log_path": str(log),
                                      "residual_file_path": str(residual_file),
                                      "excluded_z_values": []})
    registry.add(alignment)
    _qc(registry, out, series, "fine_alignment", alignment.id,
        {"status": "warning", "warnings": ["one view has a large residual"],
         "residual_mean_nm": 1.4, "residual_sd_nm": 0.3, "fiducial_count": 6,
         "global_leave_out_error_nm": 1.6, "projection_point_count": 29,
         "pruned_point_count": 1, "applied_tilt_angle_offset_deg": 0.4,
         "recommended_x_axis_tilt_deg": -0.2, "minimum_tilt_rotation_deg": 85.0})

    ctf_stack = _artifact(f"{series}:aligned_tilt_stack:bin2", ArtifactKind.ALIGNED_TILT_STACK,
                          out / "ctf_stack.mrc", series, parents=[alignment.id], binning=2)
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

    corrected_path = out / "corrected.mrc"
    with mrcfile.new(corrected_path) as handle:
        handle.set_data(rng.normal(size=(5, _SHAPE[1], _SHAPE[2])).astype(np.float32))
        handle.voxel_size = _VOXEL
    corrected = _artifact(f"{series}:aligned_tilt_stack:ctf_corrected:bin8",
                          ArtifactKind.ALIGNED_TILT_STACK, corrected_path, series,
                          parents=[ctf_model.id, alignment.id], variant="bin8",
                          pixel_spacing_angstrom=_VOXEL, binning=8,
                          parameters={"included_z_values": [4, 3, 2, 1, 0]})
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

    # pytom writes its whole job, the low-pass and template box among it.
    job = _write_json(out / "job.json", {"low_pass": 25.0, "template_shape": [16, 16, 16]})
    match = _artifact(
        f"{series}:template_match:pytom-match-pick:bin8", ArtifactKind.TEMPLATE_MATCH,
        job, series,
        parents=[tomogram.id, segmentation.id, template.id, template_mask.id], variant="bin8",
    )
    registry.add(match)

    scores = np.concatenate([rng.uniform(0.09, 0.16, 24), rng.uniform(0.05, 0.065, 12)])
    angles = rng.uniform(0.0, 180.0, size=(len(scores), 3))
    star = out / "particles.star"
    star.write_text(
        "data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n"
        "_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n_rlnLCCmax #7\n"
        + "".join(
            f"{x} {y} {z} {rot:.2f} {tilt:.2f} {psi:.2f} {s:.6f}\n"
            for (x, y, z), (rot, tilt, psi), s in zip(picks_xyz, angles, scores, strict=True)
        )
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
            "pytom_job_json_path": str(job),
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


def test_aligned_tilts_page_shows_only_views_the_stack_holds() -> None:
    """correct-ctf can leave out the highest tilts; the page must not reach for them.

    On ryr1-bin8 (2026-10-19) every series lost z 0, its most negative tilt,
    and the page failed for all five with "no slice for z value 0".
    """

    import numpy as np

    from cryoet_pipeline.reports.collect import AlignmentInfo
    from cryoet_pipeline.reports.sections.alignment import _left_out_for_ctf, _shown_views

    alignment = AlignmentInfo(
        method="gold beads",
        z_value=np.arange(7, dtype=np.int64),
        tilt_angle_deg=np.asarray([-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0]),
        # The largest residual is on the view that was left out.
        residual_nm=np.asarray([0.9, 0.2, 0.3, 0.1, 0.4, 0.2, 0.5]),
        points_per_view=None,
        point_xy=None,
        point_view=None,
        tracking_stack=None,
        tracking_pixel_angstrom=None,
        aligned_stack=None,
        aligned_pixel_angstrom=None,
        aligned_z_values=(1, 2, 3, 4, 5, 6),
        excluded_z_values=(),
        fiducial_count=None,
        tilt_axis_angle_deg=None,
    )

    views = _shown_views(alignment)

    assert views and 0 not in views
    assert views == sorted(views)
    # The worst residual still shown is the worst among the views kept.
    assert 6 in views
    # And the page says which one is missing.
    assert _left_out_for_ctf(alignment) == [-60.0]


def test_ctf_page_colours_tilts_without_thon_rings(tmp_path: Path) -> None:
    """Yellow for a tilt kept with an interpolated defocus, red for one left out.

    The defocus drawn for both is the model's, the one corrected with, not the
    value CTFFIND settled on without rings.
    """

    from matplotlib.colors import to_rgba

    from cryoet_pipeline.reports import style

    project = _project(tmp_path)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    ctf_model = registry.get("TS_A:ctf_model:ctffind5")
    # Rows follow the alignment, z 4..0; the model interpolated z 4 and z 0 and
    # left z 0 out.
    ctf_model.path.write_text(
        "1 0 0.0 0.0 0.0 3\n"
        + "".join(
            f"{view} {view} 0.0 0.0 {nm:.1f} {nm:.1f} 0.0\n"
            for view, nm in enumerate((2500.0, 3080.0, 3120.0, 3160.0, 3900.0), 1)
        )
    )
    flagged = ctf_model.model_copy(
        update={
            "parameters": {
                **ctf_model.parameters,
                "interpolated_z_values": [4, 0],
                "excluded_z_values": [0],
            }
        }
    )

    table = load_ctf(registry, flagged, _manifest("TS_A"), lineage_qc(registry, [flagged]))

    assert table.interpolated is not None and table.left_out is not None
    assert table.interpolated.tolist() == [True, False, False, False, True]
    assert table.left_out.tolist() == [False, False, False, False, True]
    # The model's defocus where it interpolated, CTFFIND's elsewhere.
    assert table.mean_defocus_um[0] == pytest.approx(2.5)
    assert table.mean_defocus_um[4] == pytest.approx(3.9)
    assert table.mean_defocus_um[2] == pytest.approx((31200 + 30200) / 2 / 1e4)

    colours = ctf._view_colours(table)
    assert tuple(colours[0]) == to_rgba(style.INTERPOLATED_COLOUR)
    assert tuple(colours[4]) == to_rgba(style.LEFT_OUT_COLOUR)

    long_warning = "2 of 5 views stopped at the low end of the fit range ... no Thon rings"
    report = SeriesReport(
        artifacts=SeriesArtifacts(manifest=_manifest("TS_A")),
        ctf=table,
        qc={"ctf_estimation": QcRecord("ctf_estimation", "warning", (long_warning,), {}, "x")},
    )
    numbers = ctf._numbers(report, table)
    assert "QC: WARN, 2 of 5 tilts without rings:" in numbers
    assert "1 left out (red), 1 kept (yellow)" in numbers
    # The full text stays on the warnings page, not in this corner.
    assert "stopped at the low end" not in numbers


# --- run times ---------------------------------------------------------------------------

_T0 = datetime(2026, 10, 20, 8, 0, tzinfo=UTC).timestamp()

# TS_A's steps as a lane ran them: (step, start, seconds, the files it wrote).
# The fixture has no aligned stack of its own, so that step's run wrote nothing
# shown, and its time is not recorded.
_LANE_RUNS = (
    ("track-fiducials", 0, 30.0, ("beads.json",)),
    ("fine-align-tilt-series", 40, 150.0, ("fine_alignment.json", "fine_alignment_qc.json")),
    ("build-ctf-stack", 200, 30.0, ("ctf_stack.mrc",)),
    ("estimate-ctf", 240, 54.0, ("ctf.defocus", "ctf_estimation_qc.json")),
    ("build-final-aligned-stack", 300, 6.0, ()),
    ("correct-ctf", 400, 11.0, ("corrected.mrc",)),
    ("reconstruct-tomogram", 420, 12.0, ("tomogram.zarr", "fine_reconstruction_qc.json")),
    ("segment-tomogram", 500, 460.0,
     ("segmentation.zarr", "tomogram_segmentation_qc.json")),
    ("prepare-template", 1000, 20.0,
     ("template.mrc", "mask.mrc", "template_preparation_qc.json")),
    ("pick-particles", 1100, 1850.0, ("job.json", "particles.star", "picking_qc.json")),
)


def _row(series: str, step: str, start: float, seconds: float, *,
         variant: str | None = None, exit_code: int = 0) -> dict[str, object]:
    """One line of step_timings.jsonl as scripts/run_series.py writes it; old rows
    name no lane."""

    finished = datetime.fromtimestamp(_T0 + start + seconds, UTC)
    row: dict[str, object] = {
        "series": series, "step": step, "seconds": seconds, "disk_delta_bytes": 0,
        "exit_code": exit_code, "finished_at": finished.isoformat(timespec="seconds"),
    }
    if variant is not None:
        row.update(lane=f"lane-{variant}", variant=variant)
    return row


def _timed_project(root: Path, *, extra: tuple[dict[str, object], ...] = ()) -> Path:
    """The synthetic project with its lane's timings, and TS_A's files dated by them."""

    project = _project(root)
    rows = [_row("TS_A", step, start, seconds, variant="bin8")
            for step, start, seconds, _ in _LANE_RUNS]
    with (project / "step_timings.jsonl").open("w") as handle:
        for row in (*rows, *extra):
            handle.write(json.dumps(row) + "\n")
    for _, start, seconds, files in _LANE_RUNS:
        for name in files:
            moment = _T0 + start + seconds - 1.0
            os.utime(project / "TS_A" / name, (moment, moment))
    return project


def _timed_reports(project: Path) -> list[SeriesReport]:
    registry = ArtifactRegistry.load(project / "artifacts.json")
    runs = load_step_runs(project)
    return [
        load_series(registry, cli._report_artifacts(registry, _manifest(series), variant="bin8"),
                    runs)
        for series in ("TS_A", "TS_B", "TS_C")
    ]


def _step_seconds(timing: Timing) -> dict[str, float | None]:
    return {one.step.name: one.seconds for one in timing.steps}


def test_a_duration_reads_in_seconds_minutes_or_hours() -> None:
    assert duration(0.2) == "under 1 s"
    assert duration(12.4) == "12 s"
    assert duration(59.6) == "1 min"
    assert duration(252.0) == "4 min 12 s"
    assert duration(3599.6) == "1 h"
    # From an hour on, to the minute.
    assert duration(3899.0) == "1 h 5 min"
    assert duration(26 * 3600 + 10 * 60 + 29) == "26 h 10 min"
    assert short_duration(50.0) == "50 s"
    assert short_duration(45 * 60.0 + 10) == "45 min"
    assert short_duration(12.7 * 3600) == "12.7 h"
    assert short_duration(130 * 3600) == "130 h"


def test_a_sum_says_at_least_when_a_step_was_not_timed() -> None:
    first, second, third = STEPS[2], STEPS[3], STEPS[4]
    partial = Timing((StepTime(first, 12.0), StepTime(second, None), StepTime(third, 30.0)))
    nothing = Timing((StepTime(first, None),))
    whole = Timing((StepTime(first, 12.0), StepTime(third, 30.0)))

    assert (partial.seconds, partial.missing) == (42.0, 1)
    assert took(partial) == "took at least 42 s"
    assert took(whole) == "took 42 s"
    assert took(nothing) == "time not recorded"
    # A page no step is known to have made says nothing about time.
    assert took(Timing()) is None
    assert (table_cell(partial), table_cell(whole), table_cell(nothing)) == (
        "≥ 42 s", "42 s", "–")
    # Broken between steps, never inside a time.
    assert panel_lines(partial, 30) == [
        "Time: coarse alignment 12 s,", "  coarse check not recorded,", "  seeding 30 s"]
    assert panel_lines(nothing, 30) == []


def test_timing_rows_are_read_forgivingly(tmp_path: Path) -> None:
    rows = [
        json.dumps(_row("TS_A", "estimate-ctf", 0, 54.0, variant="bin8")),
        json.dumps(_row("TS_A", "estimate-ctf", 100, 3.0, exit_code=1)),
        json.dumps(_row("TS_A", "reconstruct-tomogram", 200, 12.0)),
        json.dumps({"series": "TS_A", "step": "correct-ctf", "seconds": "11", "exit_code": 0,
                    "finished_at": "2026-10-20T08:00:00+00:00"}),
        json.dumps({"series": "TS_A", "step": "correct-ctf", "seconds": 11.0, "exit_code": 0}),
        '{"series": "TS_A", "step": "pick-partic',
    ]
    (tmp_path / "step_timings.jsonl").write_text("\n".join(rows) + "\n")

    runs = load_step_runs(tmp_path)

    # The failed run, the text seconds, the missing time and the cut line are dropped.
    assert [(run.step, run.seconds, run.variant) for run in runs] == [
        ("estimate-ctf", 54.0, "bin8"), ("reconstruct-tomogram", 12.0, None)]
    assert runs[0].started_at == pytest.approx(_T0)
    assert load_step_runs(tmp_path / "no-such-project") == []


def test_each_step_takes_the_time_of_the_run_that_wrote_its_result(tmp_path: Path) -> None:
    """By the files' modification times, not by which row came last.

    The first decoy is the latest run of its step. The others hold a result's
    time and end before its real run, so that each would win, as the earlier of
    two runs, without the rule it is there for. None made what the report shows.
    """

    project = _timed_project(tmp_path, extra=(
        # Another lane reconstructed TS_A later, before rows named their lane.
        _row("TS_A", "reconstruct-tomogram", 5000, 100.0),
        # Picking under another label whose span holds the bin8 picks' time.
        _row("TS_A", "pick-particles", 1050, 1899.5, variant="bin4"),
        # A run that failed, and TS_B's run at TS_A's time.
        _row("TS_A", "segment-tomogram", 940, 19.5, exit_code=1),
        _row("TS_B", "prepare-template", 1000, 25.0),
        # A step no page shows, even where its span holds a result's time.
        _row("TS_A", "visualize-picks", 425, 6.5),
    ))

    first, second, third = _timed_reports(project)

    timing = first.timing
    assert _step_seconds(timing.page("alignment")) == {
        "track-fiducials": 30.0, "fine-align-tilt-series": 150.0}
    assert _step_seconds(timing.page("ctf")) == {
        "build-ctf-stack": 30.0, "estimate-ctf": 54.0, "correct-ctf": 11.0}
    assert _step_seconds(timing.page("aligned tilts")) == {"build-final-aligned-stack": None}
    assert _step_seconds(timing.page("tomogram")) == {"reconstruct-tomogram": 12.0}
    assert _step_seconds(timing.page("membranes")) == {"segment-tomogram": 460.0}
    assert _step_seconds(timing.page("picks")) == {
        "prepare-template": 20.0, "pick-particles": 1850.0}
    assert (timing.total.seconds, timing.total.missing) == (2617.0, 1)
    assert not any(problem.startswith("run times") for problem in first.problems)
    # TS_B's tomogram came with no run of its own; TS_C has nothing to time.
    assert _step_seconds(second.timing.total) == {"reconstruct-tomogram": None}
    assert third.timing.total.steps == ()


def test_a_result_written_as_its_step_ended_is_that_steps(tmp_path: Path) -> None:
    """A row's finish is cut down to the second, and a step writes as it ends.

    Shaped as ryr1's 096 and 097 were on 2026-10-14, when both lost their time:
    096's coarse alignment was written at 15:04:10 by the run its row puts at
    15:04:00-15:04:11, and 097's fine alignment at 15:07:26 by the one at
    15:07:19-15:07:26. The fractions of a second, and the runs after them, are
    made up as those runs would have run.
    """

    def at(minute: int, second: float) -> float:
        return datetime(2026, 10, 14, 15, minute, tzinfo=UTC).timestamp() + second

    def written(name: str, kind: ArtifactKind, moment: float, *parents: str) -> Artifact:
        path = tmp_path / name
        path.write_text("")
        os.utime(path, (moment, moment))
        return Artifact(id=name, kind=kind, path=path, parent_ids=list(parents))

    rows = [
        ("096", "align-tilt-series", 10.6, "15:04:11"),
        # Its own cut-down finish puts the check's start at 15:04:10.6.
        ("096", "qc-coarse-alignment", 6.4, "15:04:17"),
        ("097", "fine-align-tilt-series", 7.0, "15:07:26"),
        ("097", "build-ctf-stack", 18.0, "15:07:44"),
    ]
    (tmp_path / "step_timings.jsonl").write_text("".join(
        json.dumps({"series": series, "step": step, "seconds": seconds, "exit_code": 0,
                    "finished_at": f"2026-10-14T{finished}+00:00"}) + "\n"
        for series, step, seconds, finished in rows
    ))
    runs = load_step_runs(tmp_path)
    stack = written("stack.st", ArtifactKind.TILT_STACK, at(3, 0.0))
    coarse = written("coarse.json", ArtifactKind.ALIGNMENT, at(4, 10.8), "stack.st")
    check = written("check.json", ArtifactKind.QC, at(4, 16.5), "coarse.json")
    record = QcRecord("coarse_alignment_report", "pass", (), {}, "check.json")
    model = written("model.fid", ArtifactKind.FIDUCIAL_MODEL, at(5, 0.0))
    fine = written("fine.json", ArtifactKind.ALIGNMENT, at(7, 26.4), "model.fid")
    fit_stack = written("fit.st", ArtifactKind.ALIGNED_TILT_STACK, at(7, 43.5), "fine.json")

    first, _ = series_timing("096", [stack, coarse, check],
                             {"coarse_alignment_report": record}, runs)
    second, _ = series_timing("097", [model, fine, fit_stack], {}, runs)

    assert _step_seconds(first.page("alignment")) == {
        "align-tilt-series": 10.6, "qc-coarse-alignment": 6.4}
    assert _step_seconds(second.page("alignment")) == {
        "track-fiducials": None, "fine-align-tilt-series": 7.0}
    assert _step_seconds(second.page("ctf")) == {"build-ctf-stack": 18.0}


def test_a_result_rewritten_outside_the_runner_has_no_time(tmp_path: Path) -> None:
    project = _timed_project(tmp_path)
    # Reconstructed again by hand a day later: the file is newer than every run.
    later = _T0 + 86400.0
    os.utime(project / "TS_A" / "tomogram.zarr", (later, later))
    os.utime(project / "TS_A" / "fine_reconstruction_qc.json", (later, later))

    report = _timed_reports(project)[0]

    assert _step_seconds(report.timing.page("tomogram")) == {"reconstruct-tomogram": None}
    assert took(report.timing.page("tomogram")) == "time not recorded"
    assert table_cell(report.timing.total) == "≥ 43 min 25 s"


def test_a_project_copied_without_file_times_says_so(tmp_path: Path) -> None:
    project = _timed_project(tmp_path)
    # As cp leaves them: every file dated now, long after the runs.
    for path in (project / "TS_A").iterdir():
        os.utime(path)

    report = _timed_reports(project)[0]

    assert report.timing.total.seconds is None
    assert any(problem.startswith("run times: the timing file has 10 runs of this series, "
                                  "but none wrote the files shown here")
               for problem in report.problems)


def test_the_steps_a_result_needed_are_read_off_its_lineage() -> None:
    def made(name: str, kind: ArtifactKind, *parents: str, **parameters: object) -> Artifact:
        return Artifact(id=name, kind=kind, path=Path(name), parent_ids=list(parents),
                        parameters=dict(parameters))

    chain = [
        made("frame", ArtifactKind.CORRECTED_PROJECTION),
        made("stack", ArtifactKind.TILT_STACK, "frame"),
        made("coarse", ArtifactKind.ALIGNMENT, "stack"),
        made("tracking", ArtifactKind.ALIGNED_TILT_STACK, "stack", "coarse"),
        made("seed", ArtifactKind.FIDUCIAL_SEED_MODEL, "tracking"),
        made("model", ArtifactKind.FIDUCIAL_MODEL, "tracking", "seed"),
        made("fine", ArtifactKind.ALIGNMENT, "tracking", "model"),
        made("fit stack", ArtifactKind.ALIGNED_TILT_STACK, "stack", "fine"),
        made("ctf", ArtifactKind.CTF_MODEL, "fit stack", "fine"),
        made("final", ArtifactKind.ALIGNED_TILT_STACK, "stack", "fine"),
        made("corrected", ArtifactKind.ALIGNED_TILT_STACK, "final", "ctf"),
        made("tomogram", ArtifactKind.TOMOGRAM, "corrected", "fine"),
        made("template", ArtifactKind.TEMPLATE, "tomogram"),
        made("picks", ArtifactKind.PICKS, "tomogram"),
        made("filtered", ArtifactKind.PICKS, "picks", filtered_from="picks"),
        # A cut of the filtered list keeps its source's key, and adds its own.
        made("cut", ArtifactKind.PICKS, "filtered", filtered_from="picks",
             selected_from="filtered"),
    ]
    qc = {"coarse_alignment_report": object()}

    assert expected_steps(chain, qc) == [
        "prepare-tilt-series", "align-tilt-series", "qc-coarse-alignment",
        "generate-fiducial-seed", "track-fiducials", "fine-align-tilt-series",
        "build-ctf-stack", "estimate-ctf", "build-final-aligned-stack", "correct-ctf",
        "reconstruct-tomogram", "prepare-template", "pick-particles", "filter-picks",
        "select-picks",
    ]
    # An imported stack, a tomogram left uncorrected, and no coarse check.
    imported = [one for one in chain if one.id not in {"frame", "corrected"}]
    imported = [one.model_copy(update={"parent_ids": ["final", "fine"]})
                if one.id == "tomogram" else one for one in imported]
    names = expected_steps(imported, {})
    assert names[0] == "import-tilt-stack"
    assert "correct-ctf" not in names and "qc-coarse-alignment" not in names


def test_motion_correction_counts_in_the_series_total(tmp_path: Path) -> None:
    """It has no page; the timed run decides which entry step it was."""

    def written(name: str, kind: ArtifactKind, moment: float, *parents: str) -> Artifact:
        path = tmp_path / name
        path.write_text("")
        os.utime(path, (_T0 + moment, _T0 + moment))
        return Artifact(id=name, kind=kind, path=path, parent_ids=list(parents))

    frame = written("frame.mrc", ArtifactKind.CORRECTED_PROJECTION, 100)
    stack = written("stack.st", ArtifactKind.TILT_STACK, 590, "frame.mrc")
    tomogram = written("tomogram.zarr", ArtifactKind.TOMOGRAM, 900, "stack.st")
    motion = _row("TS_A", "prepare-tilt-series", 0, 600.0, variant="bin8")
    (tmp_path / "step_timings.jsonl").write_text(json.dumps(motion) + "\n")
    runs = load_step_runs(tmp_path)

    corrected, _ = series_timing("TS_A", [frame, stack, tomogram], {}, runs)
    # Without its corrected frames the lineage reads as an imported stack; the run says not.
    imported, _ = series_timing("TS_A", [stack, tomogram], {}, runs)

    for found in (corrected, imported):
        assert _step_seconds(found.total) == {
            "prepare-tilt-series": 600.0, "reconstruct-tomogram": None}
        assert found.page("").steps[0].step.label == "motion correction"


def test_each_page_heading_says_how_long_its_steps_took(tmp_path: Path) -> None:
    first, second, third = _timed_reports(_timed_project(tmp_path))
    rows = [summary_row(report) for report in (first, second, third)]

    def heading(figure: Figure) -> str:
        return figure.texts[1].get_text()

    def depths(report: SeriesReport) -> list[imaging.Depth]:
        assert report.tomogram is not None
        return choose_depths(report.tomogram.shape[0], None, None, 3)

    page = alignment_pages.alignment_page(first)
    assert heading(page).endswith(" · took 3 min")
    # Each step's own time on its verdict line, with or without a verdict.
    panel = "\n".join(text.get_text() for axes in page.axes for text in axes.texts)
    assert "Fiducial tracking: took 30 s" in panel
    assert "Fine alignment: WARN · took 2 min 30 s" in panel
    assert heading(alignment_pages.aligned_tilts_page(first)).endswith(" · time not recorded")
    assert heading(ctf.ctf_page(first)).endswith(" · took 1 min 35 s")
    assert heading(tomogram_pages.tomogram_page(first, depths(first))).endswith(
        " · took 12 s")
    assert heading(tomogram_pages.membranes_page(first, depths(first))).endswith(
        " · took 7 min 40 s")
    assert heading(picking.scores_page(first, [])).endswith(" · took 31 min 10 s")
    assert heading(tomogram_pages.tomogram_page(second, depths(second))).endswith(
        " · time not recorded")
    # A page of several steps lists each one's time.
    assert first.ctf is not None
    numbers = ctf._numbers(first, first.ctf)
    assert "Time: fit stack 30 s, fit 54 s,\n  correction 11 s" in numbers

    # The series' sum in the table, and every series' on the title page.
    assert [table_cell(row.run_time) for row in rows] == ["≥ 43 min 37 s", "–", "–"]
    assert front.total_run_time(rows) == ("≥ 44 min", "run time of all steps;\n"
                                                      "2 steps not recorded")
    title = front.title_page(rows, project="p", variant="bin8", written="now", software="s")
    assert "≥ 44 min" in [text.get_text() for text in title.texts]


def test_the_report_command_reads_the_lane_timings(tmp_path: Path, monkeypatch) -> None:
    project = _timed_project(tmp_path)
    drawn: list[SeriesReport] = []

    def capture(path: Path, reports: list[SeriesReport], **_: object) -> object:
        drawn.extend(reports)
        return type("Outcome", (), {"path": path, "pages": 0, "problems": []})()

    monkeypatch.setattr("cryoet_pipeline.reports.pdf.write_report", capture)
    result = CliRunner().invoke(cli.app, ["report", "--project", str(project), "--variant",
                                          "bin8"])

    assert result.exit_code == 0, result.output
    assert _step_seconds(drawn[0].timing.page("tomogram")) == {"reconstruct-tomogram": 12.0}


def test_a_project_without_timings_says_every_time_is_missing(tmp_path: Path) -> None:
    """A project run before the runner timed its steps, or without the runner.

    It still renders: test_the_report_has_a_page_for_everything_each_series_has
    draws this same project, which has no timing file.
    """

    project = _project(tmp_path)

    reports = _timed_reports(project)

    timing = reports[0].timing
    assert timing.total.steps and timing.total.seconds is None
    assert took(timing.page("alignment")) == "time not recorded"
    assert front.total_run_time([summary_row(report) for report in reports]) == (
        "–", "run time of all steps:\nnot recorded")
    assert not any(problem.startswith("run times") for problem in reports[0].problems)


def test_a_table_of_many_steps_still_fits_the_page() -> None:
    """ryr1-bin8's ten verdict columns pushed the table past both edges (2026-10-19)."""

    steps = [step.qc_type for step in QC_STEPS] + ["ctf_correction"]
    rows = [
        SummaryRow(f"TS_{number:03d}", 26, 29, (-36.0, 42.0), 4.7, 6.9, 0.25, 5.0, 789, 291,
                   12.5, dict.fromkeys(steps, "warning"))
        for number in range(5)
    ]

    figure = front.summary_table_pages(rows)[0]()

    axes = next(axes for axes in figure.axes if axes.tables)
    assert axes.tables[0].get_window_extent().width <= axes.get_window_extent().width


def test_a_long_picking_warning_points_to_the_warnings_page() -> None:
    """Instead of running over the caption and the page's foot, as on ryr1-bin8."""

    report = _scored_report("TS_1", 30)
    report.qc = {"particle_picking": QcRecord(
        "particle_picking", "warning", ("picks spread 23.6 px through Z " * 12,), {}, "x")}
    figure = Figure()

    picking._numbers_panel(figure.add_subplot(), report)

    lines = figure.axes[0].texts[0].get_text().splitlines()
    assert lines[-2:] == ["QC: WARN", "… more on the warnings page"]
    assert len(lines) <= picking._PANEL_LINES
