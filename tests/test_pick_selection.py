"""select-picks: keep the picks at or above a score, per series or one for all."""

import json
from pathlib import Path

import pytest
from test_wedge import _manifest
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, ArtifactKind
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.pick_selection import (
    ScoreThreshold,
    measure_score_selection,
    select_picks_by_score,
    thresholds_from_json,
)

STD = 0.02
"""The search's noise scale, so 7σ is LCCmax 0.14 and 10σ is 0.20."""

SCORES = {"TS_A": [0.20, 0.15, 0.14, 0.10], "TS_B": [0.30, 0.13, 0.12]}


def _project(tmp_path: Path, *, with_stats: bool = True) -> tuple[Path, list[Path]]:
    """A registry with one tomogram and one pick list per series, as picking leaves them."""

    registry = ArtifactRegistry.empty()
    manifests = []
    for series, scores in SCORES.items():
        tomogram = Artifact(
            id=f"{series}:tomogram:fine:bin8",
            kind=ArtifactKind.TOMOGRAM,
            path=tmp_path / f"{series}.zarr",
            variant="bin8",
            parameters={"tilt_series_id": series, "tomogram_branch": "full"},
        )
        folder = tmp_path / "picks" / series
        folder.mkdir(parents=True)
        star = folder / f"{series}_full_pytom-match-pick_bin8_particles.star"
        rows = "".join(
            f"{10 + i} {20 + i} {30 + i} {score:.4f}\n" for i, score in enumerate(scores)
        )
        star.write_text(
            "data_particles\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
            "_rlnCoordinateZ #3\n_rlnLCCmax #4\n" + rows
        )
        job = folder / f"{series}_full_pytom-match-pick_bin8_job.json"
        stats = {"job_stats": {"std": STD, "search_space": 1e9}} if with_stats else {}
        job.write_text(json.dumps(stats))
        picks = Artifact(
            id=f"{series}:picks:full:pytom-match-pick:bin8",
            kind=ArtifactKind.PICKS,
            path=star,
            parent_ids=[tomogram.id],
            shape=(len(scores),),
            pixel_spacing_angstrom=9.784,
            variant="bin8",
            parameters={
                "backend": "pytom-match-pick",
                "tilt_series_id": series,
                "tomogram_branch": "full",
                "input_tomogram_id": tomogram.id,
                "pytom_job_json_path": str(job),
                "qc_path": str(folder / "picking_qc.json"),
            },
        )
        registry.add(tomogram)
        registry.add(picks)
        manifest = tmp_path / f"{series}.json"
        one = _manifest().model_copy(update={"tilt_series_id": series})
        manifest.write_text(one.model_dump_json())
        manifests.append(manifest)
    path = tmp_path / "artifacts.json"
    registry.write(path)
    return path, manifests


def _kept_scores(star: Path) -> list[float]:
    columns, rows = read_star_loop(star)
    return [float(row[columns.index("rlnLCCmax")]) for row in rows]


def _invoke(tmp_path: Path, registry: Path, manifests: list[Path], *arguments: str):
    return CliRunner().invoke(
        cli.app,
        [
            "select-picks",
            *[part for manifest in manifests for part in ("--manifest", str(manifest))],
            "--registry",
            str(registry),
            "--out",
            str(tmp_path / "out"),
            "--variant",
            "bin8",
            "--output-variant",
            "bin8-sel",
            *arguments,
        ],
    )


def test_a_threshold_in_sigma_is_converted_with_the_searchs_own_noise(tmp_path):
    registry_path, manifests = _project(tmp_path)
    picks = ArtifactRegistry.load(registry_path).get("TS_A:picks:full:pytom-match-pick:bin8")
    threshold = ScoreThreshold(10.0, "sigma", "--minimum-sigma")
    measured = measure_score_selection(picks, threshold)
    assert measured.minimum_score == pytest.approx(10.0 * STD)
    assert measured.search_sigma == STD and measured.keep_rows == [0]

    manifest = _manifest().model_copy(update={"tilt_series_id": "TS_A"})
    result = select_picks_by_score(
        picks, manifest, output_root=tmp_path / "out", label="bin8-sel", threshold=threshold
    )
    selected, qc = result.artifacts
    assert _kept_scores(selected.path) == [0.20]
    assert selected.variant == "bin8-sel" and selected.parent_ids == [picks.id]
    # What downstream reads off the pick list survives the copy.
    for key in ("input_tomogram_id", "pytom_job_json_path", "backend", "tomogram_branch"):
        assert selected.parameters[key] == picks.parameters[key]
    assert selected.parameters["score_selection"]["kept"] == 1
    assert selected.parameters["picking_qc_path"] == picks.parameters["qc_path"]
    assert qc.parameters["qc_type"] == "picks_score_selection"
    summary = json.loads(qc.path.read_text())
    assert summary["minimum_sigma"] == pytest.approx(10.0)
    assert (summary["picks"], summary["kept"]) == (4, 1)
    assert summary["kept_score_range"] == [0.20, 0.20]


def test_a_pick_exactly_at_a_raw_threshold_is_kept(tmp_path):
    registry_path, _ = _project(tmp_path)
    picks = ArtifactRegistry.load(registry_path).get("TS_A:picks:full:pytom-match-pick:bin8")
    measured = measure_score_selection(picks, ScoreThreshold(0.14, "score", "--minimum-score"))
    assert measured.keep_rows == [0, 1, 2]


def test_a_threshold_that_looks_like_the_other_unit_is_refused():
    with pytest.raises(ValueError, match="in σ"):
        ScoreThreshold(7.5, "score", "--minimum-score")
    with pytest.raises(ValueError, match="raw pytom score"):
        ScoreThreshold(0.12, "sigma", "--minimum-sigma")
    with pytest.raises(ValueError, match="not negative"):
        ScoreThreshold(-1.0, "score", "--minimum-score")


def test_sigma_needs_the_searchs_noise_scale(tmp_path):
    registry_path, _ = _project(tmp_path, with_stats=False)
    picks = ArtifactRegistry.load(registry_path).get("TS_A:picks:full:pytom-match-pick:bin8")
    with pytest.raises(ValueError, match="no noise scale"):
        measure_score_selection(picks, ScoreThreshold(7.0, "sigma", "--minimum-sigma"))
    # A raw score needs no conversion.
    assert measure_score_selection(picks, ScoreThreshold(0.15, "score", "x")).keep_rows == [0, 1]


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ({"TS_A": 7.0}, "no threshold for TS_B"),
        ({"TS_A": 7.0, "TS_B": None}, "placeholder"),
        ({"TS_A": 7.0, "TS_B": "8"}, "must be a number"),
        ({"TS_A": True, "TS_B": 8.0}, "must be a number"),
        ([7.0, 8.0], "JSON object"),
    ],
)
def test_a_thresholds_file_names_what_is_wrong_with_it(tmp_path, document, match):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        thresholds_from_json(path, "sigma", ["TS_A", "TS_B"])


def test_a_thresholds_file_may_hold_more_series_than_are_selected(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"TS_A": 7.0, "TS_B": 8.0, "TS_C": None}))
    thresholds = thresholds_from_json(path, "sigma", ["TS_A"])
    assert list(thresholds) == ["TS_A"] and thresholds["TS_A"].value == 7.0


def test_one_json_gives_each_series_its_own_threshold(tmp_path):
    registry_path, manifests = _project(tmp_path)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"TS_A": 7.0, "TS_B": 10.0}))
    result = _invoke(tmp_path, registry_path, manifests, "--thresholds-sigma", str(thresholds))
    assert result.exit_code == 0, result.output
    assert "TS_A: keeps 3 of 4 at 7σ (LCCmax 0.1400)" in result.output
    assert "TS_B: keeps 1 of 3 at 10σ (LCCmax 0.2000)" in result.output
    registry = ArtifactRegistry.load(registry_path)
    a = registry.get("TS_A:picks:full:pytom-match-pick:bin8-sel")
    b = registry.get("TS_B:picks:full:pytom-match-pick:bin8-sel")
    assert _kept_scores(a.path) == [0.20, 0.15, 0.14] and _kept_scores(b.path) == [0.30]
    # Selected by their label, they lead extraction back to their own tomogram.
    manifest = _manifest().model_copy(update={"tilt_series_id": "TS_A"})
    assert cli._picks_artifact(registry, manifest, variant="bin8-sel") == a
    tomogram = cli._tomogram_from_picks(registry, a, manifest, variant="bin8-sel")
    assert tomogram.id == "TS_A:tomogram:fine:bin8"


def test_one_threshold_serves_every_series(tmp_path):
    registry_path, manifests = _project(tmp_path)
    result = _invoke(tmp_path, registry_path, manifests, "--minimum-score", "0.13")
    assert result.exit_code == 0, result.output
    assert "TS_A: keeps 3 of 4 at LCCmax 0.13 (6.50σ)" in result.output
    registry = ArtifactRegistry.load(registry_path)
    b = registry.get("TS_B:picks:full:pytom-match-pick:bin8-sel")
    assert _kept_scores(b.path) == [0.30, 0.13]


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        (("--minimum-sigma", "7", "--minimum-score", "0.1"), "at most one"),
        (("--minimum-score", "7.5"), "in σ"),
        (("--minimum-sigma", "0.12"), "raw pytom score"),
        (("--minimum-sigma", "50"), "no pick reaches 50σ"),
    ],
)
def test_a_selection_that_cannot_be_made_changes_nothing(tmp_path, arguments, match):
    registry_path, manifests = _project(tmp_path)
    before = registry_path.read_text()
    result = _invoke(tmp_path, registry_path, manifests, *arguments)
    assert result.exit_code != 0
    assert match in " ".join(result.output.split()), result.output
    assert registry_path.read_text() == before
    assert not (tmp_path / "out").exists()


def test_one_series_keeping_nothing_leaves_the_others_unselected_too(tmp_path):
    registry_path, manifests = _project(tmp_path)
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"TS_A": 7.0, "TS_B": 20.0}))
    result = _invoke(tmp_path, registry_path, manifests, "--thresholds-sigma", str(thresholds))
    assert result.exit_code != 0 and "TS_B: no pick reaches 20σ" in result.output
    assert not list((tmp_path / "out").rglob("*_particles.star"))


def test_a_dry_run_counts_and_registers_nothing(tmp_path):
    registry_path, manifests = _project(tmp_path)
    before = registry_path.read_text()
    result = _invoke(tmp_path, registry_path, manifests, "--minimum-sigma", "7", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "TS_A: keeps 3 of 4" in result.output and "nothing registered" in result.output
    assert registry_path.read_text() == before
    assert not list((tmp_path / "out").rglob("*_particles.star"))
    assert len(list((tmp_path / "out").rglob("*_score_selection_qc.json"))) == 2


def test_a_selection_is_not_written_twice_without_overwrite(tmp_path):
    registry_path, manifests = _project(tmp_path)
    assert _invoke(tmp_path, registry_path, manifests, "--minimum-sigma", "7").exit_code == 0
    again = _invoke(tmp_path, registry_path, manifests, "--minimum-sigma", "10")
    assert again.exit_code != 0 and "already exist" in again.output
    replaced = _invoke(tmp_path, registry_path, manifests, "--minimum-sigma", "10", "--overwrite")
    assert replaced.exit_code == 0, replaced.output
    registry = ArtifactRegistry.load(registry_path)
    assert _kept_scores(registry.get("TS_A:picks:full:pytom-match-pick:bin8-sel").path) == [0.20]


def _template(tmp_path: Path) -> Path:
    return tmp_path / "out" / "picks" / "bin8_thresholds_sigma.json"


def test_a_run_with_no_threshold_writes_the_template_and_selects_nothing(tmp_path):
    registry_path, manifests = _project(tmp_path)
    before = registry_path.read_text()
    result = _invoke(tmp_path, registry_path, manifests)
    assert result.exit_code == 1, result.output
    template = _template(tmp_path)
    assert f"thresholds file: {template}" in result.output
    assert "TS_A: to fill · 4 picks, best 10.0σ, lowest 5.0σ" in result.output
    document = json.loads(template.read_text())
    assert [key for key in document if not key.startswith("_")] == ["TS_A", "TS_B"]
    assert document["TS_A"] is None and document["TS_B"] is None
    assert document["_hints"]["TS_B"]["best_sigma"] == 15.0
    assert document["_hints"]["TS_B"]["one_false_positive_sigma"] > 5.0
    assert "--minimum-sigma" in document["_how"]
    assert registry_path.read_text() == before
    assert not list((tmp_path / "out").rglob("*_particles.star"))


def test_a_filled_template_is_used_without_being_named(tmp_path):
    registry_path, manifests = _project(tmp_path)
    assert _invoke(tmp_path, registry_path, manifests).exit_code == 1
    template = _template(tmp_path)
    document = json.loads(template.read_text())
    document.update({"TS_A": 7.0, "TS_B": 10.0})
    template.write_text(json.dumps(document))
    result = _invoke(tmp_path, registry_path, manifests)
    assert result.exit_code == 0, result.output
    assert "TS_A: keeps 3 of 4 at 7σ" in result.output
    assert "TS_B: keeps 1 of 3 at 10σ" in result.output
    selected = ArtifactRegistry.load(registry_path).get("TS_A:picks:full:pytom-match-pick:bin8-sel")
    assert selected.parameters["score_selection"]["source"] == str(template)


def test_a_half_filled_template_names_what_is_left_and_keeps_what_was_filled(tmp_path):
    registry_path, manifests = _project(tmp_path)
    # The template made for TS_A alone, filled, then asked for both series.
    assert _invoke(tmp_path, registry_path, manifests[:1]).exit_code == 1
    template = _template(tmp_path)
    document = json.loads(template.read_text())
    assert list(document["_hints"]) == ["TS_A"]
    document["TS_A"] = 7.0
    template.write_text(json.dumps(document))
    result = _invoke(tmp_path, registry_path, manifests)
    assert result.exit_code == 1, result.output
    assert "TS_B: to fill" in result.output and "TS_A: to fill" not in result.output
    document = json.loads(template.read_text())
    assert document["TS_A"] == 7.0 and document["TS_B"] is None
    assert sorted(document["_hints"]) == ["TS_A", "TS_B"]


def test_one_threshold_for_all_needs_no_template(tmp_path):
    registry_path, manifests = _project(tmp_path)
    assert _invoke(tmp_path, registry_path, manifests, "--minimum-sigma", "7").exit_code == 0
    assert not _template(tmp_path).exists()


def test_a_template_value_in_the_wrong_unit_is_refused(tmp_path):
    registry_path, manifests = _project(tmp_path)
    _invoke(tmp_path, registry_path, manifests)
    template = _template(tmp_path)
    document = json.loads(template.read_text())
    document.update({"TS_A": 0.14, "TS_B": 10.0})
    template.write_text(json.dumps(document))
    result = _invoke(tmp_path, registry_path, manifests)
    assert result.exit_code != 0
    assert "raw pytom score" in " ".join(result.output.split())


def test_each_series_can_keep_what_lies_above_its_own_false_positive_height(tmp_path):
    """The line the report draws, from each search's own statistics.

    Noise of σ 0.02 over 1e9 positions and orientations reaches ~6.0σ once, so
    TS_A keeps its three picks at or above LCCmax ~0.12.
    """

    from cryoet_pipeline.pick_selection import false_positive_height_sigma

    registry_path, manifests = _project(tmp_path)
    picks = ArtifactRegistry.load(registry_path).get("TS_A:picks:full:pytom-match-pick:bin8")
    height = false_positive_height_sigma(picks)
    assert 5.9 < height < 6.1

    result = _invoke(tmp_path, registry_path, manifests, "--above-false-positive-height")
    assert result.exit_code == 0, result.output
    assert "TS_A: keeps 3 of 4" in result.output
    registry = ArtifactRegistry.load(registry_path)
    a = registry.get("TS_A:picks:full:pytom-match-pick:bin8-sel")
    assert _kept_scores(a.path) == [0.20, 0.15, 0.14]


def test_the_false_positive_height_needs_the_searchs_statistics(tmp_path):
    registry_path, manifests = _project(tmp_path, with_stats=False)
    result = _invoke(tmp_path, registry_path, manifests, "--above-false-positive-height")
    assert result.exit_code != 0
    assert "one-false-positive height is unknown" in " ".join(result.output.split())
