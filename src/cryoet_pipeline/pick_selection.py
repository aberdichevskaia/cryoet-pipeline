"""Keep the picks that score at least a threshold, as a pick list of their own.

The threshold is read off the score curves the report draws: in σ of each
tomogram's own search, as the report plots them, or as pytom's raw LCCmax, as
`scripts/qc_dynamo_picks.py` plots them and as its `thresholds.json` holds them.
One value for every tomogram, or one per tomogram from a JSON object
`{"TS_0306": 7.5, ...}`.

The per-tomogram file has a fixed place, `<project>/picks/<label>_thresholds_sigma.json`
(`default_thresholds_path`). select-picks run with no threshold writes it there with
every series null and hints on where its scores lie, and refuses to select until
each series has a number, unless one threshold for all is given instead.

The kept picks are registered under a label of their own, as `filter-picks`
registers its membrane selection: a copy of the original's record with its STAR
file narrowed. Everything downstream that reads picks — the tomogram they were
found in, the template behind the centre convention, the search's noise scale —
reads it from there unchanged, and extraction, the Dynamo export, averaging and
the report select them with `--variant <label>`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cryoet_pipeline.backends.picking import _search_statistics
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    QcStatus,
    RetentionPolicy,
    StorageRole,
    TiltSeriesManifest,
)
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.pick_geometry import write_star_subset

ScoreUnit = Literal["sigma", "score"]
SCORE_UNITS: tuple[ScoreUnit, ...] = ("sigma", "score")


@dataclass(frozen=True)
class ScoreThreshold:
    """A minimum score, its unit, and where it was given."""

    value: float
    unit: ScoreUnit
    source: str
    """The option it came from, or the JSON file's path."""

    def __post_init__(self) -> None:
        if self.unit not in SCORE_UNITS:
            raise ValueError(f"unit must be one of {', '.join(SCORE_UNITS)}, got {self.unit!r}")
        if not math.isfinite(self.value) or self.value < 0.0:
            raise ValueError(f"a threshold must be finite and not negative, got {self.value}")
        # The two units differ by a factor of ~50 on this data, and each is an
        # implausible value in the other: a raw LCCmax cannot exceed 1, and a σ
        # threshold under 1 keeps every peak a search returns. Refusing those is
        # what keeps a file of one unit from being read silently as the other.
        if self.unit == "score" and self.value > 1.0:
            raise ValueError(
                f"{self.source}: a raw pytom score of {self.value:g} is impossible (LCCmax "
                "is at most 1); a value like this is in σ, so give it as σ"
            )
        if self.unit == "sigma" and self.value < 1.0:
            raise ValueError(
                f"{self.source}: a threshold of {self.value:g}σ would keep every peak; a "
                "value like this is a raw pytom score, so give it as a score"
            )

    def describe(self) -> str:
        return f"{self.value:g}σ" if self.unit == "sigma" else f"LCCmax {self.value:g}"


def thresholds_from_json(
    path: Path, unit: ScoreUnit, series: Sequence[str]
) -> dict[str, ScoreThreshold]:
    """One threshold per tomogram from a JSON object mapping series to numbers.

    The layout `scripts/qc_dynamo_picks.py` writes, which holds raw scores, so read
    one of its files with the score unit. Every series asked for must be there with
    a number: null is that script's placeholder, and a missing series is a typo
    more often than a choice. Series in the file that are not asked for are left
    alone, so one file can serve a run over some of them.
    """

    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read thresholds from {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path} must hold a JSON object mapping each series to a number")
    missing = [name for name in series if name not in document]
    if missing:
        raise ValueError(f"{path} has no threshold for {', '.join(missing)}")
    thresholds: dict[str, ScoreThreshold] = {}
    for name in series:
        value = document[name]
        if value is None:
            raise ValueError(f"{path}: {name} is null, a placeholder; give it a number")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: the threshold for {name} must be a number, got {value!r}")
        thresholds[name] = ScoreThreshold(float(value), unit, str(path))
    return thresholds


TEMPLATE_HOW = (
    "Minimum score per tomogram, in σ of its own search, as the report's 'Scores from "
    "best to worst' plots it. Replace each null with a number and run select-picks "
    "again, or pass --minimum-sigma for one threshold for every tomogram. _hints shows "
    "where each tomogram's picks lie: a threshold above best_sigma keeps none."
)


def default_thresholds_path(output_root: Path, variant: str | None) -> Path:
    """Where select-picks keeps the per-tomogram thresholds for one pick list's label.

    One file per label, because a threshold belongs to the scores of one search:
    the same series picked again under another label gets a file of its own.
    """

    name = f"{variant}_thresholds_sigma.json" if variant else "thresholds_sigma.json"
    return output_root / "picks" / name


def score_hints(picks: Artifact) -> dict[str, float | int | None]:
    """Where a pick list's scores lie in σ, for choosing a threshold without the report open."""

    columns, rows = read_star_loop(picks.path)
    hints: dict[str, float | int | None] = {"picks": len(rows)}
    statistics = None
    job_json = picks.parameters.get("pytom_job_json_path")
    if isinstance(job_json, str):
        statistics = _search_statistics(Path(job_json))
    if statistics is None or "rlnLCCmax" not in columns or not rows:
        return hints
    position = columns.index("rlnLCCmax")
    sigma = [float(row[position]) / statistics.sigma for row in rows]
    hints.update(
        {
            "best_sigma": round(max(sigma), 2),
            "lowest_sigma": round(min(sigma), 2),
            "one_false_positive_sigma": round(statistics.one_false_positive_z, 2),
        }
    )
    return hints


def false_positive_height_sigma(picks: Artifact) -> float:
    """The one-false-positive height of the search a pick list came from, in σ.

    The line the report draws through the score curves: the score the search's
    noise alone reaches about once over the positions and orientations it tried.
    Computed from pytom's own job statistics, as the report computes it.
    """

    job_json = picks.parameters.get("pytom_job_json_path")
    statistics = _search_statistics(Path(job_json)) if isinstance(job_json, str) else None
    if statistics is None:
        raise ValueError(
            f"{picks.id} records no pytom search statistics, so its one-false-positive "
            "height is unknown; give a threshold instead"
        )
    return statistics.one_false_positive_z


@dataclass(frozen=True)
class RecommendedMultiple:
    """A count-based selection: the best `wanted`, a multiple of those above the height."""

    threshold: ScoreThreshold
    """The score of the last pick kept, as a raw LCCmax."""
    above_height: int
    """Picks above the one-false-positive height: pytom's recommendation."""
    wanted: int
    height_sigma: float


def recommended_multiple_threshold(picks: Artifact, multiple: float) -> RecommendedMultiple:
    """The score that keeps `multiple` times as many picks as lie above the height.

    The count above the one-false-positive height is what the report shows in
    blue and what pytom recommends at its default of one false positive. The best
    `multiple` times that many are kept, below the height too, so that the
    doubtful ones are there for classification as well as the sure ones. A pick
    list holds them only if its extraction went that far down: one cut off sooner
    is refused, since its best picks past the cut-off were never extracted.
    """

    if not math.isfinite(multiple) or multiple < 1.0:
        raise ValueError(f"the multiple must be at least 1, got {multiple:g}")
    height = false_positive_height_sigma(picks)
    job_json = picks.parameters.get("pytom_job_json_path")
    statistics = _search_statistics(Path(str(job_json)))
    assert statistics is not None  # false_positive_height_sigma has read it
    columns, rows = read_star_loop(picks.path)
    if "rlnLCCmax" not in columns:
        raise ValueError(f"{picks.path} has no rlnLCCmax column to select on")
    position = columns.index("rlnLCCmax")
    scores = sorted((float(row[position]) for row in rows), reverse=True)
    # Strictly above, as the report counts it.
    above = sum(1 for score in scores if score / statistics.sigma > height)
    if above == 0:
        raise ValueError(
            f"{picks.id}: no pick lies above its one-false-positive height ({height:.2f}σ), "
            "so there is nothing to take a multiple of"
        )
    wanted = math.ceil(multiple * above - 1e-9)
    if wanted > len(scores):
        raise ValueError(
            f"{picks.id}: {multiple:g} x the {above} picks above {height:.2f}σ is {wanted}, "
            f"but the list holds {len(scores)}: its extraction stopped at its own cut-off. "
            "Pick again with a larger --pytom-number-of-particles and no floor, or with "
            "none, which takes twice pytom's recommendation"
        )
    threshold = ScoreThreshold(
        scores[wanted - 1], "score", f"--recommended-multiple {multiple:g}"
    )
    return RecommendedMultiple(threshold, above, wanted, height)


def write_thresholds_template(
    path: Path, hints: dict[str, dict[str, float | int | None]]
) -> list[str]:
    """Make or extend the thresholds file: every series in `hints` present, unfilled ones null.

    A value already there is kept, whatever it is, so running again never undoes an
    edit; it is checked when it is used. The hints are refreshed for the series
    given and kept for the others. Returns the given series still without a number.
    """

    document: dict[str, object] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON ({exc}); fix or remove it") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must hold a JSON object mapping each series to a number")
        document = loaded
    values = {key: value for key, value in document.items() if not key.startswith("_")}
    for series in hints:
        values.setdefault(series, None)
    old_hints = document.get("_hints")
    merged_hints = {**(old_hints if isinstance(old_hints, dict) else {}), **hints}
    written: dict[str, object] = {series: values[series] for series in sorted(values)}
    written["_how"] = TEMPLATE_HOW
    written["_hints"] = {series: merged_hints[series] for series in sorted(merged_hints)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(written, indent=2, ensure_ascii=False) + "\n")
    return [
        series
        for series in hints
        if isinstance(values[series], bool) or not isinstance(values[series], (int, float))
    ]


@dataclass(frozen=True)
class ScoreMeasurement:
    """What a threshold keeps of one pick list, before anything is written."""

    scores: list[float]
    keep_rows: list[int]
    """STAR data-row positions, counted as `read_star_loop` and `write_star_subset` count."""
    minimum_score: float
    """The threshold as a raw LCCmax, whatever unit it was given in."""
    search_sigma: float | None


def measure_score_selection(picks: Artifact, threshold: ScoreThreshold) -> ScoreMeasurement:
    """Which picks reach the threshold.

    A threshold in σ is converted with the noise scale pytom recorded for this very
    search (`job_stats.std`), the scale the report's plots are drawn in, so the
    same σ means the same thing in every tomogram while raw scores do not. A pick
    exactly at the threshold is kept, as `extract-particles --minimum-score` keeps it.
    """

    if picks.kind is not ArtifactKind.PICKS:
        raise ValueError(f"expected picks, got {picks.kind}")
    columns, rows = read_star_loop(picks.path)
    if "rlnLCCmax" not in columns:
        raise ValueError(f"{picks.path} has no rlnLCCmax column to select on")
    position = columns.index("rlnLCCmax")
    scores = [float(row[position]) for row in rows]

    search_sigma: float | None = None
    job_json = picks.parameters.get("pytom_job_json_path")
    if isinstance(job_json, str):
        statistics = _search_statistics(Path(job_json))
        search_sigma = statistics.sigma if statistics is not None else None
    if threshold.unit == "sigma":
        if search_sigma is None:
            raise ValueError(
                f"{picks.id} records no noise scale for its search (pytom's job_stats), so a "
                "threshold in σ cannot be converted; give it as a raw score"
            )
        minimum_score = threshold.value * search_sigma
    else:
        minimum_score = threshold.value
    keep_rows = [row for row, score in enumerate(scores) if score >= minimum_score]
    return ScoreMeasurement(scores, keep_rows, minimum_score, search_sigma)


def selection_paths(
    picks: Artifact, series: str, output_root: Path, label: str
) -> tuple[Path, Path]:
    """The kept picks' STAR file and the selection's QC report, named as filter-picks names its."""

    branch = str(picks.parameters.get("tomogram_branch", "full"))
    backend = str(picks.parameters.get("backend", "picks"))
    prefix = f"{series}_{branch}_{backend}_{label}"
    return (
        output_root / "picks" / series / f"{prefix}_particles.star",
        output_root / "qc" / series / "picking" / f"{prefix}_score_selection_qc.json",
    )


@dataclass(frozen=True)
class ScoreSelection:
    artifacts: list[Artifact]
    """The kept picks and their QC; empty for a dry run."""
    summary: dict[str, object]
    report_path: Path
    star_path: Path | None


def select_picks_by_score(
    picks: Artifact,
    manifest: TiltSeriesManifest,
    *,
    output_root: Path,
    label: str,
    threshold: ScoreThreshold,
    overwrite: bool = False,
    dry_run: bool = False,
    measurement: ScoreMeasurement | None = None,
) -> ScoreSelection:
    """Keep the picks reaching `threshold`, write the QC and, unless a dry run, the picks.

    `measurement` is what `measure_score_selection` returned for the same picks and
    threshold, for a caller that checked every series before writing any.
    """

    if picks.variant == label:
        raise ValueError(
            f"the selected picks need a label of their own; {picks.id} already carries {label!r}"
        )
    measured = measurement or measure_score_selection(picks, threshold)
    scores, keep_rows = measured.scores, measured.keep_rows
    sigma = measured.search_sigma
    series = manifest.tilt_series_id
    star_path, report_path = selection_paths(picks, series, output_root, label)
    outputs = [report_path] if dry_run else [star_path, report_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"score selection outputs already exist: {joined}")
    if not dry_run and not keep_rows:
        raise ValueError(f"{series}: no pick reaches {threshold.describe()}; nothing to register")

    warnings: list[str] = []
    if scores and len(keep_rows) == len(scores):
        lowest = min(scores)
        warnings.append(
            f"the threshold keeps every pick: the lowest scores {lowest:.4f}"
            + (f" ({lowest / sigma:.2f}σ)" if sigma else "")
            + ", so the pick list's own floor set the count, not this threshold"
        )
    status = QcStatus.WARNING if warnings else QcStatus.PASS
    kept_scores = [scores[row] for row in keep_rows]
    summary: dict[str, object] = {
        "schema_version": 1,
        "picks_id": picks.id,
        "tilt_series_id": series,
        "label": label,
        "threshold": {"value": threshold.value, "unit": threshold.unit, "source": threshold.source},
        "minimum_score": measured.minimum_score,
        "minimum_sigma": measured.minimum_score / sigma if sigma else None,
        "search_sigma": sigma,
        "picks": len(scores),
        "kept": len(keep_rows),
        "kept_score_range": [min(kept_scores), max(kept_scores)] if kept_scores else None,
        "dry_run": dry_run,
        "status": status.value,
        "warnings": warnings,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if dry_run:
        return ScoreSelection([], summary, report_path, None)

    written = write_star_subset(picks.path, star_path, keep_rows)
    branch = str(picks.parameters.get("tomogram_branch", "full"))
    backend = str(picks.parameters.get("backend", "picks"))
    selected_id = f"{series}:picks:{branch}:{backend}:{label}"
    selected = Artifact.model_validate(
        {
            **picks.model_dump(mode="json"),
            "id": selected_id,
            "variant": label,
            "path": str(star_path),
            "parent_ids": [picks.id],
            "shape": [written],
            "size_bytes": star_path.stat().st_size,
            "parameters": {
                **picks.parameters,
                "selected_from": picks.id,
                "score_selection": {
                    "threshold": threshold.value,
                    "unit": threshold.unit,
                    "source": threshold.source,
                    "minimum_score": measured.minimum_score,
                    "picks": len(scores),
                    "kept": written,
                },
                "picking_qc_path": picks.parameters.get("qc_path"),
                "qc_path": str(report_path),
            },
        }
    )
    qc = Artifact(
        id=f"{series}:qc:picks_score_selection:{backend}:{label}",
        variant=label,
        kind=ArtifactKind.QC,
        path=report_path,
        parent_ids=[selected_id],
        parameters={
            "qc_type": "picks_score_selection",
            "tilt_series_id": series,
            "tomogram_branch": branch,
            "status": status.value,
        },
        storage_role=StorageRole.QC,
        retention_policy=RetentionPolicy.KEEP,
        can_recompute=True,
        size_bytes=report_path.stat().st_size,
    )
    return ScoreSelection([selected, qc], summary, report_path, star_path)
