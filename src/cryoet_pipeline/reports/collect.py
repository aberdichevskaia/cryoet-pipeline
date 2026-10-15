"""What a report shows, resolved and loaded before anything is drawn.

The rules worth testing live here rather than beside the drawing: which QC
documents belong to a result, which CTF row is which view, which dose goes
with which view, whether a correlation map agrees with the picks read off it.
None of them needs a plotting library.

Loading is per part and forgiving: a series whose CTF fit cannot be read still
gets its tomogram and picks drawn, and the reason the CTF is missing is kept
and printed. A project-wide report that stops at the first odd series would be
useless exactly when it is needed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.imod_defocus import parse_ctffind_summary, read_imod_defocus_file
from cryoet_pipeline.models import Artifact, ArtifactKind, TiltAlignment, TiltSeriesManifest
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.score_preview import (
    ScorePreview,
    read_score_preview,
    score_preview_from_mrc,
)


@dataclass(frozen=True)
class QcStep:
    qc_type: str
    label: str
    """The step's name in prose."""
    short: str
    """The step's name as a table header."""


QC_STEPS: tuple[QcStep, ...] = (
    QcStep("coarse_alignment_report", "Coarse alignment", "Coarse"),
    QcStep("fiducial_seed", "Fiducial seeding", "Seed"),
    QcStep("fiducial_tracking", "Fiducial tracking", "Track"),
    QcStep("fine_alignment", "Fine alignment", "Fine"),
    QcStep("defocus_hint", "Defocus hint", "Hint"),
    QcStep("ctf_estimation", "CTF estimation", "CTF"),
    QcStep("fine_reconstruction", "Reconstruction", "Recon"),
    QcStep("tomogram_restoration", "Denoising", "Denoise"),
    QcStep("tomogram_segmentation", "Membrane segmentation", "Membr"),
    QcStep("template_preparation", "Template", "Templ"),
    QcStep("particle_picking", "Picking", "Pick"),
    QcStep("picks_membrane_filter", "Pick filter", "Filter"),
)
"""The steps a report has a column for, in pipeline order.

A QC type not listed here still shows, after these and under its own name, so
a step added later appears without anyone remembering to add it.
"""

_NOT_VERDICTS = frozenset({"coarse_alignment_preview"})
"""QC artifacts that are pictures rather than verdicts."""

_STATUS_ORDER = {"pass": 0, "unknown": 1, "warning": 2, "fail": 3}


def qc_step(qc_type: str) -> QcStep:
    for step in QC_STEPS:
        if step.qc_type == qc_type:
            return step
    return QcStep(qc_type, qc_type.replace("_", " "), qc_type[:7])


@dataclass(frozen=True)
class SeriesArtifacts:
    """The artifacts one tilt-series' pages are drawn from, resolved by the caller.

    Resolution is the command's job, through the same lookups every other
    command uses, so that a report cannot pick a different tomogram than
    extraction would. `problems` says what could not be resolved, and why.
    """

    manifest: TiltSeriesManifest
    picks: Artifact | None = None
    tomogram: Artifact | None = None
    template: Artifact | None = None
    template_mask: Artifact | None = None
    segmentation: Artifact | None = None
    ctf_model: Artifact | None = None
    problems: tuple[str, ...] = ()

    @property
    def series(self) -> str:
        return self.manifest.tilt_series_id

    def roots(self) -> list[Artifact]:
        candidates = (
            self.picks,
            self.tomogram,
            self.template,
            self.template_mask,
            self.segmentation,
            self.ctf_model,
        )
        return [artifact for artifact in candidates if artifact is not None]


@dataclass(frozen=True)
class QcRecord:
    qc_type: str
    status: str
    """pass, warning, fail, or unknown when the document did not say."""
    warnings: tuple[str, ...]
    document: dict[str, Any]
    artifact_id: str


def lineage_qc(registry: ArtifactRegistry, roots: Sequence[Artifact]) -> dict[str, QcRecord]:
    """The QC documents of every step that led to these artifacts, by QC type.

    By lineage rather than by series and label: one project holds several
    lanes, and one series can have two reconstructions, two CTF fits, a picks
    list and a filtered one. The documents that describe this result are the
    ones attached to its ancestors, and only those.

    Two documents of one type in one lineage do not happen today; if they ever
    do, the worse status wins and both sets of warnings are kept, so neither
    is hidden.
    """

    by_id = {artifact.id: artifact for artifact in registry.artifacts}
    lineage: set[str] = set()
    pending = [root.id for root in roots]
    while pending:
        current = pending.pop()
        if current in lineage:
            continue
        lineage.add(current)
        artifact = by_id.get(current)
        if artifact is not None:
            pending.extend(artifact.parent_ids)

    records: dict[str, QcRecord] = {}
    for artifact in registry.by_kind(ArtifactKind.QC):
        qc_type = artifact.parameters.get("qc_type")
        if not isinstance(qc_type, str) or qc_type in _NOT_VERDICTS:
            continue
        if lineage.isdisjoint(artifact.parent_ids):
            continue
        record = _qc_record(artifact, qc_type)
        earlier = records.get(qc_type)
        records[qc_type] = record if earlier is None else _merge_records(earlier, record)
    return records


def _qc_record(artifact: Artifact, qc_type: str) -> QcRecord:
    document: dict[str, Any] = {}
    if artifact.path.suffix == ".json" and artifact.path.is_file():
        loaded = _read_json(artifact.path)
        if isinstance(loaded, dict):
            document = loaded
    # The document's own status first: the registry entry's copy is written
    # once at registration and is PASS for picking whatever the document says.
    raw_status = document.get("status", artifact.parameters.get("status"))
    status = str(raw_status).lower() if raw_status is not None else "unknown"
    if status not in _STATUS_ORDER:
        status = "unknown"
    warnings = tuple(str(one) for one in document.get("warnings", []) or [])
    return QcRecord(qc_type, status, warnings, document, artifact.id)


def _merge_records(first: QcRecord, second: QcRecord) -> QcRecord:
    status = max(first.status, second.status, key=_STATUS_ORDER.__getitem__)
    warnings = tuple(dict.fromkeys(first.warnings + second.warnings))
    return QcRecord(first.qc_type, status, warnings, second.document, second.artifact_id)


@dataclass(frozen=True)
class PickTable:
    """Pick coordinates and scores, with the noise scale of the search they came from."""

    x_px: np.ndarray
    y_px: np.ndarray
    z_px: np.ndarray
    score: np.ndarray
    search_sigma: float | None = None
    search_space: float | None = None
    floor_sigma: float | None = None
    """The height a peak-free search over this space reaches once, in sigma."""
    cut_off_sigma: float | None = None
    excess_at_cut_off: float | None = None
    z_sd_px: float | None = None
    uniform_z_sd_px: float | None = None

    @property
    def count(self) -> int:
        return int(self.score.size)

    @property
    def has_scores(self) -> bool:
        return self.count > 0 and bool(np.isfinite(self.score).all())

    @property
    def sigma(self) -> np.ndarray | None:
        """Scores in units of the search's noise, when pytom recorded it."""

        if self.search_sigma is None or not self.has_scores:
            return None
        return self.score / self.search_sigma

    def ranked(self) -> np.ndarray:
        """Indices from the highest score down; ties keep the file's order."""

        return np.argsort(-self.score, kind="stable")

    @property
    def above_floor(self) -> int | None:
        sigma = self.sigma
        if sigma is None or self.floor_sigma is None:
            return None
        return int(np.count_nonzero(sigma > self.floor_sigma))

    def expected_from_noise(self, z: np.ndarray) -> np.ndarray | None:
        """How many peaks noise alone puts above each height, over the whole search.

        The Gaussian tail times the number of positions and orientations
        searched: the null the picking QC derives its one-false-positive height
        from, so the two cannot disagree about what noise does.
        """

        if self.search_space is None:
            return None
        tail = np.asarray([math.erfc(float(value) / math.sqrt(2.0)) for value in z])
        return self.search_space * 0.5 * tail


def load_picks(picks: Artifact) -> PickTable:
    columns, rows = _star_table(picks.path)

    def column(name: str) -> np.ndarray:
        if name not in columns:
            return np.full(len(rows), np.nan)
        index = columns.index(name)
        return np.asarray([float(row[index]) for row in rows], dtype=np.float64)

    # A filtered list points `qc_path` at the filter's report; the search's own
    # numbers stay in the picking QC it was filtered from.
    qc_path = picks.parameters.get("picking_qc_path") or picks.parameters.get("qc_path")
    document: dict[str, Any] = {}
    if isinstance(qc_path, str) and Path(qc_path).is_file():
        loaded = _read_json(Path(qc_path))
        if isinstance(loaded, dict):
            document = loaded
    return PickTable(
        x_px=column("rlnCoordinateX"),
        y_px=column("rlnCoordinateY"),
        z_px=column("rlnCoordinateZ"),
        score=column("rlnLCCmax"),
        search_sigma=_positive(document.get("search_sigma")),
        search_space=_positive(document.get("search_space")),
        floor_sigma=_finite(document.get("one_false_positive_sigma")),
        cut_off_sigma=_finite(document.get("score_cut_off_sigma")),
        excess_at_cut_off=_finite(document.get("excess_over_noise_at_cut_off")),
        z_sd_px=_finite(document.get("coordinate_z_sd")),
        uniform_z_sd_px=_positive(document.get("uniform_z_sd")),
    )


def _star_table(path: Path) -> tuple[list[str], list[list[str]]]:
    try:
        return read_star_loop(path)
    except ValueError:
        # A header and no rows: the search found nothing above its cut-off,
        # which is a result rather than an unreadable file.
        return [], []


@dataclass(frozen=True)
class CtfTable:
    """One row per view the CTF was fitted on, in the fitted stack's order."""

    backend: str
    z_value: np.ndarray
    tilt_angle_deg: np.ndarray
    defocus1_um: np.ndarray
    defocus2_um: np.ndarray
    fit_resolution_angstrom: np.ndarray | None
    """CTFFIND's limit of a successful fit, per view; None from a backend without one."""
    dose: np.ndarray | None
    """Dose received before each view (SerialEM's PriorRecordDose), in e/A^2."""
    diagnostic_path: Path | None
    nyquist_angstrom: float | None = None
    hint_defocus_um: float | None = None
    voltage_kv: float | None = None
    spherical_aberration_mm: float | None = None
    amplitude_contrast: float | None = None
    pixel_spacing_angstrom: float | None = None

    @property
    def mean_defocus_um(self) -> np.ndarray:
        return np.asarray((self.defocus1_um + self.defocus2_um) / 2.0)

    def nearest_view(self, angle_deg: float) -> int:
        return int(np.argmin(np.abs(self.tilt_angle_deg - angle_deg)))


def load_ctf(
    registry: ArtifactRegistry,
    ctf_model: Artifact,
    manifest: TiltSeriesManifest,
    qc: dict[str, QcRecord],
) -> CtfTable:
    """Per-view CTF values, each paired with the view it was fitted on.

    The fit is paired with views exactly as the pipeline paired it when it
    wrote the `.defocus` file: row i of the fit is transform i of the alignment
    the fitted stack was built with, and each transform names its view's z
    value. The z value then finds the view's dose in the manifest. Going by z
    value rather than by position matters because a dose-symmetric series is
    written in one order and acquired in another, and because excluded views
    are in the manifest but not in the stack.
    """

    alignment = _parent_of_kind(registry, ctf_model, ArtifactKind.ALIGNMENT)
    if alignment is None:
        raise ValueError(
            f"{ctf_model.id} names no alignment among its parents, so its rows cannot be "
            "paired with views"
        )
    transforms = TiltAlignment.model_validate_json(alignment.path.read_text()).transforms
    summary = ctf_model.parameters.get("summary_path")
    fit: np.ndarray | None
    if isinstance(summary, str) and Path(summary).is_file():
        rows = sorted(parse_ctffind_summary(Path(summary)), key=lambda row: row.view)
        if [row.view for row in rows] != list(range(1, len(transforms) + 1)):
            raise ValueError(
                f"the CTFFIND summary has views {[row.view for row in rows]}, but the "
                f"alignment it was fitted with has {len(transforms)} views"
            )
        defocus1 = np.asarray([row.defocus1_angstrom / 1e4 for row in rows])
        defocus2 = np.asarray([row.defocus2_angstrom / 1e4 for row in rows])
        fit = np.asarray(
            [
                row.fit_resolution_angstrom
                if row.fit_resolution_angstrom is not None
                else np.nan
                for row in rows
            ]
        )
        if not np.isfinite(fit).any():
            fit = None
    else:
        defocus1, defocus2 = _defocus_file_values(ctf_model.path, len(transforms))
        fit = None

    z_values = np.asarray([transform.z_value for transform in transforms], dtype=np.int64)
    doses = _dose_by_z(manifest)
    dose = (
        np.asarray([doses[int(z)] for z in z_values])
        if doses is not None and all(int(z) in doses for z in z_values)
        else None
    )
    diagnostic = ctf_model.parameters.get("diagnostic_path")
    document = qc["ctf_estimation"].document if "ctf_estimation" in qc else {}
    return CtfTable(
        backend=str(ctf_model.parameters.get("backend", ctf_model.backend or "unknown")),
        z_value=z_values,
        tilt_angle_deg=np.asarray([transform.tilt_angle_deg for transform in transforms]),
        defocus1_um=defocus1,
        defocus2_um=defocus2,
        fit_resolution_angstrom=fit,
        dose=dose,
        diagnostic_path=(
            Path(diagnostic)
            if isinstance(diagnostic, str) and Path(diagnostic).is_file()
            else None
        ),
        nyquist_angstrom=_positive(document.get("nyquist_angstrom")),
        hint_defocus_um=_positive(
            ctf_model.parameters.get("hint_defocus_um", document.get("hint_defocus_um"))
        ),
        voltage_kv=_positive(ctf_model.parameters.get("voltage_kv")),
        spherical_aberration_mm=_positive(ctf_model.parameters.get("spherical_aberration_mm")),
        amplitude_contrast=_finite(ctf_model.parameters.get("amplitude_contrast")),
        pixel_spacing_angstrom=_positive(ctf_model.parameters.get("pixel_spacing_angstrom")),
    )


def _defocus_file_values(path: Path, views: int) -> tuple[np.ndarray, np.ndarray]:
    """Both defocus axes per view from an IMOD `.defocus` file, in micrometres."""

    first = np.full(views, np.nan)
    second = np.full(views, np.nan)
    for entry in read_imod_defocus_file(path):
        for view in range(entry.view_start, entry.view_end + 1):
            if 1 <= view <= views:
                first[view - 1] = entry.defocus_nm / 1000.0
                other = entry.second_defocus_nm
                second[view - 1] = (other if other is not None else entry.defocus_nm) / 1000.0
    if not np.isfinite(first).all():
        missing = [index + 1 for index in np.flatnonzero(~np.isfinite(first))]
        raise ValueError(f"{path} has no defocus for views {missing}")
    return first, second


def _dose_by_z(manifest: TiltSeriesManifest) -> dict[int, float] | None:
    """PriorRecordDose per z value, or None when the mdoc did not record it for every view."""

    doses: dict[int, float] = {}
    for image in manifest.images:
        raw = image.metadata.get("PriorRecordDose")
        if raw is None:
            return None
        try:
            doses[image.z_value] = float(str(raw))
        except ValueError:
            return None
    return doses


@dataclass(frozen=True)
class TomogramInfo:
    artifact: Artifact
    shape: tuple[int, int, int]
    """(Z, Y, X) voxels."""
    voxel_spacing_angstrom: float
    included_z_values: tuple[int, ...] | None
    excluded_z_values: tuple[int, ...]
    ctf_corrected: bool | None

    @property
    def depth_nm(self) -> float:
        return self.shape[0] * self.voxel_spacing_angstrom / 10.0


def load_tomogram_info(tomogram: Artifact) -> TomogramInfo:
    array = cast(Any, zarr.open(tomogram.path, mode="r"))
    shape = tuple(int(size) for size in array.shape)
    if len(shape) != 3:
        raise ValueError(f"{tomogram.id} is not a volume: shape {shape}")
    if tomogram.pixel_spacing_angstrom is None:
        raise ValueError(f"{tomogram.id} records no voxel size")
    included = tomogram.parameters.get("included_z_values")
    corrected = tomogram.parameters.get("ctf_corrected")
    return TomogramInfo(
        artifact=tomogram,
        shape=(shape[0], shape[1], shape[2]),
        voxel_spacing_angstrom=float(tomogram.pixel_spacing_angstrom),
        included_z_values=(
            tuple(int(z) for z in included) if isinstance(included, list) else None
        ),
        excluded_z_values=tuple(
            int(z) for z in tomogram.parameters.get("excluded_z_values", []) or []
        ),
        ctf_corrected=corrected if isinstance(corrected, bool) else None,
    )


@dataclass(frozen=True)
class TemplateInfo:
    template: np.ndarray
    """(Z, Y, X), as it was matched."""
    mask: np.ndarray | None
    voxel_spacing_angstrom: float | None
    source: str | None
    low_pass_angstrom: float | None
    mask_radius_px: float | None
    inverted: bool | None

    @property
    def box_px(self) -> int:
        return int(max(self.template.shape))


def load_template(artifacts: SeriesArtifacts, qc: dict[str, QcRecord]) -> TemplateInfo | None:
    """The template the picks were matched with: registered, or the file the picks name."""

    picks = artifacts.picks
    template_path = (
        artifacts.template.path
        if artifacts.template is not None
        else _path_parameter(picks, "pytom_template_path")
    )
    if template_path is None or not template_path.is_file():
        return None
    mask_path = (
        artifacts.template_mask.path
        if artifacts.template_mask is not None
        else _path_parameter(picks, "pytom_template_mask_path")
    )
    template, voxel = _read_mrc(template_path)
    mask = _read_mrc(mask_path)[0] if mask_path is not None and mask_path.is_file() else None
    document = qc["template_preparation"].document if "template_preparation" in qc else {}
    source = document.get("emdb_id") or document.get("source_map_path") or template_path.name
    inverted = document.get("inverted")
    return TemplateInfo(
        template=template,
        mask=mask,
        voxel_spacing_angstrom=voxel,
        source=str(source) if source else None,
        low_pass_angstrom=_positive(document.get("low_pass_angstrom")),
        mask_radius_px=_positive(document.get("mask_radius_px")),
        inverted=inverted if isinstance(inverted, bool) else None,
    )


@dataclass(frozen=True)
class CorrelationMap:
    preview: ScorePreview
    source: str
    """Where the picture came from: saved at picking, or made from a kept volume."""
    agreement: float | None
    """The share of picks whose own score is what the map holds at their voxel."""


def load_correlation_map(
    picks: Artifact, table: PickTable, *, slab_thickness_px: int
) -> CorrelationMap | None:
    preview_path = _path_parameter(picks, "pytom_scores_preview_path")
    if preview_path is not None and preview_path.is_file():
        preview = read_score_preview(preview_path)
        source = "saved at picking"
    else:
        volume_path = _path_parameter(picks, "pytom_scores_volume_path")
        if volume_path is None or not volume_path.is_file():
            return None
        preview = score_preview_from_mrc(volume_path, slab_thickness_px=slab_thickness_px)
        source = "made from the kept score volume"
    return CorrelationMap(preview, source, map_agreement(preview, table))


def map_agreement(preview: ScorePreview, table: PickTable) -> float | None:
    """The share of picks whose map value is their own score.

    pytom reads each pick's score off this map, so the slab maximum at a pick's
    voxel is at least that score. A share well below one means the map and the
    coordinates disagree about axes or origin, and circles drawn on it would be
    in the wrong places. The scores come from the STAR file, not from this map,
    which is what makes the comparison a check rather than an identity.
    """

    if not table.has_scores:
        return None
    z = np.clip(np.rint(table.z_px).astype(np.int64), 0, preview.depth - 1)
    y = np.rint(table.y_px).astype(np.int64)
    x = np.rint(table.x_px).astype(np.int64)
    _, height, width = preview.slab_max.shape
    inside = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    if not inside.any():
        return 0.0
    slabs = np.searchsorted(preview.slab_stops, z[inside], side="right")
    values = preview.slab_max[slabs, y[inside], x[inside]]
    scores = table.score[inside]
    # Half precision keeps about three significant digits.
    tolerance = 2e-3 * np.abs(scores) + 1e-4
    agreeing = int(np.count_nonzero(values >= scores - tolerance))
    return agreeing / table.count


@dataclass
class SeriesReport:
    """Everything one series' pages are drawn from, and what could not be loaded."""

    artifacts: SeriesArtifacts
    qc: dict[str, QcRecord] = field(default_factory=dict)
    tomogram: TomogramInfo | None = None
    picks: PickTable | None = None
    ctf: CtfTable | None = None
    template: TemplateInfo | None = None
    correlation: CorrelationMap | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def series(self) -> str:
        return self.artifacts.series

    @property
    def particle_radius_px(self) -> float:
        """How big a pick is drawn: the template mask's radius, else a quarter box."""

        if self.template is not None and self.template.mask_radius_px is not None:
            return self.template.mask_radius_px
        if self.template is not None:
            return self.template.box_px / 4.0
        return 5.0


def load_series(registry: ArtifactRegistry, artifacts: SeriesArtifacts) -> SeriesReport:
    report = SeriesReport(artifacts=artifacts, problems=list(artifacts.problems))
    report.qc = lineage_qc(registry, artifacts.roots())

    def attempt(what: str, load: Any) -> Any:
        try:
            return load()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report.problems.append(f"{what}: {exc}")
            return None

    if artifacts.tomogram is not None:
        tomogram = artifacts.tomogram
        report.tomogram = attempt("tomogram", lambda: load_tomogram_info(tomogram))
    if artifacts.ctf_model is not None:
        ctf_model = artifacts.ctf_model
        report.ctf = attempt(
            "CTF", lambda: load_ctf(registry, ctf_model, artifacts.manifest, report.qc)
        )
    report.template = attempt("template", lambda: load_template(artifacts, report.qc))
    if artifacts.picks is not None:
        picks = artifacts.picks
        report.picks = attempt("picks", lambda: load_picks(picks))
        if report.picks is not None:
            table = report.picks
            thickness = max(1, round(2.0 * report.particle_radius_px))
            report.correlation = attempt(
                "correlation map",
                lambda: load_correlation_map(picks, table, slab_thickness_px=thickness),
            )
    return report


@dataclass(frozen=True)
class SummaryRow:
    series: str
    views_used: int | None
    views_total: int
    tilt_range_deg: tuple[float, float] | None
    mean_defocus_um: float | None
    ctf_fit_near_zero_angstrom: float | None
    residual_nm: float | None
    membrane_percent: float | None
    picks: int | None
    above_floor: int | None
    best_sigma: float | None
    statuses: dict[str, str]


def summary_row(report: SeriesReport) -> SummaryRow:
    manifest = report.artifacts.manifest
    angle_by_z = {image.z_value: image.tilt_angle_deg for image in manifest.images}
    used: list[int] | None = None
    if report.tomogram is not None and report.tomogram.included_z_values is not None:
        used = [z for z in report.tomogram.included_z_values if z in angle_by_z]
    elif report.ctf is not None:
        used = [int(z) for z in report.ctf.z_value if int(z) in angle_by_z]
    angles = [angle_by_z[z] for z in used] if used else []

    ctf = report.ctf
    mean_defocus = float(np.mean(ctf.mean_defocus_um)) if ctf is not None else None
    if mean_defocus is None and "ctf_estimation" in report.qc:
        mean_defocus = _finite(report.qc["ctf_estimation"].document.get("mean_defocus_um"))
    fit_near_zero = None
    if ctf is not None and ctf.fit_resolution_angstrom is not None:
        value = float(ctf.fit_resolution_angstrom[ctf.nearest_view(0.0)])
        fit_near_zero = value if math.isfinite(value) else None

    membranes = None
    if "tomogram_segmentation" in report.qc:
        fraction = _finite(report.qc["tomogram_segmentation"].document.get("foreground_fraction"))
        membranes = fraction * 100.0 if fraction is not None else None

    picks = report.picks
    sigma = picks.sigma if picks is not None else None
    return SummaryRow(
        series=report.series,
        views_used=len(used) if used is not None else None,
        views_total=len(manifest.images),
        tilt_range_deg=(min(angles), max(angles)) if angles else None,
        mean_defocus_um=mean_defocus,
        ctf_fit_near_zero_angstrom=fit_near_zero,
        residual_nm=(
            _finite(report.qc["fine_alignment"].document.get("residual_mean_nm"))
            if "fine_alignment" in report.qc
            else None
        ),
        membrane_percent=membranes,
        picks=picks.count if picks is not None else None,
        above_floor=picks.above_floor if picks is not None else None,
        best_sigma=float(sigma.max()) if sigma is not None and sigma.size else None,
        statuses={qc_type: record.status for qc_type, record in report.qc.items()},
    )


def _parent_of_kind(
    registry: ArtifactRegistry, artifact: Artifact, kind: ArtifactKind
) -> Artifact | None:
    for parent_id in artifact.parent_ids:
        try:
            parent = registry.get(parent_id)
        except KeyError:
            continue
        if parent.kind is kind:
            return parent
    return None


def _path_parameter(artifact: Artifact | None, key: str) -> Path | None:
    if artifact is None:
        return None
    value = artifact.parameters.get(key)
    return Path(value) if isinstance(value, str) and value else None


def _read_mrc(path: Path) -> tuple[np.ndarray, float | None]:
    with mrcfile.open(path, permissive=True) as handle:
        data = np.asarray(handle.data, dtype=np.float32)
        voxel = float(handle.voxel_size.x)
    return data, voxel if voxel > 0 else None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0.0 else None
