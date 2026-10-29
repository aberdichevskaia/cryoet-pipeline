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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.imod_defocus import parse_ctffind_summary, read_imod_defocus_file
from cryoet_pipeline.models import Artifact, ArtifactKind, TiltAlignment, TiltSeriesManifest
from cryoet_pipeline.particles import read_star_loop
from cryoet_pipeline.reports.timing import SeriesTiming, StepRun, Timing, series_timing
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
    QcStep("picks_score_selection", "Score selection", "Cut"),
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

    ancestors = lineage(registry, roots)
    ids = {artifact.id for artifact in ancestors}
    ids.update(parent for artifact in ancestors for parent in artifact.parent_ids)
    records: dict[str, QcRecord] = {}
    for artifact in registry.by_kind(ArtifactKind.QC):
        qc_type = artifact.parameters.get("qc_type")
        if not isinstance(qc_type, str) or qc_type in _NOT_VERDICTS:
            continue
        if ids.isdisjoint(artifact.parent_ids):
            continue
        record = _qc_record(artifact, qc_type)
        earlier = records.get(qc_type)
        records[qc_type] = record if earlier is None else _merge_records(earlier, record)
    return records


def lineage(registry: ArtifactRegistry, roots: Sequence[Artifact]) -> list[Artifact]:
    """These artifacts and every ancestor of theirs the registry holds, each once."""

    by_id = {artifact.id: artifact for artifact in registry.artifacts}
    seen: set[str] = set()
    found: list[Artifact] = []
    pending = [root.id for root in roots]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        artifact = by_id.get(current)
        if artifact is not None:
            found.append(artifact)
            pending.extend(artifact.parent_ids)
    return found


REWRITE_SLACK_SECONDS = 60.0
"""How much later than a result the file it was made from may be dated and still be
the version it was made from: files one step writes together can be dated seconds
apart, in either order, and a file server's clock can run a little apart from the
node's."""


def written_again_after_use(
    artifacts: Sequence[Artifact], by_id: Mapping[str, Artifact]
) -> list[str]:
    """Each result dated after something among `artifacts` was made from it, in words.

    A step reads its inputs as they are when it runs, and the registry keeps one
    record per result, which the step's next run overwrites. A result made again
    after use leaves what was made from it on an earlier version, while this
    report reads the later one, and nothing but the files' modification times
    says so. ryr1 had it (2026-10-28): its bin8 lane remade the shared CTF fit on
    18 Oct, and the bin4 report drew that fit beside tomograms whose stacks bin4
    had corrected on 14 Oct with the one before.

    Parents are looked up in `by_id`. Parents of one kind dated after the same
    results are said together, as the tilts of a series motion-corrected again
    would be. A file that cannot be read has no date, and says nothing.
    """

    children = list({artifact.id: artifact for artifact in artifacts}.values())
    involved = {artifact.id: artifact for artifact in children}
    involved.update(
        (parent_id, by_id[parent_id])
        for child in children
        for parent_id in child.parent_ids
        if parent_id in by_id
    )
    dated = {
        artifact_id: moment
        for artifact_id, artifact in involved.items()
        if (moment := _modified(artifact.path)) is not None
    }

    later: dict[str, list[Artifact]] = {}
    for child in children:
        if child.id not in dated:
            continue
        for parent_id in dict.fromkeys(child.parent_ids):
            if parent_id in dated and dated[parent_id] > dated[child.id] + REWRITE_SLACK_SECONDS:
                later.setdefault(parent_id, []).append(child)
    together: dict[tuple[str, tuple[str, ...]], list[Artifact]] = {}
    for parent_id, made in later.items():
        key = (_artifact_label(involved[parent_id]), tuple(child.id for child in made))
        together.setdefault(key, []).append(involved[parent_id])

    lines = []
    for (label, made_ids), parents in sorted(
        together.items(), key=lambda item: min(dated[one] for one in item[0][1])
    ):
        times = sorted(dated[parent.id] for parent in parents)
        if len(parents) == 1:
            subject = f"the {label}, dated {_when(times[0])}, is"
            source, versions = "it", "an earlier version than the one"
        else:
            subject = f"{len(parents)} {_plural(label)}, dated {_span(times[0], times[-1])}, are"
            source, versions = "them", "earlier versions than the ones"
        made = sorted((involved[one] for one in made_ids), key=lambda one: dated[one.id])
        named = _listed([f"the {_artifact_label(one)} ({_when(dated[one.id])})" for one in made])
        those = "that was" if len(made) == 1 else "those were"
        lines.append(
            f"{subject} newer than {named} made from {source}, so {those} made from "
            f"{versions} this report reads."
        )
    return lines


_KIND_LABELS = {
    ArtifactKind.CORRECTED_PROJECTION: "motion-corrected tilt",
    ArtifactKind.CTF_MODEL: "CTF fit",
    ArtifactKind.SEGMENTATION: "membrane segmentation",
    ArtifactKind.TEMPLATE_MATCH: "template search",
}
"""Kinds a report calls otherwise than the registry does."""

_STACK_LABELS = {"fiducial_tracking": "tracking stack", "ctf_corrected": "CTF-corrected stack"}


def _artifact_label(artifact: Artifact) -> str:
    """What a result is called in a sentence: `CTF fit`, `fine alignment`, `aligned stack
    at bin 2`."""

    parameters = artifact.parameters
    if artifact.kind is ArtifactKind.QC:
        qc_type = parameters.get("qc_type")
        return f"{_lower_first(qc_step(qc_type).label)} QC" if isinstance(qc_type, str) else "QC"
    if artifact.kind is ArtifactKind.ALIGNMENT and parameters.get("stage") in ("coarse", "fine"):
        return f"{parameters['stage']} alignment"
    if artifact.kind is ArtifactKind.TOMOGRAM and parameters.get("alignment_stage") == "coarse":
        return "coarse tomogram"
    if artifact.kind is ArtifactKind.ALIGNED_TILT_STACK:
        purpose = parameters.get("purpose")
        if isinstance(purpose, str) and purpose in _STACK_LABELS:
            return _STACK_LABELS[purpose]
        # The CTF fit's stack and the lane's own are both aligned stacks; their
        # binnings tell them apart.
        return "aligned stack" + (f" at bin {artifact.binning}" if artifact.binning else "")
    return _KIND_LABELS.get(artifact.kind, artifact.kind.value.replace("_", " "))


def _lower_first(label: str) -> str:
    # "Fine alignment" in a sentence, but never "cTF estimation".
    return label[:1].lower() + label[1:] if label[1:2].islower() else label


def _plural(label: str) -> str:
    return label if label.endswith("s") else label + "s"


def _listed(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _when(moment: float) -> str:
    """A file's date as the report's own date is written, in this computer's time."""

    return datetime.fromtimestamp(moment).strftime("%Y-%m-%d %H:%M")


def _span(first: float, last: float) -> str:
    start, end = _when(first), _when(last)
    if start == end:
        return start
    return f"{start} to {end[11:] if start[:10] == end[:10] else end}"


def _modified(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


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
    rot_deg: np.ndarray | None = None
    """pytom's orientation for each pick, RELION's ZYZ angles; None when the list has none."""
    tilt_deg: np.ndarray | None = None
    psi_deg: np.ndarray | None = None
    low_pass_angstrom: float | None = None
    """The low-pass the search was run with, as pytom recorded it in its job."""
    template_box_px: int | None = None
    """The template box the search used, which sets where a pick's rotation centre is."""

    @property
    def count(self) -> int:
        return int(self.score.size)

    @property
    def has_orientations(self) -> bool:
        angles = (self.rot_deg, self.tilt_deg, self.psi_deg)
        return self.count > 0 and all(
            one is not None and bool(np.isfinite(one).all()) for one in angles
        )

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
    job = _pytom_job(picks)
    shape = job.get("template_shape")
    box = (
        max(shape)
        if isinstance(shape, list) and shape and all(type(one) is int for one in shape)
        else None
    )
    low_pass = _positive(job.get("low_pass"))
    if low_pass is None:
        low_pass = _positive(picks.parameters.get("pytom_low_pass_angstrom"))
    angles = [column(name) for name in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")]
    oriented = all(name in columns for name in ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"))
    return PickTable(
        rot_deg=angles[0] if oriented else None,
        tilt_deg=angles[1] if oriented else None,
        psi_deg=angles[2] if oriented else None,
        low_pass_angstrom=low_pass,
        template_box_px=box,
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


def _pytom_job(picks: Artifact) -> dict[str, Any]:
    """The job JSON pytom wrote for the search these picks came from, or nothing.

    pytom writes its whole job there, so it records the low-pass and template
    box the search actually ran with, rather than the ones a lane asked for.
    """

    path = _path_parameter(picks, "pytom_job_json_path")
    if path is None or not path.is_file():
        return {}
    loaded = _read_json(path)
    return loaded if isinstance(loaded, dict) else {}


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
    fit_binning: int | None = None
    """The binning of the stack CTFFIND fitted, relative to the tilt stack, as in the lane's
    `[binning] ctf_fit`."""
    interpolated: np.ndarray | None = None
    """Per view, True where CTFFIND found no Thon rings and the model holds a defocus
    interpolated from the fitted views; the defocus arrays then hold that value."""
    left_out: np.ndarray | None = None
    """Per view, True where such a view was also left out of the corrected stack, and so
    of the tomogram."""

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
    # Views CTFFIND found no Thon rings on: the model holds an interpolated defocus
    # for them, and that is the one corrected with, so it is the one to show.
    interpolated = np.isin(z_values, _int_list(ctf_model.parameters.get("interpolated_z_values")))
    left_out = np.isin(z_values, _int_list(ctf_model.parameters.get("excluded_z_values")))
    if interpolated.any() and ctf_model.path.is_file():
        applied1, applied2 = _defocus_file_values(ctf_model.path, len(transforms))
        defocus1 = np.where(interpolated, applied1, defocus1)
        defocus2 = np.where(interpolated, applied2, defocus2)
    doses = _dose_by_z(manifest)
    dose = (
        np.asarray([doses[int(z)] for z in z_values])
        if doses is not None and all(int(z) in doses for z in z_values)
        else None
    )
    diagnostic = ctf_model.parameters.get("diagnostic_path")
    document = qc["ctf_estimation"].document if "ctf_estimation" in qc else {}
    fitted_stack = _parent_of_kind(registry, ctf_model, ArtifactKind.ALIGNED_TILT_STACK)
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
        fit_binning=fitted_stack.binning if fitted_stack is not None else None,
        interpolated=interpolated if interpolated.any() else None,
        left_out=left_out if left_out.any() else None,
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
    mask_radius = _positive(document.get("mask_radius_px"))
    if mask_radius is None and mask is not None:
        mask_radius = _measured_mask_radius(mask)
    return TemplateInfo(
        template=template,
        mask=mask,
        voxel_spacing_angstrom=voxel,
        source=str(source) if source else None,
        low_pass_angstrom=_positive(document.get("low_pass_angstrom")),
        mask_radius_px=mask_radius,
        inverted=inverted if isinstance(inverted, bool) else None,
    )


def _measured_mask_radius(mask: np.ndarray) -> float | None:
    """Half the width of the mask's central row above half its height.

    What pytom's spherical mask has as its radius, measured off the file for a
    template that came without a preparation record.
    """

    center = [size // 2 for size in mask.shape]
    row = mask[center[0], center[1], :]
    peak = float(row.max())
    if peak <= 0.0:
        return None
    inside = int(np.count_nonzero(row >= 0.5 * peak))
    return inside / 2.0 if inside else None


@dataclass(frozen=True)
class AlignmentInfo:
    """The fine alignment of one series, per view, and the stacks to show it on."""

    method: str
    """Gold beads or patch tracking, from the tracking backend."""
    z_value: np.ndarray
    tilt_angle_deg: np.ndarray
    """The solved tilt angles, in the alignment's view order (ascending angle)."""
    residual_nm: np.ndarray | None
    """Each view's mean residual, from tiltalign's own per-view table."""
    points_per_view: np.ndarray | None
    """How many tracked points the final fit used on each view."""
    point_xy: np.ndarray | None
    """(N, 2) tracked positions, in pixels of the tracking stack."""
    point_view: np.ndarray | None
    """(N,) each position's view, as an index into `z_value`."""
    tracking_stack: Path | None
    tracking_pixel_angstrom: float | None
    aligned_stack: Path | None
    """The aligned tilt series the tomogram was reconstructed from."""
    aligned_pixel_angstrom: float | None
    aligned_z_values: tuple[int, ...] | None
    """The z value of each slice of `aligned_stack`, in its order."""
    excluded_z_values: tuple[int, ...]
    fiducial_count: int | None
    tilt_axis_angle_deg: float | None

    def nearest_view(self, angle_deg: float) -> int:
        return int(np.argmin(np.abs(self.tilt_angle_deg - angle_deg)))


def load_alignment(
    registry: ArtifactRegistry, tomogram: Artifact, qc: dict[str, QcRecord]
) -> AlignmentInfo | None:
    """Per-view residuals and tracked points of the alignment a tomogram was built with.

    The views are paired by the order every file of the fine alignment shares:
    the tracking stack, tiltalign's residual file and its per-view table all
    list the aligned views by ascending tilt, as the alignment's transforms do.
    The per-view table also prints each view's solved tilt, and a table whose
    angles do not match the transforms is not used, rather than paired wrongly.

    None for a tomogram with no alignment among its parents, such as one imported
    from elsewhere: there is nothing to show, which is not a problem.
    """

    alignment = _parent_of_kind(registry, tomogram, ArtifactKind.ALIGNMENT)
    if alignment is None:
        return None
    solved = TiltAlignment.model_validate_json(alignment.path.read_text())
    z_values = np.asarray([one.z_value for one in solved.transforms], dtype=np.int64)
    angles = np.asarray([one.tilt_angle_deg for one in solved.transforms])

    model = _parent_of_kind(registry, alignment, ArtifactKind.FIDUCIAL_MODEL)
    if model is not None:
        backend = str(model.parameters.get("backend", model.backend or ""))
        method = "patch tracking" if "patch" in backend else "gold beads"
    else:
        method = str(alignment.parameters.get("backend", alignment.backend or "unknown"))

    log = _path_parameter(alignment, "tiltalign_log_path")
    residual_nm = (
        _per_view_residuals(log.read_text(), angles) if log is not None and log.is_file() else None
    )

    points_per_view = point_xy = point_view = None
    residual_file = _path_parameter(alignment, "residual_file_path")
    if residual_file is not None and residual_file.is_file():
        points = _residual_points(residual_file)
        views = np.rint(points[:, 2]).astype(np.int64)
        if points.size and views.min() >= 0 and views.max() < len(z_values):
            point_xy = points[:, :2]
            point_view = views
            points_per_view = np.bincount(views, minlength=len(z_values))

    tracking = _parent_of_kind(registry, alignment, ArtifactKind.ALIGNED_TILT_STACK)
    aligned = _parent_of_kind(registry, tomogram, ArtifactKind.ALIGNED_TILT_STACK)
    aligned_order = aligned.parameters.get("included_z_values") if aligned is not None else None
    document = qc["fine_alignment"].document if "fine_alignment" in qc else {}
    count = document.get("fiducial_count")
    excluded = alignment.parameters.get("excluded_z_values", solved.excluded_z_values) or []
    return AlignmentInfo(
        method=method,
        z_value=z_values,
        tilt_angle_deg=angles,
        residual_nm=residual_nm,
        points_per_view=points_per_view,
        point_xy=point_xy,
        point_view=point_view,
        tracking_stack=(
            tracking.path if tracking is not None and tracking.path.is_file() else None
        ),
        tracking_pixel_angstrom=tracking.pixel_spacing_angstrom if tracking is not None else None,
        aligned_stack=aligned.path if aligned is not None and aligned.path.is_file() else None,
        aligned_pixel_angstrom=aligned.pixel_spacing_angstrom if aligned is not None else None,
        aligned_z_values=(
            tuple(int(z) for z in aligned_order) if isinstance(aligned_order, list) else None
        ),
        excluded_z_values=tuple(int(z) for z in excluded),
        fiducial_count=count if isinstance(count, int) else None,
        tilt_axis_angle_deg=_finite(
            document.get("minimum_tilt_rotation_deg", solved.tilt_axis_angle_deg)
        ),
    )


def per_view_table(log_text: str) -> list[dict[str, float]]:
    """The last per-view table tiltalign printed: one dict per view, keyed by its header.

    tiltalign prints the table once per fit, so the last one is the final fit.
    Read by its own header rather than by column position, since the columns
    depend on which variables the alignment solved for.
    """

    lines = log_text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.split()[:2] == ["view", "rotation"]]
    if not starts:
        return []
    header = lines[starts[-1]].split()
    rows = []
    for line in lines[starts[-1] + 1 :]:
        fields = line.split()
        if len(fields) != len(header):
            break
        try:
            rows.append(dict(zip(header, (float(field) for field in fields), strict=True)))
        except ValueError:
            break
    return rows


def _per_view_residuals(log_text: str, angles: np.ndarray) -> np.ndarray | None:
    rows = per_view_table(log_text)
    if len(rows) != angles.size or not all("resid-nm" in row and "tilt" in row for row in rows):
        return None
    printed = np.asarray([row["tilt"] for row in rows])
    # Printed to a tenth of a degree.
    if np.max(np.abs(printed - angles)) > 0.3:
        return None
    return np.asarray([row["resid-nm"] for row in rows])


def _residual_points(path: Path) -> np.ndarray:
    """(N, 5) rows of tiltalign's residual file: x, y, view, residual x, residual y.

    The first line declares the count; positions are in pixels of the stack the
    points were tracked on, and views count from zero.
    """

    lines = path.read_text().splitlines()
    rows = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 5:
            rows.append([float(fields[0]), float(fields[1]), float(fields[2]),
                         float(fields[-2]), float(fields[-1])])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5)


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
    alignment: AlignmentInfo | None = None
    timing: SeriesTiming = field(default_factory=SeriesTiming)
    written_again: list[str] = field(default_factory=list)
    """What these pages were drawn from that was written again after use, in words."""
    problems: list[str] = field(default_factory=list)

    @property
    def series(self) -> str:
        return self.artifacts.series

    @property
    def particle_radius_px(self) -> float:
        return self.particle_radius()[0]

    def particle_radius(self) -> tuple[float, str]:
        """How big a pick is drawn, in tomogram voxels, and where that number came from.

        The radius of the mask pytom matched with, converted from the template's
        voxels to the tomogram's. In a lane the two are the same, since the
        template is made at the tomogram's voxel size; a template given as a file
        need not be.
        """

        template = self.template
        if template is None:
            return 5.0, "a stand-in radius (no template was found for these picks)"
        scale = 1.0
        if template.voxel_spacing_angstrom is not None and self.tomogram is not None:
            scale = template.voxel_spacing_angstrom / self.tomogram.voxel_spacing_angstrom
        if template.mask_radius_px is not None:
            return template.mask_radius_px * scale, "the radius of the mask pytom matched with"
        return template.box_px / 4.0 * scale, "a quarter of the template box (no mask found)"


def load_series(
    registry: ArtifactRegistry,
    artifacts: SeriesArtifacts,
    runs: Sequence[StepRun] = (),
) -> SeriesReport:
    """One series' report; `runs` are the project's timed step runs, for the run times."""

    report = SeriesReport(artifacts=artifacts, problems=list(artifacts.problems))
    report.qc = lineage_qc(registry, artifacts.roots())
    by_id = {artifact.id: artifact for artifact in registry.artifacts}
    shown = lineage(registry, artifacts.roots()) + [
        by_id[record.artifact_id] for record in report.qc.values() if record.artifact_id in by_id
    ]
    report.timing, timing_problem = series_timing(report.series, shown, report.qc, runs)
    if timing_problem is not None:
        report.problems.append(timing_problem)
    report.written_again = written_again_after_use(shown, by_id)

    def attempt(what: str, load: Any) -> Any:
        try:
            return load()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report.problems.append(f"{what}: {exc}")
            return None

    if artifacts.tomogram is not None:
        tomogram = artifacts.tomogram
        report.tomogram = attempt("tomogram", lambda: load_tomogram_info(tomogram))
        report.alignment = attempt(
            "alignment", lambda: load_alignment(registry, tomogram, report.qc)
        )
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
    run_time: Timing = field(default_factory=Timing)
    """Every step that made this series' results, from motion correction to picking."""


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
        run_time=report.timing.total,
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


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0.0 else None
