"""How long the steps behind a report took, as the lane runner timed them.

scripts/run_series.py --execute times every step it runs, around the step's own
command and inside the Slurm job, so waiting in the queue is never part of it.
It appends one line per run to <project>/step_timings.jsonl: series, step,
seconds, exit code, when it finished, and since 2026-10-28 the lane and its
variant. The registry records no times, so that file is the only record.

Which run made what a report shows is decided by when its files were written:
the run whose span holds a file's modification time wrote it. Labels cannot
decide it alone: one project holds several lanes, the rows written before
2026-10-28 do not say which lane ran them, and a step run outside the runner, by
hand or by a recipe, leaves no row at all. Its result then falls in no run's
span, and its time is "not recorded" rather than some other run's.

The modification times only pick the run; every duration is the run's own.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryoet_pipeline.models import Artifact, ArtifactKind

TIMINGS_FILE = "step_timings.jsonl"
"""Where scripts/run_series.py writes its timings, in the project directory."""

SLACK_SECONDS = 60.0
"""How far outside a run's span a file it wrote may be dated.

The span is the finish time less the seconds, and both were approximate until
2026-10-28: the runner timed its scan of the project's size before the step as
well, and took the finish time after scanning again. A file server's clock can
also run a little apart from the node's.
"""

ROW_RESOLUTION_SECONDS = 1.0
"""The runner writes a finish time to the second, cut down, so a run ended up to
a second after its row says.

A step writes its results as it ends, so they can be dated after the row's
finish; and the next step's span, worked out from its own cut-down finish, can
start up to a second early. Missing this left ryr1's 096 coarse and 097 fine
alignment without a time (2026-10-28), both written in their step's last second.
"""


@dataclass(frozen=True)
class StepRun:
    """One successful run of one step of one series, as the runner timed it."""

    series: str
    step: str
    """The runner's name for the step, such as `fine-align-tilt-series`."""
    seconds: float
    finished_at: float
    """POSIX time."""
    variant: str | None
    """The lane's label; None in rows written before the runner recorded it."""

    @property
    def started_at(self) -> float:
        return self.finished_at - self.seconds

    def distance(self, moment: float) -> float:
        """Seconds from `moment` to this run's span, which ends within the second its
        row names; 0 inside it."""

        ended = self.finished_at + ROW_RESOLUTION_SECONDS
        return max(0.0, self.started_at - moment, moment - ended)


def load_step_runs(project: Path) -> list[StepRun]:
    """The successful runs the project's timing file records; none without the file.

    A failed run made nothing a report shows, so only exit code 0 counts. A line
    a killed job cut short, or one without what a run needs, is skipped.
    """

    try:
        lines = (project / TIMINGS_FILE).read_text().splitlines()
    except OSError:
        return []
    runs = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        run = _step_run(row)
        if run is not None:
            runs.append(run)
    return runs


def _step_run(row: Any) -> StepRun | None:
    if not isinstance(row, dict) or row.get("exit_code") != 0:
        return None
    series, step, seconds, finished = (
        row.get(key) for key in ("series", "step", "seconds", "finished_at")
    )
    if not (isinstance(series, str) and isinstance(step, str) and isinstance(finished, str)):
        return None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    try:
        moment = datetime.fromisoformat(finished)
    except ValueError:
        return None
    if moment.tzinfo is None:
        # The runner has always written UTC.
        moment = moment.replace(tzinfo=UTC)
    variant = row.get("variant")
    return StepRun(
        series=series,
        step=step,
        seconds=float(seconds),
        finished_at=moment.timestamp(),
        variant=variant if isinstance(variant, str) else None,
    )


@dataclass(frozen=True)
class Step:
    name: str
    """As scripts/run_series.py names it, and so the timing file."""
    label: str
    """What it did, in the list of a page's steps."""
    page: str
    """The page its result is drawn on; empty for motion correction, which has none."""


STEPS: tuple[Step, ...] = (
    Step("prepare-tilt-series", "motion correction", ""),
    Step("import-tilt-stack", "stack import", ""),
    Step("align-tilt-series", "coarse alignment", "alignment"),
    Step("qc-coarse-alignment", "coarse check", "alignment"),
    Step("generate-fiducial-seed", "seeding", "alignment"),
    Step("track-fiducials", "tracking", "alignment"),
    Step("fine-align-tilt-series", "fine alignment", "alignment"),
    Step("build-ctf-stack", "fit stack", "ctf"),
    Step("estimate-ctf", "fit", "ctf"),
    Step("build-final-aligned-stack", "aligned stack", "aligned tilts"),
    Step("correct-ctf", "correction", "ctf"),
    Step("reconstruct-tomogram", "reconstruction", "tomogram"),
    Step("segment-tomogram", "segmentation", "membranes"),
    Step("prepare-template", "template", "picks"),
    Step("pick-particles", "search", "picks"),
    # Never in the runner's chain, so never timed; named so that a report on a
    # filtered list says its filter's time is missing rather than leaving it out.
    Step("filter-picks", "membrane filter", "picks"),
    Step("select-picks", "score cut", "picks"),
)
"""Every step that makes something a report draws, in the runner's order."""

_ENTRY_STEPS = ("prepare-tilt-series", "import-tilt-stack")


@dataclass(frozen=True)
class StepTime:
    step: Step
    seconds: float | None
    """None: no timed run made this step's result."""


@dataclass(frozen=True)
class Timing:
    """The times of some steps: one page's, one series', or a whole report's."""

    steps: tuple[StepTime, ...] = ()

    @property
    def seconds(self) -> float | None:
        """The recorded times summed; None when none was recorded."""

        recorded = [one.seconds for one in self.steps if one.seconds is not None]
        return math.fsum(recorded) if recorded else None

    @property
    def missing(self) -> int:
        """How many of the steps have no recorded time."""

        return sum(one.seconds is None for one in self.steps)

    def of(self, *names: str) -> Timing:
        """Only the steps named."""

        return Timing(tuple(one for one in self.steps if one.step.name in names))


@dataclass(frozen=True)
class SeriesTiming:
    """Each step that made a series' results, with its time, in pipeline order."""

    steps: tuple[StepTime, ...] = ()

    def page(self, page: str) -> Timing:
        return Timing(tuple(one for one in self.steps if one.step.page == page))

    @property
    def total(self) -> Timing:
        return Timing(self.steps)


def series_timing(
    series: str,
    artifacts: Sequence[Artifact],
    qc: Mapping[str, object],
    runs: Sequence[StepRun],
) -> tuple[SeriesTiming, str | None]:
    """The steps that made `artifacts` and how long each took, and a problem to report.

    `artifacts` are everything the series' pages are drawn from, their ancestors
    and their QC documents; `qc` the QC records by type. The problem is said when
    the series has timed runs and none of them wrote any of these files, which is
    what a project copied without its files' modification times looks like.
    """

    own = [run for run in runs if run.series == series]
    made = written_by(artifacts, own)
    names = set(expected_steps(artifacts, qc)) | set(made)
    if any(name in made for name in _ENTRY_STEPS):
        # The lineage guessed which entry step it was; a timed run knows.
        names -= {name for name in _ENTRY_STEPS if name not in made}
    timing = SeriesTiming(
        tuple(
            StepTime(step, made[step.name].seconds if step.name in made else None)
            for step in STEPS
            if step.name in names
        )
    )
    problem = None
    if own and names and not made:
        runs_text = "1 run" if len(own) == 1 else f"{len(own)} runs"
        problem = (
            f"run times: the timing file has {runs_text} of this series, but none wrote the "
            "files shown here, so no time is shown; were they copied without their "
            "modification times?"
        )
    return timing, problem


def written_by(artifacts: Sequence[Artifact], runs: Sequence[StepRun]) -> dict[str, StepRun]:
    """For each step, the run that wrote some of these artifacts; the latest if several did.

    A file is the work of the run nearest its modification time, within
    SLACK_SECONDS of its span, and among runs of the same series only (the
    caller's choice of `runs`). Where the spans of two consecutive runs both
    hold it, it is the earlier run's: a step writes its results as it ends,
    and the next one needs seconds of start-up before it can write anything.
    A labelled file can only be the work of a run under the same label, when
    the run's row says which. A file no run is near was written outside the
    runner, and its step has no time.
    """

    known = {step.name for step in STEPS}
    candidates = [run for run in runs if run.step in known]
    chosen: dict[str, StepRun] = {}
    for artifact in artifacts:
        try:
            written = artifact.path.stat().st_mtime
        except OSError:
            continue
        near = [
            run
            for run in candidates
            if _same_lane(run, artifact) and run.distance(written) <= SLACK_SECONDS
        ]
        if not near:
            continue
        run = min(near, key=lambda one: (one.distance(written), one.finished_at))
        earlier = chosen.get(run.step)
        if earlier is None or run.finished_at > earlier.finished_at:
            chosen[run.step] = run
    return chosen


def _same_lane(run: StepRun, artifact: Artifact) -> bool:
    # Unlabelled results are shared by every lane, and old rows name none.
    return run.variant is None or artifact.variant is None or run.variant == artifact.variant


def expected_steps(artifacts: Sequence[Artifact], qc: Mapping[str, object]) -> list[str]:
    """The steps the lineage says ran, in pipeline order, whether or not they were timed.

    Read off what each step leaves behind: a coarse alignment is made from the
    tilt stack, a fine one from a fiducial model, a CTF-corrected stack has the
    CTF model among its parents, and a filtered pick list names the list it was
    filtered from. This is what lets a step run outside the runner say that its
    time was not recorded, instead of dropping out of the sum unseen.
    """

    by_id = {artifact.id: artifact for artifact in artifacts}
    kinds = {artifact.kind for artifact in artifacts}

    def made_from(kind: ArtifactKind, parent_kind: ArtifactKind) -> bool:
        return any(
            artifact.kind is kind
            and any(
                parent in by_id and by_id[parent].kind is parent_kind
                for parent in artifact.parent_ids
            )
            for artifact in artifacts
        )

    names: set[str] = set()
    if ArtifactKind.TILT_STACK in kinds:
        names.add(
            "prepare-tilt-series"
            if ArtifactKind.CORRECTED_PROJECTION in kinds
            else "import-tilt-stack"
        )
    if made_from(ArtifactKind.ALIGNMENT, ArtifactKind.TILT_STACK):
        names.add("align-tilt-series")
    if "coarse_alignment_report" in qc:
        names.add("qc-coarse-alignment")
    if ArtifactKind.FIDUCIAL_SEED_MODEL in kinds:
        names.add("generate-fiducial-seed")
    if ArtifactKind.FIDUCIAL_MODEL in kinds:
        names.add("track-fiducials")
    if made_from(ArtifactKind.ALIGNMENT, ArtifactKind.FIDUCIAL_MODEL):
        names.add("fine-align-tilt-series")
    if made_from(ArtifactKind.CTF_MODEL, ArtifactKind.ALIGNED_TILT_STACK):
        names.add("build-ctf-stack")
    if ArtifactKind.CTF_MODEL in kinds:
        names.add("estimate-ctf")
    if ArtifactKind.TOMOGRAM in kinds and ArtifactKind.ALIGNMENT in kinds:
        names.add("build-final-aligned-stack")
    if made_from(ArtifactKind.ALIGNED_TILT_STACK, ArtifactKind.CTF_MODEL):
        names.add("correct-ctf")
    if ArtifactKind.TOMOGRAM in kinds:
        names.add("reconstruct-tomogram")
    if ArtifactKind.SEGMENTATION in kinds:
        names.add("segment-tomogram")
    if ArtifactKind.TEMPLATE in kinds:
        names.add("prepare-template")
    for artifact in artifacts:
        if artifact.kind is not ArtifactKind.PICKS:
            continue
        # A derived list copies its source's parameters, so the key that names
        # its own parent is the one that says which step made it.
        if artifact.parameters.get("selected_from") in artifact.parent_ids:
            names.add("select-picks")
        elif artifact.parameters.get("filtered_from") in artifact.parent_ids:
            names.add("filter-picks")
        else:
            names.add("pick-particles")
    return [step.name for step in STEPS if step.name in names]


def duration(seconds: float) -> str:
    """A time in the words a report uses: `45 s`, `4 min 12 s`, `1 h 5 min`.

    Seconds are dropped from an hour or more, where they say nothing.
    """

    if seconds < 0.5:
        return "under 1 s"
    whole = round(seconds)
    if whole < 60:
        return f"{whole} s"
    if whole < 3600:
        minutes, rest = divmod(whole, 60)
        return f"{minutes} min" + (f" {rest} s" if rest else "")
    hours, minutes = divmod(round(seconds / 60.0), 60)
    return f"{hours} h" + (f" {minutes} min" if minutes else "")


def short_duration(seconds: float) -> str:
    """The same, short enough for a title page's big number: `50 s`, `45 min`, `12.7 h`."""

    if seconds < 59.5:
        return duration(seconds)
    if seconds < 3570.0:
        return f"{round(seconds / 60.0)} min"
    hours = seconds / 3600.0
    return f"{hours:.1f} h" if hours < 99.95 else f"{hours:.0f} h"


def took(timing: Timing) -> str | None:
    """How long some steps took, as a page's heading says it: `took 4 min 12 s`.

    The sum of a page's steps; `at least` when some step's time was not
    recorded, and only that it was not when none was. None when no step is
    known to have made what the page shows.
    """

    if not timing.steps:
        return None
    seconds = timing.seconds
    if seconds is None:
        return "time not recorded"
    return f"took {'at least ' if timing.missing else ''}{duration(seconds)}"


def panel_lines(timing: Timing, width: int) -> list[str]:
    """Each step's time, for the numbers panel of a page that shows several steps.

    `Time: fit stack 30 s, fit 54 s, correction 11 s`, broken between steps
    only, so that no time is split across two lines. Nothing for a single step,
    whose time the heading already says.
    """

    items = [
        f"{one.step.label} "
        + (duration(one.seconds) if one.seconds is not None else "not recorded")
        for one in timing.steps
    ]
    if len(items) < 2:
        return []
    lines = ["Time:"]
    for index, item in enumerate(items):
        piece = item + ("," if index < len(items) - 1 else "")
        if len(lines[-1]) + 1 + len(piece) > width:
            lines.append("  " + piece)
        else:
            lines[-1] += " " + piece
    return lines


def table_cell(timing: Timing) -> str:
    """A sum in a table: `1 h 5 min`, `≥ 58 min` when a step is missing, `–` with none."""

    seconds = timing.seconds
    if seconds is None:
        return "–"
    return ("≥ " if timing.missing else "") + duration(seconds)
