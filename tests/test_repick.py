"""scripts/repick.py: a lane's search again under a new label, against the old and the null."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import test_null_decoys as decoys
import typer
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import variant_label, variant_suffix
from cryoet_pipeline.models import Artifact, ArtifactKind, TiltSeriesManifest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repick  # noqa: E402

# The null-decoy tests' series: reconstructed, given a template, picked by the lane,
# with membranes segmented under the lane's label that its search did not use.
project = decoys.project
SERIES, LANE, JOB = decoys.SERIES, decoys.LANE, decoys.JOB
NOMINAL = [-40.0093, -38.0096, -36.0048, -34.0106, 57.9832]
SOLVED = [-49.77, -48.35, -46.92, -45.5, 48.11]


def _options(command: list[str]) -> dict[str, str | bool]:
    found: dict[str, str | bool] = {}
    index = 1
    while index < len(command):
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            found[command[index]] = command[index + 1]
            index += 2
        else:
            found[command[index]] = True
            index += 1
    return found


def _picks(folder: Path, name: str, rows: Sequence[tuple[float, float, float, float]],
           job: dict, *, label: str) -> Artifact:
    decoys._star(folder / f"{name}.star", rows)
    (folder / f"{name}_job.json").write_text(json.dumps(job))
    return Artifact(id=f"{SERIES}:picks:full:pytom-match-pick:{label}", variant=label,
                    kind=ArtifactKind.PICKS, path=folder / f"{name}.star",
                    parameters={"backend": "pytom-match-pick", "tilt_series_id": SERIES,
                                "pytom_job_json_path": str(folder / f"{name}_job.json")})


def test_the_search_is_the_old_one_into_a_new_label(project: Path) -> None:
    registry = ArtifactRegistry.load(project / "artifacts.json")
    old = repick.picks_labelled(registry, SERIES, "bin8")
    assert old is not None
    pick, extract = repick.repick_commands(LANE, project, SERIES, registry=registry, old=old,
                                           label="bin8-solved")
    options = _options(pick)
    # Inputs by the lane's label, as always; only what is written carries the new one.
    assert options["--variant"] == "bin8" and options["--output-variant"] == "bin8-solved"
    assert options["--pytom-template"] == old.parameters["pytom_template_path"]
    assert options["--pytom-number-of-particles"] == "600"  # the lane's: these picks do not say
    # Unmasked as the old search was, though the series has membranes under the lane's label.
    assert options.get("--no-use-segmentation-mask") is True
    assert _options(extract)["--variant"] == "bin8-solved"

    group = typer.main.get_command(cli.app)
    for command in (pick, extract):
        known = {option for param in group.commands[command[0]].params
                 for option in (*param.opts, *param.secondary_opts)}
        flags = {part for part in command[1:] if part.startswith("--")}
        assert flags <= known, sorted(flags - known)


class _PicksAgain:
    """pytom's place: two of the old picks found again and one new, the job recorded."""

    name = "pytom-match-pick"

    def __init__(self) -> None:
        self.seen: list[tuple[str, list[str]]] = []

    def pick(self, tomogram: Artifact, manifest: TiltSeriesManifest, context,
             supporting_artifacts: Sequence[Artifact] = ()) -> Artifact:
        self.seen.append((tomogram.id, [artifact.id for artifact in supporting_artifacts]))
        folder = context.output_dir / "picks" / manifest.tilt_series_id
        picks = _picks(folder, "again", [(13, 15, 10, 0.33), (25, 21, 12, 0.27),
                                         (6, 6, 6, 0.20)],
                       {**JOB, "tomogram": "/scratch/3/t.mrc", "rng_seed": 5,
                        "ts_metadata": {"tilt_angles": SOLVED}},
                       label=str(variant_label(context)))
        qc = folder / "again_qc.json"
        qc.write_text("{}\n")
        return picks.model_copy(update={
            "id": f"{manifest.tilt_series_id}:picks:full:{self.name}{variant_suffix(context)}",
            "parent_ids": [tomogram.id],
            "parameters": {**picks.parameters, "tomogram_branch": "full",
                           "input_tomogram_id": tomogram.id, "qc_path": str(qc)},
        })


def test_the_new_picks_come_from_the_lanes_tomogram_and_leave_the_old_ones(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = decoys._small_box_lane()
    registry = ArtifactRegistry.load(project / "artifacts.json")
    old = repick.picks_labelled(registry, SERIES, "bin8")
    assert old is not None
    pick, extract = repick.repick_commands(lane, project, SERIES, registry=registry, old=old,
                                           label="bin8-solved")
    backend = _PicksAgain()
    monkeypatch.setattr(cli, "_picking_backend", lambda name: backend)
    runner = CliRunner()
    for command in (pick, extract):
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 0, result.output

    (searched, supporting), = backend.seen
    assert searched == old.parameters["input_tomogram_id"] and supporting == []
    registry = ArtifactRegistry.load(project / "artifacts.json")
    assert repick.picks_labelled(registry, SERIES, "bin8") == old
    new = repick.picks_labelled(registry, SERIES, "bin8-solved")
    assert new is not None and new.parameters["input_tomogram_id"] == searched
    stacks = [artifact for artifact in registry.by_kind(ArtifactKind.PARTICLE_STACK)
              if artifact.variant == "bin8-solved"]
    assert len(stacks) == 1 and stacks[0].parameters["picks_id"] == new.id


def test_the_comparison_counts_picks_found_again_and_above_the_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = {**JOB, "random_phase_correction": False}
    old = _picks(tmp_path, "old", [(12, 15, 10, 0.30), (25, 20, 12, 0.25), (30, 30, 15, 0.12),
                                   (20, 28, 9, 0.05)],
                 {**common, "tomogram": "/s/1/t.mrc", "rng_seed": 5,
                  "ts_metadata": {"tilt_angles": NOMINAL}}, label="bin8")
    new = _picks(tmp_path, "new", [(13, 15, 10, 0.33), (25, 21, 12, 0.27), (6, 6, 6, 0.20)],
                 {**common, "tomogram": "/s/2/t.mrc", "rng_seed": 5,
                  "ts_metadata": {"tilt_angles": SOLVED}}, label="bin8-solved")
    null = _picks(tmp_path, "null", [(9, 9, 9, 0.20), (14, 30, 12, 0.11)],
                  {**common, "tomogram": "/s/3/t.mrc", "rng_seed": 77,
                   "ts_metadata": {"tilt_angles": SOLVED}}, label="bin8-null")

    result = repick.compare(old, new, null)
    assert (result.old, result.new, result.refound) == (4, 3, 2)
    assert (result.null_best, result.old_above, result.new_above, result.refound_above) == (
        0.2, 2, 2, 2)
    printed = capsys.readouterr().out
    assert "their scores, median 0.275 then and 0.300 now" in printed
    # Against the old search the tilt angles alone; against the null search nothing, its own
    # seed apart, which nothing reads without random-phase correction.
    assert ("against the old search, it differs in:\n#   ts_metadata.tilt_angles: "
            "5 values -49.77 .. 48.11 (new) vs 5 values -40.0093 .. 57.9832 (old)") in printed
    assert "against the null search: the same search, but for the tomogram" in printed

    corrected = _picks(tmp_path, "rpc", [(9, 9, 9, 0.2)],
                       {**JOB, "random_phase_correction": True, "tomogram": "/s/4/t.mrc",
                        "rng_seed": 78, "ts_metadata": {"tilt_angles": SOLVED}}, label="x")
    differences = repick.search_differences(new, corrected)
    assert differences is not None and any(line.startswith("rng_seed") for line in differences)


@pytest.mark.parametrize(("old_above", "new_above", "verdict"), [
    (100, 120, "the stage angles cost real picks"),
    (100, 105, "the angles did not matter at this binning"),
    (100, 85, "the solved angles search worse than the stage's"),
])
def test_the_summary_reads_the_rule_written_before_the_run(
    old_above: int, new_above: int, verdict: str, capsys: pytest.CaptureFixture[str]
) -> None:
    repick.summarise({"TS_A": repick.Comparison(600, 600, 400, 0.12, old_above, new_above, 90)})
    assert verdict in capsys.readouterr().out


def test_the_dry_run_prints_both_commands_and_picks_nothing(project: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "repick.py"), "--project", str(project),
         "--config", str(ROOT / "configs" / "lanes" / "ribo-mdc.toml"), "--series", SERIES,
         "--label", "bin8-solved"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--output-variant bin8-solved" in result.stdout
    assert "cryoet extract-particles" in result.stdout
    assert "searched the whole volume, as the real picks were" in result.stdout
    registry = ArtifactRegistry.load(project / "artifacts.json")
    assert repick.picks_labelled(registry, SERIES, "bin8-solved") is None
