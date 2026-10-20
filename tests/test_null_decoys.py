"""scripts/null_decoys.py: pytom's picks in a phase-randomised copy of a tomogram, as decoys."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import scipy.fft  # type: ignore[import-untyped]
import typer
import zarr
from test_wedge import _manifest
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import variant_label, variant_suffix
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, TiltSeriesManifest
from cryoet_pipeline.null_volumes import phase_randomised

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import null_decoys  # noqa: E402
import run_series  # noqa: E402

SERIES = "TS_A"
SHAPE = (24, 40, 40)
LANE = tomllib.loads((ROOT / "configs" / "lanes" / "ribo-mdc.toml").read_text())
JOB = {"template_shape": [12, 12, 12], "low_pass": 25.0, "whiten_spectrum": True,
       "tilt_angles": [-40.0, 0.0, 50.0]}


def _star(path: Path, rows: Sequence[tuple[float, float, float, float]]) -> None:
    columns = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot",
               "rlnAngleTilt", "rlnAnglePsi", "rlnLCCmax")
    lines = ["data_particles", "", "loop_",
             *(f"_{name} #{position}" for position, name in enumerate(columns, 1))]
    lines += [f"{x} {y} {z} 10.0 20.0 30.0 {score}" for x, y, z, score in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A series reconstructed, given a template and picked, as the ribo-mdc lane leaves it."""

    project = tmp_path / "project"
    (project / "manifests").mkdir(parents=True)
    manifest = _manifest().model_copy(update={"tilt_series_id": SERIES})
    (project / "manifests" / f"{SERIES}.json").write_text(manifest.model_dump_json())

    folder = project / "tomograms" / SERIES
    folder.mkdir(parents=True)
    volume = np.random.default_rng(0).normal(size=SHAPE).astype(np.float32)
    real = zarr.open(folder / "real.zarr", mode="w", shape=SHAPE, chunks=(8, 20, 40),
                     dtype=np.float32)
    real[:] = volume
    real.attrs.update({"axis_order": "zyx", "voxel_spacing_angstrom": 9.784})
    angles = folder / "real.tlt"
    angles.write_text("-40.0\n0.0\n50.0\n")
    tomogram = Artifact(
        id=f"{SERIES}:tomogram:fine:bin8", variant="bin8", kind=ArtifactKind.TOMOGRAM,
        path=folder / "real.zarr", shape=SHAPE, dtype="float32", axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=9.784, binning=8,
        parameters={"tilt_series_id": SERIES, "tomogram_branch": "full",
                    "tilt_angles_path": str(angles), "included_z_values": [0, 1, 2],
                    "imod_rec_path": str(folder / "real.rec"), "imod_rec_axis_order": "yzx",
                    "log_path": str(folder / "tilt.log")},
    )
    templates = project / "templates"
    templates.mkdir()
    # Per series and labelled, as prepare-template registers them: a lookup by any
    # other label finds this series' template and refuses it.
    template = Artifact(id=f"{SERIES}:template:pytom:bin8", variant="bin8",
                        kind=ArtifactKind.TEMPLATE, path=templates / "template.mrc",
                        parent_ids=[tomogram.id], parameters={"tilt_series_id": SERIES})
    mask = Artifact(id=f"{SERIES}:template_mask:pytom:bin8", variant="bin8",
                    kind=ArtifactKind.TEMPLATE_MASK, path=templates / "mask.mrc",
                    parent_ids=[template.id],
                    parameters={"tilt_series_id": SERIES, "template_id": template.id,
                                "mask_is_spherical": True})
    for artifact in (template, mask):
        artifact.path.touch()
    # Membranes segmented under the lane's label, which the real search did not use:
    # pick-particles looks for them by label unless told not to.
    segmentation = Artifact(id=f"{SERIES}:segmentation:membrain-seg:bin8", variant="bin8",
                            kind=ArtifactKind.SEGMENTATION, path=folder / "membranes.zarr",
                            parent_ids=[tomogram.id], parameters={"tilt_series_id": SERIES})

    picks = project / "picks" / SERIES
    _star(picks / "real.star", [(12, 15, 10, 0.30), (25, 20, 12, 0.25), (30, 30, 15, 0.12),
                                (20, 28, 9, 0.05)])
    job = picks / "real_job.json"
    job.write_text(json.dumps({**JOB, "tomogram": "/scratch/1/tomogram.mrc",
                               "template": str(template.path), "job_stats": {"std": 0.01}}))
    real_picks = Artifact(
        id=f"{SERIES}:picks:full:pytom-match-pick:bin8", variant="bin8",
        kind=ArtifactKind.PICKS, path=picks / "real.star",
        parent_ids=[tomogram.id, template.id, mask.id],
        parameters={"backend": "pytom-match-pick", "tilt_series_id": SERIES,
                    "tomogram_branch": "full", "input_tomogram_id": tomogram.id,
                    "segmentation_id": None,
                    "pytom_template_path": str(template.path),
                    "pytom_template_mask_path": str(mask.path),
                    "pytom_low_pass_angstrom": 25.0, "per_tilt_weighting": True,
                    "pytom_volume_split": [1, 1, 1], "pytom_job_json_path": str(job)},
    )
    registry = ArtifactRegistry.empty()
    registry.extend([tomogram, template, mask, segmentation, real_picks])
    registry.write(project / "artifacts.json")
    return project


def _small_box_lane() -> dict:
    """The ribo-mdc lane, with a box that fits the test's small tomogram."""

    lane = copy.deepcopy(LANE)
    lane["extract"]["box_size_px"] = 8
    return lane


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


def test_phase_randomisation_keeps_every_amplitude_the_mean_and_the_spread() -> None:
    rng = np.random.default_rng(4)
    for shape in ((9, 12, 15), (16, 20, 24)):
        z, y, x = np.indices(shape)
        blob = 4 * np.exp(-((z - shape[0] / 2) ** 2 + (y - 6) ** 2 + (x - 7) ** 2) / 8)
        volume = (blob + rng.normal(size=shape) + 2).astype(np.float32)
        null = phase_randomised(volume, np.random.default_rng([1, 2]))
        assert null.dtype == np.float32 and null.shape == shape
        before = np.abs(scipy.fft.rfftn(volume.astype(np.float64)))
        after = np.abs(scipy.fft.rfftn(null.astype(np.float64)))
        np.testing.assert_allclose(after, before, atol=1e-5 * before.max())
        assert null.mean() == pytest.approx(volume.mean(), rel=1e-5)
        assert null.std() == pytest.approx(volume.std(), rel=1e-5)
        # The blob is gone from where it was; the same seed gives the same noise.
        assert abs(np.corrcoef(volume.ravel(), null.ravel())[0, 1]) < 0.2
        np.testing.assert_array_equal(null, phase_randomised(volume, np.random.default_rng([1, 2])))
        assert not np.array_equal(null, phase_randomised(volume, np.random.default_rng([1, 3])))


def test_the_copy_is_a_labelled_child_that_pick_particles_finds_by_its_label(
    project: Path,
) -> None:
    registry = ArtifactRegistry.load(project / "artifacts.json")
    real = registry.get(f"{SERIES}:tomogram:fine:bin8")
    null, made = null_decoys.make_null_tomogram(registry, real, variant="bin8-null", seed=1)
    registry.commit(project / "artifacts.json")
    assert made and null.id == f"{SERIES}:tomogram:phase-randomised:bin8-null"
    assert null.variant == "bin8-null" and null.parent_ids == [real.id]
    # The views and tilt angles stay, so its wedge and its search are the real one's;
    # nothing names the real volume.
    assert null.parameters["tilt_angles_path"] == real.parameters["tilt_angles_path"]
    assert null.parameters["included_z_values"] == [0, 1, 2]
    assert null.parameters["tomogram_branch"] == "full"
    assert "imod_rec_path" not in null.parameters and "log_path" not in null.parameters
    assert null.parameters["phase_seed"] == null_decoys.series_seed(1, SERIES)

    source = np.asarray(zarr.open(real.path, mode="r")[:])
    written = np.asarray(zarr.open(null.path, mode="r")[:])
    expected = phase_randomised(source, np.random.default_rng(null_decoys.series_seed(1, SERIES)))
    np.testing.assert_array_equal(written, expected)
    numbers = null.parameters["null_statistics"]
    assert numbers["standard_deviation"] == pytest.approx(numbers["source_standard_deviation"],
                                                          rel=1e-5)
    assert abs(numbers["correlation_with_source"]) < 0.05

    manifest = TiltSeriesManifest.model_validate_json(
        (project / "manifests" / f"{SERIES}.json").read_text())
    reloaded = ArtifactRegistry.load(project / "artifacts.json")
    assert cli._tomogram_artifact(reloaded, manifest, variant="bin8-null").id == null.id
    assert cli._tomogram_artifact(reloaded, manifest, variant="bin8").id == real.id

    # Asked again, the registered copy is used, not made twice; another seed is refused.
    again, made = null_decoys.make_null_tomogram(reloaded, real, variant="bin8-null", seed=1)
    assert not made and again.id == null.id
    with pytest.raises(SystemExit, match="pass --overwrite"):
        null_decoys.make_null_tomogram(reloaded, real, variant="bin8-null", seed=2)


def test_the_commands_are_the_lanes_but_for_label_count_and_template(project: Path) -> None:
    registry = ArtifactRegistry.load(project / "artifacts.json")
    real = null_decoys.real_picks(registry, SERIES, "bin8")
    template, mask, spherical = null_decoys.template_of(registry, real)
    pick, extract = null_decoys.lane_commands(
        LANE, project, SERIES, real=real, variant="bin8-null", count=120, template=template,
        mask=mask, spherical_mask=spherical)
    lane = dict(run_series.build_steps(LANE, project, SERIES))

    before, after = _options(lane["pick-particles"]), _options(pick)
    changed = {flag for flag in set(before) | set(after) if before.get(flag) != after.get(flag)}
    assert changed == {"--variant", "--pytom-number-of-particles", "--pytom-template",
                       "--pytom-template-mask", "--no-pytom-non-spherical-mask",
                       "--no-use-segmentation-mask"}
    assert after["--variant"] == "bin8-null" and after["--pytom-number-of-particles"] == "120"
    assert after["--pytom-template"] == str(template)
    assert float(str(after["--pytom-cut-off"])) == 0
    assert "--pytom-number-of-false-positives" not in after
    assert _options(extract) == {**_options(lane["extract-particles"]), "--variant": "bin8-null"}

    group = typer.main.get_command(cli.app)
    for command in (pick, extract):
        known = {option for param in group.commands[command[0]].params
                 for option in (*param.opts, *param.secondary_opts)}
        flags = {part for part in command[1:] if part.startswith("--")}
        assert flags <= known, sorted(flags - known)


def test_a_lane_that_no_longer_matches_the_real_picks_is_refused(project: Path) -> None:
    registry = ArtifactRegistry.load(project / "artifacts.json")
    real = null_decoys.real_picks(registry, SERIES, "bin8")
    template, mask, _ = null_decoys.template_of(registry, real)
    older = real.model_copy(update={"parameters": {**real.parameters,
                                                   "pytom_low_pass_angstrom": 35.0}})
    with pytest.raises(SystemExit, match="config has changed"):
        null_decoys.lane_commands(LANE, project, SERIES, real=older, variant="bin8-null",
                                  count=120, template=template, mask=mask, spherical_mask=True)
    masked = real.model_copy(update={"parameters": {
        **real.parameters, "segmentation_id": f"{SERIES}:segmentation:membrain-seg:bin8"}})
    with pytest.raises(SystemExit, match="inside the membrane mask"):
        null_decoys.lane_commands(LANE, project, SERIES, real=masked, variant="bin8-null",
                                  count=120, template=template, mask=mask, spherical_mask=True)
    # No low-pass in the lane: the one the real picks were matched at, named.
    lane = copy.deepcopy(LANE)
    del lane["pick"]["low_pass_angstrom"]
    pick, _ = null_decoys.lane_commands(lane, project, SERIES, real=real, variant="bin8-null",
                                        count=120, template=template, mask=mask,
                                        spherical_mask=True)
    assert _options(pick)["--pytom-low-pass"] == "25"


def test_job_differences_name_every_setting_but_the_tomogram(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    first.write_text(json.dumps({**JOB, "tomogram": "/a/t.mrc", "job_stats": {"std": 1}}))
    second.write_text(json.dumps({**JOB, "tomogram": "/b/t.mrc", "job_stats": {"std": 2}}))
    assert null_decoys.job_differences(first, second) == []
    second.write_text(json.dumps({**JOB, "tomogram": "/b/t.mrc", "low_pass": 35.0}))
    assert null_decoys.job_differences(first, second) == ["low_pass: 25.0 (real) vs 35.0 (null)"]


class _PicksInTheNull:
    """pytom's place: three picks where pick-particles says, recorded as the backend does."""

    name = "pytom-match-pick"

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict, list[str]]] = []

    def pick(self, tomogram: Artifact, manifest: TiltSeriesManifest, context,
             supporting_artifacts: Sequence[Artifact] = ()) -> Artifact:
        self.seen.append((tomogram.id, dict(context.parameters),
                          [artifact.id for artifact in supporting_artifacts]))
        folder = context.output_dir / "picks" / manifest.tilt_series_id
        star = folder / "null.star"
        _star(star, [(12, 12, 8, 0.20), (28, 14, 12, 0.15), (16, 30, 16, 0.10)])
        job = folder / "null_job.json"
        job.write_text(json.dumps({**JOB, "tomogram": "/scratch/2/tomogram.mrc",
                                   "template": str(context.parameters["pytom_template"]),
                                   "job_stats": {"std": 0.02}}))
        qc = folder / "null_qc.json"
        qc.write_text("{}\n")
        return Artifact(
            id=f"{manifest.tilt_series_id}:picks:full:{self.name}{variant_suffix(context)}",
            variant=variant_label(context), kind=ArtifactKind.PICKS, path=star,
            parent_ids=[tomogram.id],
            parameters={"backend": self.name, "tilt_series_id": manifest.tilt_series_id,
                        "tomogram_branch": "full", "input_tomogram_id": tomogram.id,
                        "pytom_job_json_path": str(job), "qc_path": str(qc)},
        )


def test_the_null_picks_reach_the_export_as_decoys_cut_from_the_copy(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lane = _small_box_lane()
    registry_path = project / "artifacts.json"
    registry = ArtifactRegistry.load(registry_path)
    real = null_decoys.real_picks(registry, SERIES, "bin8")
    null, _ = null_decoys.make_null_tomogram(
        registry, registry.get(str(real.parameters["input_tomogram_id"])), variant="bin8-null",
        seed=1)
    registry.commit(registry_path)
    template, mask, spherical = null_decoys.template_of(registry, real)
    pick, extract = null_decoys.lane_commands(
        lane, project, SERIES, real=real, variant="bin8-null", count=3, template=template,
        mask=mask, spherical_mask=spherical)

    backend = _PicksInTheNull()
    monkeypatch.setattr(cli, "_picking_backend", lambda name: backend)
    runner = CliRunner()
    result = runner.invoke(cli.app, pick)
    assert result.exit_code == 0, result.output
    (searched, parameters, supporting), = backend.seen
    assert searched == null.id and supporting == []
    assert Path(parameters["pytom_template"]) == template
    assert Path(parameters["pytom_template_mask"]) == mask
    assert parameters["pytom_number_of_particles"] == 3
    assert parameters["pytom_non_spherical_mask"] is False
    assert parameters["pytom_low_pass"] == 25.0

    registry = ArtifactRegistry.load(registry_path)
    found, = [artifact for artifact in registry.by_kind(ArtifactKind.PICKS)
              if artifact.variant == "bin8-null"]
    null_decoys.compare_with_real(real, found)
    printed = capsys.readouterr().out
    assert "3 null peaks, best 0.200" in printed and "2 of 4 above the null's best" in printed
    assert "the same search as the real picks'" in printed

    for command in (extract, dict(run_series.build_steps(lane, project, SERIES))[
            "extract-particles"]):
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 0, result.output

    out = tmp_path / "export"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_dynamo.py"), "--project", str(project),
         "--variant", "bin8", "--decoy-variant", "bin8-null", "--series", SERIES,
         "--out", str(out)],
        check=True,
    )
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["decoy_kind"] == "pytom picks in phase-randomised tomograms"
    assert [particle["decoy"] for particle in manifest["particles"]] == [False] * 4 + [True] * 3
    assert manifest["wedge"]["ranges"] == {SERIES: [-40.0, 50.0]}
    # Every decoy voxel is one of the copy's; of the real tomogram's, only the odd value
    # two random float32 volumes share by chance (about 0.2% at this size).
    real_values = np.asarray(zarr.open(project / "tomograms" / SERIES / "real.zarr",
                                       mode="r")[:]).ravel()
    null_values = np.asarray(zarr.open(null.path, mode="r")[:]).ravel()
    for particle in manifest["particles"]:
        voxels = mrcfile.read(out / particle["file"]).ravel()
        source, other = ((null_values, real_values) if particle["decoy"]
                         else (real_values, null_values))
        assert np.isin(voxels, source).all() and np.isin(voxels, other).mean() < 0.02


def test_the_dry_run_prints_both_commands_and_makes_nothing(project: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "null_decoys.py"), "--project", str(project),
         "--config", str(ROOT / "configs" / "lanes" / "ribo-mdc.toml"), "--series", SERIES],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "cryoet pick-particles" in result.stdout and "cryoet extract-particles" in result.stdout
    assert "--variant bin8-null" in result.stdout
    registry = ArtifactRegistry.load(project / "artifacts.json")
    assert [artifact.id for artifact in registry.by_kind(ArtifactKind.TOMOGRAM)] == [
        f"{SERIES}:tomogram:fine:bin8"]
