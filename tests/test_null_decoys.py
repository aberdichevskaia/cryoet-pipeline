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
from cryoet_pipeline.backends.picking import tomogram_search_mask
from cryoet_pipeline.backends.protocols import variant_label, variant_suffix
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder, TiltSeriesManifest
from cryoet_pipeline.null_volumes import phase_randomised
from cryoet_pipeline.reports.classification import SEARCHED_DECOY_KIND

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


def _membrane_lane(dilation: int) -> dict:
    """The small-box lane searching inside membranes grown by `dilation` px, as ryr1-bin8 does."""

    lane = _small_box_lane()
    lane["segment"] = {"model": "membrain.ckpt"}
    lane["pick"]["use_segmentation_mask"] = True
    lane["pick"]["segmentation_mask_dilation_px"] = dilation
    return lane


def _membranes(project: Path) -> Artifact:
    """The fixture's segmentation, written: a slab two voxels thick across the volume."""

    registry = ArtifactRegistry.load(project / "artifacts.json")
    segmentation = registry.get(f"{SERIES}:segmentation:membrain-seg:bin8")
    store = zarr.open(segmentation.path, mode="w", shape=SHAPE, chunks=(8, 20, 40),
                      dtype=np.float32)
    data = np.zeros(SHAPE, dtype=np.float32)
    data[10:12] = 1.0
    store[:] = data
    return segmentation


def _matched_inside(project: Path, segmentation: Artifact, dilation: int) -> Artifact:
    """The real picks as a search inside `segmentation` grown by `dilation` px records them."""

    registry = ArtifactRegistry.load(project / "artifacts.json")
    real = null_decoys.real_picks(registry, SERIES, "bin8")
    tomogram = registry.get(str(real.parameters["input_tomogram_id"]))
    coverage = float(tomogram_search_mask(segmentation, tomogram=tomogram,
                                          dilation_px=dilation).mean())
    return real.model_copy(update={"parameters": {
        **real.parameters, "segmentation_id": segmentation.id,
        "tomogram_mask_coverage": coverage}})


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

    # Each search writes its tomogram mask into a folder of its own, and pytom names the
    # whitening filter after its output folder: only having a mask or not is a difference.
    first.write_text(json.dumps({**JOB, "tomogram": "/a/t.mrc", "tomogram_mask": "/tmp/a/m.mrc",
                                 "whitening_filter": "/a/out/t_whitening_filter.npy"}))
    second.write_text(json.dumps({**JOB, "tomogram": "/b/t.mrc", "tomogram_mask": "/tmp/b/m.mrc",
                                  "whitening_filter": "/b/out/t_whitening_filter.npy"}))
    assert null_decoys.job_differences(first, second) == []
    second.write_text(json.dumps({**JOB, "tomogram": "/b/t.mrc", "tomogram_mask": None}))
    assert null_decoys.job_differences(first, second) == [
        'tomogram_mask: "/tmp/a/m.mrc" (real) vs null (null)']

    # As the first MDC null run had it: the real picks matched with the tracking stack's
    # nominal angles, the copy with the reconstruction's solved ones. Each entry of a
    # nested record is named, and a list by its length and ends.
    nominal = [-40.0093, -38.0096, -36.0048, -34.0106, 57.9832]
    solved = [-49.77, -48.35, -46.92, -45.5, 48.11]
    first.write_text(json.dumps({**JOB, "ts_metadata": {"tilt_angles": nominal,
                                                        "dose_accumulation": [0, 3, 6, 9]}}))
    second.write_text(json.dumps({**JOB, "ts_metadata": {"tilt_angles": solved,
                                                         "dose_accumulation": [0, 3, 6, 9]}}))
    assert null_decoys.job_differences(first, second) == [
        "ts_metadata.tilt_angles: 5 values -40.0093 .. 57.9832 (real) vs "
        "5 values -49.77 .. 48.11 (null)"]


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
        segmentation = next((artifact for artifact in supporting_artifacts
                             if artifact.kind is ArtifactKind.SEGMENTATION), None)
        coverage = None if segmentation is None else float(tomogram_search_mask(
            segmentation, tomogram=tomogram,
            dilation_px=context.parameters.get("pytom_segmentation_dilation_px")).mean())
        job = folder / "null_job.json"
        job.write_text(json.dumps({**JOB, "tomogram": "/scratch/2/tomogram.mrc",
                                   "template": str(context.parameters["pytom_template"]),
                                   "tomogram_mask": None if segmentation is None
                                   else "/scratch/2/mask.mrc",
                                   "rng_seed": context.parameters.get("pytom_rng_seed"),
                                   "job_stats": {"std": 0.02}}))
        qc = folder / "null_qc.json"
        qc.write_text("{}\n")
        return Artifact(
            id=f"{manifest.tilt_series_id}:picks:full:{self.name}{variant_suffix(context)}",
            variant=variant_label(context), kind=ArtifactKind.PICKS, path=star,
            parent_ids=[tomogram.id],
            parameters={"backend": self.name, "tilt_series_id": manifest.tilt_series_id,
                        "tomogram_branch": "full", "input_tomogram_id": tomogram.id,
                        "segmentation_id": None if segmentation is None else segmentation.id,
                        "tomogram_mask_coverage": coverage,
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
    # The report shows these decoys by their own scores only under this exact name.
    assert manifest["decoy_kind"] == SEARCHED_DECOY_KIND
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


def test_picks_matched_inside_membranes_are_mirrored_inside_the_same_mask(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    lane = _membrane_lane(dilation=2)
    segmentation = _membranes(project)
    real = _matched_inside(project, segmentation, dilation=2)
    registry_path = project / "artifacts.json"
    registry = ArtifactRegistry.load(registry_path)
    tomogram = registry.get(str(real.parameters["input_tomogram_id"]))
    template, mask, spherical = null_decoys.template_of(registry, real)
    pick, extract = null_decoys.lane_commands(
        lane, project, SERIES, real=real, variant="bin8-null", count=3, template=template,
        mask=mask, spherical_mask=spherical)

    # The lane's own search inside membranes, named by the real picks' segmentation:
    # the copy has none under its label, and the lane's dilation stays.
    before = _options(dict(run_series.build_steps(lane, project, SERIES))["pick-particles"])
    after = _options(pick)
    changed = {flag for flag in set(before) | set(after) if before.get(flag) != after.get(flag)}
    assert changed == {"--variant", "--pytom-number-of-particles", "--pytom-template",
                       "--pytom-template-mask", "--no-pytom-non-spherical-mask",
                       "--segmentation-id"}
    assert after["--segmentation-id"] == segmentation.id
    assert after["--use-segmentation-mask"] is True and "--no-use-segmentation-mask" not in after
    assert after["--segmentation-mask-dilation-px"] == "2"
    where = null_decoys.search_mask(registry, real, tomogram, pick)
    assert where == (f"inside {segmentation.id} grown by 2 px, 25.00% of the volume, as the "
                     "real picks were")

    null, _ = null_decoys.make_null_tomogram(registry, tomogram, variant="bin8-null", seed=1)
    registry.commit(registry_path)
    backend = _PicksInTheNull()
    monkeypatch.setattr(cli, "_picking_backend", lambda name: backend)
    result = CliRunner().invoke(cli.app, pick)
    assert result.exit_code == 0, result.output
    (searched, parameters, supporting), = backend.seen
    assert searched == null.id and supporting == [segmentation.id]
    assert parameters["pytom_segmentation_dilation_px"] == 2

    # Both searches had a mask, so their jobs agree; the masks are one by what the
    # pick lists record.
    job = Path(str(real.parameters["pytom_job_json_path"]))
    job.write_text(json.dumps({**json.loads(job.read_text()),
                               "tomogram_mask": "/scratch/1/mask.mrc"}))
    found, = [artifact for artifact in ArtifactRegistry.load(registry_path).by_kind(
        ArtifactKind.PICKS) if artifact.variant == "bin8-null"]
    null_decoys.compare_with_real(real, found)
    printed = capsys.readouterr().out
    assert "the same search as the real picks'" in printed
    assert f"search mask: {segmentation.id}, 25.00% of the volume, as the real picks'" in printed
    assert "WARNING" not in printed
    unmasked = found.model_copy(update={"parameters": {
        **found.parameters, "segmentation_id": None, "tomogram_mask_coverage": None}})
    null_decoys.compare_with_real(real, unmasked)
    assert "WARNING: the null search's mask, None" in capsys.readouterr().out


def test_a_mask_other_than_the_real_searchs_is_refused_before_anything_is_made(
    project: Path,
) -> None:
    segmentation = _membranes(project)
    real = _matched_inside(project, segmentation, dilation=2)
    registry = ArtifactRegistry.load(project / "artifacts.json")
    tomogram = registry.get(str(real.parameters["input_tomogram_id"]))
    template, mask, _ = null_decoys.template_of(registry, real)

    def pick_for(lane: dict, picks: Artifact) -> list[str]:
        return null_decoys.lane_commands(lane, project, SERIES, real=picks, variant="bin8-null",
                                         count=3, template=template, mask=mask,
                                         spherical_mask=True)[0]

    # The lane grows the membranes by another margin now: another mask.
    with pytest.raises(SystemExit, match="config has changed since those picks"):
        null_decoys.search_mask(registry, real, tomogram, pick_for(_membrane_lane(3), real))
    unrecorded = real.model_copy(update={"parameters": {
        key: value for key, value in real.parameters.items()
        if key != "tomogram_mask_coverage"}})
    with pytest.raises(SystemExit, match="does not record how much"):
        null_decoys.search_mask(registry, unrecorded, tomogram,
                                pick_for(_membrane_lane(2), unrecorded))
    gone = real.model_copy(update={"parameters": {
        **real.parameters, "segmentation_id": f"{SERIES}:segmentation:membrain-seg:gone"}})
    with pytest.raises(SystemExit, match="no longer registered"):
        null_decoys.search_mask(registry, gone, tomogram, pick_for(_membrane_lane(2), gone))

    # Picks matched in the whole volume stay unmasked whatever the lane says, and no
    # segmentation is read for them.
    plain = null_decoys.real_picks(registry, SERIES, "bin8")
    pick = pick_for(_membrane_lane(2), plain)
    assert "--no-use-segmentation-mask" in pick and "--segmentation-id" not in pick
    assert "--use-segmentation-mask" not in pick
    assert "--segmentation-mask-dilation-px" not in pick
    segmentation.path.rename(segmentation.path.with_name("elsewhere.zarr"))
    assert (null_decoys.search_mask(registry, plain, tomogram, pick)
            == "the whole volume, as the real picks were")


def test_pick_particles_takes_a_segmentation_by_id_and_nothing_else(
    project: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    segmentation = _membranes(project)
    real = _matched_inside(project, segmentation, dilation=2)
    registry_path = project / "artifacts.json"
    registry = ArtifactRegistry.load(registry_path)
    null_decoys.make_null_tomogram(registry, registry.get(str(real.parameters[
        "input_tomogram_id"])), variant="bin8-null", seed=1)
    registry.commit(registry_path)
    template, mask, _ = null_decoys.template_of(registry, real)
    pick, _ = null_decoys.lane_commands(_membrane_lane(2), project, SERIES, real=real,
                                        variant="bin8-null", count=3, template=template,
                                        mask=mask, spherical_mask=True)
    backend = _PicksInTheNull()
    monkeypatch.setattr(cli, "_picking_backend", lambda name: backend)
    runner = CliRunner()

    def named(value: str) -> list[str]:
        command = list(pick)
        command[command.index("--segmentation-id") + 1] = value
        return command

    unmasked = [part for part in pick if part != "--use-segmentation-mask"]
    for command, refusal in (
            ([*unmasked, "--no-use-segmentation-mask"], "searches without one"),
            (named(f"{SERIES}:segmentation:membrain-seg:gone"), "no artifact"),
            (named(f"{SERIES}:template:pytom:bin8"), "not a segmentation")):
        result = runner.invoke(cli.app, command)
        assert result.exit_code == 2, result.output
        assert refusal in result.output
    assert backend.seen == []


def test_the_null_search_takes_the_real_searchs_noise_seed(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    # As pytom writes it when it drew the seed itself: eight random bytes.
    seed = 16624184499262119753
    registry_path = project / "artifacts.json"
    registry = ArtifactRegistry.load(registry_path)
    real = null_decoys.real_picks(registry, SERIES, "bin8")
    job = Path(str(real.parameters["pytom_job_json_path"]))
    job.write_text(json.dumps({**json.loads(job.read_text()), "rng_seed": seed}))
    template, mask, spherical = null_decoys.template_of(registry, real)
    pick, _ = null_decoys.lane_commands(
        _small_box_lane(), project, SERIES, real=real, variant="bin8-null", count=3,
        template=template, mask=mask, spherical_mask=spherical)
    assert _options(pick)["--pytom-rng-seed"] == str(seed)

    null, _ = null_decoys.make_null_tomogram(
        registry, registry.get(str(real.parameters["input_tomogram_id"])),
        variant="bin8-null", seed=1)
    registry.commit(registry_path)
    backend = _PicksInTheNull()
    monkeypatch.setattr(cli, "_picking_backend", lambda name: backend)
    result = CliRunner().invoke(cli.app, pick)
    assert result.exit_code == 0, result.output
    (_, parameters, _), = backend.seen
    assert parameters["pytom_rng_seed"] == seed
    found, = [artifact for artifact in ArtifactRegistry.load(registry_path).by_kind(
        ArtifactKind.PICKS) if artifact.variant == "bin8-null"]
    null_decoys.compare_with_real(real, found)
    printed = capsys.readouterr().out
    assert "the same search as the real picks'" in printed and "WARNING" not in printed

    # A real search whose job file is gone leaves the seed to pytom, as before.
    job.unlink()
    pick, _ = null_decoys.lane_commands(
        _small_box_lane(), project, SERIES, real=real, variant="bin8-null", count=3,
        template=template, mask=mask, spherical_mask=spherical)
    assert "--pytom-rng-seed" not in pick
