"""Lanes whose series enter as tomograms, picks or particles made elsewhere
(docs/entry_and_exit.md, Order of work item 6).

The import is the lane's first step, its questions answered in [data], and the
chain runs on from it under the lane's label. What the lane does not read is
refused, as an unread key is, and what the imports would refuse at once is named by
the dry run, before the queue. Two lanes run here from their files: a picks lane to
the particles the MRA reads, and a particles lane one series at a time.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import pytest
from test_particles_import import _folder, particles  # noqa: F401  (a fixture)
from test_picks_import import (  # noqa: F401  (scene is a fixture)
    DIAMETER,
    HIGH,
    LOW,
    PICKED,
    VOXEL,
    F,
    _pytom_star,
    _source_angles,
    scene,
)
from test_run_series_lanes import _assert_valid_commands

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.dynamo_search import dynamo_ytilt
from cryoet_pipeline.models import ArtifactKind
from cryoet_pipeline.particle_sources import registry_particles
from cryoet_pipeline.particles_import import import_particles
from cryoet_pipeline.tomogram_import import ImportRefused

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import submit_lanes  # noqa: E402

PYTOM = {
    "pytom_create_template": "$PYTOMO_MP_DIR/pytom_create_template.py",
    "pytom_create_mask": "$PYTOMO_MP_DIR/pytom_create_mask.py",
    "pytom_match_template": "$PYTOMO_MP_DIR/pytom_match_template.py",
    "pytom_extract_candidates": "$PYTOMO_MP_DIR/pytom_extract_candidates.py",
}
TOMOGRAM_DATA = {"tomogram": "data/{series}_rec.mrc", "tilt_angles": "data/{series}.tlt",
                 "orientation": "imod-standard", "processing": "raw"}


def _lane(source: str, data: dict[str, Any], **sections: Any) -> dict[str, Any]:
    return {"lane": {"name": f"ext-{source}", "input": source, "variant": "bin8"},
            "run": {"project": "outputs/ext", "series": ["TS_01"]},
            "data": data, **sections}


def tomograms_lane(**sections: Any) -> dict[str, Any]:
    return _lane("tomograms", dict(TOMOGRAM_DATA),
                 template={"emdb_id": "EMD-1234", "low_pass_angstrom": 20.0},
                 pick={"particle_diameter_angstrom": 250.0, "low_pass_angstrom": 30.0,
                       "gpu_ids": "0"},
                 extract={"box_size_px": 48}, executables=dict(PYTOM), **sections)


def picks_lane(picks_format: str = "pytom-star", **data: Any) -> dict[str, Any]:
    return _lane("picks", {**TOMOGRAM_DATA, "picks": "data/{series}_picks",
                           "picks_format": picks_format, **data},
                 pick={"particle_diameter_angstrom": 250.0}, extract={"box_size_px": 48})


def particles_lane(**data: Any) -> dict[str, Any]:
    return _lane("particles", {"particles_folder": "data/particles",
                               "particles_table": "data/particles.tbl",
                               "particles_series": {"TS_01": 1, "TS_02": 2}, **data},
                 pick={"particle_diameter_angstrom": 250.0})


# --- What each input renders ---------------------------------------------------------

FORMAT_ANSWERS = {
    "pytom-star": {"template_box_px": 36},
    "csv": {"csv_columns": ["x", "y", "z", "score"], "coordinate_units": "voxels",
            "coordinate_origin": "voxel-0"},
    "relion5-star": {"relion_tomograms_star": "data/tomograms.star"},
    "imod-model": {"imod_object": 2, "picked_in": "imod-tilt"},
    "dynamo-table": {"dynamo_tomograms": {"TS_X": 3}},
}


@pytest.mark.parametrize(("config", "steps"), [
    (tomograms_lane(), ["import-tomogram", "prepare-template", "pick-particles",
                        "visualize-picks", "extract-particles"]),
    (tomograms_lane(segment={"model": "$MEMBRAIN_CKPT"}),
     ["import-tomogram", "segment-tomogram", "prepare-template", "pick-particles",
      "visualize-picks", "extract-particles"]),
    *[(picks_lane(name, **answers),
       ["import-tomogram", "import-picks", "visualize-picks", "extract-particles"])
      for name, answers in FORMAT_ANSWERS.items()],
    (particles_lane(particles_series={"TS_X": 1, "TS_02": 2}), ["import-particles"]),
], ids=["tomograms", "tomograms+membranes", *FORMAT_ANSWERS, "particles"])
def test_each_imported_input_renders_commands_the_cli_accepts(config, steps):
    assert _assert_valid_commands(config, config["lane"]["input"]) == steps


def test_the_answers_reach_their_imports_and_the_label_every_step():
    picks = dict(run_series.build_steps(
        picks_lane("dynamo-table", dynamo_tomograms={"TS_01": 3}, picked_in="imod-tilt",
                   processing=["ctf-corrected", "denoised"], tomogram_pixel_size_angstrom=5.4),
        Path("p"), "TS_01"))
    assert picks["import-tomogram"] == [
        "import-tomogram", "--volume", "data/TS_01_rec.mrc", "--tilt-angles", "data/TS_01.tlt",
        "--orientation", "imod-standard", "--processing", "ctf-corrected,denoised",
        "--tilt-series-id", "TS_01", "--out", "p", "--pixel-size", "5.4", "--variant", "bin8"]
    assert picks["import-picks"] == [
        "import-picks", "--picks", "data/TS_01_picks", "--format", "dynamo-table",
        "--tilt-series-id", "TS_01", "--out", "p", "--particle-diameter", "250.0",
        "--tomogram-variant", "bin8", "--picked-in", "imod-tilt", "--dynamo-tomogram", "3",
        "--variant", "bin8"]
    assert picks["extract-particles"][-2:] == ["--variant", "bin8"]

    (step,) = run_series.build_steps(particles_lane(particles_pixel_size_angstrom=5.4),
                                     Path("p"), "TS_02")
    assert step[1] == [
        "import-particles", "--folder", "data/particles", "--table", "data/particles.tbl",
        "--out", "p", "--series", "1=TS_01", "--series", "2=TS_02", "--only-series", "TS_02",
        "--pixel-size", "5.4", "--variant", "bin8"]


def test_imports_are_never_given_overwrite(tmp_path):
    config = tmp_path / "lane.toml"
    config.write_text("""
[lane]
name = "ext"
input = "picks"
variant = "bin8"
[data]
tomogram = "data/{series}.mrc"
tilt_angles = "data/{series}.tlt"
orientation = "pipeline"
processing = "raw"
picks = "data/{series}.star"
picks_format = "pytom-star"
template_box_px = 36
[pick]
particle_diameter_angstrom = 250.0
[extract]
box_size_px = 48
""")
    shown = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_series.py"), "--project", "p", "--config",
         str(config), "--series", "TS_01", "--overwrite"],
        capture_output=True, text=True, check=True).stdout
    commands = [line for line in shown.splitlines() if line.startswith("cryoet ")]
    assert [line.split()[1] for line in commands] == [
        "import-tomogram", "import-picks", "visualize-picks", "extract-particles"]
    assert [line.endswith(" --overwrite") for line in commands] == [False, False, True, True]


# --- What a lane does not read is refused --------------------------------------------


def test_the_starter_config_as_a_tomograms_lane_is_told_every_section_it_does_not_read():
    config = tomllib.loads((ROOT / "configs" / "lanes" / "example.toml").read_text())
    config["lane"]["input"] = "tomograms"
    config["data"].update(TOMOGRAM_DATA)
    with pytest.raises(SystemExit) as refused:
        run_series.build_steps(config, Path("p"), "TS_01")
    message = str(refused.value)
    assert "its series enter as tomograms reconstructed elsewhere" in message
    for named in ("[align]", "[binning]", "[ctf]", "[fiducials]", "[fine_align]", "[motion]",
                  "[qc_coarse]", "[reconstruct]", "[data] frames, read with input 'frames'",
                  "[data] mdocs, read with input 'frames'", "[executables] motioncor3"):
        assert f"\n  {named}" in message, named
    for kept in ("[segment]", "[template]", "[pick]", "[extract]"):
        assert f"\n  {kept}\n" not in message + "\n", kept


@pytest.mark.parametrize(("config", "named"), [
    (picks_lane(**FORMAT_ANSWERS["pytom-star"]) | {"pick": {"particle_diameter_angstrom": 250.0,
                                                            "gpu_ids": "0"}},
     "[pick] gpu_ids"),
    (particles_lane() | {"extract": {"box_size_px": 48}}, "[extract]"),
    (picks_lane("pytom-star", template_box_px=36, csv_columns=["x", "y", "z"]),
     "[data] csv_columns, read with picks_format csv"),
    (picks_lane("relion5-star", picked_in="imod-tilt"),
     "[data] picked_in, read with picks_format pytom-star or csv or imod-model or dynamo-table"),
    (particles_lane(tomogram="data/x.mrc"), "[data] tomogram, read with input 'tomograms' or "
                                            "'picks'"),
], ids=["pick of picks", "extract of particles", "csv key with pytom", "picked_in of relion5",
        "tomogram key of particles"])
def test_what_an_imported_input_does_not_read_is_refused(config, named):
    with pytest.raises(SystemExit, match=re.escape(f"\n  {named}")):
        run_series.build_steps(config, Path("p"), "TS_01")


@pytest.mark.parametrize(("lane", "key"), [("ribo-bin8", "frames"), ("gag-bin8", "stack")])
def test_a_data_key_of_another_input_is_refused_in_the_lanes_that_reconstruct(lane, key):
    config = tomllib.loads((ROOT / "configs" / "lanes" / f"{lane}.toml").read_text())
    config["data"][key] = "data/somewhere"
    with pytest.raises(SystemExit, match=f"\\[data\\] {key}, read with input"):
        run_series.build_steps(config, Path("p"), "TS_01")


def test_a_tomograms_lane_states_its_low_pass_and_weights_by_no_ctf_it_lacks():
    config = tomograms_lane()
    del config["pick"]["low_pass_angstrom"]
    with pytest.raises(SystemExit, match="low_pass_angstrom must be set"):
        run_series.build_steps(config, Path("p"), "TS_01")
    config = tomograms_lane()
    config["pick"]["ctf_weighting"] = True
    with pytest.raises(SystemExit, match="without a CTF fit"):
        run_series.build_steps(config, Path("p"), "TS_01")
    config["pick"]["ctf_weighting"] = False
    assert "--no-pytom-ctf-weighting" in dict(
        run_series.build_steps(config, Path("p"), "TS_01"))["pick-particles"]


@pytest.mark.parametrize(("mapping", "match"), [
    ([1, 2], "must map each series"),
    ({"TS_01": True}, "must map each series"),
    ({"TS_01": 1, "TS_02": 1}, "two series the same tomogram number"),
])
def test_particles_series_maps_each_series_to_one_tomogram_number(mapping, match):
    with pytest.raises(SystemExit, match=match):
        run_series.build_steps(particles_lane(particles_series=mapping), Path("p"), "TS_01")


def test_one_dynamo_table_for_several_series_names_each_ones_tomogram():
    with pytest.raises(SystemExit, match="gives no tomogram number for TS_02"):
        run_series.build_steps(picks_lane("dynamo-table", dynamo_tomograms={"TS_01": 1}),
                               Path("p"), "TS_02")


def test_a_lane_of_imported_files_is_its_own_combination():
    combination = run_series.lane_combination
    text = run_series.combination_text
    assert text(combination(tomograms_lane())) == "tomograms, no membranes"
    assert text(combination(tomograms_lane(segment={"model": "x"}))) == "tomograms + membranes"
    assert text(combination(picks_lane())) == "picks"
    assert text(combination(particles_lane())) == "particles"
    # The test of a tomograms lane ran one with membranes in Munich on 19 November 2026.
    assert combination(tomograms_lane(segment={"model": "x"})) in run_series.RUN_ON_REAL_DATA
    assert not {combination(tomograms_lane()), combination(picks_lane()),
                combination(particles_lane())} & run_series.RUN_ON_REAL_DATA


# --- What the dry run names before the queue ----------------------------------------


def _write_lane(tmp_path: Path, config: dict[str, Any]) -> Path:
    """`config` as a lane file, its [data] paths made absolute under tmp_path."""
    lines = []
    for section, values in config.items():
        if section == "run":
            values = {**values, "project": str(tmp_path / "project")}
        lines.append(f"[{section}]")
        for key, value in values.items():
            if section == "data" and isinstance(value, str) and "/" in value:
                value = str(tmp_path / value)
            if isinstance(value, dict):
                rendered = "{" + ", ".join(f"{k} = {v!r}" for k, v in value.items()) + "}"
            elif isinstance(value, str):
                rendered = f'"{value}"'
            elif isinstance(value, bool):
                rendered = str(value).lower()
            else:
                rendered = repr(value).replace("'", '"')
            lines.append(f"{key} = {rendered}")
    lines += ["[[run.stage]]", 'name = "cpu"', f'from = "{_first(config)}"',
              f'until = "{_last(config)}"', 'time = "1:00:00"', 'sbatch = ["--partition=cpu"]']
    path = tmp_path / "lane.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _first(config: dict[str, Any]) -> str:
    return run_series.build_steps(config, Path("p"), "TS_X")[0][0]


def _last(config: dict[str, Any]) -> str:
    return run_series.build_steps(config, Path("p"), "TS_X")[-1][0]


def _touch(tmp_path: Path, *names: str) -> None:
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def test_without_series_the_lanes_find_them_in_their_files_or_their_table(tmp_path):
    _touch(tmp_path, "data/TS_02_rec.mrc", "data/TS_01_rec.mrc", "data/notes.txt")
    config = tomograms_lane()
    del config["run"]["series"]
    lane = submit_lanes.plan(tomllib.loads(_write_lane(tmp_path, config).read_text()),
                             tmp_path / "lane.toml")
    assert lane.series == ("TS_01", "TS_02")

    config = particles_lane()
    del config["run"]["series"]
    assert submit_lanes.every_series(config, tmp_path / "lane.toml") == ("TS_01", "TS_02")


def _problems(tmp_path: Path, config: dict[str, Any], series: tuple[str, ...]) -> list[str]:
    path = _write_lane(tmp_path, config)
    cfg = tomllib.loads(path.read_text())
    lane = submit_lanes.Plan(config=path, project=str(tmp_path / "project"), series=series,
                             stages=())
    return submit_lanes.import_problems(cfg, lane)


def test_the_dry_run_names_what_the_imports_would_refuse(tmp_path):
    _touch(tmp_path, "data/TS_01_rec.mrc", "data/TS_01.tlt", "data/TS_02_rec.mrc",
           "data/TS_02.tlt", "data/picks.star")
    assert _problems(tmp_path, tomograms_lane(), ("TS_01", "TS_02")) == []

    config = tomograms_lane()
    config["data"].update(orientation="aretomo", processing=["denoised", "smoothed"])
    found = _problems(tmp_path, config, ("TS_01", "TS_03"))
    assert any(line.startswith("[data] tomogram: no file") and "TS_03_rec.mrc" in line
               for line in found)
    assert any(line.startswith("[data] orientation: 'aretomo' is not on the list")
               for line in found)
    assert any(line.startswith("[data] processing: --processing takes") for line in found)

    config = picks_lane("warp-star")
    config["data"]["picks"] = "data/picks.star"
    found = _problems(tmp_path, config, ("TS_01", "TS_02"))
    assert any("[data] picks_format: 'warp-star' is not on the list yet" in line for line in found)
    config = picks_lane("pytom-star", template_box_px=36)
    config["data"]["picks"] = "data/picks.star"
    assert _problems(tmp_path, config, ("TS_01", "TS_02")) == [
        "[data] picks names one file for 2 series, and a pytom-star file holds no series of its "
        "rows; give each series its own with {series}"]
    config = picks_lane("dynamo-table")
    config["data"]["picks"] = "data/picks.star"
    assert "dynamo_tomograms says which" in _problems(tmp_path, config, ("TS_01", "TS_02"))[0]
    config = picks_lane("relion5-star", relion_tomo_name="tomo")
    config["data"]["picks"] = "data/picks.star"
    assert "relion_tomo_name names one tomogram" in _problems(tmp_path, config,
                                                              ("TS_01", "TS_02"))[0]
    config["data"]["relion_tomo_name"] = "{series}_rec"
    assert _problems(tmp_path, config, ("TS_01", "TS_02")) == []


def test_the_dry_run_holds_the_table_to_the_particles_series(tmp_path, particles):  # noqa: F811
    folder, table, _ = _folder(tmp_path, particles)  # tomograms 1 and 2
    config = particles_lane(particles_series={"TS_01": 1, "TS_03": 3})
    config["data"].update(particles_folder=str(folder), particles_table=str(table))
    found = _problems(tmp_path, config, ("TS_01", "TS_03", "TS_09"))
    assert found == [
        "[data] particles_series names no tomogram number for TS_09",
        "[data] particles_table holds rows of tomograms [2] (column 20), which [data] "
        "particles_series gives no series",
        "[data] particles_table has no rows for TS_03, by [data] particles_series",
    ]


# --- Lanes run from their files ------------------------------------------------------


def _execute(config: Path, project: Path, *series: str) -> subprocess.CompletedProcess[str]:
    # The steps run `cryoet` from PATH, as the jobs do.
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ.get("PATH", "")])
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_series.py"), "--project", str(project),
         "--config", str(config), *[part for one in series for part in ("--series", one)],
         "--execute"],
        capture_output=True, text=True, env={**os.environ, "PATH": path}, cwd=ROOT)


@pytest.mark.skipif(shutil.which("point2model") is None,
                    reason="visualize-picks writes its model with IMOD's point2model")
def test_a_picks_lane_runs_from_its_files_to_the_particles_the_mra_reads(
    tmp_path: Path, scene: dict[str, Any]  # noqa: F811
) -> None:
    # The tomogram as IMOD's standard orientation has it, z reversed, and picks
    # pytom took in it, half a voxel above their centres for an even template box.
    data = tmp_path / "data"
    data.mkdir()
    with mrcfile.new(data / "TS_01_rec.mrc") as mrc:
        mrc.set_data(np.ascontiguousarray(scene["volume"][::-1]))
        mrc.voxel_size = VOXEL
    (data / "TS_01.tlt").write_text("".join(f"{a:.1f}\n" for a in np.arange(LOW, HIGH + 0.1, 2.0)))
    to_source, _ = PICKED["imod-standard"]
    _pytom_star(data / "TS_01_picks", to_source(scene["centres"]) + 0.5,
                _source_angles(scene["angles"], F))
    config = picks_lane("pytom-star", template_box_px=16)
    config["pick"]["particle_diameter_angstrom"] = DIAMETER
    config["extract"]["box_size_px"] = 16
    project = tmp_path / "project"

    ran = _execute(_write_lane(tmp_path, config), project, "TS_01")

    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    registry = ArtifactRegistry.load(project / "artifacts.json")
    (stack,) = registry.by_kind(ArtifactKind.PARTICLE_STACK)
    assert stack.variant == "bin8"
    assert {artifact.id for artifact in registry.not_derived_lineage(stack.id)} == {
        "TS_01:tomogram:imported:bin8", "TS_01:picks:imported:bin8"}
    logged: list[str] = []
    run = registry_particles(project, "bin8", ["TS_01"], log=logged.append)
    # Dynamo's column 24 holds the pick plus 0.5; the stack's STAR keeps three decimals.
    picked = run.table[:, 23:26] - 0.5
    order = np.lexsort(picked.T[::-1])
    expected = scene["centres"][np.lexsort(scene["centres"].T[::-1])]
    np.testing.assert_allclose(picked[order], expected, atol=1e-3)
    # The MRA's wedge is the imported tomogram's own angles, as Dynamo's ytilt, and
    # not a manifest's: an imported series has none of its own.
    assert (f"TS_01 particles: wedge {LOW:+.1f} to {HIGH:+.1f} deg, from the "
            "reconstruction's solved angles") in logged, logged
    np.testing.assert_array_equal(run.table[:, 13:15],
                                  np.tile(dynamo_ytilt((LOW, HIGH)), (len(run.table), 1)))


def test_a_particles_lane_imports_its_table_one_series_at_a_time(
    tmp_path: Path, particles: list[np.ndarray]  # noqa: F811
) -> None:
    folder, table, rows = _folder(tmp_path, particles)
    config = particles_lane()
    config["data"].update(particles_folder=str(folder), particles_table=str(table))
    project = tmp_path / "project"

    ran = _execute(_write_lane(tmp_path, config), project, "TS_01", "TS_02")

    assert ran.returncode == 0, ran.stdout[-3000:] + ran.stderr[-3000:]
    run = registry_particles(project, "bin8", ["TS_01", "TS_02"], log=lambda _: None)
    assert len(run.particles) == len(rows)
    # The whole table at once is the same two runs, by their fingerprints.
    again = import_particles(folder=folder, table_path=table, out=project,
                             series={1: "TS_01", 2: "TS_02"}, variant="bin8")
    assert [result.already_imported for result in again] == [True, True]


def test_one_series_of_a_table_is_imported_only_with_the_whole_table_named(
    tmp_path: Path, particles: list[np.ndarray]  # noqa: F811
) -> None:
    folder, table, _ = _folder(tmp_path, particles)
    project = tmp_path / "project"
    with pytest.raises(ImportRefused, match="rows of tomograms \\[2\\]"):
        import_particles(folder=folder, table_path=table, out=project, series={1: "TS_01"},
                         only_series="TS_01")
    with pytest.raises(ImportRefused, match="names no tomogram number for it"):
        import_particles(folder=folder, table_path=table, out=project,
                         series={1: "TS_01", 2: "TS_02"}, only_series="TS_09")
    with pytest.raises(ImportRefused, match="no rows of tomogram 3"):
        import_particles(folder=folder, table_path=table, out=project,
                         series={1: "TS_01", 2: "TS_02", 3: "TS_03"}, only_series="TS_03")
    (one,) = import_particles(folder=folder, table_path=table, out=project,
                              series={1: "TS_01", 2: "TS_02"}, only_series="TS_02")
    assert one.tilt_series_id == "TS_02"
    assert {artifact.id for artifact in ArtifactRegistry.load(project / "artifacts.json")
            .by_kind(ArtifactKind.PARTICLE_STACK)} == {"TS_02:particle_stack:imported"}
