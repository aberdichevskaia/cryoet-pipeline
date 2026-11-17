"""The methods record: a result's lineage as the registry and the run records hold it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_tomogram_import import SHAPE, wedged_noise, write_angles, write_mrc
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.methods_record import (
    MethodsRefused,
    methods_markdown,
    methods_record,
    ordered_lineage,
    write_methods_record,
)
from cryoet_pipeline.models import Artifact, ArtifactKind
from cryoet_pipeline.runs import bytes_fingerprint, read_record, writing
from cryoet_pipeline.tomogram_import import import_tomogram


def _chain(root: Path) -> ArtifactRegistry:
    """Five views, their stack, an alignment, a tomogram and picks, made here."""

    views = [
        Artifact(id=f"TS_01:corrected_projection:{index:03d}",
                 kind=ArtifactKind.CORRECTED_PROJECTION, path=root / f"view{index}.zarr",
                 pixel_spacing_angstrom=1.35, software_versions={"motioncor3": "1.1.0"},
                 parameters={"tilt_series_id": "TS_01", "backend": "motioncor3",
                             "z_value": index, "patches": [5, 5],
                             "source_frame_file": f"/data/frames_{index}.tif"})
        for index in range(5)
    ]
    stack = Artifact(id="TS_01:tilt_stack", kind=ArtifactKind.TILT_STACK,
                     path=root / "stack.zarr", parent_ids=[view.id for view in views],
                     parameters={"tilt_series_id": "TS_01", "backend": "simple"})
    alignment = Artifact(id="TS_01:alignment:fine", kind=ArtifactKind.ALIGNMENT,
                         path=root / "alignment.json", parent_ids=[stack.id],
                         software_versions={"imod": "5.1.12"},
                         parameters={"tilt_series_id": "TS_01", "backend": "imod_tiltalign",
                                     "imod_xf_path": "/x/TS_01.xf", "rounds": 4})
    tomogram = Artifact(id="TS_01:tomogram:fine", kind=ArtifactKind.TOMOGRAM,
                        path=root / "tomogram.zarr", parent_ids=[stack.id, alignment.id],
                        pixel_spacing_angstrom=5.4, software_versions={"imod": "external"},
                        parameters={"tilt_series_id": "TS_01", "backend": "imod_tilt",
                                    "thickness": 296})
    picks = Artifact(id="TS_01:picks:pytom", kind=ArtifactKind.PICKS, path=root / "picks.star",
                     parent_ids=[tomogram.id], software_versions={"pytom-match-pick": "0.10.0"},
                     parameters={"tilt_series_id": "TS_01", "backend": "pytom",
                                 "input_tomogram_id": tomogram.id})
    qc = Artifact(id="TS_01:qc:picks", kind=ArtifactKind.QC, path=root / "qc.json",
                  parent_ids=[picks.id], parameters={"tilt_series_id": "TS_01"})
    registry = ArtifactRegistry.empty()
    registry.extend([*views, stack, alignment, tomogram, picks, qc])
    registry.commit(root / "artifacts.json")
    return registry


def test_the_lineage_comes_parents_first_with_the_views_as_one_step(tmp_path: Path) -> None:
    registry = _chain(tmp_path)

    order = [artifact.id for artifact in ordered_lineage(registry, ["TS_01:picks:pytom"])]
    record = methods_record(tmp_path, artifact_ids=["TS_01:picks:pytom"])

    assert order.index("TS_01:tilt_stack") > max(order.index(f"TS_01:corrected_projection:"
                                                             f"{index:03d}") for index in range(5))
    assert order[-3:] == ["TS_01:alignment:fine", "TS_01:tomogram:fine", "TS_01:picks:pytom"]
    names = [step["name"] for step in record["steps"]]
    assert names == ["TS_01:corrected_projection:*", "TS_01:tilt_stack", "TS_01:alignment:fine",
                     "TS_01:tomogram:fine", "TS_01:picks:pytom"]
    views = record["steps"][0]
    assert views["count"] == 5 and views["parameters"]["patches"] == [5, 5]
    assert views["varying"]["z_value"] == {"from": 0, "to": 4}
    assert record["steps"][1]["parents"] == ["TS_01:corrected_projection:*"]
    assert record["programs"]["imod"] == {"5.1.12": ["TS_01:alignment:fine"],
                                          "not recorded": ["TS_01:tomogram:fine"]}
    assert record["not_made_here"] == []

    markdown = methods_markdown(record)
    assert "| motioncor3 | 1.1.0 | TS_01:corrected_projection:* |" in markdown
    assert "z_value 0 to 4" in markdown and "rounds 4" in markdown
    # Paths stay in the JSON and out of the Markdown.
    assert "source_frame_file" in views["varying"] and "source_frame_file" not in markdown
    assert "imod_xf_path" not in markdown
    assert "not recorded for 5 of 5 steps" in markdown


def test_what_entered_from_outside_is_listed_with_its_declarations(tmp_path: Path) -> None:
    volume = write_mrc(tmp_path / "volume.mrc", wedged_noise(SHAPE, -60.0, 30.0))
    angles = write_angles(tmp_path / "volume.tlt")
    project = tmp_path / "project"
    imported = import_tomogram(volume_path=volume, tilt_angles_path=angles,
                               orientation_name="imod-parallel", processing="raw",
                               tilt_series_id="TS_01", out=project)

    record = methods_record(project, artifact_ids=[imported.tomogram.id])

    (outside,) = record["not_made_here"]
    assert outside["id"] == imported.tomogram.id and outside["provenance"] == "imported"
    assert outside["declared"]["orientation"] == "imod-parallel"
    assert outside["declared"]["processing"] == []
    fingerprints = {entry["slot"]: entry["fingerprint"] for entry in outside["inputs"]}
    assert fingerprints == {"volume": bytes_fingerprint(volume),
                            "tilt_angles": bytes_fingerprint(angles)}
    (step,) = record["steps"]
    assert step["runs"][0]["step"] == "import-tomogram"
    markdown = methods_markdown(record)
    assert "Declared: tilt_series_id TS_01, orientation imod-parallel" in markdown


def _stack(root: Path) -> None:
    registry = _chain(root)
    registry.add(Artifact(id="TS_01:particle_stack:cubic-crop:lane",
                          kind=ArtifactKind.PARTICLE_STACK, variant="lane",
                          path=root / "stack.zarr",
                          parent_ids=["TS_01:tomogram:fine", "TS_01:picks:pytom"],
                          parameters={"tilt_series_id": "TS_01", "backend": "cubic-crop",
                                      "box_size_px": 48}))
    registry.commit(root / "artifacts.json")


def test_an_mra_run_brings_its_plan_its_nodes_and_its_stacks_lineage(tmp_path: Path) -> None:
    _stack(tmp_path)
    run = tmp_path / "mra" / "run1"
    run.mkdir(parents=True)
    plan = {"particles": {"project": str(tmp_path), "label": "lane", "series": ["TS_01"],
                          "decoy_label": None}, "setup": {"channels": 2}}
    (run / "plan.json").write_text(json.dumps(plan))
    with writing(run / "setup", step="mra-setup", step_version=2, backend="dynamo-port",
                 backend_version=1, content_params={"channels": 2}):
        pass

    record = methods_record(tmp_path, run=run)

    assert record["steps"][-1]["name"] == "TS_01:particle_stack:cubic-crop:lane"
    assert record["mra"]["plan"] == plan
    (node,) = record["mra"]["nodes"]
    assert node["step"] == "mra-setup" and node["backend"] == "dynamo-port"
    assert "commit" in node["code"] or "package" in node["code"]
    assert "## MRA run" in methods_markdown(record)


def test_an_mra_run_of_an_export_folder_or_no_result_is_refused(tmp_path: Path) -> None:
    _chain(tmp_path)
    run = tmp_path / "export" / "run1"
    run.mkdir(parents=True)
    (run / "plan.json").write_text(json.dumps({"particles": "/elsewhere/export"}))
    with pytest.raises(MethodsRefused, match="export folder"):
        methods_record(tmp_path, run=run)
    with pytest.raises(MethodsRefused, match="give the result"):
        methods_record(tmp_path)
    with pytest.raises(MethodsRefused, match="no artifact"):
        methods_record(tmp_path, artifact_ids=["TS_01:picks:other"])


def test_the_record_is_a_run_and_the_command_line_writes_it(tmp_path: Path) -> None:
    _chain(tmp_path)
    written = write_methods_record(tmp_path, tmp_path / "methods",
                                   artifact_ids=["TS_01:tomogram:fine"])
    record = read_record(written.folder)
    assert record is not None and record.step == "methods-record"
    assert json.loads((written.folder / "methods.json").read_text())["steps"]
    with pytest.raises(MethodsRefused, match="never written over"):
        write_methods_record(tmp_path, tmp_path / "methods", artifact_ids=["TS_01:tomogram:fine"])

    outcome = CliRunner().invoke(cli.app, [
        "methods-record", "--project", str(tmp_path), "--artifact", "TS_01:picks:pytom",
    ])
    assert outcome.exit_code == 0, outcome.output
    assert (tmp_path / "exports" / "methods_TS_01_picks_pytom" / "methods.md").is_file()
    assert "5 steps" in outcome.output
