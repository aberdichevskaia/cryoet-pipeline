from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.models import ProjectConfig
from cryoet_pipeline.project import InitResult
from cryoet_pipeline.runtime import DevicePreference


def test_init_command_passes_normalized_project_config(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, ProjectConfig] = {}

    def fake_initialize_project(config: ProjectConfig) -> InitResult:
        captured["config"] = config
        return InitResult(
            project_path=tmp_path / "project.json",
            manifest_paths=[],
            artifact_registry_path=tmp_path / "artifacts.json",
        )

    monkeypatch.setattr(cli, "initialize_project", fake_initialize_project)

    result = CliRunner().invoke(
        cli.app,
        [
            "init",
            "--frames",
            str(tmp_path / "frames"),
            "--mdocs",
            str(tmp_path / "mdocs"),
            "--out",
            str(tmp_path / "out"),
            "--tilt-series",
            "TS_TEST",
            "--device",
            "CPU",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].tilt_series == ["TS_TEST"]
    assert captured["config"].device == DevicePreference.CPU
