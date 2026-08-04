from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.defocus_hint import (
    Ctffind5DefocusHintBackend,
    _parse_ctffind_summary,
    estimate_defocus_hint_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    DefocusHintQc,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_ctffind5_defocus_hint_writes_summary_and_qc(tmp_path: Path) -> None:
    projection = _corrected_projection_artifact(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(projection)
    executable = tmp_path / "ctffind5"
    executable.touch()
    captured: dict[str, object] = {}

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        captured["command"] = list(command)
        captured["stdin"] = stdin
        lines = (stdin or "").splitlines()
        with mrcfile.open(Path(lines[0]), permissive=True) as materialized:
            captured["materialized_input"] = np.array(materialized.data)
        diagnostic_path = Path(lines[2])
        summary_path = diagnostic_path.with_suffix(".txt")
        summary_path.write_text(
            "# columns: #1 micrograph, #2 DF1, #3 DF2, #4 azimuth, #5 phase, #6 CC, #7 fit\n"
            "1 29500.0 28500.0 12.5 0.0 0.85 4.2\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    context = BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={
            "ctffind_executable": executable,
            "ctf_voltage_kv": 300.0,
            "ctf_spherical_aberration_mm": 2.7,
        },
    )

    artifacts = estimate_defocus_hint_and_register(
        Ctffind5DefocusHintBackend(fake_runner),
        projection,
        manifest,
        context,
        registry,
    )

    hint, qc_artifact = artifacts
    assert registry.get(hint.id) == hint
    assert registry.get(qc_artifact.id) == qc_artifact
    assert hint.kind == ArtifactKind.DEFOCUS_HINT
    assert hint.parent_ids == [projection.id]
    assert hint.parameters["defocus_um"] == pytest.approx(2.9)
    assert hint.parameters["astigmatism_um"] == pytest.approx(0.05)
    assert hint.parameters["tilt_angle_deg"] == pytest.approx(0.1)

    command = captured["command"]
    assert isinstance(command, list)
    assert command == [str(executable)]
    stdin_text = captured["stdin"]
    assert isinstance(stdin_text, str)
    stdin_lines = stdin_text.splitlines()
    np.testing.assert_allclose(
        captured["materialized_input"],
        np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    assert stdin_lines[1] == "no"
    assert stdin_lines[3] == "1.350000"
    assert stdin_lines[4] == "300.000000"
    assert stdin_lines[5] == "2.700000"
    assert stdin_lines[-1] == "no"
    assert len(stdin_lines) == 20

    report = DefocusHintQc.model_validate_json(qc_artifact.path.read_text())
    assert report.status == QcStatus.PASS
    assert report.defocus_um == pytest.approx(2.9)
    assert report.astigmatism_um == pytest.approx(0.05)
    assert report.azimuth_deg == pytest.approx(12.5)
    assert report.cross_correlation == pytest.approx(0.85)
    assert report.fit_resolution_angstrom == pytest.approx(4.2)


def test_ctffind5_defocus_hint_requires_voltage(tmp_path: Path) -> None:
    projection = _corrected_projection_artifact(tmp_path)
    executable = tmp_path / "ctffind5"
    executable.touch()

    with pytest.raises(TypeError, match="ctf_voltage_kv"):
        Ctffind5DefocusHintBackend().estimate(
            projection,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "ctffind_executable": executable,
                    "ctf_spherical_aberration_mm": 2.7,
                },
            ),
        )


def test_parse_ctffind_summary(tmp_path: Path) -> None:
    path = tmp_path / "summary.txt"
    path.write_text("# header comment\n1 29500.0 28500.0 12.5 0.0 0.85 4.2\n")

    fit = _parse_ctffind_summary(path)

    assert fit.defocus1_angstrom == pytest.approx(29500.0)
    assert fit.defocus2_angstrom == pytest.approx(28500.0)
    assert fit.azimuth_deg == pytest.approx(12.5)
    assert fit.cross_correlation == pytest.approx(0.85)
    assert fit.fit_resolution_angstrom == pytest.approx(4.2)


def _corrected_projection_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "projection.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.arange(12, dtype=np.float32).reshape(3, 4))
        mrc.voxel_size = 1.35
    return Artifact(
        id="TS_TEST:corrected_projection:000",
        kind=ArtifactKind.CORRECTED_PROJECTION,
        path=path,
        shape=(3, 4),
        dtype="float32",
        axis_order=AxisOrder.YX,
        pixel_spacing_angstrom=1.35,
        parameters={
            "z_value": 0,
            "tilt_angle_deg": 0.1,
        },
    )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.1,
                subframe_path="frame.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            )
        ],
    )
