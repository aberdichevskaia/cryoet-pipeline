from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.template import PytomTemplateBackend, prepare_template_and_register
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference

_BOX_SIZE = 24
_VOXEL_SIZE = 9.504


def _blob(box_size: int, *, negative: bool) -> np.ndarray:
    axis = np.arange(box_size, dtype=np.float32) - (box_size - 1) / 2.0
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    density = np.exp(-(x**2 + y**2 + z**2) / (2.0 * (box_size / 8.0) ** 2)).astype(np.float32)
    return -density if negative else density


def _write_volume(path: Path, data: np.ndarray, voxel_size: float = _VOXEL_SIZE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size


def _source_map(tmp_path: Path) -> Path:
    path = tmp_path / "emd_2638.map"
    _write_volume(path, _blob(32, negative=False), voxel_size=1.53)
    return path


def _tomogram(tmp_path: Path) -> Artifact:
    path = tmp_path / "tomogram.zarr"
    zarr.save(path, np.zeros((4, 8, 8), dtype=np.float32))
    return Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        shape=(4, 8, 8),
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=_VOXEL_SIZE,
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.188,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frame.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.188,
                binning=1,
            )
        ],
    )


def _executables(tmp_path: Path) -> tuple[Path, Path]:
    create_template = tmp_path / "pytom_create_template.py"
    create_mask = tmp_path / "pytom_create_mask.py"
    create_template.touch()
    create_mask.touch()
    return create_template, create_mask


def _runner_factory(
    *,
    template_negative: bool = True,
    mask_box_size: int | None = None,
    captured: list[list[str]] | None = None,
):
    def runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env, stdin
        command_list = list(command)
        if captured is not None:
            captured.append(command_list)
        output = Path(command_list[command_list.index("-o") + 1])
        if "create_template" in command_list[0]:
            _write_volume(output, _blob(_BOX_SIZE, negative=template_negative))
        else:
            size = mask_box_size if mask_box_size is not None else int(
                command_list[command_list.index("-b") + 1]
            )
            _write_volume(output, np.ones((size, size, size), dtype=np.float32))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return runner


def test_missing_executable_names_the_option_it_came_from(tmp_path: Path) -> None:
    """An unset shell variable expands to a path that looks almost right.

    `$UNSET/pytom_create_template.py` arrives here as
    `/pytom_create_template.py`, and a message that only repeats the path leaves
    the reader hunting for which of two similar options carried it.
    """

    tomogram = _tomogram(tmp_path)
    _, create_mask = _executables(tmp_path)

    with pytest.raises(FileNotFoundError) as failure:
        PytomTemplateBackend(_runner_factory(captured=[])).prepare(
            _source_map(tmp_path),
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_create_template_executable": Path("/pytom_create_template.py"),
                    "pytom_create_mask_executable": create_mask,
                    "template_particle_diameter_angstrom": 150.0,
                    "template_emdb_id": "2638",
                },
            ),
        )

    message = str(failure.value)
    assert "/pytom_create_template.py" in message
    assert "--pytom-create-template-executable" in message


def test_prepare_template_draws_a_slab_mask_for_an_extended_reference(
    tmp_path: Path,
) -> None:
    """A lattice or membrane patch is thin along one axis; a sphere around it is
    mostly background.

    pytom reads -r as the x semi-axis and takes y and z separately, so the
    ellipsoid is expressed by giving the two extra radii.
    """

    tomogram = _tomogram(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    create_template, create_mask = _executables(tmp_path)
    captured: list[list[str]] = []

    prepare_template_and_register(
        PytomTemplateBackend(_runner_factory(captured=captured)),
        _source_map(tmp_path),
        tomogram,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "pytom_create_template_executable": create_template,
                "pytom_create_mask_executable": create_mask,
                "template_mask_radius_px": 12.0,
                "template_mask_radius_y_px": 12.0,
                "template_mask_radius_z_px": 3.0,
                "template_emdb_id": "2638",
            },
        ),
        registry,
    )

    mask_command = next(c for c in captured if "pytom_create_mask" in c[0])
    assert mask_command[mask_command.index("-r") + 1] == "12.000000"
    assert mask_command[mask_command.index("--radius-minor1") + 1] == "12.000000"
    assert mask_command[mask_command.index("--radius-minor2") + 1] == "3.000000"

    # Picking has to tell pytom that its rotation-invariant fast path does not
    # apply, and it can only know that if the shape was written down.
    mask = registry.get("TS_TEST:template_mask:pytom")
    assert mask.parameters["mask_is_spherical"] is False
    assert mask.parameters["mask_radius_z_px"] == pytest.approx(3.0)


def test_prepare_template_keeps_a_sphere_when_only_one_radius_is_given(
    tmp_path: Path,
) -> None:
    """The common case must not grow flags it does not need."""

    tomogram = _tomogram(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    create_template, create_mask = _executables(tmp_path)
    captured: list[list[str]] = []

    prepare_template_and_register(
        PytomTemplateBackend(_runner_factory(captured=captured)),
        _source_map(tmp_path),
        tomogram,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "pytom_create_template_executable": create_template,
                "pytom_create_mask_executable": create_mask,
                "template_mask_radius_px": 10.0,
                "template_emdb_id": "2638",
            },
        ),
        registry,
    )

    mask_command = next(c for c in captured if "pytom_create_mask" in c[0])
    assert "--radius-minor1" not in mask_command
    assert "--radius-minor2" not in mask_command
    assert registry.get("TS_TEST:template_mask:pytom").parameters["mask_is_spherical"] is True


def test_prepare_template_labels_its_outputs_for_a_lane(tmp_path: Path) -> None:
    """A lane that reaches picking needs its own template, not the other lane's.

    Everything downstream of reconstruction registers one artifact per stage, so
    without the label a second lane's template displaces the first while both
    registry entries keep pointing at one file.
    """

    tomogram = _tomogram(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    create_template, create_mask = _executables(tmp_path)

    template, mask, qc_artifact = prepare_template_and_register(
        PytomTemplateBackend(_runner_factory(captured=[])),
        _source_map(tmp_path),
        tomogram,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "pytom_create_template_executable": create_template,
                "pytom_create_mask_executable": create_mask,
                "template_particle_diameter_angstrom": 150.0,
                "template_emdb_id": "2638",
                "variant": "bin8",
            },
        ),
        registry,
    )

    assert template.id.endswith(":bin8")
    assert mask.id.endswith(":bin8")
    assert qc_artifact.id.endswith(":bin8")
    assert template.variant == "bin8"
    assert "bin8" in template.path.name, "two lanes must not share a template file"


def test_prepare_template_writes_template_mask_and_qc(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    create_template, create_mask = _executables(tmp_path)
    captured: list[list[str]] = []

    artifacts = prepare_template_and_register(
        PytomTemplateBackend(_runner_factory(captured=captured)),
        _source_map(tmp_path),
        tomogram,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "pytom_create_template_executable": create_template,
                "pytom_create_mask_executable": create_mask,
                "template_particle_diameter_angstrom": 150.0,
                "template_emdb_id": "2638",
            },
        ),
        registry,
    )

    template, mask, qc_artifact = artifacts
    assert registry.get(template.id) == template
    assert template.kind == ArtifactKind.TEMPLATE
    assert mask.kind == ArtifactKind.TEMPLATE_MASK
    assert mask.parent_ids == [template.id]
    assert mask.parameters["template_id"] == template.id
    assert template.parent_ids == [tomogram.id]
    assert template.pixel_spacing_angstrom == pytest.approx(_VOXEL_SIZE)
    assert template.shape == (_BOX_SIZE, _BOX_SIZE, _BOX_SIZE)
    assert template.parameters["emdb_id"] == "2638"
    assert template.path.is_file()
    assert mask.path.is_file()

    template_command = captured[0]
    assert template_command[template_command.index("--output-voxel-size-angstrom") + 1] == (
        f"{_VOXEL_SIZE:.6f}"
    )
    assert "--invert" in template_command

    mask_command = captured[1]
    # The mask must be built at the template's actual box size, not a guess.
    assert mask_command[mask_command.index("-b") + 1] == str(_BOX_SIZE)
    # 150 A diameter at 9.504 A/px is a radius of ~7.89 px.
    assert float(mask_command[mask_command.index("-r") + 1]) == pytest.approx(7.891, abs=1e-3)

    report = json.loads(qc_artifact.path.read_text())
    assert report["status"] == QcStatus.PASS.value
    assert report["inverted"] is True
    assert report["box_size_px"] == _BOX_SIZE
    assert report["emdb_id"] == "2638"


def test_prepare_template_warns_when_contrast_disagrees_with_invert(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)

    # invert=True is requested but the produced template is still bright.
    artifacts = PytomTemplateBackend(
        _runner_factory(template_negative=False)
    ).prepare(
        _source_map(tmp_path),
        tomogram,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CPU,
            parameters={
                "pytom_create_template_executable": create_template,
                "pytom_create_mask_executable": create_mask,
                "template_mask_radius_px": 8.0,
            },
        ),
    )

    report = json.loads(artifacts[2].path.read_text())
    assert report["status"] == QcStatus.WARNING.value
    assert any("dark (negative)" in warning for warning in report["warnings"])


def test_prepare_template_rejects_mask_box_size_mismatch(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)

    with pytest.raises(RuntimeError, match="mask box size 16 does not match"):
        PytomTemplateBackend(_runner_factory(mask_box_size=16)).prepare(
            _source_map(tmp_path),
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_create_template_executable": create_template,
                    "pytom_create_mask_executable": create_mask,
                    "template_mask_radius_px": 8.0,
                },
            ),
        )


def test_prepare_template_requires_a_way_to_size_the_mask(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)

    with pytest.raises(TypeError, match="template_mask_radius_px"):
        PytomTemplateBackend(_runner_factory()).prepare(
            _source_map(tmp_path),
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_create_template_executable": create_template,
                    "pytom_create_mask_executable": create_mask,
                },
            ),
        )


def test_prepare_template_rejects_a_mask_larger_than_the_box(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)

    with pytest.raises(ValueError) as failure:
        PytomTemplateBackend(_runner_factory()).prepare(
            _source_map(tmp_path),
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_create_template_executable": create_template,
                    "pytom_create_mask_executable": create_mask,
                    "template_mask_radius_px": 40.0,
                },
            ),
        )

    # The number to pass and the size actually available are what the
    # reader needs; the second also says whether the diameter is plausible.
    message = str(failure.value)
    assert "--template-box-size-px" in message
    assert "angstrom/pixel" in message


def test_prepare_template_rejects_a_tomogram_from_another_series(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path).model_copy(
        update={"parameters": {"tilt_series_id": "TS_OTHER", "tomogram_branch": "full"}}
    )
    create_template, create_mask = _executables(tmp_path)

    with pytest.raises(ValueError, match="tomogram is for TS_OTHER"):
        PytomTemplateBackend(_runner_factory()).prepare(
            _source_map(tmp_path),
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_create_template_executable": create_template,
                    "pytom_create_mask_executable": create_mask,
                    "template_mask_radius_px": 8.0,
                },
            ),
        )


def test_prepare_template_rejects_a_missing_source_map(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="reference map not found"):
        PytomTemplateBackend(_runner_factory()).prepare(
            tmp_path / "missing.map",
            _tomogram(tmp_path),
            _manifest(),
            BackendContext(output_dir=tmp_path / "outputs", device=DevicePreference.CPU),
        )
