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
from cryoet_pipeline.backends.template import (
    PytomTemplateBackend,
    _density_extent_px,
    _mask_radii_px,
    _mask_versus_density_warnings,
    _MaskRadii,
    _VolumeStatistics,
    prepare_template_and_register,
)
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
    template_data: np.ndarray | None = None,
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
            _write_volume(
                output,
                _blob(_BOX_SIZE, negative=template_negative)
                if template_data is None
                else template_data,
            )
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


def test_prepare_template_derives_the_mask_size_when_none_is_given(tmp_path: Path) -> None:
    """A size that was not given is measured off the map rather than refused.

    Safe to do because asking for neither used to raise, so no command line that
    already works changes behaviour. It has to be recorded as derived: a number
    the pipeline chose and a number the user chose deserve different trust.
    """

    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)

    artifacts = PytomTemplateBackend(_runner_factory()).prepare(
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

    qc = next(artifact for artifact in artifacts if artifact.kind is ArtifactKind.QC)
    report = json.loads(Path(qc.path).read_text())
    assert report["mask_radius_derived_from_density"] is True
    assert report["mask_radius_px"] > 0.0
    assert report["status"] == QcStatus.WARNING
    assert any("derived from the map itself" in warning for warning in report["warnings"])


def test_prepare_template_refuses_to_guess_a_size_for_a_map_that_fills_its_box(
    tmp_path: Path,
) -> None:
    """The EMD-2638 case: nothing about a lattice says where one particle ends."""

    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)
    axis = np.arange(_BOX_SIZE, dtype=np.float32) - (_BOX_SIZE - 1) / 2.0
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    lattice = -(1.0 + np.cos(z * 2.0 * np.pi / 6.0)).astype(np.float32) * (
        1.0 + np.cos(x * 2.0 * np.pi / 6.0)
    ).astype(np.float32)

    with pytest.raises(ValueError, match="no particle size to derive"):
        PytomTemplateBackend(_runner_factory(template_data=lattice)).prepare(
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


def test_prepare_template_still_refuses_when_there_is_no_density_at_all(
    tmp_path: Path,
) -> None:
    """A flat map has nothing to measure, so the old error is still the answer."""

    tomogram = _tomogram(tmp_path)
    create_template, create_mask = _executables(tmp_path)
    flat = np.zeros((_BOX_SIZE, _BOX_SIZE, _BOX_SIZE), dtype=np.float32)

    with pytest.raises(TypeError, match="template_mask_radius_px"):
        PytomTemplateBackend(_runner_factory(template_data=flat)).prepare(
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


def _axis_grid(box_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.arange(box_size, dtype=np.float32) - (box_size - 1) / 2.0
    return np.meshgrid(axis, axis, axis, indexing="ij")


def _measure(data: np.ndarray) -> tuple[float, float, float]:
    return _density_extent_px(data.astype(np.float32), median=float(np.median(data)))


def test_density_extent_reports_each_axis_separately() -> None:
    """An elongated map must not be described by one number.

    EMD-2638 is the case this exists for: a helical reconstruction whose density
    repeats along z, so a single diameter hides the one property that decides
    whether a spherical mask is right.
    """

    z, y, x = _axis_grid(64)
    ellipsoid = ((x / 16.0) ** 2 + (y / 8.0) ** 2 + (z / 4.0) ** 2 < 1.0).astype(np.float32)

    extent_x, extent_y, extent_z = _measure(ellipsoid)

    assert extent_x > extent_y > extent_z
    # Semi-axes 16/8/4 stand in ratio 4:2:1, and the measurement keeps it.
    assert extent_x / extent_z == pytest.approx(4.0, rel=0.15)
    assert extent_y / extent_z == pytest.approx(2.0, rel=0.15)


def test_density_extent_survives_noise() -> None:
    """Background noise must not read as density.

    Mass spread thinly over the whole box is still mass, so without removing the
    background every axis reads as the full box whatever the particle looks like.

    The noise here is heavy on purpose. At 15% every candidate measure survived,
    so a gentler test pins nothing: reading the width at a twentieth of the peak
    is exact on clean shapes and reaches the whole 64 px box at 30%, while the
    width holding 95% of the mass drifts to 31 px against a true 24.
    """

    z, y, x = _axis_grid(64)
    sphere = (x**2 + y**2 + z**2 < 12.0**2).astype(np.float32)
    noisy = sphere + np.random.default_rng(0).normal(0.0, 0.30, sphere.shape).astype(np.float32)

    clean_extent = _measure(sphere)[0]
    noisy_extent = _measure(noisy)[0]

    assert noisy_extent == pytest.approx(clean_extent, abs=3.0)
    assert noisy_extent < 28.0, "noise was counted as density and swamped the box"


def test_density_extent_reads_a_periodic_map_as_filling_the_box() -> None:
    """A lattice really does fill its box, and must be reported that way."""

    z, y, x = _axis_grid(64)
    tube = (x**2 + y**2 < 10.0**2).astype(np.float32) * (
        1.0 + np.cos(z * 2.0 * np.pi / 8.0)
    ).astype(np.float32)

    extent_x, extent_y, extent_z = _measure(tube)

    assert extent_z == pytest.approx(64.0, abs=1.0)
    assert extent_x < 24.0 and extent_y < 24.0


def test_density_extent_is_unchanged_by_inversion() -> None:
    """The template is inverted to match dark particles; the extent is the same."""

    z, y, x = _axis_grid(64)
    sphere = (x**2 + y**2 + z**2 < 12.0**2).astype(np.float32)

    assert _measure(1.0 - sphere) == pytest.approx(_measure(sphere))


def test_density_extent_measures_size_not_distance_from_the_centre() -> None:
    """An off-centre particle has its own size, not its offset."""

    z, y, x = _axis_grid(64)
    centred = (x**2 + y**2 + z**2 < 8.0**2).astype(np.float32)
    shifted = ((x - 15.0) ** 2 + y**2 + z**2 < 8.0**2).astype(np.float32)

    assert _measure(shifted) == pytest.approx(_measure(centred))


def test_mask_much_smaller_than_the_density_warns() -> None:
    statistics = _VolumeStatistics(
        box_size=64,
        minimum=0.0,
        maximum=1.0,
        median=0.0,
        voxel_size_angstrom=10.0,
        density_extent_px=(40.0, 40.0, 40.0),
    )

    warnings = _mask_versus_density_warnings(
        _MaskRadii(x=5.0, y=5.0, z=5.0), statistics, voxel_size_angstrom=10.0
    )

    assert any("cuts the particle" in warning for warning in warnings)


def test_mask_matching_the_density_is_quiet() -> None:
    statistics = _VolumeStatistics(
        box_size=64,
        minimum=0.0,
        maximum=1.0,
        median=0.0,
        voxel_size_angstrom=10.0,
        density_extent_px=(20.0, 20.0, 20.0),
    )

    warnings = _mask_versus_density_warnings(
        _MaskRadii(x=11.0, y=11.0, z=11.0), statistics, voxel_size_angstrom=10.0
    )

    assert warnings == []


def test_spherical_mask_on_anisotropic_density_warns() -> None:
    """The Gag case: a round mask over a map that is not round."""

    statistics = _VolumeStatistics(
        box_size=64,
        minimum=0.0,
        maximum=1.0,
        median=0.0,
        voxel_size_angstrom=10.0,
        density_extent_px=(18.0, 18.0, 64.0),
    )

    warnings = _mask_versus_density_warnings(
        _MaskRadii(x=10.0, y=10.0, z=10.0), statistics, voxel_size_angstrom=10.0
    )

    assert any("not isotropic" in warning for warning in warnings)
    assert any("--template-mask-radius-z-px" in warning for warning in warnings)


def _fake_pytom_scripts(tmp_path: Path) -> tuple[Path, Path]:
    """Stand-ins for pytom's helpers that write a compact blob and a cube mask."""

    import sys

    body = (
        "#!" + sys.executable + "\n"
        "import sys, numpy as np, mrcfile\n"
        "a = sys.argv\n"
        'out = a[a.index("-o") + 1]\n'
        'if "create_template" in a[0]:\n'
        "    n = 24; ax = np.arange(n) - (n - 1) / 2.0\n"
        '    z, y, x = np.meshgrid(ax, ax, ax, indexing="ij")\n'
        '    d = -np.exp(-(x**2 + y**2 + z**2) / (2 * (n / 8.0) ** 2)).astype("float32")\n'
        "else:\n"
        '    n = int(a[a.index("-b") + 1]); d = np.ones((n, n, n), dtype="float32")\n'
        "with mrcfile.new(out, overwrite=True) as m:\n"
        '    m.set_data(d.astype("float32")); m.voxel_size = ' + repr(_VOXEL_SIZE) + "\n"
    )
    paths = []
    for name in ("pytom_create_template.py", "pytom_create_mask.py"):
        path = tmp_path / name
        path.write_text(body)
        path.chmod(0o755)
        paths.append(path)
    return paths[0], paths[1]


def test_prepare_template_prints_the_size_it_measured(tmp_path: Path) -> None:
    """The derived size has to reach the terminal, not only the report.

    A run that ends in "QC status: warning" and leaves the reason in a JSON file
    tells the reader that something was noticed and not what, and a size the
    pipeline chose for itself is exactly what wants checking by eye.
    """

    from typer.testing import CliRunner

    from cryoet_pipeline import cli

    tomogram = _tomogram(tmp_path)
    registry_path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry()
    registry.add(tomogram)
    registry.write(registry_path)
    manifest_path = tmp_path / "TS_TEST.json"
    manifest_path.write_text(_manifest().model_dump_json())
    create_template, create_mask = _fake_pytom_scripts(tmp_path)

    result = CliRunner().invoke(
        cli.app,
        [
            "prepare-template",
            "--manifest", str(manifest_path),
            "--registry", str(registry_path),
            "--out", str(tmp_path / "out"),
            "--source-map", str(_source_map(tmp_path)),
            "--pytom-create-template-executable", str(create_template),
            "--pytom-create-mask-executable", str(create_mask),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "density extent:" in result.output
    assert "derived from the map, no size was given" in result.output
    # The warning text itself, not just the status line above it.
    assert "  warning: no mask size was given" in result.output


def test_density_extent_reports_a_full_box_when_the_box_is_full() -> None:
    """A lattice fills its box, and the measurement has to say so.

    This pins the property, not the bug that motivated it. The real failure was
    a noise level estimated over the whole volume — a median absolute deviation,
    which assumes most of the box is background and so measures the signal on a
    lattice: on EMD-2638 it read 6 px of a 28 px box along one axis against 24
    for the current code. No synthetic here reproduces that gap; the closest
    manages 25 against 27. The reproducer is that map, and it cannot ship as a
    fixture, so the note in `_density_extent_px` carries the finding instead.
    """

    z, y, x = _axis_grid(64)
    # Density everywhere, varying in strength: no background to estimate from.
    lattice = (
        (2.0 + np.cos(x * 2.0 * np.pi / 8.0))
        * (2.0 + np.cos(y * 2.0 * np.pi / 8.0))
        * (2.0 + np.cos(z * 2.0 * np.pi / 8.0))
    ).astype(np.float32)

    extents = _measure(lattice)

    assert min(extents) > 48.0, f"the box is full of density but read as {extents}"


def test_derived_mask_overhangs_the_particle_as_the_documentation_advises() -> None:
    """pytom-match-pick asks for a mask 10-20% wider than the particle radius.

    Checked against a sphere whose true radius is known, because the chain from
    measurement to margin has two chances to drift: the measurement reads a hard
    edge slightly under, and the margin is applied on top of that. What matters
    is where the pair lands, not either number alone.
    """

    box, true_radius = 64, 12.0
    z, y, x = _axis_grid(box)
    sphere = (x**2 + y**2 + z**2 < true_radius**2).astype(np.float32)

    context = BackendContext(
        output_dir=Path("."),
        device=DevicePreference.CPU,
        parameters={},
    )
    radii, derived = _mask_radii_px(
        context,
        voxel_size_angstrom=1.0,
        box_size=box,
        measured_extent_px=_measure(sphere),
    )

    assert derived is True
    overhang = radii.x / true_radius
    assert 1.0 < overhang < 1.25, f"mask radius {radii.x} is {overhang:.2f}x the particle's"
