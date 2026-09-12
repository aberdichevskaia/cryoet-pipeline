from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder
from cryoet_pipeline.picks_visualization import (
    resolve_sphere_radius,
    write_picks_visualization,
)
from cryoet_pipeline.runtime import DevicePreference

_VOXEL_SIZE = 9.504
_RADIUS = 7


def _tomogram(tmp_path: Path) -> Artifact:
    path = tmp_path / "tomogram.zarr"
    zarr.save(path, np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4))
    return Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        shape=(2, 3, 4),
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=_VOXEL_SIZE,
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )


def _picks(tmp_path: Path, tomogram: Artifact, *, with_scores: bool = True) -> Artifact:
    path = tmp_path / "picks.star"
    columns = ["_rlnCoordinateX #1", "_rlnCoordinateY #2", "_rlnCoordinateZ #3"]
    rows = ["10.000000 20.000000 30.000000", "1.500000 2.500000 3.500000"]
    if with_scores:
        columns.append("_rlnLCCmax #4")
        rows = [
            f"{row} {score}"
            for row, score in zip(rows, ("0.812000", "0.640000"), strict=True)
        ]
    path.write_text("data_particles\n\nloop_\n" + "\n".join(columns + rows) + "\n")
    return Artifact(
        id="TS_TEST:picks:full:pytom-match-pick",
        kind=ArtifactKind.PICKS,
        path=path,
        parent_ids=[tomogram.id],
        parameters={"tilt_series_id": "TS_TEST", "input_tomogram_id": tomogram.id},
    )


def _context(tmp_path: Path) -> BackendContext:
    executable = tmp_path / "point2model"
    executable.touch()
    return BackendContext(
        output_dir=tmp_path / "outputs",
        device=DevicePreference.CPU,
        parameters={"point2model_executable": executable},
    )


def _runner(
    captured: list[list[str]],
    *,
    returncode: int = 0,
    write_model: bool = True,
    cwds: list[Path] | None = None,
) -> object:
    """A point2model that records its argv and writes the file it promised.

    It resolves the output name against `cwd`, exactly as the real process
    does, so a command that names files the working directory does not contain
    fails here too rather than passing on a path the test happened to spell in
    full.
    """

    def run(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert env["IMOD_DIR"]
        command_list = [str(token) for token in command]
        captured.append(command_list)
        if cwds is not None:
            cwds.append(cwd)
        for option in ("-input", "-image"):
            named = cwd / command_list[command_list.index(option) + 1]
            assert named.is_file(), f"{option} names a file that is not in cwd: {named}"
        if write_model and returncode == 0:
            (cwd / command_list[command_list.index("-output") + 1]).write_text("IMOD model")
        return subprocess.CompletedProcess(
            command_list, returncode, stdout="Model created", stderr=""
        )

    return run


def _visualize(tmp_path: Path, **overrides: object) -> object:
    tomogram = overrides.pop("tomogram", None) or _tomogram(tmp_path)
    picks = overrides.pop("picks", None) or _picks(tmp_path, tomogram)
    captured = overrides.pop("captured", None)
    if captured is None:
        captured = []
    cwds = overrides.pop("cwds", None)
    return write_picks_visualization(
        tomogram=tomogram,
        picks=picks,
        out=overrides.pop("out", None) or tmp_path / "outputs",
        context=_context(tmp_path),
        sphere_radius=resolve_sphere_radius(
            requested_px=_RADIUS, template_mask=None, voxel_size_angstrom=_VOXEL_SIZE
        ),
        command_runner=_runner(captured, cwds=cwds),
        **overrides,
    )


def test_export_writes_paired_files_and_a_chimerax_script(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)

    export = _visualize(tmp_path, tomogram=tomogram, picks=picks)

    assert export.tomogram_path.is_file()
    assert export.picks_path.read_text() == picks.path.read_text()

    with mrcfile.open(export.tomogram_path, permissive=True) as mrc:
        np.testing.assert_allclose(
            mrc.data, np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        )
        assert float(mrc.voxel_size.x) == pytest.approx(_VOXEL_SIZE, abs=1e-3)

    script = export.chimerax_script_path.read_text()
    # Through ArtiaX, not as a plain volume: otherwise the tomogram is not under
    # its control and never appears in the Tomograms list.
    assert f"artiax open tomo {export.tomogram_path.name}" in script
    # ArtiaX registers both a RELION 3.1 and a RELION5 reader for .star, so the
    # format has to be pinned or ChimeraX may pick the wrong coordinate convention.
    assert f"open {export.picks_path.name} format relion\n" in script
    assert "format relion5" not in script
    # Coordinates are in tomogram voxels and ArtiaX scales them by a factor that
    # defaults to one, so without this the picks collapse into a corner.
    assert f"originScaleFactor {_VOXEL_SIZE:.6f}" in script


def test_both_viewers_read_one_tomogram_conversion(tmp_path: Path) -> None:
    """The tomogram is the largest file in the project, and it is written once.

    The 3dmod half has to point at the same MRC as the ChimeraX half, not at the
    IMOD `.rec` the reconstruction leaves behind: that copy is in a different
    axis order, and the picks are indices into this one.
    """

    captured: list[list[str]] = []
    export = _visualize(tmp_path, captured=captured)

    assert export.imod_model_path.parent == export.tomogram_path.parent
    image_argument = captured[0][captured[0].index("-image") + 1]
    assert image_argument == export.tomogram_path.name


def test_point2model_is_told_names_and_run_where_they_are(tmp_path: Path) -> None:
    """Bare names plus the directory as cwd, not paths plus the directory as cwd.

    `--out` is allowed to be relative — the README's own example passes
    `--out outputs/dev` — and a relative path in the argv would be resolved a
    second time against a cwd that already ends in the same directory, so
    point2model would fail to open a file whose path has that tail twice.
    """

    captured: list[list[str]] = []
    cwds: list[Path] = []
    export = _visualize(tmp_path, captured=captured, cwds=cwds)

    assert cwds == [export.directory]
    command = captured[0]
    for option in ("-input", "-output", "-image"):
        argument = command[command.index(option) + 1]
        assert "/" not in argument, f"{option} must name a file in cwd, got {argument}"


def test_picks_reach_the_imod_model_at_the_voxels_they_name(tmp_path: Path) -> None:
    """Half a voxel in X and Y, none in Z.

    IMOD model coordinates run 0..NX in X and Y but -0.5..NZ-0.5 in Z, so the
    centre of voxel i is at i + 0.5 in X and Y and at i in Z, while the picks
    are voxel indices in all three. Asserted on the numbers because the error it
    guards against is half a voxel: invisible in a screenshot, and wrong.
    """

    export = _visualize(tmp_path)

    lines = export.imod_points_path.read_text().splitlines()
    assert [line.split()[:3] for line in lines] == [
        ["10.500", "20.500", "30.000"],
        ["2.000", "3.000", "3.500"],
    ]
    assert export.pick_count == 2


def test_scores_ride_along_so_a_weak_pick_looks_different(tmp_path: Path) -> None:
    captured: list[list[str]] = []
    export = _visualize(tmp_path, captured=captured)

    assert [line.split()[3] for line in export.imod_points_path.read_text().splitlines()] == [
        "0.812000",
        "0.640000",
    ]
    command = captured[0]
    # `-values 1` reads the last column as a per-point value; flag 1 displays it
    # and flag 2 colours by it.
    assert command[command.index("-values") + 1] == "1"
    assert command[command.index("-flags") + 1] == "3"


def test_picks_without_scores_still_convert(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    captured: list[list[str]] = []
    export = _visualize(
        tmp_path,
        tomogram=tomogram,
        picks=_picks(tmp_path, tomogram, with_scores=False),
        captured=captured,
    )

    assert [line.split() for line in export.imod_points_path.read_text().splitlines()] == [
        ["10.500", "20.500", "30.000"],
        ["2.000", "3.000", "3.500"],
    ]
    # Values that are not there cannot be displayed, and asking for them would
    # make point2model read a coordinate as a value.
    assert "-values" not in captured[0]
    assert "-flags" not in captured[0]


@pytest.mark.parametrize("with_scores", [True, False])
def test_the_point_file_and_the_values_flag_cannot_disagree(
    tmp_path: Path, with_scores: bool
) -> None:
    """A four-number line means two different things to point2model.

    With `-values 1` the last number is the point's value. Without it, the FIRST
    number is a contour number and the rest are the coordinates — so a file
    written with scores and a command run without them does not fail, it scatters
    the picks across as many contours as there were distinct scores. The two are
    derived from one decision so the mismatch cannot be reached.
    """

    tomogram = _tomogram(tmp_path)
    captured: list[list[str]] = []
    export = _visualize(
        tmp_path,
        tomogram=tomogram,
        picks=_picks(tmp_path, tomogram, with_scores=with_scores),
        captured=captured,
    )

    widths = {len(line.split()) for line in export.imod_points_path.read_text().splitlines()}
    assert widths == {4 if with_scores else 3}
    assert ("-values" in captured[0]) is with_scores


def test_the_model_is_a_scattered_sphere_object(tmp_path: Path) -> None:
    """Without -scat the points become one contour and 3dmod draws the polygon.

    Without a sphere radius a point is one pixel, and a pick one slice away from
    the one being viewed shows nothing at all.
    """

    captured: list[list[str]] = []
    _visualize(tmp_path, captured=captured)

    command = captured[0]
    assert "-scat" in command
    assert command[command.index("-sphere") + 1] == str(_RADIUS)


def test_a_failing_point2model_is_reported_with_its_log(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)
    captured: list[list[str]] = []

    with pytest.raises(RuntimeError, match="point2model failed with exit code 1"):
        write_picks_visualization(
            tomogram=tomogram,
            picks=picks,
            out=tmp_path / "outputs",
            context=_context(tmp_path),
            sphere_radius=resolve_sphere_radius(
                requested_px=_RADIUS, template_mask=None, voxel_size_angstrom=_VOXEL_SIZE
            ),
            command_runner=_runner(captured, returncode=1, write_model=False),
        )

    log = tmp_path / "outputs" / "picks_view" / "TS_TEST" / "TS_TEST_point2model.log"
    assert log.is_file()
    assert "point2model" in log.read_text()


def test_a_silent_point2model_is_not_reported_as_success(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    captured: list[list[str]] = []

    with pytest.raises(RuntimeError, match="exited 0 but wrote no model"):
        write_picks_visualization(
            tomogram=tomogram,
            picks=_picks(tmp_path, tomogram),
            out=tmp_path / "outputs",
            context=_context(tmp_path),
            sphere_radius=resolve_sphere_radius(
                requested_px=_RADIUS, template_mask=None, voxel_size_angstrom=_VOXEL_SIZE
            ),
            command_runner=_runner(captured, write_model=False),
        )


def test_a_missing_point2model_fails_before_anything_is_written(tmp_path: Path) -> None:
    """The conversion is the step that can fail for reasons outside the project.

    Failing after the tomogram was converted would leave a directory holding
    everything but the model, which looks finished.
    """

    tomogram = _tomogram(tmp_path)
    out = tmp_path / "outputs"

    with pytest.raises(FileNotFoundError, match="point2model"):
        write_picks_visualization(
            tomogram=tomogram,
            picks=_picks(tmp_path, tomogram),
            out=out,
            context=BackendContext(
                output_dir=out,
                device=DevicePreference.CPU,
                parameters={"point2model_executable": tmp_path / "absent"},
            ),
            sphere_radius=resolve_sphere_radius(
                requested_px=_RADIUS, template_mask=None, voxel_size_angstrom=_VOXEL_SIZE
            ),
        )

    assert not (out / "picks_view").exists()


def test_export_carries_the_template_and_segmentation(tmp_path: Path) -> None:
    """Points alone answer little; the template shows orientation, the
    segmentation shows whether the picks sit on anything.

    Both live as Zarr in the pipeline and ChimeraX reads MRC, so the conversion
    has to happen here or the files are unopenable.
    """

    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)
    template_path = tmp_path / "template.zarr"
    template_values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    zarr.save(template_path, template_values)
    template = Artifact(
        id="TS_TEST:template:pytom",
        kind=ArtifactKind.TEMPLATE,
        path=template_path,
        parameters={"tilt_series_id": "TS_TEST"},
    )
    segmentation_path = tmp_path / "segmentation.zarr"
    zarr.save(segmentation_path, np.zeros((2, 3, 4), dtype=np.float32))
    segmentation = Artifact(
        id="TS_TEST:segmentation:full:membrain-seg",
        kind=ArtifactKind.SEGMENTATION,
        path=segmentation_path,
        parameters={"tilt_series_id": "TS_TEST"},
    )

    export = _visualize(
        tmp_path,
        tomogram=tomogram,
        picks=picks,
        template=template,
        segmentation=segmentation,
    )

    assert export.template_path is not None and export.template_path.is_file()
    assert export.segmentation_path is not None and export.segmentation_path.is_file()
    with mrcfile.open(export.template_path, permissive=True) as mrc:
        assert float(mrc.voxel_size.x) == pytest.approx(_VOXEL_SIZE, abs=1e-3)

    script = export.chimerax_script_path.read_text()
    assert "artiax attach #1 toParticleList #3.2.1" in script
    # attach re-opens the file beneath the particle list, so naming the model it
    # was opened under afterwards fails with "No volumes specified" and aborts
    # the rest of the script.
    # The level is set on the model that is about to be attached, because
    # attaching copies the surface as it stands.
    contour_line = next(
        line for line in script.splitlines() if line.startswith("volume #1 level")
    )
    attach_line = next(line for line in script.splitlines() if line.startswith("artiax attach"))
    assert script.index(contour_line) < script.index(attach_line)
    # Attaching alone leaves the surface hidden while the list keeps drawing its
    # own markers, so a level change appears to do nothing.
    assert "artiax show #3.2.1 surfaces" in script
    assert "artiax hide #3.2.1 markers" in script
    # The template is inverted to match dark particles, so the level has to sit
    # below its mean; ChimeraX's own default is above it, on the wrong side.
    contour = contour_line
    level = float(contour.split("level ")[1].split()[0])
    assert level < float(template_values.mean())
    assert level == pytest.approx(
        float(template_values.mean() - 2.0 * template_values.std()), abs=1e-3
    )
    assert f"open {export.segmentation_path.name}" in script
    assert "volume #2 level 0.5" in script


def test_export_without_a_template_or_segmentation_says_nothing_about_them(
    tmp_path: Path,
) -> None:
    export = _visualize(tmp_path)

    assert export.template_path is None
    assert export.segmentation_path is None
    assert "artiax attach" not in export.chimerax_script_path.read_text()


def test_export_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)
    out = tmp_path / "outputs"

    _visualize(tmp_path, tomogram=tomogram, picks=picks, out=out)
    with pytest.raises(FileExistsError, match="picks visualization already exists"):
        _visualize(tmp_path, tomogram=tomogram, picks=picks, out=out)

    export = _visualize(tmp_path, tomogram=tomogram, picks=picks, out=out, overwrite=True)
    assert export.chimerax_script_path.is_file()
    assert export.imod_model_path.is_file()


def test_export_requires_a_voxel_size(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path).model_copy(update={"pixel_spacing_angstrom": None})

    with pytest.raises(ValueError, match="voxel spacing is required"):
        _visualize(tmp_path, tomogram=tomogram, picks=_picks(tmp_path, tomogram))


def test_export_rejects_wrong_artifact_kinds(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)

    with pytest.raises(ValueError, match="expected a picks artifact"):
        _visualize(tmp_path, tomogram=tomogram, picks=tomogram)

    with pytest.raises(ValueError, match="expected a tomogram artifact"):
        _visualize(tmp_path, tomogram=picks, picks=picks)


def test_export_rejects_picks_without_coordinates(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    path = tmp_path / "no_coordinates.star"
    path.write_text("data_particles\n\nloop_\n_rlnLCCmax #1\n0.5\n")
    picks = Artifact(
        id="TS_TEST:picks:full:pytom-match-pick",
        kind=ArtifactKind.PICKS,
        path=path,
        parameters={"tilt_series_id": "TS_TEST"},
    )

    with pytest.raises(ValueError, match="missing required STAR columns"):
        _visualize(tmp_path, tomogram=tomogram, picks=picks)


def test_sphere_radius_comes_from_the_mask_the_search_used(tmp_path: Path) -> None:
    """The template mask radius is the size the picks were looked for at.

    Drawing them at that size is what makes "the circle is bigger than anything
    under it" a readable answer rather than a property of the viewer.
    """

    mask = Artifact(
        id="TS_TEST:template_mask:pytom",
        kind=ArtifactKind.TEMPLATE_MASK,
        path=tmp_path / "mask.zarr",
        pixel_spacing_angstrom=_VOXEL_SIZE,
        parameters={"mask_radius_px": 11.6},
    )

    radius = resolve_sphere_radius(
        requested_px=None, template_mask=mask, voxel_size_angstrom=_VOXEL_SIZE
    )

    assert radius.px == 12
    assert "template mask radius" in radius.source


def test_sphere_radius_is_rescaled_when_the_mask_is_at_another_voxel_size(
    tmp_path: Path,
) -> None:
    mask = Artifact(
        id="TS_TEST:template_mask:pytom",
        kind=ArtifactKind.TEMPLATE_MASK,
        path=tmp_path / "mask.zarr",
        pixel_spacing_angstrom=2.0 * _VOXEL_SIZE,
        parameters={"mask_radius_px": 6.0},
    )

    radius = resolve_sphere_radius(
        requested_px=None, template_mask=mask, voxel_size_angstrom=_VOXEL_SIZE
    )

    assert radius.px == 12
    assert "rescaled" in radius.source


def test_sphere_radius_says_when_it_is_only_a_default() -> None:
    radius = resolve_sphere_radius(
        requested_px=None, template_mask=None, voxel_size_angstrom=_VOXEL_SIZE
    )

    assert radius.px >= 1
    assert "default" in radius.source

    asked = resolve_sphere_radius(
        requested_px=3, template_mask=None, voxel_size_angstrom=_VOXEL_SIZE
    )
    assert asked.px == 3
    assert asked.source == "as given"
