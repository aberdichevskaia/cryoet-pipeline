from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artiax_export import export_picks_for_artiax
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder

_VOXEL_SIZE = 9.504


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


def _picks(tmp_path: Path, tomogram: Artifact) -> Artifact:
    path = tmp_path / "picks.star"
    path.write_text("data_particles\n\nloop_\n_rlnCoordinateX #1\n1\n")
    return Artifact(
        id="TS_TEST:picks:full:pytom-match-pick",
        kind=ArtifactKind.PICKS,
        path=path,
        parent_ids=[tomogram.id],
        parameters={"tilt_series_id": "TS_TEST", "input_tomogram_id": tomogram.id},
    )


def test_export_writes_paired_files_and_a_chimerax_script(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)

    export = export_picks_for_artiax(tomogram=tomogram, picks=picks, out=tmp_path / "outputs")

    assert export.tomogram_path.is_file()
    assert export.picks_path.read_text() == picks.path.read_text()

    with mrcfile.open(export.tomogram_path, permissive=True) as mrc:
        np.testing.assert_allclose(
            mrc.data, np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        )
        assert float(mrc.voxel_size.x) == pytest.approx(_VOXEL_SIZE, abs=1e-3)

    script = export.script_path.read_text()
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

    export = export_picks_for_artiax(
        tomogram=tomogram,
        picks=picks,
        out=tmp_path / "outputs",
        template=template,
        segmentation=segmentation,
    )

    assert export.template_path is not None and export.template_path.is_file()
    assert export.segmentation_path is not None and export.segmentation_path.is_file()
    with mrcfile.open(export.template_path, permissive=True) as mrc:
        assert float(mrc.voxel_size.x) == pytest.approx(_VOXEL_SIZE, abs=1e-3)

    script = export.script_path.read_text()
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
    tomogram = _tomogram(tmp_path)
    export = export_picks_for_artiax(
        tomogram=tomogram,
        picks=_picks(tmp_path, tomogram),
        out=tmp_path / "outputs",
    )

    assert export.template_path is None
    assert export.segmentation_path is None
    script = export.script_path.read_text()
    assert "artiax attach" not in script


def test_export_refuses_to_clobber_without_overwrite(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)
    out = tmp_path / "outputs"

    export_picks_for_artiax(tomogram=tomogram, picks=picks, out=out)
    with pytest.raises(FileExistsError, match="ArtiaX export already exists"):
        export_picks_for_artiax(tomogram=tomogram, picks=picks, out=out)

    export = export_picks_for_artiax(tomogram=tomogram, picks=picks, out=out, overwrite=True)
    assert export.script_path.is_file()


def test_export_requires_a_voxel_size(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path).model_copy(update={"pixel_spacing_angstrom": None})
    picks = _picks(tmp_path, tomogram)

    with pytest.raises(ValueError, match="voxel spacing is required"):
        export_picks_for_artiax(tomogram=tomogram, picks=picks, out=tmp_path / "outputs")


def test_export_rejects_wrong_artifact_kinds(tmp_path: Path) -> None:
    tomogram = _tomogram(tmp_path)
    picks = _picks(tmp_path, tomogram)

    with pytest.raises(ValueError, match="expected a picks artifact"):
        export_picks_for_artiax(tomogram=tomogram, picks=tomogram, out=tmp_path / "outputs")

    with pytest.raises(ValueError, match="expected a tomogram artifact"):
        export_picks_for_artiax(tomogram=picks, picks=picks, out=tmp_path / "outputs")
