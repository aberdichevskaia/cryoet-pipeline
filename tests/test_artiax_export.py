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
    assert f"open {export.tomogram_path.name}" in script
    # ArtiaX registers both a RELION 3.1 and a RELION5 reader for .star, so the
    # format has to be pinned or ChimeraX may pick the wrong coordinate convention.
    assert f"open {export.picks_path.name} format relion\n" in script
    assert "format relion5" not in script


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
