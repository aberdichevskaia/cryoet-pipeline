from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from cryoet_pipeline.cli import app
from cryoet_pipeline.map_comparison import (
    ComparedMap,
    central_section,
    central_slab_weights,
    render_map_comparison,
    sphere_mask,
    z_score_inside,
)


def _blob(size: int, centre_zyx: tuple[float, float, float], sigma: float = 2.0) -> np.ndarray:
    axis = np.arange(size, dtype=np.float64)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    cz, cy, cx = centre_zyx
    return np.exp(-((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma**2))


def test_one_voxel_slab_is_the_middle_plane_or_the_two_middle_planes():
    odd = central_slab_weights(9, 1.0)
    assert odd[4] == pytest.approx(1.0) and odd.sum() == pytest.approx(1.0)
    even = central_slab_weights(8, 1.0)
    assert even[3] == pytest.approx(0.5) and even[4] == pytest.approx(0.5)
    assert even.sum() == pytest.approx(1.0)


def test_slab_weights_are_symmetric_about_the_geometric_centre():
    for size in (8, 9):
        for thickness in (1.0, 2.0, 3.0, 4.5):
            weights = central_slab_weights(size, thickness)
            assert np.allclose(weights, weights[::-1])


@pytest.mark.parametrize(
    ("plane", "expected_row_column"),
    # Blob at (z, y, x) = (centre, centre + 6, centre + 3) in a 33 box (centre 16).
    [("xy", (22, 19)), ("xz", (16, 19)), ("yz", (16, 22))],
)
def test_sections_put_the_vertical_axis_in_rows(plane, expected_row_column):
    volume = _blob(33, (16, 22, 19))
    section = central_section(volume, plane, 1.0)
    assert np.unravel_index(np.argmax(section), section.shape) == expected_row_column


def test_z_score_uses_only_the_sphere():
    volume = torch.zeros((16, 16, 16))
    mask = sphere_mask(16, diameter_voxels=10.0)
    volume[mask] = torch.randn(int(mask.sum())) * 5 + 7
    volume[~mask] = 1000.0
    scored = z_score_inside(volume, mask)
    assert float(scored[mask].mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(scored[mask].to(torch.float64).std()) == pytest.approx(1.0, abs=1e-5)


def test_protein_bright_map_is_drawn_like_its_dark_twin(tmp_path, monkeypatch):
    dark = torch.as_tensor(-_blob(24, (11.5, 11.5, 11.5)), dtype=torch.float32)
    sections: list[np.ndarray] = []

    import cryoet_pipeline.map_comparison as module

    original = module.central_section

    def recording(volume, plane, thickness):
        result = original(volume, plane, thickness)
        sections.append(result)
        return result

    monkeypatch.setattr(module, "central_section", recording)
    render_map_comparison(
        [ComparedMap("dark", dark), ComparedMap("bright", -dark, protein_bright=True)],
        out=tmp_path / "cmp.png",
        voxel_size_angstrom=10.0,
        planes=("xy",),
        low_pass_angstrom=(40.0,),
    )
    # Row-major: (xy, dark), (xy, bright), (xy 40 A, dark), (xy 40 A, bright).
    assert len(sections) == 4
    assert np.allclose(sections[0], sections[1], atol=1e-5)
    assert np.allclose(sections[2], sections[3], atol=1e-5)
    assert sections[0].min() < -1.0  # protein is dark


def test_render_refuses_mismatched_boxes(tmp_path):
    with pytest.raises(ValueError, match="pad them"):
        render_map_comparison(
            [ComparedMap("a", torch.zeros((8, 8, 8))), ComparedMap("b", torch.zeros((9, 9, 9)))],
            out=tmp_path / "x.png",
            voxel_size_angstrom=1.0,
        )


def _write_mrc(path: Path, data: np.ndarray, voxel_size: float) -> None:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = voxel_size


def test_compare_maps_cli_pads_mixed_boxes_and_writes_the_figure(tmp_path):
    rng = np.random.default_rng(0)
    noise = 0.1 * rng.normal(size=(32,) * 3)
    _write_mrc(tmp_path / "avg.mrc", -_blob(32, (15.5,) * 3) + noise, 8.0)
    _write_mrc(tmp_path / "tpl.mrc", _blob(27, (13.0,) * 3), 8.0)
    out = tmp_path / "figure.png"

    result = CliRunner().invoke(
        app,
        [
            "compare-maps",
            "--map", "average=" + str(tmp_path / "avg.mrc"),
            "--map", "template\\nEMD-0000=" + str(tmp_path / "tpl.mrc"),
            "--protein-bright", "template\\nEMD-0000",
            "--low-pass", "35",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "padded 'template\\nEMD-0000'" in result.output
    assert out.stat().st_size > 0


def test_compare_maps_cli_rejects_mismatched_voxel_size(tmp_path):
    _write_mrc(tmp_path / "a.mrc", np.zeros((8, 8, 8)), 8.0)
    _write_mrc(tmp_path / "b.mrc", np.zeros((8, 8, 8)), 4.0)
    result = CliRunner().invoke(
        app,
        ["compare-maps", "--map", str(tmp_path / "a.mrc"), "--map", str(tmp_path / "b.mrc"),
         "--out", str(tmp_path / "x.png")],
    )
    assert result.exit_code != 0
    assert "voxel size" in result.output


def test_compare_maps_cli_takes_voxel_size_for_headerless_maps(tmp_path):
    # Dynamo writes a zero voxel size.
    _write_mrc(tmp_path / "dynamo.mrc", -_blob(16, (7.5,) * 3), 0.0)
    _write_mrc(tmp_path / "ours.mrc", -_blob(16, (7.5,) * 3), 10.0)
    arguments = ["compare-maps", "--map", str(tmp_path / "dynamo.mrc"),
                 "--map", str(tmp_path / "ours.mrc"), "--out", str(tmp_path / "x.png")]

    missing = CliRunner().invoke(app, arguments)
    assert missing.exit_code != 0 and "--voxel-size" in missing.output

    wrong = CliRunner().invoke(app, [*arguments, "--voxel-size", "8"])
    assert wrong.exit_code != 0 and "resample" in wrong.output

    right = CliRunner().invoke(app, [*arguments, "--voxel-size", "10"])
    assert right.exit_code == 0, right.output
