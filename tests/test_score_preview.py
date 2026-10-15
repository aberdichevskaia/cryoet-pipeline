from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.score_preview import (
    read_score_preview,
    score_preview_from_mrc,
    score_preview_from_volume,
    slab_bounds,
    write_score_preview,
)


def test_slabs_cover_the_depth_exactly_with_a_short_last_one() -> None:
    assert slab_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert slab_bounds(8, 4) == [(0, 4), (4, 8)]
    assert slab_bounds(3, 7) == [(0, 3)]
    with pytest.raises(ValueError):
        slab_bounds(10, 0)


def test_each_slab_holds_the_highest_score_of_its_columns() -> None:
    volume = np.random.default_rng(0).normal(size=(10, 5, 6)).astype(np.float32)

    preview = score_preview_from_volume(volume, slab_thickness_px=4)

    assert preview.slab_max.shape == (3, 5, 6)
    np.testing.assert_array_equal(preview.slab_max[0], volume[0:4].max(axis=0))
    np.testing.assert_array_equal(preview.slab_max[2], volume[8:10].max(axis=0))


def test_a_coordinate_finds_the_slab_that_holds_its_voxel() -> None:
    preview = score_preview_from_volume(np.zeros((10, 2, 2), np.float32), slab_thickness_px=4)

    assert [preview.slab_containing(z) for z in (0, 3.4, 3.6, 4, 9, 9.7, -2)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        0,
    ]


def test_a_preview_survives_the_file_at_half_precision(tmp_path: Path) -> None:
    volume = np.random.default_rng(1).uniform(0.0, 0.3, size=(6, 4, 5)).astype(np.float32)
    path = tmp_path / "scores.mrc"
    with mrcfile.new(path) as handle:
        handle.set_data(volume)
        handle.voxel_size = 9.784

    preview = score_preview_from_mrc(path, slab_thickness_px=3)
    write_score_preview(preview, tmp_path / "preview.npz")
    stored = read_score_preview(tmp_path / "preview.npz")

    assert stored.voxel_spacing_angstrom == pytest.approx(9.784, abs=1e-3)
    assert stored.slab_starts.tolist() == [0, 3]
    np.testing.assert_allclose(stored.slab_max, preview.slab_max, rtol=1e-3)


def test_a_map_with_nonfinite_scores_is_refused() -> None:
    volume = np.zeros((4, 2, 2), np.float32)
    volume[3, 1, 1] = np.nan

    with pytest.raises(ValueError, match="nonfinite"):
        score_preview_from_volume(volume, slab_thickness_px=2)
