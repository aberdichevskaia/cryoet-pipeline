"""cryoet_pipeline.map_size: the density in a map measured at a level, on shapes whose
sizes are known or found by brute force over every pair of voxels."""

import math

import mrcfile
import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.distance import pdist

from cryoet_pipeline.map_size import EDGE_NOISE, KEEP_SHARE, MapSize, measure_density, read_map

BOX = 48
VOXEL = 2.0


def ball(radius: float, at: tuple[float, float, float] = (0.0, 0.0, 0.0),
         box: int = BOX) -> np.ndarray:
    """A ball of 1s in 0s, its centre `at` voxels (z, y, x) from the box centre."""

    z, y, x = np.indices((box, box, box), dtype=np.float64) - (box - 1) / 2
    inside = (z - at[0]) ** 2 + (y - at[1]) ** 2 + (x - at[2]) ** 2 <= radius ** 2
    return inside.astype(np.float32)


def brute_force(data: np.ndarray) -> tuple[float, float]:
    """The widest extent and the farthest reach from the box centre of every voxel above
    0.5, in A, from all pairs: the measurement without its hull."""

    points = np.argwhere(data > 0.5).astype(np.float64)
    centre = (np.asarray(data.shape) - 1) / 2
    widest = (pdist(points).max() + 1.0) * VOXEL
    from_centre = (np.sqrt(((points - centre) ** 2).sum(axis=1)).max() + 0.5) * VOXEL
    return widest, from_centre


def test_a_rod_is_its_length_across_and_twice_its_half_from_the_centre():
    data = np.zeros((BOX, BOX, BOX), np.float32)
    data[23:25, 23:25, 4:44] = 1.0  # 40 voxels along x, 2 x 2 across, centred
    size = measure_density(data, VOXEL, 0.5)

    # The end voxels' centres are 39 apart along x and 1 apart each way across.
    assert size.widest_angstrom == pytest.approx((math.sqrt(39**2 + 2) + 1.0) * VOXEL)
    assert size.mask_diameter_angstrom == pytest.approx(
        2.0 * (math.sqrt(19.5**2 + 0.5) + 0.5) * VOXEL)
    assert (size.pieces, size.kept, size.kept_share, size.largest_left_out) == (1, 1, 1.0, 0.0)
    assert size.box_px == (BOX, BOX, BOX)


@pytest.mark.parametrize("shape", [
    ball(9),
    ball(6, at=(0, 0, 14)),
    np.maximum(ball(8, at=(-3, 2, -5)), ball(6, at=(2, -1, 5))),
    np.maximum(np.maximum(ball(4, at=(10, 0, 0)), ball(7)), ball(3, at=(-2, 9, 1))),
])
def test_the_hull_finds_what_every_pair_of_voxels_finds(shape):
    size = measure_density(shape, VOXEL, 0.5)
    widest, from_centre = brute_force(shape)

    assert size.kept == 1
    assert size.widest_angstrom == pytest.approx(widest)
    assert size.from_centre_angstrom == pytest.approx(from_centre)


def test_a_particle_off_the_centre_needs_a_wider_mask_than_it_is_wide():
    size = measure_density(ball(6, at=(0, 0, 14)), VOXEL, 0.5)

    assert size.widest_angstrom < 14 * VOXEL
    # 14 voxels out, and 6 more to its far side: the mask must reach past both.
    assert size.mask_diameter_angstrom > 2 * 20 * VOXEL


def test_a_complex_that_comes_apart_at_the_level_is_measured_whole():
    halves = np.maximum(ball(5, at=(0, 0, -11)), ball(5, at=(0, 0, 11)))
    size = measure_density(halves, VOXEL, 0.5)

    assert (size.pieces, size.kept) == (2, 2)
    assert size.widest_angstrom == pytest.approx(brute_force(halves)[0])


def test_specks_of_noise_are_left_out_and_counted():
    particle = ball(8)
    noisy = particle.copy()
    for corner in [(2, 2, 2), (2, 45, 3), (44, 4, 40), (40, 40, 44), (3, 30, 44)]:
        noisy[corner] = 1.0
    size = measure_density(noisy, VOXEL, 0.5)

    assert (size.pieces, size.kept) == (6, 1)
    assert size.largest_left_out == pytest.approx(1 / particle.sum())
    assert size.widest_angstrom == pytest.approx(brute_force(particle)[0])
    assert size.faces_reached == 0


@pytest.mark.parametrize(("second", "kept"), [(100, 2), (99, 1)])
def test_a_piece_is_kept_from_a_tenth_of_the_largest(second, kept):
    assert KEEP_SHARE == 0.1
    data = np.zeros((BOX, BOX, BOX), np.float32)
    data[5:15, 5:15, 5:15] = 1.0  # 1000 voxels
    flat = np.zeros(100, np.float32)
    flat[:second] = 1.0
    data[30, 30:40, 30:40] = flat.reshape(10, 10)
    size = measure_density(data, VOXEL, 0.5)

    assert size.kept == kept
    assert size.largest_left_out == (0.0 if kept == 2 else 0.099)


def test_pieces_touching_at_a_corner_are_one():
    data = np.zeros((BOX, BOX, BOX), np.float32)
    data[10:15, 10:15, 10:15] = 1.0
    data[15:20, 15:20, 15:20] = 1.0
    size = measure_density(data, VOXEL, 0.5)

    assert size.pieces == 1


def test_density_that_reaches_the_box_is_said_to():
    slab = np.zeros((BOX, BOX, BOX), np.float32)
    slab[22:26, :, :] = 1.0  # a membrane across the box, normal along z
    membrane = measure_density(np.maximum(slab, ball(8, at=(10, 0, 0))), VOXEL, 0.5)
    alone = measure_density(ball(8), VOXEL, 0.5)

    assert (membrane.faces_reached, membrane.inside_angstrom) == (4, 0.0)
    # A ball of radius 8 about the centre of a 48 box leaves 16 voxels on every side.
    assert (alone.faces_reached, alone.inside_angstrom) == (0, 16 * VOXEL)
    rod = np.zeros((BOX, BOX, BOX), np.float32)
    rod[20:28, 20:28, 1:BOX - 1] = 1.0  # one empty voxel before each x face
    assert (measure_density(rod, VOXEL, 0.5).faces_reached,
            measure_density(rod, VOXEL, 0.5).inside_angstrom) == (0, 1 * VOXEL)
    rod[20:28, 20:28, 0] = 1.0
    assert measure_density(rod, VOXEL, 0.5).faces_reached == 1


def noise(sd: float = 1.0, seed: int = 1, box: int = BOX) -> np.ndarray:
    """Gaussian noise correlated over about a voxel, as a map's is, at sd `sd`."""

    white = np.random.default_rng(seed).standard_normal((box, box, box))
    smooth = ndimage.gaussian_filter(white, 1.0, mode="wrap")
    return (sd * smooth / smooth.std()).astype(np.float32)


def membrane() -> np.ndarray:
    slab = np.zeros((BOX, BOX, BOX), np.float32)
    slab[22:26, :, :] = 1.0  # across the box, normal along z
    return np.maximum(slab, ball(8, at=(10, 0, 0)))


def test_a_level_in_the_noise_is_told_from_a_membrane_that_reaches_the_faces():
    # A particle 6 sd high in noise: at 0.5 sd the noise above the level joins it and
    # reaches every face, as on EMD-4805 at its authors' contour.
    in_noise = measure_density(noise() + 6.0 * ball(8), VOXEL, 0.5)
    # A membrane in noise of sd 0.2, at 3.5 sd: it reaches four faces by itself.
    across = measure_density(membrane() + noise(0.2), VOXEL, 0.7)

    assert in_noise.faces_reached == 6 and in_noise.in_noise
    assert in_noise.edge_below / in_noise.edge_above == pytest.approx(1.0, abs=0.15)
    assert across.faces_reached == 4 and not across.in_noise
    assert across.edge_below < 0.1 * EDGE_NOISE * across.edge_above


def test_above_the_noise_the_particle_is_measured_and_not_in_the_noise():
    size = measure_density(noise() + 6.0 * ball(8), VOXEL, 3.0)

    assert (size.faces_reached, size.kept, size.in_noise) == (0, 1, False)
    # Noise still touches the particle: the mask is a few voxels wider than the ball's.
    assert 2 * 8.5 * VOXEL <= size.mask_diameter_angstrom <= 2 * 12 * VOXEL


def test_the_noise_is_counted_about_the_median_of_the_edge_not_about_zero():
    data = noise() + 6.0 * ball(8)
    size = measure_density(data, VOXEL, 0.5)
    shifted = measure_density(data + 100.0, VOXEL, 100.5)

    assert shifted.edge_median == pytest.approx(size.edge_median + 100.0, abs=1e-3)
    assert shifted.in_noise
    assert (shifted.edge_above, shifted.edge_below) == pytest.approx(
        (size.edge_above, size.edge_below), abs=1e-3)


def test_a_particle_that_fills_most_of_its_box_is_not_taken_for_noise():
    # Clipped by every face, the ball holds most of the map, so the map's median lies
    # in the particle; the faces' median is still the noise's.
    data = noise(0.2) + 6.0 * ball(26)
    size = measure_density(data, VOXEL, 1.0)

    assert np.median(data) > 3.0
    assert abs(size.edge_median) < 0.2
    assert size.faces_reached == 6 and not size.in_noise


def test_a_map_without_noise_is_never_in_the_noise():
    # Nothing on the faces is below their median, and nothing above the level either.
    alone = measure_density(ball(8), VOXEL, 0.5)
    across = measure_density(membrane(), VOXEL, 0.5)

    assert (alone.faces_reached, alone.edge_above, alone.edge_below) == (0, 0.0, 0.0)
    assert not alone.in_noise
    assert (across.faces_reached, across.edge_below, across.in_noise) == (4, 0.0, False)


def test_the_outermost_voxels_are_each_counted_once_and_the_map_share_is_exact():
    data = np.zeros((10, 10, 10), np.float32)
    data[:, :, 0] = 1.0  # one face of 100 voxels, of the 1000 - 8**3 = 488 outermost
    size = measure_density(data, VOXEL, 0.5)

    assert size.edge_above == pytest.approx(100 / 488)
    assert size.share_of_map == pytest.approx(100 / 1000)
    # The face's rim lies on the four faces beside it.
    assert (size.faces_reached, size.in_noise) == (5, False)


def test_a_level_above_everything_is_refused_with_the_maximum():
    with pytest.raises(ValueError, match="its maximum is 1"):
        measure_density(ball(5), VOXEL, 1.5)


def test_the_level_decides_what_is_measured():
    # A core of 2 inside a shell of 1: above 0.5 the shell, above 1.5 the core.
    data = np.maximum(ball(10), 2.0 * ball(5))
    assert isinstance(measure_density(data, VOXEL, 0.5), MapSize)
    assert measure_density(data, VOXEL, 0.5).widest_angstrom == pytest.approx(
        brute_force(ball(10))[0])
    assert measure_density(data, VOXEL, 1.5).widest_angstrom == pytest.approx(
        brute_force(ball(5))[0])


# --- Reading a map ------------------------------------------------------------------


def write(path, data, voxel_size=VOXEL, compression=None):
    with mrcfile.new(path, overwrite=True, compression=compression) as mrc:
        mrc.set_data(np.asarray(data, dtype=np.float32))
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
    return path


def test_a_map_and_a_gzipped_map_are_read_with_their_voxel(tmp_path):
    data = ball(5, box=16)
    for path in (write(tmp_path / "emd_1.map", data, 2.5),
                 write(tmp_path / "emd_1.map.gz", data, 2.5, compression="gzip")):
        read, voxel = read_map(path)
        assert voxel == pytest.approx(2.5)
        np.testing.assert_array_equal(read, data)


@pytest.mark.parametrize(("data", "voxel_size", "message"), [
    (ball(5, box=16), None, "no voxel size"),
    (ball(5, box=16), (2.0, 2.0, 3.0), "unequal sides"),
    (np.ones((16, 16), np.float32), 2.0, "not a 3D map"),
])
def test_a_map_that_cannot_be_measured_as_is_is_refused(tmp_path, data, voxel_size, message):
    path = write(tmp_path / "bad.mrc", data, voxel_size)
    with pytest.raises(ValueError, match=message):
        read_map(path)
