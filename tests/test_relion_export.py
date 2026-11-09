from __future__ import annotations

import math

import numpy as np
import pytest

from cryoet_pipeline.models import AlignmentTransform
from cryoet_pipeline.relion_export import (
    ImodReconstruction,
    RelionOptics,
    RelionTilt,
    RelionView,
    defocus_angle_in_raw_frame,
    pre_exposures,
    relion_tilt,
    star_text,
    write_tomogram_set,
)


def _gl_rotation(axis, deg):
    """gravis' t3Matrix::rotation (the GL formula), transcribed for this test."""

    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    a = math.radians(deg)
    cross = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    outer = np.outer(n, n)
    return outer + math.cos(a) * (np.eye(3) - outer) + math.sin(a) * cross


def _relion_pixel(tilt: RelionTilt, voxel, image_xy, size_xyz, pixel):
    """Tomogram::setProjectionMatrix (RELION 5.0) applied to a voxel, written
    from its source rather than through the module."""

    s0 = -np.array([int(size_xyz[0] / 2), int(size_xyz[1] / 2), int(size_xyz[2] / 2)], dtype=float)
    s1 = np.array([tilt.xshift_angstrom / pixel, tilt.yshift_angstrom / pixel, 0.0])
    s2 = np.array([int(image_xy[0] / 2), int(image_xy[1] / 2), 0.0])
    r = (
        _gl_rotation((0, 0, 1), tilt.zrot_deg)
        @ _gl_rotation((0, 1, 0), tilt.ytilt_deg)
        @ _gl_rotation((1, 0, 0), tilt.xtilt_deg)
    )
    return (r @ (np.asarray(voxel, dtype=float) + s0) + s1 + s2)[:2]


def _imod_pixel(transform: AlignmentTransform, voxel, image_xy, reconstruction: ImodReconstruction):
    """Where IMOD puts a voxel of its standard-orientation tomogram in the raw image.

    newstack: aligned - c' = A (raw - c) + D about the (n - 1) / 2 centres; tilt's
    projection in the standard orientation is Ry(tilt) Rx(-XAXISTILT) about
    ((Nx-1)/2, (Ny-1)/2, (Nz-1)/2 - zshift). That projection is the convention
    scripts/relion_geometry_check.py measured on TS_01 with RELION and IMOD, not
    something this test derives.
    """

    nx, ny, nz = reconstruction.size_xyz
    centre = np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2 - reconstruction.z_shift_px])
    projected = (
        _gl_rotation((0, 1, 0), transform.tilt_angle_deg)
        @ _gl_rotation((1, 0, 0), -reconstruction.x_axis_tilt_deg)
        @ (np.asarray(voxel, dtype=float) - centre)
    )[:2]
    aligned = projected + np.array([(nx - 1) / 2, (ny - 1) / 2])
    a = np.array([[transform.a11, transform.a12], [transform.a21, transform.a22]])
    d = np.array([transform.shift_x_px, transform.shift_y_px])
    raw_centre = np.array([(image_xy[0] - 1) / 2, (image_xy[1] - 1) / 2])
    return raw_centre + np.linalg.solve(a, aligned - np.array([(nx - 1) / 2, (ny - 1) / 2]) - d)


def _transform(rng, *, scale=1.0, tilt=None):
    angle = math.radians(rng.uniform(-180, 180))
    a = scale * np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    return AlignmentTransform(
        z_value=int(rng.integers(0, 60)),
        tilt_angle_deg=float(rng.uniform(-65, 65) if tilt is None else tilt),
        a11=float(a[0, 0]),
        a12=float(a[0, 1]),
        a21=float(a[1, 0]),
        a22=float(a[1, 1]),
        shift_x_px=float(rng.uniform(-80, 80)),
        shift_y_px=float(rng.uniform(-80, 80)),
    )


@pytest.mark.parametrize("seed", range(12))
def test_relion_projects_every_voxel_where_imod_does(seed: int) -> None:
    rng = np.random.default_rng(seed)
    image_xy = (int(rng.integers(300, 401)), int(rng.integers(300, 401)))  # odd and even sizes
    reconstruction = ImodReconstruction(
        size_xyz=(
            int(rng.integers(300, 401)),
            int(rng.integers(300, 401)),
            int(rng.integers(60, 121)),
        ),
        x_axis_tilt_deg=float(rng.uniform(-3, 3)),
        z_shift_px=float(rng.uniform(-25, 25)),
    )
    pixel = float(rng.uniform(1, 20))
    transform = _transform(rng)
    tilt = relion_tilt(
        transform, image_size_xy=image_xy, reconstruction=reconstruction, pixel_size_angstrom=pixel
    )
    assert tilt.dropped_px == pytest.approx(0.0, abs=1e-9)
    for voxel in rng.uniform(0, 1, size=(20, 3)) * np.array(reconstruction.size_xyz):
        np.testing.assert_allclose(
            _relion_pixel(tilt, voxel, image_xy, reconstruction.size_xyz, pixel),
            _imod_pixel(transform, voxel, image_xy, reconstruction),
            atol=1e-8,
        )


def test_magnification_is_reported_as_dropped_and_exact_at_the_centre() -> None:
    rng = np.random.default_rng(7)
    image_xy = (400, 380)
    # No x-axis tilt, and an untilted view, so that the voxel below projects onto
    # the aligned image's corner exactly.
    reconstruction = ImodReconstruction(
        size_xyz=(390, 410, 90), x_axis_tilt_deg=0.0, z_shift_px=-4.0
    )
    transform = _transform(rng, scale=1.002, tilt=0.0)
    tilt = relion_tilt(
        transform, image_size_xy=image_xy, reconstruction=reconstruction, pixel_size_angstrom=2.0
    )
    # A magnification m moves a point at r from the centre by |1 - 1/m| r.
    half_diagonal = math.hypot(389 / 2, 409 / 2)
    assert tilt.dropped_px == pytest.approx((1 - 1 / 1.002) * half_diagonal, rel=1e-6)
    centre = reconstruction.centre()
    np.testing.assert_allclose(
        _relion_pixel(tilt, centre, image_xy, reconstruction.size_xyz, 2.0),
        _imod_pixel(transform, centre, image_xy, reconstruction),
        atol=1e-8,
    )
    corner = centre + np.array([389 / 2, 409 / 2, 0.0])
    gap = np.linalg.norm(
        _relion_pixel(tilt, corner, image_xy, reconstruction.size_xyz, 2.0)
        - _imod_pixel(transform, corner, image_xy, reconstruction)
    )
    assert gap == pytest.approx(tilt.dropped_px, rel=1e-6)


def test_ts01_first_view_as_measured() -> None:
    # EMPIAR-10164 TS_01, view z=38 at bin8: the numbers of the run that
    # scripts/relion_geometry_check.py passed on 9 November 2026.
    transform = AlignmentTransform(
        z_value=38,
        tilt_angle_deg=-53.5,
        a11=0.0869669,
        a12=0.9962116,
        a21=-0.9962116,
        a22=0.0869669,
        shift_x_px=-118.771 / 8,
        shift_y_px=39.338 / 8,
    )
    tilt = relion_tilt(
        transform,
        image_size_xy=(927, 959),
        reconstruction=ImodReconstruction((960, 928, 296), -1.62, 0.0),
        pixel_size_angstrom=5.4,
    )
    assert tilt.xtilt_deg == pytest.approx(1.62)
    assert tilt.ytilt_deg == pytest.approx(-53.5)
    assert tilt.zrot_deg == pytest.approx(85.0108631175, abs=1e-8)
    assert tilt.xshift_angstrom == pytest.approx(30.7577048660, abs=1e-6)
    assert tilt.yshift_angstrom == pytest.approx(77.1629633569, abs=1e-6)


def test_a_mirroring_transform_is_refused() -> None:
    transform = AlignmentTransform(
        z_value=0,
        tilt_angle_deg=0.0,
        a11=-1.0,
        a12=0.0,
        a21=0.0,
        a22=1.0,
        shift_x_px=0.0,
        shift_y_px=0.0,
    )
    with pytest.raises(ValueError, match="mirrors"):
        relion_tilt(
            transform,
            image_size_xy=(100, 100),
            reconstruction=ImodReconstruction((100, 100, 20), 0.0, 0.0),
            pixel_size_angstrom=1.0,
        )


def test_defocus_angle_turns_back_by_the_alignment_rotation() -> None:
    angle = math.radians(-85.0)
    transform = AlignmentTransform(
        z_value=0,
        tilt_angle_deg=0.0,
        a11=math.cos(angle),
        a12=-math.sin(angle),
        a21=math.sin(angle),
        a22=math.cos(angle),
        shift_x_px=3.0,
        shift_y_px=-2.0,
    )
    assert defocus_angle_in_raw_frame(10.0, transform) == pytest.approx(95.0)
    assert defocus_angle_in_raw_frame(100.0, transform) == pytest.approx(5.0)


def test_pre_exposure_counts_views_left_out_later() -> None:
    # Dose-symmetric order 0, +3, -3, +6, -6; z 2 (the -3) was left out of the
    # alignment, but the +6 taken after it was exposed to it all the same.
    ranks = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    doses = {0: 3.0, 1: 3.0, 2: 3.0, 3: 2.5, 4: 2.5}
    result = pre_exposures(doses, ranks)
    assert result == {0: 0.0, 1: 3.0, 2: 6.0, 3: 9.0, 4: 11.5}
    shuffled = pre_exposures({10: 1.0, 11: 2.0, 12: 4.0}, {10: 2, 11: 0, 12: 1})
    assert shuffled == {11: 0.0, 12: 2.0, 10: 6.0}


@pytest.mark.parametrize(
    "doses, ranks, message",
    [
        ({0: 1.0}, {1: 0}, "same views"),
        ({0: 1.0, 1: 1.0}, {0: 0, 1: 0}, "share"),
        ({0: -1.0}, {0: 0}, "dose"),
        ({0: float("nan")}, {0: 0}, "dose"),
    ],
)
def test_pre_exposure_refuses_bad_input(doses, ranks, message) -> None:
    with pytest.raises(ValueError, match=message):
        pre_exposures(doses, ranks)


def test_star_text_is_relion_loop_format() -> None:
    text = star_text(
        [("global", ("rlnTomoName", "rlnTomoHand", "rlnVoltage"), [("TS_01", -1, 300.0)])]
    )
    lines = text.splitlines()
    assert "# version 50001" in lines
    start = lines.index("data_global")
    assert lines[start : start + 6] == [
        "data_global",
        "",
        "loop_",
        "_rlnTomoName #1",
        "_rlnTomoHand #2",
        "_rlnVoltage #3",
    ]
    assert "TS_01 -1 300.000000" in lines


@pytest.mark.parametrize("value", ["has space", "", float("inf"), True])
def test_star_text_refuses_values_it_cannot_write(value) -> None:
    with pytest.raises((ValueError, TypeError)):
        star_text([("x", ("rlnA",), [(value,)])])


def test_star_text_refuses_a_short_row() -> None:
    with pytest.raises(ValueError, match="2 labels"):
        star_text([("x", ("rlnA", "rlnB"), [(1,)])])


def _view(section: int) -> RelionView:
    tilt = RelionTilt(
        xtilt_deg=0.5,
        ytilt_deg=-3.0,
        zrot_deg=85.0,
        xshift_angstrom=1.0,
        yshift_angstrom=-2.0,
        dropped_px=0.0,
    )
    return RelionView(
        image=f"{section}@tilt_series/TS_01.mrcs",
        nominal_tilt_deg=-3.0,
        pre_exposure=3.0,
        defocus_u_angstrom=30000.0,
        defocus_v_angstrom=29000.0,
        defocus_angle_deg=12.0,
        tilt=tilt,
    )


def test_write_tomogram_set_points_tomograms_star_at_relative_tilt_series(tmp_path) -> None:
    optics = RelionOptics(
        voltage_kv=300.0,
        spherical_aberration_mm=2.7,
        amplitude_contrast=0.07,
        original_pixel_size_angstrom=0.675,
    )
    path = write_tomogram_set(
        tmp_path, {"TS_01": ([_view(1), _view(2)], (960, 928, 296), 5.4)}, optics=optics, hand=-1
    )
    global_lines = path.read_text().splitlines()
    assert (
        "TS_01 tilt_series/TS_01.star 300.000000 2.700000 0.070000 0.675000 -1 5.400000 960 928 296"
        in global_lines
    )
    series_lines = (tmp_path / "tilt_series" / "TS_01.star").read_text().splitlines()
    assert "data_TS_01" in series_lines
    rows = [line for line in series_lines if "@tilt_series/" in line]
    assert [row.split()[0] for row in rows] == [
        "1@tilt_series/TS_01.mrcs",
        "2@tilt_series/TS_01.mrcs",
    ]
    assert rows[1].split()[1:] == [
        "-3.000000",
        "3.000000",
        "30000.000000",
        "29000.000000",
        "12.000000",
        "0.500000",
        "-3.000000",
        "85.000000",
        "1.000000",
        "-2.000000",
    ]


def test_write_tomogram_set_refuses_a_hand_that_is_not_a_sign(tmp_path) -> None:
    optics = RelionOptics(300.0, 2.7, 0.1, 1.0)
    with pytest.raises(ValueError, match="rlnTomoHand"):
        write_tomogram_set(
            tmp_path, {"TS_01": ([_view(1)], (10, 10, 10), 1.0)}, optics=optics, hand=0
        )
