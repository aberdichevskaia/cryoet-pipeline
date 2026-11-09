"""Tilt series for RELION 5: the geometry of each tilt image, and the STAR files.

RELION 5 reconstructs a tomogram, and extracts pseudo-subtomograms, from the
tilt images themselves, so a particle exported to it is only as good as the
geometry it is given. That geometry is five numbers per tilt image, which
`Tomogram::setProjectionMatrix` (src/jaz/tomography/tomogram.cpp, RELION 5.0,
commit 7f45a71) turns into the projection of a tomogram voxel `v` (tilt-series
pixels, 0-based) onto the tilt image:

    u = int(n/2) + shift / pixel + [Rz(zrot) Ry(ytilt) Rx(xtilt) (v - int(N/2))]_xy

`n` is the tilt image's size, `N` the tomogram's (rlnTomoSizeX/Y/Z), and the
rotations are right-handed (gravis' `rotation`, the GL formula).

The pipeline's tomogram is IMOD's. newstack turns each raw tilt image by its
transform, `u' - c' = A (u - c) + D` about the centres `c = (n - 1) / 2` of
0-based pixel centres, and tilt reconstructs the aligned stack with the tilt
angles, -XAXISTILT and -SHIFT 0,z. The numbers written here make RELION's
projection of a voxel equal IMOD's, voxel for voxel, in IMOD's standard
orientation (what `clip rotx` or `tilt -RotateBy90` give):

    zrot   = the rotation of A^-1
    ytilt  = the view's tilt angle, as tilt was given it
    xtilt  = -XAXISTILT
    shift  = (c - int(n/2)) - A^-1 D + M (int(N/2) - C),   M = [Rz Ry Rx]_xy
    C      = ((Nx - 1)/2, (Ny - 1)/2, (Nz - 1)/2 - zshift)

The pipeline's own tomograms are not in that orientation: its canonical volume
is tilt's output transposed (`transpose(1, 0, 2)` in the reconstruction
backend), which is the standard orientation reversed in z, a mirror image.
RELION is given the standard orientation, the one its own IMOD import gives
(ytilt is the .tlt angle there too), and whatever the pipeline hands over in
its own frame (particle positions, poses, maps) has to be mirrored in z first.

Measured on 9 November 2026 (RELION 5.0 at 7f45a71 built on a Mac, IMOD 5.1.12,
EMPIAR-10164 TS_01 of outputs/dev-ts01-registry at bin8, scripts/
relion_geometry_check.py): RELION's own reconstruction from these numbers is the
pipeline's tomogram reversed in z, NCC 0.68 against at most 0.03 for the other
seven axis reversals; in 128-voxel boxes the shift is 0.00 voxel at the centre
and at most 0.07 voxel at the four quarter positions. With ytilt and xtilt both negated it is
the pipeline's own (mirrored) frame instead, and with only one of them negated
nothing matches (NCC 0.11 at best). A reconstruction with -SHIFT 0,20 lands as
well (0.00 voxel at the centre), which fixed the sign of the z shift.

What RELION's model cannot hold is reported, not hidden: it has a rotation and
a shift per image, so the rest of IMOD's transform (tiltalign's per-view
magnification, MagOption 1 in this pipeline, and any stretch or skew) is
dropped, as RELION's own IMOD import drops it. `RelionTilt.dropped_px` is how
far that moves the corners of the aligned image; on TS_01 the magnifications
span 0.9996-1.0013, up to 0.88 bin8 pixel (4.8 A) at the corners.

Not measured yet: the astigmatism angle's turn into the raw image's frame
(`defocus_angle_in_raw_frame`), and the defocus handedness (rlnTomoHand), which
RELION's documentation says has to be determined experimentally.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cryoet_pipeline.models import AlignmentTransform

# The STAR dialect RELION 5.0 writes at the top of its files.
STAR_VERSION_LINE = "# version 50001"


@dataclass(frozen=True)
class ImodReconstruction:
    """What IMOD's tilt was given for a tomogram, in pixels of the stack it read.

    `size_xyz` is tilt's output: the aligned stack's width and height and the
    -THICKNESS; `x_axis_tilt_deg` is -XAXISTILT and `z_shift_px` the z of -SHIFT.
    """

    size_xyz: tuple[int, int, int]
    x_axis_tilt_deg: float
    z_shift_px: float

    def centre(self) -> np.ndarray:
        """IMOD's centre of rotation in the standard orientation, 0-based voxels."""

        nx, ny, nz = self.size_xyz
        return np.array([(nx - 1) / 2, (ny - 1) / 2, (nz - 1) / 2 - self.z_shift_px])


@dataclass(frozen=True)
class RelionTilt:
    """One tilt image's rlnTomoXTilt, YTilt, ZRot and X/YShiftAngst.

    `dropped_px` is the largest displacement, at the corners of the aligned
    image, between IMOD's transform and RELION's rotation-only model of it, in
    pixels of the stack the transform is in: what magnification, stretch and
    skew would have moved there. It is 0 for a pure rotation.
    """

    xtilt_deg: float
    ytilt_deg: float
    zrot_deg: float
    xshift_angstrom: float
    yshift_angstrom: float
    dropped_px: float


def rotation_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )


def rotation_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    )


def rotation_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]]
    )


def _matrix(transform: AlignmentTransform) -> np.ndarray:
    return np.array([[transform.a11, transform.a12], [transform.a21, transform.a22]], dtype=float)


def _nearest_rotation_deg(matrix: np.ndarray) -> float:
    """The angle of the rotation closest to a 2 x 2 matrix (its polar factor)."""

    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("the alignment transform mirrors the image; RELION cannot represent it")
    return math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))


def relion_tilt(
    transform: AlignmentTransform,
    *,
    image_size_xy: tuple[int, int],
    reconstruction: ImodReconstruction,
    pixel_size_angstrom: float,
) -> RelionTilt:
    """RELION's five numbers for one view, from its IMOD transform.

    `transform` maps the raw tilt image to the aligned one, its shift in pixels
    of the stack RELION will read, whose images are `image_size_xy`; the
    reconstruction is in the same pixels. See the module docstring.
    """

    if pixel_size_angstrom <= 0:
        raise ValueError("the pixel size must be positive")
    matrix = _matrix(transform)
    inverse = np.linalg.inv(matrix)
    zrot = _nearest_rotation_deg(inverse)
    ytilt = transform.tilt_angle_deg
    xtilt = -reconstruction.x_axis_tilt_deg
    projection = (rotation_z(zrot) @ rotation_y(ytilt) @ rotation_x(xtilt))[:2]

    nx, ny = image_size_xy
    image_centre_imod = np.array([(nx - 1) / 2, (ny - 1) / 2])
    image_centre_relion = np.array([nx // 2, ny // 2], dtype=float)
    size = np.array(reconstruction.size_xyz)
    shift_px = (
        (image_centre_imod - image_centre_relion)
        - inverse @ np.array([transform.shift_x_px, transform.shift_y_px])
        + projection @ (size // 2 - reconstruction.centre())
    )

    half_x, half_y = (size[0] - 1) / 2, (size[1] - 1) / 2
    corners = np.array([[half_x, half_y], [half_x, -half_y], [-half_x, half_y], [-half_x, -half_y]])
    rotation = rotation_z(zrot)[:2, :2]
    dropped = float(np.max(np.linalg.norm(corners @ (rotation - inverse).T, axis=1)))
    return RelionTilt(
        xtilt_deg=xtilt,
        ytilt_deg=ytilt,
        zrot_deg=zrot,
        xshift_angstrom=float(shift_px[0] * pixel_size_angstrom),
        yshift_angstrom=float(shift_px[1] * pixel_size_angstrom),
        dropped_px=dropped,
    )


def relion_projection(
    tilt: RelionTilt,
    *,
    image_size_xy: tuple[int, int],
    tomogram_size_xyz: tuple[int, int, int],
    pixel_size_angstrom: float,
) -> np.ndarray:
    """RELION's 3 x 4 projection of a view, as `Tomogram::setProjectionMatrix`
    builds it: `u = P @ (v, 1)`, rows x, y and the depth z."""

    rotation = rotation_z(tilt.zrot_deg) @ rotation_y(tilt.ytilt_deg) @ rotation_x(tilt.xtilt_deg)
    centre = np.array([size // 2 for size in tomogram_size_xyz], dtype=float)
    offset = np.array(
        [
            image_size_xy[0] // 2 + tilt.xshift_angstrom / pixel_size_angstrom,
            image_size_xy[1] // 2 + tilt.yshift_angstrom / pixel_size_angstrom,
            0.0,
        ]
    )
    return np.hstack([rotation, (offset - rotation @ centre)[:, None]])


def defocus_angle_in_raw_frame(azimuth_deg: float, transform: AlignmentTransform) -> float:
    """An astigmatism azimuth fitted on the aligned stack, in the raw image's frame.

    The aligned image is the raw one turned by the rotation of A, so a direction
    at `azimuth_deg` there lies at azimuth minus that rotation in the raw image;
    folded into [0, 180). Not yet checked against a fit on raw images.
    """

    turn = _nearest_rotation_deg(_matrix(transform))
    return (azimuth_deg - turn) % 180.0


def pre_exposures(
    dose_per_view: Mapping[int, float], acquisition_rank: Mapping[int, int]
) -> dict[int, float]:
    """Each view's dose received before it was taken (rlnMicrographPreExposure).

    Keyed by z value. Every acquired view counts, including those the alignment
    or the CTF left out: they were exposed all the same.
    """

    if set(dose_per_view) != set(acquisition_rank):
        raise ValueError("doses and acquisition ranks must name the same views")
    ranks = sorted(acquisition_rank.values())
    if len(set(ranks)) != len(ranks):
        raise ValueError("two views share an acquisition rank")
    order = sorted(acquisition_rank, key=acquisition_rank.__getitem__)
    received = 0.0
    result: dict[int, float] = {}
    for z_value in order:
        dose = float(dose_per_view[z_value])
        if dose < 0 or not math.isfinite(dose):
            raise ValueError(f"view z={z_value} has dose {dose}")
        result[z_value] = received
        received += dose
    return result


def _star_value(value: object) -> str:
    if isinstance(value, bool):
        raise TypeError("STAR values are numbers or names, not booleans")
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        if not math.isfinite(float(value)):
            raise ValueError("STAR values must be finite")
        return f"{float(value):.6f}"
    text = str(value)
    if not text or any(character.isspace() for character in text):
        raise ValueError(f"STAR names cannot be empty or hold whitespace: {text!r}")
    return text


def star_text(blocks: Sequence[tuple[str, Sequence[str], Sequence[Sequence[object]]]]) -> str:
    """STAR text of loop tables `(block name, labels, rows)`, as RELION writes them."""

    lines = ["", STAR_VERSION_LINE, ""]
    for name, labels, rows in blocks:
        lines += [f"data_{name}", "", "loop_"]
        lines += [f"_{label} #{index}" for index, label in enumerate(labels, start=1)]
        for row in rows:
            if len(row) != len(labels):
                raise ValueError(
                    f"a row of data_{name} has {len(row)} values for {len(labels)} labels"
                )
            lines.append(" ".join(_star_value(value) for value in row))
        lines += ["", ""]
    return "\n".join(lines)


TILT_SERIES_LABELS = (
    "rlnMicrographName",
    "rlnTomoNominalStageTiltAngle",
    "rlnMicrographPreExposure",
    "rlnDefocusU",
    "rlnDefocusV",
    "rlnDefocusAngle",
    "rlnTomoXTilt",
    "rlnTomoYTilt",
    "rlnTomoZRot",
    "rlnTomoXShiftAngst",
    "rlnTomoYShiftAngst",
)

TOMOGRAMS_LABELS = (
    "rlnTomoName",
    "rlnTomoTiltSeriesStarFile",
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
    "rlnMicrographOriginalPixelSize",
    "rlnTomoHand",
    "rlnTomoTiltSeriesPixelSize",
    "rlnTomoSizeX",
    "rlnTomoSizeY",
    "rlnTomoSizeZ",
)


@dataclass(frozen=True)
class RelionView:
    """One row of a tilt series' STAR table."""

    image: str  # "<section, from 1>@<stack>"
    nominal_tilt_deg: float
    pre_exposure: float  # e/A^2
    defocus_u_angstrom: float
    defocus_v_angstrom: float
    defocus_angle_deg: float
    tilt: RelionTilt

    def row(self) -> tuple[object, ...]:
        return (
            self.image,
            self.nominal_tilt_deg,
            self.pre_exposure,
            self.defocus_u_angstrom,
            self.defocus_v_angstrom,
            self.defocus_angle_deg,
            self.tilt.xtilt_deg,
            self.tilt.ytilt_deg,
            self.tilt.zrot_deg,
            self.tilt.xshift_angstrom,
            self.tilt.yshift_angstrom,
        )


@dataclass(frozen=True)
class RelionOptics:
    voltage_kv: float
    spherical_aberration_mm: float
    amplitude_contrast: float
    original_pixel_size_angstrom: float


def write_tomogram_set(
    folder: Path,
    series: Mapping[str, tuple[Sequence[RelionView], tuple[int, int, int], float]],
    *,
    optics: RelionOptics,
    hand: int,
) -> Path:
    """Write `<folder>/tomograms.star` and `<folder>/tilt_series/<name>.star`.

    `series` maps each tomogram's name to its views, its size in pixels of its
    stack (x, y, z) and that stack's pixel size. Paths in tomograms.star are
    relative to `folder`, which RELION is run from.
    """

    if hand not in (1, -1):
        raise ValueError("rlnTomoHand is +1 or -1")
    if not series:
        raise ValueError("no tilt series to write")
    (folder / "tilt_series").mkdir(parents=True, exist_ok=True)
    global_rows = []
    for name, (views, size_xyz, pixel_size) in series.items():
        if not views:
            raise ValueError(f"{name} has no views")
        relative = Path("tilt_series") / f"{name}.star"
        (folder / relative).write_text(
            star_text([(name, TILT_SERIES_LABELS, [view.row() for view in views])])
        )
        global_rows.append(
            (
                name,
                str(relative),
                optics.voltage_kv,
                optics.spherical_aberration_mm,
                optics.amplitude_contrast,
                optics.original_pixel_size_angstrom,
                hand,
                pixel_size,
                *size_xyz,
            )
        )
    path = folder / "tomograms.star"
    path.write_text(star_text([("global", TOMOGRAMS_LABELS, global_rows)]))
    return path
