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

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.alignment import write_binned_mrc_stack
from cryoet_pipeline.backends.dynamo_search import euler_matrix
from cryoet_pipeline.imod_defocus import read_imod_defocus_file
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    TiltAlignment,
    TiltSeriesManifest,
)
from cryoet_pipeline.rotations import matrix_to_relion_euler

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


# Particles ------------------------------------------------------------------
#
# RELION 5 places a reference point r of a particle's box in the tomogram as
# `pos + E (r - int(N/2))` (ParticleSet::getMatrix4x4), E the classic RELION
# Euler matrix of rlnAngleRot/Tilt/Psi (Euler::anglesToMatrix3), pos the
# centred coordinate over the tilt-series pixel plus (w0, h0, d0) / 2. The
# pipeline's pose is Dynamo's about N/2: the reference point r lies at
# `start + N/2 + A (r - N/2) + d` of its canonical tomogram, A = M^T of the
# angles' Dynamo matrix, the same reference-to-tomogram map RELION's E is. Both
# are carried into IMOD's standard orientation, which RELION reconstructs.

MIRROR_Z = np.diag([1.0, 1.0, -1.0])


def standard_frame_matrix(matrix: np.ndarray) -> np.ndarray:
    """A reference-to-tomogram rotation of the pipeline's frame in the standard one.

    The canonical tomogram is the standard one reversed in z, so a rotation A
    there is F A F in the standard frame, F = diag(1, 1, -1), for the reference
    mirrored as `standard_frame_reference` mirrors it.
    """

    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3 x 3 matrix, got {m.shape}")
    mirrored: np.ndarray = MIRROR_Z @ m @ MIRROR_Z
    return mirrored


def standard_frame_reference(volume_zyx: np.ndarray) -> np.ndarray:
    """A map of the pipeline's frame mirrored in z about RELION's box centre int(N/2).

    Index k goes to 2 int(N/2) - k: `np.flip` alone mirrors about (N - 1)/2,
    which for an even box moves the centre by one voxel; the roll puts it back.
    """

    volume = np.asarray(volume_zyx)
    if volume.ndim != 3:
        raise ValueError("expected a (z, y, x) volume")
    mirrored = np.flip(volume, axis=0)
    if volume.shape[0] % 2 == 0:
        mirrored = np.roll(mirrored, 1, axis=0)
    return np.ascontiguousarray(mirrored)


@dataclass(frozen=True)
class TomogramPlacement:
    """Where the pipeline's canonical tomogram sits in RELION's tomogram.

    `shape_zyx` is the pipeline's tomogram, `binning` its voxel in pixels of the
    stack RELION reads, `stack_pixel_angstrom` that stack's pixel. RELION's
    tomogram is IMOD's standard orientation of the same reconstruction, so
    `binning` times as many pixels on each axis.
    """

    shape_zyx: tuple[int, int, int]
    binning: int
    stack_pixel_angstrom: float

    def __post_init__(self) -> None:
        if type(self.binning) is not int or self.binning < 1:
            raise ValueError(f"the binning must be a positive integer, got {self.binning!r}")
        if self.stack_pixel_angstrom <= 0:
            raise ValueError("the stack's pixel size must be positive")

    def size_xyz(self) -> tuple[int, int, int]:
        nz, ny, nx = self.shape_zyx
        return (self.binning * nx, self.binning * ny, self.binning * nz)

    def reconstruction(self, *, x_axis_tilt_deg: float, z_shift_px: float) -> ImodReconstruction:
        """tilt's parameters, `z_shift_px` given in the tomogram's voxels, in stack pixels."""

        return ImodReconstruction(self.size_xyz(), x_axis_tilt_deg, self.binning * z_shift_px)

    def centred_angstrom(self, position_xyz: Sequence[float]) -> np.ndarray:
        """A point of the canonical tomogram (x, y, z, 0-based voxels) as RELION's
        rlnCenteredCoordinateX/Y/ZAngst.

        Reversed in z into the standard orientation; a voxel u of a b-times binned
        tomogram covers stack pixels b u to b u + b - 1, its centre b u + (b - 1)/2;
        RELION centres on half the size.
        """

        nz = self.shape_zyx[0]
        x, y, z = (float(value) for value in position_xyz)
        standard = np.array([x, y, (nz - 1) - z])
        stack_px = self.binning * standard + (self.binning - 1) / 2
        return (stack_px - np.array(self.size_xyz()) / 2.0) * self.stack_pixel_angstrom


def relion_pose(
    *,
    box_start_xyz: Sequence[float],
    box: int,
    dynamo_angles_deg: Sequence[float],
    dynamo_shift_px: Sequence[float],
    placement: TomogramPlacement,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """rlnCenteredCoordinate*Angst and rlnAngleRot/Tilt/Psi of a particle.

    The particle is a box of `box` voxels cut from the canonical tomogram at
    `box_start_xyz` and aligned with Dynamo's angles and shift about N/2, as the
    pipeline's MRA runs record them. RELION's box centre is int(N/2), Dynamo's
    N/2, the same voxel for the even boxes MRA works on (dynamo_search refuses
    odd ones). An odd box would put half a voxel between them, which nothing has
    checked against RELION, so it is refused.
    """

    if type(box) is not int or box < 2 or box % 2:
        raise ValueError(f"the box must be an even integer, as MRA's are; got {box!r}")
    # dynamo_search is untyped; euler_matrix returns Dynamo's 3 x 3 float matrix M.
    dynamo: np.ndarray = euler_matrix(np.asarray(dynamo_angles_deg, dtype=float))  # type: ignore[no-untyped-call]
    matrix = dynamo.T
    centre = (
        np.asarray(box_start_xyz, dtype=float) + box // 2 + np.asarray(dynamo_shift_px, dtype=float)
    )
    angles = matrix_to_relion_euler(standard_frame_matrix(matrix))
    return placement.centred_angstrom(centre), angles


PARTICLE_LABELS = (
    "rlnTomoName",
    "rlnTomoParticleName",
    "rlnCenteredCoordinateXAngst",
    "rlnCenteredCoordinateYAngst",
    "rlnCenteredCoordinateZAngst",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "rlnClassNumber",
)


@dataclass(frozen=True)
class RelionParticle:
    tomogram: str
    number: int  # from 1 within its tomogram: rlnTomoParticleName "<tomogram>/<number>"
    centred_angstrom: tuple[float, float, float]
    angles_deg: tuple[float, float, float]
    class_number: int = 1
    half_set: int | None = None


def write_particles(path: Path, particles: Sequence[RelionParticle]) -> Path:
    """`data_particles` as RELION 5 reads it straight after picking: no optics
    table (RELION makes one per tomogram from tomograms.star); rlnRandomSubset
    for every particle or for none."""

    if not particles:
        raise ValueError("no particles to write")
    halves = {particle.half_set for particle in particles}
    if None in halves and len(halves) > 1:
        raise ValueError("half sets must be given for every particle or for none")
    if not halves <= {None, 1, 2}:
        raise ValueError("rlnRandomSubset is 1 or 2")
    names = [f"{particle.tomogram}/{particle.number}" for particle in particles]
    if len(set(names)) != len(names):
        raise ValueError("two particles share a name")
    labels = PARTICLE_LABELS + (() if halves == {None} else ("rlnRandomSubset",))
    rows = []
    for particle, name in zip(particles, names, strict=True):
        row: list[object] = [
            particle.tomogram,
            name,
            *(float(value) for value in particle.centred_angstrom),
            *(float(value) for value in particle.angles_deg),
            particle.class_number,
        ]
        if particle.half_set is not None:
            row.append(particle.half_set)
        rows.append(tuple(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(star_text([("particles", labels, rows)]))
    return path


# A series from the project's registry ----------------------------------------


@dataclass(frozen=True)
class SeriesSources:
    """What a tomogram of the registry was made from."""

    name: str
    tomogram: Artifact
    alignment: TiltAlignment
    raw_stack: Artifact
    final_stack: Artifact
    ctf_model: Artifact | None
    manifest: TiltSeriesManifest


def _ancestors(registry: ArtifactRegistry, artifact: Artifact) -> list[Artifact]:
    seen: dict[str, Artifact] = {}
    queue = list(artifact.parent_ids)
    while queue:
        parent_id = queue.pop(0)
        if parent_id in seen:
            continue
        try:
            parent = registry.get(parent_id)
        except KeyError:
            continue
        seen[parent_id] = parent
        queue.extend(parent.parent_ids)
    return list(seen.values())


def _fine_alignment(artifact: Artifact) -> TiltAlignment | None:
    if artifact.kind is not ArtifactKind.ALIGNMENT:
        return None
    try:
        solved = TiltAlignment.model_validate_json(artifact.path.read_text())
    except (OSError, ValueError):
        return None
    return solved if solved.stage == "fine" else None


def _only(found: list[Artifact], what: str, tomogram: Artifact) -> Artifact:
    if len(found) != 1:
        names = ", ".join(item.id for item in found) or "none"
        raise ValueError(f"expected one {what} among {tomogram.id}'s ancestors, found {names}")
    return found[0]


def series_sources(project: Path, name: str, tomogram_id: str | None = None) -> SeriesSources:
    """The tomogram (by default `<name>:tomogram:fine`) and what it was made from."""

    registry = ArtifactRegistry.load(project / "artifacts.json")
    tomogram = registry.get(tomogram_id or f"{name}:tomogram:fine")
    ancestors = _ancestors(registry, tomogram)
    fine = [artifact for artifact in ancestors if _fine_alignment(artifact) is not None]
    alignment_artifact = _only(fine, "fine alignment", tomogram)
    raw = _only(
        [a for a in ancestors if a.kind is ArtifactKind.TILT_STACK], "raw tilt stack", tomogram
    )
    final = [
        a
        for a in ancestors
        if a.kind is ArtifactKind.ALIGNED_TILT_STACK
        and a.parameters.get("purpose") == "final_alignment"
    ]
    final_stack = _only(final, "final aligned stack", tomogram)
    models = [a for a in ancestors if a.kind is ArtifactKind.CTF_MODEL]
    manifest = TiltSeriesManifest.model_validate_json(
        (project / "manifests" / f"{name}.json").read_text()
    )
    alignment = _fine_alignment(alignment_artifact)
    assert alignment is not None
    return SeriesSources(
        name=name,
        tomogram=tomogram,
        alignment=alignment,
        raw_stack=raw,
        final_stack=final_stack,
        ctf_model=_only(models, "CTF model", tomogram) if models else None,
        manifest=manifest,
    )


def acquisition_ranks(manifest: TiltSeriesManifest) -> tuple[dict[int, int], str]:
    """Each view's place in acquisition, by z value, and what it was read from:
    the mdoc's DateTime when every view has one, else the z values' order."""

    stamps = {}
    for image in manifest.images:
        text = " ".join(str(image.metadata.get("DateTime", "")).split())
        try:
            stamps[image.z_value] = datetime.strptime(text, "%d-%b-%y %H:%M:%S")
        except ValueError:
            break
    else:
        if len(set(stamps.values())) == len(stamps):
            order = sorted(stamps, key=stamps.__getitem__)
            return {z: rank for rank, z in enumerate(order)}, "mdoc DateTime"
    order = sorted(image.z_value for image in manifest.images)
    return {z: rank for rank, z in enumerate(order)}, "z value order (no usable DateTime)"


@dataclass(frozen=True)
class ExportedSeries:
    name: str
    placement: TomogramPlacement
    raw_pixel_angstrom: float
    reconstruction: ImodReconstruction
    views: tuple[RelionView, ...]
    included_z_values: tuple[int, ...]
    optics: RelionOptics
    dropped_px_max: float
    notes: tuple[str, ...]


def export_series(
    sources: SeriesSources,
    folder: Path,
    *,
    stack_binning: int | None = None,
    dose_per_tilt: float | None = None,
) -> ExportedSeries:
    """Write `<folder>/tilt_series/<name>.mrcs` and return the series' views.

    The stack holds the raw tilt images the tomogram was reconstructed from, in
    the alignment's order, binned `stack_binning` times as the final stack bins
    them (by default as much as the tomogram is). CTF from the series' CTF model
    when it has one; dose from the mdoc when every view states one, else
    `dose_per_tilt`; every fallback is named in `notes`.
    """

    tomogram = sources.tomogram
    tomogram_binning = int(tomogram.binning or sources.final_stack.binning or 1)
    stack_binning = tomogram_binning if stack_binning is None else stack_binning
    if type(stack_binning) is not int or stack_binning < 1 or tomogram_binning % stack_binning:
        raise ValueError(
            f"the stack's binning must divide the tomogram's ({tomogram_binning}),"
            f" got {stack_binning!r}"
        )
    raw_px = float(sources.final_stack.parameters["raw_pixel_spacing_angstrom"])
    stack_px = raw_px * stack_binning
    included = tomogram.parameters.get("included_z_values") or [
        t.z_value for t in sources.alignment.transforms
    ]
    included_set = {int(z) for z in included}
    transforms = [t for t in sources.alignment.transforms if t.z_value in included_set]
    if len(transforms) != len(included_set):
        raise ValueError(f"{sources.name}: the tomogram's views are not all in the fine alignment")
    images = {image.z_value: image for image in sources.manifest.images}
    index_by_z = {image.z_value: i for i, image in enumerate(sources.manifest.images)}

    stack = folder / "tilt_series" / f"{sources.name}.mrcs"
    stack.parent.mkdir(parents=True, exist_ok=True)
    write_binned_mrc_stack(
        sources.raw_stack.path,
        stack,
        binning=stack_binning,
        pixel_spacing_angstrom=stack_px,
        section_indices=[index_by_z[t.z_value] for t in transforms],
    )
    with mrcfile.open(stack, permissive=True, header_only=True) as header:
        image_xy = (int(header.header.nx), int(header.header.ny))

    nz, ny, nx = (int(size) for size in tomogram.shape or ())
    placement = TomogramPlacement((nz, ny, nx), tomogram_binning // stack_binning, stack_px)
    reconstruction = placement.reconstruction(
        x_axis_tilt_deg=float(tomogram.parameters.get("x_axis_tilt_deg", 0.0)),
        z_shift_px=float(tomogram.parameters.get("z_shift_px", 0.0)),
    )

    pre, defocus, optics, notes = _rows_inputs(sources, dose_per_tilt, raw_px)

    views = []
    dropped = []
    for section, transform in enumerate(transforms, start=1):
        scaled = transform.model_copy(
            update={
                "shift_x_px": transform.shift_x_px / stack_binning,
                "shift_y_px": transform.shift_y_px / stack_binning,
            }
        )
        tilt = relion_tilt(
            scaled,
            image_size_xy=image_xy,
            reconstruction=reconstruction,
            pixel_size_angstrom=stack_px,
        )
        dropped.append(tilt.dropped_px)
        image = images[transform.z_value]
        if transform.z_value in defocus:
            u, v, azimuth = defocus[transform.z_value]
            angle = defocus_angle_in_raw_frame(azimuth, transform)
        else:
            u = v = 10000.0 * float(image.defocus_um or 0.0)
            angle = 0.0
        views.append(
            RelionView(
                image=f"{section}@tilt_series/{stack.name}",
                nominal_tilt_deg=image.tilt_angle_deg,
                pre_exposure=pre[transform.z_value],
                defocus_u_angstrom=u,
                defocus_v_angstrom=v,
                defocus_angle_deg=angle,
                tilt=tilt,
            )
        )
    return ExportedSeries(
        name=sources.name,
        placement=placement,
        raw_pixel_angstrom=raw_px,
        reconstruction=reconstruction,
        views=tuple(views),
        included_z_values=tuple(t.z_value for t in transforms),
        optics=optics,
        dropped_px_max=max(dropped),
        notes=tuple(notes),
    )


def _rows_inputs(
    sources: SeriesSources, dose_per_tilt: float | None, raw_px: float
) -> tuple[dict[int, float], dict[int, tuple[float, float, float]], RelionOptics, list[str]]:
    """Each view's pre-exposure and defocus, the optics, and what they fall back on."""

    notes: list[str] = []
    ranks, order_source = acquisition_ranks(sources.manifest)
    if not order_source.startswith("mdoc"):
        notes.append(f"acquisition order from the {order_source}")
    doses = {image.z_value: image.exposure_dose for image in sources.manifest.images}
    if all(dose is not None and dose > 0 for dose in doses.values()):
        dose_values = {z: float(dose or 0.0) for z, dose in doses.items()}
    elif dose_per_tilt is not None:
        dose_values = {z: float(dose_per_tilt) for z in doses}
        notes.append(f"dose {dose_per_tilt} e/A^2 per tilt as given, the mdoc states none")
    else:
        dose_values = {z: 0.0 for z in doses}
        notes.append("no dose: every pre-exposure is 0, so RELION weights no tilt by dose")
    pre = pre_exposures(dose_values, ranks)
    defocus = _defocus_by_z(sources, notes)
    if sources.ctf_model is not None:
        optics = RelionOptics(
            voltage_kv=float(sources.ctf_model.parameters["voltage_kv"]),
            spherical_aberration_mm=float(sources.ctf_model.parameters["spherical_aberration_mm"]),
            amplitude_contrast=float(sources.ctf_model.parameters["amplitude_contrast"]),
            original_pixel_size_angstrom=raw_px,
        )
    else:
        optics = RelionOptics(300.0, 2.7, 0.1, raw_px)
        notes.append("optics assumed (300 kV, Cs 2.7 mm, amplitude contrast 0.1): no CTF model")
    return pre, defocus, optics, notes


def series_notes(sources: SeriesSources, *, dose_per_tilt: float | None = None) -> list[str]:
    """What `export_series` would fall back on for this series, written nowhere: for a
    dry run, so that a missing dose can be given before the stack is written."""

    raw_px = float(sources.final_stack.parameters["raw_pixel_spacing_angstrom"])
    return _rows_inputs(sources, dose_per_tilt, raw_px)[3]


def _defocus_by_z(
    sources: SeriesSources, notes: list[str]
) -> dict[int, tuple[float, float, float]]:
    """Defocus U and V (A) and the azimuth on the fitted stack, by z value."""

    model = sources.ctf_model
    if model is None:
        notes.append("defocus nominal (the mdoc's), no astigmatism: no CTF model")
        return {}
    backend = str(model.parameters.get("backend", ""))
    if "ctffind" not in backend:
        notes.append(f"defocus from {backend} without astigmatism: only CTFFIND's angle is read")
    z_values = [int(z) for z in model.parameters.get("z_values") or []]
    result = {}
    for entry in read_imod_defocus_file(model.path):
        if not 1 <= entry.view_start <= len(z_values):
            raise ValueError(f"{model.path}: view {entry.view_start} has no z value")
        second = entry.second_defocus_nm
        azimuth = entry.astigmatism_axis_deg
        if second is None or azimuth is None or "ctffind" not in backend:
            result[z_values[entry.view_start - 1]] = (10.0 * entry.mean_defocus_nm,) * 2 + (0.0,)
        else:
            result[z_values[entry.view_start - 1]] = (
                10.0 * entry.defocus_nm,
                10.0 * second,
                azimuth,
            )
    if model.parameters.get("input_stack_id") != sources.final_stack.id:
        notes.append(
            "astigmatism angles turned by the fine alignment's rotation, though the CTF was fitted"
            f" on {model.parameters.get('input_stack_id')}"
        )
    return result


# A class of an MRA run --------------------------------------------------------


def box_start(pick_xyz: Sequence[float], box: int) -> tuple[int, int, int]:
    """The corner (x, y, z) of the box extraction cut around a pick, as
    `subtomogram._crop_start` and `_unclipped_start` place it: the rounded pick
    less half the box (Python's round, halves to even)."""

    half = box // 2
    x, y, z = (int(round(float(value))) - half for value in pick_xyz)
    return x, y, z


@dataclass(frozen=True)
class ClassMember:
    tag: int
    series: str
    stack_id: str
    pick_xyz: tuple[float, float, float]
    angles_deg: tuple[float, float, float]  # Dynamo's, after the iteration
    shifts_px: tuple[float, float, float]  # Dynamo's, about N/2, in box voxels
    cc: float


@dataclass(frozen=True)
class RunClass:
    """One class of an MRA run of the registry, as the export reads it."""

    run: Path
    project: Path
    iteration: int
    channel: int
    box: int
    pixel_angstrom: float
    members: tuple[ClassMember, ...]
    decoys_left_out: int
    not_agreed: int
    agreed_with: str | None
    average: Path
    mask: Path
    record_fingerprints: dict[str, str]


def _registry_run(run: Path) -> bool:
    plan = run / "plan.json"
    if not plan.is_file():
        return False
    return isinstance(json.loads(plan.read_text()).get("particles"), dict)


def _class_members(run: Path, channel: int, iteration: int | None) -> tuple[Any, int, set[int]]:
    from cryoet_pipeline.reports.classification_reader import read_classification

    classification = read_classification(run, compute_initial_average=False)
    if not classification.iterations:
        raise ValueError(f"{run}: no finished iteration")
    numbers = [result.number for result in classification.iterations]
    number = numbers[-1] if iteration is None else iteration
    if number not in numbers:
        raise ValueError(f"{run}: iteration {number} has not finished; finished: {numbers}")
    classes = classification.iterations[numbers.index(number)].classes
    if channel not in classes:
        raise ValueError(
            f"{run}: iteration {number} has no class {channel}; it has {sorted(classes)}"
        )
    return classification, number, {int(tag) for tag in classes[channel].members}


def read_run_class(
    run: Path,
    channel: int,
    *,
    iteration: int | None = None,
    agree: tuple[Path, int] | None = None,
) -> RunClass:
    """The members of `channel` after `iteration` (the last finished by default) of an
    MRA run of the registry, decoys left out, with their poses of that iteration and
    their picks; with `agree`, only those also in that run's class after its last
    finished iteration, as scripts/consensus_halves.py takes them."""

    from cryoet_pipeline.mra_runs import PARTICLES, node_folder
    from cryoet_pipeline.particle_tables import read_particle_table
    from cryoet_pipeline.runs import completed

    run = Path(run)
    if not _registry_run(run):
        raise ValueError(f"{run} is not an MRA run of the registry (plan.json with its particles)")
    classification, number, members = _class_members(run, channel, iteration)
    decoys = members & {int(tag) for tag in classification.decoys}
    members -= decoys
    agreed_with = None
    not_agreed = 0
    if agree is not None:
        other, other_channel = agree
        _, _, others = _class_members(Path(other), other_channel, None)
        not_agreed = len(members - others)
        members &= others
        agreed_with = f"{Path(other).name}:{other_channel}"
    if not members:
        raise ValueError(f"{run}: class {channel} of iteration {number} has no particle to export")

    assert classification.poses is not None
    poses = classification.poses((number, channel))
    row_of = {int(tag): row for row, tag in enumerate(poses.tags)}
    setup, about = read_particle_table(node_folder(run, 0) / PARTICLES)
    setup_row = {int(tag): row for row, tag in enumerate(setup["particle_id"])}
    chosen = []
    for tag in sorted(members):
        row, srow = row_of[tag], setup_row[tag]
        chosen.append(
            ClassMember(
                tag=tag,
                series=str(setup["tilt_series_id"][srow]),
                stack_id=str(setup["stack_id"][srow]),
                pick_xyz=tuple(float(setup[f"position_{axis}_px"][srow]) for axis in "xyz"),  # type: ignore[arg-type]
                angles_deg=tuple(float(value) for value in poses.angles[row]),  # type: ignore[arg-type]
                shifts_px=tuple(float(value) for value in poses.shifts[row]),  # type: ignore[arg-type]
                cc=float(poses.cc[row]),
            )
        )
    fingerprints = {}
    for name, folder in (
        ("mra_setup", node_folder(run, 0)),
        ("mra_iteration", node_folder(run, number)),
    ):
        record = completed(folder)
        if record is None:
            raise ValueError(f"{folder} has no finished run record")
        fingerprints[name] = record.fingerprint
    average = node_folder(run, number) / f"average_{channel}.mrc"
    if not average.is_file():
        raise ValueError(f"{average} is missing")
    return RunClass(
        run=run,
        project=Path(str(about["project"])),
        iteration=number,
        channel=channel,
        box=int(about["box_px"]),
        pixel_angstrom=float(about["pixel_size_angstrom"]),
        members=tuple(chosen),
        decoys_left_out=len(decoys),
        not_agreed=not_agreed,
        agreed_with=agreed_with,
        average=average,
        mask=node_folder(run, 0) / "mask.mrc",
        record_fingerprints=fingerprints,
    )


def particle_centre(member: ClassMember, box: int) -> np.ndarray:
    """Where the class's reference centre lies in the canonical tomogram (x, y, z voxels):
    the box corner plus N/2 plus the Dynamo shift."""

    return np.asarray(box_start(member.pick_xyz, box), dtype=float) + box // 2 + member.shifts_px


def remove_duplicates(
    centres: Mapping[int, np.ndarray],
    series: Mapping[int, str],
    score: Mapping[int, float],
    min_distance: float,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Particles whose centres came within `min_distance` of a better-scoring one of
    the same series are left out: what they hold is already in the set. Returns the
    kept tags and each left-out tag with the one it duplicates."""

    from scipy.spatial import cKDTree  # type: ignore[import-untyped]

    kept: list[int] = []
    dropped: list[tuple[int, int]] = []
    by_series: dict[str, list[int]] = {}
    for tag in centres:
        by_series.setdefault(series[tag], []).append(tag)
    for name in sorted(by_series):
        tags = sorted(by_series[name], key=lambda tag: (-score[tag], tag))
        points = np.array([centres[tag] for tag in tags], dtype=float)
        tree = cKDTree(points)
        taken = np.zeros(len(tags), dtype=bool)
        for index, tag in enumerate(tags):
            near = [
                other
                for other in tree.query_ball_point(points[index], r=min_distance)
                if other != index and taken[other]
            ]
            if near:
                dropped.append((tag, tags[min(near)]))
            else:
                taken[index] = True
                kept.append(tag)
    return sorted(kept), dropped


def assign_halves(
    centres: Mapping[int, np.ndarray],
    series: Mapping[int, str],
    link_distance: float,
    seed: int,
) -> tuple[dict[int, int], int]:
    """rlnRandomSubset for each particle, groups of neighbours kept together.

    Particles of one series whose centres lie within `link_distance` of each other
    (a box: their boxes overlap) are linked, and a linked group goes to one half, so
    no half holds the other's density. Within each series the groups are shuffled
    with `seed` and each goes to the half with fewer particles so far. Returns the
    halves and how many groups held more than one particle.
    """

    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    halves: dict[int, int] = {}
    linked_groups = 0
    by_series: dict[str, list[int]] = {}
    for tag in centres:
        by_series.setdefault(series[tag], []).append(tag)
    for name in sorted(by_series):
        tags = sorted(by_series[name])
        points = np.array([centres[tag] for tag in tags], dtype=float)
        pairs = cKDTree(points).query_pairs(r=link_distance)
        ordered = [[tags[index] for index in group] for group in _linked(len(tags), pairs)]
        linked_groups += sum(1 for group in ordered if len(group) > 1)
        counts = {1: 0, 2: 0}
        for group in (ordered[index] for index in rng.permutation(len(ordered))):
            half = 1 if counts[1] <= counts[2] else 2
            for tag in group:
                halves[tag] = half
            counts[half] += len(group)
    return halves, linked_groups


def _linked(count: int, pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """The groups `pairs` link among `count` items (union-find), each sorted, by first item."""

    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in pairs:
        parent[root(first)] = root(second)
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values())


def optimisation_set_text(particles: str, tomograms: str) -> str:
    """RELION 5's optimisation set: which particles and tomograms a job reads."""

    for value in (particles, tomograms):
        if not value or any(character.isspace() for character in value):
            raise ValueError(f"not a STAR file name: {value!r}")
    return "\n".join(
        [
            "",
            STAR_VERSION_LINE,
            "",
            "data_",
            "",
            f"_rlnTomoParticlesFile {particles}",
            f"_rlnTomoTomogramsFile {tomograms}",
            "",
            "",
        ]
    )
