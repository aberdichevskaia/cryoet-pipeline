"""Raw tilt movies on disk: their format, their frame count, and their frames.

MRC and TIFF movies are read. The format is taken from the file suffix, because
that is what MotionCor3 is told as well (`-InMrc` or `-InTiff`): a movie whose
suffix lies would be misread there too, so it is refused here rather than sniffed.
EER is named and refused, so that it fails as "not supported yet" instead of as
an unknown suffix.

Both readers return the frames as stored, row 0 first. Whether a TIFF and an MRC
written from the same exposure store their rows in the same order depends on the
software that wrote them, and is not checked here; a y flip between the two
mirrors the tomogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isclose
from pathlib import Path
from typing import Any, cast

import mrcfile  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from cryoet_pipeline.mrc_validation import validate_complete_mrc


class MovieFormat(StrEnum):
    MRC = "mrc"
    TIFF = "tiff"


_SUFFIXES = {
    ".mrc": MovieFormat.MRC,
    ".mrcs": MovieFormat.MRC,
    ".tif": MovieFormat.TIFF,
    ".tiff": MovieFormat.TIFF,
}


# Angstroms per unit of the TIFF ResolutionUnit tag: 2 is the inch, the TIFF
# default, and 3 the centimetre. 1, no unit, says nothing about size.
_ANGSTROM_PER_RESOLUTION_UNIT = {2: 2.54e8, 3: 1e8}
# A detector pixel is a fraction of an angstrom to tens of angstroms. Anything
# coarser is a print density, such as the 72 dpi many writers put in (3.5e6 A),
# and says nothing about the detector.
_MAX_TIFF_PIXEL_ANGSTROM = 100.0


@dataclass(frozen=True)
class MovieInfo:
    format: MovieFormat
    shape: tuple[int, int, int]  # (frames, y, x)
    # From the MRC header, or from the resolution tags SerialEM writes into a
    # TIFF; None when the file does not say.
    pixel_spacing_angstrom: float | None


def movie_format(path: Path) -> MovieFormat:
    suffix = path.suffix.lower()
    if suffix == ".eer":
        raise ValueError(f"EER movies are not supported yet: {path}")
    try:
        return _SUFFIXES[suffix]
    except KeyError:
        known = ", ".join(sorted(_SUFFIXES))
        raise ValueError(f"unknown movie format {suffix!r} (known: {known}): {path}") from None


def inspect_movie(path: Path) -> MovieInfo:
    """Format, (frames, y, x) and header pixel size, without reading the pixels."""

    fmt = movie_format(path)
    if fmt is MovieFormat.MRC:
        info = validate_complete_mrc(path)
        if len(info.shape) != 3:
            raise ValueError(
                f"{path}: expected multiframe MRC with shape (frames, y, x), got {info.shape}"
            )
        return MovieInfo(
            format=fmt,
            shape=cast(tuple[int, int, int], tuple(info.shape)),
            pixel_spacing_angstrom=info.pixel_spacing_angstrom,
        )

    with _tifffile().TiffFile(path) as tiff:
        shapes = {tuple(int(n) for n in page.shape) for page in tiff.pages}
        frames = len(tiff.pages)
        pixel_spacing = _tiff_pixel_spacing_angstrom(path, tiff.pages[0])
    if len(shapes) != 1:
        raise ValueError(f"{path}: TIFF pages differ in shape: {sorted(shapes)}")
    (frame_shape,) = shapes
    if len(frame_shape) != 2:
        raise ValueError(f"{path}: expected 2D TIFF pages, got page shape {frame_shape}")
    if frames < 2:
        # A one-page TIFF is an image, not a movie; the MRC reader refuses a 2D
        # file for the same reason.
        raise ValueError(f"{path}: expected a multiframe TIFF movie, found {frames} page")
    return MovieInfo(
        format=fmt,
        shape=(frames, frame_shape[0], frame_shape[1]),
        pixel_spacing_angstrom=pixel_spacing,
    )


def _tiff_pixel_spacing_angstrom(path: Path, page: Any) -> float | None:
    """The pixel size in the TIFF resolution tags, read as IMOD's `header` reads it.

    SerialEM records the pixel there as pixels per centimetre or per inch, and
    IMOD turns it back into angstroms; both were measured with IMOD 5.1.12 on
    synthetic files. Without a unit, without the tags, or at a print density,
    the file says nothing, and IMOD then shows 1.0, its "unknown"; this returns
    None so that the size has to be given instead.
    """

    x_tag = page.tags.get("XResolution")
    y_tag = page.tags.get("YResolution")
    if x_tag is None or y_tag is None:
        return None
    unit_tag = page.tags.get("ResolutionUnit")
    unit = 2 if unit_tag is None else int(unit_tag.value)
    angstrom_per_unit = _ANGSTROM_PER_RESOLUTION_UNIT.get(unit)
    if angstrom_per_unit is None:
        return None
    spacings = []
    for tag in (x_tag, y_tag):
        numerator, denominator = tag.value
        if numerator <= 0 or denominator <= 0:
            return None
        spacings.append(angstrom_per_unit * denominator / numerator)
    x_spacing, y_spacing = spacings
    if not isclose(x_spacing, y_spacing, rel_tol=1e-5):
        raise ValueError(
            f"{path}: TIFF X and Y resolution give different pixel sizes, "
            f"{x_spacing:g} and {y_spacing:g} angstrom"
        )
    if x_spacing > _MAX_TIFF_PIXEL_ANGSTROM:
        return None
    return float(x_spacing)


def read_movie(path: Path) -> NDArray[Any]:
    """All frames as (frames, y, x), in the file's own dtype."""

    fmt = movie_format(path)
    if fmt is MovieFormat.MRC:
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data).copy()
    else:
        with _tifffile().TiffFile(path) as tiff:
            data = np.stack([page.asarray() for page in tiff.pages])
    if data.ndim != 3:
        raise ValueError(
            f"expected a multiframe movie with shape (frames, y, x), got {data.shape}: {path}"
        )
    return data


def common_pixel_spacing(label: str, paths: list[Path]) -> float:
    """The one pixel size the movies' headers agree on, or a refusal naming why not."""

    return agreed_pixel_spacing(
        label, [(path, inspect_movie(path).pixel_spacing_angstrom) for path in paths]
    )


def agreed_pixel_spacing(label: str, spacings: list[tuple[Path, float | None]]) -> float:
    """As `common_pixel_spacing`, for header values already read."""

    without = [path for path, spacing in spacings if spacing is None]
    if without:
        raise ValueError(
            f"{label}: {without[0]} has no pixel size in its header; give the raw "
            "pixel size explicitly"
        )
    reference_path, reference = spacings[0]
    assert reference is not None
    for path, spacing in spacings[1:]:
        assert spacing is not None
        if not isclose(spacing, reference, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"{label}: inconsistent raw pixel spacing in the movie headers; "
                f"{reference_path} has {reference} angstrom, {path} has {spacing} angstrom"
            )
    return reference


def _tifffile() -> Any:
    try:
        import tifffile
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reading TIFF movies needs the tifffile package: pip install tifffile"
        ) from exc
    return tifffile
