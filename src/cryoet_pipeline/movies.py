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


@dataclass(frozen=True)
class MovieInfo:
    format: MovieFormat
    shape: tuple[int, int, int]  # (frames, y, x)
    # From the MRC header. A TIFF has none: its resolution tags are a print
    # density, not a detector pixel, so the pixel size has to be given.
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
        pixel_spacing_angstrom=None,
    )


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

    spacings = [(path, inspect_movie(path).pixel_spacing_angstrom) for path in paths]
    without = [path for path, spacing in spacings if spacing is None]
    if without:
        raise ValueError(
            f"{label}: {without[0]} has no pixel size in its header (TIFF movies "
            "never do); give the raw pixel size explicitly"
        )
    reference_path, reference = spacings[0]
    assert reference is not None
    for path, spacing in spacings[1:]:
        assert spacing is not None
        if not isclose(spacing, reference, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"{label}: inconsistent raw MRC pixel spacing; "
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
