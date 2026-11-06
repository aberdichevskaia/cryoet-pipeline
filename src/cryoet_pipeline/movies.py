"""Raw tilt movies on disk: their format, their frame count, and their frames.

MRC, TIFF and EER movies are known. The format is taken from the file suffix,
because that is what MotionCor3 is told as well (`-InMrc`, `-InTiff`, `-InEer`):
a movie whose suffix lies would be misread there too, so it is refused here
rather than sniffed.

Both readers return the frames as stored, row 0 first. Whether a TIFF and an MRC
written from the same exposure store their rows in the same order depends on the
software that wrote them, and is not checked here; a y flip between the two
mirrors the tomogram.

An EER movie (Thermo Fisher's Electron Event Representation, from Falcon
cameras) is a BigTIFF with one page per EER frame: hundreds of frames a tilt,
each a run-length code of the electron events, not an image. Its frame count,
size and pixel are read here from the page tags and the acquisition metadata
alone, which tifffile does without decoding a frame. Its frames are not read:
MotionCor3 decodes and sums them itself (`-InEer`), and nothing else here
needs them. Decoding would take imagecodecs and summing in groups as it
reads, since 728 frames of 4096 x 4096 are 12 GB as bytes.
"""

from __future__ import annotations

import struct
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
    EER = "eer"


_SUFFIXES = {
    ".mrc": MovieFormat.MRC,
    ".mrcs": MovieFormat.MRC,
    ".tif": MovieFormat.TIFF,
    ".tiff": MovieFormat.TIFF,
    ".eer": MovieFormat.EER,
}

# TIFF Compression codes of EER frames: the three versions of the encoding.
_EER_COMPRESSIONS = frozenset({65000, 65001, 65002})


# Angstroms per unit of the TIFF ResolutionUnit tag: 2 is the inch, the TIFF
# default, and 3 the centimetre. 1, no unit, says nothing about size.
_ANGSTROM_PER_RESOLUTION_UNIT = {2: 2.54e8, 3: 1e8}
# A detector pixel is a fraction of an angstrom to tens of angstroms. Anything
# coarser is a print density, such as the 72 dpi many writers put in (3.5e6 A),
# and says nothing about the detector.
_MAX_TIFF_PIXEL_ANGSTROM = 100.0


@dataclass(frozen=True)
class EerInfo:
    """What an EER file's tags and acquisition metadata say about it."""

    # The TIFF Compression code: 65000, 65001 or 65002.
    compression: int
    # The TIFF Orientation tag. Every EMPIAR-11830 movie and gain says 2; it is
    # recorded, not applied, as MotionCor3 does not apply it either.
    orientation_tag: int | None
    exposure_time_s: float | None
    total_dose_e_per_pixel: float | None
    camera: str | None
    # The gain reference the acquisition software named, which is not always
    # the one deposited with the data (EMPIAR-11830).
    gain_reference_named: str | None


@dataclass(frozen=True)
class MovieInfo:
    format: MovieFormat
    # (frames, y, x). For EER: its EER frames, and the sensor's size, which is
    # what a sum at EER sampling 1 has.
    shape: tuple[int, int, int]
    # From the MRC header, the resolution tags SerialEM writes into a TIFF, or
    # the sensor pixel size in an EER file's metadata; None when the file does
    # not say.
    pixel_spacing_angstrom: float | None
    eer: EerInfo | None = None


def movie_format(path: Path) -> MovieFormat:
    suffix = path.suffix.lower()
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
    if fmt is MovieFormat.EER:
        return _inspect_eer(path)

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


def _inspect_eer(path: Path) -> MovieInfo:
    """An EER movie's frame count, sensor size, pixel and metadata, from its tags.

    The page directories of an EER file sit at its end (EMPIAR-11830), so a
    copy that stopped short fails to open here. One whose pages and metadata
    disagree on the frame count is refused too. So is a file that starts with
    an integrated image, which tifffile keeps apart as a series of its own:
    MotionCor3 counts every page as a frame.
    """

    tifffile = _tifffile()
    try:
        with tifffile.TiffFile(path) as tiff:
            is_eer = bool(tiff.is_eer)
            series = list(tiff.series) if is_eer else []
            names = [str(item.name) for item in series]
            if is_eer and len(series) == 1:
                keyframe = series[0].keyframe
                compression = int(keyframe.compression)
                shape = tuple(int(n) for n in series[0].shape)
                orientation = keyframe.tags.valueof(274)
                metadata = dict(tiff.eer_metadata or {})
    except (struct.error, EOFError, IndexError, KeyError, tifffile.TiffFileError) as exc:
        # tifffile raises struct.error, not its own error, on some cut files.
        raise ValueError(
            f"{path}: its TIFF structure cannot be read ({exc}); an incomplete copy?"
        ) from exc
    if not is_eer:
        raise ValueError(
            f"{path}: no EER acquisition metadata (TIFF tag 65001); not an EER movie"
        )
    if len(names) != 1:
        raise ValueError(
            f"{path}: holds {', '.join(names)}, not EER frames alone; MotionCor3 would "
            "read every page as an EER frame"
        )
    if compression not in _EER_COMPRESSIONS:
        raise ValueError(f"{path}: TIFF compression {compression} is not an EER encoding")
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) != 3:
        raise ValueError(f"{path}: expected EER frames (frames, y, x), got {shape}")
    declared = metadata.get("numberOfFrames")
    if declared is not None and int(declared) != shape[0]:
        raise ValueError(
            f"{path}: {shape[0]} EER frames on disk, its metadata declares {declared}; "
            "an incomplete copy?"
        )
    return MovieInfo(
        format=MovieFormat.EER,
        shape=(shape[0], shape[1], shape[2]),
        pixel_spacing_angstrom=_eer_pixel_spacing_angstrom(path, metadata),
        eer=EerInfo(
            compression=compression,
            orientation_tag=None if orientation is None else int(orientation),
            exposure_time_s=_optional_number(metadata.get("exposureTime")),
            total_dose_e_per_pixel=_eer_dose_e_per_pixel(metadata),
            camera=None if metadata.get("commercialName") is None
            else str(metadata["commercialName"]),
            gain_reference_named=None if metadata.get("eerGainReference") is None
            else str(metadata["eerGainReference"]),
        ),
    )


def _eer_pixel_spacing_angstrom(path: Path, metadata: dict[str, Any]) -> float | None:
    """The sensor pixel size in an EER file's metadata, in angstroms.

    Thermo Fisher writes it in metres at the specimen: 1.90370344e-10 in the
    EMPIAR-11830 movies, whose mdocs say 1.90 and whose authors used 1.96.
    """

    spacings = []
    for axis in ("width", "height"):
        value = metadata.get(f"sensorPixelSize.{axis}")
        if value is None or metadata.get(f"sensorPixelSize.{axis}.unit") != "m":
            return None
        spacings.append(float(value) * 1e10)
    x_spacing, y_spacing = spacings
    if not isclose(x_spacing, y_spacing, rel_tol=1e-5):
        raise ValueError(
            f"{path}: EER sensor pixel width and height differ, "
            f"{x_spacing:g} and {y_spacing:g} angstrom"
        )
    if not 0 < x_spacing <= _MAX_TIFF_PIXEL_ANGSTROM:
        return None
    return x_spacing


def _optional_number(value: Any) -> float | None:
    return None if value is None else float(value)


def _eer_dose_e_per_pixel(metadata: dict[str, Any]) -> float | None:
    """The movie's dose from an EER file's metadata, when it is given per pixel.

    Thermo Fisher writes it as totalDose, unit "e/pixel": 6.38 to 6.53 on the
    EMPIAR-11830 movies, which the counts in their MotionCor3 sums match (job
    3890116). A dose in another unit, or none, is left out rather than guessed.
    """

    dose = metadata.get("totalDose")
    if dose is None or metadata.get("totalDose.unit") != "e/pixel":
        return None
    value = float(dose)
    return value if np.isfinite(value) and value > 0 else None


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
    """All frames as (frames, y, x), in the file's own dtype. Not for EER."""

    fmt = movie_format(path)
    if fmt is MovieFormat.EER:
        raise ValueError(
            f"{path}: EER frames are decoded by MotionCor3 (motion backend motioncor3), "
            "not read here"
        )
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
