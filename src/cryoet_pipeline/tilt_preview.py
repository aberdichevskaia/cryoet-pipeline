"""Raw tilts read before any lane: are there gold beads, and which way does the tilt axis run.

A lane config needs both before IMOD has run: block A of example.toml (beads,
`[fiducials] fiducial_diameter_nm`) or block B (patches), and `[align]
tilt_axis_angle` wherever no SerialEM mdoc gives it. scripts/show_tilts.py
draws what this module measures into a PDF, for a site with no 3dmod at hand.
Numpy and scipy only; nothing is written here.

**Orientation.** Every picture is laid out as the motion-corrected stack will
hold it, and drawn as 3dmod shows that stack: array row 0 at the bottom. An MRC
movie or stack keeps its rows. A TIFF movie is turned upside down, because
MotionCor3 reads a TIFF bottom row first (docs/motioncor3_backend.md, read from
its sources, not measured here). Angles are IMOD's, which `[align]
tilt_axis_angle` takes: from the picture's y axis, counterclockwise positive.

**The axis.** A thin specimen tilted by θ projects shortened by cos θ across
the tilt axis and unchanged along it. Of two neighbouring high tilts, the higher
one therefore matches the lower once stretched by cos θa / cos θb across the
axis, and worse when stretched across any other direction. Each pair of the
highest tilts on either side scans that direction, 2° apart and then 0.25°
around the best, by the peak of their cross-correlation; the estimate is the
median over pairs. φ and φ + 180° stretch alike, so the estimate is a line
without a sign. The two signs give tomograms of opposite hand, which no single
picture shows; an mdoc's axis has the sign.

Measured on 24 November 2026 against the mdocs' axes: EMPIAR-10164 TS_01 85.8°
and TS_43 85.2° (MRC frames summed as they are; mdoc 85.3°, tiltalign solved
85.16° on TS_01), EMPIAR-11538 TS_1002 82.6° and TS_1005 83.9° (stacks; mdoc
83.6°). The same tilts transposed read 4.75° and 6.1° (90° - φ: 4.7°, 6.4°),
and upside down 94.75° and 96.1° (180° - φ: 94.7°, 96.4°).

**The beads.** Gold is far darker than anything biological at the same size, and
round. A Laplacian of Gaussian, scale-normalised, finds dark round spots from 4
to 20 nm across on the tilt nearest 0°; a spot counts when it stands
`SPOT_MIN_STRENGTH` robust standard deviations above the filtered picture and
its centre is at most `SPOT_MAX_DARKNESS` of the mean of a ring around it. Its
diameter is the filter size that answers it best, the nearest of
`SPOT_DIAMETERS_NM`, which a defocus fringe can push up a step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore[import-untyped]

from cryoet_pipeline.backends.alignment import manifest_tilt_axis_angle
from cryoet_pipeline.models import SeriesEntry, TiltImage, TiltSeriesManifest
from cryoet_pipeline.movies import MovieFormat, movie_format

COARSE_ANGSTROM = 10.0
"""Pixel the pictures and both measurements work at: a 10 nm bead is 10 pixels across."""

FINE_ANGSTROM = 3.0
"""Pixel of the close-ups of the spots, cut from the tilt nearest 0°."""

SCAN_MAX_PX = 512
"""The axis scan bins its pictures further until neither side is longer than this."""

AXIS_MIN_TILT_DEG = 20.0
"""A pair's higher tilt must reach this: below it the stretch is under 6% and says little."""

AXIS_VIEWS_PER_SIDE = 4
"""The highest tilts read on each side of 0°, which give three neighbouring pairs a side."""

AXIS_COARSE_STEP_DEG = 2.0
AXIS_FINE_STEP_DEG = 0.25
AXIS_AGREEMENT_DEG = 10.0
"""A pair agrees with the estimate when its own best lies within this of it."""

SPOT_DIAMETERS_NM = (4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
SPOT_MIN_STRENGTH = 6.0
SPOT_MAX_DARKNESS = 0.8
"""Gold at 0° measured 0.38-0.46 on EMPIAR-10164; a lamella's darkest spots 0.68-0.9."""

GALLERY_SPOTS = 24

DARK_VIEW_FRACTION = 0.1
"""A tilt whose mean is under this share of the 0° tilt's is empty or blocked, not thick:
at 60° a slab is twice as thick, and EMPIAR-10164's TS_01 kept 53% of its 0° mean at +60°,
while its -60° tilt held 0.4% (2.75 of 624, its mdoc's MinMaxMean)."""

_BAND_SIGMAS_PX = (1.0, 12.0)

GOLD_SPREADS = 4.0
GOLD_MARGIN_ANGSTROM = 70.0


@dataclass(frozen=True)
class TiltPicture:
    """One tilt, summed over its frames, binned, in the stack's orientation."""

    z_value: int
    angle_deg: float
    source: str
    image: NDArray[np.float32]
    pixel_angstrom: float
    binning: int

    @property
    def mean(self) -> float:
        return float(self.image.mean())


@dataclass(frozen=True)
class DarkSpot:
    """A dark round spot on the tilt nearest 0°, in that picture's pixels."""

    y: float
    x: float
    diameter_nm: float
    radius_px: float
    """Half the diameter, in the picture's pixels."""
    strength: float
    """How far the filter's answer stands above the picture's, in robust standard deviations."""
    darkness: float
    """Mean of the centre over the mean of a ring around it: 1 is no darker than around."""

    @property
    def counts(self) -> bool:
        return self.darkness <= SPOT_MAX_DARKNESS


@dataclass(frozen=True)
class PairScan:
    lower_deg: float
    higher_deg: float
    candidates_deg: NDArray[np.float64]
    scores: NDArray[np.float64]
    best_deg: float
    """In [0, 180)."""
    peak: float
    """The best normalised cross-correlation: how alike the pair is at all."""


@dataclass(frozen=True)
class AxisEstimate:
    angle_deg: float | None
    """In [0, 180), or None when no pair could be scanned."""
    pairs: tuple[PairScan, ...]
    reason: str | None = None
    """Why there is no estimate."""

    @property
    def agreeing(self) -> int:
        if self.angle_deg is None:
            return 0
        return sum(
            abs(axis_difference(pair.best_deg, self.angle_deg)) <= AXIS_AGREEMENT_DEG
            for pair in self.pairs
        )

    @property
    def reading(self) -> str:
        """horizontal, vertical, diagonal, or unmeasured."""

        if self.angle_deg is None:
            return "unmeasured"
        return orientation_word(self.angle_deg)


@dataclass(frozen=True)
class RecordedAxis:
    angle_deg: float | None
    source: str


@dataclass
class SeriesPreview:
    series: str
    entry: str
    angles_deg: list[float]
    raw_pixel_angstrom: float
    pictures: list[TiltPicture]
    zero: TiltPicture
    fine: NDArray[np.float32]
    fine_pixel_angstrom: float
    fine_binning: int
    spots: list[DarkSpot]
    axis: AxisEstimate
    recorded: RecordedAxis
    notes: list[str] = field(default_factory=list)

    @property
    def bead_like(self) -> list[DarkSpot]:
        return [spot for spot in self.spots if spot.counts]

    def picture(self, angle_deg: float) -> TiltPicture:
        return min(self.pictures, key=lambda picture: abs(picture.angle_deg - angle_deg))

    def exposure_fraction(self, picture: TiltPicture) -> float | None:
        return exposure_fraction(picture, self.zero)


def exposure_fraction(picture: TiltPicture, zero: TiltPicture) -> float | None:
    """A tilt's mean over the 0° tilt's; None for pictures that are not counts (mean <= 0)."""

    reference = zero.mean
    if reference <= 0 or float(picture.image.min()) < 0:
        return None
    return picture.mean / reference


# --- Reading -----------------------------------------------------------------------------


def sum_movie(path: Path) -> NDArray[np.float32]:
    """A movie's frames added up, in the orientation the motion-corrected sum will have."""

    fmt = movie_format(path)
    if fmt is MovieFormat.EER:
        raise ValueError(
            f"{path}: EER movies are not read by this preview; their frames are decoded by "
            "MotionCor3 alone (motion backend motioncor3)"
        )
    if fmt is MovieFormat.MRC:
        with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
            data = mrc.data
            if data is None or data.ndim != 3:
                raise ValueError(f"{path}: expected a movie of shape (frames, y, x)")
            total = np.zeros(data.shape[1:], dtype=np.float32)
            for frame in data:
                total += frame
        return total

    import tifffile

    with tifffile.TiffFile(path) as tiff:
        total = np.zeros(tiff.pages[0].shape, dtype=np.float32)
        for page in tiff.pages:
            try:
                total += page.asarray()
            except (KeyError, ValueError) as error:
                if "imagecodecs" in str(error):
                    raise ValueError(
                        f"{path}: its frames are compressed ({page.compression.name}), and "
                        "tifffile reads that only with the imagecodecs package, which the "
                        "pipeline's environment does not have: pip install imagecodecs"
                    ) from error
                raise
    # MotionCor3 reads a TIFF bottom row first: its sum is this one upside down.
    return np.ascontiguousarray(total[::-1])


def stack_section(path: Path, z_value: int) -> NDArray[np.float32]:
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        return np.asarray(mrc.data[z_value], dtype=np.float32)


def block_bin(image: NDArray[Any], factor: int) -> NDArray[np.float32]:
    """Mean of `factor` x `factor` blocks; a remainder at the far edges is dropped."""

    if factor <= 1:
        return np.asarray(image, dtype=np.float32)
    height = image.shape[0] // factor
    width = image.shape[1] // factor
    trimmed = np.asarray(image[: height * factor, : width * factor], dtype=np.float32)
    return np.asarray(trimmed.reshape(height, factor, width, factor).mean(axis=(1, 3)),
                      dtype=np.float32)


def binning_for(pixel_angstrom: float, target_angstrom: float) -> int:
    return max(1, math.ceil(target_angstrom / pixel_angstrom - 1e-6))


def views_to_read(angles: Sequence[float]) -> tuple[int, list[tuple[int, int]]]:
    """The index nearest 0°, and the neighbouring pairs of high tilts the axis is read from.

    Each pair is (lower |angle|, higher |angle|). A side of 0° gives up to
    `AXIS_VIEWS_PER_SIDE` - 1 pairs from its highest tilts, each pair's higher
    tilt at least `AXIS_MIN_TILT_DEG`.
    """

    zero = min(range(len(angles)), key=lambda index: abs(angles[index]))
    positive = sorted((index for index in range(len(angles))
                       if angles[index] > 0 and index != zero), key=lambda index: -angles[index])
    negative = sorted((index for index in range(len(angles))
                       if angles[index] < 0 and index != zero), key=lambda index: angles[index])
    pairs: list[tuple[int, int]] = []
    for side in (positive, negative):
        highest = side[:AXIS_VIEWS_PER_SIDE]
        for higher, lower in zip(highest, highest[1:], strict=False):
            if abs(angles[higher]) >= AXIS_MIN_TILT_DEG and (
                abs(angles[higher]) - abs(angles[lower]) > 0.5
            ):
                pairs.append((lower, higher))
    return zero, pairs


def recorded_axis(manifest: TiltSeriesManifest) -> RecordedAxis:
    """The axis alignment would take from the manifest, by the pipeline's own rule."""

    try:
        angle = manifest_tilt_axis_angle(manifest)
    except ValueError:
        if manifest.source_mdoc is None:
            return RecordedAxis(None, "none: the angles come from file names")
        return RecordedAxis(None, f"none: {manifest.source_mdoc.name} gives no SerialEM "
                                  "RotationAngle the pipeline takes")
    name = manifest.source_mdoc.name if manifest.source_mdoc is not None else "the manifest"
    return RecordedAxis(angle, f"{name}: RotationAngle - 90")


def preview_series(
    manifest: TiltSeriesManifest,
    *,
    stack: Path | None = None,
    progress: Any = None,
) -> SeriesPreview:
    """Read the tilts the two questions need, and measure both."""

    images = sorted(manifest.images, key=lambda image: image.tilt_angle_deg)
    angles = [image.tilt_angle_deg for image in images]
    pixel = manifest.raw_pixel_spacing_angstrom
    if pixel is None or pixel <= 0:
        raise ValueError(f"{manifest.tilt_series_id}: the manifest has no raw pixel size")
    coarse = binning_for(pixel, COARSE_ANGSTROM)
    fine_binning = max(1, round(FINE_ANGSTROM / pixel))
    zero_index, pairs = views_to_read(angles)
    wanted = sorted({zero_index, 0, len(images) - 1, *(i for pair in pairs for i in pair)})

    pictures: dict[int, TiltPicture] = {}
    fine: NDArray[np.float32] | None = None
    for index in wanted:
        image = images[index]
        if progress is not None:
            progress(f"{manifest.tilt_series_id}: reading {image.tilt_angle_deg:+.1f} deg "
                     f"({_source(image, stack)})")
        full = _read(image, stack)
        if index == zero_index:
            fine = block_bin(full, fine_binning)
        pictures[index] = TiltPicture(
            z_value=image.z_value,
            angle_deg=image.tilt_angle_deg,
            source=_source(image, stack),
            image=block_bin(full, coarse),
            pixel_angstrom=pixel * coarse,
            binning=coarse,
        )
        del full
    assert fine is not None
    zero = pictures[zero_index]
    if progress is not None:
        progress(f"{manifest.tilt_series_id}: looking for dark round spots and the axis")
    spots = find_dark_spots(zero.image, zero.pixel_angstrom)
    notes: list[str] = []
    dark = set()
    for index in sorted({i for pair in pairs for i in pair}):
        fraction = exposure_fraction(pictures[index], zero)
        if fraction is not None and fraction < DARK_VIEW_FRACTION:
            dark.add(index)
            notes.append(f"the {angles[index]:+.1f} deg tilt is left out of the axis: its mean "
                         f"is {100 * fraction:.1f}% of the {zero.angle_deg:+.1f} deg tilt's, "
                         "an empty or blocked view")
    kept = [(lower, higher) for lower, higher in pairs if lower not in dark and higher not in dark]
    axis = estimate_axis([(pictures[lower], pictures[higher]) for lower, higher in kept])
    if pairs and not kept:
        axis = AxisEstimate(None, (), reason="every pair of high tilts holds an empty view")
    return SeriesPreview(
        series=manifest.tilt_series_id,
        entry=SeriesEntry(manifest.entry).value,
        angles_deg=angles,
        raw_pixel_angstrom=pixel,
        pictures=[pictures[index] for index in sorted(pictures)],
        zero=zero,
        fine=fine,
        fine_pixel_angstrom=pixel * fine_binning,
        fine_binning=fine_binning,
        spots=spots,
        axis=axis,
        recorded=recorded_axis(manifest),
        notes=notes,
    )


def _source(image: TiltImage, stack: Path | None) -> str:
    if stack is not None:
        return f"{stack.name}, section {image.z_value}"
    if image.local_frame_file is None:
        raise ValueError(f"tilt {image.tilt_angle_deg:g} names no movie file")
    return image.local_frame_file.name


def _read(image: TiltImage, stack: Path | None) -> NDArray[np.float32]:
    if stack is not None:
        return stack_section(stack, image.z_value)
    assert image.local_frame_file is not None
    return sum_movie(Path(image.local_frame_file))


# --- The axis ----------------------------------------------------------------------------


def axis_difference(first_deg: float, second_deg: float) -> float:
    """first - second as lines without a sign: in [-90, 90)."""

    return (first_deg - second_deg + 90.0) % 180.0 - 90.0


def orientation_word(angle_deg: float) -> str:
    """How a line at IMOD's angle (from y, counterclockwise) runs on the page."""

    from_vertical = abs(axis_difference(angle_deg, 0.0))
    if from_vertical >= 60.0:
        return "horizontal"
    if from_vertical <= 30.0:
        return "vertical"
    return "diagonal"


def flatten(image: NDArray[Any]) -> NDArray[np.float64]:
    """Each row's and column's mean evened out: the stripes of an uncorrected gain go."""

    values = np.asarray(image, dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    return values - values.mean(axis=1, keepdims=True)


def prepare_for_scan(image: NDArray[Any], pixel_angstrom: float, *, without_gold: bool = True
                     ) -> NDArray[np.float64]:
    """Binned to the scan's size, band-passed, in units of its spread; gold set to 0.

    Gold sits on both faces of the specimen, and between two neighbouring tilts
    the two faces move against each other across the axis. A few strong beads
    then pull the stretch that matches best off the axis, the opposite way on
    each side of 0°. On synthetic slabs 56 px thick in a 192 px picture with 25
    beads (tests/test_show_tilts.py), beads in one plane left the pairs at 84-85°
    for an axis at 84°, and beads on both faces sent them to 62-105°. So
    everything darker than `GOLD_SPREADS` spreads, and `GOLD_MARGIN_ANGSTROM`
    around it, is set to 0 and the axis is read from the specimen: over 16 such
    slabs the estimate's rms error fell from 7.5° (at most 19.6°) to 3.6° (at most
    7.9°); without beads it is 1.3°. A real slab is thinner for its width.
    """

    factor = max(1, math.ceil(max(image.shape) / SCAN_MAX_PX))
    values = flatten(block_bin(image, factor))
    low, high = _BAND_SIGMAS_PX
    values = ndimage.gaussian_filter(values, low) - ndimage.gaussian_filter(values, high)
    spread = values.std()
    if spread <= 0:
        return values
    values = values / spread
    if without_gold:
        margin = max(1, round(GOLD_MARGIN_ANGSTROM / (pixel_angstrom * factor)))
        gold = ndimage.binary_dilation(values < -GOLD_SPREADS, iterations=margin)
        values[gold] = 0.0
    return values


def _taper(shape: tuple[int, ...], fraction: float = 0.15) -> NDArray[np.float64]:
    def ramp(size: int) -> NDArray[np.float64]:
        weights = np.ones(size)
        width = max(1, int(size * fraction))
        rise = 0.5 - 0.5 * np.cos(np.pi * np.arange(width) / width)
        weights[:width] = rise
        weights[size - width:] = rise[::-1]
        return weights

    return np.outer(ramp(shape[0]), ramp(shape[1]))


def stretch_across(image: NDArray[Any], axis_deg: float, factor: float) -> NDArray[np.float64]:
    """`image` with distances across the axis multiplied by 1 / `factor`, about its centre.

    Arrays are (y, x) with y up, so IMOD's axis at angle φ runs along
    (x, y) = (-sin φ, cos φ) and across it along (cos φ, sin φ).
    """

    phi = math.radians(axis_deg)
    across = np.array([math.sin(phi), math.cos(phi)])  # (y, x)
    matrix = np.eye(2) + (factor - 1.0) * np.outer(across, across)
    centre = (np.array(image.shape, dtype=np.float64) - 1.0) / 2.0
    # Cubic: with linear interpolation the pairs' bests kept about 5° off an axis
    # lying along the pixel rows or columns (synthetic series; estimates' rms error
    # over twelve axes 1.9° linear, 1.3° cubic).
    return np.asarray(ndimage.affine_transform(
        image, matrix, offset=centre - matrix @ centre, order=3, mode="constant", cval=0.0,
    ))


def _correlation(first: NDArray[np.float64], second: NDArray[np.float64]
                 ) -> tuple[float, tuple[int, int]]:
    """Peak of the normalised cross-correlation within a quarter of the picture, and its shift."""

    spectrum = np.fft.rfft2(first) * np.conj(np.fft.rfft2(second))
    surface = np.fft.fftshift(np.fft.irfft2(spectrum, s=first.shape))
    centre_y, centre_x = (np.array(surface.shape) // 2).tolist()
    reach_y, reach_x = max(1, surface.shape[0] // 4), max(1, surface.shape[1] // 4)
    window = surface[centre_y - reach_y:centre_y + reach_y + 1,
                     centre_x - reach_x:centre_x + reach_x + 1]
    flat = int(np.argmax(window))
    peak_y, peak_x = divmod(flat, window.shape[1])
    norm = math.sqrt(float((first * first).sum()) * float((second * second).sum()))
    return (float(window.flat[flat]) / norm if norm > 0 else 0.0,
            (peak_y - reach_y, peak_x - reach_x))


def pair_factor(lower_deg: float, higher_deg: float) -> float:
    return math.cos(math.radians(higher_deg)) / math.cos(math.radians(lower_deg))


def scan_pair(lower: TiltPicture, higher: TiltPicture) -> PairScan:
    first = prepare_for_scan(lower.image, lower.pixel_angstrom)
    second = prepare_for_scan(higher.image, higher.pixel_angstrom)
    if first.shape != second.shape:
        raise ValueError("the two tilts of a pair differ in size")
    taper = _taper(first.shape)
    reference = first * taper
    factor = pair_factor(lower.angle_deg, higher.angle_deg)

    def score(candidate: float) -> float:
        return _correlation(reference, stretch_across(second, candidate, factor) * taper)[0]

    coarse = np.arange(0.0, 180.0, AXIS_COARSE_STEP_DEG)
    coarse_scores = np.array([score(candidate) for candidate in coarse])
    centre = float(coarse[int(np.argmax(coarse_scores))])
    fine = np.arange(centre - 2 * AXIS_COARSE_STEP_DEG,
                     centre + 2 * AXIS_COARSE_STEP_DEG + 1e-9, AXIS_FINE_STEP_DEG)
    fine_scores = np.array([score(candidate % 180.0) for candidate in fine])
    candidates = np.concatenate([coarse, fine % 180.0])
    scores = np.concatenate([coarse_scores, fine_scores])
    keep = np.argsort(candidates, kind="stable")
    best = int(np.argmax(fine_scores))
    return PairScan(
        lower_deg=lower.angle_deg,
        higher_deg=higher.angle_deg,
        candidates_deg=candidates[keep],
        scores=scores[keep],
        best_deg=float(fine[best] % 180.0),
        peak=float(fine_scores[best]),
    )


def estimate_axis(pairs: Sequence[tuple[TiltPicture, TiltPicture]]) -> AxisEstimate:
    if not pairs:
        return AxisEstimate(None, (), reason=(
            f"no pair of neighbouring tilts reaches {AXIS_MIN_TILT_DEG:g} deg on either side"
        ))
    scans = tuple(scan_pair(lower, higher) for lower, higher in pairs)
    return AxisEstimate(circular_median([scan.best_deg for scan in scans]), scans)


def circular_median(lines_deg: Sequence[float]) -> float:
    """The median of lines without a sign, in [0, 180): taken about their mean direction."""

    doubled = np.radians(np.asarray(lines_deg, dtype=np.float64) * 2.0)
    mean = math.degrees(math.atan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2.0
    offsets = [axis_difference(line, mean) for line in lines_deg]
    return float((mean + float(np.median(offsets))) % 180.0)


def overlay_pair(lower: TiltPicture, higher: TiltPicture, axis_deg: float
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A pair's lower tilt, and its higher one stretched across `axis_deg` and moved onto it."""

    first = prepare_for_scan(lower.image, lower.pixel_angstrom, without_gold=False)
    second = stretch_across(prepare_for_scan(higher.image, higher.pixel_angstrom,
                                             without_gold=False), axis_deg,
                            pair_factor(lower.angle_deg, higher.angle_deg))
    taper = _taper(first.shape)
    _, shift = _correlation(first * taper, second * taper)
    return first, np.asarray(ndimage.shift(second, shift, order=1, mode="constant", cval=0.0))


# --- The beads ---------------------------------------------------------------------------


def find_dark_spots(image: NDArray[Any], pixel_angstrom: float) -> list[DarkSpot]:
    """Dark round spots, strongest first, each with the filter size that answered it best."""

    values = np.asarray(image, dtype=np.float64)
    answers = []
    for diameter in SPOT_DIAMETERS_NM:
        radius_px = diameter * 10.0 / pixel_angstrom / 2.0
        sigma = max(radius_px / math.sqrt(2.0), 0.7)
        answer = sigma**2 * ndimage.gaussian_laplace(values, sigma)
        middle = float(np.median(answer))
        spread = 1.4826 * float(np.median(np.abs(answer - middle)))
        answers.append((answer - middle) / spread if spread > 0 else answer * 0.0)
    stack = np.stack(answers)
    strength = stack.max(axis=0)
    scale = stack.argmax(axis=0)
    smallest_px = SPOT_DIAMETERS_NM[0] * 10.0 / pixel_angstrom
    neighbourhood = max(3, int(round(smallest_px)) | 1)
    peaks = (strength == ndimage.maximum_filter(strength, size=neighbourhood)) & (
        strength >= SPOT_MIN_STRENGTH
    )
    rows, columns = np.nonzero(peaks)
    order = np.argsort(-strength[rows, columns])
    spots: list[DarkSpot] = []
    for index in order:
        y, x = int(rows[index]), int(columns[index])
        diameter = SPOT_DIAMETERS_NM[int(scale[y, x])]
        radius = diameter * 10.0 / pixel_angstrom / 2.0
        if any(math.hypot(y - spot.y, x - spot.x) < 2.0 * max(radius, spot.radius_px)
               for spot in spots):
            continue
        darkness = _darkness(values, y, x, radius)
        if darkness is None:
            continue
        spots.append(DarkSpot(float(y), float(x), diameter, radius, float(strength[y, x]),
                              darkness))
    return spots


def _darkness(values: NDArray[np.float64], y: int, x: int, radius: float) -> float | None:
    outer = 2.5 * radius + 2.0
    if (y - outer < 0 or x - outer < 0 or y + outer > values.shape[0] - 1
            or x + outer > values.shape[1] - 1):
        return None
    top, left = int(y - outer), int(x - outer)
    patch = values[top:int(y + outer) + 1, left:int(x + outer) + 1]
    rows, columns = np.ogrid[:patch.shape[0], :patch.shape[1]]
    distance = np.hypot(rows + top - y, columns + left - x)
    core = patch[distance <= max(1.0, 0.6 * radius)]
    ring = patch[(distance >= 1.5 * radius + 1.0) & (distance <= outer)]
    if core.size == 0 or ring.size == 0:
        return None
    around = float(ring.mean())
    if around <= 0:
        return None
    return float(core.mean()) / around


def typical_diameter(spots: Sequence[DarkSpot]) -> float | None:
    """The diameter most bead-like spots answered to, None without any."""

    counted = [spot.diameter_nm for spot in spots if spot.counts]
    if not counted:
        return None
    values, counts = np.unique(counted, return_counts=True)
    return float(values[int(np.argmax(counts))])

