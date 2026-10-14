"""A tilt-series manifest from movie file names, for data that has no mdocs.

The names are read with one regular expression, matched against the whole file
name, whose named groups say where each piece of metadata sits:

* `series` — the tilt-series id, taken verbatim (so `039` and `39` differ);
* `tilt`   — the tilt number, an integer. The images are ordered by it, and
  its rank becomes the z value, as an mdoc's ZValue would be. What the number
  means is up to whoever named the files: in EMPIAR-10164
  (`TS_01_002_-3.0.mrc`) it is the acquisition order, in the first local TIFF
  dataset (`tomogram_039_001_-42.0.tif` to `tomogram_039_029_+42.0.tif`) the
  angle order. The steps here need only an order, so either works, but in the
  second case the dose order is not in the names at all;
* `angle`  — the tilt angle in degrees. Write the group so it takes a leading
  `+`: that dataset names its positive tilts `+03.0`.

For that dataset:
`tomogram_(?P<series>\\d+)_(?P<tilt>\\d+)_(?P<angle>[-+]?\\d+(?:\\.\\d+)?)\\.tif`.

The names carry nothing else, so what an mdoc would add is absent here: dose,
nominal defocus and the tilt-axis rotation. The pixel size comes from the movie
headers (MRC, or the resolution tags SerialEM writes into a TIFF) unless it is
given; the tilt axis has to be given to alignment.

Telling the series and tilts apart from an arbitrary naming convention, without
being handed the expression, is not done here; see docs/roadmap.md.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from math import isclose
from pathlib import Path

from cryoet_pipeline.models import TiltImage, TiltSeriesManifest
from cryoet_pipeline.movies import agreed_pixel_spacing, inspect_movie

FRAME_PATTERN_GROUPS = ("series", "tilt", "angle")


@dataclass(frozen=True)
class NamedFrame:
    path: Path
    series: str
    tilt_number: int
    angle_deg: float


def compile_frame_pattern(pattern: str) -> re.Pattern[str]:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"frames pattern is not a regular expression: {exc}") from exc
    missing = [name for name in FRAME_PATTERN_GROUPS if name not in compiled.groupindex]
    if missing:
        raise ValueError(
            f"frames pattern needs the named groups {', '.join(FRAME_PATTERN_GROUPS)}; "
            f"missing {', '.join(missing)}, e.g. (?P<{missing[0]}>...)"
        )
    return compiled


def match_frames(frames_dir: Path, pattern: str) -> dict[str, list[NamedFrame]]:
    """Every file in the folder whose whole name matches, by series, in tilt order.

    A file the pattern misses is skipped when it has another suffix than the
    frames, and refused when it has theirs: a pattern that is slightly off
    matches only part of every series and says nothing. The first one written
    for `tomogram_039_016_+03.0.tif` had no `+` and matched only the negative
    tilts. Hidden files (`.DS_Store`, the `._` copies macOS leaves on shared
    drives) are never frames and are not looked at.
    """

    compiled = compile_frame_pattern(pattern)
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory not found: {frames_dir}")
    names = sorted(
        path.name
        for path in frames_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )

    by_series: dict[str, list[NamedFrame]] = defaultdict(list)
    unmatched: list[str] = []
    for name in names:
        found = compiled.fullmatch(name)
        if found is None:
            unmatched.append(name)
            continue
        try:
            tilt_number = int(found["tilt"])
            angle = float(found["angle"])
        except ValueError as exc:
            raise ValueError(
                f"{name}: the tilt group must be an integer and the angle group a "
                f"number, got tilt={found['tilt']!r}, angle={found['angle']!r}"
            ) from exc
        by_series[found["series"]].append(
            NamedFrame(
                path=frames_dir / name,
                series=found["series"],
                tilt_number=tilt_number,
                angle_deg=angle,
            )
        )
    if not by_series:
        preview = ", ".join(names[:5]) or "the folder is empty"
        raise ValueError(
            f"no file in {frames_dir} matches the frames pattern {pattern!r}; "
            f"first names: {preview}"
        )
    suffixes = {frame.path.suffix for frames in by_series.values() for frame in frames}
    missed = [name for name in unmatched if Path(name).suffix in suffixes]
    if missed:
        raise ValueError(
            f"{len(missed)} {'/'.join(sorted(suffixes))} files in {frames_dir} do not match "
            f"the frames pattern {pattern!r}, e.g. {', '.join(missed[:3])}. Fix the pattern, "
            "or move what is not a frame out of the folder; series are chosen by their "
            "ids, not by the pattern"
        )
    return {
        series: sorted(frames, key=lambda frame: frame.tilt_number)
        for series, frames in sorted(by_series.items())
    }


def manifest_from_frame_names(
    *,
    frames_dir: Path,
    pattern: str,
    tilt_series_id: str,
    raw_pixel_spacing_angstrom: float | None = None,
) -> TiltSeriesManifest:
    """The manifest of one series, with its frame counts read from the movies.

    The pixel size is read from the movie headers, which must all carry the
    same one. A given pixel size overrides them, for headers known to be wrong,
    and the manifest notes it when they say otherwise.
    """

    by_series = match_frames(frames_dir, pattern)
    frames = by_series.get(tilt_series_id)
    if frames is None:
        raise ValueError(
            f"no movies of series {tilt_series_id!r} in {frames_dir}; the frames "
            f"pattern finds the series {', '.join(sorted(by_series))}"
        )
    _refuse_duplicates(tilt_series_id, frames)

    infos = [inspect_movie(frame.path) for frame in frames]
    header_spacings = [
        (frame.path, info.pixel_spacing_angstrom) for frame, info in zip(frames, infos, strict=True)
    ]
    notes: list[str] = []
    pixel_spacing = raw_pixel_spacing_angstrom
    if pixel_spacing is None:
        pixel_spacing = agreed_pixel_spacing(tilt_series_id, header_spacings)
    else:
        in_headers = {spacing for _, spacing in header_spacings}
        (header,) = in_headers if len(in_headers) == 1 else (None,)
        if header is not None and not isclose(header, pixel_spacing, rel_tol=1e-5):
            notes.append(
                f"raw pixel spacing given as {pixel_spacing:g} angstrom, overriding "
                f"the {header:g} angstrom in the movie headers"
            )

    images = [
        TiltImage(
            z_value=z_value,
            tilt_angle_deg=frame.angle_deg,
            subframe_path=frame.path.name,
            local_frame_file=frame.path,
            num_subframes=info.shape[0],
            pixel_spacing_angstrom=pixel_spacing,
            binning=1,
            metadata={"tilt_number": frame.tilt_number},
        )
        for z_value, (frame, info) in enumerate(zip(frames, infos, strict=True))
    ]
    return TiltSeriesManifest(
        tilt_series_id=tilt_series_id,
        source_frames_pattern=pattern,
        images=images,
        raw_pixel_spacing_angstrom=pixel_spacing,
        notes=notes,
    )


def _refuse_duplicates(tilt_series_id: str, frames: list[NamedFrame]) -> None:
    by_label: dict[str, list[tuple[object, Path]]] = {
        "tilt number": [(frame.tilt_number, frame.path) for frame in frames],
        "tilt angle": [(frame.angle_deg, frame.path) for frame in frames],
    }
    for label, values in by_label.items():
        seen: dict[object, Path] = {}
        for value, path in values:
            if value in seen:
                raise ValueError(
                    f"{tilt_series_id}: two movies share {label} {value}: "
                    f"{seen[value].name} and {path.name}"
                )
            seen[value] = path
