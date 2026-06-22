from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cryoet_pipeline.models import TiltImage, TiltSeriesManifest

_SECTION_RE = re.compile(r"^\[(?P<key>[^=\]]+)=\s*(?P<value>[^\]]+)\]\s*$")


def parse_mdoc_text(
    text: str,
    *,
    source_mdoc: Path,
    tilt_series_id: str,
    frames_dir: Path | None = None,
    raw_pixel_spacing_angstrom: float | None = None,
) -> TiltSeriesManifest:
    """Parse SerialEM `.mdoc` text into a normalized manifest.

    The parser intentionally preserves unrecognized per-tilt keys in `metadata` so
    later processing steps can use them without requiring parser changes.
    """

    global_fields: dict[str, str] = {}
    sections: list[tuple[str, str, dict[str, str]]] = []
    current: tuple[str, str, dict[str, str]] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_match = _SECTION_RE.match(line)
        if section_match:
            key = section_match.group("key").strip()
            value = section_match.group("value").strip()
            current = (key, value, {})
            sections.append(current)
            continue

        if "=" not in line:
            continue

        key, value = [part.strip() for part in line.split("=", 1)]
        if current is None:
            global_fields[key] = value
        else:
            current[2][key] = value

    images: list[TiltImage] = []
    for section_key, section_value, fields in sections:
        if section_key != "ZValue":
            continue

        local_frame_file = _resolve_local_frame(fields.get("SubFramePath"), frames_dir)
        metadata: dict[str, Any] = {
            key: value
            for key, value in fields.items()
            if key
            not in {
                "TiltAngle",
                "SubFramePath",
                "NumSubFrames",
                "PixelSpacing",
                "Binning",
                "ExposureTime",
                "ExposureDose",
                "Defocus",
                "RotationAngle",
            }
        }

        images.append(
            TiltImage(
                z_value=int(section_value),
                tilt_angle_deg=float(fields["TiltAngle"]),
                subframe_path=fields["SubFramePath"],
                local_frame_file=local_frame_file,
                num_subframes=int(fields["NumSubFrames"]),
                pixel_spacing_angstrom=float(fields["PixelSpacing"]),
                binning=int(fields["Binning"]),
                exposure_time_s=_optional_float(fields.get("ExposureTime")),
                exposure_dose=_optional_float(fields.get("ExposureDose")),
                defocus_um=_optional_float(fields.get("Defocus")),
                rotation_angle_deg=_optional_float(fields.get("RotationAngle")),
                metadata=metadata,
            )
        )

    mdoc_pixel_spacing = _optional_float(global_fields.get("PixelSpacing"))
    notes: list[str] = []
    if raw_pixel_spacing_angstrom is not None and mdoc_pixel_spacing is not None:
        if abs(raw_pixel_spacing_angstrom - mdoc_pixel_spacing) > 1e-6:
            notes.append(
                "raw pixel spacing differs from mdoc pixel spacing; "
                "this usually indicates binned mdoc metadata"
            )

    return TiltSeriesManifest(
        tilt_series_id=tilt_series_id,
        source_mdoc=source_mdoc,
        images=images,
        mdoc_pixel_spacing_angstrom=mdoc_pixel_spacing,
        raw_pixel_spacing_angstrom=raw_pixel_spacing_angstrom,
        notes=notes,
    )


def _optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _resolve_local_frame(subframe_path: str | None, frames_dir: Path | None) -> Path | None:
    if subframe_path is None or frames_dir is None:
        return None
    normalized = subframe_path.replace("\\", "/")
    return frames_dir / Path(normalized).name
