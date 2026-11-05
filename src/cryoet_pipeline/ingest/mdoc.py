from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from cryoet_pipeline.models import TiltImage, TiltSeriesManifest

_SECTION_RE = re.compile(r"^\[(?P<key>[^=\]]+)=\s*(?P<value>[^\]]+)\]\s*$")
_TILT_AXIS_IN_TITLE_RE = re.compile(r"TiltAxisAngle\s*=\s*(?P<angle>[-+]?\d+(?:\.\d+)?)")


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

    Thermo Fisher's Tomography writes mdocs too ("[T = Tomography: ...]"), with
    the same keys and other meanings. Its RotationAngle is kept in `metadata`
    only, not as `rotation_angle_deg`: the tilt axis that alignment derives from
    it by SerialEM's rule (RotationAngle - 90) is 90 degrees off for them. On
    EMPIAR-11830 the axis lay along RotationAngle itself in four sessions of
    four, pointing either way, so neither field gives it; it has to be given.
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

    titles = [value for key, value, _ in sections if key.strip() == "T"]
    tomography = next((title for title in titles if title.startswith("Tomography")), None)

    images: list[TiltImage] = []
    for section_key, section_value, fields in sections:
        if section_key != "ZValue":
            continue

        local_frame_file = _resolve_local_frame(fields.get("SubFramePath"), frames_dir)
        parsed = {
            "TiltAngle",
            "SubFramePath",
            "NumSubFrames",
            "PixelSpacing",
            "Binning",
            "ExposureTime",
            "ExposureDose",
            "Defocus",
        }
        if tomography is None:
            parsed.add("RotationAngle")
        metadata: dict[str, Any] = {
            key: value for key, value in fields.items() if key not in parsed
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
                rotation_angle_deg=(
                    None if tomography is not None
                    else _optional_float(fields.get("RotationAngle"))
                ),
                metadata=metadata,
            )
        )

    mdoc_pixel_spacing = _optional_float(global_fields.get("PixelSpacing"))
    notes: list[str] = []
    if tomography is not None:
        notes.append(_tomography_note(tomography, titles, images))
    if raw_pixel_spacing_angstrom is not None and mdoc_pixel_spacing is not None:
        if abs(raw_pixel_spacing_angstrom - mdoc_pixel_spacing) > 1e-6:
            notes.append(
                "raw pixel spacing differs from mdoc pixel spacing; "
                "this usually indicates binned mdoc metadata"
            )

    manifest = TiltSeriesManifest(
        tilt_series_id=tilt_series_id,
        source_mdoc=source_mdoc,
        images=images,
        mdoc_pixel_spacing_angstrom=mdoc_pixel_spacing,
        notes=notes,
    )
    if raw_pixel_spacing_angstrom is None:
        return manifest
    return with_raw_pixel_spacing(manifest, raw_pixel_spacing_angstrom)


def with_raw_pixel_spacing(
    manifest: TiltSeriesManifest,
    raw_pixel_spacing_angstrom: float,
) -> TiltSeriesManifest:
    """Attach authoritative raw-frame spacing and record mdoc differences."""

    notes = [
        note
        for note in manifest.notes
        if not note.startswith("raw pixel spacing differs from mdoc pixel spacing")
    ]
    mdoc_pixel_spacing = manifest.mdoc_pixel_spacing_angstrom
    if (
        mdoc_pixel_spacing is not None
        and abs(raw_pixel_spacing_angstrom - mdoc_pixel_spacing) > 1e-6
    ):
        notes.append(
            "raw pixel spacing differs from mdoc pixel spacing; "
            "the raw MRC header is used as the default and may require "
            "a calibration override"
        )
    return manifest.model_copy(
        update={
            "raw_pixel_spacing_angstrom": raw_pixel_spacing_angstrom,
            "notes": notes,
        }
    )


def _optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _tomography_note(tomography: str, titles: list[str], images: list[TiltImage]) -> str:
    rotations = sorted({str(image.metadata.get("RotationAngle")) for image in images})
    axes = [
        found["angle"]
        for title in titles
        if (found := _TILT_AXIS_IN_TITLE_RE.search(title)) is not None
    ]
    said = f"RotationAngle {', '.join(rotations)}"
    if axes:
        said += f", TiltAxisAngle {axes[0]}"
    return (
        f"written by Thermo Fisher Tomography ({' '.join(tomography.split())}), "
        "not SerialEM: no tilt axis "
        f"is taken from it ({said}). SerialEM's rule, axis = RotationAngle - 90, is 90 "
        "degrees off for these mdocs; on EMPIAR-11830 the axis lay along RotationAngle, "
        "pointing either way. Give the axis explicitly ([align] tilt_axis_angle)."
    )


def _resolve_local_frame(subframe_path: str | None, frames_dir: Path | None) -> Path | None:
    if subframe_path is None or frames_dir is None:
        return None
    normalized = subframe_path.replace("\\", "/")
    return frames_dir / Path(normalized).name
