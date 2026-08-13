"""Read and write IMOD `.defocus` files, and parse CTFFIND5 summary tables.

The `.defocus` file is the interchange format between CTF estimation and CTF
correction in this pipeline: any estimation backend writes one, and
`ctfphaseflip` consumes it.

Format reference (IMOD `ctfphaseflip(1)`, "Defocus File Format"): every data
line holds at least five numbers — starting and ending view number (numbered
from 1), starting and ending tilt angle, and the defocus in nanometers with
underfocus positive. The first line may carry a version number at the end. For
versions 1 and 2 that first line also carries data; from version 3 on it is a
pure header, `flags 0 0. 0. 0 3`, where `flags` is the sum of:

    1   the file has astigmatism values
    2   the astigmatism axis angle is in radians, not degrees
    4   the file has phase shifts
    8   the phase shifts are in radians, not degrees
   16   tilt angles need to be inverted
   32   the file has cut-on frequencies

With astigmatism, the two defocus values are followed by the axis angle of the
defocus in the *first* column; IMOD accepts either defocus first as long as the
angle matches it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFOCUS_FLAG_ASTIGMATISM = 1
DEFOCUS_FLAG_ASTIGMATISM_RADIANS = 2
DEFOCUS_FLAG_PHASE_SHIFT = 4
DEFOCUS_FLAG_PHASE_SHIFT_RADIANS = 8
DEFOCUS_FLAG_INVERT_TILT_ANGLES = 16
DEFOCUS_FLAG_CUT_ON_FREQUENCY = 32


@dataclass(frozen=True)
class CtffindRow:
    """One fitted micrograph from a CTFFIND5 summary table.

    CTFFIND reports defocus in Angstrom and the additional phase shift in
    radians; `view` is the 1-based micrograph number, which for a stack input
    is the slice index.
    """

    view: int
    defocus1_angstrom: float
    defocus2_angstrom: float
    astigmatism_azimuth_deg: float
    phase_shift_rad: float
    cross_correlation: float | None
    fit_resolution_angstrom: float | None

    @property
    def mean_defocus_angstrom(self) -> float:
        return (self.defocus1_angstrom + self.defocus2_angstrom) / 2.0

    @property
    def astigmatism_angstrom(self) -> float:
        return abs(self.defocus1_angstrom - self.defocus2_angstrom) / 2.0


@dataclass(frozen=True)
class DefocusEntry:
    """One view of a parsed IMOD `.defocus` file."""

    view_start: int
    view_end: int
    angle_start_deg: float
    angle_end_deg: float
    defocus_nm: float
    second_defocus_nm: float | None = None
    astigmatism_axis_deg: float | None = None
    phase_shift: float | None = None

    @property
    def mean_defocus_nm(self) -> float:
        if self.second_defocus_nm is None:
            return self.defocus_nm
        return (self.defocus_nm + self.second_defocus_nm) / 2.0


def _iter_data_fields(path: Path) -> Iterable[tuple[int, list[str]]]:
    """Yield `(line_number, fields)` for non-empty, non-comment lines."""

    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split("#", 1)[0].strip()
        if line:
            yield line_number, line.split()


def parse_ctffind_summary(path: Path) -> list[CtffindRow]:
    """Parse a CTFFIND5 summary table, one row per fitted micrograph."""

    rows: list[CtffindRow] = []
    seen_views: set[int] = set()
    for line_number, fields in _iter_data_fields(path):
        if len(fields) < 5:
            raise ValueError(
                f"{path}:{line_number}: expected at least 5 numeric columns, got {len(fields)}"
            )
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(
                f"{path}:{line_number}: non-numeric value in CTFFIND data line"
            ) from exc

        view = int(round(values[0]))
        if not math.isclose(values[0], view, abs_tol=1e-6):
            raise ValueError(f"{path}:{line_number}: view number must be an integer")
        if view < 1:
            raise ValueError(f"{path}:{line_number}: view numbers are 1-based, got {view}")
        if view in seen_views:
            raise ValueError(f"{path}:{line_number}: duplicate view number {view}")
        seen_views.add(view)

        defocus1, defocus2, azimuth, phase_shift = values[1], values[2], values[3], values[4]
        if not all(math.isfinite(value) for value in (defocus1, defocus2, azimuth, phase_shift)):
            raise ValueError(f"{path}:{line_number}: non-finite CTF parameter")
        if defocus1 <= 0.0 or defocus2 <= 0.0:
            raise ValueError(
                f"{path}:{line_number}: expected positive underfocus values, "
                f"got {defocus1:g} and {defocus2:g} angstrom"
            )

        rows.append(
            CtffindRow(
                view=view,
                defocus1_angstrom=defocus1,
                defocus2_angstrom=defocus2,
                astigmatism_azimuth_deg=azimuth,
                phase_shift_rad=phase_shift,
                cross_correlation=values[5] if len(values) > 5 else None,
                fit_resolution_angstrom=values[6] if len(values) > 6 else None,
            )
        )

    if not rows:
        raise ValueError(f"{path}: no CTFFIND data rows found")
    rows.sort(key=lambda row: row.view)
    return rows


def write_imod_defocus_file(
    path: Path,
    rows: Sequence[CtffindRow],
    tilt_angles_deg: Sequence[float],
    *,
    include_astigmatism: bool = True,
    include_phase_shift: bool = False,
) -> None:
    """Write CTFFIND5 fits as a version-3 IMOD `.defocus` file.

    `tilt_angles_deg` must be the exact angles from the `.tlt` file handed to
    `ctfphaseflip`, indexed by view: the man page warns that rounding them
    breaks IMOD's view/angle matching. CTFFIND phase shifts are in radians, so
    the radians flag is set whenever phase shifts are written.
    """

    if not rows:
        raise ValueError("cannot write an empty defocus file")
    for row in rows:
        if row.view > len(tilt_angles_deg):
            raise ValueError(
                f"CTFFIND view {row.view} has no matching tilt angle; "
                f"{len(tilt_angles_deg)} angles were supplied"
            )

    flags = 0
    if include_astigmatism:
        flags |= DEFOCUS_FLAG_ASTIGMATISM
    if include_phase_shift:
        flags |= DEFOCUS_FLAG_PHASE_SHIFT | DEFOCUS_FLAG_PHASE_SHIFT_RADIANS

    lines = [f"{flags} 0 0.0 0.0 0.0 3"]
    for row in rows:
        tilt_angle = tilt_angles_deg[row.view - 1]
        fields = [
            str(row.view),
            str(row.view),
            f"{tilt_angle:.6f}",
            f"{tilt_angle:.6f}",
        ]
        if include_astigmatism:
            fields.extend(
                [
                    f"{row.defocus1_angstrom / 10.0:.3f}",
                    f"{row.defocus2_angstrom / 10.0:.3f}",
                    f"{row.astigmatism_azimuth_deg:.3f}",
                ]
            )
        else:
            fields.append(f"{row.mean_defocus_angstrom / 10.0:.3f}")
        if include_phase_shift:
            fields.append(f"{row.phase_shift_rad:.6f}")
        lines.append(" ".join(fields))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_imod_defocus_file(path: Path) -> list[DefocusEntry]:
    """Parse an IMOD `.defocus` file of any version into per-view entries."""

    raw_lines = [fields for _, fields in _iter_data_fields(path)]
    if not raw_lines:
        raise ValueError(f"{path}: defocus file is empty")

    flags = 0
    data_lines = raw_lines
    first = raw_lines[0]
    if len(first) == 6:
        try:
            version = int(float(first[5]))
        except ValueError as exc:
            raise ValueError(f"{path}: unreadable version number on the first line") from exc
        if version >= 3:
            flags = int(float(first[0]))
            data_lines = raw_lines[1:]
        else:
            # Versions 1 and 2 carry data on the first line, with the version
            # number appended; drop only that trailing token.
            data_lines = [first[:5], *raw_lines[1:]]

    if flags & DEFOCUS_FLAG_ASTIGMATISM_RADIANS:
        raise ValueError(f"{path}: astigmatism angles in radians are not supported")
    if flags & DEFOCUS_FLAG_CUT_ON_FREQUENCY:
        raise ValueError(f"{path}: cut-on frequency defocus files are not supported")

    expected = 5
    if flags & DEFOCUS_FLAG_ASTIGMATISM:
        expected += 2
    if flags & DEFOCUS_FLAG_PHASE_SHIFT:
        expected += 1

    entries: list[DefocusEntry] = []
    for index, fields in enumerate(data_lines):
        if len(fields) != expected:
            raise ValueError(
                f"{path}: data line {index + 1}: expected {expected} columns for "
                f"flags={flags}, got {len(fields)}"
            )
        values = [float(field) for field in fields]
        second_defocus_nm = None
        astigmatism_axis_deg = None
        cursor = 5
        if flags & DEFOCUS_FLAG_ASTIGMATISM:
            second_defocus_nm = values[cursor]
            astigmatism_axis_deg = values[cursor + 1]
            cursor += 2
        phase_shift = values[cursor] if flags & DEFOCUS_FLAG_PHASE_SHIFT else None
        entries.append(
            DefocusEntry(
                view_start=int(values[0]),
                view_end=int(values[1]),
                angle_start_deg=values[2],
                angle_end_deg=values[3],
                defocus_nm=values[4],
                second_defocus_nm=second_defocus_nm,
                astigmatism_axis_deg=astigmatism_axis_deg,
                phase_shift=phase_shift,
            )
        )

    if not entries:
        raise ValueError(f"{path}: defocus file has a header but no data lines")
    return entries
