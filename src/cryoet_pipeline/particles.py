"""Per-particle metadata and the STAR files that carry it.

Lifted out of `backends/subtomogram.py` because reading and writing a STAR file
is not a backend's job, and because keeping it there forced a dependency the
wrong way round: `fsc.py` needed `read_particle_star`, so averaging could not
reach the FSC — which it now must, since a gold-standard refinement sets each
iteration's resolution limit from the agreement between its two half-sets.

So the layering runs downhill from here: particles -> fsc -> backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParticleRecord:
    """One row of per-particle metadata carried alongside a particle stack."""

    index: int
    """Position in the array or list at hand — never a particle's name.

    Written as the slice number in the stack by extraction, rewritten with an
    offset by `_combine_particle_stacks` when several stacks are averaged
    together, and used as a subscript in roughly sixteen places. So a STAR and a
    stack are only interchangeable if they were written for one another, and a
    record cannot say which pick or which tilt-series it came from. Both facts
    are relied on; neither was stated anywhere until this line.
    """
    x_px: float
    y_px: float
    z_px: float
    rot_deg: float
    tilt_deg: float
    psi_deg: float
    shift_x_px: float
    shift_y_px: float
    shift_z_px: float
    score: float | None
    class_number: int
    random_subset: int


def read_star_loop(path: Path) -> tuple[list[str], list[list[str]]]:
    """Column names and raw string rows from the first `loop_` block."""

    lines = path.read_text().splitlines()
    columns: list[str] = []
    in_loop = False
    data_start: int | None = None
    for line_index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if line == "loop_":
            in_loop = True
            continue
        if in_loop and line.startswith("_"):
            columns.append(line.split()[0].lstrip("_"))
            continue
        if in_loop and columns and line:
            data_start = line_index
            break
    if data_start is None:
        raise ValueError(f"no STAR data rows found in {path}")
    rows = [line.split() for line in lines[data_start:] if line.strip()]
    return columns, rows


# Every name here is REQUIRED on read: `read_particle_star` collects the missing
# ones and raises. So appending a column to this tuple makes every particle STAR
# already on disk unreadable, and a new column has to be read as optional instead
# — the same shape the optional `filled.fid` output took in fine alignment.
_PARTICLE_STAR_COLUMNS = (
    "particleIndex",
    "rlnCoordinateX",
    "rlnCoordinateY",
    "rlnCoordinateZ",
    "rlnAngleRot",
    "rlnAngleTilt",
    "rlnAnglePsi",
    "shiftXPx",
    "shiftYPx",
    "shiftZPx",
    "rlnLCCmax",
    "rlnClassNumber",
    "rlnRandomSubset",
)


def write_particle_star(path: Path, records: list[ParticleRecord]) -> None:
    lines = ["data_particles", "", "loop_"]
    lines.extend(f"_{name} #{position}" for position, name in enumerate(_PARTICLE_STAR_COLUMNS, 1))
    for record in records:
        lines.append(
            " ".join(
                [
                    str(record.index),
                    f"{record.x_px:.3f}",
                    f"{record.y_px:.3f}",
                    f"{record.z_px:.3f}",
                    f"{record.rot_deg:.6f}",
                    f"{record.tilt_deg:.6f}",
                    f"{record.psi_deg:.6f}",
                    f"{record.shift_x_px:.6f}",
                    f"{record.shift_y_px:.6f}",
                    f"{record.shift_z_px:.6f}",
                    f"{record.score:.6f}" if record.score is not None else "nan",
                    str(record.class_number),
                    str(record.random_subset),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_particle_star(path: Path) -> list[ParticleRecord]:
    columns, rows = read_star_loop(path)
    missing = [name for name in _PARTICLE_STAR_COLUMNS if name not in columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    positions = {name: columns.index(name) for name in _PARTICLE_STAR_COLUMNS}
    records = []
    for fields in rows:
        score_text = fields[positions["rlnLCCmax"]]
        records.append(
            ParticleRecord(
                index=int(fields[positions["particleIndex"]]),
                x_px=float(fields[positions["rlnCoordinateX"]]),
                y_px=float(fields[positions["rlnCoordinateY"]]),
                z_px=float(fields[positions["rlnCoordinateZ"]]),
                rot_deg=float(fields[positions["rlnAngleRot"]]),
                tilt_deg=float(fields[positions["rlnAngleTilt"]]),
                psi_deg=float(fields[positions["rlnAnglePsi"]]),
                shift_x_px=float(fields[positions["shiftXPx"]]),
                shift_y_px=float(fields[positions["shiftYPx"]]),
                shift_z_px=float(fields[positions["shiftZPx"]]),
                score=None if score_text == "nan" else float(score_text),
                class_number=int(fields[positions["rlnClassNumber"]]),
                random_subset=int(fields[positions["rlnRandomSubset"]]),
            )
        )
    records.sort(key=lambda record: record.index)
    return records
