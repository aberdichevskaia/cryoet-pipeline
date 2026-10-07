"""Per-particle metadata and the STAR files that carry it.

Lifted out of `backends/subtomogram.py` because reading and writing a STAR file
is not a backend's job, and because keeping it there forced a dependency the
wrong way round: `fsc.py` needed `read_particle_star`, so averaging could not
reach the FSC — which it now must, since a gold-standard refinement sets each
iteration's resolution limit from the agreement between its two half-sets.

So the layering runs downhill from here: particles -> fsc -> backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CROP_POSE_CONVENTION = "rotation-center-v2"
"""Raw crops are unchanged; STAR shifts put each pick's rotation centre at (size - 1) / 2.

A pytom-match-pick coordinate is not that centre. pytom pastes its template so
that voxel `T // 2` lands on the reported peak (`matching.py`: a centred paste,
then `roll(flip(scores), V // 2 + 1)`), but rotates the template with voltools
about `(T - 1) / 2`, the one point every orientation leaves in place. For an
even template box the pick is therefore half a voxel above the particle's
centre along every lab axis, whatever the pose: `pick_center_offset_px`.

Measured before this marker existed, on 223 CZII ribosomes (template 48, crop
48). `geometric-center-v1` took the pick as the centre and left every particle
half a voxel off. A Dynamo search, which shares none of these conventions,
moved them by (-0.47, -0.45, -0.48) voxels, and zero shifts alone reproduced
that search's map: 26.70 A against 30.04 A.

Reading an old STAR never silently upgrades its poses. `geometric-center-v1`
shifts are half a voxel off for even templates. Unmarked cubic-crop artifacts
wrote zero shifts, which is right only for an integral pick whose template and
crop boxes have the same parity.
"""


def pytom_template_shape(job_json: Path) -> tuple[int, ...] | None:
    """The template box pytom actually matched with, from the job JSON it wrote.

    Taken from there rather than measured off the template file because that is
    the number the match was run with, and because a box read back from disk
    could have been rebuilt since.
    """

    try:
        payload = json.loads(job_json.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    shape = payload.get("template_shape")
    if not isinstance(shape, list) or not shape:
        return None
    if not all(type(item) is int and item > 0 for item in shape):
        return None
    return tuple(shape)


def pick_center_offset_px(template_box_px: int) -> float:
    """How far a pytom pick lies above its particle's rotation centre, per axis.

    Half a voxel for an even template box, where `T // 2` and `(T - 1) / 2`
    differ; nothing for an odd one, where they coincide.
    """

    if type(template_box_px) is not int or template_box_px < 1:
        raise ValueError(f"template box must be a positive int, got {template_box_px!r}")
    return 0.5 if template_box_px % 2 == 0 else 0.0


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
    half_set: int
    """Which half (1 or 2) the particle belongs to for independent refinement.

    Not random, despite the STAR column it is stored in, `rlnRandomSubset`,
    whose name is RELION's and is kept so that other programs read it.
    Extraction writes `1 + index % 2` in each tomogram's pick order, which is
    pytom's score order, so the halves alternate by score rank and split every
    tomogram evenly. Nothing recomputes it afterwards: a particle keeps its half
    through every step that carries the STAR rows.
    """


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
                    str(record.half_set),
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
                half_set=int(fields[positions["rlnRandomSubset"]]),
            )
        )
    records.sort(key=lambda record: record.index)
    return records
