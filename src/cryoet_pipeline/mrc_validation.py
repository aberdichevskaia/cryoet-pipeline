from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from pathlib import Path

import mrcfile  # type: ignore[import-untyped]
from mrcfile import utils as mrc_utils


class IncompleteMrcError(ValueError):
    """Raised when an MRC file is shorter than its header declares."""


@dataclass(frozen=True)
class MrcFileInfo:
    """Header-derived MRC dimensions and file-size expectations."""

    shape: tuple[int, ...]
    dtype: str
    pixel_spacing_angstrom: float | None
    expected_size_bytes: int
    actual_size_bytes: int


def inspect_mrc_file(path: Path) -> MrcFileInfo:
    """Read an MRC header without mapping its potentially large data block."""

    actual_size_bytes = path.stat().st_size
    if actual_size_bytes < 1024:
        raise IncompleteMrcError(
            f"incomplete MRC file {path}: the 1024-byte header is truncated; "
            f"found {actual_size_bytes} bytes"
        )

    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        nx = int(mrc.header.nx)
        ny = int(mrc.header.ny)
        nz = int(mrc.header.nz)
        extended_header_size = int(mrc.header.nsymbt)
        dtype = mrc_utils.dtype_from_mode(mrc.header.mode)
        voxel_size_x = float(mrc.voxel_size.x)
        voxel_size_y = float(mrc.voxel_size.y)

    if voxel_size_x > 0.0 and voxel_size_y > 0.0:
        if not isclose(voxel_size_x, voxel_size_y, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(
                f"MRC file {path} has inconsistent X/Y pixel spacing: "
                f"{voxel_size_x} versus {voxel_size_y} angstrom"
            )
        pixel_spacing_angstrom = voxel_size_x
    elif voxel_size_x == 0.0 and voxel_size_y == 0.0:
        pixel_spacing_angstrom = None
    else:
        raise ValueError(
            f"MRC file {path} has incomplete X/Y pixel spacing: "
            f"{voxel_size_x} versus {voxel_size_y} angstrom"
        )

    shape = (nz, ny, nx) if nz > 1 else (ny, nx)
    expected_size_bytes = (
        1024 + extended_header_size + nx * ny * nz * dtype.itemsize
    )
    return MrcFileInfo(
        shape=shape,
        dtype=str(dtype),
        pixel_spacing_angstrom=pixel_spacing_angstrom,
        expected_size_bytes=expected_size_bytes,
        actual_size_bytes=actual_size_bytes,
    )


def validate_complete_mrc(path: Path) -> MrcFileInfo:
    """Return MRC information or raise when the data block is incomplete."""

    info = inspect_mrc_file(path)
    if info.actual_size_bytes < info.expected_size_bytes:
        raise IncompleteMrcError(
            f"incomplete MRC file {path}: header shape {info.shape} requires at "
            f"least {info.expected_size_bytes} bytes, found {info.actual_size_bytes}"
        )
    return info
