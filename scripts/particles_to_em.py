#!/usr/bin/env python3
"""Copy a Dynamo particle folder from MRC to EM, for Dynamo's GPU motor.

    python scripts/particles_to_em.py --src <export>/particles --dst <export>/particles_em

The CUDA motor in Dynamo 1.1.591 (dynamo_compute_iteration_spp.cu,
volume_float::read) reads every volume as EM whatever its extension: it skips
one int, takes nx, ny, nz from the next three and the voxels from byte 512 on.
Fed one of our MRC particles (header nx, ny, nz, mode), it gets 64 x 64 x 2 voxels
of header and data, finds no correlation above its -1 starting value, and writes
its debug sentinels as the pose: gag_gpu_eo, 2026-10-12, where z came out -4004,
i.e. -4003 - nz/2 with nz = 2 = the float32 mode word. The CPU path reads MRC.

The voxels are copied unchanged, float32 to float32, so a GPU project on the EM
folder sees the same particles as the CPU project on the MRC one. Each file is
read back and compared before the next is written. Disposable, like the rest of
scripts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mrcfile
import numpy as np

# Byte 0: 6 = PC, little endian. Byte 3: 5 = float32. The motor's writer uses the
# same four bytes (pad_first = {6, 0, 0, 5}).
_EM_HEADER_START = bytes([6, 0, 0, 5])
_EM_HEADER_BYTES = 512


def write_em(path: Path, volume: np.ndarray) -> None:
    nz, ny, nx = volume.shape
    header = bytearray(_EM_HEADER_BYTES)
    header[0:4] = _EM_HEADER_START
    header[4:16] = np.array([nx, ny, nz], dtype="<i4").tobytes()
    path.write_bytes(bytes(header) + np.ascontiguousarray(volume, dtype="<f4").tobytes())


def read_em(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    nx, ny, nz = np.frombuffer(raw, dtype="<i4", count=3, offset=4)
    return np.frombuffer(raw, dtype="<f4", offset=_EM_HEADER_BYTES).reshape(nz, ny, nx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="folder of particle_NNNNN.mrc")
    parser.add_argument("--dst", type=Path, required=True, help="new folder for particle_NNNNN.em")
    args = parser.parse_args()

    sources = sorted(args.src.glob("particle_*.mrc"))
    if not sources:
        raise SystemExit(f"no particle_*.mrc in {args.src}")
    args.dst.mkdir(parents=True, exist_ok=False)

    shapes = set()
    for source in sources:
        with mrcfile.open(source, permissive=False) as handle:
            volume = np.asarray(handle.data, dtype=np.float32)
        target = args.dst / (source.stem + ".em")
        write_em(target, volume)
        if not np.array_equal(read_em(target), volume):
            raise SystemExit(f"{target}: read-back differs from {source}")
        shapes.add(volume.shape)

    print(f"{len(sources)} particles, shapes {sorted(shapes)}, written to {args.dst}")


if __name__ == "__main__":
    main()
