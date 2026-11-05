"""Per-particle tables as Parquet files (docs/run_records.md).

One particle set, one file: named, typed columns, and what the numbers mean (angle
and shift conventions, pixel size, box) in the file's own metadata, so the file says
how to read it without the code that wrote it. Values are written as given, float64
when they are float64: nothing is rounded on the way to disk.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cryoet_pipeline.runs import canonical_json

METADATA_KEY = b"cryoet_pipeline"
DYNAMO_POSES = {
    "angles": "tdrot, tilt, narot in degrees, as columns 7-9 of a Dynamo table",
    "shifts": "pixels of the particle's box, as columns 4-6 of a Dynamo table",
    "rotation_centre": "voxel N/2, zero-based, as Dynamo turns a particle",
}
"""The pose convention of the port, and so of every table the MRA writes."""


def write_particle_table(path: Path, columns: Mapping[str, np.ndarray | Sequence[Any]],
                         metadata: Mapping[str, Any]) -> None:
    """Columns of one length, in the order given, and `metadata` as JSON beside them."""

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    lengths = {name: len(values) for name, values in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"columns of different lengths: {lengths}")
    table = pa.table({name: pa.array(np.asarray(values) if not isinstance(values, list)
                                     else values) for name, values in columns.items()})
    table = table.replace_schema_metadata({METADATA_KEY: canonical_json(dict(metadata)).encode()})
    pq.write_table(table, path)


def read_particle_table(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Every column as a numpy array (text as an object array), and the metadata."""

    import pyarrow.parquet as pq

    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(METADATA_KEY)
    columns = {name: table.column(name).to_numpy(zero_copy_only=False)
               for name in table.column_names}
    return columns, json.loads(raw) if raw else {}


__all__ = ["DYNAMO_POSES", "read_particle_table", "write_particle_table"]
