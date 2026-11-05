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


def select_rows(columns: Mapping[str, np.ndarray], definition: Mapping[str, Any]) -> np.ndarray:
    """The rows a selection's definition names (docs/run_records.md), as a boolean mask:
    `{"column": c, "equals": v}` the rows whose `c` is `v`, and with `"where": w` only
    those of them whose `w` is true. Any other definition is refused, not guessed at."""

    keys = set(definition)
    if not {"column", "equals"} <= keys or keys - {"column", "equals", "where"}:
        raise ValueError(f"a selection defined as {dict(definition)} is not one this reader "
                         "knows")
    missing = [definition[key] for key in ("column", "where")
               if key in definition and definition[key] not in columns]
    if missing:
        raise ValueError(f"the table has no column {', '.join(map(str, missing))}")
    rows = np.asarray(columns[definition["column"]]) == definition["equals"]
    if "where" in definition:
        rows &= np.asarray(columns[definition["where"]], dtype=bool)
    return np.asarray(rows, dtype=bool)


__all__ = ["DYNAMO_POSES", "read_particle_table", "select_rows", "write_particle_table"]
