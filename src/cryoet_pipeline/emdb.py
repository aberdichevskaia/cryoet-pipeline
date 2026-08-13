"""Fetch reference density maps from EMDB by accession code.

Maps are distributed as gzipped CCP4 files at a predictable path, so no
directory listing or per-entry layout description is needed (unlike EMPIAR,
see `empiar.py`). Downloads are cached: an entry already present locally is
reused rather than re-fetched.
"""

from __future__ import annotations

import gzip
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

EMDB_MAP_URL_TEMPLATE = (
    "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{number}/map/emd_{number}.map.gz"
)

_ACCESSION_PATTERN = re.compile(r"^(?:emd[-_]?)?(\d{4,5})$", re.IGNORECASE)


def normalize_emdb_id(emdb_id: str) -> str:
    """Return the bare numeric part of an EMDB accession.

    Accepts `2638`, `EMD-2638`, `emd_2638`, and similar spellings.
    """

    match = _ACCESSION_PATTERN.match(emdb_id.strip())
    if match is None:
        raise ValueError(
            f"unrecognized EMDB accession {emdb_id!r}; expected forms like '2638' or 'EMD-2638'"
        )
    return match.group(1)


def emdb_map_url(emdb_id: str) -> str:
    """Return the direct download URL for one EMDB entry's primary map."""

    return EMDB_MAP_URL_TEMPLATE.format(number=normalize_emdb_id(emdb_id))


def fetch_emdb_map(
    emdb_id: str,
    *,
    cache_dir: Path,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Download and decompress one EMDB map, reusing a cached copy when present.

    Returns the path of the decompressed `.map` file.
    """

    number = normalize_emdb_id(emdb_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"emd_{number}.map"
    if destination.is_file() and not overwrite:
        _emit(progress, f"using cached EMDB map {destination}")
        return destination

    url = emdb_map_url(number)
    archive = cache_dir / f"emd_{number}.map.gz"
    partial = archive.with_suffix(archive.suffix + ".part")
    _emit(progress, f"downloading {url}")
    with urlopen(url) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)
    partial.replace(archive)

    _emit(progress, f"decompressing {archive.name}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    with gzip.open(archive, "rb") as source, temporary.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    temporary.replace(destination)
    archive.unlink()
    return destination


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
