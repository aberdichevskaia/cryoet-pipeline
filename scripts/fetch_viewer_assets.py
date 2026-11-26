#!/usr/bin/env python3
"""Fetch the libraries the result pages load, each checked against its sha256.

    python3 scripts/fetch_viewer_assets.py
    python3 scripts/fetch_viewer_assets.py --from /path/to/vendor     # no internet here

Reads configs/viewer_assets.toml and writes each file under vendor/ at the clone's
root, which git ignores; scripts/serve_results.py serves them from there. A file
already present with the right sha256 is left alone; a download or a copy whose
sha256 differs is refused and nothing is written for it.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "viewer_assets.toml"
VENDOR = ROOT / "vendor"


def listed(config: Path = CONFIG) -> list[dict[str, Any]]:
    return list(tomllib.loads(config.read_text())["file"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(entry: dict[str, Any], vendor: Path, source: Path | None) -> str:
    """Put one file in place; say what was done."""

    target = vendor / entry["path"]
    if target.is_file() and sha256(target) == entry["sha256"]:
        return f"{entry['path']}: already here"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if source is not None:
        found = source / entry["path"]
        if not found.is_file():
            found = source / Path(entry["path"]).name
        shutil.copyfile(found, partial)
        origin = str(found)
    else:
        with urllib.request.urlopen(entry["url"], timeout=120) as response, \
                partial.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        origin = entry["url"]
    got = sha256(partial)
    if got != entry["sha256"]:
        partial.unlink()
        raise ValueError(f"{entry['path']}: sha256 {got} from {origin}, expected {entry['sha256']}")
    partial.replace(target)
    return f"{entry['path']}: {target.stat().st_size} bytes from {origin}, sha256 matches"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from", dest="source", type=Path, default=None,
                        help="a folder holding the files, for a node without internet")
    parser.add_argument("--vendor", type=Path, default=VENDOR,
                        help=f"where to put them; default {VENDOR}")
    args = parser.parse_args(argv)
    failed = 0
    for entry in listed():
        try:
            print(fetch(entry, args.vendor, args.source))
        except (OSError, ValueError) as exc:
            print(f"{entry['path']}: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
