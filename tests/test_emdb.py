from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any

import pytest

from cryoet_pipeline import emdb
from cryoet_pipeline.emdb import emdb_map_url, fetch_emdb_map, normalize_emdb_id


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2638", "2638"),
        ("EMD-2638", "2638"),
        ("emd_2638", "2638"),
        ("emd2638", "2638"),
        ("  EMD-11538 ", "11538"),
    ],
)
def test_normalize_emdb_id_accepts_common_spellings(raw: str, expected: str) -> None:
    assert normalize_emdb_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "EMD-", "12", "EMD-123456"])
def test_normalize_emdb_id_rejects_nonsense(raw: str) -> None:
    with pytest.raises(ValueError, match="unrecognized EMDB accession"):
        normalize_emdb_id(raw)


def test_emdb_map_url_matches_the_published_layout() -> None:
    assert emdb_map_url("EMD-2638") == (
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2638/map/emd_2638.map.gz"
    )


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def test_fetch_emdb_map_downloads_and_decompresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"MAP payload bytes"
    requested: list[str] = []

    def fake_urlopen(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        requested.append(url)
        return _FakeResponse(gzip.compress(payload))

    monkeypatch.setattr(emdb, "urlopen", fake_urlopen)

    result = fetch_emdb_map("EMD-2638", cache_dir=tmp_path / "cache")

    assert result == tmp_path / "cache" / "emd_2638.map"
    assert result.read_bytes() == payload
    assert requested == [emdb_map_url("2638")]
    # The compressed archive is not kept around after decompression.
    assert not (tmp_path / "cache" / "emd_2638.map.gz").exists()


def test_fetch_emdb_map_reuses_the_cached_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "emd_2638.map"
    cached.write_bytes(b"already here")
    calls: list[str] = []

    def fake_urlopen(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        calls.append(url)
        return _FakeResponse(gzip.compress(b"fresh"))

    monkeypatch.setattr(emdb, "urlopen", fake_urlopen)
    messages: list[str] = []

    result = fetch_emdb_map("2638", cache_dir=cache_dir, progress=messages.append)

    assert result.read_bytes() == b"already here"
    assert calls == []
    assert any("cached" in message for message in messages)


def test_fetch_emdb_map_overwrite_refetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "emd_2638.map").write_bytes(b"stale")

    def fake_urlopen(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(gzip.compress(b"fresh"))

    monkeypatch.setattr(emdb, "urlopen", fake_urlopen)

    result = fetch_emdb_map("2638", cache_dir=cache_dir, overwrite=True)

    assert result.read_bytes() == b"fresh"
