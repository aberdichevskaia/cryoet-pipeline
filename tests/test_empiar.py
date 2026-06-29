from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.empiar import (
    build_empiar_10164_file_list,
    download_file,
    extract_listing_hrefs,
    select_frame_files,
)
from cryoet_pipeline.mrc_validation import IncompleteMrcError

LISTING = """
<a href="?C=N;O=D">Name</a>
<a href="TS_01_002_-3.0.mrc">TS_01_002_-3.0.mrc</a>
<a href="TS_43_001_3.0.mrc">TS_43_001_3.0.mrc</a>
<a href="TS_01_000_0.0.mrc">TS_01_000_0.0.mrc</a>
<a href="TS_01_001_3.0.mrc">TS_01_001_3.0.mrc</a>
<a href="TS_02_000_0.0.mrc">TS_02_000_0.0.mrc</a>
"""


def test_extract_listing_hrefs_ignores_sort_links() -> None:
    assert extract_listing_hrefs(LISTING) == [
        "TS_01_002_-3.0.mrc",
        "TS_43_001_3.0.mrc",
        "TS_01_000_0.0.mrc",
        "TS_01_001_3.0.mrc",
        "TS_02_000_0.0.mrc",
    ]


def test_select_frame_files_sorts_by_tilt_index() -> None:
    frame_names = extract_listing_hrefs(LISTING)

    assert select_frame_files(frame_names, "TS_01") == [
        "TS_01_000_0.0.mrc",
        "TS_01_001_3.0.mrc",
        "TS_01_002_-3.0.mrc",
    ]


def test_build_empiar_10164_file_list_adds_mdoc_and_frames() -> None:
    def reader(url: str) -> str:
        assert url.endswith("/frames/") or url.endswith("data/frames/")
        return LISTING

    files = build_empiar_10164_file_list(["TS_01"], listing_reader=reader)

    assert [file.relative_path for file in files] == [
        Path("data/mdoc-files/TS_01.mrc.mdoc"),
        Path("data/frames/TS_01_000_0.0.mrc"),
        Path("data/frames/TS_01_001_3.0.mrc"),
        Path("data/frames/TS_01_002_-3.0.mrc"),
    ]


def test_download_file_keeps_incomplete_mrc_as_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete_mrc = _mrc_bytes(tmp_path)
    destination = tmp_path / "downloaded.mrc"
    incomplete_response = _FakeResponse(complete_mrc[:-4])
    monkeypatch.setattr(
        "cryoet_pipeline.empiar.urlopen",
        lambda request: incomplete_response,
    )

    with pytest.raises(IncompleteMrcError, match="incomplete MRC file"):
        download_file("https://example.test/movie.mrc", destination)

    assert not destination.exists()
    assert destination.with_suffix(".mrc.part").exists()


def test_download_file_resumes_incomplete_existing_mrc(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete_mrc = _mrc_bytes(tmp_path)
    destination = tmp_path / "downloaded.mrc"
    resume_at = len(complete_mrc) // 2
    destination.write_bytes(complete_mrc[:resume_at])

    def fake_urlopen(request: Request) -> _FakeResponse:
        assert request.get_header("Range") == f"bytes={resume_at}-"
        return _FakeResponse(complete_mrc[resume_at:], status=206)

    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", fake_urlopen)

    download_file("https://example.test/movie.mrc", destination)

    assert destination.read_bytes() == complete_mrc
    assert not destination.with_suffix(".mrc.part").exists()


class _FakeResponse(BytesIO):
    def __init__(self, data: bytes, *, status: int = 200) -> None:
        super().__init__(data)
        self.status = status


def _mrc_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "source.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.arange(24, dtype=np.int8).reshape(3, 2, 4))
    return path.read_bytes()
