from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.empiar import (
    DEFAULT_EMPIAR_10164_LAYOUT,
    DatasetLayout,
    FetchGroup,
    build_empiar_file_list,
    download_file,
    empiar_base_url,
    extract_listing_hrefs,
    load_dataset_layout,
    select_group_files,
)
from cryoet_pipeline.mrc_validation import IncompleteMrcError

FRAME_LISTING = """
<a href="?C=N;O=D">Name</a>
<a href="TS_01_002_-3.0.mrc">TS_01_002_-3.0.mrc</a>
<a href="TS_43_001_3.0.mrc">TS_43_001_3.0.mrc</a>
<a href="TS_01_000_0.0.mrc">TS_01_000_0.0.mrc</a>
<a href="TS_01_001_3.0.mrc">TS_01_001_3.0.mrc</a>
<a href="TS_02_000_0.0.mrc">TS_02_000_0.0.mrc</a>
"""

MDOC_LISTING = """
<a href="TS_01.mrc.mdoc">TS_01.mrc.mdoc</a>
<a href="TS_43.mrc.mdoc">TS_43.mrc.mdoc</a>
"""

ONE_FILE_PER_SERIES_LISTING = """
<a href="TS_0002.mrc">TS_0002.mrc</a>
<a href="TS_0003.mrc">TS_0003.mrc</a>
"""


def test_extract_listing_hrefs_ignores_sort_links() -> None:
    assert extract_listing_hrefs(FRAME_LISTING) == [
        "TS_01_002_-3.0.mrc",
        "TS_43_001_3.0.mrc",
        "TS_01_000_0.0.mrc",
        "TS_01_001_3.0.mrc",
        "TS_02_000_0.0.mrc",
    ]


def test_select_group_files_matches_glob_pattern_and_sorts() -> None:
    names = extract_listing_hrefs(FRAME_LISTING)

    assert select_group_files(names, "{tilt_series}_*.mrc", "TS_01") == [
        "TS_01_000_0.0.mrc",
        "TS_01_001_3.0.mrc",
        "TS_01_002_-3.0.mrc",
    ]


def test_select_group_files_matches_exact_single_file_pattern() -> None:
    names = extract_listing_hrefs(ONE_FILE_PER_SERIES_LISTING)

    assert select_group_files(names, "{tilt_series}.mrc", "TS_0002") == ["TS_0002.mrc"]
    assert select_group_files(names, "{tilt_series}.mrc", "TS_9999") == []


def test_fetch_group_requires_tilt_series_placeholder() -> None:
    with pytest.raises(ValueError, match="tilt_series"):
        FetchGroup(remote_dir="frames", file_pattern="TS_01_*.mrc")


def test_build_empiar_file_list_with_default_10164_layout() -> None:
    def reader(url: str) -> str:
        if "mdoc-files" in url:
            return MDOC_LISTING
        return FRAME_LISTING

    files = build_empiar_file_list(
        "10164",
        ["TS_01"],
        DEFAULT_EMPIAR_10164_LAYOUT,
        listing_reader=reader,
    )

    assert [file.relative_path for file in files] == [
        Path("data/mdoc-files/TS_01.mrc.mdoc"),
        Path("data/frames/TS_01_000_0.0.mrc"),
        Path("data/frames/TS_01_001_3.0.mrc"),
        Path("data/frames/TS_01_002_-3.0.mrc"),
    ]
    assert files[0].url == (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10164/data/mdoc-files/TS_01.mrc.mdoc"
    )


def test_build_empiar_file_list_with_custom_layout() -> None:
    layout = DatasetLayout(
        groups=(FetchGroup(remote_dir="Untreated_dataset", file_pattern="{tilt_series}.mrc"),)
    )

    def reader(url: str) -> str:
        assert url.endswith("Untreated_dataset/")
        return ONE_FILE_PER_SERIES_LISTING

    files = build_empiar_file_list("11538", ["TS_0002"], layout, listing_reader=reader)

    assert [file.relative_path for file in files] == [Path("data/Untreated_dataset/TS_0002.mrc")]
    assert files[0].url == (
        "https://ftp.ebi.ac.uk/empiar/world_availability/11538/data/Untreated_dataset/TS_0002.mrc"
    )


def test_build_empiar_file_list_raises_when_pattern_matches_nothing() -> None:
    layout = DatasetLayout(
        groups=(FetchGroup(remote_dir="Untreated_dataset", file_pattern="{tilt_series}.mrc"),)
    )

    with pytest.raises(ValueError, match="no files matched"):
        build_empiar_file_list(
            "11538",
            ["TS_9999"],
            layout,
            listing_reader=lambda url: ONE_FILE_PER_SERIES_LISTING,
        )


def test_empiar_base_url_rejects_non_numeric_id() -> None:
    with pytest.raises(ValueError, match="numeric"):
        empiar_base_url("not-a-number")


def test_load_dataset_layout_parses_groups(tmp_path: Path) -> None:
    path = tmp_path / "layout.yaml"
    path.write_text(
        "groups:\n"
        "  - remote_dir: mdoc_files\n"
        '    file_pattern: "{tilt_series}.mrc.mdoc"\n'
        "  - remote_dir: Untreated_dataset\n"
        '    file_pattern: "{tilt_series}.mrc"\n'
    )

    layout = load_dataset_layout(path)

    assert layout.groups == (
        FetchGroup(remote_dir="mdoc_files", file_pattern="{tilt_series}.mrc.mdoc"),
        FetchGroup(remote_dir="Untreated_dataset", file_pattern="{tilt_series}.mrc"),
    )


def test_load_dataset_layout_rejects_missing_groups_key(tmp_path: Path) -> None:
    path = tmp_path / "layout.yaml"
    path.write_text("not_groups: []\n")

    with pytest.raises(ValueError, match="top-level 'groups'"):
        load_dataset_layout(path)


def test_load_dataset_layout_rejects_group_missing_required_key(tmp_path: Path) -> None:
    path = tmp_path / "layout.yaml"
    path.write_text("groups:\n  - remote_dir: frames\n")

    with pytest.raises(ValueError, match="file_pattern"):
        load_dataset_layout(path)


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
