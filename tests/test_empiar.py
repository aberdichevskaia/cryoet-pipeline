from pathlib import Path

from cryoet_pipeline.empiar import build_empiar_10164_file_list, extract_listing_hrefs, select_frame_files


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
