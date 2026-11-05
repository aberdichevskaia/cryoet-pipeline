from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.empiar import (
    DEFAULT_EMPIAR_10164_LAYOUT,
    DatasetLayout,
    DownloadGaveUp,
    FetchGroup,
    IncompleteDownloadError,
    RemoteFile,
    build_empiar_file_list,
    call_with_retries,
    download_file,
    download_files,
    empiar_base_url,
    extract_listing_hrefs,
    load_dataset_layout,
    select_group_files,
)
from cryoet_pipeline.mrc_validation import IncompleteMrcError

REPO = Path(__file__).resolve().parents[1]

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


# EMPIAR-11830 keeps each series in a folder of its own, and the gain per session
# (listings shortened from the live ones, 4 November 2026).
SERIES_FOLDER_LISTING = """
<a href="/empiar/world_availability/11830/data/chlamy_visual_proteomics/">Parent</a>
<a href="{series}.mdoc">{series}.mdoc</a>
<a href="{series}.st">{series}.st</a>
<a href="AreTomo/">AreTomo/</a>
<a href="frames/">frames/</a>
"""

FRAMES_2023_LISTING = """
<a href="?C=N;O=D">Name</a>
<a href="/empiar/world_availability/11830/data/chlamy_visual_proteomics/S1/">Parent</a>
<a href="Position_1_002_-12.01_20230801_175126_EER.eer">x</a>
<a href="Position_1_001_-15.01_20230801_175104_EER.eer">x</a>
"""

# EBI escapes the brackets that the 2021 names carry.
FRAMES_2021_LISTING = """
<a href="2021201_BrnoKrios_arctis_lam1_pos3_001%5b-10.00%5d_EER.eer">x</a>
<a href="2021201_BrnoKrios_arctis_lam1_pos3_002%5b-7.00%5d_EER.eer">x</a>
"""

GAIN_LISTING = """
<a href="20230807_182744_EER_GainReference.gain">x</a>
"""


def _series_folder_reader(frames: dict[str, str], calls: list[str] | None = None):
    def reader(url: str) -> str:
        if calls is not None:
            calls.append(url)
        if url.endswith("/gainref/"):
            return GAIN_LISTING
        for series, listing in frames.items():
            if url.endswith(f"/{series}/frames/"):
                return listing
            if url.endswith(f"/{series}/"):
                return SERIES_FOLDER_LISTING.format(series=series)
        raise HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    return reader


SERIES_FOLDER_LAYOUT = DatasetLayout(
    groups=(
        FetchGroup(
            remote_dir="tomo/{tilt_series}",
            file_pattern="{tilt_series}.mdoc",
            local_dir="mdocs",
        ),
        FetchGroup(
            remote_dir="tomo/{tilt_series}/frames", file_pattern="*.eer", local_dir="frames"
        ),
        FetchGroup(
            remote_dir="tomo/01082023/gainref",
            file_pattern="*.gain",
            local_dir="gain",
            shared=True,
        ),
    )
)


def test_a_group_can_name_the_series_in_its_folder_and_choose_where_files_go() -> None:
    files = build_empiar_file_list(
        "11830",
        ["S1"],
        SERIES_FOLDER_LAYOUT,
        listing_reader=_series_folder_reader({"S1": FRAMES_2023_LISTING}),
    )

    assert [file.relative_path for file in files] == [
        Path("mdocs/S1.mdoc"),
        Path("frames/Position_1_001_-15.01_20230801_175104_EER.eer"),
        Path("frames/Position_1_002_-12.01_20230801_175126_EER.eer"),
        Path("gain/20230807_182744_EER_GainReference.gain"),
    ]
    assert files[1].url == (
        "https://ftp.ebi.ac.uk/empiar/world_availability/11830/data/tomo/S1/frames/"
        "Position_1_001_-15.01_20230801_175104_EER.eer"
    )


def test_a_shared_group_is_listed_and_fetched_once_for_all_series() -> None:
    calls: list[str] = []
    files = build_empiar_file_list(
        "11830",
        ["S1", "S2"],
        SERIES_FOLDER_LAYOUT,
        listing_reader=_series_folder_reader(
            {"S1": FRAMES_2023_LISTING, "S2": FRAMES_2021_LISTING}, calls
        ),
    )

    gains = [file for file in files if file.relative_path.parent == Path("gain")]
    assert len(gains) == 1
    assert sum(url.endswith("/gainref/") for url in calls) == 1


def test_escaped_names_are_matched_and_saved_as_the_mdoc_names_them() -> None:
    files = build_empiar_file_list(
        "11830",
        ["S2"],
        SERIES_FOLDER_LAYOUT,
        listing_reader=_series_folder_reader({"S2": FRAMES_2021_LISTING}),
    )

    frames = [file for file in files if file.relative_path.parent == Path("frames")]
    assert frames[0].relative_path.name == "2021201_BrnoKrios_arctis_lam1_pos3_001[-10.00]_EER.eer"
    # The request keeps the server's own spelling.
    assert frames[0].url.endswith("lam1_pos3_001%5b-10.00%5d_EER.eer")


def test_folders_and_the_parent_link_in_a_listing_are_not_files() -> None:
    layout = DatasetLayout(groups=(FetchGroup(remote_dir="tomo/{tilt_series}", file_pattern="*"),))

    files = build_empiar_file_list(
        "11830",
        ["S1"],
        layout,
        listing_reader=_series_folder_reader({"S1": FRAMES_2023_LISTING}),
    )

    assert sorted(file.relative_path.name for file in files) == ["S1.mdoc", "S1.st"]


def test_two_remote_files_bound_for_one_local_path_are_refused() -> None:
    # Both series hold a file of the same name, and both go to frames/.
    with pytest.raises(ValueError, match="would be saved as frames/Position_1_001"):
        build_empiar_file_list(
            "11830",
            ["S1", "S2"],
            SERIES_FOLDER_LAYOUT,
            listing_reader=_series_folder_reader(
                {"S1": FRAMES_2023_LISTING, "S2": FRAMES_2023_LISTING}
            ),
        )


def test_without_local_dir_files_keep_the_depositions_own_folders() -> None:
    layout = DatasetLayout(
        groups=(FetchGroup(remote_dir="tomo/{tilt_series}/frames", file_pattern="*.eer"),)
    )

    files = build_empiar_file_list(
        "11830",
        ["S1"],
        layout,
        listing_reader=_series_folder_reader({"S1": FRAMES_2023_LISTING}),
    )

    assert files[0].relative_path == Path(
        "data/tomo/S1/frames/Position_1_001_-15.01_20230801_175104_EER.eer"
    )


def test_a_series_folder_that_does_not_exist_is_named() -> None:
    with pytest.raises(ValueError, match="no such folder: .*/tomo/S9/ .*tilt-series id"):
        build_empiar_file_list(
            "11830",
            ["S9"],
            SERIES_FOLDER_LAYOUT,
            listing_reader=_series_folder_reader({"S1": FRAMES_2023_LISTING}),
        )


def test_a_shared_group_cannot_name_the_series() -> None:
    with pytest.raises(ValueError, match="shared group"):
        FetchGroup(remote_dir="tomo/{tilt_series}", file_pattern="*.gain", shared=True)


@pytest.mark.parametrize("local_dir", ["/abs/frames", "../frames", "frames/../../x", " "])
def test_local_dir_stays_inside_the_download_root(local_dir: str) -> None:
    with pytest.raises(ValueError, match="local_dir"):
        FetchGroup(remote_dir="frames", file_pattern="{tilt_series}_*.mrc", local_dir=local_dir)


def test_load_dataset_layout_reads_local_dir_and_shared(tmp_path: Path) -> None:
    path = tmp_path / "layout.yaml"
    path.write_text(
        "groups:\n"
        "  - remote_dir: tomo/{tilt_series}/frames\n"
        '    file_pattern: "*.eer"\n'
        "    local_dir: frames\n"
        "  - remote_dir: tomo/01082023/gainref\n"
        '    file_pattern: "*.gain"\n'
        "    shared: true\n"
    )

    layout = load_dataset_layout(path)

    assert layout.groups == (
        FetchGroup(
            remote_dir="tomo/{tilt_series}/frames", file_pattern="*.eer", local_dir="frames"
        ),
        FetchGroup(remote_dir="tomo/01082023/gainref", file_pattern="*.gain", shared=True),
    )


def test_load_dataset_layout_refuses_a_misspelt_key(tmp_path: Path) -> None:
    path = tmp_path / "layout.yaml"
    path.write_text(
        "groups:\n"
        "  - remote_dir: frames\n"
        '    file_pattern: "{tilt_series}_*.mrc"\n'
        "    local-dir: frames\n"
    )

    with pytest.raises(ValueError, match="unknown keys local-dir"):
        load_dataset_layout(path)


def test_load_dataset_layout_names_the_group_a_rule_refuses(tmp_path: Path) -> None:
    path = tmp_path / "layout.yaml"
    path.write_text("groups:\n  - remote_dir: tomo/gainref\n    file_pattern: '*.gain'\n")

    with pytest.raises(ValueError, match=r"groups\[0\]: .*shared: true"):
        load_dataset_layout(path)


def test_the_11830_layout_fetches_mdoc_frames_and_the_sessions_gain() -> None:
    layout = load_dataset_layout(REPO / "configs/empiar-layouts/11830.yaml")
    series = "01082023_BrnoKrios_Arctis_WebUI_Position_1"

    def reader(url: str) -> str:
        if url.endswith("/chlamy_visual_proteomics/01082023/gainref/"):
            return GAIN_LISTING
        if url.endswith(f"/chlamy_visual_proteomics/{series}/frames/"):
            return FRAMES_2023_LISTING
        if url.endswith(f"/chlamy_visual_proteomics/{series}/"):
            return SERIES_FOLDER_LISTING.format(series=series)
        raise AssertionError(f"unexpected listing {url}")

    files = build_empiar_file_list("11830", [series], layout, listing_reader=reader)

    assert [str(file.relative_path) for file in files] == [
        f"mdocs/{series}.mdoc",
        "frames/Position_1_001_-15.01_20230801_175104_EER.eer",
        "frames/Position_1_002_-12.01_20230801_175126_EER.eer",
        "gain/20230807_182744_EER_GainReference.gain",
    ]


def test_a_download_shorter_than_announced_is_kept_as_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "movie.eer"
    response = _FakeResponse(b"x" * 60, headers={"Content-Length": "100"})
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", lambda request: response)

    with pytest.raises(IncompleteDownloadError, match="announced 100 bytes and 60 arrived"):
        download_file("https://example.test/movie.eer", destination)

    assert not destination.exists()
    assert destination.with_suffix(".eer.part").stat().st_size == 60


def test_a_resumed_download_is_checked_against_the_whole_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "gain.gain"
    destination.with_suffix(".gain.part").write_bytes(b"a" * 60)

    def fake_urlopen(request: Request) -> _FakeResponse:
        assert request.get_header("Range") == "bytes=60-"
        return _FakeResponse(
            b"b" * 40,
            status=206,
            headers={"Content-Range": "bytes 60-99/100", "Content-Length": "40"},
        )

    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", fake_urlopen)

    download_file("https://example.test/gain.gain", destination)

    assert destination.read_bytes() == b"a" * 60 + b"b" * 40


def test_a_restarted_download_is_held_to_the_whole_files_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The server ignored the Range header and sent the whole file from byte 0.
    destination = tmp_path / "movie.eer"
    destination.with_suffix(".eer.part").write_bytes(b"a" * 60)
    response = _FakeResponse(b"c" * 100, status=200, headers={"Content-Length": "100"})
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", lambda request: response)

    download_file("https://example.test/movie.eer", destination)

    assert destination.read_bytes() == b"c" * 100


class _Server:
    """Answers each request with the next of `answers`: an exception to raise, or a
    _FakeResponse; records the Range header of every request."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.ranges: list[str | None] = []

    def __call__(self, request: Request) -> object:
        self.ranges.append(request.get_header("Range"))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _http_error(code: int, headers: dict[str, str] | None = None) -> HTTPError:
    return HTTPError("https://example.test/x", code, "Forbidden", hdrs=headers, fp=None)  # type: ignore[arg-type]


def test_a_403_is_waited_out_and_the_file_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server(
        _http_error(403),
        _http_error(403),
        _FakeResponse(b"x" * 100, headers={"Content-Length": "100"}),
    )
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", server)
    waits: list[float] = []
    messages: list[str] = []
    remote = RemoteFile(
        url="https://example.test/movie.eer", relative_path=Path("frames/movie.eer")
    )

    download_files([remote], output_root=tmp_path, progress=messages.append, sleep=waits.append)

    assert (tmp_path / "frames/movie.eer").read_bytes() == b"x" * 100
    assert waits == [30.0, 60.0]
    assert "HTTP 403 Forbidden: https://example.test/movie.eer; retry 1 of 5 in 30 s" in messages


def test_a_dropped_transfer_resumes_where_it_stopped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server(
        _FakeResponse(b"a" * 60, headers={"Content-Length": "100"}),
        _FakeResponse(b"b" * 40, status=206, headers={"Content-Range": "bytes 60-99/100"}),
    )
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", server)
    remote = RemoteFile(url="https://example.test/gain.gain", relative_path=Path("gain.gain"))

    download_files([remote], output_root=tmp_path, sleep=lambda wait: None)

    assert (tmp_path / "gain.gain").read_bytes() == b"a" * 60 + b"b" * 40
    assert server.ranges == [None, "bytes=60-"]


def test_overwrite_does_not_throw_away_what_the_run_itself_fetched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "movie.eer").write_bytes(b"old")
    server = _Server(
        _FakeResponse(b"a" * 60, headers={"Content-Length": "100"}),
        _FakeResponse(b"b" * 40, status=206, headers={"Content-Range": "bytes 60-99/100"}),
    )
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", server)
    remote = RemoteFile(url="https://example.test/movie.eer", relative_path=Path("movie.eer"))

    download_files([remote], output_root=tmp_path, overwrite=True, sleep=lambda wait: None)

    assert (tmp_path / "movie.eer").read_bytes() == b"a" * 60 + b"b" * 40
    assert server.ranges == [None, "bytes=60-"]


def test_a_refused_connection_is_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server(
        URLError(ConnectionRefusedError(61, "Connection refused")),
        _FakeResponse(b"x" * 10, headers={"Content-Length": "10"}),
    )
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", server)
    remote = RemoteFile(url="https://example.test/m.eer", relative_path=Path("m.eer"))

    download_files([remote], output_root=tmp_path, sleep=lambda wait: None)

    assert (tmp_path / "m.eer").read_bytes() == b"x" * 10


def test_a_404_is_not_waited_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", _Server(_http_error(404)))
    waits: list[float] = []
    remote = RemoteFile(url="https://example.test/m.eer", relative_path=Path("m.eer"))

    with pytest.raises(HTTPError):
        download_files([remote], output_root=tmp_path, sleep=waits.append)

    assert waits == []


def test_after_the_last_retry_the_download_gives_up_and_keeps_its_part(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _Server(
        _FakeResponse(b"a" * 60, headers={"Content-Length": "100"}),
        _http_error(403),
        _http_error(403),
    )
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", server)
    waits: list[float] = []
    remote = RemoteFile(url="https://example.test/m.eer", relative_path=Path("m.eer"))

    with pytest.raises(DownloadGaveUp, match="HTTP 403 Forbidden: .*gave up after 2 retries"):
        download_files([remote], output_root=tmp_path, retries=2, sleep=waits.append)

    assert waits == [30.0, 60.0]
    assert (tmp_path / "m.eer.part").read_bytes() == b"a" * 60


def test_the_wait_doubles_up_to_ten_minutes_and_honours_retry_after() -> None:
    attempts: list[int] = []
    waits: list[float] = []

    def always_busy(attempt: int) -> None:
        attempts.append(attempt)
        raise _http_error(503, {"Retry-After": "100"} if attempt == 0 else None)

    with pytest.raises(DownloadGaveUp):
        call_with_retries(always_busy, what="x", retries=6, sleep=waits.append)

    assert attempts == list(range(7))
    assert waits == [100.0, 60.0, 120.0, 240.0, 480.0, 600.0]


def test_a_listing_is_fetched_again_after_a_403() -> None:
    answers: list[object] = [_http_error(403), GAIN_LISTING]
    layout = DatasetLayout(
        groups=(FetchGroup(remote_dir="g", file_pattern="*.gain", shared=True),)
    )

    def reader(url: str) -> str:
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return str(answer)

    files = build_empiar_file_list(
        "11830", ["S1"], layout, listing_reader=reader, sleep=lambda wait: None
    )

    assert [file.relative_path.name for file in files] == ["20230807_182744_EER_GainReference.gain"]


def test_the_command_says_how_to_resume_when_it_gives_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from cryoet_pipeline import cli

    layout = tmp_path / "layout.yaml"
    layout.write_text("groups:\n  - remote_dir: g\n    file_pattern: '*.gain'\n    shared: true\n")
    monkeypatch.setattr("cryoet_pipeline.empiar.read_text_url", lambda url: GAIN_LISTING)
    monkeypatch.setattr("cryoet_pipeline.empiar.urlopen", _Server(_http_error(403)))

    result = CliRunner().invoke(
        cli.app,
        ["download-empiar", "--out", str(tmp_path / "out"), "--empiar-id", "11830",
         "--tilt-series", "S1", "--layout", str(layout), "--retries", "0"],
    )

    assert result.exit_code == 1
    assert "download stopped: HTTP 403 Forbidden" in result.output
    assert "run the same command again" in result.output


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
    def __init__(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(data)
        self.status = status
        if headers is not None:
            self.headers = headers


def _mrc_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "source.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.arange(24, dtype=np.int8).reshape(3, 2, 4))
    return path.read_bytes()
