from __future__ import annotations

import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

EMPIAR_10164_BASE_URL = "https://ftp.ebi.ac.uk/empiar/world_availability/10164/data/"
DEFAULT_TILT_SERIES = ("TS_01", "TS_43")
_HREF_RE = re.compile(r'href="([^"]+)"')
_FRAME_RE = re.compile(r"^(?P<series>TS_\d+)_(?P<index>\d+)_[-\d.]+\.mrc$")


@dataclass(frozen=True)
class RemoteFile:
    """One remote EMPIAR file and its local relative path."""

    url: str
    relative_path: Path


def build_empiar_10164_file_list(
    tilt_series: Iterable[str] = DEFAULT_TILT_SERIES,
    *,
    base_url: str = EMPIAR_10164_BASE_URL,
    listing_reader: Callable[[str], str] | None = None,
) -> list[RemoteFile]:
    """Build the download list for selected EMPIAR-10164 tilt-series.

    The frame list is discovered from the public directory listing instead of
    hard-coding tilt angles, because this keeps the downloader robust if a user
    selects a different tilt-series later.
    """

    requested = tuple(tilt_series)
    if not requested:
        raise ValueError("at least one tilt-series id is required")

    frames_url = urljoin(base_url, "frames/")
    mdocs_url = urljoin(base_url, "mdoc-files/")
    reader = listing_reader or read_text_url
    frame_names = extract_listing_hrefs(reader(frames_url))

    files: list[RemoteFile] = []
    for series in requested:
        selected_frames = select_frame_files(frame_names, series)
        if not selected_frames:
            raise ValueError(f"no frame files found for {series} in {frames_url}")

        files.append(
            RemoteFile(
                url=urljoin(mdocs_url, f"{series}.mrc.mdoc"),
                relative_path=Path("data") / "mdoc-files" / f"{series}.mrc.mdoc",
            )
        )
        files.extend(
            RemoteFile(
                url=urljoin(frames_url, frame_name),
                relative_path=Path("data") / "frames" / frame_name,
            )
            for frame_name in selected_frames
        )

    return files


def extract_listing_hrefs(html: str) -> list[str]:
    """Extract file names from an Apache-style directory listing."""

    return [href for href in _HREF_RE.findall(html) if not href.startswith("?")]


def select_frame_files(frame_names: Iterable[str], tilt_series_id: str) -> list[str]:
    """Select and acquisition-order-sort frame files for one tilt-series."""

    prefix = f"{tilt_series_id}_"
    selected = [name for name in frame_names if name.startswith(prefix) and name.endswith(".mrc")]
    return sorted(selected, key=_frame_sort_key)


def download_files(
    files: Iterable[RemoteFile],
    *,
    output_root: Path,
    overwrite: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Download files with `.part` resume support."""

    for remote_file in files:
        destination = output_root / remote_file.relative_path
        download_file(
            remote_file.url,
            destination,
            overwrite=overwrite,
            chunk_size=chunk_size,
            progress=progress,
        )


def download_file(
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Download one file, resuming from `destination.part` when present."""

    if destination.exists() and not overwrite:
        _emit(progress, f"skip existing {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if overwrite:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    resume_at = partial.stat().st_size if partial.exists() else 0
    request = Request(url)
    if resume_at:
        request.add_header("Range", f"bytes={resume_at}-")
        mode = "ab"
        _emit(progress, f"resume {destination} from {resume_at} bytes")
    else:
        mode = "wb"
        _emit(progress, f"download {destination}")

    try:
        with urlopen(request) as response:
            if resume_at and getattr(response, "status", None) != 206:
                partial.unlink(missing_ok=True)
                mode = "wb"
                _emit(progress, f"server did not resume; restart {destination}")
            with partial.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=chunk_size)
    except HTTPError as exc:
        if exc.code == 416 and resume_at:
            partial.replace(destination)
            _emit(progress, f"complete {destination}")
            return
        raise

    partial.replace(destination)
    _emit(progress, f"complete {destination}")


def read_text_url(url: str) -> str:
    with urlopen(url) as response:
        text: str = response.read().decode("utf-8", errors="replace")
        return text


def _frame_sort_key(name: str) -> tuple[str, int, str]:
    match = _FRAME_RE.match(name)
    if match is None:
        return (name, -1, name)
    return (match.group("series"), int(match.group("index")), name)


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
