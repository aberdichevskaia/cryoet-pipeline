from __future__ import annotations

import fnmatch
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import yaml

from cryoet_pipeline.mrc_validation import IncompleteMrcError, validate_complete_mrc

EMPIAR_BASE_URL_TEMPLATE = "https://ftp.ebi.ac.uk/empiar/world_availability/{empiar_id}/data/"
DEFAULT_EMPIAR_ID = "10164"
DEFAULT_TILT_SERIES = ("TS_01", "TS_43")
_HREF_RE = re.compile(r'href="([^"]+)"')
_TILT_SERIES_PLACEHOLDER = "{tilt_series}"


@dataclass(frozen=True)
class RemoteFile:
    """One remote EMPIAR file and its local relative path."""

    url: str
    relative_path: Path


@dataclass(frozen=True)
class FetchGroup:
    """One remote subdirectory plus the filename pattern to select from it.

    `file_pattern` must contain the literal placeholder `{tilt_series}`, which
    is substituted with the requested tilt-series id before glob-matching
    (`*`/`?` wildcards supported) against the live directory listing.
    """

    remote_dir: str
    file_pattern: str

    def __post_init__(self) -> None:
        if _TILT_SERIES_PLACEHOLDER not in self.file_pattern:
            raise ValueError(
                f"file_pattern {self.file_pattern!r} must contain the "
                f"{_TILT_SERIES_PLACEHOLDER!r} placeholder"
            )


@dataclass(frozen=True)
class DatasetLayout:
    """Declarative description of where one EMPIAR deposition keeps its files."""

    groups: tuple[FetchGroup, ...]

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a dataset layout must contain at least one group")


DEFAULT_EMPIAR_10164_LAYOUT = DatasetLayout(
    groups=(
        FetchGroup(remote_dir="mdoc-files", file_pattern="{tilt_series}.mrc.mdoc"),
        FetchGroup(remote_dir="frames", file_pattern="{tilt_series}_*.mrc"),
    )
)


def load_dataset_layout(path: Path) -> DatasetLayout:
    """Load a user-authored dataset layout from a small YAML file.

    Expected shape::

        groups:
          - remote_dir: mdoc_files
            file_pattern: "{tilt_series}.mrc.mdoc"
          - remote_dir: Untreated_dataset
            file_pattern: "{tilt_series}.mrc"

    Each group names one remote subdirectory (relative to the deposition's
    `data/` root) and a glob pattern identifying files that belong to a
    requested tilt-series id within it.
    """

    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "groups" not in raw:
        raise ValueError(f"{path}: expected a mapping with a top-level 'groups' list")
    raw_groups = raw["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(f"{path}: 'groups' must be a non-empty list")

    groups: list[FetchGroup] = []
    for index, entry in enumerate(raw_groups):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: groups[{index}] must be a mapping")
        try:
            remote_dir = entry["remote_dir"]
            file_pattern = entry["file_pattern"]
        except KeyError as exc:
            raise ValueError(
                f"{path}: groups[{index}] is missing required key {exc.args[0]!r}"
            ) from exc
        if not isinstance(remote_dir, str) or not isinstance(file_pattern, str):
            raise ValueError(f"{path}: groups[{index}] 'remote_dir'/'file_pattern' must be strings")
        groups.append(FetchGroup(remote_dir=remote_dir, file_pattern=file_pattern))

    return DatasetLayout(groups=tuple(groups))


def empiar_base_url(empiar_id: str) -> str:
    """Return the public EBI FTP base URL for one EMPIAR deposition's data."""

    if not empiar_id or not empiar_id.isdigit():
        raise ValueError(f"EMPIAR id must be a numeric string, got {empiar_id!r}")
    return EMPIAR_BASE_URL_TEMPLATE.format(empiar_id=empiar_id)


def build_empiar_file_list(
    empiar_id: str,
    tilt_series: Iterable[str],
    layout: DatasetLayout,
    *,
    base_url: str | None = None,
    listing_reader: Callable[[str], str] | None = None,
) -> list[RemoteFile]:
    """Build the download list for selected tilt-series of one EMPIAR deposition.

    `layout` declares which remote subdirectories to look in and which
    filename pattern identifies files belonging to a requested tilt-series id;
    the actual file discovery still happens against the live directory
    listing, so this stays robust to naming details the layout does not need
    to spell out (frame counts, tilt angles, acquisition order).
    """

    requested = tuple(tilt_series)
    if not requested:
        raise ValueError("at least one tilt-series id is required")

    resolved_base_url = base_url if base_url is not None else empiar_base_url(empiar_id)
    reader = listing_reader or read_text_url

    files: list[RemoteFile] = []
    for group in layout.groups:
        group_url = urljoin(resolved_base_url, f"{group.remote_dir}/")
        available_names = extract_listing_hrefs(reader(group_url))
        for series in requested:
            matched_names = select_group_files(available_names, group.file_pattern, series)
            if not matched_names:
                raise ValueError(
                    f"no files matched pattern {group.file_pattern!r} for "
                    f"{series} in {group_url}"
                )
            files.extend(
                RemoteFile(
                    url=urljoin(group_url, name),
                    relative_path=Path("data") / group.remote_dir / name,
                )
                for name in matched_names
            )

    return files


def extract_listing_hrefs(html: str) -> list[str]:
    """Extract file names from an Apache-style directory listing."""

    return [href for href in _HREF_RE.findall(html) if not href.startswith("?")]


def select_group_files(
    available_names: Iterable[str],
    file_pattern: str,
    tilt_series_id: str,
) -> list[str]:
    """Select and sort the files in one listing matching one group's pattern."""

    glob_pattern = file_pattern.replace(_TILT_SERIES_PLACEHOLDER, tilt_series_id)
    return sorted(name for name in available_names if fnmatch.fnmatch(name, glob_pattern))


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
        if destination.suffix.lower() != ".mrc":
            _emit(progress, f"skip existing {destination}")
            return
        try:
            validate_complete_mrc(destination)
        except IncompleteMrcError as exc:
            partial = destination.with_suffix(destination.suffix + ".part")
            if partial.exists():
                raise FileExistsError(
                    f"both incomplete destination and partial download exist: "
                    f"{destination}, {partial}"
                ) from exc
            destination.replace(partial)
            _emit(progress, f"recover incomplete {destination}")
        else:
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
            if destination.suffix.lower() == ".mrc":
                validate_complete_mrc(partial)
            partial.replace(destination)
            _emit(progress, f"complete {destination}")
            return
        raise

    if destination.suffix.lower() == ".mrc":
        validate_complete_mrc(partial)
    partial.replace(destination)
    _emit(progress, f"complete {destination}")


def read_text_url(url: str) -> str:
    with urlopen(url) as response:
        text: str = response.read().decode("utf-8", errors="replace")
        return text


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
