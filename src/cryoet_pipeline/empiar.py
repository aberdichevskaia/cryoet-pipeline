from __future__ import annotations

import fnmatch
import http.client
import re
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urljoin
from urllib.request import Request, urlopen

import yaml

from cryoet_pipeline.mrc_validation import IncompleteMrcError, validate_complete_mrc

EMPIAR_BASE_URL_TEMPLATE = "https://ftp.ebi.ac.uk/empiar/world_availability/{empiar_id}/data/"
DEFAULT_EMPIAR_ID = "10164"
DEFAULT_TILT_SERIES = ("TS_01", "TS_43")
_HREF_RE = re.compile(r'href="([^"]+)"')
_TILT_SERIES_PLACEHOLDER = "{tilt_series}"
_LAYOUT_GROUP_KEYS = ("remote_dir", "file_pattern", "local_dir", "shared")

# EBI's file server comes and goes: on 4 and 5 November 2026 it refused
# connections for minutes at a time, and answered a download running at MDC
# with 403 Forbidden after three files had come through. Such a failure is
# retried after a wait that doubles from DEFAULT_RETRY_WAIT_S up to
# MAX_RETRY_WAIT_S: 30, 60, 120, 240 and 480 s, about 15 minutes in all.
DEFAULT_RETRIES = 5
DEFAULT_RETRY_WAIT_S = 30.0
MAX_RETRY_WAIT_S = 600.0
# 403 is among them because EBI's came and went; a 404 is a wrong path, and
# waiting does not mend it.
_TRANSIENT_HTTP_CODES = frozenset({403, 408, 425, 429, 500, 502, 503, 504})

_T = TypeVar("_T")


class IncompleteDownloadError(OSError):
    """A download ended before the server's announced size; its `.part` is kept."""


class DownloadGaveUp(OSError):
    """A transient failure outlasted every retry; what arrived is kept for a rerun."""


@dataclass(frozen=True)
class RemoteFile:
    """One remote EMPIAR file and its local relative path."""

    url: str
    relative_path: Path


@dataclass(frozen=True)
class FetchGroup:
    """One remote subdirectory plus the filename pattern to select from it.

    `{tilt_series}` is replaced by each requested tilt-series id in
    `remote_dir`, in `file_pattern` (a glob: `*`/`?` wildcards) and in
    `local_dir`, and the pattern is then matched against the live directory
    listing. A group has to name the series in its remote folder or in its
    file pattern, so that it cannot take every series' files for each one
    asked for: EMPIAR-10164 names it in the files (`{tilt_series}_*.mrc` in
    `frames/`), EMPIAR-11830 in the folders (`{tilt_series}/frames/*.eer`).

    A `shared` group is the exception. It is fetched once, whatever series are
    asked for, and so must not name one: a session's gain reference, for
    example.

    Files land under the download root in `local_dir` when it is given, and
    otherwise in `data/<remote_dir>`, the deposition's own folders.
    """

    remote_dir: str
    file_pattern: str
    local_dir: str | None = None
    shared: bool = False

    def __post_init__(self) -> None:
        names_series = (
            _TILT_SERIES_PLACEHOLDER in self.remote_dir
            or _TILT_SERIES_PLACEHOLDER in self.file_pattern
        )
        if self.shared:
            if names_series or (
                self.local_dir is not None and _TILT_SERIES_PLACEHOLDER in self.local_dir
            ):
                raise ValueError(
                    f"a shared group is fetched once for all series, so it cannot use "
                    f"the {_TILT_SERIES_PLACEHOLDER!r} placeholder: "
                    f"{self.remote_dir!r}, {self.file_pattern!r}"
                )
        elif not names_series:
            raise ValueError(
                f"file_pattern {self.file_pattern!r} or remote_dir {self.remote_dir!r} "
                f"must contain the {_TILT_SERIES_PLACEHOLDER!r} placeholder; a file "
                "fetched once for all series is a group with shared: true"
            )
        if self.local_dir is not None:
            local = PurePosixPath(self.local_dir)
            if not self.local_dir.strip() or local.is_absolute() or ".." in local.parts:
                raise ValueError(
                    f"local_dir {self.local_dir!r} must be a folder inside the download "
                    "root: relative, without '..'"
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
    requested tilt-series id within it. Two keys are optional: `local_dir`,
    the folder under the download root the files go to, and `shared: true`
    for a file fetched once for all series (see `FetchGroup`). Any other key
    is refused, so that a misspelt `local_dir` cannot leave the files in the
    default place unnoticed.
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
        unknown = sorted(str(key) for key in entry if key not in _LAYOUT_GROUP_KEYS)
        if unknown:
            raise ValueError(
                f"{path}: groups[{index}] has unknown keys {', '.join(unknown)}; "
                f"a group takes {', '.join(_LAYOUT_GROUP_KEYS)}"
            )
        try:
            remote_dir = entry["remote_dir"]
            file_pattern = entry["file_pattern"]
        except KeyError as exc:
            raise ValueError(
                f"{path}: groups[{index}] is missing required key {exc.args[0]!r}"
            ) from exc
        if not isinstance(remote_dir, str) or not isinstance(file_pattern, str):
            raise ValueError(f"{path}: groups[{index}] 'remote_dir'/'file_pattern' must be strings")
        local_dir = entry.get("local_dir")
        if local_dir is not None and not isinstance(local_dir, str):
            raise ValueError(f"{path}: groups[{index}] 'local_dir' must be a string")
        shared = entry.get("shared", False)
        if not isinstance(shared, bool):
            raise ValueError(f"{path}: groups[{index}] 'shared' must be true or false")
        try:
            groups.append(
                FetchGroup(
                    remote_dir=remote_dir,
                    file_pattern=file_pattern,
                    local_dir=local_dir,
                    shared=shared,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path}: groups[{index}]: {exc}") from exc

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
    retries: int = DEFAULT_RETRIES,
    retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[RemoteFile]:
    """Build the download list for selected tilt-series of one EMPIAR deposition.

    `layout` declares which remote subdirectories to look in and which
    filename pattern identifies files belonging to a requested tilt-series id;
    the actual file discovery still happens against the live directory
    listing, so this stays robust to naming details the layout does not need
    to spell out (frame counts, tilt angles, acquisition order).

    Names are matched and saved as the listing decodes them. EBI's server
    escapes the brackets in names such as `..._001[-10.00]_EER.eer` as
    `%5b...%5d`, and an mdoc's `SubFramePath` names the file with brackets.
    Two remote files bound for the same local path are refused, since the
    second would otherwise be skipped as already downloaded. A listing that
    fails for a passing reason is fetched again, as `download_files` does.
    """

    requested = tuple(tilt_series)
    if not requested:
        raise ValueError("at least one tilt-series id is required")

    resolved_base_url = base_url if base_url is not None else empiar_base_url(empiar_id)
    read = listing_reader or read_text_url

    def reader(url: str) -> str:
        return call_with_retries(
            lambda attempt: read(url),
            what=url,
            retries=retries,
            retry_wait_s=retry_wait_s,
            progress=progress,
            sleep=sleep,
        )

    listings: dict[str, dict[str, str]] = {}

    files: list[RemoteFile] = []
    for group in layout.groups:
        for series in (None,) if group.shared else requested:
            remote_dir = _with_series(group.remote_dir, series).strip("/")
            group_url = urljoin(resolved_base_url, f"{quote(remote_dir)}/")
            if group_url not in listings:
                listings[group_url] = _listed_files(reader, group_url)
            hrefs = listings[group_url]
            matched_names = select_group_files(hrefs, group.file_pattern, series)
            if not matched_names:
                which = "" if series is None else f" for {series}"
                raise ValueError(
                    f"no files matched pattern {group.file_pattern!r}{which} in {group_url}"
                )
            local_dir = (
                Path(_with_series(group.local_dir, series))
                if group.local_dir is not None
                else Path("data") / remote_dir
            )
            files.extend(
                RemoteFile(url=urljoin(group_url, hrefs[name]), relative_path=local_dir / name)
                for name in matched_names
            )

    return _refuse_collisions(files)


def extract_listing_hrefs(html: str) -> list[str]:
    """Extract file names from an Apache-style directory listing."""

    return [href for href in _HREF_RE.findall(html) if not href.startswith("?")]


def select_group_files(
    available_names: Iterable[str],
    file_pattern: str,
    tilt_series_id: str | None,
) -> list[str]:
    """Select and sort the files in one listing matching one group's pattern.

    `tilt_series_id` is None for a shared group, whose pattern names no series.
    """

    glob_pattern = _with_series(file_pattern, tilt_series_id)
    return sorted(name for name in available_names if fnmatch.fnmatch(name, glob_pattern))


def _with_series(text: str, tilt_series_id: str | None) -> str:
    if tilt_series_id is None:
        return text
    return text.replace(_TILT_SERIES_PLACEHOLDER, tilt_series_id)


def _listed_files(reader: Callable[[str], str], url: str) -> dict[str, str]:
    """The files a listing shows, by decoded name, with the href that fetches each.

    Folders (hrefs ending in `/`) and links out of the folder (the parent,
    written as an absolute path) are left out: no pattern should match them.
    """

    try:
        html = reader(url)
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"no such folder: {url} (HTTP 404); check the tilt-series id and the "
                "layout's remote_dir"
            ) from exc
        raise
    return {
        unquote(href): href
        for href in extract_listing_hrefs(html)
        if not href.endswith("/") and not href.startswith("/") and "://" not in href
    }


def _refuse_collisions(files: list[RemoteFile]) -> list[RemoteFile]:
    by_path: dict[Path, RemoteFile] = {}
    unique: list[RemoteFile] = []
    for remote_file in files:
        seen = by_path.setdefault(remote_file.relative_path, remote_file)
        if seen is remote_file:
            unique.append(remote_file)
        elif seen.url != remote_file.url:
            raise ValueError(
                f"two remote files would be saved as {remote_file.relative_path}: "
                f"{seen.url} and {remote_file.url}; give their groups local_dir values "
                "that keep them apart"
            )
    return unique


def download_files(
    files: Iterable[RemoteFile],
    *,
    output_root: Path,
    overwrite: bool = False,
    chunk_size: int = 8 * 1024 * 1024,
    progress: Callable[[str], None] | None = None,
    retries: int = DEFAULT_RETRIES,
    retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Download files with `.part` resume support, retrying passing failures.

    A retry resumes from the `.part` the failed attempt left. `overwrite`
    applies to the first attempt only: a retry must not throw away what that
    same run has just fetched.
    """

    for remote_file in files:
        destination = output_root / remote_file.relative_path

        def attempt_download(
            attempt: int, url: str = remote_file.url, path: Path = destination
        ) -> None:
            download_file(
                url,
                path,
                overwrite=overwrite and attempt == 0,
                chunk_size=chunk_size,
                progress=progress,
            )

        call_with_retries(
            attempt_download,
            what=remote_file.url,
            retries=retries,
            retry_wait_s=retry_wait_s,
            progress=progress,
            sleep=sleep,
        )


def call_with_retries(
    action: Callable[[int], _T],
    *,
    what: str,
    retries: int = DEFAULT_RETRIES,
    retry_wait_s: float = DEFAULT_RETRY_WAIT_S,
    progress: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> _T:
    """`action(attempt)`, again after a wait each time it fails for a passing reason.

    A failure that waiting cannot mend (a 404, a refused layout) is raised at
    once. A passing one that outlasts `retries` becomes `DownloadGaveUp`.
    """

    if retries < 0:
        raise ValueError(f"retries must be 0 or more, not {retries}")
    attempt = 0
    while True:
        try:
            return action(attempt)
        except Exception as exc:
            if not is_transient_download_error(exc):
                raise
            reason = _failure_reason(exc)
            if attempt >= retries:
                raise DownloadGaveUp(
                    f"{reason}: {what}; gave up after {retries} "
                    f"{'retry' if retries == 1 else 'retries'}"
                ) from exc
            wait = _retry_wait(exc, attempt, retry_wait_s)
            _emit(
                progress,
                f"{reason}: {what}; retry {attempt + 1} of {retries} in {wait:.0f} s",
            )
            sleep(wait)
            attempt += 1


def is_transient_download_error(exc: BaseException) -> bool:
    """Whether waiting and asking again may get past `exc`."""

    if isinstance(exc, HTTPError):
        return exc.code in _TRANSIENT_HTTP_CODES
    # A transfer cut short: the .part is kept, and the next attempt resumes it.
    if isinstance(exc, (IncompleteDownloadError, IncompleteMrcError)):
        return True
    return isinstance(
        exc, (URLError, ConnectionError, TimeoutError, http.client.HTTPException)
    )


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, URLError):
        return f"server not reached ({exc.reason})"
    if isinstance(exc, (IncompleteDownloadError, IncompleteMrcError)):
        return "transfer stopped early"
    return f"{type(exc).__name__}: {exc}"


def _retry_wait(exc: BaseException, attempt: int, retry_wait_s: float) -> float:
    """Doubling from `retry_wait_s`, longer if the server asks for it, capped."""

    wait = retry_wait_s * 2.0**attempt
    if isinstance(exc, HTTPError) and exc.headers is not None:
        asked = str(exc.headers.get("Retry-After") or "").strip()
        if asked.isdigit():
            wait = max(wait, float(asked))
    return min(wait, MAX_RETRY_WAIT_S)


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

    expected_size: int | None = None
    try:
        with urlopen(request) as response:
            if resume_at and getattr(response, "status", None) != 206:
                partial.unlink(missing_ok=True)
                mode = "wb"
                resume_at = 0
                _emit(progress, f"server did not resume; restart {destination}")
            expected_size = _announced_size(response, resume_at)
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

    # Only an MRC can be checked by its own header. Anything else (EER movies,
    # gain references, mdocs) is held to the size the server announced: a
    # connection that closes early ends the copy without an error.
    received = partial.stat().st_size
    if expected_size is not None and received != expected_size:
        raise IncompleteDownloadError(
            f"{url}: the server announced {expected_size} bytes and {received} arrived; "
            f"download again to resume from {partial}"
        )
    if destination.suffix.lower() == ".mrc":
        validate_complete_mrc(partial)
    partial.replace(destination)
    _emit(progress, f"complete {destination}")


def _announced_size(response: Any, resume_at: int) -> int | None:
    """The whole file's size as the response announces it, or None.

    An answer to a Range request names the total in Content-Range
    (`bytes 100-199/1000`); a whole-file answer gives it as Content-Length.
    """

    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    content_range = headers.get("Content-Range")
    if content_range:
        total = str(content_range).rpartition("/")[2].strip()
        return int(total) if total.isdigit() else None
    length = headers.get("Content-Length")
    if length is not None and str(length).strip().isdigit():
        return resume_at + int(str(length).strip())
    return None


def read_text_url(url: str) -> str:
    with urlopen(url) as response:
        text: str = response.read().decode("utf-8", errors="replace")
        return text


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
