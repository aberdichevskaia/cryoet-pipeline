"""A project's classification runs and their resolution, as data the page draws from.

A lane's MRA runs sit in <project>/mra/<name> in the run-record layout
(docs/run_records.md), each a folder with a plan.json; the FSC between two half runs
sits beside them as <out>_fsc.json, with <out>_mean.mrc and <out>_mask.mrc
(scripts/consensus_halves.py maps). A run is read by the classification report's own
reader (`reports.classification_reader`), so the page and the PDF show the same
classes, the same members and the same maps.

Pictures. Each map is shown by every one of its z sections, stacked top to bottom
into one PNG for the page to step through, and by its central XY, XZ and YZ sections
side by side (`reports.classification.central_views`), on one grey scale per map as
the PDF draws a map's row: values as they are, y and z drawn upwards.

Maps for Mol*. The array the reader returns is written back as a standard MRC (mapc,
mapr, maps = 1, 2, 3) at the run's pixel size, into the cache, so that Mol* draws what
the pictures show whatever layout the run was written in; Mol*'s reading of a
standard MRC was checked on a phantom (docs/dashboard.md, Status). The FSC's own maps
are standard MRCs already and are served as they are.

Nothing here writes to the runs: everything written goes into the cache, under a key
made from the times their records were written, so a run that goes on is read again.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mrcfile  # type: ignore[import-untyped]
import numpy as np

from cryoet_pipeline.dashboard.data import png_bytes, sign_inside
from cryoet_pipeline.reports.imaging import display_range

CACHE_FORMAT = 1
PLAN = "plan.json"
FSC_SUFFIX = "_fsc.json"
CLASS_FILE = re.compile(r"^class_(\d+)_(\d+)(_sections\.png|_views\.png|\.mrc)$")
FSC_MAPS = ("mean.mrc", "mask.mrc")


def section_stack(volume: np.ndarray) -> np.ndarray:
    """Every z section of a (z, y, x) map, the first at the top, each drawn with y upwards.

    In PNG row order: section z fills rows z * Y to (z + 1) * Y - 1, and within it
    the row of y is Y - 1 - y.
    """

    return np.concatenate([section[::-1] for section in np.asarray(volume, np.float32)])


def view_strip(views: list[np.ndarray]) -> np.ndarray:
    """XY, XZ and YZ side by side with a transparent gap, in PNG row order, upwards.

    The views of a cubic box are of one size; a shorter one is padded above.
    """

    height = max(view.shape[0] for view in views)
    gap = np.full((height, max(1, views[0].shape[1] // 10)), np.nan, np.float32)
    parts: list[np.ndarray] = []
    for index, view in enumerate(views):
        flipped = np.asarray(view, np.float32)[::-1]
        if flipped.shape[0] < height:
            pad = np.full((height - flipped.shape[0], flipped.shape[1]), np.nan, np.float32)
            flipped = np.concatenate([pad, flipped])
        if index:
            parts.append(gap)
        parts.append(flipped)
    return np.concatenate(parts, axis=1)


def map_pictures(volume: np.ndarray) -> tuple[bytes, bytes]:
    """A map's z sections and its central views, as two PNGs on one grey scale."""

    from cryoet_pipeline.reports.classification import central_views

    volume = np.asarray(volume, np.float32)
    views = central_views(volume)
    limits = display_range([*views, *volume])
    return (png_bytes(section_stack(volume), limits, upwards=False),
            png_bytes(view_strip(views), limits, upwards=False))


def write_mrc(path: Path, volume: np.ndarray, spacing: float | None) -> None:
    """A (z, y, x) map as a standard MRC: X fastest, Y, then Z."""

    partial = path.with_name(path.name + ".partial")
    with mrcfile.new(partial, overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
        handle.voxel_size = spacing if spacing else 1.0
    partial.replace(path)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stamp(paths: list[Path]) -> list[list[Any]]:
    return [[str(path.relative_to(path.parents[1])), path.stat().st_mtime]
            for path in sorted(paths) if path.is_file()]


@dataclass
class _Run:
    key: str
    document: dict[str, Any]
    run: Any
    """The `ClassificationRun`, kept to load its maps."""
    lock: threading.Lock = field(default_factory=threading.Lock)


class ClassificationRuns:
    """The runs and FSC curves of one folder, <project>/mra, read when asked for."""

    def __init__(self, folder: Path, *, cache_dir: Path) -> None:
        self.folder = folder
        self.cache_dir = cache_dir
        self._runs: dict[str, _Run] = {}
        self._guard = threading.Lock()

    # ---- what there is ----------------------------------------------------------------

    def _run_names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(path.name for path in self.folder.iterdir()
                      if path.is_dir() and (path / PLAN).is_file())

    def _fsc_names(self) -> list[str]:
        if not self.folder.is_dir():
            return []
        return sorted(path.name[: -len(FSC_SUFFIX)] for path in self.folder.glob(f"*{FSC_SUFFIX}"))

    def listing(self) -> dict[str, Any]:
        runs = []
        for name in self._run_names():
            entry: dict[str, Any] = {"name": name}
            try:
                document = self.run(name)
                entry.update(finished=document["finished"], planned=document["planned"])
            except Exception as exc:  # noqa: BLE001 - one unreadable run must not hide the others
                entry["problem"] = f"{type(exc).__name__}: {exc}"
            runs.append(entry)
        return {"folder": str(self.folder), "runs": runs,
                "fsc": [{"name": name} for name in self._fsc_names()]}

    # ---- runs ----------------------------------------------------------------------------

    def _run_folder(self, name: str) -> Path:
        if name not in self._run_names():
            raise KeyError(f"no classification run {name} in {self.folder}")
        return self.folder / name

    def _key(self, folder: Path) -> str:
        records = [folder / PLAN, *folder.glob("*/run.json")]
        source = [CACHE_FORMAT, folder.name, _stamp(records)]
        return hashlib.sha1(json.dumps(source).encode()).hexdigest()[:16]

    def _loaded(self, name: str) -> _Run:
        from cryoet_pipeline.reports.classification_reader import read_classification

        folder = self._run_folder(name)
        key = self._key(folder)
        with self._guard:
            known = self._runs.get(name)
            if known is not None and known.key == key:
                return known
        run = read_classification(folder, compute_initial_average=False)
        loaded = _Run(key, run_document(run, key), run)
        with self._guard:
            self._runs[name] = loaded
        return loaded

    def run(self, name: str) -> dict[str, Any]:
        return self._loaded(name).document

    def run_file(self, name: str, key: str, file: str) -> bytes:
        """A class's pictures or its map, `class_<iteration>_<channel>...`; iteration 0 is
        the starting references. Or the alignment mask, `mask.mrc`."""

        loaded = self._loaded(name)
        if loaded.key != key:
            raise KeyError(f"{name} has gone on since this page was loaded; reload it")
        folder = self.cache_dir / "mra" / name / key
        path = folder / file
        if path.is_file():
            return path.read_bytes()
        run = loaded.run
        if file == "mask.mrc":
            if run.mask is None:
                raise KeyError(f"{name} has no alignment mask")
            volume = run.mask.load()
            kind = ".mrc"
            stem = "mask"
        else:
            match = CLASS_FILE.match(file)
            if match is None:
                raise KeyError(f"no file {file}")
            number, channel, kind = int(match[1]), int(match[2]), match[3]
            volume = _class_map(run, number, channel).load()
            stem = f"class_{number}_{channel}"
        with loaded.lock:
            if not path.is_file():
                folder.mkdir(parents=True, exist_ok=True)
                if kind == ".mrc":
                    write_mrc(folder / f"{stem}.mrc", volume, run.pixel_size_angstrom)
                else:
                    sections, views = map_pictures(volume)
                    (folder / f"{stem}_sections.png").write_bytes(sections)
                    (folder / f"{stem}_views.png").write_bytes(views)
        return path.read_bytes()

    # ---- FSC -----------------------------------------------------------------------------

    def _fsc_paths(self, name: str) -> dict[str, Path]:
        if name not in self._fsc_names():
            raise KeyError(f"no FSC {name} in {self.folder}")
        paths = {"json": self.folder / f"{name}{FSC_SUFFIX}"}
        for file in FSC_MAPS:
            path = self.folder / f"{name}_{file}"
            if path.is_file():
                paths[file] = path
        return paths

    def fsc(self, name: str) -> dict[str, Any]:
        paths = self._fsc_paths(name)
        key = hashlib.sha1(json.dumps(
            [CACHE_FORMAT, name, [[one, path.stat().st_mtime] for one, path in paths.items()]]
        ).encode()).hexdigest()[:16]
        document = fsc_document(json.loads(paths["json"].read_text()), name, key)
        document["maps"] = sorted(one for one in paths if one in FSC_MAPS)
        if {"mean.mrc", "mask.mrc"} <= set(paths):
            with mrcfile.open(paths["mean.mrc"], permissive=True) as mean, \
                    mrcfile.open(paths["mask.mrc"], permissive=True) as mask:
                sign = sign_inside(np.asarray(mean.data, np.float32),
                                   np.asarray(mask.data, np.float32))
        else:
            sign = None
        document["protein_sign"] = sign if sign is not None else -1
        document["protein_sign_source"] = (
            "measured: the mean map inside the mask against outside it" if sign is not None
            else "assumed: protein dark, as in the tomograms")
        return document

    def fsc_file(self, name: str, key: str, file: str) -> bytes:
        document = self.fsc(name)
        if document["key"] != key:
            raise KeyError(f"{name} has been written again since this page was loaded; reload it")
        paths = self._fsc_paths(name)
        if file not in FSC_MAPS or file not in paths:
            raise KeyError(f"no map {file} for {name}")
        return paths[file].read_bytes()


def _class_map(run: Any, number: int, channel: int) -> Any:
    if number == 0:
        volume = run.references.get(channel)
    else:
        iteration = next((one for one in run.iterations if one.number == number), None)
        result = iteration.classes.get(channel) if iteration is not None else None
        volume = result.average if result is not None else None
    if volume is None:
        raise KeyError(f"no map for channel {channel} at iteration {number}")
    return volume


def run_document(run: Any, key: str) -> dict[str, Any]:
    """What the page shows of a `ClassificationRun`: no map, only what to ask for."""

    from cryoet_pipeline.reports.classification import class_changes

    decoys = np.asarray(sorted(run.decoys), dtype=np.int64)
    iterations: list[dict[str, Any]] = [{
        "number": 0,
        "label": "starting references",
        "searched": [],
        "summary": None,
        "classes": [{"channel": channel, "members": None, "decoys": None, "median_cc": None,
                     "note": run.reference_notes.get(channel)}
                    for channel in sorted(run.references)],
    }]
    changes = class_changes(run.iterations)
    for iteration, changed in zip(run.iterations, changes, strict=True):
        classes = []
        for channel, result in sorted(iteration.classes.items()):
            members = np.asarray(result.members)
            classes.append({
                "channel": channel,
                "members": int(len(members)),
                "decoys": int(np.isin(members, decoys).sum()) if len(decoys) else 0,
                "median_cc": _finite(result.median_cc),
                "map": result.average is not None,
            })
        iterations.append({
            "number": iteration.number,
            "label": f"iteration {iteration.number}",
            "searched": list(iteration.searched),
            "summary": iteration.summary,
            "started": iteration.started,
            "finished": iteration.finished,
            "changed": changed,
            "classes": classes,
        })
    channels = sorted({*run.references,
                       *(channel for one in run.iterations for channel in one.classes)})
    sizes = {str(channel): [len(one.classes[channel].members) if channel in one.classes else None
                            for one in run.iterations]
             for channel in channels}
    sign, sign_source = _run_sign(run)
    return {
        "name": run.name,
        "key": key,
        "software": run.software,
        "box_px": run.box_px,
        "pixel_angstrom": run.pixel_size_angstrom,
        "particles": int(len(run.tags)),
        "decoys": int(len(decoys)),
        "decoy_kind": run.decoy_kind if len(decoys) else None,
        "planned": run.planned_iterations,
        "finished": len(run.iterations),
        "channels": channels,
        "sizes": sizes,
        "iterations": iterations,
        "mask": run.mask is not None,
        "protein_sign": sign,
        "protein_sign_source": sign_source,
        "parameters": [{"group": one.group, "name": one.name, "value": one.value}
                       for one in run.parameters],
        "problems": list(run.problems),
    }


def _run_sign(run: Any) -> tuple[int, str]:
    """The protein's side for every map of a run: measured once, on the last iteration's
    largest class against the alignment mask, so that every class is drawn on one side."""

    if run.mask is not None and run.iterations:
        last = run.iterations[-1]
        largest = max(last.classes.values(), key=lambda one: len(one.members), default=None)
        if largest is not None and largest.average is not None:
            sign = sign_inside(largest.average.load(), run.mask.load())
            if sign is not None:
                return sign, (f"measured on iteration {last.number}'s class {largest.channel}, "
                              "inside the alignment mask against outside it")
    return -1, "assumed: protein dark, as in the tomograms"


def fsc_document(raw: dict[str, Any], name: str, key: str) -> dict[str, Any]:
    """The curves of scripts/consensus_halves.py's <out>_fsc.json, in spatial frequency.

    The page draws against 1 / resolution, so that the axis is linear in frequency as
    the FSC is computed. A value that is not finite, which a browser's JSON cannot
    carry, becomes null, and a resolution that is not finite the zero frequency.
    """

    spacing = _finite(raw.get("pixel_spacing_angstrom"))
    curves = {}
    for label in ("masked", "unmasked"):
        curve = raw.get(label)
        if not isinstance(curve, dict):
            continue
        resolutions = [_finite(one) for one in curve.get("resolution_angstrom", [])]
        rise = curve.get("rise_toward_nyquist")
        curves[label] = {
            "frequency": [1.0 / one if one else 0.0 for one in resolutions],
            "correlation": [_finite(one) for one in curve.get("correlation", [])],
            "falls": {level: [value for value in (_finite(one) for one in
                                                  curve.get(f"falls_{level}_angstrom") or [])
                              if value is not None]
                      for level in ("0.5", "0.143")},
            "rise": {one: _finite(value) for one, value in rise.items()}
            if isinstance(rise, dict) else None,
        }
    return {
        "name": name,
        "key": key,
        "halves": [Path(str(one)).name for one in raw.get("halves", [])],
        "iteration": raw.get("iteration"),
        "start_poses": raw.get("start_poses"),
        "pixel_angstrom": spacing,
        "nyquist_frequency": 1.0 / (2.0 * spacing) if spacing else None,
        "curves": curves,
    }
