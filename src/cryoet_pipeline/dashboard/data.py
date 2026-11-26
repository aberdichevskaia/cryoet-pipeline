"""What the result pages show, as data the page draws itself from.

A series becomes one JSON document and a set of PNG pictures: the tomogram as a
stack of slabs through z and another through y, the picks as circles placed on
each picture, and, on request, a pick's three views turned into the template's
frame. The pictures are made by the PDF report's own functions
(`reports.imaging`, `reports.sections`), so a slab on a page is drawn as the PDF
draws its slabs: the mean of about `SLAB_ANGSTROM` of depth, smoothed by one
voxel. The slabs tile the axis, each as thick as the step between them, so every
particle lies whole in at least one.

The series' tomogram, picks and template are found as `cryoet report` finds them
(`cli._report_artifacts`), so a page and the PDF show the same files.

Nothing here writes to the project but the cache, and the registry is read,
never locked. A series is cut again when its tomogram or its picks are newer
than what its cache was cut from; the cache's folder is named by a hash of
those, so a page never mixes pictures from two cuts.

Coordinates. The tomogram is (Z, Y, X) and a pick's x, y, z are voxel indices
into it, as for extraction. A picture is stored with y (or z) increasing
upwards, as the PDF and 3dmod draw it, so its first PNG row is the last row of
the volume it shows. Every circle is placed here, in the picture's own pixels
(origin at the top left corner of the PNG, pixel j covering j to j + 1), not in
the browser: the page only scales, so the one place a flip or a half voxel could
go wrong is the place the tests look at.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.models import Artifact, TiltSeriesManifest
from cryoet_pipeline.reports.collect import SeriesArtifacts, SeriesReport, load_series
from cryoet_pipeline.reports.imaging import (
    SMOOTHING_SIGMA_PX,
    block_mean,
    display_range,
    reduction_factor,
    slab_half_width_px,
    slab_mean,
    smooth,
    sphere_sections,
)

CACHE_FORMAT = 1
"""Raised whenever what a cache holds changes, so that old caches are cut again."""

MAX_PAGE_PX = 1024
"""Pictures larger than this are block-averaged down; a bin8 tomogram is not."""

PICTURE_NAME = re.compile(r"^(?:(?:xy|xz)_\d{3}|pick_\d+|template)\.png$")


def slab_centres(size: int, half: int) -> list[int]:
    """Centres of slabs 2 h + 1 thick that tile an axis from its first voxel.

    The slab around centre c spans c - h .. c + h; the last one may reach past
    the volume, where `slab_mean` clips it.
    """

    step = 2 * half + 1
    return list(range(half, size + half, step))


def slab_picture(
    volume: Any, axis: int, centre: int, half: int, max_px: int = MAX_PAGE_PX
) -> tuple[np.ndarray, int]:
    """A smoothed slab, block-averaged to at most `max_px`, and the factor it was reduced by."""

    image = smooth(slab_mean(volume, axis, centre, half))
    factor = reduction_factor(image.shape, max_px)
    return block_mean(image, factor), factor


def placed(
    across_px: np.ndarray,
    up_px: np.ndarray,
    radii_px: np.ndarray,
    *,
    factor: int,
    rows: int,
) -> list[list[float]]:
    """Circles in a picture's pixels: [pick index, column, row, radius] for each finite radius.

    `across_px` and `up_px` are voxel indices along the picture's horizontal
    axis and along the axis drawn upwards. Voxel i spans i .. i + 1, so its
    centre is at i + 0.5; reduced by `factor`, that is (i + 0.5) / factor
    across, and `rows` minus that down from the top, since row 0 of the
    stored picture is its highest.
    """

    circles = []
    for index in np.flatnonzero(np.isfinite(radii_px)):
        circles.append(
            [
                int(index),
                round((float(across_px[index]) + 0.5) / factor, 3),
                round(rows - (float(up_px[index]) + 0.5) / factor, 3),
                round(float(radii_px[index]) / factor, 3),
            ]
        )
    return circles


def band_rows(centre: int, half: int, *, factor: int, rows: int) -> list[float]:
    """Where a slab of another stack lies on a picture: its top edge, centre and bottom edge.

    The slab spans voxels centre - h .. centre + h along the picture's upward
    axis, that is from centre - h to centre + h + 1, drawn upwards, so its top
    edge is the higher number.
    """

    return [
        round(rows - (centre + half + 1) / factor, 3),
        round(rows - (centre + 0.5) / factor, 3),
        round(rows - (centre - half) / factor, 3),
    ]


def png_bytes(image: np.ndarray, limits: tuple[float, float]) -> bytes:
    """A grey PNG on the given scale, low values dark, drawn upwards; NaN is transparent."""

    low, high = limits
    span = high - low if high > low else 1.0
    finite = np.isfinite(image)
    grey = np.clip((np.where(finite, image, low) - low) / span, 0.0, 1.0)
    pixels = np.round(grey * 255.0).astype(np.uint8)[::-1]
    if finite.all():
        picture = Image.fromarray(np.ascontiguousarray(pixels), mode="L")
    else:
        alpha = np.where(finite, 255, 0).astype(np.uint8)[::-1]
        picture = Image.fromarray(np.ascontiguousarray(np.stack([pixels, alpha], axis=-1)),
                                  mode="LA")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _modified(path: Path) -> float:
    """When a file, or a zarr folder's metadata, was last written."""

    if not path.exists():
        return 0.0
    if not path.is_dir():
        return path.stat().st_mtime
    candidates = [path, path / ".zarray", path / "zarr.json", path / ".zattrs"]
    return max(one.stat().st_mtime for one in candidates if one.exists())


def cache_key(artifacts: SeriesArtifacts, max_px: int) -> str:
    """A hash of what a series' pictures are cut from."""

    def described(artifact: Artifact | None) -> list[Any] | None:
        if artifact is None:
            return None
        return [artifact.id, str(artifact.path), _modified(Path(artifact.path))]

    source = [
        CACHE_FORMAT,
        max_px,
        described(artifacts.tomogram),
        described(artifacts.picks),
        described(artifacts.template),
        described(artifacts.template_mask),
    ]
    return hashlib.sha1(json.dumps(source).encode()).hexdigest()[:16]


def _nearest(centres: list[int], values: np.ndarray) -> list[int]:
    """The slab whose centre is nearest each value; slabs tile the axis, so it holds it."""

    distance = np.abs(np.asarray(values, dtype=np.float64)[:, None] - np.asarray(centres)[None])
    return [int(one) for one in np.argmin(distance, axis=1)]


def _rounded(values: np.ndarray, digits: int) -> list[float | None]:
    return [round(float(one), digits) if np.isfinite(one) else None for one in values]


def _caption(thickness_px: int, voxel_angstrom: float) -> str:
    return (
        f"Protein is dark. Each picture is the mean of a "
        f"{thickness_px * voxel_angstrom / 10.0:.0f} nm slab ({thickness_px} voxels), smoothed "
        f"with a {SMOOTHING_SIGMA_PX:g}-voxel Gaussian, every slab of the tomogram on one grey "
        "scale (its 1st to 99th percentile). The slabs follow one another without gaps, so "
        "every particle lies whole in at least one. z is the height above the bottom of the "
        "tomogram; x, y and z increase to the right and upwards, as in 3dmod. A circle is the "
        "particle's sphere cut by the slab, as 3dmod draws spheres: full size where the slab "
        "passes through its centre, smaller just above or below it."
    )


def series_document(report: SeriesReport, folder: Path, *, max_px: int = MAX_PAGE_PX
                    ) -> dict[str, Any]:
    """Cut one series' slabs into `folder` and return the document its page is drawn from."""

    from cryoet_pipeline.reports.sections.tomogram import open_volume

    document: dict[str, Any] = {
        "series": report.series,
        "problems": list(report.problems),
        "tomogram": None,
        "picks": None,
        "template": None,
        "overlay": {},
    }
    info = report.tomogram
    if info is None:
        return document
    volume = open_volume(info.artifact.path)
    voxel = info.voxel_spacing_angstrom
    half = slab_half_width_px(voxel)
    depth, height, _ = info.shape

    stacks: dict[str, list[tuple[np.ndarray, int]]] = {}
    centres: dict[str, list[int]] = {}
    for name, axis, size in (("xy", 0, depth), ("xz", 1, height)):
        centres[name] = slab_centres(size, half)
        stacks[name] = [slab_picture(volume, axis, one, half, max_px) for one in centres[name]]
    limits = display_range([image for pictures in stacks.values() for image, _ in pictures])
    for name, pictures in stacks.items():
        for number, (image, _) in enumerate(pictures):
            (folder / f"{name}_{number:03d}.png").write_bytes(png_bytes(image, limits))

    thickness = 2 * half + 1
    # Each picture marks where the other stack's slabs cut it: an XY picture
    # the XZ slabs (y), an XZ picture the XY slabs (z).
    other = {"xy": "xz", "xz": "xy"}
    bands = {
        name: [
            band_rows(centre, half, factor=pictures[0][1], rows=pictures[0][0].shape[0])
            for centre in centres[other[name]]
        ]
        for name, pictures in stacks.items()
    }
    document["tomogram"] = {
        "id": info.artifact.id,
        "path": str(info.artifact.path),
        "shape_zyx": list(info.shape),
        "voxel_angstrom": voxel,
        "slab_half_px": half,
        "slab_thickness_px": thickness,
        "limits": list(limits),
        "caption": _caption(thickness, voxel),
        "stacks": {
            name: {
                "axis": "z" if name == "xy" else "y",
                "centres": centres[name],
                "count": len(pictures),
                "width": int(pictures[0][0].shape[1]),
                "height": int(pictures[0][0].shape[0]),
                "factor": int(pictures[0][1]),
                "bands": bands[name],
            }
            for name, pictures in stacks.items()
        },
    }

    picks = report.picks
    if picks is not None and picks.count > 0:
        radius, radius_source = report.particle_radius()
        ranked = picks.ranked()
        rank = np.empty(picks.count, dtype=int)
        rank[ranked] = np.arange(1, picks.count + 1)
        sigma = picks.sigma
        document["picks"] = {
            "count": picks.count,
            "x": _rounded(picks.x_px, 2),
            "y": _rounded(picks.y_px, 2),
            "z": _rounded(picks.z_px, 2),
            "score": _rounded(picks.score, 6),
            "sigma": _rounded(sigma, 3) if sigma is not None else None,
            "rank": [int(one) for one in rank],
            "order": [int(one) for one in ranked],
            # The slab each pick's centre lies in, through z and through y.
            "xy_slab": _nearest(centres["xy"], picks.z_px),
            "xz_slab": _nearest(centres["xz"], picks.y_px),
            "radius_px": round(float(radius), 3),
            "radius_source": radius_source,
            "search_sigma": picks.search_sigma,
            "floor_sigma": picks.floor_sigma,
            "cut_off_sigma": picks.cut_off_sigma,
            "low_pass_angstrom": picks.low_pass_angstrom,
            "oriented": picks.has_orientations,
            "id": report.artifacts.picks.id if report.artifacts.picks is not None else None,
        }
        overlay: dict[str, list[list[list[float]]]] = {"xy": [], "xz": []}
        for centre, (image, factor) in zip(centres["xy"], stacks["xy"], strict=True):
            sections = sphere_sections(picks.z_px, centre, half, radius)
            overlay["xy"].append(placed(picks.x_px, picks.y_px, sections, factor=factor,
                                        rows=image.shape[0]))
        for centre, (image, factor) in zip(centres["xz"], stacks["xz"], strict=True):
            sections = sphere_sections(picks.y_px, centre, half, radius)
            overlay["xz"].append(placed(picks.x_px, picks.z_px, sections, factor=factor,
                                        rows=image.shape[0]))
        document["overlay"] = overlay

    template = report.template
    if template is not None:
        document["template"] = {
            "box_px": template.box_px,
            "voxel_angstrom": template.voxel_spacing_angstrom,
            "source": template.source,
            "low_pass_angstrom": template.low_pass_angstrom,
            "mask_radius_px": template.mask_radius_px,
        }
    return document


def pick_tile(report: SeriesReport, index: int) -> np.ndarray:
    """One pick's XY, XZ and YZ views, as a tile of the PDF's gallery shows them."""

    from cryoet_pipeline.particles import pick_center_offset_px
    from cryoet_pipeline.reports.sections.gallery import (
        _tile_image,
        gallery_half_width,
        oriented_views,
    )
    from cryoet_pipeline.reports.sections.tomogram import open_volume
    from cryoet_pipeline.rotations import relion_euler_to_matrix

    picks = report.picks
    info = report.tomogram
    if picks is None or info is None:
        raise ValueError("this series has no picks on a tomogram")
    if not 0 <= index < picks.count:
        raise ValueError(f"no pick {index}: the list holds {picks.count}")
    half = gallery_half_width(report)
    voxel = info.voxel_spacing_angstrom
    matrix = None
    if picks.has_orientations:
        assert picks.rot_deg is not None and picks.tilt_deg is not None
        assert picks.psi_deg is not None
        matrix = relion_euler_to_matrix(
            float(picks.rot_deg[index]), float(picks.tilt_deg[index]), float(picks.psi_deg[index])
        )
    offset = (
        pick_center_offset_px(picks.template_box_px) if picks.template_box_px is not None else 0.0
    )
    views = oriented_views(
        open_volume(info.artifact.path),
        (float(picks.z_px[index]), float(picks.y_px[index]), float(picks.x_px[index])),
        matrix,
        offset_px=offset,
        half=half,
        low_pass_angstrom=picks.low_pass_angstrom,
        voxel_spacing_angstrom=voxel,
        slab_px=min(2 * slab_half_width_px(voxel) + 1, 2 * half + 1),
    )
    return _tile_image(views)


def template_tile(report: SeriesReport) -> np.ndarray:
    """The template's central views, on the grid and with the filter of a pick's tile."""

    from cryoet_pipeline.reports.sections.gallery import (
        _tile_image,
        gallery_half_width,
        template_views,
    )

    if report.template is None or report.tomogram is None:
        raise ValueError("this series has no template")
    half = gallery_half_width(report)
    voxel = report.tomogram.voxel_spacing_angstrom
    low_pass = report.picks.low_pass_angstrom if report.picks is not None else None
    return _tile_image(template_views(
        report.template, half=half, low_pass_angstrom=low_pass,
        slab_px=min(2 * slab_half_width_px(voxel) + 1, 2 * half + 1),
    ))


@dataclass
class _Series:
    key: str
    folder: Path
    document: dict[str, Any]
    report: SeriesReport | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class ResultsProject:
    """One project's result pages, read from its registry on every request.

    Reading the registry again each time costs a JSON parse and lets a page see
    series and results that appeared since it was opened. A series' pictures are
    cut on its first request and kept under `cache_dir`.
    """

    def __init__(self, project: Path, *, cache_dir: Path, variant: str | None = None,
                 max_px: int = MAX_PAGE_PX) -> None:
        self.project = project
        self.cache_dir = cache_dir
        self.variant = variant
        self.max_px = max_px
        self._series: dict[str, _Series] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _registry(self) -> ArtifactRegistry:
        path = self.project / "artifacts.json"
        if not path.is_file():
            raise FileNotFoundError(f"no registry at {path}")
        return ArtifactRegistry.load(path)

    def _manifests(self) -> dict[str, TiltSeriesManifest]:
        manifests = {}
        for path in sorted((self.project / "manifests").glob("*.json")):
            manifest = TiltSeriesManifest.model_validate_json(path.read_text())
            manifests[manifest.tilt_series_id] = manifest
        return manifests

    def _resolve(self, registry: ArtifactRegistry, manifest: TiltSeriesManifest
                 ) -> SeriesArtifacts:
        from cryoet_pipeline.cli import _report_artifacts

        return _report_artifacts(registry, manifest, variant=self.variant)

    def overview(self) -> dict[str, Any]:
        registry = self._registry()
        series = []
        for name, manifest in self._manifests().items():
            artifacts = self._resolve(registry, manifest)
            series.append({
                "id": name,
                "tomogram": artifacts.tomogram is not None,
                "picks": artifacts.picks is not None,
                "problems": list(artifacts.problems),
            })
        return {
            "project": self.project.resolve().name,
            "path": str(self.project.resolve()),
            "variant": self.variant,
            "series": series,
        }

    def _lock_for(self, name: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(name, threading.Lock())

    def _prepared(self, name: str) -> _Series:
        manifest = self._manifests().get(name)
        if manifest is None:
            raise KeyError(f"no series {name} in {self.project / 'manifests'}")
        registry = self._registry()
        artifacts = self._resolve(registry, manifest)
        key = cache_key(artifacts, self.max_px)
        with self._lock_for(name):
            known = self._series.get(name)
            if known is not None and known.key == key:
                return known
            folder = self.cache_dir / name / key
            stored = folder / "series.json"
            report = None
            if stored.is_file():
                document = json.loads(stored.read_text())
            else:
                report = load_series(registry, artifacts)
                partial = folder.with_name(f"{key}.partial")
                partial.mkdir(parents=True, exist_ok=True)
                document = series_document(report, partial, max_px=self.max_px)
                document["key"] = key
                (partial / "series.json").write_text(json.dumps(document))
                partial.replace(folder)
            prepared = _Series(key, folder, document, report)
            self._series[name] = prepared
            return prepared

    def series(self, name: str) -> dict[str, Any]:
        return self._prepared(name).document

    def picture(self, name: str, key: str, file: str) -> bytes:
        if not PICTURE_NAME.match(file):
            raise KeyError(f"no picture {file}")
        prepared = self._prepared(name)
        if prepared.key != key:
            raise KeyError(f"{name} has been cut again since this page was loaded; reload it")
        path = prepared.folder / file
        if path.is_file():
            return path.read_bytes()
        if file.startswith("pick_") or file == "template.png":
            with prepared.lock:
                if not path.is_file():
                    report = self._report(name, prepared)
                    if file == "template.png":
                        image = template_tile(report)
                    else:
                        image = pick_tile(report, int(file[5:-4]))
                    path.write_bytes(png_bytes(image, display_range([image])))
            return path.read_bytes()
        raise KeyError(f"no picture {file}")

    def _report(self, name: str, prepared: _Series) -> SeriesReport:
        if prepared.report is None:
            registry = self._registry()
            manifest = self._manifests()[name]
            prepared.report = load_series(registry, self._resolve(registry, manifest))
        return prepared.report
