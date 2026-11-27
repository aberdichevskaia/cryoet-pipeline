"""The result pages' data and server (scripts/serve_results.py).

The circles are checked against the picture itself: in the synthetic project of
test_report, the first 24 picks are dark blobs written into the volume, so each
of their circles must sit on a dark spot of the PNG the page shows. That is a
truth the placement does not come from: a flip, a missing half voxel or a wrong
row count after block-averaging moves the circles off the blobs.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import threading
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_report import _project

from cryoet_pipeline.dashboard import data as dashboard_data
from cryoet_pipeline.dashboard.data import (
    MAX_PAGE_PX,
    ResultsProject,
    band_rows,
    placed,
    protein_sign,
    slab_centres,
)
from cryoet_pipeline.dashboard.server import VIEWER_FILES, ResultsServer, tunnel_lines

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_viewer_assets  # noqa: E402
import serve_results  # noqa: E402

# test_report gives its blobs scores of 0.09 to 0.16 and its decoys 0.05 to 0.065.
BLOB_SCORE = 0.08


def _results(root: Path, max_px: int = MAX_PAGE_PX) -> ResultsProject:
    return ResultsProject(_project(root), cache_dir=root / "cache", max_px=max_px)


def _picture(results: ResultsProject, document: dict, name: str) -> np.ndarray:
    body = results.picture(document["series"], document["key"], name)
    return np.asarray(Image.open(io.BytesIO(body)).convert("L"), dtype=np.float64)


def _offsets(results: ResultsProject, stack: str) -> np.ndarray:
    """For every blob cut through its centre: the dark spot's centroid minus the circle's centre."""

    document = results.series("TS_A")
    picks = document["picks"]
    full = picks["radius_px"] / document["tomogram"]["stacks"][stack]["factor"]
    offsets = []
    for number, circles in enumerate(document["overlay"][stack]):
        image = _picture(results, document, f"{stack}_{number:03d}.png")
        background = float(np.median(image))
        for index, u, v, radius in circles:
            if picks["score"][index] < BLOB_SCORE or radius < full - 1e-6:
                continue
            # Pixel j covers j .. j + 1, so a window of pixels about (u, v) and the
            # darkness-weighted mean of their centres.
            column, row = int(np.floor(u)), int(np.floor(v))
            lo_r, hi_r = max(0, row - 2), min(image.shape[0], row + 3)
            lo_c, hi_c = max(0, column - 2), min(image.shape[1], column + 3)
            window = np.clip(background - image[lo_r:hi_r, lo_c:hi_c], 0.0, None)
            if window.sum() == 0 or hi_r - lo_r < 5 or hi_c - lo_c < 5:
                continue
            rows, columns = np.mgrid[lo_r:hi_r, lo_c:hi_c] + 0.5
            offsets.append([
                float((window * columns).sum() / window.sum()) - u,
                float((window * rows).sum() / window.sum()) - v,
            ])
    return np.asarray(offsets)


@pytest.mark.parametrize("stack", ["xy", "xz"])
@pytest.mark.parametrize("max_px", [MAX_PAGE_PX, 50])
def test_every_circle_sits_on_its_particle_in_the_picture(
    tmp_path: Path, stack: str, max_px: int
) -> None:
    results = _results(tmp_path, max_px)
    document = results.series("TS_A")
    factor = document["tomogram"]["stacks"][stack]["factor"]
    assert factor == (1 if max_px == MAX_PAGE_PX else 3)
    offsets = _offsets(results, stack)
    assert len(offsets) >= 15
    # Each blob within a pixel, and no shift common to all of them: a missing
    # half voxel would move the mean by 0.5 / factor.
    assert np.abs(offsets).max() < 1.0
    assert np.abs(offsets.mean(axis=0)).max() < 0.15 / factor + 0.05


def test_slabs_tile_the_axis_without_gaps_or_overlap() -> None:
    for size in (1, 9, 40, 41, 300):
        for half in (0, 1, 4):
            covered = []
            for centre in slab_centres(size, half):
                covered.extend(range(max(0, centre - half), min(size, centre + half + 1)))
            assert covered == list(range(size))


def test_a_voxel_centre_and_a_slab_land_where_the_picture_draws_them() -> None:
    # Voxel (0, 0) of a picture 10 rows high is its bottom left pixel, row 9.
    assert placed(np.array([0.0]), np.array([0.0]), np.array([2.0]), factor=1, rows=10) == [
        [0, 0.5, 9.5, 2.0]
    ]
    # Reduced by 2, voxel 3 lies in pixel 1 (voxels 2 and 3), at its right edge.
    assert placed(np.array([3.0]), np.array([3.0]), np.array([4.0]), factor=2, rows=5) == [
        [0, 1.75, 3.25, 2.0]
    ]
    assert placed(np.array([1.0]), np.array([1.0]), np.array([np.nan]), factor=1, rows=4) == []
    # The slab of voxels 3, 4 and 5 covers rows 4 to 7 of a picture 10 rows high.
    assert band_rows(4, 1, factor=1, rows=10) == [4.0, 5.5, 7.0]


def test_each_pick_is_shown_in_the_slabs_that_hold_its_centre(tmp_path: Path) -> None:
    results = _results(tmp_path)
    document = results.series("TS_A")
    picks = document["picks"]
    stacks = document["tomogram"]["stacks"]
    half = document["tomogram"]["slab_half_px"]
    for index in range(picks["count"]):
        number = picks["xy_slab"][index]
        assert abs(stacks["xy"]["centres"][number] - picks["z"][index]) <= half
        assert any(one[0] == index for one in document["overlay"]["xy"][number])
        number = picks["xz_slab"][index]
        assert abs(stacks["xz"]["centres"][number] - picks["y"][index]) <= half
        assert any(one[0] == index for one in document["overlay"]["xz"][number])
    ranks = sorted(picks["rank"])
    assert ranks == list(range(1, picks["count"] + 1))
    assert picks["score"][picks["order"][0]] == max(picks["score"])


def test_the_overview_says_what_each_series_has(tmp_path: Path) -> None:
    overview = _results(tmp_path).overview()
    flags = {one["id"]: (one["tomogram"], one["picks"]) for one in overview["series"]}
    assert flags == {"TS_A": (True, True), "TS_B": (True, False), "TS_C": (False, False)}


def test_a_series_without_a_tomogram_has_an_empty_page(tmp_path: Path) -> None:
    document = _results(tmp_path).series("TS_C")
    assert document["tomogram"] is None and document["picks"] is None


def test_a_series_is_cut_once_and_again_when_its_tomogram_is_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = _results(tmp_path)
    first = results.series("TS_B")
    calls = []
    original = dashboard_data.series_document

    def counted(*args: object, **kwargs: object) -> dict:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(dashboard_data, "series_document", counted)
    again = ResultsProject(results.project, cache_dir=results.cache_dir)
    assert again.series("TS_B") == first
    assert calls == []

    tomogram = Path(first["tomogram"]["path"])
    later = os.stat(tomogram).st_mtime + 60
    for path in (tomogram, *tomogram.iterdir()):
        os.utime(path, (later, later))
    remade = again.series("TS_B")
    assert calls == [1]
    assert remade["key"] != first["key"]
    with pytest.raises(KeyError, match="cut again"):
        again.picture("TS_B", first["key"], "xy_000.png")


def test_a_pick_and_the_template_are_shown_as_tiles(tmp_path: Path) -> None:
    results = _results(tmp_path)
    document = results.series("TS_A")
    tile = _picture(results, document, "pick_0.png")
    template = _picture(results, document, "template.png")
    assert tile.shape[0] > 4 and tile.shape[1] > 2 * tile.shape[0]
    assert template.shape == tile.shape
    with pytest.raises(KeyError):
        results.picture("TS_A", document["key"], "../artifacts.json")


def test_the_protein_side_is_measured_against_the_mask_before_any_record(tmp_path: Path) -> None:
    results = _results(tmp_path)
    document = results.series("TS_A")
    # test_report's template is a negative Gaussian, recorded as inverted.
    assert document["template"]["protein_sign"] == -1
    assert document["template"]["protein_sign_source"].startswith("measured")
    assert document["template"]["maps"] == ["mask.mrc", "template.mrc"]

    report = results._report("TS_A", results._prepared("TS_A"))
    template = report.template
    assert template is not None
    # The record says inverted, the map says otherwise: the map wins.
    assert protein_sign(replace(template, template=-template.template))[0] == 1
    assert protein_sign(replace(template, mask=None)) == (
        -1, "prepare-template's record of inverting it")
    assert protein_sign(replace(template, mask=None, inverted=None))[0] == 1


def test_only_the_template_and_its_mask_are_served_as_maps(tmp_path: Path) -> None:
    results = _results(tmp_path)
    document = results.series("TS_A")
    key = document["key"]
    path = results.map_file("TS_A", key, "template.mrc")
    assert path.name == "template.mrc" and path.is_file()
    assert results.map_file("TS_A", key, "mask.mrc").name == "mask.mrc"
    for wrong in ("tomogram.zarr", "../artifacts.json", "template.MRC"):
        with pytest.raises(KeyError):
            results.map_file("TS_A", key, wrong)
    with pytest.raises(KeyError, match="cut again"):
        results.map_file("TS_A", "0" * 16, "template.mrc")
    with pytest.raises(KeyError, match="no template"):
        results.map_file("TS_B", results.series("TS_B")["key"], "template.mrc")


def test_a_viewer_file_is_taken_only_with_its_sha256(tmp_path: Path) -> None:
    body = b"var molstar = {};\n"
    entry = {"path": "molstar/molstar.js", "url": "unused",
             "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
    source = tmp_path / "copy"
    (source / "molstar").mkdir(parents=True)
    (source / "molstar" / "molstar.js").write_bytes(body)
    vendor = tmp_path / "vendor"
    assert "sha256 matches" in fetch_viewer_assets.fetch(entry, vendor, source)
    assert (vendor / "molstar" / "molstar.js").read_bytes() == body
    assert "already here" in fetch_viewer_assets.fetch(entry, vendor, source)

    other = tmp_path / "other"
    (other / "molstar").mkdir(parents=True)
    (other / "molstar" / "molstar.js").write_bytes(b"something else")
    with pytest.raises(ValueError, match="sha256"):
        fetch_viewer_assets.fetch(entry, tmp_path / "fresh", other)
    assert list((tmp_path / "fresh" / "molstar").iterdir()) == []


def test_the_viewer_config_names_exactly_the_files_the_server_serves() -> None:
    entries = tomllib.loads((ROOT / "configs" / "viewer_assets.toml").read_text())["file"]
    assert sorted(one["path"] for one in entries) == sorted(VIEWER_FILES)
    for one in entries:
        assert len(one["sha256"]) == 64 and one["url"].startswith("https://")
        assert one["url"].endswith(one["path"].split("/")[-1])


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, str]]:
    assets = tmp_path / "assets"
    for name in VIEWER_FILES:
        (assets / name).parent.mkdir(parents=True, exist_ok=True)
        (assets / name).write_text(f"/* {name} */")
    server = ResultsServer(("127.0.0.1", 0), _results(tmp_path), "secret-token", assets=assets)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.port}", server.cookie_name
    finally:
        server.shutdown()
        server.server_close()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def _get(url: str, cookie: str | None = None) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, headers={"Cookie": cookie} if cookie else {})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(request) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_the_server_answers_only_with_the_token(served: tuple[str, str]) -> None:
    base, cookie_name = served
    assert _get(f"{base}/")[0] == 403
    assert _get(f"{base}/api/project")[0] == 403
    assert _get(f"{base}/?token=wrong")[0] == 403
    assert _get(f"{base}/", f"{cookie_name}=wrong")[0] == 403

    status, headers, _ = _get(f"{base}/?token=secret-token")
    assert status == 303 and headers["Location"] == "/"
    assert headers["Set-Cookie"].startswith(f"{cookie_name}=secret-token;")
    assert "HttpOnly" in headers["Set-Cookie"]

    cookie = f"{cookie_name}=secret-token"
    status, headers, body = _get(f"{base}/", cookie)
    assert status == 200 and b"<canvas" in body
    status, _, body = _get(f"{base}/api/project", cookie)
    assert status == 200 and json.loads(body)["series"][0]["id"] == "TS_A"
    status, _, body = _get(f"{base}/api/series/TS_A", cookie)
    document = json.loads(body)
    status, headers, body = _get(f"{base}/img/TS_A/{document['key']}/xy_000.png", cookie)
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert body.startswith(b"\x89PNG")
    assert _get(f"{base}/api/series/TS_X", cookie)[0] == 404
    assert _get(f"{base}/static/data.py", cookie)[0] == 404
    assert _get(f"{base}/static/%2e%2e/data.py", cookie)[0] == 404

    status, _, body = _get(f"{base}/api/project", cookie)
    assert json.loads(body)["viewer3d"] is True
    status, headers, body = _get(f"{base}/vendor/molstar/molstar.js", cookie)
    assert status == 200 and body == b"/* molstar/molstar.js */"
    assert headers["Content-Type"].startswith("text/javascript")
    assert _get(f"{base}/vendor/molstar/molstar.js")[0] == 403
    assert _get(f"{base}/vendor/molstar/other.js", cookie)[0] == 404
    status, _, body = _get(f"{base}/map/TS_A/{document['key']}/template.mrc", cookie)
    assert status == 200 and body[208:212] == b"MAP "
    assert _get(f"{base}/map/TS_A/{document['key']}/artifacts.json", cookie)[0] == 404


def test_a_lane_names_its_project_and_the_label_of_its_picks() -> None:
    folder, label = serve_results.lane_project(ROOT / "configs" / "lanes" / "ryr1-bin8.toml")
    assert folder == ROOT / "outputs" / "ryr1" and label == "bin8"


def test_a_lane_given_as_a_project_is_named_with_its_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On max, 2026-11-26: --project outputs/ryr1-bin8, the lane's name, not its project."""

    outputs = tmp_path / "outputs"
    for name in ("ryr1", "ribo-mdc"):
        (outputs / name).mkdir(parents=True)
        (outputs / name / "artifacts.json").write_text("{}")
    (outputs / "eval-runs").mkdir()
    with pytest.raises(SystemExit):
        serve_results.main(["--project", str(outputs / "ryr1-bin8")])
    error = capsys.readouterr().err
    assert "no registry at" in error
    assert ("ryr1-bin8 is a lane, whose project is outputs/ryr1 and whose picks are labelled "
            "bin8: give --lane configs/lanes/ryr1-bin8.toml") in error
    assert f"the folders in {outputs} that hold a registry: ribo-mdc, ryr1" in error
    assert serve_results.registry_hint(outputs / "nothing-like-it") == (
        f"\n  the folders in {outputs} that hold a registry: ribo-mdc, ryr1")


def test_the_tunnel_ends_on_the_node_when_only_the_node_may_connect() -> None:
    assert tunnel_lines("127.0.0.1", 8765, "node7", "login1") == [
        "ssh -N -L 8765:localhost:8765 -J login1 node7"
    ]
    assert tunnel_lines("0.0.0.0", 8765, "node7", None) == [
        "ssh -N -L 8765:node7:8765 <login node>"
    ]
