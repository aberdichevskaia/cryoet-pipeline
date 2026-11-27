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
import re
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
from cryoet_pipeline.dashboard.mra import ClassificationRuns
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


def test_the_protein_side_is_the_record_of_inverting_before_the_mask(tmp_path: Path) -> None:
    results = _results(tmp_path)
    document = results.series("TS_A")
    # test_report's template is a negative Gaussian, recorded as inverted.
    assert document["template"]["protein_sign"] == -1
    assert document["template"]["protein_sign_source"] == (
        "prepare-template's record of inverting it")
    assert document["template"]["maps"] == ["mask.mrc", "template.mrc"]

    report = results._report("TS_A", results._prepared("TS_A"))
    template = report.template
    assert template is not None
    # Munich, 27 November 2026: the Gag template, a box filled with lattice, measured
    # the other side from its record, and the page drew it reversed. The record wins,
    # and the disagreement is said.
    sign, source = protein_sign(replace(template, template=-template.template))
    assert sign == -1
    assert source.startswith("prepare-template's record of inverting it; ")
    assert "says the other side" in source
    assert protein_sign(replace(template, inverted=False)) == (
        1, "prepare-template's record of inverting it; its mean inside the mask against "
        "outside it says the other side, as it can where density fills the box, as a "
        "lattice's does")
    assert protein_sign(replace(template, mask=None)) == (
        -1, "prepare-template's record of inverting it")
    # Without a record, the mask decides; without either, protein is taken as positive.
    assert protein_sign(replace(template, inverted=None)) == (
        -1, "measured: the template's mean inside its mask against outside it")
    assert protein_sign(replace(template, template=-template.template, inverted=None))[0] == 1
    assert protein_sign(replace(template, mask=None, inverted=None)) == (
        1, "assumed: neither a mask nor a record says")


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
    assert tunnel_lines("127.0.0.1", 8765, "node7", "login1", "someone") == [
        "ssh -N -L 8765:localhost:8765 -J login1 someone@node7"
    ]
    assert tunnel_lines("0.0.0.0", 8765, "node7", None, "someone") == [
        "ssh -N -L 8765:node7:8765 <login node>"
    ]


def test_the_printed_tunnel_names_the_cluster_user_and_the_laptops_way_to_the_login() -> None:
    # max, 27 November 2026: from a laptop outside MDC, max-login2 is reached only
    # through MDC's ssh gateway, by a Host of the laptop's ~/.ssh/config, and the
    # laptop's user name is not the cluster's. The line printed then could not work.
    job = {"SLURM_JOB_ID": "3950001", "SLURM_SUBMIT_HOST": "max-login2"}
    lines = serve_results.access_lines("127.0.0.1", 41451, "t0k", node="max027",
                                       user="someone", login=None, environ=job)
    assert lines[:2] == [
        "on your laptop, open a tunnel first:",
        "  ssh -N -L 41451:localhost:41451 -J someone@max-login2 someone@max027",
    ]
    assert "  ssh -N -L 41451:localhost:41451 -J max-login2 max027" not in lines
    text = "\n".join(lines)
    assert "(someone@max-login2 is the login node as Slurm names it" in text
    assert "--login or CRYOET_SSH_LOGIN" in text
    assert "start this again with --host 0.0.0.0" in text
    assert lines[-2:] == ["then open:", "  http://localhost:41451/?token=t0k"]

    named = serve_results.access_lines("127.0.0.1", 41451, "t0k", node="max027",
                                       user="someone", login="max-login", environ=job)
    assert named[1] == "  ssh -N -L 41451:localhost:41451 -J max-login someone@max027"
    assert "Slurm names it" not in "\n".join(named)

    wide = serve_results.access_lines("0.0.0.0", 41451, "t0k", node="max027",
                                      user="someone", login="max-login", environ=job)
    assert wide[1] == "  ssh -N -L 41451:max027:41451 max-login"
    assert "--host 0.0.0.0" not in "\n".join(wide)
    assert "  ssh -N -L 41451:localhost:41451 -J max-login someone@max027)" in wide

    unnamed = serve_results.access_lines("127.0.0.1", 41451, "t0k", node="max027",
                                         user="someone", login=None,
                                         environ={"SLURM_JOB_ID": "3950001"})
    assert unnamed[1] == "  ssh -N -L 41451:localhost:41451 -J <login node> someone@max027"
    assert serve_results.access_lines("127.0.0.1", 8765, "t0k", node="mac", user="someone",
                                      login=None, environ={}) == [
        "open:", "  http://localhost:8765/?token=t0k"]


def test_a_wide_server_also_offers_the_tunnel_to_the_node_and_warns_of_vs_code() -> None:
    # Munich, 27 November 2026, replayed: from a laptop the printed tunnel reached
    # hpc-submit02, which answered "connect failed: No route to host" for the node's
    # port; the tunnel through to the node worked. VS Code, connected to
    # hpc-submit02, had forwarded the port from there and held 127.0.0.1:37425 on the
    # laptop (lsof), so the laptop's tunnel bound only [::1].
    job = {"SLURM_JOB_ID": "4101", "SLURM_SUBMIT_HOST": "hpc-submit02"}
    lines = serve_results.access_lines("0.0.0.0", 37425, "t0k", node="cpusrv83.scidom.de",
                                       user="someone", login="hpc-submit02.scidom.de",
                                       environ=job)
    assert lines[1] == "  ssh -N -L 37425:cpusrv83.scidom.de:37425 hpc-submit02.scidom.de"
    text = "\n".join(lines)
    assert 'if ssh answers "connect failed", the login node cannot reach this port' in text
    assert ("  ssh -N -L 37425:localhost:37425 -J hpc-submit02.scidom.de "
            "someone@cpusrv83.scidom.de)") in lines
    assert "VS Code may forward this port from the login node" in text
    assert "stop that forward in\n  VS Code's Ports panel" in text
    assert lines.index("then open:") > max(
        index for index, line in enumerate(lines) if "VS Code" in line)

    narrow = serve_results.access_lines("127.0.0.1", 37425, "t0k", node="cpusrv83.scidom.de",
                                        user="someone", login="hpc-submit02.scidom.de",
                                        environ=job)
    assert "connect failed" not in "\n".join(narrow)
    assert "VS Code may forward this port" in "\n".join(narrow)
    assert not any("VS Code" in line for line in serve_results.access_lines(
        "127.0.0.1", 8765, "t0k", node="mac", user="someone", login=None, environ={}))


def test_the_project_is_read_once_before_the_link_with_what_it_holds(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []
    real_import = serve_results.importlib.import_module

    def recorded(name: str) -> object:
        imported.append(name)
        return real_import(name)

    # The pipeline's modules are timed on their own, apart from reading the project.
    monkeypatch.setattr(serve_results.importlib, "import_module", recorded)
    results = _results(tmp_path)
    missing = ClassificationRuns(tmp_path / "no-mra", cache_dir=tmp_path / "cache")
    line = serve_results.read_once(results, missing)
    assert imported == ["cryoet_pipeline.cli"]
    assert re.fullmatch(
        r"read in \d+\.\d s: the pipeline's modules \d+\.\d s; 3 series, 1 with picks, "
        rf"\d+\.\d s; no classification runs: no folder {re.escape(str(tmp_path / 'no-mra'))}",
        line,
    ), line

    empty = tmp_path / "mra"
    empty.mkdir()
    line = serve_results.read_once(results, ClassificationRuns(empty, cache_dir=tmp_path / "c"))
    assert (f"0 classification runs (a run is listed by its plan.json; Dynamo's layout is not)"
            f" and 0 FSC in {empty}, ") in line

    broken = empty / "run_a"
    broken.mkdir()
    (broken / "plan.json").write_text("{")
    line = serve_results.read_once(results, ClassificationRuns(empty, cache_dir=tmp_path / "c"))
    assert f"1 classification run (1 could not be read) and 0 FSC in {empty}, " in line


def test_the_link_is_printed_only_after_the_project_was_read(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    def stop(self: ResultsServer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(serve_results.ResultsServer, "serve_forever", stop)
    project = _project(tmp_path)
    assert serve_results.main(["--project", str(project),
                               "--cache-dir", str(tmp_path / "cache")]) == 0
    printed = capsys.readouterr().out.splitlines()
    read = [index for index, line in enumerate(printed) if line.startswith("  read in ")]
    link = [index for index, line in enumerate(printed) if "/?token=" in line]
    assert len(read) == 1 and len(link) == 1 and read[0] < link[0], printed
    assert "3 series, 1 with picks" in printed[read[0]]


def test_a_project_that_cannot_be_read_is_said_not_raised(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "artifacts.json").write_text("{")
    line = serve_results.read_once(
        ResultsProject(project, cache_dir=tmp_path / "cache"),
        ClassificationRuns(project / "mra", cache_dir=tmp_path / "cache"),
    )
    assert re.fullmatch(r"reading the project took \d+\.\d s: the pipeline's modules \d+\.\d s; "
                        r"then failed: \w+: .+", line, re.DOTALL), line


def test_each_answer_is_logged_with_the_time_it_took(
        served: tuple[str, str], capsys: pytest.CaptureFixture[str]) -> None:
    base, cookie_name = served
    capsys.readouterr()
    assert _get(f"{base}/api/project", f"{cookie_name}=secret-token")[0] == 200
    assert _get(f"{base}/api/project")[0] == 403
    logged = capsys.readouterr().err.splitlines()
    assert len(logged) == 2, logged
    assert re.fullmatch(r'\d\d:\d\d:\d\d 127\.0\.0\.1 "GET /api/project HTTP/1\.1" 200 - '
                        r"\d+\.\d s", logged[0]), logged
    assert re.fullmatch(r'\d\d:\d\d:\d\d 127\.0\.0\.1 "GET /api/project HTTP/1\.1" 403 - '
                        r"\d+\.\d s", logged[1]), logged


def test_the_login_for_the_tunnel_can_be_set_once_in_the_environment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRYOET_SSH_LOGIN", "max-login")
    assert serve_results.arguments().parse_args(["--project", "p"]).login == "max-login"
    assert serve_results.arguments().parse_args(
        ["--project", "p", "--login", "other"]).login == "other"
    monkeypatch.delenv("CRYOET_SSH_LOGIN")
    assert serve_results.arguments().parse_args(["--project", "p"]).login is None
