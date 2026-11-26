"""The result pages' classification views: runs of <project>/mra and their FSC.

The run is a real one, a short MRA of test_mra_setup's twelve particles computed by
`mra_runs.run_plan`, read as the classification report reads it. The pictures are
checked against marker voxels placed by hand, so that a flip or a transposition of a
section shows; the maps handed to Mol* against the reader's own arrays.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("matplotlib")
pytest.importorskip("pyarrow")

from test_classification_reader import QUIET, _short  # noqa: E402
from test_mra_setup import _export, _plan  # noqa: E402

from cryoet_pipeline.backends.dynamo_search import _half_map_fsc  # noqa: E402
from cryoet_pipeline.dashboard.data import sign_inside  # noqa: E402
from cryoet_pipeline.dashboard.mra import (  # noqa: E402
    ClassificationRuns,
    map_pictures,
    section_stack,
)
from cryoet_pipeline.mra_runs import run_plan  # noqa: E402
from cryoet_pipeline.reports.classification import class_changes  # noqa: E402
from cryoet_pipeline.reports.classification_reader import read_classification  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from fsc_phase_randomization import falls  # noqa: E402

RUN = "run1"
FSC = "halves"


def write_fsc(folder: Path, name: str, halves: dict[int, np.ndarray], mask: np.ndarray,
              spacing: float) -> None:
    """The files scripts/consensus_halves.py maps writes, from two half maps."""

    curves = {}
    for label, curve in _half_map_fsc(halves, mask, spacing).items():
        assert isinstance(curve, dict)
        resolutions, values = curve["resolution_angstrom"], curve["correlation"]
        curves[label] = {"resolution_angstrom": resolutions, "correlation": values,
                         **{f"falls_{level}_angstrom": falls(resolutions, values, level)
                            for level in (0.5, 0.143)},
                         "rise_toward_nyquist": curve["rise_toward_nyquist"]}
    (folder / f"{name}_fsc.json").write_text(json.dumps({
        "halves": [str(folder / "refine_h1"), str(folder / "refine_h2")], "iteration": 3,
        "start_poses": "picked", "pixel_spacing_angstrom": spacing, **curves}, indent=1))
    for suffix, volume in (("mean", (halves[1] + halves[2]) / 2), ("mask", mask)):
        with mrcfile.new(folder / f"{name}_{suffix}.mrc", overwrite=True) as handle:
            handle.set_data(np.ascontiguousarray(volume, dtype=np.float32))
            handle.voxel_size = spacing


@pytest.fixture(scope="module")
def folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An export with one finished run of three iterations, and an FSC beside it."""

    export = _export(tmp_path_factory.mktemp("mra"))
    run_plan(_short(_plan(export), 3), export / RUN, **QUIET)
    run = read_classification(export / RUN, compute_initial_average=False)
    assert run.mask is not None and run.pixel_size_angstrom
    average = next(iter(run.iterations[-1].classes.values())).average
    assert average is not None
    signal = average.load()
    noise = np.random.default_rng(3)
    halves = {k: (signal + noise.normal(0.0, 0.3 * signal.std(), signal.shape)).astype(np.float32)
              for k in (1, 2)}
    write_fsc(export, FSC, halves, run.mask.load(), run.pixel_size_angstrom)
    return export


def _runs(folder: Path, tmp_path: Path) -> ClassificationRuns:
    return ClassificationRuns(folder, cache_dir=tmp_path / "cache")


def _png(body: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(body)).convert("L"), dtype=int)


def test_the_listing_holds_the_runs_and_the_fsc_curves(folder: Path, tmp_path: Path) -> None:
    listing = _runs(folder, tmp_path).listing()
    # test_mra_setup's export also holds a Dynamo project, mra_t, with no plan.json.
    assert [one["name"] for one in listing["runs"]] == [RUN]
    assert listing["runs"][0]["finished"] == 3 and listing["runs"][0]["planned"] == 3
    assert listing["fsc"] == [{"name": FSC}]
    assert ClassificationRuns(folder / "absent", cache_dir=tmp_path).listing()["runs"] == []


def test_a_run_reads_as_the_classification_report_reads_it(folder: Path, tmp_path: Path) -> None:
    document = _runs(folder, tmp_path).run(RUN)
    run = read_classification(folder / RUN, compute_initial_average=False)
    assert document["finished"] == len(run.iterations) == 3
    assert [one["number"] for one in document["iterations"]] == [0, 1, 2, 3]
    starting = document["iterations"][0]["classes"]
    assert [one["channel"] for one in starting] == sorted(run.references)
    for page, iteration in zip(document["iterations"][1:], run.iterations, strict=True):
        assert {one["channel"]: one["members"] for one in page["classes"]} == {
            channel: len(result.members) for channel, result in iteration.classes.items()}
    assert [one["changed"] for one in document["iterations"][1:]] == class_changes(run.iterations)
    for channel, sizes in document["sizes"].items():
        assert sizes == [len(one.classes[int(channel)].members) if int(channel) in one.classes
                         else None for one in run.iterations]
    assert document["box_px"] == run.box_px and document["particles"] == len(run.tags)
    json.dumps(document, allow_nan=False)


def test_every_section_and_view_puts_a_voxel_where_it_is() -> None:
    """Markers placed by hand in a 9-voxel box, read back from the PNGs."""

    box = np.zeros((9, 9, 9), np.float32)
    box[4, 1, 6] = 10.0  # on the central z section: the XY view
    box[2, 4, 7] = 20.0  # on the central y section: the XZ view
    box[6, 3, 4] = 30.0  # on the central x section: the YZ view
    sections, views = (_png(one) for one in map_pictures(box))

    # Section z fills rows 9 z to 9 z + 8, y drawn upwards: row 9 z + 8 - y.
    assert sections.shape == (81, 9)
    assert {tuple(int(one) for one in spot) for spot in np.argwhere(sections > 200)} == {
        (9 * 4 + 8 - 1, 6), (9 * 2 + 8 - 4, 7), (9 * 6 + 8 - 3, 4)}

    # XY | gap | XZ | gap | YZ, each 9 wide with a one-column gap, the second axis
    # across and the first upwards: XY (y, x), XZ (z, x), YZ (z, y).
    assert views.shape == (9, 29)
    assert {tuple(int(one) for one in spot) for spot in np.argwhere(views > 200)} == {
        (8 - 1, 6), (8 - 2, 10 + 7), (8 - 6, 20 + 3)}


def test_a_class_is_handed_to_mol_star_as_the_reader_returns_it(
    folder: Path, tmp_path: Path
) -> None:
    runs = _runs(folder, tmp_path)
    document = runs.run(RUN)
    run = read_classification(folder / RUN, compute_initial_average=False)
    last = run.iterations[-1]
    channel = next(iter(last.classes))
    body = runs.run_file(RUN, document["key"], f"class_{last.number}_{channel}.mrc")
    with mrcfile.open(_write(tmp_path / "class.mrc", body)) as handle:
        header = handle.header
        assert (int(header.mapc), int(header.mapr), int(header.maps)) == (1, 2, 3)
        assert float(handle.voxel_size.x) == pytest.approx(run.pixel_size_angstrom)
        average = last.classes[channel].average
        assert average is not None
        np.testing.assert_array_equal(handle.data, average.load())
    reference = runs.run_file(RUN, document["key"], "class_0_3.mrc")
    with mrcfile.open(_write(tmp_path / "reference.mrc", reference)) as handle:
        np.testing.assert_array_equal(handle.data, run.references[3].load())
    mask = runs.run_file(RUN, document["key"], "mask.mrc")
    with mrcfile.open(_write(tmp_path / "mask.mrc", mask)) as handle:
        assert run.mask is not None
        np.testing.assert_array_equal(handle.data, run.mask.load())

    stem = f"class_{last.number}_{channel}"
    sections = _png(runs.run_file(RUN, document["key"], f"{stem}_sections.png"))
    assert sections.shape == (run.box_px * run.box_px, run.box_px)


def _write(path: Path, body: bytes) -> Path:
    path.write_bytes(body)
    return path


def test_only_the_run_s_own_maps_are_handed_out(folder: Path, tmp_path: Path) -> None:
    runs = _runs(folder, tmp_path)
    key = runs.run(RUN)["key"]
    for wrong in ("../plan.json", "plan.json", "class_9_1.mrc", "class_1_9_views.png",
                  "class_1_1.png", "mask.png"):
        with pytest.raises(KeyError):
            runs.run_file(RUN, key, wrong)
    with pytest.raises(KeyError, match="gone on"):
        runs.run_file(RUN, "0" * 16, "mask.mrc")
    with pytest.raises(KeyError):
        runs.run("mra_t")
    with pytest.raises(KeyError):
        runs.run("../" + RUN)


def test_a_run_that_goes_on_is_read_again(folder: Path, tmp_path: Path) -> None:
    runs = _runs(folder, tmp_path)
    before = runs.run(RUN)["key"]
    record = folder / RUN / "ite_0003" / "run.json"
    stat = record.stat()
    os.utime(record, (stat.st_atime, stat.st_mtime + 30))
    try:
        assert runs.run(RUN)["key"] != before
    finally:
        os.utime(record, (stat.st_atime, stat.st_mtime))


def test_the_fsc_is_carried_in_spatial_frequency(folder: Path, tmp_path: Path) -> None:
    runs = _runs(folder, tmp_path)
    raw = json.loads((folder / f"{FSC}_fsc.json").read_text())
    document = runs.fsc(FSC)
    json.dumps(document, allow_nan=False)
    masked = document["curves"]["masked"]
    assert masked["frequency"] == pytest.approx(
        [1.0 / one for one in raw["masked"]["resolution_angstrom"]])
    assert masked["correlation"] == pytest.approx(raw["masked"]["correlation"])
    assert masked["falls"]["0.143"] == pytest.approx(raw["masked"]["falls_0.143_angstrom"])
    assert document["nyquist_frequency"] == pytest.approx(1 / (2 * raw["pixel_spacing_angstrom"]))
    assert document["halves"] == ["refine_h1", "refine_h2"]
    assert document["maps"] == ["mask.mrc", "mean.mrc"]
    assert runs.fsc_file(FSC, document["key"], "mean.mrc") == (
        folder / f"{FSC}_mean.mrc").read_bytes()
    with pytest.raises(KeyError):
        runs.fsc_file(FSC, document["key"], "half1.mrc")
    with pytest.raises(KeyError):
        runs.fsc("absent")


def test_a_value_a_browser_cannot_read_becomes_null() -> None:
    from cryoet_pipeline.dashboard.mra import fsc_document

    curve = {"resolution_angstrom": [float("inf"), 40.0, 20.0],
             "correlation": [1.0, float("nan"), 0.1],
             "falls_0.5_angstrom": [25.0], "falls_0.143_angstrom": [],
             "rise_toward_nyquist": None}
    document = fsc_document({"pixel_spacing_angstrom": 5.0, "masked": curve}, "x", "k")
    json.dumps(document, allow_nan=False)
    masked = document["curves"]["masked"]
    assert masked["frequency"] == [0.0, 1 / 40.0, 1 / 20.0]
    assert masked["correlation"] == [1.0, None, 0.1]
    assert masked["falls"] == {"0.5": [25.0], "0.143": []}
    assert "unmasked" not in document["curves"]


def test_the_protein_side_is_the_denser_side_inside_the_mask() -> None:
    volume = np.zeros((6, 6, 6), np.float32)
    mask = np.zeros_like(volume)
    mask[1:5, 1:5, 1:5] = 1.0
    volume[2:4, 2:4, 2:4] = 3.0
    assert sign_inside(volume, mask) == 1
    assert sign_inside(-volume, mask) == -1
    assert sign_inside(volume, None) is None
    assert sign_inside(volume, np.ones_like(volume)) is None
    assert sign_inside(volume, mask[:5]) is None


def test_the_server_hands_out_runs_and_curves(folder: Path, tmp_path: Path) -> None:
    import threading

    from test_dashboard import _get, _results

    from cryoet_pipeline.dashboard.server import ResultsServer

    server = ResultsServer(("127.0.0.1", 0), _results(tmp_path), "token",
                           classification=_runs(folder, tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base, cookie = f"http://127.0.0.1:{server.port}", f"{server.cookie_name}=token"
    try:
        listing = json.loads(_get(f"{base}/api/project", cookie)[2])["classification"]
        assert [one["name"] for one in listing["runs"]] == [RUN]
        run = json.loads(_get(f"{base}/api/mra/{RUN}", cookie)[2])
        status, headers, body = _get(f"{base}/mra/{RUN}/{run['key']}/class_1_1_views.png", cookie)
        assert status == 200 and headers["Content-Type"] == "image/png"
        assert _get(f"{base}/mra/{RUN}/{run['key']}/class_1_1_views.png")[0] == 403
        curve = json.loads(_get(f"{base}/api/fsc/{FSC}", cookie)[2])
        status, _, body = _get(f"{base}/fsc/{FSC}/{curve['key']}/mean.mrc", cookie)
        assert status == 200 and body == (folder / f"{FSC}_mean.mrc").read_bytes()
        assert _get(f"{base}/api/mra/mra_t", cookie)[0] == 404
        assert _get(f"{base}/fsc/{FSC}/{curve['key']}/{FSC}_fsc.json", cookie)[0] == 404
    finally:
        server.shutdown()
        server.server_close()


def test_a_stack_has_every_section_once(folder: Path) -> None:
    volume = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2).numpy()
    stacked = section_stack(volume)
    assert stacked.shape == (12, 2)
    for z in range(4):
        np.testing.assert_array_equal(stacked[3 * z: 3 * z + 3], volume[z][::-1])
