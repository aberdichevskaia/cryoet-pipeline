"""scripts/show_emdb.py: EMDB searched and an entry read against a lane, with a fake EMDB."""

import csv
import io
import json
import math
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.template import _box_size_from_diameter
from cryoet_pipeline.runtime import DevicePreference

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import show_emdb  # noqa: E402

SEARCH = "https://www.ebi.ac.uk/emdb/api/search/"
ENTRY = "https://www.ebi.ac.uk/emdb/api/entry/EMD-{number}"
EMPIAR = "https://www.ebi.ac.uk/empiar/api/entry/{id}/"


class FakeApis:
    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.search_csv = ""
        self.asked: list[str] = []

    def __call__(self, url: str) -> str:
        self.asked.append(url)
        if url.startswith(SEARCH):
            return self.search_csv
        if url not in self.pages:
            raise HTTPError(url, 404, "Not Found", None, None)
        return self.pages[url]


def search_rows(rows: list[tuple[str, str, str, str, str]]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["emdb_id", "title", "resolution", "structure_determination_method",
                     "release_date"])
    writer.writerows(rows)
    return out.getvalue()


def entry(number: str, *, title="Consensus map of a channel", method="subtomogramAveraging",
          resolution=4.9, symmetry="C4", box=256, pixel=1.68, status="REL",
          sample="A channel in the sarcoplasmic reticulum membrane",
          organism="Oryctolagus cuniculus", picked_with="EMD-90001") -> str:
    processing = {
        "final_reconstruction": {"resolution": {"valueOf_": str(resolution), "units": "Å"},
                                 "applied_symmetry": {"point_group": symmetry},
                                 "number_subtomograms_used": 13000},
        "extraction": {"number_tomograms": 180, "reference_model": picked_with,
                       "method": "pyTOM Template Matching"},
    }
    microscopy = {"microscope": "TFS KRIOS",
                  "calibrated_defocus_min": {"valueOf_": 2.0, "units": "µm"},
                  "calibrated_defocus_max": {"valueOf_": 7.0, "units": "µm"},
                  "image_recording_list": {"image_recording": [
                      {"film_or_detector_model": {"valueOf_": "GATAN K3 (6k x 4k)"}}]}}
    return json.dumps({
        "emdb_id": f"EMD-{number}",
        "admin": {"title": title, "current_status": {"code": {"valueOf_": status}},
                  "authors_list": {"author": [{"valueOf_": "Depositor D"}]}},
        "crossreferences": {"citation_list": {"primary_citation": {"citation_type": {
            "title": "A paper.", "journal": "Nat Commun", "year": 2026,
            "author": [{"valueOf_": name, "order": i} for i, name in
                       enumerate(["First A", "Second B", "Third C", "Fourth D"], 1)]}}}},
        "map": {"dimensions": {"col": box, "row": box, "sec": box},
                "pixel_spacing": {axis: {"valueOf_": pixel, "units": "Å"} for axis in "xyz"},
                "contour_list": {"contour": [{"level": 0.0787, "primary": True}]}},
        "sample": {"name": {"valueOf_": sample}, "supramolecule_list": {"supramolecule": [
            {"natural_source": [{"organism": {"valueOf_": organism, "ncbi": 9986}}]}]}},
        "structure_determination_list": {"structure_determination": [
            {"method": method, "image_processing": [processing],
             "microscopy_list": {"microscopy": [microscopy]}}]},
    })


def run(apis: FakeApis, *argv: str) -> dict:
    return show_emdb.build(show_emdb.parse_args(list(argv)), apis)


# --- Search ---------------------------------------------------------------------------


def test_a_search_is_counted_by_method_with_whole_averages_first():
    apis = FakeApis()
    apis.search_csv = search_rows([
        ("EMD-90010", "Channel, open state", "2.9", "singleParticle", "2026-11-04T00:00:00Z"),
        ("EMD-90011", "Channel, Local refinement of the core", "2.6", "subtomogramAveraging",
         "2026-11-04T00:00:00Z"),
        ("EMD-90012", "Channel in the native membrane", "6.0", "subtomogramAveraging",
         "2024-01-01T00:00:00Z"),
        ("EMD-90013", "Channel in the native membrane, consensus map", "4.9",
         "subtomogramAveraging", "2026-08-12T00:00:00Z"),
        ("EMD-90014", "A tomogram of muscle", "", "tomography", "2025-01-01T00:00:00Z"),
        ("EMD-90015", "Channel filament", "8", "helical", "2023-01-01T00:00:00Z"),
    ])
    report = run(apis, "search", "channel")

    assert report["found"] == 6 and report["exit"] == 0
    assert report["by_method"] == {"subtomogramAveraging": 3, "singleParticle": 1,
                                   "tomography": 1, "helical": 1}
    order = [row["emdb_id"] for row in report["rows"]]
    assert order == ["EMD-90013", "EMD-90012", "EMD-90011", "EMD-90010", "EMD-90015",
                     "EMD-90014"]
    flags = {row["emdb_id"]: (row["part"], row["membrane"]) for row in report["rows"]}
    assert flags["EMD-90011"] == (True, False)
    assert flags["EMD-90013"] == (False, True)


def test_a_method_is_asked_of_emdb_itself():
    apis = FakeApis()
    apis.search_csv = search_rows([])
    report = run(apis, "search", "ryanodine AND receptor", "--method", "subtomogramAveraging")

    assert report["found"] == 0 and report["exit"] == 1
    (url,) = apis.asked
    assert quote("(ryanodine AND receptor) AND structure_determination_method:"
                 "subtomogramAveraging") in url


def test_entries_made_from_the_empiar_entrys_data_are_marked():
    apis = FakeApis()
    apis.search_csv = search_rows([
        ("EMD-90020", "Ribosome", "3.1", "subtomogramAveraging", "2023-01-01T00:00:00Z"),
        ("EMD-90021", "Ribosome, other data", "4.0", "subtomogramAveraging",
         "2022-01-01T00:00:00Z"),
    ])
    apis.pages[EMPIAR.format(id="12345")] = json.dumps(
        {"EMPIAR-12345": {"cross_references": ["EMD-90020"]}})
    report = run(apis, "search", "ribosome", "--empiar", "12345")

    marked = {row["emdb_id"]: row["same_data"] for row in report["rows"]}
    assert marked == {"EMD-90020": True, "EMD-90021": False}


# --- One entry ----------------------------------------------------------------------------


def test_an_entry_is_read_from_the_paths_emdb_uses():
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90001")] = entry("90001")
    report = run(apis, "entry", "emd_90001")
    facts = report["entry"]

    assert report["exit"] == 0 and report["emdb_id"] == "EMD-90001"
    assert (facts["method"], facts["resolution"], facts["symmetry"]) == (
        "subtomogramAveraging", 4.9, "C4")
    assert (facts["map_box"], facts["map_pixel"], facts["contour"]) == (
        [256, 256, 256], [1.68, 1.68, 1.68], 0.0787)
    assert facts["organisms"] == ["Oryctolagus cuniculus"]
    assert facts["picked_with"] == "EMD-90001"
    assert facts["detectors"] == ["GATAN K3 (6k x 4k)"]
    assert facts["defocus_um"] == [2.0, 7.0]
    # The paper's authors, not the depositor.
    assert facts["authors"][:2] == ["First A", "Second B"]


def test_a_map_is_padded_resampled_and_low_passed_as_prepare_template_would():
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90001")] = entry("90001")
    report = run(apis, "entry", "EMD-90001", "--pixel", "13.44", "--diameter", "340")
    checks = report["checks"]

    assert ("the template box is 52 px = 699 A, twice the diameter; the map is 430 A across, "
            "so pytom pads it up to the box") in checks
    assert "the map is resampled from 1.68 to 13.44 A/px (8.0 times coarser)" in checks
    assert "4.9 A is finer than the template's low-pass, about 2 x 13.44 = 27 A" in checks
    assert any(check.startswith("its title or sample names a membrane") for check in checks)
    assert "it was averaged with C4 symmetry" in checks


@pytest.mark.parametrize(("diameter", "pixel"), [(340, 13.44), (250, 9.8), (130, 5.4), (8, 3)])
def test_the_box_is_prepare_templates_own(diameter, pixel):
    context = BackendContext(output_dir=Path("."), device=DevicePreference.CPU,
                             parameters={"template_particle_diameter_angstrom": diameter})
    assert show_emdb.template_box(diameter, pixel) == _box_size_from_diameter(
        context, voxel_size_angstrom=pixel)


def test_a_map_larger_than_the_box_keeps_its_own_and_one_too_small_is_said_so():
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90002")] = entry("90002", box=400, pixel=2.0)
    apis.pages[ENTRY.format(number="90003")] = entry("90003", box=64, pixel=2.0)
    large = run(apis, "entry", "90002", "--pixel", "10", "--diameter", "200")["checks"]
    small = run(apis, "entry", "90003", "--pixel", "10", "--diameter", "200")["checks"]

    assert ("the map is 800 A across, more than the 40 px = 400 A template box; pytom keeps "
            "the map's box and warns") in large
    assert ("the map is 128 A across, less than the 200 A particle: it cannot hold it"
            in small)


def test_a_coarse_map_a_region_and_a_tomogram_are_flagged():
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90004")] = entry(
        "90004", title="Local refinement of the core", resolution=30, pixel=12.0,
        sample="purified channel", symmetry="C1")
    apis.pages[ENTRY.format(number="90005")] = entry("90005", method="tomography",
                                                     sample="cells")
    checks = run(apis, "entry", "90004", "--pixel", "10", "--diameter", "200")["checks"]
    tomogram = run(apis, "entry", "90005")["checks"]

    assert "its title names a map of a region, not of the whole particle" in checks
    assert ("the map's pixel, 12 A, is coarser than the tomogram's, 10 A: it would be "
            "resampled finer") in checks
    assert ("30 A is coarser than the template's low-pass, about 2 x 10 = 20 A: the map "
            "limits the template") in checks
    assert not any("symmetry" in check for check in checks)
    assert "it is a tomogram, not an average: no template" in tomogram


def test_independence_is_read_from_the_empiar_cross_references(capsys):
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90001")] = entry("90001")
    apis.pages[EMPIAR.format(id="12345")] = json.dumps(
        {"EMPIAR-12345": {"cross_references": [{"name": "EMD-90001"}]}})
    apis.pages[EMPIAR.format(id="12346")] = json.dumps(
        {"EMPIAR-12346": {"cross_references": ["EMD-90099"]}})

    assert show_emdb.main(["entry", "EMD-90001", "--empiar", "12345"], read=apis) == 0
    assert "made from the same data" in capsys.readouterr().out
    assert show_emdb.main(["entry", "EMD-90001", "--empiar", "12346"], read=apis) == 0
    assert "EMPIAR-12346 does not cross-reference it" in capsys.readouterr().out


def test_an_entry_emdb_lacks_or_has_withdrawn_is_not_offered(capsys):
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90006")] = entry("90006", status="OBS")
    assert show_emdb.main(["entry", "EMD-99999"], read=apis) == 1
    assert "EMDB has no such entry" in capsys.readouterr().out
    report = run(apis, "entry", "90006")
    assert report["exit"] == 1
    assert "its status is OBS, not released" in report["checks"]


@pytest.mark.parametrize("argv", [
    ["entry", "EMD-12"],
    ["entry", "EMD-90001", "--pixel", "10"],
    ["entry", "EMD-90001", "--pixel", "-1", "--diameter", "200"],
    ["search", "channel", "--empiar", "EMPIAR-12345"],
])
def test_contradicting_arguments_stop_with_status_2(argv):
    apis = FakeApis()
    assert show_emdb.main(argv, read=apis) == 2
    assert apis.asked == []


def test_the_text_says_what_the_json_does(capsys):
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90001")] = entry("90001")
    argv = ["entry", "EMD-90001", "--pixel", "13.44", "--diameter", "340"]
    assert show_emdb.main([*argv, "--json"], read=apis) == 0
    report = json.loads(capsys.readouterr().out)
    assert show_emdb.main(argv, read=apis) == 0
    text = capsys.readouterr().out

    assert all(check in text for check in report["checks"])
    assert "map: 256 x 256 x 256 at 1.68 A/px = 430 A across; authors' contour 0.0787" in text
    assert "citation: A paper. Nat Commun 2026 (First A, Second B, Third C et al.)" in text


# --- Measure a map --------------------------------------------------------------------


def density(*balls: tuple[float, tuple[float, float, float]], box: int = 48) -> np.ndarray:
    """Balls of 1 in 0s: (radius, centre from the box centre as z, y, x), in voxels."""

    z, y, x = np.indices((box, box, box), dtype=np.float64) - (box - 1) / 2
    data = np.zeros((box, box, box), np.float32)
    for radius, (cz, cy, cx) in balls:
        data[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2] = 1.0
    return data


def map_file(path: Path, data: np.ndarray, voxel_size: float = 2.0) -> Path:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = voxel_size
    return path


class FakeFetch:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.asked: list[tuple[str, Path]] = []

    def __call__(self, emdb_id: str, *, cache_dir: Path, progress=None) -> Path:
        self.asked.append((emdb_id, cache_dir))
        return self.path


@pytest.fixture
def fetched(tmp_path, monkeypatch):
    fetch = FakeFetch(map_file(tmp_path / "emd_90001.map", density((10, (0, 0, 6)))))
    monkeypatch.setattr(show_emdb, "fetch_emdb_map", fetch)
    return fetch


def test_an_entrys_map_is_measured_at_the_authors_contour(fetched, tmp_path, capsys):
    from cryoet_pipeline.map_size import measure_density

    apis = FakeApis()
    apis.pages[ENTRY.format(number="90001")] = entry("90001")
    cache = tmp_path / "project" / "emdb-cache"
    argv = ["measure", "EMD-90001", "--cache", str(cache)]
    assert show_emdb.main([*argv, "--json"], read=apis) == 0
    report = json.loads(capsys.readouterr().out)
    expected = measure_density(density((10, (0, 0, 6))), 2.0, 0.0787)

    assert fetched.asked == [("EMD-90001", cache)]
    assert (report["level"], report["level_from"]) == (0.0787, "authors")
    assert report["second_level"] == pytest.approx(0.0787 * 1.5)
    assert report["first"]["widest_angstrom"] == pytest.approx(expected.widest_angstrom)
    # Six voxels off the centre: the mask's diameter, not the particle's width, is proposed.
    assert report["diameter_angstrom"] == math.ceil(expected.mask_diameter_angstrom)
    assert report["diameter_angstrom"] > expected.widest_angstrom + 2 * 6 * 2.0 - 1

    assert show_emdb.main(argv, read=apis) == 0
    text = capsys.readouterr().out
    diameter = report["diameter_angstrom"]
    assert "EMD-90001 above the authors' contour, 0.0787 (a 48 x 48 x 48 box at 2 A/px):" in text
    assert f"widest across: {expected.widest_angstrom:.0f} A" in text
    assert f"For [pick] particle_diameter_angstrom: {diameter} A, the mask's diameter" in text
    assert (f"show_emdb.py entry EMD-90001 --pixel <tomogram A/px> --diameter {diameter}"
            in text)


def test_maps_go_to_the_users_cache_unless_told(fetched, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90001")] = entry("90001")
    assert show_emdb.main(["measure", "90001"], read=apis) == 0

    assert fetched.asked == [("EMD-90001", tmp_path / "cache" / "cryoet-pipeline" / "emdb")]


def test_a_map_file_needs_a_level_and_reads_no_entry(fetched, tmp_path, capsys):
    path = map_file(tmp_path / "local.mrc", density((8, (0, 0, 0))))
    apis = FakeApis()

    assert show_emdb.main(["measure", str(path)], read=apis) == 2
    assert "give its level with --level" in capsys.readouterr().err
    assert show_emdb.main(["measure", str(path), "--level", "0.5"], read=apis) == 0
    text = capsys.readouterr().out

    assert apis.asked == [] and fetched.asked == []
    assert f"{path} above the level, 0.5" in text
    assert "For [pick] particle_diameter_angstrom:" in text
    assert "show_emdb.py entry" not in text


@pytest.mark.parametrize(("data", "why"), [
    (np.maximum(density((8, (0, 0, 0))), np.pad(np.ones((4, 48, 48), np.float32),
                                                  ((22, 22), (0, 0), (0, 0)))),
     "the density reaches the map's edge"),
    (np.kron(np.ones((6, 6, 6), np.float32), np.pad(np.ones((2, 2, 2), np.float32), 3)),
     "more than 4 pieces are measured"),
])
def test_no_diameter_is_proposed_for_what_is_not_one_particle(tmp_path, capsys, data, why):
    path = map_file(tmp_path / "many.mrc", data)
    assert show_emdb.main(["measure", str(path), "--level", "0.5", "--json"],
                          read=FakeApis()) == 0
    assert json.loads(capsys.readouterr().out)["diameter_angstrom"] is None
    assert show_emdb.main(["measure", str(path), "--level", "0.5"], read=FakeApis()) == 0
    assert f"No diameter to propose: {why}." in capsys.readouterr().out


def test_an_entry_without_a_map_or_a_contour_downloads_nothing(fetched, capsys):
    apis = FakeApis()
    apis.pages[ENTRY.format(number="90006")] = entry("90006", status="OBS")
    no_contour = json.loads(entry("90007"))
    no_contour["map"]["contour_list"]["contour"] = []
    apis.pages[ENTRY.format(number="90007")] = json.dumps(no_contour)

    assert show_emdb.main(["measure", "EMD-99999"], read=apis) == 1
    assert "EMD-99999: EMDB has no such entry" in capsys.readouterr().out
    assert show_emdb.main(["measure", "EMD-90006"], read=apis) == 1
    assert "its status is OBS, not released: it has no map" in capsys.readouterr().out
    assert show_emdb.main(["measure", "EMD-90007"], read=apis) == 1
    assert "EMDB gives no contour for it: give one with --level" in capsys.readouterr().out
    assert fetched.asked == []
    assert show_emdb.main(["measure", "EMD-90007", "--level", "0.5"], read=apis) == 0
    assert len(fetched.asked) == 1


def test_a_source_that_is_neither_a_file_nor_an_entry_stops_with_status_2(fetched, capsys):
    apis = FakeApis()
    assert show_emdb.main(["measure", "no/such/map.mrc"], read=apis) == 2
    assert "neither a map file nor an EMDB id" in capsys.readouterr().err
    assert apis.asked == [] and fetched.asked == []
