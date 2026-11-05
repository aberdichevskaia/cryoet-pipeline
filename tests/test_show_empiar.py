"""scripts/show_empiar.py: an EMPIAR entry described, and a layout checked, against a fake EBI."""

import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import show_empiar  # noqa: E402

BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/{id}/data/"
API = "https://www.ebi.ac.uk/empiar/api/entry/{id}/"
DOSE_SYMMETRIC = [0.0, 3.0, -3.0, -6.0, 6.0]


def listing(empiar_id: str, entries: list[tuple[str, str]]) -> str:
    """An Apache listing as EBI's server writes it; folders end in "/"."""

    rows = [
        '<tr><td valign="top"><img src="/icons/back.gif" alt="[PARENTDIR]"></td><td><a href='
        f'"/empiar/world_availability/{empiar_id}/">Parent Directory</a></td><td>&nbsp;</td>'
        '<td align="right">  - </td><td>&nbsp;</td></tr>'
    ]
    for name, size in entries:
        # EBI escapes brackets in lower case, as %5b and %5d.
        href = quote(name).replace("%5B", "%5b").replace("%5D", "%5d")
        rows.append(
            f'<tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]"></td><td><a href='
            f'"{href}">{name}</a></td><td align="right">2019-09-19 21:29  </td><td align="right">'
            f"{size}</td><td>&nbsp;</td></tr>"
        )
    return "<html><body><table>\n" + "\n".join(rows) + "\n</table></body></html>"


def mdoc(frames: list[str], angles=DOSE_SYMMETRIC) -> str:
    lines = ["PixelSpacing = 1.35", "Voltage = 300", ""]
    for z, (name, angle) in enumerate(zip(frames, angles, strict=True)):
        lines += [f"[ZValue = {z}]", f"TiltAngle = {angle}", "PixelSpacing = 1.35",
                  "Binning = 1", "NumSubFrames = 8", f"SubFramePath = D:\\DATA\\{name}", ""]
    return "\n".join(lines)


def api(title: str, imagesets: list[dict]) -> str:
    return json.dumps({"EMPIAR-X": {"title": title, "dataset_size": "1.0 GB",
                                    "imagesets": imagesets}})


class FakeEbi:
    def __init__(self) -> None:
        self.pages: dict[str, str] = {}
        self.asked: list[str] = []

    def __call__(self, url: str) -> str:
        self.asked.append(url)
        if url not in self.pages:
            raise HTTPError(url, 404, "Not Found", None, None)
        return self.pages[url]


def tomo_10164(ebi: FakeEbi, *, listed=5, named=5) -> None:
    """Movies in frames/, one mdoc per series in mdoc-files/."""

    base = BASE.format(id="10164")
    ebi.pages[API.format(id="10164")] = api("HIV VLPs", [{
        "name": "frames", "directory": "data/frames", "category": "micrographs - multiframe",
        "data_format": "MRC", "num_images_or_tilt_series": 2, "frames_per_image": 8,
        "pixel_width": 1.35}])
    ebi.pages[base] = listing("10164", [("frames/", "  - "), ("mdoc-files/", "  - ")])
    frames, mdocs = [], []
    for series in ("TS_01", "TS_03"):
        names = [f"{series}_{z:03d}_{a}.mrc" for z, a in enumerate(DOSE_SYMMETRIC)]
        frames += [(name, "456M") for name in names[:listed]]
        mdocs.append((f"{series}.mrc.mdoc", " 21K"))
        ebi.pages[f"{base}mdoc-files/{series}.mrc.mdoc"] = mdoc(names[:named] + [
            f"{series}_{z:03d}_x.mrc" for z in range(named, 5)])
    ebi.pages[f"{base}frames/"] = listing("10164", frames)
    ebi.pages[f"{base}mdoc-files/"] = listing("10164", mdocs)


def run(ebi: FakeEbi, *argv: str) -> dict:
    reader = show_empiar.Reader(read=ebi, retries=0, sleep=lambda seconds: None)
    return show_empiar.build(show_empiar.parse_args(list(argv)), reader)


# --- Describing an entry ---------------------------------------------------------------


def test_the_entry_and_its_folders_are_described():
    ebi = FakeEbi()
    tomo_10164(ebi)
    report = run(ebi, "10164")

    assert report["exit"] == 0
    (imageset,) = report["entry"]["imagesets"]
    assert (imageset["directory"], imageset["data_format"], imageset["pixel_width"]) == (
        "data/frames", "MRC", 1.35)
    folders = {folder["path"]: folder for folder in report["folders"]}
    assert folders[""]["folders"] == 2
    assert folders["frames"]["files"] == {".mrc": {"count": 10,
                                                   "size_bytes": 10 * 456 * 1024**2}}
    assert folders["mdoc-files"]["series_from_mdocs"] == {"count": 2, "ids": ["TS_01", "TS_03"]}


def test_a_folder_of_many_folders_is_shown_by_three():
    ebi = FakeEbi()
    base = BASE.format(id="11830")
    ebi.pages[API.format(id="11830")] = api("Chlamy", [])
    names = [f"Position_{n:02d}" for n in range(10)]
    ebi.pages[base] = listing("11830", [(f"{name}/", "  - ") for name in names])
    for name in names:
        ebi.pages[f"{base}{name}/"] = listing("11830", [(f"{name}.mdoc", " 27K")])
    report = run(ebi, "11830")

    root = report["folders"][0]
    assert (root["folders"], root["not_read"]) == (10, 7)
    read = [folder["path"] for folder in report["folders"][1:]]
    assert read == ["Position_00", "Position_05", "Position_09"]


def test_sizes_are_read_as_the_listing_rounds_them():
    assert show_empiar.parse_size(" 21K") == 21 * 1024
    assert show_empiar.parse_size("1.2G") == int(1.2 * 1024**3)
    assert show_empiar.parse_size("456") == 456
    assert show_empiar.parse_size("  - ") is None
    assert show_empiar.size_text(39 * 1024**3) == "39.0 GiB"


def test_an_entry_the_api_does_not_have_is_said_so(capsys):
    ebi = FakeEbi()
    reader = show_empiar.Reader(read=ebi, retries=0)
    assert show_empiar.main(["99999"], reader=reader) == 1
    assert "the EMPIAR API has no such entry" in capsys.readouterr().out


# --- Checking a layout -----------------------------------------------------------------


def test_each_series_is_checked_against_its_mdoc_and_a_command_is_given():
    ebi = FakeEbi()
    tomo_10164(ebi)
    report = run(ebi, "10164", "--series", "TS_01", "--series", "TS_03")

    assert report["exit"] == 0
    for row in report["check"]["series"]:
        assert row["verdict"] == "accepted"
        assert row["folders"]["data/frames"]["files"] == {".mrc": 5}
        assert row["size_bytes"] == 5 * 456 * 1024**2 + 21 * 1024
        assert (row["mdoc"]["tilts"], row["mdoc"]["frames_taken"]) == (5, 5)
        assert row["mdoc"]["missing_count"] == row["mdoc"]["extra_count"] == 0
    assert report["command"] == ("cryoet download-empiar --empiar-id 10164 --out "
                                 "data/empiar-10164 --tilt-series TS_01 --tilt-series TS_03")


def test_a_frame_the_mdoc_names_but_the_listing_lacks_is_refused():
    ebi = FakeEbi()
    tomo_10164(ebi, listed=4)
    report = run(ebi, "10164", "--series", "TS_01")

    (row,) = report["check"]["series"]
    assert row["verdict"] == "refused"
    assert (row["mdoc"]["missing_count"], row["mdoc"]["missing"]) == (1, ["TS_01_004_6.0.mrc"])
    assert report["exit"] == 1 and report["command"] is None


def test_a_pattern_that_also_takes_another_series_frames_is_refused(tmp_path):
    # {tilt_series}*.mrc for TS_1 takes TS_10's frames too; the mdoc names only TS_1's.
    ebi = FakeEbi()
    base = BASE.format(id="12345")
    ebi.pages[API.format(id="12345")] = api("Prefixes", [])
    names = {series: [f"{series}_{z:03d}_{a}.mrc" for z, a in enumerate(DOSE_SYMMETRIC)]
             for series in ("TS_1", "TS_10")}
    ebi.pages[f"{base}frames/"] = listing("12345", [(n, "100M") for ns in names.values()
                                                    for n in ns])
    ebi.pages[f"{base}mdocs/"] = listing("12345", [("TS_1.mrc.mdoc", " 21K")])
    ebi.pages[f"{base}mdocs/TS_1.mrc.mdoc"] = mdoc(names["TS_1"])
    layout = tmp_path / "12345.yaml"
    layout.write_text("groups:\n"
                      "  - remote_dir: mdocs\n    file_pattern: '{tilt_series}.mrc.mdoc'\n"
                      "  - remote_dir: frames\n    file_pattern: '{tilt_series}*.mrc'\n")
    report = run(ebi, "12345", "--layout", str(layout), "--series", "TS_1")

    (row,) = report["check"]["series"]
    assert row["verdict"] == "refused"
    assert (row["mdoc"]["frames_taken"], row["mdoc"]["extra_count"]) == (10, 5)
    assert row["mdoc"]["extra"][0] == "TS_10_000_0.0.mrc"
    assert report["exit"] == 1


def test_a_series_the_layout_finds_nothing_for_is_refused_in_the_downloaders_words():
    ebi = FakeEbi()
    tomo_10164(ebi)
    report = run(ebi, "10164", "--series", "TS_99")

    (row,) = report["check"]["series"]
    assert row["verdict"] == "refused"
    assert "no files matched pattern '{tilt_series}.mrc.mdoc' for TS_99" in row["refusal"]


def tomo_11830(ebi: FakeEbi, layout: Path) -> None:
    """A folder per series, bracketed EER names, a session's gain shared by all."""

    base = BASE.format(id="11830")
    ebi.pages[API.format(id="11830")] = api("Chlamy", [{
        "name": "tilt series", "directory": "data/cvp", "category": "tilt series",
        "data_format": "EER", "num_images_or_tilt_series": 2, "pixel_width": 1.96}])
    ebi.pages[f"{base}cvp/01122021/gainref/"] = listing("11830", [("session.gain", " 34M")])
    for series in ("01122021_pos3", "01122021_pos4"):
        names = [f"{series}_{z + 1:03d}[{a:.2f}]_EER.eer" for z, a in enumerate(DOSE_SYMMETRIC)]
        ebi.pages[f"{base}cvp/{series}/"] = listing("11830", [(f"{series}.mdoc", " 27K"),
                                                              ("frames/", "  - ")])
        ebi.pages[f"{base}cvp/{series}/{series}.mdoc"] = mdoc(names)
        ebi.pages[f"{base}cvp/{series}/frames/"] = listing("11830",
                                                           [(name, "103M") for name in names])
    layout.write_text(
        "groups:\n"
        "  - remote_dir: cvp/{tilt_series}\n    file_pattern: '{tilt_series}.mdoc'\n"
        "    local_dir: mdocs\n"
        "  - remote_dir: cvp/{tilt_series}/frames\n    file_pattern: '*.eer'\n"
        "    local_dir: frames\n"
        "  - remote_dir: cvp/01122021/gainref\n    file_pattern: '*.gain'\n"
        "    local_dir: gain\n    shared: true\n"
    )


def test_folders_per_series_bracketed_names_and_a_shared_gain(tmp_path):
    ebi = FakeEbi()
    layout = tmp_path / "11830.yaml"
    tomo_11830(ebi, layout)
    report = run(ebi, "11830", "--layout", str(layout), "--series", "01122021_pos3",
                 "--series", "01122021_pos4")

    assert report["exit"] == 0
    for row in report["check"]["series"]:
        assert row["verdict"] == "accepted"
        # The mdoc names the frames with brackets, the listing escapes them.
        assert (row["mdoc"]["tilts"], row["mdoc"]["missing_count"]) == (5, 0)
        assert row["folders"] == {"mdocs": {"files": {".mdoc": 1}, "size_bytes": 27 * 1024,
                                            "unsized": 0},
                                  "frames": {"files": {".eer": 5},
                                             "size_bytes": 5 * 103 * 1024**2, "unsized": 0}}
    # The gain is taken once, and where it comes from is said.
    assert report["check"]["shared"]["gain"]["files"] == {".gain": 1}
    assert report["check"]["shared_from"] == ["data/cvp/01122021/gainref"]
    assert report["check"]["size_bytes"] == 2 * (27 * 1024 + 5 * 103 * 1024**2) + 34 * 1024**2


def test_a_format_the_entry_does_not_declare_is_warned(tmp_path):
    ebi = FakeEbi()
    layout = tmp_path / "11830.yaml"
    tomo_11830(ebi, layout)
    ebi.pages[API.format(id="11830")] = ebi.pages[API.format(id="11830")].replace(
        '"EER"', '"TIFF"')
    report = run(ebi, "11830", "--layout", str(layout), "--series", "01122021_pos3")

    assert report["check"]["warnings"] == [
        "the entry declares TIFF for data/cvp, and the layout takes .eer files from it"]


def test_a_stack_per_series_is_left_to_the_folder_check(tmp_path):
    ebi = FakeEbi()
    base = BASE.format(id="11538")
    ebi.pages[API.format(id="11538")] = api("Ribosomes", [])
    ebi.pages[f"{base}mdoc_files/"] = listing("11538", [("TS_0002.mrc.mdoc", " 49K")])
    ebi.pages[f"{base}mdoc_files/TS_0002.mrc.mdoc"] = mdoc(
        [f"TS_0002_{z:03d}.tif" for z in range(5)])
    ebi.pages[f"{base}Untreated_dataset/"] = listing("11538", [("TS_0002.mrc", "2.0G")])
    layout = tmp_path / "11538.yaml"
    layout.write_text("groups:\n"
                      "  - remote_dir: mdoc_files\n    file_pattern: '{tilt_series}.mrc.mdoc'\n"
                      "  - remote_dir: Untreated_dataset\n    file_pattern: '{tilt_series}.mrc'\n")
    report = run(ebi, "11538", "--layout", str(layout), "--series", "TS_0002")

    (row,) = report["check"]["series"]
    assert row["verdict"] == "accepted"
    assert "assembled stack" in row["mdoc"]["note"]
    assert report["exit"] == 0


@pytest.mark.parametrize("argv", [
    ["EMPIAR-10164"],
    ["10164", "frames", "--series", "TS_01"],
    ["11538", "--layout", "{layout}"],
    ["11538", "--series", "TS_0002"],
    ["11538", "--layout", "{missing}", "--series", "TS_0002"],
    ["10164", "--series", "TS_01", "--series", "TS_01"],
])
def test_contradicting_arguments_stop_with_status_2(tmp_path, argv):
    layout = tmp_path / "x.yaml"
    layout.write_text("groups: []\n")
    filled = [arg.format(layout=layout, missing=tmp_path / "missing.yaml") for arg in argv]
    ebi = FakeEbi()
    assert show_empiar.main(filled, reader=show_empiar.Reader(read=ebi, retries=0)) == 2
    assert ebi.asked == []


def test_nothing_is_written_and_the_text_says_what_the_json_does(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ebi = FakeEbi()
    tomo_10164(ebi)
    reader = show_empiar.Reader(read=ebi, retries=0)
    assert show_empiar.main(["10164", "--series", "TS_01", "--json"], reader=reader) == 0
    report = json.loads(capsys.readouterr().out)
    assert show_empiar.main(["10164", "--series", "TS_01"], reader=reader) == 0
    text = capsys.readouterr().out

    assert list(tmp_path.iterdir()) == []
    assert report["command"] in text
    assert "TS_01: data/mdoc-files/: 1 .mrc.mdoc; data/frames/: 5 .mrc" in text
    assert "mdoc TS_01.mrc.mdoc: 5 tilts; the layout takes 5 frames" in text
