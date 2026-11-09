"""A lane's MRA (docs/mra_in_lanes.md): the [mra] section's schema, the defaults the
lab's rules give it, what they come to for given particles, and what is refused."""

import sys
import tomllib
from pathlib import Path

import pytest

from cryoet_pipeline.mra_lane import LaneMraError, MraSettings, lane_mra
from cryoet_pipeline.mra_setup import SUBSET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402


def _ryr1(mra: dict | None = None) -> dict:
    cfg = tomllib.loads((ROOT / "configs" / "lanes" / "ryr1-bin8.toml").read_text())
    if mra is not None:
        cfg["mra"] = mra
    return cfg


def test_the_lane_keys_are_the_schemas_fields():
    # run_series.py imports no package, so it lists them itself.
    assert run_series.LANE_KEYS["mra"] == frozenset(MraSettings.model_fields)


def test_the_defaults_give_the_rounds_of_the_run_that_agreed_with_dynamo():
    mra = lane_mra(_ryr1())
    assert mra.name == "bin8-ssn" and mra.label == "bin8" and mra.project == "outputs/ryr1"
    assert mra.seeds == ["subset", "subset", "noise"]
    # RyR1 at bin8: 52 px at 13.44 A. mra_pad_ssn_g2atan's rounds (53.76 -> 29.4 A, 2 global).
    rounds = mra.rounds(52, 13.44)
    assert [one.low for one in rounds] == [13, 15, 17, 20, 24]
    assert [one.dim for one in rounds] == [30, 34, 52, 52, 52]
    assert [one.is_global for one in rounds] == [True, True, False, False, False]
    assert [one.cone_range for one in rounds[2:]] == [20.7, 17.7, 14.7]  # three steps wide
    options = mra.options(13.44)
    assert options["first_lowpass"] == pytest.approx(53.76)
    assert options["last_lowpass"] == pytest.approx(1.1 * 26.88)
    assert (options["diameter"], options["global_rounds"], options["cone_steps"],
            options["antialias"], options["mask_radius"], options["mask_edge"],
            options["shift_limit"], options["subset"]) == (340.0, 2, [3.0], False, None, 2.0,
                                                           3.0, SUBSET)
    # The low-pass follows the particles' Nyquist: at bin4 the global rounds shrink the
    # box as little as at bin8.
    assert [one.dim for one in mra.rounds(102, 6.72)[:2]] == [56, 62]


def test_every_value_says_where_it_came_from():
    mra = lane_mra(_ryr1({"seeds": ["subset", "blob", "noise"], "first_lowpass_angstrom": 60.0}))
    sources = {key: value.source for key, value in mra.values.items()}
    assert sources["seeds"] == sources["first_lowpass"] == "lane"
    assert sources["diameter"] == "[pick] particle_diameter_angstrom"
    assert sources["last_lowpass"].startswith("rule") and sources["name"].startswith("rule")
    assert mra.name == "bin8-sbn"
    assert mra.lowpass_angstrom(13.44) == (60.0, pytest.approx(29.568))
    lines = "\n".join(mra.describe(52, 13.44, 2384))
    assert "60 A -> 1.1 x Nyquist" in lines and "2,384 at 13.44 A/px" in lines
    assert "first: lane; last: rule" in lines


def test_a_fixed_schedule_has_no_low_pass_and_no_global_round():
    mra = lane_mra(_ryr1({"schedule": "fixed"}))
    options = mra.options(13.44)
    assert options["global_rounds"] == 0 and "first_lowpass" not in options
    assert [one.low for one in mra.rounds(52, 13.44)] == [16]


@pytest.mark.parametrize(("mra", "words"), [
    ({"bogus": 1}, "bogus"),
    ({"name": "../x"}, "folder name"),
    ({"seeds": ["subset"]}, "at least two"),
    ({"seeds": ["map"]}, "refines from reference"),
    ({"reference": "a.mrc"}, "only of that"),
    ({"first_lowpass_nyquist": 2, "first_lowpass_angstrom": 50}, "not both"),
    ({"last_lowpass_nyquist": 0.9}, "greater than or equal to 1"),
    ({"first_lowpass_nyquist": 1.1, "last_lowpass_nyquist": 2}, "coarser than the last"),
    ({"schedule": "fixed", "last_lowpass_nyquist": 1.2}, "for 'coarse_to_fine'"),
    ({"global_rounds": 6}, "at most the schedule's rounds"),
    ({"cone_steps": [3, 3]}, "one per round"),
    ({"mask_file": "m.mrc", "mask_radius_px": 10}, "replaces the sphere"),
    ({"antialias": True, "global_rounds": 0}, "needs global_rounds"),
    ({"decoy_label": "bin8"}, "must differ"),
    ({"from_run": "bin8-ssn"}, "go together"),
    ({"from_classes": [2]}, "go together"),
    ({"start_poses": "run"}, "need from_run"),
    ({"from_run": "x", "from_classes": [2, 2]}, "each once"),
    ({"from_run": "x", "from_classes": 2, "decoy_label": "null"}, "leave out decoy_label"),
    ({"from_run": "x", "from_classes": 2, "name": "x"}, "name the new one otherwise"),
])
def test_what_cannot_run_is_refused_with_the_key(mra, words):
    with pytest.raises(LaneMraError, match=r"\[mra\]") as refused:
        lane_mra(_ryr1(mra))
    assert words in str(refused.value)


def test_a_lane_without_a_diameter_is_refused_unless_mra_gives_one():
    cfg = _ryr1()
    del cfg["pick"]["particle_diameter_angstrom"]
    with pytest.raises(LaneMraError, match="particle_diameter_angstrom"):
        lane_mra(cfg)
    assert lane_mra({**cfg, "mra": {"diameter_angstrom": 300}}).options(13.44)["diameter"] == 300


def test_a_class_run_is_named_after_its_run_and_classes():
    mra = lane_mra(_ryr1({"from_run": "bin8-ssn", "from_classes": [1, 3], "global_rounds": 0}))
    assert mra.name == "bin8-ssn-c1_3-ssn"
    assert (mra.from_run, mra.from_classes, mra.from_iteration, mra.start_poses) == (
        "bin8-ssn", [1, 3], None, "run")
    assert lane_mra(_ryr1({"from_run": "r", "from_classes": 2})).from_classes == [2]
    lines = "\n".join(lane_mra(_ryr1({"from_run": "r", "from_classes": 2})).describe())
    assert "class 2 of run r, its last iteration" in lines and "as run r read them" in lines


def test_the_setup_problems_are_found_without_the_particles():
    mra = lane_mra(_ryr1())
    assert mra.setup_problems(52, 13.44, 2384) == []
    # Two subset seeds of 300 need 600 particles.
    assert any("cannot give 2 disjoint subsets of 300" in one
               for one in mra.setup_problems(52, 13.44, 599))
    assert not lane_mra(_ryr1({"subset_particles": 200})).setup_problems(52, 13.44, 599)
    # 340 A at 13.44 A/px is a 16 px radius with the rule's 3 px: not in a 30 px box.
    assert any("does not fit the 30 px box" in one for one in mra.setup_problems(30, 13.44, 2384))
    finer = lane_mra(_ryr1({"last_lowpass_angstrom": 20.0}))
    assert any("no finer than the particles' Nyquist" in one
               for one in finer.setup_problems(52, 13.44, 2384))


def test_the_request_records_the_lane_file_and_every_source(tmp_path):
    text = (ROOT / "configs" / "lanes" / "ryr1-bin8.toml").read_text()
    mra = lane_mra(tomllib.loads(text))
    request = mra.request(Path("configs/lanes/ryr1-bin8.toml"), text, 13.44,
                          ["039", "096"], {"run": "x", "iteration": 5})
    assert request["lane_config"] == "configs/lanes/ryr1-bin8.toml"
    assert len(request["lane_config_sha256"]) == 64 and request["mra_section"] == {}
    assert request["series"] == ["039", "096"] and request["selection"]["iteration"] == 5
    assert request["resolved"]["first_lowpass_resolved_angstrom"] == pytest.approx(53.76)
    assert request["sources"]["seeds"].startswith("rule")
