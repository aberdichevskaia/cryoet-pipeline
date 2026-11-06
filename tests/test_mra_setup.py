"""The MRA setup without Dynamo: draws, mask and rounds as Dynamo's, and a plan the port runs.

The numbers below are Dynamo's own, taken on 2026-11-02 in Dynamo 1.1.511's runtime
on the Mac: MATLAB's draws after `rng(20261016)`, `dynamo_sphere`'s maps, and the
rounds scripts/dynamo_mra_setup.m printed for projects of the CZII export (box 48,
10.012 A/px), which scripts/compare_mra_setup.py then matched card for card.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_dynamo_mra_project import BOX, _project

from cryoet_pipeline.backends.dynamo_search import SearchSettings, average_class
from cryoet_pipeline.dynamo_mra_project import _particle, run_mra_project
from cryoet_pipeline.mra_setup import plan_from_export, search_rounds, seed_draws, soft_sphere
from cryoet_pipeline.reports.dynamo_project import read_dynamo_classification, read_table

PIXEL = 10.0


def test_the_order_and_orientations_are_matlabs() -> None:
    # randperm(12), then rand(1, 3): the order sorts the first 12 draws, the
    # orientations start at the 13th.
    order, orientations, noise = seed_draws(12, 4, 2)
    assert list(order + 1) == [10, 5, 12, 11, 9, 1, 3, 8, 6, 2, 4, 7]
    np.testing.assert_allclose(orientations[:3, 0] / 360,
                               [0.61113715572241423, 0.4786219387760593, 0.76454941792238618],
                               rtol=1e-15)
    assert noise.shape == (3, 4, 4, 4)
    # randperm(2384), as RyR1's export, and the orientations after it.
    order, orientations, _ = seed_draws(2384, 2, 3)
    assert list(order[:8] + 1) == [1945, 2068, 1011, 1474, 1807, 1323, 1785, 1570]
    assert int(np.sum((order + 1) * np.arange(1, 2385))) == 3383952221
    np.testing.assert_allclose(orientations[[0, 1, -1]], [
        [290.08473076052212, 128.83835719736732, 163.40743978360305],
        [117.46632860550696, 119.82477280532916, 130.8274678883964],
        [270.71559595647221, 52.828878051679531, 187.29936384234671]], rtol=1e-14)


# dynamo_sphere(radius, box, (box + 1) / 2 * [1 1 1], edge): the sum and the count of
# nonzero voxels, and the line along x through the voxel past the centre.
SPHERES = [
    (5, 16, 2, 1304.47557914, 2008, [0.201019973, 0.553425014, 0.928365886] + [1] * 10
     + [0.928365886, 0.553425014, 0.201019973]),
    (4.5, 13, 3, 1391.06551695, 2093, [0.778800786, 0.972604454] + [1] * 9
     + [0.972604454, 0.778800786]),
    (6, 16, 0, 912, 912, [0, 0] + [1] * 12 + [0, 0]),
    (16, 52, 2, 23348.082557, 28120, [0] * 7 + [0.206092253, 0.563684344, 0.935809374]
     + [1] * 32 + [0.935809374, 0.563684344, 0.206092253] + [0] * 7),
    (7.3, 21, 2.5, 3621.65571934, 5449, [0.311486274, 0.629770398, 0.924594522] + [1] * 15
     + [0.924594522, 0.629770398, 0.311486274]),
]


@pytest.mark.parametrize("radius, box, edge, total, nonzero, line", SPHERES)
def test_the_mask_is_dynamos_sphere(radius, box, edge, total, nonzero, line) -> None:
    sphere = soft_sphere(radius, box, edge)
    assert sphere.dtype == np.float32 and sphere.shape == (box,) * 3
    assert float(sphere.astype(float).sum()) == pytest.approx(total, rel=1e-11)
    assert np.count_nonzero(sphere) == nonzero
    middle = box // 2
    np.testing.assert_allclose(sphere[middle, middle, :], line, atol=5e-10)


# What scripts/dynamo_mra_setup.m printed for these variables on the CZII export.
CZII = (48, 10.0120000839)
ROUNDS = [
    ({"schedule": "coarse_to_fine", "global_rounds": 2}, [
        "round 1: 1 iteration(s), low-pass 12 px (40.0 A), every orientation, 360 deg in 7.9 deg "
        "steps refined 2 times by 2 to 2.0 deg, particles at 28 px of 48",
        "round 2: 1 iteration(s), low-pass 14 px (34.3 A), every orientation, 360 deg in 6.8 deg "
        "steps refined 2 times by 2 to 1.7 deg, particles at 32 px of 48",
        "round 3: 1 iteration(s), low-pass 16 px (30.0 A), "
        "cone and in-plane 17.7 deg in 5.9 deg steps",
        "round 4: 1 iteration(s), low-pass 18 px (26.7 A), "
        "cone and in-plane 15.9 deg in 5.3 deg steps",
        "round 5: 1 iteration(s), low-pass 22 px (21.8 A), "
        "cone and in-plane 12.9 deg in 4.3 deg steps",
    ]),
    ({"schedule": "coarse_to_fine", "lowpass_steps": "fourier", "first_lowpass": 80,
      "cone_steps": [5, 5, 3, 3, 3]}, [
        "round 1: 1 iteration(s), low-pass 6 px (80.1 A), "
        "cone and in-plane 77.0 deg in 15.4 deg steps",
        "round 2: 1 iteration(s), low-pass 10 px (48.1 A), "
        "cone and in-plane 47.0 deg in 9.4 deg steps",
        "round 3: 1 iteration(s), low-pass 14 px (34.3 A), "
        "cone and in-plane 20.4 deg in 6.8 deg steps",
        "round 4: 1 iteration(s), low-pass 18 px (26.7 A), "
        "cone and in-plane 15.9 deg in 5.3 deg steps",
        "round 5: 1 iteration(s), low-pass 22 px (21.8 A), "
        "cone and in-plane 12.9 deg in 4.3 deg steps",
    ]),
    ({"schedule": "fixed"}, [
        "round 1: 5 iteration(s), low-pass 16 px (30.0 A), "
        "cone and in-plane 15.0 deg in 5.0 deg steps",
    ]),
    ({"schedule": "coarse_to_fine", "global_rounds": 1, "global_step": 45}, [
        "round 1: 1 iteration(s), low-pass 12 px (40.0 A), every orientation, 360 deg in 45.0 deg "
        "steps refined 3 times by 2 to 5.6 deg, particles at 28 px of 48",
        "round 2: 1 iteration(s), low-pass 14 px (34.3 A), "
        "cone and in-plane 20.4 deg in 6.8 deg steps",
    ]),
    ({"schedule": "coarse_to_fine", "global_rounds": 1, "global_full_box": True,
      "diameter": 300}, [
        "round 1: 1 iteration(s), low-pass 12 px (40.0 A), every orientation, 360 deg in 7.6 deg "
        "steps refined 2 times by 2 to 1.9 deg, particles at 48 px of 48",
        "round 2: 1 iteration(s), low-pass 14 px (34.3 A), "
        "cone and in-plane 19.5 deg in 6.5 deg steps",
    ]),
]


@pytest.mark.parametrize("variables, printed", ROUNDS)
def test_the_rounds_are_the_ones_dynamos_setup_printed(variables, printed) -> None:
    rounds = search_rounds(*CZII, **variables)
    described = [r.describe(number, *CZII) for number, r in enumerate(rounds, 1)]
    assert described[:len(printed)] == printed


def test_ryr1_bin8_searches_as_its_dynamo_projects_did() -> None:
    # g2atan's variables: global rounds at 9.0 and 7.8 deg on 30 and 34 px of the 52
    # px box, as Dynamo's setup made them; three 4.9 deg steps are 14.7 deg exactly,
    # as a card held them.
    rounds = search_rounds(52, 13.44, schedule="coarse_to_fine", first_lowpass=53.76,
                           last_lowpass=29.4, diameter=340, global_rounds=2)
    assert [(r.step, r.dim, r.cone_range) for r in rounds[:2]] == [(9.0, 30, 360), (7.8, 34, 360)]
    assert rounds[-1].step == 4.9 and rounds[-1].cone_range == 14.7
    assert rounds[0].settings(52) == SearchSettings(
        cone_aperture=360, cone_sampling=9.0, inplane_range=360, inplane_sampling=9.0, refine=2,
        refine_factor=2, high=2, low=13, shift_limit=3, area_search_modus=1, dim=30)
    assert rounds[2].settings(52).dim == 0


def _export(root: Path) -> Path:
    """Twelve particles (two phantoms) and their initial.tbl, as scripts/export_dynamo.py
    lays an export out: 42 columns, the pixel size in column 36."""

    _project(root)
    table = read_table(root / "mra_t" / "results" / "ite_0001" / "starting_values"
                       / "starting_table_ref_001_ite_0001.tbl")
    table = np.pad(table, ((0, 0), (0, 42 - table.shape[1])))
    table[:, 3:6] = np.random.default_rng(7).uniform(-0.6, 0.6, size=(len(table), 3))
    table[:, 6:9] += 1 / 3  # digits a five-digit table would lose
    table[:, 33], table[:, 35] = 1, PIXEL
    (root / "initial.tbl").write_text(
        "\n".join(" ".join(f"{value:.12g}" for value in row) for row in table) + "\n")
    return root


# A search the 16 px box holds: the fixed card's band reaches 16 Fourier px, twice
# its Nyquist.
SCHEDULE = {"schedule": "coarse_to_fine", "first_lowpass": 50.0, "last_lowpass": 22.0}


def _plan(export: Path, **changes):
    options = {"seeds": ["subset", "blob", "noise"], "subset": 4, "mask_radius": 5,
               "log": lambda _: None, **SCHEDULE, **changes}
    return plan_from_export(export, **options)


def test_the_plan_is_made_from_the_export_as_dynamos_setup_made_it(tmp_path: Path) -> None:
    export = _export(tmp_path)
    plan = _plan(export)
    exported = read_table(export / "initial.tbl")
    order, orientations, noise = seed_draws(12, BOX, 3)

    # The starting table keeps the export's digits.
    np.testing.assert_array_equal(plan.table, exported)
    np.testing.assert_array_equal(plan.soft_mask, soft_sphere(5, BOX, 2).astype(float))
    # Channel 1 averages the first 4 of the order at their poses, channel 2 the next 4
    # at random orientations; channel 3 is the setup's noise.
    for channel, angles in ((1, exported[:, 6:9]), (2, orientations)):
        members = np.sort(order[(channel - 1) * 4:channel * 4])
        expected = average_class(
            [_particle(export / "particles", int(tag)) for tag in exported[members, 0]],
            angles[members], exported[members, 3:6],
            [tuple(row) for row in exported[members, 13:15]], plan.soft_mask, fmin=1).average
        np.testing.assert_array_equal(plan.references[channel],
                                      expected.astype(np.float32).astype(float))
    np.testing.assert_array_equal(plan.references[3], noise[2].astype(np.float32).astype(float))
    groups = dict(plan.seed_groups.tolist())
    assert sorted(tag for tag, channel in groups.items() if channel == 1) == sorted(
        int(exported[index, 0]) for index in order[:4])
    assert plan.reference_notes == ["1 subset 4", "2 blob 4", "3 noise 0"]
    rounds = search_rounds(BOX, PIXEL, **SCHEDULE)
    assert [r.low for r in rounds] == [3, 4, 4, 6, 7]
    assert plan.settings == [r.settings(BOX) for r in rounds]
    assert plan.mra == [True] * 5
    assert plan.setup["variables"]["seeds"] == ["subset", "blob", "noise"]
    assert plan.setup["variables"]["mask_radius_px"] == 5.0
    assert plan.setup["files"]["export"] == str(export.resolve())


def test_the_run_keeps_its_setup_and_carries_on_from_the_plan_made_again(tmp_path: Path) -> None:
    export = _export(tmp_path)
    plan = replace(_plan(export), settings=_plan(export).settings[:1])
    out = export / "mra_x_port"
    run_mra_project(plan, out, log=lambda _: None)
    assert '"schedule": "coarse_to_fine"' in (out / "settings" / "setup.json").read_text()
    manifest = (export / "mra_x_port_seeds" / "seed_manifest.txt").read_text().splitlines()
    assert len(manifest) == 12 and sum(line.split()[1] != "0" for line in manifest) == 8
    run = read_dynamo_classification(out, compute_initial_average=False)
    assert run.reference_notes == {1: "average of 4 random particles at their starting poses",
                                   2: "average of 4 random particles at random orientations",
                                   3: "white noise"}

    # Submitted again, the plan is made again and must come out as the run's.
    lines: list[str] = []
    again = replace(_plan(export), settings=_plan(export).settings[:1])
    run_mra_project(again, out, log=lines.append, resume=True)
    assert lines == ["mra_x_port: all 1 iterations finished before; nothing to compute"]
    other = replace(_plan(export, mask_radius=4), settings=_plan(export).settings[:1])
    with pytest.raises(ValueError, match=r"another plan .*\(seeds; mask\)"):
        run_mra_project(other, out, log=lambda _: None, resume=True)


@pytest.mark.parametrize("changes, refusal", [
    ({"seeds": ["subset"]}, "at least two"),
    ({"seeds": ["map"]}, "reference_file"),
    ({"subset": 7}, "12 particles cannot give 2 disjoint subsets of 7"),
    ({"mask_radius": 7}, "does not fit the 16 px box"),
    ({"global_full_box": True}, "give global_rounds too"),
    ({"schedule": "fixed"}, r"16 Fourier px, is not between .* Nyquist \(8 px\)"),
    ({"first_lowpass": 80, "last_lowpass": 40}, "2 Fourier px, is not between the high-pass"),
    ({"last_lowpass": 15}, r"Nyquist \(20.0 A\)"),
    ({"cone_steps": [3, 3]}, "one per round"),
])
def test_what_dynamos_setup_refused_or_the_port_does_not_model_is_refused(
    tmp_path: Path, changes, refusal,
) -> None:
    export = _export(tmp_path)
    with pytest.raises(ValueError, match=refusal):
        _plan(export, **changes)


def test_a_global_round_at_a_dim_that_divides_the_box_is_refused() -> None:
    # Low-pass 10 px of a 48 px box: dim 24, which Dynamo would reach by binning.
    with pytest.raises(ValueError, match="round 1 would search at dim 24 of the 48 px box"):
        search_rounds(48, 10.012, schedule="coarse_to_fine", first_lowpass=48, global_rounds=1)


def test_a_table_with_another_wedge_type_is_refused(tmp_path: Path) -> None:
    export = _export(tmp_path)
    table = read_table(export / "initial.tbl")
    table[3, 12] = 3
    (export / "initial.tbl").write_text(
        "\n".join(" ".join(f"{value:.12g}" for value in row) for row in table) + "\n")
    with pytest.raises(ValueError, match="ftype 1"):
        _plan(export)


def test_the_shift_limit_reaches_every_round_and_only_the_search(tmp_path: Path) -> None:
    export = _export(tmp_path)
    usual, narrow = _plan(export), _plan(export, shift_limit=2)
    assert {settings.shift_limit for settings in usual.settings} == {3.0}
    assert {settings.shift_limit for settings in narrow.settings} == {2.0}
    assert {settings.area_search_modus for settings in narrow.settings} == {1}
    assert {card["area_search"] for card in narrow.cards} == {"2 2 2"}
    # The setup's results do not depend on it: its variables, seeds and mask stay.
    assert narrow.setup["variables"] == usual.setup["variables"]
    for channel, reference in usual.references.items():
        np.testing.assert_array_equal(narrow.references[channel], reference)
    for wrong in (0, 8):
        with pytest.raises(ValueError, match="shift_limit must be above 0 and below half"):
            _plan(export, shift_limit=wrong)
