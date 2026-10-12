"""Every lane config renders commands the CLI accepts: known flags, required ones present."""

import copy
import sys
import tomllib
from pathlib import Path

import pytest
import typer

from cryoet_pipeline import cli
from cryoet_pipeline.backends.picking import parse_volume_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_series  # noqa: E402
import submit_lanes  # noqa: E402

LANES = sorted((ROOT / "configs" / "lanes").glob("*.toml"))


def _options(command) -> tuple[set[str], set[str]]:
    known, required = set(), set()
    for param in command.params:
        # typer bundles its own click, so options are told apart by their kind.
        if param.param_type_name == "option":
            known.update(param.opts)
            known.update(param.secondary_opts)
            if param.required:
                required.add(param.opts[0])
    return known, required


def _assert_valid_commands(config: dict, label: str) -> list[str]:
    """Every rendered step is a CLI command with known flags and its required ones."""
    group = typer.main.get_command(cli.app)
    steps = run_series.build_steps(config, Path("outputs/project"), "TS_X")
    assert steps, label
    for name, command in steps:
        sub = group.commands.get(command[0])
        assert sub is not None, f"{label}/{name}: no command {command[0]!r}"
        known, required = _options(sub)
        flags = {part for part in command[1:] if part.startswith("--")}
        assert flags <= known, f"{label}/{name}: unknown {sorted(flags - known)}"
        assert required <= flags, f"{label}/{name}: missing {sorted(required - flags)}"
    return [name for name, _ in steps]


@pytest.mark.parametrize("lane", LANES, ids=[path.stem for path in LANES])
def test_every_step_of_the_lane_is_a_valid_command(lane):
    _assert_valid_commands(tomllib.loads(lane.read_text()), lane.stem)


@pytest.mark.parametrize("lane", LANES, ids=[path.stem for path in LANES])
def test_a_volume_split_renders_as_a_valid_option_in_either_chain(lane):
    # A lane that sets no split never renders the flag, so the test above cannot
    # see a misspelt one in that lane's chain. Set it in every lane here.
    group = typer.main.get_command(cli.app)
    known, _ = _options(group.commands["pick-particles"])
    config = tomllib.loads(lane.read_text())
    if "volume_split" in config["pick"]:
        # The CLI refuses a malformed value only once the GPU job is running.
        parse_volume_split(config["pick"]["volume_split"])

    config["pick"]["volume_split"] = "3,2,1"
    split = dict(run_series.build_steps(config, Path("outputs/project"), "TS_X"))
    command = split["pick-particles"]
    assert "--pytom-volume-split" in known
    assert command[command.index("--pytom-volume-split") + 1] == "3,2,1"

    # Without the line, the command a lane rendered before the option existed.
    del config["pick"]["volume_split"]
    unsplit = dict(run_series.build_steps(config, Path("outputs/project"), "TS_X"))
    assert "--pytom-volume-split" not in unsplit["pick-particles"]


def test_an_unknown_input_and_the_old_chain_key_are_refused():
    with pytest.raises(SystemExit, match="input must be one of"):
        run_series.build_steps({"lane": {"input": "beads"}}, Path("p"), "TS_X")
    with pytest.raises(SystemExit, match="chain is now"):
        run_series.build_steps({"lane": {"chain": "frames"}}, Path("p"), "TS_X")


def _lane(stem: str) -> dict:
    return tomllib.loads((ROOT / "configs" / "lanes" / f"{stem}.toml").read_text())


def _combination(source: str, alignment: str, segment: bool) -> dict:
    """The Gag lane with its input, fiducials and segmentation swapped for the
    ribosome lane's where the combination asks for them."""
    gag, ribo = _lane("gag-bin8"), _lane("ribo-bin8")
    config = copy.deepcopy(gag)
    if source == "stack":
        config["lane"]["input"] = "stack"
        config["data"] = ribo["data"]
    if alignment == "patches":
        config["fiducials"] = ribo["fiducials"]
        config["reconstruct"] = ribo["reconstruct"]
    if not segment:
        del config["segment"]
        for key in ("use_segmentation_mask", "segmentation_mask_dilation_px"):
            config["pick"].pop(key, None)
    return config


@pytest.mark.parametrize("segment", [True, False], ids=["segment", "no-segment"])
@pytest.mark.parametrize("alignment", ["beads", "patches"])
@pytest.mark.parametrize("source", ["frames", "stack"])
def test_every_combination_renders_valid_commands(source, alignment, segment):
    config = _combination(source, alignment, segment)
    names = _assert_valid_commands(config, f"{source}/{alignment}/{segment}")
    entry = "prepare-tilt-series" if source == "frames" else "import-tilt-stack"
    assert names[0] == entry
    assert ("segment-tomogram" in names) is segment
    steps = dict(run_series.build_steps(config, Path("outputs/project"), "TS_X"))
    seed = steps["generate-fiducial-seed"]
    expected = "imod-autofidseed" if alignment == "beads" else "imod-patch-seed"
    assert seed[seed.index("--backend") + 1] == expected


EXAMPLE = ROOT / "configs" / "lanes" / "example.toml"


def _example_with_block_b() -> dict:
    """The starter config as the instructions in it say to switch to patches:
    block A deleted, one leading "# " removed from every line of block B."""
    text = EXAMPLE.read_text()
    head, rest = text.split("# --- Block A", 1)
    _, rest = rest.split("# --- Block B", 1)
    block_b, tail = rest.split("# --- end of block B", 1)
    lines = block_b.splitlines()[1:]  # the rest of the marker line
    uncommented = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines]
    return tomllib.loads(head + "\n".join(uncommented) + "\n" + tail.split("\n", 1)[1])


def test_the_starter_config_leaves_the_unchecked_choices_off():
    # Each of these is used as given, with no check against the data, and is
    # right only for some datasets: a pixel size that bypasses the check against
    # the stack, and a membrane mask that is wrong for a cytosolic particle.
    config = tomllib.loads(EXAMPLE.read_text())
    assert "pixel_size_angstrom" not in config["motion"]
    assert "raw_pixel_spacing_angstrom" not in config["fiducials"]
    assert "use_segmentation_mask" not in config["pick"]
    assert "segmentation_mask_dilation_px" not in config["pick"]
    # Block A is beads, with the fine alignment that goes with beads.
    assert config["fiducials"]["seed_backend"] == "imod-autofidseed"
    assert config["fine_align"]["surfaces_to_analyze"] == 2


def test_block_b_of_the_starter_config_switches_alignment_as_a_whole():
    config = _example_with_block_b()
    names = _assert_valid_commands(config, "example/block-b")
    assert "generate-fiducial-seed" in names
    fid, fine = config["fiducials"], config["fine_align"]
    assert (fid["seed_backend"], fid["track_backend"]) == ("imod-patch-seed", "imod-patch-track")
    assert fine["surfaces_to_analyze"] == 1
    # Two whole patches, as the comment derives it.
    assert fine["max_pruned_fraction"] == 2 / (fid["num_patches_x"] * fid["num_patches_y"])
    assert "thickness" in config["reconstruct"]
    # Nothing of block A's bead settings survives the switch.
    assert not {"tracking_binning", "fiducial_diameter_nm"} & set(fid)


def test_a_mixed_fiducial_pair_is_refused():
    config = _lane("gag-bin8")
    config["fiducials"]["track_backend"] = "imod-patch-track"
    with pytest.raises(SystemExit, match="do not belong together"):
        run_series.build_steps(config, Path("p"), "TS_X")


def test_patch_tracking_without_a_thickness_is_refused():
    config = _lane("ribo-bin8")
    del config["reconstruct"]["thickness"]
    with pytest.raises(SystemExit, match="thickness must be set"):
        run_series.build_steps(config, Path("p"), "TS_X")


def test_a_segmentation_mask_without_segmentation_is_refused():
    config = _lane("gag-bin8")
    del config["segment"]
    with pytest.raises(SystemExit, match="no \\[segment\\] section"):
        run_series.build_steps(config, Path("p"), "TS_X")


def test_the_template_comes_from_exactly_one_of_emdb_and_a_local_map():
    config = _lane("ribo-bin8")
    del config["template"]["emdb_id"]
    config["template"]["source_map"] = "maps/my_reference.mrc"
    command = dict(run_series.build_steps(config, Path("p"), "TS_X"))["prepare-template"]
    assert command[command.index("--source-map") + 1] == "maps/my_reference.mrc"
    assert "--emdb-id" not in command
    _assert_valid_commands(config, "source-map")

    config["template"]["emdb_id"] = "EMD-16721"
    with pytest.raises(SystemExit, match="exactly one of"):
        run_series.build_steps(config, Path("p"), "TS_X")
    del config["template"]["emdb_id"], config["template"]["source_map"]
    with pytest.raises(SystemExit, match="exactly one of"):
        run_series.build_steps(config, Path("p"), "TS_X")


def test_a_gain_reference_reaches_motion_correction():
    config = _lane("gag-bin8")
    config["motion"].update(gain_reference="data/gain.mrc", gain_rotation=1, gain_flip=2)
    command = dict(run_series.build_steps(config, Path("p"), "TS_X"))["prepare-tilt-series"]
    for flag, value in (("--motioncor3-gain", "data/gain.mrc"),
                        ("--motioncor3-gain-rotation", "1"),
                        ("--motioncor3-gain-flip", "2")):
        assert command[command.index(flag) + 1] == value
    _assert_valid_commands(config, "gain")


def test_a_missing_section_is_named():
    config = _lane("ribo-bin8")
    del config["ctf"]
    with pytest.raises(SystemExit, match="no \\[ctf\\] section"):
        run_series.build_steps(config, Path("p"), "TS_X")


def test_init_runs_only_for_a_new_frames_project():
    gag = _lane("gag-bin8")
    new = submit_lanes.Plan(Path("lane.toml"), "outputs/__no_such_project__", ("TS_01",), ())
    command = submit_lanes.init_command(gag, new)
    assert command[:2] == ["cryoet", "init"]
    assert command[command.index("--tilt-series") + 1] == "TS_01"
    assert submit_lanes.init_command(_lane("ribo-bin8"), new) is None


@pytest.mark.parametrize("lane", LANES, ids=[path.stem for path in LANES])
def test_the_run_stages_cover_the_chain_in_order(lane):
    # Stages are slices of the fixed chain. Covering it whole is not required by
    # the submitter, but a lane config that skips a step would be a mistake.
    config = tomllib.loads(lane.read_text())
    plan = submit_lanes.plan(config, lane)
    chain = [name for name, _ in run_series.build_steps(config, Path(plan.project), "TS_X")]
    covered = [
        name
        for stage in plan.stages
        for name, _ in run_series.slice_steps(
            run_series.build_steps(config, Path(plan.project), "TS_X"), stage.start, stage.end
        )
    ]
    assert covered == chain


def _two_stage_lane() -> dict:
    config = tomllib.loads((ROOT / "configs" / "lanes" / "ribo-bin8.toml").read_text())
    assert [stage["name"] for stage in config["run"]["stage"]] == ["cpu", "gpu"]
    return config


def test_a_gap_between_stages_is_refused():
    config = _two_stage_lane()
    config["run"]["stage"][1]["from"] = "pick-particles"
    with pytest.raises(SystemExit, match="skips prepare-template"):
        submit_lanes.plan(config, Path("lane.toml"))


def test_an_overlap_between_stages_is_refused():
    config = _two_stage_lane()
    config["run"]["stage"][1]["from"] = "reconstruct-tomogram"
    with pytest.raises(SystemExit, match="repeats reconstruct-tomogram"):
        submit_lanes.plan(config, Path("lane.toml"))


def test_a_stage_that_runs_backwards_or_names_no_step_is_refused():
    config = _two_stage_lane()
    config["run"]["stage"][0]["until"] = "import-tilt-stack"
    config["run"]["stage"][0]["from"] = "reconstruct-tomogram"
    with pytest.raises(SystemExit, match="comes after"):
        submit_lanes.plan(config, Path("lane.toml"))

    config = _two_stage_lane()
    config["run"]["stage"][0]["until"] = "reconstruct"
    with pytest.raises(SystemExit, match="unknown step 'reconstruct'"):
        submit_lanes.plan(config, Path("lane.toml"))


def test_the_environment_a_stage_needs_comes_from_its_commands():
    config = _two_stage_lane()
    plan = submit_lanes.plan(config, Path("lane.toml"))
    env = {stage.name: stage.env for stage in plan.stages}
    assert env == {"cpu": frozenset(), "gpu": frozenset({"PYTOMO_MP_DIR"})}
