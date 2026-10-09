"""Every lane config renders commands the CLI accepts: known flags, required ones present."""

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


@pytest.mark.parametrize("lane", LANES, ids=[path.stem for path in LANES])
def test_every_step_of_the_lane_is_a_valid_command(lane):
    group = typer.main.get_command(cli.app)
    config = tomllib.loads(lane.read_text())
    steps = run_series.build_steps(config, Path("outputs/project"), "TS_X")
    assert steps, lane
    for name, command in steps:
        sub = group.commands.get(command[0])
        assert sub is not None, f"{lane.stem}/{name}: no command {command[0]!r}"
        known, required = _options(sub)
        flags = {part for part in command[1:] if part.startswith("--")}
        assert flags <= known, f"{lane.stem}/{name}: unknown {sorted(flags - known)}"
        assert required <= flags, f"{lane.stem}/{name}: missing {sorted(required - flags)}"


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


def test_an_unknown_chain_is_refused():
    with pytest.raises(SystemExit, match="unknown chain"):
        run_series.build_steps({"lane": {"chain": "beads"}}, Path("p"), "TS_X")
