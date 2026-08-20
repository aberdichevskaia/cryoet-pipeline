"""Two binning lanes through the real CLI, from stack rebuild to tomogram.

Every stage already has its own tests, and every one of them passed while the
seams between stages were broken: provenance measured against direct parents, a
replaced entry pinned to its old position, a variant label filtering the very
inputs a lane parts from, CTF outputs sharing a filename between lanes. Those
only appear when a second lane is run through the whole chain, so that is what
this does — with the external programs faked, but the lookups, the registry, the
ids and the output paths all real.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from cryoet_pipeline import cli
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.ctf import (
    Ctffind5CtfEstimationBackend,
    ImodCtfPhaseFlipBackend,
)
from cryoet_pipeline.backends.final_stack import ImodFinalAlignedStackBackend
from cryoet_pipeline.backends.reconstruction import ImodTiltReconstructionBackend
from cryoet_pipeline.models import (
    AlignmentTransform,
    Artifact,
    ArtifactKind,
    AxisOrder,
    TiltAlignment,
    TiltImage,
    TiltSeriesManifest,
)

_TILT_ANGLES = [-3.1, 0.1]
_SOURCE_SHAPE = (2, 32, 24)


def test_two_binning_lanes_reach_a_tomogram_without_colliding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _seed_project(tmp_path)
    _install_fake_externals(monkeypatch, tmp_path)
    run = _command_runner_for(paths, tmp_path)

    for lane in ([], ["--variant", "bin8"]):
        binning = "4" if not lane else "8"
        run("build-final-aligned-stack", "--output-binning", binning, *lane)
        run(
            "estimate-ctf",
            "--ctf-voltage-kv",
            "300",
            "--ctf-spherical-aberration-mm",
            "2.7",
            "--ctf-expected-defocus-um",
            "3.0",
            *lane,
        )
        run("correct-ctf", *lane)
        run("reconstruct-tomogram", *lane)

    registry = ArtifactRegistry.load(paths["registry"])
    ids = {artifact.id for artifact in registry.artifacts}

    for canonical in (
        "TS_TEST:aligned_tilt_stack:final",
        "TS_TEST:ctf_model:ctffind5",
        "TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip",
        "TS_TEST:tomogram:fine",
    ):
        assert canonical in ids, f"the unlabelled lane lost {canonical}"
        assert f"{canonical}:bin8" in ids, f"the bin8 lane never produced {canonical}"

    # Two lanes writing one file is the failure that leaves both registry
    # entries pointing at whichever ran last.
    written = [artifact.path for artifact in registry.artifacts]
    assert len(written) == len(set(written)), "two artifacts share an output path"

    # The bin8 tomogram must descend from the bin8 stack, not from the other lane.
    assert registry.descends_from(
        "TS_TEST:tomogram:fine:bin8", "TS_TEST:aligned_tilt_stack:final:bin8"
    )
    assert not registry.descends_from(
        "TS_TEST:tomogram:fine:bin8", "TS_TEST:aligned_tilt_stack:final"
    )
    assert registry.descends_from(
        "TS_TEST:tomogram:fine:bin8",
        "TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip:bin8",
    ), "reconstruction must prefer this lane's CTF-corrected stack"

    binned = registry.get("TS_TEST:aligned_tilt_stack:final:bin8")
    assert binned.binning == 8
    assert binned.shape == (2, 4, 3)

    # The label is data, not only a suffix on the id: a relational schema will
    # want it as a column, and the lookup already reads the field rather than
    # parsing strings.
    for canonical in (
        "TS_TEST:aligned_tilt_stack:final",
        "TS_TEST:ctf_model:ctffind5",
        "TS_TEST:tomogram:fine",
    ):
        assert registry.get(canonical).variant is None
        assert registry.get(f"{canonical}:bin8").variant == "bin8"
    assert registry.get("TS_TEST:ctf_model:ctffind5").backend == "ctffind5"
    assert registry.get("TS_TEST:tomogram:fine").backend == "imod-tilt"


def test_a_lane_can_reuse_the_other_lanes_ctf_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fitting once at the finest binning is the point of allowing the transfer.

    The bin8 lane runs no `estimate-ctf` of its own, so the lookup falls back to
    the shared unlabelled model and correction applies it to the coarser stack.
    """

    paths = _seed_project(tmp_path)
    _install_fake_externals(monkeypatch, tmp_path)
    run = _command_runner_for(paths, tmp_path)

    run("build-final-aligned-stack", "--output-binning", "4")
    run(
        "estimate-ctf",
        "--ctf-voltage-kv",
        "300",
        "--ctf-spherical-aberration-mm",
        "2.7",
        "--ctf-expected-defocus-um",
        "3.0",
    )
    run("build-final-aligned-stack", "--output-binning", "8", "--variant", "bin8")
    run("correct-ctf", "--variant", "bin8")

    registry = ArtifactRegistry.load(paths["registry"])
    corrected = registry.get(
        "TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip:bin8"
    )
    assert corrected.parameters["ctf_model_id"] == "TS_TEST:ctf_model:ctffind5"
    assert corrected.binning == 8


def test_reconstruction_can_be_redone_after_ctf_appears_upstream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reconstruct, then add CTF, then reconstruct again — the usual order of work.

    The second reconstruction gains a parent that was registered after it, which
    is what a registry ordered parents-before-children has to accommodate: the
    entry moves rather than being refused for sitting where it used to belong.
    """

    paths = _seed_project(tmp_path)
    _install_fake_externals(monkeypatch, tmp_path)
    run = _command_runner_for(paths, tmp_path)

    run("build-final-aligned-stack", "--output-binning", "4")
    run("reconstruct-tomogram")
    before = ArtifactRegistry.load(paths["registry"]).get("TS_TEST:tomogram:fine")
    assert before.parameters["ctf_corrected"] is False

    run(
        "estimate-ctf",
        "--ctf-voltage-kv",
        "300",
        "--ctf-spherical-aberration-mm",
        "2.7",
        "--ctf-expected-defocus-um",
        "3.0",
    )
    run("correct-ctf")
    run("reconstruct-tomogram", "--overwrite")

    registry = ArtifactRegistry.load(paths["registry"])
    after = registry.get("TS_TEST:tomogram:fine")
    assert after.parameters["ctf_corrected"] is True
    assert registry.descends_from(
        after.id, "TS_TEST:aligned_tilt_stack:ctf_corrected:imod-ctfphaseflip"
    )
    # Loading is what enforces parents-before-children, so a stale position
    # would surface here rather than at the moment of replacement.
    ArtifactRegistry.model_validate(registry.model_dump())


def _command_runner_for(paths: dict[str, Path], tmp_path: Path):
    """Invoke one CLI command against this project, failing loudly on a bad exit."""

    runner = CliRunner()

    def run(command: str, *arguments: str) -> None:
        result = runner.invoke(
            cli.app,
            [
                command,
                "--manifest",
                str(paths["manifest"]),
                "--registry",
                str(paths["registry"]),
                "--out",
                str(tmp_path / "outputs"),
                *arguments,
            ],
        )
        assert result.exit_code == 0, f"{command} failed:\n{result.output}\n{result.exception}"

    return run


def _seed_project(tmp_path: Path) -> dict[str, Path]:
    """Write the manifest, the canonical stack, and a fine alignment."""

    manifest = TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=tmp_path / "TS_TEST.mdoc",
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=z_value,
                tilt_angle_deg=angle,
                subframe_path=f"frame_{z_value}.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            )
            for z_value, angle in enumerate(_TILT_ANGLES)
        ],
    )
    manifest_path = tmp_path / "TS_TEST.json"
    manifest_path.write_text(manifest.model_dump_json())

    stack_path = tmp_path / "stack.zarr"
    stack = zarr.open(
        stack_path, mode="w", shape=_SOURCE_SHAPE, chunks=_SOURCE_SHAPE, dtype=np.float32
    )
    stack[:] = np.arange(int(np.prod(_SOURCE_SHAPE)), dtype=np.float32).reshape(_SOURCE_SHAPE)

    alignment_result = TiltAlignment(
        tilt_series_id="TS_TEST",
        backend="test",
        stage="fine",
        input_stack_id="TS_TEST:tilt_stack",
        input_binning=1,
        tilt_axis_angle_deg=85.3,
        transform_semantics="global",
        transforms=[
            AlignmentTransform(
                z_value=z_value,
                tilt_angle_deg=angle,
                a11=1.0,
                a12=0.0,
                a21=0.0,
                a22=1.0,
                shift_x_px=0.0,
                shift_y_px=0.0,
            )
            for z_value, angle in enumerate(_TILT_ANGLES)
        ],
    )
    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(alignment_result.model_dump_json())

    registry = ArtifactRegistry.empty()
    registry.extend(
        [
            Artifact(
                id="TS_TEST:tilt_stack",
                kind=ArtifactKind.TILT_STACK,
                path=stack_path,
                shape=_SOURCE_SHAPE,
                dtype="float32",
                axis_order=AxisOrder.TYX,
                pixel_spacing_angstrom=1.35,
                binning=1,
                parameters={"tilt_series_id": "TS_TEST"},
            ),
            Artifact(
                id="TS_TEST:alignment:fine",
                kind=ArtifactKind.ALIGNMENT,
                path=alignment_path,
                parent_ids=["TS_TEST:tilt_stack"],
                parameters={
                    "tilt_series_id": "TS_TEST",
                    "stage": "fine",
                    "raw_pixel_spacing_angstrom": 1.35,
                    "positioning_source": "tiltalign_two_surface_analysis",
                    "recommended_x_axis_tilt_deg": 0.0,
                    "recommended_unbinned_thickness_px": 16.0,
                    "recommended_unbinned_z_shift_px": 0.0,
                    "axis_z_shift_applied_in_alignment": True,
                },
            ),
        ]
    )
    registry_path = tmp_path / "artifacts.json"
    registry.write(registry_path)
    return {"manifest": manifest_path, "registry": registry_path}


def _install_fake_externals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every stage at real backends driven by one fake program runner."""

    imod_dir = tmp_path / "imod"
    (imod_dir / "bin").mkdir(parents=True, exist_ok=True)
    for program in ("newstack", "ctfphaseflip", "tilt"):
        executable = imod_dir / "bin" / program
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
    ctffind = tmp_path / "ctffind"
    ctffind.write_text("#!/bin/sh\n")
    ctffind.chmod(0o755)

    runner = _fake_runner()
    monkeypatch.setenv("IMOD_DIR", str(imod_dir))
    monkeypatch.setattr(
        cli, "_final_aligned_stack_backend", lambda name: ImodFinalAlignedStackBackend(runner)
    )
    monkeypatch.setattr(
        cli, "_ctf_estimation_backend", lambda name: Ctffind5CtfEstimationBackend(runner)
    )
    monkeypatch.setattr(
        cli, "_ctf_correction_backend", lambda name: ImodCtfPhaseFlipBackend(runner)
    )
    monkeypatch.setattr(
        cli, "_reconstruction_backend", lambda name: ImodTiltReconstructionBackend(runner)
    )
    monkeypatch.setattr(cli, "_resolve_ctffind_executable", lambda context: ctffind, raising=False)
    monkeypatch.setattr(
        "cryoet_pipeline.backends.ctf._resolve_ctffind_executable", lambda context: ctffind
    )


def _fake_runner():
    """Stand in for newstack, CTFFIND, ctfphaseflip and tilt.

    Each writes the file the stage goes on to validate, at the shape that stage
    expects, so shape and view-count checks are exercised rather than bypassed.
    """

    def runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, env
        arguments = list(command)
        program = Path(arguments[0]).name

        if program == "newstack":
            _fake_newstack(arguments)
        elif program == "ctffind":
            assert stdin is not None, "CTFFIND is driven through stdin"
            _fake_ctffind(stdin)
        elif program == "ctfphaseflip":
            _fake_ctfphaseflip(arguments)
        elif program == "tilt":
            _fake_tilt(arguments)
        else:  # pragma: no cover - a new external would need its own fake
            raise AssertionError(f"unexpected external program: {program}")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return runner


def _fake_newstack(arguments: list[str]) -> None:
    if "-secs" in arguments:
        # One view extracted for the CTF hint pass.
        source = Path(arguments[-2])
        with mrcfile.open(source, permissive=True) as stack:
            view = np.asarray(stack.data)[int(arguments[arguments.index("-secs") + 1])]
        with mrcfile.new(Path(arguments[-1]), overwrite=True) as output:
            output.set_data(np.ascontiguousarray(view, dtype=np.float32))
        return

    source = Path(arguments[arguments.index("-input") + 1])
    size = arguments[arguments.index("-size") + 1]
    output_x, output_y = (int(value) for value in size.split(","))
    with mrcfile.open(source, permissive=True) as stack:
        views = int(np.asarray(stack.data).shape[0])
        spacing = float(stack.voxel_size.x)
    with mrcfile.new(Path(arguments[arguments.index("-output") + 1]), overwrite=True) as output:
        output.set_data(np.zeros((views, output_y, output_x), dtype=np.float32))
        output.voxel_size = spacing


def _fake_ctffind(stdin: str) -> None:
    answers = stdin.splitlines()
    with mrcfile.open(Path(answers[0]), header_only=True, permissive=True) as image:
        views = max(int(image.header.nz), 1)
    # A stack draws the movie question, a single extracted view does not.
    diagnostic = Path(answers[2] if views > 1 else answers[1])
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.write_bytes(b"diagnostic")
    diagnostic.with_suffix(".txt").write_text(
        "# fake CTFFIND output\n"
        + "".join(f"{index + 1}.0 30000.0 29500.0 -43.2 0.0 0.10 18.5\n" for index in range(views))
    )


def _fake_ctfphaseflip(arguments: list[str]) -> None:
    source = Path(arguments[arguments.index("-InputStack") + 1])
    with mrcfile.open(source, permissive=True) as stack:
        data = np.asarray(stack.data, dtype=np.float32)
        spacing = float(stack.voxel_size.x)
    with mrcfile.new(
        Path(arguments[arguments.index("-OutputFileName") + 1]), overwrite=True
    ) as output:
        output.set_data(np.ascontiguousarray(data))
        output.voxel_size = spacing


def _fake_tilt(arguments: list[str]) -> None:
    thickness = int(arguments[arguments.index("-THICKNESS") + 1])
    with mrcfile.open(Path(arguments[arguments.index("-input") + 1]), permissive=True) as stack:
        _, height, width = np.asarray(stack.data).shape
    # IMOD writes the reconstruction as (y, z, x).
    with mrcfile.new(Path(arguments[arguments.index("-output") + 1]), overwrite=True) as output:
        output.set_data(np.zeros((height, thickness, width), dtype=np.float32))
