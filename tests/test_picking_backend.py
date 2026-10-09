from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import zarr

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.picking import (
    PytomMatchPickBackend,
    _dilate,
    _extract_candidates_command,
    _included_z_values,
    _match_template_command,
    _PicksStatistics,
    _read_star_statistics,
    _write_dose_accumulation,
    _z_spread_verdict,
    parse_volume_split,
    pick_and_register,
)
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.models import (
    Artifact,
    ArtifactKind,
    AxisOrder,
    QcStatus,
    TiltImage,
    TiltSeriesManifest,
)
from cryoet_pipeline.runtime import DevicePreference


def test_pytom_match_pick_writes_star_and_qc(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    segmentation = _segmentation_artifact(tmp_path)
    manifest = _manifest()
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    registry.add(segmentation)

    match_executable = tmp_path / "pytom_match_template.py"
    match_executable.touch()
    extract_executable = tmp_path / "pytom_extract_candidates.py"
    extract_executable.touch()
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()

    captured_commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        assert env
        command_list = list(command)
        captured_commands.append(command_list)
        if command_list[0] == str(match_executable.resolve()):
            tomogram_path = Path(command_list[command_list.index("-v") + 1])
            with mrcfile.open(tomogram_path, permissive=True) as mrc:
                np.testing.assert_allclose(
                    mrc.data, np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                )
            mask_path = Path(command_list[command_list.index("--tomogram-mask") + 1])
            assert mask_path.is_file()
            destination = Path(command_list[command_list.index("-d") + 1])
            (destination / "TS_TEST_job.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0, stdout="matched", stderr="")

        job_json = Path(command_list[command_list.index("-j") + 1])
        assert job_json.is_file()
        star_path = job_json.parent / "TS_TEST_particles.star"
        star_path.write_text(
            "data_\n\n"
            "loop_\n"
            "_rlnCoordinateX #1\n"
            "_rlnCoordinateY #2\n"
            "_rlnCoordinateZ #3\n"
            "_rlnLCCmax #4\n"
            "1.000000 2.000000 3.000000 0.500000\n"
            "4.000000 5.000000 6.000000 0.700000\n"
        )
        (job_json.parent / "TS_TEST_extraction_graph.svg").write_text("<svg/>")
        return subprocess.CompletedProcess(command, 0, stdout="extracted", stderr="")

    artifacts = pick_and_register(
        PytomMatchPickBackend(fake_runner),
        tomogram,
        manifest,
        BackendContext(
            output_dir=tmp_path / "outputs",
            device=DevicePreference.CUDA,
            parameters={
                "pytom_match_template_executable": match_executable,
                "pytom_extract_candidates_executable": extract_executable,
                "pytom_template": template,
                "pytom_template_mask": template_mask,
                "pytom_gpu_ids": "0",
                "pytom_particle_diameter": 100.0,
                "pytom_z_axis_symmetry": 6,
                "pytom_number_of_particles": 1500,
            },
        ),
        registry,
        supporting_artifacts=[segmentation],
    )

    match_artifact, picks, qc_artifact = artifacts
    assert match_artifact.kind is ArtifactKind.TEMPLATE_MATCH
    assert picks.parent_ids == [match_artifact.id]
    assert registry.get(picks.id) == picks
    assert registry.get(qc_artifact.id) == qc_artifact
    assert picks.kind == ArtifactKind.PICKS
    # The lineage gained a link rather than losing one: picks descend from the
    # search, and the search from the tomogram and the segmentation that
    # constrained it. Asserted as a chain, since that is what it now is.
    assert match_artifact.parent_ids == [tomogram.id, segmentation.id]
    assert registry.descends_from(picks.id, tomogram.id)
    assert registry.descends_from(picks.id, segmentation.id)
    assert picks.shape == (2,)
    assert picks.pixel_spacing_angstrom == pytest.approx(13.5)
    assert picks.parameters["input_tomogram_id"] == tomogram.id
    assert picks.parameters["segmentation_id"] == segmentation.id
    assert picks.parameters["pytom_template_path"] == str(template.resolve())

    match_command = captured_commands[0]
    assert match_command[:1] == [str(match_executable.resolve())]
    assert match_command[match_command.index("--particle-diameter") + 1] == "100.000000"
    assert match_command[match_command.index("-g") + 1] == "0"
    # A symmetric template cuts the orientation search by its symmetry order.
    assert match_command[match_command.index("--z-axis-rotational-symmetry") + 1] == "6"


    extract_command = captured_commands[1]
    assert extract_command[:1] == [str(extract_executable.resolve())]
    assert extract_command[extract_command.index("--tomogram-mask") + 1] is not None
    # pytom requires the extraction cap, and omitting it failed the whole run
    # only after the GPU match had already been paid for.
    assert extract_command[extract_command.index("-n") + 1] == "1500"
    # Extraction falls back to the matching diameter when none is given for it.
    assert extract_command[extract_command.index("--particle-diameter") + 1] == "100.000000"

    report = json.loads(qc_artifact.path.read_text())
    assert report["status"] == QcStatus.PASS.value
    assert report["particle_count"] == 2
    assert report["mean_score"] == pytest.approx(0.6)
    assert report["input_tomogram_id"] == tomogram.id
    assert report["segmentation_id"] == segmentation.id
    assert Path(picks.parameters["command_log_path"]).is_file()
    # Whether the data was weighted is recorded, because otherwise a run without
    # dose weighting is indistinguishable from one with it.
    assert picks.parameters["per_tilt_weighting"] is False
    assert picks.parameters["dose_weighted"] is False
    # The extraction graph is kept without asking, unlike the score volumes.
    graph = Path(picks.parameters["pytom_extraction_graph_path"])
    assert graph.read_text() == "<svg/>"
    assert picks.parameters["pytom_scores_volume_path"] is None


def test_pytom_match_pick_requires_cuda_device(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)

    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        PytomMatchPickBackend().pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CPU,
                parameters={
                    "pytom_template": tmp_path / "template.mrc",
                    "pytom_template_mask": tmp_path / "mask.mrc",
                },
            ),
        )


def test_pytom_match_pick_requires_diameter_or_angular_search(tmp_path: Path) -> None:
    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("command runner should not be invoked")

    with pytest.raises(ValueError, match="exactly one of"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_number_of_particles": 100,
                    "pytom_extract_particle_diameter": 100.0,
                },
            ),
        )


def test_pytom_match_pick_declares_a_mask_that_is_not_a_sphere(tmp_path: Path) -> None:
    """pytom's fast path assumes the mask is rotation-invariant.

    It precomputes the mask's normalisation once and reuses it for every
    orientation, which is only valid for a sphere. An ellipsoidal mask overlaps
    the local density differently at each orientation, so without the flag the
    score ranks orientations instead of measuring fit — quietly.
    """

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()
    captured: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured.append(list(command))
        raise RuntimeError("stop after the command is built")

    for non_spherical, expected in ((True, True), (False, False)):
        captured.clear()
        with pytest.raises(RuntimeError, match="stop after"):
            PytomMatchPickBackend(fake_runner).pick(
                tomogram,
                _manifest(),
                BackendContext(
                    output_dir=tmp_path / f"outputs_{non_spherical}",
                    device=DevicePreference.CUDA,
                    parameters={
                        "pytom_match_template_executable": executable,
                        "pytom_extract_candidates_executable": executable,
                        "pytom_template": template,
                        "pytom_template_mask": template_mask,
                        "pytom_gpu_ids": "0",
                        "pytom_particle_diameter": 100.0,
                        "pytom_number_of_particles": 100,
                        "pytom_non_spherical_mask": non_spherical,
                    },
                ),
            )
        assert ("--non-spherical-mask" in captured[0]) is expected


def test_dilating_the_search_mask_reaches_what_lies_under_a_surface() -> None:
    """A margin is what makes an approximate segmentation useful.

    pytom zeroes the score wherever the mask does not cover, with no tolerance,
    so a mask drawn on a membrane admits only candidates centred on the membrane
    itself — not a lattice sitting a fixed distance beneath it. Growing the mask
    only has to be roughly right about where to look.
    """

    surface = np.zeros((21, 21, 21), dtype=np.float32)
    surface[10, :, :] = 1.0
    depth = 4

    assert surface[10 - depth, 10, 10] == 0.0
    assert _dilate(surface, depth - 1)[10 - depth, 10, 10] == 0.0
    assert _dilate(surface, depth)[10 - depth, 10, 10] == 1.0

    # A point grows into a box of the requested radius, and the volume keeps its shape.
    point = np.zeros((11, 11, 11), dtype=np.float32)
    point[5, 5, 5] = 1.0
    grown = _dilate(point, 2)
    assert grown.shape == point.shape
    assert int(grown.sum()) == 5**3


def test_the_margin_reaches_the_gpu_and_an_empty_mask_does_not(tmp_path: Path) -> None:
    """The margin has to be applied to the file pytom reads, not just computed.

    And a mask that admits nothing would discard every candidate while the run
    still looks successful, so it is refused instead.
    """

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()

    written_segmentations = 0

    def segmentation_of(values: np.ndarray) -> Artifact:
        nonlocal written_segmentations
        written_segmentations += 1
        path = tmp_path / f"segmentation_{written_segmentations}.zarr"
        zarr.save(path, values)
        return Artifact(
            id=f"TS_TEST:segmentation:full:membrain-seg:{written_segmentations}",
            kind=ArtifactKind.SEGMENTATION,
            path=path,
            shape=values.shape,
            parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
        )

    written: list[np.ndarray] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        mask_path = Path(command[command.index("--tomogram-mask") + 1])
        with mrcfile.open(mask_path, permissive=True) as mrc:
            written.append(np.asarray(mrc.data).copy())
        raise RuntimeError("stop after the mask is written")

    def run(values: np.ndarray, dilation: int | None) -> None:
        parameters: dict[str, object] = {
            "pytom_match_template_executable": executable,
            "pytom_extract_candidates_executable": executable,
            "pytom_template": template,
            "pytom_template_mask": template_mask,
            "pytom_gpu_ids": "0",
            "pytom_particle_diameter": 100.0,
            "pytom_number_of_particles": 100,
        }
        if dilation is not None:
            parameters["pytom_segmentation_dilation_px"] = dilation
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / f"outputs_{written_segmentations}",
                device=DevicePreference.CUDA,
                parameters=parameters,
            ),
            supporting_artifacts=[segmentation_of(values)],
        )

    # One voxel of a 2x3x4 tomogram, as a thin segmentation would be.
    sparse = np.zeros((2, 3, 4), dtype=np.float32)
    sparse[1, 1, 1] = 1.0

    with pytest.raises(RuntimeError, match="stop after"):
        run(sparse, None)
    with pytest.raises(RuntimeError, match="stop after"):
        run(sparse, 1)

    plain, grown = written
    assert plain.sum() == 1.0
    assert grown.sum() > plain.sum(), "the margin must reach the file pytom reads"

    with pytest.raises(ValueError, match="admits no voxels"):
        run(np.zeros((2, 3, 4), dtype=np.float32), None)


def test_pytom_match_pick_refuses_a_segmentation_of_another_tomogram(
    tmp_path: Path,
) -> None:
    """A mask of the wrong size is a lane mix-up, and pytom would refuse it anyway."""

    tomogram = _tomogram_artifact(tmp_path)
    segmentation_path = tmp_path / "segmentation.zarr"
    zarr.save(segmentation_path, np.ones((4, 6, 8), dtype=np.float32))
    segmentation = Artifact(
        id="TS_TEST:segmentation:full:membrain-seg",
        kind=ArtifactKind.SEGMENTATION,
        path=segmentation_path,
        shape=(4, 6, 8),
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the match must not run on a mismatched mask")

    with pytest.raises(ValueError, match="pytom requires them to match"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    "pytom_number_of_particles": 100,
                },
            ),
            supporting_artifacts=[segmentation],
        )


def test_pytom_match_pick_weights_the_template_by_the_tomogram_ctf(
    tmp_path: Path,
) -> None:
    """The template has to be weighted like the volume it is compared against."""

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()
    # A real version-3 file WITHOUT astigmatism: five columns, and a phase shift
    # in the sixth. pytom's own version-3 reader averages columns five and six as
    # two defocus axes, so handed this file directly it would read 1.5 um.
    defocus = tmp_path / "TS_TEST.defocus"
    defocus.write_text(
        "12 0 0.0 0.0 0.0 3\n"
        "1 1 0.000000 0.000000 3000.000 0.500000\n"
        "2 2 0.000000 0.000000 4000.000 0.500000\n"
    )
    captured: list[list[str]] = []

    handed_to_pytom: list[float] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        captured.append(list(command))
        written = Path(command[command.index("--defocus") + 1])
        handed_to_pytom.extend(float(line) for line in written.read_text().split())
        raise RuntimeError("stop after the command is built")

    with pytest.raises(RuntimeError, match="stop after"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    "pytom_number_of_particles": 100,
                    "pytom_ctf_defocus_path": defocus,
                    "pytom_ctf_voltage_kv": 300.0,
                    "pytom_ctf_spherical_aberration_mm": 2.7,
                    # Zero is a legitimate amplitude contrast, and the pipeline
                    # can produce it.
                    "pytom_ctf_amplitude_contrast": 0.0,
                    "pytom_tomogram_ctf_model": "phase-flip",
                },
            ),
        )

    command = captured[0]
    # Not the .defocus file: a plain list of micrometres, which has no columns
    # to misread. Handed the .defocus directly, pytom would have read 1.5 um.
    assert Path(command[command.index("--defocus") + 1]).suffix == ".txt"
    assert handed_to_pytom == pytest.approx([3.0, 4.0])
    assert command[command.index("--voltage") + 1] == "300.000000"
    assert command[command.index("--spherical-aberration") + 1] == "2.700000"
    assert command[command.index("--amplitude-contrast") + 1] == "0.000000"
    assert command[command.index("--tomogram-ctf-model") + 1] == "phase-flip"


def test_pytom_gets_the_tilt_angles_the_tomogram_was_built_from(tmp_path: Path) -> None:
    """Fine alignment drops views, so the manifest and the reconstruction differ.

    pytom models the missing wedge from these angles. Handed the collected set it
    would describe a tilt the data never contributed, and nothing downstream
    would notice.
    """

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()
    # One view fewer than the manifest, as an alignment that excluded one leaves.
    solved = tmp_path / "TS_TEST_final.tlt"
    solved.write_text("-3.100000\n")
    # A defocus for those same views. Checked against the manifest instead, this
    # one view against the manifest's two would be wrongly rejected.
    defocus = tmp_path / "TS_TEST.defocus"
    defocus.write_text("12 0 0.0 0.0 0.0 3\n1 1 0.000000 0.000000 3000.000 0.500000\n")
    handed: list[list[float]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        angles = Path(command[command.index("-a") + 1])
        handed.append([float(line) for line in angles.read_text().split()])
        raise RuntimeError("stop after the angles are written")

    with pytest.raises(RuntimeError, match="stop after"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    "pytom_number_of_particles": 100,
                    "pytom_tilt_angles_path": solved,
                    "pytom_ctf_defocus_path": defocus,
                    "pytom_ctf_voltage_kv": 300.0,
                    "pytom_ctf_spherical_aberration_mm": 2.7,
                    "pytom_ctf_amplitude_contrast": 0.07,
                },
            ),
        )

    collected = _manifest().tilt_angles_deg
    assert len(collected) == 2, "the manifest still lists every collected image"
    assert handed[0] == pytest.approx([-3.1]), "the solved angles, not the collected ones"


def test_pytom_match_pick_refuses_a_defocus_covering_other_views(tmp_path: Path) -> None:
    """Defocus and tilt angles are handed to pytom as two parallel lists.

    Nothing downstream ties one to the other, so a model fitted on a different
    set of views would weight every tilt by the wrong defocus, quietly.
    """

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()
    # One view, while the manifest has two.
    defocus = tmp_path / "TS_TEST.defocus"
    defocus.write_text("12 0 0.0 0.0 0.0 3\n1 1 0.000000 0.000000 3000.000 0.500000\n")

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the match must not run on a mismatched defocus")

    with pytest.raises(ValueError, match="must describe the same views"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    "pytom_number_of_particles": 100,
                    "pytom_ctf_defocus_path": defocus,
                    "pytom_ctf_voltage_kv": 300.0,
                    "pytom_ctf_spherical_aberration_mm": 2.7,
                    "pytom_ctf_amplitude_contrast": 0.07,
                },
            ),
        )


def test_pytom_match_pick_refuses_a_partial_ctf_description(tmp_path: Path) -> None:
    """pytom refuses a defocus without the optics; refuse it before the GPU run."""

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()
    defocus = tmp_path / "TS_TEST.defocus"
    defocus.write_text(
        "12 0 0.0 0.0 0.0 3\n"
        "1 1 0.000000 0.000000 3000.000 0.500000\n"
        "2 2 0.000000 0.000000 4000.000 0.500000\n"
    )

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the match must not run on a partial CTF description")

    with pytest.raises(ValueError, match="--spherical-aberration"):
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    "pytom_number_of_particles": 100,
                    "pytom_ctf_defocus_path": defocus,
                    "pytom_ctf_voltage_kv": 300.0,
                },
            ),
        )


def test_pytom_match_pick_checks_extraction_options_before_the_gpu_run(
    tmp_path: Path,
) -> None:
    """Everything extraction will refuse has to be refused before the match.

    pytom requires a cap on how many candidates to extract, and the check for it
    used to sit where the extract command is built — after the whole GPU match
    had run. The runner asserting it was never called is the point of the test.
    """

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the match must not run when extraction cannot succeed")

    def run(extra: dict[str, object]) -> None:
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    **extra,
                },
            ),
        )

    with pytest.raises(TypeError, match="pytom_number_of_particles"):
        run({})

    with pytest.raises(ValueError, match="at most one of"):
        run(
            {
                "pytom_number_of_particles": 100,
                "pytom_cut_off": 0.5,
                "pytom_number_of_false_positives": 1.0,
            }
        )

    # An explicit angular search leaves matching with no diameter, and extraction
    # still needs one to keep peaks apart.
    with pytest.raises(TypeError, match="extraction needs a particle diameter"):
        run(
            {
                "pytom_number_of_particles": 100,
                "pytom_particle_diameter": None,
                "pytom_angular_search": 7.0,
            }
        )


def _tomogram_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "tomogram.zarr"
    zarr.save(path, np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    return Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=path,
        shape=(2, 3, 4),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=13.5,
        binning=10,
        parameters={
            "tilt_series_id": "TS_TEST",
            "tomogram_branch": "full",
        },
    )


def _segmentation_artifact(tmp_path: Path) -> Artifact:
    path = tmp_path / "segmentation.zarr"
    zarr.save(path, np.ones((2, 3, 4), dtype=np.float32))
    return Artifact(
        id="TS_TEST:segmentation:full:membrain-seg",
        kind=ArtifactKind.SEGMENTATION,
        path=path,
        parent_ids=["TS_TEST:tomogram:fine"],
        shape=(2, 3, 4),
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=13.5,
        parameters={
            "tilt_series_id": "TS_TEST",
            "tomogram_branch": "full",
        },
    )


def _match_command(tmp_path: Path, **parameters: object) -> list[str]:
    return _match_template_command(
        Path("pytom_match_template.py"),
        BackendContext(
            output_dir=tmp_path,
            device=DevicePreference.CUDA,
            parameters={"pytom_particle_diameter": 290.0, **parameters},
        ),
        tomogram_path=tmp_path / "tomo.mrc",
        template_path=tmp_path / "template.mrc",
        template_mask_path=tmp_path / "mask.mrc",
        tilt_angles_path=tmp_path / "angles.rawtlt",
        tomogram_mask_path=None,
        destination_dir=tmp_path / "out",
        gpu_ids=["0"],
        voxel_spacing_angstrom=9.784,
        dose_path=tmp_path / "dose.txt",
    )


def test_dose_is_passed_only_alongside_per_tilt_weighting(tmp_path: Path) -> None:
    """Dose reaches pytom through per-tilt weighting and nowhere else.

    Weighting the template as a single average has nothing to do with a dose
    file, so passing one on its own would at best be ignored — and the cost of
    finding out which is a 26-minute match.
    """

    weighted = _match_command(tmp_path, pytom_per_tilt_weighting=True)
    assert "--per-tilt-weighting" in weighted
    assert "--dose-accumulation" in weighted
    assert weighted[weighted.index("--dose-accumulation") + 1].endswith("dose.txt")

    unweighted = _match_command(tmp_path)
    assert "--per-tilt-weighting" not in unweighted
    assert "--dose-accumulation" not in unweighted


def test_dose_is_omitted_when_the_mdoc_had_none(tmp_path: Path) -> None:
    command = _match_template_command(
        Path("pytom_match_template.py"),
        BackendContext(
            output_dir=tmp_path,
            device=DevicePreference.CUDA,
            parameters={
                "pytom_particle_diameter": 290.0,
                "pytom_per_tilt_weighting": True,
            },
        ),
        tomogram_path=tmp_path / "tomo.mrc",
        template_path=tmp_path / "template.mrc",
        template_mask_path=tmp_path / "mask.mrc",
        tilt_angles_path=tmp_path / "angles.rawtlt",
        tomogram_mask_path=None,
        destination_dir=tmp_path / "out",
        gpu_ids=["0"],
        voxel_spacing_angstrom=9.784,
        dose_path=None,
    )

    assert "--per-tilt-weighting" in command
    assert "--dose-accumulation" not in command


def _stats(count: int, z_sd: float | None) -> _PicksStatistics:
    return _PicksStatistics(
        particle_count=count,
        minimum_score=0.10,
        maximum_score=0.13,
        mean_score=0.11,
        coordinate_z_median=158.0,
        coordinate_z_sd=z_sd,
    )


def test_z_spread_warns_when_picks_fill_the_whole_thickness() -> None:
    """The measured case: 385 picks at 80.3 px through a 320 px tomogram.

    Uniform over the 281 px a 39 px box can be centred in spreads 81.1 px, and
    the sampling error of a standard deviation over 385 picks is 2.9 px, so 80.3
    sits a third of one standard error below uniform. Nothing about that count
    says anything — a loose enough threshold always produces plenty — and this is
    the readout that does.
    """

    uniform, warning = _z_spread_verdict(
        _stats(385, 80.3), tomogram_shape=(320, 512, 512), template_box_px=39
    )

    assert uniform == pytest.approx(281 / 12**0.5, rel=1e-3)
    assert warning is not None
    assert "not distinguishable from it" in warning
    assert "80.3" in warning
    # All three readings named, because this measurement cannot separate them —
    # and one of them, a thickness sized snugly around the specimen, is a good
    # reconstruction rather than a fault.
    assert "the picks are noise" in warning
    assert "snug around the specimen" in warning


def test_z_spread_is_silent_for_picks_confined_to_a_lamella() -> None:
    """A 177 px lamella in a 320 px volume spreads about 51 px.

    That is thirty standard errors below uniform, which is what a confined
    specimen looks like and what should pass without comment.
    """

    uniform, warning = _z_spread_verdict(
        _stats(385, 51.0), tomogram_shape=(320, 512, 512), template_box_px=39
    )

    assert uniform == pytest.approx(281 / 12**0.5, rel=1e-3)
    assert warning is None


def test_z_spread_says_nothing_without_the_numbers_to_say_it_with() -> None:
    """Silence rather than a guess when a piece is missing."""

    assert _z_spread_verdict(
        _stats(385, None), tomogram_shape=(320, 512, 512), template_box_px=39
    ) == (None, None)
    assert _z_spread_verdict(
        _stats(385, 80.3), tomogram_shape=None, template_box_px=39
    ) == (None, None)
    assert _z_spread_verdict(
        _stats(1, 80.3), tomogram_shape=(320, 512, 512), template_box_px=39
    ) == (None, None)
    # A box that does not fit leaves nowhere to centre it.
    assert _z_spread_verdict(
        _stats(385, 80.3), tomogram_shape=(32, 512, 512), template_box_px=39
    ) == (None, None)


def test_z_spread_scales_with_the_volume_rather_than_a_fixed_number() -> None:
    """The reference comes from the tomogram, not from a constant.

    The same spread is uniform in one volume and confined in a thicker one, which
    is why the comparison is computed per tomogram.
    """

    _, thin = _z_spread_verdict(
        _stats(400, 80.0), tomogram_shape=(320, 512, 512), template_box_px=39
    )
    _, thick = _z_spread_verdict(
        _stats(400, 80.0), tomogram_shape=(800, 512, 512), template_box_px=39
    )

    assert thin is not None
    assert thick is None


def _dose_manifest(doses: list[object]) -> TiltSeriesManifest:
    manifest = _manifest()
    images = [
        image.model_copy(update={"metadata": {"PriorRecordDose": dose}})
        for image, dose in zip(manifest.images, doses, strict=True)
    ]
    return manifest.model_copy(update={"images": images})


def test_dose_follows_the_views_the_tomogram_was_built_from(tmp_path: Path) -> None:
    """Fine alignment can drop views, and the dose has to drop the same ones.

    The manifest lists everything collected. Pairing the full list against a
    tomogram built from fewer views used to refuse outright, so a series with a
    single excluded view could not be picked with weighting at all.
    """

    manifest = _dose_manifest(["0.000000", "2.450890"])
    third = manifest.images[1].model_copy(
        update={"z_value": 2, "metadata": {"PriorRecordDose": "4.901780"}}
    )
    manifest = manifest.model_copy(update={"images": [*manifest.images, third]})

    output = tmp_path / "dose.txt"
    result = _write_dose_accumulation(
        manifest,
        output,
        expected_views=2,
        included_z_values=[0, 2],
    )

    assert result == output
    # The middle view is gone, and so is its dose.
    assert output.read_text() == "0.000000\n4.901780\n"


def test_dose_refuses_a_z_value_the_manifest_does_not_have(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manifest does not contain"):
        _write_dose_accumulation(
            _dose_manifest(["0.0", "2.45"]),
            tmp_path / "dose.txt",
            expected_views=2,
            included_z_values=[0, 7],
        )


def test_included_z_values_is_absent_on_a_tomogram_that_never_recorded_it() -> None:
    """Older tomograms predate the field and must keep working."""

    plain = Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=Path("tomo.zarr"),
        parameters={"tilt_series_id": "TS_TEST"},
    )
    assert _included_z_values(plain) is None

    recorded = plain.model_copy(
        update={"parameters": {**plain.parameters, "included_z_values": [0, 2]}}
    )
    assert _included_z_values(recorded) == [0, 2]


def test_dose_accumulation_uses_the_dose_already_taken(tmp_path: Path) -> None:
    """PriorRecordDose is accumulated in acquisition order, which is the point.

    A dose-symmetric series is written in tilt order and acquired in another, so
    accumulating ExposureDose down the file would weight the views inside out.
    SerialEM's running total already accounts for that, and here the second view
    in the file carries the *smaller* dose because it was acquired first.
    """

    output = tmp_path / "dose.txt"
    result = _write_dose_accumulation(
        _dose_manifest(["2.450890", "0.000000"]),
        output,
        expected_views=2,
    )

    assert result == output
    assert output.read_text() == "2.450890\n0.000000\n"


def test_dose_accumulation_is_absent_when_the_mdoc_did_not_record_it(
    tmp_path: Path,
) -> None:
    """Dose weighting is an improvement, not a requirement."""

    assert (
        _write_dose_accumulation(_manifest(), tmp_path / "dose.txt", expected_views=2)
        is None
    )


def test_dose_accumulation_refuses_a_view_count_it_cannot_pair(
    tmp_path: Path,
) -> None:
    """pytom pairs dose with tilt angles by position, so counts must agree.

    Refusing matches how the defocus file is handled: with the solved angles
    reaching this code as a bare list of numbers, a mismatch cannot be resolved
    here, and guessing would weight the wrong views.
    """

    with pytest.raises(ValueError, match="must describe the same views"):
        _write_dose_accumulation(
            _dose_manifest(["0.0", "2.45"]),
            tmp_path / "dose.txt",
            expected_views=3,
        )


def test_dose_accumulation_rejects_an_unparseable_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a number"):
        _write_dose_accumulation(
            _dose_manifest(["0.0", "not-a-dose"]),
            tmp_path / "dose.txt",
            expected_views=2,
        )


def _manifest() -> TiltSeriesManifest:
    return TiltSeriesManifest(
        tilt_series_id="TS_TEST",
        source_mdoc=Path("TS_TEST.mdoc"),
        raw_pixel_spacing_angstrom=1.35,
        images=[
            TiltImage(
                z_value=0,
                tilt_angle_deg=0.0,
                subframe_path="frame.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
            TiltImage(
                z_value=1,
                tilt_angle_deg=3.0,
                subframe_path="frame1.mrc",
                num_subframes=1,
                pixel_spacing_angstrom=1.35,
                binning=1,
            ),
        ],
    )


def test_a_zero_cut_off_means_no_threshold_rather_than_an_error() -> None:
    """Taking the top N with no significance floor has to be expressible.

    It is what the published work on EMPIAR-11538 did — top 800 candidates per
    tomogram, then RELION classification found the real 13.8% among them — and it
    is the opposite of what this pipeline did, which was to apply a floor, find
    nothing above it, and read that as an absence of particles. A floor is a
    decision rule, not a property of the data.

    The CLI advertised `min=0.0` while the backend demanded strictly positive, so
    a run that asked for zero failed after the arguments were parsed rather than
    at them — the third instance of that same split in one day.
    """

    context = BackendContext(
        output_dir=Path("/tmp"),
        device=DevicePreference.CUDA,
        parameters={
            "pytom_number_of_particles": 800,
            "pytom_cut_off": 0,
            "pytom_extract_particle_diameter": 580.0,
        },
    )

    command = _extract_candidates_command(
        Path("/x/extract.py"),
        context,
        job_json_path=Path("/x/j.json"),
        tomogram_mask_path=None,
    )

    assert "-c" in command
    assert command[command.index("-c") + 1] == "0.000000"
    # The two are alternatives: a hard cut-off replaces the false-positive budget.
    assert "--number-of-false-positives" not in command
    # And the exclusion is one true particle diameter, since pytom halves it.
    assert command[command.index("--particle-diameter") + 1] == "580.000000"


def test_a_negative_cut_off_is_still_refused() -> None:
    """Zero is a setting; below zero is not a score any correlation reaches."""

    context = BackendContext(
        output_dir=Path("/tmp"),
        device=DevicePreference.CUDA,
        parameters={
            "pytom_number_of_particles": 800,
            "pytom_cut_off": -1.0,
            # Supplied because extraction validates the diameter before it reads
            # the cut-off, so without it the wrong error comes back first.
            "pytom_extract_particle_diameter": 580.0,
        },
    )

    with pytest.raises(ValueError, match="must be finite and not negative"):
        _extract_candidates_command(
            Path("/x/extract.py"),
            context,
            job_json_path=Path("/x/j.json"),
            tomogram_mask_path=None,
        )


def test_a_pick_table_with_no_rows_is_zero_picks_not_its_header(tmp_path: Path) -> None:
    # What pytom_extract_candidates writes when nothing clears its cut-off:
    # starfile's comment line, the block and the columns, and no data. Reading
    # the comment as a row once failed the step on "package" after a full match.
    star = tmp_path / "TS_X_particles.star"
    star.write_text(
        "# Created by the starfile Python package (version 0.5.13) at 23:10:02\n"
        "\n"
        "data_particles\n"
        "\n"
        "loop_\n"
        "_rlnCoordinateX #1\n"
        "_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n"
        "_rlnAngleRot #4\n"
        "_rlnAngleTilt #5\n"
        "_rlnAnglePsi #6\n"
        "_rlnLCCmax #7\n"
    )

    statistics = _read_star_statistics(star)

    assert statistics.particle_count == 0
    assert statistics.maximum_score is None


def test_a_volume_split_reaches_pytom_only_when_asked_and_in_x_y_z_order(
    tmp_path: Path,
) -> None:
    """pytom reads --volume-split as three piece counts, x then y then z.

    1,1,1 is pytom's own default, and asking for it gives the command an unsplit
    run always got, so a lane can write the setting down without changing what
    runs.
    """

    unsplit = _match_command(tmp_path)
    assert "--volume-split" not in unsplit
    assert _match_command(tmp_path, pytom_volume_split="1,1,1") == unsplit

    # Three different counts, so a swapped pair of axes cannot pass.
    split = _match_command(tmp_path, pytom_volume_split="3,2,1")
    assert split.count("--volume-split") == 1
    at = split.index("--volume-split")
    assert split[at + 1 : at + 4] == ["3", "2", "1"]
    # The flag is all that changed.
    assert split[:at] + split[at + 4 :] == unsplit


def test_a_malformed_volume_split_is_refused_before_the_gpu_run(tmp_path: Path) -> None:
    for malformed in ("2,2", "2,2,1,1", "2,0,1", "2,-1,1", "2.5,2,1", "x,y,z", ""):
        with pytest.raises(ValueError, match="volume split"):
            parse_volume_split(malformed)
    assert parse_volume_split(" 2, 2 ,1") == (2, 2, 1)

    tomogram = _tomogram_artifact(tmp_path)
    template = tmp_path / "template.mrc"
    template.touch()
    template_mask = tmp_path / "template_mask.mrc"
    template_mask.touch()
    executable = tmp_path / "pytom_match_template.py"
    executable.touch()

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("the match must not run on a malformed split")

    def pick(split: object) -> None:
        PytomMatchPickBackend(fake_runner).pick(
            tomogram,
            _manifest(),
            BackendContext(
                output_dir=tmp_path / "outputs",
                device=DevicePreference.CUDA,
                parameters={
                    "pytom_match_template_executable": executable,
                    "pytom_extract_candidates_executable": executable,
                    "pytom_template": template,
                    "pytom_template_mask": template_mask,
                    "pytom_gpu_ids": "0",
                    "pytom_particle_diameter": 100.0,
                    "pytom_number_of_particles": 100,
                    "pytom_volume_split": split,
                },
            ),
        )

    with pytest.raises(ValueError, match="pytom_volume_split"):
        pick("2,2")
    with pytest.raises(TypeError, match="pytom_volume_split"):
        pick((2, 2, 1))


def test_the_picks_record_the_split_their_search_ran_with(tmp_path: Path) -> None:
    """pytom's job JSON does not say whether the search was split.

    The noise statistics it records do depend on it, and the false-positive
    cut-off is computed from them, so the picks carry the split. An unsplit
    search records pytom's default rather than nothing, so the two read alike.
    """

    unsplit, unsplit_match, _ = _pick_with(tmp_path / "unsplit")
    split, split_match, registry = _pick_with(tmp_path / "split", pytom_volume_split="3,2,1")

    assert "--volume-split" not in unsplit_match
    assert unsplit.parameters["pytom_volume_split"] == [1, 1, 1]
    assert "--volume-split" in split_match
    assert split.parameters["pytom_volume_split"] == [3, 2, 1]
    # And it survives the registry file the CLI writes.
    written = tmp_path / "artifacts.json"
    registry.write(written)
    reloaded = ArtifactRegistry.load(written).get(split.id)
    assert reloaded.parameters["pytom_volume_split"] == [3, 2, 1]


def _pick_with(
    root: Path, **parameters: object
) -> tuple[Artifact, list[str], ArtifactRegistry]:
    """One run through the backend on fakes: the picks, the match command, the registry."""

    root.mkdir()
    tomogram = _tomogram_artifact(root)
    registry = ArtifactRegistry.empty()
    registry.add(tomogram)
    executable = root / "pytom.py"
    executable.touch()
    template = root / "template.mrc"
    template.touch()
    template_mask = root / "template_mask.mrc"
    template_mask.touch()
    commands: list[list[str]] = []

    def fake_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        if "-j" in command:
            # Extraction: one pick, beside the job it read.
            job_json = Path(command[command.index("-j") + 1])
            (job_json.parent / "TS_TEST_particles.star").write_text(
                "data_\n\nloop_\n_rlnCoordinateZ #1\n_rlnLCCmax #2\n1.000000 0.500000\n"
            )
        else:
            destination = Path(command[command.index("-d") + 1])
            (destination / "TS_TEST_job.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    _, picks, _ = pick_and_register(
        PytomMatchPickBackend(fake_runner),
        tomogram,
        _manifest(),
        BackendContext(
            output_dir=root / "outputs",
            device=DevicePreference.CUDA,
            parameters={
                "pytom_match_template_executable": executable,
                "pytom_extract_candidates_executable": executable,
                "pytom_template": template,
                "pytom_template_mask": template_mask,
                "pytom_gpu_ids": "0",
                "pytom_particle_diameter": 100.0,
                "pytom_number_of_particles": 100,
                **parameters,
            },
        ),
        registry,
    )
    return picks, commands[0], registry
