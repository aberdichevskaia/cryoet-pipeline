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

    picks, qc_artifact = artifacts
    assert registry.get(picks.id) == picks
    assert registry.get(qc_artifact.id) == qc_artifact
    assert picks.kind == ArtifactKind.PICKS
    assert picks.parent_ids == [tomogram.id, segmentation.id]
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

    with pytest.raises(ValueError, match="would not line up with the tilt angles"):
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
