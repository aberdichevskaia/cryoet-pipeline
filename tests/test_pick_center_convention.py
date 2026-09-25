"""Where a pytom pick's particle centre is, and that extraction puts it there.

pytom-match-pick is not installed here, so its peak convention is pinned by a
replica of `pytom_tm/matching.py`: the template pasted with voxel `T // 2` at
the centre of a tomogram-sized array, `irfftn(conj(rfftn(v)) * rfftn(padded))`,
then `roll(flip(scores), floor(V / 2) + 1)`. voltools rotates it about
`(T - 1) / 2`, which `np.rot90` reproduces exactly. If pytom changes either,
these tests will not notice; they pin the arithmetic the extraction relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr
from test_subtomogram_backend import _manifest

from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.backends.subtomogram import CubicCropExtractionBackend
from cryoet_pipeline.models import Artifact, ArtifactKind, AxisOrder
from cryoet_pipeline.particles import (
    pick_center_offset_px,
    pytom_template_shape,
    read_particle_star,
)
from cryoet_pipeline.runtime import DevicePreference


def _pytom_scores(volume: np.ndarray, template: np.ndarray) -> np.ndarray:
    """pytom_tm's score map for one rotation, without its local normalisation."""

    shape_v, shape_t = np.array(volume.shape), np.array(template.shape)
    centre_t, odd, centre_v = shape_t // 2, shape_t % 2, shape_v // 2
    pad = tuple(
        slice(centre_v[i] - centre_t[i], centre_v[i] + centre_t[i] + odd[i]) for i in range(3)
    )
    padded = np.zeros_like(volume)
    padded[pad] = template
    ccc = np.fft.irfftn(
        np.conj(np.fft.rfftn(volume)) * np.fft.rfftn(padded), s=volume.shape, axes=(0, 1, 2)
    )
    shift = tuple(np.floor(shape_v / 2).astype(int) + 1)
    return np.roll(np.flip(ccc), shift, axis=(0, 1, 2))


def _particle(size: int) -> np.ndarray:
    """An asymmetric density whose centre, and rotation fixed point, is (size - 1) / 2."""

    grid = np.indices((size,) * 3).astype(float) - (size - 1) / 2
    z, y, x = grid
    return np.exp(-(x**2 / 3 + y**2 / 5 + z**2 / 2)) + 0.6 * np.exp(
        -((x - 2) ** 2 / 2 + (y + 1.5) ** 2 / 2 + (z - 1) ** 2 / 3)
    )


def _pick(volume: np.ndarray, template: np.ndarray) -> np.ndarray:
    scores = _pytom_scores(volume, template)
    return np.array(np.unravel_index(np.argmax(scores), scores.shape))


@pytest.mark.parametrize("template_box", [16, 15])
@pytest.mark.parametrize("volume_size", [64, 63])
@pytest.mark.parametrize("quarter_turns", [0, 1])
def test_a_pytom_pick_marks_template_voxel_t_half(template_box, volume_size, quarter_turns):
    template = np.rot90(_particle(template_box), k=quarter_turns, axes=(1, 2))
    volume = np.random.default_rng(1).normal(scale=0.01, size=(volume_size,) * 3)
    origin = np.array([21, 17, 30])
    volume[tuple(slice(o, o + template_box) for o in origin)] += template
    pick = _pick(volume, template)
    np.testing.assert_array_equal(pick, origin + template_box // 2)
    # The rotation-invariant centre, where the density really is.
    centre = origin + (template_box - 1) / 2
    np.testing.assert_allclose(pick - centre, pick_center_offset_px(template_box))


def test_offset_and_template_shape_helpers(tmp_path: Path):
    assert pick_center_offset_px(48) == 0.5
    assert pick_center_offset_px(47) == 0.0
    for bad in (0, -2, 48.0, True):
        with pytest.raises(ValueError):
            pick_center_offset_px(bad)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"template_shape": [48, 48, 48]}))
    assert pytom_template_shape(job) == (48, 48, 48)
    for payload in (
        {},
        {"template_shape": []},
        {"template_shape": [48, 0, 48]},
        {"template_shape": [48, True, 48]},
        {"template_shape": "48"},
    ):
        job.write_text(json.dumps(payload))
        assert pytom_template_shape(job) is None
    assert pytom_template_shape(tmp_path / "missing.json") is None


def _extract(
    tmp_path: Path, tomogram_data, pick_xyz, *, box_size, job_shape, explicit=None, declared=None
):
    zarr_path = tmp_path / "tomogram.zarr"
    zarr.save(zarr_path, tomogram_data.astype(np.float32))
    tomogram = Artifact(
        id="TS_TEST:tomogram:fine",
        kind=ArtifactKind.TOMOGRAM,
        path=zarr_path,
        shape=tomogram_data.shape,
        dtype="float32",
        axis_order=AxisOrder.ZYX,
        pixel_spacing_angstrom=9.504,
        parameters={"tilt_series_id": "TS_TEST", "tomogram_branch": "full"},
    )
    star = tmp_path / "picks.star"
    star.write_text(
        "data_particles\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n"
        "_rlnCoordinateZ #3\n_rlnAngleRot #4\n_rlnAngleTilt #5\n_rlnAnglePsi #6\n"
        "_rlnLCCmax #7\n" + " ".join(str(v) for v in pick_xyz) + " 0 0 0 0.9\n"
    )
    parameters = {"tilt_series_id": "TS_TEST", "input_tomogram_id": tomogram.id}
    if job_shape is not None:
        job = tmp_path / "job.json"
        job.write_text(json.dumps({"template_shape": list(job_shape)}))
        parameters["pytom_job_json_path"] = str(job)
    if declared is not None:  # as `--picks-star` records a stated box
        parameters["pick_template_box_px"] = declared
    picks = Artifact(
        id="TS_TEST:picks:full:pytom-match-pick",
        kind=ArtifactKind.PICKS,
        path=star,
        parent_ids=[tomogram.id],
        shape=(1,),
        parameters=parameters,
    )
    context_parameters = {"box_size_px": box_size}
    if explicit is not None:
        context_parameters["pick_template_box_px"] = explicit
    stack, _ = CubicCropExtractionBackend().extract(
        tomogram,
        picks,
        _manifest(),
        BackendContext(
            output_dir=tmp_path / "out", device=DevicePreference.CPU, parameters=context_parameters
        ),
    )
    return stack, read_particle_star(Path(stack.parameters["metadata_path"]))[0]


@pytest.mark.parametrize("template_box", [16, 15])
@pytest.mark.parametrize("box_size", [24, 23])
def test_extraction_puts_a_pytom_particle_centre_at_the_crop_centre(
    tmp_path, template_box, box_size
):
    """End to end: replica pick in, the density's own centre out, exactly."""

    template = _particle(template_box)
    volume = np.zeros((64, 64, 64))
    origin = np.array([21, 17, 30])  # z, y, x
    volume[tuple(slice(o, o + template_box) for o in origin)] += template
    pick_zyx = _pick(volume, template)
    centre_xyz = (origin + (template_box - 1) / 2)[::-1]
    stack, record = _extract(
        tmp_path,
        volume,
        pick_zyx[::-1].astype(float),
        box_size=box_size,
        job_shape=[template_box] * 3,
    )
    start_xyz = np.rint(pick_zyx[::-1]).astype(int) - box_size // 2
    shift = np.array([record.shift_x_px, record.shift_y_px, record.shift_z_px])
    np.testing.assert_allclose(start_xyz + (box_size - 1) / 2 + shift, centre_xyz, atol=1e-12)
    assert stack.parameters["pick_template_box_px"] == template_box


def test_extraction_refuses_to_guess_the_template_box(tmp_path):
    volume = np.zeros((32, 32, 32))
    common = dict(tomogram_data=volume, pick_xyz=(16.0, 16.0, 16.0), box_size=8)
    with pytest.raises(ValueError, match="record no pytom job"):
        _extract(tmp_path / "a", job_shape=None, **common)
    _, record = _extract(tmp_path / "b", job_shape=None, explicit=48, **common)
    assert record.shift_x_px == 0.0  # 16 - 0.5 - (16 - 4) - 3.5
    with pytest.raises(ValueError, match="contradicts itself.*says 47.*says 48"):
        _extract(tmp_path / "c", job_shape=[48] * 3, explicit=47, **common)
    with pytest.raises(ValueError, match="non-cubic"):
        _extract(tmp_path / "d", job_shape=[48, 48, 47], **common)
    # Imported picks carry the box stated for them along to the next extraction.
    _, record = _extract(tmp_path / "e", job_shape=None, declared=47, **common)
    assert record.shift_x_px == 0.5  # 16 - 0 - (16 - 4) - 3.5
    with pytest.raises(ValueError, match="contradicts itself"):
        _extract(tmp_path / "f", job_shape=None, declared=47, explicit=48, **common)
