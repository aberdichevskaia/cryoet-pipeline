"""Contract tests, not a second scientific parameter search."""

from dataclasses import replace
from pathlib import Path

import mrcfile
import numpy as np
import pytest
import torch
from test_wedge import _manifest, _particle_stack
from typer.testing import CliRunner

import cryoet_pipeline.backends.subtomogram as native
from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.backends.protocols import BackendContext
from cryoet_pipeline.cli import app
from cryoet_pipeline.filters import low_pass_filter_volume, low_pass_response_rfft
from cryoet_pipeline.models import ArtifactKind, SubtomogramClassificationQc
from cryoet_pipeline.particles import ParticleRecord, read_particle_star, write_particle_star
from cryoet_pipeline.rotations import relion_euler_to_matrix, resample_volume_zyx
from cryoet_pipeline.runtime import DevicePreference
from cryoet_pipeline.wedge import WedgeGeometry, hermitian_weights_rfft, wedge_present_mask_rfft

DEVICE = torch.device("cpu")


def phantom(size=24):
    axis = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.exp(-((x + 3) ** 2 + (y - 2) ** 2 + z**2) / 4) + 0.7 * torch.exp(
        -((x - 2) ** 2 + (y + 3) ** 2 + (z - 3) ** 2) / 7
    )


def settings(size=24, geometry=None):
    return native._ParticleFrameScoring(
        geometry if geometry is not None else WedgeGeometry(-45, 45),
        hermitian_weights_rfft(size, DEVICE),
        low_pass_response_rfft(
            (size,) * 3, cutoff_angstrom=60, pixel_spacing_angstrom=10, device=DEVICE
        ),
        size,
    )


def record(shift):
    return ParticleRecord(0, 0, 0, 0, 27, 48, -19, *shift, None, 1, 1)


@pytest.mark.parametrize("size", [23, 24])
def test_response_matches_existing_low_pass(size):
    volume = phantom(size)
    response = settings(size).low_pass_response
    expected = low_pass_filter_volume(volume, cutoff_angstrom=60, pixel_spacing_angstrom=10)
    actual = torch.fft.irfftn(torch.fft.rfftn(volume) * response, s=volume.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shift", [(0, 0, 0), (0.5, 0.5, 0.5), (1.25, -0.75, 0.3)])
def test_forward_pose_sign_axis_order_and_raw_only_final_resampling(shift, monkeypatch):
    reference = phantom()
    row = record(shift)
    matrix = relion_euler_to_matrix(row.rot_deg, row.tilt_deg, row.psi_deg)
    # Independent expression of q = A.T @ (u - s), then direct grid sampling.
    axis = torch.arange(24, dtype=torch.float64) - 11.5
    z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
    xyz = torch.stack((x, y, z), dim=-1)
    q = (xyz - torch.tensor(shift)) @ torch.tensor(matrix)
    raw = torch.nn.functional.grid_sample(
        reference[None, None], (q / 11.5).to(torch.float32)[None], align_corners=True
    )[0, 0]
    scoring = settings(geometry=WedgeGeometry(-35, 55, 23))
    wedge = wedge_present_mask_rfft(24, scoring.geometry, device=DEVICE)
    raw = torch.fft.irfftn(torch.fft.rfftn(raw) * wedge, s=raw.shape)
    original_raw = raw.clone()
    original_reference = reference.clone()
    calls = []
    original = native.resample_volume_zyx

    def spy(volume, *args, **kwargs):
        calls.append(volume.data_ptr())
        return original(volume, *args, **kwargs)

    monkeypatch.setattr(native, "resample_volume_zyx", spy)
    result = native._search_multi_reference(
        raw,
        [reference],
        row,
        cone=[(0, 0), (10, 90)],
        in_plane_offsets=[0],
        max_shift_px=0,
        device=DEVICE,
        particle_frame=scoring,
    )
    np.testing.assert_allclose(result.matrix, matrix, atol=1e-12)
    np.testing.assert_array_equal(result.shift, shift)
    assert result.correlation == pytest.approx(1, abs=2e-6)
    assert calls == [reference.data_ptr(), reference.data_ptr(), raw.data_ptr()]
    torch.testing.assert_close(raw, original_raw, rtol=0, atol=0)
    torch.testing.assert_close(reference, original_reference, rtol=0, atol=0)
    torch.testing.assert_close(
        result.aligned_volume, original(raw, matrix, np.array(shift), device=DEVICE)
    )


def test_multireference_assignment_uses_prediction_not_list_order():
    ref = phantom()
    row = record((0, 0, 0))
    matrix = relion_euler_to_matrix(row.rot_deg, row.tilt_deg, row.psi_deg)
    raw = resample_volume_zyx(ref, matrix.T, np.zeros(3), device=DEVICE)
    result = native._search_multi_reference(
        raw,
        [ref.flip(0), ref],
        row,
        cone=[(0, 0)],
        in_plane_offsets=[0],
        max_shift_px=0,
        device=DEVICE,
        particle_frame=settings(),
    )
    assert result.class_number == 2
    assert result.correlation == pytest.approx(1, abs=2e-6)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"shift_range_px": 1}, "shift_range_px=0"),
        ({"missing_wedge": False}, "requires missing_wedge"),
        ({"alignment_mask_diameter_angstrom": 100}, "alignment mask"),
        ({"alignment_frame": "invalid"}, "AlignmentFrame"),
    ],
)
def test_backend_refuses_unsupported_combinations_before_outputs(tmp_path, extra, message):
    stack = _particle_stack(tmp_path, [phantom()], [(0, 0, 0)])
    params = {"alignment_frame": "particle", "missing_wedge": True, "shift_range_px": 0, **extra}
    with pytest.raises(ValueError, match=message):
        native.ConeSearchRefinementBackend().average(
            [stack],
            [_manifest()],
            BackendContext(
                output_dir=tmp_path / "result", device=DevicePreference.CPU, parameters=params
            ),
        )
    assert not (tmp_path / "result").exists()


def test_actual_cli_preserves_nonzero_fixed_shifts_and_records_scorer(tmp_path):
    reference = phantom()
    row = record((0.5, -0.25, 0.75))
    matrix = relion_euler_to_matrix(row.rot_deg, row.tilt_deg, row.psi_deg)
    raw = resample_volume_zyx(
        reference, matrix.T, -matrix.T @ np.array([0.5, -0.25, 0.75]), device=DEVICE
    )
    stack = _particle_stack(tmp_path, [raw, raw], [(27, 48, -19)] * 2)
    rows = [replace(row, index=i, half_set=i + 1) for i in range(2)]
    write_particle_star(Path(stack.parameters["metadata_path"]), rows)
    registry = ArtifactRegistry.empty()
    registry.add(stack)
    registry_path = tmp_path / "artifacts.json"
    registry.write(registry_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(_manifest().model_dump_json())
    ref_path = tmp_path / "reference.mrc"
    with mrcfile.new(ref_path) as mrc:
        mrc.set_data(reference.numpy())
        mrc.voxel_size = 10
    result = CliRunner().invoke(
        app,
        [
            "average-particles",
            "--manifest",
            str(manifest),
            "--registry",
            str(registry_path),
            "--out",
            str(tmp_path / "output"),
            "--backend",
            "cone-search",
            "--device",
            "cpu",
            "--alignment-frame",
            "particle",
            "--missing-wedge",
            "--shift-range-px",
            "0",
            "--num-iterations",
            "1",
            "--cone-range-deg",
            "0",
            "--inplane-range-deg",
            "0",
            "--initial-reference",
            str(ref_path),
        ],
    )
    assert result.exit_code == 0, result.output + repr(result.exception)
    registry = ArtifactRegistry.load(registry_path)
    average = next(a for a in registry.artifacts if a.kind is ArtifactKind.SUBTOMOGRAM_AVERAGE)
    params = average.parameters
    assert params["alignment_frame"] == "particle"
    assert params["alignment_score"] == "particle-frame-wedge-ncc"
    qc = SubtomogramClassificationQc.model_validate_json(Path(params["qc_path"]).read_text())
    assert qc.schema_version == 6
    assert qc.starting_correlation_mean == pytest.approx(1, abs=2e-6)
    assert qc.mean_final_correlation == pytest.approx(qc.starting_correlation_mean, abs=1e-7)
    output_rows = read_particle_star(Path(params["metadata_path"]))
    for output, initial in zip(output_rows, rows, strict=True):
        assert (output.shift_x_px, output.shift_y_px, output.shift_z_px) == (0.5, -0.25, 0.75)
        assert output.half_set == initial.half_set
    with mrcfile.open(average.path) as mrc:
        np.testing.assert_allclose(
            mrc.data,
            resample_volume_zyx(
                raw - raw.mean(), matrix, np.array([0.5, -0.25, 0.75]), device=DEVICE
            ),
            atol=1e-7,
        )


def test_particle_frame_kernel_refuses_shift_search():
    with pytest.raises(ValueError, match="frozen shifts"):
        native._search_multi_reference(
            phantom(),
            [phantom()],
            record((0, 0, 0)),
            cone=[(0, 0)],
            in_plane_offsets=[0],
            max_shift_px=1,
            device=DEVICE,
            particle_frame=settings(),
        )
