from __future__ import annotations

import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from cryoet_pipeline.artifacts import ArtifactRegistry
from cryoet_pipeline.external_stack import import_external_tilt_stack
from cryoet_pipeline.models import ArtifactKind, AxisOrder


def _mdoc_text(tilt_count: int) -> str:
    lines = ["PixelSpacing = 1.188"]
    for index in range(tilt_count):
        lines.extend(
            [
                "",
                f"[ZValue = {index}]",
                f"TiltAngle = {float(index) * 2.0 - 20.0}",
                "PixelSpacing = 1.188",
                "Binning = 1",
                f"SubFramePath = \\\\192.168.1.1\\OffloadData\\frames\\TS_0002_{index:04d}.mrc",
                "NumSubFrames = 10",
                f"Defocus = {2.0 + index * 0.01}",
            ]
        )
    return "\n".join(lines)


def _write_stack(path: Path, *, tilt_count: int, pixel_spacing_angstrom: float = 1.188) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.arange(tilt_count * 4 * 6, dtype=np.int16).reshape(tilt_count, 4, 6))
        mrc.voxel_size = pixel_spacing_angstrom


def test_import_external_tilt_stack_keeps_a_float_source_as_float(
    tmp_path: Path,
) -> None:
    """A dtype MRC cannot hold is widened; one it can is not.

    The rule is about what the format supports, not about preferring integers:
    a float source stays float, and only something MRC has no mode for becomes
    float32.
    """

    mdoc_path = tmp_path / "TS_0003.mrc.mdoc"
    mdoc_path.write_text(_mdoc_text(tilt_count=2))
    stack_path = tmp_path / "raw" / "TS_0003.mrc"
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(np.linspace(0.0, 1.0, 2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6))
        mrc.voxel_size = 1.188

    result = import_external_tilt_stack(
        mdoc_path=mdoc_path,
        stack_path=stack_path,
        tilt_series_id="TS_0003",
        out=tmp_path / "out",
    )

    assert result.tilt_stack.dtype == "float32"
    with mrcfile.open(result.tilt_stack.path, permissive=True) as canonical:
        assert canonical.data.dtype == np.float32


def test_import_external_tilt_stack_rejects_a_nonfinite_float_source(
    tmp_path: Path,
) -> None:
    """The finiteness check still applies where non-finite values can exist."""

    mdoc_path = tmp_path / "TS_0004.mrc.mdoc"
    mdoc_path.write_text(_mdoc_text(tilt_count=2))
    stack_path = tmp_path / "raw" / "TS_0004.mrc"
    stack_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((2, 4, 6), dtype=np.float32)
    data[1, 2, 3] = np.nan
    with mrcfile.new(stack_path, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = 1.188

    with pytest.raises(ValueError, match="nonfinite"):
        import_external_tilt_stack(
            mdoc_path=mdoc_path,
            stack_path=stack_path,
            tilt_series_id="TS_0004",
            out=tmp_path / "out",
        )


def test_import_external_tilt_stack_writes_manifest_stack_and_registry(
    tmp_path: Path,
) -> None:
    mdoc_path = tmp_path / "TS_0002.mrc.mdoc"
    mdoc_path.write_text(_mdoc_text(3))
    stack_path = tmp_path / "TS_0002.mrc"
    _write_stack(stack_path, tilt_count=3)
    out = tmp_path / "out"

    result = import_external_tilt_stack(
        mdoc_path=mdoc_path,
        stack_path=stack_path,
        tilt_series_id="TS_0002",
        out=out,
    )

    assert result.manifest.tilt_series_id == "TS_0002"
    assert result.manifest.num_tilts == 3
    assert result.manifest.raw_pixel_spacing_angstrom == pytest.approx(1.188)
    assert all(image.local_frame_file is None for image in result.manifest.images)

    manifest_payload = json.loads(result.manifest_path.read_text())
    assert manifest_payload["tilt_series_id"] == "TS_0002"

    assert result.tilt_stack.kind == ArtifactKind.TILT_STACK
    assert result.tilt_stack.shape == (3, 4, 6)
    # The source is 16-bit, and so is the copy: widening it to float32 would
    # double every byte on disk without adding a digit of information.
    assert result.tilt_stack.dtype == "int16"
    assert result.tilt_stack.axis_order == AxisOrder.TYX
    assert result.tilt_stack.pixel_spacing_angstrom == pytest.approx(1.188)
    assert result.tilt_stack.parameters["source_stack_path"] == str(stack_path)

    with mrcfile.open(result.tilt_stack.path, permissive=True) as canonical:
        np.testing.assert_array_equal(
            canonical.data,
            np.arange(3 * 4 * 6, dtype=np.int16).reshape(3, 4, 6),
        )
        assert canonical.data.dtype == np.int16

    assert result.tilt_angles.kind == ArtifactKind.TILT_ANGLES
    assert result.tilt_angles.path.read_text() == "-20.000000\n-18.000000\n-16.000000\n"

    registry = ArtifactRegistry.load(result.artifact_registry_path)
    assert registry.get(result.tilt_stack.id) == result.tilt_stack
    assert registry.get(result.tilt_angles.id) == result.tilt_angles


def test_import_external_tilt_stack_rejects_tilt_count_mismatch(tmp_path: Path) -> None:
    mdoc_path = tmp_path / "TS_0002.mrc.mdoc"
    mdoc_path.write_text(_mdoc_text(3))
    stack_path = tmp_path / "TS_0002.mrc"
    _write_stack(stack_path, tilt_count=5)

    with pytest.raises(ValueError, match="mdoc has 3 ZValue tilts"):
        import_external_tilt_stack(
            mdoc_path=mdoc_path,
            stack_path=stack_path,
            tilt_series_id="TS_0002",
            out=tmp_path / "out",
        )


def test_import_external_tilt_stack_rejects_missing_mdoc(tmp_path: Path) -> None:
    stack_path = tmp_path / "TS_0002.mrc"
    _write_stack(stack_path, tilt_count=3)

    with pytest.raises(FileNotFoundError, match="mdoc file not found"):
        import_external_tilt_stack(
            mdoc_path=tmp_path / "missing.mdoc",
            stack_path=stack_path,
            tilt_series_id="TS_0002",
            out=tmp_path / "out",
        )


def test_import_external_tilt_stack_requires_overwrite_for_existing_manifest(
    tmp_path: Path,
) -> None:
    mdoc_path = tmp_path / "TS_0002.mrc.mdoc"
    mdoc_path.write_text(_mdoc_text(3))
    stack_path = tmp_path / "TS_0002.mrc"
    _write_stack(stack_path, tilt_count=3)
    out = tmp_path / "out"

    import_external_tilt_stack(
        mdoc_path=mdoc_path,
        stack_path=stack_path,
        tilt_series_id="TS_0002",
        out=out,
    )

    with pytest.raises(FileExistsError, match="manifest already exists"):
        import_external_tilt_stack(
            mdoc_path=mdoc_path,
            stack_path=stack_path,
            tilt_series_id="TS_0002",
            out=out,
        )

    result = import_external_tilt_stack(
        mdoc_path=mdoc_path,
        stack_path=stack_path,
        tilt_series_id="TS_0002",
        out=out,
        overwrite=True,
    )
    assert result.tilt_stack.path.exists()
