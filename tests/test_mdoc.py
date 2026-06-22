from pathlib import Path

from cryoet_pipeline.ingest import parse_mdoc_text


MDOC_TEXT = """PixelSpacing = 5.4
ImageFile = TS_01.mrc
ImageSize = 924 958
DataMode = 1

[T =     Tilt axis angle = 85.3, binning = 4  spot = 8  camera = 2]

[ZValue = 0]
TiltAngle = 0.000999877
PixelSpacing = 5.4
Defocus = 2.68083
RotationAngle = 175.3
ExposureTime = 0.8
Binning = 4
SubFramePath = D:\\DATA\\Flo\\frames\\TS_01_000_0.0.mrc
NumSubFrames = 8

[ZValue = 1]
TiltAngle = 3.00113
PixelSpacing = 5.4
Defocus = 2.58763
RotationAngle = 175.3
ExposureTime = 0.8
Binning = 4
SubFramePath = D:\\DATA\\Flo\\frames\\TS_01_001_3.0.mrc
NumSubFrames = 8
"""


def test_parse_mdoc_text_maps_frames_and_metadata() -> None:
    manifest = parse_mdoc_text(
        MDOC_TEXT,
        source_mdoc=Path("TS_01.mrc.mdoc"),
        tilt_series_id="TS_01",
        frames_dir=Path("/data/frames"),
        raw_pixel_spacing_angstrom=1.35,
    )

    assert manifest.tilt_series_id == "TS_01"
    assert manifest.num_tilts == 2
    assert manifest.num_subframes_set == {8}
    assert manifest.mdoc_pixel_spacing_angstrom == 5.4
    assert manifest.raw_pixel_spacing_angstrom == 1.35
    assert manifest.notes
    assert manifest.images[0].tilt_angle_deg == 0.000999877
    assert manifest.images[0].local_frame_file == Path("/data/frames/TS_01_000_0.0.mrc")
