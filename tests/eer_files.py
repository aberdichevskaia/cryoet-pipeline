"""EER-like files for tests: the layout of the EMPIAR-11830 movies, without real events."""

from __future__ import annotations

import struct
from pathlib import Path

# The acquisition metadata of an EMPIAR-11830 movie, shortened (Position_1_010).
METADATA = (
    '<metadata><item name="commercialName">Falcon 4</item>'
    '<item name="eerGainReference">ImagesForProcessing/EF-Falcon/300kV/'
    "20230428_151632_EER_GainReference.gain</item>"
    '<item name="exposureTime" unit="s">3.0278130000000001</item>'
    '<item name="numberOfFrames">{declared}</item>'
    '<item name="sensorPixelSize.height" unit="m">1.90370344e-10</item>'
    '<item name="sensorPixelSize.width" unit="m">1.90370344e-10</item>'
    '<item name="totalDose" unit="e/pixel">6.3804134669928416</item></metadata>'
)


def write_eer_like(
    path: Path,
    frames: int,
    *,
    declared: int | None = None,
    integrated_first: bool = False,
    size: int = 16,
) -> None:
    """A BigTIFF laid out as the 11830 EER files are: one strip and one IFD per
    frame, compression 65001, TIFF Orientation 2, and the acquisition metadata in
    tag 65001 of the first page. The strips are not real EER; nothing decodes them."""

    xml = METADATA.format(declared=frames if declared is None else declared).encode()
    body = bytearray(xml)
    strips = []
    pages = [1] * integrated_first + [65001] * frames
    for index in range(len(pages)):
        strips.append((16 + len(body), 10))
        body += bytes([index % 256]) * 10
    ifd_at = 16 + len(body)
    ifds = bytearray()
    for index, compression in enumerate(pages):
        entries = [
            (256, 4, 1, size),
            (257, 4, 1, size),
            (259, 3, 1, compression),
            (262, 3, 1, 1),
            (273, 16, 1, strips[index][0]),
            (274, 3, 1, 2),
            (278, 4, 1, size),
            (279, 16, 1, strips[index][1]),
        ]
        if index == 0:
            entries.append((65001, 7, len(xml), 16))
        here = ifd_at + len(ifds)
        last = index == len(pages) - 1
        following = 0 if last else here + 8 + 20 * len(entries) + 8
        ifds += struct.pack("<Q", len(entries))
        for tag, kind, count, value in entries:
            ifds += struct.pack("<HHQQ", tag, kind, count, value)
        ifds += struct.pack("<Q", following)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"II" + struct.pack("<HHHQ", 43, 8, 0, ifd_at) + bytes(body) + bytes(ifds))
