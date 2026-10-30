# The frames each mdoc names (SubFramePath): TIFFs of NumSubFrames frames at its
# ImageSize, 8-bit and Deflate-compressed as a K3 movie can be, so that a reader
# finds the frames, their count and their size. The pixels are zeros, which keeps
# 82 files at about 15 MB. Each file is dated to its tilt's DateTime and each mdoc
# to its last, as a copy that kept the acquisition's times would be.
set -euo pipefail
python3 - data/hela <<'EOF'
import os
import re
import struct
import sys
import zlib
from datetime import datetime
from pathlib import Path


def write_tiff(path, width, height, frames, strip):
    """Little-endian TIFF: per frame its one strip, then its directory."""
    out = bytearray(b"II*\x00\x00\x00\x00\x00")
    link = 4  # where the offset of the next directory goes
    for _ in range(frames):
        offset = len(out)
        out += strip + b"\x00" * (len(strip) % 2)
        # width, height, 8 bits, Deflate, black is zero, strip offset, 1 sample,
        # rows per strip, strip bytes, one plane
        entries = [(256, 4, width), (257, 4, height), (258, 3, 8), (259, 3, 8), (262, 3, 1),
                   (273, 4, offset), (277, 3, 1), (278, 4, height), (279, 4, len(strip)),
                   (284, 3, 1)]
        struct.pack_into("<I", out, link, len(out))
        out += struct.pack("<H", len(entries))
        for tag, kind, value in entries:
            packed = struct.pack("<HH", value, 0) if kind == 3 else struct.pack("<I", value)
            out += struct.pack("<HHI", tag, kind, 1) + packed
        link = len(out)
        out += b"\x00\x00\x00\x00"
    path.write_bytes(bytes(out))


folder = Path(sys.argv[1])
for mdoc in sorted(folder.glob("*.mdoc")):
    text = mdoc.read_text()
    width, height = map(int, re.search(r"^ImageSize = (\d+) (\d+)", text, re.M).groups())
    frames = int(re.search(r"^NumSubFrames = (\d+)", text, re.M)[1])
    strip = zlib.compress(bytes(width * height), 9)
    last = None
    for section in re.split(r"^\[ZValue = \d+\]\s*$", text, flags=re.M)[1:]:
        name = re.search(r"^SubFramePath = .*[\\/](\S+)\s*$", section, re.M)[1]
        when = re.search(r"^DateTime = (.+?)\s*$", section, re.M)[1]
        write_tiff(folder / name, width, height, frames, strip)
        last = datetime.strptime(" ".join(when.split()), "%d-%b-%y %H:%M:%S").timestamp()
        os.utime(folder / name, (last, last))
    if last is not None:
        os.utime(mdoc, (last, last))
        os.utime(folder, (last, last))
EOF
