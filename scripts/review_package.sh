#!/bin/bash
# Assemble everything someone would need to say whether a tomogram is sound.
#
#     ./scripts/review_package.sh TS_0005
#
# Written because the question left is visual and not ours to answer: is this a
# normal FIB-lamella tomogram of HEK293T cells, and would anyone expect to pick
# ribosomes from it at this binning. Every numerical proxy tried so far failed
# for its own reason — no valid reference existed for any of them, and a person
# who reads these daily *is* the reference.
#
# Everything here is copied rather than moved, so the project is untouched.

set -euo pipefail

SERIES="${1:?usage: review_package.sh TS_0005}"
REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
OUT=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/review-$SERIES

cd "$REPO"
mkdir -p "$OUT"

copy () {
    if [[ -e "$1" ]]; then
        cp -r "$1" "$OUT/"
        echo "  $(basename "$1")  $(du -h "$1" | cut -f1)"
    else
        echo "  MISSING: $1"
    fi
}

P=outputs/ribo
echo "collecting into $OUT"

# The volume itself, and the one slice that shows the lamella edge-on.
copy "$P/tomograms/$SERIES/${SERIES}_fine_bin8_bin8.rec"
copy "$P/qc/$SERIES/reconstruction/${SERIES}_fine_bin8_bin8_central_xz.mrc"
copy "$P/qc/$SERIES/reconstruction/${SERIES}_fine_bin8_bin8_central_xy.mrc"

# The tilt-series as it went in, so the raw data can be judged too.
copy "$P/qc/$SERIES/coarse_alignment/${SERIES}_coarse_aligned_bin16.st"

# What was matched against, at the tomogram's own voxel size.
copy "$P/templates/$SERIES/${SERIES}_pytom_bin8_template.mrc"
copy "$P/templates/$SERIES/${SERIES}_pytom_bin8_mask.mrc"

# Where the picks landed, ready to open over the tomogram.
copy "$P/artiax/$SERIES"

# Every QC document, so the numbers are checkable rather than quoted.
mkdir -p "$OUT/qc"
find "$P" -name "*_qc.json" -path "*$SERIES*" -exec cp {} "$OUT/qc/" \;
echo "  qc/  $(ls -1 "$OUT/qc" | wc -l) documents"

# A Z-averaged copy, ten slices to one: a single unfiltered bin8 slice is
# honestly noisy and that has misled this investigation more than once.
python3 - "$P/tomograms/$SERIES/${SERIES}_fine_bin8_bin8.rec" "$OUT/${SERIES}_zavg10.mrc" <<'PY'
import sys
import mrcfile
import numpy as np

source, destination = sys.argv[1], sys.argv[2]
with mrcfile.mmap(source, permissive=True, mode="r") as handle:
    volume = np.asarray(handle.data, dtype=np.float32).transpose(1, 0, 2)  # -> (Z, Y, X)
usable = volume.shape[0] // 10 * 10
averaged = volume[:usable].reshape(-1, 10, *volume.shape[1:]).mean(axis=1)
with mrcfile.new(destination, overwrite=True) as handle:
    handle.set_data(np.ascontiguousarray(averaged))
    handle.voxel_size = 9.784
print(f"  {destination.split('/')[-1]}  {averaged.shape} (ten slices averaged into one)")
PY

echo
echo "total $(du -sh "$OUT" | cut -f1)"
