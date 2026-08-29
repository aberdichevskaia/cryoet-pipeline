#!/bin/bash
# Match one series with the settings that are actually right, then report where
# the picks sit in Z.
#
#     ./scripts/match_one.sh TS_0006
#
# Runs pytom directly rather than through `pick-particles`, so the match output
# is kept and the extraction can be repeated against it for seconds instead of
# 26 minutes (roadmap section 10). Needs a GPU allocation: pytom cannot be
# imported without a CUDA runtime, whatever the tool does.
#
# The readout is the standard deviation of the picks' Z coordinate. Real
# particles sit inside the lamella; noise peaks are spread through the whole
# reconstructed thickness. For a 320 px volume with a 39 px template box, pytom
# cannot centre a box within ~19 px of either face, so uniform picks give
# sd = 282/sqrt(12) = 81.4 rather than 92.4 — compare against that, not against
# the full thickness. A 177 px lamella would give about 51.

set -euo pipefail

SERIES="${1:?usage: match_one.sh TS_0006}"
REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
WORK=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/tmp/match-$SERIES

if [[ -z "${PYTOMO_MP_DIR:-}" ]]; then
    echo "PYTOMO_MP_DIR is not set" >&2
    exit 1
fi

cd "$REPO"
mkdir -p "$WORK/match_output"

echo "==== $SERIES: preparing inputs ===="
python3 - "$WORK" "$SERIES" <<'PY'
import shutil, sys
from pathlib import Path

from cryoet_pipeline.backends.picking import _write_zarr_to_mrc
from cryoet_pipeline.imod_defocus import read_imod_defocus_file

work, series = Path(sys.argv[1]), sys.argv[2]
project = Path("outputs/ribo")

_write_zarr_to_mrc(
    project / "tomograms" / series / f"{series}_fine_bin8_bin8.zarr",
    work / f"{series}.mrc",
    voxel_spacing_angstrom=9.784,
)
shutil.copyfile(
    project / "aligned" / series / f"{series}_final_bin8_bin8.tlt",
    work / f"{series}.rawtlt",
)
entries = read_imod_defocus_file(project / "ctf" / series / f"{series}_ctffind5.defocus")
(work / f"{series}_defocus.txt").write_text(
    "".join(f"{e.mean_defocus_nm / 1000.0:.6f}\n" for e in entries)
)
print(f"  {len(entries)} views")
PY

echo
echo "==== $SERIES: match ===="
"$PYTOMO_MP_DIR/pytom_match_template.py" \
    -v "$WORK/$SERIES.mrc" \
    -t "$REPO/outputs/ribo/templates/$SERIES/${SERIES}_pytom_bin8_template.mrc" \
    -m "$REPO/outputs/ribo/templates/$SERIES/${SERIES}_pytom_bin8_mask.mrc" \
    -d "$WORK/match_output" \
    -a "$WORK/$SERIES.rawtlt" \
    --voxel-size-angstrom 9.784 \
    -g 0 \
    --particle-diameter 290.0 \
    --defocus "$WORK/${SERIES}_defocus.txt" \
    --voltage 300.0 \
    --spherical-aberration 2.7 \
    --amplitude-contrast 0.07 \
    --tomogram-ctf-model phase-flip \
    --low-pass 35 \
    --random-phase-correction \
    --per-tilt-weighting \
    --spectral-whitening

echo
echo "==== $SERIES: extract ===="
"$PYTOMO_MP_DIR/pytom_extract_candidates.py" \
    -j "$WORK/match_output/${SERIES}_job.json" \
    -n 2000 \
    --particle-diameter 290.0 \
    --number-of-false-positives 100

echo
python3 - "$WORK/match_output/${SERIES}_particles.star" "$SERIES" <<'PY'
import sys
import numpy as np

path, series = sys.argv[1], sys.argv[2]
cols, rows = [], []
for line in open(path):
    t = line.strip()
    if t.startswith("_rln"):
        cols.append(t.split()[0])
    elif t and not t.startswith(("data_", "loop_", "#")) and len(t.split()) == len(cols):
        rows.append(t.split())

z = np.array([float(r[cols.index("_rlnCoordinateZ")]) for r in rows])
s = np.array([float(r[cols.index("_rlnLCCmax")]) for r in rows])
print(f"==== {series}: result ====")
print(f"  particles {len(z)}   scores {s.min():.5f}..{s.max():.5f}")
print(f"  Z median {np.median(z):.0f}   sd {z.std():.1f}")
print(f"    uniform within the maskable region would be 81.4")
print(f"    a 177 px lamella would be about 51")
counts, edges = np.histogram(z, bins=16, range=(0, 320))
for c, lo in zip(counts, edges[:-1]):
    print(f"    z {lo:4.0f}-{lo + 20:4.0f} {c:4d} {'#' * int(c * 40 / max(counts.max(), 1))}")
PY
