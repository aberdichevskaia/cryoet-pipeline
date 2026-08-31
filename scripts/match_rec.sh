#!/bin/bash
# Match against a tomogram reconstructed by hand, rather than one the registry
# knows about.
#
#     ./scripts/match_rec.sh TS_0005 /path/to/TS_0005_flip.rec /path/to/flip.tlt flip
#
# Exists for testing geometry hypotheses: `match_one.sh` resolves the tomogram
# from the project, which is the right thing for a real run and useless when the
# point is to compare two reconstructions of the same series.
#
# Same pytom settings as match_one.sh, so the two are comparable: the low-pass
# derived from the fine-alignment residual, a phase-randomised background, CTF
# and wedge weighted per tilt, and a whitened spectrum.
#
# The readout is the standard deviation of the picks' Z coordinate. For a 320 px
# volume with a 39 px template box, pytom cannot centre a box within ~19 px of
# either face, so uniform picks give sd = 282/sqrt(12) = 81.4. A 177 px lamella
# would give about 51. Both series measured so far sit at 80.3.

set -euo pipefail

SERIES="${1:?usage: match_rec.sh SERIES rec_file tlt_file label}"
REC="${2:?missing rec file}"
TLT="${3:?missing tlt file}"
LABEL="${4:?missing label}"

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
WORK=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/tmp/match-$SERIES-$LABEL

if [[ -z "${PYTOMO_MP_DIR:-}" ]]; then
    echo "PYTOMO_MP_DIR is not set" >&2
    exit 1
fi

cd "$REPO"
mkdir -p "$WORK/match_output"

echo "==== $SERIES/$LABEL: preparing inputs ===="
python3 - "$WORK" "$SERIES" "$REC" <<'PY'
import sys
from pathlib import Path

import mrcfile
import numpy as np

from cryoet_pipeline.imod_defocus import read_imod_defocus_file

work, series, rec = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

# `tilt -PERPENDICULAR` writes X, Z, Y, which mrcfile sees as (Y, Z, X). pytom
# wants X, Y, Z in the file, so the array it reads has to be (Z, Y, X).
with mrcfile.mmap(rec, permissive=True, mode="r") as source:
    volume = np.asarray(source.data, dtype=np.float32).transpose(1, 0, 2)
with mrcfile.new(work / f"{series}.mrc", overwrite=True) as destination:
    destination.set_data(np.ascontiguousarray(volume))
    destination.voxel_size = 9.784
print(f"  tomogram {volume.shape} (Z, Y, X)")

entries = read_imod_defocus_file(
    Path("outputs/ribo") / "ctf" / series / f"{series}_ctffind5.defocus"
)
(work / f"{series}_defocus.txt").write_text(
    "".join(f"{e.mean_defocus_nm / 1000.0:.6f}\n" for e in entries)
)
print(f"  {len(entries)} views of defocus")
PY

cp "$TLT" "$WORK/$SERIES.rawtlt"
echo "  angles $(head -1 "$WORK/$SERIES.rawtlt") .. $(tail -1 "$WORK/$SERIES.rawtlt")"

echo
echo "==== $SERIES/$LABEL: match ===="
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
echo "==== $SERIES/$LABEL: extract ===="
"$PYTOMO_MP_DIR/pytom_extract_candidates.py" \
    -j "$WORK/match_output/${SERIES}_job.json" \
    -n 2000 \
    --particle-diameter 290.0 \
    --number-of-false-positives 100

echo
python3 - "$WORK/match_output/${SERIES}_particles.star" "$SERIES/$LABEL" <<'PY'
import sys
import numpy as np

path, label = sys.argv[1], sys.argv[2]
cols, rows = [], []
for line in open(path):
    t = line.strip()
    if t.startswith("_rln"):
        cols.append(t.split()[0])
    elif t and not t.startswith(("data_", "loop_", "#")) and len(t.split()) == len(cols):
        rows.append(t.split())

z = np.array([float(r[cols.index("_rlnCoordinateZ")]) for r in rows])
s = np.array([float(r[cols.index("_rlnLCCmax")]) for r in rows])
print(f"==== {label}: result ====")
print(f"  particles {len(z)}   scores {s.min():.5f}..{s.max():.5f}")
print(f"  Z median {np.median(z):.0f}   sd {z.std():.1f}")
print("    uniform within the maskable region would be 81.4")
print("    a 177 px lamella would be about 51")
print("    TS_0005 and TS_0006 both measured 80.3")
counts, edges = np.histogram(z, bins=16, range=(0, 320))
for c, lo in zip(counts, edges[:-1]):
    print(f"    z {lo:4.0f}-{lo + 20:4.0f} {c:4d} {'#' * int(c * 40 / max(counts.max(), 1))}")
PY
