#!/bin/bash
#SBATCH --job-name=ribo-pick-bisect
#SBATCH --output=slurm-ribo-bisect-%j.log
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#
# Match ONCE, extract MANY times.
#
# `pick-particles` runs the match and the extraction as one command and discards
# the match output with its temporary directory, so every threshold experiment
# costs another 76 minutes of GPU (roadmap section 10). Here the two pytom tools
# are called separately and the match output is kept, so the extraction — which
# takes seconds — can be repeated with different settings against one match.
#
# The first run matched across the whole band out to the bin8 Nyquist of 19.6 A
# and returned 2 candidates of 2000, with the cut-off at 0.11589 flush against
# the top score of 0.1184. This run adds --low-pass 35 and
# --random-phase-correction, and then asks three questions of the result:
# where the calibrated cut-off lands, where it lands when a hundred false
# positives are tolerated, and what the top 2000 scores look like with no
# statistical cut-off at all. The third is the one that says whether the volume
# contains peaks, independent of any threshold.
#
# The inputs pick-particles builds in its temporary directory are rebuilt here by
# the pipeline's own functions, so nothing about their format is guessed.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
WORK=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/tmp/bisect
SERIES=TS_0005

cd "$REPO"

export IMOD_DIR=/ictstr01/groups/shared/users/peng_marr/a.berdichevskaia/apps/IMOD/imod_5.1.12
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline
source "$IMOD_DIR/IMOD-linux.sh"
set -u

if [[ -z "${PYTOMO_MP_DIR:-}" ]]; then
    echo "PYTOMO_MP_DIR is not set" >&2
    exit 1
fi

echo "host       $(hostname)"
echo "started    $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Inside a MIG slice the line above reports the whole card; -L lists the slice.
nvidia-smi -L || true
echo

mkdir -p "$WORK/match_output"

# ---- rebuild the three inputs, using the pipeline's own code ----
python3 - "$WORK" "$SERIES" <<'PY'
import shutil, sys
from pathlib import Path

from cryoet_pipeline.backends.picking import _write_zarr_to_mrc
from cryoet_pipeline.imod_defocus import read_imod_defocus_file

work, series = Path(sys.argv[1]), sys.argv[2]
project = Path("outputs/ribo")

zarr = project / "tomograms" / series / f"{series}_fine_bin8_bin8.zarr"
_write_zarr_to_mrc(zarr, work / f"{series}.mrc", voxel_spacing_angstrom=9.784)

# The solved angles the tomogram was actually built from, not the manifest's.
shutil.copyfile(
    project / "aligned" / series / f"{series}_final_bin8_bin8.tlt",
    work / f"{series}.rawtlt",
)

# Plain micrometres, one per view: pytom's IMOD .defocus reader averages columns
# five and six, and would read our version-3 file at half its value.
entries = read_imod_defocus_file(project / "ctf" / series / f"{series}_ctffind5.defocus")
(work / f"{series}_defocus.txt").write_text(
    "".join(f"{e.mean_defocus_nm / 1000.0:.6f}\n" for e in entries)
)
print(f"prepared {len(entries)} views in {work}")
PY

# ---- match once ----
echo
echo "==== match, --low-pass 35 --random-phase-correction ===="
"$PYTOMO_MP_DIR/pytom_match_template.py" \
    -v "$WORK/$SERIES.mrc" \
    -t "$REPO/outputs/ribo/templates/$SERIES/${SERIES}_pytom_bin8_template.mrc" \
    -m "$REPO/outputs/ribo/templates/$SERIES/${SERIES}_pytom_bin8_mask.mrc" \
    -d "$WORK/match_output" \
    -a "$WORK/$SERIES.rawtlt" \
    --voxel-size-angstrom 9.784000 \
    -g 0 \
    --particle-diameter 290.000000 \
    --defocus "$WORK/${SERIES}_defocus.txt" \
    --voltage 300.000000 \
    --spherical-aberration 2.700000 \
    --amplitude-contrast 0.070000 \
    --tomogram-ctf-model phase-flip \
    --low-pass 35 \
    --random-phase-correction

JOB="$WORK/match_output/${SERIES}_job.json"

# ---- extract several times against that one match ----
run_extract () {
    local label="$1"; shift
    echo
    echo "==== extract: $label ===="
    local out="$WORK/extract_$label"
    mkdir -p "$out"
    cp "$JOB" "$out/"
    "$PYTOMO_MP_DIR/pytom_extract_candidates.py" \
        -j "$out/${SERIES}_job.json" \
        -n 2000 \
        --particle-diameter 290.000000 \
        "$@" 2>&1 | grep -viE "it/s\]$" || true
    python3 - "$out" <<'PY'
import glob, sys
star = glob.glob(f"{sys.argv[1]}/*_particles.star")
if not star:
    print("  no star file"); raise SystemExit
scores = []
for line in open(star[0]):
    parts = line.split()
    if len(parts) > 3:
        try:
            scores.append(float(parts[-2]))
        except ValueError:
            pass
if scores:
    print(f"  particles {len(scores)}   scores {min(scores):.5f}..{max(scores):.5f}"
          f"   spread {max(scores)-min(scores):.5f}")
else:
    print("  particles 0")
PY
}

run_extract "fp1_tophat"  --number-of-false-positives 1   --tophat-filter
run_extract "fp100"       --number-of-false-positives 100
run_extract "nocut"       --cut-off 0

echo
echo "finished   $(date -Is)"
