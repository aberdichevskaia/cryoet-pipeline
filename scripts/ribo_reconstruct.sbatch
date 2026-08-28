#!/bin/bash
#SBATCH --job-name=ribo-recon
#SBATCH --output=slurm-ribo-recon-%j.log
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#
# Reconstruction only, TS_0006..TS_0009. Thickness is per series and comes from
# measure_thickness.py: fill the four numbers in below, in pixels of the bin8
# tomogram. They are not derived here on purpose — the intensity fit carries a
# +-25% mean-free-path uncertainty, so a human should see the number and the R^2
# before a volume is built on it. Deriving it inside the pipeline, with that
# uncertainty recorded and a fit-quality gate, is roadmap section 12.
#
# Add the lamella (from the script) to roughly 25 px for the residual X-axis
# tilt across a 512 px field, plus a margin for edge artefacts. TS_0005 used 320
# for a 177 px lamella and that was comfortable.
#
# Reconstructing too generously is not free: it costs time and disk linearly,
# and template matching then searches empty volume, which is wasted GPU and
# invites false peaks in noise. Snug it up once the XZ slice shows where the
# lamella actually sits.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo

# ---- fill these in from measure_thickness.py, bin8 pixels ----
declare -A THICKNESS=(
    [TS_0006]=320
    [TS_0007]=320
    [TS_0008]=320
    [TS_0009]=320
)
# --------------------------------------------------------------

cd "$REPO"

export IMOD_DIR=/ictstr01/groups/shared/users/peng_marr/a.berdichevskaia/apps/IMOD/imod_5.1.12

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline
source "$IMOD_DIR/IMOD-linux.sh"
set -u

echo "host        $(hostname)"
echo "started     $(date -Is)"
echo

for series in TS_0006 TS_0007 TS_0008 TS_0009; do
    thickness="${THICKNESS[$series]}"
    echo "==== $series, thickness ${thickness} bin8 px ===="
    cryoet reconstruct-tomogram \
        --manifest "$PROJECT/manifests/${series}.json" \
        --registry "$PROJECT/artifacts.json" \
        --out "$PROJECT" \
        --variant bin8 \
        --thickness "$thickness"
    echo
done

echo "finished    $(date -Is)"
