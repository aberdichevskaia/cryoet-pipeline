#!/bin/bash
#SBATCH --job-name=ribo-pick
#SBATCH --output=slurm-ribo-pick-%j.log
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#
# Template preparation through extraction, for ONE series first.
#
# Picking is the most expensive step in the pipeline and has never been timed on
# this hardware, so measure it on TS_0005 before committing an allocation to all
# five. The GPU here is a MIG slice — 3 of 7 compute slices and 20 GB — so it is
# a fraction of an A100, not a whole one. Memory is not the constraint at bin8: a
# 512 x 512 x 320 tomogram is 335 MB, and FFT plans bring the total to a few GB.
# Compute is. Read the elapsed time from the log and size the batch of four from
# it rather than from this guess.
#
# To extend to the rest, add them to SERIES and raise --time accordingly.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo
CONFIG=configs/lanes/ribo-bin8.toml
SERIES=(TS_0005)

cd "$REPO"

export IMOD_DIR=/ictstr01/groups/shared/users/peng_marr/a.berdichevskaia/apps/IMOD/imod_5.1.12

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline
source "$IMOD_DIR/IMOD-linux.sh"
set -u

echo "host        $(hostname)"
echo "started     $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Inside a MIG slice the line above reports the whole card; -L lists the slice.
nvidia-smi -L || true
echo

# Preflight, because every one of these fails deep inside a tool after the GPU
# has already been paid for. run_series.py expands $VAR itself, but an unset
# variable expands to nothing and the literal path reaches pytom.
#
# Note the O in PYTOMO_MP_DIR. $PYTOM_MP_DIR is a different, unset name.
if [[ -z "${PYTOMO_MP_DIR:-}" ]]; then
    echo "PYTOMO_MP_DIR is not set — set it here or in the conda env" >&2
    exit 1
fi
for tool in pytom_create_template pytom_create_mask pytom_match_template \
            pytom_extract_candidates; do
    if [[ ! -f "$PYTOMO_MP_DIR/$tool.py" ]]; then
        echo "missing $PYTOMO_MP_DIR/$tool.py" >&2
        exit 1
    fi
done
# prepare-template downloads the map itself, so a missing cache is not fatal —
# only a risk. Outbound access is confirmed from the submit node and unverified
# from the compute nodes, and finding out the hard way costs a scheduled GPU
# allocation. Warn rather than refuse: the step may well succeed on its own.
if [[ ! -f "$PROJECT/emdb-cache/emd_16721.map" ]]; then
    echo "warning: EMD-16721 is not cached in $PROJECT/emdb-cache;"
    echo "         prepare-template will try to download it from EBI, which"
    echo "         needs outbound access from this node."
fi
echo "preflight   ok"
echo

for series in "${SERIES[@]}"; do
    echo "==== $series ===="
    python3 scripts/run_series.py \
        --project "$PROJECT" \
        --config "$CONFIG" \
        --series "$series" \
        --from prepare-template \
        --until extract-particles \
        --execute
    echo
done

echo "finished    $(date -Is)"
