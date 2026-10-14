#!/bin/bash
#SBATCH --job-name=ribo-lane-b-2
#SBATCH --output=slurm-ribo2-%j.log
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#
# Picks up where the first batch stopped: TS_0006 already has its fine alignment
# (rerun by hand with a raised pruning fraction), so it resumes after that step,
# and the other three start from import.
#
# Two invocations rather than one because the step range in scripts/run_series.py
# is shared by every series in a call — a real controller would carry per-series
# state instead, which is why that script is disposable.
#
# Series stay sequential: the registry is one JSON document with no locking.
#
# Reconstruction is still not here. Thickness is measured per series.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo
CONFIG=configs/lanes/ribo-bin8.toml

cd "$REPO"

export IMOD_DIR=/ictstr01/groups/shared/users/peng_marr/a.berdichevskaia/apps/IMOD/imod_5.1.12

source scripts/activate_env.sh
# `set -u` is ours, not theirs: IMOD-linux.sh appends to MANPATH assuming it
# exists.
set +u
source "$IMOD_DIR/IMOD-linux.sh"
set -u

echo "host        $(hostname)"
echo "started     $(date -Is)"
echo "cryoet      $(command -v cryoet)"
echo "imod        $(command -v tiltxcorr)"
echo "ctffind     $(command -v ctffind)"
echo

echo "==== TS_0006: resuming after fine alignment ===="
python3 scripts/run_series.py \
    --project "$PROJECT" \
    --config "$CONFIG" \
    --series TS_0006 \
    --from build-ctf-stack \
    --until correct-ctf \
    --execute

echo
echo "==== TS_0007..TS_0009: from import ===="
python3 scripts/run_series.py \
    --project "$PROJECT" \
    --config "$CONFIG" \
    --series TS_0007 \
    --series TS_0008 \
    --series TS_0009 \
    --from import-tilt-stack \
    --until correct-ctf \
    --execute

echo
echo "finished    $(date -Is)"
