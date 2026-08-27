#!/bin/bash
#SBATCH --job-name=ribo-lane-b
#SBATCH --output=slurm-ribo-%j.log
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#
# Partition and QOS are site-specific: this pair is what this cluster wants for
# CPU work, and the GPU stages later need their own.
#
# Lane B, TS_0006..TS_0009, from import through CTF correction. CPU only: every
# step here is IMOD or CTFFIND, and the GPU stages (picking, averaging) come
# later.
#
# Series run SEQUENTIALLY, and that is not a preference. ArtifactRegistry does
# read-modify-write of one JSON document with no locking, so parallel jobs
# writing outputs/ribo/artifacts.json would lose each other's entries. Until
# per-series registries exist, one job at a time is the only safe shape.
#
# Reconstruction is deliberately NOT here: thickness is measured per series from
# the Beer-Lambert intensity falloff, so it is four explicit commands afterwards
# rather than one guessed number for four different lamellae.
#
# Disposable, like scripts/run_series.py — replaced by the controller in
# docs/pipeline_engine.md.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo
CONFIG=configs/lanes/ribo-bin8.toml

cd "$REPO"

# conda is not initialised in a non-interactive batch shell, so `conda activate`
# alone would fail here.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline

export IMOD_DIR=/ictstr01/groups/shared/users/peng_marr/a.berdichevskaia/apps/IMOD/imod_5.1.12
source "$IMOD_DIR/IMOD-linux.sh"

echo "host        $(hostname)"
echo "started     $(date -Is)"
echo "cryoet      $(command -v cryoet)"
echo "imod        $(command -v tiltxcorr)"
echo "ctffind     $(command -v ctffind)"
echo

python3 scripts/run_series.py \
    --project "$PROJECT" \
    --config "$CONFIG" \
    --series TS_0006 \
    --series TS_0007 \
    --series TS_0008 \
    --series TS_0009 \
    --from import-tilt-stack \
    --until correct-ctf \
    --execute

echo
echo "finished    $(date -Is)"
