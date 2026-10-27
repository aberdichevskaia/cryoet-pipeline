#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.log
#
# scripts/torch_search_check.py on one GPU of the MDC cluster (max): our torch
# search kernel against Dynamo's own search and the numpy port, on a sample of a
# finished project's particles, with each kernel's time per candidate. Submit
# from the repo root:
#
#   PROJECT=outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan ITERATION=5 \
#       GLOBAL_STEP=9.0 GLOBAL_LOW=13 sbatch --export=ALL scripts/torch_search_check.sbatch
#
# ITERATION must be one searched at the full box (a cone round, or a global round
# set up with GLOBAL_FULL_BOX=1). PARTICLES (16 unless given; 0 for all) is the
# sample; SKIP_NUMPY=1 compares the torch kernel with Dynamo alone, for a global
# round, where numpy takes minutes per search. GLOBAL_STEP (deg) and GLOBAL_LOW
# (Fourier px) add the timing of a 360 deg search at the full box. Nothing is
# written but the log.
#
# An A40, the card the Dynamo GPU runs are asked for, so the two timings compare
# (max's gpu partition, 2026-10-26: T4, V100, A40, H100, H200, B300; 27 A40s).
# Another type with --gres=gpu:<type>:1.

set -euo pipefail

: "${PROJECT:?set PROJECT to a finished Dynamo project}"
: "${ITERATION:?set ITERATION to a finished iteration searched at the full box}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/torch_search_check.py ]] || { echo "submit from the repo root" >&2; exit 1; }
source scripts/activate_env.sh
echo "node        $(hostname)"
echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
options=()
[[ -n "${GLOBAL_STEP:-}" ]] && options+=(--global-step "$GLOBAL_STEP")
[[ -n "${GLOBAL_LOW:-}" ]] && options+=(--global-low "$GLOBAL_LOW")
[[ -n "${SKIP_NUMPY:-}" ]] && options+=(--skip-numpy)
python scripts/torch_search_check.py "$PROJECT" --iteration "$ITERATION" \
    --particles "${PARTICLES:-16}" --device cuda ${options[@]+"${options[@]}"}
