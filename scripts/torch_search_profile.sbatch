#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.log
#
# scripts/torch_search_profile.py on one A40 of the MDC cluster (max): where the
# torch search kernel's time goes on a finished Dynamo project's own searches,
# step by step, how much of it the GPU computes, and whether several processes
# sharing the card fill it. Submit from the repo root:
#
#   PROJECT=outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_cone5 \
#       sbatch --export=ALL scripts/torch_search_profile.sbatch
#
# ITERATIONS (default "1 3": cone5's 300- and 221-candidate rounds), PARTICLES
# (32 per iteration) and PROCESSES (default "1 2 4") pass through. The CPUs are
# the port run's (8 for its processes), so the numbers are that run's. Nothing is
# written but the log.

set -euo pipefail

: "${PROJECT:?set PROJECT to a Dynamo project with finished iterations}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/torch_search_profile.py ]] || { echo "submit from the repo root" >&2; exit 1; }
source scripts/activate_env.sh
echo "node        $(hostname)"
echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
read -r -a iterations <<< "${ITERATIONS:-1 3}"
read -r -a processes <<< "${PROCESSES-1 2 4}"
python scripts/torch_search_profile.py "$PROJECT" --iterations "${iterations[@]}" \
    --particles "${PARTICLES:-32}" --device cuda --processes ${processes[@]+"${processes[@]}"}
