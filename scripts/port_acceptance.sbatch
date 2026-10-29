#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.log
#
# scripts/port_acceptance.py on the MDC cluster (max): the port's acceptance check.
# Every iteration of a project Dynamo computed is run again from Dynamo's own inputs
# to it, on a seeded sample of its particles, and compared with Dynamo's answer; the
# log ends in PASS or FAIL, and the job fails with it. Run it after any change to the
# port. Submit from the repo root, once on the CPU (the numpy kernel on every CPU of
# the job, and the averaging of every particle):
#
#   PROJECT=outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan \
#       sbatch --export=ALL --job-name=accept_g2atan_cpu scripts/port_acceptance.sbatch
#
# and once on one A40 (the torch kernel in float32; the averaging runs on the CPU
# whatever the device, so it is not repeated there):
#
#   PROJECT=outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan DEVICE=cuda \
#       sbatch --export=ALL --partition=gpu --gres=gpu:a40:1 --cpus-per-task=8 \
#       --job-name=accept_g2atan_gpu scripts/port_acceptance.sbatch
#
# ITERATIONS (a list, "1 2"; default every one Dynamo finished), PARTICLES (32),
# SEED, DTYPE (float32) and PER_GPU (1) pass through. On g2atan, iterations 1 and 2
# are the two 360 deg rounds at dim 30 and 34, 3 to 5 the cones at the full box 52.
# Expected, from the per-candidate times measured so far (not this job's): on 32
# cores about 6 minutes for iteration 1, 25 for iteration 2, a minute for each cone
# and 2 for each iteration's averaging; on one A40 about 3 and 7 minutes, seconds
# for the cones.

set -euo pipefail

: "${PROJECT:?set PROJECT to a project Dynamo computed}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/port_acceptance.py ]] || { echo "submit from the repo root" >&2; exit 1; }
source scripts/activate_env.sh
device=${DEVICE:-numpy}
echo "node        $(hostname)"
echo "cpus        ${SLURM_CPUS_PER_TASK:-?}"
[[ "$device" == cuda ]] && echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
echo "started     $(date -Is)"
options=()
if [[ -n "${ITERATIONS:-}" ]]; then
    read -ra numbers <<< "$ITERATIONS"
    options+=(--iterations "${numbers[@]}")
fi
[[ -n "${SEED:-}" ]] && options+=(--seed "$SEED")
[[ "$device" != numpy ]] && options+=(--skip-averages --dtype "${DTYPE:-float32}" \
    --per-gpu "${PER_GPU:-1}")
status=0
python scripts/port_acceptance.py "$PROJECT" --device "$device" --particles "${PARTICLES:-32}" \
    ${options[@]+"${options[@]}"} || status=$?
echo "finished    $(date -Is)"
exit "$status"
