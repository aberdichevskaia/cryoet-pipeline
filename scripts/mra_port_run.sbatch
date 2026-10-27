#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.log
#
# scripts/mra_port_run.py on one GPU of the MDC cluster (max): a Dynamo MRA
# project run again with the pipeline's port beside it, compared with it
# iteration by iteration, its classes compared with chance levels, and its
# classification report written. Submit from the repo root:
#
#   PROJECT=outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_cone5 \
#       sbatch --export=ALL --job-name=mra_pad_ssn_cone5_port scripts/mra_port_run.sbatch
#
# NAME (default <project>_port), DEVICE (cuda) and DTYPE (float32) pass through.
# A project only set up, never computed by Dynamo (scripts/dynamo_mra_mdc.sbatch
# with SETUP_ONLY=1), is computed by the port alone, with nothing to compare: the
# pipeline's own runs. Its report is written all the same. Global rounds at the
# full box take hours each on one card: give them several (--gres=gpu:a40:4) or
# more time (--time=3-00:00:00).
# An A40, the card Dynamo's own GPU runs were asked for (the timings compare);
# more with --gres=gpu:a40:4 and the particles are dealt among all the job's GPUs,
# a process each. Measured on one A40 (2026-10-26, RyR1 bin8, box 52, before the
# kernel was made about four times faster on the Mac's GPU): 1.3-1.9 ms per
# candidate orientation, about half an hour for a cone round of 2,384 particles x
# 3 channels.

set -euo pipefail

: "${PROJECT:?set PROJECT to the Dynamo project to run again}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/mra_port_run.py ]] || { echo "submit from the repo root" >&2; exit 1; }
source scripts/activate_env.sh
echo "node        $(hostname)"
echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
echo "started     $(date -Is)"
name=${NAME:-$(basename "$PROJECT")_port}
python scripts/mra_port_run.py "$PROJECT" --name "$name" --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-float32}"
echo "finished    $(date -Is)"
if compgen -G "$PROJECT/results/ite_*/averages/refined_table_ref_*_ite_*.tbl" > /dev/null; then
    python scripts/compare_dynamo_classes.py "$PROJECT" "$(dirname "$PROJECT")/$name" \
        || echo "no class comparison" >&2
fi
python scripts/dynamo_classification_report.py "$(dirname "$PROJECT")/$name" \
    || echo "no report; write it with: python scripts/dynamo_classification_report.py" \
        "$(dirname "$PROJECT")/$name" >&2
