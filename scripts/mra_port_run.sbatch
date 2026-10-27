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
# NAME (default <project>_port), DEVICE (cuda), DTYPE (float32) and PER_GPU
# (processes sharing each GPU, default 1) pass through.
# A project only set up, never computed by Dynamo (scripts/dynamo_mra_mdc.sbatch
# with SETUP_ONLY=1), is computed by the port alone, with nothing to compare: the
# pipeline's own runs. Its report is written all the same. Global rounds at the
# full box take hours each on one card: 0.26 ms a candidate on an A40 (full_g1,
# 2026-10-27), so a 360 deg round in 9 deg steps (21,561 candidates a search) over
# 2,384 particles x 3 channels is about 11 h on one card, in 7.8 deg steps (32,583)
# about 17 h; give them several (--gres=gpu:a40:4) or more time (--time=3-00:00:00).
# An A40, the card Dynamo's own GPU runs were asked for (the timings compare);
# more with --gres=gpu:a40:4 and the particles are dealt among all the job's GPUs.
# Measured on one A40 (2026-10-27, scripts/torch_search_profile.sbatch, RyR1 bin8,
# box 52): one process searches a particle of cone5's first round (300 candidates)
# in 0.082 s, so a round of 2,384 particles x 3 channels is about 10 minutes of
# searching on one card. The GPU computes 87% of that time, so PER_GPU 2 or 4 gives
# nothing there. The cone5 run on two A40s took 27 minutes an iteration because its
# two processes waited on each other's CPU thread teams; GPU processes now keep one
# thread (dynamo_search_torch._start_worker). Run again so (mra_pad_ssn_cone5_port2,
# two A40s): 7.0 minutes an iteration in rounds 1-2, 5.4 in rounds 3-5, 32 minutes in
# all, the results bit for bit the first run's. The profile's rate gives 4.8 and 3.6
# minutes of searching; to see where the rest goes, each iteration's line says how
# much of it was the search, and the cards' clocks and load are sampled once a minute
# into slurm-<job name>-<job id>-gpus.csv and summarized at the end.

set -euo pipefail

: "${PROJECT:?set PROJECT to the Dynamo project to run again}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/mra_port_run.py ]] || { echo "submit from the repo root" >&2; exit 1; }
source scripts/activate_env.sh
echo "node        $(hostname)"
echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
echo "started     $(date -Is)"
name=${NAME:-$(basename "$PROJECT")_port}
cards="slurm-${SLURM_JOB_NAME:-mra_port_run}-${SLURM_JOB_ID:-local}-gpus.csv"
nvidia-smi --format=csv,noheader,nounits -l 60 \
    --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,power.draw \
    > "$cards" 2>&1 &
sampler=$!
python scripts/mra_port_run.py "$PROJECT" --name "$name" --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-float32}" --per-gpu "${PER_GPU:-1}"
{ kill "$sampler" && wait "$sampler"; } 2>/dev/null || true
echo "finished    $(date -Is)"
# The samples taken while a card was at least half busy: did it hold its clock?
awk -F', ' '$2 ~ /^[0-9]+$/ && $5 + 0 >= 50 {
        g = $2; n[g]++; clock[g] += $3; busy[g] += $5; top[g] = $4 + 0
        if (!(g in low) || $3 + 0 < low[g]) low[g] = $3 + 0
        if ($6 + 0 > hot[g]) hot[g] = $6 + 0
        if ($7 + 0 > watts[g]) watts[g] = $7 + 0
    }
    END {
        for (g in n) printf "gpu %s, %d samples while busy: SM clock %.0f MHz on average, " \
            "%d at the lowest, %d its top; %.0f%% busy; up to %d C and %.0f W\n", g, n[g],
            clock[g] / n[g], low[g], top[g], busy[g] / n[g], hot[g], watts[g]
    }' "$cards" | sort || echo "no gpu summary" >&2
echo "gpu samples $cards"
if compgen -G "$PROJECT/results/ite_*/averages/refined_table_ref_*_ite_*.tbl" > /dev/null; then
    python scripts/compare_dynamo_classes.py "$PROJECT" "$(dirname "$PROJECT")/$name" \
        || echo "no class comparison" >&2
fi
python scripts/dynamo_classification_report.py "$(dirname "$PROJECT")/$name" \
    || echo "no report; write it with: python scripts/dynamo_classification_report.py" \
        "$(dirname "$PROJECT")/$name" >&2
