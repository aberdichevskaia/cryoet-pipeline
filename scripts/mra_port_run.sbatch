#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.log
#
# An MRA classification computed by the pipeline's port on the MDC cluster (max),
# on four A40s, with its classification report. A new run is one command from the
# repo root, with the variables scripts/dynamo_mra_mdc.sbatch takes for Dynamo:
#
#   EXPORT=outputs/ryr1/dynamo/bin8-2x-pad NAME=mra_pad_ssn_g2atan2 \
#       SEEDS="subset subset noise" SCHEDULE=coarse_to_fine FIRST_LOWPASS=53.76 \
#       LAST_LOWPASS=29.4 DIAMETER=340 GLOBAL_ROUNDS=2 \
#       sbatch --export=ALL --job-name=mra_pad_ssn_g2atan2 scripts/mra_port_run.sbatch
#
# (g2atan's own variables, which Dynamo ran on 32 cores as mra_pad_ssn_g2atan.)
# The job makes the Dynamo project $EXPORT/$NAME unless it is there (Dynamo's own
# setup: scripts/dynamo_mra_mdc.sbatch with SETUP_ONLY=1, run inside this job while
# the cards wait, not yet timed on max; it averages 300 particles for each subset or
# blob seed and writes every iteration's card, the plan the port follows), computes
# it with the port as $EXPORT/${NAME}_port, and writes
# $EXPORT/${NAME}_port_classification_report.pdf, TEMPLATE drawn in it as in
# Dynamo's reports. The project is made for 32 cores, so Dynamo can compute the
# same one for a comparison (scripts/dynamo_mra_mdc.sbatch, the same EXPORT and
# NAME). GPUS, Dynamo's own GPU setting, is refused here: the port searches on
# every GPU the job has.
#
# A job that runs out of time, or stops, is submitted again as it was: the project
# is not made again, and the port carries on from its last finished iteration
# (`iteration N ok` in ${NAME}_port/log.txt). Every iteration starts from what the
# one before it wrote, as Dynamo's do, so the run ends as an unstopped one would.
# A run that finished is only reported again.
#
# Or PROJECT=<a Dynamo project>, set up or computed (instead of EXPORT): the port
# runs it again beside it as <project>_port (NAME names it otherwise), and when
# Dynamo computed the project, compares the two iteration by iteration and their
# classes with chance levels; so were the mirrors of cone5, g2full and g2atan made:
#
#   PROJECT=outputs/ryr1/dynamo/bin8-2x-pad/mra_pad_ssn_g2atan \
#       sbatch --export=ALL --job-name=mra_pad_ssn_g2atan_port scripts/mra_port_run.sbatch
#
# DEVICE (cuda), DTYPE (float32) and PER_GPU (processes sharing each GPU, 1) pass
# through; fewer cards with --gres=gpu:a40:1, and more time with it. DEVICE=numpy
# runs the numpy kernel on the job's CPUs instead.
#
# Measured on A40s (RyR1 bin8, 2,384 particles, 3 channels, box 52). g2atan
# (mra_pad_ssn_g2atan_port, 2026-10-28, four cards): the two 360 deg rounds at dim
# 30 and 34 took 51 and 105 minutes, the three cones 3.4 minutes each, 2 h 48 min
# in all, against Dynamo's 30 h 20 min on 32 cores; a card searched a candidate in
# 0.077 ms at dim 30, 0.106 ms at dim 34 and 0.26-0.28 ms at the full box
# (full_g1; g2full_port, its global rounds at 52, 7.7 h on four cards). One card
# would take about 10.7 h for g2atan. About 1.5 minutes of every iteration are
# outside the search, the class averages on the main process's one CPU. The
# particles are held in memory as float64 (2,384 at box 52 take 2.7 GB). The
# cards' clocks and load are sampled once a minute into
# slurm-<job name>-<job id>-gpus.csv and summarized at the end: on g2atan every
# busy sample held 1740 MHz.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/mra_port_run.py ]] || { echo "submit from the repo root" >&2; exit 1; }
if [[ -n "${EXPORT:-}" ]]; then
    [[ -z "${PROJECT:-}" ]] || { echo "give EXPORT and NAME for a new run, or PROJECT; not both" >&2; exit 1; }
    : "${NAME:?set NAME to the name of the project}"
    if [[ -n "${GPUS:-}" ]]; then
        echo "GPUS is Dynamo's own GPU setting; the port searches on every GPU of the job" \
             "(--gres=gpu:a40:N)" >&2
        exit 1
    fi
    PROJECT=$EXPORT/$NAME
    name=${NAME}_port
    if [[ ! -f "$PROJECT/cards/ite_0001/card_ite_ite_0001.card" ]]; then
        # Made for 32 cores, which the setup reads from SLURM_CPUS_PER_TASK, whatever
        # this job has; a refusal of the setup ends this job, on this log.
        SETUP_ONLY=1 SLURM_CPUS_PER_TASK=32 bash scripts/dynamo_mra_mdc.sbatch
    else
        echo "set up      $PROJECT was made before: the port follows its cards, and the" \
             "setup's variables are not read"
    fi
else
    : "${PROJECT:?set PROJECT to a Dynamo project, or EXPORT and NAME for a new run}"
    name=${NAME:-$(basename "$PROJECT")_port}
fi
run=$(dirname "$PROJECT")/$name
source scripts/activate_env.sh
echo "node        $(hostname)"
echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
echo "started     $(date -Is)"
echo "project     $PROJECT"
echo "port run    $run"
cards="slurm-${SLURM_JOB_NAME:-mra_port_run}-${SLURM_JOB_ID:-local}-gpus.csv"
nvidia-smi --format=csv,noheader,nounits -l 60 \
    --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,power.draw \
    > "$cards" 2>&1 &
sampler=$!
python scripts/mra_port_run.py "$PROJECT" --name "$name" --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-float32}" --per-gpu "${PER_GPU:-1}" --workers "${SLURM_CPUS_PER_TASK:-1}"
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
    python scripts/compare_dynamo_classes.py "$PROJECT" "$run" || echo "no class comparison" >&2
fi
report_options=()
[[ -n "${TEMPLATE:-}" ]] && report_options+=(--template "$TEMPLATE")
if python scripts/dynamo_classification_report.py "$run" \
        ${report_options[@]+"${report_options[@]}"}; then
    echo "report      ${run}_classification_report.pdf"
else
    echo "no report; write it with: python scripts/dynamo_classification_report.py $run" \
         "${TEMPLATE:+--template $TEMPLATE}" >&2
fi
