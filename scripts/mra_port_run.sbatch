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
#   EXPORT=outputs/ryr1/dynamo/bin8-2x-pad NAME=mra_pad_ssn_g2atan3 \
#       SEEDS="subset subset noise" SCHEDULE=coarse_to_fine FIRST_LOWPASS=53.76 \
#       LAST_LOWPASS=29.4 DIAMETER=340 GLOBAL_ROUNDS=2 \
#       sbatch --export=ALL --job-name=mra_pad_ssn_g2atan3 scripts/mra_port_run.sbatch
#
# (g2atan's own variables, which Dynamo ran on 32 cores as mra_pad_ssn_g2atan.)
# Or, with no export at all, from the particle stacks of a pipeline project's
# registry, read where they are (docs/mra_from_registry.md): SOURCE, the project;
# LANE, the stacks' label (the lane's variant); SERIES, the tilt-series in the order
# to number them; DECOY_LANE for decoy stacks after them; the run is
# $SOURCE/mra/$NAME:
#
#   SOURCE=outputs/ryr1 LANE=bin8 SERIES="039 096 097 125 161" NAME=mra_ssn \
#       SEEDS="subset subset noise" SCHEDULE=coarse_to_fine FIRST_LOWPASS=53.76 \
#       LAST_LOWPASS=29.4 DIAMETER=340 \
#       sbatch --export=ALL --job-name=mra_ssn scripts/mra_port_run.sbatch
#
# Its particles, tags and starting table are the ones an export of the same LANE and
# SERIES gives (one conversion, cryoet_pipeline.particle_sources); the setup's record
# names each stack it read. The rest of this header says EXPORT where SOURCE works the
# same. Dynamo takes no part. The job makes the run's plan from EXPORT and those variables
# with the pipeline's setup (cryoet_pipeline.mra_setup; until 2026-11-02 Dynamo's,
# scripts/dynamo_mra_setup.m, ran inside this job): the same particles in each subset
# or blob seed, averaged as dynamo_average averages, the same mask and the same
# search in every iteration, but new noise in a noise seed (MATLAB's randn is not
# numpy's) and the export's own digits in the starting table, where Dynamo's were
# rounded to five; scripts/compare_mra_setup.py checks a plan against a project
# Dynamo's setup made. It then computes the plan with the port as $EXPORT/$NAME, in
# the run-record layout (docs/run_records.md): plan.json, then setup/, ite_0001/, ...,
# each with its run.json, the poses as Parquet and the maps as MRC; and writes
# $EXPORT/${NAME}_classification_report.pdf, TEMPLATE drawn in it as in Dynamo's
# reports. A run begun before 2026-11-05 in Dynamo's layout, $EXPORT/${NAME}_port
# with its settings/setup.json, is carried on in that layout under that name.
# GPUS, Dynamo's own GPU setting, is refused here: the port
# searches on every GPU the job has; so are SYMMETRY (but c1) and AXIS_OFFSET, which
# the port does not model. A NAME that Dynamo has a project of in EXPORT, or a setup
# under way (its ${NAME}_seeds and the rest), is refused for a new run: that
# project's port run is made and carried on with PROJECT, below.
#
# A job that runs out of time, or stops, is submitted again as it was: the plan is
# made again from the same variables and must be the run's (each finished node's
# run.json holds the fingerprint of what it was computed from: the setup's variables
# and the export's table and manifest, each iteration's search), and the port carries
# on after its last finished node; a node of another plan is refused. (A run in
# Dynamo's layout is checked as it was: seeds and mask to float32's last bits, the
# starting table and every finished iteration's search exactly, `iteration N ok` in
# its log.txt.) Every iteration starts from what the
# one before it wrote, as Dynamo's do, so the run ends as an unstopped one would:
# g2atan2's first job ran out of time in iteration 2 (2026-10-29), the next carried
# it on from iteration 1 and agreed with Dynamo's g2atan as the unstopped mirror
# did, iteration by iteration. A run that finished is only reported again.
#
# One job at a time on a run. The job holds a lock on <run>.lock, beside the run
# folder ($NAME.lock for a new run: the lock scripts/dynamo_mra_mdc.sbatch takes for a
# Dynamo project of that NAME, so neither starts while the other runs), from the
# setup to the report
# (scripts/hold_run_lock.py). A second job of the same run, started while the
# first is on it, stops at once with REFUSED, the first job's number, node and
# start, having run nothing; one that waited in the queue and starts after the
# first has ended carries the run on. (On 2026-10-29 two jobs carrying g2atan2 on,
# submitted a minute apart, computed iterations 2-5 into the same files for two
# hours; the folder held one run in the end only because both computed the same
# numbers.) On max (2026-10-30) the lock kept out a job on another node (see
# scripts/dynamo_mra_mdc.sbatch, which locks with the same holder), and a port job
# on maxg22 ran itself again under its lock from SLURM's copy of this script.
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

# This script, to run again under the run's lock below: SLURM runs a copy of it,
# by an absolute path in its spool; run by hand, it is found from here.
script=$0
[[ "$script" = /* ]] || script=$PWD/$script
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f scripts/mra_port_run.py ]] || { echo "submit from the repo root" >&2; exit 1; }
if [[ -n "${SOURCE:-}" || -n "${EXPORT:-}" ]]; then
    if [[ -n "${PROJECT:-}" || ( -n "${SOURCE:-}" && -n "${EXPORT:-}" ) ]]; then
        echo "give SOURCE or EXPORT, with NAME, for a new run, or PROJECT; one of them" >&2
        exit 1
    fi
    : "${NAME:?set NAME to the name of the run}"
    : "${SEEDS:?set SEEDS, e.g. SEEDS=\"subset subset noise\", to set the run up}"
    if [[ -n "${GPUS:-}" ]]; then
        echo "GPUS is Dynamo's own GPU setting; the port searches on every GPU of the job" \
             "(--gres=gpu:a40:N)" >&2
        exit 1
    fi
    if [[ -n "${SYMMETRY:-}" && "$SYMMETRY" != c1 ]] || [[ -n "${AXIS_OFFSET:-}" ]]; then
        echo "SYMMETRY and AXIS_OFFSET are Dynamo's; the port models no symmetry" >&2
        exit 1
    fi
    if [[ -n "${SOURCE:-}" ]]; then
        : "${SERIES:?set SERIES to the tilt-series to read, in order, e.g. SERIES=\"TS_01 TS_43\"}"
        name=$NAME
        run=$SOURCE/mra/$name
        mkdir -p "$SOURCE/mra"
    # A run begun in Dynamo's layout before 2026-11-05 is carried on in it, under its
    # name; every other run is NAME, in the run-record layout.
    elif [[ -f "$EXPORT/${NAME}_port/settings/setup.json" && ! -e "$EXPORT/$NAME/plan.json" ]]
    then
        name=${NAME}_port
        run=$EXPORT/$name
    else
        name=$NAME
        run=$EXPORT/$name
    fi
else
    : "${PROJECT:?set PROJECT to a Dynamo project, or EXPORT and NAME for a new run}"
    name=${NAME:-$(basename "$PROJECT")_port}
    run=$(dirname "$PROJECT")/$name
fi
# The rest of the job runs again under the run's lock, the setup, the port and the
# report: a second job of the same run stops here, naming the first. The pipeline's
# env only gives the lock its python here; the job activates it after.
if [[ "${MRA_PORT_RUN_LOCK:-}" != "$run.lock" ]]; then
    export MRA_PORT_RUN_LOCK=$run.lock
    holder=$(source scripts/activate_env.sh > /dev/null && command -v python) \
        || { echo "REFUSED: no pipeline env found to hold $run.lock with" >&2; exit 1; }
    exec "$holder" scripts/hold_run_lock.py "$run.lock" -- bash "$script"
fi
if [[ -n "${EXPORT:-}" && ! -f "$run/plan.json" && ! -f "$run/settings/setup.json" ]]; then
    # A new run. Dynamo's own files of the name belong to a Dynamo project, or to a
    # setup a Dynamo job is making, whose port run is PROJECT's.
    dynamo=()
    for item in "$NAME" "${NAME}_seeds" "${NAME}_fmasks" "${NAME}_tables" "${NAME}.vpr"; do
        if [[ -e "$EXPORT/$item" ]]; then dynamo+=("$item"); fi
    done
    if (( ${#dynamo[@]} )); then
        {
            echo "REFUSED: $EXPORT holds Dynamo's ${dynamo[*]}, a Dynamo project of this name"
            echo "or its setup. Its port run is made, or carried on, with PROJECT=$EXPORT/$NAME;"
            echo "a new run needs another NAME."
        } >&2
        exit 1
    fi
fi
source scripts/activate_env.sh
echo "lock        $run.lock, held by this job to its end"
echo "node        $(hostname)"
echo "gpus        ${CUDA_VISIBLE_DEVICES:-none visible}: $(nvidia-smi -L 2>&1 | paste -sd ';' -)"
echo "started     $(date -Is)"
if [[ -n "${SOURCE:-}" ]]; then
    echo "source      $SOURCE, label ${LANE:-none}, series $SERIES${DECOY_LANE:+, decoys $DECOY_LANE}"
    read -ra series <<< "$SERIES"
    from=(--source "$SOURCE" --seeds "$SEEDS" --series "${series[@]}")
    if [[ -n "${LANE:-}" ]]; then from+=(--lane "$LANE"); fi
    if [[ -n "${DECOY_LANE:-}" ]]; then from+=(--decoy-lane "$DECOY_LANE"); fi
elif [[ -n "${EXPORT:-}" ]]; then
    echo "export      $EXPORT, set up by the pipeline"
    from=(--export "$EXPORT" --seeds "$SEEDS")
else
    echo "project     $PROJECT"
    from=("$PROJECT")
fi
if [[ -n "${SOURCE:-}${EXPORT:-}" ]]; then
    for pair in SCHEDULE:--schedule FIRST_LOWPASS:--first-lowpass LAST_LOWPASS:--last-lowpass \
            DIAMETER:--diameter LOWPASS_STEPS:--lowpass-steps MASK_RADIUS:--mask-radius \
            MASK_EDGE:--mask-edge MASK_FILE:--mask-file REFERENCE:--reference \
            GLOBAL_ROUNDS:--global-rounds GLOBAL_STEP:--global-step CONE_STEPS:--cone-steps \
            DATA_FOLDER:--data-folder; do
        variable=${pair%%:*}
        if [[ -n "${!variable:-}" ]]; then from+=("${pair#*:}" "${!variable}"); fi
    done
    case "${GLOBAL_FULL_BOX:-}" in
        "" | 0 | false) ;;
        *) from+=(--global-full-box) ;;
    esac
fi
echo "port run    $run"
cards="slurm-${SLURM_JOB_NAME:-mra_port_run}-${SLURM_JOB_ID:-local}-gpus.csv"
nvidia-smi --format=csv,noheader,nounits -l 60 \
    --query-gpu=timestamp,index,clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,power.draw \
    > "$cards" 2>&1 &
sampler=$!
python scripts/mra_port_run.py "${from[@]}" --name "$name" --device "${DEVICE:-cuda}" \
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
if [[ -n "${PROJECT:-}" ]] \
        && compgen -G "$PROJECT/results/ite_*/averages/refined_table_ref_*_ite_*.tbl" > /dev/null; then
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
