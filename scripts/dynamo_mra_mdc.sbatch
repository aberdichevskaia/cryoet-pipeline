#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.log
#
# One Dynamo multireference (MRA) project on the MDC cluster (max): make it in an
# export folder of scripts/export_dynamo.py with scripts/dynamo_mra_setup.m,
# compute it with Dynamo standalone on this job's cores, then write its
# classification report beside it. Submit from the repo root:
#
#   EXPORT=<export folder> NAME=mra_nss SEEDS="subset subset noise" \
#       sbatch --export=ALL --job-name=mra_nss scripts/dynamo_mra_mdc.sbatch
#
# SEEDS lists one starting reference per channel: subset, blob or noise (see the
# setup script). SCHEDULE=coarse_to_fine searches in 5 rounds from a 40 A
# low-pass to a 22 A one, the angular step following the resolution;
# FIRST_LOWPASS and LAST_LOWPASS (A) change its ends, DIAMETER (A, the
# particle's widest, 290 unless given) its steps; the steps are equal in A, or in
# frequency with LOWPASS_STEPS=fourier. The alignment and classification mask is
# the particle's radius from DIAMETER plus 3 px with a 2 px soft edge, unless
# MASK_RADIUS and MASK_EDGE (px) say otherwise, or MASK_FILE gives a mask of any
# shape in the particles' box (scripts/make_mra_mask.py makes one that leaves a
# membrane out; a relative path is taken from the repo root). SYMMETRY (c4, ...)
# is imposed about the box's z axis in every round, c1 unless given. Without
# SCHEDULE the search is the fixed one. A project
# that already exists in EXPORT is not made again: the job goes on computing it
# from the iteration Dynamo has recorded as next, so a job that ran out of time
# is resubmitted as it was.
#
# Dynamo is the group's shared installation, activated with its shipped MATLAB
# Runtime 9.9. dynamo_activate_linux.sh alone tries to compile OpenMP wrappers
# inside the read-only installation and fails, and without those wrappers only
# destination standalone works, which the setup script asks for. Each job keeps
# the Runtime's cache in a folder of its own on /fast: max's /tmp holds 100 MB.
#
# Start each of several jobs a quarter of an hour after the previous one starts
# (sbatch --dependency=after:<job>+15). Every Dynamo parallel pool keeps its job
# files in the one folder the Runtime keeps under ~/.matlab. On the Mac two pools
# started together failed to open there (2026-10-16) and left a job record that
# stopped every later pool until it was moved aside; a separate HOME per job only
# stopped every pool from opening. A job whose pool does not open stops with a
# message, computes nothing, and lists the pool job records MATLAB cannot read;
# submitted again, it carries on where the project stopped.

set -euo pipefail

: "${EXPORT:?set EXPORT to the export folder (particles/, initial.tbl)}"
: "${NAME:?set NAME to the project name}"
REPO=${SLURM_SUBMIT_DIR:-$PWD}
DYNAMO_ROOT=${DYNAMO_ROOT:-/fast/AG_Kudryashev/artsemi/software/Dynamo}
CORES=${SLURM_CPUS_PER_TASK:-8}
JOB_DIR=${SCRATCH_ROOT:-/fast/AG_Kudryashev/anna.berdichevskaia/tmp}/dynamo_${NAME}_${SLURM_JOB_ID:-$$}
[[ -f "$REPO/scripts/dynamo_mra_setup.m" ]] || { echo "submit from the repo root" >&2; exit 1; }
if [[ -n "${MASK_FILE:-}" ]]; then
    [[ "$MASK_FILE" = /* ]] || MASK_FILE=$REPO/$MASK_FILE
    [[ -f "$MASK_FILE" ]] || { echo "no MASK_FILE $MASK_FILE" >&2; exit 1; }
fi

export MCR_CACHE_ROOT=$JOB_DIR/mcr_cache
export TMPDIR=$JOB_DIR/tmp
mkdir -p "$MCR_CACHE_ROOT" "$TMPDIR"

# Not written for set -eu: the activation tests unset arguments, and the generic
# script it calls tries the wrapper compilation that fails here.
set +eu
source "$DYNAMO_ROOT/dynamo_activate_linux_shipped_MCR.sh" > "$JOB_DIR/activation.log" 2>&1
set -eu
for binary in dynamo dynamo_execute_project; do
    if ! command -v "$binary" > /dev/null; then
        echo "no $binary on PATH after activating $DYNAMO_ROOT; see $JOB_DIR/activation.log" >&2
        exit 1
    fi
done

cd "$EXPORT"
echo "node        $(hostname)"
echo "started     $(date -Is)"
echo "project     $EXPORT/$NAME, $CORES cores"
echo "dynamo      $(command -v dynamo_execute_project)"

card=$NAME/cards/ite_0001/card_ite_ite_0001.card
if [[ ! -d "$NAME" ]]; then
    : "${SEEDS:?set SEEDS, e.g. SEEDS=\"subset subset noise\", to make the project}"
    cells=$(printf "'%s', " $SEEDS)
    options=""
    [[ -n "${SCHEDULE:-}" ]] && options+="schedule = '$SCHEDULE'; "
    [[ -n "${FIRST_LOWPASS:-}" ]] && options+="first_lowpass = $FIRST_LOWPASS; "
    [[ -n "${LAST_LOWPASS:-}" ]] && options+="last_lowpass = $LAST_LOWPASS; "
    [[ -n "${DIAMETER:-}" ]] && options+="diameter = $DIAMETER; "
    [[ -n "${LOWPASS_STEPS:-}" ]] && options+="lowpass_steps = '$LOWPASS_STEPS'; "
    [[ -n "${MASK_RADIUS:-}" ]] && options+="mask_radius = $MASK_RADIUS; "
    [[ -n "${MASK_EDGE:-}" ]] && options+="mask_edge = $MASK_EDGE; "
    [[ -n "${MASK_FILE:-}" ]] && options+="mask_file = '$MASK_FILE'; "
    [[ -n "${SYMMETRY:-}" ]] && options+="symmetry = '$SYMMETRY'; "
    dynamo "project = '$NAME'; seeds = {${cells%, }}; cores = $CORES; ${options}eval(fileread('$REPO/scripts/dynamo_mra_setup.m'))" 2>&1 | tee "${NAME}_setup.log"
    if [[ ! -f "$card" ]]; then
        echo "the setup did not make $NAME; see $EXPORT/${NAME}_setup.log" >&2
        exit 1
    fi
fi
made_for=$(sed -n 's/^how_many_processors=\([0-9]*\);$/\1/p' "$card")
if [[ "$made_for" != "$CORES" ]]; then
    echo "$NAME was made for $made_for cores and this job has $CORES; submit with" \
         "--cpus-per-task=$made_for" >&2
    exit 1
fi

log=${NAME}_run_${SLURM_JOB_ID:-$$}.log
times=${NAME}_iteration_times_${SLURM_JOB_ID:-$$}.txt
: > "$times"

# Dynamo's output passed on unchanged, plus a line for every iteration it
# finishes saying how long that took by Dynamo's own clock, from "Starting
# iteration" to "Iteration finished" (the MRA plugin's few seconds after it are
# left out). Its stamps rather than when the lines arrive, so buffering in the
# pipe cannot skew them. Clock times only: an iteration past midnight is counted
# right, one longer than a day would not be.
iteration_times() {
    local line round="?" start="" end="" seconds
    while IFS= read -r line; do
        printf '%s\n' "$line"
        if [[ $line =~ ^Iteration\ [0-9]+\ corresponding\ to\ round\ ([0-9]+) ]]; then
            round=${BASH_REMATCH[1]}
        elif [[ $line =~ ^Starting\ iteration\ at\ .*\ ([0-9]{2}):([0-9]{2}):([0-9]{2}) ]]; then
            start=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 \
                     + 10#${BASH_REMATCH[3]}))
        elif [[ $line =~ ^Iteration\ finished\ at\ .*\ ([0-9]{2}):([0-9]{2}):([0-9]{2}) ]]; then
            end=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
        elif [[ $line =~ ^iteration\ ([0-9]+)\ ok ]] && [[ -n $start && -n $end ]]; then
            seconds=$(( (end - start + 86400) % 86400 ))
            printf '>>> iteration %s (round %s) took %d min %02d s\n' "${BASH_REMATCH[1]}" \
                "$round" $((seconds / 60)) $((seconds % 60)) | tee -a "$times"
            start=""
            end=""
        fi
    done
}

dynamo_execute_project "$NAME" 2>&1 | iteration_times | tee "$log"
echo "finished    $(date -Is)"
if [[ -s "$times" ]]; then
    echo "iteration times, by Dynamo's clock:"
    cat "$times"
    awk '{ total += $(NF - 3) * 60 + $(NF - 1) }
         END { printf "together %d h %02d min\n", total / 3600, total % 3600 / 60 }' "$times"
fi
if grep -q "Cannot parallelize the project" "$log"; then
    echo "Dynamo could not open its parallel pool and computed nothing; submit again later" >&2
    # Two pools started together can leave a pool job record that blocks every pool
    # after it: on the Mac one Job*.metadata.txt read 'concurrentconcurrent', and
    # MATLAB then failed with "Unknown type" until that job's files were moved away.
    odd=$(find "$HOME/.matlab" -path '*local_cluster_jobs*' -name 'Job*.metadata.txt' \
          -exec grep -L -x -E 'concurrent|independent|communicating' {} + 2>/dev/null || true)
    if [[ -n "$odd" ]]; then
        echo "pool job records that MATLAB cannot read; move each one's Job<N>* files aside:" >&2
        echo "$odd" >&2
    fi
    exit 1
fi
grep -E "contains [0-9]+ tags|No tags assigned|iteration [0-9]+ ok" "$log" | tail -n 20 || true

# The report needs the pipeline's env; the classes are on disk whether or not it works.
# A subshell, because activate_env.sh exits when it finds no env manager.
cd "$REPO"
if (source scripts/activate_env.sh && python scripts/dynamo_classification_report.py "$EXPORT/$NAME"); then
    echo "report      $EXPORT/${NAME}_classification_report.pdf"
else
    echo "no report; write it with: python scripts/dynamo_classification_report.py $EXPORT/$NAME" >&2
fi
