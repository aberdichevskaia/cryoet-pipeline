#!/bin/bash
#SBATCH --output=slurm-%x-%j.log
#
# One stage of one lane: scripts/run_series.py over SERIES, steps FROM..UNTIL.
# Partition, GPU, time and job name come from the sbatch command line, so the
# same file serves the CPU and the GPU stages; scripts/submit_lanes.py submits
# them chained with afterok, from the [run] section of the lane config.
#
# Required in the environment (sbatch --export):
#   PROJECT  CONFIG  SERIES (space-separated)  FROM  UNTIL
# and, for the stages that use them, MOTIONCOR3, MEMBRAIN_EXE, MEMBRAIN_CKPT,
# PYTOMO_MP_DIR. OVERWRITE=1 adds --overwrite to every step, for re-running a
# step that already wrote part of its output.
#
# Jobs of one project may run at the same time, one per series: each command
# merges its own entries into the registry under a lock when it finishes
# (ArtifactRegistry.commit), so jobs no longer erase each other's. Two jobs on
# the SAME series must still not overlap: they write the same output files, and
# the later registry commit is refused. Inside one job, series run in turn.
#
# Disposable, like scripts/run_series.py.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
cd "$REPO"

for name in PROJECT CONFIG SERIES FROM UNTIL; do
    if [[ -z "${!name:-}" ]]; then
        echo "$name is not set; submit through scripts/submit_lanes.py" >&2
        exit 1
    fi
done

export IMOD_DIR=/ictstr01/groups/shared/users/peng_marr/a.berdichevskaia/apps/IMOD/imod_5.1.12

# `set -u` is for our code, not for environment scripts that were not written
# for it (IMOD-linux.sh appends to an unset MANPATH; conda's hook likewise).
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline
source "$IMOD_DIR/IMOD-linux.sh"
set -u
# MotionCor3 needs the env's libtiff.so.6.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

STEPS=$(python3 scripts/run_series.py --project "$PROJECT" --config "$CONFIG" \
    --series X --from "$FROM" --until "$UNTIL" --list-steps-range)

echo "job         $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname)"
echo "started     $(date -Is)"
echo "project     $PROJECT   config $CONFIG"
echo "series      $SERIES"
echo "steps       $(echo $STEPS)"
echo "cryoet      $(command -v cryoet)"
echo "imod        $(command -v tiltxcorr || echo 'not on PATH')"
echo "ctffind     $(command -v ctffind5 || command -v ctffind || echo 'not on PATH')"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true
# Inside a MIG slice the line above reports the whole card: job 41075712 had a
# 3g.20gb slice and printed 40960 MiB. -L lists the slice itself.
nvidia-smi -L 2>/dev/null || true
echo

# Preflight: each of these fails deep inside a tool otherwise, after the
# allocation has been paid for.
fail() { echo "preflight: $*" >&2; exit 1; }
needs() { grep -qx "$1" <<<"$STEPS"; }
gpu_cap() { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1; }
cuobjdump=$(command -v cuobjdump || echo /usr/local/cuda/bin/cuobjdump)
archs() { "$cuobjdump" "$1" "$2" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' '; }
# A cubin runs on its own major version at an equal or higher minor; PTX is
# compiled at load time for any GPU at least as new as its target. The cluster's
# MotionCor3 was once rebuilt for sm_61 alone, which an A100 cannot run.
runs_on() {  # runs_on CAP BINARY
    local cap=${1/./} arch
    for arch in $(archs -lelf "$2"); do
        arch=${arch#sm_}
        (( arch / 10 == cap / 10 && arch <= cap )) && return 0
    done
    for arch in $(archs -lptx "$2"); do
        (( ${arch#sm_} <= cap )) && return 0
    done
    return 1
}

if needs prepare-tilt-series; then
    [[ -x "${MOTIONCOR3:-}" ]] || fail "MOTIONCOR3 is not an executable: '${MOTIONCOR3:-}'"
    # The loader names only the first library it misses; ldd names them all.
    missing=$(ldd "$MOTIONCOR3" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
    [[ -z "$missing" ]] || fail "MotionCor3 cannot load: ${missing}- put their directory on LD_LIBRARY_PATH before submitting"
    cap=$(gpu_cap)
    if [[ -z "$cap" || ! -x "$cuobjdump" ]]; then
        echo "preflight: no compute capability or cuobjdump here; MotionCor3's GPU code unchecked"
    elif ! runs_on "$cap" "$MOTIONCOR3"; then
        fail "MotionCor3 has nothing this GPU (compute $cap) can run." \
            "cubins: $(archs -lelf "$MOTIONCOR3")ptx: $(archs -lptx "$MOTIONCOR3")" \
            "Rebuild it, or give the motion stage another GPU in the config's [[run.stage]]."
    fi
fi
if needs segment-tomogram; then
    [[ -x "${MEMBRAIN_EXE:-}" ]] || fail "MEMBRAIN_EXE is not an executable: '${MEMBRAIN_EXE:-}'"
    [[ -f "${MEMBRAIN_CKPT:-}" ]] || fail "MEMBRAIN_CKPT is not a file: '${MEMBRAIN_CKPT:-}'"
    cap=$(gpu_cap)
    if [[ -n "$cap" ]] && ! awk -v c="$cap" 'BEGIN { exit !(c + 0 >= 7.5) }'; then
        fail "MemBrain-seg needs compute capability >= 7.5, this GPU has $cap"
    fi
fi
if needs prepare-template || needs pick-particles; then
    [[ -n "${PYTOMO_MP_DIR:-}" ]] || fail "PYTOMO_MP_DIR is not set (note the O)"
    for tool in pytom_create_template pytom_create_mask pytom_match_template \
                pytom_extract_candidates; do
        [[ -f "$PYTOMO_MP_DIR/$tool.py" ]] || fail "missing $PYTOMO_MP_DIR/$tool.py"
    done
fi
if needs prepare-template; then
    emdb=$(python3 -c "import tomllib; print(tomllib.load(open('$CONFIG','rb'))['template']['emdb_id'].split('-')[1])")
    [[ -f "$PROJECT/emdb-cache/emd_${emdb}.map" ]] \
        || echo "warning: EMD-$emdb not cached in $PROJECT/emdb-cache; prepare-template will" \
                "try to download it, which needs outbound access from this node"
fi
if needs estimate-ctf; then
    command -v ctffind >/dev/null || command -v ctffind5 >/dev/null || fail "no ctffind on PATH"
fi
# Every step commits to the registry under a lock and refuses to without one that
# holds across nodes (no flock on this mount, or localflock); find that out here,
# not after the first step has run.
python3 - "$PROJECT/artifacts.json" <<'EOF' || fail "cannot lock the registry from this node (above)"
import sys
from pathlib import Path

from cryoet_pipeline.artifacts import registry_lock

with registry_lock(Path(sys.argv[1]), timeout=60):
    pass
EOF
echo "preflight   ok"
echo

read -ra SERIES_LIST <<<"$SERIES"
args=()
for series in "${SERIES_LIST[@]}"; do args+=(--series "$series"); done

python3 scripts/run_series.py \
    --project "$PROJECT" \
    --config "$CONFIG" \
    "${args[@]}" \
    --from "$FROM" \
    --until "$UNTIL" \
    ${OVERWRITE:+--overwrite} \
    --execute

echo
echo "finished    $(date -Is)"
