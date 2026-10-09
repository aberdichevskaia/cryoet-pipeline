#!/bin/bash
# Submit today's lanes as chained Slurm jobs, from the repo root on the submit node:
#
#   scripts/submit_lanes.sh ribo        # EMPIAR-11538, five series, lane ribo-bin8
#   scripts/submit_lanes.sh hiv         # EMPIAR-10164, TS_01 + TS_43, lane gag-bin8
#   scripts/submit_lanes.sh ribo hiv    # both; the two projects run side by side
#
# Each lane is a chain of stages joined by afterok, so a stage starts only when the
# previous one finished without a failed series: it needs that stage's outputs. The
# registry no longer forbids jobs of one project overlapping (each command merges its
# entries under a lock), but two jobs on the same series still must not. Before
# submitting: data downloaded, `init` run for hiv, EMDB maps cached — see the runbook.
#
# Set these before running (or export them in your shell):
#   MOTIONCOR3     MotionCor3 binary                      (hiv)
#   MEMBRAIN_EXE   membrain executable                    (hiv)
#   MEMBRAIN_CKPT  MemBrain-seg checkpoint, .ckpt         (hiv)
#   PYTOMO_MP_DIR  directory with the pytom_*.py scripts  (both; note the O)
# Optional: MOTION_GRES, the GPU for MotionCor3 (default: the A100 slice), if the
# MotionCor3 build has no kernels for it; lane.sbatch checks and says so.

set -euo pipefail

CPU=(--partition=cpu_p --qos=cpu_normal --cpus-per-task=8 --mem=32G)
GPU=(--partition=gpu_p --qos=gpu_normal --gres=gpu:a100_3g.20gb:1 --cpus-per-task=8 --mem=64G)
MOTION_GPU=(--partition=gpu_p --qos=gpu_normal --gres="${MOTION_GRES:-gpu:a100_3g.20gb:1}"
            --cpus-per-task=8 --mem=64G)

stage() {  # stage NAME DEPENDENCY PROJECT CONFIG SERIES FROM UNTIL TIME RESOURCES...
    local name=$1 dependency=$2 time=$8 id
    local after=()
    [[ -n "$dependency" ]] && after=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    # Through the environment rather than --export=VAR=value, whose list is
    # comma-separated: SERIES holds spaces, and nothing has to survive a parser.
    id=$(PROJECT=$3 CONFIG=$4 SERIES=$5 FROM=$6 UNTIL=$7 \
        sbatch --parsable --export=ALL "${@:9}" ${after[@]+"${after[@]}"} \
            --time="$time" --job-name="$name" scripts/lane.sbatch)
    echo "${id%%;*}"  # --parsable prints "id;cluster" on a multi-cluster setup
}

[[ $# -gt 0 ]] || { echo "usage: $0 ribo|hiv [ribo|hiv]" >&2; exit 1; }
: "${PYTOMO_MP_DIR:?set PYTOMO_MP_DIR}"

for lane in "$@"; do
    case "$lane" in
    ribo)
        P=outputs/ribo2 C=configs/lanes/ribo-bin8.toml
        S="TS_0201 TS_0388 TS_0487 TS_0572 TS_0636"
        r1=$(stage ribo2-cpu "" "$P" "$C" "$S" import-tilt-stack reconstruct-tomogram \
            10:00:00 "${CPU[@]}")
        r2=$(stage ribo2-gpu "$r1" "$P" "$C" "$S" prepare-template extract-particles \
            16:00:00 "${GPU[@]}")
        echo "ribo: cpu $r1 -> gpu $r2"
        ;;
    hiv)
        : "${MOTIONCOR3:?set MOTIONCOR3}" "${MEMBRAIN_EXE:?set MEMBRAIN_EXE}"
        : "${MEMBRAIN_CKPT:?set MEMBRAIN_CKPT}"
        P=outputs/hiv C=configs/lanes/gag-bin8.toml S="TS_01 TS_43"
        h1=$(stage hiv-motion "" "$P" "$C" "$S" prepare-tilt-series prepare-tilt-series \
            3:00:00 "${MOTION_GPU[@]}")
        h2=$(stage hiv-cpu "$h1" "$P" "$C" "$S" align-tilt-series reconstruct-tomogram \
            8:00:00 "${CPU[@]}")
        h3=$(stage hiv-gpu "$h2" "$P" "$C" "$S" segment-tomogram extract-particles \
            16:00:00 "${GPU[@]}")
        echo "hiv: motion $h1 -> cpu $h2 -> gpu $h3"
        ;;
    *)
        echo "unknown lane '$lane' (ribo or hiv)" >&2
        exit 1
        ;;
    esac
done
echo "watch: squeue -u \$USER   logs: slurm-<job name>-<id>.log"
