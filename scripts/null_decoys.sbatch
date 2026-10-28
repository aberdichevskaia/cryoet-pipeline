#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.log
#
# Decoys chosen by template matching, for the classification control, on the MDC
# cluster (max): scripts/null_decoys.py over SERIES (per series a phase-randomised
# copy of the tomogram, the lane's pytom pick on it and the lane's extraction),
# then, with EXPORT, scripts/export_dynamo.py: every real particle of the lane, and
# after them these decoys. Submit from the repo root:
#
#   python3 scripts/submit_recipe.py null-decoys configs/lanes/ribo-mdc.toml \
#       --set EXPORT=outputs/ribo-mdc/dynamo/export_null --dry-run
#
# submit_recipe.py takes the lane's series unless --series names others, runs it
# where the lane picks, checks it first and journals it in the project. Given to
# sbatch by hand (--export=ALL, with PROJECT, CONFIG, SERIES and the options below
# in the environment) it runs the same, without any of that.
#
# Partition, card and time are the lane's GPU stage (configs/lanes/ribo-mdc.toml),
# for the same five pytom searches. Optional: COUNT (peaks kept per tomogram, 120),
# SEED (1), VARIANT (the lane's label plus -null), OVERWRITE=1 (make the copies and
# pick again; submit_recipe.py --overwrite sets it). Needs PYTOMO_MP_DIR, as the
# lane does. The export is written only when every series got through, into EXPORT,
# which must not hold anything yet. A job that stops on a series has finished the
# ones before it: resubmit with SERIES from that series on (its copy, if made, is
# used again) and without EXPORT, then run the export with all the series by hand;
# it takes minutes on a CPU.

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$PWD}
[[ -f "$REPO/scripts/null_decoys.py" ]] || { echo "submit from the repo root" >&2; exit 1; }
cd "$REPO"
: "${PROJECT:?set PROJECT, e.g. outputs/ribo-mdc}"
: "${CONFIG:?set CONFIG to the lane config the real picks were made with}"
: "${SERIES:?set SERIES, space-separated}"
: "${PYTOMO_MP_DIR:?set PYTOMO_MP_DIR (note the O), as for the lane}"
for tool in pytom_match_template pytom_extract_candidates; do
    [[ -f "$PYTOMO_MP_DIR/$tool.py" ]] || { echo "missing $PYTOMO_MP_DIR/$tool.py" >&2; exit 1; }
done
if [[ -n "${EXPORT:-}" && -d "$EXPORT" ]] && [[ -n "$(ls -A "$EXPORT")" ]]; then
    echo "$EXPORT already holds files; name a new EXPORT" >&2
    exit 1
fi

source scripts/activate_env.sh
# As in lane.sbatch: the env's libraries first.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

LANE=$(python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["lane"]["variant"])' "$CONFIG")
VARIANT=${VARIANT:-$LANE-null}
read -ra SERIES_LIST <<<"$SERIES"
series_args=()
for series in "${SERIES_LIST[@]}"; do series_args+=(--series "$series"); done
options=(--variant "$VARIANT")
[[ -n "${COUNT:-}" ]] && options+=(--count "$COUNT")
[[ -n "${SEED:-}" ]] && options+=(--seed "$SEED")
[[ "${OVERWRITE:-0}" == 1 ]] && options+=(--overwrite)

echo "job         ${SLURM_JOB_NAME:-null-decoys} (${SLURM_JOB_ID:-local}) on $(hostname)"
echo "started     $(date -Is)"
echo "project     $PROJECT   config $CONFIG   lane $LANE -> $VARIANT"
echo "series      $SERIES"
echo "overwrite   $([[ "${OVERWRITE:-0}" == 1 ]] && echo yes || echo no)"
echo "cryoet      $(command -v cryoet)"
nvidia-smi -L 2>/dev/null || true
echo

python scripts/null_decoys.py --project "$PROJECT" --config "$CONFIG" "${series_args[@]}" \
    "${options[@]}" --execute

if [[ -n "${EXPORT:-}" ]]; then
    echo
    python scripts/export_dynamo.py --project "$PROJECT" --variant "$LANE" \
        --decoy-variant "$VARIANT" "${series_args[@]}" --out "$EXPORT"
fi
echo
echo "finished    $(date -Is)"
