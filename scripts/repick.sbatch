#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%x-%j.log
#
# A lane's tomograms picked again under a new label on the MDC cluster (max), as
# the lane's picks were made, and compared with them and with the null decoys'
# search: scripts/repick.py over SERIES (the lane's pick-particles and
# extract-particles, then the comparison). With EXPORT, scripts/export_dynamo.py
# then writes the new picks followed by the null decoys, which were picked the same
# way. Submit from the repo root:
#
#   PROJECT=outputs/ribo-mdc CONFIG=configs/lanes/ribo-mdc.toml \
#   SERIES="TS_0306 TS_0463 TS_0493 TS_0564 TS_0623" LABEL=bin8-solved \
#       sbatch --export=ALL --job-name=repick scripts/repick.sbatch
#
# Partition, card and time are the lane's GPU stage (configs/lanes/ribo-mdc.toml),
# for the same five pytom searches. Optional: EXPORT (a folder that holds nothing
# yet; written only when every series got through), NULL_LABEL (the lane's label
# plus -null), OVERWRITE=1 (pick again under LABEL). Needs PYTOMO_MP_DIR, as the
# lane does. A job that stops on a series has finished the ones before it:
# resubmit with SERIES from that series on and without EXPORT, then print the whole
# comparison with
#
#   python scripts/repick.py --project ... --config ... --series ... --label ... --compare-only

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-$PWD}
[[ -f "$REPO/scripts/repick.py" ]] || { echo "submit from the repo root" >&2; exit 1; }
cd "$REPO"
: "${PROJECT:?set PROJECT, e.g. outputs/ribo-mdc}"
: "${CONFIG:?set CONFIG to the lane config the picks were made with}"
: "${SERIES:?set SERIES, space-separated}"
: "${LABEL:?set LABEL to the label of the new picks, e.g. bin8-solved}"
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
NULL_LABEL=${NULL_LABEL:-$LANE-null}
read -ra SERIES_LIST <<<"$SERIES"
series_args=()
for series in "${SERIES_LIST[@]}"; do series_args+=(--series "$series"); done
options=(--label "$LABEL" --null-label "$NULL_LABEL")
[[ "${OVERWRITE:-0}" == 1 ]] && options+=(--overwrite)

echo "job         ${SLURM_JOB_NAME:-repick} (${SLURM_JOB_ID:-local}) on $(hostname)"
echo "started     $(date -Is)"
echo "project     $PROJECT   config $CONFIG   lane $LANE -> $LABEL, null $NULL_LABEL"
echo "series      $SERIES"
echo "cryoet      $(command -v cryoet)"
nvidia-smi -L 2>/dev/null || true
echo

python scripts/repick.py --project "$PROJECT" --config "$CONFIG" "${series_args[@]}" \
    "${options[@]}" --execute

if [[ -n "${EXPORT:-}" ]]; then
    echo
    python scripts/export_dynamo.py --project "$PROJECT" --variant "$LABEL" \
        --decoy-variant "$NULL_LABEL" "${series_args[@]}" --out "$EXPORT"
fi
echo
echo "finished    $(date -Is)"
