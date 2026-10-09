#!/bin/bash
#SBATCH --output=slurm-%x-%j.log
#
# The pipeline's tail for one lane, after the pick review: re-extract every
# series at its threshold from scripts/qc_dynamo_picks.py, then refine them all
# together with independent halves. Submit from the repo root, e.g.
#
#   PROJECT=outputs/ribo2 CONFIG=configs/lanes/ribo-bin8.toml \
#   SERIES="TS_0201 TS_0388 TS_0487 TS_0572 TS_0636" \
#   THRESHOLDS=outputs/ribo2/dynamo/pick-qc/thresholds.json MASK=381.6 \
#       sbatch --partition=cpu_p --qos=cpu_normal --cpus-per-task=32 --mem=64G \
#           --time=24:00:00 --job-name=ribo2-avg scripts/average_lane.sbatch
#
# MASK is the alignment mask: a diameter in A, or an MRC file in the particle
# box, which is used as Dynamo uses one, so a mask made for the Dynamo project
# can serve both. Optional: ITERATIONS (default 3), START_LOWPASS_ANGSTROM
# (the halves' common start; default 60), OVERWRITE=1, and VARIANT to average
# picks under another label than the lane's, such as the ones `cryoet
# filter-picks --output-variant` registers.
#
# The thresholds keep LCCmax >= the minimum, the rule the Dynamo selection
# uses, so the two branches start from the same particles. Box and variant come
# from the lane config. Disposable, like scripts/lane.sbatch.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
cd "$REPO"

for name in PROJECT CONFIG SERIES THRESHOLDS MASK; do
    if [[ -z "${!name:-}" ]]; then
        echo "$name is not set; see the header of this script" >&2
        exit 1
    fi
done

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline
set -u

lane() {  # lane SECTION KEY: one value of the lane config
    python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))[sys.argv[2]][sys.argv[3]])' \
        "$CONFIG" "$1" "$2"
}
threshold() {  # threshold SERIES: its minimum score, refusing the null placeholder
    python3 - "$THRESHOLDS" "$1" <<'EOF'
import json
import sys

value = json.load(open(sys.argv[1])).get(sys.argv[2])
if isinstance(value, bool) or not isinstance(value, (int, float)):
    sys.exit(f"{sys.argv[1]}: no numeric threshold for {sys.argv[2]}")
print(value)
EOF
}

box=$(lane extract box_size_px)
variant=${VARIANT:-$(lane lane variant)}
if [[ -f "$MASK" ]]; then
    mask=(--alignment-mask "$MASK")
else
    mask=(--alignment-mask-diameter-angstrom "$MASK")
fi

# Every threshold before any extraction, so a missing one stops the job here
# rather than after half the series were overwritten.
read -ra SERIES_LIST <<<"$SERIES"
minimums=()
for series in "${SERIES_LIST[@]}"; do minimums+=("$(threshold "$series")"); done

echo "job         $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname)"
echo "started     $(date -Is)"
echo "project     $PROJECT   variant $variant   box $box   mask ${mask[*]}"
echo

manifests=()
for i in "${!SERIES_LIST[@]}"; do
    series=${SERIES_LIST[$i]}
    echo "# $series: LCCmax >= ${minimums[$i]}"
    cryoet extract-particles \
        --manifest "$PROJECT/manifests/$series.json" \
        --registry "$PROJECT/artifacts.json" \
        --out "$PROJECT" \
        --box-size-px "$box" \
        --variant "$variant" \
        --minimum-score "${minimums[$i]}" \
        --overwrite
    manifests+=(--manifest "$PROJECT/manifests/$series.json")
done

echo
echo "# average-particles over ${#SERIES_LIST[@]} series"
cryoet average-particles \
    "${manifests[@]}" \
    --registry "$PROJECT/artifacts.json" \
    --out "$PROJECT" \
    --variant "$variant" \
    --gold-standard \
    --num-iterations "${ITERATIONS:-3}" \
    --shift-search-mode 1 \
    ${START_LOWPASS_ANGSTROM:+--gold-standard-start-lowpass-angstrom "$START_LOWPASS_ANGSTROM"} \
    "${mask[@]}" \
    ${OVERWRITE:+--overwrite}

echo
echo "finished    $(date -Is)"
