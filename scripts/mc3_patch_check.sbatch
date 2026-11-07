#!/bin/bash
#SBATCH --job-name=mc3-patch-check
#SBATCH --output=slurm-%x-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#
# Do MotionCor3's patches cut a tilt's sum into displaced tiles? Reruns a few
# tilts of a series exactly as the lane's motion step ran them (the command in
# its artifacts.json: same executable, gain, flip, FmIntFile and flags), once
# as they were and once with -Patch 1 1, and measures how far each tile of the
# patch grid moved. scripts/mc3_patch_check.py says how, and states the
# prediction for EMPIAR-11830 Position_1 before the run.
#
# Submit from the root of the clone the lane ran from, with the GPU flags of the
# lane's motion stage (the recorded MotionCor3 must run on the node), e.g. at MDC:
#
#     sbatch -p gpu --gres=gpu:1 \
#         --export=ALL,PROJECT=outputs/<lane>,SERIES=<tilt-series id> \
#         scripts/mc3_patch_check.sbatch
#
# Required: PROJECT (the lane's project folder, holding artifacts.json) and
# SERIES. Optional: ANGLES (default "-75 45 -15 -45 15": the two ends, the
# first tilt and two in between) and OUT (outputs/mc3-patch-check/<job id>).
# Two MotionCor3 runs per tilt, about 0.13 GB of sums per tilt under OUT.
# Nothing in PROJECT is written. The verdict is in the log, compare.png in OUT.
# The job ends 0 once the comparison is made, whatever the verdict; only a step
# that fails ends it otherwise. To judge an earlier OUT again, without a GPU:
#
#     python3 scripts/mc3_patch_check.py compare --out <OUT>
#
# Disposable.

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?not running under Slurm}
[[ -f "$REPO/scripts/mc3_patch_check.py" ]] \
    || { echo "submit from the root of a cryoet-pipeline clone, not $REPO" >&2; exit 1; }
cd "$REPO"

for name in PROJECT SERIES; do
    [[ -n "${!name:-}" ]] || { echo "$name is not set" >&2; exit 1; }
done
[[ -f "$PROJECT/artifacts.json" ]] || { echo "no artifacts.json in PROJECT=$PROJECT" >&2; exit 1; }
ANGLES=${ANGLES:--75 45 -15 -45 15}
OUT=${OUT:-outputs/mc3-patch-check/$SLURM_JOB_ID}

source scripts/activate_env.sh
# MotionCor3 needs the env's libtiff.so.6, as in lane.sbatch.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "job        $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), $(date -Is)"
echo "project    $PROJECT   series $SERIES"
echo "angles     $ANGLES   out $OUT"
nvidia-smi -L 2>/dev/null || true
echo

# shellcheck disable=SC2086  # ANGLES is a list
python3 scripts/mc3_patch_check.py plan --project "$PROJECT" --series "$SERIES" \
    --angles $ANGLES --out "$OUT"
motioncor3=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["tilts"][0]["runs"]["p5"]["command"][0])' "$OUT/plan.json")
missing=$(ldd "$motioncor3" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
[[ -z "$missing" ]] || { echo "MotionCor3 cannot load: ${missing}" >&2; exit 1; }
echo

python3 scripts/mc3_patch_check.py run --out "$OUT"
echo
status=0
python3 scripts/mc3_patch_check.py compare --out "$OUT" || status=$?
echo
echo "finished   $(date -Is)"
exit "$status"
