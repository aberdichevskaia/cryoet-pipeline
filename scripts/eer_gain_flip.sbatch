#!/bin/bash
#SBATCH --job-name=eer-gain-flip
#SBATCH --output=slurm-%x-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#
# Which -FlipGain MotionCor3 needs to apply a Falcon `.gain` to EER movies,
# measured with the MotionCor3 the lanes run, before the pipeline's EER input
# relies on a value. The sources predict 1 (scripts/eer_gain_flip.py says why).
# A wrong flip raises no error: it leaves the gain's pattern in every sum.
#
# The data, from the repo root: one EMPIAR-11830 series (41 movies, ~6.5 GB) with
# its mdoc and its session's gain.
#
#     cryoet download-empiar --out data/empiar-11830 --empiar-id 11830 \
#         --layout configs/empiar-layouts/11830.yaml \
#         --tilt-series 01082023_BrnoKrios_Arctis_WebUI_Position_1
#
# Then, at MDC (the GPU flags are the site's; Munich has its own partition and
# QOS):
#
#     sbatch -p gpu --gres=gpu:1 --export=ALL,FRAMES=data/empiar-11830/frames,GAIN=data/empiar-11830/gain/20230807_182744_EER_GainReference.gain scripts/eer_gain_flip.sbatch
#
# Required: FRAMES (the folder of .eer files), GAIN (the .gain) and MOTIONCOR3
# (the executable, as lane.sbatch needs it; at MDC the .bashrc sets it on compute
# nodes only, so submit from one). Optional: COUNT movies (3, the first in name
# order: for this series the first three acquired, at -15, -12 and -18 degrees),
# FRACTIONS (10), FLIPS ("0 1") and OUT (outputs/eer-gain-flip/<job id>).
#
# Per movie, MotionCor3 sums the EER frames with -Align 0, once without the gain
# and once with it at each -FlipGain in FLIPS: EER sampling 1 (4K), no Fourier
# binning, no dose weighting. scripts/eer_gain_flip.py fmint writes each movie's
# FmIntFile from its header first, and compare then judges the sums and prints
# the verdict beside the prediction. A run with the gain whose output lacks
# "Gain reference has been loaded" stops the job: MotionCor3 only warns when it
# cannot use a gain (any mode but 32-bit float) and sums without one. About
# 0.2 GB per movie and flip under OUT.
#
# Disposable.

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?not running under Slurm}
[[ -f "$REPO/scripts/eer_gain_flip.py" ]] \
    || { echo "submit from the root of a cryoet-pipeline clone, not $REPO" >&2; exit 1; }
cd "$REPO"

for name in FRAMES GAIN MOTIONCOR3; do
    [[ -n "${!name:-}" ]] || { echo "$name is not set" >&2; exit 1; }
done
[[ -d "$FRAMES" ]] || { echo "FRAMES is not a folder: $FRAMES" >&2; exit 1; }
[[ -f "$GAIN" ]] || { echo "GAIN is not a file: $GAIN" >&2; exit 1; }
[[ -x "$MOTIONCOR3" ]] || { echo "MOTIONCOR3 is not an executable: '$MOTIONCOR3'" >&2; exit 1; }
COUNT=${COUNT:-3}
FRACTIONS=${FRACTIONS:-10}
FLIPS=${FLIPS:-0 1}
OUT=${OUT:-outputs/eer-gain-flip/$SLURM_JOB_ID}

source scripts/activate_env.sh
# MotionCor3 needs the env's libtiff.so.6, as in lane.sbatch.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
missing=$(ldd "$MOTIONCOR3" 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
[[ -z "$missing" ]] || { echo "MotionCor3 cannot load: ${missing}" >&2; exit 1; }

movies=()
while IFS= read -r movie; do
    movies+=("$movie")
done < <(find "$FRAMES" -maxdepth 1 -name '*.eer' | sort | head -n "$COUNT")
(( ${#movies[@]} > 0 )) || { echo "no .eer files in $FRAMES" >&2; exit 1; }
mkdir -p "$OUT/logs"

echo "job        $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), $(date -Is)"
echo "motioncor3 $MOTIONCOR3 (sha256 $(sha256sum "$MOTIONCOR3" | cut -c1-16))"
echo "gain       $GAIN"
echo "movies     ${#movies[@]} of $(find "$FRAMES" -maxdepth 1 -name '*.eer' | wc -l) in $FRAMES"
echo "flips      $FLIPS   fractions $FRACTIONS   out $OUT"
nvidia-smi -L 2>/dev/null || true
echo

sum_movie() {  # movie stem label [gain flags]
    local movie=$1 stem=$2 label=$3
    shift 3
    local log="$OUT/logs/${stem}_${label}.out"
    if ! "$MOTIONCOR3" -InEer "$movie" -OutMrc "$OUT/${stem}_${label}.mrc" \
            -FmIntFile "$OUT/$stem.fmint" -EerSampling 1 -FtBin 1 -Align 0 -FmDose 0 \
            -PixSize "$pixel" -kV 300 -Cs 0 -Gpu 0 -LogDir "$OUT/logs" "$@" >"$log" 2>&1; then
        echo "MotionCor3 failed on $stem ($label); the end of $log:" >&2
        tail -20 "$log" >&2
        exit 1
    fi
    [[ -f "$OUT/${stem}_${label}.mrc" ]] \
        || { echo "MotionCor3 wrote no $OUT/${stem}_${label}.mrc; see $log" >&2; exit 1; }
}

for movie in "${movies[@]}"; do
    stem=$(basename "$movie" .eer)
    echo "== $stem"
    facts=$(python3 scripts/eer_gain_flip.py fmint "$movie" --fractions "$FRACTIONS" \
        --out "$OUT/$stem.fmint" --gain "$GAIN")
    sed 's/^/   /' <<<"$facts"
    # With -Align 0 and no dose weighting MotionCor3 uses the pixel size for nothing.
    pixel=$(awk '$1 == "pixel_angstrom" && $2 != "None" {print $2}' <<<"$facts")
    pixel=${pixel:-1.0}

    sum_movie "$movie" "$stem" nogain
    if grep -q "Gain reference has been loaded" "$OUT/logs/${stem}_nogain.out"; then
        echo "MotionCor3 loaded a gain in the run without one; see $OUT/logs/${stem}_nogain.out" >&2
        exit 1
    fi
    for flip in $FLIPS; do
        sum_movie "$movie" "$stem" "flip$flip" -Gain "$GAIN" -FlipGain "$flip"
        grep -q "Gain reference has been loaded" "$OUT/logs/${stem}_flip$flip.out" \
            || { echo "MotionCor3 did not use the gain (flip $flip); see $OUT/logs/${stem}_flip$flip.out" >&2; exit 1; }
    done
done

echo
echo "MotionCor3's first lines, first run:"
head -5 "$OUT/logs/$(basename "${movies[0]}" .eer)_nogain.out" | sed 's/^/   /'
echo
python3 scripts/eer_gain_flip.py compare "$OUT"
echo
echo "finished   $(date -Is)"
