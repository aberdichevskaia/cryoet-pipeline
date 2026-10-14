#!/bin/bash
#SBATCH --job-name=ribo-decoy-cls
#SBATCH --output=slurm-ribo-decoy-cls-s%a-%j.log
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=1,7
#
# The control: four-class classification of RANDOM-position boxes, with the same
# settings and the same two seeds as the real bin8-top800-cls4 run.
#
# Why it exists. The real run gave four class averages that look alike and do not
# look like ribosomes, and two of the four reproduce across seeds beyond what
# four hundred shuffles reach. Both facts are equally consistent with a weak real
# population and with an artefact shared by every box, and nothing computed on
# the real picks alone separates those readings. This does: same tomograms, same
# depths, same box count, same starting orientations, same settings, coordinates
# random. Its reproducibility is the baseline the real run's 36% and 33% have to
# be read against.
#
# --array=1,7 IS the seed list: task 1 runs seed 1, task 7 runs seed 7, matching
# bin8-top800-cls4-s1 and -s7. The two tasks are independent and run in parallel.
#
# THE SETTINGS ARE COPIED FROM THE REAL RUN'S QC AND MUST NOT BE "IMPROVED".
# --particle-diameter-angstrom and --target-resolution-angstrom are deliberately
# absent so they fall back to the same defaults the real run used (the box,
# 469.6 A, and Nyquist, 19.6 A). Those defaults are wrong -- the particle is
# 290 A and the references are filtered to 60 A, so the Crowther floor should sit
# near 11.9 deg rather than the 2.39 deg that ran -- and fixing them is a
# SEPARATE experiment afterwards. A control that fixes them controls a run
# nobody did.
#
# GPU memory. Every particle is loaded onto the device (subtomogram.py:1937
# passes device= into torch.as_tensor) and the loop holds raw_volumes, aligned
# and new_aligned at once:
#
#   one copy   4000 x 48^3 x 4 B = 1.77 GB
#   peak       three copies      = 5.3 GB, plus references and FFT workspace
#
# so a 20 GB MIG slice is roughly four times what is needed. Compute is the
# constraint. If this gres is slow to schedule, any GPU will do.
#
# Runtime is UNMEASURED on GPU: about 4000 particles x 4 references x 5
# iterations x ~25 orientations = 2e6 rotations of a 48^3 box, each with a shift
# search. 8 hours is a guess with headroom. Read the elapsed time out of the log
# and size later runs from that. There are no checkpoints and nothing is written
# until the run finishes, so a job that hits the wall clock loses everything.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo
SERIES=(TS_0005 TS_0006 TS_0007 TS_0008 TS_0009)
VARIANT=decoy
CLASSES=4
SEED="${SLURM_ARRAY_TASK_ID:?submit with sbatch --array, or the seed is unset}"
LABEL="${VARIANT}-cls${CLASSES}-s${SEED}"

cd "$REPO"

source scripts/activate_env.sh

echo "host        $(hostname)"
echo "started     $(date -Is)"
echo "seed        $SEED"
echo "label       $LABEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Inside a MIG slice the line above reports the whole card; -L lists the slice.
nvidia-smi -L || true
echo

# A registry copy per task, because both array tasks would otherwise write the
# same file: each loads it at startup, adds its artifacts and writes at the end,
# so whoever finishes second silently drops the other's entries.
#
# This is NOT for the duplicate-id collision -- that was a real bug, now fixed
# in 5a14a55: a class average's id carried no variant, so every classification
# of one scope produced the same four ids while the maps on disk were correctly
# separated. Before that fix this run needed --overwrite, which against the real
# registry would have replaced the REAL run's entries with the control's.
#
# Nothing downstream needs these entries anyway: compare_classifications.py
# reads the STAR and QC files by path, and the control is registered temporary.
REGISTRY="$PROJECT/artifacts-${LABEL}.json"
cp "$PROJECT/artifacts.json" "$REGISTRY"
echo "registry    $REGISTRY (copy: the real artifacts.json is not touched)"
echo

# Preflight. Each of these otherwise fails after the GPU has been paid for.
MANIFESTS=()
for series in "${SERIES[@]}"; do
    manifest="$PROJECT/manifests/$series.json"
    if [[ ! -f "$manifest" ]]; then
        echo "missing manifest: $manifest" >&2
        exit 1
    fi
    stack="$PROJECT/particles/$series/${series}_cubic-crop_${VARIANT}_particles.zarr"
    if [[ ! -d "$stack" ]]; then
        echo "missing control particle stack: $stack" >&2
        echo "run decoy_picks.py, then extract-particles --variant $VARIANT" >&2
        exit 1
    fi
    MANIFESTS+=(--manifest "$manifest")
done

# The maps for THIS label must not already exist. The backend refuses too, but
# only after loading every stack onto the GPU, so fail here instead.
if compgen -G "$PROJECT/qc/*/subtomogram_averaging/*_${LABEL}_qc.json" > /dev/null; then
    echo "output for $LABEL already exists:" >&2
    ls -1 "$PROJECT"/qc/*/subtomogram_averaging/*_"${LABEL}"_qc.json >&2
    echo "delete it, or pick a different label, rather than overwriting a result" >&2
    exit 1
fi
echo "preflight   ok, ${#SERIES[@]} series"
echo

cryoet average-particles \
    "${MANIFESTS[@]}" \
    --registry "$REGISTRY" \
    --out "$PROJECT" \
    --backend cone-search \
    --num-classes "$CLASSES" \
    --cone-range-deg 30 \
    --cone-sampling-deg 10 \
    --inplane-range-deg 30 \
    --inplane-sampling-deg 10 \
    --shift-range-px 2 \
    --num-iterations 5 \
    --seed "$SEED" \
    --variant "$VARIANT" \
    --output-variant "$LABEL" \
    --device cuda

echo
echo "finished    $(date -Is)"
echo
echo "next: compare the two seeds against each other,"
echo "  python3 scripts/compare_classifications.py \\"
echo "    --first  $PROJECT/averages/*/*_${VARIANT}-cls${CLASSES}-s1_particles.star \\"
echo "    --second $PROJECT/averages/*/*_${VARIANT}-cls${CLASSES}-s7_particles.star"
echo "and read the result against the real run's 36% and 33%."
