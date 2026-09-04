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
# The control: four-class classification of RANDOM-position boxes, run with the
# same settings and the same two seeds as the real bin8-top800-cls4 run.
#
# Why this exists. The real run produced four class averages that look alike and
# do not look like ribosomes, and two of the four reproduce across seeds above
# what four hundred shuffles reach. Both observations are equally consistent with
# a weak real population and with an artefact shared by every box, and nothing
# computed on the real picks alone separates those readings. This control does:
# same tomograms, same depths, same box count, same starting orientations, same
# settings, coordinates random. Its own reproducibility is the baseline the real
# run's 36% and 33% have to be read against.
#
# THE SETTINGS ARE COPIED FROM THE REAL RUN'S QC AND MUST NOT BE "IMPROVED".
# In particular --particle-diameter-angstrom and --target-resolution-angstrom are
# deliberately absent, so that they fall back to the same defaults the real run
# used (the box, 469.6 A, and Nyquist, 19.6 A). Those defaults are wrong -- the
# particle is 290 A and the references are filtered to 60 A, so the Crowther
# floor should be near 11.9 deg rather than the 2.39 deg that ran -- and fixing
# them is a SEPARATE experiment for afterwards. A control that fixes them
# controls a run that was never done.
#
# The array index IS the seed: --array=1,7 gives seeds 1 and 7, matching
# bin8-top800-cls4-s1 and -s7. The two tasks are independent and run in
# parallel.
#
# GPU memory. Every particle is loaded onto the device
# (subtomogram.py:1937 passes device= into torch.as_tensor), and the loop holds
# raw_volumes, aligned and new_aligned at once:
#
#   one copy   4000 x 48^3 x 4 B = 1.77 GB
#   peak       three copies       = 5.3 GB, plus references and FFT workspace
#
# so the 20 GB MIG slice is comfortable. Compute is the constraint, not memory.
#
# Runtime is UNMEASURED on GPU. The arithmetic is about 4000 particles x 4
# references x 5 iterations x ~25 orientations = 2e6 rotations of a 48^3 box,
# each with a shift search. 8 hours is a guess with headroom; read the elapsed
# time out of the log and size later runs from that rather than from this
# comment. There are no checkpoints, and nothing is written until the run
# finishes, so a job that hits the wall clock loses everything.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo
SERIES=(TS_0005 TS_0006 TS_0007 TS_0008 TS_0009)
VARIANT=decoy
CLASSES=4
SEED="${SLURM_ARRAY_TASK_ID:?this script expects to run as an array job}"
LABEL="${VARIANT}-cls${CLASSES}-s${SEED}"

cd "$REPO"

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cryoet-pipeline
set -u

echo "host        $(hostname)"
echo "started     $(date -Is)"
echo "seed        $SEED"
echo "label       $LABEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Each array task writes the registry it was given, and both tasks would
# otherwise write the same file: each loads it at startup, adds its artifacts,
# and writes at the end, so whichever finishes second silently drops the
# other's entries. A per-seed copy removes the race. Nothing downstream needs
# these entries -- compare_classifications.py reads the STAR and QC files by
# path -- and the control is marked temporary anyway, so a throwaway registry
# is the honest place for its bookkeeping.
REGISTRY="$PROJECT/artifacts-${LABEL}.json"
cp "$PROJECT/artifacts.json" "$REGISTRY"
echo "registry    $REGISTRY (copy, so the two seeds cannot clobber each other)"

# Preflight. Every check here is for something that otherwise fails after the
# GPU has been paid for: the stacks are read at startup but the refusal for an
# existing output comes later, and a missing manifest is an argument error
# thrown inside the CLI.
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
        echo "run decoy_picks.py and then extract-particles --variant $VARIANT" >&2
        exit 1
    fi
    MANIFESTS+=(--manifest "$manifest")
done

# The backend refuses rather than overwriting, which is right, but it refuses
# after loading every stack. Fail here instead.
if compgen -G "$PROJECT/qc/*/subtomogram_averaging/*_${LABEL}_qc.json" > /dev/null; then
    echo "output for $LABEL already exists:" >&2
    ls -1 "$PROJECT"/qc/*/subtomogram_averaging/*_"${LABEL}"_qc.json >&2
    echo "pass --overwrite deliberately, or use a different --output-variant" >&2
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
