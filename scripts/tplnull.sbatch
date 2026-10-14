#!/bin/bash
#SBATCH --job-name=ribo-tplnull
#SBATCH --output=slurm-ribo-tplnull-%a-%j.log
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:a100_3g.20gb:1
# A 20 GB slice is enough: sacct shows both runs that finished (40090844_1,
# 40094531_0, 2026-09-04) on a100_3g.20gb. This once asked for a whole card on a
# misreading. Inside a MIG slice nvidia-smi --query-gpu describes the parent card,
# so the log's "A100-PCIE-40GB" was not a whole card, and 27983 MiB of memory.used
# cannot have been this job's own: it is more than the slice holds.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0,1
#
# How much does a global orientation search against the template print the
# template into boxes that have nothing at their centre?
#
# This is the last question open in lane B. Two results already stand: the picks
# sit on real, spatially concentrated density rather than noise (radial ratio 2.4
# against a control's 1.00), and two of four classes reproduce across seeds where
# the control's reproduce nothing. Neither says the density is a RIBOSOME.
#
# The confound is recorded in roadmap section 4: averaging particles at their
# template-matched orientations cannot test whether the picks are real, because
# noise oriented by a global search against a template averages to +0.321
# correlation with that template against +0.035 at random angles. So "the average
# resembles a ribosome" is evidence only insofar as it exceeds what that effect
# produces on its own — and nobody had measured that on this data.
#
# --array=0,1 runs both halves of the measurement at once:
#   task 0   the real picks           starting from the template, global search
#   task 1   the random-position boxes  the same, on boxes centred on nothing
# Task 1's agreement with the template is the floor. Task 0 is meaningful to the
# extent it clears it.
#
# THE PAIR IS WHAT READS, NOT EITHER NUMBER. Both tasks use a 15 degree step,
# which is coarser than pytom's own increment for this particle — degrees(35/290)
# = 6.9 deg — and a coarser search has fewer chances to find a fit to noise. So
# this UNDERSTATES the template-printing effect, in the direction that flatters
# the real data. It is still a valid comparison because both sides are treated
# identically; what it does not give is the absolute size of the bias pytom
# itself introduced. That would need 6.9 deg, roughly ten times the orientations
# and a day of GPU, and is only worth paying for if this pair comes out close.
#
# COST, counted from cone_offsets and _symmetric_offsets rather than estimated.
# Per particle, summed over the three iterations:
#
#   iter 1  cone 180 deg @ 15.00  ->  106 directions x 25 in-plane =  2650
#   iter 2  cone  90 deg @  7.50  ->  128            x 25          =  3200
#   iter 3  cone  45 deg @  3.75  ->  133            x 25          =  3325
#                                                       per particle  9175
#
# The count RISES across iterations: refine_factor halves the range and the step
# together, but the ring count stays put while each ring's circumference is
# sampled more finely, so narrowing the search does not make it cheaper here.
#
# MEMORY is not measured. The particle stack is 4000 x 48^3 x 4 B = 1.77 GB, and
# the loop holds raw_volumes, aligned and new_aligned at once for 5.3 GB. The
# search adds the batch of rotated volumes and its FFT workspace (2650
# orientations per particle in the first iteration against 24 in a
# classification iteration), so 5.3 GB is a lower bound. The peak is below the
# slice's 20 GB, since both runs finished there; the 27983 MiB once recorded here
# was the whole card (see the --gres note above).
#
# The classification runs cost 296 per particle by the same arithmetic (8 cone
# directions x 3 in-plane x 4 classes for three iterations, then 1 x 1 once the
# Crowther floor clamps), and one measured 14 minutes on an A100. So this is 31x
# that, or about 7.2 hours.
#
# A first version of this script asked for 8 hours on a guess of "roughly three",
# which left 48 minutes of margin on a job that writes nothing until it finishes
# and takes no checkpoints: hitting the wall clock loses everything. Hence 16.

set -euo pipefail

REPO=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/cryoet-pipeline
PROJECT=outputs/ribo
SERIES=(TS_0005 TS_0006 TS_0007 TS_0008 TS_0009)
BOX=48                    # the extraction box, configs/lanes/ribo-bin8.toml

# Index 0 is the real lane, index 1 the control. Same everything else.
VARIANTS=(bin8-top800 decoy)
LABELS=(real-tplnull decoy-tplnull)
INDEX="${SLURM_ARRAY_TASK_ID:?submit with sbatch --array=0,1}"
VARIANT="${VARIANTS[$INDEX]}"
LABEL="${LABELS[$INDEX]}"

cd "$REPO"

# No IMOD here: averaging calls nothing from it. Sourcing IMOD-linux.sh in this
# job only adds a path that can go stale and fail a run that never needed it.
source scripts/activate_env.sh

echo "host        $(hostname)"
echo "started     $(date -Is)"
echo "lane        $VARIANT -> $LABEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# Inside a MIG slice the line above reports the whole card; -L lists the slice.
nvidia-smi -L || true
echo

# Both tasks must start from the SAME reference, or the two sides of the
# comparison measure agreement with different things.
#
# And it has to be the PADDED template. A prepared template is 39 voxels, the
# particle box is 48, and the two commands that consume it disagree about whose
# job the reconciliation is: compute-fsc pads centrally (cli.py:3805) while
# _load_reference_mrc refuses outright (subtomogram.py:1616). Padding once into
# a file makes --initial-reference here and --model-map later point at the same
# object, which is easier to trust than two reconciliations agreeing.
#
# Not created here on purpose: both array tasks would write the same file at the
# same moment. Make it once before submitting.
RAW_TPL=$(ls "$PROJECT"/templates/TS_0005/*_template.mrc | head -1)
TPL="${RAW_TPL%.mrc}_pad${BOX}.mrc"
if [[ ! -f "$TPL" ]]; then
    echo "no padded template at $TPL" >&2
    echo "make it once, then resubmit:" >&2
    echo "  python3 scripts/pad_map.py --map $RAW_TPL --box $BOX" >&2
    exit 1
fi
echo "template    $TPL"

# A registry copy per task: both would otherwise write the same file, each
# loading it at startup and writing at the end, so whoever finishes second drops
# the other's entries. These are control runs and nothing downstream reads their
# registry entries — compute-fsc takes map paths directly.
REGISTRY="$PROJECT/artifacts-${LABEL}.json"
cp "$PROJECT/artifacts.json" "$REGISTRY"
echo "registry    $REGISTRY (copy: the real artifacts.json is not touched)"

MANIFESTS=()
for series in "${SERIES[@]}"; do
    manifest="$PROJECT/manifests/$series.json"
    if [[ ! -f "$manifest" ]]; then
        echo "missing manifest: $manifest" >&2
        exit 1
    fi
    stack="$PROJECT/particles/$series/${series}_cubic-crop_${VARIANT}_particles.zarr"
    if [[ ! -d "$stack" ]]; then
        echo "missing particle stack for lane $VARIANT: $stack" >&2
        exit 1
    fi
    MANIFESTS+=(--manifest "$manifest")
done

if compgen -G "$PROJECT/qc/*/subtomogram_averaging/*_${LABEL}_qc.json" > /dev/null; then
    echo "output for $LABEL already exists:" >&2
    ls -1 "$PROJECT"/qc/*/subtomogram_averaging/*_"${LABEL}"_qc.json >&2
    exit 1
fi
echo "preflight   ok, ${#SERIES[@]} series"
echo

cryoet average-particles \
    "${MANIFESTS[@]}" \
    --registry "$REGISTRY" \
    --out "$PROJECT" \
    --backend cone-search \
    --num-classes 1 \
    --initial-reference "$TPL" \
    --cone-range-deg 180 \
    --inplane-range-deg 360 \
    --cone-sampling-deg 15 \
    --inplane-sampling-deg 15 \
    --num-iterations 3 \
    --variant "$VARIANT" \
    --output-variant "$LABEL" \
    --device cuda

echo
echo "finished    $(date -Is)"
echo
echo "next, once BOTH tasks have finished — the same FSC against the same"
echo "template for each, unfiltered maps, no mask, so the two are treated alike:"
echo
for label in real-tplnull decoy-tplnull; do
    echo "  cryoet compute-fsc --mode map-to-model \\"
    echo "    --average-map $PROJECT/averages/*/*_${label}_average.mrc \\"
    echo "    --model-map $TPL \\"
    echo "    --out $PROJECT/fsc/${label}-vs-template.json \\"
    echo "    --plot $PROJECT/fsc/${label}-vs-template.png --device cpu"
done
echo
echo "Read the pair, not either number: decoy-tplnull is the floor that a global"
echo "search prints from nothing, and real-tplnull means something only above it."
