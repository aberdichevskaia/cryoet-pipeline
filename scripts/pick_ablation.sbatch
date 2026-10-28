#!/bin/bash
#SBATCH --job-name=pick-ablation
#SBATCH --output=slurm-%x-%A_%a.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4%1
#
# Which settings let pytom see the ribosomes in the lane's tomograms?
#
#     python3 scripts/submit_recipe.py pick-ablation configs/lanes/ribo-mdc.toml \
#         --series TS_0463 --dry-run
#
# submit_recipe.py runs it where the lane picks, checks it first and journals it in
# the project; its dry run refuses today's lanes, as the job's own check does (see
# the result below). Given to sbatch by hand, with PROJECT, CONFIG and SERIES in the
# environment, it runs the same, without any of that.
#
# What is known (local copies, 2026-10-15): on TS_0005 the lane's old score map
# put 44% of its top 100 peaks in the specimen band, where noise puts 48%; a
# plain normalised cross-correlation with the same template on the same volume
# put 100% there. Adding pytom's two corrections to that plain match, one at a
# time and together (TS_0005; pipeline TS_0002 in brackets):
#
#   |CTF| weighting           67%  (74%)
#   random-phase correction   84%  (80%)
#   both                      44%        noise would give 45%
#
# Anna's August etomo reconstruction of TS_0002 kept 98% under |CTF| weighting,
# so the corrections are fatal here because the lane's tomograms are weak in the
# band they lean on, not because they are wrong in themselves. All of that is a
# stand-in matcher (no wedge, no dose, 1000 random orientations); real pytom is
# what this job measures. A colleague's advice is tested alongside: low-pass
# near Nyquist, a mask 3 px past the particle with a soft edge, and a box twice
# the particle (pytom's tutorial uses 2.8x).
#
# Five matches of one series, one at a time (%1), each under its own output
# variant beside the lane's picks, each keeping its score volume (~1 GB). Picks a
# run before this one left under these labels are replaced only with OVERWRITE=1
# (submit_recipe.py --overwrite):
#
#   task  label              template               low-pass  random-phase  CTF
#   0     bin8-abl-lane      lane's old: 39 px box,  35 A      on            on
#                            17 px hard-edged mask
#   1     bin8-abl-new25     60 px box, 18 px mask   25 A      on            on
#                            with a 1 px soft edge
#   2     bin8-abl-new25-nr  the same                25 A      off           on
#   3     bin8-abl-new25-pl  the same                25 A      off           off
#   4     bin8-abl-new35-pl  the same                35 A      off           off
#
# Task 0 reproduces the lane's map; 0 -> 1 is the advice on top of the lane's
# corrections; 1 -> 2 -> 3 removes the corrections; 3 -> 4 is the low-pass alone.
#
# Result, 2026-10-15 (jobs 3640899 TS_0463, 3643100 TS_0306): task 3 won on both,
# 93% and 99% in the band where noise gives 40% and 75%, against the lane's 89%
# and 71%; full table in configs/lanes/ribo-mdc.toml [pick]. The ribo lanes now
# run task 3's settings, so this job's check below refuses them: it was written
# for the lane as it was, and is kept as the record of how the choice was made.
# Every task extracts the lane's best-N with no floor, so the five pick lists are
# the same size and comparable to each other and to a biologist's.
#
# Each task ends with one line from scripts/score_band_readout.py: how many of the
# map's 100 best peaks sit in the specimen band, against what noise would put
# there. Collect them with  grep -h "top-100" slurm-*pick-ablation-<jobid>_*.log
#
# The two templates are made here by prepare-template under their own output
# variants, so the lane's registered template is left alone either way.
#
# Disposable.

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?not running under Slurm}
[[ -f "$REPO/scripts/run_series.py" ]] \
    || { echo "submit from the root of a cryoet-pipeline clone, not $REPO" >&2; exit 1; }
cd "$REPO"

for name in PROJECT CONFIG SERIES PYTOMO_MP_DIR; do
    if [[ -z "${!name:-}" ]]; then
        echo "$name is not set" >&2
        exit 1
    fi
done

source scripts/activate_env.sh
set -u

lane_command() {
    python3 scripts/run_series.py --project "$PROJECT" --config "$CONFIG" \
        --series "$SERIES" --from "$1" --until "$1" | grep "^cryoet $1"
}

PICK=$(lane_command pick-particles)
PREP=$(lane_command prepare-template)
[[ -n "$PICK" && -n "$PREP" ]] || { echo "run_series.py printed no command" >&2; exit 1; }
for wanted in " --pytom-random-phase-correction" " --pytom-low-pass " \
              " --pytom-cut-off 0" " --template-box-size-px " " --template-mask-sigma-px "; do
    [[ "$PICK $PREP" == *"$wanted"* ]] \
        || { echo "the lane no longer renders '$wanted'; this job assumes it does" >&2; exit 1; }
done

# The lane renders the new template settings; the old ones are the lane's
# template as it was registered before 2026-10-15.
NEW_TEMPLATE="$PREP"
OLD_TEMPLATE=$(sed -E -e 's/ --template-box-size-px [^ ]+//' \
                      -e 's/ --template-mask-sigma-px [^ ]+//' \
                      -e 's/ --template-mask-radius-px [^ ]+/ --template-mask-radius-px 17/' \
                      <<<"$PREP")

template_for() {  # label prepare-command -> "template mask" paths, made if missing
    local label=$1 command=$2 dir="$PROJECT/templates/$SERIES"
    local template="$dir/${SERIES}_pytom_${label}_template.mrc"
    local mask="$dir/${SERIES}_pytom_${label}_mask.mrc"
    if [[ ! -f "$template" || ! -f "$mask" ]]; then
        eval "$command --output-variant $label" >&2
    fi
    [[ -f "$template" && -f "$mask" ]] \
        || { echo "prepare-template wrote no $template / $mask" >&2; exit 1; }
    echo "$template $mask"
}

without_rpc() { sed -e 's/ --pytom-random-phase-correction//' <<<"$1"; }
with_low_pass() { sed -E -e "s/ --pytom-low-pass [^ ]+/ --pytom-low-pass $2/" <<<"$1"; }

task=${SLURM_ARRAY_TASK_ID:?submit as an array job}
case "$task" in
    0) label=bin8-abl-lane;     tpl=bin8-abl-tpl39; prep=$OLD_TEMPLATE
       command=$(with_low_pass "$PICK" 35) ;;
    1) label=bin8-abl-new25;    tpl=bin8-abl-tpl60; prep=$NEW_TEMPLATE
       command=$(with_low_pass "$PICK" 25) ;;
    2) label=bin8-abl-new25-nr; tpl=bin8-abl-tpl60; prep=$NEW_TEMPLATE
       command=$(without_rpc "$(with_low_pass "$PICK" 25)") ;;
    3) label=bin8-abl-new25-pl; tpl=bin8-abl-tpl60; prep=$NEW_TEMPLATE
       command="$(without_rpc "$(with_low_pass "$PICK" 25)") --no-pytom-ctf-weighting" ;;
    4) label=bin8-abl-new35-pl; tpl=bin8-abl-tpl60; prep=$NEW_TEMPLATE
       command="$(without_rpc "$(with_low_pass "$PICK" 35)") --no-pytom-ctf-weighting" ;;
    *) echo "no task $task" >&2; exit 1 ;;
esac

# Assigned first: a failure inside the substitution then stops the job, which it
# would not do inside a here-string.
paths=$(template_for "$tpl" "$prep")
read -r template mask <<<"$paths"

echo "job       $SLURM_JOB_NAME ($SLURM_ARRAY_JOB_ID task $task) on $(hostname), $(date -Is)"
echo "series    $SERIES in $PROJECT"
echo "variant   $label   template $template"
echo "overwrite $([[ -n "${OVERWRITE:-}" ]] && echo yes || echo no)"
nvidia-smi -L 2>/dev/null || true
echo

# eval: the printed commands name the executables through $PYTOMO_MP_DIR.
eval "$command --pytom-template $template --pytom-template-mask $mask \
    --output-variant $label --pytom-keep-score-volume ${OVERWRITE:+--overwrite}"

echo
tomogram=$(ls -d "$PROJECT/tomograms/$SERIES/"*bin8*.zarr | head -1)
python3 scripts/score_band_readout.py "$tomogram" \
    "$PROJECT/picks/$SERIES/"*"_${label}_scores.mrc"
echo "finished  $(date -Is)"
