#!/bin/bash
#SBATCH --job-name=ctf-window
#SBATCH --output=slurm-%x-%j.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Why do the ryr1 CTF fits fail past ~12 degrees, while the ribosome ones did not?
#
#     sbatch --partition=normal \
#         --export=ALL,PROJECT=outputs/ryr1,CONFIG=configs/lanes/ryr1-bin8.toml,SERIES=039 \
#         scripts/ctf_window_test.sbatch
#
# IMOD_DIR has to name the MDC install (lane.sbatch falls back to Munich's).
#
# What is known (report-bin4.pdf, written 2026-10-15; per-view values read off
# its CTF plots): on every series the views fitted to 7-9 A sit inside about
# +-12 degrees, and nearly every other view stops at 29.7 A, the 30 A low end of
# the fit range, so its defocus means nothing. 161 is the exception, fitted from
# -13 to +39. On the fitted views the defocus falls steadily with the tilt:
#
#   039  4.98 um at -11  ->  4.43 at +10     096  6.23 at -10  ->  5.71 at  +5
#   097  5.37 at -12     ->  4.87 at  +3     125  6.37 at -16  ->  5.80 at  +8
#   161  5.76 at -13     ->  5.03 at +39     (the weakest trend, the widest fit)
#
# The per-view pass searches +-1000 A around the 0-degree fit
# (DEFAULT_DEFOCUS_SEARCH_WINDOW_ANGSTROM), and the fitted views already span
# 5000 A: carried on to +-42 degrees, the trend puts the far views ~1 um off the
# window. The guess is that a view whose defocus is outside the window is
# fitted only when CTFFIND's refinement walks back to it, and from ~12 degrees
# on it does not. Not shown yet, which is what this job is for.
#
# The job rebuilds the series' unlabelled bin-2 CTF stack first (--overwrite):
# the registered one predates the newstack -size fix and is cropped, with about
# 30% of each view blank. Then it fits that stack four ways, each under its own
# variant, so the lane's own CTF model is not touched:
#
#   variant      window      CTFFIND5 sample tilt
#   ctfw1        +-1000 A    no      the lane as it ran, on the uncropped stack
#   ctfw5        +-5000 A    no
#   ctfw10       +-10000 A   no
#   ctfw5-tilt   +-5000 A    yes
#
# ctfw1 -> ctfw5 -> ctfw10 tests the window alone; ctfw5 -> ctfw5-tilt adds
# CTFFIND5's own fit of the gradient across each view. If the window is the
# cause, the fitted views should reach far past 12 degrees from ctfw5 on, and
# their defocus should still follow the line ctfw1 fitted near 0. If the window
# is not the cause, ctfw1 and ctfw5 look alike and the cliff stays.
#
# Ends with scripts/ctf_fit_table.py: per view, defocus and fit limit for all
# four, and per variant the number of views fitted and the defocus line.
#
# Disposable.

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:?not running under Slurm}
[[ -f "$REPO/scripts/run_series.py" ]] \
    || { echo "submit from the root of a cryoet-pipeline clone, not $REPO" >&2; exit 1; }
cd "$REPO"

for name in PROJECT CONFIG SERIES IMOD_DIR; do
    if [[ -z "${!name:-}" ]]; then
        echo "$name is not set" >&2
        exit 1
    fi
done

source scripts/activate_env.sh
set +u
source "$IMOD_DIR/IMOD-linux.sh"
set -u
command -v ctffind >/dev/null || command -v ctffind5 >/dev/null \
    || { echo "no ctffind on PATH" >&2; exit 1; }

lane_command() {
    python3 scripts/run_series.py --project "$PROJECT" --config "$CONFIG" \
        --series "$SERIES" --from "$1" --until "$1" | grep "^cryoet "
}

STACK=$(lane_command build-ctf-stack)
FIT=$(lane_command estimate-ctf)
[[ -n "$STACK" && -n "$FIT" ]] || { echo "run_series.py printed no command" >&2; exit 1; }
# The lane may set its own window or tilt; each variant here sets both.
FIT=$(sed -E -e 's/ --ctffind-defocus-search-window-angstrom [^ ]+//' \
             -e 's/ --(no-)?ctffind-determine-sample-tilt//' <<<"$FIT")

echo "job       $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), $(date -Is)"
echo "series    $SERIES in $PROJECT"
echo "ctffind   $(command -v ctffind5 || command -v ctffind)"
echo

eval "$STACK --overwrite"

fit() {  # label window [extra flags]
    local label=$1 window=$2
    shift 2
    echo
    echo "== $label: +-$window A $*"
    eval "$FIT --ctffind-defocus-search-window-angstrom $window $* --variant $label --overwrite"
}

fit ctfw1 1000
fit ctfw5 5000
fit ctfw10 10000
fit ctfw5-tilt 5000 --ctffind-determine-sample-tilt

echo
python3 scripts/ctf_fit_table.py "$PROJECT" "$SERIES" ctfw1 ctfw5 ctfw10 ctfw5-tilt
echo "finished  $(date -Is)"
