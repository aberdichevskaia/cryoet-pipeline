#!/bin/bash
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.log
#
# Compute one Dynamo project that is already set up and unfolded in the
# console, the way the MRA arms ran: Dynamo 1.1.591 standalone, 16 cores.
#
#   DIR=<folder holding the project> NAME=<project> \
#       sbatch --export=ALL --job-name=<project> scripts/dynamo_project.sbatch
#
# The project's -cores has to match --cpus-per-task. Disposable, like the rest
# of scripts/.

set -euo pipefail

: "${DIR:?set DIR to the folder that holds the project}"
: "${NAME:?set NAME to the project name}"

D=/lustre/groups/shared/users/peng_marr/a.berdichevskaia/apps/dynamo-1.1.591
cd "$DIR"

# The MCR unpacks itself here once per job; node-local /tmp is invisible from
# the submit node, which is where a crash gets read.
export MCR_CACHE_ROOT="/lustre/groups/shared/users/peng_marr/a.berdichevskaia/tmp/dynamo_mcr_${SLURM_JOB_ID}"
mkdir -p "$MCR_CACHE_ROOT"

echo "node        $(hostname)"
echo "started     $(date -Is)"
echo "project     $DIR/$NAME"

"$D/bin/run_dynamo_execute_project.sh" "$D" "$NAME" 2>&1 | tee "${NAME}_run.log"

echo "finished    $(date -Is)"
