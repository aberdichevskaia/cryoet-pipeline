# Activates the cryoet-pipeline env in a Slurm job. Sourced, not run, by the job
# scripts (scripts/*.sbatch) once they are in the repo root; leaves ENV_MANAGER
# saying how it did it.
#
# conda where there is one (the Munich cluster), micromamba otherwise (MDC
# Berlin). A batch shell reads no .bashrc, so the micromamba function its shell
# init defines is not here: the executable is found on PATH, or through
# MAMBA_EXE, which that init exports and --export=ALL carries into the job.
# ENV_MANAGER=conda|micromamba picks one on a machine that has both.
#
# `set -u` is for our code, not for hooks that were not written for it: it is
# off while they run and on again after, as the job scripts run with it.

if [[ -z "${ENV_MANAGER:-}" ]]; then
    if command -v conda >/dev/null; then
        ENV_MANAGER=conda
    elif command -v micromamba >/dev/null || [[ -x "${MAMBA_EXE:-}" ]]; then
        ENV_MANAGER=micromamba
    else
        echo "neither conda nor micromamba found; put one on PATH before submitting" >&2
        exit 1
    fi
fi

set +u
case "$ENV_MANAGER" in
    conda)
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate cryoet-pipeline
        ;;
    micromamba)
        micromamba_exe=$(command -v micromamba || echo "$MAMBA_EXE")
        eval "$("$micromamba_exe" shell hook --shell bash)"
        micromamba activate cryoet-pipeline
        ;;
    *)
        echo "ENV_MANAGER must be conda or micromamba, not '$ENV_MANAGER'" >&2
        exit 1
        ;;
esac
set -u
