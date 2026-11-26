# The pipeline's settings for a Stockholm course account on cryoem.se. The
# course's installation copies this file to ~/cryoet/session.sh.
#
# In every new terminal, before working with the pipeline:
#
#     module load tomobear/2026
#     source ~/cryoet/session.sh
#
# Not in ~/.bashrc on purpose: the same accounts serve the Warp/M part of the
# workshop, which should see none of this. Slurm jobs take these variables from
# the shell they are submitted from, so submit from a terminal where both lines
# ran.

if [[ -z "${MEMBRAIN_CKPT:-}" || -z "${CRYOET_LIBRARY:-}" ]]; then
    echo "session.sh: load the course module first: module load tomobear/2026" >&2
    return 1 2>/dev/null || exit 1
fi

# Everything the installation wrote is in ~/cryoet: the environments, the
# downloads pip and micromamba keep, and the clone with its outputs.
export CRYOET_WORK=$HOME/cryoet
export MAMBA_ROOT_PREFIX=$CRYOET_WORK/micromamba
export PIP_CACHE_DIR=$CRYOET_WORK/cache/pip
# The jobs activate the pipeline's environment with micromamba.
export ENV_MANAGER=micromamba
command -v micromamba >/dev/null || module load micromamba/2.5.0
eval "$(micromamba shell hook --shell bash)"

# The cluster's programs, as a course account ran them on 24 November 2026:
# IMOD 5.2.1, CTFFIND 5.0.2 (the cluster's CTFFIND 4 is refused), MotionCor3
# (its folder says 1.1.1, it reports 1.1.2) and AreTomo2 (its folder says
# 1.1.2, it reports 1.1.3), the last two with CUDA 12's libraries. A value the module already set is kept.
export IMOD_DIR=${IMOD_DIR:-/opt/cryoem/imod/5.2.1}
export CTFFIND5=${CTFFIND5:-/opt/cryoem/ctffind/5.0.2/bin/ctffind}
export MOTIONCOR3=${MOTIONCOR3:-/opt/cryoem/motioncor3/1.1.1/bin/MotionCor3}
export ARETOMO=${ARETOMO:-/opt/cryoem/aretomo2/1.1.2/bin/AreTomo2}
case ":${LD_LIBRARY_PATH:-}:" in
    *:/opt/tcbsys/cuda/12.6.2/lib64:*) ;;
    *) export LD_LIBRARY_PATH=/opt/tcbsys/cuda/12.6.2/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ;;
esac

# pytom-match-pick and MemBrain-seg, each in an environment of its own.
export PYTOMO_MP_DIR=$MAMBA_ROOT_PREFIX/envs/pytom_tm/bin
export MEMBRAIN_EXE=$MAMBA_ROOT_PREFIX/envs/membrain/bin/membrain
# CuPy's CUB off: pytom-match-pick's CuPy (14.2.0, built for CUDA 12.9) cannot
# compile its CUB reduction here, and picking failed at its first angle on 25
# November 2026; the same kernel compiles and sums right without CUB. Playbook
# entry I-9.
export CUPY_ACCELERATORS=

# The clone's root and the pipeline's environment, once they exist.
if [[ -d $CRYOET_WORK/cryoet-pipeline ]]; then
    cd "$CRYOET_WORK/cryoet-pipeline" || return 1 2>/dev/null || exit 1
fi
if [[ -x $MAMBA_ROOT_PREFIX/envs/cryoet-pipeline/bin/python ]]; then
    micromamba activate cryoet-pipeline
    echo "pipeline session: $(pwd), environment cryoet-pipeline"
fi
