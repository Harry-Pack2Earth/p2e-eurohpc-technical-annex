#!/bin/bash
# slurm/md_proxy_gpu.sbatch
#
# Phase-8 / ② LATER — ACC single-GPU job: MD gas-diffusion barrier corroborator (SCAFFOLD, NOT BUILT).
# Estimates penetrant D (MSD) + S (Widom) in an amorphous cell -> P=D*S, an independent forward witness for the
# ML barrier prediction with no measured P0. Mirrors mlip_embed_gpu.sbatch exactly (bind-mounted scripts, the
# singularity module load, smoke-first via --limit, durable shards on the project store).
#
# NOT RUNNABLE until scripts/md_proxy_openmm.py stubs are filled AND container/mn5-md.def is built+pushed.
# The driver currently REFUSES to fabricate a result (raises at the MD step), so a premature submit fails loud
# rather than staging fake numbers. See docs/phase8-md-barrier-scaffold.md. Reconfirm bsc_acct before spend.
#
# Submit (once built) via: scripts/mn5/submit_gpu.sh md [--qos acc_debug|acc_ehpc] [--time HH:MM:SS] [--limit N]
#
#SBATCH --job-name=p2e-md-barrier
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.err

set -euo pipefail

echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SIF="${SIF:-${PROJECT_ROOT}/container/mn5-md.sif}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
SHARDS_DIR="${SHARDS_DIR:-${PROJECT_ROOT}/var/md/shards}"
LIMIT_ARG=()
[[ -n "${MD_LIMIT:-}" ]] && LIMIT_ARG=(--limit "${MD_LIMIT}")

if [[ ! -f "${SIF}" ]]; then
  echo "FATAL: ${SIF} not found. Build container/mn5-md.def (Rosetta amd64 path) + sync_push.sh --container."
  echo "This method is a SCAFFOLD — see docs/phase8-md-barrier-scaffold.md."
  exit 1
fi

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}" "${SHARDS_DIR}"
cd "${JOB_SCRATCH}"

module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

BIND_ARGS=(--bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}")

# Run the bind-mounted driver (code from the project store, container supplies only the environment).
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv "${BIND_ARGS[@]}" "${SIF}" \
    python "${PROJECT_ROOT}/scripts/md_proxy_openmm.py" \
      --shards "${SHARDS_DIR}" "${LIMIT_ARG[@]}"

mkdir -p "${RESULTS_DIR}/md/${SLURM_JOB_ID}"
rsync -a "${SHARDS_DIR}/" "${RESULTS_DIR}/md/${SLURM_JOB_ID}/" 2>/dev/null || true
echo "End:          $(date -u +%FT%TZ)"
echo "Next:         pull results/md/<job>/, review, then wire BarrierMDCheck (kind 'md') in corroboration.py."
