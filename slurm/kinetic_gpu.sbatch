#!/bin/bash
# slurm/kinetic_gpu.sbatch
#
# ACC single-GPU job: the kinetic track via the real `train kinetic <FAMILY>
# --label rate_k_per_day [--log-space] --datasets DIR` (see
# src/p2e_formulations_pipeline/cli.py + models/kinetic_targets.py). FAMILY
# is positional. Valid --label values: rate_k_per_day | asymptote_A |
# endpoint_value | time_to_50 | time_to_90.
#
# NOTE (confirmed against the shipped command body): `train kinetic`
# currently trains the tabular model zoo ON the kinetic parameter (composition
# -> rate constant) under grouped-CV — a true Fick/Avrami-constrained PINN/ODE
# is deferred. Accordingly there is no --runs and no --checkpoint-dir flag;
# the CLI only prints a JSON report to stdout, which this script captures to
# a file (see COMMAND section below). This template stays queued on ACC
# (single GPU) ahead of that future PINN track landing; today's run doesn't
# need the GPU, so don't read GPU utilization here as a bug.
#
# Single GPU by design — no torchrun/rendezvous machinery here (that's for
# multi-GPU/multi-node DDP; see the hpc-dl-perf skill's
# 08-slurm-launch-patterns.md and slurm/nn_gpu.sbatch's header note).
#
# Submit via scripts/mn5/submit_gpu.sh kinetic --family F [--label L]
# [--log-space] [--nsys] — it derives --cpus-per-task from --gpus (20 CPU/GPU
# rule) and passes PROJECT_ROOT/SCRATCH_ROOT/DATASETS_DIR/RESULTS_DIR/FAMILY/
# LABEL/LOG_SPACE/NSYS_PROFILE via --export. The #SBATCH headers below are
# safe literal 1-GPU defaults for direct `sbatch slurm/kinetic_gpu.sbatch`
# use; submit_gpu.sh's CLI flags override them.
#
#SBATCH --job-name=p2e-kinetic
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.err

set -euo pipefail

# ==============================================================================
# Preflight
# ==============================================================================
echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "JobName:      ${SLURM_JOB_NAME:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "CPUsPerTask:  ${SLURM_CPUS_PER_TASK:-unset}"
echo "GPUs:         ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Partition:    ${SLURM_JOB_PARTITION:-unset}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# CRITICAL (confirmed BSC gotcha): srun no longer inherits --cpus-per-task
# from #SBATCH. Every GPU job that uses srun must export this or the srun'd
# process silently gets 1 CPU.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# ==============================================================================
# Paths — literal BSC fallbacks so this still runs if launched without
# submit_gpu.sh's --export.
# ==============================================================================
ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-formulations.sif}"
DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/var/model_datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
# Default to one of the kinetic-bearing families (dataset_kinetic_biodegradation.jsonl
# etc. — see var/model_datasets/). Override via submit_gpu.sh --family.
FAMILY="${FAMILY:-biodegradation}"
LABEL="${LABEL:-rate_k_per_day}"   # rate_k_per_day | asymptote_A | endpoint_value | time_to_50 | time_to_90
LOG_SPACE="${LOG_SPACE:-0}"        # 1 -> pass --log-space (train on log10 of the label)

echo "Family:       ${FAMILY}"
echo "Label:        ${LABEL}"
echo "Log-space:    ${LOG_SPACE}"
echo "Container:    ${SIF}"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}"
cd "${JOB_SCRATCH}"

# Caches — keep off $HOME (small/rate-limited on HPC clusters); harmless
# forward-compat for the future PINN/ODE track, unused by today's tabular fit.
export XDG_CACHE_HOME="${JOB_SCRATCH}/.cache"
export TRITON_CACHE_DIR="${JOB_SCRATCH}/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"

# ==============================================================================
# COMMAND — the exact CLI invocation for this job. Adjust here.
# ==============================================================================
P2E_CMD=(p2e-formulations train kinetic "${FAMILY}"
  --label "${LABEL}"
  --datasets "${DATASETS_DIR}")
if [[ "${LOG_SPACE}" == "1" ]]; then
  P2E_CMD+=(--log-space)
fi

# Optional Nsight Systems profiling pass (hpc-dl-perf skill,
# 01-tracing-toolchain.md). OFF by default — toggle with `submit_gpu.sh
# kinetic --nsys` (sets NSYS_PROFILE=1) or `export NSYS_PROFILE=1` before
# sbatch. Only meaningful once the real GPU-bound PINN/ODE track lands;
# today's tabular fit is CPU-bound so a trace mostly shows idle GPU.
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
  NSYS_BIN="${NSYS_BIN:-nsys}"
  TRACE_DIR="${JOB_SCRATCH}/nsys"
  mkdir -p "${TRACE_DIR}"
  echo "nsys:         ON -> ${TRACE_DIR}"
  P2E_CMD=("${NSYS_BIN}" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none --cpuctxsw=none --cuda-memory-usage=true \
    --force-overwrite=true \
    --output "${TRACE_DIR}/report-%h-%p" \
    "${P2E_CMD[@]}")
fi

echo "Command:      ${P2E_CMD[*]}"

# `train kinetic` has no --runs flag — it prints its report as JSON to stdout
# and persists nothing else on its own. Capture that JSON to a file (via
# `tee`, so it's still visible live in the SLURM .out log) so there's a
# discrete artifact to stage back below.
REPORT_FILE="${JOB_SCRATCH}/kinetic_report.json"
# Singularity is a MODULE on MN5, not on the default PATH — load it or `singularity exec` fails instantly
# with "execve(): singularity: No such file or directory".
module load singularity/4.1.5 2>/dev/null || module load singularity
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" \
    --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    "${SIF}" \
    "${P2E_CMD[@]}" | tee "${REPORT_FILE}"

# ==============================================================================
# Stage results back to the durable project store — scratch is not backed up.
# ==============================================================================
mkdir -p "${RESULTS_DIR}/kinetic/${FAMILY}/${SLURM_JOB_ID}"
rsync -a --exclude '.cache/' "${JOB_SCRATCH}/" "${RESULTS_DIR}/kinetic/${FAMILY}/${SLURM_JOB_ID}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Report:       ${RESULTS_DIR}/kinetic/${FAMILY}/${SLURM_JOB_ID}/kinetic_report.json"
