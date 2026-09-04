#!/bin/bash
# slurm/bake_off_array.sbatch
#
# GPP CPU array job: one array task = one property family, matching the real
# `train bake-off --family <F> --datasets DIR --runs DIR` (see
# src/p2e_formulations_pipeline/cli.py). Upstream --family is repeatable and
# --model is repeatable (default: config default_models — the whole point of
# a bake-off is to sweep the full zoo, so this script deliberately never
# passes --model). It writes runs_dir/leaderboard.json + per-model cards.
#
# Submit via scripts/mn5/submit_array.sh bake-off --family f1 --family f2
# ... — it builds the manifest (one family per line), sizes --array=0-N%K,
# and passes MANIFEST/PROJECT_ROOT/SCRATCH_ROOT/DATASETS_DIR/RESULTS_DIR via
# --export. The #SBATCH headers below are safe literal defaults for direct
# `sbatch slurm/bake_off_array.sbatch` use; submit_array.sh's CLI flags
# override them.
#
#SBATCH --job-name=p2e-bake-off
#SBATCH -A ehpc838
#SBATCH -q gp_ehpc
#SBATCH --array=0-0%4
#SBATCH -c 8
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%A_%a.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%A_%a.err

set -euo pipefail

# ==============================================================================
# Preflight — always know what you're looking at six weeks from now.
# ==============================================================================
echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "ArrayJobID:   ${SLURM_ARRAY_JOB_ID:-unset}"
echo "ArrayTaskID:  ${SLURM_ARRAY_TASK_ID:-unset}"
echo "JobName:      ${SLURM_JOB_NAME:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "CPUsPerTask:  ${SLURM_CPUS_PER_TASK:-unset}"
echo "Partition:    ${SLURM_JOB_PARTITION:-unset}"
echo "SubmitDir:    ${SLURM_SUBMIT_DIR:-unset}"
echo "Start:        $(date -u +%FT%TZ)"

# ==============================================================================
# Paths — literal BSC fallbacks so this still runs if launched without
# submit_array.sh's --export (e.g. `source bsc_project ehpc838` not sourced).
# ==============================================================================
ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
MANIFEST="${MANIFEST:-${SCRATCH_ROOT}/manifests/bake_off_latest.tsv}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-formulations.sif}"
DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/var/model_datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found at ${MANIFEST} — submit via scripts/mn5/submit_array.sh bake-off, which writes and exports it." >&2
  exit 1
fi

TASK_LINE=$((SLURM_ARRAY_TASK_ID + 1))
FAMILY="$(sed -n "${TASK_LINE}p" "${MANIFEST}")"
if [[ -z "${FAMILY}" ]]; then
  echo "ERROR: no manifest line ${TASK_LINE} in ${MANIFEST} (array task ${SLURM_ARRAY_TASK_ID})" >&2
  exit 1
fi
echo "Family:       ${FAMILY}"
echo "Manifest:     ${MANIFEST}"
echo "Container:    ${SIF}"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${JOB_SCRATCH}"
cd "${JOB_SCRATCH}"

# ==============================================================================
# Per-task MLflow store — each array task logs to its OWN store dir (unique by
# job+task id), so concurrent tasks never race on one GPFS file store. They are
# consolidated locally after pull with `p2e-formulations train tracking-merge`
# (see docs/mn5-workflow.md). ExperimentTracker honours MLFLOW_TRACKING_URI.
# ==============================================================================
TASK_TAG="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MLRUNS_TASK_DIR="${RESULTS_DIR}/mlruns_tasks/${TASK_TAG}"
mkdir -p "${MLRUNS_TASK_DIR}"
echo "MLflow store: ${MLRUNS_TASK_DIR}"

# ==============================================================================
# COMMAND — the exact CLI invocation for this task. Adjust here if `train
# bake-off`'s flags change.
# ==============================================================================
P2E_CMD=(p2e-formulations train bake-off
  --family "${FAMILY}"
  --datasets "${DATASETS_DIR}"
  --runs "${JOB_SCRATCH}")

echo "Command:      ${P2E_CMD[*]}"

singularity exec \
  --env "MLFLOW_TRACKING_URI=file://${MLRUNS_TASK_DIR}" \
  --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" \
  --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}" \
  "${SIF}" \
  "${P2E_CMD[@]}"

# ==============================================================================
# Stage results back to the durable project store — scratch is not backed up.
# ==============================================================================
mkdir -p "${RESULTS_DIR}/${FAMILY}"
rsync -a "${JOB_SCRATCH}/" "${RESULTS_DIR}/${FAMILY}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Results:      ${RESULTS_DIR}/${FAMILY}"
