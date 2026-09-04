#!/bin/bash
# slurm/tune_array.sbatch
#
# GPP CPU array job: one array task = one (family, model) PAIR, matching the
# real `train tune <FAMILY> --model <M> [--n-trials <N>] --datasets DIR` (see
# src/p2e_formulations_pipeline/cli.py / models/tuning.py). FAMILY is
# positional; there is no --runs flag — `tune()` returns a result object and
# the CLI only prints it as JSON, so this script captures that stdout JSON
# into a file for staging back (see COMMAND section below).
#
# IMPORTANT: `tune()` runs all N trials in ONE process and does NOT yet
# support shared/distributed Optuna storage (no --optuna-storage flag
# upstream). So the array unit is the (family, model) pair, not the trial —
# parallelism is across family x model, not across trials. Distributed
# Optuna-over-RDB is a future enhancement once that flag exists; there is no
# shared-storage plumbing in this script to remove-if-wrong, because there
# never was a working version of it upstream to plumb into.
#
# Submit via scripts/mn5/submit_array.sh tune --family F [--family F ...]
# --model M [--model M ...] [--n-trials N] — it builds the manifest
# (family\tmodel\tn_trials per line, one row per family x model pair), sizes
# --array=0-N%K, and passes MANIFEST/PROJECT_ROOT/SCRATCH_ROOT/DATASETS_DIR/
# RESULTS_DIR via --export. The #SBATCH headers below are safe literal
# defaults for direct `sbatch slurm/tune_array.sbatch` use; submit_array.sh's
# CLI flags override them.
#
#SBATCH --job-name=p2e-tune
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
# Preflight
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
# submit_array.sh's --export.
# ==============================================================================
ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
MANIFEST="${MANIFEST:-${SCRATCH_ROOT}/manifests/tune_latest.tsv}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-formulations.sif}"
DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/var/model_datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest not found at ${MANIFEST} — submit via scripts/mn5/submit_array.sh tune, which writes and exports it." >&2
  exit 1
fi

TASK_LINE=$((SLURM_ARRAY_TASK_ID + 1))
LINE="$(sed -n "${TASK_LINE}p" "${MANIFEST}")"
if [[ -z "${LINE}" ]]; then
  echo "ERROR: no manifest line ${TASK_LINE} in ${MANIFEST} (array task ${SLURM_ARRAY_TASK_ID})" >&2
  exit 1
fi
IFS=$'\t' read -r FAMILY MODEL N_TRIALS <<< "${LINE}"
echo "Family:       ${FAMILY}"
echo "Model:        ${MODEL}"
echo "N trials:     ${N_TRIALS:-<config default>}"
echo "Manifest:     ${MANIFEST}"
echo "Container:    ${SIF}"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${JOB_SCRATCH}"
cd "${JOB_SCRATCH}"

# ==============================================================================
# Per-task MLflow store — one store dir per array task (unique by job+task id),
# so concurrent trials never race on a shared GPFS file store. Consolidated
# locally after pull with `p2e-formulations train tracking-merge`. The tune run
# logs best HP + full search space + honest scores here.
# ==============================================================================
TASK_TAG="${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MLRUNS_TASK_DIR="${RESULTS_DIR}/mlruns_tasks/${TASK_TAG}"
mkdir -p "${MLRUNS_TASK_DIR}"
echo "MLflow store: ${MLRUNS_TASK_DIR}"

# ==============================================================================
# COMMAND — the exact CLI invocation for this task. Adjust here if `train
# tune`'s flags change. FAMILY is positional; --n-trials is only added when
# the manifest carries a value, otherwise `tune()` uses config's tune.n_trials
# default (50 as of this writing — see config/modelling.yaml).
# ==============================================================================
P2E_CMD=(p2e-formulations train tune "${FAMILY}"
  --model "${MODEL}"
  --datasets "${DATASETS_DIR}")
if [[ -n "${N_TRIALS}" ]]; then
  P2E_CMD+=(--n-trials "${N_TRIALS}")
fi

echo "Command:      ${P2E_CMD[*]}"

# `train tune` has no --runs flag — it prints its TuneResult as JSON to
# stdout and persists nothing else on its own. Capture that JSON to a file
# (via `tee`, so it's still visible live in the SLURM .out log) so there's a
# discrete artifact to stage back below.
REPORT_FILE="${JOB_SCRATCH}/tune_report.json"
singularity exec \
  --env "MLFLOW_TRACKING_URI=file://${MLRUNS_TASK_DIR}" \
  --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" \
  --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}" \
  "${SIF}" \
  "${P2E_CMD[@]}" | tee "${REPORT_FILE}"

# ==============================================================================
# Stage results back to the durable project store — scratch is not backed up.
# ==============================================================================
mkdir -p "${RESULTS_DIR}/tune/${FAMILY}_${MODEL}"
rsync -a "${JOB_SCRATCH}/" "${RESULTS_DIR}/tune/${FAMILY}_${MODEL}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Report:       ${RESULTS_DIR}/tune/${FAMILY}_${MODEL}/tune_report.json"
