#!/bin/bash
# MN5 loop smoke test — pure-stdlib job on the ACC (GPU) debug queue. Proves transfer -> sbatch -> results-back.
#
# NOTE: account ehpc838 is GPU-only — `bsc_queues` shows only acc_* queues (no GPP). So this runs on ACC.
# The job itself is CPU-only stdlib (no GPU needed); we just use the acc_debug QOS because it's the only debug
# queue available. A non-fatal nvidia-smi/module probe at the end harvests stage-2 info (GPU + Python/CUDA).
#
# Submit from an ACC login node (alogin1/alogin2), from your project dir on GPFS (NOT /gpfs/home):
#   ssh pack423506@alogin1.bsc.es
#   cd /gpfs/projects/ehpc838/pack423506
#   sbatch slurm/smoke_test.sbatch
#   squeue -j <jobid>              # watch it schedule/run (debug queue is fast)
#   tail -f smoke_test_<jobid>.out # or read logs/ after it finishes

#SBATCH --job-name=smoke_test
#SBATCH -A ehpc838
#SBATCH -q acc_debug             # <-- ACC debug QOS (2h). Only debug queue for ehpc838 (GPU-only account).
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH -o smoke_test_%j.out
#SBATCH -e smoke_test_%j.err
# If SLURM rejects a CPU-only job on this queue, uncomment the next two lines (request 1 GPU, BSC 20-CPU/GPU rule):
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=20

set -euo pipefail

# --- paths (override by exporting before sbatch, or edit here) ---
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
DATASET="${DATASET:-$PROJECT_ROOT/var/model_datasets/dataset_tensile_strength.csv}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/smoke}"
mkdir -p "$RESULTS_DIR"

# --- preflight audit trail ---
echo "JobID:        ${SLURM_JOB_ID:-n/a}"
echo "Node:         $(hostname)"
echo "SubmitDir:    ${SLURM_SUBMIT_DIR:-n/a}"
echo "Dataset:      $DATASET"
echo "ResultsDir:   $RESULTS_DIR"

# --- load a Python (pure-stdlib job, so any python3 works) ---
module load python 2>/dev/null || echo "note: 'module load python' unavailable; falling back to python3 on PATH"
PY="$(command -v python3 || command -v python)"
echo "Python:       $PY ($("$PY" --version 2>&1))"

# --- run ---
"$PY" "$PROJECT_ROOT/scripts/mn5/smoke_job.py" "$DATASET" "$RESULTS_DIR/smoke_result.json"

echo "[smoke] done -> $RESULTS_DIR/smoke_result.json"

# --- stage-2 recon (non-fatal; harvests what the real training run needs) ---
echo "=== stage-2 recon (non-fatal) ==="
nvidia-smi -L 2>/dev/null || echo "no nvidia-smi / no GPU visible to this job"
module avail 2>&1 | grep -iE "python|pytorch|cuda|cudnn|singularity" | head -30 || echo "module avail: nothing matched or module unavailable"
