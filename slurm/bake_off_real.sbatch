#!/bin/bash -l
# NOTE: login shell (-l) is REQUIRED so lmod's `module` command is initialized in the batch environment.
# A plain `#!/bin/bash` batch shell does not source lmod, so `module load` fails (and under `set -e` kills
# the job before it starts) — that is exactly what sank the first attempt.
#
# MN5 stage-2 REAL run: offline-install the CPU training stack from the pre-built wheel bundle, then run the
# grouped-CV bake-off and stage the leaderboard back. No internet needed on the node — everything installs
# from wheels shipped by scripts/mn5/build_wheels.sh. First run uses the ACC debug queue as a CPU-only job
# (no GPU requested; smoke test 42836356 proved CPU-only runs on acc_debug).
#
# Prereqs on GPFS (push from laptop via a transfer node BEFORE submitting):
#   /gpfs/projects/ehpc838/pack423506/var/mn5_bundle/wheels/   <- from scripts/mn5/build_wheels.sh
#   /gpfs/projects/ehpc838/pack423506/var/model_datasets/      <- the dataset_*.csv to train on
#   /gpfs/projects/ehpc838/pack423506/slurm/bake_off_real.sbatch
#
# Submit from an ACC login node (account ehpc838 is GPU-only; ACC jobs go from alogin, not glogin):
#   ssh pack423506@alogin1.bsc.es
#   cd /gpfs/projects/ehpc838/pack423506
#   sbatch slurm/bake_off_real.sbatch          # or: FAMILY=elongation_at_break sbatch slurm/bake_off_real.sbatch
#   squeue -j <jobid>
#   # result -> results/bake_off/leaderboard.json  (pull back with rsync)

#SBATCH --job-name=bakeoff_real
#SBATCH -A ehpc838
#SBATCH -q acc_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH -o bakeoff_real_%j.out
#SBATCH -e bakeoff_real_%j.err

set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
WHEELS="${WHEELS:-$PROJECT/var/mn5_bundle/wheels}"
VENV="${VENV:-$PROJECT/venv312}"
DATASETS="${DATASETS:-$PROJECT/var/model_datasets}"
RESULTS="${RESULTS:-$PROJECT/results/bake_off}"
FAMILY="${FAMILY:-tensile_strength}"
mkdir -p "$RESULTS"

echo "Node:     $(hostname)"
echo "Project:  $PROJECT"
echo "Wheels:   $WHEELS"
echo "Family:   $FAMILY"

# --- python 3.12 module. python/3.12.1 is HIERARCHICAL — Lmod requires the intel toolchain loaded first:
#     first `intel`, then `mkl` (discovered by loading them one at a time). Load the full chain then python;
#     it gives cp312, which is all the manylinux wheel bundle needs. Confirm the chain interactively on
#     alogin1 (`module purge; module load intel mkl python/3.12.1 && python --version`) before submitting. ---
module load intel mkl python/3.12.1
echo "Python:   $(command -v python) ($(python --version 2>&1))"
# Hard assert: the cp312 wheel bundle CANNOT install into any other Python. Fail loud if the module didn't take.
PYVER="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.12" ] || { echo "ERROR: need Python 3.12 for the cp312 wheels, got $PYVER — module load failed"; exit 1; }

# --- build the venv once with the module python, then ISOLATE it from the module's PYTHONHOME/PYTHONPATH
#     (the python module exports both; left set they shadow our pinned deps or break the venv python). ---
NEED_INSTALL=0
if [ ! -x "$VENV/bin/p2e-formulations" ]; then
  echo "[setup] creating venv $VENV"
  python -m venv "$VENV"
  "$VENV/bin/python" -m ensurepip --default-pip 2>/dev/null || true   # safety if the module python lacks pip
  NEED_INSTALL=1
else
  echo "[setup] reusing existing venv $VENV"
fi
unset PYTHONHOME PYTHONPATH        # from here on the venv is fully self-contained

if [ "$NEED_INSTALL" = 1 ]; then
  echo "[setup] offline install from $WHEELS"
  "$VENV/bin/pip" install --no-index --find-links "$WHEELS" \
    p2e-formulations-data-pipeline numpy scikit-learn pandas joblib lightgbm
fi

# --- sanity: the CLI import closure resolves with the offline stack ---
"$VENV/bin/python" -c "import p2e_formulations_pipeline.cli; print('cli import OK')"

# --- run the grouped-CV bake-off; stage the leaderboard + model cards into results/ ---
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
"$VENV/bin/p2e-formulations" train bake-off \
  --family "$FAMILY" \
  --datasets "$DATASETS" \
  --runs "$RESULTS" \
  --root "$PROJECT"

echo "[bakeoff] done -> $RESULTS/leaderboard.json"
