#!/bin/bash -l
# NOTE: login shell (-l) is REQUIRED so lmod's `module` command is initialized in the batch environment
# (same reason as bake_off_real.sbatch — a plain `#!/bin/bash` does not source lmod and `module load` dies
# under set -e before the job starts).
#
# MN5 Optuna HP-tuning run for ALL tensile_strength zoo models, on the PROVEN CPU venv+wheel path (the same
# offline-install pattern as bake_off_real.sbatch, job 42843426). Nested grouped-by-paper CV: the inner
# folds drive each Optuna trial's HP search; the untouched outer fold is scored once for the honest RMSE.
# One process per model, looped sequentially (tune() runs all N trials in-process — no shared Optuna store
# yet). Each model's TuneResult JSON (best_params + best_inner_score + honest_outer_rmse) is staged back.
#
# Prereqs on GPFS (push from laptop via a transfer node BEFORE submitting — scripts/mn5/sync_push.sh):
#   var/mn5_bundle/wheels/     <- REBUILT bundle incl. optuna (scripts/mn5/build_wheels.sh)
#   var/model_datasets/        <- the re-encoded dataset_tensile_strength.csv (descriptors+physics+conditions)
#   config/modelling.yaml      <- carries tune.search_spaces (rf/gbm/lightgbm/ridge/knn)
#   slurm/tune_tensile.sbatch
#
# Submit from an ACC login node (ehpc838 is GPU-only; ACC jobs go from alogin):
#   ssh pack423506@alogin1.bsc.es
#   cd /gpfs/projects/ehpc838/pack423506
#   sbatch slurm/tune_tensile.sbatch                       # 50 trials/model (config default)
#   N_TRIALS=100 sbatch slurm/tune_tensile.sbatch          # override the budget
#   squeue -j <jobid>
#   # results -> results/tune/tensile_<model>.json  (pull back with scripts/mn5/sync_pull.sh)

#SBATCH --job-name=tune_tensile
#SBATCH -A ehpc838
#SBATCH -q acc_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o tune_tensile_%j.out
#SBATCH -e tune_tensile_%j.err

set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
WHEELS="${WHEELS:-$PROJECT/var/mn5_bundle/wheels}"
VENV="${VENV:-$PROJECT/venv312}"
DATASETS="${DATASETS:-$PROJECT/var/model_datasets}"
RESULTS="${RESULTS:-$PROJECT/results/tune}"
FAMILY="${FAMILY:-tensile_strength}"
# global_mean has no hyperparameters — every other zoo model gets tuned.
MODELS="${MODELS:-random_forest gbm lightgbm ridge knn}"
N_TRIALS="${N_TRIALS:-50}"
mkdir -p "$RESULTS"

echo "Node:     $(hostname)"
echo "Project:  $PROJECT"
echo "Family:   $FAMILY"
echo "Models:   $MODELS"
echo "Trials:   $N_TRIALS"

# --- python 3.12 module chain (hierarchical: intel -> mkl -> python/3.12.1), gives cp312 for the wheels ---
module load intel mkl python/3.12.1
PYVER="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.12" ] || { echo "ERROR: need Python 3.12 for the cp312 wheels, got $PYVER — module load failed"; exit 1; }

# --- venv from the module python, isolated from the module's PYTHONHOME/PYTHONPATH ---
NEED_INSTALL=0
if [ ! -x "$VENV/bin/p2e-formulations" ]; then
  echo "[setup] creating venv $VENV"
  python -m venv "$VENV"
  "$VENV/bin/python" -m ensurepip --default-pip 2>/dev/null || true
  NEED_INSTALL=1
else
  echo "[setup] reusing existing venv $VENV"
fi
unset PYTHONHOME PYTHONPATH

if [ "$NEED_INSTALL" = 1 ]; then
  echo "[setup] offline install from $WHEELS (incl. optuna + pysqlite3 for tuning)"
  "$VENV/bin/pip" install --no-index --find-links "$WHEELS" \
    p2e-formulations-data-pipeline numpy scikit-learn pandas joblib lightgbm optuna pysqlite3-binary
fi

# --- sqlite3 shim: MN5's module Python 3.12.1 (INTEL) ships a broken _sqlite3 (undefined symbol
#     sqlite3_deserialize); optuna imports sqlite3 eagerly even for in-memory studies. Install a
#     sitecustomize into the venv that swaps in pysqlite3 (modern bundled SQLite) whenever the stdlib
#     sqlite3 fails to import. Written every run (idempotent, cheap). ---
SITE="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SITE/sitecustomize.py" <<'PYEOF'
import sys
try:
    import sqlite3  # noqa: F401  (stdlib; broken on MN5's INTEL Python build)
except Exception:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
PYEOF
echo "[setup] wrote sqlite3->pysqlite3 shim to $SITE/sitecustomize.py"

# --- sanity: CLI + optuna both resolve from the offline stack (optuna import exercises the sqlite shim) ---
"$VENV/bin/python" -c "import p2e_formulations_pipeline.cli, optuna; print('cli + optuna import OK', optuna.__version__)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# --- tune each model; capture the TuneResult JSON per model (honest outer RMSE lives in it) ---
for M in $MODELS; do
  echo "[tune] $FAMILY / $M ($N_TRIALS trials) ..."
  "$VENV/bin/p2e-formulations" train tune "$FAMILY" \
    --model "$M" \
    --n-trials "$N_TRIALS" \
    --datasets "$DATASETS" \
    --root "$PROJECT" | tee "$RESULTS/${FAMILY}_${M}.json"
done

echo "[tune] done -> $RESULTS/${FAMILY}_<model>.json"
