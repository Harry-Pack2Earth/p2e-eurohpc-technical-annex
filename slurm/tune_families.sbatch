#!/bin/bash -l
# NOTE: login shell (-l) REQUIRED so lmod's `module` is initialised (see bake_off_real.sbatch).
#
# MN5 Optuna HP-tuning sweep over the WINNABLE property families x all zoo models, on the proven CPU
# venv+wheel path (same as tune_tensile.sbatch). Nested grouped-by-paper CV: inner folds drive each
# trial's search, the untouched outer fold gives the honest RMSE. One process per (family, model); JSON
# per combo staged back.
#
# HONEST SCOPE: tuning extracts the last few points and rescues weak model defaults — it does NOT raise
# the ceiling. The ceiling movers are registry coverage, latent variables, and lab data (see
# docs/formulation-encoding-standard.md). Run this for the tuned demo baseline, not to reach 0.9.
#
# Prereqs on GPFS (push from laptop FIRST — the datasets were re-encoded with coverage + physics config):
#   scripts/mn5/sync_push.sh --code --data --bundle
#
# Submit from an ACC login node:
#   ssh pack423506@alogin1.bsc.es
#   cd /gpfs/projects/ehpc838/pack423506/p2e-formulations-modelling
#   sbatch slurm/tune_families.sbatch
#   # override scope: FAMILIES="melting_temperature glass_transition_temperature" sbatch slurm/tune_families.sbatch
#   squeue -u pack423506
#   # results -> results/tune/<family>_<model>.json  (pull: scripts/mn5/sync_pull.sh --results)

#SBATCH --job-name=tune_families
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH -o tune_families_%j.out
#SBATCH -e tune_families_%j.err

set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
WHEELS="${WHEELS:-$PROJECT/var/mn5_bundle/wheels}"
VENV="${VENV:-$PROJECT/venv312}"
DATASETS="${DATASETS:-$PROJECT/var/model_datasets}"
RESULTS="${RESULTS:-$PROJECT/results/tune}"
# The winnable set (physics-Δ + ML). Override with FAMILIES=... at submit time.
FAMILIES="${FAMILIES:-melting_temperature glass_transition_temperature crystallinity water_contact_angle crystallization_enthalpy elongation_at_break tensile_modulus film_geometry thermal_degradation_onset}"
# Full CPU zoo. mlp (torch NN) is NOT here — it needs [torch] + a GPU sweep (separate); this bundle is
# CPU-only wheels. xgboost skips gracefully if its wheel is missing from the bundle.
MODELS="${MODELS:-random_forest extra_trees gbm lightgbm xgboost ridge elastic_net svr knn}"   # global_mean has no HP
N_TRIALS="${N_TRIALS:-50}"
mkdir -p "$RESULTS"

echo "Node:     $(hostname)"
echo "Families: $FAMILIES"
echo "Models:   $MODELS   Trials: $N_TRIALS"

# --- python 3.12 module chain (hierarchical intel -> mkl -> python/3.12.1); gives cp312 for the wheels ---
module load intel mkl python/3.12.1
PYVER="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.12" ] || { echo "ERROR: need Python 3.12, got $PYVER"; exit 1; }

NEED_INSTALL=0
if [ ! -x "$VENV/bin/p2e-formulations" ]; then
  echo "[setup] creating venv $VENV"; python -m venv "$VENV"
  "$VENV/bin/python" -m ensurepip --default-pip 2>/dev/null || true; NEED_INSTALL=1
else
  echo "[setup] reusing venv $VENV"
fi
unset PYTHONHOME PYTHONPATH

if [ "$NEED_INSTALL" = 1 ]; then
  echo "[setup] offline install from $WHEELS (optuna + pysqlite3)"
  "$VENV/bin/pip" install --no-index --find-links "$WHEELS" \
    p2e-formulations-data-pipeline numpy scikit-learn pandas joblib lightgbm optuna pysqlite3-binary
  # xgboost declares nvidia-nccl-cu12 (x86_64-only) as a hard dep, absent from the arm64-built bundle →
  # --no-deps installs just the CPU xgboost wheel (numpy/scipy already present; nccl is multi-GPU-only).
  "$VENV/bin/pip" install --no-index --no-deps --find-links "$WHEELS" xgboost
fi

# --- sqlite3 shim for MN5's broken INTEL _sqlite3 (undefined symbol sqlite3_deserialize); optuna needs it ---
SITE="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
cat > "$SITE/sitecustomize.py" <<'PYEOF'
import sys
try:
    import sqlite3  # noqa: F401
except Exception:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    sys.modules["sqlite3.dbapi2"] = pysqlite3.dbapi2
PYEOF

"$VENV/bin/python" -c "import p2e_formulations_pipeline.cli, optuna; print('imports OK', optuna.__version__)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
for FAM in $FAMILIES; do
  if [ ! -f "$DATASETS/dataset_${FAM}.csv" ]; then echo "[skip] no dataset_${FAM}.csv"; continue; fi
  for M in $MODELS; do
    echo "[tune] $FAM / $M ($N_TRIALS trials) ..."
    "$VENV/bin/p2e-formulations" train tune "$FAM" \
      --model "$M" --n-trials "$N_TRIALS" --datasets "$DATASETS" --root "$PROJECT" \
      | tee "$RESULTS/${FAM}_${M}.json" || echo "[warn] $FAM/$M failed, continuing"
  done
done

echo "[tune] sweep done -> $RESULTS/<family>_<model>.json"
