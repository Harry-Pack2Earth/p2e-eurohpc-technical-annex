#!/bin/bash -l
# PARALLEL Optuna tuning sweep — the fast path. One SLURM ARRAY TASK = one (family, model) pair, so the
# whole family×model grid runs CONCURRENTLY across CPU cores instead of a sequential loop on one node
# (tune_families.sbatch). 81 combos that took ~2-3 h serially finish in ~10 min when 14+ run at once.
#
# QUEUE: ehpc838 is an ACC-only allocation (no GPP grant) — so this uses -q acc_ehpc and runs on ACC
# nodes' CPU cores (80 core / node). gp_ehpc from alogin1 fails with "Invalid qos specification". We do
# NOT request a GPU (no --gres): see WHY-CPU below.
#
# WHY CPU, NOT GPU: the zoo is tree/linear/kernel algorithms on 58–500-row tables. Those are CPU
# algorithms; a GPU sits idle or is SLOWER (host↔device transfer dwarfs the compute at this size). The
# speed lever here is task-parallelism across cores (this array), not the GPU. The GPU is for the torch NN
# / kinetic-PINN track only (nn_gpu.sbatch, kinetic_gpu.sbatch).
#
# Uses the PROVEN venv+wheel+sqlite-shim deployment (same as tune_families.sbatch), NOT the unbuilt
# singularity container that tune_array.sbatch expects. A flock guards the one-time venv build so
# concurrent array tasks don't race to create it.
#
# Prereqs (laptop): bash scripts/mn5/build_wheels.sh && scripts/mn5/sync_push.sh --code --data --bundle
# Submit (alogin1), array size = (#families × #models) − 1 = 9×9−1 = 80; %14 = max concurrent tasks
# (8 cores each → ~2 ACC nodes' worth). Raise/lower %N to taste:
#   cd /gpfs/projects/ehpc838/pack423506/p2e-formulations-modelling
#   sbatch --array=0-80%14 slurm/tune_families_array.sbatch
#   squeue -u pack423506           # watch tasks fan out
#   # results -> results/tune/<family>_<model>.json  (pull: scripts/mn5/sync_pull.sh --results)

#SBATCH --job-name=tune_fam_array
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-80%14
#SBATCH -o tune_fam_array_%A_%a.out
#SBATCH -e tune_fam_array_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
WHEELS="${WHEELS:-$PROJECT/var/mn5_bundle/wheels}"
VENV="${VENV:-$PROJECT/venv312}"
DATASETS="${DATASETS:-$PROJECT/var/model_datasets}"
RESULTS="${RESULTS:-$PROJECT/results/tune}"
N_TRIALS="${N_TRIALS:-50}"
mkdir -p "$RESULTS"

# Deterministic grid — array task index maps to exactly one (family, model). Keep these in sync with the
# --array size in the submit command: N tasks = (#FAMILIES × #MODELS).
FAMILIES=(melting_temperature glass_transition_temperature crystallinity water_contact_angle \
          crystallization_enthalpy elongation_at_break tensile_modulus tensile_strength thermal_degradation_onset)
MODELS=(random_forest extra_trees gbm lightgbm xgboost ridge elastic_net svr knn)

N_MODELS=${#MODELS[@]}
IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job (sbatch --array=...)}"
FAM="${FAMILIES[$((IDX / N_MODELS))]}"
M="${MODELS[$((IDX % N_MODELS))]}"
echo "Node $(hostname)  task ${IDX} -> ${FAM} / ${M}  (${N_TRIALS} trials)"

module load intel mkl python/3.12.1
PYVER="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.12" ] || { echo "ERROR: need Python 3.12, got $PYVER"; exit 1; }

# --- one-time venv build, flock-guarded so concurrent array tasks don't race to create it ---
(
  flock 9
  # GATE ON .ready, NOT ON THE BINARY. `bin/p2e-formulations` appears EARLY in the p2e wheel's install,
  # while `.ready` is touched only after everything is in place. Testing the binary means a task can see a
  # HALF-BUILT venv and conclude it is finished. That is what failed on 2026-08-13 (job 44552976, tasks
  # 4/5/6/18): flock does not serialise across nodes on GPFS, three nodes entered this block at once
  # (0 on as01r1b17, 2 on as01r5b17, 19 on as04r4b22), a second node saw the binary another node's pip had
  # just created, skipped the build, found no .p2e_wheel sha and ran --force-reinstall into the same
  # directory. Two pips, one venv: "OSError: No such file or directory: .../bin/p2e-formulations".
  if [ ! -f "$VENV/.ready" ]; then
    echo "[setup] building venv $VENV (first task only)"
    python -m venv "$VENV"
    "$VENV/bin/python" -m ensurepip --default-pip 2>/dev/null || true
    unset PYTHONHOME PYTHONPATH
    "$VENV/bin/pip" install --no-index --find-links "$WHEELS" \
      p2e-formulations-data-pipeline numpy scikit-learn pandas joblib lightgbm optuna pysqlite3-binary
    # xgboost's wheel declares nvidia-nccl-cu12 (x86_64-only) as a hard dep; the arm64-built bundle never
    # downloaded it (marker false on mac), so a normal offline resolve backtracks and FAILS. --no-deps
    # installs just the CPU xgboost wheel — numpy/scipy are already present and nccl is multi-GPU-only.
    "$VENV/bin/pip" install --no-index --no-deps --find-links "$WHEELS" xgboost
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
    touch "$VENV/.ready"
  fi
) 9>"$VENV.lock"
unset PYTHONHOME PYTHONPATH

# Fail loud + early if the (flock'd) venv build didn't produce a usable install, instead of dying with a
# cryptic error deep in the tune. A broken venv is rebuilt on the next submit (the flock gate re-checks).
# WAIT for a build in progress on ANOTHER NODE before declaring failure. Since flock is not cross-node on
# GPFS, a task can arrive while a peer is mid-build; without this it reads "not ready" as "broken" and dies
# in seconds. Bounded so a genuinely failed build still fails the task rather than holding the slot.
for _ in $(seq 1 120); do
  [ -f "$VENV/.ready" ] && break
  echo "[setup] venv not ready yet — another task is building it; waiting 5s"
  sleep 5
done

# Fail loud + early if the venv build did not produce a usable install.
[ -f "$VENV/.ready" ] || { echo "FATAL: venv never became ready after 10 min — check the building task's log, fix the bundle, rm -rf venv312, resubmit." >&2; exit 1; }
[ -x "$VENV/bin/p2e-formulations" ] || { echo "FATAL: venv build failed — p2e-formulations missing. Check the building task's log, fix the bundle, rm -rf venv312, resubmit." >&2; exit 1; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ ! -f "$DATASETS/dataset_${FAM}.csv" ]; then echo "[skip] no dataset_${FAM}.csv"; exit 0; fi
"$VENV/bin/p2e-formulations" train tune "$FAM" \
  --model "$M" --n-trials "$N_TRIALS" --datasets "$DATASETS" --root "$PROJECT" \
  | tee "$RESULTS/${FAM}_${M}.json"
echo "[tune] ${FAM}/${M} done"
