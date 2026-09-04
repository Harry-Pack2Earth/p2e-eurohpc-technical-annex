#!/bin/bash -l
# slurm/full_sweep_array.sbatch  (workstream #11 — one-command MN5 FULL-SWEEP pre-stage)
#
# PARALLEL full-zoo bake-off across EVERY encoded property family. One SLURM ARRAY TASK = one family, via
# the real `train bake-off --family <F> --datasets DIR --runs DIR` (see src/p2e_formulations_pipeline/cli.py
# train_bake_off). Omitting --model deliberately — that lets bake-off sweep the WHOLE config
# `train.default_models` zoo for this family, which is the entire point of a bake-off (see
# config/modelling.yaml). This is the "retrain EVERYTHING" step: run it the moment a fresh `modelling encode`
# has landed a bigger var/model_datasets/ (e.g. after the ~700-paper corpus expansion is ingested).
#
# QUEUE: ehpc838 is an ACC-only allocation (confirmed — no GPP grant) -> -q acc_ehpc, NOT gp_ehpc (that
# fails "Invalid qos specification"). No --gres requested: this is tree/linear/kernel algorithms on
# ~10-750 row tables — CPU-bound, a GPU would sit idle or add transfer overhead for no gain (same reasoning
# as slurm/tune_families_array.sbatch).
#
# Uses the PROVEN venv+wheel deployment (same $PROJECT/venv312 as tune_families(_array).sbatch /
# bake_off_real.sbatch — shared on purpose so a full_tune_array.sbatch submitted in the same session reuses
# the already-built venv instead of rebuilding it). A flock guards the one-time build so concurrent array
# tasks (from this script AND full_tune_array.sbatch) never race to create it.
#
# FAMILY DISCOVERY is DYNAMIC (not a hardcoded list) — every array task independently globs
# $DATASETS/dataset_*.csv, sorts, and drops EXCLUDE_FAMILIES, so this script automatically covers whatever
# families the latest `modelling encode` produced without editing the script. This block MUST stay
# logically identical to scripts/mn5/full_sweep_plan.sh (used locally to plan --array sizing) and to
# slurm/full_tune_array.sbatch's own copy — if you change one, change all three.
#
# Prereqs (laptop): bash scripts/mn5/build_wheels.sh && scripts/mn5/sync_push_dataset.sh
#   (or directly: scripts/mn5/sync_push.sh --code --data --bundle)
# Plan the array size BEFORE submitting (laptop, reads your local var/model_datasets — same glob the
# cluster side uses, so the count matches as long as you pushed what you planned against):
#   scripts/mn5/full_sweep_plan.sh --mode sweep
# Submit (alogin1 — ACC login node; ehpc838 has no GPP queue):
#   ssh pack423506@alogin1.bsc.es
#   cd /gpfs/projects/ehpc838/pack423506/p2e-formulations-modelling   # (or `source bsc_project ehpc838` layout)
#   sbatch --array=0-<N-1>%<K> slurm/full_sweep_array.sbatch          # N from full_sweep_plan.sh above
#   squeue -u pack423506
#   # results -> results/full_sweep/<family>/leaderboard.json + model cards
#   # pull: scripts/mn5/sync_pull_results.sh

#SBATCH --job-name=full_sweep_array
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --array=0-0%1
#SBATCH -o full_sweep_array_%A_%a.out
#SBATCH -e full_sweep_array_%A_%a.err

set -euo pipefail

echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "ArrayJobID:   ${SLURM_ARRAY_JOB_ID:-unset}"
echo "ArrayTaskID:  ${SLURM_ARRAY_TASK_ID:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "Start:        $(date -u +%FT%TZ)"

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
WHEELS="${WHEELS:-$PROJECT/var/mn5_bundle/wheels}"
VENV="${VENV:-$PROJECT/venv312}"
DATASETS="${DATASETS:-$PROJECT/var/model_datasets}"
# RESULTS ARE NAMESPACED BY ROW — see the matching block in full_tune_array.sbatch. Flat paths meant a
# second row in flight overwrote the first's leaderboards file-for-file, silently, leaving a directory
# that looks like one complete experiment but holds a mixture of two.
ROW_TAG="$(basename "${P2E_MODELLING_CONFIG:-modelling.yaml}" .yaml)"
RESULTS_DEFAULT="$PROJECT/results/full_sweep/$ROW_TAG"
RESULTS="${RESULTS:-$RESULTS_DEFAULT}"
mkdir -p "$RESULTS"
echo "Row tag:      $ROW_TAG"
echo "Results:      $RESULTS"
# See full_tune_array.sbatch: an inherited RESULTS silently defeats row namespacing, which already cost
# one row's attributability. Warn rather than forbid.
if [ "$RESULTS" != "$RESULTS_DEFAULT" ]; then
  echo "WARNING: RESULTS was overridden from the environment (default: $RESULTS_DEFAULT)"
  case "$RESULTS" in
    *"$ROW_TAG"*) : ;;
    *) echo "         !! it does NOT contain the row tag '$ROW_TAG' -- rows can collide here." ;;
  esac
fi

# ==============================================================================
# Family discovery — MUST match scripts/mn5/full_sweep_plan.sh and slurm/full_tune_array.sbatch verbatim.
# ==============================================================================
EXCLUDE_FAMILIES=(film_geometry)   # keep in sync with config/modelling.yaml train.exclude_families

FAMILIES=()
while IFS= read -r csv; do
  name="$(basename "$csv" .csv)"
  name="${name#dataset_}"
  skip=0
  for ex in "${EXCLUDE_FAMILIES[@]}"; do
    [ "$name" = "$ex" ] && { skip=1; break; }
  done
  [ "$skip" -eq 1 ] && continue
  FAMILIES+=("$name")
done < <(find "$DATASETS" -maxdepth 1 -name 'dataset_*.csv' | sort)

N_FAM=${#FAMILIES[@]}
IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job (sbatch --array=...)}"
[ "$IDX" -lt "$N_FAM" ] || { echo "FATAL: array task $IDX >= discovered family count $N_FAM — re-run scripts/mn5/full_sweep_plan.sh --mode sweep and resubmit with the right --array size." >&2; exit 1; }
FAM="${FAMILIES[$IDX]}"
echo "Families found: $N_FAM"
echo "Task $IDX -> family=$FAM"

# --- python 3.12 module chain (hierarchical: intel -> mkl -> python/3.12.1) ---
module load intel mkl python/3.12.1
PYVER="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.12" ] || { echo "ERROR: need Python 3.12, got $PYVER"; exit 1; }

# --- one-time venv build, flock-guarded so concurrent array tasks (incl. full_tune_array.sbatch tasks
#     sharing this same $VENV) don't race to create it. Mirrors tune_families_array.sbatch exactly. ---
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
    # xgboost's wheel declares nvidia-nccl-cu12 (x86_64-only) as a hard dep; an arm64-built bundle never
    # downloaded it (marker false on mac) -> a normal offline resolve backtracks and FAILS. --no-deps
    # installs just the CPU xgboost wheel (numpy/scipy already present; nccl is multi-GPU-only).
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
  # ALWAYS refresh the p2e package when the pushed wheel changed — even on a pre-existing venv. Without this,
  # the build `if` above skips reinstall and every task silently runs STALE code after a code change (e.g. an
  # old ModellingConfig rejecting a new config field -> all tasks die at config-load). Guarded by wheel mtime
  # so it reinstalls once per new push, not on every array task.
  # Compare the wheel's CONTENT HASH, not its path — the version is pinned at 0.1.0 so a rebuild keeps
  # the same filename, and path-equality skipped the reinstall forever after the first push. See the
  # matching block in full_tune_array.sbatch; both arrays share this venv, so they must agree.
  CUR_WHEEL="$(ls -t "$WHEELS"/p2e_formulations*.whl 2>/dev/null | head -1)"
  CUR_SUM=""
  [ -n "$CUR_WHEEL" ] && CUR_SUM="$(sha256sum "$CUR_WHEEL" | cut -d' ' -f1)"
  if [ -n "$CUR_SUM" ] && [ "$CUR_SUM" != "$(cat "$VENV/.p2e_wheel" 2>/dev/null)" ]; then
    echo "[setup] refreshing p2e package from $CUR_WHEEL (sha256 ${CUR_SUM:0:12})"
    "$VENV/bin/pip" install --no-index --no-deps --force-reinstall --find-links "$WHEELS" p2e-formulations-data-pipeline
    # ORDERING IS LOAD-BEARING WITHIN A NODE. The sha is written LAST, after the binary is back, so a task
    # sharing this filesystem *and this lock* can only skip the block once the binary exists. Write this
    # line before the reinstall and same-node tasks lose that protection.
    #
    # RETRACTION, and read this before trusting the measurement. This comment originally claimed the
    # ordering, not the flock, was what prevented the 2026-08-13 exit-127 failures, on the strength of a
    # 12-way concurrent harness that could not reproduce them with the flock removed. The harness ran on
    # ONE machine. The real cause — see the .ready gate above — needed THREE NODES, because flock does not
    # serialise across nodes on GPFS. A single-node harness structurally cannot produce a cross-node race,
    # so its clean result meant nothing and the conclusion drawn from it was wrong.
    #
    # The instrument was even checked and did discriminate (remove the lock, the refresh runs 12× instead
    # of once) — but it discriminated on lock-present/absent while the variable that mattered was
    # lock-honoured/ignored, which it never varied and could not. Checking that a harness responds to SOME
    # change is not checking it responds to the change you are actually asking about.
    echo "$CUR_SUM" > "$VENV/.p2e_wheel"
  fi
) 9>"$VENV.lock"
unset PYTHONHOME PYTHONPATH

# Fail loud + early if the (flock'd) venv build didn't produce a usable install.
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

# --- per-task MLflow store, so concurrent tasks never race on one GPFS file store. Consolidated after
#     pull with `p2e-formulations train tracking-merge` (scripts/mn5/sync_pull_results.sh does this). ---
TASK_TAG="full_sweep_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${IDX}"
MLRUNS_TASK_DIR="$PROJECT/results/mlruns_tasks/$TASK_TAG"
mkdir -p "$MLRUNS_TASK_DIR"
export MLFLOW_TRACKING_URI="file://$MLRUNS_TASK_DIR"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ ! -f "$DATASETS/dataset_${FAM}.csv" ]; then
  echo "[skip] no dataset_${FAM}.csv (should not happen — discovered from the same glob)"; exit 0
fi

mkdir -p "$RESULTS/$FAM"
"$VENV/bin/p2e-formulations" train bake-off \
  --family "$FAM" \
  --datasets "$DATASETS" \
  --runs "$RESULTS/$FAM" \
  --root "$PROJECT"

echo "[full_sweep] $FAM done -> $RESULTS/$FAM/leaderboard.json"
echo "End: $(date -u +%FT%TZ)"
