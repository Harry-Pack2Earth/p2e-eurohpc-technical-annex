#!/bin/bash -l
# slurm/full_tune_array.sbatch  (workstream #11 — one-command MN5 FULL-SWEEP pre-stage)
#
# PARALLEL Optuna tuning sweep across EVERY encoded family x the full model zoo. One SLURM ARRAY TASK = one
# (family, model) pair, via `train tune <FAMILY> --model <M> --n-trials <N> --datasets DIR` (FAMILY
# positional; see src/p2e_formulations_pipeline/cli.py train_tune_cmd). This is the dynamic-family sibling
# of slurm/tune_families_array.sbatch, which hardcodes a 9-family "winnable set" — this one covers whatever
# `modelling encode` produced, so it needs no editing when the corpus expansion lands new families.
#
# NOTE (workstream #2, built in parallel): once a `train tune-gate` command lands (verify-or-keep gate that
# only promotes a tuned HP set into config/modelling.yaml if it beats the current champion on the untouched
# outer fold), swap the CMD block below from `train tune` to `train tune-gate` — same positional/flags
# expected, so the swap should be a one-line change. Until then this runs plain `train tune`, which only
# emits a JSON report; promoting a result into family_params is still a manual step (see
# config/modelling.yaml's family_params comments for the current convention).
#
# QUEUE: ehpc838 is an ACC-only allocation -> -q acc_ehpc, NOT gp_ehpc ("Invalid qos specification"). No
# --gres requested — same CPU-bound reasoning as tune_families_array.sbatch (tree/linear/kernel algorithms
# on small tables; a GPU adds no speedup here, see that script's WHY-CPU note).
#
# Uses the PROVEN venv+wheel deployment at the SAME $PROJECT/venv312 as full_sweep_array.sbatch /
# tune_families(_array).sbatch / bake_off_real.sbatch — shared on purpose: submit both full_sweep_array and
# full_tune_array in the same session and the second one reuses whichever built the venv first. A flock
# guards the one-time build so concurrent array tasks never race to create it.
#
# FAMILY DISCOVERY is DYNAMIC — every array task independently globs $DATASETS/dataset_*.csv, sorts, and
# drops EXCLUDE_FAMILIES; the MODEL list is a fixed mirror of config/modelling.yaml's train.default_models
# minus global_mean (no HPs to tune). This block MUST stay logically identical to
# scripts/mn5/full_sweep_plan.sh (--mode tune, used locally to plan --array sizing) and to
# slurm/full_sweep_array.sbatch's family-discovery copy — if you change one, change all three.
#
# Prereqs (laptop): bash scripts/mn5/build_wheels.sh && scripts/mn5/sync_push_dataset.sh
#   (or directly: scripts/mn5/sync_push.sh --code --data --bundle)
# Plan the array size BEFORE submitting (laptop):
#   scripts/mn5/full_sweep_plan.sh --mode tune
# Submit (alogin1 — ACC login node; ehpc838 has no GPP queue):
#   ssh pack423506@alogin1.bsc.es
#   cd /gpfs/projects/ehpc838/pack423506/p2e-formulations-modelling
#   sbatch --array=0-<N-1>%<K> slurm/full_tune_array.sbatch          # N from full_sweep_plan.sh above
#   squeue -u pack423506
#   # results -> results/full_tune/<family>_<model>.json
#   # pull: scripts/mn5/sync_pull_results.sh

#SBATCH --job-name=full_tune_array
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-0%1
#SBATCH -o full_tune_array_%A_%a.out
#SBATCH -e full_tune_array_%A_%a.err

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
# RESULTS ARE NAMESPACED BY ROW. This path used to be flat, so every queued row wrote
# results/full_tune/<family>_<model>.json over the previous row's file of the same name. Two rows in
# flight at once therefore destroyed each other's output silently — no error, no warning, just a
# directory holding a mixture of two experiments that looks exactly like one complete experiment.
# The tag is derived the same way the Optuna study names are, so rows can never collide anywhere.
ROW_TAG="$(basename "${P2E_MODELLING_CONFIG:-modelling.yaml}" .yaml)"
RESULTS_DEFAULT="$PROJECT/results/full_tune/$ROW_TAG"
RESULTS="${RESULTS:-$RESULTS_DEFAULT}"
N_TRIALS="${N_TRIALS:-50}"
mkdir -p "$RESULTS"
echo "Row tag:      $ROW_TAG"
echo "Results:      $RESULTS"
# An INHERITED RESULTS silently defeats the row namespacing. This is not hypothetical: array 44418191 was
# submitted with P2E_MODELLING_CONFIG=modelling.row3 but RESULTS was still exported in the submitting
# shell from an earlier row-8 command, so row 3's tune wrote into results/tune_row8/ alongside another
# row's output, and the two became unseparable. Keep the override (it is genuinely useful) but make it
# impossible to do by accident: if RESULTS does not mention the row tag, say so loudly.
if [ "$RESULTS" != "$RESULTS_DEFAULT" ]; then
  echo "WARNING: RESULTS was overridden from the environment."
  echo "         default would have been: $RESULTS_DEFAULT"
  case "$RESULTS" in
    *"$ROW_TAG"*) echo "         (it does contain the row tag '$ROW_TAG', so rows stay separable)" ;;
    *) echo "         !! it does NOT contain the row tag '$ROW_TAG' -- results from different rows can" ;
       echo "         !! collide here. Unset RESULTS unless you meant this." ;;
  esac
fi

# ==============================================================================
# Family discovery — MUST match scripts/mn5/full_sweep_plan.sh and slurm/full_sweep_array.sbatch verbatim.
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

# Full zoo minus global_mean (no HPs to tune) — mirrors config/modelling.yaml train.default_models minus
# global_mean, same convention as tune_families_array.sbatch.
# Overridable via P2E_TUNE_MODELS (space-separated). The DEFAULT is byte-identical to the historical
# list on purpose: this sbatch is re-read by every array task at task start, and N_MODELS drives the
# index math (FAM = FAMILIES[IDX / N_MODELS]). Changing the default while an array is queued would
# silently REMAP every unstarted task to a different (family, model) pair — measured: with 9 models
# task 701 is (fam77, model8); with 10 it becomes (fam70, model1). So growing the list is opt-in per
# submission, never a default edit.
MODELS=(${P2E_TUNE_MODELS:-random_forest extra_trees gbm lightgbm xgboost ridge elastic_net svr knn})

N_FAM=${#FAMILIES[@]}
N_MODELS=${#MODELS[@]}
N_TOTAL=$((N_FAM * N_MODELS))
IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job (sbatch --array=...)}"
[ "$IDX" -lt "$N_TOTAL" ] || { echo "FATAL: array task $IDX >= discovered family*model count $N_TOTAL — re-run scripts/mn5/full_sweep_plan.sh --mode tune and resubmit with the right --array size." >&2; exit 1; }
FAM="${FAMILIES[$((IDX / N_MODELS))]}"
M="${MODELS[$((IDX % N_MODELS))]}"
echo "Families found: $N_FAM   Models: $N_MODELS   Total tasks: $N_TOTAL"
echo "Task $IDX -> family=$FAM model=$M ($N_TRIALS trials)"

# --- python 3.12 module chain (hierarchical: intel -> mkl -> python/3.12.1) ---
module load intel mkl python/3.12.1
PYVER="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[ "$PYVER" = "3.12" ] || { echo "ERROR: need Python 3.12, got $PYVER"; exit 1; }

# --- one-time venv build, flock-guarded so concurrent array tasks (incl. full_sweep_array.sbatch tasks
#     sharing this same $VENV) don't race to create it. Identical to full_sweep_array.sbatch. ---
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
  # Compare the wheel's CONTENT HASH, not its path. Our version is pinned at 0.1.0, so every rebuild
  # produces the byte-identical FILENAME p2e_formulations_data_pipeline-0.1.0-py3-none-any.whl — the old
  # path-equality test therefore matched on the first push and skipped the reinstall forever after,
  # which is precisely the stale-code failure the guard above was written to prevent. The stored value
  # was previously a path; a path never equals a sha256, so the first run after this change reinstalls
  # once and then settles.
  CUR_WHEEL="$(ls -t "$WHEELS"/p2e_formulations*.whl 2>/dev/null | head -1)"
  CUR_SUM=""
  [ -n "$CUR_WHEEL" ] && CUR_SUM="$(sha256sum "$CUR_WHEEL" | cut -d' ' -f1)"
  if [ -n "$CUR_SUM" ] && [ "$CUR_SUM" != "$(cat "$VENV/.p2e_wheel" 2>/dev/null)" ]; then
    echo "[setup] refreshing p2e package from $CUR_WHEEL (sha256 ${CUR_SUM:0:12})"
    "$VENV/bin/pip" install --no-index --no-deps --force-reinstall --find-links "$WHEELS" p2e-formulations-data-pipeline
    # Sha written LAST, after the binary is back — see the measured note in full_sweep_array.sbatch. That
    # ordering, not the flock, is what stops a task that skips this block from reaching the `-x` check
    # while another task has the package uninstalled. Both arrays share $VENV, so both must keep it.
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
if [ ! -f "$DATASETS/dataset_${FAM}.csv" ]; then echo "[skip] no dataset_${FAM}.csv"; exit 0; fi

"$VENV/bin/p2e-formulations" train tune-gate "$FAM" \
  --model "$M" --n-trials "$N_TRIALS" --datasets "$DATASETS" --root "$PROJECT" \
  | tee "$RESULTS/${FAM}_${M}.json"
echo "[full_tune] ${FAM}/${M} done -> $RESULTS/${FAM}_${M}.json"
echo "End: $(date -u +%FT%TZ)"
