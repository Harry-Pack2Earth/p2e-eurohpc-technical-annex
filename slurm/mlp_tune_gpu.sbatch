#!/bin/bash -l
# slurm/mlp_tune_gpu.sbatch — ROW 10's re-run: tune `mlp` per family, on the GPU track.
#
# WHY THIS EXISTS AND WHY IT IS NOT A SLOT IN full_tune_array.sbatch.
#
# Row 10 (job 44578421) put `mlp` into the CPU tune array and every one of its 80 results came back EMPTY
# with `KeyError: "Unknown model 'mlp'"`. Two independent causes, both now understood:
#
#   1. `mlp` is registered lazily (models/base._OPTIONAL_MODULES) so that tabular code paths never import
#      torch. `runner.py` called `ensure_registered` before its bake-off; `tune_and_gate` did not. Fixed in
#      f5f47bd — the tune path now wakes it.
#   2. The offline CPU wheel bundle carries NO torch, so `mlp` cannot import on the CPU array whatever the
#      code does. Adding a ~200 MB torch wheel to a bundle that every one of 720 CPU array tasks installs
#      is the wrong trade for one model.
#
# So `mlp` runs where torch already lives: inside the Singularity image, exactly as `backbone` and `setnet`
# do. GPU is incidental — a small MLP over a few hundred tabular rows does not need one — but the SIF is
# the CUDA image and acc_ehpc is the allocation we have, so that is the cheapest correct home for it.
#
# WHAT IT MEASURES. One array task = one family, `train tune-gate <family> --model mlp`. The gate has no
# incumbent for mlp (no family_params entry), so it will adopt whatever it tunes; the verdict is therefore
# NOT the adopt/keep flag but the comparison of mlp's tuned pooled R2 against that family's current
# champion. Read it with scripts/row_verdict.py against the baseline, or straight off the scoreboard.
#
# RESULTS GO IN THEIR OWN DIRECTORY (modelling.mlp), not into the baseline's. The baseline tune predates
# the row-4/row-7 promotions, so mixing a freshly-measured mlp result into it would put two config states
# in one directory with nothing recording the difference.
#
# Prereqs (laptop), IN THIS ORDER:
#   container/build_local.sh --docker --force && scripts/mn5/sync_push.sh --container
#     — the SIF must contain f5f47bd, or every task fails exactly as row 10 did. `--code` does NOT reach
#       the GPU track.
#   scripts/mn5/sync_push.sh --code            # config + this sbatch
#
# Submit (alogin1). 80 families; throttle low — these are whole GPUs for a small model:
#   sbatch --array=0-79%4 slurm/mlp_tune_gpu.sbatch
#   # results -> results/full_tune/modelling.mlp/<family>_mlp.json

#SBATCH --job-name=p2e-mlp-tune
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --array=0-0%1
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%A_%a.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/ehpc838/pack423506/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-formulations.sif}"
DATASETS="${DATASETS:-${PROJECT_ROOT}/var/model_datasets}"
RESULTS="${RESULTS:-${PROJECT_ROOT}/results/full_tune/modelling.mlp}"
N_TRIALS="${N_TRIALS:-50}"

echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "ArrayTaskID:  ${SLURM_ARRAY_TASK_ID:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "Start:        $(date -u +%FT%TZ)"

# FAMILY DISCOVERY MUST MATCH full_sweep_array.sbatch / full_tune_array.sbatch EXACTLY — same glob, same
# sort, same exclusion. If it drifts, an array index means a different family here than it does there, and
# every cross-referenced index in the runbooks silently points at the wrong row.
EXCLUDE_FAMILIES=(film_geometry)
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

IDX="${SLURM_ARRAY_TASK_ID:?must run as an array job (sbatch --array=...)}"
if [ "$IDX" -ge "${#FAMILIES[@]}" ]; then
  echo "[skip] index $IDX >= ${#FAMILIES[@]} families"; exit 0
fi
FAM="${FAMILIES[$IDX]}"
echo "Families found: ${#FAMILIES[@]}   Task $IDX -> family=$FAM model=mlp ($N_TRIALS trials)"

if [ ! -f "$DATASETS/dataset_${FAM}.csv" ]; then echo "[skip] no dataset_${FAM}.csv"; exit 0; fi
mkdir -p "$RESULTS"

# Singularity is a MODULE on MN5, not on the default PATH (the observed nn_gpu fast-fail).
module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"
echo "SIF:          ${SIF}"

# NO P2E_MODELLING_CONFIG: this runs under the PRODUCTION config on purpose, so mlp is measured against the
# same conditions the served champions were. Setting it here is how a row picks up a stale config by
# accident (see scripts/mn5/submit_row.sh's header for what that already cost).
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" \
    --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}" \
    "${SIF}" \
    p2e-formulations train tune-gate "$FAM" \
      --model mlp --n-trials "$N_TRIALS" --datasets "$DATASETS" --root "$PROJECT_ROOT" \
  | tee "${RESULTS}/${FAM}_mlp.json"

echo "[mlp_tune] ${FAM}/mlp done -> ${RESULTS}/${FAM}_mlp.json"
echo "End:          $(date -u +%FT%TZ)"
