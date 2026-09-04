#!/bin/bash
# slurm/setnet_gpu.sbatch
#
# LEVER #12 — FINAL RUN of the set model (`train setnet --arm pool|attn`). Tune first with
# slurm/setnet_tune_gpu.sbatch, then point PARAMS_JSON at that job's setnet_tune_<arm>.json.
#
# ONE multi-task model, not a family x model grid, so it gets its own GPU job and is NEVER a slot in the
# CPU tune array (slurm/full_tune_array.sbatch). That is what makes lever #12 additive at the
# infrastructure level: MODELS/N_MODELS there are untouched, so adding this cannot re-index or perturb a
# queued or running array.
#
# RUN BOTH ARMS, as two separate jobs:
#   ARM=pool  scripts/mn5/submit_gpu.sh setnet-tune     # Deep Sets — the control
#   ARM=attn  scripts/mn5/submit_gpu.sh setnet-tune     # Set transformer
# `pool` is not a warm-up for `attn`. Both replace the encoder's role-bucket averaging with per-component
# tokens; only `attn` additionally lets components interact. Running attn alone cannot say which half paid,
# and the corpus argues the split matters — median 2 components per formulation, p90 4, so a typical
# formulation is 1-6 pairs, which is not where attention usually earns its keep.
#
# Each arm gets its OWN study. Sharing one would confound the ablation: a gap between arms could be the
# architecture or could be that one arm drew luckier hyperparameters.
#
#SBATCH --job-name=p2e-setnet
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=06:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.err

set -euo pipefail

echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "JobName:      ${SLURM_JOB_NAME:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "GPUs:         ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# CRITICAL (confirmed BSC gotcha): srun no longer inherits --cpus-per-task from #SBATCH.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-formulations.sif}"
DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/var/model_datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"

# Same 22-family multi-task set as the backbone, deliberately: #12 must be comparable to the model it is
# being measured against, and the set model needs multi-task pooling far more than the backbone does
# (the component encoder is shared across every family, and no single family has enough PAPERS —
# tensile_strength, the largest, has 637).
# Family list from the PUSHED FILE by default — env evaporates with its shell, and that is exactly how
# the 2026-08-18 wide-59 finals silently trained the legacy 22 (see setnet_tune_gpu.sbatch for the log
# evidence). The FINAL must use the same set its tune searched, so both read the same committed file.
FAMILIES_FILE="${FAMILIES_FILE:-${PROJECT_ROOT}/config/setnet_families.txt}"
if [[ -z "${FAMILIES:-}" && -s "${FAMILIES_FILE}" ]]; then
  FAMILIES="$(tr -d '\n' < "${FAMILIES_FILE}")"
  FAMILY_SOURCE="file:${FAMILIES_FILE}"
else
  FAMILY_SOURCE="${FAMILIES:+env}"
fi
FAMILIES="${FAMILIES:-tensile_strength,elongation_at_break,tensile_modulus,melting_temperature,glass_transition_temperature,crystallinity,water_uptake,melting_enthalpy,water_contact_angle,thermal_degradation_temperature,light_transmittance,thermal_degradation_peak,disintegration,color_l_star,thermal_degradation_onset,color_b_star,color_a_star,crystallization_temperature,biodegradation,cold_crystallization_temperature,antimicrobial_inhibition_zone,residual_char_mass}"
FAMILY_SOURCE="${FAMILY_SOURCE:-LEGACY-DEFAULT-22 (no env, no file)}"
echo "FamilySource: ${FAMILY_SOURCE}   ($(( $(tr -cd ',' <<< "${FAMILIES}" | wc -c) + 1 )) families)"
ARM="${ARM:-attn}"
EPOCHS="${EPOCHS:-400}"   # overridden by the tuned params when PARAMS_JSON is set

BATCH="${BATCH:-256}"
AMP="${AMP:-1}"
# Feature-space parity with the trees for the DENSE half (the set half is #12's own). Accept either
# spelling: the singularity call sets P2E_BACKBONE_FULL_FEATURES from this var, so exporting that name at
# submit time would otherwise be silently overridden here.
FULL_FEATURES="${FULL_FEATURES:-${P2E_BACKBONE_FULL_FEATURES:-1}}"
# 3 seeds HERE (1 per tune trial). This is the number that gets reported, and a net's fold-to-fold seed
# variance is far larger than a tree's, so a single draw compared against a tuned tree pays a variance
# penalty on top of every other difference.
SEEDS="${SEEDS:-3}"
MISSING_MASK="${MISSING_MASK:-1}"

echo "Arm:          ${ARM}"
echo "Families:     ${FAMILIES}"
echo "Epochs:       ${EPOCHS}"
echo "FullFeatures/Seeds/MissingMask: ${FULL_FEATURES} / ${SEEDS} / ${MISSING_MASK}"
echo "Container:    ${SIF}"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${ARM}_${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}"
cd "${JOB_SCRATCH}"

export XDG_CACHE_HOME="${JOB_SCRATCH}/.cache"
export TRITON_CACHE_DIR="${JOB_SCRATCH}/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"

IFS=',' read -r -a FAMILY_ARR <<< "${FAMILIES}"
P2E_CMD=(p2e-formulations train setnet --arm "${ARM}")
for f in "${FAMILY_ARR[@]}"; do
  [[ -z "${f}" ]] && continue
  P2E_CMD+=(--family "${f}")
done
# --root IS REQUIRED, not cosmetic — see the note in setnet_tune_gpu.sbatch. Without it a row config is
# silently ignored and the run trains with no leak denylist while reporting normal numbers.
P2E_CMD+=(--datasets "${DATASETS_DIR}" --runs "${JOB_SCRATCH}" --root "${PROJECT_ROOT}" \
          --epochs "${EPOCHS}")
# TUNED PARAMS from a prior setnet_tune_gpu job. Without it this trains at library defaults, which is the
# UNTUNED arm — and an untuned result settles nothing in either direction (docs/ab-instrument-tuning-bias).
if [[ -n "${PARAMS_JSON:-}" ]]; then
  P2E_CMD+=(--params "${PARAMS_JSON}")
  echo "TunedParams:  ${PARAMS_JSON}"
fi

# THE TWO OUTPUTS THAT MAKE LEVER #12'S ACTUAL CASE, and which this sbatch did not request until
# 2026-08-14. Both arms ran on 22 families that day and produced NEITHER, so the only evidence available
# was a win/lose count against the trees — and models/nn/setnet.py's own docstrings say that number is
# insufficient:
#
#   --collect-oof        "a pooled R2 says whether #12 wins, but not whether it is REDUNDANT. Two models
#                        can score the same and be worth combining, or score the same and be the same
#                        model wearing different clothes — only the row-level residuals distinguish
#                        those." Feeds the complementarity test and the stacking DAG.
#   --export-embeddings  "the most ADDITIVE path in lever #12: it transfers the representation without
#                        the predictor, so an existing champion can be improved rather than replaced."
#                        Out-of-fold by construction, so trees can consume z_comp as features without
#                        carrying held-out papers into their own evaluation.
#
# ON by default: they are cheap (both are read off loops that already run) and they are the point. Set
# COLLECT_OOF=0 / EXPORT_EMBEDDINGS=0 only to reproduce an older run exactly.
[[ "${COLLECT_OOF:-1}" == "1" ]] && P2E_CMD+=(--collect-oof)
[[ "${EXPORT_EMBEDDINGS:-1}" == "1" ]] && P2E_CMD+=(--export-embeddings)
# THE SERVING ARTIFACT, on by default since 2026-08-20: the finals are the runs whose winners serving
# needs (residual_char_mass +0.112 among them), and an artifact costs one extra full-data fit per seed.
# SAVE_ARTIFACT=0 to skip.
[[ "${SAVE_ARTIFACT:-1}" == "1" ]] && P2E_CMD+=(--artifact-dir "${JOB_SCRATCH}/artifact_${ARM}")

echo "Command:      ${P2E_CMD[*]}"

module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" \
    --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    --env P2E_SETNET_BATCH="${BATCH}" \
    --env P2E_SETNET_AMP="${AMP}" \
    --env P2E_SETNET_SEEDS="${SEEDS}" \
    --env P2E_BACKBONE_FULL_FEATURES="${FULL_FEATURES}" \
    --env P2E_BACKBONE_MISSING_MASK="${MISSING_MASK}" \
    "${SIF}" \
    "${P2E_CMD[@]}"

mkdir -p "${RESULTS_DIR}/setnet/${SLURM_JOB_ID}"
rsync -a --exclude '.cache/' "${JOB_SCRATCH}/" "${RESULTS_DIR}/setnet/${SLURM_JOB_ID}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Results:      ${RESULTS_DIR}/setnet/${SLURM_JOB_ID}"
