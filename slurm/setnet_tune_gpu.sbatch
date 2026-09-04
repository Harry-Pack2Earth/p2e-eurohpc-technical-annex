#!/bin/bash
# slurm/setnet_tune_gpu.sbatch
#
# LEVER #12 — OPTUNA-TUNES the set model (`train tune-setnet --arm pool|attn`).
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
#SBATCH --job-name=p2e-setnet-tune
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
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
# THE FAMILY LIST IS A PUSHED FILE, NOT SHELL STATE. Three submissions have now silently fallen through
# to the hand-list below because a FAMILIES env var evaporated with its login shell (empty substitution
# 2026-08-17; unset $FAM_V37 on the top-ups AND the original wide-59 tunes 2026-08-18 — zero --family
# flags in every log of that chain). An empty env var is indistinguishable from "use the default", which
# makes the default a trap. Resolution order: explicit FAMILIES env (ad-hoc runs) > the committed
# config/setnet_families.txt (pushed by --code, versioned, survives any shell) > the legacy 22 hand-list,
# LOUDLY, as a last resort.
FAMILIES_FILE="${FAMILIES_FILE:-${PROJECT_ROOT}/config/setnet_families.txt}"
if [[ -z "${FAMILIES:-}" && -s "${FAMILIES_FILE}" ]]; then
  FAMILIES="$(tr -d '\n' < "${FAMILIES_FILE}")"
  FAMILY_SOURCE="file:${FAMILIES_FILE}"
else
  FAMILY_SOURCE="${FAMILIES:+env}"
fi
FAMILIES="${FAMILIES:-tensile_strength,elongation_at_break,tensile_modulus,melting_temperature,glass_transition_temperature,crystallinity,water_uptake,melting_enthalpy,water_contact_angle,thermal_degradation_temperature,light_transmittance,thermal_degradation_peak,disintegration,color_l_star,thermal_degradation_onset,color_b_star,color_a_star,crystallization_temperature,biodegradation,cold_crystallization_temperature,antimicrobial_inhibition_zone,residual_char_mass}"
FAMILY_SOURCE="${FAMILY_SOURCE:-LEGACY-DEFAULT-22 (no env, no file — almost certainly wrong for a wide run)}"
FAMILY_COUNT="$(( $(tr -cd ',' <<< "${FAMILIES}" | wc -c) + 1 ))"
echo "FamilySource: ${FAMILY_SOURCE}   (${FAMILY_COUNT} families)"
ARM="${ARM:-attn}"
N_TRIALS="${N_TRIALS:-40}"

BATCH="${BATCH:-256}"
AMP="${AMP:-1}"
# Feature-space parity with the trees for the DENSE half (the set half is #12's own). Accept either
# spelling: the singularity call sets P2E_BACKBONE_FULL_FEATURES from this var, so exporting that name at
# submit time would otherwise be silently overridden here.
FULL_FEATURES="${FULL_FEATURES:-${P2E_BACKBONE_FULL_FEATURES:-1}}"
# Seeds per trial: 1 here, 3 in the final run. Every trial pays the multiplier, so at N_TRIALS=40 seeds=3
# is a 3x GPU-hour bill for noise reduction that only the reported number actually needs.
SEEDS="${SEEDS:-1}"
MISSING_MASK="${MISSING_MASK:-1}"

echo "Arm:          ${ARM}"
echo "Families:     ${FAMILIES}"
echo "Optuna trials: ${N_TRIALS}"
echo "FullFeatures/Seeds/MissingMask: ${FULL_FEATURES} / ${SEEDS} / ${MISSING_MASK}"
echo "Container:    ${SIF}"

# DURABLE OPTUNA STUDY (ported from nn_tune_gpu.sbatch on 2026-08-18, one day after both wide-55 arms
# hit their 12h wall clock with IN-MEMORY studies and lost every trial — ~24 GPU-hours, and the per-trial
# scratch reports could not reconstruct them because three searched HPs are not echoed in a report).
# With a sqlite study, resubmitting the SAME command tops the study back up to N_TRIALS instead of
# restarting. The study name embeds the config tag and the arm: arms must never pool trials.
STUDY_TAG="$(basename "${P2E_MODELLING_CONFIG:-modelling.yaml}" .yaml)"
STUDY_NAME="${STUDY_NAME:-setnet_${STUDY_TAG}_${ARM}_f${FAMILY_COUNT}}"
STUDY_DB="${STUDY_DB:-${SCRATCH_ROOT}/optuna/${STUDY_NAME}.db}"
mkdir -p "$(dirname "${STUDY_DB}")"
echo "Study:        ${STUDY_NAME}"
echo "Study DB:     ${STUDY_DB}$([[ -f "${STUDY_DB}" ]] && echo '  (EXISTS — this run RESUMES it)')"

# PARALLEL WORKERS. Submit as an ARRAY to share one study across several GPUs:
#   FAMILIES=... sbatch --array=0-3 slurm/setnet_tune_gpu.sbatch    # 4 workers, ONE budget of N_TRIALS
# The budget belongs to the STUDY (the CLI stops a worker once the study reaches N_TRIALS), so raising
# the array size shortens the wall clock rather than multiplying the bill. Task 0 is the leader — it
# computes the baseline and the final evaluation and writes the canonical setnet_tune_<arm>.json.
WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
N_WORKERS="${N_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

# Workers never share a scratch dir; the leader's report path stays predictable for PARAMS_JSON.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${ARM}_${SLURM_ARRAY_JOB_ID}_w${WORKER_INDEX}"
  RUN_TAG="${SLURM_ARRAY_JOB_ID}/w${WORKER_INDEX}"
else
  JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${ARM}_${SLURM_JOB_ID}"
  RUN_TAG="${SLURM_JOB_ID}"
fi
mkdir -p "${JOB_SCRATCH}"
cd "${JOB_SCRATCH}"

export XDG_CACHE_HOME="${JOB_SCRATCH}/.cache"
export TRITON_CACHE_DIR="${JOB_SCRATCH}/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"

IFS=',' read -r -a FAMILY_ARR <<< "${FAMILIES}"
P2E_CMD=(p2e-formulations train tune-setnet --arm "${ARM}")
for f in "${FAMILY_ARR[@]}"; do
  [[ -z "${f}" ]] && continue
  P2E_CMD+=(--family "${f}")
done
# --root IS REQUIRED, not cosmetic. P2E_MODELLING_CONFIG takes a path resolved relative to --root, and
# the CLI fallback repo_root() only finds the repo from a SOURCE checkout — inside the container the
# package lives in site-packages, so it lands on a python lib dir with no config/ in it. Without this a
# `row10` setnet job would train on the DEFAULT config: no row flag and NO LEAK DENYLIST, with nothing in
# the output to say so. Same defect fixed for the NN jobs in 6dd7fd9.
P2E_CMD+=(--datasets "${DATASETS_DIR}" --runs "${JOB_SCRATCH}" --root "${PROJECT_ROOT}" \
          --n-trials "${N_TRIALS}" \
          --storage "${STUDY_DB}" --study-name "${STUDY_NAME}" \
          --worker-index "${WORKER_INDEX}" --n-workers "${N_WORKERS}")

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

# Under an array, stage per worker (setnet/<array_job_id>/wN) — same layout the backbone chain uses, so
# a final's PARAMS_JSON is results/setnet/<array_job_id>/w0/setnet_tune_<arm>.json, knowable at submit.
mkdir -p "${RESULTS_DIR}/setnet/${RUN_TAG}"
rsync -a --exclude '.cache/' "${JOB_SCRATCH}/" "${RESULTS_DIR}/setnet/${RUN_TAG}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Results:      ${RESULTS_DIR}/setnet/${RUN_TAG}"
