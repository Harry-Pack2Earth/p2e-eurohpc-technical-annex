#!/bin/bash
# slurm/mlip_embed_gpu.sbatch
#
# Phase-8 / 8a.1 — ACC single-GPU job: relax each component's repeat-unit geometry with a pretrained Orb
# MLIP and build registries/component_mlip_embeddings.json (the desc_mlip_* table). Runs
# scripts/build_geometries.py (RDKit, if geometries are missing) then scripts/build_mlip_embeddings.py
# inside container/p2e-mlip.sif.
#
# ~69 components = one sub-2h single-GPU batch (NOT the "thousands of GPU-hours" the roadmap implied). This
# is the CHEAP go/no-go input: once the table lands, `train mlip-ablation` decides whether the whole physics
# funnel is worth building. Idempotent: per-structure embedding shards live on the DURABLE project store, so
# a job killed at the wall/allocation limit resumes cleanly on re-submit (shard-skip).
#
# Submit via scripts/mn5/submit_gpu.sh mlip [--gpus 1] [--qos acc_debug|acc_ehpc] [--time HH:MM:SS]
#   [--model orb_v2] [--pca-dims 4] [--dry-run]. Orb weights must be staged to GPFS first (no node internet)
#   and pointed at via ORB_WEIGHTS.
#
#SBATCH --job-name=p2e-mlip-embed
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/p2e-formulations-modelling/logs/%x_%j.err

set -euo pipefail

# ==============================================================================
# Preflight
# ==============================================================================
echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "JobName:      ${SLURM_JOB_NAME:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "CPUsPerTask:  ${SLURM_CPUS_PER_TASK:-unset}"
echo "GPUs:         ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# CRITICAL (confirmed BSC gotcha): srun no longer inherits --cpus-per-task from #SBATCH.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# ==============================================================================
# Paths — literal BSC fallbacks so this still runs if launched without submit_gpu.sh's --export.
# ==============================================================================
ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-mlip.sif}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"

# Durable inputs/outputs on the PROJECT store (scratch is wiped; shards must survive a re-submit).
GEOM_DIR="${GEOM_DIR:-${PROJECT_ROOT}/var/mlip/geometries}"
SHARDS_DIR="${SHARDS_DIR:-${PROJECT_ROOT}/var/mlip/shards}"
DESCRIPTORS="${PROJECT_ROOT}/src/p2e_formulations_pipeline/modelling/registries/component_descriptors.json"
COMPONENTS="${PROJECT_ROOT}/src/p2e_formulations_pipeline/modelling/registries/components.json"

# Pretrained Orb weights staged to GPFS (NO node internet). Point the loader at them.
ORB_WEIGHTS="${ORB_WEIGHTS:-${PROJECT_ROOT}/var/mlip/orb_weights}"
ORB_MODEL="${ORB_MODEL:-orb_v2}"
PCA_DIMS="${PCA_DIMS:-4}"
# SMOKE: MLIP_LIMIT set (via submit_gpu.sh --limit N) relaxes only N structures on the cheap acc_debug queue,
# verifying weights-load + the version-sensitive embedding hook + GPU BEFORE the full acc_ehpc run. The script
# diverts the small table to a .smoke sidecar, so a smoke run can never clobber the committed embedding table.
LIMIT_ARG=()
[[ -n "${MLIP_LIMIT:-}" ]] && LIMIT_ARG=(--limit "${MLIP_LIMIT}")

echo "Container:    ${SIF}"
echo "Geometries:   ${GEOM_DIR}"
echo "Shards:       ${SHARDS_DIR}"
echo "Orb model:    ${ORB_MODEL}  weights: ${ORB_WEIGHTS}  pca-dims: ${PCA_DIMS}${MLIP_LIMIT:+  SMOKE-limit: ${MLIP_LIMIT}}"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}" "${GEOM_DIR}" "${SHARDS_DIR}"
cd "${JOB_SCRATCH}"

export XDG_CACHE_HOME="${JOB_SCRATCH}/.cache"
export ORB_MODELS_CACHE_DIR="${ORB_WEIGHTS}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${XDG_CACHE_HOME}"

OUT_TABLE="${JOB_SCRATCH}/component_mlip_embeddings.json"

BIND_ARGS=(
  --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}"
  --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}"
)

# Singularity is a MODULE on MN5, not on the default PATH. Without this load, `singularity exec` below fails
# instantly with "execve(): singularity: No such file or directory" (matches nn_gpu.sbatch / kinetic_gpu.sbatch).
module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

# NOTE: we run the scripts from the BIND-MOUNTED ${PROJECT_ROOT}/scripts (NOT the /opt/p2e/scripts copies
# baked into the .sif). The container supplies only the environment (python/rdkit/orb/torch); the code comes
# from the project store, so a `sync_push.sh --code` iterates the pipeline WITHOUT a 9.7 GB image rebuild.
#
# ==============================================================================
# 1) Build geometries (RDKit, CPU) if none present — capped repeat-unit XYZ + frozen protocol.
# ==============================================================================
if ! ls "${GEOM_DIR}/xyz/"*.xyz >/dev/null 2>&1; then
  echo "No geometries found -> building with RDKit (protocol frozen: capped monomer)."
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    singularity exec --nv "${BIND_ARGS[@]}" "${SIF}" \
      python "${PROJECT_ROOT}/scripts/build_geometries.py" \
        --descriptors "${DESCRIPTORS}" --out "${GEOM_DIR}"
fi

# ==============================================================================
# 2) Relax + embed on the GPU; PCA-reduce; write the desc_mlip_* table.
#    Shard-skip resume is handled inside the script (skips components already cached under --shards).
# ==============================================================================
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv "${BIND_ARGS[@]}" \
    --env ORB_MODELS_CACHE_DIR="${ORB_MODELS_CACHE_DIR}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    "${SIF}" \
    python "${PROJECT_ROOT}/scripts/build_mlip_embeddings.py" \
      --geometries "${GEOM_DIR}" \
      --components "${COMPONENTS}" \
      --shards "${SHARDS_DIR}" \
      --model "${ORB_MODEL}" \
      --pca-dims "${PCA_DIMS}" \
      --weights-dir "${ORB_WEIGHTS}" \
      "${LIMIT_ARG[@]}" \
      --out "${OUT_TABLE}"

# ==============================================================================
# Stage the table back to the durable results store. It is a FEATURE input: review it, then commit into
# src/.../registries/component_mlip_embeddings.json (the desc_mlip_* aggregator picks it up automatically).
# ==============================================================================
# In SMOKE mode build_mlip_embeddings diverts the table to a .smoke.json sidecar (never the committed name),
# so stage THAT file — otherwise the rsync of the canonical name fails (exit 23) even though the run succeeded.
ACTUAL_TABLE="${OUT_TABLE}"
[[ -n "${MLIP_LIMIT:-}" ]] && ACTUAL_TABLE="${OUT_TABLE%.json}.smoke.json"
mkdir -p "${RESULTS_DIR}/mlip/${SLURM_JOB_ID}"
rsync -a "${ACTUAL_TABLE}" "${RESULTS_DIR}/mlip/${SLURM_JOB_ID}/" || echo "WARN: table ${ACTUAL_TABLE} not found to stage"
rsync -a "${GEOM_DIR}/manifest.json" "${GEOM_DIR}/protocol.json" "${RESULTS_DIR}/mlip/${SLURM_JOB_ID}/" 2>/dev/null || true

echo "End:          $(date -u +%FT%TZ)"
echo "Table:        ${RESULTS_DIR}/mlip/${SLURM_JOB_ID}/$(basename "${ACTUAL_TABLE}")"
echo "Next:         pull it, review, and commit into src/.../registries/component_mlip_embeddings.json,"
echo "              then run: p2e-formulations train mlip-ablation --family melting_temperature ... (the 8a A/B gate)"
