#!/bin/bash
# slurm/ie_vllm_predict.sbatch
# =============================================================================
# W1/W6 — run the MERGED v2 extraction model over the 60-paper / 252-example GOLD
# set with vLLM (tensor-parallel + paged attention). The proper-engine counterpart
# to slurm/ie_predict.sbatch (HF device_map): minutes not hours, no per-window OOM.
#
# PREREQ (one-time), all path-isolated to the ie_extract namespace:
#   1. Merge the adapter into the base (in ie-train.sif, no GPU needed):
#        singularity exec --bind <binds> container/ie-train.sif \
#          python scripts/ie/merge_adapter.py \
#            --base /gpfs/projects/ehpc838/models/Qwen3.5-27B \
#            --adapter results/ie_sft/44769991 \
#            --out results/ie_sft/44769991_merged
#   2. Build + push the vLLM SIF:
#        bash container/build_local.sh --docker --def container/ie-vllm.def --out container/ie-vllm.sif
#        scripts/mn5/sync_push.sh --container
#
# Usage: sbatch slurm/ie_vllm_predict.sbatch      (IE_JOBID=<jobid> to pick the merged run)
#
# TODO(mn5-user): the #SBATCH --chdir/-o/-e lines can't expand vars; MN5 user hard-coded
# to pack423506 (as in ie_predict.sbatch). Change it in the four #SBATCH lines if yours differs.
# =============================================================================
#SBATCH --job-name=ie-vllm
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/ie_extract
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/ie_extract/logs/%x_%j.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/ie_extract/logs/%x_%j.err

set -euo pipefail

echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "GPUs:         ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

ACCOUNT="ehpc838"
MN5_USER="${USER:-pack423506}"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${MN5_USER}/p2e-formulations-modelling}"
ISCRATCH="${ISCRATCH:-/gpfs/scratch/${ACCOUNT}/${MN5_USER}/ie_extract}"
SIF="${SIF:-${PROJECT_ROOT}/container/ie-vllm.sif}"
IE_JOBID="${IE_JOBID:-44769991}"
MERGED="${MERGED:-${PROJECT_ROOT}/results/ie_sft/${IE_JOBID}_merged}"
# v3 uses var/ie_extract_v3/gold_eval.jsonl (uncapped). Override GOLD for v2's old set.
GOLD="${GOLD:-${PROJECT_ROOT}/var/ie_extract_v3/gold_eval.jsonl}"
PREDICT_PY="${PROJECT_ROOT}/scripts/ie/predict_vllm.py"
IE_TP="${IE_TP:-2}"
# v3 targets reach ~5517 tok and prompts ~8615 tok — sized so inference NEVER truncates (the bug v3
# fixed in training). Do not lower these below the dataset's target/prompt max.
IE_MAX_NEW="${IE_MAX_NEW:-6144}"
IE_MAX_MODEL_LEN="${IE_MAX_MODEL_LEN:-16384}"
OUT="${ISCRATCH}/predictions/pred_v3_vllm_${IE_JOBID}.jsonl"

echo "MergedModel:  ${MERGED}"
echo "Gold:         ${GOLD}"
echo "Out:          ${OUT}"
echo "TP / max_new: ${IE_TP} / ${IE_MAX_NEW}"

# Fail CLOSED — a missing merged model / SIF scores garbage or dies late.
[[ -f "${SIF}" ]]                            || { echo "vLLM SIF not found: ${SIF} (build container/ie-vllm.def, sync_push --container)" >&2; exit 1; }
[[ -f "${MERGED}/config.json" ]]             || { echo "Merged model not found: ${MERGED} (run scripts/ie/merge_adapter.py first)" >&2; exit 1; }
[[ -f "${GOLD}" ]]                           || { echo "Gold set not found: ${GOLD}" >&2; exit 1; }
[[ -f "${PREDICT_PY}" ]]                     || { echo "predict_vllm.py not found: ${PREDICT_PY}" >&2; exit 1; }

mkdir -p "${ISCRATCH}/predictions" "${ISCRATCH}/logs" "${ISCRATCH}/.cache/huggingface/hub"
export XDG_CACHE_HOME="${ISCRATCH}/.cache"
export HF_HOME="${ISCRATCH}/.cache/huggingface"
export HF_HUB_CACHE="${ISCRATCH}/.cache/huggingface/hub"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
# vLLM's own worker parallelism; keep OMP modest to avoid oversubscription on the node.
export OMP_NUM_THREADS=4

module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

BINDS="${PROJECT_ROOT}:${PROJECT_ROOT},${ISCRATCH}:${ISCRATCH}"

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${BINDS}" \
    --pwd "${PROJECT_ROOT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" --env HF_HOME="${HF_HOME}" --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env VLLM_DO_NOT_TRACK=1 \
    --env TOKENIZERS_PARALLELISM=false --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    "${SIF}" \
    python3 "${PREDICT_PY}" \
      --gold "${GOLD}" \
      --model "${MERGED}" \
      --tensor-parallel-size "${IE_TP}" \
      --max-new-tokens "${IE_MAX_NEW}" \
      --max-model-len "${IE_MAX_MODEL_LEN}" \
      --out "${OUT}"

DEST="${PROJECT_ROOT}/results/ie_sft/${IE_JOBID}"
mkdir -p "${DEST}"
cp -f "${OUT}" "${DEST}/pred_v3_vllm_${IE_JOBID}.jsonl"

echo "End:          $(date -u +%FT%TZ)"
echo "Predictions:  ${DEST}/pred_v3_vllm_${IE_JOBID}.jsonl"
echo "Score at home: python scripts/ie/evaluate.py --gold var/ie_extract_v3/gold_eval.jsonl --pred results/ie_sft/${IE_JOBID}/pred_v3_vllm_${IE_JOBID}.jsonl"
