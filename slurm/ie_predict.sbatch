#!/bin/bash
# slurm/ie_predict.sbatch
# =============================================================================
# W1 — run the fine-tuned v2 extraction adapter over the 60-paper / 252-example
# GOLD set and write predictions in the shape scripts/ie/evaluate.py scores.
# This is the "predict" step of train (W3) -> predict (here) -> evaluate (W1);
# it produces the actual v2 STRICT + field-F1 + hallucination number.
#
# Inference-only, single node. Loads base bf16 with device_map="auto" (pipeline
# parallel across the node's GPUs for memory headroom; ~1-GPU compute speed — the
# 27B bf16 base is ~55 GB so it needs >1 card), wraps the LoRA adapter, generates
# greedily with thinking OFF (the v1 inference contract). 252 gens, EOS-bounded by
# the SFT — ~1 h. The BARE base (no adapter) is NOT run here on purpose: un-tuned it
# has no EOS discipline and would generate to --max-new-tokens on every example
# (~15 h). Run that separately with IE_BASELINE=1 + a bounded --max-new-tokens.
#
# PATH ISOLATION (same hard rule as ie_sft.sbatch): everything writable lives under
# the ie_extract scratch namespace, never the modelling lane's runs dir.
#
# Usage (MN5 login node, after the training adapter is on the durable store):
#   sbatch slurm/ie_predict.sbatch                 # v2 adapter on job 44769991
#   IE_JOBID=<jobid> sbatch slurm/ie_predict.sbatch
#   IE_BASELINE=1 IE_MAX_NEW=1024 sbatch slurm/ie_predict.sbatch   # bare-base baseline
#
# TODO(mn5-user): the #SBATCH --chdir/-o/-e lines can't expand shell vars, so the MN5
# user is hard-coded to pack423506 (as in ie_sft.sbatch). Change it in the four
# #SBATCH lines if your MN5 user differs; the body derives everything from ${USER}.
# =============================================================================
#SBATCH --job-name=ie-predict
#SBATCH -A ehpc838
# acc_ehpc is the ONLY valid QOS on this ACC allocation (MEASURED, modelling lane).
#SBATCH -q acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# 2 GPUs, NOT 4: predict.py's device_map="auto" is PIPELINE parallel (one GPU active at a
# time during generate), so 4 vs 2 GPUs is IDENTICAL wall-clock here. The 54 GB bf16 base
# splits comfortably across 2x64 GB. Asking for a whole 4-GPU node made this job starve
# behind the modelling lane's single-GPU full_tune array (SLURM backfill fills 1-GPU holes
# with array tasks; a 4-GPU whole-node job waits for 4 co-located GPUs to drain at once).
# 2 co-located GPUs free far more often -> schedules promptly. 20 CPU/GPU allocation cap -> 40.
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=40
#SBATCH --time=03:00:00
#SBATCH --chdir=/gpfs/scratch/ehpc838/pack423506/ie_extract
#SBATCH -o /gpfs/scratch/ehpc838/pack423506/ie_extract/logs/%x_%j.out
#SBATCH -e /gpfs/scratch/ehpc838/pack423506/ie_extract/logs/%x_%j.err

set -euo pipefail

# ==============================================================================
# Preflight banner
# ==============================================================================
echo "JobID:        ${SLURM_JOB_ID:-unset}"
echo "JobName:      ${SLURM_JOB_NAME:-unset}"
echo "NodeList:     ${SLURM_JOB_NODELIST:-unset}"
echo "GPUs:         ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# ==============================================================================
# Paths — isolated to the ie_extract namespace (identical derivation to ie_sft).
# ==============================================================================
ACCOUNT="ehpc838"
MN5_USER="${USER:-pack423506}"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${MN5_USER}/p2e-formulations-modelling}"
ISCRATCH="${ISCRATCH:-/gpfs/scratch/${ACCOUNT}/${MN5_USER}/ie_extract}"
MODELS_DIR="${MODELS_DIR:-/gpfs/projects/${ACCOUNT}/models}"
BASE_MODEL="${BASE_MODEL:-${MODELS_DIR}/Qwen3.5-27B}"
SIF="${SIF:-${PROJECT_ROOT}/container/ie-train.sif}"

# Which trained run to evaluate (the adapter lives on the durable store the training
# job rsynced it to: results/ie_sft/<jobid>). Default = the first real run.
IE_JOBID="${IE_JOBID:-44769991}"
ADAPTER="${ADAPTER:-${PROJECT_ROOT}/results/ie_sft/${IE_JOBID}}"
GOLD="${GOLD:-${PROJECT_ROOT}/var/ie_extract/gold_eval.jsonl}"
# IE_MULTIPASS=1 runs the test-time-compute driver (pass-1 + critic pass + merge + gates) instead of
# single-pass predict.py. Same arg surface (--gold/--base/--adapter/--max-new-tokens/--batch-size/--out),
# so only the script path + the output tag change. Ablation via --no-pass2/--no-gate is a code edit if
# needed; the default here is the full harness.
if [[ "${IE_MULTIPASS:-0}" == "1" ]]; then
  PREDICT_PY="${PROJECT_ROOT}/scripts/ie/predict_multipass.py"
else
  PREDICT_PY="${PROJECT_ROOT}/scripts/ie/predict.py"
fi
IE_MAX_NEW="${IE_MAX_NEW:-6144}"   # v3 targets ~5517 tok; 6144 clears both v2 and v3 (no inference truncation)
IE_BATCH="${IE_BATCH:-32}"         # sequences decoded together — the throughput lever (GatedDeltaNet
                                   # decode is pure-torch here; batch-1 leaves the GPU ~32% idle)

# Multipass harness knobs (docs/qwen-harness-experiments.md): IE_PASSES (total passes) + IE_VARIANT
# (pass-2+ prompt: critic|qwen_complete|qwen_attrib) + IE_SEED (a pred file to CHAIN from — this model
# critic-passes another model's output) + IE_CORRECT=1 (attribution-correction pass). All baked into
# RUN_TAG so every experiment writes a DISTINCT pred file (no clobber).
IE_PASSES="${IE_PASSES:-2}"
IE_VARIANT="${IE_VARIANT:-critic}"
MULTIPASS_ARGS=()
if [[ "${IE_BASELINE:-0}" == "1" ]]; then
  RUN_TAG="base"
  ADAPTER_ARGS=()   # no --adapter => bare base model
elif [[ "${IE_MULTIPASS:-0}" == "1" ]]; then
  MULTIPASS_ARGS=(--passes "${IE_PASSES}" --prompt-variant "${IE_VARIANT}")
  TAG_SUFFIX=""
  if [[ -n "${IE_SEED:-}" ]]; then
    MULTIPASS_ARGS+=(--seed-pred "${IE_SEED}")
    TAG_SUFFIX="_seed$(basename "${IE_SEED}" .jsonl | sed 's/^pred_//;s/^multipass_//')"
  fi
  if [[ "${IE_CORRECT:-0}" == "1" ]]; then
    MULTIPASS_ARGS+=(--correct)
    TAG_SUFFIX="${TAG_SUFFIX}_corr"
  fi
  if [[ -n "${IE_SHARD:-}" ]]; then   # data-parallel: i/K -> K jobs; fold i into the tag so outputs are distinct
    MULTIPASS_ARGS+=(--shard "${IE_SHARD}")
    TAG_SUFFIX="${TAG_SUFFIX}_sh$(echo "${IE_SHARD}" | tr '/' 'of')"
  fi
  RUN_TAG="multipass_${IE_VARIANT}_p${IE_PASSES}${TAG_SUFFIX}_${IE_JOBID}"
  ADAPTER_ARGS=(--adapter "${ADAPTER}")
else
  RUN_TAG="v2_${IE_JOBID}"
  ADAPTER_ARGS=(--adapter "${ADAPTER}")
fi
# IE_TAG namespaces a run over a NON-gold corpus (e.g. IE_TAG=barrier) so its output does not clobber
# the gold-eval predictions (which share the same VARIANT/PASSES/JOBID RUN_TAG).
RUN_TAG="${IE_TAG:+${IE_TAG}_}${RUN_TAG}"
OUT="${ISCRATCH}/predictions/pred_${RUN_TAG}.jsonl"

echo "MN5_USER:     ${MN5_USER}"
echo "BaseModel:    ${BASE_MODEL}"
echo "Adapter:      ${IE_BASELINE:+<none: bare base baseline>}${IE_BASELINE:-${ADAPTER}}"
echo "Gold:         ${GOLD}"
echo "Out:          ${OUT}"
echo "MaxNewTokens: ${IE_MAX_NEW}"

# ==============================================================================
# Sanity checks — fail CLOSED (a fail-open here scores garbage against gold).
# ==============================================================================
[[ -f "${SIF}" ]]        || { echo "SIF not found: ${SIF}" >&2; exit 1; }
[[ -d "${BASE_MODEL}" ]] || { echo "Base model not found: ${BASE_MODEL}" >&2; exit 1; }
[[ -f "${GOLD}" ]]       || { echo "Gold set not found: ${GOLD} — sync_push --data-src var/ie_extract" >&2; exit 1; }
[[ -f "${PREDICT_PY}" ]] || { echo "predict.py not found: ${PREDICT_PY}" >&2; exit 1; }
if [[ "${IE_BASELINE:-0}" != "1" ]]; then
  [[ -f "${ADAPTER}/adapter_config.json" ]] || { echo "Adapter not found: ${ADAPTER}/adapter_config.json (is IE_JOBID=${IE_JOBID} right?)" >&2; exit 1; }
fi

# ==============================================================================
# Caches + offline env (reuse the isolated caches; no internet on compute nodes).
# ==============================================================================
mkdir -p "${ISCRATCH}/predictions" "${ISCRATCH}/logs" \
         "${ISCRATCH}/.cache/huggingface/hub" "${ISCRATCH}/.cache/triton"
export XDG_CACHE_HOME="${ISCRATCH}/.cache"
export HF_HOME="${ISCRATCH}/.cache/huggingface"
export HF_HUB_CACHE="${ISCRATCH}/.cache/huggingface/hub"
export TRITON_CACHE_DIR="${ISCRATCH}/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

BINDS="${PROJECT_ROOT}:${PROJECT_ROOT},${ISCRATCH}:${ISCRATCH},${MODELS_DIR}:${MODELS_DIR}"

# ==============================================================================
# Predict. device_map="auto" inside predict.py shards the base across the node's
# GPUs; --nv exposes them. Run from PROJECT_ROOT so the script's relative repo
# assumptions hold; the script itself only needs torch/transformers/peft.
# ==============================================================================
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${BINDS}" \
    --pwd "${PROJECT_ROOT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" --env HF_HOME="${HF_HOME}" --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
    --env TOKENIZERS_PARALLELISM=false --env OMP_NUM_THREADS=1 \
    "${SIF}" \
    python "${PREDICT_PY}" \
      --gold "${GOLD}" \
      --base "${BASE_MODEL}" \
      "${ADAPTER_ARGS[@]}" \
      "${MULTIPASS_ARGS[@]}" \
      --max-new-tokens "${IE_MAX_NEW}" \
      --batch-size "${IE_BATCH}" \
      --out "${OUT}"

# ==============================================================================
# Stage predictions to the durable store next to the adapter, so sync_pull grabs
# them and evaluate.py (local, no GPU) scores them at home.
# ==============================================================================
DEST="${PROJECT_ROOT}/results/ie_sft/${IE_JOBID}"
mkdir -p "${DEST}"
cp -f "${OUT}" "${DEST}/pred_${RUN_TAG}.jsonl"

echo "End:          $(date -u +%FT%TZ)"
echo "Predictions:  ${DEST}/pred_${RUN_TAG}.jsonl"
echo "Score at home: python scripts/ie/evaluate.py --gold var/ie_extract/gold_eval.jsonl --pred results/ie_sft/${IE_JOBID}/pred_${RUN_TAG}.jsonl"
