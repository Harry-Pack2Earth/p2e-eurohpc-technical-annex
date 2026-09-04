#!/bin/bash
# slurm/ie_sft.sbatch
# =============================================================================
# W3 — v2 IN-HOUSE EXTRACTION MODEL fine-tune (materials-IE), single-node 4xH100.
#
# Reproduces Raul's recovered v1 recipe (bf16 LoRA r32/a64, lr1e-4, 2ep, seq8192,
# seed17) via axolotl + DeepSpeed ZeRO-3 + Apptainer, on OUR ehpc838 allocation.
# This is the proven course stack (aif-dl-optimization-parallelization-training
# hands-on-m5) adapted to this repo's SLURM conventions (see slurm/nn_gpu.sbatch).
#
# PATH ISOLATION (hard requirement — never collide with the modelling lane):
#   scratch namespace : /gpfs/scratch/ehpc838/<user>/ie_extract/   (NOT .../p2e-formulations-modelling)
#   job name          : ie-sft
#   container         : container/ie-train.sif  (LLM stack; NOT p2e-formulations.sif)
#   dataset           : <project>/var/ie_extract/{sft_train,sft_dev}.jsonl
#   base model        : /gpfs/projects/ehpc838/models/Qwen3.5-27B (shared, read-only)
#
# AUTHOR-ONLY artifact. Do NOT sbatch this until the PREFLIGHT in
# scripts/ie/README-ie-train.md passes (base model present, dataset synced, SIF
# built+synced, gold papers held out).
#
# TODO(mn5-user): the #SBATCH --chdir/-o/-e lines below cannot expand shell vars,
# so the MN5 user is hard-coded to this project's account (pack423506). If your MN5
# user differs, change pack423506 in the four #SBATCH lines below (the body derives
# everything else from ${USER}).
# =============================================================================
#SBATCH --job-name=ie-sft
#SBATCH -A ehpc838
# MEASURED (modelling lane): -q acc_ehpc is the ONLY valid QOS on this ACC-only
# allocation — gp_ehpc is INVALID and the submit is rejected. Do not "fix" it.
#SBATCH -q acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
# 18h with margin (v1 fit in 13.1h). Hard cap on this allocation is 72h; stay well
# under. --requeue: assume the wall/preemption kills this at least once — SLURM
# requeues it and the body auto-resumes from the latest SCRATCH checkpoint.
#SBATCH --time=18:00:00
#SBATCH --requeue
# ABSOLUTE -o/-e/--chdir into the isolated scratch logs/ (SLURM won't expand vars or
# mkdir these — the PREFLIGHT creates logs/).
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
echo "CPUsPerTask:  ${SLURM_CPUS_PER_TASK:-unset}"
echo "GPUs:         ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-unset}}"
echo "Partition:    ${SLURM_JOB_PARTITION:-unset}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
# Bare nvidia-smi header too: its top-right "CUDA Version" is the driver's max supported CUDA,
# the fact that decides whether the cu130 image runs here. Captured once on the record (the
# base-image choice inferred driver 595 -> CUDA >= 13.0; this is where that gets proven).
nvidia-smi | head -4 || true

# CRITICAL (confirmed BSC gotcha, same as nn_gpu.sbatch): srun no longer inherits
# --cpus-per-task from #SBATCH. Without this export the srun'd rank silently gets 1
# CPU and the dataloader starves.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# ==============================================================================
# Paths — all path-isolated to the ie_extract namespace.
# ==============================================================================
ACCOUNT="ehpc838"
MN5_USER="${USER:-pack423506}"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${MN5_USER}/p2e-formulations-modelling}"
ISCRATCH="${ISCRATCH:-/gpfs/scratch/${ACCOUNT}/${MN5_USER}/ie_extract}"      # DEDICATED, isolated
MODELS_DIR="${MODELS_DIR:-/gpfs/projects/${ACCOUNT}/models}"                 # shared base-model store
BASE_MODEL="${BASE_MODEL:-${MODELS_DIR}/Qwen3.5-27B}"
SIF="${SIF:-${PROJECT_ROOT}/container/ie-train.sif}"                         # LLM training image, NOT p2e-formulations.sif
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${PROJECT_ROOT}/config/ie_sft.yaml}"
# v3 dataset (uncapped, exhaustive spec). Must match the datasets path in config/ie_sft.yaml —
# the guard checks the SAME dir axolotl reads, so a missing v3 push fails closed here (not at
# preprocess). Override IE_DATA_DIR for a different set.
IE_DATA_DIR="${IE_DATA_DIR:-${PROJECT_ROOT}/var/ie_extract_v3}"
TRAIN_JSONL="${IE_DATA_DIR}/sft_train.jsonl"
DEV_JSONL="${IE_DATA_DIR}/sft_dev.jsonl"

# MEASURED (modelling lane): single node → use STANDALONE torchrun, skip c10d
# rendezvous entirely (MASTER_ADDR / rdzv_endpoint / node_rank are multi-node pain
# we do not need and that flakes at rank init). One node, 4 local workers.
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# JOB-UNIQUE render + deepspeed paths. These were fixed filenames, so two concurrent training jobs
# (e.g. the q35 vs q38 base comparison) rendered to the SAME file — one job's sed edits clobbered the
# other's config mid-flight (observed: "no gradient_accumulation_steps key" when the grep raced a
# concurrent rewrite). Scoping by SLURM_JOB_ID lets any number of trainings run at once. The
# dataset_prepared_path is already per-config (distinct output tags), so it does not collide.
RENDERED_CONFIG="${ISCRATCH}/generated-configs/ie_sft.rendered.${SLURM_JOB_ID:-manual}.yaml"
DEEPSPEED_CONFIG="${ISCRATCH}/ds_zero2.${SLURM_JOB_ID:-manual}.json"

echo "MN5_USER:     ${MN5_USER}"
echo "ProjectRoot:  ${PROJECT_ROOT}"
echo "Scratch:      ${ISCRATCH}"
echo "BaseModel:    ${BASE_MODEL}"
echo "Container:    ${SIF}"

# ==============================================================================
# Sanity checks — fail CLOSED before burning the allocation (fail-open here would
# silently train on the wrong base or a leaked dataset).
# ==============================================================================
[[ -f "${SIF}" ]]            || { echo "SIF not found: ${SIF} (build container/ie-train.def, then sync_push)" >&2; exit 1; }
[[ -d "${BASE_MODEL}" ]]     || { echo "Base model not found: ${BASE_MODEL}" >&2; exit 1; }
[[ -f "${TRAIN_JSONL}" ]]    || { echo "Train set not found: ${TRAIN_JSONL} (run W2, then sync_push --data-src var/ie_extract)" >&2; exit 1; }
[[ -f "${DEV_JSONL}" ]]      || { echo "Dev set not found: ${DEV_JSONL}" >&2; exit 1; }
[[ -f "${CONFIG_TEMPLATE}" ]] || { echo "Config template not found: ${CONFIG_TEMPLATE}" >&2; exit 1; }

# ==============================================================================
# Isolated dirs + caches (keep off $HOME; same allocator-fragmentation mitigation
# as nn_gpu.sbatch).
# ==============================================================================
mkdir -p "${ISCRATCH}/generated-configs" "${ISCRATCH}/prepared-datasets" \
         "${ISCRATCH}/outputs" "${ISCRATCH}/logs" "${ISCRATCH}/.cache/huggingface/datasets" \
         "${ISCRATCH}/.cache/triton"
export XDG_CACHE_HOME="${ISCRATCH}/.cache"
export HF_HOME="${ISCRATCH}/.cache/huggingface"
# Set HF_HUB_CACHE EXPLICITLY (do not rely on the HF_HOME/hub default — inside the container it
# was resolving elsewhere, so the pre-staged flash-attn2 kernel was never found and FA2 died).
# transformers 5.5 fetches FA2 as a hub kernel (kernels-community/flash-attn2); offline it must
# be pre-cached HERE. Staged once from ~/ie-fa2-kernel (see runbook).
export HF_HUB_CACHE="${ISCRATCH}/.cache/huggingface/hub"
mkdir -p "${HF_HUB_CACHE}"
# JOB-UNIQUE write-heavy caches. Two concurrent trainings converting the SAME sft_train.jsonl into a
# SHARED HF datasets cache race on the arrow write -> one dies with DatasetGenerationError (observed:
# q38 died while q35 loaded the identical data fine). Triton can likewise race on concurrent kernel
# compiles. Scope both by job id. HF_HUB_CACHE stays shared — it is offline, read-only model/kernel
# weights, no write contention.
export HF_DATASETS_CACHE="${ISCRATCH}/.cache/huggingface/datasets.${SLURM_JOB_ID:-manual}"
export TRITON_CACHE_DIR="${ISCRATCH}/.cache/triton.${SLURM_JOB_ID:-manual}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRITON_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# Offline: MN5 compute nodes have no outbound internet; the base model is already
# staged on gpfs. Force every hub client offline so a resolve attempt can't hang.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AXOLOTL_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
# One OpenMP runtime per rank (avoids the libomp duplicate-runtime segfault class
# noted in project memory for the serve path).
export OMP_NUM_THREADS=1
# NCCL_DEBUG is verbose (floods the log at scale) — gate it behind a smoke/env flag,
# never always-on. `IE_NCCL_DEBUG=1 sbatch …` (or IE_SMOKE=1) to turn it on.
if [[ "${IE_NCCL_DEBUG:-0}" == "1" || "${IE_SMOKE:-0}" == "1" ]]; then
  export NCCL_DEBUG=INFO
  echo "NCCL_DEBUG:   INFO (smoke/debug)"
fi

# ==============================================================================
# Render the checked-in config template: substitute the __MN5_USER__ token into a
# scratch copy (mirrors the course's render_axolotl_config.sh). The repo file stays
# a template; axolotl only ever reads the rendered copy.
# ==============================================================================
sed "s/__MN5_USER__/${MN5_USER}/g" "${CONFIG_TEMPLATE}" > "${RENDERED_CONFIG}"
if grep -q "__[A-Z0-9_]\+__" "${RENDERED_CONFIG}"; then
  echo "Unresolved template placeholders remain in ${RENDERED_CONFIG}" >&2
  exit 1
fi
echo "Rendered:     ${RENDERED_CONFIG}"

# ==============================================================================
# Preserve effective batch across GPU counts (quality invariant).
# effective_batch = micro_batch_size * gradient_accumulation_steps * n_gpu. The recipe targets 32.
# n_gpu varies — a 4-GPU whole-node run vs a 2-GPU run that SCHEDULES far faster (measured: 2 co-
# located GPUs free in minutes while a 4-GPU node sat pending >1 day). So DERIVE grad_accum from
# GPUS_PER_NODE instead of hardcoding: hardcoding 16 for a 2-GPU run would silently make a later
# 4-GPU run eff-batch 64 (a real, invisible quality change — the exact drift to avoid). Derived =>
# eff batch is ALWAYS TARGET_EFF_BATCH, identical optimisation dynamics on any GPU count.
# ==============================================================================
TARGET_EFF_BATCH="${TARGET_EFF_BATCH:-32}"
MICRO_BATCH="$(awk -F': *' '/^micro_batch_size:/{print $2; exit}' "${RENDERED_CONFIG}" | awk '{print $1}')"
MICRO_BATCH="${MICRO_BATCH:-1}"
if (( TARGET_EFF_BATCH % (MICRO_BATCH * GPUS_PER_NODE) != 0 )); then
  echo "ERROR: eff-batch ${TARGET_EFF_BATCH} not divisible by micro(${MICRO_BATCH})*gpus(${GPUS_PER_NODE}) — set TARGET_EFF_BATCH" >&2
  exit 1
fi
GRAD_ACCUM=$(( TARGET_EFF_BATCH / (MICRO_BATCH * GPUS_PER_NODE) ))
if ! grep -q '^gradient_accumulation_steps:' "${RENDERED_CONFIG}"; then
  echo "ERROR: no gradient_accumulation_steps key in ${RENDERED_CONFIG} to override" >&2
  exit 1
fi
sed -i "s/^gradient_accumulation_steps:.*/gradient_accumulation_steps: ${GRAD_ACCUM}/" "${RENDERED_CONFIG}"
echo "EffBatch:     ${TARGET_EFF_BATCH} = micro ${MICRO_BATCH} x grad_accum ${GRAD_ACCUM} x gpus ${GPUS_PER_NODE}"

# ==============================================================================
# Materialise the DeepSpeed ZeRO-3 config into the isolated scratch dir (kept out
# of the repo so only the 5 named W3 files are checked in).
#
# WHY ZeRO-3 (not ZeRO-2): the base is FROZEN (LoRA), so only the ~54 GB bf16 PARAM
# tensor matters for memory — optimiser/grad state exists only for the tiny LoRA
# MEASURED (smoke OOM, 54 GB/GPU): a FROZEN base is NOT sharded by ZeRO — ZeRO only partitions
# TRAINABLE params, so both ZeRO-2 AND ZeRO-3 REPLICATE the full base on every GPU. bf16 base =
# 54 GB/GPU → OOM on 64 GB cards. FIX = QLoRA (config: load_in_4bit) → 4-bit base ~14 GB/GPU,
# fits with headroom. With a 14 GB base params don't need sharding, so ZeRO-2 (shard the tiny
# LoRA optimiser/grads) — NOT ZeRO-3, which is incompatible with bitsandbytes 4-bit params.
# ==============================================================================
cat > "${DEEPSPEED_CONFIG}" <<'DSJSON'
{
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 2,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_scatter": true
  },
  "gradient_clipping": 1.0,
  "train_micro_batch_size_per_gpu": "auto"
}
DSJSON
echo "DeepSpeed:    ${DEEPSPEED_CONFIG}"

# ==============================================================================
# Singularity is a MODULE on MN5, not on PATH (same as nn_gpu.sbatch).
# ==============================================================================
module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

BINDS="${PROJECT_ROOT}:${PROJECT_ROOT},${ISCRATCH}:${ISCRATCH},${MODELS_DIR}:${MODELS_DIR}"

# FAIL FAST: transformers 5.5 loads FA2 as an offline hub-kernel from HF_HUB_CACHE. If it is not
# pre-cached there the model load dies ~40s in with a cryptic LocalEntryNotFoundError. Check it
# here so a missing kernel fails in SECONDS with a clear message. Stage once from ~/ie-fa2-kernel.
if [[ -d "${HF_HUB_CACHE}/models--kernels-community--flash-attn2/snapshots" ]]; then
  echo "FA2 kernel:   present in ${HF_HUB_CACHE}"
else
  echo "FA2 kernel:   not cached — OK, config uses sdp_attention (only needed if flash_attention: true)"
fi

# ==============================================================================
# Step 1 — tokenise/pack the dataset once (CPU; no --nv needed). Axolotl caches the
# result under dataset_prepared_path (isolated). Cheap and makes the GPU step's
# first log line be training, not preprocessing.
# ==============================================================================
# --cpus-per-task is passed EXPLICITLY on every srun (MEASURED: inheritance from
# #SBATCH is unreliable even WITH the export above — belt and braces).
srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec \
    --bind "${BINDS}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" --env HF_HOME="${HF_HOME}" --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
    --env AXOLOTL_DO_NOT_TRACK=1 --env DO_NOT_TRACK=1 \
    "${SIF}" \
    axolotl preprocess "${RENDERED_CONFIG}"

# ==============================================================================
# Step 2 — train. Single-node 4-GPU via STANDALONE torchrun (no c10d rendezvous).
# axolotl's `--launcher torchrun -- <args>` forwards <args> to torchrun; --standalone
# spins up a local rendezvous on a free port and needs no MASTER_ADDR / node_rank.
#
# Resumable: on a --requeue after the wall clock kills us, resume from the latest
# checkpoint in the isolated SCRATCH output_dir (checkpoints are step-based and land
# in scratch, which is NOT backed up — that's why the job end rsyncs the adapter to
# projects/results). Override save cadence here (not in the frozen recipe config) so
# an interrupted 18h run loses at most ~SAVE_STEPS steps, not a whole epoch.
# ==============================================================================
# DERIVE OUT_DIR from the config's output_dir — do NOT hardcode. A stale hardcoded 'ie-sft-v1'
# here (while the config wrote ie-sft-v3) made the requeue-resume scan find v2's rank-32
# checkpoints and inject resume_from_checkpoint into a rank-64 run -> "size mismatch ... lora_B
# [.,32] vs [.,64]" at load. Parse it straight out of the rendered config so the two can never drift.
OUT_DIR="$(awk -F': *' '/^output_dir:/{print $2; exit}' "${RENDERED_CONFIG}" | tr -d '"'"'"' ')"
[[ -n "${OUT_DIR}" ]] || { echo "could not parse output_dir from ${RENDERED_CONFIG}" >&2; exit 1; }
echo "OutDir:       ${OUT_DIR} (derived from config)"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

# Run-mode settings are BAKED INTO THE RENDERED CONFIG, not passed as axolotl CLI flags.
# MEASURED: this axolotl forwards UNRECOGNISED CLI flags to torchrun, which rejects them
# (the first smoke died `torchrun: error: unrecognized arguments: --max_steps`). The frozen
# template stays clean; only the scratch render gets these. IE_SMOKE=1 is the gate: load the
# model + ONE training step + exit, saving nothing — a broken image dies at load in seconds.
# The template already sets `save_strategy: "epoch"`, so REPLACE that key in place (a second
# `save_strategy:` would be a duplicate YAML key). The other keys are absent from the template,
# so appending them adds no duplicate.
if [[ "${IE_SMOKE:-0}" == "1" ]]; then
  echo "SMOKE:        max_steps=1, save_strategy=no — model load + 1 training step gate"
  sed -i 's/^save_strategy:.*/save_strategy: "no"/' "${RENDERED_CONFIG}"
  printf '\n# --- smoke override (sbatch, IE_SMOKE=1) ---\nmax_steps: 1\n' >> "${RENDERED_CONFIG}"
else
  echo "SaveCadence:  every ${SAVE_STEPS} steps, keep ${SAVE_TOTAL_LIMIT}"
  sed -i 's/^save_strategy:.*/save_strategy: steps/' "${RENDERED_CONFIG}"
  printf '\n# --- save cadence (sbatch) ---\nsave_steps: %s\nsave_total_limit: %s\n' \
    "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" >> "${RENDERED_CONFIG}"
fi

# Requeue-resume: also via the rendered config (same CLI-forwarding reason). Not in the
# template, so a plain append is safe.
LATEST_CKPT="$(ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
if [[ -n "${LATEST_CKPT}" ]]; then
  echo "Resume:       ${LATEST_CKPT} (requeue detected)"
  printf '\n# --- resume (sbatch, requeue) ---\nresume_from_checkpoint: "%s"\n' "${LATEST_CKPT}" >> "${RENDERED_CONFIG}"
else
  echo "Resume:       none (fresh start)"
fi

srun --ntasks=1 --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${BINDS}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    --env HF_HOME="${HF_HOME}" --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env HF_DATASETS_OFFLINE=1 \
    --env AXOLOTL_DO_NOT_TRACK=1 --env DO_NOT_TRACK=1 \
    --env TOKENIZERS_PARALLELISM=false --env OMP_NUM_THREADS=1 \
    ${NCCL_DEBUG:+--env NCCL_DEBUG="${NCCL_DEBUG}"} \
    "${SIF}" \
    axolotl train "${RENDERED_CONFIG}" \
      --launcher torchrun -- \
        --standalone \
        --nnodes=1 \
        --nproc-per-node="${GPUS_PER_NODE}"

# ==============================================================================
# Stage the FINAL adapter to the durable project store INSIDE the job (scratch is
# NOT backed up and is purged). The LoRA adapter is small (~600 MB); pull it home
# with scripts/mn5/sync_pull*.sh. Intermediate checkpoint-*/ dirs are excluded —
# only the final adapter + trainer state graduate to projects/results.
# ==============================================================================
DEST="${PROJECT_ROOT}/results/ie_sft/${SLURM_JOB_ID}"
mkdir -p "${DEST}"
rsync -a --exclude '.cache/' --exclude 'checkpoint-*/' "${OUT_DIR}/" "${DEST}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Adapter:      ${DEST}"
