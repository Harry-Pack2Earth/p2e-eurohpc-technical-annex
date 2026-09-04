#!/bin/bash
# slurm/nn_gpu.sbatch
#
# ACC single-GPU job: trains the multi-task NN backbone (shared trunk +
# per-family heads) via the real `train backbone --family F [--family F ...]
# --epochs N --datasets DIR --runs DIR` (needs [torch]; see
# src/p2e_formulations_pipeline/cli.py + models/nn/backbone.py). --family is
# repeatable and REQUIRED upstream — there is no single-family mode. It
# writes runs_dir/backbone_report.json.
#
# Single GPU by design — no torchrun/rendezvous machinery here (that's for
# multi-GPU/multi-node DDP; see the hpc-dl-perf skill's
# 08-slurm-launch-patterns.md). The plan explicitly scopes DDP/Megatron out
# unless a sweep genuinely benefits — on ~500-750 row tabular-derived data a
# single H100 is not the bottleneck.
#
# Submit via scripts/mn5/submit_gpu.sh nn [--gpus 1] --family F [--family F
# ...] [--epochs N] [--nsys] — it derives --cpus-per-task from --gpus (20
# CPU/GPU rule) and passes PROJECT_ROOT/SCRATCH_ROOT/DATASETS_DIR/
# RESULTS_DIR/FAMILIES/EPOCHS/NSYS_PROFILE via --export. The #SBATCH headers
# below are safe literal 1-GPU defaults for direct `sbatch slurm/nn_gpu.sbatch`
# use; submit_gpu.sh's CLI flags override them.
#
#SBATCH --job-name=p2e-backbone
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
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
echo "Partition:    ${SLURM_JOB_PARTITION:-unset}"
echo "Start:        $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# CRITICAL (confirmed BSC gotcha): srun no longer inherits --cpus-per-task
# from #SBATCH. Every GPU job that uses srun must export this or the srun'd
# process silently gets 1 CPU.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"

# ==============================================================================
# Paths — literal BSC fallbacks so this still runs if launched without
# submit_gpu.sh's --export.
# ==============================================================================
ACCOUNT="ehpc838"
PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/projects/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/gpfs/scratch/${ACCOUNT}/${USER:-pack423506}/p2e-formulations-modelling}"
SIF="${SIF:-${PROJECT_ROOT}/container/p2e-formulations.sif}"
DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/var/model_datasets}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/results}"
# Tier-1 RICH set for the 29k rematch — the six 1.8k+-row families plus the
# two buildable thermal ones (melting_enthalpy ~1.5k, crystallization_temperature
# ~0.7k) so data-poor thermal families borrow strength from the shared trunk.
# Override via submit_gpu.sh --family (repeatable) or edit here.
# Multi-task family set. Widened 8 -> 22 on 2026-08-04: every family with >=600 rows AND >=60 papers in
# the v12 encoding. This is not cosmetic — the multi-task premise is that data-rich families lend
# representation strength to data-poor ones, and the backbone has lost to trees three times while only
# ever seeing 8 tasks. 14 of these (barrier/optical/colour/degradation) only became viable through the
# lever-#8 signal recovery, so this is the first honest test of the premise.
#
# STALE ON v35, MEASURED 2026-08-10. This hand-list no longer matches the rule it claims to implement:
# `train backbone-families --min-papers 60 --min-rows 600` selects 24 families on v35, not these 22. The
# two missing are tensile_strength_film (1096 rows / 207 papers) and tensile_strength_moulded (1512 / 196)
# — v35's split of tensile_strength, and among the best-evidenced families in the corpus. Rows 3/6/8 all
# ran blind to them. Do NOT quietly paste them in: changing the set changes the objective (mean per-family
# R2), so it belongs in its own arm. Use FAMILY_MIN_PAPERS below rather than editing this line.
FAMILIES="${FAMILIES:-tensile_strength,elongation_at_break,tensile_modulus,melting_temperature,glass_transition_temperature,crystallinity,water_uptake,melting_enthalpy,water_contact_angle,thermal_degradation_temperature,light_transmittance,thermal_degradation_peak,disintegration,color_l_star,thermal_degradation_onset,color_b_star,color_a_star,crystallization_temperature,biodegradation,cold_crystallization_temperature,antimicrobial_inhibition_zone,residual_char_mass}"
EPOCHS="${EPOCHS:-400}"   # data-rich default; train backbone's own default is also 400

# GPU-track perf levers (hpc-dl-perf 04). Forwarded into the container as
# P2E_BACKBONE_* env; train_backbone reads them when the CLI can't pass a flag.
# Override per-run: `BATCH=512 AMP=1 COMPILE=1 submit_gpu.sh nn ...` (submit_gpu.sh
# --export=ALL carries them through) or edit here.
BATCH="${BATCH:-256}"     # minibatch size (full-batch no longer needed at 29k obs)
AMP="${AMP:-1}"           # bf16 autocast on CUDA (no-op fp32 on CPU)
COMPILE="${COMPILE:-0}"   # torch.compile OFF: inductor JIT can't find libcuda.so in the NGC container
                          # (driver at /usr/local/cuda/compat/lib, "libcuda.so cannot found!") and gives
                          # ~zero gain for a small tabular MLP. Override COMPILE=1 only if libcuda is linked.

# FEATURE-SPACE PARITY WITH THE TREES. The NN's default feature space is a drifted SUBSET of the trees':
# 0 descriptors vs 85-91, 6 numeric conditions vs 31, 3 categoricals vs 17 (see nn/unified.py). Every
# recorded "the NN is not competitive" result was measured against a model reading materially less
# information than its competitors. This flag existed but was reachable only from a python process, never
# from slurm, so a GPU tune would have spent its whole budget finding the best CRIPPLED model.
# FULL_FEATURES=0 is the deliberate legacy-narrow ablation arm, not the default.
# Accept EITHER name — the singularity call below sets P2E_BACKBONE_FULL_FEATURES from this var, so
# exporting that spelling at submit time would otherwise be silently overridden here.
FULL_FEATURES="${FULL_FEATURES:-${P2E_BACKBONE_FULL_FEATURES:-1}}"
SEEDS="${SEEDS:-3}"       # predictions averaged over N seeds; NN fold variance >> a tree's
MISSING_MASK="${MISSING_MASK:-1}"   # was-missing indicator per median-filled column (LightGBM has this natively)

echo "Families:     ${FAMILIES}"
echo "Epochs:       ${EPOCHS}"
echo "Batch/AMP/Compile: ${BATCH} / ${AMP} / ${COMPILE}"
echo "FullFeatures/Seeds/MissingMask: ${FULL_FEATURES} / ${SEEDS} / ${MISSING_MASK}"
echo "Container:    ${SIF}"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_SCRATCH}"
cd "${JOB_SCRATCH}"

# Caches — keep off $HOME (small/rate-limited on HPC clusters); allocator
# fragmentation mitigation per hpc-dl-perf's memory-and-allocator guidance.
export XDG_CACHE_HOME="${JOB_SCRATCH}/.cache"
export TRITON_CACHE_DIR="${JOB_SCRATCH}/.cache/triton"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"

# ==============================================================================
# COMMAND — the exact CLI invocation for this job. Adjust here.
# --family is repeatable upstream (one flag occurrence per family, built from
# the comma-separated FAMILIES var below); there is no --model flag for
# backbone (single shared-trunk architecture, not a model choice).
# ==============================================================================
P2E_CMD=(p2e-formulations train backbone)
# FAMILY_MIN_PAPERS selects the set BY RULE instead of by the hand-list above, and the two are mutually
# exclusive upstream (the CLI refuses both). Use it for the widened arm:
#
#   FAMILY_MIN_PAPERS=20 scripts/mn5/submit_gpu.sh nn ...
#
# THIS IS A DIFFERENT ARM, NOT AN UPGRADE. The objective is mean per-family R2, so a 55-family score and a
# 24-family score are not the same statistic and must never be diffed. Run it as its own row.
if [[ -n "${FAMILY_MIN_PAPERS:-}" ]]; then
  P2E_CMD+=(--min-papers "${FAMILY_MIN_PAPERS}" --min-rows "${FAMILY_MIN_ROWS:-0}")
  echo "FamilySet:    RULE papers>=${FAMILY_MIN_PAPERS} rows>=${FAMILY_MIN_ROWS:-0} (FAMILIES list IGNORED)"
else
  IFS=',' read -r -a FAMILY_ARR <<< "${FAMILIES}"
  for f in "${FAMILY_ARR[@]}"; do
    [[ -z "${f}" ]] && continue
    P2E_CMD+=(--family "${f}")
  done
  echo "FamilySet:    LIST of ${#FAMILY_ARR[@]} (from FAMILIES)"
fi
# --root IS REQUIRED, not cosmetic — P2E_MODELLING_CONFIG resolves relative to it, and the CLI fallback
# repo_root() only finds the repo from a SOURCE checkout, not from site-packages inside the container.
# Without it load_modelling_config FAILS OPEN to all-defaults and a row config is silently ignored.
P2E_CMD+=(--datasets "${DATASETS_DIR}" --runs "${JOB_SCRATCH}" --root "${PROJECT_ROOT}")
if [[ -n "${EPOCHS}" ]]; then
  P2E_CMD+=(--epochs "${EPOCHS}")
fi
# TUNED PARAMS from a prior `train tune-backbone` job. Point PARAMS_JSON at that job's
# backbone_tune.json (under RESULTS_DIR/backbone/<jobid>/) to run the FINAL model at the searched
# hyperparameters; without it this trains at library defaults, which is the untuned arm.
if [[ -n "${PARAMS_JSON:-}" ]]; then
  P2E_CMD+=(--params "${PARAMS_JSON}")
  echo "TunedParams:  ${PARAMS_JSON}"
fi

# THE SERVING ARTIFACT, on by default since 2026-08-20 (SAVE_ARTIFACT=0 to skip): one extra full-data fit
# per seed buys a servable, self-verifying artifact — the missing half of every backbone final to date.
[[ "${SAVE_ARTIFACT:-1}" == "1" ]] && P2E_CMD+=(--artifact-dir "${JOB_SCRATCH}/artifact_backbone")

# Optional Nsight Systems profiling pass (hpc-dl-perf skill,
# 01-tracing-toolchain.md). OFF by default — toggle with `submit_gpu.sh nn
# --nsys` (sets NSYS_PROFILE=1) or `export NSYS_PROFILE=1` before sbatch.
# Traces the WHOLE command as written; for a cropped window, add explicit
# cudaProfilerStart/Stop calls around the training loop and switch to
# --capture-range=cudaProfilerApi.
if [[ "${NSYS_PROFILE:-0}" == "1" ]]; then
  NSYS_BIN="${NSYS_BIN:-nsys}"
  TRACE_DIR="${JOB_SCRATCH}/nsys"
  mkdir -p "${TRACE_DIR}"
  echo "nsys:         ON -> ${TRACE_DIR}"
  P2E_CMD=("${NSYS_BIN}" profile \
    --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none --cpuctxsw=none --cuda-memory-usage=true \
    --force-overwrite=true \
    --output "${TRACE_DIR}/report-%h-%p" \
    "${P2E_CMD[@]}")
fi

echo "Command:      ${P2E_CMD[*]}"

# Singularity is a MODULE on MN5, not on the default PATH. Without this load, the `singularity exec` below
# fails instantly with "execve(): singularity: No such file or directory" (the observed nn_gpu fast-fail).
module load singularity/4.1.5 2>/dev/null || module load singularity
echo "Singularity:  $(command -v singularity || echo 'NOT FOUND')"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  singularity exec --nv \
    --bind "${PROJECT_ROOT}":"${PROJECT_ROOT}" \
    --bind "${SCRATCH_ROOT}":"${SCRATCH_ROOT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    --env P2E_BACKBONE_BATCH="${BATCH}" \
    --env P2E_BACKBONE_AMP="${AMP}" \
    --env P2E_BACKBONE_COMPILE="${COMPILE}" \
    --env P2E_BACKBONE_FULL_FEATURES="${FULL_FEATURES}" \
    --env P2E_BACKBONE_SEEDS="${SEEDS}" \
    --env P2E_BACKBONE_MISSING_MASK="${MISSING_MASK}" \
    "${SIF}" \
    "${P2E_CMD[@]}"

# ==============================================================================
# Stage results back to the durable project store — scratch is not backed up.
# `train backbone` writes backbone_report.json (+ any persisted models) under
# --runs (== JOB_SCRATCH here); only the local .cache/ is excluded.
# ==============================================================================
mkdir -p "${RESULTS_DIR}/backbone/${SLURM_JOB_ID}"
rsync -a --exclude '.cache/' "${JOB_SCRATCH}/" "${RESULTS_DIR}/backbone/${SLURM_JOB_ID}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Results:      ${RESULTS_DIR}/backbone/${SLURM_JOB_ID}"
