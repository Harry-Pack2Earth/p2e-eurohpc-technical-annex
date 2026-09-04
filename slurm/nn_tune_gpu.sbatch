#!/bin/bash
# slurm/nn_gpu.sbatch
#
# ACC single-GPU job: OPTUNA-TUNES the multi-task NN backbone (`train tune-backbone`), so the NN is judged
# on the same terms as the trees, which each get 50 trials per family from the CPU tune array. The backbone
# is ONE multi-task model, not a family x model grid, so it gets its own job rather than a slot in that
# array. `epochs` is a SEARCHED parameter here — do not pass a fixed --epochs.
#
# Derived from nn_gpu.sbatch. Original header:
# ACC single-GPU job: trains the multi-task NN backbone (shared trunk +
# per-family heads) via the real `train backbone --family F [--family F ...]
# --datasets DIR --runs DIR --n-trials N` (needs [torch]+[training]; see
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
# ...] [--nsys] — it derives --cpus-per-task from --gpus (20
# CPU/GPU rule) and passes PROJECT_ROOT/SCRATCH_ROOT/DATASETS_DIR/
# RESULTS_DIR/FAMILIES/N_TRIALS/NSYS_PROFILE via --export. The #SBATCH headers
# below are safe literal 1-GPU defaults for direct `sbatch slurm/nn_gpu.sbatch`
# use; submit_gpu.sh's CLI flags override them.
#
#SBATCH --job-name=p2e-backbone-tune
#SBATCH -A ehpc838
#SBATCH -q acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
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
# STALE ON v35, MEASURED 2026-08-10 — same note as slurm/nn_gpu.sbatch. The rule this list claims to
# implement selects 24 families on v35, not these 22; tensile_strength_film (207 papers) and
# tensile_strength_moulded (196) are missing. Widen via FAMILY_MIN_PAPERS as its own arm, not by editing
# this line, and keep this file's setting identical to nn_gpu.sbatch's.
FAMILIES="${FAMILIES:-tensile_strength,elongation_at_break,tensile_modulus,melting_temperature,glass_transition_temperature,crystallinity,water_uptake,melting_enthalpy,water_contact_angle,thermal_degradation_temperature,light_transmittance,thermal_degradation_peak,disintegration,color_l_star,thermal_degradation_onset,color_b_star,color_a_star,crystallization_temperature,biodegradation,cold_crystallization_temperature,antimicrobial_inhibition_zone,residual_char_mass}"
N_TRIALS="${N_TRIALS:-40}"   # Optuna trials; each trial is a full grouped-CV backbone fit

# GPU-track perf levers (hpc-dl-perf 04). Forwarded into the container as
# P2E_BACKBONE_* env; train_backbone reads them when the CLI can't pass a flag.
# Override per-run: `BATCH=512 AMP=1 COMPILE=1 submit_gpu.sh nn ...` (submit_gpu.sh
# --export=ALL carries them through) or edit here.
BATCH="${BATCH:-256}"     # minibatch size (full-batch no longer needed at 29k obs)
AMP="${AMP:-1}"           # bf16 autocast on CUDA (no-op fp32 on CPU)
COMPILE="${COMPILE:-0}"   # torch.compile OFF: inductor JIT can't find libcuda.so in the NGC container
                          # (driver at /usr/local/cuda/compat/lib, "libcuda.so cannot found!") and gives
                          # ~zero gain for a small tabular MLP. Override COMPILE=1 only if libcuda is linked.

# FEATURE-SPACE PARITY WITH THE TREES — see nn/unified.py. Without this the search spends its entire
# budget finding the best model that cannot see a single descriptor, which is the handicapped comparison
# the whole tune exists to retire. FULL_FEATURES=0 is the legacy-narrow ABLATION arm; run it as a second
# job if you want the with/without contrast, do not make it the default.
# Accept EITHER name. The sbatch var is FULL_FEATURES, but the thing a person remembers is the one the
# code reads (P2E_BACKBONE_FULL_FEATURES) — and because the singularity call sets that name explicitly
# from this var, exporting it at submit time would be SILENTLY OVERRIDDEN here. Reading it as a fallback
# makes both spellings work instead of one of them lying.
FULL_FEATURES="${FULL_FEATURES:-${P2E_BACKBONE_FULL_FEATURES:-1}}"
# Seeds per trial. 1 by default HERE (and 3 in the final run): every trial pays this multiplier, so at
# N_TRIALS=40 seeds=3 is a 3x GPU-hour bill. The tuned optimum is then re-fit with seed averaging by
# nn_gpu.sbatch, which is where the reported number comes from.
SEEDS="${SEEDS:-1}"
MISSING_MASK="${MISSING_MASK:-1}"

# DURABLE OPTUNA STUDY. Each trial is a full grouped-CV fit over 22 families, so an in-memory study that
# hits the 12h wall discards every completed trial, not just the one in flight. The db therefore lives on
# SCRATCH_ROOT under a STABLE name — deliberately NOT under JOB_SCRATCH, whose path contains
# $SLURM_JOB_ID and so would be abandoned by the very resubmit that needs it. Resubmitting the same
# command tops the study back up to N_TRIALS.
#
# The name encodes BOTH the config row and the arm. Two studies must never merge: narrow and aligned are
# different feature spaces, and row8 and row9 are different experiments — pooling either would let the
# sampler carry one run's conclusions into another and make best_params a mixture.
STUDY_TAG="$(basename "${P2E_MODELLING_CONFIG:-modelling.yaml}" .yaml)"
[[ "${FULL_FEATURES}" == "1" ]] && ARM="aligned" || ARM="narrow"
STUDY_NAME="${STUDY_NAME:-backbone_${STUDY_TAG}_${ARM}}"
STUDY_DB="${STUDY_DB:-${SCRATCH_ROOT}/optuna/${STUDY_NAME}.db}"
mkdir -p "$(dirname "${STUDY_DB}")"

echo "Families:     ${FAMILIES}"
echo "Optuna trials: ${N_TRIALS}"
echo "Batch/AMP/Compile: ${BATCH} / ${AMP} / ${COMPILE}"
echo "FullFeatures/Seeds/MissingMask: ${FULL_FEATURES} / ${SEEDS} / ${MISSING_MASK}"
echo "Study:        ${STUDY_NAME}"
echo "Study DB:     ${STUDY_DB}$([[ -f "${STUDY_DB}" ]] && echo '  (EXISTS — this run RESUMES it)')"
echo "Container:    ${SIF}"

# PARALLEL WORKERS. Submit as an ARRAY to share one study across several GPUs:
#
#   sbatch --array=0-3 slurm/nn_tune_gpu.sbatch     # 4 workers, one budget of N_TRIALS between them
#
# Each task is an independent Optuna worker against the same STUDY_DB. The trial budget is the STUDY's,
# not each worker's — the CLI stops a worker as soon as the study reaches N_TRIALS — so raising the array
# size shortens the tune instead of multiplying its cost. Task 0 is the LEADER: it alone computes the
# default-hyperparameter baseline and the final full-fold evaluation, and it alone writes
# backbone_tune.json. Point PARAMS_JSON at the leader's copy (.../w0/backbone_tune.json).
WORKER_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
N_WORKERS="${N_WORKERS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
# Namespaced so workers never share a scratch dir, and so the leader's report sits at a predictable path
# rather than under whichever per-task job id SLURM happened to assign.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_TAG="${SLURM_ARRAY_JOB_ID}/w${SLURM_ARRAY_TASK_ID}"
else
  RUN_TAG="${SLURM_JOB_ID}"
fi
echo "Worker:       ${WORKER_INDEX} of ${N_WORKERS}$([[ "${WORKER_INDEX}" == "0" ]] && echo '  (LEADER)' || echo '  (contributor)')"

JOB_SCRATCH="${SCRATCH_ROOT}/runs/${SLURM_JOB_NAME}_${RUN_TAG//\//_}"
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
P2E_CMD=(p2e-formulations train tune-backbone)
# FAMILY_MIN_PAPERS selects the set BY RULE instead of by the hand-list above. It MUST match the final
# run's setting: a study is tuned for the set it searched, so tuning a 24-family trunk and running a
# 55-family final applies hyperparameters chosen for a different objective. Set it on BOTH jobs or neither,
# and give the widened arm its OWN --study-name so it never tops up the narrow set's study.
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
# --root IS REQUIRED, not cosmetic. P2E_MODELLING_CONFIG takes a path relative to --root, and the CLI's
# fallback is repo_root() = Path(__file__).parents[3], which resolves to the repo only from a SOURCE
# checkout — inside the container the package is in site-packages, so it lands on a python lib dir with no
# config/ in it. load_modelling_config would then have FAILED OPEN and returned all-defaults, so a row
# config would be silently ignored while the job looked like it ran fine. The CPU sweep/tune arrays
# already pass this (full_sweep_array.sbatch, full_tune_array.sbatch); the GPU jobs did not.
P2E_CMD+=(--datasets "${DATASETS_DIR}" --runs "${JOB_SCRATCH}" --root "${PROJECT_ROOT}" \
          --n-trials "${N_TRIALS}" --storage "${STUDY_DB}" --study-name "${STUDY_NAME}" \
          --worker-index "${WORKER_INDEX}" --n-workers "${N_WORKERS}")

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
mkdir -p "${RESULTS_DIR}/backbone/${RUN_TAG}"
rsync -a --exclude '.cache/' "${JOB_SCRATCH}/" "${RESULTS_DIR}/backbone/${RUN_TAG}/"

echo "End:          $(date -u +%FT%TZ)"
echo "Results:      ${RESULTS_DIR}/backbone/${RUN_TAG}"
