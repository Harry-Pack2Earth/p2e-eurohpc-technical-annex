# Software manifest

Everything runs inside Apptainer images with base image tags pinned. Date-coded
tags are bumped deliberately and `torch.cuda.is_available()` is re-verified after
any bump, because silent regressions from base-image changes are a known trap.

## Container images

### `p2e-formulations.sif` (13.2 GB) - modelling lane

| | |
|---|---|
| Base | `nvcr.io/nvidia/pytorch:25.01-py3` |
| Python | 3.12 |
| CUDA | 12.6 (bundled with the NGC base, matched to its torch build) |
| Definition | `container/mn5.def` |

Installs the repository's own `[modelling]`, `[training]` and `[boost]` extras:
numpy (>=1.26), scikit-learn (>=1.5), pandas (>=2.2), mlflow (>=2.14),
optuna (>=3.6), shap (>=0.45), joblib (>=1.4), lightgbm (>=4.3), xgboost (>=2.0).

It deliberately does **not** pip-install a second torch wheel over the NGC build.
The NGC image ships torch matched to its bundled CUDA and cuDNN; installing a
generic PyPI torch on top risks silently shadowing the optimised build.

The base was moved from `24.10-py3` to `25.01-py3` because 24.10 ships Python
3.10 and this package requires >=3.11.

### `ie-train.sif` (8.4 GB) - extraction model fine-tuning

| | |
|---|---|
| Base | `docker.io/axolotlai/axolotl:main-20260817-py3.12-cu130-2.11.0` |
| Python | 3.12 |
| CUDA | 13.0 |
| Definition | `container/ie-train.def` |

Adds `flash-linear-attention` and `causal-conv1d` (both with pure-torch
fallbacks if the build fails), plus the p2e package installed `--no-deps`.

CUDA 13.0 in this image is supported by the MareNostrum 5 driver (595.71.05,
which supports CUDA 13.0 and below). This was checked against the live driver
before the image was adopted.

### `ie-vllm.sif` (8.9 GB) - extraction model inference

| | |
|---|---|
| Base | `docker.io/vllm/vllm-openai:v0.18.0` |
| Definition | `container/ie-vllm.def` |

### `p2e-mlip.sif` (10.4 GB) - MLIP structural embeddings

| | |
|---|---|
| Base | `nvcr.io/nvidia/pytorch:24.10-py3` |
| Definition | `container/mn5-mlip.def` |

Kept deliberately separate from the modelling image so that the faster-moving
MLIP stack (orb-models, nvalchemi) cannot destabilise the proven training image.
Pretrained weights are not baked in; they are staged to GPFS and pointed at by
path at run time, because compute nodes have no network access.

## Build and transfer

Images cannot be built on MareNostrum 5: compute and login nodes have no outbound
internet and no fakeroot. The workflow is:

1. Build locally (or via a remote builder) from the definition files here.
2. `rsync` the resulting `.sif` to `/gpfs/projects/ehpc838/`.
3. Reference it by absolute path in the submission script.

Python packages needed outside an image are supplied as a pre-built offline wheel
bundle, also staged to GPFS.

## Runtime environment settings

Applied consistently across jobs:

| Setting | Reason |
|---|---|
| `OMP_NUM_THREADS` set explicitly per rank | one OpenMP runtime per rank; avoids a duplicate-runtime fault traced in this stack |
| `TOKENIZERS_PARALLELISM=false` | avoids fork warnings and contention in dataloader workers |
| `srun --cpus-per-task` passed explicitly | `srun` does not inherit `--cpus-per-task` from `#SBATCH` on this system; without it the srun'd rank silently receives 1 CPU |
| `HF_HUB_OFFLINE` / local model paths | no network at run time |

## Languages and tooling

Python, SQL, Bash, SLURM. Experiment tracking with MLflow. Optuna for
hyperparameter search. DeepSpeed ZeRO-2 and axolotl for the language-model
fine-tune. BoTorch for multi-objective optimisation (CPU).
