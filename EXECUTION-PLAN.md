# Execution plan

What we run, in what order, at what measured cost. Figures marked MEASURED come
from completed jobs on allocation `ehpc838` during the previous access period.

## Job shapes

| Script | GPUs | CPUs | Wall | Workload |
|---|---|---|---|---|
| `ie_sft.sbatch` | 4 | 80 | 18:00 | 27B LoRA fine-tune. **MEASURED 13.1 h** |
| `ie_vllm_predict.sbatch` | 2 | 40 | 01:00 | vLLM batch inference over the corpus |
| `ie_predict.sbatch` | 2 | 40 | 03:00 | HuggingFace-path batch inference |
| `nn_gpu.sbatch` | 1 | 20 | 04:00 | multi-task neural backbone training |
| `nn_tune_gpu.sbatch` | 1 | 20 | 12:00 | backbone hyperparameter search |
| `setnet_gpu.sbatch` | 1 | 20 | 06:00 | set-transformer over formulation components |
| `setnet_tune_gpu.sbatch` | 1 | 20 | 12:00 | set-transformer tuning |
| `kinetic_gpu.sbatch` | 1 | 20 | 04:00 | degradation kinetics models |
| `mlip_embed_gpu.sbatch` | 1 | 20 | 02:00 | MLIP structural embeddings |
| `full_tune_array.sbatch` | CPU | 8/task | 02:00 | per-family hyperparameter arrays |
| `full_sweep_array.sbatch` | CPU | 8/task | 00:30 | per-family model bake-off |

Nothing in the workload requires more than 4 GPUs or more than one node. The
fine-tune is the largest single job and uses exactly one full ACC node.

## Sequence for one full cycle

```
1. Corpus growth        extraction model inference over newly acquired papers
                        ie_vllm_predict.sbatch, 2 GPU, ~1 h per batch

2. Encode               datastore to per-family model-ready frames
                        CPU, off-cluster

3. Bake-off             every model type against every family, fixed defaults
                        full_sweep_array.sbatch, array of 98, 8 CPU per task

4. Tune                 Optuna search, family x model
                        full_tune_array.sbatch, array of 98 x 9, 8 CPU per task

5. GPU model training   backbone, set-transformer, kinetics
                        nn_gpu / setnet_gpu / kinetic_gpu, 1 GPU each

6. Promote              grouped cross-validation verdict, physics corroboration
                        gates, champion selection per family. CPU.

7. Optimise             BoTorch expected hypervolume improvement over the
                        surrogate ensemble. CPU.

8. Laboratory feedback  results ingested, uncertainty threshold triggers a
                        return to step 3
```

The extraction model fine-tune (`ie_sft.sbatch`) runs on its own cadence,
whenever enough new training examples have accumulated to justify an iteration.
It is not part of every cycle.

## Measured cost basis for the request

| | |
|---|---|
| Previous allocation | 5,000 GPU hours |
| Consumed at 2026-09-03 | 58.48 kCPU-hours, which is 58%, roughly 2,924 GPU hours |
| Run rate | approximately 40 GPU hours per day over 73 days |
| Projected at expiry | approximately 74% |

At the same run rate, a three-month period costs approximately 3,600 GPU hours.

Extraction lane cost per iteration:

| | |
|---|---|
| Fine-tune | 4 GPUs x 18 h = 72 GPU hours (measured 13.1 h, so typically ~52) |
| Full evaluation | approximately 25 GPU hours |
| Inference batch | 2 GPUs x 1 h = 2 GPU hours per batch |
| Five iterations per quarter | approximately 485 GPU hours |

The request of 5,000 GPU hours covers the measured modelling run rate with
headroom for the extraction lane. It is based on observed consumption rather than
on a projected increase.

## Storage and transfer basis

| | |
|---|---|
| Working set | 1.4 TB locally |
| Container images on GPFS | approximately 41 GB across four `.sif` files |
| One 27B bf16 checkpoint | approximately 54 GB before optimizer states |
| Result pull per synchronisation | approximately 61 GB |
| Synchronisation frequency | several times per week |

The previous request of 400 GB storage and 200 GB transfer was set before the
modelling components were operational and proved too low in practice.

## Failure modes handled

| Failure | Handling |
|---|---|
| Wall-clock kill or preemption during the fine-tune | `--requeue` plus resume from the latest scratch checkpoint |
| Out-of-memory during training | sequence-window rebuild; diagnosed and fixed during the previous allocation |
| Out-of-memory during inference | explicit per-device memory cap replacing `device_map` strings; diagnosed and fixed during the previous allocation |
| Duplicate OpenMP runtime fault | one OpenMP runtime per rank, set explicitly |
| `srun` receiving 1 CPU instead of the requested count | `--cpus-per-task` passed explicitly to `srun`, not inherited |
| Container or driver mismatch | `torch.cuda.is_available()` asserted at job start, fails loudly |
| Corpus growth changing the family count | sweep and tune arrays auto-discover families from the dataset directory rather than a hardcoded list |
