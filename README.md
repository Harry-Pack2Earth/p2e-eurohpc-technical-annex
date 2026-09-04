# Pack2Earth Formulation Modelling System: technical annex

Public technical annex supporting a EuroHPC AI Factories access application for
time on **MareNostrum 5 ACC** (BSC).

This annex exists so that a technical assessor can evaluate the proposal without
requesting access to anything. It contains the actual SLURM submission scripts and
Apptainer container definitions we run on MareNostrum 5, a pinned software
manifest, and the execution plan.

The full source repository is private because it holds formulation data relating
to patented and client-confidential material. Read access can be granted on
request to harry@pack2earth.com. Nothing needed for technical assessment is
withheld here.

**Applicant:** Pack2Earth SL, Barcelona, Spain (SME)
**Allocation:** `ehpc838`, MareNostrum 5 ACC
**Previous application:** EHPC-AIF-2026PG01-174

---

## What the system does

Pack2Earth develops home-compostable plastic formulations. The system reduces
laboratory time by proposing a small number of high-quality candidate
formulations for physical validation, instead of broad trial-and-error screening.

Four components, all currently operational:

1. **Data extraction and materials datastore.** Literature, supplier datasheets
   and internal laboratory results parsed into a typed datastore with full
   provenance per datapoint. Currently 1,965 papers, 86,363 observations,
   16,983 formulations.
2. **In-house extraction model.** An open-weights 27B language model fine-tuned
   on MareNostrum 5 to perform the extraction in component 1.
3. **Property prediction models.** 98 encoded property families, 1,177 tracked
   model runs, validated by grouped cross-validation with the source paper as
   grouping key.
4. **Multi-objective optimisation and active learning.** BoTorch expected
   hypervolume improvement over the joint formulation space, with laboratory
   results fed back to trigger retraining.

---

## Repository contents

```
slurm/       every SLURM submission script we run on MareNostrum 5 ACC
container/   Apptainer definition files, base images pinned by tag
SOFTWARE-MANIFEST.md   versions, and how the offline build works
EXECUTION-PLAN.md      what we run, in what order, at what measured cost
```

---

## Hardware compatibility

Verified in production over 359 completed jobs on this allocation, not assumed.

| | |
|---|---|
| Machine | MareNostrum 5 ACC |
| QoS | `acc_ehpc` (the only valid QoS on an ACC-only allocation; `gp_ehpc` is rejected at submit) |
| Node shape | 4 x NVIDIA H100, 80 CPUs |
| CPUs per GPU | 20, which is how every script derives `--cpus-per-task` |
| Driver | 595.71.05, supporting CUDA 13.0 and below |
| Container runtime | Apptainer with `--nv` |

Every GPU job asserts `torch.cuda.is_available()` and prints an `nvidia-smi`
banner into its job log before doing any work, so a driver or container
mismatch fails loudly at job start rather than silently falling back to CPU.

---

## Constraints this design accounts for

**No outbound internet on compute or login nodes.** Container images are built
off-cluster and transferred as `.sif` files. Python packages not baked into an
image are supplied as a pre-built offline wheel bundle staged to GPFS. Model
weights are staged read-only under `/gpfs/projects/ehpc838/models/`.

**No fakeroot for image building on the cluster.** All images are built locally
or via a remote builder, then rsynced in.

**Wall-clock limits and preemption.** The long fine-tune runs with `--requeue`
and resumes from the most recent checkpoint, so an interruption costs one
checkpoint interval rather than the run.

**Two lanes sharing one allocation.** The extraction and modelling workloads use
separate scratch namespaces, separate container images and separate job names, so
neither can overwrite the other's artefacts.

---

## Licence and contact

Scripts and definitions in this annex are published for technical assessment.

Harry Jones, AI Lead, Pack2Earth SL: harry@pack2earth.com
