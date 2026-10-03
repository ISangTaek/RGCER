# V10-S1 implementation and frozen comparison

This is a method-screening implementation, not a result or a claim of superiority.
The ten entries below are distinct architecture families. Learning rates, controls,
ablations and seeds are not counted as additional methods. All code is a new
PyTorch implementation from the described constructions; no external model
weights or copied third-party source are bundled.

## Method correspondence

| ID / code | Primary method source | Preserved construction | Molecular adaptation and limits |
|---|---|---|---|
| F01 / `LAMEL` | [LAMeL, Digital Discovery 2026](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00443h), [full method](https://arxiv.org/html/2509.13527v1) | Source prediction mean, centered prediction-space ridge, molecular residual ridge | Frozen nonlinear source heads replace the original linear/graphlet models. Labels use each task's training scale. The residual is not constrained to be orthogonal. This analytic method uses the complete pretrained source bank; it does not optimize a new cross-task network. |
| F02 / `MODERNNCA` | [ModernNCA, ICLR 2025](https://openreview.net/pdf?id=JytL2MrlLT) | Nonlinear embedding; negative squared distance softmax; explicit weighted neighbor labels | Shared task-conditioned MLP on graph/function features. Stochastic same-task candidate subset: half the eligible pool, bounded at 128; full legal target train context at validation. No PLR numerical embedding or paper-scale HPO. Group exclusion is stronger than excluding only self. |
| F03 / `TABR` | [TabR, ICLR 2024](https://arxiv.org/html/2307.14338v2) | Key-only squared distance, top-96 retrieval, value = label embedding + key-difference correction, residual feed-forward prediction | TabR-S-like linear encoder and small predictor, dropout 0, shared task metadata and bounded training context. Original retrieval equation 5 is retained. This is a multitask adaptation, not an official benchmark reproduction. |
| F04 / `TABM` | [TabM, ICLR 2025](https://arxiv.org/html/2410.24210v2) | Shared weight matrices and rank-one branch factors, 32 branches, independent output heads, mean of individual training losses, mean prediction | Two width-64 layers, no numerical feature embedding, common task-conditioned inputs. It is not a 32-seed ensemble. All source/target tasks update the same network. |
| F05 / `INP` | [Informed Neural Processes, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/2679c5d2e32b545eb0c5aa4cd626096e-Paper-Conference.pdf) | Context set encoder plus knowledge encoder, fused Gaussian global latent, context prior / context-plus-query posterior ELBO | Known task-name population, route, endpoint and dataset domain replace textual knowledge. These fields are factual metadata; no biological mechanism or theorem of improvement is inferred from them. |
| F06 / `DNP` | [Distance-informed Neural Processes, NeurIPS 2025](https://arxiv.org/html/2508.18903v1) | Global and local Gaussian latents, Laplace distance attention, KL terms, two-sided singular-value penalty | Local variance follows literal equation 6 via log-sum-exp; no claim that its far-distance limit is standard normal. Exact SVD replaces iterative singular-value approximation. Width reduction has a null space, so no global bi-Lipschitz guarantee is claimed. LeakyReLU .1; singular-value interval [.1, 2], penalty .01. |
| F07 / `BSA_TNP` | [BSA-TNP, AISTATS 2026](https://proceedings.mlr.press/v300/jenson26a.html), [architecture](https://arxiv.org/html/2506.09163v2) | Three KRBlocks; context attends context and queries attend context using shared subnetworks; learned distance bias | Single-head dense attention on bounded context, with squared distance on standardized graph/function coordinates. No spatial invariance claim or blocked-attention memory-performance claim. It is a molecular KRBlock adaptation. |
| F08 / `KERNELICL` | [KernelICL, arXiv 2026-02](https://arxiv.org/html/2602.02162v1) | Context-dependent symmetric embeddings: both keys and queries pass through the same label-free query route; explicit Gaussian label kernel | Regression MSE, two shared attention blocks, no original TabICL column/row foundation pretraining or classification objective. It is a KernelICL-inspired regression model, not pretrained TabICL replication. |
| F09 / `DANP` | [Dimension Agnostic Neural Processes, ICLR 2025](https://arxiv.org/html/2502.20661v1) | Scalar feature/label tokens, distinct sinusoidal positions, dimension attention and pooling; masked context transformer plus latent path | Molecular/function coordinates and metadata form the dimensions; one output label. Two KRBlocks implement the context mask. Each scene is trained separately, so cross-scene dimension-transfer superiority is not tested. |
| F10 / `FCR` | Proposed in this project; shares prediction-space motivation with LAMeL | Source functions and molecular features, label-conditioned feature map, differentiable centered ridge adaptation | Source-task episodes hide their own source head. Same shared architecture across every source and target task within a scene. Novelty and empirical benefit remain unestablished. |

The source functions are deterministic transforms of frozen graph features. They
add a learned coordinate system and prior, not independent molecular measurements.
Source pretraining has already seen source train labels. Masking a source task's
own head does not turn the source episodes into unseen-task validation.

## Common inputs and training

`v10_feature_cache.py` verifies the existing source loader, graph identities,
train-only scalers and the original scene population. It extracts raw graph
features and central source-head outputs once per scene. All methods share this
cache. The source function columns remain in source train-scaled coordinates;
graph coordinates are standardized over all source and target train observations
(observations, not unique molecules). The domain and task metadata are encoded
from task names, with no fabricated species taxonomy. Cache column order and
metadata vocabulary are recorded.

ToxAcute uses Animal56 + Human3; A uses Nonhuman104 + Human5; B uses Animal56 +
external-only Human5 with the existing all-Tox-partition canonical exclusion.
No target test inference is available in the V10 runner. Context labels are
same-task train labels only. Sample IDs, canonical molecules and groups must be
disjoint between context and query. Query labels have a separate training-loss
argument and are rejected for validation queries.

Seed 42, 40 full target epochs, query batch 16, context cap 128, hidden width 64,
AdamW weight decay 1e-5 and gradient clipping 1. Two predeclared learning rates
are configuration 0 = 1e-3 and configuration 1 = 3e-4. A source task contributes
16 rotating query observations per epoch (or its entire smaller population),
with group-safe subdivision if necessary. Every target observation appears once
as a query per epoch. Source episodes are distributed across target updates;
the loss at an update is target loss plus the mean of its assigned source losses.
If an update has no assigned source episode, it has target loss only.
All source tasks participate every epoch, but source observations are not fully
traversed every epoch. Actual context/query identities and coverage are saved.

The analytic methods fit every target task using all its legal train rows. The
two LAMeL penalty pairs (source, residual) are (.1, 1) and (1, 10); ordinary ridge
penalties are .1 and 1. The definition is mean MSE plus penalty times squared
coefficient norm, with an unpenalized intercept.

MSE is used for F02/F03/F04/F08/F10 and the learned controls. F05/F06/F09 use
Gaussian NLL with the context/union-posterior KL; F07 uses Gaussian NLL. In latent
models, prediction integrates decoder means with 16 fixed antithetic prior
samples. Local draws are keyed to sample identity, and global draws to task, so
predictions are invariant to query order and batch partition. Training posterior
labels are never used by prediction. These uncertainty models are screened on
point RMSE; no calibration advantage is claimed.

Four controls are `TARGET_MLP` (same graph/functions, MSE, target episodes only),
`FCR_NO_FUNCTIONS` (zero functions, identical dimensions/capacity, all episodes),
`FCR_TARGET_ONLY` (same bank and FCR, target episodes only), and `RIDGE` (graph
features only, target train labels). Source/encoder assets are inherited by all
controls; these controls cannot establish the value of source pretraining itself.

There are 84 scene-method-configuration jobs: 72 neural trajectories / 2,880
model epochs plus 12 analytic jobs. Analytic jobs fit 52 task-specific regressors
in total. They are not counted as neural epochs. No cost calibration or separate
discarded smoke stage is inserted. Finite loss, gradients and parameters are
checked in the actual training trajectory.

## Selection, verification and handoff

Within each configuration choose the first minimum validation macro-RMSE epoch;
then compare two configurations and keep configuration 0 on exact ties. This is
development selection on a reused validation set. Old seed-42 comparators from
066/077/084 are bound by file hashes and re-scored on the identical population.
The strongest old result and strongest matched control are both required. A
single candidate must beat both in all three scenes to enter bounded confirmation
review; selecting a different method in each scene is not a unified method.
Report every endpoint and descriptive paired group-bootstrap intervals. No
automatic extra seeds, test inference, per-scene architecture switch or HPO round.

Independent CPU verification rebuilds every target train/validation graph feature
and two fixed random source rows per source task. It checks the entire cache file
SHA, source identities and feature statistics. This is not a full raw-graph
re-extraction of all source observations. Every selected checkpoint is replayed
using freshly extracted target inputs. Every saved epoch's metric and full
sampling schedule is independently checked; training itself is not repeated.
Cache graph comparison uses rtol 1e-3 / atol 2e-5; selected prediction comparison
uses rtol 1e-4 / atol 2e-5. Numerical failure is reported, not silently retried.

Execution requires an exact published commit, separate WSL and server code gates,
the trusted existing data/source assets, and empty output/attempt locations.
Use the separately issued GLM card for WSL -> trusted Paramiko -> server -> SFTP.
This document alone is not a READY card and does not authorize a server run.
