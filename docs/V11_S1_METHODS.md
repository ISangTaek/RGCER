# V11-S1: chemical multi-view joint training

Status: implementation for a bounded development screen, not scientific acceptance.
The runnable entry point is `v11_s1_screen.py`; exact architecture and optimizer
values are `v11_models.CONFIGS`, mirrored in `configs/v11_s1_method_manifest.json`.
Upstream repository revisions and inspected source-file SHA-256 values are pinned
in that manifest. No upstream package, dataset or weight download is required by
the execution card. These are independent core adaptations, not reproductions of
nine complete author benchmarks or nine newly published 2026 methods.

## Input, training and scope

All ten candidates and the concatenation control see the same molecular input:
the accepted frozen 085 graph embedding `h`, source-function outputs, an explicit
own-head mask, Morgan 2048-bit radius-2 chirality fingerprints, 24 RDKit 2D
descriptors with 24 missingness indicators, and known task metadata. The graph
control deliberately sees only the graph/function view and task metadata.
RDKit is pinned to `2025.09.6` (`rdkit==2025.9.6` in the existing environment).
The exact descriptor list is `v11_features.NAMES`; molecules are parsed from the
accepted canonical SMILES without a new standardization step. Invalid molecules
fail before training. Nonfinite descriptors receive a training-median imputation
and a missingness bit. Entirely missing fields use zero and are recorded.
Median, mean, scale and RealMLP robust statistics use every legal source/target
training observation with its original multiplicity. Validation labels never enter
features; validation molecules never fit statistics. No test rows are accessed.

The flat order is `[h, masked F, mask, Morgan, standardized D, D_missing, metadata]`.
Token methods use every input dimension exactly once: h/F/mask blocks of at most
16, 32 Morgan blocks of 64, one descriptor-plus-missing token per descriptor, and
one token per task field (population, route, endpoint, domain). This blocking is a
declared adaptation from the authors' scalar tokenization and limits sequence
length without selecting or dropping fingerprint bits. Each block has its own
learned projection; ExcelFormer also has the corresponding multiplicative gate.

Each setting has one shared model across its complete source and target task
pool: ToxAcute 56+3, A 104+5, B 56+5. Each epoch covers every target training row
once and 16 rotating queries from every source task. Source losses are averaged
with total weight one per target batch. This does not traverse every source row
each epoch. Context excludes query sample IDs, canonical identities and groups.
All methods use standardized per-task labels, MSE, batch size 16, gradient norm
clip 1, and exactly 40 epochs. The two configurations use a method's declared
base learning rates and 0.3 times all of them. There is no third configuration,
cost probe, additional pretraining, source-task removal or fresh seed.

## Method-specific cores and adaptations

| ID / implementation | Retained core and formal configuration | Declared changes / exclusions |
|---|---|---|
| C01 `REALMLP_MV` | [RealMLP](https://arxiv.org/abs/2407.04491), published TD-S regression variant: three 256-unit layers, NTK linear scaling, normal hidden weights/biases, zero output initialization, Mish, learned input scaling, train median/IQR with range fallback, smooth clipping at 3; Adam-equivalent zero-decay AdamW, beta2 .95; lr .07, bias factor .1, input-scale factor 6; coslog4 schedule | This is **TD-S**, not full TD's periodic embeddings/parametric activation. Metadata is already one-hot; output normalization uses the frozen task scalers. Schedule progress spans this protocol's 40 epochs instead of the author's 256. |
| C02 `TABM_MV` | [TabM](https://arxiv.org/abs/2410.24210), k=32 BatchEnsemble, three width-512 ReLU blocks, shared dense weights, separate input/output fast weights and biases, first adapter random signs, subsequent ones, independent member output layers, member-wise MSE and mean prediction; dropout .1; AdamW lr .002, decay .0003 | No numerical embedding module; shared task batch across members; fixed multi-view train normalization; output biases initialized to zero. No author HPO/early stopping. |
| C03 `FT_TRANSFORMER_MV` | [FT-Transformer](https://arxiv.org/abs/2106.11959), width192, three blocks/eight heads, pre-LN except first attention, ReGLU FFN factor4/3, CLS readout; attention dropout .2, FFN .1; AdamW .0001/decay .00001 | Multi-view block tokenizer replaces scalar tokens; no KV compression. All tokens are computed in the final block (same CLS operation). |
| C04 `EXCELFORMER_MV` | [ExcelFormer](https://arxiv.org/abs/2301.02819), gated input projection, semi-permeable causal attention, attenuated QKV initialization scale .1, TangLU, learned token pooling; width192, three blocks/eight heads; AdamW .0001/decay .00001 | Fixed semantic block order, no validation-dependent ordering; no supervised feature-importance reordering or cross-example mixup. FFN dropout .1 is explicit in this adaptation. |
| C05 `AMFORMER_MV` | [AMFormer](https://arxiv.org/abs/2402.02334), separate additive and multiplicative prompt attention, log/exp product branch, learned branch/token pooling and residual GEGLU FFN; width192, three blocks/eight heads,32 prompts per branch; AdamW .0001/decay .00001 | Uses the author's dense-prompt option, not nondifferentiable top-k token gathering. Product rescaling is **within each row**, with denominator floor1e-6, instead of batch-global min/max, so another query cannot alter a prediction. Fixed32 prompts and a flattened regression head; dropout .1. Independent formula implementation, no upstream GPL source is redistributed. |
| C06 `T2G_FORMER_MV` | [T2G-Former](https://arxiv.org/abs/2211.16887), shared head/tail weight transform, diagonal relation weights, separate normalized column head/tail embeddings, straight-through hard topology, no diagonal edges, no readout key; FT-style width192/depth3/heads8; AdamW .0001/decay .00001 | Feature-block graph, not the molecular atom graph. No topology freezing schedule, KV compression or supervised token ordering. |
| C07 `MAMBULAR_MV` | [Mambular](https://arxiv.org/abs/2408.06291), width64, four RMSNorm residual Mamba blocks, expansion2, depthwise causal convolution4, state128, input-dependent delta/B/C, learned negative A and skip D, SiLU gating, average readout; AdamW .0001/decay .000001 | Uses the inspected native PyTorch sequential selective scan. No external CUDA extension, bidirectionality, feature shuffle or added interaction layer. Feature-block tokenization; no larger head network. |
| C08 `GRANDE_MV` | [GRANDE](https://arxiv.org/abs/2309.17130), inspected author's current regression core:1024 hard axis-aligned depth5 trees,50 sampled columns per tree, softmax/straight-through feature selection, softsign/straight-through thresholding, leaf predictions and instance-dependent expert weights; dropout .2; Adam rates weights/index/threshold/leaf=.001/.01/.05/.05 | Uses current regression implementation, not a claim to repeat the original classification benchmark. Fixed PyTorch seed42 feature subsets replace NumPy draws. No bootstrap/subsample or separate numeric embeddings. All four parameter groups update in one optimizer call; if all experts drop, keep the largest-weight expert (only material in tiny tests). |
| C09 `MODERNNCA_MV` | [ModernNCA](https://arxiv.org/abs/2407.03257), learned embedding128, one BN–Linear512–ReLU–Dropout.1–Linear128 block and final BN, Euclidean-distance softmax neighbor regression, train context sampling .5, temperature1; AdamW .01/decay .0002 | Explicit project configuration: one MLP block, no PLR numerical embeddings, differing from the inspected default's zero blocks plus PLR. Training BN jointly processes disjoint context/query features to support singleton queries. The bank contains only legal same-task training context; **query labels are never inserted into the bank**. Evaluation uses running BN and all legal training context. |
| C10 `TCF` | Task-conditioned chemical fusion hypothesis: three64-dimensional two-layer view encoders,64-dimensional task encoder, rank4 task-conditioned trilinear interactions for G–P/G–D/P–D,448→64→1 decoder; AdamW .001/decay .0001 | Low-rank interaction draws on [LMF](https://aclanthology.org/P18-1209/). Newness and usefulness remain unproven; the actual winner, not the name TCF, determines follow-up. |

These adaptations preserve distinct inductive biases. Their fixed finite screen
does not establish the performance ceiling of any original publication. Width,
optimizer, numeric preprocessing, tokenization and removed training strategies
are part of the interpretation of a negative or positive result.

## Controls, identity and independent verification

`MV_CONCAT_MLP` has a single hidden ReLU layer; its integer width is chosen from
the parameter-count formula closest to TCF. `TCF_GRAPH_ONLY` keeps the exact graph
and task encoders and adjusts only its readout hidden width. Both match TCF's
trainable parameter count within5%, use the same two learning rates and cannot
be tuned after results. Exact counts are recorded in every receipt.

The accepted086 inner ZIP and3356 content hashes are locked. A314-file necessary
snapshot preserves all085 references, historical results, fixed mixtures, six086
methods, and the frozen graph caches. The strongest old macro-RMSE thresholds are
ToxAcute=.8722816066441325, A=1.0447224161678876, B=1.0833168556201014. Old reference
metric checking permits only1e-12 floating-point reduction tails, not population
or identity changes. New metrics are independently recalculated from every
epoch's predictions; optimizer states, step counts, schedules, exact selected
epoch/configuration, checkpoint SHA and query/context identities are checked.

The three independent CPU verification processes validate the entire byte-locked
graph cache and independently replay all target graphs plus two fixed rows per
source task against that cache (the inherited085 audit scope). They regenerate
chemical features for **all unique canonical SMILES**, refit all training feature
statistics, and replay selected checkpoints on the exact frozen cache. Graph
audit and checkpoint replay are separate: small raw-graph CPU reduction tails
must not silently refit statistics or replace the features used to train a hard
tree. This does not claim a full raw-graph replay of every source observation.

Local tests cover all12 full configurations at the104-source shape, meaningful
core gradients, source/task coverage, label isolation, query batching invariance,
malformed caches, tampered optimizer/checkpoints, and the all-three-settings gate.
Additional local comparisons against hash-locked author definitions check selected
FT/Excel/T2G attention, Mamba scan, GRANDE regression and ModernNCA evaluation
forward/parameter gradients, plus RealMLP clipping/schedule. TabM is checked against
explicit independent member matrices. AMFormer's intentional row-normalization
change is checked for batch invariance and both prompt branches' gradients; its
full original forward is not claimed equivalent. No author training benchmark,
real-data performance, GPU experiment or scientific acceptance is implied.

## Selection and stop

Exactly72 trajectories = (10 candidates+2 controls)×3 settings×2 configs×seed42;
2880 epochs and60480 batch optimizer updates (720/1120/680 per trajectory).
Every candidate must strictly beat the strongest frozen reference and strongest
new control in **each** setting. Ties select the earliest epoch then config0.
Passing families rank by minimum relative gain, mean relative gain, summed
trainable parameters across settings, then declared C01–C10 order. Endpoint RMSEs
and paired group bootstrap differences are retained. Repeated development-set
selection is not corrected by this bootstrap and is not independent confirmation.
No passing family means review/stop, not automatic extra configurations, seeds or
reopening the ended V9/DCR branch. A passing family still requires Codex/user
review before a separate bounded confirmation/ablation plan.
