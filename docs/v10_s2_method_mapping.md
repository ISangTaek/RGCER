# V10-S2 dual-context screen

This document freezes implementations, not performance or novelty claims. S2
compares ten architecture candidates: six accepted S1 results are reused and
four new information flows are trained. Learning rates, controls and prediction
averages are not counted as architecture families. S1 code remains unchanged.

## Evidence and method correspondence

The design follows two constructions: [TabR](https://arxiv.org/html/2307.14338v2)
provides distance-weighted label/value retrieval with a key-difference correction;
[BSA-TNP, AISTATS 2026](https://proceedings.mlr.press/v300/jenson26a.html)
provides shared context/context and query/context KRBlocks. Their combination
here is a project hypothesis, not an implementation or theorem attributed to
either paper. The molecular port uses dense bounded-context attention and has
no claimed spatial invariance or Biased Scan Attention scaling advantage.

The accepted S1 implementations of ModernNCA, TabR, TabM, INP, BSA-TNP and DANP
are retained without retraining. Their differences from the papers remain in
[the S1 mapping](v10_method_mapping.md). All fourteen S1 methods, all 114 older
reference comparisons and four fixed prediction mixtures remain in the S2
reference scoreboard, including the methods outside the retained six.

No inference about universal superiority or paper-level novelty follows from
the S1 single-seed development results. The terms "retrieval" and "relation"
describe information flow. In S1, top-96 includes all eligible training context
in ToxAcute and B, so a local-versus-global mechanism has not been established.

## Frozen inputs and equations

For a task, let `x=[h,F,mask,metadata]`, with frozen graph feature `h`, frozen
source central-head outputs `F`, own-source-head mask and factual task fields.
Only source and target training rows fit the existing graph-feature statistics;
every label scaler is task-train fitted. All are reused byte-for-byte from 085.
The source functions are deterministic transforms of graph features, not new
independent molecular measurements. Each scenario has its own trained weights;
all source and target tasks within that scenario share the same model.

The shared projection is `u=Linear(x,64)`. The retrieval module uses
`k=Linear(u,64)`, selects up to 96 eligible context rows, and returns

```
r(q) = sum_i softmax(-||k(q)-k(i)||^2)_i *
       [Linear(y_i,64) + Linear(ReLU(Linear(k(q)-k(i),64)),64,bias=False)]
```

Context rows are sorted by task/canonical/group/sample identity before a stable
distance sort. This makes exact cutoff ties independent of input permutation.
Blocked context-internal pairs receive zero weight. If a context row has no
eligible internal neighbors, its retrieval residual is exactly zero and the
event is recorded. Query/context identity exclusion is never relaxed.

The relation module embeds `[u_c,y_c,1]` and `[u_q,0,0]` with a shared linear
map and applies three width-64 KRBlocks. Each block uses context as keys and
values for both its context and query updates. It has learned squared-distance
bias computed in the original standardized/masked input coordinates. There is
no query/query attention. Labels of query observations never enter prediction.

| ID | Implementation | Information flow |
|---|---|---|
| C07 | `DCR_PARALLEL` | Retrieval and relation run from the same projection; one `MLP([u_q,r_q,a_q],64,1)` predicts the value. |
| C08 | `DCR_CONTEXT_FIRST` | Relation updates context and query states first; retrieval keys and correction values then use those states. The single decoder receives `[u_q,r_q,a_q]`. |
| C09 | `DCR_RETRIEVAL_FIRST` | Retrieval enriches both context and query projections before the relation module. Context-internal retrieval excludes the same sample, canonical molecule or group. Decoder receives `[u_q,r_q,a_q]`. |
| C10 | `DCR_CONTEXT_METRIC` | A symmetric mean of `MLP([u_c,y_c],64,64)` produces task-context summary `z`; `softplus(Linear(z,64))+1e-4` supplies positive metric factors. Retrieval distance is `||s*(k_q-k_c)||^2/64`. One decoder receives `[u_q,z,r_q]`. |

C07-C09 share the same parameter schema but have distinct tested information
flows. C10 has no KRBlocks and conditions the retrieval metric directly. None
routes by scenario, switches to an old model according to validation labels,
uses per-scenario mixture coefficients, or merges pretrained parameter vectors.
All four use MSE and the same episode schedule.

## Additional controls

`BSA_MSE` has exactly the S1 BSA-TNP module and state schema, including its scale
output, but trains the predictive mean with MSE instead of Gaussian NLL. No
calibration claim is made for its unused scale output.

`JOINT_MLP_CAP` uses the same `x`, source/target episode schedule and MSE, with
two equal-width SiLU hidden layers and no context-label path. For input dimension
`d` and main hidden size `h`, C07 has `d*h+31*h^2+43*h+4` parameters, including
three scalar KRBlock bandwidths. The control has `w^2+(d+3)*w+1` parameters;
choose the nearest integer width analytically (smaller width on exact tie).
The difference must be at most 5%; there is no validation-based width search.

The actual frozen input has 96 graph coordinates. For ToxAcute/A/B, C07-C09
have 144772/151556/144900 parameters; the capacity control widths are 280/254/279
with 145041/151893/144802 parameters. C10 has 52609/59393/52737 parameters. The
capacity match is to C07-C09, not C10; a future mechanism claim for C10 requires
its own suitable capacity comparison.

The four untrained prediction controls are equal means of TabR+BSA-TNP,
TabR+DANP, BSA-TNP+DANP and all three. They use the accepted S1 selected
checkpoints/predictions in original task label units. Pairing is by full sample
identity, not file order. No coefficients or component configurations are tuned
using S2 outcomes. Their post-hoc origin in S1 is preserved explicitly.

## Matrix, selection and reference ownership

Four new architectures and two new controls, three scenarios and two preset
learning rates give 36 new neural trajectories. Each trains 40 full target
epochs at seed42, hidden64, query16, context128, AdamW decay1e-5 and clip1.
Each source task contributes 16 rotating query observations per epoch; target
loss plus the average assigned source losses is used at a step. All source tasks
participate, but source observations are not all traversed every epoch. Actual
coverage is saved. No real warmup/cost-only run or new source pretraining is added.

The matrix is 1440 model epochs / 30240 optimizer updates: 720/1120/680 updates
per ToxAcute/A/B trajectory. Source cache producer commit is the immutable
`9111793be3adf87ca335be8f44aa86065e38427e`, distinct from the S2 producer commit.
`configs/v10_s2_reference_lock.json` binds the accepted 085 checksum manifest,
all 6385 original content files at import, a compact 239-file snapshot, and the
unchanged S1 implementation (source/config text hashes normalize CRLF to LF;
all experiment asset hashes remain byte-exact). The compact snapshot includes three original
caches, all 84 selected prediction/receipt pairs and older reference subsets;
omitted S1 epoch histories/checkpoints remain in the immutable original archive.
The `unpack-reference` entry point first verifies the accepted inner ZIP SHA,
rejects duplicate/unsafe/link members, and extracts into a new directory before
checking all original file hashes. The prior raw run directory is not used as
an archive substitute: its package-only checksum manifest may be absent.

Select the earliest minimum validation macro-RMSE within each configuration,
then the better of the two configurations (configuration0 on exact tie). Every
new architecture must beat the strongest frozen reference and the strongest
new control separately in all three scenarios to become a development candidate.
If several pass, rank by their worst-scenario relative improvement, then mean
relative improvement, then summed parameter count across scenarios, then the
declared C07-C10 order. Do not construct a winner by switching architectures
between scenarios. Failure stops this fixed branch; success requires review.

All endpoints and paired group-bootstrap intervals are retained. Bootstrap
intervals describe reused development data and are not independent confirmation.
No S2 test inference or automatic extra seeds/ablations are authorized by this
code. Model performance and scientific acceptance remain pending real results.

## Validation and execution boundary

Tests cover train-only query-loss guards, source own-head masking, context-label
effects, empty-neighbor handling, independent retrieval equations, gradients,
context/query permutation and query chunk invariance, exact distance ties,
architecture distinctions, capacity arithmetic, identity-bound reference reuse,
population pairing, complete training/checkpoint replay, tampered receipts and
optimizers, ranking and partial-failure packaging. Synthetic graph tests exercise
the actual Graphormer interface without real datasets or teacher weights.

Server verification uses three independent CPU processes. Each re-extracts all
target train/validation graph features plus two fixed random source rows per
task, then replays every selected S2 checkpoint and checks every saved epoch and
schedule. Source raw-graph replay is sampled, while file hashes and row identities
are complete. Frozen graph tolerances are rtol1e-3/atol2e-5; selected prediction
tolerances are rtol1e-4/atol2e-5. No retry with looser tolerances is permitted.

Use the separate exact-commit GLM card: current WSL validation, trusted Paramiko
server synchronization/validation, bounded experiment, independent replay, SFTP
SHA-checked return. This document alone is not a READY execution card.
