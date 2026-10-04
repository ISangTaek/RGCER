# D2-S1: initialization, backbone plasticity and joint supervision

This stage tests a bounded scientific question using one existing Graphormer.
It is a controlled training comparison, not a new method or a claim that any
architecture will dominate the previous models. Test and calibration inference
are unavailable in this runner. All selection uses the existing development
validation populations, which are already exposed through earlier work.

| Condition | Encoder initialization | Encoder updates | Training labels |
| --- | --- | --- | --- |
| FJ | Reviewed source seed 42 | Frozen | Source and target |
| PT | Same reviewed source | Trainable | Target only |
| PJ | Same reviewed source | Trainable | Source and target |
| RT | Fresh random construction | Trainable | Target only |
| RJ | Same fresh random construction | Trainable | Source and target |

All five conditions have the same independent per-task readout: LayerNorm,
96-to-96 linear, GELU, dropout 0.1 and a scalar linear output. All source and
target heads are newly initialized using a seed derived from the task name.
Historical source heads are only inspected by the source-data identity checker;
they are never copied into a D2 model. Target-only source heads receive no
gradient, optimizer state or updates. Random encoders are constructed directly;
they never load a source tensor. The eight-layer, four-head, width-96 molecular
Graphormer and its graph preprocessing match the bound historical architecture.

Dropout remains active in the encoder during training in **all five conditions**,
including FJ. Freezing changes `requires_grad`, not stochastic forward behavior.
All evaluation uses `eval()`. Target and source dropout have separate per-update
RNG streams; adding a source forward does not advance the target stream.

## Bounded S1 matrix and training clock

Three settings (ToxAcute, PubChem A and PubChem B), seed 42 only. FJ has one
configuration; each of PT/PJ/RT/RJ has encoder LR 0.0001 and 0.001. This is **27
trajectories**, including all controls. Head LR is always 0.001. AdamW uses
weight decay 0.00001, default betas/epsilon, constant LR and global gradient
clipping at norm 1. No warmup, early stopping, scheduler, router or residual
adapter is added. Every condition uses standardized-label MSE and train-only
task scalers. Predictions are returned to the original task label scale.

Source and target batch sizes are both 32. Within a source pass each legal
source observation appears exactly once, short tails retained. Source task
batches are shuffled together. Target tasks independently follow complete
shuffled passes, cycling as necessary. Each joint update sums one target-batch
mean MSE and one source-batch mean MSE (source weight 1), then takes one shared
optimizer step. PT/RT take the same target batch at the same update index but
do not fetch a source graph or label. Source counts from an authenticated
checkpoint provide their count-only clock; source datasets/scalers are not
constructed or fitted for PT/RT. The underlying CSV loader may validate whole
file bytes, but no source labels enter PT/RT loss or selection.

Each trajectory has `40 * sum(ceil(source_task_count / 32))` optimizer updates.
Joint runs complete 40 source passes; target-only runs complete zero source
passes but exactly match joint target exposures and update counts. Target data
are therefore repeatedly sampled. This explicitly tests a balanced two-stream
recipe; it is not equivalent to one target epoch per source epoch. Target
sampling remains proportional across observed target-task batches. The last
target pass may be partial; its exact counts are reported.

| Setting | Target tasks / train observations | Validation observations | Source tasks / train observations |
| --- | ---: | ---: | ---: |
| ToxAcute | 3 / 262 | 39 | 56 / 78,063 |
| PubChem A | 5 / 407 | 133 | 104 / 75,872 |
| PubChem B | 5 / 261 | 83 | 56 / 78,063 |

All three data/asset preflights run before the first GPU update. Exact historical
file, population and encoder identities are bound by `p1d4_identity_lock.json`
and the existing source lock. Route A uses its nonhuman104 source; ToxAcute and B
use Animal56. Route B retains the original cross-database exact-canonical
exclusion, without inventing a cross-database scaffold-disjoint claim.
RDKit must be the existing 2025.09.6 build; the runner records PyTorch, CUDA,
RDKit and device versions without installing or upgrading anything.

The source history before D2 is **40 epochs for FJ/PT/PJ, zero for RT/RJ**. Those
historical exposures are not made equal by matching the D2 clock. D2 also replaces
historical quantile heads/loss with a common scalar head/MSE; the five within-D2
contrasts control this change, but a comparison to historical RPT is not a
single-factor intervention.

## Selection, verification and interpretation

Validation is evaluated at updates 1,2,4,8,16,32,64,128,256,512, then every 1,024
updates, at every complete source-clock pass and at the final update, after
deduplication. Early checkpoints protect PT/RT against missing an optimum long
before the first complete source pass. All conditions have the same selection
opportunities. The earliest minimum endpoint-macro RMSE wins within each run.
Step zero is not a selectable checkpoint. The source-pass training-loss curve,
validation predictions and all endpoint results are retained.

Report PJ-FJ, RJ-PJ, PJ-PT and RJ-RT **separately within each LR stratum**; the
same FJ serves both strata. Do not choose a different LR/condition per scene and
present the resulting collage as a unified architecture. S1 does not automatically
choose replication configurations. Codex reviews both LR strata, convergence,
endpoint harms and uncertainty before freezing the four remaining seeds 43-46.
Single-seed S1 is not confirmation, and extra seeds do not create unexposed data.

The first two real updates of each trajectory are its smoke check. Nonzero
gradient roles, encoder/head changes and frozen behavior must pass before the
trajectory continues. These updates are retained, not discarded extra trials.
Each finished run is then checked in a fresh CPU process: input identity,
training schedule, exact exposure counts, every validation metric, selected
checkpoint, frozen and inactive tensors, optimizer configuration/moments/step
counts, and selected-checkpoint prediction replay (absolute tolerance 0.0002).
CPU/GPU numerical agreement is scoped to that stated tolerance, not bit identity.

If training is still improving at the boundary, the negative result is
**convergence unresolved under this bounded recipe**, not proof that joint
learning or random initialization cannot work. No automatic extension is
permitted. If a well-trained RJ does not reproducibly improve on PJ/PT, end the
unified-from-scratch hypothesis instead of adding more upper modules. Positive
development results require paired replication and comparison with the locked
strong historical methods before becoming a method claim.

The released CLI owns at most four independent single-GPU processes on idle
physical GPUs 0-3. It does not change installations, connect to a server, kill
other jobs, retry failures or resume attempts. A failed child stops new dispatch;
already-running jobs finish and retain their facts. Failed output is packaged as
partial. `CONTENT_PASS` only means verified content, never scientific acceptance.

The transfer ZIP contains JSON, logs and JUnit XML with a SHA manifest. Best and
last tensor checkpoints stay on the server and are identified by SHA in every
run receipt. WSL and server code tests are separate; neither substitutes for the
server's real GPU/asset checks. Execution requires the current complete GLM card.
