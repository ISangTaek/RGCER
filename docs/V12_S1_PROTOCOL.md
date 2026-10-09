# V12 S1 analytic and relationship mechanism protocol

This stage implements C3 and a prerequisite relationship test for C1. It is not
the subsequent ten-method architecture screen and cannot establish a unified
winner. The source graph networks are reused, not retrained. Real fits run on
the experiment server; workstation and WSL tests use synthetic inputs.

## Frozen inputs and qualification

`configs/v12_s1_input_lock.json` binds 25 existing 087 files, including the
accepted manifests, raw function/chemical caches, source/target train records,
target validation identities, and the current historical references. No test or
calibration inputs are loaded. Wrong hashes, paths, populations, unknown tasks,
duplicate cells and target train/validation group overlap fail closed.

The existing teachers were fit on their complete source train sets. They do not
qualify as strict chemical-group out-of-fold teachers. C3's target-label CV is
conditional on these fixed source assets; it is not unseen-chemistry evidence.
C1's first relation test uses observed source labels, with both molecular ends
held out of the fitted relation. It does not use or validate source predictions.
No additional teacher training is needed for this limited stage.

The data adapters preserve published log-dose values and task identities. This
stage does not reverse-transform units or certify record-level exposure/assay
semantics. C1's first test requires the same route and endpoint within one bank;
cross-endpoint pairs and C4 remain unqualified pending semantic evidence.

## C3 and matched controls

The inputs are eight source-function principal coordinates and 80 chemical
coordinates: a fixed seed-42 Rademacher projection of the 2048-bit Morgan vector
to 32 dimensions, 24 descriptors, and 24 missing indicators. Descriptor medians,
PCA, centering and scaling use unique source train molecules only. Inherited V11
statistics using all target train observations are not reused. Every source task
fits standardized labels against the source coordinates, with ridge precision
10. Task coefficients estimate an equally task-weighted mean/covariance. The
covariance has 50% diagonal shrinkage and a 0.01 diagonal floor. The source PCA
dimension is bounded by available rank for synthetic fixtures, but is eight in
the production banks.

C3 jointly fits source and structure coefficients with an unpenalized intercept.
The source block uses the empirical prior; structure coefficients have zero mean
and tenfold diagonal precision. Two fixed multipliers are 1 and 10. This is a
conditional empirical-Bayes posterior mean, not calibrated uncertainty including
the uncertainty of the estimated prior.

Five matched conditions: C3, same-input ridge, same-input RBF kernel ridge,
C3 with zero prior mean, and C3 without the chemical residual. The zero-mean
ablation removes the prior mean only, retaining its covariance. These controls
and configurations are not counted as new architecture families.

Three deterministic structure-group folds use the first 16 hexadecimal digits
of SHA256(`V12S1/42/` + group), modulo three. All target tasks remain. Labels and
scales for a fold fit use its target training portion only. Configuration is
selected from pooled out-of-fold target **training** predictions by endpoint-macro
RMSE, with the first configuration winning exact ties. This CV score is a tuning
score, not an unbiased nested outer evaluation. The selected models are then
reported on the old development validation population; validation scores cannot
choose configurations. Both configurations remain in the returned evidence.

Each scene has 5 conditions × 2 configurations × (3 folds + 1 full train fit).
Across Human3 + Human5 + Human5 this is **520 small endpoint fits**, plus **216
source-task prior regressions**. This is not 736 neural training trajectories.
All three source banks and every legal target train row participate. No epochs,
new seeds, backbone optimization, test access or GPU use are authorized here.

## C1 observed relation diagnostic

Evaluate both directions of each qualified source-task pair in three chemical
group folds. A pair needs at least 12 common molecules and six groups, with all
three folds having at least six train molecules, three held molecules and two
held groups. Raw descriptor availability is also required. The returned
eligibility table includes every excluded pair and unrepresented source task.
These exclusions do not remove tasks from C3 training.

Read-only qualification of the locked inputs gives **74 eligible task pairs /
444 directed fold cases / 41 represented source tasks** for Animal56 and **88 /
528 / 64** for Nonhuman104. These counts are checked before fitting. The remaining
source tasks still participate in C3's complete prior estimation. Across both
banks there are 972 relation cases with three small slope fits each; B reuses
the Animal56 diagnostic only after source labels and raw chemistry match.

Select up to 128 train and 64 held molecules deterministically by hashed
canonical identity. Ring offsets 1 and 7 give each molecule at most four pair
appearances. Pair weights allocate total likelihood mass equal to the number of
unique groups. Standardize each task using the selected train molecules only.
Each four-cell record uses two real molecule labels at each of two source tasks;
both ends of every held pair belong to the held fold.

Fit a zero-intercept global slope, a chemical-conditioned slope, and a matched
shuffled-condition control. Symmetric conditions are Morgan Tanimoto similarity,
mean MolLogP, mean TPSA and absolute molecular-weight change. Condition statistics
use train pairs only; ridge precision is one. Multiplying a symmetric condition
by the source difference preserves zero differences and exchange antisymmetry.

The diagnostic's held source labels would not be available in deployment. Scores
therefore answer whether an observed conditional relation exists, not whether
a C1 model can predict Human3/Human5 or transfer to B. Pair/fold/task repetitions
are dependent and must not be treated as independent samples. Preserve individual
pair identities and predictions for grouped sensitivity review; a mean across
task-pair folds is descriptive, not a significance test. Tox and B share the same
source bank; labels and raw chemistry must match before reusing its diagnostic.

## Commands and independent verification

`python v12_s1.py matrix` prints the bounded protocol. `preflight` binds the full
execution commit, code files and 087 inputs before any fit. `run` requires a new
output directory and uses one CPU numerical thread. `verify` in a fresh process
recomputes all fits, reloads saved model coefficients, repeats predictions and
selection, regenerates relation cases, and checks against the frozen inputs.
Its receipt must be outside the immutable run directory. Updating a checksum
manifest cannot make a wrong sample, label or fitted model pass replay.

Code tests: `tests/test_v12_analytic.py`, `tests/test_v12_relation.py`,
`tests/test_v12_s1.py`. The synthetic end-to-end test reduces task counts in a
fixture only; no CLI switch permits replacing the production matrix with a toy.

After replay, the receipt is `PASS_INDEPENDENT_CPU_REFIT_AND_REPLAY` with
`PENDING_CODEX_REVIEW` scientific status. It does not automatically advance a
candidate, certify C1 deployment, confirm a mechanism, or authorize V12-S2.
