# MultiTask Toxic Prediction Reborn

Multitask molecular toxicology prediction framework built on Graphormer-style
graph transformers. The central experiment is **ToxAcute**: 59 toxicity
endpoints (56 animal LD50/LDLo endpoints across species × route, plus 3 human
oral TDLo endpoints), where animal source endpoints are transferred to human
target endpoints via cross-endpoint routing.

## Project Overview

Three architecture variants share one backbone and are selected via `--arch`:

| Architecture | Role |
|---|---|
| `Graphormer` | Base model (HPS) — direct multitask prediction, no transfer |
| `Graphormer_prompt` | Prompt-conditioned variant with `router_mode` (static/dynamic) |
| `Graphormer_rgcer` | **Response-Guided Cross-Endpoint Routing (RGCER)** — the main novel method; routes information from source endpoints based on their preliminary responses, molecule-conditioned top-k sparse routing, explicit NULL (no-transfer) route, FiLM + adapter conditioning |

Key design features:

- **Prediction heads** (`architecture/prediction_heads.py`): `point` or
  `quantile` mode. Quantile mode (default) regresses lower/upper quantiles
  (default 0.05/0.95) on **log-scale** labels; classification datasets are
  forced to `point` mode (see below). Predictions are decoded to mg/kg via
  `utils.calculate_mgkg`.
- **Conformal calibration** (`conformal.py`): CQR with signed conformity
  scores (`max(lower − y, y − upper)`), fitted per task on a dedicated
  calibration split; `qhat` widens/shrinks the base interval at inference.
- **Group-safe data splits** (`split_manifest.py`): one global
  scaffold-based (or random) train/validation/calibration/test manifest
  (default 0.70/0.10/0.10/0.10) shared across all tasks, so a molecule group
  never appears in two splits.
- **RGCER ablation flags** (`--rgcer_use_source_response`,
  `--rgcer_use_target_response`, `--rgcer_use_molecule_query`,
  `--rgcer_use_sparse_routing`, `--rgcer_use_null_route`,
  `--rgcer_use_film`, `--rgcer_use_adapter`, `--rgcer_use_base_aux_loss`,
  `--rgcer_fallback_space {prediction,representation}`,
  `--rgcer_transfer_mechanism {endpoint_router,response_stacking,target_only}`)
  — the last is the mutually exclusive transfer mechanism/baseline switch.
  RGCER flags are only honored by `Graphormer_rgcer`; other architectures
  warn and ignore non-default values.
- **ToxAcute task scopes** (`--toxacute_task_scope`): `human3` (3 human
  TDLo targets), `animal56` (56 animal sources), `all59` (default). Canonical
  task lists live in `architecture/toxacute_tasks.py`.

Supported datasets (`--dataset`): `toxacute` (default) plus classic Molecule-
Net tasks — regression: `qm7`, `esol`, `freesolv`, `lipophilicity`, `qm8`,
`qm9`; classification: `hiv`, `bace`, `bbbp`, `muv`, `tox21`, `sider`,
`clintox` (each with fixed task subsets defined in `main.task_names_for_params`).

## Tech Stack & Environment

- **Python 3.10.19** in the conda environment **`icl`** (per `requirements.txt`).
  The prebuilt `algos` Cython extension is **cp310-only**; rebuilding for
  another Python ABI requires `python setup.py build_ext --inplace`.
- The default `python` on this machine's PATH is Anaconda **3.13** — always
  run commands inside the `icl` env (e.g. `conda activate icl` or
  `conda run -n icl python ...`).
- PyTorch 2.10.0+cu126, PyTorch Geometric 2.8.0.post1, RDKit 2025.9.6,
  numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2, Cython 3.2.9, pytest 9.0.3.
- If a dependency is missing, **ask the user which conda environment to use**
  before installing anything (do not create/choose environments on your own).

## Repository Layout

```
main.py               CLI entry point (train/test/inference)
trainer.py            Trainer: task-wise training, warmup, conformal fit,
                      strict v4 checkpointing, routing aggregation (large file)
config.py             prepare_args(): maps CLI params -> arch_args + optimizer
experiment_config.py  Named experiment defaults (raw CSV path, preprocessed
                      dir, smoke/main run dirs, phase0 aliases)
dataset.py            PreprocessedDatasetWrapper (.pt files), DataCollator
                      (CPU-only padding), DataloaderWrapper (global splits)
preprocess_data.py    SMILES -> versioned graph tensors per task + split manifest
split_manifest.py     Group-safe (scaffold) train/val/cal/test manifest I/O
conformal.py          ConformalCalibrator (CQR, per-task qhat)
metric.py             ClsMetric (AUROC/AUPRC), RegMetric (RMSE/R2), interval metrics
loss.py               BCELoss, MSELoss, QuantileRegressionLoss
molecular_features.py Atom/bond feature schema (FEATURE_SCHEMA_VERSION),
                      canonical SMILES, scaffold extraction
record.py             PerformanceMeter (legacy best-val/test bookkeeping)
visualize.py          Compat shim re-exporting analysis helpers
weighting/            Loss balancing: EW, UW, DWA
architecture/
  Graphormer.py               Base encoder
  Graphormer_prompt.py        Prompt-conditioned encoder
  Graphormer_rgcer.py         RGCER encoder
  graphormer_backbone.py      Shared backbone
  prediction_heads.py         TaskPredictionHead (point/quantile)
  response_guided_router.py   RGCER router
  molecule_adaptive_prompt.py Prompt utilities
  toxacute_tasks.py           ToxAcute task registry
algos/algos.pyx, algos.pyx, setup.py   Cython extension (compiled in-place)
analysis/           Library-only (no CLI): interval_analysis, negative_transfer,
                    routing_analysis
tests/              28 pytest modules (contracts, RGCER ablations, conformal,
                    splits, trainer semantics, ...)
configs/            phase0_toxacute_3task.json
data/               Raw CSV (git-ignored; expected: data/toxacute.csv)
processed_graph_data/  Preprocessed .pt graphs, one subdirectory per task
artifacts/          Run outputs (git-ignored)
```

## Key Commands

All commands run from the project root inside the `icl` conda env.

```bash
# Rebuild the Cython extension (only needed for non-cp310 Python or after
# editing algos/algos.pyx)
python setup.py build_ext --inplace

# Preprocess raw CSV into per-task graph tensors + global split manifest
python preprocess_data.py \
    --raw_csv_path data/toxacute.csv \
    --task_list all \
    --output_dir processed_graph_data \
    --splitting scaffold \
    --valid_size 0.1 --calibration_size 0.1 --test_size 0.1 \
    --seed 42

# Train (RGCER, all 59 ToxAcute endpoints)
python main.py --mode train \
    --dataset toxacute --arch Graphormer_rgcer --toxacute_task_scope all59 \
    --preprocessed_data_dir processed_graph_data \
    --save_path artifacts/runs/toxacute_all59_rgcer \
    --experiment_tag full --seed 42 \
    --bs 64 --epochs 100 --gpu_id 0

# Train a base-architecture smoke run (3 human tasks)
python main.py --mode train \
    --dataset toxacute --arch Graphormer --toxacute_task_scope human3 \
    --save_path artifacts/runs/toxacute_smoke_human3 \
    --experiment_tag full

# Evaluate a checkpoint on the test split
python main.py --mode test --load_path <path/to/checkpoint> \
    --preprocessed_data_dir processed_graph_data \
    --save_path <run_dir>

# Single-molecule inference
python main.py --mode single_inference --smiles "CCO" \
    --load_path <path/to/checkpoint> \
    --preprocessed_data_dir processed_graph_data

# Batch inference on one task (writes CSV with median/lower/upper in log and mg/kg)
python main.py --mode batch_inference --load_path <path/to/checkpoint> \
    --preprocessed_data_dir processed_graph_data \
    --inference_task human_oral_TDLo \
    --inference_output_path artifacts/results/toxacute_predictions.csv

# Unit tests
pytest tests/
```

## Conventions & Invariants

- **Output layout**: `--save_path` is rewritten to
  `<save_path>/<experiment_tag>/seed_<seed>/` and receives `args.json`,
  `architecture_summary.txt`, `metrics.json`, `routing_summary.json`, and the
  best checkpoint (`<prefix>_<tag>_seed<seed>`).
- **Checkpoints are strict v5 for formal DataStore V2 runs** (legacy
  non-V2 paths remain v4): they must contain `checkpoint_version=5`,
  `model_state`, `optimizer_state`, `weighting_state`, `epoch`, a
  `data_config` block binding the DataStore fingerprint/build id/split
  manifest hash, and are loaded with `strict=True`. `test` /
  `single_inference` / `batch_inference` modes require `--load_path`
  pointing at a full checkpoint. `architecture_config` additionally pins
  behaviour-affecting sizes/rates (`head_dropout`, `hidden_dim`,
  `a_layers`, `adapter_ratio`, …) so a resume cannot silently change them.
- **Split protocol**: build order is chemistry scan → Manifest-V3 planning →
  hard preflight PASS → LMDB shards. Plan first with
  `preprocess_data.py --plan_split_only ...` (writes the manifest plus
  `split_report.json` carrying hard-constraint status), then pass the
  approved file to `--build_datastore_v2 --split_manifest_path <file>`.
  The planner pins oversized scaffolds (e.g. benzene) to train, balances
  endpoint *label presence* (never values) with human3 priority, and
  enforces acyclic eval coverage plus a 50% single-group dominance cap on
  every evaluation split. The store only accepts version-3 manifests.
- **Conformal scope & alpha floor**: `--conformal_scope human3|all_tasks`
  (default human3) selects which endpoints receive CQR states; point
  metrics always cover every task and intervals are labelled conformal
  only where qhat exists. Scoped endpoints below
  `ceil((1-alpha)/alpha)` calibration rows fail preflight
  (`FAIL_CONFORMAL_RANK`) before training starts; counts between the
  floor and 30 warn (`LOW_CALIBRATION_STABILITY`).
- **Response-profile determinism**: the RGCER router consumes head
  evaluations computed with dropout pinned off (`deterministic=True`) and
  no grad, so identical molecules yield identical routing evidence in
  train mode; supervised head calls keep normal dropout.
- **Split ratios** must satisfy `vs + calibration_size + ts < 1.0`. The split
  manifest is the single source of truth for all tasks — never re-split
  per task.
- **Prediction mode**: `quantile` is the default and only makes sense for
  regression datasets; classification datasets (`hiv`, `bace`, `bbbp`, `muv`,
  `tox21`, `sider`, `clintox`) are automatically forced to `point` with a
  RuntimeWarning.
- **Validation rules** (in `main.validate_params`): `hidden_dim` must be
  divisible by both `a_heads` and `prompt_heads`; sparse routing requires
  `router_top_k > 0`; `DWA` weighting requires `tasks_per_update > 1`;
  `adapter_ratio` in (0, 1].
- **Labels are log-scale**: model outputs are log values; convert to mg/kg
  with `calculate_mgkg` when reporting.
- **Task-wise training**: one use of every task batch per epoch
  (`tasks_per_update` controls how many tasks update per optimizer step);
  HPS warmup (`--hps_warmup_epochs`, default 10) runs before routing is
  enabled (`--routing_enabled`). Exposure is data-proportional by default
  (`--task_sampling proportional`); `human_target_floor` adds one extra
  full pass of each human loader per epoch for ablations, and every epoch
  records `schedule_diagnostics` (per-task batches, update fractions,
  human3 fraction) in the history/metrics output.
- RDKit app logging is disabled globally in `main.py`.
- `data/`, `artifacts/`, `processed_graph_data/` contents are git-ignored —
  large data files never go into git.
- Code style: plain modern Python with `from __future__ import annotations`,
  type hints, dataclasses for metadata, small module-private helpers prefixed
  with `_`. Tests are behavior/contract-oriented (one file per concern).

## Working Notes

- `experiment_config.py` hardcodes `TOXACUTE_PREPROCESSED_DIR` to this
  machine's absolute path (`D:\PROJECT\MTL\...`); override with
  `--preprocessed_data_dir` when moving the project.
- The `algos` package must be importable (compiled in-place `.pyd`/`.so`)
  before preprocessing or training; it is imported by `preprocess_data.py`.
- `analysis/` modules are import-only utilities (no `main`/argparse) — they
  are meant to be used from notebooks or test code. `visualize.py` is only a
  re-export shim for the retired visualization entry point.
- Graph tensors carry `feature_schema_version`
  (`atom_v2_bond_v1_pathavg_v1`); the trainer verifies it matches
  `molecular_features.FEATURE_SCHEMA_VERSION` on load.
