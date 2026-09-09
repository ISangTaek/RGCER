# Linux server handoff

This code has local Human3 smoke coverage only. It has not run server HPO, five-seed training, calibration, formal test, or Figure generation. Pulling the code does not authorize those runs.

## 1. Isolated environment

```bash
cd /home/shangzeli/RGCER
python3 -m venv --system-site-packages .venv-baselines
source .venv-baselines/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-baselines.txt
```

Do not install into or mutate the existing server RGCER environment. Confirm that the new venv can import the existing compatible PyTorch/CUDA runtime before proceeding.

## 2. Official sources and GROVER_base

Choose explicit server paths outside the repository's tracked tree:

```bash
ASSET_ROOT=/home/shangzeli/baseline_assets
mkdir -p "$ASSET_ROOT"

git clone https://github.com/chemprop/chemprop.git "$ASSET_ROOT/chemprop-1.6.1"
git -C "$ASSET_ROOT/chemprop-1.6.1" checkout f3d1bff19a6e1b03d28e9cfabdf4c80dd8c67382

git clone https://github.com/tencent-ailab/grover.git "$ASSET_ROOT/grover"
git -C "$ASSET_ROOT/grover" checkout 40b6d97098e4508687912f3c05eca369fc2c6213

git clone https://github.com/LuJiangTHU/Acute_Toxicity_FSL.git "$ASSET_ROOT/Acute_Toxicity_FSL"
git -C "$ASSET_ROOT/Acute_Toxicity_FSL" checkout 704edcb951ab397e31561bc130154d01e0103b3c
```

Download GROVER_base only from the official GROVER README (Google Drive file id `1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl` or its listed OneDrive mirror), then place it at a chosen server path. The required SHA-256 is:

```text
47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de
```

The loader rejects a different hash, any missing encoder key, and any encoder shape mismatch.

## 3. Fail-closed READY preflight

Set the real frozen DataStore build path; do not copy `.tmp` from a workstation.

```bash
DATASTORE=/home/shangzeli/RGCER/data/toxacute_datastore_v2/builds/toxacute-v2-7b7bd62a6457
GROVER_WEIGHT="$ASSET_ROOT/grover_base_download"

python scripts/baseline_ready.py \
  --datastore "$DATASTORE" \
  --chemprop-source "$ASSET_ROOT/chemprop-1.6.1" \
  --grover-source "$ASSET_ROOT/grover" \
  --grover-pretrained "$GROVER_WEIGHT" \
  --toxacol-source "$ASSET_ROOT/Acute_Toxicity_FSL" \
  --output-json /home/shangzeli/baseline_ready.json
```

Only a zero exit code and `status=READY` count. Skipped integration tests do not count as READY.

## 4. Portable targeted tests

```bash
python -m pytest -q tests/baselines \
  --baseline-datastore "$DATASTORE" \
  --chemprop-source "$ASSET_ROOT/chemprop-1.6.1" \
  --grover-source "$ASSET_ROOT/grover" \
  --grover-pretrained "$GROVER_WEIGHT" \
  --toxacol-source "$ASSET_ROOT/Acute_Toxicity_FSL"
```

Without these parameters, ordinary external-asset tests may report skips. That mode is useful for source-only checks but is not an execution gate.

## 5. Local-size smoke commands

Every output directory must be new.

```bash
python scripts/baseline_smoke.py --method rf --datastore "$DATASTORE" --output /tmp/baseline-rf-smoke --device cpu --seed 42
python scripts/baseline_smoke.py --method attentivefp --datastore "$DATASTORE" --output /tmp/baseline-afp-smoke --device cpu --seed 42
python scripts/baseline_smoke.py --method dmpnn --datastore "$DATASTORE" --output /tmp/baseline-dmpnn-smoke --device cpu --seed 42 --source-root "$ASSET_ROOT/chemprop-1.6.1"
python scripts/baseline_smoke.py --method grover --datastore "$DATASTORE" --output /tmp/baseline-grover-smoke --device cpu --seed 42 --source-root "$ASSET_ROOT/grover" --pretrained "$GROVER_WEIGHT"
python scripts/baseline_smoke.py --method toxacol --datastore "$DATASTORE" --output /tmp/baseline-toxacol-smoke --device cpu --seed 42
```

Verify each completed smoke directory:

```bash
python scripts/verify_baseline_smoke.py --run-dir /tmp/baseline-rf-smoke
```

## 6. Formal train/validation interface

Generate the deterministic trial files only when verifying a checkout; generation does not run HPO:

```bash
python scripts/generate_baseline_trials.py
```

The full training interface requires an explicit method, zero-based trial index, seed, DataStore, and new output directory. Examples:

```bash
python scripts/baseline_train.py --method rf --trial 0 --seed 42 --datastore "$DATASTORE" --output /home/shangzeli/baseline_runs/rf-t00-s42 --device cuda:0

python scripts/baseline_train.py --method dmpnn --trial 0 --seed 42 --datastore "$DATASTORE" --output /home/shangzeli/baseline_runs/dmpnn-t00-s42 --device cuda:0 --source-root "$ASSET_ROOT/chemprop-1.6.1"

python scripts/baseline_train.py --method grover --trial 0 --seed 42 --datastore "$DATASTORE" --output /home/shangzeli/baseline_runs/grover-t00-s42 --device cuda:0 --source-root "$ASSET_ROOT/grover" --pretrained "$GROVER_WEIGHT"
```

TOXACol formal mode automatically uses the full joint59 train union; all other methods use the full Human3 train union. All methods select only by complete Human3 validation macro-RMSE. Do not launch the trial matrix until a separate execution card authorizes it.

## 7. Independent checkpoint inference

Validation is the only enabled split. D-MPNN and GROVER inference require the matching source checkout but do not require the original pretrained file because the trained checkpoint contains the complete finetuned state.

```bash
python scripts/baseline_predict.py \
  --checkpoint /home/shangzeli/baseline_runs/dmpnn-t00-s42/checkpoint.pt \
  --method dmpnn \
  --datastore "$DATASTORE" \
  --source-root "$ASSET_ROOT/chemprop-1.6.1" \
  --split validation \
  --device cuda:0 \
  --output /home/shangzeli/baseline_predictions/dmpnn-t00-s42-validation
```

Requests for `calibration` or `test` fail closed. Enabling those splits requires a later reviewed authorization change; this interface does not imply that authorization.

## 8. Output and limits

Training writes the resolved effective configuration, data manifest, scaler, history, original-unit validation predictions and metrics, model/source audit, checkpoint and SHA-256, logs, command/exit status, and an independent reload report. Smoke additionally writes the exact selected sample manifest and supports the independent smoke verifier.

Do not commit DataStore files, checkpoints, weights, logs, `.tmp`, or server asset checkouts. Do not report HPO, five seeds, calibration, formal test, or server execution as complete until they have actually run and been separately accepted.
