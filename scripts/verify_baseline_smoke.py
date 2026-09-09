#!/usr/bin/env python
"""Independently verify one baseline smoke artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.artifacts import verify_run
from baselines.checkpoint import protocol_identity
from baselines.constants import HUMAN3_TASKS
from baselines.metrics import regression_metrics
from baselines.utils import read_json, sha256_file, write_json


REQUIRED = (
    "config_resolved.json", "environment.json", "data_manifest.json", "scaler.json",
    "metrics.json", "history.json", "validation_predictions.jsonl", "model_audit.json",
    "reload_verification.json", "run_result.json", "command.json", "stdout.log", "stderr.log",
)


def verify(run_dir: str | Path) -> dict:
    root = Path(run_dir).resolve()
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Run is missing required artifacts: {missing}")
    config = read_json(root / "config_resolved.json")
    if config.get("mode") != "human3_smoke" or config.get("allowed_splits") != ["train", "validation"]:
        raise ValueError("Run config is not locked Human3 train/validation smoke mode")
    for key, expected in protocol_identity().items():
        if config.get(key) != expected:
            raise ValueError(f"Frozen identity mismatch for {key}")
    manifest = read_json(root / "data_manifest.json")
    if manifest["train"]["count"] > 96 or manifest["validation"]["count"] > 24:
        raise ValueError("Smoke sample union exceeds protocol limits")
    if set(manifest) & {"test", "calibration"}:
        raise ValueError("Smoke data manifest exposes a forbidden split")
    records = [json.loads(line) for line in (root / "validation_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if [row["sample_id"] for row in records] != manifest["validation"]["sample_ids"]:
        raise ValueError("Prediction rows do not match validation sample identity/order")
    truth = np.full((len(records), 3), np.nan, dtype=np.float64)
    pred = np.empty((len(records), 3), dtype=np.float64)
    for row_index, row in enumerate(records):
        if row.get("split") != "validation":
            raise ValueError("Prediction artifact contains a non-validation row")
        for task_index, task in enumerate(HUMAN3_TASKS):
            if row["y_true"][task] is not None:
                truth[row_index, task_index] = float(row["y_true"][task])
            pred[row_index, task_index] = float(row["y_pred"][task])
    recomputed = regression_metrics(truth, pred, HUMAN3_TASKS)
    saved = read_json(root / "metrics.json")
    if json.dumps(recomputed, sort_keys=True) != json.dumps(saved, sort_keys=True):
        raise ValueError("Saved metrics do not exactly match independent recomputation")
    reload_result = read_json(root / "reload_verification.json")
    if reload_result.get("passed") is not True:
        raise ValueError("Independent checkpoint reload verification did not pass")
    checkpoint = root / reload_result["checkpoint"]
    if not checkpoint.exists() or sha256_file(checkpoint) != reload_result["checkpoint_sha256"]:
        raise ValueError("Checkpoint is missing or its SHA-256 differs")
    result = {
        "status": "passed", "method": config["method"], "run_dir": str(root),
        "validation_rows": len(records), "metrics_recomputed": True,
        "checkpoint_sha256_verified": True, "forbidden_splits_absent": True,
    }
    write_json(root / "verification.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    result = verify_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
