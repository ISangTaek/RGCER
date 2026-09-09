#!/usr/bin/env python
"""Load a baseline checkpoint independently and predict an authorized split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.constants import HUMAN3_TASKS, METHODS
from baselines.data import load_authorized_validation
from baselines.inference import predict_partition
from baselines.metrics import regression_metrics
from baselines.utils import require_nonexistent_output, sha256_file, write_json


def _write_rows(path: Path, partition, predictions: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, sample_id in enumerate(partition.sample_ids):
            for column, task in enumerate(HUMAN3_TASKS):
                truth = partition.labels_human3[row, column]
                record = {
                    "sample_id": sample_id,
                    "row_index": int(partition.raw_row_indices[row]),
                    "split": partition.split,
                    "endpoint": task,
                    "y_true": float(truth) if np.isfinite(truth) else None,
                    "y_pred": float(predictions[row, column]),
                    "mask": bool(np.isfinite(truth)),
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=METHODS + ("attentivefp",))
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-root", help="Locked Chemprop/GROVER source tree when required")
    parser.add_argument("--split", default="validation", choices=("validation", "calibration", "test"))
    args = parser.parse_args()
    method = "afp" if args.method == "attentivefp" else args.method
    partition = load_authorized_validation(args.datastore, method or "rf", split=args.split)
    output = require_nonexistent_output(args.output)
    try:
        predictions, payload, scaler = predict_partition(
            args.checkpoint,
            partition,
            device=args.device,
            source_root=args.source_root,
            expected_method=method,
        )
        _write_rows(output / "predictions.jsonl", partition, predictions)
        metrics = regression_metrics(partition.labels_human3, predictions, HUMAN3_TASKS)
        write_json(output / "metrics.json", metrics)
        write_json(output / "inference_manifest.json", {
            "status": "PASS",
            "method": payload["method"],
            "model_type": payload["model_type"],
            "split": partition.split,
            "sample_count": len(partition.sample_ids),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "task_names": list(scaler.task_names),
            "feature_schema": payload["feature_schema"],
            "restored_scaler_from_checkpoint": True,
            "restored_config_from_checkpoint": True,
        })
    finally:
        close = getattr(partition.graph_records, "close", None)
        if close is not None:
            close()
    print(f"PASS validation predictions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
