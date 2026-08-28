"""Aggregate seed runs after enforcing a shared data/config contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS


# These fields identify the output location or contain process-local objects;
# they are intentionally excluded so a seed directory can move independently
# without changing the experiment contract.
_NON_EXPERIMENT_ARG_KEYS = {
    "seed",
    "save_path",
    "load_path",
    "data_store_dir",
    "preprocessed_data_dir",
    "inference_output_path",
    "datastore_metadata",
    "datastore_context",
    "label_providers",
    "ckpt_name",
}


def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_run(path: Path, args: dict) -> int:
    if "seed" in args:
        return int(args["seed"])
    match = re.search(r"seed_(\d+)$", path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot determine model seed for run: {path}")


def _find_checkpoint(run: Path) -> Path | None:
    checkpoints = sorted(run.glob("*_best.pt"))
    return checkpoints[0] if checkpoints else None


def _stable_experiment_config(args: dict) -> dict:
    """Return JSON-stable run settings; only ``seed`` may vary by contract."""

    return {
        key: value
        for key, value in args.items()
        if key not in _NON_EXPERIMENT_ARG_KEYS
    }


def _contract(run: Path, args: dict) -> dict:
    preflight = _read_json(run / "data_preflight.json", {}) or {}
    effective = _read_json(run / "effective_config.json", None)
    checkpoint_data = {}
    checkpoint_path = _find_checkpoint(run)
    if checkpoint_path is not None:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_data = checkpoint.get("data_config", {})
        if effective is None:
            effective = checkpoint.get("effective_rgcer_config")
        if not args.get("task_names"):
            args = {**args, "task_names": checkpoint.get("task_names")}
    effective = effective or {
        "architecture": args.get("arch"),
        "transfer_mechanism": args.get("rgcer_transfer_mechanism"),
    }
    task_names = args.get("task_names") or checkpoint_data.get("task_names") or []
    return {
        "datastore_fingerprint": preflight.get("datastore_fingerprint") or checkpoint_data.get("datastore_fingerprint"),
        "split_manifest_hash": preflight.get("split_manifest_hash") or checkpoint_data.get("split_manifest_hash"),
        "raw_csv_sha256": preflight.get("raw_csv_sha256") or checkpoint_data.get("raw_csv_sha256"),
        "feature_schema_version": preflight.get("feature_schema_version") or checkpoint_data.get("feature_schema_version"),
        "max_path_distance": preflight.get("max_path_distance") or checkpoint_data.get("max_path_distance"),
        "task_names": list(task_names),
        "effective_config": effective,
        "experiment_config": _stable_experiment_config(args),
        "prediction_config": {
            key: args.get(key)
            for key in ("prediction_mode", "lower_quantile", "upper_quantile", "conformal_alpha", "fit_conformal")
        },
    }


def _load_run(run: Path) -> dict:
    args = _read_json(run / "args.json", {}) or {}
    metrics = _read_json(run / "metrics.json", {}) or {}
    return {
        "path": run,
        "seed": _seed_from_run(run, args),
        "contract": _contract(run, args),
        "metrics": metrics,
        "auxiliary_task_names": set(args.get("auxiliary_task_names", [])),
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def validate_contract(runs: list[dict]) -> None:
    if not runs:
        raise ValueError("No runs were found")
    reference = runs[0]["contract"]
    for field in (
        "datastore_fingerprint",
        "split_manifest_hash",
        "raw_csv_sha256",
        "feature_schema_version",
        "max_path_distance",
        "task_names",
        "effective_config",
        "experiment_config",
        "prediction_config",
    ):
        if reference.get(field) in (None, "", []):
            raise ValueError(f"Run contract is missing required field: {field}")
    for run in runs[1:]:
        for field in reference:
            if _canonical(run["contract"].get(field)) != _canonical(reference.get(field)):
                raise ValueError(
                    f"Run contract mismatch for {field!r}: {runs[0]['path']} vs {run['path']}"
                )
    seeds = [run["seed"] for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate model seeds: {seeds}")


def _test_payload(run: dict) -> dict:
    metrics = run["metrics"]
    return metrics.get("test") or metrics.get("validation") or {}


def _aggregate_records(records: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    grouped = {}
    for record in records:
        key = tuple(record.get(name) for name in group_keys)
        grouped.setdefault(key, []).append(float(record["value"]))
    rows = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        rows.append(
            {
                **dict(zip(group_keys, key)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
            }
        )
    return rows


def aggregate_runs(experiment_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    experiment_dir = Path(experiment_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else experiment_dir
    run_paths = sorted(path for path in experiment_dir.iterdir() if path.is_dir() and (path / "metrics.json").exists())
    runs = [_load_run(path) for path in run_paths]
    validate_contract(runs)

    metric_records = []
    endpoint_records = []
    routing_records = []
    for run in runs:
        payload = _test_payload(run)
        auxiliary_tasks = run["auxiliary_task_names"]
        for task, values in (payload.get("tasks", {}) or {}).items():
            for metric, value in values.items():
                if value is None or not math.isfinite(float(value)):
                    continue
                endpoint_records.append({"task": task, "metric": metric, "value": value})
                if task not in auxiliary_tasks:
                    metric_records.append({"metric": metric, "value": value})
        for task, values in (payload.get("routing", {}) or {}).items():
            for metric, value in values.items():
                if value is None or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    continue
                routing_records.append({"task": task, "metric": metric, "value": value})

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "aggregate_metrics": output_dir / "aggregate_metrics.csv",
        "per_endpoint_metrics": output_dir / "per_endpoint_metrics.csv",
        "human3_metrics": output_dir / "human3_metrics.csv",
        "routing_metrics": output_dir / "routing_metrics.csv",
    }
    tables = {
        outputs["aggregate_metrics"]: _aggregate_records(metric_records, ("metric",)),
        outputs["per_endpoint_metrics"]: _aggregate_records(endpoint_records, ("task", "metric")),
        outputs["routing_metrics"]: _aggregate_records(routing_records, ("task", "metric")),
    }
    human3 = [record for record in endpoint_records if record.get("task") in HUMAN_TARGET_TASKS]
    tables[outputs["human3_metrics"]] = _aggregate_records(human3, ("task", "metric"))
    for path, rows in tables.items():
        fieldnames = sorted({field for row in rows for field in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames or ["mean", "std", "n_seeds"])
            writer.writeheader()
            writer.writerows(rows)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate compatible seed runs")
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    for path in aggregate_runs(args.experiment_dir, args.output_dir).values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
