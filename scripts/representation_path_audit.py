"""Representation-path audit over finished D1 best checkpoints (D3 plan §5-§13).

For each seed directory the script rebuilds the run from its own ``args.json``,
loads the strict v6 best checkpoint, replays the **validation** forward pass
only (calibration/test never iterated), computes the per-sample
representation-chain norms, and writes:

- REPRESENTATION_PATH_AUDIT_SUMMARY.csv   (per seed x human task aggregates)
- REPRESENTATION_PATH_AUDIT_PER_SAMPLE.csv (plan §12 columns)

Usage:
  python scripts/representation_path_audit.py \
      --run_dirs artifacts/runs/diag_no_null/d1_no_null_e40/seed_42 ... \
      [--output_dir .] [--gpu_id 1]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from architecture.toxacute_tasks import HUMAN_TARGET_TASKS
from config import prepare_args
from dataset import DataCollator
from main import (
    _build_model_components,
    _device_from_params,
    _loaders,
    _resolve_data_store,
    build_task_dict,
    build_parser,
    task_names_for_params,
    validate_params,
)
from representation_audit import (
    AUDIT_PER_SAMPLE_FIELDS,
    AUDIT_SUMMARY_FIELDS,
    compute_representation_metrics,
)
from reproducibility import seed_everything
from trainer import Trainer
import weighting as weighting_method


def _write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _finite_mean(values):
    finite = [value for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def audit_seed(run_dir: Path, gpu_id, seed: int):
    params = build_parser().parse_args([])
    saved = json_load(run_dir / "args.json")
    for key, value in saved.items():
        setattr(params, key, value)
    if gpu_id is not None:
        params.gpu_id = gpu_id
    validate_params(params)

    best_epoch = read_best_epoch(run_dir / "diagnostics" / "epoch_summary.csv")
    if best_epoch is None:
        raise SystemExit(f"No best epoch recorded under {run_dir}")

    seed_everything(params.seed)
    task_names = task_names_for_params(params)
    kwargs, optim_param = prepare_args(params)
    encoder_class, architecture_class, decoders = _build_model_components(
        params, task_names, _device_from_params(params)
    )
    collator = DataCollator(
        spatial_pos_max_clip=params.spatial_pos_clip,
        max_node_filter=None,
    )
    trainer = Trainer(
        task_dict=build_task_dict(params, task_names),
        weighting=weighting_method.__dict__[params.weighting],
        architecture=architecture_class,
        encoder_class=encoder_class,
        decoders=decoders,
        optim_param=optim_param,
        args=params,
        save_path=str(run_dir),
        load_path=None,
        **kwargs,
    )
    trainer.load_checkpoint(run_dir / f"{params.ckpt_name}_best.pt")
    store = _resolve_data_store(params, task_names, required=True)
    trainer.sample_row_index = {
        str(sample_id): int(row_index)
        for sample_id, row_index in zip(store.sample_ids.tolist(), store.row_indices.tolist())
    }
    loaders = _loaders(params, task_names, collator)

    trainer.model.eval()
    per_sample_rows = []
    summary_rows = []
    human_tasks = [task for task in task_names if task in set(HUMAN_TARGET_TASKS)]
    with torch.no_grad():
        for task in trainer.task_name:
            if task not in human_tasks:
                continue
            loader = loaders["val"].get(task)
            if loader is None:
                continue
            task_rows = []
            for batch in loader:
                if getattr(batch, "get", lambda *_: None)("is_empty", False):
                    continue
                batch = batch.to(trainer.device)
                output, diagnostics = trainer._forward_task(
                    batch, task, best_epoch, return_aux=True, routing_enabled_override=True
                )
                if not diagnostics:
                    continue
                rep = compute_representation_metrics(trainer.model, task, diagnostics)
                labels = batch.y.reshape(-1).float().cpu()
                base_pred = trainer.decode_task_output(
                    task, diagnostics["base_raw"], apply_conformal=False
                )["median"].cpu().reshape(-1)
                route_pred = trainer.decode_task_output(
                    task, output[task], apply_conformal=False
                )["median"].cpu().reshape(-1)
                sample_ids = list(getattr(batch, "sample_id", None) or [])
                weights = diagnostics.get("source_weights")
                count = labels.numel()
                for index in range(count):
                    row = {
                        "seed": seed,
                        "sample_id": sample_ids[index] if index < len(sample_ids) else "",
                        "row_index": trainer.sample_row_index.get(
                            str(sample_ids[index]) if index < len(sample_ids) else "", ""
                        ),
                        "task": task,
                        "label": float(labels[index]),
                        "base_prediction": float(base_pred[index]),
                        "route_prediction": float(route_pred[index]),
                    }
                    for field in (
                        "h_norm",
                        "source_context_norm",
                        "source_context_ratio",
                        "film_delta_norm",
                        "film_delta_ratio",
                        "adapter_delta_norm",
                        "adapter_delta_ratio",
                        "route_rep_delta_norm",
                        "route_rep_delta_ratio",
                        "gamma_abs_mean",
                        "beta_abs_mean",
                        "routing_entropy",
                        "routing_variance",
                    ):
                        values = rep.get(field)
                        row[field] = float(values[index]) if values is not None and index < values.numel() else float("nan")
                    if weights is not None and index < weights.size(0):
                        top_weight, top_index = weights[index].float().max(dim=0)
                        row["top1_source"] = (
                            task_names[int(top_index)] if int(top_index) < len(task_names) else f"column_{int(top_index)}"
                        )
                        row["top1_weight"] = float(top_weight)
                    else:
                        row["top1_source"] = ""
                        row["top1_weight"] = float("nan")
                    task_rows.append(row)
            per_sample_rows.extend(task_rows)
            if task_rows:
                summary = {"seed": seed, "task": task, "n": len(task_rows)}
                for field in AUDIT_SUMMARY_FIELDS[3:]:
                    if field == "route_minus_base_abs_mean":
                        deltas = [
                            abs(row["route_prediction"] - row["base_prediction"])
                            for row in task_rows
                        ]
                        summary[field] = _finite_mean(deltas)
                    else:
                        base_field = field[: -len("_mean")]
                        summary[field] = _finite_mean([row.get(base_field) for row in task_rows])
                summary_rows.append(summary)
    return per_sample_rows, summary_rows, best_epoch


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def read_best_epoch(path: Path):
    if not path.exists():
        return None
    best_rows = [
        row
        for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if row["is_best"] == "True"
    ]
    if not best_rows:
        return None
    return max(int(row["epoch"]) for row in best_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        default=[
            "artifacts/runs/diag_no_null/d1_no_null_e40/seed_42",
            "artifacts/runs/diag_no_null/d1_no_null_e40/seed_43",
            "artifacts/runs/diag_no_null/d1_no_null_e40/seed_44",
        ],
    )
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--gpu_id", default="1")
    args = parser.parse_args()

    all_per_sample = []
    all_summary = []
    for run_dir in args.run_dirs:
        run_path = Path(run_dir)
        seed = int(run_path.name.replace("seed_", ""))
        per_sample, summary, best_epoch = audit_seed(run_path, args.gpu_id, seed)
        all_per_sample.extend(per_sample)
        all_summary.extend(summary)
        print(f"audited {run_path} (best epoch {best_epoch}): {len(per_sample)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "REPRESENTATION_PATH_AUDIT_PER_SAMPLE.csv", AUDIT_PER_SAMPLE_FIELDS, all_per_sample)
    _write_csv(output_dir / "REPRESENTATION_PATH_AUDIT_SUMMARY.csv", AUDIT_SUMMARY_FIELDS, all_summary)
    print(f"wrote {output_dir / 'REPRESENTATION_PATH_AUDIT_SUMMARY.csv'}")
    print(f"wrote {output_dir / 'REPRESENTATION_PATH_AUDIT_PER_SAMPLE.csv'}")


if __name__ == "__main__":
    main()
