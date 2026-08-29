"""Regenerate best-epoch per-sample diagnostics from a finished D1 checkpoint.

The in-run best-epoch dumps written before the source-weight concatenation fix
carry corrupted ``column_*`` source names.  This script rebuilds the run from
its own ``args.json``, loads the strict v6 best checkpoint, replays the
**validation** forward pass only (calibration/test are never built into
iterators that get consumed), and rewrites:

- diagnostics/best_validation_human3_predictions.csv
- diagnostics/best_validation_human3_routing.csv
- diagnostics/routing_source_frequency.json

Usage:
  python scripts/regenerate_best_diagnostics.py \
      --run_dir artifacts/runs/diag_no_null/d1_no_null_e40/seed_42 [--gpu_id 1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import prepare_args
from dataset import DataCollator
from main import (
    _build_model_components,
    _device_from_params,
    _loaders,
    build_task_dict,
    build_parser,
    task_names_for_params,
    validate_params,
)
from reproducibility import seed_everything
from run_diagnostics import RunDiagnosticsWriter
from trainer import Trainer
import weighting as weighting_method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--gpu_id", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    params = build_parser().parse_args([])
    saved = json.loads((run_dir / "args.json").read_text(encoding="utf-8"))
    for key, value in saved.items():
        setattr(params, key, value)
    if args.gpu_id is not None:
        params.gpu_id = args.gpu_id
    validate_params(params)

    best_summary = run_dir / "diagnostics" / "epoch_summary.csv"
    best_epoch = None
    import csv

    with best_summary.open(newline="", encoding="utf-8") as handle:
        best_rows = [row for row in csv.DictReader(handle) if row["is_best"] == "True"]
    if best_rows:
        best_epoch = max(int(row["epoch"]) for row in best_rows)
    if best_epoch is None:
        raise SystemExit(f"No best epoch recorded under {best_summary}")

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
    checkpoint_path = run_dir / f"{params.ckpt_name}_best.pt"
    trainer.load_checkpoint(checkpoint_path)

    store = _resolve_data_store(params, task_names, required=True)
    trainer.sample_row_index = {
        str(sample_id): int(row_index)
        for sample_id, row_index in zip(store.sample_ids.tolist(), store.row_indices.tolist())
    }
    loaders = _loaders(params, task_names, collator)

    trainer.model.eval()
    _, _, route_records = trainer._collect_predictions(
        loaders["val"],
        epoch=best_epoch,
        routing_enabled_override=True,
    )
    writer = RunDiagnosticsWriter(
        run_dir / "diagnostics",
        task_names=list(trainer.task_name),
        sample_row_index=trainer.sample_row_index,
    )
    writer.note_best_epoch(best_epoch, route_records)
    writer.write_best_artifacts(best_epoch)
    print(f"regenerated best-epoch diagnostics for epoch {best_epoch} in {run_dir}")


if __name__ == "__main__":
    main()
