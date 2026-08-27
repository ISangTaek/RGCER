"""Benchmark DataStore V2 startup and random-read throughput."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import DataCollator
from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset


def _tree_stats(root: Path) -> dict[str, float | int]:
    """Collect file count, byte count, and recursive scan time."""

    started = time.perf_counter()
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file():
            file_count += 1
            total_bytes += path.stat().st_size
    return {
        "path": str(root),
        "file_count": file_count,
        "bytes": total_bytes,
        "scan_seconds": time.perf_counter() - started,
    }


def _parse_workers(value: str) -> list[int]:
    workers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        count = int(item)
        if count < 0:
            raise ValueError("worker counts must be non-negative")
        if count not in workers:
            workers.append(count)
    if not workers:
        raise ValueError("at least one worker count is required")
    return workers


def _benchmark_loader(dataset, *, batch_size: int, num_workers: int, max_batches: int) -> dict:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=DataCollator(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    iterator = iter(loader)
    consumed = 0
    non_empty = 0
    first_batch_seconds = None
    started = time.perf_counter()
    try:
        for _ in range(min(int(max_batches), len(loader))):
            batch_started = time.perf_counter()
            batch = next(iterator)
            elapsed = time.perf_counter() - batch_started
            if first_batch_seconds is None:
                first_batch_seconds = elapsed
            consumed += 1
            if not batch.get("is_empty", False):
                non_empty += 1
    finally:
        shutdown = getattr(iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    elapsed = time.perf_counter() - started
    return {
        "num_workers": num_workers,
        "dataset_size": len(dataset),
        "batches": consumed,
        "non_empty_batches": non_empty,
        "first_batch_seconds": first_batch_seconds,
        "elapsed_seconds": elapsed,
        "batches_per_second": consumed / elapsed if elapsed > 0 else 0.0,
    }


def benchmark_datastore(
    root: str | Path,
    *,
    task_name: str,
    split: str,
    batch_size: int = 64,
    max_nodes: int | None = None,
    max_batches: int = 100,
    workers: list[int] | None = None,
    legacy_root: str | Path | None = None,
    strict: bool = False,
) -> dict:
    root = Path(root).resolve()
    workers = [0, 2, 4] if workers is None else list(workers)
    if batch_size <= 0 or max_batches <= 0:
        raise ValueError("batch_size and max_batches must be positive")

    result = {"v2": {}, "legacy": None}
    if legacy_root is not None:
        legacy_path = Path(legacy_root).resolve()
        result["legacy"] = _tree_stats(legacy_path)

    opened = time.perf_counter()
    store = ToxAcuteDataStore.resolve(root)
    open_seconds = time.perf_counter() - opened
    try:
        if strict:
            store.validate(strict=True, expected_task_names=[task_name])
        dataset_started = time.perf_counter()
        dataset = ToxAcuteTaskDataset(
            store,
            task_name,
            split=split,
            max_nodes=max_nodes,
        )
        dataset_init_seconds = time.perf_counter() - dataset_started
        result["v2"].update(
            {
                "tree": _tree_stats(store.root),
                "open_seconds": open_seconds,
                "dataset_init_seconds": dataset_init_seconds,
                "build_id": store.build_id,
                "datastore_fingerprint": store.fingerprint,
                "task_name": task_name,
                "split": split,
                "max_nodes": max_nodes,
                "loaders": [
                    _benchmark_loader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        max_batches=max_batches,
                    )
                    for num_workers in workers
                ],
            }
        )
    finally:
        store.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DataStore V2 random reads")
    parser.add_argument("--root", "--data_store_dir", dest="root", required=True)
    parser.add_argument("--task", dest="task_name", required=True)
    parser.add_argument("--split", choices=["train", "validation", "calibration", "test"], default="train")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_nodes", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--workers", default="0,2,4", help="Comma-separated worker counts")
    parser.add_argument("--legacy_root", default=None, help="Optional V1 tree for file-count comparison")
    parser.add_argument("--strict", action="store_true", help="Run strict per-graph validation before timing")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()
    payload = benchmark_datastore(
        args.root,
        task_name=args.task_name,
        split=args.split,
        batch_size=args.batch_size,
        max_nodes=args.max_nodes,
        max_batches=args.max_batches,
        workers=_parse_workers(args.workers),
        legacy_root=args.legacy_root,
        strict=args.strict,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + os.linesep, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
