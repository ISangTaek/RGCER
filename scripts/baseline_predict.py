#!/usr/bin/env python
"""Load a baseline checkpoint independently and predict an authorized split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.checkpoint import protocol_identity
from baselines.constants import HUMAN3_TASKS, METHODS
from baselines.data import derive_authorized_sample_expectation, load_authorized_inference_split
from baselines.inference import predict_partition
from baselines.inference import load_checkpoint
from baselines.inference_artifacts import (
    InferenceManifestExpectation,
    inference_metrics,
    prediction_rows,
    sample_manifest_payload,
    verify_inference_artifacts,
)
from baselines.inference_authorization import authorize_test_inference, current_inference_identity
from baselines.utils import require_nonexistent_output, sha256_file, write_json


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _actual_device(method: str, requested_device: str) -> str:
    if method == "rf":
        return "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"Requested unavailable device: {requested_device}")
        index = torch.cuda.current_device() if device.index is None else device.index
        return f"cuda:{index}"
    return str(device)


def _source_commit_for_manifest(
    payload: dict,
    method: str,
    source_root: str | None,
    preflight_commit: str | None,
) -> str | None:
    expected = payload["feature_schema"].get("source_commit")
    if method not in ("dmpnn", "grover"):
        if preflight_commit is not None:
            raise ValueError(f"Unexpected external source commit for method {method}")
        return expected
    if source_root is None:
        raise ValueError(f"{method} checkpoint inference requires --source-root")
    actual = preflight_commit
    if actual is None:
        actual = subprocess.run(
            ["git", "-C", str(Path(source_root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if actual != expected:
        raise ValueError(f"{method} source commit differs from the checkpoint feature schema")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=METHODS + ("attentivefp",))
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-root", help="Locked Chemprop/GROVER source tree when required")
    parser.add_argument("--split", default="validation", choices=("validation", "calibration", "test"))
    parser.add_argument("--authorization", help="Codex-issued fixed-checkpoint test authorization JSON")
    parser.add_argument("--authorization-sha256", help="Expected SHA-256 of --authorization")
    parser.add_argument("--task-id", help="Task id that must exactly match the authorization")
    parser.add_argument("--asset-id", help="Unique authorization asset to export")
    args = parser.parse_args(argv)
    requested_method = "afp" if args.method == "attentivefp" else args.method
    authorization_fields = (args.authorization, args.authorization_sha256, args.task_id, args.asset_id)
    if args.split == "calibration":
        raise PermissionError("Calibration prediction remains locked in Stage 3C2A")
    context = None
    if args.split == "test":
        missing = [
            flag
            for flag, value in (
                ("--authorization", args.authorization),
                ("--authorization-sha256", args.authorization_sha256),
                ("--task-id", args.task_id),
                ("--asset-id", args.asset_id),
                ("--method", requested_method),
            )
            if value is None
        ]
        if missing:
            raise PermissionError(f"Test prediction requires all authorization fields; missing={missing}")
        context = authorize_test_inference(
            authorization_path=args.authorization,
            authorization_sha256=args.authorization_sha256,
            task_id=args.task_id,
            asset_id=args.asset_id,
            method=requested_method,
            checkpoint=args.checkpoint,
            repository_root=ROOT,
            source_root=args.source_root,
        )
        payload = context.checkpoint_payload
        scaler = context.scaler
        method = context.asset.method
        seed = context.asset.seed
        trial = context.asset.trial
        checkpoint_sha256_before = context.checkpoint_sha256_before
        inference_identity = {
            "git_commit": context.inference_git_commit,
            "implementation_sha256": context.inference_implementation_sha256,
        }
    else:
        if any(value is not None for value in authorization_fields):
            raise ValueError("Authorization/task/asset arguments are only accepted with --split test")
        payload, scaler = load_checkpoint(args.checkpoint, expected_method=requested_method)
        method = payload["method"]
        seed = payload["seed"]
        trial = payload["config"].get("trial_index")
        checkpoint_sha256_before = sha256_file(args.checkpoint)
        inference_identity = current_inference_identity(ROOT)
    checkpoint_path = Path(args.checkpoint).resolve()
    asset_id = context.asset.asset_id if context is not None else None
    source_root = str(Path(args.source_root).resolve()) if args.source_root else None
    source_commit = _source_commit_for_manifest(
        payload,
        method,
        source_root,
        context.source_commit if context is not None else None,
    )
    actual_device = _actual_device(method, args.device)
    frozen_data_identity = protocol_identity()
    manifest_expectation = InferenceManifestExpectation(
        checkpoint_path=str(checkpoint_path),
        checkpoint_bytes=checkpoint_path.stat().st_size,
        checkpoint_sha256=checkpoint_sha256_before,
        method=method,
        model_type=payload["model_type"],
        seed=seed,
        trial=trial,
        asset_id=asset_id,
        task_order=tuple(HUMAN3_TASKS),
        checkpoint_task_names=tuple(scaler.task_names),
        feature_schema=dict(payload["feature_schema"]),
        restored_scaler_from_checkpoint=True,
        restored_config_from_checkpoint=True,
        authorization_required=context is not None,
        authorization_path=context.authorization_path if context is not None else None,
        authorization_sha256=context.authorization_sha256 if context is not None else None,
        authorization_task_id=context.authorization.task_id if context is not None else None,
        authorization_asset_id=asset_id,
        training_git_commit=context.asset.training_git_commit if context is not None else None,
        training_implementation_sha256=(
            context.asset.training_implementation_sha256 if context is not None else None
        ),
        inference_git_commit=inference_identity["git_commit"],
        inference_implementation_sha256=inference_identity["implementation_sha256"],
        protocol_sha256=frozen_data_identity["protocol_sha256"],
        datastore_build_id=frozen_data_identity["datastore_build_id"],
        datastore_fingerprint=frozen_data_identity["datastore_fingerprint"],
        split_manifest_hash=frozen_data_identity["split_manifest_hash"],
        datastore_argument=str(Path(args.datastore).resolve()),
        source_root=source_root,
        source_commit=source_commit,
        device=actual_device,
    )
    output = require_nonexistent_output(args.output)
    partition = load_authorized_inference_split(
        args.datastore,
        method,
        split=args.split,
        test_grant=context.grant if context is not None else None,
    )
    try:
        predictions, reloaded_payload, reloaded_scaler = predict_partition(
            args.checkpoint,
            partition,
            device=args.device,
            source_root=args.source_root,
            expected_method=method,
        )
        if reloaded_payload["method"] != payload["method"] or tuple(reloaded_scaler.task_names) != tuple(scaler.task_names):
            raise ValueError("Independently reloaded checkpoint contract differs from the preflight checkpoint")
        rows = prediction_rows(
            partition,
            predictions,
            asset_id=asset_id,
            method=method,
            seed=seed,
        )
        _write_rows(output / "predictions.jsonl", rows)
        metrics = inference_metrics(partition.labels_human3, predictions, partition.split)
        write_json(output / "metrics.json", metrics)
        sample_manifest = sample_manifest_payload(
            partition.split,
            partition.sample_ids,
            partition.raw_row_indices,
            partition.labels_human3,
        )
        write_json(output / "sample_manifest.json", sample_manifest)
        write_json(output / "inference_manifest.json", {
            "schema_version": 1,
            "status": "EXECUTED_NOT_SCIENTIFIC_PASS",
            "method": manifest_expectation.method,
            "model_type": manifest_expectation.model_type,
            "seed": manifest_expectation.seed,
            "trial": manifest_expectation.trial,
            "asset_id": manifest_expectation.asset_id,
            "split": partition.split,
            "sample_count": len(partition.sample_ids),
            "prediction_row_count": len(rows),
            "sample_identity_hash": sample_manifest["sample_identity_hash"],
            "checkpoint": {
                "path": manifest_expectation.checkpoint_path,
                "bytes": manifest_expectation.checkpoint_bytes,
                "sha256": manifest_expectation.checkpoint_sha256,
            },
            "task_order": list(manifest_expectation.task_order),
            "checkpoint_task_names": list(manifest_expectation.checkpoint_task_names),
            "feature_schema": manifest_expectation.feature_schema,
            "restored_scaler_from_checkpoint": manifest_expectation.restored_scaler_from_checkpoint,
            "restored_config_from_checkpoint": manifest_expectation.restored_config_from_checkpoint,
            "training_identity": {
                "git_commit": manifest_expectation.training_git_commit,
                "implementation_sha256": manifest_expectation.training_implementation_sha256,
            },
            "inference_identity": {
                "git_commit": manifest_expectation.inference_git_commit,
                "implementation_sha256": manifest_expectation.inference_implementation_sha256,
            },
            "authorization": {
                "required": manifest_expectation.authorization_required,
                "path": manifest_expectation.authorization_path,
                "sha256": manifest_expectation.authorization_sha256,
                "task_id": manifest_expectation.authorization_task_id,
                "asset_id": manifest_expectation.authorization_asset_id,
            },
            "data_identity": {
                "protocol_sha256": manifest_expectation.protocol_sha256,
                "datastore_build_id": manifest_expectation.datastore_build_id,
                "datastore_fingerprint": manifest_expectation.datastore_fingerprint,
                "split_manifest_hash": manifest_expectation.split_manifest_hash,
                "datastore_argument": manifest_expectation.datastore_argument,
            },
            "source_root": manifest_expectation.source_root,
            "source_commit": manifest_expectation.source_commit,
            "device": manifest_expectation.device,
        })
    finally:
        close = getattr(partition.graph_records, "close", None)
        if close is not None:
            close()
    expectation = derive_authorized_sample_expectation(
        args.datastore,
        method,
        split=args.split,
        test_grant=context.grant if context is not None else None,
    )
    verify_inference_artifacts(
        output,
        expectation,
        manifest_expectation,
    )
    print(f"PASS {args.split} predictions -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
