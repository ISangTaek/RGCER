"""Global, group-safe train/validation/calibration/test manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "test": 0.10,
}


def create_split_manifest(
    records: Iterable[dict],
    *,
    splitting: str = "scaffold",
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> dict:
    records = [dict(record) for record in records]
    ratios = dict(DEFAULT_RATIOS if ratios is None else ratios)
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"ratios must contain exactly {SPLIT_NAMES}")
    if any(float(ratios[name]) < 0 for name in SPLIT_NAMES) or not np.isclose(
        sum(float(ratios[name]) for name in SPLIT_NAMES), 1.0
    ):
        raise ValueError("split ratios must be non-negative and sum to 1")
    if splitting not in {"scaffold", "random"}:
        raise ValueError(f"Unsupported splitting method: {splitting}")
    if not records:
        raise ValueError("Cannot create a split manifest from zero records")
    if any(not record.get("sample_id") for record in records):
        raise ValueError("Every manifest record needs a non-empty sample_id")

    group_field = "scaffold" if splitting == "scaffold" else "canonical_smiles"
    groups: dict[str, list[dict]] = {}
    for record in records:
        group_key = record.get(group_field) or record.get("canonical_smiles") or record["sample_id"]
        groups.setdefault(str(group_key), []).append(record)

    rng = np.random.default_rng(seed)
    grouped = list(groups.values())
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)

    target = np.array([ratios[name] * len(records) for name in SPLIT_NAMES], dtype=float)
    counts = np.zeros(len(SPLIT_NAMES), dtype=float)
    assignments: dict[str, str] = {}
    for group in grouped:
        size = len(group)
        candidate_scores = []
        for split_index in range(len(SPLIT_NAMES)):
            projected = counts.copy()
            projected[split_index] += size
            score = float(np.square((projected - target) / np.maximum(target, 1.0)).sum())
            candidate_scores.append(score)
        chosen = int(np.argmin(candidate_scores))
        split_name = SPLIT_NAMES[chosen]
        counts[chosen] += size
        for record in group:
            sample_id = record["sample_id"]
            if sample_id in assignments:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id}")
            assignments[sample_id] = split_name

    output_records = []
    for record in records:
        record["split"] = assignments[record["sample_id"]]
        output_records.append(record)

    manifest = {
        "manifest_version": 1,
        "splitting": splitting,
        "seed": seed,
        "ratios": ratios,
        "records": output_records,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Split manifest has no records")
    sample_ids = [record.get("sample_id") for record in records]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Split manifest sample_id values must be unique and non-empty")

    by_split: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for record in records:
        split = record.get("split")
        if split not in by_split:
            raise ValueError(f"Unknown split in manifest: {split}")
        by_split[split].add(record["sample_id"])

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            if by_split[left] & by_split[right]:
                raise AssertionError(f"Sample IDs overlap between {left} and {right}")

    for field in ("canonical_smiles", "scaffold"):
        locations: dict[str, str] = {}
        for record in records:
            value = record.get(field)
            if not value:
                continue
            split = record["split"]
            previous = locations.setdefault(value, split)
            if previous != split:
                raise AssertionError(f"{field} crosses split: {value}")


def write_manifest(manifest: dict, path: str | Path) -> None:
    validate_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def manifest_hash(manifest: dict) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
