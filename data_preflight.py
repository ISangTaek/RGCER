"""Pre-training validation of preprocessed graph data directories.

Hard failures (missing directory, missing/inconsistent manifest, empty
task/split) raise :class:`DataPreflightError` before any model is built.
Coverage concerns (thin calibration, acyclic molecules missing from
evaluation splits, one scaffold dominating a split, unexpected directories)
are collected as warnings so a known limitation is visible without aborting.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

from split_manifest import ACYCLIC_SCAFFOLD, SPLIT_NAMES, load_manifest
from toxacute_datastore import ToxAcuteDataStore


class DataPreflightError(ValueError):
    """Preprocessed data is missing or inconsistent with the requested run."""


def _manifest_distribution_stats(manifest: dict) -> dict:
    """Global split/group statistics computed straight from manifest records."""

    records = manifest["records"]
    split_counts = Counter(record["split"] for record in records)
    total = max(len(records), 1)
    ratios = manifest.get("ratios", {})
    actual_ratios = {name: round(split_counts.get(name, 0) / total, 6) for name in SPLIT_NAMES}
    ratio_deviation = {
        name: round(
            100.0 * (split_counts.get(name, 0) / total - float(ratios.get(name, 0.0))),
            4,
        )
        for name in SPLIT_NAMES
    }

    acyclic = [record for record in records if record.get("scaffold") == ACYCLIC_SCAFFOLD]
    acyclic_by_split = Counter(record["split"] for record in acyclic)

    def _group_key(record):
        # Manifest v2 records carry the authoritative grouping key directly;
        # v1 manifests re-derive it so legacy trees stay auditable.
        stored = record.get("split_group")
        if stored:
            return str(stored)
        return str(record.get("canonical_smiles") or record["sample_id"])

    group_sizes = Counter(_group_key(record) for record in records)
    largest_group_name, largest_group_size = ("", 0)
    if group_sizes:
        largest_group_name, largest_group_size = group_sizes.most_common(1)[0]
    largest_group_splits = Counter(
        record["split"] for record in records if _group_key(record) == largest_group_name
    )
    largest_group_split, largest_in_split = ("", 0)
    if largest_group_splits:
        largest_group_split, largest_in_split = largest_group_splits.most_common(1)[0]

    canonical_counts = Counter(
        str(record.get("canonical_smiles") or "") for record in records
    )
    del canonical_counts[""]

    largest_group_by_split: dict[str, dict] = {}
    for split_name in ("validation", "calibration", "test"):
        placed = Counter()
        for record in records:
            if record.get("split") != split_name:
                continue
            key = str(record.get("split_group") or record.get("canonical_smiles") or record["sample_id"])
            placed[key] += 1
        if not placed:
            continue
        largest_key, largest_size = max(placed.items(), key=lambda item: item[1])
        largest_group_by_split[split_name] = {
            "key": largest_key,
            "size": int(largest_size),
            "fraction": round(largest_size / max(split_counts.get(split_name, 0), 1), 4),
        }

    return {
        "actual_split_counts": {name: int(split_counts.get(name, 0)) for name in SPLIT_NAMES},
        "actual_split_ratios": actual_ratios,
        "ratio_deviation_pct": ratio_deviation,
        "acyclic_count": len(acyclic),
        "acyclic_by_split": {name: int(acyclic_by_split.get(name, 0)) for name in SPLIT_NAMES},
        "largest_split_group": {
            "key": largest_group_name,
            "size": int(largest_group_size),
            "split": largest_group_split,
            "fraction_of_split": round(
                largest_in_split / max(split_counts.get(largest_group_split, 0), 1), 4
            ),
        },
        "largest_group_by_split": largest_group_by_split,
        "num_unique_canonical_smiles": len(canonical_counts),
        "num_duplicate_canonical_groups": sum(
            1 for count in canonical_counts.values() if count > 1
        ),
        "duplicate_canonical_row_count": sum(
            count for count in canonical_counts.values() if count > 1
        ),
    }


def _acyclic_concentration_error(distribution: Mapping) -> str | None:
    """Formal scaffold runs must not strand acyclic chemistry out of evals.

    Review §37 upgrades the old "two missing splits" rule: with a meaningful
    acyclic population, *any* evaluation split without acyclic chemistry is
    a hard failure.
    """

    if int(distribution.get("acyclic_count", 0)) < 100:
        return None
    by_split = distribution.get("acyclic_by_split", {})
    missing = [
        name
        for name in ("validation", "calibration", "test")
        if by_split.get(name, 0) == 0
    ]
    if missing:
        present = [name for name in SPLIT_NAMES if by_split.get(name, 0)]
        return (
            f"FAIL_ACYCLIC_MISSING_EVAL: {distribution['acyclic_count']} acyclic "
            f"molecules never reach {', '.join(missing)} "
            f"(present in {', '.join(present) or 'nowhere'})"
        )
    return None


def _eval_dominance_errors(distribution: Mapping) -> list[str]:
    """Any evaluation split whose largest group owns >=50% is a hard failure."""

    failures = []
    largest = distribution.get("largest_group_by_split") or {}
    for split_name in ("validation", "calibration", "test"):
        entry = largest.get(split_name)
        if not entry:
            continue
        if float(entry.get("fraction", 0.0)) >= 0.50:
            failures.append(
                f"FAIL_EVAL_GROUP_DOMINANCE:{split_name}:{entry.get('key')}="
                f"{float(entry['fraction']):.2f}"
            )
    return failures


def _conformal_rank_failures(
    task_rows: Sequence[Mapping],
    *,
    conformal_task_names: Sequence[str],
    conformal_alpha: float,
) -> list[str]:
    """Split-conformal finite-rank floors per scoped endpoint (review §37)."""

    if not conformal_task_names:
        return []
    from conformal import minimum_calibration_size

    minimum = minimum_calibration_size(conformal_alpha)
    counts = {row["task"]: int(row.get("calibration", 0)) for row in task_rows}
    failures = []
    for task in conformal_task_names:
        count = counts.get(task)
        if count is None:
            continue
        if count < minimum:
            failures.append(
                f"FAIL_CONFORMAL_RANK:{task}:calibration={count}:minimum={minimum}"
            )
    return failures


def run_datastore_preflight(
    store_or_root,
    task_names,
    *,
    max_nodes_filter=None,
    min_calibration_size: int = 30,
    require_calibration: bool = True,
    ratio_tolerance_pct: float = 5.0,
    conformal_alpha: float | None = None,
    conformal_task_names=None,
    priority_task_names=None,
) -> dict:
    """Validate V2 task coverage before a model or DataLoader is created.

    Alpha-aware calibration rules (review §37): scoped endpoints below
    ``minimum_calibration_size(alpha)`` are hard failures; counts between the
    floor and the 30-sample stability threshold only warn.
    """

    from conformal import minimum_calibration_size

    scoped_conformal = [str(name) for name in (conformal_task_names or [])]
    priority_names = [str(name) for name in (priority_task_names or [])]

    store = store_or_root if isinstance(store_or_root, ToxAcuteDataStore) else ToxAcuteDataStore.resolve(store_or_root)
    try:
        store.validate(strict=False, expected_task_names=task_names)
        rows = []
        failures: list[str] = []
        minimum_rank = (
            minimum_calibration_size(conformal_alpha) if conformal_alpha else None
        )
        for task_name in task_names:
            all_indices = store.get_task_indices(task_name, split=None)
            counts = {
                "train": len(store.get_task_indices(task_name, split="train", max_nodes=max_nodes_filter)),
                "validation": len(store.get_task_indices(task_name, split="validation", max_nodes=max_nodes_filter)),
                "calibration": len(store.get_task_indices(task_name, split="calibration", max_nodes=max_nodes_filter)),
                "test": len(store.get_task_indices(task_name, split="test", max_nodes=max_nodes_filter)),
            }
            excluded_nodes = 0
            if max_nodes_filter is not None:
                excluded_nodes = int(np.sum(store.num_nodes[all_indices] > int(max_nodes_filter)))
            status = "OK"
            if task_name in scoped_conformal and minimum_rank is not None and counts["calibration"] < minimum_rank:
                status = f"FAIL_CONFORMAL_RANK"
            elif counts["train"] == 0:
                status = "FAIL_EMPTY_TRAIN"
            elif counts["validation"] == 0:
                status = "FAIL_EMPTY_VAL"
            elif require_calibration and counts["calibration"] == 0:
                status = "FAIL_EMPTY_CAL"
            elif counts["test"] == 0:
                status = "FAIL_EMPTY_TEST"
            elif task_name in scoped_conformal and counts["calibration"] < int(min_calibration_size):
                status = "LOW_CALIBRATION_STABILITY"
            elif counts["calibration"] < int(min_calibration_size):
                status = "LOW_CALIBRATION"
            if status.startswith("FAIL_"):
                failures.append(f"{task_name}={status}")
            rows.append(
                {
                    "task": task_name,
                    **counts,
                    "excluded_nodes": excluded_nodes,
                    "low_calibration": counts["calibration"] < int(min_calibration_size),
                    "conformal_scoped": task_name in scoped_conformal,
                    "status": status,
                }
            )
        metadata = store.metadata
        report_warnings: list[str] = []

        manifest_path = store.root / "split_manifest.json"
        distribution = {}
        if manifest_path.exists():
            distribution = _manifest_distribution_stats(load_manifest(manifest_path))
            deviations = distribution.get("ratio_deviation_pct", {})
            for split_name, deviation in deviations.items():
                if abs(float(deviation)) > float(ratio_tolerance_pct):
                    report_warnings.append(
                        f"actual {split_name} share deviates by {deviation:+.1f}pp "
                        f"from the configured ratio ({distribution['actual_split_ratios'][split_name]:.3f})"
                    )
        concentration_error = _acyclic_concentration_error(distribution)
        if concentration_error:
            failures.append(concentration_error)
        failures.extend(_eval_dominance_errors(distribution))
        failures.extend(
            _conformal_rank_failures(
                rows,
                conformal_task_names=scoped_conformal,
                conformal_alpha=conformal_alpha or 0.10,
            )
        )
        if priority_names:
            per_task = {row["task"]: row for row in rows}
            for task in priority_names:
                entry = per_task.get(task)
                if entry is None:
                    continue
                if entry["validation"] < 10 or entry["test"] < 10:
                    failures.append(
                        f"FAIL_PRIORITY_COVERAGE:{task}:validation={entry['validation']}:"
                        f"test={entry['test']}:minimum=10"
                    )
        if failures:
            raise DataPreflightError(
                "DataStore V2 formal preflight failed (" + "; ".join(failures) + ")"
            )
        return {
            "format": "toxacute_datastore",
            "format_version": 2,
            "build_id": store.build_id,
            "datastore_fingerprint": store.fingerprint,
            "split_manifest_hash": store.split_manifest_hash,
            "raw_csv_sha256": store.raw_csv_sha256,
            "feature_schema_version": metadata.get("feature_schema_version"),
            "graph_layout": metadata.get("graph_layout"),
            "max_path_distance": metadata.get("max_path_distance"),
            "conformal_scope_tasks": scoped_conformal,
            "conformal_alpha": conformal_alpha,
            "priority_task_names": priority_names,
            "task_names": list(task_names),
            "max_nodes_filter": max_nodes_filter,
            **distribution,
            "tasks": rows,
            "warnings": report_warnings,
            "status": "OK",
        }
    finally:
        if not isinstance(store_or_root, ToxAcuteDataStore):
            store.close()


def run_data_preflight(
    base_dir,
    task_names,
    *,
    splitting: str = "scaffold",
    valid_size: float = 0.1,
    calibration_size: float = 0.1,
    test_size: float = 0.1,
    split_seed: int = 42,
    min_calibration_size: int = 30,
) -> dict:
    """Check the preprocessed tree against the requested run configuration.

    Returns a JSON-serializable report: per-task split counts, global split
    sizes, scaffold concentration, acyclic coverage, and a warning list.
    """

    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise DataPreflightError(f"Preprocessed data directory not found: {base_dir}")
    manifest_path = base_dir / "split_manifest.json"
    if not manifest_path.exists():
        raise DataPreflightError(
            f"Global split manifest not found: {manifest_path}. Run preprocess_data.py first."
        )
    manifest = load_manifest(manifest_path)
    if manifest.get("splitting") != splitting:
        raise DataPreflightError(
            f"Manifest splitting={manifest.get('splitting')!r} does not match requested {splitting!r}; "
            "re-run preprocess_data.py with the matching --splitting"
        )
    expected_ratios = {
        "train": 1.0 - valid_size - calibration_size - test_size,
        "validation": valid_size,
        "calibration": calibration_size,
        "test": test_size,
    }
    actual_ratios = manifest.get("ratios", {})
    for name, expected in expected_ratios.items():
        actual = actual_ratios.get(name)
        if actual is None or abs(float(actual) - float(expected)) > 1e-6:
            raise DataPreflightError(
                f"Manifest ratio for {name!r} ({actual!r}) does not match the requested split "
                f"configuration ({expected!r}); re-run preprocess_data.py with matching sizes"
            )
    if int(manifest.get("seed")) != int(split_seed):
        raise DataPreflightError(
            f"Manifest seed={manifest.get('seed')!r} does not match requested split_seed={split_seed!r}; "
            "pass --split_seed matching the preprocessing seed or re-preprocess"
        )

    records = manifest["records"]
    split_of_row = {int(record["row_index"]): record["split"] for record in records}
    split_sizes = Counter(record["split"] for record in records)
    warnings: list[str] = []
    distribution = _manifest_distribution_stats(manifest)
    for split_name, deviation in distribution["ratio_deviation_pct"].items():
        if abs(float(deviation)) > 5.0:
            warnings.append(
                f"actual {split_name} share deviates by {deviation:+.1f}pp "
                f"from the configured ratio ({distribution['actual_split_ratios'][split_name]:.3f})"
            )
    concentration_error = _acyclic_concentration_error(distribution)
    if concentration_error:
        raise DataPreflightError(concentration_error)

    scaffold_sizes = Counter(
        record["scaffold"] for record in records if record.get("scaffold")
    )
    largest_groups = []
    for scaffold, size in scaffold_sizes.most_common(5):
        group_splits = Counter(
            record["split"] for record in records if record.get("scaffold") == scaffold
        )
        dominant, dominant_count = group_splits.most_common(1)[0]
        largest_groups.append(
            {
                "scaffold": scaffold,
                "size": size,
                "split": dominant,
                "fraction_of_split": round(
                    dominant_count / max(split_sizes.get(dominant, 1), 1), 4
                ),
            }
        )
    for group in largest_groups:
        if group["split"] != "train" and group["fraction_of_split"] > 0.5:
            warnings.append(
                f"scaffold {group['scaffold']!r} dominates {group['split']} "
                f"({group['fraction_of_split']:.0%} of {split_sizes[group['split']]} samples)"
            )

    acyclic = [record for record in records if record.get("scaffold") == ACYCLIC_SCAFFOLD]
    acyclic_by_split = Counter(record["split"] for record in acyclic)
    if acyclic:
        missing = [
            name
            for name in ("validation", "calibration", "test")
            if acyclic_by_split.get(name, 0) == 0
        ]
        if missing:
            present = [name for name in SPLIT_NAMES if acyclic_by_split.get(name, 0) > 0]
            warnings.append(
                f"{len(acyclic)} acyclic molecules are absent from {', '.join(missing)} "
                f"(acyclic chemistry appears only in {', '.join(present)}); "
                "evaluation coverage excludes acyclic chemistry for those splits"
            )

    tasks_report: dict[str, dict[str, int]] = {}
    for task in task_names:
        task_dir = base_dir / str(task)
        if not task_dir.is_dir():
            raise DataPreflightError(f"Preprocessed task directory not found: {task_dir}")
        per_split: Counter = Counter()
        for path in task_dir.glob("data_*.pt"):
            parts = path.stem.split("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                warnings.append(f"{task}: unrecognized data file name {path.name}")
                continue
            row_index = int(parts[1])
            split = split_of_row.get(row_index)
            if split is None:
                warnings.append(
                    f"{task}: {path.name} (row_index={row_index}) has no manifest record"
                )
                continue
            per_split[split] += 1
        empty_splits = [name for name in SPLIT_NAMES if per_split.get(name, 0) == 0]
        if empty_splits:
            raise DataPreflightError(
                f"Task {task!r} has no samples in: {', '.join(empty_splits)} "
                f"(found {dict(per_split)})"
            )
        if per_split["calibration"] < min_calibration_size:
            warnings.append(
                f"Task {task!r} has only {per_split['calibration']} calibration samples "
                f"(< min_calibration_size={min_calibration_size}); "
                "conformal calibration will fall back for this task"
            )
        tasks_report[str(task)] = {name: int(per_split.get(name, 0)) for name in SPLIT_NAMES}

    task_dir_names = {str(task) for task in task_names}
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir() and entry.name not in task_dir_names:
            warnings.append(f"Unexpected directory in preprocessed data: {entry.name}")

    return {
        "base_dir": str(base_dir),
        "manifest_path": str(manifest_path),
        "manifest_records": len(records),
        "splitting": splitting,
        "seed": int(manifest["seed"]),
        "manifest_version": int(manifest.get("manifest_version", 1)),
        "split_sizes": {name: int(split_sizes.get(name, 0)) for name in SPLIT_NAMES},
        **distribution,
        "largest_scaffold_groups": largest_groups,
        "tasks": tasks_report,
        "warnings": warnings,
    }
