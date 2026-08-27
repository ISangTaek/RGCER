"""Global, group-safe train/validation/calibration/test manifests.

Manifest version 3 (current) is the constrained scaffold splitter:
oversized scaffold groups are pinned to train, and the remaining groups are
placed by a deterministic multi-candidate greedy pass that balances overall
size, endpoint *label presence* (never label values), human-priority
coverage, acyclic chemistry, and conformal calibration feasibility, subject
to hard constraints (scaffold/canonical isolation, evaluation acyclic
coverage, no >50% single group in any evaluation split).  Version 2 grouped
acyclic chemistry by canonical SMILES only; version 1 grouped everything by
canonical SMILES.  Both legacy versions stay validatable for audit, but the
formal DataStore accepts manifests generated with version 3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SPLIT_NAMES = ("train", "validation", "calibration", "test")
EVAL_SPLITS = ("validation", "calibration", "test")
DEFAULT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "test": 0.10,
}
MANIFEST_VERSION = 3
SUPPORTED_MANIFEST_VERSIONS = {1, 2, 3}
ACYCLIC_SCAFFOLD = "__ACYCLIC__"
V3_SPLIT_ALGORITHM = "constrained_scaffold_v3"
RANDOM_SPLIT_ALGORITHM = "canonical_size_greedy_v1"

# The >50% single-group rule targets full-scale evaluation splits; tiny
# synthetic splits (a handful of molecules) cannot satisfy it in principle.
_MIN_EVAL_TARGET_FOR_DOMINANCE_RULE = 12.0

# Soft objective weights; priority endpoints must dominate.
_V3_W_SIZE = 1.0
_V3_W_ALL_LABELS = 1.0
_V3_W_PRIORITY = 5.0
_V3_W_ACYCLIC = 2.0
_EVAL_DOMINANCE_LIMIT = 0.50
_MIN_ACYCLIC_FOR_COVERAGE_RULE = 100

_EPS = 1e-12


def split_group_key(record: dict, splitting: str) -> str:
    """Group key deciding which chemistry must stay inside one split.

    Scaffold splitting groups cyclic molecules by scaffold and acyclic
    molecules by canonical SMILES; random splitting always groups by
    canonical SMILES.
    """

    canonical = str(record.get("canonical_smiles") or record["sample_id"])
    if splitting == "scaffold":
        scaffold = str(record.get("scaffold") or "")
        if scaffold and scaffold != ACYCLIC_SCAFFOLD:
            return f"scaffold::{scaffold}"
        return f"acyclic::{canonical}"
    return f"canonical::{canonical}"


def _group_records(records: list[dict], splitting: str):
    groups: dict[str, list[dict]] = {}
    for record in records:
        group_key = split_group_key(record, splitting)
        record["split_group"] = group_key
        groups.setdefault(group_key, []).append(record)
    return groups


def _target_vector(ratios: dict[str, float], totals: np.ndarray | float | None):
    """Per-split targets for one accounting column."""

    if totals is None:
        return {name: 0.0 for name in SPLIT_NAMES}
    total = float(totals)
    return {name: ratios[name] * total for name in SPLIT_NAMES}


def _sq_deviation(projected: float, target: float) -> float:
    if target <= _EPS:
        return 0.0 if projected <= _EPS else float("inf")
    value = (projected - target) / max(target, 1.0)
    return float(value * value)


class _PlannerState:
    """Running per-split tallies used by the constrained greedy pass."""

    def __init__(
        self,
        ratios: dict[str, float],
        num_samples: int,
        label_width: int,
        priority_width: int = 0,
    ):
        self.ratios = ratios
        self.num_samples = int(num_samples)
        self.label_width = label_width
        self.priority_width = priority_width
        # Absolute per-split targets; ratio alone is a share, not a count.
        self.size_targets = {name: ratios[name] * float(num_samples) for name in SPLIT_NAMES}
        self.sizes = {name: 0 for name in SPLIT_NAMES}
        self.labels = {name: np.zeros(label_width, dtype=np.int64) for name in SPLIT_NAMES}
        self.priority = {name: np.zeros(priority_width, dtype=np.int64) for name in SPLIT_NAMES}
        self.acyclic = {name: 0 for name in SPLIT_NAMES}
        self.group_sizes: dict[str, dict[str, int]] = {name: {} for name in SPLIT_NAMES}

    def add(self, split_name: str, size: int, labels: np.ndarray, acyclic: int, group_key: str, priority=None):
        self.sizes[split_name] += size
        if size:
            self.labels[split_name] += labels
            if priority is not None and self.priority_width:
                self.priority[split_name] += priority
        self.acyclic[split_name] += acyclic
        self.group_sizes[split_name][group_key] = self.group_sizes[split_name].get(group_key, 0) + size

    def _component_deviation(self, projected, target):
        return _sq_deviation(float(projected), float(target))

    def placement_delta(
        self,
        split_name: str,
        size: int,
        labels: np.ndarray,
        acyclic: int,
        *,
        label_targets: dict[str, np.ndarray] | None,
        priority_targets: dict[str, np.ndarray] | None,
        priority_vector: np.ndarray | None,
        acyclic_total: int,
    ) -> float:
        """Change in the global soft score if this group joins ``split_name``.

        Comparing *marginal* scores (not each split's absolute deviation) is
        what lets an under-filled evaluation split win over a large split
        that is already near — or past — its target.
        """

        ratio = self.ratios[split_name]
        if ratio <= 0:
            return float("inf")

        target = self.size_targets[split_name]
        old, new = self.sizes[split_name], self.sizes[split_name] + size
        delta = _V3_W_SIZE * (
            self._component_deviation(new, target) - self._component_deviation(old, target)
        )
        if self.label_width:
            old_vec = self.labels[split_name]
            new_vec = old_vec + labels
            drifts = []
            for t in range(self.label_width):
                if label_targets[split_name][t] > 0:
                    drifts.append(
                        self._component_deviation(new_vec[t], label_targets[split_name][t])
                        - self._component_deviation(old_vec[t], label_targets[split_name][t])
                    )
            if drifts:
                delta += _V3_W_ALL_LABELS * float(np.mean(drifts))
        if self.priority_width and priority_targets is not None and priority_vector is not None:
            old_pri = self.priority[split_name]
            new_pri = old_pri + priority_vector
            drifts = []
            for t in range(self.priority_width):
                if priority_targets[split_name][t] > 0:
                    drifts.append(
                        self._component_deviation(new_pri[t], priority_targets[split_name][t])
                        - self._component_deviation(old_pri[t], priority_targets[split_name][t])
                    )
            if drifts:
                delta += _V3_W_PRIORITY * float(np.mean(drifts))
        acy_target = ratio * acyclic_total
        old_acy = self.acyclic[split_name]
        delta += _V3_W_ACYCLIC * (
            self._component_deviation(old_acy + acyclic, acy_target)
            - self._component_deviation(old_acy, acy_target)
        )
        return delta

    def soft_score(self, *, label_targets, priority_targets, acyclic_total: int) -> float:
        score = 0.0
        for name in SPLIT_NAMES:
            score += _V3_W_SIZE * self._component_deviation(
                self.sizes[name], self.size_targets[name]
            )
            if self.label_width:
                deviations = [
                    self._component_deviation(self.labels[name][t], label_targets[name][t])
                    for t in range(self.label_width)
                    if label_targets[name][t] > 0
                ]
                if deviations:
                    score += _V3_W_ALL_LABELS * float(np.mean(deviations))
            if self.priority_width:
                deviations = [
                    self._component_deviation(self.priority[name][t], priority_targets[name][t])
                    for t in range(self.priority_width)
                    if priority_targets[name][t] > 0
                ]
                if deviations:
                    score += _V3_W_PRIORITY * float(np.mean(deviations))
            score += _V3_W_ACYCLIC * self._component_deviation(
                self.acyclic[name], self.ratios[name] * acyclic_total
            )
        return score




def _hard_constraint_failures(
    state: _PlannerState,
    *,
    assignments: dict[str, str],
    ratios: dict[str, float],
    acyclic_total: int,
    priority_keys: Sequence[int],
    conformal_columns: Sequence[int],
    task_names: Sequence[str],
    group_meta: dict[str, dict],
    minimum_calibration: int,
    min_priority_eval_count: int,
    enforce_dominance: bool = True,
) -> list[str]:
    failures: list[str] = []
    active_evals = [name for name in EVAL_SPLITS if ratios[name] > 0]

    for name in active_evals:
        if state.sizes[name] == 0:
            failures.append(f"{name}_empty")
        if acyclic_total >= _MIN_ACYCLIC_FOR_COVERAGE_RULE and state.acyclic[name] == 0:
            failures.append(f"FAIL_ACYCLIC_MISSING_EVAL:{name}")
        placed = state.group_sizes[name]
        if placed and enforce_dominance:
            largest_key, largest_size = max(placed.items(), key=lambda item: item[1])
            fraction = largest_size / max(state.sizes[name], 1)
            if fraction >= _EVAL_DOMINANCE_LIMIT:
                failures.append(
                    f"FAIL_EVAL_GROUP_DOMINANCE:{name}:{largest_key}={fraction:.2f}"
                )

    # Priority and conformal coverage are accumulated from per-group label
    # metadata so constraints always reflect exactly which molecules landed
    # in each split.
    def totals_for(columns: Sequence[int]) -> dict[str, np.ndarray]:
        width = len(columns)
        totals = {name: np.zeros(max(width, 1), dtype=np.int64) for name in SPLIT_NAMES}
        for group_key, meta in group_meta.items():
            split_name = assignments[group_key]
            if width:
                totals[split_name][:width] += np.asarray(meta["labels"])[list(columns)]
        return totals

    if len(priority_keys):
        priority_totals = totals_for(priority_keys)
        minimum_by_split = {
            "validation": min_priority_eval_count,
            "test": min_priority_eval_count,
        }
        for split_name, minimum in minimum_by_split.items():
            if ratios[split_name] <= 0:
                continue
            for offset, column in enumerate(priority_keys):
                count = int(priority_totals[split_name][offset])
                if count < minimum:
                    failures.append(
                        f"FAIL_PRIORITY_COVERAGE:task={task_names[column]}:"
                        f"split={split_name}:count={count}:minimum={minimum}"
                    )

    if len(conformal_columns):
        conformal_totals = totals_for(conformal_columns)["calibration"]
        for offset, column in enumerate(conformal_columns):
            count = int(conformal_totals[offset])
            if count < minimum_calibration:
                failures.append(
                    f"FAIL_CONFORMAL_RANK:task={task_names[column]}:split=calibration:"
                    f"count={count}:minimum={minimum_calibration}"
                )
    return failures


def create_split_manifest(
    records: Iterable[dict],
    *,
    splitting: str = "scaffold",
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    task_names: Sequence[str] | None = None,
    label_presence=None,
    priority_task_names: Sequence[str] | None = None,
    conformal_task_names: Sequence[str] | None = None,
    conformal_alpha: float = 0.10,
    num_candidates: int = 64,
    oversized_eval_fraction: float = 0.5,
    min_priority_eval_count: int = 10,
    source_csv_sha256: str | None = None,
    conformal_scope: str = "human3",
    enforce_scale_constraints: bool = True,
) -> dict:
    """Create a split manifest.

    ``random`` splitting keeps the simple size-greedy assignment (no labels
    used).  ``scaffold`` splitting uses the constrained v3 planner described
    in :data:`V3_SPLIT_ALGORITHM`; it never splits one scaffold across splits
    and refuses — loudly — when every candidate violates a hard constraint.

    ``enforce_scale_constraints`` exists for unit tests that exercise the
    grouping/isolation semantics on toy corpora (two groups cannot fill three
    evaluation splits); formal callers must keep the default.
    """

    from conformal import minimum_calibration_size

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

    groups = _group_records(records, splitting)

    if splitting == "random":
        assignments = _greedy_size_assignment(groups, ratios, seed)
    else:
        assignments = _constrained_scaffold_assignment(
            groups,
            records=records,
            ratios=ratios,
            seed=seed,
            task_names=list(task_names) if task_names else [],
            label_presence=label_presence,
            priority_task_names=list(priority_task_names) if priority_task_names else [],
            conformal_task_names=list(conformal_task_names) if conformal_task_names else [],
            conformal_alpha=float(conformal_alpha),
            num_candidates=int(num_candidates),
            oversized_eval_fraction=float(oversized_eval_fraction),
            min_priority_eval_count=int(min_priority_eval_count),
            enforce_scale_constraints=bool(enforce_scale_constraints),
        )

    output_records = []
    for record in records:
        record["split"] = assignments[record["split_group"]]
        output_records.append(record)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "split_algorithm": V3_SPLIT_ALGORITHM if splitting == "scaffold" else RANDOM_SPLIT_ALGORITHM,
        "splitting": splitting,
        "seed": seed,
        "ratios": ratios,
        "records": output_records,
    }
    if splitting == "scaffold":
        manifest["source_csv_sha256"] = source_csv_sha256
        manifest["constraints"] = {
            "oversized_eval_fraction": float(oversized_eval_fraction),
            "num_candidates": int(num_candidates),
            "min_priority_eval_count": int(min_priority_eval_count),
            "conformal_scope": conformal_scope,
            "conformal_alpha": float(conformal_alpha),
            "priority_tasks": list(priority_task_names or []),
            "conformal_tasks": list(conformal_task_names or []),
        }
    validate_manifest(manifest)
    return manifest


def _greedy_size_assignment(groups: dict[str, list[dict]], ratios: dict[str, float], seed: int):
    rng = np.random.default_rng(seed)
    ordered = list(groups.items())
    rng.shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)

    num_samples = sum(len(members) for _, members in ordered)
    target = {name: ratios[name] * num_samples for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    assignments: dict[str, str] = {}
    for group_key, members in ordered:
        size = len(members)
        best_split, best_score = None, None
        for split_name in SPLIT_NAMES:
            projected = counts[split_name] + size
            deviation = _sq_deviation(projected, target[split_name])
            if best_score is None or deviation < best_score:
                best_split, best_score = split_name, deviation
        counts[best_split] += size
        assignments[group_key] = best_split
    return assignments


def _constrained_scaffold_assignment(
    groups: dict[str, list[dict]],
    *,
    records: list[dict],
    ratios: dict[str, float],
    seed: int,
    task_names: Sequence[str],
    label_presence,
    priority_task_names: Sequence[str],
    conformal_task_names: Sequence[str],
    conformal_alpha: float,
    num_candidates: int,
    oversized_eval_fraction: float,
    min_priority_eval_count: int,
    enforce_scale_constraints: bool = True,
) -> dict[str, str]:
    from conformal import minimum_calibration_size

    if (priority_task_names or conformal_task_names) and label_presence is None:
        raise ValueError(
            "label_presence is required when priority/conformal task coverage "
            "constraints are requested"
        )

    label_matrix: np.ndarray | None = None
    if label_presence is not None:
        label_matrix = np.asarray(label_presence, dtype=np.int64)
        if label_matrix.shape != (len(records), len(task_names)):
            raise ValueError(
                f"label_presence shape {label_matrix.shape} does not match "
                f"[{len(records)}, {len(task_names)}]"
            )

    num_samples = len(records)
    eval_ratios = {name: float(ratios[name]) for name in EVAL_SPLITS if ratios[name] > 0}
    if not eval_ratios:
        raise ValueError("At least one evaluation split must have a positive ratio")
    smallest_eval_target = min(eval_ratios[name] * num_samples for name in eval_ratios)
    oversized_threshold = oversized_eval_fraction * smallest_eval_target
    minimum_calibration = minimum_calibration_size(conformal_alpha)
    enforce_dominance = smallest_eval_target >= _MIN_EVAL_TARGET_FOR_DOMINANCE_RULE

    priority_columns = [task_names.index(name) for name in priority_task_names if name in task_names]
    missing_priority = [name for name in priority_task_names if name not in task_names]
    if missing_priority:
        raise ValueError(f"priority tasks absent from task_names: {missing_priority}")
    conformal_columns = [
        task_names.index(name)
        for name in conformal_task_names
        if name in task_names
    ]
    label_index_of_record = {id(record): index for index, record in enumerate(records)}

    group_meta: dict[str, dict] = {}
    for group_key, members in groups.items():
        member_indices = [label_index_of_record[id(record)] for record in members]
        labels = (
            label_matrix[member_indices].sum(axis=0) if label_matrix is not None else np.zeros(0, dtype=np.int64)
        )
        group_meta[group_key] = {
            "members": members,
            "size": len(members),
            "labels": labels,
            "acyclic": sum(
                1
                for record in members
                if str(record.get("scaffold") or "") == ACYCLIC_SCAFFOLD
            ),
            "priority": labels[priority_columns] if len(priority_columns) else np.zeros(0, dtype=np.int64),
            "conformal": labels[conformal_columns] if len(conformal_columns) else np.zeros(1, dtype=np.int64),
        }

    # The oversized-pinning rule mirrors the c1ccccc1-at-scale problem; on a
    # toy corpus every group would exceed the threshold, so apply it only for
    # realistically sized evaluation targets.
    enforce_oversize_guard = smallest_eval_target >= _MIN_EVAL_TARGET_FOR_DOMINANCE_RULE
    if enforce_oversize_guard:
        forced_train = sorted(
            key for key, meta in group_meta.items() if meta["size"] > oversized_threshold
        )
    else:
        forced_train = []
    remaining = [key for key in group_meta if key not in set(forced_train)]

    total_acyclic = sum(meta["acyclic"] for meta in group_meta.values())
    use_labels = label_matrix is not None
    label_width = len(task_names) if use_labels else 0
    priority_width = len(priority_columns)
    label_targets = (
        {
            name: ratios[name] * label_matrix.sum(axis=0).astype(np.float64)
            for name in SPLIT_NAMES
        }
        if use_labels
        else None
    )
    priority_targets = (
        {
            name: ratios[name] * label_matrix[:, priority_columns].sum(axis=0).astype(np.float64)
            for name in SPLIT_NAMES
        }
        if priority_width
        else None
    )

    def check(assignments: dict[str, str], state: _PlannerState) -> list[str]:
        if not enforce_scale_constraints:
            # Test-only escape hatch; group atomicity is still guaranteed by
            # construction.
            return []
        return _hard_constraint_failures(
            state,
            assignments=assignments,
            ratios=ratios,
            acyclic_total=total_acyclic,
            priority_keys=priority_columns,
            conformal_columns=conformal_columns,
            task_names=task_names,
            group_meta=group_meta,
            minimum_calibration=minimum_calibration,
            min_priority_eval_count=min_priority_eval_count,
            enforce_dominance=enforce_dominance,
        )

    best_candidate = None
    all_failures: list[list[str]] = []

    for candidate_index in range(max(1, num_candidates)):
        rng = np.random.default_rng(seed + candidate_index)
        jitter = {key: float(rng.random()) for key in remaining}

        def sort_key(group_key):
            meta = group_meta[group_key]
            priority_score = float(meta["priority"].sum()) if priority_width else 0.0
            label_score = float(meta["labels"].sum()) if use_labels else 0.0
            return (-priority_score, -label_score, -meta["size"], jitter[group_key])

        order = sorted(remaining, key=sort_key)

        state = _PlannerState(ratios, num_samples, label_width, priority_width)
        assignments: dict[str, str] = {}

        for group_key in forced_train:
            meta = group_meta[group_key]
            state.add(
                "train",
                meta["size"],
                meta["labels"],
                meta["acyclic"],
                group_key,
                priority=meta["priority"],
            )
            assignments[group_key] = "train"

        for group_key in order:
            meta = group_meta[group_key]
            scored = [
                (
                    state.placement_delta(
                        split_name,
                        meta["size"],
                        meta["labels"],
                        meta["acyclic"],
                        label_targets=label_targets,
                        priority_targets=priority_targets,
                        priority_vector=meta["priority"],
                        acyclic_total=total_acyclic,
                    ),
                    split_name,
                )
                for split_name in SPLIT_NAMES
            ]
            finite = [(cost, name) for cost, name in scored if np.isfinite(cost)]
            chosen = min(finite)[1] if finite else "train"
            state.add(
                chosen,
                meta["size"],
                meta["labels"],
                meta["acyclic"],
                group_key,
                priority=meta["priority"],
            )
            assignments[group_key] = chosen

        failures = check(assignments, state)
        if failures:
            all_failures.append(failures)
            continue
        score = state.soft_score(
            label_targets=label_targets,
            priority_targets=priority_targets,
            acyclic_total=total_acyclic,
        )
        if best_candidate is None or score < best_candidate[0]:
            best_candidate = (score, assignments)

    if best_candidate is None:
        flattened = sorted({failure for run in all_failures[-4:] for failure in run})
        raise ValueError(
            "Constrained scaffold split found no feasible candidate in "
            f"{num_candidates} runs; hard-constraint failures observed: "
            f"{flattened}. Scaffold atomicity cannot be relaxed to fix this — "
            "adjust split ratios or priority expectations instead."
        )
    return best_candidate[1]


def validate_manifest(manifest: dict) -> None:
    splitting = manifest.get("splitting")
    if splitting not in {"random", "scaffold"}:
        raise ValueError(f"Unsupported splitting method in manifest: {splitting!r}")
    ratios = manifest.get("ratios")
    if not isinstance(ratios, dict) or set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Manifest ratios must contain exactly {SPLIT_NAMES}")
    if any(float(ratios[name]) < 0 for name in SPLIT_NAMES) or not np.isclose(
        sum(float(ratios[name]) for name in SPLIT_NAMES), 1.0
    ):
        raise ValueError("Manifest split ratios must be non-negative and sum to 1")
    if "seed" not in manifest:
        raise ValueError("Split manifest must record its seed")
    records = manifest.get("records", [])
    if not records:
        raise ValueError("Split manifest has no records")
    sample_ids = [record.get("sample_id") for record in records]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Split manifest sample_id values must be unique and non-empty")

    manifest_version = int(manifest.get("manifest_version", 1))
    if manifest_version not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(f"Unsupported split manifest version: {manifest_version}")
    if manifest_version == 3:
        algorithm = manifest.get("split_algorithm")
        if algorithm not in {V3_SPLIT_ALGORITHM, RANDOM_SPLIT_ALGORITHM}:
            raise ValueError(
                "Version-3 manifests must declare a known split_algorithm "
                f"({V3_SPLIT_ALGORITHM!r} or {RANDOM_SPLIT_ALGORITHM!r}); found "
                f"{algorithm!r}"
            )

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

    if manifest_version >= 2:
        # v2/v3 store explicit grouping keys whose membership must match the
        # canonical re-derivation and never cross splits.
        for record in records:
            expected_group = split_group_key(record, splitting)
            if record.get("split_group") != expected_group:
                raise ValueError(
                    f"split_group {record.get('split_group')!r} does not match "
                    f"{expected_group!r} for sample_id={record['sample_id']!r}"
                )
        locations: dict[str, str] = {}
        for record in records:
            value = record["split_group"]
            split = record["split"]
            previous = locations.setdefault(value, split)
            if previous != split:
                raise AssertionError(f"split_group crosses split: {value}")
    else:
        # Version 1 legacy audit path: derive isolation from raw fields.
        fields = ["canonical_smiles"]
        if splitting == "scaffold":
            fields.append("scaffold")
        for field in fields:
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
