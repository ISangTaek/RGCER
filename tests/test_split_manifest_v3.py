"""Manifest V3 constrained scaffold splitter (P0-1 / review §9-15, §44)."""

import numpy as np
import pytest

from conformal import minimum_calibration_size
from split_manifest import (
    ACYCLIC_SCAFFOLD,
    V3_SPLIT_ALGORITHM,
    create_split_manifest,
    manifest_hash,
    validate_manifest,
)

RATIOS = {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}


def _cyclic_records(num_groups, group_size, first_scaffold="c1ccccc1"):
    records = []
    index = 0
    for g in range(num_groups):
        scaffold = first_scaffold if g == 0 else f"SC{g:04d}"
        for _ in range(group_size):
            records.append(
                {
                    "sample_id": f"c{index}",
                    "row_index": index,
                    "canonical_smiles": f"C{index}CCO",
                    "scaffold": scaffold,
                }
            )
            index += 1
    return records


def _acyclic_records(count, *, duplicate_prefix=0, shared_canonical="CCO-dup"):
    records = []
    for i in range(count):
        canonical = shared_canonical if i < duplicate_prefix else f"F{i}Cl"
        records.append(
            {
                "sample_id": f"a{i}",
                "row_index": 10_000 + i,
                "canonical_smiles": canonical,
                "scaffold": ACYCLIC_SCAFFOLD,
            }
        )
    return records


# §44 Test 1: oversized scaffold pinned to train.
def test_oversized_scaffold_is_pinned_to_train():
    # Giant (200) dwarfs every evaluation target (28), while the remaining
    # small scaffolds (2 each) keep the evaluation splits placeable.
    records = []
    index = 0
    for i in range(200):
        records.append({"sample_id": f"g{index}", "row_index": index,
                        "canonical_smiles": f"G{index}", "scaffold": "c1ccccc1"})
        index += 1
    for g in range(40):
        scaffold = f"SMALL_{g}"
        for _ in range(2):
            records.append({"sample_id": f"s{index}", "row_index": index,
                            "canonical_smiles": f"C{index}N", "scaffold": scaffold})
            index += 1

    manifest = create_split_manifest(records, splitting="scaffold", ratios=RATIOS, seed=42)
    giant = {r["split"] for r in manifest["records"] if r["scaffold"] == "c1ccccc1"}
    assert giant == {"train"}
    validate_manifest(manifest)


# §44 Tests 2+3: scaffold and duplicate-canonical atomicity.
def test_scaffold_and_duplicate_canonical_groups_never_cross_splits():
    records = _cyclic_records(30, group_size=4) + _acyclic_records(40, duplicate_prefix=6)
    manifest = create_split_manifest(records, splitting="scaffold", ratios=RATIOS, seed=3)

    def splits_of(predicate):
        return {r["split"] for r in manifest["records"] if predicate(r)}

    for scaffold in {r["scaffold"] for r in manifest["records"]} - {ACYCLIC_SCAFFOLD}:
        assert len(splits_of(lambda r, s=scaffold: r["scaffold"] == s)) == 1

    duplicated = {
        r["canonical_smiles"]
        for r in manifest["records"]
        if r["scaffold"] == ACYCLIC_SCAFFOLD and r["canonical_smiles"] == "CCO-dup"
    }
    # Every member of the duplicated acyclic structure shares exactly one split.
    members = [r for r in manifest["records"] if r["canonical_smiles"] == "CCO-dup"]
    assert len({r["split"] for r in members}) == 1


# §44 Test 4: evaluation splits see acyclic chemistry.
def test_evaluation_splits_receive_acyclic_chemistry():
    records = _cyclic_records(15, group_size=8) + _acyclic_records(80)
    manifest = create_split_manifest(records, splitting="scaffold", ratios=RATIOS, seed=11)
    by_split = {}
    for record in manifest["records"]:
        if record["scaffold"] == ACYCLIC_SCAFFOLD:
            by_split.setdefault(record["split"], 0)
            by_split[record["split"]] += 1
    for split in ("validation", "calibration", "test"):
        assert by_split.get(split, 0) > 0


# §44 Test 5: sparse human labels still cover val/test/calibration floors.
def test_priority_label_presence_floors_are_enforced():
    tasks = ["human_oral_TDLo", "women_oral_TDLo", "rat_oral_LD50"]
    records = _cyclic_records(60, group_size=2) + _acyclic_records(60)
    rng = np.random.default_rng(0)
    presence = (rng.random((len(records), len(tasks))) > 0.55).astype(np.int64)

    minimum = 4
    manifest = create_split_manifest(
        records,
        splitting="scaffold",
        ratios={"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        seed=5,
        task_names=tasks,
        label_presence=presence,
        priority_task_names=tasks[:2],
        conformal_task_names=tasks,
        conformal_alpha=0.30,
        min_priority_eval_count=minimum,
        num_candidates=16,
    )

    calibration_minimum = minimum_calibration_size(0.30)
    split_of = {record["sample_id"]: record["split"] for record in manifest["records"]}
    for task_index, task in enumerate(tasks[:2]):
        per_split = {"train": 0, "validation": 0, "calibration": 0, "test": 0}
        for offset, record in enumerate(records):
            if presence[offset][task_index]:
                per_split[split_of[record["sample_id"]]] += 1
        assert per_split["validation"] >= minimum, task
        assert per_split["test"] >= minimum, task
    # Conformal scope: every named endpoint reaches the finite-rank floor.
    for task_index in range(len(tasks)):
        cal_count = sum(
            presence[offset][task_index]
            for offset, record in enumerate(records)
            if split_of[record["sample_id"]] == "calibration"
        )
        assert cal_count >= calibration_minimum


def test_impossible_priority_floor_fails_loudly_without_scaffold_splitting():
    tasks = ["human_oral_TDLo"]
    records = _cyclic_records(12, group_size=2)  # only 24 molecules
    # No human labels at all -> floor cannot be met by any candidate.
    presence = np.zeros((len(records), len(tasks)), dtype=np.int64)
    with pytest.raises(ValueError, match="FAIL_PRIORITY_COVERAGE"):
        create_split_manifest(
            records,
            splitting="scaffold",
            ratios={"train": 0.4, "validation": 0.2, "calibration": 0.2, "test": 0.2},
            seed=1,
            task_names=tasks,
            label_presence=presence,
            priority_task_names=["human_oral_TDLo"],
            min_priority_eval_count=3,
            num_candidates=4,
        )


# §44 Test 6: determinism under identical inputs.
def test_identical_inputs_yield_identical_manifest_hash():
    records_a = _cyclic_records(25, group_size=3) + _acyclic_records(30, duplicate_prefix=3)
    records_b = [dict(r) for r in records_a]
    tasks = ["t1", "t2"]
    rng = np.random.default_rng(9)
    presence = rng.random((len(records_a), len(tasks))) > 0.5

    common = dict(
        splitting="scaffold",
        ratios=RATIOS,
        seed=17,
        task_names=tasks,
        label_presence=presence,
        priority_task_names=["t1"],
        min_priority_eval_count=3,
        num_candidates=8,
    )
    m1 = create_split_manifest(records_a, **common)
    m2 = create_split_manifest(records_b, **common)
    assert manifest_hash(m1) == manifest_hash(m2)
    assert m1["split_algorithm"] == V3_SPLIT_ALGORITHM
    assert m1["constraints"]["priority_tasks"] == ["t1"]
