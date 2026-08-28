from collections import defaultdict
from pathlib import Path

import pytest

from split_manifest import (
    ACYCLIC_SCAFFOLD,
    MANIFEST_VERSION,
    SPLIT_NAMES,
    create_split_manifest,
    load_manifest,
    manifest_hash,
    split_group_key,
    validate_manifest,
    write_manifest,
)

RATIOS = {"train": 0.5, "validation": 0.2, "calibration": 0.2, "test": 0.1}


def test_manifest_keeps_canonical_and_scaffold_groups_together(tmp_path: Path):
    records = [
        {"sample_id": "a1", "canonical_smiles": "CCO", "scaffold": "acyclic_a"},
        {"sample_id": "a2", "canonical_smiles": "CCO", "scaffold": "acyclic_a"},
        {"sample_id": "b1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene"},
        {"sample_id": "c1", "canonical_smiles": "CCN", "scaffold": "acyclic_c"},
        {"sample_id": "d1", "canonical_smiles": "CCC", "scaffold": "acyclic_d"},
        {"sample_id": "e1", "canonical_smiles": "CCCl", "scaffold": "acyclic_e"},
        {"sample_id": "f1", "canonical_smiles": "CCBr", "scaffold": "acyclic_f"},
        {"sample_id": "g1", "canonical_smiles": "CCF", "scaffold": "acyclic_g"},
    ]
    # Grouping/isolation semantics only; scale-dependent constraints need a
    # realistic corpus and are covered by the dedicated v3 tests.
    manifest = create_split_manifest(records, splitting="random", ratios={
        "train": 0.5, "validation": 0.2, "calibration": 0.2, "test": 0.1
    })
    path = tmp_path / "split_manifest.json"
    write_manifest(manifest, path)
    loaded = load_manifest(path)
    assert manifest_hash(manifest) == manifest_hash(loaded)

    locations = {}
    for record in loaded["records"]:
        for field in ("canonical_smiles", "scaffold"):
            value = record[field]
            locations.setdefault((field, value), record["split"])
            assert locations[(field, value)] == record["split"]


def test_empty_subset_is_really_empty():
    from dataset import DataloaderWrapper

    loader = DataloaderWrapper._loader([], [], "validation", 4, 0, lambda x: x)
    assert len(loader) == 0
    assert len(loader.dataset) == 0


def test_random_manifest_does_not_require_scaffold_isolation():
    records = [
        {
            "sample_id": f"sample_{index}",
            "canonical_smiles": "CCO" if index < 2 else f"CC{index}",
            "scaffold": "shared_scaffold",
        }
        for index in range(8)
    ]
    manifest = create_split_manifest(
        records,
        splitting="random",
        ratios={"train": 0.5, "validation": 0.2, "calibration": 0.2, "test": 0.1},
    )
    locations = {}
    for record in manifest["records"]:
        key = record["canonical_smiles"]
        locations.setdefault(key, record["split"])
        assert locations[key] == record["split"]


def test_v2_manifest_records_split_group(tmp_path: Path):
    records = [
        {"sample_id": "cyc_1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene"},
        {"sample_id": "cyc_2", "canonical_smiles": "c1ccc(O)cc1", "scaffold": "benzene"},
        {"sample_id": "acy_1", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD},
        {"sample_id": "acy_2", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD},
    ]
    manifest = create_split_manifest(records, ratios=RATIOS, enforce_scale_constraints=False)
    assert manifest["manifest_version"] == MANIFEST_VERSION == 3
    by_id = {record["sample_id"]: record for record in manifest["records"]}
    assert by_id["cyc_1"]["split_group"] == "scaffold::benzene"
    assert by_id["cyc_2"]["split_group"] == "scaffold::benzene"
    assert by_id["acy_1"]["split_group"] == f"acyclic::CCO"
    assert by_id["acy_2"]["split_group"] == f"acyclic::CCO"
    for record in manifest["records"]:
        assert record["split_group"] == split_group_key(record, "scaffold")

    path = tmp_path / "split_manifest.json"
    write_manifest(manifest, path)
    loaded = load_manifest(path)
    assert manifest_hash(manifest) == manifest_hash(loaded)
    for record in loaded["records"]:
        assert record["split_group"] == split_group_key(record, "scaffold")

    random_manifest = create_split_manifest(
        [dict(record) for record in records], splitting="random", ratios=RATIOS
    )
    for record in random_manifest["records"]:
        expected = f"canonical::{record['canonical_smiles']}"
        assert record["split_group"] == expected == split_group_key(record, "random")


def test_acyclic_chemistry_spreads_across_splits():
    # A cyclic anchor fills the train target so the greedy balancer has room
    # to spread the acyclic singletons across every evaluation split.
    records = [
        {"sample_id": f"cyc_{i}", "canonical_smiles": f"c1ccccc1-{i}", "scaffold": "benzene"}
        for i in range(14)
    ]
    for i in range(20):
        smiles = f"CC{'C' * (i % 10)}{'O' if i < 10 else 'N'}"
        records.append(
            {"sample_id": f"acy_{i}", "canonical_smiles": smiles, "scaffold": ACYCLIC_SCAFFOLD}
        )
    manifest = create_split_manifest(records)

    acyclic_splits = {
        record["split"] for record in manifest["records"] if record["scaffold"] == ACYCLIC_SCAFFOLD
    }
    assert acyclic_splits == set(SPLIT_NAMES)

    structure_split = {}
    for record in manifest["records"]:
        if record["scaffold"] == ACYCLIC_SCAFFOLD:
            structure_split.setdefault(record["canonical_smiles"], record["split"])
            assert structure_split[record["canonical_smiles"]] == record["split"]


def test_scaffold_split_keeps_cyclic_together_and_spreads_acyclic():
    records = [
        {"sample_id": f"cyc_{i}", "canonical_smiles": f"c1ccccc1-{i}", "scaffold": "benzene"}
        for i in range(14)
    ]
    for i in range(4):
        records.append(
            {"sample_id": f"eth_{i}", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD}
        )
    for i in range(16):
        smiles = f"CC{'C' * i}N"
        records.append(
            {"sample_id": f"acy_{i}", "canonical_smiles": smiles, "scaffold": ACYCLIC_SCAFFOLD}
        )
    manifest = create_split_manifest(records)

    benzene_splits = {record["split"] for record in manifest["records"] if record["scaffold"] == "benzene"}
    assert len(benzene_splits) == 1

    eth_splits = {
        record["split"] for record in manifest["records"] if record["canonical_smiles"] == "CCO"
    }
    assert len(eth_splits) == 1

    acyclic_splits = {
        record["split"] for record in manifest["records"] if record["scaffold"] == ACYCLIC_SCAFFOLD
    }
    assert acyclic_splits == set(SPLIT_NAMES)

    by_split = defaultdict(list)
    for record in manifest["records"]:
        by_split[record["split"]].append(record)
    assert all(by_split[name] for name in SPLIT_NAMES)


def test_v2_rejects_split_group_crossing_splits():
    records = [
        {"sample_id": "acy_1", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD},
        {"sample_id": "acy_2", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD},
        {"sample_id": "cyc_1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene"},
        {"sample_id": "cyc_2", "canonical_smiles": "c1ccc(O)cc1", "scaffold": "benzene"},
    ]
    manifest = create_split_manifest(records, ratios=RATIOS, enforce_scale_constraints=False)
    by_id = {record["sample_id"]: record for record in manifest["records"]}
    by_id["acy_1"]["split"] = "train"
    by_id["acy_2"]["split"] = "test"
    with pytest.raises(AssertionError, match="split_group crosses split"):
        validate_manifest(manifest)


def test_v2_rejects_missing_split_group():
    records = [
        {"sample_id": "cyc_1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene"},
        {"sample_id": "acy_1", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD},
    ]
    manifest = create_split_manifest(records, ratios=RATIOS, enforce_scale_constraints=False)
    for record in manifest["records"]:
        del record["split_group"]
    with pytest.raises(ValueError, match="split_group None does not match"):
        validate_manifest(manifest)


def test_v2_rejects_tampered_split_group():
    records = [
        {"sample_id": "cyc_1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene"},
        {"sample_id": "acy_1", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD},
    ]
    manifest = create_split_manifest(records, ratios=RATIOS, enforce_scale_constraints=False)
    manifest["records"][0]["split_group"] = "scaffold::other"
    with pytest.raises(ValueError, match="does not match"):
        validate_manifest(manifest)


def _legacy_v1_manifest() -> dict:
    # Legacy shape: no split_group field, and every acyclic row shared the
    # single __ACYCLIC__ super-group, so all acyclic chemistry sat in one split.
    return {
        "manifest_version": 1,
        "splitting": "scaffold",
        "seed": 42,
        "ratios": dict(RATIOS),
        "records": [
            {"sample_id": "acy_1", "canonical_smiles": "CCO", "scaffold": ACYCLIC_SCAFFOLD, "split": "train"},
            {"sample_id": "acy_2", "canonical_smiles": "CCN", "scaffold": ACYCLIC_SCAFFOLD, "split": "train"},
            {"sample_id": "cyc_1", "canonical_smiles": "c1ccccc1", "scaffold": "benzene", "split": "validation"},
            {"sample_id": "cyc_2", "canonical_smiles": "c1ccc(O)cc1", "scaffold": "benzene", "split": "validation"},
        ],
    }


def test_v1_manifest_still_validates():
    validate_manifest(_legacy_v1_manifest())


def test_v1_manifest_rejects_crossing_super_group():
    manifest = _legacy_v1_manifest()
    manifest["records"][1]["split"] = "test"
    with pytest.raises(AssertionError, match="scaffold crosses split"):
        validate_manifest(manifest)
