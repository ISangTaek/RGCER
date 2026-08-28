import json

import numpy as np
import pandas as pd
import pytest
import torch

from preprocess_data import get_graph_data_from_smiles
from split_manifest import MANIFEST_VERSION
from toxacute_datastore import ToxAcuteDataStore, ToxAcuteTaskDataset, build_datastore_v2


def _write_raw(path):
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(6)],
            "Pubchem CID": list(range(6)),
            "IUPAC Name": ["" for _ in range(6)],
            "SMILES": ["CC", "CCC", "CCO", "CCN", "c1ccccc1", "CCCl"],
            "smiles": ["CC", "CCC", "CCO", "CCN", "c1ccccc1", "CCCl"],
            "InChIKey": ["" for _ in range(6)],
            "task_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "task_b": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0],
        }
    ).to_csv(path, index=False)


def test_datastore_v2_build_read_and_task_views(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)

    build = build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
        commit_every=2,
    )

    assert (build / "READY").exists()
    assert (root / "CURRENT").read_text(encoding="utf-8").strip() == build.name
    assert not list(build.rglob("data_*.pt"))

    store = ToxAcuteDataStore.resolve(root)
    assert store.labels.shape == (6, 2)
    assert store.metadata["max_path_distance"] == 6
    assert store.lmdb_entry_count() == 6
    record = store.get_graph_record(0)
    assert "y" not in record
    assert "task_name" not in record

    task_a_train = ToxAcuteTaskDataset(store, "task_a", split="train")
    assert all(np.isfinite(store.get_label(int(index), "task_a")) for index in task_a_train.indices)
    sample_id = task_a_train.get_sample_id(0)
    assert sample_id.startswith("row_")
    store.close()


def test_datastore_v2_sample_id_does_not_read_graph(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    dataset = ToxAcuteTaskDataset(store, "task_a", split=None)

    original = store.get_graph_record
    store.get_graph_record = lambda index: (_ for _ in ()).throw(AssertionError("graph read"))
    try:
        assert dataset.get_sample_id(0).startswith("row_")
    finally:
        store.get_graph_record = original
        store.close()


def test_datastore_v1_v2_graph_feature_parity(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    try:
        global_index = int(np.where(store.sample_ids == "row_2")[0][0])
        v1_data = get_graph_data_from_smiles("CCO", label_val=30.0, max_path_distance=6)
        v2_record = store.get_graph_record(global_index)
        for field in ("x", "in_degree", "out_degree", "spatial_pos", "attn_edge_type", "edge_input"):
            assert torch.equal(getattr(v1_data, field), v2_record[field])
        assert store.get_label(global_index, "task_b") == 30.0
    finally:
        store.close()


def test_failed_build_preserves_active_current(tmp_path, monkeypatch):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    first_build = build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        lmdb_map_size_gb=0.001,
    )
    current_before = (root / "CURRENT").read_text(encoding="utf-8")

    def fail_graph(*args, **kwargs):
        raise RuntimeError("synthetic graph failure")

    monkeypatch.setattr("preprocess_data.get_graph_features_from_smiles", fail_graph, raising=False)
    with pytest.raises(ValueError, match="No valid graph records"):
        build_datastore_v2(
            raw_csv,
            root,
            task_names=["task_a", "task_b"],
            splitting="random",
            valid_size=0.2,
            calibration_size=0.2,
            test_size=0.2,
            lmdb_map_size_gb=0.001,
        )

    assert (root / "CURRENT").read_text(encoding="utf-8") == current_before
    assert first_build.exists()
    assert not list((root / "builds").glob(".tmp-*"))


def test_datastore_rejects_legacy_v1_split_manifest(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build = build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        lmdb_map_size_gb=0.001,
    )
    assert build is not None

    manifest_path = build / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == MANIFEST_VERSION
    manifest["manifest_version"] = 1
    for record in manifest["records"]:
        record.pop("split_group", None)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="DataStore requires split manifest version 3"):
        ToxAcuteDataStore.resolve(root)


def test_build_rejects_v1_approved_split_manifest(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)

    smiles_by_row = ["CC", "CCC", "CCO", "CCN", "c1ccccc1", "CCCl"]
    split_by_row = ["train", "train", "validation", "calibration", "test", "train"]
    approved = tmp_path / "approved_v1.json"
    approved.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "splitting": "random",
                "seed": 42,
                "ratios": {"train": 0.4, "validation": 0.2, "calibration": 0.2, "test": 0.2},
                "records": [
                    {
                        "sample_id": f"row_{row}",
                        "canonical_smiles": smiles_by_row[row],
                        "split": split_by_row[row],
                    }
                    for row in range(6)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Approved split manifest must be version 3"):
        build_datastore_v2(
            raw_csv,
            root,
            task_names=["task_a", "task_b"],
            splitting="random",
            valid_size=0.2,
            calibration_size=0.2,
            test_size=0.2,
            split_seed=42,
            lmdb_map_size_gb=0.001,
            split_manifest_path=approved,
        )
    assert not list((root / "builds").glob(".tmp-*"))
