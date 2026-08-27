"""Read-side contract: labels, index filtering, graph deserialization, errors."""

import numpy as np
import pandas as pd
import pytest
import torch

from toxacute_datastore import (
    SPLIT_CODES,
    ToxAcuteDataStore,
    ToxAcuteTaskDataset,
    build_datastore_v2,
)


def _write_raw(path):
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(6)],
            "Pubchem CID": list(range(6)),
            "IUPAC Name": ["" for _ in range(6)],
            "SMILES": [""] * 6,
            "smiles": ["CC", "CCC", "CCO", "CCN", "c1ccccc1", "CCCl"],
            "InChIKey": ["" for _ in range(6)],
            "task_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "task_b": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0],
        }
    ).to_csv(path, index=False)


@pytest.fixture()
def store(tmp_path):
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
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )
    resolved = ToxAcuteDataStore.resolve(root)
    try:
        yield resolved
    finally:
        resolved.close()


def test_get_label_returns_dense_matrix_values(store):
    assert store.labels.shape == (6, 2)
    assert store.get_label(0, "task_a") == 1.0
    assert store.get_label(2, "task_b") == 30.0
    assert np.isnan(store.get_label(2, "task_a"))
    assert np.isnan(store.get_label(1, "task_b"))
    with pytest.raises(KeyError):
        store.get_label(0, "missing_task")


def test_get_task_indices_filters_finite_labels_and_split(store):
    all_a = store.get_task_indices("task_a")
    assert len(all_a) == 5
    assert all(np.isfinite(store.get_label(int(index), "task_a")) for index in all_a)

    # Five labelled samples across four splits: a split may legitimately be
    # empty, but each labelled sample belongs to exactly one split.
    per_split = {}
    for split_name, code in SPLIT_CODES.items():
        indices = store.get_task_indices("task_a", split=split_name)
        per_split[split_name] = indices
        assert len(np.intersect1d(indices, all_a)) == len(indices)
        assert np.all(store.split_codes[indices] == code)
    assert sum(len(indices) for indices in per_split.values()) == len(all_a)

    # "val" is accepted as the short alias for "validation".
    val_indices = store.get_task_indices("task_a", split="val")
    assert np.all(store.split_codes[val_indices] == SPLIT_CODES["validation"])
    with pytest.raises(ValueError):
        store.get_task_indices("task_a", split="bogus")


def test_get_task_indices_respects_max_nodes(store):
    all_indices = store.get_task_indices("task_a")
    small_indices = store.get_task_indices("task_a", max_nodes=4)
    # c1ccccc1 has six nodes; every other fixture molecule has four or fewer.
    benzene_indices = np.where(store.num_nodes > 4)[0]
    assert len(benzene_indices) == 1
    assert len(small_indices) == len(all_indices) - 1
    assert np.all(store.num_nodes[small_indices] <= 4)
    assert np.isin(int(benzene_indices[0]), small_indices).item() is False
    with pytest.raises(ValueError):
        store.get_task_indices("task_a", max_nodes=0)


def test_get_graph_record_shapes_match_index_metadata(store):
    for index in range(len(store.row_indices)):
        record = store.get_graph_record(index)
        expected_nodes = int(store.num_nodes[index])
        assert record["x"].shape == (expected_nodes, record["x"].shape[1])
        assert record["in_degree"].shape == (expected_nodes,)
        assert record["out_degree"].shape == (expected_nodes,)
        assert record["spatial_pos"].shape == (expected_nodes, expected_nodes)
        assert record["edge_input"].ndim >= 2
        assert record["num_nodes"] == expected_nodes
        assert record["sample_id"] == str(store.sample_ids[index])
    with pytest.raises(IndexError):
        store.get_graph_record(len(store.row_indices) + 1)
    with pytest.raises(IndexError):
        store.get_graph_record(-1)


def test_get_graph_data_attaches_label_and_task(store):
    index = int(store.get_task_indices("task_a")[0])
    data = store.get_graph_data(index, task_name="task_a")
    assert torch.equal(
        data.y, torch.tensor([store.get_label(index, "task_a")], dtype=torch.float)
    )
    assert data.task_name == "task_a"
    assert data.num_nodes == data.x.size(0)
    assert data.sample_id == str(store.sample_ids[index])
    assert data.feature_schema_version == store.metadata["feature_schema_version"]

    explicit = store.get_graph_data(index, label=99.0)
    assert float(explicit.y.item()) == 99.0
    assert "task_name" not in explicit


def test_task_dataset_iterates_graphs_with_labels(store):
    dataset = ToxAcuteTaskDataset(store, "task_b", split="train")
    assert 0 < len(dataset) < 6
    first = dataset[0]
    global_index = int(dataset.indices[0])
    assert first.y.item() == store.get_label(global_index, "task_b")
    assert first.task_name == "task_b"
    assert first.x.shape[0] == int(store.num_nodes[global_index])
    assert first.smiles == store.get_graph_record(global_index)["canonical_smiles"]
