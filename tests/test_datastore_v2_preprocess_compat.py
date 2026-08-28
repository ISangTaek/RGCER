"""V1/V2 preprocessing compatibility: feature parity and legacy outputs."""

import io

import pandas as pd
import torch

from molecular_features import FEATURE_SCHEMA_VERSION
from preprocess_data import get_graph_data_from_smiles, get_graph_features_from_smiles
from toxacute_datastore import ToxAcuteDataStore, build_datastore_v2


def test_v2_record_is_task_free_and_shape_consistent():
    record = get_graph_features_from_smiles("CCO", sample_id="row_2", max_path_distance=6)

    for forbidden in ("y", "label", "task_name", "edge_index", "edge_attr"):
        assert forbidden not in record
    assert record["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert record["graph_record_version"] == 1
    assert record["sample_id"] == "row_2"
    assert record["num_nodes"] == 3
    assert record["x"].shape[0] == 3
    assert record["in_degree"].shape == (3,)
    assert record["out_degree"].shape == (3,)
    assert record["spatial_pos"].shape == (3, 3)
    assert record["attn_edge_type"].ndim == 3
    assert record["edge_input"].ndim == 4
    assert record["edge_input"].shape[0] == 3
    assert record["edge_input"].shape[2] == 6
    assert record["canonical_smiles"] == "CCO"
    assert record["raw_smiles"] == "CCO"


def test_v1_v2_tensor_parity_and_label_separation():
    label_value = 12.5
    v2_record = get_graph_features_from_smiles("CCN", sample_id="row_1", max_path_distance=8)
    v1_data = get_graph_data_from_smiles(
        "CCN", label_value, sample_id="row_1", task_name="task_a", max_path_distance=8
    )

    for field in ("x", "in_degree", "out_degree", "spatial_pos", "attn_edge_type", "edge_input"):
        assert torch.equal(getattr(v1_data, field), v2_record[field]), field
    # Edge tensors exist only on the legacy object and stay consistent with it.
    assert v1_data.edge_index.shape[1] > 0
    assert v1_data.edge_attr.shape[1] == v1_data.attn_edge_type.shape[2]

    assert float(v1_data.y.item()) == label_value
    assert v1_data.task_name == "task_a"
    assert v1_data.sample_id == "row_1"
    assert v1_data.feature_schema_version == FEATURE_SCHEMA_VERSION
    assert v1_data.canonical_smiles == "CCN"


def test_v1_data_still_serializes_for_legacy_consumers():
    data = get_graph_data_from_smiles("c1ccccc1", 3.0, task_name="legacy")
    buffer = io.BytesIO()
    torch.save(data, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=False)
    assert torch.equal(restored.x, data.x)
    assert restored.task_name == "legacy"
    assert restored.edge_index is not None


def test_invalid_smiles_raises_value_error():
    for smiles in ("not_a_smiles", ""):
        try:
            get_graph_features_from_smiles(smiles)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {smiles!r}")


def test_datastore_roundtrip_preserves_v2_record_fields(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(4)],
            "Pubchem CID": list(range(4)),
            "IUPAC Name": [""] * 4,
            "SMILES": [""] * 4,
            "smiles": ["CC", "CCO", "CCN", "CCCl"],
            "InChIKey": [""] * 4,
            "task_a": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(raw_csv, index=False)
    build_datastore_v2(
        raw_csv,
        tmp_path / "datastore",
        task_names=["task_a"],
        splitting="random",
        valid_size=0.25,
        calibration_size=0.25,
        test_size=0.25,
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(tmp_path / "datastore")
    try:
        for global_index in range(len(store.row_indices)):
            record = store.get_graph_record(global_index)
            for field in (
                "x",
                "in_degree",
                "out_degree",
                "spatial_pos",
                "attn_edge_type",
                "edge_input",
                "raw_smiles",
                "canonical_smiles",
                "scaffold",
                "sample_id",
                "num_nodes",
                "feature_schema_version",
                "graph_record_version",
            ):
                assert field in record, field
    finally:
        store.close()
