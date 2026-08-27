"""Full-pipeline DataStore V2 build contract: build, READY, CURRENT, artifacts."""

import json

import numpy as np
import pandas as pd

from molecular_features import FEATURE_SCHEMA_VERSION
from toxacute_datastore import (
    GRAPH_RECORD_VERSION,
    ToxAcuteDataStore,
    build_datastore_v2,
)


def _write_raw(path):
    """Eight rows: six buildable molecules, one unparseable SMILES, one missing SMILES."""
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(8)],
            "Pubchem CID": list(range(8)),
            "IUPAC Name": ["" for _ in range(8)],
            "SMILES": [""] * 8,
            "smiles": [
                "CC",
                "CCC",
                "CCO",
                "CCN",
                "c1ccccc1",
                "CCCl",
                "not_a_smiles",
                np.nan,
            ],
            "InChIKey": ["" for _ in range(8)],
            "task_a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0],
            "task_b": [10.0, np.nan, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0],
        }
    ).to_csv(path, index=False)


def _build(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    return raw_csv, root, build_datastore_v2(
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
        commit_every=4,
    )


def test_build_produces_ready_artifact_with_current(tmp_path):
    _, root, build = _build(tmp_path)

    assert (build / "READY").exists()
    assert build.parent.name == "builds"
    assert (root / "CURRENT").read_text(encoding="utf-8").strip() == build.name
    assert not list(build.rglob("data_*.pt"))
    assert not list(root.rglob("data_*.pt"))
    for required in (
        "datastore.json",
        "split_manifest.json",
        "index.npz",
        "labels.npy",
        "task_stats.json",
        "preprocess_errors.json",
        "data_preflight.json",
        "graph_shards",
    ):
        assert (build / required).exists(), required


def test_build_records_invalid_rows_as_errors(tmp_path):
    _, root, build = _build(tmp_path)
    errors = json.loads((build / "preprocess_errors.json").read_text(encoding="utf-8"))

    assert len(errors) == 2
    row_indexes = sorted(error["row_index"] for error in errors)
    assert row_indexes == [6, 7]
    assert all("error" in error for error in errors)

    labels = np.load(build / "labels.npy")
    # Rows 6 and 7 never become graph samples; row 2 lacks task_a and row 1
    # lacks task_b, so the observed cells are NaN.
    assert np.isnan(labels[2, 0])
    assert np.isnan(labels[1, 1])
    assert labels[2, 1] == 30.0


def test_datastore_metadata_is_internally_consistent(tmp_path):
    _, root, build = _build(tmp_path)
    metadata = json.loads((build / "datastore.json").read_text(encoding="utf-8"))

    assert metadata["format"] == "toxacute_datastore"
    assert metadata["format_version"] == 2
    assert metadata["graph_record_version"] == GRAPH_RECORD_VERSION
    assert metadata["build_id"].startswith("toxacute-v2-")
    assert metadata["build_id"] == build.name
    assert len(metadata["datastore_fingerprint"]) == 64
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["max_path_distance"] == 6
    assert metadata["task_names"] == ["task_a", "task_b"]
    assert metadata["num_samples"] == 6
    assert metadata["lmdb_entries"] == 6
    assert metadata["num_tasks"] == 2
    assert metadata["labels_shape"] == [6, 2]
    assert metadata["num_observed_labels"] == 9
    assert metadata["raw_csv_sha256"]
    assert metadata["split_manifest_hash"]

    manifest = json.loads((build / "split_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["records"]) == 6

    store = ToxAcuteDataStore.resolve(root)
    try:
        assert store.fingerprint == metadata["datastore_fingerprint"]
        assert store.split_manifest_hash == metadata["split_manifest_hash"]
        assert store.raw_csv_sha256 == metadata["raw_csv_sha256"]
        assert store.validate(strict=True) is None
    finally:
        store.close()


def test_task_stats_match_label_matrix(tmp_path):
    _, root, build = _build(tmp_path)
    del root
    task_stats = json.loads((build / "task_stats.json").read_text(encoding="utf-8"))

    assert task_stats["num_samples"] == 6
    assert task_stats["tasks"]["task_a"]["total"] == 5
    assert task_stats["tasks"]["task_b"]["total"] == 4
    for task_name in ("task_a", "task_b"):
        stats = task_stats["tasks"][task_name]
        assert (
            stats["train"]
            + stats["validation"]
            + stats["calibration"]
            + stats["test"]
            == stats["total"]
        )
        assert stats["train"] > 0
        assert task_name not in task_stats["empty_train_tasks"]


def test_context_exposes_run_identity(tmp_path):
    _, root, build = _build(tmp_path)
    store = ToxAcuteDataStore.resolve(root)
    try:
        context = store.context
        assert context.root == str(store.root)
        assert context.build_id == build.name
        assert context.fingerprint == store.fingerprint
        assert context.max_path_distance == 6
        assert context.task_names == ("task_a", "task_b")
        assert context.feature_schema_version == FEATURE_SCHEMA_VERSION
    finally:
        store.close()
