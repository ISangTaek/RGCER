from pathlib import Path

import numpy as np
import pytest

import baselines.data as data_module
from baselines.constants import HUMAN3_TASKS, SPLIT_MANIFEST_HASH, TOXACUTE_TASKS
from baselines.data import (
    GraphRecordView,
    derive_authorized_sample_expectation,
    load_authorized_inference_split,
    load_authorized_validation,
    load_formal_train_validation,
    load_human3_smoke,
)
from toxacute_datastore import SPLIT_CODES


class SyntheticStore:
    def __init__(self, entries):
        self.root = Path("synthetic-store")
        self.sample_ids = np.asarray([entry["sample_id"] for entry in entries])
        self.split_codes = np.asarray([SPLIT_CODES[entry["split"]] for entry in entries], dtype=np.int8)
        self.task_names = list(TOXACUTE_TASKS)
        self.labels = np.full((len(entries), len(TOXACUTE_TASKS)), np.nan, dtype=np.float32)
        self._records = []
        for index, entry in enumerate(entries):
            for task, value in entry.get("labels", {}).items():
                self.labels[index, self.task_names.index(task)] = value
            self._records.append({"sample_id": entry["sample_id"], "raw_smiles": entry.get("smiles", "CC")})
        self.metadata = {"feature_schema_version": "synthetic"}
        self.closed = False

    def task_index(self, task):
        return self.task_names.index(task)

    def get_graph_record(self, index):
        return self._records[int(index)]

    def close(self):
        self.closed = True


def _synthetic_open(entries):
    store = SyntheticStore(entries)
    manifest = {
        entry["sample_id"]: {
            "sample_id": entry["sample_id"],
            "row_index": entry["row_index"],
            "split": entry.get("manifest_split", entry["split"]),
        }
        for entry in entries
    }
    return store, manifest, {str(sample_id): index for index, sample_id in enumerate(store.sample_ids)}


def test_human3_smoke_identity_order_and_limits(baseline_datastore):
    data = load_human3_smoke(baseline_datastore)
    assert len(data.train.sample_ids) <= 96
    assert len(data.validation.sample_ids) <= 24
    assert set(data.train.sample_ids).isdisjoint(data.validation.sample_ids)
    assert np.all(np.diff(data.train.raw_row_indices) >= 0)
    assert np.all(np.diff(data.validation.raw_row_indices) >= 0)
    for task in HUMAN3_TASKS:
        assert len(data.selection_by_task[task]["train"]) == 32
        assert len(data.selection_by_task[task]["validation"]) == 8


def test_human3_smoke_never_materializes_formal_splits(baseline_datastore):
    manifest = load_human3_smoke(baseline_datastore).to_manifest()
    assert "test" not in manifest
    assert "calibration" not in manifest


def test_human3_smoke_afp_can_use_shared_lazy_graph_path(baseline_datastore):
    data = load_human3_smoke(baseline_datastore, lazy_graphs=True)
    try:
        assert data.graph_record_mode == "shared_lazy_datastore"
        assert data.to_manifest()["graph_record_mode"] == "shared_lazy_datastore"
        assert data.train.graph_records._owner is data.validation.graph_records._owner
        assert data.train.graph_records[0]["sample_id"] == data.train.sample_ids[0]
        assert data.validation.graph_records[0]["sample_id"] == data.validation.sample_ids[0]
    finally:
        data.close()


def test_formal_afp_lazy_views_share_one_store_across_train_validation_train(baseline_datastore):
    data = load_formal_train_validation(baseline_datastore, "afp")
    owner = data._graph_store_owner
    assert owner is not None
    assert isinstance(data.train.graph_records, GraphRecordView)
    assert isinstance(data.validation.graph_records, GraphRecordView)
    assert data.train.graph_records._owner is owner
    assert data.validation.graph_records._owner is owner
    try:
        first_train = data.train.graph_records[0]
        assert owner.is_open
        opened_shards = set(owner._store._envs)
        first_validation = data.validation.graph_records[0]
        second_train = data.train.graph_records[0]
        assert set(owner._store._envs) == opened_shards
        assert first_train["sample_id"] == second_train["sample_id"]
        assert first_validation["sample_id"] == data.validation.sample_ids[0]
    finally:
        data.close()
    assert not owner.is_open


def test_formal_afp_lazy_context_releases_after_exception_and_can_reload(baseline_datastore):
    data = load_formal_train_validation(baseline_datastore, "afp")
    owner = data._graph_store_owner
    with pytest.raises(RuntimeError, match="intentional lazy-reader failure"):
        with data:
            data.train.graph_records[0]
            raise RuntimeError("intentional lazy-reader failure")
    assert owner is not None and not owner.is_open

    reloaded = load_formal_train_validation(baseline_datastore, "afp")
    try:
        assert reloaded.validation.graph_records[0]["sample_id"] == reloaded.validation.sample_ids[0]
        assert reloaded.train.graph_records[0]["sample_id"] == reloaded.train.sample_ids[0]
    finally:
        reloaded.close()


def test_authorized_synthetic_test_selection_is_sorted_and_preserves_missing_labels(
    monkeypatch,
    authorized_rf_context,
):
    entries = [
        {"sample_id": "b", "row_index": 20, "split": "test", "labels": {HUMAN3_TASKS[0]: 1.0}},
        {"sample_id": "v", "row_index": 5, "split": "validation", "labels": {HUMAN3_TASKS[0]: 9.0}},
        {"sample_id": "a", "row_index": 10, "split": "test", "labels": {HUMAN3_TASKS[1]: 2.0}},
        {"sample_id": "c", "row_index": 30, "split": "test", "labels": {}},
        {"sample_id": "d", "row_index": 25, "split": "test", "labels": {HUMAN3_TASKS[2]: 3.0}},
    ]
    monkeypatch.setattr(data_module, "_open_validated", lambda unused: _synthetic_open(entries))
    partition = load_authorized_inference_split(
        "unused",
        "rf",
        split="test",
        test_grant=authorized_rf_context.grant,
    )
    try:
        assert partition.sample_ids == ("a", "b", "d")
        assert partition.raw_row_indices.tolist() == [10, 20, 25]
        assert np.isfinite(partition.labels_human3).sum(axis=1).tolist() == [1, 1, 1]
        assert partition.labels_human3.shape == (3, 3)
    finally:
        partition.graph_records.close()
    expectation = derive_authorized_sample_expectation(
        "unused",
        "rf",
        split="test",
        test_grant=authorized_rf_context.grant,
    )
    assert expectation.sample_ids == partition.sample_ids
    assert expectation.raw_row_indices.tolist() == partition.raw_row_indices.tolist()


def test_authorized_synthetic_test_rejects_cross_split_manifest_mix(monkeypatch, authorized_rf_context):
    entries = [{
        "sample_id": "mixed",
        "row_index": 1,
        "split": "test",
        "manifest_split": "validation",
        "labels": {HUMAN3_TASKS[0]: 1.0},
    }]
    monkeypatch.setattr(data_module, "_open_validated", lambda unused: _synthetic_open(entries))
    with pytest.raises(ValueError, match="Cross-source sample/split mismatch"):
        load_authorized_inference_split(
            "unused",
            "rf",
            split="test",
            test_grant=authorized_rf_context.grant,
        )


def test_datastore_duplicate_sample_id_is_rejected_before_inference(monkeypatch):
    entries = [
        {"sample_id": "duplicate", "row_index": 0, "split": "validation", "labels": {}},
        {"sample_id": "duplicate", "row_index": 1, "split": "validation", "labels": {}},
    ]
    store = SyntheticStore(entries)
    store.build_id = data_module.DATASTORE_BUILD_ID
    store.fingerprint = data_module.DATASTORE_FINGERPRINT
    store.split_manifest_hash = data_module.SPLIT_MANIFEST_HASH
    manifest = {
        "records": [{"sample_id": "duplicate", "row_index": 0, "split": "validation"}],
    }
    monkeypatch.setattr(data_module.ToxAcuteDataStore, "resolve", lambda unused: store)
    monkeypatch.setattr(data_module, "load_manifest", lambda unused: manifest)
    monkeypatch.setattr(data_module, "manifest_hash", lambda unused: SPLIT_MANIFEST_HASH)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        data_module._open_validated("unused")
    assert store.closed


def test_inference_afp_lazy_reader_closes_and_can_reopen_validation(baseline_datastore):
    partition = load_authorized_inference_split(baseline_datastore, "afp", split="validation")
    owner = partition.graph_records._owner
    assert partition.graph_records[0]["sample_id"] == partition.sample_ids[0]
    assert owner.is_open
    partition.graph_records.close()
    assert not owner.is_open
    reloaded = load_authorized_inference_split(baseline_datastore, "afp", split="validation")
    try:
        assert reloaded.graph_records[0]["sample_id"] == reloaded.sample_ids[0]
    finally:
        reloaded.graph_records.close()


def test_calibration_refuses_even_a_valid_test_grant(authorized_rf_context):
    with pytest.raises(PermissionError, match="Calibration"):
        load_authorized_inference_split(
            "unused",
            "rf",
            split="calibration",
            test_grant=authorized_rf_context.grant,
        )


def test_test_refuses_without_validated_grant():
    with pytest.raises(PermissionError, match="fully validated"):
        load_authorized_inference_split("unused", "rf", split="test")


@pytest.mark.parametrize("split", ["calibration", "test"])
def test_checkpoint_prediction_keeps_locked_splits_closed(split):
    with pytest.raises(PermissionError, match="remain locked"):
        load_authorized_validation("unused", "rf", split=split)
