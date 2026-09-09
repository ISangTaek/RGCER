import numpy as np
import pytest

from baselines.constants import HUMAN3_TASKS
from baselines.data import (
    GraphRecordView,
    load_authorized_validation,
    load_formal_train_validation,
    load_human3_smoke,
)


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


@pytest.mark.parametrize("split", ["calibration", "test"])
def test_checkpoint_prediction_keeps_locked_splits_closed(split):
    with pytest.raises(PermissionError, match="remain locked"):
        load_authorized_validation("unused", "rf", split=split)
