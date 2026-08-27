from dataset import DataCollator, DataloaderWrapper
from tests.test_datastore_v2_contract import _write_raw
from toxacute_datastore import ToxAcuteDataStore, build_datastore_v2


def test_datastore_v2_loader_uses_lazy_multiworker_reads(tmp_path):
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
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    wrapper = DataloaderWrapper(
        task_list=["task_a"],
        data_store=store,
        batch_size=2,
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        num_workers=2,
        collate_fn_for_loader=DataCollator(),
        max_nodes_filter=20,
    )
    loaders = wrapper.get_data_loaders()["task_a"]
    batches = list(loaders["train"])
    assert batches
    assert all(not batch.is_empty for batch in batches)
    assert sum(batch.y.numel() for batch in batches) == len(loaders["train"].dataset)
    store.close()
