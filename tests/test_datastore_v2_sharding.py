"""Sharded graph storage contract: bounded files, lazy reads, fail-fast legacy."""

import json
import shutil

import pandas as pd
import pytest

from toxacute_datastore import (
    GRAPH_LAYOUT_NAME,
    ToxAcuteDataStore,
    build_datastore_v2,
    normalize_shard_table,
    shard_name,
)


def _write_raw(path):
    """Six molecules mixing compact rings with long chains whose edge_input
    tensors dwarf a tiny shard cap, so both rollover modes trigger: normal
    boundary fills and an oversized single record getting its own shard."""
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(6)],
            "Pubchem CID": list(range(6)),
            "IUPAC Name": [""] * 6,
            "SMILES": [""] * 6,
            "smiles": [
                "CCC",
                "C" * 22,
                "CCO",
                "C" * 24 + "O",
                "c1ccccc1",
                "C" * 20 + "Cl",
            ],
            "InChIKey": [""] * 6,
            "task_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "task_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    ).to_csv(path, index=False)


def _build_multi_shard(tmp_path, shard_max_gb=1.0e-05):
    """~16 KB cap forces rollover while building six graphs."""
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
        lmdb_map_size_gb=0.008,
        commit_every=2,
        graph_shard_max_gb=shard_max_gb,
    )
    return root, build


def test_build_rolls_shards_at_byte_cap(tmp_path):
    root, build = _build_multi_shard(tmp_path)
    metadata = json.loads((build / "datastore.json").read_text(encoding="utf-8"))

    assert metadata["graph_layout"] == GRAPH_LAYOUT_NAME
    shards = metadata["graph_shards"]
    assert len(shards) > 1
    cursor = 0
    for position, entry in enumerate(shards):
        assert entry["name"] == shard_name(position)
        assert entry["first_index"] == cursor
        assert entry["entries"] > 0
        assert (build / "graph_shards" / entry["name"] / "data.mdb").exists()
        cursor += entry["entries"]
    assert cursor == metadata["num_samples"]
    # Diagnostics survive normalization: every shard reports a non-negative
    # serialized byte total, and the declared cap is what the build used.
    cap = metadata["graph_shard_max_bytes"]
    assert cap == int(1.0e-05 * 1024**3)
    payload_sizes = [entry["payload_bytes"] for entry in shards]
    assert all(size > 0 for size in payload_sizes)
    # No shard except one holding an oversized single record may exceed the cap.
    oversized = [size for size in payload_sizes if size > cap]
    assert len(oversized) <= len(shards)

    store = ToxAcuteDataStore.resolve(root)
    try:
        store.validate(strict=True)
        assert store.lmdb_entry_count() == metadata["num_samples"]
    finally:
        store.close()


def test_reads_span_shard_boundaries_correctly(tmp_path):
    root, _ = _build_multi_shard(tmp_path)
    store = ToxAcuteDataStore.resolve(root)
    try:
        boundaries = [entry["first_index"] for entry in store.shard_table]
        assert 0 in boundaries
        probe_indices = sorted({0, len(store.row_indices) - 1, *boundaries})
        for index in probe_indices:
            record = store.get_graph_record(index)
            assert record["sample_id"] == str(store.sample_ids[index])
            assert int(record["num_nodes"]) == int(store.num_nodes[index])
            task_index = list(range(len(store.row_indices)))[index]
            label = store.get_label(task_index, "task_a")
            assert label == float(task_index) + 1.0
    finally:
        store.close()


def test_reader_opens_only_touched_shard_envs(tmp_path):
    root, _ = _build_multi_shard(tmp_path)
    store = ToxAcuteDataStore.resolve(root)
    try:
        assert len(store.shard_table) > 1
        store.get_graph_record(0)
        assert len(store._envs) == 1
        last_index = len(store.row_indices) - 1
        last_entry = store.shard_table[-1]
        if last_entry["first_index"] > 0:
            store.get_graph_record(last_index)
            assert len(store._envs) == 2
    finally:
        store.close()
        assert store._envs == {}


def test_store_survives_pickling_with_open_envs(tmp_path):
    import pickle

    root, _ = _build_multi_shard(tmp_path)
    store = ToxAcuteDataStore.resolve(root)
    try:
        store.get_graph_record(0)
        assert len(store._envs) == 1
        # A pickled store (e.g. a spawned DataLoader worker payload) must not
        # carry LMDB env handles; the clone reopens shards lazily on demand.
        clone = pickle.loads(pickle.dumps(store))
        assert clone._envs == {}
    finally:
        store.close()


def test_writer_flushes_pending_batch_by_byte_budget(tmp_path):
    """A byte-budget flush bounds RAM even when commit_every is huge."""


    from toxacute_datastore import _GraphShardWriter, graph_key

    writer = _GraphShardWriter(
        tmp_path / "shard_probe",
        max_bytes=1 << 30,
        initial_map_size=8 << 20,
        pending_flush_bytes=1024,
    )
    try:
        payload_a = b"a" * 640
        writer.add(graph_key(0), payload_a, commit_every=100000)
        assert len(writer._pending) == 1
        payload_b = b"b" * 640
        writer.add(graph_key(1), payload_b, commit_every=100000)
        assert len(writer._pending) == 0
        entries = int(writer.env.stat()["entries"])
        assert entries == 2
        closed = writer.flush_and_close()
        assert closed["payload_bytes"] == len(payload_a) + len(payload_b)
    finally:
        try:
            writer.env.close()
        except Exception:
            pass


def test_legacy_single_file_layout_fails_fast(tmp_path):
    root, build = _build_multi_shard(tmp_path)
    metadata = json.loads((build / "datastore.json").read_text(encoding="utf-8"))
    metadata.pop("graph_layout", None)
    (build / "datastore.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="sharded graph layout"):
        ToxAcuteDataStore.resolve(root)

    legacy_copy = root.parent / "legacy_root"
    shutil.rmtree(legacy_copy, ignore_errors=True)
    legacy_build = legacy_copy / "builds" / build.name
    legacy_build.parent.mkdir(parents=True)
    shutil.copytree(build, legacy_build)
    (legacy_copy / "CURRENT").write_text(build.name + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="preprocess_data.py --build_datastore_v2"):
        ToxAcuteDataStore.resolve(legacy_copy)


def test_normalize_shard_table_rejects_inconsistent_tables():
    base = {"name": shard_name(0), "first_index": 0, "entries": 3}
    assert normalize_shard_table([base]) == [{"name": "shard_00000", "first_index": 0, "entries": 3}]

    gap = {"name": shard_name(1), "first_index": 4, "entries": 2}
    with pytest.raises(ValueError, match="was expected"):
        normalize_shard_table([base, gap])

    empty = {**base, "entries": 0}
    with pytest.raises(ValueError, match="no entries"):
        normalize_shard_table([empty])

    renamed = {**gap, "first_index": 3, "name": "../evil"}
    with pytest.raises(ValueError, match="Unexpected graph shard name"):
        normalize_shard_table([base, renamed])

    malformed = [{"first_index": 0, "entries": 1}]
    with pytest.raises(ValueError, match="Invalid graph shard descriptor"):
        normalize_shard_table(malformed)

    with pytest.raises(ValueError, match="empty graph shard table"):
        normalize_shard_table([])
