"""Dedup contract: one LMDB graph per row, labels shared across all tasks."""

import numpy as np
import pandas as pd

from toxacute_datastore import ToxAcuteDataStore, build_datastore_v2


def _write_raw(path, n_tasks=2):
    smiles = ["CC", "CCC", "CCO", "CCN", "c1ccccc1", "CCCl"]
    frame = pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(len(smiles))],
            "Pubchem CID": list(range(len(smiles))),
            "IUPAC Name": ["" for _ in smiles],
            "SMILES": [""] * len(smiles),
            "smiles": smiles,
            "InChIKey": ["" for _ in smiles],
        }
    )
    # Distinct, fully observed labels per task so columns never alias.
    for column_index, task_name in enumerate([f"task_{name}" for name in "abcdefgh"[:n_tasks]]):
        frame[task_name] = [value + (column_index + 1) * 10.0 for value in range(len(smiles))]
    return frame.to_csv(path, index=False), [f"task_{name}" for name in "abcdefgh"[:n_tasks]]


def test_one_graph_per_row_shared_across_tasks(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv, n_tasks=2)
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
    )
    store = ToxAcuteDataStore.resolve(root)
    try:
        num_rows = len(store.row_indices)
        # Same six raw rows, two tasks: the store still holds six graphs.
        assert store.metadata["num_samples"] == num_rows == 6
        assert store.lmdb_entry_count() == 6
        assert store.labels.shape == (6, 2)
        assert store.metadata["task_names"] == ["task_a", "task_b"]

        # A molecule present in both tasks resolves to one identical record,
        # while its per-task labels stay independent.
        row_2 = int(np.where(store.sample_ids == "row_2")[0][0])
        record = store.get_graph_record(row_2)
        assert record["canonical_smiles"] == "CCO"
        # Fixture columns: task_a = row + 10, task_b = row + 20.
        assert store.get_label(row_2, "task_a") == 12.0
        assert store.get_label(row_2, "task_b") == 22.0
    finally:
        store.close()


def test_graphs_are_never_duplicated_for_multiple_task_views(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv, n_tasks=3)
    build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b", "task_c"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    try:
        assert store.labels.shape == (6, 3)
        assert store.lmdb_entry_count() == 6
        # Every task view indexes the same global rows.
        for task_name in ("task_a", "task_b", "task_c"):
            assert set(map(int, store.get_task_indices(task_name))) == set(range(6))
    finally:
        store.close()
