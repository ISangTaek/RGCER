"""Atomic activation contract: READY -> rename -> CURRENT, idempotent rebuilds."""

import pandas as pd
import pytest

from toxacute_datastore import ToxAcuteDataStore, build_datastore_v2


def _write_raw(path, extra_row=None):
    smiles = ["CC", "CCC", "CCO", "CCN", "c1ccccc1", "CCCl"]
    task_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    task_b = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    if extra_row is not None:
        smiles.append(extra_row)
        task_a.append(7.0)
        task_b.append(70.0)
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(len(smiles))],
            "Pubchem CID": list(range(len(smiles))),
            "IUPAC Name": ["" for _ in smiles],
            "SMILES": [""] * len(smiles),
            "smiles": smiles,
            "InChIKey": ["" for _ in smiles],
            "task_a": task_a,
            "task_b": task_b,
        }
    ).to_csv(path, index=False)


def _build_args(tmp_path, root):
    return dict(
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )


def test_build_leaves_no_temp_artifacts_and_sets_current(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build = build_datastore_v2(raw_csv, root, **_build_args(tmp_path, root))

    assert (build / "READY").exists()
    assert (root / "CURRENT").exists()
    assert (root / "CURRENT").read_text(encoding="utf-8").strip() == build.name
    assert not list((root / "builds").glob(".tmp-*"))
    assert not list(root.glob(".CURRENT-*"))


def test_identical_rebuild_is_idempotent(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    first = build_datastore_v2(raw_csv, root, **_build_args(tmp_path, root))
    current_before = (root / "CURRENT").read_text(encoding="utf-8")

    second = build_datastore_v2(raw_csv, root, **_build_args(tmp_path, root))

    # Fingerprint is content-based, so identical input re-resolves to the
    # same build instead of creating a duplicate directory.
    assert second.name == first.name
    assert (root / "CURRENT").read_text(encoding="utf-8") == current_before
    assert [path.name for path in (root / "builds").iterdir()] == [first.name]
    assert not list((root / "builds").glob(".tmp-*"))


def test_changed_input_creates_new_build_and_switches_current(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    first = build_datastore_v2(raw_csv, root, **_build_args(tmp_path, root))

    raw_csv_v2 = tmp_path / "raw_v2.csv"
    _write_raw(raw_csv_v2, extra_row="CC(=O)O")
    second = build_datastore_v2(raw_csv_v2, root, **_build_args(tmp_path, root))

    assert second.name != first.name
    assert (root / "CURRENT").read_text(encoding="utf-8").strip() == second.name
    assert (first / "READY").exists(), "previous build must stay intact"
    assert (second / "READY").exists()
    assert not list((root / "builds").glob(".tmp-*"))

    store = ToxAcuteDataStore.resolve(root)
    try:
        assert store.root == second
        assert store.metadata["num_samples"] == 7
    finally:
        store.close()


def test_resolve_rejects_root_without_current(tmp_path):
    empty = tmp_path / "empty_root"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        ToxAcuteDataStore.resolve(empty)


def test_resolve_rejects_current_pointing_at_missing_build(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build_datastore_v2(raw_csv, root, **_build_args(tmp_path, root))

    (root / "CURRENT").write_text("toxacute-v2-000000000000\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ToxAcuteDataStore.resolve(root)
