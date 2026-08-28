"""Pre-LMDB manifest provenance contract (review §26-30, §48)."""

import pytest

from toxacute_datastore import _split_report, _verify_approved_manifest


def _record(sample_id, row_index, smiles="CCO", scaffold="CCO"):
    return {
        "sample_id": sample_id,
        "row_index": row_index,
        "canonical_smiles": smiles,
        "scaffold": scaffold,
        "split": "train",
        "split_group": scaffold,
    }


def _manifest(records, sha="a" * 64):
    return {
        "manifest_version": 3,
        "split_algorithm": "constrained_scaffold_v3",
        "splitting": "scaffold",
        "seed": 42,
        "ratios": {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        "source_csv_sha256": sha,
        "records": records,
    }


def _csv_records():
    return [
        _record("row_0", 0),
        _record("row_1", 1, smiles="CCN", scaffold="CCN"),
        _record("row_2", 2, smiles="CCC", scaffold="CCC"),
    ]


def _verify(manifest, records=None, sha="a" * 64):
    _verify_approved_manifest(
        manifest,
        records if records is not None else _csv_records(),
        raw_csv_sha256=sha,
        splitting="scaffold",
        split_seed=42,
        ratios={"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
    )


def test_valid_manifest_passes():
    _verify(_manifest(_csv_records()))


def test_missing_csv_row_fails():
    manifest = _manifest(_csv_records())
    manifest["records"] = manifest["records"][:2]
    with pytest.raises(ValueError, match="record count"):
        _verify(manifest)


def test_duplicate_manifest_row_fails():
    records = _csv_records()
    records.append(dict(records[0]))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _verify(_manifest(records))


def test_reordered_manifest_fails_sequence_contract():
    records = list(reversed(_csv_records()))
    with pytest.raises(ValueError, match="sequence"):
        _verify(_manifest(records))


def test_changed_chemistry_fails():
    manifest = _manifest(_csv_records())
    manifest["records"][1]["canonical_smiles"] = "CCX"
    with pytest.raises(ValueError, match="canonical_smiles"):
        _verify(manifest)


def test_raw_csv_sha_mismatch_fails():
    with pytest.raises(ValueError, match="sha256"):
        _verify(_manifest(_csv_records(), sha="a" * 64), sha="b" * 64)


def test_extra_csv_row_missing_from_manifest_fails():
    csv_records = _csv_records()
    csv_records.append(_record("row_3", 3, smiles="CCF", scaffold="CCF"))
    with pytest.raises(ValueError, match="record count"):
        _verify(_manifest(_csv_records()), records=csv_records)


def _report(manifest, presence_rows):
    import numpy as np

    return _split_report(
        manifest,
        num_samples=len(manifest["records"]),
        parse_failures=0,
        raw_csv_sha256="a" * 64,
        labels_presence=np.array(presence_rows, dtype=np.int64),
        record_sample_ids=[f"row_{index}" for index in range(len(presence_rows))],
        task_names=["human_oral_TDLo"],
        priority_task_names=["human_oral_TDLo"],
        conformal_task_names=[],
        conformal_alpha=0.10,
        splitting="random",
    )


def test_label_presence_aligns_by_sample_id_not_position():
    # Manifest records deliberately reordered relative to the presence
    # matrix rows: row_1 (labelled) listed first, row_0 (unlabelled) second.
    manifest = {
        "manifest_version": 3,
        "ratios": {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
        "records": [
            {"sample_id": "row_1", "split": "calibration", "scaffold": "CCN", "split_group": "CCN"},
            {"sample_id": "row_0", "split": "train", "scaffold": "CCO", "split_group": "CCO"},
        ],
    }
    presence_rows = [
        [0],  # row_0: no label
        [1],  # row_1: labelled
    ]

    report = _report(manifest, presence_rows)

    assert report["human3_counts"]["calibration"]["human_oral_TDLo"] == 1
