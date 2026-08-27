"""Plan-before-build flow and approved-manifest provenance (review §30-35)."""

import pandas as pd
import pytest

from toxacute_datastore import (
    _verify_approved_manifest,
    build_datastore_v2,
    plan_toxacute_split,
)


def _write_raw(path, rows=30):
    pd.DataFrame(
        {
            "TAID": [f"T-{i}" for i in range(rows)],
            "Pubchem CID": list(range(rows)),
            "IUPAC Name": [""] * rows,
            "SMILES": [""] * rows,
            "smiles": [f"{'C' * (i % 6) or 'C'}CO" if i % 5 else f"CC(F){'C' * i}N" for i in range(rows)],
            "InChIKey": [""] * rows,
            "task_a": [float(i % 4) for i in range(rows)],
            "task_b": [float(i % 7) for i in range(rows)],
        }
    ).to_csv(path, index=False)


def test_plan_split_only_writes_v3_manifest_and_passing_report(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    _write_raw(raw_csv)

    manifest, report = plan_toxacute_split(
        raw_csv,
        task_names=["task_a", "task_b"],
        splitting="scaffold",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        conformal_alpha=0.10,
        num_candidates=8,
    )

    assert manifest["manifest_version"] == 3
    assert manifest["split_algorithm"] == "constrained_scaffold_v3"
    assert len(manifest["source_csv_sha256"]) == 64
    assert report["status"] == "PASS"
    assert report["hard_constraint_failures"] == []
    assert report["valid_molecules"] == len(manifest["records"])
    assert set(report["split_counts"]) == {"train", "validation", "calibration", "test"}


def test_verify_approved_manifest_rejects_swapped_smiles(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    _write_raw(raw_csv)
    manifest, report = plan_toxacute_split(
        raw_csv,
        task_names=["task_a", "task_b"],
        splitting="scaffold",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
    )

    # §34: the same row index with a different SMILES must invalidate the
    # approved manifest even though every sample_id still resolves. The
    # sha256 gate (§33) fires first — bypass it to exercise the row-level
    # identity check directly.
    from toxacute_datastore import scan_chemistry

    tampered_csv = tmp_path / "tampered.csv"
    frame = pd.read_csv(raw_csv)
    frame.loc[0, "smiles"] = "CCCN"
    frame.to_csv(tampered_csv, index=False)
    scan = scan_chemistry(tampered_csv, ["task_a", "task_b"])

    planned_ratios = manifest["ratios"]
    with pytest.raises(ValueError, match="does not match the CSV"):
        _verify_approved_manifest(
            manifest,
            scan["records"],
            raw_csv_sha256=str(manifest["source_csv_sha256"]),
            splitting="scaffold",
            split_seed=42,
            ratios=planned_ratios,
        )

    # And the provenance hash itself rejects a genuinely different CSV.
    with pytest.raises(ValueError, match="source_csv_sha256 mismatch"):
        _verify_approved_manifest(
            manifest,
            scan["records"],
            raw_csv_sha256=scan["raw_csv_sha256"],
            splitting="scaffold",
            split_seed=42,
            ratios=planned_ratios,
        )


def test_build_accepts_and_persists_approved_manifest(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    manifest, report = plan_toxacute_split(
        raw_csv,
        task_names=["task_a", "task_b"],
        splitting="scaffold",
        valid_size=0.25,
        calibration_size=0.25,
        test_size=0.25,
        num_candidates=8,
    )
    assert report["status"] == "PASS"

    manifest_path = tmp_path / "approved.json"
    import json

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    build = build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="scaffold",
        valid_size=0.25,
        calibration_size=0.25,
        test_size=0.25,
        split_seed=42,
        max_path_distance=4,
        lmdb_map_size_gb=0.008,
        commit_every=4,
        graph_shard_max_gb=1.0e-05,
        split_manifest_path=manifest_path,
    )
    stored = json.loads((build / "split_report.json").read_text(encoding="utf-8"))
    assert stored["status"] == "PASS"
    assert (build / "READY").exists()
