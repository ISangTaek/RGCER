import json

import pytest

from scripts.aggregate_runs import aggregate_runs


def _write_run(root, seed, split_hash="split-1", lr=1e-3, auxiliary=False):
    run = root / f"seed_{seed}"
    run.mkdir()
    (run / "args.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "task_names": ["human_oral_TDLo"],
                "prediction_mode": "quantile",
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "conformal_alpha": 0.1,
                "fit_conformal": True,
                "lr": lr,
                "auxiliary_task_names": ["shuffled_human_oral_TDLo"] if auxiliary else [],
            }
        ),
        encoding="utf-8",
    )
    (run / "effective_config.json").write_text(
        json.dumps({"effective": {"architecture": "Graphormer_rgcer"}}), encoding="utf-8"
    )
    (run / "data_preflight.json").write_text(
        json.dumps(
            {
                "datastore_fingerprint": "store-1",
                "split_manifest_hash": split_hash,
                "raw_csv_sha256": "raw-1",
                "feature_schema_version": "schema-1",
                "max_path_distance": 8,
            }
        ),
        encoding="utf-8",
    )
    tasks = {"human_oral_TDLo": {"RMSE": float(seed)}}
    if auxiliary:
        tasks["shuffled_human_oral_TDLo"] = {"RMSE": 999.0}
    (run / "metrics.json").write_text(
        json.dumps({"test": {"tasks": tasks}}),
        encoding="utf-8",
    )


def test_aggregate_runs_checks_contract_and_writes_tables(tmp_path):
    _write_run(tmp_path, 42)
    _write_run(tmp_path, 3407)
    outputs = aggregate_runs(tmp_path)
    assert outputs["aggregate_metrics"].exists()
    assert "mean" in outputs["aggregate_metrics"].read_text(encoding="utf-8")


def test_aggregate_runs_rejects_split_mismatch(tmp_path):
    _write_run(tmp_path, 42, split_hash="split-1")
    _write_run(tmp_path, 3407, split_hash="split-2")
    with pytest.raises(ValueError, match="split_manifest_hash"):
        aggregate_runs(tmp_path)


def test_aggregate_runs_rejects_training_config_mismatch(tmp_path):
    _write_run(tmp_path, 42, lr=1e-3)
    _write_run(tmp_path, 3407, lr=2e-3)
    with pytest.raises(ValueError, match="experiment_config"):
        aggregate_runs(tmp_path)


def test_aggregate_runs_excludes_auxiliary_tasks_from_formal_tables(tmp_path):
    _write_run(tmp_path, 42, auxiliary=True)
    _write_run(tmp_path, 3407, auxiliary=True)
    outputs = aggregate_runs(tmp_path)

    aggregate = outputs["aggregate_metrics"].read_text(encoding="utf-8")
    human3 = outputs["human3_metrics"].read_text(encoding="utf-8")
    endpoint = outputs["per_endpoint_metrics"].read_text(encoding="utf-8")
    assert "999.0" not in aggregate
    assert "shuffled_human_oral_TDLo" not in human3
    assert "shuffled_human_oral_TDLo" in endpoint
