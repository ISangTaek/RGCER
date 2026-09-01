"""D7 selector aggregation contracts (fifth review §62; plan §49-§56, §61-§66, §82-§83).

Covers: Stage-B conditional B0A (only for the CSDT family), the
b1-confirmation fallback branch, duplicate-epoch rejection, stage identity
cross-checks (manifest/datastore), the Stage-B women gate, and the D7-0 CLST
audit's matched-teacher contract.
"""

import json

import pytest

from tests.test_d6_aggregate import _flat_macro, _write_run

HUMAN_TASKS = ["man_oral_TDLo", "women_oral_TDLo", "human_oral_TDLo"]
ENDPOINTS = {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.40, "human_oral_TDLo": 1.30}


def _write_d6_ref(root, reference, seed, stable, *, stage="d6_stage_b", epochs=40, women=1.40):
    last = epochs - 1
    endpoints = {
        "man_oral_TDLo": 1.0,
        "women_oral_TDLo": women,
        "human_oral_TDLo": 1.3,
    }
    _write_run(
        root / stage,
        reference,
        f"d6_{reference}_e{epochs}",
        seed,
        _flat_macro(last, stable),
        endpoint_by_epoch={epoch: dict(endpoints) for epoch in range(last - 4, epochs)},
        args_epochs=epochs,
    )


def _write_d7_run(
    root,
    candidate,
    seed,
    stable,
    *,
    stage="d7_stage_b",
    epochs=40,
    women=1.40,
    manifest="manifest-test",
    fingerprint="datastore-test",
    duplicate_epoch=False,
    contract=True,
    mode=None,
):
    tag = f"d7_{candidate}_e{epochs}"
    run_dir = root / stage / candidate / tag / f"seed_{seed}"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    freeze = {"s1": epochs, "s2": 0, "s3": 5}.get(candidate, 0)
    multiplier = 0.1 if candidate in ("s2", "s3") else 1.0
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "dataset": "toxacute",
                "toxacute_task_scope": "human3",
                "train_eval_scope": "validation_only",
                "fit_conformal": False,
                "epochs": epochs,
                "split_seed": 42,
                "freeze_backbone_epochs": freeze,
                "backbone_lr_multiplier": multiplier,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": candidate,
        "manifest_sha256": manifest,
        "datastore_fingerprint": fingerprint,
        "feature_schema_version": "schema-test",
    }
    if contract:
        metadata["d7_artifact_contract"] = {
            "mode": mode or {"s1": "b1", "s2": "b1", "s3": "b1", "o4": "csdt", "o5": "csdt_bounded"}[candidate],
            "human_seed": seed,
            "split_manifest_hash": manifest,
            "datastore_fingerprint": fingerprint,
            "feature_schema_version": "schema-test",
        }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    last = epochs - 1
    with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,val_human3_macro_rmse\n")
        written = set()
        for epoch in range(epochs):
            handle.write(f"{epoch},{stable}\n")
            written.add(epoch)
            if duplicate_epoch and epoch == last - 1:
                handle.write(f"{epoch},{stable}\n")  # duplicate row (resume artifact)
    with (diagnostics / "human3_path_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,task,path,rmse\n")
        for epoch in range(last - 4, epochs):
            for task, value in {
                "man_oral_TDLo": 1.0,
                "women_oral_TDLo": women,
                "human_oral_TDLo": 1.3,
            }.items():
                handle.write(f"{epoch},{task},final,{value}\n")
    return run_dir


def _write_stage_b_world(tmp_path, *, candidates=("s2", "s3"), b0a_present=True):
    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    for seed, b1_stable in ((42, 1.2355), (44, 1.2400), (46, 1.2300)):
        _write_d6_ref(d6_root, "b0", seed, 1.2648 + 0.001 * (seed % 3), women=1.4226)
        _write_d6_ref(d6_root, "b1", seed, b1_stable, women=1.4816)
        if b0a_present:
            _write_d6_ref(d6_root, "b0a", seed, 1.2648, women=1.4226)
    for candidate, stable in {
        "s2": 1.22,
        "s3": 1.225,
        "o4": 1.21,
        "o5": 1.215,
    }.items():
        if candidate in candidates:
            for seed in (42, 44, 46):
                _write_d7_run(d7_root, candidate, seed, stable)
    return d6_root, d7_root


# ----------------------------------------------------------------------
# P1-1: Stage B conditional B0A
# ----------------------------------------------------------------------
def test_stage_b_does_not_require_b0a_for_s_only_top2(tmp_path):
    from scripts.d7_aggregate import stage_b

    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s2", "s3"), b0a_present=False)
    trace = {}
    result = stage_b(d7_root, d6_root, ["s2", "s3"], [42, 44, 46], tmp_path, trace)
    assert result["gates"]["s2"]["gate_pass"] is True
    assert result["gates"]["s3"]["gate_pass"] is True


def test_stage_b_requires_b0a_if_o4_present(tmp_path):
    from scripts.d7_aggregate import stage_b

    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s2", "o4"), b0a_present=False)
    with pytest.raises(SystemExit, match="b0a"):
        stage_b(d7_root, d6_root, ["s2", "o4"], [42, 44, 46], tmp_path, {})


def test_stage_b_requires_b0a_if_o5_present(tmp_path):
    from scripts.d7_aggregate import stage_b

    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s3", "o5"), b0a_present=False)
    with pytest.raises(SystemExit, match="b0a"):
        stage_b(d7_root, d6_root, ["s3", "o5"], [42, 44, 46], tmp_path, {})


def test_stage_b_o4_passes_anchor_gate_when_b0a_present(tmp_path):
    from scripts.d7_aggregate import stage_b

    # B0A=1.2648 vs O4=1.21 -> positive anchor semantic gain.
    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s2", "o4"), b0a_present=True)
    trace = {}
    result = stage_b(d7_root, d6_root, ["s2", "o4"], [42, 44, 46], tmp_path, trace)
    assert result["gates"]["o4"]["anchor_semantic_gain_mean"] > 0
    assert result["gates"]["o4"]["gate_pass"] is True


# ----------------------------------------------------------------------
# P1-2: b1-confirmation branch
# ----------------------------------------------------------------------
def test_b1_confirmation_three_seed_gate_pass(tmp_path):
    from scripts.d7_aggregate import b1_confirmation

    d6_root = tmp_path / "d6"
    for seed, (b0, b1) in ((42, (1.2648, 1.2355)), (44, (1.2700, 1.2400)), (46, (1.2600, 1.2300))):
        _write_d6_ref(d6_root, "b0", seed, b0)
        _write_d6_ref(d6_root, "b1", seed, b1)
    trace = {}
    result = b1_confirmation(d6_root, [42, 44, 46], tmp_path, trace)
    assert result["gate_pass"] is True
    assert result["positive_seeds"] == 3
    assert result["mean_stable_gain"] == pytest.approx(0.0298 + 0.0063, abs=0.01)


def test_b1_confirmation_failure_reports_transfer_engineering_stop(tmp_path):
    from scripts.d7_aggregate import b1_confirmation

    d6_root = tmp_path / "d6"
    # B1 beats B0 only at seed 44 -> mean gain <= 0 and <2/3 positive.
    for seed, (b0, b1) in ((42, (1.2300, 1.2600)), (44, (1.2700, 1.2400)), (46, (1.2600, 1.2900))):
        _write_d6_ref(d6_root, "b0", seed, b0)
        _write_d6_ref(d6_root, "b1", seed, b1)
    trace = {}
    result = b1_confirmation(d6_root, [42, 44, 46], tmp_path, trace)
    assert result["gate_pass"] is False
    assert "STOP_ANIMAL_TO_HUMAN_TRANSFER_ENGINEERING" in result["stop_reason"]
    assert (tmp_path / "D7B_B1_CONFIRMATION.csv").exists()
    assert (tmp_path / "D7B_B1_CONFIRMATION_TRACE.json").exists()


# ----------------------------------------------------------------------
# P1-7: duplicate-epoch guard
# ----------------------------------------------------------------------
def test_d7_rejects_duplicate_epoch(tmp_path):
    from scripts.d7_aggregate import summarize_run

    run_dir = _write_d7_run(tmp_path, "s2", 42, 1.20, duplicate_epoch=True)
    row = summarize_run(run_dir, "s2", 42, 39)
    assert row["status"] == "DUPLICATE_EPOCHS"


# ----------------------------------------------------------------------
# P1-6: stage identity cross-check
# ----------------------------------------------------------------------
def test_d7_rejects_mixed_manifest(tmp_path):
    from scripts.d7_aggregate import stage_b

    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s2", "s3"), b0a_present=False)
    s3_run = d7_root / "d7_stage_b" / "s3" / "d7_s3_e40" / "seed_42" / "run_metadata.json"
    metadata = json.loads(s3_run.read_text(encoding="utf-8"))
    metadata["manifest_sha256"] = "manifest-other"
    metadata["d7_artifact_contract"]["split_manifest_hash"] = "manifest-other"
    s3_run.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SystemExit, match="data identity"):
        stage_b(d7_root, d6_root, ["s2", "s3"], [42, 44, 46], tmp_path, {})


def test_d7_rejects_mixed_datastore(tmp_path):
    from scripts.d7_aggregate import stage_b

    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s2", "s3"), b0a_present=False)
    s2_run = d7_root / "d7_stage_b" / "s2" / "d7_s2_e40" / "seed_42" / "run_metadata.json"
    metadata = json.loads(s2_run.read_text(encoding="utf-8"))
    metadata["datastore_fingerprint"] = "datastore-other"
    metadata["d7_artifact_contract"]["datastore_fingerprint"] = "datastore-other"
    s2_run.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(SystemExit, match="data identity"):
        stage_b(d7_root, d6_root, ["s2", "s3"], [42, 44, 46], tmp_path, {})


# ----------------------------------------------------------------------
# P1-8: Stage-B women gate
# ----------------------------------------------------------------------
def test_stage_b_women_gate(tmp_path):
    from scripts.d7_aggregate import stage_b

    # S3 wins on macro but its women mean (1.60) sits 0.118 above the B1
    # women mean (1.4816) — beyond the 0.03 margin, so the gate must fail
    # even though 2/3 endpoints are non-worse.
    d6_root, d7_root = _write_stage_b_world(tmp_path, candidates=("s2", "s3"), b0a_present=False)
    for seed in (42, 44, 46):
        s3_run = d7_root / "d7_stage_b" / "s3" / "d7_s3_e40" / f"seed_{seed}"
        _rewrite_endpoints(s3_run, women=1.60)
    trace = {}
    result = stage_b(d7_root, d6_root, ["s2", "s3"], [42, 44, 46], tmp_path, trace)
    assert result["gates"]["s3"]["women_gate_pass"] is False
    assert result["gates"]["s3"]["gate_pass"] is False
    assert result["gates"]["s3"]["women_degradation_vs_b1"] > 0.03
    # S2 keeps women at the B0 level (1.40 < B1's 1.4816) -> gate open.
    assert result["gates"]["s2"]["women_gate_pass"] is True


def _rewrite_endpoints(run_dir, women):
    path = run_dir / "diagnostics" / "human3_path_metrics.csv"
    lines = ["epoch,task,path,rmse\n"]
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        epoch, task, path_name, value = line.split(",")
        if task == "women_oral_TDLo":
            value = str(women)
        lines.append(f"{epoch},{task},{path_name},{value}\n")
    path.write_text("".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------
# §63: D7-0 CLST audit matched-teacher contract
# ----------------------------------------------------------------------
from tests.test_d6_counterfactual import _teacher_payload, _write_teacher  # noqa: E402

from scripts.d7_clst_audit import verify_audit_teachers  # noqa: E402


def _audit_pair(tmp_path, *, shuffle_seed=42, manifest=None, shuffle_init="init-hash"):
    real = _write_teacher(tmp_path, "real", _teacher_payload(False))
    shuffle_payload = _teacher_payload(True, seed=shuffle_seed)
    if manifest is not None:
        shuffle_payload["split_manifest_hash"] = manifest
    shuffle = _write_teacher(tmp_path, "shuffle", shuffle_payload, init_hash=shuffle_init)
    return real, shuffle


def test_d7_clst_audit_requires_matched_teacher_seed(tmp_path):
    real, shuffle = _audit_pair(tmp_path, shuffle_seed=43)
    with pytest.raises(SystemExit, match="base seed"):
        verify_audit_teachers(real, shuffle, 29, expected_model_seed=42)


def test_d7_clst_audit_requires_same_manifest(tmp_path):
    real, shuffle = _audit_pair(tmp_path, manifest="other-manifest")
    with pytest.raises(SystemExit, match="split_manifest_hash"):
        verify_audit_teachers(real, shuffle, 29, expected_model_seed=42)


def test_d7_clst_audit_requires_same_initial_hash(tmp_path):
    real, shuffle = _audit_pair(tmp_path, shuffle_init="different-hash")
    with pytest.raises(SystemExit, match="initial_model_sha256"):
        verify_audit_teachers(real, shuffle, 29, expected_model_seed=42)


def test_d7_clst_audit_accepts_matched_pair(tmp_path):
    real, shuffle = _audit_pair(tmp_path)
    matched = verify_audit_teachers(real, shuffle, 29, expected_model_seed=42)
    assert matched["teacher_initial_model_sha256"] == "init-hash"
    for key in (
        "split_manifest_hash",
        "datastore_fingerprint",
        "feature_schema_version",
        "animal_shuffle_mapping_sha256",
    ):
        assert matched.get(key) not in (None, ""), key

def test_clst_direct_audit_uses_exact_probe_set():
    # Sixth-review §26: audit n_samples must equal the probe size exactly —
    # never the whole loader batch that happened to contain a probe molecule.
    from scripts.d7_clst_audit import _fixed_probe_batches
    from tests.test_d7_candidates import _StubLoader

    loaders = {"task_a": _StubLoader([f"id{i}" for i in range(12)])}
    batches, probe_ids = _fixed_probe_batches(loaders, probe_size=8)
    assert len(probe_ids) == 8
    assert probe_ids == sorted(probe_ids)
    total = sum(len(batch.sample_id) for batch in batches)
    assert total == 8
    observed = {sid for batch in batches for sid in batch.sample_id}
    assert observed == set(probe_ids)
