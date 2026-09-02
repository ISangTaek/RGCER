"""D8 aggregation contracts (plan §16-§19, §62-§79, §99).

Covers: the D8-0 5-seed formal gate (S1 vs B1 and S1 vs B0), endpoint and
selection-sensitivity checks, the D8-A adaptation-locus screen with
STOP_METHOD_ENGINEERING fallback, women/human/representation/functional gates,
and the D8-B 5-seed strong/modest confirmation.
"""

import json

import pytest

from tests.test_d6_aggregate import _flat_macro, _write_run

HUMAN_ENDPOINTS = {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.48, "human_oral_TDLo": 1.30}
D6_SEEDS = [42, 43, 44, 45, 46]


def _write_d6_ref(root, reference, seed, stable):
    _write_run(
        root / "d6_stage_b",
        reference,
        f"d6_{reference}_e40",
        seed,
        _flat_macro(39, stable),
        endpoint_by_epoch={epoch: dict(HUMAN_ENDPOINTS) for epoch in range(35, 40)},
        args_epochs=40,
    )


def _write_d7_s1_run(root, seed, stable, *, stage="d7_stage_b", epochs=40, endpoints=None):
    run_dir = root / stage / "s1" / f"d7_s1_e{epochs}" / f"seed_{seed}"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
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
                "freeze_backbone_epochs": epochs,
                "backbone_lr_multiplier": 1.0,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": "s1",
        "manifest_sha256": "manifest-test",
        "datastore_fingerprint": "datastore-test",
        "feature_schema_version": "schema-test",
        "split_seed": 42,
        "d7_artifact_contract": {
            "mode": "b1",
            "human_seed": seed,
            "split_manifest_hash": "manifest-test",
            "datastore_fingerprint": "datastore-test",
            "feature_schema_version": "schema-test",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,val_human3_macro_rmse\n")
        for epoch in range(epochs):
            handle.write(f"{epoch},{stable}\n")
    with (diagnostics / "human3_path_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,task,path,rmse\n")
        for epoch in range(epochs - 5, epochs):
            for task, value in (endpoints or HUMAN_ENDPOINTS).items():
                handle.write(f"{epoch},{task},final,{value}\n")
    return run_dir


def _write_d8_candidate_run(
    root,
    candidate,
    seed,
    stable,
    *,
    epochs=20,
    endpoints=None,
    women=None,
    feature_drift=0.01,
    forgetting_relative=0.03,
    with_mechanism=True,
    multiplier=None,
    blocks=None,
):
    run_dir = root / "d8_stage_a" if epochs == 20 else root / "d8_stage_b"
    tag = f"d8_{candidate}_e{epochs}"
    run_dir = run_dir / candidate / tag / f"seed_{seed}"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    scope = {"a1": (1, 0.01), "a2": (2, 0.01), "o6": (1, 0.02)}[candidate]
    endpoint_values = dict(endpoints or HUMAN_ENDPOINTS)
    if women is not None:
        endpoint_values["women_oral_TDLo"] = women
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
                "freeze_backbone_epochs": 0,
                "trainable_last_blocks": blocks if blocks is not None else scope[0],
                "backbone_lr_multiplier": multiplier if multiplier is not None else scope[1],
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": candidate,
        "manifest_sha256": "manifest-test",
        "datastore_fingerprint": "datastore-test",
        "feature_schema_version": "schema-test",
        "split_seed": 42,
        "d7_artifact_contract": {
            "mode": "b1",
            "human_seed": seed,
            "split_manifest_hash": "manifest-test",
            "datastore_fingerprint": "datastore-test",
            "feature_schema_version": "schema-test",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,val_human3_macro_rmse\n")
        for epoch in range(epochs):
            handle.write(f"{epoch},{stable}\n")
    with (diagnostics / "human3_path_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,task,path,rmse\n")
        for epoch in range(max(epochs - 5, 0), epochs):
            for task, value in endpoint_values.items():
                handle.write(f"{epoch},{task},final,{value}\n")
    if with_mechanism:
        with (diagnostics / "d7_representation_drift.csv").open("w", encoding="utf-8") as handle:
            handle.write("epoch,state,drift_anchor_type,backbone_param_drift,early_block_drift,late_block_drift,feature_drift,human3_rmse\n")
            handle.write(f"-1,baseline,animal_pretrained,0,0,0,0,\n")
            handle.write(f"{epochs - 1},post_epoch,animal_pretrained,0.005,0.004,0.006,{feature_drift},\n")
        if forgetting_relative is not None:
            (run_dir / "functional_forgetting.json").write_text(
                json.dumps(
                    {
                        "functional_forgetting_relative": forgetting_relative,
                        "functional_forgetting_abs": 0.04,
                        "functional_forgetting_exact_zero": False,
                        "provenance_verified": True,
                        "animal_tasks_worsened": 2,
                        "animal_tasks_improved": 1,
                        "animal_tasks_unchanged": 53,
                        "delta_q25": -0.01,
                        "delta_median": 0.0,
                        "delta_q75": 0.01,
                        "delta_q90": 0.02,
                        "delta_max": 0.03,
                        "per_task_delta": {f"t{i}": 0.001 * i for i in range(56)},
                    }
                ),
                encoding="utf-8",
            )
    return run_dir


def _write_s1_20e(root, seed, stable, endpoints=None):
    endpoint_values = dict(endpoints or HUMAN_ENDPOINTS)
    run_dir = root / "d7_stage_a" / "s1" / "d7_s1_e20" / f"seed_{seed}"
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "dataset": "toxacute",
                "toxacute_task_scope": "human3",
                "train_eval_scope": "validation_only",
                "fit_conformal": False,
                "epochs": 20,
                "split_seed": 42,
                "freeze_backbone_epochs": 20,
                "backbone_lr_multiplier": 1.0,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": "s1",
        "manifest_sha256": "manifest-test",
        "datastore_fingerprint": "datastore-test",
        "feature_schema_version": "schema-test",
        "split_seed": 42,
        "d7_artifact_contract": {
            "mode": "b1",
            "human_seed": seed,
            "split_manifest_hash": "manifest-test",
            "datastore_fingerprint": "datastore-test",
            "feature_schema_version": "schema-test",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,val_human3_macro_rmse\n")
        for epoch in range(20):
            handle.write(f"{epoch},{stable}\n")
    with (diagnostics / "human3_path_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,task,path,rmse\n")
        for epoch in range(15, 20):
            for task, value in endpoint_values.items():
                handle.write(f"{epoch},{task},final,{value}\n")
    return run_dir


# ----------------------------------------------------------------------
# D8-0: 5-seed formal gate
# ----------------------------------------------------------------------
def test_d8_0_gate_passes_when_s1_locks_the_win(tmp_path):
    from scripts.d8_aggregate import d8_0

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    s1_endpoints = {"man_oral_TDLo": 0.95, "women_oral_TDLo": 1.35, "human_oral_TDLo": 1.28}
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40 + 0.01 * index)
        _write_d6_ref(d6_root, "b1", seed, 1.25 + 0.01 * index)
        _write_d7_s1_run(d7_root, seed, 1.09, endpoints=s1_endpoints)
        _write_functional_forgetting_json(d6_root, "b1", seed, exact_zero=False)
        _write_functional_forgetting_json(d7_root, "s1", seed, exact_zero=True)
    trace = {}
    result = d8_0(d6_root, d7_root, D6_SEEDS, tmp_path, trace)
    assert result["gate_pass"] is True
    assert result["s1_vs_b1"]["positive"] == 5
    assert result["s1_vs_b1"]["mean"] > 0.05
    assert result["s1_vs_b0"]["mean"] > 0.05
    assert (tmp_path / "D8_0_GATE.json").exists()
    assert (tmp_path / "D8_0_5SEED_SUMMARY.csv").exists()


def test_d8_0_gate_fails_when_a_seed_reverses(tmp_path):
    from scripts.d8_aggregate import d8_0

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        # seed45: B1 beats S1 -> only 4/5 positive and mean dips below 0.05.
        b1_stable = 1.02 if seed == 45 else 1.25 + 0.005 * index
        _write_d6_ref(d6_root, "b1", seed, b1_stable)
        _write_d7_s1_run(d7_root, seed, 1.09)
    trace = {}
    result = d8_0(d6_root, d7_root, D6_SEEDS, tmp_path, trace)
    assert result["gate_pass"] is False
    assert "STOP" in result["stop_reason"]


def test_d8_0_marks_selection_sensitive(tmp_path):
    from scripts.d8_aggregate import d8_0

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    s1_endpoints = {"man_oral_TDLo": 0.95, "women_oral_TDLo": 1.35, "human_oral_TDLo": 1.28}
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        _write_d6_ref(d6_root, "b1", seed, 1.25)
        # S1 stable wins, but its BEST rmse is worse than B1's best: B1's
        # trajectory dips early (best ~1.00) then degrades to its 1.25 stable.
        _write_d7_s1_run(d7_root, seed, 1.09, endpoints=s1_endpoints)
        b1_summary = (
            d6_root / "d6_stage_b" / "b1" / "d6_b1_e40" / f"seed_{seed}" / "diagnostics" / "epoch_summary.csv"
        )
        with b1_summary.open("w", encoding="utf-8") as handle:
            handle.write("epoch,val_human3_macro_rmse\n")
            for epoch in range(40):
                handle.write(f"{epoch},{1.00 if epoch < 30 else 1.25}\n")
        s1_run = d7_root / "d7_stage_b" / "s1" / "d7_s1_e40" / f"seed_{seed}"
        diagnostics = s1_run / "diagnostics"
        with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
            handle.write("epoch,val_human3_macro_rmse\n")
            for epoch in range(40):
                handle.write(f"{epoch},{1.30 if epoch < 35 else 1.09}\n")
    trace = {}
    result = d8_0(d6_root, d7_root, D6_SEEDS, tmp_path, trace)
    assert result["selection_sensitive"] is True


# ----------------------------------------------------------------------
# D8-A: adaptation-locus screen (§62-§66, §70/§106)
# ----------------------------------------------------------------------
D8_A_SEEDS = [42, 45]


def test_d8_a_stops_method_engineering_when_nothing_beats_s1(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
        for candidate in ("a1", "a2", "o6"):
            _write_d8_candidate_run(d8_root, candidate, seed, 1.10)  # worse than S1
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    assert result["top1"] is None
    assert result["selection_mode"] == "STOP_METHOD_ENGINEERING"
    assert "S1 is the final transfer strategy" in result["stop_reason"]


def test_d8_a_promotes_candidate_that_beats_s1(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
        _write_d8_candidate_run(d8_root, "a1", seed, 1.05)  # gain 0.02 both seeds
        _write_d8_candidate_run(d8_root, "a2", seed, 1.10)
        _write_d8_candidate_run(d8_root, "o6", seed, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    assert result["top1"] == "a1"
    assert result["selection_mode"] == "PROMOTED"
    assert result["gates"]["a1"]["gate_pass"] is True


def test_d8_a_human_endpoint_block(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
        # O6 wins macro but degrades human_oral_TDLo by >0.05 -> blocked (§64).
        _write_d8_candidate_run(
            d8_root,
            "o6",
            seed,
            1.05,
            endpoints={"man_oral_TDLo": 0.9, "women_oral_TDLo": 1.4, "human_oral_TDLo": 1.40},
        )
        _write_d8_candidate_run(d8_root, "a1", seed, 1.10)
        _write_d8_candidate_run(d8_root, "a2", seed, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    assert result["gates"]["o6"]["human_gate"] is False
    assert result["gates"]["o6"]["gate_pass"] is False


def test_d8_a_representation_gate_requires_low_feature_drift(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
        _write_d8_candidate_run(d8_root, "a1", seed, 1.05, feature_drift=0.05)  # > 0.02
        _write_d8_candidate_run(d8_root, "a2", seed, 1.10)
        _write_d8_candidate_run(d8_root, "o6", seed, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    assert result["gates"]["a1"]["representation_gate"] is False
    assert result["gates"]["a1"]["gate_pass"] is False


def test_d8_a_functional_gate_requires_low_forgetting(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
        _write_d8_candidate_run(
            d8_root, "a1", seed, 1.05, forgetting_relative=0.20
        )  # > 0.05
        _write_d8_candidate_run(d8_root, "a2", seed, 1.10)
        _write_d8_candidate_run(d8_root, "o6", seed, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    assert result["gates"]["a1"]["functional_gate"] is False
    assert result["gates"]["a1"]["gate_pass"] is False


# ----------------------------------------------------------------------
# D8-B: Top-1 5-seed confirmation (§75-§78)
# ----------------------------------------------------------------------
def test_d8_b_strong_pass_and_women_block(tmp_path):
    from scripts.d8_aggregate import d8_b

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        _write_d6_ref(d6_root, "b1", seed, 1.25)
        # S1 with women at 1.35: a candidate at 1.40 is blocked, at 1.30 passes.
        _write_d7_s1_run_with_endpoints(d7_root, seed, 1.09, women=1.35)
        _write_d8_candidate_run(
            d8_root, "o6", seed, 1.06, epochs=40, women=1.30
        )
    trace = {}
    result = d8_b(d8_root, d7_root, d6_root, "o6", D6_SEEDS, tmp_path, trace)
    assert result["gate_pass"] is True
    assert result["strength"] == "STRONG"
    assert result["women_non_worse"] is True


def test_d8_b_women_regression_blocks(tmp_path):
    from scripts.d8_aggregate import d8_b

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        _write_d6_ref(d6_root, "b1", seed, 1.25)
        _write_d7_s1_run_with_endpoints(d7_root, seed, 1.09, women=1.30)
        _write_d8_candidate_run(d8_root, "o6", seed, 1.06, epochs=40, women=1.40)
    trace = {}
    result = d8_b(d8_root, d7_root, d6_root, "o6", D6_SEEDS, tmp_path, trace)
    assert result["women_non_worse"] is False
    assert result["gate_pass"] is False


def _write_d7_s1_run_with_endpoints(root, seed, stable, women):
    run_dir = _write_d7_s1_run(root, seed, stable)
    path = run_dir / "diagnostics" / "human3_path_metrics.csv"
    lines = ["epoch,task,path,rmse\n"]
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        epoch, task, path_name, value = line.split(",")
        if task == "women_oral_TDLo":
            value = str(women)
        lines.append(f"{epoch},{task},{path_name},{value}\n")
    path.write_text("".join(lines), encoding="utf-8")
    return run_dir

# ----------------------------------------------------------------------
# Sixth-D8 review P0-2: endpoint gate on cross-seed MEANS (§9-§14)
# ----------------------------------------------------------------------
def test_d8_a_endpoint_gate_uses_endpoint_means_across_seeds(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    per_seed_endpoints = {
        42: {"man_oral_TDLo": 0.8, "women_oral_TDLo": 1.4, "human_oral_TDLo": 1.7},
        45: {"man_oral_TDLo": 1.4, "women_oral_TDLo": 1.4, "human_oral_TDLo": 1.0},
    }
    for seed in D8_A_SEEDS:
        # S1 reference at (1.0, 1.48, 1.30); candidate wins 2/3 WITHIN each
        # seed, but the cross-seed means only keep women non-worse.
        _write_s1_20e(d7_root, seed, 1.07)
        _write_d8_candidate_run(d8_root, "a1", seed, 1.05, endpoints=per_seed_endpoints[seed])
        _write_d8_candidate_run(d8_root, "a2", seed, 1.10)
        _write_d8_candidate_run(d8_root, "o6", seed, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    gate = result["gates"]["a1"]
    assert gate["endpoint_non_worse"] == 1  # only women survives the mean gate
    assert gate["endpoint_gate_pass"] is False
    assert gate["gate_pass"] is False
    assert gate["mean_gain_vs_s1"] == pytest.approx(0.02)  # macro alone would pass


# ----------------------------------------------------------------------
# Sixth-D8 review P0-3/P0-4: feature drift completeness (§17-§21)
# ----------------------------------------------------------------------
def test_d8_a_requires_feature_drift_for_both_seeds(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
    drift_by_seed = {42: 0.01, 45: 0.08}
    for seed, drift in drift_by_seed.items():
        _write_d8_candidate_run(d8_root, "a1", seed, 1.05, feature_drift=drift)
    _write_d8_candidate_run(d8_root, "a2", 42, 1.10)
    _write_d8_candidate_run(d8_root, "o6", 42, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    gate = result["gates"]["a1"]
    assert gate["representation_gate"] is False
    assert gate["complete_feature_drift"] is True
    assert gate["max_feature_drift"] == pytest.approx(0.08)
    assert gate["gate_pass"] is False


def test_d8_b_requires_feature_drift_for_all_five_seeds(tmp_path):
    from scripts.d8_aggregate import d8_b

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        _write_d6_ref(d6_root, "b1", seed, 1.25)
        _write_d7_s1_run_with_endpoints(d7_root, seed, 1.09, women=1.30)
        # seed 45 silently omits the drift file -> mechanism evidence 4/5.
        with_drift = seed != 45
        _write_d8_candidate_run(
            d8_root, "o6", seed, 1.06, epochs=40,
            women=1.25, feature_drift=0.01, with_mechanism=with_drift,
        )
    trace = {}
    result = d8_b(d8_root, d7_root, d6_root, "o6", D6_SEEDS, tmp_path, trace)
    assert result["complete_feature_drift"] is False
    assert result["mechanism_gate"] is False
    assert result["gate_pass"] is False


# ----------------------------------------------------------------------
# Sixth-D8 review P0-5: functional forgetting completeness (§22-§26)
# ----------------------------------------------------------------------
def test_d8_a_requires_functional_forgetting_for_both_seeds(tmp_path):
    from scripts.d8_aggregate import d8_a

    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for seed in D8_A_SEEDS:
        _write_s1_20e(d7_root, seed, 1.07)
        _write_d8_candidate_run(d8_root, "a1", seed, 1.05, with_mechanism=(seed == 42))
        _write_d8_candidate_run(d8_root, "a2", seed, 1.10)
        _write_d8_candidate_run(d8_root, "o6", seed, 1.10)
    trace = {}
    result = d8_a(d8_root, d7_root, D8_A_SEEDS, tmp_path, trace)
    gate = result["gates"]["a1"]
    assert gate["complete_forgetting"] is False
    assert gate["functional_gate"] is False
    assert gate["gate_pass"] is False


def test_d8_b_requires_functional_forgetting_for_all_five_seeds(tmp_path):
    from scripts.d8_aggregate import d8_b

    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    d8_root = tmp_path / "d8"
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        _write_d6_ref(d6_root, "b1", seed, 1.25)
        _write_d7_s1_run_with_endpoints(d7_root, seed, 1.09, women=1.30)
        _write_d8_candidate_run(
            d8_root, "o6", seed, 1.06, epochs=40, women=1.25,
            forgetting_relative=0.03, with_mechanism=(seed != 45),
        )
    trace = {}
    result = d8_b(d8_root, d7_root, d6_root, "o6", D6_SEEDS, tmp_path, trace)
    assert result["complete_forgetting"] is False
    assert result["mechanism_gate"] is False
    assert result["gate_pass"] is False


# ----------------------------------------------------------------------
# Sixth-D8 review P1-3: D8-0 audit completeness (§49-§54)
# ----------------------------------------------------------------------
def _write_functional_forgetting_json(root, reference, seed, *, exact_zero, verified=True):
    if reference == "b1":
        run_dir = root / "d6_stage_b" / "b1" / "d6_b1_e40" / f"seed_{seed}"
    else:
        run_dir = root / "d7_stage_b" / "s1" / "d7_s1_e40" / f"seed_{seed}"
    payload = {
        "functional_forgetting_relative": 0.0 if exact_zero else 0.20,
        "functional_forgetting_abs": 0.0 if exact_zero else 0.25,
        "functional_forgetting_exact_zero": exact_zero,
        "provenance_verified": verified,
        "per_task_delta": {f"t{i}": 0.0 if exact_zero else 0.01 for i in range(56)},
        "animal_tasks_worsened": 0 if exact_zero else 56,
        "animal_tasks_improved": 0,
        "animal_tasks_unchanged": 56 if exact_zero else 0,
        "delta_q25": 0.0, "delta_median": 0.0, "delta_q75": 0.0,
        "delta_q90": 0.0, "delta_max": 0.0 if exact_zero else 0.05,
    }
    (run_dir / "functional_forgetting.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _d8_0_world(tmp_path):
    d6_root = tmp_path / "d6"
    d7_root = tmp_path / "d7"
    s1_endpoints = {"man_oral_TDLo": 0.95, "women_oral_TDLo": 1.35, "human_oral_TDLo": 1.28}
    for index, seed in enumerate(D6_SEEDS):
        _write_d6_ref(d6_root, "b0", seed, 1.40)
        _write_d6_ref(d6_root, "b1", seed, 1.25 + 0.005 * index)
        _write_d7_s1_run(d7_root, seed, 1.09, endpoints=s1_endpoints)
    return d6_root, d7_root


def test_d8_0_requires_complete_s1_b1_functional_audit(tmp_path):
    from scripts.d8_aggregate import d8_0

    d6_root, d7_root = _d8_0_world(tmp_path)
    # B1 audit present for only 4/5 seeds, S1 complete and exact-zero.
    for index, seed in enumerate(D6_SEEDS):
        _write_functional_forgetting_json(d6_root, "b1", seed, exact_zero=False)
        if seed != 46:
            _write_functional_forgetting_json(d7_root, "s1", seed, exact_zero=True)
    trace = {}
    result = d8_0(d6_root, d7_root, D6_SEEDS, tmp_path, trace)
    assert result["performance_gate_pass"] is True
    assert result["functional_audit_complete"] is False
    assert result["gate_pass"] is False
    assert "functional forgetting audit incomplete" in result["stop_reason"]


def test_d8_0_requires_s1_exact_zero_functional_forgetting(tmp_path):
    from scripts.d8_aggregate import d8_0

    d6_root, d7_root = _d8_0_world(tmp_path)
    for index, seed in enumerate(D6_SEEDS):
        _write_functional_forgetting_json(d6_root, "b1", seed, exact_zero=False)
        # S1 audit complete, but one seed violates the exact-zero invariant.
        _write_functional_forgetting_json(
            d7_root, "s1", seed, exact_zero=(seed != 45)
        )
    trace = {}
    result = d8_0(d6_root, d7_root, D6_SEEDS, tmp_path, trace)
    assert result["functional_invariant_failure"] is True
    assert result["functional_audit_complete"] is False
    assert result["gate_pass"] is False
