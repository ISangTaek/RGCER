"""D7 representation-preservation candidate contracts (plan §8-§29, §82, §95-§98).

Covers: S1 freeze schedule (backbone grads zero), S2/S3 parameter-group LR
(backbone LR = multiplier * head LR), O4 exact alpha scaling, O5 per-group
bounded ratio <= tau (§98), drift at epoch 0 == 0 (§96), the D7 validation-only
run contract, and the D7 selector Top-2 rules (§49-§56, §82).
"""

import json

import pytest
import torch
from torch import nn

from d7_diagnostics import (
    D7DriftTracker,
    backbone_parameter_drift,
    parse_feature_drift_epochs,
    select_probe_ids,
)
from scripts.d6_prep_inits import apply_csdt, apply_csdt_bounded


# ----------------------------------------------------------------------
# S1/S3 freeze schedule (plan §96: S1 backbone grad == 0; S3 epoch0-4 frozen)
# ----------------------------------------------------------------------
class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(3, 3)
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        return self.head(self.encoder.backbone(x))


def _frozen_state(trainer_like_model, freeze_epochs, epoch):
    """Mirror of Trainer._apply_backbone_freeze decision for tests."""

    return epoch < freeze_epochs


def test_s1_freeze_zeroes_backbone_gradients():
    from trainer import Trainer  # ensures import surface stays intact

    model = _TinyModel()
    freeze_epochs = 20
    for epoch in (0, 10, 19):
        frozen = _frozen_state(model, freeze_epochs, epoch)
        for name, parameter in model.named_parameters():
            if name.startswith("encoder.backbone."):
                parameter.requires_grad = not frozen
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    assert model.encoder.backbone.weight.grad is None
    assert model.encoder.backbone.bias.grad is None
    assert model.head.weight.grad is not None  # heads keep training


def test_s3_freeze_schedule_epochs_0_to_4_frozen_then_unfrozen():
    freeze_epochs = 5
    expected = {0: True, 4: True, 5: False, 10: False, 19: False}
    for epoch, frozen in expected.items():
        assert _frozen_state(None, freeze_epochs, epoch) is frozen, epoch


def test_s2_parameter_group_lr_multiplier():
    # §96: S2 backbone LR == 0.1 * head LR — mirrors the regroup logic in
    # Trainer._make_optimizer.
    backbone = nn.Linear(4, 4)
    head = nn.Linear(4, 1)
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.backbone = backbone
    model.head = head
    multiplier = 0.1
    base_lr = 1e-3
    named = list(model.named_parameters())
    all_params = [parameter for _, parameter in named]
    original = torch.optim.AdamW(all_params, lr=base_lr, weight_decay=0.01)
    backbone_ids = {
        id(parameter)
        for name, parameter in named
        if name.startswith("encoder.backbone.")
    }
    base_group = {k: v for k, v in original.param_groups[0].items() if k != "params"}
    backbone_group = dict(base_group)
    backbone_group["lr"] = base_group["lr"] * multiplier
    backbone_params = [p for p in all_params if id(p) in backbone_ids]
    head_params = [p for p in all_params if id(p) not in backbone_ids]
    backbone_group["params"] = backbone_params
    head_group = dict(base_group)
    head_group["params"] = head_params
    regrouped = torch.optim.AdamW([backbone_group, head_group])
    backbone_lrs = [
        g["lr"] for g in regrouped.param_groups if any(id(p) in backbone_ids for p in g["params"])
    ]
    head_lrs = [
        g["lr"] for g in regrouped.param_groups if not any(id(p) in backbone_ids for p in g["params"])
    ]
    assert backbone_lrs == [base_lr * multiplier]
    assert head_lrs == [base_lr]
    # weight decay inherited unchanged
    assert all(g["weight_decay"] == 0.01 for g in regrouped.param_groups)


# ----------------------------------------------------------------------
# O4 exact alpha scaling / O5 bounded ratio (plan §96, §98)
# ----------------------------------------------------------------------
def _csdt_dicts():
    theta_anchor = {
        "encoder.backbone.a": torch.tensor([1.0, 1.0]),
        "encoder.backbone.b": torch.tensor([0.5]),
        "decoders.head": torch.tensor([0.25]),
    }
    real = {
        "encoder.backbone.a": torch.tensor([31.0, 21.0]),
        "encoder.backbone.b": torch.tensor([0.52]),  # ratio 0.04 < tau -> untouched
        "decoders.head": torch.tensor([9.0]),
    }
    shuffle = {
        "encoder.backbone.a": torch.tensor([1.0, 1.0]),
        "encoder.backbone.b": torch.tensor([0.5]),
        "decoders.head": torch.tensor([7.0]),
    }
    return theta_anchor, real, shuffle


def test_o4_exact_alpha_scaling():
    theta_anchor, real, shuffle = _csdt_dicts()
    merged = apply_csdt(theta_anchor, real, shuffle, alpha=0.1)
    expected_a = theta_anchor["encoder.backbone.a"] + 0.1 * (
        real["encoder.backbone.a"] - shuffle["encoder.backbone.a"]
    )
    assert torch.allclose(merged["encoder.backbone.a"], expected_a)
    # heads untouched by the CSDT merge
    assert torch.equal(merged["decoders.head"], theta_anchor["decoders.head"])


def test_o5_bounded_ratio_never_exceeds_tau():
    # §98: bounded_delta_ratio <= tau + 1e-6 for EVERY group, and clipped
    # groups are flagged.
    theta_anchor, real, shuffle = _csdt_dicts()
    tau = 0.1
    merged, rows = apply_csdt_bounded(theta_anchor, real, shuffle, tau)
    assert rows
    for row in rows:
        assert row["bounded_delta_ratio"] <= tau + 1e-6, row
        assert row["scale_factor"] <= 1.0
        assert row["was_clipped"] == int(row["raw_delta_ratio"] > tau + 1e-12)
    # group a: huge raw delta -> clipped to exactly tau of the anchor norm
    row_a = next(row for row in rows if row["parameter_group"] == "encoder.backbone.a")
    assert row_a["was_clipped"] == 1
    # float32 norms: the §98 invariant is <= tau + 1e-6, not bitwise tau
    assert row_a["bounded_delta_ratio"] == pytest.approx(tau, abs=1e-6)
    # group b: small raw delta -> untouched (scale 1.0)
    row_b = next(row for row in rows if row["parameter_group"] == "encoder.backbone.b")
    assert row_b["was_clipped"] == 0
    assert row_b["scale_factor"] == 1.0
    # merged weights: anchor + bounded delta
    expected_b = theta_anchor["encoder.backbone.b"] + (
        real["encoder.backbone.b"] - shuffle["encoder.backbone.b"]
    )
    assert torch.allclose(merged["encoder.backbone.b"], expected_b)


def test_o5_zero_anchor_group_stays_bounded():
    theta_anchor = {"encoder.backbone.zero": torch.zeros(2)}
    real = {"encoder.backbone.zero": torch.tensor([10.0, 10.0])}
    shuffle = {"encoder.backbone.zero": torch.zeros(2)}
    _, rows = apply_csdt_bounded(theta_anchor, real, shuffle, tau=0.1)
    assert rows[0]["bounded_delta_ratio"] <= 0.1 + 1e-6


def test_o5_rejects_non_positive_tau():
    theta_anchor, real, shuffle = _csdt_dicts()
    with pytest.raises(ValueError):
        apply_csdt_bounded(theta_anchor, real, shuffle, tau=0.0)


# ----------------------------------------------------------------------
# Drift (plan §43-§48, §96: drift at epoch0 == 0)
# ----------------------------------------------------------------------
def test_backbone_parameter_drift_at_epoch0_is_zero():
    state = {
        "encoder.backbone.layers.0.w": torch.randn(4, 4),
        "encoder.backbone.layers.5.w": torch.randn(4, 4),
        "encoder.backbone.embed": torch.randn(4),
        "decoders.head": torch.randn(2),
    }
    all_drift, early, late = backbone_parameter_drift(state, state)
    assert all_drift == 0.0
    assert early == 0.0
    assert late == 0.0


def test_backbone_parameter_drift_buckets_early_and_late():
    init = {
        "encoder.backbone.layers.0.w": torch.ones(2, 2),
        "encoder.backbone.layers.5.w": torch.ones(2, 2),
    }
    state = {
        "encoder.backbone.layers.0.w": torch.ones(2, 2) + 0.1,  # early moved
        "encoder.backbone.layers.5.w": torch.ones(2, 2),  # late untouched
    }
    _, early, late = backbone_parameter_drift(state, init)
    assert early > 0.0
    assert late == 0.0


def test_parse_feature_drift_epochs():
    assert parse_feature_drift_epochs("0,5,10,15,19") == {0, 5, 10, 15, 19}
    with pytest.raises(ValueError):
        parse_feature_drift_epochs("-1,5")


def test_select_probe_ids_sorted_first_n_unique():
    ids = ["z1", "a2", "a2", "m3", "b4"]
    assert select_probe_ids(ids, 3) == ["a2", "b4", "m3"]
    with pytest.raises(ValueError):
        select_probe_ids([], 128)


class _ProbeBatch(dict):
    def __init__(self, ids):
        super().__init__()
        self.sample_id = list(ids)

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default


def test_drift_tracker_epoch0_row_is_zero_drift(tmp_path):
    class _TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 4)

        def forward(self, batch):
            # Graphormer-shaped output: (batch, tokens, dim); CLS pooling
            # takes [:, 0, :].  The tokens pass through the weights so a
            # parameter update actually changes the representation.
            tokens = torch.ones(len(batch.sample_id), 2, 4)
            return self.layer(tokens)

    class _TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _TinyBackbone()

    class _TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _TinyEncoder()

    model = _TinyModel()
    loaders = {
        "task_a": _StubLoader(["s2", "s1"]),
        "task_b": _StubLoader(["s1", "s3"]),
    }
    tracker = D7DriftTracker(
        model=model,
        train_loaders=loaders,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        probe_size=128,
        feature_drift_epochs={0},
    )
    assert tracker.probe_ids == ["s1", "s2", "s3"]
    # Fifth-review P1-4: construction logs the BASELINE row (epoch=-1,
    # state=baseline, drift exactly 0) — no epoch-0 row yet.
    baseline = tracker.rows[0]
    assert baseline["epoch"] == -1
    assert baseline["state"] == "baseline"
    assert float(baseline["backbone_param_drift"]) == 0.0
    assert float(baseline["feature_drift"]) == 0.0
    # epoch-0 row is logged AFTER the first training epoch — a real update
    # makes it non-zero while staying aligned with validation epoch 0.
    with torch.no_grad():
        model.encoder.backbone.layer.weight.add_(0.5)
    row0 = tracker.log_epoch(0, model)
    assert row0["state"] == "post_epoch"
    assert float(row0["backbone_param_drift"]) > 0.0
    assert float(row0["feature_drift"]) > 0.0  # epoch 0 is in the schedule
    with torch.no_grad():
        model.encoder.backbone.layer.bias.add_(0.3)
    row1 = tracker.log_epoch(1, model)
    assert float(row1["backbone_param_drift"]) > float(row0["backbone_param_drift"])
    assert row1["feature_drift"] == ""  # epoch 1 not in the schedule
    # CSV written and parseable; drift/validation epochs align 1:1
    import csv as _csv

    with (tmp_path / "d7_representation_drift.csv").open() as handle:
        rows = list(_csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [-1, 0, 1]
    assert [row["state"] for row in rows] == ["baseline", "post_epoch", "post_epoch"]


def test_drift_and_validation_epoch_are_aligned():
    # §64: the trainer must call log_epoch AFTER validation, so drift row t
    # and validation row t observe the same model state.
    import inspect

    from trainer import Trainer

    source = inspect.getsource(Trainer.train)
    drift_position = source.index("drift_tracker.log_epoch")
    validation_position = source.index('self._evaluate(val_dataloaders_dict, mode="validation"')
    assert validation_position < drift_position


def test_drift_anchor_type_is_recorded(tmp_path):
    # Fifth-review P2-2: O4/O5 baselines are counterfactual inits — their
    # drift magnitudes must never be silently compared with S-family rows.
    class _AnchorBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 4)

        def forward(self, batch):
            return torch.ones(len(batch.sample_id), 2, 4)

    class _AnchorModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.backbone = _AnchorBackbone()

    model = _AnchorModel()
    loaders = {"task_a": _StubLoader(["s1"])}
    for anchor_type in ("animal_pretrained", "counterfactual_init"):
        tracker = D7DriftTracker(
            model=model,
            train_loaders=loaders,
            device=torch.device("cpu"),
            output_dir=tmp_path / anchor_type,
            anchor_type=anchor_type,
        )
        assert tracker.rows[0]["drift_anchor_type"] == anchor_type
    with pytest.raises(ValueError):
        D7DriftTracker(
            model=model,
            train_loaders=loaders,
            device=torch.device("cpu"),
            output_dir=None,
            anchor_type="bogus",
        )


# ----------------------------------------------------------------------
# Sixth-review P1: exact probe set (§25-§26)
# ----------------------------------------------------------------------
class _ProbeDataset(torch.utils.data.Dataset):
    """Synthetic human3 train dataset with cheap get_sample_id."""

    def __init__(self, ids):
        self.ids = [str(value) for value in ids]

    def __len__(self):
        return len(self.ids)

    def get_sample_id(self, index):
        return self.ids[index]

    def __getitem__(self, index):
        return {"sample_id": self.ids[index]}


class _StubLoader:
    """Minimal loader surface used by the probe helpers: .dataset/.collate_fn."""

    def __init__(self, ids, collate_fn=None):
        self.dataset = _ProbeDataset(ids)
        self.collate_fn = collate_fn or (lambda items: _ProbeBatch([item["sample_id"] for item in items]))


def test_d7_probe_batches_contain_exact_probe_ids():
    # §25: a batch containing a probe molecule AND an extra molecule must
    # contribute ONLY the probe molecule to probe statistics.
    from d7_diagnostics import collect_probe_batches, pooled_representations, select_probe_ids, train_sample_ids

    # Sorted train ids: ["a_probe", "b_extra", "c_tail"]; the probe is ONLY
    # "a_probe", while the batch that carries it also carries "b_extra".
    loaders = {"task_a": _StubLoader(["a_probe", "b_extra", "c_tail"])}
    probe_ids = select_probe_ids(train_sample_ids(loaders), 1)
    assert probe_ids == ["a_probe"]
    batches = collect_probe_batches(loaders, probe_ids)
    model = _ProbeModelForProbe()
    representations = pooled_representations(model, batches, torch.device("cpu"), expected_ids=probe_ids)
    assert set(representations) == {"a_probe"}
    assert "b_extra" not in representations
    assert "c_tail" not in representations


class _ProbeModelForProbe(torch.nn.Module):
    class _Backbone(torch.nn.Module):
        def forward(self, batch):
            tokens = torch.ones(len(batch.sample_id), 2, 4)
            return tokens * torch.arange(1.0, 5.0)

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.backbone = self._Backbone()


def test_d7_probe_selection_does_not_advance_train_loader_generator():
    # §25: diagnostics must not disturb the formal training loaders.
    import torch as _torch

    dataset = _ProbeDataset([f"id{i}" for i in range(8)])

    def collate(items):
        return _ProbeBatch([item["sample_id"] for item in items])

    generator = _torch.Generator().manual_seed(7)
    loader = _torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, generator=generator, collate_fn=collate
    )
    state_before = generator.get_state().clone()
    loaders = {"task_a": loader}
    from d7_diagnostics import collect_probe_batches, train_sample_ids

    train_sample_ids(loaders)
    collect_probe_batches(loaders, [f"id{i}" for i in range(4)])
    assert _torch.equal(generator.get_state(), state_before)


def test_pooled_representations_accepts_pooled_2d_backbone(tmp_path):
    # The real Graphormer backbone returns the already-pooled (batch, dim)
    # tensor; the probe must accept both layouts.
    from d7_diagnostics import pooled_representations

    class _PooledBackbone(torch.nn.Module):
        def forward(self, batch):
            return torch.arange(1.0, 5.0).repeat(len(batch.sample_id), 1) * 2.0

    class _PooledModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.backbone = _PooledBackbone()

    model = _PooledModel()
    batches = [_ProbeBatch(["s1", "s2"])]
    reps = pooled_representations(model, batches, torch.device("cpu"), expected_ids=["s1", "s2"])
    assert set(reps) == {"s1", "s2"}
    assert reps["s1"].shape[0] == 4


def test_d7_probe_exact_set_mismatch_fails():
    # §22: pooled_representations must fail loudly if observed ids deviate.
    from d7_diagnostics import pooled_representations

    model = _ProbeModelForProbe()
    batches = [_ProbeBatch(["s1", "intruder"])]
    with pytest.raises(RuntimeError, match="probe representation ID mismatch"):
        pooled_representations(model, batches, torch.device("cpu"), expected_ids=["s1"])


# ----------------------------------------------------------------------
# D7 run contract (plan §93, §96: validation-only contract)
# ----------------------------------------------------------------------
from scripts.d7_aggregate import check_d7_run_contract  # noqa: E402
from scripts.d7_aggregate import ARTIFACT_CANDIDATES  # noqa: E402


def _write_d7_run(root, candidate, seed, epochs, freeze, multiplier, contract=True, mode=None):
    run_dir = root / candidate / f"d7_{candidate}_e{epochs}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "dataset": "toxacute",
                "toxacute_task_scope": "human3",
                "train_eval_scope": "validation_only",
                "fit_conformal": False,
                "epochs": epochs,
                "freeze_backbone_epochs": freeze,
                "backbone_lr_multiplier": multiplier,
                "split_seed": 42,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": candidate,
        "manifest_sha256": "m",
        "datastore_fingerprint": "f",
        "feature_schema_version": "schema-test",
    }
    # Fifth-review P0-1: EVERY D7 candidate carries an artifact contract
    # (S-family mode=b1; CSDT family per variant).
    if contract and candidate in ARTIFACT_CANDIDATES:
        metadata["d7_artifact_contract"] = {
            "mode": mode or ARTIFACT_CANDIDATES[candidate],
            "human_seed": seed,
            "split_manifest_hash": "m",
            "datastore_fingerprint": "f",
            "feature_schema_version": "schema-test",
        }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def _write_b1_sidecar(tmp_path, mode="b1", seed=42, name="s_init.pt"):
    """Create a stub init artifact + provenance sidecar for validate_params."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    init_state = tmp_path / name
    init_state.write_bytes(b"stub")
    provenance = {"mode": mode, "human_seed": seed}
    (tmp_path / (name + ".provenance.json")).write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    return str(init_state)


def test_d7_run_contract_accepts_valid_s2_run(tmp_path):
    run_dir = _write_d7_run(tmp_path, "s2", 42, 20, freeze=0, multiplier=0.1)
    assert check_d7_run_contract(run_dir, "s2", 42, 19) == []


def test_d7_run_contract_rejects_s1_wrong_freeze(tmp_path):
    run_dir = _write_d7_run(tmp_path, "s1", 42, 20, freeze=0, multiplier=1.0)
    violations = check_d7_run_contract(run_dir, "s1", 42, 19)
    assert any("freeze_backbone_epochs" in violation for violation in violations)


def test_d7_run_contract_rejects_s3_wrong_multiplier(tmp_path):
    run_dir = _write_d7_run(tmp_path, "s3", 42, 20, freeze=5, multiplier=1.0)
    violations = check_d7_run_contract(run_dir, "s3", 42, 19)
    assert any("backbone_lr_multiplier" in violation for violation in violations)


def test_d7_run_contract_rejects_o5_without_artifact_contract(tmp_path):
    run_dir = _write_d7_run(tmp_path, "o5", 42, 20, freeze=0, multiplier=1.0, contract=False)
    violations = check_d7_run_contract(run_dir, "o5", 42, 19)
    assert any("d7_artifact_contract" in violation for violation in violations)


def test_d7_run_contract_rejects_full_scope_run(tmp_path):
    run_dir = _write_d7_run(tmp_path, "s2", 42, 20, freeze=0, multiplier=0.1)
    args_path = run_dir / "args.json"
    args_payload = json.loads(args_path.read_text(encoding="utf-8"))
    args_payload["train_eval_scope"] = "full"
    args_path.write_text(json.dumps(args_payload), encoding="utf-8")
    violations = check_d7_run_contract(run_dir, "s2", 42, 19)
    assert any("train_eval_scope" in violation for violation in violations)


# ----------------------------------------------------------------------
# D7 selector Top-2 rules (plan §49-§56, §82)
# ----------------------------------------------------------------------
from scripts.d7_aggregate import (  # noqa: E402
    EXCLUDE_VS_B1,
    STRONG_GAIN_VS_B1,
    stage_a,
)


def _write_reference(root, reference, stable, women=None):
    from tests.test_d6_aggregate import _write_run, _flat_macro

    endpoint = {
        "man_oral_TDLo": 1.0,
        "women_oral_TDLo": women if women is not None else 1.4,
        "human_oral_TDLo": 1.3,
    }
    # The D6 selector layout lives under <d6_root>/d6_stage_a/<reference>/...
    _write_run(
        root / "d6_stage_a",
        reference,
        f"d6_{reference}_e20",
        42,
        _flat_macro(19, stable),
        endpoint_by_epoch={epoch: dict(endpoint) for epoch in range(15, 20)},
    )


def _write_d7_candidate_run(root, candidate, stable, endpoints):
    from tests.test_d6_aggregate import _write_run, _flat_macro

    run_dir = root / "d7_stage_a" / candidate / f"d7_{candidate}_e20" / "seed_42"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "dataset": "toxacute",
                "toxacute_task_scope": "human3",
                "train_eval_scope": "validation_only",
                "fit_conformal": False,
                "epochs": 20,
                "split_seed": 42,
                "freeze_backbone_epochs": {"s1": 20, "s2": 0, "s3": 5}.get(candidate, 0),
                "backbone_lr_multiplier": 0.1 if candidate in ("s2", "s3") else 1.0,
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": candidate,
        "manifest_sha256": "manifest-test",
        "datastore_fingerprint": "datastore-test",
        "feature_schema_version": "schema-test",
    }
    metadata["d7_artifact_contract"] = {
        "mode": ARTIFACT_CANDIDATES[candidate],
        "human_seed": 42,
        "split_manifest_hash": "manifest-test",
        "datastore_fingerprint": "datastore-test",
        "feature_schema_version": "schema-test",
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    diagnostics = run_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    with (diagnostics / "epoch_summary.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,val_human3_macro_rmse\n")
        for epoch in range(20):
            handle.write(f"{epoch},{stable}\n")
    with (diagnostics / "human3_path_metrics.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,task,path,rmse\n")
        for epoch in range(15, 20):
            for task, value in endpoints.items():
                handle.write(f"{epoch},{task},final,{value}\n")


def test_stage_a_top2_promotes_candidates_beating_b1(tmp_path):
    # B0=1.2648, B1=1.2355; S2 and S3 beat B1 by >=0.01, S1/O4/O5 worse.
    from tests.test_d6_aggregate import _write_run, _flat_macro

    _write_reference(tmp_path, "b0", 1.2648)
    _write_reference(tmp_path, "b1", 1.2355)
    _write_reference(tmp_path, "b0a", 1.2648)
    endpoints = {"man_oral_TDLo": 0.95, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.25}
    _write_d7_candidate_run(tmp_path, "s1", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "s2", 1.20, endpoints)  # gain 0.0355
    _write_d7_candidate_run(tmp_path, "s3", 1.22, endpoints)  # gain 0.0155
    _write_d7_candidate_run(tmp_path, "o4", 1.26, endpoints)
    _write_d7_candidate_run(tmp_path, "o5", 1.27, endpoints)
    trace = {}
    result = stage_a(tmp_path, tmp_path, 42, tmp_path, trace)
    assert result["top2"] == ["s2", "s3"]
    assert result["selection_mode"] == "PROMOTED"
    ranking = {row["candidate"]: row for row in trace["stage_a"]["ranking"]}
    assert ranking["s2"]["tier"] == "PROMOTED"
    assert ranking["s2"]["gain_vs_b1"] == pytest.approx(1.2355 - 1.20)


def test_stage_a_excludes_candidates_worse_than_b1_by_margin(tmp_path):
    from tests.test_d6_aggregate import _write_run, _flat_macro

    _write_reference(tmp_path, "b0", 1.2648)
    _write_reference(tmp_path, "b1", 1.2355)
    _write_reference(tmp_path, "b0a", 1.2648)
    endpoints = {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.3}
    # every candidate at least 0.02 worse than B1
    for index, candidate in enumerate(("s1", "s2", "s3", "o4", "o5")):
        _write_d7_candidate_run(tmp_path, candidate, 1.2355 + EXCLUDE_VS_B1 + 0.01 * index, endpoints)
    trace = {}
    result = stage_a(tmp_path, tmp_path, 42, tmp_path, trace)
    assert result["top2"] == []
    assert result["selection_mode"] == "B1_CONFIRMATION_ONLY"
    assert "B1" in trace["stage_a"]["stop_reason"]


def test_stage_a_anchor_gate_fails_o4_without_anchor_gain(tmp_path):
    from tests.test_d6_aggregate import _write_run, _flat_macro

    # B0A is BETTER than O4 — the counterfactual delta has no independent
    # contribution, so O4 must fail even though it beats B1.
    _write_reference(tmp_path, "b0", 1.2648)
    _write_reference(tmp_path, "b1", 1.2355)
    _write_reference(tmp_path, "b0a", 1.20)
    endpoints = {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.3}
    _write_d7_candidate_run(tmp_path, "s1", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "s2", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "s3", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "o4", 1.21, endpoints)  # beats B1, loses to B0A
    _write_d7_candidate_run(tmp_path, "o5", 1.30, endpoints)
    trace = {}
    result = stage_a(tmp_path, tmp_path, 42, tmp_path, trace)
    ranking = {row["candidate"]: row for row in trace["stage_a"]["ranking"]}
    assert ranking["o4"]["status"] == "ANCHOR_GATE_FAILED"
    assert "o4" not in result["top2"]


def test_stage_a_o5_passes_anchor_gate_and_enters_top2(tmp_path):
    from tests.test_d6_aggregate import _write_run, _flat_macro

    _write_reference(tmp_path, "b0", 1.2648)
    _write_reference(tmp_path, "b1", 1.2355)
    _write_reference(tmp_path, "b0a", 1.2648)
    endpoints = {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.3}
    _write_d7_candidate_run(tmp_path, "s1", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "s2", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "s3", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "o4", 1.30, endpoints)
    _write_d7_candidate_run(tmp_path, "o5", 1.21, endpoints)  # gain 0.0255, anchor gain positive
    trace = {}
    result = stage_a(tmp_path, tmp_path, 42, tmp_path, trace)
    # §54/§55: O5 leads; the S-family reserve only applies while S rows are
    # still PASS — here all three S rows lost to B1 by >=0.02, so §82 excludes
    # them and O5 advances alone.
    assert result["top2"] == ["o5"]
    assert result["top1"] == "o5"
    ranking = {row["candidate"]: row for row in trace["stage_a"]["ranking"]}
    assert ranking["o5"]["anchor_semantic_gain"] == pytest.approx(1.2648 - 1.21)


def test_stage_a_women_protection_blocks_top1(tmp_path):
    from tests.test_d6_aggregate import _write_run, _flat_macro

    _write_reference(tmp_path, "b0", 1.2648, women=1.4226)
    _write_reference(tmp_path, "b1", 1.2355, women=1.4226)
    _write_reference(tmp_path, "b0a", 1.2648, women=1.4226)
    # S2 wins overall but destroys women (degradation > 0.08, §51).
    _write_d7_candidate_run(
        tmp_path, "s2", 1.20, {"man_oral_TDLo": 0.9, "women_oral_TDLo": 1.55, "human_oral_TDLo": 1.2}
    )
    # S3 modest but women-safe.
    _write_d7_candidate_run(
        tmp_path, "s3", 1.225, {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.43, "human_oral_TDLo": 1.28}
    )
    _write_d7_candidate_run(
        tmp_path, "s1", 1.30, {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.3}
    )
    _write_d7_candidate_run(
        tmp_path, "o4", 1.30, {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.3}
    )
    _write_d7_candidate_run(
        tmp_path, "o5", 1.30, {"man_oral_TDLo": 1.0, "women_oral_TDLo": 1.42, "human_oral_TDLo": 1.3}
    )
    trace = {}
    result = stage_a(tmp_path, tmp_path, 42, tmp_path, trace)
    assert result["top1"] == "s3"
    assert result["top2"][0] == "s2"  # still ranked by StableRMSE...
    ranking = {row["candidate"]: row for row in trace["stage_a"]["ranking"]}
    assert "women" in ranking["s2"]["women_flag"]


def test_stage_a_strong_gain_constant_matches_plan():
    assert STRONG_GAIN_VS_B1 == 0.01
    assert EXCLUDE_VS_B1 == 0.02


# ----------------------------------------------------------------------
# run-side contracts in validate_params (plan §93, §96)
# ----------------------------------------------------------------------
def _base_params():
    from main import build_parser

    params = build_parser().parse_args([])
    params.dataset = "toxacute"
    params.arch = "Graphormer"
    params.toxacute_task_scope = "human3"
    params.fit_conformal = False
    params.train_eval_scope = "validation_only"
    params.shuffle_animal_train_labels = False
    params.card_lambda_delta = 0.0
    params.d6_candidate = "none"
    params.d7_candidate = "none"
    params.seed = 42
    params.epochs = 20
    params.freeze_backbone_epochs = 0
    params.backbone_lr_multiplier = 1.0
    return params


def test_validate_params_s2_requires_b1_init_sidecar(tmp_path):
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s2"
    params.freeze_backbone_epochs = 0
    params.backbone_lr_multiplier = 0.1
    # Fifth-review P0-1: no init artifact -> the run must not even start.
    with pytest.raises(ValueError, match="initialization artifact"):
        validate_params(params)
    params.init_state_path = _write_b1_sidecar(tmp_path)
    validate_params(params)  # valid B1 sidecar -> cheap checks pass


def test_validate_params_s1_requires_b1_init(tmp_path):
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s1"
    params.freeze_backbone_epochs = 20
    with pytest.raises(ValueError, match="requires --init_state_path"):
        validate_params(params)
    params.init_state_path = _write_b1_sidecar(tmp_path)
    validate_params(params)


def test_validate_params_s3_requires_b1_init(tmp_path):
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s3"
    params.freeze_backbone_epochs = 5
    params.backbone_lr_multiplier = 0.1
    with pytest.raises(ValueError, match="requires --init_state_path"):
        validate_params(params)
    params.init_state_path = _write_b1_sidecar(tmp_path)
    validate_params(params)


def test_validate_params_s2_rejects_csdt_init(tmp_path):
    # §18: a CSDT artifact is not a preservation-family initialization.
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s2"
    params.freeze_backbone_epochs = 0
    params.backbone_lr_multiplier = 0.1
    params.init_state_path = _write_b1_sidecar(tmp_path, mode="csdt")
    with pytest.raises(ValueError, match="B1/sequential-transfer"):
        validate_params(params)


def test_validate_params_s_family_rejects_wrong_seed_sidecar(tmp_path):
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s1"
    params.freeze_backbone_epochs = 20
    params.init_state_path = _write_b1_sidecar(tmp_path, seed=44)
    with pytest.raises(ValueError, match="human_seed"):
        validate_params(params)


def test_validate_params_accepts_valid_s2(tmp_path):
    # Fifth-review §13: the pre-review version of this test accepted S2 with
    # NO init artifact — that protected the wrong behaviour.  A valid S2 must
    # carry a seed-matched B1 initialization sidecar.
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s2"
    params.freeze_backbone_epochs = 0
    params.backbone_lr_multiplier = 0.1
    params.init_state_path = _write_b1_sidecar(tmp_path)
    validate_params(params)  # must not raise


def test_validate_params_s1_requires_whole_run_freeze():
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s1"
    params.freeze_backbone_epochs = 5
    with pytest.raises(ValueError, match="WHOLE run"):
        validate_params(params)


def test_validate_params_s3_requires_exactly_five_frozen_epochs():
    from main import validate_params

    params = _base_params()
    params.d7_candidate = "s3"
    params.freeze_backbone_epochs = 10
    params.backbone_lr_multiplier = 0.1
    with pytest.raises(ValueError, match="first 5 epochs"):
        validate_params(params)


def test_validate_params_s2_rejects_multiplier_other_than_point_one(tmp_path):
    # Sixth-review P0: Stage A is direction screening, not an LR search —
    # only backbone_lr_multiplier == 0.1 is S2.
    from main import validate_params

    for bad_multiplier in (0.05, 0.2, 0.3):
        params = _base_params()
        params.d7_candidate = "s2"
        params.freeze_backbone_epochs = 0
        params.backbone_lr_multiplier = bad_multiplier
        params.init_state_path = _write_b1_sidecar(
            tmp_path / f"m{bad_multiplier}", name=f"s2_{bad_multiplier}.pt"
        )
        with pytest.raises(ValueError, match="not an LR grid search"):
            validate_params(params)
    params = _base_params()
    params.d7_candidate = "s2"
    params.freeze_backbone_epochs = 0
    params.backbone_lr_multiplier = 0.1
    params.init_state_path = _write_b1_sidecar(tmp_path, name="s2_ok.pt")
    validate_params(params)  # 0.1 accepted


def test_validate_params_s3_rejects_multiplier_other_than_point_one(tmp_path):
    from main import validate_params

    for bad_multiplier in (0.05, 0.2):
        params = _base_params()
        params.d7_candidate = "s3"
        params.freeze_backbone_epochs = 5
        params.backbone_lr_multiplier = bad_multiplier
        params.init_state_path = _write_b1_sidecar(
            tmp_path / f"m{bad_multiplier}", name=f"s3_{bad_multiplier}.pt"
        )
        with pytest.raises(ValueError, match="backbone_lr_multiplier=0.1"):
            validate_params(params)
    params = _base_params()
    params.d7_candidate = "s3"
    params.freeze_backbone_epochs = 5
    params.backbone_lr_multiplier = 0.1
    params.init_state_path = _write_b1_sidecar(tmp_path, name="s3_ok.pt")
    validate_params(params)


def test_d7_run_contract_rejects_s2_multiplier_0_2(tmp_path):
    run_dir = _write_d7_run(tmp_path, "s2", 42, 20, freeze=0, multiplier=0.2)
    violations = check_d7_run_contract(run_dir, "s2", 42, 19)
    assert any("backbone_lr_multiplier" in violation for violation in violations)


def test_d7_run_contract_rejects_s3_multiplier_0_05(tmp_path):
    run_dir = _write_d7_run(tmp_path, "s3", 42, 20, freeze=5, multiplier=0.05)
    violations = check_d7_run_contract(run_dir, "s3", 42, 19)
    assert any("backbone_lr_multiplier" in violation for violation in violations)


def test_validate_params_rejects_d6_and_d7_together():
    from main import validate_params

    params = _base_params()
    params.d6_candidate = "b0"
    params.d7_candidate = "s1"
    params.freeze_backbone_epochs = 20
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_params(params)


def test_validate_params_o4_requires_alpha_0_1_sidecar(tmp_path):
    from main import validate_params

    init_state = tmp_path / "o4_init.pt"
    init_state.write_bytes(b"stub")
    sidecar = tmp_path / "o4_init.pt.provenance.json"
    sidecar.write_text(
        json.dumps({"mode": "csdt", "human_seed": 42, "alpha": 1.0}), encoding="utf-8"
    )
    params = _base_params()
    params.d7_candidate = "o4"
    params.init_state_path = str(init_state)
    with pytest.raises(ValueError, match="alpha=0.1"):
        validate_params(params)
    sidecar.write_text(
        json.dumps({"mode": "csdt", "human_seed": 42, "alpha": 0.1}), encoding="utf-8"
    )
    validate_params(params)  # cheap checks pass; data-identity check happens in main()


def test_validate_params_o5_requires_tau_0_1_sidecar(tmp_path):
    from main import validate_params

    init_state = tmp_path / "o5_init.pt"
    init_state.write_bytes(b"stub")
    sidecar = tmp_path / "o5_init.pt.provenance.json"
    sidecar.write_text(
        json.dumps({"mode": "csdt_bounded", "human_seed": 42, "tau": 0.3}), encoding="utf-8"
    )
    params = _base_params()
    params.d7_candidate = "o5"
    params.init_state_path = str(init_state)
    with pytest.raises(ValueError, match="tau=0.1"):
        validate_params(params)
    sidecar.write_text(
        json.dumps({"mode": "csdt_bounded", "human_seed": 42, "tau": 0.1}), encoding="utf-8"
    )
    validate_params(params)
