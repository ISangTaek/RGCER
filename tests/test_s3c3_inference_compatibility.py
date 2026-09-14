"""S3C3-R2 inference-only historical checkpoint compatibility contract."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from torch import nn

from checkpoint_inference_compatibility import (
    ADAPTATION_FIELDS,
    CHECKPOINT_CONFIGURATION_FIELDS,
    HUMAN3_TASKS,
    AssetCompatibilityRule,
    InferenceCompatibilityError,
    InferenceCompatibilityPolicy,
    _prepare_inference_checkpoint_with_policy,
    sha256_file,
)
from loss import MSELoss
from trainer import Trainer
from weighting.EW import EW


class _TinyBatch:
    def __init__(self, values):
        self.x = torch.as_tensor(values, dtype=torch.float32)


class _TinyHuman3Architecture(nn.Module):
    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        super().__init__()
        del encoder_class, decoders, device, args, kwargs
        self.encoder = nn.Identity()
        self.heads = nn.ModuleDict(
            {task: nn.Linear(4, 3) for task in task_name}
        )

    def forward(
        self,
        batch,
        task_name=None,
        return_all_tasks=False,
        return_aux=False,
        source_mask=None,
        **kwargs,
    ):
        del source_mask, kwargs
        tasks = list(self.heads) if return_all_tasks else [task_name]
        result = {task: self.heads[task](batch.x) for task in tasks}
        if return_aux:
            return result, {}
        return result


def _adaptation(method="B1"):
    if method == "B0":
        d6_candidate, d7_candidate, freeze = "b0", "none", 0
        feature_epochs = "0,5,10,15,19"
    elif method == "B1":
        d6_candidate, d7_candidate, freeze = "b1", "none", 0
        feature_epochs = "0,5,10,15,20,25,30,35,39"
    elif method == "RPT":
        d6_candidate, d7_candidate, freeze = "none", "s1", 40
        feature_epochs = "0,5,10,15,20,25,30,35,39"
    else:
        raise AssertionError(method)
    return {
        "d6_candidate": d6_candidate,
        "d7_candidate": d7_candidate,
        "freeze_backbone_epochs": freeze,
        "backbone_lr_multiplier": 1.0,
        "trainable_last_blocks": 0,
        "retention_probe_per_task": 16,
        "retention_damage_threshold": 0.02,
        "feature_drift_probe_size": 128,
        "feature_drift_epochs": feature_epochs,
    }


def _metadata():
    return {
        "format_version": 2,
        "build_id": "synthetic-human3-v2",
        "datastore_fingerprint": "a" * 64,
        "raw_csv_sha256": "b" * 64,
        "feature_schema_version": "atom_v2_bond_v1_pathavg_v1",
        "max_path_distance": 8,
        "splitting": "scaffold",
        "split_seed": 42,
        "split_ratios": {
            "train": 0.7,
            "validation": 0.1,
            "calibration": 0.1,
            "test": 0.1,
        },
        "split_manifest_hash": "c" * 64,
    }


def _args(tmp_path, *, method="B1", mode="train", split=None):
    adaptation = _adaptation(method)
    return SimpleNamespace(
        seed=42,
        gpu_id="cpu",
        mode=mode,
        inference_split=split,
        save_path=str(tmp_path),
        load_path=None,
        preprocessed_data_dir=None,
        data_store_dir=None,
        datastore_metadata=_metadata(),
        max_path_distance=8,
        max_nodes_filter=512,
        prediction_mode="quantile",
        conformal_alpha=0.10,
        min_calibration_size=1,
        fit_conformal=False,
        hps_warmup_epochs=0,
        lambda_base=0.25,
        lambda_quantile=1.0,
        router_top_k=2,
        router_temperature=1.0,
        routing_enabled=True,
        rgcer_use_source_response=True,
        rgcer_use_target_response=True,
        rgcer_use_molecule_query=True,
        rgcer_use_sparse_routing=True,
        rgcer_use_null_route=True,
        rgcer_use_film=True,
        rgcer_use_adapter=True,
        rgcer_use_base_aux_loss=True,
        rgcer_fallback_space="prediction",
        rgcer_transfer_mechanism="endpoint_router",
        rgcer_source_policy="all_except_target",
        explicitly_allowed_auxiliary_sources=(),
        exclude_target_from_sources=True,
        use_factorized_prompt=False,
        lower_quantile=0.05,
        upper_quantile=0.95,
        ckpt_name="model",
        edge_bias_mode="path",
        spatial_pos_clip=20,
        hidden_dim=4,
        a_layers=1,
        a_heads=1,
        mid_dim=4,
        head_hidden_dim=4,
        head_dropout=0.0,
        adapter_ratio=0.25,
        response_hidden_dim=4,
        task_sampling="proportional",
        conformal_scope="human3",
        selection_scope="human3",
        card_lambda_delta=0.0,
        card_bottleneck=32,
        effective_rgcer_config=None,
        dataset="toxacute",
        splitting="scaffold",
        split_seed=42,
        arch="Graphormer",
        train_eval_scope="validation_only",
        epochs=40,
        **adaptation,
    )


def _task_dict():
    return {
        task: {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]}
        for task in HUMAN3_TASKS
    }


def _trainer(tmp_path, *, method="B1", mode="train", split=None, load_path=None, compatibility=None):
    args = _args(tmp_path, method=method, mode=mode, split=split)
    args.load_path = str(load_path) if load_path is not None else None
    return Trainer(
        task_dict=_task_dict(),
        weighting=EW,
        architecture=_TinyHuman3Architecture,
        encoder_class=nn.Identity,
        decoders=nn.ModuleDict(),
        optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        args=args,
        save_path=tmp_path,
        load_path=load_path,
        inference_compatibility=compatibility,
    )


def _source_payload(tmp_path, *, method="B1"):
    source = _trainer(tmp_path / "source", method=method)
    source.task_scalers = {
        task: {"mean": float(index + 1), "std": float(index + 2), "count": 3}
        for index, task in enumerate(HUMAN3_TASKS)
    }
    checkpoint_path = source._save_checkpoint(0, "model_best.pt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return source, payload


def _write_payload(tmp_path, payload, *, name="candidate.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _policy_for(path, payload, *, method="B1", migration_required=True, adaptation=None):
    expected_adaptation = dict(adaptation or _adaptation(method))
    model_identity = copy.deepcopy(payload["architecture_config"])
    for name in ADAPTATION_FIELDS:
        model_identity.pop(name, None)
    data_identity = copy.deepcopy(payload["data_config"])
    policy = InferenceCompatibilityPolicy(
        version="synthetic-test-policy-v1",
        assets=(
            AssetCompatibilityRule(
                run_id=f"synthetic_{method}_s42",
                method=method,
                seed=42,
                checkpoint_sha256=sha256_file(path),
                checkpoint_size_bytes=Path(path).stat().st_size,
                best_epoch=0,
                migration_required=migration_required,
                adaptation_config=MappingProxyType(expected_adaptation),
            ),
        ),
        task_names=HUMAN3_TASKS,
        configuration_identity=MappingProxyType(
            {
                "dataset": "toxacute",
                "splitting": "scaffold",
                "split_seed": 42,
                "prediction_mode": "quantile",
            }
        ),
        model_identity=MappingProxyType(model_identity),
        data_identity=MappingProxyType(data_identity),
    )
    return policy


def _legacy_payload(payload):
    result = copy.deepcopy(payload)
    for name in ADAPTATION_FIELDS:
        result["architecture_config"].pop(name)
    for name in (
        "trainable_last_blocks",
        "retention_probe_per_task",
        "retention_damage_threshold",
    ):
        result["configuration"].pop(name)
    return result


def _prepare(path, policy, *, method="B1", operation="batch_inference", split="validation"):
    return _prepare_inference_checkpoint_with_policy(
        path,
        run_id=f"synthetic_{method}_s42",
        method=method,
        seed=42,
        operation=operation,
        split=split,
        expected_policy_sha256=policy.computed_sha256(),
        policy=policy,
    )


@pytest.mark.parametrize("method", ["B1", "RPT"])
def test_exact_legacy_b1_and_s1_assets_build_only_the_audited_view(tmp_path, method):
    _, full = _source_payload(tmp_path, method=method)
    legacy = _legacy_payload(full)
    path = _write_payload(tmp_path, legacy, name=f"legacy-{method}.pt")
    policy = _policy_for(path, full, method=method, migration_required=True)

    prepared = _prepare(path, policy, method=method)

    assert prepared.compatibility.migration_applied is True
    assert tuple(item["field"] for item in prepared.compatibility.additions) == ADAPTATION_FIELDS
    sources = {item["field"]: item["source"] for item in prepared.compatibility.additions}
    assert all(
        sources[name] == "checkpoint_configuration"
        for name in CHECKPOINT_CONFIGURATION_FIELDS
    )
    assert all(
        sources[name] == "codex_inference_compatibility_constant"
        for name in set(ADAPTATION_FIELDS) - set(CHECKPOINT_CONFIGURATION_FIELDS)
    )
    assert prepared.compatibility.architecture_config_view == full["architecture_config"]
    assert all(name not in legacy["architecture_config"] for name in ADAPTATION_FIELDS)


def test_full_new_b0_checkpoint_uses_zero_migration_and_predictions_are_unchanged(tmp_path):
    source, full = _source_payload(tmp_path, method="B0")
    path = _write_payload(tmp_path, full, name="full-b0.pt")
    policy = _policy_for(path, full, method="B0", migration_required=False)
    prepared = _prepare(path, policy, method="B0")
    assert prepared.compatibility.migration_applied is False
    assert prepared.compatibility.additions == ()

    strict = _trainer(
        tmp_path / "strict",
        method="B0",
        mode="batch_inference",
        split="validation",
        load_path=path,
    )
    compatible = _trainer(
        tmp_path / "compatible",
        method="B0",
        mode="batch_inference",
        split="validation",
        load_path=path,
        compatibility=prepared.compatibility,
    )
    batch = _TinyBatch([[1, 2, 3, 4], [4, 3, 2, 1]])
    strict_predictions = strict.predict_all_tasks(batch)
    compatible_predictions = compatible.predict_all_tasks(batch)
    for task in HUMAN3_TASKS:
        torch.testing.assert_close(strict_predictions[task], compatible_predictions[task])
        torch.testing.assert_close(
            source.model.state_dict()[f"heads.{task}.weight"],
            compatible.model.state_dict()[f"heads.{task}.weight"],
        )


@pytest.mark.parametrize("value", [None, "0", False])
def test_legacy_explicit_null_wrong_type_or_bool_is_not_missing(tmp_path, value):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    legacy["architecture_config"]["trainable_last_blocks"] = value
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match="all nine"):
        _prepare(path, policy)


@pytest.mark.parametrize("field", CHECKPOINT_CONFIGURATION_FIELDS)
def test_every_checkpoint_configuration_source_field_is_required(tmp_path, field):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    legacy["configuration"].pop(field)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match=field):
        _prepare(path, policy)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("d6_candidate", "none"),
        ("d7_candidate", "s1"),
        ("freeze_backbone_epochs", True),
        ("backbone_lr_multiplier", 1),
        ("feature_drift_probe_size", False),
        ("feature_drift_epochs", None),
    ],
)
def test_configuration_source_conflict_or_wrong_type_is_rejected(tmp_path, field, bad_value):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    legacy["configuration"][field] = bad_value
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match=field):
        _prepare(path, policy)


@pytest.mark.parametrize(
    ("location", "field", "bad_value"),
    [
        ("checkpoint", "checkpoint_version", 5),
        ("checkpoint", "epoch", 1),
        ("checkpoint", "task_names", list(reversed(HUMAN3_TASKS))),
        ("configuration", "seed", 43),
        ("configuration", "dataset", "other"),
        ("data_config", "datastore_fingerprint", "d" * 64),
    ],
)
def test_bad_schema_epoch_task_seed_config_or_data_identity_is_rejected(
    tmp_path, location, field, bad_value
):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    target = legacy if location == "checkpoint" else legacy[location]
    target[field] = bad_value
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError):
        _prepare(path, policy)


def test_checkpoint_sha_size_and_policy_sha_are_all_binding(tmp_path):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(InferenceCompatibilityError, match="size"):
        _prepare(path, policy)

    clean_path = _write_payload(tmp_path, legacy, name="clean.pt")
    clean_policy = _policy_for(clean_path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match="policy SHA"):
        _prepare_inference_checkpoint_with_policy(
            clean_path,
            run_id="synthetic_B1_s42",
            method="B1",
            seed=42,
            operation="batch_inference",
            split="validation",
            expected_policy_sha256="0" * 64,
            policy=clean_policy,
        )


def test_changed_inference_constant_is_rejected_even_under_rehashed_policy(tmp_path):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    changed = replace(
        policy,
        compatibility_constants=MappingProxyType(
            {
                "trainable_last_blocks": 1,
                "retention_probe_per_task": 16,
                "retention_damage_threshold": 0.02,
            }
        ),
    )
    with pytest.raises(InferenceCompatibilityError, match="constants"):
        _prepare(path, changed)


def test_partial_missing_and_extra_legacy_patterns_are_rejected(tmp_path):
    _, full = _source_payload(tmp_path)
    partial = copy.deepcopy(full)
    partial["architecture_config"].pop("d6_candidate")
    partial_path = _write_payload(tmp_path, partial, name="partial.pt")
    full_policy = _policy_for(partial_path, full, migration_required=False)
    with pytest.raises(InferenceCompatibilityError, match="d6_candidate"):
        _prepare(partial_path, full_policy)

    legacy = _legacy_payload(full)
    legacy["architecture_config"]["d6_candidate"] = "b1"
    legacy_path = _write_payload(tmp_path, legacy, name="extra.pt")
    legacy_policy = _policy_for(legacy_path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match="all nine"):
        _prepare(legacy_path, legacy_policy)


def test_unapproved_run_candidate_and_seed_are_rejected(tmp_path):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match="not uniquely authorized"):
        _prepare_inference_checkpoint_with_policy(
            path,
            run_id="unknown",
            method="B1",
            seed=42,
            operation="batch_inference",
            split="validation",
            expected_policy_sha256=policy.computed_sha256(),
            policy=policy,
        )
    with pytest.raises(InferenceCompatibilityError, match="method/seed"):
        _prepare_inference_checkpoint_with_policy(
            path,
            run_id="synthetic_B1_s42",
            method="RPT",
            seed=42,
            operation="batch_inference",
            split="validation",
            expected_policy_sha256=policy.computed_sha256(),
            policy=policy,
        )
    with pytest.raises(InferenceCompatibilityError, match="method/seed"):
        _prepare_inference_checkpoint_with_policy(
            path,
            run_id="synthetic_B1_s42",
            method="B1",
            seed=43,
            operation="batch_inference",
            split="validation",
            expected_policy_sha256=policy.computed_sha256(),
            policy=policy,
        )


@pytest.mark.parametrize("operation", ["train", "resume", "calibration", "test"])
def test_compatibility_entry_rejects_train_resume_calibration_and_implicit_test(tmp_path, operation):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match="batch_inference"):
        _prepare(path, policy, operation=operation)


def test_old_strict_candidate_resume_rejection_is_unchanged():
    stored = _adaptation("B0")
    current = _adaptation("B1")
    with pytest.raises(ValueError, match="d6_candidate"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_synthetic_human3_save_compatible_load_predict_is_finite_and_bitwise_bound(tmp_path):
    source, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    path = _write_payload(tmp_path, legacy)
    before_checkpoint_sha = sha256_file(path)
    policy = _policy_for(path, full, migration_required=True)
    prepared = _prepare(path, policy)
    loaded = _trainer(
        tmp_path / "loaded",
        mode="batch_inference",
        split="validation",
        load_path=path,
        compatibility=prepared.compatibility,
    )

    assert list(loaded.task_name) == list(HUMAN3_TASKS)
    assert list(loaded.model.state_dict()) == list(full["model_state"])
    for name, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor.detach().cpu(), full["model_state"][name])
    batch = _TinyBatch([[0, 1, 2, 3], [3, 2, 1, 0]])
    predictions = loaded.predict_all_tasks(batch)
    assert list(predictions) == list(HUMAN3_TASKS)
    for task in HUMAN3_TASKS:
        decoded = loaded.decode_task_output(task, predictions[task], apply_conformal=False)
        assert torch.isfinite(decoded["median"]).all()
        assert torch.isfinite(decoded["lower"]).all()
        assert torch.isfinite(decoded["upper"]).all()
    assert sha256_file(path) == before_checkpoint_sha
    assert all(
        torch.equal(source.model.state_dict()[name], loaded.model.state_dict()[name])
        for name in source.model.state_dict()
    )
    with pytest.raises(RuntimeError, match="training or resume"):
        loaded.train({}, {})
    with pytest.raises(RuntimeError, match="calibration"):
        loaded._fit_conformal({})


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.__setitem__("quantile_config", {"lower": 0.1, "upper": 0.9}), "quantile_config"),
        (lambda payload: payload.__setitem__("routing_enabled", 1), "routing_enabled"),
        (lambda payload: payload["task_scalers"][HUMAN3_TASKS[0]].__setitem__("std", 0.0), "task_scalers"),
        (lambda payload: payload["architecture_config"].__setitem__("hidden_dim", 8), "hidden_dim"),
    ],
)
def test_quantile_routing_scaler_and_dimension_contracts_still_reject(
    tmp_path, mutation, message
):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    mutation(legacy)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    with pytest.raises(InferenceCompatibilityError, match=message):
        _prepare(path, policy)


def test_tensor_shape_contract_still_rejects_inside_strict_trainer_load(tmp_path):
    _, full = _source_payload(tmp_path)
    legacy = _legacy_payload(full)
    tensor_name = next(name for name in legacy["model_state"] if name.endswith("weight"))
    legacy["model_state"][tensor_name] = torch.zeros(1)
    path = _write_payload(tmp_path, legacy)
    policy = _policy_for(path, full, migration_required=True)
    prepared = _prepare(path, policy)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _trainer(
            tmp_path / "bad-tensor",
            mode="batch_inference",
            split="validation",
            load_path=path,
            compatibility=prepared.compatibility,
        )
