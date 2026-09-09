import copy

import pytest

from baselines.checkpoint import StrictBest, protocol_identity, validate_checkpoint_contract, validate_protocol_identity
from baselines.config import resolve_training_config
from baselines.contracts import expected_feature_schema
from baselines.constants import HUMAN3_TASKS
from baselines.models.dmpnn import NoamLikeScheduler
from baselines.models.toxacol import toxacol_learning_rate
from baselines.scaling import TaskScaler
from baselines.trials import finalize_learning_rates, generate_trials, rf_full_grid


class OptimizerStub:
    def __init__(self):
        self.param_groups = [{"lr": 0.0}]


def test_strict_best_preserves_earlier_tie(tmp_path):
    best = StrictBest(tmp_path / "best")
    writes = []
    assert best.consider(1.0, 0, lambda path: writes.append((path, 0)))
    assert not best.consider(1.0, 1, lambda path: writes.append((path, 1)))
    assert best.best_epoch == 0
    assert len(writes) == 1


def test_protocol_checkpoint_identity_rejects_drift():
    identity = protocol_identity()
    validate_protocol_identity(identity)
    identity["split_manifest_hash"] = "wrong"
    with pytest.raises(ValueError, match="split_manifest_hash"):
        validate_protocol_identity(identity)


def test_noam_schedule_hits_max_then_final():
    optimizer = OptimizerStub()
    scheduler = NoamLikeScheduler(optimizer, warmup_epochs=1, total_epochs=2, steps_per_epoch=2,
                                  init_lr=1e-5, max_lr=1e-3, final_lr=1e-5)
    assert scheduler.step() < 1e-3
    assert scheduler.step() == pytest.approx(1e-3)
    scheduler.step()
    assert scheduler.step() == pytest.approx(1e-5)


def test_toxacol_lut_is_zero_based_first_strict_upper_bound():
    assert toxacol_learning_rate(0, [1, 3], [0.1, 0.01, 0.001]) == 0.1
    assert toxacol_learning_rate(1, [1, 3], [0.1, 0.01, 0.001]) == 0.01
    assert toxacol_learning_rate(3, [1, 3], [0.1, 0.01, 0.001]) == 0.001


def test_trial_lists_are_deterministic_unique_and_exact_size():
    expected = {"rf": 50, "afp": 20, "dmpnn": 20, "grover": 10, "toxacol": 10}
    for method, count in expected.items():
        first = finalize_learning_rates(method, generate_trials(method, 42))
        second = finalize_learning_rates(method, generate_trials(method, 42))
        assert first == second
        assert len(first) == count
        assert len({repr(sorted(row.items())) for row in first}) == count


def test_rf_frozen_grid_fields_candidates_and_sampling():
    grid = rf_full_grid()
    assert len(grid) == 54
    assert set().union(*(row.keys() for row in grid)) == {
        "n_estimators", "max_depth", "min_samples_leaf", "max_features"
    }
    assert {row["n_estimators"] for row in grid} == {500, 1000}
    assert {row["max_depth"] for row in grid} == {None, 10, 30}
    assert {row["min_samples_leaf"] for row in grid} == {1, 2, 4}
    assert {row["max_features"] for row in grid} == {"sqrt", 0.3, 1.0}
    sampled = generate_trials("rf", 42)
    assert sampled == [grid[index] for index in __import__("numpy").random.RandomState(42).choice(54, 50, replace=False)]


def test_smoke_and_formal_configs_share_resolver_but_effective_values_differ():
    first = resolve_training_config("afp", role="smoke", seed=42, device="cpu")
    second = resolve_training_config("afp", role="smoke", seed=42, device="cpu", overrides={"lr": 0.0005})
    assert first["training"]["lr"] == 0.001
    assert second["training"]["lr"] == 0.0005
    assert first["smoke_override"] and second["smoke_override"]
    formal = resolve_training_config("toxacol", role="formal", seed=42, device="cpu", trial_index=0)
    assert formal["training_task_scope"] == "joint59"
    assert formal["training"]["epochs"] == 120
    assert formal["training"]["batch_size"] == 32


def _valid_rf_checkpoint():
    config = resolve_training_config("rf", role="smoke", seed=42, device="cpu")
    scaler = TaskScaler.fit(__import__("numpy").ones((2, 3)), HUMAN3_TASKS, allow_empty=False)
    return {
        **protocol_identity(),
        "method": "rf",
        "model_type": config["model_type"],
        "task_names": list(HUMAN3_TASKS),
        "feature_schema": expected_feature_schema("rf"),
        "config": config,
        "scaler": scaler.to_dict(),
    }


@pytest.mark.parametrize("mutation,match", [
    (lambda payload: payload.update(task_names=list(reversed(HUMAN3_TASKS))), "task order"),
    (lambda payload: payload.update(feature_schema={"schema_hash": "wrong"}), "feature schema"),
    (lambda payload: payload.pop("scaler"), "missing scaler"),
])
def test_checkpoint_contract_rejects_task_schema_and_scaler_drift(mutation, match):
    payload = copy.deepcopy(_valid_rf_checkpoint())
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        validate_checkpoint_contract(payload, expected_method="rf")
