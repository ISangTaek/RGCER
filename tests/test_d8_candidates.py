"""D8 contracts (plan §20-§57, §97-§99).

Covers: functional forgetting statistics (S1 exact zero), A1/A2 last-block
grad scope, O6 deterministic train-only probe, retention trigger semantics
(fire -> next epoch grad 0 -> never unfreeze), and the D8 selector's
STOP_METHOD_ENGINEERING fallback.
"""

import json

import pytest
import torch
from torch import nn
from pathlib import Path

from d8_retention import (
    D8RetentionController,
    build_hybrid_model,
    build_probe_manifest,
    collate_probe_batches,
    evaluate_animal_rmse,
    functional_forgetting_stats,
    select_source_retention_probe,
)


# ----------------------------------------------------------------------
# Functional forgetting statistics (§23-§28, §99: S1 forgetting == 0)
# ----------------------------------------------------------------------
def test_functional_forgetting_s1_is_exactly_zero():
    # §25: an S1 run's backbone is bitwise the teacher backbone, so the
    # hybrid predictions are identical -> forgetting exactly 0.
    teacher = {f"task{i}": 1.0 + i * 0.01 for i in range(8)}
    stats = functional_forgetting_stats(teacher, dict(teacher))
    assert stats["functional_forgetting_abs"] == 0.0
    assert stats["functional_forgetting_exact_zero"] is True
    assert stats["functional_forgetting_relative"] == 0.0
    assert stats["animal_tasks_unchanged"] == 8
    assert stats["animal_tasks_improved"] == 0
    assert stats["animal_tasks_worsened"] == 0


def test_functional_forgetting_counts_and_distribution():
    teacher = {f"t{i}": 1.0 for i in range(10)}
    current = {f"t{i}": 1.0 + 0.1 * (i - 4) for i in range(10)}  # 4 better, 2 equal, 4 worse
    stats = functional_forgetting_stats(teacher, current)
    assert stats["animal_tasks_improved"] == 4
    assert stats["animal_tasks_unchanged"] == 1  # only the exact i=4 delta
    assert stats["animal_tasks_worsened"] == 5
    assert stats["delta_max"] == pytest.approx(0.5)
    assert stats["delta_median"] == pytest.approx(0.0)
    # macro delta = mean of the per-task deltas = +0.05
    assert stats["functional_forgetting_abs"] == pytest.approx(0.05)


class _StubEncoder(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        return self.backbone(batch)


class _AnimalModel(nn.Module):
    """Animal56-shaped stub: constant backbone + per-task quantile decoders."""

    def __init__(self, tasks, scale=1.0, offset=0.0):
        super().__init__()
        self.encoder = _StubEncoder(_ConstBackbone(scale))
        self.decoders = nn.ModuleDict(
            {task: _MedianDecoder(task, offset=offset) for task in tasks}
        )
        self.prediction_mode = "quantile"


class _ConstBackbone(nn.Module):
    def __init__(self, scale):
        super().__init__()
        # scale is a Parameter so it appears in state_dict() and is swapped
        # by build_hybrid_model exactly like a real backbone tensor.
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.layers = nn.ModuleList([_PassLayer() for _ in range(8)])

    def forward(self, batch):
        rows = len(batch.sample_id)
        base = torch.arange(1.0, 5.0).repeat(rows, 1)
        return base * self.scale


class _PassLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.weight


class _MedianDecoder(nn.Module):
    """Real quantile layout: channels = [median, softplus(lower_w), softplus(upper_w)]
    (architecture/prediction_heads.decode_prediction reads median from [..., 0:1])."""

    def __init__(self, task, offset=0.0):
        super().__init__()
        self.task = task
        self.offset = offset

    def forward(self, representation):
        base = representation[:, 0]
        median = base + self.offset
        width = torch.ones_like(median)
        return torch.stack([median, width, width], dim=1)


def test_hybrid_model_swaps_only_backbone():
    tasks = ["t1", "t2"]
    teacher = _AnimalModel(tasks, scale=1.0)
    # The training model's state dict carries the encoder.backbone. prefix.
    current_backbone = {
        f"encoder.backbone.{key}": value
        for key, value in _ConstBackbone(scale=3.0).state_dict().items()
    }
    hybrid = build_hybrid_model(teacher, current_backbone)
    # decoders stay teacher-frozen
    assert hybrid.decoders["t1"].offset == 0.0
    # backbone swapped
    assert float(hybrid.encoder.backbone.scale.detach()) == 3.0


# ----------------------------------------------------------------------
# A1/A2 trainable scope (§39-§43, §99)
# ----------------------------------------------------------------------
def test_a1_scope_only_last_block_and_head_receive_gradients():
    from trainer import Trainer  # import surface stays intact

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])
            self.final_norm = nn.Linear(4, 4)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.final_norm(x)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.backbone = _Backbone()
            self.head = nn.Linear(4, 1)

        def forward(self, x):
            return self.head(self.encoder.backbone(x))

    model = _Model()
    # D8 A1: freeze everything except the LAST backbone block; heads trainable.
    trainable_indices = {7}
    for name, parameter in model.named_parameters():
        if name.startswith("encoder.backbone."):
            trainable = False
            if name.startswith("encoder.backbone.layers."):
                trainable = int(name.split(".")[3]) in trainable_indices
            parameter.requires_grad = trainable
    loss = model(torch.ones(2, 4)).sum()
    loss.backward()
    for index, layer in enumerate(model.encoder.backbone.layers):
        if index in trainable_indices:
            assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
        else:
            assert layer.weight.grad is None, index
    assert model.encoder.backbone.final_norm.weight.grad is None  # frozen too
    assert model.head.weight.grad is not None


def test_a2_scope_last_two_blocks_receive_gradients():
    trainable_indices = {6, 7}
    frozen = [i for i in range(8) if i not in trainable_indices]
    assert trainable_indices | set(frozen) == set(range(8))
    assert min(trainable_indices) == 6


# ----------------------------------------------------------------------
# O6 retention probe (§49-§51, §96, §99)
# ----------------------------------------------------------------------
class _StubStore:
    def __init__(self, train_ids_by_task):
        self.train_ids_by_task = train_ids_by_task
        self.sample_ids = []
        lookup = {}
        for task, ids in train_ids_by_task.items():
            for sample_id in ids:
                if sample_id not in lookup:
                    lookup[sample_id] = len(self.sample_ids)
                    self.sample_ids.append(sample_id)
        self._index_of = lookup
        self.requested_splits = []

    def get_task_indices(self, task, split=None, max_nodes=None):
        self.requested_splits.append(split)
        assert split == "train", "O6 probe may only read the TRAIN split"
        return [self._index_of[sample_id] for sample_id in self.train_ids_by_task[task]]


def test_o6_probe_is_deterministic_and_train_only():
    store = _StubStore(
        {
            "task_a": ["id_x", "id_y", "id_z", "id_w"],
            "task_b": ["id_b1", "id_b2", "id_b3"],
        }
    )
    probe_a = select_source_retention_probe(store, ["task_a", "task_b"], per_task=2)
    probe_b = select_source_retention_probe(store, ["task_a", "task_b"], per_task=2)
    assert probe_a == probe_b  # deterministic across calls
    assert all(len(ids) == 2 for ids in probe_a.values())
    assert set(store.requested_splits) == {"train"}  # §96: never validation
    manifest = build_probe_manifest(probe_a)
    manifest_again = build_probe_manifest(probe_b)
    assert manifest["manifest_sha256"] == manifest_again["manifest_sha256"]
    assert manifest["probe_total"] == 4


def test_o6_probe_manifest_changes_with_membership():
    store_a = _StubStore({"task_a": ["m1", "m2", "m3"]})
    store_b = _StubStore({"task_a": ["m1", "m2", "m4"]})
    manifest_a = build_probe_manifest(select_source_retention_probe(store_a, ["task_a"], 2))
    manifest_b = build_probe_manifest(select_source_retention_probe(store_b, ["task_a"], 2))
    assert manifest_a["manifest_sha256"] != manifest_b["manifest_sha256"]


# ----------------------------------------------------------------------
# O6 retention trigger (§53-§57, §67-§68, §99)
# ----------------------------------------------------------------------
class _TaskBatch(dict):
    def __init__(self, ids, y):
        super().__init__()
        self.sample_id = list(ids)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default


def _controller_world(tmp_path):
    """Build a controller over stub teacher/hybrid models (decoded medians
    encode the per-task offset, so RMSE follows the offsets)."""

    tasks = ["t1", "t2"]
    teacher = _AnimalModel(tasks, offset=0.0)
    hybrid_model = _AnimalModel(tasks, offset=0.0)

    # Probe batches with zero labels: RMSE comes from the decode offset.
    batches = [("t1", _TaskBatch(["p1", "p2"], [0.0, 0.0])), ("t2", _TaskBatch(["p3"], [0.0]))]
    scalers = {task: {"mean": 0.0, "std": 1.0} for task in tasks}

    controller = D8RetentionController(
        model=hybrid_model,
        teacher_model=teacher,
        animal_tasks=tasks,
        task_scalers=scalers,
        device=torch.device("cpu"),
        probe_batches=batches,
        output_dir=tmp_path,
        threshold=0.02,
    )
    return controller, hybrid_model


def test_o6_trigger_freezes_next_epoch_and_never_unfreezes(tmp_path):
    from d8_retention import freeze_last_block

    controller, model = _controller_world(tmp_path)
    # Manually drive the trigger: first epoch damage below threshold...
    controller._teacher_model.decoders["t1"].offset = 0.0
    row = controller.log_epoch(0, model)
    assert controller.triggered is False
    assert row["backbone_trainable"] == 1
    # ... then above threshold.
    controller._teacher_model.decoders["t1"].offset = 1.0
    controller._teacher_model.decoders["t2"].offset = 1.0
    row = controller.log_epoch(5, model)
    assert controller.triggered is True
    assert controller.trigger_epoch == 5
    # §68: epoch 6 START re-freezes (pre_epoch) -> every last-block parameter
    # has requires_grad=False, so the optimizer applies no update (grad None).
    from types import SimpleNamespace

    from trainer import Trainer

    # The trainer applies the D8 trainable scope first...
    scope_holder = SimpleNamespace(
        args=SimpleNamespace(trainable_last_blocks=1, freeze_backbone_epochs=0),
        model=model,
    )
    Trainer._apply_trainable_last_blocks(scope_holder)
    assert model.encoder.backbone.layers[0].weight.requires_grad is False
    assert model.encoder.backbone.layers[7].weight.requires_grad is True
    # ...then the controller re-freezes the last block once triggered.
    controller.pre_epoch(6, model)
    last_block = model.encoder.backbone.layers[7]
    assert all(parameter.requires_grad is False for parameter in last_block.parameters())
    # §56: once frozen, never unfreezes (even when the scope re-applies).
    for epoch in (7, 8, 9):
        Trainer._apply_trainable_last_blocks(scope_holder)
        controller.pre_epoch(epoch, model)
        assert all(parameter.requires_grad is False for parameter in last_block.parameters())
        assert model.encoder.backbone.layers[0].weight.requires_grad is False
    # The trigger fires at the END of epoch 5: that row still reports the
    # trainable state during the epoch; epoch 6's row (P1-1) reports the REAL
    # continued damage with triggered=1.
    assert controller.trigger_epoch == 5
    row6 = controller.log_epoch(6, model)
    assert int(row6["triggered"]) == 1
    assert float(row6["retention_damage_train"]) > 0.0  # real damage, not zeroed
    # P1-4: the firing epoch row itself carries triggered=1 AND
    # trigger_fired_this_epoch=1; the next epoch only keeps triggered=1.
    assert int(controller.rows[2]["trigger_fired_this_epoch"]) == 1
    assert int(row6["trigger_fired_this_epoch"]) == 0
    assert [int(row["triggered"]) for row in controller.rows] == [0, 0, 1, 1]
    assert [row["state"] for row in controller.rows] == [
        "baseline", "post_epoch", "post_epoch", "post_epoch",
    ]
    csv_rows = (tmp_path / "d8_retention_trigger.csv").read_text().strip().splitlines()
    assert len(csv_rows) == 5  # header + baseline + 3 logged epochs
    assert "triggered" in csv_rows[0]


def test_o6_below_threshold_never_triggers(tmp_path):
    controller, model = _controller_world(tmp_path)
    for epoch in range(5):
        row = controller.log_epoch(epoch, model)
        assert controller.triggered is False
        assert float(row["retention_damage_train"]) == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# D8 selector fallback (§70/§106): no candidate beats S1 -> STOP
# ----------------------------------------------------------------------
from scripts.d8_aggregate import (  # noqa: E402
    D8_CANDIDATE_SCOPE,
    d8_a,
    check_d8_run_contract,
)


def test_d8_run_contract_rejects_wrong_multiplier(tmp_path):
    run_dir = tmp_path / "a1" / "d8_a1_e20" / "seed_42"
    run_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "dataset": "toxacute",
                "toxacute_task_scope": "human3",
                "train_eval_scope": "validation_only",
                "fit_conformal": False,
                "epochs": 20,
                "freeze_backbone_epochs": 0,
                "trainable_last_blocks": 1,
                "backbone_lr_multiplier": 0.02,  # wrong: a1 locks 0.01
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "d7_candidate": "a1",
        "manifest_sha256": "m",
        "datastore_fingerprint": "f",
        "feature_schema_version": "s",
        "d7_artifact_contract": {
            "mode": "b1",
            "human_seed": 42,
            "split_manifest_hash": "m",
            "datastore_fingerprint": "f",
            "feature_schema_version": "s",
        },
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    violations = check_d8_run_contract(run_dir, "a1", 42, 19)
    assert any("backbone_lr_multiplier" in violation for violation in violations)

# ----------------------------------------------------------------------
# Sixth-D8 review P0-1: SSE/N aggregation over ALL batches (§6-§8)
# ----------------------------------------------------------------------
def test_evaluate_animal_rmse_aggregates_all_batches():
    # batch1: 2 samples with error 0; batch2: 1 sample with error 3.
    # Correct endpoint RMSE = sqrt(9/3) = sqrt(3); the broken implementation
    # that kept only the LAST batch reported 3.0.
    from d8_retention import evaluate_animal_rmse

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _StubEncoder(_ConstBackbone(scale=1.0))
            self.decoders = nn.ModuleDict({"t1": _MedianDecoder("t1")})

    model = _Model()
    # decoded median = backbone[:, 0] = 1.0 -> y must be 1.0 for zero error
    batch1 = _TaskBatch(["a", "b"], [1.0, 1.0])
    batch2 = _TaskBatch(["c"], [-2.0])  # error = 1 - (-2) = 3
    macro, per_task = evaluate_animal_rmse(
        model, [("t1", batch1), ("t1", batch2)], torch.device("cpu"),
        {"t1": {"mean": 0.0, "std": 1.0}}, ["t1"],
    )
    assert per_task["t1"] == pytest.approx(3 ** 0.5)
    assert macro == pytest.approx(3 ** 0.5)


# ----------------------------------------------------------------------
# Sixth-D8 review P1-1: post-trigger damage is REAL, never zeroed (§36-§40)
# ----------------------------------------------------------------------
def test_o6_post_trigger_damage_is_not_reset_to_zero(tmp_path):
    controller, model = _controller_world(tmp_path)
    controller.log_epoch(0, model)  # damage 0, no trigger
    controller._teacher_model.decoders["t1"].offset = 1.0
    controller._teacher_model.decoders["t2"].offset = 1.0
    controller.log_epoch(5, model)  # damage 1.0 -> trigger
    assert controller.triggered is True
    # Real continued drift: perturb the (already drifted) backbone — the
    # post-trigger log must reflect the TRUE damage, not a synthetic 0.
    with torch.no_grad():
        model.encoder.backbone.scale.add_(2.0)
    row = controller.log_epoch(6, model)
    assert float(row["retention_damage_train"]) > 0.0
    assert float(row["retention_damage_train"]) == pytest.approx(3.0)


# ----------------------------------------------------------------------
# Sixth-D8 review P1-2: O6 trigger state survives resume (§41-§48)
# ----------------------------------------------------------------------
def test_o6_resume_preserves_trigger_state(tmp_path):
    controller, model = _controller_world(tmp_path)
    controller._teacher_model.decoders["t1"].offset = 1.0
    controller._teacher_model.decoders["t2"].offset = 1.0
    controller.log_epoch(5, model)
    assert controller.triggered is True

    state = controller.state_dict()
    assert state["triggered"] is True and state["trigger_epoch"] == 5

    fresh = D8RetentionController(
        model=model,
        teacher_model=controller._teacher_model,
        animal_tasks=controller.animal_tasks,
        task_scalers=controller.task_scalers,
        device=torch.device("cpu"),
        probe_batches=controller.probe_batches,
        output_dir=tmp_path / "resumed",
        threshold=0.02,
    )
    fresh.load_state_dict(state)
    assert fresh.triggered is True
    assert fresh.trigger_epoch == 5
    # §68: the first resumed epoch start keeps the last block frozen.
    fresh.pre_epoch(6, model)
    assert all(
        parameter.requires_grad is False
        for parameter in model.encoder.backbone.layers[7].parameters()
    )
    # threshold mismatch must fail loudly
    with pytest.raises(ValueError, match="threshold mismatch"):
        fresh.load_state_dict({**state, "threshold": 0.05})


def test_trainer_checkpoint_carries_retention_state():
    # §45: the checkpoint payload must embed the controller state so a resume
    # can restore it; the field stays optional for non-O6 runs.
    import inspect

    from trainer import Trainer

    source = inspect.getsource(Trainer._checkpoint_payload)
    assert "d8_retention_state" in source
    load_source = inspect.getsource(Trainer.load_checkpoint)
    assert "pending_d8_retention_state" in load_source


# ----------------------------------------------------------------------
# Sixth-D8 review P0-6: matched-teacher provenance binding (§27-§34)
# ----------------------------------------------------------------------
import hashlib  # noqa: E402

from scripts.d8_functional_forgetting import (  # noqa: E402
    _load_run_backbone,
    verify_run_provenance,
    verify_teacher_bundle,
)


def _functional_forgetting_world(
    tmp_path,
    *,
    teacher_seed=42,
    sidecar_sha=None,
    sidecar_seed=42,
    contract_manifest="m",
    run_payload_manifest="m",
    with_run_checkpoint=True,
):
    from tests.test_d6_counterfactual import _teacher_payload, _write_teacher

    teacher_dir = _write_teacher(tmp_path, "teacher", _teacher_payload(False, seed=teacher_seed))
    checkpoint_sha = hashlib.sha256((teacher_dir / "teacher_last.pt").read_bytes()).hexdigest()

    init_path = tmp_path / "b1_init.pt"
    init_path.write_bytes(b"stub-init")
    provenance = {
        "mode": "b1",
        "teacher_real_run_dir": str(teacher_dir),
        "teacher_real_checkpoint_sha256": sidecar_sha if sidecar_sha is not None else checkpoint_sha,
        "expected_teacher_epoch": 29,
        "teacher_initial_model_sha256": "teacher-init-hash",
        "human_seed": sidecar_seed,
        "output_sha256": hashlib.sha256(init_path.read_bytes()).hexdigest(),
    }
    (tmp_path / "b1_init.pt.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

    run_dir = tmp_path / "run"
    (run_dir / "diagnostics").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "d7_candidate": "o6",
                "d7_artifact_contract": {
                    "mode": "b1",
                    "human_seed": 42,
                    "split_manifest_hash": contract_manifest,
                    "datastore_fingerprint": "f",
                    "feature_schema_version": "s",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "args.json").write_text(
        json.dumps({"init_state_path": str(init_path)}), encoding="utf-8"
    )
    if with_run_checkpoint:
        torch.save(
            {
                "model_state": {"encoder.backbone.w": torch.zeros(1)},
                "split_manifest_hash": run_payload_manifest,
                "data_config": {"datastore_fingerprint": "f"},
                "feature_schema_version": "s",
            },
            run_dir / "graphormer_last.pt",
        )
    return run_dir, teacher_dir


def test_functional_forgetting_rejects_wrong_teacher_seed(tmp_path):
    run_dir, teacher_dir = _functional_forgetting_world(tmp_path, teacher_seed=43)
    bundle = verify_run_provenance(run_dir, teacher_dir)
    payload = torch.load(Path(teacher_dir) / "teacher_last.pt", weights_only=False)
    with pytest.raises(ValueError, match="base_seed"):
        verify_teacher_bundle(bundle, Path(teacher_dir) / "teacher_last.pt", payload)


def test_functional_forgetting_rejects_wrong_teacher_checkpoint(tmp_path):
    run_dir, teacher_dir = _functional_forgetting_world(tmp_path, sidecar_sha="0" * 64)
    bundle = verify_run_provenance(run_dir, teacher_dir)
    payload = torch.load(Path(teacher_dir) / "teacher_last.pt", weights_only=False)
    with pytest.raises(ValueError, match="sha256 does not match"):
        verify_teacher_bundle(bundle, Path(teacher_dir) / "teacher_last.pt", payload)


def test_functional_forgetting_rejects_wrong_manifest(tmp_path):
    run_dir, teacher_dir = _functional_forgetting_world(
        tmp_path, contract_manifest="m", run_payload_manifest="other-manifest"
    )
    bundle = verify_run_provenance(run_dir, teacher_dir)
    with pytest.raises(ValueError, match="data identity|split_manifest_hash"):
        _load_run_backbone(run_dir, bundle["contract"])

# ----------------------------------------------------------------------
# Seventh-D8 review P0-3: O6 protocol locks (§28-§33)
# ----------------------------------------------------------------------
from tests.test_d7_candidates import _write_b1_sidecar  # noqa: E402


def _o6_params(tmp_path, *, probe_per_task=16, threshold=0.02):
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
    params.d7_candidate = "o6"
    params.seed = 42
    params.epochs = 20
    params.freeze_backbone_epochs = 0
    params.backbone_lr_multiplier = 0.02
    params.trainable_last_blocks = 1
    params.feature_drift_epochs = "0,5,10,15,19"
    params.retention_probe_per_task = probe_per_task
    params.retention_damage_threshold = threshold
    params.init_state_path = _write_b1_sidecar(tmp_path, name="o6_init.pt")
    return params


def test_o6_rejects_probe_size_not_16(tmp_path):
    from main import validate_params

    params = _o6_params(tmp_path, probe_per_task=15)
    with pytest.raises(ValueError, match="retention_probe_per_task=16"):
        validate_params(params)
    params = _o6_params(tmp_path, probe_per_task=32)
    with pytest.raises(ValueError, match="retention_probe_per_task=16"):
        validate_params(params)
    validate_params(_o6_params(tmp_path))  # 16 passes


def test_o6_rejects_trigger_threshold_not_002(tmp_path):
    from main import validate_params

    for bad in (0.01, 0.05, 0.123):
        params = _o6_params(tmp_path, threshold=bad)
        with pytest.raises(ValueError, match="retention_damage_threshold=0.02"):
            validate_params(params)
    validate_params(_o6_params(tmp_path))  # 0.02 passes


def test_o6_rejects_retention_teacher_wrong_seed(tmp_path):
    # P0-4: the retention teacher must be seed-matched to the candidate —
    # the single-real contract rejects any other base seed.
    from tests.test_d6_counterfactual import _teacher_payload, _write_teacher

    run_dir = _write_teacher(tmp_path, "teacher", _teacher_payload(False, seed=43))
    payload = torch.load(run_dir / "teacher_last.pt", weights_only=False)
    from scripts.d6_prep_inits import _verify_single_real_teacher

    with pytest.raises(SystemExit, match="base seed"):
        _verify_single_real_teacher(
            payload, run_dir, expected_model_seed=42, expected_epoch=29
        )


def test_o6_run_metadata_records_retention_teacher_sha():
    # P0-4 (§42-§44): the run metadata writer must persist the probe manifest
    # hash and the bound retention teacher identity.
    import inspect

    from main import _write_run_metadata

    source = inspect.getsource(_write_run_metadata)
    for field in (
        "source_train_probe_manifest_sha256",
        "o6_retention_teacher_checkpoint_sha256",
        "o6_retention_teacher_initial_model_sha256",
    ):
        assert field in source
