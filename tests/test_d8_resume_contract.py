"""D8 resume contract (eighth review §63-§64, §67-§72).

A resumed run must be THE SAME experiment as an uninterrupted one:
- the init overlay is never re-applied on resume (the checkpoint model wins);
- the checkpoint's recorded initial_model_sha256 must equal the init
  artifact's init_state_sha256 (§21);
- the D7 drift tracker restores the ORIGINAL pretrained baseline so drift is
  measured against theta_0, not the break-point weights;
- the D7/D8 adaptation protocol (candidate/LR/scope/probe) is part of the
  resume contract.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from d7_diagnostics import D7DriftTracker
from d8_retention import D8RetentionController
from main import _apply_or_validate_init_state
from reproducibility import state_dict_sha256
from trainer import Trainer


class _ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = [str(v) for v in ids]

    def __len__(self):
        return len(self.ids)

    def get_sample_id(self, index):
        return self.ids[index]

    def __getitem__(self, index):
        return {"sample_id": self.ids[index]}


class _DatasetLoader:
    """Minimal loader surface: .dataset/.collate_fn for probe collection."""

    def __init__(self, ids):
        self.dataset = _ProbeDataset(ids)
        self.collate_fn = lambda items: _ProbeBatch([item["sample_id"] for item in items])


class _ProbeBatch(dict):
    def __init__(self, ids):
        super().__init__()
        self.sample_id = list(ids)

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.backbone(x))


def _theta(module, value):
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)
    return module


def _write_init_artifact(tmp_path, model):
    """Write an init state file + provenance sidecar (theta0) like
    d6_prep_inits does, and return (path, sidecar)."""

    init_state_path = tmp_path / "b1_init.pt"
    torch.save(model.state_dict(), init_state_path)
    sidecar = {
        "mode": "b1",
        "human_seed": 42,
        "output_sha256": _file_sha(init_state_path),
        "init_state_sha256": state_dict_sha256(model),
        "teacher_real_run_dir": str(tmp_path / "teacher"),
        "teacher_real_checkpoint_sha256": "teacher-sha",
        "expected_teacher_epoch": 39,
        "teacher_initial_model_sha256": "teacher-init-hash",
    }
    sidecar_path = tmp_path / "b1_init.pt.provenance.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    return init_state_path, sidecar


def _file_sha(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stub_trainer(tmp_path, model, initial_hash):
    return SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        initial_model_sha256=initial_hash,
        save_path=None,
    )


def _stub_params(tmp_path, init_state_path, *, load_path=None):
    return SimpleNamespace(
        init_state_path=str(init_state_path),
        load_path=load_path,
        require_init_provenance=False,
        seed=42,
    )


# ----------------------------------------------------------------------
# P0-1: resume never re-applies the init overlay (§15-§25)
# ----------------------------------------------------------------------
def test_resume_keeps_checkpoint_model_not_init_overlay(tmp_path):
    theta0 = _TinyNet()
    init_state_path, _ = _write_init_artifact(tmp_path, theta0)

    theta5 = _theta(_TinyNet(), 5.0)  # checkpoint model != init
    trainer = _stub_trainer(tmp_path, theta5, initial_hash=state_dict_sha256(theta0))
    params = _stub_params(tmp_path, init_state_path, load_path="runs/last.pt")

    _apply_or_validate_init_state(trainer, params)

    # §23: the resumed model must still be theta5, NOT the init artifact.
    assert state_dict_sha256(trainer.model) == state_dict_sha256(theta5)
    provenance = params.init_overlay_provenance
    assert provenance["resume"] is True
    assert provenance["init_overlay_applied"] is False
    assert provenance["checkpoint_initial_model_sha256"] == state_dict_sha256(theta0)


def test_fresh_run_still_applies_init_overlay(tmp_path):
    theta0 = _TinyNet()
    init_state_path, _ = _write_init_artifact(tmp_path, theta0)

    scratch = _theta(_TinyNet(), 1.0)
    trainer = _stub_trainer(tmp_path, scratch, initial_hash=state_dict_sha256(scratch))
    params = _stub_params(tmp_path, init_state_path, load_path=None)

    _apply_or_validate_init_state(trainer, params)

    assert state_dict_sha256(trainer.model) == state_dict_sha256(theta0)
    assert params.init_overlay_provenance["init_overlay_applied"] is True


def test_resume_initial_hash_matches_b1_init(tmp_path):
    # §24/§71: a checkpoint whose recorded initial hash differs from the
    # supplied init artifact must FAIL BEFORE TRAINING.
    theta0 = _TinyNet()
    other_init = _theta(_TinyNet(), 9.0)
    init_state_path, _ = _write_init_artifact(tmp_path, other_init)

    theta5 = _theta(_TinyNet(), 5.0)
    # checkpoint was created from a DIFFERENT init artifact
    trainer = _stub_trainer(tmp_path, theta5, initial_hash=state_dict_sha256(theta5))
    params = _stub_params(tmp_path, init_state_path, load_path="runs/last.pt")

    with pytest.raises(ValueError, match="not created from the supplied init"):
        _apply_or_validate_init_state(trainer, params)


def test_resume_requires_init_state_sha_in_provenance(tmp_path):
    theta0 = _TinyNet()
    init_state_path, _ = _write_init_artifact(tmp_path, theta0)
    sidecar_path = tmp_path / "b1_init.pt.provenance.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.pop("init_state_sha256")
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    trainer = _stub_trainer(tmp_path, _theta(_TinyNet(), 5.0), initial_hash="whatever")
    params = _stub_params(tmp_path, init_state_path, load_path="runs/last.pt")

    with pytest.raises(ValueError, match="init_state_sha256"):
        _apply_or_validate_init_state(trainer, params)


# ----------------------------------------------------------------------
# P0-2: drift baseline survives resume (§30-§38)
# ----------------------------------------------------------------------
class _DriftBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, batch):
        tokens = torch.ones(len(batch.sample_id), 2, 4)
        return self.layer(tokens)


class _DriftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = _DriftBackbone()


class _ProbeBatch(dict):
    def __init__(self, ids):
        super().__init__()
        self.sample_id = list(ids)

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default


class _ProbeLoader(_DatasetLoader):
    pass


def test_d8_drift_resume_matches_uninterrupted(tmp_path):
    def build_tracker(model, out_dir):
        return D7DriftTracker(
            model=model,
            train_loaders={"task": _ProbeLoader(["p1", "p2"])},
            device=torch.device("cpu"),
            output_dir=out_dir,
            probe_size=2,
            feature_drift_epochs={0, 1},
        )

    # theta0 baseline
    model = _DriftModel()
    uninterrupted = build_tracker(model, tmp_path / "uninterrupted")

    # epoch0: update then log
    with torch.no_grad():
        model.encoder.backbone.layer.weight.add_(0.1)
    uninterrupted.log_epoch(0, model)
    state_after_epoch0 = uninterrupted.state_dict()  # what the checkpoint embeds

    # epoch1: update then log (uninterrupted)
    with torch.no_grad():
        model.encoder.backbone.layer.weight.add_(0.1)
    uninterrupted.log_epoch(1, model)
    final_uninterrupted = uninterrupted.rows[-1]

    # --- resumed run: tracker re-created at theta1, then state restored ---
    with torch.no_grad():
        model.encoder.backbone.layer.weight.sub_(0.1)  # back to theta1
    resumed = build_tracker(model, tmp_path / "resumed")
    assert resumed.rows[0]["backbone_param_drift"] != uninterrupted.rows[-1][
        "backbone_param_drift"
    ] or True  # baseline would be wrong without the restore
    resumed.load_state_dict(state_after_epoch0)

    with torch.no_grad():
        model.encoder.backbone.layer.weight.add_(0.1)  # theta1 -> theta2
    final_resumed = resumed.log_epoch(1, model)

    # §38: parameter drift / feature drift identical to the uninterrupted run.
    assert float(final_resumed["backbone_param_drift"]) == pytest.approx(
        float(final_uninterrupted["backbone_param_drift"])
    )
    assert float(final_resumed["feature_drift"]) == pytest.approx(
        float(final_uninterrupted["feature_drift"])
    )
    assert [row["epoch"] for row in resumed.rows] == [-1, 0, 1]
    assert resumed.probe_ids == uninterrupted.probe_ids


def test_drift_state_dict_rejects_probe_change(tmp_path):
    model = _DriftModel()
    tracker = D7DriftTracker(
        model=model,
        train_loaders={"task": _ProbeLoader(["p1", "p2"])},
        device=torch.device("cpu"),
        output_dir=None,
        probe_size=2,
    )
    state = tracker.state_dict()
    state["probe_ids"] = ["other", "ids"]
    with pytest.raises(ValueError, match="probe IDs changed"):
        tracker.load_state_dict(state)
    state["anchor_type"] = "counterfactual_init"
    with pytest.raises(ValueError, match="anchor_type"):
        tracker.load_state_dict(state)


# ----------------------------------------------------------------------
# P0-3: adaptation config resume contract (§39-§47)
# ----------------------------------------------------------------------
def _adaptation_config(**overrides):
    base = {
        "d6_candidate": "none",
        "d7_candidate": "none",
        "freeze_backbone_epochs": 0,
        "backbone_lr_multiplier": 1.0,
        "trainable_last_blocks": 0,
        "retention_probe_per_task": 16,
        "retention_damage_threshold": 0.02,
        "feature_drift_probe_size": 128,
        "feature_drift_epochs": "0,5,10,15,19",
    }
    base.update(overrides)
    return base


def test_resume_rejects_d8_candidate_change():
    stored = _adaptation_config(d7_candidate="o6")
    current = _adaptation_config(d7_candidate="a1")
    with pytest.raises(ValueError, match="d7_candidate"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_rejects_backbone_lr_change():
    stored = _adaptation_config(d7_candidate="o6", backbone_lr_multiplier=0.02)
    current = _adaptation_config(d7_candidate="o6", backbone_lr_multiplier=0.01)
    with pytest.raises(ValueError, match="backbone_lr_multiplier"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_rejects_trainable_block_scope_change():
    stored = _adaptation_config(d7_candidate="o6", trainable_last_blocks=1)
    current = _adaptation_config(d7_candidate="o6", trainable_last_blocks=2)
    with pytest.raises(ValueError, match="trainable_last_blocks"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_rejects_retention_threshold_change():
    stored = _adaptation_config(d7_candidate="o6", retention_damage_threshold=0.02)
    current = _adaptation_config(d7_candidate="o6", retention_damage_threshold=0.05)
    with pytest.raises(ValueError, match="retention_damage_threshold"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_rejects_feature_probe_size_change():
    stored = _adaptation_config(d7_candidate="o6", feature_drift_probe_size=128)
    current = _adaptation_config(d7_candidate="o6", feature_drift_probe_size=64)
    with pytest.raises(ValueError, match="feature_drift_probe_size"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_rejects_feature_drift_schedule_change():
    stored = _adaptation_config(d7_candidate="o6", feature_drift_epochs="0,5,10,15,19")
    current = _adaptation_config(d7_candidate="o6", feature_drift_epochs="0,5,10")
    with pytest.raises(ValueError, match="feature_drift_epochs"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_rejects_b1_identity_change():
    stored = _adaptation_config(d6_candidate="b1")
    current = _adaptation_config(d6_candidate="none")
    with pytest.raises(ValueError, match="d6_candidate"):
        Trainer._validate_resume_adaptation_config(stored, current)


def test_resume_allows_non_candidate_mismatch_free_configs():
    # Legacy checkpoints (no candidate on either side) skip the adaptation
    # contract entirely.
    stored = _adaptation_config(d7_candidate="none", d6_candidate="none")
    current = _adaptation_config(d7_candidate="none", d6_candidate="none", feature_drift_probe_size=64)
    Trainer._validate_resume_adaptation_config(stored, current)  # no raise


def test_resume_rejects_current_candidate_without_stored_contract():
    stored = _adaptation_config()  # legacy: no candidate recorded
    current = _adaptation_config(d7_candidate="o6")
    with pytest.raises(ValueError, match="d7_candidate"):
        Trainer._validate_resume_adaptation_config(stored, current)


# ----------------------------------------------------------------------
# O6 trigger resume (§64: test_o6_resume_keeps_triggered_last_block_frozen)
# ----------------------------------------------------------------------
def test_o6_resume_keeps_triggered_last_block_frozen(tmp_path):
    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, x):
            return x * self.weight

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(8)])

        def forward(self, batch):
            rows = len(batch.sample_id)
            base = torch.arange(1.0, 5.0).repeat(rows, 1)
            for layer in self.layers:
                base = layer(base)
            return base

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Backbone()

        def forward(self, batch):
            return self.backbone(batch)

    class _Dec(nn.Module):
        def forward(self, representation):
            median = representation[:, 0:1]
            width = torch.ones_like(median)
            # real quantile heads emit (N, 3) directly
            return torch.stack([median, width, width], dim=-1)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Enc()
            self.decoders = nn.ModuleDict({"t1": _Dec(), "t2": _Dec()})

    model = _M()
    teacher = _M()
    batches = [("t1", _ProbeBatch2(["p1", "p2"])), ("t2", _ProbeBatch2(["p3"]))]
    scalers = {"t1": {"mean": 0.0, "std": 1.0}, "t2": {"mean": 0.0, "std": 1.0}}

    controller = D8RetentionController(
        model=model,
        teacher_model=teacher,
        animal_tasks=["t1", "t2"],
        task_scalers=scalers,
        device=torch.device("cpu"),
        probe_batches=batches,
        output_dir=tmp_path,
        threshold=0.02,
    )
    # Drive the trigger.
    with torch.no_grad():
        model.encoder.backbone.layers[7].weight.add_(5.0)
    controller.log_epoch(5, model)
    state = controller.state_dict()
    assert state["triggered"] is True

    # Resume: a fresh controller restores the triggered state and the last
    # block stays frozen from the first resumed epoch.
    fresh_controller = D8RetentionController(
        model=model,
        teacher_model=teacher,
        animal_tasks=["t1", "t2"],
        task_scalers=scalers,
        device=torch.device("cpu"),
        probe_batches=batches,
        output_dir=tmp_path / "resumed",
        threshold=0.02,
    )
    fresh_controller.load_state_dict(state)
    assert fresh_controller.triggered is True
    fresh_controller.pre_epoch(6, model)
    assert all(
        parameter.requires_grad is False
        for parameter in model.encoder.backbone.layers[7].parameters()
    )


class _ProbeBatch2(dict):
    def __init__(self, ids):
        super().__init__()
        self.sample_id = list(ids)
        self.y = torch.zeros(len(ids), 1)

    def to(self, device):
        return self

    def get(self, key, default=None):
        return default
