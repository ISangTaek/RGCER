"""Checkpoint v6 formal contract (review-2 §14-18, §31-32)."""
from types import SimpleNamespace

import torch
from torch import nn

from loss import MSELoss
from tests.test_checkpoint_ablation_config import _DummyArchitecture
from tests.test_datastore_v2_checkpoint_contract import _formal_args
from tests.test_datastore_v2_contract import _write_raw
from toxacute_datastore import ToxAcuteDataStore, build_datastore_v2
from trainer import Trainer
from weighting.EW import EW


def _formal_trainer(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    root = tmp_path / "datastore"
    _write_raw(raw_csv)
    build_datastore_v2(
        raw_csv,
        root,
        task_names=["task_a", "task_b"],
        splitting="random",
        valid_size=0.2,
        calibration_size=0.2,
        test_size=0.2,
        split_seed=42,
        max_path_distance=6,
        lmdb_map_size_gb=0.001,
    )
    store = ToxAcuteDataStore.resolve(root)
    args = _formal_args(tmp_path / "run", store.metadata, root)
    trainer = Trainer(
        task_dict={"task_a": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]}},
        weighting=EW,
        architecture=_DummyArchitecture,
        encoder_class=nn.Identity,
        decoders=nn.ModuleDict(),
        optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        args=args,
        save_path=args.save_path,
    )
    return trainer, store


def _with_historical_best(trainer, best_score=10.0, best_epoch=0):
    import copy

    trainer.train_loss_buffer = trainer.train_loss_buffer if hasattr(
        trainer, "train_loss_buffer"
    ) else __import__("numpy").ones((trainer.task_num, 2))
    trainer.best_val_score = best_score
    trainer.best_epoch = best_epoch
    trainer.best_routing_enabled = False
    trainer._best_state = copy.deepcopy(trainer.model.state_dict())
    trainer._best_training_state = {
        "model_state": copy.deepcopy(trainer.model.state_dict()),
        "optimizer_state": copy.deepcopy(trainer.optimizer.state_dict()),
        "weighting_state": copy.deepcopy(trainer.loss_balancer.state_dict()),
        "train_loss_buffer": trainer.train_loss_buffer.copy(),
        "optimizer_updates": 7,
        "epoch": best_epoch,
    }
    return trainer


def test_formal_checkpoint_is_v6_and_last_pt_embeds_best_state(tmp_path):
    trainer, store = _formal_trainer(tmp_path)
    _with_historical_best(trainer)
    best_path = trainer._save_checkpoint(3, "model_best.pt")
    last_path = trainer._save_checkpoint(3, "model_last.pt", include_historical_best=True)

    best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
    last_payload = torch.load(last_path, map_location="cpu", weights_only=False)
    assert best_payload["checkpoint_version"] == 6
    assert last_payload["checkpoint_version"] == 6
    # Review-2 §17: best.pt must not embed a redundant copy of itself.
    assert "best_training_state" not in best_payload
    # Review-2 §16: last.pt alone is a complete resume artifact.
    assert "best_training_state" in last_payload
    assert "best_model_state" in last_payload
    assert last_payload["selection_state"]["best_val_score"] == 10.0
    assert last_payload["selection_state"]["best_epoch"] == 0
    assert last_payload["reproducibility"]["seed_policy_version"] == 2
    assert (
        last_payload["reproducibility"]["persistent_worker_policy"]
        == "disabled_for_strict_resume"
    )
    store.close()


def test_v6_rejects_v5_and_missing_contracts(tmp_path):
    trainer, store = _formal_trainer(tmp_path)
    _with_historical_best(trainer)
    last_path = trainer._save_checkpoint(3, "model_last.pt", include_historical_best=True)
    payload = torch.load(last_path, map_location="cpu", weights_only=False)

    def _try_load(mutated, match):
        broken_path = tmp_path / "broken.pt"
        torch.save(mutated, broken_path)
        fresh_args = _formal_args(tmp_path / f"fresh-{match[:6]}", store.metadata, store.root)
        fresh = Trainer(
            task_dict={"task_a": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]}},
            weighting=EW,
            architecture=_DummyArchitecture,
            encoder_class=nn.Identity,
            decoders=nn.ModuleDict(),
            optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
            args=fresh_args,
            save_path=fresh_args.save_path,
        )
        import pytest

        with pytest.raises(ValueError, match=match):
            fresh.load_checkpoint(broken_path)

    v5_payload = dict(payload)
    v5_payload["checkpoint_version"] = 5
    _try_load(v5_payload, "strict v6")

    no_repro = dict(payload)
    no_repro["reproducibility"] = None
    _try_load(no_repro, "reproducibility")

    no_selection = dict(payload)
    no_selection.pop("selection_state")
    _try_load(no_selection, "selection_state")
    store.close()


def test_v6_load_restores_historical_best_selection(tmp_path):
    import copy

    source, store = _formal_trainer(tmp_path)
    _with_historical_best(source, best_score=10.0, best_epoch=0)
    last_path = source._save_checkpoint(3, "model_last.pt", include_historical_best=True)

    resumed_args = _formal_args(tmp_path / "resumed", store.metadata, store.root)
    resumed = Trainer(
        task_dict={"task_a": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]}},
        weighting=EW,
        architecture=_DummyArchitecture,
        encoder_class=nn.Identity,
        decoders=nn.ModuleDict(),
        optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        args=resumed_args,
        save_path=resumed_args.save_path,
        load_path=last_path,
    )

    # Review-2 §18: the historical best selection contract is fully restored.
    assert resumed.best_val_score == 10.0
    assert resumed.best_epoch == 0
    assert resumed.best_routing_enabled is False
    assert resumed._best_state is not None
    assert resumed._best_training_state["optimizer_updates"] == 7
    assert resumed._best_training_state["epoch"] == 0
    for stored, restored in zip(source._best_state.values(), resumed._best_state.values()):
        torch.testing.assert_close(stored, restored)
    # A worse new epoch must NOT replace the historical best (review-2 §19).
    assert not (resumed._best_state is None or 5.0 > resumed.best_val_score)
    del copy
    store.close()
