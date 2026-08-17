from types import SimpleNamespace

import pytest
import torch
from torch import nn

from loss import MSELoss
from trainer import Trainer
from weighting.EW import EW


class _DummyArchitecture(nn.Module):
    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        super().__init__()
        del task_name, encoder_class, decoders, device, args, kwargs
        self.encoder = nn.Identity()
        self.projection = nn.Linear(4, 1)


def _args(tmp_path, *, use_null=True):
    return SimpleNamespace(
        seed=42,
        gpu_id="cpu",
        prediction_mode="point",
        conformal_alpha=0.10,
        min_calibration_size=1,
        mode="train",
        save_path=str(tmp_path),
        load_path=None,
        preprocessed_data_dir=None,
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
        rgcer_use_null_route=use_null,
        rgcer_use_film=True,
        rgcer_use_adapter=True,
        rgcer_use_base_aux_loss=True,
        rgcer_fallback_space="prediction",
        rgcer_transfer_mechanism="endpoint_router",
        exclude_target_from_sources=True,
        use_factorized_prompt=False,
        lower_quantile=0.05,
        upper_quantile=0.95,
        ckpt_name="model",
    )


def _trainer(tmp_path, *, use_null=True, load_path=None):
    args = _args(tmp_path, use_null=use_null)
    args.load_path = load_path
    task_dict = {"task": {"metrics": ["RMSE"], "loss_fn": MSELoss(), "weight": [-1, 1]}}
    return Trainer(
        task_dict=task_dict,
        weighting=EW,
        architecture=_DummyArchitecture,
        encoder_class=nn.Identity,
        decoders=nn.ModuleDict(),
        optim_param={"optim": "adamw", "lr": 1e-3, "weight_decay": 0.0},
        args=args,
        save_path=tmp_path,
        load_path=load_path,
    )


def test_checkpoint_v4_records_and_rejects_changed_rgcer_flags(tmp_path):
    source = _trainer(tmp_path / "source", use_null=True)
    checkpoint_path = source._save_checkpoint(0, "model_best.pt")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == 4
    assert payload["rgcer_config"]["rgcer_use_null_route"] is True
    assert payload["rgcer_config"]["rgcer_fallback_space"] == "prediction"
    assert payload["rgcer_config"]["rgcer_transfer_mechanism"] == "endpoint_router"

    with pytest.raises(ValueError, match="rgcer_use_null_route"):
        _trainer(tmp_path / "changed", use_null=False, load_path=checkpoint_path)
