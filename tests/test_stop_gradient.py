from unittest.mock import patch

import torch

from tests.test_rgcer_contract import build_model
from dataset import DataCollator
from preprocess_data import get_graph_data_from_smiles


def test_source_heads_do_not_receive_response_profile_gradient():
    model, tasks = build_model()
    model.train()
    batch = DataCollator()([get_graph_data_from_smiles("CCO", 1.0)])
    predictions = model(batch, task_name=tasks[0])
    predictions[tasks[0]].sum().backward()
    target_head = model.decoders[tasks[0]]
    assert target_head.network[-1].weight.grad is not None
    router_grad = model.encoder.task_conditioner.router.query_projection[1].weight.grad
    assert router_grad is not None
    assert torch.isfinite(router_grad).all()
    adapter_grad = model.encoder.task_conditioner.adapter.adapter_up.weight.grad
    assert adapter_grad is not None
    assert torch.isfinite(adapter_grad).all()
    for task in tasks[1:]:
        gradient = model.decoders[task].network[-1].weight.grad
        assert gradient is None or torch.allclose(gradient, torch.zeros_like(gradient))


def test_backbone_is_detached_on_route_only_path():
    model, tasks = build_model()
    model.train()
    batch = DataCollator()([get_graph_data_from_smiles("CCO", 1.0)])
    captured = {}

    def capture(_module, _inputs, output):
        output.retain_grad()
        captured["h"] = output

    handle = model.encoder.backbone.register_forward_hook(capture)
    router = model.encoder.task_conditioner.router
    original = router.forward

    def force_route_only(*args, **kwargs):
        context, source, joint, null, entropy = original(*args, **kwargs)
        return context, source, joint, torch.zeros_like(null), entropy

    try:
        with patch.object(router, "forward", side_effect=force_route_only):
            model(batch, task_name=tasks[0])[tasks[0]].sum().backward()
    finally:
        handle.remove()
    assert captured["h"].grad is None or torch.allclose(captured["h"].grad, torch.zeros_like(captured["h"].grad))
