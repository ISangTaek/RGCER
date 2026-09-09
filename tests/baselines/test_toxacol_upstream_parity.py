from __future__ import annotations

import importlib.util

import numpy as np
import torch

from baselines.models.toxacol import ToxACoLNet, endpoint_feature_matrix


def _load_upstream(path):
    source = path / "models" / "CorrelationNet.py"
    spec = importlib.util.spec_from_file_location("locked_toxacol_correlation_net", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_weights(local, upstream):
    with torch.no_grad():
        upstream.tail_w.copy_(local.tail_weight)
        for index in range(4):
            source_fcl = local.dnn[index]
            target_fcl = upstream.dnn[index].layers
            target_fcl.Linear.weight.copy_(source_fcl.linear.weight)
            target_fcl.Linear.bias.copy_(source_fcl.linear.bias)
            target_fcl.BatchNorm.weight.copy_(source_fcl.batch_norm.weight)
            target_fcl.BatchNorm.bias.copy_(source_fcl.batch_norm.bias)
            target_fcl.BatchNorm.running_mean.copy_(source_fcl.batch_norm.running_mean)
            target_fcl.BatchNorm.running_var.copy_(source_fcl.batch_norm.running_var)
            upstream.gcn[index].W.copy_(local.gcn[index].weight)
            upstream.correlation_net[index].W.copy_(local.correlation[index].weight)


def test_toxacol_forward_and_gradient_match_locked_upstream(toxacol_source):
    torch.manual_seed(7)
    adjacency = np.eye(59, dtype=np.float32)
    endpoint, _ = endpoint_feature_matrix()
    local = ToxACoLNet(adjacency, endpoint, dropout=0.1)
    module = _load_upstream(toxacol_source)
    module.CorrelationNet.init_adjacency_matrix = lambda self, file_read=None: torch.as_tensor(adjacency)
    module.CorrelationNet.init_gcn_input = lambda self, file_read=None: torch.as_tensor(endpoint)
    upstream = module.CorrelationNet({
        "in_features_dnn": 1024,
        "in_features_gcn": 26,
        "out_features": [768, 512, 384, 64],
        "num_layers": 4,
        "Dropout_p": 0.1,
        "num_task": 59,
    })
    _copy_weights(local, upstream)
    local.eval()
    upstream.eval()
    left = torch.randn(4, 1024, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    left_output = local(left)
    right_output = upstream(right)
    torch.testing.assert_close(left_output, right_output, rtol=1e-6, atol=1e-6)
    left_output.square().mean().backward()
    right_output.square().mean().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(local.tail_weight.grad, upstream.tail_w.grad, rtol=1e-5, atol=1e-6)


def test_toxacol_initialization_ranges_match_upstream_semantics():
    endpoint, _ = endpoint_feature_matrix()
    model = ToxACoLNet(np.eye(59, dtype=np.float32), endpoint)
    for layer in list(model.gcn) + list(model.correlation):
        assert float(layer.weight.detach().min()) >= -0.05
        assert float(layer.weight.detach().max()) <= 0.05
    assert float(model.tail_weight.detach().min()) >= 0.0
    assert float(model.tail_weight.detach().max()) <= 0.1
