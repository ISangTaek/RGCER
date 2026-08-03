from types import SimpleNamespace

import torch
from torch import nn

from architecture.Graphormer_rgcer import Encoder, Graphormer_rgcer
from architecture.prediction_heads import TaskPredictionHead
from dataset import DataCollator
from preprocess_data import get_graph_data_from_smiles


def build_model():
    tasks = ["task_a", "task_b", "task_c"]
    args = SimpleNamespace(
        a_heads=2,
        a_layers=1,
        hidden_dim=16,
        mid_dim=32,
        t_layers=1,
        prompt_heads=4,
        prompt_dropout=0.0,
        adapter_ratio=0.25,
        use_factorized_prompt=False,
        task_residual_scale=0.1,
        router_dim=12,
        router_top_k=2,
        router_temperature=1.0,
        exclude_target_from_sources=True,
        response_hidden_dim=12,
        edge_bias_mode="path",
        prediction_mode="quantile",
        head_hidden_dim=8,
        head_dropout=0.0,
    )
    heads = nn.ModuleDict({task: TaskPredictionHead(16, "quantile", 8, 0.0) for task in tasks})
    return Graphormer_rgcer(tasks, Encoder, heads, torch.device("cpu"), args), tasks


def test_rgcer_single_all_and_source_mask_contract():
    model, tasks = build_model()
    model.eval()
    batch = DataCollator()([
        get_graph_data_from_smiles("CCO", 1.0),
        get_graph_data_from_smiles("CC", 2.0),
    ])
    predictions, diagnostics = model(batch, task_name="task_a", return_aux=True)
    assert predictions["task_a"].shape == (2, 3)
    assert diagnostics["response_profile"].shape == (2, 3, 1)
    assert diagnostics["source_weights"].shape == (2, 3)
    assert diagnostics["joint_source_weights"].shape == (2, 3)
    assert torch.all((diagnostics["null_weight"] >= 0) & (diagnostics["null_weight"] <= 1))
    all_predictions, all_diagnostics = model(batch, return_all_tasks=True, return_aux=True)
    assert list(all_predictions) == tasks
    assert list(all_diagnostics) == tasks
    mask = torch.zeros(2, 3, dtype=torch.bool)
    _, masked = model(batch, task_name="task_a", return_aux=True, source_mask=mask)
    assert torch.allclose(masked["null_weight"], torch.ones(2, 1))
    assert torch.allclose(masked["final_representation"], masked["base_representation"])
    assert torch.allclose(masked["final_raw"], masked["base_raw"])
