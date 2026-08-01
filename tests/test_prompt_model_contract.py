from types import SimpleNamespace

import torch
from torch import nn

from architecture.Graphormer_prompt import Encoder, Graphormer_prompt
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
        prompt_layers=1,
        prompt_ffn_dim=32,
        prompt_dropout=0.0,
        adapter_ratio=0.25,
        prompt_gate_init=-2.0,
    )
    decoders = nn.ModuleDict({task: nn.Linear(16, 1) for task in tasks})
    return Graphormer_prompt(tasks, Encoder, decoders, torch.device("cpu"), args)


def test_single_and_all_task_forward_contract():
    model = build_model().eval()
    batch = DataCollator()([get_graph_data_from_smiles("CCO", 1.0)])
    single = model(batch, task_name="task_a")
    all_tasks = model(batch, return_all_tasks=True)
    assert list(single) == ["task_a"]
    assert set(all_tasks) == {"task_a", "task_b", "task_c"}
    assert all(value.shape == (1, 1) for value in all_tasks.values())


def test_no_cross_batch_state_and_different_batch_sizes():
    model = build_model().eval()
    collator = DataCollator()
    batch_a = collator([get_graph_data_from_smiles("CC", 1.0) for _ in range(3)])
    batch_b = collator([get_graph_data_from_smiles("CCO", 1.0) for _ in range(5)])
    first = model(batch_a, task_name="task_a")["task_a"]
    _ = model(batch_b, task_name="task_b")
    second = model(batch_a, task_name="task_a")["task_a"]
    assert first.shape == (3, 1)
    assert torch.allclose(first, second)
