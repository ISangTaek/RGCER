from types import SimpleNamespace

import torch
from torch import nn

from architecture.Graphormer_prompt import Encoder, Graphormer_prompt
from dataset import DataCollator
from preprocess_data import get_graph_data_from_smiles


def test_prompt_forward_is_stateless_and_task_conditioned():
    torch.manual_seed(11)
    tasks = ["man", "women", "human"]
    args = SimpleNamespace(a_heads=2, a_layers=1, hidden_dim=16, mid_dim=32, t_heads=2)
    decoders = nn.ModuleDict({task: nn.Linear(16, 1) for task in tasks})
    model = Graphormer_prompt(tasks, Encoder, decoders, torch.device("cpu"), args).eval()
    batch = DataCollator()([get_graph_data_from_smiles("CCO", 1.0)])

    first = model(batch, task_name="man", mode="test")["man"]
    other = model(batch, task_name="women", mode="test")["women"]
    repeated = model(batch, task_name="man", mode="test")["man"]
    assert first.shape == other.shape == repeated.shape == (1, 1)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other)

    model.train()
    output = model(batch, task_name="human", mode="train")["human"].sum()
    output.backward()
    assert model.encoder.task_conditioner.prompts.grad is not None
    assert model.encoder.task_conditioner.prompts.grad.abs().sum() > 0
