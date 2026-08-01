"""Stateless task-conditioned Graphormer architecture.

Task relations are modelled only in the prompt bank.  Molecular examples from
different tasks are never paired or stacked, so a forward call is independent
of the order and history of previous calls.
"""

from __future__ import annotations

import torch
from torch import nn

from architecture.abstract_arch import AbsArchitecture
from architecture.graphormer_backbone import MolecularGraphormerBackbone


class TaskPromptConditioner(nn.Module):
    """Prompt self-attention followed by FiLM and a small low-rank adapter."""

    def __init__(self, num_tasks, hidden_dim, num_heads=2, dropout=0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by prompt attention heads")
        self.prompts = nn.Parameter(torch.empty(num_tasks, hidden_dim))
        nn.init.normal_(self.prompts, std=0.02)
        self.relation_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.relation_norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, hidden_dim * 3)
        bottleneck = max(hidden_dim // 4, 1)
        self.adapter_down = nn.Linear(hidden_dim, bottleneck)
        self.adapter_up = nn.Linear(bottleneck, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, representation, task_index):
        prompt_tokens = self.prompts.unsqueeze(0)
        related, _ = self.relation_attention(prompt_tokens, prompt_tokens, prompt_tokens)
        related = self.relation_norm(prompt_tokens + self.dropout(related))
        prompt = related[:, task_index, :].expand(representation.size(0), -1)
        gamma, beta, gate = self.film(prompt).chunk(3, dim=-1)
        conditioned = (1.0 + torch.tanh(gamma)) * representation + beta
        conditioned = self.output_norm(conditioned)
        adapter = self.adapter_up(torch.nn.functional.gelu(self.adapter_down(conditioned)))
        return conditioned + torch.sigmoid(gate) * adapter


class Encoder(nn.Module):
    def __init__(
        self,
        atoms_num_heads,
        task_name,
        atoms_embedders_num,
        atoms_hidden_dim,
        atoms_dropout_rate,
        atoms_input_dropout_rate,
        atoms_ffn_dim,
        atoms_reshape_dim,
        atoms_attention_dropout_rate,
        atoms_readout_dim,
        moles_hidden_dim,
        task_layers,
        device,
        task_num_heads,
        task_dropout_rate,
    ):
        super().__init__()
        del atoms_input_dropout_rate, atoms_reshape_dim, task_layers, device
        self.task_name = tuple(task_name)
        self.backbone = MolecularGraphormerBackbone(
            hidden_dim=atoms_hidden_dim,
            num_heads=atoms_num_heads,
            num_layers=atoms_embedders_num,
            ffn_dim=atoms_ffn_dim,
            dropout=max(float(atoms_dropout_rate), float(atoms_attention_dropout_rate)),
            spatial_pos_max_clip=20,
        )
        self.backbone_readout = (
            nn.Identity()
            if atoms_readout_dim == moles_hidden_dim
            else nn.Linear(atoms_readout_dim, moles_hidden_dim)
        )
        self.task_conditioner = TaskPromptConditioner(
            len(self.task_name), moles_hidden_dim, num_heads=task_num_heads, dropout=task_dropout_rate
        )

    def forward(self, batch, task_name=None, mode=None):
        del mode
        if task_name not in self.task_name:
            raise KeyError(f"Unknown task: {task_name}")
        representation = self.backbone_readout(self.backbone(batch))
        return self.task_conditioner(representation, self.task_name.index(task_name))


class Graphormer_prompt(AbsArchitecture):
    """Relation-aware prompt-conditioned Graphormer with task-specific heads."""

    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        super().__init__(task_name, encoder_class, decoders, device, **kwargs)
        prompt_heads = getattr(args, "t_heads", 2)
        # Multi-head attention is only over the task prompt bank.  The molecular
        # batch remains a single-task batch throughout the complete forward pass.
        self.encoder = encoder_class(
            atoms_num_heads=args.a_heads,
            task_name=task_name,
            atoms_embedders_num=args.a_layers,
            atoms_hidden_dim=args.hidden_dim,
            atoms_dropout_rate=0.1,
            atoms_input_dropout_rate=0.0,
            atoms_ffn_dim=args.mid_dim,
            atoms_reshape_dim=args.hidden_dim,
            atoms_attention_dropout_rate=0.1,
            atoms_readout_dim=args.hidden_dim,
            moles_hidden_dim=args.hidden_dim,
            task_layers=1,
            device=device,
            task_num_heads=prompt_heads,
            task_dropout_rate=0.1,
        )

    def forward(self, inputs, task_name=None, mode=None):
        if task_name is None:
            raise ValueError("task_name is required for task-conditioned forward")
        representation = self.encoder(inputs, task_name=task_name, mode=mode)
        return {task_name: self.decoders[task_name](representation)}
