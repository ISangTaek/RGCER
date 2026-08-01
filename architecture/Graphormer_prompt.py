"""Relation-aware, stateless Prompt Graphormer."""

from __future__ import annotations

import torch
from torch import nn

from architecture.abstract_arch import AbsArchitecture
from architecture.graphormer_backbone import MolecularGraphormerBackbone
from architecture.task_prompt import RelationAwareTaskConditioner


class Encoder(nn.Module):
    """Shared chemical backbone plus task-space prompt conditioning."""

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
        prompt_layers=1,
        prompt_ffn_dim=None,
        adapter_ratio=0.25,
        prompt_gate_init=-2.0,
    ):
        super().__init__()
        del atoms_input_dropout_rate, atoms_reshape_dim, task_layers, device
        if atoms_readout_dim != moles_hidden_dim:
            raise ValueError("atoms_readout_dim must equal moles_hidden_dim for prompt conditioning")
        self.task_name = list(task_name)
        self.backbone = MolecularGraphormerBackbone(
            hidden_dim=atoms_hidden_dim,
            num_heads=atoms_num_heads,
            num_layers=atoms_embedders_num,
            ffn_dim=atoms_ffn_dim,
            dropout=max(float(atoms_dropout_rate), float(atoms_attention_dropout_rate)),
            spatial_pos_max_clip=20,
        )
        self.task_conditioner = RelationAwareTaskConditioner(
            self.task_name,
            hidden_dim=moles_hidden_dim,
            prompt_heads=task_num_heads,
            prompt_layers=prompt_layers,
            prompt_ffn_dim=prompt_ffn_dim,
            prompt_dropout=task_dropout_rate,
            adapter_ratio=adapter_ratio,
            gate_init=prompt_gate_init,
        )

    def encode_backbone(self, batch):
        return self.backbone(batch)

    def forward(self, batch, task_name=None, return_all_tasks=False, mode=None):
        del mode
        representation = self.encode_backbone(batch)
        if return_all_tasks:
            return self.task_conditioner.condition_all(representation)
        if task_name is None:
            raise ValueError("task_name is required unless return_all_tasks=True")
        return self.task_conditioner(representation, task_name)


class Graphormer_prompt(AbsArchitecture):
    """Shared Graphormer with relation-aware task-specific decoders."""

    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        super().__init__(task_name, encoder_class, decoders, device, **kwargs)
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
            task_layers=getattr(args, "t_layers", 1),
            device=device,
            task_num_heads=getattr(args, "prompt_heads", getattr(args, "t_heads", 4)),
            task_dropout_rate=getattr(args, "prompt_dropout", 0.1),
            prompt_layers=getattr(args, "prompt_layers", 1),
            prompt_ffn_dim=getattr(args, "prompt_ffn_dim", args.hidden_dim * 2),
            adapter_ratio=getattr(args, "adapter_ratio", 0.25),
            prompt_gate_init=getattr(args, "prompt_gate_init", -2.0),
        )

    def forward(self, inputs, task_name=None, return_all_tasks=False, mode=None):
        representations = self.encoder(
            inputs,
            task_name=task_name,
            return_all_tasks=return_all_tasks,
            mode=mode,
        )
        if return_all_tasks:
            return {
                task: self.decoders[task](representation)
                for task, representation in representations.items()
            }
        if task_name not in self.decoders:
            raise KeyError(f"Unknown task: {task_name}")
        return {task_name: self.decoders[task_name](representations)}
