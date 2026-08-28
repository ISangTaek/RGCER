"""Graphormer with stateless molecule-adaptive task Prompt routing."""

from __future__ import annotations

from typing import Optional, Sequence

from torch import nn

from architecture.abstract_arch import AbsArchitecture
from architecture.graphormer_backbone import MolecularGraphormerBackbone
from architecture.molecule_adaptive_prompt import MoleculeAdaptiveTaskConditioner
from architecture.toxacute_tasks import parse_toxacute_task_name


def _infer_factorized_prompt(task_names: Sequence[str], requested: Optional[bool]) -> bool:
    """Use factorized endpoint metadata for ToxAcute, residual-only otherwise."""

    if requested is not None:
        return bool(requested)
    try:
        for task_name in task_names:
            parse_toxacute_task_name(task_name)
    except ValueError:
        return False
    return True


class Encoder(nn.Module):
    """Shared chemical backbone followed by task conditioning."""

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
        prompt_layers=0,
        prompt_ffn_dim=None,
        adapter_ratio=0.25,
        prompt_gate_init=-2.0,
        use_factorized_prompt=None,
        task_residual_scale=0.1,
        router_mode="dynamic",
        router_dim=None,
        router_top_k=8,
        router_temperature=1.0,
        exclude_target_from_sources=True,
        edge_bias_mode="path",
        spatial_pos_max_clip=20,
        metadata_overrides=None,
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
            spatial_pos_max_clip=spatial_pos_max_clip,
            edge_bias_mode=edge_bias_mode,
        )
        self.task_conditioner = MoleculeAdaptiveTaskConditioner(
            self.task_name,
            hidden_dim=moles_hidden_dim,
            use_factorized_prompt=_infer_factorized_prompt(self.task_name, use_factorized_prompt),
            task_residual_scale=task_residual_scale,
            prompt_heads=task_num_heads,
            prompt_layers=prompt_layers,
            prompt_ffn_dim=prompt_ffn_dim,
            prompt_dropout=task_dropout_rate,
            router_mode=router_mode,
            router_dim=router_dim,
            router_top_k=router_top_k,
            router_temperature=router_temperature,
            exclude_target_from_sources=exclude_target_from_sources,
            adapter_ratio=adapter_ratio,
            gate_init=prompt_gate_init,
            metadata_overrides=metadata_overrides,
        )

    def encode_backbone(self, batch):
        return self.backbone(batch)

    def forward(
        self,
        batch,
        task_name=None,
        return_all_tasks=False,
        return_aux=False,
        source_mask=None,
        mode=None,
    ):
        del mode  # Retained only for compatibility with older callers.
        representation = self.encode_backbone(batch)
        if return_all_tasks:
            return self.task_conditioner.condition_all(
                representation,
                return_aux=return_aux,
                source_mask=source_mask,
            )
        if task_name is None:
            raise ValueError("task_name is required unless return_all_tasks=True")
        return self.task_conditioner(
            representation,
            task_name,
            return_aux=return_aux,
            source_mask=source_mask,
        )


class Graphormer_prompt(AbsArchitecture):
    """Shared Graphormer with independent decoders and adaptive routing."""

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
            prompt_layers=getattr(args, "prompt_layers", 0),
            prompt_ffn_dim=getattr(args, "prompt_ffn_dim", args.hidden_dim * 2),
            adapter_ratio=getattr(args, "adapter_ratio", 0.25),
            prompt_gate_init=getattr(args, "prompt_gate_init", -2.0),
            use_factorized_prompt=getattr(args, "use_factorized_prompt", None),
            task_residual_scale=getattr(args, "task_residual_scale", 0.1),
            router_mode=getattr(args, "router_mode", "dynamic"),
            router_dim=getattr(args, "router_dim", None),
            router_top_k=getattr(args, "router_top_k", 8),
            router_temperature=getattr(args, "router_temperature", 1.0),
            exclude_target_from_sources=getattr(args, "exclude_target_from_sources", True),
            edge_bias_mode=getattr(args, "edge_bias_mode", "path"),
            spatial_pos_max_clip=kwargs.get(
                "spatial_pos_max_clip", getattr(args, "spatial_pos_clip", 20)
            ),
            metadata_overrides=getattr(args, "auxiliary_metadata_overrides", None),
        )

    def forward(
        self,
        inputs,
        task_name=None,
        return_all_tasks=False,
        return_aux=False,
        source_mask=None,
        mode=None,
    ):
        conditioned = self.encoder(
            inputs,
            task_name=task_name,
            return_all_tasks=return_all_tasks,
            return_aux=return_aux,
            source_mask=source_mask,
            mode=mode,
        )
        if return_all_tasks:
            if return_aux:
                representations, diagnostics = conditioned
            else:
                representations, diagnostics = conditioned, None
            predictions = {
                task: self.decoders[task](representations[task])
                for task in self.task_name
            }
            return (predictions, diagnostics) if return_aux else predictions

        if task_name is None:
            raise ValueError("task_name is required unless return_all_tasks=True")
        if task_name not in self.decoders:
            raise KeyError(f"Unknown task: {task_name}")
        if return_aux:
            representation, diagnostics = conditioned
        else:
            representation, diagnostics = conditioned, None
        predictions = {task_name: self.decoders[task_name](representation)}
        return (predictions, diagnostics) if return_aux else predictions


__all__ = ["Encoder", "Graphormer_prompt"]
