"""Response-Guided Conservative Endpoint Routing Graphormer (RGCER)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from architecture.abstract_arch import AbsArchitecture
from architecture.graphormer_backbone import MolecularGraphormerBackbone
from architecture.prediction_heads import (
    blend_decoded_predictions,
    decode_prediction,
    encode_decoded_prediction,
    point_from_raw,
)
from architecture.response_guided_router import RGCERTaskConditioner
from architecture.toxacute_tasks import parse_toxacute_task_name


def _infer_factorized_prompt(task_names: Sequence[str], requested: Optional[bool]) -> bool:
    if requested is not None:
        return bool(requested)
    try:
        for task_name in task_names:
            parse_toxacute_task_name(task_name)
    except ValueError:
        return False
    return True


class Encoder(nn.Module):
    """Backbone plus response-guided conditioner; no Prompt self-attention."""

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
        task_num_heads=None,
        task_dropout_rate=0.1,
        prompt_layers=0,
        prompt_ffn_dim=None,
        adapter_ratio=0.25,
        prompt_gate_init=None,
        use_factorized_prompt=None,
        task_residual_scale=0.1,
        router_dim=None,
        router_top_k=8,
        router_temperature=1.0,
        exclude_target_from_sources=True,
        response_hidden_dim=None,
        edge_bias_mode="path",
        use_source_response=True,
        use_target_response=True,
        use_molecule_query=True,
        use_sparse_routing=True,
        use_null_route=True,
        use_film=True,
        use_adapter=True,
        transfer_mechanism="endpoint_router",
        spatial_pos_max_clip=20,
        metadata_overrides=None,
    ):
        super().__init__()
        del atoms_input_dropout_rate, atoms_reshape_dim, task_layers, device
        del task_num_heads, prompt_layers, prompt_ffn_dim, prompt_gate_init
        if atoms_readout_dim != moles_hidden_dim:
            raise ValueError("atoms_readout_dim must equal moles_hidden_dim")
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
        self.task_conditioner = RGCERTaskConditioner(
            task_names=self.task_name,
            hidden_dim=moles_hidden_dim,
            use_factorized_prompt=_infer_factorized_prompt(self.task_name, use_factorized_prompt),
            task_residual_scale=task_residual_scale,
            metadata_overrides=metadata_overrides,
            router_dim=router_dim,
            router_top_k=router_top_k,
            router_temperature=router_temperature,
            exclude_target_from_sources=exclude_target_from_sources,
            adapter_ratio=adapter_ratio,
            dropout=task_dropout_rate,
            response_hidden_dim=response_hidden_dim,
            use_source_response=use_source_response,
            use_target_response=use_target_response,
            use_molecule_query=use_molecule_query,
            use_sparse_routing=use_sparse_routing,
            use_null_route=use_null_route,
            use_film=use_film,
            use_adapter=use_adapter,
            transfer_mechanism=transfer_mechanism,
        )

    def encode_backbone(self, batch):
        return self.backbone(batch)


class Graphormer_rgcer(AbsArchitecture):
    """RGCER model with prediction-space HPS fallback and shared task heads."""

    is_rgcer = True

    def __init__(self, task_name, encoder_class, decoders, device, args, **kwargs):
        super().__init__(task_name, encoder_class, decoders, device, **kwargs)
        self.prediction_mode = getattr(args, "prediction_mode", "quantile")
        self.rgcer_fallback_space = getattr(args, "rgcer_fallback_space", "prediction")
        if self.rgcer_fallback_space not in {"prediction", "representation"}:
            raise ValueError("rgcer_fallback_space must be 'prediction' or 'representation'")
        self.rgcer_transfer_mechanism = getattr(args, "rgcer_transfer_mechanism", "endpoint_router")
        if self.rgcer_transfer_mechanism not in {"endpoint_router", "response_stacking", "target_only"}:
            raise ValueError("Unsupported RGCER transfer mechanism")
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
            task_num_heads=getattr(args, "prompt_heads", None),
            task_dropout_rate=getattr(args, "prompt_dropout", 0.1),
            prompt_layers=0,
            adapter_ratio=getattr(args, "adapter_ratio", 0.25),
            use_factorized_prompt=getattr(args, "use_factorized_prompt", None),
            task_residual_scale=getattr(args, "task_residual_scale", 0.1),
            router_dim=getattr(args, "router_dim", None),
            router_top_k=getattr(args, "router_top_k", 8),
            router_temperature=getattr(args, "router_temperature", 1.0),
            exclude_target_from_sources=getattr(args, "exclude_target_from_sources", True),
            response_hidden_dim=getattr(args, "response_hidden_dim", None),
            edge_bias_mode=getattr(args, "edge_bias_mode", "path"),
            use_source_response=getattr(args, "rgcer_use_source_response", True),
            use_target_response=getattr(args, "rgcer_use_target_response", True),
            use_molecule_query=getattr(args, "rgcer_use_molecule_query", True),
            use_sparse_routing=getattr(args, "rgcer_use_sparse_routing", True),
            use_null_route=getattr(args, "rgcer_use_null_route", True),
            use_film=getattr(args, "rgcer_use_film", True),
            use_adapter=getattr(args, "rgcer_use_adapter", True),
            transfer_mechanism=self.rgcer_transfer_mechanism,
            spatial_pos_max_clip=kwargs.get(
                "spatial_pos_max_clip", getattr(args, "spatial_pos_clip", 20)
            ),
            metadata_overrides=getattr(args, "auxiliary_metadata_overrides", None),
        )

    def _head_mode(self, task):
        return getattr(self.decoders[task], "mode", self.prediction_mode)

    def _response_profile(self, h):
        # Review §25-28: the router consumes these preliminary responses as
        # routing evidence, so they must be deterministic even in train mode
        # (heads called with deterministic=True pin their dropout off) and
        # must never feed gradients back into the source heads or backbone.
        with torch.no_grad():
            detached_h = h.detach()
            responses = []
            for task in self.task_name:
                raw = self.decoders[task](detached_h, deterministic=True)
                responses.append(point_from_raw(raw, self._head_mode(task)))
            return torch.stack(responses, dim=1)

    @staticmethod
    def _decoded_diagnostics(base_decoded, route_decoded, final_decoded):
        return {
            "base_prediction": base_decoded.median,
            "route_prediction": route_decoded.median,
            "final_prediction": final_decoded.median,
            "base_lower": base_decoded.lower,
            "base_upper": base_decoded.upper,
            "route_lower": route_decoded.lower,
            "route_upper": route_decoded.upper,
            "final_lower": final_decoded.lower,
            "final_upper": final_decoded.upper,
        }

    def _base_diagnostics(self, task, index, h, base_raw, response_profile=None):
        null = torch.ones(h.size(0), 1, device=h.device, dtype=h.dtype)
        task_count = response_profile.size(1) if response_profile is not None else 0
        base_decoded = decode_prediction(base_raw, self._head_mode(task))
        return {
            "target_task": task,
            "target_index": index,
            "response_profile": response_profile
            if response_profile is not None
            else torch.zeros(h.size(0), 0, 1, device=h.device, dtype=h.dtype),
            "source_weights": torch.zeros(h.size(0), task_count, device=h.device, dtype=h.dtype),
            "joint_source_weights": torch.zeros(h.size(0), task_count, device=h.device, dtype=h.dtype),
            "routing_entropy": torch.zeros(h.size(0), device=h.device, dtype=h.dtype),
            "task_context": torch.zeros_like(h),
            "gamma": torch.zeros_like(h),
            "beta": torch.zeros_like(h),
            "adapter_output": torch.zeros_like(h),
            "base_representation": h,
            "route_representation": h,
            "final_representation": h,
            "base_raw": base_raw,
            "route_raw": base_raw,
            "final_raw": base_raw,
            "null_weight": null,
            **self._decoded_diagnostics(base_decoded, base_decoded, base_decoded),
        }

    def _forward_task(self, h, task, response_profile, return_aux, source_mask, routing_enabled):
        index = self.task_name.index(task)
        base_raw = self.decoders[task](h)
        if not routing_enabled:
            diagnostics = self._base_diagnostics(task, index, h, base_raw, response_profile)
            return {task: base_raw}, diagnostics if return_aux else None

        # The router sees a detached shared representation.  Backbone learning
        # therefore comes from the HPS/base path; routing can still train its
        # own projections, response encoder, and shared adapter.
        route_representation, router_diagnostics = self.encoder.task_conditioner(
            h.detach(),
            task,
            response_profile,
            return_aux=True,
            source_mask=source_mask,
        )
        null_weight = router_diagnostics.null_weight
        route_raw = self.decoders[task](route_representation)
        mode = self._head_mode(task)
        base_decoded = decode_prediction(base_raw, mode)
        route_decoded = decode_prediction(route_raw, mode)

        if self.rgcer_fallback_space == "prediction":
            final_decoded = blend_decoded_predictions(base_decoded, route_decoded, null_weight)
            blended_raw = encode_decoded_prediction(final_decoded, mode)
            # Preserve exact endpoint outputs at the two interpretable
            # extremes, including quantile raw tensors in train mode.
            final_raw = torch.where(null_weight.eq(1.0), base_raw, blended_raw)
            final_raw = torch.where(null_weight.eq(0.0), route_raw, final_raw)
            # There is no single representation corresponding to a nonlinear
            # prediction-space convex combination.
            final_representation = None
        else:
            mixed_representation = null_weight * h + (1.0 - null_weight) * route_representation
            # Legacy compatibility ablation only.  ``where`` preserves the
            # exact HPS tensor for a hard NULL decision.
            final_representation = torch.where(null_weight.eq(1.0), h, mixed_representation)
            mixed_raw = self.decoders[task](final_representation)
            final_raw = torch.where(null_weight.eq(1.0), base_raw, mixed_raw)
            final_decoded = decode_prediction(final_raw, mode)
        predictions = {task: final_raw}
        if not return_aux:
            return predictions, None
        diagnostics = router_diagnostics.as_dict()
        diagnostics.update(
            {
                "base_representation": h,
                "route_representation": route_representation,
                "final_representation": final_representation,
                "base_raw": base_raw,
                "route_raw": route_raw,
                "final_raw": final_raw,
                **self._decoded_diagnostics(base_decoded, route_decoded, final_decoded),
            }
        )
        return predictions, diagnostics

    def forward(
        self,
        inputs,
        task_name=None,
        return_all_tasks=False,
        return_aux=False,
        source_mask=None,
        routing_enabled=True,
    ):
        h = self.encoder.encode_backbone(inputs)
        if return_all_tasks:
            # Keep warm-up diagnostics compatible with routed diagnostics.
            response_profile = self._response_profile(h)
            predictions = {}
            diagnostics = {}
            for task in self.task_name:
                result, task_diagnostics = self._forward_task(
                    h, task, response_profile, return_aux, source_mask, routing_enabled
                )
                predictions.update(result)
                if return_aux:
                    diagnostics[task] = task_diagnostics
            return (predictions, diagnostics) if return_aux else predictions

        if task_name is None:
            raise ValueError("task_name is required unless return_all_tasks=True")
        if task_name not in self.decoders:
            raise KeyError(f"Unknown task: {task_name}")
        # The profile is detached from the backbone/head gradients.  Computing it
        # during HPS warm-up keeps diagnostics shape-stable without training the
        # router or letting preliminary responses affect the HPS loss.
        response_profile = self._response_profile(h)
        result, diagnostics = self._forward_task(
            h, task_name, response_profile, return_aux, source_mask, routing_enabled
        )
        return (result, diagnostics) if return_aux else result


__all__ = ["Encoder", "Graphormer_rgcer"]
