"""Molecule-adaptive, task-conditioned Prompt routing.

The module only relates task prompts to one another.  Molecular batches are
encoded independently and enter the router as the current sample's shared
Graphormer representation; no other task's molecular batch is read or cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from architecture.toxacute_tasks import TaskMetadata, build_task_metadata


@dataclass
class RoutingDiagnostics:
    target_task: str
    target_index: int
    routing_weights: torch.Tensor
    transfer_gate: torch.Tensor
    routing_entropy: torch.Tensor
    task_context: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor


def _ordered_vocab(values: Iterable[str]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for value in values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


class FactorizedTaskPromptBank(nn.Module):
    """Build prompts from endpoint factors plus a task-specific residual."""

    def __init__(
        self,
        task_names: Sequence[str],
        hidden_dim: int,
        use_factorized_prompt: bool = True,
        task_residual_scale: float = 0.1,
        metadata_overrides: Optional[Dict[str, dict]] = None,
    ) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("task_names cannot be empty")
        if len(set(task_names)) != len(task_names):
            raise ValueError("task_names must be unique")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if task_residual_scale < 0:
            raise ValueError("task_residual_scale must be non-negative")

        self.task_names = list(task_names)
        self.hidden_dim = int(hidden_dim)
        self.use_factorized_prompt = bool(use_factorized_prompt)
        self.task_residual_scale = float(task_residual_scale)
        self.task_residual = nn.Parameter(torch.empty(len(task_names), hidden_dim))
        nn.init.normal_(self.task_residual, mean=0.0, std=0.02)

        self.metadata: List[TaskMetadata] = []
        self.organism_vocab: Dict[str, int] = {}
        self.route_vocab: Dict[str, int] = {}
        self.measurement_vocab: Dict[str, int] = {}
        self.population_vocab: Dict[str, int] = {}
        self.organism_embedding: Optional[nn.Embedding] = None
        self.route_embedding: Optional[nn.Embedding] = None
        self.measurement_embedding: Optional[nn.Embedding] = None
        self.population_embedding: Optional[nn.Embedding] = None

        if not self.use_factorized_prompt:
            return

        metadata_overrides = metadata_overrides or {}
        parsed_metadata = []
        for task_name in self.task_names:
            override = metadata_overrides.get(task_name)
            if override is None:
                parsed_metadata.extend(build_task_metadata([task_name]))
                continue
            required = {
                key: override[key]
                for key in ("task_name", "organism", "route", "measurement", "population")
                if key in override
            }
            required["task_name"] = task_name
            missing = {"organism", "route", "measurement", "population"}.difference(required)
            if missing:
                raise ValueError(f"Metadata override for {task_name!r} is missing {sorted(missing)}")
            parsed_metadata.append(TaskMetadata(**required))
        self.metadata = parsed_metadata
        self.organism_vocab = _ordered_vocab(item.organism for item in self.metadata)
        self.route_vocab = _ordered_vocab(item.route for item in self.metadata)
        self.measurement_vocab = _ordered_vocab(item.measurement for item in self.metadata)
        self.population_vocab = _ordered_vocab(item.population for item in self.metadata)

        self.register_buffer(
            "organism_ids",
            torch.tensor([self.organism_vocab[item.organism] for item in self.metadata], dtype=torch.long),
        )
        self.register_buffer(
            "route_ids",
            torch.tensor([self.route_vocab[item.route] for item in self.metadata], dtype=torch.long),
        )
        self.register_buffer(
            "measurement_ids",
            torch.tensor(
                [self.measurement_vocab[item.measurement] for item in self.metadata], dtype=torch.long
            ),
        )
        self.register_buffer(
            "population_ids",
            torch.tensor([self.population_vocab[item.population] for item in self.metadata], dtype=torch.long),
        )

        self.organism_embedding = nn.Embedding(len(self.organism_vocab), hidden_dim)
        self.route_embedding = nn.Embedding(len(self.route_vocab), hidden_dim)
        self.measurement_embedding = nn.Embedding(len(self.measurement_vocab), hidden_dim)
        self.population_embedding = nn.Embedding(len(self.population_vocab), hidden_dim)
        for embedding in (
            self.organism_embedding,
            self.route_embedding,
            self.measurement_embedding,
            self.population_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

    def forward(self) -> torch.Tensor:
        residual = self.task_residual_scale * self.task_residual
        if not self.use_factorized_prompt:
            return residual
        return (
            self.organism_embedding(self.organism_ids)
            + self.route_embedding(self.route_ids)
            + self.measurement_embedding(self.measurement_ids)
            + self.population_embedding(self.population_ids)
            + residual
        )


class PromptRelationLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        normalized = self.norm1(x)
        attention_output, attention = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attention_output)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x, attention


class PromptRelationEncoder(nn.Module):
    """Global task-prompt self-attention, independent of molecular batches."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.hidden_dim = int(hidden_dim)
        ffn_dim = hidden_dim * 2 if ffn_dim is None else int(ffn_dim)
        self.layers = nn.ModuleList(
            [PromptRelationLayer(hidden_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.__dict__["_compat_prompt_bank"] = None

    def bind_prompt_bank(self, parameter: nn.Parameter) -> None:
        """Expose the factorized bank residual for old diagnostic callers.

        The reference is deliberately kept outside the module registry so the
        parameter is not duplicated in the state dict.
        """

        self.__dict__["_compat_prompt_bank"] = parameter

    @property
    def prompt_bank(self) -> Optional[nn.Parameter]:
        return self.__dict__.get("_compat_prompt_bank")

    def forward(self, raw_prompts: torch.Tensor, return_attention: bool = False):
        if raw_prompts.ndim != 2:
            raise ValueError("raw_prompts must have shape [T, D]")
        x = raw_prompts.unsqueeze(0)
        attention_per_layer = []
        for layer in self.layers:
            x, attention = layer(x, return_attention=return_attention)
            if return_attention:
                if attention is None:
                    raise RuntimeError("Prompt attention was not returned")
                attention_per_layer.append(attention.squeeze(0))
        contextual = self.final_norm(x).squeeze(0)
        if not return_attention:
            return contextual, None
        if attention_per_layer:
            attention = torch.stack(attention_per_layer, dim=0)
        else:
            attention = torch.empty(
                0,
                0,
                raw_prompts.size(0),
                raw_prompts.size(0),
                device=raw_prompts.device,
                dtype=raw_prompts.dtype,
            )
        return contextual, attention


class MoleculeAdaptiveSparseRouter(nn.Module):
    """Route each molecule/target pair over contextual task prompts."""

    def __init__(
        self,
        hidden_dim: int,
        router_dim: Optional[int] = None,
        top_k: int = 8,
        temperature: float = 1.0,
        exclude_target: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        router_dim = hidden_dim if router_dim is None else int(router_dim)
        if hidden_dim <= 0 or router_dim <= 0:
            raise ValueError("hidden_dim and router_dim must be positive")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.hidden_dim = int(hidden_dim)
        self.router_dim = router_dim
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.exclude_target = bool(exclude_target)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, router_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_dim, router_dim),
        )
        self.key_projection = nn.Linear(hidden_dim, router_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.context_norm = nn.LayerNorm(hidden_dim)

    def _sparse_softmax(self, logits: torch.Tensor, available_sources: int) -> torch.Tensor:
        if available_sources <= 0:
            return torch.zeros_like(logits)
        if self.top_k == 0 or self.top_k >= available_sources:
            return torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(logits, k=min(self.top_k, available_sources), dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(dim=-1, index=top_indices, src=top_values)
        return torch.softmax(sparse_logits, dim=-1)

    def forward(
        self,
        molecular_representation: torch.Tensor,
        contextual_prompts: torch.Tensor,
        target_index: int,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if molecular_representation.ndim != 2:
            raise ValueError("molecular_representation must have shape [B, D]")
        if contextual_prompts.ndim != 2:
            raise ValueError("contextual_prompts must have shape [T, D]")
        batch_size, hidden_dim = molecular_representation.shape
        task_count, prompt_dim = contextual_prompts.shape
        if hidden_dim != self.hidden_dim or prompt_dim != self.hidden_dim:
            raise ValueError("molecular and prompt dimensions must equal hidden_dim")
        if not 0 <= target_index < task_count:
            raise IndexError(f"target_index={target_index} is outside [0, {task_count})")

        if source_mask is None:
            allowed = torch.ones(
                batch_size, task_count, dtype=torch.bool, device=contextual_prompts.device
            )
        else:
            source_mask = source_mask.to(device=contextual_prompts.device, dtype=torch.bool)
            if source_mask.ndim == 1:
                if source_mask.numel() != task_count:
                    raise ValueError("source_mask must have shape [T] or [B, T]")
                allowed = source_mask.unsqueeze(0).expand(batch_size, -1).clone()
            elif source_mask.ndim == 2 and source_mask.shape == (batch_size, task_count):
                allowed = source_mask.clone()
            else:
                raise ValueError("source_mask must have shape [T] or [B, T]")
        if self.exclude_target:
            allowed[:, target_index] = False

        target_prompt = contextual_prompts[target_index].expand(batch_size, -1)
        query = self.query_projection(torch.cat((molecular_representation, target_prompt), dim=-1))
        keys = self.key_projection(contextual_prompts)
        values = self.value_projection(contextual_prompts)
        logits = torch.matmul(query, keys.transpose(0, 1)) / sqrt(self.router_dim)
        logits = logits / self.temperature

        available_sources = allowed.sum(dim=-1)
        logits = logits.masked_fill(~allowed, float("-inf"))
        if self.top_k and self.top_k < task_count:
            top_indices = torch.topk(logits, k=min(self.top_k, task_count), dim=-1).indices
            selected = torch.zeros_like(allowed)
            selected.scatter_(1, top_indices, True)
            allowed = allowed & selected
            logits = logits.masked_fill(~allowed, float("-inf"))
            available_sources = allowed.sum(dim=-1)
        safe_logits = torch.where(
            available_sources.unsqueeze(-1) > 0,
            logits,
            torch.zeros_like(logits),
        )
        routing_weights = torch.softmax(safe_logits, dim=-1) * allowed.to(logits.dtype)
        source_context = torch.matmul(routing_weights, values)

        task_context = self.context_norm(target_prompt + source_context)
        safe_weights = routing_weights.clamp_min(1e-12)
        routing_entropy = -(routing_weights * safe_weights.log()).sum(dim=-1)
        return task_context, routing_weights, routing_entropy


class PromptConditionedFiLMAdapter(nn.Module):
    """Shared FiLM/bottleneck adapter with a sample-wise transfer gate."""

    def __init__(
        self,
        hidden_dim: int,
        adapter_ratio: float = 0.25,
        dropout: float = 0.1,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < adapter_ratio <= 1.0:
            raise ValueError("adapter_ratio must be in (0, 1]")
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = max(1, int(round(hidden_dim * adapter_ratio)))
        self.film_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2))
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.adapter_down = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.adapter_up = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.adapter_dropout = nn.Dropout(dropout)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters(gate_init)

    def _reset_parameters(self, gate_init: float) -> None:
        film_linear = self.film_head[-1]
        nn.init.normal_(film_linear.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(film_linear.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.adapter_up.bias)
        gate_linear = self.gate_head[-1]
        nn.init.normal_(gate_linear.weight, mean=0.0, std=1e-3)
        nn.init.constant_(gate_linear.bias, gate_init)

    def forward(self, molecular_representation: torch.Tensor, task_context: torch.Tensor, return_aux=False):
        if molecular_representation.ndim != 2 or task_context.ndim != 2:
            raise ValueError("molecular_representation and task_context must have shape [B, D]")
        if molecular_representation.shape != task_context.shape:
            raise ValueError("molecular_representation and task_context must have the same shape")
        raw_gamma, beta = self.film_head(task_context).chunk(2, dim=-1)
        gamma = torch.tanh(raw_gamma)
        conditioned = self.condition_norm((1.0 + gamma) * molecular_representation + beta)
        adapter_hidden = self.adapter_dropout(F.gelu(self.adapter_down(conditioned)))
        adapter_output = self.adapter_up(adapter_hidden)
        transfer_gate = torch.sigmoid(
            self.gate_head(torch.cat((molecular_representation, task_context), dim=-1))
        )
        task_representation = molecular_representation + transfer_gate * adapter_output
        if not return_aux:
            return task_representation
        return task_representation, {
            "transfer_gate": transfer_gate,
            "gamma": gamma,
            "beta": beta,
            "adapter_output": adapter_output,
        }


class MoleculeAdaptiveTaskConditioner(nn.Module):
    """Complete stateless factorized Prompt + adaptive routing conditioner."""

    VALID_ROUTER_MODES = {"static", "dynamic"}

    def __init__(
        self,
        task_names: Sequence[str],
        hidden_dim: int,
        use_factorized_prompt: bool = True,
        task_residual_scale: float = 0.1,
        prompt_layers: int = 1,
        prompt_heads: int = 4,
        prompt_ffn_dim: Optional[int] = None,
        prompt_dropout: float = 0.1,
        router_mode: str = "dynamic",
        router_dim: Optional[int] = None,
        router_top_k: int = 8,
        router_temperature: float = 1.0,
        exclude_target_from_sources: bool = True,
        adapter_ratio: float = 0.25,
        gate_init: float = -2.0,
        metadata_overrides: Optional[Dict[str, dict]] = None,
    ) -> None:
        super().__init__()
        if not task_names or len(set(task_names)) != len(task_names):
            raise ValueError("task_names must be non-empty and unique")
        if router_mode not in self.VALID_ROUTER_MODES:
            raise ValueError(f"router_mode must be one of {sorted(self.VALID_ROUTER_MODES)}")
        self.task_names = list(task_names)
        self.task_to_index = {name: index for index, name in enumerate(self.task_names)}
        self.hidden_dim = int(hidden_dim)
        self.router_mode = router_mode
        self.exclude_target_from_sources = bool(exclude_target_from_sources)
        self.prompt_bank = FactorizedTaskPromptBank(
            self.task_names,
            hidden_dim,
            use_factorized_prompt=use_factorized_prompt,
            task_residual_scale=task_residual_scale,
            metadata_overrides=metadata_overrides,
        )
        self.relation_encoder = PromptRelationEncoder(
            hidden_dim,
            num_heads=prompt_heads,
            num_layers=prompt_layers,
            ffn_dim=prompt_ffn_dim,
            dropout=prompt_dropout,
        )
        self.relation_encoder.bind_prompt_bank(self.prompt_bank.task_residual)
        self.router = None
        if router_mode == "dynamic":
            self.router = MoleculeAdaptiveSparseRouter(
                hidden_dim,
                router_dim=router_dim,
                top_k=router_top_k,
                temperature=router_temperature,
                exclude_target=exclude_target_from_sources,
                dropout=prompt_dropout,
            )
        self.film_adapter = PromptConditionedFiLMAdapter(
            hidden_dim,
            adapter_ratio=adapter_ratio,
            dropout=prompt_dropout,
            gate_init=gate_init,
        )

    def _task_index(self, task_name: str) -> int:
        if task_name not in self.task_to_index:
            raise KeyError(f"Unknown task_name={task_name!r}; available: {self.task_names}")
        return self.task_to_index[task_name]

    def encode_prompts(self, return_attention: bool = False):
        raw_prompts = self.prompt_bank()
        contextual_prompts, attention = self.relation_encoder(
            raw_prompts,
            return_attention=return_attention,
        )
        return raw_prompts, contextual_prompts, attention

    def _condition_with_prompts(
        self,
        molecular_representation: torch.Tensor,
        task_name: str,
        contextual_prompts: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ):
        target_index = self._task_index(task_name)
        batch_size = molecular_representation.size(0)
        if self.router_mode == "dynamic":
            if self.router is None:
                raise RuntimeError("Dynamic routing requested without a router")
            task_context, routing_weights, routing_entropy = self.router(
                molecular_representation,
                contextual_prompts,
                target_index,
                source_mask=source_mask,
            )
        else:
            task_context = contextual_prompts[target_index].expand(batch_size, -1)
            routing_weights = torch.zeros(
                batch_size,
                len(self.task_names),
                dtype=molecular_representation.dtype,
                device=molecular_representation.device,
            )
            routing_entropy = torch.zeros(
                batch_size, dtype=molecular_representation.dtype, device=molecular_representation.device
            )
        task_representation, film_aux = self.film_adapter(
            molecular_representation, task_context, return_aux=True
        )
        diagnostics = RoutingDiagnostics(
            target_task=task_name,
            target_index=target_index,
            routing_weights=routing_weights,
            transfer_gate=film_aux["transfer_gate"],
            routing_entropy=routing_entropy,
            task_context=task_context,
            gamma=film_aux["gamma"],
            beta=film_aux["beta"],
        )
        return task_representation, diagnostics

    def forward(
        self,
        molecular_representation: torch.Tensor,
        task_name: str,
        return_aux: bool = False,
        source_mask: Optional[torch.Tensor] = None,
    ):
        _raw, contextual_prompts, _attention = self.encode_prompts(return_attention=False)
        representation, diagnostics = self._condition_with_prompts(
            molecular_representation, task_name, contextual_prompts, source_mask=source_mask
        )
        return (representation, diagnostics) if return_aux else representation

    def condition_all(
        self,
        molecular_representation: torch.Tensor,
        return_aux: bool = False,
        source_mask: Optional[torch.Tensor] = None,
    ):
        _raw, contextual_prompts, _attention = self.encode_prompts(return_attention=False)
        representations: Dict[str, torch.Tensor] = {}
        diagnostics: Dict[str, RoutingDiagnostics] = {}
        for task_name in self.task_names:
            representation, task_diagnostics = self._condition_with_prompts(
                molecular_representation,
                task_name,
                contextual_prompts,
                source_mask=source_mask,
            )
            representations[task_name] = representation
            diagnostics[task_name] = task_diagnostics
        return (representations, diagnostics) if return_aux else representations

    @torch.no_grad()
    def prompt_diagnostics(self) -> Dict[str, torch.Tensor]:
        contextual, attention = self.relation_encoder(self.prompt_bank(), return_attention=True)
        if attention is None:
            raise RuntimeError("Prompt attention diagnostics were not returned")
        return {
            "raw_prompts": self.prompt_bank().detach().clone(),
            "contextual_prompts": contextual.detach().clone(),
            "prompt_attention": attention.detach().clone(),
        }


def count_router_parameters(model_or_conditioner: nn.Module) -> int:
    """Count trainable parameters in the adaptive task conditioner."""

    conditioner = getattr(getattr(model_or_conditioner, "encoder", None), "task_conditioner", None)
    if conditioner is None:
        conditioner = model_or_conditioner
    if not isinstance(conditioner, MoleculeAdaptiveTaskConditioner):
        raise TypeError("Expected a MoleculeAdaptiveTaskConditioner or a model containing one")
    return sum(parameter.numel() for parameter in conditioner.parameters() if parameter.requires_grad)


__all__ = [
    "FactorizedTaskPromptBank",
    "MoleculeAdaptiveSparseRouter",
    "MoleculeAdaptiveTaskConditioner",
    "PromptConditionedFiLMAdapter",
    "PromptRelationEncoder",
    "RoutingDiagnostics",
    "count_router_parameters",
]
