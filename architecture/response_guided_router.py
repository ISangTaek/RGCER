"""Response-guided conservative endpoint routing for RGCER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from architecture.molecule_adaptive_prompt import FactorizedTaskPromptBank


@dataclass
class RGCERDiagnostics:
    target_task: str
    target_index: int
    response_profile: torch.Tensor
    source_weights: torch.Tensor
    joint_source_weights: torch.Tensor
    null_weight: torch.Tensor
    routing_entropy: torch.Tensor
    task_context: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    adapter_output: torch.Tensor | None = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "target_task": self.target_task,
            "target_index": self.target_index,
            "response_profile": self.response_profile,
            "source_weights": self.source_weights,
            "joint_source_weights": self.joint_source_weights,
            "null_weight": self.null_weight,
            "routing_entropy": self.routing_entropy,
            "task_context": self.task_context,
            "gamma": self.gamma,
            "beta": self.beta,
            "adapter_output": self.adapter_output,
        }


class EndpointResponseEncoder(nn.Module):
    """Encode one standardized preliminary endpoint response per task."""

    def __init__(self, hidden_dim: int, response_hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        response_hidden_dim = hidden_dim if response_hidden_dim is None else int(response_hidden_dim)
        if response_hidden_dim <= 0:
            raise ValueError("response_hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(1, response_hidden_dim),
            nn.GELU(),
            nn.Linear(response_hidden_dim, hidden_dim),
        )

    def forward(self, response_profile: torch.Tensor) -> torch.Tensor:
        if response_profile.ndim != 3 or response_profile.size(-1) != 1:
            raise ValueError("response_profile must have shape [B, T, 1]")
        return self.network(torch.tanh(response_profile))


def _expand_source_mask(source_mask, batch_size, task_count, device):
    if source_mask is None:
        return torch.ones(batch_size, task_count, dtype=torch.bool, device=device)
    source_mask = source_mask.to(device=device, dtype=torch.bool)
    if source_mask.ndim == 1:
        if source_mask.numel() != task_count:
            raise ValueError("source_mask must have shape [T] or [B, T]")
        return source_mask.unsqueeze(0).expand(batch_size, -1).clone()
    if source_mask.ndim == 2 and source_mask.shape == (batch_size, task_count):
        return source_mask.clone()
    raise ValueError("source_mask must have shape [T] or [B, T]")


class ResponseGuidedEndpointRouter(nn.Module):
    """Route over source endpoint tokens plus an explicit NULL token."""

    def __init__(
        self,
        hidden_dim: int,
        router_dim: int | None = None,
        top_k: int = 8,
        temperature: float = 1.0,
        exclude_target: bool = True,
        dropout: float = 0.1,
        response_hidden_dim: int | None = None,
        use_source_response: bool = True,
        use_target_response: bool = True,
        use_molecule_query: bool = True,
        use_sparse_routing: bool = True,
        use_null_route: bool = True,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        router_dim = hidden_dim if router_dim is None else int(router_dim)
        if router_dim <= 0 or top_k < 0 or temperature <= 0:
            raise ValueError("router_dim must be positive, top_k non-negative, temperature positive")
        if use_sparse_routing and top_k <= 0:
            raise ValueError("Sparse routing requires top_k > 0")
        self.hidden_dim = int(hidden_dim)
        self.router_dim = router_dim
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.exclude_target = bool(exclude_target)
        self.use_source_response = bool(use_source_response)
        self.use_target_response = bool(use_target_response)
        self.use_molecule_query = bool(use_molecule_query)
        self.use_sparse_routing = bool(use_sparse_routing)
        self.use_null_route = bool(use_null_route)
        self.response_encoder = EndpointResponseEncoder(hidden_dim, response_hidden_dim)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, router_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_dim, router_dim),
        )
        self.key_projection = nn.Linear(hidden_dim, router_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.null_token = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.null_token, mean=0.0, std=0.02)
        self.null_key_projection = nn.Linear(hidden_dim, router_dim, bias=False)
        self.context_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        molecular_representation: torch.Tensor,
        task_prompts: torch.Tensor,
        response_profile: torch.Tensor,
        target_index: int,
        source_mask: Optional[torch.Tensor] = None,
    ):
        if molecular_representation.ndim != 2:
            raise ValueError("molecular_representation must have shape [B, D]")
        if task_prompts.ndim != 2:
            raise ValueError("task_prompts must have shape [T, D]")
        batch_size, hidden_dim = molecular_representation.shape
        task_count, prompt_dim = task_prompts.shape
        if hidden_dim != self.hidden_dim or prompt_dim != self.hidden_dim:
            raise ValueError("molecular and prompt dimensions must equal hidden_dim")
        if response_profile.shape != (batch_size, task_count, 1):
            raise ValueError("response_profile must have shape [B, T, 1]")
        if not 0 <= target_index < task_count:
            raise IndexError(f"target_index={target_index} is outside [0, {task_count})")

        response_embeddings = self.response_encoder(response_profile)
        prompt_batch = task_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        source_response_embeddings = (
            response_embeddings if self.use_source_response else torch.zeros_like(response_embeddings)
        )
        source_tokens = prompt_batch + source_response_embeddings
        target_prompt = prompt_batch[:, target_index, :]
        target_response = response_embeddings[:, target_index, :]
        target_response_term = (
            target_response if self.use_target_response else torch.zeros_like(target_response)
        )
        molecule_term = (
            molecular_representation
            if self.use_molecule_query
            else torch.zeros_like(molecular_representation)
        )
        query = self.query_projection(
            torch.cat((molecule_term, target_prompt, target_response_term), dim=-1)
        )
        keys = self.key_projection(source_tokens)
        values = self.value_projection(source_tokens)
        logits = torch.einsum("bd,btd->bt", query, keys) / (self.router_dim**0.5)
        logits = logits / self.temperature
        null_key = self.null_key_projection(self.null_token).expand(batch_size, -1)
        null_logits = (query * null_key).sum(dim=-1, keepdim=True) / (self.router_dim**0.5)
        null_logits = null_logits / self.temperature
        allowed = _expand_source_mask(source_mask, batch_size, task_count, logits.device)
        if self.exclude_target:
            allowed[:, target_index] = False

        # Top-k is applied only to real source endpoints; NULL is never removed.
        masked_logits = logits.masked_fill(~allowed, float("-inf"))
        if self.use_sparse_routing and self.top_k < task_count:
            top_indices = torch.topk(masked_logits, k=min(self.top_k, task_count), dim=-1).indices
            selected = torch.zeros_like(allowed)
            selected.scatter_(1, top_indices, True)
            allowed = allowed & selected
            masked_logits = logits.masked_fill(~allowed, float("-inf"))

        available = allowed.sum(dim=-1)
        if not self.use_null_route and torch.any(available.eq(0)):
            raise ValueError("No real source endpoints available while NULL route is disabled")
        safe_source_logits = torch.where(
            available.unsqueeze(-1) > 0,
            masked_logits,
            torch.zeros_like(masked_logits),
        )
        conditional_weights = torch.softmax(safe_source_logits, dim=-1) * allowed.to(logits.dtype)

        if self.use_null_route:
            joint_logits = torch.cat(
                (
                    null_logits,
                    torch.where(
                        available.unsqueeze(-1) > 0,
                        masked_logits,
                        torch.full_like(masked_logits, float("-inf")),
                    ),
                ),
                dim=-1,
            )
            joint_weights = torch.softmax(joint_logits, dim=-1)
            null_weight = joint_weights[:, :1]
            joint_source_weights = joint_weights[:, 1:]
            entropy_weights = joint_weights
        else:
            null_weight = torch.zeros(batch_size, 1, device=logits.device, dtype=logits.dtype)
            joint_source_weights = conditional_weights
            entropy_weights = conditional_weights
        source_context = torch.bmm(conditional_weights.unsqueeze(1), values).squeeze(1)
        task_context = self.context_norm(target_prompt + source_context)
        entropy = -(entropy_weights * entropy_weights.clamp_min(1e-12).log()).sum(dim=-1)
        return task_context, conditional_weights, joint_source_weights, null_weight, entropy


class SharedFiLMAdapter(nn.Module):
    """One shared, ungated FiLM bottleneck adapter for all endpoints."""

    def __init__(
        self,
        hidden_dim: int,
        adapter_ratio: float = 0.25,
        dropout: float = 0.1,
        use_film: bool = True,
        use_adapter: bool = True,
    ):
        super().__init__()
        if not 0.0 < adapter_ratio <= 1.0:
            raise ValueError("adapter_ratio must be in (0, 1]")
        self.hidden_dim = int(hidden_dim)
        self.use_film = bool(use_film)
        self.use_adapter = bool(use_adapter)
        self.bottleneck_dim = max(1, int(round(hidden_dim * adapter_ratio)))
        self.film_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 2))
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.adapter_down = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.adapter_up = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.film_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.film_head[-1].bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, h: torch.Tensor, task_context: torch.Tensor):
        if self.use_film:
            raw_gamma, beta = self.film_head(task_context).chunk(2, dim=-1)
            gamma = torch.tanh(raw_gamma)
            conditioned = self.condition_norm((1.0 + gamma) * h + beta)
        else:
            gamma = torch.zeros_like(h)
            beta = torch.zeros_like(h)
            conditioned = h

        if self.use_adapter:
            adapter_output = self.adapter_up(self.dropout(F.gelu(self.adapter_down(conditioned))))
            route_representation = h + adapter_output
        else:
            adapter_output = torch.zeros_like(h)
            route_representation = conditioned
        return route_representation, {
            "gamma": gamma,
            "beta": beta,
            "adapter_output": adapter_output,
        }


class ResponseStackingConditioner(nn.Module):
    """Use the complete preliminary response vector without endpoint routing."""

    def __init__(
        self,
        task_count: int,
        hidden_dim: int,
        adapter_ratio: float = 0.25,
        dropout: float = 0.1,
        use_film: bool = True,
        use_adapter: bool = True,
    ):
        super().__init__()
        if task_count <= 0 or hidden_dim <= 0:
            raise ValueError("task_count and hidden_dim must be positive")
        self.task_count = int(task_count)
        self.hidden_dim = int(hidden_dim)
        self.response_network = nn.Sequential(
            nn.Linear(task_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        target_prompt: torch.Tensor,
        response_profile: torch.Tensor,
    ):
        if response_profile.ndim != 3 or response_profile.shape[1:] != (self.task_count, 1):
            raise ValueError("response_profile must have shape [B, T, 1]")
        if target_prompt.shape != h.shape:
            raise ValueError("target_prompt and h must have matching shapes")
        response_context = self.response_network(response_profile.squeeze(-1))
        return self.context_norm(target_prompt + response_context)


class RGCERTaskConditioner(nn.Module):
    """Factorized endpoint prompts + response-guided router + shared adapter."""

    def __init__(
        self,
        task_names: Sequence[str],
        hidden_dim: int,
        use_factorized_prompt: bool = True,
        task_residual_scale: float = 0.1,
        router_dim: int | None = None,
        router_top_k: int = 8,
        router_temperature: float = 1.0,
        exclude_target_from_sources: bool = True,
        adapter_ratio: float = 0.25,
        dropout: float = 0.1,
        response_hidden_dim: int | None = None,
        use_source_response: bool = True,
        use_target_response: bool = True,
        use_molecule_query: bool = True,
        use_sparse_routing: bool = True,
        use_null_route: bool = True,
        use_film: bool = True,
        use_adapter: bool = True,
        transfer_mechanism: str = "endpoint_router",
    ):
        super().__init__()
        if not task_names or len(set(task_names)) != len(task_names):
            raise ValueError("task_names must be non-empty and unique")
        self.task_names = list(task_names)
        self.task_to_index = {name: index for index, name in enumerate(self.task_names)}
        valid_mechanisms = {"endpoint_router", "response_stacking", "target_only"}
        if transfer_mechanism not in valid_mechanisms:
            raise ValueError(f"transfer_mechanism must be one of {sorted(valid_mechanisms)}")
        self.transfer_mechanism = transfer_mechanism
        self.use_source_response = bool(use_source_response)
        self.use_target_response = bool(use_target_response)
        self.use_molecule_query = bool(use_molecule_query)
        self.use_sparse_routing = bool(use_sparse_routing)
        self.use_null_route = bool(use_null_route)
        self.use_film = bool(use_film)
        self.use_adapter = bool(use_adapter)
        self.prompt_bank = FactorizedTaskPromptBank(
            self.task_names,
            hidden_dim,
            use_factorized_prompt=use_factorized_prompt,
            task_residual_scale=task_residual_scale,
        )
        self.router = ResponseGuidedEndpointRouter(
            hidden_dim=hidden_dim,
            router_dim=router_dim,
            top_k=router_top_k,
            temperature=router_temperature,
            exclude_target=exclude_target_from_sources,
            dropout=dropout,
            response_hidden_dim=response_hidden_dim,
            use_source_response=self.use_source_response,
            use_target_response=self.use_target_response,
            use_molecule_query=self.use_molecule_query,
            use_sparse_routing=self.use_sparse_routing,
            use_null_route=self.use_null_route,
        )
        self.adapter = SharedFiLMAdapter(
            hidden_dim,
            adapter_ratio=adapter_ratio,
            dropout=dropout,
            use_film=self.use_film,
            use_adapter=self.use_adapter,
        )
        self.response_stacking = (
            ResponseStackingConditioner(
                task_count=len(self.task_names),
                hidden_dim=hidden_dim,
                adapter_ratio=adapter_ratio,
                dropout=dropout,
                use_film=self.use_film,
                use_adapter=self.use_adapter,
            )
            if transfer_mechanism == "response_stacking"
            else None
        )

    def _task_index(self, task_name):
        if task_name not in self.task_to_index:
            raise KeyError(f"Unknown task_name={task_name!r}; available: {self.task_names}")
        return self.task_to_index[task_name]

    def forward(self, h, task_name, response_profile, return_aux=False, source_mask=None):
        target_index = self._task_index(task_name)
        prompts = self.prompt_bank()
        batch_size = h.size(0)
        if self.transfer_mechanism == "endpoint_router":
            task_context, source_weights, joint_source_weights, null_weight, entropy = self.router(
                h,
                prompts,
                response_profile,
                target_index,
                source_mask=source_mask,
            )
            route_representation, adapter_aux = self.adapter(h, task_context)
        elif self.transfer_mechanism == "response_stacking":
            if self.response_stacking is None:
                raise RuntimeError("response_stacking conditioner is not initialized")
            target_prompt = prompts[target_index].unsqueeze(0).expand(batch_size, -1)
            task_context = self.response_stacking(
                h,
                target_prompt,
                response_profile,
            )
            route_representation, adapter_aux = self.adapter(h, task_context)
            source_weights = torch.zeros(
                batch_size, len(self.task_names), device=h.device, dtype=h.dtype
            )
            joint_source_weights = torch.zeros_like(source_weights)
            null_weight = torch.zeros(batch_size, 1, device=h.device, dtype=h.dtype)
            entropy = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
        else:
            target_prompt = prompts[target_index].unsqueeze(0).expand(batch_size, -1)
            task_context = target_prompt
            route_representation, adapter_aux = self.adapter(h, task_context)
            source_weights = torch.zeros(
                batch_size, len(self.task_names), device=h.device, dtype=h.dtype
            )
            joint_source_weights = torch.zeros_like(source_weights)
            null_weight = torch.zeros(batch_size, 1, device=h.device, dtype=h.dtype)
            entropy = torch.zeros(batch_size, device=h.device, dtype=h.dtype)
            response_profile = torch.zeros_like(response_profile)
        diagnostics = RGCERDiagnostics(
            target_task=task_name,
            target_index=target_index,
            response_profile=response_profile,
            source_weights=source_weights,
            joint_source_weights=joint_source_weights,
            null_weight=null_weight,
            routing_entropy=entropy,
            task_context=task_context,
            gamma=adapter_aux["gamma"],
            beta=adapter_aux["beta"],
            adapter_output=adapter_aux["adapter_output"],
        )
        if return_aux:
            return route_representation, diagnostics
        return route_representation


__all__ = [
    "EndpointResponseEncoder",
    "RGCERDiagnostics",
    "RGCERTaskConditioner",
    "ResponseGuidedEndpointRouter",
    "ResponseStackingConditioner",
    "SharedFiLMAdapter",
]
