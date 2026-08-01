"""Relation-aware task prompts and task-conditioned molecular adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PromptDiagnostics:
    raw_prompts: torch.Tensor
    contextual_prompts: torch.Tensor
    attention: torch.Tensor


class TaskPromptRelationEncoder(nn.Module):
    """Learn task relations only inside a learnable prompt bank."""

    def __init__(
        self,
        num_tasks: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.num_tasks = num_tasks
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.prompt_bank = nn.Parameter(torch.empty(num_tasks, hidden_dim))
        nn.init.normal_(self.prompt_bank, mean=0.0, std=0.02)
        ffn_dim = hidden_dim * 2 if ffn_dim is None else int(ffn_dim)
        self.layers = nn.ModuleList(
            [
                _PromptRelationLayer(hidden_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.prompt_bank.unsqueeze(0)
        attention_per_layer = []
        for layer in self.layers:
            x, attention = layer(x, return_attention=return_attention)
            if return_attention:
                if attention is None:
                    raise RuntimeError("Prompt relation layer returned no attention")
                attention_per_layer.append(attention.squeeze(0))
        contextual_prompts = self.final_norm(x).squeeze(0)
        if not return_attention:
            return contextual_prompts, None
        if attention_per_layer:
            attention = torch.stack(attention_per_layer, dim=0)
        else:
            attention = torch.empty(
                0,
                self.num_heads,
                self.num_tasks,
                self.num_tasks,
                device=contextual_prompts.device,
                dtype=contextual_prompts.dtype,
            )
        return contextual_prompts, attention


class _PromptRelationLayer(nn.Module):
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

    def forward(self, x: torch.Tensor, return_attention: bool):
        normalized = self.norm1(x)
        attention_out, attention = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attention_out)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x, attention


class PromptFiLMAdapter(nn.Module):
    """FiLM conditioning plus a shared low-rank residual adapter."""

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
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = max(1, int(round(hidden_dim * adapter_ratio)))
        self.prompt_norm = nn.LayerNorm(hidden_dim)
        self.film_head = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate_head = nn.Linear(hidden_dim, 1)
        self.condition_norm = nn.LayerNorm(hidden_dim)
        self.adapter_down = nn.Linear(hidden_dim, self.bottleneck_dim)
        self.adapter_up = nn.Linear(self.bottleneck_dim, hidden_dim)
        self.adapter_dropout = nn.Dropout(dropout)
        self._reset_parameters(gate_init)

    def _reset_parameters(self, gate_init: float) -> None:
        nn.init.normal_(self.film_head.weight, std=1e-3)
        nn.init.zeros_(self.film_head.bias)
        nn.init.normal_(self.gate_head.weight, std=1e-3)
        nn.init.constant_(self.gate_head.bias, gate_init)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.normal_(self.adapter_up.weight, std=1e-3)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, molecular_representation: torch.Tensor, task_prompt: torch.Tensor, return_aux=False):
        if molecular_representation.ndim != 2:
            raise ValueError("molecular_representation must have shape [B, D]")
        if task_prompt.ndim != 1:
            raise ValueError("task_prompt must have shape [D]")
        if molecular_representation.size(-1) != self.hidden_dim or task_prompt.size(-1) != self.hidden_dim:
            raise ValueError("Molecular representation and prompt dimensions must equal hidden_dim")
        prompt = self.prompt_norm(task_prompt)
        gamma, beta = self.film_head(prompt).chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        conditioned = self.condition_norm(
            (1.0 + gamma.unsqueeze(0)) * molecular_representation + beta.unsqueeze(0)
        )
        adapter = self.adapter_up(
            self.adapter_dropout(F.gelu(self.adapter_down(conditioned)))
        )
        gate = torch.sigmoid(self.gate_head(prompt))
        output = molecular_representation + gate.view(1, 1) * adapter
        if not return_aux:
            return output
        return output, {
            "gamma": gamma,
            "beta": beta,
            "gate": gate,
            "adapter_norm": adapter.norm(dim=-1).mean(),
        }


class RelationAwareTaskConditioner(nn.Module):
    def __init__(
        self,
        task_names: List[str],
        hidden_dim: int,
        prompt_heads: int = 4,
        prompt_layers: int = 1,
        prompt_ffn_dim: Optional[int] = None,
        prompt_dropout: float = 0.1,
        adapter_ratio: float = 0.25,
        gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if not task_names or len(set(task_names)) != len(task_names):
            raise ValueError("task_names must be non-empty and unique")
        self.task_names = list(task_names)
        self.task_to_index = {name: index for index, name in enumerate(self.task_names)}
        self.relation_encoder = TaskPromptRelationEncoder(
            len(self.task_names),
            hidden_dim,
            num_heads=prompt_heads,
            num_layers=prompt_layers,
            ffn_dim=prompt_ffn_dim,
            dropout=prompt_dropout,
        )
        self.adapter = PromptFiLMAdapter(
            hidden_dim,
            adapter_ratio=adapter_ratio,
            dropout=prompt_dropout,
            gate_init=gate_init,
        )

    def _task_index(self, task_name: str) -> int:
        if task_name not in self.task_to_index:
            raise KeyError(f"Unknown task_name {task_name!r}; available: {self.task_names}")
        return self.task_to_index[task_name]

    def forward(self, molecular_representation, task_name: str, return_aux: bool = False):
        contextual, _ = self.relation_encoder(return_attention=False)
        return self.adapter(
            molecular_representation,
            contextual[self._task_index(task_name)],
            return_aux=return_aux,
        )

    def condition_all(self, molecular_representation) -> Dict[str, torch.Tensor]:
        contextual, _ = self.relation_encoder(return_attention=False)
        return {
            name: self.adapter(molecular_representation, contextual[index])
            for index, name in enumerate(self.task_names)
        }

    @torch.no_grad()
    def get_prompt_diagnostics(self) -> PromptDiagnostics:
        contextual, attention = self.relation_encoder(return_attention=True)
        if attention is None:
            raise RuntimeError("Prompt attention diagnostics were not produced")
        return PromptDiagnostics(
            raw_prompts=self.relation_encoder.prompt_bank.detach().clone(),
            contextual_prompts=contextual.detach().clone(),
            attention=attention.detach().clone(),
        )
