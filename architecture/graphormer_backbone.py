"""Shared chemical Graphormer backbone used by all Phase 2 models."""

from __future__ import annotations

import torch
from torch import nn

from molecular_features import ATOM_CARDINALITIES, BOND_CARDINALITIES


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        batch_size, sequence_length, hidden_dim = x.shape
        q = self.q_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, sequence_length, hidden_dim)
        return self.out_proj(output)


class GraphormerBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_heads, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias, key_padding_mask, node_padding_mask):
        residual = x
        x = self.attention_norm(x)
        x = residual + self.attention_dropout(
            self.attention(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        )
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        if node_padding_mask is not None:
            node_values = x[:, 1:, :].masked_fill(node_padding_mask.unsqueeze(-1), 0.0)
            x = torch.cat([x[:, :1, :], node_values], dim=1)
        return x


class MolecularGraphormerBackbone(nn.Module):
    """Graphormer with categorical atom and multi-hop chemical bond encoding."""

    def __init__(
        self,
        hidden_dim,
        num_heads,
        num_layers,
        ffn_dim,
        dropout=0.1,
        spatial_pos_max_clip=20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_pos_max_clip = int(spatial_pos_max_clip)
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, hidden_dim, padding_idx=0) for cardinality in ATOM_CARDINALITIES]
        )
        self.in_degree_embedding = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.out_degree_embedding = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.spatial_embedding = nn.Embedding(self.spatial_pos_max_clip + 2, num_heads, padding_idx=0)
        self.direct_bond_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, num_heads, padding_idx=0) for cardinality in BOND_CARDINALITIES]
        )
        # edge_input has one extra index because the raw Cython output uses
        # -1 for padding and the collator shifts it to 0.
        self.path_bond_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, num_heads, padding_idx=0) for cardinality in BOND_CARDINALITIES]
        )
        self.graph_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.graph_token_distance = nn.Parameter(torch.zeros(1, num_heads, 1))
        nn.init.normal_(self.graph_token, std=0.02)
        nn.init.normal_(self.graph_token_distance, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [GraphormerBlock(hidden_dim, ffn_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def _encode_atoms(self, x, in_degree, out_degree):
        if x.ndim != 3 or x.size(-1) != len(self.atom_embeddings):
            raise ValueError(f"Expected x=[batch,nodes,{len(self.atom_embeddings)}], got {tuple(x.shape)}")
        node_feature = sum(embedding(x[..., field_id]) for field_id, embedding in enumerate(self.atom_embeddings))
        node_feature = node_feature + self.in_degree_embedding(in_degree.clamp_min(0).clamp_max(63))
        node_feature = node_feature + self.out_degree_embedding(out_degree.clamp_min(0).clamp_max(63))
        return node_feature

    def _encode_attention_bias(self, batch, n_nodes):
        attn_bias = batch.attn_bias
        if attn_bias.shape[-2:] != (n_nodes + 1, n_nodes + 1):
            raise ValueError("attn_bias shape is inconsistent with padded node count")
        bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).clone()

        spatial = batch.spatial_pos.clamp_min(0).clamp_max(self.spatial_pos_max_clip + 1)
        spatial_bias = self.spatial_embedding(spatial).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + spatial_bias

        direct_edge = batch.attn_edge_type
        direct_bias = 0.0
        for field_id, embedding in enumerate(self.direct_bond_embeddings):
            direct_bias = direct_bias + embedding(direct_edge[..., field_id])
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + direct_bias.permute(0, 3, 1, 2)

        path_edge = batch.edge_input
        path_bias = 0.0
        for field_id, embedding in enumerate(self.path_bond_embeddings):
            path_bias = path_bias + embedding(path_edge[..., field_id]).sum(dim=3)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + path_bias.permute(0, 3, 1, 2)

        bias[:, :, 1:, 0:1] = bias[:, :, 1:, 0:1] + self.graph_token_distance.unsqueeze(2)
        bias[:, :, 0:1, :] = bias[:, :, 0:1, :] + self.graph_token_distance.unsqueeze(2)
        return bias

    def forward(self, batch):
        required = ("x", "in_degree", "out_degree", "spatial_pos", "attn_bias", "attn_edge_type", "edge_input")
        missing = [name for name in required if not hasattr(batch, name)]
        if missing:
            raise ValueError(f"Graphormer batch is missing fields: {missing}")
        x = batch.x.long()
        batch_size, n_nodes = x.shape[:2]
        node_feature = self._encode_atoms(batch.x, batch.in_degree.long(), batch.out_degree.long())
        graph_token = self.graph_token.expand(batch_size, -1, -1)
        sequence = torch.cat([graph_token, node_feature], dim=1)
        sequence = self.input_dropout(sequence)

        node_padding_mask = getattr(batch, "padding_mask", None)
        if node_padding_mask is None:
            node_padding_mask = batch.x.eq(0).all(dim=-1)
        key_padding_mask = torch.cat(
            [torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device), node_padding_mask], dim=1
        )
        attn_bias = self._encode_attention_bias(batch, n_nodes)
        for layer in self.layers:
            sequence = layer(sequence, attn_bias, key_padding_mask, node_padding_mask)
        return self.final_norm(sequence)[:, 0, :]
