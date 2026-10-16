"""Shared chemical Graphormer backbone used by all Phase 2 models."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from molecular_features import ATOM_CARDINALITIES, BOND_CARDINALITIES, FEATURE_SCHEMA_VERSION, FeatureSpec


@dataclass(frozen=True)
class BackboneTokens:
    graph_token: torch.Tensor
    atom_tokens: torch.Tensor
    atom_mask: torch.Tensor
    attentions: tuple[torch.Tensor, ...] = ()


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

    def forward(self, x, attn_bias=None, key_padding_mask=None, *, return_attention=False):
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
        if key_padding_mask is not None:
            # Padded queries have no meaningful attribution, even though the
            # graph token keeps their softmax finite.
            weights = weights.masked_fill(key_padding_mask[:, None, :, None], 0.0)
        attention = weights
        weights = self.dropout(weights)
        output = torch.matmul(weights, v).transpose(1, 2).reshape(batch_size, sequence_length, hidden_dim)
        output = self.out_proj(output)
        return (output, attention) if return_attention else output


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

    def forward(self, x, attn_bias, key_padding_mask, node_padding_mask, *, return_attention=False):
        residual = x
        x = self.attention_norm(x)
        attended = self.attention(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask,
                                  return_attention=return_attention)
        if return_attention:
            attended, attention = attended
        x = residual + self.attention_dropout(attended)
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        if node_padding_mask is not None:
            node_values = x[:, 1:, :].masked_fill(node_padding_mask.unsqueeze(-1), 0.0)
            x = torch.cat([x[:, :1, :], node_values], dim=1)
        return (x, attention) if return_attention else x


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
        edge_bias_mode="path",
        feature_spec: FeatureSpec | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.spatial_pos_max_clip = int(spatial_pos_max_clip)
        if edge_bias_mode not in {"path", "direct", "direct_plus_path"}:
            raise ValueError("edge_bias_mode must be 'path', 'direct', or 'direct_plus_path'")
        self.edge_bias_mode = edge_bias_mode
        self.feature_spec = feature_spec
        if feature_spec is not None and edge_bias_mode != "direct":
            raise ValueError("Explicit PGT FeatureSpec requires direct edge bias")
        atom_cardinalities = feature_spec.atom_cardinalities if feature_spec else ATOM_CARDINALITIES
        bond_cardinalities = feature_spec.bond_cardinalities if feature_spec else BOND_CARDINALITIES
        if feature_spec is not None:
            self.register_buffer("_feature_spec_identity", torch.tensor(list(bytes.fromhex(feature_spec.sha256)), dtype=torch.uint8))
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, hidden_dim, padding_idx=0) for cardinality in atom_cardinalities]
        )
        self.in_degree_embedding = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.out_degree_embedding = nn.Embedding(64, hidden_dim, padding_idx=0)
        self.spatial_embedding = nn.Embedding(self.spatial_pos_max_clip + 2, num_heads, padding_idx=0)
        self.direct_bond_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, num_heads, padding_idx=0) for cardinality in bond_cardinalities]
        )
        # edge_input has one extra index because the raw Cython output uses
        # -1 for padding and the collator shifts it to 0.
        self.path_bond_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 1, num_heads, padding_idx=0) for cardinality in bond_cardinalities]
            if feature_spec is None else []
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

        if self.edge_bias_mode in {"direct", "direct_plus_path"}:
            direct_edge = batch.attn_edge_type
            direct_bias = 0.0
            for field_id, embedding in enumerate(self.direct_bond_embeddings):
                direct_bias = direct_bias + embedding(direct_edge[..., field_id])
            bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + direct_bias.permute(0, 3, 1, 2)

        if self.edge_bias_mode in {"path", "direct_plus_path"}:
            path_edge = batch.edge_input
            valid_hops = path_edge.ne(0).any(dim=-1)
            path_length = valid_hops.sum(dim=3).clamp_min(1)
            path_bias = 0.0
            for field_id, embedding in enumerate(self.path_bond_embeddings):
                path_bias = path_bias + embedding(path_edge[..., field_id]).sum(dim=3)
            path_bias = path_bias / path_length.unsqueeze(-1)
            path_bias = path_bias.masked_fill(~valid_hops.any(dim=3).unsqueeze(-1), 0.0)
            bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + path_bias.permute(0, 3, 1, 2)

        bias[:, :, 1:, 0:1] = bias[:, :, 1:, 0:1] + self.graph_token_distance.unsqueeze(2)
        bias[:, :, 0:1, :] = bias[:, :, 0:1, :] + self.graph_token_distance.unsqueeze(2)
        return bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        identity = state_dict.get(prefix + "_feature_spec_identity")
        if self.feature_spec is not None:
            expected = torch.tensor(list(bytes.fromhex(self.feature_spec.sha256)), dtype=torch.uint8)
            if (not isinstance(identity, torch.Tensor) or identity.dtype != torch.uint8 or
                    not torch.equal(identity.detach().cpu(), expected)):
                raise RuntimeError("FeatureSpec mismatch: new features require matching new weights, even with strict=False")
        elif identity is not None:
            raise RuntimeError("PGT FeatureSpec weights cannot be loaded into a legacy backbone")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def encode_tokens(self, batch, *, return_attention=False, atom_embeddings=None):
        required = ("x", "in_degree", "out_degree", "spatial_pos", "attn_bias", "attn_edge_type")
        if self.edge_bias_mode != "direct":
            required += ("edge_input",)
        missing = [name for name in required if not hasattr(batch, name)]
        if missing:
            raise ValueError(f"Graphormer batch is missing fields: {missing}")
        if self.feature_spec is not None:
            if (getattr(batch, "feature_schema_version", None) != self.feature_spec.version or
                    getattr(batch, "feature_spec_sha256", None) != self.feature_spec.sha256):
                raise ValueError("Batch FeatureSpec mismatch; old caches are not PGT inputs")
        elif getattr(batch, "feature_schema_version", FEATURE_SCHEMA_VERSION) not in (None, FEATURE_SCHEMA_VERSION):
            raise ValueError("Explicit FeatureSpec is required for non-legacy batches")
        x = batch.x.long()
        batch_size, n_nodes = x.shape[:2]
        node_feature = self._encode_atoms(x, batch.in_degree.long(), batch.out_degree.long())
        if atom_embeddings is not None:
            if atom_embeddings.shape != node_feature.shape or atom_embeddings.device != node_feature.device:
                raise ValueError("Atom embedding override must match the complete padded input")
            node_feature = atom_embeddings
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
        attentions = []
        for layer in self.layers:
            result = layer(sequence, attn_bias, key_padding_mask, node_padding_mask, return_attention=return_attention)
            if return_attention:
                sequence, attention = result
                attentions.append(attention)
            else:
                sequence = result
        sequence = self.final_norm(sequence)
        atoms = sequence[:, 1:, :].masked_fill(node_padding_mask.unsqueeze(-1), 0.0)
        return BackboneTokens(sequence[:, 0, :], atoms, ~node_padding_mask, tuple(attentions))

    def forward(self, batch):
        """Preserve the graph-only interface and legacy checkpoint key layout."""
        return self.encode_tokens(batch).graph_token
