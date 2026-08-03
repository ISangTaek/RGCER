from types import SimpleNamespace

import torch

from architecture.graphormer_backbone import MolecularGraphormerBackbone


def _synthetic_batch():
    edge_input = torch.zeros(1, 3, 3, 3, 4, dtype=torch.long)
    # 0 -> 1 is one hop; 0 -> 2 is a two-hop path with bond codes 1 and 2.
    edge_input[0, 0, 1, 0, :] = 1
    edge_input[0, 0, 2, 0, :] = 1
    edge_input[0, 0, 2, 1, :] = 2
    return SimpleNamespace(
        attn_bias=torch.zeros(1, 4, 4),
        spatial_pos=torch.zeros(1, 3, 3, dtype=torch.long),
        attn_edge_type=torch.zeros(1, 3, 3, 4, dtype=torch.long),
        edge_input=edge_input,
    )


def _set_path_weights(model):
    with torch.no_grad():
        model.spatial_embedding.weight.zero_()
        model.graph_token_distance.zero_()
        for embedding in model.path_bond_embeddings:
            embedding.weight.zero_()
            embedding.weight[1].fill_(1.0)
            embedding.weight[2].fill_(2.0)


def test_path_bias_averages_effective_hops_and_ignores_padding():
    model = MolecularGraphormerBackbone(
        hidden_dim=8, num_heads=2, num_layers=1, ffn_dim=16, dropout=0.0, edge_bias_mode="path"
    )
    _set_path_weights(model)
    bias = model._encode_attention_bias(_synthetic_batch(), 3)
    # Four bond fields, each field contributes 1 on the one-hop path and
    # (1 + 2) / 2 on the two-hop path.
    assert torch.allclose(bias[0, :, 1, 2], torch.full((2,), 4.0))
    assert torch.allclose(bias[0, :, 1, 3], torch.full((2,), 6.0))


def test_direct_and_path_modes_are_explicit_and_finite():
    batch = _synthetic_batch()
    direct = MolecularGraphormerBackbone(
        hidden_dim=8, num_heads=2, num_layers=1, ffn_dim=16, dropout=0.0, edge_bias_mode="direct"
    )
    path = MolecularGraphormerBackbone(
        hidden_dim=8, num_heads=2, num_layers=1, ffn_dim=16, dropout=0.0, edge_bias_mode="path"
    )
    with torch.no_grad():
        direct.spatial_embedding.weight.zero_()
        path.spatial_embedding.weight.zero_()
        direct.graph_token_distance.zero_()
        path.graph_token_distance.zero_()
        batch.attn_edge_type[0, 0, 1, :] = 1
        for embedding in direct.direct_bond_embeddings:
            embedding.weight.zero_()
            embedding.weight[1].fill_(2.0)
        for embedding in path.path_bond_embeddings:
            embedding.weight.zero_()
            embedding.weight[1].fill_(1.0)
    direct_bias = direct._encode_attention_bias(batch, 3)
    path_bias = path._encode_attention_bias(batch, 3)
    assert torch.isfinite(direct_bias).all()
    assert torch.isfinite(path_bias).all()
    assert not torch.allclose(direct_bias, path_bias)
