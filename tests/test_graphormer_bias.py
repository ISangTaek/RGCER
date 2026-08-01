import copy

import torch

from architecture.graphormer_backbone import MolecularGraphormerBackbone
from dataset import DataCollator
from preprocess_data import get_graph_data_from_smiles


def test_same_topology_has_different_chemical_edge_bias():
    torch.manual_seed(7)
    single = get_graph_data_from_smiles("CC", 1.0)
    double = get_graph_data_from_smiles("C=C", 1.0)
    collator = DataCollator()
    single_batch = collator([single])
    double_batch = collator([double])
    model = MolecularGraphormerBackbone(
        hidden_dim=16, num_heads=2, num_layers=1, ffn_dim=32, dropout=0.0
    ).eval()

    # Their shortest-path geometry is identical; only chemical edge fields differ.
    assert torch.equal(single_batch.spatial_pos, double_batch.spatial_pos)
    bias_single = model._encode_attention_bias(single_batch, 2)
    bias_double = model._encode_attention_bias(double_batch, 2)
    assert not torch.allclose(bias_single, bias_double)

    representation_single = model(single_batch)
    representation_double = model(double_batch)
    assert not torch.allclose(representation_single, representation_double)


def test_bias_is_stable_for_zero_padding_fields():
    graph = get_graph_data_from_smiles("CC", 1.0)
    collator = DataCollator()
    batch = collator([graph])
    altered = copy.deepcopy(batch)
    altered.attn_edge_type.zero_()
    altered.edge_input.zero_()
    model = MolecularGraphormerBackbone(
        hidden_dim=16, num_heads=2, num_layers=1, ffn_dim=32, dropout=0.0
    ).eval()
    output = model(altered)
    assert torch.isfinite(output).all()
