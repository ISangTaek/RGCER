import numpy as np
import torch
from torch_geometric.data import Batch

from baselines.data import load_human3_smoke
from baselines.features import afp_graph, afp_schema
from baselines.models.afp import AttentiveFPRegressor
from baselines.models.dmpnn import ChempropDMPNN
from baselines.models.toxacol import ToxACoLNet, endpoint_feature_matrix, task_adjacency


def test_afp_real_graph_and_no_edge_molecule_are_legal(baseline_datastore):
    data = load_human3_smoke(baseline_datastore)
    graph = afp_graph(data.train.graph_records[0])
    schema = afp_schema()
    model = AttentiveFPRegressor(schema.atom_dim, schema.bond_dim)
    output = model(Batch.from_data_list([graph]))
    assert output.shape == (1, 3)
    record = dict(data.train.graph_records[0])
    record["x"] = record["x"][:1]
    record["spatial_pos"] = torch.zeros((1, 1), dtype=torch.long)
    record["attn_edge_type"] = torch.zeros((1, 1, 4), dtype=torch.long)
    output = model(Batch.from_data_list([afp_graph(record)]))
    assert output.shape == (1, 3)


def test_chemprop_native_dmpnn_forward_and_two_layer_head(chemprop_source):
    model = ChempropDMPNN(chemprop_source, torch.device("cpu"))
    assert model(["CC", "O"]).shape == (2, 3)
    linears = [module for module in model.head if isinstance(module, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [(300, 300), (300, 3)]


def test_toxacol_shapes_and_graph_constant_policy():
    endpoint, schema = endpoint_feature_matrix()
    labels = np.full((20, 59), np.nan)
    labels[:, 0] = 1.0
    labels[:, 1] = 1.0
    adjacency, audit = task_adjacency(labels)
    assert audit["undirected_edges"] == 0
    assert adjacency.shape == (59, 59)
    model = ToxACoLNet(adjacency, endpoint)
    assert model(torch.zeros((2, 1024))).shape == (2, 59)
    assert schema["species"][11:14] == ["man", "women", "human"]
