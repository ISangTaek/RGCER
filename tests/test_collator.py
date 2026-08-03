from preprocess_data import get_graph_data_from_smiles
from dataset import DataCollator


def test_collator_pads_all_graphormer_fields_on_cpu():
    short = get_graph_data_from_smiles("C", 1.0, sample_id="short")
    long = get_graph_data_from_smiles("CCO", 2.0, sample_id="long")
    batch = DataCollator()([short, long])

    assert batch.x.device.type == "cpu"
    assert tuple(batch.x.shape) == (2, 3, 9)
    assert tuple(batch.spatial_pos.shape) == (2, 3, 3)
    assert tuple(batch.attn_edge_type.shape) == (2, 3, 3, 4)
    assert tuple(batch.edge_input.shape) == (2, 3, 3, 8, 4)
    assert tuple(batch.padding_mask.shape) == (2, 3)
    assert batch.padding_mask[0, 1:].all()
    assert not batch.padding_mask[1].any()
    assert batch.y.shape == (2, 1)


def test_collator_returns_empty_batch_after_filtering():
    graph = get_graph_data_from_smiles("CCO", 1.0)
    batch = DataCollator(max_node_filter=1)([graph])
    assert batch.is_empty is True
    assert batch.x.numel() == 0
