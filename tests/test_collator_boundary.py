"""§9.3: spatial_pos_max_clip is the largest *visible* distance, not masked."""

import torch

from dataset import DataCollator


class _Item:
    """Just enough of a graph record for the collator's attention path."""

    def __init__(self, spatial_pos):
        size = len(spatial_pos)
        self.spatial_pos = torch.tensor(spatial_pos, dtype=torch.long)
        self.x = torch.zeros(size, 1, dtype=torch.long)
        self.y = torch.zeros(1, dtype=torch.float)
        self.in_degree = torch.zeros(size, dtype=torch.long)
        self.out_degree = torch.zeros(size, dtype=torch.long)
        self.attn_edge_type = torch.zeros((size, size, 4), dtype=torch.long)
        self.edge_input = torch.zeros((size, size, 6, 4), dtype=torch.long)

    def __len__(self):
        return int(self.x.size(0))


def test_distance_equal_to_clip_is_not_masked():
    clip = 8
    items = [
        _Item(
            [
                [0, 1, clip],
                [1, 0, clip],
                [clip, clip, 0],
            ]
        )
    ]
    bias = DataCollator(spatial_pos_max_clip=clip)(items).attn_bias[0]
    visible = torch.isfinite(bias)
    # Pairs at exactly max_clip attend in both directions...
    assert bool(visible[1, 3]) and bool(visible[3, 1])
    assert bool(visible[2, 3]) and bool(visible[3, 2])


def test_distances_beyond_clip_and_disconnected_pairs_are_masked():
    clip = 4
    items = [_Item([[0, clip, clip + 7], [clip, 0, 510], [clip + 7, 510, 0]])]
    node_bias = DataCollator(spatial_pos_max_clip=clip)(items).attn_bias[0][1:4, 1:4]
    # d == clip stays visible; d == clip+7 and disconnected (510) are masked.
    assert torch.isfinite(node_bias[0, 1]) and torch.isfinite(node_bias[1, 0])
    assert float(node_bias[0, 2]) == float("-inf")
    assert float(node_bias[2, 0]) == float("-inf")
    assert float(node_bias[1, 2]) == float("-inf")
    assert float(node_bias[2, 1]) == float("-inf")


def test_padded_keys_remain_masked_regardless_of_distance_semantics():
    items = [_Item([[0, 1], [1, 0]])]
    bias = DataCollator(spatial_pos_max_clip=20)(items).attn_bias[0]
    assert bool(torch.isneginf(bias[1:3, 3:]).all())
