"""V4 explicit fixed task knowledge; no free task-ID residual parameters."""
import numpy as np
import torch
from torch import nn

PRIOR_METHODS = ('N1P-Meta', 'N1P-Tox', 'N1P-Hybrid', 'N1P-Shuffle')


def known(value):
    value = str(value).lower()
    return value not in ('', 'unknown', 'unreported', 'unspecified', 'none') and 'unspecified' not in value


class PriorPrompts(nn.Module):
    def __init__(self, method, metadata, adjacency, dim):
        super().__init__()
        n = len(metadata)
        blocks = []
        for field in ('organism', 'route', 'measurement', 'population'):
            vocab = sorted({row[field] for row in metadata})
            blocks.append(np.eye(len(vocab), dtype=np.float32)[[vocab.index(row[field]) for row in metadata]])
        features = np.concatenate(blocks, axis=1)
        if adjacency is None or np.asarray(adjacency).shape != (n, n):
            raise ValueError('N1P requires the current visible-training task graph')
        empirical = (np.asarray(adjacency) > 0).astype(np.float32)
        np.fill_diagonal(empirical, 0)
        semantic = np.zeros((n, n), dtype=np.float32)
        for i, left in enumerate(metadata):
            for j, right in enumerate(metadata):
                if i != j and left['measurement'] == right['measurement']:
                    semantic[i, j] = sum(.5 for field in ('organism', 'route')
                        if known(left[field]) and left[field] == right[field])
        weights = np.zeros_like(empirical) if method == 'N1P-Meta' else empirical
        if method in ('N1P-Hybrid', 'N1P-Shuffle'):
            weights = np.maximum(empirical, .25 * semantic)
        permutation = np.arange(n)
        if method == 'N1P-Shuffle':
            permutation = np.random.default_rng(20261028).permutation(n)
            weights = weights[permutation][:, permutation]
        graph = weights + np.eye(n, dtype=np.float32)
        inv = graph.sum(1) ** -.5
        graph = graph * inv[:, None] * inv[None, :]
        self.register_buffer('features', torch.tensor(features))
        self.register_buffer('relation_weights', torch.tensor(weights.copy()))
        self.register_buffer('adjacency', torch.tensor(graph))
        self.register_buffer('permutation', torch.tensor(permutation))
        self.w1 = nn.Linear(features.shape[1], 32, bias=False)
        self.w2 = nn.Linear(32, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self):
        h = torch.nn.functional.gelu(self.adjacency @ self.w1(self.features))
        return .02 * self.norm(self.adjacency @ self.w2(h))
