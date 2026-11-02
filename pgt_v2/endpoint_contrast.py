"""N5: matched endpoint contrasts on contextual atom representations."""
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


METHODS = ('N5', 'N5-Shuffle')
FIELDS = ('organism', 'route', 'measurement', 'population')


def endpoint_design(metadata, counts):
    """Give each available one-field contrast equal weight within a target."""
    attributes = [tuple(row[field] for field in FIELDS) for row in metadata]
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (len(attributes),) or not np.all(np.isfinite(counts) & (counts > 0)):
        raise ValueError('N5 requires the visible training count of each endpoint')
    reliability = counts / (counts + 32)
    columns = []
    for values in zip(*attributes):
        vocabulary = sorted(set(values))
        columns.append(np.eye(len(vocabulary), dtype=np.float32)[[vocabulary.index(v) for v in values]])
    reference = np.zeros((len(attributes), len(attributes)), dtype=np.float32)
    # Missing species or route cannot establish a matched experimental condition.
    comparable = [('unspecified' not in organism and route != 'unreported')
                  for organism, route, _, _ in attributes]
    for target, conditions in enumerate(attributes):
        if not comparable[target]:
            continue
        by_field = defaultdict(list)
        for peer, other in enumerate(attributes):
            if not comparable[peer]:
                continue
            differences = [i for i, (a, b) in enumerate(zip(conditions, other)) if a != b]
            if len(differences) == 1:
                by_field[differences[0]].append(peer)
        for peers in by_field.values():
            reference[target, peers] = reliability[peers] / (len(by_field) * reliability[peers].sum())
    return torch.from_numpy(np.concatenate(columns, axis=1)), torch.from_numpy(reference)


class MatchedEndpointContrast(nn.Module):
    def __init__(self, dim, metadata, counts, *, shuffle=False):
        super().__init__()
        features, reference = endpoint_design(metadata, counts)
        if shuffle:
            # Reassign the entire semantic design, including its reference graph.
            # A cyclic reassignment within each count band preserves its membership.
            order = np.argsort(counts, kind='stable')
            permutation = np.arange(len(order))
            rng = np.random.default_rng(20261102)
            for band in np.array_split(order, min(3, max(1, len(order) // 2))):
                cycle = rng.permutation(band)
                permutation[cycle] = np.roll(cycle, 1)
            features = features[permutation]
            reference = reference[permutation][:, permutation]
        self.register_buffer('features', features)
        self.register_buffer('reference', reference)
        self.register_buffer('isolated', (reference.sum(1) == 0).float())
        self.query = nn.Linear(features.shape[1], 16, bias=False)
        self.key = nn.Linear(dim, 16, bias=False)
        self.value = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.value.weight)

    def forward(self, atoms, mask, *, return_aux=False):
        queries = F.normalize(self.query(self.features), dim=-1)
        keys = F.normalize(self.key(atoms), dim=-1)
        logits = 4.0 * torch.einsum('td,bnd->btn', queries, keys)
        attention = logits.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        uniform = mask.to(atoms.dtype) / mask.sum(-1, keepdim=True)
        reference = self.reference @ attention + self.isolated[None, :, None] * uniform[:, None]
        signed_weight = attention - reference
        contribution = signed_weight * self.value(atoms).transpose(1, 2)
        details = dict(target_attention=attention, reference_attention=reference,
                       prompt_atom_contribution=contribution) if return_aux else {}
        return contribution.sum(-1), details
