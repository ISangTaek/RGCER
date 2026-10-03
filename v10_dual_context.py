"""V10-S2: four context information flows and two additional controls.

No query labels, query/query attention, or scenario-specific routing. The S1
implementations are kept unchanged; these are proposed molecular adaptations.
"""
from dataclasses import replace
import math

import torch
from torch import nn
from torch.nn import functional as F

import v10_context_models as base

REUSED = ('MODERNNCA', 'TABR', 'TABM', 'INP', 'BSA_TNP', 'DANP')
NEW = ('DCR_PARALLEL', 'DCR_CONTEXT_FIRST', 'DCR_RETRIEVAL_FIRST', 'DCR_CONTEXT_METRIC')
CONTROLS = ('BSA_MSE', 'JOINT_MLP_CAP')
CANDIDATES = REUSED + NEW


def canonical_context(e):
    """Canonical order makes top-k ties independent of the input permutation."""
    order = sorted(range(len(e.context_rows)), key=lambda i: (
        e.context_rows[i].task, e.context_rows[i].canonical,
        e.context_rows[i].group, e.context_rows[i].sample_id))
    return replace(e, context_rows=[e.context_rows[i] for i in order],
                   context_features=e.context_features[order],
                   context_functions=e.context_functions[order],
                   context_labels=e.context_labels[order])


def legal_context_pairs(rows, device):
    return torch.tensor([[a.sample_id != b.sample_id and a.canonical != b.canonical
                          and a.group != b.group for b in rows] for a in rows],
                        dtype=torch.bool, device=device)


class Retrieval(nn.Module):
    """TabR label/value equation with identity-stable ties and masked support."""
    def __init__(self, hidden):
        super().__init__()
        self.key = nn.Linear(hidden, hidden)
        self.label = nn.Linear(1, hidden)
        self.correction = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden, bias=False))

    def forward(self, c, q, labels, *, legal=None, metric=None):
        kc, kq = self.key(c), self.key(q)
        difference = kq[:, None, :] - kc[None, :, :]
        distance = difference.square().sum(-1) if metric is None else (
            (difference * metric).square().sum(-1) / kc.shape[-1])
        allowed = torch.ones_like(distance, dtype=torch.bool) if legal is None else legal
        base.require(allowed.shape == distance.shape and allowed.dtype == torch.bool,
                     'retrieval mask shape/type')
        distance = distance.masked_fill(~allowed, float('inf'))
        # Stable sorting, preceded by canonical identity order, fixes exact ties.
        indices = distance.argsort(dim=-1, stable=True)[:, :min(96, len(c))]
        selected = allowed.gather(1, indices)
        logits = -distance.gather(1, indices)
        nonempty = selected.any(-1, keepdim=True)
        logits = torch.where(nonempty, logits, torch.zeros_like(logits))
        weights = logits.softmax(-1) * selected.to(logits.dtype)
        value = self.label(labels)[indices] + self.correction(
            kq[:, None, :] - kc[indices])
        return (weights[..., None] * value).sum(1), int((~nonempty).sum().item())


class Relation(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embed = nn.Linear(hidden + 2, hidden)
        self.blocks = nn.ModuleList([base.KRBlock(hidden) for _ in range(3)])

    def forward(self, c, q, labels, raw_c, raw_q):
        hc = self.embed(torch.cat((c, labels, torch.ones_like(labels)), -1))
        hq = self.embed(torch.cat((q, q.new_zeros(len(q), 2)), -1))
        for block in self.blocks:
            hc, hq = block(hc, hq, raw_c, raw_q)
        return hc, hq


class DualContext(base.Inputs):
    def __init__(self, *args, kind, **kwargs):
        super().__init__(*args, **kwargs)
        base.require(kind in NEW, 'DCR information flow')
        self.kind = kind
        self.project = nn.Linear(self.input_dim, self.hidden)
        self.retrieval = Retrieval(self.hidden)
        if kind == 'DCR_CONTEXT_METRIC':
            self.summary = base.mlp(self.hidden + 1, self.hidden, self.hidden)
            self.metric = nn.Linear(self.hidden, self.hidden)
        else:
            self.relation = Relation(self.hidden)
        self.decoder = base.mlp(3 * self.hidden, self.hidden, 1)
        self.empty_context_rows = 0

    def forward(self, episode):
        episode.validate(self.feature_dim, len(self.source_tasks), self.metadata_dim)
        e = canonical_context(episode)
        c, q = self.inputs(e)
        uc, uq = self.project(c), self.project(q)
        self.empty_context_rows = 0
        if self.kind == 'DCR_CONTEXT_METRIC':
            z = self.summary(torch.cat((uc, e.context_labels), -1)).mean(0, keepdim=True)
            metric = F.softplus(self.metric(z)) + 1e-4
            r, _ = self.retrieval(uc, uq, e.context_labels, metric=metric)
            return self.decoder(torch.cat((uq, z.expand(len(q), -1), r), -1))
        if self.kind == 'DCR_CONTEXT_FIRST':
            ac, aq = self.relation(uc, uq, e.context_labels, c, q)
            r, _ = self.retrieval(ac, aq, e.context_labels)
        else:
            r, _ = self.retrieval(uc, uq, e.context_labels)
            if self.kind == 'DCR_RETRIEVAL_FIRST':
                rc, self.empty_context_rows = self.retrieval(
                    uc, uc, e.context_labels,
                    legal=legal_context_pairs(e.context_rows, uc.device))
                ac, aq = self.relation(uc + rc, uq + r, e.context_labels, c, q)
            else:
                ac, aq = self.relation(uc, uq, e.context_labels, c, q)
        return self.decoder(torch.cat((uq, r, aq), -1))


class BSAMSE(base.BSATNP):
    """Same S1 architecture, including unused scale output; mean MSE only."""
    def loss(self, e, labels):
        base.check_training_labels(e, labels)
        return F.mse_loss(self(e), labels)


def parallel_parameter_count(input_dim, hidden):
    # Projection + retrieval + relation embedding + 3 KRBlocks + single decoder.
    # Each KRBlock also has one learned scalar distance bandwidth.
    return input_dim * hidden + 31 * hidden * hidden + 43 * hidden + 4


def capacity_width(input_dim, hidden):
    target = parallel_parameter_count(input_dim, hidden)
    # Two equal hidden layers: w^2 + (input_dim+3)w + 1 parameters.
    root = (math.sqrt((input_dim + 3)**2 + 4 * (target - 1)) - input_dim - 3) / 2
    choices = {max(1, math.floor(root)), max(1, math.ceil(root))}
    width = min(choices, key=lambda w: (abs(w*w + (input_dim+3)*w + 1 - target), w))
    actual = width*width + (input_dim+3)*width + 1
    base.require(abs(actual/target - 1) <= .05, 'capacity control mismatch')
    return width, target


class JointMLPCap(base.Inputs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width, self.target_parameter_count = capacity_width(self.input_dim, self.hidden)
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.width), nn.SiLU(),
                                 nn.Linear(self.width, self.width), nn.SiLU(), nn.Linear(self.width, 1))

    def forward(self, e):
        return self.net(self.inputs(e)[1])


def build(name, feature_dim, source_tasks, metadata_dim, hidden=64):
    args = (feature_dim, source_tasks, metadata_dim)
    if name in NEW:
        return DualContext(*args, hidden=hidden, kind=name)
    base.require(name in CONTROLS, 'new trainable S2 method only')
    return (BSAMSE if name == 'BSA_MSE' else JointMLPCap)(*args, hidden=hidden)


def describe(model):
    result = dict(parameters=sum(p.numel() for p in model.parameters()), loss='MSE')
    if isinstance(model, JointMLPCap):
        result.update(width=model.width, target_parameters=model.target_parameter_count)
    return result


def diagnostics(model):
    return dict(empty_context_retrieval_rows=getattr(model, 'empty_context_rows', 0))
