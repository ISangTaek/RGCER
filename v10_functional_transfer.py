"""Research prototypes for function-space and support-conditioned regression.

The two-stage ridge predictor borrows the *prediction-space* construction of
LAMeL (https://doi.org/10.1039/D5DD00443H). Nonlinear pretrained source heads
are an extension, not a reproduction of its linear/graphlet implementation.
FunctionalContextRidge is our proposed differentiable episodic extension.
This module has no real-data experiment, checkpoint selection, or test access.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def require(ok, message):
    if not ok:
        raise ValueError(message)


def matrix(x, name, *, columns=None):
    require(isinstance(x, Tensor) and x.ndim == 2 and x.shape[0] > 0 and x.shape[1] > 0,
            name + ': nonempty matrix required')
    require(x.dtype in (torch.float32, torch.float64) and bool(torch.isfinite(x).all()),
            name + ': finite float32/float64 required')
    require(columns is None or x.shape[1] == columns, name + ': column count')


def compatible(reference, *others):
    require(all(x.device == reference.device and x.dtype == reference.dtype for x in others),
            'inconsistent dtype/device')


def ridge_coefficients(x: Tensor, y: Tensor, penalty: Tensor) -> Tensor:
    """Solve mean squared error + penalty * squared norm; no inverse or jitter.

    The caller centers x/y to leave the intercept unpenalized. Selecting the
    smaller primal/dual system changes only arithmetic, not the objective.
    A failed solve remains a failure; it cannot silently change regularization.
    """
    matrix(x, 'ridge x'); matrix(y, 'ridge y')
    require(x.shape[0] == y.shape[0], 'ridge row count')
    require(isinstance(penalty, Tensor) and penalty.numel() == 1
            and bool(torch.isfinite(penalty).all()) and bool(penalty > 0), 'positive finite penalty')
    compatible(x, y, penalty)
    n, d = x.shape
    if d <= n:
        gram = x.T @ x + n * penalty * torch.eye(d, dtype=x.dtype, device=x.device)
        return torch.linalg.solve(gram, x.T @ y)
    gram = x @ x.T + n * penalty * torch.eye(n, dtype=x.dtype, device=x.device)
    return x.T @ torch.linalg.solve(gram, y)


@dataclass(frozen=True)
class FunctionalFit:
    """Fitted tensors only: the held-out query has no route to the labels."""
    feature_mean: Tensor
    source_center_mean: Tensor
    intercept: Tensor
    source_coefficients: Tensor
    residual_coefficients: Tensor

    def predict(self, features: Tensor, source_predictions: Tensor) -> Tensor:
        matrix(features, 'query features', columns=self.feature_mean.numel())
        matrix(source_predictions, 'query source predictions', columns=self.source_coefficients.shape[0])
        require(features.shape[0] == source_predictions.shape[0], 'query row count')
        compatible(features, source_predictions, self.feature_mean, self.source_center_mean,
                   self.intercept, self.source_coefficients, self.residual_coefficients)
        average = source_predictions.mean(dim=1, keepdim=True)
        centered = source_predictions - average
        return (average + self.intercept
                + (centered - self.source_center_mean) @ self.source_coefficients
                + (features - self.feature_mean) @ self.residual_coefficients)


def fit_functional_ridge(features: Tensor, source_predictions: Tensor, labels: Tensor,
                         *, source_penalty=.1, residual_penalty=1.) -> FunctionalFit:
    """A centered LAMeL-style prediction transfer plus molecular residual.

    Source columns are model predictions in each source task's train-scaled
    coordinates, never measurements of the query. Target labels must be on
    the target task's training scale. No cross-database raw-unit equality is
    assumed. The residual is unconstrained; it is not mathematically
    orthogonal to the source span, and we make no such claim.
    """
    matrix(features, 'features'); matrix(source_predictions, 'source predictions')
    matrix(labels, 'labels', columns=1)
    require(features.shape[0] == source_predictions.shape[0] == labels.shape[0], 'fit row count')
    compatible(features, source_predictions, labels)
    average = source_predictions.mean(dim=1, keepdim=True)
    centered = source_predictions - average
    fmean, cmean = features.mean(0, keepdim=True), centered.mean(0, keepdim=True)
    intercept = (labels - average).mean(0, keepdim=True)
    cy = labels - average - intercept
    coef = ridge_coefficients(centered - cmean, cy, features.new_tensor(source_penalty))
    residual = cy - (centered - cmean) @ coef
    rcoef = ridge_coefficients(features - fmean, residual, features.new_tensor(residual_penalty))
    return FunctionalFit(*(x.detach().clone() for x in (fmean, cmean, intercept, coef, rcoef)))


class FrozenFunctionBank(nn.Module):
    """Immutable copied encoder and verified source heads; no target decoders.

    Caller must verify source checkpoint provenance (e.g. v9_joint_source.load)
    before construction. This small component does not replace that verifier.
    """
    def __init__(self, encoder: nn.Module, source_heads: nn.ModuleDict):
        super().__init__()
        require(isinstance(source_heads, nn.ModuleDict) and len(source_heads) > 0, 'source heads')
        self.encoder = deepcopy(encoder)
        self.heads = deepcopy(source_heads)
        self.tasks = tuple(self.heads)
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, graph_batch):
        require(not any(m.training for m in self.modules())
                and not any(p.requires_grad for p in self.parameters()), 'function bank must be frozen/eval')
        features = self.encoder(graph_batch)
        matrix(features, 'encoder features')
        outputs = []
        for task in self.tasks:
            raw = self.heads[task](features)
            matrix(raw, 'source head', columns=3)
            require(raw.shape[0] == features.shape[0], 'source head row count')
            # RGCER quantile heads store the train-scaled central output first.
            outputs.append(raw[:, :1])
        return features, torch.cat(outputs, dim=1)


@dataclass(frozen=True)
class RowIdentity:
    sample_id: str
    canonical: str
    group: str
    split: str
    task: str


@dataclass(frozen=True)
class Episode:
    task: str
    context_rows: Sequence[RowIdentity]
    query_rows: Sequence[RowIdentity]
    context_features: Tensor
    context_functions: Tensor
    context_labels: Tensor
    query_features: Tensor
    query_functions: Tensor
    metadata: Tensor

    def validate(self, feature_dim, function_dim, metadata_dim):
        require(type(self.task) is str and bool(self.task), 'episode task')
        tensors = (self.context_features, self.context_functions, self.context_labels,
                   self.query_features, self.query_functions, self.metadata)
        for value, name, columns in zip(tensors, ('context features','context functions','context labels',
                'query features','query functions','metadata'),
                (feature_dim, function_dim, 1, feature_dim, function_dim, metadata_dim)):
            matrix(value, name, columns=columns)
        compatible(tensors[0], *tensors[1:])
        require(len(self.context_rows) == self.context_features.shape[0] == self.context_functions.shape[0]
                == self.context_labels.shape[0] and len(self.context_rows) >= 2, 'context population')
        require(len(self.query_rows) == self.query_features.shape[0] == self.query_functions.shape[0],
                'query population')
        require(self.metadata.shape[0] == 1, 'one task metadata vector')
        for role, rows in (('context',self.context_rows), ('query',self.query_rows)):
            require(all(isinstance(r, RowIdentity) and all(type(v) is str and bool(v) for v in
                    (r.sample_id,r.canonical,r.group,r.split,r.task)) and r.task == self.task for r in rows),
                    role + ' row identity/task')
            require(len({r.sample_id for r in rows}) == len(rows), role + ' duplicate sample')
            require(all(r.split == 'train' if role == 'context' else r.split in ('train','validation') for r in rows),
                    role + ' split')
        for field in ('sample_id', 'canonical', 'group'):
            require(not {getattr(r,field) for r in self.context_rows}
                    & {getattr(r,field) for r in self.query_rows}, 'context/query overlap: '+field)


class FunctionalContextRidge(nn.Module):
    """One trainable episodic regressor shared by all source and target tasks.

    Source prediction features + molecular features + task metadata -> shared
    feature network -> label-conditioned feature map -> differentiable ridge.
    Query labels are deliberately absent from Episode and forward arguments.
    This is a proposed architecture, not a paper reimplementation or result.
    """
    def __init__(self, feature_dim: int, source_tasks: Sequence[str], metadata_dim: int,
                 hidden=32):
        super().__init__()
        require(type(feature_dim) is int and feature_dim > 0 and type(metadata_dim) is int
                and metadata_dim > 0 and type(hidden) is int and hidden >= 2, 'dimensions')
        require(bool(source_tasks) and all(type(t) is str and bool(t) for t in source_tasks)
                and len(set(source_tasks)) == len(source_tasks), 'unique source tasks')
        self.feature_dim, self.metadata_dim = feature_dim, metadata_dim
        self.source_tasks = tuple(source_tasks)
        self.features = nn.Sequential(nn.Linear(feature_dim+2*len(source_tasks)+metadata_dim, hidden),
                                      nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(hidden+1+metadata_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, 2*hidden))
        self.log_penalty = nn.Parameter(torch.tensor(-2.))

    def forward(self, episode: Episode) -> Tensor:
        require(isinstance(episode, Episode), 'episode required')
        episode.validate(self.feature_dim, len(self.source_tasks), self.metadata_dim)
        c, q = episode.context_features, episode.query_features
        compatible(c, next(self.parameters()))
        available = c.new_ones((1, len(self.source_tasks)))
        if episode.task in self.source_tasks:
            # A source episode must not learn to copy its own pretrained head.
            available[0, self.source_tasks.index(episode.task)] = 0
        def encode(x, functions):
            n = x.shape[0]
            return self.features(torch.cat((x, functions*available, available.expand(n,-1),
                                           episode.metadata.expand(n,-1)), dim=1))
        hc, hq = encode(c, episode.context_functions), encode(q, episode.query_functions)
        mean = episode.context_labels.mean(dim=0, keepdim=True)
        scale = episode.context_labels.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        cy = (episode.context_labels - mean)/scale
        summary = self.context(torch.cat((hc, cy, episode.metadata.expand(c.shape[0],-1)),dim=1)).mean(0,keepdim=True)
        slope, offset = summary.chunk(2, dim=1)
        zc = F.silu(hc * (1+.1*torch.tanh(slope)) + offset)
        zq = F.silu(hq * (1+.1*torch.tanh(slope)) + offset)
        zmean = zc.mean(0, keepdim=True)
        coef = ridge_coefficients(zc-zmean, cy, F.softplus(self.log_penalty)+1e-4)
        return mean + scale * ((zq-zmean) @ coef)
