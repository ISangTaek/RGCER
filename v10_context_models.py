"""Ten V10 regression architectures and four matched controls.

These are molecular/multitask adaptations, not benchmark reproductions.
Exact differences and primary references live in docs/v10_method_mapping.md.
Prediction accepts Episode only; posterior labels enter loss() only on train.
"""
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from v10_functional_transfer import (Episode, FunctionalContextRidge,
    fit_functional_ridge, ridge_coefficients, require)

CANDIDATES = ('LAMEL', 'MODERNNCA', 'TABR', 'TABM', 'INP', 'DNP',
              'BSA_TNP', 'KERNELICL', 'DANP', 'FCR')
CONTROLS = ('TARGET_MLP', 'FCR_NO_FUNCTIONS', 'FCR_TARGET_ONLY', 'RIDGE')
ANALYTIC = ('LAMEL', 'RIDGE')
TARGET_ONLY = ('TARGET_MLP', 'FCR_TARGET_ONLY', *ANALYTIC)


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


def squared_distance(a, b):
    # Direct differences avoid cancellation/negative distances near zero.
    return (a[:, None, :] - b[None, :, :]).square().sum(-1)


def gaussian(raw):
    mean, scale = raw.chunk(2, -1)
    return mean, .1 + .9 * F.softplus(scale)


def kl(q, p):
    qm, qs = q; pm, ps = p
    return (torch.log(ps / qs) + (qs.square() + (qm-pm).square()) /
            (2 * ps.square()) - .5).sum(-1).mean()


def nll(raw, y):
    mean, scale = gaussian(raw)
    return (.5*((y-mean)/scale).square() + scale.log() + .5*math.log(2*math.pi)).mean()


def fixed_noise(keys, dim, like, draws=16):
    """Antithetic MC, independent of device, task order and query batching."""
    arrays = []
    for key in keys:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % (2**63-1)
        z = torch.randn(draws//2, dim, generator=torch.Generator().manual_seed(seed))
        arrays.append(torch.cat((z, -z), 0))
    return torch.stack(arrays, 1).to(like)


class Inputs(nn.Module):
    def __init__(self, feature_dim, source_tasks, metadata_dim, hidden=64):
        super().__init__()
        self.feature_dim, self.metadata_dim = feature_dim, metadata_dim
        self.source_tasks, self.hidden = tuple(source_tasks), hidden
        self.input_dim = feature_dim + 2*len(source_tasks) + metadata_dim

    def inputs(self, e):
        require(isinstance(e, Episode), 'Episode required')
        e.validate(self.feature_dim, len(self.source_tasks), self.metadata_dim)
        mask = e.context_features.new_ones(1, len(self.source_tasks))
        if e.task in self.source_tasks: mask[0, self.source_tasks.index(e.task)] = 0
        def join(x, p):
            return torch.cat((x, p*mask, mask.expand(len(x), -1), e.metadata.expand(len(x), -1)), -1)
        return join(e.context_features, e.context_functions), join(e.query_features, e.query_functions)

    def loss(self, e, labels):
        check_training_labels(e, labels)
        return F.mse_loss(self(e), labels)


def check_training_labels(e, labels):
    require(all(r.split == 'train' for r in e.query_rows), 'posterior/loss accepts train queries only')
    require(labels.shape == (len(e.query_rows), 1) and labels.device == e.query_features.device
            and labels.dtype == e.query_features.dtype and bool(torch.isfinite(labels).all()), 'training labels')


class ModernNCA(Inputs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.embedding = mlp(self.input_dim, self.hidden, self.hidden)

    def forward(self, e):
        c, q = self.inputs(e)
        return (-squared_distance(self.embedding(q), self.embedding(c))).softmax(-1) @ e.context_labels


class TabR(Inputs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); h = self.hidden
        self.encoder = nn.Linear(self.input_dim, h)
        self.key = nn.Linear(h, h)
        self.label = nn.Linear(1, h)
        self.correction = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h, bias=False))
        self.predictor = nn.Sequential(nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, e):
        c, q = self.inputs(e); hc, hq = self.encoder(c), self.encoder(q)
        kc, kq = self.key(hc), self.key(hq)
        distance, indices = squared_distance(kq, kc).topk(min(96, len(c)), largest=False, sorted=True)
        values = self.label(e.context_labels)[indices] + self.correction(kq[:, None, :] - kc[indices])
        retrieval = ((-distance).softmax(-1)[..., None]*values).sum(1)
        return self.predictor(hq + retrieval)


class EnsembleLinear(nn.Module):
    def __init__(self, i, o, k=32, first=False):
        super().__init__(); self.weight = nn.Parameter(torch.empty(o, i))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.r = nn.Parameter(torch.empty(k, i)); self.s = nn.Parameter(torch.ones(k, o))
        self.bias = nn.Parameter(torch.zeros(k, o))
        if first: nn.init.normal_(self.r, mean=0, std=1)
        else: nn.init.ones_(self.r)

    def forward(self, x):
        return F.linear(x*self.r, self.weight)*self.s + self.bias


class TabM(Inputs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); h = self.hidden
        self.layers = nn.Sequential(EnsembleLinear(self.input_dim, h, first=True), nn.ReLU(),
                                    EnsembleLinear(h, h), nn.ReLU())
        self.head_weight = nn.Parameter(torch.randn(32, h)/math.sqrt(h))
        self.head_bias = nn.Parameter(torch.zeros(32))

    def branches(self, e):
        _, q = self.inputs(e)
        return (self.layers(q[:, None, :])*self.head_weight).sum(-1)+self.head_bias

    def forward(self, e): return self.branches(e).mean(-1, keepdim=True)

    def loss(self, e, labels):
        check_training_labels(e, labels)
        return (self.branches(e)-labels).square().mean()


class TargetMLP(Inputs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); self.net = mlp(self.input_dim, self.hidden, 1)

    def forward(self, e): return self.net(self.inputs(e)[1])


class KRBlock(nn.Module):
    """Shared context/context and query/context subnetworks; no query/query path."""
    def __init__(self, h, kernel=True):
        super().__init__(); self.q = nn.Linear(h, h); self.k = nn.Linear(h, h)
        self.v = nn.Linear(h, h); self.out = nn.Linear(h, h)
        self.norm1, self.norm2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = mlp(h, 2*h, h)
        self.log_bandwidth = nn.Parameter(torch.tensor(0.)) if kernel else None

    def forward(self, c, q, xc, xq):
        k, v = self.k(c), self.v(c)
        def update(z, x):
            score = self.q(z) @ k.T / math.sqrt(k.shape[-1])
            if self.log_bandwidth is not None:
                score = score - squared_distance(x, xc)/(xc.shape[-1]*(F.softplus(self.log_bandwidth)+1e-4))
            z = self.norm1(z + self.out(score.softmax(-1) @ v))
            return self.norm2(z + self.ff(z))
        return update(c, xc), update(q, xq)


class BSATNP(Inputs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); h = self.hidden
        self.embed = mlp(self.input_dim+2, h, h)
        self.blocks = nn.ModuleList([KRBlock(h) for _ in range(3)])
        self.decoder = mlp(h, h, 2)

    def distribution(self, e):
        c, q = self.inputs(e)
        hc = self.embed(torch.cat((c, e.context_labels, torch.ones_like(e.context_labels)), -1))
        hq = self.embed(torch.cat((q, q.new_zeros(len(q), 2)), -1))
        for block in self.blocks: hc, hq = block(hc, hq, c, q)
        return self.decoder(hq)

    def forward(self, e): return self.distribution(e)[:, :1]

    def loss(self, e, labels):
        check_training_labels(e, labels); return nll(self.distribution(e), labels)


class KernelICL(Inputs):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw); h = self.hidden
        self.embed = mlp(self.input_dim+2, h, h)
        self.blocks = nn.ModuleList([KRBlock(h, kernel=False) for _ in range(2)])
        self.log_gamma = nn.Parameter(torch.tensor(0.))

    def symmetric_embeddings(self, e):
        c, q = self.inputs(e)
        hc = self.embed(torch.cat((c, e.context_labels, torch.ones_like(e.context_labels)), -1))
        # Both context keys and queries are re-embedded as label-free queries.
        x = torch.cat((c, q)); z = self.embed(torch.cat((x, x.new_zeros(len(x), 2)), -1))
        for block in self.blocks: hc, z = block(hc, z, c, x)
        return z[:len(c)], z[len(c):]

    def forward(self, e):
        c, q = self.symmetric_embeddings(e)
        weights = (-squared_distance(q, c)*(F.softplus(self.log_gamma)+1e-4)).softmax(-1)
        return weights @ e.context_labels


class DimensionAggregator(nn.Module):
    """DANP-style scalar/position tokens, attention, x pooling and y token."""
    def __init__(self, h):
        super().__init__(); self.h = h; self.value = nn.Linear(1, h, bias=False)
        self.attn = nn.MultiheadAttention(h, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(h); self.combine = nn.Linear(2*h, h)

    def forward(self, x, y):
        positions = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)[:, None]
        divisor = torch.exp(torch.arange(0, self.h, 2, device=x.device, dtype=x.dtype)*(-math.log(10000)/self.h))
        pe = x.new_zeros(x.shape[-1]+1, self.h)
        pe[:-1, 0::2] = torch.sin(positions*divisor); pe[:-1, 1::2] = torch.cos(positions*divisor)
        pe[-1, 0::2] = 1  # y uses swapped cosine/sine, output dimension 0
        tokens = self.value(torch.cat((x, y), -1)[..., None]) + pe
        z = self.norm(tokens + self.attn(tokens, tokens, tokens, need_weights=False)[0])
        return self.combine(torch.cat((z[:, :-1].mean(1), z[:, -1]), -1))


class LatentProcess(Inputs):
    """INP, DNP and DANP paths share only Gaussian/ELBO utilities."""
    def __init__(self, *args, kind, **kw):
        super().__init__(*args, **kw); self.kind = kind; h = self.hidden
        self.pair = mlp(self.input_dim+1, h, h)
        self.knowledge = mlp(self.metadata_dim, h, h) if kind == 'INP' else None
        self.global_dist = mlp(2*h if kind == 'INP' else h, h, 2*h)
        if kind == 'DNP':
            self.distance = nn.Sequential(nn.Linear(self.input_dim, h), nn.LeakyReLU(.1), nn.Linear(h, h))
            self.local = mlp(self.input_dim+1, h, 2*h)
        if kind == 'DANP':
            self.dab = DimensionAggregator(h)
            self.blocks = nn.ModuleList([KRBlock(h, kernel=False) for _ in range(2)])
        self.decoder = mlp(self.input_dim+h+(h if kind in ('DNP','DANP') else 0), h, 2)

    def global_distribution(self, x, y, metadata):
        encoded = self.dab(x, y) if self.kind == 'DANP' else self.pair(torch.cat((x, y), -1))
        summary = encoded.mean(0, keepdim=True)
        if self.knowledge is not None: summary = torch.cat((summary, self.knowledge(metadata)), -1)
        return gaussian(self.global_dist(summary))

    def local_distribution(self, c, y, q):
        weights = (-squared_distance(self.distance(q), self.distance(c)).clamp_min(1e-12).sqrt()/math.sqrt(self.hidden)).softmax(-1)
        mean, logvar = self.local(torch.cat((c, y), -1)).chunk(2, -1)
        # Literal equation 6, stably evaluated; no unproved OOD-prior claim.
        log_variance = torch.logsumexp(weights[..., None]*logvar[None, :, :], dim=1)
        return weights @ mean, torch.exp(.5*log_variance)

    def deterministic_path(self, c, cy, q):
        if self.kind != 'DANP': return q.new_empty(len(q), 0)
        hc, hq = self.dab(c, cy), self.dab(q, q.new_zeros(len(q), 1))
        for block in self.blocks: hc, hq = block(hc, hq, c, q)
        return hq

    def forward(self, e):
        c, q = self.inputs(e); pg = self.global_distribution(c, e.context_labels, e.metadata)
        zg = pg[0][None] + pg[1][None]*fixed_noise(['global:'+e.task], self.hidden, q)
        zg = zg.expand(-1, len(q), -1)
        if self.kind == 'DNP':
            pl = self.local_distribution(c, e.context_labels, q)
            noise = fixed_noise(['local:'+e.task+':'+r.sample_id for r in e.query_rows], self.hidden, q)
            extra = pl[0][None] + pl[1][None]*noise
        else: extra = self.deterministic_path(c, e.context_labels, q)[None].expand(16, -1, -1)
        raw = self.decoder(torch.cat((q[None].expand(16, -1, -1), zg, extra), -1))
        return raw[..., :1].mean(0)

    def loss(self, e, labels):
        check_training_labels(e, labels); c, q = self.inputs(e)
        pg = self.global_distribution(c, e.context_labels, e.metadata)
        allx, ally = torch.cat((c, q)), torch.cat((e.context_labels, labels))
        qg = self.global_distribution(allx, ally, e.metadata)
        zg = (qg[0]+qg[1]*torch.randn_like(qg[0])).expand(len(q), -1)
        regularizer = kl(qg, pg)/len(q)
        if self.kind == 'DNP':
            pl = self.local_distribution(c, e.context_labels, q)
            ql = gaussian(self.local(torch.cat((q, labels), -1)))
            extra = ql[0] + ql[1]*torch.randn_like(ql[0]); regularizer = regularizer + kl(ql, pl)
            for layer in self.distance:
                if isinstance(layer, nn.Linear):
                    s = torch.linalg.svdvals(layer.weight)
                    regularizer = regularizer + .01*(F.relu(.1-s[-1]).square()+F.relu(s[0]-2).square())
        else: extra = self.deterministic_path(c, e.context_labels, q)
        return nll(self.decoder(torch.cat((q, zg, extra), -1)), labels) + regularizer


class FCR(FunctionalContextRidge):
    def loss(self, e, labels):
        check_training_labels(e, labels); return F.mse_loss(self(e), labels)


def build(name, feature_dim, source_tasks, metadata_dim, hidden=64):
    require(name in CANDIDATES+CONTROLS and name not in ANALYTIC, 'trainable method')
    args = (feature_dim, source_tasks, metadata_dim)
    cls = dict(MODERNNCA=ModernNCA, TABR=TabR, TABM=TabM, BSA_TNP=BSATNP,
               KERNELICL=KernelICL, TARGET_MLP=TargetMLP)
    if name in ('INP', 'DNP', 'DANP'): return LatentProcess(*args, hidden=hidden, kind=name)
    if name.startswith('FCR'): return FCR(*args, hidden=hidden)
    return cls[name](*args, hidden=hidden)


def closed_form(name, e, config):
    e.validate(e.context_features.shape[1], e.context_functions.shape[1], e.metadata.shape[1])
    c, q, y = e.context_features.double(), e.query_features.double(), e.context_labels.double()
    if name == 'LAMEL':
        fit = fit_functional_ridge(c, e.context_functions.double(), y,
                                  source_penalty=(.1, 1.)[config], residual_penalty=(1., 10.)[config])
        return fit.predict(q, e.query_functions.double()).to(e.query_features)
    require(name == 'RIDGE', 'analytic method')
    xm, ym = c.mean(0, keepdim=True), y.mean(0, keepdim=True)
    return (ym+(q-xm)@ridge_coefficients(c-xm, y-ym, c.new_tensor((.1, 1.)[config]))).to(e.query_features)
