"""Ten distinct graph-method candidates plus a target-priority control.

These are explicitly scoped graph adaptations, not reproductions of the
papers' language/vision benchmarks. Hyperparameters are fixed before screening.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

import v9_target_priority as priority
from v9_cost_probe import require
from v9_c2a_source_stats import targets
from loss import QuantileRegressionLoss

CANDIDATES = ('TPRS', 'ORTHO_LORA', 'ALIGN_LORA', 'ASE', 'MTL_LORA', 'MTLORA_BLOCK',
              'MASA', 'LIME', 'CONFIG', 'DB_MTL')
METHODS = (*CANDIDATES, 'TPO_FT')
FULL = ('TPRS', 'TPO_FT', 'CONFIG', 'DB_MTL')
SPEC = dict(schema='graph_portfolio_v1', candidates=list(CANDIDATES), controls=['TPO_FT'],
            learning_rate=.001, head_lr=.001, weight_decay=.00001, adapter_scale=1.,
            plain_rank=8, align_rank=16, align_lambda=.01, align_variance_floor=.0001,
            experts=4, mtl_rank=4, mtl_temperature=.05, ase_rank=2, ase_shared=1, ase_top_k=2,
            block_rank=4, block_routing_groups=4, spectral_lambda=.01, spectral_epsilon=.000001,
            spectral_basis='detached_SVD_reweighted_B_for_stability', balance_lambda=.01,
            masa_rank=8, masa_A_count=2, masa_layer_group=2,
            lime_experts=4, lime_threshold=.7, lime_temperature=.5, lime_route_blend=.5,
            lime_window='one_atom_no_order_dependent_ngram', lime_importance=.1, lime_kl=.01,
            db_beta=.9, db_epsilon=1e-8, db_objectives='target_domain_and_auxiliary_domain_EMA',
            priority=priority.SPEC, clipping='separate_shared_target_source_norm_1')


def param(*shape):
    value = torch.empty(*shape)
    nn.init.kaiming_uniform_(value, a=math.sqrt(5))
    return nn.Parameter(value)


def atom_mean(value, mask):
    require(value.ndim == 3 and value.shape[:2] == (len(mask), mask.shape[1]+1), 'atom activation shape')
    weight = mask.to(value.dtype).unsqueeze(-1)
    return (value[:, 1:]*weight).sum(1)/weight.sum(1).clamp_min(1)


def balance(weights, mask):
    mean = atom_mean(weights, mask).mean(0)
    return len(mean)*(mean.square().sum())-1


class Adapter(nn.Module):
    def __init__(self, nin, nout, tasks, kind):
        super().__init__(); self.kind = kind; self.nout = nout
        rank = {'ALIGN_LORA': 16, 'ASE': 2, 'MTL_LORA': 4, 'MTLORA_BLOCK': 4}.get(kind, 8)
        if kind != 'MASA':
            self.A = param(rank, nin)
        if kind in ('MTL_LORA', 'MTLORA_BLOCK'):
            self.B = nn.Parameter(torch.zeros(4, nout, rank))
        elif kind == 'ASE':
            del self.A
            self.A = nn.Parameter(torch.stack([param(rank, nin).detach() for _ in range(4)]))
            self.B = nn.Parameter(torch.zeros(4, nout, rank))
        else:
            self.B = nn.Parameter(torch.zeros(nout, rank))
        if kind == 'MTL_LORA':
            self.task_transform = nn.Parameter(torch.eye(rank).repeat(tasks, 1, 1))
            self.task_weights = nn.Parameter(torch.randn(tasks, 4)*.02)
        if kind == 'ASE':
            self.router = nn.Linear(nin, 4, bias=False)
            self.task_embedding = nn.Parameter(torch.randn(tasks, nin)*.02)
        if kind == 'MTLORA_BLOCK':
            self.router = nn.Linear(nin, 16, bias=False)
            require(nout % 4 == 0, 'routing dimension divisible by four')
        if kind == 'LIME':
            self.modulators = nn.Parameter(torch.ones(4, nout)+torch.randn(4, nout)*.01)

    def forward(self, x, original, task, mask, bank=None):
        reg = x.new_zeros(()); feature = None
        if self.kind == 'MASA':
            delta = F.linear(F.linear(x, bank.sum(0)), self.B)
        elif self.kind == 'ASE':
            logits = self.router(x+self.task_embedding[task])
            selected = logits[..., 1:].topk(2, dim=-1).indices+1
            active = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, selected, True)
            active[..., 0] = True
            weights = logits.masked_fill(~active, float('-inf')).softmax(-1)
            experts = torch.stack([F.linear(F.linear(x, a), b) for a, b in zip(self.A, self.B)], -2)
            delta = (experts*weights.unsqueeze(-1)).sum(-2)
        else:
            low = F.linear(x, self.A)
            if self.kind == 'MTL_LORA':
                low = F.linear(low, self.task_transform[task])
                weights = (self.task_weights[task]/.05).softmax(0)
                delta = F.linear(low, (self.B*weights[:, None, None]).sum(0))
            elif self.kind == 'MTLORA_BLOCK':
                weights = self.router(x).reshape(*x.shape[:-1], 4, 4).softmax(-2)
                experts = torch.stack([F.linear(low, b) for b in self.B], -2)
                delta = (experts*weights.repeat_interleave(self.nout//4, dim=-1)).sum(-2)
                reg = .01*balance(weights.mean(-1), mask)
                # Degenerate zero B has non-unique SVD vectors. Treat spectral
                # bases/weights as fixed within this step; gradients reach B.
                adjusted = []
                for b in self.B:
                    with torch.no_grad():
                        _, sigma, vh = torch.linalg.svd(b.float(), full_matrices=False)
                        w = torch.exp(-sigma/sigma.mean().clamp_min(1e-6))
                        transform = (vh.T*w)@vh
                    adjusted.append(b@transform.to(b))
                reg = reg+.01*sum((a.T@b).square().sum() for i, a in enumerate(adjusted) for b in adjusted[i+1:])
            else:
                delta = F.linear(low, self.B)
                if self.kind == 'ALIGN_LORA':
                    feature = atom_mean(low, mask)
                if self.kind == 'LIME':
                    logits = (.5*F.layer_norm(original, (original.shape[-1],))[..., :4]+
                              .5*F.layer_norm(delta, (delta.shape[-1],))[..., :4])/.5
                    all_weights = logits.softmax(-1)
                    active = all_weights >= .7*all_weights.amax(-1, keepdim=True)
                    weights = all_weights*active
                    weights = weights/weights.sum(-1, keepdim=True)
                    delta = delta*(weights@self.modulators)
                    mean = atom_mean(all_weights, mask).mean(0)
                    reg = .1*balance(all_weights, mask)+.01*(mean*(mean.clamp_min(1e-8).log()+math.log(4))).sum()
        return delta, reg, feature


class PortfolioGraph(nn.Module):
    def __init__(self, base, source, kind):
        super().__init__(); self.method = kind
        self.encoder = deepcopy(base.encoder).requires_grad_(kind in FULL)
        self.decoders = deepcopy(base.decoders); self.decoders.update(deepcopy(source.heads))
        self.decoders.requires_grad_(True)
        self.target_tasks = list(base.task_name); self.source_tasks = list(source.tasks)
        self.task_name = self.target_tasks+self.source_tasks; self.task_num = len(self.task_name)
        require(len(set(self.task_name)) == len(self.task_name), 'disjoint task scope')
        self.adapters = nn.ModuleDict(); self.bank = nn.ParameterDict(); self.private = nn.ModuleDict()
        self._active = None; self._mask = None; self._regs = []; self._features = {}
        if kind in FULL:
            return
        selected = targets(self.encoder)
        if kind == 'MTLORA_BLOCK':
            selected = {f'layers.{i}.{name}': getattr(layer, name)
                        for i, layer in enumerate(self.encoder.backbone.layers) for name in ('attention', 'ffn')}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            for name, module in selected.items():
                key = name.replace('.', '__')
                nin = module.in_features if isinstance(module, nn.Linear) else self.encoder.backbone.hidden_dim
                nout = module.out_features if isinstance(module, nn.Linear) else nin
                self.adapters[key] = Adapter(nin, nout, self.task_num, kind)
                bank_key = None
                if kind == 'MASA':
                    parts = name.split('.'); bank_key = f'g{int(parts[1])//2}__'+('__'.join(parts[2:]))
                    if bank_key not in self.bank:
                        self.bank[bank_key] = nn.Parameter(torch.stack([param(8, nin).detach() for _ in range(2)]))
                def hook(mod, args, original, key=key, bank_key=bank_key):
                    require(self._active in self.task_name, 'explicit portfolio task')
                    delta, reg, feature = self.adapters[key](args[0], original, self.task_name.index(self._active),
                                                             self._mask, None if bank_key is None else self.bank[bank_key])
                    if self.training:
                        self._regs.append(reg)
                        if feature is not None:
                            self._features[key] = feature
                    return original+delta
                module.register_forward_hook(hook)

    def train(self, mode=True):
        super().train(mode)
        if self.method not in FULL:
            self.encoder.eval()
        return self

    def forward(self, batch, task_name=None, return_aux=False):
        require(task_name in self.task_name and self._active is None, 'explicit task/no nested call')
        self._active = task_name; self._mask = batch.node_mask
        self._regs = []; self._features = {}
        try:
            h = self.encoder(batch)
        finally:
            self._active = None; self._mask = None
        raw = self.decoders[task_name](h); output = {task_name: raw}
        if not return_aux:
            return output
        return output, dict(final_raw=raw, base_raw=raw, route_raw=raw, final_representation=h,
                            base_representation=h, route_representation=h, null_weight=raw.new_ones((len(raw), 1)))

    def regularization(self):
        return sum(self._regs, next(self.parameters()).new_zeros(()))


def build(base, source, kind, encoder_lr, device):
    require(kind in METHODS and encoder_lr == .001, 'fixed portfolio method/lr')
    if kind in ('TPRS', 'TPO_FT'):
        return priority.build(base, source, kind, encoder_lr, device)
    model = PortfolioGraph(base, source, kind).to(device)
    shared = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('decoders.')]
    groups = [dict(params=shared, lr=.001, scope='shared'), dict(params=list(model.decoders.parameters()), lr=.001, scope='heads')]
    return model, torch.optim.AdamW(groups, weight_decay=.00001)


def dot(x, y):
    return sum((a.double()*b.double()).sum() for a, b in zip(x, y))


def config_gradient(target, source):
    """Two-objective ConFIG direction; explicit finite degenerate cases."""
    nt = math.sqrt(float(dot(target, target))); ns = math.sqrt(float(dot(source, source)))
    if nt == 0 or ns == 0:
        return [a+b for a, b in zip(target, source)]
    unit = [a/nt+b/ns for a, b in zip(target, source)]
    norm = math.sqrt(float(dot(unit, unit)))
    if norm < 1e-8:
        return [torch.zeros_like(a) for a in target]
    unit = [u/norm for u in unit]
    length = float(dot(target, unit)+dot(source, unit))
    return [u*length for u in unit]


def symmetric_projection(target, source):
    product = float(dot(target, source)); nt = float(dot(target, target)); ns = float(dot(source, source))
    if product < 0 and nt > 0 and ns > 0:
        return [a+b-a*(product/nt)-b*(product/ns) for a, b in zip(target, source)]
    return [a+b for a, b in zip(target, source)]


def alignment(first, second):
    require(set(first) == set(second) and bool(first), 'alignment layers')
    values = []
    for key in first:
        a, b = first[key], second[key]
        ma, mb = a.mean(0), b.mean(0)
        va, vb = a.var(0, unbiased=False).clamp_min(.0001), b.var(0, unbiased=False).clamp_min(.0001)
        values.append(.25*((va+(ma-mb).square())/vb+(vb+(ma-mb).square())/va-2).mean())
    return torch.stack(values).mean()


def algorithm_state(optimizer):
    return {name: [v.detach().cpu().clone() for v in pair] for name, pair in getattr(optimizer, '_v9_ema', {}).items()}


def step(trainer, setting, batch, task, epoch, source, item, optimizer, device):
    model = trainer.model; kind = model.method
    if kind in ('TPRS', 'TPO_FT'):
        r = priority.step(trainer, setting, batch, task, epoch, source, item, optimizer, device)
        return dict(target=r['target'], auxiliary=r['auxiliary'], regularization=0., gradient_norms=r['gradient_norms'],
                    rule='TARGET_PRIORITY', diagnostics=r['blocks'])
    optimizer.zero_grad(set_to_none=True)
    if setting == 'ToxAcute':
        target_loss, _ = trainer._training_step(batch, task, epoch)
    else:
        scaler = trainer.scalers[task]
        target_loss = QuantileRegressionLoss().compute_loss(model(batch, task_name=task)[task],
            (batch.y.reshape(-1, 1)-scaler['mean'])/scaler['std'])
    require(torch.isfinite(target_loss).item(), 'finite target loss')
    reg_t = model.regularization(); features_t = model._features
    objective = torch.log(target_loss+1e-8) if kind == 'DB_MTL' else target_loss+reg_t
    objective.backward(retain_graph=kind == 'ALIGN_LORA')
    saved = {n: p.grad.detach() if p.grad is not None else None for n, p in model.named_parameters() if p.requires_grad}
    optimizer.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[0] if str(device).startswith('cuda') else []):
        torch.manual_seed(42000000+item['step'])
        ab = source.batch(item['task'], item['indices'], device); scaler = source.scalers[item['task']]
        auxiliary_loss = QuantileRegressionLoss().compute_loss(model(ab, task_name=item['task'])[item['task']],
            (ab.y.reshape(-1, 1)-scaler['mean'])/scaler['std'])
        require(torch.isfinite(auxiliary_loss).item(), 'finite auxiliary loss')
        reg_a = model.regularization()
        if kind == 'ALIGN_LORA':
            reg_a = reg_a+.01*alignment(features_t, model._features)
        objective = torch.log(auxiliary_loss+1e-8) if kind == 'DB_MTL' else auxiliary_loss+reg_a
        objective.backward()
    shared = []; scopes = dict(encoder=[], target=[], source=[])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('decoders.'):
            if name.split('.')[1] in model.target_tasks:
                require(p.grad is None, 'auxiliary target head leakage'); p.grad = saved[name]; scopes['target'].append(p)
            else:
                require(saved[name] is None, 'target source head leakage'); scopes['source'].append(p)
        else:
            scopes['encoder'].append(p)
            if saved[name] is not None or p.grad is not None:
                t = torch.zeros_like(p) if saved[name] is None else saved[name]
                a = torch.zeros_like(p) if p.grad is None else p.grad
                shared.append((name, p, t, a))
    tg = [v[2] for v in shared]; ag = [v[3] for v in shared]
    require(all(torch.isfinite(g).all().item() for g in tg+ag), 'finite shared gradients')
    if kind == 'CONFIG':
        merged = config_gradient(tg, ag)
    elif kind == 'ORTHO_LORA':
        merged = [None]*len(shared)
        for factor in ('A', 'B'):
            idx = [i for i, (n, _, _, _) in enumerate(shared) if n.endswith('.'+factor)]
            values = symmetric_projection([tg[i] for i in idx], [ag[i] for i in idx])
            for i, value in zip(idx, values):
                merged[i] = value
        require(all(g is not None for g in merged), 'Ortho A/B scopes')
    elif kind == 'DB_MTL':
        old = getattr(optimizer, '_v9_ema', {})
        for n, p, t, a in shared:
            prior = old.get(n, (torch.zeros_like(t), torch.zeros_like(a)))
            old[n] = (.9*prior[0]+.1*t, .9*prior[1]+.1*a)
        optimizer._v9_ema = old
        et = [old[n][0] for n, _, _, _ in shared]; ea = [old[n][1] for n, _, _, _ in shared]
        nt = math.sqrt(float(dot(et, et))); na = math.sqrt(float(dot(ea, ea))); scale = max(nt, na)
        merged = [(t/(nt+1e-8)+a/(na+1e-8))*scale for t, a in zip(et, ea)]
    else:
        merged = [t+a for t, a in zip(tg, ag)]
    diagnostics = dict(target_norm=math.sqrt(float(dot(tg, tg))), auxiliary_norm=math.sqrt(float(dot(ag, ag))),
                       dot_before=float(dot(tg, ag)), dot_target_after=float(dot(tg, merged)), dot_auxiliary_after=float(dot(ag, merged)))
    for (_, p, _, _), value in zip(shared, merged):
        p.grad = value
    norms = {scope: float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)) for scope, parameters in scopes.items()}
    optimizer.step()
    # No graph references survive a completed step or evaluation.
    model._features = {}; model._regs = []
    return dict(target=float(target_loss.detach()), auxiliary=float(auxiliary_loss.detach()),
                regularization=float((reg_t+reg_a).detach()), gradient_norms=norms, rule=kind, diagnostics=diagnostics)


def parameter_counts(model):
    result = {key: 0 for key in ('encoder', 'decoders', 'private', 'adapters', 'bank', 'trainable', 'total')}
    for name, p in model.named_parameters():
        result[name.split('.')[0]] += p.numel(); result['total'] += p.numel()
        if p.requires_grad:
            result['trainable'] += p.numel()
    return result
