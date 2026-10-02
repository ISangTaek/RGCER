"""Target-prioritized full graph tuning, optionally with endpoint-private residuals.

The projection is a minibatch Euclidean gradient constraint, not an AdamW or
generalization guarantee. Source tasks never traverse the private adapters.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.nn import functional as F

from v9_c2a_source_stats import targets
from v9_cost_probe import require
from loss import QuantileRegressionLoss


METHODS = ('TPO_FT', 'TPRS')
SPEC = dict(private_rank=4, private_scale=1., private_layers='last_half',
            private_projections='Q,V,FFN0,FFN3', private_lr=.001, head_lr=.001,
            weight_decay=.00001, gradient_rule='asymmetric_project_then_cap_at_target_norm',
            gradient_blocks='each_encoder_layer_and_remaining_encoder_parameters',
            clipping='separate_encoder_target_and_source_scopes_each_norm_1',
            gradient_arithmetic='float64_reductions_model_dtype_updates')


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.A = nn.Parameter(torch.empty(4, module.in_features, dtype=module.weight.dtype))
        self.B = nn.Parameter(torch.zeros(module.out_features, 4, dtype=module.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, value):
        return F.linear(F.linear(value, self.A), self.B)


class TargetGraph(nn.Module):
    def __init__(self, base, source, method):
        super().__init__()
        require(method in METHODS, 'target-priority method')
        self.method = method
        self.encoder = deepcopy(base.encoder).requires_grad_(True)
        self.decoders = deepcopy(base.decoders)
        self.decoders.update(deepcopy(source.heads))
        self.decoders.requires_grad_(True)
        self.target_tasks = list(base.task_name)
        self.source_tasks = list(source.tasks)
        self.task_name = self.target_tasks + self.source_tasks
        self.task_num = len(self.task_name)
        require(len(set(self.task_name)) == len(self.task_name), 'disjoint task scopes')
        self.private = nn.ModuleDict()
        self._active = None
        if method == 'TPRS':
            selected = {n: m for n, m in targets(self.encoder).items()
                        if int(n.split('.')[1]) >= len(self.encoder.backbone.layers)//2}
            with torch.random.fork_rng(devices=[]):
                # CPU initialization must not reset the GPU dropout stream.
                torch.random.default_generator.manual_seed(42)
                for task in self.target_tasks:
                    self.private[task] = nn.ModuleDict({n.replace('.', '__'): Residual(m) for n, m in selected.items()})
            self.private.to(next(self.encoder.parameters()).device)
            for name, module in selected.items():
                key = name.replace('.', '__')
                def hook(mod, args, result, key=key):
                    require(self._active in self.task_name, 'explicit encoder task required')
                    if self._active in self.private:
                        return result + self.private[self._active][key](args[0])
                    return result
                module.register_forward_hook(hook)

    def forward(self, inputs, task_name=None, return_aux=False):
        require(task_name in self.task_name and self._active is None, 'explicit task/no nested encoder')
        self._active = task_name
        try:
            h = self.encoder(inputs)
        finally:
            self._active = None
        raw = self.decoders[task_name](h)
        out = {task_name: raw}
        if not return_aux:
            return out
        return out, dict(final_raw=raw, base_raw=raw, route_raw=raw, final_representation=h,
                         base_representation=h, route_representation=h, null_weight=raw.new_ones((len(raw), 1)))


def build(base, source, method, encoder_lr, device):
    require(encoder_lr in (.0001, .001), 'fixed encoder lr')
    model = TargetGraph(base, source, method).to(device)
    groups = [dict(params=list(model.encoder.parameters()), lr=encoder_lr, scope='encoder'),
              dict(params=list(model.decoders.parameters()), lr=.001, scope='heads')]
    if model.private:
        groups.append(dict(params=list(model.private.parameters()), lr=.001, scope='private'))
    return model, torch.optim.AdamW(groups, weight_decay=.00001)


def block_name(name):
    parts = name.split('.')
    return '.'.join(parts[:4]) if parts[:3] == ['encoder', 'backbone', 'layers'] else 'encoder.other'


def merge_block(target, auxiliary):
    require(len(target) == len(auxiliary) and bool(target), 'gradient block dimensions')
    pairs = [(t, a) for t, a in zip(target, auxiliary) if t is not None or a is not None]
    require(bool(pairs), 'empty gradient block')
    tensors = [v for pair in pairs for v in pair if v is not None]
    require(all(torch.isfinite(v).all().item() for v in tensors), 'finite block gradients')
    zero = tensors[0].new_zeros((), dtype=torch.float64)
    tt = sum((t.double().square().sum() for t, _ in pairs if t is not None), zero)
    aa = sum((a.double().square().sum() for _, a in pairs if a is not None), zero)
    dot = sum(((t.double()*a.double()).sum() for t, a in pairs if t is not None and a is not None), zero)
    coefficient = min(float(dot/tt), 0.) if float(tt) > 0 else 0.
    adjusted = []
    for t, a in zip(target, auxiliary):
        if t is None and a is None:
            adjusted.append(None)
        elif float(tt) == 0:
            adjusted.append(torch.zeros_like(t if t is not None else a))
        else:
            value = torch.zeros_like(t) if a is None else a.clone()
            if t is not None:
                value.add_(t, alpha=-coefficient)
            adjusted.append(value)
    pp = sum((a.double().square().sum() for a in adjusted if a is not None), zero)
    scale = min(1., math.sqrt(float(tt/pp))) if float(pp) > 0 else 1.
    final_dot = sum(((t.double()*a.double()).sum()*scale for t, a in zip(target, adjusted)
                     if t is not None and a is not None), zero)
    final_norm = math.sqrt(float(pp))*scale
    target_norm = math.sqrt(float(tt))
    tolerance = 1e-6*max(math.sqrt(float(tt*aa)), 1e-12)
    require(float(final_dot) >= -tolerance and final_norm <= target_norm*(1+1e-6)+1e-12, 'gradient constraint')
    merged = [None if t is None and a is None else
              (torch.zeros_like(a) if t is None else t.clone()).add_(a, alpha=scale)
              for t, a in zip(target, adjusted)]
    return merged, dict(target_norm=target_norm, auxiliary_norm=math.sqrt(float(aa)),
                        dot_before=float(dot), dot_after=float(final_dot), auxiliary_norm_after=final_norm,
                        projection_coefficient=coefficient, auxiliary_scale=scale, tolerance=tolerance)


def step(trainer, setting, batch, task, epoch, source, item, optimizer, device):
    model = trainer.model
    optimizer.zero_grad(set_to_none=True)
    if setting == 'ToxAcute':
        target_loss, _ = trainer._training_step(batch, task, epoch)
    else:
        scaler = trainer.scalers[task]
        target_loss = QuantileRegressionLoss().compute_loss(model(batch, task_name=task)[task],
            (batch.y.reshape(-1, 1)-scaler['mean'])/scaler['std'])
    require(torch.isfinite(target_loss).item(), 'finite target loss')
    target_loss.backward()
    saved = {n: p.grad.detach() if p.grad is not None else None for n, p in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[0] if str(device).startswith('cuda') else []):
        torch.manual_seed(42000000+item['step'])
        ab = source.batch(item['task'], item['indices'], device)
        scaler = source.scalers[item['task']]
        auxiliary_loss = QuantileRegressionLoss().compute_loss(model(ab, task_name=item['task'])[item['task']],
            (ab.y.reshape(-1, 1)-scaler['mean'])/scaler['std'])
        require(torch.isfinite(auxiliary_loss).item(), 'finite auxiliary loss')
        auxiliary_loss.backward()
    blocks = {}
    scopes = dict(encoder=[], target=[], source=[])
    for name, p in model.named_parameters():
        if name.startswith('encoder.'):
            blocks.setdefault(block_name(name), []).append((name, p))
            scopes['encoder'].append(p)
        elif name.startswith('private.') or name.split('.')[1] in model.target_tasks:
            require(p.grad is None, 'auxiliary reached target-private parameters')
            p.grad = saved[name]
            scopes['target'].append(p)
        else:
            require(saved[name] is None, 'target reached source head')
            scopes['source'].append(p)
    diagnostics = {}
    for name, parameters in blocks.items():
        if all(saved[n] is None and p.grad is None for n, p in parameters):
            continue
        grads, diagnostics[name] = merge_block([saved[n] for n, _ in parameters], [p.grad for _, p in parameters])
        for (_, p), gradient in zip(parameters, grads):
            p.grad = gradient
    norms = {scope: float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
             for scope, parameters in scopes.items()}
    optimizer.step()
    return dict(target=float(target_loss.detach()), auxiliary=float(auxiliary_loss.detach()),
                gradient_norms=norms, blocks=diagnostics, auxiliary_private_gradient=False)


def parameter_counts(model):
    return {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ('encoder', 'decoders', 'private')}
