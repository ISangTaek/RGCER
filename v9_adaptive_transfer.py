"""Train-internal policy calibration and full-training replay for V9-S6.

These are named, bounded graph adaptations of method components, not claims
to reproduce the papers. No controller in this module accepts outer labels.
"""
from copy import deepcopy
import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from v9_cost_probe import require, digest
from v9_method_portfolio import PortfolioGraph
from loss import QuantileRegressionLoss

GRADIENT_METHODS = ('AUTO_LAMBDA_P', 'OL_AUX_P', 'SLGRAD_P', 'GIST_HOLDOUT_P', 'GIST_SUBSPACE_P')
CANDIDATES = (*GRADIENT_METHODS, 'FORKMERGE_P', 'WISE_FT_P', 'LINES_P', 'TIES_P', 'PROTECTED_GATE')
CONTROLS = ('TARGET_FULL', 'TARGET_LAST', 'TARGET_FROZEN', 'JOINT_FULL')
ENGINES = (*CONTROLS, *GRADIENT_METHODS, 'FORK_TARGET', 'FORK_JOINT')
METHODS = (*CANDIDATES, *CONTROLS, 'TARGET_ADAPTIVE')
GRID = (0., .25, .5, .75, 1.)
SPEC = dict(schema='adaptive_transfer_graph_v1', candidates=list(CANDIDATES), controls=list(CONTROLS),
            lrs=[.0001, .001], head_lr=.001, weight_decay=.00001, max_epochs=40,
            inner_fraction=.2, split_seed=4206, meta_examples_per_task=1,
            probe='last_transformer_ffn_output_weight', auxiliary_initial_weight=.1,
            controller_step=.05, ol_horizon=4, selection_fraction=.5, subspace_energy=.95,
            ties_density=.2, coefficient_grid=list(GRID), gate_ridge=.1,
            policy='train_internal_calibration_then_frozen_schedule_full_train_refit',
            full_refit_target_sampling='original_full_pass_each_epoch',
            inner_sampling='filter_full_batch_plan_repeat_fit_batch_only_if_empty',
            clipping='separate_encoder_target_source_norm_1', warmup_epochs=0,
            selection='inner_macro_RMSE_first_epoch_then_lower_LR', test_access=False)


def partition(rows, fraction=.2):
    """Label-blind deterministic scaffold partition with every task on both sides."""
    require(rows and all(r['split'] == 'train' for r in rows), 'train-only inner split')
    tasks = list(dict.fromkeys(r['task'] for r in rows))
    groups = sorted({r['group'] for r in rows}); require(len(groups) >= 2, 'inner groups')
    best = None
    for attempt in range(1024):
        order = list(groups); random.Random(4206+attempt).shuffle(order)
        chosen = set(order[:max(1, min(len(groups)-1, round(len(groups)*fraction)))])
        by_task = {t: [r for r in rows if r['task'] == t] for t in tasks}
        ratios = [sum(r['group'] in chosen for r in by_task[t])/len(by_task[t]) for t in tasks]
        if any(v == 0 or v == 1 for v in ratios):
            continue
        score = sum(abs(v-fraction) for v in ratios)
        if best is None or score < best[0]:
            best = score, attempt, chosen
    require(best is not None, 'cannot form inner split with all tasks; stop, do not change split')
    fit = {}; meta = {}
    for t in tasks:
        subset = [r for r in rows if r['task'] == t]
        fit[t] = [i for i, r in enumerate(subset) if r['group'] not in best[2]]
        meta[t] = [i for i, r in enumerate(subset) if r['group'] in best[2]]
    # Canonical identity is independently checked, even if group metadata is bad.
    fc = {r['canonical'] for r in rows if r['group'] not in best[2]}
    mc = {r['canonical'] for r in rows if r['group'] in best[2]}
    require(not fc & mc, 'inner canonical overlap')
    return dict(fit=fit, meta=meta, groups=sorted(best[2]), attempt=best[1],
                train_sha256=digest(rows), fraction=fraction)


def inner_scalers(rows, split):
    result = {}
    for task, indices in split['fit'].items():
        values = [r['label'] for r in rows if r['task'] == task]
        x = [values[i] for i in indices]; mean = math.fsum(x)/len(x)
        std = math.sqrt(math.fsum((v-mean)**2 for v in x)/len(x))
        result[task] = dict(mean=mean, std=std if std > 1e-12 else 1., count=len(x))
    return result


def fit_plan(plan, split):
    result = []
    for item in plan:
        allowed = set(split['fit'][item['task']])
        chosen = [i for i in item['indices'] if i in allowed]
        if not chosen:
            chosen = split['fit'][item['task']][:len(item['indices'])]
        result.append(dict(task=item['task'], indices=chosen))
    return result


def source_mask(source, item, training, split, setting):
    held = [r for r in training if r['group'] in set(split['groups'])]
    canonical = {r['canonical'] for r in held}; groups = set(split['groups'])
    return [r['canonical'] not in canonical and (setting == 'B' or r['group'] not in groups)
            for r in (source.by_task[item['task']][i] for i in item['indices'])]


def build(base, source, kind, lr, device):
    require(kind in ENGINES and lr in SPEC['lrs'], 'fixed engine/lr')
    model = PortfolioGraph(base, source, 'CONFIG').to(device)
    model.profile = 'FROZEN' if kind == 'TARGET_FROZEN' else 'LAST' if kind == 'TARGET_LAST' else 'FULL'
    last = len(model.encoder.backbone.layers)-1
    for name, p in model.encoder.named_parameters():
        p.requires_grad_(model.profile == 'FULL' or (model.profile == 'LAST' and
                         (name.startswith(f'backbone.layers.{last}.') or name.startswith(('backbone.final_norm.', 'readout.')))))
    groups = [dict(params=[p for p in model.encoder.parameters() if p.requires_grad], lr=lr, scope='encoder'),
              dict(params=list(model.decoders.parameters()), lr=.001, scope='heads')]
    return model, torch.optim.AdamW(groups, weight_decay=.00001)


def training_mode(model):
    model.train()
    if model.profile != 'FULL':
        model.encoder.eval()
        if model.profile == 'LAST':
            model.encoder.backbone.layers[-1].train()
            model.encoder.backbone.final_norm.train()
            model.encoder.readout.train()


def probe_parameter(model):
    layer = model.encoder.backbone.layers[-1].ffn
    linear = [m for m in layer if isinstance(m, nn.Linear)][-1]
    require(linear.weight.requires_grad, 'trainable shared probe')
    return linear.weight


def losses(model, batch, task, scaler):
    raw = model(batch, task_name=task)[task]
    y = (batch.y.reshape(-1, 1)-scaler['mean'])/scaler['std']
    # Exact individual QuantileRegressionLoss values, not a different proxy loss.
    return torch.stack([QuantileRegressionLoss().compute_loss(raw[i:i+1], y[i:i+1]) for i in range(len(raw))])


def gradient_rows(values, parameter):
    return torch.stack([torch.autograd.grad(v, parameter, retain_graph=True)[0].detach().reshape(-1)
                        for v in values])


def cosine_scores(values, target):
    return F.normalize(values.double(), dim=1, eps=1e-12) @ F.normalize(target.double().reshape(1, -1), dim=1, eps=1e-12).T


def subspace_scores(values, targets):
    """GIST spectral projector and max projected cosine; no whitening substitution."""
    x = targets.double()
    _, sigma, vh = torch.linalg.svd(x, full_matrices=False)
    if float(sigma.square().sum()) == 0:
        return values.new_zeros(len(values)).double()
    energy = sigma.square().cumsum(0)/sigma.square().sum()
    rank = int(torch.searchsorted(energy, .95).item())+1
    projection = vh[:rank].T
    a = F.normalize(values.double()@projection, dim=1, eps=1e-12)
    b = F.normalize(x@projection, dim=1, eps=1e-12)
    return (a@b.T).amax(1)


def hard_weights(scores, allowed):
    idx = [i for i, ok in enumerate(allowed) if ok and float(scores[i]) > 0]
    idx.sort(key=lambda i: (-float(scores[i]), i))
    chosen = idx[:math.ceil(sum(allowed)*.5)]
    return [1./len(chosen) if i in chosen else 0. for i in range(len(allowed))]


def learn_weights(model, kind, target_batch, target_task, target_scaler, ab, atask, ascaler,
                  meta_batches, allowed, state, lr):
    """Only internal training batches enter this controller. Optimizer never steps here."""
    require(kind in GRADIENT_METHODS and any(allowed), 'controller kind/population')
    model.eval(); parameter = probe_parameter(model)
    auxiliary = losses(model, ab, atask, ascaler)
    ga = gradient_rows(auxiliary, parameter)
    gm = []
    for task, batch, scaler in meta_batches:
        gm.append(gradient_rows(losses(model, batch, task, scaler), parameter))
    gm = torch.cat(gm)
    target = losses(model, target_batch, target_task, target_scaler).mean()
    gt = torch.autograd.grad(target, parameter)[0].detach().reshape(-1)
    mask = torch.tensor(allowed, device=ga.device, dtype=torch.bool)
    average = ga[mask].mean(0)
    if kind == 'AUTO_LAMBDA_P':
        prior = state.setdefault(atask, dict(weight=.1, signals=[]))
        saved = parameter.detach().clone()
        try:
            with torch.no_grad():
                parameter.add_(-lr*(gt+prior['weight']*average).reshape_as(parameter))
            future = []
            for task, batch, scaler in meta_batches:
                v = losses(model, batch, task, scaler).mean()
                future.append(torch.autograd.grad(v, parameter)[0].detach().reshape(-1))
            signal = float(cosine_scores(average[None], torch.stack(future).mean(0))[0, 0])
        finally:
            with torch.no_grad():
                parameter.copy_(saved)
        prior['weight'] = min(1., max(0., prior['weight']+.05*signal))
        weights = [prior['weight']/sum(allowed) if ok else 0. for ok in allowed]
    elif kind == 'OL_AUX_P':
        prior = state.setdefault(atask, dict(weight=.1, signals=[]))
        signal = math.tanh(float((gt.double()*average.double()).sum()) /
                           max(float(target.detach())*float(auxiliary.detach()[mask].mean()), 1e-8))
        prior['signals'].append(signal)
        if len(prior['signals']) == 4:
            prior['weight'] = min(1., max(0., prior['weight']+.05*math.fsum(prior['signals'])/4))
            prior['signals'] = []
        weights = [prior['weight']/sum(allowed) if ok else 0. for ok in allowed]
    elif kind == 'SLGRAD_P':
        scores = (ga.double()@gm.double().mean(0)).clamp_min(0)*mask
        total = float(scores.sum())
        weights = (scores/total).tolist() if total > 0 else [0.]*len(allowed)
    else:
        scores = (cosine_scores(ga, gm.mean(0)).reshape(-1) if kind == 'GIST_HOLDOUT_P'
                  else subspace_scores(ga, gm))
        weights = hard_weights(scores, allowed)
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in weights) and sum(weights) <= 1.000001,
            'bounded controller weights')
    return weights


def update(model, optimizer, batch, task, scaler, source, auxiliary, weights, device, step):
    """Paired target updates; zero auxiliary weights exactly recover target-only."""
    training_mode(model); optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(420000+step)
    target = losses(model, batch, task, scaler).mean()
    require(torch.isfinite(target).item(), 'finite target loss')
    target.backward(); auxiliary_loss = 0.
    require(len(weights) == len(auxiliary['indices']) and all(math.isfinite(w) and 0 <= w <= 1 for w in weights)
            and sum(weights) <= 1.000001, 'auxiliary weight vector')
    if sum(weights) > 0:
        with torch.random.fork_rng(devices=[0] if str(device).startswith('cuda') else []):
            torch.manual_seed(42000000+step)
            ab = source.batch(auxiliary['task'], auxiliary['indices'], device)
            values = losses(model, ab, auxiliary['task'], source.scalers[auxiliary['task']])
            loss = (values*values.new_tensor(weights)).sum()
            require(torch.isfinite(loss).item(), 'finite auxiliary loss')
            loss.backward(); auxiliary_loss = float(loss.detach())
    scopes = dict(encoder=[p for p in model.encoder.parameters() if p.requires_grad],
                  target=[p for t in model.target_tasks for p in model.decoders[t].parameters()],
                  source=[p for t in model.source_tasks for p in model.decoders[t].parameters()])
    norms = {k: float(torch.nn.utils.clip_grad_norm_(p, 1., error_if_nonfinite=True)) for k, p in scopes.items()}
    optimizer.step()
    return dict(target=float(target.detach()), auxiliary=auxiliary_loss, norms=norms,
                accepted=sum(w > 0 for w in weights), weight_sum=math.fsum(weights))


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def interpolate(first, second, alpha, depth=False):
    require(set(first) == set(second) and 0 <= alpha <= 1, 'merge input')
    layers = [int(k.split('.')[3]) for k in first if k.startswith('encoder.backbone.layers.')]
    last = max(layers) if layers else 0
    result = {}
    for key, x in first.items():
        y = second[key]; require(x.shape == y.shape and x.dtype == y.dtype, 'merge tensor')
        if not x.is_floating_point():
            require(torch.equal(x, y), 'merge immutable buffer'); result[key] = x.clone(); continue
        scale = alpha
        if depth and key.startswith('encoder.'):
            if key.startswith('encoder.backbone.layers.'):
                scale *= int(key.split('.')[3])/last if last else 1.
            elif not key.startswith(('encoder.backbone.final_norm.', 'encoder.readout.')):
                scale = 0.
        result[key] = x+scale*(y-x)
    return result


def ties_merge(initial, first, second, parameter_names, density=.2):
    keys = [k for k in initial if k in parameter_names and
            (k.startswith('encoder.') or k.startswith('decoders.') and k.split('.')[1] in parameter_names['target_tasks'])]
    # A global top-k over the target predictor, not per-tensor padding of the density.
    deltas = torch.stack([torch.cat([(s[k]-initial[k]).reshape(-1) for k in keys]) for s in (first, second)])
    keep = max(1, math.ceil(deltas.shape[1]*density))
    idx = torch.argsort(deltas.abs(), dim=1, descending=True, stable=True)[:, :keep]
    mask = torch.zeros_like(deltas, dtype=torch.bool).scatter_(1, idx, True)
    trimmed = deltas*mask; sign = trimmed.sum(0).sign()
    matching = (trimmed.sign() == sign[None]) & (trimmed != 0)
    merged = (trimmed*matching).sum(0)/matching.sum(0).clamp_min(1)
    result = {k: v.clone() for k, v in first.items()}; offset = 0
    for k in keys:
        n = initial[k].numel(); result[k] = initial[k]+merged[offset:offset+n].reshape_as(initial[k]); offset += n
    return result


def blend_rows(first, second, coefficients):
    require(len(first) == len(second), 'prediction blend population')
    result = []
    for x, y in zip(first, second):
        require({k: v for k, v in x.items() if k != 'prediction'} ==
                {k: v for k, v in y.items() if k != 'prediction'}, 'prediction blend identity')
        a = coefficients[x['task']]; require(0 <= a <= 1, 'prediction blend coefficient')
        result.append(dict(x, prediction=x['prediction']+a*(y['prediction']-x['prediction'])))
    return result


def gate_coefficients(first, second):
    """Low-capacity ridge shrinkage, then leave-one-group-out stability veto."""
    tasks = list(dict.fromkeys(r['task'] for r in first)); result = {}
    for task in tasks:
        pairs = [(x, y) for x, y in zip(first, second) if x['task'] == task]
        require(pairs and all(x['sample_id'] == y['sample_id'] for x, y in pairs), 'gate identity')
        def coefficient(items):
            numerator = math.fsum((y['prediction']-x['prediction'])*(x['label']-x['prediction']) for x, y in items)
            denom = math.fsum((y['prediction']-x['prediction'])**2 for x, y in items)
            return min(1., max(0., numerator/(1.1*denom))) if denom > 0 else 0.
        groups = {x['group'] for x, _ in pairs}
        alpha = coefficient(pairs)
        gains = []
        for g in groups:
            fit = [(x, y) for x, y in pairs if x['group'] != g]
            a = coefficient(fit)
            gains.append(math.fsum((x['label']-x['prediction'])**2-
                (x['label']-(x['prediction']+a*(y['prediction']-x['prediction'])))**2
                for x, y in pairs if x['group'] == g))
        # A conservative empirical veto, not a population non-inferiority guarantee.
        result[task] = alpha if len(groups) >= 2 and math.fsum(gains) > 0 and sum(v > 0 for v in gains) > len(gains)/2 else 0.
    return result
