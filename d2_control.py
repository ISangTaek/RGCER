"""Same-architecture initialization/plasticity/supervision controls.

No data discovery, holdout access, remote execution or permission is performed
here. S1 is a bounded learning-rate comparison, not a novel-method screen.
"""
import math
import random

import torch
from torch import nn

from architecture.Graphormer import Encoder
from architecture.prediction_heads import TaskPredictionHead
from reproducibility import stable_seed, state_dict_sha256
from v9_cost_probe import digest, require

CONDITIONS = ('FJ', 'PT', 'PJ', 'RT', 'RJ')
SETTINGS = ('ToxAcute', 'A', 'B')
SPEC = dict(source_passes=40, source_batch_size=32, target_batch_size=32,
            head_lr=.001, weight_decay=1e-5, grad_clip=1., source_weight=1.,
            loss='train_standardized_mse', lr_schedule='constant',
            validation_every=1024, early_validation=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            early_stop=False, selection='earliest_minimum_validation_endpoint_macro_rmse',
            encoder_dropout='train_in_all_five_conditions', source_heads='fresh_per_task',
            target_sampling='complete_shuffled_passes_cycled_to_source_step_count')


def jobs():
    return [dict(id=f'{s}_{c}_{r}_s42', setting=s, condition=c, rate=r, seed=42,
                 encoder_lr=0. if c == 'FJ' else {'low': .0001, 'high': .001}[r])
            for s in SETTINGS for c in CONDITIONS
            for r in (('fixed',) if c == 'FJ' else ('low', 'high'))]


def check_job(job):
    require(type(job) is dict and digest(job) in {digest(j) for j in jobs()}, 'outside D2-S1 matrix')


def encoder(args, seed):
    """Construct random weights directly, without loading or perturbing a source."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stable_seed(seed, 'd2', 'encoder'))
        return Encoder(args.a_heads, [], args.a_layers, args.hidden_dim, .1, 0.,
                       args.mid_dim, args.hidden_dim, .1, args.hidden_dim, 'cpu',
                       edge_bias_mode=args.edge_bias_mode,
                       spatial_pos_max_clip=args.spatial_pos_clip)


class Control(nn.Module):
    def __init__(self, args, pretrained, targets, sources, condition, seed):
        super().__init__()
        require(condition in CONDITIONS and type(seed) is int and seed in range(42, 47), 'condition/seed')
        require(targets and sources and not set(targets) & set(sources), 'disjoint nonempty task roles')
        self.condition = condition
        self.target_tasks, self.source_tasks = list(targets), list(sources)
        self.encoder = encoder(args, seed)
        random_sha = state_dict_sha256(self.encoder)
        if condition in ('FJ', 'PT', 'PJ'):
            self.encoder.load_state_dict(pretrained, strict=True)
            require(state_dict_sha256(self.encoder) != random_sha, 'pretrained equals random initialization')
        self.encoder.requires_grad_(condition != 'FJ')
        self.decoders = nn.ModuleDict()
        # Head RNG is per task and independent of encoder initialization and role.
        for task in list(targets) + list(sources):
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(stable_seed(seed, 'd2', 'head', task))
                self.decoders[task] = TaskPredictionHead(
                    args.hidden_dim, mode='point', head_hidden_dim=args.head_hidden_dim,
                    dropout=args.head_dropout)
        if condition in ('PT', 'RT'):
            for task in sources:
                self.decoders[task].requires_grad_(False)

    @property
    def joint(self):
        return self.condition in ('FJ', 'PJ', 'RJ')

    def forward(self, batch, task):
        require(task in self.target_tasks or (self.joint and task in self.source_tasks),
                'source supervision forbidden for target-only control')
        return self.decoders[task](self.encoder(batch)).reshape(-1)


def optimizer(model, encoder_lr, spec=SPEC):
    groups = []
    if model.condition != 'FJ':
        require(encoder_lr > 0, 'positive encoder LR')
        groups.append(dict(params=list(model.encoder.parameters()), lr=encoder_lr, role='encoder'))
    groups.append(dict(params=[p for p in model.decoders.parameters() if p.requires_grad],
                       lr=spec['head_lr'], role='heads'))
    return torch.optim.AdamW(groups, weight_decay=spec['weight_decay'])


def full_pass(counts, size, seed, role, epoch):
    require(type(size) is int and size > 0 and counts and
            all(type(n) is int and n > 0 for n in counts.values()), 'positive batch/counts')
    batches = []
    for task, n in counts.items():
        order = list(range(n))
        random.Random(stable_seed(seed, role, epoch, task)).shuffle(order)
        batches.extend(dict(task=task, indices=order[i:i+size]) for i in range(0, n, size))
    random.Random(stable_seed(seed, role, epoch, 'batch_order')).shuffle(batches)
    return batches


def schedule(target_counts, source_counts, seed, spec=SPEC):
    """Identical target stream in all conditions; source completes 40 full passes.

    Source entries in PT/RT are a count-only clock. The trainer must never fetch
    their graphs/labels. Target short tails and pass boundaries are retained.
    """
    target_pass = 0
    target = iter(full_pass(target_counts, spec['target_batch_size'], seed, 'target', target_pass))
    step = 0
    for epoch in range(spec['source_passes']):
        source = full_pass(source_counts, spec['source_batch_size'], seed, 'source', epoch)
        for i, item in enumerate(source):
            try:
                t = next(target)
            except StopIteration:
                target_pass += 1
                target = iter(full_pass(target_counts, spec['target_batch_size'], seed, 'target', target_pass))
                t = next(target)
            step += 1
            yield dict(step=step, source_pass=epoch+1, source_pass_end=i == len(source)-1,
                       target_pass=target_pass+1, target=t, source=item)


def update_count(source_counts, spec=SPEC):
    return spec['source_passes'] * sum(math.ceil(n / spec['source_batch_size']) for n in source_counts.values())


def evaluation_steps(source_counts, spec=SPEC):
    total = update_count(source_counts, spec)
    width = total // spec['source_passes']
    return sorted({total} | {v for v in spec['early_validation'] if 0 < v <= total}
                  | set(range(spec['validation_every'], total+1, spec['validation_every']))
                  | set(range(width, total+1, width)))


def standardized_loss(model, batch, task, scaler):
    require(scaler['std'] > 0 and math.isfinite(scaler['mean']) and math.isfinite(scaler['std']), 'scaler')
    prediction = model(batch, task)
    y = (batch.y.reshape(-1) - scaler['mean']) / scaler['std']
    require(prediction.shape == y.shape and y.numel() > 0, 'loss shape')
    loss = (prediction - y).square().mean()
    require(bool(torch.isfinite(loss)), 'nonfinite loss')
    return loss


def update(model, opt, data, item, seed, device, spec=SPEC, *, audit=True):
    model.train()  # Includes dropout in FJ: only requires_grad differs from PJ.
    opt.zero_grad(set_to_none=True)
    devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    losses = {}
    for role in ('target', 'source') if model.joint else ('target',):
        row = item[role]
        batch = data.batch(role, 'train', row['task'], row['indices'], device)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(stable_seed(seed, 'd2', role, item['step']))
            if devices:
                torch.cuda.manual_seed(stable_seed(seed, 'd2', role, item['step']))
            loss = standardized_loss(model, batch, row['task'], data.scalers[role][row['task']])
            (loss * (spec['source_weight'] if role == 'source' else 1.)).backward()
        losses[role] = float(loss.detach())
    norms = None
    if audit:
        norms = dict(encoder=sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None),
                 target=sum(float(p.grad.abs().sum()) for t in model.target_tasks for p in model.decoders[t].parameters() if p.grad is not None),
                 source=sum(float(p.grad.abs().sum()) for t in model.source_tasks for p in model.decoders[t].parameters() if p.grad is not None))
        require(model.condition != 'FJ' or norms['encoder'] == 0, 'frozen encoder gradient')
        require(model.joint or norms['source'] == 0, 'forbidden source gradient')
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        spec['grad_clip'], error_if_nonfinite=True)
    opt.step()
    return dict(losses=losses, gradient_l1=norms, gradient_norm=float(norm))


def select(history):
    require(history and all(math.isfinite(r['metrics']['macro_rmse']) for r in history), 'finite history')
    return min(history, key=lambda r: (r['metrics']['macro_rmse'], r['step']))


def paired_report(results):
    """Both LR strata retained. Do not compare independently picked LR winners."""
    require(len(results) == len(jobs()) and {r['job']['id'] for r in results} == {j['id'] for j in jobs()}, 'complete S1 matrix')
    mapping = {r['job']['id']: r for r in results}
    rows = []
    for scene in SETTINGS:
        for rate in ('low', 'high'):
            values = {c: mapping[f'{scene}_{c}_{"fixed" if c == "FJ" else rate}_s42']['selected']['metrics']['macro_rmse']
                      for c in CONDITIONS}
            effects = {f'{a}-{b}': values[a]-values[b] for a, b in [('PJ', 'FJ'), ('RJ', 'PJ'), ('PJ', 'PT'), ('RJ', 'RT')]}
            rows.append(dict(setting=scene, rate=rate, macro_rmse=values, paired_differences=effects))
    return dict(rows=rows, negative_difference_favors_first=True,
                scientific_status='DEVELOPMENT_SINGLE_SEED_PENDING_REVIEW', unified_superiority_confirmed=False,
                automatic_replication_authorized=False, test_accessed=False)
